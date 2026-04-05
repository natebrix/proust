import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .api import create_session
from .corpus import get_canonical_chapter, get_canonical_structure, get_chapter_body, get_chapter_info


@dataclass(frozen=True)
class ReaderEditionSpec:
    id: str
    title: str
    language: str
    description: str = ""
    source: str = "file"
    model: str = "fr_core_news_sm"
    chapter_start: int = 1
    chapter_end: int = 486


READER_EDITIONS = {
    "fr-original": ReaderEditionSpec(
        id="fr-original",
        title="À la recherche du temps perdu",
        language="fr",
        description="texte intégral restructuré selon les divisions canoniques de Proust",
        source="canonical",
        chapter_end=18,
    )
}


def chapter_slug(chapter_number):
    return f"{chapter_number:03}"


def paragraph_id(index):
    return f"p-{index}"


def sentence_id(index):
    return f"s-{index}"


def build_reader_structure(chapter_summaries):
    volumes = []

    for chapter in chapter_summaries:
        volume_id = f"v{chapter['volumeNumber']}"
        volume = next((item for item in volumes if item["id"] == volume_id), None)

        if volume is None:
            volume = {
                "id": volume_id,
                "number": chapter["volumeNumber"],
                "title": chapter["volumeTitle"],
                "chapterIds": [],
                "parts": [],
            }
            volumes.append(volume)

        volume["chapterIds"].append(chapter["id"])

        if chapter["partNumber"] is None:
            continue

        part_id = f"{volume_id}-p{chapter['partNumber']}"
        part = next((item for item in volume["parts"] if item["id"] == part_id), None)

        if part is None:
            part = {
                "id": part_id,
                "number": chapter["partNumber"],
                "title": chapter["partTitle"],
                "chapterIds": [],
            }
            volume["parts"].append(part)

        part["chapterIds"].append(chapter["id"])

    return volumes


def summarize_reader_chapter(chapter):
    summary = {
        "id": chapter["chapterId"],
        "number": chapter["chapterNumber"],
        "title": chapter["title"],
        "paragraphCount": chapter["paragraphCount"],
        "sentenceCount": chapter["sentenceCount"],
        "prevChapterId": chapter["prevChapterId"],
        "nextChapterId": chapter["nextChapterId"],
    }

    for field in (
        "volumeNumber",
        "volumeTitle",
        "partNumber",
        "partTitle",
        "sectionTitle",
        "sourceChapterRange",
        "sourceAnchorIds",
    ):
        if field in chapter:
            summary[field] = chapter[field]

    return summary


def build_reader_chapter(session, edition, chapter_number, source=None):
    html = session.get_proust_page(chapter_number, source=source)
    title = get_chapter_info(html)
    body = get_chapter_body(html)
    paragraphs = []

    for paragraph_index, paragraph in enumerate(body.find_all("p"), start=1):
        text = session.preprocess(paragraph.get_text(" ", strip=True))
        sentence_records = [
            {
                "id": sentence_id(sentence_index),
                "index": sentence_index,
                "text": sentence.text,
            }
            for sentence_index, sentence in enumerate(
                session.get_sentences(text),
                start=1,
            )
        ]
        paragraphs.append(
            {
                "id": paragraph_id(paragraph_index),
                "index": paragraph_index,
                "text": text,
                "sentences": sentence_records,
            }
        )

    return {
        "edition": edition,
        "chapterId": chapter_slug(chapter_number),
        "chapterNumber": chapter_number,
        "title": title,
        "paragraphCount": len(paragraphs),
        "sentenceCount": sum(len(paragraph["sentences"]) for paragraph in paragraphs),
        "paragraphs": paragraphs,
    }


def build_reader_manifest(editions, default_edition):
    return {
        "project": "In Search of Lost Time",
        "defaultEdition": default_edition,
        "editions": editions,
    }


def export_reader_dataset(output_dir, edition_specs=None, default_edition="fr-original"):
    output_path = Path(output_dir)
    specs = list(edition_specs or READER_EDITIONS.values())
    manifest_editions = []

    for spec in specs:
        if spec.source == "canonical":
            canonical_structure = get_canonical_structure(edition=spec.id)
            chapter_summaries = []

            for chapter_id in [chapter["id"] for chapter in canonical_structure]:
                chapter = get_canonical_chapter(chapter_id, edition=spec.id)
                chapter_file = output_path / "editions" / spec.id / "chapters" / f"{chapter['chapterId']}.json"
                chapter_file.parent.mkdir(parents=True, exist_ok=True)
                chapter_file.write_text(json.dumps(chapter, ensure_ascii=False, indent=2) + "\n")
                chapter_summaries.append(summarize_reader_chapter(chapter))

            manifest_editions.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "language": spec.language,
                    "description": spec.description,
                    "chapterCount": len(chapter_summaries),
                    "isComplete": True,
                    "structure": build_reader_structure(chapter_summaries),
                    "chapters": chapter_summaries,
                }
            )
            continue

        session = create_session(model=spec.model, default_source=spec.source)
        chapter_summaries = []

        for chapter_number in range(spec.chapter_start, spec.chapter_end + 1):
            chapter = build_reader_chapter(
                session,
                edition=spec.id,
                chapter_number=chapter_number,
                source=spec.source,
            )
            previous_chapter = chapter_slug(chapter_number - 1) if chapter_number > spec.chapter_start else None
            next_chapter = chapter_slug(chapter_number + 1) if chapter_number < spec.chapter_end else None
            chapter["prevChapterId"] = previous_chapter
            chapter["nextChapterId"] = next_chapter

            chapter_file = output_path / "editions" / spec.id / "chapters" / f"{chapter['chapterId']}.json"
            chapter_file.parent.mkdir(parents=True, exist_ok=True)
            chapter_file.write_text(json.dumps(chapter, ensure_ascii=False, indent=2) + "\n")
            chapter_summaries.append(summarize_reader_chapter(chapter))

        manifest_editions.append(
            {
                "id": spec.id,
                "title": spec.title,
                "language": spec.language,
                "description": spec.description,
                "chapterCount": len(chapter_summaries),
                "chapters": chapter_summaries,
            }
        )

    manifest = build_reader_manifest(manifest_editions, default_edition=default_edition)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export ISLT reader data.")
    parser.add_argument("--output", required=True, help="Directory where reader JSON should be written.")
    parser.add_argument(
        "--edition",
        action="append",
        dest="editions",
        choices=sorted(READER_EDITIONS),
        help="Edition id to export. May be repeated. Defaults to all known editions.",
    )
    args = parser.parse_args(argv)

    selected_specs = [READER_EDITIONS[edition_id] for edition_id in args.editions] if args.editions else None
    export_reader_dataset(args.output, edition_specs=selected_specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
