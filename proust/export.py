import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .corpus import get_canonical_chapter, get_canonical_structure


@dataclass(frozen=True)
class ReaderEditionSpec:
    id: str
    title: str
    language: str
    description: str = ""


READER_EDITIONS = {
    "fr-original": ReaderEditionSpec(
        id="fr-original",
        title="À la recherche du temps perdu",
        language="fr",
        description="texte intégral restructuré selon les divisions canoniques de Proust",
    )
}


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
    ):
        if field in chapter:
            summary[field] = chapter[field]

    return summary


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
