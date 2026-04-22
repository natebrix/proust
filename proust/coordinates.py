from dataclasses import dataclass
import json

from .corpus import preprocess
from .export import CANONICAL_CHAPTER_SPECS
from .paths import CANONICAL_ISLT_DATA_DIR, DATA_DIR


READER_BASE_URL = "https://nathanbrixius.com/projects/islt"


@dataclass(frozen=True)
class CanonicalParagraphRef:
    chapter_id: str
    paragraph_index: int
    paragraph_id: str
    source_chapter_id: int
    source_paragraph_index: int


def annotation_unit_id(chapter_id, paragraph_start, paragraph_end=None):
    end = paragraph_end if paragraph_end is not None else paragraph_start
    return f"{chapter_id}#p-{paragraph_start}" if end == paragraph_start else f"{chapter_id}#p-{paragraph_start}-p-{end}"


def reader_url(edition_id, chapter_id, paragraph_start=None):
    url = f"{READER_BASE_URL}/{edition_id}/{chapter_id}"
    if paragraph_start is not None:
        url += f"#p-{paragraph_start}"
    return url


def reader_urls(chapter_id, paragraph_start=None):
    return {
        "fr-original": reader_url("fr-original", chapter_id, paragraph_start=paragraph_start),
        "en-moncrieff": reader_url("en-moncrieff", chapter_id, paragraph_start=paragraph_start),
    }


def map_source_paragraph_to_canonical(source_chapter_id, source_paragraph_index):
    source_text = _load_source_paragraph_text(source_chapter_id, source_paragraph_index)
    candidate_texts = {
        _normalize_text(source_text),
        _normalize_text(preprocess(source_text)),
    }

    for chapter_spec in CANONICAL_CHAPTER_SPECS:
        if chapter_spec.start_chapter_id <= source_chapter_id <= chapter_spec.end_chapter_id:
            chapter_data = _load_canonical_chapter_json(chapter_spec.id)
            for paragraph in chapter_data["paragraphs"]:
                paragraph_text = _normalize_text(paragraph["text"])
                if paragraph_text in candidate_texts:
                    return CanonicalParagraphRef(
                        chapter_id=chapter_spec.id,
                        paragraph_index=paragraph["index"],
                        paragraph_id=paragraph["id"],
                        source_chapter_id=source_chapter_id,
                        source_paragraph_index=source_paragraph_index,
                    )
            break

    raise ValueError(
        f"Could not map source chapter {source_chapter_id:03} paragraph {source_paragraph_index} to a canonical paragraph."
    )


def _load_source_paragraph_text(source_chapter_id, source_paragraph_index):
    from bs4 import BeautifulSoup

    html = (DATA_DIR / f"islt_fr_{source_chapter_id:03}.html").read_text()
    body = BeautifulSoup(html, features="html.parser").body.find("div", attrs={"class": "field-item"})
    paragraphs = body.find_all("p")
    return paragraphs[source_paragraph_index - 1].get_text(" ", strip=True)


def _load_canonical_chapter_json(chapter_id):
    chapter_path = CANONICAL_ISLT_DATA_DIR / "editions" / "fr-original" / "chapters" / f"{chapter_id}.json"
    return json.loads(chapter_path.read_text())


def _normalize_text(text):
    return " ".join(text.split())
