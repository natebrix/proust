import json
from dataclasses import dataclass
from pathlib import Path

from .api import create_session
from .corpus import apply_alias_replacements
from .coordinates import annotation_unit_id, reader_urls
from .export import CANONICAL_CHAPTER_SPECS
from .paths import CANONICAL_ISLT_DATA_DIR


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "prompt.md"


DEFAULT_STARTER_ALIAS_MAP = {
    "Swann": {
        "aliases": ["Swann", "M. Swann", "Charles Swann"],
        "notes": "Charles Swann",
    },
    "Legrandin": {
        "aliases": ["Legrandin", "M. Legrandin"],
        "notes": "",
    },
    "Mme de Villeparisis": {
        "aliases": ["Mme de Villeparisis", "Madame de Villeparisis"],
        "notes": "",
    },
    "Mme de Cambremer": {
        "aliases": ["Mme de Cambremer", "Madame de Cambremer"],
        "notes": "Legrandin's sister",
    },
    "M. Vinteuil": {
        "aliases": ["M. Vinteuil", "Vinteuil"],
        "notes": "",
    },
    "la mère du narrateur": {
        "aliases": ["maman", "ma mère"],
        "notes": "",
    },
}


@dataclass(frozen=True)
class AnnotationUnitSpec:
    chapter_id: str
    paragraph_start: int
    paragraph_end: int | None = None
    notes: str = ""


STARTER_UNITS = (
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=17,
        notes="Narrated prestige around Swann.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=274,
        paragraph_end=275,
        notes="Legrandin's reaction to Guermantes exposes snobbery.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=312,
        paragraph_end=313,
        notes="Swann is praised personally and diminished socially.",
    ),
)


RUN_007_UNITS = (
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=16,
        notes="Swann enters through family memory and inherited judgment.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=17,
        notes="Narrated prestige around Swann.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=18,
        notes="Family caste assumptions keep Swann socially misrecognized.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=19,
        notes="Counterfactual revelation intensifies the gap between Swann's worlds.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=20,
        notes="Swann's real aristocratic ties are discounted by the family frame.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=21,
        paragraph_end=22,
        notes="Swann is used familiarly while denied full prestige.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=23,
        paragraph_end=24,
        notes="Mme de Villeparisis is lowered rather than Swann being raised.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=25,
        paragraph_end=26,
        notes="External ratification partly repositions Swann in family judgment.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=27,
        paragraph_end=28,
        notes="Dinner-table handling of Swann mixes awkward esteem and misfire.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=29,
        paragraph_end=30,
        notes="Swann becomes an explicit comparison point for anxious exclusion.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=33,
        notes="Post-dinner family judgment repositions Swann through age and marriage.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=270,
        notes="Legrandin's public self-abasement becomes visibly legible.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=271,
        paragraph_end=273,
        notes="Prelude to the Guermantes question keeps Legrandin's self-staging in focus.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=274,
        paragraph_end=275,
        notes="Legrandin's reaction to Guermantes exposes snobbery.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=276,
        paragraph_end=277,
        notes="The family plans a Balbec test while Legrandin preemptively ornaments the terrain.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=278,
        paragraph_end=279,
        notes="Legrandin evades direct questioning about Balbec ties.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=280,
        paragraph_end=282,
        notes="Legrandin escalates evasive abstraction rather than answer plainly.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=283,
        paragraph_end=285,
        notes="Balbec evasion culminates in refusal to disclose his sister connection.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=310,
        paragraph_end=311,
        notes="Vinteuil is socially broken down by grief, shame, and local judgment.",
    ),
    AnnotationUnitSpec(
        chapter_id="v1-p1-combray",
        paragraph_start=312,
        paragraph_end=313,
        notes="Swann is praised personally and diminished socially.",
    ),
)


def load_prompt_template(path=PROMPT_PATH):
    return Path(path).read_text()


def _get_chapter_spec(chapter_id):
    for spec in CANONICAL_CHAPTER_SPECS:
        if spec.id == chapter_id:
            return spec
    raise ValueError(f'Unknown canonical chapter id "{chapter_id}".')


def _load_canonical_chapter_json(chapter_id):
    chapter_path = CANONICAL_ISLT_DATA_DIR / "editions" / "fr-original" / "chapters" / f"{chapter_id}.json"
    return json.loads(chapter_path.read_text())


def _flatten_alias_map(alias_map):
    replacements = {}

    for canonical_name, entry in alias_map.items():
        aliases = entry.get("aliases", [])
        for alias in aliases:
            replacements[alias] = canonical_name

    return replacements


def _preprocess_annotation_text(text, alias_map):
    text = text.replace("; –", ";").replace("– ;", ";")
    replacements = _flatten_alias_map(alias_map)
    return apply_alias_replacements(text, replacements)


def get_canonical_chapter_paragraphs(session, chapter_id, alias_map=None):
    del session
    chapter_data = _load_canonical_chapter_json(chapter_id)
    paragraphs = []
    active_alias_map = alias_map or {}

    for paragraph in chapter_data["paragraphs"]:
        text = paragraph["text"]
        paragraphs.append(
            {
                "index": paragraph["index"],
                "raw_text": text,
                "preprocessed_text": _preprocess_annotation_text(text, active_alias_map),
            }
        )

    return paragraphs


def build_annotation_unit(
    chapter_id,
    paragraph_start,
    paragraph_end=None,
    prior_context_paragraphs=0,
    alias_map=None,
    session=None,
):
    end = paragraph_end if paragraph_end is not None else paragraph_start
    if end < paragraph_start:
        raise ValueError("paragraph_end must be greater than or equal to paragraph_start")

    active_alias_map = alias_map or DEFAULT_STARTER_ALIAS_MAP
    active_session = session or create_session(aliases={}, nlp=None)
    paragraphs = get_canonical_chapter_paragraphs(active_session, chapter_id, alias_map=active_alias_map)

    if paragraph_start < 1 or end > len(paragraphs):
        raise ValueError(f"Paragraph range p-{paragraph_start}..p-{end} is out of bounds for {chapter_id}.")

    selected = paragraphs[paragraph_start - 1 : end]
    prior_start = max(1, paragraph_start - prior_context_paragraphs)
    prior = paragraphs[prior_start - 1 : paragraph_start - 1]

    return {
        "unit_id": annotation_unit_id(chapter_id, paragraph_start, end),
        "source_text": "islt_fr",
        "chapter_id": chapter_id,
        "paragraph_start": paragraph_start,
        "paragraph_end": end,
        "raw_text": "\n\n".join(item["raw_text"] for item in selected),
        "preprocessed_text": "\n\n".join(item["preprocessed_text"] for item in selected),
        "reader_urls": reader_urls(chapter_id, paragraph_start=paragraph_start),
        "source_chapter_range": {
            "start": f"{_get_chapter_spec(chapter_id).start_chapter_id:03}",
            "end": f"{_get_chapter_spec(chapter_id).end_chapter_id:03}",
        },
        "alias_map": active_alias_map,
        "prior_context": "\n\n".join(item["preprocessed_text"] for item in prior),
    }


def render_prompt_input(unit_payload, prompt_template=None):
    template = prompt_template or load_prompt_template()
    prior_context = unit_payload["prior_context"] or "[none]"
    return (
        template.replace("{{ALIAS_MAP}}", json.dumps(unit_payload["alias_map"], ensure_ascii=False, indent=2))
        .replace("{{PRIOR_CONTEXT}}", prior_context)
        .replace("{{PASSAGE}}", unit_payload["preprocessed_text"])
    )


def build_starter_units(alias_map=None, session=None):
    return [
        build_annotation_unit(
            unit.chapter_id,
            unit.paragraph_start,
            paragraph_end=unit.paragraph_end,
            prior_context_paragraphs=1,
            alias_map=alias_map,
            session=session,
        )
        | {"notes": unit.notes}
        for unit in STARTER_UNITS
    ]
