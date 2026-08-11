"""Stage the prompt-v2 A/B evaluation (registry design migration step 2).

See proust/docs/character_registry_design.md ("Prompt v2" section, migration
step 2): "Switch annotation input to raw text + reference sheet; prompt v2.
A/B a handful of units against v1 to measure drift before committing."

This script PREPARES a 10-unit A/B run under outputs/ab-run-001/. It does not
call any model: annotation happens later via a separate orchestrated run over
prompts-a/ and prompts-b/, writing into annotations-a/ and annotations-b/
(raw-a/ and raw-b/ for the raw model responses). scripts/compare_ab_run.py
consumes whatever has landed there, tolerating partial completion.

For each of the 10 legacy exemplars (chosen because they were accepted
annotations that exercise the closed-world alias-map gaps the registry design
doc calls out -- Rachel, Mme de Marsantes, l'amie de Mlle Vinteuil, the
dame-en-rose units, etc.) this script:

  1. Bridges the legacy (fan-site) paragraph span to the new (Wikisource)
     paragraph span using outputs/source-migration-map.json, with the same
     anchor semantics scripts/remap_artifact_paragraphs.py uses for its
     "phase D" reader-link bridge: a span START resolves through the nearest
     FOLLOWING mapped paragraph when the anchor itself has no direct new-
     edition counterpart (never starts earlier than intended); a span END
     resolves through the nearest PRECEDING mapped paragraph.
  2. Builds the unit payload on the NEW (current canonical, Wikisource) text
     via proust.annotation.build_annotation_unit, using the LEGACY alias map
     lifted verbatim from the accepted run's unit payload -- this is what
     prompt v1 (arm A) gets rewritten text from.
  3. Renders prompt v1 (arm A) the normal way (proust.annotation.
     render_prompt_input) and prompt v2 (arm B) via the small render_prompt_v2
     helper below, which injects Registry.load().reference_sheet(chapter_id)
     as {{REFERENCE_SHEET}} and passes the UNALTERED raw_text as the passage
     (prompt v2's own text says "the text is Proust's original, unaltered").
  4. Copies the accepted legacy annotation in as a fixed reference point.

Usage:  python scripts/prepare_ab_run.py [--repo PATH] [--out outputs/ab-run-001]
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from proust.annotation import build_annotation_unit, render_prompt_input  # noqa: E402
from proust.registry import Registry  # noqa: E402

MIGRATION_MAP_PATH = REPO / "outputs" / "source-migration-map.json"
OUTPUTS_DIR = REPO / "outputs"
DEFAULT_AB_RUN_DIR = OUTPUTS_DIR / "ab-run-001"
PROMPT_V2_PATH = REPO / "proust" / "prompts" / "prompt_v2.md"

EMPTY_DIRS = ("annotations-a", "annotations-b", "raw-a", "raw-b")


@dataclass(frozen=True)
class UnitSpec:
    chapter_id: str
    legacy_start: int
    legacy_end: int
    accepted_run: str
    notes: str
    verify_keywords: tuple  # substrings expected to survive in the new-span text


# The 10 legacy exemplars from the mission brief. `accepted_run` is the
# `source_run` outputs/coverage-audit-current.json records for this exact
# legacy unit id (that field already resolves "latest run per unit" per the
# audit's own duplicate_resolution policy). `verify_keywords` is a cheap
# sanity net, not a substitute for reading the passage: prepare() asserts
# each keyword survives in the bridged new-span raw text.
UNIT_SPECS = (
    UnitSpec(
        "v7-p4-le-bal-de-tetes", 96, 100, "run-554",
        "la Berma / Rachel duel -- the Rachel-exclusion test",
        ("Rachel", "la Berma"),
    ),
    UnitSpec(
        "v7-p2-m-de-charlus-pendant-la-guerre", 76, 80, "run-544",
        "the Marsantes chimera scene (Françoise's grief-reaction to Mme de Marsantes)",
        ("Marsantes",),
    ),
    UnitSpec(
        "v1-p1-combray", 111, 115, "run-291",
        "oncle Adolphe / la dame en rose (introduction of oncle Adolphe, "
        "immediately preceding the dame-en-rose reveal)",
        ("oncle Adolphe",),
    ),
    UnitSpec(
        "v1-p1-combray", 311, 315, "run-305",
        "Montjouvain -- l'amie de Mlle Vinteuil test",
        ("Vinteuil", "Montjouvain"),
    ),
    UnitSpec(
        "v3-p1", 51, 55, "run-392",
        "heavy variant-flattening zone per the audit",
        ("Berma",),
    ),
    UnitSpec(
        "v3-p1", 306, 310, "run-404",
        "Saint-Loup / duchesse control with accepted scores",
        ("Saint-Loup", "duchesse"),
    ),
    UnitSpec(
        "v1-p2-un-amour-de-swann", 105, 106, "run-028",
        "third-person control",
        ("Verdurin",),
    ),
    UnitSpec(
        "v4-p2", 346, 350, "run-494",
        "roster-rich Verdurin salon",
        ("Cambremer", "Verdurin"),
    ),
    UnitSpec(
        "v5", 121, 125, "run-508",
        "narrator-intimate Albertine unit",
        ("Albertine",),
    ),
    UnitSpec(
        "v2-p1-autour-de-mme-swann", 241, 250, "run-103",
        "Norpois demolishes Bergotte -- known contested reading",
        ("Norpois", "Bergotte"),
    ),
)


def legacy_unit_id(spec: UnitSpec) -> str:
    return f"{spec.chapter_id}#p-{spec.legacy_start}-p-{spec.legacy_end}"


# --------------------------------------------------------------- span bridge


def load_migration_map(path=MIGRATION_MAP_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chapter_pairings(migration_map: dict, chapter_id: str) -> list:
    for chapter in migration_map["chapters"]:
        if chapter["chapterId"] == chapter_id:
            return chapter["pairings"]
    raise ValueError(f"No migration map entry for chapter {chapter_id!r}.")


def _paragraph_number(paragraph_id: str) -> int:
    return int(paragraph_id.split("-", 1)[1])


def build_chapter_lookup(pairings: list) -> dict:
    """old paragraph index -> {"start": new index, "end": new index}.

    Same anchor semantics as scripts/remap_artifact_paragraphs.py's
    build_chapter_lookup: a directly-mapped old paragraph uses its own
    pairing's [min(new_ids), max(new_ids)]; an old_only paragraph (no
    Wikisource counterpart -- stripped apparatus, a legacy-only artifact)
    borrows from its nearest mapped neighbour, preferring the FOLLOWING
    paragraph for "start" and the PRECEDING paragraph for "end" so a bridged
    span never silently grows past its intended edges.
    """
    direct_start, direct_end = {}, {}
    all_old_ids = []
    for pairing in pairings:
        old_ids = [_paragraph_number(pid) for pid in pairing["old_ids"]]
        new_ids = [_paragraph_number(pid) for pid in pairing["new_ids"]]
        all_old_ids.extend(old_ids)
        if not new_ids:
            continue
        start, end = min(new_ids), max(new_ids)
        for old_index in old_ids:
            direct_start[old_index] = start
            direct_end[old_index] = end

    mapped_ids = sorted(direct_start)

    def nearest_at_or_after(value):
        index = bisect.bisect_left(mapped_ids, value)
        return mapped_ids[index] if index < len(mapped_ids) else None

    def nearest_before(value):
        index = bisect.bisect_left(mapped_ids, value)
        return mapped_ids[index - 1] if index > 0 else None

    lookup = {}
    for old_index in sorted(set(all_old_ids)):
        if old_index in direct_start:
            lookup[old_index] = {"start": direct_start[old_index], "end": direct_end[old_index]}
            continue
        following = nearest_at_or_after(old_index)
        preceding = nearest_before(old_index)
        start = direct_start[following] if following is not None else (
            direct_end[preceding] if preceding is not None else None
        )
        end = direct_end[preceding] if preceding is not None else (
            direct_start[following] if following is not None else None
        )
        lookup[old_index] = {"start": start, "end": end}
    return lookup


def bridge_span(pairings: list, old_start: int, old_end: int) -> tuple:
    """Bridge a legacy [old_start, old_end] paragraph span to the new edition.

    Returns (new_start, new_end). Raises ValueError if either edge cannot be
    resolved at all (only possible for a chapter with no mapped paragraphs).
    """
    lookup = build_chapter_lookup(pairings)
    if old_start not in lookup or lookup[old_start]["start"] is None:
        raise ValueError(f"Cannot bridge span start p-{old_start}: unmapped.")
    if old_end not in lookup or lookup[old_end]["end"] is None:
        raise ValueError(f"Cannot bridge span end p-{old_end}: unmapped.")
    return lookup[old_start]["start"], lookup[old_end]["end"]


# ------------------------------------------------------------- v2 rendering


def load_prompt_v2_template(path=PROMPT_V2_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt_v2_input(unit_payload: dict, reference_sheet: dict, prompt_template: str | None = None) -> str:
    """Render prompt v2 for a unit payload.

    Unlike render_prompt_input (prompt v1, ALIAS_MAP + alias-rewritten
    preprocessed_text), prompt v2's own text is explicit that the passage is
    "Proust's original, unaltered" and the reference sheet is "advisory
    context, not a constraint" -- so this uses unit_payload["raw_text"], not
    the alias-rewritten preprocessed_text. Added alongside
    proust.annotation.render_prompt_input; that function is untouched.
    """
    template = prompt_template or load_prompt_v2_template()
    prior_context = unit_payload["prior_context"] or "[none]"
    return (
        template.replace("{{REFERENCE_SHEET}}", json.dumps(reference_sheet, ensure_ascii=False, indent=2))
        .replace("{{PRIOR_CONTEXT}}", prior_context)
        .replace("{{PASSAGE}}", unit_payload["raw_text"])
    )


# ------------------------------------------------------------------- build


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_accepted_alias_map(spec: UnitSpec) -> dict:
    unit_path = OUTPUTS_DIR / spec.accepted_run / "units" / f"{legacy_unit_id(spec)}.json"
    if not unit_path.exists():
        raise FileNotFoundError(
            f"Accepted run unit payload not found for {legacy_unit_id(spec)!r}: {unit_path}"
        )
    return _read_json(unit_path)["alias_map"]


def load_accepted_annotation(spec: UnitSpec) -> dict:
    annotation_path = OUTPUTS_DIR / spec.accepted_run / "annotations" / f"{legacy_unit_id(spec)}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Accepted annotation not found for {legacy_unit_id(spec)!r}: {annotation_path}"
        )
    return _read_json(annotation_path)


def prepare_unit(spec: UnitSpec, migration_map: dict, registry: Registry, ab_run_dir: Path) -> dict:
    pairings = chapter_pairings(migration_map, spec.chapter_id)
    new_start, new_end = bridge_span(pairings, spec.legacy_start, spec.legacy_end)

    alias_map = load_accepted_alias_map(spec)
    unit_payload = build_annotation_unit(
        spec.chapter_id,
        new_start,
        paragraph_end=new_end,
        prior_context_paragraphs=1,
        alias_map=alias_map,
    )
    new_unit_id = unit_payload["unit_id"]

    missing_keywords = [kw for kw in spec.verify_keywords if kw not in unit_payload["raw_text"]]
    if missing_keywords:
        raise ValueError(
            f"Bridged span {spec.chapter_id}#p-{new_start}-p-{new_end} (from legacy "
            f"{legacy_unit_id(spec)}) is missing expected material: {missing_keywords}. "
            "The migration-map bridge may need a manual +/-1 paragraph adjustment for "
            "this unit."
        )

    reference_sheet = registry.reference_sheet(chapter_id=spec.chapter_id)

    prompt_a = render_prompt_input(unit_payload)
    prompt_b = render_prompt_v2_input(unit_payload, reference_sheet)
    for placeholder in ("{{ALIAS_MAP}}", "{{PRIOR_CONTEXT}}", "{{PASSAGE}}"):
        assert placeholder not in prompt_a, f"{placeholder} left unresolved in prompt A for {new_unit_id}"
    for placeholder in ("{{REFERENCE_SHEET}}", "{{PRIOR_CONTEXT}}", "{{PASSAGE}}"):
        assert placeholder not in prompt_b, f"{placeholder} left unresolved in prompt B for {new_unit_id}"

    accepted_annotation = load_accepted_annotation(spec)

    _write_json(ab_run_dir / "units" / f"{new_unit_id}.json", unit_payload)
    (ab_run_dir / "prompts-a").mkdir(parents=True, exist_ok=True)
    (ab_run_dir / "prompts-a" / f"{new_unit_id}.md").write_text(prompt_a, encoding="utf-8")
    (ab_run_dir / "prompts-b").mkdir(parents=True, exist_ok=True)
    (ab_run_dir / "prompts-b" / f"{new_unit_id}.md").write_text(prompt_b, encoding="utf-8")
    _write_json(ab_run_dir / "accepted" / f"{legacy_unit_id(spec)}.json", accepted_annotation)

    return {
        "legacy_unit_id": legacy_unit_id(spec),
        "chapter_id": spec.chapter_id,
        "legacy_span": {"start": spec.legacy_start, "end": spec.legacy_end},
        "new_span": {"start": new_start, "end": new_end},
        "new_unit_id": new_unit_id,
        "accepted_run": spec.accepted_run,
        "notes": spec.notes,
        "accepted_alias_map_size": len(alias_map),
        "reference_sheet_entity_count": len(reference_sheet),
    }


def prepare_ab_run(ab_run_dir: Path = DEFAULT_AB_RUN_DIR, migration_map_path: Path = MIGRATION_MAP_PATH) -> dict:
    migration_map = load_migration_map(migration_map_path)
    registry = Registry.load()

    manifest_units = [prepare_unit(spec, migration_map, registry, ab_run_dir) for spec in UNIT_SPECS]

    for name in EMPTY_DIRS:
        directory = ab_run_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").write_text("", encoding="utf-8")

    manifest = {
        "ab_run_id": ab_run_dir.name,
        "description": (
            "Prompt-v2 A/B evaluation: arm A = prompt v1 + legacy alias map "
            "(lifted from the accepted run), arm B = prompt v2 + registry "
            "reference sheet. Both arms run on the current (Wikisource) "
            "canonical text; annotation happens in a later, separate run."
        ),
        "prompt_a_template": "proust/prompts/prompt.md",
        "prompt_b_template": "proust/prompts/prompt_v2.md",
        "migration_map": str(migration_map_path.relative_to(REPO)) if migration_map_path.is_relative_to(REPO) else str(migration_map_path),
        "registry_path": "characters.yaml",
        "units": manifest_units,
    }
    _write_json(ab_run_dir / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_AB_RUN_DIR), help="Output ab-run directory.")
    parser.add_argument("--migration-map", default=str(MIGRATION_MAP_PATH), help="source-migration-map.json path.")
    args = parser.parse_args(argv)

    manifest = prepare_ab_run(Path(args.out), Path(args.migration_map))

    print(f"Prepared {len(manifest['units'])} units under {args.out}")
    for entry in manifest["units"]:
        legacy = entry["legacy_unit_id"]
        new_id = entry["new_unit_id"]
        print(f"  {legacy} -> {new_id}  (ref_sheet={entry['reference_sheet_entity_count']}, "
              f"legacy_alias_map={entry['accepted_alias_map_size']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
