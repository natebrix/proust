from collections import defaultdict
import csv
import json
from math import comb
from pathlib import Path
import re

from . import runner
from .app_exports import build_chapter_overlay_data
from .paths import ALIASES_CSV


# Reviewed same-person merges applied when accepted annotation identities were
# canonicalized upstream (see proust/docs/character_alias_normalization_plan.md
# and commit "Canonicalize accepted character identities upstream"). Run-level
# unit alias maps and the root aliases.csv predate that rewrite and still use
# these pre-canonicalization names as their canonical bucket keys, while the
# accepted corpus (and therefore build_chapter_overlay_data) always uses the
# name on the right. Any alias-derived canonical bucket keyed by one of these
# left-hand names must be translated to the right-hand accepted identity
# before candidate additions can be compared against scored_characters, or
# every relevant unit would show a false "candidate addition" for a character
# who is already scored under the modern name.
REVIEWED_CANONICAL_MERGES = {
    "Saint-Loup": "Robert de Saint-Loup",
    "princesse des Laumes": "duchesse de Guermantes",
    "Charlus": "baron de Charlus",
    "Mme Swann": "Odette",
    "la grand-mère du narrateur": "la grand-mère",
    "Vinteuil": "M. Vinteuil",
    "Mme de Saint-Euverte": "marquise de Saint-Euverte",
    # Not one of the seven person-identity splits reviewed in the plan doc
    # above (that set only records "princesse des Laumes" -> "duchesse de
    # Guermantes"), but the same target person: aliases.csv itself records
    # "duchesse de Guermantes,Mme de Guermantes" as a same-bucket relationship,
    # and several run-level unit alias maps still use "Mme de Guermantes" as
    # their raw roster key (the pre-canonicalization generic-title form).
    # Left untranslated, literal occurrences of "Mme de Guermantes" in
    # raw_text would be bucketed as a separate, never-scored character
    # instead of folding into the accepted "duchesse de Guermantes" identity.
    "Mme de Guermantes": "duchesse de Guermantes",
    # Accent-variant identity splits found during the supplement mass pass
    # (2026-08-06): the audit universe carried both spellings, and supplement
    # annotators normalize French orthography unpredictably. French-correct
    # forms win.
    "Remi": "Rémi",
    "la jeune ouvriere": "la jeune ouvrière",
}


# The accepted character named "le narrateur" is a normal scored character
# like any other; the narrator heuristic below is a separate, purely lexical
# signal (first-person marker density) used to flag units where the narrator
# is plausibly present in a way worth reviewing for scoring.
NARRATOR_CHARACTER = "le narrateur"

MIN_SURFACE_FORM_LENGTH = 3

# First-person markers with word boundaries. je/j' get matched case
# insensitively because French capitalizes the first word of a sentence
# ("Je ...", "J'avais ..."), and we want sentence-initial occurrences to
# count. Third-person narrated sections (e.g. much of "Un amour de Swann")
# naturally produce a low count here -- that is correct behavior, not a bug:
# those passages are not narrated in the first person, so they should not be
# flagged as narrator-presence candidates.
FIRST_PERSON_PATTERNS = (
    re.compile(r"\bj'(?=\w)", re.IGNORECASE),
    re.compile(r"\bje\b", re.IGNORECASE),
    re.compile(r"\bme\b", re.IGNORECASE),
    re.compile(r"\bmoi\b", re.IGNORECASE),
    re.compile(r"\bmon\b", re.IGNORECASE),
    re.compile(r"\bma\b", re.IGNORECASE),
    re.compile(r"\bmes\b", re.IGNORECASE),
)

# Quoted spans distort the narrator heuristic: a character's own dialogue can
# be full of first-person markers ("je", "moi", "mon", ...) that say nothing
# about whether the narrating voice is present. This is especially acute in
# third-person material like "Un amour de Swann", where dialogue between
# Swann, Odette, and the Verdurins is otherwise-unnarrated but still riddled
# with first-person speech. Two quoting conventions are stripped before
# counting: « ... » guillemet spans (which may run across a paragraph break,
# hence DOTALL), and paragraphs that open with the quotation-dash convention
# ("– " or "— " at the very start of a paragraph, the rest of that paragraph
# being the spoken line).
QUOTED_GUILLEMET_SPAN_PATTERN = re.compile(r"«[^»]*»", re.DOTALL)
QUOTATION_DASH_CHARACTERS = ("–", "—")  # en dash, em dash


def _canonicalize(name):
    return REVIEWED_CANONICAL_MERGES.get(name, name)


def translate_canonical_name(name):
    # Public wrapper around _canonicalize for other modules (e.g.
    # proust/supplement.py) that need to translate a raw unit alias_map's
    # canonical key to the accepted canonical identity without duplicating
    # REVIEWED_CANONICAL_MERGES.
    return _canonicalize(name)


def _run_id_sort_key(run_id):
    match = re.search(r"(\d+)$", run_id)
    return (int(match.group(1)), run_id) if match else (-1, run_id)


def _read_json_or_none(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _read_alias_csv_pairs(path=ALIASES_CSV):
    alias_path = Path(path)
    if not alias_path.exists():
        return []
    pairs = []
    with alias_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            surface = row[0].strip()
            canonical = row[1].strip()
            if surface and canonical:
                pairs.append((surface, canonical))
    return pairs


def _iter_run_unit_payloads(run_dirs):
    for run_dir in run_dirs:
        units_dir = Path(run_dir) / "units"
        if not units_dir.exists():
            continue
        for unit_path in sorted(units_dir.glob("*.json")):
            payload = _read_json_or_none(unit_path)
            if payload is not None:
                yield payload


def _build_surface_index(run_dirs):
    index = {}

    for unit_payload in _iter_run_unit_payloads(run_dirs):
        alias_map = unit_payload.get("alias_map") or {}
        for canonical_raw, entry in alias_map.items():
            canonical = _canonicalize(canonical_raw)
            aliases = (entry or {}).get("aliases") or []
            for surface in aliases:
                if isinstance(surface, str) and len(surface) >= MIN_SURFACE_FORM_LENGTH:
                    index[surface] = canonical

    for surface, canonical_raw in _read_alias_csv_pairs():
        if len(surface) >= MIN_SURFACE_FORM_LENGTH:
            index[surface] = _canonicalize(canonical_raw)

    # A canonical character's own accepted name is always a valid surface
    # form for that character. This pass runs last and unconditionally wins
    # any conflict, which resolves stale aliases.csv rows such as
    # "duchesse de Guermantes,Mme de Guermantes" (a pre-canonicalization row
    # that would otherwise route the literal text "duchesse de Guermantes"
    # to the never-accepted bucket "Mme de Guermantes" instead of to itself).
    for canonical in set(index.values()):
        if len(canonical) >= MIN_SURFACE_FORM_LENGTH:
            index[canonical] = canonical

    return index


def _select_preferred_unit_sources(run_dirs):
    # Mirrors the "latest_reviewed_run_wins" duplicate-unit resolution that
    # build_chapter_overlay_data uses, but resolves file paths relative to
    # the run_dir actually passed in rather than trusting run.json's stored
    # "directories" (which can go stale if the repository is later checked
    # out under a different absolute path/home directory).
    preferred_run_path_by_unit = {}
    preferred_sort_key_by_unit = {}

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        manifest = _read_json_or_none(run_path / "run.json")
        if manifest is None:
            continue
        run_id = manifest.get("run_id", run_path.name)
        sort_key = _run_id_sort_key(run_id)
        annotations_dir = run_path / "annotations"

        for unit_id in manifest.get("unit_ids", []):
            annotation = _read_json_or_none(annotations_dir / f"{unit_id}.json")
            if annotation is None:
                continue
            if runner.validate_annotation_result(annotation, expected_unit_id=unit_id):
                continue
            existing_key = preferred_sort_key_by_unit.get(unit_id)
            if existing_key is None or sort_key > existing_key:
                preferred_sort_key_by_unit[unit_id] = sort_key
                preferred_run_path_by_unit[unit_id] = run_path

    sources = {}
    for unit_id, run_path in preferred_run_path_by_unit.items():
        unit_payload = _read_json_or_none(run_path / "units" / f"{unit_id}.json")
        if unit_payload is None:
            continue
        manifest = _read_json_or_none(run_path / "run.json") or {}
        sources[unit_id] = {
            "raw_text": unit_payload.get("raw_text") or "",
            "alias_map": unit_payload.get("alias_map") or {},
            "source_run": manifest.get("run_id", run_path.name),
        }
    return sources


def _strip_quoted_spans(raw_text):
    without_guillemets = QUOTED_GUILLEMET_SPAN_PATTERN.sub(" ", raw_text)

    kept_paragraphs = []
    for paragraph in without_guillemets.split("\n\n"):
        if paragraph.lstrip().startswith(QUOTATION_DASH_CHARACTERS):
            continue
        kept_paragraphs.append(paragraph)
    return "\n\n".join(kept_paragraphs)


def _count_narrator_markers(raw_text):
    stripped_text = _strip_quoted_spans(raw_text)
    return sum(len(pattern.findall(stripped_text)) for pattern in FIRST_PERSON_PATTERNS)


def _find_candidate_additions(raw_text, surface_index, scored_characters):
    # Longest-surface-form-first matching with span claiming: once a longer
    # surface form (e.g. "Mme de Cambremer") consumes a span of raw_text, no
    # shorter surface form (e.g. a bare "Cambremer" bucket for a different
    # character) is allowed to also match inside that span.
    ordered_surfaces = sorted(surface_index, key=lambda surface: (-len(surface), surface))
    claimed = [False] * len(raw_text)
    matches_by_character = defaultdict(lambda: {"surface_forms": set(), "occurrence_count": 0})

    for surface in ordered_surfaces:
        canonical = surface_index[surface]
        pattern = re.compile(r"\b" + re.escape(surface) + r"\b")
        for match in pattern.finditer(raw_text):
            start, end = match.span()
            if end == start:
                continue
            if any(claimed[start:end]):
                continue
            for position in range(start, end):
                claimed[position] = True
            bucket = matches_by_character[canonical]
            bucket["surface_forms"].add(surface)
            bucket["occurrence_count"] += 1

    candidate_additions = [
        {
            "character": canonical,
            "matched_surface_forms": sorted(info["surface_forms"]),
            "occurrence_count": info["occurrence_count"],
        }
        for canonical, info in matches_by_character.items()
        if canonical not in scored_characters
    ]
    candidate_additions.sort(key=lambda row: row["character"])
    return candidate_additions


def _projected_new_matches(n_before, n_after):
    return comb(n_after, 2) - comb(n_before, 2)


def build_coverage_audit(run_dirs, narrator_min_first_person=2):
    if not run_dirs:
        raise ValueError("At least one run directory is required for a coverage audit.")

    overlay_dataset = build_chapter_overlay_data(run_dirs)
    surface_index = _build_surface_index(run_dirs)
    unit_sources = _select_preferred_unit_sources(run_dirs)

    unit_rows = []
    character_totals = defaultdict(
        lambda: {
            "flagged_unit_count": 0,
            "projected_new_matches_without_narrator": 0,
            "projected_new_matches_with_narrator": 0,
        }
    )
    chapter_totals = defaultdict(
        lambda: {
            "unit_count": 0,
            "flagged_unit_count": 0,
            "narrator_candidate_unit_count": 0,
            "candidate_addition_count": 0,
        }
    )

    flagged_unit_count = 0
    candidate_addition_count = 0
    narrator_candidate_unit_count = 0
    projected_new_matches_without_narrator = 0
    projected_new_matches_with_narrator = 0
    missing_raw_text_unit_ids = []

    for chapter in overlay_dataset["chapters"]:
        chapter_id = chapter["chapterId"]
        for unit in chapter["units"]:
            unit_id = unit["unitId"]
            scored_characters = sorted({row["character"] for row in unit["characters"]})
            scored_character_set = set(scored_characters)

            source = unit_sources.get(unit_id)
            if source is None:
                missing_raw_text_unit_ids.append(unit_id)
                raw_text = ""
                source_run = None
            else:
                raw_text = source["raw_text"]
                source_run = source["source_run"]

            candidate_additions = _find_candidate_additions(raw_text, surface_index, scored_character_set)
            narrator_first_person_count = _count_narrator_markers(raw_text)
            narrator_candidate = (
                narrator_first_person_count >= narrator_min_first_person
                and NARRATOR_CHARACTER not in scored_character_set
            )

            n_before = len(scored_characters)
            n_after_without_narrator = n_before + len(candidate_additions)
            n_after_with_narrator = n_after_without_narrator + (1 if narrator_candidate else 0)
            gain_without_narrator = _projected_new_matches(n_before, n_after_without_narrator)
            gain_with_narrator = _projected_new_matches(n_before, n_after_with_narrator)

            unit_row = {
                "unit_id": unit_id,
                "chapter_id": chapter_id,
                "source_run": source_run,
                "scored_characters": scored_characters,
                "candidate_additions": candidate_additions,
                "narrator_first_person_count": narrator_first_person_count,
                "narrator_candidate": narrator_candidate,
                "projected_new_matches_without_narrator": gain_without_narrator,
                "projected_new_matches_with_narrator": gain_with_narrator,
            }
            unit_rows.append(unit_row)

            chapter_bucket = chapter_totals[chapter_id]
            chapter_bucket["unit_count"] += 1
            chapter_bucket["candidate_addition_count"] += len(candidate_additions)

            is_flagged = bool(candidate_additions)
            if is_flagged:
                flagged_unit_count += 1
                chapter_bucket["flagged_unit_count"] += 1
            if narrator_candidate:
                narrator_candidate_unit_count += 1
                chapter_bucket["narrator_candidate_unit_count"] += 1

            candidate_addition_count += len(candidate_additions)
            projected_new_matches_without_narrator += gain_without_narrator
            projected_new_matches_with_narrator += gain_with_narrator

            if is_flagged or narrator_candidate:
                participants = scored_character_set | {row["character"] for row in candidate_additions}
                if narrator_candidate:
                    participants = participants | {NARRATOR_CHARACTER}
                for character in participants:
                    totals = character_totals[character]
                    totals["flagged_unit_count"] += 1
                    totals["projected_new_matches_without_narrator"] += gain_without_narrator
                    totals["projected_new_matches_with_narrator"] += gain_with_narrator

    character_summary = [
        {
            "character": character,
            "flagged_unit_count": totals["flagged_unit_count"],
            "projected_new_matches_without_narrator": totals["projected_new_matches_without_narrator"],
            "projected_new_matches_with_narrator": totals["projected_new_matches_with_narrator"],
        }
        for character, totals in character_totals.items()
    ]
    character_summary.sort(
        key=lambda row: (
            -row["projected_new_matches_with_narrator"],
            -row["projected_new_matches_without_narrator"],
            row["character"],
        )
    )

    chapter_summary = [
        {"chapter_id": chapter_id, **totals}
        for chapter_id, totals in sorted(chapter_totals.items())
    ]

    return {
        "coverage_audit_version": "coverage_audit_v1",
        "metadata": {
            "unit_count": len(unit_rows),
            "flagged_unit_count": flagged_unit_count,
            "candidate_addition_count": candidate_addition_count,
            "narrator_candidate_unit_count": narrator_candidate_unit_count,
            "params": {"narrator_min_first_person": narrator_min_first_person},
            "duplicate_resolution": overlay_dataset["duplicate_resolution"],
            "missing_raw_text_unit_ids": missing_raw_text_unit_ids,
        },
        "units": unit_rows,
        "summary": {
            "characters": character_summary,
            "chapters": chapter_summary,
            "corpus_totals": {
                "projected_new_matches_without_narrator": projected_new_matches_without_narrator,
                "projected_new_matches_with_narrator": projected_new_matches_with_narrator,
            },
        },
    }
