import json
from datetime import datetime, timezone
from pathlib import Path

from . import runner
from .coverage import NARRATOR_CHARACTER, translate_canonical_name


# Supplements are a separate artifact family (see
# proust/docs/coverage_supplement_plan.md): outputs/supplement-run-*,
# schema version "annotation_supplement_v1", same run-directory layout
# (run.json, units/, prompts/, raw/, annotations/) as a regular annotation
# run, so existing status/score/report tooling applies unchanged.
SUPPLEMENT_SCHEMA_VERSION = "annotation_supplement_v1"
SUPPLEMENT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "supplement_prompt.md"


def _supplement_unit_filename(unit_id):
    return f"{unit_id}.json"


def _supplement_prompt_filename(unit_id):
    return f"{unit_id}.md"


def _supplement_annotation_filename(unit_id):
    return f"{unit_id}.json"


def load_supplement_prompt_template(path=SUPPLEMENT_PROMPT_PATH):
    return Path(path).read_text()


def _translate_alias_map(alias_map):
    # Translates a raw unit alias_map's canonical keys (which may still use
    # pre-canonicalization names, see coverage.REVIEWED_CANONICAL_MERGES) to
    # the accepted canonical identities used by the accepted corpus, so a
    # supplement annotator sees the same character names the accepted
    # annotation and candidate list use. Reuses coverage.translate_canonical_name
    # rather than duplicating the merge table.
    translated = {}
    for canonical_raw, entry in (alias_map or {}).items():
        canonical = translate_canonical_name(canonical_raw)
        bucket = translated.setdefault(canonical, {"aliases": []})
        for alias in (entry or {}).get("aliases") or []:
            if alias not in bucket["aliases"]:
                bucket["aliases"].append(alias)
    return translated


def _unit_candidate_characters(audit_row):
    # "le narrateur" can independently arise from surface-form candidate
    # matching (candidate_additions) and from the narrator_candidate
    # heuristic; only add it once.
    candidate_characters = [addition["character"] for addition in audit_row.get("candidate_additions", [])]
    if audit_row.get("narrator_candidate") and NARRATOR_CHARACTER not in candidate_characters:
        candidate_characters.append(NARRATOR_CHARACTER)
    return candidate_characters


def render_supplement_prompt(unit_payload, prompt_template=None):
    template = prompt_template or load_supplement_prompt_template()
    prior_context = unit_payload.get("prior_context") or "(none provided)"
    return (
        template.replace("{{ALIAS_MAP}}", json.dumps(unit_payload["alias_map"], ensure_ascii=False, indent=2))
        .replace(
            "{{ACCEPTED_ANNOTATION}}",
            json.dumps(unit_payload["accepted_annotation"], ensure_ascii=False, indent=2),
        )
        .replace("{{CANDIDATES}}", json.dumps(unit_payload["candidate_characters"], ensure_ascii=False, indent=2))
        .replace("{{PRIOR_CONTEXT}}", prior_context)
        .replace("{{PASSAGE}}", unit_payload["raw_text"])
    )


def prepare_supplement_run(audit_path, unit_ids, output_dir, run_id=None, notes="", source_outputs_dir="outputs"):
    audit = json.loads(Path(audit_path).read_text())
    units_by_id = {row["unit_id"]: row for row in audit.get("units", [])}

    requested_unit_ids = list(unit_ids)
    missing_unit_ids = sorted(set(requested_unit_ids) - set(units_by_id))
    if missing_unit_ids:
        raise ValueError(
            f'Unit ids not present in coverage audit "{audit_path}": {", ".join(missing_unit_ids)}'
        )

    output_path = Path(output_dir)
    resolved_run_id = run_id or output_path.name
    directories = runner._ensure_run_directories(output_path)
    prompt_template = load_supplement_prompt_template()

    source_run_ids = set()
    written_unit_ids = []

    for unit_id in requested_unit_ids:
        audit_row = units_by_id[unit_id]
        source_run = audit_row.get("source_run")
        if not source_run:
            raise ValueError(f'Unit "{unit_id}" has no recorded source_run in the coverage audit.')
        source_run_ids.add(source_run)
        source_run_path = Path(source_outputs_dir) / source_run

        accepted_unit_payload = runner._read_json(source_run_path / "units" / f"{unit_id}.json")
        accepted_annotation = runner._read_json(source_run_path / "annotations" / f"{unit_id}.json")

        supplement_unit_payload = dict(accepted_unit_payload)
        supplement_unit_payload["alias_map"] = _translate_alias_map(accepted_unit_payload.get("alias_map"))
        supplement_unit_payload["accepted_annotation"] = accepted_annotation
        supplement_unit_payload["candidate_characters"] = _unit_candidate_characters(audit_row)
        supplement_unit_payload["supplement_of_run"] = source_run

        (directories["units"] / _supplement_unit_filename(unit_id)).write_text(
            json.dumps(supplement_unit_payload, ensure_ascii=False, indent=2) + "\n"
        )
        (directories["prompts"] / _supplement_prompt_filename(unit_id)).write_text(
            render_supplement_prompt(supplement_unit_payload, prompt_template=prompt_template)
        )

        written_unit_ids.append(unit_id)

    manifest = {
        "run_id": resolved_run_id,
        "run_type": "supplement",
        "schema_version": SUPPLEMENT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_path": str(SUPPLEMENT_PROMPT_PATH.resolve()),
        "unit_ids": written_unit_ids,
        # Portable, relative-to-run-dir directory paths (see deliverable 1 in
        # runner.py's manifest path portability fix) rather than baked
        # absolute paths.
        "directories": {name: name for name in directories},
        "alias_map": {},
        "notes": notes,
        "derived_from": {
            "audit_path": str(Path(audit_path)),
            "source_run_ids": sorted(source_run_ids),
        },
        "automation": None,
        "benchmark": None,
    }
    runner._write_run_manifest(output_path, manifest)
    return manifest


def _augment_with_known_sources(annotation, accepted_characters, candidate_characters, roster_characters=None):
    # Supplement annotations list only the candidate characters they scored in
    # characters_present, but accepted characters, unscored candidates, and any
    # named character from the unit's alias roster remain legitimate event
    # sources (e.g. a hostess praising a candidate without being scored herself).
    # The base validator requires sources to appear in characters_present, so
    # validate against a view that includes all known unit characters as
    # implicit presences. Targets stay restricted to candidates separately.
    known = (
        list(accepted_characters or [])
        + list(candidate_characters or [])
        + list(roster_characters or [])
    )
    if not isinstance(annotation, dict) or not known:
        return annotation
    present = list(annotation.get("characters_present") or [])
    listed = {
        row.get("canonical_name")
        for row in present
        if isinstance(row, dict)
    }
    for name in known:
        if name in listed:
            continue
        listed.add(name)
        present.append(
            {
                "canonical_name": name,
                "surface_forms": [name],
                "presence_type": "implicit",
                "presence_confidence": 1.0,
            }
        )
    view = dict(annotation)
    view["characters_present"] = present
    return view


def normalize_supplement_ambiguities(annotation):
    # The production reducer keeps at most one ambiguity, and only when an event
    # carries an uncertain stance; the accepted corpus is post-reduction. The
    # scorer charges every character in the unit per ambiguity flag, so supplement
    # annotations must follow the same rule or their net scores skew negative
    # relative to accepted characters in the same unit.
    if not isinstance(annotation, dict):
        return annotation
    normalized = dict(annotation)
    events = normalized.get("appraisal_events") or []
    ambiguities = normalized.get("ambiguities") or []
    keep = (
        [ambiguities[0]]
        if ambiguities
        and any(
            isinstance(event, dict) and event.get("narrative_stance") == "uncertain"
            for event in events
        )
        else []
    )
    normalized["ambiguities"] = keep
    return normalized


def validate_supplement_result(annotation, expected_unit_id=None, accepted_characters=None, candidate_characters=None, roster_characters=None):
    validation_view = _augment_with_known_sources(annotation, accepted_characters, candidate_characters, roster_characters)
    errors = list(runner.validate_annotation_result(validation_view, expected_unit_id=expected_unit_id))
    if not isinstance(annotation, dict):
        return errors

    accepted = set(accepted_characters) if accepted_characters is not None else None
    candidates = set(candidate_characters) if candidate_characters is not None else None

    for index, effect in enumerate(annotation.get("status_effects") or []):
        if not isinstance(effect, dict):
            continue
        prefix = f"status_effects[{index}]"
        character = effect.get("character")
        if accepted is not None and character in accepted:
            errors.append(f'{prefix}.character "{character}" must not overlap an already-accepted character.')
        if candidates is not None and character not in candidates:
            errors.append(f'{prefix}.character "{character}" must be one of the candidate characters.')

    for index, event in enumerate(annotation.get("appraisal_events") or []):
        if not isinstance(event, dict):
            continue
        prefix = f"appraisal_events[{index}]"
        target = event.get("target")
        if accepted is not None and target in accepted:
            errors.append(f'{prefix}.target "{target}" must not overlap an already-accepted character.')
        if candidates is not None and target not in candidates:
            errors.append(f'{prefix}.target "{target}" must be one of the candidate characters.')

        event_id = event.get("event_id")
        if isinstance(event_id, str) and event_id and not event_id.startswith("S"):
            errors.append(f'{prefix}.event_id "{event_id}" must use the "S" prefix for supplement events.')

    return errors


def _accent_insensitive_key(name):
    import unicodedata

    return unicodedata.normalize("NFKD", (name or "").lower()).encode("ascii", "ignore").decode()


def canonicalize_supplement_names(annotation, known_characters):
    # Reconcile annotation character names against the unit's known universe:
    # first through the reviewed merge table, then accent-insensitively toward
    # the known form when exactly one known character matches. Annotators
    # normalize French orthography unpredictably (Remi/Rémi, ouvriere/ouvrière),
    # and an unreconciled spelling would split one person into two identities
    # downstream. Ambiguous or unknown names are left unchanged for the
    # validator to reject.
    if not isinstance(annotation, dict):
        return annotation
    known = [translate_canonical_name(name) for name in known_characters]
    by_key = {}
    for name in known:
        by_key.setdefault(_accent_insensitive_key(name), set()).add(name)

    def reconcile(name):
        translated = translate_canonical_name(name)
        if translated in known:
            return translated
        matches = by_key.get(_accent_insensitive_key(translated)) or set()
        if len(matches) == 1:
            return next(iter(matches))
        return translated

    result = json.loads(json.dumps(annotation))
    for row in result.get("characters_present") or []:
        if isinstance(row, dict) and isinstance(row.get("canonical_name"), str):
            row["canonical_name"] = reconcile(row["canonical_name"])
    for event in result.get("appraisal_events") or []:
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("target"), str):
            event["target"] = reconcile(event["target"])
        source = event.get("source")
        if isinstance(source, str) and source not in ("narrator", "collective_social_voice", "unknown"):
            reconciled = reconcile(source)
            # One-off named figures outside the tracked universe (visiting
            # royalty, historical names) are legitimate evaluators but can
            # never be tracked characters; the schema's bucket for them is
            # "unknown", and the event evidence text preserves the attribution.
            event["source"] = reconciled if reconciled in known else "unknown"
    for effect in result.get("status_effects") or []:
        if isinstance(effect, dict) and isinstance(effect.get("character"), str):
            effect["character"] = reconcile(effect["character"])
    return result


def write_supplement_result(run_dir, unit_id, annotation):
    run_path = Path(run_dir)
    unit_payload = runner._read_json(run_path / "units" / _supplement_unit_filename(unit_id))
    candidate_characters = [
        translate_canonical_name(name) for name in unit_payload.get("candidate_characters") or []
    ]
    accepted_annotation = unit_payload.get("accepted_annotation") or {}
    accepted_characters = [
        translate_canonical_name(character.get("canonical_name"))
        for character in accepted_annotation.get("characters_present", [])
    ]

    known_universe = (
        candidate_characters
        + accepted_characters
        + list((unit_payload.get("alias_map") or {}).keys())
    )
    annotation = canonicalize_supplement_names(annotation, known_universe)
    annotation = normalize_supplement_ambiguities(annotation)
    errors = validate_supplement_result(
        annotation,
        expected_unit_id=unit_id,
        accepted_characters=accepted_characters,
        candidate_characters=candidate_characters,
        roster_characters=[
            translate_canonical_name(name) for name in (unit_payload.get("alias_map") or {})
        ],
    )
    if errors:
        raise ValueError(f'Invalid supplement annotation for unit "{unit_id}": {"; ".join(errors)}')

    annotation_path = run_path / "annotations" / _supplement_annotation_filename(unit_id)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n")
    return annotation_path
