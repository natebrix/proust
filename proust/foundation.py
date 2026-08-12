"""Foundation re-annotation machinery (registry design migration steps 2-3).

The foundation pass re-annotates the WHOLE canonical corpus from scratch on
the current fr-original text (Wikisource, 4,779 paragraphs over 18 chapters)
with prompt v2 + a per-chapter registry reference sheet, replacing the legacy
closed-world alias-map pass. See proust/docs/character_registry_design.md.

Shape of the machinery, mirroring the supplement pass:

- derive_foundation_unit_specs: fresh ~5-paragraph windows over a chapter (or
  the whole corpus) in canonical order.
- prepare_foundation_run: standard run directory (run.json, units/, prompts/,
  raw/, annotations/) whose prompts are prompt v2 with {{REFERENCE_SHEET}}
  filled from Registry.load().reference_sheet(chapter_id). The registry is
  read fresh at preparation time, and run.json records the sha256 of
  characters.yaml the run was prepared under (migration step 5).
- validate_foundation_result / write_foundation_result: the v1 validator's
  checks extended to accept the optional "resolution" field prompt v2 puts on
  characters_present entries. runner.validate_annotation_result is left
  untouched; this module wraps it over a v1-shaped view.

Storage decision: annotations/ hold the v1-shaped annotation (no "resolution"
key) so every existing downstream consumer -- runner.get_run_status,
_score_run_outcomes, build_outcome_report, the coverage audit, the aggregation
surfaces -- keeps working unchanged against foundation runs. The resolution
information is not lost: raw/ keeps the model's verbatim output, and
write_foundation_result also writes a small per-unit record under
resolutions/ so the unresolved-name triage inventory survives re-gating.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import runner
from .annotation import AnnotationUnitSpec, build_annotation_unit, _load_canonical_chapter_json
from .coordinates import annotation_unit_id
from .export import CANONICAL_CHAPTER_SPECS
from .registry import REGISTRY_PATH, Registry


FOUNDATION_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "prompt_v2.md"
FOUNDATION_PROMPT_VERSION = "v2"
FOUNDATION_RUN_TYPE = "foundation"
FOUNDATION_SCHEMA_VERSION = "annotation_foundation_v1"

DEFAULT_WINDOW = 5
DEFAULT_RUN_SIZE = 40
DEFAULT_RUN_PREFIX = "foundation-run-"
DEFAULT_OUTPUTS_DIR = "outputs"
DEFAULT_PLAN_PATH = "outputs/foundation-plan.json"

ALLOWED_RESOLUTIONS = {"resolved", "unresolved"}

_UNIT_ID_SPAN = re.compile(r"^p-(\d+)(?:-p-(\d+))?$")


def canonical_chapter_ids():
    return [spec.id for spec in CANONICAL_CHAPTER_SPECS]


def _chapter_spec(chapter_id):
    for spec in CANONICAL_CHAPTER_SPECS:
        if spec.id == chapter_id:
            return spec
    raise ValueError(f'Unknown canonical chapter id "{chapter_id}".')


def _chapter_paragraph_texts(chapter_id):
    """Raw paragraph texts of a canonical chapter, in paragraph order."""
    chapter = _load_canonical_chapter_json(chapter_id)
    return [paragraph["text"] for paragraph in chapter["paragraphs"]]


def _unit_filename(unit_id):
    return f"{unit_id}.json"


def _prompt_filename(unit_id):
    # runner._prompt_filename's ".txt" convention, so runner's status,
    # automation, and reprocessing paths work on foundation runs unchanged.
    return f"{unit_id}.txt"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


# --------------------------------------------------------------- unit windows


def foundation_unit_id(spec):
    return annotation_unit_id(spec.chapter_id, spec.paragraph_start, spec.paragraph_end)


def parse_foundation_unit_id(unit_id):
    """Inverse of foundation_unit_id: "v5#p-7-p-11" -> AnnotationUnitSpec."""
    chapter_id, separator, span = unit_id.partition("#")
    match = _UNIT_ID_SPAN.match(span) if separator else None
    if not match:
        raise ValueError(f'Malformed foundation unit id "{unit_id}".')
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return AnnotationUnitSpec(chapter_id=chapter_id, paragraph_start=start, paragraph_end=end)


def _trim_empty_edges(paragraph_texts, start, end):
    """Shrink [start, end] past empty paragraphs at either edge.

    Returns None when the whole window is empty text (nothing to annotate).
    Interior empty paragraphs are kept: dropping them would split one window
    into two non-contiguous spans, which the unit id convention cannot express.
    """
    while start <= end and not paragraph_texts[start - 1].strip():
        start += 1
    while end >= start and not paragraph_texts[end - 1].strip():
        end -= 1
    if start > end:
        return None
    return start, end


def derive_foundation_unit_specs(chapter_id=None, window=DEFAULT_WINDOW):
    """Fresh annotation windows over the current canonical text.

    Consecutive spans of `window` paragraphs in paragraph order; a chapter's
    tail keeps whatever remainder is left (at least one paragraph). Windows
    are trimmed past empty paragraphs at their edges, and a window that would
    contain only empty paragraphs is skipped entirely.

    With chapter_id=None this walks every canonical chapter in reader order.
    """
    if window < 1:
        raise ValueError("window must be at least 1 paragraph.")

    chapter_ids = [chapter_id] if chapter_id else canonical_chapter_ids()
    specs = []

    for current_chapter_id in chapter_ids:
        _chapter_spec(current_chapter_id)
        paragraph_texts = _chapter_paragraph_texts(current_chapter_id)
        paragraph_count = len(paragraph_texts)
        start = 1
        while start <= paragraph_count:
            end = min(start + window - 1, paragraph_count)
            trimmed = _trim_empty_edges(paragraph_texts, start, end)
            if trimmed is not None:
                specs.append(
                    AnnotationUnitSpec(
                        chapter_id=current_chapter_id,
                        paragraph_start=trimmed[0],
                        paragraph_end=trimmed[1],
                    )
                )
            start = end + 1

    return specs


# ------------------------------------------------------------------ preparing


def registry_content_hash(path=REGISTRY_PATH):
    """sha256 of characters.yaml, recorded in run.json (migration step 5)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_foundation_prompt_template(path=FOUNDATION_PROMPT_PATH):
    return Path(path).read_text()


def render_foundation_prompt(unit_payload, reference_sheet, prompt_template=None):
    """Render prompt v2 for a unit payload.

    Prompt v2's own text is explicit that the passage is "Proust's original,
    unaltered" and that the reference sheet is advisory context, so the
    passage is the unit's raw_text and the sheet goes in verbatim as
    pretty-printed JSON.
    """
    template = prompt_template or load_foundation_prompt_template()
    prior_context = unit_payload.get("prior_context") or "[none]"
    return (
        template.replace("{{REFERENCE_SHEET}}", json.dumps(reference_sheet, ensure_ascii=False, indent=2))
        .replace("{{PRIOR_CONTEXT}}", prior_context)
        .replace("{{PASSAGE}}", unit_payload["raw_text"])
    )


def build_foundation_unit(chapter_id, paragraph_start, paragraph_end=None, prior_context_paragraphs=1):
    """Unit payload on the current canonical text, with no alias rewriting.

    build_annotation_unit falls back to DEFAULT_STARTER_ALIAS_MAP whenever it
    is handed an empty alias map, and derives prior_context from the
    alias-rewritten text. The foundation pass rewrites nothing, so the payload
    is normalized afterwards: empty alias_map, preprocessed_text == raw_text,
    and prior context taken from the raw paragraphs.
    """
    unit = build_annotation_unit(
        chapter_id,
        paragraph_start,
        paragraph_end=paragraph_end,
        prior_context_paragraphs=prior_context_paragraphs,
    )
    paragraph_texts = _chapter_paragraph_texts(chapter_id)
    prior_start = max(1, paragraph_start - prior_context_paragraphs)
    unit["alias_map"] = {}
    unit["preprocessed_text"] = unit["raw_text"]
    unit["prior_context"] = "\n\n".join(paragraph_texts[prior_start - 1 : paragraph_start - 1])
    return unit


def prepare_foundation_run(output_dir, chapter_id, unit_specs=None, run_id=None, notes=""):
    """Prepare one foundation run directory for a single chapter.

    The registry is loaded fresh here, so a reference sheet always reflects
    characters.yaml as of preparation time; run.json records the sha256 that
    sheet was rendered from.
    """
    output_path = Path(output_dir)
    resolved_run_id = run_id or output_path.name
    selected_specs = list(unit_specs) if unit_specs is not None else derive_foundation_unit_specs(chapter_id)

    foreign = sorted({spec.chapter_id for spec in selected_specs} - {chapter_id})
    if foreign:
        raise ValueError(
            f'Foundation run "{resolved_run_id}" is scoped to chapter "{chapter_id}" but was given '
            f'units from: {", ".join(foreign)}. One run renders one chapter reference sheet.'
        )

    directories = runner._ensure_run_directories(output_path)
    registry = Registry.load()
    reference_sheet = registry.reference_sheet(chapter_id=chapter_id)
    prompt_template = load_foundation_prompt_template()

    unit_ids = []
    for spec in selected_specs:
        unit = build_foundation_unit(
            spec.chapter_id,
            spec.paragraph_start,
            paragraph_end=spec.paragraph_end,
            prior_context_paragraphs=1,
        )
        if spec.notes:
            unit["notes"] = spec.notes
        unit_id = unit["unit_id"]
        unit_ids.append(unit_id)

        _write_json(directories["units"] / _unit_filename(unit_id), unit)
        prompt_text = render_foundation_prompt(unit, reference_sheet, prompt_template=prompt_template)
        (directories["prompts"] / _prompt_filename(unit_id)).write_text(prompt_text)

    _write_json(output_path / "reference-sheet.json", reference_sheet)

    manifest = {
        "run_id": resolved_run_id,
        "run_type": FOUNDATION_RUN_TYPE,
        "schema_version": FOUNDATION_SCHEMA_VERSION,
        "prompt_version": FOUNDATION_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Portable paths only: a basename here resolves under proust/prompts/,
        # and the directory names resolve under the run directory itself.
        "prompt_path": "proust/prompts/prompt_v2.md",
        "chapter_id": chapter_id,
        "unit_ids": unit_ids,
        "directories": {name: name for name in directories},
        "alias_map": {},
        "registry": {
            "path": "characters.yaml",
            "content_sha256": registry_content_hash(),
            "reference_sheet_path": "reference-sheet.json",
            "reference_sheet_entity_count": len(reference_sheet),
        },
        "notes": notes,
        "automation": None,
        "benchmark": None,
    }
    runner._write_run_manifest(output_path, manifest)
    return manifest


# ---------------------------------------------------------------- batch plan


def build_foundation_plan(
    window=DEFAULT_WINDOW,
    run_size=DEFAULT_RUN_SIZE,
    outputs_dir=DEFAULT_OUTPUTS_DIR,
    run_prefix=DEFAULT_RUN_PREFIX,
):
    """Split the whole corpus into <= run_size unit batches, in canonical order.

    Runs never straddle a chapter boundary: one run renders one chapter's
    reference sheet, so a chapter with more than run_size units simply gets
    several consecutive runs.
    """
    if run_size < 1:
        raise ValueError("run_size must be at least 1 unit.")

    runs = []
    chapters = []
    volumes = {}

    for chapter_id in canonical_chapter_ids():
        spec = _chapter_spec(chapter_id)
        chapter_specs = derive_foundation_unit_specs(chapter_id, window=window)
        chapter_run_ids = []
        for offset in range(0, len(chapter_specs), run_size):
            batch = chapter_specs[offset : offset + run_size]
            run_id = f"{run_prefix}{len(runs) + 1:03d}"
            chapter_run_ids.append(run_id)
            runs.append(
                {
                    "runId": run_id,
                    "runDir": f"{outputs_dir}/{run_id}",
                    "chapterId": chapter_id,
                    "volume": spec.volume_number,
                    "unitCount": len(batch),
                    "unitIds": [foundation_unit_id(unit_spec) for unit_spec in batch],
                }
            )
        chapters.append(
            {
                "chapterId": chapter_id,
                "volume": spec.volume_number,
                "volumeTitle": spec.volume_title,
                "unitCount": len(chapter_specs),
                "runIds": chapter_run_ids,
            }
        )
        volume = volumes.setdefault(
            spec.volume_number,
            {"volume": spec.volume_number, "volumeTitle": spec.volume_title, "unitCount": 0, "runCount": 0},
        )
        volume["unitCount"] += len(chapter_specs)
        volume["runCount"] += len(chapter_run_ids)

    return {
        "plan_version": "foundation_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": FOUNDATION_PROMPT_VERSION,
        "window": window,
        "run_size": run_size,
        "registry_content_sha256": registry_content_hash(),
        "unit_count": sum(run["unitCount"] for run in runs),
        "run_count": len(runs),
        "volumes": [volumes[key] for key in sorted(volumes)],
        "chapters": chapters,
        "runs": runs,
    }


def write_foundation_plan(plan, path=DEFAULT_PLAN_PATH):
    return _write_json(Path(path), plan)


def load_foundation_plan(path=DEFAULT_PLAN_PATH):
    return json.loads(Path(path).read_text())


def plan_run_entry(plan, run_id):
    for run in plan["runs"]:
        if run["runId"] == run_id:
            return run
    raise ValueError(f'Run "{run_id}" is not in the foundation plan.')


def prepare_planned_run(plan, run_id, output_dir=None, notes=""):
    """Prepare one run named by a plan built with build_foundation_plan."""
    entry = plan_run_entry(plan, run_id)
    specs = [parse_foundation_unit_id(unit_id) for unit_id in entry["unitIds"]]
    return prepare_foundation_run(
        output_dir or entry["runDir"],
        entry["chapterId"],
        specs,
        run_id=run_id,
        notes=notes or f'foundation pass, {entry["chapterId"]} ({entry["unitCount"]} units)',
    )


# ----------------------------------------------------------------- validation


def _v1_annotation_view(annotation):
    """The annotation as prompt v1's schema sees it: no "resolution" key."""
    if not isinstance(annotation, dict):
        return annotation
    characters = annotation.get("characters_present")
    if not isinstance(characters, list):
        return annotation
    view = dict(annotation)
    view["characters_present"] = [
        {key: value for key, value in character.items() if key != "resolution"}
        if isinstance(character, dict)
        else character
        for character in characters
    ]
    return view


def validate_foundation_result(annotation, expected_unit_id=None):
    """runner.validate_annotation_result plus prompt v2's "resolution" field.

    An unresolved character is valid output, not an error: prompt v2 asks the
    annotator to name every materially involved character even when the
    reference sheet has no entry for them. Only a malformed resolution value
    is an error.
    """
    errors = list(
        runner.validate_annotation_result(_v1_annotation_view(annotation), expected_unit_id=expected_unit_id)
    )
    if not isinstance(annotation, dict):
        return errors

    characters = annotation.get("characters_present")
    if isinstance(characters, list):
        for index, character in enumerate(characters):
            if not isinstance(character, dict) or "resolution" not in character:
                continue
            resolution = character["resolution"]
            if resolution not in ALLOWED_RESOLUTIONS:
                errors.append(
                    f"characters_present[{index}].resolution must be one of: "
                    f"{', '.join(sorted(ALLOWED_RESOLUTIONS))}."
                )
    return errors


def resolution_summary(annotation, unit_id=None):
    """Per-unit resolution record: how many named characters went unresolved."""
    characters = []
    if isinstance(annotation, dict) and isinstance(annotation.get("characters_present"), list):
        characters = [character for character in annotation["characters_present"] if isinstance(character, dict)]

    unresolved = [
        character.get("canonical_name")
        for character in characters
        if character.get("resolution") == "unresolved"
    ]
    return {
        "unit_id": unit_id or (annotation.get("unit_id") if isinstance(annotation, dict) else None),
        "character_count": len(characters),
        "unresolved_count": len(unresolved),
        "unresolved_names": unresolved,
    }


_RECONCILE_REGISTRY = None


def _registry():
    global _RECONCILE_REGISTRY
    if _RECONCILE_REGISTRY is None:
        _RECONCILE_REGISTRY = Registry.load()
    return _RECONCILE_REGISTRY


def reconcile_foundation_names(annotation, chapter_id=None):
    """Unify character-name variants through the registry at ingest.

    Annotators legitimately write surface variants ("M. de Cambremer") for a
    character they list canonically elsewhere ("marquis de Cambremer"): the
    registry resolves both to one entity, so all names in the annotation are
    rewritten to that entity's canonical annotation name. Names the registry
    cannot resolve (open-world discoveries) or resolves ambiguously are left
    untouched; a registry-resolved name also corrects the entry's
    "resolution" flag to "resolved".
    """
    if not isinstance(annotation, dict):
        return annotation
    registry = _registry()

    def unify(name):
        if not isinstance(name, str) or name in ("narrator", "collective_social_voice", "unknown"):
            return name
        resolution = registry.resolve(name, chapter_id=chapter_id)
        if resolution.status != "resolved":
            return name
        entity = registry.entities[resolution.entity_id]
        return (entity.annotation_names[0] if entity.annotation_names else entity.display_name)

    result = json.loads(json.dumps(annotation))
    for row in result.get("characters_present") or []:
        if isinstance(row, dict) and isinstance(row.get("canonical_name"), str):
            unified = unify(row["canonical_name"])
            if unified != row["canonical_name"]:
                row["canonical_name"] = unified
            # correct the flag only when it holds a legal value or is absent;
            # a malformed value must survive for the validator to reject
            if row.get("resolution") in (None, "resolved", "unresolved") and (
                registry.resolve(row["canonical_name"], chapter_id=chapter_id).status == "resolved"
            ):
                row["resolution"] = "resolved"
    present_names = {
        row.get("canonical_name")
        for row in result.get("characters_present") or []
        if isinstance(row, dict)
    }
    for event in result.get("appraisal_events") or []:
        if isinstance(event, dict):
            if isinstance(event.get("source"), str):
                unified_source = unify(event["source"])
                # An event source that is neither a listed character nor a
                # special value cannot validate. Named open-world sources are
                # listed in characters_present by the prompt's own rules, so
                # what reaches this branch is descriptor sources ("le patron")
                # and stray unresolvables: the schema's bucket for those is
                # "unknown", with the evidence text preserving the attribution.
                if unified_source not in present_names and unified_source not in (
                    "narrator", "collective_social_voice", "unknown"
                ):
                    unified_source = "unknown"
                event["source"] = unified_source
            if isinstance(event.get("target"), str):
                event["target"] = unify(event["target"])
    for effect in result.get("status_effects") or []:
        if isinstance(effect, dict) and isinstance(effect.get("character"), str):
            effect["character"] = unify(effect["character"])
    return result


def write_foundation_result(run_dir, unit_id, annotation):
    """Validate and write one foundation annotation.

    annotations/<unit_id>.json holds the v1-shaped annotation so all existing
    scoring and reporting tooling applies unchanged; resolutions/<unit_id>.json
    keeps the resolution record for triage.
    """
    run_path = Path(run_dir)
    if isinstance(annotation, dict) and not annotation.get("unit_id"):
        annotation = dict(annotation, unit_id=unit_id)

    chapter_id = unit_id.split("#", 1)[0] if isinstance(unit_id, str) else None
    annotation = reconcile_foundation_names(annotation, chapter_id=chapter_id)
    errors = validate_foundation_result(annotation, expected_unit_id=unit_id)
    if errors:
        raise ValueError(f'Invalid foundation annotation for unit "{unit_id}": {"; ".join(errors)}')

    annotation_path = _write_json(
        run_path / "annotations" / _unit_filename(unit_id), _v1_annotation_view(annotation)
    )
    _write_json(run_path / "resolutions" / _unit_filename(unit_id), resolution_summary(annotation, unit_id=unit_id))
    return annotation_path
