# Annotation Runner Notes

## Current helper surface

The first annotation layer is now available through the `proust.annotation` helpers.

Primary functions:

- `build_annotation_unit(chapter_id, paragraph_start, paragraph_end=None, prior_context_paragraphs=0, alias_map=None, session=None)`
- `build_starter_units(alias_map=None, session=None)`
- `render_prompt_input(unit_payload, prompt_template=None)`

The run preparation layer is available through `proust.runner`.

Primary functions:

- `prepare_annotation_run(output_dir, run_id=None, unit_specs=None, alias_map=None, prompt_path=None, notes="")`
- `write_raw_response(run_dir, unit_id, raw_text)`
- `write_annotation_result(run_dir, unit_id, annotation)`
- `validate_annotation_result(annotation, expected_unit_id=None)`
- `get_run_status(run_dir)`
- `mark_run_as_benchmark(run_dir, label="reviewed benchmark")`
- `compare_run_to_benchmark(run_dir, benchmark_run_dir)`
- `prepare_annotation_run_from_existing(source_run_dir, output_dir, run_id=None, notes="")`
- `run_openai_annotation(source_run_dir, output_dir, model="gpt-5", overwrite=False, limit=None, api_key=None)`
- `run_automated_batch(source_run_dir, output_dir, model="gpt-5", overwrite=False, limit=None, poll_interval=5.0, timeout=None, progress_stream=None, max_mixed_units_per_lens=3)`
- `score_run_local_outcomes(run_dir)`
- `score_run_inclusion_outcomes(run_dir)`
- `score_run_prestige_outcomes(run_dir)`
- `build_outcome_report(run_dir)`

## Starter units

`build_starter_units()` returns three canonical prompt payloads:

- `v1-p1-combray#p-17`
- `v1-p1-combray#p-274-p-275`
- `v1-p1-combray#p-312-p-313`

Each payload includes:

- canonical `unit_id`
- canonical `chapter_id`
- paragraph span
- `raw_text`
- `preprocessed_text`
- `reader_urls` for French and English
- `alias_map`
- optional `prior_context`

## Typical usage

```python
from proust import build_starter_units, prepare_annotation_run, render_prompt_input

units = build_starter_units()

for unit in units:
    prompt_text = render_prompt_input(unit)
    print(unit["unit_id"])
    print(unit["reader_urls"]["fr-original"])

prepare_annotation_run("outputs/run-001", notes="starter batch")
```

To validate a reviewed run and mark it as a benchmark:

```python
from proust import get_run_status, mark_run_as_benchmark

status = get_run_status("outputs/run-001")
print(status["summary"])

mark_run_as_benchmark("outputs/run-001", label="starter reviewed benchmark")
```

CLI equivalents:

```bash
python -m proust status --run outputs/run-001
python -m proust status --run outputs/run-001 --write-benchmark --label "starter reviewed benchmark"
python -m proust compare --run outputs/run-002 --benchmark outputs/run-001
python -m proust score --run outputs/run-008
python -m proust score --run outputs/run-008 --lens inclusion
python -m proust score --run outputs/run-008 --lens prestige
python -m proust report --run outputs/run-008
python -m proust report --run outputs/run-008 --lens inclusion
python -m proust report --run outputs/run-008 --lens prestige
python -m proust automate --source-run outputs/run-002 --output outputs/run-003 --model gpt-5
python -m proust wait --run outputs/run-003 --reduce --report
python -m proust batch --source-run outputs/run-002 --output outputs/run-003 --model gpt-5
```

Important CLI note:

- `python -m proust prepare` only scaffolds an empty run directory and manifest
- it does not currently expose `unit_specs` or `alias_map`
- production source runs with explicit paragraph spans are therefore prepared through `prepare_annotation_run(...)` in Python, or by cloning an existing source run with `prepare_annotation_run_from_existing(...)`

`batch` is the one-command version of the normal production loop:

- launch automation from a prepared source run
- wait for completion
- reprocess with reduction
- build all three lens reports
- return exit code `0` when the batch is clean
- return exit code `2` when a conservative review gate is tripped

Current review-gate triggers are intentionally narrow:

- any parse errors
- any validation errors
- any cross-lens sign flip for the same unit/character
- more than `3` mixed units in a single lens by default

To compare another run against the reviewed benchmark:

```python
from proust import compare_run_to_benchmark

comparison = compare_run_to_benchmark("outputs/run-002", "outputs/run-001")
print(comparison["summary"])
```

## Recommended output layout

For each annotation run, persist:

- prompt payload JSON keyed by `unit_id`
- raw model response keyed by `unit_id`
- normalized annotation JSON keyed by `unit_id`
- run metadata including prompt version and model settings

Suggested directory shape:

```text
outputs/
  run-001/
    units/
      v1-p1-combray#p-17.json
    raw/
      v1-p1-combray#p-17.txt
    annotations/
      v1-p1-combray#p-17.json
    run.json
```

`raw/` is optional for lightweight manual review. A run is benchmark-ready when each declared unit has:

- a unit payload in `units/`
- a rendered prompt in `prompts/`
- a valid reviewed annotation in `annotations/`

The validation layer checks the reviewed annotation schema directly against the current prompt contract:

- exact top-level keys only
- required nested keys only
- enum values for event types, stance, polarity, and status dimensions
- `based_on_events` must reference declared event ids
- event targets and status-effect characters must appear in `characters_present`

`get_run_status(...)` reports per-unit status and a run summary:

- `annotation_valid`
- `review_state`
- `benchmark_ready`
- counts for prompts, raw files, annotations, and reviewed units

`mark_run_as_benchmark(...)` writes a `benchmark` block into `run.json` with:

- benchmark label
- validation timestamp
- reviewed unit ids
- valid annotation count
- pending unit count
- overall `benchmark_ready` status

`compare_run_to_benchmark(...)` compares a candidate run against a reviewed benchmark and reports:

- shared, benchmark-only, and run-only unit counts
- exact annotation matches on shared units
- differing reviewed annotations
- missing reviewed annotations on either side

This is intentionally a filesystem-level comparison, not a semantic scorer. It is meant to answer:

- did the candidate run cover the same units?
- did it produce reviewed annotations for them?
- where did the saved JSON diverge from the benchmark?

## Lightweight local outcome scoring

`score_run_local_outcomes(...)` is the first downstream transformation from reduced annotations into candidate local "winning" and "losing" signals.

Its purpose is limited:

- score each validated unit locally, not globally
- keep the transformation legible and revisable
- use only the current reduced schema

Current v1 rules:

- event polarity contributes directional signal on the target character
- status deltas carry most of the weight
- `social_status` and `inclusion_exclusion` count slightly more than `general_appraisal`
- `ironized` and `uncertain` events are discounted so unstable rhetorical moments do not over-register
- each ambiguity flag applies a small penalty to keep unstable units from reading as clean wins or losses

The output includes:

- per-unit per-character `event_score`
- per-unit per-character `status_score`
- per-unit per-character `net_score`
- a coarse label: `win`, `loss`, `mixed`, or `neutral`
- run-level character totals

This is intentionally a lightweight exploratory layer, not a canonical literary interpretation.

## Outcome report layer

`build_outcome_report(...)` is a compact downstream report built directly on top of `local_outcome_v1`.

It is meant to make the score output easier to read, not to introduce a second scoring scheme.

The report includes:

- per-character summaries with total scores and label counts
- an ordered per-unit timeline
- top positive units
- top negative units
- mixed units called out explicitly

This is the preferred first downstream view for collaborative review of a scored run.

The next recommended downstream comparison is not a retuning of `local_outcome_v1`, but a parallel lens comparison built from the same reduced annotations:

- a prestige-weighted view
- an inclusion-weighted view

This is motivated by cases like Swann, where local prestige and local incorporation can diverge sharply.

The currently implemented alternatives are:

- `prestige_outcome_v1` via `--lens prestige`
- `inclusion_outcome_v1` via `--lens inclusion`

## Automated run workflow

The automated path assumes:

- a reviewed or candidate source run already exists
- its `units/` and `prompts/` directories define the candidate set
- the automated outputs should be written to a fresh run directory

Typical source-run preparation for an explicit batch currently happens through Python, not the CLI:

```python
import json
from pathlib import Path

from proust.annotation import AnnotationUnitSpec
from proust.runner import prepare_annotation_run

alias_map = json.loads(Path("outputs/run-399/run.json").read_text())["alias_map"]

unit_specs = [
    AnnotationUnitSpec("v3-p1", 281, 285),
    AnnotationUnitSpec("v3-p1", 286, 290),
    AnnotationUnitSpec("v3-p1", 291, 295),
    AnnotationUnitSpec("v3-p1", 296, 300),
    AnnotationUnitSpec("v3-p1", 301, 305),
    AnnotationUnitSpec("v3-p1", 306, 310),
    AnnotationUnitSpec("v3-p1", 311, 315),
    AnnotationUnitSpec("v3-p1", 316, 320),
]

prepare_annotation_run(
    "outputs/run-403",
    unit_specs=unit_specs,
    alias_map=alias_map,
    notes="source batch for v3-p1 p-281 through p-320",
)
```

Operational rule:

- carry forward the alias map from the last accepted source run unless a real coverage gap appears
- define explicit `AnnotationUnitSpec` spans for the next contiguous batch
- then launch `python -m proust batch --source-run ... --output ... --model gpt-5`

Typical flow:

```python
from proust import run_openai_annotation

run_openai_annotation(
    "outputs/run-002",
    "outputs/run-003",
    model="gpt-5",
)
```

What this does:

- clones the unit set and prompts from the source run into the output run
- requests one model completion per prompt through the OpenAI Responses API
- writes raw model text into `raw/`
- parses JSON responses
- validates them against the current annotation schema
- writes valid annotations into `annotations/`
- records an `automation` block in `run.json`

The automated run keeps the source run untouched, so a reviewed baseline like `run-002` remains stable.

## Current review workflow

The current project phase is no longer benchmark construction.

The default workflow for new scaled runs is:

1. prepare the next contiguous batch
2. automate it to a fresh run directory
3. apply reduction and scoring
4. read the report outputs first
5. inspect individual units only if the reports show a genuinely surprising signal

This means report-reading is now the primary review surface.

Unit-by-unit inspection is now an exception path used for:

- an implausible character arc
- a strong unexplained divergence between report lenses
- a recurring directional mistake in the same passage type
- a suspected alias, parsing, or reduction failure that affects multiple units

Unit-by-unit inspection is not the default response to:

- familiar tolerated edge cases
- mild weighting disagreements
- isolated debatable mixed-unit judgments
- lack of exact equality with an older benchmark run

## Scaling rule of thumb

Continue scaling in modest contiguous batches while both of these remain true:

- the reports are directionally plausible
- no recurring failure class is distorting downstream interpretation

Shift to larger automated chunks once the current contiguous block is comfortably larger and the reports remain stable.

Pause scaling and intervene only when a recurring failure class begins to matter at the report level.

## Long-run monitoring

Some prepared prompts are now large enough that automated requests can take a while even when they are succeeding.

Do not assume an API failure just because the command is quiet for a long stretch.

When checking a live automated run, prefer inspecting `run.json` and the output directories:

- `automation.in_progress`
- `automation.completed_unit_count`
- `automation.requested_unit_count`
- files appearing in `raw/`
- files appearing in `annotations/`

If a run is progressing, let it continue.

If a run appears stuck, first verify whether units are still being written before retrying.

If partial progress exists and the process was interrupted, rerunning `automate` is acceptable because completed units will normally be skipped unless `--overwrite` is used.

If you want the CLI to block until a run finishes and then post-process it, use:

```bash
python -m proust wait --run outputs/run-003 --reduce --report
```

`wait` polls `run.json` until `automation.in_progress` becomes `false`, prints progress updates when counts change, and can chain:

- `reprocess --overwrite --reduce`
- `report --lens local`
- `report --lens prestige`
- `report --lens inclusion`

If you are driving many long runs from Codex unified exec sessions, periodically close completed sessions or reuse still-relevant ones.

Otherwise, the session limit warning can become the practical bottleneck before the annotation pipeline itself does.

## Full-corpus procedure

The project now has a dedicated full-corpus operating procedure in:

- [full_corpus_runbook.md](/Users/nathan_brixius/dev/proust/proust/docs/full_corpus_runbook.md:1)

That runbook now governs:

- chapter order
- batch size
- cursor advancement
- retry and resumption behavior
- stop conditions
- monitoring expectations

## Historical note

Build a tiny batch runner that:

- calls `build_starter_units()`
- renders prompt text with `render_prompt_input()`
- sends the prompt to the chosen model
- stores raw and normalized outputs by `unit_id`

The first half of this now exists:

- `prepare_annotation_run(...)` writes `units/`, `prompts/`, `raw/`, `annotations/`, and `run.json`
- `write_raw_response(...)` persists raw model output
- `write_annotation_result(...)` persists normalized annotation JSON

The remaining step is model execution.
