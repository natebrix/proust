# Full-Corpus Runbook

This document defines the operating procedure for the first full-corpus automation pass over the canonical ISLT chapter set.

It is the practical companion to:

- [annotation_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_plan.md:1)
- [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1)
- [annotation_runner.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_runner.md:1)

## Goal

Run one end-to-end automated annotation pass over the canonical ISLT chapters using the current prompt, reducer, and scoring stack without reopening routine validation questions chapter by chapter.

## Corpus order

Use the canonical chapter order already exported in the repo:

1. `v1-p1-combray`
2. `v1-p2-un-amour-de-swann`
3. `v1-p3-noms-de-pays-le-nom`
4. `v2-p1-autour-de-mme-swann`
5. `v2-p2-noms-de-pays-le-pays`
6. `v3-p1`
7. `v3-p2`
8. `v4-p1`
9. `v4-p2`
10. `v5`
11. `v6-p1`
12. `v6-p2`
13. `v6-p3`
14. `v6-p4`
15. `v7-p1-a-tansonville`
16. `v7-p2-m-de-charlus-pendant-la-guerre`
17. `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`
18. `v7-p4-le-bal-de-tetes`

## Unit shape

Default full-corpus unit shape:

- `5` paragraphs per unit
- `8` units per source batch
- therefore a standard source batch covers `40` paragraphs

At chapter boundaries:

- use smaller trailing units only when the remaining span does not divide evenly into `5`
- do not cross chapter boundaries inside a unit
- do not mix chapters inside a source batch unless a chapter tail would otherwise leave a tiny stranded remainder

## Alias policy

Use the current stable carried-forward alias map as the default starting map for the full-corpus pass.

Operational rule:

- carry the latest stable alias map forward from the previous accepted source run
- extend it only when new terrain exposes a real coverage gap
- avoid broad speculative alias growth
- treat title collisions conservatively

## Run naming and pairing

For the full-corpus pass, continue using paired source/output runs:

- one prepared source run
- one automated output run derived from it

The operative pairing is:

- source run defines `units/`, `prompts/`, `alias_map`, and notes
- output run contains `raw/`, `annotations/`, automation metadata, and reports

## Standard batch procedure

For each batch:

1. prepare the next contiguous source batch from the current cursor
2. automate it to a fresh output run
3. wait for completion
4. reduce and report it
5. read the three report lenses first
6. advance the cursor unless a genuine stop condition appears

Default commands:

```bash
python -m proust batch --source-run outputs/run-SOURCE --output outputs/run-OUTPUT --model gpt-5
```

Current source-batch preparation note:

- `python -m proust prepare` only creates an empty run scaffold
- explicit source batches are currently prepared via `prepare_annotation_run(...)` with `AnnotationUnitSpec` spans and a carried-forward alias map
- until a dedicated batch-prep CLI exists, treat the Python preparation path as the operational standard

Equivalent manual split form when needed:

```bash
python -m proust automate --source-run outputs/run-SOURCE --output outputs/run-OUTPUT --model gpt-5
python -m proust wait --run outputs/run-OUTPUT --reduce --report
```

Operational preference:

- use `batch` as the default command for new production runs
- fall back to the split `automate` plus `wait --reduce --report` sequence when resuming a partial run or diagnosing an issue
- preserve the last accepted source run as the alias-map baseline for the next prepared batch

## Monitoring rule

For a live automated run, check:

- `automation.in_progress`
- `automation.completed_unit_count`
- `automation.requested_unit_count`
- files appearing in `raw/`
- files appearing in `annotations/`

If progress is visible, let the run continue.

If a request fails explicitly:

- treat the manifest as the source of truth
- preserve completed work
- rerun `automate` on the same source/output pair after the external issue is resolved
- do not use `--overwrite` unless you explicitly want to replace completed outputs

## Review rule during full-corpus automation

Do not reopen routine unit-level review by default.

Inspect units only when reports show a genuinely surprising signal, such as:

- report-level inversion relative to nearby context
- sharp unexplained divergence between lenses
- repeated wrong-direction outputs in the same passage type
- new alias or parsing failure affecting multiple units
- operational ambiguity that prevents distinguishing slowness from failure

Do not pause for:

- familiar tolerated edge cases
- mild weighting disagreements
- isolated mixed or debatable units
- failure to match a past benchmark exactly

## Stop conditions

Pause the full-corpus pass only if one of the following appears:

- a recurring failure class begins to distort reports
- one lens becomes systematically misleading in a recurrent passage type
- alias coverage drops materially in a new terrain zone
- automation behavior becomes too brittle to trust operationally

If none of those appears, keep advancing.

## Resumption rule

If the pass is interrupted:

- identify the last completed automated output run
- check whether the next output run is partial
- if partial, rerun `automate` against the same source/output pair
- once complete, run `wait --reduce --report`
- then continue with the next source batch from the next untouched cursor

## Initial cursor

The first full-corpus pass starts at:

- `v1-p1-combray#p-1`

The first standard batch therefore covers:

- `v1-p1-combray#p-1-p-5`
- `v1-p1-combray#p-6-p-10`
- `v1-p1-combray#p-11-p-15`
- `v1-p1-combray#p-16-p-20`
- `v1-p1-combray#p-21-p-25`
- `v1-p1-combray#p-26-p-30`

## Current operating decision

Under the current project standard, full-corpus automation is justified.

The default operating stance is:

- keep the prompt, reducer, and lenses fixed
- annotate the corpus contiguously in standard batches
- monitor operational health
- intervene only for real report-level or orchestration-level problems

Current production cadence:

- `8` units per batch is now the default operating cadence
- keep `5` paragraphs per unit
- preserve the chapter-boundary rule for short trailing units
