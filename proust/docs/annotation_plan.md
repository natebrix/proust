# ISLT Annotation Plan

This document is the durable strategy document for the annotation project.

It should stay focused on:

- project goal
- current standard
- durable operating rules
- intervention thresholds
- the current strategic phase

Detailed run-by-run checkpoints and local judgments live in:

- [annotation_log.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_log.md:1)

For the shortest operational handoff, see:

- [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1)

For current artifact discovery, see:

- [outputs_guide.md](/Users/nathan_brixius/dev/proust/proust/docs/outputs_guide.md:1)

## Goal

Use prompt-based analysis on passages from *À la recherche du temps perdu* to produce structured literary-social annotations that can be transformed into different notions of "winning" and "losing."

The project is trying to maintain a corpus that is:

- reproducible
- directionally trustworthy
- operationally stable
- usable for downstream literary-social analysis
- usable for app-facing derivative products

The project is not trying to produce:

- benchmark-perfect agreement on every unit
- indefinite close review of the whole novel
- a permanently hand-curated annotation workflow

## Current standard

The current standard is not benchmark equality.

The current standard is:

- directional trustworthiness at the report level

That means the project is allowed to tolerate:

- familiar edge cases
- mild weighting disagreements
- local mixed or debatable units
- imperfect compression of complex passages

The project should not tolerate:

- recurring report-level inversion
- a lens becoming systematically misleading in a recurrent passage type
- alias or parsing failure that materially degrades coverage
- operational behavior that makes the accepted corpus unreliable

## Current phase

The project is no longer in pre-full-corpus validation.

The current phase is:

- post-production normalized aggregate analysis

That means:

- the canonical full-corpus annotation pass is complete
- the accepted annotation JSON should remain fixed by default
- the normalized aggregate review surface is now the default character surface
- new work should primarily happen in downstream analysis and app-facing exports

The main questions are now:

- what downstream analysis surfaces best expose the literary-social structure of the corpus?
- what app-facing derivative data products should be built on top of those surfaces?

The main questions are no longer:

- whether full-corpus automation is justified
- whether more transfer or stress evidence is required before production use
- whether the project should reopen routine prompt or reducer tuning

## Current stack assumptions

Assume the following unless new evidence forces a change:

- the current prompt is strong enough to keep using
- the current reducer is strong enough to keep using
- the current scoring lenses are strong enough for exploratory analysis
- prompt, reducer, schema, and alias changes should be targeted interventions, not routine companions to each run

## Minimal schema

The reduced annotation schema should continue to center four sections:

### 1. `characters_present`

Purpose:

- record which canonical characters are explicitly present or clearly implicated

### 2. `appraisal_events`

Purpose:

- capture meaningful local evaluative or status-relevant moves in the passage

### 3. `status_effects`

Purpose:

- translate events into local position changes for each affected character

Recommended dimensions:

- `social_status`
- `rhetorical_position`
- `emotional_position`
- `inclusion_exclusion`
- `general_appraisal`

### 4. `ambiguities`

Purpose:

- preserve uncertainty without collapsing it into false precision

## Alias strategy

The alias map remains useful and should be preserved.

For current work:

- use canonical human-readable character names
- resolve only names supported by the alias map
- treat ambiguous surface forms conservatively
- prefer bounded alias maintenance over broad alias proliferation
- avoid generic title aliases when they risk colliding with distinct titled figures in the same run

At the aggregate layer:

- use the reviewed explicit normalization map where needed
- prefer explicit reviewed merges over broad heuristic identity collapsing
- keep source annotations unchanged unless a later high-value reason clearly justifies rewriting them

## Scoring lenses

Keep the current three-lens comparison:

- `local_outcome_v1`
- `prestige_outcome_v1`
- `inclusion_outcome_v1`

These lenses are meant to be compared, not collapsed into one final notion of value.

Their purpose is to help distinguish:

- prestige from belonging
- rhetorical advantage from social incorporation
- mixed outcomes from clean local wins or losses

## Operating rule

For any new annotation batch or maintenance run:

1. prepare the source run
2. automate to a fresh output run
3. reduce and score it
4. read the three report lenses first
5. inspect units only if the reports show a genuinely surprising signal

Operational clarification:

- the current source-run preparation path is `prepare_annotation_run(...)` with explicit unit specs and a carried-forward alias map
- the CLI `prepare` command only scaffolds an empty run and should not be treated as sufficient for production batch setup

Examples of genuinely surprising signals:

- a character arc appears inverted relative to nearby context
- one lens diverges sharply from the others without a clear textual reason
- the same passage type starts producing the same wrong directional result repeatedly
- a new alias, parsing, or reduction problem affects multiple units at once

Examples that do **not** justify close review by default:

- a familiar tolerated edge case
- mild weighting disagreements
- one or two debatable mixed-unit outcomes
- failure to match an older benchmark exactly

## Stop rule

Do not keep reopening interpretive review once the report-level evidence remains stable.

The default should be:

- keep the prompt, reducer, and lenses fixed
- keep accepted annotation JSON fixed
- treat the normalized aggregate surface as the default analysis surface
- intervene only if downstream analysis exposes a genuinely new report-level failure class

For downstream analysis work:

- read aggregate artifacts first
- only drill back into unit-level evidence when the aggregate picture shows a genuinely surprising signal
- prefer new downstream views over source-annotation rewriting

## Runtime rule

For long automated runs:

- do not assume silence means failure
- check `run.json` for `automation.in_progress` and `automation.completed_unit_count`
- check whether files are appearing in `raw/` and `annotations/`
- retry only when there is evidence that progress has actually stopped

Operationally, the default path remains:

- `wait --reduce --report`

The pipeline should be treated as healthy when manifest and file progress continue, even if the terminal stays quiet for a long stretch.

## Intervention threshold

Do **not** treat the current stack as permanently fixed if any of the following appears:

- a recurring failure class starts to distort downstream character arcs
- one lens becomes systematically misleading in a recurrent passage type
- new terrain exposes alias or parsing gaps that materially degrade coverage
- aggregated corpus summaries expose hidden distortion not visible batch by batch
- app-facing exports require structural data not recoverable from the current annotation surface

If that happens, do a targeted intervention:

- prompt revision
- reducer heuristic
- alias-map extension
- narrow benchmark or stress-pack addition
- aggregation/export-layer improvement

Do not reopen the whole stack by default.

## Current strategic surface

The project should now treat these as the default current surfaces:

- normalized corpus review
- character cross-lens analysis
- character-by-chapter cross-lens analysis
- character profile cards
- `chapter_overlay_v2`

These are the current bridges between:

- accepted source annotations
- aggregate literary-social analysis
- app-facing presentation

## Default next move

If work resumes from this plan, the default next move is:

1. keep the accepted annotation JSON unchanged
2. treat the normalized corpus-review surface as canonical
3. extend downstream analysis before reopening annotation production
4. treat app-facing exports as derivative layers over the accepted normalized corpus
5. only reopen source annotation or stack behavior if a later report-level problem clearly justifies it
