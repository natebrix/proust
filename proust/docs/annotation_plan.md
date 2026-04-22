# ISLT Annotation Plan

This document is the durable strategy document for the annotation project.

It should stay focused on:

- goals
- operating criteria
- review rules
- automation-readiness criteria
- intervention thresholds
- the next-phase checklist

Detailed run-by-run checkpoints and local judgments live in:

- [annotation_log.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_log.md:1)

For the shortest operational handoff, see:

- [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1)

## Goal

Use prompt-based analysis on selected passages from *À la recherche du temps perdu* to produce structured literary-social annotations that can later be transformed into different notions of "winning" and "losing."

The project is trying to build a corpus that is:

- reproducible
- directionally trustworthy
- operationally scalable
- usable for downstream literary-social analysis

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
- operational behavior that makes unattended scaling unreliable

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

Recommended dimensions for v1:

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

The operative rule is:

- carry forward stable recurring aliases confidently
- expect light local alias refresh near new terrain
- do not assume one large global alias map must be perfected before scaling

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

## Current operating rule

For each new batch:

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

## Runtime rule

For long automated runs:

- do not assume silence means failure
- check `run.json` for `automation.in_progress` and `automation.completed_unit_count`
- check whether files are appearing in `raw/` and `annotations/`
- retry only when there is evidence that progress has actually stopped

Operationally, the default path remains:

- `wait --reduce --report`

The pipeline should be treated as healthy when manifest and file progress continue, even if the terminal stays quiet for a long stretch.

## Current phase

The project is no longer trying to prove that small-scale annotation works.

The project is now in:

- pre-full-corpus automation validation

This phase sits after:

- benchmarked reliability
- transfer checks
- modest exploratory scaling
- larger automated scaling in at least two substantial zones

The current question is no longer:

- can the stack scale beyond its first benchmark terrain?

The current question is:

- what additional evidence is still needed before it is responsible to automate the rest of ISLT?

## What has already been proved

The existing accepted corpus already supports these conclusions:

- the annotation stack works beyond the original benchmark set
- report-first review is now the real workflow, not just an aspiration
- contiguous batch automation can run cleanly across substantial spans
- the stack remains usable across at least two materially different accepted zones
- remaining misses are mostly weighting, compression, or local interpretive issues rather than recurring directional inversion
- bounded alias maintenance has been enough so far
- the default orchestration path is workable for long runs

Those points are no longer the main uncertainty.

## What still must be proved before full-corpus automation

Before automating the rest of ISLT, the project should gather evidence in four areas.

### 1. Terrain-transfer proof

Current status:

- effectively complete
- `v5` now counts as strong positive evidence for this proof
- `v7-p2-m-de-charlus-pendant-la-guerre` now also counts as a completed successful contrasting zone
- `v6-p1` now also counts as a completed successful contrasting zone
- the project should now stop adding more chapter zones by inertia and move to the stress pack unless a new concern appears

The stack should be tested on several deliberately contrasting zones rather than only by continuing familiar adjacency.

What this must answer:

- does the stack remain directionally trustworthy when local dynamics change substantially?

The target should be:

- `2` to `4` additional contrasting zones

Those zones should stress differences such as:

- social density
- narrator-to-scene ratio
- family/interior material
- diffuse or unnamed-group appraisal
- political or salon-talk material
- indirect or weakly signaled social movement

Success standard:

- no new recurring failure class that distorts reports
- no repeated lens inversion serious enough to force routine close review
- no alias collapse that materially degrades coverage

### 2. Adverse-case proof

Current status:

- complete
- `run-276` now counts as the first completed adverse-case stress pack
- the pack did not expose a new recurring failure class
- the next active checklist item is now the production-style dry run

The stack should face a small explicit stress test rather than only ordinary scaling.

What this must answer:

- when the passage type is hard, do failures remain local and tolerable?
- or do they become systematically misleading at the report level?

The target should be:

- one curated stress pack of roughly `20` to `40` units

That pack should include examples like:

- sparse reflective passages
- strongly ironic narrator framing
- title-heavy aristocratic passages with collision risk
- crowded scenes with many peripheral named figures
- passages with mixed prestige and inclusion signals
- passages where focal character assignment is easy to blur

Success standard:

- failures may be visible
- failures should stay local
- the stack should not start producing systematically misleading higher-level shapes

### 3. Operational proof at production scale

Current status:

- complete
- the first unattended multi-batch dry run completed across `run-278`, `run-280`, and `run-282`
- the chain did encounter one explicit quota interruption in `run-282`, but the interruption was diagnosable, resumable, and then successfully resumed to completion
- terminal-state handling after API interruption has now been patched in the runner

The project has proven batch-scale automation.

It still needs one explicit proof that unattended multi-batch operation is reliable enough for production use.

What this must answer:

- can the pipeline run at real scale without close supervision and without silently poisoning the corpus?

The target should be:

- one dry run of the intended unattended operating mode across multiple consecutive batches

This should verify:

- resumability
- idempotence
- manifest accuracy
- retry behavior
- partial-failure detection
- practical monitoring signals

Success standard:

- a stalled request or interrupted run is detectable
- reruns do not create ambiguous corpus state
- the operator can tell the difference between slowness and failure
- interrupted runs should not be left looking live when the parent process has already exited

### 4. Corpus-level sanity proof

Current status:

- complete
- the first explicit corpus sanity review has already been run over the accepted corpus
- current judgment:
  - the aggregate surface remains literarily sane
  - cross-lens disagreement remains low
  - no hidden large-scale distortion has appeared
  - focal narrowness remains the main watchpoint rather than corpus corruption

The project should run one lightweight downstream analysis over the accepted corpus accumulated so far.

What this must answer:

- do many acceptable local runs aggregate into a literarily sane corpus surface?

This matters because some failure classes are only visible after aggregation.

The target should be:

- one explicit corpus sanity review over the accepted corpus

Useful checks include:

- most positive and negative characters by lens
- per-character volatility across units
- disagreement rate between the three lenses
- fraction of units that are neutral, mixed, win, or loss
- extreme-score units
- runs or chapters with unusually narrow character surfaces

Success standard:

- outputs need not be perfect
- outputs should still look literarily and operationally sane
- no hidden large-scale distortion should become visible only at aggregation time

## What does not need to be proved

The project does **not** need the following before full-corpus automation:

- benchmark-like confidence in every new chapter
- a universal alias map prepared in advance
- exact stability of all three lenses in every local case
- elimination of all mixed or debatable units
- another long period of routine unit-by-unit review

Requiring those would be overly conservative relative to the actual project standard.

## Full-corpus automation threshold

The project should treat full-corpus automation as justified when all of the following are true:

- `2` to `4` additional contrasting zones complete acceptably
- none of those zones produces a new recurring failure class that distorts reports
- one adverse-case stress pack shows failures remain local rather than structurally misleading
- one unattended multi-batch dry run shows the orchestration is operationally stable
- one corpus-level sanity review does not reveal major hidden distortion

If those conditions hold, the project should stop asking for much more evidence and move to full-corpus automation with monitoring.

## Intervention threshold

Do **not** keep scaling blindly if any of the following appears:

- a recurring failure class starts to distort downstream character arcs
- one lens becomes systematically misleading in a recurrent passage type
- new terrain exposes alias or parsing gaps that materially degrade coverage
- report-level shapes become surprising often enough that selective review is no longer selective
- aggregated corpus summaries expose hidden distortion not visible batch by batch
- unattended orchestration proves too brittle to trust operationally

If that happens, pause scaling and do a targeted intervention:

- prompt revision
- reducer heuristic
- alias-map extension
- narrow benchmark or stress-pack addition
- orchestration or retry-path improvement

## Next-phase checklist

This is the checklist that should govern the next phase.

### A. Contrastive zones

Current status:

- `v5` should now be treated as substantially complete for this proof
- `v7-p2-m-de-charlus-pendant-la-guerre` should now also be treated as complete for this proof
- `v6-p1` should now also be treated as complete for this proof
- the project should stop extending these zones by inertia unless a specific unresolved question arises
- this proof is complete for current project purposes

- choose `2` to `4` chapter zones for contrast rather than adjacency
- prefer zones that differ in social density and rhetorical mode from the accepted corpus
- run them with the normal report-first workflow
- record whether any new recurring failure class appears

Current shortlisted zones for this proof:

- `v5` (`La Prisonnière`)
  - primary reason:
    - intimate, domestic, and jealousy-heavy dynamics
  - main test:
    - whether the stack remains useful when emotional and possessive dynamics dominate over salon-style public ranking
- `v7-p2-m-de-charlus-pendant-la-guerre`
  - primary reason:
    - wartime and politically reconfigured social conditions
  - main test:
    - whether the current lenses remain interpretable outside the familiar Guermantes / salon equilibrium
- `v6-p1` or `v6-p2` (`Albertine disparue`)
  - primary reason:
    - grief, absence, retrospection, and weaker explicit social motion
  - main test:
    - whether sparse reflective terrain remains directionally usable rather than collapsing into emptiness or noise
- one Swann-side bridge zone:
  - `v2-p1-autour-de-mme-swann`
  - or `v1-p2-un-amour-de-swann`
  - primary reason:
    - a different but still legible social cluster that is not just more Guermantes / Charlus adjacency
  - main test:
    - transfer into a socially explicit but differently organized network

Preferred order:

1. optional Swann-side bridge zone only if extra transfer breadth is later desired

Reason for this order:

- `v5` has already supplied the intimate and psychological narrowing test
- `v7-p2-m-de-charlus-pendant-la-guerre` has already supplied the wartime and historical-social reconfiguration test
- `v6-p1` has now supplied the sparse reflective aftermath test
- the Swann-side zone remains available only if the project later decides it wants optional extra transfer breadth

### B. Stress pack

Current status:

- complete
- `run-276` completed the first stress pack cleanly
- no parse or validation failures appeared
- the adverse-case categories remained directionally coherent at the report level
- failures remained local rather than structurally misleading

- curate a `20` to `40` unit adverse-case pack
- include irony, sparsity, title collision risk, crowding, and mixed-signal cases
- automate, reduce, and report it
- judge whether errors remain local or become structurally misleading

Recommended first stress pack:

- use one curated pack of `20` units rather than many tiny sub-packs
- mix old known failure cases with newer transfer-proven terrain so the result tests both historical weaknesses and present behavior
- prefer units already mentioned in project docs as conceptually useful or operationally risky

Proposed `20`-unit shortlist:

1. irony / narrator-stance drift
   - `v1-p1-combray#p-20`
   - `v1-p1-combray#p-274-p-275`
   - `v1-p1-combray#p-278-p-279`
   - `v3-p1#p-186-p-195`
2. dominant-movement discipline and mixed prestige/inclusion
   - `v1-p1-combray#p-21-p-22`
   - `v1-p1-combray#p-25-p-26`
   - `v1-p1-combray#p-312-p-313`
   - `v2-p2-noms-de-pays-le-pays#p-61-p-65`
3. title-heavy or collision-risk social material
   - `v2-p2-noms-de-pays-le-pays#p-111-p-115`
   - `v4-p2#p-1-p-5`
   - `v4-p2#p-91-p-95`
   - `v2-p2-noms-de-pays-le-pays#p-166-p-170`
4. sparse reflective and grief-driven terrain
   - `v6-p1#p-1-p-5`
   - `v6-p1#p-96-p-100`
   - `v6-p1#p-101-p-105`
   - `v6-p1#p-106-p-110`
5. focal blur, collateral figures, and mixed-signal local fields
   - `v6-p1#p-116-p-120`
   - `v7-p2-m-de-charlus-pendant-la-guerre#p-46-p-50`
   - `v7-p2-m-de-charlus-pendant-la-guerre#p-51-p-55`
   - `v2-p1-autour-de-mme-swann#p-311-p-315`

Why this shortlist:

- it directly targets the known historical failure classes from `run_003_failure_modes.md`
- it includes the earlier `princesse` collision terrain without reintroducing the bad alias practice
- it includes several units where prestige and inclusion are known to diverge
- it includes sparse reflective aftermath units where emptiness, over-reading, or false directional certainty would be obvious
- it includes units where secondary figures are easy to over-surface or wrongly center

Execution preference:

- prepare the full pack as one run if convenient
- if operationally easier, split it into two labeled `10`-unit halves while preserving the same category balance
- in either case, judge the pack at the report level first, then inspect unit failures only where the report shape suggests a real structural problem

Completed result:

- `run-276` completed the full `20`-unit pack
- `20/20` units completed
- `0` parse errors
- `0` validation errors
- the pack remained substantively coherent across:
  - irony / narrator-stance drift
  - mixed prestige / inclusion material
  - title-heavy and collision-risk passages
  - sparse reflective and grief-driven terrain
  - focal blur and collateral-figure pressure
- current judgment:
  - the adverse-case proof is now satisfied for project purposes

### C. Production-style dry run

Current status:

- complete
- first explicit dry run launched as three unattended consecutive batches in `v2-p1-autour-de-mme-swann`
- source sequence:
  - `run-277`
  - `run-279`
  - `run-281`
- automated sequence:
  - `run-278`
  - `run-280`
  - `run-282`
- outcome:
  - `run-278` completed cleanly
  - `run-280` completed cleanly
  - `run-282` was interrupted once by `429 insufficient_quota` after `1/6` units
  - the run was then resumed successfully once credits were replenished
  - resumed execution correctly skipped the already-completed unit and finished the remaining `5/5`
  - `run-282` completed cleanly after resumption
  - the runner now writes terminal failure state on request exceptions instead of leaving interrupted runs looking live

- define the intended unattended multi-batch operating procedure
- run multiple consecutive batches with minimal intervention
- verify resumability, retry behavior, and manifest correctness
- document the operational rule for when to rerun versus wait

Current judgment:

- unattended chaining itself looks viable
- slowness versus failure was distinguishable in practice
- resumability is now explicitly demonstrated because completed annotations were preserved and skipped on rerun
- manifest interruption-state handling has been patched
- this should count as completed operational proof for current project purposes

### D. Corpus sanity review

Current status:

- complete
- no major hidden distortion appeared in the first corpus-level sanity pass

- aggregate the accepted corpus accumulated so far
- compute a small set of sanity summaries across characters, runs, and lenses
- inspect extreme and outlier cases
- decide whether the corpus still looks literarily sane at scale

### E. Final go/no-go decision

- compare the evidence from A through D against the full-corpus automation threshold
- if the threshold is met, move to full-corpus automation with monitoring
- if the threshold is not met, identify the smallest targeted intervention rather than reopening the whole stack

## Default next move

The default next move is:

1. keep `annotation_plan.md` as the strategic checklist document
2. treat `current_state.md` as the shortest operational handoff
3. treat `annotation_log.md` as the historical record of accepted runs and judgments
4. compare the accumulated evidence against the full-corpus automation threshold
5. move to full-corpus automation with monitoring unless a new contrary signal appears
