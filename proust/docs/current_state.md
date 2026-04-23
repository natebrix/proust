# Current State

This file is the shortest current checkpoint for the annotation project.

It is meant to answer two questions quickly:

1. where the project stands now
2. what the next session should do by default

For the longer running history, decisions, and examples, see:

- [annotation_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_plan.md:1)
- [annotation_log.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_log.md:1)
- [annotation_runner.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_runner.md:1)
- [full_corpus_runbook.md](/Users/nathan_brixius/dev/proust/proust/docs/full_corpus_runbook.md:1)

## Current phase

The project is in:

- post-production aggregate analysis and normalization planning

This now means:

- benchmark-building is complete
- terrain-transfer proof is effectively complete
- the first corpus sanity review is complete
- the first adverse-case stress pack is complete
- the first production-style dry run is now complete
- full-corpus automation has now completed the canonical ISLT chapter pass
- the final production-corpus sanity/aggregation review is complete
- annotation `explanation` fields have been normalized to English
- a refreshed full-corpus review has been generated over all currently accepted annotation outputs
- a character-alias audit has been completed
- the first character-alias normalization plan has been reviewed and documented

The current question is no longer:

- can the prompt and reducer survive new literary terrain?
- should the project proceed to full-corpus automation?
- what final corpus-level sanity checks should be run on the completed production corpus?

The current question is:

- what aggregate-layer normalization and downstream analysis surfaces should be built from the accepted production corpus?

## Current stack status

Assume the following unless a new batch shows otherwise:

- the current prompt is good enough to keep using
- the current reducer is good enough to keep using
- the current scoring lenses are good enough for exploratory analysis
- prompt, reducer, schema, and alias changes should be targeted interventions, not routine companions to each run

The current project standard is not benchmark equality.

The current standard is:

- directional trustworthiness at the report level

Current read:

- no recurring report-level inversion has appeared
- cross-lens disagreement remains low at the corpus level
- the completed production corpus has `0` cross-lens positive/negative sign-flip examples
- the stress pack did not expose a hidden structural failure class
- the main watchpoint remains focal narrowness, not corpus corruption
- the dry run demonstrated viable unattended chaining, explicit failure visibility, and practical resumability
- interruption-state handling has now been patched in the runner
- controlled chapter-internal parallelism has now been repeatedly validated in production chapters
- the current aggregate review surface is good enough to expose identity-splitting issues that were not obvious batch by batch

## Current review rule

For each new automated batch sequence:

1. prepare the source run
2. automate to a fresh output run
3. reduce and score it
4. read the three report lenses first
5. inspect units only if the reports show a genuinely surprising signal

Operational clarification:

- source-run preparation currently means building a run with `prepare_annotation_run(...)` and explicit `AnnotationUnitSpec` spans
- `python -m proust prepare` is only a scaffold command and is not sufficient for full-corpus batch preparation

Examples of genuinely surprising signals:

- a character arc looks inverted relative to nearby material
- one lens diverges sharply from the others without a clear textual reason
- the same passage type starts producing the same wrong directional result repeatedly
- a new alias, parsing, or reduction problem affects multiple units at once
- a multi-batch run becomes operationally ambiguous enough that the operator cannot distinguish slowness from failure

Examples that do not justify close review by default:

- a familiar tolerated edge case
- mild weighting disagreements
- one or two debatable mixed-unit outcomes
- failure to match an older benchmark exactly

## Current stop rule

Do not keep reopening interpretive review once the report-level evidence remains stable.

The default should be:

- keep the prompt, reducer, and lenses fixed
- move to full-corpus automation with monitoring
- intervene only if a new recurring failure class or orchestration failure appears

For long automated runs:

- do not assume silence means failure
- check `run.json` for `automation.in_progress` and `automation.completed_unit_count`
- check whether files are appearing in `raw/` and `annotations/`
- retry only when there is evidence that progress has actually stopped

## Latest accepted evidence

The current accepted evidence now includes:

- sustained chapter-scale and widened-aperture scaling in earlier accepted zones
- a completed corpus sanity review over the accepted corpus
- completed successful terrain-transfer zones in:
  - `v5`
  - `v7-p2-m-de-charlus-pendant-la-guerre`
  - `v6-p1`
- a completed `20`-unit adverse-case stress pack in `run-276`
- a completed production-style dry run through `run-278`, `run-280`, and the resumed `run-282`
- a refreshed current corpus review over all accepted annotation outputs:
  - [corpus-review-current.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current.json:1)
  - [corpus-review-current.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current.md:1)
- a completed character alias audit:
  - [character-alias-audit-current.json](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.json:1)
  - [character-alias-audit-current.md](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.md:1)
- a reviewed character alias normalization plan:
  - [character_alias_normalization_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/character_alias_normalization_plan.md:1)

Stress-pack result:

- `20/20` units completed
- `0` parse errors
- `0` validation errors
- hard cases remained directionally coherent rather than merely schema-valid
- failures, where present, remained local and tolerable rather than structurally misleading

Key interpretive takeaways from `run-276`:

- `v6-p1` remained nuanced rather than collapsing into a rigid `Albertine` rule
- wartime `Charlus` / `Morel` terrain stayed legible without focal hijack
- title-heavy and prestige-friction passages remained plausible under the existing alias discipline
- the pack did not reveal a new adverse-case class that would justify retuning the stack before larger automation

## Current judgment

The project has completed the full-corpus automation and first aggregate refresh successfully.

Current conclusion:

- keep source annotations fixed
- treat the accepted annotation corpus as stable enough for aggregate-layer refinement
- use explicit reviewed mappings rather than broad alias heuristics
- make the next changes in downstream aggregation, not in prompt/reducer/schema behavior

## Default next move

If work resumes from this checkpoint, the next default move is:

1. keep the accepted annotation JSON unchanged
2. implement optional aggregate-layer character normalization using only the reviewed explicit mapping in [character_alias_normalization_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/character_alias_normalization_plan.md:1)
3. generate normalized aggregate artifacts beside the current unnormalized ones
4. write a normalization diff that shows what changed in top character totals, rankings, and cross-lens summaries
5. only consider source-annotation rewriting after the normalized aggregate surface has been reviewed and accepted

## Latest checkpoint

The current production pass has completed `v3-p2`, completed the short `v4-p1` chapter, completed `v4-p2`, completed `v5`, completed `v6-p1`, completed `v6-p2`, completed `v6-p3`, completed the short `v6-p4` chapter, completed the short `v7-p1-a-tansonville` chapter, completed `v7-p2-m-de-charlus-pendant-la-guerre`, completed `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`, and completed `v7-p4-le-bal-de-tetes`.

Completed through the latest accepted outputs:

- `run-466`: output for `v3-p2#p-601-p-640`
- `run-468`: output for `v3-p2#p-641-p-680`
- `run-470`: output for `v3-p2#p-681-p-720`
- `run-472`: output for `v3-p2#p-721-p-730`
- `run-476`: output for `v3-p2#p-731-p-733`
- `run-474`: output for `v4-p1#p-1-p-22`
- `run-478`: output for `v4-p2#p-1-p-40`
- `run-480`: output for `v4-p2#p-41-p-80`
- `run-482`: output for `v4-p2#p-81-p-120`
- `run-484`: output for `v4-p2#p-121-p-160`
- `run-486`: output for `v4-p2#p-161-p-200`
- `run-488`: output for `v4-p2#p-201-p-240`
- `run-490`: output for `v4-p2#p-241-p-280`
- `run-492`: output for `v4-p2#p-281-p-320`
- `run-494`: output for `v4-p2#p-321-p-360`
- `run-496`: output for `v4-p2#p-361-p-400`
- `run-498`: output for `v4-p2#p-401-p-440`
- `run-500`: output for `v4-p2#p-441-p-450`
- `run-502`: output for `v5#p-1-p-40`
- `run-504`: output for `v5#p-41-p-80`
- `run-506`: output for `v5#p-81-p-120`
- `run-508`: output for `v5#p-121-p-160`
- `run-510`: output for `v5#p-161-p-200`
- `run-512`: output for `v5#p-201-p-240`
- `run-514`: output for `v5#p-241-p-280`
- `run-516`: output for `v5#p-281-p-320`
- `run-518`: output for `v5#p-321-p-360`
- `run-520`: output for `v5#p-361-p-400`
- `run-522`: output for `v5#p-401-p-428`
- `run-524`: output for `v6-p1#p-1-p-40`
- `run-526`: output for `v6-p1#p-41-p-80`
- `run-528`: output for `v6-p1#p-81-p-120`
- `run-530`: output for `v6-p2#p-1-p-40`
- `run-532`: output for `v6-p2#p-41-p-72`
- `run-534`: output for `v6-p3#p-1-p-40`
- `run-536`: output for `v6-p3#p-41-p-69`
- `run-538`: output for `v6-p4#p-1-p-25`
- `run-540`: output for `v7-p1-a-tansonville#p-1-p-25`
- `run-542`: output for `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-40`
- `run-544`: output for `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-80`
- `run-546`: output for `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-40`
- `run-548`: output for `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-41-p-45`
- `run-550`: output for `v7-p4-le-bal-de-tetes#p-1-p-40`
- `run-552`: output for `v7-p4-le-bal-de-tetes#p-41-p-80`
- `run-554`: output for `v7-p4-le-bal-de-tetes#p-81-p-120`
- `run-556`: output for `v7-p4-le-bal-de-tetes#p-121-p-141`

Latest mechanical result:

- explanation-language normalization committed in `cfe7d33`
- current corpus review workflow and refreshed artifacts committed in `499ad3a`
- character alias audit committed in `522d51e`
- reviewed character alias normalization plan committed in `82137dc`

Current aggregate corpus counts:

- `271` annotated runs discovered under `outputs/`
- `1684` declared units
- `1684` valid annotations
- cross-lens sign-flip examples in the refreshed corpus review: `0`

Current stopping point:

- the corpus review surface has exposed a small number of identity splits (`Charlus` / `baron de Charlus`, `Mme Swann` / `Odette`, etc.)
- those have now been audited and converted into an explicit reviewed normalization plan
- the next session should begin with aggregate-layer normalization, not new annotation production

- `run-466`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-468`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-470`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-472`: `2/2` completed, `0` parse errors, `0` validation errors
- `run-476`: `1/1` completed, `0` parse errors, `0` validation errors
- `run-474`: `5/5` completed, `0` parse errors, `0` validation errors
- `run-478`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-480`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-482`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-484`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-486`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-488`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-490`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-492`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-494`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-496`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-498`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-500`: `2/2` completed, `0` parse errors, `0` validation errors
- `run-502`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-504`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-506`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-508`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-510`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-512`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-514`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-516`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-518`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-520`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-522`: `6/6` completed, `0` parse errors, `0` validation errors
- `run-524`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-526`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-528`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-530`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-532`: `7/7` completed, `0` parse errors, `0` validation errors
- `run-534`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-536`: `6/6` completed, `0` parse errors, `0` validation errors
- `run-538`: `5/5` completed, `0` parse errors, `0` validation errors
- `run-540`: `5/5` completed, `0` parse errors, `0` validation errors
- `run-542`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-544`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-546`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-548`: `1/1` completed, `0` parse errors, `0` validation errors
- `run-550`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-552`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-554`: `8/8` completed, `0` parse errors, `0` validation errors
- `run-556`: `5/5` completed, `0` parse errors, `0` validation errors
- all passed the review gate

Latest review surface:

- `run-466` was clean:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-468` was clean:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 2 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-470` was clean enough for the higher-variance closing Guermantes material:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 2 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-472` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-476` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-474` was acceptable as a short `v4-p1` sequential opener/full-chapter pass:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 3 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-478` was a clean `v4-p2` sequential opener:
  - mixed counts `{ inclusion: 2, local: 1, prestige: 2 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-480` was clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-482` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-484` was acceptable:
  - mixed counts `{ inclusion: 2, local: 1, prestige: 1 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-486` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-488` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-490` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-492` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-494` was acceptable:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-496` was acceptable:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 1 }`
  - `2` label disagreements
  - `2` direction disagreements
  - `0` sign flips
- `run-498` was acceptable at the mixed-unit threshold:
  - mixed counts `{ inclusion: 3, local: 1, prestige: 3 }`
  - `2` label disagreements
  - `1` direction disagreement
  - `0` sign flips
- `run-500` was clean as the short `v4-p2` tail:
  - mixed counts `{ inclusion: 0, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-502` was a clean `v5` opener:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-504` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-506` was clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-508` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-510` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-512` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-514` was clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-516` was clean:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-518` was acceptable:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 1 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-520` was clean:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-522` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-524` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-526` was clean:
  - mixed counts `{ inclusion: 0, local: 1, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-528` was acceptable:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 2 }`
  - `2` label disagreements
  - `1` direction disagreement
  - `0` sign flips
- `run-530` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-532` was acceptable at the mixed-unit threshold:
  - mixed counts `{ inclusion: 3, local: 2, prestige: 3 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-534` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-536` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-538` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-540` was acceptable:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-542` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-544` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-546` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-548` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-550` was clean:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-552` was acceptable:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-554` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-556` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Current operational judgment:

- `v3-p2` is complete and closed without a review-gate stop condition
- `v4-p1` is complete and acceptable; its mixed pressure reached but did not exceed the review threshold
- `v4-p2` opened sequentially without a review-gate stop condition, then resumed controlled chapter-internal parallel mode cleanly
- `v4-p2` is complete through `p-450`
- `run-498` reaches the mixed-unit threshold but does not exceed it, and no sign flip appears
- `run-500` closes the chapter without a review-gate stop condition
- `v5` is complete through `p-428`
- `v6-p1` is complete through `p-120`
- `v6-p2` is complete through `p-72`
- `run-532` reaches the mixed-unit threshold in inclusion and prestige but does not exceed it, and no sign flip appears
- `v6-p3` is complete through `p-69`
- `v6-p4` is complete through `p-25`
- `v7-p1-a-tansonville` is complete through `p-25`
- `v7-p2-m-de-charlus-pendant-la-guerre` is complete through `p-80`
- `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle` is complete through `p-45`
- `v7-p4-le-bal-de-tetes` is complete through `p-141`
- the canonical ISLT production pass is complete through the final exported chapter
- after reboot, do not rely on old terminal sessions; use the filesystem manifests as the source of truth
