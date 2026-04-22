# ISLT Annotation Log

This document preserves the detailed run history, checkpoint judgments, and local evidence that support the current annotation strategy.

It is the companion to:

- [annotation_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_plan.md:1), which now holds the durable goals, criteria, and phase logic
- [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1), which holds the shortest operational handoff

What belongs here:

- run-by-run checkpoints
- representative misses
- report readings
- runtime and alias notes
- local assessment decisions

What does not need to stay here:

- the stable phase model
- the durable transition criteria
- the standing default workflow

## Goal

Use prompt-based analysis on selected passages from *À la recherche du temps perdu* to produce structured literary-social annotations that can later be transformed into different notions of "winning" and "losing."

The first phase should stay narrow:

- work on a small subset of ISLT
- preserve the existing alias normalization approach
- avoid broad model or schema complexity before one end-to-end run works

## Immediate priorities

1. Make annotation runs explicit and reproducible.

- Avoid relying on process-global NLP or alias state for the new annotation workflow.
- Run annotation jobs from an explicit session/config object.
- Keep passage selection, alias map, prompt version, and output files tied together.

2. Define a stable text unit for prompting.

- Use deterministic passage windows derived from the source text.
- Each unit should carry enough provenance to be rerun or audited later.
- Minimum metadata should include source chapter/page id, paragraph range, raw text, and preprocessed text.

3. Start with a minimal structured annotation schema.

- Capture only the literary-social primitives needed for later transformation.
- Prefer a small number of reliable fields over a large ontology.
- Preserve ambiguity instead of forcing a single winner/loser conclusion.

4. Build a small batch runner.

- Select a subset of passages.
- Render prompt inputs consistently.
- Save raw model output.
- Validate and normalize structured JSON output.
- Preserve provenance for each passage.

5. Do only the cleanup that supports the annotation workflow.

- Add packaging and dependency metadata when the runner shape is clear.
- Defer broader refactors in legacy analytics code unless they block annotation work.

## Alias strategy

The current alias map is useful and should be kept for the first pass.

For now:

- use canonical human-readable character names
- resolve only names supported by the alias map
- treat ambiguous surface forms conservatively

Later:

- extend alias entries with notes and optional stable ids
- support more careful handling for family roles, narrator-centered references, and ambiguous social titles

## Minimal first-pass schema

The prompt currently asks for a fairly rich output. For the first subset, use a reduced schema with four core sections.

### 1. `characters_present`

Purpose:
- record which canonical characters are explicitly present or clearly implicated

Fields:
- `canonical_name`
- `surface_forms`
- `presence_type`: `explicit | implicit`
- `presence_confidence`

### 2. `appraisal_events`

Purpose:
- capture the meaningful local evaluative or status-relevant moves in the passage

Fields:
- `event_id`
- `source`
- `target`
- `type`
- `polarity`
- `narrative_stance`
- `confidence`
- `evidence`
- `explanation`

Notes:
- keep event types limited to the prompt's core categories
- prefer fewer high-value events
- do not require intensity, directness, or evidence mode in the first pass unless they prove necessary during evaluation

### 3. `status_effects`

Purpose:
- translate events into local position changes for each affected character

Fields:
- `character`
- `dimension`
- `delta`
- `based_on_events`
- `confidence`
- `explanation`

Recommended dimensions for v1:
- `social_status`
- `rhetorical_position`
- `emotional_position`
- `inclusion_exclusion`
- `general_appraisal`

Notes:
- keep `delta` on the existing `-2` to `+2` scale
- allow mixed outcomes across dimensions

### 4. `ambiguities`

Purpose:
- preserve uncertainty without collapsing it into false precision

Fields:
- list of short strings

Use for:
- ironic narration
- uncertain evaluator
- unclear alias resolution
- unstable target assignment

## Why this schema first

This keeps the first pipeline legible and testable.

It is enough to support later transformations into:

- local advantage/disadvantage scores
- rivalry or prestige graphs
- social inclusion/exclusion measures
- alternative "winning" definitions built from the same annotations

It also avoids premature complexity in areas where prompt behavior may still change.

## Practical next steps

1. Review and tighten `proust/prompts/prompt.md` so it matches the reduced first-pass schema.
2. Choose a small subset of passages from ISLT for manual inspection and prompt iteration.
3. Define the on-disk format for:

- passage inputs
- annotation outputs
- run metadata

4. Implement a minimal runner that:

- loads passages
- injects alias map and optional prior context
- writes prompt-ready payloads
- stores raw and normalized responses

5. Evaluate the first batch before extending aliases or scoring logic.

## Current v1 state

The first phase is now far enough along to treat the current setup as a working v1 rather than an open-ended prompt experiment.

Current practical baselines:

- `outputs/run-002` is the reviewed benchmark set
- `outputs/run-006` is the current automated baseline after reduction
- the prompt and reducer are good enough for modest-scale exploratory use even though some semantic misses remain

The remaining misses are now concentrated in a small number of hard interpretive cases rather than broad structural instability.

## Next phase

The next phase should shift from prompt-fidelity work to small-scale deployment and downstream usefulness.

### 1. Freeze the current annotation stack as v1

- keep the current prompt
- keep the current reducer
- preserve `run-002` as the reviewed benchmark
- preserve reduced `run-006` as the current automated baseline

### 2. Scale to a modest larger subset

Prepare a new run on a somewhat larger Combray subset, for example:

- `20` to `40` canonical units
- still concentrated in socially dense passages
- still small enough for targeted review if a failure class becomes important

The goal is not benchmark-perfect matching.

The goal is to see whether the current pipeline is stable enough to produce analyzable structure at modest scale.

### 3. Evaluate downstream usefulness, not just annotation fidelity

Begin using the annotations as inputs to lightweight exploratory summaries such as:

- positive vs negative appraisal counts by character
- net local movement by status dimension
- narrator-led versus socially voiced valuation
- inclusion/exclusion patterns

This is the point where the project starts to test candidate notions of "winning" and "losing."

### 4. Review errors by class, not one passage at a time

For the larger run, ask:

- which failure classes remain common enough to matter downstream?
- which misses are acceptable for exploratory analysis?
- which misses actually distort the early scoring ideas?

This should guide whether the next investment goes into:

- better reduction rules
- better alias coverage
- narrower passage selection
- or first-pass scoring experiments

## Recommended immediate next action

If work resumes from this checkpoint, the next concrete step should be:

1. prepare `run-007` on a larger canonical subset
2. automate it with the current prompt
3. apply the reducer
4. generate a lightweight summary report over the resulting annotations

That will test whether the annotation layer is already useful for exploratory literary-social analysis, rather than only for benchmark matching.

## run-008 review note

The first larger exploratory reduced run has now been completed as `outputs/run-008`.

### Decision

Treat `run-008` as **acceptable for exploratory downstream use**.

This does **not** mean that the reduced annotations are benchmark-equal.

It means:

- the dominant local movements are usually correct
- the focal characters are usually correct
- compact evasive or diminishing scenes are now handled with reasonable discipline
- the remaining misses are mostly weighting and edge-case granularity problems, not broad structural instability

### What the review showed

Benchmark-overlap passages reviewed during this pass were broadly acceptable:

- `v1-p1-combray#p-17` is effectively clean
- `v1-p1-combray#p-21-p-22` is acceptable, though still somewhat over-compressed toward exclusion
- `v1-p1-combray#p-274-p-275` is acceptable, with a mild drift from `social_status` loss toward `rhetorical_position` loss
- `v1-p1-combray#p-312-p-313` is a meaningful success because it preserves the genuinely mixed Swann-centered structure instead of collapsing onto `M. Vinteuil`

New-unit review also supports keeping the current stack:

- `v1-p1-combray#p-25-p-26` is usable, though its negative downstream effect is under-realized
- `v1-p1-combray#p-270` is strong
- `v1-p1-combray#p-276-p-277` is acceptable
- `v1-p1-combray#p-278-p-279` is strong
- `v1-p1-combray#p-280-p-282` is strong

### Current representative edge case

`v1-p1-combray#p-271-p-273` should be treated as the representative soft spot of the current prompt-plus-reducer stack.

The likely issue is:

- transitional lyrical setup units can still be over-read as genuine positive movement

In this case, Legrandin's rhetorical charm and the narrator's temporary susceptibility are over-credited as a local `narrated_elevation`, even though the unit functions mainly as unstable setup before the sharper Guermantes exposure.

### Decision on heuristics

Do **not** change the reducer heuristics yet.

Reason:

- the current stack is good enough for exploratory analysis
- the known miss class is narrow and intelligible
- reopening the reducer now would likely trade a stable known edge case for broader drift elsewhere

For now, record the edge case and move forward.

## Next stage

The next stage should shift from annotation-shape review to downstream analytical use.

Recommended next move after this checkpoint:

1. compact the current state
2. resume with `run-008` as the current exploratory reduced run
3. begin defining the first lightweight transformation from annotations into candidate notions of local "winning" and "losing"

That transformation work should start by using the existing summary structure rather than by reopening prompt or reducer tuning.

## Local outcome v1

The first downstream transformation should remain deliberately simple.

Initial rule:

- treat each unit as a local scene-level outcome surface
- let `status_effects` carry most of the score
- let `appraisal_events` contribute directional pressure on the same character
- discount `ironized` and `uncertain` events so unstable rhetorical passages do not count like clean wins
- apply a small ambiguity penalty so edge cases do not register as clean local victories

Initial interpretation target:

- this is not yet a full theory of "winning"
- it is a first local advantage/disadvantage pass that can later be revised
- the point is to see whether the reduced annotations produce plausible local signals at all

Practical implementation shape:

- compute per-unit per-character `event_score`
- compute per-unit per-character `status_score`
- combine them into a local `net_score`
- assign a coarse local label such as `win`, `loss`, `mixed`, or `neutral`
- inspect whether the resulting rankings match recent manual review

If this lightweight layer is usable, the next iteration can add:

- sharper treatment of mixed two-movement units
- explicit handling of counterpoint or ironic uplift
- alternative scoring definitions built from the same reduced annotations

### Review checkpoint

The first focused review of `local_outcome_v1` is sufficient to keep it without further tuning.

Reviewed units:

- `v1-p1-combray#p-271-p-273`: accepted as `mixed`
- `v1-p1-combray#p-312-p-313`: accepted as `mixed` with a slight negative lean
- `v1-p1-combray#p-25-p-26`: accepted as a mildly over-positive `win`, understood as an upstream annotation-weighting artifact rather than a scoring-layer problem
- `v1-p1-combray#p-21-p-22`: accepted as a clear `loss`
- `v1-p1-combray#p-17`: accepted as a clear `win`

Decision:

- keep `local_outcome_v1` as the current exploratory scoring layer
- do not tune it further at this stage
- assume that remaining mild misweights are acceptable for higher-level exploratory findings

### Outcome report checkpoint

The first read of `outcome_report_v1` over `run-008` suggests that the current stack is already capable of producing plausible arc-level readings, not only plausible unit-level labels.

Current evidence:

- Swann's sequence reads as oscillatory rather than steadily triumphant
- repeated prestige elevations coexist with real local exclusion
- Legrandin's sequence reads as a sustained degradation arc with one unstable mixed interruption
- the mixed units are sparse and meaningful rather than noisy

Working arc readings from `run-008`:

- Swann is repeatedly revealed as high-prestige but not securely incorporated into the local milieu
- Legrandin undergoes progressive discredit through exposure rather than a single decisive collapse

This is a meaningful downstream success condition.

It suggests that the current annotation plus reduction plus scoring stack is already producing analyzable literary-social structure.

## Next downstream lens

The next scoring move should not replace `local_outcome_v1`.

It should define an alternative lens from the same reduced annotations so that different notions of local "winning" can be compared directly.

The first useful contrast is:

- `prestige_outcome_v1`
- `inclusion_outcome_v1`

### Why this contrast first

Swann already shows why this distinction matters.

In the current subset:

- he often wins on prestige revelation and social rank
- he does not always win on local incorporation or practical belonging

So a single undifferentiated notion of "winning" is less informative than two explicitly different ones.

### Proposed emphasis

`prestige_outcome_v1` should emphasize:

- `social_status`
- `prestige_association`
- positive or negative appraisals tied to recognized standing

`inclusion_outcome_v1` should emphasize:

- `inclusion_exclusion`
- `snub`
- practical admission, refusal, sidelining, or instrumentalization inside a local social setting

### Expected use

The point is not to crown one lens as correct.

The point is to compare them and ask:

- which characters win on prestige but lose on belonging?
- which characters are locally included without commanding prestige?
- where do these two notions diverge passage by passage?

That contrast should be especially useful later for socially volatile figures such as Charlus.

### Three-lens checkpoint

The first comparison across `local_outcome_v1`, `prestige_outcome_v1`, and `inclusion_outcome_v1` produces a meaningful conceptual result for Swann.

Working reading:

- under `prestige_outcome_v1`, Swann is clearly ahead overall
- under `inclusion_outcome_v1`, Swann is slightly behind overall
- under `local_outcome_v1`, Swann remains oscillatory and mildly positive overall

Interpretive value:

- prestige lens captures Swann's rank, elite adjacency, and status revelation
- inclusion lens captures his insecure local belonging and practical exclusion
- local lens preserves the collision between those two facts rather than resolving them into a single axis

This is the first real demonstration that the project can distinguish between different kinds of local "winning" without changing the underlying annotation layer.

Representative units:

- `v1-p1-combray#p-21-p-22` is the clearest inclusion-loss case
- `v1-p1-combray#p-25-p-26` is the clearest prestige-versus-belonging contrast case
- `v1-p1-combray#p-312-p-313` shows how the same unit can read as `mixed`, `loss`, or near-neutral depending on which notion of value is foregrounded

## run-010 transfer check

The first small transfer check beyond the reviewed Swann / Legrandin / Vinteuil subset has now been completed as `outputs/run-010`.

Source slice:

- `v1-p1-combray#p-326`
- `v1-p1-combray#p-327-p-333`
- `v1-p1-combray#p-334-p-335`
- `v1-p1-combray#p-336-p-338`
- `v1-p1-combray#p-339-p-343`
- `v1-p1-combray#p-344-p-345`

This Montjouvain / Mlle Vinteuil sequence was useful as a transfer test because it puts pressure on the current stack in a different way from the earlier socially coded material:

- narrator judgment becomes more morally severe and more explicit
- the scene mixes hesitation, tenderness, complicity, and desecration
- several units risk being over-collapsed into flat blame

### Review result

Treat `run-010` as **acceptable for exploratory downstream use**.

Reviewed units:

- `v1-p1-combray#p-327-p-333`: keep as-is; slightly too one-sided in explanation, but directionally correct as unstable solicitation plus rhetorical timidity rather than neutral description
- `v1-p1-combray#p-336-p-338`: keep as-is; a clean local shift into deliberate profanation framing
- `v1-p1-combray#p-339-p-343`: keep as-is; strongest compression risk in the run, but still acceptable
- `v1-p1-combray#p-344-p-345`: keep as-is; terminal sealing unit, properly handled as a clear narrator-driven diminishment

### Transfer conclusion

The current prompt-plus-reducer-plus-lens stack survives this transfer check without requiring any tuning.

What held:

- the stack preserves the difference between unstable setup and explicit degrading culmination
- it does not falsely neutralize morally severe material simply because the prose remains nuanced
- it still produces sparse, interpretable local movements rather than noisy over-tagging

### Known compression pattern

`v1-p1-combray#p-339-p-343` should be remembered as a representative **mode-collapse** case.

The reduced schema compresses distinct forms of degradation into one negative movement:

- the friend's open brutality
- Mlle Vinteuil's softer, hypocritical complicity

This is a real loss of nuance, but not yet a damaging one for exploratory downstream use.

Decision:

- do not retune the reducer or scoring layers on the basis of this slice
- keep using the current stack for one-step-beyond-reviewed transfer checks
- revisit only if this same compression pattern begins to accumulate across later material

## run-012 transfer check

The second transfer check has now been completed as `outputs/run-012`.

Source slice:

- `v7-p2-m-de-charlus-pendant-la-guerre#p-54`
- `v7-p2-m-de-charlus-pendant-la-guerre#p-55`
- `v7-p2-m-de-charlus-pendant-la-guerre#p-57`
- `v7-p2-m-de-charlus-pendant-la-guerre#p-58`

This Charlus sequence was chosen as a prestige-heavy, socially unstable test outside the earlier Combray material.

It was useful because it puts pressure on the current stack in a different way:

- rank and social force remain highly visible
- local belonging and reciprocity are unstable or absent
- rhetorical brilliance and rhetorical weakness can coexist in the same passage
- later-style Charlus material risks over-collapsing forceful but compromised presence into flat loss

### Review result

Treat `run-012` as **acceptable for exploratory downstream use**.

Reviewed units:

- `v7-p2-m-de-charlus-pendant-la-guerre#p-55`: keep as-is; a clean local humiliation and exclusion case with Morel holding the initiative
- `v7-p2-m-de-charlus-pendant-la-guerre#p-54`: keep as-is; good prestige-versus-belonging case, with hauteur failing to conceal dependence
- `v7-p2-m-de-charlus-pendant-la-guerre#p-57`: keep as-is; mildly over-compressed, but essentially correct as narrator-trimmed rhetorical force
- `v7-p2-m-de-charlus-pendant-la-guerre#p-58`: keep as-is; a small but telling local self-exposure, correctly kept at slight diminishment

### Transfer conclusion

The current stack survives this later Charlus test without requiring any tuning.

What held:

- the stack distinguishes explicit exclusion from more diffuse narrator-driven diminishment
- it handles prestige that remains socially legible even when local control is lost
- it continues to produce plausible local signals in later material with a different social and stylistic texture

### Known soft spot

`v7-p2-m-de-charlus-pendant-la-guerre#p-57` should be remembered as a representative **counterweighted grandeur** case.

The reduced schema slightly under-registers Charlus's real local force:

- he is expansive, commanding, and rhetorically alive in the unit
- the narrator nevertheless explicitly cuts back his authority

The result is slightly flatter than a full human reading, but still acceptable for exploratory downstream use.

Decision:

- do not retune the reducer or scoring layers on the basis of this slice
- treat Charlus as a successful prestige/inclusion transfer check
- proceed to the final planned transfer check before deciding on modest exploratory scaling

## run-014 transfer check

The third and final planned transfer check has now been completed as `outputs/run-014`.

Source slice:

- `v2-p2-noms-de-pays-le-pays#p-216`
- `v2-p2-noms-de-pays-le-pays#p-220`
- `v2-p2-noms-de-pays-le-pays#p-221`

This final slice was chosen as a multi-party social-positioning test rather than another single-character instability case.

It was useful because it puts pressure on the current stack in a different way:

- several named figures are materially involved at once
- invitation, reception, denial, and redistributed favor overlap in one short sequence
- the risk is not only compression, but wrong target selection or overly neat segmentation

### Review result

Treat `run-014` as **acceptable for exploratory downstream use**.

Reviewed units:

- `v2-p2-noms-de-pays-le-pays#p-220`: keep as-is; slightly neatly segmented, but acceptable as grandmother elevation plus Charlus salon efficacy
- `v2-p2-noms-de-pays-le-pays#p-221`: keep as-is; clean narrator-targeted exclusion and snub

Supporting read:

- `v2-p2-noms-de-pays-le-pays#p-216` also behaves plausibly as an invitation-driven inclusion signal for the narrator

### Transfer conclusion

The current stack survives the final multi-party transfer check without requiring any tuning.

What held:

- the stack can distribute local effects across more than one target in adjacent units
- it can distinguish between grandmother elevation and narrator exclusion rather than forcing one focal target for the whole scene
- it remains usable even when a scene could have been segmented in more than one defensible way

### Known soft spot

`v2-p2-noms-de-pays-le-pays#p-220` should be remembered as a representative **mild over-segmentation** case.

The current split is slightly neater than a full human reading:

- Charlus's salon efficacy and the grandmother's rise are captured in `p-220`
- the narrator-targeted denial is largely deferred to `p-221`

This is still acceptable for exploratory downstream use because the adjacent-unit structure preserves the overall local movement.

Decision:

- do not retune the reducer or scoring layers on the basis of this slice
- treat the final planned transfer check as successful

## Transfer-Check Decision

The planned transfer-check phase is now complete.

Outcome:

- the current annotation-plus-reducer-plus-lens stack has passed three distinct transfer checks
- no new failure mode discovered in these checks is serious enough to justify reopening the reducer
- the remaining soft spots are narrow, intelligible, and acceptable for exploratory downstream interpretation

The project should now move from transfer validation to **modest exploratory scaling**.

Recommended next step:

- prepare the first modest larger run using the current stack unchanged
- keep manual review selective rather than exhaustive
- review only representative edge cases, major divergences across lenses, and arc-anchoring units

## First modest larger run design

The first modest larger run should expand an already transfer-checked social zone rather than jump immediately into wholly new terrain.

Recommended source slice:

- chapter: `v2-p2-noms-de-pays-le-pays`
- local arc: the Charlus / Mme de Villeparisis / grandmother salon sequence around the Grand-Hôtel invitation
- purpose: test whether the current stack can sustain a coherent multi-unit social arc across invitation, repositioning, selective exclusion, prestige display, and retrospective admiration

Why this slice first:

- `p-216`, `p-220`, and `p-221` were already transfer-checked and judged acceptable
- the surrounding paragraphs add interpretive density without introducing an unmanageably large cast
- the sequence should read meaningfully across all three implemented lenses:
  - local outcome
  - prestige outcome
  - inclusion outcome

Proposed unit set for the first scaled source run:

- `v2-p2-noms-de-pays-le-pays#p-211-p-213`
- `v2-p2-noms-de-pays-le-pays#p-214-p-215`
- `v2-p2-noms-de-pays-le-pays#p-216`
- `v2-p2-noms-de-pays-le-pays#p-220`
- `v2-p2-noms-de-pays-le-pays#p-221`
- `v2-p2-noms-de-pays-le-pays#p-222`
- `v2-p2-noms-de-pays-le-pays#p-223-p-224`
- `v2-p2-noms-de-pays-le-pays#p-225-p-226`

Selective review rule for this run:

- do **not** review every unit by default
- review arc-anchoring units first
- review any unit where the three lenses diverge sharply
- review any unit that looks like counterweighted grandeur, over-segmentation, or narrator-target ambiguity

If this run is broadly acceptable, the next scaling step should be a somewhat larger candidate set in a new chapter rather than more transfer-style spot checks.

### Follow-up technical note

The boundary-safe alias replacement fix removed the serious substring-corruption bug caused by short aliases like `je` and `moi`.

However, first-person alias substitution still produces some grammatically awkward prompt text in inflected contexts, for example:

- `je ne doutais` -> `le narrateur ne doutais`
- `je sens`-style constructions becoming `le narrateur` plus a first-person verb form

This is much less damaging than the earlier corruption bug, and it is acceptable for current exploratory review.

Do **not** interrupt the current scaling pass to solve it.

But record it as a later cleanup target if narrator-oriented aliasing becomes more central to larger automated runs.

### Review note on the clean rerun

The clean rerun for this first modest larger batch is `outputs/run-018`.

Selective review outcome:

- `p-214-p-215`: keep
- `p-220`: representative reduction miss
- `p-221`: keep
- `p-223-p-224`: keep
- `p-225-p-226`: keep

Interpretive note on `p-220`:

- the raw model output correctly preserved two distinct local movements:
  - narrator-targeted snub
  - grandmother-targeted elevation by Charlus's public framing
- the reduced annotation under-compressed the unit by retaining only the narrator's exclusion
- this should be remembered as the representative reduction miss in `run-018`

Decision:

- treat `run-018` as acceptable for exploratory downstream use
- stop detailed unit-by-unit review for this batch
- do **not** reopen the reducer on the basis of this miss alone

Next move:

- prepare the next somewhat larger candidate set in a new chapter

## run-020 review note

The second scaled reduced run is `outputs/run-020`.

Selective review outcome:

- `p-65-p-67`: keep
- `p-77-p-78`: representative reduction miss

Interpretive note on `p-77-p-78`:

- the raw model output preserved two real local movements:
  - narrator-targeted self-diminishment after Norpois's judgment
  - Bergotte-targeted social exclusion via the Vienna anecdote
- the reduced annotation retained only the Bergotte-side exclusion
- this should be remembered as the representative reduction miss in `run-020`

Decision:

- treat `run-020` as acceptable for exploratory downstream use
- continue scaling without reopening the reducer
- avoid renewed detailed review unless a materially worse miss appears

Next move:

- prepare the next scaled candidate set in another chapter

## run-022 review note

The third scaled reduced run is `outputs/run-022`.

Selective review outcome:

- `p-13-p-14`: keep
- `p-18-p-19`: keep
- `p-37-p-38`: keep

Interpretive note on `p-37-p-38`:

- the raw model output preserved two real local movements:
  - grandfather-targeted prestige discredit of Swann through the Verdurin association
  - successful inclusion of Swann into the Verdurin circle through Odette's mediation
- the reduced annotation retained only the dominant prestige loss
- this is an acceptable reduction, not a representative miss

Decision:

- treat `run-022` as acceptable for exploratory downstream use
- continue scaling without reopening the reducer
- remember `p-37-p-38` as a clean example of acceptable compression where secondary inclusion is dropped in favor of the dominant prestige movement

Next move:

- proceed to the next scaled candidate set unless a new failure class appears

## run-024 review note

The fourth scaled reduced run is `outputs/run-024`.

Selective review outcome:

- `p-47-p-48`: representative reduction miss
- `p-55-p-57`: keep

Interpretive note on `p-47-p-48`:

- the raw model output preserved the correct dominant structure:
  - narrator-endorsed elevation of Swann through his excellent first impression and tact inside the Verdurin circle
  - a smaller negative counter-movement when M. Verdurin pushes back on Swann's mockery of the pianist's aunt
- the reduced annotation dropped the dominant positive movement and retained only the smaller negative counterpoint
- this should be remembered as the representative reduction miss in `run-024`

Interpretive note on `p-55-p-57`:

- the raw model output preserved two real local movements:
  - Mme Verdurin publicly overrules M. Verdurin
  - the pianist is implicitly protected as a favored insider
- the reduced annotation retained only the more central authority movement inside the salon
- this is acceptable compression, not a miss

Decision:

- treat `run-024` as acceptable for exploratory downstream use
- continue scaling without reopening the reducer
- remember `p-47-p-48` as a true reduction inversion, not merely a dropped secondary signal

Next move:

- proceed to the next scaled candidate set unless a broader inversion pattern emerges

## run-026 review note

The fifth scaled reduced run is `outputs/run-026`.

Selective review outcome:

- `p-69-p-72`: keep

Interpretive note on `p-69-p-72`:

- the raw model output preserved a small salon-control movement:
  - Mme Verdurin checks the painter's joke and keeps command of the conversational floor
  - the painter takes a minor rhetorical loss
- the reduced annotation retains that same local movement
- the passage remains playful, but the retained signal is still acceptable

Decision:

- treat `run-026` as acceptable for exploratory downstream use
- reduce review intensity further from this point
- only stop to inspect when a report surfaces a clearly surprising signal

Next move:

- continue scaling in modest batches and interrupt only for surprising report-level outcomes

## run-036 review note

The tenth scaled reduced run is `outputs/run-036`.

This batch triggered the new stop condition because the report-level shape became clearly surprising.

Selective review outcome:

- `p-176-p-182`: representative annotation-stage miss
- `p-183-p-184`: representative reduction-stage miss in the same local failure class

Interpretive note on `p-176-p-182`:

- the raw annotation itself failed
- it retained only Swann's timidity, indirectness, and reliance on the catleya pretext
- it dropped the dominant consummatory movement:
  - Odette's receptive assent
  - the realization of Swann's long-desired kiss and possession
  - the emergence of a durable shared erotic language
- this is an annotation-layer miss, not merely a reduction miss

Interpretive note on `p-183-p-184`:

- the raw annotation preserved the correct two-part structure:
  - social diminishment as others read Swann as "tenu" by a woman
  - narrator-endorsed emotional renewal through love, music, and repeated intimacy
- the reduced annotation retained only the social diminishment
- this is a reduction-stage miss in the same local semantic zone

Broader failure class:

- the current stack is weaker on erotic-consummation and affective-renewal passages when the dominant positive movement is braided with:
  - timidity
  - dependence
  - social diminishment
  - ironic or qualifying narrator framing
- in these passages, the model stack tends to over-select weakness, awkwardness, or dependency and under-select the dominant local success or renewal

Decision:

- do not treat `run-036` as a normal fast-path keep
- treat this batch as a boundary-finding result
- do not continue blind scaling through the same Swann/Odette consummation zone without adjustment or a deliberate decision to tolerate this failure mode

Next move:

- decide whether to revise the prompt for this failure class or to leave the current prompt as-is and redirect scaling toward regions where the stack is already reliable

Prompt revision and tiny regression check:

- revised `proust/prompts/prompt.md` narrowly for the documented failure class
- added an explicit rule to distinguish:
  - mode of attainment
  - attained local outcome
- added an explicit positive example for hesitant but successful consummation / affective fulfillment

Targeted regression set:

- prepared `outputs/run-037` as a four-unit source run:
  - `v1-p1-combray#p-17` as a stable positive control
  - `v1-p1-combray#p-312-p-313` as a stable mixed control
  - `v1-p2-un-amour-de-swann#p-176-p-182` as the annotation-stage regression target
  - `v1-p2-un-amour-de-swann#p-183-p-184` as the reduction-zone regression target
- automated to `outputs/run-038` and reprocessed with reduction

Regression result:

- `p-176-p-182` improved in the intended direction
  - old behavior: retained only timidity / pretext / indirectness
  - new behavior: preserved realized intimacy and the shared `faire catleya` code as the dominant local movement
  - local outcome now reads as a Swann win rather than a loss
- `p-183-p-184` improved in the intended direction
  - old behavior: retained only the social diminishment of being seen as "tenu" by a woman
  - new behavior: preserved narrator-endorsed happiness, repose, and renovation as the dominant local movement
  - local outcome now reads as a Swann win rather than a loss

Control behavior:

- `p-17` remained stable and directionally unchanged
- `p-312-p-313` remained a mixed passage with admiration plus social discredit
  - the new output was slightly more compressed than the reviewed benchmark
  - it preserved the same essential mixed shape, so this looks acceptable for now

Working conclusion:

- the narrow prompt intervention appears to address the identified failure class without obvious collateral damage in the tiny check set
- this merits one nearby follow-up check before resuming broader scaling through the same Swann/Odette region

Nearby follow-up check:

- prepared `outputs/run-039` and automated/reduced to `outputs/run-040`
- tested three adjacent units:
  - `v1-p2-un-amour-de-swann#p-167-p-170`
  - `v1-p2-un-amour-de-swann#p-171-p-175`
  - `v1-p2-un-amour-de-swann#p-185-p-186`

Result:

- `p-171-p-175` remained directionally stable as a modest local Swann gain
- `p-185-p-186` remained directionally stable as a Swann emotional loss, with an additional acceptable positive appraisal of Odette
- `p-167-p-170` shifted too far
  - previous reading: anxious pursuit and obsessive need dominated
  - new reading: the ending relief at finding Odette was promoted to the dominant local outcome
  - this suggests the new consummation / renewal rule is slightly over-broad and can incorrectly absorb mere relief at reunion into a positive local win

Interpretive refinement now indicated:

- the repair should apply when the passage culminates in:
  - realized intimacy
  - mutual receptivity
  - successful consummation
  - narrator-endorsed renewal
- it should not automatically fire for:
  - anxious search
  - reunion relief
  - the end of uncertainty before consummation has actually occurred

Operational conclusion:

- do not rerun prior batches yet
- first tighten the prompt one notch so that reunion relief does not get conflated with consummation or renewal

Second prompt tightening check:

- tightened the prompt further to say that reunion relief or the mere end of uncertainty should not count as consummation or renewal unless the current passage itself reaches realized intimacy or explicit narrator-endorsed transformation
- reran the same nearby three-unit check as `outputs/run-041`

Result:

- `p-171-p-175` remained acceptable as a modest gain
- `p-185-p-186` remained acceptable as a Swann loss
- `p-167-p-170` still flipped incorrectly to a Swann win

Interpretive conclusion:

- the current prompt-only fix is still too weakly specified at the `p-167-p-170` boundary
- the model continues to treat sudden reunion relief as dominant local success even when the larger unit is better read as anxious pursuit ending in temporary relief

Next recommendation:

- do not rerun older runs yet
- make one more prompt revision, but this time with a sharper constraint:
  - successful consummation or renewal must be realized **within the current passage**
  - do not let an uplifting ending sentence override a passage whose dominant body is anxious search, obsession, or dependency
  - when a passage is structurally weighted toward distress and only ends in relief, prefer a mixed or negative reading unless the positive culmination is clearly the main narrated point

Third prompt revision check:

- revised the prompt again to say explicitly that a single closing note should not outweigh the dominant movement of the passage unless that culmination is plainly the point
- reran the same nearby three-unit check as `outputs/run-042`

Result:

- `p-171-p-175` remained acceptable as a modest gain
- `p-185-p-186` remained acceptable as a Swann loss
- `p-167-p-170` still flipped to a Swann win

Conclusion:

- the intended weighting principle is now present in the prompt, but the model is still not reliably obeying it on this boundary case
- further prompt-only tweaking may have diminishing returns here

Practical recommendation:

- still do not rerun earlier runs
- either:
  - accept `p-167-p-170` as a tolerated prompt-level miss while keeping the broader revision, or
  - introduce a downstream heuristic / benchmark note for this exact failure shape rather than continuing prompt iteration indefinitely

## Stage transition standard

The project should not move between phases by feel alone. The transition standard should be:

### 1. Benchmark-fidelity phase -> transfer-check phase

Move on when all of the following are true:

- there is at least one reviewed benchmark run that covers the core schema and prompt behavior
- benchmark review shows the stack is usable even if not benchmark-equal
- remaining misses are intelligible and local rather than global or schema-breaking

### 2. Transfer-check phase -> modest exploratory scaling

Move on when all of the following are true:

- at least three transfer checks in distinct local settings have been judged acceptable
- no unresolved failure class appears to invert the dominant local movement across a whole type of passage
- downstream reports remain directionally useful on those transfer checks

### 3. Modest exploratory scaling -> larger automated scaling

Move on when all of the following are true:

- multiple contiguous scaled runs in sequence are acceptable without prompt or reducer retuning
- review has already shifted from unit-by-unit inspection to selective report-led spot checks
- surprising outputs are now exceptions rather than the normal review experience
- the stack is producing analyzable higher-level structure for at least one sustained character zone

Operationally, a practical threshold for this project is:

- roughly two or three more contiguous scaled batches that do not produce a new failure class
- no need to reopen the reducer or prompt during those batches
- no report-level inversion serious enough to force renewed close review

If those conditions hold, the next stage should be:

- larger automated chunks
- report-first inspection
- unit review only for genuinely surprising signals

### 4. Larger automated scaling -> intervention phase

Do **not** keep scaling blindly if any of the following appears:

- a recurring failure class starts to distort downstream character arcs
- one lens becomes systematically misleading in a recurrent passage type
- new terrain exposes alias or parsing gaps that materially degrade coverage
- report-level shapes become surprising often enough that selective review is no longer selective

If that happens, pause scaling and do a targeted intervention:

- prompt revision
- reducer heuristic
- alias-map extension
- narrow benchmark addition

Current status:

- benchmark-fidelity phase: complete
- transfer-check phase: complete
- modest exploratory scaling: in progress
- current objective: reach the larger-automated-scaling threshold rather than continue indefinite close review

Recent scaling note:

- `run-084`, `v2-p1-autour-de-mme-swann#p-37-p-44` is a representative acceptable-but-noted reduction miss
- the passage stages a successful Norpois prestige performance before the family while also subjecting his rhetoric to narrator irony
- the stack over-selected the ironic demystification and returned a local Norpois loss
- working judgment: this belongs to the already-known class where ironic undercutting can outweigh the dominant enacted prestige or social success
- action: note it, keep scaling, and do not reopen prompt or reducer on this basis alone

Checkpoint note for resumption:

- the current contiguous scaling pass has continued through `run-101`
- completed automated reduced runs in this latest block:
  - `run-097` covering `p-145-p-150`, `p-151-p-159`, `p-160-p-169`
  - `run-099` covering `p-170-p-179`, `p-180-p-185`, `p-186-p-195`
  - `run-101` covering `p-196-p-204`, `p-205-p-214`, `p-215-p-221`
- working judgment on those runs:
  - `run-097` is acceptable as a coherent negative Swann/Odette social-hunger and social-ceiling stretch
  - `run-099` is acceptable and shows a useful cross-lens divergence in `p-186-p-195`, where Swann reads as a prestige win but only mixed on inclusion
  - `run-101` is acceptable and gives a plausible Bergotte mini-arc: early diminishment, then mixed or neutral correction, then renewed win
- no new prompt-level or reducer-level failure class appeared in those batches
- the only active operational risk is unified exec session pressure during automation/report work; this is not an interpretive problem, but it may force a resume if the shell environment becomes unstable

If work resumes from this checkpoint, the next concrete step should be:

1. continue the same contiguous pass immediately after `v2-p1-autour-de-mme-swann#p-215-p-221`
2. prepare the next source batch as `run-102`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. stop only if a genuinely surprising report signal appears or if session pressure makes continued execution unreliable

## Current operating mode

The project is no longer in a benchmark-building phase.

It is also not in a phase where every new unit should be hand-walked or manually normalized.

The current mode is:

- transfer-check while scaling
- build a materially larger exploratory corpus
- use reports as the primary review surface
- inspect individual units only when a report produces a genuinely surprising signal

This means the practical goal is not:

- manually annotate all of ISLT at close-reading granularity

The practical goal is:

- build enough confidence that the current prompt-plus-reducer-plus-lens stack transfers beyond the original benchmark zone
- scale through modest contiguous batches while the outputs remain directionally trustworthy
- stop spending review time at the unit level unless higher-level reports indicate a real interpretive problem

## Phase model

Use this as the current working phase map.

### 1. Early phase: benchmarked reliability

Purpose:

- establish a reviewed baseline
- verify schema and prompt discipline
- identify broad failure modes

This phase is complete.

### 2. Current phase: transfer-check while scaling

Purpose:

- continue through nearby or contiguous material
- confirm that the stack stays usable outside the original benchmark terrain
- treat recurring distortions, not isolated misses, as the main thing to watch

This phase is the current default.

### 3. Next phase: larger automated runs with report-led spot checks

Purpose:

- move to larger contiguous chunks
- read report shapes first
- inspect units only when the report surface looks genuinely wrong or unstable

This is the next phase threshold we are actively trying to reach.

### 4. Later phase: downstream corpus analysis

Purpose:

- use the growing corpus to study recurring social-literary structures across characters, sections, and lenses
- compare prestige, inclusion, and mixed local-outcome patterns at larger scale

This remains the destination.

## Current decision rule

When a new contiguous batch has been automated, reduced, and scored:

1. read the three report lenses first
2. ask whether the batch produces a plausible higher-level shape
3. inspect individual units only if the reports show something genuinely surprising

Examples of genuinely surprising signals:

- a character arc appears inverted relative to the surrounding context
- one lens diverges sharply from the others in a way that does not seem textually motivated
- the same passage type starts producing the same wrong directional result repeatedly
- a new zone exposes alias, parsing, or reduction problems that distort multiple units at once

Examples of signals that do **not** justify dropping into close review by default:

- a familiar tolerated edge case
- mild weighting disagreements within an otherwise plausible arc
- one or two locally debatable mixed-unit judgments
- imperfect benchmark equality in terrain no longer being used as a benchmark task

## Stop condition for close review

Do not keep extending unit-by-unit manual checking indefinitely.

Close review should now be treated as an exception path, not the standard workflow.

The default should be:

- continue scaling in modest contiguous batches
- keep the current prompt and reducer unless a recurring failure class starts to matter downstream
- spend review effort at the report level first

## Practical next-step rule

For the next stretch of work, use this operating rule:

1. continue a bit further in the current contiguous Swann-centered region
2. once the corpus is comfortably larger, shift to bigger automated chunks
3. inspect units only when a report throws off a genuinely surprising signal
4. intervene only if a recurring failure class begins to distort downstream results

The destination is:

- downstream analysis over a larger exploratory corpus

The destination is not:

- indefinite close review of every annotation unit

## Resumption note

If work resumes after a pause, assume the following unless new evidence forces a change:

- the current stack is good enough to keep scaling
- report-led inspection is now the primary review method
- unit-level inspection is reserved for surprising report output
- prompt, reducer, schema, and alias changes should be targeted interventions rather than routine companions to each run

## run-103 scaling checkpoint

The contiguous scaling pass has now continued through `run-103`.

Source batch:

- `run-102` covering:
  - `v2-p1-autour-de-mme-swann#p-222-p-230`
  - `v2-p1-autour-de-mme-swann#p-231-p-240`
  - `v2-p1-autour-de-mme-swann#p-241-p-250`

Automated reduced output:

- `run-103`

Report-first reading:

- `v2-p1-autour-de-mme-swann#p-222-p-230` reads as a clear Norpois loss
- `v2-p1-autour-de-mme-swann#p-231-p-240` reads as a clear Swann prestige win
- `v2-p1-autour-de-mme-swann#p-241-p-250` reads as a local rhetorical win for `la mère du narrateur`

Working judgment:

- treat `run-103` as acceptable for continued scaling
- no obvious new recurring failure class appeared at the report level
- the mother-centered final unit is somewhat narrower in focus than the surrounding Swann material, but not surprising enough to justify reopening close review or retuning the stack

Operational note:

- this batch needed the expanded alias map carried forward from `run-101`; do not prepare later batches from the default starter alias map for this region

If work resumes from this checkpoint, the next concrete step should be:

1. continue immediately after `v2-p1-autour-de-mme-swann#p-241-p-250`
2. prepare the next source batch as `run-104`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-105 scaling checkpoint

The contiguous scaling pass has now continued through `run-105`.

Source batch:

- `run-104` covering:
  - `v2-p1-autour-de-mme-swann#p-251-p-260`
  - `v2-p1-autour-de-mme-swann#p-261-p-270`
  - `v2-p1-autour-de-mme-swann#p-271-p-280`

Automated reduced output:

- `run-105`

Report-first reading:

- `v2-p1-autour-de-mme-swann#p-251-p-260` reads as a local Bergotte win
- `v2-p1-autour-de-mme-swann#p-261-p-270` reads as an Odette loss
- `v2-p1-autour-de-mme-swann#p-271-p-280` reads primarily as another Bergotte win, with Odette flattened toward neutral in the reports

Working judgment:

- treat `run-105` as acceptable for continued scaling
- no obvious new recurring failure class appeared at the report level
- the final unit contains more local friction around Odette and Gilberte than the report surface preserves, but this looks like ordinary reduction pressure rather than a new report-level inversion

Operational note:

- the API automation for this batch completed in a staggered way and left a misleading `automation` summary in `run-105/run.json`
- the raw files, reduced annotations, and report outputs are complete and usable
- this is a runner bookkeeping issue, not an interpretive blocker

If work resumes from this checkpoint, the next concrete step should be:

1. continue immediately after `v2-p1-autour-de-mme-swann#p-271-p-280`
2. prepare the next source batch as `run-106`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-107 scaling checkpoint

The contiguous scaling pass has now continued through `run-107`.

Source batch:

- `run-106` covering:
  - `v2-p1-autour-de-mme-swann#p-281-p-285`
  - `v2-p1-autour-de-mme-swann#p-286-p-290`
  - `v2-p1-autour-de-mme-swann#p-291-p-300`

Automated reduced output:

- `run-107`

Report-first reading:

- `v2-p1-autour-de-mme-swann#p-281-p-285` reads as a Gilberte loss on the inclusion axis
- `v2-p1-autour-de-mme-swann#p-286-p-290` reads as an Odette social-position win
- `v2-p1-autour-de-mme-swann#p-291-p-300` reads as a Mme Cottard loss paired with a smaller Odette rhetorical win

Working judgment:

- treat `run-107` as acceptable for continued scaling
- no obvious new recurring failure class appeared at the report level
- the Odette-positive readings are textually grounded in local hostess control and conversational tact, so they do not require intervention

Operational note:

- automation again completed in a staggered way and required repeated one-unit requests before all files appeared
- this remains a runner/API execution nuisance rather than an interpretive blocker

If work resumes from this checkpoint, the next concrete step should be:

1. continue immediately after `v2-p1-autour-de-mme-swann#p-291-p-300`
2. prepare the next source batch as `run-108`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-109 scaling checkpoint

The contiguous scaling pass has now continued through `run-109`.

Source batch:

- `run-108` covering:
  - `v2-p1-autour-de-mme-swann#p-301-p-305`
  - `v2-p1-autour-de-mme-swann#p-306-p-310`
  - `v2-p1-autour-de-mme-swann#p-311-p-315`

Automated reduced output:

- `run-109`

Report-first reading:

- `v2-p1-autour-de-mme-swann#p-301-p-305` reads as a Swann loss
- `v2-p1-autour-de-mme-swann#p-306-p-310` reads as a docteur Cottard loss
- `v2-p1-autour-de-mme-swann#p-311-p-315` reads as an Odette mixed-to-positive unit: a local prestige win in the prestige lens, but only mixed in the inclusion lens

Working judgment:

- treat `run-109` as acceptable for continued scaling
- no obvious new recurring failure class appeared at the report level
- the final unit is a useful lens-divergence case rather than a distortion: Odette looks socially elevated while also carrying local negative appraisal pressure

## Runtime note

The apparent automation "failure" on this batch was not an API-limit problem.

What happened instead:

- prompts in this region are large
- requests are processed sequentially
- the old runner behavior made long quiet periods look like stalls

The runner now writes automation progress incrementally into `run.json` while a request loop is still in progress.

For future runs:

- monitor `automation.in_progress`
- monitor `automation.completed_unit_count`
- check `raw/` and `annotations/` before assuming failure
- retry only when there is evidence that progress has actually stopped

If work resumes from this checkpoint, the next concrete step should be:

1. continue immediately after `v2-p1-autour-de-mme-swann#p-311-p-315`
2. prepare the next source batch as `run-110`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-111 scaling checkpoint

The contiguous scaling pass has now continued through `run-111`.

Source batch:

- `run-110` covering:
  - `v2-p1-autour-de-mme-swann#p-316-p-320`
  - `v2-p1-autour-de-mme-swann#p-321-p-325`
  - `v2-p1-autour-de-mme-swann#p-326-p-330`

Automated reduced output:

- `run-111`

Report-first reading:

- `v2-p1-autour-de-mme-swann#p-316-p-320` reads as an Odette aesthetic-elevation win
- `v2-p1-autour-de-mme-swann#p-321-p-325` reads as a Gilberte emotional-power win
- `v2-p1-autour-de-mme-swann#p-326-p-330` reads as a strong Odette social-status win

Working judgment:

- treat `run-111` as acceptable for continued scaling
- the final batch of this chapter is strongly Odette-positive, but the reduced annotations show that this is textually grounded in sustained narrator-led elevation rather than a new distortion
- the Gilberte unit is best understood as an affective-power formulation, not as a reconciliation or social incorporation signal

### Pass conclusion

This completes the current contiguous pass through `v2-p1-autour-de-mme-swann`.

At the report level, the chapter now yields a usable higher-level shape:

- repeated Odette prestige and hostess authority gains
- repeated Swann losses or humiliations in adjacent social framing
- meaningful lens divergence in the middle of the pass
- a late chapter turn toward more explicit Odette elevation

This is a good stopping point for this contiguous block.

## Runtime confirmation

`run-111` also confirms that the new progress-writing runner behavior is working as intended.

Observed pattern:

- the command remained quiet for long stretches
- `run.json` showed `automation.in_progress: true`
- `completed_unit_count` moved from `0` to `1` to `2` to `3`
- the final command exit then wrote the completed summary

So future long runs should be monitored through progress fields rather than treated as failed during quiet periods.

If work resumes from this checkpoint, the next concrete step should be:

1. choose the next contiguous chapter or section to extend the exploratory corpus
2. prepare the next source batch as `run-112`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-113 transfer checkpoint

The first transfer batch into `v2-p2-noms-de-pays-le-pays` has now been completed as `run-113`.

Source batch:

- `run-112` covering:
  - `v2-p2-noms-de-pays-le-pays#p-11-p-15`
  - `v2-p2-noms-de-pays-le-pays#p-16-p-20`
  - `v2-p2-noms-de-pays-le-pays#p-21-p-25`

Automated reduced output:

- `run-113`

Preparation note:

- `la grand-mère` was added to the carried-forward alias map for this section

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-11-p-15` reads as a Françoise win
- `v2-p2-noms-de-pays-le-pays#p-16-p-20` reads as a grandmother win
- `v2-p2-noms-de-pays-le-pays#p-21-p-25` reads as a second grandmother win

Working judgment:

- treat `run-113` as acceptable for continued scaling
- the uniformly positive shape is not a new distortion; the reduced annotations show genuine narrator-led elevation of Françoise and the grandmother in this opening cluster
- this is therefore a successful transfer into a somewhat different affective and familial register, not just socially competitive salon material

Runtime note:

- the new progress-writing behavior again worked as intended
- `run.json` showed the live transition from `completed_unit_count: 0` to `1` to `2` to `3`
- the final unit was slower, but the run remained clearly healthy throughout

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-114`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-115 transfer checkpoint

The second transfer batch inside `v2-p2-noms-de-pays-le-pays` has now been completed as `run-115`.

Source batch:

- `run-114` covering:
  - `v2-p2-noms-de-pays-le-pays#p-61-p-65`
  - `v2-p2-noms-de-pays-le-pays#p-66-p-70`
  - `v2-p2-noms-de-pays-le-pays#p-71-p-75`

Automated reduced output:

- `run-115`

Preparation note:

- `M. de Stermaria` was added to the carried-forward alias map for this cluster

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-61-p-65` reads as a local loss for the grandmother on inclusion and a local loss or mixed result for `Mme de Villeparisis`
- `v2-p2-noms-de-pays-le-pays#p-66-p-70` reads as a `Mme de Cambremer` prestige win
- `v2-p2-noms-de-pays-le-pays#p-71-p-75` reads as a `M. de Stermaria` prestige win

Working judgment:

- treat `run-115` as a strong and useful transfer result
- this is not just acceptable; it is a conceptually valuable batch because it produces exactly the kind of prestige/inclusion structure the project is meant to compare downstream
- especially useful is `p-61-p-65`, where the grandmother's principled refusal to recognize `Mme de Villeparisis` lowers local inclusion even though the negative social event is directed outward

This is a meaningful confirmation that the current stack can capture:

- prestige carried by rank and reception
- exclusion or self-isolation effects
- the divergence between social standing and local incorporation

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-116`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-117 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-117`.

Source batch:

- `run-116` covering:
  - `v2-p2-noms-de-pays-le-pays#p-76-p-80`
  - `v2-p2-noms-de-pays-le-pays#p-81-p-85`
  - `v2-p2-noms-de-pays-le-pays#p-86-p-90`

Automated reduced output:

- `run-117`

Preparation notes:

- the carried-forward alias map was extended modestly for this cluster with:
  - `Aimé`
  - `Mlle de Stermaria`
  - `marquis de Cambremer`

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-76-p-80` reads as a local `M. de Stermaria` loss
- `v2-p2-noms-de-pays-le-pays#p-81-p-85` reads as a local `Mlle de Stermaria` loss or mixed-negative result
- `v2-p2-noms-de-pays-le-pays#p-86-p-90` reads as a Françoise loss

Working judgment:

- treat `run-117` as acceptable for continued scaling
- the batch remains directionally coherent across all three lenses
- `p-81-p-85` is the mildest case, but the reduced reading still tracks the narrator's emphasis on Mlle de Stermaria's hardness and limited sympathy, tempered by sensual attraction
- `p-86-p-90` compresses a broader social weave into a Françoise-centered local loss; this looks like ordinary reduction pressure rather than a new recurring failure class

Runtime note:

- the long-run monitoring rule was reaffirmed here
- `run-117` again spent a long time in silence before the final unit completed
- but `run.json` showed healthy progress from `completed_unit_count: 0` to `2` before finishing at `3`
- the correct practical response is still to monitor progress fields before retrying or assuming API failure

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-118`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-119 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-119`.

Source batch:

- `run-118` covering:
  - `v2-p2-noms-de-pays-le-pays#p-91-p-95`
  - `v2-p2-noms-de-pays-le-pays#p-96-p-100`
  - `v2-p2-noms-de-pays-le-pays#p-101-p-105`

Automated reduced output:

- `run-119`

Preparation notes:

- the carried-forward alias map was extended modestly for this cluster with:
  - `princesse de Luxembourg`
  - `le père du narrateur`
  - `Mme Blandais`
  - `Mme Poncin`
- a real alias collision appeared during preparation:
  - the existing bare alias `princesse` under `princesse des Laumes` was partially rewriting `princesse de Luxembourg` in preprocessed text
  - for this run, the bare `princesse` alias was removed from the carried-forward map before automation

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-91-p-95` reads as a local loss for the grandmother
- `v2-p2-noms-de-pays-le-pays#p-96-p-100` reads as a win for the narrator's father
- `v2-p2-noms-de-pays-le-pays#p-101-p-105` reads as a local `Mme de Villeparisis` loss

Working judgment:

- treat `run-119` as acceptable for continued scaling
- the three reports are strongly coherent with one another after the alias fix
- `p-91-p-95` correctly captures condescending aristocratic benevolence as local diminishment rather than hospitality
- `p-96-p-100` correctly captures the local elevation of the narrator's father through Villeparisis's magnifying attention
- `p-101-p-105` correctly captures the gossip-circle suspicion directed at `Mme de Villeparisis`

Method note:

- this batch produced a useful targeted intervention rule
- when a new titled figure enters a section, avoid generic title aliases like bare `princesse` if they can collide with a distinct titled character in the same run
- this was not an API or runtime problem; it was a real alias-map hygiene issue caught before automation

Runtime note:

- `run-119` again showed the same long silent interval before completion
- despite that silence, the run completed cleanly with `3/3` written annotations and no parse or validation errors

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-120`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-121 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-121`.

Source batch:

- `run-120` covering:
  - `v2-p2-noms-de-pays-le-pays#p-106-p-110`
  - `v2-p2-noms-de-pays-le-pays#p-111-p-115`
  - `v2-p2-noms-de-pays-le-pays#p-116-p-120`

Automated reduced output:

- `run-121`

Preparation note:

- the no-bare-`princesse` alias rule from `run-119` was carried forward unchanged into this batch

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-106-p-110` reads as neutral
- `v2-p2-noms-de-pays-le-pays#p-111-p-115` reads as a `princesse de Luxembourg` loss
- `v2-p2-noms-de-pays-le-pays#p-116-p-120` reads as a grandmother win

Working judgment:

- treat `run-121` as acceptable for continued scaling
- `p-111-p-115` is the clearest result: the bourgeois misunderstanding of aristocratic status is correctly captured as a local discrediting of the princess
- `p-106-p-110` is sparse and its neutral reduction is acceptable; this looks like an ordinary thin-unit outcome rather than a structural miss
- `p-116-p-120` initially looked surprising at report level, but the reduced annotation is defensible: it captures the narrator's local admiration for the grandmother's composed, practical independence in handling the doctor's advice

Runtime note:

- `run-121` followed the now-familiar pattern of long silent requests with eventual healthy completion
- progress monitoring again worked as intended and no retry was needed

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-122`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-123 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-123`.

Source batch:

- `run-122` covering:
  - `v2-p2-noms-de-pays-le-pays#p-121-p-125`
  - `v2-p2-noms-de-pays-le-pays#p-126-p-130`
  - `v2-p2-noms-de-pays-le-pays#p-131-p-135`

Automated reduced output:

- `run-123`

Preparation note:

- the no-bare-`princesse` alias rule was carried forward unchanged again

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-121-p-125` reads as a mixed-positive `Mme de Villeparisis` result
- `v2-p2-noms-de-pays-le-pays#p-126-p-130` reads as a `Mme de Villeparisis` loss
- `v2-p2-noms-de-pays-le-pays#p-131-p-135` reads as a Bergotte win

Working judgment:

- treat `run-123` as acceptable for continued scaling
- `p-121-p-125` and `p-126-p-130` together produce a useful internally differentiated reading of `Mme de Villeparisis`: cultured local authority in one unit, then narrowed and socially biased literary judgment in the next
- `p-131-p-135` initially looked surprising at report level and therefore justified inspection
- the reduced annotation is defensible: the unit does contain a real Bergotte-valued prestige signal, so this is not a random alias hallucination or parser collapse

Method note:

- this batch is a good example of the current review rule working as intended
- the report flagged a genuinely surprising signal
- a quick reduced-annotation inspection was enough to show that the signal was textually grounded, so no prompt or schema intervention is needed

Runtime note:

- `run-123` again completed cleanly after a long silent interval
- progress monitoring via `run.json` remained the correct operational response

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-124`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-125 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-125`.

Source batch:

- `run-124` covering:
  - `v2-p2-noms-de-pays-le-pays#p-136-p-140`
  - `v2-p2-noms-de-pays-le-pays#p-141-p-145`
  - `v2-p2-noms-de-pays-le-pays#p-146-p-150`

Automated reduced output:

- `run-125`

Preparation note:

- the no-bare-`princesse` alias rule was carried forward unchanged again

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-136-p-140` reads as neutral
- `v2-p2-noms-de-pays-le-pays#p-141-p-145` reads as a `Mme de Villeparisis` win
- `v2-p2-noms-de-pays-le-pays#p-146-p-150` reads as a `Mme de Villeparisis` win

Working judgment:

- treat `run-125` as acceptable for continued scaling
- this batch is narrow in cast and slightly surprising in emphasis, because all of the non-neutral signal concentrates on `Mme de Villeparisis`
- quick reduced-annotation inspection showed the two positive units are still textually defensible:
  - one captures narrator admiration for her aesthetic discernment
  - the other captures her local authority and positioning around literary judgment
- this looks like ordinary compression of a socially thinner stretch, not a new distortion pattern

Runtime note:

- `run-125` completed cleanly with `3/3` valid annotations and no pending units

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-126`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-127 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-127`.

Source batch:

- `run-126` covering:
  - `v2-p2-noms-de-pays-le-pays#p-151-p-155`
  - `v2-p2-noms-de-pays-le-pays#p-156-p-160`
  - `v2-p2-noms-de-pays-le-pays#p-161-p-165`

Automated reduced output:

- `run-127`

Preparation note:

- the carried-forward alias map remained stable; no new intervention was needed for this batch

Report-first reading:

- `v2-p2-noms-de-pays-le-pays#p-151-p-155` reads as a `Mme de Villeparisis` loss
- `v2-p2-noms-de-pays-le-pays#p-156-p-160` again reads as a `Mme de Villeparisis` loss
- `v2-p2-noms-de-pays-le-pays#p-161-p-165` turns back toward a `Mme de Villeparisis` win

Working judgment:

- treat `run-127` as acceptable for continued scaling
- the pattern looks like a coherent local oscillation rather than model instability:
  - two units of diminishment or narrowing
  - then one recovery into local positive standing
- grandmother signal appears alongside this sequence, but not in a way that suggests alias failure or report incoherence

Runtime note:

- `run-127` completed cleanly with `3/3` valid annotations and no pending units

If work resumes from this checkpoint, the next concrete step should be:

1. continue further into `v2-p2-noms-de-pays-le-pays`
2. prepare the next source batch as `run-128`
3. automate to a fresh output run, reduce it, and read the three report lenses
4. inspect units only if a genuinely surprising report signal appears

## run-129 transfer checkpoint

The next contiguous Balbec batch has now been completed as `run-129`.

Source batch:

- `run-128` covering:
  - `v2-p2-noms-de-pays-le-pays#p-166-p-170`
  - `v2-p2-noms-de-pays-le-pays#p-171-p-175`
  - `v2-p2-noms-de-pays-le-pays#p-176-p-180`

Automated reduced output:

- `run-129`

Preparation notes:

- the carried-forward map remained stable
- the next Saint-Loup cluster required two modest alias additions:
  - `Robert de Saint-Loup`
  - `M. de Marsantes`

Report-first reading:

- across all three lenses, the batch is dominated by `Robert de Saint-Loup`
- `v2-p2-noms-de-pays-le-pays#p-166-p-170` reads as mixed in the local lens and neutral in the inclusion and prestige lenses
- `v2-p2-noms-de-pays-le-pays#p-171-p-175` reads as a Saint-Loup win
- `v2-p2-noms-de-pays-le-pays#p-176-p-180` reads as the strongest Saint-Loup win of the batch, especially in inclusion

Working judgment:

- treat `run-129` as a strong transfer result
- the report shape is exactly the kind of directional pattern the current phase is meant to test:
  - anticipatory or ambivalent first contact
  - then clear local elevation
  - then strong incorporation and admiration
- the first unit's weaker outcome is not a problem signal; it is intelligible compression of mixed valuation, where admiration is partly offset by discredit-by-association and emotional drag
- no new recurring failure class appears here

Runtime note:

- `run-129` again completed cleanly after the familiar long silent interval
- reduction and reporting both succeeded with `3/3` valid annotations and no pending units

Assessment note:

- this is a good place to pause and step back
- the contiguous transfer-check stretch is now materially larger, and the evidence burden is starting to shift from unit-level validation toward higher-level corpus use

## run-131 larger-scaling checkpoint

The first larger automated scaling batch has now been completed as `run-131`.

Source batch:

- `run-130` covering:
  - `v2-p2-noms-de-pays-le-pays#p-181-p-185`
  - `v2-p2-noms-de-pays-le-pays#p-186-p-190`
  - `v2-p2-noms-de-pays-le-pays#p-191-p-195`
  - `v2-p2-noms-de-pays-le-pays#p-196-p-200`
  - `v2-p2-noms-de-pays-le-pays#p-201-p-205`
  - `v2-p2-noms-de-pays-le-pays#p-206-p-210`

Automated reduced output:

- `run-131`

Preparation notes:

- this was the first 6-unit contiguous batch in the larger-automated-scaling phase
- the carried-forward Balbec map was extended modestly with:
  - `Bloch`
  - `prince des Laumes`

Report-first reading:

- the front of the batch reads as a Saint-Loup/Bloch contrast
- `v2-p2-noms-de-pays-le-pays#p-196-p-200` gives a mixed but prestige-positive Charlus result
- the back of the batch is dominated by strong `Mme de Villeparisis` prestige gains through Guermantes revelation

Working judgment:

- treat `run-131` as a successful first proof of the new phase
- the larger batch completed cleanly and still yielded a coherent multi-unit arc
- no report-level surprise appeared that justified dropping back into routine unit inspection

## run-133 larger-scaling checkpoint

The second larger automated scaling batch has now been completed as `run-133`.

Source batch:

- `run-132` covering:
  - `v2-p2-noms-de-pays-le-pays#p-211-p-215`
  - `v2-p2-noms-de-pays-le-pays#p-216-p-220`
  - `v2-p2-noms-de-pays-le-pays#p-221-p-225`
  - `v2-p2-noms-de-pays-le-pays#p-226-p-230`
  - `v2-p2-noms-de-pays-le-pays#p-231-p-235`
  - `v2-p2-noms-de-pays-le-pays#p-236-p-240`

Automated reduced output:

- `run-133`

Report-first reading:

- this batch is centered on Charlus and the grandmother
- `la grand-mère` receives the clearest positive local inclusion signal
- Charlus oscillates across the batch rather than flattening into one direction:
  - local lift
  - then a negative stretch
  - then recovery

Working judgment:

- treat `run-133` as another successful larger-scaling batch
- the reports preserve a credible Charlus-centered instability rather than collapsing him into noise or one-note verdict
- no new failure class appears

## run-135 larger-scaling checkpoint

The third larger automated scaling batch has now been completed as `run-135`.

Source batch:

- `run-134` covering:
  - `v2-p2-noms-de-pays-le-pays#p-241-p-245`
  - `v2-p2-noms-de-pays-le-pays#p-246-p-250`
  - `v2-p2-noms-de-pays-le-pays#p-251-p-255`
  - `v2-p2-noms-de-pays-le-pays#p-256-p-260`
  - `v2-p2-noms-de-pays-le-pays#p-261-p-265`
  - `v2-p2-noms-de-pays-le-pays#p-266-p-270`

Automated reduced output:

- `run-135`

Preparation note:

- the register shifted here from enacted salon interaction into more reflective and essayistic Bloch-père material
- the carried-forward map was extended modestly with:
  - `Bloch père`

Report-first reading:

- the report surface became much sparser and more uniformly negative than in `run-131` or `run-133`
- `baron de Charlus` reads as a local loss
- `Bloch` reads as a local rhetorical loss
- `Bloch père` reads as the dominant negative figure in the reflective back half
- the lyric / quotation-fragment units `p-251-p-255` and `p-256-p-260` disappeared from the report surface entirely

Spot-check result:

- this did justify a targeted review
- the investigation showed:
  - the negative readings are text-grounded, not parser noise
  - the empty fragment units were already empty in raw model output, not erased by reduction
  - the raw Bloch-père unit is slightly richer than the reduced result, but the reduction pressure remains intelligible rather than pathological

Working judgment:

- treat `run-135` as acceptable
- it is best understood as sparse demystification-heavy material rather than a new recurring failure class
- record one caution for future scaling:
  - reflective and expository stretches are more likely to yield empty units and narrowed negative report shapes than socially enacted scene material

## run-137 through run-151 chapter-completion checkpoint

The larger automated scaling pass across the remainder of `v2-p2-noms-de-pays-le-pays` has now been carried through the end of the chapter.

Covered source batches:

- `run-136` / `run-137`:
  - `v2-p2-noms-de-pays-le-pays#p-271-p-275`
  - `v2-p2-noms-de-pays-le-pays#p-276-p-280`
  - `v2-p2-noms-de-pays-le-pays#p-281-p-285`
  - `v2-p2-noms-de-pays-le-pays#p-286-p-290`
  - `v2-p2-noms-de-pays-le-pays#p-291-p-295`
  - `v2-p2-noms-de-pays-le-pays#p-296-p-300`
- `run-138` / `run-139`:
  - `v2-p2-noms-de-pays-le-pays#p-301-p-305`
  - `v2-p2-noms-de-pays-le-pays#p-306-p-310`
  - `v2-p2-noms-de-pays-le-pays#p-311-p-315`
  - `v2-p2-noms-de-pays-le-pays#p-316-p-320`
  - `v2-p2-noms-de-pays-le-pays#p-321-p-325`
  - `v2-p2-noms-de-pays-le-pays#p-326-p-330`
- `run-140` / `run-141`:
  - `v2-p2-noms-de-pays-le-pays#p-331-p-335`
  - `v2-p2-noms-de-pays-le-pays#p-336-p-340`
  - `v2-p2-noms-de-pays-le-pays#p-341-p-345`
  - `v2-p2-noms-de-pays-le-pays#p-346-p-350`
  - `v2-p2-noms-de-pays-le-pays#p-351-p-355`
  - `v2-p2-noms-de-pays-le-pays#p-356-p-360`
- `run-142` / `run-143`:
  - `v2-p2-noms-de-pays-le-pays#p-361-p-365`
  - `v2-p2-noms-de-pays-le-pays#p-366-p-370`
  - `v2-p2-noms-de-pays-le-pays#p-371-p-375`
  - `v2-p2-noms-de-pays-le-pays#p-376-p-380`
  - `v2-p2-noms-de-pays-le-pays#p-381-p-385`
  - `v2-p2-noms-de-pays-le-pays#p-386-p-390`
- `run-144` / `run-145`:
  - `v2-p2-noms-de-pays-le-pays#p-391-p-395`
  - `v2-p2-noms-de-pays-le-pays#p-396-p-400`
  - `v2-p2-noms-de-pays-le-pays#p-401-p-405`
  - `v2-p2-noms-de-pays-le-pays#p-406-p-410`
  - `v2-p2-noms-de-pays-le-pays#p-411-p-415`
  - `v2-p2-noms-de-pays-le-pays#p-416-p-420`
- `run-146` / `run-147`:
  - `v2-p2-noms-de-pays-le-pays#p-421-p-425`
  - `v2-p2-noms-de-pays-le-pays#p-426-p-430`
  - `v2-p2-noms-de-pays-le-pays#p-431-p-435`
  - `v2-p2-noms-de-pays-le-pays#p-436-p-440`
  - `v2-p2-noms-de-pays-le-pays#p-441-p-445`
  - `v2-p2-noms-de-pays-le-pays#p-446-p-450`
- `run-148` / `run-149`:
  - `v2-p2-noms-de-pays-le-pays#p-451-p-455`
  - `v2-p2-noms-de-pays-le-pays#p-456-p-460`
  - `v2-p2-noms-de-pays-le-pays#p-461-p-465`
  - `v2-p2-noms-de-pays-le-pays#p-466-p-470`
  - `v2-p2-noms-de-pays-le-pays#p-471-p-475`
  - `v2-p2-noms-de-pays-le-pays#p-476-p-480`
- `run-150` / `run-151`:
  - `v2-p2-noms-de-pays-le-pays#p-481-p-485`
  - `v2-p2-noms-de-pays-le-pays#p-486-p-490`
  - `v2-p2-noms-de-pays-le-pays#p-491-p-492`

Preparation notes:

- the larger-scaling workflow remained stable through the rest of the chapter:
  - prepare contiguous source batch
  - automate to fresh output run
  - reduce and score
  - inspect reports first
  - inspect units only when genuinely surprised
- the carried-forward Balbec map picked up a few useful additions during this span:
  - `le directeur`
  - `Dreyfus`
  - `jeune blonde de Rivebelle`
- no new alias collision on the order of the earlier bare `princesse` problem appeared

Report-first reading:

- the chapter remained report-readable across long contiguous spans without forcing routine return to unit-by-unit validation
- the accepted larger-scaling batches preserved plausible local oscillation rather than collapsing the chapter into one-note verdicts
- recurrent patterns included:
  - positive and mixed painter / Elstir material
  - oscillating Albertine valuation
  - Bloch and Bloch-père negative or exclusion-heavy stretches
  - Villeparisis, Saint-Loup, and related social-world figures moving through intelligible gains and losses
- the chapter-end batch `run-151` was especially simple and coherent:
  - only `le directeur` appeared on the report surface
  - local loss at `p-481-p-485`
  - neutral middle unit
  - mild rebound at `p-491-p-492`

Working judgment:

- treat the completed `v2-p2-noms-de-pays-le-pays` pass as successful larger automated scaling
- no new recurring failure class emerged across the back half of the chapter
- the prompt and reducer remained usable without retuning
- remaining misses continue to look like tolerable compression or weighting artifacts rather than systematic directional inversion

Operational notes worth carrying forward:

- long silent runs remained normal all the way through `run-151`
- the right health check remained:
  - `run.json` counters
  - `raw/` file count
  - `annotations/` file count
- chapter boundaries should not trigger automatic prompt or reducer changes
- when starting the next chapter, refresh aliases lightly for new local figures rather than assuming the full Balbec-era map should be copied forward unchanged without review

Recommended next step:

1. treat `v2-p2-noms-de-pays-le-pays` as complete for the current scaling pass
2. start the opening contiguous batch of the next chapter
3. keep the same report-first larger-scaling workflow
4. intervene only if the next chapter produces a genuinely surprising recurring signal

## run-153 through run-209 ongoing `v3-p1` checkpoint

The larger automated scaling pass across `v3-p1` has now been carried from the chapter opening through `p-720`.

Covered source batches:

- `run-152` / `run-153` through `p-120`
- `run-154` / `run-155` through `p-150`
- `run-156` / `run-157` through `p-180`
- `run-158` / `run-159` through `p-210`
- `run-160` / `run-161` through `p-240`
- `run-162` / `run-163` through `p-270`
- `run-164` / `run-165` through `p-300`
- `run-166` / `run-167` through `p-330`
- `run-168` / `run-169` through `p-360`
- `run-170` / `run-171` through `p-390`
- `run-172` / `run-173` through `p-420`
- `run-174` / `run-175` through `p-450`
- `run-176` / `run-177` through `p-480`
- `run-178` / `run-179` through `p-510`
- `run-180` / `run-181` through `p-540`
- `run-182` / `run-183` through `p-570`
- `run-184` / `run-185` through `p-600`
- `run-186` / `run-187` through `p-630`
- `run-188` / `run-189` through `p-660`
- `run-190` / `run-191` through `p-690`
- `run-208` / `run-209` through `p-720`

Operational notes:

- the `wait --reduce --report` path is now the default long-run orchestration workflow
- the new waiting flow solved the recurring manual-polling problem:
  - it blocks on manifest completion
  - it chains reprocessing and report generation
  - it makes long silent runs easier to treat as normal rather than suspicious
- one earlier stalled-looking run turned out to be a dead automate process rather than an API limit issue
- after that debugging, the safe rule became:
  - use `wait` for ordinary monitoring
  - verify the process separately only when manifest/file progress truly stops

Alias and carry-forward notes:

- the earlier Balbec-region additions remained stable across the pass
- the later Guermantes-region additions also held without destabilizing the map:
  - `duchesse de Guermantes`
  - `Jupien`
- no new collision on the order of the earlier bare `princesse` issue appeared

Report-first reading:

- the chapter has remained report-readable over a long contiguous span without forcing a return to routine unit-by-unit validation
- accepted recent batches kept producing plausible local shapes rather than flattening into noise
- recurrent patterns included:
  - repeated mixed or negative `Robert de Saint-Loup` surfaces that still read as text-plausible rather than system failure
  - repeated negative or exclusion-heavy `Bloch` stretches that looked narratively grounded, even when visually dominant
  - ongoing oscillation between `Mme de Villeparisis`, `duchesse de Guermantes`, `duc de Guermantes`, and related Guermantes-world figures
  - periodic `Norpois`, `Legrandin`, `Dreyfus`, and `Odette` appearances that remained locally intelligible
- the chapter also continued to show that:
  - reflective or sparse stretches can narrow the report surface
  - but those stretches have not turned into a recurring failure class

Working judgment:

- treat the `v3-p1` pass through `p-720` as another successful large-scale transfer proof
- no recurring failure class has emerged across this long sequence
- the prompt and reducer remain usable without retuning
- the orchestration layer now also feels mature enough for longer unattended runs
- the main remaining question is strategic rather than technical:
  - whether to finish the chapter for neatness
  - or widen the aperture now that confidence in transfer is materially higher

Recommended next step:

1. update the durable state docs to reflect the completed `v3-p1` stretch through `p-720`
2. take stock at the project level rather than immediately extending the same chapter by inertia
3. if widening the aperture, choose the next terrain for comparative value rather than simple adjacency
4. keep the same report-first workflow and intervention threshold

## run-211 first widened-aperture checkpoint

The first widened-aperture batch has now been completed successfully in `v4-p2`.

Source batch:

- `run-210` covering:
  - `v4-p2#p-1-p-5`
  - `v4-p2#p-6-p-10`
  - `v4-p2#p-11-p-15`
  - `v4-p2#p-16-p-20`
  - `v4-p2#p-21-p-25`
  - `v4-p2#p-26-p-30`

Automated reduced output:

- `run-211`

Preparation notes:

- this was the first batch chosen explicitly for widened-aperture contrast rather than chapter adjacency
- `v4-p2` was selected because it offers new terrain while still overlapping with the Guermantes / Charlus world enough to test transfer rather than force a cold restart
- the opening alias refresh was light and local:
  - `princesse de Guermantes`
  - `duc de Châtellerault`
  - `M. de Vaugoubert`
  - `Mme de Vaugoubert`

Report-first reading:

- the batch completed cleanly:
  - `6/6/6`
  - no parse errors
  - no validation errors
- the report surface remained coherent in the new zone:
  - `princesse de Guermantes` opened positively
  - `duc de Châtellerault` read as a local social loss
  - `M. de Vaugoubert` came through as mixed-to-negative
  - `baron de Charlus` read as negative across two units
- nothing in the surface suggested prompt drift, reducer instability, or alias breakdown

Working judgment:

- treat `run-211` as the first successful practical proof that widening the aperture now works
- the stack transferred into `v4-p2` without requiring renewed benchmark-style caution
- the right next move is not to retreat to `v3-p1` completion by inertia
- the right next move is to continue at least a little further into `v4-p2` and see whether the widened-aperture behavior stays stable beyond the opening sample

Recommended next step:

1. prepare the next contiguous `v4-p2` batch
2. automate it with the same `wait --reduce --report` workflow
3. keep review report-first
4. inspect units only if a genuinely surprising signal appears

## run-213 through run-229 ongoing widened-aperture `v4-p2` checkpoint

The widened-aperture pass across `v4-p2` has now continued from the opening sample through `p-300`.

Covered source batches:

- `run-212` / `run-213` through `p-60`
- `run-214` / `run-215` through `p-90`
- `run-216` / `run-217` through `p-120`
- `run-218` / `run-219` through `p-150`
- `run-220` / `run-221` through `p-180`
- `run-222` / `run-223` through `p-210`
- `run-224` / `run-225` through `p-240`
- `run-226` / `run-227` through `p-270`
- `run-228` / `run-229` through `p-300`

Operational notes:

- the `wait --reduce --report` path remained effective through the entire widened-aperture continuation
- long runtimes continued to be normal rather than exceptional
- a new operational constraint did appear in the Codex environment:
  - too many completed unified exec sessions can accumulate
  - session cleanup or reuse is therefore worth doing periodically
  - this is an interface/orchestration concern, not a pipeline-quality concern

Report-first reading:

- the widened-aperture pass kept producing plausible local structures rather than collapsing into generic negativity
- the surface remained variable in a useful way:
  - some batches were socially crowded and mixed
  - some were sparse and narrowed to one or two figures
  - some were harsher negative stretches
  - others reopened into more varied social motion
- recurring figures across the pass included:
  - `baron de Charlus`
  - `princesse de Guermantes`
  - `M. de Vaugoubert`
  - `duchesse de Guermantes`
  - `Swann`
  - `Françoise`
  - `la grand-mère`
  - `le directeur`
  - `Aimé`
  - `Mme de Cambremer`
  - `princesse de Parme`
- sparse batches did occur, but they read like local narrative texture rather than annotation failure

Working judgment:

- treat the widened-aperture `v4-p2` continuation through `run-229` as successful
- the stack has now shown sustained transfer beyond the original contiguous chapter-scale zone
- no recurring failure class has emerged in the widened pass
- the main practical caution at this point is session hygiene in the Codex environment, not prompt, reducer, or alias instability

Recommended next step:

1. keep the current widened-aperture `v4-p2` pass going if more coverage is useful
2. or pause for a comparative project-level assessment, since the evidence for broader scaling is now substantial
3. if continuing, keep the same report-first workflow and close old exec sessions periodically

## run-231 and run-234 widened-aperture `v4-p2` continuation checkpoint

The widened-aperture pass across `v4-p2` has now continued from `p-300` through `p-360`.

Covered source batches:

- `run-230` / `run-231` through `p-330`
- `run-232` / `run-234` through `p-360`

Operational notes:

- both continuation batches completed cleanly under the same workflow:
  - `6/6/6`
  - no parse errors
  - no validation errors
- `run-231` had already been present as a completed automated batch and only needed the usual `wait --reduce --report` post-processing to restore the review surface cleanly
- the next local source batch was then prepared as `run-232`, continuing the same five-paragraph widened-aperture windows:
  - `v4-p2#p-331-p-335`
  - `v4-p2#p-336-p-340`
  - `v4-p2#p-341-p-345`
  - `v4-p2#p-346-p-350`
  - `v4-p2#p-351-p-355`
  - `v4-p2#p-356-p-360`
- one non-pipeline operational issue did recur:
  - a sandboxed `automate` attempt appeared to hang on its first request without writing files
  - rerunning the same batch with escalated network access succeeded immediately
  - this looks like an environment/network-permission issue rather than a prompt, reducer, or schema issue

Report-first reading:

- `run-231` remained coherent across `p-301` to `p-330`
- the surface was still dominated by plausible local negatives rather than generic collapse:
  - `Albertine`
  - `baron de Charlus`
  - `Saniette`
  - `docteur Cottard`
- `Robert de Saint-Loup` was the only positive or mixed counterweight in that slice, which read as a local textual feature rather than a scoring malfunction
- `run-234` remained equally coherent across `p-331` to `p-360`
- the next slice showed continued variation rather than monotone negativity:
  - repeated local diminishment for `M. Verdurin`
  - a clear negative turn for `baron de Charlus`
  - a sharper local loss for `marquis de Cambremer`
  - a strong exclusionary loss for `Saniette`
  - a clear positive reopening for `docteur Cottard`
- the three lenses stayed aligned on the main directional shapes even when magnitude changed:
  - `docteur Cottard` remained positive across all three
  - `Saniette` remained sharply negative across all three
  - `M. Verdurin`, `baron de Charlus`, and `marquis de Cambremer` stayed negative across all three
- nothing in either continuation batch suggested a new recurring alias problem, parse breakdown, or reducer drift

Working judgment:

- treat `run-231` and `run-234` as accepted widened-aperture checkpoints
- `v4-p2` now remains accepted through `p-360` of `450`
- the widened-aperture mode continues to produce report-readable output over a long contiguous stretch without reopening prompt or reducer work
- the main practical caution remains execution environment behavior during API calls, not annotation quality

Recommended next step:

1. continue the widened-aperture `v4-p2` pass to the end of the chapter unless there is a stronger reason to pause for corpus-level comparison now
2. keep using `wait --reduce --report` as the default review path
3. if a local `automate` run again shows no manifest or file progress, treat escalated network rerun as the first practical fix

## run-236 widened-aperture `v4-p2` continuation checkpoint

The widened-aperture pass across `v4-p2` has now continued through `p-390`.

Covered source batch:

- `run-235` / `run-236` through `p-390`

Operational notes:

- the source batch was prepared as:
  - `v4-p2#p-361-p-365`
  - `v4-p2#p-366-p-370`
  - `v4-p2#p-371-p-375`
  - `v4-p2#p-376-p-380`
  - `v4-p2#p-381-p-385`
  - `v4-p2#p-386-p-390`
- the automated run completed cleanly:
  - `6/6/6`
  - no parse errors
  - no validation errors
- the run was operationally slow enough to look stalled at first, but manifest and file checks eventually showed steady progress:
  - it moved from `0/6` to `3/6`, then `4/6`, then `5/6`, then completion
  - this reinforces the existing rule:
    - prefer manifest/file checks over visual silence
    - intervene only when progress genuinely stops

Report-first reading:

- this batch was notably narrower than the immediately previous one
- only two characters surfaced in the scored reports:
  - `baron de Charlus`
  - `la grand-mère`
- that narrowness did not read like parser collapse or alias failure
- instead it read like a locally concentrated stretch of the narrative:
  - `baron de Charlus` first appeared in a positive local elevation
  - then moved through several negative units in succession
  - the main negative forms were narrated diminishment and discredit association
- the three lenses stayed directionally aligned throughout:
  - all three preserved the initial Charlus win in `v4-p2#p-361-p-365`
  - all three read the later Charlus-centered windows as losses
  - `la grand-mère` stayed neutral in the one narrow unit where she surfaced
- the strongest loss signal in the batch landed in the Charlus material around:
  - `v4-p2#p-376-p-380`
  - and, in prestige terms, also `v4-p2#p-381-p-385`

Working judgment:

- treat `run-236` as an accepted checkpoint
- `v4-p2` now remains accepted through `p-390` of `450`
- the widened-aperture workflow continues to hold even when the report surface narrows dramatically around one focal figure
- no new recurring failure class is visible here

Recommended next step:

1. continue the widened-aperture pass toward the end of `v4-p2`
2. keep the same report-first review rule
3. do not overreact to narrow character surfaces when the three lenses still agree and the outputs remain text-plausible

## run-238 widened-aperture `v4-p2` continuation checkpoint

The widened-aperture pass across `v4-p2` has now continued through `p-420`.

Covered source batch:

- `run-237` / `run-238` through `p-420`

Operational notes:

- the source batch was prepared as:
  - `v4-p2#p-391-p-395`
  - `v4-p2#p-396-p-400`
  - `v4-p2#p-401-p-405`
  - `v4-p2#p-406-p-410`
  - `v4-p2#p-411-p-415`
  - `v4-p2#p-416-p-420`
- the automated run completed cleanly:
  - `6/6/6`
  - no parse errors
  - no validation errors
- startup was again slow enough to look suspicious at first, but manifest/file checks showed eventual forward motion:
  - `0/6`
  - then `1/6`
  - then `2/6`
  - then `3/6`
  - then `4/6`
  - then `5/6`
  - then completion
- this batch reinforces the same operational rule as the previous long run:
  - do not infer failure from silence alone
  - verify progress from `run.json` and output files before intervening

Report-first reading:

- the batch reopened into a more varied local surface than the narrow Charlus-heavy stretch in `run-236`
- recurring figures included:
  - `baron de Charlus`
  - `Albertine`
  - `Morel`
  - `Mme Verdurin`
- the main local shapes remained interpretable:
  - an early Charlus loss
  - a positive Albertine unit
  - a sharp Morel loss
  - a strong Charlus reopening in `v4-p2#p-406-p-410`
  - a mixed Charlus unit in `v4-p2#p-411-p-415`
  - another Charlus loss at the end of the batch
- the three lenses diverged in degree but not in a troubling way:
  - `Albertine` remained positive across all three
  - `Morel` remained negative across all three
  - the Charlus sequence remained legible as oscillating rather than incoherent
  - the mixed Charlus unit stayed mixed or near-neutral rather than flipping implausibly
- nothing here suggests parser degradation, alias failure, or reducer drift

Working judgment:

- treat `run-238` as an accepted checkpoint
- `v4-p2` now remains accepted through `p-420` of `450`
- the widened-aperture workflow continues to hold on mixed and swinging local material, not just on narrow or monotone stretches
- no new recurring failure class is visible

Recommended next step:

1. continue the final `v4-p2` stretch through the end of the chapter
2. keep using the same report-first review rule
3. once the chapter is complete, decide whether the next move should be another widened-aperture comparison zone or a corpus-level assessment

## run-240 final widened-aperture `v4-p2` checkpoint

The widened-aperture pass across `v4-p2` has now reached the end of the chapter.

Covered source batch:

- `run-239` / `run-240` through `p-450`

Operational notes:

- the final source batch was prepared as:
  - `v4-p2#p-421-p-425`
  - `v4-p2#p-426-p-430`
  - `v4-p2#p-431-p-435`
  - `v4-p2#p-436-p-440`
  - `v4-p2#p-441-p-445`
  - `v4-p2#p-446-p-450`
- the automated run completed cleanly:
  - `6/6/6`
  - no parse errors
  - no validation errors
- by the time it was checked, the run had already finished, so the only remaining task was the normal `wait --reduce --report` pass

Report-first reading:

- the final stretch was narrower and cleaner than some of the more socially crowded middle batches
- the surface still remained varied enough to be useful:
  - `baron de Charlus` opened with a positive unit
  - `prince de Guermantes` read as a local loss
  - `le directeur` read as a local loss
  - `marquis de Cambremer` took the strongest negative signal in the batch
  - `Bloch` read as a local loss
  - `la mère du narrateur` closed the chapter slice with a local emotional loss
- the three lenses stayed directionally aligned:
  - Charlus stayed positive in the opening unit across all three
  - the later figures remained negative across all three
  - no implausible inversion or unexplained divergence appeared
- the batch therefore reads as a coherent chapter ending rather than as a degraded tail section

Working judgment:

- treat `run-240` as an accepted checkpoint
- treat `v4-p2` as a completed widened-aperture pass through `p-450`
- across the full `v4-p2` sequence, no new recurring failure class emerged
- the main lesson from this chapter is now strategic rather than technical:
  - the current stack appears strong enough to keep scaling
  - the better next question is where to widen next and what cross-run analysis to begin

Recommended next step:

1. pause for a comparative assessment across the accepted `v3-p1` and completed `v4-p2` material
2. decide whether to:
   - widen into another contrasting chapter zone
   - or start a corpus-level analysis pass over the accepted exploratory corpus
3. preserve the same report-first intervention threshold if scaling continues

## corpus-review-001 accepted corpus sanity checkpoint

The first corpus-level sanity review has now been completed over the accepted exploratory corpus.

Reviewed corpus:

- accepted `v3-p1` scaling through `run-209`
- completed `v4-p2` widened-aperture scaling through `run-240`

Corpus size:

- `36` accepted runs
- `186` declared units
- `186` valid annotations

Headline findings:

- aggregate polarity remained negative-leaning but not pathologically one-note:
  - positive events: `64`
  - negative events: `114`
  - mixed events: `2`
- cross-lens disagreement stayed low:
  - label disagreement rate: `0.041`
  - direction disagreement rate: `0.031`
  - positive-versus-negative sign flips: `0`
- extreme positive and negative units still looked text-plausible rather than randomly targeted
- the main watchpoint surfaced at the corpus level was focal narrowness:
  - many runs average close to one scored character per scored unit

What mattered most in the disagreement reading:

- disagreements were mostly threshold or mixedness cases
- they did not take the form:
  - one lens reading a clear win
  - another lens reading a clear loss
- representative disagreement cases included:
  - `M. de Vaugoubert`
  - `Robert de Saint-Loup`
  - `baron de Charlus`
  - `la grand-mère`
- these looked like bounded mixed cases rather than a systemic inversion problem

Working judgment:

- the accepted corpus passes a first sanity review
- no hidden large-scale distortion became visible only after aggregation
- the current stack still looks usable enough to continue the automation-readiness checklist
- the main remaining concern is not corpus corruption but whether future terrain will stress the current tendency toward narrow focal surfaces

Recommended next step:

1. mark the corpus sanity proof as complete
2. move to contrastive terrain checks rather than repeating more of the same scaling
3. keep focal narrowness in mind as an explicit watchpoint when choosing the next zones

## contrastive-zone shortlist checkpoint

The project has now chosen a specific shortlist of contrasting terrain checks for the next validation phase.

Chosen shortlist:

- `v5` (`La Prisonnière`)
- `v7-p2-m-de-charlus-pendant-la-guerre`
- `v6-p1` or `v6-p2` (`Albertine disparue`)
- one Swann-side bridge zone:
  - `v2-p1-autour-de-mme-swann`
  - or `v1-p2-un-amour-de-swann`

Why this set:

- the accepted corpus already covers aristocratic and salon-heavy public dynamics well
- the remaining uncertainty is not whether those zones still work
- the remaining uncertainty is whether the stack generalizes when:
  - the social field becomes more intimate
  - the historical-social context changes
  - the material becomes more reflective or absence-driven
  - the relationship network changes while remaining socially legible

Working order:

1. `v5`
2. `v7-p2-m-de-charlus-pendant-la-guerre`
3. `v6-p1` or `v6-p2`
4. `v2-p1-autour-de-mme-swann` or `v1-p2-un-amour-de-swann`

Why `v5` comes first:

- it is the strongest immediate test of intimate and psychological narrowing
- it should stress the focal narrowness watchpoint in a new way
- it is therefore the highest-value first contrastive zone

Recommended next step:

1. prepare the opening `v5` source batch
2. keep the normal widened-aperture five-paragraph window pattern
3. read the three report lenses first before deciding whether the new terrain is opening cleanly

## run-242 first v5 contrastive opening checkpoint

`run-242` automated the prepared `v5` opening source batch (`run-241`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch opened cleanly in intimate/domestic terrain rather than collapsing
- the reported cast was narrow but plausible for the material:
  - `Albertine`
  - `Françoise`
- `Albertine` showed a coherent oscillating profile across the opening:
  - `v5#p-1-p-5`: strong positive opening
  - `v5#p-6-p-10`: negative turn through `discredit_association`
  - `v5#p-16-p-20`: sharper negative turn through `snub`
  - `v5#p-21-p-25`: positive reopening
- `Françoise` surfaced as socially and rhetorically strong in the central windows
- the lens split remained interpretable rather than noisy:
  - local: `Françoise` strongest overall
  - prestige: `Françoise` clearly strongest, `Albertine` only mildly net positive
  - inclusion: `Albertine` strongest overall

Judgment:

- this should count as the first successful `v5` contrastive checkpoint
- it does not remove focal narrowness as a watchpoint
- but it does show that focal narrowness can remain text-plausible in terrain very different from the previously accepted salon-heavy corpus

Recommended next step:

1. continue `v5` contiguously for another opening batch if more evidence is desired from the same intimate zone
2. keep the existing prompt and reduction stack unchanged unless a later contrastive run produces an actually surprising report signal

## run-244 second v5 contrastive checkpoint

`run-244` automated the second contiguous `v5` source batch (`run-243`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch was narrower and more negative than `run-242`, but not in a way that suggested stack failure
- the surfaced cast remained coherent for the terrain:
  - `Albertine`
  - `Andrée`
  - `duchesse de Guermantes`
- `Albertine` registered as a sustained loss across four windows:
  - `v5#p-31-p-35`
  - `v5#p-41-p-45`
  - `v5#p-46-p-50`
  - `v5#p-51-p-55`
- the dominant event shape was repeated `narrated_diminishment`
- the sharpest negative point landed in `v5#p-46-p-50`
- `Andrée` opened positively in `v5#p-36-p-40`
- `duchesse de Guermantes` opened positively in `v5#p-56-p-60`
- the three lenses stayed tightly aligned:
  - all three scored `Albertine` as net negative across the batch
  - all three preserved the positive `Andrée` window
  - all three preserved the positive Guermantes reopening at the end

Judgment:

- this should count as the second successful `v5` contrastive checkpoint
- it strengthens the case that the current stack can survive a long psychologically constricted and locally monotone patch without losing interpretability
- the run therefore increases confidence that intimate/domestic terrain is not itself a hidden failure mode

Recommended next step:

1. continue `v5` contiguously into the next opening-social expansion rather than stopping at the first negative interior stretch
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-246 third v5 contrastive checkpoint

`run-246` automated the third contiguous `v5` source batch (`run-245`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch widened cleanly back into a more aristocratic and socially explicit register
- the surfaced cast was:
  - `duchesse de Guermantes`
  - `duc de Guermantes`
  - `baron de Charlus`
  - `Morel`
- `duchesse de Guermantes` opened positively in:
  - `v5#p-61-p-65`
  - `v5#p-66-p-70`
- then the batch turned into a coherent negative sequence:
  - `duc de Guermantes` negative in `v5#p-71-p-75`
  - `baron de Charlus` negative in `v5#p-76-p-80`
  - `Morel` negative in `v5#p-81-p-85`
  - `Morel` negative again in `v5#p-86-p-90`
- the three lenses stayed tightly aligned on all of this:
  - positive Guermantes opening across all three
  - negative `duc de Guermantes`, `baron de Charlus`, and `Morel` sequence across all three
  - no noisy lens disagreement and no implausible cast surfacing

Judgment:

- this should count as the third successful `v5` contrastive checkpoint
- it increases confidence that the current stack can move from intimate and psychologically narrow material back into denser Guermantes/Charlus/Morel terrain without losing interpretability
- this matters for automation readiness because it shows that the stack is not merely surviving one narrow mode of `v5`, but transitioning coherently across local terrain changes inside the same volume

Recommended next step:

1. continue `v5` contiguously into the next Albertine/Andree suspicion and sleep material
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-248 fourth v5 contrastive checkpoint

`run-248` automated the fourth contiguous `v5` source batch (`run-247`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch narrowed back to the intimate Albertine/Andree field, which fit the suspicion and sleep material
- the surfaced cast was:
  - `Albertine`
  - `Andrée`
- `Albertine` was not monotone in this batch:
  - roughly neutral in `v5#p-91-p-95`
  - positive in `v5#p-101-p-105`
  - sharply negative in `v5#p-106-p-110`
  - negative again in `v5#p-111-p-115`
  - mixed or near-neutral in `v5#p-116-p-120`, depending on lens
- `Andrée` appeared once, in `v5#p-96-p-100`, and landed negative across all three lenses
- the three lenses stayed broadly aligned on the main shape:
  - Albertine positive at `v5#p-101-p-105`
  - Albertine strongly negative at `v5#p-106-p-110`
  - Andree negative at `v5#p-96-p-100`
  - only mild edge variation where one lens called a unit mixed or neutral while another flattened it further toward neutral

Judgment:

- this should count as the fourth successful `v5` contrastive checkpoint
- it increases confidence that the stack can handle a genuinely unstable local field of suspicion, concealment, possession, aesthetic elevation, and diminishment without obvious prompt or reducer drift
- this is useful automation-readiness evidence because it is neither the socially broader Guermantes material nor a simple monotone Albertine-negative trough, but a more internally volatile local patch

Recommended next step:

1. continue `v5` contiguously into the next jealousy-memory material rather than stopping at this mixed interior stretch
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-250 fifth v5 contrastive checkpoint

`run-250` automated the fifth contiguous `v5` source batch (`run-249`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch stayed very narrow and centered on the jealousy-memory field
- the surfaced cast was:
  - `Albertine`
  - `Aimé`
- `Albertine` carried almost the entire batch:
  - near-neutral in `v5#p-121-p-125`
  - negative in `v5#p-126-p-130`
  - strongly positive in `v5#p-131-p-135`
  - strongly negative in `v5#p-136-p-140`
  - neutral in `v5#p-141-p-145`
  - negative again in `v5#p-146-p-150`
- `Aimé` surfaced only as a neutral presence in the final window
- the three lenses stayed aligned on the main shape:
  - same large Albertine oscillation across all three
  - positive spike in `v5#p-131-p-135`
  - strongest negative in `v5#p-136-p-140`
  - `Aimé` flattened to neutral rather than being forced into a noisy polarity

Judgment:

- this should count as the fifth successful `v5` contrastive checkpoint
- it increases confidence that the current stack can sustain a very narrow jealousy-memory loop without drifting into noise or inventing cast breadth where the text does not warrant it
- this is useful automation-readiness evidence because it shows the stack can remain interpretable even when the terrain is dominated by one figure's oscillating value rather than by broader social motion

Recommended next step:

1. continue `v5` contiguously into the Verdurin / Andree / telephone material
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-252 sixth v5 contrastive checkpoint

`run-252` automated the sixth contiguous `v5` source batch (`run-251`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the batch was notably narrower and more monotone than `run-250`
- the surfaced cast was:
  - `Albertine`
  - `Françoise`
- `Albertine` carried five of the six windows and was negative almost throughout:
  - negative in `v5#p-151-p-155`
  - negative in `v5#p-156-p-160`
  - slight softening only in `v5#p-161-p-165`, which landed near-neutral or mixed depending on lens
  - negative in `v5#p-166-p-170`
  - negative in `v5#p-171-p-175`
- `Françoise` appeared in `v5#p-176-p-180` and was negative there
- the three lenses stayed tightly aligned:
  - `Albertine` negative across almost the whole batch
  - `Françoise` negative in the final window
  - only the `v5#p-161-p-165` window softened into neutral or mixed

Judgment:

- this should count as the sixth successful `v5` contrastive checkpoint
- it reinforces the main watchpoint without turning it into evidence of failure:
  - even when the prose begins to broaden toward other actors, the reduced surface can remain locked on the narrator's Albertine-centered evaluative field
- that is still useful automation-readiness evidence because the stack remained coherent and non-noisy even under this continued focal narrowing

Recommended next step:

1. continue `v5` contiguously into the next Verdurin-plan and street-sound material to see whether the surface reopens
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-254 seventh v5 contrastive checkpoint

`run-254` automated the seventh contiguous `v5` source batch (`run-253`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this was the first recent `v5` batch that clearly reopened after the narrow monotone trough in `run-252`
- the surfaced cast was:
  - `Andrée`
  - `Albertine`
- `Andrée` appeared in `v5#p-181-p-185` and landed negative
- `Albertine` then moved through a more articulated sequence:
  - negative in `v5#p-186-p-190`
  - soft negative or mixed in `v5#p-191-p-195`
  - mixed or near-neutral in `v5#p-196-p-200`
  - positive in `v5#p-201-p-205`
  - positive again in `v5#p-206-p-210`
- the three lenses stayed aligned on the broad shape:
  - `Andrée` negative across all three
  - early Albertine negative across all three
  - late Albertine positive across all three
  - only mild mid-batch variation in how quickly the reopening was scored

Judgment:

- this should count as the seventh successful `v5` contrastive checkpoint
- it increases confidence that the current stack is not trapped in a single repeated Albertine-negative mode
- this is useful automation-readiness evidence because the surface reopened on its own, without any prompt or reducer intervention

Recommended next step:

1. continue `v5` contiguously into the next Albertine / household / Bergotte material
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-256 eighth v5 contrastive checkpoint

`run-256` automated the eighth contiguous `v5` source batch (`run-255`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch narrowed again after the reopening in `run-254`
- the surfaced field was mostly just:
  - `Albertine`
  - `la mère du narrateur` once, neutrally, as collateral presence
- `v5#p-211-p-215` produced no scored character outcome
- the five scored Albertine units formed a mostly negative sequence:
  - loss in `v5#p-216-p-220`
  - loss in `v5#p-221-p-225`
  - mixed or near-neutral in `v5#p-226-p-230`
  - loss in `v5#p-231-p-235`
  - strong loss in `v5#p-236-p-240`
- the three lenses stayed aligned on the broad shape:
  - `local` read four losses plus one mixed unit
  - `prestige` softened `v5#p-226-p-230` to near-neutral because rhetorical elevation partly offset local diminishment
  - `inclusion` read all five Albertine units as losses and gave the deepest troughs to `v5#p-231-p-235` and `v5#p-236-p-240`
- `v5#p-226-p-230` was the useful test case in the batch:
  - Albertine was locally diminished by the narrator's control over her movement
  - but she was also elevated in rhetorical and aesthetic standing
  - the stack preserved that mixed structure instead of flattening it
- `v5#p-236-p-240` stayed sharply negative but also carried an explicit ambiguity flag because the narrator foregrounds the unreliability of his own jealousy and memory

Judgment:

- this should count as the eighth successful `v5` contrastive checkpoint
- it increases confidence that the current prompt and reduction stack can stay coherent even when the surface narrows again after a temporary reopening
- the batch did not expose a new failure mode; if anything, it strengthened confidence that the stack can preserve mixed local structure inside a jealousy-saturated zone

Recommended next step:

1. continue `v5` contiguously into the next Bergotte / household / Albertine material
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-258 ninth v5 contrastive checkpoint

`run-258` automated the ninth contiguous `v5` source batch (`run-257`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch widened again after the narrow `run-256` field
- the surfaced cast was:
  - `Françoise`
  - `Morel`
  - `Albertine`
  - `Bergotte`
- the per-unit sequence was strongly differentiated:
  - `Françoise` negative in `v5#p-241-p-245`
  - `Morel` negative in `v5#p-246-p-250`
  - `Albertine` negative in `v5#p-251-p-255`
  - `Albertine` more sharply negative in `v5#p-256-p-260`
  - `Albertine` negative again in `v5#p-261-p-265`
  - `Bergotte` clearly positive in `v5#p-266-p-270`
- the three lenses stayed tightly aligned on the broad shape:
  - all three kept `Françoise` negative
  - all three kept `Morel` negative
  - all three kept the whole three-unit `Albertine` block negative
  - all three saw `Bergotte` as a clean positive close
- `v5#p-251-p-255` carried one ambiguity flag, but not enough to change the broad reading

Judgment:

- this should count as the ninth successful `v5` contrastive checkpoint
- it increases confidence that the current stack is not getting trapped in a single repetitive Albertine-jealousy mode
- this is useful automation-readiness evidence because the surface widened on its own, with no prompt or reducer intervention

Recommended next step:

1. continue `v5` contiguously into the next Bergotte / Albertine / household material
2. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-260 tenth v5 contrastive checkpoint

`run-260` automated the tenth contiguous `v5` source batch (`run-259`) and also completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch stayed broad rather than collapsing back into a single-character field
- the surfaced cast was:
  - `Albertine`
  - `Morel`
  - `Swann`
  - `baron de Charlus`
- the per-unit sequence was strongly differentiated:
  - `Albertine` negative in `v5#p-271-p-275`
  - `Morel` negative in `v5#p-276-p-280`
  - `Swann` clearly positive in `v5#p-281-p-285`
  - `baron de Charlus` negative in `v5#p-286-p-290`
  - `baron de Charlus` mixed in `v5#p-291-p-295`
  - `baron de Charlus` more sharply negative in `v5#p-296-p-300`
- the three lenses stayed aligned on the broad shape:
  - all three kept `Albertine` negative
  - all three kept `Morel` negative
  - all three saw `Swann` as a clear positive unit
  - all three read the three-unit `Charlus` sequence as negative, mixed, then negative
- `v5#p-291-p-295` was the useful unit here:
  - `Charlus` was locally diminished
  - but he also received a rhetorical elevation signal
  - the stack preserved that mixed shape instead of flattening it

Judgment:

- this should count as the tenth successful `v5` contrastive checkpoint
- it further increases confidence that the current stack is not getting trapped in a single narrow local mode inside `v5`
- `v5` now looks like strong terrain-transfer evidence rather than the default place to keep accumulating more proof by inertia

Recommended next step:

1. log `v5` as strong terrain-transfer evidence and stop treating it as the default next chapter
2. finish terrain-transfer proof in a deliberately contrasting zone
3. keep the prompt and reduction stack unchanged unless a later contrastive run produces a genuinely surprising signal

## run-262 first v7-p2 contrasting-zone checkpoint

`run-262` automated the opening source batch for `v7-p2-m-de-charlus-pendant-la-guerre` (`run-261`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this was the first batch in a deliberately contrasting zone after the extended `v5` evidence run
- the surfaced cast was:
  - `Elstir`
  - `Mme Verdurin`
  - `Robert de Saint-Loup`
  - `Françoise`
- the per-unit sequence was strongly differentiated:
  - `Elstir` positive in `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-5`
  - `Mme Verdurin` positive in `v7-p2-m-de-charlus-pendant-la-guerre#p-6-p-10`
  - `Mme Verdurin` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-11-p-15`
  - `Robert de Saint-Loup` positive in `v7-p2-m-de-charlus-pendant-la-guerre#p-16-p-20`
  - `Françoise` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-21-p-25`
  - `Robert de Saint-Loup` positive again in `v7-p2-m-de-charlus-pendant-la-guerre#p-26-p-30`
- the three lenses stayed aligned on the broad shape:
  - `Elstir` positive across all three
  - `Mme Verdurin` oscillating from positive to negative across all three
  - `Robert de Saint-Loup` positive in both units across all three
  - `Françoise` negative across all three

Judgment:

- this should count as the first successful checkpoint in `v7-p2-m-de-charlus-pendant-la-guerre`
- it is strong early evidence that the stack transfers into wartime and socially reconfigured terrain without a new recurring failure class
- the zone remains worth continuing contiguously while the reports stay this readable

Recommended next step:

1. continue `v7-p2-m-de-charlus-pendant-la-guerre` contiguously into the next batch
2. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-264 second v7-p2 contrasting-zone checkpoint

`run-264` automated the second source batch for `v7-p2-m-de-charlus-pendant-la-guerre` (`run-263`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the zone remained readable rather than collapsing into a single wartime prestige signal
- the surfaced cast was:
  - `Robert de Saint-Loup`
  - `Mme Verdurin`
  - `Brichot`
  - `Morel`
  - `baron de Charlus`
- the per-unit sequence was strongly differentiated:
  - `Robert de Saint-Loup` positive in `v7-p2-m-de-charlus-pendant-la-guerre#p-31-p-35`
  - `Mme Verdurin` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-36-p-40`
  - `Mme Verdurin` more sharply negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-45`
  - `Brichot` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-46-p-50`
  - `Mme Verdurin` neutral collateral in `v7-p2-m-de-charlus-pendant-la-guerre#p-46-p-50`
  - `Morel` mixed in `v7-p2-m-de-charlus-pendant-la-guerre#p-51-p-55`
  - `baron de Charlus` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-51-p-55`
  - `baron de Charlus` negative again in `v7-p2-m-de-charlus-pendant-la-guerre#p-56-p-60`
- the three lenses stayed aligned on the broad shape:
  - `Robert de Saint-Loup` positive across all three
  - both `Mme Verdurin` units negative across all three
  - `Brichot` negative across all three
  - both `Charlus` units negative across all three
  - `Morel` mixed across all three

Judgment:

- this should count as the second successful checkpoint in `v7-p2-m-de-charlus-pendant-la-guerre`
- it strengthens the early transfer evidence from `run-262`
- the zone remains worth continuing contiguously while the reports stay this readable

Recommended next step:

1. continue `v7-p2-m-de-charlus-pendant-la-guerre` into the next batch
2. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-266 third v7-p2 contrasting-zone checkpoint

`run-266` automated the closing source batch for `v7-p2-m-de-charlus-pendant-la-guerre` (`run-265`) and completed cleanly.

Mechanical result:

- `4/4` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- the closing batch was narrower than the previous one, but still clearly readable as terrain rather than system collapse
- the surfaced cast was:
  - `baron de Charlus`
  - `M. de Marsantes`
- the per-unit sequence was:
  - `baron de Charlus` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-61-p-65`
  - `baron de Charlus` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-66-p-70`
  - `baron de Charlus` negative in `v7-p2-m-de-charlus-pendant-la-guerre#p-71-p-75`
  - `M. de Marsantes` positive in `v7-p2-m-de-charlus-pendant-la-guerre#p-76-p-80`
- the three lenses stayed aligned on the broad shape:
  - all three kept the entire three-unit `Charlus` sequence negative
  - all three saw `M. de Marsantes` as a clean positive closing unit
  - no instability or sign inversion appeared at the chapter end

Judgment:

- this should count as the third successful checkpoint in `v7-p2-m-de-charlus-pendant-la-guerre`
- together with `run-262` and `run-264`, it is enough to treat `v7-p2-m-de-charlus-pendant-la-guerre` as a completed successful contrasting zone
- the chapter did not expose a new recurring failure class

Recommended next step:

1. mark `v7-p2-m-de-charlus-pendant-la-guerre` complete as contrasting-zone evidence
2. decide whether the next terrain-transfer move should be `v6-p1` / `v6-p2` or the Swann-side bridge zone
3. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-268 first v6-p1 contrasting-zone checkpoint

`run-268` automated the opening source batch for `v6-p1` (`run-267`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this was the first batch in the grief-and-absence terrain of `Albertine disparue`
- the surfaced field was narrow but not empty:
  - `Albertine`
  - `Bloch`
  - `Robert de Saint-Loup`
- the per-unit sequence was:
  - `Albertine` positive in `v6-p1#p-1-p-5`
  - `Albertine` positive in `v6-p1#p-6-p-10`
  - `Albertine` positive in `v6-p1#p-11-p-15`
  - `Bloch` negative in `v6-p1#p-16-p-20`
  - `Robert de Saint-Loup` negative in `v6-p1#p-21-p-25`
  - `Albertine` positive again in `v6-p1#p-26-p-30`
- the three lenses stayed aligned on the broad shape:
  - all three saw the four `Albertine` units as positive
  - all three kept `Bloch` negative
  - all three kept `Robert de Saint-Loup` negative
- the important interpretive point is that `Albertine` appears to be scoring positive through emotional dominance in the narrator's field, not through ordinary prestige or social advantage

Judgment:

- this should count as the first successful checkpoint in `v6-p1`
- the batch did not collapse into emptiness or noise, which is the first thing this terrain had to prove
- this is a delicate but defensible reading rather than an obvious failure
- the next batch is high-information because it will show whether this grief-driven `Albertine` pattern remains text-plausible or starts becoming misleading at the report level

Recommended next step:

1. continue `v6-p1` into the next batch
2. watch whether the `Albertine`-positive pattern remains literarily plausible
3. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-270 second v6-p1 contrasting-zone checkpoint

`run-270` automated the second source batch for `v6-p1` (`run-269`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch was the key follow-up test for whether the opening `Albertine`-positive reading in grief would become a rigid pattern
- the surfaced field remained narrow but became more differentiated:
  - `Albertine`
  - `Robert de Saint-Loup`
- the per-unit sequence was:
  - `Albertine` negative in `v6-p1#p-31-p-35`
  - no scored character in `v6-p1#p-36-p-40`
  - `Albertine` negative in `v6-p1#p-41-p-45`
  - `Albertine` positive in `v6-p1#p-46-p-50`
  - `Albertine` negative in `v6-p1#p-51-p-55`
  - `Robert de Saint-Loup` negative in `v6-p1#p-56-p-60`
- the three lenses stayed aligned on the broad shape:
  - all three saw the three negative `Albertine` units as negative
  - all three saw the one positive `Albertine` unit as positive
  - all three kept `Robert de Saint-Loup` negative
  - no cross-lens inversion or mixed-unit noise appeared

Judgment:

- this should count as the second successful checkpoint in `v6-p1`
- it materially improves confidence in the zone
- the opening `Albertine`-positive result now looks like a local grief-terrain reading rather than a fixed system distortion
- `v6-p1` remains worth continuing contiguously while the reports stay this readable

Recommended next step:

1. continue `v6-p1` into the next batch
2. keep watching whether the narrow field remains text-plausible rather than misleading
3. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-272 third v6-p1 contrasting-zone checkpoint

`run-272` automated the third source batch for `v6-p1` (`run-271`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch moved deeper into memorial and retrospective aftermath without losing report readability
- the surfaced cast was:
  - `Françoise`
  - `Aimé`
  - `Albertine`
- the per-unit sequence was:
  - `Françoise` negative in `v6-p1#p-61-p-65`
  - `Albertine` negative in `v6-p1#p-66-p-70`
  - `Aimé` positive in `v6-p1#p-71-p-75`
  - `Albertine` positive in `v6-p1#p-76-p-80`
  - `Albertine` negative in `v6-p1#p-81-p-85`
  - `Albertine` mixed in `v6-p1#p-86-p-90`
- the three lenses stayed aligned on the broad shape:
  - `Françoise` negative across all three
  - `Aimé` positive across all three
  - `Albertine` alternating across loss, win, and mixed outcomes across all three
  - no cross-lens inversion or mixed-unit noise appeared

Judgment:

- this should count as the third successful checkpoint in `v6-p1`
- it materially strengthens the case that `v6-p1` is genuine contrasting-zone evidence rather than a fragile edge case
- the zone is now showing stable local oscillation rather than a single rigid reading rule

Recommended next step:

1. continue `v6-p1` into the next batch
2. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-274 fourth v6-p1 contrasting-zone checkpoint

`run-274` automated the fourth source batch for `v6-p1` (`run-273`) and completed cleanly.

Mechanical result:

- `6/6` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this batch pushed deeper into inquiry, retrospection, and self-contradictory aftermath without losing report readability
- the surfaced cast was:
  - `Albertine`
  - `Andrée`
  - `Aimé`
- the per-unit sequence was:
  - `Albertine` negative in `v6-p1#p-91-p-95`
  - `Albertine` negative in `v6-p1#p-96-p-100`, with `Aimé` mixed collateral
  - `Albertine` mixed or near-neutral in `v6-p1#p-101-p-105`
  - `Albertine` negative in `v6-p1#p-106-p-110`
  - `Andrée` positive in `v6-p1#p-111-p-115`
  - `Albertine` negative and `Andrée` mixed in `v6-p1#p-116-p-120`
- the three lenses stayed aligned on the broad shape:
  - `Albertine` mostly negative across the span
  - `Andrée` positive once and mixed once across all three lenses
  - `Aimé` mixed collateral rather than overcommitted
  - no cross-lens inversion or runaway noise appeared

Judgment:

- this should count as the fourth successful checkpoint in `v6-p1`
- it materially strengthens the case that `v6-p1` is successful contrasting-zone evidence
- taken together with the earlier `v6-p1` batches, terrain-transfer is now complete enough to stop chapter scaling and move to the stress pack

Recommended next step:

1. stop chapter scaling
2. begin the adverse-case stress pack
3. keep the prompt and reduction stack unchanged unless a later run produces a genuinely surprising signal

## run-276 first adverse-case stress-pack checkpoint

`run-276` automated the first curated adverse-case stress pack from `run-275` and completed cleanly.

Mechanical result:

- `20/20` completed
- `0` parse errors
- `0` validation errors

Interpretive result:

- this was the first explicit proof that the stack can face deliberately difficult passage types without producing a new structural failure class
- the pack covered:
  - irony and narrator-stance drift
  - mixed prestige and inclusion signals
  - title-heavy and collision-risk material
  - sparse reflective and grief-driven terrain
  - focal blur and collateral-figure pressure
- the report surface remained coherent rather than merely schema-valid:
  - `Swann` ranged across positive, negative, and neutral local movement depending on the specific Combray or salon passage
  - `Albertine` alternated across win, loss, and mixed outcomes in `v6-p1` rather than collapsing into a rigid positive or negative rule
  - `baron de Charlus` remained consistently negative in the wartime selections
  - `Morel` remained collateral and mixed rather than hijacking the `Charlus` material
  - `Odette` surfaced as a clear positive in the `v2-p1-autour-de-mme-swann` social-ascent passage
- no cross-lens inversion, alias collapse, or broad report-level distortion appeared

Judgment:

- this should count as a successful first adverse-case proof
- the stress pack did not reveal a hidden failure mode that would justify retuning the prompt, reducer, or alias discipline before larger automation
- the project should now treat the interpretive validation phase as substantially complete

Recommended next step:

1. record the stress pack as completed evidence in the strategic docs
2. begin the production-style dry run
3. use that dry run to decide whether full-corpus automation is now justified

## run-278 / run-280 / run-282 first production-style dry-run checkpoint

`run-278`, `run-280`, and `run-282` were launched as the first explicit unattended multi-batch dry run.

Dry-run shape:

- source runs:
  - `run-277`
  - `run-279`
  - `run-281`
- automated output runs:
  - `run-278`
  - `run-280`
  - `run-282`
- orchestration mode:
  - unattended chained `automate`
  - then `wait --reduce --report`
  - repeated across three consecutive batches in `v2-p1-autour-de-mme-swann`

Mechanical result:

- `run-278` completed cleanly:
  - `6/6` completed
  - `0` parse errors
  - `0` validation errors
- `run-280` completed cleanly:
  - `6/6` completed
  - `0` parse errors
  - `0` validation errors
- `run-282` started cleanly and wrote one completed unit:
  - `1/6` completed
  - `0` parse errors
  - `0` validation errors so far
- the parent process then exited on an explicit OpenAI API error:
  - `429 insufficient_quota`

Interpretive result:

- the completed batches did not reveal any new report-level interpretive problem
- the operationally interesting result was not literary but procedural:
  - unattended chaining across multiple batches worked
  - progress remained visible through manifest and file creation
  - failure was explicit rather than silent

Operational judgment:

- this should count as a genuinely useful first production-style dry run
- it showed that the operator can distinguish ordinary slowness from a real hard failure
- it also showed that completed work is preserved, so practical resumability looks good
- the main orchestration gap now exposed is specific:
  - after the parent process exited on the quota failure, `run-282` still showed `automation.in_progress: true`
- that means the current manifest semantics are not yet production-clean under interruption, even though the run state is still understandable to a careful operator

Recommended next step:

1. record the dry-run evidence in the strategic docs
2. decide whether to patch interruption-state handling before resuming
3. treat this as an operational gap rather than a prompt, reducer, or interpretive gap

## run-282 resumed completion of the first production-style dry run

`run-282` was later resumed after API credits were replenished and then completed cleanly.

Mechanical result:

- the resumed `automate` call correctly recognized the already completed unit and restarted as a `5`-unit request rather than duplicating work
- resumed batch result:
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
- final run state:
  - all `6/6` units are now present in `raw/` and `annotations/`
  - `wait --reduce --report` completed cleanly

Interpretive result:

- the closing stretch remained report-readable rather than noisy:
  - `Swann` negative across five units
  - `Odette` negative once
  - `la mère du narrateur` neutral once
- no new cross-lens inversion or local-report instability appeared in the resumed batch

Operational judgment:

- this completes the first production-style dry run
- the dry run now demonstrates all of the operational behaviors that mattered for this phase:
  - unattended chaining across multiple batches
  - visible distinction between ordinary slowness and explicit hard failure
  - preservation of completed work under interruption
  - successful resumption without duplicate writes
- the quota interruption should be treated as an external account-state issue rather than a stack failure
- the only genuine orchestration defect surfaced during the interruption was terminal manifest state after request failure, and that has now been patched in the runner

Recommended next step:

1. record the operational proof as complete in the strategic docs
2. compare the full evidence against the full-corpus automation threshold
3. proceed to full-corpus automation with monitoring

## v1-p2-un-amour-de-swann completed in full-corpus production mode

The full automated pass over `v1-p2-un-amour-de-swann` has now been completed through `run-340`.

Mechanical result:

- the chapter advanced cleanly from `run-310` through `run-340`
- the final tail batch `run-340` completed cleanly:
  - `2/2` completed
  - `0` parse errors
  - `0` validation errors
- the chapter closed without any review-gate stop condition

Interpretive result:

- the chapter showed stable report-level behavior across multiple distinct terrains:
  - extended Swann decline
  - prestige-heavy Guermantes and salon material
  - late local upward turns that were registered rather than flattened away
- the final tail remained coherent:
  - `Swann` mixed at `p-586-p-590`
  - `Swann` negative at `p-591`
- no recurring cross-lens inversion or alias-driven corruption appeared during the chapter pass

Operational judgment:

- this is now a chapter-scale production proof, not just a batch-scale proof
- the current prompt, reducer, and review-gate pattern remained usable without chapter-specific retuning
- the default next move is to continue directly into `v1-p3-noms-de-pays-le-nom`

## v1-p3-noms-de-pays-le-nom completed in full-corpus production mode

The full automated pass over `v1-p3-noms-de-pays-le-nom` has now been completed through `run-344`.

Mechanical result:

- the chapter opened and closed cleanly in two production batches:
  - `run-342` for `p-1` through `p-40`
  - `run-344` for `p-41` through `p-67`
- the closing batch `run-344` completed cleanly:
  - `6/6` completed
  - `0` parse errors
  - `0` validation errors
- the chapter closed without any review-gate stop condition

Interpretive result:

- the chapter surface remained distinct from `v1-p2` rather than mechanically inheriting its pattern
- the opening terrain registered:
  - positive `Swann`
  - a structured early `Gilberte` arc with rise, mixed middle, and decline
- the tail remained coherent:
  - `Swann` mixed but net negative
  - `Odette` turning from a local loss to late wins
  - `la mère du narrateur` neutral
- the final batch showed especially strong operational cleanliness:
  - no mixed units in any lens
  - no cross-lens label disagreements
  - no direction disagreements
  - no sign flips

Operational judgment:

- this is a second chapter-scale production proof under unchanged prompt, reducer, and review-gate settings
- the stack continues to transfer cleanly across chapter boundaries without chapter-specific retuning
- the default next move is to continue directly into `v2-p1-autour-de-mme-swann`

## v2-p1-autour-de-mme-swann completed with stable chapter-internal parallelism

The full automated pass over `v2-p1-autour-de-mme-swann` has now been completed through `run-362`.

Mechanical result:

- the chapter was advanced through a mix of:
  - clean chapter-internal parallel pairs
  - a final single tail batch at the chapter boundary
- the closing batch `run-362` completed cleanly:
  - `3/3` completed
  - `0` parse errors
  - `0` validation errors
- the chapter closed without any review-gate stop condition

Interpretive result:

- the chapter sustained coherent report-level behavior across a wide range of social and rhetorical terrain:
  - strong Norpois stretches
  - salon and prestige-friction clusters
  - mixed Swann / Odette / Gilberte local turns that remained legible rather than noisy
- the final tail remained especially clean:
  - `Odette` strongly positive in the final two units
  - `Gilberte` neutral in the opening tail unit
  - no mixed units
  - no cross-lens disagreements
  - no sign flips

Operational judgment:

- this is now the strongest operational proof yet for the current rule:
  - parallel pairs are acceptable inside long, already-stable chapters
  - chapter boundaries should still collapse back to a single tail batch
- the project now has direct evidence that controlled parallelism can increase throughput without visibly degrading the report surface in stable terrain
- the default next move is to open `v2-p2-noms-de-pays-le-pays` sequentially for one batch, then re-evaluate whether to promote it back to chapter-internal parallel mode

## v2-p2-noms-de-pays-le-pays completed with stable but higher-variance parallelism

The full automated pass over `v2-p2-noms-de-pays-le-pays` has now been completed through `run-388`.

Mechanical result:

- the chapter opened with one clean sequential probe batch:
  - `run-364`
- after that opening probe, the chapter was advanced successfully through controlled chapter-internal parallel pairs
- the final pair completed cleanly:
  - `run-386`: `8/8` completed, `0` parse errors, `0` validation errors
  - `run-388`: `3/3` completed, `0` parse errors, `0` validation errors
- the chapter closed without any review-gate stop condition

Interpretive result:

- `v2-p2` was somewhat noisier than `v2-p1`, but the noise stayed bounded rather than compounding
- key terrain remained report-readable across the chapter:
  - extended `la grand-mère` and family material
  - mixed social-friction passages around `Mme de Villeparisis`, `Mme/M. de Cambremer`, and related prestige figures
  - later positive `Elstir` and `Albertine` stretches
  - localized losses for `Bloch`, `Bloch père`, `le directeur`, and others where the surrounding terrain plausibly called for them
- no recurring cross-lens inversion and no sign-flip pattern emerged
- the chapter tail remained especially clean:
  - `Françoise` positive at the close
  - `le directeur` negative once, then neutral

Operational judgment:

- this chapter strengthens the controlled-parallel rule in a more demanding way than `v2-p1` did:
  - parallel pairs remained usable even in somewhat higher-variance terrain
  - the right response to chapter-level uncertainty was not immediate rollback, but closer watch plus boundary discipline
- the chapter still supports the standing rule:
  - use parallel pairs inside long, already-stable chapters
  - open new chapters sequentially
  - collapse back to a single tail batch at chapter boundaries
- the default next move is to open `v3-p1` sequentially for one batch, then decide whether it should be promoted back to chapter-internal parallel mode

## v3-p1 completed with stable large-chapter parallelism plus one successful sequential probe

The full automated pass over `v3-p1` has now been completed through `run-434`.

Mechanical result:

- the chapter opened with one sequential probe batch:
  - `run-390`
- it then advanced successfully through long controlled parallel pairs across most of the chapter
- one later noisy patch triggered a deliberate single sequential probe:
  - `run-424`
- that probe came back fully clean and justified returning to parallel mode
- the final tail batch `run-434` completed cleanly:
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
- the chapter closed without any review-gate stop condition

Interpretive result:

- `v3-p1` sustained coherent report-level behavior across very broad terrain:
  - extended Saint-Loup arcs
  - Guermantes prestige and salon material
  - Balbec-adjacent social ranking and snub sequences
  - later Charlus / Morel / Bloch stretches
  - quiet closing material around `la grand-mère` and `Albertine`
- most of the chapter remained operationally quiet under parallel execution
- the one noisier zone was bounded rather than contagious:
  - `run-422` reached the mixed-unit threshold without crossing it
  - the next sequential probe `run-424` returned `0/0/0` mixed pressure and `0` cross-lens disagreement
  - later parallel pairs returned to a clean pattern
- the final tail remained especially clean:
  - `la grand-mère` negative in the closing two larger tail units
  - `Albertine` negative in the last short unit
  - no mixed units
  - no cross-lens disagreement
  - no sign flips

Operational judgment:

- this is the strongest large-chapter production proof yet for the current operating rule:
  - use parallel pairs inside long, already-stable chapters
  - if a local patch becomes noisier, insert one sequential probe rather than immediately abandoning parallelism
  - return to parallel mode when the probe confirms that the noise is local rather than structural
  - still collapse back to a single tail batch at the chapter boundary
- the chapter materially strengthens confidence that the current prompt, reducer, and review-gate stack can tolerate large-scale automated throughput without hidden degradation
- the default next move is to open `v3-p2` sequentially for one batch, then decide whether to promote it back to chapter-internal parallel mode

## Restart checkpoint: v3-p2 production pass through paragraph 596

The full-corpus production pass has advanced partway through `v3-p2`.

Accepted state before reboot:

- `v3-p2` opened sequentially with `run-436`, then promoted back to controlled chapter-internal parallel mode
- the current accepted endpoint is paragraph `596`
- latest accepted output pair:
  - `run-462`: `v3-p2#p-521-p-556`
  - `run-464`: `v3-p2#p-561-p-596`
- both latest outputs completed with `8/8` units, `0` parse errors, and `0` validation errors
- both passed the review gate

Latest review details:

- `run-462` was fully clean:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-464` showed bounded local variance:
  - mixed counts `{ inclusion: 2, local: 1, prestige: 1 }`
  - `2` label disagreements
  - `1` direction disagreement
  - `0` sign flips

Operational judgment:

- the current `v3-p2` stretch remains suitable for controlled parallel pairs
- the higher-variance Guermantes/social-prestige terrain is noisy but not structurally unstable
- old terminal sessions should be ignored after reboot; the run manifests are the source of truth
- the next restart move is to prepare source `run-465` for `v3-p2#p-601-p-640` and source `run-467` for `v3-p2#p-641-p-680`, then launch output `run-466` and output `run-468`

## v3-p2 completed and v4-p1 opened sequentially

The production pass has now completed `v3-p2` and the short `v4-p1` chapter.

Mechanical result:

- `run-466`: `v3-p2#p-601-p-640`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-468`: `v3-p2#p-641-p-680`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-470`: `v3-p2#p-681-p-720`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-472`: `v3-p2#p-721-p-730`
  - `2/2` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-476`: `v3-p2#p-731-p-733`
  - `1/1` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-474`: `v4-p1#p-1-p-22`
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-466`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-468`:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 2 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-470`:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 2 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-472`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips
- `run-476`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-474`:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 3 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Operational judgment:

- `v3-p2` closed cleanly after the final `p-731-p-733` tail was filled
- `v4-p1` was short enough that the sequential opener was also the full chapter pass
- `v4-p1` reached but did not exceed the mixed-unit review threshold in the prestige lens
- no parse, validation, or sign-flip stop condition appeared
- the next default move is to open `v4-p2` sequentially with source `run-477` for `v4-p2#p-1-p-40`, then launch output `run-478`

## v4-p2 opened sequentially

The `v4-p2` production pass has opened with the planned sequential probe batch.

Mechanical result:

- `run-478`: `v4-p2#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- mixed counts `{ inclusion: 2, local: 1, prestige: 2 }`
- `1` label disagreement
- `0` direction disagreements
- `0` sign flips

Report-first reading:

- the opener surfaced the expected title-heavy Guermantes terrain:
  - `duc de Châtellerault`
  - `baron de Charlus`
  - `M. de Vaugoubert`
  - `marquise de Gallardon`
  - `princesse de Guermantes`
- the dominant movement was local negative pressure around rank, exposure, and social diminishment
- mixed cases stayed bounded:
  - `M. de Vaugoubert` in `v4-p2#p-16-p-20`
  - `princesse de Guermantes` in `v4-p2#p-36-p-40`
- no title-collision or alias-collapse problem appeared in the opener

Operational judgment:

- the new-chapter sequential probe did its job and returned cleanly
- `v4-p2` can now be promoted back to controlled chapter-internal parallel mode
- the next default move is to prepare source `run-479` for `v4-p2#p-41-p-80` and source `run-481` for `v4-p2#p-81-p-120`, then launch output `run-480` and output `run-482`

## v4-p2 controlled parallel continuation through paragraph 120

The `v4-p2` production pass has continued in controlled chapter-internal parallel mode.

Prepared sources:

- `run-479`: `v4-p2#p-41-p-80`
- `run-481`: `v4-p2#p-81-p-120`

Mechanical result:

- `run-480`: `v4-p2#p-41-p-80`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-482`: `v4-p2#p-81-p-120`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-480`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-482`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-480` is dominated by negative local and inclusion pressure around social exclusion and rank:
  - repeated losses for `M. de Vaugoubert`
  - a strong exclusion loss for `Mme de Chaussepierre` in `v4-p2#p-66-p-70`
  - a Swann exclusion loss in `v4-p2#p-76-p-80`
- `run-482` shifts into a broader Guermantes/Saint-Loup/Swann field:
  - Swann and Robert de Saint-Loup carry the largest losses
  - Charlus has one loss in `v4-p2#p-106-p-110` and one win in `v4-p2#p-116-p-120`
  - the only mixed unit is Robert de Saint-Loup in `v4-p2#p-111-p-115`
- no cross-lens disagreement, direction disagreement, or sign flip appeared in either run

Operational judgment:

- the chapter-internal parallel promotion was successful
- `v4-p2` is now accepted through `p-120`
- the next default move is to prepare source `run-483` for `v4-p2#p-121-p-160` and source `run-485` for `v4-p2#p-161-p-200`, then launch output `run-484` and output `run-486`

## v4-p2 controlled parallel continuation through paragraph 200

The `v4-p2` production pass has continued with the next controlled parallel pair.

Prepared sources:

- `run-483`: `v4-p2#p-121-p-160`
- `run-485`: `v4-p2#p-161-p-200`

Mechanical result:

- `run-484`: `v4-p2#p-121-p-160`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-486`: `v4-p2#p-161-p-200`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-484`:
  - mixed counts `{ inclusion: 2, local: 1, prestige: 1 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips
- `run-486`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-484` moves through a Guermantes/Swann social-friction stretch:
  - the duchesse de Guermantes is elevated in `v4-p2#p-141-p-145` and `v4-p2#p-146-p-150`
  - Swann carries repeated losses across `v4-p2#p-126-p-140`
  - marquise de Gallardon and marquise de Saint-Euverte carry the strongest inclusion/social losses
  - bounded mixed cases appear for Charlus in `v4-p2#p-121-p-125` and the duchesse de Guermantes in `v4-p2#p-141-p-145`
- `run-486` shifts into Albertine/Odette/Gilberte material:
  - Odette has the clearest wins in `v4-p2#p-176-p-180` and `v4-p2#p-181-p-185`
  - la grand-mère has a clean positive appraisal in `v4-p2#p-191-p-195`
  - Albertine has losses in `v4-p2#p-161-p-165` and `v4-p2#p-196-p-200`
  - Gilberte, Françoise, and le directeur appear as local losses
- the only cross-lens disagreement in this pair is the bounded `run-484` duchesse case; no sign flip appears

Operational judgment:

- the controlled parallel pass remains usable through the shift from title-heavy Guermantes material into the Albertine/Odette/Gilberte field
- `v4-p2` is now accepted through `p-200`
- the next default move is to prepare source `run-487` for `v4-p2#p-201-p-240` and source `run-489` for `v4-p2#p-241-p-280`, then launch output `run-488` and output `run-490`

## v4-p2 controlled parallel continuation through paragraph 280

The `v4-p2` production pass has continued with another controlled parallel pair.

Prepared sources:

- `run-487`: `v4-p2#p-201-p-240`
- `run-489`: `v4-p2#p-241-p-280`

Mechanical result:

- `run-488`: `v4-p2#p-201-p-240`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-490`: `v4-p2#p-241-p-280`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-488`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-490`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-488` is a clean hotel/social-evaluation stretch:
  - princesse de Parme and Mme de Cambremer carry the main positive movements
  - le directeur, docteur Cottard, Albertine, and la grand-mère carry local negative pressure
  - Mme de Cambremer appears with both a win and a loss, but without mixed-unit or cross-lens instability
- `run-490` is narrower and fully coherent:
  - M. Nissim Bernard has the strongest win in `v4-p2#p-261-p-265`
  - Albertine and the narrator's mother also register wins
  - Mme de Cambremer carries two losses in `v4-p2#p-241-p-250`
- neither run produced mixed units, label disagreements, direction disagreements, or sign flips

Operational judgment:

- the controlled parallel pass remains clean through `v4-p2#p-280`
- `v4-p2` is now accepted through `p-280`
- the next default move is to prepare source `run-491` for `v4-p2#p-281-p-320` and source `run-493` for `v4-p2#p-321-p-360`, then launch output `run-492` and output `run-494`

## v4-p2 controlled parallel continuation through paragraph 360

The `v4-p2` production pass has continued through the next controlled parallel pair.

Prepared sources:

- `run-491`: `v4-p2#p-281-p-320`
- `run-493`: `v4-p2#p-321-p-360`

Mechanical result:

- `run-492`: `v4-p2#p-281-p-320`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-494`: `v4-p2#p-321-p-360`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-492`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips
- `run-494`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-492` stays readable across Albertine, Saint-Loup, Charlus, and Verdurin terrain:
  - Albertine carries the strongest negative pressure, especially in `v4-p2#p-306-p-310`
  - Mme Verdurin and the narrator's father carry the clearest wins
  - Robert de Saint-Loup is the only mixed case, stable across all three lenses in `v4-p2#p-306-p-310`
  - Charlus has a clean loss in `v4-p2#p-311-p-315`
- `run-494` shifts into Verdurin/Cottard/Cambremer/Morel material:
  - Saniette has the strongest loss in `v4-p2#p-351-p-355`
  - M. Verdurin, docteur Cottard, and marquis de Cambremer carry repeated losses
  - Morel is the only mixed/disagreement case in `v4-p2#p-341-p-345`
- no sign flip appeared in either run

Operational judgment:

- the controlled parallel pass remains usable through `v4-p2#p-360`
- the bounded Morel disagreement in `run-494` is not a stop condition
- `v4-p2` is now accepted through `p-360`
- the next default move is to prepare source `run-495` for `v4-p2#p-361-p-400` and source `run-497` for `v4-p2#p-401-p-440`, then launch output `run-496` and output `run-498`

## v4-p2 controlled parallel continuation through paragraph 440

The `v4-p2` production pass has continued through the next controlled parallel pair.

Prepared sources:

- `run-495`: `v4-p2#p-361-p-400`
- `run-497`: `v4-p2#p-401-p-440`

Mechanical result:

- `run-496`: `v4-p2#p-361-p-400`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed
- `run-498`: `v4-p2#p-401-p-440`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-496`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 1 }`
  - `2` label disagreements
  - `2` direction disagreements
  - `0` sign flips
- `run-498`:
  - mixed counts `{ inclusion: 3, local: 1, prestige: 3 }`
  - `2` label disagreements
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-496` is a Charlus/Saniette/Albertine/Verdurin stretch with mostly negative pressure:
  - Saniette carries the strongest inclusion loss in `v4-p2#p-396-p-400`
  - Albertine and Mme Verdurin also carry clear losses
  - Charlus is mostly negative across `v4-p2#p-376-p-395`, with a bounded mixed/positive-status case in `v4-p2#p-361-p-365`
  - la grand-mère remains neutral
- `run-498` is a higher-variance Charlus/Morel closing stretch:
  - Charlus has the clearest win in `v4-p2#p-406-p-410` and another win in `v4-p2#p-421-p-425`
  - Charlus also carries a loss in `v4-p2#p-416-p-420` and mixed cases in `v4-p2#p-411-p-415` and `v4-p2#p-426-p-430`
  - Morel, M. de Chevregny, and Brichot carry the clearest losses
  - Mme Verdurin remains neutral where she appears in the report surface
- no sign flip appeared in either run

Operational judgment:

- the controlled parallel pass remains usable through `v4-p2#p-440`
- `run-498` reaches the mixed-unit threshold but does not exceed it, and its lack of sign flips makes it acceptable to continue
- `v4-p2` is now accepted through `p-440`
- the next default move is to prepare source `run-499` for `v4-p2#p-441-p-450`, then launch output `run-500`

## v4-p2 final tail through paragraph 450

The `v4-p2` production pass has reached the end of the chapter.

Prepared source:

- `run-499`: `v4-p2#p-441-p-450`

Mechanical result:

- `run-500`: `v4-p2#p-441-p-450`
  - `2/2` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-500`:
  - mixed counts `{ inclusion: 0, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- the final tail is narrow and stable:
  - Bloch carries a loss in `v4-p2#p-441-p-445` across all three lenses
  - Albertine closes the chapter as neutral in the inclusion and prestige lenses
  - the local lens marks Albertine as mixed in `v4-p2#p-446-p-450`, preserving the positive emotional/elevating pressure alongside discrediting association
- this is a bounded label disagreement, not a directional disagreement or sign flip

Operational judgment:

- `v4-p2` is now complete through `p-450`
- the short tail closes without a review-gate stop condition
- the next default move is to enter the next canonical full-corpus chapter, preparing source `run-501` for `v5#p-1-p-40`, then launching output `run-502`

## v5 opening production batch through paragraph 40

The full-corpus production pass has entered the next canonical chapter, `v5`.

Prepared source:

- `run-501`: `v5#p-1-p-40`

Mechanical result:

- `run-502`: `v5#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-502`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `v5` opens with a narrow but coherent field:
  - Albertine carries the main negative pressure, with losses in `v5#p-6-p-10`, `v5#p-16-p-20`, `v5#p-31-p-35`, and `v5#p-36-p-40`
  - Albertine also has a clear positive reopening in `v5#p-21-p-25`
  - Françoise carries the strongest wins in `v5#p-11-p-15` and, outside the inclusion mixed label, `v5#p-16-p-20`
  - Bloch appears once as a loss in `v5#p-1-p-5`
- the only mixed unit is Françoise in the inclusion lens at `v5#p-16-p-20`
- no sign flip appears

Operational judgment:

- the transition from `v4-p2` into `v5` is clean at the report level
- the opening `v5` surface is narrow, but expected for the chapter terrain and not a stop condition
- `v5` is now accepted through `p-40`
- the next default move is to prepare source `run-503` for `v5#p-41-p-80`, then launch output `run-504`

## v5 production continuation through paragraph 80

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-503`: `v5#p-41-p-80`

Mechanical result:

- `run-504`: `v5#p-41-p-80`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-504`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-504` is fully clean across all three lenses:
  - Albertine carries early losses in `v5#p-41-p-55`, with a neutral unit in `v5#p-56-p-60`
  - duchesse de Guermantes carries the main positive movement in `v5#p-56-p-70`
  - duc de Guermantes carries a loss in `v5#p-71-p-75`
  - baron de Charlus carries a loss in `v5#p-76-p-80`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v5` remains clean through the transition from Albertine-centered material into Guermantes/Charlus terrain
- `v5` is now accepted through `p-80`
- the next default move is to prepare source `run-505` for `v5#p-81-p-120`, then launch output `run-506`

## v5 production continuation through paragraph 120

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-505`: `v5#p-81-p-120`

Mechanical result:

- `run-506`: `v5#p-81-p-120`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-506`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-506` stays coherent across Morel, Andrée, and Albertine material:
  - Morel carries losses in `v5#p-81-p-90`
  - Andrée carries a loss in `v5#p-96-p-100`
  - Albertine oscillates, with wins in `v5#p-101-p-105` and `v5#p-116-p-120`
  - Albertine also carries losses in `v5#p-106-p-110` and `v5#p-111-p-115`
- the lone label/direction disagreement is bounded around Albertine's mixed positive/negative pressure in `v5#p-106-p-110`
- no mixed units or sign flips appear

Operational judgment:

- `v5` remains stable through another psychologically volatile Albertine stretch
- the disagreement is local and expected, not a stop condition
- `v5` is now accepted through `p-120`
- the next default move is to prepare source `run-507` for `v5#p-121-p-160`, then launch output `run-508`

## v5 production continuation through paragraph 160

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-507`: `v5#p-121-p-160`

Mechanical result:

- `run-508`: `v5#p-121-p-160`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-508`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-508` is narrow but fully clean:
  - Albertine carries nearly the whole batch
  - Albertine has a clear win in `v5#p-131-p-135`
  - Albertine is neutral in `v5#p-141-p-145`
  - Aimé appears neutrally in `v5#p-146-p-150`
  - Albertine carries losses through most of the surrounding units, strongest in `v5#p-136-p-140`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v5` remains stable through another narrow Albertine-centered sequence
- the narrow surface is expected in this terrain and is not a stop condition
- `v5` is now accepted through `p-160`
- the next default move is to prepare source `run-509` for `v5#p-161-p-200`, then launch output `run-510`

## v5 production continuation through paragraph 200

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-509`: `v5#p-161-p-200`

Mechanical result:

- `run-510`: `v5#p-161-p-200`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-510`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-510` is fully clean and negative across the reported surface:
  - Albertine carries five losses, including `v5#p-161-p-165`, `v5#p-166-p-170`, and `v5#p-186-p-200`
  - Françoise carries losses in `v5#p-171-p-180`
  - Andrée carries a loss in `v5#p-181-p-185`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v5` remains stable through a negative, still narrow domestic/intimate stretch
- the fully aligned negative surface is not a stop condition
- `v5` is now accepted through `p-200`
- the next default move is to prepare source `run-511` for `v5#p-201-p-240`, then launch output `run-512`

## v5 production continuation through paragraph 240

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-511`: `v5#p-201-p-240`

Mechanical result:

- `run-512`: `v5#p-201-p-240`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-512`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-512` is fully clean with a small positive counterweight in a mostly Albertine-negative stretch:
  - Albertine has a win in `v5#p-201-p-205`
  - Françoise has a win in `v5#p-221-p-225`
  - la mère du narrateur appears neutrally in `v5#p-231-p-235`
  - Albertine carries losses in `v5#p-206-p-210`, `v5#p-216-p-220`, and `v5#p-226-p-240`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v5` remains stable through another mostly Albertine-centered domestic stretch
- the clean cross-lens surface supports continuing without intervention
- `v5` is now accepted through `p-240`
- the next default move is to prepare source `run-513` for `v5#p-241-p-280`, then launch output `run-514`

## v5 production continuation through paragraph 280

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-513`: `v5#p-241-p-280`

Mechanical result:

- `run-514`: `v5#p-241-p-280`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-514`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-514` remains coherent with one bounded prestige-lens mixed unit:
  - Bergotte carries the only clear positive inclusion result, a win in `v5#p-266-p-270`
  - Albertine carries losses in `v5#p-241-p-245`, `v5#p-251-p-265`, and a near-neutral inclusion result in `v5#p-271-p-275`
  - Morel carries losses in `v5#p-246-p-250` and `v5#p-276-p-280`
  - the lone mixed unit is prestige-only for Albertine in `v5#p-271-p-275`
- no direction disagreements or sign flips appear

Operational judgment:

- `v5` remains stable through another Albertine-heavy stretch with a Bergotte counterweight and Morel losses
- the single prestige mixed unit is bounded and does not represent a stop condition
- `v5` is now accepted through `p-280`
- the next default move is to prepare source `run-515` for `v5#p-281-p-320`, then launch output `run-516`

## v5 production continuation through paragraph 320

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-515`: `v5#p-281-p-320`

Mechanical result:

- `run-516`: `v5#p-281-p-320`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-516`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-516` shifts into a Charlus-centered negative sequence with one bounded mixed unit:
  - Charlus carries the inclusion mixed unit in `v5#p-291-p-295`
  - Charlus also carries losses in `v5#p-296-p-320`
  - Swann carries the strongest inclusion loss in `v5#p-286-p-290`
  - Brichot carries a loss in `v5#p-281-p-285`
- no direction disagreements or sign flips appear

Operational judgment:

- `v5` remains stable through the transition from Albertine-heavy material into Charlus-centered social diminishment
- the single inclusion mixed unit is bounded and below the review threshold
- `v5` is now accepted through `p-320`
- the next default move is to prepare source `run-517` for `v5#p-321-p-360`, then launch output `run-518`

## v5 production continuation through paragraph 360

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-517`: `v5#p-321-p-360`

Mechanical result:

- `run-518`: `v5#p-321-p-360`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-518`:
  - mixed counts `{ inclusion: 2, local: 2, prestige: 1 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-518` remains acceptable through a more mixed Charlus/social-prestige stretch:
  - M. Vinteuil carries clear wins in `v5#p-326-p-335`
  - Charlus carries mixed units in `v5#p-321-p-325` and `v5#p-341-p-345`
  - Charlus also carries losses in `v5#p-351-p-360`
  - Mme Verdurin carries the strongest inclusion loss in `v5#p-336-p-340`
  - Brichot carries a loss in `v5#p-341-p-345`
  - la reine de Naples appears neutrally in `v5#p-356-p-360`
- the lone direction disagreement is bounded, and no sign flips appear

Operational judgment:

- `v5` remains stable through a mixed Charlus-centered social and prestige passage
- the mixed counts remain below threshold and no sign flip appears, so this is not a stop condition
- `v5` is now accepted through `p-360`
- the next default move is to prepare source `run-519` for `v5#p-361-p-400`, then launch output `run-520`

## v5 production continuation through paragraph 400

The `v5` production pass has continued through the next standard batch.

Prepared source:

- `run-519`: `v5#p-361-p-400`

Mechanical result:

- `run-520`: `v5#p-361-p-400`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-520`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-520` returns to a mostly Albertine-centered sequence with a small positive counterweight:
  - M. Verdurin carries a win in `v5#p-361-p-365`
  - Albertine carries wins in `v5#p-381-p-385` and `v5#p-386-p-390`
  - Albertine carries losses in `v5#p-366-p-380` and `v5#p-391-p-400`
  - Françoise appears neutrally in `v5#p-391-p-395`
  - the only mixed unit is Albertine in `v5#p-386-p-390`
- the single direction disagreement is bounded, and no sign flips appear

Operational judgment:

- `v5` remains stable through another Albertine-heavy passage with limited positive counterpressure
- the single mixed unit and direction disagreement remain below threshold and do not represent a stop condition
- `v5` is now accepted through `p-400`
- the next default move is to prepare source `run-521` for `v5#p-401-p-440`, then launch output `run-522`

## v5 final tail through paragraph 428

The `v5` production pass has reached the end of the chapter.

Prepared source:

- `run-521`: `v5#p-401-p-428`
  - this is a six-unit chapter tail because `v5` ends at paragraph `428`

Mechanical result:

- `run-522`: `v5#p-401-p-428`
  - `6/6` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-522`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-522` closes `v5` cleanly:
  - M. Vinteuil carries the clearest win in `v5#p-401-p-405`
  - Albertine carries the only mixed unit in `v5#p-406-p-410`
  - Albertine carries losses in `v5#p-411-p-425`
  - Albertine closes neutrally in the short tail `v5#p-426-p-428`
- no label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v5` closes without a review-gate stop condition
- the tail confirms the same stable interpretive pattern seen across the recent `v5` batches
- `v5` is now complete through `p-428`
- the next default move is to enter `v6-p1`, preparing source `run-523` for `v6-p1#p-1-p-40`, then launching output `run-524`

## v6-p1 opening production batch through paragraph 40

The full-corpus production pass has entered the next canonical chapter, `v6-p1`.

Prepared source:

- `run-523`: `v6-p1#p-1-p-40`

Mechanical result:

- `run-524`: `v6-p1#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-524`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-524` opens `v6-p1` with a fully clean surface:
  - Albertine carries wins in `v6-p1#p-1-p-15` and `v6-p1#p-26-p-30`
  - Robert de Saint-Loup carries a win in `v6-p1#p-16-p-20`
  - Robert de Saint-Loup carries a loss in `v6-p1#p-21-p-25`
  - Albertine carries the strongest loss in `v6-p1#p-31-p-35`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- the transition from `v5` into `v6-p1` is clean at the report level
- `v6-p1` opens with a narrow but coherent Albertine/Saint-Loup surface
- `v6-p1` is now accepted through `p-40`
- the next default move is to prepare source `run-525` for `v6-p1#p-41-p-80`, then launch output `run-526`

## v6-p1 production continuation through paragraph 80

The `v6-p1` production pass has continued through the next standard batch.

Prepared source:

- `run-525`: `v6-p1#p-41-p-80`

Mechanical result:

- `run-526`: `v6-p1#p-41-p-80`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-526`:
  - mixed counts `{ inclusion: 0, local: 1, prestige: 1 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-526` remains coherent with one bounded Albertine mixed case outside the inclusion lens:
  - Albertine carries the strongest loss in `v6-p1#p-41-p-45`
  - Robert de Saint-Loup carries a loss in `v6-p1#p-56-p-60`
  - Françoise carries a loss in `v6-p1#p-61-p-65`
  - Albertine is neutral in inclusion but mixed in local/prestige at `v6-p1#p-66-p-70`
  - Aimé carries a win in `v6-p1#p-71-p-75`
  - Albertine carries the strongest win in `v6-p1#p-76-p-80`
- no direction disagreements or sign flips appear

Operational judgment:

- `v6-p1` remains stable through the second production batch
- the local/prestige mixed unit is bounded and does not represent a stop condition
- `v6-p1` is now accepted through `p-80`
- the next default move is to prepare source `run-527` for `v6-p1#p-81-p-120`, then launch output `run-528`

## v6-p1 final tail through paragraph 120

The `v6-p1` production pass has reached the end of the chapter.

Prepared source:

- `run-527`: `v6-p1#p-81-p-120`

Mechanical result:

- `run-528`: `v6-p1#p-81-p-120`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-528`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 2 }`
  - `2` label disagreements
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-528` closes `v6-p1` with bounded mixed pressure around Albertine:
  - Swann carries a win in `v6-p1#p-81-p-85`
  - Albertine has mixed pressure in `v6-p1#p-86-p-90` and `v6-p1#p-101-p-105`
  - Albertine carries losses in `v6-p1#p-91-p-110`
  - Aimé appears neutrally in `v6-p1#p-96-p-100`
  - Andrée carries a win in `v6-p1#p-111-p-115` and a loss in `v6-p1#p-116-p-120`
- the direction disagreement is bounded, and no sign flips appear

Operational judgment:

- `v6-p1` closes without a review-gate stop condition
- the higher mixed count in the tail remains below threshold and fits the reflective aftermath terrain
- `v6-p1` is now complete through `p-120`
- the next default move is to enter `v6-p2`, preparing source `run-529` for `v6-p2#p-1-p-40`, then launching output `run-530`

## v6-p2 opening production batch through paragraph 40

The full-corpus production pass has entered the next canonical chapter, `v6-p2`.

Prepared source:

- `run-529`: `v6-p2#p-1-p-40`

Mechanical result:

- `run-530`: `v6-p2#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-530`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-530` opens `v6-p2` with a fully clean but wider social surface:
  - Gilberte carries wins in `v6-p2#p-16-p-20` and `v6-p2#p-26-p-30`
  - Mlle d'Eporcheville carries a win in `v6-p2#p-6-p-10`
  - Albertine, la mere du narrateur, and the duchesse de Guermantes appear neutrally
  - Gilberte carries losses in `v6-p2#p-21-p-25` and `v6-p2#p-36-p-40`
  - Swann carries a loss in `v6-p2#p-31-p-35`
  - Francoise carries a loss in `v6-p2#p-11-p-15`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- the transition from `v6-p1` into `v6-p2` is clean at the report level
- `v6-p2` opens with a broader but fully aligned social field
- `v6-p2` is now accepted through `p-40`
- the next default move is to prepare source `run-531` for `v6-p2#p-41-p-72`, then launch output `run-532`

## v6-p2 final tail through paragraph 72

The `v6-p2` production pass has reached the end of the chapter.

Prepared source:

- `run-531`: `v6-p2#p-41-p-72`

Mechanical result:

- `run-532`: `v6-p2#p-41-p-72`
  - `7/7` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-532`:
  - mixed counts `{ inclusion: 3, local: 2, prestige: 3 }`
  - `1` label disagreement
  - `1` direction disagreement
  - `0` sign flips

Report-first reading:

- `run-532` closes `v6-p2` with a coherent but more negative social field:
  - Gilberte is mixed in `v6-p2#p-41-p-45`
  - Bloch carries a loss in `v6-p2#p-46-p-50`
  - Swann carries the strongest inclusion loss in `v6-p2#p-51-p-55`
  - Albertine carries losses in `v6-p2#p-56-p-60` and `v6-p2#p-61-p-65`
  - Andree is neutral in `v6-p2#p-61-p-65` and carries a loss in `v6-p2#p-66-p-70`
  - Albertine and Andree are both mixed in the short tail `v6-p2#p-71-p-72`
- the mixed counts reach the review threshold in inclusion and prestige but do not exceed it
- the one label disagreement and one direction disagreement are bounded, and no sign flips appear

Operational judgment:

- `v6-p2` closes without a review-gate stop condition
- the tail is higher variance than the opener, but the variance is localized in mixed Albertine/Andree tail material
- `v6-p2` is now complete through `p-72`
- the next default move is to enter `v6-p3`, preparing source `run-533` for `v6-p3#p-1-p-40`, then launching output `run-534`

## v6-p3 opening production batch through paragraph 40

The full-corpus production pass has entered the next canonical chapter, `v6-p3`.

Prepared source:

- `run-533`: `v6-p3#p-1-p-40`

Mechanical result:

- `run-534`: `v6-p3#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-534`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-534` opens `v6-p3` with a fully clean and narrow report surface:
  - la mere du narrateur carries a win in `v6-p3#p-1-p-5`
  - Mme de Villeparisis carries losses in `v6-p3#p-6-p-10` and `v6-p3#p-26-p-30`
  - Norpois carries losses in `v6-p3#p-11-p-15` and `v6-p3#p-21-p-25`
  - Mme de Villeparisis carries a win in `v6-p3#p-31-p-35`
  - Mme de Villeparisis is neutral in `v6-p3#p-36-p-40`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- the transition from `v6-p2` into `v6-p3` is clean at the report level
- the opener is socially narrow but fully aligned across lenses
- `v6-p3` is now accepted through `p-40`
- the next default move is to prepare source `run-535` for `v6-p3#p-41-p-69`, then launch output `run-536`

## v6-p3 final tail through paragraph 69

The `v6-p3` production pass has reached the end of the chapter.

Prepared source:

- `run-535`: `v6-p3#p-41-p-69`

Mechanical result:

- `run-536`: `v6-p3#p-41-p-69`
  - `6/6` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-536`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-536` closes `v6-p3` with a fully clean and compact report surface:
  - Norpois carries a win in `v6-p3#p-41-p-45`
  - Norpois carries a loss in `v6-p3#p-46-p-50`
  - Albertine carries the strongest inclusion loss in `v6-p3#p-51-p-55`
  - la mere du narrateur is neutral in `v6-p3#p-56-p-60`
  - la mere du narrateur carries wins in `v6-p3#p-61-p-65` and `v6-p3#p-66-p-69`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v6-p3` closes without a review-gate stop condition
- the chapter tail remains fully aligned across all three lenses
- `v6-p3` is now complete through `p-69`
- the next default move is to enter the short `v6-p4` chapter, preparing source `run-537` for `v6-p4#p-1-p-25`, then launching output `run-538`

## v6-p4 full short chapter through paragraph 25

The `v6-p4` production pass has completed the full short chapter.

Prepared source:

- `run-537`: `v6-p4#p-1-p-25`

Mechanical result:

- `run-538`: `v6-p4#p-1-p-25`
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-538`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-538` completes `v6-p4` with a fully clean short-chapter surface:
  - Robert de Saint-Loup carries a win in `v6-p4#p-1-p-5`
  - Legrandin carries the strongest inclusion win in `v6-p4#p-6-p-10`
  - Robert de Saint-Loup carries losses in `v6-p4#p-11-p-15` and `v6-p4#p-16-p-20`
  - Gilberte carries a loss in `v6-p4#p-21-p-25`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v6-p4` closes without a review-gate stop condition
- the full short chapter is clean across all three lenses
- `v6-p4` is now complete through `p-25`
- the next default move is to enter the short `v7-p1-a-tansonville` chapter, preparing source `run-539` for `v7-p1-a-tansonville#p-1-p-25`, then launching output `run-540`

## v7-p1-a-tansonville full short chapter through paragraph 25

The full-corpus production pass has entered volume 7 and completed the short opening Tansonville chapter.

Prepared source:

- `run-539`: `v7-p1-a-tansonville#p-1-p-25`

Mechanical result:

- `run-540`: `v7-p1-a-tansonville#p-1-p-25`
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-540`:
  - mixed counts `{ inclusion: 1, local: 0, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-540` opens volume 7 with a stable, Saint-Loup-centered negative surface:
  - Robert de Saint-Loup carries losses in `v7-p1-a-tansonville#p-1-p-5` and `v7-p1-a-tansonville#p-11-p-15`
  - Robert de Saint-Loup is mixed only in the inclusion lens in `v7-p1-a-tansonville#p-6-p-10`
  - Françoise is neutral in `v7-p1-a-tansonville#p-6-p-10`
  - Mme Bontemps carries a loss in `v7-p1-a-tansonville#p-16-p-20`
  - Swann carries the lone win in `v7-p1-a-tansonville#p-21-p-25`
- the one label disagreement is the bounded Saint-Loup mixed-versus-neutral distinction in `v7-p1-a-tansonville#p-6-p-10`
- no direction disagreement or sign flip appears

Operational judgment:

- `v7-p1-a-tansonville` closes without a review-gate stop condition
- the full short chapter is acceptable across all three lenses
- `v7-p1-a-tansonville` is now complete through `p-25`
- the next default move is to enter `v7-p2-m-de-charlus-pendant-la-guerre`, preparing source `run-541` for `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-40`, then launching output `run-542`

## v7-p2 opening production batch through paragraph 40

The full-corpus production pass has opened `v7-p2-m-de-charlus-pendant-la-guerre`.

Prepared source:

- `run-541`: `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-40`

Mechanical result:

- `run-542`: `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-542`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-542` opens `v7-p2` with a fully clean cross-lens surface:
  - Elstir carries a win in `v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-5`
  - Mme Verdurin carries a win in `v7-p2-m-de-charlus-pendant-la-guerre#p-6-p-10`, then losses in `v7-p2-m-de-charlus-pendant-la-guerre#p-11-p-15` and `v7-p2-m-de-charlus-pendant-la-guerre#p-36-p-40`
  - Robert de Saint-Loup carries the strongest inclusion win in `v7-p2-m-de-charlus-pendant-la-guerre#p-16-p-20`
  - Françoise carries a loss in `v7-p2-m-de-charlus-pendant-la-guerre#p-21-p-25`
  - Gilberte carries a win in `v7-p2-m-de-charlus-pendant-la-guerre#p-26-p-30`
  - baron de Charlus carries a loss in `v7-p2-m-de-charlus-pendant-la-guerre#p-31-p-35`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v7-p2` opens without a review-gate stop condition
- the opening wartime/Charlus material is fully aligned across all three lenses
- `v7-p2-m-de-charlus-pendant-la-guerre` is now accepted through `p-40`
- the next default move is to prepare source `run-543` for `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-80`, then launch output `run-544`

## v7-p2 final batch through paragraph 80

The `v7-p2-m-de-charlus-pendant-la-guerre` production pass has reached the end of the chapter.

Prepared source:

- `run-543`: `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-80`

Mechanical result:

- `run-544`: `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-80`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-544`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-544` closes `v7-p2` with a coherent Charlus-negative tail:
  - Mme Verdurin carries a loss in `v7-p2-m-de-charlus-pendant-la-guerre#p-41-p-45`
  - baron de Charlus carries losses from `v7-p2-m-de-charlus-pendant-la-guerre#p-46-p-50` through `v7-p2-m-de-charlus-pendant-la-guerre#p-71-p-75`
  - Morel is mixed in `v7-p2-m-de-charlus-pendant-la-guerre#p-51-p-55` across all three lenses
  - M. de Marsantes carries the lone win in `v7-p2-m-de-charlus-pendant-la-guerre#p-76-p-80`
- the only mixed unit is the Morel signal in `p-51-p-55`, and it is aligned across all lenses
- no label disagreement, direction disagreement, or sign flip appears

Operational judgment:

- `v7-p2` closes without a review-gate stop condition
- the final wartime/Charlus material is strongly negative but report-coherent
- `v7-p2-m-de-charlus-pendant-la-guerre` is now complete through `p-80`
- the next default move is to enter `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`, preparing source `run-545` for `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-40`, then launching output `run-546`

## v7-p3 opening production batch through paragraph 40

The full-corpus production pass has opened `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`.

Prepared source:

- `run-545`: `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-40`

Mechanical result:

- `run-546`: `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-546`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-546` opens `v7-p3` with a fully clean cross-lens surface:
  - baron de Charlus carries losses in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-5` and `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-6-p-10`
  - M. Vinteuil carries a win in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-11-p-15`
  - Swann carries a loss in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-16-p-20`
  - Bergotte carries a loss in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-21-p-25`, then a win in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-26-p-30`
  - Albertine carries a win in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-36-p-40`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v7-p3` opens without a review-gate stop condition
- the opening matinee material is fully aligned across all three lenses
- `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle` is now accepted through `p-40`
- the next default move is to prepare source `run-547` for `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-41-p-45`, then launch output `run-548`

## v7-p3 final tail through paragraph 45

The `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle` production pass has reached the end of the chapter.

Prepared source:

- `run-547`: `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-41-p-45`

Mechanical result:

- `run-548`: `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-41-p-45`
  - `1/1` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-548`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-548` closes `v7-p3` with a compact, fully aligned tail:
  - duc de Guermantes carries a loss in `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-41-p-45`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v7-p3` closes without a review-gate stop condition
- the short tail is fully aligned across all three lenses
- `v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle` is now complete through `p-45`
- the next default move is to enter `v7-p4-le-bal-de-tetes`, preparing source `run-549` for `v7-p4-le-bal-de-tetes#p-1-p-40`, then launching output `run-550`

## v7-p4 opening production batch through paragraph 40

The full-corpus production pass has opened `v7-p4-le-bal-de-tetes`.

Prepared source:

- `run-549`: `v7-p4-le-bal-de-tetes#p-1-p-40`

Mechanical result:

- `run-550`: `v7-p4-le-bal-de-tetes#p-1-p-40`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-550`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 1 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-550` opens `v7-p4` with a coherent, socially sharp opening surface:
  - duc de Châtellerault carries a loss in `v7-p4-le-bal-de-tetes#p-1-p-5`
  - Bloch carries losses in `v7-p4-le-bal-de-tetes#p-6-p-10` and `v7-p4-le-bal-de-tetes#p-16-p-20`
  - Legrandin carries the strongest inclusion loss in `v7-p4-le-bal-de-tetes#p-11-p-15`
  - princesse de Guermantes carries a loss in `v7-p4-le-bal-de-tetes#p-21-p-25`
  - Odette is mixed in `v7-p4-le-bal-de-tetes#p-31-p-35` across all three lenses
  - Mme Verdurin carries a win in `v7-p4-le-bal-de-tetes#p-36-p-40`
- the only mixed unit is the Odette signal in `p-31-p-35`, and it is aligned across all lenses
- no label disagreement, direction disagreement, or sign flip appears

Operational judgment:

- `v7-p4` opens without a review-gate stop condition
- the opening `Bal de tetes` material is acceptable across all three lenses
- `v7-p4-le-bal-de-tetes` is now accepted through `p-40`
- the next default move is to prepare source `run-551` for `v7-p4-le-bal-de-tetes#p-41-p-80`, then launch output `run-552`

## v7-p4 continuation production batch through paragraph 80

The `v7-p4-le-bal-de-tetes` production pass has continued through the next 40-paragraph span.

Prepared source:

- `run-551`: `v7-p4-le-bal-de-tetes#p-41-p-80`

Mechanical result:

- `run-552`: `v7-p4-le-bal-de-tetes#p-41-p-80`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-552`:
  - mixed counts `{ inclusion: 1, local: 1, prestige: 0 }`
  - `1` label disagreement
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-552` keeps the `Bal de tetes` social-recognition surface coherent:
  - Morel carries a win in `v7-p4-le-bal-de-tetes#p-41-p-45`
  - Bloch carries a win in `v7-p4-le-bal-de-tetes#p-46-p-50`
  - Swann carries a loss in `v7-p4-le-bal-de-tetes#p-51-p-55`
  - duchesse de Guermantes carries a loss in `v7-p4-le-bal-de-tetes#p-56-p-60`
  - Bloch is mixed in inclusion and local, but neutral in prestige, in `v7-p4-le-bal-de-tetes#p-61-p-65`
  - Gilberte is neutral and Robert de Saint-Loup carries a win in `v7-p4-le-bal-de-tetes#p-66-p-70`
  - Gilberte carries a loss in `v7-p4-le-bal-de-tetes#p-71-p-75`
  - la Berma carries the strongest inclusion loss in `v7-p4-le-bal-de-tetes#p-76-p-80`
- the only cross-lens label disagreement is the bounded Bloch mixed-versus-neutral distinction in `p-61-p-65`
- no direction disagreement or sign flip appears

Operational judgment:

- `v7-p4` continues without a review-gate stop condition
- the middle `Bal de tetes` material is acceptable across all three lenses
- `v7-p4-le-bal-de-tetes` is now accepted through `p-80`
- the next default move is to prepare source `run-553` for `v7-p4-le-bal-de-tetes#p-81-p-120`, then launch output `run-554`

## v7-p4 continuation production batch through paragraph 120

The `v7-p4-le-bal-de-tetes` production pass has completed the last full 40-paragraph batch before the chapter tail.

Prepared source:

- `run-553`: `v7-p4-le-bal-de-tetes#p-81-p-120`

Mechanical result:

- `run-554`: `v7-p4-le-bal-de-tetes#p-81-p-120`
  - `8/8` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-554`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-554` is fully clean across all three lenses:
  - Bloch carries a loss in `v7-p4-le-bal-de-tetes#p-81-p-85`
  - la Berma carries a win in `v7-p4-le-bal-de-tetes#p-86-p-90`, then a loss in `v7-p4-le-bal-de-tetes#p-96-p-100`
  - baron de Charlus carries a loss in `v7-p4-le-bal-de-tetes#p-91-p-95`
  - duc de Guermantes carries losses in `v7-p4-le-bal-de-tetes#p-101-p-105` and `v7-p4-le-bal-de-tetes#p-106-p-110`
  - Gilberte carries a loss and duchesse de Guermantes is neutral in `v7-p4-le-bal-de-tetes#p-111-p-115`
  - Françoise carries a win in `v7-p4-le-bal-de-tetes#p-116-p-120`
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v7-p4` continues without a review-gate stop condition
- the late `Bal de tetes` material is fully aligned across all three lenses
- `v7-p4-le-bal-de-tetes` is now accepted through `p-120`
- the next default move is to prepare source `run-555` for the final tail, `v7-p4-le-bal-de-tetes#p-121-p-141`, then launch output `run-556`

## v7-p4 final tail through paragraph 141

The `v7-p4-le-bal-de-tetes` production pass has reached the end of the chapter and the end of the canonical ISLT chapter list.

Prepared source:

- `run-555`: `v7-p4-le-bal-de-tetes#p-121-p-141`

Mechanical result:

- `run-556`: `v7-p4-le-bal-de-tetes#p-121-p-141`
  - `5/5` completed
  - `0` parse errors
  - `0` validation errors
  - review gate passed

Review surface:

- `run-556`:
  - mixed counts `{ inclusion: 0, local: 0, prestige: 0 }`
  - `0` label disagreements
  - `0` direction disagreements
  - `0` sign flips

Report-first reading:

- `run-556` closes `v7-p4` with a fully clean cross-lens surface:
  - Albertine carries a loss in `v7-p4-le-bal-de-tetes#p-121-p-125`
  - la grand-mère is neutral in `v7-p4-le-bal-de-tetes#p-126-p-130`
  - Elstir is neutral in `v7-p4-le-bal-de-tetes#p-131-p-135`
  - duc de Guermantes carries a loss in `v7-p4-le-bal-de-tetes#p-136-p-140`
  - `v7-p4-le-bal-de-tetes#p-141` completed mechanically without adding a scored report entry
- no mixed units, label disagreements, direction disagreements, or sign flips appear

Operational judgment:

- `v7-p4` closes without a review-gate stop condition
- the final tail is fully aligned across all three lenses
- `v7-p4-le-bal-de-tetes` is now complete through `p-141`
- the canonical ISLT production pass is now complete through the final exported chapter
- the next default move is a final corpus-level sanity/aggregation review over the accepted production outputs
