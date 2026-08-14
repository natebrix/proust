# Prestige and inclusion enrichment — design note

Status: **PROPOSED** (2026-08-14). Probe evidence in `outputs/budget-probe-001/`;
A/B evaluation staged in `outputs/enrichment-ab-001/`. Adoption gates below.

## Problem

Under scoring v2 the three lenses rank very different numbers of characters:
advantage 35, prestige 8, inclusion 9 (of 288). The ranking criterion is a
maximum uncertainty (rating band ≤ 200 points), so the gap traces to evidence
volume: 3,475 decided advantage comparisons versus 1,184 prestige and 531
inclusion. Part of that gap is Proust's text — most scenes stage a scene-level
contest, fewer visibly move standing or belonging. But part is structural in
prompt v2:

1. **No dimension criteria.** The five status dimensions are listed as bare
   names. `social_status` benefits from worked examples; `inclusion_exclusion`
   has none anywhere in the prompt.
2. **Budget crowding.** "Usually 1, never more than 2" status effects per
   character forces standing and belonging movements to compete with the
   scene-level movement for the same slots. 28% of character-unit annotations
   in the foundation corpus sit exactly at the 2-effect ceiling.
3. **Event ceiling.** Effects must cite `based_on_events`, and events are
   hard-capped at 2 per unit. In practice every foundation annotation of a
   dense salon unit uses exactly 2 events, which caps how many characters can
   move at all.

The goal is to raise prestige/inclusion evidence where the text supports it,
without manufacturing signal. Loosening elicitation invites a demand effect —
annotators finding standing movements because they were told to look — which
would tighten rating bands around induced evidence. Every change below is
therefore paired with a hardening, and adoption is gated on a control set.

## The standard

### Dimension criteria (new)

* `social_status` — a movement or display of standing **witnessed inside the
  world of the passage**. Someone present, or society's reported voice, must
  register it: deference given or withheld, a reception or invitation that
  marks rank, a public snub, a reputation spoken of as risen or fallen. A
  private, unshared judgment is `general_appraisal`, not `social_status`.
* `inclusion_exclusion` — a **boundary event**: a circle with an inside and an
  outside, and a character shown crossing that line or barred at it —
  introduced or not, greeted or cut, invited or left out, absorbed into the
  group or held at its edge. Mere presence at a gathering is not inclusion;
  absence is not exclusion; the boundary must be shown moving.
* The advantage-family dimensions (`general_appraisal`, `rhetorical_position`,
  `emotional_position`) get short definitions for symmetry.

### Effect budget (replaces "usually 1, never more than 2")

* Record every distinct movement that meets its dimension's criterion; no
  fixed per-character total.
* At most **one effect per dimension per character**; a second effect in the
  same dimension only for clearly separate moments citing different
  `based_on_events`.
* For one character, a single event grounds at most **one** effect among the
  three advantage-family dimensions (they are facets of the same contest).
  The same event may additionally ground a `social_status` or
  `inclusion_exclusion` effect when that criterion is independently met —
  a witticism that wins the room is both the duel and the standing it
  visibly confers.

### Event budget (the real lever)

* Default 1, ordinarily at most 2 — unchanged for ordinary passages.
* A dense social scene may carry up to **4 events** when each grounds a
  distinct witnessed movement the others do not cover. All existing
  anti-fragmentation rules stay in force.

Prompt: `proust/prompts/prompt_v2_1.md` (diff against `prompt_v2.md`).

## Probe evidence (budget-probe-001, 2026-08-13)

24 units: 14 social-dense, 5 silent-solitary controls, 10 total controls.
Arm B = criteria + open effect budget; baseline = foundation annotations.

* **No floodgates.** The criteria filter more than the open budget adds:
  55 effects on social units versus 73 at baseline. Dropped baseline effects
  were mostly state-descriptions or inferential stretches; kept effects are
  cleanly witnessed.
* **Zero demand effect.** All silent controls returned empty annotations in
  arm B; no prestige/inclusion effect appeared on any control unit in either
  arm.
* **Budget calibration.** Uncapped, no character ever carried more than one
  prestige or inclusion effect per unit — salon density expresses itself as
  more *characters* moving, not one character moving twice. Four multi-effect
  cases appeared in the advantage family: three legitimate (distinct events,
  different dimensions), one restatement (two facets of one event). This is
  why the rule is per-dimension plus one-facet-per-event, not the
  one-per-family rule originally proposed.
* **Event ceiling confirmed.** Both arms produced exactly 2 events on every
  unit; character coverage, not effect budget, is what the ceiling constrains.
  (Disclosure: the probe prompt left the base prompt's Task-rule 2-effect cap
  in place while its status_effects section said "no fixed cap" — the probe
  was therefore more conservative than designed. Its conclusions are lower
  bounds; v2.1 is internally consistent.)

## A/B evaluation and adoption gates

`outputs/enrichment-ab-001/`: 40 fresh units disjoint from the probe —
20 social-dense, 10 medium, 10 controls (5 silent, 5 intimate). Arm B =
prompt v2.1. Baseline = foundation annotations. A 10-unit **variance arm**
re-annotates social units with unmodified prompt v2 to measure re-roll noise,
so criteria effects can be distinguished from sampling effects.

Gates, in order of authority:

1. **Control flatness (hard).** No prestige/inclusion effects on silent
   controls; no rise on intimate controls beyond what the variance arm shows
   for re-roll noise. A demand effect fails the pass regardless of coverage
   gains.
2. **Distinctness discipline (hard).** No same-family pair of effects for one
   character citing the same single event; same-dimension seconds cite
   distinct events. Violations are counted and any above 5% of multi-effect
   cases fails.
3. **Coverage direction (target).** More characters carry prestige/inclusion
   effects on social-dense units than baseline, with each new effect passing
   spot review for witnessed/boundary grounding.
4. **Sign consistency (advisory).** Where baseline and arm B weigh the same
   character-unit-family, signs agree at a rate comparable to the variance
   arm's self-agreement; flips are reviewed individually.

Adoption means: full re-annotation pass with prompt v2.1 over the foundation
unit grid, gated batch-by-batch like foundation-run-001..034, then scoring v2
re-fit and re-promotion. Scoring formulas are unchanged; only evidence
elicitation changes. Nathan gates adoption on the A/B report.

## Adoption record

* 2026-08-14 — **Family-boundary fix adopted** (Nathan): the
  `inclusion_exclusion` criterion generalized from "a circle" to "an interior
  with an exterior" that need not be a social set — family table, household,
  bedroom door, clan, club, theatre box all qualify. Motivated by the A/B's
  one wrong drop (the goodnight-kiss dining-room exclusion) and by the
  corpus: ~21% of all foundation inclusion evidence is family/domestic.
* 2026-08-14 — **Boundary illustration adopted** (Nathan): the criterion now
  names the range of qualifying interiors — family table, household, clan,
  club, theatre box, the circle of a conversation, the intimacy of
  tutoiement, an institution, a nation, a clandestine fraternity — as
  illustration, not as a closed list. Grounded in the corpus census (173
  inclusion effects classified) and the novel's own boundary systems the
  corpus under-captures (Dreyfus-era salon realignments, the Sodome I
  recognition network).
* 2026-08-14 — **Staged parity is a CANDIDATE rule** (Nathan), not adopted:
  a delta-0 status effect records a lens staged between characters with a
  witnessed even outcome, scored as a draw. The bright line: uncertainty is
  not parity — *null is not the same as zero* — and co-presence is not
  parity. Mechanically already supported (tie band + WHR draw model,
  verified). Because the vacuous-draw disease of 2026-08-12 is the failure
  mode, adoption is gated on `parity-probe-001` (prompt
  `prompt_v2_2_candidate.md`): 8 temptation units (many characters present,
  little baseline movement), 8 social-dense, 8 controls. Pass requires
  every delta-0 to cite a staged comparison and zero delta-0s born of mere
  co-presence.
