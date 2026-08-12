# Character ELO Plan

> **Superseded as the current rating plan (2026-08-12).** Everything below is
> the reasoning that produced the ELO, Glicko-2, and v1 Whole-History Rating
> surfaces, and it is still the best account of WHY the corpus can be rated at
> all — pairwise comparison within a unit, narrative time as the axis, and
> uncertainty reported rather than hidden. What it does not describe is the
> input those systems now consume: scoring v2 derives one movement per
> character per lens from the annotator's own calibrated status effects,
> partitions the dimensions so the three lenses cannot leak into one another,
> and puts every uncertainty signal into a comparison's WEIGHT instead of its
> direction. Scoring v2 was adopted after a validation gate that used the
> surfaces planned here as its baselines, and the current standings, journey
> timelines, and character pages are built from it. Read
> [scoring_v2_design.md](scoring_v2_design.md) for the current formula and its
> adoption record, and keep this document for the history and the rating-theory
> rationale it still carries.

This document defines the first precise ELO-style rating plan for the annotation corpus.

It is intentionally narrow.

The first target is:

- a per-character ELO artifact for the `advantage` lens

If that artifact is interpretable and stable, the same framework can later be extended to:

- `prestige`
- `inclusion`

## Why ELO Is Now Plausible

Earlier in the project, ELO was premature because the corpus surface was not yet stable enough.

That is no longer true.

The project now has:

- a completed full-corpus accepted annotation surface
- reviewed same-person source canonicalization
- stable per-unit lens scoring
- deterministic duplicate-run resolution for aggregate outputs

So the question is no longer whether we can compute a rating mechanically.

The real question is:

- can we define a defensible notion of a match between characters?

This plan answers that question for a first-pass `advantage`-only implementation.

## Goal

Produce a derived artifact that answers:

1. which characters most often come out ahead of other characters in the immediate scene
2. which characters most often come out behind
3. how strong that pattern remains after repeated pairwise updates across the full corpus

This should be treated as:

- a ranking heuristic over local relative advantage

It should not be treated as:

- a metaphysical truth about character importance
- a replacement for the existing net-score and percentile surfaces

## First Scope

Version `v1` should use only:

- the accepted canonicalized corpus surface
- the `advantage` lens

It should not initially try to:

- combine all three lenses into one rating
- infer direct confrontation from event semantics
- model chapter-to-chapter temporal rating changes

The first implementation should be corpus-wide and static.

## Core Match Definition

The basic unit of competition is:

- one annotated unit

Within a unit, the competitors are:

- the normalized characters who receive an `advantage` `net_score`

For every unordered pair of distinct characters in the same unit:

- compare their `advantage` `net_score`
- assign a result

This creates a pairwise local contest from the unit-level outcome surface.

## Result Rule

Given characters `A` and `B` in the same unit:

- if `advantage_net_score(A) - advantage_net_score(B) > epsilon`, `A` wins and `B` loses
- if `advantage_net_score(B) - advantage_net_score(A) > epsilon`, `B` wins and `A` loses
- otherwise the result is a draw

The initial `v1` default should use:

- `epsilon = 0.25`

Rationale:

- it avoids treating tiny score differences as decisive victories
- it aligns with the existing scorer’s notion that very small values are close to neutral terrain

The chosen `epsilon` must be recorded in the output artifact metadata.

## Which Characters Count In A Unit

For `v1`, include a character in pairwise comparison if:

- the character appears in the normalized `advantage` timeline for that unit

Do not additionally filter by:

- label
- dominant status dimension
- chapter

This keeps the first pass simple and tied to the existing scoring surface.

Characters with `neutral` or very small scores may still participate, because:

- their relative position can still matter for pairwise ordering

## Duplicate Resolution

The ELO surface must inherit the same duplicate policy used in the current aggregate exports:

- `latest_reviewed_run_wins`

That means:

- if the same `unit_id` appears in multiple reviewed runs, only the latest reviewed accepted unit counts

The implementation should reuse the same preferred-run logic already used by:

- chapter overlays
- character chapter analysis

## Identity Rule

The default ELO surface should use:

- the accepted source-canonicalized character identities

That is, it should run on the same identity surface as:

- `corpus-review-current`
- `character-cross-lens-current`
- `chapter-overlays-current`

Historical normalized artifacts remain useful for comparison, but they are no longer the default source for current ELO computation.

## Rating Algorithm

Use standard Elo updates.

Suggested `v1` defaults:

- initial rating: `1500`
- `K = 24`

Expected score for `A` against `B`:

- `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

Observed score:

- win = `1.0`
- draw = `0.5`
- loss = `0.0`

Update:

- `R_A' = R_A + K * (S_A - E_A)`
- `R_B' = R_B + K * (S_B - E_B)`

These parameters should be recorded in the artifact metadata.

## Match Ordering

Elo is path-dependent, so ordering matters.

The `v1` implementation should use a deterministic, corpus-stable ordering:

1. canonical chapter order
2. paragraph start within chapter
3. paragraph end within chapter
4. `unit_id`
5. lexicographic character pair order

This does not claim to reconstruct narrative time perfectly.

It does guarantee:

- reproducibility
- stable comparison across runs

The output should state clearly that ratings are deterministic under a fixed corpus order.

## Multi-Character Units

Units often include more than two characters.

In `v1`, treat each unit as a complete pairwise round-robin among participating characters.

If a unit has `n` included characters, it contributes:

- `n * (n - 1) / 2` pairwise matches

This is acceptable for `v1`, but it should be acknowledged as a modeling choice rather than an obvious truth.

Interpretive caution:

- a large social scene creates more pairwise ELO pressure than a two-character scene

That is not necessarily wrong, but it should be visible in diagnostics.

## Diagnostics Required

The output must not only give ratings.

It must also include enough evidence to judge whether the ratings are meaningful.

Minimum per-character diagnostics:

- `character`
- `elo`
- `match_count`
- `win_count`
- `loss_count`
- `draw_count`
- `unit_count`
- `mean_advantage_net_score`
- `top_positive_unit`
- `top_negative_unit`

Minimum global diagnostics:

- `character_count`
- `match_count`
- `draw_rate`
- `epsilon`
- `k_factor`
- `initial_rating`
- `ordering_rule`
- `duplicate_resolution`

## Recommended Output Shape

The first artifact should be:

- `outputs/character-elo-advantage-current.json`
- `outputs/character-elo-advantage-current.md`

Suggested top-level fields:

```json
{
  "character_elo_version": "character_elo_advantage_v1",
  "lens": "advantage",
  "source_review_version": "corpus_sanity_review_v1",
  "duplicate_resolution": "latest_reviewed_run_wins",
  "initial_rating": 1500,
  "k_factor": 24,
  "epsilon": 0.25,
  "ordering_rule": "canonical_chapter_then_paragraph_then_unit_then_pair",
  "character_count": 0,
  "match_count": 0,
  "draw_rate": 0.0,
  "characters": []
}
```

Suggested per-character row:

```json
{
  "character": "Swann",
  "elo": 1472.3,
  "match_count": 188,
  "win_count": 72,
  "loss_count": 97,
  "draw_count": 19,
  "unit_count": 263,
  "mean_advantage_net_score": -0.84,
  "top_positive_unit": {
    "unit_id": "v2-p1-autour-de-mme-swann#p-17-p-21",
    "net_score": 1.4
  },
  "top_negative_unit": {
    "unit_id": "v1-p2-un-amour-de-swann#p-88-p-92",
    "net_score": -2.3
  }
}
```

## Markdown Companion

The markdown artifact should be readable as an analysis surface, not just a dump.

Recommended sections:

1. methodology summary
2. top-rated characters
3. lowest-rated characters
4. highest-match-count characters
5. largest rating-minus-mean-score mismatches
6. full character table

The most interesting section may be:

- characters whose ELO rank differs sharply from simple mean score rank

That would reveal whether repeated pairwise local “wins” tell a different story from raw aggregate net totals.

## Interpretive Expectations

This ELO surface should answer a narrower question than the existing aggregate reports.

It is not:

- “who is most positive overall?”

It is closer to:

- “who repeatedly comes out ahead of the other scored figures sharing the same scenes?”

That means the ELO surface may diverge from existing net-score rankings in meaningful ways.

For example:

- a character with many mildly positive relative outcomes could outrank a character with fewer but larger isolated gains
- a character with broad repeated social defeats could rate especially poorly even if some individual units are strongly positive

That divergence is a feature, not necessarily a bug.

## Known Limits

This method has real limitations.

### 1. Not all scenes are zero-sum

Multiple characters can all lose or all gain in a unit.

Pairwise comparison forces relative ordering even when the scene is not naturally competitive.

### 2. Units with many characters create many matches

Large scenes contribute disproportionately to the rating system.

That may be acceptable, but it should remain visible in diagnostics.

### 3. Lens-specific meaning

This plan is most natural for `advantage`.

It is less obviously natural for:

- `prestige`
- `inclusion`

Those can still be done later, but they should be treated as extensions, not assumed equivalents.

### 4. Path dependence

Elo ratings depend on match order.

The deterministic ordering solves reproducibility, not ontology.

## Reasons To Start With Advantage

`advantage` is the best first lens because it already means:

- who comes out ahead or behind in the immediate scene

That is the closest thing in the project to a literal competitive outcome.

By contrast:

- `prestige` is more field-level and reputational
- `inclusion` is more relational and affiliative

Those may still be fruitful later, but they are less intuitive as “match” surfaces.

## Success Criteria For V1

The first ELO artifact should be considered successful if:

1. the ranking is reproducible
2. the top and bottom characters are interpretable against known corpus patterns
3. the result adds something not already obvious from net-score totals alone
4. the diagnostics make it clear where the method is strong and where it is strained

The first artifact should not be considered successful merely because:

- it produces a neat ranking

## Default Next Step

If work proceeds on this plan, the next concrete task should be:

1. implement `build_character_elo(...)` for the `advantage` lens only
2. generate the JSON and markdown artifacts
3. compare the resulting top and bottom ratings with:
   - `character-cross-lens-current`
   - `character-annotation-counts-current`
4. decide whether the method is strong enough to extend to `prestige` and `inclusion`

## Prestige Extension (2026-08-06)

`build_character_elo` and `build_character_elo_timeline` now accept any lens in
`scoring.SCORING_LENS_ORDER`, and the machinery has been extended to `prestige`.

The match rule is unchanged. Within a unit, every pair of scored characters is still
compared by `net_score` under the chosen lens, with the same `epsilon = 0.25` draw
band and the same Elo update. Nothing about match construction, ordering, or the
rating algorithm is lens-specific.

What changes is interpretation, not mechanics. `advantage` is a scene-level tactical
read: who came out ahead in the immediate exchange. `prestige` is field-level and
reputational: it tracks visible social standing rather than momentary tactical
position. A prestige "win" in a unit means this character's visible standing moved up
relative to the co-present character in that passage -- closer to a move in the
Faubourg's ongoing reputational tournament (who is received where, whose name carries
weight, whose judgment is deferred to) than to a skirmish outcome. Two characters can
both "lose" a scene tactically (`advantage`) while one of them still gains standing
from simply being seen in the room (`prestige`) -- that divergence is expected and is
part of why the two lenses are tracked separately rather than combined.

Schema-wise, the extension is additive. Every per-character ELO row now carries a
lens-generic `mean_net_score` field; for `advantage` specifically, the row also keeps
the original `mean_advantage_net_score` key as a deprecated alias with the same value,
so the committed advantage artifacts remain a strict subset of the new schema. The
same pattern applies to the timeline's per-point `net_score` / `label` fields versus
the original `advantage_net_score` / `advantage_label` keys. Version strings follow
the lens: `character_elo_{lens}_v1` and `character_elo_timeline_{lens}_v1`, except
that the `advantage` lens keeps its original strings
(`character_elo_advantage_v1`, `character_elo_timeline_v1`) exactly, so nothing about
the already-committed advantage artifacts changes.

`advantage` remains the default lens everywhere -- the CLI's `--lens` flag on
`character-elo`, `character-elo-timeline`, and `elo-supplement-diff` defaults to
`advantage`, and default output filenames for `advantage` are unchanged
(`character-elo-advantage-current.*`, `character-elo-advantage-supplemented-current.*`,
`character-elo-advantage-timeline-supplemented-current.*`,
`character-elo-supplement-diff-current.*`). Other lenses get parallel filenames with
the lens name in place of `advantage` (e.g. `character-elo-prestige-current.*`), except
that the diff artifact for `advantage` has no lens component in its name at all
(`character-elo-supplement-diff-current.*`) while other lenses get one
(`character-elo-prestige-supplement-diff-current.*`) -- an asymmetry inherited
directly from the pre-existing advantage filenames, which were never renamed.

`inclusion` is not generated as a production artifact yet, but the same code path
supports it without further changes, since the guard now accepts any lens in
`SCORING_LENS_ORDER` rather than only `advantage`.

## Inclusion Extension (2026-08-06)

The `inclusion` artifact family completes the triptych, generated from the same
supplemented corpus surface (`1801` matches, `70` characters):

- `outputs/character-elo-inclusion-current.{json,md}` (baseline)
- `outputs/character-elo-inclusion-supplemented-current.{json,md}`
- `outputs/character-elo-inclusion-timeline-supplemented-current.{json,md}`
- `outputs/character-elo-inclusion-supplement-diff-current.{json,md}`

Interpretation: an inclusion "win" means the character's belonging or acceptance
moved up relative to a co-present character in that passage. It is the most
relational of the three lenses — closer to "who is being absorbed by the room"
than to who wins it or who outranks whom.

Cross-lens reading (characters with `>= 30` matches in all three lenses):

- the extremes are stable across the whole triptych (Jupien/Aimé/Elstir/
  le narrateur/Gilberte high; Charlus/Brichot/Swann/Saniette low), which is the
  triptych's own robustness check
- the diagnostic signal is divergence: `Odette` is the corpus's clearest
  prestige-without-belonging case (prestige rank `9` vs inclusion rank `18`
  among reliable characters) — standing rises while absorption lags; `Gilberte`
  is the inverse pattern (inclusion `2`), absorbed by every world she enters;
  `Bloch` carries the worst mean inclusion (`-1.072`) of any reliable character,
  the corpus's most-snubbed aspirant
- as with prestige, the pairwise-forcing caveat applies: co-presence is not
  competition; a large salon scene creates more inclusion pressure than an
  intimate one

## Glicko-2 Extension (2026-08-07)

A Glicko-2 rating surface now exists alongside the ELO artifacts
(`proust/glicko2.py`, `character-glicko2-{lens}-supplemented-current.{json,md}`,
`python -m proust character-glicko2 --lens ... --include-supplements`).

Why Glicko-2 rather than TrueSkill: public-domain exact specification
(implemented per Glickman's paper, with the paper's worked example as a
regression test), a straightforward stdlib implementation, and an unchanged
match definition — the same pairwise-with-epsilon-draws structure the ELO uses.
Parameters: initial `1500/350/0.06`, `tau 0.5`, `epsilon 0.25`.

Design choices:

- **rating periods are canonical chapters**: all matches within a chapter use
  opponents' pre-chapter state and update simultaneously, removing within-chapter
  order dependence
- **provisional means uncertain, not merely under-matched**: a character is
  provisional when `RD > 100`; ranked listings sort by the conservative rating
  (`rating - 2*RD`). This supersedes the presentation-level min-match cutoff
  with a principled criterion; it is stricter (it also flags concentrated
  low-match figures like Saniette) while legitimately admitting low-match,
  low-variance figures like Aimé and Jupien to the ranked set

Findings worth keeping:

- Aimé and Jupien, excluded from ranked standings under the 30-match rule
  despite topping raw ELO, are non-provisional under RD and lead the
  conservative-rating standings
- high-volume characters (Swann, Charlus, Albertine, Françoise) rank notably
  better under Glicko-2 than under ELO: sequential fixed-K ELO accumulates
  path-dependent drift from early matches against unstabilized opponents,
  while Glicko-2's per-period batching and `g(phi)` down-weighting of
  uncertain opponents suppress that drift. The directional story is unchanged
  (Swann remains the lowest-rated major figure, at `1357 ± 83` on `328`
  matches), but ELO's exact rank order in the mid-table should be treated as
  the noisier of the two readings
- the reader-facing standings consume the Glicko-2 surface; the per-passage
  ELO timeline remains the dossier journey chart (chapter-period Glicko is too
  coarse for the narrative arc view)

## Whole-History Rating Extension (2026-08-08)

WHR (Coulom 2008) is now the unified rating surface (`proust/whr.py`,
`proust/character_whr.py`, `character-whr` / `character-whr-timeline` CLI,
`character-whr-{lens}-supplemented-current.*` artifacts): per-character rating
trajectories over narrative time (`cumulative_unit_index`), a Wiener-process
prior (`w2 = 5` Elo²/unit, chosen by one-step-ahead predictive log-loss over
`{5, 15, 35, 60}`), Bradley-Terry likelihood with half-win/half-loss draws,
tridiagonal Newton fitting, and posterior bands (`± 2 sigma`). Two modes:

- **smoothed** ("in retrospect"): full-history MAP — every moment informed by
  the whole novel
- **filtered** ("as you read"): prefix fits — each moment informed only by what
  the reader has seen

Validation findings (recorded so no surface ever overclaims):

1. **Sequential ELO remains the best one-step-ahead predictor** (log-loss
   0.672 vs WHR-filtered 0.706 vs Glicko-2 0.728, advantage lens; same
   ordering on all lenses; result survives freezing ELO at unit boundaries and
   giving WHR variance-aware predictions). WHR's MAP is overconfident on a
   coarse signal with a 16% draw rate; ELO's bounded step is effective
   shrinkage. WHR's case is the trajectory and the band, not accuracy.
2. Final WHR ratings agree with Glicko-2 at Pearson `0.986` / Spearman `0.981`
   (non-provisional intersection), so the standings story is stable across
   systems.
3. Bands do not narrow monotonically: the narrator's filtered band collapses
   `196 -> 78` across v1->v2, then all bands widen modestly through v5-v7
   because trajectory-end nodes are one-sidedly supported and late appearance
   density thins. Reader-facing copy must not present end-of-book widening as
   growing interpretive ambiguity; the honest gloss is that the book ends and
   evidence stops accumulating.
4. The two modes diverge exactly where retrospection should: at settled nodes
   the largest gaps are Odette in *Un amour de Swann* (filtered `1311 ± 195`
   vs smoothed `1507` — the whole-history fit refuses the courtship-era
   verdict because she ends as Mme de Forcheville), Charlus's Balbec entrance
   (filtered `1686` vs smoothed `1542`), and the duchesse named inside Swann's
   story (`1415` vs `1557`). Filtered tracks the scene; smoothed tracks the
   book.

Unification decision (reviewed 2026-08-08): the reader-facing standings and
journey charts both consume WHR — standings from final smoothed ratings
(conservative = rating − band, provisional when band > 200), journeys from the
trajectories with a mode toggle. ELO and Glicko-2 remain as artifacts and as
the write-up's comparison points; ELO additionally remains the best forecaster
and should be cited as such.

For the eventual formal write-up: the chosen estimator is itself an act of
retrospection — a smoothing method in which the whole history revises every
moment, applied to a novel whose subject is exactly that revision. The method
agrees with its material; this belongs in the write-up as more than a
footnote.
