# Character ELO Plan

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
- `character_normalization`

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
  "character_normalization": {
    "applied": true,
    "map": {}
  },
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
