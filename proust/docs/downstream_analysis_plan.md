# Downstream Analysis Plan

This document records the default next analysis step after the normalized corpus-review pass.

It assumes:

- the accepted annotation JSON remains fixed
- the reviewed aggregate-layer character normalization is accepted
- the normalized corpus-review artifacts are now the default per-character aggregate surface

Primary inputs:

- [corpus-review-current-normalized.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.json:1)
- [corpus-review-current-normalized.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.md:1)
- [corpus-review-normalization-diff.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-normalization-diff.md:1)

General artifact map:

- [outputs_guide.md](/Users/nathan_brixius/dev/proust/proust/docs/outputs_guide.md:1)

## Immediate goal

The first lightweight downstream analysis artifact has now been generated and should be treated as the default character-level view of the corpus:

- [character-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.json:1)
- [character-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.md:1)

The next downstream step should extend that artifact rather than restart from scratch.

The first artifact should answer:

1. who the major aggregate winners and losers are under each lens
2. which characters move most across lenses
3. which characters are most internally volatile across units
4. which results are broad corpus-shape findings versus local high-variance phenomena

## Current checkpoint

The active next checkpoint is now:

- build and read a character-by-chapter cross-lens report for the highest-rank-spread and highest-volatility figures

That artifact has now been generated:

- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)
- [character-chapter-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.md:1)

It covers the union of the current top rank-spread and top volatility characters, including:

- `Odette`
- `Robert de Saint-Loup`
- `Gilberte`
- `Mme de Villeparisis`
- `Swann`
- `Albertine`
- `baron de Charlus`

The first app-facing derivative artifacts have now also been generated:

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

The chapter overlay surface now includes additive deterministic prose summaries as `chapter_overlay_v2`.

## Recommended first artifact

The first artifact is a character-centered downstream report.

Recommended sections:

- top positive and negative characters by lens
- one row per character with `advantage`, `prestige`, and `inclusion` net scores side by side
- cross-lens rank movement for the same character
- unit-count context for each character
- volatility summary for characters with repeated appearances

This should stay aggregate-first and should not reopen routine unit interpretation.

## Guardrails

- do not rewrite accepted annotation JSON
- do not add new alias heuristics beyond the reviewed explicit normalization map
- do not treat ranking movement alone as an interpretive finding without unit-count context
- do not re-open prompt, reducer, or schema tuning unless a new report-level problem appears

## Default next move

If work resumes from here, the default next move is:

1. read the generated character-by-chapter cross-lens artifact first
2. use it to identify which chapter-level accumulations are driving the largest lens splits
3. treat the character profile cards and `chapter_overlay_v2` overlays as the current app-facing export layer
4. if app-facing export work continues, prefer richer editorial framing over new structural export work
5. keep extending aggregate analysis before drilling into individual units
6. only drill into units if the chapter-level picture shows a genuinely surprising pattern
