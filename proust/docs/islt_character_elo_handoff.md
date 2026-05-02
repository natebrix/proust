# ISLT Character ELO Handoff

This document is the shortest engineering brief for a frontend session that adds an ELO plot to the existing `islt` character pages.

The goal is narrow:

- keep the current character page intact
- add one reader-facing `advantage` ELO timeline
- use the exported data directly
- avoid recomputing analytics in the app

The `islt` app lives in:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt`

## Current Inputs

Primary data:

- [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)
- [character-elo-advantage-timeline-current.json](/Users/nathan_brixius/dev/proust/outputs/character-elo-advantage-timeline-current.json:1)

Related context:

- [islt_character_pages_handoff.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_character_pages_handoff.md:1)
- [character_elo_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/character_elo_plan.md:1)

Reference plots already generated from the same artifact:

- [swann-advantage-elo-timeline.png](/Users/nathan_brixius/dev/proust/outputs/plots/swann-advantage-elo-timeline.png:1)
- [odette-advantage-elo-timeline.png](/Users/nathan_brixius/dev/proust/outputs/plots/odette-advantage-elo-timeline.png:1)
- [swann-odette-advantage-elo-comparison.png](/Users/nathan_brixius/dev/proust/outputs/plots/swann-odette-advantage-elo-comparison.png:1)

## What The App Should Assume

The app should assume:

- the lens is `advantage`
- the timeline is sparse
- a point exists only when that character appears in an annotated unit
- ratings begin from a flat common prior, so the shape and relative movement matter more than the absolute number
- `cumulative_unit_index` is the best first x-axis

The app should not assume:

- it needs to smooth the series
- it needs to fill gaps where the character is absent
- it needs to derive chapter boundaries from raw text files
- it needs to read `run-*` directories

## Data Shape

Top-level fields in `character-elo-advantage-timeline-current.json`:

- `character_elo_timeline_version`
- `lens`
- `timeline_type`
- `tracked_characters`
- `points`

Each point has:

- `character`
- `elo`
- `advantage_net_score`
- `advantage_label`
- `unit_character_count`
- `corpus_position`

Each `corpus_position` has:

- `volume_number`
- `chapter_id`
- `chapter_title`
- `chapter_index`
- `unit_id`
- `unit_index_within_chapter`
- `cumulative_unit_index`
- `paragraph_start`
- `paragraph_end`
- `cumulative_paragraph_index`
- `cumulative_paragraph_index_end`
- `cumulative_word_count`
- `cumulative_word_count_end`

## First Rendering Target

Add one compact timeline module to the character page.

Recommended placement:

- below the portrait and editorial summary
- above `Distinguishing Passages`

Recommended first display:

- one raw ELO line
- x-axis keyed to `cumulative_unit_index`
- light vertical markers for volume boundaries
- subtle baseline at `1500`
- hover state showing:
  - chapter title
  - unit id
  - ELO
  - `advantage_label`
  - `advantage_net_score`

This should read as:

- a compact history of how the character's immediate scene-level advantage changes across the novel

It should not read as:

- a stock price chart
- a replacement for the lens table
- a smoothed interpretive abstraction

## Display Policy

Use the ELO chart as a secondary interpretive element, not the page headline metric.

Good framing:

- `Advantage ELO Across ISLT`
- `Scene-level advantage over time`

Avoid:

- treating ELO as the character's final canonical rank
- explaining the chart with too much game vocabulary
- requiring the reader to understand Elo to benefit from the page

The clean mental model is:

- when the line rises, the character tends to come out ahead more often in the scenes where they appear
- when it falls, the character tends to come out behind

## Likely App Seams

Probable files to inspect first:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt/characters/[slug]/page.tsx`
- `/Users/nathan_brixius/dev/brixius-web/lib/islt.ts`
- any existing chart or data-loader utilities already used on character pages

Recommended split:

1. loader helper in `lib/islt.ts`
2. small `CharacterEloChart` component
3. wire that component into the character page

## Suggested Loader Contract

The app does not need the whole timeline at once on the page component.

A small helper should:

- load `character-elo-advantage-timeline-current.json`
- filter `points` to one character
- return:
  - `character`
  - `points`
  - derived `volumeStarts` keyed by `cumulative_unit_index`

That keeps chart code simple and avoids repeated filtering in render.

## Nice First Enhancements

These are optional after the first pass:

- point color by `advantage_label`
- chapter-boundary tooltips
- click-through from a point to the relevant reader unit
- shared comparison plot for selected pairs like `Swann` and `Odette`

Do not block the first implementation on these.

## Validation Characters

Use these to sanity-check the chart behavior:

- `Swann`
  - long decline with only partial later recovery
- `Odette`
  - strong rise, high middle peak, stable late plateau
- `duchesse de Guermantes`
  - strong upward trajectory
- `Albertine`
  - more negative late movement

If those look qualitatively wrong, the issue is probably:

- character filtering
- x-axis selection
- volume-boundary placement
- mishandling sparse points

## Short Prompt

Use something like this:

`Please add an ELO timeline to the existing ISLT character pages using ../proust/outputs/character-elo-advantage-timeline-current.json and ../proust/outputs/character-pages-current.json. Treat this as a compact secondary module on the character page, not a new page type. Use a raw unsmoothed line for the character's sparse advantage ELO points, keyed to cumulative_unit_index, with light volume markers and a 1500 baseline. Please read proust/docs/islt_character_elo_handoff.md and proust/docs/islt_character_pages_handoff.md first, and avoid recomputing analytics in the app.`
