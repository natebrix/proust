# ISLT Chapter Summaries Handoff

This document is a concrete engineering brief for a partner Codex instance working in:

- `/Users/nathan_brixius/dev/brixius-web`

The goal is to integrate the new chapter-summary export into the existing `islt` app with minimal recomputation in the web layer.

## Objective

Add a chapter-level framing layer to `islt` chapter pages using the exported Proust data product:

- [chapter-summaries-current.json](/Users/nathan_brixius/dev/proust/outputs/chapter-summaries-current.json:1)

This layer should sit between:

- existing chapter reader text
- existing or planned unit overlays
- existing character pages

In practice, this means:

- a compact chapter banner or summary panel
- top chapter characters
- chapter lens profile
- distinguishing passages

## Current App Seams

These are the most relevant current `islt` files:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt/[edition]/[chapter]/page.tsx`
- `/Users/nathan_brixius/dev/brixius-web/components/islt-reader.tsx`
- `/Users/nathan_brixius/dev/brixius-web/lib/islt.ts`

What they currently do:

- `lib/islt.ts`
  - loads canonical ISLT chapter data from `data/islt`
  - exports `getIsltChapter(...)`
  - defines the current chapter and edition types

- `app/projects/islt/[edition]/[chapter]/page.tsx`
  - is the main chapter route
  - already loads the chapter payload through `getIsltChapter(...)`
  - renders the chapter shell, title, nav, and `IsltReader`

- `components/islt-reader.tsx`
  - renders the paragraph and sentence text
  - already supports paragraph anchors and sentence highlighting
  - should probably stay focused on prose rendering rather than chapter-summary aggregation

## New Input Data

Primary input:

- `/Users/nathan_brixius/dev/proust/outputs/chapter-summaries-current.json`

Supporting inputs already in use or likely to matter next:

- `/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json`
- `/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json`
- `/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json`

Reference docs:

- [chapter_summary_export_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/chapter_summary_export_plan.md:1)
- [islt_app_integration_ideas.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_app_integration_ideas.md:1)
- [chapter_overlay_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/chapter_overlay_schema.md:1)
- [character_page_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_page_schema.md:1)

## Data Contract Notes

The chapter summary export is:

- chapter-keyed
- normalized on character identity
- reader-facing
- chapter-density aware

Each chapter row includes:

- `chapter_id`
- `chapter_title`
- `reader_link`
- `unit_count`
- `tonal_archetype`
- `lens_profile`
- `top_characters`
- `strongest_split_character`
- `distinguishing_passages`
- `summary`

Each `top_characters` row includes:

- `character`
- `unit_count`
- `impact_mass`
- `dominant_lens`
- `lens_signature`
- `advantage.net_score`
- `prestige.net_score`
- `inclusion.net_score`

Each `lens_profile` row includes:

- `net_score_total`
- `signed_density`
- `intensity_density`
- `direction`
- `chapter_rank`
- `chapter_percentile`
- `intensity_above_median`

Each `distinguishing_passages` row includes:

- `unit_id`
- `paragraph_start`
- `paragraph_end`
- `reader_link`
- `summary`
- `dominant_character`
- `impact_mass`
- `lens_signature`

## Important Display Policy

The project now treats:

- `advantage`
- `prestige`
- `inclusion`

as the canonical lens names.

For reader-facing displays:

- use chapter-level direction and rank as the primary lens summary
- use `impact_mass` for top chapter characters
- keep raw scores as secondary detail
- avoid character-in-chapter percentiles as the primary display

Examples:

- `negative inclusion, 3rd among chapters`
- `Robert de Saint-Loup carries the most impact mass in this chapter`

Avoid leading with values like:

- `15% advantage`

without clear context for what the number means.

## Recommended Integration Shape

### 1. Load The New Artifact In `lib/islt.ts`

Add a small loader for chapter-summary data.

Suggested approach:

- define a new type like `IsltChapterSummaryExport`
- define a per-chapter row type like `IsltChapterSocialSummary`
- add a cached JSON loader that reads:
  - `../proust/outputs/chapter-summaries-current.json`
  - or a copied local app-facing mirror, if that is the convention in `brixius-web`

Recommended helper names:

- `getIsltChapterSummaryExport()`
- `getIsltChapterSocialSummary(chapterId: string)`

The app should not recompute chapter analytics from raw annotation files.

### 2. Extend `app/projects/islt/[edition]/[chapter]/page.tsx`

This is the natural place for the new framing layer.

Suggested flow:

1. load the existing chapter via `getIsltChapter(...)`
2. load the chapter summary row via a new helper
3. render a chapter summary panel above `IsltReader`

The summary panel should likely live:

- below the chapter title/nav area
- above the full prose reader

### 3. Keep `IsltReader` Focused On Text

Do not push chapter-summary logic into `components/islt-reader.tsx` unless there is a clear need.

The reader component already has a clean responsibility:

- paragraph/sentence rendering
- paragraph anchors
- sentence highlighting

Chapter summary framing should stay one level up in the page shell.

## Minimal First Render

The first implementation does not need to be elaborate.

A good minimal banner should render:

- chapter `summary`
- tonal archetype
- chapter lens profile
- top `3` chapter characters by impact mass
- `2` to `3` distinguishing passages

That is enough to make the chapter legible as a social field without turning the page into a dashboard.

## Suggested Componentization

Likely new component:

- `components/islt-chapter-summary.tsx`

Possible props:

- chapter summary row
- optional active lens

Possible subsections:

- summary text
- tonal archetype
- lens profile
- top characters
- distinguishing passages

If the session wants to keep it even lighter, rendering inline in `page.tsx` is fine for a first pass.

## Recommended UI Priorities

1. show the deterministic chapter summary sentence
2. show the tonal archetype and chapter lens profile
3. show a compact `top_characters` row or table
4. show distinguishing passages

This order matches the data hierarchy:

- editorial summary first
- chapter field type second
- supporting structure third

## Good First Review Chapters

Use these to validate the output:

- `v2-p1-autour-de-mme-swann`
  - strong Odette-driven positive/negative contrast

- `v3-p1`
  - clear Guermantes-field structure with a real split figure

- `v5`
  - strongly negative chapter with concentrated Albertine pressure

- `v7-p2-m-de-charlus-pendant-la-guerre`
  - small but legible wartime field with compact negative concentration

## What The Partner Session Should Not Do

It should not:

- recompute lens totals from raw `run-*` data
- rename lenses again
- reintroduce character-in-chapter percentiles as the main display
- move chapter-summary logic into the prose rendering component without a clear need
- block on sentence-level evidence mapping

## Implementation Suggestion

Recommended sequence:

1. add a chapter-summary loader in `lib/islt.ts`
2. add a chapter-summary panel component
3. render it in `app/projects/islt/[edition]/[chapter]/page.tsx`
4. verify the four review chapters above
5. refine layout and copy density only after the data is visibly working

## Success Condition

The implementation is successful if a chapter page now gives the reader, before the prose begins:

- a compact statement of the chapter's social shape
- a visible clue about who matters most
- an immediate sense of which character is most split across lenses
- a path from chapter framing down into the text and over to character pages
