# ISLT Character Pages Handoff

This document is the shortest transition note for a future session focused on rendering character pages in the separate `islt` app.

The purpose is:

- give the rendering session a clean starting point
- identify the current page-ready data products
- clarify what should be treated as fixed data contract versus UI interpretation

The `islt` app lives in:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt`

## Current Status

The annotation project now has the data pieces needed for a first pilot character-page implementation:

- cross-lens profile-card data
- chapter-driver data
- chapter overlay data with deterministic unit and chapter summaries
- portrait assets
- a pilot character-page schema

The annotation side should now be treated as supplying page-ready data rather than asking the app to reconstruct analysis from raw runs.

## Primary Inputs

For the rendering session, these are the most important source artifacts:

- [character_page_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_page_schema.md:1)
- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)
- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)

Related context:

- [islt_app_integration_ideas.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_app_integration_ideas.md:1)
- [outputs_guide.md](/Users/nathan_brixius/dev/proust/proust/docs/outputs_guide.md:1)

Portrait assets:

- `/Users/nathan_brixius/dev/brixius-web/public/projects/islt/portraits`

## Pilot Characters

The initial page set should stay intentionally small:

- `Odette`
- `Robert de Saint-Loup`
- `Swann`
- `Albertine`
- `baron de Charlus`

This set was chosen to cover:

- strong cross-lens split
- broad annotation footprint
- strongly negative concentration
- high volatility across several terrains

## What The App Should Assume

The app should assume:

- normalized character identity is canonical
- page data is derived from the accepted normalized corpus surface
- the page artifact already separates computed versus editorial layers
- chapter links should route into the existing French reader pages

The app should not assume:

- it needs to read raw `run-*` directories
- it needs to recompute lens totals
- it needs to derive page explainer copy from scratch

## Rendering Targets

A first pilot page should probably render:

- portrait
- character name
- short `subheading`
- longer summary paragraph
- lens score table or compact lens stat block
- top driving chapters
- reading path links
- notable unit links

This is enough for a meaningful first character page without trying to solve every future feature at once.

## Likely Route Shape

The app session can choose the exact route convention, but a likely page family would be something like:

- `/projects/islt/characters/odette`
- `/projects/islt/characters/robert-de-saint-loup`

The annotation side does not need to enforce this. It only needs to supply stable `slug` values.

## Portrait Handling

The page artifact is intended to provide:

- one `default` portrait
- a list of discovered `variants`

The rendering session should follow `islt` conventions for image display, but it should not need to reverse-engineer filenames.

This is especially useful for figures like `Odette`, where multiple meaningful portrait variants already exist.

## Recommended First Rendering Pass

The cleanest first rendering pass would be:

1. implement one reusable character-page component
2. wire it to the pilot artifact
3. render the five pilot characters
4. reuse existing `islt` typography, layout, and navigation conventions
5. add links from chapter rows into the existing reader pages

That keeps the session focused on actual page design and information architecture instead of data wrangling.

## What The Rendering Session Should Ignore

The rendering session should not spend time:

- redesigning the annotation export contract from scratch
- inventing a different character identity system
- solving sentence-level evidence linking
- reopening source-annotation logic

Those are separate problems and should not block the first character pages.

## Recommended Next Data Check

Before rendering starts, the session should verify the current character-page artifact exists and inspect:

- one split figure like `Odette`
- one frequent but stable figure like `Swann`
- one broad volatile figure like `baron de Charlus`

That should be enough to validate the data contract against real page needs.
