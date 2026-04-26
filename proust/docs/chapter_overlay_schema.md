# Chapter Overlay Schema

This document defines the first app-facing JSON schema for chapter-level annotation overlays in the `islt` reader.

The goal is:

- one stable minimal export for inline reader overlays
- keyed by canonical chapter id
- built from the accepted normalized aggregate surface
- sufficient for paragraph shading, lens toggles, and character chips

This schema is intentionally narrow. It is meant to support a clean `v1` exporter before adding editorial or prose enrichment.

Related documents:

- [islt_app_integration_ideas.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_app_integration_ideas.md:1)
- [character_profile_card_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_profile_card_schema.md:1)

Primary source data for the exporter:

- normalized `build_outcome_report(...)` unit timelines
- canonical chapter metadata from [export.py](/Users/nathan_brixius/dev/proust/proust/export.py:1)

## Design Principles

The first schema should be:

- chapter-oriented
- paragraph-range-based
- normalized by reviewed character identity
- explicit about all three scoring lenses
- small enough for direct app consumption

The first schema should not require:

- sentence-level span mapping
- new prompt calls
- generated prose summaries
- the web app reading raw `run-*` directories

## Recommended Output Layout

Suggested artifact layout:

- `outputs/chapter-overlays-current/manifest.json`
- `outputs/chapter-overlays-current/chapters/<chapter-id>.json`

This keeps chapter fetches simple in the app and avoids one very large monolithic JSON file.

## Top-Level Manifest Shape

Recommended manifest shape:

```json
{
  "chapter_overlay_version": "chapter_overlay_v1",
  "source_review_version": "corpus_sanity_review_v1",
  "character_normalization": {
    "applied": true,
    "map": {
      "Charlus": "baron de Charlus"
    }
  },
  "chapter_count": 75,
  "duplicate_resolution": "latest_reviewed_run_wins",
  "chapters": [
    {
      "chapterId": "v1-p1-combray",
      "title": "Combray",
      "path": "chapters/v1-p1-combray.json",
      "unitCount": 17,
      "characterCount": 9
    }
  ]
}
```

Historical corpus overlaps are possible because later accepted runs may supersede earlier reviewed units.

For the overlay export, the manifest should therefore record:

- `duplicate_resolution: "latest_reviewed_run_wins"`

That gives the app one canonical inline surface per unit id.

## Chapter File Shape

Recommended chapter file shape:

```json
{
  "chapter_overlay_version": "chapter_overlay_v1",
  "chapterId": "v1-p2-un-amour-de-swann",
  "chapterNumber": 2,
  "title": "Un amour de Swann",
  "volumeNumber": 1,
  "volumeTitle": "Du côté de chez Swann",
  "partNumber": 2,
  "partTitle": "Un amour de Swann",
  "sectionTitle": null,
  "characterNormalizationApplied": true,
  "units": [
    {
      "unitId": "v1-p2-un-amour-de-swann#p-17-p-21",
      "paragraphStart": 17,
      "paragraphEnd": 21,
      "dominantCharacter": "Odette",
      "characters": [
        {
          "character": "Odette",
          "dominantStatusDimension": "social_status",
          "local": {
            "netScore": -1.638,
            "label": "loss"
          },
          "prestige": {
            "netScore": 1.245,
            "label": "mixed"
          },
          "inclusion": {
            "netScore": -2.214,
            "label": "loss"
          }
        }
      ]
    }
  ]
}
```

## Required Unit Fields

Each unit should contain:

- `unitId`
- `paragraphStart`
- `paragraphEnd`
- `dominantCharacter`
- `characters`

### `unitId`

Canonical annotation unit id, for example:

- `v1-p1-combray#p-17`
- `v7-p4-le-bal-de-tetes#p-121-p-125`

This is the stable bridge back to the annotation corpus.

### `paragraphStart` and `paragraphEnd`

Integer paragraph bounds extracted from `unitId`.

These are the direct app hooks for paragraph-range shading.

### `dominantCharacter`

Single character name chosen for compact UI defaults.

The first implementation should keep this rule simple:

- choose the character with the largest absolute `local` net score in the unit
- break ties by absolute `prestige`, then absolute `inclusion`, then character name

This is only a convenience field. The app should still receive the full `characters` list.

### `characters`

Array of normalized per-character entries for the unit.

Each entry should contain:

- `character`
- `dominantStatusDimension`
- `local`
- `prestige`
- `inclusion`

## Required Per-Lens Fields

Each lens object should contain:

- `netScore`
- `label`

Example:

```json
{
  "local": {
    "netScore": -2.6,
    "label": "loss"
  }
}
```

This is enough for:

- overlay color
- sign marker
- lens toggles
- compact chips

## Sorting Rules

To keep the app logic simple, exported arrays should already be stably sorted.

Recommended sorting:

- manifest `chapters`: canonical chapter order
- chapter `units`: by `paragraphStart`, then `paragraphEnd`, then `unitId`
- unit `characters`: descending maximum absolute net score across the three lenses, then character name

## Why No Prose Summary In V1

Prose summary is useful, but it is not required for the first inline overlay pass.

Leaving it out of `v1` has two advantages:

- it keeps the exporter purely structural and deterministic
- it lets the app team build rendering against a stable minimal contract first

The first app version can already do useful work with:

- paragraph ranges
- lens labels
- character chips
- dominant character defaults

## Planned V2 Extension

After `chapter_overlay_v1` is working, the next additive step should be `chapter_overlay_v2` with optional prose summaries.

Recommended `v2` additions:

- unit-level `summary`
- optionally chapter-level `summary`

Example additive unit field:

```json
{
  "summary": "Narrator-led diminishment of Albertine's emotional standing."
}
```

That should remain additive rather than structural, so the app can adopt `v1` first without blocking on summary generation.

## Recommended Next Implementation Step

Implement a `build_chapter_overlay_data(...)` exporter that:

1. builds normalized per-lens outcome reports
2. groups unit timeline entries by chapter and `unitId`
3. emits one chapter JSON file per canonical chapter
4. writes a small manifest with version and chapter metadata

That will create the main missing data bridge between the annotation project and the `islt` reader.
