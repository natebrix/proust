# Chapter Overlay Schema

This document defines the current app-facing JSON schema for chapter-level annotation overlays in the `islt` reader.

The goal is:

- one stable minimal export for inline reader overlays
- keyed by canonical chapter id
- built from the accepted canonicalized corpus surface
- sufficient for paragraph shading, lens toggles, and character chips

The schema started as a narrow structural `v1` export. The current artifact is `chapter_overlay_v2`, which keeps the `v1` structure and adds deterministic prose summaries.

Related documents:

- [islt_app_integration_ideas.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_app_integration_ideas.md:1)
- [character_profile_card_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_profile_card_schema.md:1)

Primary source data for the exporter:

- canonicalized `build_outcome_report(...)` unit timelines
- canonical chapter metadata from [export.py](/Users/nathan_brixius/dev/proust/proust/export.py:1)

## Design Principles

The overlay schema should be:

- chapter-oriented
- paragraph-range-based
- canonicalized by reviewed same-person identity
- explicit about all three scoring lenses
- small enough for direct app consumption

The structural layer should not require:

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
  "chapter_overlay_version": "chapter_overlay_v2",
  "source_review_version": "corpus_sanity_review_v1",
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
  "chapter_overlay_version": "chapter_overlay_v2",
  "chapterId": "v1-p2-un-amour-de-swann",
  "chapterNumber": 2,
  "title": "Un amour de Swann",
  "volumeNumber": 1,
  "volumeTitle": "Du côté de chez Swann",
  "partNumber": 2,
  "partTitle": "Un amour de Swann",
  "sectionTitle": null,
  "summary": "This chapter contains 235 annotated units, centered on Swann, Odette, and Mme Verdurin. Overall it is inclusion loss-heavy, advantage loss-heavy, and prestige loss-heavy.",
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
          "advantage": {
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
      ],
      "summary": "Odette loses social status in advantage and inclusion; shows mixed social status in prestige."
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
- `summary` in `v2`

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

- choose the character with the largest absolute `advantage` net score in the unit
- break ties by absolute `prestige`, then absolute `inclusion`, then character name

This is only a convenience field. The app should still receive the full `characters` list.

### `characters`

Array of normalized per-character entries for the unit.

Each entry should contain:

- `character`
- `dominantStatusDimension`
- `advantage`
- `prestige`
- `inclusion`

## Required Per-Lens Fields

Each lens object should contain:

- `netScore`
- `label`

Example:

```json
{
  "advantage": {
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

## Summary Fields In V2

`chapter_overlay_v2` adds deterministic prose summaries while keeping the `v1` structural contract intact.

### Chapter `summary`

Short aggregate sentence for the chapter.

Current behavior:

- reports annotated unit count
- names the most common dominant characters
- describes whether each lens is broadly win-heavy, loss-heavy, mixed, or balanced

### Unit `summary`

Short deterministic sentence for the unit.

Current behavior:

- uses the highest-salience one or two non-neutral characters
- describes whether they gain, lose, or split by lens
- uses the dominant status dimension as the prose noun phrase

## Sorting Rules

To keep the app logic simple, exported arrays should already be stably sorted.

Recommended sorting:

- manifest `chapters`: canonical chapter order
- chapter `units`: by `paragraphStart`, then `paragraphEnd`, then `unitId`
- unit `characters`: descending maximum absolute net score across the three lenses, then character name

## V1 And V2

`chapter_overlay_v1` deliberately omitted prose summaries.

That kept the first exporter:

- it keeps the exporter purely structural and deterministic
- it lets the app team build rendering against a stable minimal contract first

`chapter_overlay_v2` is the additive follow-on:

- same manifest and chapter structure
- same per-unit lens data
- added deterministic chapter and unit summaries

The summaries remain additive rather than structural, so rendering can still rely on the original `v1` data shape if needed.
