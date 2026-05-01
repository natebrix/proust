# Chapter Summary Export Plan

This document defines the chapter-facing app export after:

- character profile cards
- character pages
- chapter overlays

Its purpose is to provide the missing middle layer between:

- corpus-level character interpretation
- paragraph-range overlay evidence

The target is a chapter-keyed export that supports:

- chapter framing banners
- chapter sidebars
- chapter landing summaries
- blog-friendly chapter interpretation

## Why `v1` Was Not Enough

The first chapter-summary export was structurally useful, but it leaned too hard on report-like statistics:

- the prose summary described the export rather than the chapter
- chapter interpretation was too dependent on character rows
- character-in-chapter percentiles were not very intuitive for readers

What we want instead is a chapter-centered surface:

1. a prose summary about the chapter itself
2. a tonal archetype for the chapter
3. chapter-level lens densities and chapter-vs-chapter ranks
4. top characters by chapter impact mass
5. distinguishing passages

## Proposed Artifact

Recommended artifact family:

- `outputs/chapter-summaries-current.json`
- optionally `outputs/chapter-summaries-current.md`

Version target:

- `chapter_summary_export_v2`

## Core Model

Each chapter should package five things:

1. `summary`
2. `tonal_archetype`
3. `lens_profile`
4. `top_characters`
5. `distinguishing_passages`

## Lens Density Model

For each chapter and each lens, we compute two chapter-level values:

- `signed_density = sum(net_scores) / unit_count`
- `intensity_density = sum(abs(net_scores)) / unit_count`

`signed_density` answers:

- does this chapter tilt positive or negative in advantage, prestige, or inclusion?

`intensity_density` answers:

- how strongly is that lens activated in the chapter, regardless of sign?

This keeps the metric length-invariant across short and long chapters.

## Tonal Archetypes

The tonal archetype is based on whether each lens has `intensity_density` above the median chapter for that lens.

Lens order:

- `advantage`
- `prestige`
- `inclusion`

Archetype map:

- `000` → `Diffuse`
- `100` → `Confrontational`
- `010` → `Ceremonial`
- `001` → `Intimate`
- `110` → `Competitive`
- `101` → `Volatile`
- `011` → `Social`
- `111` → `Totalizing`

This is an intensity classification, not a polarity classification. Direction remains a separate field in `lens_profile`.

## Recommended `v2` Shape

```json
{
  "chapter_summary_export_version": "chapter_summary_export_v2",
  "character_normalization": {
    "applied": true
  },
  "intensity_medians": {
    "advantage": 3.22,
    "prestige": 2.88,
    "inclusion": 3.41
  },
  "chapters": [
    {
      "chapter_id": "v3-p1",
      "chapter_title": "Le Côté de Guermantes I",
      "reader_link": "/projects/islt/fr-original/v3-p1",
      "unit_count": 132,
      "summary": "The chapter reads as a totalizing social field, with negative inclusion and positive prestige pressure doing most of the work. Robert de Saint-Loup, duchesse de Guermantes, and Mme de Villeparisis carry the largest share of the chapter's social movement.",
      "tonal_archetype": {
        "label": "Totalizing",
        "intense_lenses": ["advantage", "prestige", "inclusion"],
        "intensity_signature": {
          "advantage": true,
          "prestige": true,
          "inclusion": true
        }
      },
      "lens_profile": {
        "advantage": {
          "net_score_total": -12.4,
          "signed_density": -0.094,
          "intensity_density": 4.221,
          "direction": "negative",
          "chapter_rank": 4,
          "chapter_percentile": 77,
          "intensity_above_median": true
        }
      },
      "top_characters": [
        {
          "character": "Robert de Saint-Loup",
          "unit_count": 44,
          "impact_mass": 41.587,
          "dominant_lens": "inclusion",
          "lens_signature": "advantage negative, prestige positive, inclusion negative",
          "advantage": { "net_score": -9.402 },
          "prestige": { "net_score": 5.885 },
          "inclusion": { "net_score": -26.3 }
        }
      ],
      "distinguishing_passages": [
        {
          "unit_id": "v3-p1#p-71-p-75",
          "paragraph_start": 71,
          "paragraph_end": 75,
          "reader_link": "/projects/islt/fr-original/v3-p1#p-71",
          "summary": "Robert de Saint-Loup gains prestige while losing inclusion.",
          "dominant_character": "Robert de Saint-Loup",
          "impact_mass": 8.441,
          "lens_signature": {
            "advantage": "negative",
            "prestige": "positive",
            "inclusion": "negative"
          }
        }
      ]
    }
  ]
}
```

## Field Notes

### `summary`

This should describe the chapter as a social field, not the export.

It should usually answer:

- what kind of pressure organizes the chapter
- which characters absorb the most movement
- whether there is a strong split figure worth naming

The summary should stay short and deterministic.

### `tonal_archetype`

This gives the chapter a compact shorthand:

- low-intensity everywhere → `Diffuse`
- high-intensity everywhere → `Totalizing`
- strong prestige plus inclusion without high advantage → `Social`

This is intended as a reader-friendly interpretive hook, closer to a chapter mood or field type than a technical metric.

### `lens_profile`

This is the chapter-level lens table.

Each lens should expose:

- `net_score_total`
- `signed_density`
- `intensity_density`
- `direction`
- `chapter_rank`
- `chapter_percentile`
- `intensity_above_median`

Reader-facing surfaces should emphasize:

- direction
- density rank

Raw totals are still useful as supporting detail.

### `top_characters`

These should no longer be sorted by unit count alone or displayed with chapter-local percentiles.

Recommended rule:

- sort by descending `impact_mass`
- break ties by descending `unit_count`

Where:

- `impact_mass = sum(abs(advantage) + abs(prestige) + abs(inclusion))`

These rows answer:

- which characters are most socially moved by this chapter?

### `distinguishing_passages`

These should be the strongest candidate units for inline links, excerpt callouts, or blog citations.

Recommended rule:

- sort units by descending passage `impact_mass`
- keep a short limit, usually `3` to `5`

Each row should include:

- unit id
- paragraph span
- reader link
- summary
- dominant character
- passage impact mass
- per-lens direction signature

## Relationship To Existing Artifacts

This layer should sit between:

- `character-pages-current.json`
- `chapter-overlays-current/`

So the interpretive stack becomes:

- character pages explain figures across the corpus
- chapter summaries explain the chapter as a field
- chapter overlays expose paragraph-range evidence

## Display Policy

Recommended reader-facing policy:

- use `advantage / prestige / inclusion`
- use chapter-level rank and direction for lens summaries
- avoid character-in-chapter percentiles as the primary display
- use impact mass and prose summaries for chapter character panels

## Implementation Checkpoint

The next completed checkpoint is:

1. implement `chapter_summary_export_v2`
2. regenerate current chapter summaries
3. update the ISLT handoff doc to the new contract
4. validate on:
   - `v1-p1-combray`
   - `v2-p1-autour-de-mme-swann`
   - `v3-p1`
   - `v7-p2-m-de-charlus-pendant-la-guerre`
