# Character Page Schema

This document defines a first app-facing JSON schema for full character pages in the `islt` reader.

The goal is:

- one stable derived JSON artifact per normalized character
- built on top of existing analysis outputs rather than new scoring logic
- explicit about computed versus editorial fields
- easy for the `islt` app to render using its existing conventions

This schema is a page-level extension of the existing profile-card work.

Related documents:

- [character_profile_card_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_profile_card_schema.md:1)
- [chapter_overlay_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/chapter_overlay_schema.md:1)
- [islt_app_integration_ideas.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_app_integration_ideas.md:1)

Primary source artifacts:

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)
- [character-annotation-counts-current.json](/Users/nathan_brixius/dev/proust/outputs/character-annotation-counts-current.json:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

Portrait source directory:

- `/Users/nathan_brixius/dev/brixius-web/public/projects/islt/portraits`

## Design Principles

The first schema should be:

- character-centric
- normalized by reviewed identity
- strong enough to power a real page, not just a card
- explicit about what is computed versus what is editorial
- portrait-aware without hard-coding rendering choices

The first schema should not require:

- recomputing analysis inside the app
- sentence-level span mapping
- new prompt calls
- the app reading raw `run-*` directories

## Recommended Output File

Suggested artifact names:

- `outputs/character-pages-current.json`
- optionally `outputs/character-pages-current.md`

## Top-Level Shape

Recommended top-level JSON shape:

```json
{
  "character_pages_version": "character_pages_v1",
  "source_review_version": "corpus_sanity_review_v1",
  "character_normalization": {
    "applied": true,
    "map": {
      "Charlus": "baron de Charlus"
    }
  },
  "character_count": 5,
  "pages": [
    {
      "...": "character page"
    }
  ]
}
```

## Required Page Fields

Each page should contain these required fields.

```json
{
  "character": "Odette",
  "slug": "odette",
  "portrait": {
    "default": "/projects/islt/portraits/odette-default-vermeer-proustian-20260425-1432.png",
    "variants": [
      {
        "variant": "default",
        "style": "vermeer-proustian",
        "src": "/projects/islt/portraits/odette-default-vermeer-proustian-20260425-1432.png"
      },
      {
        "variant": "prestige-radiant",
        "style": "vermeer-proustian",
        "src": "/projects/islt/portraits/odette-prestige-radiant-vermeer-proustian-20260425-1356.png"
      }
    ]
  },
  "profile": {
    "annotation_unit_count": 88,
    "rank_spread": 48,
    "max_score_span": 7.422,
    "selected_by": ["rank_spread", "volatility"],
    "lens_scores": {
      "advantage": {
        "net_score": -1.638,
        "rank": 28,
        "unit_count": 88,
        "dominant_status_dimension": "social_status",
        "score_span": 6.598,
        "mean_score": -0.019
      },
      "prestige": {
        "net_score": 15.86,
        "rank": 4,
        "unit_count": 88,
        "dominant_status_dimension": "social_status",
        "score_span": 7.422,
        "mean_score": 0.18
      },
      "inclusion": {
        "net_score": -23.464,
        "rank": 52,
        "unit_count": 88,
        "dominant_status_dimension": "social_status",
        "score_span": 6.155,
        "mean_score": -0.267
      }
    }
  },
  "editorial": {
    "dek": "Prestige-positive but inclusion-negative, with the split concentrated in Swann- and Guermantes-adjacent chapters.",
    "summary": "Odette is one of the sharpest cross-lens split figures in the corpus: she rises strongly in prestige while remaining much more unstable in belonging and immediate advantage.",
    "why_interesting": [
      "Her prestige and inclusion readings diverge far more than her raw frequency alone would predict.",
      "Her profile is driven by a small number of chapters rather than uniform treatment across the novel."
    ],
    "primary_pattern": "prestige_positive_inclusion_negative"
  },
  "top_chapters": [
    {
      "chapter_id": "v2-p1-autour-de-mme-swann",
      "chapter_title": "À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann",
      "advantage": {
        "net_score": 19.699,
        "unit_count": 32
      },
      "prestige": {
        "net_score": 24.138,
        "unit_count": 32
      },
      "inclusion": {
        "net_score": 8.419,
        "unit_count": 32
      },
      "reader_link": "/projects/islt/fr-original/v2-p1-autour-de-mme-swann"
    }
  ],
  "reading_path": [
    {
      "chapter_id": "v2-p1-autour-de-mme-swann",
      "label": "Prestige ascent around Mme Swann",
      "reader_link": "/projects/islt/fr-original/v2-p1-autour-de-mme-swann"
    },
    {
      "chapter_id": "v1-p2-un-amour-de-swann",
      "label": "Negative counterweight in Swann's love",
      "reader_link": "/projects/islt/fr-original/v1-p2-un-amour-de-swann"
    }
  ]
}
```

## Field Groups

### `character`

Normalized character identity key.

This should always be the reviewed aggregate-layer name, not an unnormalized alias.

### `slug`

Stable app-facing identifier for URLs, asset matching, and local page routing.

This should usually be:

- lowercase
- ASCII slug
- derived from the normalized character name

Examples:

- `Odette` -> `odette`
- `baron de Charlus` -> `baron-de-charlus`
- `Robert de Saint-Loup` -> `robert-de-saint-loup`

### `portrait`

Portrait metadata for the page.

This should not encode rendering decisions such as layout, cropping, or which variant the app must prefer visually. It should only provide the available assets cleanly.

Required subfields:

- `default`
- `variants`

Each `variants` row should include:

- `variant`
- `style`
- `src`

The current portrait directory suggests the app can expect patterns like:

- `odette-default-vermeer-proustian-20260425-1432.png`
- `odette-prestige-radiant-vermeer-proustian-20260425-1356.png`
- `swann-default-elstir-20260425-1432.png`

That means the page schema should preserve:

- character
- variant
- style

without forcing the app to reverse-engineer filenames.

### `profile`

This is the computed analysis core.

It should largely embed the existing character-profile-card data unchanged:

- `annotation_unit_count`
- `rank_spread`
- `max_score_span`
- `selected_by`
- `lens_scores`

This keeps the page artifact aligned with the existing card artifact instead of inventing a second incompatible representation.

### `editorial`

This is the human-readable interpretation layer.

Required subfields for `v1`:

- `dek`
- `summary`
- `why_interesting`
- `primary_pattern`

This is where the page stops being just analytics and starts being literary framing.

### `top_chapters`

This should carry the highest-value chapter drivers for the character.

The practical rule should remain:

- sort chapters by maximum absolute net score across the three lenses
- keep the top `3` to `5`

Recommended subfields:

- `chapter_id`
- `chapter_title`
- `advantage`
- `prestige`
- `inclusion`
- `reader_link`

### `reading_path`

This is the most app-specific but also one of the most useful fields.

It should be a short curated list of suggested reading destinations for the page.

Each row should include:

- `chapter_id`
- `label`
- `reader_link`

This gives the app a way to say “where should the reader go next?” without needing new logic.

## Computed Versus Editorial Boundary

The page schema should keep this split explicit.

Computed fields:

- `character`
- `slug`
- `portrait`
- `profile`
- `top_chapters`

Editorial fields:

- `editorial`
- `reading_path`

This matters because the computed layer can be regenerated mechanically, while the editorial layer should be easy to review and revise without disturbing the numeric structure.

## Suggested Portrait Handling

The page artifact should not assume a single portrait style forever.

A good first rule is:

- include all discovered portrait variants for the character
- mark one image as `default`

That lets the `islt` app follow its own conventions while still having:

- fallback image stability
- optional style switching later
- optional variant-specific use for especially split figures like `Odette`

## Optional Enrichment Fields

These are useful, but not required for `v1`.

### `chapter_count`

Number of chapters in which the character appears.

Useful for distinguishing:

- broad corpus presence
- concentrated chapter-driven figures

### `notable_units`

Curated links to specific overlay units.

Example:

```json
{
  "notable_units": [
    {
      "unit_id": "v2-p1-autour-de-mme-swann#p-101-p-105",
      "label": "Prestige-heavy salon rise",
      "reader_link": "/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-101"
    }
  ]
}
```

This is useful, but it is more precise than the first page schema strictly requires.

### `related_characters`

Small list of associated figures for exploration.

Examples:

- `Swann` for `Odette`
- `Morel` for `baron de Charlus`
- `Albertine` for `Swann`

This is a strong future app feature, but should remain optional in `v1`.

## Suggested Sorting

The top-level `pages` array should probably be sorted by:

1. descending `annotation_unit_count`
2. then descending `rank_spread`
3. then character name

That keeps the page dataset aligned with both:

- importance by corpus footprint
- importance by interpretive interest

## Recommended First Characters

The first pages should be written for a small high-signal set:

- `Odette`
- `Robert de Saint-Loup`
- `Swann`
- `Albertine`
- `baron de Charlus`

This gives the first page set a mix of:

- strong cross-lens split
- broad annotation footprint
- concentrated chapter drivers
- familiar literary centrality

## Recommended Next Implementation Step

Implement a `character_pages_v1` artifact that:

1. embeds the existing profile-card analysis
2. attaches chapter-driver rows and reader links
3. maps each character to discovered portrait assets
4. adds a small reviewed editorial layer for the pilot characters

That would produce a real page-ready dataset without asking the app to invent either the analysis layer or the interpretation layer on its own.
