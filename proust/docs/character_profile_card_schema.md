# Character Profile Card Schema

This document defines a first app-facing JSON schema for cross-lens character profile cards.

The goal is:

- one stable derived JSON artifact
- keyed by normalized character identity
- easy for the `islt` app to consume
- built from existing analysis outputs rather than new interpretive logic

Primary source artifacts:

- [character-annotation-counts-current.json](/Users/nathan_brixius/dev/proust/outputs/character-annotation-counts-current.json:1)
- [character-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.json:1)
- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)

## Design Principles

The first schema should be:

- character-centric
- normalized by reviewed identity
- small enough for direct app rendering
- explicit about what is computed versus editorial

The first schema should not require:

- sentence-level span mapping
- new prompt calls
- per-event reconstruction inside the app

## Recommended Output File

Suggested artifact names:

- `outputs/character-profile-cards-current.json`
- optionally `outputs/character-profile-cards-current.md`

## Top-Level Shape

Recommended top-level JSON shape:

```json
{
  "character_profile_cards_version": "character_profile_cards_v1",
  "source_review_version": "corpus_sanity_review_v1",
  "character_normalization": {
    "applied": true,
    "map": {
      "Charlus": "baron de Charlus"
    }
  },
  "character_count": 62,
  "cards": [
    {
      "...": "character card"
    }
  ]
}
```

## Required Card Fields

Each card should contain these required fields.

```json
{
  "character": "Odette",
  "annotation_unit_count": 88,
  "rank_spread": 48,
  "max_score_span": 7.422,
  "selected_by": ["rank_spread", "volatility"],
  "lens_scores": {
    "advantage": {
      "net_score": -1.638,
      "percentile": 56,
      "rank": 28,
      "unit_count": 88,
      "dominant_status_dimension": "social_status",
      "score_span": 6.598,
      "mean_score": -0.019
    },
    "prestige": {
      "net_score": 15.86,
      "percentile": 95,
      "rank": 4,
      "unit_count": 88,
      "dominant_status_dimension": "social_status",
      "score_span": 7.422,
      "mean_score": 0.18
    },
    "inclusion": {
      "net_score": -23.464,
      "percentile": 16,
      "rank": 52,
      "unit_count": 88,
      "dominant_status_dimension": "social_status",
      "score_span": 6.155,
      "mean_score": -0.267
    }
  },
  "top_chapters": [
    {
      "chapter_id": "v2-p1-autour-de-mme-swann",
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
      }
    }
  ]
}
```

## Field Meanings

### `character`

Normalized character identity key.

This should always be the reviewed aggregate-layer name, not an unnormalized alias.

### `annotation_unit_count`

Primary ranking field for “most annotated.”

This is the count of annotated units in which the character appears on the current canonicalized corpus surface.

### `rank_spread`

Difference between the character’s best and worst rank across:

- `advantage`
- `prestige`
- `inclusion`

This is one of the most useful quick signals for “cross-lens instability.”

### `max_score_span`

Largest within-lens volatility span for the character across units.

This is the best quick signal for “internal volatility.”

### `selected_by`

Optional but useful.

Suggested values:

- `rank_spread`
- `volatility`
- `annotation_count`
- `manual`

This helps the app explain why a character is being surfaced.

### `lens_scores`

Required.

Each lens should include:

- `net_score`
- `percentile`
- `rank`
- `unit_count`
- `dominant_status_dimension`
- `score_span`
- `mean_score`

This is the core of the profile card.

### `top_chapters`

Required.

This should contain the most important chapter-level accumulations for the character.

The first implementation should not try to solve this with a new abstract metric.

A practical rule is:

- sort chapter rows by maximum absolute chapter net score across the three lenses
- keep the top `3` to `5`

That is enough for the app to say where the character’s profile is coming from.

## Optional Enrichment Fields

These are useful, but not required for `v1`.

### `summary`

Short human-readable sentence, for example:

- `Prestige-positive but inclusion-negative overall, driven heavily by v2-p1 and v1-p2.`

This is useful for cards, but it is editorialized derived text rather than core numeric structure.

It should be optional in `v1`.

### `chapter_count`

Number of chapters in which the character appears.

This can help distinguish broad corpus presence from a concentrated chapter footprint.

### `primary_pattern`

Small categorical label, such as:

- `prestige_positive_inclusion_negative`
- `consistently_negative`
- `cross_lens_split`
- `broad_positive`

This is useful later for filtering, but should remain optional.

### `links`

App-friendly URLs.

Example:

```json
{
  "links": {
    "reader": "/projects/islt/fr-original/v2-p1-autour-de-mme-swann",
    "chapter_analysis_anchor": "/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-1"
  }
}
```

This is convenient, but not necessary for the first schema.

## Suggested Sorting

The top-level `cards` array should probably be sorted by:

1. descending `annotation_unit_count`
2. then descending `rank_spread`
3. then character name

That keeps the list useful for both:

- “most important” character discovery
- “most analytically interesting” character discovery

## Minimal `v1` Schema

If the goal is the smallest useful app contract, `v1` can be just:

```json
{
  "character": "Odette",
  "annotation_unit_count": 88,
  "rank_spread": 48,
  "max_score_span": 7.422,
  "selected_by": ["rank_spread", "volatility"],
  "lens_scores": {
    "advantage": { "net_score": -1.638, "percentile": 56, "rank": 28, "unit_count": 88 },
    "prestige": { "net_score": 15.86, "percentile": 95, "rank": 4, "unit_count": 88 },
    "inclusion": { "net_score": -23.464, "percentile": 16, "rank": 52, "unit_count": 88 }
  },
  "top_chapters": [
    {
      "chapter_id": "v2-p1-autour-de-mme-swann",
      "advantage": { "net_score": 19.699, "unit_count": 32 },
      "prestige": { "net_score": 24.138, "unit_count": 32 },
      "inclusion": { "net_score": 8.419, "unit_count": 32 }
    }
  ]
}
```

That is already enough to render a very good card.

## Recommended Next Step

The next practical step is:

1. implement `build_character_profile_cards(...)`
2. generate `character-profile-cards-current.json`
3. keep the first version purely derived from current artifacts

The first implementation should avoid generating prose summaries until the numeric card shape is stable.
