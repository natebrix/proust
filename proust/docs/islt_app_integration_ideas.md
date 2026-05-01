# ISLT App Integration Ideas

This document sketches how the annotation corpus and downstream analysis artifacts could be used inside the `islt` reader app in:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt`

It focuses on:

1. how hard it is to map annotation `unit_id`s onto the reader text
2. a practical implementation plan
3. product ideas for inline and aggregate presentation

## Short Answer

The paragraph-level mapping is straightforward.

The `islt` reader already renders canonical chapter paragraphs with stable ids like:

- `p-1`
- `p-121`

The annotation corpus already uses unit ids like:

- `v7-p4-le-bal-de-tetes#p-121-p-125`
- `v2-p2-noms-de-pays-le-pays#p-211-p-213`
- `v1-p1-combray#p-17`

That means the chapter id and paragraph range are already encoded in a directly compatible format.

For paragraph-range overlays, the mapping difficulty is low.

For sentence-level overlays, the mapping difficulty is higher, because the current annotations are unit-level and paragraph-range-based rather than sentence-addressed.

## Exact Mapping Fit

The compatibility is unusually clean:

- reader route chapter ids use canonical chapter ids like `v1-p1-combray`
- reader paragraph ids use canonical paragraph ids like `p-17`
- annotation `unit_id` is built as `chapter_id#p-start` or `chapter_id#p-start-p-end`

The existing helper in [coordinates.py](/Users/nathan_brixius/dev/proust/proust/coordinates.py:21) already defines this format:

```python
def annotation_unit_id(chapter_id, paragraph_start, paragraph_end=None):
    end = paragraph_end if paragraph_end is not None else paragraph_start
    return f"{chapter_id}#p-{paragraph_start}" if end == paragraph_start else f"{chapter_id}#p-{paragraph_start}-p-{end}"
```

The `islt` reader currently renders chapter paragraphs as sections with ids like `p-121`, using chapter data from:

- [lib/islt.ts](/Users/nathan_brixius/dev/brixius-web/lib/islt.ts:1)
- [components/islt-reader.tsx](/Users/nathan_brixius/dev/brixius-web/components/islt-reader.tsx:1)

So the translation rule is simple:

1. parse `unit_id`
2. split at `#` to get `chapter_id`
3. parse the starting and ending paragraph numbers
4. mark all reader paragraphs whose `index` falls inside that range

Example:

- `v7-p4-le-bal-de-tetes#p-121-p-125`

maps to:

- chapter route `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`
- paragraph ids `p-121`, `p-122`, `p-123`, `p-124`, `p-125`

## Difficulty Assessment

### Easy

- paragraph-range highlighting
- showing unit cards or chips above the first paragraph in a unit
- adding a lens toggle
- adding character chips per unit
- linking from aggregate artifacts to chapter routes with paragraph anchors

### Moderate

- chapter minimaps or heat strips
- character-focus mode inside a chapter
- chapter-level summary banners built from the downstream analysis artifacts
- search/filter by normalized character identity

### Harder

- sentence-level evidence highlighting
- event-by-event inline rendering
- auto-linking quoted `evidence` snippets back to exact sentence spans

Those are harder because the current corpus does not store sentence-addressed spans. It stores:

- paragraph-range units
- appraisal events
- status effects
- textual evidence strings

but not exact sentence ids.

## Completed Foundations

Several of the original app-integration ideas are now complete as data products:

- chapter overlay export exists as `chapter_overlay_v2`
- cross-lens character profile cards exist as a stable JSON contract
- character pages now exist as a fuller editorial/app-facing JSON contract
- lens naming is now standardized as `advantage / prestige / inclusion`
- character-facing lens display should now be treated as percentile-first, with raw score and rank preserved as additive detail

Current key artifacts:

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

For the `islt` app, this means the main remaining work is no longer inventing a character layer. It is:

- rendering and linking the existing character data products
- adding chapter-level framing between character pages and inline overlays
- supporting chapter-level exploration such as character focus mode and chapter summary banners

## Best Current UI Ideas

These are the highest-value ideas that fit the current data with low friction.

### 1. Unit Range Overlays

Lightly shade the full annotated paragraph range.

At the first paragraph of the unit, show a compact header:

- major characters
- dominant direction
- active lens score summary

Example:

- `Odette  prestige +`
- `Swann  advantage -`
- `Albertine  inclusion -`

This is the best first feature because it maps directly to the unit structure you already have.

### 2. Lens Toggle

Add a chapter-level toggle:

- `Advantage`
- `Prestige`
- `Inclusion`

Then recolor and relabel the same unit overlays by lens.

This would express the central analytical result of the project without needing a complex new interface.

### 3. Character Chips

For each annotated unit, show normalized character chips with directional summaries.

Good default chip content:

- character name
- sign
- maybe the dominant status dimension

Example:

- `Robert de Saint-Loup  prestige +`
- `Odette  inclusion -`
- `baron de Charlus  advantage -`

### 4. Chapter Heat Strip

Add a slim chapter minimap showing where positive, negative, or mixed units cluster.

This would help a reader see chapter shape before reading any individual overlay text.

### 5. Character Focus Mode

Let the reader pick one character and show only units involving that character.

This becomes especially useful now that the corpus has:

- normalized character identities
- cross-lens rank spread data
- chapter-level distribution data

Good first candidates:

- `Odette`
- `Robert de Saint-Loup`
- `baron de Charlus`
- `Albertine`
- `Swann`

## Higher-Level Ideas Using What We Learned

The aggregate artifacts create opportunities beyond simple inline annotation.

### Cross-Lens Character Profiles

This is now complete enough as a derived data product and should be treated as a rendering problem, not a missing-analysis problem.

Current artifacts:

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-profile-cards-current.md](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.md:1)
- [character_profile_card_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_profile_card_schema.md:1)

Each current card already includes:

- overall advantage / prestige / inclusion totals
- percentile in each lens, with higher = better
- current rank in each lens
- rank spread
- top chapters driving the result

Recommended display policy:

- use percentile as the primary reader-facing measure
- keep rank as secondary precision
- keep raw score as optional deeper analytic detail

So for the `islt` app, the remaining work here is rendering and linking, not inventing a new analysis layer.

### Chapter Framing Banners

At the top of selected chapters, show a short structured summary such as:

- a chapter-centered prose summary
- a tonal archetype
- chapter lens densities and ranks
- top characters by impact mass
- distinguishing passages

This would work well for chapters like:

- `v2-p1-autour-de-mme-swann`
- `v3-p1`
- `v5`
- `v7-p2-m-de-charlus-pendant-la-guerre`

### “Why This Character Is Interesting” Cards

Use the downstream analysis artifacts to identify characters whose social reading changes dramatically by lens.

Examples already surfaced:

- `Odette`: prestige-positive but inclusion-negative overall
- `Robert de Saint-Loup`: prestige-positive but inclusion-negative, concentrated heavily in `v3-p1`
- `Mme de Villeparisis`: chapter-structured lens split rather than simple noise

These can become editorial entry points for the app.

## Data Products The App Would Want

The cleanest app integration path is not to have the web app compute everything from raw `run-*` directories.

Instead, export small chapter-oriented JSON artifacts from the annotation project.

For the completed character-card JSON contract and current artifact, see:

- [character_profile_card_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_profile_card_schema.md:1)
- [character_page_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/character_page_schema.md:1)
- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)
- [islt_character_pages_handoff.md](/Users/nathan_brixius/dev/proust/proust/docs/islt_character_pages_handoff.md:1)

For the chapter overlay export contract, see:

- [chapter_overlay_schema.md](/Users/nathan_brixius/dev/proust/proust/docs/chapter_overlay_schema.md:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

### Recommended Export 1: Chapter Annotation Overlay Data

One file per chapter, or a chapter-keyed manifest.

Suggested shape:

```json
{
  "chapterId": "v7-p4-le-bal-de-tetes",
  "units": [
    {
      "unitId": "v7-p4-le-bal-de-tetes#p-121-p-125",
      "paragraphStart": 121,
      "paragraphEnd": 125,
      "characters": [
        {
          "character": "Albertine",
          "dominantStatusDimension": "general_appraisal",
          "advantage": { "netScore": -2.6, "label": "loss" },
          "prestige": { "netScore": -1.9, "label": "loss" },
          "inclusion": { "netScore": -2.8, "label": "loss" }
        }
      ],
      "dominantCharacter": "Albertine",
    }
  ]
}
```

This is enough for unit shading, chips, and lens toggles.

This export now exists as the current app-facing dataset, and the current `chapter_overlay_v2` payload also includes additive deterministic chapter and unit summaries.

### Recommended Export 2: Chapter Character Summary Data

One file per chapter summarizing major character totals inside that chapter.

Suggested shape:

```json
{
  "chapterId": "v3-p1",
  "characters": [
    {
      "character": "Robert de Saint-Loup",
      "advantage": -9.402,
      "prestige": 5.885,
      "inclusion": -26.3,
      "unitCount": 132
    }
  ]
}
```

This is enough for:

- chapter headers
- chapter framing banners
- sidebars
- character-focus mode

The current project already has most of this information in aggregate form via:

- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)

So the remaining gap is mainly app-oriented packaging by chapter rather than new analysis.

The next planning document for this layer is:

- [chapter_summary_export_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/chapter_summary_export_plan.md:1)

The first current artifact is now available at:

- [chapter-summaries-current.json](/Users/nathan_brixius/dev/proust/outputs/chapter-summaries-current.json:1)
- [chapter-summaries-current.md](/Users/nathan_brixius/dev/proust/outputs/chapter-summaries-current.md:1)

The current chapter-summary layer should now be treated as:

- `chapter_summary_export_v2`
- chapter-centered rather than report-centered
- based on chapter lens densities rather than character-in-chapter percentiles
- impact-mass based for top chapter characters

### Recommended Export 3: Current Canonical Aggregate Manifest

One small manifest that points the app at the current canonical analysis surfaces:

- normalized corpus review
- current character cross-lens analysis
- current character chapter analysis

This keeps the app aligned with whichever artifacts are current.

This should now also include:

- current character profile cards

## Concrete Implementation Plan

### Phase 1: Minimal Inline Annotation

Goal:

- make annotations visible in the chapter reader with minimal UI risk

Build:

1. export chapter overlay JSON from the annotation project
2. load overlay JSON in the `islt` chapter page
3. shade annotated paragraph ranges
4. add a `Advantage / Prestige / Inclusion` toggle
5. show normalized character chips and a one-line unit summary

This phase is enough to make the project legible inside the text.

### Phase 2: Character-Aware Reading

Goal:

- make the app useful for exploring the strongest corpus-level findings

Build:

1. render the existing character profile/page JSON in the app
2. character-focus filter
3. chapter-level character summary panel
4. links from character cards/pages to the paragraph ranges where the score is coming from

This phase would let a reader move from aggregate finding to textual evidence efficiently.

### Phase 3: Chapter-Structured Analysis

Goal:

- bridge the gap between whole-corpus findings and local reading

Build:

1. chapter summary export `v2`
2. chapter framing banner
3. chapter heat strip
4. “why this chapter matters for this character” summaries using the chapter cross-lens artifact

This is where the downstream analysis work becomes editorially strong.

## Most Practical First Feature

If only one feature should be built first, it should be:

- paragraph-range overlays with a lens toggle and normalized character chips

Why:

- mapping is direct
- it uses the current data almost as-is
- it expresses the key analytical result clearly
- it avoids cluttering the sentence text

## Important Constraint

The current annotation corpus is paragraph-range-based, not sentence-addressed.

So the first app integration should stay paragraph- and unit-based.

Sentence-level highlighting should be treated as a later enhancement, not a requirement for the first version.

## Recommended Next Technical Step

The character profile/page layers and chapter overlay export are now in place.

The next technical step on the data side is a chapter-summary export that sits between character pages and inline overlays.

Current app-facing inputs already available:

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)
- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

If app-facing data work continues beyond that, the next highest-value additive layer is:

- chapter-summary export
- then chapter framing/editorial summarization built on top of it
