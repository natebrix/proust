# Outputs Guide

This document is a high-level guide to the artifacts under `outputs/`.

It is not a chronological log.

Its purpose is to answer:

1. what kinds of outputs exist
2. which outputs are current versus historical
3. what a new session should read first
4. where to look for granular run material versus aggregate analysis

For project history and judgments, see:

- [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1)
- [annotation_log.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_log.md:1)
- [downstream_analysis_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/downstream_analysis_plan.md:1)

## Reading Order

If you are re-entering the project and want the shortest useful path, read in this order:

1. [current_state.md](/Users/nathan_brixius/dev/proust/proust/docs/current_state.md:1)
2. [corpus-review-current-normalized.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.md:1)
3. [character-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.md:1)
4. [character-chapter-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.md:1)
5. [annotation_log.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_log.md:1) only if you need the historical path

## Output Families

The `outputs/` directory now contains five main artifact families:

1. granular `run-*` directories
2. corpus-review artifacts
3. alias-audit and normalization artifacts
4. downstream character-level analysis artifacts
5. character annotation-count artifacts
6. character profile-card artifacts
7. character page artifacts for the `islt` app
8. chapter overlay artifacts for the `islt` app
9. historical milestone artifacts kept for comparison

## Granular Runs

The `run-*` directories are the basic production units.

Typical examples:

- `outputs/run-016`
- `outputs/run-276`
- `outputs/run-556`

Each run directory generally contains:

- `run.json`
- `units/`
- `prompts/`
- `raw/`
- `annotations/`

Conceptually:

- source runs define units, prompts, alias maps, and notes
- output runs contain raw model output, reduced annotations, automation state, and reports

If you need the mechanics of how a run is structured or prepared, use:

- [annotation_runner.md](/Users/nathan_brixius/dev/proust/proust/docs/annotation_runner.md:1)
- [full_corpus_runbook.md](/Users/nathan_brixius/dev/proust/proust/docs/full_corpus_runbook.md:1)

Use granular runs when you need:

- unit-level evidence
- run-level validation details
- raw versus reduced annotation comparison
- the exact source of a later aggregate finding

Do not start with granular runs if the question is corpus-level.

## Current Canonical Aggregate Surfaces

These are the current default aggregate artifacts.

### 1. Unnormalized Current Corpus Review

- [corpus-review-current.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current.json:1)
- [corpus-review-current.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current.md:1)

Purpose:

- refreshed whole-corpus aggregate review over accepted annotations before aggregate-layer character normalization

Use when:

- you want the pre-normalization surface for comparison

### 2. Normalized Current Corpus Review

- [corpus-review-current-normalized.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.json:1)
- [corpus-review-current-normalized.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.md:1)

Purpose:

- current default corpus-level review surface
- per-character summaries after reviewed explicit aggregate-layer name merges

Use when:

- you want the current default corpus-wide character surface
- you want top positive and negative characters by lens
- you want the current cross-lens stability summary

This is the default corpus-review artifact to read first.

## Alias And Normalization Artifacts

These explain how the project moved from split character identities to a cleaner aggregate surface.

### Alias Audit

- [character-alias-audit-current.json](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.json:1)
- [character-alias-audit-current.md](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.md:1)

Purpose:

- identify candidate duplicate character identities using annotation usage, `aliases.csv`, and run-level alias maps

Use when:

- you want to see why certain names were treated as merge candidates

### Normalization Plan

- [character_alias_normalization_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/character_alias_normalization_plan.md:1)

Purpose:

- record the reviewed explicit merge decisions

### Normalization Diff

- [corpus-review-normalization-diff.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-normalization-diff.md:1)

Purpose:

- show exactly what changed between the unnormalized and normalized aggregate surfaces

Use when:

- you want to verify that normalization cleaned up identity splits without changing the broader corpus-level judgment

## Downstream Analysis Artifacts

These are the first post-normalization analysis layers built on top of the accepted corpus.

### Character Cross-Lens Analysis

- [character-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.json:1)
- [character-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.md:1)

Purpose:

- compare `local`, `prestige`, and `inclusion` for each normalized character
- surface rank spread and volatility

Use when:

- you want to know which characters diverge most across lenses
- you want to find high-volatility characters before drilling down by chapter

### Character Chapter Cross-Lens Analysis

- [character-chapter-cross-lens-current.json](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.json:1)
- [character-chapter-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.md:1)

Purpose:

- show where the largest cross-lens splits and highest-volatility characters are concentrated chapter by chapter

Use when:

- you want to convert an abstract character-level anomaly into a chapter-structured reading
- you want to know which chapters are driving `Odette`, `Robert de Saint-Loup`, `Mme de Villeparisis`, `Swann`, `Albertine`, or `baron de Charlus`

This is the current active checkpoint artifact for downstream reading.

### Character Annotation Counts

- [character-annotation-counts-current.json](/Users/nathan_brixius/dev/proust/outputs/character-annotation-counts-current.json:1)
- [character-annotation-counts-current.md](/Users/nathan_brixius/dev/proust/outputs/character-annotation-counts-current.md:1)

Purpose:

- list every normalized character in descending order of annotation-unit count

Use when:

- you want the simplest answer to "which characters are most annotated?"
- you want a compact current ranking of the most textually active figures in the accepted corpus

### Character Profile Cards

- [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)
- [character-profile-cards-current.md](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.md:1)

Purpose:

- provide an app-facing, per-character JSON contract for cross-lens profile cards

Use when:

- you want one stable object per character with annotation count, lens scores, rank spread, volatility, and top driving chapters
- you want to feed a reader app or another presentation layer without recomputing analysis logic there

### Character Pages

- [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)
- [character-pages-current.md](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.md:1)

Purpose:

- provide a pilot page-ready artifact for a small high-signal character set
- combine computed profile data, chapter drivers, portrait metadata, and editorial explainer fields

Use when:

- you want a full character-page surface rather than a compact card
- you want a rendering-oriented handoff for the `islt` app

### Chapter Overlays

- [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)
- [v1-p1-combray.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/chapters/v1-p1-combray.json:1)

Purpose:

- provide chapter-keyed, paragraph-range overlay JSON for the `islt` reader
- bridge normalized aggregate/unit-level annotation data into an app-friendly form
- include deterministic chapter and unit summaries in the current `chapter_overlay_v2` surface

Use when:

- you want inline chapter overlays, lens toggles, or character chips in the reader
- you want one canonical per-unit overlay surface, with later reviewed runs superseding earlier duplicate units

## Historical Milestone Artifacts

Some aggregate files are important historical checkpoints but are not the current default surface.

### Early Corpus Sanity Check

- [corpus-review-001.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-001.json:1)

Purpose:

- first explicit corpus-level sanity proof over an earlier accepted subset

### Final Pre-Refresh Corpus Review

- [corpus-review-final.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-final.json:1)

Purpose:

- final aggregate review from the end of the production pass before later refresh and normalization work

Use these when:

- you want to compare phases of the project
- you want to reconstruct how the aggregate surface evolved over time

Do not treat them as the current default reading surface.

## Which Artifact Answers Which Question

If the question is:

- "What is the current corpus-wide picture?"  
  Read [corpus-review-current-normalized.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-current-normalized.md:1)

- "Which characters diverge most across lenses?"  
  Read [character-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-cross-lens-current.md:1)

- "Which chapters are driving those divergences?"  
  Read [character-chapter-cross-lens-current.md](/Users/nathan_brixius/dev/proust/outputs/character-chapter-cross-lens-current.md:1)

- "Which characters are most annotated overall?"  
  Read [character-annotation-counts-current.md](/Users/nathan_brixius/dev/proust/outputs/character-annotation-counts-current.md:1)

- "What should a cross-lens character card contain?"  
  Read [character-profile-cards-current.json](/Users/nathan_brixius/dev/proust/outputs/character-profile-cards-current.json:1)

- "What should a first full character page contain?"  
  Read [character-pages-current.json](/Users/nathan_brixius/dev/proust/outputs/character-pages-current.json:1)

- "What should the `islt` app use for inline paragraph-range overlays?"  
  Read [chapter-overlays-current/manifest.json](/Users/nathan_brixius/dev/proust/outputs/chapter-overlays-current/manifest.json:1)

- "Why were these names merged?"  
  Read [character-alias-audit-current.md](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.md:1) and [character_alias_normalization_plan.md](/Users/nathan_brixius/dev/proust/proust/docs/character_alias_normalization_plan.md:1)

- "What exactly changed after normalization?"  
  Read [corpus-review-normalization-diff.md](/Users/nathan_brixius/dev/proust/outputs/corpus-review-normalization-diff.md:1)

- "Where did a specific finding come from at the unit level?"  
  Trace back to the relevant `run-*` directory and then to `annotations/` and `raw/`

## Practical Rule

Default to the highest available aggregate layer first.

That means:

1. current normalized corpus review
2. character cross-lens analysis
3. character chapter cross-lens analysis
4. only then the relevant `run-*` directories

This keeps the project aligned with its current report-first and aggregate-first review rule.
