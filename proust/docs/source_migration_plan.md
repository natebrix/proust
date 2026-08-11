# Source Migration Plan: fr-original to Wikisource

Decision (Nathan, 2026-08-11): migrate the French text from the historical
fan-site transcription (marcel-proust.com, unknown editorial lineage,
audiobook-collation artifacts in the posthumous volumes) to the most
authoritative complete public source: **French Wikisource**, which carries the
original NRF/Gallimard editions, scan-backed and revision-controlled. The
guiding principle is a strong foundation: text authority and annotation
authority are the same project.

## Why Wikisource

- complete: all seven volumes present in the original edition structure
- scan-backed: transcluded from proofread page images (Page: namespace)
- citable and immutable: every page has a revision id; we pin and record the
  exact revids we consume
- public domain text with a versioned, community-proofread transcription

## Phases

1. **R — mapping (recon).** Verified mapping from our 18 canonical chapter ids
   to ordered Wikisource pages, with boundary anchors where our chapter
   divisions fall inside their pages, revids, and scan-backed status:
   `outputs/wikisource-mapping.json`.
2. **B — fetch + build + align (staged).** Throttled fetcher with a provenance
   manifest (title + revid per page); paragraph extraction from rendered HTML
   (page-number markers, footnotes, and header apparatus stripped); new
   canonical chapters in the existing schema; sentence segmentation
   (fr_core_news_sm). Then old-to-new paragraph alignment per chapter
   (similarity-based, not equality: the transcriptions differ) producing
   `outputs/source-migration-map.json` (old p-N -> new p-N or split/merge
   ranges) and a divergence report. Gates: every old paragraph maps (or its
   absence is explained and reviewed); divergence classified; nothing applies
   without review.
3. **D — apply + interim remap.** Apply new chapters; remap paragraph
   references in the app-facing artifacts (timeline corpus positions, dossier
   passage links, chapter-summary passages) through the migration map so the
   site stays coherent; rebuild and verify sampled deep links; provenance
   documented in an edition statement.

## Consequences accepted

- Paragraph numbering changes; the migration map is the bridge. Historical
  unit ids (`v1-p1-combray#p-17`) remain identifiers of the LEGACY text's
  spans; they are remapped for display but not renamed.
- The accepted annotation corpus and the supplement corpus become legacy
  surfaces at re-annotation time (registry design migration steps 2-3), which
  now runs against the Wikisource text with prompt v2 and the character
  registry. The re-annotation supersedes rather than patches.
- `en-moncrieff` is out of scope for this migration and keeps its current
  source until its own provenance review.

## Provenance policy

Every consumed Wikisource page is recorded with title, revid, and timestamp in
`data/wikisource/manifest.json`. The edition carries an edition statement
(source, editions transcribed, retrieval dates, revids) surfaced on the site's
methods page. Text updates are deliberate re-pins to newer revids, never
silent refetches.
