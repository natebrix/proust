# Edition Statement: fr-original

This statement covers the `fr-original` edition of *À la recherche du temps
perdu* served by the ISLT app (`data/islt/editions/fr-original`). It does not
cover `en-moncrieff`, which is out of scope for this migration and keeps its
own provenance until its own review (`proust/docs/source_migration_plan.md`).

## Source

The text is transcribed from **French Wikisource**
(`https://fr.wikisource.org`), the French-language sister project of
Wikisource: a community-proofread transcription transcluded from scanned page
images (the `Page:` namespace) of the original printed editions. All 24 pages
consumed by this edition are `scan_backed = true` per
`outputs/wikisource-mapping.json` — every paragraph traces to a photographed
page, not to an unverified retyping.

This replaces the edition's original source, an unattributed fan-site
transcription (marcel-proust.com) of unknown editorial lineage that carried
audiobook-collation artifacts in the posthumous volumes (see "Apparatus
exclusion policy" below, and `proust/docs/source_migration_plan.md`). The
guiding principle for the change: text authority and annotation authority
should be the same project.

## Provenance: pages, revisions, retrieval

Every page is fetched **by revision id**, never "latest" — `oldid=<revid>`,
so a refetch can never silently change the text — through the MediaWiki parse
API (`scripts/fetch_wikisource_pages.py`). Retrieval ran on **2026-08-11**
(`fetched_at: 2026-08-11T22:00:15Z`, `data/wikisource/manifest.json`). Full
provenance (title, revid, timestamp, sha256 of the cached HTML, byte count)
for all 24 pages is recorded there; the table below is the canonical-chapter
mapping (`outputs/wikisource-mapping.json`) joined against it.

| Wikisource page | revid | revision timestamp | canonical chapter(s) |
| --- | ---: | --- | --- |
| Du côté de chez Swann/Partie 1 | 4966056 | 2015-02-17T14:18:19Z | v1-p1-combray |
| Du côté de chez Swann/Partie 2 | 6010454 | 2016-07-02T15:30:58Z | v1-p2-un-amour-de-swann |
| Du côté de chez Swann/Partie 3 | 4966058 | 2015-02-17T14:18:48Z | v1-p3-noms-de-pays-le-nom |
| À l'ombre des jeunes filles en fleurs/Première partie | 3939419 | 2013-02-22T20:55:19Z | v2-p1-autour-de-mme-swann |
| À l'ombre des jeunes filles en fleurs/Deuxième partie | 14982744 | 2025-03-12T10:16:41Z | v2-p1-autour-de-mme-swann, v2-p2-noms-de-pays-le-pays |
| À l'ombre des jeunes filles en fleurs/Troisième partie | 2222082 | 2011-02-17T20:29:36Z | v2-p2-noms-de-pays-le-pays |
| Le Côté de Guermantes/Première partie | 3939407 | 2013-02-22T20:51:42Z | v3-p1 |
| Le Côté de Guermantes/Deuxième partie | 4270224 | 2013-10-27T18:02:56Z | v3-p1, v3-p2 |
| Le Côté de Guermantes/Troisième partie | 4497021 | 2014-03-21T22:41:21Z | v3-p2 |
| Sodome et Gomorrhe/Partie 1 | 4497996 | 2014-03-22T17:13:51Z | v4-p1 |
| Sodome et Gomorrhe/Partie 2 - chapitre 1 | 10484547 | 2020-06-02T09:26:19Z | v4-p2 |
| Sodome et Gomorrhe/Partie 2 - chapitre 2 | 4616362 | 2014-06-20T08:33:06Z | v4-p2 |
| Sodome et Gomorrhe/Partie 2 - chapitre 3 | 7292642 | 2018-04-03T12:40:56Z | v4-p2 |
| Sodome et Gomorrhe/Partie 2 - chapitre 4 | 4641264 | 2014-07-13T16:34:39Z | v4-p2 |
| La Prisonnière/Chapitre 1 | 4684429 | 2014-08-21T09:43:13Z | v5 |
| La Prisonnière/Chapitre 2 | 4725952 | 2014-09-20T17:03:34Z | v5 |
| La Prisonnière/Chapitre 3 | 4748205 | 2014-10-07T12:02:51Z | v5 |
| Albertine disparue/Chapitre I | 4766819 | 2014-10-22T11:17:21Z | v6-p1 |
| Albertine disparue/Chapitre II | 4775678 | 2014-10-29T13:16:25Z | v6-p2 |
| Albertine disparue/Chapitre III | 4784837 | 2014-11-04T12:36:42Z | v6-p3 |
| Albertine disparue/Chapitre IV | 4793526 | 2014-11-10T19:25:46Z | v6-p4 |
| Le Temps retrouvé/I | 4805044 | 2014-11-19T21:37:56Z | v7-p1-a-tansonville, v7-p2-m-de-charlus-pendant-la-guerre |
| Le Temps retrouvé/II | 4831288 | 2014-12-03T19:45:32Z | v7-p2-m-de-charlus-pendant-la-guerre |
| Le Temps retrouvé/III | 4873511 | 2014-12-23T16:06:42Z | v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle, v7-p4-le-bal-de-tetes |

Two canonical chapter boundaries fall *inside* a single Wikisource page
rather than at a page break — the novel's own part divisions do not always
coincide with how Wikisource split its pages. `v2-p1`/`v2-p2` split inside "À
l'ombre .../Deuxième partie", and `v3-p1`/`v3-p2` split inside "Le Côté de
Guermantes/Deuxième partie", and `v7-p1`/`v7-p2` and `v7-p3`/`v7-p4` inside
"Le Temps retrouvé/I" and "/III" respectively. Each boundary was verified by
locating the exact sentence Wikisource's own wikitext marks with a section
break (or, for Le Temps retrouvé, a verified verbatim anchor phrase) — see the
per-chapter `notes` in `outputs/wikisource-mapping.json` for the specific
anchor text used.

## Mixed scan editions

Wikisource's transcription of *À la recherche du temps perdu* is not scanned
from one uniform printing; the underlying page scans (`data/wikisource/pages/*.html`,
`itemprop="datePublished"` / `volumeNumber` metadata) span five different
Gallimard printings across the 24 tomes:

| Volume(s) | Scan edition | Tome(s) | Canonical chapters |
| --- | --- | --- | --- |
| Du côté de chez Swann, Partie 1-2 | 1946 Gallimard ("édition définitive") | 1 | v1-p1-combray, v1-p2-un-amour-de-swann |
| Du côté de chez Swann, Partie 3 | 1919 NRF original | 2 | v1-p3-noms-de-pays-le-nom |
| À l'ombre des jeunes filles en fleurs | 1919 NRF original | 3, 4, 5 | v2-p1-autour-de-mme-swann, v2-p2-noms-de-pays-le-pays |
| Le Côté de Guermantes | 1921 NRF original | 6, 7, 8 | v3-p1, v3-p2 |
| Sodome et Gomorrhe | 1924 NRF original | 9, 10 | v4-p1, v4-p2 |
| La Prisonnière | 1946 Gallimard | 11, 12 | v5 |
| Albertine disparue | 1927 Gallimard | 13 | v6-p1, v6-p2, v6-p3, v6-p4 |
| Le Temps retrouvé | 1927 Gallimard | 14 | v7-p1-a-tansonville, v7-p2-m-de-charlus-pendant-la-guerre, v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle, v7-p4-le-bal-de-tetes |

Practically: the first three volumes (Swann's Way through Le Côté de
Guermantes and Sodome et Gomorrhe) are the 1919-1924 NRF first editions,
Proust's own lifetime text apart from Partie 1-2 of Swann, which Wikisource
happens to have scanned from the later 1946 "édition définitive" Gallimard
printing (post-Proust editorial revision by Gallimard, published after his
1922 death) — the same 1946 printing used for La Prisonnière. The two most
posthumous volumes, Albertine disparue and Le Temps retrouvé — assembled and
published by Proust's brother Robert and Gallimard from his manuscripts and
proofs — are both scanned from the 1927 Gallimard printing. This mixture is a
property of what Wikisource happened to scan, not an editorial choice made
here; the per-chapter table above lets any given passage's textual lineage be
traced to a specific scanned tome.

## Apparatus exclusion policy

The rendered Wikisource HTML carries transcription and site apparatus that is
not Proust's text, all stripped before paragraph extraction
(`scripts/build_wikisource_chapters.py`):

- header/footer/navigation templates (`div.ws-noexport`)
- transcluded page-number markers (`span.pagenum`)
- footnote markers and the footnote list (`sup.reference` / `ol.references`)
  — e.g. a Robert Proust editorial note appended after the body text of
  Albertine disparue Chapitre II
- per-chapter argument summaries (`div.alineanegatif`)
- chapter headings (`CHAPITRE PREMIER`, part titles, etc.)
- the `⁂` section-break glyphs

What each chapter had stripped is recorded in
`data/islt/editions/fr-original-ws/stripped-apparatus.json` so the alignment
step (`scripts/align_migration_map.py`) could account for every corresponding
paragraph on the legacy side. Everything else is kept verbatim apart from
whitespace collapse — Wikisource's apostrophes (U+2019), guillemets, accents,
and ellipses are this edition's typography.

The legacy (fan-site) source separately carried its own apparatus that has no
place in the text either: site-navigation boilerplate, a "FIN du roman ..."
colophon, and — in the posthumous volumes La Prisonnière above all — inserted
passages explicitly marked `[----Ajout Gallimard----]` or attributed to "l'édition
sonore Thélème," i.e. text spliced in from a *different* edition or an
audiobook collation rather than transcribed from any single printed source.
None of that carries over: the Wikisource-derived text contains no such
markers (verified: no built reader page contains the string `[----Ajout
Gallimard----`), and the migration map records each legacy-only paragraph of
this kind with an `editorial_marker` / `editorial_marker_block` annotation
(`outputs/source-migration-map.json`) rather than silently mapping it
somewhere.

## Re-pin policy

Every consumed page is pinned to an exact revision id, recorded with title,
revid, and timestamp in `data/wikisource/manifest.json`
(`"note": "Pinned revisions only: pages are fetched by oldid, never by
latest. Text updates are deliberate re-pins in the mapping, never silent
refetches."`). Wikisource's own text can and does keep improving after these
revids were pinned (community proofreading is ongoing); this edition does not
track that automatically. Adopting a newer revision of any page is a
**deliberate, reviewed re-pin**: update the revid in
`outputs/wikisource-mapping.json`, refetch just that page, rerun the
alignment (`scripts/align_migration_map.py`) and, if it applies cleanly, the
paragraph remap (`scripts/remap_artifact_paragraphs.py`) — the same phase
B/D pipeline used for this migration, never a silent overwrite of the cached
HTML or the chapter JSON.

## Related documents

- `proust/docs/source_migration_plan.md` — the migration plan (phases R/B/D)
  this statement closes out.
- `outputs/wikisource-mapping.json` — chapter-to-page mapping, boundary
  anchors, scan-backed status, per-chapter provenance notes.
- `data/wikisource/manifest.json` — page-level provenance (revid, timestamp,
  sha256).
- `outputs/source-migration-map.json` / `outputs/source-migration-divergence.md`
  — the old-to-new paragraph alignment and its review surface.
