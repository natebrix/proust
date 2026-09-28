# ISLT Fortune Arcs Handoff

This is the engineering brief for a frontend session that adds a **fortune
arc** to the existing `islt` character pages.

The goal is narrow:

- keep the current character page intact
- add one compact chart: how the novel treats this character over time
- read the exported data directly
- recompute nothing in the app

The `islt` app lives in:

- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt`

## What a fortune arc is

Each line is one character's **own** trajectory: a smoothed level of how the
passages that involve them leave them, gaining or losing standing, from the
first page to the last. It is not a head-to-head rating: nobody has to be
beaten for the line to rise. The model and its checks are in
[fortune_design.md](fortune_design.md).

Ratings use an Elo-style scale so readers can hold onto them:

- `1500` means the passages around that point leave the character where they
  were
- a gap of 200 points between two characters (in one lens) means the
  higher one's next passage goes better about 76% of the time; 100 points is
  about 64%, 400 about 91%
- every point carries `sd`, one standard deviation of uncertainty

## Input

One file, rebuilt by `python3 scripts/build_fortune.py`:

- `../proust/outputs/character-fortune-current.json` (~0.7 MB)

Do not read `outputs/fortune/*.json`; those are the analysis files.

## Data shape

Top level:

| field | meaning |
| --- | --- |
| `character_fortune_version` | `character_fortune_v1` |
| `corpus`, `view` | provenance (`enrichment`, `person`) |
| `x_axis` | x is the fraction of the novel's words before a point, 0 to 1 |
| `rating_center` | `1500` |
| `clear_move_rule` | how `clear` below is decided (`max_order_p`, `min_level_change`) |
| `lenses` | per lens: `points_per_level`, `arc_evidence`, `shows_arcs` |
| `volumes` | `[{volume, title, x}]`: where each of the seven volumes starts |
| `chapters` | `[{chapter_id, chapter_title, volume, x}]`: where each chapter starts |
| `characters` | one entry per character with enough passages for an arc (42) |

Each character:

| field | meaning |
| --- | --- |
| `character` | display name, the same string as in `character-pages-current.json` |
| `slug` | the character-page slug; join on this |
| `has_character_page` | `true` for the 23 characters with pages |
| `key` | registry id (stable; not for display) |
| `lenses` | `overall`, `advantage`, `prestige` where the character has enough passages |

Each lens entry:

| field | meaning |
| --- | --- |
| `passages_count` | passages behind the line |
| `line` | `[[x, rating, sd], ...]`, the smoothed arc, in order |
| `passages` | `[{x, outcome, unit_id, chapter_title, reader_link}]`, one dot per passage |
| `start`, `end` | `{x, rating, sd, chapter_title}` |
| `biggest_fall`, `biggest_rise` | `{points, from, to, order_p, clear}` or `null` |

`lenses.inclusion.shows_arcs` is `false`: its annotations are too sparse
to show arcs, so no character carries an `inclusion` entry. Do not add one.

## First rendering target

One chart module on the character page.

- Placement: below the portrait and editorial summary, near the existing
  Elo/timeline module if there is one.
- Lens: `overall` by default. If you add a switch, offer only lenses where
  `shows_arcs` is true and the character has an entry.
- Line: `line` as a 2px line, with a band of `rating ± sd` as a light fill.
- Dots: `passages` as small, low-contrast dots at `(x, outcome)`. Clamp them
  to the chart's y-range; individual passages swing much further than the
  line.
- Axis: x from 0 to 1, with light vertical rules at `volumes[].x` labelled
  I–VII. Faint horizontal rule at 1500.
- Y-range: fixed across characters within a lens, so arcs are comparable.
  `1500 ± 3 × points_per_level` covers every line and its band (about
  840–2160 for `overall`; the widest band is Vinteuil's late peak).
- Mark `biggest_fall` (or `biggest_rise` if it is the larger) with two
  endpoint dots **only when `clear` is true**.
- Hover: nearest `line` point shows `rating ± sd` and the chapter title
  (from `chapters`, by x). A dot can link to its `reader_link`.

It should read as: "the shape of what the novel does to this person".
It should not read as: a stock chart, a rank, or a verdict on the character.

## Display policy

Good framing:

- `Fortune across the novel`
- one sentence: "How the passages involving {name} leave them, from Combray
  to the Bal de têtes. 1500 is even; higher is better."

Say the move when it is `clear`, and only then: "Biggest fall: 479 points,
from Jeunes Filles to the Matinée."

Avoid:

- ranking characters by `end.rating` on the page (use the existing standings
  for rankings; fortune is about shape)
- describing a move whose `clear` is false as a finding
- comparing ratings across lenses (the scales differ slightly)

## Likely app seams

- `/Users/nathan_brixius/dev/brixius-web/lib/islt.ts`: a loader
  `getCharacterFortune(slug)` that reads the file once and returns
  `{ lenses, volumes, chapters, entry }` for one slug, or `null`
- a small `CharacterFortuneChart` component
- `/Users/nathan_brixius/dev/brixius-web/app/projects/islt/characters/[slug]/page.tsx`:
  render the component when the loader returns an entry

## Validation characters

If these look wrong, check x-axis units, the y-range, and lens selection
before anything else.

| character | expected in `overall` |
| --- | --- |
| baron de Charlus | high through Jeunes Filles (~1627), steep decline from Sodome et Gomorrhe, ends ~1150–1175; `biggest_fall.clear` true |
| M. Vinteuil | low in Combray, high in La Prisonnière; `biggest_rise.clear` true |
| Albertine | falls from Balbec to La Prisonnière; `clear` true |
| Mme Verdurin | rises to a wartime peak (~1509), then sinks at the Bal de têtes; no clear move |
| le narrateur | dense dots, a hump through Guermantes (~1589), ~1413 at his last annotated passage in M. de Charlus pendant la guerre; the vocation passages have no annotations, so the line stops before them |

## Short prompt

`Please add a fortune-arc chart to the existing ISLT character pages using ../proust/outputs/character-fortune-current.json. Read proust/docs/islt_fortune_arcs_handoff.md first and follow its data shape, rendering target and display policy. Join on slug. Default to the overall lens; offer advantage and prestige only where shows_arcs is true and the character has an entry. Draw the smoothed line with a ±sd band, faint per-passage dots, volume rules and a 1500 baseline, and mark the biggest move only when clear is true. Do not recompute anything in the app.`
