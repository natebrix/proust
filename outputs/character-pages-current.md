# Character Pages (scoring v2)

- Analysis version: `character_pages_v2`
- Scoring version: `scoring_v2`
- Source corpus summary: `scoring_v2_corpus_summary_v1`
- View: `name`
- Character count: `21`
- Corpus: `foundation`

## Profile shape

`profile.lens_scores[lens]` is scoring v2 and no longer carries v1 net scores, percentiles, or score spans. Its keys are:

- `rating`, `band`, `conservative_rating`: the weighted-WHR standing at the character's last node, the `2*sigma` band around it, and `rating - band`
- `rank`, `non_provisional_count`: dense rank by conservative rating among the lens's ranked characters, and how many characters that set holds. `rank` is `null` whenever `provisional` is true -- a wide band is missing evidence, not a low placement
- `provisional`: true when the band still exceeds the fit's threshold
- `appearances`: annotated units the character is present in (lens-independent)
- `mean_movement`, `mean_absolute_movement`: direction and intensity per appearing unit. Both are means, never sums, so appearing often cannot raise either
- `labels`: positive / negative / mixed / neutral unit counts in this lens
- `comparison_count`: weighted comparisons the character took part in

`profile.archetype_signs` gives the sign of each lens's rating against the initial rating: the three-way signature the lens-polarity archetypes are read from. `top_chapters` and `notable_units` are selected by v2 absolute movement, and a notable unit's label is the annotator's own explanation of the largest effect in it.

## le narrateur

- Slug: `le-narrateur`
- Portrait default: `/projects/islt/portraits/le-narrateur-default-vermeer-proustian-20260807-1130.png`
- Annotation units: `316`
- Archetype signs: `advantage -1, prestige +1, inclusion +1`
- Pattern: `relational_positive_understated`

He loses the scene and keeps the room: near the bottom in scene-level advantage, yet first in belonging and near the top in standing among the figures the novel lets us measure.

The narrator is the novel's "I": nearly every scene passes through him, and scene by scene the scenes go badly — snubs registered, composure lost, comparisons endured. Yet across the whole book his welcome never runs out: he ranks first in belonging and near the top in visible standing among the characters the text weighs often enough to judge. The rooms keep receiving the man the scenes keep wounding, and that split — lived defeat inside durable acceptance — is the book's central irony made measurable.

Why interesting:

- His three readings pull apart more sharply than anyone's: last third in scene-level advantage, first in belonging — the same passages, weighed differently.
- Because the whole novel passes through him, he is measured against more of the cast than any other figure, so his readings are among the most certain in the book.
- His suffering is local and his acceptance is cumulative: no single scene secures his place, and no single defeat costs it.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1369 ± 87 | 1281.5 | 26 of 35 | 316 | -0.304 | 0.6984 | 65/135/5/111 |
| prestige | 1702 ± 171 | 1531.3 | 2 of 8 | 316 | +0.026 | 0.0629 | 16/6/0/294 |
| inclusion | 1520 ± 101 | 1418.8 | 1 of 9 | 316 | +0.051 | 0.2198 | 33/28/0/255 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v5 | 38 | -0.583 | +0.039 | -0.022 |
| v2-p1-autour-de-mme-swann | 34 | -0.561 | +0.071 | +0.172 |
| v3-p1 | 57 | -0.063 | -0.017 | +0.087 |
| v2-p2-noms-de-pays-le-pays | 40 | -0.292 | -0.001 | -0.132 |
| v3-p2 | 48 | -0.021 | +0.03 | +0.269 |

Reading path:

- Balbec thresholds: the machinery of being received: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Guermantes admission: the observer absorbed: `/projects/islt/fr-original/v3-p2`
- The bal de têtes: survivor among the masks: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- The narrator's local self-regard and sense of public standing rise sharply upon reading his own published, admired article.: `/projects/islt/fr-original/v6-p2#p-11`
- The narrator explicitly indicts his past self as ungrateful, selfish, and cruel toward his grandmother.: `/projects/islt/fr-original/v4-p2#p-191`
- The narrator is directly and repeatedly praised as exceptionally intelligent, placed in the company of Elstir and implicitly of great novelists.: `/projects/islt/fr-original/v3-p1#p-206`

## Swann

- Slug: `swann`
- Portrait default: `/projects/islt/portraits/swann-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `202`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `broad_negative`

The most heavily measured man in the novel after the narrator and the duchesse: his scenes run relentlessly negative, but his standing in advantage still holds in the upper third.

Swann is staged constantly — 202 scenes, more than all but the narrator and the duchesse — and scene by scene the losses pile up: 81 negative outcomes against 42 positive in advantage alone, a real negative pull with real intensity. Yet his overall standing there is not a collapse: he ranks 13th of the 35 characters substantial enough to size, solidly in the upper third — presence and volume outweighing any run of bad scenes. Belonging is a cleaner loss, 7th of the 9 characters the novel stages enough to size there. Prestige is where the evidence runs thinnest: his scenes trend only faintly negative, and the novel does not stage him in high-status contests often enough to rank him there at all.

Why interesting:

- He is the most heavily measured man in the pilot set — more scenes than anyone but the narrator and the duchesse — so his readings carry unusual weight.
- His advantage story splits: relentlessly negative scene by scene (81 losses to 42 gains), yet his standing still lands in the upper third (13th of 35).
- Belonging is where he loses cleanly — 7th of the 9 characters ranked there — while prestige stays too thin to size.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1475 ± 111 | 1363.8 | 13 of 35 | 202 | -0.314 | 0.6576 | 42/81/0/79 |
| prestige | 1385 ± 221 | 1164.3 | insufficient evidence | 202 | -0.014 | 0.1934 | 15/20/0/167 |
| inclusion | 1287 ± 127 | 1159.9 | 7 of 9 | 202 | -0.067 | 0.1169 | 4/16/0/182 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 98 | -0.502 | -0.013 | -0.094 |
| v2-p1-autour-de-mme-swann | 25 | -0.245 | -0.025 | 0.0 |
| v3-p2 | 20 | +0.118 | -0.048 | -0.007 |
| v4-p2 | 15 | -0.481 | -0.232 | -0.113 |
| v1-p1-combray | 11 | -0.314 | +0.34 | 0.0 |

Reading path:

- Primary negative concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Early counterweight and setup: `/projects/islt/fr-original/v1-p1-combray`
- Later negative reinforcement: `/projects/islt/fr-original/v4-p2`

Notable units:

- The narrator's explicit verdicts — lazy-minded, uninventive, as much a liar as Odette, no less egoistic, not made better by his love of truth — lower Swann sharply within the passage.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-536`
- Swann loses all leverage: he must beg intermediaries for a meeting, is refused in public, and ends the passage weeping and desiring death to escape the monotony of his pursuit.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-411`
- Swann is left talking aloud to himself in the Bois, convulsed with disgust and jealousy, imagining Odette laughing at him beside a rival; he has lost all leverage over the salon and over Odette's evening.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-361`

## duchesse de Guermantes

- Slug: `duchesse-de-guermantes`
- Portrait default: `/projects/islt/portraits/duchesse-de-guermantes-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `199`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `uniform_positive`

One of the few figures the novel weighs in every register — and she holds the elite in standing and belonging while her scenes cut both ways.

duchesse de Guermantes is among the handful of characters the text stages often enough to measure in all three lenses at once. Her standing and her belonging hold at the top of that small measurable circle; her scene-level outcomes are mixed, because her chief instrument — the wit — wounds its owner as often as it crowns her. She is not uniformly triumphant; she is durably central, which in this novel is the rarer thing.

Why interesting:

- She is one of the only characters with enough comparative evidence to be ranked in standing and belonging, not just scene-level advantage — a measure of how much of the book is staged around her.
- Her scene outcomes and her standing tell different stories: individual evenings are won and lost, the position endures.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1461 ± 104 | 1356.5 | 14 of 35 | 199 | +0.051 | 0.4851 | 65/46/4/84 |
| prestige | 1588 ± 167 | 1421.0 | 6 of 8 | 199 | +0.163 | 0.2065 | 29/6/0/164 |
| inclusion | 1479 ± 163 | 1315.8 | 3 of 9 | 199 | -0.004 | 0.004 | 0/1/0/198 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 83 | -0.004 | +0.158 | 0.0 |
| v3-p1 | 53 | +0.213 | +0.26 | -0.015 |
| v5 | 8 | +0.27 | +0.094 | 0.0 |
| v7-p4-le-bal-de-tetes | 11 | -0.511 | -0.071 | 0.0 |
| v4-p2 | 15 | -0.04 | +0.165 | 0.0 |

Reading path:

- High Guermantes concentration: `/projects/islt/fr-original/v3-p1`
- Continued positive confirmation: `/projects/islt/fr-original/v3-p2`
- Late reinforcing appearances: `/projects/islt/fr-original/v4-p2`

Notable units:

- The narrator's sustained account of her decayed wit and comparison to the socially diminished Mme de Villeparisis marks a clear negative shift in how she is presented.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-76`
- The sustained narrator analysis reframes her famous discernment as arbitrary, capricious, and hollowed out by boredom, and her wit as calculated malice rehearsed for an audience.: `/projects/islt/fr-original/v3-p2#p-316`
- She is praised as a gifted, authentic storyteller whose speech is likened to a living museum of French history.: `/projects/islt/fr-original/v5#p-61`

## Robert de Saint-Loup

- Slug: `robert-de-saint-loup`
- Portrait default: `/projects/islt/portraits/saint-loup-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `168`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

One of the few figures ranked in all three registers at once — solid in scene-level advantage and belonging, but near the bottom of the small circle the novel sizes in prestige.

Saint-Loup is one of the few characters the novel stages enough to rank in advantage, prestige, and belonging all at once. His footing is comfortably above the middle in scene-level advantage (15th of 35, a mild pull across 168 appearances split 57 losses to 39 gains) and solid in belonging (4th of 9). But in prestige — the register his aristocratic bearing would predict he'd own — he ranks next to last of the eight characters substantial enough to size there (7th of 8), his scene-by-scene movement essentially flat rather than commanding. He is present and accepted more than he is deferred to.

Why interesting:

- He is one of the only figures ranked in all three lenses at once, a completeness the novel affords barely a handful of its cast.
- His prestige position inverts what his rank and bearing would suggest: 7th of the 8 characters substantial enough to size there, essentially flat scene by scene.
- His advantage losses outnumber his gains (57 to 39 across 168 scenes) yet he still lands comfortably above the midpoint of the ranked set (15th of 35) — breadth of presence outweighing any single bad run.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1457 ± 107 | 1350.5 | 15 of 35 | 168 | -0.107 | 0.6038 | 39/57/1/71 |
| prestige | 1589 ± 184 | 1404.9 | 7 of 8 | 168 | -0.0 | 0.1601 | 13/16/0/139 |
| inclusion | 1427 ± 163 | 1264.4 | 4 of 9 | 168 | -0.011 | 0.0205 | 1/2/0/165 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 88 | -0.175 | -0.046 | -0.01 |
| v2-p2-noms-de-pays-le-pays | 26 | +0.114 | +0.046 | -0.069 |
| v3-p2 | 18 | -0.058 | +0.077 | +0.044 |
| v7-p1-a-tansonville | 4 | -1.525 | +0.17 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 5 | +1.22 | +0.16 | 0.0 |

Reading path:

- Main prestige / inclusion divergence: `/projects/islt/fr-original/v3-p1`
- Earlier positive concentration: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Late negative pressure: `/projects/islt/fr-original/v7-p1-a-tansonville`

Notable units:

- Saint-Loup is locally reduced to visible suffering under a woman who controls the scene; his agony is displayed to a rival rather than answered.: `/projects/islt/fr-original/v3-p1#p-396`
- The passage is organized around exposing Saint-Loup: his marriage neglected, his women a useless screen, his elegance reinterpreted as furtive concealment of vice.: `/projects/islt/fr-original/v7-p1-a-tansonville#p-1`
- His once-private military theorizing is shown to have anticipated real historical developments and to align with expert critical analysis, posthumously enhancing his intellectual reputation.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-61`

## Albertine

- Slug: `albertine`
- Portrait default: `/projects/islt/portraits/albertine-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `146`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `broad_negative_advantage_standing_holds`

Her scenes run negative and volatile — more internally conflicted than any other figure in the pilot set — yet her standing in advantage holds near the top quarter; belonging is where she is dead last.

Albertine's scenes are the most volatile of anyone examined here: 40 positive outcomes, 56 negative, and 10 explicitly mixed — passages where the text registers gain and loss in the same breath — more internal conflict than any other pilot figure. That volatility pulls her mean movement in advantage negative, yet her standing there remains strong: 9th of the 35 characters substantial enough to rank, solidly in the top quarter. Belonging tells a starker story: she is last of the nine characters the novel stages enough to size there, the clearest exclusion in the set. Prestige carries almost no signal at all, and the novel does not stage her in that register often enough to rank her.

Why interesting:

- She has more mixed-outcome scenes than any other pilot figure — 10 of 146 appearances register gain and loss simultaneously, a genuine internal split rather than a one-directional slide.
- Her advantage standing (9th of 35) sits well above what her negative mean movement alone would suggest — volatility, not collapse.
- In belonging she is dead last of the nine ranked characters — the clean, unambiguous loss in her profile.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1483 ± 96 | 1386.9 | 9 of 35 | 146 | -0.173 | 0.7048 | 40/56/10/40 |
| prestige | 1329 ± 257 | 1072.1 | insufficient evidence | 146 | -0.017 | 0.0514 | 4/4/0/138 |
| inclusion | 1293 ± 167 | 1126.1 | 9 of 9 | 146 | -0.071 | 0.0901 | 2/10/0/134 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v5 | 57 | -0.241 | -0.087 | -0.107 |
| v6-p1 | 21 | -0.077 | 0.0 | -0.172 |
| v2-p2-noms-de-pays-le-pays | 22 | +0.353 | +0.083 | 0.0 |
| v3-p2 | 16 | -0.126 | +0.044 | 0.0 |
| v4-p2 | 19 | -0.394 | 0.0 | -0.032 |

Reading path:

- Main negative concentration in La Prisonnière: `/projects/islt/fr-original/v5`
- Afterlife of loss in Albertine disparue: `/projects/islt/fr-original/v6-p1`
- Continuing exclusion pressure: `/projects/islt/fr-original/v6-p2`

Notable units:

- Albertine's remembered image is reduced from a radiant girl to an unflattering, coarse, aged figure resembling Mme Bontemps.: `/projects/islt/fr-original/v6-p3#p-51`
- Albertine's apparent freedom and happiness restore her, in the narrator's eyes, to the elevated, desirable status she held at the start of his infatuation.: `/projects/islt/fr-original/v6-p1#p-51`
- Albertine's repeated, successive admissions of lying about the Balbec trip, her relations with Mlle Vinteuil, and the Andrée and Léa deceptions locally destroy her credibility and expose her as compulsively mendacious.: `/projects/islt/fr-original/v5#p-341`

## Odette

- Slug: `odette`
- Portrait default: `/projects/islt/portraits/odette-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `142`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

One of the few women the novel ranks in two registers at once — mid-table in scene-level advantage and in belonging alike — with a lean toward prestige too thinly staged to size.

Odette is substantial enough for the novel to rank in two separate registers: 22nd of 35 in scene-level advantage, 5th of 9 in belonging, each direction mildly negative rather than sharply so. Her clearest lean is toward prestige, where her scenes trend positive, but the novel does not stage her in enough high-status scenes to size that claim. The picture is not a dramatic split so much as steady, unglamorous footing in the rooms the book lets us measure, with prestige left an open question.

Why interesting:

- She is one of only nine characters the novel stages enough to rank in belonging at all, and she sits mid-table there (5th of 9), ahead of Swann, Mme Verdurin, and Albertine.
- Her prestige movement leans positive, but it is the one register the novel doesn't stage enough to rank — the reverse of where the old reading placed her strength.
- In scene-level advantage, negative scenes outnumber positive ones only modestly (36 to 23 across 142 appearances), landing her 22nd of 35 — a mild pull, not a severe one.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1435 ± 123 | 1311.8 | 22 of 35 | 142 | -0.134 | 0.3862 | 23/36/3/80 |
| prestige | 1689 ± 203 | 1485.5 | insufficient evidence | 142 | +0.039 | 0.151 | 12/9/0/121 |
| inclusion | 1348 ± 154 | 1194.5 | 5 of 9 | 142 | -0.037 | 0.0711 | 2/7/0/133 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 67 | -0.16 | -0.012 | -0.004 |
| v2-p1-autour-de-mme-swann | 34 | -0.038 | +0.033 | -0.101 |
| v1-p3-noms-de-pays-le-nom | 6 | -0.123 | +0.572 | -0.12 |
| v4-p2 | 3 | -0.5 | +0.903 | +0.6 |
| v3-p1 | 7 | -0.236 | -0.114 | -0.386 |

Reading path:

- Mild prestige lean around Mme Swann: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Negative counterweight in Swann's love: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Later reversals in Guermantes-adjacent society: `/projects/islt/fr-original/v3-p1`

Notable units:

- She is exposed as a practiced but unskillful liar whose fabrications are transparent to Swann, and the narrator generalizes her behavior as habitual mendacity.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-331`
- Swann's speech directly denies Odette's status as a person and calls her contemptible and unintelligent, sharply lowering how she is evaluated within the passage.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-371`
- She is explicitly barred from the valued social space: admitted once as a nuisance to be forewarned about, never to be received again, and avoided in person by the duchesse.: `/projects/islt/fr-original/v3-p1#p-686`

## baron de Charlus

- Slug: `baron-de-charlus`
- Portrait default: `/projects/islt/portraits/charlus-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `119`
- Archetype signs: `advantage -1, prestige -1, inclusion +1`
- Pattern: `prestige_declining_advantage_negative`

Last of the eight ranked in prestige despite scenes that lean mildly positive — a standing that erodes late in the book even as day-to-day encounters do not; lower-third in scene-level advantage, and belonging too faint to size.

Charlus's prestige is the clearest case of standing pulling against direction in the pilot set: his individual scenes trend mildly positive on balance, yet he ends last of the eight characters substantial enough to rank in that register — a decline concentrated in the book's later volumes (his prestige rating falls sharply across the wartime chapter) rather than visible scene by scene. His scene-level advantage is more straightforwardly negative: intensely volatile, among the largest swing patterns examined here, and ranked in the lower third, 23rd of 35. Belonging, by contrast, is barely touched — the novel does not stage him there often enough to size a claim, and what little there is sits essentially flat.

Why interesting:

- He is the sharpest case where a mildly positive day-to-day trend and a last-place standing coexist — his overall prestige position erodes late, not scene by scene.
- His advantage volatility is among the highest measured: swings between gain and loss are large and frequent, even though his overall standing (23rd of 35) is merely lower-third, not catastrophic.
- Belonging carries almost no signal — the quietest of his three readings.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1407 ± 98 | 1309.3 | 23 of 35 | 119 | -0.3 | 0.7039 | 29/45/2/43 |
| prestige | 1447 ± 160 | 1287.2 | 8 of 8 | 119 | +0.041 | 0.2677 | 13/12/1/93 |
| inclusion | 1786 ± 235 | 1550.5 | insufficient evidence | 119 | +0.012 | 0.0118 | 2/0/0/117 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v4-p2 | 33 | -0.394 | +0.31 | +0.042 |
| v5 | 18 | -0.444 | -0.051 | 0.0 |
| v3-p2 | 17 | -0.159 | -0.038 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 7 | -1.322 | -0.624 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 11 | -0.103 | +0.154 | 0.0 |

Reading path:

- Salon-world negative pressure: `/projects/islt/fr-original/v4-p2`
- Late negative cluster with Morel: `/projects/islt/fr-original/v5`
- Wartime degradation: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

Notable units:

- Charlus is clearly lowered in this passage: wounded, mincing, credulous, playing at slang before boys who lie to him, he is presented as a dupe of the underworld theatre he pays to have staged.: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre#p-51`
- Charlus is clearly diminished as the narrator lays bare his moral and physical decline, self-delusion, and the collapse of the mask he once maintained.: `/projects/islt/fr-original/v5#p-281`
- Charlus's aged, made-up appearance exposed by daylight and his undignified, transactional pursuit of Morel diminish him relative to his usual polished, commanding social image.: `/projects/islt/fr-original/v4-p2#p-301`

## duc de Guermantes

- Slug: `duc-de-guermantes`
- Portrait default: `/projects/islt/portraits/duc-de-guermantes-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `110`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `advantage_reversed_high_title`

The formal title promises status the numbers don't deliver: one of the most lopsided negative scene-level readings measured, while prestige and belonging register almost no movement at all.

duc de Guermantes carries the book's most prestigious hereditary title, but his measured standing does not follow it: in scene-level advantage he ranks 27th of 35, with one of the most lopsided negative records examined here — 48 losing outcomes against just 5 positive ones across 110 appearances. Prestige and belonging tell a different, quieter story: the novel simply does not stage him often enough in either register to size a claim, and what little is there sits essentially flat rather than negative. The reversal is real, but it belongs to advantage alone — his title does not protect him in the room, even as the more rarefied registers stay nearly silent on him.

Why interesting:

- His advantage record is one of the most lopsided negative readings in the pilot set: 48 losses to just 5 gains, landing him 27th of 35 despite his formal rank.
- Prestige and belonging are not negative so much as nearly empty — mean movement close to zero in both, the novel giving him too little space there to size any claim.
- He sharpens the distinction between inherited title and measured standing: the numbers separate the two cleanly.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1396 ± 116 | 1279.8 | 27 of 35 | 110 | -0.351 | 0.4264 | 5/48/1/56 |
| prestige | 1367 ± 205 | 1162.3 | insufficient evidence | 110 | -0.005 | 0.1043 | 5/4/0/101 |
| inclusion | 1413 ± 223 | 1189.9 | insufficient evidence | 110 | 0.0 | 0.0 | 0/0/0/110 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 63 | -0.337 | +0.022 | 0.0 |
| v3-p1 | 21 | -0.459 | +0.165 | 0.0 |
| v4-p2 | 13 | -0.35 | 0.0 | 0.0 |
| v7-p4-le-bal-de-tetes | 3 | -0.307 | -1.2 | 0.0 |
| v5 | 1 | -0.64 | -1.8 | 0.0 |

Reading path:

- Primary Guermantes counterexample: `/projects/islt/fr-original/v3-p2`
- Late decline reinforcement: `/projects/islt/fr-original/v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`
- Final negative return: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- Explicit narrator judgment exposes him as prideful rather than genuinely knowledgeable about art.: `/projects/islt/fr-original/v3-p2#p-526`
- The duc is locally diminished by the narrator's exposure of his marital brutality and the vain, imitative nature of his proudest rhetorical flourish.: `/projects/islt/fr-original/v3-p1#p-626`
- The narrator's direct, endorsed commentary condemns the duc's tactlessness and self-regard amid a death in the household, clearly lowering his local standing.: `/projects/islt/fr-original/v3-p2#p-61`

## Françoise

- Slug: `francoise`
- Portrait default: `/projects/islt/portraits/francoise-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `82`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `advantage_top_ranked_positive`

The single highest standing in scene-level advantage of anyone the novel measures — first of 35 — even though her day-to-day scenes are close to an even split of gain and loss.

Françoise holds the highest standing in scene-level advantage of any character substantial enough to rank — first of 35. It is not a story of one-sided triumph: her scenes split nearly evenly, 20 positive against 18 negative across 82 appearances, and her mean movement is only mildly positive. Her standing comes from consistency and breadth rather than a run of decisive wins. In prestige and belonging the novel does not stage her often enough to rank her, though what evidence exists leans mildly positive in prestige and essentially flat in belonging.

Why interesting:

- She ranks first of the 35 characters substantial enough to size in scene-level advantage — the single highest standing in the register the novel measures most, ahead of every aristocrat, artist, and lover in the set.
- That standing is not built on one-sided scenes: her outcomes split almost evenly (20 positive, 18 negative), so the ranking reflects sustained, durable footing rather than a hot streak.
- Prestige and belonging stay unranked — too little is staged there to size a claim, though both lean toward positive or flat rather than negative.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1648 ± 141 | 1506.9 | 1 of 35 | 82 | +0.12 | 0.5288 | 20/18/2/42 |
| prestige | 1560 ± 272 | 1287.7 | insufficient evidence | 82 | +0.036 | 0.071 | 6/2/0/74 |
| inclusion | 1595 ± 271 | 1324.3 | insufficient evidence | 82 | -0.018 | 0.0177 | 0/2/0/80 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p1-combray | 22 | +0.141 | 0.0 | 0.0 |
| v3-p2 | 8 | -0.404 | -0.177 | 0.0 |
| v2-p1-autour-de-mme-swann | 4 | +1.165 | +0.2 | 0.0 |
| v4-p2 | 8 | -0.2 | 0.0 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 7 | +0.529 | +0.2 | 0.0 |

Reading path:

- Early domestic concentration: `/projects/islt/fr-original/v1-p1-combray`
- Strongest positive concentration, in Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- The rare negative pocket in an otherwise positive record: `/projects/islt/fr-original/v4-p2`

Notable units:

- Françoise is portrayed as devoted, hard-working, and morally admirable, explicitly contrasted with servants whose surface charm masks an 'inéducable nullité.': `/projects/islt/fr-original/v1-p1-combray#p-56`
- Françoise is wounded by the narrator's calculated cruelty, reacting with a breathless, barely intelligible response.: `/projects/islt/fr-original/v4-p2#p-166`
- Françoise's reputed saintliness and tenderness are locally undercut by the narrator's revelation of her calculated, wasp-like cruelty toward dependents she can dominate.: `/projects/islt/fr-original/v1-p1-combray#p-261`

## Mme Verdurin

- Slug: `mme-verdurin`
- Portrait default: `/projects/islt/portraits/mme-verdurin-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `82`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Ranked in all three registers — rare completeness — with prestige alone bucking the trend: mid-table and positive there while advantage and belonging pull down.

Mme Verdurin is one of the few figures the novel stages enough to rank in advantage, prestige, and belonging all at once. Prestige is where she holds real ground: 5th of the eight characters substantial enough to size there, with a genuinely positive scene-by-scene trend. Advantage tells a harsher story in the moment — scenes run heavily negative, 28 losses against only 5 gains — yet her overall standing still lands in the top third (12th of 35), presence and consistency outweighing the lopsided texture. Belonging is her clearest weak point: 8th of 9, near the bottom of the small ranked circle.

Why interesting:

- She is ranked in all three registers, a completeness only a handful of characters in the pilot set achieve.
- Prestige runs counter to her salon-world reputation: she holds a genuinely positive, mid-table standing there (5th of 8) even as her other two readings run negative.
- Her advantage scenes are lopsidedly negative (28 to 5) yet her standing still holds in the top third — volume and durability outweighing the texture of individual encounters.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1488 ± 121 | 1367.7 | 12 of 35 | 82 | -0.299 | 0.3451 | 5/28/0/49 |
| prestige | 1638 ± 170 | 1468.2 | 5 of 8 | 82 | +0.07 | 0.2228 | 11/8/0/63 |
| inclusion | 1334 ± 181 | 1152.6 | 8 of 9 | 82 | -0.056 | 0.0743 | 1/3/0/78 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 44 | -0.362 | +0.052 | 0.0 |
| v4-p2 | 14 | -0.319 | -0.041 | +0.054 |
| v7-p2-m-de-charlus-pendant-la-guerre | 6 | -0.317 | +0.663 | 0.0 |
| v5 | 9 | +0.073 | -0.278 | -0.411 |
| v7-p4-le-bal-de-tetes | 3 | -0.483 | +0.567 | 0.0 |

Reading path:

- Primary Verdurin-world concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Late negative counterpoint: `/projects/islt/fr-original/v5`
- Wartime reversal zone: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

Notable units:

- She is displaced from the center of her own reception: unrecognized, unpresented, left alone while her guests form a group apart around Charlus.: `/projects/islt/fr-original/v5#p-301`
- Mme Verdurin is clearly diminished as the narrator unmasks her declared horror at wartime deaths as superficial performance beneath which lies petty physical contentment.: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre#p-31`
- She is systematically ignored, unrecognized by departing guests, and structurally excluded from credit at her own salon.: `/projects/islt/fr-original/v5#p-311`

## la grand-mère

- Slug: `la-grand-mere`
- Portrait default: `/projects/islt/portraits/la-grand-mere-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `80`
- Archetype signs: `advantage +1, prestige -1, inclusion +1`
- Pattern: `advantage_strongly_positive`

Second of 35 in scene-level advantage — one of the highest standings in the pilot set — built on genuinely positive scenes; prestige and belonging stay too thin to rank, belonging leaning mildly negative.

la grand-mère holds one of the highest standings measured in scene-level advantage: 2nd of the 35 characters substantial enough to rank, behind only Françoise. Unlike several of the book's other high-standing figures, this position is not a story of standing surviving bad scenes: her outcomes are genuinely positive, 25 gains against 17 losses across 80 appearances. Prestige and belonging are both too thinly staged to rank: prestige reads essentially flat, and belonging leans mildly negative, though neither rests on enough evidence to call a real standing.

Why interesting:

- She ranks 2nd of 35 in scene-level advantage — one of only two figures in the pilot set, with Françoise, whose standing there is this high.
- Unlike Swann, Norpois, or Albertine, her high standing is matched by genuinely positive scenes rather than surviving a negative-leaning texture (25 gains to 17 losses).
- Belonging leans mildly negative in direction, but the novel does not stage her there often enough to rank the claim.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1654 ± 165 | 1489.4 | 2 of 35 | 80 | +0.138 | 0.639 | 25/17/2/36 |
| prestige | 1461 ± 336 | 1125.1 | insufficient evidence | 80 | +0.012 | 0.0548 | 2/2/0/76 |
| inclusion | 1523 ± 218 | 1304.9 | insufficient evidence | 80 | -0.049 | 0.0489 | 0/4/0/76 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 14 | -0.133 | 0.0 | -0.054 |
| v2-p2-noms-de-pays-le-pays | 31 | +0.242 | +0.032 | -0.052 |
| v3-p1 | 10 | -0.037 | 0.0 | -0.156 |
| v4-p2 | 6 | +0.607 | 0.0 | 0.0 |
| v1-p1-combray | 12 | +0.125 | 0.0 | 0.0 |

Reading path:

- Main positive concentration, in Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Early family-world footing: `/projects/islt/fr-original/v1-p1-combray`
- Guermantes-world counterweight: `/projects/islt/fr-original/v3-p1`

Notable units:

- The narrator's « Sans toi je ne pourrais pas vivre » gives her total emotional leverage, which she then uses to counsel him toward a harder heart.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-161`
- She is shown, through the eyes of passersby like Legrandin, as visibly disheveled and overwhelmed, a stark diminishment of her usual composed public presence.: `/projects/islt/fr-original/v3-p2#p-6`
- Both the dream vision and the photograph mark her with the visible signs of fatal illness, an 'air de condamnée à mort' that maman experiences as an insult done to her mother's face.: `/projects/islt/fr-original/v4-p2#p-206`

## Mme de Villeparisis

- Slug: `mme-de-villeparisis`
- Portrait default: `/projects/islt/portraits/mme-de-villeparisis-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `79`
- Archetype signs: `advantage -1, prestige -1, inclusion +1`
- Pattern: `advantage_midtable_thin_elsewhere`

A mid-table figure in scene-level advantage (19th of 35); the novel stages her too rarely in prestige and belonging to rank her in either, though her scenes lean mildly opposite ways there.

Mme de Villeparisis is substantial enough to rank in scene-level advantage, where she lands almost exactly at the middle of the measured cast (19th of 35), a mild negative pull across 79 appearances. In prestige and belonging the novel simply doesn't stage her often enough to size a standing: what evidence exists tilts in opposite directions, mildly positive in prestige and mildly negative in belonging, but neither claim can be made with confidence. She reads less as a split figure than as a moderate, largely unremarkable presence whose one measurable register places her squarely in the pack.

Why interesting:

- She is ranked in advantage while going unranked in both other registers — a genuinely partial picture rather than a clean split.
- Her prestige and belonging readings point in opposite directions, but both rest on too little staged evidence to call either a real standing.
- Her advantage placement (19th of 35) is almost exactly the median of the ranked cast — a useful baseline case of moderate, unremarkable footing.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1484 ± 144 | 1340.8 | 19 of 35 | 79 | -0.139 | 0.3257 | 12/19/0/48 |
| prestige | 1453 ± 250 | 1202.3 | insufficient evidence | 79 | +0.02 | 0.24 | 9/7/0/63 |
| inclusion | 1550 ± 204 | 1345.6 | insufficient evidence | 79 | -0.029 | 0.0294 | 0/2/0/77 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 39 | +0.006 | +0.03 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 21 | -0.131 | +0.123 | -0.111 |
| v3-p2 | 9 | -0.388 | -0.167 | 0.0 |
| v6-p3 | 5 | -0.841 | -0.132 | 0.0 |
| v1-p1-combray | 1 | -0.78 | 0.0 | 0.0 |

Reading path:

- Main split concentration: `/projects/islt/fr-original/v3-p1`
- Brief positive lean in Balbec prestige: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Late negative counterweight: `/projects/islt/fr-original/v6-p3`

Notable units:

- Her aristocratic standing is locally erased: servants classify her as a tiresome foreigner unfit for a smart hotel, and her rank is legible only to the narrator.: `/projects/islt/fr-original/v6-p3#p-6`
- Within the passage, Mme de Villeparisis's local standing rises sharply in the narrator's retrospective account once her close Guermantes kinship is revealed.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-201`
- The narrator explicitly and at length attributes her lasting déchéance mondaine and déclassement to qualities that alienate fashionable society, illustrated by Mme Leroi's disdainful greetings.: `/projects/islt/fr-original/v3-p1#p-416`

## Gilberte

- Slug: `gilberte`
- Portrait default: `/projects/islt/portraits/gilberte-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `76`
- Archetype signs: `advantage -1, prestige +1, inclusion +1`
- Pattern: `inclusion_positive_prestige_positive_advantage_negative`

Second only to the narrator in belonging (2nd of 9) and comfortably placed in prestige (4th of 8) — her weakest register is scene-level advantage, where she sits below the middle of the ranked cast.

Gilberte's strongest reading is belonging: she ranks 2nd of the nine characters substantial enough to size there, behind only the narrator himself, with a scene-by-scene trend that's essentially flat but tilted positive. Prestige is comfortably positive too, 4th of the eight characters ranked in that register. Scene-level advantage is where she is weakest — 21st of 35, a mild negative pull across 76 appearances (19 negative outcomes to 11 positive). The overall shape is a young woman more secure in standing and welcome than she is in any single encounter.

Why interesting:

- She ranks 2nd of 9 in belonging, second only to the narrator himself — an unusually high position for a figure this compact.
- She is one of the few characters ranked in all three registers at once, and the pattern runs opposite to what her salon polish might suggest: belonging and prestige are her strengths, scene-level advantage her weak point.
- Her advantage losses outnumber her gains (19 to 11), though the pull is mild rather than severe (21st of 35).

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1418 ± 105 | 1312.5 | 21 of 35 | 76 | -0.063 | 0.4118 | 11/19/2/44 |
| prestige | 1643 ± 162 | 1481.8 | 4 of 8 | 76 | +0.098 | 0.1337 | 8/2/0/66 |
| inclusion | 1545 ± 174 | 1371.1 | 2 of 9 | 76 | +0.008 | 0.0345 | 2/1/0/73 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v6-p2 | 8 | -0.449 | +0.305 | +0.072 |
| v2-p1-autour-de-mme-swann | 28 | -0.077 | 0.0 | 0.0 |
| v1-p3-noms-de-pays-le-nom | 8 | +0.667 | 0.0 | 0.0 |
| v6-p4 | 5 | -0.24 | +0.508 | 0.0 |
| v1-p1-combray | 4 | +0.745 | +0.61 | 0.0 |

Reading path:

- Early positive concentration: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom`
- Mme Swann-world extension: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Late instability in belonging: `/projects/islt/fr-original/v6-p2`

Notable units:

- The passage raises her to the level of a Poussin sky and an apparition of the gods, making her the luminous centre of the episode.: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom#p-6`
- The same reversal that erases her father's name raises her sharply: she inherits an enormous fortune, becomes 'une des plus riches héritières de France', and is adopted into the noble name of Forcheville.: `/projects/islt/fr-original/v6-p2#p-16`
- The marriage carries Gilberte into the Guermantes as marquise de Saint-Loup, and society people who had ignored her now seek her out and study her.: `/projects/islt/fr-original/v6-p4#p-1`

## Bloch

- Slug: `bloch`
- Portrait default: `/projects/islt/portraits/bloch-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `71`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Near the bottom of the ranked cast in scene-level advantage (32nd of 35, his clearest weak point) — yet third of the eight characters ranked in prestige, a standing that holds despite scenes that lean mildly negative there too.

Bloch's advantage reading is one of the harshest in the pilot set: 32nd of 35, with a heavily lopsided scene record (37 negative outcomes against 6 positive) across 71 appearances. Belonging is also a genuine loss, 6th of the nine characters ranked there, moderately negative in direction. But prestige breaks the pattern entirely: he ranks 3rd of the eight characters substantial enough to size there, a real standing, even though his individual prestige scenes lean mildly negative on balance. He is cut down constantly in the room, and still commands more prestige-standing than all but two of the book's most-measured figures.

Why interesting:

- His advantage reading is among the harshest measured — 32nd of 35, with negative scenes outnumbering positive ones better than six to one.
- Prestige inverts the expectation his advantage reading sets up: he ranks 3rd of 8, ahead of every pilot figure but Gilberte and Mme Verdurin in that register, even though the scenes themselves skew mildly negative.
- Belonging sits in between — a real, ranked loss (6th of 9) but not the worst in the small circle.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1349 ± 121 | 1227.8 | 32 of 35 | 71 | -0.589 | 0.7328 | 6/37/1/27 |
| prestige | 1689 ± 187 | 1502.4 | 3 of 8 | 71 | -0.067 | 0.1586 | 3/8/0/60 |
| inclusion | 1362 ± 184 | 1177.9 | 6 of 9 | 71 | -0.12 | 0.1775 | 3/10/0/58 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 25 | -0.672 | -0.112 | -0.165 |
| v2-p2-noms-de-pays-le-pays | 11 | -0.626 | -0.134 | 0.0 |
| v1-p1-combray | 5 | -0.378 | 0.0 | -0.852 |
| v7-p4-le-bal-de-tetes | 9 | -0.456 | +0.296 | -0.009 |
| v4-p2 | 7 | -0.579 | 0.0 | -0.007 |

Reading path:

- Primary Guermantes-world humiliation zone: `/projects/islt/fr-original/v3-p1`
- Early negative setup: `/projects/islt/fr-original/v1-p1-combray`
- Continued social diminishment: `/projects/islt/fr-original/v3-p2`

Notable units:

- The narrator's sustained comic portrait — schoolboy provisions at the trial, nervous erethism, the imperious supper giving only the illusion of power — clearly lowers Bloch within this passage.: `/projects/islt/fr-original/v3-p1#p-616`
- Bloch is portrayed as insincere and manipulative, badmouthing both the narrator and Saint-Loup to each other while performing exaggerated tenderness toward both, and the narrator explicitly disbelieves his protestations.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-186`
- Bloch is clearly diminished: his silence is unmasked as envy and his eventual comment as a self-serving insult delivered under the pretext of friendly tact.: `/projects/islt/fr-original/v6-p2#p-46`

## Norpois

- Slug: `norpois`
- Portrait default: `/projects/islt/portraits/norpois-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `63`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `authority_positive`

Seventh of 35 in scene-level advantage — a genuinely high standing — even though his individual scenes run mildly negative more often than not; prestige and belonging lean positive too, but too rarely staged to rank.

Norpois holds one of the highest standings in scene-level advantage of any figure examined here — 7th of 35 — a mark of durable authority across the book. That standing does not mean his scenes are one-sided wins: taken individually they skew mildly negative, 18 losses against 14 gains across 63 appearances, so the high standing reflects sustained position more than a run of triumphs. In prestige and belonging the evidence runs too thin to rank him, though both lean toward positive or flat rather than negative. He reads as a man whose authority is bigger than any single room he stands in.

Why interesting:

- His standing in advantage (7th of 35) is one of the highest in the pilot set, yet his individual scenes lean mildly negative (18 losses to 14 gains) — authority that outlasts any given encounter.
- Prestige and belonging both lean positive-to-flat in direction, but the novel simply doesn't stage him often enough in either to rank him there.
- He is a clean example of standing decoupled from scene-by-scene texture — durable position built on more than winning individual exchanges.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1573 ± 165 | 1408.3 | 7 of 35 | 63 | -0.157 | 0.442 | 14/18/0/31 |
| prestige | 1562 ± 286 | 1276.2 | insufficient evidence | 63 | +0.048 | 0.073 | 4/1/0/58 |
| inclusion | 1569 ± 321 | 1248.3 | insufficient evidence | 63 | 0.0 | 0.0 | 0/0/0/63 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p1-autour-de-mme-swann | 27 | -0.062 | +0.083 | 0.0 |
| v3-p1 | 24 | -0.049 | +0.065 | 0.0 |
| v6-p3 | 7 | -0.6 | -0.114 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 1 | -1.74 | 0.0 | 0.0 |
| v5 | 1 | -0.72 | 0.0 | 0.0 |

Reading path:

- Main authority concentration: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Secondary Guermantes reinforcement: `/projects/islt/fr-original/v3-p1`
- Late echo of rhetorical force: `/projects/islt/fr-original/v6-p3`

Notable units:

- Every voice in the room, and the narrator behind them, treats Norpois as tedious, stale, and malicious; he ends the passage labelled « très mauvaise langue ».: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-216`
- Norpois is locally diminished as the narrator exposes his loss of former reserve, naive political self-importance, and senile garrulousness.: `/projects/islt/fr-original/v6-p3#p-11`
- The narrator's explicit, endorsed judgment that Norpois's praise rests on 'no true taste' clearly diminishes him locally, exposing his pronouncement as empty flattery rather than genuine artistic discernment.: `/projects/islt/fr-original/v3-p1#p-771`

## docteur Cottard

- Slug: `docteur-cottard`
- Portrait default: `/projects/islt/portraits/docteur-cottard-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `43`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `advantage_negative_lower_tier`

A lower-third standing in scene-level advantage (25th of 35), driven by real negative intensity rather than mere frequency; prestige and belonging are both too thinly staged to rank, and lean mildly negative-to-flat.

Cottard's clearest measured trait is scene-level advantage: 25th of 35, a genuinely negative record — 19 losing outcomes against 7 positive across 43 appearances, with real intensity behind the pull rather than a flat accumulation. Prestige and belonging are both too rarely staged to size a standing: prestige leans mildly negative, belonging registers almost no movement at all. He reads as a moderate, real loser in the room rather than a broadly damaged figure across every register.

Why interesting:

- His advantage losses substantially outnumber his gains (19 to 7), giving him real negative intensity rather than a flat accumulation, though his standing (25th of 35) stops short of the book's worst cases.
- Prestige and belonging stay unranked — the novel simply doesn't stage him often enough in either for a standing to form.
- He is a useful moderate case: genuinely negative where measured, silent where not, without the volatility of figures like Charlus or Bloch.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1460 ± 168 | 1291.7 | 25 of 35 | 43 | -0.22 | 0.6104 | 7/19/0/17 |
| prestige | 1413 ± 292 | 1121.5 | insufficient evidence | 43 | -0.035 | 0.0349 | 0/2/0/41 |
| inclusion | 1372 ± 277 | 1094.6 | insufficient evidence | 43 | 0.0 | 0.0 | 0/0/0/43 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 23 | -0.383 | 0.0 | 0.0 |
| v2-p1-autour-de-mme-swann | 6 | +0.236 | 0.0 | 0.0 |
| v4-p2 | 9 | -0.35 | -0.167 | 0.0 |
| v3-p2 | 1 | +1.86 | 0.0 | 0.0 |
| v3-p1 | 2 | -0.4 | 0.0 | 0.0 |

Reading path:

- Primary negative concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Continued negative pressure: `/projects/islt/fr-original/v4-p2`
- Positive counterweight in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`

Notable units:

- Cottard's local standing rises sharply as the narrator credits him with a rare, almost heroic grandeur in his critical medical judgment, despite being otherwise deemed insignificant and common.: `/projects/islt/fr-original/v3-p2#p-21`
- Cottard is clearly elevated: his quip produces collective laughter and explicit verbal praise from the hostess.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-216`
- Cottard is locally diminished by the narrator's sustained irony about his snobbish vanity, misjudged social hierarchies, and callous prioritizing of the Verdurin salon over professional and family duty.: `/projects/islt/fr-original/v4-p2#p-311`

## la mère du narrateur

- Slug: `la-mere-du-narrateur`
- Portrait default: `/projects/islt/portraits/la-mere-du-narrateur-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `40`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `familial_positive`

Sixth of 35 in scene-level advantage — one of the highest standings among family and household figures — built on real positive scenes; prestige and belonging stay too thin to size, belonging leaning mildly negative.

la mère du narrateur holds one of the strongest advantage standings measured: 6th of the 35 characters substantial enough to rank, with genuinely positive scenes behind it — 10 gains against 6 losses across 40 appearances. Prestige and belonging are both too rarely staged to size a standing: prestige registers almost no movement at all, and belonging leans mildly negative in direction, though neither rests on enough evidence for a real claim. Where the novel measures her, she is quietly but genuinely strong; elsewhere it simply does not measure her enough to say.

Why interesting:

- She ranks 6th of 35 in scene-level advantage — among the highest standings in the pilot set, and built on genuinely more gains than losses (10 to 6).
- Belonging leans mildly negative in direction, a small but real complication in what is otherwise a clean positive profile, though far too little is staged there to rank the claim.
- Prestige is essentially untouched: the novel gives her almost no scenes in that register at all.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1580 ± 167 | 1412.6 | 6 of 35 | 40 | +0.087 | 0.3057 | 10/6/0/24 |
| prestige | 1529 ± 304 | 1225.2 | insufficient evidence | 40 | 0.0 | 0.0 | 0/0/0/40 |
| inclusion | 1784 ± 387 | 1397.1 | insufficient evidence | 40 | -0.018 | 0.0175 | 0/1/0/39 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 7 | +0.085 | 0.0 | 0.0 |
| v1-p1-combray | 7 | +0.417 | 0.0 | 0.0 |
| v4-p2 | 2 | -0.865 | 0.0 | 0.0 |
| v2-p1-autour-de-mme-swann | 9 | +0.099 | 0.0 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 2 | +0.35 | 0.0 | 0.0 |

Reading path:

- Foundational domestic context, and her strongest positive concentration: `/projects/islt/fr-original/v1-p1-combray`
- Largest positive presence, in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Continued positive presence in Guermantes-adjacent scenes: `/projects/islt/fr-original/v3-p2`

Notable units:

- The mother is idealized by the narrator as an irreplaceable, incomparable source of complete love, set explicitly above any hypothetical substitute or later mistress.: `/projects/islt/fr-original/v1-p1-combray#p-361`
- She loses all standing as an agent in the scene, convulsed and thoughtless with grief at the foot of the bed.: `/projects/islt/fr-original/v3-p2#p-86`
- The mother is shown consumed by an all-effacing grief that submerges her own distinct traits ('son bon sens, sa gaîté moqueuse') and remakes her in her dead mother's image, a severe local diminishment of her own individual selfhood.: `/projects/islt/fr-original/v4-p2#p-196`

## Bergotte

- Slug: `bergotte`
- Portrait default: `/projects/islt/portraits/bergotte-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `36`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `advantage_positive_high_standing`

Third of 35 in scene-level advantage — one of the very highest standings in the pilot set — with real positive scenes behind it; prestige leans positive too, but the novel does not stage him there often enough to rank it.

Bergotte holds one of the highest standings measured in scene-level advantage: 3rd of 35, behind only Françoise and la grand-mère, with genuinely positive scenes underwriting it — 12 gains against 8 losses across 36 appearances, and real intensity in the swings. Prestige leans the same direction, but the novel simply doesn't stage him in enough high-status scenes to rank him there; belonging is almost entirely untouched. His authority, where it is measured at all, is one of the clearest and highest-standing positives in the book.

Why interesting:

- He ranks 3rd of 35 in scene-level advantage — one of the highest standings of any figure examined here, ahead of every aristocrat in the pilot set.
- His positive standing is matched by genuinely positive scenes (12 gains to 8 losses), not just a favorable reading on thin evidence.
- Prestige and belonging both go unranked — the novel simply doesn't stage him in those registers often enough, even though the little evidence there leans positive or flat.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1631 ± 175 | 1456.0 | 3 of 35 | 36 | +0.145 | 0.6619 | 12/8/0/16 |
| prestige | 1655 ± 403 | 1252.0 | insufficient evidence | 36 | +0.07 | 0.0697 | 2/0/0/34 |
| inclusion | 1556 ± 428 | 1128.0 | insufficient evidence | 36 | 0.0 | 0.0 | 0/0/0/36 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p1-autour-de-mme-swann | 13 | -0.112 | +0.058 | 0.0 |
| v3-p1 | 4 | +0.522 | 0.0 | 0.0 |
| v5 | 3 | +0.953 | 0.0 | 0.0 |
| v1-p1-combray | 5 | +0.368 | 0.0 | 0.0 |
| v3-p2 | 2 | -0.8 | +0.88 | 0.0 |

Reading path:

- Largest positive presence, in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Strong positive concentration in Guermantes-world scenes: `/projects/islt/fr-original/v3-p1`
- Early positive footing: `/projects/islt/fr-original/v1-p1-combray`

Notable units:

- Bergotte's contradiction is presented as strengthening his interlocutor, making the final judgment a joint work; this is the passage's model of real intellectual force.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-216`
- Bergotte is locally elevated to the status of an all-but-worshipped authority through the narrator's escalating admiration.: `/projects/islt/fr-original/v1-p1-combray#p-186`
- Bergotte's oddities of speech and origin, initially read as affectation or vulgarity, are revealed by the narrator as the living root of his literary genius.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-201`

## Legrandin

- Slug: `legrandin`
- Portrait default: `/projects/islt/portraits/legrandin-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `24`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `advantage_negative_too_thin_to_rank`

The novel doesn't stage him often enough to rank him in any register, but where it does, scene-level advantage is nearly one-directional — the steepest negative lean of any figure in the pilot set.

Legrandin is not substantial enough for the novel to rank in any register — advantage, prestige, or belonging all fall short of the evidence needed. But where he is staged, the direction is stark: in scene-level advantage his 24 appearances split 15 negative to just 1 positive, the steepest negative lean of any figure in this set, with real intensity behind it. Prestige and belonging carry almost no signal at all, essentially flat. He is a case of a real, sharp negative pattern that the novel simply doesn't stage often enough to certify with a standing.

Why interesting:

- His advantage scenes are the most lopsided in the pilot set — 15 negative to 1 positive — yet even this is not enough appearances for the novel to rank him.
- He is a clean illustration of the gap between a strong signal and a certified standing: intensity without enough staged evidence to rank.
- Prestige and belonging are essentially silent for him — flat, thin, and unranked.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1348 ± 203 | 1144.4 | insufficient evidence | 24 | -0.627 | 0.7313 | 1/15/0/8 |
| prestige | 1586 ± 314 | 1272.1 | insufficient evidence | 24 | +0.002 | 0.1396 | 1/2/0/21 |
| inclusion | 1446 ± 497 | 949.3 | insufficient evidence | 24 | 0.0 | 0.0 | 0/0/0/24 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p1-combray | 8 | -0.9 | -0.106 | 0.0 |
| v3-p1 | 7 | -0.664 | -0.114 | 0.0 |
| v7-p4-le-bal-de-tetes | 3 | -0.6 | 0.0 | 0.0 |
| v6-p4 | 2 | -0.325 | +0.85 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 1 | -0.75 | 0.0 | 0.0 |

Reading path:

- Primary negative concentration, in Guermantes-adjacent society: `/projects/islt/fr-original/v3-p1`
- Early negative concentration: `/projects/islt/fr-original/v1-p1-combray`
- Final, sharpest negative return in diminished society: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- The narrator's opening promise to 'definitively change opinion' of Legrandin, followed by sustained, explicit exposure of his hypocrisy culminating in the flat verdict 'il était snob,' marks a clear and pronounced local diminishment.: `/projects/islt/fr-original/v1-p1-combray#p-266`
- The narrator sharply lowers Legrandin by likening his evasive fabrications to a criminal's wasted, misapplied labor, undercutting his pose of refined sincerity.: `/projects/islt/fr-original/v1-p1-combray#p-281`
- Legrandin is rendered a hollow, ghostly version of his former vivacious self, a clear local decline in the narrator's estimation of him.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-11`

## Mme de Cambremer

- Slug: `mme-de-cambremer`
- Portrait default: `/projects/islt/portraits/mme-de-cambremer-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `20`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `compact_negative`

Near the very bottom of the ranked cast in scene-level advantage (33rd of 35) — one of the most severe standings measured — with prestige and belonging both leaning negative too, though too thinly staged in either to rank.

Mme de Cambremer appears rarely, only 20 times across the book, but wherever she is staged, the reading is severe: 33rd of the 35 characters substantial enough to rank in scene-level advantage, near the very bottom, with 9 negative outcomes against just 2 positive. Prestige and belonging are both too thin to rank, but neither offers relief: prestige leans substantially negative in direction, belonging mildly so. She is a small, sharply negative presence rather than a broadly damaged one, rare enough to stay legible, consistent enough to matter.

Why interesting:

- She ranks 33rd of 35 in scene-level advantage — one of the most severe standings measured in the entire pilot set, third from the bottom.
- Her rarity does not soften the reading: prestige, though too thin to rank, leans substantially negative in direction, reinforcing rather than complicating the advantage picture.
- She is useful as a small-scale confirmation that severe negative standing is not limited to the book's most frequent figures.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1401 ± 180 | 1220.7 | 33 of 35 | 20 | -0.452 | 0.5635 | 2/9/0/9 |
| prestige | 1064 ± 324 | 739.5 | insufficient evidence | 20 | -0.213 | 0.3835 | 1/4/0/15 |
| inclusion | 1162 ± 344 | 817.8 | insufficient evidence | 20 | -0.062 | 0.062 | 0/1/0/19 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 3 | -1.45 | -0.587 | 0.0 |
| v4-p2 | 8 | -0.273 | -0.314 | -0.155 |
| v1-p2-un-amour-de-swann | 3 | -0.8 | -0.567 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 2 | +0.35 | +0.85 | 0.0 |
| v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle | 1 | -0.8 | 0.0 | 0.0 |

Reading path:

- Primary negative concentration: `/projects/islt/fr-original/v4-p2`
- Sharpest negative intensity, in Guermantes-adjacent scenes: `/projects/islt/fr-original/v3-p1`
- Supporting negative evidence: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- She is portrayed as a boring, pretentious 'monster' whose company and speech are ridiculed at length.: `/projects/islt/fr-original/v3-p1#p-466`
- She is relentlessly and elaborately ridiculed as bovine, absent and unable to respond, reduced entirely to an object of mockery.: `/projects/islt/fr-original/v3-p1#p-606`
- She is marked as structurally outside the true aristocratic circle, present only by charitable sufferance and unable even to claim acquaintance with the two cousins she watches so closely.: `/projects/islt/fr-original/v3-p1#p-76`

## M. Vinteuil

- Slug: `m-vinteuil`
- Portrait default: `/projects/islt/portraits/m-vinteuil-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `15`
- Archetype signs: `advantage +1, prestige -1, inclusion +1`
- Pattern: `rehabilitated_positive`

A genuine reversal within a small set of scenes: strongly negative early, strongly positive late — the novel doesn't stage him often enough to rank the outcome, but the arc itself is among the most dramatic swings measured.

M. Vinteuil's appearances are few, only 15 across the book, but they trace one of the most dramatic arcs in the pilot set: strongly negative early, concentrated in Combray, and strongly positive later, in the La Prisonnière material, with his overall mean movement in scene-level advantage ending up mildly positive despite the rough start. The swings are among the largest measured here — his individual scenes move more, on average, than almost any other figure's — but there are simply too few of them for the novel to certify a standing. Prestige leans mildly negative and belonging is essentially untouched, both far too thin to size. He is a genuine reversal case, not a stable positive one, even if the evidence stays too sparse to rank.

Why interesting:

- His scenes swing more dramatically than almost any other figure examined here — strongly negative early, strongly positive late — even though the total appearances are too few to rank the outcome.
- The reversal is chapter-shaped, not incidental: the early material (Combray) is where the negative concentrates, and the later material (La Prisonnière) is where the recovery happens.
- He is a genuine case of an arc rather than a static reading, best understood by following the sequence rather than a single number.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1575 ± 210 | 1365.2 | insufficient evidence | 15 | +0.125 | 0.9655 | 5/5/1/4 |
| prestige | 1498 ± 483 | 1015.2 | insufficient evidence | 15 | -0.127 | 0.1267 | 0/1/0/14 |
| inclusion | 1667 ± 533 | 1134.0 | insufficient evidence | 15 | 0.0 | 0.0 | 0/0/0/15 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p1-combray | 6 | -0.489 | -0.317 | 0.0 |
| v5 | 4 | +0.91 | 0.0 | 0.0 |
| v1-p2-un-amour-de-swann | 3 | +0.39 | 0.0 | 0.0 |
| v3-p2 | 1 | 0.0 | 0.0 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 1 | 0.0 | 0.0 | 0.0 |

Reading path:

- Main late positive recovery: `/projects/islt/fr-original/v5`
- Early negative counterweight: `/projects/islt/fr-original/v1-p1-combray`
- Intermediate positive reinforcement: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- Vinteuil is portrayed as an artist of supreme, unique genius whose septet is a triumphant masterpiece that eclipses even his celebrated sonata.: `/projects/islt/fr-original/v5#p-306`
- He falls to the bottom of Combray's esteem — an object of gossip and of Percepied's public joke — and behaves accordingly, deferring to people who were formerly beneath him.: `/projects/islt/fr-original/v1-p1-combray#p-306`
- Vinteuil is locally elevated to the status of a sublime, almost superhuman artistic genius through Swann's admiring meditation, endorsed by the narrator.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-531`
