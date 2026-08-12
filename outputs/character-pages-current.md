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

The most-scored figure in the novel, overwhelmingly shaped by repeated immediate, social, and emotional losses.

Swann dominates the novel by sheer presence, and his overall pattern remains broadly and repeatedly negative across all three lenses.

Why interesting:

- He is both the most-scored character and one of the clearest broad negative cases.
- His profile is not a narrow anomaly but a book-wide social pattern, especially in Un amour de Swann.

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

An ever-present aristocratic figure whose prestige often holds even where belonging and immediate advantage give way.

Robert de Saint-Loup appears throughout the book and shows one of the largest lens spreads in the novel, especially where aristocratic polish and emotional belonging pull apart.

Why interesting:

- He is central enough to matter structurally, not just as a curiosity of one chapter.
- His strongest divergence is chapter-shaped rather than flat across the book, especially in the Guermantes material.

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
- Pattern: `broad_negative`

A constant presence across the book whose strongest shaping comes from imprisonment, suspicion, disappearance, and loss.

Albertine is one of the largest and most persistently negative figures in the novel, with her strongest shaping concentrated in the prison and disappearance chapters.

Why interesting:

- Her presence is both extensive and highly concentrated in a few major late narrative blocks.
- She helps distinguish broad negative centrality from the more split prestige/inclusion cases.

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

Prestige-positive but inclusion-negative, with her sharpest gains and reversals concentrated in a few high-pressure chapters.

Odette is one of the clearest cross-lens split figures in the novel: she rises strongly in prestige while remaining far more unstable in belonging and immediate advantage.

Why interesting:

- Her prestige and inclusion readings diverge much more sharply than how often she appears would predict.
- Her profile is driven by a few concentrated chapter zones rather than a flat, book-wide pattern.

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

- Prestige ascent around Mme Swann: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
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
- Pattern: `volatile_broad_negative`

A highly volatile major figure whose negative scores are spread across salon, sexual, and wartime terrains.

baron de Charlus is a constant presence and a highly volatile figure, his negative scores spread across salon, sexual, and wartime configurations rather than one single narrative block.

Why interesting:

- He is too frequent and too spread out to read as a one-zone anomaly.
- His profile shows how a major character can be broadly negative without collapsing into a single repeated scene type.

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
- Pattern: `prestige_expectation_reversed`

A revealing reversal-of-expectation figure: high rank without correspondingly positive scores.

duc de Guermantes is one of the novel's most revealing reversals of expectation: despite formal rank, his scores are broadly negative across all three lenses.

Why interesting:

- He demonstrates that aristocratic title alone does not guarantee a positive reading in the novel.
- His profile sharpens the distinction between formal status and actual advantage or belonging.

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
- Pattern: `broad_negative_with_reversals`

A frequent recurring figure whose overall downward pull includes a few brief local reversals.

Françoise accumulates as a broadly negative figure across the book, though her profile is not flat: a small number of chapters briefly reverse the trend before the longer downward pull returns.

Why interesting:

- She is frequent enough to matter, but not in the same aristocratic pattern as Swann or Charlus.
- Her profile is useful for distinguishing domestic/local authority from broader belonging and valuation.

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
- Late negative reinforcement: `/projects/islt/fr-original/v5`
- Brief local reversal: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`

Notable units:

- Françoise is portrayed as devoted, hard-working, and morally admirable, explicitly contrasted with servants whose surface charm masks an 'inéducable nullité.': `/projects/islt/fr-original/v1-p1-combray#p-56`
- Françoise is wounded by the narrator's calculated cruelty, reacting with a breathless, barely intelligible response.: `/projects/islt/fr-original/v4-p2#p-166`
- Françoise's reputed saintliness and tenderness are locally undercut by the narrator's revelation of her calculated, wasp-like cruelty toward dependents she can dominate.: `/projects/islt/fr-original/v1-p1-combray#p-261`

## Mme Verdurin

- Slug: `mme-verdurin`
- Portrait default: `/projects/islt/portraits/mme-verdurin-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `82`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `broad_negative`

A salon figure whose overall reading is broadly negative across all three lenses.

Mme Verdurin is one of the clearest broadly negative salon figures in the novel, with losses in advantage, prestige, and inclusion all reinforcing rather than offsetting one another.

Why interesting:

- She is central enough to shape multiple social zones without becoming a prestige split case.
- Her pattern helps define the novel's recurrent salon-world negativity.

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
- Pattern: `inclusion_negative`

A recurring family figure whose harshest pressure falls on belonging and broader valuation.

la grand-mère accumulates as one of the book's more strongly negative recurring figures, with the harshest pressure falling on inclusion and broad valuation rather than on a narrow prestige story alone.

Why interesting:

- She is a strongly negative recurring figure outside the main salon/aristocratic pattern.
- Her profile is useful for showing how intimate and familial figures can be damaged without being prestige-centered.

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

- Early family-world pressure: `/projects/islt/fr-original/v1-p1-combray`
- Guermantes-world counterweight: `/projects/islt/fr-original/v3-p1`
- Main split concentration: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`

Notable units:

- The narrator's « Sans toi je ne pourrais pas vivre » gives her total emotional leverage, which she then uses to counsel him toward a harder heart.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-161`
- She is shown, through the eyes of passersby like Legrandin, as visibly disheveled and overwhelmed, a stark diminishment of her usual composed public presence.: `/projects/islt/fr-original/v3-p2#p-6`
- Both the dream vision and the photograph mark her with the visible signs of fatal illness, an 'air de condamnée à mort' that maman experiences as an insult done to her mother's face.: `/projects/islt/fr-original/v4-p2#p-206`

## Mme de Villeparisis

- Slug: `mme-de-villeparisis`
- Portrait default: `/projects/islt/portraits/mme-de-villeparisis-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `79`
- Archetype signs: `advantage -1, prestige -1, inclusion +1`
- Pattern: `prestige_positive_inclusion_negative`

A moderate but revealing split figure, relatively strong in prestige while advantage and inclusion drift downward.

Mme de Villeparisis is one of the clearest moderate split figures in the novel: she remains comparatively strong in prestige while advantage and inclusion drift downward or oscillate by chapter.

Why interesting:

- She shows the lens split in a quieter register than Odette or Saint-Loup.
- Her chapter distribution helps distinguish sustained social authority from weaker interpersonal footing.

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
- Prestige support zone: `/projects/islt/fr-original/v3-p2`
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
- Pattern: `advantage_positive_prestige_positive_inclusion_negative`

A compact but telling figure whose strength in prestige and immediate advantage does not translate into equal belonging.

Gilberte is a compact but revealing cross-lens figure: she scores very well in prestige and immediate advantage, yet her inclusion profile remains markedly less secure.

Why interesting:

- She is a smaller but especially clear example of lens divergence.
- Her divergence is visible in just a few chapters, without needing a large presence in the book to make the case.

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
- Pattern: `broad_negative`

A strongly negative recurring figure whose losses reinforce each other across all three lenses.

Bloch is one of the clearest cases of consistent negative treatment in the novel, with repeated losses in advantage, prestige, and inclusion reinforcing each other rather than splitting apart.

Why interesting:

- He is one of the cleanest examples of consistent multi-lens damage.
- His profile helps clarify the difference between broad negative treatment and more prestige-divergent cases.

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

A strongly positive figure whose main force comes from durable rhetorical and social authority rather than intimacy.

Norpois is a strongly positive figure across all three lenses, driven less by intimacy than by durable rhetorical authority and socially legible judgment.

Why interesting:

- He helps separate prestige-positive authority from the more emotionally charged positive figures.
- His positivity is anchored in repeated interpretive command rather than one dramatic narrative reversal.

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
- Pattern: `swann_world_negative`

A mid-tier negative figure shaped by a strong Swann-world concentration and then smaller, uneven later reversals.

docteur Cottard is a mid-tier negative figure whose profile is shaped by one strong Swann-world concentration, then complicated by smaller later recoveries and uneven prestige moments.

Why interesting:

- He is a useful moderate case rather than an extreme outlier in the novel.
- His chapter pattern helps show how one major concentration can dominate an otherwise mixed profile.

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
- Brief positive counterweight: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom`
- Later positive echo: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`

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

A quiet but high-performing recurring figure, especially strong in advantage and inclusion.

la mère du narrateur is a quietly high-performing figure across all three lenses, with especially strong advantage and inclusion values driven by stable interpretive and familial force.

Why interesting:

- She offers a positive familial counterpoint to the more socially competitive figures.
- Her profile shows how recurring emotional authority can register positively without requiring prestige spectacle.

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

- Main positive family-world concentration: `/projects/islt/fr-original/v6-p3`
- Earlier positive presence: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom`
- Foundational domestic context: `/projects/islt/fr-original/v1-p1-combray`

Notable units:

- The mother is idealized by the narrator as an irreplaceable, incomparable source of complete love, set explicitly above any hypothetical substitute or later mistress.: `/projects/islt/fr-original/v1-p1-combray#p-361`
- She loses all standing as an agent in the scene, convulsed and thoughtless with grief at the foot of the bed.: `/projects/islt/fr-original/v3-p2#p-86`
- The mother is shown consumed by an all-effacing grief that submerges her own distinct traits ('son bon sens, sa gaîté moqueuse') and remakes her in her dead mother's image, a severe local diminishment of her own individual selfhood.: `/projects/islt/fr-original/v4-p2#p-196`

## Bergotte

- Slug: `bergotte`
- Portrait default: `/projects/islt/portraits/bergotte-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `36`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `rehabilitated_positive`

A symbolic positive figure whose literary authority carries strongly across advantage and prestige.

Bergotte is one of the novel's clearest positive symbolic figures, with his literary authority translating into very high advantage and prestige across several distinct narrative zones.

Why interesting:

- He is strongly positive without being primarily a belonging-driven figure.
- His profile is sharply chapter-shaped, which makes him useful for showing how strongly positive scores can arise from uneven terrain.

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

- Main late positive recovery: `/projects/islt/fr-original/v5`
- Early negative counterweight: `/projects/islt/fr-original/v1-p1-combray`
- Intermediate positive reinforcement: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- Bergotte's contradiction is presented as strengthening his interlocutor, making the final judgment a joint work; this is the passage's model of real intellectual force.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-216`
- Bergotte is locally elevated to the status of an all-but-worshipped authority through the narrator's escalating admiration.: `/projects/islt/fr-original/v1-p1-combray#p-186`
- Bergotte's oddities of speech and origin, initially read as affectation or vulgarity, are revealed by the narrator as the living root of his literary genius.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-201`

## Legrandin

- Slug: `legrandin`
- Portrait default: `/projects/islt/portraits/legrandin-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `24`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `awkward_negative`

A broadly negative recurring figure marked by repeated self-positioning failures and social discredit.

Legrandin is a broadly negative figure whose profile is shaped by repeated discredit and awkward self-positioning, even though a few isolated passages briefly interrupt the downward pattern.

Why interesting:

- He is a clear recurring loser without depending on a single chapter block.
- His profile sharpens the category of embarrassment-driven negative treatment.

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

- Primary negative concentration: `/projects/islt/fr-original/v1-p1-combray`
- Late positive interruption: `/projects/islt/fr-original/v6-p4`
- Final return in diminished society: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

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

A compact but stable negative case: rare in the novel, but strongly downward wherever she appears.

Mme de Cambremer is a compact but stable negative case: she doesn't appear as often as the novel's biggest characters, but what is there reads overwhelmingly downward in advantage, prestige, and inclusion.

Why interesting:

- She is useful as a smaller-scale confirmation that consistent multi-lens negativity is not limited to the biggest characters.
- Her appearances are sparse enough to stay legible but numerous enough to matter.

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

- Primary negative concentration: `/projects/islt/fr-original/v3-p1`
- Supporting negative evidence: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Prestige-world reinforcement: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`

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

A surprising overall positive whose late recoveries outweigh strongly negative early material.

M. Vinteuil is one of the more surprising positive figures in the novel: despite some strongly negative early material, his scores end up decisively positive, especially in inclusion.

Why interesting:

- He is a genuine reversal case rather than a merely stable positive.
- His profile is sharply chapter-shaped, which makes him useful for showing how an overall positive reading can arise from uneven terrain.

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
