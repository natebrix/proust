# Character Pages (scoring v2)

- Analysis version: `character_pages_v2`
- Scoring version: `scoring_v2`
- Source corpus summary: `scoring_v2_corpus_summary_v1`
- View: `name`
- Character count: `23`
- Corpus: `enrichment`

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
- Annotation units: `209`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `relational_positive_understated`

He loses more scenes than he wins, yet he ranks 1st of 8 in belonging, 4th of 14 in standing and 7th of 31 in the scenes.

Nearly every passage in the novel passes through the narrator, and the scenes often go badly for him: 168 wins against 200 losses, with 81 passages leaving him worse off and 46 better. Across the book his welcome holds all the same. He ranks 1st of 8 in belonging and 4th of 14 in standing, and his place in the scenes, 7th of 31, owes as much to how often he is seen as to what he wins, since no one's rating is more certain. His fortune rises through Le Côté de Guermantes, peaks in the Guermantes salons, and falls in La Prisonnière. The book's last turn, the discovery of his vocation in L'Adoration perpétuelle, depends on no one's regard but his own, so it leaves no trace in his ratings.

Why interesting:

- The rooms keep receiving him while the scenes keep wounding him. That split is the book's central irony about its narrator.
- Because the whole novel passes through him, his rating in the scenes has the narrowest uncertainty of anyone's.
- His one lasting victory, the vocation he finds in L'Adoration perpétuelle, happens alone, where no other character can grant or refuse it.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1513 ± 82 | 1430.8 | 7 of 31 | 209 | -0.201 | 0.6321 | 46/81/7/75 |
| prestige | 1648 ± 128 | 1520.9 | 4 of 14 | 209 | +0.061 | 0.1003 | 18/4/0/187 |
| inclusion | 1598 ± 101 | 1496.7 | 1 of 8 | 209 | +0.077 | 0.3553 | 36/26/1/146 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v5 | 38 | -0.286 | +0.019 | 0.0 |
| v2-p1-autour-de-mme-swann | 26 | -0.439 | +0.055 | +0.417 |
| v3-p2 | 35 | +0.013 | +0.245 | +0.071 |
| v3-p1 | 36 | -0.176 | +0.054 | +0.191 |
| v2-p2-noms-de-pays-le-pays | 28 | -0.148 | +0.035 | -0.21 |

Reading path:

- Balbec: new people, new rooms: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Received by the Guermantes: `/projects/islt/fr-original/v3-p2`
- The Bal de têtes: the survivor among the masks: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- He is reduced to recalling her, begging, and being refused, and the withheld kiss governs everything that follows.: `/projects/islt/fr-original/v5#p-381`
- The revelation stops his breath and reopens his jealousy; he is the dupe of Albertine and Andrée, and the passage insists that what matters in her life is sheltered exactly where he does not think to look.: `/projects/islt/fr-original/v5#p-376`
- He is the one who needs, suffers and watches; his surveillance is both humiliating to him and, as it turns out, useless.: `/projects/islt/fr-original/v5#p-221`

## duchesse de Guermantes

- Slug: `duchesse-de-guermantes`
- Portrait default: `/projects/islt/portraits/duchesse-de-guermantes-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `183`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `uniform_positive`

First in the scenes, first in standing, and second only to the narrator in belonging.

The duchesse ranks 1st of 31 in the scenes, 1st of 14 in standing and 2nd of 8 in belonging, behind only the narrator. No one else places in the top three of every measure. Her scenes bear it out, with 225 wins against 92 losses, the wit carrying the evening far more often than it fails her. Her fortune rises into Le Côté de Guermantes, where the narrator enters her world, and declines after it. By the Bal de têtes her standing has clearly fallen, as the aging duchesse takes up with actresses and Rachel outshines la Berma.

Why interesting:

- She tops two of the three measures and is second in the third, the most complete high placement in the book.
- Her scene record, 225 wins and 92 losses, is the most one-sided winning record of any often-seen character.
- Her standing is also one of the few clear declines at the end of the book. The queen of the Faubourg fades at the Bal de têtes.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1602 ± 93 | 1508.5 | 1 of 31 | 183 | +0.049 | 0.4899 | 62/41/8/72 |
| prestige | 1706 ± 108 | 1598.2 | 1 of 14 | 183 | +0.216 | 0.2704 | 38/4/0/141 |
| inclusion | 1613 ± 162 | 1451.1 | 2 of 8 | 183 | 0.0 | 0.0 | 0/0/0/183 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 82 | +0.08 | +0.24 | 0.0 |
| v3-p1 | 45 | +0.126 | +0.216 | 0.0 |
| v7-p4-le-bal-de-tetes | 9 | -0.561 | -0.078 | 0.0 |
| v1-p2-un-amour-de-swann | 15 | -0.051 | +0.137 | 0.0 |
| v4-p2 | 14 | -0.085 | +0.34 | 0.0 |

Reading path:

- The Guermantes seen from outside: `/projects/islt/fr-original/v3-p1`
- Dinner at the Guermantes: `/projects/islt/fr-original/v3-p2`
- The princesse's soirée: `/projects/islt/fr-original/v4-p2`

Notable units:

- The narrator's direct, superlative condemnation of her wit as knowingly false and cruel clearly diminishes her locally.: `/projects/islt/fr-original/v3-p2#p-476`
- The narrator sustains an emphatic diagnosis of her judgments as arbitrary and untruthful, a sharp local diminishment of her celebrated discernment.: `/projects/islt/fr-original/v3-p2#p-316`
- The princesse's unqualified declaration that nothing could lower Oriane in her esteem clearly elevates her standing in the scene.: `/projects/islt/fr-original/v3-p2#p-361`

## Swann

- Slug: `swann`
- Portrait default: `/projects/islt/portraits/swann-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `177`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `broad_negative`

Swann is one of the most present men in the novel and one of its steadiest losers, 19th of 31 in the scenes, 10th of 14 in standing and 7th of 8 in belonging.

Swann appears in 177 passages, more than anyone but the narrator and the duchesse, and the scenes go against him: 144 wins, 197 losses, and 83 passages that leave him worse off against 46 that leave him better. He ranks 19th of 31 in the scenes and 10th of 14 in standing, a modest place for the man Combray never knew dined with princes, because the novel shows his standing mostly after his marriage to Odette has cost him. In belonging he is 7th of 8. His fortune falls from Combray to Albertine disparue II, where the Guermantes will not speak his name, the second largest fall among the novel's main figures after Charlus's.

Why interesting:

- He appears in so many passages that his decline is no accident of a few scenes.
- His standing is shown mostly on its way down, after the marriage, so the Swann the novel lets us watch in society is already the diminished one.
- He is 7th of 8 in belonging. The man who once dined with princes ends as a name the Guermantes avoid.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1475 ± 102 | 1372.3 | 19 of 31 | 177 | -0.317 | 0.7741 | 46/83/3/45 |
| prestige | 1530 ± 134 | 1395.3 | 10 of 14 | 177 | +0.024 | 0.1659 | 15/15/2/145 |
| inclusion | 1359 ± 125 | 1234.0 | 7 of 8 | 177 | -0.12 | 0.1975 | 8/20/0/149 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 100 | -0.393 | +0.05 | -0.124 |
| v2-p1-autour-de-mme-swann | 22 | -0.108 | +0.003 | 0.0 |
| v3-p2 | 15 | +0.12 | -0.047 | +0.047 |
| v4-p2 | 11 | -0.797 | -0.054 | -0.214 |
| v6-p2 | 7 | -0.91 | -0.093 | -0.914 |

Reading path:

- Un amour de Swann: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Combray: the neighbor who dines with princes: `/projects/islt/fr-original/v1-p1-combray`
- The last evenings, ill and unwelcome: `/projects/islt/fr-original/v4-p2`

Notable units:

- The narrator shows his elevated disgust to be a factitious pose invented minutes earlier, so the tirade lowers the speaker rather than its objects.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-361`
- Swann is decisively barred from the Verdurin circle: his failed scheme to get invited fails outright, and afterward he is not even mentioned in their conversation.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-366`
- He is placed outside the Bayreuth party he was asked to pay for — the letter does not mention him, and the guests' presence is understood to bar his own.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-391`

## Robert de Saint-Loup

- Slug: `robert-de-saint-loup`
- Portrait default: `/projects/islt/portraits/saint-loup-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `138`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `broad_presence_middling`

Saint-Loup is in more scenes than almost anyone, and ranks in the middle of them, 16th of 31, and 9th of 14 in standing.

Saint-Loup is one of the most present figures in the novel, and his record is close to even: 16th of 31 in the scenes, with 105 wins and 108 losses. In standing, the measure his name should command, he is 9th of 14, behind his uncle Charlus, his great-aunt Mme de Villeparisis, Odette, and Gilberte, the woman he marries. His belonging is staged too rarely to rank. His fortune dips in Le Côté de Guermantes I, where Rachel and the barracks at Doncières fill his scenes, then rises to the end of the book, where the soldier killed in the war is remembered well.

Why interesting:

- His standing sits below his family's: 9th of 14, behind his uncle and his great-aunt, and behind Odette.
- His scene record is almost exactly even, 105 wins to 108 losses. Presence, more than triumph, holds his place.
- His fortune climbs through the last volumes, from the low of Doncières and Rachel to the war that kills him.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1486 ± 100 | 1386.5 | 16 of 31 | 138 | -0.132 | 0.6397 | 37/57/3/41 |
| prestige | 1553 ± 144 | 1409.3 | 9 of 14 | 138 | +0.047 | 0.1162 | 11/7/0/120 |
| inclusion | 1490 ± 202 | 1287.1 | insufficient evidence | 138 | -0.024 | 0.0235 | 0/3/0/135 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 81 | -0.258 | +0.043 | -0.031 |
| v2-p2-noms-de-pays-le-pays | 21 | +0.017 | +0.086 | -0.036 |
| v3-p2 | 13 | +0.108 | +0.159 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 5 | +1.408 | +0.16 | 0.0 |
| v7-p1-a-tansonville | 3 | -1.0 | -0.033 | 0.0 |

Reading path:

- Doncières, Rachel, and the family salons: `/projects/islt/fr-original/v3-p1`
- The friendship begins at Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Tansonville: the unhappy marriage: `/projects/islt/fr-original/v7-p1-a-tansonville`

Notable units:

- Robert's own words show total emotional subjugation: self-blame, anguished devotion, and willingness to sacrifice his own peace to appease Rachel.: `/projects/islt/fr-original/v3-p1#p-791`
- His entrance is met with staged, mobilized deference from the entire staff and is explicitly ranked above even Foix's standing in the patron's eyes.: `/projects/islt/fr-original/v3-p2#p-236`
- The passage retracts every unfavourable impression left by Tansonville and restores him as brave, delicate, and artistically intelligent.: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre#p-16`

## Albertine

- Slug: `albertine`
- Portrait default: `/projects/islt/portraits/albertine-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `126`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `volatile_scenes_standing_holds`

Albertine falls from the girl on the beach at Balbec to the captive of La Prisonnière, one of the clearest declines in the novel, while her scenes split almost evenly.

Albertine's scenes are among the most conflicted in the book, with 80 wins, 84 losses and more mixed passages than most characters. She ranks 12th of 31 in the scenes. Her standing and her belonging are both staged too seldom to rank, and much of her confinement is the narrator's doing, one man shutting her in more than society shutting her out. Her fortune falls clearly, from its high point at Balbec in Noms de pays : le pays to its low in La Prisonnière. The decline shows in her scenes as well, which keep turning against her into Albertine disparue.

Why interesting:

- Hers is one of the clearest declines in the novel. The girl the narrator first sees on the Balbec beach becomes the prisoner of his apartment.
- Her scenes split nearly evenly, 80 wins to 84 losses, with an unusual share of mixed passages, the sign of a character pulled both ways at once.
- Her exclusion is private. It is the narrator who shuts her in, and the world's verdict on her is staged too rarely to rank.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1504 ± 90 | 1413.3 | 12 of 31 | 126 | -0.203 | 0.7437 | 35/60/5/26 |
| prestige | 1511 ± 226 | 1284.3 | insufficient evidence | 126 | +0.01 | 0.0469 | 4/2/0/120 |
| inclusion | 1676 ± 254 | 1422.3 | insufficient evidence | 126 | -0.013 | 0.0618 | 3/3/0/120 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v5 | 54 | -0.329 | -0.014 | -0.058 |
| v6-p1 | 20 | -0.288 | 0.0 | 0.0 |
| v3-p2 | 14 | +0.24 | 0.0 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 17 | +0.242 | +0.133 | +0.086 |
| v4-p2 | 15 | -0.43 | -0.067 | 0.0 |

Reading path:

- La Prisonnière: `/projects/islt/fr-original/v5`
- After her flight and death: `/projects/islt/fr-original/v6-p1`
- Forgetting Albertine: `/projects/islt/fr-original/v6-p2`

Notable units:

- Each new admission further destroys Albertine's credibility, culminating in the narrator's blanket judgment that nothing she says can be trusted.: `/projects/islt/fr-original/v5#p-341`
- Albertine is admiringly portrayed by the narrator as unexpectedly devoted, gentle, and almost innocently generous in the moments following their intimacy, a narrator-endorsed elevation of her character in this scene.: `/projects/islt/fr-original/v3-p2#p-146`
- She holds the leverage: her keeper is exhausted, jealous and dependent, must invent daily pretexts to hold her, and she quietly secures the chauffeur's silence without his ever suspecting it.: `/projects/islt/fr-original/v5#p-221`

## Odette

- Slug: `odette`
- Portrait default: `/projects/islt/portraits/odette-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `124`
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Odette ends the novel with more standing than most of the people who once refused to receive her, while her scenes stay even and her welcome stays thin.

Odette ranks 3rd of 14 in standing, behind only the duchesse de Guermantes and Morel. In her scenes she sits at the middle, 15th of 31, with 112 wins to 107 losses, and in belonging she is 5th of 8. This is the pattern Proust gives her: the Faubourg learns to defer to Mme Swann, and later to Mme de Forcheville, long before it lets her in. Across the novel her standing edges up from Combray to Sodome et Gomorrhe II and holds to the end, while the passages as a whole drift slowly against her, from 1482 in Combray to 1346 at the Bal de têtes. Neither movement is large enough to call a clear arc.

Why interesting:

- Standing is her strongest measure, and it is the one the salons fought hardest. The demi-mondaine of Un amour de Swann ends above most of the Faubourg.
- Her three rankings pull apart: high standing, even scenes, modest belonging. She is received as a name before she is received as a person.
- Her scene record is nearly even, which suits a woman who advances by marriage and patience more than by winning rooms.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1520 ± 120 | 1400.3 | 15 of 31 | 124 | -0.081 | 0.5035 | 26/40/2/56 |
| prestige | 1710 ± 137 | 1573.2 | 3 of 14 | 124 | +0.107 | 0.1687 | 13/4/1/106 |
| inclusion | 1417 ± 157 | 1259.7 | 5 of 8 | 124 | -0.094 | 0.1066 | 1/8/0/115 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 63 | -0.018 | 0.0 | +0.013 |
| v2-p1-autour-de-mme-swann | 31 | -0.041 | +0.066 | -0.133 |
| v3-p1 | 7 | -0.193 | +0.34 | -0.594 |
| v1-p3-noms-de-pays-le-nom | 3 | +0.313 | +1.367 | 0.0 |
| v4-p2 | 3 | -0.25 | +0.867 | 0.0 |

Reading path:

- Mme Swann's salon: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Swann's love, and the Verdurin circle: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Glimpsed from the Guermantes world: `/projects/islt/fr-original/v3-p1`

Notable units:

- Odette's mere passage provokes public curiosity and a presumption of importance among strangers, a clear public marking of elevated standing.: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom#p-56`
- Swann's aunt refuses to receive Mme Swann and organizes other women to do likewise: a direct, witnessed exclusion.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-61`
- Odette is diminished by the narrator's detailed, unsympathetic exposure of her as a practiced but poorly-armed liar whose deceptions unravel under scrutiny and whose distress signals something further being concealed.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-331`

## baron de Charlus

- Slug: `baron-de-charlus`
- Portrait default: `/projects/islt/portraits/charlus-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `110`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `ranked_everywhere_late_fall`

The largest fall in the novel, from his first appearances at Balbec to the ruined old man of L'Adoration perpétuelle.

Charlus ranks in all three measures: 13th of 31 in the scenes, 7th of 14 in standing and 3rd of 8 in belonging. Those middling places average a long height and a steep fall. His fortune is the largest decline in the book, from 1627 at Balbec to 1149 in L'Adoration perpétuelle, and the fall is clear in his scenes, in his standing, and in his fortune as a whole. It runs through the Verdurins' expulsion of him in La Prisonnière and the wartime Paris of M. de Charlus pendant la guerre, and it ends with the old baron bowing to Mme de Saint-Euverte, a woman he once refused to acknowledge.

Why interesting:

- His is the largest fall in the novel, 479 points, and the only one that is clear in the scenes, in standing and overall at once.
- His middling ranks hide the shape of his story, a long summit and a collapse averaged into a place near the middle.
- He is 3rd of 8 in belonging. The novel installs the clubbable baron everywhere before it evicts him.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1502 ± 92 | 1410.0 | 13 of 31 | 110 | -0.256 | 0.7058 | 28/46/5/31 |
| prestige | 1550 ± 110 | 1440.0 | 7 of 14 | 110 | +0.032 | 0.269 | 16/11/1/82 |
| inclusion | 1562 ± 155 | 1406.6 | 3 of 8 | 110 | +0.011 | 0.0705 | 3/2/0/105 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v4-p2 | 34 | -0.185 | +0.192 | +0.05 |
| v5 | 15 | -0.979 | -0.057 | -0.22 |
| v7-p2-m-de-charlus-pendant-la-guerre | 8 | -1.086 | -0.419 | 0.0 |
| v3-p2 | 14 | +0.081 | 0.0 | 0.0 |
| v3-p1 | 11 | -0.089 | 0.0 | 0.0 |

Reading path:

- The Verdurin salon at la Raspelière: `/projects/islt/fr-original/v4-p2`
- The expulsion from the Verdurins: `/projects/islt/fr-original/v5`
- Wartime Paris and Jupien's hotel: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

Notable units:

- The narrator's extended commentary presents Charlus's collapse of aristocratic pride, laid bare by his illness, as proof of how perishable worldly grandeur and human pride are.: `/projects/islt/fr-original/v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1`
- Charlus is diminished as his once-carefully-hidden vice now surfaces uncontrollably in his manner and speech, aging and exposing him.: `/projects/islt/fr-original/v5#p-281`
- His grandiose self-delusion, obliviousness to Morel's obvious displeasure, and public spectacle of shouting 'Alleluia!' alone expose him as pathetically self-deceived.: `/projects/islt/fr-original/v4-p2#p-396`

## duc de Guermantes

- Slug: `duc-de-guermantes`
- Portrait default: `/projects/islt/portraits/duc-de-guermantes-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `97`
- Archetype signs: `advantage -1, prestige -1, inclusion +1`
- Pattern: `title_and_scenes_low`

The head of the Guermantes family ranks last in standing, 14th of 14, and 27th of 31 in the scenes.

The duc de Guermantes ranks 14th of 14 in standing and 27th of 31 in the scenes, where he loses 126 times and wins 75. Passages that cut him outnumber those that lift him 53 to 5: the Jockey Club election he loses, the dying cousin he will not mourn for fear of missing a costume ball, his wife's wit at his expense. His belonging is staged too rarely to rank. His fortune falls from his first appearance, as the prince des Laumes of Un amour de Swann, to La Prisonnière, and stays low to the end, where the old duc is Odette's lover.

Why interesting:

- He is last of 14 in standing and 27th of 31 in the scenes, which suits the book's running joke about him.
- Among the great aristocrats, his scenes are the most one-sided against him, with 5 passages lifting him and 53 cutting him.
- Against his wife the contrast is complete. She leads two measures; he reaches the top half of none.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1423 ± 105 | 1318.0 | 27 of 31 | 97 | -0.507 | 0.5645 | 5/53/4/35 |
| prestige | 1467 ± 168 | 1299.7 | 14 of 14 | 97 | -0.012 | 0.0614 | 2/2/0/93 |
| inclusion | 1565 ± 206 | 1358.6 | insufficient evidence | 97 | 0.0 | 0.0 | 0/0/0/97 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 58 | -0.47 | +0.012 | 0.0 |
| v3-p1 | 15 | -0.645 | +0.113 | 0.0 |
| v4-p2 | 13 | -0.592 | 0.0 | 0.0 |
| v5 | 1 | -1.8 | -1.88 | 0.0 |
| v7-p4-le-bal-de-tetes | 3 | -0.439 | -0.567 | 0.0 |

Reading path:

- Dinner at the Guermantes: `/projects/islt/fr-original/v3-p2`
- The matinée: `/projects/islt/fr-original/v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`
- The Bal de têtes: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- His scheduling of a dying relative's death around his own entertainments is a stark exposure of callous self-interest.: `/projects/islt/fr-original/v3-p2#p-626`
- A publicly registered defeat before his own world: denied the presidency that was his turn, and left «sur le carreau» in favour of a nobody.: `/projects/islt/fr-original/v5#p-71`
- The duc is clearly diminished in this passage: the narrator exposes his self-importance and obtuseness, and his later misreading of the grieving mother as merely disagreeable compounds the same portrait of a man unable to register others' suffering.: `/projects/islt/fr-original/v3-p2#p-61`

## Mme Verdurin

- Slug: `mme-verdurin`
- Portrait default: `/projects/islt/portraits/mme-verdurin-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `78`
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Mme Verdurin ranks 6th of 14 in standing and 10th of 31 in the scenes, and she is last of 8 in belonging.

Mme Verdurin ranks 6th of 14 in standing, 10th of 31 in the scenes, and 8th of 8 in belonging. Her standing rises across the book, from the bourgeois patronne of Un amour de Swann to the princesse de Guermantes of the last chapters. Her scenes are harsh, with 33 passages leaving her worse off against 4 that leave her better, and her belonging is the lowest of anyone ranked there. The woman who built the most exclusive little clan in Paris is rarely shown securely inside anything. Her fortune peaks in wartime Paris, in M. de Charlus pendant la guerre, and sinks again at the Bal de têtes, where the new princesse is a figure of fun.

Why interesting:

- Her three ranks tell three different stories: upper half in standing, upper third in the scenes, and last in belonging.
- She loses belonging at her own parties. At the soirée Charlus arranges in her salon, his aristocratic guests walk past her to thank him.
- Her title arrives as a joke. By the Bal de têtes the new princesse de Guermantes is mocked, and her fortune falls from its wartime peak.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1537 ± 122 | 1415.4 | 10 of 31 | 78 | -0.336 | 0.4254 | 4/33/0/41 |
| prestige | 1586 ± 127 | 1459.1 | 6 of 14 | 78 | +0.129 | 0.2362 | 13/4/0/61 |
| inclusion | 1357 ± 168 | 1189.0 | 8 of 8 | 78 | -0.055 | 0.0549 | 0/3/0/75 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 46 | -0.322 | +0.047 | 0.0 |
| v4-p2 | 13 | -0.608 | +0.195 | -0.055 |
| v5 | 5 | +0.176 | +0.136 | -0.712 |
| v7-p4-le-bal-de-tetes | 4 | -0.777 | +0.425 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 5 | -0.1 | +0.602 | 0.0 |

Reading path:

- The little clan: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- The soirée Charlus arranges, and his expulsion: `/projects/islt/fr-original/v5`
- Wartime: the salon at its height: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

Notable units:

- Her possessive, envy-driven manipulation of guests and casual denigration of an absent friend expose her as controlling rather than generous.: `/projects/islt/fr-original/v4-p2#p-341`
- Deference is withheld from her in her own house before the whole room: unrecognized, unpresented to, compared to a theatre usherette, and doubted to exist at all.: `/projects/islt/fr-original/v5#p-311`
- The guests bypass her entirely as hostess, addressing only Charlus and discussing her dismissively within earshot instead of greeting her as mistress of the house.: `/projects/islt/fr-original/v5#p-301`

## Mme de Villeparisis

- Slug: `mme-de-villeparisis`
- Portrait default: `/projects/islt/portraits/mme-de-villeparisis-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `73`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `advantage_strong_prestige_ranked`

The hostess of the Balbec hotel and the Paris matinée ranks 4th of 31 in the scenes and 8th of 14 in standing.

Mme de Villeparisis ranks 4th of 31 in the scenes, on 72 wins and 38 losses, and 8th of 14 in standing. Her matinée in Le Côté de Guermantes is some of the most closely staged social machinery in the novel, and she runs it, taking as much credit as the guests who shine there. Her belonging is harder to place. Everyone receives her and no one knows quite where to seat her, and the novel stages it too rarely to rank. Her fortune declines gently from Balbec to Albertine disparue III, where she is last seen in Venice with Norpois.

Why interesting:

- Among often-seen characters, only the duchesse wins a larger share of her scenes.
- Her standing is real but unsettled. She is grand enough to bring the princesse de Luxembourg to the Balbec hotel and still doubtful to the Guermantes.
- She fades quietly, last seen in Venice with Norpois, both of them old.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1595 ± 134 | 1461.4 | 4 of 31 | 73 | -0.077 | 0.3693 | 14/19/2/38 |
| prestige | 1561 ± 148 | 1412.8 | 8 of 14 | 73 | -0.016 | 0.2053 | 7/9/0/57 |
| inclusion | 1532 ± 215 | 1316.7 | insufficient evidence | 73 | 0.0 | 0.0 | 0/0/0/73 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 39 | +0.05 | +0.004 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 20 | -0.038 | -0.025 | 0.0 |
| v3-p2 | 7 | -0.559 | -0.114 | 0.0 |
| v6-p3 | 5 | -0.422 | 0.0 | 0.0 |
| v1-p1-combray | 1 | -0.8 | 0.0 | 0.0 |

Reading path:

- Her matinée: `/projects/islt/fr-original/v3-p1`
- Balbec, and the princesse de Luxembourg: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Venice with Norpois: `/projects/islt/fr-original/v6-p3`

Notable units:

- Her standing is shown fallen and seen to be fallen: duchesses no longer come except from duty of kinship, the snobs avoid her rooms, and Mme Leroi's freezing bow is the public form of it.: `/projects/islt/fr-original/v3-p1#p-416`
- The narrator's private reassessment of her as fundamentally unaristocratic, her name and title self-assumed, clearly lowers her standing in his eyes even though she remains outwardly unchanged toward him.: `/projects/islt/fr-original/v3-p1#p-841`
- Her standing rises sharply in the narrator's own private reappraisal once her close kinship to the Guermantes is revealed.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-201`

## Bloch

- Slug: `bloch`
- Portrait default: `/projects/islt/portraits/bloch-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `64`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `broad_negative`

Bloch is near the bottom wherever others can see him, 30th of 31 in the scenes, 13th of 14 in standing and 6th of 8 in belonging.

Bloch ranks 30th of 31 in the scenes, where he loses 100 times and wins 31, and 43 passages leave him worse off against 8 that leave him better. He is 13th of 14 in standing and 6th of 8 in belonging. The novel gives him the gaffes, the wrong clothes, the family manners, and later the new name, Jacques du Rozier. His success as a playwright is real but happens mostly offstage, and the rooms the novel shows are the ones that cost him. His fortune recovers a little from Le Côté de Guermantes II to the Bal de têtes, where he is an established man of letters.

Why interesting:

- His scene record, 31 wins and 100 losses, is the most one-sided losing record of any often-seen character.
- All three measures agree about him, which is rare. Together they make him the book's most relentless study of the outsider the salons will not absorb.
- His late rise is small but real. By the Bal de têtes, the young defer to him.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1317 ± 129 | 1187.7 | 30 of 31 | 64 | -0.692 | 0.8975 | 8/43/2/11 |
| prestige | 1479 ± 173 | 1305.7 | 13 of 14 | 64 | -0.046 | 0.0934 | 2/5/0/57 |
| inclusion | 1412 ± 174 | 1237.4 | 6 of 8 | 64 | -0.152 | 0.2444 | 3/10/0/51 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 24 | -0.734 | 0.0 | -0.122 |
| v2-p2-noms-de-pays-le-pays | 13 | -0.723 | -0.045 | -0.189 |
| v1-p1-combray | 6 | -0.633 | 0.0 | -0.59 |
| v7-p4-le-bal-de-tetes | 7 | -0.44 | 0.0 | +0.127 |
| v3-p2 | 3 | -1.053 | -0.253 | -0.58 |

Reading path:

- Mme de Villeparisis's matinée: `/projects/islt/fr-original/v3-p1`
- Combray: the school friend the family distrusts: `/projects/islt/fr-original/v1-p1-combray`
- Among the Guermantes: `/projects/islt/fr-original/v3-p2`

Notable units:

- Bloch is bluntly called idiotic and an imbecile by the father after his pretentious non-answer.: `/projects/islt/fr-original/v1-p1-combray#p-176`
- A second, more shocking gaffe -- mocking a guest's outdated predictions and implying senility -- is explicitly framed by the narrator as exposing Bloch's poor upbringing.: `/projects/islt/fr-original/v3-p1#p-536`
- The narration's summary judgment is severe and unqualified: ill-bred, neurotic, snobbish, and blind to the fault he detects in others.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-181`

## Françoise

- Slug: `francoise`
- Portrait default: `/projects/islt/portraits/francoise-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `61`
- Archetype signs: `advantage +1, prestige -1, inclusion -1`
- Pattern: `advantage_high_durable`

The family's cook ranks 5th of 31 in the scenes, ahead of most of the aristocrats in the novel.

Françoise ranks 5th of 31 in the scenes, on a winning record of 51 wins, 38 losses and 11 draws. The kitchen, the sickroom and the servants' table are her ground, and the novel lets her win there more often than most of the aristocrats win in theirs. The deference she commands from footmen, doctors and households is standing too, but the novel shows it too seldom to rank. Her fortune peaks in Autour de Mme Swann and settles back near where it began.

Why interesting:

- She wins more of her scenes than all but four characters, on genuinely winning footing.
- The deference she commands counts as standing on the same scale as a duchesse's, though the novel shows it too rarely to rank.
- Her record is durable more than dazzling: 51 wins, 38 losses and 11 draws.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1581 ± 133 | 1448.1 | 5 of 31 | 61 | +0.086 | 0.5864 | 20/17/0/24 |
| prestige | 1480 ± 222 | 1258.5 | insufficient evidence | 61 | +0.052 | 0.0516 | 3/0/0/58 |
| inclusion | 1337 ± 375 | 961.5 | insufficient evidence | 61 | -0.013 | 0.0128 | 0/1/0/60 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p2 | 8 | +0.145 | 0.0 | 0.0 |
| v1-p1-combray | 9 | -0.137 | 0.0 | 0.0 |
| v2-p1-autour-de-mme-swann | 4 | +1.24 | +0.425 | 0.0 |
| v4-p2 | 8 | -0.206 | 0.0 | 0.0 |
| v3-p1 | 10 | -0.118 | +0.07 | 0.0 |

Reading path:

- The Combray kitchen: `/projects/islt/fr-original/v1-p1-combray`
- Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Paris, and the second Balbec: `/projects/islt/fr-original/v4-p2`

Notable units:

- Françoise is clearly elevated by the narrator's extended comparison of her household perceptiveness to near-scientific, quasi-divinatory expertise.: `/projects/islt/fr-original/v3-p2#p-121`
- Françoise is left vulnerable and suffering, provoked into breathless distress by the narrator's deliberate cruelty and display of power over her through money spent on someone she dislikes.: `/projects/islt/fr-original/v4-p2#p-166`
- The passage decisively lowers the evaluation of Françoise by exposing deliberate, patient cruelty toward the kitchen maid and other non-family dependents beneath her celebrated gentleness.: `/projects/islt/fr-original/v1-p1-combray#p-261`

## Gilberte

- Slug: `gilberte`
- Portrait default: `/projects/islt/portraits/gilberte-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `57`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `inclusion_positive_prestige_positive_advantage_negative`

Swann's daughter changes her name twice and moves further inside society each time, ending 5th of 14 in standing and 4th of 8 in belonging.

Gilberte ranks 5th of 14 in standing, 4th of 8 in belonging and 17th of 31 in the scenes, where she wins and loses almost equally, 64 to 61. Her standing and belonging follow the novel's study of changed names: Mlle Swann, then Mlle de Forcheville, then the marquise de Saint-Loup, each name opening a door the last one could not. The salon that would not receive Mlle Swann receives Mlle de Forcheville. Her scenes run the other way. Her fortune in them falls clearly, from the Champs-Élysées of Noms de pays : le nom to the Bal de têtes.

Why interesting:

- The Guermantes salon that would not receive Mlle Swann receives Mlle de Forcheville, one of the plainest boundary crossings in the book.
- Her strengths run opposite to her father's. His standing and belonging collapse as hers grow.
- Her scenes, even overall, decline clearly across the novel, from the girl of the Champs-Élysées to the marquise of the Bal de têtes.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1503 ± 119 | 1384.0 | 17 of 31 | 57 | -0.028 | 0.4766 | 12/18/0/27 |
| prestige | 1630 ± 154 | 1476.1 | 5 of 14 | 57 | +0.085 | 0.174 | 6/2/0/49 |
| inclusion | 1517 ± 181 | 1336.2 | 4 of 8 | 57 | +0.05 | 0.0712 | 2/1/1/53 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p1-autour-de-mme-swann | 24 | -0.002 | 0.0 | -0.025 |
| v6-p2 | 7 | -0.414 | +0.343 | 0.0 |
| v1-p3-noms-de-pays-le-nom | 6 | +0.68 | +0.133 | 0.0 |
| v6-p4 | 4 | -0.532 | +0.01 | +0.44 |
| v7-p4-le-bal-de-tetes | 5 | -0.468 | -0.156 | 0.0 |

Reading path:

- The Champs-Élysées: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom`
- The Swann household: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Mlle de Forcheville: `/projects/islt/fr-original/v6-p2`

Notable units:

- Gilberte is intensely idealized and elevated in the narrator's perception, her mere name carrying overwhelming poetic and emotional value.: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom#p-6`
- Her standing visibly rises inside the world of the passage: people who had never noticed her now seek presentations and comment on the match.: `/projects/islt/fr-original/v6-p4#p-1`
- Her name loses its purchasing power as she spends it on a milieu that depreciates it, and she ends by receiving no one of the society she had wanted.: `/projects/islt/fr-original/v6-p4#p-6`

## Norpois

- Slug: `norpois`
- Portrait default: `/projects/islt/portraits/norpois-default-vermeer-proustian-20260425-1432.png`
- Annotation units: `54`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `reputation_ranked_scenes_even`

The ambassador is deferred to everywhere and wins about half his scenes, ranking 11th of 14 in standing and 20th of 31 in the scenes.

Norpois ranks 11th of 14 in standing and 20th of 31 in the scenes, where his record is almost a perfect draw: 45 wins, 44 losses and 12 draws. The deference paid to an ambassador is shown constantly, even in passages where his conversation wins nothing, which is very close to the joke the novel tells about him. His fortune slides from Le Côté de Guermantes I to the wartime chapters, where his newspaper articles have become a target of the narrator's irony.

Why interesting:

- The novel stages the ceremony around him relentlessly, and mostly with irony.
- His scene record is nearly a perfect draw, 45-44-12. The master of official language neither wins nor loses rooms.
- His standing outlasts his scenes: 11th of 14 in standing, 20th of 31 in the scenes.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1507 ± 135 | 1371.4 | 20 of 31 | 54 | -0.069 | 0.4894 | 16/17/1/20 |
| prestige | 1545 ± 196 | 1348.5 | 11 of 14 | 54 | +0.101 | 0.1274 | 6/1/0/47 |
| inclusion | 1579 ± 232 | 1347.5 | insufficient evidence | 54 | +0.013 | 0.0133 | 1/0/0/53 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p1-autour-de-mme-swann | 21 | -0.069 | +0.222 | 0.0 |
| v3-p1 | 22 | +0.115 | +0.004 | +0.033 |
| v6-p3 | 6 | -0.398 | 0.0 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 1 | -0.9 | 0.0 | 0.0 |
| v2-p2-noms-de-pays-le-pays | 1 | -0.88 | +0.7 | 0.0 |

Reading path:

- Dinner with the narrator's parents: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Mme de Villeparisis's matinée: `/projects/islt/fr-original/v3-p1`
- Venice: `/projects/islt/fr-original/v6-p3`

Notable units:

- Norpois is openly ridiculed as tedious and intellectually hollow by both the narrator's analysis and the direct mockery of Bergotte and Mme Swann.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-216`
- Norpois's standing is repeatedly and publicly confirmed: sought after across the political spectrum, praised in print, and granted a notable royal audience.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-1`
- Norpois's authority within the family is shown as effectively unquestionable, overturning the father's established positions on two separate matters with a single word.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-6`

## la grand-mère

- Slug: `la-grand-mere`
- Portrait default: `/projects/islt/portraits/la-grand-mere-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `48`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `advantage_strongly_positive`

The narrator's grandmother wins her scenes, 9th of 31, and her fortune rises through the book, most of all after her death.

The narrator's grandmother ranks 9th of 31 in the scenes, on 37 wins and 31 losses, with more passages leaving her better off than worse. Her belonging and standing are staged too rarely to rank, though both lean warmly upward, including the princesse de Luxembourg's greeting at Balbec. Her fortune rises from Combray to Sodome et Gomorrhe II, where the narrator, a year after her death, finally grieves for her. It is one of the few arcs in the book that climb after the character has died.

Why interesting:

- Her scenes genuinely go her way, with 17 passages leaving her better off against 14 worse, which is rarer in this novel than it sounds.
- The princesse de Luxembourg's greeting at Balbec, treating a bourgeois grandmother as her equal, is one of the plainest displays of standing in the book.
- Her fortune peaks in the passages of grief in Sodome et Gomorrhe II, when the narrator understands for the first time that she is gone.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1563 ± 145 | 1418.6 | 9 of 31 | 48 | +0.177 | 0.6675 | 17/14/0/17 |
| prestige | 1601 ± 235 | 1366.0 | insufficient evidence | 48 | +0.053 | 0.1185 | 3/2/0/43 |
| inclusion | 1752 ± 257 | 1494.8 | insufficient evidence | 48 | +0.047 | 0.0792 | 3/1/0/44 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p2-noms-de-pays-le-pays | 24 | +0.219 | +0.105 | +0.063 |
| v3-p1 | 9 | -0.289 | 0.0 | +0.08 |
| v3-p2 | 4 | +1.54 | 0.0 | 0.0 |
| v1-p1-combray | 7 | -0.327 | 0.0 | 0.0 |
| v4-p2 | 3 | +0.817 | 0.0 | 0.0 |

Reading path:

- Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Combray: `/projects/islt/fr-original/v1-p1-combray`
- Paris, and the telephone call from Doncières: `/projects/islt/fr-original/v3-p1`

Notable units:

- She is venerated almost to sanctification — her face, her hair, even the partition wall she knocks through are described as spiritualized by contact with her tenderness.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-36`
- She pleads in vain and is described as already defeated ('vaincue d'avance'), departing sad and discouraged, though bearing it with a gentle, self-effacing smile.: `/projects/islt/fr-original/v1-p1-combray#p-11`
- She is elevated by an explicit comparison to professional caregivers, her pity and devotion framed as vaster and more selfless than any paid or vowed care.: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays#p-31`

## docteur Cottard

- Slug: `docteur-cottard`
- Portrait default: `/projects/islt/portraits/docteur-cottard-default-vermeer-proustian-20260425-1609.png`
- Annotation units: `37`
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `advantage_positive_texture_mocking`

The Verdurins' doctor wins more scenes than he loses, 14th of 31 in the scenes, while the narration laughs at him.

Cottard ranks 14th of 31 in the scenes, on 50 wins and 43 losses. His wins come mostly from the Verdurin salon's crowded evenings, where his puns land with the faithful and his diagnoses impress them. He is a figure of fun in the telling, with 19 passages mocking him against 9 that flatter him, yet the outcomes go his way, which is precisely Proust's joke about medicine. His standing as an eminent specialist is shown too rarely to rank. His fortune rises from Un amour de Swann to Le Côté de Guermantes II, the doctor making his way up.

Why interesting:

- The narration mocks him, 19 passages against 9, while the scenes keep handing him the win.
- He is both the idiot of the salon and the great clinician, and his record carries both.
- His standing as a specialist is asserted in the later volumes but shown too rarely to rank.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1555 ± 148 | 1407.1 | 14 of 31 | 37 | -0.165 | 0.7129 | 9/19/1/8 |
| prestige | 1530 ± 206 | 1324.0 | insufficient evidence | 37 | +0.057 | 0.0568 | 3/0/0/34 |
| inclusion | 1425 ± 258 | 1167.1 | insufficient evidence | 37 | 0.0 | 0.0 | 0/0/0/37 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p2-un-amour-de-swann | 22 | -0.359 | +0.034 | 0.0 |
| v4-p2 | 8 | -0.393 | +0.081 | 0.0 |
| v2-p1-autour-de-mme-swann | 4 | +0.7 | +0.175 | 0.0 |
| v3-p2 | 1 | +1.76 | 0.0 | 0.0 |
| v7-p1-a-tansonville | 1 | +0.96 | 0.0 | 0.0 |

Reading path:

- The little clan: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- La Raspelière: `/projects/islt/fr-original/v4-p2`
- The Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`

Notable units:

- Events prove his imperious prescription right against the family's objections, and the household that had hidden its disobedience ends by crowning him a great clinician.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-116`
- His decisive competence in a medical crisis is framed as a form of unexpected greatness, elevating him above his usual ordinariness.: `/projects/islt/fr-original/v3-p2#p-21`
- The narrator explicitly frames him as stupid and incredulous, then as gullible enough to be talked out of his own correct astonishment.: `/projects/islt/fr-original/v1-p2-un-amour-de-swann#p-116`

## Morel

- Slug: `morel`
- Portrait default: `/projects/islt/portraits/morel-default-vermeer-proustian-20260813-0900.png`
- Annotation units: `35`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_high_scene_negative`

The violinist ranks 2nd of 14 in standing, behind only the duchesse de Guermantes, and loses most of his scenes, where he is 25th of 31.

Morel ranks 2nd of 14 in standing, behind only the duchesse de Guermantes. His talent, and the protection Charlus buys for it, set him above dukes and barons in the salons. The scenes themselves go against him. He is 25th of 31 in the scenes, and for every passage that leaves him better off, four or more leave him worse. His belonging is staged too rarely to rank. His fortune reaches its low point in La Prisonnière, where he breaks with Charlus, and recovers by the Bal de têtes, where the wartime deserter has become a decorated and respected man.

Why interesting:

- His standing rests on protectors, Charlus above all, which makes it real and precarious at once.
- His scenes are sharply negative: 22 passages leave him worse off and 5 leave him better.
- His fortune bottoms out in La Prisonnière and climbs back by the Bal de têtes, where the deserter ends the novel honored.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1476 ± 134 | 1342.5 | 25 of 31 | 35 | -0.718 | 0.8773 | 5/22/0/8 |
| prestige | 1770 ± 188 | 1582.1 | 2 of 14 | 35 | +0.206 | 0.2063 | 6/0/0/29 |
| inclusion | 1499 ± 229 | 1269.9 | insufficient evidence | 35 | -0.041 | 0.0414 | 0/2/0/33 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v5 | 13 | -0.964 | +0.186 | -0.058 |
| v4-p2 | 12 | -0.634 | +0.117 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 4 | -0.25 | +0.41 | 0.0 |
| v7-p4-le-bal-de-tetes | 1 | 0.0 | +1.76 | 0.0 |
| v6-p2 | 1 | -1.7 | 0.0 | 0.0 |

Reading path:

- The Verdurin salon, under Charlus's patronage: `/projects/islt/fr-original/v4-p2`
- The break with Charlus: `/projects/islt/fr-original/v5`
- Wartime: the protégé outlives the protector: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

Notable units:

- The narrator's extended, explicit exposure of his cynical calculation and self-deceiving venality strongly diminishes him.: `/projects/islt/fr-original/v5#p-86`
- Morel is exposed as viciously cruel toward a defenseless woman and, per the narrator's aside about his cowardice, flees as soon as Jupien is heard returning.: `/projects/islt/fr-original/v5#p-236`
- The narrator's analysis strips away Morel's momentary display of shame and reveals a habitual, mercenary cruelty toward the women he seduces, clearly diminishing him.: `/projects/islt/fr-original/v5#p-266`

## Rachel

- Slug: `rachel`
- Portrait default: `/projects/islt/portraits/rachel-default-vermeer-proustian-20260813-0900.png`
- Annotation units: `29`
- Archetype signs: `advantage +1, prestige +1, inclusion +1`
- Pattern: `scenes_strong_standing_unranked`

Saint-Loup's mistress becomes the celebrated actress of the last chapter, and she ranks 8th of 31 in the scenes.

Rachel appears across the whole novel, from the young woman Saint-Loup keeps in Le Côté de Guermantes to the actress whose recital empties la Berma's salon at the Bal de têtes. She ranks 8th of 31 in the scenes, with 27 wins, 20 losses and 9 draws. Her standing looks high, but the novel shows it in only 10 passages, too few to rank, and her belonging is thinner still. Her fortune rises from Le Côté de Guermantes II to the end of the book, though not steeply enough to call a clear arc.

Why interesting:

- Her triumph at the Bal de têtes is one of the sharpest reversals in the novel. The woman once sold for twenty francs is celebrated at the princesse's matinée while la Berma waits for guests who never come.
- She wins more of her scenes than she loses, 27 to 20, which is striking for a character the narrator first meets as a woman for sale.
- Her standing is the open question. It looks high, but the novel shows it too seldom to rank.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1600 ± 173 | 1427.0 | 8 of 31 | 29 | -0.216 | 0.6092 | 7/14/0/8 |
| prestige | 1639 ± 224 | 1415.2 | insufficient evidence | 29 | +0.041 | 0.2003 | 2/2/0/25 |
| inclusion | 1648 ± 462 | 1185.3 | insufficient evidence | 29 | +0.025 | 0.0248 | 1/0/0/28 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 17 | -0.06 | -0.136 | 0.0 |
| v7-p4-le-bal-de-tetes | 6 | -0.45 | +0.583 | +0.12 |
| v3-p2 | 3 | -0.603 | 0.0 | 0.0 |
| v2-p1-autour-de-mme-swann | 1 | -0.72 | 0.0 | 0.0 |
| v4-p2 | 1 | 0.0 | 0.0 | 0.0 |

Reading path:

- "Rachel quand du Seigneur" and Saint-Loup's love: `/projects/islt/fr-original/v3-p1`
- Her recital, and la Berma's empty salon: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- Rachel is exposed as capable of premeditated, orchestrated cruelty against a vulnerable rival, a serious local diminishment even though the narrator hesitates to voice it aloud.: `/projects/islt/fr-original/v3-p1#p-371`
- Paris itself reports her as the real hostess of a Guermantes matinée and the duchesse's chosen friend; her local standing rises sharply.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-66`
- Rachel visibly enacts and registers the reversal of fortune, condescendingly receiving the once-illustrious Berma's children before onlookers.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-81`

## la mère du narrateur

- Slug: `la-mere-du-narrateur`
- Portrait default: `/projects/islt/portraits/la-mere-du-narrateur-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `28`
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `familial_positive`

The narrator's mother has the cleanest winning record in the book, 6th of 31 in the scenes, with nine passages lifting her for every one that cuts.

The narrator's mother ranks 6th of 31 in the scenes, the highest of anyone in the family, on 34 wins and 16 losses. Nine passages leave her better off for every one that leaves her worse, the most consistently favorable record of any ranked character. Her authority is domestic and effective: the goodnight kiss, the verdicts the household accepts, the quiet management of the father. Her standing and belonging are staged too rarely to rank, and her belonging leans slightly negative, the cost of being the one who decides who reaches the child rather than the one admitted anywhere herself.

Why interesting:

- Nine passages lift her for every one that cuts, the most favorable ratio of any ranked character.
- She ranks 6th of 31 in the scenes, above everyone else in the family and above most of the salons.
- Her belonging leans against her, a quiet irony. The guardian of the family's inside is rarely shown crossing into anyone else's.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1605 ± 169 | 1435.9 | 6 of 31 | 28 | +0.273 | 0.3177 | 9/1/0/18 |
| prestige | 1751 ± 280 | 1470.6 | insufficient evidence | 28 | +0.005 | 0.0482 | 1/1/0/26 |
| inclusion | 1399 ± 231 | 1168.4 | insufficient evidence | 28 | -0.108 | 0.1618 | 1/3/0/24 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p1-combray | 6 | +0.581 | 0.0 | 0.0 |
| v3-p2 | 5 | +0.34 | 0.0 | 0.0 |
| v6-p2 | 2 | 0.0 | +0.375 | -0.85 |
| v2-p1-autour-de-mme-swann | 6 | +0.055 | -0.1 | +0.012 |
| v2-p2-noms-de-pays-le-pays | 2 | +0.7 | 0.0 | 0.0 |

Reading path:

- Combray and the goodnight kiss: `/projects/islt/fr-original/v1-p1-combray`
- Paris: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- After the grandmother's death: `/projects/islt/fr-original/v3-p2`

Notable units:

- She is explicitly called an admirable reader, praised at length for her tact, tenderness, and interpretive skill in reading aloud to the narrator.: `/projects/islt/fr-original/v1-p1-combray#p-36`
- The mother is elevated through the narrator's sympathetic portrayal of the depth and totality of her grief and love.: `/projects/islt/fr-original/v3-p2#p-76`
- Maman is pointedly shut out of the princesse's courtesy — ignored, unaddressed for the visit, and denied even a parting handshake despite having been specifically summoned.: `/projects/islt/fr-original/v6-p2#p-61`

## Bergotte

- Slug: `bergotte`
- Portrait default: `/projects/islt/portraits/bergotte-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `27`
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `advantage_positive_reputation_offstage`

The admired writer wins a little more often than he loses, 21st of 31 in the scenes, while his fame happens mostly offstage.

Bergotte ranks 21st of 31 in the scenes, with 25 wins and 21 losses. The aura of the name is real, and it belongs to his standing, which leans high but is staged too rarely to rank. The novel shows him less than his fame makes it feel: the author of the narrator's youth, met at the Swanns' table, and later the sick old man who dies before Vermeer's View of Delft. His fortune rises to La Prisonnière, where he dies before the little patch of yellow wall, and slips in the last volume.

Why interesting:

- His scenes are winning but modest, 25 wins to 21 losses, closer to the dying man before the Vermeer than to the legend at the Swanns' table.
- His standing leans among the highest of anyone the novel shows too rarely to rank. The fame is real, and the novel mostly keeps it offstage.
- Like Norpois, he is more a reputation than a presence in the rooms the novel shows.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1549 ± 181 | 1368.0 | 21 of 31 | 27 | +0.127 | 0.8517 | 11/8/0/8 |
| prestige | 1779 ± 468 | 1311.7 | insufficient evidence | 27 | +0.037 | 0.1467 | 2/2/0/23 |
| inclusion | 1268 ± 492 | 776.0 | insufficient evidence | 27 | 0.0 | 0.0 | 0/0/0/27 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v2-p1-autour-de-mme-swann | 13 | -0.039 | -0.061 | 0.0 |
| v3-p1 | 4 | +0.489 | 0.0 | 0.0 |
| v5 | 2 | +1.3 | 0.0 | 0.0 |
| v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle | 2 | -0.445 | -0.34 | 0.0 |
| v1-p1-combray | 3 | +0.62 | +0.24 | 0.0 |

Reading path:

- Lunch at the Swanns': `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- The Guermantes world: `/projects/islt/fr-original/v3-p1`
- Combray: the books before the man: `/projects/islt/fr-original/v1-p1-combray`

Notable units:

- The narrator's private aesthetic reverence for Bergotte's style and thought is the passage's dominant evaluative movement, praising him without qualification.: `/projects/islt/fr-original/v1-p1-combray#p-186`
- The narrator explicitly and emphatically ranks Bergotte's genius above the wit and distinction of his childhood entourage, crediting him with transforming mediocre material into art in a way they could not.: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann#p-201`
- Bergotte is posthumously elevated by the narrator's framing of his books as angelic and his death as a kind of resurrection.: `/projects/islt/fr-original/v5#p-261`

## Legrandin

- Slug: `legrandin`
- Portrait default: `/projects/islt/portraits/legrandin-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `23`
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `advantage_negative_prestige_performed`

The snob loses nearly every scene he is in, 8 wins against 28 losses, while he performs a standing he does not have.

Legrandin loses 28 of his scenes and wins 8, and 15 passages leave him worse off against 2 that leave him better, one of the lowest ratings in the book. He appears in too few passages for his scenes to be ranked. What the novel records of him is his performance of standing: the bows calibrated for aristocratic eyes, the exquisite phrases, the snobbery he denounces in others. His standing, too thinly staged to rank, leans upward, because the pose is what the narration keeps witnessing. His fortune improves late, once he has become the comte de Méséglise. His profile is snobbery complete, the floor of the scenes and the ceiling of the pose.

Why interesting:

- He loses more than three scenes for every one he wins, though in too few passages to be ranked.
- His standing leans among the highest of those shown too rarely to rank, because what the narration witnesses is the performance of standing.
- The pairing, the floor of the scenes and the ceiling of the pose, is the anatomy of snobbery.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1285 ± 208 | 1076.9 | insufficient evidence | 23 | -0.547 | 0.7439 | 2/15/0/6 |
| prestige | 1760 ± 298 | 1462.8 | insufficient evidence | 23 | +0.013 | 0.1435 | 1/2/0/20 |
| inclusion | 1394 ± 494 | 900.6 | insufficient evidence | 23 | 0.0 | 0.0 | 0/0/0/23 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v3-p1 | 9 | -0.783 | -0.072 | 0.0 |
| v1-p1-combray | 8 | -0.455 | -0.106 | 0.0 |
| v6-p4 | 1 | -0.8 | +1.8 | 0.0 |
| v7-p4-le-bal-de-tetes | 2 | -0.9 | 0.0 | 0.0 |
| v5 | 1 | +0.7 | 0.0 | 0.0 |

Reading path:

- The Paris salons: `/projects/islt/fr-original/v3-p1`
- Combray: the snob on the church steps: `/projects/islt/fr-original/v1-p1-combray`
- The Bal de têtes: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

Notable units:

- The forger comparison is the sharpest and most explicit indictment in the sequence, decisively confirming Legrandin's absurd, self-defeating evasiveness rather than leaving any residual ambiguity.: `/projects/islt/fr-original/v1-p1-combray#p-281`
- He passes from isolated invitations to a genuine social position, and the duc de Guermantes' cover makes him the comte de Méséglise for a whole generation.: `/projects/islt/fr-original/v6-p4#p-6`
- Legrandin is transformed from a colorful, quick-witted figure into a pale, silent phantom of himself.: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes#p-11`

## Mme de Cambremer

- Slug: `mme-de-cambremer`
- Portrait default: `/projects/islt/portraits/mme-de-cambremer-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `22`
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `compact_negative`

Legrandin's sister, the young marquise, is last of 31 in the scenes, and no passage about her leaves her better off.

Mme de Cambremer ranks 31st of 31 in the scenes, last of every ranked character, with 15 wins and 41 losses and no passage that leaves her better off. Her standing leans hard downward but is staged too rarely to rank. She is the provincial snob, Legrandin's sister, whose pretensions every Parisian room declines to honor, and Charlus's treatment of her at la Raspelière is one of the book's plainest snubs. She marks the floor of the scenes as the duchesse marks their ceiling.

Why interesting:

- She is last in the scenes, and all 17 passages that move her leave her worse off.
- The novel shows her rarely and defeats her reliably.
- Her fortune reaches its low in Le Côté de Guermantes I and recovers somewhat by the last volume.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1317 ± 192 | 1125.5 | 31 of 31 | 22 | -0.831 | 0.8309 | 0/17/0/5 |
| prestige | 1378 ± 255 | 1122.1 | insufficient evidence | 22 | -0.064 | 0.0636 | 0/2/0/20 |
| inclusion | 1355 ± 260 | 1095.1 | insufficient evidence | 22 | -0.146 | 0.2082 | 1/3/0/18 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v4-p2 | 9 | -0.828 | -0.156 | -0.113 |
| v3-p1 | 3 | -1.753 | 0.0 | -0.567 |
| v1-p2-un-amour-de-swann | 4 | -1.008 | 0.0 | 0.0 |
| v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle | 1 | -0.82 | 0.0 | 0.0 |
| v7-p2-m-de-charlus-pendant-la-guerre | 1 | -0.72 | 0.0 | 0.0 |

Reading path:

- La Raspelière: `/projects/islt/fr-original/v4-p2`
- The Guermantes world: `/projects/islt/fr-original/v3-p1`
- Mme de Saint-Euverte's soirée: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- The narration's verdict on her is sustained and severe: her opinions are shown to be secondhand, her enthusiasm performed, her advanced theory nonsense, her erudition a form of snobbery.: `/projects/islt/fr-original/v4-p2#p-231`
- She is savaged by repeated, escalating bovine mockery in her absence, thoroughly discredited in the company's eyes.: `/projects/islt/fr-original/v3-p1#p-606`
- Mme de Cambremer is harshly mocked as vulgar, tiresome, and socially impossible.: `/projects/islt/fr-original/v3-p1#p-466`

## M. Vinteuil

- Slug: `m-vinteuil`
- Portrait default: `/projects/islt/portraits/m-vinteuil-default-vermeer-proustian-20260425-1923.png`
- Annotation units: `9`
- Archetype signs: `advantage +1, prestige +1, inclusion +0`
- Pattern: `rehabilitated_positive`

The humiliated piano teacher of Combray becomes, after his death, the composer whose septet transfigures La Prisonnière, the largest rise in the novel.

M. Vinteuil appears in only 9 passages, too few to rank in any measure, and their order tells a complete story. In Combray he is the shy music teacher shamed by his daughter's reputation. In Un amour de Swann his sonata, not yet known to be his, becomes the anthem of Swann's love. In La Prisonnière his septet, deciphered from his notes by his daughter's friend, reveals him as a great composer. His fortune rises from 1449 in Combray to 1917 in La Prisonnière, a clear rise and the largest in the book, earned entirely after his death.

Why interesting:

- His is the largest rise in the book, and it comes entirely after his death, through his music.
- The arc follows the chapters exactly: shame in Combray, the sonata in Un amour de Swann, the septet in La Prisonnière.
- His daughter's friend, who helped shame him, is the one who restores his work, one of the novel's strangest reparations.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1654 ± 268 | 1386.0 | insufficient evidence | 9 | +0.02 | 1.1667 | 4/3/1/1 |
| prestige | 1691 ± 351 | 1340.2 | insufficient evidence | 9 | +0.078 | 0.3 | 1/1/0/7 |
| inclusion | 1500 ± 700 | 800.0 | insufficient evidence | 9 | 0.0 | 0.0 | 0/0/0/9 |

Top chapters (by absolute movement):

| Chapter | Units | Advantage | Prestige | Inclusion |
| --- | --- | --- | --- | --- |
| v1-p1-combray | 5 | -0.868 | -0.2 | 0.0 |
| v1-p2-un-amour-de-swann | 3 | +0.867 | 0.0 | 0.0 |
| v5 | 1 | +1.92 | +1.7 | 0.0 |

Reading path:

- The septet: `/projects/islt/fr-original/v5`
- Combray: `/projects/islt/fr-original/v1-p1-combray`
- The sonata: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- The narration's verdict on him rises to the highest possible: an original of the rank of the greatest, whose work outranks everything previously known of him.: `/projects/islt/fr-original/v5#p-306`
- Vinteuil is savagely mocked after his death, reduced to a contemptuous epithet ('le vilain singe') in a scene the narrator frames as ritual desecration of his memory.: `/projects/islt/fr-original/v1-p1-combray#p-331`
- He is mocked and blamed by village gossip for tolerating his daughter's companion.: `/projects/islt/fr-original/v1-p1-combray#p-306`
