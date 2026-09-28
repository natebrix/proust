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

He loses the scene and keeps the room: his individual scenes still run against him, yet he is first in belonging, 4th of 14 in prestige, and 7th of 31 in advantage on the sheer certainty of the evidence.

The narrator is the novel's "I": nearly every scene passes through him, and scene by scene the scenes still go badly — 200 decided losses against 168 wins, with negative passages far outnumbering positive ones. Yet across the whole book his welcome never runs out: he ranks 1st of 8 in belonging and 4th of 14 in prestige, and his place in scene-level advantage (7th of 31) is less a verdict on his victories than on his measurability — no one in the book is weighed more often or more surely, and that certainty holds his floor where flashier figures wobble. The rooms keep receiving the man the scenes keep wounding; the split between lived defeat and durable acceptance remains the book's central irony made measurable.

Why interesting:

- His scene outcomes still lean against him — more decided losses than wins, negative passages nearly two to one — while all three of his standings sit in the upper half: the same passages, weighed differently.
- Because the whole novel passes through him, he is measured against more of the cast than any other figure, so his readings are the most certain in the book — his advantage rating carries the narrowest uncertainty of anyone's.
- His suffering is local and his acceptance is cumulative: no single scene secures his place, and no single defeat costs it.

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

- Balbec thresholds: the machinery of being received: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Guermantes admission: the observer absorbed: `/projects/islt/fr-original/v3-p2`
- The bal de têtes: survivor among the masks: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

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

First in scene-level advantage, first in prestige, and second only to the narrator in belonging — the most complete dominance the novel measures.

The duchesse holds the top of the book: 1st of 31 in scene-level advantage, 1st of 14 in prestige, and 2nd of 8 in belonging, behind only the narrator — no one else places in the top three of every register. Her scenes back it up: 225 decided wins against 92 losses, the wit crowning her far more often than it cuts her. Counted passage by passage, so that a crowded salon weighs as one scene rather than a dozen, she moves up past Forcheville in the scenes and past Morel in standing: her dominance is spread across the book, not piled up in a few full rooms. She is the book's measured establishment, and the measurements agree.

Why interesting:

- She ranks first in two registers and second in the third — the most complete high placement in the measured cast.
- Her scene record (225 wins, 92 losses across 354 decided comparisons) is the most lopsidedly victorious of any heavily-measured figure: the wit wins far more evenings than it loses.
- The only character above her anywhere is the narrator, in belonging: the observer the salons absorb outranks the hostess who admits him.

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

- High Guermantes concentration: `/projects/islt/fr-original/v3-p1`
- Continued positive confirmation: `/projects/islt/fr-original/v3-p2`
- Late reinforcing appearances: `/projects/islt/fr-original/v4-p2`

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

One of the most heavily measured men in the novel, and measured losing: below the middle in scene-level advantage, in the lower half of a prestige field he once led from the shadows, next to last in belonging.

Swann is staged constantly — 386 decided comparisons in advantage alone, more than anyone but the narrator — and the scenes go against him: 197 losses to 144 wins, with negative passages far outnumbering positive. His scene-level advantage sits below the middle (19th of 31). Prestige lands in the lower half (10th of 14) — a sobering number for the man Combray never realized dined with princes, because the novel stages his standing mostly in decline, through the marriage that costs him the rooms he owned. Belonging is his cleanest loss: 7th of 8, the elegant man who ends the book steered around as an embarrassment.

Why interesting:

- He is among the most heavily measured figures in the book, so his negative readings carry unusual evidentiary weight — this is not a small-sample verdict.
- His prestige rank (10th of 14) captures the tragedy structurally: the novel stages his standing almost entirely on its way down, after the marriage, so the measured Swann is the diminished one.
- Belonging near the bottom (7th of 8) squares with the book's late cruelty: the name unspeakable in the Guermantes household his person once graced.

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

- Primary negative concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Early counterweight and setup: `/projects/islt/fr-original/v1-p1-combray`
- Later negative reinforcement: `/projects/islt/fr-original/v4-p2`

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

Present everywhere and first nowhere: mid-table in scene-level advantage, below the middle of the prestige field his name would predict he'd own, and his belonging now too thinly staged to rank.

Saint-Loup is one of the most heavily staged figures in the novel, and the measurement finds breadth rather than dominance. His scene-level advantage sits mid-table (16th of 31, wins and losses nearly even across 234 decided comparisons). In prestige — the register his aristocratic bearing would predict he'd own — he ranks 9th of 14, behind his uncle Charlus and his great-aunt Villeparisis, and behind Odette and Gilberte, the woman he marries. His belonging, counted passage by passage, is staged too rarely to rank. He is accepted more than he is deferred to, a Guermantes who spends the name rather than banks it.

Why interesting:

- His prestige position inverts what his rank and bearing would suggest: 9th of the 14 characters the novel sizes there, below his own family and below Odette.
- His advantage record is almost perfectly even (105 wins, 108 losses across 234 decided comparisons): breadth of presence, not a run of triumphs, is what holds his place.
- His belonging leaned slightly negative and is now unranked: many passages include him, few stage him crossing a threshold.

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

- Main prestige / inclusion divergence: `/projects/islt/fr-original/v3-p1`
- Earlier positive concentration: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Late negative pressure: `/projects/islt/fr-original/v7-p1-a-tansonville`

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

Volatile in the scenes and in the upper half of them (12th of 31) — while her standing and her belonging are both, counted passage by passage, still open questions.

Albertine's scenes remain among the most conflicted measured — wins and losses nearly even (80 to 84), with more explicitly mixed passages than most of the cast — and her scene-level advantage sits in the upper half, 12th of 31. Prestige leans slightly upward, but the evidence comes from few passages and no longer supports a rank. Belonging, which an older reading ranked dead last, is also unranked: the sequestration chapters stage fewer true boundary events than that reading counted. Her exclusion was real, but much of it was the narrator's arrangement rather than the world's verdict — and the measurement respects that difference.

Why interesting:

- Her belonging reading changed more than anyone's: from dead last to unranked, because the boundary criteria distinguish being shut in by one man from being shut out by the world.
- Her standing is staged mostly through the elegance the narrator cultivates — real, but in too few passages to rank.
- Her scene volatility persists: near-even outcomes with an unusual share of explicitly mixed passages, a genuine internal split rather than a slide.

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

- Main negative concentration in La Prisonnière: `/projects/islt/fr-original/v5`
- Afterlife of loss in Albertine disparue: `/projects/islt/fr-original/v6-p1`
- Continuing exclusion pressure: `/projects/islt/fr-original/v6-p2`

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

Ranked in all three registers, and highest where the old reading couldn't see her: 3rd of 14 in prestige — the demi-mondaine ends the book outranking most of the Faubourg.

Odette is one of the eight figures the novel ranks in every register, and her strongest is the one the evidence used to leave open: prestige, where she stands 3rd of 14, behind only the duchesse de Guermantes and Morel. Her scene-level advantage sits at the exact middle (15th of 31, wins and losses nearly even across 248 decided comparisons), and belonging sits mid-low (5th of 8). The shape is the novel's longest social climb made measurable: the woman the salons refused to receive ends with a certified standing above most of the people who refused her.

Why interesting:

- Her prestige standing — 3rd of 14 — was invisible to the old reading, which had too little staged evidence to size her there at all; the enriched reading certifies the climb.
- The three registers disagree about her in the most Proustian way: standing high, scenes even, belonging modest — received as a name long before she is received as a person.
- In scene-level advantage her record is nearly balanced (112 wins, 107 losses), steady unglamorous footing rather than a dramatic arc.

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

- Mild prestige lean around Mme Swann: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Negative counterweight in Swann's love: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Later reversals in Guermantes-adjacent society: `/projects/islt/fr-original/v3-p1`

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

Ranked in all three registers — upper half in the scenes, mid-table in prestige, 3rd of 8 in belonging — a great position, measured on its way to a great fall.

Where the oldest evidence left Charlus last in prestige and unrankable in belonging, the witnessed-standing and boundary criteria certify what the novel actually stages for most of its length: a man of real position — 13th of 31 in scene-level advantage, 7th of 14 in prestige, 3rd of 8 in belonging. His ranks are middling for a baron because the book stages him high and then brings him down, and a rank averages the two. The fall lives in the trajectory: the wartime chapters and the Verdurin expulsion drag his late ratings down from a summit the earlier volumes spent thousands of pages building. He is the book's great instance of position as altitude — measured high precisely so the descent can be measured too.

Why interesting:

- He is one of the eight figures ranked in all three registers, with belonging 3rd of 8 — the clubbable baron the novel installs everywhere before it evicts him.
- His fall is a trajectory fact more than a rank fact: the standing is high through the early volumes and collapses at the end, which is precisely the shape the novel wrote.
- His scene record (133 wins, 115 losses) is positive overall, a reminder of how long the novel lets him win before it stops.

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

- Salon-world negative pressure: `/projects/islt/fr-original/v4-p2`
- Late negative cluster with Morel: `/projects/islt/fr-original/v5`
- Wartime degradation: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

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

The title earns a rank and it is the last one — 14th of 14 in prestige — while the rooms go against him too: 27th of 31 in scene-level advantage.

The duc's title is measured, and it lands at the floor: 14th of 14 in prestige, a Guermantes name that commands ceremony and not much more. It does not rescue his scenes either. In scene-level advantage he sits 27th of 31, losing 126 decided comparisons against 75 wins, with passages that cut him outnumbering those that lift him ten to one — the Jockey Club defeat, the deceptions endured, the wife's wit at his expense. His belonging stays too thin to rank. The gap between name and man has closed from the wrong side: in the passages the novel actually stages, neither holds.

Why interesting:

- His two ranks quantify the book's running joke about him: last of 14 in prestige, 27th of 31 in scene-level advantage.
- His negative scene texture is the most lopsided of the great aristocrats (5 positive passages against 53) — the comedy of the duc is structural, not incidental.
- Against his wife the comparison is total: she leads in two registers; he cracks the top half of none.

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

- Primary Guermantes counterexample: `/projects/islt/fr-original/v3-p2`
- Late decline reinforcement: `/projects/islt/fr-original/v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle`
- Final negative return: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

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

Ranked in all three registers, and the three disagree completely: 6th of 14 in prestige, 10th of 31 in the scenes, dead last of 8 in belonging — the hostess the book crowns and never seats.

Mme Verdurin is one of the eight figures ranked in advantage, prestige, and belonging at once, and the measurement sharpens her contradiction to its final form. Prestige: 6th of 14, real certified standing, ending as it does in the princesse de Guermantes title. Advantage: 10th of 31, though the texture is brutal — passages that lift her are outnumbered eight to one by passages that cut. Belonging: dead last, 8th of 8. The woman who built the century's most exclusive interior is, by the book's own staging, never securely inside anything — bypassed at her own soirées, mocked in her own title. The clan was a fortress built by someone the walls never protected.

Why interesting:

- Her three ranks tell three different stories — upper-half standing, upper-third scenes, last-place belonging — the widest three-way disagreement in the measured cast.
- Her last place in belonging is earned at her own parties: the corpus's adjudicated divergences include guests bypassing her as hostess while a queen rescues her, and the Faubourg mocking her as princesse.
- Her prestige is the book's great manufactured standing — built, purchased, and finally titled — and the numbers certify it while refusing it warmth.

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

- Primary Verdurin-world concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Late negative counterpoint: `/projects/islt/fr-original/v5`
- Wartime reversal zone: `/projects/islt/fr-original/v7-p2-m-de-charlus-pendant-la-guerre`

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

The quiet riser: 4th of 31 in scene-level advantage and ranked 8th of 14 in prestige — the salonnière the old evidence mistook for background.

Mme de Villeparisis is one of the clearest promotions since the oldest reading: 4th of 31 in scene-level advantage (72 decided wins against 38 losses), with a ranked standing in prestige (8th of 14) besides. The rise is not mysterious — her matinées are among the book's most heavily staged social machinery, and the witnessed-standing criteria credit the hostess who runs the room rather than only the guests who shine in it. Belonging alone stays too thin to rank, the famous ambiguity of her position — received by everyone, placed by no one — surviving as an honestly open question.

Why interesting:

- Her advantage rank rose from the median of the oldest reading to 4th of 31 — the stricter criteria found the authority her matinées actually exercise.
- She is the foundation corpus's one adjudicated case of prestige-without-belonging at Balbec, and the current reading preserves exactly that shape: ranked standing, unrankable belonging.
- Her win rate (72 to 38) is among the strongest of any heavily measured figure — quiet dominance the old reading's thin evidence could not see.

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

- Main split concentration: `/projects/islt/fr-original/v3-p1`
- Brief positive lean in Balbec prestige: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Late negative counterweight: `/projects/islt/fr-original/v6-p3`

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

Near the bottom everywhere the room can see him: 30th of 31 in the scenes, 13th of 14 in prestige, 6th of 8 in belonging.

Bloch's advantage reading is among the harshest measured — 30th of 31, losses outnumbering wins better than three to one (100 to 31), negative passages five to one. Prestige, once a startling 3rd-of-8 in a tiny early field, now places him 13th of 14, next to last — which is what the text has staged all along: the gaffes, the wrong clothes, the name changed to Jacques du Rozier. Belonging completes the picture at 6th of 8. His late success as a dramatist is real but arrives mostly offstage; the rooms the novel actually stages are the ones that cost him. The consistency across all three registers is the point: the book's most relentless study of the socially unabsorbed.

Why interesting:

- His early 3rd-of-8 prestige rank was a small-field artifact; the current reading places him 13th of 14 in prestige — a demotion that brings the number into line with every scene the novel wrote him.
- His advantage record (31-100-15) is the most lopsided of any heavily measured figure — being cut down in the room is his structural role.
- All three registers agree on him, which they do for almost no one else — and their agreement is itself the reading.

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

- Primary Guermantes-world humiliation zone: `/projects/islt/fr-original/v3-p1`
- Early negative setup: `/projects/islt/fr-original/v1-p1-combray`
- Continued social diminishment: `/projects/islt/fr-original/v3-p2`

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

Top five in the scenes she was born to win — 5th of 31 in advantage — while her standing, witnessed in a handful of passages, stays too thin to rank.

Françoise holds 5th of 31 in scene-level advantage on a genuinely winning record (51 decided wins to 38 losses), in a field that includes the salon figures the stricter criteria promoted. Her prestige is witnessed — the deference of footmen, doctors, and households counts as standing too — but only in a handful of passages, too few to rank. Belonging stays unranked, the servant's position at the family's center and margin at once remaining, fittingly, unmeasurable.

Why interesting:

- Her old first-place advantage rank was partly a small-field artifact; 5th of 31 in scene-level advantage, on a real winning record, is the sturdier claim.
- The witnessed-standing criterion is blind to class, exactly as the novel's own attention is: her prestige evidence exists, a servant measured in the register built for duchesses, even if it is too sparse to rank.
- Her scenes stay nearly even (51-38-11): durable footing, not a hot streak, is what the ranking reflects.

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

- Early domestic concentration: `/projects/islt/fr-original/v1-p1-combray`
- Strongest positive concentration, in Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- The rare negative pocket in an otherwise positive record: `/projects/islt/fr-original/v4-p2`

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

Ranked in all three registers and strongest in prestige (5th of 14) — the girl who changes names twice and lands, each time, further inside.

Gilberte is ranked everywhere the novel measures: 5th of 14 in prestige, 4th of 8 in belonging, 17th of 31 in scene-level advantage, her scenes themselves nearly even (64 decided wins, 61 losses). Her standing and her belonging are the registers that fit the book's great study in absorbed identity — Swann's daughter becoming Mlle de Forcheville becoming the marquise de Saint-Loup, each name a door that opens on a room the last one couldn't enter. The corpus catches the mechanism directly: her walk into the Guermantes salon under her new name is one of its cleanest boundary events.

Why interesting:

- Her belonging rank rests on the novel's most explicit boundary machinery: the same salon that would not receive Mlle Swann receives Mlle de Forcheville.
- She is one of the eight characters ranked in all three registers, with the strengths running opposite to her father's — his standing and belonging collapse as hers compound.
- Her scene record is almost perfectly even (64-61): she never dominates a room, and never needs to; the names do the work.

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

- Early positive concentration: `/projects/islt/fr-original/v1-p3-noms-de-pays-le-nom`
- Mme Swann-world extension: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Late instability in belonging: `/projects/islt/fr-original/v6-p2`

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

The diplomat demoted by better evidence: from 7th in the oldest reading to 20th of 31 in scene-level advantage, while prestige ranks him 11th of 14 — the authority was always reputation more than performance.

Norpois is the clearest deflation in the measured cast. The oldest evidence placed him 7th in scene-level advantage; the current reading places him 20th of 31, his scenes an almost perfect draw (45 decided wins, 44 losses). What he has instead is a ranked prestige standing, 11th of 14 — because the deference paid to an ambassador is witnessed constantly, even in the passages where his actual conversation wins nothing. The two numbers together are truer than the old one alone: a man received everywhere as an authority and fought to a standstill in most rooms — which is very close to the joke the novel itself tells about him.

Why interesting:

- His demotion (7th in the oldest reading, 20th of 31 now) is the cleanest case of reputation mistaken for scene-level performance; the current criteria separate the two registers and rank him in each honestly.
- His prestige rank rests on the most repeatable of witnessed displays — the ceremony that attends an ambassador — which the novel stages relentlessly and mostly ironically.
- His scene record (45-44-12) is nearly a perfect draw: the wielder of official language neither wins nor loses rooms, which is its own diplomatic verdict.

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

- Main authority concentration: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Secondary Guermantes reinforcement: `/projects/islt/fr-original/v3-p1`
- Late echo of rhetorical force: `/projects/islt/fr-original/v6-p3`

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

Top-ten in scene-level advantage (9th of 31) on genuinely winning scenes — and the belonging that once read as mildly negative now leans warmly upward, though still too rarely staged to rank.

la grand-mère holds 9th of 31 in scene-level advantage, on scenes that genuinely go her way (37 decided wins against 31 losses, positive passages outnumbering negative). The quiet correction in her profile is belonging: an older reading had it leaning mildly negative, but the family-boundary criterion — which counts the household's interior as a real inside — turned the direction warmly positive, though the evidence stays too thin to rank. Prestige leans upward too, on famously literal witnessed ground: the princesse de Luxembourg signifying her equality at Balbec. Where the novel measures her, she is strong; where it doesn't, it at least no longer misreads her.

Why interesting:

- Her belonging direction reversed under the family-boundary fix — the reading that counted the dining-room door and the goodnight kiss found the warmth the society-only criterion had missed.
- Her high advantage standing is matched by genuinely positive scenes, not survival on volume — rarer than it sounds in this book.
- Her prestige evidence includes the corpus's single most explicit staged-equality display: a princess signaling that a bourgeois grandmother is her peer.

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

- Main positive concentration, in Balbec: `/projects/islt/fr-original/v2-p2-noms-de-pays-le-pays`
- Early family-world footing: `/projects/islt/fr-original/v1-p1-combray`
- Guermantes-world counterweight: `/projects/islt/fr-original/v3-p1`

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

A winning record in the salon's crowded rooms, now weighed passage by passage: 14th of 31 in scene-level advantage, while his standing stays too thin to rank.

Cottard's scene record is genuinely winning (50 decided wins to 43 losses), and it places him 14th of 31 in scene-level advantage. His wins live in the Verdurin salon's dense scenes: the puns that land in the clan, the diagnoses that awe the faithful, the professorship that arrives. A crowded salon produces many pairwise comparisons from one passage, and his wins come disproportionately from the most crowded ones: an earlier count that weighed every pair in full lifted him to 5th, but weighed passage by passage his record is close to even — about 25 wins' worth to 24 — and he sits at the middle, where the novel's double portrait of him belongs. His prestige, the eminent-specialist reputation the later volumes assert, is witnessed in too few passages to rank. He remains a buffoon in texture — passages that mock him outnumber those that flatter — but the outcomes go his way, which is precisely Proust's joke about medicine.

Why interesting:

- His rank is the clearest case of crowd size mistaken for strength: his wins cluster in the fullest salons, and weighing each passage once moved him from 5th to 14th.
- The texture-versus-outcome split is his signature: the narration laughs at him constantly while the scenes keep handing him the win.
- The ranked scene-winner with an unranked prestige squares with the book's double portrait of the idiot who is also the great clinician.

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

- Primary negative concentration: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`
- Continued negative pressure: `/projects/islt/fr-original/v4-p2`
- Positive counterweight in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`

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

Second in prestige, behind only the duchesse, and in the bottom quarter of the scenes: the violinist commands the register the salons keep and loses more rooms than he wins.

Morel ranks 2nd of 14 in prestige, behind only the duchesse de Guermantes: his talent, and the protections it buys, place him above dukes, barons and Verdurins alike. Scene by scene the story runs the other way: he sits 25th of 31 in scene-level advantage, and the texture of those scenes is sharply negative — for every passage that lifts him, more than four cut him down. His belonging is still staged too rarely to rank. He remains the book's cleanest case of prestige without ground under it: the reputation ascends while the man, room by room, gives ground.

Why interesting:

- The clearest standing-versus-scene split in the measured cast: 2nd of 14 in prestige, 25th of 31 in scene-level advantage, with heavily negative scene texture.
- His prestige moves through protectors — Charlus above all — which makes his standing real and his position precarious at once.
- Belonging stays unranked: the salons prize the violinist and never quite seat the man.

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

- The Verdurin salon: talent under patronage: `/projects/islt/fr-original/v4-p2`
- The rupture with Charlus: `/projects/islt/fr-original/v5`
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

From bit-player to the duchesse's intimate: 8th of 31 in scene-level advantage, while her standing — high, but staged in too few passages — is back to an open question.

Rachel is staged across the whole arc of the novel — Saint-Loup's mistress, working actress, and at the end the celebrated artist whose reading empties la Berma's salon. The scenes certify her: 8th of 31 in scene-level advantage. Her prestige leans high, but its 24 comparisons come from only 10 passages, and counted passage by passage they are too few to rank; belonging is thinner still. The woman the theatre once priced at twenty francs ends the book winning the rooms she enters, even if the novel stages her standing too sparingly to certify it.

Why interesting:

- Her late triumph over la Berma at the bal de têtes is one of the sharpest single reversals the novel stages — celebrated in the same room that once priced her.
- Her scene record is ranked and strong (8th of 31 in advantage); her standing is the register where the evidence runs out, not where she falls short.
- For most of the book she was structurally invisible to measurement at all; the open reading of the full cast is what put her on the board.

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

- "Rachel quand du Seigneur": the theatre world's pricing: `/projects/islt/fr-original/v3-p1`
- The bal de têtes: her reading, la Berma's empty salon: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

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

Sixth of 31 in scene-level advantage — the highest family standing in the book — on the cleanest winning record of any measured figure: nine passages lift her for every one that cuts.

la mère du narrateur holds one of the strongest scene records in the measured cast: 6th of 31 in advantage, 34 decided wins against 16 losses, and a passage texture of nine positive to one negative — no one else the novel weighs comes out so consistently ahead. Her authority is entirely domestic and entirely effective: the goodnight-kiss economy, the moral verdicts the household defers to, the quiet management of the father. Prestige and belonging both remain too thin to rank, and belonging still leans mildly negative — the cost of being the boundary-keeper, the one who decides who is admitted to the child rather than the one admitted anywhere herself.

Why interesting:

- Her passage texture (+9/−1) is the cleanest positive of any measured figure — quiet domestic authority, near-perfectly effective.
- She ranks 6th of 31 in scene-level advantage, above every other member of the family and in a field that includes the promoted salon figures.
- Belonging still leans against her, a fine irony the numbers preserve: the guardian of the family's inside is rarely staged crossing into anyone else's.

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

- Foundational domestic context, and her strongest positive concentration: `/projects/islt/fr-original/v1-p1-combray`
- Largest positive presence, in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Continued positive presence in Guermantes-adjacent scenes: `/projects/islt/fr-original/v3-p2`

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

From 3rd in the oldest reading to 21st of 31 in scene-level advantage: the great author's standing was always more reputation than scene, and his prestige leans high but stays too thin to rank.

Bergotte is one of the measured cast's honest demotions: from 3rd in the oldest reading to 21st of 31 in scene-level advantage, his record still winning (25 decided wins to 21 losses) but no longer extraordinary. What the oldest reading counted as scene-dominance was largely the aura of the name — and the stricter criteria route that aura where it belongs, into prestige, where his lean is among the strongest measured but the staging stays too sparse to certify a rank. Belonging is nearly silent. He remains a strong positive presence where the book actually stages him; the correction is that the book stages him less than his fame made it feel.

Why interesting:

- His demotion mirrors Norpois's: the stricter criteria separate the witnessed aura of a reputation from the outcomes of actual scenes, and rank each honestly.
- His prestige lean is among the highest of any unranked figure — the fame is real; the novel simply conducts it offstage.
- His measured scenes still lean positive (25-21), a genuine but modest authority — closer to the dying man at the Vermeer than to the legend at the dinner table.

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

- Largest positive presence, in the Mme Swann circle: `/projects/islt/fr-original/v2-p1-autour-de-mme-swann`
- Strong positive concentration in Guermantes-world scenes: `/projects/islt/fr-original/v3-p1`
- Early positive footing: `/projects/islt/fr-original/v1-p1-combray`

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

Among the lowest ratings in the scenes, but on too few passages to rank: 40 comparisons from 23 passages — while his unrankable prestige lean is, absurdly and perfectly, among the highest in the book.

Legrandin's scene-level advantage rating is among the lowest anyone receives, on scenes that go against him more than three to one (8 decided wins, 28 losses) and passages that cut him seven to one. Counted passage by passage, though, his 40 comparisons come from only 23 passages, too few to certify a rank. And the joke only this book would build survives intact: his prestige lean, also too thinly staged to rank, is among the steepest upward of any unranked figure — because what the novel witnesses of him is precisely his performances of standing, the bows calibrated for aristocratic eyes, the syntax of the exquisite. The snob loses every real room while broadcasting, constantly and measurably, the standing he doesn't have.

Why interesting:

- His scene rating is near the floor, but the evidence behind it is thin: what the novel stages of him is concentrated in a few scenes, not spread across the book.
- His unranked prestige lean is among the highest measured, an artifact of what the novel stages about him: not standing, but the performance of standing.
- The pairing — floor of the scenes, ceiling of the pose — is the complete anatomy of snobbery in two numbers.

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

- Primary negative concentration, in Guermantes-adjacent society: `/projects/islt/fr-original/v3-p1`
- Early negative concentration: `/projects/islt/fr-original/v1-p1-combray`
- Final, sharpest negative return in diminished society: `/projects/islt/fr-original/v7-p4-le-bal-de-tetes`

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

Last of 31 in scene-level advantage — the certified floor of the scenes — while her standing, once ranked last, is now too thinly staged to rank at all.

Mme de Cambremer is measured severely: 31st of 31 in scene-level advantage, last of everyone the novel ranks there, on a record of 15 decided wins to 41 losses without a single positively-toned passage. Her prestige leans hard downward, but counted passage by passage its evidence is too thin to rank. The bottom rank is fitting rather than cruel: her position in the book is precisely the provincial grande dame whose standing every Parisian room quietly declines to honor — Charlus's engineered humiliation of her at la Raspelière is one of the corpus's textbook witnessed snubs. She anchors the floor of the scenes the way the duchesse anchors their ceiling, and the table needs both.

Why interesting:

- She is the certified last place in scene-level advantage — the floor of the ranked field.
- Not one of her measured passages is positively toned (0 for, 17 against): the harshest texture in the ranked cast.
- She confirms that severe loss doesn't require constant presence: the novel stages her rarely and beats her reliably.

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

- Primary negative concentration: `/projects/islt/fr-original/v4-p2`
- Sharpest negative intensity, in Guermantes-adjacent scenes: `/projects/islt/fr-original/v3-p1`
- Supporting negative evidence: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

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

A genuine reversal within a small set of scenes: strongly negative early, strongly positive late — the novel doesn't stage him often enough to rank the outcome, but the arc itself is among the most dramatic swings measured.

M. Vinteuil's appearances are few, but they trace one of the most dramatic arcs in the pilot set: strongly negative early, concentrated in Combray, and strongly positive later, in the La Prisonnière material, with his overall movement in scene-level advantage ending up mildly positive despite the rough start — a shape the enriched reading preserves intact, still too thinly staged to rank in any register. The swings are among the largest measured here — his individual scenes move more, on average, than almost any other figure's — but there are simply too few of them for the novel to certify a standing. Prestige leans mildly negative and belonging is essentially untouched, both far too thin to size. He is a genuine reversal case, not a stable positive one, even if the evidence stays too sparse to rank.

Why interesting:

- His scenes swing more dramatically than almost any other figure examined here — strongly negative early, strongly positive late — even though the total appearances are too few to rank the outcome.
- The reversal is chapter-shaped, not incidental: the early material (Combray) is where the negative concentrates, and the later material (La Prisonnière) is where the recovery happens.
- He is a genuine case of an arc rather than a static reading, best understood by following the sequence rather than a single number.

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

- Main late positive recovery: `/projects/islt/fr-original/v5`
- Early negative counterweight: `/projects/islt/fr-original/v1-p1-combray`
- Intermediate positive reinforcement: `/projects/islt/fr-original/v1-p2-un-amour-de-swann`

Notable units:

- The narration's verdict on him rises to the highest possible: an original of the rank of the greatest, whose work outranks everything previously known of him.: `/projects/islt/fr-original/v5#p-306`
- Vinteuil is savagely mocked after his death, reduced to a contemptuous epithet ('le vilain singe') in a scene the narrator frames as ritual desecration of his memory.: `/projects/islt/fr-original/v1-p1-combray#p-331`
- He is mocked and blamed by village gossip for tolerating his daughter's companion.: `/projects/islt/fr-original/v1-p1-combray#p-306`
