# Character Pages (scoring v2)

- Analysis version: `character_pages_v2`
- Scoring version: `scoring_v2`
- Source corpus summary: `scoring_v2_corpus_summary_v1`
- View: `name`
- Character count: `23`

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

He loses the scene and keeps the room: his individual scenes still run against him, yet he is first in belonging, 4th of 22 in standing, and held mid-table in advantage by the sheer certainty of the evidence.

The narrator is the novel's "I": nearly every scene passes through him, and scene by scene the scenes still go badly — 200 decided losses against 168 wins, with negative passages far outnumbering positive ones. Yet across the whole book his welcome never runs out: he ranks first in belonging, 4th of 22 in visible standing, and his advantage position (10th of 41) is less a verdict on his victories than on his measurability — no one in the book is weighed more often or more surely, and that certainty holds his floor where flashier figures wobble. The rooms keep receiving the man the scenes keep wounding; the split between lived defeat and durable acceptance remains the book's central irony made measurable.

Why interesting:

- His scene outcomes still lean against him — more decided losses than wins, negative passages nearly two to one — while all three of his standings sit in the upper half: the same passages, weighed differently.
- Because the whole novel passes through him, he is measured against more of the cast than any other figure, so his readings are the most certain in the book — his rating carries the narrowest uncertainty of anyone's.
- His suffering is local and his acceptance is cumulative: no single scene secures his place, and no single defeat costs it.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1513 ± 73 | 1439.9 | 10 of 41 | 209 | -0.201 | 0.6321 | 46/81/7/75 |
| prestige | 1633 ± 118 | 1515.5 | 4 of 22 | 209 | +0.061 | 0.1003 | 18/4/0/187 |
| inclusion | 1602 ± 100 | 1502.1 | 1 of 9 | 209 | +0.077 | 0.3553 | 36/26/1/146 |

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

Second in every register the novel measures — advantage, prestige, and belonging alike — the most complete dominance in the book, and in each case second to a different rival.

The duchesse now holds the same rank three times over: 2nd of 41 in scene-level advantage, 2nd of 22 in prestige, 2nd of 9 in belonging — no one else places in the top three of every register. Her scenes back it up: 225 decided wins against 92 losses, the wit crowning her far more often than it cuts her. And the trio of figures who edge her out reads like the novel's own commentary — Forcheville in the scenes, Morel in standing, the narrator in belonging: a brute, a protégé, and an observer, each beating the queen of the Faubourg at exactly one game. She is the book's measured establishment, and the measurements agree.

Why interesting:

- She is second in all three registers at once — the most complete high placement in the measured cast — and to a different character each time.
- Her scene record (225 wins, 92 losses across 354 decided comparisons) is the most lopsidedly victorious of any heavily-measured figure: the wit wins far more evenings than it loses.
- The old reading had her mid-table in advantage; the witnessed-standing criteria found the deference the salons actually pay her.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1594 ± 80 | 1514.1 | 2 of 41 | 183 | +0.049 | 0.4899 | 62/41/8/72 |
| prestige | 1683 ± 99 | 1583.8 | 2 of 22 | 183 | +0.216 | 0.2704 | 38/4/0/141 |
| inclusion | 1619 ± 159 | 1460.2 | 2 of 9 | 183 | 0.0 | 0.0 | 0/0/0/183 |

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
- Archetype signs: `advantage -1, prestige -1, inclusion -1`
- Pattern: `broad_negative`

One of the most heavily measured men in the novel, and measured losing: below the middle in scene-level advantage, mid-table in a prestige field he once led from the shadows, near the bottom in belonging.

Swann is staged constantly — 386 decided comparisons in advantage alone, more than anyone but the narrator — and the scenes go against him: 197 losses to 144 wins, with negative passages far outnumbering positive. His advantage standing sits below the middle (26th of 41). Prestige, newly measurable for him, lands mid-table (12th of 22) — a sobering number for the man Combray never realized dined with princes, because the novel stages his standing mostly in decline, through the marriage that costs him the rooms he owned. Belonging is his cleanest loss: 8th of 9, the elegant man who ends the book steered around as an embarrassment.

Why interesting:

- He is among the most heavily measured figures in the book, so his negative readings carry unusual evidentiary weight — this is not a small-sample verdict.
- His prestige rank (12th of 22) captures the tragedy structurally: the novel stages his standing almost entirely on its way down, after the marriage, so the measured Swann is the diminished one.
- Belonging near the bottom (8th of 9) squares with the book's late cruelty: the name unspeakable in the Guermantes household his person once graced.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1458 ± 88 | 1369.2 | 26 of 41 | 177 | -0.317 | 0.7741 | 46/83/3/45 |
| prestige | 1494 ± 129 | 1364.8 | 12 of 22 | 177 | +0.024 | 0.1659 | 15/15/2/145 |
| inclusion | 1346 ± 122 | 1224.0 | 8 of 9 | 177 | -0.12 | 0.1975 | 8/20/0/149 |

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
- Archetype signs: `advantage +1, prestige -1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Ranked in all three registers — solidly mid-table in scene-level advantage and belonging, but in the lower half of the prestige field his name would predict he'd own.

Saint-Loup remains one of the few characters the novel stages enough to rank in advantage, prestige, and belonging all at once. His footing is mid-table in scene-level advantage (14th of 41, wins and losses nearly even across 234 decided comparisons) and solid in belonging (5th of 9). But in prestige — the register his aristocratic bearing would predict he'd own — he ranks 16th of 22, the lower third of the measured field, his standing resting on presence more than deference. He is accepted more than he is deferred to, a Guermantes who spends the name rather than banks it.

Why interesting:

- He is one of the few figures ranked in all three lenses at once, a completeness the novel affords barely a handful of its cast.
- His prestige position inverts what his rank and bearing would suggest: 16th of the 22 characters the novel sizes there, behind Rachel — his own mistress — and Odette.
- His advantage record is almost perfectly even (105 wins, 108 losses across 234 decided comparisons): breadth of presence, not a run of triumphs, is what holds his place.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1514 ± 84 | 1429.8 | 14 of 41 | 138 | -0.132 | 0.6397 | 37/57/3/41 |
| prestige | 1476 ± 125 | 1351.3 | 16 of 22 | 138 | +0.047 | 0.1162 | 11/7/0/120 |
| inclusion | 1486 ± 195 | 1291.3 | 5 of 9 | 138 | -0.024 | 0.0235 | 0/3/0/135 |

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

Volatile in the scenes and newly ranked in standing (13th of 22) — while belonging, once her clearest loss, has become her open question: the stricter reading un-ranked it.

Albertine's scenes remain among the most conflicted measured — wins and losses nearly even (80 to 84), with more explicitly mixed passages than most of the cast — and her advantage standing holds mid-table, 13th of 41. Prestige, unmeasurable before, now ranks her 13th of 22: the captive girl carries more certified standing than the duc de Guermantes. The starkest change is belonging: the old reading ranked her dead last, but under the stricter boundary criteria the sequestration chapters stage fewer true boundary events than the old reading counted, and what remains is too thin to rank. Her exclusion was real, but much of it was the narrator's arrangement rather than the world's verdict — and the measurement now respects that difference.

Why interesting:

- Her belonging reading changed more than anyone's: from dead last to unranked, because the boundary criteria distinguish being shut in by one man from being shut out by the world.
- She is newly ranked in prestige (13th of 22) — the novel does stage her standing, through the elegance the narrator cultivates and the world appraises.
- Her scene volatility persists in the new reading: near-even outcomes with an unusual share of explicitly mixed passages, a genuine internal split rather than a slide.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1509 ± 79 | 1430.1 | 13 of 41 | 126 | -0.203 | 0.7437 | 35/60/5/26 |
| prestige | 1549 ± 185 | 1364.3 | 13 of 22 | 126 | +0.01 | 0.0469 | 4/2/0/120 |
| inclusion | 1723 ± 245 | 1477.5 | insufficient evidence | 126 | -0.013 | 0.0618 | 3/3/0/120 |

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
- Archetype signs: `advantage -1, prestige +1, inclusion -1`
- Pattern: `prestige_positive_inclusion_negative`

Ranked in all three registers, and highest where the old reading couldn't see her: 3rd of 22 in prestige — the demi-mondaine ends the book outranking most of the Faubourg.

Odette is now one of the few figures the novel ranks in every register, and her strongest is the one the evidence used to leave open: prestige, where she stands 3rd of 22, behind only Morel and the duchesse de Guermantes. Her scene-level advantage holds mid-table (20th of 41, wins and losses nearly even across 248 decided comparisons), and belonging sits mid-low (6th of 9). The shape is the novel's longest social climb made measurable: the woman the salons refused to receive ends with a certified standing above most of the people who refused her.

Why interesting:

- Her prestige standing — 3rd of 22 — was invisible to the old reading, which had too little staged evidence to rank her there at all; the enriched reading certifies the climb.
- The three registers disagree about her in the most Proustian way: standing high, scenes even, belonging modest — received as a name long before she is received as a person.
- In scene-level advantage her record is nearly balanced (112 wins, 107 losses), steady unglamorous footing rather than a dramatic arc.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1493 ± 98 | 1394.7 | 20 of 41 | 124 | -0.081 | 0.5035 | 26/40/2/56 |
| prestige | 1686 ± 124 | 1561.9 | 3 of 22 | 124 | +0.107 | 0.1687 | 13/4/1/106 |
| inclusion | 1402 ± 153 | 1248.7 | 6 of 9 | 124 | -0.094 | 0.1066 | 1/8/0/115 |

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

The stricter reading restores the baron: top-quarter in scene-level advantage, 5th of 22 in prestige, and newly ranked 3rd of 9 in belonging — a great position, measured on its way to a great fall.

The enriched reading transforms Charlus's profile more than almost anyone's. Where the old evidence left him last in prestige and unrankable in belonging, the witnessed-standing and boundary criteria now certify what the novel actually stages for most of its length: a man of enormous measured position — 10th of 41 in scene-level advantage, 5th of 22 in prestige, 3rd of 9 in belonging. The fall is still in the data, but it lives in the trajectory rather than the rank: the wartime chapters and the Verdurin expulsion drag his late ratings down from a summit the earlier volumes spent thousands of pages building. He is the book's great instance of position as altitude — measured high precisely so the descent can be measured too.

Why interesting:

- All three of his readings improved under stricter criteria — evidence that his old low ranks were artifacts of unwitnessed-judgment noise, not of the text.
- He is now ranked in all three registers, one of the few, with belonging 3rd of 9 — the clubbable baron the novel installs everywhere before it evicts him.
- His fall is a trajectory fact, not a rank fact: the standing is high across the book and collapses at its end, which is precisely the shape the novel wrote.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1514 ± 74 | 1439.9 | 10 of 41 | 110 | -0.256 | 0.7058 | 28/46/5/31 |
| prestige | 1591 ± 93 | 1497.9 | 5 of 22 | 110 | +0.032 | 0.269 | 16/11/1/82 |
| inclusion | 1571 ± 146 | 1424.3 | 3 of 9 | 110 | +0.011 | 0.0705 | 3/2/0/105 |

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
- Pattern: `advantage_reversed_high_title`

The title now earns a rank — 14th of 22 in prestige — but the rooms still go against him: 31st of 41 in scene-level advantage, with the book's most lopsided losing texture among its great names.

The enriched reading finally measures the duc's title: he ranks 14th of 22 in prestige, a real if middling standing built on the ceremony that attends a Guermantes. It does not rescue his scenes. In scene-level advantage he sits 31st of 41, losing 126 decided comparisons against 75 wins, with passages that cut him outnumbering those that lift him ten to one — the Jockey Club defeat, the deceptions endured, the wife's wit at his expense. His belonging stays too thin to rank. The gap between the two measured registers is now his profile: the name commands deference the man cannot hold onto in any actual room.

Why interesting:

- His prestige and advantage ranks now quantify the book's running joke about him: 14th of 22 as a name, 31st of 41 as a presence.
- His negative scene texture is the most lopsided of the great aristocrats (5 positive passages against 53) — the comedy of the duc is structural, not incidental.
- Against his wife the comparison is total: she is 2nd in every register; he cracks the top half of none.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1435 ± 87 | 1347.7 | 31 of 41 | 97 | -0.507 | 0.5645 | 5/53/4/35 |
| prestige | 1496 ± 138 | 1357.3 | 14 of 22 | 97 | -0.012 | 0.0614 | 2/2/0/93 |
| inclusion | 1562 ± 202 | 1360.0 | insufficient evidence | 97 | 0.0 | 0.0 | 0/0/0/97 |

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

Ranked in all three registers, and the three disagree completely: 6th of 22 in prestige, mid-table in the scenes, dead last of 9 in belonging — the hostess the book crowns and never seats.

Mme Verdurin remains one of the few figures ranked in advantage, prestige, and belonging at once, and the enriched reading sharpens her contradiction to its final form. Prestige: 6th of 22, real certified standing, ending as it does in the princesse de Guermantes title. Advantage: 12th of 41, though the texture is brutal — passages that lift her are outnumbered eight to one by passages that cut. Belonging: dead last, 9th of 9. The woman who built the century's most exclusive interior is, by the book's own staging, never securely inside anything — bypassed at her own soirées, mocked in her own title. The clan was a fortress built by someone the walls never protected.

Why interesting:

- Her three ranks tell three different stories — top-third standing, mid-table scenes, last-place belonging — the widest three-way disagreement in the measured cast.
- Her last place in belonging is earned at her own parties: the corpus's adjudicated divergences include guests bypassing her as hostess while a queen rescues her, and the Faubourg mocking her as princesse.
- Her prestige is the book's great manufactured standing — built, purchased, and finally titled — and the numbers certify it while refusing it warmth.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1535 ± 96 | 1439.3 | 12 of 41 | 78 | -0.336 | 0.4254 | 4/33/0/41 |
| prestige | 1574 ± 105 | 1468.6 | 6 of 22 | 78 | +0.129 | 0.2362 | 13/4/0/61 |
| inclusion | 1334 ± 156 | 1178.0 | 9 of 9 | 78 | -0.055 | 0.0549 | 0/3/0/75 |

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

The quiet riser of the stricter reading: 6th of 41 in scene-level advantage and newly ranked 10th of 22 in prestige — the salonnière the old evidence mistook for background.

Mme de Villeparisis is one of the enriched reading's clearest promotions: from the exact middle of the old table to 6th of 41 in scene-level advantage (72 decided wins against 38 losses), with a new ranked standing in prestige (10th of 22) besides. The rise is not mysterious — her matinées are among the book's most heavily staged social machinery, and the witnessed-standing criteria credit the hostess who runs the room rather than only the guests who shine in it. Belonging alone stays too thin to rank, the famous ambiguity of her position — received by everyone, placed by no one — surviving as an honestly open question.

Why interesting:

- Her advantage rank jumped from the median to 6th of 41 — the stricter criteria found the authority her matinées actually exercise.
- She is the foundation corpus's one adjudicated case of prestige-without-belonging at Balbec, and the enriched reading preserves exactly that shape: ranked standing, unrankable belonging.
- Her win rate (72 to 38) is among the strongest of any non-family figure — quiet dominance the old reading's thin evidence could not see.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1580 ± 118 | 1461.3 | 6 of 41 | 73 | -0.077 | 0.3693 | 14/19/2/38 |
| prestige | 1527 ± 131 | 1395.9 | 10 of 22 | 73 | -0.016 | 0.2053 | 7/9/0/57 |
| inclusion | 1524 ± 208 | 1316.4 | insufficient evidence | 73 | 0.0 | 0.0 | 0/0/0/73 |

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

Near the bottom everywhere the room can see him: 36th of 41 in the scenes, lower-third in prestige (18th of 22), 7th of 9 in belonging — the old reading's flattering prestige rank was an artifact, and it's gone.

Bloch's advantage reading remains among the harshest measured — 36th of 41, losses outnumbering wins better than three to one (100 to 31), negative passages five to one. What changed is prestige: the old, tiny field ranked him a startling 3rd of 8; the enriched field of 22 places him 18th, which is what the text has staged all along — the gaffes, the wrong clothes, the name changed to Jacques du Rozier. Belonging completes the picture at 7th of 9. His late success as a dramatist is real but arrives mostly offstage; the rooms the novel actually stages are the ones that cost him. The consistency across all three registers is now the point: the book's most relentless study of the socially unabsorbed.

Why interesting:

- His old 3rd-of-8 prestige rank was a small-field artifact that the enriched reading corrects to 18th of 22 — a demotion that brings the number into line with every scene the novel wrote him.
- His advantage record (31-100-15) is the most lopsided of any heavily measured figure — being cut down in the room is his structural role.
- All three registers now agree on him, which they do for almost no one else — and their agreement is itself the reading.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1306 ± 110 | 1196.1 | 36 of 41 | 64 | -0.692 | 0.8975 | 8/43/2/11 |
| prestige | 1482 ± 156 | 1325.7 | 18 of 22 | 64 | -0.046 | 0.0934 | 2/5/0/57 |
| inclusion | 1409 ± 172 | 1236.7 | 7 of 9 | 64 | -0.152 | 0.2444 | 3/10/0/51 |

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
- Archetype signs: `advantage +1, prestige +1, inclusion -1`
- Pattern: `advantage_high_durable`

Top-ten in the scenes she was born to win — 7th of 41 in advantage — and newly ranked in prestige (19th of 22): the kitchen has a standing the salons must now be measured against.

Françoise no longer holds the top rank the sparser reading gave her, but her position remains formidable and better founded: 7th of 41 in scene-level advantage, on a genuinely winning record (51 decided wins to 38 losses), amid a field that now includes the salon figures the stricter criteria promoted past her. And she gains something the old reading could not give her: a ranked prestige standing (19th of 22) — the deference of footmen, doctors, and households is witnessed standing too, and the enriched reading counts it. Belonging stays unranked, the servant's position at the family's center and margin at once remaining, fittingly, unmeasurable.

Why interesting:

- Her old first-place advantage rank was partly a small-field artifact; her new 7th of 41, on a real winning record, is the sturdier claim.
- She is ranked in prestige at all — a servant measured in the register built for duchesses — because the witnessed-standing criterion is blind to class, exactly as the novel's own attention is.
- Her scenes stay nearly even (51-38-11): durable footing, not a hot streak, is what the ranking has always reflected.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1575 ± 114 | 1461.2 | 7 of 41 | 61 | +0.086 | 0.5864 | 20/17/0/24 |
| prestige | 1514 ± 198 | 1315.7 | 19 of 22 | 61 | +0.052 | 0.0516 | 3/0/0/58 |
| inclusion | 1322 ± 370 | 951.9 | insufficient evidence | 61 | -0.013 | 0.0128 | 0/1/0/60 |

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

Ranked in all three registers and strongest in belonging (4th of 9) — the girl who changes names twice and lands, each time, further inside.

Gilberte remains ranked everywhere the novel measures: 4th of 9 in belonging, 8th of 22 in prestige, 17th of 41 in scene-level advantage, her scenes themselves nearly even (64 decided wins, 61 losses). Belonging is still her strongest register, fitting for the book's great study in absorbed identity — Swann's daughter becoming Mlle de Forcheville becoming the marquise de Saint-Loup, each name a door that opens on a room the last one couldn't enter. The corpus catches the mechanism directly: her walk into the Guermantes salon under her new name is one of its cleanest boundary events.

Why interesting:

- Her belonging rank rests on the novel's most explicit boundary machinery: the same salon that would not receive Mlle Swann receives Mlle de Forcheville.
- She is one of the few characters ranked in all three registers at once, with the strengths running opposite to her father's — his belonging collapses as hers compounds.
- Her scene record is almost perfectly even (64-61): she never dominates a room, and never needs to; the names do the work.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1505 ± 94 | 1410.7 | 17 of 41 | 57 | -0.028 | 0.4766 | 12/18/0/27 |
| prestige | 1539 ± 124 | 1415.4 | 8 of 22 | 57 | +0.085 | 0.174 | 6/2/0/49 |
| inclusion | 1554 ± 164 | 1389.9 | 4 of 9 | 57 | +0.05 | 0.0712 | 2/1/1/53 |

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

The diplomat demoted by better evidence: from 7th to 24th of 41 in scene-level advantage, though prestige now ranks him (11th of 22) — the authority was always reputation more than performance.

Norpois is the enriched reading's clearest deflation. The old evidence placed him 7th in scene-level advantage; the stricter criteria place him 24th of 41, his scenes an almost perfect draw (45 decided wins, 44 losses). What he gains instead is a ranked prestige standing, 11th of 22 — because the deference paid to an ambassador is witnessed constantly, even in the passages where his actual conversation wins nothing. The two numbers together are truer than the old one alone: a man received everywhere as an authority and fought to a standstill in most rooms — which is very close to the joke the novel itself tells about him.

Why interesting:

- His demotion (7th to 24th) is the cleanest case of the old reading mistaking reputation for scene-level performance; the new criteria separate the two registers and rank him in each honestly.
- His prestige rank rests on the most repeatable of witnessed displays — the ceremony that attends an ambassador — which the novel stages relentlessly and mostly ironically.
- His scene record (45-44-12) is nearly a perfect draw: the wielder of official language neither wins nor loses rooms, which is its own diplomatic verdict.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1503 ± 121 | 1381.9 | 24 of 41 | 54 | -0.069 | 0.4894 | 16/17/1/20 |
| prestige | 1569 ± 177 | 1392.2 | 11 of 22 | 54 | +0.101 | 0.1274 | 6/1/0/47 |
| inclusion | 1594 ± 225 | 1369.1 | insufficient evidence | 54 | +0.013 | 0.0133 | 1/0/0/53 |

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

Top-ten in scene-level advantage (8th of 41) on genuinely winning scenes — and the belonging that once read as mildly negative now leans warmly upward, though still too rarely staged to rank.

la grand-mère holds 8th of 41 in scene-level advantage, on scenes that genuinely go her way (37 decided wins against 31 losses, positive passages outnumbering negative). The quiet correction in her profile is belonging: the old reading had it leaning mildly negative, but the family-boundary criterion — which counts the household's interior as a real inside — turned the direction warmly positive, though the evidence stays too thin to rank. Prestige leans upward too, on famously literal witnessed ground: the princesse de Luxembourg signifying her equality at Balbec. Where the novel measures her, she is strong; where it doesn't, it at least no longer misreads her.

Why interesting:

- Her belonging direction reversed under the family-boundary fix — the reading that counted the dining-room door and the goodnight kiss found the warmth the society-only criterion had missed.
- Her high advantage standing is matched by genuinely positive scenes, not survival on volume — rarer than it sounds in this book.
- Her prestige evidence includes the corpus's single most explicit staged-equality display: a princess signaling that a bourgeois grandmother is her peer.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1578 ± 129 | 1448.7 | 8 of 41 | 48 | +0.177 | 0.6675 | 17/14/0/17 |
| prestige | 1632 ± 217 | 1415.0 | insufficient evidence | 48 | +0.053 | 0.1185 | 3/2/0/43 |
| inclusion | 1755 ± 244 | 1511.5 | insufficient evidence | 48 | +0.047 | 0.0792 | 3/1/0/44 |

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

The steepest riser in the stricter reading: from the lower third to 5th of 41 in scene-level advantage, with a new ranked standing in prestige (15th of 22) — the buffoon was winning his rooms all along.

Cottard is the enriched reading's biggest surprise: 5th of 41 in scene-level advantage, up from the old lower third, on a genuinely winning record (50 decided wins to 43 losses). The mechanism is the opened event budget — in the Verdurin salon's dense scenes, the old two-event ceiling had room for the hosts and the victims, and Cottard's small, constant victories (the puns that land in the clan, the diagnoses that awe the faithful, the professorship that arrives) fell off the sheet. Counted, they compound. Prestige now ranks him too (15th of 22), the eminent-specialist reputation the later volumes keep asserting. He remains a buffoon in texture — passages that mock him outnumber those that flatter — but the outcomes go his way, which is precisely Proust's joke about medicine.

Why interesting:

- His rise from 25th to 5th is the single largest promotion of the enrichment pass — a coverage artifact corrected, not a reinterpretation: his wins were always in the text, below the old event ceiling.
- The texture-versus-outcome split is his signature: the narration laughs at him constantly while the scenes keep handing him the win.
- His two ranked standings — scene-winner, mid-table name — square exactly with the book's double portrait of the idiot who is also the great clinician.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1582 ± 119 | 1463.9 | 5 of 41 | 37 | -0.165 | 0.7129 | 9/19/1/8 |
| prestige | 1537 ± 183 | 1353.8 | 15 of 22 | 37 | +0.057 | 0.0568 | 3/0/0/34 |
| inclusion | 1410 ± 246 | 1163.5 | insufficient evidence | 37 | 0.0 | 0.0 | 0/0/0/37 |

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
- Archetype signs: `advantage -1, prestige +1, inclusion +1`
- Pattern: `prestige_first_scene_negative`

First in standing across a field of twenty-two, and below the middle of the scenes: the violinist commands the register the salons keep and loses more rooms than he wins.

Morel holds the highest standing of the twenty-two characters the novel stages often enough to rank in prestige — his talent, and the protections it buys, place him above dukes and duchesses alike. Scene by scene the story runs the other way: he sits below the middle in scene-level advantage (21st of 41), and the texture of those scenes is sharply negative — for every passage that lifts him, more than four cut him down. His belonging is still staged too rarely to rank. He remains the book's cleanest case of prestige without ground under it: the reputation ascends while the man, room by room, gives ground.

Why interesting:

- The clearest standing-versus-scene split in the measured cast: first of 22 in prestige, below the middle of 41 in scene-level advantage, with heavily negative scene texture.
- His prestige moves through protectors — Charlus above all — which makes his standing real and his position precarious at once.
- Belonging stays unranked: the salons prize the violinist and never quite seat the man.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1488 ± 99 | 1388.3 | 21 of 41 | 35 | -0.718 | 0.8773 | 5/22/0/8 |
| prestige | 1773 ± 164 | 1608.1 | 1 of 22 | 35 | +0.206 | 0.2063 | 6/0/0/29 |
| inclusion | 1513 ± 219 | 1293.4 | insufficient evidence | 35 | -0.041 | 0.0414 | 0/2/0/33 |

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
- Pattern: `standing_rises_belonging_thin`

From bit-player to the duchesse's intimate — and now the numbers certify it: ranked 7th of 22 in prestige, top-quarter in the scenes, with only belonging still too thin to call.

Rachel is staged across the whole arc of the novel — Saint-Loup's mistress, working actress, and at the end the celebrated artist whose reading empties la Berma's salon. The new reading ranks her in two registers at once: 9th of 41 in scene-level advantage and 7th of 22 in prestige, the steep late climb no longer a lean but a certified standing. Belonging remains her open question — still staged too rarely to rank — though what the novel now weighs there no longer points away. The woman the theatre once priced at twenty francs ends the book measurable beside duchesses.

Why interesting:

- Her late triumph over la Berma at the bal de têtes is one of the sharpest single reversals the novel stages — celebrated in the same room that once priced her.
- The stricter reading promoted her: what was a provisional upward lean in standing is now a ranked 7th of 22, one of the few figures whose position strengthened as the evidence hardened.
- For most of the book she was structurally invisible to measurement at all; the open reading of the full cast is what put her on the board.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1579 ± 132 | 1446.3 | 9 of 41 | 29 | -0.216 | 0.6092 | 7/14/0/8 |
| prestige | 1640 ± 183 | 1456.7 | 7 of 22 | 29 | +0.041 | 0.2003 | 2/2/0/25 |
| inclusion | 1713 ± 418 | 1294.6 | insufficient evidence | 29 | +0.025 | 0.0248 | 1/0/0/28 |

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

Fourth of 41 in scene-level advantage — the highest family standing in the book — on the cleanest winning record of any measured figure: nine passages lift her for every one that cuts.

la mère du narrateur holds the strongest scene record in the measured cast: 4th of 41 in advantage, 34 decided wins against 16 losses, and a passage texture of nine positive to one negative — no one else the novel weighs comes out so consistently ahead. Her authority is entirely domestic and entirely effective: the goodnight-kiss economy, the moral verdicts the household defers to, the quiet management of the father. Prestige and belonging both remain too thin to rank, and belonging still leans mildly negative — the cost of being the boundary-keeper, the one who decides who is admitted to the child rather than the one admitted anywhere herself.

Why interesting:

- Her passage texture (+9/−1) is the cleanest positive of any measured figure — quiet domestic authority, near-perfectly effective.
- She climbed to 4th of 41 in a field that now includes the promoted salon figures — family standing holding its own against the drawing rooms.
- Belonging still leans against her, a fine irony the numbers preserve: the guardian of the family's inside is rarely staged crossing into anyone else's.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1620 ± 146 | 1473.8 | 4 of 41 | 28 | +0.273 | 0.3177 | 9/1/0/18 |
| prestige | 1731 ± 266 | 1464.5 | insufficient evidence | 28 | +0.005 | 0.0482 | 1/1/0/26 |
| inclusion | 1392 ± 228 | 1163.5 | insufficient evidence | 28 | -0.108 | 0.1618 | 1/3/0/24 |

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

From 3rd to 19th of 41 in scene-level advantage: the great author's standing was always more reputation than scene, and the stricter reading files the reputation under prestige — where it leans high but stays too thin to rank.

Bergotte is one of the enriched reading's honest demotions: from 3rd to 19th of 41 in scene-level advantage, his record still winning (25 decided wins to 21 losses) but no longer extraordinary. What the old reading counted as scene-dominance was largely the aura of the name — and the stricter criteria route that aura where it belongs, into prestige, where his lean is among the strongest measured but the staging stays too sparse to certify a rank. Belonging is nearly silent. He remains a strong positive presence where the book actually stages him; the correction is that the book stages him less than his fame made it feel.

Why interesting:

- His demotion mirrors Norpois's: the stricter criteria separate the witnessed aura of a reputation from the outcomes of actual scenes, and rank each honestly.
- His prestige lean is among the highest of any unranked figure — the fame is real; the novel simply conducts it offstage.
- His measured scenes still lean positive (25-21), a genuine but modest authority — closer to the dying man at the Vermeer than to the legend at the dinner table.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1562 ± 157 | 1405.2 | 19 of 41 | 27 | +0.127 | 0.8517 | 11/8/0/8 |
| prestige | 1777 ± 469 | 1308.1 | insufficient evidence | 27 | +0.037 | 0.1467 | 2/2/0/23 |
| inclusion | 1266 ± 491 | 774.7 | insufficient evidence | 27 | 0.0 | 0.0 | 0/0/0/27 |

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

Now ranked, and ranked where he always belonged: 39th of 41 in scene-level advantage, near the very bottom — while his unrankable prestige lean is, absurdly and perfectly, the highest in the book.

The enriched reading finally certifies Legrandin: 39th of 41 in scene-level advantage, near the very bottom of everyone the novel measures, on scenes that go against him seven to one (8 decided wins, 28 losses). And it adds the joke only this book would build: his prestige lean, still too thinly staged to rank, is the steepest upward of any unranked figure — because what the novel witnesses of him is precisely his performances of standing, the bows calibrated for aristocratic eyes, the syntax of the exquisite. The snob loses every real room while broadcasting, constantly and measurably, the standing he doesn't have.

Why interesting:

- He graduated from unrankable to a certified place near the very bottom — the stricter reading's evidence was enough to make his losses official.
- His unranked prestige lean is the highest measured, an artifact of what the novel stages about him: not standing, but the performance of standing.
- The pairing — floor of the scenes, ceiling of the pose — is the complete anatomy of snobbery in two numbers.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1294 ± 166 | 1128.5 | 39 of 41 | 23 | -0.547 | 0.7439 | 2/15/0/6 |
| prestige | 1887 ± 241 | 1646.3 | insufficient evidence | 23 | +0.013 | 0.1435 | 1/2/0/20 |
| inclusion | 1387 ± 495 | 891.7 | insufficient evidence | 23 | 0.0 | 0.0 | 0/0/0/23 |

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

Near the bottom in both registers the novel now measures her in: 34th of 41 in scene-level advantage and last — 22nd of 22 — in prestige, the certified floor of the standing table.

Mme de Cambremer is now measured twice, and severely both times: 34th of 41 in scene-level advantage, on a record of 15 decided wins to 41 losses without a single positively-toned passage, and dead last of the 22 characters ranked in prestige. The bottom rank is fitting rather than cruel: her position in the book is precisely the provincial grande dame whose standing every Parisian room quietly declines to honor — Charlus's engineered humiliation of her at la Raspelière is one of the corpus's textbook witnessed snubs. She anchors the floor of the prestige table the way the duchesse anchors its ceiling, and the table needs both.

Why interesting:

- She is the certified last place in prestige — the enriched field's floor — where the old reading could only call her lean negative.
- Not one of her measured passages is positively toned (0 for, 17 against): the harshest texture in the ranked cast.
- She confirms that severe standing loss doesn't require constant presence: the novel stages her rarely and beats her reliably.

| Lens | Standing | Conservative | Rank | Appearances | Mean m | Mean abs m | +/-/mixed/neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| advantage | 1368 ± 140 | 1228.0 | 34 of 41 | 22 | -0.831 | 0.8309 | 0/17/0/5 |
| prestige | 1442 ± 196 | 1246.5 | 22 of 22 | 22 | -0.064 | 0.0636 | 0/2/0/20 |
| inclusion | 1316 ± 245 | 1071.5 | insufficient evidence | 22 | -0.146 | 0.2082 | 1/3/0/18 |

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
| advantage | 1658 ± 217 | 1441.0 | insufficient evidence | 9 | +0.02 | 1.1667 | 4/3/1/1 |
| prestige | 1715 ± 298 | 1417.3 | insufficient evidence | 9 | +0.078 | 0.3 | 1/1/0/7 |
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
