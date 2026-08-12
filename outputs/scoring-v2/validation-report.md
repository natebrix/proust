# Scoring v2 validation report (staged, pre-adoption)

Corpus: foundation, 34 runs, 963 reviewed units, 963 narrative time points. Comparisons per lens: {'advantage': 3475, 'prestige': 1184, 'inclusion': 531}.

Formula: `proust/scoring_v2.py`, exactly as specified in `proust/docs/scoring_v2_design.md`. Ratings: weighted WHR (`proust/whr.py`), smoothed and filtered, on the `cumulative_unit_index` narrative axis. Everything here is staged under `outputs/scoring-v2/`; adoption is a separate reviewed decision.

w2 selected per lens/view: advantage/name = 15, advantage/person = 15, inclusion/name = 5, inclusion/person = 5, prestige/name = 60, prestige/person = 60

## 1. Lens orthogonality

The design predicts cross-lens rating correlations should FALL against v1: v1's weight tables blended every dimension into every lens, v2's projection partitions them.

| pair | v2 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.0980 (n=288) |
| advantage vs inclusion | +0.2193 (n=288) |
| prestige vs inclusion | +0.0670 (n=288) |
| **mean abs** | **0.1281** |

| pair | v1 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.9852 (n=288) |
| advantage vs inclusion | +0.9897 (n=288) |
| prestige vs inclusion | +0.9736 (n=288) |
| **mean abs** | **0.9828** |

| pair | v2 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | -0.6667 (n=8) |
| advantage vs inclusion | -0.6500 (n=9) |
| prestige vs inclusion | +0.2000 (n=6) |
| **mean abs** | **0.5056** |

| pair | v1 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.9847 (n=91) |
| advantage vs inclusion | +0.9804 (n=91) |
| prestige vs inclusion | +0.9627 (n=91) |
| **mean abs** | **0.9759** |

**Verdict**: mean |rho| 0.983 (v1) -> 0.128 (v2): prediction held.

## 2. Bootstrap stability

50 unit-level resamples with replacement, both formulas scored on the same drawn corpora; ranks are taken over the characters both formulas rate non-provisionally on the full corpus. Lower rank standard deviation is more stable.

| lens | field | v2 mean sd | v1 mean sd | v2 median sd | v1 median sd | v2 non-prov | v1 non-prov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 35 | 5.912 | 5.799 | 6.074 | 5.835 | 35 | 91 |
| prestige | 8 | 1.643 | 1.808 | 1.539 | 1.893 | 8 | 91 |
| inclusion | 9 | 1.57 | 2.298 | 1.712 | 2.337 | 9 | 91 |

### 2b. Frequency confounding

The design's fourth principle is that frequency must not masquerade as strength. Ratings are no longer sums, so nothing accumulates with appearances -- but the standings rank by rating MINUS band, and a band narrows with evidence. Where a lens's ratings are tightly packed and its bands are not, the ranking is mostly a comparison count. Spearman rho against comparison count, over each formula's own non-provisional set:

| lens | formula | conservative vs count | rating vs count | band vs count | rating spread | band spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| advantage | v2 | 0.122 | -0.199 | -0.891 | 424.2 | 110.6 |
| advantage | v1 | 0.44 | -0.121 | -0.925 | 607.4 | 132.3 |
| prestige | v2 | -0.524 | -0.595 | -0.548 | 370.4 | 35.4 |
| prestige | v1 | 0.447 | -0.088 | -0.927 | 554.1 | 132.6 |
| inclusion | v2 | 0.05 | -0.283 | -0.883 | 257.8 | 83.1 |
| inclusion | v1 | 0.434 | -0.17 | -0.927 | 605.5 | 133.4 |

## 3. Predictive sanity

One-step-ahead over the v2 comparisons in narrative order. Every system is scored UNWEIGHTED (one comparison, one prediction): ELO and Glicko-2 have no notion of a game weight, so a weighted loss would not be comparable to theirs. The WHR fits themselves DO use the weights. Cross-formula comparison against v1's own numbers is meaningless here -- the comparisons differ -- so only the within-v2 ordering is informative.

| lens/view | system | log-loss | Brier | comparisons |
| --- | --- | ---: | ---: | ---: |
| advantage/name | whr_filtered | 0.712944 | 0.255273 | 3475 |
| advantage/name | whr_filtered_deflated | 0.700120 | 0.251245 | 3475 |
| advantage/name | elo_sequential | 0.639731 | 0.224781 | 3475 |
| advantage/name | elo_unit_frozen | 0.690754 | 0.247897 | 3475 |
| advantage/name | glicko2_chapter_period | 0.750061 | 0.268868 | 3475 |
| advantage/person | whr_filtered | 0.712578 | 0.255024 | 3474 |
| advantage/person | whr_filtered_deflated | 0.699967 | 0.251108 | 3474 |
| advantage/person | elo_sequential | 0.639673 | 0.224759 | 3474 |
| advantage/person | elo_unit_frozen | 0.690789 | 0.247900 | 3474 |
| advantage/person | glicko2_chapter_period | 0.776715 | 0.276820 | 3474 |
| prestige/name | whr_filtered | 0.726454 | 0.256436 | 1184 |
| prestige/name | whr_filtered_deflated | 0.700606 | 0.249012 | 1184 |
| prestige/name | elo_sequential | 0.621480 | 0.215890 | 1184 |
| prestige/name | elo_unit_frozen | 0.686416 | 0.244655 | 1184 |
| prestige/name | glicko2_chapter_period | 0.751339 | 0.269992 | 1184 |
| prestige/person | whr_filtered | 0.728047 | 0.256969 | 1183 |
| prestige/person | whr_filtered_deflated | 0.701779 | 0.249448 | 1183 |
| prestige/person | elo_sequential | 0.621698 | 0.215988 | 1183 |
| prestige/person | elo_unit_frozen | 0.686573 | 0.244724 | 1183 |
| prestige/person | glicko2_chapter_period | 0.753169 | 0.270541 | 1183 |
| inclusion/name | whr_filtered | 0.720449 | 0.255018 | 531 |
| inclusion/name | whr_filtered_deflated | 0.694206 | 0.247104 | 531 |
| inclusion/name | elo_sequential | 0.629650 | 0.218810 | 531 |
| inclusion/name | elo_unit_frozen | 0.682716 | 0.243506 | 531 |
| inclusion/name | glicko2_chapter_period | 0.830490 | 0.299996 | 531 |
| inclusion/person | whr_filtered | 0.719676 | 0.254638 | 531 |
| inclusion/person | whr_filtered_deflated | 0.693599 | 0.246804 | 531 |
| inclusion/person | elo_sequential | 0.629459 | 0.218717 | 531 |
| inclusion/person | elo_unit_frozen | 0.682495 | 0.243397 | 531 |
| inclusion/person | glicko2_chapter_period | 0.828472 | 0.298909 | 531 |

### w2 selection

| lens/view | w2 | log-loss |
| --- | ---: | ---: |
| advantage/name | 5 | 0.714412 |
| advantage/name | 15 **(selected)** | 0.712944 |
| advantage/name | 35 | 0.712998 |
| advantage/name | 60 | 0.714222 |
| advantage/person | 5 | 0.714020 |
| advantage/person | 15 **(selected)** | 0.712578 |
| advantage/person | 35 | 0.712692 |
| advantage/person | 60 | 0.713984 |
| prestige/name | 5 | 0.732717 |
| prestige/name | 15 | 0.729890 |
| prestige/name | 35 | 0.727362 |
| prestige/name | 60 **(selected)** | 0.726454 |
| prestige/person | 5 | 0.734353 |
| prestige/person | 15 | 0.731528 |
| prestige/person | 35 | 0.728982 |
| prestige/person | 60 **(selected)** | 0.728047 |
| inclusion/name | 5 **(selected)** | 0.720449 |
| inclusion/name | 15 | 0.721532 |
| inclusion/name | 35 | 0.723758 |
| inclusion/name | 60 | 0.726438 |
| inclusion/person | 5 **(selected)** | 0.719676 |
| inclusion/person | 15 | 0.720813 |
| inclusion/person | 35 | 0.723126 |
| inclusion/person | 60 | 0.725895 |

## 4. Literary panel (pre-registered)

Each claim comes from the design doc; each operationalization was fixed before the ratings were read. Name view, and the standings referred to are the non-provisional set.

**5/8 claims pass.**

| claim | verdict |
| --- | --- |
| the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens | FAIL |
| Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all) | PASS |
| Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set | FAIL |
| Odette's prestige above her inclusion: prestige rating > inclusion rating | PASS |
| Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4 | PASS |
| the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median | FAIL |
| Saniette last or near it: bottom 10% of the non-provisional advantage set | PASS |
| l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons | PASS |

### duchesse — FAIL

the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens

| lens | rating | band | rank | non provisional count | rank percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| advantage | 1460.6 | 104.1 | 14 | 35 | 0.4 |
| prestige | 1588.4 | 167.4 | 6 | 8 | 0.75 |
| inclusion | 1479 | 163.2 | 3 | 9 | 0.333 |

### rachel — PASS

Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all)

| lens | rating | band | rank | unit count | comparison count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1497.5 | 148.3 | 16 | 43 | 94 | False |
| prestige | 1761.5 | 233 | - | 43 | 33 | True |
| inclusion | 1169.3 | 331.9 | - | 43 | 13 | True |

### bloch — FAIL

Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set

| lens | rating | rank | rank percentile | non provisional count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: |
| inclusion | 1362.3 | 6 | 0.667 | 9 | False |

### odette — PASS

Odette's prestige above her inclusion: prestige rating > inclusion rating

| lens | rating | rank | mean movement |
| --- | ---: | ---: | ---: |
| prestige | 1688.7 | - | 0.039 |
| inclusion | 1348.5 | 5 | -0.037 |

### charlus — PASS

Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4

| lens | first rating | last rating | early volume mean | late volume mean | early node count | late node count | rating | rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1533.5 | 1407.3 | 1488.1 | 1409.4 | 81 | 33 | 1407.3 | 23 |
| prestige | 1622.6 | 1446.9 | 1673.8 | 1496.1 | 39 | 22 | 1446.9 | 8 |
| inclusion | 1779.7 | 1785.9 | 1782.7 | 1785.7 | 18 | 8 | 1785.9 | - |

### narrator — FAIL

the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median

| lens | rating | band | rank | rank percentile | median band | unit count | comparison count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1368.7 | 87.2 | 26 | 0.743 | 148.3 | 316 | 816 |
| prestige | 1702 | - | 2 | - | - | - | - |
| inclusion | 1520.1 | - | 1 | - | - | - | - |

### saniette — PASS

Saniette last or near it: bottom 10% of the non-provisional advantage set

| lens | rating | band | rank | rank percentile | provisional | non provisional count | mean movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1230 | 197.8 | 35 | 1 | False | 35 | -1.13 |

### amie — PASS

l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons

| lens | unit count | comparison count | rating | band | rank | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 12 | 24 | 1700.1 | 241.9 | - | True |

## 5. Headline standings (name view, non-provisional)

### advantage — top 15 of 35 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Françoise | 1647.8 | 140.9 | 1506.9 | 82 | 141 | +0.120 | 0.529 |
| 2 | la grand-mère | 1654.2 | 164.8 | 1489.4 | 80 | 142 | +0.138 | 0.639 |
| 3 | Bergotte | 1630.8 | 174.8 | 1456.0 | 36 | 79 | +0.145 | 0.662 |
| 4 | Elstir | 1609.7 | 154.9 | 1454.8 | 29 | 70 | +0.476 | 0.677 |
| 5 | Aimé | 1616.0 | 183.9 | 1432.1 | 18 | 40 | +0.153 | 0.248 |
| 6 | la mère du narrateur | 1579.5 | 166.9 | 1412.6 | 40 | 81 | +0.087 | 0.306 |
| 7 | Norpois | 1573.0 | 164.7 | 1408.3 | 63 | 121 | -0.157 | 0.442 |
| 8 | princesse de Guermantes | 1576.7 | 174.1 | 1402.6 | 25 | 58 | +0.129 | 0.470 |
| 9 | Albertine | 1482.9 | 96.0 | 1386.9 | 146 | 298 | -0.173 | 0.705 |
| 10 | prince de Guermantes | 1542.3 | 163.6 | 1378.7 | 22 | 61 | -0.093 | 0.338 |
| 11 | le père du narrateur | 1559.8 | 187.0 | 1372.8 | 24 | 47 | +0.007 | 0.135 |
| 12 | Mme Verdurin | 1488.5 | 120.8 | 1367.7 | 82 | 162 | -0.299 | 0.345 |
| 13 | Swann | 1474.9 | 111.1 | 1363.8 | 202 | 446 | -0.314 | 0.658 |
| 14 | duchesse de Guermantes | 1460.6 | 104.1 | 1356.5 | 199 | 452 | +0.051 | 0.485 |
| 15 | Robert de Saint-Loup | 1457.4 | 106.9 | 1350.5 | 168 | 338 | -0.107 | 0.604 |

Bottom 5, advantage:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 35 | Saniette | 1230.0 | 197.8 | 1032.2 |
| 34 | Morel | 1303.8 | 120.8 | 1183.0 |
| 33 | Mme de Cambremer | 1400.7 | 180.0 | 1220.7 |
| 32 | Bloch | 1349.2 | 121.4 | 1227.8 |
| 31 | la Berma | 1419.8 | 185.7 | 1234.1 |

### prestige — top 15 of 8 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Morel | 1817.3 | 195.1 | 1622.2 | 32 | 41 | +0.077 | 0.128 |
| 2 | le narrateur | 1702.0 | 170.7 | 1531.3 | 316 | 187 | +0.026 | 0.063 |
| 3 | Bloch | 1689.0 | 186.6 | 1502.4 | 71 | 55 | -0.067 | 0.159 |
| 4 | Gilberte | 1643.3 | 161.5 | 1481.8 | 76 | 77 | +0.098 | 0.134 |
| 5 | Mme Verdurin | 1637.7 | 169.5 | 1468.2 | 82 | 94 | +0.070 | 0.223 |
| 6 | duchesse de Guermantes | 1588.4 | 167.4 | 1421.0 | 199 | 197 | +0.163 | 0.206 |
| 7 | Robert de Saint-Loup | 1588.6 | 183.7 | 1404.9 | 168 | 128 | -0.000 | 0.160 |
| 8 | baron de Charlus | 1446.9 | 159.7 | 1287.2 | 119 | 164 | +0.041 | 0.268 |

Bottom 5, prestige:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 8 | baron de Charlus | 1446.9 | 159.7 | 1287.2 |
| 7 | Robert de Saint-Loup | 1588.6 | 183.7 | 1404.9 |
| 6 | duchesse de Guermantes | 1588.4 | 167.4 | 1421.0 |
| 5 | Mme Verdurin | 1637.7 | 169.5 | 1468.2 |
| 4 | Gilberte | 1643.3 | 161.5 | 1481.8 |

### inclusion — top 15 of 9 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | le narrateur | 1520.1 | 101.3 | 1418.8 | 316 | 223 | +0.051 | 0.220 |
| 2 | Gilberte | 1544.9 | 173.8 | 1371.1 | 76 | 33 | +0.008 | 0.035 |
| 3 | duchesse de Guermantes | 1479.0 | 163.2 | 1315.8 | 199 | 41 | -0.004 | 0.004 |
| 4 | Robert de Saint-Loup | 1427.2 | 162.8 | 1264.4 | 168 | 44 | -0.011 | 0.021 |
| 5 | Odette | 1348.5 | 154.0 | 1194.5 | 142 | 64 | -0.037 | 0.071 |
| 6 | Bloch | 1362.3 | 184.4 | 1177.9 | 71 | 38 | -0.120 | 0.177 |
| 7 | Swann | 1287.1 | 127.2 | 1159.9 | 202 | 99 | -0.067 | 0.117 |
| 8 | Mme Verdurin | 1333.9 | 181.3 | 1152.6 | 82 | 35 | -0.056 | 0.074 |
| 9 | Albertine | 1292.8 | 166.7 | 1126.1 | 146 | 50 | -0.071 | 0.090 |

Bottom 5, inclusion:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 9 | Albertine | 1292.8 | 166.7 | 1126.1 |
| 8 | Mme Verdurin | 1333.9 | 181.3 | 1152.6 |
| 7 | Swann | 1287.1 | 127.2 | 1159.9 |
| 6 | Bloch | 1362.3 | 184.4 | 1177.9 |
| 5 | Odette | 1348.5 | 154.0 | 1194.5 |

## 6. Person view

The person view aggregates on registry entity ids with `person_view_merge` links applied, so the two era names of one man become one player; `keep_separate` links (the post-V7 princesse de Guermantes, who is Mme Verdurin holding a dead woman's title) never merge.

| lens | merged | name-view rows | person-view row | mean abs rank shift | self-pairings dropped |
| --- | --- | --- | --- | ---: | ---: |
| advantage | le-peintre -> elstir | le peintre r=1729 rank=- units=8; Elstir r=1610 rank=4 units=29 | elstir r=1630 rank=3 units=37 | 0.114 | 1 |
| advantage | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1570 rank=- units=3; duc de Guermantes r=1396 rank=27 units=110 | duc-de-guermantes r=1400 rank=27 units=113 | 0.114 | 1 |
| prestige | le-peintre -> elstir | le peintre r=1590 rank=- units=8; Elstir r=1336 rank=- units=29 | elstir r=1394 rank=- units=37 | 0.0 | 1 |
| prestige | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1477 rank=- units=3; duc de Guermantes r=1367 rank=- units=110 | duc-de-guermantes r=1366 rank=- units=113 | 0.0 | 1 |
| inclusion | le-peintre -> elstir | le peintre r=1611 rank=- units=8; Elstir r=1574 rank=- units=29 | elstir r=1612 rank=- units=37 | 0.0 | 0 |
| inclusion | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1500 rank=- units=3; duc de Guermantes r=1413 rank=- units=110 | duc-de-guermantes r=1414 rank=- units=113 | 0.0 | 0 |

Largest rank shifts between the two views (name-view rank minus person-view rank; the two views rank different fields, so a shift is not by itself a finding):

| lens | character | person key | name rank | person rank | shift | rating shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| advantage | Bergotte | bergotte | 3 | 4 | -1 | +1.8 |
| advantage | Elstir | elstir | 4 | 3 | +1 | +20.0 |
| advantage | M. Verdurin | m-verdurin | 18 | 17 | +1 | -1.5 |
| advantage | comte de Forcheville | comte-de-forcheville | 17 | 18 | -1 | -4.8 |
| advantage | Aimé | aime | 5 | 5 | +0 | +2.3 |
| prestige | Bloch | bloch | 3 | 3 | +0 | +0.0 |
| prestige | Gilberte | gilberte | 4 | 4 | +0 | -0.3 |
| prestige | Mme Verdurin | mme-verdurin | 5 | 5 | +0 | +0.2 |
| prestige | Morel | morel | 1 | 1 | +0 | +0.1 |
| prestige | Robert de Saint-Loup | saint-loup | 7 | 7 | +0 | +0.0 |
| inclusion | Albertine | albertine | 9 | 9 | +0 | +1.6 |
| inclusion | Bloch | bloch | 6 | 6 | +0 | +1.0 |
| inclusion | Gilberte | gilberte | 2 | 2 | +0 | +1.3 |
| inclusion | Mme Verdurin | mme-verdurin | 8 | 8 | +0 | +0.9 |
| inclusion | Odette | odette | 5 | 5 | +0 | +1.1 |

## 7. Reading notes: where the implementation had to choose

The design doc leaves four points open; each was resolved once, in code, and is recorded here so the review can overrule it.

1. **kappa is scoped to the lens.** "The mean confidence of c's effects in u" is read as the effects that MOVE c in this lens. A character with only a `social_status` effect is therefore a zero-effect character under advantage and falls back to presence confidence there, while carrying that effect's confidence under prestige. The alternative (pooling all five dimensions into one kappa) would let a lens's weights be set by evidence that lens is defined not to see.
2. **Label precedence.** A movement past the tie band names itself first; the sign-conflict test decides only within the band. So a character with a big positive movement and one small negative effect reads positive, not mixed. Mixed still REQUIRES a genuine sign conflict, which is the clause the doc makes binding.
3. **Predictive scores are unweighted.** The WHR fits use the weights; the scoring of predictions does not, because ELO and Glicko-2 have no weight to use and a weighted loss would not be comparable to theirs.
4. **w2 is selected per lens AND per view**, independently, by the same one-step-ahead log-loss rule v1 uses.

Deferred, as the design doc says: dossier lens cards (dominant dimension, percentile), the archetype rewrite, and the person/name UI toggle -- all app-facing, all after the adoption gate. The corpus summary carries the sign triple the archetype would use.

Wall clock: 87.2 s for the validation battery; 540.057 s for the build it reads.

