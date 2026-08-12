# Scoring v2 validation report (staged, pre-adoption)

Corpus: foundation, 34 runs, 963 reviewed units, 963 narrative time points. Comparisons per lens: {'advantage': 5756, 'prestige': 5756, 'inclusion': 5756}.

Formula: `proust/scoring_v2.py`, exactly as specified in `proust/docs/scoring_v2_design.md`. Ratings: weighted WHR (`proust/whr.py`), smoothed and filtered, on the `cumulative_unit_index` narrative axis. Everything here is staged under `outputs/scoring-v2/`; adoption is a separate reviewed decision.

w2 selected per lens/view: advantage/name = 15, advantage/person = 15, inclusion/name = 5, inclusion/person = 5, prestige/name = 15, prestige/person = 15

## 1. Lens orthogonality

The design predicts cross-lens rating correlations should FALL against v1: v1's weight tables blended every dimension into every lens, v2's projection partitions them.

| pair | v2 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.1761 (n=288) |
| advantage vs inclusion | +0.1795 (n=288) |
| prestige vs inclusion | +0.0549 (n=288) |
| **mean abs** | **0.1368** |

| pair | v1 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.9852 (n=288) |
| advantage vs inclusion | +0.9897 (n=288) |
| prestige vs inclusion | +0.9736 (n=288) |
| **mean abs** | **0.9828** |

| pair | v2 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.1741 (n=60) |
| advantage vs inclusion | +0.2314 (n=60) |
| prestige vs inclusion | +0.1137 (n=66) |
| **mean abs** | **0.1731** |

| pair | v1 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.9847 (n=91) |
| advantage vs inclusion | +0.9804 (n=91) |
| prestige vs inclusion | +0.9627 (n=91) |
| **mean abs** | **0.9759** |

**Verdict**: mean |rho| 0.983 (v1) -> 0.137 (v2): prediction held.

## 2. Bootstrap stability

50 unit-level resamples with replacement, both formulas scored on the same drawn corpora; ranks are taken over the characters both formulas rate non-provisionally on the full corpus. Lower rank standard deviation is more stable.

| lens | field | v2 mean sd | v1 mean sd | v2 median sd | v1 median sd | v2 non-prov | v1 non-prov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 60 | 9.91 | 10.167 | 10.199 | 10.067 | 60 | 91 |
| prestige | 66 | 9.519 | 10.975 | 9.714 | 11.041 | 66 | 91 |
| inclusion | 70 | 9.064 | 11.935 | 9.098 | 11.939 | 70 | 91 |

### 2b. Frequency confounding

The design's fourth principle is that frequency must not masquerade as strength. Ratings are no longer sums, so nothing accumulates with appearances -- but the standings rank by rating MINUS band, and a band narrows with evidence. Where a lens's ratings are tightly packed and its bands are not, the ranking is mostly a comparison count. Spearman rho against comparison count, over each formula's own non-provisional set:

| lens | formula | conservative vs count | rating vs count | band vs count | rating spread | band spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| advantage | v2 | 0.443 | -0.087 | -0.904 | 332.3 | 122.4 |
| advantage | v1 | 0.44 | -0.121 | -0.925 | 607.4 | 132.3 |
| prestige | v2 | 0.731 | 0.068 | -0.923 | 288.9 | 125.6 |
| prestige | v1 | 0.447 | -0.088 | -0.927 | 554.1 | 132.6 |
| inclusion | v2 | 0.934 | -0.15 | -0.981 | 118.3 | 142.8 |
| inclusion | v1 | 0.434 | -0.17 | -0.927 | 605.5 | 133.4 |

## 3. Predictive sanity

One-step-ahead over the v2 comparisons in narrative order. Every system is scored UNWEIGHTED (one comparison, one prediction): ELO and Glicko-2 have no notion of a game weight, so a weighted loss would not be comparable to theirs. The WHR fits themselves DO use the weights. Cross-formula comparison against v1's own numbers is meaningless here -- the comparisons differ -- so only the within-v2 ordering is informative.

| lens/view | system | log-loss | Brier | comparisons |
| --- | --- | ---: | ---: | ---: |
| advantage/name | whr_filtered | 0.710754 | 0.256290 | 5756 |
| advantage/name | whr_filtered_deflated | 0.702428 | 0.253394 | 5756 |
| advantage/name | elo_sequential | 0.664231 | 0.236127 | 5756 |
| advantage/name | elo_unit_frozen | 0.696975 | 0.251322 | 5756 |
| advantage/name | glicko2_chapter_period | 0.723843 | 0.262102 | 5756 |
| advantage/person | whr_filtered | 0.710220 | 0.256034 | 5755 |
| advantage/person | whr_filtered_deflated | 0.702080 | 0.253221 | 5755 |
| advantage/person | elo_sequential | 0.664197 | 0.236112 | 5755 |
| advantage/person | elo_unit_frozen | 0.696956 | 0.251312 | 5755 |
| advantage/person | glicko2_chapter_period | 0.726389 | 0.263152 | 5755 |
| prestige/name | whr_filtered | 0.700859 | 0.253272 | 5756 |
| prestige/name | whr_filtered_deflated | 0.698047 | 0.252171 | 5756 |
| prestige/name | elo_sequential | 0.682428 | 0.244749 | 5756 |
| prestige/name | elo_unit_frozen | 0.697130 | 0.251805 | 5756 |
| prestige/name | glicko2_chapter_period | 0.707798 | 0.256266 | 5756 |
| prestige/person | whr_filtered | 0.700803 | 0.253245 | 5755 |
| prestige/person | whr_filtered_deflated | 0.698010 | 0.252152 | 5755 |
| prestige/person | elo_sequential | 0.682463 | 0.244767 | 5755 |
| prestige/person | elo_unit_frozen | 0.697147 | 0.251814 | 5755 |
| prestige/person | glicko2_chapter_period | 0.707779 | 0.256256 | 5755 |
| inclusion/name | whr_filtered | 0.697158 | 0.251801 | 5756 |
| inclusion/name | whr_filtered_deflated | 0.695895 | 0.251281 | 5756 |
| inclusion/name | elo_sequential | 0.689974 | 0.248414 | 5756 |
| inclusion/name | elo_unit_frozen | 0.695510 | 0.251118 | 5756 |
| inclusion/name | glicko2_chapter_period | 0.700971 | 0.253614 | 5756 |
| inclusion/person | whr_filtered | 0.697122 | 0.251783 | 5755 |
| inclusion/person | whr_filtered_deflated | 0.695867 | 0.251267 | 5755 |
| inclusion/person | elo_sequential | 0.689972 | 0.248414 | 5755 |
| inclusion/person | elo_unit_frozen | 0.695511 | 0.251118 | 5755 |
| inclusion/person | glicko2_chapter_period | 0.700997 | 0.253628 | 5755 |

### w2 selection

| lens/view | w2 | log-loss |
| --- | ---: | ---: |
| advantage/name | 5 | 0.711218 |
| advantage/name | 15 **(selected)** | 0.710754 |
| advantage/name | 35 | 0.711479 |
| advantage/name | 60 | 0.712824 |
| advantage/person | 5 | 0.710655 |
| advantage/person | 15 **(selected)** | 0.710220 |
| advantage/person | 35 | 0.710971 |
| advantage/person | 60 | 0.712341 |
| prestige/name | 5 | 0.700893 |
| prestige/name | 15 **(selected)** | 0.700859 |
| prestige/name | 35 | 0.701139 |
| prestige/name | 60 | 0.701636 |
| prestige/person | 5 | 0.700822 |
| prestige/person | 15 **(selected)** | 0.700803 |
| prestige/person | 35 | 0.701101 |
| prestige/person | 60 | 0.701609 |
| inclusion/name | 5 **(selected)** | 0.697158 |
| inclusion/name | 15 | 0.697185 |
| inclusion/name | 35 | 0.697320 |
| inclusion/name | 60 | 0.697512 |
| inclusion/person | 5 **(selected)** | 0.697122 |
| inclusion/person | 15 | 0.697152 |
| inclusion/person | 35 | 0.697291 |
| inclusion/person | 60 | 0.697485 |

## 4. Literary panel (pre-registered)

Each claim comes from the design doc; each operationalization was fixed before the ratings were read. Name view, and the standings referred to are the non-provisional set.

**6/8 claims pass.**

| claim | verdict |
| --- | --- |
| the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens | PASS |
| Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all) | PASS |
| Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set | FAIL |
| Odette's prestige above her inclusion: prestige rating > inclusion rating | PASS |
| Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4 | PASS |
| the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median | FAIL |
| Saniette last or near it: bottom 10% of the non-provisional advantage set | PASS |
| l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons | PASS |

### duchesse — PASS

the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens

| lens | rating | band | rank | non provisional count | rank percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| advantage | 1484.4 | 80.5 | 19 | 60 | 0.317 |
| prestige | 1516.6 | 79.9 | 9 | 66 | 0.136 |
| inclusion | 1503 | 62.1 | 4 | 70 | 0.057 |

### rachel — PASS

Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all)

| lens | rating | band | rank | unit count | comparison count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1514.1 | 113.3 | 21 | 43 | 146 | False |
| prestige | 1571.8 | 113 | 2 | 43 | 146 | False |
| inclusion | 1485.5 | 93.3 | 36 | 43 | 146 | False |

### bloch — FAIL

Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set

| lens | rating | rank | rank percentile | non provisional count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: |
| inclusion | 1495.6 | 8 | 0.114 | 70 | False |

### odette — PASS

Odette's prestige above her inclusion: prestige rating > inclusion rating

| lens | rating | rank | mean movement |
| --- | ---: | ---: | ---: |
| prestige | 1537.9 | 7 | 0.039 |
| inclusion | 1495.7 | 10 | -0.037 |

### charlus — PASS

Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4

| lens | first rating | last rating | early volume mean | late volume mean | early node count | late node count | rating | rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1545.1 | 1459.5 | 1515.9 | 1458.8 | 84 | 34 | 1459.5 | 28 |
| prestige | 1516.2 | 1485 | 1545.3 | 1506.6 | 84 | 34 | 1485 | 16 |
| inclusion | 1516.9 | 1515.7 | 1516.1 | 1516.7 | 84 | 34 | 1515.7 | 1 |

### narrator — FAIL

the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median

| lens | rating | band | rank | rank percentile | median band | unit count | comparison count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1421 | 76.1 | 42 | 0.7 | 130.3 | 316 | 1093 |
| prestige | 1531.3 | - | 3 | - | - | - | - |
| inclusion | 1505.1 | - | 2 | - | - | - | - |

### saniette — PASS

Saniette last or near it: bottom 10% of the non-provisional advantage set

| lens | rating | band | rank | rank percentile | provisional | non provisional count | mean movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1300.8 | 174.1 | 60 | 1 | False | 60 | -1.13 |

### amie — PASS

l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons

| lens | unit count | comparison count | rating | band | rank | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 12 | 44 | 1605.5 | 158.3 | 8 | False |

## 5. Headline standings (name view, non-provisional)

### advantage — top 15 of 60 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Françoise | 1601.4 | 121.9 | 1479.5 | 82 | 217 | +0.120 | 0.529 |
| 2 | la grand-mère | 1600.0 | 127.9 | 1472.1 | 80 | 225 | +0.138 | 0.639 |
| 3 | Elstir | 1578.7 | 121.5 | 1457.2 | 29 | 106 | +0.476 | 0.677 |
| 4 | le peintre | 1605.4 | 150.0 | 1455.4 | 8 | 42 | +0.279 | 0.279 |
| 5 | Bergotte | 1597.4 | 145.1 | 1452.3 | 36 | 129 | +0.145 | 0.662 |
| 6 | Jupien | 1575.0 | 124.1 | 1450.9 | 18 | 68 | +0.339 | 0.512 |
| 7 | Aimé | 1576.4 | 127.3 | 1449.1 | 18 | 79 | +0.153 | 0.248 |
| 8 | l'amie de Mlle Vinteuil | 1605.5 | 158.3 | 1447.2 | 12 | 44 | +0.068 | 0.068 |
| 9 | le grand-père du narrateur | 1633.1 | 190.7 | 1442.4 | 16 | 63 | +0.069 | 0.069 |
| 10 | la mère du narrateur | 1543.3 | 115.4 | 1427.9 | 40 | 144 | +0.087 | 0.306 |
| 11 | Mlle Vinteuil | 1553.3 | 125.8 | 1427.5 | 15 | 71 | -0.152 | 0.264 |
| 12 | Swann | 1503.3 | 83.6 | 1419.7 | 202 | 667 | -0.314 | 0.658 |
| 13 | Norpois | 1566.2 | 149.5 | 1416.7 | 63 | 180 | -0.157 | 0.442 |
| 14 | Robert de Saint-Loup | 1498.5 | 84.9 | 1413.6 | 168 | 508 | -0.107 | 0.604 |
| 15 | le père du narrateur | 1566.1 | 154.4 | 1411.7 | 24 | 90 | +0.007 | 0.135 |

Bottom 5, advantage:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 60 | Saniette | 1300.8 | 174.1 | 1126.7 |
| 59 | Mme d'Arpajon | 1374.2 | 163.8 | 1210.4 |
| 58 | M. de Vaugoubert | 1396.1 | 169.1 | 1227.0 |
| 57 | comtesse Molé | 1432.2 | 173.9 | 1258.3 |
| 56 | la Berma | 1456.1 | 161.9 | 1294.2 |

### prestige — top 15 of 66 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Morel | 1589.4 | 96.2 | 1493.2 | 32 | 152 | +0.077 | 0.128 |
| 2 | Rachel | 1571.8 | 113.0 | 1458.8 | 43 | 146 | +0.067 | 0.178 |
| 3 | le narrateur | 1531.3 | 73.6 | 1457.7 | 316 | 1093 | +0.026 | 0.063 |
| 4 | Bloch | 1548.4 | 92.7 | 1455.7 | 71 | 270 | -0.067 | 0.159 |
| 5 | Gilberte | 1533.7 | 83.9 | 1449.8 | 76 | 312 | +0.098 | 0.134 |
| 6 | Mme Verdurin | 1537.1 | 93.5 | 1443.6 | 82 | 311 | +0.070 | 0.223 |
| 7 | Odette | 1537.9 | 94.4 | 1443.5 | 142 | 462 | +0.039 | 0.151 |
| 8 | Robert de Saint-Loup | 1525.4 | 83.3 | 1442.1 | 168 | 508 | -0.000 | 0.160 |
| 9 | duchesse de Guermantes | 1516.6 | 79.9 | 1436.7 | 199 | 662 | +0.163 | 0.206 |
| 10 | Albertine | 1506.7 | 84.8 | 1421.9 | 146 | 387 | -0.017 | 0.051 |
| 11 | Andrée | 1524.4 | 104.6 | 1419.8 | 31 | 114 | +0.023 | 0.023 |
| 12 | Jupien | 1529.4 | 118.8 | 1410.6 | 18 | 68 | +0.094 | 0.094 |
| 13 | M. Verdurin | 1532.8 | 122.5 | 1410.3 | 27 | 110 | -0.032 | 0.087 |
| 14 | Mlle Vinteuil | 1529.7 | 121.0 | 1408.7 | 15 | 71 | +0.000 | 0.000 |
| 15 | princesse de Parme | 1510.4 | 105.3 | 1405.1 | 38 | 130 | +0.095 | 0.095 |

Bottom 5, prestige:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 66 | Saniette | 1300.5 | 177.2 | 1123.3 |
| 65 | marquis de Cambremer | 1375.0 | 141.1 | 1233.9 |
| 64 | la Berma | 1408.0 | 165.8 | 1242.2 |
| 63 | Mme d'Arpajon | 1410.2 | 161.1 | 1249.1 |
| 62 | Mme de Sévigné | 1480.4 | 195.3 | 1285.1 |

### inclusion — top 15 of 70 non-provisional (288 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baron de Charlus | 1515.7 | 61.6 | 1454.1 | 119 | 485 | +0.012 | 0.012 |
| 2 | le narrateur | 1505.1 | 55.0 | 1450.1 | 316 | 1093 | +0.051 | 0.220 |
| 3 | Gilberte | 1514.4 | 68.4 | 1446.0 | 76 | 312 | +0.008 | 0.035 |
| 4 | duchesse de Guermantes | 1503.0 | 62.1 | 1440.9 | 199 | 662 | -0.004 | 0.004 |
| 5 | Robert de Saint-Loup | 1502.1 | 65.2 | 1436.9 | 168 | 508 | -0.011 | 0.021 |
| 6 | duc de Guermantes | 1504.5 | 73.2 | 1431.3 | 110 | 401 | +0.000 | 0.000 |
| 7 | Morel | 1502.0 | 79.2 | 1422.8 | 32 | 152 | -0.024 | 0.024 |
| 8 | Bloch | 1495.6 | 75.0 | 1420.6 | 71 | 270 | -0.120 | 0.177 |
| 9 | Mme de Villeparisis | 1510.5 | 90.1 | 1420.4 | 79 | 236 | -0.029 | 0.029 |
| 10 | Odette | 1495.7 | 75.4 | 1420.3 | 142 | 462 | -0.037 | 0.071 |
| 11 | la mère du narrateur | 1514.0 | 94.1 | 1419.9 | 40 | 144 | -0.018 | 0.018 |
| 12 | Mlle Vinteuil | 1529.8 | 111.9 | 1417.9 | 15 | 71 | +0.000 | 0.000 |
| 13 | Swann | 1482.0 | 64.7 | 1417.3 | 202 | 667 | -0.067 | 0.117 |
| 14 | Brichot | 1501.5 | 85.0 | 1416.5 | 21 | 135 | +0.000 | 0.000 |
| 15 | Françoise | 1503.1 | 88.0 | 1415.1 | 82 | 217 | -0.018 | 0.018 |

Bottom 5, inclusion:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 70 | Saniette | 1426.4 | 146.2 | 1280.2 |
| 69 | Mme Féré | 1497.6 | 197.8 | 1299.8 |
| 68 | M. de Crécy | 1497.6 | 195.2 | 1302.4 |
| 67 | marquise de Gallardon | 1501.9 | 192.7 | 1309.2 |
| 66 | tante Léonie | 1462.4 | 147.5 | 1314.9 |

## 6. Person view

The person view aggregates on registry entity ids with `person_view_merge` links applied, so the two era names of one man become one player; `keep_separate` links (the post-V7 princesse de Guermantes, who is Mme Verdurin holding a dead woman's title) never merge.

| lens | merged | name-view rows | person-view row | mean abs rank shift | self-pairings dropped |
| --- | --- | --- | --- | ---: | ---: |
| advantage | le-peintre -> elstir | le peintre r=1605 rank=4 units=8; Elstir r=1579 rank=3 units=29 | elstir r=1584 rank=3 units=37 | 1.367 | 1 |
| advantage | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1532 rank=41 units=3; duc de Guermantes r=1449 rank=39 units=110 | duc-de-guermantes r=1451 rank=38 units=113 | 1.367 | 1 |
| prestige | le-peintre -> elstir | le peintre r=1518 rank=30 units=8; Elstir r=1501 rank=27 units=29 | elstir r=1503 rank=27 units=37 | 1.0 | 1 |
| prestige | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1495 rank=58 units=3; duc de Guermantes r=1456 rank=40 units=110 | duc-de-guermantes r=1456 rank=40 units=113 | 1.0 | 1 |
| inclusion | le-peintre -> elstir | le peintre r=1515 rank=43 units=8; Elstir r=1504 rank=23 units=29 | elstir r=1506 rank=19 units=37 | 1.743 | 1 |
| inclusion | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1503 rank=60 units=3; duc de Guermantes r=1504 rank=6 units=110 | duc-de-guermantes r=1505 rank=6 units=113 | 1.743 | 1 |

Largest rank shifts between the two views (name-view rank minus person-view rank; the two views rank different fields, so a shift is not by itself a finding):

| lens | character | person key | name rank | person rank | shift | rating shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| advantage | Albertine | albertine | 17 | 14 | +3 | +1.2 |
| advantage | Mme de Villeparisis | mme-de-villeparisis | 32 | 29 | +3 | +0.9 |
| advantage | prince des Laumes | duc-de-guermantes | 41 | 38 | +3 | -80.5 |
| advantage | princesse de Parme | princesse-de-parme | 47 | 44 | +3 | +1.9 |
| advantage | Bloch | bloch | 44 | 42 | +2 | +1.1 |
| prestige | prince des Laumes | duc-de-guermantes | 58 | 40 | +18 | -39.7 |
| prestige | le peintre | elstir | 30 | 27 | +3 | -15.2 |
| prestige | M. d'Argencourt | m-d-argencourt | 61 | 59 | +2 | -0.1 |
| prestige | Mme Sazerat | mme-sazerat | 60 | 58 | +2 | +0.0 |
| prestige | Mme d'Arpajon | mme-d-arpajon | 63 | 61 | +2 | -0.2 |
| inclusion | prince des Laumes | duc-de-guermantes | 60 | 6 | +54 | +1.6 |
| inclusion | le peintre | elstir | 43 | 19 | +24 | -9.0 |
| inclusion | Elstir | elstir | 23 | 19 | +4 | +1.8 |
| inclusion | M. Ski | ski | 65 | 63 | +2 | +0.2 |
| inclusion | M. de Crécy | M. de Crécy | 68 | 66 | +2 | +0.1 |

## 7. Reading notes: where the implementation had to choose

The design doc leaves four points open; each was resolved once, in code, and is recorded here so the review can overrule it.

1. **kappa is scoped to the lens.** "The mean confidence of c's effects in u" is read as the effects that MOVE c in this lens. A character with only a `social_status` effect is therefore a zero-effect character under advantage and falls back to presence confidence there, while carrying that effect's confidence under prestige. The alternative (pooling all five dimensions into one kappa) would let a lens's weights be set by evidence that lens is defined not to see.
2. **Label precedence.** A movement past the tie band names itself first; the sign-conflict test decides only within the band. So a character with a big positive movement and one small negative effect reads positive, not mixed. Mixed still REQUIRES a genuine sign conflict, which is the clause the doc makes binding.
3. **Predictive scores are unweighted.** The WHR fits use the weights; the scoring of predictions does not, because ELO and Glicko-2 have no weight to use and a weighted loss would not be comparable to theirs.
4. **w2 is selected per lens AND per view**, independently, by the same one-step-ahead log-loss rule v1 uses.

Deferred, as the design doc says: dossier lens cards (dominant dimension, percentile), the archetype rewrite, and the person/name UI toggle -- all app-facing, all after the adoption gate. The corpus summary carries the sign triple the archetype would use.

Wall clock: 126.8 s for the validation battery; 2160.312 s for the build it reads.

