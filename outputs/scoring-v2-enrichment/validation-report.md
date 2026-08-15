# Scoring v2 validation report (staged, pre-adoption)

Corpus: None, 34 runs, 963 reviewed units, 963 narrative time points. Comparisons per lens: {'advantage': 2708, 'prestige': 954, 'inclusion': 565}.

Formula: `proust/scoring_v2.py`, exactly as specified in `proust/docs/scoring_v2_design.md`. Ratings: weighted WHR (`proust/whr.py`), smoothed and filtered, on the `cumulative_unit_index` narrative axis. Everything here is staged under `outputs/scoring-v2/`; adoption is a separate reviewed decision.

w2 selected per lens/view: advantage/name = 5, advantage/person = 5, inclusion/name = 5, inclusion/person = 5, prestige/name = 5, prestige/person = 5

## 1. Lens orthogonality

The design predicts cross-lens rating correlations should FALL against v1: v1's weight tables blended every dimension into every lens, v2's projection partitions them.

| pair | v2 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.2744 (n=193) |
| advantage vs inclusion | +0.1340 (n=193) |
| prestige vs inclusion | +0.0002 (n=193) |
| **mean abs** | **0.1362** |

| pair | v1 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.9852 (n=288) |
| advantage vs inclusion | +0.9897 (n=288) |
| prestige vs inclusion | +0.9736 (n=288) |
| **mean abs** | **0.9828** |

| pair | v2 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.2761 (n=22) |
| advantage vs inclusion | +0.3667 (n=9) |
| prestige vs inclusion | +0.3333 (n=9) |
| **mean abs** | **0.3254** |

| pair | v1 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.9847 (n=91) |
| advantage vs inclusion | +0.9804 (n=91) |
| prestige vs inclusion | +0.9627 (n=91) |
| **mean abs** | **0.9759** |

**Verdict**: mean |rho| 0.983 (v1) -> 0.136 (v2): prediction held.

## 2. Bootstrap stability

50 unit-level resamples with replacement, both formulas scored on the same drawn corpora; ranks are taken over the characters both formulas rate non-provisionally on the full corpus. Lower rank standard deviation is more stable.

| lens | field | v2 mean sd | v1 mean sd | v2 median sd | v1 median sd | v2 non-prov | v1 non-prov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 41 | 6.8 | 6.917 | 6.935 | 6.566 | 41 | 91 |
| prestige | 22 | 4.009 | 3.939 | 3.727 | 3.823 | 22 | 91 |
| inclusion | 9 | 1.395 | 1.888 | 1.46 | 2.003 | 9 | 91 |

### 2b. Frequency confounding

The design's fourth principle is that frequency must not masquerade as strength. Ratings are no longer sums, so nothing accumulates with appearances -- but the standings rank by rating MINUS band, and a band narrows with evidence. Where a lens's ratings are tightly packed and its bands are not, the ranking is mostly a comparison count. Spearman rho against comparison count, over each formula's own non-provisional set:

| lens | formula | conservative vs count | rating vs count | band vs count | rating spread | band spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| advantage | v2 | 0.451 | 0.054 | -0.961 | 456.2 | 124.5 |
| advantage | v1 | 0.44 | -0.121 | -0.925 | 607.4 | 132.3 |
| prestige | v2 | 0.554 | 0.223 | -0.958 | 330.6 | 105.4 |
| prestige | v1 | 0.447 | -0.088 | -0.927 | 554.1 | 132.6 |
| inclusion | v2 | 0.226 | 0.084 | -0.962 | 284.9 | 95.3 |
| inclusion | v1 | 0.434 | -0.17 | -0.927 | 605.5 | 133.4 |

## 3. Predictive sanity

One-step-ahead over the v2 comparisons in narrative order. Every system is scored UNWEIGHTED (one comparison, one prediction): ELO and Glicko-2 have no notion of a game weight, so a weighted loss would not be comparable to theirs. The WHR fits themselves DO use the weights. Cross-formula comparison against v1's own numbers is meaningless here -- the comparisons differ -- so only the within-v2 ordering is informative.

| lens/view | system | log-loss | Brier | comparisons |
| --- | --- | ---: | ---: | ---: |
| advantage/name | whr_filtered | 0.713182 | 0.255718 | 2708 |
| advantage/name | whr_filtered_deflated | 0.698182 | 0.250603 | 2708 |
| advantage/name | elo_sequential | 0.654221 | 0.231186 | 2708 |
| advantage/name | elo_unit_frozen | 0.684708 | 0.245236 | 2708 |
| advantage/name | glicko2_chapter_period | 0.714807 | 0.256846 | 2708 |
| advantage/person | whr_filtered | 0.713729 | 0.255878 | 2708 |
| advantage/person | whr_filtered_deflated | 0.698414 | 0.250672 | 2708 |
| advantage/person | elo_sequential | 0.654110 | 0.231154 | 2708 |
| advantage/person | elo_unit_frozen | 0.684616 | 0.245206 | 2708 |
| advantage/person | glicko2_chapter_period | 0.713007 | 0.256107 | 2708 |
| prestige/name | whr_filtered | 0.755528 | 0.270845 | 954 |
| prestige/name | whr_filtered_deflated | 0.724183 | 0.260902 | 954 |
| prestige/name | elo_sequential | 0.644968 | 0.227261 | 954 |
| prestige/name | elo_unit_frozen | 0.689898 | 0.248206 | 954 |
| prestige/name | glicko2_chapter_period | 0.798204 | 0.285332 | 954 |
| prestige/person | whr_filtered | 0.754954 | 0.270719 | 954 |
| prestige/person | whr_filtered_deflated | 0.723689 | 0.260763 | 954 |
| prestige/person | elo_sequential | 0.644788 | 0.227163 | 954 |
| prestige/person | elo_unit_frozen | 0.689699 | 0.248097 | 954 |
| prestige/person | glicko2_chapter_period | 0.797549 | 0.285122 | 954 |
| inclusion/name | whr_filtered | 0.740263 | 0.262942 | 565 |
| inclusion/name | whr_filtered_deflated | 0.711020 | 0.254037 | 565 |
| inclusion/name | elo_sequential | 0.645951 | 0.226850 | 565 |
| inclusion/name | elo_unit_frozen | 0.690794 | 0.248223 | 565 |
| inclusion/name | glicko2_chapter_period | 0.763966 | 0.273191 | 565 |
| inclusion/person | whr_filtered | 0.739496 | 0.262560 | 565 |
| inclusion/person | whr_filtered_deflated | 0.710281 | 0.253708 | 565 |
| inclusion/person | elo_sequential | 0.645817 | 0.226783 | 565 |
| inclusion/person | elo_unit_frozen | 0.690647 | 0.248150 | 565 |
| inclusion/person | glicko2_chapter_period | 0.763996 | 0.273228 | 565 |

### w2 selection

| lens/view | w2 | log-loss |
| --- | ---: | ---: |
| advantage/name | 5 **(selected)** | 0.713182 |
| advantage/name | 15 | 0.713534 |
| advantage/name | 35 | 0.715735 |
| advantage/name | 60 | 0.718819 |
| advantage/person | 5 **(selected)** | 0.713729 |
| advantage/person | 15 | 0.713909 |
| advantage/person | 35 | 0.715898 |
| advantage/person | 60 | 0.718829 |
| prestige/name | 5 **(selected)** | 0.755528 |
| prestige/name | 15 | 0.756952 |
| prestige/name | 35 | 0.761009 |
| prestige/name | 60 | 0.766297 |
| prestige/person | 5 **(selected)** | 0.754954 |
| prestige/person | 15 | 0.756416 |
| prestige/person | 35 | 0.760532 |
| prestige/person | 60 | 0.765878 |
| inclusion/name | 5 **(selected)** | 0.740263 |
| inclusion/name | 15 | 0.741692 |
| inclusion/name | 35 | 0.744933 |
| inclusion/name | 60 | 0.749333 |
| inclusion/person | 5 **(selected)** | 0.739496 |
| inclusion/person | 15 | 0.740948 |
| inclusion/person | 35 | 0.744227 |
| inclusion/person | 60 | 0.748671 |

## 4. Literary panel (pre-registered)

Each claim comes from the design doc; each operationalization was fixed before the ratings were read. Name view, and the standings referred to are the non-provisional set.

**7/8 claims pass.**

| claim | verdict |
| --- | --- |
| the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens | PASS |
| Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all) | PASS |
| Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set | PASS |
| Odette's prestige above her inclusion: prestige rating > inclusion rating | PASS |
| Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4 | PASS |
| the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median | FAIL |
| Saniette last or near it: bottom 10% of the non-provisional advantage set | PASS |
| l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons | PASS |

### duchesse — PASS

the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens

| lens | rating | band | rank | non provisional count | rank percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| advantage | 1593.8 | 79.7 | 2 | 41 | 0.049 |
| prestige | 1683 | 99.2 | 2 | 22 | 0.091 |
| inclusion | 1619.2 | 159 | 2 | 9 | 0.222 |

### rachel — PASS

Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all)

| lens | rating | band | rank | unit count | comparison count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1578.7 | 132.4 | 9 | 29 | 56 | False |
| prestige | 1639.8 | 183.1 | 7 | 29 | 24 | False |
| inclusion | 1712.6 | 418 | - | 29 | 6 | True |

### bloch — PASS

Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set

| lens | rating | rank | rank percentile | non provisional count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: |
| inclusion | 1408.6 | 7 | 0.778 | 9 | False |

### odette — PASS

Odette's prestige above her inclusion: prestige rating > inclusion rating

| lens | rating | rank | mean movement |
| --- | ---: | ---: | ---: |
| prestige | 1685.9 | 3 | 0.107 |
| inclusion | 1401.8 | 6 | -0.094 |

### charlus — PASS

Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4

| lens | first rating | last rating | early volume mean | late volume mean | early node count | late node count | rating | rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1580.9 | 1513.6 | 1552.1 | 1514.8 | 68 | 33 | 1513.6 | 10 |
| prestige | 1606.3 | 1590.7 | 1604.8 | 1594.7 | 28 | 21 | 1590.7 | 5 |
| inclusion | 1583.1 | 1570.6 | 1578.8 | 1571.6 | 26 | 9 | 1570.6 | 3 |

### narrator — FAIL

the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median

| lens | rating | band | rank | rank percentile | median band | unit count | comparison count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1513.2 | 73.3 | 11 | 0.268 | 132.4 | 209 | 399 |
| prestige | 1633.4 | - | 4 | - | - | - | - |
| inclusion | 1601.9 | - | 1 | - | - | - | - |

### saniette — PASS

Saniette last or near it: bottom 10% of the non-provisional advantage set

| lens | rating | band | rank | rank percentile | provisional | non provisional count | mean movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1255.9 | 173 | 41 | 1 | False | 41 | -0.846 |

### amie — PASS

l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons

| lens | unit count | comparison count | rating | band | rank | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 7 | 17 | 1719.9 | 233.9 | - | True |

## 5. Headline standings (name view, non-provisional)

### advantage — top 15 of 41 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | comte de Forcheville | 1712.1 | 162.7 | 1549.4 | 28 | 62 | +0.139 | 0.181 |
| 2 | duchesse de Guermantes | 1593.8 | 79.7 | 1514.1 | 183 | 354 | +0.049 | 0.490 |
| 3 | M. Verdurin | 1621.9 | 123.5 | 1498.4 | 32 | 84 | -0.176 | 0.267 |
| 4 | la mère du narrateur | 1619.8 | 146.0 | 1473.8 | 28 | 54 | +0.273 | 0.318 |
| 5 | docteur Cottard | 1582.5 | 118.6 | 1463.9 | 37 | 107 | -0.165 | 0.713 |
| 6 | Mme de Villeparisis | 1579.6 | 118.3 | 1461.3 | 73 | 118 | -0.077 | 0.369 |
| 7 | Françoise | 1574.8 | 113.6 | 1461.2 | 61 | 100 | +0.086 | 0.586 |
| 8 | la grand-mère | 1577.5 | 128.8 | 1448.7 | 48 | 74 | +0.177 | 0.667 |
| 9 | Rachel | 1578.7 | 132.4 | 1446.3 | 29 | 56 | -0.216 | 0.609 |
| 10 | baron de Charlus | 1513.6 | 73.7 | 1439.9 | 110 | 283 | -0.256 | 0.706 |
| 11 | le narrateur | 1513.2 | 73.3 | 1439.9 | 209 | 399 | -0.201 | 0.632 |
| 12 | Andrée | 1577.8 | 138.1 | 1439.7 | 25 | 54 | -0.061 | 0.659 |
| 13 | Mme Verdurin | 1534.9 | 95.6 | 1439.3 | 78 | 181 | -0.336 | 0.425 |
| 14 | Albertine | 1509.4 | 79.3 | 1430.1 | 126 | 183 | -0.203 | 0.744 |
| 15 | Robert de Saint-Loup | 1513.8 | 84.0 | 1429.8 | 138 | 234 | -0.132 | 0.640 |

Bottom 5, advantage:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 41 | Saniette | 1255.9 | 173.0 | 1082.9 |
| 40 | Legrandin | 1294.4 | 165.9 | 1128.5 |
| 39 | marquise de Gallardon | 1322.1 | 189.5 | 1132.6 |
| 38 | Mme d'Arpajon | 1346.6 | 172.3 | 1174.3 |
| 37 | Bloch | 1306.1 | 110.0 | 1196.1 |

### prestige — top 15 of 22 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Morel | 1772.6 | 164.5 | 1608.1 | 35 | 43 | +0.206 | 0.206 |
| 2 | duchesse de Guermantes | 1683.0 | 99.2 | 1583.8 | 183 | 156 | +0.216 | 0.270 |
| 3 | Odette | 1685.9 | 124.0 | 1561.9 | 124 | 77 | +0.107 | 0.169 |
| 4 | le narrateur | 1633.4 | 117.9 | 1515.5 | 209 | 102 | +0.061 | 0.100 |
| 5 | baron de Charlus | 1590.7 | 92.8 | 1497.9 | 110 | 132 | +0.032 | 0.269 |
| 6 | Mme Verdurin | 1573.7 | 105.1 | 1468.6 | 78 | 104 | +0.129 | 0.236 |
| 7 | Rachel | 1639.8 | 183.1 | 1456.7 | 29 | 24 | +0.041 | 0.200 |
| 8 | Gilberte | 1539.1 | 123.7 | 1415.4 | 57 | 53 | +0.085 | 0.174 |
| 9 | M. Verdurin | 1588.1 | 188.7 | 1399.4 | 32 | 25 | +0.000 | 0.000 |
| 10 | Mme de Villeparisis | 1527.1 | 131.2 | 1395.9 | 73 | 58 | -0.016 | 0.205 |
| 11 | Norpois | 1569.4 | 177.2 | 1392.2 | 54 | 38 | +0.101 | 0.127 |
| 12 | Swann | 1493.6 | 128.8 | 1364.8 | 177 | 114 | +0.024 | 0.166 |
| 13 | Albertine | 1548.9 | 184.6 | 1364.3 | 126 | 29 | +0.010 | 0.047 |
| 14 | duc de Guermantes | 1495.6 | 138.3 | 1357.3 | 97 | 56 | -0.012 | 0.061 |
| 15 | docteur Cottard | 1536.6 | 182.8 | 1353.8 | 37 | 30 | +0.057 | 0.057 |

Bottom 5, prestige:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 22 | Mme de Cambremer | 1442.0 | 195.5 | 1246.5 |
| 21 | princesse de Parme | 1475.7 | 179.9 | 1295.8 |
| 20 | Brichot | 1478.9 | 171.2 | 1307.7 |
| 19 | Françoise | 1513.9 | 198.2 | 1315.7 |
| 18 | Bloch | 1481.9 | 156.2 | 1325.7 |

### inclusion — top 15 of 9 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | le narrateur | 1601.9 | 99.8 | 1502.1 | 209 | 186 | +0.077 | 0.355 |
| 2 | duchesse de Guermantes | 1619.2 | 159.0 | 1460.2 | 183 | 45 | +0.000 | 0.000 |
| 3 | baron de Charlus | 1570.6 | 146.3 | 1424.3 | 110 | 48 | +0.011 | 0.070 |
| 4 | Gilberte | 1553.6 | 163.7 | 1389.9 | 57 | 37 | +0.050 | 0.071 |
| 5 | Robert de Saint-Loup | 1486.4 | 195.1 | 1291.3 | 138 | 25 | -0.024 | 0.024 |
| 6 | Odette | 1401.8 | 153.1 | 1248.7 | 124 | 59 | -0.094 | 0.107 |
| 7 | Bloch | 1408.6 | 171.9 | 1236.7 | 64 | 37 | -0.152 | 0.244 |
| 8 | Swann | 1346.3 | 122.3 | 1224.0 | 177 | 103 | -0.120 | 0.198 |
| 9 | Mme Verdurin | 1334.3 | 156.3 | 1178.0 | 78 | 43 | -0.055 | 0.055 |

Bottom 5, inclusion:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 9 | Mme Verdurin | 1334.3 | 156.3 | 1178.0 |
| 8 | Swann | 1346.3 | 122.3 | 1224.0 |
| 7 | Bloch | 1408.6 | 171.9 | 1236.7 |
| 6 | Odette | 1401.8 | 153.1 | 1248.7 |
| 5 | Robert de Saint-Loup | 1486.4 | 195.1 | 1291.3 |

## 6. Person view

The person view aggregates on registry entity ids with `person_view_merge` links applied, so the two era names of one man become one player; `keep_separate` links (the post-V7 princesse de Guermantes, who is Mme Verdurin holding a dead woman's title) never merge.

| lens | merged | name-view rows | person-view row | mean abs rank shift | self-pairings dropped |
| --- | --- | --- | --- | ---: | ---: |
| advantage | le-peintre -> elstir | le peintre r=1728 rank=- units=8; Elstir r=1506 rank=33 units=18 | elstir r=1560 rank=18 units=26 | 0.927 | 0 |
| advantage | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1209 rank=- units=1; duc de Guermantes r=1435 rank=32 units=97 | duc-de-guermantes r=1431 rank=33 units=98 | 0.927 | 0 |
| prestige | le-peintre -> elstir | le peintre r=1792 rank=- units=8; Elstir r=1572 rank=- units=18 | elstir r=1749 rank=- units=26 | 0.182 | 0 |
| prestige | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1548 rank=- units=1; duc de Guermantes r=1496 rank=14 units=97 | duc-de-guermantes r=1500 rank=14 units=98 | 0.182 | 0 |
| inclusion | le-peintre -> elstir | le peintre r=1660 rank=- units=8; Elstir r=1609 rank=- units=18 | elstir r=1669 rank=- units=26 | 0.0 | 0 |
| inclusion | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1500 rank=- units=1; duc de Guermantes r=1562 rank=- units=97 | duc-de-guermantes r=1563 rank=- units=98 | 0.0 | 0 |

Largest rank shifts between the two views (name-view rank minus person-view rank; the two views rank different fields, so a shift is not by itself a finding):

| lens | character | person key | name rank | person rank | shift | rating shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| advantage | Elstir | elstir | 33 | 18 | +15 | +54.6 |
| advantage | Mme Cottard | mme-cottard | 23 | 26 | -3 | -7.2 |
| advantage | Andrée | andree | 12 | 10 | +2 | +2.4 |
| advantage | Françoise | francoise | 7 | 5 | +2 | -0.3 |
| advantage | baron de Charlus | charlus | 10 | 12 | -2 | -1.3 |
| prestige | Albertine | albertine | 13 | 12 | +1 | +1.6 |
| prestige | Robert de Saint-Loup | saint-loup | 16 | 15 | +1 | +4.8 |
| prestige | Swann | swann | 12 | 13 | -1 | +0.4 |
| prestige | docteur Cottard | docteur-cottard | 15 | 16 | -1 | -0.5 |
| prestige | Bloch | bloch | 18 | 18 | +0 | +2.4 |
| inclusion | Bloch | bloch | 7 | 7 | +0 | +2.5 |
| inclusion | Gilberte | gilberte | 4 | 4 | +0 | +1.4 |
| inclusion | Mme Verdurin | mme-verdurin | 9 | 9 | +0 | +1.0 |
| inclusion | Odette | odette | 6 | 6 | +0 | +1.3 |
| inclusion | Robert de Saint-Loup | saint-loup | 5 | 5 | +0 | +3.9 |

## 7. Reading notes: where the implementation had to choose

The design doc leaves four points open; each was resolved once, in code, and is recorded here so the review can overrule it.

1. **kappa is scoped to the lens.** "The mean confidence of c's effects in u" is read as the effects that MOVE c in this lens. A character with only a `social_status` effect is therefore a zero-effect character under advantage and falls back to presence confidence there, while carrying that effect's confidence under prestige. The alternative (pooling all five dimensions into one kappa) would let a lens's weights be set by evidence that lens is defined not to see.
2. **Label precedence.** A movement past the tie band names itself first; the sign-conflict test decides only within the band. So a character with a big positive movement and one small negative effect reads positive, not mixed. Mixed still REQUIRES a genuine sign conflict, which is the clause the doc makes binding.
3. **Predictive scores are unweighted.** The WHR fits use the weights; the scoring of predictions does not, because ELO and Glicko-2 have no weight to use and a weighted loss would not be comparable to theirs.
4. **w2 is selected per lens AND per view**, independently, by the same one-step-ahead log-loss rule v1 uses.

Deferred, as the design doc says: dossier lens cards (dominant dimension, percentile), the archetype rewrite, and the person/name UI toggle -- all app-facing, all after the adoption gate. The corpus summary carries the sign triple the archetype would use.

Wall clock: 54.7 s for the validation battery; 467.602 s for the build it reads.

