# Scoring v2 validation report (staged, pre-adoption)

Corpus: enrichment, 34 runs, 963 reviewed units, 963 narrative time points. Comparisons per lens: {'advantage': 2708, 'prestige': 954, 'inclusion': 565}.

Formula: `proust/scoring_v2.py`, exactly as specified in `proust/docs/scoring_v2_design.md`. Ratings: weighted WHR (`proust/whr.py`), smoothed and filtered, on the `cumulative_unit_index` narrative axis. Everything here is staged under `outputs/scoring-v2/`; adoption is a separate reviewed decision.

w2 selected per lens/view: advantage/name = 5, advantage/person = 5, inclusion/name = 5, inclusion/person = 5, prestige/name = 5, prestige/person = 5

## 1. Lens orthogonality

The design predicts cross-lens rating correlations should FALL against v1: v1's weight tables blended every dimension into every lens, v2's projection partitions them.

| pair | v2 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.2582 (n=193) |
| advantage vs inclusion | +0.1205 (n=193) |
| prestige vs inclusion | -0.0426 (n=193) |
| **mean abs** | **0.1404** |

| pair | v1 Spearman (all rated) |
| --- | ---: |
| advantage vs prestige | +0.9852 (n=288) |
| advantage vs inclusion | +0.9897 (n=288) |
| prestige vs inclusion | +0.9736 (n=288) |
| **mean abs** | **0.9828** |

| pair | v2 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.5077 (n=14) |
| advantage vs inclusion | +0.3095 (n=8) |
| prestige vs inclusion | +0.5238 (n=8) |
| **mean abs** | **0.4470** |

| pair | v1 Spearman (non-provisional) |
| --- | ---: |
| advantage vs prestige | +0.9847 (n=91) |
| advantage vs inclusion | +0.9804 (n=91) |
| prestige vs inclusion | +0.9627 (n=91) |
| **mean abs** | **0.9759** |

**Verdict**: mean |rho| 0.983 (v1) -> 0.14 (v2): prediction held.

## 2. Bootstrap stability

50 unit-level resamples with replacement, both formulas scored on the same drawn corpora; ranks are taken over the characters both formulas rate non-provisionally on the full corpus. Lower rank standard deviation is more stable.

| lens | field | v2 mean sd | v1 mean sd | v2 median sd | v1 median sd | v2 non-prov | v1 non-prov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 31 | 5.289 | 5.83 | 5.281 | 5.366 | 31 | 91 |
| prestige | 14 | 2.355 | 2.615 | 2.148 | 2.554 | 14 | 91 |
| inclusion | 8 | 1.186 | 1.621 | 1.313 | 1.712 | 8 | 91 |

### 2b. Frequency confounding

The design's fourth principle is that frequency must not masquerade as strength. Ratings are no longer sums, so nothing accumulates with appearances -- but the standings rank by rating MINUS band, and a band narrows with evidence. Where a lens's ratings are tightly packed and its bands are not, the ranking is mostly a comparison count. Spearman rho against comparison count, over each formula's own non-provisional set:

| lens | formula | conservative vs count | rating vs count | band vs count | rating spread | band spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| advantage | v2 | 0.243 | -0.235 | -0.919 | 382.3 | 116.1 |
| advantage | v1 | 0.44 | -0.121 | -0.925 | 607.4 | 132.3 |
| prestige | v2 | 0.442 | 0.231 | -0.982 | 302.6 | 90.6 |
| prestige | v1 | 0.447 | -0.088 | -0.927 | 554.1 | 132.6 |
| inclusion | v2 | 0.311 | 0.216 | -0.97 | 255.4 | 79.6 |
| inclusion | v1 | 0.434 | -0.17 | -0.927 | 605.5 | 133.4 |

## 3. Predictive sanity

One-step-ahead over the v2 comparisons in narrative order. Every system is scored UNWEIGHTED (one comparison, one prediction): ELO and Glicko-2 have no notion of a game weight, so a weighted loss would not be comparable to theirs. The WHR fits themselves DO use the weights. Cross-formula comparison against v1's own numbers is meaningless here -- the comparisons differ -- so only the within-v2 ordering is informative.

| lens/view | system | log-loss | Brier | comparisons |
| --- | --- | ---: | ---: | ---: |
| advantage/name | whr_filtered | 0.703012 | 0.252055 | 2708 |
| advantage/name | whr_filtered_deflated | 0.689635 | 0.247186 | 2708 |
| advantage/name | elo_sequential | 0.654221 | 0.231186 | 2708 |
| advantage/name | elo_unit_frozen | 0.684708 | 0.245236 | 2708 |
| advantage/name | glicko2_chapter_period | 0.714807 | 0.256846 | 2708 |
| advantage/person | whr_filtered | 0.703215 | 0.252083 | 2708 |
| advantage/person | whr_filtered_deflated | 0.689698 | 0.247181 | 2708 |
| advantage/person | elo_sequential | 0.654164 | 0.231186 | 2708 |
| advantage/person | elo_unit_frozen | 0.684676 | 0.245242 | 2708 |
| advantage/person | glicko2_chapter_period | 0.713018 | 0.256112 | 2708 |
| prestige/name | whr_filtered | 0.750464 | 0.270556 | 954 |
| prestige/name | whr_filtered_deflated | 0.721365 | 0.260737 | 954 |
| prestige/name | elo_sequential | 0.644968 | 0.227261 | 954 |
| prestige/name | elo_unit_frozen | 0.689898 | 0.248206 | 954 |
| prestige/name | glicko2_chapter_period | 0.798204 | 0.285332 | 954 |
| prestige/person | whr_filtered | 0.749899 | 0.270404 | 954 |
| prestige/person | whr_filtered_deflated | 0.720930 | 0.260599 | 954 |
| prestige/person | elo_sequential | 0.644850 | 0.227199 | 954 |
| prestige/person | elo_unit_frozen | 0.689767 | 0.248130 | 954 |
| prestige/person | glicko2_chapter_period | 0.797737 | 0.285161 | 954 |
| inclusion/name | whr_filtered | 0.742116 | 0.263963 | 565 |
| inclusion/name | whr_filtered_deflated | 0.711925 | 0.254816 | 565 |
| inclusion/name | elo_sequential | 0.645951 | 0.226850 | 565 |
| inclusion/name | elo_unit_frozen | 0.690794 | 0.248223 | 565 |
| inclusion/name | glicko2_chapter_period | 0.763966 | 0.273191 | 565 |
| inclusion/person | whr_filtered | 0.741314 | 0.263556 | 565 |
| inclusion/person | whr_filtered_deflated | 0.711164 | 0.254473 | 565 |
| inclusion/person | elo_sequential | 0.645817 | 0.226783 | 565 |
| inclusion/person | elo_unit_frozen | 0.690647 | 0.248150 | 565 |
| inclusion/person | glicko2_chapter_period | 0.763996 | 0.273228 | 565 |

### w2 selection

| lens/view | w2 | log-loss |
| --- | ---: | ---: |
| advantage/name | 5 **(selected)** | 0.703012 |
| advantage/name | 15 | 0.703194 |
| advantage/name | 35 | 0.704356 |
| advantage/name | 60 | 0.706151 |
| advantage/person | 5 **(selected)** | 0.703215 |
| advantage/person | 15 | 0.703319 |
| advantage/person | 35 | 0.704374 |
| advantage/person | 60 | 0.706086 |
| prestige/name | 5 **(selected)** | 0.750464 |
| prestige/name | 15 | 0.751280 |
| prestige/name | 35 | 0.754418 |
| prestige/name | 60 | 0.758920 |
| prestige/person | 5 **(selected)** | 0.749899 |
| prestige/person | 15 | 0.750766 |
| prestige/person | 35 | 0.753978 |
| prestige/person | 60 | 0.758546 |
| inclusion/name | 5 **(selected)** | 0.742116 |
| inclusion/name | 15 | 0.743318 |
| inclusion/name | 35 | 0.746319 |
| inclusion/name | 60 | 0.750540 |
| inclusion/person | 5 **(selected)** | 0.741314 |
| inclusion/person | 15 | 0.742537 |
| inclusion/person | 35 | 0.745573 |
| inclusion/person | 60 | 0.749832 |

## 4. Literary panel (pre-registered)

Each claim comes from the design doc; each operationalization was fixed before the ratings were read. Name view, and the standings referred to are the non-provisional set.

**6/8 claims pass.**

| claim | verdict |
| --- | --- |
| the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens | PASS |
| Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all) | PASS |
| Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set | PASS |
| Odette's prestige above her inclusion: prestige rating > inclusion rating | PASS |
| Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4 | PASS |
| the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median | FAIL |
| Saniette last or near it: bottom 10% of the non-provisional advantage set | FAIL |
| l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons | PASS |

### duchesse — PASS

the duchesse de Guermantes's standing among the corpus elite: non-provisional and ranked in the top 10% of the non-provisional set in at least one lens

| lens | rating | band | rank | non provisional count | rank percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| advantage | 1601.6 | 93.1 | 1 | 31 | 0.032 |
| prestige | 1706.2 | 108 | 1 | 14 | 0.071 |
| inclusion | 1612.7 | 161.6 | 2 | 8 | 0.25 |

### rachel — PASS

Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at least one lens (the closed-world corpus could not see her at all)

| lens | rating | band | rank | unit count | comparison count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1600.4 | 173.4 | 8 | 29 | 56 | False |
| prestige | 1639.4 | 224.2 | - | 29 | 24 | True |
| inclusion | 1647.6 | 462.3 | - | 29 | 6 | True |

### bloch — PASS

Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set

| lens | rating | rank | rank percentile | non provisional count | provisional |
| --- | ---: | ---: | ---: | ---: | ---: |
| inclusion | 1411.6 | 6 | 0.75 | 8 | False |

### odette — PASS

Odette's prestige above her inclusion: prestige rating > inclusion rating

| lens | rating | rank | mean movement |
| --- | ---: | ---: | ---: |
| prestige | 1710.2 | 3 | 0.107 |
| inclusion | 1416.7 | 5 | -0.094 |

### charlus — PASS

Charlus's trajectory declining across the late volumes: mean smoothed advantage rating over volumes 5-7 below the mean over volumes 1-4

| lens | first rating | last rating | early volume mean | late volume mean | early node count | late node count | rating | rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1548.5 | 1501.5 | 1527.7 | 1503.2 | 68 | 33 | 1501.5 | 13 |
| prestige | 1561.4 | 1550.2 | 1558.3 | 1551.9 | 28 | 21 | 1550.2 | 7 |
| inclusion | 1572.2 | 1561.9 | 1569.2 | 1562.6 | 26 | 9 | 1561.9 | 3 |

### narrator — FAIL

the narrator mid-table with a tight band: advantage rank in the middle third of the non-provisional set, band below its median

| lens | rating | band | rank | rank percentile | median band | unit count | comparison count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1513.3 | 82.5 | 7 | 0.226 | 143.9 | 209 | 399 |
| prestige | 1648.4 | - | 4 | - | - | - | - |
| inclusion | 1597.8 | - | 1 | - | - | - | - |

### saniette — FAIL

Saniette last or near it: bottom 10% of the non-provisional advantage set

| lens | rating | band | rank | rank percentile | provisional | non provisional count | mean movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 1310.7 | 239.1 | - | - | True | 31 | -0.846 |

### amie — PASS

l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons

| lens | unit count | comparison count | rating | band | rank | provisional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | 7 | 17 | 1736 | 300.4 | - | True |

## 5. Headline standings (name view, non-provisional)

### advantage — top 15 of 31 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | duchesse de Guermantes | 1601.6 | 93.1 | 1508.5 | 183 | 354 | +0.049 | 0.490 |
| 2 | comte de Forcheville | 1699.1 | 193.8 | 1505.3 | 28 | 62 | +0.139 | 0.181 |
| 3 | M. Verdurin | 1646.9 | 160.2 | 1486.7 | 32 | 84 | -0.176 | 0.267 |
| 4 | Mme de Villeparisis | 1595.0 | 133.6 | 1461.4 | 73 | 118 | -0.077 | 0.369 |
| 5 | Françoise | 1581.1 | 133.0 | 1448.1 | 61 | 100 | +0.086 | 0.586 |
| 6 | la mère du narrateur | 1604.6 | 168.7 | 1435.9 | 28 | 54 | +0.273 | 0.318 |
| 7 | le narrateur | 1513.3 | 82.5 | 1430.8 | 209 | 399 | -0.201 | 0.632 |
| 8 | Rachel | 1600.4 | 173.4 | 1427.0 | 29 | 56 | -0.216 | 0.609 |
| 9 | la grand-mère | 1563.2 | 144.6 | 1418.6 | 48 | 74 | +0.177 | 0.667 |
| 10 | Mme Verdurin | 1537.2 | 121.8 | 1415.4 | 78 | 181 | -0.336 | 0.425 |
| 11 | le père du narrateur | 1613.7 | 198.6 | 1415.1 | 21 | 44 | +0.078 | 0.287 |
| 12 | Albertine | 1503.7 | 90.4 | 1413.3 | 126 | 183 | -0.203 | 0.744 |
| 13 | baron de Charlus | 1501.5 | 91.5 | 1410.0 | 110 | 283 | -0.256 | 0.706 |
| 14 | docteur Cottard | 1555.2 | 148.1 | 1407.1 | 37 | 107 | -0.165 | 0.713 |
| 15 | Odette | 1519.8 | 119.5 | 1400.3 | 124 | 248 | -0.081 | 0.503 |

Bottom 5, advantage:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 31 | Mme de Cambremer | 1317.2 | 191.7 | 1125.5 |
| 30 | Bloch | 1316.8 | 129.1 | 1187.7 |
| 29 | prince de Guermantes | 1463.5 | 192.9 | 1270.6 |
| 28 | Mme de Marsantes | 1480.1 | 176.2 | 1303.9 |
| 27 | duc de Guermantes | 1422.9 | 104.9 | 1318.0 |

### prestige — top 15 of 14 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | duchesse de Guermantes | 1706.2 | 108.0 | 1598.2 | 183 | 156 | +0.216 | 0.270 |
| 2 | Morel | 1769.8 | 187.7 | 1582.1 | 35 | 43 | +0.206 | 0.206 |
| 3 | Odette | 1710.2 | 137.0 | 1573.2 | 124 | 77 | +0.107 | 0.169 |
| 4 | le narrateur | 1648.4 | 127.5 | 1520.9 | 209 | 102 | +0.061 | 0.100 |
| 5 | Gilberte | 1630.2 | 154.1 | 1476.1 | 57 | 53 | +0.085 | 0.174 |
| 6 | Mme Verdurin | 1586.5 | 127.4 | 1459.1 | 78 | 104 | +0.129 | 0.236 |
| 7 | baron de Charlus | 1550.2 | 110.2 | 1440.0 | 110 | 132 | +0.032 | 0.269 |
| 8 | Mme de Villeparisis | 1561.1 | 148.3 | 1412.8 | 73 | 58 | -0.016 | 0.205 |
| 9 | Robert de Saint-Loup | 1553.3 | 144.0 | 1409.3 | 138 | 74 | +0.047 | 0.116 |
| 10 | Swann | 1529.6 | 134.3 | 1395.3 | 177 | 114 | +0.024 | 0.166 |
| 11 | Norpois | 1544.8 | 196.3 | 1348.5 | 54 | 38 | +0.101 | 0.127 |
| 12 | Brichot | 1532.9 | 198.6 | 1334.3 | 17 | 32 | -0.008 | 0.352 |
| 13 | Bloch | 1478.7 | 173.0 | 1305.7 | 64 | 45 | -0.046 | 0.093 |
| 14 | duc de Guermantes | 1467.2 | 167.5 | 1299.7 | 97 | 56 | -0.012 | 0.061 |

Bottom 5, prestige:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 14 | duc de Guermantes | 1467.2 | 167.5 | 1299.7 |
| 13 | Bloch | 1478.7 | 173.0 | 1305.7 |
| 12 | Brichot | 1532.9 | 198.6 | 1334.3 |
| 11 | Norpois | 1544.8 | 196.3 | 1348.5 |
| 10 | Swann | 1529.6 | 134.3 | 1395.3 |

### inclusion — top 15 of 8 non-provisional (193 rated)

| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | le narrateur | 1597.8 | 101.1 | 1496.7 | 209 | 186 | +0.077 | 0.355 |
| 2 | duchesse de Guermantes | 1612.7 | 161.6 | 1451.1 | 183 | 45 | +0.000 | 0.000 |
| 3 | baron de Charlus | 1561.9 | 155.3 | 1406.6 | 110 | 48 | +0.011 | 0.070 |
| 4 | Gilberte | 1516.9 | 180.7 | 1336.2 | 57 | 37 | +0.050 | 0.071 |
| 5 | Odette | 1416.7 | 157.0 | 1259.7 | 124 | 59 | -0.094 | 0.107 |
| 6 | Bloch | 1411.6 | 174.2 | 1237.4 | 64 | 37 | -0.152 | 0.244 |
| 7 | Swann | 1358.8 | 124.8 | 1234.0 | 177 | 103 | -0.120 | 0.198 |
| 8 | Mme Verdurin | 1357.3 | 168.3 | 1189.0 | 78 | 43 | -0.055 | 0.055 |

Bottom 5, inclusion:

| rank | character | rating | band | conservative |
| ---: | --- | ---: | ---: | ---: |
| 8 | Mme Verdurin | 1357.3 | 168.3 | 1189.0 |
| 7 | Swann | 1358.8 | 124.8 | 1234.0 |
| 6 | Bloch | 1411.6 | 174.2 | 1237.4 |
| 5 | Odette | 1416.7 | 157.0 | 1259.7 |
| 4 | Gilberte | 1516.9 | 180.7 | 1336.2 |

## 6. Person view

The person view aggregates on registry entity ids with `person_view_merge` links applied, so the two era names of one man become one player; `keep_separate` links (the post-V7 princesse de Guermantes, who is Mme Verdurin holding a dead woman's title) never merge.

| lens | merged | name-view rows | person-view row | mean abs rank shift | self-pairings dropped |
| --- | --- | --- | --- | ---: | ---: |
| advantage | le-peintre -> elstir | le peintre r=1684 rank=- units=8; Elstir r=1538 rank=- units=18 | elstir r=1584 rank=16 units=26 | 0.71 | 0 |
| advantage | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1308 rank=- units=1; duc de Guermantes r=1423 rank=27 units=97 | duc-de-guermantes r=1420 rank=28 units=98 | 0.71 | 0 |
| prestige | le-peintre -> elstir | le peintre r=1805 rank=- units=8; Elstir r=1596 rank=- units=18 | elstir r=1766 rank=- units=26 | 0.143 | 0 |
| prestige | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1547 rank=- units=1; duc de Guermantes r=1467 rank=14 units=97 | duc-de-guermantes r=1473 rank=14 units=98 | 0.143 | 0 |
| inclusion | le-peintre -> elstir | le peintre r=1664 rank=- units=8; Elstir r=1599 rank=- units=18 | elstir r=1662 rank=- units=26 | 0.0 | 0 |
| inclusion | prince-des-laumes -> duc-de-guermantes | prince des Laumes r=1500 rank=- units=1; duc de Guermantes r=1565 rank=- units=97 | duc-de-guermantes r=1566 rank=- units=98 | 0.0 | 0 |

Largest rank shifts between the two views (name-view rank minus person-view rank; the two views rank different fields, so a shift is not by itself a finding):

| lens | character | person key | name rank | person rank | shift | rating shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| advantage | Mme Verdurin | mme-verdurin | 10 | 13 | -3 | -10.5 |
| advantage | Albertine | albertine | 12 | 10 | +2 | +1.3 |
| advantage | Brichot | brichot | 22 | 24 | -2 | -2.1 |
| advantage | Swann | swann | 19 | 21 | -2 | -1.8 |
| advantage | Andrée | andree | 18 | 19 | -1 | +1.7 |
| prestige | Mme de Villeparisis | mme-de-villeparisis | 8 | 9 | -1 | +2.8 |
| prestige | Robert de Saint-Loup | saint-loup | 9 | 8 | +1 | +6.6 |
| prestige | Bloch | bloch | 13 | 13 | +0 | +2.9 |
| prestige | Brichot | brichot | 12 | 12 | +0 | +0.9 |
| prestige | Gilberte | gilberte | 5 | 5 | +0 | +2.3 |
| inclusion | Bloch | bloch | 6 | 6 | +0 | +2.8 |
| inclusion | Gilberte | gilberte | 4 | 4 | +0 | +1.4 |
| inclusion | Mme Verdurin | mme-verdurin | 8 | 8 | +0 | +1.0 |
| inclusion | Odette | odette | 5 | 5 | +0 | +1.3 |
| inclusion | Swann | swann | 7 | 7 | +0 | +1.2 |

## 7. Reading notes: where the implementation had to choose

The design doc leaves four points open; each was resolved once, in code, and is recorded here so the review can overrule it.

1. **kappa is scoped to the lens.** "The mean confidence of c's effects in u" is read as the effects that MOVE c in this lens. A character with only a `social_status` effect is therefore a zero-effect character under advantage and falls back to presence confidence there, while carrying that effect's confidence under prestige. The alternative (pooling all five dimensions into one kappa) would let a lens's weights be set by evidence that lens is defined not to see.
2. **Label precedence.** A movement past the tie band names itself first; the sign-conflict test decides only within the band. So a character with a big positive movement and one small negative effect reads positive, not mixed. Mixed still REQUIRES a genuine sign conflict, which is the clause the doc makes binding.
3. **Predictive scores are unweighted.** The WHR fits use the weights; the scoring of predictions does not, because ELO and Glicko-2 have no weight to use and a weighted loss would not be comparable to theirs.
4. **w2 is selected per lens AND per view**, independently, by the same one-step-ahead log-loss rule v1 uses.

Deferred, as the design doc says: dossier lens cards (dominant dimension, percentile), the archetype rewrite, and the person/name UI toggle -- all app-facing, all after the adoption gate. The corpus summary carries the sign triple the archetype would use.

Wall clock: 53.5 s for the validation battery; 431.611 s for the build it reads.

