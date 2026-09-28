# Fortune arcs

Each character's own arc: a smoothed level of their scoring v2 movements, passage by
passage, from `proust/fortune.py`, in the PERSON view (registry entities; names and titles
of one person pooled). Drift and noise are chosen once per lens by pooled marginal
likelihood; nothing is tuned per character.

Ratings are Elo-style: 1500 is a level of 0 (passages leave the character where they were),
and the points-per-level factor comes from the fitted passage noise, so a gap of D points
means one character's next passage goes better than the other's about as often as a
D-point Elo favourite wins (100 points ≈ 64%, 200 ≈ 76%, 400 ≈ 91%). `±` is one posterior
standard deviation.

`order p` asks whether the ORDER of a character's passages made the arc: the share of
random reshufflings of their outcomes in time that give an equal or bigger move. Around
40 characters are tested per lens, so a handful under 0.05 are expected by chance; lean on
the ones near 0.01.

`arc evidence` is how much better (in log-likelihood) the fitted model explains the
movements than one in which no character's fortune ever changes.

| lens | observations | people | q | sigma² | points per level | arc evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 1874 | 168 | 0.016 | 0.9 | 220 | 32.4 |
| advantage | 1610 | 151 | 0.004 | 0.7 | 250 | 11.9 |
| prestige | 356 | 78 | 0.008 | 0.7 | 250 | 7.7 |
| inclusion | 200 | 52 | 0.004 | 0.9 | 220 | 0.5 |

## Person view rulings

- merged by `person_view_merge`: le peintre → Elstir, prince des Laumes → duc de Guermantes
- reviewed passage ruling: "princesse de Guermantes" in `v7-p4-le-bal-de-tetes#p-61-p-65` → Mme Verdurin
- no names left ambiguous by chapter-scoped registry forms

## overall

People with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 89 | 0.001 | 1627 ± 68 → 1149 ± 73 (479 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| la Berma | 13 | 0.051 | 1657 ± 83 → 1300 ± 114 (358 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Temps retrouvé — IV. Le Bal de têtes |
| Swann | 147 | 0.033 | 1472 ± 57 → 1185 ± 78 (287 pts) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Saniette | 11 | 0.200 | 1314 ± 79 → 1049 ± 126 (265 pts) · Du Côté de Chez Swann — II. Un amour de Swann → La Prisonnière |
| Albertine | 103 | 0.005 | 1590 ± 60 → 1334 ± 53 (256 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → La Prisonnière |
| duchesse de Guermantes | 132 | 0.016 | 1638 ± 60 → 1395 ± 82 (243 pts) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 182 | 0.033 | 1589 ± 45 → 1365 ± 53 (224 pts) · Le Côté de Guermantes — II → La Prisonnière |
| duc de Guermantes | 64 | 0.058 | 1379 ± 86 → 1160 ± 98 (219 pts) · Du Côté de Chez Swann — II. Un amour de Swann → La Prisonnière |
| marquise de Saint-Euverte | 8 | 0.511 | 1349 ± 91 → 1151 ± 88 (198 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Sodome et Gomorrhe — II |
| Brichot | 15 | 0.145 | 1512 ± 82 → 1333 ± 132 (179 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — II. M. de Charlus pendant la guerre |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| M. Vinteuil | 8 | 0.004 | 1449 ± 72 → 1917 ± 180 (468 pts) · Du Côté de Chez Swann — I. Combray → La Prisonnière |
| Mme Bontemps | 8 | 0.008 | 1427 ± 84 → 1701 ± 138 (274 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| la grand-mère | 36 | 0.043 | 1465 ± 71 → 1682 ± 106 (216 pts) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Morel | 33 | 0.019 | 1299 ± 62 → 1497 ± 101 (198 pts) · La Prisonnière → Le Temps retrouvé — IV. Le Bal de têtes |
| marquise de Saint-Euverte | 8 | 0.035 | 1151 ± 88 → 1345 ± 167 (194 pts) · Sodome et Gomorrhe — II → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| Robert de Saint-Loup | 105 | 0.075 | 1410 ± 40 → 1584 ± 97 (174 pts) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| Jupien | 9 | 0.562 | 1589 ± 82 → 1759 ± 93 (170 pts) · Le Côté de Guermantes — I → Sodome et Gomorrhe — I |
| Andrée | 19 | 0.201 | 1420 ± 86 → 1561 ± 139 (140 pts) · La Prisonnière → Le Temps retrouvé — IV. Le Bal de têtes |
| Mme Verdurin | 49 | 0.117 | 1370 ± 74 → 1509 ± 83 (139 pts) · Sodome et Gomorrhe — II → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Françoise | 39 | 0.429 | 1504 ± 70 → 1641 ± 74 (137 pts) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |


## advantage

People with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 79 | 0.001 | 1549 ± 61 → 1228 ± 63 (321 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| Gilberte | 30 | 0.002 | 1575 ± 61 → 1375 ± 79 (200 pts) · Du Côté de Chez Swann — III. Noms de pays : le nom → Le Temps retrouvé — IV. Le Bal de têtes |
| Albertine | 100 | 0.003 | 1539 ± 50 → 1352 ± 56 (187 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Albertine disparue — III |
| Odette | 68 | 0.042 | 1491 ± 37 → 1338 ± 80 (152 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Albertine disparue — IV |
| duchesse de Guermantes | 111 | 0.038 | 1554 ± 40 → 1420 ± 74 (134 pts) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| Saniette | 10 | 0.220 | 1327 ± 80 → 1228 ± 83 (99 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Sodome et Gomorrhe — II |
| Andrée | 18 | 0.110 | 1510 ± 66 → 1424 ± 75 (86 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → La Prisonnière |
| Swann | 132 | 0.277 | 1416 ± 48 → 1331 ± 87 (85 pts) · Le Côté de Guermantes — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Norpois | 34 | 0.122 | 1498 ± 55 → 1414 ± 81 (85 pts) · Le Côté de Guermantes — I → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Mme de Marsantes | 13 | 0.041 | 1366 ± 78 → 1281 ± 120 (84 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Albertine disparue — IV |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| M. Vinteuil | 8 | 0.003 | 1460 ± 74 → 1616 ± 135 (156 pts) · Du Côté de Chez Swann — I. Combray → La Prisonnière |
| la grand-mère | 31 | 0.040 | 1491 ± 64 → 1642 ± 79 (151 pts) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Robert de Saint-Loup | 97 | 0.079 | 1421 ± 32 → 1538 ± 76 (116 pts) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 134 | 0.437 | 1403 ± 41 → 1470 ± 47 (67 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Sodome et Gomorrhe — II |
| le père du narrateur | 9 | 0.176 | 1497 ± 85 → 1564 ± 114 (66 pts) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| M. Verdurin | 13 | 0.021 | 1381 ± 70 → 1443 ± 111 (63 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Elstir | 19 | 0.639 | 1566 ± 71 → 1623 ± 62 (57 pts) · Du Côté de Chez Swann — II. Un amour de Swann → À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays |
| Françoise | 37 | 0.626 | 1514 ± 63 → 1570 ± 58 (56 pts) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |
| docteur Cottard | 29 | 0.342 | 1430 ± 50 → 1485 ± 113 (55 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — I. A Tansonville |
| Morel | 27 | 0.090 | 1231 ± 55 → 1284 ± 75 (52 pts) · La Prisonnière → Le Temps retrouvé — II. M. de Charlus pendant la guerre |


## prestige

People with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 28 | 0.002 | 1649 ± 80 → 1357 ± 98 (293 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — IV. Le Bal de têtes |
| duchesse de Guermantes | 42 | 0.003 | 1785 ± 51 → 1548 ± 97 (237 pts) · Le Côté de Guermantes — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Swann | 32 | 0.120 | 1583 ± 60 → 1422 ± 157 (161 pts) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Robert de Saint-Loup | 18 | 0.103 | 1650 ± 77 → 1523 ± 111 (127 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Albertine disparue — IV |
| le narrateur | 22 | 0.065 | 1681 ± 70 → 1557 ± 123 (124 pts) · Le Côté de Guermantes — II → Albertine disparue — I |
| Mme de Villeparisis | 16 | 0.320 | 1545 ± 74 → 1492 ± 94 (54 pts) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Côté de Guermantes — II |
| Odette | 18 | 0.661 | 1746 ± 99 → 1714 ± 144 (32 pts) · Sodome et Gomorrhe — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Gilberte | 8 | 0.479 | 1631 ± 93 → 1602 ± 110 (29 pts) · Albertine disparue — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Mme Verdurin | 17 | 0.732 | 1656 ± 92 → 1634 ± 82 (23 pts) · Sodome et Gomorrhe — II → La Prisonnière |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Odette | 18 | 0.511 | 1646 ± 79 → 1746 ± 99 (100 pts) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Mme Verdurin | 17 | 0.349 | 1607 ± 80 → 1703 ± 108 (96 pts) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — IV. Le Bal de têtes |
| duchesse de Guermantes | 42 | 0.669 | 1705 ± 79 → 1785 ± 51 (80 pts) · Du Côté de Chez Swann — I. Combray → Le Côté de Guermantes — II |
| le narrateur | 22 | 0.602 | 1638 ± 77 → 1681 ± 70 (43 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Côté de Guermantes — II |
| Gilberte | 8 | 0.830 | 1619 ± 100 → 1631 ± 93 (12 pts) · Du Côté de Chez Swann — III. Noms de pays : le nom → Albertine disparue — II |
| Robert de Saint-Loup | 18 | 0.931 | 1523 ± 111 → 1530 ± 119 (8 pts) · Albertine disparue — IV → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Mme de Villeparisis | 16 | 0.704 | 1501 ± 67 → 1505 ± 70 (4 pts) · Le Côté de Guermantes — I → Le Côté de Guermantes — I |
| Swann | 32 | 0.974 | 1571 ± 54 → 1573 ± 52 (2 pts) · Du Côté de Chez Swann — I. Combray → Du Côté de Chez Swann — II. Un amour de Swann |
| baron de Charlus | 28 | 0.994 | 1607 ± 69 → 1607 ± 68 (0 pts) · Sodome et Gomorrhe — II → Sodome et Gomorrhe — II |


## inclusion

People with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Swann | 28 | 0.088 | 1372 ± 56 → 1261 ± 86 (111 pts) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Odette | 9 | 0.201 | 1314 ± 71 → 1220 ± 127 (94 pts) · Du Côté de Chez Swann — I. Combray → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 63 | 0.171 | 1571 ± 43 → 1518 ± 73 (53 pts) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Sodome et Gomorrhe — II |
| Bloch | 13 | 0.982 | 1339 ± 68 → 1338 ± 69 (1 pts) · Le Côté de Guermantes — I → Le Côté de Guermantes — I |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Bloch | 13 | 0.045 | 1327 ± 71 → 1412 ± 109 (85 pts) · Du Côté de Chez Swann — I. Combray → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 63 | 0.856 | 1541 ± 59 → 1571 ± 43 (30 pts) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |

