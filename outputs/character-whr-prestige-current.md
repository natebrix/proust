# Character Whole-History Rating

- Analysis version: `character_whr_prestige_v1`
- Lens: `prestige`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `288`
- Match count: `5756`
- Time point count: `840`
- Node count: `3007`
- Draw rate: `0.322`
- Draw model: `half_win_half_loss`
- w2: `15.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.83`s, filtered `130.411`s (all w2 candidates `591.865`s)
- Convergence: smoothed `29` sweeps (converged: `True`), filtered `840` fits / `13184` sweeps, `0` of them unconverged
- Corpus: `foundation`

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.722023 | 0.259189 | 5756 | filtered WHR at w2=15 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.711012 | 0.255895 | 5756 | filtered WHR at w2=15, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.658562 | 0.233349 | 5756 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.697635 | 0.251293 | 5756 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.728281 | 0.263611 | 5756 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.722288 | 0.25951 | 98.936 |
| 15.0 | 0.722023 | 0.259189 | 130.411 |
| 35.0 | 0.723711 | 0.259618 | 164.682 |
| 60.0 | 0.726215 | 0.260369 | 197.836 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Prestige |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docteur du Boulbon | 1770 ± 188 | 1581.9 | 188.0 | 27 | 19-3-5 | 6 | 6 | -0.435 |
| le peintre | 1641 ± 118 | 1523.2 | 118.2 | 42 | 16-4-22 | 8 | 8 | -0.296 |
| Bergotte | 1645 ± 127 | 1517.7 | 127.4 | 129 | 51-31-47 | 36 | 32 | -0.094 |
| Françoise | 1615 ± 108 | 1507.3 | 107.7 | 217 | 99-50-68 | 82 | 76 | -0.267 |
| Rachel | 1596 ± 93 | 1503.1 | 92.9 | 146 | 52-49-45 | 43 | 43 | -0.939 |
| Aimé | 1580 ± 99 | 1481.2 | 99.1 | 79 | 27-13-39 | 18 | 18 | -0.472 |
| l'amie de Mlle Vinteuil | 1606 ± 128 | 1478.4 | 127.6 | 44 | 17-6-21 | 12 | 12 | -0.346 |
| Rémi | 1650 ± 174 | 1475.7 | 173.9 | 17 | 5-0-12 | 3 | 3 | -0.533 |
| Mme Verdurin | 1555 ± 82 | 1473.1 | 81.7 | 311 | 100-98-113 | 82 | 82 | -0.738 |
| Elstir | 1569 ± 100 | 1469.7 | 99.7 | 106 | 39-31-36 | 29 | 24 | +0.034 |
| Morel | 1546 ± 84 | 1461.6 | 84.3 | 152 | 48-51-53 | 32 | 31 | -0.876 |
| Jupien | 1556 ± 96 | 1459.8 | 95.9 | 68 | 23-13-32 | 18 | 18 | +0.031 |
| Mlle Vinteuil | 1559 ± 100 | 1459.3 | 100.1 | 71 | 21-15-35 | 15 | 15 | -0.665 |
| le père du narrateur | 1593 ± 136 | 1456.7 | 136.0 | 90 | 34-21-35 | 24 | 24 | -0.68 |
| la grand-mère | 1565 ± 109 | 1456.5 | 108.8 | 225 | 94-67-64 | 80 | 74 | -0.29 |
| M. Verdurin | 1558 ± 104 | 1453.9 | 103.6 | 110 | 36-25-49 | 27 | 27 | -0.64 |
| Bloch | 1528 ± 81 | 1446.5 | 81.0 | 270 | 78-112-80 | 71 | 70 | -1.407 |
| la mère du narrateur | 1532 ± 97 | 1435.5 | 97.0 | 144 | 55-36-53 | 40 | 40 | -0.421 |
| prince de Guermantes | 1552 ± 116 | 1435.2 | 116.4 | 124 | 41-27-56 | 22 | 22 | -0.815 |
| Mme Cottard | 1625 ± 190 | 1434.7 | 190.3 | 33 | 17-7-9 | 11 | 11 | -0.383 |
| Norpois | 1568 ± 134 | 1433.8 | 133.7 | 180 | 79-53-48 | 63 | 62 | -0.592 |
| Robert de Saint-Loup | 1505 ± 73 | 1431.8 | 72.9 | 508 | 160-212-136 | 168 | 154 | -0.548 |
| Odette | 1509 ± 80 | 1429.7 | 79.6 | 462 | 164-168-130 | 142 | 138 | -0.625 |
| Mme Sazerat | 1582 ± 160 | 1421.9 | 160.0 | 20 | 7-2-11 | 6 | 6 | -0.673 |
| Mme de Charlus | 1611 ± 189 | 1421.5 | 189.4 | 15 | 5-1-9 | 2 | 2 | -0.8 |
| marquis de Bréauté | 1529 ± 110 | 1418.6 | 110.0 | 101 | 26-22-53 | 19 | 19 | -0.931 |
| le grand-père du narrateur | 1583 ± 167 | 1415.9 | 166.9 | 63 | 26-7-30 | 16 | 16 | -0.612 |
| Andrée | 1503 ± 90 | 1413.2 | 90.0 | 114 | 37-42-35 | 31 | 31 | -0.712 |
| Mme de Surgis | 1545 ± 132 | 1413.0 | 132.2 | 42 | 16-11-15 | 9 | 9 | -0.802 |
| Dreyfus | 1523 ± 111 | 1412.7 | 110.6 | 58 | 13-11-34 | 7 | 7 | -0.77 |
| duchesse de Guermantes | 1479 ± 68 | 1411.2 | 68.2 | 662 | 331-180-151 | 199 | 194 | -0.076 |
| comte de Forcheville | 1517 ± 112 | 1404.9 | 112.3 | 112 | 56-19-37 | 25 | 25 | -0.29 |
| Mme Goupil | 1573 ± 170 | 1402.9 | 169.9 | 17 | 5-1-11 | 2 | 2 | -0.8 |
| la marquise douairière de Cambremer | 1532 ± 129 | 1402.9 | 129.4 | 31 | 10-5-16 | 6 | 6 | +0.083 |
| le narrateur | 1467 ± 65 | 1401.8 | 64.9 | 1093 | 403-491-199 | 316 | 315 | -0.718 |
| docteur Cottard | 1493 ± 102 | 1391.2 | 101.8 | 194 | 46-64-84 | 43 | 43 | -0.865 |
| Charcot | 1587 ± 197 | 1389.4 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| M. Reinach | 1587 ± 197 | 1389.4 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| Mme de Villeparisis | 1499 ± 111 | 1388.2 | 111.1 | 236 | 89-92-55 | 79 | 78 | -0.637 |
| Mme Leroi | 1581 ± 195 | 1386.7 | 194.7 | 13 | 8-5-0 | 5 | 5 | -1.092 |
| Gilberte | 1456 ± 70 | 1385.2 | 70.5 | 312 | 112-104-96 | 76 | 74 | -0.457 |
| M. Vinteuil | 1505 ± 122 | 1383.3 | 122.1 | 61 | 18-19-24 | 15 | 15 | -0.422 |
| Albertine | 1461 ± 78 | 1383.1 | 77.8 | 387 | 149-156-82 | 146 | 126 | -0.778 |
| marquise de Saint-Euverte | 1501 ± 118 | 1382.7 | 118.5 | 72 | 16-29-27 | 13 | 13 | -1.982 |
| Mme de Sévigné | 1544 ± 162 | 1381.2 | 162.3 | 25 | 7-5-13 | 4 | 4 | -0.019 |
| Mme de Marsantes | 1482 ± 102 | 1379.5 | 102.0 | 107 | 20-31-56 | 21 | 20 | -1.001 |
| Swann | 1451 ± 72 | 1379.3 | 71.8 | 667 | 212-305-150 | 202 | 198 | -0.817 |
| baron de Charlus | 1449 ± 71 | 1377.3 | 71.4 | 485 | 193-159-133 | 119 | 118 | -0.723 |
| Legrandin | 1481 ± 104 | 1376.6 | 104.2 | 83 | 16-27-40 | 24 | 20 | -1.178 |
| M. Ski | 1532 ± 156 | 1376.3 | 156.1 | 21 | 4-1-16 | 2 | 2 | -0.4 |
| Brichot | 1464 ± 88 | 1376.2 | 88.0 | 135 | 28-32-75 | 21 | 21 | -0.777 |
| M. de Chevregny | 1549 ± 173 | 1376.1 | 173.3 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| M. de Crécy | 1549 ± 173 | 1376.1 | 173.3 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| Mme Féré | 1549 ± 173 | 1376.1 | 173.3 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| général de Froberville | 1526 ± 156 | 1370.3 | 156.1 | 27 | 7-4-16 | 7 | 7 | -0.589 |
| prince des Laumes | 1514 ± 147 | 1366.7 | 147.2 | 27 | 4-3-20 | 3 | 3 | -0.8 |
| M. Nissim Bernard | 1495 ± 134 | 1361.6 | 133.7 | 39 | 8-10-21 | 10 | 7 | -1.315 |
| tante Léonie | 1522 ± 161 | 1361.0 | 161.0 | 38 | 11-22-5 | 22 | 20 | -0.717 |
| Bloch père | 1489 ± 131 | 1357.9 | 131.3 | 47 | 10-12-25 | 8 | 8 | -1.771 |
| Esther | 1537 ± 181 | 1356.3 | 181.1 | 14 | 3-2-9 | 2 | 2 | -1.0 |
| princesse de Luxembourg | 1508 ± 152 | 1355.7 | 152.0 | 25 | 6-7-12 | 6 | 6 | -0.773 |
| Mme Bontemps | 1480 ± 125 | 1354.4 | 125.4 | 54 | 13-13-28 | 13 | 13 | -0.643 |
| princesse de Guermantes | 1466 ± 113 | 1353.3 | 112.6 | 113 | 42-30-41 | 25 | 25 | -0.252 |
| duc de Chartres | 1532 ± 187 | 1345.0 | 186.8 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| prince de Chimay | 1532 ± 187 | 1345.0 | 186.8 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| princesse de Parme | 1438 ± 96 | 1341.8 | 96.5 | 130 | 37-62-31 | 38 | 38 | -0.658 |
| le jeune marquis de Cambremer | 1536 ± 195 | 1341.2 | 194.7 | 12 | 2-1-9 | 1 | 1 | -1.2 |
| prince d’Agrigente | 1513 ± 183 | 1330.0 | 183.0 | 15 | 3-2-10 | 2 | 2 | -0.8 |
| le directeur | 1461 ± 136 | 1325.6 | 135.8 | 39 | 10-17-12 | 11 | 11 | -0.63 |
| comtesse Molé | 1461 ± 136 | 1324.7 | 136.2 | 34 | 5-9-20 | 6 | 6 | -1.142 |
| prince de Foix | 1516 ± 196 | 1319.5 | 196.2 | 14 | 4-4-6 | 3 | 3 | -0.893 |
| M. d'Argencourt | 1469 ± 158 | 1310.7 | 158.5 | 56 | 20-18-18 | 14 | 12 | -1.2 |
| Céline | 1497 ± 187 | 1310.2 | 186.6 | 16 | 3-6-7 | 2 | 2 | -1.14 |
| duc de Guermantes | 1396 ± 86 | 1309.3 | 86.4 | 401 | 123-172-106 | 110 | 107 | -0.985 |
| Rosemonde | 1475 ± 170 | 1305.0 | 170.3 | 20 | 5-7-8 | 4 | 4 | -0.7 |
| Goncourt | 1474 ± 171 | 1302.4 | 171.3 | 16 | 2-3-11 | 2 | 2 | -0.8 |
| M. de Vaugoubert | 1434 ± 137 | 1297.1 | 137.2 | 35 | 6-12-17 | 9 | 8 | -1.131 |
| général de Monserfeuil | 1454 ± 166 | 1288.1 | 166.3 | 18 | 5-8-5 | 4 | 4 | -1.289 |
| Mme de Cambremer | 1384 ± 102 | 1282.4 | 101.8 | 112 | 13-54-45 | 20 | 19 | -1.561 |
| le petit Cambremer | 1459 ± 187 | 1272.4 | 186.8 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| princesse de Silistrie | 1459 ± 187 | 1272.4 | 186.8 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| la Berma | 1400 ± 141 | 1259.4 | 141.1 | 62 | 19-24-19 | 19 | 16 | -0.451 |
| oncle Adolphe | 1455 ± 197 | 1258.0 | 197.1 | 20 | 4-11-5 | 6 | 5 | -1.52 |
| Balzac | 1422 ± 184 | 1237.6 | 184.1 | 18 | 2-4-12 | 2 | 2 | -0.8 |
| Mme d'Arpajon | 1343 ± 133 | 1209.7 | 133.2 | 37 | 6-22-9 | 8 | 8 | -1.72 |
| marquis de Cambremer | 1316 ± 120 | 1196.2 | 119.5 | 45 | 6-24-15 | 6 | 6 | -1.12 |
| princesse Sherbatoff | 1342 ± 173 | 1169.0 | 172.6 | 19 | 4-12-3 | 5 | 5 | -0.701 |
| Mme de Franquetot | 1303 ± 171 | 1132.3 | 170.7 | 23 | 4-13-6 | 3 | 3 | -0.837 |
| Mme d'Heudicourt | 1300 ± 182 | 1117.3 | 182.5 | 18 | 3-12-3 | 5 | 5 | -1.469 |
| marquise de Gallardon | 1312 ± 198 | 1114.7 | 197.5 | 19 | 1-12-6 | 7 | 7 | -2.104 |
| Saniette | 1216 ± 167 | 1048.5 | 167.3 | 35 | 2-25-8 | 9 | 8 | -2.784 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1999 ± 363 | 363.1 | 14 | 1 | 1 | 888 |
| marquis de Beausergent | 1966 ± 373 | 373.0 | 12 | 1 | 1 | 923 |
| Mme Elstir | 1937 ± 386 | 386.4 | 7 | 1 | 1 | 333 |
| Mlle de Saint-Loup | 1935 ± 388 | 387.7 | 7 | 2 | 2 | 940 |
| Céleste Albaret | 1931 ± 276 | 275.7 | 17 | 3 | 3 | 806 |
| la reine de Naples | 1899 ± 275 | 275.1 | 17 | 3 | 3 | 828 |
| prince de Saxe | 1859 ± 428 | 428.1 | 3 | 1 | 1 | 365 |
| Marie | 1834 ± 319 | 318.7 | 7 | 1 | 1 | 737 |
| colonel Picquart | 1832 ± 429 | 428.9 | 4 | 1 | 1 | 481 |
| Mme de Chaussepierre | 1825 ± 431 | 430.7 | 4 | 1 | 1 | 777 |
| Mme de Grouchy | 1818 ± 437 | 437.2 | 4 | 1 | 1 | 598 |
| duchesse de La Trémoïlle | 1809 ± 442 | 441.5 | 3 | 1 | 1 | 119 |
| Maeterlinck | 1790 ± 354 | 354.2 | 5 | 1 | 1 | 469 |
| marquis Maurice de Vaudémont | 1786 ± 464 | 463.8 | 2 | 1 | 1 | 353 |
| duc de Sidonia | 1772 ± 466 | 466.4 | 2 | 1 | 1 | 684 |
| Bibi | 1754 ± 477 | 477.1 | 2 | 1 | 1 | 579 |
| Mlle Bloch | 1748 ± 475 | 474.6 | 2 | 1 | 1 | 732 |
| Lady Israels | 1746 ± 476 | 475.5 | 2 | 1 | 1 | 232 |
| le commandant Duroc | 1742 ± 477 | 477.1 | 2 | 1 | 1 | 396 |
| monsieur Vallenères | 1742 ± 478 | 478.1 | 2 | 1 | 1 | 457 |
| Eulalie | 1741 ± 241 | 241.3 | 16 | 7 | 7 | 796 |
| Gribelin | 1723 ± 318 | 318.3 | 6 | 1 | 1 | 482 |
| Léa | 1714 ± 215 | 215.4 | 14 | 4 | 4 | 852 |
| Émilie Daltier | 1701 ± 388 | 388.1 | 3 | 1 | 1 | 839 |
| Victurnien | 1695 ± 256 | 256.1 | 8 | 2 | 2 | 704 |
| Bismarck | 1689 ± 332 | 332.1 | 4 | 1 | 1 | 210 |
| Herbinger | 1686 ± 385 | 385.4 | 3 | 1 | 1 | 108 |
| Létourville | 1680 ± 386 | 386.2 | 3 | 1 | 1 | 921 |
| Duroc | 1673 ± 520 | 520.0 | 2 | 1 | 1 | 395 |
| docteur Dieulafoy | 1668 ± 531 | 531.4 | 1 | 1 | 1 | 548 |
| le pianiste | 1665 ± 228 | 227.8 | 10 | 3 | 3 | 124 |
| Théodore | 1660 ± 417 | 417.4 | 2 | 1 | 1 | 59 |
| elle | 1656 ± 537 | 537.0 | 1 | 1 | 1 | 430 |
| marquis du Lau | 1655 ± 328 | 328.0 | 5 | 2 | 2 | 869 |
| Dechambre | 1654 ± 400 | 400.5 | 3 | 1 | 1 | 745 |
| grand-duc héritier de Luxembourg | 1646 ± 232 | 231.8 | 9 | 2 | 2 | 644 |
| Mlle de Stermaria | 1646 ± 224 | 223.6 | 10 | 5 | 5 | 577 |
| la duchesse d'Alençon | 1642 ± 295 | 295.3 | 6 | 1 | 1 | 628 |
| les La Trémoïlle | 1642 ± 257 | 256.9 | 7 | 1 | 1 | 118 |
| duc d'Aumale | 1638 ± 352 | 351.6 | 4 | 2 | 2 | 664 |
| Flora | 1631 ± 240 | 240.0 | 8 | 1 | 1 | 4 |
| M. de Courgivaux | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| Mme de Villebon | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| Arnulphe | 1629 ± 321 | 320.6 | 4 | 1 | 1 | 703 |
| M. d'Orsan | 1628 ± 206 | 206.1 | 11 | 1 | 1 | 177 |
| Marie-Aynard | 1624 ± 257 | 256.9 | 7 | 1 | 1 | 480 |
| Victurnienne | 1624 ± 257 | 256.9 | 7 | 1 | 1 | 480 |
| cousine Poictiers | 1617 ± 293 | 293.0 | 5 | 1 | 1 | 414 |
| duc de Poictiers | 1617 ± 293 | 293.0 | 5 | 1 | 1 | 414 |
| M. Vibert | 1606 ± 357 | 357.4 | 3 | 1 | 1 | 618 |
| Mme de Stermaria | 1602 ± 288 | 288.4 | 5 | 1 | 1 | 566 |
| Mme de Sagan | 1592 ± 359 | 359.1 | 3 | 1 | 1 | 485 |
| Mme Legrandin mère | 1591 ± 242 | 242.2 | 8 | 1 | 1 | 266 |
| Victoire | 1591 ± 242 | 242.2 | 8 | 1 | 1 | 266 |
| le baron Bréau-Chenut | 1587 ± 260 | 260.3 | 7 | 1 | 1 | 229 |
| le vieux père Chenut | 1587 ± 260 | 260.3 | 7 | 1 | 1 | 229 |
| Sarah Bernhardt | 1586 ± 260 | 260.0 | 7 | 1 | 1 | 908 |
| le jeune prince de Foix | 1586 ± 260 | 260.0 | 7 | 1 | 1 | 908 |
| vicomte de Courvoisier | 1586 ± 260 | 260.0 | 7 | 1 | 1 | 908 |
| Manet | 1582 ± 289 | 289.2 | 5 | 1 | 1 | 637 |
| d'Orléans | 1579 ± 289 | 289.1 | 5 | 1 | 1 | 325 |
| le grand-duc Wladimir | 1575 ± 366 | 366.0 | 3 | 1 | 1 | 689 |
| M. de Beauserfeuil | 1574 ± 250 | 250.3 | 7 | 1 | 1 | 644 |
| Lady Israël | 1573 ± 293 | 292.8 | 5 | 1 | 1 | 491 |
| Mlle de l’Orgeville | 1572 ± 358 | 358.1 | 3 | 1 | 1 | 892 |
| jeune blonde de Rivebelle | 1567 ± 267 | 267.2 | 6 | 2 | 2 | 326 |
| la Charité de Giotto | 1564 ± 497 | 497.3 | 1 | 1 | 1 | 49 |
| M. de Bornier | 1563 ± 310 | 310.4 | 5 | 1 | 1 | 609 |
| Élisabeth | 1561 ± 267 | 266.8 | 6 | 1 | 1 | 791 |
| duchesse de Létourville | 1560 ± 289 | 289.4 | 5 | 1 | 1 | 912 |
| baron de Guermantes | 1560 ± 611 | 610.6 | 1 | 1 | 1 | 452 |
| Sir Rufus Israël | 1553 ± 259 | 258.9 | 7 | 1 | 1 | 459 |
| M. de La Rochefoucauld | 1553 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| duchesse de La Rochefoucauld | 1553 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| duchesse de Praslin | 1553 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| le marquis de Ganançay | 1550 ± 285 | 285.4 | 6 | 1 | 1 | 367 |
| le marquis de Palancy | 1550 ± 285 | 285.4 | 6 | 1 | 1 | 367 |
| Octave | 1543 ± 332 | 332.4 | 4 | 2 | 2 | 875 |
| docteur Percepied | 1537 ± 313 | 312.6 | 4 | 1 | 1 | 58 |
| M. Barrère | 1533 ± 494 | 494.3 | 1 | 1 | 1 | 884 |
| Mlle d'Éporcheville | 1533 ± 212 | 212.2 | 10 | 2 | 2 | 865 |
| L’excellent écrivain G… | 1530 ± 317 | 317.1 | 4 | 1 | 1 | 448 |
| Lady Rufus Israël | 1527 ± 267 | 266.7 | 6 | 1 | 1 | 868 |
| comtesse de Monteriender | 1521 ± 313 | 313.0 | 4 | 1 | 1 | 176 |
| Coquelin | 1520 ± 288 | 288.0 | 5 | 1 | 1 | 198 |
| Mme Trombert | 1518 ± 314 | 313.5 | 4 | 1 | 1 | 231 |
| d’Orgeville | 1517 ± 247 | 246.6 | 7 | 1 | 1 | 701 |
| Napoléon III | 1516 ± 238 | 238.5 | 8 | 1 | 1 | 186 |
| Mme Putbus | 1515 ± 233 | 233.1 | 8 | 1 | 1 | 792 |
| la jeune ouvriere | 1512 ± 403 | 402.7 | 2 | 1 | 1 | 96 |
| M. Swann, le père | 1512 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| le comte de Paris | 1512 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| le prince de Galles | 1512 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| Mme de Montmorency | 1506 ± 203 | 203.4 | 11 | 1 | 1 | 718 |
| Mme de Rochechouart | 1506 ± 203 | 203.4 | 11 | 1 | 1 | 718 |
| M. Carnot | 1506 ± 221 | 220.7 | 9 | 1 | 1 | 663 |
| Mme Carnot | 1506 ± 221 | 220.7 | 9 | 1 | 1 | 663 |
| M. de Marsantes | 1502 ± 251 | 250.9 | 7 | 2 | 2 | 509 |
| Mme Timoléon d'Amoncourt | 1502 ± 225 | 225.0 | 9 | 1 | 1 | 694 |
| M. Arthur Meyer | 1501 ± 264 | 264.3 | 6 | 1 | 1 | 911 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| M. de Miribel | 1498 ± 314 | 313.5 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Henry | 1498 ± 314 | 313.5 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Picquart | 1498 ± 314 | 313.5 | 4 | 1 | 1 | 476 |
| comte de Paris | 1497 ± 214 | 213.6 | 10 | 3 | 3 | 219 |
| Thibaud | 1495 ± 232 | 232.2 | 8 | 1 | 1 | 780 |
| M. de Goncourt | 1493 ± 234 | 234.5 | 8 | 1 | 1 | 897 |
| prince de Sagan | 1492 ± 251 | 250.8 | 7 | 1 | 1 | 710 |
| Léonor de Cambremer | 1492 ± 202 | 201.9 | 12 | 1 | 1 | 923 |
| Liszt | 1490 ± 276 | 276.2 | 6 | 1 | 1 | 440 |
| Mme Ristori | 1490 ± 276 | 276.2 | 6 | 1 | 1 | 440 |
| M. Grevy | 1487 ± 348 | 348.3 | 3 | 1 | 1 | 94 |
| Poullein | 1482 ± 418 | 417.6 | 2 | 2 | 2 | 601 |
| princesse d'Épinay | 1481 ± 204 | 204.4 | 12 | 3 | 3 | 608 |
| Dostoïevski | 1481 ± 264 | 263.5 | 6 | 1 | 1 | 836 |
| Sainte-Beuve | 1481 ± 250 | 250.4 | 7 | 1 | 1 | 583 |
| le capitaine | 1479 ± 402 | 402.5 | 2 | 1 | 1 | 375 |
| le professeur E… | 1477 ± 444 | 444.1 | 2 | 1 | 1 | 684 |
| l'abbé Poiré | 1475 ± 212 | 211.5 | 10 | 1 | 1 | 708 |
| prince d'Agrigente | 1474 ± 414 | 414.0 | 2 | 2 | 2 | 922 |
| prince Von | 1472 ± 236 | 236.5 | 8 | 3 | 3 | 641 |
| Barrès | 1468 ± 225 | 224.6 | 9 | 1 | 1 | 661 |
| Clémenceau | 1468 ± 225 | 224.6 | 9 | 1 | 1 | 661 |
| comtesse douairière d'Argencourt | 1465 ± 212 | 212.0 | 10 | 1 | 1 | 590 |
| duchesse de Gallardon douairière | 1465 ± 212 | 212.0 | 10 | 1 | 1 | 590 |
| marquis de Fierbois | 1465 ± 212 | 212.0 | 10 | 1 | 1 | 590 |
| Vigny | 1463 ± 404 | 403.8 | 2 | 1 | 1 | 294 |
| Gisèle | 1460 ± 200 | 200.3 | 14 | 5 | 5 | 812 |
| Madame Elstir | 1455 ± 267 | 267.4 | 6 | 1 | 1 | 341 |
| les demoiselles d’Ambresac | 1455 ± 267 | 267.4 | 6 | 1 | 1 | 341 |
| M. de Chateaubriand | 1446 ± 209 | 209.3 | 11 | 2 | 2 | 870 |
| Mme de Vaugoubert | 1444 ± 232 | 232.3 | 9 | 2 | 2 | 822 |
| princesse Mathilde | 1443 ± 260 | 260.1 | 7 | 2 | 2 | 595 |
| le bâtonnier | 1440 ± 380 | 379.8 | 3 | 1 | 1 | 284 |
| D'Annunzio | 1437 ± 290 | 290.0 | 5 | 1 | 1 | 693 |
| le roi Théodose | 1432 ± 250 | 249.8 | 8 | 3 | 3 | 693 |
| M. d'Herweck | 1421 ± 288 | 288.5 | 5 | 2 | 2 | 699 |
| Beauserfeuil | 1401 ± 357 | 356.6 | 3 | 1 | 1 | 662 |
| Théodose Cadet | 1400 ± 356 | 356.5 | 3 | 1 | 1 | 665 |
| Cartier | 1389 ± 348 | 347.5 | 4 | 1 | 1 | 777 |
| Antoine | 1389 ± 400 | 400.2 | 3 | 1 | 1 | 358 |
| Prince Henri d'Orléans | 1378 ± 422 | 421.5 | 2 | 1 | 1 | 483 |
| duc de Châtellerault | 1378 ± 248 | 248.1 | 10 | 5 | 5 | 683 |
| professeur E… | 1373 ± 342 | 341.7 | 4 | 2 | 2 | 685 |
| comtesse G… | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| vicomtesse de Saint-Fiacre | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| marquise de Citri | 1367 ± 418 | 417.5 | 2 | 1 | 1 | 700 |
| M. de Stermaria | 1365 ± 225 | 225.4 | 10 | 4 | 4 | 280 |
| le prince Von | 1357 ± 228 | 228.2 | 10 | 2 | 2 | 640 |
| M. de Grouchy | 1354 ± 241 | 241.3 | 10 | 4 | 4 | 601 |
| M. Molé | 1350 ± 248 | 247.8 | 8 | 1 | 1 | 295 |
| M. de Bouillon | 1350 ± 248 | 247.8 | 8 | 1 | 1 | 295 |
| Musset | 1350 ± 248 | 247.8 | 8 | 1 | 1 | 295 |
| Victor Hugo | 1350 ± 248 | 247.8 | 8 | 1 | 1 | 295 |
| ma grand'tante | 1346 ± 538 | 538.1 | 1 | 1 | 1 | 1 |
| prince Foggi | 1345 ± 538 | 537.7 | 1 | 1 | 1 | 883 |
| la marquise | 1333 ± 532 | 531.6 | 1 | 1 | 1 | 528 |
| princesse de Nassau | 1329 ± 530 | 530.0 | 1 | 1 | 1 | 931 |
| les Courvoisier | 1306 ± 321 | 321.2 | 5 | 1 | 1 | 595 |
| Mme de Morienval | 1306 ± 294 | 293.9 | 6 | 1 | 1 | 367 |
| duchesse de Luxembourg | 1306 ± 294 | 293.9 | 6 | 1 | 1 | 367 |
| Marie Gineste | 1305 ± 512 | 512.5 | 2 | 1 | 1 | 736 |
| le grand-duc héritier de Luxembourg | 1303 ± 519 | 519.3 | 1 | 1 | 1 | 581 |
| Madame d'Ambresac | 1293 ± 493 | 493.1 | 2 | 1 | 1 | 366 |
| le curé | 1281 ± 492 | 492.4 | 2 | 1 | 1 | 42 |
| prince de Léon | 1279 ± 488 | 488.5 | 2 | 1 | 1 | 775 |
| capitaine de Borodino | 1278 ± 213 | 212.9 | 14 | 5 | 5 | 459 |
| Maurice | 1274 ± 305 | 304.9 | 7 | 1 | 1 | 908 |
| le prince von *** | 1271 ± 483 | 482.7 | 2 | 1 | 1 | 498 |
| Mme de Souvré | 1266 ± 246 | 245.8 | 11 | 2 | 2 | 687 |
| M. Bontemps | 1260 ± 329 | 329.4 | 9 | 2 | 2 | 899 |
| Dumont | 1255 ± 481 | 481.2 | 2 | 1 | 1 | 30 |
| le diplomate belge | 1253 ± 481 | 481.0 | 2 | 1 | 1 | 493 |
| Mme Blatin | 1248 ± 476 | 476.1 | 2 | 1 | 1 | 195 |
| M. de Luxembourg | 1243 ± 472 | 471.7 | 2 | 1 | 1 | 645 |
| l'historien de la Fronde | 1227 ± 457 | 456.8 | 3 | 1 | 1 | 453 |
| Mme de Simiane | 1221 ± 452 | 452.5 | 3 | 1 | 1 | 269 |
| prince de Faffenheim | 1217 ± 451 | 451.4 | 3 | 2 | 2 | 500 |
| la cousine d'Oriane | 1198 ± 444 | 444.3 | 3 | 1 | 1 | 606 |
| vicomtesse d'Égremont | 1196 ± 444 | 444.5 | 3 | 1 | 1 | 593 |
| Monsieur Vallenères | 1194 ± 443 | 442.6 | 3 | 1 | 1 | 472 |
| princesse d'Iéna | 1180 ± 443 | 442.9 | 3 | 1 | 1 | 166 |
| l'ambassadrice de Turquie | 1162 ± 425 | 425.4 | 4 | 1 | 1 | 690 |
| Mme Blandais | 1150 ± 422 | 422.3 | 4 | 2 | 2 | 288 |
| M. Pierre | 1146 ± 422 | 421.6 | 4 | 2 | 2 | 452 |
| Alix | 1136 ± 326 | 325.9 | 9 | 3 | 3 | 445 |
| Mme de Varambon | 1135 ± 419 | 419.2 | 4 | 2 | 2 | 648 |
| Mme Iéna | 1126 ± 410 | 410.2 | 5 | 1 | 1 | 635 |
| le prince de Faffenheim | 1126 ± 410 | 410.0 | 5 | 1 | 1 | 497 |
| ma grand’tante | 1118 ± 403 | 403.0 | 7 | 1 | 1 | 2 |
| l'empereur | 1118 ± 413 | 413.3 | 4 | 1 | 1 | 640 |
| Picquart | 1092 ± 398 | 397.8 | 8 | 2 | 2 | 482 |
| colonel de Froberville | 1072 ± 310 | 309.7 | 14 | 1 | 1 | 696 |
| M. de Vigny | 996 ± 368 | 368.4 | 8 | 1 | 1 | 295 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Céleste Albaret | 3 | t736: 1933 ± 274 | t806: 1931 ± 276 | t806: 1931 ± 276 | t736: 1933 ± 274 |
| Mlle d'Oloron | 1 | t888: 1999 ± 363 | t888: 1999 ± 363 | t888: 1999 ± 363 | t888: 1999 ± 363 |
| la reine de Naples | 3 | t628: 1900 ± 276 | t828: 1899 ± 275 | t828: 1899 ± 275 | t628: 1900 ± 276 |
| marquis de Beausergent | 1 | t923: 1966 ± 373 | t923: 1966 ± 373 | t923: 1966 ± 373 | t923: 1966 ± 373 |
| docteur du Boulbon | 6 | t248: 1748 ± 178 | t725: 1770 ± 188 | t248: 1748 ± 178 | t523: 1774 ± 165 |
| Mme Elstir | 1 | t333: 1937 ± 386 | t333: 1937 ± 386 | t333: 1937 ± 386 | t333: 1937 ± 386 |
| Mlle de Saint-Loup | 2 | t939: 1935 ± 388 | t940: 1935 ± 388 | t939: 1935 ± 388 | t939: 1935 ± 388 |
| le peintre | 8 | t89: 1650 ± 121 | t186: 1641 ± 118 | t186: 1641 ± 118 | t114: 1651 ± 117 |
| Bergotte | 32 | t28: 1531 ± 116 | t941: 1645 ± 127 | t28: 1531 ± 116 | t941: 1645 ± 127 |
| Marie | 1 | t737: 1834 ± 319 | t737: 1834 ± 319 | t737: 1834 ± 319 | t737: 1834 ± 319 |
| Françoise | 76 | t2: 1623 ± 95 | t940: 1615 ± 108 | t536: 1589 ± 84 | t59: 1628 ± 91 |
| Rachel | 43 | t251: 1460 ± 125 | t939: 1596 ± 93 | t469: 1438 ± 80 | t939: 1596 ± 93 |
| Eulalie | 7 | t19: 1699 ± 202 | t796: 1741 ± 241 | t19: 1699 ± 202 | t796: 1741 ± 241 |
| Léa | 4 | t807: 1712 ± 211 | t852: 1714 ± 215 | t807: 1712 ± 211 | t852: 1714 ± 215 |
| Aimé | 18 | t279: 1518 ± 131 | t890: 1580 ± 99 | t282: 1518 ± 130 | t791: 1585 ± 92 |
| l'amie de Mlle Vinteuil | 12 | t58: 1602 ± 145 | t855: 1606 ± 128 | t58: 1602 ± 145 | t762: 1609 ± 126 |
| Rémi | 3 | t101: 1651 ± 177 | t177: 1650 ± 174 | t177: 1650 ± 174 | t101: 1651 ± 177 |
| Mme Verdurin | 82 | t70: 1486 ± 74 | t934: 1555 ± 82 | t124: 1485 ± 63 | t927: 1555 ± 80 |
| Elstir | 24 | t269: 1528 ± 112 | t904: 1569 ± 100 | t617: 1523 ± 98 | t898: 1569 ± 98 |
| Morel | 31 | t501: 1428 ± 127 | t928: 1546 ± 84 | t501: 1428 ± 127 | t928: 1546 ± 84 |
| Jupien | 18 | t356: 1610 ± 145 | t913: 1556 ± 96 | t888: 1555 ± 93 | t356: 1610 ± 145 |
| Mlle Vinteuil | 15 | t45: 1555 ± 135 | t855: 1559 ± 100 | t61: 1553 ± 133 | t762: 1563 ± 101 |
| le père du narrateur | 24 | t4: 1584 ± 98 | t550: 1593 ± 136 | t197: 1580 ± 88 | t547: 1593 ± 136 |
| la grand-mère | 74 | t1: 1568 ± 96 | t917: 1565 ± 109 | t412: 1545 ± 78 | t731: 1576 ± 95 |
| M. Verdurin | 27 | t70: 1530 ± 98 | t904: 1558 ± 104 | t745: 1529 ± 93 | t904: 1558 ± 104 |
| Bloch | 70 | t29: 1365 ± 126 | t940: 1528 ± 81 | t29: 1365 ± 126 | t930: 1528 ± 78 |
| Victurnien | 2 | t703: 1695 ± 256 | t704: 1695 ± 256 | t703: 1695 ± 256 | t703: 1695 ± 256 |
| le pianiste | 3 | t85: 1667 ± 228 | t124: 1665 ± 228 | t124: 1665 ± 228 | t85: 1667 ± 228 |
| Maeterlinck | 1 | t469: 1790 ± 354 | t469: 1790 ± 354 | t469: 1790 ± 354 | t469: 1790 ± 354 |
| la mère du narrateur | 40 | t4: 1609 ± 108 | t888: 1532 ± 97 | t888: 1532 ± 97 | t4: 1609 ± 108 |
| prince de Guermantes | 22 | t477: 1511 ± 104 | t927: 1552 ± 116 | t477: 1511 ± 104 | t827: 1552 ± 97 |
| Mme Cottard | 11 | t87: 1636 ± 141 | t756: 1625 ± 190 | t756: 1625 ± 190 | t186: 1641 ± 133 |
| Norpois | 62 | t201: 1577 ± 76 | t915: 1568 ± 134 | t350: 1554 ± 80 | t201: 1577 ± 76 |
| Robert de Saint-Loup | 154 | t298: 1492 ± 71 | t939: 1505 ± 73 | t477: 1430 ± 54 | t911: 1507 ± 64 |
| prince de Saxe | 1 | t365: 1859 ± 428 | t365: 1859 ± 428 | t365: 1859 ± 428 | t365: 1859 ± 428 |
| Odette | 138 | t21: 1562 ± 80 | t938: 1509 ± 80 | t490: 1460 ± 80 | t21: 1562 ± 80 |
| M. d'Orsan | 1 | t177: 1628 ± 206 | t177: 1628 ± 206 | t177: 1628 ± 206 | t177: 1628 ± 206 |
| Mlle de Stermaria | 5 | t280: 1591 ± 234 | t577: 1646 ± 224 | t280: 1591 ± 234 | t576: 1646 ± 223 |
| Mme Sazerat | 6 | t416: 1612 ± 204 | t882: 1582 ± 160 | t881: 1582 ± 160 | t416: 1612 ± 204 |
| Mme de Charlus | 2 | t621: 1609 ± 190 | t855: 1611 ± 189 | t621: 1609 ± 190 | t855: 1611 ± 189 |
| marquis de Bréauté | 19 | t157: 1519 ± 130 | t938: 1529 ± 110 | t450: 1510 ± 111 | t938: 1529 ± 110 |
| le grand-père du narrateur | 16 | t2: 1642 ± 102 | t549: 1583 ± 167 | t549: 1583 ± 167 | t31: 1646 ± 99 |
| grand-duc héritier de Luxembourg | 2 | t540: 1652 ± 236 | t644: 1646 ± 232 | t644: 1646 ± 232 | t540: 1652 ± 236 |
| Andrée | 31 | t341: 1512 ± 105 | t875: 1503 ± 90 | t782: 1490 ± 76 | t345: 1512 ± 104 |
| Mme de Surgis | 9 | t687: 1552 ± 112 | t817: 1545 ± 132 | t817: 1545 ± 132 | t687: 1552 ± 112 |
| Dreyfus | 7 | t324: 1533 ± 130 | t708: 1523 ± 111 | t708: 1523 ± 111 | t421: 1534 ± 114 |
| duchesse de Guermantes | 194 | t67: 1599 ± 125 | t939: 1479 ± 68 | t939: 1479 ± 68 | t412: 1676 ± 67 |
| comte de Forcheville | 25 | t110: 1693 ± 89 | t938: 1517 ± 112 | t865: 1515 ± 110 | t124: 1694 ± 86 |
| Gribelin | 1 | t482: 1723 ± 318 | t482: 1723 ± 318 | t482: 1723 ± 318 | t482: 1723 ± 318 |
| Mme Goupil | 2 | t870: 1573 ± 170 | t871: 1573 ± 170 | t870: 1573 ± 170 | t871: 1573 ± 170 |
| la marquise douairière de Cambremer | 6 | t158: 1510 ± 202 | t761: 1532 ± 129 | t158: 1510 ± 202 | t761: 1532 ± 129 |
| colonel Picquart | 1 | t481: 1832 ± 429 | t481: 1832 ± 429 | t481: 1832 ± 429 | t481: 1832 ± 429 |
| le narrateur | 315 | t4: 1441 ± 85 | t941: 1467 ± 65 | t56: 1440 ± 77 | t623: 1574 ± 51 |
| Mme de Chaussepierre | 1 | t777: 1825 ± 431 | t777: 1825 ± 431 | t777: 1825 ± 431 | t777: 1825 ± 431 |
| Flora | 1 | t4: 1631 ± 240 | t4: 1631 ± 240 | t4: 1631 ± 240 | t4: 1631 ± 240 |
| docteur Cottard | 43 | t71: 1461 ± 86 | t923: 1493 ± 102 | t523: 1453 ± 91 | t761: 1494 ± 75 |
| Charcot | 1 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 |
| M. Reinach | 1 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 |
| Mme de Villeparisis | 78 | t3: 1466 ± 141 | t882: 1499 ± 111 | t590: 1465 ± 72 | t477: 1509 ± 60 |
| Mme Leroi | 5 | t436: 1582 ± 195 | t506: 1581 ± 195 | t505: 1581 ± 195 | t436: 1582 ± 195 |
| Gilberte | 74 | t37: 1603 ± 108 | t939: 1456 ± 70 | t939: 1456 ± 70 | t37: 1603 ± 108 |
| les La Trémoïlle | 1 | t118: 1642 ± 257 | t118: 1642 ± 257 | t118: 1642 ± 257 | t118: 1642 ± 257 |
| M. Vinteuil | 15 | t45: 1517 ± 123 | t898: 1505 ± 122 | t898: 1505 ± 122 | t176: 1532 ± 124 |
| Albertine | 126 | t229: 1564 ± 100 | t918: 1461 ± 78 | t873: 1458 ± 61 | t345: 1599 ± 76 |
| marquise de Saint-Euverte | 13 | t163: 1338 ± 168 | t938: 1501 ± 118 | t163: 1338 ± 168 | t938: 1501 ± 118 |
| Mme de Grouchy | 1 | t598: 1818 ± 437 | t598: 1818 ± 437 | t598: 1818 ± 437 | t598: 1818 ± 437 |
| Mme de Sévigné | 4 | t269: 1563 ± 171 | t729: 1544 ± 162 | t729: 1544 ± 162 | t269: 1563 ± 171 |
| Mme de Marsantes | 20 | t232: 1455 ± 140 | t890: 1482 ± 102 | t421: 1447 ± 101 | t890: 1482 ± 102 |
| Swann | 198 | t2: 1517 ± 77 | t938: 1451 ± 72 | t708: 1438 ± 55 | t2: 1517 ± 77 |
| baron de Charlus | 118 | t56: 1590 ± 110 | t938: 1449 ± 71 | t912: 1443 ± 63 | t521: 1591 ± 69 |
| Legrandin | 20 | t17: 1397 ± 156 | t930: 1481 ± 104 | t266: 1396 ± 129 | t888: 1483 ± 98 |
| M. Ski | 2 | t748: 1525 ± 157 | t825: 1532 ± 156 | t748: 1525 ± 157 | t825: 1532 ± 156 |
| Brichot | 21 | t111: 1525 ± 126 | t923: 1464 ± 88 | t905: 1464 ± 84 | t118: 1525 ± 125 |
| M. de Chevregny | 1 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 |
| M. de Crécy | 1 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 |
| Mme Féré | 1 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 | t761: 1549 ± 173 |
| général de Froberville | 7 | t157: 1518 ± 158 | t696: 1526 ± 156 | t157: 1518 ± 158 | t696: 1526 ± 156 |
| duchesse de La Trémoïlle | 1 | t119: 1809 ± 442 | t119: 1809 ± 442 | t119: 1809 ± 442 | t119: 1809 ± 442 |
| Marie-Aynard | 1 | t480: 1624 ± 257 | t480: 1624 ± 257 | t480: 1624 ± 257 | t480: 1624 ± 257 |
| Victurnienne | 1 | t480: 1624 ± 257 | t480: 1624 ± 257 | t480: 1624 ± 257 | t480: 1624 ± 257 |
| prince des Laumes | 3 | t177: 1559 ± 155 | t596: 1514 ± 147 | t596: 1514 ± 147 | t177: 1559 ± 155 |
| M. Nissim Bernard | 7 | t315: 1498 ± 171 | t923: 1495 ± 134 | t923: 1495 ± 134 | t509: 1502 ± 145 |
| tante Léonie | 20 | t8: 1495 ± 126 | t361: 1522 ± 161 | t56: 1490 ± 123 | t361: 1522 ± 161 |
| Bloch père | 8 | t313: 1431 ± 137 | t923: 1489 ± 131 | t314: 1430 ± 137 | t923: 1489 ± 131 |
| Bismarck | 1 | t210: 1689 ± 332 | t210: 1689 ± 332 | t210: 1689 ± 332 | t210: 1689 ± 332 |
| Esther | 2 | t791: 1537 ± 181 | t792: 1537 ± 181 | t791: 1537 ± 181 | t791: 1537 ± 181 |
| princesse de Luxembourg | 6 | t283: 1501 ± 163 | t730: 1508 ± 152 | t283: 1501 ± 163 | t644: 1512 ± 148 |
| Mme Bontemps | 13 | t229: 1515 ± 124 | t899: 1480 ± 125 | t899: 1480 ± 125 | t229: 1515 ± 124 |
| princesse de Guermantes | 25 | t363: 1576 ± 114 | t932: 1466 ± 113 | t932: 1466 ± 113 | t367: 1577 ± 113 |
| Mme Legrandin mère | 1 | t266: 1591 ± 242 | t266: 1591 ± 242 | t266: 1591 ± 242 | t266: 1591 ± 242 |
| Victoire | 1 | t266: 1591 ± 242 | t266: 1591 ± 242 | t266: 1591 ± 242 | t266: 1591 ± 242 |
| la duchesse d'Alençon | 1 | t628: 1642 ± 295 | t628: 1642 ± 295 | t628: 1642 ± 295 | t628: 1642 ± 295 |
| duc de Chartres | 1 | t696: 1532 ± 187 | t696: 1532 ± 187 | t696: 1532 ± 187 | t696: 1532 ± 187 |
| prince de Chimay | 1 | t696: 1532 ± 187 | t696: 1532 ± 187 | t696: 1532 ± 187 | t696: 1532 ± 187 |
| princesse de Parme | 38 | t363: 1411 ± 127 | t724: 1438 ± 96 | t570: 1411 ± 78 | t724: 1438 ± 96 |
| le jeune marquis de Cambremer | 1 | t890: 1536 ± 195 | t890: 1536 ± 195 | t890: 1536 ± 195 | t890: 1536 ± 195 |
| prince d’Agrigente | 2 | t630: 1518 ± 187 | t870: 1513 ± 183 | t870: 1513 ± 183 | t630: 1518 ± 187 |
| marquis du Lau | 2 | t775: 1648 ± 328 | t869: 1655 ± 328 | t775: 1648 ± 328 | t869: 1655 ± 328 |
| Sarah Bernhardt | 1 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 |
| le jeune prince de Foix | 1 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 |
| vicomte de Courvoisier | 1 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 | t908: 1586 ± 260 |
| le baron Bréau-Chenut | 1 | t229: 1587 ± 260 | t229: 1587 ± 260 | t229: 1587 ± 260 | t229: 1587 ± 260 |
| le vieux père Chenut | 1 | t229: 1587 ± 260 | t229: 1587 ± 260 | t229: 1587 ± 260 | t229: 1587 ± 260 |
| le directeur | 11 | t270: 1474 ± 135 | t737: 1461 ± 136 | t737: 1461 ± 136 | t270: 1474 ± 135 |
| comtesse Molé | 6 | t668: 1445 ± 129 | t870: 1461 ± 136 | t668: 1445 ± 129 | t870: 1461 ± 136 |
| cousine Poictiers | 1 | t414: 1617 ± 293 | t414: 1617 ± 293 | t414: 1617 ± 293 | t414: 1617 ± 293 |
| duc de Poictiers | 1 | t414: 1617 ± 293 | t414: 1617 ± 293 | t414: 1617 ± 293 | t414: 1617 ± 293 |
| M. de Beauserfeuil | 1 | t644: 1574 ± 250 | t644: 1574 ± 250 | t644: 1574 ± 250 | t644: 1574 ± 250 |
| marquis Maurice de Vaudémont | 1 | t353: 1786 ± 464 | t353: 1786 ± 464 | t353: 1786 ± 464 | t353: 1786 ± 464 |
| Mlle d'Éporcheville | 2 | t863: 1533 ± 212 | t865: 1533 ± 212 | t865: 1533 ± 212 | t863: 1533 ± 212 |
| prince de Foix | 3 | t580: 1491 ± 196 | t908: 1516 ± 196 | t580: 1491 ± 196 | t908: 1516 ± 196 |
| Mme de Stermaria | 1 | t566: 1602 ± 288 | t566: 1602 ± 288 | t566: 1602 ± 288 | t566: 1602 ± 288 |
| Émilie Daltier | 1 | t839: 1701 ± 388 | t839: 1701 ± 388 | t839: 1701 ± 388 | t839: 1701 ± 388 |
| M. d'Argencourt | 12 | t453: 1534 ± 102 | t911: 1469 ± 158 | t911: 1469 ± 158 | t464: 1535 ± 99 |
| Céline | 2 | t4: 1468 ± 185 | t266: 1497 ± 187 | t4: 1468 ± 185 | t266: 1497 ± 187 |
| duc de Guermantes | 107 | t362: 1467 ± 99 | t938: 1396 ± 86 | t938: 1396 ± 86 | t464: 1474 ± 71 |
| Arnulphe | 1 | t703: 1629 ± 321 | t703: 1629 ± 321 | t703: 1629 ± 321 | t703: 1629 ± 321 |
| duc de Sidonia | 1 | t684: 1772 ± 466 | t684: 1772 ± 466 | t684: 1772 ± 466 | t684: 1772 ± 466 |
| Rosemonde | 4 | t345: 1457 ± 168 | t729: 1475 ± 170 | t345: 1457 ± 168 | t727: 1475 ± 170 |
| Mme de Montmorency | 1 | t718: 1506 ± 203 | t718: 1506 ± 203 | t718: 1506 ± 203 | t718: 1506 ± 203 |
| Mme de Rochechouart | 1 | t718: 1506 ± 203 | t718: 1506 ± 203 | t718: 1506 ± 203 | t718: 1506 ± 203 |
| Goncourt | 2 | t896: 1474 ± 171 | t898: 1474 ± 171 | t896: 1474 ± 171 | t896: 1474 ± 171 |
| Herbinger | 1 | t108: 1686 ± 385 | t108: 1686 ± 385 | t108: 1686 ± 385 | t108: 1686 ± 385 |
| jeune blonde de Rivebelle | 2 | t325: 1567 ± 267 | t326: 1567 ± 267 | t325: 1567 ± 267 | t325: 1567 ± 267 |
| M. de Vaugoubert | 8 | t209: 1480 ± 189 | t822: 1434 ± 137 | t778: 1432 ± 132 | t209: 1480 ± 189 |
| Sir Rufus Israël | 1 | t459: 1553 ± 259 | t459: 1553 ± 259 | t459: 1553 ± 259 | t459: 1553 ± 259 |
| Létourville | 1 | t921: 1680 ± 386 | t921: 1680 ± 386 | t921: 1680 ± 386 | t921: 1680 ± 386 |
| Élisabeth | 1 | t791: 1561 ± 267 | t791: 1561 ± 267 | t791: 1561 ± 267 | t791: 1561 ± 267 |
| Manet | 1 | t637: 1582 ± 289 | t637: 1582 ± 289 | t637: 1582 ± 289 | t637: 1582 ± 289 |
| d'Orléans | 1 | t325: 1579 ± 289 | t325: 1579 ± 289 | t325: 1579 ± 289 | t325: 1579 ± 289 |
| Léonor de Cambremer | 1 | t923: 1492 ± 202 | t923: 1492 ± 202 | t923: 1492 ± 202 | t923: 1492 ± 202 |
| général de Monserfeuil | 4 | t628: 1455 ± 166 | t631: 1454 ± 166 | t631: 1454 ± 166 | t628: 1455 ± 166 |
| duc d'Aumale | 2 | t366: 1622 ± 347 | t664: 1638 ± 352 | t366: 1622 ± 347 | t664: 1638 ± 352 |
| M. Carnot | 1 | t663: 1506 ± 221 | t663: 1506 ± 221 | t663: 1506 ± 221 | t663: 1506 ± 221 |
| Mme Carnot | 1 | t663: 1506 ± 221 | t663: 1506 ± 221 | t663: 1506 ± 221 | t663: 1506 ± 221 |
| M. de La Rochefoucauld | 1 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 |
| duchesse de La Rochefoucauld | 1 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 |
| duchesse de Praslin | 1 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 | t297: 1553 ± 268 |
| comte de Paris | 3 | t192: 1495 ± 214 | t219: 1497 ± 214 | t192: 1495 ± 214 | t219: 1497 ± 214 |
| Mme de Cambremer | 19 | t165: 1370 ± 143 | t923: 1384 ± 102 | t694: 1349 ± 87 | t923: 1384 ± 102 |
| Mme Putbus | 1 | t792: 1515 ± 233 | t792: 1515 ± 233 | t792: 1515 ± 233 | t792: 1515 ± 233 |
| Lady Israël | 1 | t491: 1573 ± 293 | t491: 1573 ± 293 | t491: 1573 ± 293 | t491: 1573 ± 293 |
| Napoléon III | 1 | t186: 1516 ± 238 | t186: 1516 ± 238 | t186: 1516 ± 238 | t186: 1516 ± 238 |
| Bibi | 1 | t579: 1754 ± 477 | t579: 1754 ± 477 | t579: 1754 ± 477 | t579: 1754 ± 477 |
| princesse d'Épinay | 3 | t593: 1481 ± 204 | t608: 1481 ± 204 | t608: 1481 ± 204 | t593: 1481 ± 204 |
| Mme Timoléon d'Amoncourt | 1 | t694: 1502 ± 225 | t694: 1502 ± 225 | t694: 1502 ± 225 | t694: 1502 ± 225 |
| Mlle Bloch | 1 | t732: 1748 ± 475 | t732: 1748 ± 475 | t732: 1748 ± 475 | t732: 1748 ± 475 |
| le petit Cambremer | 1 | t888: 1459 ± 187 | t888: 1459 ± 187 | t888: 1459 ± 187 | t888: 1459 ± 187 |
| princesse de Silistrie | 1 | t888: 1459 ± 187 | t888: 1459 ± 187 | t888: 1459 ± 187 | t888: 1459 ± 187 |
| duchesse de Létourville | 1 | t912: 1560 ± 289 | t912: 1560 ± 289 | t912: 1560 ± 289 | t912: 1560 ± 289 |
| Lady Israels | 1 | t232: 1746 ± 476 | t232: 1746 ± 476 | t232: 1746 ± 476 | t232: 1746 ± 476 |
| d’Orgeville | 1 | t701: 1517 ± 247 | t701: 1517 ± 247 | t701: 1517 ± 247 | t701: 1517 ± 247 |
| le commandant Duroc | 1 | t396: 1742 ± 477 | t396: 1742 ± 477 | t396: 1742 ± 477 | t396: 1742 ± 477 |
| le marquis de Ganançay | 1 | t367: 1550 ± 285 | t367: 1550 ± 285 | t367: 1550 ± 285 | t367: 1550 ± 285 |
| le marquis de Palancy | 1 | t367: 1550 ± 285 | t367: 1550 ± 285 | t367: 1550 ± 285 | t367: 1550 ± 285 |
| l'abbé Poiré | 1 | t708: 1475 ± 212 | t708: 1475 ± 212 | t708: 1475 ± 212 | t708: 1475 ± 212 |
| monsieur Vallenères | 1 | t457: 1742 ± 478 | t457: 1742 ± 478 | t457: 1742 ± 478 | t457: 1742 ± 478 |
| Thibaud | 1 | t780: 1495 ± 232 | t780: 1495 ± 232 | t780: 1495 ± 232 | t780: 1495 ± 232 |
| Lady Rufus Israël | 1 | t868: 1527 ± 267 | t868: 1527 ± 267 | t868: 1527 ± 267 | t868: 1527 ± 267 |
| Gisèle | 5 | t342: 1417 ± 212 | t812: 1460 ± 200 | t342: 1417 ± 212 | t812: 1460 ± 200 |
| la Berma | 16 | t21: 1570 ± 132 | t936: 1400 ± 141 | t934: 1400 ± 141 | t21: 1570 ± 132 |
| M. de Goncourt | 1 | t897: 1493 ± 234 | t897: 1493 ± 234 | t897: 1493 ± 234 | t897: 1493 ± 234 |
| oncle Adolphe | 5 | t21: 1385 ± 172 | t501: 1455 ± 197 | t21: 1385 ± 172 | t501: 1455 ± 197 |
| Dechambre | 1 | t745: 1654 ± 400 | t745: 1654 ± 400 | t745: 1654 ± 400 | t745: 1654 ± 400 |
| comtesse douairière d'Argencourt | 1 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 |
| duchesse de Gallardon douairière | 1 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 |
| marquis de Fierbois | 1 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 | t590: 1465 ± 212 |
| M. de Bornier | 1 | t609: 1563 ± 310 | t609: 1563 ± 310 | t609: 1563 ± 310 | t609: 1563 ± 310 |
| M. de Marsantes | 2 | t299: 1493 ± 262 | t509: 1502 ± 251 | t299: 1493 ± 262 | t509: 1502 ± 251 |
| M. Swann, le père | 1 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 |
| le comte de Paris | 1 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 |
| le prince de Galles | 1 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 | t2: 1512 ± 261 |
| M. Vibert | 1 | t618: 1606 ± 357 | t618: 1606 ± 357 | t618: 1606 ± 357 | t618: 1606 ± 357 |
| Barrès | 1 | t661: 1468 ± 225 | t661: 1468 ± 225 | t661: 1468 ± 225 | t661: 1468 ± 225 |
| Clémenceau | 1 | t661: 1468 ± 225 | t661: 1468 ± 225 | t661: 1468 ± 225 | t661: 1468 ± 225 |
| Théodore | 1 | t59: 1660 ± 417 | t59: 1660 ± 417 | t59: 1660 ± 417 | t59: 1660 ± 417 |
| prince de Sagan | 1 | t710: 1492 ± 251 | t710: 1492 ± 251 | t710: 1492 ± 251 | t710: 1492 ± 251 |
| Balzac | 2 | t295: 1393 ± 190 | t898: 1422 ± 184 | t295: 1393 ± 190 | t898: 1422 ± 184 |
| M. Arthur Meyer | 1 | t911: 1501 ± 264 | t911: 1501 ± 264 | t911: 1501 ± 264 | t911: 1501 ± 264 |
| M. de Chateaubriand | 2 | t294: 1411 ± 242 | t870: 1446 ± 209 | t294: 1411 ± 242 | t870: 1446 ± 209 |
| prince Von | 3 | t623: 1473 ± 235 | t641: 1472 ± 236 | t641: 1472 ± 236 | t623: 1473 ± 235 |
| Mme de Sagan | 1 | t485: 1592 ± 359 | t485: 1592 ± 359 | t485: 1592 ± 359 | t485: 1592 ± 359 |
| Coquelin | 1 | t198: 1520 ± 288 | t198: 1520 ± 288 | t198: 1520 ± 288 | t198: 1520 ± 288 |
| Sainte-Beuve | 1 | t583: 1481 ± 250 | t583: 1481 ± 250 | t583: 1481 ± 250 | t583: 1481 ± 250 |
| docteur Percepied | 1 | t58: 1537 ± 313 | t58: 1537 ± 313 | t58: 1537 ± 313 | t58: 1537 ± 313 |
| Dostoïevski | 1 | t836: 1481 ± 264 | t836: 1481 ± 264 | t836: 1481 ± 264 | t836: 1481 ± 264 |
| Liszt | 1 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 |
| Mme Ristori | 1 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 |
| Mlle de l’Orgeville | 1 | t892: 1572 ± 358 | t892: 1572 ± 358 | t892: 1572 ± 358 | t892: 1572 ± 358 |
| L’excellent écrivain G… | 1 | t448: 1530 ± 317 | t448: 1530 ± 317 | t448: 1530 ± 317 | t448: 1530 ± 317 |
| Mme de Vaugoubert | 2 | t686: 1439 ± 242 | t822: 1444 ± 232 | t686: 1439 ± 242 | t822: 1444 ± 232 |
| Octave | 2 | t340: 1498 ± 324 | t875: 1543 ± 332 | t340: 1498 ± 324 | t875: 1543 ± 332 |
| Mme d'Arpajon | 8 | t597: 1337 ± 135 | t718: 1343 ± 133 | t612: 1335 ± 134 | t718: 1343 ± 133 |
| le grand-duc Wladimir | 1 | t689: 1575 ± 366 | t689: 1575 ± 366 | t689: 1575 ± 366 | t689: 1575 ± 366 |
| comtesse de Monteriender | 1 | t176: 1521 ± 313 | t176: 1521 ± 313 | t176: 1521 ± 313 | t176: 1521 ± 313 |
| Mme Trombert | 1 | t231: 1518 ± 314 | t231: 1518 ± 314 | t231: 1518 ± 314 | t231: 1518 ± 314 |
| marquis de Cambremer | 6 | t277: 1415 ± 166 | t761: 1316 ± 120 | t761: 1316 ± 120 | t277: 1415 ± 166 |
| Madame Elstir | 1 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 |
| les demoiselles d’Ambresac | 1 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 |
| M. de Miribel | 1 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 |
| le lieutenant-colonel Henry | 1 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 |
| le lieutenant-colonel Picquart | 1 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 | t476: 1498 ± 314 |
| princesse Mathilde | 2 | t238: 1421 ± 268 | t595: 1443 ± 260 | t238: 1421 ± 268 | t595: 1443 ± 260 |
| le roi Théodose | 3 | t208: 1434 ± 256 | t693: 1432 ± 250 | t693: 1432 ± 250 | t208: 1434 ± 256 |
| princesse Sherbatoff | 5 | t742: 1344 ± 173 | t757: 1342 ± 173 | t757: 1342 ± 173 | t742: 1344 ± 173 |
| Duroc | 1 | t395: 1673 ± 520 | t395: 1673 ± 520 | t395: 1673 ± 520 | t395: 1673 ± 520 |
| D'Annunzio | 1 | t693: 1437 ± 290 | t693: 1437 ± 290 | t693: 1437 ± 290 | t693: 1437 ± 290 |
| M. de Stermaria | 4 | t275: 1365 ± 225 | t280: 1365 ± 225 | t279: 1365 ± 225 | t275: 1365 ± 225 |
| M. Grevy | 1 | t94: 1487 ± 348 | t94: 1487 ± 348 | t94: 1487 ± 348 | t94: 1487 ± 348 |
| docteur Dieulafoy | 1 | t548: 1668 ± 531 | t548: 1668 ± 531 | t548: 1668 ± 531 | t548: 1668 ± 531 |
| M. d'Herweck | 2 | t698: 1421 ± 288 | t699: 1421 ± 288 | t698: 1421 ± 288 | t698: 1421 ± 288 |
| Mme de Franquetot | 3 | t158: 1412 ± 217 | t923: 1303 ± 171 | t923: 1303 ± 171 | t158: 1412 ± 217 |
| duc de Châtellerault | 5 | t488: 1387 ± 242 | t683: 1378 ± 248 | t682: 1378 ± 248 | t488: 1387 ± 242 |
| le prince Von | 2 | t625: 1358 ± 228 | t640: 1357 ± 228 | t640: 1357 ± 228 | t625: 1358 ± 228 |
| elle | 1 | t430: 1656 ± 537 | t430: 1656 ± 537 | t430: 1656 ± 537 | t430: 1656 ± 537 |
| Mme d'Heudicourt | 5 | t602: 1300 ± 183 | t609: 1300 ± 182 | t603: 1300 ± 182 | t608: 1300 ± 182 |
| marquise de Gallardon | 7 | t158: 1278 ± 209 | t711: 1312 ± 198 | t158: 1278 ± 209 | t710: 1312 ± 197 |
| M. de Grouchy | 4 | t587: 1352 ± 241 | t601: 1354 ± 241 | t587: 1352 ± 241 | t601: 1354 ± 241 |
| la jeune ouvriere | 1 | t96: 1512 ± 403 | t96: 1512 ± 403 | t96: 1512 ± 403 | t96: 1512 ± 403 |
| M. Molé | 1 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 |
| M. de Bouillon | 1 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 |
| Musset | 1 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 |
| Victor Hugo | 1 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 | t295: 1350 ± 248 |
| M. de Courgivaux | 1 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 |
| Mme de Villebon | 1 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 |
| le capitaine | 1 | t375: 1479 ± 402 | t375: 1479 ± 402 | t375: 1479 ± 402 | t375: 1479 ± 402 |
| la Charité de Giotto | 1 | t49: 1564 ± 497 | t49: 1564 ± 497 | t49: 1564 ± 497 | t49: 1564 ± 497 |
| capitaine de Borodino | 5 | t379: 1280 ± 214 | t459: 1278 ± 213 | t459: 1278 ± 213 | t402: 1280 ± 212 |
| Poullein | 2 | t600: 1482 ± 418 | t601: 1482 ± 418 | t601: 1482 ± 418 | t600: 1482 ± 418 |
| prince d'Agrigente | 2 | t586: 1459 ± 406 | t922: 1474 ± 414 | t586: 1459 ± 406 | t922: 1474 ± 414 |
| le bâtonnier | 1 | t284: 1440 ± 380 | t284: 1440 ± 380 | t284: 1440 ± 380 | t284: 1440 ± 380 |
| Vigny | 1 | t294: 1463 ± 404 | t294: 1463 ± 404 | t294: 1463 ± 404 | t294: 1463 ± 404 |
| Saniette | 8 | t121: 1233 ± 201 | t820: 1216 ± 167 | t820: 1216 ± 167 | t661: 1239 ± 159 |
| Beauserfeuil | 1 | t662: 1401 ± 357 | t662: 1401 ± 357 | t662: 1401 ± 357 | t662: 1401 ± 357 |
| Théodose Cadet | 1 | t665: 1400 ± 356 | t665: 1400 ± 356 | t665: 1400 ± 356 | t665: 1400 ± 356 |
| Cartier | 1 | t777: 1389 ± 348 | t777: 1389 ± 348 | t777: 1389 ± 348 | t777: 1389 ± 348 |
| M. Barrère | 1 | t884: 1533 ± 494 | t884: 1533 ± 494 | t884: 1533 ± 494 | t884: 1533 ± 494 |
| le professeur E… | 1 | t684: 1477 ± 444 | t684: 1477 ± 444 | t684: 1477 ± 444 | t684: 1477 ± 444 |
| professeur E… | 2 | t533: 1368 ± 339 | t685: 1373 ± 342 | t533: 1368 ± 339 | t685: 1373 ± 342 |
| Mme de Souvré | 2 | t591: 1263 ± 249 | t687: 1266 ± 246 | t591: 1263 ± 249 | t687: 1266 ± 246 |
| Mme de Morienval | 1 | t367: 1306 ± 294 | t367: 1306 ± 294 | t367: 1306 ± 294 | t367: 1306 ± 294 |
| duchesse de Luxembourg | 1 | t367: 1306 ± 294 | t367: 1306 ± 294 | t367: 1306 ± 294 | t367: 1306 ± 294 |
| Antoine | 1 | t358: 1389 ± 400 | t358: 1389 ± 400 | t358: 1389 ± 400 | t358: 1389 ± 400 |
| les Courvoisier | 1 | t595: 1306 ± 321 | t595: 1306 ± 321 | t595: 1306 ± 321 | t595: 1306 ± 321 |
| Maurice | 1 | t908: 1274 ± 305 | t908: 1274 ± 305 | t908: 1274 ± 305 | t908: 1274 ± 305 |
| Prince Henri d'Orléans | 1 | t483: 1378 ± 422 | t483: 1378 ± 422 | t483: 1378 ± 422 | t483: 1378 ± 422 |
| marquise de Citri | 1 | t700: 1367 ± 418 | t700: 1367 ± 418 | t700: 1367 ± 418 | t700: 1367 ± 418 |
| baron de Guermantes | 1 | t452: 1560 ± 611 | t452: 1560 ± 611 | t452: 1560 ± 611 | t452: 1560 ± 611 |
| M. Bontemps | 2 | t229: 1224 ± 296 | t899: 1260 ± 329 | t229: 1224 ± 296 | t899: 1260 ± 329 |
| comtesse G… | 1 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 |
| vicomtesse de Saint-Fiacre | 1 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 |
| Alix | 3 | t440: 1136 ± 326 | t445: 1136 ± 326 | t441: 1136 ± 326 | t440: 1136 ± 326 |
| ma grand'tante | 1 | t1: 1346 ± 538 | t1: 1346 ± 538 | t1: 1346 ± 538 | t1: 1346 ± 538 |
| prince Foggi | 1 | t883: 1345 ± 538 | t883: 1345 ± 538 | t883: 1345 ± 538 | t883: 1345 ± 538 |
| la marquise | 1 | t528: 1333 ± 532 | t528: 1333 ± 532 | t528: 1333 ± 532 | t528: 1333 ± 532 |
| Madame d'Ambresac | 1 | t366: 1293 ± 493 | t366: 1293 ± 493 | t366: 1293 ± 493 | t366: 1293 ± 493 |
| princesse de Nassau | 1 | t931: 1329 ± 530 | t931: 1329 ± 530 | t931: 1329 ± 530 | t931: 1329 ± 530 |
| Marie Gineste | 1 | t736: 1305 ± 512 | t736: 1305 ± 512 | t736: 1305 ± 512 | t736: 1305 ± 512 |
| prince de Léon | 1 | t775: 1279 ± 488 | t775: 1279 ± 488 | t775: 1279 ± 488 | t775: 1279 ± 488 |
| le curé | 1 | t42: 1281 ± 492 | t42: 1281 ± 492 | t42: 1281 ± 492 | t42: 1281 ± 492 |
| le prince von *** | 1 | t498: 1271 ± 483 | t498: 1271 ± 483 | t498: 1271 ± 483 | t498: 1271 ± 483 |
| le grand-duc héritier de Luxembourg | 1 | t581: 1303 ± 519 | t581: 1303 ± 519 | t581: 1303 ± 519 | t581: 1303 ± 519 |
| Dumont | 1 | t30: 1255 ± 481 | t30: 1255 ± 481 | t30: 1255 ± 481 | t30: 1255 ± 481 |
| le diplomate belge | 1 | t493: 1253 ± 481 | t493: 1253 ± 481 | t493: 1253 ± 481 | t493: 1253 ± 481 |
| Mme Blatin | 1 | t195: 1248 ± 476 | t195: 1248 ± 476 | t195: 1248 ± 476 | t195: 1248 ± 476 |
| M. de Luxembourg | 1 | t645: 1243 ± 472 | t645: 1243 ± 472 | t645: 1243 ± 472 | t645: 1243 ± 472 |
| l'historien de la Fronde | 1 | t453: 1227 ± 457 | t453: 1227 ± 457 | t453: 1227 ± 457 | t453: 1227 ± 457 |
| Mme de Simiane | 1 | t269: 1221 ± 452 | t269: 1221 ± 452 | t269: 1221 ± 452 | t269: 1221 ± 452 |
| prince de Faffenheim | 2 | t499: 1217 ± 451 | t500: 1217 ± 451 | t499: 1217 ± 451 | t499: 1217 ± 451 |
| colonel de Froberville | 1 | t696: 1072 ± 310 | t696: 1072 ± 310 | t696: 1072 ± 310 | t696: 1072 ± 310 |
| la cousine d'Oriane | 1 | t606: 1198 ± 444 | t606: 1198 ± 444 | t606: 1198 ± 444 | t606: 1198 ± 444 |
| vicomtesse d'Égremont | 1 | t593: 1196 ± 444 | t593: 1196 ± 444 | t593: 1196 ± 444 | t593: 1196 ± 444 |
| Monsieur Vallenères | 1 | t472: 1194 ± 443 | t472: 1194 ± 443 | t472: 1194 ± 443 | t472: 1194 ± 443 |
| princesse d'Iéna | 1 | t166: 1180 ± 443 | t166: 1180 ± 443 | t166: 1180 ± 443 | t166: 1180 ± 443 |
| l'ambassadrice de Turquie | 1 | t690: 1162 ± 425 | t690: 1162 ± 425 | t690: 1162 ± 425 | t690: 1162 ± 425 |
| Mme Blandais | 2 | t284: 1150 ± 422 | t288: 1150 ± 422 | t288: 1150 ± 422 | t284: 1150 ± 422 |
| M. Pierre | 2 | t438: 1146 ± 421 | t452: 1146 ± 422 | t452: 1146 ± 422 | t438: 1146 ± 421 |
| le prince de Faffenheim | 1 | t497: 1126 ± 410 | t497: 1126 ± 410 | t497: 1126 ± 410 | t497: 1126 ± 410 |
| Mme Iéna | 1 | t635: 1126 ± 410 | t635: 1126 ± 410 | t635: 1126 ± 410 | t635: 1126 ± 410 |
| Mme de Varambon | 2 | t616: 1135 ± 418 | t648: 1135 ± 419 | t648: 1135 ± 419 | t616: 1135 ± 418 |
| ma grand’tante | 1 | t2: 1118 ± 403 | t2: 1118 ± 403 | t2: 1118 ± 403 | t2: 1118 ± 403 |
| l'empereur | 1 | t640: 1118 ± 413 | t640: 1118 ± 413 | t640: 1118 ± 413 | t640: 1118 ± 413 |
| Picquart | 2 | t395: 1095 ± 398 | t482: 1092 ± 398 | t482: 1092 ± 398 | t395: 1095 ± 398 |
| M. de Vigny | 1 | t295: 996 ± 368 | t295: 996 ± 368 | t295: 996 ± 368 | t295: 996 ± 368 |
