# Character Whole-History Rating

- Analysis version: `character_whr_advantage_v1`
- Lens: `advantage`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `288`
- Match count: `5756`
- Time point count: `840`
- Node count: `3007`
- Draw rate: `0.321`
- Draw model: `half_win_half_loss`
- w2: `15.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.7`s, filtered `129.659`s (all w2 candidates `589.027`s)
- Convergence: smoothed `29` sweeps (converged: `True`), filtered `840` fits / `13203` sweeps, `0` of them unconverged
- Corpus: `foundation`

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.721232 | 0.258915 | 5756 | filtered WHR at w2=15 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.710343 | 0.255674 | 5756 | filtered WHR at w2=15, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.657827 | 0.232994 | 5756 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.696603 | 0.250774 | 5756 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.726877 | 0.262853 | 5756 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.721429 | 0.259204 | 98.716 |
| 15.0 | 0.721232 | 0.258915 | 129.659 |
| 35.0 | 0.722996 | 0.259381 | 164.491 |
| 60.0 | 0.725528 | 0.260149 | 196.161 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docteur du Boulbon | 1771 ± 188 | 1582.6 | 188.0 | 27 | 19-3-5 | 6 | 6 | -0.392 |
| le peintre | 1640 ± 118 | 1522.3 | 118.2 | 42 | 16-4-22 | 8 | 8 | -0.202 |
| Bergotte | 1646 ± 128 | 1519.0 | 127.5 | 129 | 52-31-46 | 36 | 32 | -0.062 |
| Françoise | 1625 ± 108 | 1517.3 | 108.0 | 217 | 101-48-68 | 82 | 76 | -0.26 |
| Rachel | 1585 ± 92 | 1492.8 | 92.5 | 146 | 52-53-41 | 43 | 43 | -1.086 |
| Elstir | 1580 ± 100 | 1480.1 | 100.0 | 106 | 40-29-37 | 29 | 24 | +0.174 |
| Aimé | 1579 ± 99 | 1479.9 | 99.0 | 79 | 27-14-38 | 18 | 18 | -0.418 |
| Rémi | 1650 ± 174 | 1476.2 | 174.0 | 17 | 5-0-12 | 3 | 3 | -0.533 |
| l'amie de Mlle Vinteuil | 1595 ± 127 | 1468.1 | 126.9 | 44 | 17-7-20 | 12 | 12 | -0.325 |
| le père du narrateur | 1600 ± 136 | 1464.4 | 136.1 | 90 | 35-22-33 | 24 | 24 | -0.753 |
| M. Verdurin | 1568 ± 104 | 1463.4 | 104.1 | 110 | 38-23-49 | 27 | 27 | -0.687 |
| Odette | 1543 ± 80 | 1462.9 | 80.0 | 462 | 167-157-138 | 142 | 138 | -0.718 |
| la grand-mère | 1568 ± 109 | 1459.4 | 108.9 | 225 | 93-66-66 | 80 | 74 | -0.325 |
| Mlle Vinteuil | 1558 ± 100 | 1457.9 | 100.1 | 71 | 21-15-35 | 15 | 15 | -0.714 |
| Jupien | 1552 ± 96 | 1455.8 | 95.9 | 68 | 23-14-31 | 18 | 18 | +0.118 |
| Mme Verdurin | 1532 ± 81 | 1450.2 | 81.4 | 311 | 93-96-122 | 82 | 82 | -0.893 |
| Morel | 1534 ± 84 | 1450.0 | 84.2 | 152 | 47-53-52 | 32 | 31 | -1.066 |
| Bloch | 1530 ± 81 | 1449.1 | 81.0 | 270 | 79-111-80 | 71 | 70 | -1.701 |
| Mme Sazerat | 1600 ± 162 | 1438.4 | 162.0 | 20 | 8-2-10 | 6 | 6 | -0.734 |
| la mère du narrateur | 1534 ± 97 | 1436.5 | 97.1 | 144 | 55-36-53 | 40 | 40 | -0.419 |
| Robert de Saint-Loup | 1507 ± 73 | 1434.2 | 72.9 | 508 | 166-213-129 | 168 | 154 | -0.602 |
| Norpois | 1567 ± 134 | 1432.9 | 133.8 | 180 | 80-54-46 | 63 | 62 | -0.65 |
| prince de Guermantes | 1543 ± 116 | 1426.4 | 116.3 | 124 | 42-30-52 | 22 | 22 | -0.843 |
| Mme de Charlus | 1610 ± 189 | 1420.4 | 189.3 | 15 | 5-1-9 | 2 | 2 | -0.8 |
| le grand-père du narrateur | 1585 ± 167 | 1418.0 | 167.0 | 63 | 26-7-30 | 16 | 16 | -0.627 |
| Mme de Surgis | 1547 ± 132 | 1414.2 | 132.4 | 42 | 16-11-15 | 9 | 9 | -0.967 |
| marquis de Bréauté | 1524 ± 110 | 1413.6 | 109.9 | 101 | 26-22-53 | 19 | 19 | -0.934 |
| Andrée | 1503 ± 90 | 1413.0 | 90.0 | 114 | 36-42-36 | 31 | 31 | -0.795 |
| Dreyfus | 1522 ± 111 | 1411.7 | 110.7 | 58 | 13-11-34 | 7 | 7 | -0.794 |
| Mme Cottard | 1597 ± 189 | 1408.1 | 188.8 | 33 | 16-7-10 | 11 | 11 | -0.431 |
| duchesse de Guermantes | 1476 ± 68 | 1407.8 | 68.2 | 662 | 334-177-151 | 199 | 194 | -0.075 |
| Mme Goupil | 1577 ± 170 | 1406.8 | 170.0 | 17 | 5-1-11 | 2 | 2 | -0.8 |
| comte de Forcheville | 1510 ± 112 | 1398.0 | 112.3 | 112 | 55-18-39 | 25 | 25 | -0.312 |
| docteur Cottard | 1500 ± 102 | 1397.7 | 101.8 | 194 | 46-63-85 | 43 | 43 | -0.978 |
| le narrateur | 1459 ± 65 | 1394.4 | 64.9 | 1093 | 400-508-185 | 316 | 315 | -0.85 |
| la marquise douairière de Cambremer | 1522 ± 129 | 1393.5 | 128.7 | 31 | 9-5-17 | 6 | 6 | +0.132 |
| Gilberte | 1462 ± 70 | 1391.6 | 70.4 | 312 | 114-103-95 | 76 | 74 | -0.516 |
| tante Léonie | 1551 ± 161 | 1390.0 | 160.7 | 38 | 12-22-4 | 22 | 20 | -0.865 |
| Charcot | 1587 ± 197 | 1389.7 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| M. Reinach | 1587 ± 197 | 1389.7 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| Mme Leroi | 1580 ± 195 | 1385.5 | 194.8 | 13 | 8-5-0 | 5 | 5 | -1.147 |
| Mme de Sévigné | 1544 ± 162 | 1382.0 | 162.3 | 25 | 7-5-13 | 4 | 4 | +0.097 |
| M. Vinteuil | 1504 ± 122 | 1381.6 | 122.0 | 61 | 18-19-24 | 15 | 15 | -0.388 |
| Mme de Villeparisis | 1492 ± 111 | 1380.7 | 111.3 | 236 | 90-93-53 | 79 | 78 | -0.726 |
| Albertine | 1458 ± 78 | 1380.6 | 77.9 | 387 | 149-156-82 | 146 | 126 | -0.868 |
| Swann | 1451 ± 72 | 1379.5 | 71.8 | 667 | 205-303-159 | 202 | 198 | -1.004 |
| Brichot | 1465 ± 88 | 1376.8 | 88.1 | 135 | 28-33-74 | 21 | 21 | -0.909 |
| marquise de Saint-Euverte | 1495 ± 118 | 1376.2 | 118.5 | 72 | 15-28-29 | 13 | 13 | -2.104 |
| M. de Chevregny | 1548 ± 173 | 1375.3 | 173.2 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| M. de Crécy | 1548 ± 173 | 1375.3 | 173.2 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| Mme Féré | 1548 ± 173 | 1375.3 | 173.2 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| Mme de Marsantes | 1477 ± 102 | 1374.8 | 102.2 | 107 | 20-34-53 | 21 | 20 | -1.22 |
| M. Ski | 1531 ± 156 | 1374.3 | 156.3 | 21 | 4-1-16 | 2 | 2 | -0.4 |
| baron de Charlus | 1446 ± 71 | 1374.2 | 71.4 | 485 | 189-160-136 | 119 | 118 | -0.809 |
| général de Froberville | 1529 ± 156 | 1372.4 | 156.3 | 27 | 7-4-16 | 7 | 7 | -0.622 |
| M. Nissim Bernard | 1505 ± 134 | 1370.9 | 133.7 | 39 | 9-10-20 | 10 | 7 | -1.591 |
| Bloch père | 1500 ± 131 | 1368.7 | 131.3 | 47 | 11-11-25 | 8 | 8 | -1.942 |
| Mme Bontemps | 1493 ± 125 | 1368.1 | 124.9 | 54 | 13-12-29 | 13 | 13 | -0.651 |
| Legrandin | 1469 ± 104 | 1365.1 | 104.2 | 83 | 15-28-40 | 24 | 20 | -1.39 |
| prince des Laumes | 1512 ± 147 | 1364.7 | 147.2 | 27 | 4-3-20 | 3 | 3 | -0.8 |
| Esther | 1537 ± 181 | 1356.0 | 181.1 | 14 | 3-2-9 | 2 | 2 | -1.0 |
| princesse de Luxembourg | 1508 ± 152 | 1355.9 | 151.9 | 25 | 6-7-12 | 6 | 6 | -0.816 |
| princesse de Guermantes | 1461 ± 112 | 1348.9 | 112.5 | 113 | 41-31-41 | 25 | 25 | -0.268 |
| duc de Chartres | 1531 ± 187 | 1343.9 | 187.1 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| prince de Chimay | 1531 ± 187 | 1343.9 | 187.1 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| le directeur | 1477 ± 135 | 1342.0 | 135.4 | 39 | 11-16-12 | 11 | 11 | -0.828 |
| le jeune marquis de Cambremer | 1536 ± 195 | 1341.0 | 194.7 | 12 | 2-1-9 | 1 | 1 | -1.2 |
| princesse de Parme | 1436 ± 96 | 1339.1 | 96.5 | 130 | 36-63-31 | 38 | 38 | -0.822 |
| comtesse Molé | 1470 ± 136 | 1333.5 | 136.1 | 34 | 6-9-19 | 6 | 6 | -1.365 |
| prince d’Agrigente | 1516 ± 183 | 1333.3 | 183.0 | 15 | 3-2-10 | 2 | 2 | -0.8 |
| Céline | 1517 ± 186 | 1331.2 | 185.6 | 16 | 4-6-6 | 2 | 2 | -1.225 |
| prince de Foix | 1515 ± 196 | 1318.7 | 196.1 | 14 | 4-4-6 | 3 | 3 | -0.95 |
| duc de Guermantes | 1405 ± 86 | 1318.5 | 86.2 | 401 | 123-171-107 | 110 | 107 | -1.136 |
| général de Monserfeuil | 1474 ± 166 | 1308.8 | 165.6 | 18 | 5-7-6 | 4 | 4 | -1.511 |
| M. d'Argencourt | 1464 ± 158 | 1305.1 | 158.4 | 56 | 19-18-19 | 14 | 12 | -1.286 |
| Rosemonde | 1474 ± 170 | 1303.4 | 170.3 | 20 | 5-7-8 | 4 | 4 | -0.7 |
| Goncourt | 1474 ± 171 | 1302.3 | 171.2 | 16 | 2-3-11 | 2 | 2 | -0.8 |
| M. de Vaugoubert | 1432 ± 137 | 1294.3 | 137.3 | 35 | 6-12-17 | 9 | 8 | -1.463 |
| Mme de Cambremer | 1391 ± 102 | 1288.9 | 101.7 | 112 | 12-53-47 | 20 | 19 | -1.709 |
| oncle Adolphe | 1470 ± 196 | 1274.7 | 195.6 | 20 | 5-11-4 | 6 | 5 | -1.8 |
| le petit Cambremer | 1460 ± 187 | 1273.3 | 187.0 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| princesse de Silistrie | 1460 ± 187 | 1273.3 | 187.0 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| la Berma | 1397 ± 141 | 1256.6 | 140.8 | 62 | 19-24-19 | 19 | 16 | -0.309 |
| Balzac | 1422 ± 184 | 1238.3 | 184.0 | 18 | 2-4-12 | 2 | 2 | -0.8 |
| marquis de Cambremer | 1325 ± 118 | 1206.6 | 118.4 | 45 | 7-24-14 | 6 | 6 | -1.173 |
| Mme d'Arpajon | 1330 ± 135 | 1195.2 | 135.0 | 37 | 6-23-8 | 8 | 8 | -1.85 |
| marquise de Gallardon | 1362 ± 188 | 1174.0 | 187.6 | 19 | 1-10-8 | 7 | 7 | -2.18 |
| princesse Sherbatoff | 1336 ± 173 | 1162.3 | 173.4 | 19 | 5-13-1 | 5 | 5 | -0.884 |
| Mme d'Heudicourt | 1322 ± 178 | 1143.8 | 178.2 | 18 | 3-11-4 | 5 | 5 | -1.7 |
| Mme de Franquetot | 1306 ± 170 | 1135.6 | 170.2 | 23 | 4-13-6 | 3 | 3 | -1.088 |
| Saniette | 1163 ± 181 | 981.9 | 181.3 | 35 | 1-27-7 | 9 | 8 | -3.455 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 2000 ± 363 | 362.8 | 14 | 1 | 1 | 888 |
| marquis de Beausergent | 1967 ± 372 | 372.5 | 12 | 1 | 1 | 923 |
| Mme Elstir | 1938 ± 386 | 386.1 | 7 | 1 | 1 | 333 |
| Mlle de Saint-Loup | 1935 ± 388 | 387.7 | 7 | 2 | 2 | 940 |
| Céleste Albaret | 1934 ± 276 | 275.5 | 17 | 3 | 3 | 806 |
| la reine de Naples | 1898 ± 275 | 275.4 | 17 | 3 | 3 | 828 |
| prince de Saxe | 1863 ± 427 | 427.1 | 3 | 1 | 1 | 365 |
| Marie | 1838 ± 318 | 318.0 | 7 | 1 | 1 | 737 |
| colonel Picquart | 1832 ± 429 | 428.6 | 4 | 1 | 1 | 481 |
| Mme de Chaussepierre | 1826 ± 430 | 430.3 | 4 | 1 | 1 | 777 |
| Mme de Grouchy | 1814 ± 440 | 440.0 | 4 | 1 | 1 | 598 |
| duchesse de La Trémoïlle | 1810 ± 441 | 441.2 | 3 | 1 | 1 | 119 |
| marquis Maurice de Vaudémont | 1798 ± 460 | 459.5 | 2 | 1 | 1 | 353 |
| Maeterlinck | 1788 ± 355 | 355.2 | 5 | 1 | 1 | 469 |
| Eulalie | 1774 ± 248 | 247.5 | 16 | 7 | 7 | 796 |
| Mlle Bloch | 1756 ± 472 | 471.6 | 2 | 1 | 1 | 732 |
| Bibi | 1751 ± 478 | 478.0 | 2 | 1 | 1 | 579 |
| Lady Israels | 1744 ± 476 | 476.5 | 2 | 1 | 1 | 232 |
| Victurnien | 1744 ± 271 | 271.4 | 8 | 2 | 2 | 704 |
| monsieur Vallenères | 1742 ± 478 | 477.8 | 2 | 1 | 1 | 457 |
| le commandant Duroc | 1741 ± 478 | 477.5 | 2 | 1 | 1 | 396 |
| duc de Sidonia | 1738 ± 486 | 486.4 | 2 | 1 | 1 | 684 |
| Gribelin | 1724 ± 318 | 318.4 | 6 | 1 | 1 | 482 |
| Léa | 1715 ± 216 | 215.7 | 14 | 4 | 4 | 852 |
| Émilie Daltier | 1702 ± 389 | 388.7 | 3 | 1 | 1 | 839 |
| Bismarck | 1689 ± 332 | 332.3 | 4 | 1 | 1 | 210 |
| Herbinger | 1687 ± 385 | 385.3 | 3 | 1 | 1 | 108 |
| Létourville | 1679 ± 387 | 386.7 | 3 | 1 | 1 | 921 |
| Duroc | 1675 ± 519 | 519.4 | 2 | 1 | 1 | 395 |
| Théodore | 1669 ± 416 | 415.9 | 2 | 1 | 1 | 59 |
| docteur Dieulafoy | 1667 ± 532 | 531.6 | 1 | 1 | 1 | 548 |
| elle | 1657 ± 536 | 536.4 | 1 | 1 | 1 | 430 |
| marquis du Lau | 1656 ± 328 | 327.9 | 5 | 2 | 2 | 869 |
| Dechambre | 1651 ± 404 | 403.5 | 3 | 1 | 1 | 745 |
| Mlle de Stermaria | 1647 ± 223 | 223.0 | 10 | 5 | 5 | 577 |
| la duchesse d'Alençon | 1646 ± 295 | 294.8 | 6 | 1 | 1 | 628 |
| grand-duc héritier de Luxembourg | 1643 ± 232 | 231.5 | 9 | 2 | 2 | 644 |
| les La Trémoïlle | 1642 ± 257 | 257.1 | 7 | 1 | 1 | 118 |
| Arnulphe | 1637 ± 324 | 324.1 | 4 | 1 | 1 | 703 |
| duc d'Aumale | 1636 ± 351 | 351.3 | 4 | 2 | 2 | 664 |
| Flora | 1635 ± 240 | 239.8 | 8 | 1 | 1 | 4 |
| M. de Courgivaux | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| Mme de Villebon | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| M. d'Orsan | 1628 ± 206 | 206.2 | 11 | 1 | 1 | 177 |
| le pianiste | 1628 ± 222 | 221.8 | 10 | 3 | 3 | 124 |
| Marie-Aynard | 1623 ± 257 | 257.1 | 7 | 1 | 1 | 480 |
| Victurnienne | 1623 ± 257 | 257.1 | 7 | 1 | 1 | 480 |
| cousine Poictiers | 1621 ± 294 | 293.6 | 5 | 1 | 1 | 414 |
| duc de Poictiers | 1621 ± 294 | 293.6 | 5 | 1 | 1 | 414 |
| M. Vibert | 1609 ± 357 | 357.3 | 3 | 1 | 1 | 618 |
| Mme de Stermaria | 1601 ± 289 | 288.6 | 5 | 1 | 1 | 566 |
| le baron Bréau-Chenut | 1594 ± 258 | 257.9 | 7 | 1 | 1 | 229 |
| le vieux père Chenut | 1594 ± 258 | 257.9 | 7 | 1 | 1 | 229 |
| Mme Legrandin mère | 1593 ± 242 | 242.4 | 8 | 1 | 1 | 266 |
| Victoire | 1593 ± 242 | 242.4 | 8 | 1 | 1 | 266 |
| Mme de Sagan | 1592 ± 359 | 358.9 | 3 | 1 | 1 | 485 |
| Sarah Bernhardt | 1584 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| le jeune prince de Foix | 1584 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| vicomte de Courvoisier | 1584 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| Manet | 1579 ± 289 | 289.3 | 5 | 1 | 1 | 637 |
| d'Orléans | 1578 ± 289 | 289.0 | 5 | 1 | 1 | 325 |
| M. de Beauserfeuil | 1573 ± 250 | 250.1 | 7 | 1 | 1 | 644 |
| Mlle de l’Orgeville | 1573 ± 358 | 358.2 | 3 | 1 | 1 | 892 |
| le grand-duc Wladimir | 1571 ± 367 | 366.9 | 3 | 1 | 1 | 689 |
| Lady Israël | 1570 ± 293 | 293.4 | 5 | 1 | 1 | 491 |
| jeune blonde de Rivebelle | 1566 ± 267 | 267.2 | 6 | 2 | 2 | 326 |
| M. de Bornier | 1564 ± 310 | 310.3 | 5 | 1 | 1 | 609 |
| Élisabeth | 1561 ± 267 | 266.8 | 6 | 1 | 1 | 791 |
| baron de Guermantes | 1560 ± 610 | 610.5 | 1 | 1 | 1 | 452 |
| duchesse de Létourville | 1556 ± 290 | 289.5 | 5 | 1 | 1 | 912 |
| M. de La Rochefoucauld | 1554 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| duchesse de La Rochefoucauld | 1554 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| duchesse de Praslin | 1554 ± 268 | 268.1 | 6 | 1 | 1 | 297 |
| Sir Rufus Israël | 1553 ± 260 | 259.5 | 7 | 1 | 1 | 459 |
| M. de Marsantes | 1548 ± 253 | 253.1 | 7 | 2 | 2 | 509 |
| le marquis de Ganançay | 1546 ± 285 | 285.1 | 6 | 1 | 1 | 367 |
| le marquis de Palancy | 1546 ± 285 | 285.1 | 6 | 1 | 1 | 367 |
| prince de Sagan | 1541 ± 251 | 250.6 | 7 | 1 | 1 | 710 |
| Octave | 1541 ± 332 | 332.0 | 4 | 2 | 2 | 875 |
| Mlle d'Éporcheville | 1535 ± 212 | 212.5 | 10 | 2 | 2 | 865 |
| M. de Goncourt | 1534 ± 236 | 235.8 | 8 | 1 | 1 | 897 |
| docteur Percepied | 1534 ± 313 | 312.6 | 4 | 1 | 1 | 58 |
| M. Barrère | 1533 ± 494 | 494.2 | 1 | 1 | 1 | 884 |
| L’excellent écrivain G… | 1529 ± 318 | 317.6 | 4 | 1 | 1 | 448 |
| Lady Rufus Israël | 1527 ± 266 | 266.4 | 6 | 1 | 1 | 868 |
| Coquelin | 1522 ± 289 | 288.6 | 5 | 1 | 1 | 198 |
| Mme Trombert | 1520 ± 314 | 313.5 | 4 | 1 | 1 | 231 |
| comtesse de Monteriender | 1518 ± 313 | 313.0 | 4 | 1 | 1 | 176 |
| d’Orgeville | 1515 ± 246 | 246.5 | 7 | 1 | 1 | 701 |
| Mme Putbus | 1515 ± 233 | 233.1 | 8 | 1 | 1 | 792 |
| Napoléon III | 1514 ± 238 | 238.4 | 8 | 1 | 1 | 186 |
| M. Swann, le père | 1513 ± 262 | 261.5 | 7 | 1 | 1 | 2 |
| le comte de Paris | 1513 ± 262 | 261.5 | 7 | 1 | 1 | 2 |
| le prince de Galles | 1513 ± 262 | 261.5 | 7 | 1 | 1 | 2 |
| prince Von | 1508 ± 235 | 235.3 | 8 | 3 | 3 | 641 |
| Mme de Montmorency | 1508 ± 204 | 203.6 | 11 | 1 | 1 | 718 |
| Mme de Rochechouart | 1508 ± 204 | 203.6 | 11 | 1 | 1 | 718 |
| M. Carnot | 1504 ± 221 | 220.6 | 9 | 1 | 1 | 663 |
| Mme Carnot | 1504 ± 221 | 220.6 | 9 | 1 | 1 | 663 |
| Mme Timoléon d'Amoncourt | 1501 ± 225 | 224.7 | 9 | 1 | 1 | 694 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| M. de Miribel | 1499 ± 313 | 313.3 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Henry | 1499 ± 313 | 313.3 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Picquart | 1499 ± 313 | 313.3 | 4 | 1 | 1 | 476 |
| M. Arthur Meyer | 1497 ± 264 | 264.2 | 6 | 1 | 1 | 911 |
| comte de Paris | 1497 ± 214 | 213.8 | 10 | 3 | 3 | 219 |
| Thibaud | 1495 ± 232 | 232.3 | 8 | 1 | 1 | 780 |
| Léonor de Cambremer | 1494 ± 202 | 201.9 | 12 | 1 | 1 | 923 |
| Liszt | 1490 ± 276 | 276.0 | 6 | 1 | 1 | 440 |
| Mme Ristori | 1490 ± 276 | 276.0 | 6 | 1 | 1 | 440 |
| M. Grevy | 1484 ± 348 | 348.4 | 3 | 1 | 1 | 94 |
| princesse d'Épinay | 1482 ± 204 | 204.2 | 12 | 3 | 3 | 608 |
| Dostoïevski | 1480 ± 264 | 263.5 | 6 | 1 | 1 | 836 |
| Sainte-Beuve | 1479 ± 250 | 250.3 | 7 | 1 | 1 | 583 |
| le capitaine | 1478 ± 402 | 402.4 | 2 | 1 | 1 | 375 |
| l'abbé Poiré | 1474 ± 211 | 211.3 | 10 | 1 | 1 | 708 |
| prince d'Agrigente | 1469 ± 414 | 413.9 | 2 | 2 | 2 | 922 |
| Poullein | 1468 ± 425 | 424.9 | 2 | 2 | 2 | 601 |
| Vigny | 1465 ± 404 | 403.8 | 2 | 1 | 1 | 294 |
| Barrès | 1463 ± 226 | 225.9 | 9 | 1 | 1 | 661 |
| Clémenceau | 1463 ± 226 | 225.9 | 9 | 1 | 1 | 661 |
| comtesse douairière d'Argencourt | 1461 ± 212 | 212.2 | 10 | 1 | 1 | 590 |
| duchesse de Gallardon douairière | 1461 ± 212 | 212.2 | 10 | 1 | 1 | 590 |
| marquis de Fierbois | 1461 ± 212 | 212.2 | 10 | 1 | 1 | 590 |
| Gisèle | 1458 ± 200 | 200.2 | 14 | 5 | 5 | 812 |
| Madame Elstir | 1455 ± 267 | 267.4 | 6 | 1 | 1 | 341 |
| les demoiselles d’Ambresac | 1455 ± 267 | 267.4 | 6 | 1 | 1 | 341 |
| M. de Chateaubriand | 1449 ± 209 | 209.4 | 11 | 2 | 2 | 870 |
| princesse Mathilde | 1442 ± 260 | 260.2 | 7 | 2 | 2 | 595 |
| le bâtonnier | 1442 ± 380 | 379.9 | 3 | 1 | 1 | 284 |
| Mme de Vaugoubert | 1440 ± 232 | 232.3 | 9 | 2 | 2 | 822 |
| M. de Stermaria | 1436 ± 215 | 215.1 | 10 | 4 | 4 | 280 |
| D'Annunzio | 1435 ± 290 | 289.9 | 5 | 1 | 1 | 693 |
| le roi Théodose | 1430 ± 250 | 249.8 | 8 | 3 | 3 | 693 |
| M. d'Herweck | 1423 ± 289 | 288.6 | 5 | 2 | 2 | 699 |
| Beauserfeuil | 1402 ± 357 | 356.8 | 3 | 1 | 1 | 662 |
| Théodose Cadet | 1402 ± 357 | 356.7 | 3 | 1 | 1 | 665 |
| M. Bontemps | 1396 ± 291 | 291.0 | 9 | 2 | 2 | 899 |
| Antoine | 1394 ± 402 | 401.6 | 3 | 1 | 1 | 358 |
| la jeune ouvriere | 1392 ± 420 | 420.1 | 2 | 1 | 1 | 96 |
| le prince Von | 1392 ± 223 | 223.1 | 10 | 2 | 2 | 640 |
| Cartier | 1391 ± 347 | 347.4 | 4 | 1 | 1 | 777 |
| Prince Henri d'Orléans | 1379 ± 421 | 421.4 | 2 | 1 | 1 | 483 |
| duc de Châtellerault | 1375 ± 248 | 248.0 | 10 | 5 | 5 | 683 |
| la Charité de Giotto | 1374 ± 554 | 553.8 | 1 | 1 | 1 | 49 |
| comtesse G… | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| vicomtesse de Saint-Fiacre | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| professeur E… | 1368 ± 341 | 341.1 | 4 | 2 | 2 | 685 |
| M. Molé | 1351 ± 248 | 247.9 | 8 | 1 | 1 | 295 |
| M. de Bouillon | 1351 ± 248 | 247.9 | 8 | 1 | 1 | 295 |
| Musset | 1351 ± 248 | 247.9 | 8 | 1 | 1 | 295 |
| Victor Hugo | 1351 ± 248 | 247.9 | 8 | 1 | 1 | 295 |
| ma grand'tante | 1347 ± 539 | 538.7 | 1 | 1 | 1 | 1 |
| prince Foggi | 1345 ± 538 | 537.5 | 1 | 1 | 1 | 883 |
| princesse de Nassau | 1330 ± 530 | 530.5 | 1 | 1 | 1 | 931 |
| la marquise | 1328 ± 529 | 529.3 | 1 | 1 | 1 | 528 |
| Monsieur Vallenères | 1314 ± 385 | 385.2 | 3 | 1 | 1 | 472 |
| le professeur E… | 1308 ± 504 | 504.4 | 2 | 1 | 1 | 684 |
| princesse d'Iéna | 1306 ± 394 | 393.9 | 3 | 1 | 1 | 166 |
| M. de Grouchy | 1306 ± 254 | 253.5 | 10 | 4 | 4 | 601 |
| les Courvoisier | 1305 ± 321 | 321.2 | 5 | 1 | 1 | 595 |
| le grand-duc héritier de Luxembourg | 1304 ± 520 | 519.6 | 1 | 1 | 1 | 581 |
| Marie Gineste | 1304 ± 512 | 512.2 | 2 | 1 | 1 | 736 |
| Mme de Morienval | 1303 ± 293 | 293.3 | 6 | 1 | 1 | 367 |
| duchesse de Luxembourg | 1303 ± 293 | 293.3 | 6 | 1 | 1 | 367 |
| le curé | 1293 ± 499 | 499.0 | 2 | 1 | 1 | 42 |
| Madame d'Ambresac | 1289 ± 492 | 491.5 | 2 | 1 | 1 | 366 |
| prince de Léon | 1279 ± 489 | 488.6 | 2 | 1 | 1 | 775 |
| Maurice | 1271 ± 305 | 304.7 | 7 | 1 | 1 | 908 |
| le prince von *** | 1271 ± 483 | 482.6 | 2 | 1 | 1 | 498 |
| Mme de Souvré | 1262 ± 246 | 246.0 | 11 | 2 | 2 | 687 |
| le diplomate belge | 1255 ± 482 | 481.9 | 2 | 1 | 1 | 493 |
| Dumont | 1254 ± 481 | 481.1 | 2 | 1 | 1 | 30 |
| Mme Blatin | 1246 ± 476 | 475.9 | 2 | 1 | 1 | 195 |
| capitaine de Borodino | 1246 ± 223 | 222.9 | 14 | 5 | 5 | 459 |
| M. de Luxembourg | 1244 ± 472 | 472.0 | 2 | 1 | 1 | 645 |
| marquise de Citri | 1230 ± 467 | 467.2 | 2 | 1 | 1 | 700 |
| l'historien de la Fronde | 1226 ± 457 | 456.7 | 3 | 1 | 1 | 453 |
| Mme de Simiane | 1224 ± 454 | 453.7 | 3 | 1 | 1 | 269 |
| prince de Faffenheim | 1217 ± 451 | 451.3 | 3 | 2 | 2 | 500 |
| la cousine d'Oriane | 1197 ± 444 | 443.9 | 3 | 1 | 1 | 606 |
| vicomtesse d'Égremont | 1197 ± 445 | 444.8 | 3 | 1 | 1 | 593 |
| l'ambassadrice de Turquie | 1162 ± 425 | 425.2 | 4 | 1 | 1 | 690 |
| Mme Blandais | 1152 ± 423 | 423.1 | 4 | 2 | 2 | 288 |
| M. Pierre | 1146 ± 422 | 421.7 | 4 | 2 | 2 | 452 |
| Alix | 1136 ± 326 | 325.9 | 9 | 3 | 3 | 445 |
| Mme de Varambon | 1134 ± 419 | 419.0 | 4 | 2 | 2 | 648 |
| Mme Iéna | 1125 ± 410 | 410.1 | 5 | 1 | 1 | 635 |
| l'empereur | 1125 ± 415 | 414.8 | 4 | 1 | 1 | 640 |
| le prince de Faffenheim | 1123 ± 409 | 409.2 | 5 | 1 | 1 | 497 |
| ma grand’tante | 1119 ± 403 | 403.3 | 7 | 1 | 1 | 2 |
| Picquart | 1093 ± 398 | 398.1 | 8 | 2 | 2 | 482 |
| M. de Vigny | 997 ± 368 | 368.5 | 8 | 1 | 1 | 295 |
| colonel de Froberville | 992 ± 361 | 361.0 | 14 | 1 | 1 | 696 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Céleste Albaret | 3 | t736: 1936 ± 274 | t806: 1934 ± 276 | t806: 1934 ± 276 | t736: 1936 ± 274 |
| Mlle d'Oloron | 1 | t888: 2000 ± 363 | t888: 2000 ± 363 | t888: 2000 ± 363 | t888: 2000 ± 363 |
| la reine de Naples | 3 | t628: 1899 ± 276 | t828: 1898 ± 275 | t828: 1898 ± 275 | t628: 1899 ± 276 |
| marquis de Beausergent | 1 | t923: 1967 ± 372 | t923: 1967 ± 372 | t923: 1967 ± 372 | t923: 1967 ± 372 |
| docteur du Boulbon | 6 | t248: 1749 ± 178 | t725: 1771 ± 188 | t248: 1749 ± 178 | t523: 1774 ± 165 |
| Mme Elstir | 1 | t333: 1938 ± 386 | t333: 1938 ± 386 | t333: 1938 ± 386 | t333: 1938 ± 386 |
| Mlle de Saint-Loup | 2 | t939: 1935 ± 388 | t940: 1935 ± 388 | t939: 1935 ± 388 | t939: 1935 ± 388 |
| Eulalie | 7 | t19: 1741 ± 207 | t796: 1774 ± 248 | t19: 1741 ± 207 | t796: 1774 ± 248 |
| le peintre | 8 | t89: 1649 ± 122 | t186: 1640 ± 118 | t186: 1640 ± 118 | t114: 1650 ± 117 |
| Marie | 1 | t737: 1838 ± 318 | t737: 1838 ± 318 | t737: 1838 ± 318 | t737: 1838 ± 318 |
| Bergotte | 32 | t28: 1535 ± 116 | t941: 1646 ± 128 | t28: 1535 ± 116 | t941: 1646 ± 128 |
| Françoise | 76 | t2: 1632 ± 95 | t940: 1625 ± 108 | t536: 1598 ± 84 | t59: 1638 ± 91 |
| Léa | 4 | t807: 1713 ± 212 | t852: 1715 ± 216 | t807: 1713 ± 212 | t852: 1715 ± 216 |
| Rachel | 43 | t251: 1450 ± 125 | t939: 1585 ± 92 | t469: 1427 ± 81 | t938: 1585 ± 92 |
| Elstir | 24 | t269: 1538 ± 112 | t904: 1580 ± 100 | t617: 1535 ± 98 | t898: 1580 ± 98 |
| Aimé | 18 | t279: 1510 ± 130 | t890: 1579 ± 99 | t279: 1510 ± 130 | t791: 1583 ± 92 |
| Rémi | 3 | t101: 1651 ± 177 | t177: 1650 ± 174 | t177: 1650 ± 174 | t101: 1651 ± 177 |
| Victurnien | 2 | t703: 1744 ± 271 | t704: 1744 ± 271 | t703: 1744 ± 271 | t704: 1744 ± 271 |
| l'amie de Mlle Vinteuil | 12 | t58: 1596 ± 145 | t855: 1595 ± 127 | t823: 1594 ± 123 | t762: 1597 ± 125 |
| le père du narrateur | 24 | t4: 1585 ± 98 | t550: 1600 ± 136 | t197: 1580 ± 88 | t547: 1600 ± 136 |
| M. Verdurin | 27 | t70: 1544 ± 98 | t904: 1568 ± 104 | t745: 1541 ± 94 | t904: 1568 ± 104 |
| Odette | 138 | t21: 1568 ± 80 | t938: 1543 ± 80 | t490: 1465 ± 80 | t21: 1568 ± 80 |
| la grand-mère | 74 | t1: 1571 ± 96 | t917: 1568 ± 109 | t412: 1545 ± 78 | t731: 1581 ± 95 |
| Mlle Vinteuil | 15 | t45: 1552 ± 135 | t855: 1558 ± 100 | t61: 1550 ± 133 | t762: 1561 ± 101 |
| Jupien | 18 | t356: 1596 ± 145 | t913: 1552 ± 96 | t888: 1551 ± 93 | t356: 1596 ± 145 |
| Mme Verdurin | 82 | t70: 1483 ± 74 | t934: 1532 ± 81 | t86: 1483 ± 68 | t927: 1532 ± 79 |
| Morel | 31 | t501: 1423 ± 127 | t928: 1534 ± 84 | t501: 1423 ± 127 | t928: 1534 ± 84 |
| Bloch | 70 | t29: 1367 ± 126 | t940: 1530 ± 81 | t29: 1367 ± 126 | t931: 1530 ± 78 |
| Mme Sazerat | 6 | t416: 1628 ± 206 | t882: 1600 ± 162 | t870: 1600 ± 161 | t416: 1628 ± 206 |
| la mère du narrateur | 40 | t4: 1611 ± 108 | t888: 1534 ± 97 | t888: 1534 ± 97 | t4: 1611 ± 108 |
| prince de Saxe | 1 | t365: 1863 ± 427 | t365: 1863 ± 427 | t365: 1863 ± 427 | t365: 1863 ± 427 |
| Robert de Saint-Loup | 154 | t298: 1492 ± 71 | t939: 1507 ± 73 | t477: 1433 ± 54 | t911: 1509 ± 64 |
| Maeterlinck | 1 | t469: 1788 ± 355 | t469: 1788 ± 355 | t469: 1788 ± 355 | t469: 1788 ± 355 |
| Norpois | 62 | t201: 1577 ± 76 | t915: 1567 ± 134 | t350: 1554 ± 80 | t201: 1577 ± 76 |
| prince de Guermantes | 22 | t477: 1505 ± 104 | t927: 1543 ± 116 | t477: 1505 ± 104 | t708: 1543 ± 73 |
| Mlle de Stermaria | 5 | t280: 1594 ± 232 | t577: 1647 ± 223 | t280: 1594 ± 232 | t576: 1647 ± 223 |
| M. d'Orsan | 1 | t177: 1628 ± 206 | t177: 1628 ± 206 | t177: 1628 ± 206 | t177: 1628 ± 206 |
| Mme de Charlus | 2 | t621: 1608 ± 190 | t855: 1610 ± 189 | t621: 1608 ± 190 | t855: 1610 ± 189 |
| le grand-père du narrateur | 16 | t2: 1645 ± 101 | t549: 1585 ± 167 | t547: 1585 ± 167 | t30: 1648 ± 99 |
| Mme de Surgis | 9 | t687: 1553 ± 112 | t817: 1547 ± 132 | t817: 1547 ± 132 | t687: 1553 ± 112 |
| marquis de Bréauté | 19 | t157: 1520 ± 130 | t938: 1524 ± 110 | t450: 1512 ± 111 | t623: 1526 ± 86 |
| Andrée | 31 | t341: 1505 ± 105 | t875: 1503 ± 90 | t781: 1490 ± 76 | t345: 1506 ± 104 |
| Dreyfus | 7 | t324: 1532 ± 130 | t708: 1522 ± 111 | t708: 1522 ± 111 | t421: 1533 ± 114 |
| grand-duc héritier de Luxembourg | 2 | t540: 1649 ± 236 | t644: 1643 ± 232 | t644: 1643 ± 232 | t540: 1649 ± 236 |
| Mme Cottard | 11 | t87: 1625 ± 140 | t756: 1597 ± 189 | t756: 1597 ± 189 | t186: 1629 ± 132 |
| duchesse de Guermantes | 194 | t67: 1622 ± 126 | t939: 1476 ± 68 | t938: 1476 ± 68 | t412: 1683 ± 67 |
| Mme Goupil | 2 | t870: 1577 ± 170 | t871: 1577 ± 170 | t870: 1577 ± 170 | t871: 1577 ± 170 |
| le pianiste | 3 | t85: 1629 ± 222 | t124: 1628 ± 222 | t124: 1628 ± 222 | t85: 1629 ± 222 |
| Gribelin | 1 | t482: 1724 ± 318 | t482: 1724 ± 318 | t482: 1724 ± 318 | t482: 1724 ± 318 |
| colonel Picquart | 1 | t481: 1832 ± 429 | t481: 1832 ± 429 | t481: 1832 ± 429 | t481: 1832 ± 429 |
| comte de Forcheville | 25 | t110: 1695 ± 89 | t938: 1510 ± 112 | t928: 1509 ± 111 | t124: 1696 ± 86 |
| docteur Cottard | 43 | t71: 1455 ± 86 | t923: 1500 ± 102 | t77: 1455 ± 84 | t897: 1500 ± 97 |
| Mme de Chaussepierre | 1 | t777: 1826 ± 430 | t777: 1826 ± 430 | t777: 1826 ± 430 | t777: 1826 ± 430 |
| Flora | 1 | t4: 1635 ± 240 | t4: 1635 ± 240 | t4: 1635 ± 240 | t4: 1635 ± 240 |
| le narrateur | 315 | t4: 1437 ± 85 | t941: 1459 ± 65 | t806: 1437 ± 46 | t623: 1567 ± 51 |
| la marquise douairière de Cambremer | 6 | t158: 1503 ± 201 | t761: 1522 ± 129 | t158: 1503 ± 201 | t761: 1522 ± 129 |
| Gilberte | 74 | t37: 1604 ± 108 | t939: 1462 ± 70 | t939: 1462 ± 70 | t37: 1604 ± 108 |
| tante Léonie | 20 | t8: 1513 ± 126 | t361: 1551 ± 161 | t56: 1509 ± 123 | t361: 1551 ± 161 |
| Charcot | 1 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 |
| M. Reinach | 1 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 | t523: 1587 ± 197 |
| Mme Leroi | 5 | t436: 1581 ± 195 | t506: 1580 ± 195 | t505: 1580 ± 195 | t436: 1581 ± 195 |
| les La Trémoïlle | 1 | t118: 1642 ± 257 | t118: 1642 ± 257 | t118: 1642 ± 257 | t118: 1642 ± 257 |
| Mme de Sévigné | 4 | t269: 1565 ± 171 | t729: 1544 ± 162 | t729: 1544 ± 162 | t269: 1565 ± 171 |
| M. Vinteuil | 15 | t45: 1514 ± 123 | t898: 1504 ± 122 | t898: 1504 ± 122 | t176: 1529 ± 124 |
| Mme de Villeparisis | 78 | t3: 1471 ± 141 | t882: 1492 ± 111 | t590: 1464 ± 72 | t472: 1508 ± 60 |
| Albertine | 126 | t229: 1562 ± 100 | t918: 1458 ± 78 | t873: 1457 ± 61 | t345: 1596 ± 76 |
| Swann | 198 | t2: 1512 ± 77 | t938: 1451 ± 72 | t718: 1441 ± 56 | t2: 1512 ± 77 |
| Brichot | 21 | t111: 1513 ± 126 | t923: 1465 ± 88 | t905: 1464 ± 84 | t118: 1513 ± 125 |
| marquise de Saint-Euverte | 13 | t163: 1339 ± 169 | t938: 1495 ± 118 | t163: 1339 ± 169 | t938: 1495 ± 118 |
| M. de Chevregny | 1 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 |
| M. de Crécy | 1 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 |
| Mme Féré | 1 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 | t761: 1548 ± 173 |
| Mme de Marsantes | 20 | t232: 1442 ± 141 | t890: 1477 ± 102 | t421: 1434 ± 101 | t890: 1477 ± 102 |
| M. Ski | 2 | t748: 1524 ± 157 | t825: 1531 ± 156 | t748: 1524 ± 157 | t825: 1531 ± 156 |
| Mme de Grouchy | 1 | t598: 1814 ± 440 | t598: 1814 ± 440 | t598: 1814 ± 440 | t598: 1814 ± 440 |
| baron de Charlus | 118 | t56: 1584 ± 110 | t938: 1446 ± 71 | t912: 1441 ± 63 | t522: 1584 ± 68 |
| général de Froberville | 7 | t157: 1523 ± 158 | t696: 1529 ± 156 | t157: 1523 ± 158 | t696: 1529 ± 156 |
| M. Nissim Bernard | 7 | t315: 1516 ± 171 | t923: 1505 ± 134 | t923: 1505 ± 134 | t509: 1520 ± 145 |
| duchesse de La Trémoïlle | 1 | t119: 1810 ± 441 | t119: 1810 ± 441 | t119: 1810 ± 441 | t119: 1810 ± 441 |
| Bloch père | 8 | t313: 1448 ± 137 | t923: 1500 ± 131 | t315: 1448 ± 137 | t761: 1502 ± 117 |
| Mme Bontemps | 13 | t229: 1520 ± 124 | t899: 1493 ± 125 | t898: 1493 ± 125 | t229: 1520 ± 124 |
| Marie-Aynard | 1 | t480: 1623 ± 257 | t480: 1623 ± 257 | t480: 1623 ± 257 | t480: 1623 ± 257 |
| Victurnienne | 1 | t480: 1623 ± 257 | t480: 1623 ± 257 | t480: 1623 ± 257 | t480: 1623 ± 257 |
| Legrandin | 20 | t17: 1395 ± 156 | t930: 1469 ± 104 | t266: 1394 ± 129 | t761: 1475 ± 96 |
| prince des Laumes | 3 | t177: 1558 ± 155 | t596: 1512 ± 147 | t596: 1512 ± 147 | t177: 1558 ± 155 |
| Bismarck | 1 | t210: 1689 ± 332 | t210: 1689 ± 332 | t210: 1689 ± 332 | t210: 1689 ± 332 |
| Esther | 2 | t791: 1537 ± 181 | t792: 1537 ± 181 | t791: 1537 ± 181 | t791: 1537 ± 181 |
| princesse de Luxembourg | 6 | t283: 1501 ± 163 | t730: 1508 ± 152 | t283: 1501 ± 163 | t644: 1512 ± 148 |
| la duchesse d'Alençon | 1 | t628: 1646 ± 295 | t628: 1646 ± 295 | t628: 1646 ± 295 | t628: 1646 ± 295 |
| Mme Legrandin mère | 1 | t266: 1593 ± 242 | t266: 1593 ± 242 | t266: 1593 ± 242 | t266: 1593 ± 242 |
| Victoire | 1 | t266: 1593 ± 242 | t266: 1593 ± 242 | t266: 1593 ± 242 | t266: 1593 ± 242 |
| princesse de Guermantes | 25 | t363: 1565 ± 113 | t932: 1461 ± 112 | t932: 1461 ± 112 | t366: 1565 ± 113 |
| duc de Chartres | 1 | t696: 1531 ± 187 | t696: 1531 ± 187 | t696: 1531 ± 187 | t696: 1531 ± 187 |
| prince de Chimay | 1 | t696: 1531 ± 187 | t696: 1531 ± 187 | t696: 1531 ± 187 | t696: 1531 ± 187 |
| le directeur | 11 | t270: 1504 ± 134 | t737: 1477 ± 135 | t737: 1477 ± 135 | t270: 1504 ± 134 |
| le jeune marquis de Cambremer | 1 | t890: 1536 ± 195 | t890: 1536 ± 195 | t890: 1536 ± 195 | t890: 1536 ± 195 |
| princesse de Parme | 38 | t363: 1408 ± 127 | t724: 1436 ± 96 | t570: 1408 ± 78 | t724: 1436 ± 96 |
| marquis Maurice de Vaudémont | 1 | t353: 1798 ± 460 | t353: 1798 ± 460 | t353: 1798 ± 460 | t353: 1798 ± 460 |
| le baron Bréau-Chenut | 1 | t229: 1594 ± 258 | t229: 1594 ± 258 | t229: 1594 ± 258 | t229: 1594 ± 258 |
| le vieux père Chenut | 1 | t229: 1594 ± 258 | t229: 1594 ± 258 | t229: 1594 ± 258 | t229: 1594 ± 258 |
| comtesse Molé | 6 | t668: 1457 ± 129 | t870: 1470 ± 136 | t668: 1457 ± 129 | t870: 1470 ± 136 |
| prince d’Agrigente | 2 | t630: 1521 ± 187 | t870: 1516 ± 183 | t870: 1516 ± 183 | t630: 1521 ± 187 |
| Céline | 2 | t4: 1493 ± 184 | t266: 1517 ± 186 | t4: 1493 ± 184 | t266: 1517 ± 186 |
| marquis du Lau | 2 | t775: 1649 ± 328 | t869: 1656 ± 328 | t775: 1649 ± 328 | t869: 1656 ± 328 |
| cousine Poictiers | 1 | t414: 1621 ± 294 | t414: 1621 ± 294 | t414: 1621 ± 294 | t414: 1621 ± 294 |
| duc de Poictiers | 1 | t414: 1621 ± 294 | t414: 1621 ± 294 | t414: 1621 ± 294 | t414: 1621 ± 294 |
| Sarah Bernhardt | 1 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 |
| le jeune prince de Foix | 1 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 |
| vicomte de Courvoisier | 1 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 | t908: 1584 ± 260 |
| M. de Beauserfeuil | 1 | t644: 1573 ± 250 | t644: 1573 ± 250 | t644: 1573 ± 250 | t644: 1573 ± 250 |
| Mlle d'Éporcheville | 2 | t863: 1535 ± 213 | t865: 1535 ± 212 | t865: 1535 ± 212 | t863: 1535 ± 213 |
| prince de Foix | 3 | t580: 1490 ± 196 | t908: 1515 ± 196 | t580: 1490 ± 196 | t908: 1515 ± 196 |
| duc de Guermantes | 107 | t362: 1466 ± 99 | t938: 1405 ± 86 | t938: 1405 ± 86 | t464: 1472 ± 71 |
| Émilie Daltier | 1 | t839: 1702 ± 389 | t839: 1702 ± 389 | t839: 1702 ± 389 | t839: 1702 ± 389 |
| Arnulphe | 1 | t703: 1637 ± 324 | t703: 1637 ± 324 | t703: 1637 ± 324 | t703: 1637 ± 324 |
| Mme de Stermaria | 1 | t566: 1601 ± 289 | t566: 1601 ± 289 | t566: 1601 ± 289 | t566: 1601 ± 289 |
| général de Monserfeuil | 4 | t628: 1475 ± 166 | t631: 1474 ± 166 | t631: 1474 ± 166 | t628: 1475 ± 166 |
| M. d'Argencourt | 12 | t453: 1528 ± 102 | t911: 1464 ± 158 | t911: 1464 ± 158 | t464: 1528 ± 99 |
| Mme de Montmorency | 1 | t718: 1508 ± 204 | t718: 1508 ± 204 | t718: 1508 ± 204 | t718: 1508 ± 204 |
| Mme de Rochechouart | 1 | t718: 1508 ± 204 | t718: 1508 ± 204 | t718: 1508 ± 204 | t718: 1508 ± 204 |
| Rosemonde | 4 | t345: 1456 ± 168 | t729: 1474 ± 170 | t345: 1456 ± 168 | t727: 1474 ± 170 |
| Goncourt | 2 | t896: 1474 ± 171 | t898: 1474 ± 171 | t898: 1474 ± 171 | t896: 1474 ± 171 |
| Herbinger | 1 | t108: 1687 ± 385 | t108: 1687 ± 385 | t108: 1687 ± 385 | t108: 1687 ± 385 |
| jeune blonde de Rivebelle | 2 | t325: 1566 ± 267 | t326: 1566 ± 267 | t325: 1566 ± 267 | t325: 1566 ± 267 |
| M. de Goncourt | 1 | t897: 1534 ± 236 | t897: 1534 ± 236 | t897: 1534 ± 236 | t897: 1534 ± 236 |
| M. de Marsantes | 2 | t299: 1537 ± 264 | t509: 1548 ± 253 | t299: 1537 ± 264 | t509: 1548 ± 253 |
| M. de Vaugoubert | 8 | t209: 1477 ± 190 | t822: 1432 ± 137 | t778: 1430 ± 133 | t209: 1477 ± 190 |
| Élisabeth | 1 | t791: 1561 ± 267 | t791: 1561 ± 267 | t791: 1561 ± 267 | t791: 1561 ± 267 |
| Sir Rufus Israël | 1 | t459: 1553 ± 260 | t459: 1553 ± 260 | t459: 1553 ± 260 | t459: 1553 ± 260 |
| Létourville | 1 | t921: 1679 ± 387 | t921: 1679 ± 387 | t921: 1679 ± 387 | t921: 1679 ± 387 |
| Léonor de Cambremer | 1 | t923: 1494 ± 202 | t923: 1494 ± 202 | t923: 1494 ± 202 | t923: 1494 ± 202 |
| prince de Sagan | 1 | t710: 1541 ± 251 | t710: 1541 ± 251 | t710: 1541 ± 251 | t710: 1541 ± 251 |
| Manet | 1 | t637: 1579 ± 289 | t637: 1579 ± 289 | t637: 1579 ± 289 | t637: 1579 ± 289 |
| d'Orléans | 1 | t325: 1578 ± 289 | t325: 1578 ± 289 | t325: 1578 ± 289 | t325: 1578 ± 289 |
| Mme de Cambremer | 19 | t165: 1363 ± 143 | t923: 1391 ± 102 | t694: 1350 ± 87 | t923: 1391 ± 102 |
| M. de La Rochefoucauld | 1 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 |
| duchesse de La Rochefoucauld | 1 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 |
| duchesse de Praslin | 1 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 | t297: 1554 ± 268 |
| duc d'Aumale | 2 | t366: 1620 ± 347 | t664: 1636 ± 351 | t366: 1620 ± 347 | t664: 1636 ± 351 |
| Mlle Bloch | 1 | t732: 1756 ± 472 | t732: 1756 ± 472 | t732: 1756 ± 472 | t732: 1756 ± 472 |
| M. Carnot | 1 | t663: 1504 ± 221 | t663: 1504 ± 221 | t663: 1504 ± 221 | t663: 1504 ± 221 |
| Mme Carnot | 1 | t663: 1504 ± 221 | t663: 1504 ± 221 | t663: 1504 ± 221 | t663: 1504 ± 221 |
| comte de Paris | 3 | t192: 1495 ± 214 | t219: 1497 ± 214 | t192: 1495 ± 214 | t219: 1497 ± 214 |
| Mme Putbus | 1 | t792: 1515 ± 233 | t792: 1515 ± 233 | t792: 1515 ± 233 | t792: 1515 ± 233 |
| princesse d'Épinay | 3 | t593: 1482 ± 204 | t608: 1482 ± 204 | t608: 1482 ± 204 | t595: 1482 ± 204 |
| Lady Israël | 1 | t491: 1570 ± 293 | t491: 1570 ± 293 | t491: 1570 ± 293 | t491: 1570 ± 293 |
| Mme Timoléon d'Amoncourt | 1 | t694: 1501 ± 225 | t694: 1501 ± 225 | t694: 1501 ± 225 | t694: 1501 ± 225 |
| Napoléon III | 1 | t186: 1514 ± 238 | t186: 1514 ± 238 | t186: 1514 ± 238 | t186: 1514 ± 238 |
| oncle Adolphe | 5 | t21: 1406 ± 169 | t501: 1470 ± 196 | t21: 1406 ± 169 | t501: 1470 ± 196 |
| le petit Cambremer | 1 | t888: 1460 ± 187 | t888: 1460 ± 187 | t888: 1460 ± 187 | t888: 1460 ± 187 |
| princesse de Silistrie | 1 | t888: 1460 ± 187 | t888: 1460 ± 187 | t888: 1460 ± 187 | t888: 1460 ± 187 |
| Bibi | 1 | t579: 1751 ± 478 | t579: 1751 ± 478 | t579: 1751 ± 478 | t579: 1751 ± 478 |
| prince Von | 3 | t623: 1509 ± 234 | t641: 1508 ± 235 | t641: 1508 ± 235 | t623: 1509 ± 234 |
| d’Orgeville | 1 | t701: 1515 ± 246 | t701: 1515 ± 246 | t701: 1515 ± 246 | t701: 1515 ± 246 |
| Lady Israels | 1 | t232: 1744 ± 476 | t232: 1744 ± 476 | t232: 1744 ± 476 | t232: 1744 ± 476 |
| duchesse de Létourville | 1 | t912: 1556 ± 290 | t912: 1556 ± 290 | t912: 1556 ± 290 | t912: 1556 ± 290 |
| monsieur Vallenères | 1 | t457: 1742 ± 478 | t457: 1742 ± 478 | t457: 1742 ± 478 | t457: 1742 ± 478 |
| le commandant Duroc | 1 | t396: 1741 ± 478 | t396: 1741 ± 478 | t396: 1741 ± 478 | t396: 1741 ± 478 |
| Thibaud | 1 | t780: 1495 ± 232 | t780: 1495 ± 232 | t780: 1495 ± 232 | t780: 1495 ± 232 |
| l'abbé Poiré | 1 | t708: 1474 ± 211 | t708: 1474 ± 211 | t708: 1474 ± 211 | t708: 1474 ± 211 |
| le marquis de Ganançay | 1 | t367: 1546 ± 285 | t367: 1546 ± 285 | t367: 1546 ± 285 | t367: 1546 ± 285 |
| le marquis de Palancy | 1 | t367: 1546 ± 285 | t367: 1546 ± 285 | t367: 1546 ± 285 | t367: 1546 ± 285 |
| Lady Rufus Israël | 1 | t868: 1527 ± 266 | t868: 1527 ± 266 | t868: 1527 ± 266 | t868: 1527 ± 266 |
| Gisèle | 5 | t342: 1414 ± 212 | t812: 1458 ± 200 | t342: 1414 ± 212 | t812: 1458 ± 200 |
| la Berma | 16 | t21: 1571 ± 132 | t936: 1397 ± 141 | t936: 1397 ± 141 | t21: 1571 ± 132 |
| M. de Bornier | 1 | t609: 1564 ± 310 | t609: 1564 ± 310 | t609: 1564 ± 310 | t609: 1564 ± 310 |
| Théodore | 1 | t59: 1669 ± 416 | t59: 1669 ± 416 | t59: 1669 ± 416 | t59: 1669 ± 416 |
| M. Vibert | 1 | t618: 1609 ± 357 | t618: 1609 ± 357 | t618: 1609 ± 357 | t618: 1609 ± 357 |
| M. Swann, le père | 1 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 |
| le comte de Paris | 1 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 |
| le prince de Galles | 1 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 | t2: 1513 ± 262 |
| duc de Sidonia | 1 | t684: 1738 ± 486 | t684: 1738 ± 486 | t684: 1738 ± 486 | t684: 1738 ± 486 |
| comtesse douairière d'Argencourt | 1 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 |
| duchesse de Gallardon douairière | 1 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 |
| marquis de Fierbois | 1 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 | t590: 1461 ± 212 |
| Dechambre | 1 | t745: 1651 ± 404 | t745: 1651 ± 404 | t745: 1651 ± 404 | t745: 1651 ± 404 |
| M. de Chateaubriand | 2 | t294: 1414 ± 242 | t870: 1449 ± 209 | t294: 1414 ± 242 | t870: 1449 ± 209 |
| Balzac | 2 | t295: 1394 ± 190 | t898: 1422 ± 184 | t295: 1394 ± 190 | t898: 1422 ± 184 |
| Barrès | 1 | t661: 1463 ± 226 | t661: 1463 ± 226 | t661: 1463 ± 226 | t661: 1463 ± 226 |
| Clémenceau | 1 | t661: 1463 ± 226 | t661: 1463 ± 226 | t661: 1463 ± 226 | t661: 1463 ± 226 |
| Mme de Sagan | 1 | t485: 1592 ± 359 | t485: 1592 ± 359 | t485: 1592 ± 359 | t485: 1592 ± 359 |
| Coquelin | 1 | t198: 1522 ± 289 | t198: 1522 ± 289 | t198: 1522 ± 289 | t198: 1522 ± 289 |
| M. Arthur Meyer | 1 | t911: 1497 ± 264 | t911: 1497 ± 264 | t911: 1497 ± 264 | t911: 1497 ± 264 |
| Sainte-Beuve | 1 | t583: 1479 ± 250 | t583: 1479 ± 250 | t583: 1479 ± 250 | t583: 1479 ± 250 |
| M. de Stermaria | 4 | t275: 1437 ± 215 | t280: 1436 ± 215 | t279: 1436 ± 215 | t275: 1437 ± 215 |
| docteur Percepied | 1 | t58: 1534 ± 313 | t58: 1534 ± 313 | t58: 1534 ± 313 | t58: 1534 ± 313 |
| Dostoïevski | 1 | t836: 1480 ± 264 | t836: 1480 ± 264 | t836: 1480 ± 264 | t836: 1480 ± 264 |
| Mlle de l’Orgeville | 1 | t892: 1573 ± 358 | t892: 1573 ± 358 | t892: 1573 ± 358 | t892: 1573 ± 358 |
| Liszt | 1 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 |
| Mme Ristori | 1 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 | t440: 1490 ± 276 |
| L’excellent écrivain G… | 1 | t448: 1529 ± 318 | t448: 1529 ± 318 | t448: 1529 ± 318 | t448: 1529 ± 318 |
| Octave | 2 | t340: 1496 ± 324 | t875: 1541 ± 332 | t340: 1496 ± 324 | t875: 1541 ± 332 |
| Mme de Vaugoubert | 2 | t686: 1435 ± 242 | t822: 1440 ± 232 | t686: 1435 ± 242 | t822: 1440 ± 232 |
| Mme Trombert | 1 | t231: 1520 ± 314 | t231: 1520 ± 314 | t231: 1520 ± 314 | t231: 1520 ± 314 |
| marquis de Cambremer | 6 | t277: 1424 ± 165 | t761: 1325 ± 118 | t761: 1325 ± 118 | t277: 1424 ± 165 |
| comtesse de Monteriender | 1 | t176: 1518 ± 313 | t176: 1518 ± 313 | t176: 1518 ± 313 | t176: 1518 ± 313 |
| le grand-duc Wladimir | 1 | t689: 1571 ± 367 | t689: 1571 ± 367 | t689: 1571 ± 367 | t689: 1571 ± 367 |
| Mme d'Arpajon | 8 | t597: 1325 ± 137 | t718: 1330 ± 135 | t687: 1323 ± 133 | t718: 1330 ± 135 |
| Madame Elstir | 1 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 |
| les demoiselles d’Ambresac | 1 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 | t341: 1455 ± 267 |
| M. de Miribel | 1 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 |
| le lieutenant-colonel Henry | 1 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 |
| le lieutenant-colonel Picquart | 1 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 | t476: 1499 ± 313 |
| princesse Mathilde | 2 | t238: 1421 ± 268 | t595: 1442 ± 260 | t238: 1421 ± 268 | t595: 1442 ± 260 |
| le roi Théodose | 3 | t208: 1432 ± 256 | t693: 1430 ± 250 | t693: 1430 ± 250 | t208: 1432 ± 256 |
| marquise de Gallardon | 7 | t158: 1331 ± 200 | t711: 1362 ± 188 | t158: 1331 ± 200 | t710: 1362 ± 188 |
| le prince Von | 2 | t625: 1393 ± 222 | t640: 1392 ± 223 | t640: 1392 ± 223 | t625: 1393 ± 222 |
| princesse Sherbatoff | 5 | t742: 1339 ± 174 | t757: 1336 ± 173 | t757: 1336 ± 173 | t742: 1339 ± 174 |
| Duroc | 1 | t395: 1675 ± 519 | t395: 1675 ± 519 | t395: 1675 ± 519 | t395: 1675 ± 519 |
| D'Annunzio | 1 | t693: 1435 ± 290 | t693: 1435 ± 290 | t693: 1435 ± 290 | t693: 1435 ± 290 |
| Mme d'Heudicourt | 5 | t602: 1322 ± 178 | t609: 1322 ± 178 | t602: 1322 ± 178 | t608: 1322 ± 178 |
| M. Grevy | 1 | t94: 1484 ± 348 | t94: 1484 ± 348 | t94: 1484 ± 348 | t94: 1484 ± 348 |
| Mme de Franquetot | 3 | t158: 1416 ± 216 | t923: 1306 ± 170 | t923: 1306 ± 170 | t158: 1416 ± 216 |
| docteur Dieulafoy | 1 | t548: 1667 ± 532 | t548: 1667 ± 532 | t548: 1667 ± 532 | t548: 1667 ± 532 |
| M. d'Herweck | 2 | t698: 1423 ± 289 | t699: 1423 ± 289 | t699: 1423 ± 289 | t698: 1423 ± 289 |
| duc de Châtellerault | 5 | t488: 1385 ± 242 | t683: 1375 ± 248 | t683: 1375 ± 248 | t488: 1385 ± 242 |
| elle | 1 | t430: 1657 ± 536 | t430: 1657 ± 536 | t430: 1657 ± 536 | t430: 1657 ± 536 |
| M. Bontemps | 2 | t229: 1320 ± 256 | t899: 1396 ± 291 | t229: 1320 ± 256 | t899: 1396 ± 291 |
| M. Molé | 1 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 |
| M. de Bouillon | 1 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 |
| Musset | 1 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 |
| Victor Hugo | 1 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 | t295: 1351 ± 248 |
| M. de Courgivaux | 1 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 |
| Mme de Villebon | 1 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 |
| le capitaine | 1 | t375: 1478 ± 402 | t375: 1478 ± 402 | t375: 1478 ± 402 | t375: 1478 ± 402 |
| le bâtonnier | 1 | t284: 1442 ± 380 | t284: 1442 ± 380 | t284: 1442 ± 380 | t284: 1442 ± 380 |
| Vigny | 1 | t294: 1465 ± 404 | t294: 1465 ± 404 | t294: 1465 ± 404 | t294: 1465 ± 404 |
| prince d'Agrigente | 2 | t586: 1454 ± 406 | t922: 1469 ± 414 | t586: 1454 ± 406 | t922: 1469 ± 414 |
| M. de Grouchy | 4 | t587: 1305 ± 253 | t601: 1306 ± 254 | t587: 1305 ± 253 | t601: 1306 ± 254 |
| Beauserfeuil | 1 | t662: 1402 ± 357 | t662: 1402 ± 357 | t662: 1402 ± 357 | t662: 1402 ± 357 |
| Théodose Cadet | 1 | t665: 1402 ± 357 | t665: 1402 ± 357 | t665: 1402 ± 357 | t665: 1402 ± 357 |
| Cartier | 1 | t777: 1391 ± 347 | t777: 1391 ± 347 | t777: 1391 ± 347 | t777: 1391 ± 347 |
| Poullein | 2 | t600: 1468 ± 425 | t601: 1468 ± 425 | t601: 1468 ± 425 | t600: 1468 ± 425 |
| M. Barrère | 1 | t884: 1533 ± 494 | t884: 1533 ± 494 | t884: 1533 ± 494 | t884: 1533 ± 494 |
| professeur E… | 2 | t533: 1364 ± 339 | t685: 1368 ± 341 | t533: 1364 ± 339 | t685: 1368 ± 341 |
| capitaine de Borodino | 5 | t379: 1248 ± 224 | t459: 1246 ± 223 | t459: 1246 ± 223 | t379: 1248 ± 224 |
| Mme de Souvré | 2 | t591: 1260 ± 249 | t687: 1262 ± 246 | t591: 1260 ± 249 | t687: 1262 ± 246 |
| Mme de Morienval | 1 | t367: 1303 ± 293 | t367: 1303 ± 293 | t367: 1303 ± 293 | t367: 1303 ± 293 |
| duchesse de Luxembourg | 1 | t367: 1303 ± 293 | t367: 1303 ± 293 | t367: 1303 ± 293 | t367: 1303 ± 293 |
| Antoine | 1 | t358: 1394 ± 402 | t358: 1394 ± 402 | t358: 1394 ± 402 | t358: 1394 ± 402 |
| les Courvoisier | 1 | t595: 1305 ± 321 | t595: 1305 ± 321 | t595: 1305 ± 321 | t595: 1305 ± 321 |
| Saniette | 8 | t121: 1174 ± 214 | t820: 1163 ± 181 | t820: 1163 ± 181 | t661: 1187 ± 173 |
| la jeune ouvriere | 1 | t96: 1392 ± 420 | t96: 1392 ± 420 | t96: 1392 ± 420 | t96: 1392 ± 420 |
| Maurice | 1 | t908: 1271 ± 305 | t908: 1271 ± 305 | t908: 1271 ± 305 | t908: 1271 ± 305 |
| Prince Henri d'Orléans | 1 | t483: 1379 ± 421 | t483: 1379 ± 421 | t483: 1379 ± 421 | t483: 1379 ± 421 |
| baron de Guermantes | 1 | t452: 1560 ± 610 | t452: 1560 ± 610 | t452: 1560 ± 610 | t452: 1560 ± 610 |
| Monsieur Vallenères | 1 | t472: 1314 ± 385 | t472: 1314 ± 385 | t472: 1314 ± 385 | t472: 1314 ± 385 |
| princesse d'Iéna | 1 | t166: 1306 ± 394 | t166: 1306 ± 394 | t166: 1306 ± 394 | t166: 1306 ± 394 |
| la Charité de Giotto | 1 | t49: 1374 ± 554 | t49: 1374 ± 554 | t49: 1374 ± 554 | t49: 1374 ± 554 |
| comtesse G… | 1 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 |
| vicomtesse de Saint-Fiacre | 1 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 |
| Alix | 3 | t440: 1136 ± 326 | t445: 1136 ± 326 | t440: 1136 ± 326 | t440: 1136 ± 326 |
| ma grand'tante | 1 | t1: 1347 ± 539 | t1: 1347 ± 539 | t1: 1347 ± 539 | t1: 1347 ± 539 |
| prince Foggi | 1 | t883: 1345 ± 538 | t883: 1345 ± 538 | t883: 1345 ± 538 | t883: 1345 ± 538 |
| le professeur E… | 1 | t684: 1308 ± 504 | t684: 1308 ± 504 | t684: 1308 ± 504 | t684: 1308 ± 504 |
| princesse de Nassau | 1 | t931: 1330 ± 530 | t931: 1330 ± 530 | t931: 1330 ± 530 | t931: 1330 ± 530 |
| la marquise | 1 | t528: 1328 ± 529 | t528: 1328 ± 529 | t528: 1328 ± 529 | t528: 1328 ± 529 |
| Madame d'Ambresac | 1 | t366: 1289 ± 492 | t366: 1289 ± 492 | t366: 1289 ± 492 | t366: 1289 ± 492 |
| le curé | 1 | t42: 1293 ± 499 | t42: 1293 ± 499 | t42: 1293 ± 499 | t42: 1293 ± 499 |
| Marie Gineste | 1 | t736: 1304 ± 512 | t736: 1304 ± 512 | t736: 1304 ± 512 | t736: 1304 ± 512 |
| prince de Léon | 1 | t775: 1279 ± 489 | t775: 1279 ± 489 | t775: 1279 ± 489 | t775: 1279 ± 489 |
| le prince von *** | 1 | t498: 1271 ± 483 | t498: 1271 ± 483 | t498: 1271 ± 483 | t498: 1271 ± 483 |
| le grand-duc héritier de Luxembourg | 1 | t581: 1304 ± 520 | t581: 1304 ± 520 | t581: 1304 ± 520 | t581: 1304 ± 520 |
| le diplomate belge | 1 | t493: 1255 ± 482 | t493: 1255 ± 482 | t493: 1255 ± 482 | t493: 1255 ± 482 |
| Dumont | 1 | t30: 1254 ± 481 | t30: 1254 ± 481 | t30: 1254 ± 481 | t30: 1254 ± 481 |
| M. de Luxembourg | 1 | t645: 1244 ± 472 | t645: 1244 ± 472 | t645: 1244 ± 472 | t645: 1244 ± 472 |
| Mme Blatin | 1 | t195: 1246 ± 476 | t195: 1246 ± 476 | t195: 1246 ± 476 | t195: 1246 ± 476 |
| Mme de Simiane | 1 | t269: 1224 ± 454 | t269: 1224 ± 454 | t269: 1224 ± 454 | t269: 1224 ± 454 |
| l'historien de la Fronde | 1 | t453: 1226 ± 457 | t453: 1226 ± 457 | t453: 1226 ± 457 | t453: 1226 ± 457 |
| prince de Faffenheim | 2 | t499: 1217 ± 451 | t500: 1217 ± 451 | t499: 1217 ± 451 | t499: 1217 ± 451 |
| marquise de Citri | 1 | t700: 1230 ± 467 | t700: 1230 ± 467 | t700: 1230 ± 467 | t700: 1230 ± 467 |
| la cousine d'Oriane | 1 | t606: 1197 ± 444 | t606: 1197 ± 444 | t606: 1197 ± 444 | t606: 1197 ± 444 |
| vicomtesse d'Égremont | 1 | t593: 1197 ± 445 | t593: 1197 ± 445 | t593: 1197 ± 445 | t593: 1197 ± 445 |
| l'ambassadrice de Turquie | 1 | t690: 1162 ± 425 | t690: 1162 ± 425 | t690: 1162 ± 425 | t690: 1162 ± 425 |
| Mme Blandais | 2 | t284: 1152 ± 423 | t288: 1152 ± 423 | t288: 1152 ± 423 | t284: 1152 ± 423 |
| M. Pierre | 2 | t438: 1146 ± 421 | t452: 1146 ± 422 | t452: 1146 ± 422 | t438: 1146 ± 421 |
| ma grand’tante | 1 | t2: 1119 ± 403 | t2: 1119 ± 403 | t2: 1119 ± 403 | t2: 1119 ± 403 |
| Mme Iéna | 1 | t635: 1125 ± 410 | t635: 1125 ± 410 | t635: 1125 ± 410 | t635: 1125 ± 410 |
| Mme de Varambon | 2 | t616: 1134 ± 418 | t648: 1134 ± 419 | t648: 1134 ± 419 | t616: 1134 ± 418 |
| le prince de Faffenheim | 1 | t497: 1123 ± 409 | t497: 1123 ± 409 | t497: 1123 ± 409 | t497: 1123 ± 409 |
| l'empereur | 1 | t640: 1125 ± 415 | t640: 1125 ± 415 | t640: 1125 ± 415 | t640: 1125 ± 415 |
| Picquart | 2 | t395: 1096 ± 398 | t482: 1093 ± 398 | t482: 1093 ± 398 | t395: 1096 ± 398 |
| colonel de Froberville | 1 | t696: 992 ± 361 | t696: 992 ± 361 | t696: 992 ± 361 | t696: 992 ± 361 |
| M. de Vigny | 1 | t295: 997 ± 368 | t295: 997 ± 368 | t295: 997 ± 368 | t295: 997 ± 368 |
