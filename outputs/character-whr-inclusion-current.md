# Character Whole-History Rating

- Analysis version: `character_whr_inclusion_v1`
- Lens: `inclusion`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `288`
- Match count: `5756`
- Time point count: `840`
- Node count: `3007`
- Draw rate: `0.327`
- Draw model: `half_win_half_loss`
- w2: `15.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.667`s, filtered `129.642`s (all w2 candidates `589.338`s)
- Convergence: smoothed `28` sweeps (converged: `True`), filtered `840` fits / `13169` sweeps, `0` of them unconverged
- Corpus: `foundation`

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.721337 | 0.259116 | 5756 | filtered WHR at w2=15 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.710687 | 0.255929 | 5756 | filtered WHR at w2=15, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.65856 | 0.233305 | 5756 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.696859 | 0.250866 | 5756 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.723757 | 0.262021 | 5756 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.721636 | 0.259469 | 98.758 |
| 15.0 | 0.721337 | 0.259116 | 129.642 |
| 35.0 | 0.722898 | 0.259491 | 163.547 |
| 60.0 | 0.725267 | 0.260213 | 197.391 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docteur du Boulbon | 1764 ± 188 | 1576.4 | 188.0 | 27 | 19-3-5 | 6 | 6 | -0.469 |
| Françoise | 1627 ± 108 | 1519.1 | 107.9 | 217 | 100-48-69 | 82 | 76 | -0.369 |
| Bergotte | 1641 ± 128 | 1513.1 | 127.5 | 129 | 52-31-46 | 36 | 32 | -0.199 |
| le peintre | 1631 ± 118 | 1512.7 | 118.2 | 42 | 16-4-22 | 8 | 8 | -0.298 |
| Rachel | 1582 ± 92 | 1489.5 | 92.5 | 146 | 52-53-41 | 43 | 43 | -1.09 |
| Elstir | 1582 ± 100 | 1482.1 | 100.3 | 106 | 42-29-35 | 29 | 24 | +0.014 |
| Aimé | 1580 ± 99 | 1481.2 | 99.3 | 79 | 27-13-39 | 18 | 18 | -0.45 |
| M. Verdurin | 1568 ± 104 | 1463.6 | 104.3 | 110 | 38-23-49 | 27 | 27 | -0.64 |
| l'amie de Mlle Vinteuil | 1590 ± 127 | 1463.0 | 127.1 | 44 | 17-6-21 | 12 | 12 | -0.361 |
| Jupien | 1557 ± 96 | 1460.4 | 96.2 | 68 | 23-12-33 | 18 | 18 | -0.063 |
| la grand-mère | 1566 ± 109 | 1456.8 | 109.1 | 225 | 93-65-67 | 80 | 74 | -0.444 |
| le père du narrateur | 1593 ± 136 | 1456.6 | 136.0 | 90 | 34-22-34 | 24 | 24 | -0.79 |
| Rémi | 1622 ± 172 | 1449.7 | 171.8 | 17 | 4-0-13 | 3 | 3 | -0.533 |
| Mlle Vinteuil | 1548 ± 100 | 1447.6 | 100.1 | 71 | 19-15-37 | 15 | 15 | -0.693 |
| Morel | 1531 ± 84 | 1446.4 | 84.2 | 152 | 47-52-53 | 32 | 31 | -1.02 |
| Bloch | 1526 ± 81 | 1445.1 | 80.8 | 270 | 78-111-81 | 71 | 70 | -1.609 |
| Mme Sazerat | 1595 ± 162 | 1433.3 | 162.0 | 20 | 8-2-10 | 6 | 6 | -0.692 |
| la mère du narrateur | 1528 ± 97 | 1430.6 | 97.1 | 144 | 55-36-53 | 40 | 40 | -0.477 |
| Norpois | 1562 ± 134 | 1427.9 | 133.7 | 180 | 79-54-47 | 63 | 62 | -0.659 |
| Robert de Saint-Loup | 1498 ± 73 | 1425.1 | 72.8 | 508 | 167-208-133 | 168 | 154 | -0.623 |
| prince de Guermantes | 1540 ± 116 | 1424.1 | 116.3 | 124 | 42-29-53 | 22 | 22 | -0.797 |
| Odette | 1501 ± 80 | 1421.1 | 79.5 | 462 | 147-154-161 | 142 | 138 | -0.748 |
| Mme de Charlus | 1605 ± 190 | 1415.1 | 189.5 | 15 | 5-1-9 | 2 | 2 | -0.8 |
| Mme Verdurin | 1496 ± 81 | 1414.4 | 81.2 | 311 | 93-104-114 | 82 | 82 | -0.909 |
| marquis de Bréauté | 1522 ± 110 | 1412.2 | 110.0 | 101 | 26-21-54 | 19 | 19 | -0.811 |
| Mme de Surgis | 1544 ± 132 | 1411.7 | 132.3 | 42 | 16-11-15 | 9 | 9 | -0.99 |
| Dreyfus | 1518 ± 111 | 1407.5 | 110.6 | 58 | 13-11-34 | 7 | 7 | -0.77 |
| le grand-père du narrateur | 1574 ± 167 | 1407.2 | 166.7 | 63 | 25-7-31 | 16 | 16 | -0.664 |
| Mme Leroi | 1601 ± 198 | 1402.8 | 198.3 | 13 | 8-4-1 | 5 | 5 | -0.994 |
| Andrée | 1492 ± 90 | 1402.3 | 90.1 | 114 | 35-42-37 | 31 | 31 | -0.815 |
| Mme Goupil | 1569 ± 170 | 1399.4 | 169.9 | 17 | 5-1-11 | 2 | 2 | -0.8 |
| duchesse de Guermantes | 1466 ± 68 | 1397.8 | 68.1 | 662 | 329-177-156 | 199 | 194 | -0.255 |
| docteur Cottard | 1498 ± 102 | 1396.6 | 101.9 | 194 | 48-64-82 | 43 | 43 | -0.899 |
| Mme Bontemps | 1517 ± 126 | 1391.7 | 125.6 | 54 | 15-11-28 | 13 | 13 | -0.575 |
| le narrateur | 1455 ± 65 | 1390.4 | 64.9 | 1093 | 397-501-195 | 316 | 315 | -0.845 |
| comte de Forcheville | 1501 ± 112 | 1389.1 | 112.2 | 112 | 55-18-39 | 25 | 25 | -0.4 |
| Charcot | 1581 ± 197 | 1383.9 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| M. Reinach | 1581 ± 197 | 1383.9 | 197.2 | 12 | 3-2-7 | 1 | 1 | -0.8 |
| tante Léonie | 1544 ± 161 | 1383.5 | 161.0 | 38 | 12-22-4 | 22 | 20 | -0.825 |
| Brichot | 1470 ± 88 | 1382.2 | 88.0 | 135 | 30-32-73 | 21 | 21 | -0.877 |
| marquise de Saint-Euverte | 1499 ± 118 | 1381.0 | 118.3 | 72 | 16-27-29 | 13 | 13 | -1.784 |
| Gilberte | 1451 ± 70 | 1380.7 | 70.4 | 312 | 112-103-97 | 76 | 74 | -0.582 |
| Mme de Marsantes | 1481 ± 102 | 1379.2 | 102.1 | 107 | 19-33-55 | 21 | 20 | -1.234 |
| la marquise douairière de Cambremer | 1506 ± 128 | 1378.2 | 128.3 | 31 | 9-6-16 | 6 | 6 | -0.063 |
| Albertine | 1455 ± 78 | 1377.2 | 77.9 | 387 | 147-156-84 | 146 | 126 | -0.887 |
| baron de Charlus | 1448 ± 71 | 1376.8 | 71.3 | 485 | 185-155-145 | 119 | 118 | -0.8 |
| M. Vinteuil | 1498 ± 122 | 1376.2 | 122.1 | 61 | 18-19-24 | 15 | 15 | -0.444 |
| Mme de Villeparisis | 1487 ± 111 | 1375.5 | 111.2 | 236 | 89-94-53 | 79 | 78 | -0.749 |
| Mme de Sévigné | 1538 ± 162 | 1375.3 | 162.3 | 25 | 7-5-13 | 4 | 4 | -0.065 |
| Mme Cottard | 1563 ± 188 | 1375.2 | 187.5 | 33 | 16-8-9 | 11 | 11 | -0.43 |
| Swann | 1444 ± 72 | 1372.3 | 71.8 | 667 | 207-308-152 | 202 | 198 | -1.023 |
| M. de Chevregny | 1543 ± 173 | 1370.3 | 173.1 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| M. de Crécy | 1543 ± 173 | 1370.3 | 173.1 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| Mme Féré | 1543 ± 173 | 1370.3 | 173.1 | 16 | 4-1-11 | 1 | 1 | -0.4 |
| général de Froberville | 1526 ± 156 | 1370.2 | 156.1 | 27 | 7-4-16 | 7 | 7 | -0.596 |
| princesse de Luxembourg | 1521 ± 152 | 1368.8 | 152.0 | 25 | 7-6-12 | 6 | 6 | -0.782 |
| M. Ski | 1524 ± 156 | 1367.8 | 156.2 | 21 | 4-1-16 | 2 | 2 | -0.4 |
| M. Nissim Bernard | 1501 ± 134 | 1367.6 | 133.8 | 39 | 9-10-20 | 10 | 7 | -1.502 |
| le directeur | 1501 ± 135 | 1366.6 | 134.8 | 39 | 11-14-14 | 11 | 11 | -0.851 |
| Bloch père | 1495 ± 131 | 1364.0 | 131.3 | 47 | 11-11-25 | 8 | 8 | -1.614 |
| le jeune marquis de Cambremer | 1558 ± 196 | 1361.6 | 196.3 | 12 | 2-0-10 | 1 | 1 | -1.2 |
| Legrandin | 1464 ± 104 | 1359.4 | 104.2 | 83 | 15-28-40 | 24 | 20 | -1.22 |
| prince des Laumes | 1505 ± 147 | 1357.6 | 147.1 | 27 | 4-3-20 | 3 | 3 | -0.8 |
| Esther | 1533 ± 181 | 1351.4 | 181.2 | 14 | 3-2-9 | 2 | 2 | -1.0 |
| duc de Chartres | 1529 ± 187 | 1341.6 | 187.1 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| prince de Chimay | 1529 ± 187 | 1341.6 | 187.1 | 14 | 2-0-12 | 1 | 1 | -0.8 |
| princesse de Guermantes | 1453 ± 112 | 1340.4 | 112.4 | 113 | 41-32-40 | 25 | 25 | -0.403 |
| comtesse Molé | 1465 ± 136 | 1329.2 | 136.0 | 34 | 6-9-19 | 6 | 6 | -1.288 |
| prince d’Agrigente | 1512 ± 183 | 1328.9 | 183.0 | 15 | 3-2-10 | 2 | 2 | -0.8 |
| princesse de Parme | 1423 ± 97 | 1326.3 | 96.7 | 130 | 35-65-30 | 38 | 38 | -0.839 |
| Céline | 1509 ± 186 | 1323.9 | 185.5 | 16 | 4-6-6 | 2 | 2 | -1.14 |
| général de Monserfeuil | 1488 ± 166 | 1322.1 | 165.9 | 18 | 6-7-5 | 4 | 4 | -1.481 |
| prince de Foix | 1514 ± 196 | 1317.6 | 196.1 | 14 | 4-4-6 | 3 | 3 | -0.893 |
| duc de Guermantes | 1401 ± 86 | 1315.2 | 86.1 | 401 | 120-171-110 | 110 | 107 | -1.042 |
| oncle Adolphe | 1505 ± 192 | 1312.8 | 192.5 | 20 | 4-7-9 | 6 | 5 | -1.773 |
| M. d'Argencourt | 1460 ± 158 | 1301.3 | 158.4 | 56 | 19-18-19 | 14 | 12 | -1.123 |
| Rosemonde | 1467 ± 170 | 1296.8 | 170.2 | 20 | 5-7-8 | 4 | 4 | -0.7 |
| Goncourt | 1468 ± 171 | 1296.4 | 171.2 | 16 | 2-3-11 | 2 | 2 | -0.8 |
| M. de Vaugoubert | 1430 ± 137 | 1293.0 | 137.3 | 35 | 6-12-17 | 9 | 8 | -1.383 |
| Mme de Cambremer | 1384 ± 102 | 1282.7 | 101.7 | 112 | 12-54-46 | 20 | 19 | -1.51 |
| le petit Cambremer | 1454 ± 187 | 1267.0 | 186.8 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| princesse de Silistrie | 1454 ± 187 | 1267.0 | 186.8 | 14 | 1-3-10 | 1 | 1 | -0.8 |
| la Berma | 1396 ± 140 | 1256.0 | 140.1 | 62 | 19-24-19 | 19 | 16 | -0.336 |
| Mme d'Arpajon | 1375 ± 129 | 1245.7 | 129.3 | 37 | 7-20-10 | 8 | 8 | -1.53 |
| Balzac | 1419 ± 184 | 1234.6 | 184.0 | 18 | 2-4-12 | 2 | 2 | -0.8 |
| marquis de Cambremer | 1330 ± 117 | 1212.2 | 117.4 | 45 | 7-23-15 | 6 | 6 | -1.016 |
| marquise de Gallardon | 1373 ± 184 | 1188.6 | 184.2 | 19 | 2-10-7 | 7 | 7 | -1.717 |
| princesse Sherbatoff | 1328 ± 173 | 1155.3 | 173.2 | 19 | 5-13-1 | 5 | 5 | -0.787 |
| Mme d'Heudicourt | 1317 ± 178 | 1138.8 | 177.8 | 18 | 3-11-4 | 5 | 5 | -1.482 |
| Mme de Franquetot | 1302 ± 170 | 1132.4 | 170.0 | 23 | 4-13-6 | 3 | 3 | -1.092 |
| Saniette | 1159 ± 181 | 977.7 | 181.2 | 35 | 1-27-7 | 9 | 8 | -3.263 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1995 ± 364 | 364.1 | 14 | 1 | 1 | 888 |
| marquis de Beausergent | 1964 ± 373 | 373.3 | 12 | 1 | 1 | 923 |
| la reine de Naples | 1954 ± 310 | 309.9 | 17 | 3 | 3 | 828 |
| Mme Elstir | 1933 ± 388 | 387.5 | 7 | 1 | 1 | 333 |
| Céleste Albaret | 1929 ± 276 | 276.3 | 17 | 3 | 3 | 806 |
| prince de Saxe | 1858 ± 428 | 428.1 | 3 | 1 | 1 | 365 |
| Mlle de Saint-Loup | 1839 ± 335 | 335.2 | 7 | 2 | 2 | 940 |
| Marie | 1836 ± 318 | 317.5 | 7 | 1 | 1 | 737 |
| colonel Picquart | 1830 ± 429 | 429.3 | 4 | 1 | 1 | 481 |
| Mme de Chaussepierre | 1824 ± 431 | 431.0 | 4 | 1 | 1 | 777 |
| Mme de Grouchy | 1819 ± 436 | 436.5 | 4 | 1 | 1 | 598 |
| duchesse de La Trémoïlle | 1804 ± 443 | 443.3 | 3 | 1 | 1 | 119 |
| marquis Maurice de Vaudémont | 1797 ± 459 | 459.0 | 2 | 1 | 1 | 353 |
| Eulalie | 1792 ± 254 | 254.3 | 16 | 7 | 7 | 796 |
| Maeterlinck | 1784 ± 355 | 355.4 | 5 | 1 | 1 | 469 |
| Bibi | 1765 ± 470 | 469.6 | 2 | 1 | 1 | 579 |
| Mlle Bloch | 1761 ± 469 | 469.3 | 2 | 1 | 1 | 732 |
| Victurnien | 1741 ± 272 | 271.7 | 8 | 2 | 2 | 704 |
| le commandant Duroc | 1741 ± 478 | 477.5 | 2 | 1 | 1 | 396 |
| monsieur Vallenères | 1739 ± 479 | 479.0 | 2 | 1 | 1 | 457 |
| Lady Israels | 1738 ± 479 | 479.0 | 2 | 1 | 1 | 232 |
| duc de Sidonia | 1736 ± 487 | 486.8 | 2 | 1 | 1 | 684 |
| Gribelin | 1719 ± 319 | 318.7 | 6 | 1 | 1 | 482 |
| Léa | 1712 ± 216 | 215.9 | 14 | 4 | 4 | 852 |
| Émilie Daltier | 1700 ± 389 | 389.1 | 3 | 1 | 1 | 839 |
| Bismarck | 1683 ± 332 | 332.5 | 4 | 1 | 1 | 210 |
| Herbinger | 1682 ± 386 | 386.3 | 3 | 1 | 1 | 108 |
| Duroc | 1677 ± 519 | 518.6 | 2 | 1 | 1 | 395 |
| Létourville | 1675 ± 387 | 387.4 | 3 | 1 | 1 | 921 |
| docteur Dieulafoy | 1663 ± 533 | 533.3 | 1 | 1 | 1 | 548 |
| Théodore | 1663 ± 416 | 416.1 | 2 | 1 | 1 | 59 |
| elle | 1659 ± 535 | 535.4 | 1 | 1 | 1 | 430 |
| marquis du Lau | 1650 ± 328 | 328.4 | 5 | 2 | 2 | 869 |
| la duchesse d'Alençon | 1649 ± 300 | 299.5 | 6 | 1 | 1 | 628 |
| Dechambre | 1647 ± 404 | 404.3 | 3 | 1 | 1 | 745 |
| Mlle de Stermaria | 1642 ± 223 | 223.1 | 10 | 5 | 5 | 577 |
| grand-duc héritier de Luxembourg | 1638 ± 231 | 231.4 | 9 | 2 | 2 | 644 |
| les La Trémoïlle | 1635 ± 257 | 257.1 | 7 | 1 | 1 | 118 |
| Arnulphe | 1634 ± 324 | 324.2 | 4 | 1 | 1 | 703 |
| duc d'Aumale | 1632 ± 352 | 351.6 | 4 | 2 | 2 | 664 |
| M. de Courgivaux | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| Mme de Villebon | 1630 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| Flora | 1627 ± 240 | 239.6 | 8 | 1 | 1 | 4 |
| Marie-Aynard | 1619 ± 257 | 256.9 | 7 | 1 | 1 | 480 |
| Victurnienne | 1619 ± 257 | 256.9 | 7 | 1 | 1 | 480 |
| M. d'Orsan | 1618 ± 206 | 206.0 | 11 | 1 | 1 | 177 |
| cousine Poictiers | 1616 ± 293 | 292.9 | 5 | 1 | 1 | 414 |
| duc de Poictiers | 1616 ± 293 | 292.9 | 5 | 1 | 1 | 414 |
| M. Vibert | 1606 ± 358 | 357.7 | 3 | 1 | 1 | 618 |
| Mme de Stermaria | 1595 ± 289 | 288.8 | 5 | 1 | 1 | 566 |
| Mme de Sagan | 1589 ± 359 | 359.2 | 3 | 1 | 1 | 485 |
| le pianiste | 1587 ± 218 | 217.6 | 10 | 3 | 3 | 124 |
| Mme Legrandin mère | 1586 ± 243 | 242.6 | 8 | 1 | 1 | 266 |
| Victoire | 1586 ± 243 | 242.6 | 8 | 1 | 1 | 266 |
| Sarah Bernhardt | 1585 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| le jeune prince de Foix | 1585 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| vicomte de Courvoisier | 1585 ± 260 | 260.1 | 7 | 1 | 1 | 908 |
| le baron Bréau-Chenut | 1584 ± 259 | 258.9 | 7 | 1 | 1 | 229 |
| le vieux père Chenut | 1584 ± 259 | 258.9 | 7 | 1 | 1 | 229 |
| d'Orléans | 1582 ± 289 | 288.6 | 5 | 1 | 1 | 325 |
| le grand-duc Wladimir | 1579 ± 363 | 363.2 | 3 | 1 | 1 | 689 |
| Manet | 1574 ± 290 | 289.6 | 5 | 1 | 1 | 637 |
| M. de Beauserfeuil | 1570 ± 250 | 250.1 | 7 | 1 | 1 | 644 |
| jeune blonde de Rivebelle | 1570 ± 267 | 266.8 | 6 | 2 | 2 | 326 |
| M. de Goncourt | 1570 ± 240 | 239.9 | 8 | 1 | 1 | 897 |
| Mlle de l’Orgeville | 1567 ± 358 | 358.5 | 3 | 1 | 1 | 892 |
| Lady Israël | 1566 ± 293 | 293.2 | 5 | 1 | 1 | 491 |
| M. de Bornier | 1564 ± 309 | 308.9 | 5 | 1 | 1 | 609 |
| baron de Guermantes | 1559 ± 611 | 611.4 | 1 | 1 | 1 | 452 |
| duchesse de Létourville | 1557 ± 290 | 289.5 | 5 | 1 | 1 | 912 |
| Élisabeth | 1556 ± 267 | 267.0 | 6 | 1 | 1 | 791 |
| Sir Rufus Israël | 1550 ± 259 | 259.4 | 7 | 1 | 1 | 459 |
| M. de La Rochefoucauld | 1550 ± 268 | 268.3 | 6 | 1 | 1 | 297 |
| duchesse de La Rochefoucauld | 1550 ± 268 | 268.3 | 6 | 1 | 1 | 297 |
| duchesse de Praslin | 1550 ± 268 | 268.3 | 6 | 1 | 1 | 297 |
| M. de Marsantes | 1545 ± 253 | 253.0 | 7 | 2 | 2 | 509 |
| le marquis de Ganançay | 1545 ± 285 | 284.7 | 6 | 1 | 1 | 367 |
| le marquis de Palancy | 1545 ± 285 | 284.7 | 6 | 1 | 1 | 367 |
| Mme de Montmorency | 1540 ± 205 | 205.0 | 11 | 1 | 1 | 718 |
| Mme de Rochechouart | 1540 ± 205 | 205.0 | 11 | 1 | 1 | 718 |
| prince de Sagan | 1539 ± 250 | 250.3 | 7 | 1 | 1 | 710 |
| Octave | 1535 ± 332 | 331.9 | 4 | 2 | 2 | 875 |
| M. Barrère | 1530 ± 494 | 494.1 | 1 | 1 | 1 | 884 |
| Mlle d'Éporcheville | 1525 ± 212 | 212.4 | 10 | 2 | 2 | 865 |
| docteur Percepied | 1524 ± 313 | 312.9 | 4 | 1 | 1 | 58 |
| L’excellent écrivain G… | 1523 ± 317 | 317.3 | 4 | 1 | 1 | 448 |
| Lady Rufus Israël | 1522 ± 266 | 266.4 | 6 | 1 | 1 | 868 |
| Coquelin | 1515 ± 288 | 288.4 | 5 | 1 | 1 | 198 |
| Mme Trombert | 1515 ± 314 | 313.6 | 4 | 1 | 1 | 231 |
| princesse de Nassau | 1513 ± 493 | 493.3 | 1 | 1 | 1 | 931 |
| d’Orgeville | 1511 ± 247 | 246.6 | 7 | 1 | 1 | 701 |
| Mme Putbus | 1511 ± 233 | 233.1 | 8 | 1 | 1 | 792 |
| comtesse de Monteriender | 1510 ± 313 | 312.8 | 4 | 1 | 1 | 176 |
| M. Swann, le père | 1505 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| le comte de Paris | 1505 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| le prince de Galles | 1505 ± 261 | 261.2 | 7 | 1 | 1 | 2 |
| prince Von | 1505 ± 235 | 235.3 | 8 | 3 | 3 | 641 |
| Napoléon III | 1504 ± 238 | 238.1 | 8 | 1 | 1 | 186 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| M. Carnot | 1499 ± 221 | 220.7 | 9 | 1 | 1 | 663 |
| Mme Carnot | 1499 ± 221 | 220.7 | 9 | 1 | 1 | 663 |
| Mme Timoléon d'Amoncourt | 1498 ± 225 | 224.7 | 9 | 1 | 1 | 694 |
| Prince Henri d'Orléans | 1497 ± 407 | 406.6 | 2 | 1 | 1 | 483 |
| M. de Miribel | 1496 ± 313 | 313.4 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Henry | 1496 ± 313 | 313.4 | 4 | 1 | 1 | 476 |
| le lieutenant-colonel Picquart | 1496 ± 313 | 313.4 | 4 | 1 | 1 | 476 |
| M. Arthur Meyer | 1495 ± 264 | 264.3 | 6 | 1 | 1 | 911 |
| Liszt | 1493 ± 274 | 273.5 | 6 | 1 | 1 | 440 |
| Mme Ristori | 1493 ± 274 | 273.5 | 6 | 1 | 1 | 440 |
| Thibaud | 1491 ± 232 | 232.3 | 8 | 1 | 1 | 780 |
| Léonor de Cambremer | 1490 ± 202 | 201.9 | 12 | 1 | 1 | 923 |
| comte de Paris | 1489 ± 214 | 213.6 | 10 | 3 | 3 | 219 |
| Poullein | 1480 ± 417 | 417.4 | 2 | 2 | 2 | 601 |
| le capitaine | 1479 ± 402 | 402.3 | 2 | 1 | 1 | 375 |
| M. Grevy | 1478 ± 348 | 348.4 | 3 | 1 | 1 | 94 |
| Dostoïevski | 1476 ± 264 | 263.5 | 6 | 1 | 1 | 836 |
| Sainte-Beuve | 1474 ± 250 | 250.3 | 7 | 1 | 1 | 583 |
| princesse d'Épinay | 1474 ± 204 | 204.3 | 12 | 3 | 3 | 608 |
| l'abbé Poiré | 1469 ± 211 | 211.4 | 10 | 1 | 1 | 708 |
| le roi Théodose | 1463 ± 246 | 245.7 | 8 | 3 | 3 | 693 |
| prince d'Agrigente | 1463 ± 414 | 414.2 | 2 | 2 | 2 | 922 |
| Barrès | 1461 ± 226 | 225.9 | 9 | 1 | 1 | 661 |
| Clémenceau | 1461 ± 226 | 225.9 | 9 | 1 | 1 | 661 |
| Vigny | 1460 ± 404 | 403.7 | 2 | 1 | 1 | 294 |
| comtesse douairière d'Argencourt | 1456 ± 212 | 212.1 | 10 | 1 | 1 | 590 |
| duchesse de Gallardon douairière | 1456 ± 212 | 212.1 | 10 | 1 | 1 | 590 |
| marquis de Fierbois | 1456 ± 212 | 212.1 | 10 | 1 | 1 | 590 |
| Madame Elstir | 1453 ± 267 | 267.2 | 6 | 1 | 1 | 341 |
| les demoiselles d’Ambresac | 1453 ± 267 | 267.2 | 6 | 1 | 1 | 341 |
| Gisèle | 1451 ± 200 | 200.1 | 14 | 5 | 5 | 812 |
| duc de Châtellerault | 1448 ± 233 | 233.0 | 10 | 5 | 5 | 683 |
| M. de Chateaubriand | 1444 ± 209 | 209.3 | 11 | 2 | 2 | 870 |
| D'Annunzio | 1439 ± 289 | 289.2 | 5 | 1 | 1 | 693 |
| Mme de Vaugoubert | 1437 ± 233 | 232.8 | 9 | 2 | 2 | 822 |
| le bâtonnier | 1436 ± 379 | 379.4 | 3 | 1 | 1 | 284 |
| princesse Mathilde | 1434 ± 260 | 260.2 | 7 | 2 | 2 | 595 |
| M. de Stermaria | 1434 ± 215 | 215.0 | 10 | 4 | 4 | 280 |
| M. d'Herweck | 1421 ± 288 | 288.3 | 5 | 2 | 2 | 699 |
| princesse d'Iéna | 1404 ± 370 | 369.7 | 3 | 1 | 1 | 166 |
| Beauserfeuil | 1398 ± 357 | 356.6 | 3 | 1 | 1 | 662 |
| Théodose Cadet | 1397 ± 356 | 356.5 | 3 | 1 | 1 | 665 |
| la jeune ouvriere | 1389 ± 420 | 419.8 | 2 | 1 | 1 | 96 |
| Cartier | 1388 ± 347 | 347.3 | 4 | 1 | 1 | 777 |
| Antoine | 1387 ± 400 | 399.9 | 3 | 1 | 1 | 358 |
| le prince Von | 1384 ± 223 | 223.2 | 10 | 2 | 2 | 640 |
| M. de Luxembourg | 1382 ± 419 | 419.3 | 2 | 1 | 1 | 645 |
| comtesse G… | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 589 |
| vicomtesse de Saint-Fiacre | 1370 ± 552 | 551.9 | 1 | 1 | 1 | 924 |
| la Charité de Giotto | 1369 ± 551 | 551.1 | 1 | 1 | 1 | 49 |
| professeur E… | 1366 ± 341 | 340.8 | 4 | 2 | 2 | 685 |
| marquise de Citri | 1365 ± 417 | 417.4 | 2 | 1 | 1 | 700 |
| M. de Grouchy | 1352 ± 240 | 240.2 | 10 | 4 | 4 | 601 |
| M. Molé | 1346 ± 248 | 247.7 | 8 | 1 | 1 | 295 |
| M. de Bouillon | 1346 ± 248 | 247.7 | 8 | 1 | 1 | 295 |
| Musset | 1346 ± 248 | 247.7 | 8 | 1 | 1 | 295 |
| Victor Hugo | 1346 ± 248 | 247.7 | 8 | 1 | 1 | 295 |
| ma grand'tante | 1343 ± 536 | 536.4 | 1 | 1 | 1 | 1 |
| prince Foggi | 1343 ± 536 | 536.4 | 1 | 1 | 1 | 883 |
| la marquise | 1325 ± 528 | 528.0 | 1 | 1 | 1 | 528 |
| Monsieur Vallenères | 1310 ± 385 | 384.6 | 3 | 1 | 1 | 472 |
| le professeur E… | 1307 ± 504 | 503.8 | 2 | 1 | 1 | 684 |
| M. Bontemps | 1305 ± 307 | 307.0 | 9 | 2 | 2 | 899 |
| Mme de Morienval | 1302 ± 293 | 292.8 | 6 | 1 | 1 | 367 |
| duchesse de Luxembourg | 1302 ± 293 | 292.8 | 6 | 1 | 1 | 367 |
| Marie Gineste | 1301 ± 511 | 511.2 | 2 | 1 | 1 | 736 |
| le grand-duc héritier de Luxembourg | 1301 ± 519 | 518.6 | 1 | 1 | 1 | 581 |
| les Courvoisier | 1298 ± 321 | 320.8 | 5 | 1 | 1 | 595 |
| le curé | 1294 ± 501 | 500.7 | 2 | 1 | 1 | 42 |
| Madame d'Ambresac | 1286 ± 490 | 490.1 | 2 | 1 | 1 | 366 |
| prince de Léon | 1276 ± 487 | 487.3 | 2 | 1 | 1 | 775 |
| Maurice | 1272 ± 305 | 304.8 | 7 | 1 | 1 | 908 |
| le prince von *** | 1269 ± 482 | 481.8 | 2 | 1 | 1 | 498 |
| Mme de Souvré | 1263 ± 245 | 244.9 | 11 | 2 | 2 | 687 |
| le diplomate belge | 1254 ± 481 | 481.2 | 2 | 1 | 1 | 493 |
| Dumont | 1252 ± 480 | 479.6 | 2 | 1 | 1 | 30 |
| capitaine de Borodino | 1244 ± 223 | 222.6 | 14 | 5 | 5 | 459 |
| Mme Blatin | 1244 ± 475 | 474.8 | 2 | 1 | 1 | 195 |
| l'historien de la Fronde | 1223 ± 455 | 455.1 | 3 | 1 | 1 | 453 |
| Mme de Simiane | 1223 ± 453 | 453.1 | 3 | 1 | 1 | 269 |
| prince de Faffenheim | 1214 ± 450 | 450.2 | 3 | 2 | 2 | 500 |
| Alix | 1202 ± 291 | 291.0 | 9 | 3 | 3 | 445 |
| la cousine d'Oriane | 1193 ± 443 | 442.7 | 3 | 1 | 1 | 606 |
| vicomtesse d'Égremont | 1192 ± 443 | 443.1 | 3 | 1 | 1 | 593 |
| l'ambassadrice de Turquie | 1158 ± 424 | 424.0 | 4 | 1 | 1 | 690 |
| Mme Blandais | 1147 ± 421 | 421.2 | 4 | 2 | 2 | 288 |
| M. Pierre | 1143 ± 421 | 420.7 | 4 | 2 | 2 | 452 |
| Mme de Varambon | 1127 ± 417 | 417.0 | 4 | 2 | 2 | 648 |
| Mme Iéna | 1121 ± 409 | 409.0 | 5 | 1 | 1 | 635 |
| l'empereur | 1120 ± 413 | 413.1 | 4 | 1 | 1 | 640 |
| le prince de Faffenheim | 1119 ± 408 | 407.8 | 5 | 1 | 1 | 497 |
| ma grand’tante | 1113 ± 401 | 401.2 | 7 | 1 | 1 | 2 |
| Picquart | 1090 ± 397 | 397.3 | 8 | 2 | 2 | 482 |
| M. de Vigny | 994 ± 368 | 367.7 | 8 | 1 | 1 | 295 |
| colonel de Froberville | 991 ± 361 | 360.6 | 14 | 1 | 1 | 696 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Céleste Albaret | 3 | t736: 1931 ± 274 | t806: 1929 ± 276 | t806: 1929 ± 276 | t737: 1931 ± 274 |
| la reine de Naples | 3 | t628: 1952 ± 309 | t828: 1954 ± 310 | t628: 1952 ± 309 | t822: 1955 ± 310 |
| Mlle d'Oloron | 1 | t888: 1995 ± 364 | t888: 1995 ± 364 | t888: 1995 ± 364 | t888: 1995 ± 364 |
| marquis de Beausergent | 1 | t923: 1964 ± 373 | t923: 1964 ± 373 | t923: 1964 ± 373 | t923: 1964 ± 373 |
| docteur du Boulbon | 6 | t248: 1742 ± 178 | t725: 1764 ± 188 | t248: 1742 ± 178 | t523: 1768 ± 165 |
| Mme Elstir | 1 | t333: 1933 ± 388 | t333: 1933 ± 388 | t333: 1933 ± 388 | t333: 1933 ± 388 |
| Eulalie | 7 | t19: 1766 ± 215 | t796: 1792 ± 254 | t42: 1766 ± 213 | t796: 1792 ± 254 |
| Françoise | 76 | t2: 1619 ± 95 | t940: 1627 ± 108 | t535: 1591 ± 84 | t833: 1629 ± 79 |
| Marie | 1 | t737: 1836 ± 318 | t737: 1836 ± 318 | t737: 1836 ± 318 | t737: 1836 ± 318 |
| Bergotte | 32 | t28: 1526 ± 116 | t941: 1641 ± 128 | t28: 1526 ± 116 | t941: 1641 ± 128 |
| le peintre | 8 | t89: 1640 ± 121 | t186: 1631 ± 118 | t186: 1631 ± 118 | t114: 1641 ± 117 |
| Mlle de Saint-Loup | 2 | t939: 1839 ± 335 | t940: 1839 ± 335 | t939: 1839 ± 335 | t939: 1839 ± 335 |
| Léa | 4 | t807: 1709 ± 212 | t852: 1712 ± 216 | t807: 1709 ± 212 | t852: 1712 ± 216 |
| Rachel | 43 | t251: 1440 ± 125 | t939: 1582 ± 92 | t469: 1421 ± 81 | t938: 1582 ± 92 |
| Elstir | 24 | t269: 1540 ± 112 | t904: 1582 ± 100 | t617: 1535 ± 98 | t898: 1582 ± 99 |
| Aimé | 18 | t279: 1508 ± 130 | t890: 1580 ± 99 | t282: 1508 ± 130 | t791: 1582 ± 92 |
| Victurnien | 2 | t703: 1741 ± 272 | t704: 1741 ± 272 | t703: 1741 ± 272 | t704: 1741 ± 272 |
| M. Verdurin | 27 | t70: 1531 ± 98 | t904: 1568 ± 104 | t72: 1531 ± 97 | t904: 1568 ± 104 |
| l'amie de Mlle Vinteuil | 12 | t58: 1598 ± 146 | t855: 1590 ± 127 | t823: 1589 ± 123 | t63: 1598 ± 146 |
| Jupien | 18 | t356: 1609 ± 145 | t913: 1557 ± 96 | t888: 1556 ± 94 | t356: 1609 ± 145 |
| la grand-mère | 74 | t1: 1560 ± 96 | t917: 1566 ± 109 | t412: 1542 ± 78 | t721: 1581 ± 95 |
| le père du narrateur | 24 | t4: 1573 ± 98 | t550: 1593 ± 136 | t47: 1570 ± 93 | t547: 1593 ± 136 |
| Rémi | 3 | t101: 1621 ± 175 | t177: 1622 ± 172 | t101: 1621 ± 175 | t177: 1622 ± 172 |
| Mlle Vinteuil | 15 | t45: 1522 ± 135 | t855: 1548 ± 100 | t61: 1520 ± 133 | t762: 1550 ± 101 |
| Morel | 31 | t501: 1439 ± 127 | t928: 1531 ± 84 | t501: 1439 ± 127 | t928: 1531 ± 84 |
| Bloch | 70 | t29: 1362 ± 125 | t940: 1526 ± 81 | t31: 1361 ± 125 | t930: 1526 ± 77 |
| Mme Sazerat | 6 | t416: 1623 ± 206 | t882: 1595 ± 162 | t870: 1595 ± 161 | t416: 1623 ± 206 |
| la mère du narrateur | 40 | t4: 1603 ± 108 | t888: 1528 ± 97 | t888: 1528 ± 97 | t4: 1603 ± 108 |
| prince de Saxe | 1 | t365: 1858 ± 428 | t365: 1858 ± 428 | t365: 1858 ± 428 | t365: 1858 ± 428 |
| Maeterlinck | 1 | t469: 1784 ± 355 | t469: 1784 ± 355 | t469: 1784 ± 355 | t469: 1784 ± 355 |
| Norpois | 62 | t201: 1566 ± 76 | t915: 1562 ± 134 | t350: 1549 ± 80 | t201: 1566 ± 76 |
| Robert de Saint-Loup | 154 | t298: 1498 ± 71 | t939: 1498 ± 73 | t477: 1437 ± 54 | t911: 1500 ± 64 |
| prince de Guermantes | 22 | t477: 1504 ± 104 | t927: 1540 ± 116 | t477: 1504 ± 104 | t708: 1542 ± 73 |
| Odette | 138 | t21: 1567 ± 80 | t938: 1501 ± 80 | t501: 1430 ± 80 | t21: 1567 ± 80 |
| Mlle de Stermaria | 5 | t280: 1590 ± 232 | t577: 1642 ± 223 | t280: 1590 ± 232 | t576: 1642 ± 223 |
| Mme de Charlus | 2 | t621: 1603 ± 190 | t855: 1605 ± 190 | t621: 1603 ± 190 | t855: 1605 ± 190 |
| Mme Verdurin | 82 | t70: 1474 ± 74 | t934: 1496 ± 81 | t86: 1474 ± 68 | t761: 1502 ± 62 |
| marquis de Bréauté | 19 | t157: 1514 ± 130 | t938: 1522 ± 110 | t450: 1509 ± 111 | t623: 1525 ± 86 |
| Mme de Surgis | 9 | t687: 1550 ± 112 | t817: 1544 ± 132 | t817: 1544 ± 132 | t687: 1550 ± 112 |
| M. d'Orsan | 1 | t177: 1618 ± 206 | t177: 1618 ± 206 | t177: 1618 ± 206 | t177: 1618 ± 206 |
| Dreyfus | 7 | t324: 1528 ± 130 | t708: 1518 ± 111 | t708: 1518 ± 111 | t421: 1528 ± 114 |
| le grand-père du narrateur | 16 | t2: 1632 ± 101 | t549: 1574 ± 167 | t547: 1574 ± 166 | t31: 1635 ± 98 |
| grand-duc héritier de Luxembourg | 2 | t540: 1644 ± 236 | t644: 1638 ± 231 | t644: 1638 ± 231 | t540: 1644 ± 236 |
| Mme Leroi | 5 | t436: 1601 ± 198 | t506: 1601 ± 198 | t439: 1600 ± 198 | t506: 1601 ± 198 |
| Andrée | 31 | t341: 1498 ± 105 | t875: 1492 ± 90 | t782: 1480 ± 76 | t345: 1499 ± 104 |
| Gribelin | 1 | t482: 1719 ± 319 | t482: 1719 ± 319 | t482: 1719 ± 319 | t482: 1719 ± 319 |
| colonel Picquart | 1 | t481: 1830 ± 429 | t481: 1830 ± 429 | t481: 1830 ± 429 | t481: 1830 ± 429 |
| Mme Goupil | 2 | t870: 1569 ± 170 | t871: 1569 ± 170 | t870: 1569 ± 170 | t871: 1569 ± 170 |
| duchesse de Guermantes | 194 | t67: 1617 ± 125 | t939: 1466 ± 68 | t938: 1466 ± 68 | t412: 1669 ± 67 |
| docteur Cottard | 43 | t71: 1448 ± 86 | t923: 1498 ± 102 | t77: 1448 ± 84 | t897: 1499 ± 98 |
| Mme de Chaussepierre | 1 | t777: 1824 ± 431 | t777: 1824 ± 431 | t777: 1824 ± 431 | t777: 1824 ± 431 |
| Mme Bontemps | 13 | t229: 1520 ± 124 | t899: 1517 ± 126 | t661: 1510 ± 119 | t349: 1522 ± 124 |
| le narrateur | 315 | t4: 1437 ± 85 | t941: 1455 ± 65 | t806: 1434 ± 46 | t623: 1563 ± 51 |
| comte de Forcheville | 25 | t110: 1686 ± 89 | t938: 1501 ± 112 | t928: 1500 ± 111 | t124: 1687 ± 86 |
| Flora | 1 | t4: 1627 ± 240 | t4: 1627 ± 240 | t4: 1627 ± 240 | t4: 1627 ± 240 |
| Charcot | 1 | t523: 1581 ± 197 | t523: 1581 ± 197 | t523: 1581 ± 197 | t523: 1581 ± 197 |
| M. Reinach | 1 | t523: 1581 ± 197 | t523: 1581 ± 197 | t523: 1581 ± 197 | t523: 1581 ± 197 |
| tante Léonie | 20 | t8: 1507 ± 126 | t361: 1544 ± 161 | t56: 1503 ± 123 | t361: 1544 ± 161 |
| Mme de Grouchy | 1 | t598: 1819 ± 436 | t598: 1819 ± 436 | t598: 1819 ± 436 | t598: 1819 ± 436 |
| Brichot | 21 | t111: 1511 ± 126 | t923: 1470 ± 88 | t825: 1466 ± 74 | t118: 1512 ± 125 |
| marquise de Saint-Euverte | 13 | t163: 1344 ± 168 | t938: 1499 ± 118 | t163: 1344 ± 168 | t938: 1499 ± 118 |
| Gilberte | 74 | t37: 1594 ± 108 | t939: 1451 ± 70 | t939: 1451 ± 70 | t37: 1594 ± 108 |
| Mme de Marsantes | 20 | t232: 1432 ± 141 | t890: 1481 ± 102 | t421: 1423 ± 101 | t890: 1481 ± 102 |
| les La Trémoïlle | 1 | t118: 1635 ± 257 | t118: 1635 ± 257 | t118: 1635 ± 257 | t118: 1635 ± 257 |
| la marquise douairière de Cambremer | 6 | t158: 1490 ± 200 | t761: 1506 ± 128 | t158: 1490 ± 200 | t761: 1506 ± 128 |
| Albertine | 126 | t229: 1546 ± 100 | t918: 1455 ± 78 | t873: 1453 ± 61 | t347: 1584 ± 76 |
| baron de Charlus | 118 | t56: 1567 ± 109 | t938: 1448 ± 71 | t912: 1444 ± 63 | t523: 1572 ± 68 |
| M. Vinteuil | 15 | t45: 1505 ± 123 | t898: 1498 ± 122 | t898: 1498 ± 122 | t176: 1520 ± 124 |
| Mme de Villeparisis | 78 | t3: 1461 ± 141 | t882: 1487 ± 111 | t590: 1459 ± 72 | t473: 1503 ± 60 |
| Mme de Sévigné | 4 | t269: 1559 ± 171 | t729: 1538 ± 162 | t729: 1538 ± 162 | t269: 1559 ± 171 |
| Mme Cottard | 11 | t87: 1608 ± 139 | t756: 1563 ± 188 | t756: 1563 ± 188 | t186: 1611 ± 131 |
| Swann | 198 | t2: 1504 ± 77 | t938: 1444 ± 72 | t718: 1433 ± 56 | t2: 1504 ± 77 |
| M. de Chevregny | 1 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 |
| M. de Crécy | 1 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 |
| Mme Féré | 1 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 | t761: 1543 ± 173 |
| général de Froberville | 7 | t157: 1521 ± 157 | t696: 1526 ± 156 | t157: 1521 ± 157 | t696: 1526 ± 156 |
| le pianiste | 3 | t85: 1586 ± 218 | t124: 1587 ± 218 | t85: 1586 ± 218 | t124: 1587 ± 218 |
| princesse de Luxembourg | 6 | t283: 1533 ± 163 | t730: 1521 ± 152 | t730: 1521 ± 152 | t325: 1534 ± 159 |
| M. Ski | 2 | t748: 1517 ± 157 | t825: 1524 ± 156 | t748: 1517 ± 157 | t825: 1524 ± 156 |
| M. Nissim Bernard | 7 | t315: 1513 ± 171 | t923: 1501 ± 134 | t923: 1501 ± 134 | t509: 1517 ± 145 |
| le directeur | 11 | t270: 1513 ± 134 | t737: 1501 ± 135 | t737: 1501 ± 135 | t270: 1513 ± 134 |
| Bloch père | 8 | t313: 1443 ± 137 | t923: 1495 ± 131 | t314: 1443 ± 137 | t761: 1497 ± 117 |
| Marie-Aynard | 1 | t480: 1619 ± 257 | t480: 1619 ± 257 | t480: 1619 ± 257 | t480: 1619 ± 257 |
| Victurnienne | 1 | t480: 1619 ± 257 | t480: 1619 ± 257 | t480: 1619 ± 257 | t480: 1619 ± 257 |
| le jeune marquis de Cambremer | 1 | t890: 1558 ± 196 | t890: 1558 ± 196 | t890: 1558 ± 196 | t890: 1558 ± 196 |
| duchesse de La Trémoïlle | 1 | t119: 1804 ± 443 | t119: 1804 ± 443 | t119: 1804 ± 443 | t119: 1804 ± 443 |
| Legrandin | 20 | t17: 1389 ± 156 | t930: 1464 ± 104 | t266: 1387 ± 129 | t761: 1469 ± 96 |
| prince des Laumes | 3 | t177: 1549 ± 155 | t596: 1505 ± 147 | t596: 1505 ± 147 | t177: 1549 ± 155 |
| Esther | 2 | t791: 1533 ± 181 | t792: 1533 ± 181 | t791: 1533 ± 181 | t791: 1533 ± 181 |
| Bismarck | 1 | t210: 1683 ± 332 | t210: 1683 ± 332 | t210: 1683 ± 332 | t210: 1683 ± 332 |
| la duchesse d'Alençon | 1 | t628: 1649 ± 300 | t628: 1649 ± 300 | t628: 1649 ± 300 | t628: 1649 ± 300 |
| Mme Legrandin mère | 1 | t266: 1586 ± 243 | t266: 1586 ± 243 | t266: 1586 ± 243 | t266: 1586 ± 243 |
| Victoire | 1 | t266: 1586 ± 243 | t266: 1586 ± 243 | t266: 1586 ± 243 | t266: 1586 ± 243 |
| duc de Chartres | 1 | t696: 1529 ± 187 | t696: 1529 ± 187 | t696: 1529 ± 187 | t696: 1529 ± 187 |
| prince de Chimay | 1 | t696: 1529 ± 187 | t696: 1529 ± 187 | t696: 1529 ± 187 | t696: 1529 ± 187 |
| princesse de Guermantes | 25 | t363: 1558 ± 113 | t932: 1453 ± 112 | t932: 1453 ± 112 | t366: 1559 ± 113 |
| marquis Maurice de Vaudémont | 1 | t353: 1797 ± 459 | t353: 1797 ± 459 | t353: 1797 ± 459 | t353: 1797 ± 459 |
| Mme de Montmorency | 1 | t718: 1540 ± 205 | t718: 1540 ± 205 | t718: 1540 ± 205 | t718: 1540 ± 205 |
| Mme de Rochechouart | 1 | t718: 1540 ± 205 | t718: 1540 ± 205 | t718: 1540 ± 205 | t718: 1540 ± 205 |
| M. de Goncourt | 1 | t897: 1570 ± 240 | t897: 1570 ± 240 | t897: 1570 ± 240 | t897: 1570 ± 240 |
| comtesse Molé | 6 | t668: 1452 ± 129 | t870: 1465 ± 136 | t668: 1452 ± 129 | t870: 1465 ± 136 |
| prince d’Agrigente | 2 | t630: 1517 ± 187 | t870: 1512 ± 183 | t870: 1512 ± 183 | t630: 1517 ± 187 |
| princesse de Parme | 38 | t363: 1396 ± 127 | t724: 1423 ± 97 | t570: 1395 ± 78 | t724: 1423 ± 97 |
| le baron Bréau-Chenut | 1 | t229: 1584 ± 259 | t229: 1584 ± 259 | t229: 1584 ± 259 | t229: 1584 ± 259 |
| le vieux père Chenut | 1 | t229: 1584 ± 259 | t229: 1584 ± 259 | t229: 1584 ± 259 | t229: 1584 ± 259 |
| Sarah Bernhardt | 1 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 |
| le jeune prince de Foix | 1 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 |
| vicomte de Courvoisier | 1 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 | t908: 1585 ± 260 |
| Céline | 2 | t4: 1486 ± 184 | t266: 1509 ± 186 | t4: 1486 ± 184 | t266: 1509 ± 186 |
| cousine Poictiers | 1 | t414: 1616 ± 293 | t414: 1616 ± 293 | t414: 1616 ± 293 | t414: 1616 ± 293 |
| duc de Poictiers | 1 | t414: 1616 ± 293 | t414: 1616 ± 293 | t414: 1616 ± 293 | t414: 1616 ± 293 |
| général de Monserfeuil | 4 | t628: 1488 ± 166 | t631: 1488 ± 166 | t631: 1488 ± 166 | t628: 1488 ± 166 |
| marquis du Lau | 2 | t775: 1643 ± 328 | t869: 1650 ± 328 | t775: 1643 ± 328 | t869: 1650 ± 328 |
| M. de Beauserfeuil | 1 | t644: 1570 ± 250 | t644: 1570 ± 250 | t644: 1570 ± 250 | t644: 1570 ± 250 |
| prince de Foix | 3 | t580: 1489 ± 196 | t908: 1514 ± 196 | t580: 1489 ± 196 | t908: 1514 ± 196 |
| duc de Guermantes | 107 | t362: 1459 ± 99 | t938: 1401 ± 86 | t938: 1401 ± 86 | t464: 1465 ± 71 |
| Mlle d'Éporcheville | 2 | t863: 1526 ± 213 | t865: 1525 ± 212 | t865: 1525 ± 212 | t863: 1526 ± 213 |
| oncle Adolphe | 5 | t21: 1461 ± 163 | t501: 1505 ± 192 | t21: 1461 ± 163 | t501: 1505 ± 192 |
| Émilie Daltier | 1 | t839: 1700 ± 389 | t839: 1700 ± 389 | t839: 1700 ± 389 | t839: 1700 ± 389 |
| Arnulphe | 1 | t703: 1634 ± 324 | t703: 1634 ± 324 | t703: 1634 ± 324 | t703: 1634 ± 324 |
| Mme de Stermaria | 1 | t566: 1595 ± 289 | t566: 1595 ± 289 | t566: 1595 ± 289 | t566: 1595 ± 289 |
| jeune blonde de Rivebelle | 2 | t325: 1570 ± 267 | t326: 1570 ± 267 | t325: 1570 ± 267 | t325: 1570 ± 267 |
| M. d'Argencourt | 12 | t453: 1524 ± 102 | t911: 1460 ± 158 | t911: 1460 ± 158 | t464: 1524 ± 99 |
| Rosemonde | 4 | t345: 1450 ± 168 | t729: 1467 ± 170 | t345: 1450 ± 168 | t727: 1467 ± 170 |
| Goncourt | 2 | t896: 1468 ± 171 | t898: 1468 ± 171 | t898: 1468 ± 171 | t896: 1468 ± 171 |
| Herbinger | 1 | t108: 1682 ± 386 | t108: 1682 ± 386 | t108: 1682 ± 386 | t108: 1682 ± 386 |
| Bibi | 1 | t579: 1765 ± 470 | t579: 1765 ± 470 | t579: 1765 ± 470 | t579: 1765 ± 470 |
| d'Orléans | 1 | t325: 1582 ± 289 | t325: 1582 ± 289 | t325: 1582 ± 289 | t325: 1582 ± 289 |
| M. de Vaugoubert | 8 | t209: 1478 ± 189 | t822: 1430 ± 137 | t778: 1428 ± 132 | t209: 1478 ± 189 |
| M. de Marsantes | 2 | t299: 1534 ± 264 | t509: 1545 ± 253 | t299: 1534 ± 264 | t509: 1545 ± 253 |
| Mlle Bloch | 1 | t732: 1761 ± 469 | t732: 1761 ± 469 | t732: 1761 ± 469 | t732: 1761 ± 469 |
| Sir Rufus Israël | 1 | t459: 1550 ± 259 | t459: 1550 ± 259 | t459: 1550 ± 259 | t459: 1550 ± 259 |
| Élisabeth | 1 | t791: 1556 ± 267 | t791: 1556 ± 267 | t791: 1556 ± 267 | t791: 1556 ± 267 |
| prince de Sagan | 1 | t710: 1539 ± 250 | t710: 1539 ± 250 | t710: 1539 ± 250 | t710: 1539 ± 250 |
| Léonor de Cambremer | 1 | t923: 1490 ± 202 | t923: 1490 ± 202 | t923: 1490 ± 202 | t923: 1490 ± 202 |
| Létourville | 1 | t921: 1675 ± 387 | t921: 1675 ± 387 | t921: 1675 ± 387 | t921: 1675 ± 387 |
| Manet | 1 | t637: 1574 ± 290 | t637: 1574 ± 290 | t637: 1574 ± 290 | t637: 1574 ± 290 |
| Mme de Cambremer | 19 | t165: 1348 ± 143 | t923: 1384 ± 102 | t446: 1341 ± 116 | t923: 1384 ± 102 |
| M. de La Rochefoucauld | 1 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 |
| duchesse de La Rochefoucauld | 1 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 |
| duchesse de Praslin | 1 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 | t297: 1550 ± 268 |
| duc d'Aumale | 2 | t366: 1616 ± 347 | t664: 1632 ± 352 | t366: 1616 ± 347 | t664: 1632 ± 352 |
| M. Carnot | 1 | t663: 1499 ± 221 | t663: 1499 ± 221 | t663: 1499 ± 221 | t663: 1499 ± 221 |
| Mme Carnot | 1 | t663: 1499 ± 221 | t663: 1499 ± 221 | t663: 1499 ± 221 | t663: 1499 ± 221 |
| Mme Putbus | 1 | t792: 1511 ± 233 | t792: 1511 ± 233 | t792: 1511 ± 233 | t792: 1511 ± 233 |
| comte de Paris | 3 | t192: 1488 ± 214 | t219: 1489 ± 214 | t192: 1488 ± 214 | t219: 1489 ± 214 |
| Mme Timoléon d'Amoncourt | 1 | t694: 1498 ± 225 | t694: 1498 ± 225 | t694: 1498 ± 225 | t694: 1498 ± 225 |
| Lady Israël | 1 | t491: 1566 ± 293 | t491: 1566 ± 293 | t491: 1566 ± 293 | t491: 1566 ± 293 |
| princesse d'Épinay | 3 | t593: 1475 ± 204 | t608: 1474 ± 204 | t608: 1474 ± 204 | t593: 1475 ± 204 |
| prince Von | 3 | t623: 1505 ± 234 | t641: 1505 ± 235 | t641: 1505 ± 235 | t623: 1505 ± 234 |
| duchesse de Létourville | 1 | t912: 1557 ± 290 | t912: 1557 ± 290 | t912: 1557 ± 290 | t912: 1557 ± 290 |
| le petit Cambremer | 1 | t888: 1454 ± 187 | t888: 1454 ± 187 | t888: 1454 ± 187 | t888: 1454 ± 187 |
| princesse de Silistrie | 1 | t888: 1454 ± 187 | t888: 1454 ± 187 | t888: 1454 ± 187 | t888: 1454 ± 187 |
| Napoléon III | 1 | t186: 1504 ± 238 | t186: 1504 ± 238 | t186: 1504 ± 238 | t186: 1504 ± 238 |
| d’Orgeville | 1 | t701: 1511 ± 247 | t701: 1511 ± 247 | t701: 1511 ± 247 | t701: 1511 ± 247 |
| le commandant Duroc | 1 | t396: 1741 ± 478 | t396: 1741 ± 478 | t396: 1741 ± 478 | t396: 1741 ± 478 |
| monsieur Vallenères | 1 | t457: 1739 ± 479 | t457: 1739 ± 479 | t457: 1739 ± 479 | t457: 1739 ± 479 |
| le marquis de Ganançay | 1 | t367: 1545 ± 285 | t367: 1545 ± 285 | t367: 1545 ± 285 | t367: 1545 ± 285 |
| le marquis de Palancy | 1 | t367: 1545 ± 285 | t367: 1545 ± 285 | t367: 1545 ± 285 | t367: 1545 ± 285 |
| Lady Israels | 1 | t232: 1738 ± 479 | t232: 1738 ± 479 | t232: 1738 ± 479 | t232: 1738 ± 479 |
| Thibaud | 1 | t780: 1491 ± 232 | t780: 1491 ± 232 | t780: 1491 ± 232 | t780: 1491 ± 232 |
| l'abbé Poiré | 1 | t708: 1469 ± 211 | t708: 1469 ± 211 | t708: 1469 ± 211 | t708: 1469 ± 211 |
| la Berma | 16 | t21: 1556 ± 131 | t936: 1396 ± 140 | t936: 1396 ± 140 | t36: 1556 ± 129 |
| Lady Rufus Israël | 1 | t868: 1522 ± 266 | t868: 1522 ± 266 | t868: 1522 ± 266 | t868: 1522 ± 266 |
| M. de Bornier | 1 | t609: 1564 ± 309 | t609: 1564 ± 309 | t609: 1564 ± 309 | t609: 1564 ± 309 |
| Gisèle | 5 | t342: 1408 ± 212 | t812: 1451 ± 200 | t342: 1408 ± 212 | t812: 1451 ± 200 |
| duc de Sidonia | 1 | t684: 1736 ± 487 | t684: 1736 ± 487 | t684: 1736 ± 487 | t684: 1736 ± 487 |
| M. Vibert | 1 | t618: 1606 ± 358 | t618: 1606 ± 358 | t618: 1606 ± 358 | t618: 1606 ± 358 |
| Théodore | 1 | t59: 1663 ± 416 | t59: 1663 ± 416 | t59: 1663 ± 416 | t59: 1663 ± 416 |
| Mme d'Arpajon | 8 | t597: 1368 ± 131 | t718: 1375 ± 129 | t614: 1366 ± 130 | t718: 1375 ± 129 |
| comtesse douairière d'Argencourt | 1 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 |
| duchesse de Gallardon douairière | 1 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 |
| marquis de Fierbois | 1 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 | t590: 1456 ± 212 |
| M. Swann, le père | 1 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 |
| le comte de Paris | 1 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 |
| le prince de Galles | 1 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 | t2: 1505 ± 261 |
| Dechambre | 1 | t745: 1647 ± 404 | t745: 1647 ± 404 | t745: 1647 ± 404 | t745: 1647 ± 404 |
| Barrès | 1 | t661: 1461 ± 226 | t661: 1461 ± 226 | t661: 1461 ± 226 | t661: 1461 ± 226 |
| Clémenceau | 1 | t661: 1461 ± 226 | t661: 1461 ± 226 | t661: 1461 ± 226 | t661: 1461 ± 226 |
| M. de Chateaubriand | 2 | t294: 1409 ± 242 | t870: 1444 ± 209 | t294: 1409 ± 242 | t870: 1444 ± 209 |
| Balzac | 2 | t295: 1390 ± 190 | t898: 1419 ± 184 | t295: 1390 ± 190 | t898: 1419 ± 184 |
| M. Arthur Meyer | 1 | t911: 1495 ± 264 | t911: 1495 ± 264 | t911: 1495 ± 264 | t911: 1495 ± 264 |
| Mme de Sagan | 1 | t485: 1589 ± 359 | t485: 1589 ± 359 | t485: 1589 ± 359 | t485: 1589 ± 359 |
| Coquelin | 1 | t198: 1515 ± 288 | t198: 1515 ± 288 | t198: 1515 ± 288 | t198: 1515 ± 288 |
| Sainte-Beuve | 1 | t583: 1474 ± 250 | t583: 1474 ± 250 | t583: 1474 ± 250 | t583: 1474 ± 250 |
| Liszt | 1 | t440: 1493 ± 274 | t440: 1493 ± 274 | t440: 1493 ± 274 | t440: 1493 ± 274 |
| Mme Ristori | 1 | t440: 1493 ± 274 | t440: 1493 ± 274 | t440: 1493 ± 274 | t440: 1493 ± 274 |
| M. de Stermaria | 4 | t275: 1434 ± 215 | t280: 1434 ± 215 | t279: 1434 ± 215 | t275: 1434 ± 215 |
| le roi Théodose | 3 | t208: 1475 ± 252 | t693: 1463 ± 246 | t693: 1463 ± 246 | t208: 1475 ± 252 |
| le grand-duc Wladimir | 1 | t689: 1579 ± 363 | t689: 1579 ± 363 | t689: 1579 ± 363 | t689: 1579 ± 363 |
| duc de Châtellerault | 5 | t488: 1456 ± 226 | t683: 1448 ± 233 | t683: 1448 ± 233 | t488: 1456 ± 226 |
| marquis de Cambremer | 6 | t277: 1425 ± 164 | t761: 1330 ± 117 | t761: 1330 ± 117 | t277: 1425 ± 164 |
| Dostoïevski | 1 | t836: 1476 ± 264 | t836: 1476 ± 264 | t836: 1476 ± 264 | t836: 1476 ± 264 |
| docteur Percepied | 1 | t58: 1524 ± 313 | t58: 1524 ± 313 | t58: 1524 ± 313 | t58: 1524 ± 313 |
| Mlle de l’Orgeville | 1 | t892: 1567 ± 358 | t892: 1567 ± 358 | t892: 1567 ± 358 | t892: 1567 ± 358 |
| L’excellent écrivain G… | 1 | t448: 1523 ± 317 | t448: 1523 ± 317 | t448: 1523 ± 317 | t448: 1523 ± 317 |
| Mme de Vaugoubert | 2 | t686: 1432 ± 242 | t822: 1437 ± 233 | t686: 1432 ± 242 | t822: 1437 ± 233 |
| Octave | 2 | t340: 1490 ± 324 | t875: 1535 ± 332 | t340: 1490 ± 324 | t875: 1535 ± 332 |
| Mme Trombert | 1 | t231: 1515 ± 314 | t231: 1515 ± 314 | t231: 1515 ± 314 | t231: 1515 ± 314 |
| comtesse de Monteriender | 1 | t176: 1510 ± 313 | t176: 1510 ± 313 | t176: 1510 ± 313 | t176: 1510 ± 313 |
| marquise de Gallardon | 7 | t158: 1353 ± 195 | t711: 1373 ± 184 | t158: 1353 ± 195 | t710: 1373 ± 184 |
| Madame Elstir | 1 | t341: 1453 ± 267 | t341: 1453 ± 267 | t341: 1453 ± 267 | t341: 1453 ± 267 |
| les demoiselles d’Ambresac | 1 | t341: 1453 ± 267 | t341: 1453 ± 267 | t341: 1453 ± 267 | t341: 1453 ± 267 |
| M. de Miribel | 1 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 |
| le lieutenant-colonel Henry | 1 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 |
| le lieutenant-colonel Picquart | 1 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 | t476: 1496 ± 313 |
| princesse Mathilde | 2 | t238: 1413 ± 268 | t595: 1434 ± 260 | t238: 1413 ± 268 | t595: 1434 ± 260 |
| le prince Von | 2 | t625: 1386 ± 222 | t640: 1384 ± 223 | t640: 1384 ± 223 | t625: 1386 ± 222 |
| Duroc | 1 | t395: 1677 ± 519 | t395: 1677 ± 519 | t395: 1677 ± 519 | t395: 1677 ± 519 |
| princesse Sherbatoff | 5 | t742: 1332 ± 174 | t757: 1328 ± 173 | t757: 1328 ± 173 | t742: 1332 ± 174 |
| D'Annunzio | 1 | t693: 1439 ± 289 | t693: 1439 ± 289 | t693: 1439 ± 289 | t693: 1439 ± 289 |
| Mme d'Heudicourt | 5 | t602: 1316 ± 178 | t609: 1317 ± 178 | t603: 1316 ± 178 | t608: 1317 ± 178 |
| M. d'Herweck | 2 | t698: 1421 ± 288 | t699: 1421 ± 288 | t698: 1421 ± 288 | t698: 1421 ± 288 |
| Mme de Franquetot | 3 | t158: 1414 ± 216 | t923: 1302 ± 170 | t923: 1302 ± 170 | t158: 1414 ± 216 |
| docteur Dieulafoy | 1 | t548: 1663 ± 533 | t548: 1663 ± 533 | t548: 1663 ± 533 | t548: 1663 ± 533 |
| M. Grevy | 1 | t94: 1478 ± 348 | t94: 1478 ± 348 | t94: 1478 ± 348 | t94: 1478 ± 348 |
| elle | 1 | t430: 1659 ± 535 | t430: 1659 ± 535 | t430: 1659 ± 535 | t430: 1659 ± 535 |
| M. de Grouchy | 4 | t587: 1351 ± 240 | t601: 1352 ± 240 | t587: 1351 ± 240 | t601: 1352 ± 240 |
| M. Molé | 1 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 |
| M. de Bouillon | 1 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 |
| Musset | 1 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 |
| Victor Hugo | 1 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 | t295: 1346 ± 248 |
| Prince Henri d'Orléans | 1 | t483: 1497 ± 407 | t483: 1497 ± 407 | t483: 1497 ± 407 | t483: 1497 ± 407 |
| M. de Courgivaux | 1 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 | t924: 1630 ± 552 |
| Mme de Villebon | 1 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 | t589: 1630 ± 552 |
| le capitaine | 1 | t375: 1479 ± 402 | t375: 1479 ± 402 | t375: 1479 ± 402 | t375: 1479 ± 402 |
| Poullein | 2 | t600: 1480 ± 417 | t601: 1480 ± 417 | t601: 1480 ± 417 | t600: 1480 ± 417 |
| Vigny | 1 | t294: 1460 ± 404 | t294: 1460 ± 404 | t294: 1460 ± 404 | t294: 1460 ± 404 |
| le bâtonnier | 1 | t284: 1436 ± 379 | t284: 1436 ± 379 | t284: 1436 ± 379 | t284: 1436 ± 379 |
| prince d'Agrigente | 2 | t586: 1448 ± 406 | t922: 1463 ± 414 | t586: 1448 ± 406 | t922: 1463 ± 414 |
| Cartier | 1 | t777: 1388 ± 347 | t777: 1388 ± 347 | t777: 1388 ± 347 | t777: 1388 ± 347 |
| Beauserfeuil | 1 | t662: 1398 ± 357 | t662: 1398 ± 357 | t662: 1398 ± 357 | t662: 1398 ± 357 |
| Théodose Cadet | 1 | t665: 1397 ± 356 | t665: 1397 ± 356 | t665: 1397 ± 356 | t665: 1397 ± 356 |
| M. Barrère | 1 | t884: 1530 ± 494 | t884: 1530 ± 494 | t884: 1530 ± 494 | t884: 1530 ± 494 |
| princesse d'Iéna | 1 | t166: 1404 ± 370 | t166: 1404 ± 370 | t166: 1404 ± 370 | t166: 1404 ± 370 |
| professeur E… | 2 | t533: 1361 ± 338 | t685: 1366 ± 341 | t533: 1361 ± 338 | t685: 1366 ± 341 |
| capitaine de Borodino | 5 | t379: 1247 ± 224 | t459: 1244 ± 223 | t459: 1244 ± 223 | t379: 1247 ± 224 |
| princesse de Nassau | 1 | t931: 1513 ± 493 | t931: 1513 ± 493 | t931: 1513 ± 493 | t931: 1513 ± 493 |
| Mme de Souvré | 2 | t591: 1260 ± 248 | t687: 1263 ± 245 | t591: 1260 ± 248 | t687: 1263 ± 245 |
| Mme de Morienval | 1 | t367: 1302 ± 293 | t367: 1302 ± 293 | t367: 1302 ± 293 | t367: 1302 ± 293 |
| duchesse de Luxembourg | 1 | t367: 1302 ± 293 | t367: 1302 ± 293 | t367: 1302 ± 293 | t367: 1302 ± 293 |
| M. Bontemps | 2 | t229: 1276 ± 271 | t899: 1305 ± 307 | t229: 1276 ± 271 | t899: 1305 ± 307 |
| Antoine | 1 | t358: 1387 ± 400 | t358: 1387 ± 400 | t358: 1387 ± 400 | t358: 1387 ± 400 |
| Saniette | 8 | t121: 1169 ± 213 | t820: 1159 ± 181 | t820: 1159 ± 181 | t661: 1183 ± 173 |
| les Courvoisier | 1 | t595: 1298 ± 321 | t595: 1298 ± 321 | t595: 1298 ± 321 | t595: 1298 ± 321 |
| la jeune ouvriere | 1 | t96: 1389 ± 420 | t96: 1389 ± 420 | t96: 1389 ± 420 | t96: 1389 ± 420 |
| Maurice | 1 | t908: 1272 ± 305 | t908: 1272 ± 305 | t908: 1272 ± 305 | t908: 1272 ± 305 |
| M. de Luxembourg | 1 | t645: 1382 ± 419 | t645: 1382 ± 419 | t645: 1382 ± 419 | t645: 1382 ± 419 |
| marquise de Citri | 1 | t700: 1365 ± 417 | t700: 1365 ± 417 | t700: 1365 ± 417 | t700: 1365 ± 417 |
| baron de Guermantes | 1 | t452: 1559 ± 611 | t452: 1559 ± 611 | t452: 1559 ± 611 | t452: 1559 ± 611 |
| Monsieur Vallenères | 1 | t472: 1310 ± 385 | t472: 1310 ± 385 | t472: 1310 ± 385 | t472: 1310 ± 385 |
| Alix | 3 | t440: 1202 ± 291 | t445: 1202 ± 291 | t440: 1202 ± 291 | t440: 1202 ± 291 |
| comtesse G… | 1 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 | t589: 1370 ± 552 |
| vicomtesse de Saint-Fiacre | 1 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 | t924: 1370 ± 552 |
| la Charité de Giotto | 1 | t49: 1369 ± 551 | t49: 1369 ± 551 | t49: 1369 ± 551 | t49: 1369 ± 551 |
| ma grand'tante | 1 | t1: 1343 ± 536 | t1: 1343 ± 536 | t1: 1343 ± 536 | t1: 1343 ± 536 |
| prince Foggi | 1 | t883: 1343 ± 536 | t883: 1343 ± 536 | t883: 1343 ± 536 | t883: 1343 ± 536 |
| le professeur E… | 1 | t684: 1307 ± 504 | t684: 1307 ± 504 | t684: 1307 ± 504 | t684: 1307 ± 504 |
| la marquise | 1 | t528: 1325 ± 528 | t528: 1325 ± 528 | t528: 1325 ± 528 | t528: 1325 ± 528 |
| Madame d'Ambresac | 1 | t366: 1286 ± 490 | t366: 1286 ± 490 | t366: 1286 ± 490 | t366: 1286 ± 490 |
| le curé | 1 | t42: 1294 ± 501 | t42: 1294 ± 501 | t42: 1294 ± 501 | t42: 1294 ± 501 |
| Marie Gineste | 1 | t736: 1301 ± 511 | t736: 1301 ± 511 | t736: 1301 ± 511 | t736: 1301 ± 511 |
| prince de Léon | 1 | t775: 1276 ± 487 | t775: 1276 ± 487 | t775: 1276 ± 487 | t775: 1276 ± 487 |
| le prince von *** | 1 | t498: 1269 ± 482 | t498: 1269 ± 482 | t498: 1269 ± 482 | t498: 1269 ± 482 |
| le grand-duc héritier de Luxembourg | 1 | t581: 1301 ± 519 | t581: 1301 ± 519 | t581: 1301 ± 519 | t581: 1301 ± 519 |
| le diplomate belge | 1 | t493: 1254 ± 481 | t493: 1254 ± 481 | t493: 1254 ± 481 | t493: 1254 ± 481 |
| Dumont | 1 | t30: 1252 ± 480 | t30: 1252 ± 480 | t30: 1252 ± 480 | t30: 1252 ± 480 |
| Mme Blatin | 1 | t195: 1244 ± 475 | t195: 1244 ± 475 | t195: 1244 ± 475 | t195: 1244 ± 475 |
| Mme de Simiane | 1 | t269: 1223 ± 453 | t269: 1223 ± 453 | t269: 1223 ± 453 | t269: 1223 ± 453 |
| l'historien de la Fronde | 1 | t453: 1223 ± 455 | t453: 1223 ± 455 | t453: 1223 ± 455 | t453: 1223 ± 455 |
| prince de Faffenheim | 2 | t499: 1214 ± 450 | t500: 1214 ± 450 | t499: 1214 ± 450 | t499: 1214 ± 450 |
| la cousine d'Oriane | 1 | t606: 1193 ± 443 | t606: 1193 ± 443 | t606: 1193 ± 443 | t606: 1193 ± 443 |
| vicomtesse d'Égremont | 1 | t593: 1192 ± 443 | t593: 1192 ± 443 | t593: 1192 ± 443 | t593: 1192 ± 443 |
| l'ambassadrice de Turquie | 1 | t690: 1158 ± 424 | t690: 1158 ± 424 | t690: 1158 ± 424 | t690: 1158 ± 424 |
| Mme Blandais | 2 | t284: 1147 ± 421 | t288: 1147 ± 421 | t284: 1147 ± 421 | t284: 1147 ± 421 |
| M. Pierre | 2 | t438: 1143 ± 420 | t452: 1143 ± 421 | t452: 1143 ± 421 | t438: 1143 ± 420 |
| Mme Iéna | 1 | t635: 1121 ± 409 | t635: 1121 ± 409 | t635: 1121 ± 409 | t635: 1121 ± 409 |
| ma grand’tante | 1 | t2: 1113 ± 401 | t2: 1113 ± 401 | t2: 1113 ± 401 | t2: 1113 ± 401 |
| le prince de Faffenheim | 1 | t497: 1119 ± 408 | t497: 1119 ± 408 | t497: 1119 ± 408 | t497: 1119 ± 408 |
| Mme de Varambon | 2 | t616: 1127 ± 416 | t648: 1127 ± 417 | t648: 1127 ± 417 | t616: 1127 ± 416 |
| l'empereur | 1 | t640: 1120 ± 413 | t640: 1120 ± 413 | t640: 1120 ± 413 | t640: 1120 ± 413 |
| Picquart | 2 | t395: 1094 ± 397 | t482: 1090 ± 397 | t482: 1090 ± 397 | t395: 1094 ± 397 |
| colonel de Froberville | 1 | t696: 991 ± 361 | t696: 991 ± 361 | t696: 991 ± 361 | t696: 991 ± 361 |
| M. de Vigny | 1 | t295: 994 ± 368 | t295: 994 ± 368 | t295: 994 ± 368 | t295: 994 ± 368 |
