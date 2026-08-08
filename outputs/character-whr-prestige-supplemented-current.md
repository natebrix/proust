# Character Whole-History Rating

- Analysis version: `character_whr_prestige_v1`
- Lens: `prestige`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `70`
- Match count: `1801`
- Time point count: `686`
- Node count: `1829`
- Draw rate: `0.162`
- Draw model: `half_win_half_loss`
- w2: `5.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.093`s, filtered `19.923`s (all w2 candidates `107.142`s)
- Convergence: smoothed `18` sweeps (converged: `True`), filtered `686` fits / `9083` sweeps, `0` of them unconverged
- Supplemented: `true` (runs: supplement-run-001, supplement-run-002, supplement-run-003, supplement-run-004, supplement-run-005, supplement-run-006, supplement-run-007, supplement-run-008, supplement-run-009, supplement-run-010, supplement-run-011, supplement-run-012, supplement-run-013, supplement-run-014, supplement-run-015, supplement-run-016, supplement-run-017, supplement-run-018, supplement-run-019, supplement-run-020, supplement-run-021, supplement-run-022, supplement-run-023, supplement-run-024, supplement-run-025, supplement-run-026, supplement-run-027, supplement-run-028, supplement-run-029)

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.72552 | 0.261259 | 1801 | filtered WHR at w2=5 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.712916 | 0.257114 | 1801 | filtered WHR at w2=5, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.67656 | 0.241879 | 1801 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.692593 | 0.249447 | 1801 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.734111 | 0.266045 | 1801 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.72552 | 0.261259 | 19.923 |
| 15.0 | 0.72723 | 0.261836 | 23.261 |
| 35.0 | 0.731955 | 0.263588 | 28.875 |
| 60.0 | 0.737718 | 0.265712 | 35.083 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Prestige |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jupien | 1741 ± 166 | 1575.1 | 165.8 | 29 | 23-5-1 | 16 | 16 | +0.547 |
| Aimé | 1731 ± 182 | 1549.5 | 181.9 | 23 | 17-3-3 | 14 | 13 | +0.372 |
| duchesse de Guermantes | 1561 ± 78 | 1482.8 | 78.0 | 244 | 140-68-36 | 163 | 134 | +0.387 |
| le narrateur | 1558 ± 77 | 1480.7 | 77.3 | 281 | 130-96-55 | 128 | 128 | +0.006 |
| Elstir | 1598 ± 120 | 1477.3 | 120.4 | 52 | 28-13-11 | 30 | 24 | +1.039 |
| Robert de Saint-Loup | 1554 ± 93 | 1461.7 | 92.6 | 137 | 65-56-16 | 134 | 77 | +0.082 |
| Andrée | 1582 ± 129 | 1453.3 | 128.7 | 36 | 18-11-7 | 22 | 20 | -0.028 |
| la mère du narrateur | 1560 ± 107 | 1452.4 | 107.2 | 92 | 45-30-17 | 54 | 45 | +0.243 |
| princesse de Guermantes | 1603 ± 152 | 1451.0 | 151.9 | 25 | 15-8-2 | 18 | 14 | +0.605 |
| la grand-mère | 1548 ± 101 | 1446.5 | 101.1 | 112 | 41-40-31 | 71 | 58 | -0.051 |
| Mme de Villeparisis | 1557 ± 112 | 1444.3 | 112.4 | 90 | 41-30-19 | 69 | 55 | -0.037 |
| Odette | 1541 ± 98 | 1443.1 | 98.0 | 205 | 100-75-30 | 123 | 104 | -0.042 |
| Gilberte | 1538 ± 95 | 1442.3 | 95.3 | 84 | 42-34-8 | 56 | 36 | +0.135 |
| Françoise | 1508 ± 89 | 1419.6 | 88.6 | 131 | 55-54-22 | 89 | 63 | -0.08 |
| M. Vinteuil | 1592 ± 174 | 1417.9 | 173.6 | 26 | 14-9-3 | 21 | 15 | +0.475 |
| le père du narrateur | 1543 ± 133 | 1409.8 | 133.2 | 48 | 23-15-10 | 28 | 25 | +0.071 |
| Mme Verdurin | 1502 ± 93 | 1408.3 | 93.2 | 151 | 64-61-26 | 80 | 70 | -0.223 |
| le peintre | 1532 ± 133 | 1399.5 | 132.9 | 29 | 13-7-9 | 13 | 13 | +0.022 |
| Albertine | 1460 ± 75 | 1385.2 | 74.7 | 161 | 62-86-13 | 137 | 93 | -0.348 |
| Morel | 1483 ± 107 | 1376.5 | 106.7 | 50 | 22-21-7 | 24 | 22 | -0.811 |
| baron de Charlus | 1450 ± 78 | 1372.0 | 78.0 | 155 | 65-71-19 | 108 | 78 | -0.566 |
| duc de Guermantes | 1442 ± 89 | 1352.3 | 89.2 | 138 | 46-72-20 | 80 | 67 | -0.593 |
| docteur Cottard | 1462 ± 119 | 1342.7 | 119.0 | 89 | 30-33-26 | 54 | 44 | -0.452 |
| Legrandin | 1485 ± 146 | 1339.8 | 145.5 | 33 | 13-17-3 | 23 | 19 | -0.636 |
| Norpois | 1447 ± 110 | 1337.1 | 110.1 | 96 | 30-45-21 | 72 | 56 | -0.023 |
| M. Verdurin | 1486 ± 162 | 1323.8 | 162.0 | 32 | 12-11-9 | 22 | 20 | -0.648 |
| la Berma | 1515 ± 192 | 1323.0 | 192.2 | 14 | 6-6-2 | 10 | 9 | +0.382 |
| Mme Bontemps | 1466 ± 144 | 1322.1 | 143.9 | 29 | 11-13-5 | 14 | 14 | -0.788 |
| Bergotte | 1441 ± 123 | 1318.5 | 122.6 | 56 | 22-27-7 | 39 | 28 | +0.353 |
| Bloch | 1410 ± 96 | 1314.6 | 95.9 | 113 | 33-64-16 | 63 | 55 | -0.802 |
| princesse de Parme | 1427 ± 117 | 1310.0 | 117.4 | 45 | 13-25-7 | 23 | 20 | -0.066 |
| M. de Marsantes | 1479 ± 169 | 1309.8 | 169.2 | 21 | 7-10-4 | 12 | 11 | +0.035 |
| Swann | 1389 ± 91 | 1298.4 | 90.7 | 328 | 108-172-48 | 270 | 183 | -0.474 |
| comte de Forcheville | 1446 ± 158 | 1287.5 | 158.1 | 49 | 21-19-9 | 25 | 23 | +0.227 |
| Mme de Cambremer | 1415 ± 137 | 1277.6 | 137.2 | 37 | 16-18-3 | 23 | 18 | -0.721 |
| Mme Cottard | 1431 ± 154 | 1276.9 | 154.1 | 27 | 10-12-5 | 15 | 12 | +0.015 |
| le directeur | 1419 ± 145 | 1274.2 | 144.7 | 32 | 7-15-10 | 18 | 16 | -0.589 |
| marquis de Cambremer | 1426 ± 160 | 1266.8 | 159.6 | 21 | 7-10-4 | 9 | 8 | -0.545 |
| général de Froberville | 1373 ± 165 | 1208.7 | 164.6 | 22 | 6-11-5 | 8 | 7 | -0.742 |
| marquis de Bréauté | 1356 ± 166 | 1190.5 | 165.6 | 23 | 5-13-5 | 7 | 7 | -1.184 |
| marquise de Saint-Euverte | 1348 ± 170 | 1177.8 | 169.7 | 19 | 6-12-1 | 7 | 7 | -1.53 |
| Brichot | 1302 ± 140 | 1163.0 | 139.5 | 34 | 8-23-3 | 13 | 13 | -0.716 |
| marquise de Gallardon | 1333 ± 180 | 1153.3 | 180.0 | 20 | 5-14-1 | 12 | 12 | -1.325 |
| M. de Vaugoubert | 1249 ± 182 | 1067.1 | 181.8 | 23 | 4-18-1 | 10 | 9 | -0.66 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Octave | 1868 ± 427 | 426.9 | 4 | 2 | 2 | 1086 |
| Mlle d'Éporcheville | 1792 ± 458 | 458.4 | 2 | 1 | 1 | 1074 |
| Mme Blandais | 1783 ± 461 | 461.3 | 2 | 2 | 2 | 470 |
| jeune blonde de Rivebelle | 1778 ± 463 | 463.0 | 2 | 1 | 1 | 519 |
| le pianiste | 1773 ± 348 | 347.7 | 6 | 3 | 3 | 112 |
| Rémi | 1699 ± 372 | 372.0 | 4 | 3 | 3 | 153 |
| le grand-père du narrateur | 1671 ± 225 | 224.9 | 16 | 8 | 8 | 1143 |
| la reine de Naples | 1666 ± 532 | 531.9 | 1 | 1 | 1 | 1035 |
| princesse de Luxembourg | 1616 ± 224 | 223.6 | 10 | 4 | 4 | 842 |
| Napoléon III | 1611 ± 355 | 355.2 | 3 | 1 | 1 | 605 |
| Mme de Vaugoubert | 1550 ± 278 | 278.4 | 7 | 3 | 3 | 1029 |
| duc de Châtellerault | 1530 ± 235 | 235.2 | 9 | 6 | 6 | 1135 |
| Mlle de Stermaria | 1505 ± 292 | 291.7 | 5 | 3 | 2 | 463 |
| M. Nissim Bernard | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| marquis de Forestelle | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| prince des Laumes | 1490 ± 314 | 314.1 | 4 | 1 | 1 | 789 |
| M. Ski | 1485 ± 355 | 354.8 | 3 | 2 | 2 | 1138 |
| Remi | 1478 ± 404 | 403.6 | 2 | 1 | 1 | 159 |
| Dreyfus | 1446 ± 237 | 236.6 | 8 | 3 | 3 | 679 |
| M. de Chevregny | 1432 ± 373 | 372.6 | 3 | 1 | 1 | 963 |
| M. de Stermaria | 1394 ± 256 | 255.7 | 8 | 6 | 4 | 773 |
| Mme de Chaussepierre | 1360 ± 363 | 362.7 | 3 | 1 | 1 | 893 |
| Saniette | 1284 ± 202 | 202.4 | 21 | 12 | 10 | 1036 |
| oncle Adolphe | 1178 ± 431 | 431.0 | 4 | 4 | 3 | 30 |
| M. Grevy | 1166 ± 337 | 337.2 | 6 | 2 | 2 | 134 |
| Bloch père | 1142 ± 332 | 331.7 | 7 | 4 | 3 | 507 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Jupien | 16 | t551: 1738 ± 170 | t1154: 1741 ± 166 | t551: 1738 ± 170 | t1154: 1741 ± 166 |
| Aimé | 13 | t456: 1719 ± 183 | t1068: 1731 ± 182 | t456: 1719 ± 183 | t995: 1732 ± 180 |
| duchesse de Guermantes | 134 | t75: 1544 ± 106 | t1156: 1561 ± 78 | t75: 1544 ± 106 | t764: 1588 ± 55 |
| le narrateur | 128 | t9: 1502 ± 95 | t1144: 1558 ± 77 | t9: 1502 ± 95 | t1144: 1558 ± 77 |
| Elstir | 24 | t451: 1602 ± 116 | t1147: 1598 ± 120 | t1110: 1597 ± 118 | t556: 1603 ± 109 |
| Robert de Saint-Loup | 77 | t480: 1531 ± 76 | t1147: 1554 ± 93 | t678: 1522 ± 68 | t1147: 1554 ± 93 |
| Andrée | 20 | t534: 1572 ± 131 | t1113: 1582 ± 129 | t534: 1572 ± 131 | t1113: 1582 ± 129 |
| la mère du narrateur | 45 | t1: 1560 ± 100 | t1098: 1560 ± 107 | t723: 1558 ± 90 | t330: 1566 ± 87 |
| princesse de Guermantes | 14 | t561: 1618 ± 161 | t1153: 1603 ± 152 | t1147: 1603 ± 152 | t561: 1618 ± 161 |
| la grand-mère | 58 | t1: 1508 ± 105 | t1068: 1548 ± 101 | t1: 1508 ± 105 | t1068: 1548 ± 101 |
| le grand-père du narrateur | 8 | t1: 1698 ± 203 | t1143: 1671 ± 225 | t1143: 1671 ± 225 | t7: 1698 ± 203 |
| Mme de Villeparisis | 55 | t7: 1526 ± 124 | t1093: 1557 ± 112 | t7: 1526 ± 124 | t1091: 1557 ± 112 |
| Odette | 104 | t44: 1496 ± 79 | t1155: 1541 ± 98 | t44: 1496 ± 79 | t1154: 1541 ± 98 |
| Gilberte | 36 | t44: 1519 ± 118 | t1156: 1538 ± 95 | t442: 1511 ± 89 | t1152: 1538 ± 95 |
| Octave | 2 | t17: 1857 ± 425 | t1086: 1868 ± 427 | t17: 1857 ± 425 | t1086: 1868 ± 427 |
| le pianiste | 3 | t78: 1773 ± 347 | t112: 1773 ± 348 | t107: 1773 ± 348 | t78: 1773 ± 347 |
| Françoise | 63 | t9: 1548 ± 101 | t1126: 1508 ± 89 | t721: 1506 ± 75 | t9: 1548 ± 101 |
| M. Vinteuil | 15 | t52: 1538 ± 143 | t1129: 1592 ± 174 | t52: 1538 ± 143 | t1129: 1592 ± 174 |
| le père du narrateur | 25 | t8: 1577 ± 115 | t1005: 1543 ± 133 | t1005: 1543 ± 133 | t52: 1578 ± 113 |
| Mme Verdurin | 70 | t78: 1455 ± 75 | t1141: 1502 ± 93 | t133: 1454 ± 70 | t1141: 1502 ± 93 |
| le peintre | 13 | t89: 1528 ± 133 | t226: 1532 ± 133 | t89: 1528 ± 133 | t176: 1534 ± 131 |
| princesse de Luxembourg | 4 | t469: 1607 ± 224 | t842: 1616 ± 224 | t469: 1607 ± 224 | t783: 1616 ± 222 |
| Albertine | 93 | t514: 1506 ± 85 | t1133: 1460 ± 75 | t1045: 1458 ± 64 | t514: 1506 ± 85 |
| Morel | 22 | t945: 1489 ± 103 | t1123: 1483 ± 107 | t1112: 1483 ± 106 | t945: 1489 ± 103 |
| baron de Charlus | 78 | t267: 1528 ± 109 | t1154: 1450 ± 78 | t1154: 1450 ± 78 | t267: 1528 ± 109 |
| duc de Guermantes | 67 | t612: 1444 ± 82 | t1155: 1442 ± 89 | t1134: 1442 ± 88 | t843: 1451 ± 66 |
| docteur Cottard | 44 | t78: 1460 ± 88 | t1118: 1462 ± 119 | t942: 1456 ± 106 | t383: 1468 ± 87 |
| Legrandin | 19 | t23: 1482 ± 145 | t1137: 1485 ± 146 | t59: 1482 ± 143 | t518: 1491 ± 130 |
| Norpois | 56 | t338: 1421 ± 85 | t1132: 1447 ± 110 | t418: 1419 ± 82 | t844: 1448 ± 92 |
| Mlle d'Éporcheville | 1 | t1074: 1792 ± 458 | t1074: 1792 ± 458 | t1074: 1792 ± 458 | t1074: 1792 ± 458 |
| Rémi | 3 | t151: 1699 ± 372 | t153: 1699 ± 372 | t151: 1699 ± 372 | t151: 1699 ± 372 |
| M. Verdurin | 20 | t82: 1465 ± 128 | t1118: 1486 ± 162 | t82: 1465 ± 128 | t1118: 1486 ± 162 |
| la Berma | 9 | t340: 1528 ± 194 | t1153: 1515 ± 192 | t1153: 1515 ± 192 | t344: 1528 ± 194 |
| Mme Bontemps | 14 | t397: 1463 ± 139 | t1112: 1466 ± 144 | t543: 1460 ± 137 | t1020: 1466 ± 140 |
| Mme Blandais | 2 | t467: 1783 ± 461 | t470: 1783 ± 461 | t467: 1783 ± 461 | t467: 1783 ± 461 |
| Bergotte | 28 | t41: 1460 ± 127 | t1132: 1441 ± 123 | t1131: 1441 ± 122 | t41: 1460 ± 127 |
| jeune blonde de Rivebelle | 1 | t519: 1778 ± 463 | t519: 1778 ± 463 | t519: 1778 ± 463 | t519: 1778 ± 463 |
| Bloch | 55 | t39: 1410 ± 117 | t1151: 1410 ± 96 | t504: 1396 ± 85 | t1143: 1410 ± 95 |
| princesse de Parme | 20 | t396: 1442 ± 128 | t924: 1427 ± 117 | t794: 1420 ± 110 | t396: 1442 ± 128 |
| M. de Marsantes | 11 | t628: 1466 ± 156 | t1126: 1479 ± 169 | t689: 1465 ± 152 | t1126: 1479 ± 169 |
| Swann | 183 | t3: 1448 ± 69 | t1144: 1389 ± 91 | t1143: 1389 ± 91 | t3: 1448 ± 69 |
| duc de Châtellerault | 6 | t653: 1540 ± 226 | t1135: 1530 ± 235 | t1135: 1530 ± 235 | t653: 1540 ± 226 |
| comte de Forcheville | 23 | t169: 1462 ± 102 | t1076: 1446 ± 158 | t1076: 1446 ± 158 | t218: 1464 ± 102 |
| Mme de Cambremer | 18 | t270: 1468 ± 131 | t1133: 1415 ± 137 | t1133: 1415 ± 137 | t270: 1468 ± 131 |
| Mme Cottard | 12 | t179: 1454 ± 145 | t959: 1431 ± 154 | t959: 1431 ± 154 | t179: 1454 ± 145 |
| le directeur | 16 | t453: 1421 ± 138 | t1115: 1419 ± 145 | t1115: 1419 ± 145 | t463: 1421 ± 138 |
| Mme de Vaugoubert | 3 | t884: 1550 ± 277 | t1029: 1550 ± 278 | t1029: 1550 ± 278 | t884: 1550 ± 277 |
| marquis de Cambremer | 8 | t458: 1444 ± 171 | t1137: 1426 ± 160 | t946: 1420 ± 154 | t458: 1444 ± 171 |
| Napoléon III | 1 | t605: 1611 ± 355 | t605: 1611 ± 355 | t605: 1611 ± 355 | t605: 1611 ± 355 |
| Mlle de Stermaria | 2 | t458: 1505 ± 292 | t463: 1505 ± 292 | t458: 1505 ± 292 | t458: 1505 ± 292 |
| Dreyfus | 3 | t677: 1446 ± 237 | t679: 1446 ± 237 | t677: 1446 ± 237 | t677: 1446 ± 237 |
| général de Froberville | 7 | t268: 1357 ± 159 | t895: 1373 ± 165 | t268: 1357 ± 159 | t895: 1373 ± 165 |
| marquis de Bréauté | 7 | t268: 1345 ± 178 | t1152: 1356 ± 166 | t268: 1345 ± 178 | t1152: 1356 ± 166 |
| marquise de Saint-Euverte | 7 | t279: 1332 ± 187 | t916: 1348 ± 170 | t279: 1332 ± 187 | t893: 1348 ± 169 |
| prince des Laumes | 1 | t789: 1490 ± 314 | t789: 1490 ± 314 | t789: 1490 ± 314 | t789: 1490 ± 314 |
| Brichot | 13 | t171: 1310 ± 162 | t1120: 1302 ± 140 | t1120: 1302 ± 140 | t201: 1310 ± 160 |
| marquise de Gallardon | 12 | t270: 1358 ± 177 | t909: 1333 ± 180 | t908: 1333 ± 180 | t270: 1358 ± 177 |
| M. de Stermaria | 4 | t458: 1387 ± 249 | t773: 1394 ± 256 | t458: 1387 ± 249 | t773: 1394 ± 256 |
| la reine de Naples | 1 | t1035: 1666 ± 532 | t1035: 1666 ± 532 | t1035: 1666 ± 532 | t1035: 1666 ± 532 |
| M. Ski | 2 | t943: 1486 ± 352 | t1138: 1485 ± 355 | t1138: 1485 ± 355 | t943: 1486 ± 352 |
| Saniette | 10 | t169: 1277 ± 172 | t1036: 1284 ± 202 | t186: 1276 ± 171 | t956: 1284 ± 199 |
| Remi | 1 | t159: 1478 ± 404 | t159: 1478 ± 404 | t159: 1478 ± 404 | t159: 1478 ± 404 |
| M. de Vaugoubert | 9 | t350: 1253 ± 197 | t1115: 1249 ± 182 | t889: 1236 ± 176 | t350: 1253 ± 197 |
| M. de Chevregny | 1 | t963: 1432 ± 373 | t963: 1432 ± 373 | t963: 1432 ± 373 | t963: 1432 ± 373 |
| Mme de Chaussepierre | 1 | t893: 1360 ± 363 | t893: 1360 ± 363 | t893: 1360 ± 363 | t893: 1360 ± 363 |
| M. Grevy | 2 | t133: 1166 ± 337 | t134: 1166 ± 337 | t133: 1166 ± 337 | t133: 1166 ± 337 |
| Bloch père | 3 | t485: 1142 ± 332 | t507: 1142 ± 332 | t505: 1142 ± 332 | t485: 1142 ± 332 |
| oncle Adolphe | 3 | t27: 1178 ± 431 | t30: 1178 ± 431 | t27: 1178 ± 431 | t27: 1178 ± 431 |
