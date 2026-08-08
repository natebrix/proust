# Character Whole-History Rating

- Analysis version: `character_whr_inclusion_v1`
- Lens: `inclusion`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `70`
- Match count: `1801`
- Time point count: `686`
- Node count: `1829`
- Draw rate: `0.156`
- Draw model: `half_win_half_loss`
- w2: `5.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.097`s, filtered `20.006`s (all w2 candidates `108.461`s)
- Convergence: smoothed `19` sweeps (converged: `True`), filtered `686` fits / `9150` sweeps, `0` of them unconverged
- Supplemented: `true` (runs: supplement-run-001, supplement-run-002, supplement-run-003, supplement-run-004, supplement-run-005, supplement-run-006, supplement-run-007, supplement-run-008, supplement-run-009, supplement-run-010, supplement-run-011, supplement-run-012, supplement-run-013, supplement-run-014, supplement-run-015, supplement-run-016, supplement-run-017, supplement-run-018, supplement-run-019, supplement-run-020, supplement-run-021, supplement-run-022, supplement-run-023, supplement-run-024, supplement-run-025, supplement-run-026, supplement-run-027, supplement-run-028, supplement-run-029)

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.720066 | 0.259364 | 1801 | filtered WHR at w2=5 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.709015 | 0.255669 | 1801 | filtered WHR at w2=5, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.673814 | 0.240473 | 1801 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.689723 | 0.247932 | 1801 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.72831 | 0.263859 | 1801 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.720066 | 0.259364 | 20.006 |
| 15.0 | 0.721936 | 0.259994 | 23.655 |
| 35.0 | 0.726899 | 0.261747 | 29.339 |
| 60.0 | 0.73287 | 0.263834 | 35.461 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aimé | 1762 ± 189 | 1572.5 | 189.1 | 23 | 18-3-2 | 14 | 13 | +0.376 |
| Jupien | 1726 ± 158 | 1568.6 | 157.6 | 29 | 21-5-3 | 16 | 16 | +0.451 |
| le narrateur | 1582 ± 77 | 1504.3 | 77.4 | 281 | 134-99-48 | 128 | 128 | -0.112 |
| Elstir | 1611 ± 121 | 1489.5 | 121.2 | 52 | 30-14-8 | 30 | 24 | +0.899 |
| duchesse de Guermantes | 1567 ± 78 | 1489.1 | 77.7 | 244 | 137-74-33 | 163 | 134 | +0.13 |
| Gilberte | 1577 ± 96 | 1480.8 | 95.8 | 84 | 45-33-6 | 56 | 36 | -0.157 |
| Andrée | 1609 ± 130 | 1479.0 | 130.4 | 36 | 20-11-5 | 22 | 20 | -0.058 |
| Robert de Saint-Loup | 1568 ± 92 | 1475.1 | 92.5 | 137 | 64-54-19 | 134 | 77 | -0.035 |
| la mère du narrateur | 1576 ± 107 | 1469.2 | 107.3 | 92 | 47-29-16 | 54 | 45 | +0.211 |
| princesse de Guermantes | 1620 ± 151 | 1468.4 | 151.4 | 25 | 15-8-2 | 18 | 14 | +0.25 |
| la grand-mère | 1568 ± 102 | 1466.5 | 101.5 | 112 | 43-40-29 | 71 | 58 | -0.165 |
| Mme de Villeparisis | 1573 ± 112 | 1460.4 | 112.4 | 90 | 41-28-21 | 69 | 55 | -0.14 |
| Françoise | 1524 ± 89 | 1435.4 | 88.8 | 131 | 57-53-21 | 89 | 63 | -0.199 |
| M. Vinteuil | 1598 ± 174 | 1424.2 | 173.7 | 26 | 13-8-5 | 21 | 15 | +0.458 |
| Mme Verdurin | 1518 ± 94 | 1423.9 | 93.7 | 151 | 66-56-29 | 80 | 70 | -0.401 |
| baron de Charlus | 1488 ± 78 | 1410.5 | 77.7 | 155 | 70-66-19 | 108 | 78 | -0.6 |
| le peintre | 1543 ± 133 | 1410.3 | 133.0 | 29 | 13-7-9 | 13 | 13 | -0.078 |
| Odette | 1505 ± 98 | 1407.3 | 97.9 | 205 | 98-76-31 | 123 | 104 | -0.256 |
| le père du narrateur | 1539 ± 133 | 1405.6 | 133.0 | 48 | 22-17-9 | 28 | 25 | -0.227 |
| princesse de Parme | 1508 ± 114 | 1394.0 | 113.9 | 45 | 19-22-4 | 23 | 20 | -0.137 |
| Albertine | 1461 ± 75 | 1386.1 | 75.3 | 161 | 60-90-11 | 137 | 93 | -0.606 |
| Morel | 1490 ± 107 | 1382.5 | 107.2 | 50 | 21-21-8 | 24 | 22 | -1.004 |
| Legrandin | 1529 ± 146 | 1382.1 | 146.5 | 33 | 15-16-2 | 23 | 19 | -0.678 |
| duc de Guermantes | 1456 ± 89 | 1366.8 | 88.9 | 138 | 43-70-25 | 80 | 67 | -0.662 |
| M. Verdurin | 1524 ± 164 | 1360.0 | 164.2 | 32 | 15-11-6 | 22 | 20 | -0.68 |
| Norpois | 1469 ± 110 | 1358.6 | 110.1 | 96 | 31-44-21 | 72 | 56 | -0.112 |
| Bloch | 1433 ± 96 | 1337.5 | 95.6 | 113 | 33-62-18 | 63 | 55 | -1.072 |
| Bergotte | 1446 ± 123 | 1323.4 | 122.7 | 56 | 22-28-6 | 39 | 28 | +0.045 |
| docteur Cottard | 1443 ± 120 | 1322.8 | 119.8 | 89 | 30-37-22 | 54 | 44 | -0.543 |
| marquis de Cambremer | 1475 ± 160 | 1315.7 | 159.5 | 21 | 8-9-4 | 9 | 8 | -0.833 |
| Swann | 1402 ± 91 | 1311.0 | 91.1 | 328 | 108-182-38 | 270 | 183 | -0.814 |
| la Berma | 1504 ± 193 | 1310.3 | 193.2 | 14 | 6-7-1 | 10 | 9 | +0.197 |
| M. de Marsantes | 1480 ± 170 | 1310.2 | 170.1 | 21 | 6-10-5 | 12 | 11 | -0.019 |
| comte de Forcheville | 1468 ± 159 | 1309.2 | 159.1 | 49 | 21-15-13 | 25 | 23 | +0.042 |
| Mme Bontemps | 1424 ± 146 | 1278.2 | 146.3 | 29 | 9-15-5 | 14 | 14 | -0.864 |
| général de Froberville | 1437 ± 162 | 1274.8 | 161.8 | 22 | 8-9-5 | 8 | 7 | -0.691 |
| Mme Cottard | 1412 ± 156 | 1255.8 | 156.0 | 27 | 9-14-4 | 15 | 12 | -0.191 |
| Mme de Cambremer | 1384 ± 139 | 1244.5 | 139.3 | 37 | 12-20-5 | 23 | 18 | -1.046 |
| marquise de Saint-Euverte | 1410 ± 167 | 1242.7 | 166.9 | 19 | 8-11-0 | 7 | 7 | -1.218 |
| marquise de Gallardon | 1410 ± 171 | 1239.1 | 171.4 | 20 | 6-12-2 | 12 | 12 | -1.514 |
| le directeur | 1388 ± 150 | 1238.4 | 149.5 | 32 | 8-19-5 | 18 | 16 | -0.771 |
| marquis de Bréauté | 1394 ± 164 | 1230.0 | 163.6 | 23 | 5-12-6 | 7 | 7 | -1.164 |
| Brichot | 1261 ± 149 | 1112.2 | 148.7 | 34 | 5-24-5 | 13 | 13 | -0.918 |
| M. de Vaugoubert | 1259 ± 181 | 1077.4 | 181.3 | 23 | 4-18-1 | 10 | 9 | -0.862 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Octave | 1879 ± 424 | 423.8 | 4 | 2 | 2 | 1086 |
| jeune blonde de Rivebelle | 1786 ± 460 | 460.2 | 2 | 1 | 1 | 519 |
| le grand-père du narrateur | 1719 ± 232 | 232.5 | 16 | 8 | 8 | 1143 |
| le pianiste | 1703 ± 311 | 310.7 | 6 | 3 | 3 | 112 |
| Rémi | 1698 ± 372 | 372.4 | 4 | 3 | 3 | 153 |
| la reine de Naples | 1682 ± 525 | 524.9 | 1 | 1 | 1 | 1035 |
| Mme Blandais | 1662 ± 412 | 412.4 | 2 | 2 | 2 | 470 |
| princesse de Luxembourg | 1631 ± 223 | 222.6 | 10 | 4 | 4 | 842 |
| Napoléon III | 1623 ± 354 | 354.4 | 3 | 1 | 1 | 605 |
| Mlle de Stermaria | 1580 ± 303 | 303.3 | 5 | 3 | 2 | 463 |
| Mlle d'Éporcheville | 1552 ± 403 | 402.9 | 2 | 1 | 1 | 1074 |
| duc de Châtellerault | 1507 ± 236 | 236.1 | 9 | 6 | 6 | 1135 |
| M. Ski | 1505 ± 355 | 354.7 | 3 | 2 | 2 | 1138 |
| M. Nissim Bernard | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| marquis de Forestelle | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| Remi | 1480 ± 404 | 404.2 | 2 | 1 | 1 | 159 |
| Mme de Vaugoubert | 1467 ± 265 | 265.4 | 7 | 3 | 3 | 1029 |
| Dreyfus | 1460 ± 236 | 236.3 | 8 | 3 | 3 | 679 |
| prince des Laumes | 1444 ± 318 | 318.4 | 4 | 1 | 1 | 789 |
| M. de Stermaria | 1416 ± 258 | 258.0 | 8 | 6 | 4 | 773 |
| M. Grevy | 1306 ± 284 | 284.2 | 6 | 2 | 2 | 134 |
| Bloch père | 1287 ± 279 | 278.9 | 7 | 4 | 3 | 507 |
| M. de Chevregny | 1201 ± 453 | 453.4 | 3 | 1 | 1 | 963 |
| oncle Adolphe | 1180 ± 432 | 432.0 | 4 | 4 | 3 | 30 |
| Mme de Chaussepierre | 1163 ± 434 | 434.5 | 3 | 1 | 1 | 893 |
| Saniette | 1155 ± 238 | 237.5 | 21 | 12 | 10 | 1036 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Aimé | 13 | t456: 1748 ± 189 | t1068: 1762 ± 189 | t456: 1748 ± 189 | t995: 1762 ± 187 |
| Jupien | 16 | t551: 1723 ± 163 | t1154: 1726 ± 158 | t876: 1723 ± 153 | t1154: 1726 ± 158 |
| le narrateur | 128 | t9: 1506 ± 95 | t1144: 1582 ± 77 | t9: 1506 ± 95 | t1144: 1582 ± 77 |
| Elstir | 24 | t451: 1616 ± 116 | t1147: 1611 ± 121 | t1110: 1610 ± 119 | t556: 1617 ± 110 |
| duchesse de Guermantes | 134 | t75: 1565 ± 106 | t1156: 1567 ± 78 | t75: 1565 ± 106 | t687: 1594 ± 58 |
| le grand-père du narrateur | 8 | t1: 1734 ± 211 | t1143: 1719 ± 232 | t744: 1718 ± 222 | t39: 1735 ± 210 |
| Gilberte | 36 | t44: 1528 ± 118 | t1156: 1577 ± 96 | t327: 1522 ± 96 | t1153: 1577 ± 96 |
| Andrée | 20 | t534: 1593 ± 132 | t1113: 1609 ± 130 | t534: 1593 ± 132 | t1113: 1609 ± 130 |
| Robert de Saint-Loup | 77 | t480: 1551 ± 76 | t1147: 1568 ± 92 | t678: 1541 ± 68 | t1147: 1568 ± 92 |
| la mère du narrateur | 45 | t1: 1579 ± 100 | t1098: 1576 ± 107 | t1093: 1576 ± 107 | t327: 1585 ± 88 |
| princesse de Guermantes | 14 | t561: 1637 ± 160 | t1153: 1620 ± 151 | t1147: 1620 ± 151 | t561: 1637 ± 160 |
| la grand-mère | 58 | t1: 1525 ± 105 | t1068: 1568 ± 102 | t1: 1525 ± 105 | t1068: 1568 ± 102 |
| Mme de Villeparisis | 55 | t7: 1560 ± 124 | t1093: 1573 ± 112 | t789: 1559 ± 87 | t1091: 1573 ± 112 |
| Octave | 2 | t17: 1868 ± 422 | t1086: 1879 ± 424 | t17: 1868 ± 422 | t1086: 1879 ± 424 |
| Françoise | 63 | t9: 1566 ± 102 | t1126: 1524 ± 89 | t721: 1524 ± 75 | t9: 1566 ± 102 |
| M. Vinteuil | 15 | t52: 1544 ± 143 | t1129: 1598 ± 174 | t52: 1544 ± 143 | t1129: 1598 ± 174 |
| Mme Verdurin | 70 | t78: 1473 ± 75 | t1141: 1518 ± 94 | t117: 1473 ± 71 | t1141: 1518 ± 94 |
| baron de Charlus | 78 | t267: 1556 ± 109 | t1154: 1488 ± 78 | t1152: 1488 ± 77 | t267: 1556 ± 109 |
| le peintre | 13 | t89: 1539 ± 133 | t226: 1543 ± 133 | t89: 1539 ± 133 | t177: 1544 ± 131 |
| princesse de Luxembourg | 4 | t469: 1623 ± 223 | t842: 1631 ± 223 | t469: 1623 ± 223 | t783: 1632 ± 221 |
| Odette | 104 | t44: 1505 ± 79 | t1155: 1505 ± 98 | t687: 1496 ± 76 | t297: 1516 ± 60 |
| le père du narrateur | 25 | t8: 1568 ± 114 | t1005: 1539 ± 133 | t1005: 1539 ± 133 | t8: 1568 ± 114 |
| princesse de Parme | 20 | t396: 1524 ± 126 | t924: 1508 ± 114 | t790: 1502 ± 106 | t396: 1524 ± 126 |
| le pianiste | 3 | t78: 1703 ± 310 | t112: 1703 ± 311 | t78: 1703 ± 310 | t112: 1703 ± 311 |
| Albertine | 93 | t514: 1503 ± 86 | t1133: 1461 ± 75 | t1042: 1460 ± 64 | t514: 1503 ± 86 |
| Morel | 22 | t945: 1495 ± 103 | t1123: 1490 ± 107 | t1112: 1489 ± 106 | t945: 1495 ± 103 |
| Legrandin | 19 | t23: 1515 ± 145 | t1137: 1529 ± 146 | t60: 1515 ± 143 | t744: 1531 ± 132 |
| duc de Guermantes | 67 | t612: 1459 ± 82 | t1155: 1456 ± 89 | t1111: 1455 ± 85 | t842: 1464 ± 66 |
| M. Verdurin | 20 | t82: 1500 ± 129 | t1118: 1524 ± 164 | t82: 1500 ± 129 | t1118: 1524 ± 164 |
| Norpois | 56 | t338: 1428 ± 85 | t1132: 1469 ± 110 | t355: 1428 ± 83 | t844: 1471 ± 92 |
| Bloch | 55 | t39: 1432 ± 116 | t1151: 1433 ± 96 | t483: 1414 ± 86 | t1143: 1433 ± 95 |
| jeune blonde de Rivebelle | 1 | t519: 1786 ± 460 | t519: 1786 ± 460 | t519: 1786 ± 460 | t519: 1786 ± 460 |
| Rémi | 3 | t151: 1698 ± 372 | t153: 1698 ± 372 | t151: 1698 ± 372 | t151: 1698 ± 372 |
| Bergotte | 28 | t41: 1468 ± 127 | t1132: 1446 ± 123 | t1131: 1446 ± 123 | t41: 1468 ± 127 |
| docteur Cottard | 44 | t78: 1460 ± 88 | t1118: 1443 ± 120 | t942: 1438 ± 107 | t338: 1468 ± 86 |
| marquis de Cambremer | 8 | t458: 1484 ± 171 | t1137: 1475 ± 160 | t945: 1469 ± 154 | t458: 1484 ± 171 |
| Swann | 183 | t3: 1443 ± 69 | t1144: 1402 ± 91 | t1143: 1402 ± 91 | t82: 1444 ± 59 |
| la Berma | 9 | t340: 1520 ± 195 | t1153: 1504 ± 193 | t1153: 1504 ± 193 | t344: 1520 ± 195 |
| M. de Marsantes | 11 | t628: 1466 ± 156 | t1126: 1480 ± 170 | t689: 1465 ± 153 | t1126: 1480 ± 170 |
| comte de Forcheville | 23 | t169: 1487 ± 104 | t1076: 1468 ± 159 | t1076: 1468 ± 159 | t218: 1488 ± 104 |
| Mme Bontemps | 14 | t397: 1424 ± 141 | t1112: 1424 ± 146 | t543: 1422 ± 139 | t982: 1427 ± 141 |
| Mlle de Stermaria | 2 | t458: 1580 ± 303 | t463: 1580 ± 303 | t458: 1580 ± 303 | t458: 1580 ± 303 |
| général de Froberville | 7 | t268: 1438 ± 155 | t895: 1437 ± 162 | t894: 1436 ± 162 | t268: 1438 ± 155 |
| duc de Châtellerault | 6 | t653: 1522 ± 227 | t1135: 1507 ± 236 | t1135: 1507 ± 236 | t653: 1522 ± 227 |
| Napoléon III | 1 | t605: 1623 ± 354 | t605: 1623 ± 354 | t605: 1623 ± 354 | t605: 1623 ± 354 |
| Mme Cottard | 12 | t179: 1412 ± 147 | t959: 1412 ± 156 | t959: 1412 ± 156 | t438: 1420 ± 138 |
| Mme Blandais | 2 | t467: 1662 ± 412 | t470: 1662 ± 412 | t467: 1662 ± 412 | t470: 1662 ± 412 |
| Mme de Cambremer | 18 | t270: 1413 ± 133 | t1133: 1384 ± 139 | t1133: 1384 ± 139 | t270: 1413 ± 133 |
| marquise de Saint-Euverte | 7 | t279: 1391 ± 184 | t916: 1410 ± 167 | t279: 1391 ± 184 | t893: 1410 ± 166 |
| marquise de Gallardon | 12 | t270: 1422 ± 169 | t909: 1410 ± 171 | t908: 1410 ± 171 | t275: 1423 ± 169 |
| le directeur | 16 | t453: 1396 ± 143 | t1115: 1388 ± 150 | t1115: 1388 ± 150 | t463: 1396 ± 143 |
| marquis de Bréauté | 7 | t268: 1397 ± 177 | t1152: 1394 ± 164 | t784: 1387 ± 156 | t268: 1397 ± 177 |
| Dreyfus | 3 | t677: 1460 ± 236 | t679: 1460 ± 236 | t677: 1460 ± 236 | t677: 1460 ± 236 |
| Mme de Vaugoubert | 3 | t884: 1468 ± 264 | t1029: 1467 ± 265 | t1029: 1467 ± 265 | t884: 1468 ± 264 |
| M. de Stermaria | 4 | t458: 1409 ± 252 | t773: 1416 ± 258 | t458: 1409 ± 252 | t773: 1416 ± 258 |
| la reine de Naples | 1 | t1035: 1682 ± 525 | t1035: 1682 ± 525 | t1035: 1682 ± 525 | t1035: 1682 ± 525 |
| M. Ski | 2 | t943: 1506 ± 352 | t1138: 1505 ± 355 | t1138: 1505 ± 355 | t943: 1506 ± 352 |
| Mlle d'Éporcheville | 1 | t1074: 1552 ± 403 | t1074: 1552 ± 403 | t1074: 1552 ± 403 | t1074: 1552 ± 403 |
| prince des Laumes | 1 | t789: 1444 ± 318 | t789: 1444 ± 318 | t789: 1444 ± 318 | t789: 1444 ± 318 |
| Brichot | 13 | t171: 1275 ± 169 | t1120: 1261 ± 149 | t1120: 1261 ± 149 | t201: 1275 ± 168 |
| M. de Vaugoubert | 9 | t350: 1263 ± 197 | t1115: 1259 ± 181 | t885: 1246 ± 175 | t350: 1263 ± 197 |
| Remi | 1 | t159: 1480 ± 404 | t159: 1480 ± 404 | t159: 1480 ± 404 | t159: 1480 ± 404 |
| M. Grevy | 2 | t133: 1306 ± 284 | t134: 1306 ± 284 | t133: 1306 ± 284 | t133: 1306 ± 284 |
| Bloch père | 3 | t485: 1288 ± 279 | t507: 1287 ± 279 | t505: 1287 ± 279 | t485: 1288 ± 279 |
| Saniette | 10 | t169: 1164 ± 210 | t1036: 1155 ± 238 | t1036: 1155 ± 238 | t169: 1164 ± 210 |
| oncle Adolphe | 3 | t27: 1180 ± 432 | t30: 1180 ± 432 | t27: 1180 ± 432 | t27: 1180 ± 432 |
| M. de Chevregny | 1 | t963: 1201 ± 453 | t963: 1201 ± 453 | t963: 1201 ± 453 | t963: 1201 ± 453 |
| Mme de Chaussepierre | 1 | t893: 1163 ± 434 | t893: 1163 ± 434 | t893: 1163 ± 434 | t893: 1163 ± 434 |
