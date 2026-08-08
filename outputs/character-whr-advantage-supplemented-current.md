# Character Whole-History Rating

- Analysis version: `character_whr_advantage_v1`
- Lens: `advantage`
- Source review version: `corpus_sanity_review_v1`
- Mode: `both`
- Time axis: `cumulative_unit_index`
- Character count: `70`
- Match count: `1801`
- Time point count: `686`
- Node count: `1829`
- Draw rate: `0.157`
- Draw model: `half_win_half_loss`
- w2: `5.0` Elo² per unit of narrative time (selected by `sequential_one_step_ahead_log_loss` from `[5.0, 15.0, 35.0, 60.0]`)
- Epsilon: `0.25`
- Initial rating / RD: `1500.0` / `350.0`
- Provisional band threshold: `200.0` Elo
- Wall clock: smoothed `0.097`s, filtered `19.91`s (all w2 candidates `107.346`s)
- Convergence: smoothed `19` sweeps (converged: `True`), filtered `686` fits / `9139` sweeps, `0` of them unconverged
- Supplemented: `true` (runs: supplement-run-001, supplement-run-002, supplement-run-003, supplement-run-004, supplement-run-005, supplement-run-006, supplement-run-007, supplement-run-008, supplement-run-009, supplement-run-010, supplement-run-011, supplement-run-012, supplement-run-013, supplement-run-014, supplement-run-015, supplement-run-016, supplement-run-017, supplement-run-018, supplement-run-019, supplement-run-020, supplement-run-021, supplement-run-022, supplement-run-023, supplement-run-024, supplement-run-025, supplement-run-026, supplement-run-027, supplement-run-028, supplement-run-029)

Ratings are shown as `rating ± band`, where the band is `2*sigma` from the per-node posterior variance -- an approximate 95% interval, conditional on the other characters' trajectories. Ranked listings sort by the conservative rating `rating - band` (i.e. `rating - 2*sigma`), the same conservative convention the Glicko-2 surface uses, so the two are read the same way. A character is provisional when their band exceeds `200.0` Elo, which is Glicko-2's `RD > 100` said about the same quantity.

## Predictive Comparison

Sequential one-step-ahead prediction over every match in narrative order, each match predicted from prior information only. Lower is better for both columns.

| System | Log Loss | Brier | Matches | Basis |
| --- | --- | --- | --- | --- |
| `whr_filtered` | 0.717813 | 0.258194 | 1801 | filtered WHR at w2=5 Elo^2 per unit, previous node's rating |
| `whr_filtered_deflated` | 0.706347 | 0.254424 | 1801 | filtered WHR at w2=5, previous node's rating deflated by its posterior variance |
| `elo_sequential` | 0.671605 | 0.239456 | 1801 | sequential ELO, K=24, expected score from the pre-match ratings |
| `elo_unit_frozen` | 0.68739 | 0.246856 | 1801 | sequential ELO, K=24, expected score frozen at the unit boundary |
| `glicko2_chapter_period` | 0.727686 | 0.263596 | 1801 | Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary |

sequential one-step-ahead over all matches in narrative order; each match is predicted from prior information only, and draws are scored as half a win plus half a loss for every system. Systems freeze at different boundaries: filtered WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual match -- so elo_sequential alone can see the other pairings of the unit it is predicting, which are driven by the same net scores. elo_unit_frozen is the like-for-like row.

### w2 Selection

| w2 (Elo² per unit) | Log Loss | Brier | Filtered Seconds |
| --- | --- | --- | --- |
| 5.0 | 0.717813 | 0.258194 | 19.91 |
| 15.0 | 0.719009 | 0.258493 | 23.4 |
| 35.0 | 0.723165 | 0.259869 | 28.966 |
| 60.0 | 0.728502 | 0.261676 | 35.07 |

## Final Standings

Final smoothed rating at each character's last node, ordered by conservative rating.

| Character | Rating | Conservative | Band | Matches | W-L-D | Units | Nodes | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jupien | 1735 ± 162 | 1573.5 | 161.5 | 29 | 22-5-2 | 16 | 16 | +0.64 |
| le narrateur | 1578 ± 77 | 1500.3 | 77.4 | 281 | 135-96-50 | 128 | 128 | -0.036 |
| Elstir | 1615 ± 122 | 1493.4 | 121.6 | 52 | 30-13-9 | 30 | 24 | +1.217 |
| duchesse de Guermantes | 1565 ± 78 | 1487.0 | 77.9 | 244 | 141-71-32 | 163 | 134 | +0.33 |
| Andrée | 1604 ± 130 | 1474.6 | 129.6 | 36 | 20-12-4 | 22 | 20 | -0.02 |
| la mère du narrateur | 1576 ± 108 | 1468.5 | 107.5 | 92 | 47-28-17 | 54 | 45 | +0.301 |
| Robert de Saint-Loup | 1560 ± 92 | 1467.4 | 92.5 | 137 | 64-56-17 | 134 | 77 | +0.055 |
| Gilberte | 1562 ± 96 | 1466.9 | 95.6 | 84 | 43-33-8 | 56 | 36 | +0.022 |
| la grand-mère | 1560 ± 101 | 1458.6 | 101.4 | 112 | 42-41-29 | 71 | 58 | -0.103 |
| princesse de Guermantes | 1601 ± 150 | 1450.4 | 150.4 | 25 | 15-9-1 | 18 | 14 | +0.543 |
| Mme de Villeparisis | 1562 ± 112 | 1450.2 | 112.3 | 90 | 40-30-20 | 69 | 55 | -0.078 |
| M. Vinteuil | 1606 ± 175 | 1431.7 | 174.6 | 26 | 15-9-2 | 21 | 15 | +0.651 |
| Françoise | 1519 ± 89 | 1430.1 | 88.8 | 131 | 57-55-19 | 89 | 63 | -0.144 |
| Mme Verdurin | 1524 ± 94 | 1429.9 | 93.6 | 151 | 65-55-31 | 80 | 70 | -0.347 |
| Odette | 1521 ± 98 | 1422.7 | 98.0 | 205 | 99-73-33 | 123 | 104 | -0.155 |
| Morel | 1515 ± 107 | 1407.8 | 107.3 | 50 | 23-19-8 | 24 | 22 | -1.118 |
| le père du narrateur | 1527 ± 133 | 1393.9 | 132.8 | 48 | 22-18-8 | 28 | 25 | -0.037 |
| baron de Charlus | 1468 ± 78 | 1390.5 | 77.9 | 155 | 67-69-19 | 108 | 78 | -0.672 |
| Albertine | 1463 ± 75 | 1388.2 | 75.2 | 161 | 62-90-9 | 137 | 93 | -0.519 |
| le peintre | 1519 ± 132 | 1386.6 | 132.3 | 29 | 13-8-8 | 13 | 13 | +0.023 |
| Norpois | 1470 ± 110 | 1359.9 | 110.0 | 96 | 33-44-19 | 72 | 56 | -0.048 |
| M. Verdurin | 1513 ± 164 | 1349.0 | 164.0 | 32 | 14-11-7 | 22 | 20 | -0.814 |
| duc de Guermantes | 1438 ± 89 | 1348.3 | 89.3 | 138 | 43-73-22 | 80 | 67 | -0.754 |
| Legrandin | 1492 ± 146 | 1346.1 | 146.3 | 33 | 13-17-3 | 23 | 19 | -0.786 |
| Bloch | 1428 ± 96 | 1332.9 | 95.6 | 113 | 33-63-17 | 63 | 55 | -1.091 |
| princesse de Parme | 1449 ± 116 | 1332.3 | 116.4 | 45 | 14-24-7 | 23 | 20 | -0.125 |
| Bergotte | 1449 ± 123 | 1326.6 | 122.6 | 56 | 23-28-5 | 39 | 28 | +0.319 |
| M. de Marsantes | 1488 ± 169 | 1319.2 | 169.2 | 21 | 7-10-4 | 12 | 11 | +0.026 |
| docteur Cottard | 1439 ± 120 | 1318.8 | 119.8 | 89 | 27-35-27 | 54 | 44 | -0.597 |
| Mme Bontemps | 1456 ± 145 | 1310.8 | 144.9 | 29 | 10-14-5 | 14 | 14 | -1.005 |
| Mme Cottard | 1465 ± 154 | 1310.8 | 154.1 | 27 | 12-12-3 | 15 | 12 | -0.103 |
| la Berma | 1501 ± 193 | 1308.4 | 192.8 | 14 | 6-7-1 | 10 | 9 | +0.469 |
| marquis de Cambremer | 1454 ± 159 | 1295.0 | 159.4 | 21 | 7-9-5 | 9 | 8 | -0.88 |
| Swann | 1383 ± 91 | 1291.7 | 91.4 | 328 | 107-181-40 | 270 | 183 | -0.728 |
| Mme de Cambremer | 1420 ± 138 | 1282.4 | 137.6 | 37 | 14-17-6 | 23 | 18 | -1.001 |
| le directeur | 1426 ± 146 | 1280.7 | 145.6 | 32 | 8-16-8 | 18 | 16 | -0.821 |
| comte de Forcheville | 1428 ± 159 | 1268.5 | 159.4 | 49 | 20-19-10 | 25 | 23 | +0.134 |
| général de Froberville | 1410 ± 163 | 1247.5 | 162.6 | 22 | 6-9-7 | 8 | 7 | -0.867 |
| marquis de Bréauté | 1402 ± 162 | 1239.9 | 162.1 | 23 | 6-12-5 | 7 | 7 | -1.438 |
| marquise de Saint-Euverte | 1383 ± 168 | 1214.4 | 168.3 | 19 | 7-11-1 | 7 | 7 | -1.58 |
| marquise de Gallardon | 1338 ± 180 | 1157.2 | 180.4 | 20 | 5-14-1 | 12 | 12 | -1.604 |
| Brichot | 1271 ± 146 | 1125.0 | 146.2 | 34 | 4-22-8 | 13 | 13 | -0.971 |
| M. de Vaugoubert | 1253 ± 181 | 1071.6 | 181.2 | 23 | 4-18-1 | 10 | 9 | -0.897 |

## Provisional Characters

Characters whose band is still wider than the provisional threshold -- too little evidence for the rating to mean much.

| Character | Rating | Band | Matches | Units | Nodes | Last Time |
| --- | --- | --- | --- | --- | --- | --- |
| Octave | 1876 ± 425 | 424.7 | 4 | 2 | 2 | 1086 |
| Aimé | 1808 ± 204 | 204.2 | 23 | 14 | 13 | 1068 |
| Mme Blandais | 1786 ± 460 | 460.4 | 2 | 2 | 2 | 470 |
| jeune blonde de Rivebelle | 1785 ± 461 | 460.6 | 2 | 1 | 1 | 519 |
| le pianiste | 1699 ± 311 | 311.2 | 6 | 3 | 3 | 112 |
| Rémi | 1696 ± 373 | 372.9 | 4 | 3 | 3 | 153 |
| la reine de Naples | 1675 ± 528 | 528.2 | 1 | 1 | 1 | 1035 |
| le grand-père du narrateur | 1671 ± 225 | 224.8 | 16 | 8 | 8 | 1143 |
| Mlle d'Éporcheville | 1669 ± 412 | 411.6 | 2 | 1 | 1 | 1074 |
| princesse de Luxembourg | 1622 ± 224 | 223.5 | 10 | 4 | 4 | 842 |
| Napoléon III | 1620 ± 355 | 354.7 | 3 | 1 | 1 | 605 |
| duc de Châtellerault | 1537 ± 235 | 234.7 | 9 | 6 | 6 | 1135 |
| Mlle de Stermaria | 1518 ± 291 | 291.4 | 5 | 3 | 2 | 463 |
| M. Ski | 1501 ± 355 | 354.8 | 3 | 2 | 2 | 1138 |
| M. Nissim Bernard | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| marquis de Forestelle | 1500 ± 700 | 700.0 | 0 | 1 | 0 | None |
| prince des Laumes | 1496 ± 314 | 313.9 | 4 | 1 | 1 | 789 |
| Remi | 1478 ± 404 | 404.4 | 2 | 1 | 1 | 159 |
| Mme de Vaugoubert | 1459 ± 265 | 265.0 | 7 | 3 | 3 | 1029 |
| Dreyfus | 1455 ± 236 | 236.5 | 8 | 3 | 3 | 679 |
| M. de Stermaria | 1408 ± 257 | 257.0 | 8 | 6 | 4 | 773 |
| M. de Chevregny | 1338 ± 402 | 402.1 | 3 | 1 | 1 | 963 |
| M. Grevy | 1299 ± 284 | 283.8 | 6 | 2 | 2 | 134 |
| Bloch père | 1222 ± 300 | 300.0 | 7 | 4 | 3 | 507 |
| oncle Adolphe | 1180 ± 432 | 431.7 | 4 | 4 | 3 | 30 |
| Mme de Chaussepierre | 1151 ± 432 | 431.6 | 3 | 1 | 1 | 893 |
| Saniette | 1113 ± 250 | 249.6 | 21 | 12 | 10 | 1036 |

## Trajectory Summaries

First, last, lowest, and highest point of each character's SMOOTHED trajectory (`t<time>: rating ± band`, time being the cumulative unit index). The full point-by-point trajectories, smoothed and filtered, live in the JSON artifact.

| Character | Points | First | Last | Lowest | Highest |
| --- | --- | --- | --- | --- | --- |
| Aimé | 13 | t456: 1801 ± 205 | t1068: 1808 ± 204 | t456: 1801 ± 205 | t963: 1808 ± 201 |
| Jupien | 16 | t551: 1732 ± 166 | t1154: 1735 ± 162 | t551: 1732 ± 166 | t1154: 1735 ± 162 |
| le narrateur | 128 | t9: 1511 ± 95 | t1144: 1578 ± 77 | t9: 1511 ± 95 | t1144: 1578 ± 77 |
| Elstir | 24 | t451: 1624 ± 117 | t1147: 1615 ± 122 | t1110: 1615 ± 119 | t543: 1625 ± 110 |
| duchesse de Guermantes | 134 | t75: 1553 ± 106 | t1156: 1565 ± 78 | t75: 1553 ± 106 | t687: 1593 ± 58 |
| Andrée | 20 | t534: 1585 ± 132 | t1113: 1604 ± 130 | t534: 1585 ± 132 | t1113: 1604 ± 130 |
| la mère du narrateur | 45 | t1: 1579 ± 100 | t1098: 1576 ± 108 | t1098: 1576 ± 108 | t397: 1588 ± 86 |
| Robert de Saint-Loup | 77 | t480: 1544 ± 76 | t1147: 1560 ± 92 | t678: 1532 ± 68 | t1147: 1560 ± 92 |
| Gilberte | 36 | t44: 1524 ± 118 | t1156: 1562 ± 96 | t327: 1517 ± 96 | t1153: 1562 ± 95 |
| la grand-mère | 58 | t1: 1514 ± 105 | t1068: 1560 ± 101 | t8: 1514 ± 104 | t1068: 1560 ± 101 |
| Octave | 2 | t17: 1864 ± 423 | t1086: 1876 ± 425 | t17: 1864 ± 423 | t1086: 1876 ± 425 |
| princesse de Guermantes | 14 | t561: 1616 ± 159 | t1153: 1601 ± 150 | t1147: 1601 ± 150 | t561: 1616 ± 159 |
| Mme de Villeparisis | 55 | t7: 1543 ± 124 | t1093: 1562 ± 112 | t7: 1543 ± 124 | t1091: 1562 ± 112 |
| le grand-père du narrateur | 8 | t1: 1698 ± 203 | t1143: 1671 ± 225 | t1143: 1671 ± 225 | t39: 1698 ± 202 |
| M. Vinteuil | 15 | t52: 1553 ± 144 | t1129: 1606 ± 175 | t52: 1553 ± 144 | t1129: 1606 ± 175 |
| Françoise | 63 | t9: 1555 ± 102 | t1126: 1519 ± 89 | t721: 1517 ± 75 | t9: 1555 ± 102 |
| Mme Verdurin | 70 | t78: 1465 ± 75 | t1141: 1524 ± 94 | t125: 1465 ± 70 | t1141: 1524 ± 94 |
| Odette | 104 | t44: 1504 ± 79 | t1155: 1521 ± 98 | t44: 1504 ± 79 | t954: 1524 ± 87 |
| Morel | 22 | t945: 1521 ± 104 | t1123: 1515 ± 107 | t1112: 1515 ± 106 | t945: 1521 ± 104 |
| princesse de Luxembourg | 4 | t469: 1614 ± 224 | t842: 1622 ± 224 | t469: 1614 ± 224 | t783: 1622 ± 222 |
| le père du narrateur | 25 | t8: 1552 ± 114 | t1005: 1527 ± 133 | t1005: 1527 ± 133 | t52: 1552 ± 112 |
| baron de Charlus | 78 | t267: 1544 ± 109 | t1154: 1468 ± 78 | t1152: 1468 ± 78 | t267: 1544 ± 109 |
| Albertine | 93 | t514: 1512 ± 86 | t1133: 1463 ± 75 | t1039: 1462 ± 64 | t514: 1512 ± 86 |
| le pianiste | 3 | t78: 1698 ± 311 | t112: 1699 ± 311 | t78: 1698 ± 311 | t112: 1699 ± 311 |
| le peintre | 13 | t89: 1514 ± 132 | t226: 1519 ± 132 | t89: 1514 ± 132 | t178: 1520 ± 130 |
| Norpois | 56 | t338: 1435 ± 84 | t1132: 1470 ± 110 | t357: 1434 ± 83 | t844: 1473 ± 92 |
| M. Verdurin | 20 | t82: 1485 ± 129 | t1118: 1513 ± 164 | t82: 1485 ± 129 | t1118: 1513 ± 164 |
| duc de Guermantes | 67 | t612: 1443 ± 82 | t1155: 1438 ± 89 | t1134: 1438 ± 88 | t843: 1449 ± 66 |
| Legrandin | 19 | t23: 1483 ± 145 | t1137: 1492 ± 146 | t59: 1483 ± 143 | t744: 1493 ± 132 |
| Bloch | 55 | t39: 1425 ± 116 | t1151: 1428 ± 96 | t504: 1406 ± 84 | t1143: 1428 ± 95 |
| princesse de Parme | 20 | t396: 1465 ± 127 | t924: 1449 ± 116 | t789: 1441 ± 109 | t396: 1465 ± 127 |
| Bergotte | 28 | t41: 1473 ± 127 | t1132: 1449 ± 123 | t1129: 1449 ± 122 | t41: 1473 ± 127 |
| Mme Blandais | 2 | t467: 1786 ± 460 | t470: 1786 ± 460 | t467: 1786 ± 460 | t467: 1786 ± 460 |
| jeune blonde de Rivebelle | 1 | t519: 1785 ± 461 | t519: 1785 ± 461 | t519: 1785 ± 461 | t519: 1785 ± 461 |
| Rémi | 3 | t151: 1696 ± 373 | t153: 1696 ± 373 | t151: 1696 ± 373 | t151: 1696 ± 373 |
| M. de Marsantes | 11 | t628: 1475 ± 156 | t1126: 1488 ± 169 | t690: 1474 ± 152 | t1126: 1488 ± 169 |
| docteur Cottard | 44 | t78: 1447 ± 88 | t1118: 1439 ± 120 | t942: 1434 ± 107 | t383: 1455 ± 87 |
| Mme Bontemps | 14 | t397: 1446 ± 140 | t1112: 1456 ± 145 | t440: 1446 ± 138 | t982: 1457 ± 139 |
| Mme Cottard | 12 | t179: 1482 ± 145 | t959: 1465 ± 154 | t959: 1465 ± 154 | t179: 1482 ± 145 |
| la Berma | 9 | t340: 1519 ± 195 | t1153: 1501 ± 193 | t1153: 1501 ± 193 | t344: 1519 ± 194 |
| duc de Châtellerault | 6 | t653: 1547 ± 226 | t1135: 1537 ± 235 | t1135: 1537 ± 235 | t653: 1547 ± 226 |
| marquis de Cambremer | 8 | t458: 1471 ± 171 | t1137: 1454 ± 159 | t945: 1448 ± 153 | t458: 1471 ± 171 |
| Swann | 183 | t3: 1440 ± 69 | t1144: 1383 ± 91 | t1143: 1383 ± 91 | t3: 1440 ± 69 |
| Mme de Cambremer | 18 | t270: 1462 ± 131 | t1133: 1420 ± 138 | t1133: 1420 ± 138 | t270: 1462 ± 131 |
| le directeur | 16 | t453: 1431 ± 139 | t1115: 1426 ± 146 | t1115: 1426 ± 146 | t463: 1431 ± 138 |
| comte de Forcheville | 23 | t169: 1443 ± 104 | t1076: 1428 ± 159 | t1076: 1428 ± 159 | t218: 1445 ± 104 |
| Napoléon III | 1 | t605: 1620 ± 355 | t605: 1620 ± 355 | t605: 1620 ± 355 | t605: 1620 ± 355 |
| Mlle d'Éporcheville | 1 | t1074: 1669 ± 412 | t1074: 1669 ± 412 | t1074: 1669 ± 412 | t1074: 1669 ± 412 |
| général de Froberville | 7 | t268: 1399 ± 156 | t895: 1410 ± 163 | t283: 1399 ± 155 | t895: 1410 ± 163 |
| marquis de Bréauté | 7 | t268: 1394 ± 176 | t1152: 1402 ± 162 | t784: 1391 ± 155 | t1152: 1402 ± 162 |
| Mlle de Stermaria | 2 | t458: 1518 ± 291 | t463: 1518 ± 291 | t458: 1518 ± 291 | t458: 1518 ± 291 |
| Dreyfus | 3 | t677: 1455 ± 236 | t679: 1455 ± 236 | t677: 1455 ± 236 | t677: 1455 ± 236 |
| marquise de Saint-Euverte | 7 | t279: 1366 ± 185 | t916: 1383 ± 168 | t279: 1366 ± 185 | t893: 1383 ± 168 |
| Mme de Vaugoubert | 3 | t884: 1460 ± 264 | t1029: 1459 ± 265 | t1029: 1459 ± 265 | t884: 1460 ± 264 |
| prince des Laumes | 1 | t789: 1496 ± 314 | t789: 1496 ± 314 | t789: 1496 ± 314 | t789: 1496 ± 314 |
| marquise de Gallardon | 12 | t270: 1362 ± 177 | t909: 1338 ± 180 | t908: 1338 ± 180 | t274: 1362 ± 177 |
| M. de Stermaria | 4 | t458: 1401 ± 251 | t773: 1408 ± 257 | t458: 1401 ± 251 | t773: 1408 ± 257 |
| la reine de Naples | 1 | t1035: 1675 ± 528 | t1035: 1675 ± 528 | t1035: 1675 ± 528 | t1035: 1675 ± 528 |
| M. Ski | 2 | t943: 1501 ± 352 | t1138: 1501 ± 355 | t1138: 1501 ± 355 | t943: 1501 ± 352 |
| Brichot | 13 | t171: 1282 ± 167 | t1120: 1271 ± 146 | t1120: 1271 ± 146 | t201: 1282 ± 166 |
| Remi | 1 | t159: 1478 ± 404 | t159: 1478 ± 404 | t159: 1478 ± 404 | t159: 1478 ± 404 |
| M. de Vaugoubert | 9 | t350: 1257 ± 197 | t1115: 1253 ± 181 | t889: 1240 ± 175 | t350: 1257 ± 197 |
| M. Grevy | 2 | t133: 1299 ± 284 | t134: 1299 ± 284 | t133: 1299 ± 284 | t133: 1299 ± 284 |
| M. de Chevregny | 1 | t963: 1338 ± 402 | t963: 1338 ± 402 | t963: 1338 ± 402 | t963: 1338 ± 402 |
| Bloch père | 3 | t485: 1222 ± 300 | t507: 1222 ± 300 | t505: 1222 ± 300 | t485: 1222 ± 300 |
| Saniette | 10 | t169: 1119 ± 223 | t1036: 1113 ± 250 | t1036: 1113 ± 250 | t169: 1119 ± 223 |
| oncle Adolphe | 3 | t27: 1180 ± 432 | t30: 1180 ± 432 | t30: 1180 ± 432 | t27: 1180 ± 432 |
| Mme de Chaussepierre | 1 | t893: 1151 ± 432 | t893: 1151 ± 432 | t893: 1151 ± 432 | t893: 1151 ± 432 |
