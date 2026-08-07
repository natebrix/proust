# Character Glicko-2

- Analysis version: `character_glicko2_advantage_v1`
- Lens: `advantage`
- Source review version: `corpus_sanity_review_v1`
- Character count: `70`
- Match count: `1801`
- Draw rate: `0.157`
- Initial rating: `1500.0`
- Initial RD: `350.0`
- Initial volatility: `0.06`
- Tau: `0.5`
- Epsilon: `0.25`
- Rating period rule: `canonical_chapter`
- Period count: `18`
- Provisional RD threshold: `100.0`
- Supplemented: `true` (runs: supplement-run-001, supplement-run-002, supplement-run-003, supplement-run-004, supplement-run-005, supplement-run-006, supplement-run-007, supplement-run-008, supplement-run-009, supplement-run-010, supplement-run-011, supplement-run-012, supplement-run-013, supplement-run-014, supplement-run-015, supplement-run-016, supplement-run-017, supplement-run-018, supplement-run-019, supplement-run-020, supplement-run-021, supplement-run-022, supplement-run-023, supplement-run-024, supplement-run-025, supplement-run-026, supplement-run-027, supplement-run-028, supplement-run-029)

Ratings are shown as `rating ± 2*RD`, an approximate 95% confidence band. Characters whose RD exceeds the provisional threshold are excluded from the top/bottom/divergence tables below (but never from the full character table), since a rating built on very little evidence is noise.

## Top Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aimé | 1794 ± 194 | 1599.4 | 97.1 | 0.06 | 23 | 19-2-2 | 14 | +0.497 |
| Jupien | 1749 ± 153 | 1595.8 | 76.6 | 0.06 | 29 | 22-5-2 | 16 | +0.64 |
| duchesse de Guermantes | 1559 ± 75 | 1483.4 | 37.7 | 0.0599 | 244 | 141-71-32 | 163 | +0.33 |
| Elstir | 1598 ± 123 | 1474.9 | 61.4 | 0.06 | 52 | 30-13-9 | 30 | +1.217 |
| le narrateur | 1552 ± 77 | 1474.9 | 38.6 | 0.0599 | 281 | 135-96-50 | 128 | -0.036 |
| la mère du narrateur | 1559 ± 109 | 1449.9 | 54.5 | 0.06 | 92 | 47-28-17 | 54 | +0.301 |
| Robert de Saint-Loup | 1535 ± 89 | 1445.8 | 44.5 | 0.0603 | 137 | 64-56-17 | 134 | +0.055 |
| Andrée | 1579 ± 139 | 1439.6 | 69.6 | 0.06 | 36 | 20-12-4 | 22 | -0.02 |
| Gilberte | 1534 ± 94 | 1439.3 | 47.1 | 0.0602 | 84 | 43-33-8 | 56 | +0.022 |
| princesse de Guermantes | 1596 ± 168 | 1427.6 | 84.2 | 0.06 | 25 | 15-9-1 | 18 | +0.543 |

## Bottom Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M. de Vaugoubert | 1246 ± 198 | 1047.4 | 99.2 | 0.0601 | 23 | 4-18-1 | 10 | -0.897 |
| Brichot | 1250 ± 155 | 1095.8 | 77.4 | 0.06 | 34 | 4-22-8 | 13 | -0.971 |
| général de Froberville | 1377 ± 199 | 1177.7 | 99.6 | 0.06 | 22 | 6-9-7 | 8 | -0.867 |
| marquis de Bréauté | 1396 ± 173 | 1222.1 | 86.7 | 0.06 | 23 | 6-12-5 | 7 | -1.438 |
| Mme de Cambremer | 1375 ± 148 | 1227.5 | 73.8 | 0.06 | 37 | 14-17-6 | 23 | -1.001 |
| le directeur | 1400 ± 152 | 1248.4 | 76.0 | 0.06 | 32 | 8-16-8 | 18 | -0.821 |
| princesse de Parme | 1402 ± 134 | 1267.5 | 67.1 | 0.0602 | 45 | 14-24-7 | 23 | -0.125 |
| Swann | 1358 ± 83 | 1274.8 | 41.4 | 0.0603 | 328 | 107-181-40 | 270 | -0.728 |
| Mme Cottard | 1448 ± 166 | 1282.1 | 83.1 | 0.06 | 27 | 12-12-3 | 15 | -0.103 |
| comte de Forcheville | 1436 ± 153 | 1283.6 | 76.4 | 0.06 | 49 | 20-19-10 | 25 | +0.134 |

## Provisional Characters

Characters whose RD is still above the provisional threshold -- their rating should be treated as unstable.

| Character | Rating | RD | Matches | Units | Last Period |
| --- | --- | --- | --- | --- | --- |
| Octave | 1823 ± 361 | 180.4 | 4 | 2 | v6-p2 |
| jeune blonde de Rivebelle | 1760 ± 456 | 228.1 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| la reine de Naples | 1721 ± 506 | 252.8 | 1 | 1 | v5 |
| le pianiste | 1716 ± 347 | 173.7 | 6 | 3 | v1-p2-un-amour-de-swann |
| Mme Blandais | 1690 ± 470 | 235.1 | 2 | 2 | v2-p2-noms-de-pays-le-pays |
| le grand-père du narrateur | 1664 ± 256 | 128.2 | 16 | 8 | v7-p4-le-bal-de-tetes |
| Mlle d'Éporcheville | 1659 ± 412 | 206.0 | 2 | 1 | v6-p2 |
| Rémi | 1645 ± 371 | 185.4 | 4 | 3 | v1-p2-un-amour-de-swann |
| princesse de Luxembourg | 1615 ± 251 | 125.6 | 10 | 4 | v3-p2 |
| Mlle de Stermaria | 1604 ± 368 | 183.8 | 5 | 3 | v3-p2 |
| Napoléon III | 1592 ± 361 | 180.4 | 3 | 1 | v3-p1 |
| Mme de Vaugoubert | 1551 ± 300 | 150.2 | 7 | 3 | v5 |
| marquise de Saint-Euverte | 1505 ± 243 | 121.4 | 19 | 7 | v4-p2 |
| M. Nissim Bernard | 1500 ± 703 | 351.5 | 0 | 1 | v4-p2 |
| marquis de Forestelle | 1500 ± 705 | 352.6 | 0 | 1 | v1-p2-un-amour-de-swann |
| duc de Châtellerault | 1499 ± 246 | 122.8 | 9 | 6 | v7-p4-le-bal-de-tetes |
| la Berma | 1496 ± 211 | 105.5 | 14 | 10 | v7-p4-le-bal-de-tetes |
| prince des Laumes | 1478 ± 337 | 168.3 | 4 | 1 | v3-p2 |
| M. Ski | 1450 ± 363 | 181.5 | 3 | 2 | v7-p4-le-bal-de-tetes |
| M. de Stermaria | 1436 ± 303 | 151.4 | 8 | 6 | v3-p2 |
| Remi | 1412 ± 466 | 233.2 | 2 | 1 | v1-p2-un-amour-de-swann |
| Dreyfus | 1386 ± 272 | 136.1 | 8 | 3 | v3-p1 |
| marquis de Cambremer | 1372 ± 209 | 104.4 | 21 | 9 | v7-p4-le-bal-de-tetes |
| M. de Chevregny | 1330 ± 386 | 193.2 | 3 | 1 | v4-p2 |
| marquise de Gallardon | 1262 ± 203 | 101.6 | 20 | 12 | v4-p2 |
| M. Grevy | 1218 ± 350 | 175.0 | 6 | 2 | v1-p2-un-amour-de-swann |
| Bloch père | 1191 ± 296 | 148.1 | 7 | 4 | v2-p2-noms-de-pays-le-pays |
| oncle Adolphe | 1165 ± 426 | 212.9 | 4 | 4 | v1-p1-combray |
| Mme de Chaussepierre | 1117 ± 396 | 197.9 | 3 | 1 | v4-p2 |
| Saniette | 1082 ± 213 | 106.6 | 21 | 12 | v5 |

## Largest Glicko-vs-ELO Rank Divergences

| Character | Glicko Rank | ELO Rank | Delta | Rating | ELO |
| --- | --- | --- | --- | --- | --- |
| baron de Charlus | 20 | 68 | -48 | 1464 ± 82 | 1367.676 |
| duc de Guermantes | 26 | 64 | -38 | 1421 ± 88 | 1426.987 |
| Albertine | 24 | 60 | -36 | 1434 ± 83 | 1448.802 |
| Françoise | 18 | 53 | -35 | 1480 ± 92 | 1466.009 |
| Swann | 39 | 67 | -28 | 1358 ± 83 | 1379.419 |
| Bergotte | 31 | 57 | -26 | 1422 ± 120 | 1462.187 |
| Bloch | 34 | 58 | -24 | 1397 ± 96 | 1456.97 |
| docteur Cottard | 32 | 55 | -23 | 1415 ± 114 | 1463.35 |
| Mme Bontemps | 29 | 47 | -18 | 1457 ± 151 | 1479.422 |
| Legrandin | 27 | 44 | -17 | 1464 ± 152 | 1484.987 |

## Character Table

| Character | Rating | Conservative | RD | Volatility | Glicko Rank | ELO Rank | Provisional | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aimé | 1794 ± 194 | 1599.4 | 97.1 | 0.06 | 1 | 1 | False | 23 | 19-2-2 | 14 | +0.497 |
| Jupien | 1749 ± 153 | 1595.8 | 76.6 | 0.06 | 2 | 2 | False | 29 | 22-5-2 | 16 | +0.64 |
| duchesse de Guermantes | 1559 ± 75 | 1483.4 | 37.7 | 0.0599 | 3 | 14 | False | 244 | 141-71-32 | 163 | +0.33 |
| Elstir | 1598 ± 123 | 1474.9 | 61.4 | 0.06 | 4 | 4 | False | 52 | 30-13-9 | 30 | +1.217 |
| le narrateur | 1552 ± 77 | 1474.9 | 38.6 | 0.0599 | 5 | 3 | False | 281 | 135-96-50 | 128 | -0.036 |
| Octave | 1823 ± 361 | 1462.4 | 180.4 | 0.06 | 6 | 15 | True | 4 | 4-0-0 | 2 | +1.025 |
| la mère du narrateur | 1559 ± 109 | 1449.9 | 54.5 | 0.06 | 7 | 13 | False | 92 | 47-28-17 | 54 | +0.301 |
| Robert de Saint-Loup | 1535 ± 89 | 1445.8 | 44.5 | 0.0603 | 8 | 7 | False | 137 | 64-56-17 | 134 | +0.055 |
| Andrée | 1579 ± 139 | 1439.6 | 69.6 | 0.06 | 9 | 9 | False | 36 | 20-12-4 | 22 | -0.02 |
| Gilberte | 1534 ± 94 | 1439.3 | 47.1 | 0.0602 | 10 | 5 | False | 84 | 43-33-8 | 56 | +0.022 |
| princesse de Guermantes | 1596 ± 168 | 1427.6 | 84.2 | 0.06 | 11 | 11 | False | 25 | 15-9-1 | 18 | +0.543 |
| la grand-mère | 1517 ± 101 | 1416.1 | 50.6 | 0.06 | 12 | 8 | False | 112 | 42-41-29 | 71 | -0.103 |
| M. Vinteuil | 1591 ± 179 | 1412.0 | 89.3 | 0.06 | 13 | 10 | False | 26 | 15-9-2 | 21 | +0.651 |
| le grand-père du narrateur | 1664 ± 256 | 1407.7 | 128.2 | 0.06 | 14 | 6 | True | 16 | 11-2-3 | 8 | +0.385 |
| Mme Verdurin | 1495 ± 91 | 1404.0 | 45.7 | 0.0601 | 15 | 12 | False | 151 | 65-55-31 | 80 | -0.347 |
| Morel | 1516 ± 116 | 1399.5 | 58.1 | 0.06 | 16 | 30 | False | 50 | 23-19-8 | 24 | -1.118 |
| Mme de Villeparisis | 1508 ± 109 | 1399.1 | 54.3 | 0.06 | 17 | 18 | False | 90 | 40-30-20 | 69 | -0.078 |
| Françoise | 1480 ± 92 | 1387.7 | 46.0 | 0.0604 | 18 | 53 | False | 131 | 57-55-19 | 89 | -0.144 |
| Odette | 1473 ± 86 | 1387.4 | 42.9 | 0.06 | 19 | 32 | False | 205 | 99-73-33 | 123 | -0.155 |
| baron de Charlus | 1464 ± 82 | 1382.5 | 40.8 | 0.0604 | 20 | 68 | False | 155 | 67-69-19 | 108 | -0.672 |
| le pianiste | 1716 ± 347 | 1368.1 | 173.7 | 0.06 | 21 | 16 | True | 6 | 4-0-2 | 3 | +0.863 |
| le père du narrateur | 1510 ± 144 | 1365.7 | 72.2 | 0.06 | 22 | 26 | False | 48 | 22-18-8 | 28 | -0.037 |
| princesse de Luxembourg | 1615 ± 251 | 1363.6 | 125.6 | 0.06 | 23 | 17 | True | 10 | 5-2-3 | 4 | -0.795 |
| Albertine | 1434 ± 83 | 1350.5 | 41.5 | 0.06 | 24 | 60 | False | 161 | 62-90-9 | 137 | -0.519 |
| Norpois | 1437 ± 101 | 1335.4 | 50.7 | 0.0601 | 25 | 39 | False | 96 | 33-44-19 | 72 | -0.048 |
| duc de Guermantes | 1421 ± 88 | 1332.9 | 44.2 | 0.06 | 26 | 64 | False | 138 | 43-73-22 | 80 | -0.754 |
| Legrandin | 1464 ± 152 | 1311.7 | 75.9 | 0.06 | 27 | 44 | False | 33 | 13-17-3 | 23 | -0.786 |
| M. Verdurin | 1475 ± 168 | 1306.8 | 84.2 | 0.06 | 28 | 19 | False | 32 | 14-11-7 | 22 | -0.814 |
| Mme Bontemps | 1457 ± 151 | 1305.8 | 75.5 | 0.06 | 29 | 47 | False | 29 | 10-14-5 | 14 | -1.005 |
| jeune blonde de Rivebelle | 1760 ± 456 | 1303.3 | 228.1 | 0.06 | 30 | 22 | True | 2 | 2-0-0 | 1 | +1.614 |
| Bergotte | 1422 ± 120 | 1302.1 | 60.1 | 0.06 | 31 | 57 | False | 56 | 23-28-5 | 39 | +0.319 |
| docteur Cottard | 1415 ± 114 | 1301.5 | 57.0 | 0.06 | 32 | 55 | False | 89 | 27-35-27 | 54 | -0.597 |
| le peintre | 1493 ± 191 | 1301.4 | 95.6 | 0.06 | 33 | 21 | False | 29 | 13-8-8 | 13 | +0.023 |
| Bloch | 1397 ± 96 | 1300.9 | 48.2 | 0.06 | 34 | 58 | False | 113 | 33-63-17 | 63 | -1.091 |
| M. de Marsantes | 1461 ± 169 | 1292.5 | 84.3 | 0.06 | 35 | 45 | False | 21 | 7-10-4 | 12 | +0.026 |
| la Berma | 1496 ± 211 | 1284.7 | 105.5 | 0.06 | 36 | 31 | True | 14 | 6-7-1 | 10 | +0.469 |
| comte de Forcheville | 1436 ± 153 | 1283.6 | 76.4 | 0.06 | 37 | 46 | False | 49 | 20-19-10 | 25 | +0.134 |
| Mme Cottard | 1448 ± 166 | 1282.1 | 83.1 | 0.06 | 38 | 42 | False | 27 | 12-12-3 | 15 | -0.103 |
| Swann | 1358 ± 83 | 1274.8 | 41.4 | 0.0603 | 39 | 67 | False | 328 | 107-181-40 | 270 | -0.728 |
| Rémi | 1645 ± 371 | 1274.4 | 185.4 | 0.06 | 40 | 20 | True | 4 | 3-0-1 | 3 | +0.693 |
| princesse de Parme | 1402 ± 134 | 1267.5 | 67.1 | 0.0602 | 41 | 41 | False | 45 | 14-24-7 | 23 | -0.125 |
| marquise de Saint-Euverte | 1505 ± 243 | 1262.6 | 121.4 | 0.06 | 42 | 50 | True | 19 | 7-11-1 | 7 | -1.58 |
| duc de Châtellerault | 1499 ± 246 | 1253.1 | 122.8 | 0.06 | 43 | 36 | True | 9 | 4-4-1 | 6 | -0.816 |
| Mme de Vaugoubert | 1551 ± 300 | 1250.8 | 150.2 | 0.06 | 44 | 29 | True | 7 | 3-2-2 | 3 | -0.165 |
| le directeur | 1400 ± 152 | 1248.4 | 76.0 | 0.06 | 45 | 62 | False | 32 | 8-16-8 | 18 | -0.821 |
| Mlle d'Éporcheville | 1659 ± 412 | 1247.1 | 206.0 | 0.06 | 46 | 24 | True | 2 | 1-0-1 | 1 | +1.46 |
| Mlle de Stermaria | 1604 ± 368 | 1236.0 | 183.8 | 0.06 | 47 | 28 | True | 5 | 3-2-0 | 3 | +0.112 |
| Napoléon III | 1592 ± 361 | 1231.6 | 180.4 | 0.06 | 48 | 25 | True | 3 | 1-0-2 | 1 | 0.0 |
| Mme de Cambremer | 1375 ± 148 | 1227.5 | 73.8 | 0.06 | 49 | 63 | False | 37 | 14-17-6 | 23 | -1.001 |
| marquis de Bréauté | 1396 ± 173 | 1222.1 | 86.7 | 0.06 | 50 | 52 | False | 23 | 6-12-5 | 7 | -1.438 |
| Mme Blandais | 1690 ± 470 | 1219.8 | 235.1 | 0.06 | 51 | 23 | True | 2 | 2-0-0 | 2 | -0.955 |
| la reine de Naples | 1721 ± 506 | 1215.4 | 252.8 | 0.06 | 52 | 27 | True | 1 | 1-0-0 | 1 | 0.0 |
| général de Froberville | 1377 ± 199 | 1177.7 | 99.6 | 0.06 | 53 | 49 | False | 22 | 6-9-7 | 8 | -0.867 |
| marquis de Cambremer | 1372 ± 209 | 1163.7 | 104.4 | 0.0602 | 54 | 40 | True | 21 | 7-9-5 | 9 | -0.88 |
| prince des Laumes | 1478 ± 337 | 1142.0 | 168.3 | 0.06 | 55 | 38 | True | 4 | 2-2-0 | 1 | -1.33 |
| M. de Stermaria | 1436 ± 303 | 1133.6 | 151.4 | 0.06 | 56 | 51 | True | 8 | 2-5-1 | 6 | -0.284 |
| Dreyfus | 1386 ± 272 | 1113.8 | 136.1 | 0.06 | 57 | 43 | True | 8 | 2-3-3 | 3 | -0.658 |
| Brichot | 1250 ± 155 | 1095.8 | 77.4 | 0.06 | 58 | 69 | False | 34 | 4-22-8 | 13 | -0.971 |
| M. Ski | 1450 ± 363 | 1087.6 | 181.5 | 0.06 | 59 | 33 | True | 3 | 1-1-1 | 2 | -1.218 |
| marquise de Gallardon | 1262 ± 203 | 1058.6 | 101.6 | 0.06 | 60 | 65 | True | 20 | 5-14-1 | 12 | -1.604 |
| M. de Vaugoubert | 1246 ± 198 | 1047.4 | 99.2 | 0.0601 | 61 | 66 | False | 23 | 4-18-1 | 10 | -0.897 |
| Remi | 1412 ± 466 | 945.3 | 233.2 | 0.06 | 62 | 37 | True | 2 | 1-1-0 | 1 | +0.616 |
| M. de Chevregny | 1330 ± 386 | 944.0 | 193.2 | 0.06 | 63 | 48 | True | 3 | 0-2-1 | 1 | -1.8 |
| Bloch père | 1191 ± 296 | 895.1 | 148.1 | 0.06 | 64 | 61 | True | 7 | 0-5-2 | 4 | -1.543 |
| Saniette | 1082 ± 213 | 868.7 | 106.6 | 0.06 | 65 | 70 | True | 21 | 2-19-0 | 12 | -2.128 |
| M. Grevy | 1218 ± 350 | 867.8 | 175.0 | 0.06 | 66 | 56 | True | 6 | 1-4-1 | 2 | -0.873 |
| M. Nissim Bernard | 1500 ± 703 | 796.9 | 351.5 | 0.06 | 67 | 34 | True | 0 | 0-0-0 | 1 | +3.096 |
| marquis de Forestelle | 1500 ± 705 | 794.7 | 352.6 | 0.06 | 68 | 35 | True | 0 | 0-0-0 | 1 | +1.72 |
| oncle Adolphe | 1165 ± 426 | 739.1 | 212.9 | 0.06 | 69 | 59 | True | 4 | 0-4-0 | 4 | -1.861 |
| Mme de Chaussepierre | 1117 ± 396 | 721.6 | 197.9 | 0.06 | 70 | 54 | True | 3 | 0-3-0 | 1 | -2.867 |
