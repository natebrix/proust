# Character Standings — advantage (scoring v2)

- Standings version: `character_standings_advantage_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_advantage_name_view_v1` (`outputs/scoring-v2/scoring-v2-advantage-name-view-ratings.json`)
- Lens / view: `advantage` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`41` ranked, `152` without sufficient evidence)
- Comparisons: `2708` (mean weight `0.6453`, draw rate `0.106`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-advantage-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `41` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | comte de Forcheville | 1712 ± 163 | 1549.4 | 62 | 44-14-4 | 28 | +0.139 | 0.1814 |
| 2 | duchesse de Guermantes | 1594 ± 80 | 1514.1 | 354 | 225-92-37 | 183 | +0.049 | 0.4899 |
| 3 | M. Verdurin | 1622 ± 124 | 1498.4 | 84 | 47-26-11 | 32 | -0.176 | 0.2672 |
| 4 | la mère du narrateur | 1620 ± 146 | 1473.8 | 54 | 34-16-4 | 28 | +0.273 | 0.3177 |
| 5 | docteur Cottard | 1582 ± 119 | 1463.9 | 107 | 50-43-14 | 37 | -0.165 | 0.7129 |
| 6 | Mme de Villeparisis | 1580 ± 118 | 1461.3 | 118 | 72-38-8 | 73 | -0.077 | 0.3693 |
| 7 | Françoise | 1575 ± 114 | 1461.2 | 100 | 51-38-11 | 61 | +0.086 | 0.5864 |
| 8 | la grand-mère | 1578 ± 129 | 1448.7 | 74 | 37-31-6 | 48 | +0.177 | 0.6675 |
| 9 | Rachel | 1579 ± 132 | 1446.3 | 56 | 27-20-9 | 29 | -0.216 | 0.6092 |
| 10 | baron de Charlus | 1514 ± 74 | 1439.9 | 283 | 133-115-35 | 110 | -0.256 | 0.7058 |
| 10 | le narrateur | 1513 ± 73 | 1439.9 | 399 | 168-200-31 | 209 | -0.201 | 0.6321 |
| 11 | Andrée | 1578 ± 138 | 1439.7 | 54 | 30-19-5 | 25 | -0.061 | 0.659 |
| 12 | Mme Verdurin | 1535 ± 96 | 1439.3 | 181 | 78-83-20 | 78 | -0.336 | 0.4254 |
| 13 | Albertine | 1509 ± 79 | 1430.1 | 183 | 80-84-19 | 126 | -0.203 | 0.7437 |
| 14 | Robert de Saint-Loup | 1514 ± 84 | 1429.8 | 234 | 105-108-21 | 138 | -0.132 | 0.6397 |
| 15 | le père du narrateur | 1590 ± 166 | 1424.1 | 44 | 25-16-3 | 21 | +0.078 | 0.2873 |
| 16 | le grand-père du narrateur | 1606 ± 186 | 1419.5 | 25 | 16-9-0 | 11 | +0.004 | 0.182 |
| 17 | Gilberte | 1505 ± 94 | 1410.7 | 139 | 64-61-14 | 57 | -0.028 | 0.4766 |
| 18 | Brichot | 1535 ± 127 | 1408.2 | 62 | 27-24-11 | 17 | -0.262 | 0.6579 |
| 19 | Bergotte | 1562 ± 157 | 1405.2 | 47 | 25-21-1 | 27 | +0.127 | 0.8517 |
| 20 | Odette | 1493 ± 98 | 1394.7 | 248 | 112-107-29 | 124 | -0.081 | 0.5035 |
| 21 | Morel | 1488 ± 99 | 1388.3 | 101 | 41-50-10 | 35 | -0.718 | 0.8773 |
| 22 | Mme Cottard | 1551 ± 165 | 1386.1 | 32 | 15-14-3 | 15 | -0.093 | 0.2876 |
| 23 | princesse de Parme | 1506 ± 120 | 1386.0 | 82 | 37-37-8 | 36 | -0.121 | 0.2381 |
| 24 | Norpois | 1503 ± 121 | 1381.9 | 101 | 45-44-12 | 54 | -0.069 | 0.4894 |
| 25 | la Berma | 1560 ± 182 | 1378.6 | 26 | 11-11-4 | 13 | +0.351 | 1.2738 |
| 26 | Swann | 1458 ± 88 | 1369.2 | 386 | 144-197-45 | 177 | -0.317 | 0.7741 |
| 27 | prince de Guermantes | 1497 ± 138 | 1358.5 | 43 | 19-21-3 | 13 | -0.449 | 0.9777 |
| 28 | Mme Bontemps | 1547 ± 189 | 1358.1 | 24 | 12-10-2 | 13 | -0.295 | 0.2954 |
| 29 | Mme de Marsantes | 1489 ± 137 | 1352.1 | 47 | 20-25-2 | 21 | -0.449 | 0.5832 |
| 30 | princesse de Guermantes | 1486 ± 138 | 1348.9 | 53 | 21-22-10 | 19 | -0.076 | 0.4858 |
| 31 | duc de Guermantes | 1435 ± 87 | 1347.7 | 234 | 75-126-33 | 97 | -0.507 | 0.5645 |
| 32 | Elstir | 1506 ± 161 | 1345.0 | 41 | 15-16-10 | 18 | +0.332 | 0.7734 |
| 33 | marquis de Bréauté | 1480 ± 136 | 1344.6 | 56 | 25-25-6 | 17 | -0.244 | 0.5146 |
| 34 | Mme de Cambremer | 1368 ± 140 | 1228.0 | 65 | 15-41-9 | 22 | -0.831 | 0.8309 |
| 35 | comtesse Molé | 1422 ± 198 | 1224.1 | 22 | 8-12-2 | 6 | -0.363 | 0.3633 |
| 36 | Bloch | 1306 ± 110 | 1196.1 | 146 | 31-100-15 | 64 | -0.692 | 0.8975 |
| 37 | Mme d'Arpajon | 1347 ± 172 | 1174.3 | 34 | 8-19-7 | 10 | -0.791 | 0.791 |
| 38 | marquise de Gallardon | 1322 ± 190 | 1132.6 | 29 | 5-20-4 | 10 | -0.751 | 0.7514 |
| 39 | Legrandin | 1294 ± 166 | 1128.5 | 40 | 8-28-4 | 23 | -0.547 | 0.7439 |
| 40 | Saniette | 1256 ± 173 | 1082.9 | 44 | 6-34-4 | 12 | -0.846 | 1.0578 |

## Insufficient comparative evidence

The `152` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| le jeune marquis de Cambremer | 1980 ± 398 | 398.1 | 10 | 1 | +0.75 | 0.75 |
| Aimé | 1939 ± 294 | 293.5 | 25 | 9 | +0.268 | 0.2678 |
| Mlle d'Oloron | 1842 ± 422 | 421.7 | 7 | 2 | 0.0 | 0.0 |
| M. de Crécy | 1841 ± 428 | 428.0 | 6 | 1 | +0.6 | 0.6 |
| grand-duc héritier de Luxembourg | 1816 ± 442 | 442.5 | 5 | 2 | +0.85 | 0.85 |
| Mlle de Stermaria | 1802 ± 354 | 353.9 | 8 | 4 | +0.39 | 0.69 |
| le petit Cambremer | 1801 ± 443 | 442.7 | 5 | 1 | 0.0 | 0.0 |
| Eulalie | 1792 ± 368 | 367.9 | 5 | 3 | +0.507 | 0.96 |
| Victurnien | 1791 ± 466 | 466.2 | 3 | 2 | +0.4 | 0.4 |
| Mme Elstir | 1789 ± 455 | 454.8 | 4 | 1 | +0.75 | 0.75 |
| le pianiste | 1768 ± 292 | 291.7 | 10 | 5 | +0.17 | 0.41 |
| Jupien | 1763 ± 201 | 200.7 | 31 | 15 | +0.444 | 0.5852 |
| Bibi | 1757 ± 468 | 468.4 | 3 | 1 | +0.75 | 0.75 |
| M. Swann, le père | 1742 ± 488 | 487.8 | 3 | 1 | +0.8 | 0.8 |
| Lady Israels | 1734 ± 490 | 489.7 | 2 | 1 | 0.0 | 0.0 |
| Mme de Villebon | 1732 ± 485 | 485.3 | 3 | 1 | +1.5 | 1.5 |
| le peintre | 1728 ± 236 | 236.4 | 16 | 8 | +0.119 | 0.2945 |
| cousine Poictiers | 1720 ± 510 | 510.0 | 2 | 1 | +0.55 | 0.55 |
| l'amie de Mlle Vinteuil | 1720 ± 234 | 233.9 | 17 | 7 | +0.029 | 0.4857 |
| Maurice | 1719 ± 364 | 363.9 | 6 | 2 | 0.0 | 0.0 |
| tante Léonie | 1718 ± 238 | 238.3 | 13 | 9 | +0.347 | 0.5071 |
| vicomte de Courvoisier | 1714 ± 503 | 502.8 | 3 | 1 | 0.0 | 0.0 |
| Mme de Valcourt | 1712 ± 497 | 497.2 | 3 | 1 | 0.0 | 0.0 |
| Léa | 1710 ± 506 | 505.9 | 2 | 1 | 0.0 | 0.0 |
| Théodore | 1709 ± 442 | 441.5 | 2 | 1 | +0.8 | 0.8 |
| docteur Percepied | 1708 ± 385 | 385.0 | 4 | 1 | 0.0 | 0.0 |
| M. de Grouchy | 1705 ± 370 | 369.8 | 5 | 3 | 0.0 | 0.0 |
| duc de Sidonia | 1696 ± 517 | 516.6 | 2 | 1 | 0.0 | 0.0 |
| M. de Saint-Candé | 1691 ± 508 | 507.9 | 3 | 1 | +0.6 | 0.6 |
| prince de Sagan | 1683 ± 405 | 405.2 | 4 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1666 ± 222 | 221.6 | 18 | 9 | +0.027 | 0.2933 |
| M. de Vaudémont | 1664 ± 539 | 539.0 | 1 | 1 | 0.0 | 0.0 |
| la reine de Naples | 1662 ± 313 | 313.1 | 8 | 4 | +0.212 | 0.2125 |
| M. Vinteuil | 1658 ± 217 | 216.8 | 21 | 9 | +0.02 | 1.1667 |
| Mlle d'Éporcheville | 1650 ± 554 | 554.5 | 1 | 1 | 0.0 | 0.0 |
| duchesse de Létourville | 1645 ± 559 | 558.6 | 1 | 1 | 0.0 | 0.0 |
| Dumont | 1640 ± 583 | 582.9 | 1 | 1 | 0.0 | 0.0 |
| marquis de Surgis | 1639 ± 403 | 402.8 | 4 | 1 | 0.0 | 0.0 |
| princesse d'Orvillers | 1638 ± 469 | 468.8 | 4 | 1 | +0.62 | 0.62 |
| docteur du Boulbon | 1630 ± 295 | 294.6 | 7 | 4 | +0.97 | 1.21 |
| Larivière | 1628 ± 460 | 460.0 | 2 | 1 | +1.86 | 1.86 |
| princesse Mathilde | 1622 ± 392 | 392.2 | 4 | 2 | 0.0 | 0.0 |
| Dieulafoy | 1622 ± 451 | 451.3 | 2 | 1 | +1.9 | 1.9 |
| la marquise | 1617 ± 490 | 490.2 | 2 | 3 | -1.0 | 1.0 |
| le vicomte de Courvoisier | 1617 ± 382 | 382.4 | 4 | 1 | 0.0 | 0.0 |
| Rémi | 1614 ± 467 | 467.1 | 2 | 2 | 0.0 | 0.0 |
| marquis de Palancy | 1610 ± 569 | 568.7 | 1 | 2 | 0.0 | 0.0 |
| M. de Courgivaux | 1609 ± 574 | 573.5 | 1 | 1 | +0.7 | 0.7 |
| Rosemonde | 1608 ± 409 | 409.1 | 3 | 1 | 0.0 | 0.0 |
| Céline | 1606 ± 332 | 331.5 | 6 | 1 | +0.39 | 0.39 |
| marquis de Cambremer | 1606 ± 232 | 231.7 | 21 | 4 | -0.19 | 0.54 |
| duc de La Trémoïlle | 1605 ± 328 | 327.7 | 6 | 1 | 0.0 | 0.0 |
| Mme de Franquetot | 1605 ± 582 | 581.9 | 1 | 1 | 0.0 | 0.0 |
| la marquise douairière de Cambremer | 1604 ± 335 | 335.4 | 10 | 5 | +0.52 | 0.52 |
| Mme de Chaussepierre | 1603 ± 315 | 315.2 | 6 | 2 | 0.0 | 0.0 |
| M. de Goncourt | 1601 ± 358 | 357.7 | 6 | 1 | 0.0 | 0.0 |
| Mme d'Hunolstein | 1598 ± 386 | 386.0 | 4 | 1 | 0.0 | 0.0 |
| M. de Chaussepierre | 1595 ± 382 | 382.5 | 3 | 1 | 0.0 | 0.0 |
| M. d'Herweck | 1594 ± 434 | 433.7 | 3 | 2 | 0.0 | 0.0 |
| M. de Luxembourg | 1589 ± 515 | 514.8 | 2 | 1 | +0.78 | 0.78 |
| Mme Sazerat | 1585 ± 318 | 318.3 | 8 | 4 | -0.466 | 0.466 |
| baron de Guermantes | 1584 ± 584 | 584.0 | 2 | 2 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1581 ± 472 | 471.5 | 2 | 1 | +1.56 | 1.56 |
| princesse Sherbatoff | 1579 ± 256 | 256.0 | 10 | 3 | +0.017 | 1.05 |
| Mlle Vinteuil | 1579 ± 234 | 234.3 | 17 | 8 | -0.349 | 0.6913 |
| Céleste Albaret | 1562 ± 320 | 319.5 | 7 | 3 | +0.653 | 0.6533 |
| Mme Leroi | 1561 ± 298 | 298.1 | 10 | 6 | -0.1 | 0.1 |
| grand-duc Wladimir | 1552 ± 351 | 351.4 | 5 | 1 | 0.0 | 0.0 |
| commandant Duroc | 1542 ± 422 | 421.5 | 3 | 2 | +0.25 | 0.25 |
| princesse de Luxembourg | 1532 ± 373 | 373.1 | 5 | 4 | -0.098 | 0.0975 |
| Mme G... | 1524 ± 490 | 490.4 | 2 | 1 | 0.0 | 0.0 |
| les Iéna | 1512 ± 374 | 374.2 | 6 | 2 | -0.3 | 0.3 |
| général de Monserfeuil | 1510 ± 368 | 368.2 | 4 | 2 | -0.7 | 0.7 |
| Octave | 1508 ± 247 | 247.2 | 16 | 3 | -0.18 | 1.3533 |
| M. Barrère | 1502 ± 520 | 520.0 | 1 | 1 | -0.8 | 0.8 |
| le directeur | 1501 ± 262 | 261.9 | 10 | 6 | -0.505 | 0.755 |
| M. de Stermaria | 1501 ± 324 | 323.8 | 7 | 4 | -0.205 | 0.205 |
| Arnulphe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vallenères | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | -1.56 | 1.56 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | +0.65 | 0.65 |
| prince Von | 1497 ± 251 | 251.4 | 12 | 5 | -0.502 | 0.6744 |
| le roi Théodose | 1483 ± 348 | 347.8 | 7 | 3 | +0.517 | 0.9167 |
| comte Arnulphe | 1481 ± 438 | 438.5 | 2 | 1 | -0.8 | 0.8 |
| M. d'Argencourt | 1481 ± 231 | 230.6 | 17 | 10 | -0.493 | 0.593 |
| Dreyfus | 1480 ± 666 | 665.9 | 1 | 1 | -0.5 | 0.5 |
| Poullein | 1462 ± 349 | 349.4 | 6 | 3 | -0.573 | 0.5733 |
| général de Froberville | 1460 ± 214 | 214.3 | 17 | 8 | -0.364 | 0.3637 |
| comtesse de Monteriender | 1458 ± 401 | 401.2 | 3 | 1 | -0.75 | 0.75 |
| oncle Adolphe | 1458 ± 363 | 362.7 | 6 | 4 | -0.18 | 0.53 |
| prince de Foix | 1452 ± 227 | 226.9 | 16 | 5 | -0.554 | 0.7264 |
| Sainte-Beuve | 1447 ± 544 | 543.9 | 1 | 1 | -0.68 | 0.68 |
| la jeune ouvriere | 1441 ± 382 | 382.3 | 3 | 1 | -1.44 | 1.44 |
| Flora | 1436 ± 318 | 317.8 | 6 | 1 | 0.0 | 0.0 |
| duc de Châtellerault | 1435 ± 258 | 258.1 | 10 | 4 | -0.67 | 0.88 |
| Gibergue | 1431 ± 371 | 371.0 | 4 | 2 | 0.0 | 0.0 |
| princesse de Silistrie | 1428 ± 306 | 306.5 | 8 | 1 | -0.85 | 0.85 |
| Antoine | 1414 ± 598 | 598.0 | 1 | 1 | -1.2 | 1.2 |
| Marie Gineste | 1405 ± 420 | 420.3 | 3 | 2 | +0.21 | 0.21 |
| marquise de Saint-Euverte | 1397 ± 206 | 206.3 | 20 | 9 | -0.488 | 0.4878 |
| vicomtesse de Saint-Fiacre | 1391 ± 574 | 573.5 | 1 | 1 | -1.6 | 1.6 |
| prince de Faffenheim | 1384 ± 359 | 358.8 | 4 | 3 | -0.262 | 0.602 |
| Gisèle | 1383 ± 290 | 290.2 | 9 | 4 | -0.412 | 0.8125 |
| Dechambre | 1375 ± 375 | 374.7 | 5 | 1 | -0.75 | 0.75 |
| Alix | 1372 ± 276 | 275.7 | 9 | 4 | -0.808 | 0.8085 |
| Victor | 1370 ± 550 | 549.9 | 2 | 1 | -0.7 | 0.7 |
| comte de Paris | 1366 ± 560 | 560.0 | 1 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1362 ± 312 | 311.6 | 8 | 2 | -0.03 | 1.67 |
| capitaine de Borodino | 1360 ± 366 | 366.0 | 5 | 5 | -0.538 | 0.838 |
| princesse de Nassau | 1353 ± 470 | 470.4 | 2 | 1 | -0.75 | 0.75 |
| M. de Vaugoubert | 1349 ± 234 | 233.5 | 17 | 7 | -0.41 | 0.884 |
| Mme de Citri | 1340 ± 420 | 419.9 | 4 | 1 | -1.8 | 1.8 |
| M. de Beautreillis | 1335 ± 450 | 450.5 | 4 | 1 | -0.55 | 0.55 |
| Mme Blandais | 1333 ± 525 | 524.9 | 2 | 2 | -0.325 | 0.325 |
| ma grand'tante | 1333 ± 256 | 256.5 | 15 | 4 | -1.065 | 1.065 |
| prince Foggi | 1332 ± 536 | 535.9 | 1 | 1 | 0.0 | 0.0 |
| Madame d'Ambresac | 1331 ± 537 | 537.3 | 1 | 1 | 0.0 | 0.0 |
| duc d'Aumale | 1328 ± 534 | 534.4 | 1 | 1 | 0.0 | 0.0 |
| princesse d'Épinay | 1328 ± 347 | 347.4 | 6 | 2 | -0.735 | 0.735 |
| M. de Palancy | 1317 ± 440 | 440.5 | 3 | 1 | -0.72 | 0.72 |
| Israël | 1316 ± 523 | 522.7 | 2 | 1 | -1.7 | 1.7 |
| Mme de Vaugoubert | 1312 ± 553 | 553.1 | 1 | 1 | -1.7 | 1.7 |
| spécialiste X... | 1310 ± 506 | 505.5 | 2 | 1 | -1.72 | 1.72 |
| le prince de Faffenheim | 1309 ± 513 | 512.8 | 2 | 1 | -0.382 | 0.382 |
| M. Nissim Bernard | 1302 ± 323 | 323.4 | 9 | 6 | -0.902 | 0.902 |
| Mlle Bloch | 1294 ± 552 | 551.9 | 1 | 1 | -0.6 | 0.6 |
| professeur E... | 1290 ± 492 | 491.8 | 3 | 3 | -0.973 | 1.4267 |
| Majesté | 1289 ± 503 | 502.9 | 2 | 1 | -1.7 | 1.7 |
| duc de Guastalla | 1265 ± 485 | 484.7 | 3 | 1 | -0.6 | 0.6 |
| Potain | 1262 ± 478 | 477.6 | 5 | 1 | -0.6 | 0.6 |
| Bloch père | 1261 ± 218 | 217.7 | 25 | 7 | -1.086 | 1.0857 |
| l'empereur Guillaume | 1256 ± 476 | 475.8 | 3 | 1 | -1.66 | 1.66 |
| M. Bontemps | 1251 ± 409 | 408.7 | 6 | 4 | -0.52 | 0.52 |
| M. Ski | 1240 ± 323 | 323.3 | 8 | 2 | -1.21 | 1.21 |
| Mme d'Heudicourt | 1221 ± 462 | 461.8 | 3 | 1 | -1.56 | 1.56 |
| prince des Laumes | 1209 ± 462 | 461.7 | 4 | 1 | -1.76 | 1.76 |
| M. Pierre | 1206 ± 388 | 387.7 | 8 | 4 | -0.573 | 0.5725 |
| Mme de Montmorency | 1202 ± 386 | 385.8 | 6 | 1 | -0.8 | 0.8 |
| Mme Putbus | 1200 ± 461 | 460.8 | 4 | 1 | -1.7 | 1.7 |
| Mme de Souvré | 1197 ± 330 | 330.3 | 8 | 3 | -0.8 | 0.8 |
| le professeur E… | 1194 ± 451 | 450.6 | 4 | 2 | -1.225 | 1.225 |
| Mme de Varambon | 1192 ± 452 | 451.9 | 4 | 1 | -1.7 | 1.7 |
| Mme Blatin | 1191 ± 371 | 371.0 | 10 | 3 | -1.34 | 1.34 |
| princesse de Caprarola | 1187 ± 440 | 440.3 | 4 | 1 | -0.8 | 0.8 |
| marquise d'Amoncourt | 1182 ± 448 | 447.9 | 4 | 1 | -0.82 | 0.82 |
| vicomtesse d'Égremont | 1182 ± 452 | 452.2 | 3 | 1 | -1.6 | 1.6 |
| M. de Bornier | 1178 ± 327 | 327.3 | 10 | 3 | -1.13 | 1.13 |
| Mme de Mortemart | 1176 ± 362 | 361.6 | 9 | 1 | -1.056 | 1.056 |
| la cousine d'Oriane | 1138 ± 418 | 418.4 | 6 | 2 | -1.195 | 1.195 |
| le bâtonnier | 1120 ± 405 | 404.9 | 10 | 5 | -1.182 | 1.182 |
| colonel de Froberville | 1096 ± 400 | 399.9 | 8 | 2 | -1.7 | 1.7 |
