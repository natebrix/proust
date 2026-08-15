# Character Standings — prestige (scoring v2)

- Standings version: `character_standings_prestige_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_prestige_name_view_v1` (`outputs/scoring-v2/scoring-v2-prestige-name-view-ratings.json`)
- Lens / view: `prestige` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`22` ranked, `171` without sufficient evidence)
- Comparisons: `954` (mean weight `0.6622`, draw rate `0.058`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-prestige-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `22` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Morel | 1773 ± 164 | 1608.1 | 43 | 33-9-1 | 35 | +0.206 | 0.2063 |
| 2 | duchesse de Guermantes | 1683 ± 99 | 1583.8 | 156 | 113-38-5 | 183 | +0.216 | 0.2704 |
| 3 | Odette | 1686 ± 124 | 1561.9 | 77 | 46-24-7 | 124 | +0.107 | 0.1687 |
| 4 | le narrateur | 1633 ± 118 | 1515.5 | 102 | 59-38-5 | 209 | +0.061 | 0.1003 |
| 5 | baron de Charlus | 1591 ± 93 | 1497.9 | 132 | 65-54-13 | 110 | +0.032 | 0.269 |
| 6 | Mme Verdurin | 1574 ± 105 | 1468.6 | 104 | 53-44-7 | 78 | +0.129 | 0.2362 |
| 7 | Rachel | 1640 ± 183 | 1456.7 | 24 | 14-10-0 | 29 | +0.041 | 0.2003 |
| 8 | Gilberte | 1539 ± 124 | 1415.4 | 53 | 23-29-1 | 57 | +0.085 | 0.174 |
| 9 | M. Verdurin | 1588 ± 189 | 1399.4 | 25 | 14-10-1 | 32 | 0.0 | 0.0 |
| 10 | Mme de Villeparisis | 1527 ± 131 | 1395.9 | 58 | 23-28-7 | 73 | -0.016 | 0.2053 |
| 11 | Norpois | 1569 ± 177 | 1392.2 | 38 | 21-16-1 | 54 | +0.101 | 0.1274 |
| 12 | Swann | 1494 ± 129 | 1364.8 | 114 | 44-60-10 | 177 | +0.024 | 0.1659 |
| 13 | Albertine | 1549 ± 185 | 1364.3 | 29 | 16-13-0 | 126 | +0.01 | 0.0469 |
| 14 | duc de Guermantes | 1496 ± 138 | 1357.3 | 56 | 20-35-1 | 97 | -0.012 | 0.0614 |
| 15 | docteur Cottard | 1537 ± 183 | 1353.8 | 30 | 15-14-1 | 37 | +0.057 | 0.0568 |
| 16 | Robert de Saint-Loup | 1476 ± 125 | 1351.3 | 74 | 30-44-0 | 138 | +0.047 | 0.1162 |
| 17 | princesse de Guermantes | 1544 ± 197 | 1346.4 | 21 | 9-11-1 | 19 | +0.213 | 0.3916 |
| 18 | Bloch | 1482 ± 156 | 1325.7 | 45 | 16-26-3 | 64 | -0.046 | 0.0934 |
| 19 | Françoise | 1514 ± 198 | 1315.7 | 22 | 7-13-2 | 61 | +0.052 | 0.0516 |
| 20 | Brichot | 1479 ± 171 | 1307.7 | 32 | 12-18-2 | 17 | -0.008 | 0.3518 |
| 21 | princesse de Parme | 1476 ± 180 | 1295.8 | 30 | 8-21-1 | 36 | +0.046 | 0.0875 |
| 22 | Mme de Cambremer | 1442 ± 196 | 1246.5 | 27 | 9-18-0 | 22 | -0.064 | 0.0636 |

## Insufficient comparative evidence

The `171` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| Alix | 1936 ± 421 | 420.6 | 4 | 4 | +0.2 | 0.2 |
| Mlle d'Oloron | 1936 ± 270 | 269.5 | 19 | 2 | +1.27 | 1.27 |
| Legrandin | 1887 ± 241 | 241.1 | 18 | 23 | +0.013 | 0.1435 |
| M. de Chaussepierre | 1846 ± 379 | 379.2 | 4 | 1 | +1.78 | 1.78 |
| Mme de Chaussepierre | 1839 ± 384 | 384.4 | 4 | 2 | +0.82 | 0.82 |
| docteur du Boulbon | 1822 ± 464 | 464.4 | 2 | 4 | +0.188 | 0.1875 |
| le peintre | 1792 ± 311 | 311.4 | 8 | 8 | +0.306 | 0.3063 |
| Bergotte | 1777 ± 469 | 468.7 | 4 | 27 | +0.037 | 0.1467 |
| docteur Percepied | 1770 ± 489 | 489.4 | 2 | 1 | 0.0 | 0.0 |
| l'amie de Mlle Vinteuil | 1750 ± 369 | 369.1 | 5 | 7 | 0.0 | 0.0 |
| Aimé | 1738 ± 385 | 385.2 | 5 | 9 | +0.064 | 0.0644 |
| vicomte de Courvoisier | 1731 ± 497 | 496.6 | 3 | 1 | +0.55 | 0.55 |
| la mère du narrateur | 1731 ± 266 | 266.4 | 10 | 28 | +0.005 | 0.0482 |
| prince de Guermantes | 1726 ± 296 | 295.7 | 9 | 13 | +0.131 | 0.1308 |
| Mlle Vinteuil | 1726 ± 385 | 385.2 | 4 | 8 | 0.0 | 0.0 |
| le petit Cambremer | 1724 ± 325 | 324.8 | 8 | 1 | +0.8 | 0.8 |
| Rosemonde | 1724 ± 496 | 496.0 | 2 | 1 | 0.0 | 0.0 |
| Madame d'Ambresac | 1723 ± 496 | 496.2 | 2 | 1 | +0.75 | 0.75 |
| Octave | 1721 ± 329 | 329.3 | 9 | 3 | +0.547 | 0.5467 |
| M. Vinteuil | 1715 ± 298 | 297.6 | 8 | 9 | +0.078 | 0.3 |
| Lady Israels | 1714 ± 532 | 531.7 | 1 | 1 | 0.0 | 0.0 |
| duc de Sidonia | 1703 ± 529 | 529.4 | 1 | 1 | 0.0 | 0.0 |
| le professeur E… | 1703 ± 529 | 529.4 | 1 | 2 | 0.0 | 0.0 |
| le vicomte de Courvoisier | 1698 ± 525 | 525.2 | 2 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1697 ± 211 | 210.6 | 19 | 9 | +0.347 | 0.5133 |
| la marquise | 1657 ± 574 | 574.5 | 1 | 3 | +0.2 | 0.2 |
| la grand-mère | 1632 ± 217 | 217.0 | 18 | 48 | +0.053 | 0.1185 |
| M. de Vaudémont | 1626 ± 564 | 564.3 | 1 | 1 | +0.7 | 0.7 |
| M. Nissim Bernard | 1623 ± 563 | 563.4 | 2 | 6 | +0.1 | 0.1 |
| comtesse Molé | 1615 ± 226 | 225.9 | 14 | 6 | -0.01 | 0.5767 |
| M. Bontemps | 1614 ± 407 | 406.6 | 3 | 4 | +0.425 | 0.425 |
| tante Léonie | 1610 ± 399 | 398.6 | 3 | 9 | +0.189 | 0.1889 |
| le père du narrateur | 1597 ± 237 | 236.8 | 14 | 21 | -0.001 | 0.0676 |
| Andrée | 1594 ± 234 | 233.5 | 17 | 25 | +0.043 | 0.0928 |
| Maurice | 1582 ± 484 | 484.2 | 2 | 2 | 0.0 | 0.0 |
| Jupien | 1576 ± 206 | 205.5 | 19 | 15 | +0.099 | 0.208 |
| Elstir | 1572 ± 435 | 434.6 | 3 | 18 | +0.094 | 0.0944 |
| comte de Forcheville | 1570 ± 225 | 225.4 | 18 | 28 | +0.087 | 0.0875 |
| duc de La Trémoïlle | 1568 ± 606 | 606.4 | 1 | 1 | 0.0 | 0.0 |
| la marquise douairière de Cambremer | 1568 ± 380 | 379.8 | 5 | 5 | +0.34 | 0.34 |
| Mme Bontemps | 1566 ± 269 | 269.3 | 10 | 13 | +0.18 | 0.18 |
| prince des Laumes | 1548 ± 629 | 629.0 | 1 | 1 | 0.0 | 0.0 |
| Mme Cottard | 1548 ± 245 | 245.3 | 12 | 15 | +0.05 | 0.05 |
| Mme de Valcourt | 1543 ± 386 | 385.5 | 4 | 1 | 0.0 | 0.0 |
| marquis de Bréauté | 1536 ± 244 | 244.2 | 13 | 17 | 0.0 | 0.0 |
| la reine de Naples | 1523 ± 291 | 290.8 | 8 | 4 | 0.0 | 0.0 |
| Mme Leroi | 1514 ± 332 | 331.5 | 6 | 6 | -0.47 | 0.7033 |
| colonel de Froberville | 1509 ± 486 | 485.6 | 2 | 2 | 0.0 | 0.0 |
| la Berma | 1508 ± 214 | 214.2 | 17 | 13 | -0.015 | 0.5169 |
| Antoine | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Bibi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dieulafoy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dreyfus | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Eulalie | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| Gisèle | 1500 ± 700 | 700.0 | 0 | 4 | 0.0 | 0.0 |
| Israël | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Léa | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Barrère | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Bornier | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| M. de Courgivaux | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Grouchy | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| M. de Luxembourg | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Palancy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Saint-Candé | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle d'Éporcheville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Blandais | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Mme Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme G... | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Putbus | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Sazerat | 1500 ± 700 | 700.0 | 0 | 4 | 0.0 | 0.0 |
| Mme d'Heudicourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme d'Hunolstein | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Citri | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Vaugoubert | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Villebon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Potain | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Rémi | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Sainte-Beuve | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| commandant Duroc | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| comtesse de Monteriender | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc de Guastalla | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| grand-duc Wladimir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| l'empereur Guillaume | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la cousine d'Oriane | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| la jeune ouvriere | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Surgis | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince Foggi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince Von | 1500 ± 700 | 700.0 | 0 | 5 | 0.0 | 0.0 |
| prince de Faffenheim | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| princesse d'Orvillers | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| princesse de Nassau | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| professeur E... | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| spécialiste X... | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomtesse de Saint-Fiacre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Franquetot | 1497 ± 522 | 522.1 | 2 | 1 | 0.0 | 0.0 |
| Mme de Marsantes | 1497 ± 232 | 231.7 | 16 | 21 | +0.036 | 0.101 |
| duchesse de Létourville | 1486 ± 478 | 477.8 | 2 | 1 | 0.0 | 0.0 |
| Céleste Albaret | 1484 ± 355 | 355.1 | 4 | 3 | +0.043 | 0.0433 |
| Marie Gineste | 1484 ± 355 | 355.1 | 4 | 2 | +0.065 | 0.065 |
| Victurnien | 1479 ± 452 | 452.4 | 3 | 2 | 0.0 | 0.0 |
| comte Arnulphe | 1476 ± 455 | 455.2 | 2 | 1 | 0.0 | 0.0 |
| marquis de Palancy | 1468 ± 400 | 399.8 | 4 | 2 | +0.615 | 0.615 |
| M. d'Herweck | 1464 ± 440 | 439.9 | 3 | 2 | 0.0 | 0.0 |
| Dechambre | 1450 ± 626 | 626.5 | 1 | 1 | 0.0 | 0.0 |
| M. de Vaugoubert | 1449 ± 369 | 369.4 | 6 | 7 | +0.237 | 0.2371 |
| princesse de Caprarola | 1448 ± 372 | 371.5 | 4 | 1 | 0.0 | 0.0 |
| cousine Poictiers | 1442 ± 619 | 619.2 | 1 | 1 | 0.0 | 0.0 |
| M. de Beautreillis | 1442 ± 620 | 619.7 | 1 | 1 | 0.0 | 0.0 |
| Mme de Varambon | 1442 ± 620 | 619.7 | 1 | 1 | 0.0 | 0.0 |
| le directeur | 1440 ± 409 | 408.9 | 3 | 6 | 0.0 | 0.0 |
| M. d'Argencourt | 1437 ± 294 | 293.7 | 10 | 10 | -0.072 | 0.072 |
| baron de Guermantes | 1433 ± 608 | 608.3 | 1 | 2 | 0.0 | 0.0 |
| le bâtonnier | 1426 ± 390 | 390.2 | 5 | 5 | +0.02 | 0.28 |
| marquise d'Amoncourt | 1423 ± 597 | 597.3 | 1 | 1 | 0.0 | 0.0 |
| le jeune marquis de Cambremer | 1421 ± 598 | 597.5 | 1 | 1 | 0.0 | 0.0 |
| duc d'Aumale | 1421 ± 594 | 594.3 | 1 | 1 | 0.0 | 0.0 |
| Poullein | 1420 ± 593 | 593.2 | 1 | 3 | 0.0 | 0.0 |
| Mme de Montmorency | 1418 ± 595 | 594.7 | 1 | 1 | 0.0 | 0.0 |
| Arnulphe | 1418 ± 593 | 592.6 | 1 | 1 | 0.0 | 0.0 |
| vicomtesse d'Égremont | 1414 ± 588 | 587.6 | 1 | 1 | 0.0 | 0.0 |
| M. Pierre | 1414 ± 586 | 586.4 | 1 | 4 | 0.0 | 0.0 |
| le grand-père du narrateur | 1413 ± 382 | 381.8 | 4 | 11 | 0.0 | 0.0 |
| prince de Sagan | 1408 ± 582 | 581.9 | 1 | 1 | 0.0 | 0.0 |
| Mlle Bloch | 1406 ± 587 | 587.1 | 1 | 1 | 0.0 | 0.0 |
| général de Froberville | 1400 ± 468 | 468.4 | 3 | 8 | 0.0 | 0.0 |
| Mme de Souvré | 1400 ± 391 | 391.0 | 4 | 3 | 0.0 | 0.0 |
| princesse de Luxembourg | 1390 ± 408 | 408.0 | 4 | 4 | -0.125 | 0.125 |
| Victor | 1388 ± 576 | 576.4 | 1 | 1 | 0.0 | 0.0 |
| général de Monserfeuil | 1386 ± 566 | 565.8 | 1 | 2 | 0.0 | 0.0 |
| oncle Adolphe | 1385 ± 576 | 576.3 | 1 | 4 | 0.0 | 0.0 |
| Théodore | 1377 ± 560 | 560.0 | 1 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1373 ± 550 | 550.2 | 2 | 2 | 0.0 | 0.0 |
| princesse d'Épinay | 1370 ± 543 | 543.4 | 2 | 2 | 0.0 | 0.0 |
| M. Swann, le père | 1369 ± 561 | 560.6 | 1 | 1 | 0.0 | 0.0 |
| Céline | 1366 ± 558 | 557.5 | 1 | 1 | 0.0 | 0.0 |
| Flora | 1366 ± 558 | 557.5 | 1 | 1 | 0.0 | 0.0 |
| marquis de Cambremer | 1365 ± 322 | 322.2 | 9 | 4 | -0.15 | 0.15 |
| comte de Paris | 1361 ± 555 | 555.4 | 1 | 1 | 0.0 | 0.0 |
| princesse Sherbatoff | 1358 ± 546 | 545.7 | 2 | 3 | 0.0 | 0.0 |
| Majesté | 1354 ± 538 | 537.5 | 2 | 1 | -1.64 | 1.64 |
| Larivière | 1352 ± 558 | 558.1 | 1 | 1 | 0.0 | 0.0 |
| ma grand'tante | 1351 ± 441 | 440.8 | 3 | 4 | 0.0 | 0.0 |
| Mme d'Arpajon | 1337 ± 302 | 302.2 | 12 | 10 | -0.075 | 0.075 |
| Dumont | 1335 ± 557 | 557.0 | 1 | 1 | -1.5 | 1.5 |
| M. de Crécy | 1334 ± 526 | 525.8 | 2 | 1 | 0.0 | 0.0 |
| Mme Blatin | 1334 ± 520 | 520.5 | 2 | 3 | 0.0 | 0.0 |
| M. Vallenères | 1332 ± 542 | 541.9 | 2 | 1 | -0.5 | 0.5 |
| princesse de Silistrie | 1328 ± 520 | 519.8 | 3 | 1 | 0.0 | 0.0 |
| le prince de Faffenheim | 1322 ± 520 | 520.5 | 2 | 1 | 0.0 | 0.0 |
| princesse Mathilde | 1304 ± 507 | 506.8 | 3 | 2 | 0.0 | 0.0 |
| M. Ski | 1302 ± 510 | 510.2 | 2 | 2 | 0.0 | 0.0 |
| duc de Châtellerault | 1290 ± 500 | 500.5 | 2 | 4 | 0.0 | 0.0 |
| marquise de Saint-Euverte | 1288 ± 249 | 249.3 | 19 | 9 | -0.684 | 0.8622 |
| Saniette | 1273 ± 244 | 244.4 | 22 | 12 | -0.262 | 0.2617 |
| Mme de Mortemart | 1256 ± 345 | 345.1 | 9 | 1 | -0.8 | 0.8 |
| M. de Stermaria | 1255 ± 478 | 477.6 | 3 | 4 | 0.0 | 0.0 |
| Mlle de Stermaria | 1255 ± 478 | 477.6 | 3 | 4 | 0.0 | 0.0 |
| le roi Théodose | 1252 ± 484 | 483.5 | 4 | 3 | 0.0 | 0.0 |
| grand-duc héritier de Luxembourg | 1246 ± 481 | 480.9 | 3 | 2 | -0.7 | 0.7 |
| marquise de Gallardon | 1246 ± 383 | 383.3 | 9 | 10 | -0.155 | 0.155 |
| prince de Foix | 1243 ± 478 | 478.4 | 3 | 5 | 0.0 | 0.0 |
| Gibergue | 1239 ± 469 | 468.8 | 3 | 2 | -0.375 | 0.375 |
| les Iéna | 1237 ± 472 | 472.3 | 5 | 2 | -0.3 | 0.3 |
| M. de Goncourt | 1216 ± 451 | 450.7 | 7 | 1 | -0.7 | 0.7 |
| Bloch père | 1198 ± 444 | 443.6 | 6 | 7 | -0.086 | 0.0857 |
| le pianiste | 1196 ± 437 | 436.9 | 7 | 5 | -0.15 | 0.15 |
| capitaine de Borodino | 1133 ± 415 | 414.9 | 7 | 5 | -0.596 | 0.596 |
