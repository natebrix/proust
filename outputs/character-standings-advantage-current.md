# Character Standings — advantage (scoring v2)

- Standings version: `character_standings_advantage_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_advantage_name_view_v1` (`outputs/scoring-v2-enrichment/scoring-v2-advantage-name-view-ratings.json`)
- Lens / view: `advantage` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`31` ranked, `162` without sufficient evidence)
- Comparisons: `2708` (mean weight `0.3829`, draw rate `0.106`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `enrichment`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2-enrichment/scoring-v2-advantage-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `31` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | duchesse de Guermantes | 1602 ± 93 | 1508.5 | 354 | 225-92-37 | 183 | +0.049 | 0.4899 |
| 2 | comte de Forcheville | 1699 ± 194 | 1505.3 | 62 | 44-14-4 | 28 | +0.139 | 0.1814 |
| 3 | M. Verdurin | 1647 ± 160 | 1486.7 | 84 | 47-26-11 | 32 | -0.176 | 0.2672 |
| 4 | Mme de Villeparisis | 1595 ± 134 | 1461.4 | 118 | 72-38-8 | 73 | -0.077 | 0.3693 |
| 5 | Françoise | 1581 ± 133 | 1448.1 | 100 | 51-38-11 | 61 | +0.086 | 0.5864 |
| 6 | la mère du narrateur | 1605 ± 169 | 1435.9 | 54 | 34-16-4 | 28 | +0.273 | 0.3177 |
| 7 | le narrateur | 1513 ± 82 | 1430.8 | 399 | 168-200-31 | 209 | -0.201 | 0.6321 |
| 8 | Rachel | 1600 ± 173 | 1427.0 | 56 | 27-20-9 | 29 | -0.216 | 0.6092 |
| 9 | la grand-mère | 1563 ± 145 | 1418.6 | 74 | 37-31-6 | 48 | +0.177 | 0.6675 |
| 10 | Mme Verdurin | 1537 ± 122 | 1415.4 | 181 | 78-83-20 | 78 | -0.336 | 0.4254 |
| 11 | le père du narrateur | 1614 ± 199 | 1415.1 | 44 | 25-16-3 | 21 | +0.078 | 0.2873 |
| 12 | Albertine | 1504 ± 90 | 1413.3 | 183 | 80-84-19 | 126 | -0.203 | 0.7437 |
| 13 | baron de Charlus | 1502 ± 92 | 1410.0 | 283 | 133-115-35 | 110 | -0.256 | 0.7058 |
| 14 | docteur Cottard | 1555 ± 148 | 1407.1 | 107 | 50-43-14 | 37 | -0.165 | 0.7129 |
| 15 | Odette | 1520 ± 120 | 1400.3 | 248 | 112-107-29 | 124 | -0.081 | 0.5035 |
| 16 | Robert de Saint-Loup | 1486 ± 100 | 1386.5 | 234 | 105-108-21 | 138 | -0.132 | 0.6397 |
| 17 | Gilberte | 1503 ± 119 | 1384.0 | 139 | 64-61-14 | 57 | -0.028 | 0.4766 |
| 18 | Andrée | 1550 ± 173 | 1376.8 | 54 | 30-19-5 | 25 | -0.061 | 0.659 |
| 19 | Swann | 1475 ± 102 | 1372.3 | 386 | 144-197-45 | 177 | -0.317 | 0.7741 |
| 20 | Norpois | 1507 ± 135 | 1371.4 | 101 | 45-44-12 | 54 | -0.069 | 0.4894 |
| 21 | Bergotte | 1549 ± 181 | 1368.0 | 47 | 25-21-1 | 27 | +0.127 | 0.8517 |
| 22 | Brichot | 1534 ± 177 | 1357.0 | 62 | 27-24-11 | 17 | -0.262 | 0.6579 |
| 23 | princesse de Guermantes | 1531 ± 182 | 1349.1 | 53 | 21-22-10 | 19 | -0.076 | 0.4858 |
| 24 | princesse de Parme | 1489 ± 144 | 1345.1 | 82 | 37-37-8 | 36 | -0.121 | 0.2381 |
| 25 | Morel | 1476 ± 134 | 1342.5 | 101 | 41-50-10 | 35 | -0.718 | 0.8773 |
| 26 | marquis de Bréauté | 1502 ± 184 | 1318.3 | 56 | 25-25-6 | 17 | -0.244 | 0.5146 |
| 27 | duc de Guermantes | 1423 ± 105 | 1318.0 | 234 | 75-126-33 | 97 | -0.507 | 0.5645 |
| 28 | Mme de Marsantes | 1480 ± 176 | 1303.9 | 47 | 20-25-2 | 21 | -0.449 | 0.5832 |
| 29 | prince de Guermantes | 1464 ± 193 | 1270.6 | 43 | 19-21-3 | 13 | -0.449 | 0.9777 |
| 30 | Bloch | 1317 ± 129 | 1187.7 | 146 | 31-100-15 | 64 | -0.692 | 0.8975 |
| 31 | Mme de Cambremer | 1317 ± 192 | 1125.5 | 65 | 15-41-9 | 22 | -0.831 | 0.8309 |

## Insufficient comparative evidence

The `162` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| Aimé | 1888 ± 366 | 365.8 | 25 | 9 | +0.268 | 0.2678 |
| Mlle de Stermaria | 1772 ± 390 | 389.5 | 8 | 4 | +0.39 | 0.69 |
| le jeune marquis de Cambremer | 1751 ± 509 | 508.7 | 10 | 1 | +0.75 | 0.75 |
| grand-duc héritier de Luxembourg | 1750 ± 482 | 481.8 | 5 | 2 | +0.85 | 0.85 |
| Jupien | 1746 ± 251 | 251.1 | 31 | 15 | +0.444 | 0.5852 |
| Eulalie | 1742 ± 400 | 399.8 | 5 | 3 | +0.507 | 0.96 |
| le pianiste | 1738 ± 339 | 338.7 | 10 | 5 | +0.17 | 0.41 |
| l'amie de Mlle Vinteuil | 1736 ± 300 | 300.4 | 17 | 7 | +0.029 | 0.4857 |
| Victurnien | 1726 ± 510 | 509.5 | 3 | 2 | +0.4 | 0.4 |
| M. de Crécy | 1722 ± 498 | 498.2 | 6 | 1 | +0.6 | 0.6 |
| Mlle d'Oloron | 1714 ± 496 | 496.5 | 7 | 2 | 0.0 | 0.0 |
| tante Léonie | 1712 ± 279 | 279.4 | 13 | 9 | +0.347 | 0.5071 |
| Mme Elstir | 1697 ± 518 | 517.5 | 4 | 1 | +0.75 | 0.75 |
| M. de Grouchy | 1688 ± 436 | 435.5 | 5 | 3 | 0.0 | 0.0 |
| Lady Israels | 1686 ± 524 | 524.1 | 2 | 1 | 0.0 | 0.0 |
| Bibi | 1686 ± 520 | 519.5 | 3 | 1 | +0.75 | 0.75 |
| M. Swann, le père | 1684 ± 539 | 538.9 | 3 | 1 | +0.8 | 0.8 |
| le peintre | 1684 ± 289 | 288.7 | 16 | 8 | +0.119 | 0.2945 |
| Mme de Villebon | 1673 ± 527 | 527.4 | 3 | 1 | +1.5 | 1.5 |
| Maurice | 1673 ± 451 | 451.4 | 6 | 2 | 0.0 | 0.0 |
| cousine Poictiers | 1673 ± 546 | 546.5 | 2 | 1 | +0.55 | 0.55 |
| le grand-père du narrateur | 1672 ± 257 | 257.0 | 25 | 11 | +0.004 | 0.182 |
| Théodore | 1670 ± 492 | 492.5 | 2 | 1 | +0.8 | 0.8 |
| Mme de Surgis | 1668 ± 266 | 266.4 | 18 | 9 | +0.027 | 0.2933 |
| Léa | 1666 ± 540 | 540.5 | 2 | 1 | 0.0 | 0.0 |
| M. de Vaudémont | 1661 ± 540 | 540.3 | 1 | 1 | 0.0 | 0.0 |
| le petit Cambremer | 1661 ± 541 | 540.7 | 5 | 1 | 0.0 | 0.0 |
| Dumont | 1660 ± 582 | 582.1 | 1 | 1 | 0.0 | 0.0 |
| M. Vinteuil | 1654 ± 268 | 267.7 | 21 | 9 | +0.02 | 1.1667 |
| duc de Sidonia | 1653 ± 546 | 546.1 | 2 | 1 | 0.0 | 0.0 |
| Mlle d'Éporcheville | 1650 ± 555 | 554.6 | 1 | 1 | 0.0 | 0.0 |
| docteur Percepied | 1648 ± 498 | 498.5 | 4 | 1 | 0.0 | 0.0 |
| duchesse de Létourville | 1641 ± 560 | 560.2 | 1 | 1 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1638 ± 562 | 561.9 | 3 | 1 | 0.0 | 0.0 |
| la Berma | 1634 ± 246 | 246.5 | 26 | 13 | +0.351 | 1.2738 |
| M. d'Herweck | 1632 ± 467 | 466.9 | 3 | 2 | 0.0 | 0.0 |
| M. de Saint-Candé | 1629 ± 559 | 558.9 | 3 | 1 | +0.6 | 0.6 |
| princesse Mathilde | 1628 ± 457 | 456.7 | 4 | 2 | 0.0 | 0.0 |
| Mme de Valcourt | 1627 ± 566 | 565.6 | 3 | 1 | 0.0 | 0.0 |
| prince de Sagan | 1618 ± 516 | 515.9 | 4 | 1 | 0.0 | 0.0 |
| la marquise | 1613 ± 491 | 490.6 | 2 | 3 | -1.0 | 1.0 |
| Rémi | 1610 ± 468 | 467.7 | 2 | 2 | 0.0 | 0.0 |
| la reine de Naples | 1610 ± 360 | 360.3 | 8 | 4 | +0.212 | 0.2125 |
| M. de Courgivaux | 1609 ± 574 | 573.5 | 1 | 1 | +0.7 | 0.7 |
| la marquise douairière de Cambremer | 1604 ± 391 | 390.7 | 10 | 5 | +0.52 | 0.52 |
| Mme de Franquetot | 1603 ± 583 | 583.0 | 1 | 1 | 0.0 | 0.0 |
| Larivière | 1601 ± 510 | 510.4 | 2 | 1 | +1.86 | 1.86 |
| marquis de Surgis | 1598 ± 508 | 507.6 | 4 | 1 | 0.0 | 0.0 |
| Dieulafoy | 1596 ± 500 | 500.1 | 2 | 1 | +1.9 | 1.9 |
| marquis de Palancy | 1595 ± 579 | 579.3 | 1 | 2 | 0.0 | 0.0 |
| Mme de Chaussepierre | 1592 ± 422 | 421.9 | 6 | 2 | 0.0 | 0.0 |
| baron de Guermantes | 1589 ± 580 | 579.8 | 2 | 2 | 0.0 | 0.0 |
| Mlle Vinteuil | 1588 ± 301 | 300.8 | 17 | 8 | -0.349 | 0.6913 |
| princesse d'Orvillers | 1587 ± 557 | 557.4 | 4 | 1 | +0.62 | 0.62 |
| princesse Sherbatoff | 1585 ± 352 | 351.6 | 10 | 3 | +0.017 | 1.05 |
| Céline | 1580 ± 489 | 489.0 | 6 | 1 | +0.39 | 0.39 |
| Mme Leroi | 1574 ± 350 | 350.0 | 10 | 6 | -0.1 | 0.1 |
| marquis de Cambremer | 1573 ± 357 | 356.6 | 21 | 4 | -0.19 | 0.54 |
| M. de Chaussepierre | 1572 ± 486 | 485.6 | 3 | 1 | 0.0 | 0.0 |
| duc de La Trémoïlle | 1571 ± 498 | 498.2 | 6 | 1 | 0.0 | 0.0 |
| Céleste Albaret | 1570 ± 394 | 394.4 | 7 | 3 | +0.653 | 0.6533 |
| Mlle de Saint-Loup | 1570 ± 518 | 518.1 | 2 | 1 | +1.56 | 1.56 |
| M. de Luxembourg | 1569 ± 559 | 558.8 | 2 | 1 | +0.78 | 0.78 |
| Rosemonde | 1567 ± 499 | 498.6 | 3 | 1 | 0.0 | 0.0 |
| le vicomte de Courvoisier | 1567 ± 522 | 521.7 | 4 | 1 | 0.0 | 0.0 |
| Mme d'Hunolstein | 1567 ± 505 | 505.1 | 4 | 1 | 0.0 | 0.0 |
| commandant Duroc | 1562 ± 473 | 472.9 | 3 | 2 | +0.25 | 0.25 |
| M. de Goncourt | 1560 ± 526 | 526.4 | 6 | 1 | 0.0 | 0.0 |
| Mme Sazerat | 1553 ± 364 | 363.6 | 8 | 4 | -0.466 | 0.466 |
| docteur du Boulbon | 1550 ± 367 | 367.0 | 7 | 4 | +0.97 | 1.21 |
| Mme Cottard | 1545 ± 202 | 202.5 | 32 | 15 | -0.093 | 0.2876 |
| grand-duc Wladimir | 1540 ± 493 | 493.3 | 5 | 1 | 0.0 | 0.0 |
| Elstir | 1538 ± 209 | 209.1 | 41 | 18 | +0.332 | 0.7734 |
| le roi Théodose | 1537 ± 408 | 407.8 | 7 | 3 | +0.517 | 0.9167 |
| princesse de Luxembourg | 1526 ± 408 | 408.2 | 5 | 4 | -0.098 | 0.0975 |
| Mme Bontemps | 1511 ± 238 | 238.2 | 24 | 13 | -0.295 | 0.2954 |
| Mme G... | 1509 ± 542 | 541.6 | 2 | 1 | 0.0 | 0.0 |
| M. Barrère | 1503 ± 520 | 520.1 | 1 | 1 | -0.8 | 0.8 |
| Arnulphe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vallenères | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | -1.56 | 1.56 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | +0.65 | 0.65 |
| M. d'Argencourt | 1499 ± 300 | 299.5 | 17 | 10 | -0.493 | 0.593 |
| comte Arnulphe | 1493 ± 486 | 486.5 | 2 | 1 | -0.8 | 0.8 |
| général de Monserfeuil | 1492 ± 432 | 431.7 | 4 | 2 | -0.7 | 0.7 |
| Octave | 1492 ± 360 | 360.2 | 16 | 3 | -0.18 | 1.3533 |
| le directeur | 1491 ± 302 | 302.2 | 10 | 6 | -0.505 | 0.755 |
| prince de Foix | 1488 ± 292 | 291.5 | 16 | 5 | -0.554 | 0.7264 |
| Poullein | 1482 ± 395 | 394.8 | 6 | 3 | -0.573 | 0.5733 |
| Dreyfus | 1475 ± 658 | 658.5 | 1 | 1 | -0.5 | 0.5 |
| les Iéna | 1474 ± 464 | 463.5 | 6 | 2 | -0.3 | 0.3 |
| oncle Adolphe | 1470 ± 390 | 389.6 | 6 | 4 | -0.18 | 0.53 |
| prince Von | 1468 ± 318 | 318.3 | 12 | 5 | -0.502 | 0.6744 |
| Flora | 1467 ± 476 | 476.5 | 6 | 1 | 0.0 | 0.0 |
| comtesse de Monteriender | 1467 ± 490 | 490.4 | 3 | 1 | -0.75 | 0.75 |
| M. de Stermaria | 1458 ± 345 | 345.0 | 7 | 4 | -0.205 | 0.205 |
| comtesse Molé | 1458 ± 273 | 272.9 | 22 | 6 | -0.363 | 0.3633 |
| général de Froberville | 1457 ± 262 | 262.2 | 17 | 8 | -0.364 | 0.3637 |
| marquise de Saint-Euverte | 1451 ± 256 | 255.9 | 20 | 9 | -0.488 | 0.4878 |
| la jeune ouvriere | 1450 ± 473 | 472.7 | 3 | 1 | -1.44 | 1.44 |
| Marie Gineste | 1450 ± 451 | 451.0 | 3 | 2 | +0.21 | 0.21 |
| Gibergue | 1442 ± 442 | 441.7 | 4 | 2 | 0.0 | 0.0 |
| princesse de Silistrie | 1441 ± 467 | 467.3 | 8 | 1 | -0.85 | 0.85 |
| duc de Châtellerault | 1436 ± 315 | 315.4 | 10 | 4 | -0.67 | 0.88 |
| Sainte-Beuve | 1427 ± 549 | 549.0 | 1 | 1 | -0.68 | 0.68 |
| ma grand'tante | 1419 ± 342 | 342.5 | 15 | 4 | -1.065 | 1.065 |
| Dechambre | 1418 ± 482 | 482.5 | 5 | 1 | -0.75 | 0.75 |
| prince de Faffenheim | 1414 ± 373 | 373.2 | 4 | 3 | -0.262 | 0.602 |
| Antoine | 1411 ± 596 | 596.5 | 1 | 1 | -1.2 | 1.2 |
| capitaine de Borodino | 1401 ± 403 | 403.3 | 5 | 5 | -0.538 | 0.838 |
| M. de Beautreillis | 1399 ± 536 | 535.9 | 4 | 1 | -0.55 | 0.55 |
| Victor | 1397 ± 576 | 576.0 | 2 | 1 | -0.7 | 0.7 |
| Mme de Citri | 1391 ± 516 | 515.8 | 4 | 1 | -1.8 | 1.8 |
| vicomtesse de Saint-Fiacre | 1391 ± 574 | 573.5 | 1 | 1 | -1.6 | 1.6 |
| M. de Vaugoubert | 1386 ± 283 | 283.3 | 17 | 7 | -0.41 | 0.884 |
| princesse de Nassau | 1384 ± 518 | 518.1 | 2 | 1 | -0.75 | 0.75 |
| Mme Blandais | 1376 ± 560 | 559.7 | 2 | 2 | -0.325 | 0.325 |
| prince d'Agrigente | 1374 ± 388 | 387.9 | 8 | 2 | -0.03 | 1.67 |
| M. Nissim Bernard | 1373 ± 392 | 391.7 | 9 | 6 | -0.902 | 0.902 |
| princesse d'Épinay | 1371 ± 425 | 424.7 | 6 | 2 | -0.735 | 0.735 |
| comte de Paris | 1365 ± 560 | 559.7 | 1 | 1 | 0.0 | 0.0 |
| Gisèle | 1364 ± 352 | 352.1 | 9 | 4 | -0.412 | 0.8125 |
| M. de Palancy | 1358 ± 516 | 515.7 | 3 | 1 | -0.72 | 0.72 |
| Israël | 1352 ± 554 | 553.6 | 2 | 1 | -1.7 | 1.7 |
| professeur E... | 1352 ± 542 | 541.6 | 3 | 3 | -0.973 | 1.4267 |
| Mme d'Arpajon | 1351 ± 243 | 243.2 | 34 | 10 | -0.791 | 0.791 |
| Madame d'Ambresac | 1349 ± 545 | 544.8 | 1 | 1 | 0.0 | 0.0 |
| Alix | 1348 ± 335 | 335.1 | 9 | 4 | -0.808 | 0.8085 |
| duc d'Aumale | 1346 ± 542 | 542.1 | 1 | 1 | 0.0 | 0.0 |
| spécialiste X... | 1342 ± 533 | 532.9 | 2 | 1 | -1.72 | 1.72 |
| prince Foggi | 1333 ± 536 | 536.5 | 1 | 1 | 0.0 | 0.0 |
| l'empereur Guillaume | 1328 ± 530 | 529.5 | 3 | 1 | -1.66 | 1.66 |
| Mme de Vaugoubert | 1326 ± 555 | 555.1 | 1 | 1 | -1.7 | 1.7 |
| M. Bontemps | 1324 ± 442 | 442.3 | 6 | 4 | -0.52 | 0.52 |
| Potain | 1322 ± 520 | 520.4 | 5 | 1 | -0.6 | 0.6 |
| Mlle Bloch | 1320 ± 554 | 554.2 | 1 | 1 | -0.6 | 0.6 |
| M. Ski | 1314 ± 423 | 423.3 | 8 | 2 | -1.21 | 1.21 |
| le prince de Faffenheim | 1312 ± 514 | 514.4 | 2 | 1 | -0.382 | 0.382 |
| Saniette | 1311 ± 239 | 239.1 | 44 | 12 | -0.846 | 1.0578 |
| prince des Laumes | 1308 ± 537 | 536.8 | 4 | 1 | -1.76 | 1.76 |
| Mme Putbus | 1304 ± 536 | 535.9 | 4 | 1 | -1.7 | 1.7 |
| Mme de Montmorency | 1301 ± 478 | 477.7 | 6 | 1 | -0.8 | 0.8 |
| marquise de Gallardon | 1296 ± 268 | 268.3 | 29 | 10 | -0.751 | 0.7514 |
| Mme de Varambon | 1295 ± 517 | 516.7 | 4 | 1 | -1.7 | 1.7 |
| Majesté | 1290 ± 503 | 502.8 | 2 | 1 | -1.7 | 1.7 |
| Legrandin | 1285 ± 208 | 208.1 | 40 | 23 | -0.547 | 0.7439 |
| marquise d'Amoncourt | 1282 ± 513 | 512.9 | 4 | 1 | -0.82 | 0.82 |
| vicomtesse d'Égremont | 1278 ± 510 | 510.1 | 3 | 1 | -1.6 | 1.6 |
| Mme d'Heudicourt | 1278 ± 500 | 500.4 | 3 | 1 | -1.56 | 1.56 |
| duc de Guastalla | 1267 ± 485 | 485.4 | 3 | 1 | -0.6 | 0.6 |
| Mme de Mortemart | 1267 ± 448 | 447.8 | 9 | 1 | -1.056 | 1.056 |
| Mme Blatin | 1254 ± 435 | 435.2 | 10 | 3 | -1.34 | 1.34 |
| M. de Bornier | 1250 ± 401 | 401.0 | 10 | 3 | -1.13 | 1.13 |
| Bloch père | 1245 ± 295 | 295.0 | 25 | 7 | -1.086 | 1.0857 |
| le professeur E… | 1243 ± 476 | 475.9 | 4 | 2 | -1.225 | 1.225 |
| M. Pierre | 1221 ± 410 | 410.2 | 8 | 4 | -0.573 | 0.5725 |
| la cousine d'Oriane | 1213 ± 459 | 458.6 | 6 | 2 | -1.195 | 1.195 |
| princesse de Caprarola | 1199 ± 444 | 444.0 | 4 | 1 | -0.8 | 0.8 |
| Mme de Souvré | 1192 ± 390 | 389.5 | 8 | 3 | -0.8 | 0.8 |
| colonel de Froberville | 1174 ± 436 | 435.6 | 8 | 2 | -1.7 | 1.7 |
| le bâtonnier | 1150 ± 420 | 420.2 | 10 | 5 | -1.182 | 1.182 |
