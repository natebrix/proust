# Character Standings — advantage (scoring v2)

- Standings version: `character_standings_advantage_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_advantage_name_view_v1` (`outputs/scoring-v2/scoring-v2-advantage-name-view-ratings.json`)
- Lens / view: `advantage` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `288` (`35` ranked, `253` without sufficient evidence)
- Comparisons: `3475` (mean weight `0.5697`, draw rate `0.064`)
- w2: `15.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `foundation`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-advantage-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `35` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Françoise | 1648 ± 141 | 1506.9 | 141 | 92-40-9 | 82 | +0.12 | 0.5288 |
| 2 | la grand-mère | 1654 ± 165 | 1489.4 | 142 | 88-50-4 | 80 | +0.138 | 0.639 |
| 3 | Bergotte | 1631 ± 175 | 1456.0 | 79 | 47-31-1 | 36 | +0.145 | 0.6619 |
| 4 | Elstir | 1610 ± 155 | 1454.8 | 70 | 39-21-10 | 29 | +0.476 | 0.6772 |
| 5 | Aimé | 1616 ± 184 | 1432.1 | 40 | 27-12-1 | 18 | +0.153 | 0.2479 |
| 6 | la mère du narrateur | 1580 ± 167 | 1412.6 | 81 | 49-32-0 | 40 | +0.087 | 0.3057 |
| 7 | Norpois | 1573 ± 165 | 1408.3 | 121 | 71-47-3 | 63 | -0.157 | 0.442 |
| 8 | princesse de Guermantes | 1577 ± 174 | 1402.6 | 58 | 41-16-1 | 25 | +0.129 | 0.4701 |
| 9 | Albertine | 1483 ± 96 | 1386.9 | 298 | 143-120-35 | 146 | -0.173 | 0.7048 |
| 10 | prince de Guermantes | 1542 ± 164 | 1378.7 | 61 | 34-25-2 | 22 | -0.093 | 0.3377 |
| 11 | le père du narrateur | 1560 ± 187 | 1372.8 | 47 | 26-20-1 | 24 | +0.007 | 0.1348 |
| 12 | Mme Verdurin | 1488 ± 121 | 1367.7 | 162 | 70-79-13 | 82 | -0.299 | 0.3451 |
| 13 | Swann | 1475 ± 111 | 1363.8 | 446 | 185-233-28 | 202 | -0.314 | 0.6576 |
| 14 | duchesse de Guermantes | 1461 ± 104 | 1356.5 | 452 | 269-154-29 | 199 | +0.051 | 0.4851 |
| 15 | Robert de Saint-Loup | 1457 ± 107 | 1350.5 | 338 | 153-166-19 | 168 | -0.107 | 0.6038 |
| 16 | Rachel | 1498 ± 148 | 1349.2 | 94 | 42-47-5 | 43 | -0.318 | 0.6765 |
| 17 | comte de Forcheville | 1542 ± 194 | 1348.3 | 61 | 42-16-3 | 25 | +0.022 | 0.2001 |
| 18 | M. Verdurin | 1524 ± 177 | 1346.8 | 47 | 28-15-4 | 27 | -0.111 | 0.2367 |
| 19 | Mme de Villeparisis | 1484 ± 144 | 1340.8 | 154 | 78-68-8 | 79 | -0.139 | 0.3257 |
| 20 | Andrée | 1466 ± 134 | 1331.6 | 78 | 33-34-11 | 31 | -0.084 | 0.4071 |
| 21 | Gilberte | 1418 ± 105 | 1312.5 | 187 | 88-86-13 | 76 | -0.063 | 0.4118 |
| 22 | Odette | 1435 ± 123 | 1311.8 | 283 | 126-131-26 | 142 | -0.134 | 0.3862 |
| 23 | baron de Charlus | 1407 ± 98 | 1309.3 | 308 | 141-139-28 | 119 | -0.3 | 0.7039 |
| 24 | marquis de Bréauté | 1489 ± 192 | 1297.1 | 40 | 19-16-5 | 19 | -0.111 | 0.1453 |
| 25 | docteur Cottard | 1460 ± 168 | 1291.7 | 106 | 41-56-9 | 43 | -0.22 | 0.6104 |
| 26 | le narrateur | 1369 ± 87 | 1281.5 | 816 | 286-474-56 | 316 | -0.304 | 0.6984 |
| 27 | duc de Guermantes | 1396 ± 116 | 1279.8 | 272 | 106-149-17 | 110 | -0.351 | 0.4264 |
| 28 | Mme de Marsantes | 1457 ± 187 | 1270.6 | 49 | 20-26-3 | 21 | -0.427 | 0.4396 |
| 29 | Brichot | 1411 ± 145 | 1266.1 | 57 | 24-29-4 | 21 | -0.285 | 0.3255 |
| 30 | princesse de Parme | 1372 ± 127 | 1245.5 | 94 | 32-57-5 | 38 | -0.271 | 0.3107 |
| 31 | la Berma | 1420 ± 186 | 1234.1 | 41 | 18-21-2 | 19 | +0.489 | 0.8981 |
| 32 | Bloch | 1349 ± 121 | 1227.8 | 173 | 57-104-12 | 71 | -0.589 | 0.7328 |
| 33 | Mme de Cambremer | 1401 ± 180 | 1220.7 | 45 | 16-27-2 | 20 | -0.452 | 0.5635 |
| 34 | Morel | 1304 ± 121 | 1183.0 | 92 | 26-61-5 | 32 | -0.536 | 0.7081 |
| 35 | Saniette | 1230 ± 198 | 1032.2 | 28 | 7-21-0 | 9 | -1.13 | 1.1298 |

## Insufficient comparative evidence

The `253` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1910 ± 405 | 405.1 | 14 | 3 | +0.897 | 0.8967 |
| Céleste Albaret | 1858 ± 342 | 342.1 | 17 | 3 | +1.56 | 1.56 |
| Mme Elstir | 1835 ± 432 | 431.5 | 7 | 1 | +0.78 | 0.78 |
| marquis de Beausergent | 1834 ± 423 | 422.8 | 12 | 1 | +0.72 | 0.72 |
| Eulalie | 1789 ± 315 | 315.4 | 13 | 7 | -0.014 | 0.1857 |
| docteur du Boulbon | 1783 ± 276 | 276.1 | 20 | 6 | +0.002 | 0.7283 |
| Mme de Grouchy | 1759 ± 474 | 473.9 | 4 | 1 | +0.408 | 0.408 |
| colonel Picquart | 1753 ± 470 | 469.8 | 4 | 1 | +1.7 | 1.7 |
| le grand-père du narrateur | 1742 ± 275 | 274.8 | 29 | 16 | +0.069 | 0.0688 |
| Marie | 1739 ± 408 | 407.5 | 7 | 1 | +0.7 | 0.7 |
| duchesse de La Trémoïlle | 1736 ± 483 | 483.0 | 3 | 1 | +0.8 | 0.8 |
| le peintre | 1729 ± 258 | 257.6 | 18 | 8 | +0.279 | 0.279 |
| Léa | 1728 ± 402 | 401.8 | 7 | 4 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1725 ± 405 | 404.7 | 7 | 2 | +1.8 | 1.8 |
| la Charité de Giotto | 1721 ± 539 | 539.1 | 1 | 1 | -1.36 | 1.36 |
| Rémi | 1714 ± 418 | 417.8 | 5 | 3 | 0.0 | 0.0 |
| Lady Israels | 1712 ± 497 | 496.8 | 2 | 1 | 0.0 | 0.0 |
| Mme Goupil | 1708 ± 491 | 491.2 | 4 | 2 | 0.0 | 0.0 |
| prince de Sagan | 1705 ± 515 | 515.0 | 2 | 1 | 0.0 | 0.0 |
| Manet | 1702 ± 512 | 512.5 | 2 | 1 | 0.0 | 0.0 |
| l'amie de Mlle Vinteuil | 1700 ± 242 | 241.9 | 24 | 12 | +0.068 | 0.0683 |
| docteur Percepied | 1698 ± 518 | 518.3 | 2 | 1 | 0.0 | 0.0 |
| Jupien | 1694 ± 220 | 220.4 | 29 | 18 | +0.339 | 0.5117 |
| Victurnien | 1692 ± 512 | 511.7 | 2 | 2 | 0.0 | 0.0 |
| prince d’Agrigente | 1688 ± 530 | 529.9 | 2 | 2 | 0.0 | 0.0 |
| Théodore | 1687 ± 454 | 453.8 | 2 | 1 | +1.76 | 1.76 |
| L’excellent écrivain G… | 1686 ± 534 | 533.6 | 2 | 1 | 0.0 | 0.0 |
| princesse d'Iéna | 1684 ± 526 | 526.5 | 2 | 1 | 0.0 | 0.0 |
| Maeterlinck | 1682 ± 442 | 441.9 | 4 | 1 | 0.0 | 0.0 |
| M. d'Orsan | 1675 ± 526 | 525.5 | 2 | 1 | 0.0 | 0.0 |
| le jeune marquis de Cambremer | 1673 ± 523 | 522.6 | 3 | 1 | 0.0 | 0.0 |
| grand-duc héritier de Luxembourg | 1670 ± 398 | 398.0 | 5 | 2 | +0.88 | 0.88 |
| monsieur Vallenères | 1667 ± 532 | 531.9 | 2 | 1 | 0.0 | 0.0 |
| la duchesse d'Alençon | 1666 ± 456 | 455.9 | 4 | 1 | 0.0 | 0.0 |
| Mme de Montmorency | 1658 ± 535 | 534.8 | 2 | 1 | 0.0 | 0.0 |
| Mme de Charlus | 1658 ± 399 | 398.8 | 6 | 2 | 0.0 | 0.0 |
| le commandant Duroc | 1652 ± 540 | 540.4 | 2 | 1 | +0.78 | 0.78 |
| Bibi | 1651 ± 541 | 541.1 | 2 | 1 | +0.7 | 0.7 |
| M. de Chevregny | 1649 ± 535 | 535.4 | 2 | 1 | 0.0 | 0.0 |
| M. de Crécy | 1649 ± 535 | 535.4 | 2 | 1 | 0.0 | 0.0 |
| Mme Féré | 1649 ± 535 | 535.4 | 2 | 1 | 0.0 | 0.0 |
| Gribelin | 1646 ± 418 | 417.5 | 6 | 1 | +0.55 | 0.55 |
| Mme de Rochechouart | 1645 ± 546 | 546.2 | 2 | 1 | 0.0 | 0.0 |
| duchesse de Létourville | 1641 ± 543 | 543.4 | 2 | 1 | 0.0 | 0.0 |
| Mme Sazerat | 1641 ± 362 | 362.4 | 6 | 6 | -0.2 | 0.2 |
| M. de Goncourt | 1641 ± 553 | 552.7 | 2 | 1 | 0.0 | 0.0 |
| Mme de Sévigné | 1640 ± 306 | 306.3 | 11 | 4 | +0.465 | 0.465 |
| Mlle d'Oloron | 1638 ± 546 | 546.5 | 2 | 1 | 0.0 | 0.0 |
| le petit Cambremer | 1638 ± 546 | 546.5 | 2 | 1 | 0.0 | 0.0 |
| princesse de Silistrie | 1638 ± 546 | 546.5 | 2 | 1 | 0.0 | 0.0 |
| Létourville | 1638 ± 543 | 543.2 | 2 | 1 | 0.0 | 0.0 |
| marquis Maurice de Vaudémont | 1637 ± 550 | 550.2 | 1 | 1 | 0.0 | 0.0 |
| M. Swann, le père | 1635 ± 550 | 549.5 | 2 | 1 | 0.0 | 0.0 |
| le comte de Paris | 1635 ± 550 | 549.5 | 2 | 1 | 0.0 | 0.0 |
| le prince de Galles | 1635 ± 550 | 549.5 | 2 | 1 | 0.0 | 0.0 |
| Herbinger | 1634 ± 567 | 566.7 | 1 | 1 | 0.0 | 0.0 |
| Duroc | 1632 ± 554 | 553.6 | 2 | 1 | +1.5 | 1.5 |
| Victoire | 1631 ± 550 | 550.5 | 2 | 1 | 0.0 | 0.0 |
| docteur Dieulafoy | 1630 ± 556 | 555.6 | 1 | 1 | +1.86 | 1.86 |
| Arnulphe | 1628 ± 566 | 566.2 | 1 | 1 | 0.0 | 0.0 |
| Bismarck | 1627 ± 435 | 434.7 | 4 | 1 | +0.7 | 0.7 |
| Mme Legrandin mère | 1626 ± 555 | 554.6 | 2 | 1 | 0.0 | 0.0 |
| Mme Cottard | 1619 ± 262 | 261.9 | 23 | 11 | +0.011 | 0.3384 |
| M. de Courgivaux | 1617 ± 565 | 565.4 | 1 | 1 | +1.6 | 1.6 |
| Mlle Vinteuil | 1615 ± 202 | 201.8 | 34 | 15 | -0.152 | 0.2637 |
| tante Léonie | 1614 ± 205 | 205.3 | 28 | 22 | -0.326 | 0.48 |
| Coquelin | 1611 ± 586 | 586.0 | 1 | 1 | 0.0 | 0.0 |
| marquis du Lau | 1608 ± 384 | 383.7 | 4 | 2 | +0.85 | 0.85 |
| elle | 1608 ± 581 | 581.1 | 1 | 1 | +0.56 | 0.56 |
| Flora | 1606 ± 429 | 428.9 | 4 | 1 | 0.0 | 0.0 |
| le baron Bréau-Chenut | 1603 ± 430 | 430.2 | 4 | 1 | 0.0 | 0.0 |
| le vieux père Chenut | 1603 ± 430 | 430.2 | 4 | 1 | 0.0 | 0.0 |
| M. Grevy | 1603 ± 583 | 582.9 | 1 | 1 | 0.0 | 0.0 |
| Maurice | 1601 ± 583 | 583.4 | 1 | 1 | 0.0 | 0.0 |
| comtesse G… | 1601 ± 583 | 582.9 | 1 | 1 | 0.0 | 0.0 |
| d’Orgeville | 1600 ± 592 | 591.9 | 1 | 1 | 0.0 | 0.0 |
| Mlle Bloch | 1598 ± 589 | 588.6 | 1 | 1 | 0.0 | 0.0 |
| M. Ski | 1596 ± 390 | 390.0 | 5 | 2 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1596 ± 589 | 588.6 | 1 | 1 | 0.0 | 0.0 |
| Mme de Stermaria | 1596 ± 432 | 431.9 | 3 | 1 | 0.0 | 0.0 |
| les La Trémoïlle | 1596 ± 591 | 590.7 | 1 | 1 | 0.0 | 0.0 |
| Charcot | 1595 ± 395 | 394.9 | 5 | 1 | 0.0 | 0.0 |
| M. Reinach | 1595 ± 395 | 394.9 | 5 | 1 | 0.0 | 0.0 |
| Sarah Bernhardt | 1592 ± 593 | 592.8 | 1 | 1 | 0.0 | 0.0 |
| le jeune prince de Foix | 1592 ± 593 | 592.8 | 1 | 1 | 0.0 | 0.0 |
| Marie-Aynard | 1591 ± 593 | 593.4 | 1 | 1 | 0.0 | 0.0 |
| Victurnienne | 1587 ± 598 | 597.7 | 1 | 1 | 0.0 | 0.0 |
| M. de Luxembourg | 1585 ± 503 | 503.0 | 2 | 1 | +0.62 | 0.62 |
| Mme de Sagan | 1585 ± 595 | 595.2 | 1 | 1 | 0.0 | 0.0 |
| M. de La Rochefoucauld | 1578 ± 442 | 442.5 | 3 | 1 | 0.0 | 0.0 |
| M. de Grouchy | 1576 ± 326 | 326.1 | 7 | 4 | +0.188 | 0.1875 |
| duchesse de La Rochefoucauld | 1576 ± 438 | 437.8 | 3 | 1 | 0.0 | 0.0 |
| duchesse de Praslin | 1576 ± 438 | 437.8 | 3 | 1 | 0.0 | 0.0 |
| oncle Adolphe | 1576 ± 283 | 283.3 | 16 | 6 | -0.212 | 0.4517 |
| Mme de Chaussepierre | 1576 ± 401 | 400.6 | 4 | 1 | 0.0 | 0.0 |
| M. Vinteuil | 1575 ± 210 | 209.9 | 37 | 15 | +0.125 | 0.9655 |
| duc de Poictiers | 1572 ± 424 | 423.5 | 3 | 1 | 0.0 | 0.0 |
| Élisabeth | 1571 ± 486 | 485.7 | 3 | 1 | 0.0 | 0.0 |
| prince des Laumes | 1570 ± 359 | 358.8 | 6 | 3 | 0.0 | 0.0 |
| cousine Poictiers | 1564 ± 415 | 415.3 | 3 | 1 | 0.0 | 0.0 |
| le roi Théodose | 1560 ± 430 | 429.8 | 4 | 3 | +0.183 | 0.1833 |
| jeune blonde de Rivebelle | 1557 ± 441 | 441.4 | 3 | 2 | 0.0 | 0.0 |
| d'Orléans | 1555 ± 461 | 461.4 | 3 | 1 | 0.0 | 0.0 |
| baron de Guermantes | 1554 ± 619 | 619.3 | 1 | 1 | 0.0 | 0.0 |
| le capitaine | 1554 ± 522 | 522.2 | 2 | 1 | +0.55 | 0.55 |
| Mlle de Stermaria | 1552 ± 296 | 296.1 | 10 | 5 | +0.176 | 0.524 |
| M. de Beauserfeuil | 1547 ± 444 | 444.5 | 3 | 1 | 0.0 | 0.0 |
| Mlle d'Éporcheville | 1547 ± 475 | 475.1 | 3 | 2 | 0.0 | 0.0 |
| la marquise douairière de Cambremer | 1545 ± 276 | 275.8 | 11 | 6 | +0.372 | 0.6217 |
| Sir Rufus Israël | 1544 ± 467 | 467.1 | 3 | 1 | 0.0 | 0.0 |
| le pianiste | 1544 ± 341 | 341.1 | 5 | 3 | +0.283 | 0.2833 |
| M. de Stermaria | 1541 ± 438 | 438.0 | 3 | 4 | -0.2 | 0.2 |
| Mme Leroi | 1541 ± 307 | 306.8 | 9 | 5 | -0.21 | 0.47 |
| duc de Sidonia | 1540 ± 522 | 522.5 | 2 | 1 | -0.6 | 0.6 |
| Esther | 1536 ± 396 | 395.9 | 5 | 2 | 0.0 | 0.0 |
| Mlle de l’Orgeville | 1535 ± 512 | 511.5 | 2 | 1 | 0.0 | 0.0 |
| le marquis de Palancy | 1534 ± 444 | 444.2 | 4 | 1 | 0.0 | 0.0 |
| le marquis de Ganançay | 1534 ± 512 | 511.8 | 4 | 1 | 0.0 | 0.0 |
| Liszt | 1532 ± 647 | 646.8 | 1 | 1 | 0.0 | 0.0 |
| Mme Ristori | 1532 ± 647 | 646.8 | 1 | 1 | 0.0 | 0.0 |
| M. de Bornier | 1531 ± 467 | 467.0 | 4 | 1 | 0.0 | 0.0 |
| princesse Mathilde | 1528 ± 455 | 455.4 | 3 | 2 | 0.0 | 0.0 |
| M. Vibert | 1528 ± 552 | 552.3 | 1 | 1 | 0.0 | 0.0 |
| M. Barrère | 1526 ± 561 | 560.8 | 1 | 1 | -0.7 | 0.7 |
| prince de Saxe | 1524 ± 500 | 500.0 | 2 | 1 | 0.0 | 0.0 |
| prince de Chimay | 1521 ± 663 | 663.3 | 1 | 1 | 0.0 | 0.0 |
| duc de Chartres | 1520 ± 664 | 664.3 | 1 | 1 | 0.0 | 0.0 |
| général de Froberville | 1516 ± 329 | 328.9 | 11 | 7 | -0.093 | 0.0929 |
| Goncourt | 1511 ± 400 | 399.7 | 4 | 2 | 0.0 | 0.0 |
| Monsieur Vallenères | 1509 ± 491 | 491.4 | 2 | 1 | 0.0 | 0.0 |
| Lady Israël | 1508 ± 465 | 465.3 | 2 | 1 | 0.0 | 0.0 |
| Léonor de Cambremer | 1505 ± 581 | 581.1 | 2 | 1 | 0.0 | 0.0 |
| comtesse de Monteriender | 1502 ± 422 | 422.5 | 2 | 1 | 0.0 | 0.0 |
| princesse de Luxembourg | 1502 ± 288 | 288.2 | 11 | 6 | -0.125 | 0.125 |
| comte de Paris | 1501 ± 402 | 401.8 | 4 | 3 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | +0.3 | 0.3 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | -1.5 | 1.5 |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | -1.7 | 1.7 |
| M. Carnot | 1494 ± 493 | 493.2 | 2 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1490 ± 217 | 217.1 | 18 | 9 | -0.264 | 0.2644 |
| Barrès | 1489 ± 482 | 482.4 | 2 | 1 | 0.0 | 0.0 |
| Clémenceau | 1489 ± 482 | 482.4 | 2 | 1 | 0.0 | 0.0 |
| Mme Carnot | 1488 ± 497 | 496.9 | 2 | 1 | 0.0 | 0.0 |
| comtesse douairière d'Argencourt | 1486 ± 481 | 480.7 | 2 | 1 | 0.0 | 0.0 |
| duchesse de Gallardon douairière | 1486 ± 481 | 480.7 | 2 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Henry | 1486 ± 494 | 493.9 | 2 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Picquart | 1486 ± 494 | 493.9 | 2 | 1 | 0.0 | 0.0 |
| marquis de Fierbois | 1483 ± 487 | 487.4 | 2 | 1 | 0.0 | 0.0 |
| M. de Miribel | 1482 ± 497 | 496.9 | 2 | 1 | 0.0 | 0.0 |
| Napoléon III | 1480 ± 456 | 456.3 | 3 | 1 | 0.0 | 0.0 |
| Octave | 1477 ± 394 | 393.5 | 4 | 2 | +0.02 | 1.82 |
| Mme Putbus | 1476 ± 490 | 490.5 | 2 | 1 | 0.0 | 0.0 |
| Vigny | 1474 ± 475 | 475.1 | 2 | 1 | -0.75 | 0.75 |
| Dostoïevski | 1474 ± 488 | 488.3 | 2 | 1 | 0.0 | 0.0 |
| Mme Iéna | 1472 ± 498 | 497.6 | 2 | 1 | 0.0 | 0.0 |
| Sainte-Beuve | 1469 ± 483 | 482.8 | 2 | 1 | 0.0 | 0.0 |
| Lady Rufus Israël | 1468 ± 467 | 467.3 | 2 | 1 | 0.0 | 0.0 |
| le bâtonnier | 1466 ± 467 | 466.7 | 2 | 1 | 0.0 | 0.0 |
| princesse d'Épinay | 1466 ± 323 | 323.1 | 7 | 3 | 0.0 | 0.0 |
| Rosemonde | 1464 ± 274 | 274.4 | 12 | 4 | 0.0 | 0.0 |
| Émilie Daltier | 1464 ± 484 | 483.7 | 2 | 1 | 0.0 | 0.0 |
| Poullein | 1463 ± 491 | 490.8 | 2 | 2 | -0.53 | 0.53 |
| Dreyfus | 1457 ± 229 | 229.4 | 22 | 7 | -0.01 | 0.2043 |
| Dechambre | 1454 ± 443 | 442.8 | 3 | 1 | -0.7 | 0.7 |
| M. d'Argencourt | 1452 ± 235 | 234.6 | 29 | 14 | -0.323 | 0.3227 |
| Madame Elstir | 1451 ± 438 | 438.3 | 3 | 1 | 0.0 | 0.0 |
| les demoiselles d’Ambresac | 1451 ± 438 | 438.3 | 3 | 1 | 0.0 | 0.0 |
| Mme Timoléon d'Amoncourt | 1449 ± 482 | 481.7 | 2 | 1 | 0.0 | 0.0 |
| Bloch père | 1443 ± 267 | 267.1 | 18 | 8 | -0.756 | 0.7562 |
| Dumont | 1443 ± 619 | 619.2 | 1 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1440 ± 456 | 456.2 | 2 | 2 | -0.07 | 1.77 |
| l'abbé Poiré | 1439 ± 436 | 435.6 | 3 | 1 | 0.0 | 0.0 |
| le grand-duc Wladimir | 1438 ± 470 | 470.3 | 2 | 1 | 0.0 | 0.0 |
| Thibaud | 1438 ± 428 | 428.0 | 4 | 1 | 0.0 | 0.0 |
| Balzac | 1433 ± 406 | 406.4 | 5 | 2 | 0.0 | 0.0 |
| Antoine | 1430 ± 609 | 609.0 | 1 | 1 | 0.0 | 0.0 |
| D'Annunzio | 1425 ± 476 | 476.2 | 2 | 1 | 0.0 | 0.0 |
| M. Nissim Bernard | 1425 ± 250 | 249.5 | 19 | 10 | -0.651 | 0.731 |
| M. Molé | 1424 ± 527 | 527.3 | 2 | 1 | 0.0 | 0.0 |
| Musset | 1424 ± 527 | 527.3 | 2 | 1 | 0.0 | 0.0 |
| Victor Hugo | 1424 ± 527 | 527.3 | 2 | 1 | 0.0 | 0.0 |
| Mme Bontemps | 1423 ± 215 | 214.8 | 26 | 13 | -0.195 | 0.1954 |
| M. de Bouillon | 1422 ± 530 | 529.5 | 2 | 1 | 0.0 | 0.0 |
| Gisèle | 1420 ± 308 | 307.5 | 9 | 5 | -0.848 | 0.848 |
| le directeur | 1417 ± 231 | 231.2 | 20 | 11 | -0.494 | 0.4945 |
| Mme Trombert | 1413 ± 598 | 597.5 | 1 | 1 | 0.0 | 0.0 |
| Théodose Cadet | 1408 ± 446 | 445.9 | 3 | 1 | -0.78 | 0.78 |
| ma grand'tante | 1401 ± 586 | 586.1 | 1 | 1 | -0.7 | 0.7 |
| M. de Chateaubriand | 1400 ± 455 | 454.8 | 3 | 2 | -1.075 | 1.075 |
| Céline | 1400 ± 323 | 322.7 | 10 | 2 | -0.425 | 0.425 |
| Mme de Villebon | 1399 ± 583 | 582.9 | 1 | 1 | -0.75 | 0.75 |
| prince Foggi | 1397 ± 585 | 584.8 | 1 | 1 | -0.68 | 0.68 |
| M. Bontemps | 1391 ± 340 | 340.3 | 9 | 2 | -0.03 | 1.73 |
| marquise de Citri | 1390 ± 480 | 479.5 | 2 | 1 | -2.12 | 2.12 |
| M. de Marsantes | 1389 ± 440 | 440.1 | 3 | 2 | +0.36 | 0.36 |
| général de Monserfeuil | 1388 ± 283 | 282.9 | 11 | 4 | -0.578 | 0.5775 |
| vicomtesse de Saint-Fiacre | 1383 ± 565 | 565.4 | 1 | 1 | -1.76 | 1.76 |
| capitaine de Borodino | 1382 ± 311 | 310.9 | 8 | 5 | -0.5 | 0.5 |
| Prince Henri d'Orléans | 1381 ± 493 | 493.3 | 2 | 1 | -0.7 | 0.7 |
| Madame d'Ambresac | 1380 ± 562 | 561.5 | 1 | 1 | 0.0 | 0.0 |
| M. d'Herweck | 1374 ± 460 | 460.1 | 2 | 2 | -0.35 | 0.35 |
| le grand-duc héritier de Luxembourg | 1374 ± 580 | 580.1 | 1 | 1 | -0.6 | 0.6 |
| M. Arthur Meyer | 1366 ± 574 | 574.3 | 1 | 1 | 0.0 | 0.0 |
| le prince Von | 1365 ± 330 | 330.5 | 8 | 2 | -0.141 | 0.141 |
| la jeune ouvriere | 1363 ± 575 | 575.4 | 1 | 1 | 0.0 | 0.0 |
| la marquise | 1361 ± 568 | 568.1 | 1 | 1 | -0.8 | 0.8 |
| le prince von *** | 1355 ± 545 | 545.1 | 2 | 1 | -0.516 | 0.516 |
| prince de Léon | 1351 ± 554 | 553.5 | 1 | 1 | 0.0 | 0.0 |
| le curé | 1350 ± 532 | 532.3 | 2 | 1 | -1.7 | 1.7 |
| Legrandin | 1348 ± 203 | 203.3 | 37 | 24 | -0.627 | 0.7313 |
| Beauserfeuil | 1344 ± 474 | 474.1 | 3 | 1 | -0.55 | 0.55 |
| marquis de Cambremer | 1339 ± 232 | 231.7 | 15 | 6 | -0.31 | 0.31 |
| Marie Gineste | 1337 ± 547 | 547.0 | 2 | 1 | 0.0 | 0.0 |
| Mme Blandais | 1331 ± 411 | 411.0 | 4 | 2 | -0.85 | 0.85 |
| duc d'Aumale | 1329 ± 438 | 437.6 | 3 | 2 | -0.325 | 0.325 |
| les Courvoisier | 1324 ± 395 | 395.4 | 5 | 1 | -0.82 | 0.82 |
| le professeur E… | 1311 ± 520 | 519.8 | 2 | 1 | -1.7 | 1.7 |
| marquise de Saint-Euverte | 1310 ± 205 | 205.1 | 35 | 13 | -0.414 | 0.4138 |
| le diplomate belge | 1305 ± 518 | 517.8 | 2 | 1 | -0.45 | 0.45 |
| princesse de Nassau | 1301 ± 544 | 544.5 | 1 | 1 | -1.64 | 1.64 |
| prince de Foix | 1299 ± 379 | 378.9 | 6 | 3 | -0.283 | 0.2833 |
| ma grand’tante | 1298 ± 418 | 417.8 | 7 | 1 | -0.85 | 0.85 |
| professeur E… | 1296 ± 418 | 417.9 | 4 | 2 | -1.155 | 1.155 |
| comtesse Molé | 1295 ± 284 | 284.4 | 13 | 6 | -0.543 | 0.543 |
| Cartier | 1290 ± 414 | 414.4 | 4 | 1 | -0.8 | 0.8 |
| duc de Châtellerault | 1289 ± 386 | 386.1 | 6 | 5 | -0.784 | 0.784 |
| Mme d'Arpajon | 1288 ± 204 | 204.0 | 27 | 8 | -0.569 | 0.5687 |
| Mme de Simiane | 1280 ± 484 | 484.4 | 3 | 1 | -0.8 | 0.8 |
| prince de Faffenheim | 1274 ± 488 | 488.0 | 3 | 2 | -0.94 | 0.94 |
| l'historien de la Fronde | 1268 ± 487 | 487.1 | 3 | 1 | -0.432 | 0.432 |
| Mme de Morienval | 1261 ± 394 | 394.3 | 6 | 1 | -0.8 | 0.8 |
| duchesse de Luxembourg | 1261 ± 394 | 394.3 | 6 | 1 | -0.8 | 0.8 |
| Mme de Vaugoubert | 1252 ± 460 | 459.7 | 3 | 2 | -1.22 | 1.22 |
| Mme Blatin | 1247 ± 484 | 484.4 | 2 | 1 | -1.84 | 1.84 |
| Mme de Varambon | 1237 ± 406 | 405.5 | 4 | 2 | -1.26 | 1.26 |
| princesse Sherbatoff | 1233 ± 246 | 246.1 | 17 | 5 | -0.75 | 0.7496 |
| Mme de Souvré | 1232 ± 334 | 334.2 | 9 | 2 | -0.45 | 0.45 |
| la cousine d'Oriane | 1232 ± 470 | 469.9 | 3 | 1 | -0.85 | 0.85 |
| M. de Vaugoubert | 1225 ± 270 | 269.7 | 15 | 9 | -0.801 | 0.8011 |
| l'ambassadrice de Turquie | 1212 ± 456 | 456.4 | 4 | 1 | -0.75 | 0.75 |
| marquise de Gallardon | 1212 ± 311 | 311.3 | 12 | 7 | -0.496 | 0.4957 |
| vicomtesse d'Égremont | 1210 ± 457 | 457.3 | 3 | 1 | -0.6 | 0.6 |
| M. Pierre | 1197 ± 456 | 456.5 | 4 | 2 | -1.663 | 1.663 |
| l'empereur | 1190 ± 455 | 455.4 | 4 | 1 | -1.76 | 1.76 |
| Mme d'Heudicourt | 1190 ± 280 | 279.9 | 15 | 5 | -0.872 | 0.872 |
| prince Von | 1186 ± 388 | 388.1 | 7 | 3 | -0.843 | 0.8433 |
| le prince de Faffenheim | 1181 ± 440 | 440.5 | 5 | 1 | -2.31 | 2.31 |
| Picquart | 1157 ± 432 | 431.8 | 7 | 2 | -0.7 | 0.7 |
| Mme de Franquetot | 1156 ± 289 | 289.3 | 15 | 3 | -0.587 | 0.5867 |
| Alix | 1117 ± 409 | 408.8 | 9 | 3 | -1.502 | 1.502 |
| M. de Vigny | 1107 ± 406 | 406.5 | 8 | 1 | -1.8 | 1.8 |
| colonel de Froberville | 1007 ± 374 | 374.5 | 14 | 1 | -1.86 | 1.86 |
