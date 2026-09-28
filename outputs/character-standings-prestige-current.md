# Character Standings — prestige (scoring v2)

- Standings version: `character_standings_prestige_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_prestige_name_view_v1` (`outputs/scoring-v2-enrichment/scoring-v2-prestige-name-view-ratings.json`)
- Lens / view: `prestige` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`14` ranked, `179` without sufficient evidence)
- Comparisons: `954` (mean weight `0.4732`, draw rate `0.058`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `enrichment`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2-enrichment/scoring-v2-prestige-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `14` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | duchesse de Guermantes | 1706 ± 108 | 1598.2 | 156 | 113-38-5 | 183 | +0.216 | 0.2704 |
| 2 | Morel | 1770 ± 188 | 1582.1 | 43 | 33-9-1 | 35 | +0.206 | 0.2063 |
| 3 | Odette | 1710 ± 137 | 1573.2 | 77 | 46-24-7 | 124 | +0.107 | 0.1687 |
| 4 | le narrateur | 1648 ± 128 | 1520.9 | 102 | 59-38-5 | 209 | +0.061 | 0.1003 |
| 5 | Gilberte | 1630 ± 154 | 1476.1 | 53 | 23-29-1 | 57 | +0.085 | 0.174 |
| 6 | Mme Verdurin | 1586 ± 127 | 1459.1 | 104 | 53-44-7 | 78 | +0.129 | 0.2362 |
| 7 | baron de Charlus | 1550 ± 110 | 1440.0 | 132 | 65-54-13 | 110 | +0.032 | 0.269 |
| 8 | Mme de Villeparisis | 1561 ± 148 | 1412.8 | 58 | 23-28-7 | 73 | -0.016 | 0.2053 |
| 9 | Robert de Saint-Loup | 1553 ± 144 | 1409.3 | 74 | 30-44-0 | 138 | +0.047 | 0.1162 |
| 10 | Swann | 1530 ± 134 | 1395.3 | 114 | 44-60-10 | 177 | +0.024 | 0.1659 |
| 11 | Norpois | 1545 ± 196 | 1348.5 | 38 | 21-16-1 | 54 | +0.101 | 0.1274 |
| 12 | Brichot | 1533 ± 199 | 1334.3 | 32 | 12-18-2 | 17 | -0.008 | 0.3518 |
| 13 | Bloch | 1479 ± 173 | 1305.7 | 45 | 16-26-3 | 64 | -0.046 | 0.0934 |
| 14 | duc de Guermantes | 1467 ± 168 | 1299.7 | 56 | 20-35-1 | 97 | -0.012 | 0.0614 |

## Insufficient comparative evidence

The `179` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| Alix | 1865 ± 442 | 442.4 | 4 | 4 | +0.2 | 0.2 |
| Mlle d'Oloron | 1848 ± 365 | 365.1 | 19 | 2 | +1.27 | 1.27 |
| docteur du Boulbon | 1821 ± 467 | 467.3 | 2 | 4 | +0.188 | 0.1875 |
| le peintre | 1805 ± 310 | 310.1 | 8 | 8 | +0.306 | 0.3063 |
| Bergotte | 1779 ± 468 | 467.6 | 4 | 27 | +0.037 | 0.1467 |
| Legrandin | 1760 ± 298 | 297.6 | 18 | 23 | +0.013 | 0.1435 |
| la mère du narrateur | 1751 ± 280 | 280.5 | 10 | 28 | +0.005 | 0.0482 |
| Madame d'Ambresac | 1740 ± 492 | 492.1 | 2 | 1 | +0.75 | 0.75 |
| M. de Chaussepierre | 1736 ± 465 | 464.6 | 4 | 1 | +1.78 | 1.78 |
| Aimé | 1732 ± 388 | 388.3 | 5 | 9 | +0.064 | 0.0644 |
| Mme de Chaussepierre | 1729 ± 471 | 470.7 | 4 | 2 | +0.82 | 0.82 |
| Lady Israels | 1721 ± 531 | 530.6 | 1 | 1 | 0.0 | 0.0 |
| Octave | 1717 ± 431 | 431.4 | 9 | 3 | +0.547 | 0.5467 |
| l'amie de Mlle Vinteuil | 1715 ± 421 | 421.1 | 5 | 7 | 0.0 | 0.0 |
| prince de Guermantes | 1707 ± 352 | 351.6 | 9 | 13 | +0.131 | 0.1308 |
| docteur Percepied | 1702 ± 539 | 538.9 | 2 | 1 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1692 ± 534 | 534.2 | 3 | 1 | +0.55 | 0.55 |
| M. Vinteuil | 1691 ± 351 | 350.6 | 8 | 9 | +0.078 | 0.3 |
| duc de Sidonia | 1684 ± 535 | 534.8 | 1 | 1 | 0.0 | 0.0 |
| le professeur E… | 1684 ± 535 | 534.8 | 1 | 2 | 0.0 | 0.0 |
| le petit Cambremer | 1683 ± 429 | 429.0 | 8 | 1 | +0.8 | 0.8 |
| Mlle Vinteuil | 1677 ± 452 | 451.5 | 4 | 8 | 0.0 | 0.0 |
| la marquise | 1662 ± 574 | 574.2 | 1 | 3 | +0.2 | 0.2 |
| Rosemonde | 1651 ± 547 | 546.8 | 2 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1650 ± 241 | 241.1 | 19 | 9 | +0.347 | 0.5133 |
| Rachel | 1639 ± 224 | 224.2 | 24 | 29 | +0.041 | 0.2003 |
| Andrée | 1629 ± 302 | 301.7 | 17 | 25 | +0.043 | 0.0928 |
| le vicomte de Courvoisier | 1626 ± 575 | 574.7 | 2 | 1 | 0.0 | 0.0 |
| M. Nissim Bernard | 1626 ± 560 | 560.5 | 2 | 6 | +0.1 | 0.1 |
| M. de Vaudémont | 1620 ± 567 | 567.0 | 1 | 1 | +0.7 | 0.7 |
| la Berma | 1608 ± 249 | 249.0 | 17 | 13 | -0.015 | 0.5169 |
| tante Léonie | 1605 ± 401 | 401.1 | 3 | 9 | +0.189 | 0.1889 |
| M. Bontemps | 1602 ± 495 | 494.9 | 3 | 4 | +0.425 | 0.425 |
| comtesse Molé | 1602 ± 313 | 313.0 | 14 | 6 | -0.01 | 0.5767 |
| princesse de Guermantes | 1601 ± 225 | 224.6 | 21 | 19 | +0.213 | 0.3916 |
| la grand-mère | 1601 ± 235 | 234.8 | 18 | 48 | +0.053 | 0.1185 |
| M. Verdurin | 1596 ± 231 | 230.7 | 25 | 32 | 0.0 | 0.0 |
| Elstir | 1596 ± 430 | 430.2 | 3 | 18 | +0.094 | 0.0944 |
| le père du narrateur | 1586 ± 294 | 293.8 | 14 | 21 | -0.001 | 0.0676 |
| comte de Forcheville | 1582 ± 238 | 238.5 | 18 | 28 | +0.087 | 0.0875 |
| Jupien | 1574 ± 249 | 249.2 | 19 | 15 | +0.099 | 0.208 |
| Mme Bontemps | 1569 ± 331 | 331.0 | 10 | 13 | +0.18 | 0.18 |
| duc de La Trémoïlle | 1563 ± 610 | 610.5 | 1 | 1 | 0.0 | 0.0 |
| Maurice | 1560 ± 484 | 484.2 | 2 | 2 | 0.0 | 0.0 |
| la marquise douairière de Cambremer | 1547 ± 407 | 407.4 | 5 | 5 | +0.34 | 0.34 |
| prince des Laumes | 1547 ± 631 | 630.7 | 1 | 1 | 0.0 | 0.0 |
| Mme Cottard | 1542 ± 286 | 285.8 | 12 | 15 | +0.05 | 0.05 |
| Mme Leroi | 1532 ± 362 | 362.4 | 6 | 6 | -0.47 | 0.7033 |
| docteur Cottard | 1530 ± 206 | 206.1 | 30 | 37 | +0.057 | 0.0568 |
| Mme de Valcourt | 1527 ± 534 | 533.6 | 4 | 1 | 0.0 | 0.0 |
| Mme de Marsantes | 1516 ± 282 | 282.5 | 16 | 21 | +0.036 | 0.101 |
| Albertine | 1511 ± 226 | 226.3 | 29 | 126 | +0.01 | 0.0469 |
| marquis de Bréauté | 1509 ± 303 | 302.8 | 13 | 17 | 0.0 | 0.0 |
| colonel de Froberville | 1506 ± 559 | 558.6 | 2 | 2 | 0.0 | 0.0 |
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
| Mme de Franquetot | 1498 ± 585 | 584.6 | 2 | 1 | 0.0 | 0.0 |
| la reine de Naples | 1494 ± 389 | 388.8 | 8 | 4 | 0.0 | 0.0 |
| Françoise | 1480 ± 222 | 221.7 | 22 | 61 | +0.052 | 0.0516 |
| Céleste Albaret | 1478 ± 430 | 430.1 | 4 | 3 | +0.043 | 0.0433 |
| Marie Gineste | 1478 ± 430 | 430.1 | 4 | 2 | +0.065 | 0.065 |
| duchesse de Létourville | 1476 ± 537 | 537.3 | 2 | 1 | 0.0 | 0.0 |
| princesse de Caprarola | 1468 ± 489 | 489.2 | 4 | 1 | 0.0 | 0.0 |
| comte Arnulphe | 1465 ± 503 | 503.4 | 2 | 1 | 0.0 | 0.0 |
| M. d'Herweck | 1453 ± 497 | 497.4 | 3 | 2 | 0.0 | 0.0 |
| Dechambre | 1450 ± 626 | 626.2 | 1 | 1 | 0.0 | 0.0 |
| Victurnien | 1449 ± 488 | 488.2 | 3 | 2 | 0.0 | 0.0 |
| marquis de Palancy | 1448 ± 470 | 470.2 | 4 | 2 | +0.615 | 0.615 |
| princesse de Parme | 1448 ± 213 | 213.1 | 30 | 36 | +0.046 | 0.0875 |
| M. de Beautreillis | 1444 ± 622 | 621.8 | 1 | 1 | 0.0 | 0.0 |
| Mme de Varambon | 1444 ± 622 | 621.8 | 1 | 1 | 0.0 | 0.0 |
| cousine Poictiers | 1444 ± 620 | 620.4 | 1 | 1 | 0.0 | 0.0 |
| le bâtonnier | 1438 ± 386 | 386.2 | 5 | 5 | +0.02 | 0.28 |
| M. de Vaugoubert | 1437 ± 412 | 412.4 | 6 | 7 | +0.237 | 0.2371 |
| baron de Guermantes | 1435 ± 610 | 609.8 | 1 | 2 | 0.0 | 0.0 |
| le grand-père du narrateur | 1430 ± 385 | 385.3 | 4 | 11 | 0.0 | 0.0 |
| le jeune marquis de Cambremer | 1428 ± 603 | 602.9 | 1 | 1 | 0.0 | 0.0 |
| marquise d'Amoncourt | 1426 ± 600 | 599.6 | 1 | 1 | 0.0 | 0.0 |
| duc d'Aumale | 1426 ± 599 | 598.6 | 1 | 1 | 0.0 | 0.0 |
| le directeur | 1424 ± 462 | 461.8 | 3 | 6 | 0.0 | 0.0 |
| Mme de Montmorency | 1424 ± 600 | 599.7 | 1 | 1 | 0.0 | 0.0 |
| Poullein | 1422 ± 596 | 595.6 | 1 | 3 | 0.0 | 0.0 |
| M. d'Argencourt | 1419 ± 372 | 371.6 | 10 | 10 | -0.072 | 0.072 |
| prince d'Agrigente | 1418 ± 594 | 593.8 | 2 | 2 | 0.0 | 0.0 |
| vicomtesse d'Égremont | 1417 ± 590 | 589.9 | 1 | 1 | 0.0 | 0.0 |
| M. Pierre | 1416 ± 588 | 588.1 | 1 | 4 | 0.0 | 0.0 |
| prince de Sagan | 1413 ± 586 | 585.8 | 1 | 1 | 0.0 | 0.0 |
| princesse de Silistrie | 1413 ± 590 | 589.5 | 3 | 1 | 0.0 | 0.0 |
| Mlle Bloch | 1407 ± 588 | 587.8 | 1 | 1 | 0.0 | 0.0 |
| général de Froberville | 1406 ± 473 | 473.3 | 3 | 8 | 0.0 | 0.0 |
| Arnulphe | 1404 ± 582 | 582.1 | 1 | 1 | 0.0 | 0.0 |
| princesse de Luxembourg | 1397 ± 409 | 408.6 | 4 | 4 | -0.125 | 0.125 |
| oncle Adolphe | 1394 ± 581 | 581.3 | 1 | 4 | 0.0 | 0.0 |
| Mme de Souvré | 1392 ± 468 | 468.2 | 4 | 3 | 0.0 | 0.0 |
| général de Monserfeuil | 1391 ± 569 | 569.2 | 1 | 2 | 0.0 | 0.0 |
| Victor | 1389 ± 577 | 577.3 | 1 | 1 | 0.0 | 0.0 |
| Mme Blatin | 1388 ± 566 | 566.1 | 2 | 3 | 0.0 | 0.0 |
| Majesté | 1384 ± 566 | 565.9 | 2 | 1 | -1.64 | 1.64 |
| M. de Crécy | 1380 ± 569 | 568.7 | 2 | 1 | 0.0 | 0.0 |
| M. Swann, le père | 1380 ± 566 | 566.2 | 1 | 1 | 0.0 | 0.0 |
| Larivière | 1379 ± 570 | 569.7 | 1 | 1 | 0.0 | 0.0 |
| Mme de Cambremer | 1378 ± 255 | 255.4 | 27 | 22 | -0.064 | 0.0636 |
| Céline | 1376 ± 563 | 563.2 | 1 | 1 | 0.0 | 0.0 |
| Flora | 1376 ± 563 | 563.2 | 1 | 1 | 0.0 | 0.0 |
| Théodore | 1375 ± 559 | 558.9 | 1 | 1 | 0.0 | 0.0 |
| princesse d'Épinay | 1374 ± 546 | 546.2 | 2 | 2 | 0.0 | 0.0 |
| comte de Paris | 1373 ± 561 | 561.0 | 1 | 1 | 0.0 | 0.0 |
| M. Ski | 1371 ± 562 | 562.0 | 2 | 2 | 0.0 | 0.0 |
| ma grand'tante | 1368 ± 446 | 445.7 | 3 | 4 | 0.0 | 0.0 |
| le prince de Faffenheim | 1360 ± 553 | 552.6 | 2 | 1 | 0.0 | 0.0 |
| princesse Sherbatoff | 1356 ± 545 | 544.8 | 2 | 3 | 0.0 | 0.0 |
| princesse Mathilde | 1351 ± 538 | 538.5 | 3 | 2 | 0.0 | 0.0 |
| Dumont | 1341 ± 558 | 558.5 | 1 | 1 | -1.5 | 1.5 |
| M. Vallenères | 1334 ± 545 | 544.7 | 2 | 1 | -0.5 | 0.5 |
| Mme de Mortemart | 1330 ± 471 | 471.0 | 9 | 1 | -0.8 | 0.8 |
| Mme d'Arpajon | 1316 ± 401 | 400.6 | 12 | 10 | -0.075 | 0.075 |
| duc de Châtellerault | 1312 ± 511 | 511.1 | 2 | 4 | 0.0 | 0.0 |
| Gibergue | 1310 ± 514 | 514.4 | 3 | 2 | -0.375 | 0.375 |
| marquis de Cambremer | 1309 ± 394 | 393.9 | 9 | 4 | -0.15 | 0.15 |
| M. de Stermaria | 1286 ± 501 | 500.8 | 3 | 4 | 0.0 | 0.0 |
| Mlle de Stermaria | 1286 ± 501 | 500.8 | 3 | 4 | 0.0 | 0.0 |
| marquise de Gallardon | 1271 ± 430 | 429.6 | 9 | 10 | -0.155 | 0.155 |
| le roi Théodose | 1271 ± 497 | 496.6 | 4 | 3 | 0.0 | 0.0 |
| le pianiste | 1270 ± 480 | 479.9 | 7 | 5 | -0.15 | 0.15 |
| prince de Foix | 1270 ± 488 | 488.0 | 3 | 5 | 0.0 | 0.0 |
| marquise de Saint-Euverte | 1268 ± 294 | 293.7 | 19 | 9 | -0.684 | 0.8622 |
| Saniette | 1252 ± 300 | 299.7 | 22 | 12 | -0.262 | 0.2617 |
| grand-duc héritier de Luxembourg | 1249 ± 483 | 483.2 | 3 | 2 | -0.7 | 0.7 |
| Bloch père | 1229 ± 460 | 459.5 | 6 | 7 | -0.086 | 0.0857 |
| M. de Goncourt | 1229 ± 456 | 455.5 | 7 | 1 | -0.7 | 0.7 |
| les Iéna | 1228 ± 471 | 470.8 | 5 | 2 | -0.3 | 0.3 |
| capitaine de Borodino | 1168 ± 427 | 426.8 | 7 | 5 | -0.596 | 0.596 |
