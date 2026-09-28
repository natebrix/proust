# Character Standings — inclusion (scoring v2)

- Standings version: `character_standings_inclusion_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_inclusion_name_view_v1` (`outputs/scoring-v2-enrichment/scoring-v2-inclusion-name-view-ratings.json`)
- Lens / view: `inclusion` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`8` ranked, `185` without sufficient evidence)
- Comparisons: `565` (mean weight `0.5946`, draw rate `0.023`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `enrichment`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2-enrichment/scoring-v2-inclusion-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `8` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | le narrateur | 1598 ± 101 | 1496.7 | 186 | 108-73-5 | 209 | +0.077 | 0.3553 |
| 2 | duchesse de Guermantes | 1613 ± 162 | 1451.1 | 45 | 31-12-2 | 183 | 0.0 | 0.0 |
| 3 | baron de Charlus | 1562 ± 155 | 1406.6 | 48 | 28-17-3 | 110 | +0.011 | 0.0705 |
| 4 | Gilberte | 1517 ± 181 | 1336.2 | 37 | 18-15-4 | 57 | +0.05 | 0.0712 |
| 5 | Odette | 1417 ± 157 | 1259.7 | 59 | 24-35-0 | 124 | -0.094 | 0.1066 |
| 6 | Bloch | 1412 ± 174 | 1237.4 | 37 | 13-24-0 | 64 | -0.152 | 0.2444 |
| 7 | Swann | 1359 ± 125 | 1234.0 | 103 | 33-69-1 | 177 | -0.12 | 0.1975 |
| 8 | Mme Verdurin | 1357 ± 168 | 1189.0 | 43 | 16-27-0 | 78 | -0.055 | 0.0549 |

## Insufficient comparative evidence

The `185` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| le jeune marquis de Cambremer | 1918 ± 394 | 393.5 | 10 | 1 | +0.83 | 0.83 |
| Victurnien | 1787 ± 411 | 411.3 | 6 | 2 | +0.725 | 0.725 |
| M. de Stermaria | 1774 ± 501 | 501.3 | 2 | 4 | 0.0 | 0.0 |
| la grand-mère | 1752 ± 257 | 256.7 | 17 | 48 | +0.047 | 0.0792 |
| Brichot | 1736 ± 494 | 493.7 | 4 | 17 | 0.0 | 0.0 |
| Mlle d'Oloron | 1732 ± 429 | 428.7 | 8 | 2 | +0.85 | 0.85 |
| Mme d'Arpajon | 1721 ± 501 | 501.1 | 4 | 10 | 0.0 | 0.0 |
| Bibi | 1711 ± 524 | 523.6 | 1 | 1 | 0.0 | 0.0 |
| la reine de Naples | 1707 ± 496 | 495.9 | 4 | 4 | 0.0 | 0.0 |
| princesse Sherbatoff | 1707 ± 525 | 524.7 | 1 | 3 | 0.0 | 0.0 |
| Arnulphe | 1705 ± 483 | 482.8 | 3 | 1 | +0.7 | 0.7 |
| prince Von | 1699 ± 532 | 532.0 | 1 | 5 | 0.0 | 0.0 |
| comte de Forcheville | 1695 ± 316 | 316.1 | 17 | 28 | +0.033 | 0.0886 |
| M. d'Argencourt | 1690 ± 507 | 506.8 | 4 | 10 | 0.0 | 0.0 |
| le bâtonnier | 1686 ± 544 | 544.0 | 1 | 5 | 0.0 | 0.0 |
| le directeur | 1677 ± 552 | 551.8 | 1 | 6 | 0.0 | 0.0 |
| Albertine | 1676 ± 254 | 253.7 | 18 | 126 | -0.013 | 0.0618 |
| Mlle de Stermaria | 1675 ± 553 | 552.6 | 1 | 4 | 0.0 | 0.0 |
| Octave | 1667 ± 481 | 481.3 | 3 | 3 | 0.0 | 0.0 |
| le peintre | 1664 ± 522 | 522.1 | 2 | 8 | 0.0 | 0.0 |
| M. de Palancy | 1663 ± 539 | 539.4 | 1 | 1 | 0.0 | 0.0 |
| M. de Saint-Candé | 1663 ± 539 | 539.4 | 1 | 1 | 0.0 | 0.0 |
| Mme Sazerat | 1651 ± 534 | 533.6 | 3 | 4 | 0.0 | 0.0 |
| Rachel | 1648 ± 462 | 462.3 | 6 | 29 | +0.025 | 0.0248 |
| Eulalie | 1644 ± 534 | 533.9 | 2 | 3 | +0.533 | 0.5333 |
| Lady Israels | 1636 ± 553 | 552.6 | 1 | 1 | 0.0 | 0.0 |
| Andrée | 1630 ± 396 | 396.4 | 5 | 25 | 0.0 | 0.0 |
| duc de Châtellerault | 1626 ± 386 | 385.6 | 3 | 4 | 0.0 | 0.0 |
| princesse de Nassau | 1622 ± 564 | 564.3 | 1 | 1 | 0.0 | 0.0 |
| colonel de Froberville | 1614 ± 562 | 561.6 | 2 | 2 | 0.0 | 0.0 |
| marquis de Palancy | 1610 ± 569 | 568.7 | 1 | 2 | 0.0 | 0.0 |
| le grand-père du narrateur | 1609 ± 381 | 381.3 | 6 | 11 | 0.0 | 0.0 |
| baron de Guermantes | 1606 ± 578 | 577.5 | 1 | 2 | 0.0 | 0.0 |
| marquis de Bréauté | 1605 ± 441 | 440.8 | 5 | 17 | 0.0 | 0.0 |
| princesse de Caprarola | 1600 ± 578 | 578.5 | 1 | 1 | 0.0 | 0.0 |
| Elstir | 1599 ± 416 | 416.5 | 3 | 18 | 0.0 | 0.0 |
| Mlle Bloch | 1596 ± 588 | 588.1 | 1 | 1 | +0.55 | 0.55 |
| ma grand'tante | 1593 ± 588 | 587.8 | 1 | 4 | 0.0 | 0.0 |
| Mme de Mortemart | 1591 ± 586 | 586.1 | 2 | 1 | 0.0 | 0.0 |
| Mme Bontemps | 1589 ± 505 | 504.8 | 3 | 13 | +0.128 | 0.1277 |
| M. de Crécy | 1588 ± 588 | 588.1 | 2 | 1 | 0.0 | 0.0 |
| Norpois | 1579 ± 232 | 231.6 | 18 | 54 | +0.013 | 0.0133 |
| Mme de Montmorency | 1573 ± 602 | 601.8 | 1 | 1 | 0.0 | 0.0 |
| duc de Guermantes | 1565 ± 206 | 206.3 | 23 | 97 | 0.0 | 0.0 |
| Mme de Franquetot | 1549 ± 627 | 626.9 | 1 | 1 | 0.0 | 0.0 |
| Mlle Vinteuil | 1547 ± 629 | 628.9 | 1 | 8 | 0.0 | 0.0 |
| prince de Guermantes | 1532 ± 288 | 287.7 | 13 | 13 | +0.058 | 0.0577 |
| Mme de Villeparisis | 1532 ± 215 | 214.8 | 20 | 73 | 0.0 | 0.0 |
| Dechambre | 1530 ± 649 | 649.4 | 1 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1530 ± 340 | 340.0 | 7 | 9 | 0.0 | 0.0 |
| Mme de Marsantes | 1521 ± 280 | 280.1 | 10 | 21 | 0.0 | 0.0 |
| Jupien | 1519 ± 450 | 450.3 | 3 | 15 | +0.184 | 0.184 |
| marquise de Gallardon | 1510 ± 294 | 294.1 | 9 | 10 | -0.095 | 0.245 |
| princesse de Parme | 1509 ± 339 | 339.1 | 5 | 36 | 0.0 | 0.0 |
| capitaine de Borodino | 1506 ± 452 | 452.4 | 3 | 5 | -0.136 | 0.136 |
| Antoine | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Céleste Albaret | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| Céline | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dieulafoy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dreyfus | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dumont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Flora | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Gibergue | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Larivière | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Léa | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Barrère | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Ski | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. Swann, le père | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vallenères | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vinteuil | 1500 ± 700 | 700.0 | 0 | 9 | 0.0 | 0.0 |
| M. de Beautreillis | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Bornier | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| M. de Chaussepierre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Courgivaux | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Goncourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Grouchy | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| M. de Luxembourg | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Vaudémont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Vaugoubert | 1500 ± 700 | 700.0 | 0 | 7 | 0.0 | 0.0 |
| Madame d'Ambresac | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Majesté | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Marie Gineste | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Maurice | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Mlle d'Éporcheville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme G... | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme d'Heudicourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Citri | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Varambon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Vaugoubert | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Villebon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Potain | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Poullein | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Rosemonde | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Rémi | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Sainte-Beuve | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Théodore | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Victor | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| commandant Duroc | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| comte Arnulphe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comte de Paris | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comtesse de Monteriender | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| cousine Poictiers | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| docteur Percepied | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc d'Aumale | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc de La Trémoïlle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc de Sidonia | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Létourville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| grand-duc Wladimir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| grand-duc héritier de Luxembourg | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| général de Froberville | 1500 ± 700 | 700.0 | 0 | 8 | 0.0 | 0.0 |
| l'amie de Mlle Vinteuil | 1500 ± 700 | 700.0 | 0 | 7 | 0.0 | 0.0 |
| l'empereur Guillaume | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la cousine d'Oriane | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| la jeune ouvriere | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la marquise | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| le prince de Faffenheim | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le professeur E… | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| le roi Théodose | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| les Iéna | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Surgis | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquise d'Amoncourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince Foggi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| prince de Faffenheim | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| prince de Sagan | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince des Laumes | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| princesse d'Épinay | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| professeur E... | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| spécialiste X... | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomtesse d'Égremont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomtesse de Saint-Fiacre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Morel | 1499 ± 229 | 228.8 | 20 | 35 | -0.041 | 0.0414 |
| Mme Leroi | 1492 ± 477 | 476.6 | 2 | 6 | 0.0 | 0.0 |
| Robert de Saint-Loup | 1490 ± 202 | 202.5 | 25 | 138 | -0.024 | 0.0235 |
| oncle Adolphe | 1488 ± 397 | 397.3 | 3 | 4 | -0.45 | 0.45 |
| M. Verdurin | 1482 ± 279 | 278.8 | 12 | 32 | 0.0 | 0.0 |
| Aimé | 1468 ± 646 | 646.0 | 1 | 9 | 0.0 | 0.0 |
| docteur du Boulbon | 1451 ± 629 | 628.8 | 1 | 4 | 0.0 | 0.0 |
| Mme Cottard | 1444 ± 293 | 293.4 | 10 | 15 | +0.001 | 0.0947 |
| princesse de Guermantes | 1433 ± 271 | 270.7 | 10 | 19 | -0.038 | 0.0379 |
| docteur Cottard | 1425 ± 258 | 258.1 | 13 | 37 | 0.0 | 0.0 |
| Gisèle | 1423 ± 481 | 480.7 | 3 | 4 | -0.38 | 0.38 |
| Israël | 1406 ± 589 | 588.9 | 1 | 1 | 0.0 | 0.0 |
| M. Nissim Bernard | 1404 ± 588 | 588.1 | 1 | 6 | 0.0 | 0.0 |
| la mère du narrateur | 1399 ± 231 | 230.8 | 17 | 28 | -0.108 | 0.1618 |
| princesse Mathilde | 1397 ± 580 | 580.1 | 1 | 2 | 0.0 | 0.0 |
| Legrandin | 1394 ± 494 | 493.8 | 2 | 23 | 0.0 | 0.0 |
| M. Pierre | 1394 ± 578 | 577.5 | 1 | 4 | -0.41 | 0.41 |
| Alix | 1392 ± 574 | 574.3 | 1 | 4 | 0.0 | 0.0 |
| le petit Cambremer | 1385 ± 574 | 574.3 | 2 | 1 | 0.0 | 0.0 |
| princesse de Silistrie | 1385 ± 574 | 574.3 | 2 | 1 | 0.0 | 0.0 |
| général de Monserfeuil | 1377 ± 562 | 561.8 | 1 | 2 | 0.0 | 0.0 |
| Mme Putbus | 1369 ± 568 | 567.8 | 1 | 1 | 0.0 | 0.0 |
| princesse d'Orvillers | 1369 ± 568 | 567.8 | 1 | 1 | 0.0 | 0.0 |
| princesse de Luxembourg | 1368 ± 544 | 544.3 | 3 | 4 | 0.0 | 0.0 |
| le pianiste | 1363 ± 462 | 462.3 | 3 | 5 | 0.0 | 0.0 |
| prince de Foix | 1362 ± 336 | 336.0 | 6 | 5 | -0.15 | 0.15 |
| M. Bontemps | 1359 ± 552 | 552.2 | 2 | 4 | 0.0 | 0.0 |
| le père du narrateur | 1358 ± 300 | 299.5 | 12 | 21 | -0.11 | 0.1095 |
| Mme de Cambremer | 1355 ± 260 | 260.0 | 17 | 22 | -0.146 | 0.2082 |
| Françoise | 1337 ± 375 | 375.3 | 7 | 61 | -0.013 | 0.0128 |
| la Berma | 1320 ± 389 | 388.9 | 8 | 13 | -0.143 | 0.1431 |
| comtesse Molé | 1309 ± 303 | 303.0 | 8 | 6 | -0.307 | 0.3067 |
| Mme de Chaussepierre | 1302 ± 429 | 428.6 | 6 | 2 | -0.41 | 0.41 |
| Mme de Souvré | 1297 ± 365 | 364.9 | 6 | 3 | -0.547 | 0.5467 |
| marquise de Saint-Euverte | 1288 ± 283 | 282.6 | 12 | 9 | -0.167 | 0.1667 |
| Mme de Valcourt | 1286 ± 415 | 415.2 | 9 | 1 | -0.8 | 0.8 |
| duc de Guastalla | 1281 ± 489 | 488.9 | 3 | 1 | -0.65 | 0.65 |
| Mme Blandais | 1278 ± 512 | 512.3 | 2 | 2 | -0.575 | 0.575 |
| tante Léonie | 1278 ± 448 | 448.5 | 7 | 9 | -0.182 | 0.1822 |
| marquis de Cambremer | 1272 ± 414 | 413.5 | 7 | 4 | -0.2 | 0.2 |
| Bergotte | 1268 ± 492 | 492.2 | 3 | 27 | 0.0 | 0.0 |
| Bloch père | 1250 ± 477 | 476.7 | 4 | 7 | -0.15 | 0.15 |
| Mme d'Hunolstein | 1246 ± 467 | 466.7 | 4 | 1 | -1.6 | 1.6 |
| le vicomte de Courvoisier | 1219 ± 458 | 458.2 | 7 | 1 | -1.8 | 1.8 |
| la marquise douairière de Cambremer | 1216 ± 472 | 472.3 | 4 | 5 | -0.34 | 0.34 |
| Mme Blatin | 1193 ± 452 | 451.6 | 4 | 3 | -0.48 | 0.48 |
| M. d'Herweck | 1118 ± 414 | 413.5 | 7 | 2 | -1.6 | 1.6 |
| Saniette | 1095 ± 315 | 314.8 | 16 | 12 | -0.508 | 0.5083 |
