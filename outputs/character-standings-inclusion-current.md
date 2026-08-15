# Character Standings — inclusion (scoring v2)

- Standings version: `character_standings_inclusion_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_inclusion_name_view_v1` (`outputs/scoring-v2/scoring-v2-inclusion-name-view-ratings.json`)
- Lens / view: `inclusion` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `193` (`9` ranked, `184` without sufficient evidence)
- Comparisons: `565` (mean weight `0.6792`, draw rate `0.023`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-inclusion-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `9` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | le narrateur | 1602 ± 100 | 1502.1 | 186 | 108-73-5 | 209 | +0.077 | 0.3553 |
| 2 | duchesse de Guermantes | 1619 ± 159 | 1460.2 | 45 | 31-12-2 | 183 | 0.0 | 0.0 |
| 3 | baron de Charlus | 1571 ± 146 | 1424.3 | 48 | 28-17-3 | 110 | +0.011 | 0.0705 |
| 4 | Gilberte | 1554 ± 164 | 1389.9 | 37 | 18-15-4 | 57 | +0.05 | 0.0712 |
| 5 | Robert de Saint-Loup | 1486 ± 195 | 1291.3 | 25 | 11-14-0 | 138 | -0.024 | 0.0235 |
| 6 | Odette | 1402 ± 153 | 1248.7 | 59 | 24-35-0 | 124 | -0.094 | 0.1066 |
| 7 | Bloch | 1409 ± 172 | 1236.7 | 37 | 13-24-0 | 64 | -0.152 | 0.2444 |
| 8 | Swann | 1346 ± 122 | 1224.0 | 103 | 33-69-1 | 177 | -0.12 | 0.1975 |
| 9 | Mme Verdurin | 1334 ± 156 | 1178.0 | 43 | 16-27-0 | 78 | -0.055 | 0.0549 |

## Insufficient comparative evidence

The `184` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| le jeune marquis de Cambremer | 1930 ± 392 | 392.2 | 10 | 1 | +0.83 | 0.83 |
| M. de Stermaria | 1831 ± 463 | 463.4 | 2 | 4 | 0.0 | 0.0 |
| Victurnien | 1816 ± 381 | 380.7 | 6 | 2 | +0.725 | 0.725 |
| Mlle d'Oloron | 1782 ± 373 | 373.2 | 8 | 2 | +0.85 | 0.85 |
| Arnulphe | 1762 ± 427 | 427.0 | 3 | 1 | +0.7 | 0.7 |
| la grand-mère | 1755 ± 244 | 243.9 | 17 | 48 | +0.047 | 0.0792 |
| Brichot | 1748 ± 484 | 483.5 | 4 | 17 | 0.0 | 0.0 |
| Mme d'Arpajon | 1739 ± 486 | 486.1 | 4 | 10 | 0.0 | 0.0 |
| Albertine | 1723 ± 245 | 245.2 | 18 | 126 | -0.013 | 0.0618 |
| la reine de Naples | 1722 ± 485 | 484.6 | 4 | 4 | 0.0 | 0.0 |
| Rachel | 1713 ± 418 | 418.0 | 6 | 29 | +0.025 | 0.0248 |
| Bibi | 1712 ± 524 | 523.5 | 1 | 1 | 0.0 | 0.0 |
| princesse Sherbatoff | 1708 ± 524 | 524.3 | 1 | 3 | 0.0 | 0.0 |
| M. d'Argencourt | 1707 ± 493 | 493.0 | 4 | 10 | 0.0 | 0.0 |
| prince Von | 1700 ± 532 | 531.8 | 1 | 5 | 0.0 | 0.0 |
| Mme Sazerat | 1688 ± 505 | 504.9 | 3 | 4 | 0.0 | 0.0 |
| le bâtonnier | 1684 ± 544 | 544.4 | 1 | 5 | 0.0 | 0.0 |
| comte de Forcheville | 1680 ± 283 | 283.4 | 17 | 28 | +0.033 | 0.0886 |
| Mlle de Stermaria | 1676 ± 553 | 552.6 | 1 | 4 | 0.0 | 0.0 |
| le directeur | 1675 ± 552 | 552.3 | 1 | 6 | 0.0 | 0.0 |
| Octave | 1672 ± 439 | 439.2 | 3 | 3 | 0.0 | 0.0 |
| le peintre | 1660 ± 525 | 524.6 | 2 | 8 | 0.0 | 0.0 |
| M. de Palancy | 1658 ± 542 | 541.8 | 1 | 1 | 0.0 | 0.0 |
| M. de Saint-Candé | 1658 ± 542 | 541.8 | 1 | 1 | 0.0 | 0.0 |
| Andrée | 1636 ± 376 | 376.2 | 5 | 25 | 0.0 | 0.0 |
| Eulalie | 1633 ± 542 | 541.6 | 2 | 3 | +0.533 | 0.5333 |
| le grand-père du narrateur | 1632 ± 375 | 375.4 | 6 | 11 | 0.0 | 0.0 |
| Lady Israels | 1631 ± 555 | 555.3 | 1 | 1 | 0.0 | 0.0 |
| duc de Châtellerault | 1627 ± 386 | 385.9 | 3 | 4 | 0.0 | 0.0 |
| Mme de Mortemart | 1626 ± 549 | 548.8 | 2 | 1 | 0.0 | 0.0 |
| princesse de Nassau | 1620 ± 565 | 564.9 | 1 | 1 | 0.0 | 0.0 |
| M. de Crécy | 1618 ± 555 | 554.7 | 2 | 1 | 0.0 | 0.0 |
| Mme Bontemps | 1613 ± 453 | 452.7 | 3 | 13 | +0.128 | 0.1277 |
| colonel de Froberville | 1611 ± 563 | 562.9 | 2 | 2 | 0.0 | 0.0 |
| Elstir | 1609 ± 422 | 422.4 | 3 | 18 | 0.0 | 0.0 |
| baron de Guermantes | 1606 ± 578 | 577.5 | 1 | 2 | 0.0 | 0.0 |
| marquis de Palancy | 1598 ± 577 | 576.9 | 1 | 2 | 0.0 | 0.0 |
| Mlle Bloch | 1596 ± 588 | 588.1 | 1 | 1 | +0.55 | 0.55 |
| Norpois | 1594 ± 225 | 224.9 | 18 | 54 | +0.013 | 0.0133 |
| princesse de Caprarola | 1594 ± 583 | 583.4 | 1 | 1 | 0.0 | 0.0 |
| ma grand'tante | 1590 ± 590 | 589.6 | 1 | 4 | 0.0 | 0.0 |
| Mme de Montmorency | 1581 ± 596 | 595.7 | 1 | 1 | 0.0 | 0.0 |
| marquis de Bréauté | 1572 ± 400 | 400.0 | 5 | 17 | 0.0 | 0.0 |
| duc de Guermantes | 1562 ± 202 | 201.6 | 23 | 97 | 0.0 | 0.0 |
| Mme de Franquetot | 1551 ± 625 | 625.4 | 1 | 1 | 0.0 | 0.0 |
| Mlle Vinteuil | 1547 ± 629 | 629.1 | 1 | 8 | 0.0 | 0.0 |
| prince de Guermantes | 1541 ± 258 | 257.5 | 13 | 13 | +0.058 | 0.0577 |
| Dechambre | 1530 ± 650 | 650.0 | 1 | 1 | 0.0 | 0.0 |
| Jupien | 1525 ± 451 | 451.2 | 3 | 15 | +0.184 | 0.184 |
| Mme de Villeparisis | 1524 ± 208 | 208.1 | 20 | 73 | 0.0 | 0.0 |
| Mme de Surgis | 1522 ± 333 | 333.0 | 7 | 9 | 0.0 | 0.0 |
| Morel | 1513 ± 219 | 219.4 | 20 | 35 | -0.041 | 0.0414 |
| princesse de Parme | 1507 ± 340 | 340.1 | 5 | 36 | 0.0 | 0.0 |
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
| M. Verdurin | 1500 ± 264 | 263.9 | 12 | 32 | 0.0 | 0.0 |
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
| Mme de Marsantes | 1500 ± 272 | 272.0 | 10 | 21 | 0.0 | 0.0 |
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
| marquise de Gallardon | 1495 ± 286 | 285.8 | 9 | 10 | -0.095 | 0.245 |
| Mme Leroi | 1488 ± 478 | 477.9 | 2 | 6 | 0.0 | 0.0 |
| oncle Adolphe | 1480 ± 398 | 397.8 | 3 | 4 | -0.45 | 0.45 |
| Aimé | 1469 ± 648 | 648.3 | 1 | 9 | 0.0 | 0.0 |
| capitaine de Borodino | 1463 ± 426 | 425.7 | 3 | 5 | -0.136 | 0.136 |
| docteur du Boulbon | 1452 ± 630 | 629.5 | 1 | 4 | 0.0 | 0.0 |
| princesse de Guermantes | 1450 ± 263 | 262.7 | 10 | 19 | -0.038 | 0.0379 |
| Gisèle | 1443 ± 451 | 451.4 | 3 | 4 | -0.38 | 0.38 |
| Mme Cottard | 1424 ± 274 | 274.5 | 10 | 15 | +0.001 | 0.0947 |
| docteur Cottard | 1410 ± 246 | 246.2 | 13 | 37 | 0.0 | 0.0 |
| Israël | 1404 ± 588 | 587.9 | 1 | 1 | 0.0 | 0.0 |
| M. Nissim Bernard | 1404 ± 588 | 588.1 | 1 | 6 | 0.0 | 0.0 |
| princesse Mathilde | 1396 ± 579 | 578.9 | 1 | 2 | 0.0 | 0.0 |
| M. Pierre | 1394 ± 578 | 577.5 | 1 | 4 | -0.41 | 0.41 |
| la mère du narrateur | 1392 ± 228 | 228.2 | 17 | 28 | -0.108 | 0.1618 |
| Alix | 1391 ± 574 | 573.8 | 1 | 4 | 0.0 | 0.0 |
| Legrandin | 1387 ± 495 | 495.2 | 2 | 23 | 0.0 | 0.0 |
| le pianiste | 1383 ± 418 | 418.0 | 3 | 5 | 0.0 | 0.0 |
| général de Monserfeuil | 1378 ± 562 | 562.3 | 1 | 2 | 0.0 | 0.0 |
| le père du narrateur | 1366 ± 278 | 277.7 | 12 | 21 | -0.11 | 0.1095 |
| Mme Putbus | 1364 ± 566 | 565.9 | 1 | 1 | 0.0 | 0.0 |
| princesse d'Orvillers | 1364 ± 566 | 565.9 | 1 | 1 | 0.0 | 0.0 |
| prince de Foix | 1359 ± 336 | 336.0 | 6 | 5 | -0.15 | 0.15 |
| le petit Cambremer | 1348 ± 536 | 536.0 | 2 | 1 | 0.0 | 0.0 |
| princesse de Silistrie | 1348 ± 536 | 536.0 | 2 | 1 | 0.0 | 0.0 |
| comtesse Molé | 1342 ± 284 | 283.7 | 8 | 6 | -0.307 | 0.3067 |
| princesse de Luxembourg | 1332 ± 516 | 515.5 | 3 | 4 | 0.0 | 0.0 |
| M. Bontemps | 1324 ± 519 | 518.8 | 2 | 4 | 0.0 | 0.0 |
| Françoise | 1322 ± 370 | 369.9 | 7 | 61 | -0.013 | 0.0128 |
| Mme de Cambremer | 1316 ± 245 | 244.9 | 17 | 22 | -0.146 | 0.2082 |
| marquise de Saint-Euverte | 1315 ± 266 | 265.7 | 12 | 9 | -0.167 | 0.1667 |
| Mme de Souvré | 1293 ± 364 | 364.1 | 6 | 3 | -0.547 | 0.5467 |
| Mme Blandais | 1279 ± 512 | 511.6 | 2 | 2 | -0.575 | 0.575 |
| duc de Guastalla | 1276 ± 487 | 486.7 | 3 | 1 | -0.65 | 0.65 |
| Mme de Chaussepierre | 1266 ± 360 | 359.6 | 6 | 2 | -0.41 | 0.41 |
| Bergotte | 1266 ± 491 | 491.0 | 3 | 27 | 0.0 | 0.0 |
| la Berma | 1258 ± 366 | 366.3 | 8 | 13 | -0.143 | 0.1431 |
| Bloch père | 1255 ± 480 | 480.5 | 4 | 7 | -0.15 | 0.15 |
| Mme de Valcourt | 1251 ± 350 | 349.7 | 9 | 1 | -0.8 | 0.8 |
| Mme d'Hunolstein | 1249 ± 468 | 468.1 | 4 | 1 | -1.6 | 1.6 |
| marquis de Cambremer | 1244 ± 359 | 359.4 | 7 | 4 | -0.2 | 0.2 |
| tante Léonie | 1237 ± 401 | 400.9 | 7 | 9 | -0.182 | 0.1822 |
| la marquise douairière de Cambremer | 1223 ± 472 | 472.1 | 4 | 5 | -0.34 | 0.34 |
| Mme Blatin | 1192 ± 452 | 452.2 | 4 | 3 | -0.48 | 0.48 |
| le vicomte de Courvoisier | 1137 ± 417 | 416.8 | 7 | 1 | -1.8 | 1.8 |
| M. d'Herweck | 1123 ± 413 | 413.3 | 7 | 2 | -1.6 | 1.6 |
| Saniette | 1091 ± 316 | 315.5 | 16 | 12 | -0.508 | 0.5083 |
