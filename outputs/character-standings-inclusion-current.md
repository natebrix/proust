# Character Standings — inclusion (scoring v2)

- Standings version: `character_standings_inclusion_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_inclusion_name_view_v1` (`outputs/scoring-v2/scoring-v2-inclusion-name-view-ratings.json`)
- Lens / view: `inclusion` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `288` (`9` ranked, `279` without sufficient evidence)
- Comparisons: `531` (mean weight `0.5886`, draw rate `0.006`)
- w2: `5.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `foundation`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-inclusion-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `9` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | le narrateur | 1520 ± 101 | 1418.8 | 223 | 121-100-2 | 316 | +0.051 | 0.2198 |
| 2 | Gilberte | 1545 ± 174 | 1371.1 | 33 | 22-11-0 | 76 | +0.008 | 0.0345 |
| 3 | duchesse de Guermantes | 1479 ± 163 | 1315.8 | 41 | 24-17-0 | 199 | -0.004 | 0.004 |
| 4 | Robert de Saint-Loup | 1427 ± 163 | 1264.4 | 44 | 19-25-0 | 168 | -0.011 | 0.0205 |
| 5 | Odette | 1348 ± 154 | 1194.5 | 64 | 24-40-0 | 142 | -0.037 | 0.0711 |
| 6 | Bloch | 1362 ± 184 | 1177.9 | 38 | 11-26-1 | 71 | -0.12 | 0.1775 |
| 7 | Swann | 1287 ± 127 | 1159.9 | 99 | 30-69-0 | 202 | -0.067 | 0.1169 |
| 8 | Mme Verdurin | 1334 ± 181 | 1152.6 | 35 | 13-22-0 | 82 | -0.056 | 0.0743 |
| 9 | Albertine | 1293 ± 167 | 1126.1 | 50 | 11-38-1 | 146 | -0.071 | 0.0901 |

## Insufficient comparative evidence

The `279` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| Mme Sazerat | 1830 ± 476 | 476.5 | 2 | 6 | 0.0 | 0.0 |
| Victurnien | 1827 ± 462 | 462.3 | 4 | 2 | +0.75 | 0.75 |
| baron de Charlus | 1786 ± 235 | 235.4 | 32 | 119 | +0.012 | 0.0118 |
| la mère du narrateur | 1784 ± 387 | 387.1 | 8 | 40 | -0.018 | 0.0175 |
| le père du narrateur | 1717 ± 335 | 334.7 | 11 | 24 | -0.068 | 0.0683 |
| marquis de Cambremer | 1712 ± 507 | 507.4 | 2 | 6 | 0.0 | 0.0 |
| Eulalie | 1701 ± 524 | 523.5 | 2 | 7 | +0.243 | 0.2429 |
| Mlle Vinteuil | 1692 ± 498 | 497.6 | 5 | 15 | 0.0 | 0.0 |
| Céline | 1690 ± 523 | 522.8 | 2 | 2 | 0.0 | 0.0 |
| la reine de Naples | 1680 ± 518 | 517.7 | 2 | 3 | 0.0 | 0.0 |
| Rosemonde | 1680 ± 524 | 523.5 | 2 | 4 | 0.0 | 0.0 |
| M. d'Argencourt | 1674 ± 512 | 512.5 | 4 | 14 | 0.0 | 0.0 |
| Mlle Bloch | 1672 ± 526 | 526.0 | 2 | 1 | +0.75 | 0.75 |
| M. Vinteuil | 1667 ± 533 | 533.1 | 2 | 15 | 0.0 | 0.0 |
| l'amie de Mlle Vinteuil | 1667 ± 516 | 516.4 | 4 | 12 | 0.0 | 0.0 |
| prince Von | 1657 ± 552 | 552.1 | 1 | 3 | 0.0 | 0.0 |
| Bibi | 1654 ± 555 | 554.8 | 1 | 1 | 0.0 | 0.0 |
| princesse Sherbatoff | 1654 ± 428 | 427.7 | 5 | 5 | +0.32 | 0.32 |
| princesse de Luxembourg | 1646 ± 564 | 564.3 | 1 | 6 | 0.0 | 0.0 |
| M. de Stermaria | 1643 ± 560 | 560.0 | 1 | 4 | 0.0 | 0.0 |
| Mme d'Arpajon | 1641 ± 565 | 564.8 | 1 | 8 | 0.0 | 0.0 |
| M. de Marsantes | 1635 ± 571 | 570.9 | 1 | 2 | 0.0 | 0.0 |
| Dostoïevski | 1632 ± 572 | 571.9 | 1 | 1 | 0.0 | 0.0 |
| Coquelin | 1627 ± 574 | 573.7 | 1 | 1 | 0.0 | 0.0 |
| Flora | 1627 ± 572 | 572.2 | 1 | 1 | 0.0 | 0.0 |
| le grand-père du narrateur | 1623 ± 324 | 324.2 | 11 | 16 | -0.048 | 0.0475 |
| Mme Legrandin mère | 1621 ± 580 | 579.9 | 1 | 1 | 0.0 | 0.0 |
| Victoire | 1621 ± 580 | 579.9 | 1 | 1 | 0.0 | 0.0 |
| Lady Israels | 1611 ± 566 | 566.2 | 1 | 1 | 0.0 | 0.0 |
| le peintre | 1611 ± 562 | 562.4 | 2 | 8 | 0.0 | 0.0 |
| prince de Guermantes | 1601 ± 388 | 388.5 | 6 | 22 | 0.0 | 0.0 |
| Françoise | 1595 ± 271 | 270.8 | 13 | 82 | -0.018 | 0.0177 |
| princesse de Nassau | 1594 ± 586 | 585.6 | 1 | 1 | 0.0 | 0.0 |
| duc de Sidonia | 1589 ± 592 | 592.0 | 1 | 1 | 0.0 | 0.0 |
| Mme de Surgis | 1587 ± 499 | 498.9 | 2 | 9 | 0.0 | 0.0 |
| Mme de Sagan | 1581 ± 598 | 597.9 | 1 | 1 | 0.0 | 0.0 |
| Mme de Vaugoubert | 1581 ± 597 | 597.0 | 1 | 2 | 0.0 | 0.0 |
| Herbinger | 1580 ± 599 | 599.1 | 1 | 1 | 0.0 | 0.0 |
| M. de Vaugoubert | 1580 ± 598 | 598.5 | 1 | 9 | 0.0 | 0.0 |
| la Berma | 1578 ± 435 | 434.7 | 3 | 19 | 0.0 | 0.0 |
| Brichot | 1575 ± 427 | 427.4 | 4 | 21 | 0.0 | 0.0 |
| docteur du Boulbon | 1575 ± 604 | 604.3 | 1 | 6 | 0.0 | 0.0 |
| Elstir | 1574 ± 407 | 407.3 | 4 | 29 | 0.0 | 0.0 |
| Lady Rufus Israël | 1573 ± 602 | 602.0 | 1 | 1 | 0.0 | 0.0 |
| le pianiste | 1573 ± 603 | 602.6 | 1 | 3 | 0.0 | 0.0 |
| le bâtonnier | 1572 ± 605 | 604.9 | 1 | 1 | 0.0 | 0.0 |
| Norpois | 1569 ± 321 | 321.0 | 8 | 63 | 0.0 | 0.0 |
| Mme Goupil | 1568 ± 608 | 608.4 | 1 | 2 | 0.0 | 0.0 |
| Léa | 1563 ± 614 | 613.9 | 1 | 4 | 0.0 | 0.0 |
| les La Trémoïlle | 1563 ± 614 | 614.3 | 1 | 1 | 0.0 | 0.0 |
| comte de Forcheville | 1561 ± 295 | 294.9 | 15 | 25 | -0.004 | 0.064 |
| Mme de Charlus | 1560 ± 618 | 617.8 | 1 | 2 | 0.0 | 0.0 |
| Bergotte | 1556 ± 428 | 427.5 | 4 | 36 | 0.0 | 0.0 |
| Mme de Villeparisis | 1550 ± 204 | 204.4 | 24 | 79 | -0.029 | 0.0294 |
| Jupien | 1544 ± 633 | 633.0 | 1 | 18 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1544 ± 633 | 633.0 | 1 | 1 | 0.0 | 0.0 |
| Sarah Bernhardt | 1542 ± 636 | 635.9 | 1 | 1 | 0.0 | 0.0 |
| le jeune prince de Foix | 1542 ± 636 | 635.9 | 1 | 1 | 0.0 | 0.0 |
| marquis de Bréauté | 1540 ± 637 | 637.4 | 1 | 19 | 0.0 | 0.0 |
| duc de Châtellerault | 1537 ± 403 | 403.2 | 3 | 5 | 0.0 | 0.0 |
| Mme Timoléon d'Amoncourt | 1536 ± 642 | 642.3 | 1 | 1 | 0.0 | 0.0 |
| Mme de Franquetot | 1536 ± 642 | 642.3 | 1 | 3 | 0.0 | 0.0 |
| Dechambre | 1531 ± 648 | 647.6 | 1 | 1 | 0.0 | 0.0 |
| Mme de Marsantes | 1529 ± 352 | 351.7 | 6 | 21 | 0.0 | 0.0 |
| Andrée | 1526 ± 375 | 375.3 | 6 | 31 | -0.024 | 0.0242 |
| la grand-mère | 1523 ± 218 | 218.3 | 21 | 80 | -0.049 | 0.0489 |
| Mme Cottard | 1523 ± 540 | 540.3 | 2 | 11 | 0.0 | 0.0 |
| Aimé | 1518 ± 448 | 448.1 | 4 | 18 | 0.0 | 0.0 |
| Mme Bontemps | 1516 ± 447 | 447.3 | 3 | 13 | +0.115 | 0.1154 |
| Alix | 1511 ± 497 | 497.0 | 2 | 3 | 0.0 | 0.0 |
| Antoine | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Arnulphe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Beauserfeuil | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Bismarck | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Bloch père | 1500 ± 700 | 700.0 | 0 | 8 | 0.0 | 0.0 |
| Cartier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Charcot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Céleste Albaret | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| D'Annunzio | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dreyfus | 1500 ± 700 | 700.0 | 0 | 7 | 0.0 | 0.0 |
| Duroc | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Esther | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Goncourt | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Gribelin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Lady Israël | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Léonor de Cambremer | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Létourville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| L’excellent écrivain G… | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Arthur Meyer | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Barrère | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Carnot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Grevy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Pierre | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. Reinach | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Ski | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. Swann, le père | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vibert | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. d'Herweck | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. d'Orsan | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Beauserfeuil | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Bornier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Chateaubriand | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. de Chevregny | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Courgivaux | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Crécy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Goncourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de La Rochefoucauld | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Luxembourg | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Miribel | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Madame Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Madame d'Ambresac | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Maeterlinck | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Manet | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Marie | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Marie Gineste | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Marie-Aynard | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle d'Oloron | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Mlle de Stermaria | 1500 ± 700 | 700.0 | 0 | 5 | 0.0 | 0.0 |
| Mlle de l’Orgeville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Blatin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Carnot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Féré | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Leroi | 1500 ± 700 | 700.0 | 0 | 5 | 0.0 | 0.0 |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Putbus | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Trombert | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme d'Heudicourt | 1500 ± 700 | 700.0 | 0 | 5 | 0.0 | 0.0 |
| Mme de Chaussepierre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Grouchy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Montmorency | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Morienval | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Rochechouart | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Simiane | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Stermaria | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Sévigné | 1500 ± 700 | 700.0 | 0 | 4 | 0.0 | 0.0 |
| Mme de Villebon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Monsieur Vallenères | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Napoléon III | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Octave | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Picquart | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Poullein | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Prince Henri d'Orléans | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Rémi | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| Sir Rufus Israël | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Thibaud | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Théodore | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Théodose Cadet | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Victurnienne | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Vigny | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| baron de Guermantes | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| colonel Picquart | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| colonel de Froberville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comte de Paris | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| comtesse G… | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comtesse Molé | 1500 ± 700 | 700.0 | 0 | 6 | 0.0 | 0.0 |
| comtesse de Monteriender | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comtesse douairière d'Argencourt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| cousine Poictiers | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| d'Orléans | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| docteur Dieulafoy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| docteur Percepied | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc d'Aumale | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| duc de Chartres | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duc de Poictiers | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Gallardon douairière | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de La Rochefoucauld | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de La Trémoïlle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Luxembourg | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Létourville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Praslin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| d’Orgeville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| elle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| grand-duc héritier de Luxembourg | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| général de Froberville | 1500 ± 700 | 700.0 | 0 | 7 | 0.0 | 0.0 |
| jeune blonde de Rivebelle | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| l'abbé Poiré | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| l'ambassadrice de Turquie | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| l'empereur | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| l'historien de la Fronde | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la cousine d'Oriane | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la duchesse d'Alençon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la jeune ouvriere | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la marquise | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la marquise douairière de Cambremer | 1500 ± 700 | 700.0 | 0 | 6 | 0.0 | 0.0 |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le baron Bréau-Chenut | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le capitaine | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le commandant Duroc | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le comte de Paris | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le curé | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le diplomate belge | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le grand-duc Wladimir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le grand-duc héritier de Luxembourg | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le jeune marquis de Cambremer | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Henry | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Picquart | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le marquis de Ganançay | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le marquis de Palancy | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le petit Cambremer | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le prince Von | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| le prince de Faffenheim | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le prince de Galles | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le roi Théodose | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| le vieux père Chenut | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| les Courvoisier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| les demoiselles d’Ambresac | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| ma grand'tante | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| ma grand’tante | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis Maurice de Vaudémont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Fierbois | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis du Lau | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| marquise de Citri | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| monsieur Vallenères | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince Foggi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| prince de Chimay | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince de Faffenheim | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| prince de Léon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince de Sagan | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince de Saxe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince des Laumes | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| princesse Mathilde | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| princesse d'Iéna | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| princesse d'Épinay | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| princesse de Silistrie | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| professeur E… | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| vicomtesse d'Égremont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomtesse de Saint-Fiacre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Élisabeth | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Émilie Daltier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le directeur | 1486 ± 443 | 442.6 | 3 | 11 | 0.0 | 0.0 |
| Morel | 1475 ± 328 | 327.9 | 7 | 32 | -0.024 | 0.0244 |
| marquise de Gallardon | 1455 ± 497 | 496.8 | 2 | 7 | 0.0 | 0.0 |
| M. Verdurin | 1454 ± 398 | 397.7 | 5 | 27 | 0.0 | 0.0 |
| Legrandin | 1446 ± 497 | 496.7 | 2 | 24 | 0.0 | 0.0 |
| prince de Foix | 1417 ± 500 | 499.8 | 2 | 3 | 0.0 | 0.0 |
| marquise de Saint-Euverte | 1416 ± 378 | 377.9 | 5 | 13 | -0.058 | 0.0577 |
| Mlle d'Éporcheville | 1414 ± 605 | 605.0 | 1 | 2 | 0.0 | 0.0 |
| duc de Guermantes | 1413 ± 223 | 222.9 | 18 | 110 | 0.0 | 0.0 |
| Liszt | 1395 ± 590 | 589.8 | 1 | 1 | 0.0 | 0.0 |
| Mme Ristori | 1395 ± 590 | 589.8 | 1 | 1 | 0.0 | 0.0 |
| la Charité de Giotto | 1395 ± 579 | 578.6 | 1 | 1 | -1.6 | 1.6 |
| M. Nissim Bernard | 1392 ± 434 | 433.9 | 4 | 10 | -0.13 | 0.13 |
| Sainte-Beuve | 1383 ± 576 | 576.3 | 1 | 1 | 0.0 | 0.0 |
| prince d’Agrigente | 1380 ± 574 | 574.1 | 1 | 2 | 0.0 | 0.0 |
| Balzac | 1378 ± 576 | 575.7 | 1 | 2 | 0.0 | 0.0 |
| M. Molé | 1378 ± 576 | 575.7 | 1 | 1 | 0.0 | 0.0 |
| M. de Bouillon | 1378 ± 576 | 575.7 | 1 | 1 | 0.0 | 0.0 |
| M. de Vigny | 1378 ± 576 | 575.7 | 1 | 1 | 0.0 | 0.0 |
| Musset | 1378 ± 576 | 575.7 | 1 | 1 | 0.0 | 0.0 |
| Victor Hugo | 1378 ± 576 | 575.7 | 1 | 1 | 0.0 | 0.0 |
| général de Monserfeuil | 1377 ± 572 | 571.6 | 1 | 4 | 0.0 | 0.0 |
| le professeur E… | 1376 ± 557 | 556.9 | 2 | 1 | -0.7 | 0.7 |
| docteur Cottard | 1372 ± 277 | 277.0 | 12 | 43 | 0.0 | 0.0 |
| M. Bontemps | 1372 ± 569 | 568.7 | 1 | 2 | 0.0 | 0.0 |
| M. de Grouchy | 1367 ± 562 | 561.8 | 1 | 4 | 0.0 | 0.0 |
| princesse de Parme | 1362 ± 316 | 315.6 | 8 | 38 | 0.0 | 0.0 |
| Barrès | 1356 ± 572 | 571.8 | 1 | 1 | 0.0 | 0.0 |
| Clémenceau | 1356 ± 572 | 571.8 | 1 | 1 | 0.0 | 0.0 |
| Dumont | 1349 ± 545 | 545.0 | 2 | 1 | -0.72 | 0.72 |
| Mme de Souvré | 1347 ± 404 | 404.2 | 4 | 2 | -0.34 | 0.34 |
| capitaine de Borodino | 1346 ± 566 | 565.9 | 1 | 5 | -0.136 | 0.136 |
| le prince von *** | 1332 ± 529 | 529.2 | 2 | 1 | -0.83 | 0.83 |
| princesse de Guermantes | 1330 ± 326 | 325.9 | 8 | 25 | -0.03 | 0.03 |
| Mme de Varambon | 1324 ± 560 | 560.2 | 1 | 2 | -0.35 | 0.35 |
| oncle Adolphe | 1320 ± 412 | 412.4 | 5 | 6 | -0.317 | 0.3167 |
| Mme Blandais | 1312 ± 509 | 508.6 | 3 | 2 | -0.35 | 0.35 |
| tante Léonie | 1216 ± 400 | 400.5 | 8 | 22 | -0.039 | 0.0386 |
| Maurice | 1188 ± 438 | 438.3 | 7 | 1 | -0.85 | 0.85 |
| Gisèle | 1181 ± 464 | 464.0 | 3 | 5 | -0.336 | 0.336 |
| Rachel | 1169 ± 332 | 331.9 | 13 | 43 | -0.061 | 0.0607 |
| Mme de Cambremer | 1162 ± 344 | 344.4 | 12 | 20 | -0.062 | 0.062 |
| Mme Iéna | 1158 ± 440 | 439.7 | 5 | 1 | -1.76 | 1.76 |
| Saniette | 1057 ± 401 | 400.9 | 8 | 9 | -0.6 | 0.6 |
