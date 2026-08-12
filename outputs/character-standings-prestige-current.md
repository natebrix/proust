# Character Standings — prestige (scoring v2)

- Standings version: `character_standings_prestige_name_view_v2`
- Scoring version: `scoring_v2`
- Source fit: `scoring_v2_prestige_name_view_v1` (`outputs/scoring-v2/scoring-v2-prestige-name-view-ratings.json`)
- Lens / view: `prestige` / `name`
- Time axis: `cumulative_unit_index`
- Characters: `288` (`8` ranked, `280` without sufficient evidence)
- Comparisons: `1184` (mean weight `0.5796`, draw rate `0.018`)
- w2: `60.0` Elo² per unit of narrative time (selected by `one_step_ahead_log_loss_on_v2_comparisons`)
- Provisional band threshold: `200.0` Elo
- Rank rule: `dense_rank_by_conservative_rating`
- Corpus: `foundation`

Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's posterior variance -- an approximate 95% interval conditional on the other characters' trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a character has to be both high and well-measured to place.

The point-by-point trajectories behind these standings are not repeated here; they live in `outputs/scoring-v2/scoring-v2-prestige-name-view-ratings.json` and, for the pilot cast, in the `character-journey-*-timeline-current` artifacts.

## Ranked

The `8` characters the corpus compared often enough for the rating to mean something (band at or under `200.0` Elo), by conservative rating, densely ranked.

| Rank | Character | Rating | Conservative | Comparisons | W-L-D | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Morel | 1817 ± 195 | 1622.2 | 41 | 29-10-2 | 32 | +0.077 | 0.1281 |
| 2 | le narrateur | 1702 ± 171 | 1531.3 | 187 | 123-62-2 | 316 | +0.026 | 0.0629 |
| 3 | Bloch | 1689 ± 187 | 1502.4 | 55 | 22-31-2 | 71 | -0.067 | 0.1586 |
| 4 | Gilberte | 1643 ± 162 | 1481.8 | 77 | 47-28-2 | 76 | +0.098 | 0.1337 |
| 5 | Mme Verdurin | 1638 ± 170 | 1468.2 | 94 | 52-41-1 | 82 | +0.07 | 0.2228 |
| 6 | duchesse de Guermantes | 1588 ± 167 | 1421.0 | 197 | 129-65-3 | 199 | +0.163 | 0.2065 |
| 7 | Robert de Saint-Loup | 1589 ± 184 | 1404.9 | 128 | 47-80-1 | 168 | -0.0 | 0.1601 |
| 8 | baron de Charlus | 1447 ± 160 | 1287.2 | 164 | 95-67-2 | 119 | +0.041 | 0.2677 |

## Insufficient comparative evidence

The `280` characters whose band is still wider than `200.0` Elo. THIS IS NOT THE BOTTOM OF THE TABLE ABOVE. These characters were not compared often enough for a standing to exist: the rating shown is where the fit currently sits, and it is listed here only so the reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which is an ordering of the fit's current guesses and not of the characters.

| Character | Rating | Band | Comparisons | Units | Mean m | Mean abs m |
| --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1909 ± 354 | 354.0 | 14 | 1 | +1.7 | 1.7 |
| Céleste Albaret | 1847 ± 431 | 431.3 | 7 | 3 | +0.25 | 0.25 |
| Mme de Chaussepierre | 1825 ± 440 | 440.0 | 4 | 1 | +1.7 | 1.7 |
| marquis Maurice de Vaudémont | 1767 ± 481 | 481.3 | 2 | 1 | +0.7 | 0.7 |
| Rachel | 1762 ± 233 | 233.0 | 33 | 43 | +0.067 | 0.1781 |
| duc d'Aumale | 1725 ± 499 | 499.1 | 2 | 2 | +0.35 | 0.35 |
| cousine Poictiers | 1713 ± 545 | 545.4 | 1 | 1 | 0.0 | 0.0 |
| duc de Poictiers | 1713 ± 545 | 545.4 | 1 | 1 | 0.0 | 0.0 |
| duc de Sidonia | 1712 ± 546 | 545.5 | 1 | 1 | 0.0 | 0.0 |
| le professeur E… | 1712 ± 546 | 545.5 | 1 | 1 | 0.0 | 0.0 |
| marquis du Lau | 1711 ± 515 | 515.2 | 2 | 2 | +0.35 | 0.35 |
| prince de Saxe | 1702 ± 536 | 535.5 | 3 | 1 | +0.55 | 0.55 |
| prince Von | 1696 ± 521 | 521.2 | 2 | 3 | 0.0 | 0.0 |
| général de Monserfeuil | 1695 ± 543 | 543.2 | 2 | 4 | 0.0 | 0.0 |
| M. de Vaugoubert | 1694 ± 440 | 440.0 | 6 | 9 | +0.189 | 0.1889 |
| Odette | 1689 ± 203 | 203.2 | 97 | 142 | +0.039 | 0.151 |
| la reine de Naples | 1680 ± 408 | 408.3 | 3 | 3 | 0.0 | 0.0 |
| Mme de Stermaria | 1679 ± 579 | 578.7 | 1 | 1 | 0.0 | 0.0 |
| docteur Dieulafoy | 1676 ± 555 | 554.9 | 1 | 1 | +0.8 | 0.8 |
| M. Verdurin | 1673 ± 264 | 264.3 | 30 | 27 | -0.032 | 0.087 |
| Dreyfus | 1670 ± 553 | 553.0 | 3 | 7 | 0.0 | 0.0 |
| Mme de Surgis | 1659 ± 265 | 265.0 | 19 | 9 | +0.089 | 0.2711 |
| Mme Cottard | 1657 ± 558 | 557.8 | 2 | 11 | 0.0 | 0.0 |
| Andrée | 1656 ± 538 | 538.2 | 2 | 31 | +0.023 | 0.0226 |
| Bergotte | 1655 ± 403 | 402.9 | 9 | 36 | +0.07 | 0.0697 |
| Mme de Vaugoubert | 1644 ± 565 | 565.1 | 1 | 2 | 0.0 | 0.0 |
| jeune blonde de Rivebelle | 1641 ± 572 | 572.5 | 1 | 2 | 0.0 | 0.0 |
| Liszt | 1641 ± 548 | 548.2 | 2 | 1 | 0.0 | 0.0 |
| Mme Ristori | 1641 ± 548 | 548.2 | 2 | 1 | 0.0 | 0.0 |
| comte de Forcheville | 1639 ± 319 | 319.4 | 21 | 25 | +0.124 | 0.124 |
| Mlle Vinteuil | 1632 ± 442 | 442.2 | 6 | 15 | 0.0 | 0.0 |
| d'Orléans | 1632 ± 581 | 581.2 | 1 | 1 | 0.0 | 0.0 |
| prince de Chimay | 1626 ± 555 | 554.8 | 2 | 1 | 0.0 | 0.0 |
| duc de Chartres | 1625 ± 556 | 555.9 | 2 | 1 | 0.0 | 0.0 |
| Jupien | 1620 ± 267 | 266.8 | 15 | 18 | +0.094 | 0.0944 |
| général de Froberville | 1620 ± 512 | 511.7 | 5 | 7 | 0.0 | 0.0 |
| Sir Rufus Israël | 1616 ± 562 | 561.7 | 2 | 1 | 0.0 | 0.0 |
| M. Arthur Meyer | 1616 ± 475 | 475.2 | 3 | 1 | 0.0 | 0.0 |
| Mme Trombert | 1614 ± 584 | 584.3 | 1 | 1 | 0.0 | 0.0 |
| le commandant Duroc | 1605 ± 586 | 585.7 | 1 | 1 | 0.0 | 0.0 |
| colonel Picquart | 1604 ± 581 | 581.3 | 1 | 1 | 0.0 | 0.0 |
| Lady Israël | 1601 ± 584 | 583.8 | 1 | 1 | 0.0 | 0.0 |
| docteur du Boulbon | 1601 ± 515 | 514.9 | 3 | 6 | 0.0 | 0.0 |
| le pianiste | 1599 ± 480 | 480.1 | 2 | 3 | +0.26 | 0.26 |
| Mme de Villebon | 1597 ± 587 | 587.0 | 1 | 1 | 0.0 | 0.0 |
| baron de Guermantes | 1597 ± 587 | 587.0 | 1 | 1 | 0.0 | 0.0 |
| Mme de Grouchy | 1596 ± 576 | 576.3 | 2 | 1 | 0.0 | 0.0 |
| Marie-Aynard | 1593 ± 592 | 592.4 | 1 | 1 | 0.0 | 0.0 |
| le peintre | 1590 ± 432 | 431.5 | 3 | 8 | 0.0 | 0.0 |
| prince de Guermantes | 1590 ± 262 | 261.9 | 23 | 22 | -0.037 | 0.0373 |
| Victurnienne | 1588 ± 597 | 596.8 | 1 | 1 | 0.0 | 0.0 |
| monsieur Vallenères | 1587 ± 595 | 595.4 | 1 | 1 | 0.0 | 0.0 |
| Legrandin | 1586 ± 314 | 313.7 | 11 | 24 | +0.002 | 0.1396 |
| princesse de Luxembourg | 1585 ± 430 | 429.8 | 3 | 6 | 0.0 | 0.0 |
| le grand-père du narrateur | 1583 ± 372 | 372.2 | 5 | 16 | 0.0 | 0.0 |
| Mme de Franquetot | 1582 ± 455 | 455.4 | 4 | 3 | +0.25 | 0.25 |
| Cartier | 1580 ± 484 | 484.0 | 2 | 1 | 0.0 | 0.0 |
| princesse Sherbatoff | 1577 ± 343 | 343.0 | 7 | 5 | +0.212 | 0.492 |
| Mme Bontemps | 1576 ± 455 | 455.0 | 5 | 13 | +0.054 | 0.0538 |
| Mme de Rochechouart | 1575 ± 515 | 514.6 | 2 | 1 | 0.0 | 0.0 |
| Duroc | 1571 ± 606 | 605.7 | 1 | 1 | 0.0 | 0.0 |
| le directeur | 1568 ± 365 | 364.8 | 11 | 11 | +0.154 | 0.1545 |
| Mme de Montmorency | 1566 ± 503 | 502.8 | 2 | 1 | 0.0 | 0.0 |
| Norpois | 1562 ± 286 | 285.8 | 38 | 63 | +0.048 | 0.073 |
| Françoise | 1560 ± 272 | 271.8 | 36 | 82 | +0.036 | 0.071 |
| tante Léonie | 1559 ± 458 | 457.7 | 4 | 22 | +0.034 | 0.0341 |
| princesse de Parme | 1557 ± 232 | 232.2 | 22 | 38 | +0.095 | 0.0947 |
| le grand-duc Wladimir | 1549 ± 469 | 469.4 | 2 | 1 | 0.0 | 0.0 |
| comtesse Molé | 1544 ± 389 | 388.6 | 6 | 6 | 0.0 | 0.0 |
| le père du narrateur | 1544 ± 330 | 329.9 | 16 | 24 | -0.042 | 0.0917 |
| la marquise douairière de Cambremer | 1538 ± 358 | 357.6 | 9 | 6 | +0.117 | 0.3833 |
| marquis de Fierbois | 1536 ± 511 | 511.4 | 2 | 1 | 0.0 | 0.0 |
| duchesse de Gallardon douairière | 1532 ± 508 | 507.7 | 2 | 1 | 0.0 | 0.0 |
| comtesse douairière d'Argencourt | 1531 ± 506 | 506.5 | 2 | 1 | 0.0 | 0.0 |
| la mère du narrateur | 1529 ± 304 | 304.1 | 20 | 40 | 0.0 | 0.0 |
| prince de Foix | 1529 ± 427 | 427.3 | 3 | 3 | 0.0 | 0.0 |
| prince d’Agrigente | 1518 ± 553 | 552.9 | 2 | 2 | 0.0 | 0.0 |
| Mlle d'Éporcheville | 1517 ± 549 | 548.6 | 2 | 2 | 0.0 | 0.0 |
| marquis de Bréauté | 1513 ± 284 | 283.8 | 19 | 19 | -0.157 | 0.1574 |
| Mme de Marsantes | 1512 ± 328 | 327.5 | 14 | 21 | +0.041 | 0.0405 |
| Mme Timoléon d'Amoncourt | 1506 ± 473 | 473.0 | 2 | 1 | 0.0 | 0.0 |
| Arnulphe | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Balzac | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Barrès | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Beauserfeuil | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Bibi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Charcot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Clémenceau | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Céline | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Dechambre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dostoïevski | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Dumont | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Esther | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Eulalie | 1500 ± 700 | 700.0 | 0 | 7 | 0.0 | 0.0 |
| Flora | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Gribelin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Herbinger | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| La Moussaye | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Lady Israels | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Léa | 1500 ± 700 | 700.0 | 0 | 4 | 0.0 | 0.0 |
| Léonor de Cambremer | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Barrère | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Carnot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Molé | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Reinach | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vibert | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. d'Orsan | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Beauserfeuil | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Bornier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Bouillon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Courgivaux | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de La Rochefoucauld | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Marsantes | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| M. de Miribel | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. de Vigny | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Madame Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Maeterlinck | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Manet | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Marie Gineste | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Maurice | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle Bloch | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mlle de Saint-Loup | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Mlle de l’Orgeville | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Carnot | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Elstir | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Iéna | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Legrandin mère | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Poncin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme Putbus | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme d'Heudicourt | 1500 ± 700 | 700.0 | 0 | 5 | 0.0 | 0.0 |
| Mme de Charlus | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Mme de Sagan | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Simiane | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Mme de Varambon | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Musset | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Napoléon III | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Octave | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Poullein | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| Prince Henri d'Orléans | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Périgot (Joseph) | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Rémi | 1500 ± 700 | 700.0 | 0 | 3 | 0.0 | 0.0 |
| Sarah Bernhardt | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Thibaud | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Théodose Cadet | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Victoire | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Victor Hugo | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Vigny | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| comtesse de Monteriender | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de La Rochefoucauld | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de La Trémoïlle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| duchesse de Praslin | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| elle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| grand-duc héritier de Luxembourg | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| l'abbé Poiré | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| l'empereur | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la Charité de Giotto | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la cousine d'Oriane | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la duchesse d'Alençon | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la jeune ouvriere | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la marquise | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| la « marquise » | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le baron Bréau-Chenut | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le bâtonnier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le capitaine | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le curé | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le diplomate belge | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le jeune prince de Foix | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Henry | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le lieutenant-colonel Picquart | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le prince Von | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| le prince von *** | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| le vieux père Chenut | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| les La Trémoïlle | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| les demoiselles d’Ambresac | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| ma grand'tante | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquis de Beausergent | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| marquise de Citri | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince Foggi | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| prince d'Agrigente | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| princesse de Nassau | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| professeur E… | 1500 ± 700 | 700.0 | 0 | 2 | 0.0 | 0.0 |
| vicomte de Courvoisier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| vicomtesse de Saint-Fiacre | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Élisabeth | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| Émilie Daltier | 1500 ± 700 | 700.0 | 0 | 1 | 0.0 | 0.0 |
| M. Vinteuil | 1498 ± 483 | 482.8 | 6 | 15 | -0.127 | 0.1267 |
| marquise de Saint-Euverte | 1497 ± 256 | 256.1 | 30 | 13 | -0.421 | 0.5469 |
| duchesse de Létourville | 1496 ± 479 | 479.0 | 2 | 1 | 0.0 | 0.0 |
| docteur Percepied | 1490 ± 477 | 477.2 | 2 | 1 | 0.0 | 0.0 |
| l'amie de Mlle Vinteuil | 1489 ± 478 | 477.9 | 2 | 12 | 0.0 | 0.0 |
| duc de Châtellerault | 1487 ± 471 | 470.9 | 2 | 5 | 0.0 | 0.0 |
| M. d'Herweck | 1481 ± 382 | 382.4 | 4 | 2 | -0.85 | 0.85 |
| prince des Laumes | 1477 ± 466 | 465.8 | 3 | 3 | 0.0 | 0.0 |
| M. de Crécy | 1477 ± 498 | 497.5 | 3 | 1 | 0.0 | 0.0 |
| Mme Féré | 1477 ± 498 | 497.5 | 3 | 1 | 0.0 | 0.0 |
| M. de Chevregny | 1476 ± 498 | 498.4 | 3 | 1 | 0.0 | 0.0 |
| Brichot | 1474 ± 266 | 265.6 | 21 | 21 | +0.039 | 0.1133 |
| M. Nissim Bernard | 1466 ± 646 | 645.5 | 1 | 10 | 0.0 | 0.0 |
| Marie | 1466 ± 646 | 645.5 | 1 | 1 | 0.0 | 0.0 |
| Mme Sazerat | 1464 ± 377 | 377.0 | 5 | 6 | 0.0 | 0.0 |
| princesse d'Épinay | 1463 ± 448 | 447.9 | 3 | 3 | 0.0 | 0.0 |
| la grand-mère | 1461 ± 336 | 336.3 | 31 | 80 | +0.012 | 0.0548 |
| M. Bontemps | 1460 ± 518 | 517.8 | 2 | 2 | +0.3 | 0.3 |
| Mme de Villeparisis | 1453 ± 250 | 250.4 | 72 | 79 | +0.02 | 0.24 |
| Mme Leroi | 1452 ± 329 | 328.7 | 7 | 5 | -0.14 | 0.14 |
| Létourville | 1449 ± 629 | 628.9 | 1 | 1 | 0.0 | 0.0 |
| princesse de Guermantes | 1448 ± 279 | 278.8 | 29 | 25 | +0.053 | 0.1812 |
| M. d'Argencourt | 1444 ± 318 | 318.0 | 14 | 14 | -0.118 | 0.1179 |
| le jeune marquis de Cambremer | 1444 ± 624 | 623.8 | 1 | 1 | 0.0 | 0.0 |
| Antoine | 1444 ± 622 | 622.1 | 1 | 1 | 0.0 | 0.0 |
| Lady Rufus Israël | 1440 ± 617 | 617.4 | 1 | 1 | 0.0 | 0.0 |
| M. de Chateaubriand | 1439 ± 616 | 615.9 | 1 | 2 | 0.0 | 0.0 |
| Mme Goupil | 1439 ± 616 | 615.9 | 1 | 2 | 0.0 | 0.0 |
| d’Orgeville | 1439 ± 617 | 617.3 | 1 | 1 | 0.0 | 0.0 |
| les Courvoisier | 1436 ± 612 | 611.5 | 1 | 1 | 0.0 | 0.0 |
| prince de Léon | 1434 ± 610 | 609.7 | 1 | 1 | 0.0 | 0.0 |
| le marquis de Ganançay | 1433 ± 621 | 620.8 | 1 | 1 | 0.0 | 0.0 |
| M. Ski | 1433 ± 556 | 555.6 | 2 | 2 | 0.0 | 0.0 |
| Goncourt | 1432 ± 611 | 611.1 | 1 | 2 | 0.0 | 0.0 |
| prince de Sagan | 1432 ± 609 | 609.2 | 1 | 1 | 0.0 | 0.0 |
| Gisèle | 1432 ± 608 | 608.4 | 1 | 5 | 0.0 | 0.0 |
| Rosemonde | 1432 ± 608 | 608.4 | 1 | 4 | 0.0 | 0.0 |
| L’excellent écrivain G… | 1431 ± 607 | 607.4 | 1 | 1 | 0.0 | 0.0 |
| oncle Adolphe | 1430 ± 440 | 439.7 | 8 | 6 | -0.142 | 0.1417 |
| M. de Goncourt | 1428 ± 611 | 610.7 | 1 | 1 | 0.0 | 0.0 |
| Bloch père | 1426 ± 394 | 394.3 | 7 | 8 | -0.212 | 0.2125 |
| Madame d'Ambresac | 1425 ± 599 | 599.3 | 1 | 1 | 0.0 | 0.0 |
| Sainte-Beuve | 1425 ± 603 | 602.7 | 1 | 1 | 0.0 | 0.0 |
| l'historien de la Fronde | 1420 ± 594 | 593.7 | 1 | 1 | 0.0 | 0.0 |
| Mme de Souvré | 1419 ± 446 | 446.3 | 4 | 2 | 0.0 | 0.0 |
| Bismarck | 1419 ± 608 | 607.6 | 1 | 1 | 0.0 | 0.0 |
| D'Annunzio | 1415 ± 594 | 593.8 | 1 | 1 | 0.0 | 0.0 |
| docteur Cottard | 1413 ± 292 | 291.9 | 33 | 43 | -0.035 | 0.0349 |
| Victurnien | 1410 ± 589 | 588.6 | 1 | 2 | 0.0 | 0.0 |
| le prince de Faffenheim | 1407 ± 592 | 592.1 | 1 | 1 | 0.0 | 0.0 |
| comte de Paris | 1403 ± 428 | 428.0 | 3 | 3 | 0.0 | 0.0 |
| M. Pierre | 1403 ± 587 | 587.0 | 1 | 2 | -0.35 | 0.35 |
| comtesse G… | 1403 ± 587 | 587.0 | 1 | 1 | -0.7 | 0.7 |
| Mme de Morienval | 1400 ± 586 | 586.5 | 1 | 1 | 0.0 | 0.0 |
| duchesse de Luxembourg | 1400 ± 586 | 586.5 | 1 | 1 | 0.0 | 0.0 |
| le marquis de Palancy | 1398 ± 584 | 584.1 | 1 | 1 | 0.0 | 0.0 |
| le comte de Paris | 1393 ± 582 | 582.4 | 1 | 1 | 0.0 | 0.0 |
| le prince de Galles | 1393 ± 582 | 582.4 | 1 | 1 | 0.0 | 0.0 |
| Théodore | 1392 ± 579 | 578.6 | 1 | 1 | 0.0 | 0.0 |
| Mme de Sévigné | 1389 ± 620 | 620.2 | 2 | 4 | 0.0 | 0.0 |
| ma grand’tante | 1386 ± 576 | 575.8 | 1 | 1 | 0.0 | 0.0 |
| Swann | 1385 ± 221 | 220.9 | 149 | 202 | -0.014 | 0.1934 |
| M. Swann, le père | 1384 ± 573 | 573.4 | 1 | 1 | 0.0 | 0.0 |
| Coquelin | 1378 ± 576 | 575.6 | 1 | 1 | 0.0 | 0.0 |
| duc de Guermantes | 1367 ± 205 | 204.9 | 90 | 110 | -0.005 | 0.1043 |
| Mme Blandais | 1362 ± 568 | 568.4 | 1 | 2 | -0.35 | 0.35 |
| princesse de Silistrie | 1361 ± 541 | 541.4 | 3 | 1 | 0.0 | 0.0 |
| le petit Cambremer | 1360 ± 541 | 540.7 | 3 | 1 | 0.0 | 0.0 |
| M. Grevy | 1356 ± 557 | 556.9 | 1 | 1 | 0.0 | 0.0 |
| Elstir | 1336 ± 422 | 421.8 | 9 | 29 | -0.026 | 0.0259 |
| Albertine | 1329 ± 257 | 257.2 | 31 | 146 | -0.017 | 0.0514 |
| le grand-duc héritier de Luxembourg | 1323 ± 543 | 542.8 | 1 | 1 | 0.0 | 0.0 |
| M. de Luxembourg | 1322 ± 532 | 531.8 | 2 | 1 | -0.7 | 0.7 |
| le roi Théodose | 1318 ± 591 | 591.4 | 3 | 3 | 0.0 | 0.0 |
| Aimé | 1317 ± 558 | 558.4 | 4 | 18 | -0.039 | 0.0389 |
| Mme Blatin | 1317 ± 509 | 508.7 | 2 | 1 | -0.85 | 0.85 |
| Mme d'Arpajon | 1313 ± 305 | 305.2 | 17 | 8 | -0.286 | 0.2863 |
| Mlle de Stermaria | 1310 ± 496 | 495.7 | 3 | 5 | -0.15 | 0.15 |
| prince de Faffenheim | 1308 ± 515 | 514.9 | 2 | 2 | 0.0 | 0.0 |
| Picquart | 1308 ± 519 | 519.1 | 2 | 2 | -0.35 | 0.35 |
| Monsieur Vallenères | 1284 ± 497 | 496.9 | 3 | 1 | -0.83 | 0.83 |
| l'ambassadrice de Turquie | 1269 ± 480 | 480.0 | 4 | 1 | -0.75 | 0.75 |
| princesse Mathilde | 1257 ± 555 | 555.1 | 3 | 2 | 0.0 | 0.0 |
| vicomtesse d'Égremont | 1255 ± 476 | 475.9 | 3 | 1 | -1.7 | 1.7 |
| M. de Grouchy | 1252 ± 416 | 416.1 | 8 | 4 | -0.357 | 0.3575 |
| marquise de Gallardon | 1239 ± 486 | 486.2 | 8 | 7 | -0.5 | 0.5 |
| capitaine de Borodino | 1232 ± 334 | 334.5 | 11 | 5 | -0.196 | 0.436 |
| Alix | 1231 ± 398 | 397.8 | 7 | 3 | -0.283 | 0.2833 |
| colonel de Froberville | 1219 ± 340 | 339.9 | 14 | 1 | -0.88 | 0.88 |
| princesse d'Iéna | 1203 ± 494 | 493.9 | 3 | 1 | -0.78 | 0.78 |
| la Berma | 1200 ± 432 | 431.9 | 13 | 19 | -0.193 | 0.1926 |
| M. de Stermaria | 1176 ± 405 | 404.8 | 7 | 4 | -0.307 | 0.3075 |
| marquis de Cambremer | 1123 ± 288 | 288.2 | 21 | 6 | -0.133 | 0.1333 |
| Mme de Cambremer | 1064 ± 324 | 324.0 | 33 | 20 | -0.213 | 0.3835 |
| Saniette | 962 ± 387 | 387.1 | 18 | 9 | -0.44 | 0.44 |
