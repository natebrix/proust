# Character Glicko-2

- Analysis version: `character_glicko2_inclusion_v1`
- Lens: `inclusion`
- Source review version: `corpus_sanity_review_v1`
- Character count: `288`
- Match count: `5756`
- Draw rate: `0.327`
- Initial rating: `1500.0`
- Initial RD: `350.0`
- Initial volatility: `0.06`
- Tau: `0.5`
- Epsilon: `0.25`
- Rating period rule: `canonical_chapter`
- Period count: `18`
- Provisional RD threshold: `100.0`
- Corpus: `foundation`

Ratings are shown as `rating ± 2*RD`, an approximate 95% confidence band. Characters whose RD exceeds the provisional threshold are excluded from the top/bottom/divergence tables below (but never from the full character table), since a rating built on very little evidence is noise.

## Top Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1856 ± 197 | 1659.5 | 98.4 | 0.06 | 14 | 14-0-0 | 1 | +0.39 |
| docteur du Boulbon | 1754 ± 165 | 1589.8 | 82.3 | 0.06 | 27 | 19-3-5 | 6 | -0.469 |
| Françoise | 1633 ± 82 | 1550.6 | 41.0 | 0.0602 | 217 | 100-48-69 | 82 | -0.369 |
| comte de Forcheville | 1632 ± 100 | 1532.1 | 50.0 | 0.0605 | 112 | 55-18-39 | 25 | -0.4 |
| Bergotte | 1622 ± 93 | 1528.9 | 46.4 | 0.06 | 129 | 52-31-46 | 36 | -0.199 |
| Léa | 1720 ± 195 | 1525.0 | 97.3 | 0.06 | 14 | 8-0-6 | 4 | -0.7 |
| le peintre | 1675 ± 160 | 1514.7 | 80.1 | 0.06 | 42 | 16-4-22 | 8 | -0.298 |
| M. Verdurin | 1596 ± 95 | 1501.2 | 47.3 | 0.06 | 110 | 38-23-49 | 27 | -0.64 |
| le grand-père du narrateur | 1640 ± 146 | 1493.7 | 73.1 | 0.06 | 63 | 25-7-31 | 16 | -0.664 |
| Elstir | 1578 ± 90 | 1487.4 | 45.2 | 0.0601 | 106 | 42-29-35 | 29 | +0.014 |

## Bottom Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Saniette | 1204 ± 158 | 1046.9 | 78.8 | 0.0601 | 35 | 1-27-7 | 9 | -3.263 |
| Mme de Franquetot | 1279 ± 170 | 1109.7 | 84.8 | 0.0601 | 23 | 4-13-6 | 3 | -1.092 |
| Mme d'Heudicourt | 1333 ± 199 | 1133.8 | 99.4 | 0.06 | 18 | 3-11-4 | 5 | -1.482 |
| marquis de Cambremer | 1318 ± 134 | 1183.3 | 67.2 | 0.0601 | 45 | 7-23-15 | 6 | -1.016 |
| princesse Sherbatoff | 1359 ± 172 | 1186.9 | 86.2 | 0.06 | 19 | 5-13-1 | 5 | -0.787 |
| Rosemonde | 1429 ± 191 | 1238.2 | 95.3 | 0.06 | 20 | 5-7-8 | 4 | -0.7 |
| M. de Vaugoubert | 1399 ± 159 | 1239.7 | 79.5 | 0.06 | 35 | 6-12-17 | 9 | -1.383 |
| Mme d'Arpajon | 1396 ± 147 | 1248.5 | 73.5 | 0.06 | 37 | 7-20-10 | 8 | -1.53 |
| tante Léonie | 1439 ± 181 | 1258.0 | 90.4 | 0.0601 | 38 | 12-22-4 | 22 | -0.825 |
| le petit Cambremer | 1460 ± 197 | 1263.5 | 98.4 | 0.06 | 14 | 1-3-10 | 1 | -0.8 |

## Provisional Characters

Characters whose RD is still above the provisional threshold -- their rating should be treated as unstable.

| Character | Rating | RD | Matches | Units | Last Period |
| --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1918 ± 229 | 114.7 | 17 | 3 | v5 |
| Mme de Grouchy | 1877 ± 377 | 188.7 | 4 | 1 | v3-p2 |
| Céleste Albaret | 1866 ± 214 | 107.0 | 17 | 3 | v5 |
| prince de Saxe | 1848 ± 375 | 187.5 | 3 | 1 | v3-p1 |
| marquis de Beausergent | 1825 ± 203 | 101.4 | 12 | 1 | v7-p4-le-bal-de-tetes |
| Mme de Chaussepierre | 1823 ± 339 | 169.6 | 4 | 1 | v5 |
| Mme Elstir | 1818 ± 273 | 136.3 | 7 | 1 | v2-p2-noms-de-pays-le-pays |
| marquis Maurice de Vaudémont | 1810 ± 468 | 234.0 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Trémoïlle | 1799 ± 423 | 211.6 | 3 | 1 | v1-p2-un-amour-de-swann |
| Eulalie | 1796 ± 237 | 118.4 | 16 | 7 | v5 |
| colonel Picquart | 1791 ± 349 | 174.5 | 4 | 1 | v3-p1 |
| Marie | 1788 ± 272 | 136.1 | 7 | 1 | v4-p2 |
| Mlle de Saint-Loup | 1783 ± 251 | 125.4 | 7 | 2 | v7-p4-le-bal-de-tetes |
| Maeterlinck | 1783 ± 329 | 164.3 | 5 | 1 | v3-p1 |
| duc de Sidonia | 1771 ± 455 | 227.3 | 2 | 1 | v4-p2 |
| Lady Israels | 1766 ± 456 | 228.0 | 2 | 1 | v2-p1-autour-de-mme-swann |
| Mlle Bloch | 1762 ± 426 | 213.1 | 2 | 1 | v4-p2 |
| Mlle de Stermaria | 1758 ± 270 | 134.9 | 10 | 5 | v3-p2 |
| Duroc | 1751 ± 454 | 227.1 | 2 | 1 | v3-p1 |
| duc d'Aumale | 1744 ± 377 | 188.7 | 4 | 2 | v3-p2 |
| Herbinger | 1735 ± 408 | 204.1 | 3 | 1 | v1-p2-un-amour-de-swann |
| Victurnien | 1734 ± 267 | 133.7 | 8 | 2 | v4-p2 |
| le commandant Duroc | 1726 ± 413 | 206.5 | 2 | 1 | v3-p1 |
| Gribelin | 1725 ± 315 | 157.3 | 6 | 1 | v3-p1 |
| Émilie Daltier | 1718 ± 358 | 179.1 | 3 | 1 | v5 |
| marquis du Lau | 1714 ± 308 | 153.9 | 5 | 2 | v6-p2 |
| Bibi | 1709 ± 425 | 212.5 | 2 | 1 | v3-p2 |
| Bismarck | 1704 ± 353 | 176.4 | 4 | 1 | v2-p1-autour-de-mme-swann |
| elle | 1689 ± 502 | 251.2 | 1 | 1 | v3-p1 |
| monsieur Vallenères | 1685 ± 427 | 213.7 | 2 | 1 | v3-p1 |
| Dechambre | 1683 ± 368 | 183.8 | 3 | 1 | v4-p2 |
| Rémi | 1677 ± 222 | 111.2 | 17 | 3 | v1-p2-un-amour-de-swann |
| Létourville | 1676 ± 353 | 176.5 | 3 | 1 | v7-p4-le-bal-de-tetes |
| grand-duc héritier de Luxembourg | 1673 ± 250 | 125.1 | 9 | 2 | v3-p2 |
| les La Trémoïlle | 1663 ± 329 | 164.3 | 7 | 1 | v1-p2-un-amour-de-swann |
| M. de Courgivaux | 1662 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| Mme de Villebon | 1662 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| baron de Guermantes | 1662 ± 585 | 292.6 | 1 | 1 | v3-p1 |
| docteur Dieulafoy | 1660 ± 502 | 251.0 | 1 | 1 | v3-p2 |
| la duchesse d'Alençon | 1650 ± 314 | 157.0 | 6 | 1 | v3-p2 |
| Poullein | 1649 ± 485 | 242.6 | 2 | 2 | v3-p2 |
| M. Vibert | 1645 ± 374 | 187.0 | 3 | 1 | v3-p2 |
| le pianiste | 1642 ± 271 | 135.5 | 10 | 3 | v1-p2-un-amour-de-swann |
| Marie-Aynard | 1636 ± 311 | 155.5 | 7 | 1 | v3-p1 |
| Victurnienne | 1636 ± 311 | 155.5 | 7 | 1 | v3-p1 |
| Mme de Stermaria | 1632 ± 309 | 154.6 | 5 | 1 | v3-p2 |
| cousine Poictiers | 1625 ± 321 | 160.7 | 5 | 1 | v3-p1 |
| duc de Poictiers | 1625 ± 321 | 160.7 | 5 | 1 | v3-p1 |
| M. d'Orsan | 1624 ± 269 | 134.7 | 11 | 1 | v1-p2-un-amour-de-swann |
| Théodore | 1624 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| M. de Bornier | 1620 ± 346 | 173.2 | 5 | 1 | v3-p2 |
| Lady Rufus Israël | 1617 ± 280 | 140.2 | 6 | 1 | v6-p2 |
| Flora | 1602 ± 336 | 168.2 | 8 | 1 | v1-p1-combray |
| Manet | 1600 ± 316 | 158.0 | 5 | 1 | v3-p2 |
| Lady Israël | 1600 ± 327 | 163.5 | 5 | 1 | v3-p1 |
| Mme Leroi | 1600 ± 207 | 103.7 | 13 | 5 | v3-p1 |
| Arnulphe | 1594 ± 362 | 181.0 | 4 | 1 | v4-p2 |
| le grand-duc Wladimir | 1593 ± 366 | 182.9 | 3 | 1 | v4-p2 |
| Sarah Bernhardt | 1589 ± 287 | 143.3 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| le jeune prince de Foix | 1589 ± 287 | 143.3 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| vicomte de Courvoisier | 1589 ± 287 | 143.3 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| Mlle de l’Orgeville | 1588 ± 352 | 176.1 | 3 | 1 | v6-p4 |
| le baron Bréau-Chenut | 1586 ± 308 | 154.1 | 7 | 1 | v2-p1-autour-de-mme-swann |
| le vieux père Chenut | 1586 ± 308 | 154.1 | 7 | 1 | v2-p1-autour-de-mme-swann |
| M. de Goncourt | 1582 ± 237 | 118.4 | 8 | 1 | v7-p1-a-tansonville |
| M. de Beauserfeuil | 1581 ± 278 | 138.9 | 7 | 1 | v3-p2 |
| Charcot | 1577 ± 225 | 112.3 | 12 | 1 | v3-p1 |
| M. Reinach | 1577 ± 225 | 112.3 | 12 | 1 | v3-p1 |
| d'Orléans | 1575 ± 351 | 175.6 | 5 | 1 | v2-p2-noms-de-pays-le-pays |
| prince de Sagan | 1571 ± 277 | 138.5 | 7 | 1 | v4-p2 |
| Mlle d'Éporcheville | 1569 ± 224 | 111.9 | 10 | 2 | v6-p2 |
| M. de Marsantes | 1568 ± 304 | 151.9 | 7 | 2 | v3-p1 |
| jeune blonde de Rivebelle | 1568 ± 334 | 167.2 | 6 | 2 | v2-p2-noms-de-pays-le-pays |
| duc de Chartres | 1566 ± 208 | 103.9 | 14 | 1 | v4-p2 |
| prince de Chimay | 1566 ± 208 | 103.9 | 14 | 1 | v4-p2 |
| le marquis de Ganançay | 1566 ± 350 | 174.9 | 6 | 1 | v3-p1 |
| le marquis de Palancy | 1566 ± 350 | 174.9 | 6 | 1 | v3-p1 |
| duchesse de Létourville | 1565 ± 291 | 145.5 | 5 | 1 | v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle |
| Mme de Montmorency | 1564 ± 221 | 110.7 | 11 | 1 | v4-p2 |
| Mme de Rochechouart | 1564 ± 221 | 110.7 | 11 | 1 | v4-p2 |
| le jeune marquis de Cambremer | 1564 ± 202 | 100.8 | 12 | 1 | v6-p4 |
| Mme de Sagan | 1559 ± 369 | 184.7 | 3 | 1 | v3-p1 |
| Élisabeth | 1558 ± 284 | 142.0 | 6 | 1 | v5 |
| Mme Legrandin mère | 1558 ± 280 | 140.0 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victoire | 1558 ± 280 | 140.0 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Mme Timoléon d'Amoncourt | 1557 ± 244 | 121.9 | 9 | 1 | v4-p2 |
| M. de Chateaubriand | 1554 ± 285 | 142.7 | 11 | 2 | v6-p2 |
| comtesse de Monteriender | 1548 ± 352 | 175.9 | 4 | 1 | v1-p2-un-amour-de-swann |
| princesse d'Épinay | 1540 ± 241 | 120.6 | 12 | 3 | v3-p2 |
| Coquelin | 1539 ± 321 | 160.3 | 5 | 1 | v1-p3-noms-de-pays-le-nom |
| prince d’Agrigente | 1537 ± 202 | 101.1 | 15 | 2 | v6-p2 |
| Mme Trombert | 1532 ± 347 | 173.3 | 4 | 1 | v2-p1-autour-de-mme-swann |
| Sir Rufus Israël | 1530 ± 282 | 141.0 | 7 | 1 | v3-p1 |
| M. Arthur Meyer | 1528 ± 269 | 134.7 | 6 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| M. Barrère | 1528 ± 500 | 250.2 | 1 | 1 | v6-p3 |
| Napoléon III | 1523 ± 307 | 153.4 | 8 | 1 | v1-p2-un-amour-de-swann |
| Mme Putbus | 1519 ± 251 | 125.4 | 8 | 1 | v5 |
| M. de La Rochefoucauld | 1516 ± 327 | 163.3 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Rochefoucauld | 1516 ± 327 | 163.3 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de Praslin | 1516 ± 327 | 163.3 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| Thibaud | 1514 ± 244 | 121.9 | 8 | 1 | v5 |
| Dostoïevski | 1512 ± 273 | 136.5 | 6 | 1 | v5 |
| comte de Paris | 1509 ± 250 | 125.1 | 10 | 3 | v2-p1-autour-de-mme-swann |
| Gisèle | 1506 ± 257 | 128.7 | 14 | 5 | v5 |
| M. Carnot | 1506 ± 252 | 126.1 | 9 | 1 | v3-p2 |
| Mme Carnot | 1506 ± 252 | 126.1 | 9 | 1 | v3-p2 |
| oncle Adolphe | 1505 ± 219 | 109.7 | 20 | 6 | v3-p1 |
| Liszt | 1504 ± 304 | 151.8 | 6 | 1 | v3-p1 |
| Mme Ristori | 1504 ± 304 | 151.8 | 6 | 1 | v3-p1 |
| l'abbé Poiré | 1502 ± 224 | 112.0 | 10 | 1 | v4-p2 |
| La Moussaye | 1500 ± 703 | 351.4 | 0 | 1 | v5 |
| M. Swann, le père | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| Mme Poncin | 1500 ± 704 | 352.2 | 0 | 1 | v2-p2-noms-de-pays-le-pays |
| Périgot (Joseph) | 1500 ± 704 | 351.9 | 0 | 1 | v3-p2 |
| docteur Percepied | 1500 ± 426 | 212.9 | 4 | 1 | v1-p1-combray |
| la « marquise » | 1500 ± 704 | 352.0 | 0 | 1 | v3-p1 |
| le comte de Paris | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| le prince de Galles | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| Octave | 1497 ± 426 | 213.0 | 4 | 2 | v6-p2 |
| d’Orgeville | 1495 ± 269 | 134.4 | 7 | 1 | v4-p2 |
| Léonor de Cambremer | 1494 ± 203 | 101.4 | 12 | 1 | v7-p4-le-bal-de-tetes |
| le capitaine | 1489 ± 413 | 206.5 | 2 | 1 | v3-p1 |
| Mme de Vaugoubert | 1488 ± 257 | 128.6 | 9 | 2 | v5 |
| L’excellent écrivain G… | 1488 ± 351 | 175.3 | 4 | 1 | v3-p1 |
| le roi Théodose | 1488 ± 289 | 144.7 | 8 | 3 | v4-p2 |
| prince d'Agrigente | 1487 ± 451 | 225.5 | 2 | 2 | v7-p4-le-bal-de-tetes |
| D'Annunzio | 1485 ± 319 | 159.4 | 5 | 1 | v4-p2 |
| Sainte-Beuve | 1485 ± 266 | 132.8 | 7 | 1 | v3-p2 |
| comtesse douairière d'Argencourt | 1484 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| duchesse de Gallardon douairière | 1484 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| marquis de Fierbois | 1484 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| prince Von | 1484 ± 259 | 129.3 | 8 | 3 | v3-p2 |
| princesse de Nassau | 1484 ± 496 | 248.1 | 1 | 1 | v7-p4-le-bal-de-tetes |
| M. Grevy | 1476 ± 423 | 211.6 | 3 | 1 | v1-p2-un-amour-de-swann |
| M. de Miribel | 1476 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Henry | 1476 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Picquart | 1476 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| princesse Mathilde | 1476 ± 317 | 158.7 | 7 | 2 | v3-p2 |
| prince de Foix | 1475 ± 214 | 107.0 | 14 | 3 | v7-p2-m-de-charlus-pendant-la-guerre |
| Céline | 1472 ± 245 | 122.5 | 16 | 2 | v2-p2-noms-de-pays-le-pays |
| Prince Henri d'Orléans | 1468 ± 427 | 213.6 | 2 | 1 | v3-p1 |
| le bâtonnier | 1466 ± 405 | 202.3 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| Barrès | 1465 ± 249 | 124.3 | 9 | 1 | v3-p2 |
| Clémenceau | 1465 ± 249 | 124.3 | 9 | 1 | v3-p2 |
| M. de Luxembourg | 1454 ± 434 | 217.0 | 2 | 1 | v3-p2 |
| la jeune ouvriere | 1451 ± 444 | 221.9 | 2 | 1 | v1-p2-un-amour-de-swann |
| Théodose Cadet | 1450 ± 372 | 186.2 | 3 | 1 | v3-p2 |
| Beauserfeuil | 1448 ± 372 | 186.2 | 3 | 1 | v3-p2 |
| Vigny | 1448 ± 484 | 242.0 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| M. d'Herweck | 1447 ± 300 | 150.0 | 5 | 2 | v4-p2 |
| M. de Stermaria | 1441 ± 255 | 127.5 | 10 | 4 | v2-p2-noms-de-pays-le-pays |
| duc de Châtellerault | 1438 ± 246 | 122.8 | 10 | 5 | v4-p2 |
| M. Molé | 1434 ± 299 | 149.7 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| M. de Bouillon | 1434 ± 299 | 149.7 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Musset | 1434 ± 299 | 149.7 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victor Hugo | 1434 ± 299 | 149.7 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Antoine | 1433 ± 412 | 206.2 | 3 | 1 | v3-p1 |
| marquise de Citri | 1423 ± 413 | 206.7 | 2 | 1 | v4-p2 |
| le prince Von | 1415 ± 243 | 121.3 | 10 | 2 | v3-p2 |
| Madame Elstir | 1398 ± 334 | 167.2 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| les demoiselles d’Ambresac | 1398 ± 334 | 167.2 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| Cartier | 1396 ± 339 | 169.6 | 4 | 1 | v5 |
| M. de Grouchy | 1364 ± 254 | 126.8 | 10 | 4 | v3-p2 |
| professeur E… | 1363 ± 370 | 184.9 | 4 | 2 | v4-p2 |
| marquise de Gallardon | 1362 ± 222 | 111.0 | 19 | 7 | v4-p2 |
| princesse d'Iéna | 1356 ± 460 | 229.9 | 3 | 1 | v1-p2-un-amour-de-swann |
| prince Foggi | 1351 ± 500 | 250.2 | 1 | 1 | v6-p3 |
| comtesse G… | 1338 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| la Charité de Giotto | 1338 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| ma grand'tante | 1338 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| le diplomate belge | 1313 ± 429 | 214.4 | 2 | 1 | v3-p1 |
| le prince von *** | 1303 ± 415 | 207.4 | 2 | 1 | v3-p1 |
| les Courvoisier | 1301 ± 346 | 173.0 | 5 | 1 | v3-p2 |
| la marquise | 1294 ± 503 | 251.3 | 1 | 1 | v3-p1 |
| prince de Léon | 1290 ± 455 | 227.6 | 2 | 1 | v5 |
| Mme de Souvré | 1289 ± 266 | 133.1 | 11 | 2 | v4-p2 |
| le professeur E… | 1286 ± 455 | 227.3 | 2 | 1 | v4-p2 |
| le grand-duc héritier de Luxembourg | 1284 ± 504 | 252.1 | 1 | 1 | v3-p2 |
| Monsieur Vallenères | 1284 ± 419 | 209.5 | 3 | 1 | v3-p1 |
| Marie Gineste | 1281 ± 452 | 226.2 | 2 | 1 | v4-p2 |
| M. Bontemps | 1267 ± 285 | 142.5 | 9 | 2 | v7-p2-m-de-charlus-pendant-la-guerre |
| Maurice | 1263 ± 287 | 143.3 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| l'ambassadrice de Turquie | 1260 ± 326 | 163.1 | 4 | 1 | v4-p2 |
| Mme de Morienval | 1256 ± 350 | 174.9 | 6 | 1 | v3-p1 |
| duchesse de Luxembourg | 1256 ± 350 | 174.9 | 6 | 1 | v3-p1 |
| vicomtesse d'Égremont | 1254 ± 403 | 201.6 | 3 | 1 | v3-p2 |
| Dumont | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| Madame d'Ambresac | 1253 ± 512 | 256.0 | 2 | 1 | v3-p1 |
| le curé | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| prince de Faffenheim | 1252 ± 361 | 180.7 | 3 | 2 | v3-p1 |
| capitaine de Borodino | 1250 ± 208 | 104.1 | 14 | 5 | v3-p1 |
| l'historien de la Fronde | 1244 ± 399 | 199.5 | 3 | 1 | v3-p1 |
| Mme de Simiane | 1244 ± 426 | 213.2 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| Mme Blatin | 1233 ± 438 | 219.1 | 2 | 1 | v1-p3-noms-de-pays-le-nom |
| la cousine d'Oriane | 1233 ± 368 | 184.2 | 3 | 1 | v3-p2 |
| Alix | 1219 ± 253 | 126.5 | 9 | 3 | v3-p1 |
| l'empereur | 1197 ± 366 | 182.8 | 4 | 1 | v3-p2 |
| le prince de Faffenheim | 1196 ± 305 | 152.7 | 5 | 1 | v3-p1 |
| Mme Iéna | 1189 ± 316 | 157.8 | 5 | 1 | v3-p2 |
| Mme de Varambon | 1178 ± 355 | 177.5 | 4 | 2 | v3-p2 |
| Mme Blandais | 1178 ± 367 | 183.6 | 4 | 2 | v2-p2-noms-de-pays-le-pays |
| colonel de Froberville | 1157 ± 208 | 103.9 | 14 | 1 | v4-p2 |
| Picquart | 1150 ± 281 | 140.3 | 8 | 2 | v3-p1 |
| M. Pierre | 1136 ± 372 | 186.2 | 4 | 2 | v3-p1 |
| M. de Vigny | 1133 ± 299 | 149.7 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| ma grand’tante | 1105 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |

## Largest Glicko-vs-ELO Rank Divergences

| Character | Glicko Rank | ELO Rank | Delta | Rating | ELO |
| --- | --- | --- | --- | --- | --- |
| duchesse de Guermantes | 34 | 282 | -248 | 1515 ± 59 | 1374.47 |
| Albertine | 46 | 284 | -238 | 1495 ± 68 | 1371.816 |
| baron de Charlus | 43 | 281 | -238 | 1499 ± 60 | 1390.62 |
| Gilberte | 51 | 286 | -235 | 1484 ± 61 | 1330.433 |
| princesse de Guermantes | 47 | 273 | -226 | 1516 ± 91 | 1419.165 |
| duc de Guermantes | 67 | 287 | -220 | 1458 ± 67 | 1327.969 |
| Robert de Saint-Loup | 44 | 252 | -208 | 1493 ± 60 | 1461.689 |
| Brichot | 53 | 260 | -207 | 1495 ± 80 | 1449.343 |
| Swann | 59 | 262 | -203 | 1470 ± 61 | 1448.202 |
| Andrée | 64 | 246 | -182 | 1488 ± 91 | 1465.981 |

## Character Table

| Character | Rating | Conservative | RD | Volatility | Glicko Rank | ELO Rank | Provisional | Matches | W-L-D | Units | Mean Inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1918 ± 229 | 1689.1 | 114.7 | 0.06 | 1 | 1 | True | 17 | 16-0-1 | 3 | +0.184 |
| Mlle d'Oloron | 1856 ± 197 | 1659.5 | 98.4 | 0.06 | 2 | 4 | False | 14 | 14-0-0 | 1 | +0.39 |
| Céleste Albaret | 1866 ± 214 | 1651.5 | 107.0 | 0.06 | 3 | 3 | True | 17 | 16-1-0 | 3 | +1.2 |
| marquis de Beausergent | 1825 ± 203 | 1622.7 | 101.4 | 0.06 | 4 | 6 | True | 12 | 12-0-0 | 1 | -0.224 |
| docteur du Boulbon | 1754 ± 165 | 1589.8 | 82.3 | 0.06 | 5 | 5 | False | 27 | 19-3-5 | 6 | -0.469 |
| Eulalie | 1796 ± 237 | 1559.1 | 118.4 | 0.06 | 6 | 8 | True | 16 | 12-2-2 | 7 | +0.074 |
| Françoise | 1633 ± 82 | 1550.6 | 41.0 | 0.0602 | 7 | 9 | False | 217 | 100-48-69 | 82 | -0.369 |
| Mme Elstir | 1818 ± 273 | 1545.2 | 136.3 | 0.06 | 8 | 14 | True | 7 | 7-0-0 | 1 | +0.384 |
| Mlle de Saint-Loup | 1783 ± 251 | 1532.5 | 125.4 | 0.06 | 9 | 17 | True | 7 | 6-0-1 | 2 | +1.288 |
| comte de Forcheville | 1632 ± 100 | 1532.1 | 50.0 | 0.0605 | 10 | 88 | False | 112 | 55-18-39 | 25 | -0.4 |
| Bergotte | 1622 ± 93 | 1528.9 | 46.4 | 0.06 | 11 | 2 | False | 129 | 52-31-46 | 36 | -0.199 |
| Léa | 1720 ± 195 | 1525.0 | 97.3 | 0.06 | 12 | 16 | False | 14 | 8-0-6 | 4 | -0.7 |
| Marie | 1788 ± 272 | 1515.8 | 136.1 | 0.06 | 13 | 19 | True | 7 | 6-1-0 | 1 | -0.24 |
| le peintre | 1675 ± 160 | 1514.7 | 80.1 | 0.06 | 14 | 26 | False | 42 | 16-4-22 | 8 | -0.298 |
| M. Verdurin | 1596 ± 95 | 1501.2 | 47.3 | 0.06 | 15 | 15 | False | 110 | 38-23-49 | 27 | -0.64 |
| Mme de Grouchy | 1877 ± 377 | 1499.2 | 188.7 | 0.06 | 16 | 27 | True | 4 | 4-0-0 | 1 | +0.008 |
| le grand-père du narrateur | 1640 ± 146 | 1493.7 | 73.1 | 0.06 | 17 | 13 | False | 63 | 25-7-31 | 16 | -0.664 |
| Mlle de Stermaria | 1758 ± 270 | 1488.2 | 134.9 | 0.0601 | 18 | 25 | True | 10 | 6-3-1 | 5 | -0.671 |
| Elstir | 1578 ± 90 | 1487.4 | 45.2 | 0.0601 | 19 | 10 | False | 106 | 42-29-35 | 29 | +0.014 |
| Aimé | 1587 ± 100 | 1486.5 | 50.2 | 0.06 | 20 | 23 | False | 79 | 27-13-39 | 18 | -0.45 |
| Rachel | 1562 ± 78 | 1484.5 | 38.9 | 0.0607 | 21 | 11 | False | 146 | 52-53-41 | 43 | -1.09 |
| Mme de Chaussepierre | 1823 ± 339 | 1483.8 | 169.6 | 0.06 | 22 | 29 | True | 4 | 4-0-0 | 1 | +0.79 |
| la grand-mère | 1567 ± 84 | 1482.8 | 42.1 | 0.0605 | 23 | 38 | False | 225 | 93-65-67 | 80 | -0.444 |
| prince de Guermantes | 1570 ± 91 | 1478.7 | 45.5 | 0.0605 | 24 | 7 | False | 124 | 42-29-53 | 22 | -0.797 |
| l'amie de Mlle Vinteuil | 1613 ± 135 | 1478.0 | 67.6 | 0.06 | 25 | 21 | False | 44 | 17-6-21 | 12 | -0.361 |
| Jupien | 1576 ± 102 | 1473.6 | 51.1 | 0.06 | 26 | 44 | False | 68 | 23-12-33 | 18 | -0.063 |
| Norpois | 1563 ± 89 | 1473.4 | 44.6 | 0.0599 | 27 | 22 | False | 180 | 79-54-47 | 63 | -0.659 |
| la mère du narrateur | 1563 ± 89 | 1473.4 | 44.7 | 0.0599 | 28 | 179 | False | 144 | 55-36-53 | 40 | -0.477 |
| prince de Saxe | 1848 ± 375 | 1473.2 | 187.5 | 0.06 | 29 | 39 | True | 3 | 3-0-0 | 1 | +0.37 |
| Morel | 1546 ± 77 | 1468.8 | 38.5 | 0.0602 | 30 | 12 | False | 152 | 47-52-53 | 32 | -1.02 |
| Victurnien | 1734 ± 267 | 1466.8 | 133.7 | 0.06 | 31 | 20 | True | 8 | 5-0-3 | 2 | +1.058 |
| le père du narrateur | 1580 ± 120 | 1460.2 | 59.9 | 0.0599 | 32 | 18 | False | 90 | 34-22-34 | 24 | -0.79 |
| marquis de Bréauté | 1550 ± 93 | 1456.7 | 46.6 | 0.0599 | 33 | 49 | False | 101 | 26-21-54 | 19 | -0.811 |
| duchesse de Guermantes | 1515 ± 59 | 1456.6 | 29.4 | 0.0664 | 34 | 282 | False | 662 | 329-177-156 | 199 | -0.255 |
| Mme Verdurin | 1523 ± 67 | 1455.9 | 33.7 | 0.0599 | 35 | 143 | False | 311 | 93-104-114 | 82 | -0.909 |
| Mlle Vinteuil | 1564 ± 109 | 1455.3 | 54.6 | 0.06 | 36 | 89 | False | 71 | 19-15-37 | 15 | -0.693 |
| Rémi | 1677 ± 222 | 1454.6 | 111.2 | 0.06 | 37 | 32 | True | 17 | 4-0-13 | 3 | -0.533 |
| Mme Cottard | 1620 ± 166 | 1454.5 | 83.0 | 0.06 | 38 | 36 | False | 33 | 16-8-9 | 11 | -0.43 |
| Maeterlinck | 1783 ± 329 | 1454.0 | 164.3 | 0.06 | 39 | 28 | True | 5 | 4-0-1 | 1 | -0.8 |
| Bloch | 1515 ± 68 | 1447.7 | 33.8 | 0.0609 | 40 | 30 | False | 270 | 78-111-81 | 71 | -1.609 |
| Odette | 1513 ± 66 | 1447.2 | 33.0 | 0.0611 | 41 | 182 | False | 462 | 147-154-161 | 142 | -0.748 |
| colonel Picquart | 1791 ± 349 | 1441.6 | 174.5 | 0.06 | 42 | 42 | True | 4 | 4-0-0 | 1 | +1.555 |
| baron de Charlus | 1499 ± 60 | 1439.2 | 29.9 | 0.0619 | 43 | 281 | False | 485 | 185-155-145 | 119 | -0.8 |
| Robert de Saint-Loup | 1493 ± 60 | 1433.2 | 29.9 | 0.0608 | 44 | 252 | False | 508 | 167-208-133 | 168 | -0.623 |
| Mme de Charlus | 1623 ± 192 | 1430.4 | 96.1 | 0.06 | 45 | 35 | False | 15 | 5-1-9 | 2 | -0.8 |
| Albertine | 1495 ± 68 | 1427.0 | 34.1 | 0.0604 | 46 | 284 | False | 387 | 147-156-84 | 146 | -0.887 |
| princesse de Guermantes | 1516 ± 91 | 1425.8 | 45.3 | 0.0602 | 47 | 273 | False | 113 | 41-32-40 | 25 | -0.403 |
| Mme Goupil | 1606 ± 180 | 1425.7 | 89.9 | 0.06 | 48 | 37 | False | 17 | 5-1-11 | 2 | -0.8 |
| Mme de Surgis | 1554 ± 130 | 1424.6 | 64.9 | 0.06 | 49 | 43 | False | 42 | 16-11-15 | 9 | -0.99 |
| grand-duc héritier de Luxembourg | 1673 ± 250 | 1422.8 | 125.1 | 0.06 | 50 | 31 | True | 9 | 4-1-4 | 2 | +0.419 |
| Gilberte | 1484 ± 61 | 1422.5 | 30.6 | 0.0604 | 51 | 286 | False | 312 | 112-103-97 | 76 | -0.582 |
| docteur Cottard | 1499 ± 81 | 1417.7 | 40.6 | 0.06 | 52 | 101 | False | 194 | 48-64-82 | 43 | -0.899 |
| Brichot | 1495 ± 80 | 1415.4 | 39.9 | 0.06 | 53 | 260 | False | 135 | 30-32-73 | 21 | -0.877 |
| le narrateur | 1478 ± 63 | 1415.2 | 31.4 | 0.0831 | 54 | 221 | False | 1093 | 397-501-195 | 316 | -0.845 |
| Dreyfus | 1537 ± 123 | 1413.6 | 61.6 | 0.06 | 55 | 77 | False | 58 | 13-11-34 | 7 | -0.77 |
| Gribelin | 1725 ± 315 | 1410.2 | 157.3 | 0.06 | 56 | 33 | True | 6 | 5-1-0 | 1 | +0.04 |
| Mme Bontemps | 1525 ± 115 | 1409.3 | 57.7 | 0.0599 | 57 | 137 | False | 54 | 15-11-28 | 13 | -0.575 |
| Mme de Villeparisis | 1495 ± 86 | 1409.2 | 42.9 | 0.0605 | 58 | 228 | False | 236 | 89-94-53 | 79 | -0.749 |
| Swann | 1470 ± 61 | 1409.0 | 30.5 | 0.0645 | 59 | 262 | False | 667 | 207-308-152 | 202 | -1.023 |
| Mme Sazerat | 1599 ± 192 | 1406.4 | 96.2 | 0.06 | 60 | 24 | False | 20 | 8-2-10 | 6 | -0.692 |
| marquis du Lau | 1714 ± 308 | 1406.0 | 153.9 | 0.06 | 61 | 46 | True | 5 | 4-1-0 | 2 | +1.032 |
| M. Vinteuil | 1520 ± 118 | 1401.7 | 59.0 | 0.06 | 62 | 220 | False | 61 | 18-19-24 | 15 | -0.444 |
| marquise de Saint-Euverte | 1505 ± 106 | 1399.0 | 52.9 | 0.0602 | 63 | 70 | False | 72 | 16-27-29 | 13 | -1.784 |
| Andrée | 1488 ± 91 | 1396.5 | 45.7 | 0.0599 | 64 | 246 | False | 114 | 35-42-37 | 31 | -0.815 |
| M. Ski | 1559 ± 166 | 1392.9 | 82.9 | 0.06 | 65 | 98 | False | 21 | 4-1-16 | 2 | -0.4 |
| Mme Leroi | 1600 ± 207 | 1392.1 | 103.7 | 0.06 | 66 | 40 | True | 13 | 8-4-1 | 5 | -0.994 |
| duc de Guermantes | 1458 ± 67 | 1391.4 | 33.3 | 0.0602 | 67 | 287 | False | 401 | 120-171-110 | 110 | -1.042 |
| M. d'Argencourt | 1514 ± 124 | 1389.7 | 62.1 | 0.06 | 68 | 218 | False | 56 | 19-18-19 | 14 | -1.123 |
| général de Froberville | 1554 ± 171 | 1382.8 | 85.6 | 0.06 | 69 | 41 | False | 27 | 7-4-16 | 7 | -0.596 |
| duchesse de La Trémoïlle | 1799 ± 423 | 1375.6 | 211.6 | 0.06 | 70 | 45 | True | 3 | 3-0-0 | 1 | +0.665 |
| Mme de Marsantes | 1468 ± 95 | 1373.9 | 47.3 | 0.06 | 71 | 214 | False | 107 | 19-33-55 | 21 | -1.234 |
| le pianiste | 1642 ± 271 | 1370.8 | 135.5 | 0.06 | 72 | 69 | True | 10 | 4-2-4 | 3 | +0.438 |
| comtesse Molé | 1515 ± 148 | 1367.7 | 73.8 | 0.06 | 73 | 157 | False | 34 | 6-9-19 | 6 | -1.288 |
| M. Nissim Bernard | 1497 ± 130 | 1366.9 | 64.8 | 0.06 | 74 | 219 | False | 39 | 9-10-20 | 10 | -1.502 |
| duc d'Aumale | 1744 ± 377 | 1366.8 | 188.7 | 0.06 | 75 | 73 | True | 4 | 3-1-0 | 2 | +0.247 |
| le directeur | 1504 ± 138 | 1365.2 | 69.2 | 0.06 | 76 | 254 | False | 39 | 11-14-14 | 11 | -0.851 |
| le jeune marquis de Cambremer | 1564 ± 202 | 1362.0 | 100.8 | 0.06 | 77 | 108 | True | 12 | 2-0-10 | 1 | -1.2 |
| Bloch père | 1488 ± 128 | 1359.7 | 64.1 | 0.06 | 78 | 180 | False | 47 | 11-11-25 | 8 | -1.614 |
| Émilie Daltier | 1718 ± 358 | 1359.7 | 179.1 | 0.06 | 79 | 65 | True | 3 | 2-0-1 | 1 | -0.4 |
| duc de Chartres | 1566 ± 208 | 1358.0 | 103.9 | 0.06 | 80 | 47 | True | 14 | 2-0-12 | 1 | -0.8 |
| prince de Chimay | 1566 ± 208 | 1358.0 | 103.9 | 0.06 | 81 | 54 | True | 14 | 2-0-12 | 1 | -0.8 |
| Legrandin | 1457 ± 101 | 1356.7 | 50.4 | 0.0601 | 82 | 247 | False | 83 | 15-28-40 | 24 | -1.22 |
| prince des Laumes | 1532 ± 176 | 1355.5 | 88.2 | 0.06 | 83 | 125 | False | 27 | 4-3-20 | 3 | -0.8 |
| M. d'Orsan | 1624 ± 269 | 1355.0 | 134.7 | 0.06 | 84 | 58 | True | 11 | 2-0-9 | 1 | -0.8 |
| princesse de Parme | 1455 ± 101 | 1353.7 | 50.7 | 0.06 | 85 | 34 | False | 130 | 35-65-30 | 38 | -0.839 |
| la marquise douairière de Cambremer | 1503 ± 151 | 1352.3 | 75.5 | 0.06 | 86 | 81 | False | 31 | 9-6-16 | 6 | -0.063 |
| Charcot | 1577 ± 225 | 1352.2 | 112.3 | 0.06 | 87 | 92 | True | 12 | 3-2-7 | 1 | -0.8 |
| M. Reinach | 1577 ± 225 | 1352.2 | 112.3 | 0.06 | 88 | 97 | True | 12 | 3-2-7 | 1 | -0.8 |
| Bismarck | 1704 ± 353 | 1351.1 | 176.4 | 0.06 | 89 | 72 | True | 4 | 3-1-0 | 1 | +0.214 |
| princesse de Luxembourg | 1518 ± 167 | 1350.8 | 83.4 | 0.06 | 90 | 130 | False | 25 | 7-6-12 | 6 | -0.782 |
| M. de Chevregny | 1542 ± 196 | 1346.4 | 97.9 | 0.06 | 91 | 61 | False | 16 | 4-1-11 | 1 | -0.4 |
| M. de Crécy | 1542 ± 196 | 1346.4 | 97.9 | 0.06 | 92 | 63 | False | 16 | 4-1-11 | 1 | -0.4 |
| Mme Féré | 1542 ± 196 | 1346.4 | 97.9 | 0.06 | 93 | 67 | False | 16 | 4-1-11 | 1 | -0.4 |
| la Berma | 1457 ± 111 | 1346.3 | 55.4 | 0.0602 | 94 | 276 | False | 62 | 19-24-19 | 19 | -0.336 |
| Mlle d'Éporcheville | 1569 ± 224 | 1345.3 | 111.9 | 0.06 | 95 | 114 | True | 10 | 3-2-5 | 2 | -0.6 |
| M. de Goncourt | 1582 ± 237 | 1344.8 | 118.4 | 0.06 | 96 | 109 | True | 8 | 2-0-6 | 1 | -1.2 |
| Mme de Montmorency | 1564 ± 221 | 1343.1 | 110.7 | 0.06 | 97 | 126 | True | 11 | 2-0-9 | 1 | -0.8 |
| Mme de Rochechouart | 1564 ± 221 | 1343.1 | 110.7 | 0.06 | 98 | 128 | True | 11 | 2-0-9 | 1 | -0.8 |
| marquis Maurice de Vaudémont | 1810 ± 468 | 1341.7 | 234.0 | 0.06 | 99 | 48 | True | 2 | 2-0-0 | 1 | +0.843 |
| Esther | 1539 ± 199 | 1340.1 | 99.4 | 0.06 | 100 | 110 | False | 14 | 3-2-9 | 2 | -1.0 |
| Lady Rufus Israël | 1617 ± 280 | 1336.7 | 140.2 | 0.06 | 101 | 116 | True | 6 | 2-1-3 | 1 | -0.4 |
| la duchesse d'Alençon | 1650 ± 314 | 1336.2 | 157.0 | 0.06 | 102 | 50 | True | 6 | 3-1-2 | 1 | -0.8 |
| Mlle Bloch | 1762 ± 426 | 1335.9 | 213.1 | 0.06 | 103 | 59 | True | 2 | 2-0-0 | 1 | +1.343 |
| prince d’Agrigente | 1537 ± 202 | 1335.1 | 101.1 | 0.06 | 104 | 122 | True | 15 | 3-2-10 | 2 | -0.8 |
| les La Trémoïlle | 1663 ± 329 | 1334.7 | 164.3 | 0.06 | 105 | 56 | True | 7 | 2-0-5 | 1 | -0.8 |
| Herbinger | 1735 ± 408 | 1326.7 | 204.1 | 0.06 | 106 | 53 | True | 3 | 2-0-1 | 1 | -0.8 |
| Marie-Aynard | 1636 ± 311 | 1325.3 | 155.5 | 0.06 | 107 | 52 | True | 7 | 2-0-5 | 1 | -0.8 |
| Victurnienne | 1636 ± 311 | 1325.3 | 155.5 | 0.06 | 108 | 51 | True | 7 | 2-0-5 | 1 | -0.8 |
| Létourville | 1676 ± 353 | 1323.5 | 176.5 | 0.06 | 109 | 80 | True | 3 | 2-0-1 | 1 | -0.8 |
| Mme de Stermaria | 1632 ± 309 | 1322.9 | 154.6 | 0.06 | 110 | 83 | True | 5 | 2-1-2 | 1 | -0.8 |
| duc de Sidonia | 1771 ± 455 | 1316.4 | 227.3 | 0.06 | 111 | 57 | True | 2 | 2-0-0 | 1 | -0.88 |
| Dechambre | 1683 ± 368 | 1315.9 | 183.8 | 0.06 | 112 | 74 | True | 3 | 2-0-1 | 1 | -0.96 |
| Mme Timoléon d'Amoncourt | 1557 ± 244 | 1313.2 | 121.9 | 0.06 | 113 | 104 | True | 9 | 2-1-6 | 1 | -0.4 |
| le commandant Duroc | 1726 ± 413 | 1313.0 | 206.5 | 0.06 | 114 | 71 | True | 2 | 2-0-0 | 1 | +0.256 |
| Goncourt | 1485 ± 174 | 1310.9 | 87.2 | 0.06 | 115 | 238 | False | 16 | 2-3-11 | 2 | -0.8 |
| Lady Israels | 1766 ± 456 | 1310.1 | 228.0 | 0.06 | 116 | 79 | True | 2 | 2-0-0 | 1 | 0.0 |
| cousine Poictiers | 1625 ± 321 | 1303.6 | 160.7 | 0.06 | 117 | 93 | True | 5 | 2-1-2 | 1 | -0.4 |
| duc de Poictiers | 1625 ± 321 | 1303.6 | 160.7 | 0.06 | 118 | 96 | True | 5 | 2-1-2 | 1 | -0.4 |
| M. de Beauserfeuil | 1581 ± 278 | 1303.3 | 138.9 | 0.06 | 119 | 62 | True | 7 | 2-1-4 | 1 | -0.8 |
| Sarah Bernhardt | 1589 ± 287 | 1302.0 | 143.3 | 0.06 | 120 | 94 | True | 7 | 2-0-5 | 1 | -0.8 |
| le jeune prince de Foix | 1589 ± 287 | 1302.0 | 143.3 | 0.06 | 121 | 86 | True | 7 | 2-0-5 | 1 | -0.8 |
| vicomte de Courvoisier | 1589 ± 287 | 1302.0 | 143.3 | 0.06 | 122 | 100 | True | 7 | 2-0-5 | 1 | -0.8 |
| princesse d'Épinay | 1540 ± 241 | 1298.3 | 120.6 | 0.06 | 123 | 105 | True | 12 | 4-3-5 | 3 | -0.533 |
| Duroc | 1751 ± 454 | 1296.6 | 227.1 | 0.06 | 124 | 68 | True | 2 | 2-0-0 | 1 | +1.205 |
| Mme de Sévigné | 1483 ± 188 | 1294.6 | 94.1 | 0.06 | 125 | 102 | False | 25 | 7-5-13 | 4 | -0.065 |
| prince de Sagan | 1571 ± 277 | 1294.0 | 138.5 | 0.06 | 126 | 85 | True | 7 | 1-0-6 | 1 | -0.8 |
| général de Monserfeuil | 1489 ± 195 | 1293.6 | 97.7 | 0.06 | 127 | 195 | False | 18 | 6-7-5 | 4 | -1.481 |
| Léonor de Cambremer | 1494 ± 203 | 1291.3 | 101.4 | 0.06 | 128 | 185 | True | 12 | 1-1-10 | 1 | -0.8 |
| oncle Adolphe | 1505 ± 219 | 1285.9 | 109.7 | 0.06 | 129 | 224 | True | 20 | 4-7-9 | 6 | -1.773 |
| Manet | 1600 ± 316 | 1284.5 | 158.0 | 0.06 | 130 | 75 | True | 5 | 1-0-4 | 1 | -0.8 |
| Bibi | 1709 ± 425 | 1284.2 | 212.5 | 0.06 | 131 | 55 | True | 2 | 2-0-0 | 1 | +0.16 |
| Mme de Cambremer | 1374 ± 94 | 1280.0 | 46.9 | 0.0601 | 132 | 280 | False | 112 | 12-54-46 | 20 | -1.51 |
| Mme Legrandin mère | 1558 ± 280 | 1277.7 | 140.0 | 0.06 | 133 | 76 | True | 8 | 2-0-6 | 1 | -0.8 |
| Victoire | 1558 ± 280 | 1277.7 | 140.0 | 0.06 | 134 | 82 | True | 8 | 2-0-6 | 1 | -0.8 |
| l'abbé Poiré | 1502 ± 224 | 1277.7 | 112.0 | 0.06 | 135 | 138 | True | 10 | 1-2-7 | 1 | -0.8 |
| le baron Bréau-Chenut | 1586 ± 308 | 1277.7 | 154.1 | 0.06 | 136 | 90 | True | 7 | 3-1-3 | 1 | -0.8 |
| le vieux père Chenut | 1586 ± 308 | 1277.7 | 154.1 | 0.06 | 137 | 95 | True | 7 | 3-1-3 | 1 | -0.8 |
| Balzac | 1462 ± 184 | 1277.4 | 92.2 | 0.06 | 138 | 240 | False | 18 | 2-4-12 | 2 | -0.8 |
| Élisabeth | 1558 ± 284 | 1274.4 | 142.0 | 0.06 | 139 | 123 | True | 6 | 2-1-3 | 1 | -1.2 |
| duchesse de Létourville | 1565 ± 291 | 1274.2 | 145.5 | 0.06 | 140 | 131 | True | 5 | 2-1-2 | 1 | -0.8 |
| M. de Bornier | 1620 ± 346 | 1274.0 | 173.2 | 0.06 | 141 | 64 | True | 5 | 3-1-1 | 1 | -1.2 |
| Lady Israël | 1600 ± 327 | 1272.6 | 163.5 | 0.06 | 142 | 103 | True | 5 | 2-1-2 | 1 | -0.4 |
| M. Vibert | 1645 ± 374 | 1270.6 | 187.0 | 0.06 | 143 | 78 | True | 3 | 1-0-2 | 1 | -0.4 |
| Thibaud | 1514 ± 244 | 1270.0 | 121.9 | 0.06 | 144 | 168 | True | 8 | 2-2-4 | 1 | -0.8 |
| Mme Putbus | 1519 ± 251 | 1268.4 | 125.4 | 0.06 | 145 | 135 | True | 8 | 1-1-6 | 1 | -0.8 |
| M. de Chateaubriand | 1554 ± 285 | 1268.1 | 142.7 | 0.06 | 146 | 213 | True | 11 | 1-3-7 | 2 | -1.849 |
| Flora | 1602 ± 336 | 1265.5 | 168.2 | 0.06 | 147 | 60 | True | 8 | 3-1-4 | 1 | -0.8 |
| M. de Marsantes | 1568 ± 304 | 1263.9 | 151.9 | 0.06 | 148 | 124 | True | 7 | 2-1-4 | 2 | -0.312 |
| le petit Cambremer | 1460 ± 197 | 1263.5 | 98.4 | 0.06 | 149 | 225 | False | 14 | 1-3-10 | 1 | -0.8 |
| princesse de Silistrie | 1460 ± 197 | 1263.5 | 98.4 | 0.06 | 150 | 223 | False | 14 | 1-3-10 | 1 | -0.8 |
| prince de Foix | 1475 ± 214 | 1260.9 | 107.0 | 0.06 | 151 | 156 | True | 14 | 4-4-6 | 3 | -0.893 |
| comte de Paris | 1509 ± 250 | 1259.1 | 125.1 | 0.06 | 152 | 207 | True | 10 | 3-4-3 | 3 | -0.667 |
| M. Arthur Meyer | 1528 ± 269 | 1259.0 | 134.7 | 0.06 | 153 | 169 | True | 6 | 2-2-2 | 1 | -0.8 |
| tante Léonie | 1439 ± 181 | 1258.0 | 90.4 | 0.0601 | 154 | 256 | False | 38 | 12-22-4 | 22 | -0.825 |
| monsieur Vallenères | 1685 ± 427 | 1257.3 | 213.7 | 0.06 | 155 | 66 | True | 2 | 2-0-0 | 1 | -0.8 |
| M. Carnot | 1506 ± 252 | 1253.9 | 126.1 | 0.06 | 156 | 139 | True | 9 | 1-1-7 | 1 | -0.8 |
| Mme Carnot | 1506 ± 252 | 1253.9 | 126.1 | 0.06 | 157 | 142 | True | 9 | 1-1-7 | 1 | -0.8 |
| Gisèle | 1506 ± 257 | 1248.7 | 128.7 | 0.06 | 158 | 206 | True | 14 | 3-6-5 | 5 | -2.15 |
| Mme d'Arpajon | 1396 ± 147 | 1248.5 | 73.5 | 0.06 | 159 | 270 | False | 37 | 7-20-10 | 8 | -1.53 |
| Sir Rufus Israël | 1530 ± 282 | 1248.2 | 141.0 | 0.06 | 160 | 115 | True | 7 | 3-1-3 | 1 | -0.8 |
| comtesse douairière d'Argencourt | 1484 ± 244 | 1240.2 | 121.9 | 0.06 | 161 | 204 | True | 10 | 1-2-7 | 1 | -0.8 |
| duchesse de Gallardon douairière | 1484 ± 244 | 1240.2 | 121.9 | 0.06 | 162 | 201 | True | 10 | 1-2-7 | 1 | -0.8 |
| marquis de Fierbois | 1484 ± 244 | 1240.2 | 121.9 | 0.06 | 163 | 212 | True | 10 | 1-2-7 | 1 | -0.8 |
| M. de Vaugoubert | 1399 ± 159 | 1239.7 | 79.5 | 0.06 | 164 | 253 | False | 35 | 6-12-17 | 9 | -1.383 |
| Dostoïevski | 1512 ± 273 | 1238.5 | 136.5 | 0.06 | 165 | 177 | True | 6 | 1-1-4 | 1 | -0.8 |
| Rosemonde | 1429 ± 191 | 1238.2 | 95.3 | 0.06 | 166 | 203 | False | 20 | 5-7-8 | 4 | -0.7 |
| Mlle de l’Orgeville | 1588 ± 352 | 1235.5 | 176.1 | 0.06 | 167 | 136 | True | 3 | 1-0-2 | 1 | -0.8 |
| jeune blonde de Rivebelle | 1568 ± 334 | 1233.3 | 167.2 | 0.06 | 168 | 99 | True | 6 | 2-1-3 | 2 | -0.4 |
| Arnulphe | 1594 ± 362 | 1232.2 | 181.0 | 0.06 | 169 | 91 | True | 4 | 1-0-3 | 1 | -0.4 |
| Mme de Vaugoubert | 1488 ± 257 | 1231.3 | 128.6 | 0.06 | 170 | 234 | True | 9 | 1-3-5 | 2 | -1.734 |
| Céline | 1472 ± 245 | 1227.0 | 122.5 | 0.06 | 171 | 183 | True | 16 | 4-6-6 | 2 | -1.14 |
| le grand-duc Wladimir | 1593 ± 366 | 1226.9 | 182.9 | 0.06 | 172 | 106 | True | 3 | 2-1-0 | 1 | -0.4 |
| d’Orgeville | 1495 ± 269 | 1226.5 | 134.4 | 0.06 | 173 | 132 | True | 7 | 1-1-5 | 1 | -0.8 |
| prince Von | 1484 ± 259 | 1225.2 | 129.3 | 0.06 | 174 | 134 | True | 8 | 3-3-2 | 3 | -1.463 |
| d'Orléans | 1575 ± 351 | 1223.8 | 175.6 | 0.06 | 175 | 107 | True | 5 | 2-1-2 | 1 | -0.8 |
| Sainte-Beuve | 1485 ± 266 | 1219.0 | 132.8 | 0.06 | 176 | 181 | True | 7 | 1-2-4 | 1 | -0.8 |
| Coquelin | 1539 ± 321 | 1218.4 | 160.3 | 0.06 | 177 | 166 | True | 5 | 1-1-3 | 1 | -0.8 |
| Barrès | 1465 ± 249 | 1216.1 | 124.3 | 0.06 | 178 | 148 | True | 9 | 1-1-7 | 1 | -0.8 |
| Clémenceau | 1465 ± 249 | 1216.1 | 124.3 | 0.06 | 179 | 150 | True | 9 | 1-1-7 | 1 | -0.8 |
| Napoléon III | 1523 ± 307 | 1216.0 | 153.4 | 0.06 | 180 | 198 | True | 8 | 1-2-5 | 1 | -0.8 |
| le marquis de Ganançay | 1566 ± 350 | 1216.0 | 174.9 | 0.06 | 181 | 84 | True | 6 | 3-1-2 | 1 | -0.8 |
| le marquis de Palancy | 1566 ± 350 | 1216.0 | 174.9 | 0.06 | 182 | 87 | True | 6 | 3-1-2 | 1 | -0.8 |
| Liszt | 1504 ± 304 | 1200.6 | 151.8 | 0.06 | 183 | 140 | True | 6 | 2-1-3 | 1 | -0.8 |
| Mme Ristori | 1504 ± 304 | 1200.6 | 151.8 | 0.06 | 184 | 141 | True | 6 | 2-1-3 | 1 | -0.8 |
| le roi Théodose | 1488 ± 289 | 1198.3 | 144.7 | 0.06 | 185 | 173 | True | 8 | 2-3-3 | 3 | -0.189 |
| comtesse de Monteriender | 1548 ± 352 | 1195.6 | 175.9 | 0.06 | 186 | 155 | True | 4 | 1-1-2 | 1 | 0.0 |
| duc de Châtellerault | 1438 ± 246 | 1192.0 | 122.8 | 0.06 | 187 | 189 | True | 10 | 2-5-3 | 5 | -1.422 |
| Mme de Sagan | 1559 ± 369 | 1189.7 | 184.7 | 0.06 | 188 | 118 | True | 3 | 1-0-2 | 1 | -0.4 |
| M. de La Rochefoucauld | 1516 ± 327 | 1189.5 | 163.3 | 0.06 | 189 | 127 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de La Rochefoucauld | 1516 ± 327 | 1189.5 | 163.3 | 0.06 | 190 | 117 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de Praslin | 1516 ± 327 | 1189.5 | 163.3 | 0.06 | 191 | 119 | True | 6 | 2-1-3 | 1 | -0.8 |
| elle | 1689 ± 502 | 1186.9 | 251.2 | 0.06 | 192 | 129 | True | 1 | 1-0-0 | 1 | -0.05 |
| princesse Sherbatoff | 1359 ± 172 | 1186.9 | 86.2 | 0.06 | 193 | 277 | False | 19 | 5-13-1 | 5 | -0.787 |
| M. de Stermaria | 1441 ± 255 | 1185.9 | 127.5 | 0.06 | 194 | 250 | True | 10 | 3-5-2 | 4 | -1.108 |
| Mme Trombert | 1532 ± 347 | 1185.7 | 173.3 | 0.06 | 195 | 176 | True | 4 | 1-1-2 | 1 | -0.4 |
| marquis de Cambremer | 1318 ± 134 | 1183.3 | 67.2 | 0.0601 | 196 | 285 | False | 45 | 7-23-15 | 6 | -1.016 |
| le prince Von | 1415 ± 243 | 1171.9 | 121.3 | 0.06 | 197 | 211 | True | 10 | 3-5-2 | 2 | -1.226 |
| D'Annunzio | 1485 ± 319 | 1166.0 | 159.4 | 0.06 | 198 | 178 | True | 5 | 1-2-2 | 1 | -0.4 |
| Poullein | 1649 ± 485 | 1163.8 | 242.6 | 0.06 | 199 | 145 | True | 2 | 1-1-0 | 2 | -0.575 |
| princesse Mathilde | 1476 ± 317 | 1158.2 | 158.7 | 0.06 | 200 | 187 | True | 7 | 2-3-2 | 2 | -0.6 |
| docteur Dieulafoy | 1660 ± 502 | 1157.8 | 251.0 | 0.06 | 201 | 121 | True | 1 | 1-0-0 | 1 | +2.699 |
| M. d'Herweck | 1447 ± 300 | 1147.3 | 150.0 | 0.06 | 202 | 164 | True | 5 | 2-3-0 | 2 | -1.57 |
| M. Swann, le père | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 203 | 146 | True | 7 | 1-1-5 | 1 | -0.8 |
| le comte de Paris | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 204 | 158 | True | 7 | 1-1-5 | 1 | -0.8 |
| le prince de Galles | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 205 | 159 | True | 7 | 1-1-5 | 1 | -0.8 |
| marquise de Gallardon | 1362 ± 222 | 1140.4 | 111.0 | 0.06 | 206 | 266 | True | 19 | 2-10-7 | 7 | -1.717 |
| L’excellent écrivain G… | 1488 ± 351 | 1137.1 | 175.3 | 0.06 | 207 | 133 | True | 4 | 1-1-2 | 1 | -0.8 |
| M. Molé | 1434 ± 299 | 1134.9 | 149.7 | 0.06 | 208 | 208 | True | 8 | 1-2-5 | 1 | -0.8 |
| M. de Bouillon | 1434 ± 299 | 1134.9 | 149.7 | 0.06 | 209 | 205 | True | 8 | 1-2-5 | 1 | -0.8 |
| Musset | 1434 ± 299 | 1134.9 | 149.7 | 0.06 | 210 | 197 | True | 8 | 1-2-5 | 1 | -0.8 |
| Victor Hugo | 1434 ± 299 | 1134.9 | 149.7 | 0.06 | 211 | 194 | True | 8 | 1-2-5 | 1 | -0.8 |
| Mme d'Heudicourt | 1333 ± 199 | 1133.8 | 99.4 | 0.06 | 212 | 268 | False | 18 | 3-11-4 | 5 | -1.482 |
| M. de Grouchy | 1364 ± 254 | 1110.6 | 126.8 | 0.06 | 213 | 245 | True | 10 | 2-7-1 | 4 | -0.741 |
| Mme de Franquetot | 1279 ± 170 | 1109.7 | 84.8 | 0.0601 | 214 | 279 | False | 23 | 4-13-6 | 3 | -1.092 |
| Théodore | 1624 ± 514 | 1109.6 | 257.0 | 0.06 | 215 | 113 | True | 2 | 1-0-1 | 1 | +1.638 |
| M. de Miribel | 1476 ± 372 | 1103.5 | 186.2 | 0.06 | 216 | 170 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Henry | 1476 ± 372 | 1103.5 | 186.2 | 0.06 | 217 | 175 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Picquart | 1476 ± 372 | 1103.5 | 186.2 | 0.06 | 218 | 174 | True | 4 | 1-1-2 | 1 | -0.8 |
| M. de Courgivaux | 1662 ± 581 | 1081.7 | 290.3 | 0.06 | 219 | 111 | True | 1 | 1-0-0 | 1 | +1.854 |
| Mme de Villebon | 1662 ± 585 | 1077.6 | 292.4 | 0.06 | 220 | 112 | True | 1 | 1-0-0 | 1 | -1.0 |
| Théodose Cadet | 1450 ± 372 | 1077.2 | 186.2 | 0.06 | 221 | 202 | True | 3 | 1-2-0 | 1 | -2.348 |
| baron de Guermantes | 1662 ± 585 | 1077.2 | 292.6 | 0.06 | 222 | 120 | True | 1 | 1-0-0 | 1 | -0.4 |
| Beauserfeuil | 1448 ± 372 | 1076.0 | 186.2 | 0.06 | 223 | 210 | True | 3 | 1-2-0 | 1 | -0.84 |
| le capitaine | 1489 ± 413 | 1076.0 | 206.5 | 0.06 | 224 | 167 | True | 2 | 1-1-0 | 1 | +0.04 |
| docteur Percepied | 1500 ± 426 | 1074.1 | 212.9 | 0.06 | 225 | 160 | True | 4 | 1-1-2 | 1 | -0.8 |
| Octave | 1497 ± 426 | 1071.2 | 213.0 | 0.06 | 226 | 171 | True | 4 | 2-2-0 | 2 | -0.473 |
| Madame Elstir | 1398 ± 334 | 1064.2 | 167.2 | 0.06 | 227 | 196 | True | 6 | 1-2-3 | 1 | -0.8 |
| les demoiselles d’Ambresac | 1398 ± 334 | 1064.2 | 167.2 | 0.06 | 228 | 188 | True | 6 | 1-2-3 | 1 | -0.8 |
| le bâtonnier | 1466 ± 405 | 1061.3 | 202.3 | 0.06 | 229 | 147 | True | 3 | 1-1-1 | 1 | -0.4 |
| Cartier | 1396 ± 339 | 1057.2 | 169.6 | 0.06 | 230 | 227 | True | 4 | 1-3-0 | 1 | -1.635 |
| M. Grevy | 1476 ± 423 | 1053.0 | 211.6 | 0.06 | 231 | 162 | True | 3 | 1-1-1 | 1 | -0.4 |
| Saniette | 1204 ± 158 | 1046.9 | 78.8 | 0.0601 | 232 | 288 | False | 35 | 1-27-7 | 9 | -3.263 |
| capitaine de Borodino | 1250 ± 208 | 1042.3 | 104.1 | 0.06 | 233 | 278 | True | 14 | 2-11-1 | 5 | -1.769 |
| Prince Henri d'Orléans | 1468 ± 427 | 1040.7 | 213.6 | 0.06 | 234 | 165 | True | 2 | 1-1-0 | 1 | -1.401 |
| prince d'Agrigente | 1487 ± 451 | 1036.1 | 225.5 | 0.06 | 235 | 163 | True | 2 | 1-1-0 | 2 | -0.37 |
| M. Barrère | 1528 ± 500 | 1027.4 | 250.2 | 0.06 | 236 | 149 | True | 1 | 0-0-1 | 1 | -1.401 |
| Mme de Souvré | 1289 ± 266 | 1023.1 | 133.1 | 0.06 | 237 | 267 | True | 11 | 2-9-0 | 2 | -1.916 |
| Antoine | 1433 ± 412 | 1020.4 | 206.2 | 0.06 | 238 | 216 | True | 3 | 0-2-1 | 1 | -0.8 |
| M. de Luxembourg | 1454 ± 434 | 1019.6 | 217.0 | 0.06 | 239 | 199 | True | 2 | 0-1-1 | 1 | -0.394 |
| marquise de Citri | 1423 ± 413 | 1010.0 | 206.7 | 0.06 | 240 | 190 | True | 2 | 0-1-1 | 1 | -2.945 |
| la jeune ouvriere | 1451 ± 444 | 1007.4 | 221.9 | 0.06 | 241 | 172 | True | 2 | 0-1-1 | 1 | -0.4 |
| professeur E… | 1363 ± 370 | 992.9 | 184.9 | 0.06 | 242 | 237 | True | 4 | 1-3-0 | 2 | -1.529 |
| princesse de Nassau | 1484 ± 496 | 987.4 | 248.1 | 0.06 | 243 | 144 | True | 1 | 0-0-1 | 1 | -2.45 |
| M. Bontemps | 1267 ± 285 | 982.3 | 142.5 | 0.06 | 244 | 269 | True | 9 | 1-7-1 | 2 | -0.483 |
| Maurice | 1263 ± 287 | 976.0 | 143.3 | 0.06 | 245 | 265 | True | 7 | 1-6-0 | 1 | -3.169 |
| Alix | 1219 ± 253 | 965.6 | 126.5 | 0.06 | 246 | 272 | True | 9 | 0-7-2 | 3 | -2.733 |
| Vigny | 1448 ± 484 | 963.9 | 242.0 | 0.06 | 247 | 161 | True | 2 | 1-1-0 | 1 | -1.632 |
| les Courvoisier | 1301 ± 346 | 954.5 | 173.0 | 0.06 | 248 | 242 | True | 5 | 1-4-0 | 1 | -1.456 |
| colonel de Froberville | 1157 ± 208 | 949.7 | 103.9 | 0.06 | 249 | 283 | True | 14 | 0-14-0 | 1 | -3.75 |
| l'ambassadrice de Turquie | 1260 ± 326 | 933.2 | 163.1 | 0.06 | 250 | 259 | True | 4 | 0-4-0 | 1 | -2.263 |
| Mme de Morienval | 1256 ± 350 | 906.5 | 174.9 | 0.06 | 251 | 251 | True | 6 | 1-4-1 | 1 | -1.44 |
| duchesse de Luxembourg | 1256 ± 350 | 906.5 | 174.9 | 0.06 | 252 | 249 | True | 6 | 1-4-1 | 1 | -1.44 |
| princesse d'Iéna | 1356 ± 460 | 896.2 | 229.9 | 0.06 | 253 | 200 | True | 3 | 1-2-0 | 1 | -1.437 |
| le prince de Faffenheim | 1196 ± 305 | 890.6 | 152.7 | 0.06 | 254 | 264 | True | 5 | 0-5-0 | 1 | -4.907 |
| prince de Faffenheim | 1252 ± 361 | 890.2 | 180.7 | 0.06 | 255 | 236 | True | 3 | 0-3-0 | 2 | -1.401 |
| le prince von *** | 1303 ± 415 | 888.1 | 207.4 | 0.06 | 256 | 226 | True | 2 | 0-2-0 | 1 | -3.672 |
| le diplomate belge | 1313 ± 429 | 884.6 | 214.4 | 0.06 | 257 | 230 | True | 2 | 0-2-0 | 1 | -2.05 |
| Mme Iéna | 1189 ± 316 | 873.8 | 157.8 | 0.06 | 258 | 263 | True | 5 | 0-5-0 | 1 | -4.716 |
| Picquart | 1150 ± 281 | 869.2 | 140.3 | 0.06 | 259 | 274 | True | 8 | 0-8-0 | 2 | -1.725 |
| Monsieur Vallenères | 1284 ± 419 | 864.5 | 209.5 | 0.06 | 260 | 222 | True | 3 | 0-2-1 | 1 | -1.916 |
| la cousine d'Oriane | 1233 ± 368 | 864.3 | 184.2 | 0.06 | 261 | 248 | True | 3 | 0-3-0 | 1 | -1.615 |
| vicomtesse d'Égremont | 1254 ± 403 | 850.8 | 201.6 | 0.06 | 262 | 243 | True | 3 | 0-3-0 | 1 | -2.33 |
| prince Foggi | 1351 ± 500 | 850.1 | 250.2 | 0.06 | 263 | 186 | True | 1 | 0-1-0 | 1 | -1.574 |
| l'historien de la Fronde | 1244 ± 399 | 845.4 | 199.5 | 0.06 | 264 | 241 | True | 3 | 0-3-0 | 1 | -1.286 |
| prince de Léon | 1290 ± 455 | 835.4 | 227.6 | 0.06 | 265 | 235 | True | 2 | 0-2-0 | 1 | -0.4 |
| M. de Vigny | 1133 ± 299 | 833.6 | 149.7 | 0.06 | 266 | 275 | True | 8 | 0-8-0 | 1 | -2.681 |
| l'empereur | 1197 ± 366 | 831.8 | 182.8 | 0.06 | 267 | 255 | True | 4 | 0-4-0 | 1 | -2.639 |
| le professeur E… | 1286 ± 455 | 831.3 | 227.3 | 0.06 | 268 | 229 | True | 2 | 0-2-0 | 1 | -3.742 |
| Marie Gineste | 1281 ± 452 | 828.1 | 226.2 | 0.06 | 269 | 217 | True | 2 | 0-2-0 | 1 | -0.4 |
| Mme de Varambon | 1178 ± 355 | 823.3 | 177.5 | 0.06 | 270 | 257 | True | 4 | 0-4-0 | 2 | -2.781 |
| Mme de Simiane | 1244 ± 426 | 817.6 | 213.2 | 0.06 | 271 | 244 | True | 3 | 0-3-0 | 1 | -1.378 |
| Mme Blandais | 1178 ± 367 | 810.8 | 183.6 | 0.06 | 272 | 258 | True | 4 | 0-4-0 | 2 | -2.538 |
| La Moussaye | 1500 ± 703 | 797.2 | 351.4 | 0.06 | 273 | 151 | True | 0 | 0-0-0 | 1 | -0.4 |
| Périgot (Joseph) | 1500 ± 704 | 796.3 | 351.9 | 0.06 | 274 | 153 | True | 0 | 0-0-0 | 1 | -2.073 |
| la « marquise » | 1500 ± 704 | 796.0 | 352.0 | 0.06 | 275 | 154 | True | 0 | 0-0-0 | 1 | -2.525 |
| Mme Poncin | 1500 ± 704 | 795.7 | 352.2 | 0.06 | 276 | 152 | True | 0 | 0-0-0 | 1 | +0.131 |
| Mme Blatin | 1233 ± 438 | 794.4 | 219.1 | 0.06 | 277 | 239 | True | 2 | 0-2-0 | 1 | -2.711 |
| la marquise | 1294 ± 503 | 791.4 | 251.3 | 0.06 | 278 | 215 | True | 1 | 0-1-0 | 1 | -1.575 |
| le grand-duc héritier de Luxembourg | 1284 ± 504 | 780.1 | 252.1 | 0.06 | 279 | 209 | True | 1 | 0-1-0 | 1 | -1.198 |
| M. Pierre | 1136 ± 372 | 763.2 | 186.2 | 0.06 | 280 | 261 | True | 4 | 0-4-0 | 2 | -3.099 |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 757.1 | 290.3 | 0.06 | 281 | 193 | True | 1 | 0-1-0 | 1 | -2.218 |
| comtesse G… | 1338 ± 585 | 752.9 | 292.4 | 0.06 | 282 | 191 | True | 1 | 0-1-0 | 1 | -1.751 |
| ma grand’tante | 1105 ± 353 | 751.8 | 176.5 | 0.06 | 283 | 271 | True | 7 | 0-7-0 | 1 | -1.48 |
| la Charité de Giotto | 1338 ± 587 | 750.7 | 293.5 | 0.06 | 284 | 184 | True | 1 | 0-1-0 | 1 | -4.565 |
| ma grand'tante | 1338 ± 587 | 750.7 | 293.5 | 0.06 | 285 | 192 | True | 1 | 0-1-0 | 1 | -0.96 |
| Madame d'Ambresac | 1253 ± 512 | 740.8 | 256.0 | 0.06 | 286 | 233 | True | 2 | 0-2-0 | 1 | 0.0 |
| Dumont | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 287 | 232 | True | 2 | 0-2-0 | 1 | -2.864 |
| le curé | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 288 | 231 | True | 2 | 0-2-0 | 1 | -2.125 |
