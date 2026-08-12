# Character Glicko-2

- Analysis version: `character_glicko2_advantage_v1`
- Lens: `advantage`
- Source review version: `corpus_sanity_review_v1`
- Character count: `288`
- Match count: `5756`
- Draw rate: `0.321`
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

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1858 ± 197 | 1661.1 | 98.4 | 0.06 | 14 | 14-0-0 | 1 | +1.41 |
| docteur du Boulbon | 1758 ± 165 | 1593.4 | 82.5 | 0.06 | 27 | 19-3-5 | 6 | -0.392 |
| Françoise | 1634 ± 82 | 1551.5 | 41.0 | 0.0601 | 217 | 101-48-68 | 82 | -0.26 |
| comte de Forcheville | 1635 ± 100 | 1535.3 | 49.9 | 0.0605 | 112 | 55-18-39 | 25 | -0.312 |
| Bergotte | 1624 ± 93 | 1531.4 | 46.4 | 0.06 | 129 | 52-31-46 | 36 | -0.062 |
| Léa | 1724 ± 195 | 1528.9 | 97.4 | 0.06 | 14 | 8-0-6 | 4 | -0.7 |
| le peintre | 1675 ± 160 | 1515.2 | 80.1 | 0.06 | 42 | 16-4-22 | 8 | -0.202 |
| le grand-père du narrateur | 1647 ± 146 | 1500.4 | 73.2 | 0.06 | 63 | 26-7-30 | 16 | -0.627 |
| M. Verdurin | 1594 ± 95 | 1498.9 | 47.5 | 0.06 | 110 | 38-23-49 | 27 | -0.687 |
| Morel | 1570 ± 79 | 1491.1 | 39.5 | 0.0604 | 152 | 47-53-52 | 32 | -1.066 |

## Bottom Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Saniette | 1199 ± 158 | 1040.9 | 79.1 | 0.0601 | 35 | 1-27-7 | 9 | -3.455 |
| Mme de Franquetot | 1281 ± 170 | 1111.0 | 84.8 | 0.0601 | 23 | 4-13-6 | 3 | -1.088 |
| Mme d'Heudicourt | 1334 ± 199 | 1134.3 | 99.7 | 0.06 | 18 | 3-11-4 | 5 | -1.7 |
| marquis de Cambremer | 1299 ± 136 | 1163.0 | 67.8 | 0.0601 | 45 | 7-24-14 | 6 | -1.173 |
| princesse Sherbatoff | 1354 ± 174 | 1180.2 | 86.8 | 0.06 | 19 | 5-13-1 | 5 | -0.884 |
| Mme d'Arpajon | 1346 ± 149 | 1196.7 | 74.7 | 0.06 | 37 | 6-23-8 | 8 | -1.85 |
| Rosemonde | 1434 ± 190 | 1243.4 | 95.0 | 0.06 | 20 | 5-7-8 | 4 | -0.7 |
| M. de Vaugoubert | 1407 ± 160 | 1247.2 | 80.1 | 0.06 | 35 | 6-12-17 | 9 | -1.463 |
| tante Léonie | 1437 ± 180 | 1256.3 | 90.2 | 0.0601 | 38 | 12-22-4 | 22 | -0.865 |
| le petit Cambremer | 1462 ± 197 | 1265.5 | 98.4 | 0.06 | 14 | 1-3-10 | 1 | -0.8 |

## Provisional Characters

Characters whose RD is still above the provisional threshold -- their rating should be treated as unstable.

| Character | Rating | RD | Matches | Units | Last Period |
| --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1893 ± 226 | 112.9 | 17 | 3 | v5 |
| Mme de Grouchy | 1880 ± 378 | 189.1 | 4 | 1 | v3-p2 |
| Céleste Albaret | 1873 ± 212 | 106.1 | 17 | 3 | v5 |
| prince de Saxe | 1842 ± 374 | 187.0 | 3 | 1 | v3-p1 |
| Mlle de Saint-Loup | 1831 ± 251 | 125.5 | 7 | 2 | v7-p4-le-bal-de-tetes |
| marquis de Beausergent | 1826 ± 203 | 101.4 | 12 | 1 | v7-p4-le-bal-de-tetes |
| Mme de Chaussepierre | 1825 ± 339 | 169.7 | 4 | 1 | v5 |
| Mme Elstir | 1820 ± 272 | 136.2 | 7 | 1 | v2-p2-noms-de-pays-le-pays |
| marquis Maurice de Vaudémont | 1815 ± 470 | 234.8 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Trémoïlle | 1799 ± 423 | 211.6 | 3 | 1 | v1-p2-un-amour-de-swann |
| Marie | 1791 ± 272 | 136.2 | 7 | 1 | v4-p2 |
| colonel Picquart | 1790 ± 349 | 174.5 | 4 | 1 | v3-p1 |
| Maeterlinck | 1785 ± 325 | 162.5 | 5 | 1 | v3-p1 |
| duc de Sidonia | 1776 ± 456 | 227.8 | 2 | 1 | v4-p2 |
| Eulalie | 1772 ± 232 | 116.1 | 16 | 7 | v5 |
| Mlle de Stermaria | 1768 ± 271 | 135.6 | 10 | 5 | v3-p2 |
| Mlle Bloch | 1763 ± 426 | 213.1 | 2 | 1 | v4-p2 |
| Lady Israels | 1760 ± 455 | 227.5 | 2 | 1 | v2-p1-autour-de-mme-swann |
| duc d'Aumale | 1748 ± 378 | 189.2 | 4 | 2 | v3-p2 |
| Duroc | 1747 ± 454 | 227.0 | 2 | 1 | v3-p1 |
| Victurnien | 1738 ± 268 | 133.9 | 8 | 2 | v4-p2 |
| Herbinger | 1735 ± 408 | 204.1 | 3 | 1 | v1-p2-un-amour-de-swann |
| le commandant Duroc | 1724 ± 413 | 206.3 | 2 | 1 | v3-p1 |
| Émilie Daltier | 1722 ± 359 | 179.4 | 3 | 1 | v5 |
| Gribelin | 1722 ± 313 | 156.7 | 6 | 1 | v3-p1 |
| marquis du Lau | 1716 ± 308 | 154.0 | 5 | 2 | v6-p2 |
| Bibi | 1710 ± 425 | 212.3 | 2 | 1 | v3-p2 |
| Rémi | 1707 ± 223 | 111.3 | 17 | 3 | v1-p2-un-amour-de-swann |
| Bismarck | 1705 ± 354 | 177.1 | 4 | 1 | v2-p1-autour-de-mme-swann |
| monsieur Vallenères | 1690 ± 426 | 213.2 | 2 | 1 | v3-p1 |
| le pianiste | 1688 ± 271 | 135.5 | 10 | 3 | v1-p2-un-amour-de-swann |
| Dechambre | 1687 ± 368 | 184.1 | 3 | 1 | v4-p2 |
| elle | 1685 ± 502 | 251.0 | 1 | 1 | v3-p1 |
| Létourville | 1679 ± 353 | 176.6 | 3 | 1 | v7-p4-le-bal-de-tetes |
| grand-duc héritier de Luxembourg | 1666 ± 250 | 125.2 | 9 | 2 | v3-p2 |
| les La Trémoïlle | 1663 ± 329 | 164.3 | 7 | 1 | v1-p2-un-amour-de-swann |
| M. de Courgivaux | 1662 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| Mme de Villebon | 1662 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| baron de Guermantes | 1662 ± 585 | 292.6 | 1 | 1 | v3-p1 |
| docteur Dieulafoy | 1661 ± 502 | 251.0 | 1 | 1 | v3-p2 |
| Poullein | 1653 ± 487 | 243.6 | 2 | 2 | v3-p2 |
| la duchesse d'Alençon | 1651 ± 315 | 157.4 | 6 | 1 | v3-p2 |
| M. Vibert | 1646 ± 375 | 187.4 | 3 | 1 | v3-p2 |
| Mme de Stermaria | 1636 ± 310 | 154.9 | 5 | 1 | v3-p2 |
| Marie-Aynard | 1631 ± 310 | 154.9 | 7 | 1 | v3-p1 |
| Victurnienne | 1631 ± 310 | 154.9 | 7 | 1 | v3-p1 |
| M. d'Orsan | 1626 ± 270 | 134.9 | 11 | 1 | v1-p2-un-amour-de-swann |
| Théodore | 1624 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| M. de Bornier | 1622 ± 347 | 173.5 | 5 | 1 | v3-p2 |
| cousine Poictiers | 1622 ± 321 | 160.3 | 5 | 1 | v3-p1 |
| duc de Poictiers | 1622 ± 321 | 160.3 | 5 | 1 | v3-p1 |
| Lady Rufus Israël | 1620 ± 281 | 140.3 | 6 | 1 | v6-p2 |
| Manet | 1607 ± 318 | 158.8 | 5 | 1 | v3-p2 |
| Flora | 1602 ± 336 | 168.2 | 8 | 1 | v1-p1-combray |
| Arnulphe | 1597 ± 363 | 181.3 | 4 | 1 | v4-p2 |
| Lady Israël | 1596 ± 325 | 162.6 | 5 | 1 | v3-p1 |
| le grand-duc Wladimir | 1591 ± 369 | 184.3 | 3 | 1 | v4-p2 |
| le baron Bréau-Chenut | 1589 ± 310 | 154.9 | 7 | 1 | v2-p1-autour-de-mme-swann |
| le vieux père Chenut | 1589 ± 310 | 154.9 | 7 | 1 | v2-p1-autour-de-mme-swann |
| Mlle de l’Orgeville | 1588 ± 352 | 176.0 | 3 | 1 | v6-p4 |
| Sarah Bernhardt | 1586 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| le jeune prince de Foix | 1586 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| vicomte de Courvoisier | 1586 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| Charcot | 1581 ± 225 | 112.5 | 12 | 1 | v3-p1 |
| M. Reinach | 1581 ± 225 | 112.5 | 12 | 1 | v3-p1 |
| d'Orléans | 1579 ± 350 | 175.0 | 5 | 1 | v2-p2-noms-de-pays-le-pays |
| Mme Leroi | 1577 ± 207 | 103.4 | 13 | 5 | v3-p1 |
| Mlle d'Éporcheville | 1573 ± 224 | 112.0 | 10 | 2 | v6-p2 |
| M. de Beauserfeuil | 1572 ± 278 | 139.1 | 7 | 1 | v3-p2 |
| prince de Sagan | 1572 ± 279 | 139.6 | 7 | 1 | v4-p2 |
| jeune blonde de Rivebelle | 1571 ± 333 | 166.7 | 6 | 2 | v2-p2-noms-de-pays-le-pays |
| duc de Chartres | 1568 ± 208 | 104.0 | 14 | 1 | v4-p2 |
| prince de Chimay | 1568 ± 208 | 104.0 | 14 | 1 | v4-p2 |
| le marquis de Ganançay | 1567 ± 350 | 174.8 | 6 | 1 | v3-p1 |
| le marquis de Palancy | 1567 ± 350 | 174.8 | 6 | 1 | v3-p1 |
| Mme de Sagan | 1565 ± 369 | 184.6 | 3 | 1 | v3-p1 |
| M. de Marsantes | 1563 ± 302 | 151.2 | 7 | 2 | v3-p1 |
| Mme Legrandin mère | 1562 ± 279 | 139.5 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victoire | 1562 ± 279 | 139.5 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Élisabeth | 1562 ± 284 | 142.0 | 6 | 1 | v5 |
| duchesse de Létourville | 1561 ± 291 | 145.7 | 5 | 1 | v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle |
| Mme Timoléon d'Amoncourt | 1559 ± 244 | 121.9 | 9 | 1 | v4-p2 |
| M. de Chateaubriand | 1556 ± 286 | 143.0 | 11 | 2 | v6-p2 |
| comtesse de Monteriender | 1548 ± 352 | 175.9 | 4 | 1 | v1-p2-un-amour-de-swann |
| princesse d'Épinay | 1544 ± 242 | 121.2 | 12 | 3 | v3-p2 |
| M. de Goncourt | 1543 ± 237 | 118.4 | 8 | 1 | v7-p1-a-tansonville |
| le jeune marquis de Cambremer | 1541 ± 201 | 100.7 | 12 | 1 | v6-p4 |
| prince d’Agrigente | 1539 ± 202 | 101.1 | 15 | 2 | v6-p2 |
| Coquelin | 1538 ± 321 | 160.6 | 5 | 1 | v1-p3-noms-de-pays-le-nom |
| Sir Rufus Israël | 1533 ± 282 | 140.9 | 7 | 1 | v3-p1 |
| Mme de Montmorency | 1533 ± 222 | 110.9 | 11 | 1 | v4-p2 |
| Mme de Rochechouart | 1533 ± 222 | 110.9 | 11 | 1 | v4-p2 |
| M. Arthur Meyer | 1532 ± 269 | 134.7 | 6 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| Mme Trombert | 1530 ± 346 | 173.0 | 4 | 1 | v2-p1-autour-de-mme-swann |
| M. Barrère | 1530 ± 501 | 250.5 | 1 | 1 | v6-p3 |
| Thibaud | 1526 ± 245 | 122.3 | 8 | 1 | v5 |
| Napoléon III | 1523 ± 307 | 153.4 | 8 | 1 | v1-p2-un-amour-de-swann |
| Mme Putbus | 1521 ± 251 | 125.4 | 8 | 1 | v5 |
| M. de La Rochefoucauld | 1518 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Rochefoucauld | 1518 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de Praslin | 1518 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| Dostoïevski | 1513 ± 273 | 136.5 | 6 | 1 | v5 |
| M. Carnot | 1508 ± 252 | 126.1 | 9 | 1 | v3-p2 |
| Mme Carnot | 1508 ± 252 | 126.1 | 9 | 1 | v3-p2 |
| comte de Paris | 1508 ± 251 | 125.3 | 10 | 3 | v2-p1-autour-de-mme-swann |
| Liszt | 1507 ± 303 | 151.6 | 6 | 1 | v3-p1 |
| Mme Ristori | 1507 ± 303 | 151.6 | 6 | 1 | v3-p1 |
| l'abbé Poiré | 1504 ± 224 | 112.1 | 10 | 1 | v4-p2 |
| Mme de Vaugoubert | 1504 ± 261 | 130.3 | 9 | 2 | v5 |
| Gisèle | 1503 ± 254 | 126.8 | 14 | 5 | v5 |
| La Moussaye | 1500 ± 703 | 351.4 | 0 | 1 | v5 |
| M. Swann, le père | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| Mme Poncin | 1500 ± 704 | 352.2 | 0 | 1 | v2-p2-noms-de-pays-le-pays |
| Octave | 1500 ± 422 | 210.8 | 4 | 2 | v6-p2 |
| Périgot (Joseph) | 1500 ± 704 | 351.9 | 0 | 1 | v3-p2 |
| docteur Percepied | 1500 ± 426 | 212.9 | 4 | 1 | v1-p1-combray |
| la « marquise » | 1500 ± 704 | 352.0 | 0 | 1 | v3-p1 |
| le comte de Paris | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| le prince de Galles | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| d’Orgeville | 1499 ± 269 | 134.5 | 7 | 1 | v4-p2 |
| Léonor de Cambremer | 1494 ± 203 | 101.4 | 12 | 1 | v7-p4-le-bal-de-tetes |
| Sainte-Beuve | 1489 ± 267 | 133.4 | 7 | 1 | v3-p2 |
| L’excellent écrivain G… | 1489 ± 349 | 174.3 | 4 | 1 | v3-p1 |
| comtesse douairière d'Argencourt | 1488 ± 244 | 122.1 | 10 | 1 | v3-p2 |
| duchesse de Gallardon douairière | 1488 ± 244 | 122.1 | 10 | 1 | v3-p2 |
| marquis de Fierbois | 1488 ± 244 | 122.1 | 10 | 1 | v3-p2 |
| le capitaine | 1488 ± 413 | 206.3 | 2 | 1 | v3-p1 |
| prince d'Agrigente | 1487 ± 451 | 225.7 | 2 | 2 | v7-p4-le-bal-de-tetes |
| prince Von | 1484 ± 259 | 129.5 | 8 | 3 | v3-p2 |
| princesse Mathilde | 1479 ± 319 | 159.4 | 7 | 2 | v3-p2 |
| M. de Miribel | 1478 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Henry | 1478 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Picquart | 1478 ± 372 | 186.2 | 4 | 1 | v3-p1 |
| M. Grevy | 1476 ± 423 | 211.6 | 3 | 1 | v1-p2-un-amour-de-swann |
| Céline | 1476 ± 244 | 122.2 | 16 | 2 | v2-p2-noms-de-pays-le-pays |
| D'Annunzio | 1475 ± 321 | 160.3 | 5 | 1 | v4-p2 |
| prince de Foix | 1473 ± 214 | 107.0 | 14 | 3 | v7-p2-m-de-charlus-pendant-la-guerre |
| oncle Adolphe | 1469 ± 229 | 114.3 | 20 | 6 | v3-p1 |
| le bâtonnier | 1467 ± 405 | 202.4 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| Barrès | 1467 ± 248 | 124.1 | 9 | 1 | v3-p2 |
| Clémenceau | 1467 ± 248 | 124.1 | 9 | 1 | v3-p2 |
| le roi Théodose | 1453 ± 305 | 152.7 | 8 | 3 | v4-p2 |
| M. d'Herweck | 1453 ± 300 | 150.2 | 5 | 2 | v4-p2 |
| Théodose Cadet | 1451 ± 373 | 186.6 | 3 | 1 | v3-p2 |
| la jeune ouvriere | 1451 ± 444 | 221.9 | 2 | 1 | v1-p2-un-amour-de-swann |
| Beauserfeuil | 1450 ± 373 | 186.6 | 3 | 1 | v3-p2 |
| Vigny | 1448 ± 484 | 242.0 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| M. de Stermaria | 1443 ± 256 | 127.8 | 10 | 4 | v2-p2-noms-de-pays-le-pays |
| M. Molé | 1436 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| M. de Bouillon | 1436 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Musset | 1436 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victor Hugo | 1436 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Antoine | 1430 ± 411 | 205.7 | 3 | 1 | v3-p1 |
| le prince Von | 1415 ± 243 | 121.6 | 10 | 2 | v3-p2 |
| Madame Elstir | 1403 ± 333 | 166.7 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| les demoiselles d’Ambresac | 1403 ± 333 | 166.7 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| Cartier | 1398 ± 339 | 169.7 | 4 | 1 | v5 |
| professeur E… | 1366 ± 370 | 185.2 | 4 | 2 | v4-p2 |
| duc de Châtellerault | 1353 ± 254 | 127.0 | 10 | 5 | v4-p2 |
| M. Bontemps | 1352 ± 291 | 145.6 | 9 | 2 | v7-p2-m-de-charlus-pendant-la-guerre |
| prince Foggi | 1352 ± 501 | 250.5 | 1 | 1 | v6-p3 |
| marquise de Gallardon | 1350 ± 234 | 117.0 | 19 | 7 | v4-p2 |
| Prince Henri d'Orléans | 1347 ± 427 | 213.7 | 2 | 1 | v3-p1 |
| comtesse G… | 1338 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| la Charité de Giotto | 1338 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| ma grand'tante | 1338 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| M. de Luxembourg | 1324 ± 436 | 217.8 | 2 | 1 | v3-p2 |
| M. de Grouchy | 1324 ± 254 | 127.2 | 10 | 4 | v3-p2 |
| princesse de Nassau | 1310 ± 496 | 248.0 | 1 | 1 | v7-p4-le-bal-de-tetes |
| le diplomate belge | 1307 ± 426 | 212.9 | 2 | 1 | v3-p1 |
| marquise de Citri | 1305 ± 414 | 207.0 | 2 | 1 | v4-p2 |
| le prince von *** | 1304 ± 415 | 207.6 | 2 | 1 | v3-p1 |
| les Courvoisier | 1304 ± 349 | 174.3 | 5 | 1 | v3-p2 |
| la marquise | 1296 ± 502 | 251.1 | 1 | 1 | v3-p1 |
| prince de Léon | 1292 ± 456 | 227.8 | 2 | 1 | v5 |
| Monsieur Vallenères | 1290 ± 418 | 209.2 | 3 | 1 | v3-p1 |
| le professeur E… | 1289 ± 456 | 227.8 | 2 | 1 | v4-p2 |
| Mme de Souvré | 1288 ± 266 | 133.1 | 11 | 2 | v4-p2 |
| Marie Gineste | 1281 ± 453 | 226.3 | 2 | 1 | v4-p2 |
| le grand-duc héritier de Luxembourg | 1280 ± 505 | 252.7 | 1 | 1 | v3-p2 |
| l'ambassadrice de Turquie | 1263 ± 327 | 163.3 | 4 | 1 | v4-p2 |
| Maurice | 1261 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| prince de Faffenheim | 1258 ± 361 | 180.7 | 3 | 2 | v3-p1 |
| princesse d'Iéna | 1258 ± 460 | 229.9 | 3 | 1 | v1-p2-un-amour-de-swann |
| Mme de Morienval | 1257 ± 350 | 174.8 | 6 | 1 | v3-p1 |
| duchesse de Luxembourg | 1257 ± 350 | 174.8 | 6 | 1 | v3-p1 |
| capitaine de Borodino | 1256 ± 207 | 103.6 | 14 | 5 | v3-p1 |
| vicomtesse d'Égremont | 1255 ± 404 | 202.1 | 3 | 1 | v3-p2 |
| Dumont | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| Madame d'Ambresac | 1253 ± 512 | 256.0 | 2 | 1 | v3-p1 |
| le curé | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| l'historien de la Fronde | 1248 ± 396 | 198.1 | 3 | 1 | v3-p1 |
| Mme de Simiane | 1246 ± 428 | 213.9 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| la cousine d'Oriane | 1233 ± 368 | 184.2 | 3 | 1 | v3-p2 |
| Mme Blatin | 1228 ± 440 | 219.8 | 2 | 1 | v1-p3-noms-de-pays-le-nom |
| le prince de Faffenheim | 1199 ± 305 | 152.6 | 5 | 1 | v3-p1 |
| l'empereur | 1198 ± 366 | 183.2 | 4 | 1 | v3-p2 |
| Mme Iéna | 1188 ± 316 | 158.0 | 5 | 1 | v3-p2 |
| Alix | 1184 ± 252 | 126.0 | 9 | 3 | v3-p1 |
| Mme Blandais | 1179 ± 367 | 183.7 | 4 | 2 | v2-p2-noms-de-pays-le-pays |
| Mme de Varambon | 1178 ± 356 | 177.9 | 4 | 2 | v3-p2 |
| colonel de Froberville | 1159 ± 208 | 104.0 | 14 | 1 | v4-p2 |
| Picquart | 1150 ± 280 | 139.9 | 8 | 2 | v3-p1 |
| M. Pierre | 1142 ± 372 | 185.9 | 4 | 2 | v3-p1 |
| M. de Vigny | 1133 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| ma grand’tante | 1105 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |

## Largest Glicko-vs-ELO Rank Divergences

| Character | Glicko Rank | ELO Rank | Delta | Rating | ELO |
| --- | --- | --- | --- | --- | --- |
| duchesse de Guermantes | 38 | 281 | -243 | 1521 ± 59 | 1388.589 |
| Albertine | 46 | 284 | -238 | 1499 ± 68 | 1365.03 |
| baron de Charlus | 44 | 282 | -238 | 1496 ± 60 | 1387.013 |
| Gilberte | 49 | 286 | -237 | 1489 ± 61 | 1341.323 |
| duc de Guermantes | 65 | 287 | -222 | 1461 ± 67 | 1330.266 |
| princesse de Guermantes | 47 | 269 | -222 | 1522 ± 91 | 1431.084 |
| Brichot | 62 | 265 | -203 | 1486 ± 80 | 1439.337 |
| Swann | 57 | 259 | -202 | 1473 ± 61 | 1452.732 |
| Robert de Saint-Loup | 43 | 241 | -198 | 1497 ± 60 | 1471.52 |
| la Berma | 91 | 275 | -184 | 1454 ± 111 | 1410.334 |

## Character Table

| Character | Rating | Conservative | RD | Volatility | Glicko Rank | ELO Rank | Provisional | Matches | W-L-D | Units | Mean Advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1893 ± 226 | 1667.1 | 112.9 | 0.06 | 1 | 4 | True | 17 | 16-1-0 | 3 | +0.363 |
| Mlle d'Oloron | 1858 ± 197 | 1661.1 | 98.4 | 0.06 | 2 | 3 | False | 14 | 14-0-0 | 1 | +1.41 |
| Céleste Albaret | 1873 ± 212 | 1660.8 | 106.1 | 0.06 | 3 | 2 | True | 17 | 16-1-0 | 3 | +1.858 |
| marquis de Beausergent | 1826 ± 203 | 1623.4 | 101.4 | 0.06 | 4 | 6 | True | 12 | 12-0-0 | 1 | -0.08 |
| docteur du Boulbon | 1758 ± 165 | 1593.4 | 82.5 | 0.06 | 5 | 5 | False | 27 | 19-3-5 | 6 | -0.392 |
| Mlle de Saint-Loup | 1831 ± 251 | 1579.8 | 125.5 | 0.06 | 6 | 14 | True | 7 | 7-0-0 | 2 | +1.925 |
| Françoise | 1634 ± 82 | 1551.5 | 41.0 | 0.0601 | 7 | 12 | False | 217 | 101-48-68 | 82 | -0.26 |
| Mme Elstir | 1820 ± 272 | 1547.8 | 136.2 | 0.06 | 8 | 15 | True | 7 | 7-0-0 | 1 | +0.78 |
| Eulalie | 1772 ± 232 | 1539.6 | 116.1 | 0.06 | 9 | 8 | True | 16 | 11-2-3 | 7 | -0.021 |
| comte de Forcheville | 1635 ± 100 | 1535.3 | 49.9 | 0.0605 | 10 | 67 | False | 112 | 55-18-39 | 25 | -0.312 |
| Bergotte | 1624 ± 93 | 1531.4 | 46.4 | 0.06 | 11 | 1 | False | 129 | 52-31-46 | 36 | -0.062 |
| Léa | 1724 ± 195 | 1528.9 | 97.4 | 0.06 | 12 | 16 | False | 14 | 8-0-6 | 4 | -0.7 |
| Marie | 1791 ± 272 | 1518.9 | 136.2 | 0.06 | 13 | 19 | True | 7 | 6-1-0 | 1 | -0.1 |
| le peintre | 1675 ± 160 | 1515.2 | 80.1 | 0.06 | 14 | 28 | False | 42 | 16-4-22 | 8 | -0.202 |
| Mme de Grouchy | 1880 ± 378 | 1502.3 | 189.1 | 0.06 | 15 | 30 | True | 4 | 4-0-0 | 1 | +0.144 |
| le grand-père du narrateur | 1647 ± 146 | 1500.4 | 73.2 | 0.06 | 16 | 13 | False | 63 | 26-7-30 | 16 | -0.627 |
| M. Verdurin | 1594 ± 95 | 1498.9 | 47.5 | 0.06 | 17 | 18 | False | 110 | 38-23-49 | 27 | -0.687 |
| Mlle de Stermaria | 1768 ± 271 | 1496.5 | 135.6 | 0.0601 | 18 | 27 | True | 10 | 6-3-1 | 5 | -0.702 |
| Morel | 1570 ± 79 | 1491.1 | 39.5 | 0.0604 | 19 | 11 | False | 152 | 47-53-52 | 32 | -1.066 |
| Aimé | 1586 ± 100 | 1485.3 | 50.1 | 0.06 | 20 | 34 | False | 79 | 27-14-38 | 18 | -0.418 |
| Mme de Chaussepierre | 1825 ± 339 | 1485.3 | 169.7 | 0.06 | 21 | 32 | True | 4 | 4-0-0 | 1 | +1.81 |
| Rémi | 1707 ± 223 | 1484.4 | 111.3 | 0.06 | 22 | 29 | True | 17 | 5-0-12 | 3 | -0.533 |
| Elstir | 1574 ± 91 | 1483.6 | 45.3 | 0.0601 | 23 | 10 | False | 106 | 40-29-37 | 29 | +0.174 |
| la grand-mère | 1567 ± 84 | 1482.8 | 42.1 | 0.0605 | 24 | 40 | False | 225 | 93-66-66 | 80 | -0.325 |
| Rachel | 1558 ± 77 | 1481.0 | 38.6 | 0.0608 | 25 | 9 | False | 146 | 52-53-41 | 43 | -1.086 |
| l'amie de Mlle Vinteuil | 1615 ± 134 | 1480.8 | 66.9 | 0.06 | 26 | 26 | False | 44 | 17-7-20 | 12 | -0.325 |
| Mme Verdurin | 1546 ± 68 | 1478.9 | 33.8 | 0.0598 | 27 | 38 | False | 311 | 93-96-122 | 82 | -0.893 |
| prince de Guermantes | 1570 ± 91 | 1478.6 | 45.7 | 0.0606 | 28 | 7 | False | 124 | 42-30-52 | 22 | -0.843 |
| Norpois | 1567 ± 89 | 1477.3 | 44.6 | 0.0599 | 29 | 21 | False | 180 | 80-54-46 | 63 | -0.65 |
| Odette | 1543 ± 66 | 1476.8 | 33.0 | 0.0608 | 30 | 44 | False | 462 | 167-157-138 | 142 | -0.718 |
| la mère du narrateur | 1565 ± 89 | 1475.9 | 44.7 | 0.0599 | 31 | 169 | False | 144 | 55-36-53 | 40 | -0.419 |
| Mme Cottard | 1637 ± 166 | 1470.7 | 83.0 | 0.06 | 32 | 24 | False | 33 | 16-7-10 | 11 | -0.431 |
| Victurnien | 1738 ± 268 | 1469.9 | 133.9 | 0.06 | 33 | 20 | True | 8 | 5-0-3 | 2 | +0.762 |
| prince de Saxe | 1842 ± 374 | 1468.5 | 187.0 | 0.06 | 34 | 39 | True | 3 | 3-0-0 | 1 | +0.865 |
| Jupien | 1569 ± 101 | 1468.2 | 50.6 | 0.06 | 35 | 53 | False | 68 | 23-14-31 | 18 | +0.118 |
| Mlle Vinteuil | 1577 ± 109 | 1467.6 | 54.6 | 0.06 | 36 | 65 | False | 71 | 21-15-35 | 15 | -0.714 |
| le père du narrateur | 1584 ± 120 | 1464.4 | 60.0 | 0.0599 | 37 | 17 | False | 90 | 35-22-33 | 24 | -0.753 |
| duchesse de Guermantes | 1521 ± 59 | 1461.6 | 29.6 | 0.0673 | 38 | 281 | False | 662 | 334-177-151 | 199 | -0.075 |
| Maeterlinck | 1785 ± 325 | 1459.7 | 162.5 | 0.06 | 39 | 31 | True | 5 | 4-0-1 | 1 | -0.8 |
| marquis de Bréauté | 1548 ± 93 | 1454.5 | 46.7 | 0.0599 | 40 | 57 | False | 101 | 26-22-53 | 19 | -0.934 |
| Bloch | 1519 ± 68 | 1450.9 | 33.9 | 0.0609 | 41 | 22 | False | 270 | 79-111-80 | 71 | -1.701 |
| colonel Picquart | 1790 ± 349 | 1441.3 | 174.5 | 0.06 | 42 | 43 | True | 4 | 4-0-0 | 1 | +2.15 |
| Robert de Saint-Loup | 1497 ± 60 | 1437.4 | 30.0 | 0.0611 | 43 | 241 | False | 508 | 166-213-129 | 168 | -0.602 |
| baron de Charlus | 1496 ± 60 | 1436.6 | 29.9 | 0.0617 | 44 | 282 | False | 485 | 189-160-136 | 119 | -0.809 |
| Mme de Charlus | 1627 ± 193 | 1434.5 | 96.3 | 0.06 | 45 | 36 | False | 15 | 5-1-9 | 2 | -0.8 |
| Albertine | 1499 ± 68 | 1431.2 | 34.1 | 0.0604 | 46 | 284 | False | 387 | 149-156-82 | 146 | -0.868 |
| princesse de Guermantes | 1522 ± 91 | 1431.0 | 45.3 | 0.0602 | 47 | 269 | False | 113 | 41-31-41 | 25 | -0.268 |
| Mme Goupil | 1609 ± 180 | 1429.6 | 89.9 | 0.06 | 48 | 37 | False | 17 | 5-1-11 | 2 | -0.8 |
| Gilberte | 1489 ± 61 | 1427.7 | 30.6 | 0.0604 | 49 | 286 | False | 312 | 114-103-95 | 76 | -0.516 |
| Mme de Surgis | 1556 ± 130 | 1426.2 | 65.1 | 0.06 | 50 | 42 | False | 42 | 16-11-15 | 9 | -0.967 |
| le narrateur | 1481 ± 62 | 1419.1 | 31.0 | 0.0809 | 51 | 186 | False | 1093 | 400-508-185 | 316 | -0.85 |
| le pianiste | 1688 ± 271 | 1417.1 | 135.5 | 0.06 | 52 | 47 | True | 10 | 4-1-5 | 3 | +0.759 |
| Dreyfus | 1540 ± 123 | 1416.5 | 61.5 | 0.06 | 53 | 71 | False | 58 | 13-11-34 | 7 | -0.794 |
| grand-duc héritier de Luxembourg | 1666 ± 250 | 1415.8 | 125.2 | 0.06 | 54 | 33 | True | 9 | 4-1-4 | 2 | +0.73 |
| docteur Cottard | 1495 ± 81 | 1413.3 | 40.7 | 0.06 | 55 | 124 | False | 194 | 46-63-85 | 43 | -0.978 |
| Mme de Villeparisis | 1498 ± 86 | 1412.6 | 42.9 | 0.0605 | 56 | 216 | False | 236 | 90-93-53 | 79 | -0.726 |
| Swann | 1473 ± 61 | 1411.9 | 30.4 | 0.0642 | 57 | 259 | False | 667 | 205-303-159 | 202 | -1.004 |
| Mme Sazerat | 1602 ± 192 | 1409.2 | 96.2 | 0.06 | 58 | 25 | False | 20 | 8-2-10 | 6 | -0.734 |
| Gribelin | 1722 ± 313 | 1408.4 | 156.7 | 0.06 | 59 | 35 | True | 6 | 5-1-0 | 1 | +0.15 |
| marquis du Lau | 1716 ± 308 | 1407.7 | 154.0 | 0.06 | 60 | 46 | True | 5 | 4-1-0 | 2 | +1.647 |
| Andrée | 1498 ± 91 | 1406.9 | 45.7 | 0.0599 | 61 | 224 | False | 114 | 36-42-36 | 31 | -0.795 |
| Brichot | 1486 ± 80 | 1406.2 | 40.0 | 0.06 | 62 | 265 | False | 135 | 28-33-74 | 21 | -0.909 |
| M. Vinteuil | 1523 ± 118 | 1405.1 | 59.0 | 0.06 | 63 | 210 | False | 61 | 18-19-24 | 15 | -0.388 |
| M. Ski | 1563 ± 167 | 1395.5 | 83.7 | 0.06 | 64 | 91 | False | 21 | 4-1-16 | 2 | -0.4 |
| duc de Guermantes | 1461 ± 67 | 1394.7 | 33.3 | 0.0601 | 65 | 287 | False | 401 | 123-171-107 | 110 | -1.136 |
| M. d'Argencourt | 1516 ± 123 | 1392.8 | 61.7 | 0.06 | 66 | 209 | False | 56 | 19-18-19 | 14 | -1.286 |
| marquise de Saint-Euverte | 1495 ± 106 | 1389.0 | 53.1 | 0.0602 | 67 | 101 | False | 72 | 15-28-29 | 13 | -2.104 |
| Mme Bontemps | 1502 ± 116 | 1386.4 | 57.8 | 0.06 | 68 | 238 | False | 54 | 13-12-29 | 13 | -0.651 |
| général de Froberville | 1556 ± 171 | 1384.3 | 85.6 | 0.06 | 69 | 41 | False | 27 | 7-4-16 | 7 | -0.622 |
| duchesse de La Trémoïlle | 1799 ± 423 | 1375.6 | 211.6 | 0.06 | 70 | 45 | True | 3 | 3-0-0 | 1 | +0.91 |
| Mme de Marsantes | 1468 ± 94 | 1373.8 | 47.1 | 0.0599 | 71 | 236 | False | 107 | 20-34-53 | 21 | -1.22 |
| comtesse Molé | 1520 ± 148 | 1372.5 | 73.9 | 0.06 | 72 | 149 | False | 34 | 6-9-19 | 6 | -1.365 |
| M. Nissim Bernard | 1500 ± 130 | 1370.8 | 64.8 | 0.06 | 73 | 213 | False | 39 | 9-10-20 | 10 | -1.591 |
| Mme Leroi | 1577 ± 207 | 1370.6 | 103.4 | 0.06 | 74 | 51 | True | 13 | 8-5-0 | 5 | -1.147 |
| duc d'Aumale | 1748 ± 378 | 1369.2 | 189.2 | 0.06 | 75 | 75 | True | 4 | 3-1-0 | 2 | +0.505 |
| princesse de Parme | 1469 ± 102 | 1367.0 | 50.8 | 0.06 | 76 | 23 | False | 130 | 36-63-31 | 38 | -0.822 |
| la marquise douairière de Cambremer | 1516 ± 151 | 1364.9 | 75.7 | 0.06 | 77 | 62 | False | 31 | 9-5-17 | 6 | +0.132 |
| Émilie Daltier | 1722 ± 359 | 1363.7 | 179.4 | 0.06 | 78 | 73 | True | 3 | 2-0-1 | 1 | -0.4 |
| duc de Chartres | 1568 ± 208 | 1360.3 | 104.0 | 0.06 | 79 | 48 | True | 14 | 2-0-12 | 1 | -0.8 |
| prince de Chimay | 1568 ± 208 | 1360.3 | 104.0 | 0.06 | 80 | 55 | True | 14 | 2-0-12 | 1 | -0.8 |
| Bloch père | 1488 ± 129 | 1359.1 | 64.3 | 0.06 | 81 | 177 | False | 47 | 11-11-25 | 8 | -1.942 |
| prince des Laumes | 1536 ± 176 | 1359.0 | 88.2 | 0.06 | 82 | 119 | False | 27 | 4-3-20 | 3 | -0.8 |
| Legrandin | 1458 ± 101 | 1357.4 | 50.4 | 0.06 | 83 | 246 | False | 83 | 15-28-40 | 24 | -1.39 |
| M. d'Orsan | 1626 ± 270 | 1356.3 | 134.9 | 0.06 | 84 | 59 | True | 11 | 2-0-9 | 1 | -0.8 |
| Charcot | 1581 ± 225 | 1356.0 | 112.5 | 0.06 | 85 | 89 | True | 12 | 3-2-7 | 1 | -0.8 |
| M. Reinach | 1581 ± 225 | 1356.0 | 112.5 | 0.06 | 86 | 92 | True | 12 | 3-2-7 | 1 | -0.8 |
| Bismarck | 1705 ± 354 | 1350.4 | 177.1 | 0.06 | 87 | 78 | True | 4 | 3-1-0 | 1 | +0.548 |
| Mlle d'Éporcheville | 1573 ± 224 | 1348.6 | 112.0 | 0.06 | 88 | 110 | True | 10 | 3-2-5 | 2 | -0.6 |
| le directeur | 1484 ± 139 | 1346.0 | 69.3 | 0.06 | 89 | 264 | False | 39 | 11-16-12 | 11 | -0.828 |
| marquis Maurice de Vaudémont | 1815 ± 470 | 1345.6 | 234.8 | 0.06 | 90 | 49 | True | 2 | 2-0-0 | 1 | +1.414 |
| la Berma | 1454 ± 111 | 1342.9 | 55.7 | 0.0601 | 91 | 275 | False | 62 | 19-24-19 | 19 | -0.309 |
| Esther | 1542 ± 199 | 1342.8 | 99.4 | 0.06 | 92 | 107 | False | 14 | 3-2-9 | 2 | -1.0 |
| M. de Chevregny | 1539 ± 197 | 1341.5 | 98.6 | 0.06 | 93 | 66 | False | 16 | 4-1-11 | 1 | -0.4 |
| M. de Crécy | 1539 ± 197 | 1341.5 | 98.6 | 0.06 | 94 | 70 | False | 16 | 4-1-11 | 1 | -0.4 |
| Mme Féré | 1539 ± 197 | 1341.5 | 98.6 | 0.06 | 95 | 74 | False | 16 | 4-1-11 | 1 | -0.4 |
| le jeune marquis de Cambremer | 1541 ± 201 | 1339.9 | 100.7 | 0.06 | 96 | 140 | True | 12 | 2-1-9 | 1 | -1.2 |
| Lady Rufus Israël | 1620 ± 281 | 1339.2 | 140.3 | 0.06 | 97 | 112 | True | 6 | 2-1-3 | 1 | -0.4 |
| Mlle Bloch | 1763 ± 426 | 1336.5 | 213.1 | 0.06 | 98 | 61 | True | 2 | 2-0-0 | 1 | +1.28 |
| prince d’Agrigente | 1539 ± 202 | 1336.4 | 101.1 | 0.06 | 99 | 123 | True | 15 | 3-2-10 | 2 | -0.8 |
| la duchesse d'Alençon | 1651 ± 315 | 1336.0 | 157.4 | 0.06 | 100 | 50 | True | 6 | 3-1-2 | 1 | -0.8 |
| les La Trémoïlle | 1663 ± 329 | 1334.7 | 164.3 | 0.06 | 101 | 60 | True | 7 | 2-0-5 | 1 | -0.8 |
| princesse de Luxembourg | 1493 ± 166 | 1327.2 | 83.1 | 0.06 | 102 | 160 | False | 25 | 6-7-12 | 6 | -0.816 |
| Herbinger | 1735 ± 408 | 1326.7 | 204.1 | 0.06 | 103 | 64 | True | 3 | 2-0-1 | 1 | -0.8 |
| Mme de Stermaria | 1636 ± 310 | 1326.6 | 154.9 | 0.06 | 104 | 88 | True | 5 | 2-1-2 | 1 | -0.8 |
| Létourville | 1679 ± 353 | 1325.4 | 176.6 | 0.06 | 105 | 84 | True | 3 | 2-0-1 | 1 | -0.8 |
| Marie-Aynard | 1631 ± 310 | 1321.6 | 154.9 | 0.06 | 106 | 54 | True | 7 | 2-0-5 | 1 | -0.8 |
| Victurnienne | 1631 ± 310 | 1321.6 | 154.9 | 0.06 | 107 | 52 | True | 7 | 2-0-5 | 1 | -0.8 |
| duc de Sidonia | 1776 ± 456 | 1320.6 | 227.8 | 0.06 | 108 | 58 | True | 2 | 2-0-0 | 1 | -1.0 |
| Dechambre | 1687 ± 368 | 1319.1 | 184.1 | 0.06 | 109 | 79 | True | 3 | 2-0-1 | 1 | -1.1 |
| Mme Timoléon d'Amoncourt | 1559 ± 244 | 1315.3 | 121.9 | 0.06 | 110 | 102 | True | 9 | 2-1-6 | 1 | -0.4 |
| Goncourt | 1487 ± 175 | 1312.9 | 87.3 | 0.06 | 111 | 232 | False | 16 | 2-3-11 | 2 | -0.8 |
| le commandant Duroc | 1724 ± 413 | 1311.7 | 206.3 | 0.06 | 112 | 82 | True | 2 | 2-0-0 | 1 | +0.628 |
| Mme de Montmorency | 1533 ± 222 | 1310.9 | 110.9 | 0.06 | 113 | 147 | True | 11 | 2-1-8 | 1 | -0.8 |
| Mme de Rochechouart | 1533 ± 222 | 1310.9 | 110.9 | 0.06 | 114 | 148 | True | 11 | 2-1-8 | 1 | -0.8 |
| M. de Goncourt | 1543 ± 237 | 1306.4 | 118.4 | 0.06 | 115 | 138 | True | 8 | 1-0-7 | 1 | -1.2 |
| Lady Israels | 1760 ± 455 | 1304.6 | 227.5 | 0.06 | 116 | 83 | True | 2 | 2-0-0 | 1 | 0.0 |
| princesse d'Épinay | 1544 ± 242 | 1301.8 | 121.2 | 0.06 | 117 | 99 | True | 12 | 4-3-5 | 3 | -0.533 |
| cousine Poictiers | 1622 ± 321 | 1300.9 | 160.3 | 0.06 | 118 | 95 | True | 5 | 2-1-2 | 1 | -0.4 |
| duc de Poictiers | 1622 ± 321 | 1300.9 | 160.3 | 0.06 | 119 | 98 | True | 5 | 2-1-2 | 1 | -0.4 |
| Sarah Bernhardt | 1586 ± 286 | 1299.9 | 143.2 | 0.06 | 120 | 104 | True | 7 | 2-0-5 | 1 | -0.8 |
| le jeune prince de Foix | 1586 ± 286 | 1299.9 | 143.2 | 0.06 | 121 | 97 | True | 7 | 2-0-5 | 1 | -0.8 |
| vicomte de Courvoisier | 1586 ± 286 | 1299.9 | 143.2 | 0.06 | 122 | 106 | True | 7 | 2-0-5 | 1 | -0.8 |
| Mme de Sévigné | 1484 ± 189 | 1295.7 | 94.3 | 0.06 | 123 | 94 | False | 25 | 7-5-13 | 4 | +0.097 |
| M. de Beauserfeuil | 1572 ± 278 | 1294.0 | 139.1 | 0.06 | 124 | 69 | True | 7 | 2-1-4 | 1 | -0.8 |
| Duroc | 1747 ± 454 | 1293.4 | 227.0 | 0.06 | 125 | 76 | True | 2 | 2-0-0 | 1 | +1.708 |
| prince de Sagan | 1572 ± 279 | 1292.6 | 139.6 | 0.06 | 126 | 86 | True | 7 | 1-0-6 | 1 | -0.8 |
| Léonor de Cambremer | 1494 ± 203 | 1291.7 | 101.4 | 0.06 | 127 | 183 | True | 12 | 1-1-10 | 1 | -0.8 |
| Manet | 1607 ± 318 | 1289.6 | 158.8 | 0.06 | 128 | 77 | True | 5 | 1-0-4 | 1 | -0.8 |
| Bibi | 1710 ± 425 | 1285.8 | 212.3 | 0.06 | 129 | 56 | True | 2 | 2-0-0 | 1 | +0.3 |
| Mme Legrandin mère | 1562 ± 279 | 1283.4 | 139.5 | 0.06 | 130 | 80 | True | 8 | 2-0-6 | 1 | -0.8 |
| Victoire | 1562 ± 279 | 1283.4 | 139.5 | 0.06 | 131 | 85 | True | 8 | 2-0-6 | 1 | -0.8 |
| Thibaud | 1526 ± 245 | 1281.6 | 122.3 | 0.06 | 132 | 164 | True | 8 | 2-2-4 | 1 | -0.8 |
| l'abbé Poiré | 1504 ± 224 | 1280.2 | 112.1 | 0.06 | 133 | 134 | True | 10 | 1-2-7 | 1 | -0.8 |
| Balzac | 1464 ± 185 | 1279.3 | 92.3 | 0.06 | 134 | 233 | False | 18 | 2-4-12 | 2 | -0.8 |
| le baron Bréau-Chenut | 1589 ± 310 | 1279.3 | 154.9 | 0.06 | 135 | 93 | True | 7 | 3-1-3 | 1 | -0.8 |
| le vieux père Chenut | 1589 ± 310 | 1279.3 | 154.9 | 0.06 | 136 | 100 | True | 7 | 3-1-3 | 1 | -0.8 |
| Mme de Cambremer | 1372 ± 94 | 1278.5 | 46.9 | 0.0601 | 137 | 280 | False | 112 | 12-53-47 | 20 | -1.709 |
| Élisabeth | 1562 ± 284 | 1277.5 | 142.0 | 0.06 | 138 | 120 | True | 6 | 2-1-3 | 1 | -1.2 |
| M. de Bornier | 1622 ± 347 | 1275.2 | 173.5 | 0.06 | 139 | 68 | True | 5 | 3-1-1 | 1 | -1.2 |
| général de Monserfeuil | 1468 ± 196 | 1272.5 | 98.0 | 0.06 | 140 | 214 | False | 18 | 5-7-6 | 4 | -1.511 |
| Lady Israël | 1596 ± 325 | 1270.9 | 162.6 | 0.06 | 141 | 103 | True | 5 | 2-1-2 | 1 | -0.4 |
| M. Vibert | 1646 ± 375 | 1270.9 | 187.4 | 0.06 | 142 | 81 | True | 3 | 1-0-2 | 1 | -0.4 |
| Mme Putbus | 1521 ± 251 | 1270.4 | 125.4 | 0.06 | 143 | 130 | True | 8 | 1-1-6 | 1 | -0.8 |
| M. de Chateaubriand | 1556 ± 286 | 1270.3 | 143.0 | 0.06 | 144 | 208 | True | 11 | 1-3-7 | 2 | -2.132 |
| duchesse de Létourville | 1561 ± 291 | 1269.7 | 145.7 | 0.06 | 145 | 133 | True | 5 | 2-1-2 | 1 | -0.8 |
| Flora | 1602 ± 336 | 1265.5 | 168.2 | 0.06 | 146 | 63 | True | 8 | 3-1-4 | 1 | -0.8 |
| le petit Cambremer | 1462 ± 197 | 1265.5 | 98.4 | 0.06 | 147 | 219 | False | 14 | 1-3-10 | 1 | -0.8 |
| princesse de Silistrie | 1462 ± 197 | 1265.5 | 98.4 | 0.06 | 148 | 217 | False | 14 | 1-3-10 | 1 | -0.8 |
| monsieur Vallenères | 1690 ± 426 | 1264.0 | 213.2 | 0.06 | 149 | 72 | True | 2 | 2-0-0 | 1 | -0.8 |
| M. Arthur Meyer | 1532 ± 269 | 1262.3 | 134.7 | 0.06 | 150 | 167 | True | 6 | 2-2-2 | 1 | -0.8 |
| M. de Marsantes | 1563 ± 302 | 1260.8 | 151.2 | 0.06 | 151 | 121 | True | 7 | 2-1-4 | 2 | -0.24 |
| prince de Foix | 1473 ± 214 | 1259.3 | 107.0 | 0.06 | 152 | 159 | True | 14 | 4-4-6 | 3 | -0.95 |
| comte de Paris | 1508 ± 251 | 1257.0 | 125.3 | 0.06 | 153 | 203 | True | 10 | 3-4-3 | 3 | -0.667 |
| tante Léonie | 1437 ± 180 | 1256.3 | 90.2 | 0.0601 | 154 | 253 | False | 38 | 12-22-4 | 22 | -0.865 |
| M. Carnot | 1508 ± 252 | 1255.7 | 126.1 | 0.06 | 155 | 135 | True | 9 | 1-1-7 | 1 | -0.8 |
| Mme Carnot | 1508 ± 252 | 1255.7 | 126.1 | 0.06 | 156 | 136 | True | 9 | 1-1-7 | 1 | -0.8 |
| Sir Rufus Israël | 1533 ± 282 | 1251.4 | 140.9 | 0.06 | 157 | 118 | True | 7 | 3-1-3 | 1 | -0.8 |
| Gisèle | 1503 ± 254 | 1249.6 | 126.8 | 0.06 | 158 | 192 | True | 14 | 3-6-5 | 5 | -2.229 |
| M. de Vaugoubert | 1407 ± 160 | 1247.2 | 80.1 | 0.06 | 159 | 252 | False | 35 | 6-12-17 | 9 | -1.463 |
| comtesse douairière d'Argencourt | 1488 ± 244 | 1243.6 | 122.1 | 0.06 | 160 | 196 | True | 10 | 1-2-7 | 1 | -0.8 |
| duchesse de Gallardon douairière | 1488 ± 244 | 1243.6 | 122.1 | 0.06 | 161 | 193 | True | 10 | 1-2-7 | 1 | -0.8 |
| marquis de Fierbois | 1488 ± 244 | 1243.6 | 122.1 | 0.06 | 162 | 206 | True | 10 | 1-2-7 | 1 | -0.8 |
| Rosemonde | 1434 ± 190 | 1243.4 | 95.0 | 0.06 | 163 | 194 | False | 20 | 5-7-8 | 4 | -0.7 |
| Mme de Vaugoubert | 1504 ± 261 | 1243.0 | 130.3 | 0.06 | 164 | 228 | True | 9 | 1-3-5 | 2 | -2.0 |
| oncle Adolphe | 1469 ± 229 | 1240.3 | 114.3 | 0.0601 | 165 | 251 | True | 20 | 5-11-4 | 6 | -1.8 |
| Dostoïevski | 1513 ± 273 | 1239.8 | 136.5 | 0.06 | 166 | 172 | True | 6 | 1-1-4 | 1 | -0.8 |
| jeune blonde de Rivebelle | 1571 ± 333 | 1237.6 | 166.7 | 0.06 | 167 | 105 | True | 6 | 2-1-3 | 2 | -0.4 |
| Mlle de l’Orgeville | 1588 ± 352 | 1235.9 | 176.0 | 0.06 | 168 | 132 | True | 3 | 1-0-2 | 1 | -0.8 |
| Arnulphe | 1597 ± 363 | 1234.5 | 181.3 | 0.06 | 169 | 96 | True | 4 | 1-0-3 | 1 | -0.4 |
| Céline | 1476 ± 244 | 1231.7 | 122.2 | 0.06 | 170 | 179 | True | 16 | 4-6-6 | 2 | -1.225 |
| d’Orgeville | 1499 ± 269 | 1229.7 | 134.5 | 0.06 | 171 | 128 | True | 7 | 1-1-5 | 1 | -0.8 |
| d'Orléans | 1579 ± 350 | 1228.5 | 175.0 | 0.06 | 172 | 109 | True | 5 | 2-1-2 | 1 | -0.8 |
| prince Von | 1484 ± 259 | 1225.2 | 129.5 | 0.06 | 173 | 131 | True | 8 | 3-3-2 | 3 | -1.66 |
| Sainte-Beuve | 1489 ± 267 | 1222.3 | 133.4 | 0.06 | 174 | 178 | True | 7 | 1-2-4 | 1 | -0.8 |
| le grand-duc Wladimir | 1591 ± 369 | 1222.1 | 184.3 | 0.06 | 175 | 108 | True | 3 | 2-1-0 | 1 | -0.4 |
| Barrès | 1467 ± 248 | 1218.3 | 124.1 | 0.06 | 176 | 144 | True | 9 | 1-1-7 | 1 | -0.8 |
| Clémenceau | 1467 ± 248 | 1218.3 | 124.1 | 0.06 | 177 | 146 | True | 9 | 1-1-7 | 1 | -0.8 |
| le marquis de Ganançay | 1567 ± 350 | 1217.0 | 174.8 | 0.06 | 178 | 87 | True | 6 | 3-1-2 | 1 | -0.8 |
| le marquis de Palancy | 1567 ± 350 | 1217.0 | 174.8 | 0.06 | 179 | 90 | True | 6 | 3-1-2 | 1 | -0.8 |
| Coquelin | 1538 ± 321 | 1216.6 | 160.6 | 0.06 | 180 | 166 | True | 5 | 1-1-3 | 1 | -0.8 |
| Napoléon III | 1523 ± 307 | 1216.0 | 153.4 | 0.06 | 181 | 198 | True | 8 | 1-2-5 | 1 | -0.8 |
| Liszt | 1507 ± 303 | 1203.6 | 151.6 | 0.06 | 182 | 137 | True | 6 | 2-1-3 | 1 | -0.8 |
| Mme Ristori | 1507 ± 303 | 1203.6 | 151.6 | 0.06 | 183 | 139 | True | 6 | 2-1-3 | 1 | -0.8 |
| Mme d'Arpajon | 1346 ± 149 | 1196.7 | 74.7 | 0.06 | 184 | 276 | False | 37 | 6-23-8 | 8 | -1.85 |
| Mme de Sagan | 1565 ± 369 | 1196.1 | 184.6 | 0.06 | 185 | 116 | True | 3 | 1-0-2 | 1 | -0.4 |
| comtesse de Monteriender | 1548 ± 352 | 1195.6 | 175.9 | 0.06 | 186 | 150 | True | 4 | 1-1-2 | 1 | 0.0 |
| M. de La Rochefoucauld | 1518 ± 327 | 1190.9 | 163.6 | 0.06 | 187 | 126 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de La Rochefoucauld | 1518 ± 327 | 1190.9 | 163.6 | 0.06 | 188 | 115 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de Praslin | 1518 ± 327 | 1190.9 | 163.6 | 0.06 | 189 | 117 | True | 6 | 2-1-3 | 1 | -0.8 |
| M. de Stermaria | 1443 ± 256 | 1187.3 | 127.8 | 0.06 | 190 | 249 | True | 10 | 3-5-2 | 4 | -1.359 |
| Mme Trombert | 1530 ± 346 | 1184.1 | 173.0 | 0.06 | 191 | 175 | True | 4 | 1-1-2 | 1 | -0.4 |
| elle | 1685 ± 502 | 1183.0 | 251.0 | 0.06 | 192 | 127 | True | 1 | 1-0-0 | 1 | +0.02 |
| princesse Sherbatoff | 1354 ± 174 | 1180.2 | 86.8 | 0.06 | 193 | 277 | False | 19 | 5-13-1 | 5 | -0.884 |
| le prince Von | 1415 ± 243 | 1171.8 | 121.6 | 0.06 | 194 | 201 | True | 10 | 3-5-2 | 2 | -1.263 |
| Poullein | 1653 ± 487 | 1165.7 | 243.6 | 0.06 | 195 | 141 | True | 2 | 1-1-0 | 2 | -0.66 |
| marquis de Cambremer | 1299 ± 136 | 1163.0 | 67.8 | 0.0601 | 196 | 285 | False | 45 | 7-24-14 | 6 | -1.173 |
| princesse Mathilde | 1479 ± 319 | 1160.0 | 159.4 | 0.06 | 197 | 182 | True | 7 | 2-3-2 | 2 | -0.6 |
| docteur Dieulafoy | 1661 ± 502 | 1158.9 | 251.0 | 0.06 | 198 | 125 | True | 1 | 1-0-0 | 1 | +3.83 |
| D'Annunzio | 1475 ± 321 | 1154.2 | 160.3 | 0.06 | 199 | 176 | True | 5 | 1-2-2 | 1 | -0.4 |
| M. d'Herweck | 1453 ± 300 | 1152.4 | 150.2 | 0.06 | 200 | 163 | True | 5 | 2-3-0 | 2 | -2.204 |
| le roi Théodose | 1453 ± 305 | 1147.4 | 152.7 | 0.06 | 201 | 197 | True | 8 | 2-4-2 | 3 | -0.14 |
| M. Swann, le père | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 202 | 142 | True | 7 | 1-1-5 | 1 | -0.8 |
| le comte de Paris | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 203 | 155 | True | 7 | 1-1-5 | 1 | -0.8 |
| le prince de Galles | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 204 | 156 | True | 7 | 1-1-5 | 1 | -0.8 |
| L’excellent écrivain G… | 1489 ± 349 | 1140.2 | 174.3 | 0.06 | 205 | 129 | True | 4 | 1-1-2 | 1 | -0.8 |
| M. Molé | 1436 ± 300 | 1135.9 | 149.9 | 0.06 | 206 | 202 | True | 8 | 1-2-5 | 1 | -0.8 |
| M. de Bouillon | 1436 ± 300 | 1135.9 | 149.9 | 0.06 | 207 | 200 | True | 8 | 1-2-5 | 1 | -0.8 |
| Musset | 1436 ± 300 | 1135.9 | 149.9 | 0.06 | 208 | 188 | True | 8 | 1-2-5 | 1 | -0.8 |
| Victor Hugo | 1436 ± 300 | 1135.9 | 149.9 | 0.06 | 209 | 185 | True | 8 | 1-2-5 | 1 | -0.8 |
| Mme d'Heudicourt | 1334 ± 199 | 1134.3 | 99.7 | 0.06 | 210 | 267 | False | 18 | 3-11-4 | 5 | -1.7 |
| marquise de Gallardon | 1350 ± 234 | 1116.1 | 117.0 | 0.06 | 211 | 270 | True | 19 | 1-10-8 | 7 | -2.18 |
| Mme de Franquetot | 1281 ± 170 | 1111.0 | 84.8 | 0.0601 | 212 | 279 | False | 23 | 4-13-6 | 3 | -1.088 |
| Théodore | 1624 ± 514 | 1109.6 | 257.0 | 0.06 | 213 | 111 | True | 2 | 1-0-1 | 1 | +2.26 |
| M. de Miribel | 1478 ± 372 | 1105.2 | 186.2 | 0.06 | 214 | 168 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Henry | 1478 ± 372 | 1105.2 | 186.2 | 0.06 | 215 | 174 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Picquart | 1478 ± 372 | 1105.2 | 186.2 | 0.06 | 216 | 173 | True | 4 | 1-1-2 | 1 | -0.8 |
| duc de Châtellerault | 1353 ± 254 | 1099.2 | 127.0 | 0.06 | 217 | 244 | True | 10 | 1-6-3 | 5 | -1.636 |
| M. de Courgivaux | 1662 ± 581 | 1081.7 | 290.3 | 0.06 | 218 | 113 | True | 1 | 1-0-0 | 1 | +2.42 |
| Octave | 1500 ± 422 | 1078.3 | 210.8 | 0.06 | 219 | 165 | True | 4 | 2-2-0 | 2 | -0.375 |
| Théodose Cadet | 1451 ± 373 | 1078.1 | 186.6 | 0.06 | 220 | 199 | True | 3 | 1-2-0 | 1 | -2.258 |
| Mme de Villebon | 1662 ± 585 | 1077.6 | 292.4 | 0.06 | 221 | 114 | True | 1 | 1-0-0 | 1 | -1.15 |
| baron de Guermantes | 1662 ± 585 | 1077.2 | 292.6 | 0.06 | 222 | 122 | True | 1 | 1-0-0 | 1 | -0.4 |
| Beauserfeuil | 1450 ± 373 | 1076.8 | 186.6 | 0.06 | 223 | 204 | True | 3 | 1-2-0 | 1 | -0.95 |
| le capitaine | 1488 ± 413 | 1075.1 | 206.3 | 0.06 | 224 | 170 | True | 2 | 1-1-0 | 1 | +0.15 |
| docteur Percepied | 1500 ± 426 | 1074.1 | 212.9 | 0.06 | 225 | 157 | True | 4 | 1-1-2 | 1 | -0.8 |
| Madame Elstir | 1403 ± 333 | 1069.7 | 166.7 | 0.06 | 226 | 195 | True | 6 | 1-2-3 | 1 | -0.8 |
| les demoiselles d’Ambresac | 1403 ± 333 | 1069.7 | 166.7 | 0.06 | 227 | 187 | True | 6 | 1-2-3 | 1 | -0.8 |
| M. de Grouchy | 1324 ± 254 | 1069.1 | 127.2 | 0.06 | 228 | 258 | True | 10 | 2-8-0 | 4 | -0.841 |
| le bâtonnier | 1467 ± 405 | 1062.5 | 202.4 | 0.06 | 229 | 143 | True | 3 | 1-1-1 | 1 | -0.4 |
| M. Bontemps | 1352 ± 291 | 1061.2 | 145.6 | 0.06 | 230 | 262 | True | 9 | 2-7-0 | 2 | -0.251 |
| Cartier | 1398 ± 339 | 1058.5 | 169.7 | 0.06 | 231 | 221 | True | 4 | 1-3-0 | 1 | -1.965 |
| M. Grevy | 1476 ± 423 | 1053.0 | 211.6 | 0.06 | 232 | 161 | True | 3 | 1-1-1 | 1 | -0.4 |
| capitaine de Borodino | 1256 ± 207 | 1048.8 | 103.6 | 0.06 | 233 | 278 | True | 14 | 2-11-1 | 5 | -1.962 |
| Saniette | 1199 ± 158 | 1040.9 | 79.1 | 0.0601 | 234 | 288 | False | 35 | 1-27-7 | 9 | -3.455 |
| prince d'Agrigente | 1487 ± 451 | 1035.5 | 225.7 | 0.06 | 235 | 162 | True | 2 | 1-1-0 | 2 | -0.3 |
| M. Barrère | 1530 ± 501 | 1029.0 | 250.5 | 0.06 | 236 | 145 | True | 1 | 0-0-1 | 1 | -1.59 |
| Mme de Souvré | 1288 ± 266 | 1021.3 | 133.1 | 0.06 | 237 | 268 | True | 11 | 2-9-0 | 2 | -1.735 |
| Antoine | 1430 ± 411 | 1019.0 | 205.7 | 0.06 | 238 | 212 | True | 3 | 0-2-1 | 1 | -0.8 |
| la jeune ouvriere | 1451 ± 444 | 1007.4 | 221.9 | 0.06 | 239 | 171 | True | 2 | 0-1-1 | 1 | -0.4 |
| professeur E… | 1366 ± 370 | 995.8 | 185.2 | 0.06 | 240 | 237 | True | 4 | 1-3-0 | 2 | -1.805 |
| Maurice | 1261 ± 286 | 974.5 | 143.2 | 0.06 | 241 | 266 | True | 7 | 1-6-0 | 1 | -2.498 |
| Vigny | 1448 ± 484 | 963.9 | 242.0 | 0.06 | 242 | 158 | True | 2 | 1-1-0 | 1 | -1.852 |
| les Courvoisier | 1304 ± 349 | 954.9 | 174.3 | 0.06 | 243 | 242 | True | 5 | 1-4-0 | 1 | -1.62 |
| colonel de Froberville | 1159 ± 208 | 951.2 | 104.0 | 0.06 | 244 | 283 | True | 14 | 0-14-0 | 1 | -4.744 |
| l'ambassadrice de Turquie | 1263 ± 327 | 936.0 | 163.3 | 0.06 | 245 | 257 | True | 4 | 0-4-0 | 1 | -2.945 |
| Alix | 1184 ± 252 | 932.5 | 126.0 | 0.06 | 246 | 274 | True | 9 | 0-8-1 | 3 | -3.354 |
| Prince Henri d'Orléans | 1347 ± 427 | 919.4 | 213.7 | 0.06 | 247 | 207 | True | 2 | 0-1-1 | 1 | -1.667 |
| Mme de Morienval | 1257 ± 350 | 907.7 | 174.8 | 0.06 | 248 | 250 | True | 6 | 1-4-1 | 1 | -1.6 |
| duchesse de Luxembourg | 1257 ± 350 | 907.7 | 174.8 | 0.06 | 249 | 248 | True | 6 | 1-4-1 | 1 | -1.6 |
| prince de Faffenheim | 1258 ± 361 | 896.9 | 180.7 | 0.06 | 250 | 235 | True | 3 | 0-3-0 | 2 | -1.508 |
| le prince de Faffenheim | 1199 ± 305 | 894.2 | 152.6 | 0.06 | 251 | 263 | True | 5 | 0-5-0 | 1 | -5.178 |
| marquise de Citri | 1305 ± 414 | 891.1 | 207.0 | 0.06 | 252 | 223 | True | 2 | 0-2-0 | 1 | -3.51 |
| le prince von *** | 1304 ± 415 | 889.1 | 207.6 | 0.06 | 253 | 220 | True | 2 | 0-2-0 | 1 | -3.177 |
| M. de Luxembourg | 1324 ± 436 | 889.0 | 217.8 | 0.06 | 254 | 231 | True | 2 | 0-2-0 | 1 | -0.69 |
| le diplomate belge | 1307 ± 426 | 881.7 | 212.9 | 0.06 | 255 | 225 | True | 2 | 0-2-0 | 1 | -1.88 |
| Mme Iéna | 1188 ± 316 | 872.0 | 158.0 | 0.06 | 256 | 261 | True | 5 | 0-5-0 | 1 | -3.59 |
| Monsieur Vallenères | 1290 ± 418 | 871.6 | 209.2 | 0.06 | 257 | 218 | True | 3 | 0-2-1 | 1 | -2.474 |
| Picquart | 1150 ± 280 | 870.1 | 139.9 | 0.06 | 258 | 273 | True | 8 | 0-8-0 | 2 | -2.132 |
| la cousine d'Oriane | 1233 ± 368 | 864.6 | 184.2 | 0.06 | 259 | 247 | True | 3 | 0-3-0 | 1 | -1.939 |
| l'historien de la Fronde | 1248 ± 396 | 852.0 | 198.1 | 0.06 | 260 | 240 | True | 3 | 0-3-0 | 1 | -1.48 |
| prince Foggi | 1352 ± 501 | 851.4 | 250.5 | 0.06 | 261 | 184 | True | 1 | 0-1-0 | 1 | -1.78 |
| vicomtesse d'Égremont | 1255 ± 404 | 850.8 | 202.1 | 0.06 | 262 | 243 | True | 3 | 0-3-0 | 1 | -3.51 |
| prince de Léon | 1292 ± 456 | 836.0 | 227.8 | 0.06 | 263 | 234 | True | 2 | 0-2-0 | 1 | -0.4 |
| M. de Vigny | 1133 ± 300 | 833.6 | 149.9 | 0.06 | 264 | 272 | True | 8 | 0-8-0 | 1 | -3.167 |
| le professeur E… | 1289 ± 456 | 833.2 | 227.8 | 0.06 | 265 | 222 | True | 2 | 0-2-0 | 1 | -3.82 |
| l'empereur | 1198 ± 366 | 831.0 | 183.2 | 0.06 | 266 | 254 | True | 4 | 0-4-0 | 1 | -3.114 |
| Marie Gineste | 1281 ± 453 | 828.5 | 226.3 | 0.06 | 267 | 215 | True | 2 | 0-2-0 | 1 | -0.4 |
| Mme de Varambon | 1178 ± 356 | 821.9 | 177.9 | 0.06 | 268 | 255 | True | 4 | 0-4-0 | 2 | -2.945 |
| Mme de Simiane | 1246 ± 428 | 818.3 | 213.9 | 0.06 | 269 | 245 | True | 3 | 0-3-0 | 1 | -1.62 |
| princesse de Nassau | 1310 ± 496 | 814.1 | 248.0 | 0.06 | 270 | 180 | True | 1 | 0-1-0 | 1 | -2.86 |
| Mme Blandais | 1179 ± 367 | 811.4 | 183.7 | 0.06 | 271 | 256 | True | 4 | 0-4-0 | 2 | -2.859 |
| princesse d'Iéna | 1258 ± 460 | 797.8 | 229.9 | 0.06 | 272 | 230 | True | 3 | 0-2-1 | 1 | -1.96 |
| La Moussaye | 1500 ± 703 | 797.2 | 351.4 | 0.06 | 273 | 151 | True | 0 | 0-0-0 | 1 | -0.4 |
| Périgot (Joseph) | 1500 ± 704 | 796.3 | 351.9 | 0.06 | 274 | 153 | True | 0 | 0-0-0 | 1 | -2.425 |
| la « marquise » | 1500 ± 704 | 796.0 | 352.0 | 0.06 | 275 | 154 | True | 0 | 0-0-0 | 1 | -2.95 |
| Mme Poncin | 1500 ± 704 | 795.7 | 352.2 | 0.06 | 276 | 152 | True | 0 | 0-0-0 | 1 | +0.347 |
| la marquise | 1296 ± 502 | 794.1 | 251.1 | 0.06 | 277 | 211 | True | 1 | 0-1-0 | 1 | -1.795 |
| Mme Blatin | 1228 ± 440 | 788.3 | 219.8 | 0.06 | 278 | 239 | True | 2 | 0-2-0 | 1 | -3.773 |
| le grand-duc héritier de Luxembourg | 1280 ± 505 | 774.1 | 252.7 | 0.06 | 279 | 205 | True | 1 | 0-1-0 | 1 | -1.41 |
| M. Pierre | 1142 ± 372 | 769.9 | 185.9 | 0.06 | 280 | 260 | True | 4 | 0-4-0 | 2 | -3.527 |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 757.1 | 290.3 | 0.06 | 281 | 191 | True | 1 | 0-1-0 | 1 | -2.66 |
| comtesse G… | 1338 ± 585 | 752.9 | 292.4 | 0.06 | 282 | 189 | True | 1 | 0-1-0 | 1 | -1.941 |
| ma grand’tante | 1105 ± 353 | 751.8 | 176.5 | 0.06 | 283 | 271 | True | 7 | 0-7-0 | 1 | -1.65 |
| la Charité de Giotto | 1338 ± 587 | 750.7 | 293.5 | 0.06 | 284 | 181 | True | 1 | 0-1-0 | 1 | -4.105 |
| ma grand'tante | 1338 ± 587 | 750.7 | 293.5 | 0.06 | 285 | 190 | True | 1 | 0-1-0 | 1 | -1.1 |
| Madame d'Ambresac | 1253 ± 512 | 740.8 | 256.0 | 0.06 | 286 | 229 | True | 2 | 0-2-0 | 1 | 0.0 |
| Dumont | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 287 | 227 | True | 2 | 0-2-0 | 1 | -2.28 |
| le curé | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 288 | 226 | True | 2 | 0-2-0 | 1 | -2.55 |
