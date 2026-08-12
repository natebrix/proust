# Character Glicko-2

- Analysis version: `character_glicko2_prestige_v1`
- Lens: `prestige`
- Source review version: `corpus_sanity_review_v1`
- Character count: `288`
- Match count: `5756`
- Draw rate: `0.322`
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

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Prestige |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mlle d'Oloron | 1862 ± 197 | 1665.1 | 98.5 | 0.06 | 14 | 14-0-0 | 1 | +1.92 |
| docteur du Boulbon | 1762 ± 165 | 1596.6 | 82.4 | 0.06 | 27 | 19-3-5 | 6 | -0.435 |
| comte de Forcheville | 1642 ± 100 | 1542.5 | 49.9 | 0.0604 | 112 | 56-19-37 | 25 | -0.29 |
| Françoise | 1624 ± 82 | 1542.4 | 41.0 | 0.0602 | 217 | 99-50-68 | 82 | -0.267 |
| Léa | 1724 ± 195 | 1529.4 | 97.3 | 0.06 | 14 | 8-0-6 | 4 | -0.7 |
| Bergotte | 1621 ± 93 | 1528.5 | 46.4 | 0.06 | 129 | 51-31-47 | 36 | -0.094 |
| le peintre | 1680 ± 160 | 1519.3 | 80.2 | 0.06 | 42 | 16-4-22 | 8 | -0.296 |
| Morel | 1585 ± 79 | 1505.4 | 39.6 | 0.0604 | 152 | 48-51-53 | 32 | -0.876 |
| le grand-père du narrateur | 1649 ± 146 | 1502.4 | 73.1 | 0.06 | 63 | 26-7-30 | 16 | -0.612 |
| Mme Verdurin | 1565 ± 68 | 1497.4 | 33.8 | 0.0598 | 311 | 100-98-113 | 82 | -0.738 |

## Bottom Rated Characters

| Character | Rating | Conservative | RD | Volatility | Matches | W-L-D | Units | Mean Prestige |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Saniette | 1247 ± 154 | 1092.9 | 76.9 | 0.0601 | 35 | 2-25-8 | 9 | -2.784 |
| Mme d'Heudicourt | 1306 ± 199 | 1107.3 | 99.3 | 0.06 | 18 | 3-12-3 | 5 | -1.469 |
| Mme de Franquetot | 1280 ± 170 | 1109.7 | 85.0 | 0.0601 | 23 | 4-13-6 | 3 | -0.837 |
| marquis de Cambremer | 1288 ± 136 | 1152.7 | 67.8 | 0.0601 | 45 | 6-24-15 | 6 | -1.12 |
| princesse Sherbatoff | 1357 ± 174 | 1183.3 | 86.8 | 0.06 | 19 | 4-12-3 | 5 | -0.701 |
| Mme d'Arpajon | 1357 ± 150 | 1207.6 | 74.9 | 0.06 | 37 | 6-22-9 | 8 | -1.72 |
| tante Léonie | 1415 ± 180 | 1235.3 | 89.9 | 0.06 | 38 | 11-22-5 | 22 | -0.717 |
| Rosemonde | 1436 ± 190 | 1245.7 | 95.0 | 0.06 | 20 | 5-7-8 | 4 | -0.7 |
| général de Monserfeuil | 1442 ± 196 | 1246.1 | 97.8 | 0.06 | 18 | 5-8-5 | 4 | -1.289 |
| M. de Vaugoubert | 1410 ± 160 | 1250.1 | 80.2 | 0.06 | 35 | 6-12-17 | 9 | -1.131 |

## Provisional Characters

Characters whose RD is still above the provisional threshold -- their rating should be treated as unstable.

| Character | Rating | RD | Matches | Units | Last Period |
| --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1892 ± 223 | 111.6 | 17 | 3 | v5 |
| Mme de Grouchy | 1873 ± 376 | 188.2 | 4 | 1 | v3-p2 |
| Céleste Albaret | 1867 ± 211 | 105.3 | 17 | 3 | v5 |
| Mlle de Saint-Loup | 1830 ± 251 | 125.3 | 7 | 2 | v7-p4-le-bal-de-tetes |
| marquis de Beausergent | 1824 ± 203 | 101.5 | 12 | 1 | v7-p4-le-bal-de-tetes |
| Mme Elstir | 1823 ± 272 | 136.1 | 7 | 1 | v2-p2-noms-de-pays-le-pays |
| prince de Saxe | 1822 ± 370 | 185.2 | 3 | 1 | v3-p1 |
| Mme de Chaussepierre | 1821 ± 339 | 169.5 | 4 | 1 | v5 |
| marquis Maurice de Vaudémont | 1816 ± 470 | 235.0 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Trémoïlle | 1804 ± 424 | 212.0 | 3 | 1 | v1-p2-un-amour-de-swann |
| colonel Picquart | 1795 ± 349 | 174.5 | 4 | 1 | v3-p1 |
| Mlle de Stermaria | 1785 ± 274 | 136.8 | 10 | 5 | v3-p2 |
| duc de Sidonia | 1780 ± 456 | 228.2 | 2 | 1 | v4-p2 |
| Marie | 1778 ± 272 | 136.0 | 7 | 1 | v4-p2 |
| Maeterlinck | 1775 ± 322 | 161.2 | 5 | 1 | v3-p1 |
| Lady Israels | 1760 ± 455 | 227.5 | 2 | 1 | v2-p1-autour-de-mme-swann |
| Duroc | 1753 ± 454 | 227.2 | 2 | 1 | v3-p1 |
| Eulalie | 1747 ± 228 | 114.2 | 16 | 7 | v5 |
| duc d'Aumale | 1741 ± 377 | 188.3 | 4 | 2 | v3-p2 |
| le pianiste | 1740 ± 271 | 135.7 | 10 | 3 | v1-p2-un-amour-de-swann |
| Herbinger | 1740 ± 409 | 204.4 | 3 | 1 | v1-p2-un-amour-de-swann |
| le commandant Duroc | 1731 ± 412 | 206.2 | 2 | 1 | v3-p1 |
| Mlle Bloch | 1729 ± 425 | 212.4 | 2 | 1 | v4-p2 |
| Émilie Daltier | 1721 ± 358 | 179.2 | 3 | 1 | v5 |
| marquis du Lau | 1715 ± 308 | 153.9 | 5 | 2 | v6-p2 |
| Gribelin | 1711 ± 311 | 155.6 | 6 | 1 | v3-p1 |
| Bibi | 1711 ± 424 | 212.2 | 2 | 1 | v3-p2 |
| Rémi | 1711 ± 223 | 111.4 | 17 | 3 | v1-p2-un-amour-de-swann |
| Victurnien | 1707 ± 268 | 134.1 | 8 | 2 | v4-p2 |
| elle | 1692 ± 503 | 251.3 | 1 | 1 | v3-p1 |
| Dechambre | 1689 ± 366 | 183.0 | 3 | 1 | v4-p2 |
| monsieur Vallenères | 1689 ± 426 | 213.1 | 2 | 1 | v3-p1 |
| Létourville | 1681 ± 353 | 176.6 | 3 | 1 | v7-p4-le-bal-de-tetes |
| Bismarck | 1680 ± 354 | 176.9 | 4 | 1 | v2-p1-autour-de-mme-swann |
| grand-duc héritier de Luxembourg | 1666 ± 250 | 125.1 | 9 | 2 | v3-p2 |
| les La Trémoïlle | 1666 ± 329 | 164.4 | 7 | 1 | v1-p2-un-amour-de-swann |
| M. de Courgivaux | 1662 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| Mme de Villebon | 1662 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| baron de Guermantes | 1662 ± 585 | 292.6 | 1 | 1 | v3-p1 |
| docteur Dieulafoy | 1659 ± 502 | 251.1 | 1 | 1 | v3-p2 |
| Poullein | 1646 ± 483 | 241.6 | 2 | 2 | v3-p2 |
| la duchesse d'Alençon | 1645 ± 314 | 157.2 | 6 | 1 | v3-p2 |
| M. Vibert | 1638 ± 373 | 186.6 | 3 | 1 | v3-p2 |
| Mme de Stermaria | 1637 ± 309 | 154.6 | 5 | 1 | v3-p2 |
| M. d'Orsan | 1628 ± 270 | 134.9 | 11 | 1 | v1-p2-un-amour-de-swann |
| Marie-Aynard | 1624 ± 308 | 153.9 | 7 | 1 | v3-p1 |
| Victurnienne | 1624 ± 308 | 153.9 | 7 | 1 | v3-p1 |
| Théodore | 1624 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| Lady Rufus Israël | 1621 ± 281 | 140.3 | 6 | 1 | v6-p2 |
| M. de Bornier | 1616 ± 346 | 173.0 | 5 | 1 | v3-p2 |
| cousine Poictiers | 1615 ± 318 | 159.1 | 5 | 1 | v3-p1 |
| duc de Poictiers | 1615 ± 318 | 159.1 | 5 | 1 | v3-p1 |
| Manet | 1607 ± 317 | 158.6 | 5 | 1 | v3-p2 |
| Flora | 1602 ± 336 | 168.2 | 8 | 1 | v1-p1-combray |
| Arnulphe | 1598 ± 363 | 181.6 | 4 | 1 | v4-p2 |
| le baron Bréau-Chenut | 1593 ± 310 | 154.8 | 7 | 1 | v2-p1-autour-de-mme-swann |
| le vieux père Chenut | 1593 ± 310 | 154.8 | 7 | 1 | v2-p1-autour-de-mme-swann |
| le grand-duc Wladimir | 1591 ± 369 | 184.6 | 3 | 1 | v4-p2 |
| Sarah Bernhardt | 1590 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| le jeune prince de Foix | 1590 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| vicomte de Courvoisier | 1590 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| Mlle de l’Orgeville | 1590 ± 352 | 176.1 | 3 | 1 | v6-p4 |
| la jeune ouvriere | 1589 ± 445 | 222.3 | 2 | 1 | v1-p2-un-amour-de-swann |
| Lady Israël | 1587 ± 323 | 161.5 | 5 | 1 | v3-p1 |
| Charcot | 1584 ± 225 | 112.6 | 12 | 1 | v3-p1 |
| M. Reinach | 1584 ± 225 | 112.6 | 12 | 1 | v3-p1 |
| d'Orléans | 1581 ± 350 | 174.9 | 5 | 1 | v2-p2-noms-de-pays-le-pays |
| Mlle d'Éporcheville | 1573 ± 224 | 111.9 | 10 | 2 | v6-p2 |
| Mme Leroi | 1573 ± 206 | 103.2 | 13 | 5 | v3-p1 |
| jeune blonde de Rivebelle | 1573 ± 333 | 166.5 | 6 | 2 | v2-p2-noms-de-pays-le-pays |
| M. de Beauserfeuil | 1573 ± 278 | 138.9 | 7 | 1 | v3-p2 |
| le marquis de Ganançay | 1568 ± 349 | 174.6 | 6 | 1 | v3-p1 |
| le marquis de Palancy | 1568 ± 349 | 174.6 | 6 | 1 | v3-p1 |
| duc de Chartres | 1567 ± 209 | 104.3 | 14 | 1 | v4-p2 |
| prince de Chimay | 1567 ± 209 | 104.3 | 14 | 1 | v4-p2 |
| Élisabeth | 1566 ± 284 | 142.2 | 6 | 1 | v5 |
| Mme de Sagan | 1565 ± 369 | 184.6 | 3 | 1 | v3-p1 |
| Mme Legrandin mère | 1564 ± 279 | 139.6 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victoire | 1564 ± 279 | 139.6 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de Létourville | 1564 ± 292 | 146.0 | 5 | 1 | v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle |
| Mme Timoléon d'Amoncourt | 1562 ± 244 | 121.9 | 9 | 1 | v4-p2 |
| M. de Chateaubriand | 1555 ± 285 | 142.7 | 11 | 2 | v6-p2 |
| comtesse de Monteriender | 1550 ± 352 | 176.1 | 4 | 1 | v1-p2-un-amour-de-swann |
| le jeune marquis de Cambremer | 1546 ± 202 | 100.8 | 12 | 1 | v6-p4 |
| Coquelin | 1542 ± 321 | 160.3 | 5 | 1 | v1-p3-noms-de-pays-le-nom |
| princesse d'Épinay | 1539 ± 242 | 120.8 | 12 | 3 | v3-p2 |
| prince d’Agrigente | 1537 ± 202 | 101.0 | 15 | 2 | v6-p2 |
| Sir Rufus Israël | 1536 ± 282 | 140.9 | 7 | 1 | v3-p1 |
| M. Arthur Meyer | 1535 ± 270 | 134.8 | 6 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| Mme Trombert | 1533 ± 346 | 173.0 | 4 | 1 | v2-p1-autour-de-mme-swann |
| Mme de Montmorency | 1532 ± 222 | 110.8 | 11 | 1 | v4-p2 |
| Mme de Rochechouart | 1532 ± 222 | 110.8 | 11 | 1 | v4-p2 |
| M. Barrère | 1531 ± 501 | 250.6 | 1 | 1 | v6-p3 |
| Thibaud | 1527 ± 245 | 122.5 | 8 | 1 | v5 |
| Napoléon III | 1525 ± 307 | 153.6 | 8 | 1 | v1-p2-un-amour-de-swann |
| Mme Putbus | 1524 ± 251 | 125.7 | 8 | 1 | v5 |
| M. de La Rochefoucauld | 1520 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de La Rochefoucauld | 1520 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| duchesse de Praslin | 1520 ± 327 | 163.6 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| prince de Sagan | 1518 ± 281 | 140.7 | 7 | 1 | v4-p2 |
| Dostoïevski | 1515 ± 273 | 136.7 | 6 | 1 | v5 |
| comte de Paris | 1511 ± 250 | 125.0 | 10 | 3 | v2-p1-autour-de-mme-swann |
| Liszt | 1510 ± 303 | 151.5 | 6 | 1 | v3-p1 |
| Mme Ristori | 1510 ± 303 | 151.5 | 6 | 1 | v3-p1 |
| M. Carnot | 1509 ± 252 | 126.0 | 9 | 1 | v3-p2 |
| Mme Carnot | 1509 ± 252 | 126.0 | 9 | 1 | v3-p2 |
| Mme de Vaugoubert | 1508 ± 262 | 130.8 | 9 | 2 | v5 |
| l'abbé Poiré | 1507 ± 225 | 112.3 | 10 | 1 | v4-p2 |
| Octave | 1507 ± 423 | 211.3 | 4 | 2 | v6-p2 |
| Gisèle | 1505 ± 253 | 126.6 | 14 | 5 | v5 |
| M. de Goncourt | 1504 ± 237 | 118.3 | 8 | 1 | v7-p1-a-tansonville |
| d’Orgeville | 1501 ± 269 | 134.5 | 7 | 1 | v4-p2 |
| La Moussaye | 1500 ± 703 | 351.4 | 0 | 1 | v5 |
| M. Swann, le père | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| Mme Poncin | 1500 ± 704 | 352.2 | 0 | 1 | v2-p2-noms-de-pays-le-pays |
| Périgot (Joseph) | 1500 ± 704 | 351.9 | 0 | 1 | v3-p2 |
| docteur Percepied | 1500 ± 426 | 212.9 | 4 | 1 | v1-p1-combray |
| la Charité de Giotto | 1500 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| la « marquise » | 1500 ± 704 | 352.0 | 0 | 1 | v3-p1 |
| le comte de Paris | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| le prince de Galles | 1500 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |
| M. de Marsantes | 1498 ± 301 | 150.4 | 7 | 2 | v3-p1 |
| le capitaine | 1494 ± 412 | 206.2 | 2 | 1 | v3-p1 |
| prince d'Agrigente | 1493 ± 456 | 228.2 | 2 | 2 | v7-p4-le-bal-de-tetes |
| Léonor de Cambremer | 1492 ± 203 | 101.5 | 12 | 1 | v7-p4-le-bal-de-tetes |
| Sainte-Beuve | 1490 ± 267 | 133.4 | 7 | 1 | v3-p2 |
| comtesse douairière d'Argencourt | 1490 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| duchesse de Gallardon douairière | 1490 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| marquis de Fierbois | 1490 ± 244 | 121.9 | 10 | 1 | v3-p2 |
| M. Grevy | 1481 ± 424 | 212.0 | 3 | 1 | v1-p2-un-amour-de-swann |
| Barrès | 1479 ± 247 | 123.5 | 9 | 1 | v3-p2 |
| Clémenceau | 1479 ± 247 | 123.5 | 9 | 1 | v3-p2 |
| M. de Miribel | 1479 ± 372 | 186.1 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Henry | 1479 ± 372 | 186.1 | 4 | 1 | v3-p1 |
| le lieutenant-colonel Picquart | 1479 ± 372 | 186.1 | 4 | 1 | v3-p1 |
| princesse Mathilde | 1479 ± 318 | 158.9 | 7 | 2 | v3-p2 |
| L’excellent écrivain G… | 1479 ± 345 | 172.7 | 4 | 1 | v3-p1 |
| D'Annunzio | 1477 ± 321 | 160.5 | 5 | 1 | v4-p2 |
| prince de Foix | 1477 ± 214 | 107.0 | 14 | 3 | v7-p2-m-de-charlus-pendant-la-guerre |
| le bâtonnier | 1471 ± 405 | 202.7 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| Céline | 1466 ± 250 | 124.8 | 16 | 2 | v2-p2-noms-de-pays-le-pays |
| le roi Théodose | 1455 ± 306 | 153.1 | 8 | 3 | v4-p2 |
| M. d'Herweck | 1452 ± 300 | 150.0 | 5 | 2 | v4-p2 |
| oncle Adolphe | 1452 ± 232 | 115.8 | 20 | 6 | v3-p1 |
| Théodose Cadet | 1450 ± 372 | 185.8 | 3 | 1 | v3-p2 |
| Vigny | 1449 ± 484 | 242.0 | 2 | 1 | v2-p2-noms-de-pays-le-pays |
| Beauserfeuil | 1448 ± 372 | 185.8 | 3 | 1 | v3-p2 |
| prince Von | 1444 ± 258 | 129.0 | 8 | 3 | v3-p2 |
| M. Molé | 1438 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| M. de Bouillon | 1438 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Musset | 1438 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| Victor Hugo | 1438 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| le professeur E… | 1437 ± 456 | 228.2 | 2 | 1 | v4-p2 |
| marquise de Citri | 1426 ± 414 | 206.9 | 2 | 1 | v4-p2 |
| Antoine | 1418 ± 406 | 203.1 | 3 | 1 | v3-p1 |
| Madame Elstir | 1405 ± 333 | 166.5 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| les demoiselles d’Ambresac | 1405 ± 333 | 166.5 | 6 | 1 | v2-p2-noms-de-pays-le-pays |
| Cartier | 1395 ± 339 | 169.5 | 4 | 1 | v5 |
| le prince Von | 1374 ± 242 | 121.2 | 10 | 2 | v3-p2 |
| professeur E… | 1373 ± 370 | 184.9 | 4 | 2 | v4-p2 |
| M. de Grouchy | 1364 ± 253 | 126.4 | 10 | 4 | v3-p2 |
| M. de Stermaria | 1360 ± 255 | 127.7 | 10 | 4 | v2-p2-noms-de-pays-le-pays |
| prince Foggi | 1353 ± 501 | 250.6 | 1 | 1 | v6-p3 |
| Prince Henri d'Orléans | 1349 ± 427 | 213.6 | 2 | 1 | v3-p1 |
| duc de Châtellerault | 1348 ± 253 | 126.7 | 10 | 5 | v4-p2 |
| comtesse G… | 1338 ± 585 | 292.4 | 1 | 1 | v3-p2 |
| ma grand'tante | 1338 ± 587 | 293.5 | 1 | 1 | v1-p1-combray |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 290.3 | 1 | 1 | v7-p4-le-bal-de-tetes |
| M. de Luxembourg | 1322 ± 433 | 216.3 | 2 | 1 | v3-p2 |
| princesse de Nassau | 1308 ± 496 | 248.2 | 1 | 1 | v7-p4-le-bal-de-tetes |
| le prince von *** | 1306 ± 416 | 207.8 | 2 | 1 | v3-p1 |
| les Courvoisier | 1303 ± 348 | 174.1 | 5 | 1 | v3-p2 |
| le diplomate belge | 1301 ± 420 | 210.1 | 2 | 1 | v3-p1 |
| la marquise | 1301 ± 501 | 250.7 | 1 | 1 | v3-p1 |
| marquise de Gallardon | 1296 ± 245 | 122.6 | 19 | 7 | v4-p2 |
| prince de Léon | 1290 ± 455 | 227.6 | 2 | 1 | v5 |
| Mme de Souvré | 1287 ± 268 | 133.8 | 11 | 2 | v4-p2 |
| capitaine de Borodino | 1286 ± 207 | 103.4 | 14 | 5 | v3-p1 |
| Marie Gineste | 1284 ± 453 | 226.6 | 2 | 1 | v4-p2 |
| le grand-duc héritier de Luxembourg | 1274 ± 507 | 253.3 | 1 | 1 | v3-p2 |
| Maurice | 1264 ± 286 | 143.2 | 7 | 1 | v7-p2-m-de-charlus-pendant-la-guerre |
| l'ambassadrice de Turquie | 1264 ± 327 | 163.5 | 4 | 1 | v4-p2 |
| Mme de Morienval | 1260 ± 349 | 174.6 | 6 | 1 | v3-p1 |
| duchesse de Luxembourg | 1260 ± 349 | 174.6 | 6 | 1 | v3-p1 |
| prince de Faffenheim | 1258 ± 362 | 181.0 | 3 | 2 | v3-p1 |
| Dumont | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| Madame d'Ambresac | 1253 ± 512 | 256.0 | 2 | 1 | v3-p1 |
| le curé | 1253 ± 514 | 257.0 | 2 | 1 | v1-p1-combray |
| vicomtesse d'Égremont | 1253 ± 402 | 201.0 | 3 | 1 | v3-p2 |
| Mme de Simiane | 1246 ± 428 | 213.9 | 3 | 1 | v2-p2-noms-de-pays-le-pays |
| l'historien de la Fronde | 1239 ± 392 | 195.8 | 3 | 1 | v3-p1 |
| M. Bontemps | 1238 ± 292 | 145.8 | 9 | 2 | v7-p2-m-de-charlus-pendant-la-guerre |
| Mme Blatin | 1233 ± 438 | 219.1 | 2 | 1 | v1-p3-noms-de-pays-le-nom |
| la cousine d'Oriane | 1232 ± 370 | 184.8 | 3 | 1 | v3-p2 |
| Monsieur Vallenères | 1206 ± 419 | 209.3 | 3 | 1 | v3-p1 |
| le prince de Faffenheim | 1202 ± 305 | 152.6 | 5 | 1 | v3-p1 |
| l'empereur | 1194 ± 365 | 182.6 | 4 | 1 | v3-p2 |
| Mme Iéna | 1185 ± 315 | 157.6 | 5 | 1 | v3-p2 |
| colonel de Froberville | 1184 ± 209 | 104.3 | 14 | 1 | v4-p2 |
| Alix | 1182 ± 251 | 125.5 | 9 | 3 | v3-p1 |
| Mme Blandais | 1182 ± 368 | 183.9 | 4 | 2 | v2-p2-noms-de-pays-le-pays |
| Mme de Varambon | 1172 ± 355 | 177.5 | 4 | 2 | v3-p2 |
| princesse d'Iéna | 1159 ± 460 | 229.9 | 3 | 1 | v1-p2-un-amour-de-swann |
| Picquart | 1150 ± 278 | 139.1 | 8 | 2 | v3-p1 |
| M. Pierre | 1141 ± 372 | 185.8 | 4 | 2 | v3-p1 |
| M. de Vigny | 1135 ± 300 | 149.9 | 8 | 1 | v2-p2-noms-de-pays-le-pays |
| ma grand’tante | 1105 ± 353 | 176.5 | 7 | 1 | v1-p1-combray |

## Largest Glicko-vs-ELO Rank Divergences

| Character | Glicko Rank | ELO Rank | Delta | Rating | ELO |
| --- | --- | --- | --- | --- | --- |
| duchesse de Guermantes | 35 | 281 | -246 | 1524 ± 59 | 1392.071 |
| Albertine | 48 | 284 | -236 | 1503 ± 68 | 1376.649 |
| baron de Charlus | 43 | 279 | -236 | 1502 ± 60 | 1396.438 |
| Gilberte | 52 | 286 | -234 | 1486 ± 61 | 1337.768 |
| duc de Guermantes | 66 | 287 | -221 | 1456 ± 67 | 1323.953 |
| princesse de Guermantes | 45 | 263 | -218 | 1529 ± 91 | 1438.66 |
| Odette | 38 | 252 | -214 | 1521 ± 66 | 1459.843 |
| Brichot | 59 | 264 | -205 | 1488 ± 80 | 1438.272 |
| Swann | 55 | 251 | -196 | 1474 ± 61 | 1460.527 |
| Robert de Saint-Loup | 47 | 241 | -194 | 1496 ± 60 | 1469.412 |

## Character Table

| Character | Rating | Conservative | RD | Volatility | Glicko Rank | ELO Rank | Provisional | Matches | W-L-D | Units | Mean Prestige |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| la reine de Naples | 1892 ± 223 | 1668.6 | 111.6 | 0.06 | 1 | 4 | True | 17 | 16-1-0 | 3 | +0.184 |
| Mlle d'Oloron | 1862 ± 197 | 1665.1 | 98.5 | 0.06 | 2 | 3 | False | 14 | 14-0-0 | 1 | +1.92 |
| Céleste Albaret | 1867 ± 211 | 1656.3 | 105.3 | 0.06 | 3 | 2 | True | 17 | 16-1-0 | 3 | +1.52 |
| marquis de Beausergent | 1824 ± 203 | 1621.5 | 101.5 | 0.06 | 4 | 6 | True | 12 | 12-0-0 | 1 | -0.224 |
| docteur du Boulbon | 1762 ± 165 | 1596.6 | 82.4 | 0.06 | 5 | 5 | False | 27 | 19-3-5 | 6 | -0.435 |
| Mlle de Saint-Loup | 1830 ± 251 | 1579.8 | 125.3 | 0.06 | 6 | 13 | True | 7 | 7-0-0 | 2 | +1.473 |
| Mme Elstir | 1823 ± 272 | 1550.8 | 136.1 | 0.06 | 7 | 15 | True | 7 | 7-0-0 | 1 | +0.544 |
| comte de Forcheville | 1642 ± 100 | 1542.5 | 49.9 | 0.0604 | 8 | 47 | False | 112 | 56-19-37 | 25 | -0.29 |
| Françoise | 1624 ± 82 | 1542.4 | 41.0 | 0.0602 | 9 | 16 | False | 217 | 99-50-68 | 82 | -0.267 |
| Léa | 1724 ± 195 | 1529.4 | 97.3 | 0.06 | 10 | 17 | False | 14 | 8-0-6 | 4 | -0.7 |
| Bergotte | 1621 ± 93 | 1528.5 | 46.4 | 0.06 | 11 | 1 | False | 129 | 51-31-47 | 36 | -0.094 |
| le peintre | 1680 ± 160 | 1519.3 | 80.2 | 0.06 | 12 | 27 | False | 42 | 16-4-22 | 8 | -0.296 |
| Eulalie | 1747 ± 228 | 1518.9 | 114.2 | 0.06 | 13 | 11 | True | 16 | 11-3-2 | 7 | -0.168 |
| Mlle de Stermaria | 1785 ± 274 | 1511.3 | 136.8 | 0.0601 | 14 | 25 | True | 10 | 6-3-1 | 5 | -0.775 |
| Marie | 1778 ± 272 | 1505.6 | 136.0 | 0.06 | 15 | 22 | True | 7 | 6-1-0 | 1 | -0.24 |
| Morel | 1585 ± 79 | 1505.4 | 39.6 | 0.0604 | 16 | 8 | False | 152 | 48-51-53 | 32 | -0.876 |
| le grand-père du narrateur | 1649 ± 146 | 1502.4 | 73.1 | 0.06 | 17 | 10 | False | 63 | 26-7-30 | 16 | -0.612 |
| Mme Verdurin | 1565 ± 68 | 1497.4 | 33.8 | 0.0598 | 18 | 14 | False | 311 | 100-98-113 | 82 | -0.738 |
| Mme de Grouchy | 1873 ± 376 | 1496.3 | 188.2 | 0.06 | 19 | 28 | True | 4 | 4-0-0 | 1 | -0.06 |
| l'amie de Mlle Vinteuil | 1628 ± 134 | 1493.8 | 67.0 | 0.06 | 20 | 19 | False | 44 | 17-6-21 | 12 | -0.346 |
| Rachel | 1570 ± 77 | 1493.4 | 38.4 | 0.0607 | 21 | 9 | False | 146 | 52-49-45 | 43 | -0.939 |
| Aimé | 1593 ± 100 | 1492.7 | 50.2 | 0.06 | 22 | 31 | False | 79 | 27-13-39 | 18 | -0.472 |
| Mme Cottard | 1658 ± 166 | 1491.6 | 83.2 | 0.06 | 23 | 18 | False | 33 | 17-7-9 | 11 | -0.383 |
| M. Verdurin | 1584 ± 95 | 1488.9 | 47.4 | 0.06 | 24 | 26 | False | 110 | 36-25-49 | 27 | -0.64 |
| Rémi | 1711 ± 223 | 1487.9 | 111.4 | 0.06 | 25 | 29 | True | 17 | 5-0-12 | 3 | -0.533 |
| prince de Guermantes | 1578 ± 91 | 1486.2 | 45.6 | 0.0605 | 26 | 7 | False | 124 | 41-27-56 | 22 | -0.815 |
| la grand-mère | 1570 ± 84 | 1485.8 | 42.1 | 0.0604 | 27 | 40 | False | 225 | 94-67-64 | 80 | -0.29 |
| Mme de Chaussepierre | 1821 ± 339 | 1482.0 | 169.5 | 0.06 | 28 | 34 | True | 4 | 4-0-0 | 1 | +2.32 |
| Norpois | 1568 ± 89 | 1478.7 | 44.6 | 0.0599 | 29 | 24 | False | 180 | 79-53-48 | 63 | -0.592 |
| la mère du narrateur | 1567 ± 89 | 1477.8 | 44.7 | 0.0599 | 30 | 169 | False | 144 | 55-36-53 | 40 | -0.421 |
| Jupien | 1576 ± 102 | 1474.1 | 51.0 | 0.06 | 31 | 50 | False | 68 | 23-13-32 | 18 | +0.031 |
| Elstir | 1564 ± 91 | 1473.4 | 45.3 | 0.0601 | 32 | 12 | False | 106 | 39-31-36 | 29 | +0.034 |
| Mlle Vinteuil | 1579 ± 109 | 1469.9 | 54.6 | 0.06 | 33 | 63 | False | 71 | 21-15-35 | 15 | -0.665 |
| le pianiste | 1740 ± 271 | 1468.8 | 135.7 | 0.06 | 34 | 41 | True | 10 | 5-1-4 | 3 | +0.699 |
| duchesse de Guermantes | 1524 ± 59 | 1465.1 | 29.5 | 0.0669 | 35 | 281 | False | 662 | 331-180-151 | 199 | -0.076 |
| le père du narrateur | 1584 ± 120 | 1464.5 | 60.0 | 0.0599 | 36 | 20 | False | 90 | 34-21-35 | 24 | -0.68 |
| marquis de Bréauté | 1549 ± 93 | 1455.9 | 46.6 | 0.0599 | 37 | 45 | False | 101 | 26-22-53 | 19 | -0.931 |
| Odette | 1521 ± 66 | 1454.8 | 33.0 | 0.061 | 38 | 252 | False | 462 | 164-168-130 | 142 | -0.625 |
| Maeterlinck | 1775 ± 322 | 1452.7 | 161.2 | 0.06 | 39 | 32 | True | 5 | 4-0-1 | 1 | -0.8 |
| prince de Saxe | 1822 ± 370 | 1452.0 | 185.2 | 0.06 | 40 | 39 | True | 3 | 3-0-0 | 1 | +0.975 |
| Bloch | 1517 ± 68 | 1448.8 | 33.9 | 0.0611 | 41 | 23 | False | 270 | 78-112-80 | 71 | -1.407 |
| colonel Picquart | 1795 ± 349 | 1445.7 | 174.5 | 0.06 | 42 | 42 | True | 4 | 4-0-0 | 1 | +1.725 |
| baron de Charlus | 1502 ± 60 | 1441.9 | 30.0 | 0.0621 | 43 | 279 | False | 485 | 193-159-133 | 119 | -0.723 |
| Victurnien | 1707 ± 268 | 1438.7 | 134.1 | 0.06 | 44 | 30 | True | 8 | 4-0-4 | 2 | +0.286 |
| princesse de Guermantes | 1529 ± 91 | 1437.8 | 45.4 | 0.0602 | 45 | 263 | False | 113 | 42-30-41 | 25 | -0.252 |
| Mme de Charlus | 1629 ± 192 | 1436.9 | 96.2 | 0.06 | 46 | 36 | False | 15 | 5-1-9 | 2 | -0.8 |
| Robert de Saint-Loup | 1496 ± 60 | 1435.6 | 30.1 | 0.0614 | 47 | 241 | False | 508 | 160-212-136 | 168 | -0.548 |
| Albertine | 1503 ± 68 | 1435.0 | 34.1 | 0.0605 | 48 | 284 | False | 387 | 149-156-82 | 146 | -0.778 |
| Mme Goupil | 1609 ± 180 | 1429.4 | 89.9 | 0.06 | 49 | 37 | False | 17 | 5-1-11 | 2 | -0.8 |
| le narrateur | 1491 ± 61 | 1429.4 | 30.6 | 0.079 | 50 | 144 | False | 1093 | 403-491-199 | 316 | -0.718 |
| Mme de Surgis | 1556 ± 130 | 1426.1 | 65.2 | 0.06 | 51 | 44 | False | 42 | 16-11-15 | 9 | -0.802 |
| Gilberte | 1486 ± 61 | 1425.0 | 30.6 | 0.0604 | 52 | 286 | False | 312 | 112-104-96 | 76 | -0.457 |
| Dreyfus | 1540 ± 123 | 1416.5 | 61.5 | 0.06 | 53 | 67 | False | 58 | 13-11-34 | 7 | -0.77 |
| grand-duc héritier de Luxembourg | 1666 ± 250 | 1416.1 | 125.1 | 0.06 | 54 | 33 | True | 9 | 4-1-4 | 2 | +0.509 |
| Swann | 1474 ± 61 | 1413.7 | 30.4 | 0.0644 | 55 | 251 | False | 667 | 212-305-150 | 202 | -0.817 |
| Mme de Villeparisis | 1499 ± 86 | 1413.0 | 42.9 | 0.0606 | 56 | 178 | False | 236 | 89-92-55 | 79 | -0.637 |
| Andrée | 1504 ± 91 | 1412.4 | 45.6 | 0.0599 | 57 | 220 | False | 114 | 37-42-35 | 31 | -0.712 |
| docteur Cottard | 1492 ± 81 | 1410.5 | 40.7 | 0.0601 | 58 | 141 | False | 194 | 46-64-84 | 43 | -0.865 |
| Brichot | 1488 ± 80 | 1408.3 | 40.1 | 0.0599 | 59 | 264 | False | 135 | 28-32-75 | 21 | -0.777 |
| marquis du Lau | 1715 ± 308 | 1407.4 | 153.9 | 0.06 | 60 | 48 | True | 5 | 4-1-0 | 2 | +1.462 |
| M. Vinteuil | 1525 ± 118 | 1407.3 | 59.0 | 0.06 | 61 | 208 | False | 61 | 18-19-24 | 15 | -0.422 |
| Gribelin | 1711 ± 311 | 1400.3 | 155.6 | 0.06 | 62 | 35 | True | 6 | 5-1-0 | 1 | +0.04 |
| M. Ski | 1565 ± 167 | 1398.0 | 83.7 | 0.06 | 63 | 84 | False | 21 | 4-1-16 | 2 | -0.4 |
| marquise de Saint-Euverte | 1499 ± 106 | 1392.2 | 53.2 | 0.0602 | 64 | 71 | False | 72 | 16-29-27 | 13 | -1.982 |
| M. d'Argencourt | 1513 ± 123 | 1390.6 | 61.4 | 0.06 | 65 | 193 | False | 56 | 20-18-18 | 14 | -1.2 |
| duc de Guermantes | 1456 ± 67 | 1389.9 | 33.3 | 0.0601 | 66 | 287 | False | 401 | 123-172-106 | 110 | -0.985 |
| Mme de Marsantes | 1478 ± 94 | 1384.7 | 46.9 | 0.0599 | 67 | 219 | False | 107 | 20-31-56 | 21 | -1.001 |
| général de Froberville | 1556 ± 172 | 1384.2 | 85.8 | 0.06 | 68 | 43 | False | 27 | 7-4-16 | 7 | -0.589 |
| Mme Sazerat | 1576 ± 193 | 1382.7 | 96.5 | 0.06 | 69 | 38 | False | 20 | 7-2-11 | 6 | -0.673 |
| Mme Bontemps | 1498 ± 115 | 1382.6 | 57.7 | 0.06 | 70 | 247 | False | 54 | 13-13-28 | 13 | -0.643 |
| duchesse de La Trémoïlle | 1804 ± 424 | 1380.2 | 212.0 | 0.06 | 71 | 46 | True | 3 | 3-0-0 | 1 | +0.665 |
| la marquise douairière de Cambremer | 1530 ± 152 | 1378.3 | 75.9 | 0.06 | 72 | 49 | False | 31 | 10-5-16 | 6 | +0.083 |
| princesse de Parme | 1471 ± 102 | 1369.6 | 50.9 | 0.06 | 73 | 21 | False | 130 | 37-62-31 | 38 | -0.658 |
| Legrandin | 1469 ± 101 | 1368.6 | 50.4 | 0.0601 | 74 | 218 | False | 83 | 16-27-40 | 24 | -1.178 |
| comtesse Molé | 1518 ± 151 | 1367.2 | 75.3 | 0.06 | 75 | 168 | False | 34 | 5-9-20 | 6 | -1.142 |
| Mme Leroi | 1573 ± 206 | 1366.9 | 103.2 | 0.06 | 76 | 52 | True | 13 | 8-5-0 | 5 | -1.092 |
| duc d'Aumale | 1741 ± 377 | 1364.4 | 188.3 | 0.06 | 77 | 77 | True | 4 | 3-1-0 | 2 | +0.637 |
| Émilie Daltier | 1721 ± 358 | 1363.0 | 179.2 | 0.06 | 78 | 72 | True | 3 | 2-0-1 | 1 | -0.4 |
| prince des Laumes | 1536 ± 176 | 1360.1 | 88.1 | 0.06 | 79 | 107 | False | 27 | 4-3-20 | 3 | -0.8 |
| Charcot | 1584 ± 225 | 1359.2 | 112.6 | 0.06 | 80 | 87 | True | 12 | 3-2-7 | 1 | -0.8 |
| M. Reinach | 1584 ± 225 | 1359.2 | 112.6 | 0.06 | 81 | 90 | True | 12 | 3-2-7 | 1 | -0.8 |
| M. Nissim Bernard | 1489 ± 130 | 1359.1 | 65.0 | 0.06 | 82 | 223 | False | 39 | 8-10-21 | 10 | -1.315 |
| duc de Chartres | 1567 ± 209 | 1358.5 | 104.3 | 0.06 | 83 | 51 | True | 14 | 2-0-12 | 1 | -0.8 |
| prince de Chimay | 1567 ± 209 | 1358.5 | 104.3 | 0.06 | 84 | 59 | True | 14 | 2-0-12 | 1 | -0.8 |
| M. d'Orsan | 1628 ± 270 | 1357.9 | 134.9 | 0.06 | 85 | 60 | True | 11 | 2-0-9 | 1 | -0.8 |
| Mlle d'Éporcheville | 1573 ± 224 | 1349.5 | 111.9 | 0.06 | 86 | 108 | True | 10 | 3-2-5 | 2 | -0.6 |
| Esther | 1546 ± 199 | 1346.5 | 99.6 | 0.06 | 87 | 105 | False | 14 | 3-2-9 | 2 | -1.0 |
| la Berma | 1458 ± 111 | 1346.5 | 55.6 | 0.0601 | 88 | 273 | False | 62 | 19-24-19 | 19 | -0.451 |
| marquis Maurice de Vaudémont | 1816 ± 470 | 1346.3 | 235.0 | 0.06 | 89 | 54 | True | 2 | 2-0-0 | 1 | +1.574 |
| le jeune marquis de Cambremer | 1546 ± 202 | 1344.8 | 100.8 | 0.06 | 90 | 136 | True | 12 | 2-1-9 | 1 | -1.2 |
| Bloch père | 1473 ± 130 | 1342.5 | 65.2 | 0.06 | 91 | 209 | False | 47 | 10-12-25 | 8 | -1.771 |
| Lady Rufus Israël | 1621 ± 281 | 1340.3 | 140.3 | 0.06 | 92 | 114 | True | 6 | 2-1-3 | 1 | -0.4 |
| M. de Chevregny | 1537 ± 198 | 1339.3 | 99.0 | 0.06 | 93 | 65 | False | 16 | 4-1-11 | 1 | -0.4 |
| M. de Crécy | 1537 ± 198 | 1339.3 | 99.0 | 0.06 | 94 | 70 | False | 16 | 4-1-11 | 1 | -0.4 |
| Mme Féré | 1537 ± 198 | 1339.3 | 99.0 | 0.06 | 95 | 74 | False | 16 | 4-1-11 | 1 | -0.4 |
| les La Trémoïlle | 1666 ± 329 | 1337.3 | 164.4 | 0.06 | 96 | 61 | True | 7 | 2-0-5 | 1 | -0.8 |
| prince d’Agrigente | 1537 ± 202 | 1335.2 | 101.0 | 0.06 | 97 | 120 | True | 15 | 3-2-10 | 2 | -0.8 |
| princesse de Luxembourg | 1498 ± 166 | 1332.3 | 83.0 | 0.06 | 98 | 160 | False | 25 | 6-7-12 | 6 | -0.773 |
| Herbinger | 1740 ± 409 | 1330.8 | 204.4 | 0.06 | 99 | 68 | True | 3 | 2-0-1 | 1 | -0.8 |
| la duchesse d'Alençon | 1645 ± 314 | 1330.4 | 157.2 | 0.06 | 100 | 53 | True | 6 | 3-1-2 | 1 | -0.8 |
| Mme de Stermaria | 1637 ± 309 | 1328.1 | 154.6 | 0.06 | 101 | 88 | True | 5 | 2-1-2 | 1 | -0.8 |
| Létourville | 1681 ± 353 | 1327.5 | 176.6 | 0.06 | 102 | 82 | True | 3 | 2-0-1 | 1 | -0.8 |
| Bismarck | 1680 ± 354 | 1326.5 | 176.9 | 0.06 | 103 | 81 | True | 4 | 2-0-2 | 1 | +0.343 |
| le directeur | 1463 ± 139 | 1324.3 | 69.3 | 0.06 | 104 | 267 | False | 39 | 10-17-12 | 11 | -0.63 |
| duc de Sidonia | 1780 ± 456 | 1323.6 | 228.2 | 0.06 | 105 | 58 | True | 2 | 2-0-0 | 1 | -0.88 |
| Dechambre | 1689 ± 366 | 1323.3 | 183.0 | 0.06 | 106 | 76 | True | 3 | 2-0-1 | 1 | -0.96 |
| le commandant Duroc | 1731 ± 412 | 1318.2 | 206.2 | 0.06 | 107 | 80 | True | 2 | 2-0-0 | 1 | +0.328 |
| Mme Timoléon d'Amoncourt | 1562 ± 244 | 1317.9 | 121.9 | 0.06 | 108 | 97 | True | 9 | 2-1-6 | 1 | -0.4 |
| Marie-Aynard | 1624 ± 308 | 1316.0 | 153.9 | 0.06 | 109 | 56 | True | 7 | 2-0-5 | 1 | -0.8 |
| Victurnienne | 1624 ± 308 | 1316.0 | 153.9 | 0.06 | 110 | 55 | True | 7 | 2-0-5 | 1 | -0.8 |
| Goncourt | 1488 ± 174 | 1313.0 | 87.2 | 0.06 | 111 | 234 | False | 16 | 2-3-11 | 2 | -0.8 |
| Mme de Montmorency | 1532 ± 222 | 1310.1 | 110.8 | 0.06 | 112 | 152 | True | 11 | 2-1-8 | 1 | -0.8 |
| Mme de Rochechouart | 1532 ± 222 | 1310.1 | 110.8 | 0.06 | 113 | 153 | True | 11 | 2-1-8 | 1 | -0.8 |
| Lady Israels | 1760 ± 455 | 1304.6 | 227.5 | 0.06 | 114 | 85 | True | 2 | 2-0-0 | 1 | 0.0 |
| Mlle Bloch | 1729 ± 425 | 1304.3 | 212.4 | 0.06 | 115 | 64 | True | 2 | 2-0-0 | 1 | +1.142 |
| Sarah Bernhardt | 1590 ± 286 | 1303.7 | 143.2 | 0.06 | 116 | 102 | True | 7 | 2-0-5 | 1 | -0.8 |
| le jeune prince de Foix | 1590 ± 286 | 1303.7 | 143.2 | 0.06 | 117 | 95 | True | 7 | 2-0-5 | 1 | -0.8 |
| vicomte de Courvoisier | 1590 ± 286 | 1303.7 | 143.2 | 0.06 | 118 | 104 | True | 7 | 2-0-5 | 1 | -0.8 |
| Duroc | 1753 ± 454 | 1298.3 | 227.2 | 0.06 | 119 | 79 | True | 2 | 2-0-0 | 1 | +1.273 |
| princesse d'Épinay | 1539 ± 242 | 1297.9 | 120.8 | 0.06 | 120 | 96 | True | 12 | 4-3-5 | 3 | -0.533 |
| cousine Poictiers | 1615 ± 318 | 1296.8 | 159.1 | 0.06 | 121 | 99 | True | 5 | 2-1-2 | 1 | -0.4 |
| duc de Poictiers | 1615 ± 318 | 1296.8 | 159.1 | 0.06 | 122 | 100 | True | 5 | 2-1-2 | 1 | -0.4 |
| M. de Beauserfeuil | 1573 ± 278 | 1294.8 | 138.9 | 0.06 | 123 | 66 | True | 7 | 2-1-4 | 1 | -0.8 |
| Mme de Sévigné | 1483 ± 189 | 1294.6 | 94.4 | 0.06 | 124 | 91 | False | 25 | 7-5-13 | 4 | -0.019 |
| Manet | 1607 ± 317 | 1289.8 | 158.6 | 0.06 | 125 | 75 | True | 5 | 1-0-4 | 1 | -0.8 |
| Léonor de Cambremer | 1492 ± 203 | 1289.2 | 101.5 | 0.06 | 126 | 189 | True | 12 | 1-1-10 | 1 | -0.8 |
| Bibi | 1711 ± 424 | 1286.2 | 212.2 | 0.06 | 127 | 57 | True | 2 | 2-0-0 | 1 | +0.16 |
| Mme Legrandin mère | 1564 ± 279 | 1285.2 | 139.6 | 0.06 | 128 | 78 | True | 8 | 2-0-6 | 1 | -0.8 |
| Victoire | 1564 ± 279 | 1285.2 | 139.6 | 0.06 | 129 | 86 | True | 8 | 2-0-6 | 1 | -0.8 |
| le baron Bréau-Chenut | 1593 ± 310 | 1283.2 | 154.8 | 0.06 | 130 | 93 | True | 7 | 3-1-3 | 1 | -0.8 |
| le vieux père Chenut | 1593 ± 310 | 1283.2 | 154.8 | 0.06 | 131 | 98 | True | 7 | 3-1-3 | 1 | -0.8 |
| l'abbé Poiré | 1507 ± 225 | 1282.8 | 112.3 | 0.06 | 132 | 130 | True | 10 | 1-2-7 | 1 | -0.8 |
| Élisabeth | 1566 ± 284 | 1281.9 | 142.2 | 0.06 | 133 | 118 | True | 6 | 2-1-3 | 1 | -1.2 |
| Thibaud | 1527 ± 245 | 1281.6 | 122.5 | 0.06 | 134 | 165 | True | 8 | 2-2-4 | 1 | -0.8 |
| Balzac | 1465 ± 185 | 1280.6 | 92.3 | 0.06 | 135 | 228 | False | 18 | 2-4-12 | 2 | -0.8 |
| Mme de Cambremer | 1367 ± 94 | 1273.1 | 46.9 | 0.0601 | 136 | 282 | False | 112 | 13-54-45 | 20 | -1.561 |
| Mme Putbus | 1524 ± 251 | 1272.8 | 125.7 | 0.06 | 137 | 128 | True | 8 | 1-1-6 | 1 | -0.8 |
| duchesse de Létourville | 1564 ± 292 | 1272.2 | 146.0 | 0.06 | 138 | 129 | True | 5 | 2-1-2 | 1 | -0.8 |
| M. de Bornier | 1616 ± 346 | 1270.4 | 173.0 | 0.06 | 139 | 69 | True | 5 | 3-1-1 | 1 | -1.2 |
| M. de Chateaubriand | 1555 ± 285 | 1269.3 | 142.7 | 0.06 | 140 | 210 | True | 11 | 1-3-7 | 2 | -1.773 |
| le petit Cambremer | 1465 ± 197 | 1268.3 | 98.5 | 0.06 | 141 | 216 | False | 14 | 1-3-10 | 1 | -0.8 |
| princesse de Silistrie | 1465 ± 197 | 1268.3 | 98.5 | 0.06 | 142 | 214 | False | 14 | 1-3-10 | 1 | -0.8 |
| M. de Goncourt | 1504 ± 237 | 1266.9 | 118.3 | 0.06 | 143 | 179 | True | 8 | 1-1-6 | 1 | -1.2 |
| Flora | 1602 ± 336 | 1265.5 | 168.2 | 0.06 | 144 | 62 | True | 8 | 3-1-4 | 1 | -0.8 |
| M. Arthur Meyer | 1535 ± 270 | 1264.9 | 134.8 | 0.06 | 145 | 163 | True | 6 | 2-2-2 | 1 | -0.8 |
| M. Vibert | 1638 ± 373 | 1264.6 | 186.6 | 0.06 | 146 | 83 | True | 3 | 1-0-2 | 1 | -0.4 |
| Lady Israël | 1587 ± 323 | 1263.8 | 161.5 | 0.06 | 147 | 101 | True | 5 | 2-1-2 | 1 | -0.4 |
| monsieur Vallenères | 1689 ± 426 | 1262.7 | 213.1 | 0.06 | 148 | 73 | True | 2 | 2-0-0 | 1 | -0.8 |
| prince de Foix | 1477 ± 214 | 1262.6 | 107.0 | 0.06 | 149 | 159 | True | 14 | 4-4-6 | 3 | -0.893 |
| comte de Paris | 1511 ± 250 | 1261.3 | 125.0 | 0.06 | 150 | 200 | True | 10 | 3-4-3 | 3 | -0.667 |
| M. Carnot | 1509 ± 252 | 1257.1 | 126.0 | 0.06 | 151 | 132 | True | 9 | 1-1-7 | 1 | -0.8 |
| Mme Carnot | 1509 ± 252 | 1257.1 | 126.0 | 0.06 | 152 | 133 | True | 9 | 1-1-7 | 1 | -0.8 |
| Sir Rufus Israël | 1536 ± 282 | 1254.6 | 140.9 | 0.06 | 153 | 117 | True | 7 | 3-1-3 | 1 | -0.8 |
| Gisèle | 1505 ± 253 | 1252.1 | 126.6 | 0.06 | 154 | 186 | True | 14 | 3-6-5 | 5 | -1.768 |
| M. de Vaugoubert | 1410 ± 160 | 1250.1 | 80.2 | 0.06 | 155 | 250 | False | 35 | 6-12-17 | 9 | -1.131 |
| Mme de Vaugoubert | 1508 ± 262 | 1246.5 | 130.8 | 0.06 | 156 | 226 | True | 9 | 1-3-5 | 2 | -1.648 |
| général de Monserfeuil | 1442 ± 196 | 1246.1 | 97.8 | 0.06 | 157 | 235 | False | 18 | 5-8-5 | 4 | -1.289 |
| comtesse douairière d'Argencourt | 1490 ± 244 | 1246.0 | 121.9 | 0.06 | 158 | 188 | True | 10 | 1-2-7 | 1 | -0.8 |
| duchesse de Gallardon douairière | 1490 ± 244 | 1246.0 | 121.9 | 0.06 | 159 | 185 | True | 10 | 1-2-7 | 1 | -0.8 |
| marquis de Fierbois | 1490 ± 244 | 1246.0 | 121.9 | 0.06 | 160 | 202 | True | 10 | 1-2-7 | 1 | -0.8 |
| Rosemonde | 1436 ± 190 | 1245.7 | 95.0 | 0.06 | 161 | 190 | False | 20 | 5-7-8 | 4 | -0.7 |
| Dostoïevski | 1515 ± 273 | 1241.8 | 136.7 | 0.06 | 162 | 171 | True | 6 | 1-1-4 | 1 | -0.8 |
| jeune blonde de Rivebelle | 1573 ± 333 | 1239.9 | 166.5 | 0.06 | 163 | 103 | True | 6 | 2-1-3 | 2 | -0.4 |
| Mlle de l’Orgeville | 1590 ± 352 | 1237.4 | 176.1 | 0.06 | 164 | 131 | True | 3 | 1-0-2 | 1 | -0.8 |
| prince de Sagan | 1518 ± 281 | 1237.0 | 140.7 | 0.06 | 165 | 123 | True | 7 | 1-1-5 | 1 | -0.8 |
| tante Léonie | 1415 ± 180 | 1235.3 | 89.9 | 0.06 | 166 | 260 | False | 38 | 11-22-5 | 22 | -0.717 |
| Arnulphe | 1598 ± 363 | 1234.7 | 181.6 | 0.06 | 167 | 94 | True | 4 | 1-0-3 | 1 | -0.4 |
| Barrès | 1479 ± 247 | 1232.2 | 123.5 | 0.06 | 168 | 137 | True | 9 | 1-1-7 | 1 | -0.8 |
| Clémenceau | 1479 ± 247 | 1232.2 | 123.5 | 0.06 | 169 | 139 | True | 9 | 1-1-7 | 1 | -0.8 |
| d’Orgeville | 1501 ± 269 | 1232.2 | 134.5 | 0.06 | 170 | 125 | True | 7 | 1-1-5 | 1 | -0.8 |
| d'Orléans | 1581 ± 350 | 1231.1 | 174.9 | 0.06 | 171 | 110 | True | 5 | 2-1-2 | 1 | -0.8 |
| Sainte-Beuve | 1490 ± 267 | 1223.6 | 133.4 | 0.06 | 172 | 176 | True | 7 | 1-2-4 | 1 | -0.8 |
| le grand-duc Wladimir | 1591 ± 369 | 1221.6 | 184.6 | 0.06 | 173 | 111 | True | 3 | 2-1-0 | 1 | -0.4 |
| Coquelin | 1542 ± 321 | 1221.1 | 160.3 | 0.06 | 174 | 164 | True | 5 | 1-1-3 | 1 | -0.8 |
| oncle Adolphe | 1452 ± 232 | 1220.7 | 115.8 | 0.0601 | 175 | 257 | True | 20 | 4-11-5 | 6 | -1.52 |
| le marquis de Ganançay | 1568 ± 349 | 1219.3 | 174.6 | 0.06 | 176 | 89 | True | 6 | 3-1-2 | 1 | -0.8 |
| le marquis de Palancy | 1568 ± 349 | 1219.3 | 174.6 | 0.06 | 177 | 92 | True | 6 | 3-1-2 | 1 | -0.8 |
| Napoléon III | 1525 ± 307 | 1218.0 | 153.6 | 0.06 | 178 | 201 | True | 8 | 1-2-5 | 1 | -0.8 |
| Céline | 1466 ± 250 | 1215.8 | 124.8 | 0.06 | 179 | 211 | True | 16 | 3-6-7 | 2 | -1.14 |
| Mme d'Arpajon | 1357 ± 150 | 1207.6 | 74.9 | 0.06 | 180 | 276 | False | 37 | 6-22-9 | 8 | -1.72 |
| Liszt | 1510 ± 303 | 1206.6 | 151.5 | 0.06 | 181 | 134 | True | 6 | 2-1-3 | 1 | -0.8 |
| Mme Ristori | 1510 ± 303 | 1206.6 | 151.5 | 0.06 | 182 | 135 | True | 6 | 2-1-3 | 1 | -0.8 |
| comtesse de Monteriender | 1550 ± 352 | 1198.4 | 176.1 | 0.06 | 183 | 146 | True | 4 | 1-1-2 | 1 | 0.0 |
| M. de Marsantes | 1498 ± 301 | 1197.3 | 150.4 | 0.06 | 184 | 154 | True | 7 | 1-1-5 | 2 | -0.312 |
| Mme de Sagan | 1565 ± 369 | 1196.0 | 184.6 | 0.06 | 185 | 116 | True | 3 | 1-0-2 | 1 | -0.4 |
| M. de La Rochefoucauld | 1520 ± 327 | 1193.1 | 163.6 | 0.06 | 186 | 119 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de La Rochefoucauld | 1520 ± 327 | 1193.1 | 163.6 | 0.06 | 187 | 106 | True | 6 | 2-1-3 | 1 | -0.8 |
| duchesse de Praslin | 1520 ± 327 | 1193.1 | 163.6 | 0.06 | 188 | 109 | True | 6 | 2-1-3 | 1 | -0.8 |
| elle | 1692 ± 503 | 1188.9 | 251.3 | 0.06 | 189 | 124 | True | 1 | 1-0-0 | 1 | -0.12 |
| Mme Trombert | 1533 ± 346 | 1187.3 | 173.0 | 0.06 | 190 | 175 | True | 4 | 1-1-2 | 1 | -0.4 |
| prince Von | 1444 ± 258 | 1185.5 | 129.0 | 0.06 | 191 | 170 | True | 8 | 2-3-3 | 3 | -1.435 |
| princesse Sherbatoff | 1357 ± 174 | 1183.3 | 86.8 | 0.06 | 192 | 277 | False | 19 | 4-12-3 | 5 | -0.701 |
| Poullein | 1646 ± 483 | 1162.3 | 241.6 | 0.06 | 193 | 138 | True | 2 | 1-1-0 | 2 | -0.54 |
| princesse Mathilde | 1479 ± 318 | 1161.2 | 158.9 | 0.06 | 194 | 182 | True | 7 | 2-3-2 | 2 | -0.6 |
| docteur Dieulafoy | 1659 ± 502 | 1156.4 | 251.1 | 0.06 | 195 | 122 | True | 1 | 1-0-0 | 1 | +3.605 |
| D'Annunzio | 1477 ± 321 | 1156.3 | 160.5 | 0.06 | 196 | 177 | True | 5 | 1-2-2 | 1 | -0.4 |
| marquis de Cambremer | 1288 ± 136 | 1152.7 | 67.8 | 0.0601 | 197 | 285 | False | 45 | 6-24-15 | 6 | -1.12 |
| M. d'Herweck | 1452 ± 300 | 1152.4 | 150.0 | 0.06 | 198 | 166 | True | 5 | 2-3-0 | 2 | -2.28 |
| le roi Théodose | 1455 ± 306 | 1148.9 | 153.1 | 0.06 | 199 | 203 | True | 8 | 2-4-2 | 3 | -0.189 |
| M. Swann, le père | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 200 | 142 | True | 7 | 1-1-5 | 1 | -0.8 |
| le comte de Paris | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 201 | 155 | True | 7 | 1-1-5 | 1 | -0.8 |
| le prince de Galles | 1500 ± 353 | 1147.0 | 176.5 | 0.06 | 202 | 156 | True | 7 | 1-1-5 | 1 | -0.8 |
| la jeune ouvriere | 1589 ± 445 | 1144.0 | 222.3 | 0.06 | 203 | 126 | True | 2 | 0-0-2 | 1 | -0.4 |
| M. Molé | 1438 ± 300 | 1137.8 | 149.9 | 0.06 | 204 | 198 | True | 8 | 1-2-5 | 1 | -0.8 |
| M. de Bouillon | 1438 ± 300 | 1137.8 | 149.9 | 0.06 | 205 | 194 | True | 8 | 1-2-5 | 1 | -0.8 |
| Musset | 1438 ± 300 | 1137.8 | 149.9 | 0.06 | 206 | 184 | True | 8 | 1-2-5 | 1 | -0.8 |
| Victor Hugo | 1438 ± 300 | 1137.8 | 149.9 | 0.06 | 207 | 183 | True | 8 | 1-2-5 | 1 | -0.8 |
| L’excellent écrivain G… | 1479 ± 345 | 1133.6 | 172.7 | 0.06 | 208 | 127 | True | 4 | 1-1-2 | 1 | -0.8 |
| le prince Von | 1374 ± 242 | 1131.8 | 121.2 | 0.06 | 209 | 221 | True | 10 | 2-5-3 | 2 | -1.225 |
| M. de Grouchy | 1364 ± 253 | 1111.2 | 126.4 | 0.06 | 210 | 243 | True | 10 | 2-7-1 | 4 | -0.911 |
| Mme de Franquetot | 1280 ± 170 | 1109.7 | 85.0 | 0.0601 | 211 | 280 | False | 23 | 4-13-6 | 3 | -0.837 |
| Théodore | 1624 ± 514 | 1109.6 | 257.0 | 0.06 | 212 | 115 | True | 2 | 1-0-1 | 1 | +1.818 |
| Mme d'Heudicourt | 1306 ± 199 | 1107.3 | 99.3 | 0.06 | 213 | 268 | False | 18 | 3-12-3 | 5 | -1.469 |
| M. de Miribel | 1479 ± 372 | 1107.0 | 186.1 | 0.06 | 214 | 167 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Henry | 1479 ± 372 | 1107.0 | 186.1 | 0.06 | 215 | 174 | True | 4 | 1-1-2 | 1 | -0.8 |
| le lieutenant-colonel Picquart | 1479 ± 372 | 1107.0 | 186.1 | 0.06 | 216 | 173 | True | 4 | 1-1-2 | 1 | -0.8 |
| M. de Stermaria | 1360 ± 255 | 1105.0 | 127.7 | 0.06 | 217 | 262 | True | 10 | 2-6-2 | 4 | -1.359 |
| duc de Châtellerault | 1348 ± 253 | 1094.5 | 126.7 | 0.06 | 218 | 242 | True | 10 | 1-6-3 | 5 | -1.313 |
| Saniette | 1247 ± 154 | 1092.9 | 76.9 | 0.0601 | 219 | 288 | False | 35 | 2-25-8 | 9 | -2.784 |
| Octave | 1507 ± 423 | 1084.5 | 211.3 | 0.06 | 220 | 162 | True | 4 | 2-2-0 | 2 | -0.334 |
| le capitaine | 1494 ± 412 | 1081.8 | 206.2 | 0.06 | 221 | 172 | True | 2 | 1-1-0 | 1 | +0.04 |
| M. de Courgivaux | 1662 ± 581 | 1081.7 | 290.3 | 0.06 | 222 | 112 | True | 1 | 1-0-0 | 1 | +2.018 |
| capitaine de Borodino | 1286 ± 207 | 1079.8 | 103.4 | 0.06 | 223 | 278 | True | 14 | 2-10-2 | 5 | -1.731 |
| Théodose Cadet | 1450 ± 372 | 1078.0 | 185.8 | 0.06 | 224 | 204 | True | 3 | 1-2-0 | 1 | -1.794 |
| Mme de Villebon | 1662 ± 585 | 1077.6 | 292.4 | 0.06 | 225 | 113 | True | 1 | 1-0-0 | 1 | -1.0 |
| baron de Guermantes | 1662 ± 585 | 1077.2 | 292.6 | 0.06 | 226 | 121 | True | 1 | 1-0-0 | 1 | -0.4 |
| Beauserfeuil | 1448 ± 372 | 1076.6 | 185.8 | 0.06 | 227 | 205 | True | 3 | 1-2-0 | 1 | -0.84 |
| docteur Percepied | 1500 ± 426 | 1074.1 | 212.9 | 0.06 | 228 | 151 | True | 4 | 1-1-2 | 1 | -0.8 |
| Madame Elstir | 1405 ± 333 | 1072.2 | 166.5 | 0.06 | 229 | 199 | True | 6 | 1-2-3 | 1 | -0.8 |
| les demoiselles d’Ambresac | 1405 ± 333 | 1072.2 | 166.5 | 0.06 | 230 | 192 | True | 6 | 1-2-3 | 1 | -0.8 |
| le bâtonnier | 1471 ± 405 | 1065.3 | 202.7 | 0.06 | 231 | 143 | True | 3 | 1-1-1 | 1 | -0.4 |
| M. Grevy | 1481 ± 424 | 1056.6 | 212.0 | 0.06 | 232 | 157 | True | 3 | 1-1-1 | 1 | -0.4 |
| Cartier | 1395 ± 339 | 1056.1 | 169.5 | 0.06 | 233 | 222 | True | 4 | 1-3-0 | 1 | -1.635 |
| marquise de Gallardon | 1296 ± 245 | 1051.1 | 122.6 | 0.06 | 234 | 274 | True | 19 | 1-12-6 | 7 | -2.104 |
| prince d'Agrigente | 1493 ± 456 | 1036.5 | 228.2 | 0.06 | 235 | 161 | True | 2 | 1-1-0 | 2 | -0.237 |
| M. Barrère | 1531 ± 501 | 1029.5 | 250.6 | 0.06 | 236 | 145 | True | 1 | 0-0-1 | 1 | -1.352 |
| Mme de Souvré | 1287 ± 268 | 1019.5 | 133.8 | 0.06 | 237 | 266 | True | 11 | 2-9-0 | 2 | -1.315 |
| marquise de Citri | 1426 ± 414 | 1011.9 | 206.9 | 0.06 | 238 | 191 | True | 2 | 0-1-1 | 1 | -2.79 |
| Antoine | 1418 ± 406 | 1011.8 | 203.1 | 0.06 | 239 | 213 | True | 3 | 0-2-1 | 1 | -0.8 |
| professeur E… | 1373 ± 370 | 1002.9 | 184.9 | 0.06 | 240 | 230 | True | 4 | 1-3-0 | 2 | -1.484 |
| le professeur E… | 1437 ± 456 | 980.6 | 228.2 | 0.06 | 241 | 181 | True | 2 | 0-1-1 | 1 | -2.884 |
| Maurice | 1264 ± 286 | 978.1 | 143.2 | 0.06 | 242 | 265 | True | 7 | 1-6-0 | 1 | -1.68 |
| colonel de Froberville | 1184 ± 209 | 975.1 | 104.3 | 0.06 | 243 | 283 | True | 14 | 0-13-1 | 1 | -4.448 |
| Vigny | 1449 ± 484 | 965.5 | 242.0 | 0.06 | 244 | 158 | True | 2 | 1-1-0 | 1 | -1.562 |
| les Courvoisier | 1303 ± 348 | 955.0 | 174.1 | 0.06 | 245 | 238 | True | 5 | 1-4-0 | 1 | -1.456 |
| M. Bontemps | 1238 ± 292 | 946.1 | 145.8 | 0.06 | 246 | 270 | True | 9 | 1-8-0 | 2 | -0.123 |
| l'ambassadrice de Turquie | 1264 ± 327 | 937.1 | 163.5 | 0.06 | 247 | 256 | True | 4 | 0-4-0 | 1 | -2.856 |
| Alix | 1182 ± 251 | 931.6 | 125.5 | 0.06 | 248 | 275 | True | 9 | 0-8-1 | 3 | -2.814 |
| Prince Henri d'Orléans | 1349 ± 427 | 922.1 | 213.6 | 0.06 | 249 | 206 | True | 2 | 0-1-1 | 1 | -1.401 |
| la Charité de Giotto | 1500 ± 587 | 913.0 | 293.5 | 0.06 | 250 | 140 | True | 1 | 0-0-1 | 1 | -2.635 |
| Mme de Morienval | 1260 ± 349 | 910.8 | 174.6 | 0.06 | 251 | 248 | True | 6 | 1-4-1 | 1 | -1.44 |
| duchesse de Luxembourg | 1260 ± 349 | 910.8 | 174.6 | 0.06 | 252 | 246 | True | 6 | 1-4-1 | 1 | -1.44 |
| le prince de Faffenheim | 1202 ± 305 | 896.6 | 152.6 | 0.06 | 253 | 261 | True | 5 | 0-5-0 | 1 | -3.937 |
| prince de Faffenheim | 1258 ± 362 | 895.8 | 181.0 | 0.06 | 254 | 232 | True | 3 | 0-3-0 | 2 | -1.091 |
| le prince von *** | 1306 ± 416 | 890.5 | 207.8 | 0.06 | 255 | 217 | True | 2 | 0-2-0 | 1 | -2.106 |
| M. de Luxembourg | 1322 ± 433 | 889.2 | 216.3 | 0.06 | 256 | 231 | True | 2 | 0-2-0 | 1 | -1.024 |
| le diplomate belge | 1301 ± 420 | 881.0 | 210.1 | 0.06 | 257 | 225 | True | 2 | 0-2-0 | 1 | -1.255 |
| Picquart | 1150 ± 278 | 872.0 | 139.1 | 0.06 | 258 | 272 | True | 8 | 0-8-0 | 2 | -1.982 |
| Mme Iéna | 1185 ± 315 | 870.0 | 157.6 | 0.06 | 259 | 259 | True | 5 | 0-5-0 | 1 | -2.226 |
| la cousine d'Oriane | 1232 ± 370 | 861.9 | 184.8 | 0.06 | 260 | 245 | True | 3 | 0-3-0 | 1 | -1.615 |
| prince Foggi | 1353 ± 501 | 851.8 | 250.6 | 0.06 | 261 | 187 | True | 1 | 0-1-0 | 1 | -1.504 |
| vicomtesse d'Égremont | 1253 ± 402 | 850.5 | 201.0 | 0.06 | 262 | 240 | True | 3 | 0-3-0 | 1 | -3.7 |
| l'historien de la Fronde | 1239 ± 392 | 847.0 | 195.8 | 0.06 | 263 | 237 | True | 3 | 0-3-0 | 1 | -1.163 |
| M. de Vigny | 1135 ± 300 | 835.5 | 149.9 | 0.06 | 264 | 271 | True | 8 | 0-8-0 | 1 | -2.681 |
| prince de Léon | 1290 ± 455 | 835.4 | 227.6 | 0.06 | 265 | 233 | True | 2 | 0-2-0 | 1 | -0.4 |
| Marie Gineste | 1284 ± 453 | 830.6 | 226.6 | 0.06 | 266 | 215 | True | 2 | 0-2-0 | 1 | -0.4 |
| l'empereur | 1194 ± 365 | 828.3 | 182.6 | 0.06 | 267 | 253 | True | 4 | 0-4-0 | 1 | -2.639 |
| Mme de Simiane | 1246 ± 428 | 818.3 | 213.9 | 0.06 | 268 | 244 | True | 3 | 0-3-0 | 1 | -1.296 |
| Mme de Varambon | 1172 ± 355 | 816.8 | 177.5 | 0.06 | 269 | 254 | True | 4 | 0-4-0 | 2 | -2.31 |
| Mme Blandais | 1182 ± 368 | 814.2 | 183.9 | 0.06 | 270 | 255 | True | 4 | 0-4-0 | 2 | -2.429 |
| princesse de Nassau | 1308 ± 496 | 811.3 | 248.2 | 0.06 | 271 | 180 | True | 1 | 0-1-0 | 1 | -2.368 |
| la marquise | 1301 ± 501 | 799.4 | 250.7 | 0.06 | 272 | 212 | True | 1 | 0-1-0 | 1 | -1.516 |
| La Moussaye | 1500 ± 703 | 797.2 | 351.4 | 0.06 | 273 | 147 | True | 0 | 0-0-0 | 1 | -0.4 |
| Périgot (Joseph) | 1500 ± 704 | 796.3 | 351.9 | 0.06 | 274 | 149 | True | 0 | 0-0-0 | 1 | -2.02 |
| la « marquise » | 1500 ± 704 | 796.0 | 352.0 | 0.06 | 275 | 150 | True | 0 | 0-0-0 | 1 | -2.44 |
| Mme Poncin | 1500 ± 704 | 795.7 | 352.2 | 0.06 | 276 | 148 | True | 0 | 0-0-0 | 1 | +0.119 |
| Mme Blatin | 1233 ± 438 | 794.4 | 219.1 | 0.06 | 277 | 236 | True | 2 | 0-2-0 | 1 | -3.476 |
| Monsieur Vallenères | 1206 ± 419 | 786.9 | 209.3 | 0.06 | 278 | 239 | True | 3 | 0-3-0 | 1 | -2.604 |
| M. Pierre | 1141 ± 372 | 769.5 | 185.8 | 0.06 | 279 | 258 | True | 4 | 0-4-0 | 2 | -2.821 |
| le grand-duc héritier de Luxembourg | 1274 ± 507 | 767.8 | 253.3 | 0.06 | 280 | 207 | True | 1 | 0-1-0 | 1 | -1.198 |
| vicomtesse de Saint-Fiacre | 1338 ± 581 | 757.1 | 290.3 | 0.06 | 281 | 197 | True | 1 | 0-1-0 | 1 | -2.128 |
| comtesse G… | 1338 ± 585 | 752.9 | 292.4 | 0.06 | 282 | 195 | True | 1 | 0-1-0 | 1 | -1.864 |
| ma grand’tante | 1105 ± 353 | 751.8 | 176.5 | 0.06 | 283 | 269 | True | 7 | 0-7-0 | 1 | -1.48 |
| ma grand'tante | 1338 ± 587 | 750.7 | 293.5 | 0.06 | 284 | 196 | True | 1 | 0-1-0 | 1 | -0.96 |
| Madame d'Ambresac | 1253 ± 512 | 740.8 | 256.0 | 0.06 | 285 | 229 | True | 2 | 0-2-0 | 1 | 0.0 |
| Dumont | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 286 | 224 | True | 2 | 0-2-0 | 1 | -1.568 |
| le curé | 1253 ± 514 | 738.6 | 257.0 | 0.06 | 287 | 227 | True | 2 | 0-2-0 | 1 | -2.04 |
| princesse d'Iéna | 1159 ± 460 | 699.4 | 229.9 | 0.06 | 288 | 249 | True | 3 | 0-3-0 | 1 | -2.085 |
