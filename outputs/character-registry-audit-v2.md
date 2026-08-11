# Character Registry Audit v2

## A. Historical text-rewriting blast radius

- Unit files where preprocessed differs from raw: **2827**
- Distinct substitution rules observed: **4166**
- Annotated unit files with SEVERE substitutions (descriptor/mangled): **630**
- Annotated unit files with possessive normalization only: **515**
- Annotated unit files with name-variant flattening (e.g. Mme de Guermantes -> duchesse de Guermantes): **556**

| class | orig | new | count | example |
| --- | --- | --- | --- | --- |
| variant_flattening | `la Mme de` | `la duchesse de` | 430 | run-158/v3-p1#p-51-p-55 |
| variant_flattening | `de Mme de` | `de duchesse de` | 316 | run-152/v3-p1#p-1-p-5 |
| variant_flattening | `à Mme de` | `à duchesse de` | 150 | run-162/v3-p1#p-86-p-90 |
| variant_flattening | `que Mme de` | `que duchesse de` | 134 | run-152/v3-p1#p-1-p-5 |
| variant_flattening | `chez Mme de` | `chez duchesse de` | 71 | run-156/v3-p1#p-31-p-35 |
| mangled | `Mme de` | `Mme de M. de` | 67 | run-188/v3-p1#p-391-p-395 |
| variant_flattening | `dit M. de` | `dit marquis de` | 56 | run-232/v4-p2#p-346-p-350 |
| variant_flattening | `pour Mme de` | `pour duchesse de` | 52 | run-172/v3-p1#p-176-p-180 |
| descriptor_substitution | `de la duchesse de` | `de duchesse de Guermantes de` | 52 | run-190/v3-p1#p-421-p-425 |
| variant_flattening | `dit Mme de` | `dit duchesse de` | 52 | run-192/v3-p1#p-471-p-475 |
| variant_flattening | `chez Mme de` | `chez marquise de` | 50 | run-065/v1-p2-un-amour-de-swann#p-436-p-442 |
| variant_flattening | `de M. de` | `de marquis de` | 47 | run-222/v4-p2#p-186-p-190 |
| variant_flattening | `à M. de` | `à marquis de` | 44 | run-061/v1-p2-un-amour-de-swann#p-382-p-386 |
| variant_flattening | `de Mme de` | `de marquise de` | 44 | run-067/v1-p2-un-amour-de-swann#p-447-p-460 |
| variant_flattening | `« Mme de` | `« duchesse de` | 44 | run-190/v3-p1#p-436-p-440 |
| possessive_shift | `que ma grand'mère avait` | `que la grand-mère avait` | 34 | run-152/v3-p1#p-11-p-15 |
| possessive_shift | `de ma grand'mère et` | `de la grand-mère et` | 33 | run-112/v2-p2-noms-de-pays-le-pays#p-16-p-20 |
| variant_flattening | `répondit Mme de` | `répondit duchesse de` | 30 | run-194/v3-p1#p-486-p-490 |
| variant_flattening | `» Mme de` | `» duchesse de` | 28 | run-214/v4-p2#p-76-p-80 |
| variant_flattening | `que M. de` | `que marquis de` | 28 | run-232/v4-p2#p-346-p-350 |
| variant_flattening | `que Mme de` | `que marquise de` | 26 | run-067/v1-p2-un-amour-de-swann#p-461-p-468 |
| descriptor_substitution | `à la duchesse de` | `à duchesse de Guermantes de` | 23 | run-218/v4-p2#p-126-p-130 |
| variant_flattening | `avec Mme de` | `avec duchesse de` | 22 | run-172/v3-p1#p-176-p-180 |
| variant_flattening | `et Mme de` | `et duchesse de` | 22 | run-218/v4-p2#p-146-p-150 |
| possessive_shift | `que ma grand'mère ne` | `que la grand-mère ne` | 20 | run-114/v2-p2-noms-de-pays-le-pays#p-61-p-65 |
| possessive_shift | `de ma grand'mère, je` | `de la grand-mère, je` | 20 | run-136/v2-p2-noms-de-pays-le-pays#p-291-p-295 |
| variant_flattening | `répondit M. de` | `répondit marquis de` | 20 | run-232/v4-p2#p-346-p-350 |
| descriptor_substitution | `que maman me` | `que la mère du narrateur me` | 18 | run-073/v1-p3-noms-de-pays-le-nom#p-10-p-12 |
| possessive_shift | `à ma mère :` | `à la mère du narrateur :` | 18 | run-073/v1-p3-noms-de-pays-le-nom#p-13-p-18 |
| descriptor_substitution | `à maman :` | `à la mère du narrateur :` | 18 | run-089/v2-p1-autour-de-mme-swann#p-89-p-98 |
| possessive_shift | `de mon père et` | `de le père du narrateur et` | 18 | run-122/v2-p2-noms-de-pays-le-pays#p-131-p-135 |
| variant_flattening | `où Mme de` | `où duchesse de` | 18 | run-152/v3-p1#p-1-p-5 |
| possessive_shift | `que ma grand'mère eût` | `que la grand-mère eût` | 17 | run-226/v4-p2#p-256-p-260 |
| deletion | `de mon oncle` | `de oncle` | 16 | run-091/v2-p1-autour-de-mme-swann#p-105-p-113 |
| possessive_shift | `de ma grand'mère. Elle` | `de la grand-mère. Elle` | 16 | run-120/v2-p2-noms-de-pays-le-pays#p-116-p-120 |
| variant_flattening | `ajouta Mme de` | `ajouta duchesse de` | 16 | run-194/v3-p1#p-491-p-495 |
| variant_flattening | `– Mme de` | `– duchesse de` | 16 | run-208/v3-p1#p-711-p-715 |
| variant_flattening | `à Mme de` | `à marquise de` | 14 | run-067/v1-p2-un-amour-de-swann#p-447-p-460 |
| possessive_shift | `que ma mère était` | `que la mère du narrateur était` | 14 | run-089/v2-p1-autour-de-mme-swann#p-89-p-98 |
| possessive_shift | `à ma grand'mère et` | `à la grand-mère et` | 14 | run-114/v2-p2-noms-de-pays-le-pays#p-61-p-65 |

(showing 40 non-expansion rules; full list in the JSON)

## B. Exclusion gaps (mentioned in original text, absent from characters_present)

Latest-run annotations scanned: 1185 (selection rule: highest run number per unit)

| entity | units w/ gap | strong (≥2 mentions) | total mentions |
| --- | --- | --- | --- |
| Odette | 264 | 165 | 1176 |
| Swann | 269 | 143 | 838 |
| baron de Charlus | 222 | 120 | 829 |
| la mère du narrateur | 258 | 120 | 649 |
| M. Verdurin | 193 | 112 | 564 |
| duchesse de Guermantes | 210 | 108 | 585 |
| Gilberte | 147 | 97 | 660 |
| Albertine | 127 | 96 | 921 |
| Robert de Saint-Loup | 193 | 92 | 512 |
| Françoise | 174 | 85 | 502 |
| Mme Verdurin | 143 | 78 | 531 |
| la grand-mère | 172 | 74 | 428 |
| le père du narrateur | 133 | 57 | 273 |
| Bloch | 89 | 51 | 299 |
| Mme de Villeparisis | 120 | 49 | 236 |
| comte de Forcheville | 84 | 47 | 196 |
| duc de Guermantes | 108 | 46 | 228 |
| Morel | 55 | 45 | 433 |
| docteur Cottard | 85 | 41 | 278 |
| Elstir | 93 | 39 | 212 |
| Andrée | 57 | 38 | 239 |
| Bergotte | 78 | 36 | 175 |
| Brichot | 59 | 34 | 271 |
| Norpois | 74 | 33 | 174 |
| Jupien | 42 | 29 | 195 |
| Rachel | 44 | 28 | 187 |
| Dreyfus | 63 | 24 | 112 |
| Mme Bontemps | 53 | 23 | 125 |
| M. Vinteuil | 48 | 23 | 109 |
| princesse de Guermantes | 69 | 22 | 112 |

## C. Phantom presences (listed present, no support in original text)

| entity | substitution-induced | no-surface-match |
| --- | --- | --- |
| duchesse de Guermantes | 8 | 5 |
| docteur Cottard | 5 | 6 |
| Swann | 0 | 10 |
| duc de Guermantes | 0 | 8 |
| Robert de Saint-Loup | 0 | 7 |
| le directeur | 0 | 7 |
| Odette | 0 | 6 |
| le peintre | 0 | 6 |
| Albertine | 0 | 6 |
| la grand-mère | 3 | 1 |
| princesse de Guermantes | 0 | 4 |
| le narrateur | 0 | 4 |
| Mme de Villeparisis | 0 | 4 |
| oncle Adolphe | 0 | 3 |
| Mme de Cambremer | 0 | 3 |
| Mme Verdurin | 0 | 2 |
| la mère du narrateur | 0 | 2 |
| Norpois | 1 | 1 |
| baron de Charlus | 0 | 2 |
| le père du narrateur | 0 | 1 |
| princesse de Parme | 0 | 1 |
| le pianiste | 0 | 1 |
| Mme Cottard | 0 | 1 |
| Legrandin | 0 | 1 |
| marquis de Cambremer | 0 | 1 |
| Mlle de Stermaria | 0 | 1 |
| la reine de Naples | 0 | 1 |
| Mlle d'Éporcheville | 0 | 1 |
| Elstir | 0 | 1 |

Ambiguous surfaces routed to triage: prince de Guermantes (58), Mlle d'Éporcheville (14)

## D. Chimera adjudication

### M. de Marsantes — units listing him present

**run-147/v2-p2-noms-de-pays-le-pays#p-441-p-445**
- (no 'de Marsantes' string in raw text at all)

**run-266/v7-p2-m-de-charlus-pendant-la-guerre#p-76-p-80**
- [Mme] …t, une fois mobilisé, pour fuir devant le danger. « Pauvre dame, disait-elle en pensant à Mme de Marsantes, qu'est-ce qu'elle a dû pleurer quand elle a appris la mort de son garçon ! Si encore ell…
- [Mme] …perçait la curiosité cruelle de la paysanne. Sans doute Françoise plaignait la douleur de Mme de Marsantes de tout son coeur, mais elle regrettait de ne pas connaître la forme que cette douleur av…

**run-446/v3-p2#p-231-p-235**
- (no 'de Marsantes' string in raw text at all)

**run-544/v7-p2-m-de-charlus-pendant-la-guerre#p-76-p-80**
- [Mme] …t, une fois mobilisé, pour fuir devant le danger. « Pauvre dame, disait-elle en pensant à Mme de Marsantes, qu'est-ce qu'elle a dû pleurer quand elle a appris la mort de son garçon ! Si encore ell…
- [Mme] …perçait la curiosité cruelle de la paysanne. Sans doute Françoise plaignait la douleur de Mme de Marsantes de tout son coeur, mais elle regrettait de ne pas connaître la forme que cette douleur av…

### Octave — units listing him present

| unit | bare 'Octave' | 'Mme Octave' |
| --- | --- | --- |

## E. Candidate names not in the registry (honorific patterns, count ≥ 3)

| candidate | count |
| --- | --- |
| M. de Bréauté | 53 |
| M. d'Argencourt | 47 |
| Mme d'Arpajon | 38 |
| Mme de Sévigné | 36 |
| Mme de Surgis | 34 |
| Mme Leroi | 28 |
| comtesse Molé | 21 |
| Mme de Souvré | 19 |
| Mme Molé | 18 |
| Mme Putbus | 18 |
| Mme Goupil | 16 |
| marquis de Saint-Loup | 16 |
| Mme de Sainte-Euverte | 14 |
| duc d'Aumale | 13 |
| docteur du Boulbon | 12 |
| marquise de Cambremer | 11 |
| Mme de Franquetot | 11 |
| Mme de Montmorency | 11 |
| Mme de Mortemart | 11 |
| Mlle de Saint-Loup | 11 |
| docteur Percepied | 10 |
| M. Bontemps | 10 |
| princesse Mathilde | 9 |
| Mme de Parme | 9 |
| Mme de Valcourt | 9 |
| baronne Putbus | 8 |
| duc de Chartres | 8 |
| Mme Blatin | 8 |
| Madame de Sévigné | 8 |
| capitaine de Borodino | 8 |
| M. de Grouchy | 8 |
| princesse d'Épinay | 8 |
| Mme de Varambon | 8 |
| duc de Brabant | 8 |
| princesse de Caprarola | 8 |
| marquise de Saint-Loup | 8 |
| duc d'Orléans | 7 |
| duchesse de Vendôme | 7 |
| Mme de Beausergent | 7 |
| tante Villeparisis | 7 |
| Mlle d'Ambresac | 7 |
| Mme de Citri | 7 |
| M. de Froberville | 6 |
| M. de Chateaubriand | 6 |
| duchesse de Luxembourg | 6 |
| duchesse de Montmorency | 6 |
| Mme d'Épinay | 6 |
| Mme de Duras | 6 |
| princesse de Sagan | 5 |
| duchesse de La Rochefoucauld | 5 |
| M. de Borodino | 5 |
| M. Decazes | 5 |
| princesse de Silistrie | 5 |
| général de Saint-Joseph | 5 |
| Mme de Villebon | 5 |
| Mlle de Guermantes | 5 |
| duc de Guastalla | 5 |
| M. de Luxembourg | 5 |
| Mme de Villemur | 5 |
| colonel de Froberville | 5 |

(full candidate list in the JSON: 135 entries)
