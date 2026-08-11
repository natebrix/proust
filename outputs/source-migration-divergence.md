# Source migration divergence report

Legacy `fr-original` (marcel-proust.com transcription) aligned to the
staged `fr-original-ws` chapters built from the pinned French Wikisource
revisions (`outputs/wikisource-mapping.json`, `data/wikisource/manifest.json`).
Similarity is a token-level `difflib` ratio on typography-normalized text
(NFC, unified apostrophes/quotes/dashes/ellipses, lowercased); it is not an
equality test, because the two transcriptions genuinely differ.

Gates: every old paragraph paired or annotated; new_only < 2% per chapter; median 1:1 similarity >= 0.95.

## Summary

| chapter | old | new | 1:1 | split | merge | old_only | new_only | median | p10 | min | word div | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1-p1-combray | 371 | 367 | 359 | 1 | 5 | 1 | 0 | 1.000 | 0.996 | 0.923 | 0.18% | PASS |
| v1-p2-un-amour-de-swann | 591 | 587 | 583 | 0 | 4 | 0 | 0 | 1.000 | 1.000 | 0.938 | 0.08% | PASS |
| v1-p3-noms-de-pays-le-nom | 67 | 64 | 61 | 0 | 3 | 0 | 0 | 1.000 | 0.999 | 0.992 | 0.05% | PASS |
| v2-p1-autour-de-mme-swann | 332 | 320 | 308 | 0 | 12 | 0 | 0 | 1.000 | 0.998 | 0.975 | 0.11% | PASS |
| v2-p2-noms-de-pays-le-pays | 492 | 479 | 467 | 0 | 12 | 0 | 0 | 1.000 | 0.995 | 0.857 | 0.22% | PASS |
| v3-p1 | 902 | 897 | 892 | 0 | 5 | 0 | 0 | 1.000 | 1.000 | 0.978 | 0.01% | PASS |
| v3-p2 | 733 | 717 | 702 | 1 | 13 | 4 | 0 | 1.000 | 1.000 | 0.981 | 0.01% | PASS |
| v4-p1 | 22 | 21 | 17 | 1 | 2 | 0 | 0 | 1.000 | 0.999 | 0.997 | 0.03% | PASS |
| v4-p2 | 450 | 417 | 387 | 2 | 26 | 7 | 0 | 1.000 | 1.000 | 0.909 | 0.08% | PASS |
| v5 | 428 | 389 | 361 | 0 | 28 | 11 | 0 | 1.000 | 1.000 | 0.875 | 0.06% | PASS |
| v6-p1 | 120 | 112 | 105 | 0 | 7 | 0 | 0 | 1.000 | 0.997 | 0.986 | 0.13% | PASS |
| v6-p2 | 72 | 69 | 66 | 0 | 3 | 0 | 0 | 1.000 | 1.000 | 0.992 | 0.04% | PASS |
| v6-p3 | 69 | 68 | 68 | 0 | 0 | 1 | 0 | 1.000 | 1.000 | 0.981 | 0.05% | PASS |
| v6-p4 | 25 | 23 | 21 | 0 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00% | PASS |
| v7-p1-a-tansonville | 25 | 24 | 23 | 0 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00% | PASS |
| v7-p2-m-de-charlus-pendant-la-guerre | 80 | 66 | 56 | 0 | 10 | 3 | 0 | 1.000 | 1.000 | 1.000 | 0.00% | PASS |
| v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle | 45 | 39 | 34 | 0 | 5 | 1 | 0 | 1.000 | 1.000 | 1.000 | 0.00% | PASS |
| v7-p4-le-bal-de-tetes | 141 | 120 | 109 | 0 | 11 | 9 | 0 | 1.000 | 1.000 | 1.000 | 0.00% | PASS |

## Per chapter

### v1-p1-combray

- paragraphs: old 371, new 367
- pairings: one_to_one 359, split 1, merge 5, old_only 1, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9963, min 0.9231
- word-level divergence on 1:1 pairs: 0.18%

**old_only (1)** — legacy paragraphs with no counterpart:

- `p-49` [wikisource_apparatus] 'II'

**weakest pairings (5)** — none below 0.75, these are the least similar:

- merge `p-171,p-172` -> `p-170` (0.900)
  - old: '« De ce timide Israëlite Quoi ! vous guidez ici les pas ! »'
  - new: '« De ce timide Israélite Quoi, vous guidez ici les pas ! »'
- one_to_one `p-225` -> `p-222` (0.923)
  - old: "– Monsieur le Curé, qu'est-ce que l'on me disait qu'il y a un artiste qui a installé son chevalet da…"
  - new: '— Monsieur le Curé, qu’est-ce que l’on me disait qu’il y a un artiste qui a installé son chevalet da…'
- one_to_one `p-75` -> `p-75` (0.952)
  - old: "– La fille de M. Pupin ! Oh ! je vous crois bien, ma pauvre Françoise ! Avec cela que je ne l'aurais…"
  - new: '— La fille à M. Pupin ! Oh ! je vous crois bien, ma pauvre Françoise ! Avec cela que je ne l’aurais …'
- one_to_one `p-84` -> `p-84` (0.957)
  - old: "– Comme si je ne connaissais pas le chien de Mme Sazerat ! répondait ma tante donc l'esprit critique…"
  - new: '— Comme si je ne connaissais pas le chien de Mme Sazerat ! répondait ma tante dont l’esprit critique…'
- one_to_one `p-331` -> `p-327` (0.960)
  - old: "– Quand je dis nous voir, je veux dire nous voir lire ; c'est assommant, quelque chose insignifiante…"
  - new: '— Quand je dis nous voir, je veux dire nous voir lire ; c’est assommant, quelque chose insignifiante…'

### v1-p2-un-amour-de-swann

- paragraphs: old 591, new 587
- pairings: one_to_one 583, split 0, merge 4, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9375
- word-level divergence on 1:1 pairs: 0.08%

**weakest pairings (5)** — none below 0.75, these are the least similar:

- one_to_one `p-293` -> `p-292` (0.938)
  - old: 'Tour le monde se retira fort tard. Les premiers mots de Cottard à sa femme furent :'
  - new: 'Tout le monde se retira fort tard. Les premiers mots de Cottard à sa femme furent :'
- one_to_one `p-73` -> `p-73` (0.950)
  - old: "– Mais je ne dis absolument rien. Voyons, docteur, je vous prends à témoin : est-ce que j'ai dit que…"
  - new: '— Mais je ne dis absolument rien. Voyons, docteur, je vous prends à témoins : est-ce que j’ai dit qu…'
- one_to_one `p-99` -> `p-99` (0.962)
  - old: "– Ah ! madame Verdurin, dit Cottard, sur un ton de marivaudage, vous oubliez que vous parlez d'un de…"
  - new: '— Ah ! madame Verdurin, dit Cottard, sur un ton de marivaudage, vous oubliez que vous parlez d’un de…'
- one_to_one `p-481` -> `p-478` (0.979)
  - old: '– Je ne vois aucun mal à ce que ce soit ancien, répondit sèchement la princesse, mais en tous cas ce…'
  - new: '— Je ne vois aucun mal à ce que ce soit ancien, répondit sèchement la princesse, mais en tous cas ce…'
- one_to_one `p-218` -> `p-217` (0.980)
  - old: "– À ce point de vue-là, c'était extraordinaire, mais cela ne semblait pas d'un art, comme on dit, tr…"
  - new: '— À ce point de vue-là, c’était extraordinaire, mais cela ne me semblait pas d’un art, comme on dit,…'

### v1-p3-noms-de-pays-le-nom

- paragraphs: old 67, new 64
- pairings: one_to_one 61, split 0, merge 3, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9985, min 0.9922
- word-level divergence on 1:1 pairs: 0.05%

### v2-p1-autour-de-mme-swann

- paragraphs: old 332, new 320
- pairings: one_to_one 308, split 0, merge 12, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9976, min 0.9750
- word-level divergence on 1:1 pairs: 0.11%

**weakest pairings (5)** — none below 0.75, these are the least similar:

- one_to_one `p-245` -> `p-239` (0.975)
  - old: "– Naturellement ! reprit-il. Cela prouve bien que c'est un esprit faux et malveillant. Mon pauvre fi…"
  - new: '— Naturellement ! reprit-il. Cela prouve bien que c’est un esprit faux et malveillant. Mon pauvre fi…'
- one_to_one `p-47` -> `p-46` (0.985)
  - old: "M. de Norpois leva les yeux au ciel d'un air de dire : Ah ! celui-là ! « D'abord, c'est un acte d'in…"
  - new: 'M. de Norpois leva les yeux au ciel d’un air de dire : Ah ! celui-là ! « D’abord, c’est un acte d’in…'
- one_to_one `p-49` -> `p-48` (0.985)
  - old: "– Mais oui, c'est un projet tout à fait attrayant et dont je me réjouis. J'aimerais beaucoup faire a…"
  - new: '— Mais oui, c’est un projet tout à fait attrayant dont je me réjouis. J’aimerais beaucoup faire avec…'
- one_to_one `p-96` -> `p-95` (0.985)
  - old: "– C'est extraordinaire qu'il ait dîné chez les Swann et qu'il y ait trouvé en somme des gens régulie…"
  - new: '— C’est extraordinaire qu’il ait dîné chez les Swann et qu’il y ait trouvé en somme des gens régulie…'
- one_to_one `p-36` -> `p-36` (0.987)
  - old: "Ma mère comptait beaucoup sur la salade d'ananas et de truffes. Mais l'Ambassadeur après avoir exerc…"
  - new: 'Ma mère comptait beaucoup sur la salade d’ananas et de truffes. Mais l’Ambassadeur après avoir exerc…'

### v2-p2-noms-de-pays-le-pays

- paragraphs: old 492, new 479
- pairings: one_to_one 467, split 0, merge 12, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9953, min 0.8571
- word-level divergence on 1:1 pairs: 0.22%

**weakest pairings (5)** — none below 0.75, these are the least similar:

- one_to_one `p-339` -> `p-334` (0.857)
  - old: 'Ma grand-mère ouvrait la porte de ma chambre, je lui posais mille questions sur la famille Legrandin…'
  - new: 'Ma grand’mère ouvrait la porte de ma chambre, je lui posais quelques questions sur la famille Legran…'
- one_to_one `p-320` -> `p-315` (0.903)
  - old: "On frappa ; c'était Aimé qui avait tenu à m'apporter lui-même les dernières listes d'étrangers."
  - new: 'On frappa ; c’était Aimé qui avait tenu à m’apporter lui-même les dernières listes des étrangers.'
- one_to_one `p-70` -> `p-69` (0.968)
  - old: '– Mais vous avez eu tort, je vous le répète, répondit le bâtonnier enhardi maintenant que le danger …'
  - new: '— Mais vous avez eu tort, je vous le répète, répondit le bâtonnier enhardi maintenant que le danger …'
- one_to_one `p-355` -> `p-350` (0.974)
  - old: "Ma grand-mère, à qui j'avais raconté mon entrevue avec Elstir et qui se réjouissait de tout le profi…"
  - new: 'Ma grand’mère, à qui j’avais raconté mon entrevue avec Elstir et qui se réjouissait de tout le profi…'
- one_to_one `p-377` -> `p-372` (0.974)
  - old: "« Il n'y a pas de jour qu'une ou l'autre d'entre elles ne passe devant l'atelier et n'entre me faire…"
  - new: '« Il n’y a pas de jour qu’une ou l’autre d’entre elles ne passe devant l’atelier et n’entre me faire…'

### v3-p1

- paragraphs: old 902, new 897
- pairings: one_to_one 892, split 0, merge 5, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9778
- word-level divergence on 1:1 pairs: 0.01%

**weakest pairings (3)** — none below 0.75, these are the least similar:

- one_to_one `p-399` -> `p-398` (0.978)
  - old: '– Écoute, pour le dernière fois, je te jure que tu auras beau faire, tu pourras avoir dans huit jour…'
  - new: '— Écoute, pour la dernière fois, je te jure que tu auras beau faire, tu pourras avoir dans huit jour…'
- one_to_one `p-767` -> `p-764` (0.986)
  - old: "J'aurais pourtant voulu avoir des renseignements non seulement sur Mme de Guermantes mais sur tous l…"
  - new: 'J’aurais pourtant voulu avoir des renseignements non seulement sur Mme de Guermantes mais sur tous l…'
- one_to_one `p-614` -> `p-612` (0.989)
  - old: "– Oh ! mon Dieu, monsieur, les rois et les reines, à notre époque ce n'est pas grand'chose ! dit M. …"
  - new: '— Oh ! mon Dieu, monsieur, les rois et les reines, à notre époque ce n’est pas grand’chose ! dit M. …'

### v3-p2

- paragraphs: old 733, new 717
- pairings: one_to_one 702, split 1, merge 13, old_only 4, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9808
- word-level divergence on 1:1 pairs: 0.01%

**old_only (4)** — legacy paragraphs with no counterpart:

- `p-95` [structural_spacer] ''
- `p-96` [wikisource_apparatus] 'Chapitre deuxième'
- `p-97` [structural_spacer] ''
- `p-98` [wikisource_apparatus] "Visite d'Albertine. Perspective d'un riche mariage pour quelques amis de Saint-Loup. L'esprit des Gu…"

**weakest pairings (2)** — none below 0.75, these are the least similar:

- one_to_one `p-554` -> `p-542` (0.981)
  - old: '– Ah ! je ne suis pas de votre avis, dit Mme de Guermantes, qui trouvait que le prince allemand manq…'
  - new: '— Ah ! je ne suis pas de votre avis, dit Mme de Guermantes, qui trouvait que le prince allemand manq…'
- one_to_one `p-322` -> `p-311` (0.988)
  - old: 'Ainsi grâce, une fois, à Taquin le Superbe, une autre fois à un autre mot, ces visites du duc et de …'
  - new: 'Ainsi grâce, une fois, à Taquin le Superbe, une autre fois à un autre mot, ces visites du duc et de …'

### v4-p1

- paragraphs: old 22, new 21
- pairings: one_to_one 17, split 1, merge 2, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9993, min 0.9972
- word-level divergence on 1:1 pairs: 0.03%

### v4-p2

- paragraphs: old 450, new 417
- pairings: one_to_one 387, split 2, merge 26, old_only 7, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9093
- word-level divergence on 1:1 pairs: 0.08%

**old_only (7)** — legacy paragraphs with no counterpart:

- `p-185` [wikisource_apparatus] 'Les intermittences du coeur'
- `p-214` [wikisource_apparatus] 'Chapitre deuxième'
- `p-215` [wikisource_apparatus] "Les mystères d'Albertine. – Les jeunes filles qu'elle voit dans la glace. – La dame inconnue. – Le l…"
- `p-365` [wikisource_apparatus] 'Chapitre troisième'
- `p-366` [wikisource_apparatus] "Tristesses de M. de Charlus. Son duel fictif. Les stations du « Transatlantique ». Fatigué d'Alberti…"
- `p-444` [wikisource_apparatus] 'Chapitre quatrième'
- `p-445` [wikisource_apparatus] 'Brusque revirement vers Albertine. Désolation au lever du soleil. Je pars immédiatement avec Alberti…'

**weakest pairings (2)** — none below 0.75, these are the least similar:

- one_to_one `p-441` -> `p-411` (0.909)
  - old: "Il suffit, de la sorte, qu'accidentellement, absurdement, un incident (ici la mise en présence d'Alb…"
  - new: 'Amitiés plus belles que celle de Bloch ne serait pas, du reste, beaucoup dire. Il avait tous les déf…'
- one_to_one `p-440` -> `p-410` (0.943)
  - old: "Pendant ces retours (comme à l'aller), je disais à Albertine de se vêtir, car je savais bien qu'à Am…"
  - new: 'Pendant ces retours (comme à l’aller), je disais à Albertine de se vêtir, car je savais bien qu’à Am…'

### v5

- paragraphs: old 428, new 389
- pairings: one_to_one 361, split 0, merge 28, old_only 11, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.8750
- word-level divergence on 1:1 pairs: 0.06%

**old_only (11)** — legacy paragraphs with no counterpart:

- `p-275` [wikisource_apparatus] 'Chapitre deuxième'
- `p-276` [wikisource_apparatus] 'Les Verdurin se brouillent avec M. de Charlus.'
- `p-292` [editorial_marker_block] "« C'est comme ça, Brichot, que vous vous promenez la nuit avec un beau jeune homme, dit-il en nous a…"
- `p-293` [editorial_marker] '[----Ajout Gallimard---- Et vous, mon cher, comment allez-vous ? me dit-il en quittant son ton plais…'
- `p-294` [editorial_marker_block] "En tout cas, même si je me trompe sur ce qu'il eût pu réaliser dans la moindre page, il eût rendu un…"
- `p-295` [editorial_marker_block] '"Oui, elle sait se vêtir ou plus exactement s\'habiller, reprit M. de Charlus au sujet d\'Albertine. M…'
- `p-296` [editorial_marker_block] '"Hé bien ! Baron", interrompit Brichot, craignant que j\'eusse du chagrin de ces derniers mots, car i…'
- `p-363` [wikisource_apparatus] 'Chapitre troisième'
- `p-364` [wikisource_apparatus] "Disparition d'Albertine"
- `p-386` [editorial_marker] "[L'édition sonore Thélème reprend ici un texte Gallimard, situé plus tôt dans l'édition originale (-…"
- `p-387` [editorial_marker_block] "------- Il n'y eut qu'un moment où j'eus pour elle une espèce de haine qui ne fit qu'aviver mon beso…"

**weakest pairings (4)** — none below 0.75, these are the least similar:

- one_to_one `p-419` -> `p-382` (0.875)
  - old: 'Soutiendrait les éclairs qui partent de vos yeux ?'
  - new: 'Soutiendrait les éclairs qui partent de ses yeux.'
- merge `p-297,p-298` -> `p-281` (0.974)
  - old: "Je vous ai dérangés, vous aviez l'air de vous amuser comme deux petites folles, et vous n'aviez pas …"
  - new: '« C’est comme ça, Brichot, que vous vous promenez la nuit avec un beau jeune homme, dit-il en nous a…'
- one_to_one `p-151` -> `p-147` (0.983)
  - old: "Pourtant, quand, le lendemain, Bloch m'eut envoyé la photographie de sa cousine Esther, je m'empress…"
  - new: 'Pourtant, quand, le lendemain, Bloch m’eut envoyé la photographie de sa cousine Esther, je m’empress…'
- one_to_one `p-157` -> `p-153` (0.986)
  - old: 'Albertine laissait parfois traîner dans ses propos tel ou tel de ces précieux amalgames, que je me h…'
  - new: 'Albertine laissait parfois traîner dans ses propos tel ou tel de ces précieux amalgames, que je me h…'

### v6-p1

- paragraphs: old 120, new 112
- pairings: one_to_one 105, split 0, merge 7, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 0.9965, min 0.9855
- word-level divergence on 1:1 pairs: 0.13%

**weakest pairings (1)** — none below 0.75, these are the least similar:

- one_to_one `p-48` -> `p-44` (0.986)
  - old: "Le temps passe, et peu à peu tout ce qu'on disait par mensonge devient vrai, je l'avais trop expérim…"
  - new: 'Le temps passe, et peu à peu tout ce qu’on disait par mensonge devient vrai, je l’avais trop expérim…'

### v6-p2

- paragraphs: old 72, new 69
- pairings: one_to_one 66, split 0, merge 3, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9917
- word-level divergence on 1:1 pairs: 0.04%

### v6-p3

- paragraphs: old 69, new 68
- pairings: one_to_one 68, split 0, merge 0, old_only 1, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9808
- word-level divergence on 1:1 pairs: 0.05%

**old_only (1)** — legacy paragraphs with no counterpart:

- `p-61` [section_break] '* * *'

**weakest pairings (1)** — none below 0.75, these are the least similar:

- one_to_one `p-64` -> `p-63` (0.981)
  - old: "Ma pensée, sans doute pour ne pas envisager une résolution à prendre, s'occupait tout entière à suiv…"
  - new: 'Ma pensée, sans doute pour ne pas envisager une résolution à prendre, s’occupait tout entière à suiv…'

### v6-p4

- paragraphs: old 25, new 23
- pairings: one_to_one 21, split 0, merge 2, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 0.9995
- word-level divergence on 1:1 pairs: 0.00%

### v7-p1-a-tansonville

- paragraphs: old 25, new 24
- pairings: one_to_one 23, split 0, merge 1, old_only 0, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 1.0000
- word-level divergence on 1:1 pairs: 0.00%

### v7-p2-m-de-charlus-pendant-la-guerre

- paragraphs: old 80, new 66
- pairings: one_to_one 56, split 0, merge 10, old_only 3, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 1.0000
- word-level divergence on 1:1 pairs: 0.00%

**old_only (3)** — legacy paragraphs with no counterpart:

- `p-5` [wikisource_apparatus] 'Chapitre II'
- `p-6` [wikisource_apparatus] 'M. de Charlus pendant la guerre ; ses opinions, ses plaisirs'
- `p-42` [section_break] '* * *'

### v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle

- paragraphs: old 45, new 39
- pairings: one_to_one 34, split 0, merge 5, old_only 1, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 1.0000
- word-level divergence on 1:1 pairs: 0.00%

**old_only (1)** — legacy paragraphs with no counterpart:

- `p-8` [section_break] '* * *'

### v7-p4-le-bal-de-tetes

- paragraphs: old 141, new 120
- pairings: one_to_one 109, split 0, merge 11, old_only 9, new_only 0
- 1:1 similarity: median 1.0000, p10 1.0000, min 1.0000
- word-level divergence on 1:1 pairs: 0.00%

**old_only (9)** — legacy paragraphs with no counterpart:

- `p-55` [section_break] '* * *'
- `p-64` [section_break] '* * *'
- `p-69` [section_break] '* * *'
- `p-80` [section_break] '* * *'
- `p-97` [section_break] '* * *'
- `p-138` [section_break] '--'
- `p-139` [colophon] 'FIN du roman A LA RECHERCHE DU TEMPS PERDU de'
- `p-140` [colophon_block] 'MARCEL PROUST'
- `p-141` [site_navigation] '--> Retour à la première page : 001 : [---- I ---- Combray] Longtemps, je me suis couché de bonne he…'

