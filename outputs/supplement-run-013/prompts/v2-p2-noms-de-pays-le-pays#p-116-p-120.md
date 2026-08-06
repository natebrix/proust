You are annotating a French passage from Marcel Proust's *À la recherche du temps perdu* for **local appraisal events** and **character status effects**.

This is a **supplemental coverage pass**. The passage has already been annotated once. That accepted annotation captured the dominant local movement and its focal characters, and it is **fixed** — you must not re-score, revise, or contradict it.

Your job is narrower: judge whether any of the **additional candidate characters** listed below are **materially involved** in the local social or evaluative dynamics of the passage, and score **only those characters**.

## Inputs

You will be given:

1. A French passage.
2. An alias map for named characters.
3. The **accepted annotation** for this passage (characters already scored, with their events and status effects). This is fixed context, not a draft to improve.
4. A **candidate list** of additional characters detected in the passage text but not scored in the accepted annotation. The candidate list may include `le narrateur`.
5. Optionally, brief prior context from the immediately preceding window.

## Scope rules

* Score **only** characters from the candidate list. Never emit events or status effects whose target is an already-scored character.
* An already-scored character **may** appear as the `source` of an event targeting a candidate character.
* The candidate list is a mechanical screen, not a quota. Most candidates are peripheral mentions and should be **omitted**.
* Include a candidate only if omitting them would misrepresent how the passage locally positions its participants.
* Resolve references to the **canonical character name** using the alias map.
* Work primarily from the passage itself. Use prior context only for local disambiguation.
* Do not invent motives, unstated events, or long-run arc interpretations.
* Prefer the **smallest sufficient reading** of the passage.
* An **empty result** (`appraisal_events: []`, `status_effects: []`, and only trivially-present `characters_present`) is a valid, common, and expected outcome. Do not manufacture weak events to justify a candidate.

## The narrator as participant

`le narrateur` may appear in the candidate list. Distinguish carefully between two roles:

* **The narrating voice** — the retrospective "I" who tells, evaluates, and ironizes. This voice remains an evaluation `source` (use `"source": "narrator"` as in the accepted annotation). The voice is **never** a scored character.
* **The in-scene self** — the protagonist as a participant in the staged scene: he is received or snubbed, favored or dismissed, gains or loses composure, standing, or emotional leverage relative to the people in the room. This in-scene self is scored as the character `le narrateur`.

Score `le narrateur` only when the passage **stages** him as a social participant:

* he is included in or excluded from valued company
* another character defers to, favors, dismisses, or dominates him
* he gains or loses emotional leverage in a staged interaction (e.g., with Albertine or Gilberte)
* the scene's social outcome lands on him as a participant, not merely through him as a lens

Do **not** score `le narrateur` when:

* he is only the perceiving or remembering consciousness
* the passage is essayistic reflection, description, or generalization
* his "loss" or "gain" exists only at the level of retrospective commentary

In third-person stretches (notably *Un amour de Swann*), `le narrateur` should almost never be scored.

## What to detect

For candidate characters, track the same local shifts as the first pass:

* praise, blame, admiration, snub
* prestige or discredit by association
* narrated elevation or diminishment
* inclusion in or exclusion from valued social space
* signs that another character depends on, yields to, or dismisses them

## Interpretive principles

All interpretive rules of the first pass apply unchanged:

* judge only the local evaluative and social dynamics of the supplied passage
* do not judge morality, factual correctness, long-term importance, or desert
* distinguish who evaluates, who is targeted, and whether the passage endorses, neutrally reports, ironizes, or leaves uncertain that evaluation
* respect quoted speech, free indirect style, irony, and narrator distance
* do not force zero-sum logic — a candidate can gain or lose independently of the already-scored characters
* the consummation-and-renewal rule from the first pass applies: do not collapse attained intimacy or narrator-endorsed renewal into diminishment merely because the path was hesitant or dependent

## Relation to the accepted annotation

* The accepted annotation defines the dominant local movement. Do not restate it.
* Your events should cover the **remaining** participants' positioning, which is often quieter: a hostess's successful reception, a rival's eclipse, a servant's competence acknowledged, the narrator's admission or exclusion.
* If a candidate's only involvement is as part of the movement already captured (e.g., a collective source of an existing snub), and the passage gives them no distinct local outcome of their own, omit them.
* Never emit an event that reverses the direction of an accepted event for the same interaction. If you believe the accepted annotation is wrong, record that in `ambiguities` — do not correct it through scoring.

## Task

1. From the candidate list, identify which characters (if any) are materially involved in the local movement.
2. Extract only the **significant** appraisal or status-relevant events involving them.
3. Record only the dominant local status effects for those characters.
4. Note ambiguity only when it materially changes the reading.
5. Prefer fewer, high-quality events. Default to **0 or 1** events. Never more than **3** events total, and only reach 3 when distinct candidates have genuinely distinct movements.
6. Never more than **2 status effects** for a single character.

## Output

Return valid JSON only, in exactly the first-pass schema:

{
"characters_present": [
{
"canonical_name": "string",
"surface_forms": ["string"],
"presence_type": "explicit | implicit",
"presence_confidence": 0.0
}
],
"appraisal_events": [
{
"event_id": "S1",
"source": "canonical character name | narrator | collective_social_voice | unknown",
"target": "canonical character name",
"type": "praise | blame | admiration | snub | prestige_association | discredit_association | narrated_elevation | narrated_diminishment | other",
"polarity": "positive | negative | mixed",
"narrative_stance": "endorsed | neutral_report | ironized | uncertain",
"confidence": 0.0,
"evidence": "brief quotation or paraphrase from the passage",
"explanation": "1-2 sentence explanation in English"
}
],
"status_effects": [
{
"character": "canonical character name",
"dimension": "general_appraisal | social_status | rhetorical_position | emotional_position | inclusion_exclusion",
"delta": -2,
"based_on_events": ["S1"],
"confidence": 0.0,
"explanation": "brief explanation in English"
}
],
"ambiguities": [
"string"
]
}

Schema guidance:

* `characters_present` lists only the candidate characters you actually scored (or judged explicitly implicit-but-material). Do not relist already-scored characters.
* Event ids use the `S` prefix (`S1`, `S2`, ...) so supplement events are distinguishable from first-pass events (`E1`, ...).
* `status_effects` targets must be candidate characters only.
* Delta scale, dimensions, stance values, and confidence conventions are identical to the first pass:
  * delta: -2 clearly diminished ... +2 clearly elevated
  * be conservative when irony, layered narration, or reference resolution makes interpretation unstable
* `explanation` fields must be written in English.
* `ambiguities` defaults to an empty list.

## Important rules

* Candidate characters only. Canonical names only.
* The accepted annotation is fixed; never re-score its characters.
* An empty supplement is a good supplement when the candidates are peripheral.
* Do not add a winner/loser verdict, a summary object, or fields beyond the schema.
* Do not turn one movement into a chain of micro-events.
* Do not add balancing effects unless both directions are central for that candidate.

## Inputs begin below

### Alias map

{
  "Swann": {
    "aliases": [
      "Swann",
      "M. Swann",
      "Charles Swann"
    ]
  },
  "Legrandin": {
    "aliases": [
      "Legrandin",
      "M. Legrandin"
    ]
  },
  "Mme de Villeparisis": {
    "aliases": [
      "Mme de Villeparisis",
      "Madame de Villeparisis"
    ]
  },
  "Mme de Cambremer": {
    "aliases": [
      "Mme de Cambremer",
      "Madame de Cambremer"
    ]
  },
  "M. Vinteuil": {
    "aliases": [
      "M. Vinteuil",
      "Vinteuil"
    ]
  },
  "la mère du narrateur": {
    "aliases": [
      "maman",
      "ma mère"
    ]
  },
  "Odette": {
    "aliases": [
      "Odette",
      "Odette de Crécy",
      "Odette de Crecy",
      "Mme de Crécy",
      "Mme de Crecy"
    ]
  },
  "Mme Verdurin": {
    "aliases": [
      "Mme Verdurin",
      "Madame Verdurin"
    ]
  },
  "M. Verdurin": {
    "aliases": [
      "M. Verdurin",
      "Monsieur Verdurin",
      "Verdurin"
    ]
  },
  "comte de Forcheville": {
    "aliases": [
      "Forcheville",
      "comte de Forcheville",
      "M. de Forcheville"
    ]
  },
  "Brichot": {
    "aliases": [
      "Brichot",
      "M. Brichot"
    ]
  },
  "docteur Cottard": {
    "aliases": [
      "Cottard",
      "docteur Cottard",
      "le docteur"
    ]
  },
  "Mme Cottard": {
    "aliases": [
      "Mme Cottard",
      "Madame Cottard"
    ]
  },
  "Saniette": {
    "aliases": [
      "Saniette"
    ]
  },
  "le peintre": {
    "aliases": [
      "le peintre",
      "peintre"
    ]
  },
  "marquis de Forestelle": {
    "aliases": [
      "marquis de Forestelle",
      "M. de Forestelle",
      "Forestelle"
    ]
  },
  "baron de Charlus": {
    "aliases": [
      "baron de Charlus",
      "Charlus"
    ]
  },
  "oncle Adolphe": {
    "aliases": [
      "mon oncle Adolphe",
      "oncle Adolphe",
      "Adolphe"
    ]
  },
  "marquise de Saint-Euverte": {
    "aliases": [
      "marquise de Saint-Euverte",
      "Mme de Saint-Euverte",
      "Saint-Euverte"
    ]
  },
  "général de Froberville": {
    "aliases": [
      "général de Froberville",
      "general de Froberville",
      "Froberville"
    ]
  },
  "marquis de Bréauté": {
    "aliases": [
      "marquis de Bréauté",
      "marquis de Breaute",
      "Bréauté",
      "Breaute"
    ]
  },
  "marquise de Gallardon": {
    "aliases": [
      "marquise de Gallardon",
      "Mme de Gallardon",
      "Gallardon"
    ]
  },
  "duc de Guermantes": {
    "aliases": [
      "duc de Guermantes"
    ]
  },
  "princesse de Parme": {
    "aliases": [
      "princesse de Parme"
    ]
  },
  "M. d'Orsan": {
    "aliases": [
      "M. d'Orsan",
      "d'Orsan",
      "Orsan"
    ]
  },
  "Rémi": {
    "aliases": [
      "Rémi",
      "Remi"
    ]
  },
  "comtesse de Monteriender": {
    "aliases": [
      "comtesse de Monteriender",
      "Mme de Monteriender",
      "Monteriender"
    ]
  },
  "Napoléon III": {
    "aliases": [
      "Napoléon III",
      "Napoleon III"
    ]
  },
  "Gilberte": {
    "aliases": [
      "Gilberte"
    ]
  },
  "Françoise": {
    "aliases": [
      "Françoise",
      "Francoise"
    ]
  },
  "la Berma": {
    "aliases": [
      "la Berma",
      "Berma"
    ]
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "le marquis de Norpois"
    ]
  },
  "la grand-mère": {
    "aliases": [
      "ma grand-mère",
      "grand-mère",
      "ma grand'mère",
      "grand'mère",
      "la grand-mère"
    ]
  },
  "M. de Stermaria": {
    "aliases": [
      "M. de Stermaria",
      "de Stermaria",
      "Stermaria"
    ]
  },
  "Aimé": {
    "aliases": [
      "Aimé",
      "Aime"
    ]
  },
  "Mlle de Stermaria": {
    "aliases": [
      "Mlle de Stermaria"
    ]
  },
  "marquis de Cambremer": {
    "aliases": [
      "marquis de Cambremer",
      "M. de Cambremer"
    ]
  },
  "princesse de Luxembourg": {
    "aliases": [
      "princesse de Luxembourg",
      "La princesse de Luxembourg"
    ]
  },
  "le père du narrateur": {
    "aliases": [
      "mon père",
      "votre père"
    ]
  },
  "Mme Blandais": {
    "aliases": [
      "Mme Blandais",
      "Madame Blandais"
    ]
  },
  "Mme Poncin": {
    "aliases": [
      "Mme Poncin",
      "Madame Poncin"
    ]
  },
  "Robert de Saint-Loup": {
    "aliases": [
      "Saint-Loup",
      "Robert de Saint-Loup",
      "marquis de Saint-Loup-en-Bray",
      "le neveu de Mme de Villeparisis"
    ]
  },
  "M. de Marsantes": {
    "aliases": [
      "M. de Marsantes",
      "Marsantes",
      "Saint-Loup de Saint-Loup"
    ]
  },
  "Bloch": {
    "aliases": [
      "Bloch",
      "Bloch fils"
    ]
  },
  "prince des Laumes": {
    "aliases": [
      "prince des Laumes"
    ]
  },
  "Bloch père": {
    "aliases": [
      "Bloch père"
    ]
  },
  "le directeur": {
    "aliases": [
      "le directeur",
      "directeur"
    ]
  },
  "Dreyfus": {
    "aliases": [
      "Dreyfus"
    ]
  },
  "jeune blonde de Rivebelle": {
    "aliases": [
      "jeune blonde",
      "jeune blonde à l'air triste"
    ]
  },
  "duchesse de Guermantes": {
    "aliases": [
      "duchesse de Guermantes",
      "Mme de Guermantes",
      "Madame de Guermantes",
      "la duchesse"
    ]
  },
  "Jupien": {
    "aliases": [
      "Jupien"
    ]
  },
  "princesse de Guermantes": {
    "aliases": [
      "princesse de Guermantes",
      "princesse de Guermantes-Bavière",
      "Mme de Guermantes-Bavière"
    ]
  },
  "duc de Châtellerault": {
    "aliases": [
      "duc de Châtellerault",
      "M. de Châtellerault",
      "Châtellerault"
    ]
  },
  "M. de Vaugoubert": {
    "aliases": [
      "M. de Vaugoubert",
      "Vaugoubert"
    ]
  },
  "Mme de Vaugoubert": {
    "aliases": [
      "Mme de Vaugoubert",
      "Madame de Vaugoubert"
    ]
  },
  "Albertine": {
    "aliases": [
      "Albertine"
    ]
  },
  "Andrée": {
    "aliases": [
      "Andrée",
      "Andree"
    ]
  },
  "Mme Bontemps": {
    "aliases": [
      "Mme Bontemps",
      "Madame Bontemps"
    ]
  },
  "Morel": {
    "aliases": [
      "Morel"
    ]
  },
  "Elstir": {
    "aliases": [
      "Elstir"
    ]
  },
  "prince de Léon": {
    "aliases": [
      "prince de Léon",
      "prince de Leon",
      "Léon",
      "Leon"
    ]
  },
  "marquis du Lau": {
    "aliases": [
      "marquis du Lau",
      "du Lau"
    ]
  },
  "Mme de Chaussepierre": {
    "aliases": [
      "Mme de Chaussepierre",
      "Madame de Chaussepierre",
      "Chaussepierre"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Mme de Villeparisis",
      "surface_forms": [
        "Mme de Villeparisis",
        "la marquise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Mme de Villeparisis",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.72,
      "evidence": "Il savait qu'une personne qui amène ses gens avec soi ... donne peu de pourboires ... que les nobles de l'ancien faubourg Saint-Germain agissent de même. Mme de Villeparisis appartenait à la fois à ces deux catégories. Le chasseur ... en concluait qu'il n'avait rien à attendre de la marquise.",
      "explanation": "The passage aligns Mme de Villeparisis with categories (guests with their own servants; nobles of the faubourg) that hotel staff regard as poor tippers, leading the groom to withhold service. This socially discredits her in the hotel context by association rather than by any explicit action of hers."
    }
  ],
  "status_effects": [
    {
      "character": "Mme de Villeparisis",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.7,
      "explanation": "She is locally excluded from the groom's attentions (he does not assist her), owing to the staff’s expectation that nobles like her tip little and rely on their own servants."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-116-p-120"
}

### Candidate characters

[
  "Mme Blandais",
  "Mme de Cambremer",
  "la grand-mère",
  "le directeur",
  "le narrateur"
]

### Prior local context (optional)

Il ne faut, d'ailleurs, pas croire que ce malentendu fut momentané comme ceux qui se forment au deuxième acte d'un vaudeville pour se dissiper au dernier. Mme de Luxembourg, nièce du roi d'Angleterre et de l'empereur d'Autriche, et Mme de Villeparisis parurent toujours, quand la première venait chercher la seconde pour se promener en voiture, deux drôlesses de l'espèce de celles dont on se gare difficilement dans les villes d'eaux. Les trois quarts des hommes du faubourg Saint-Germain passent aux yeux d'une bonne partie de la bourgeoisie pour des décavés crapuleux (qu'ils sont d'ailleurs quelquefois individuellement) et que, par conséquent, personne ne reçoit. La bourgeoisie est trop honnête en cela, car leurs tares ne les empêcheraient nullement d'être reçus avec la plus grande faveur là où elle ne le sera jamais. Et eux s'imaginent tellement que la bourgeoisie le sait qu'ils affectent une simplicité en ce qui les concerne, un dénigrement pour leurs amis particulièrement « à la côte », qui achève le malentendu. Si par hasard un homme du grand monde est en rapports avec la petite bourgeoisie parce qu'il se trouve, étant extrêmement riche, avoir la présidence des plus importantes sociétés financières, la bourgeoisie qui voit enfin un noble digne d'être grand bourgeois jurerait qu'il ne fraye pas avec le marquis joueur et ruiné qu'elle croit d'autant plus dénué de relations qu'il est plus aimable. Et elle n'en revient pas quand le duc, président du conseil d'administration de la colossale Affaire, donne pour femme à son fils la fille du marquis joueur, mais dont le nom est le plus ancien de France, de même qu'un souverain fera plutôt épouser à son fils la fille d'un roi détrôné que d'un président de la république en fonctions. C'est dire que les deux mondes ont l'un de l'autre une vue aussi chimérique que les habitants d'une plage située à une des extrémités de la baie de Balbec ont de la plage située à l'autre extrémité : de Rivebelle on voit un peu Marcouville l'Orgueilleuse ; mais cela même trompe, car on croit qu'on est vu de Marcouville d'où au contraire les splendeurs de Rivebelle sont en grande partie invisibles.

### Passage

Le médecin de Balbec appelé pour un accès de fièvre que j'avais eu, ayant estimé que je ne devrais pas rester toute la journée au bord de la mer, en plein soleil, par les grandes chaleurs, et rédigé à mon usage quelques ordonnances pharmaceutiques, ma grand'mère prit les ordonnances avec un respect apparent où je reconnus tout de suite sa ferme décision de n'en faire exécuter aucune, mais tint compte du conseil en matière d'hygiène et accepta l'offre de Mme de Villeparisis de nous faire faire quelques promenades en voiture. J'allais et venais, jusqu'à l'heure du déjeuner, de ma chambre à celle de ma grand'mère. Elle ne donnait pas directement sur la mer comme la mienne mais prenait jour de trois côtés différents : sur un coin de la digue, sur une cour et sur la campagne, et était meublée autrement avec des fauteuils brodés de filigranes métalliques et de fleurs roses d'où semblait émaner l'agréable et fraîche odeur qu'on trouvait en entrant. Et à cette heure où des rayons venus d'expositions, et comme d'heures différentes, brisaient les angles du mur, à côté d'un reflet de la plage mettaient sur la commode un reposoir diapré comme les fleurs du sentier, suspendaient à la paroi les ailes repliées, tremblantes et tièdes d'une clarté prête à reprendre son vol, chauffaient comme un bain un carré de tapis provincial devant la fenêtre de la courette que le soleil festonnait comme une vigne, ajoutaient au charme et à la complexité de la décoration mobilière en semblant exfolier la soie fleurie des fauteuils et détacher leur passementerie, cette chambre que je traversais un moment avant de m'habiller pour la promenade, avait l'air d'un prisme où se décomposaient les couleurs de la lumière du dehors, d'une ruche où les sucs de la journée que j'allais goûter étaient dissociés, épars, enivrants et visibles, d'un jardin de l'espérance qui se dissolvait en une palpitation de rayons d'argent et de pétales de rose. Mais avant tout j'avais ouvert mes rideaux dans l'impatience de savoir quelle était la Mer qui jouait ce matin-là au bord du rivage, comme une Néréide. Car chacune de ces Mers ne restait jamais plus d'un jour. Le lendemain il y en avait une autre qui parfois lui ressemblait. Mais je ne vis jamais deux fois la même.

Il y en avait qui étaient d'une beauté si rare qu'en les apercevant mon plaisir était encore accru par la surprise. Par quel privilège, un matin plutôt qu'un autre, la fenêtre en s'entr'ouvrant découvrit-elle à mes yeux émerveillés la nymphe Glaukonomèné, dont la beauté paresseuse et qui respirait mollement avait la transparence d'une vaporeuse émeraude à travers laquelle je voyais affluer les éléments pondérables qui la coloraient ? Elle faisait jouer le soleil avec un sourire alangui par une brume invisible qui n'était qu'un espace vide réservé autour de sa surface translucide rendue ainsi plus abrégée et plus saisissante, comme ces déesses que le sculpteur détache sur le reste du bloc qu'il ne daigne pas dégrossir. Telle, dans sa couleur unique, elle nous invitait à la promenade sur ces routes grossières et terriennes, d'où, installés dans la calèche de Mme de Villeparisis, nous apercevions tout le jour et sans jamais l'atteindre la fraîcheur de sa molle palpitation.

Mme de Villeparisis faisait atteler de bonne heure, pour que nous eussions le temps d'aller soit jusqu'à Saint-Mars-le-Vêtu, soit jusqu'aux rochers de Quetteholme ou à quelque autre but d'excursion qui, pour une voiture assez lente, était fort lointain et demandait toute la journée. Dans ma joie de la longue promenade que nous allions entreprendre, je fredonnais quelque air récemment écouté, et je faisais les cent pas en attendant que Mme de Villeparisis fût prête. Si c'était dimanche, sa voiture n'était pas seule devant l'hôtel ; plusieurs fiacres loués attendaient, non seulement les personnes qui étaient invitées au château de Féterne chez Mme de Cambremer, mais celles qui plutôt que de rester là comme des enfants punis déclaraient que le dimanche était un jour assommant à Balbec et partaient dès après déjeuner se cacher dans une plage voisine ou visiter quelque site, et même souvent, quand on demandait à Mme Blandais si elle avait été chez les Cambremer, elle répondait péremptoirement : « Non, nous étions aux cascades du Bec », comme si c'était là la seule raison pour laquelle elle n'avait pas passé la journée à Féterne. Et le bâtonnier disait charitablement :

– Je vous envie, j'aurais bien changé avec vous, c'est autrement intéressant.

À côté des voitures, devant le porche où j'attendais, était planté comme un arbrisseau d'une espèce rare un jeune chasseur qui ne frappait pas moins les yeux par l'harmonie singulière de ses cheveux colorés, que par son épiderme de plante. À l'intérieur dans le hall qui correspondait au narthex ou église des Catéchumènes, des églises romanes, et où les personnes qui n'habitaient pas l'hôtel avaient le droit de passer, les camarades du groom « extérieur » ne travaillaient pas beaucoup plus que lui mais exécutaient du moins quelques mouvements. Il est probable que le matin ils aidaient au nettoyage. Mais l'après-midi ils restaient là seulement comme des choristes qui, même quand ils ne servent à rien, demeurent en scène pour ajouter à la figuration. Le Directeur général, celui qui me faisait si peur, comptait augmenter considérablement leur nombre l'année suivante, car il « voyait grand ». Et sa décision affligeait beaucoup le directeur de l'Hôtel, lequel trouvait que tous ces enfants n'étaient que des « faiseurs d'embarras » entendant par là qu'ils embarrassaient le passage et ne servaient à rien. Du moins entre le déjeuner et le dîner, entre les sorties et les rentrées des clients remplissaient-ils le vide de l'action comme ces élèves de Mme de Maintenon qui sous le costume de jeunes israélites font intermède chaque fois qu'Esther ou Joad s'en vont. Mais le chasseur du dehors, aux nuances précieuses, à la taille élancée et frêle, non loin duquel j'attendais que la marquise descendît, gardait une immobilité à laquelle s'ajoutait de la mélancolie, car ses frères aînés avaient quitté l'hôtel pour des destinées plus brillantes et il se sentait isolé sur cette terre étrangère. Enfin Mme de Villeparisis arrivait. S'occuper de sa voiture et l'y faire monter eût peut-être dû faire partie des fonctions du chasseur. Mais il savait qu'une personne qui amène ses gens avec soi se fait servir par eux, et d'habitude donne peu de pourboires dans un hôtel, que les nobles de l'ancien faubourg Saint-Germain agissent de même. Mme de Villeparisis appartenait à la fois à ces deux catégories. Le chasseur arborescent en concluait qu'il n'avait rien à attendre de la marquise ; en laissant le maître d'hôtel et la femme de chambre de celle-ci l'installer avec ses affaires, il rêvait tristement au sort envié de ses frères et conservait son immobilité végétale.
