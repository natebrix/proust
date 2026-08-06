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
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.95
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.72,
      "evidence": "« j'appuierais mes hésitations au calme de Robert de Saint-Loup » … « aussitôt commandés par mon ami »",
      "explanation": "The narrator frames Robert de Saint-Loup as a calming, decisive companion who will steady his hesitations and promptly order dishes, signaling deference and esteem."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "He is locally valued for composure and leadership in the dining scene, gaining esteem through the narrator's reliance."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-316-p-320"
}

### Candidate characters

[
  "Aimé",
  "le narrateur"
]

### Prior local context (optional)

Je sortis de l'ascenseur, mais au lieu d'aller vers ma chambre je m'engageai plus avant dans le couloir, car à cette heure-là le valet de chambre de l'étage, quoiqu'il craignît les courants d'air, avait ouvert la fenêtre du bout, laquelle regardait, au lieu de la mer, le côté de la colline et de la vallée, mais ne les laissait jamais voir, car ses vitres, d'un verre opaque, étaient le plus souvent fermées. Je m'arrêtai devant elle en une courte station et le temps de faire mes dévotions à la « vue » que pour une fois elle découvrait au delà de la colline à laquelle était adossé l'hôtel et qui ne contenait qu'une maison posée à quelque distance, mais à laquelle la perspective et la lumière du soir en lui conservant son volume donnait une ciselure précieuse et un écrin de velours, comme à une de ces architectures en miniature, petit temple ou petite chapelle d'orfèvrerie et d'émaux qui servent de reliquaires et qu'on n'expose qu'à de rares jours à la vénération des fidèles. Mais cet instant d'adoration avait déjà trop duré, car le valet de chambre qui tenait d'une main un trousseau de clefs et de l'autre me saluait en touchant sa calotte de sacristain, mais sans la soulever à cause de l'air pur et frais du soir, venait refermer comme ceux d'une châsse les deux battants de la croisée et dérobait à mon adoration le monument réduit et la relique d'or. J'entrai dans ma chambre.

### Passage

Au fur et à mesure que la saison s'avança changea le tableau que j'y trouvais dans la fenêtre. D'abord il faisait grand jour, et sombre seulement s'il faisait mauvais temps ; alors, dans le verre glauque et qu'elle boursouflait de ses vagues rondes, la mer, sertie entre les montants de fer de ma croisée comme dans les plombs d'un vitrail, effilochait sur toute la profonde bordure rocheuse de la baie des triangles empennés d'une immobile écume linéamentée avec la délicatesse d'une plume ou d'un duvet dessinés par Pisanello, et fixés par cet émail blanc, inaltérable et crémeux qui figure une couche de neige dans les verreries de Gallé.

Bientôt les jours diminuèrent et au moment où j'entrais dans la chambre, le ciel violet semblait stigmatisé par la figure raide, géométrique, passagère et fulgurante du soleil (pareille à la représentation de quelque signe miraculeux, de quelque apparition mystique), s'inclinait vers la mer sur la charnière de l'horizon comme un tableau religieux au-dessus du maître-autel, tandis que les parties différentes du couchant exposées dans les glaces des bibliothèques basses en acajou qui couraient le long des murs et que je rapportais par la pensée à la merveilleuse peinture dont elles étaient détachées semblaient comme ces scènes différentes que quelque maître ancien exécuta jadis pour une confrérie sur une châsse, et dont on exhibe à côté les uns des autres dans une salle de musée les volets séparés que l'imagination seule du visiteur remet à leur place sur les prédelles du retable. Quelques semaines plus tard, quand je remontais, le soleil était déjà couché. Pareille à celle que je voyais à Combray au-dessus du Calvaire à mes retours de promenade et quand je m'apprêtais à descendre avant le dîner à la cuisine, une bande de ciel rouge au-dessus de la mer compacte et coupante comme de la gelée de viande, puis bientôt, sur la mer déjà froide et bleue comme le poisson appelé mulet, le ciel, du même rose qu'un de ces saumons que nous nous ferions servir tout à l'heure à Rivebelle, ravivaient le plaisir que j'allais avoir à me mettre en habit pour partir dîner. Sur la mer, tout près du rivage, essayaient de s'élever, les unes par-dessus les autres, à étages de plus en plus larges, des vapeurs d'un noir de suie mais aussi d'un poli, d'une consistance d'agate, d'une pesanteur visible, si bien que les plus élevées penchant au-dessus de la tige déformée et jusqu'en dehors du centre de gravité de celles qui les avaient soutenues jusqu'ici, semblaient sur le point d'entraîner cet échafaudage déjà à demi hauteur du ciel et de le précipiter dans la mer. La vue d'un vaisseau qui s'éloignait comme un voyageur de nuit me donnait cette même impression que j'avais eue en wagon, d'être affranchi des nécessités du sommeil et de la claustration dans une chambre. D'ailleurs je ne me sentais pas emprisonné dans celle où j'étais puisque dans une heure j'allais la quitter pour monter en voiture. Je me jetais sur mon lit ; et, comme si j'avais été sur la couchette d'un des bateaux que je voyais assez près de moi et que la nuit on s'étonnerait de voir se déplacer lentement dans l'obscurité, comme des cygnes assombris et silencieux mais qui ne dorment pas, j'étais de tous côtés entouré des images de la mer.

Mais bien souvent ce n'était, en effet, que des images ; j'oubliais que sous leur couleur se creusait le triste vide de la plage, parcouru par le vent inquiet du soir, que j'avais si anxieusement ressenti à mon arrivée à Balbec ; d'ailleurs, même dans ma chambre, tout occupé des jeunes filles que j'avais vu passer, je n'étais plus dans des dispositions assez calmes ni assez désintéressées pour que pussent se produire en moi des impressions vraiment profondes de beauté. L'attente du dîner à Rivebelle rendait mon humeur plus frivole encore et ma pensée, habitant à ces moments-là la surface de mon corps que j'allais habiller pour tâcher de paraître le plus plaisant possible aux regards féminins qui me dévisageraient dans le restaurant illuminé, était incapable de mettre de la profondeur derrière la couleur des choses. Et si, sous ma fenêtre, le vol inlassable et doux des martinets et des hirondelles n'avait pas monté comme un jet d'eau, comme un feu d'artifice de vie, unissant l'intervalle de ses hautes fusées par la filée immobile et blanche de longs sillages horizontaux, sans le miracle charmant de ce phénomène naturel et local qui rattachait à la réalité les paysages que j'avais devant les yeux, j'aurais pu croire qu'ils n'étaient qu'un choix, chaque jour renouvelé, de peintures qu'on montrait arbitrairement dans l'endroit où je me trouvais et sans qu'elles eussent de rapport nécessaire avec lui. Une fois c'était une exposition d'estampes japonaises : à côté de la mince découpure de soleil rouge et rond comme la lune, un nuage jaune paraissait un lac contre lequel des glaives noirs se profilaient ainsi que les arbres de sa rive, une barre d'un rose tendre que je n'avais jamais revu depuis ma première boîte de couleurs s'enflait comme un fleuve sur les deux rives duquel des bateaux semblaient attendre à sec qu'on vînt les tirer pour les mettre à flot. Et avec le regard dédaigneux, ennuyé et frivole d'un amateur ou d'une femme parcourant, entre deux visites mondaines, une galerie, je me disais : « C'est curieux ce coucher de soleil, c'est différent, mais enfin j'en ai déjà vu d'aussi délicats, d'aussi étonnants que celui-ci. » J'avais plus de plaisir les soirs où un navire absorbé et fluidifié par l'horizon apparaissait tellement de la même couleur que lui, ainsi que dans une toile impressionniste, qu'il semblait aussi de la même matière, comme si on n'eût fait que découper son avant et les cordages en lesquels elle s'était amincie et filigranée dans le bleu vaporeux du ciel. Parfois l'océan emplissait presque toute ma fenêtre, surélevée qu'elle était par une bande de ciel bordée en haut seulement d'une ligne qui était du même bleu que celui de la mer, mais qu'à cause de cela je croyais être la mer encore et ne devant sa couleur différente qu'à un effet d'éclairage. Un autre jour la mer n'était peinte que dans la partie basse de la fenêtre dont tout le reste était rempli de tant de nuages poussés les uns contre les autres par bandes horizontales, que les carreaux avaient l'air, par une préméditation ou une spécialité de l'artiste, de présenter une « étude de nuages », cependant que les différentes vitrines de la bibliothèque montrant des nuages semblables mais dans une autre partie de l'horizon et diversement colorés par la lumière, paraissaient offrir comme la répétition, chère à certains maîtres contemporains, d'un seul et même effet, pris toujours à des heures différentes, mais qui maintenant avec l'immobilité de l'art pouvaient être tous vus ensemble dans une même pièce, exécutés au pastel et mis sous verre. Et parfois sur le ciel et la mer uniformément gris, un peu de rose s'ajoutait avec un raffinement exquis, cependant qu'un petit papillon qui s'était endormi au bas de la fenêtre semblait apposer avec ses ailes, au bas de cette « harmonie gris et rose » dans le goût de celles de Whistler, la signature favorite du maître de Chelsea. Le rose même disparaissait, il n'y avait plus rien à regarder. Je me mettais debout un instant et avant de m'étendre de nouveau je fermais les grands rideaux. Au-dessus d'eux, je voyais de mon lit la raie de clarté qui y restait encore, s'assombrissant, s'amincissant progressivement, mais c'est sans m'attrister et sans lui donner de regret que je laissais ainsi mourir au haut des rideaux l'heure où d'habitude j'étais à table, car je savais que ce jour-ci était d'une autre sorte que les autres, plus long comme ceux du pôle que la nuit interrompt seulement quelques minutes ; je savais que de la chrysalide de ce crépuscule se préparait à sortir, par une radieuse métamorphose, la lumière éclatante du restaurant de Rivebelle. Je me disais : « Il est temps » ; je m'étirais, sur le lit, je me levais, j'achevais ma toilette ; et je trouvais du charme à ces instants inutiles, allégés de tout fardeau matériel, où tandis qu'en bas les autres dînaient, je n'employais les forces accumulées pendant l'inactivité de cette fin de journée qu'à sécher mon corps, à passer un smoking, à attacher ma cravate, à faire tous ces gestes que guidait déjà le plaisir attendu de revoir cette femme que j'avais remarquée la dernière fois à Rivebelle, qui avait paru me regarder, n'était peut-être sortie un instant de table que dans l'espoir que je la suivrais ; c'est avec joie que j'ajoutais à moi tous ces appâts pour me donner entier et dispos à une vie nouvelle, libre, sans souci, où j'appuierais mes hésitations au calme de Saint-Loup et choisirais, entre les espèces de l'histoire naturelle et les provenances de tous les pays, celles qui, composant les plats inusités, aussitôt commandés par mon ami, auraient tenté ma gourmandise ou mon imagination.

Et tout à la fin, les jours vinrent où je ne pouvais plus rentrer de la digue par la salle à manger, ses vitres n'étaient plus ouvertes, car il faisait nuit dehors, et l'essaim des pauvres et des curieux attirés par le flamboiement qu'ils ne pouvaient atteindre pendait, en noires grappes morfondues par la bise, aux parois lumineuses et glissantes de la ruche de verre.

On frappa ; c'était Aimé qui avait tenu à m'apporter lui-même les dernières listes d'étrangers.
