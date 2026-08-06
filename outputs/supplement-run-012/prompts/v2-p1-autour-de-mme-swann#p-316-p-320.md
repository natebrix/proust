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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette",
        "Madame Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« enveloppaient Odette de quelque chose de noble »; « On sentait qu'elle ne s'habillait pas seulement pour la commodité… elle était entourée de sa toilette comme de l'appareil délicat et spiritualisé d'une civilisation »; « Madame Swann, n'est-ce pas, c'est toute une époque ? »",
      "explanation": "The narrator exalts Odette's attire and style, conferring nobility, tradition, and individuality, and suggests an implicit social recognition by the 'young people.'"
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Odette is strongly elevated by a long aesthetic and cultural praise of her outfit, presented as noble and significant."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-316-p-320"
}

### Candidate characters

[
  "Françoise",
  "Gilberte",
  "Remi",
  "Swann",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Swann avait dans sa chambre, au lieu des belles photographies qu'on faisait maintenant de sa femme, et où la même expression énigmatique et victorieuse laissait reconnaître, quels que fussent la robe et le chapeau, sa silhouette et son visage triomphants, un petit daguerréotype ancien tout simple, antérieur à ce type, et duquel la jeunesse et la beauté d'Odette, non encore trouvées par elle, semblaient absentes. Mais sans doute Swann, fidèle ou revenu à une conception différente, goûtait-il dans la jeune femme grêle aux yeux pensifs, aux traits las, à l'attitude suspendue entre la marche et l'immobilité, une grâce plus botticellienne. Il aimait encore en effet à voir en sa femme un Botticelli. Odette qui au contraire cherchait non à faire ressortir, mais à compenser, à dissimuler ce qui, en elle-même, ne lui plaisait pas, ce qui était peut-être, pour un artiste, son « caractère », mais que, comme femme, elle trouvait des défauts, ne voulait pas entendre parler de ce le peintre. Swann possédait une merveilleuse écharpe orientale, bleue et rose, qu'il avait achetée parce que c'était exactement celle de la Vierge du Magnificat. Mais Odette ne voulait pas la porter. Une fois seulement elle laissa son mari lui commander une toilette toute criblée de pâquerettes, de bluets, de myosotis et de campanules d'après la Primavera du Printemps. Parfois, le soir, quand elle était fatiguée, il me faisait remarquer tout bas comme elle donnait sans s'en rendre compte à ses mains pensives le mouvement délié, un peu tourmenté de la Vierge qui trempe sa plume dans l'encrier que lui tend l'ange, avant d'écrire sur le livre saint où est déjà tracé le mot Magnificat. Mais il ajoutait : « Surtout ne le lui dites pas, il suffirait qu'elle le sût pour qu'elle fît autrement. »

### Passage

Sauf à ces moments d'involontaire fléchissement où Swann essayait de retrouver la mélancolique cadence botticellienne, le corps d'Odette était maintenant découpé en une seule silhouette cernée tout entière par une « ligne » qui, pour suivre le contour de la femme, avait abandonné les chemins accidentés, les rentrants et les sortants factices, les lacis, l'éparpillement composite des modes d'autrefois, mais qui aussi, là où c'était l'anatomie qui se trompait en faisant des détours inutiles en deçà ou au delà du tracé idéal, savait rectifier d'un trait hardi les écarts de la nature, suppléer, pour toute une partie du parcours, aux défaillances aussi bien de la chair que des étoffes. Les coussins, le « strapontin » de l'affreuse « tournure » avaient disparu ainsi que ces corsages à basques qui, dépassant la jupe et raidis par des baleines, avaient ajouté si longtemps à Odette un ventre postiche et lui avaient donné l'air d'être composée de pièces disparates qu'aucune individualité ne reliait. La verticale des « effilés » et la courbe des ruches avaient cédé la place à l'inflexion d'un corps qui faisait palpiter la soie comme la sirène bat l'onde et donnait à la percaline une expression humaine, maintenant qu'il s'était dégagé, comme une forme organisée et vivante, du long chaos et de l'enveloppement nébuleux des modes détrônées. Mais Odette cependant avait voulu, avait su garder un vestige de certaines d'entre elles, au milieu même de celles qui les avaient remplacées. Quand le soir, ne pouvant travailler et étant assuré que Gilberte était au théâtre avec des amies, j'allais à l'improviste chez ses parents, je trouvais souvent Odette dans quelque élégant déshabillé dont la jupe, d'un de ces beaux tons sombres, rouge foncé ou orange, qui avaient l'air d'avoir une signification particulière parce qu'ils n'étaient plus à la mode, était obliquement traversée d'une rampe ajourée et large de dentelle noire qui faisait penser aux volants d'autrefois. Quand par un jour encore froid de printemps elle m'avait, avant ma brouille avec sa fille, emmené au Jardin d'Acclimatation, sous sa veste qu'elle entr'ouvrait plus ou moins selon qu'elle se réchauffait en marchant, le « dépassant » en dents de scie de sa chemisette avait l'air du revers entrevu de quelque gilet absent, pareil à l'un de ceux qu'elle avait portés quelques années plus tôt et dont elle aimait que les bords eussent ce léger déchiquetage ; et sa cravate – de cet « écossais » auquel elle était restée fidèle, mais en adoucissant tellement les tons (le rouge devenu rose et le bleu lilas) que l'on aurait presque cru à un de ces taffetas gorge de pigeon qui étaient la dernière nouveauté – était nouée de telle façon sous son menton, sans qu'on pût voir où elle était attachée, qu'on pensait invinciblement à ces « brides » de chapeaux qui ne se portaient plus. Pour peu qu'elle sût « durer » encore quelque temps ainsi, les jeunes gens, essayant de comprendre ses toilettes, diraient : « Madame Swann, n'est-ce pas, c'est toute une époque ? » Comme dans un beau style qui superpose des formes différentes et que fortifie une tradition cachée, dans la toilette de Odette, ces souvenirs incertains de gilets, ou de boucles, parfois une tendance aussitôt réprimée au « saute en barque », et jusqu'à une allusion lointaine et vague au « suivez-moi jeune homme », faisaient circuler sous la forme concrète la ressemblance inachevée d'autres plus anciennes qu'on n'aurait pu y trouver effectivement réalisées par la couturière ou la modiste, mais auxquelles on pensait sans cesse, et enveloppaient Odette de quelque chose de noble – peut-être parce que l'inutilité même de ces atours faisait qu'ils semblaient répondre à un but plus qu'utilitaire, peut-être à cause du vestige conservé des années passées, ou encore d'une sorte d'individualité vestimentaire, particulière à cette femme et qui donnait à ses mises les plus différentes un même air de famille. On sentait qu'elle ne s'habillait pas seulement pour la commodité ou la parure de son corps ; elle était entourée de sa toilette comme de l'appareil délicat et spiritualisé d'une civilisation.

Quand Gilberte, qui d'habitude donnait ses goûters le jour où recevait sa mère, devait au contraire être absente et qu'à cause de cela je pouvais aller au « Choufleury » de Odette, je la trouvais vêtue de quelque belle robe, certaines en taffetas, d'autres en faille, ou en velours, ou en crêpe de Chine, ou en satin, ou en soie, et qui non point lâches comme les déshabillés qu'elle revêtait ordinairement à la maison, mais combinées comme pour la sortie au dehors, donnaient cet après-midi-là à son oisiveté chez elle quelque chose d'alerte et d'agissant. Et sans doute la simplicité hardie de leur coupe était bien appropriée à sa taille et à ses mouvements dont les manches avaient l'air d'être la couleur, changeante selon les jours ; on aurait dit qu'il y avait soudain de la décision dans le velours bleu, une humeur facile dans le taffetas blanc, et qu'une sorte de réserve suprême et pleine de distinction dans la façon d'avancer le bras avait, pour devenir visible, revêtu l'apparence brillante du sourire des grands sacrifices, du crêpe de Chine noir. Mais en même temps, à ces robes si vives la complication des « garnitures » sans utilité pratique, sans raison d'être visible, ajoutait quelque chose de désintéressé, de pensif, de secret, qui s'accordait à la mélancolie que Odette gardait toujours au moins dans la cernure de ses yeux et les phalanges de ses mains. Sous la profusion des porte-bonheur en saphir, des trèfles à quatre feuilles d'émail, des médailles d'argent, des médaillons d'or, des amulettes de turquoise, des chaînettes de rubis, des châtaignes de topaze, il y avait dans la robe elle-même tel dessin colorié poursuivant sur un empiècement rapporté son existence antérieure, telle rangée de petits boutons de satin qui ne boutonnaient rien et ne pouvaient pas se déboutonner, une soutache cherchant à faire plaisir avec la minutie, la discrétion d'un rappel délicat, lesquels, tout autant que les bijoux, avaient l'air – n'ayant sans cela aucune justification possible – de déceler une intention, d'être un gage de tendresse, de retenir une confidence, de répondre à une superstition, de garder le souvenir d'une guérison, d'un voeu, d'un amour ou d'une philippine. Et parfois, dans le velours bleu du corsage un soupçon de crevé Henri II, dans la robe de satin noir un léger renflement qui, soit aux manches, près des épaules, faisaient penser aux « gigots » 1830, soit, au contraire, sous la jupe « aux paniers » Louis XV, donnaient à la robe un air imperceptible d'être un costume, et en insinuant sous la vie présente comme une réminiscence indiscernable du passé, mêlaient à la personne de Odette le charme de certaines héroïnes historiques ou romanesques. Et si je lui faisais remarquer : « Je ne joue pas au golf comme plusieurs de mes amies, disait-elle. Je n'aurais aucune excuse à être comme elles, vêtues de sweaters. »

Dans la confusion du salon, revenant de reconduire une visite, ou prenant une assiette de gâteaux pour les offrir à une autre, Odette en passant près de moi, me prenait une seconde à part : « Je suis spécialement chargée par Gilberte de vous inviter à déjeuner pour après-demain. Comme je n'étais pas certaine de vous voir, j'allais vous écrire si vous n'étiez pas venu. » Je continuais à résister. Et cette résistance me coûtait de moins en moins, parce qu'on a beau aimer le poison qui vous fait du mal, quand on en est privé par quelque nécessité, depuis déjà un certain temps, on ne peut pas ne pas attacher quelque prix au repos qu'on ne connaissait plus, à l'absence d'émotions et de souffrances. Si l'on n'est pas tout à fait sincère en se disant qu'on ne voudra jamais revoir celle qu'on aime, on ne le serait pas non plus en disant qu'on veut la revoir. Car, sans doute, on ne peut supporter son absence qu'en se la promettant courte, en pensant au jour où on se retrouvera, mais d'autre part on sent à quel point ces rêves quotidiens d'une réunion prochaine et sans cesse ajournée sont moins douloureux que ne serait une entrevue qui pourrait être suivie de jalousie, de sorte que la nouvelle qu'on va revoir celle qu'on aime donnerait une commotion peu agréable. Ce qu'on recule maintenant de jour en jour, ce n'est plus la fin de l'intolérable anxiété causée par la séparation, c'est le recommencement redouté d'émotions sans issue. Comme à une telle entrevue on préfère le souvenir docile qu'on complète à son gré de rêveries où celle qui, dans la réalité, ne vous aime pas vous fait au contraire des déclarations, quand vous êtes tout seul ; ce souvenir qu'on peut arriver, en y mêlant peu à peu beaucoup de ce qu'on désire, à rendre aussi doux qu'on veut, comme on le préfère à l'entretien ajourné où on aurait affaire à un être à qui on ne dicterait plus à son gré les paroles qu'on désire, mais dont on subirait les nouvelles froideurs, les violences inattendues. Nous savons tous, quand nous n'aimons plus, que l'oubli, même le souvenir vague ne causent pas tant de souffrances que l'amour malheureux. C'est d'un tel oubli anticipé que je préférais sans me l'avouer, la reposante douceur.

D'ailleurs, ce qu'une telle cure de détachement psychique et d'isolement peut avoir de pénible le devient de moins en moins pour une autre raison, c'est qu'elle affaiblit, en attendant de la guérir, cette idée fixe qu'est un amour. Le mien était encore assez fort pour que je tinsse à reconquérir tout mon prestige aux yeux de Gilberte, lequel, par ma séparation volontaire, devait, me semblait-il, grandir progressivement, de sorte que chacune de ces calmes et tristes journées où je ne la voyais pas, venant chacune après l'autre, sans interruption, sans prescription (quand un fâcheux ne se mêlait pas de mes affaires), était une journée non pas perdue, mais gagnée. Inutilement gagnée peut-être, car bientôt on pourrait me déclarer guéri. La résignation, modalité de l'habitude, permet à certaines forces de s'accroître indéfiniment. Celles si infimes, que j'avais pour supporter mon chagrin, le premier soir de ma brouille avec Gilberte, avaient été portées depuis lors à une puissance incalculable. Seulement la tendance de tout ce qui existe à se prolonger est parfois coupée de brusques impulsions auxquelles nous nous concédons avec d'autant moins de scrupules de nous laisser aller que nous savons pendant combien de jours, de mois, nous avons pu, nous pourrions encore, nous priver. Et souvent, c'est quand la bourse où l'on épargne va être pleine qu'on la vide tout d'un coup, c'est sans attendre le résultat du traitement et quand déjà on s'est habitué à lui, qu'on le cesse. Et un jour où Odette me redisait ses habituelles paroles sur le plaisir que Gilberte aurait à me voir, mettant ainsi le bonheur dont je me privais déjà depuis si longtemps comme à la portée de ma main, je fus bouleversé en comprenant qu'il était encore possible de le goûter ; et j'eus peine à attendre le lendemain ; je venais de me résoudre à aller surprendre Gilberte avant son dîner.

Ce qui m'aida à patienter tout l'espace d'une journée fut un projet que je fis. Du moment que tout était oublié, que j'étais réconcilié avec Gilberte, je ne voulais plus la voir qu'en amoureux. Tous les jours elle recevrait de moi les plus belles fleurs qui fussent. Et si Odette, bien qu'elle n'eût pas le droit d'être une mère trop sévère, ne me permettait pas des envois de fleurs quotidiens, je trouverais des cadeaux plus précieux et moins fréquents. Mes parents ne me donnaient pas assez d'argent pour acheter des choses chères. Je songeai à une grande potiche de vieux Chine qui me venait de ma tante Léonie et dont maman prédisait chaque jour que Françoise allait venir en lui disant : « A s'est décollée » et qu'il n'en resterait rien. Dans ces conditions n'était-il pas plus sage de la vendre, de la vendre pour pouvoir faire tout le plaisir que je voudrais à Gilberte. Il me semblait que je pourrais bien en tirer mille francs. Je la fis envelopper, l'habitude m'avait empêché de jamais la voir ; m'en séparer eut au moins un avantage qui fut de me faire faire sa connaissance. Je l'emportai avec moi avant d'aller chez les Swann, et en donnant leur adresse au cocher, je lui dis de prendre par les Champs-Élysées, au coin desquels était le magasin d'un grand marchand de chinoiseries que connaissait mon père. À ma grande surprise, il m'offrit séance tenante de la potiche non pas mille, mais dix mille francs. Je pris ces billets avec ravissement ; pendant toute une année, je pourrais combler chaque jour Gilberte de roses et de lilas. Quand je fus remonté dans la voiture en quittant le marchand, le cocher, tout naturellement, comme les Swann demeuraient près du Bois, se trouva, au lieu du chemin habituel, descendre l'avenue des Champs-Élysées. Il avait déjà dépassé le coin de la rue de Berri, quand, dans le crépuscule, je crus reconnaître, très près de la maison des Swann mais allant dans la direction inverse et s'en éloignant, Gilberte qui marchait lentement, quoique d'un pas délibéré, à côté d'un jeune homme avec qui elle causait et duquel je ne pus distinguer le visage. Je me soulevai dans la voiture, voulant faire arrêter, puis j'hésitai. Les deux promeneurs étaient déjà un peu loin et les deux lignes douces et parallèles que traçait leur lente promenade allaient s'estompant dans l'ombre élyséenne. Bientôt j'arrivai devant la maison de Gilberte. Je fus reçu par Odette : « Oh ! elle va être désolée, me dit-elle, je ne sais pas comment elle n'est pas là. Elle a eu très chaud tantôt à un cours, elle m'a dit qu'elle voulait aller prendre un peu l'air avec une de ses amies. – Je crois que je l'ai aperçue avenue des Champs-Élysées. – Je ne pense pas que ce fût elle. En tous cas ne le dites pas à son père, il n'aime pas qu'elle sorte à ces heures-là. Good evening. » Je partis, dis au cocher de reprendre le même chemin, mais ne retrouvai pas les deux promeneurs. Où avaient-ils été ? Que se disaient-ils dans le soir, de cet air confidentiel ?
