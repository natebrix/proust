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
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur"
      ],
      "presence_type": "implicit",
      "presence_confidence": 0.92
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Les arguments de Norpois (en matière d'art) étaient sans réplique parce qu'ils étaient sans réalité. » Bergotte l'appelle « un vieux serin » et raille « la provision de sottises », Odette le trouve « ennuyeux comme la pluie » et « d'un vaseux » et ajoute qu'il est « très mauvaise langue ».",
      "explanation": "The narrator devalues Norpois's intellectual authority and reports converging jabs from Bergotte and Odette that ridicule him as a conversational partner and speaker. Swann attempts a slight rebalancing, but does not overturn this movement."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "rhetorical_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "His credit as a mind and interlocutor is significantly lowered by the narrator's judgment and the jabs from Bergotte and Odette describing him as empty, boring, and foolish."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-221-p-225"
}

### Candidate characters

[
  "Bergotte",
  "Gilberte",
  "Odette",
  "Swann",
  "la Berma",
  "le narrateur"
]

### Prior local context (optional)

– Non, non, dit Bergotte, sauf dans la scène où elle avoue sa passion à Œnone et où elle fait avec la main le mouvement d'Hégeso dans la stèle du Céramique, c'est un art bien plus ancien qu'elle ranime. Je parlais des Koraï de l'ancien Éréchthéion, et je reconnais qu'il n'y a peut-être rien qui soit aussi loin de l'art de Racine, mais il y a tant déjà de choses dans Phèdre..., une de plus... Oh ! et puis, si, elle est bien jolie la petite Phèdre du VIe siècle, la verticalité du bras, la boucle du cheveu qui « fait marbre », si, tout de même, c'est très fort d'avoir trouvé tout ça. Il y a là beaucoup plus d'antiquité que dans bien des livres qu'on appelle cette année « antiques ».

### Passage

Comme Bergotte avait adressé dans un de ses livres une invocation célèbre à ces statues archaïques, les paroles qu'il prononçait en ce moment étaient fort claires pour moi et me donnaient une nouvelle raison de m'intéresser au jeu de la Berma. Je tâchais de la revoir dans mon souvenir, telle qu'elle avait été dans cette scène où je me rappelais qu'elle avait élevé le bras à la hauteur de l'épaule. Et je me disais : « Voilà l'Hespéride d'Olympie ; voilà la soeur d'une de ces admirables orantes de l'Acropole ; voilà ce que c'est qu'un art noble. » Mais pour que ces pensées pussent m'embellir le geste de la Berma, il aurait fallu que Bergotte me les eût fournies avant la représentation. Alors pendant que cette attitude de l'actrice existait effectivement devant moi, à ce moment où la chose qui a lieu a encore la plénitude de la réalité, j'aurais pu essayer d'en extraire l'idée de sculpture archaïque. Mais de la Berma dans cette scène, ce que je gardais c'était un souvenir qui n'était plus modifiable, mince comme une image dépourvue de ces dessous profonds du présent qui se laissent creuser et d'où l'on peut tirer véridiquement quelque chose de nouveau, une image à laquelle on ne peut imposer rétroactivement une interprétation qui ne serait plus susceptible de vérification, de sanction objective. Pour se mêler à la conversation, Odette me demanda si Gilberte avait pensé à me donner ce que Bergotte avait écrit sur Phèdre. « J'ai une fille si étourdie », ajouta-t-elle. Bergotte eut un sourire de modestie et protesta que c'étaient des pages sans importance. « Mais c'est si ravissant ce petit opuscule, ce petit tract », dit Odette pour se montrer bonne maîtresse de maison, pour faire croire qu'elle avait lu la brochure, et aussi parce qu'elle n'aimait pas seulement complimenter Bergotte, mais faire un choix entre les choses qu'il écrivait, le diriger. Et à vrai dire elle l'inspira, d'une autre façon, du reste, qu'elle ne crut. Mais enfin il y a entre ce que fut l'élégance du salon de Odette et tout un côté de l'oeuvre de Bergotte des rapports tels que chacun des deux peut être alternativement, pour les vieillards d'aujourd'hui, un commentaire de l'autre.

Je me laissais aller à raconter mes impressions. Souvent Bergotte ne les trouvait pas justes, mais il me laissait parler. Je lui dis que j'avais aimé cet éclairage vert qu'il y a au moment où Phèdre lève le bras. « Ah ! vous feriez très plaisir au décorateur qui est un grand artiste, je le lui raconterai parce qu'il est très fier de cette lumière-là. Moi je dois dire que je ne l'aime pas beaucoup, ça baigne tout dans une espèce de machine glauque, la petite Phèdre là dedans fait trop branche de corail au fond d'un aquarium. Vous direz que ça fait ressortir le côté cosmique du drame. Ça c'est vrai. Tout de même ce serait mieux pour une pièce qui se passerait chez Neptune. Je sais bien qu'il y a là de la vengeance de Neptune. Mon Dieu, je ne demande pas qu'on ne pense qu'à Port-Royal, mais enfin, tout de même, ce que Racine a raconté ce ne sont pas les amours des oursins. Mais enfin c'est ce que mon ami a voulu et c'est très fort tout de même et, au fond, c'est assez joli. Oui, enfin vous avez aimé ça, vous avez compris, n'est-ce pas, au fond nous pensons de même là-dessus, c'est un peu insensé ce qu'il a fait, n'est-ce pas, mais enfin c'est très intelligent. » Et quand l'avis de Bergotte était ainsi contraire au mien, il ne me réduisait nullement au silence, à l'impossibilité de rien répondre, comme eût fait celui de Norpois. Cela ne prouve pas que les opinions de Bergotte fussent moins valables que celles de l'Ambassadeur, au contraire. Une idée forte communique un peu de sa force au contradicteur. Participant à la valeur universelle des esprits, elle s'insère, se greffe en l'esprit de celui qu'elle réfute, au milieu d'idées adjacentes, à l'aide desquelles, reprenant quelque avantage, il la complète, la rectifie ; si bien que la sentence finale est en quelque sorte l'oeuvre des deux personnes qui discutaient. C'est aux idées qui ne sont pas, à proprement parler, des idées, aux idées qui ne tenant à rien, ne trouvent aucun point d'appui, aucun rameau fraternel dans l'esprit de l'adversaire, que celui-ci, aux prises avec le pur vide, ne trouve rien à répondre. Les arguments de Norpois (en matière d'art) étaient sans réplique parce qu'ils étaient sans réalité.

Bergotte n'écartant pas mes objections, je lui avouai qu'elles avaient été méprisées par Norpois. « Mais c'est un vieux serin, répondit-il ; il vous a donné des coups de bec parce qu'il croit toujours avoir devant lui un échaudé ou une seiche. – Comment ! vous connaissez Norpois, me dit Swann. – Oh ! il est ennuyeux comme la pluie, interrompit sa femme qui avait grande confiance dans le jugement de Bergotte et craignait sans doute que Norpois ne nous eût dit du mal d'elle. J'ai voulu causer avec lui après le dîner, je ne sais pas si c'est l'âge ou la digestion, mais je l'ai trouvé d'un vaseux. Il semble qu'on aurait eu besoin de le doper ! – Oui, n'est-ce pas, dit Bergotte, il est bien obligé de se taire assez souvent pour ne pas épuiser avant la fin de la soirée la provision de sottises qui empèsent le jabot de la chemise et maintiennent le gilet blanc. – Je trouve Bergotte et ma femme bien sévères, dit Swann qui avait pris chez lui « l'emploi » d'homme de bon sens. Je reconnais que Norpois ne peut pas vous intéresser beaucoup, mais à un autre point de vue (car Swann aimait à recueillir les beautés de la « vie »), il est quelqu'un d'assez curieux, d'assez curieux comme « amant ». Quand il était secrétaire à Rome, ajouta-t-il, après s'être assuré que Gilberte ne pouvait pas entendre, il avait à Paris une maîtresse dont il était éperdu et il trouvait le moyen de faire le voyage deux fois par semaine pour la voir deux heures. C'était du reste une femme très intelligente et ravissante à ce moment-là, c'est une douairière maintenant. Et il en a eu beaucoup d'autres dans l'intervalle. Moi je serais devenu fou s'il avait fallu que la femme que j'aimais habitât Paris pendant que j'étais retenu à Rome. Pour les gens nerveux il faudrait toujours qu'ils aimassent, comme disent les gens du peuple, « au-dessous d'eux » afin qu'une question d'intérêt mît la femme qu'ils aiment à leur discrétion. » À ce moment Swann s'aperçut de l'application que je pouvais faire de cette maxime à lui et à Odette. Et comme même chez les êtres supérieurs, au moment où ils semblent planer avec vous au-dessus de la vie, l'amour-propre reste mesquin, il fut pris d'une mauvaise humeur contre moi. Mais cela ne se manifesta que par l'inquiétude de son regard. Il ne me dit rien au moment même. Il ne faut pas trop s'en étonner. Quand Racine, selon un récit d'ailleurs controuvé, mais dont la matière se répète tous les jours dans la vie de Paris, fit allusion à Scarron devant Louis XIV, le plus puissant roi du monde ne dit rien le soir même au poète. Et c'est le lendemain que celui-ci tomba en disgrâce.

Mais comme une théorie désire d'être exprimée entièrement, Swann, après cette minute d'irritation et ayant essuyé le verre de son monocle, compléta sa pensée en ces mots qui devaient plus tard prendre dans mon souvenir la valeur d'un avertissement prophétique et duquel je ne sus pas tenir compte. « Cependant le danger de ce genre d'amours est que la sujétion de la femme calme un moment la jalousie de l'homme mais la rend aussi plus exigeante. Il arrive à faire vivre sa maîtresse comme ces prisonniers qui sont jour et nuit éclairés pour être mieux gardés. Et cela finit généralement par des drames. »

Je revins à Norpois. « Ne vous y fiez pas, il est au contraire très mauvaise langue », dit Odette avec un accent qui me parut d'autant plus signifier que Norpois avait mal parlé d'elle, que Swann regarda sa femme d'un air de réprimande et comme pour l'empêcher d'en dire davantage.
