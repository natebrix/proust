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
        "La Marquise de Villeparisis",
        "Mme de Villeparisis",
        "cette dame"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère",
        "la vieille dame",
        "ma grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "la grand-mère",
      "target": "Mme de Villeparisis",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.72,
      "evidence": "elle se contenta de détourner les yeux et eut l'air de ne pas voir Mme de Villeparisis qui, comprenant ... regarda à son tour dans le vague. Elle s'éloigna, et je restai dans mon isolement.",
      "explanation": "Following her travel principle of incognito, the grandmother feigns not to recognize Mme de Villeparisis, prompting a reciprocal non-recognition that forecloses the anticipated prestige-by-association."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.68,
      "explanation": "By declining recognition, she self-excludes from a valuable social link in the hotel, maintaining distance rather than gaining association."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-61-p-65"
}

### Candidate characters

[
  "Françoise",
  "Legrandin",
  "M. de Stermaria",
  "Mlle de Stermaria",
  "Odette",
  "Swann",
  "le directeur",
  "le narrateur",
  "marquis de Cambremer"
]

### Prior local context (optional)

Car j'avais remarqué sa fille dès son entrée, son joli visage pâle et presque bleuté, ce qu'il y avait de particulier dans le port de sa haute taille, dans sa démarche, et qui m'évoquait avec raison son hérédité, son éducation aristocratique et d'autant plus clairement que je savais son nom – comme ces thèmes expressifs inventés par des musiciens de génie et qui peignent splendidement le scintillement de la flamme, le bruissement du fleuve et la paix de la campagne, pour les auditeurs qui, en parcourant préalablement le livret, ont aiguillé leur imagination dans la bonne voie. La « race », en ajoutant aux charmes de Mlle de Stermaria l'idée de leur cause, les rendait plus intelligibles, plus complets. Elle les faisait aussi plus désirables, annonçant qu'ils étaient peu accessibles, comme un prix élevé ajoute à la valeur d'un objet qui nous a plu. Et la tige héréditaire donnait à ce teint composé de sucs choisis la saveur d'un fruit exotique ou d'un cru célèbre.

### Passage

Or, un hasard mit tout d'un coup entre nos mains le moyen de nous donner à ma grand'mère et à moi, pour tous les habitants de l'hôtel, un prestige immédiat. En effet, dès ce premier jour, au moment où la vieille dame descendait de chez elle, exerçant, grâce au valet de pied qui la précédait, à la femme de chambre qui courait derrière avec un livre et une couverture oubliés, une action sur les âmes et excitant chez tous une curiosité et un respect auxquels il fut visible qu'échappait moins que personne M. de Stermaria, le directeur se pencha vers ma grand'mère, et par amabilité (comme on montre le Shah de Perse ou la Reine Ranavalo à un spectateur obscur qui ne peut évidemment avoir aucune relation avec le puissant souverain, mais peut trouver intéressant de l'avoir vu à quelques pas), il lui coula dans l'oreille : « La Marquise de Villeparisis », cependant qu'au même moment cette dame apercevant ma grand'mère ne pouvait retenir un regard de joyeuse surprise.

On peut penser que l'apparition soudaine, sous les traits d'une petite vieille, de la plus puissante des fées ne m'aurait pas causé plus de plaisir, dénué comme j'étais de tout recours pour m'approcher de Mlle de Stermaria, dans un pays où je ne connaissais personne. J'entends personne au point de vue pratique. Esthétiquement, le nombre des types humains est trop restreint pour qu'on n'ait pas bien souvent, dans quelque endroit qu'on aille, la joie de revoir des gens de connaissance, même sans les chercher dans les tableaux des vieux maîtres, comme faisait Swann. C'est ainsi que dès les premiers jours de notre séjour à Balbec, il m'était arrivé de rencontrer Legrandin, le concierge de Swann, et Odette elle-même, devenus, le premier, garçon de café, le second un étranger de passage que je ne revis pas, et la dernière un maître baigneur. Et une sorte d'aimantation attire et retient si inséparablement les uns auprès les autres certains caractères de physionomie et de mentalité que quand la nature introduit ainsi une personne dans un nouveau corps elle ne la mutile pas trop. Legrandin changé en garçon de café gardait intacts sa stature, le profil de son nez et une partie du menton ; Odette dans le sexe masculin et la condition de maître baigneur avait été suivie non seulement par sa physionomie habituelle, mais même par une certaine manière de parler. Seulement elle ne pouvait pas m'être de plus d'utilité entourée de sa ceinture rouge et hissant, à la moindre houle, le drapeau qui interdit les bains, car les maîtres baigneurs sont prudents, sachant rarement nager, qu'elle ne l'eût pu dans la fresque de la Vie de Moïse où Swann l'avait reconnue jadis sous les traits de la fille de Jethro. Tandis que cette Mme de Villeparisis était bien la véritable, elle n'avait pas été victime d'un enchantement qui l'eût dépouillée de sa puissance, mais était capable au contraire d'en mettre un à la disposition de la mienne qu'il centuplerait, et grâce auquel, comme si j'avais été porté par les ailes d'un oiseau fabuleux, j'allais franchir en quelques instants les distances sociales infinies, au moins à Balbec, qui me séparaient de Mlle de Stermaria.

Malheureusement, s'il y avait quelqu'un qui, plus que quiconque, vécût enfermé dans son univers particulier, c'était ma grand'mère. Elle ne m'aurait même pas méprisé, elle ne m'aurait pas compris, si elle avait su que j'attachais de l'importance à l'opinion, que j'éprouvais de l'intérêt pour la personne de gens dont elle ne remarquait seulement pas l'existence et dont elle devait quitter Balbec sans avoir retenu le nom ; je n'osais pas lui avouer que si ces mêmes gens l'avaient vu causer avec Mme de Villeparisis, j'en aurais eu un grand plaisir, parce que je sentais que la marquise avait du prestige dans l'hôtel et que son amitié nous eût posés aux yeux de M. de Stermaria. Non d'ailleurs que l'amie de ma grand'mère me représentât le moins du monde une personne de l'aristocratie : j'étais trop habitué à son nom devenu familier à mes oreilles avant que mon esprit s'arrêtât sur lui, quand tout enfant je l'entendais prononcer à la maison ; et son titre n'y ajoutait qu'une particularité bizarre comme aurait fait un prénom peu usité, ainsi qu'il arrive dans les noms de rue où on n'aperçoit rien de plus noble dans la rue Lord-Byron, dans la si populaire et vulgaire rue Rochechouart, ou dans la rue de Gramont que dans la rue Léonce-Reynaud ou la rue Hippolyte-Lebas. Mme de Villeparisis ne me faisait pas plus penser à une personne d'un monde spécial que son cousin Mac Mahon que je ne différenciais pas de M. Carnot, président de la République comme lui, et de Raspail dont Françoise avait acheté la photographie avec celle de Pie IX. Ma grand'mère avait pour principe qu'en voyage on ne doit plus avoir de relations, qu'on ne va pas au bord de la mer pour voir des gens, qu'on a tout le temps pour cela à Paris, qu'ils vous feraient perdre en politesses, en banalités, le temps précieux qu'il faut passer tout entier au grand air, devant les vagues ; et trouvant plus commode de supposer que cette opinion était partagée par tout le monde et qu'elle autorisait entre de vieux amis que le hasard mettait en présence dans le même hôtel la fiction d'un incognito réciproque, au nom que lui cita le directeur, elle se contenta de détourner les yeux et eut l'air de ne pas voir Mme de Villeparisis qui, comprenant que ma grand'mère ne tenait pas à faire de reconnaissances, regarda à son tour dans le vague. Elle s'éloigna, et je restai dans mon isolement comme un naufragé de qui a paru s'approcher un vaisseau, lequel a disparu ensuite sans s'être arrêté.

Elle prenait aussi ses repas dans la salle à manger, mais à l'autre bout. Elle ne connaissait aucune des personnes qui habitaient l'hôtel ou y venaient en visite, pas même M. de Cambremer ; en effet, je vis qu'il ne la saluait pas, un jour où il avait accepté avec sa femme une invitation à déjeuner du bâtonnier, lequel, ivre de l'honneur d'avoir le gentilhomme à sa table, évitait ses amis des autres jours et se contentait de leur adresser de loin un clignement d'oeil pour faire à cet événement historique une allusion toutefois assez discrète pour qu'elle ne pût pas être interprétée comme une invite à s'approcher.

– Eh bien, j'espère que vous vous mettez bien, que vous êtes un homme chic, lui dit le soir la femme du premier président.
