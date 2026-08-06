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
      "canonical_name": "duchesse de Guermantes",
      "surface_forms": [
        "duchesse de Guermantes",
        "Mme des Laumes"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    },
    {
      "canonical_name": "marquise de Gallardon",
      "surface_forms": [
        "marquise de Gallardon"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "duchesse de Guermantes",
      "target": "marquise de Gallardon",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Elle rit: « Mais le mieux du monde ! »; regard « étonné et rieur » signifiant qu’elle n’avait pas autorisé Gallardon à l’appeler par son prénom; « Mais je connais ce quintette, je peux te dire tout de suite... que je l’aime ! »",
      "explanation": "Through mocking laughter, a sign of distancing on Gallardon's familiarity, followed by a response that dispenses with visiting her by giving her opinion on the spot, the duchess belittles and dismisses Gallardon. The narrator additionally emphasizes Gallardon's pleading posture, reinforcing the effect of snub."
    }
  ],
  "status_effects": [
    {
      "character": "duchesse de Guermantes",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She imposes her superiority through wit and disguised refusal, controlling the exchange with ease."
    },
    {
      "character": "marquise de Gallardon",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "She appears dependent on the duchess's opinion and is politely but clearly kept at a distance and rejected."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-451-p-455"
}

### Candidate characters

[
  "le narrateur"
]

### Prior local context (optional)

Vivant dans une famille provinciale qui avait peu de relations, n'allant guère au bal, elle s'était grisée dans la solitude de son manoir, à ralentir, à précipiter la danse de tous ces couples imaginaires, à les égrener comme des fleurs, à quitter un moment le bal pour entendre le vent souffler dans les sapins, au bord du lac, et à y voir tout d'un coup s'avancer, plus différent de tout ce qu'on a jamais rêvé que ne sont les amants de la terre, un mince jeune homme à la voix un peu chantante, étrangère et fausse, en gants blancs. Mais aujourd'hui la beauté démodée de cette musique semblait défraîchie. Privée depuis quelques années de l'estime des connaisseurs, elle avait perdu son honneur et son charme et ceux mêmes dont le goût est mauvais n'y trouvaient plus qu'un plaisir inavoué et médiocre. Mme de Cambremer jeta un regard furtif derrière elle. Elle savait que sa jeune bru (pleine de respect pour sa nouvelle famille, sauf en ce qui touchait les choses de l'esprit sur lesquelles, sachant jusqu'à l'harmonie et jusqu'au grec, elle avait des lumières spéciales) méprisait Chopin et souffrait quand elle en entendait jouer. Mais loin de la surveillance de cette wagnérienne qui était plus loin avec un groupe de personnes de son âge, Mme de Cambremer se laissait aller à des impressions délicieuses. La princesse des Laumes les éprouvait aussi. Sans être par nature douée pour la musique, elle avait reçu il y a quinze ans les leçons qu'un professeur de piano du faubourg Saint-Germain, femme de génie qui avait été à la fin de sa vie réduite à la misère, avait recommencé, à l'âge de soixante-dix ans, à donner aux filles et aux petites-filles de ses anciennes élèves. Elle était morte aujourd'hui. Mais sa méthode, son beau son, renaissaient parfois sous les doigts de ses élèves, même de celles qui étaient devenues pour le reste des personnes médiocres, avaient abandonné la musique et n'ouvraient presque plus jamais un piano. Aussi Mme des Laumes put-elle secouer la tête, en pleine connaissance de cause, avec une appréciation juste de la façon dont le pianiste jouait ce prélude qu'elle savait par coeur. La fin de la phrase commencée chanta d'elle-même sur ses lèvres. Et elle murmura « c'est toujours charmant », avec un double ch au commencement du mot qui était une marque de délicatesse et dont elle sentait ses lèvres si romanesquement froissées comme une belle fleur, qu'elle harmonisa instinctivement son regard avec elles en lui donnant à ce moment-là une sorte de sentimentalité et de vague. Cependant marquise de Gallardon était en train de se dire qu'il était fâcheux qu'elle n'eût que bien rarement l'occasion de rencontrer la princesse des Laumes, car elle souhaitait lui donner une leçon en ne répondant pas à son salut. Elle ne savait pas que sa cousine fût là. Un mouvement de tête de Mme de Franquetot la lui découvrit. Aussitôt elle se précipita vers elle en dérangeant tout le monde ; mais désireuse de garder un air hautain et glacial qui rappelât à tous qu'elle ne désirait pas avoir de relations avec une personne chez qui on pouvait se trouver nez à nez avec la princesse Mathilde, et au-devant de qui elle n'avait pas à aller car elle n'était pas « sa contemporaine », elle voulut pourtant compenser cet air de hauteur et de réserve par quelque propos qui justifiât sa démarche et forçât la princesse à engager la conversation ; aussi une fois arrivée près de sa cousine, marquise de Gallardon, avec un visage dur, une main tendue comme une carte forcée, lui dit : « Comment va ton mari ? » de la même voix soucieuse que si le prince avait été gravement malade. La princesse éclatant d'un rire qui lui était particulier et qui était destiné à la fois à montrer aux autres qu'elle se moquait de quelqu'un et aussi à se faire paraître plus jolie en concentrant les traits de son visage autour de sa bouche animée et de son regard brillant, lui répondit :

### Passage

– Mais le mieux du monde !

Et elle rit encore. Cependant tout en redressant sa taille et refroidissant sa mine, inquiète encore pourtant de l'état du prince, Mme de Gallardon dit à sa cousine :

– Mme de Guermantes (ici Mme des Laumes regarda d'un air étonné et rieur un tiers invisible vis-à-vis duquel elle semblait tenir à attester qu'elle n'avait jamais autorisé Mme de Gallardon à l'appeler par son prénom), je tiendrais beaucoup à ce que tu viennes un moment demain soir chez moi entendre un quintette avec clarinette de Mozart. Je voudrais avoir ton appréciation.

Elle semblait non pas adresser une invitation, mais demander un service, et avoir besoin de l'avis de la princesse sur le quintette de Mozart, comme si ç'avait été un plat de la composition d'une nouvelle cuisinière sur les talents de laquelle il lui eût été précieux de recueillir l'opinion d'un gourmet.

– Mais je connais ce quintette, je peux te dire tout de suite... que je l'aime !
