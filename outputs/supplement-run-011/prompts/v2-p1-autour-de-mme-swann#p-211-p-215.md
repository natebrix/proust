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
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bergotte",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« il avait été original et créateur comme écrivain »; « il y avait dans le style de Bergotte une sorte d'harmonie »; son « sévérité de goût... était au contraire le secret de sa force »; des plus jeunes « employaient les mêmes adverbes, les mêmes prépositions qu'il répétait sans cesse ».",
      "explanation": "The narrator elevates Bergotte as an original and influential writer: his style is harmonious, his rigor is the source of his strength, and young authors inadvertently reproduce his tics and constructions."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Bergotte",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« bien des années plus tard, quand il n'eut plus de talent... il se répéta... “c'est assez exact, cela n'est pas inutile à mon pays” »; « il continuait à simuler la déférence envers des écrivains médiocres pour arriver à être prochainement académicien »; « des ruses de gentleman voleur de fourchettes... pour se rapprocher du fauteuil académique ». ",
      "explanation": "The narrator diminishes him by showing his late artistic decadence and his ambitious and concealed maneuvers for the Academy, as well as a strategic deference towards less gifted people."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Locally elevated as an original, harmonious writer and shaper of his time."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-211-p-215"
}

### Candidate characters

[
  "Norpois",
  "Odette",
  "le narrateur"
]

### Prior local context (optional)

Ces jeunes Bergotte – le futur écrivain et ses frères et soeurs – n'étaient sans doute pas supérieurs, au contraire, à des jeunes gens plus fins, plus spirituels qui trouvaient les Bergotte bien bruyants, voire un peu vulgaires, agaçants dans leurs plaisanteries qui caractérisaient le « genre » moitié prétentieux, moitié bêta, de la maison. Mais le génie, même le grand talent, vient moins d'éléments intellectuels et d'affinement spécial supérieurs à ceux d'autrui, que de la faculté de les transformer, de les transposer. Pour faire chauffer un liquide avec une lampe électrique, il ne s'agit pas d'avoir la plus forte lampe possible, mais une dont le courant puisse cesser d'éclairer, être dérivé et donner, au lieu de lumière, de la chaleur. Pour se promener dans les airs, il n'est pas nécessaire d'avoir l'automobile la plus puissante, mais une automobile qui ne continuant pas de courir à terre et coupant d'une verticale la ligne qu'elle suivait soit capable de convertir en force ascensionnelle sa vitesse horizontale. De même ceux qui produisent des oeuvres géniales ne sont pas ceux qui vivent dans le milieu le plus délicat, qui ont la conversation la plus brillante, la culture la plus étendue, mais ceux qui ont eu le pouvoir, cessant brusquement de vivre pour eux-mêmes, de rendre leur personnalité pareille à un miroir, de telle sorte que leur vie, si médiocre d'ailleurs qu'elle pouvait être mondainement et même, dans un certain sens, intellectuellement parlant, s'y reflète, le génie consistant dans le pouvoir réfléchissant et non dans la qualité intrinsèque du spectacle reflété. Le jour où le jeune Bergotte put montrer au monde de ses lecteurs le salon de mauvais goût où il avait passé son enfance et les causeries pas très drôles qu'il y tenait avec ses frères, ce jour-là il monta plus haut que les amis de sa famille, plus spirituels et plus distingués : ceux-ci dans leurs belles Rolls-Royce pourraient rentrer chez eux en témoignant un peu de mépris pour la vulgarité des Bergotte ; mais lui, de son modeste appareil qui venait enfin de « décoller », il les survolait.

### Passage

C'était, non plus avec des membres de sa famille, mais avec certains écrivains de son temps que d'autres traits de son élocution lui étaient communs. De plus jeunes qui commençaient à le renier et prétendaient n'avoir aucune parenté intellectuelle avec lui, la manifestaient sans le vouloir en employant les mêmes adverbes, les mêmes prépositions qu'il répétait sans cesse, en construisant les phrases de la même manière, en parlant sur le même ton amorti, ralenti, par réaction contre le langage éloquent et facile d'une génération précédente. Peut-être ces jeunes gens – on en verra qui étaient dans ce cas – n'avaient-ils pas connu Bergotte. Mais sa façon de penser, inoculée en eux, y avait développé ces altérations de la syntaxe et de l'accent qui sont en relation nécessaire avec l'originalité intellectuelle. Relation qui demande à être interprétée d'ailleurs. Ainsi Bergotte, s'il ne devait rien à personne dans sa façon d'écrire, tenait sa façon de parler d'un de ses vieux camarades, merveilleux causeur dont il avait subi l'ascendant, qu'il imitait sans le vouloir dans la conversation, mais qui, lui, étant moins doué, n'avait jamais écrit de livres vraiment supérieurs. De sorte que si l'on s'en était tenu à l'originalité du débit, Bergotte eût été étiqueté disciple, écrivain de seconde main, alors que, influencé par son ami dans le domaine de la causerie, il avait été original et créateur comme écrivain. Sans doute encore pour se séparer de la précédente génération, trop amie des abstractions, des grands lieux communs, quand Bergotte voulait dire du bien d'un livre, ce qu'il faisait valoir, ce qu'il citait c'était toujours quelque scène faisant image, quelque tableau sans signification rationnelle. « Ah ! si ! disait-il, c'est bien ! il y a une petite fille en châle orange, ah ! c'est bien », ou encore : « Oh ! oui, il y a un passage où il y a un régiment qui traverse la ville, ah ! oui, c'est bien ! » Pour le style, il n'était pas tout à fait de son temps (et restait du reste fort exclusivement de son pays, il détestait Tolstoï, George Eliot, Ibsen et Dostoïewski) car le mot qui revenait toujours quand il voulait faire l'éloge d'un style, c'était le mot « doux ». « Si, j'aime, tout de même mieux le Chateaubriand d'Atala que celui de René, il me semble que c'est plus doux. » Il disait ce mot-là comme un médecin à qui un malade assure que le lait lui fait mal à l'estomac et qui répond : « C'est pourtant bien doux. » Et il est vrai qu'il y avait dans le style de Bergotte une sorte d'harmonie pareille à celle pour laquelle les anciens donnaient à certains de leurs orateurs des louanges dont nous concevons difficilement la nature, habitués que nous sommes à nos langues modernes où on ne cherche pas ce genre d'effets.

Il disait aussi, avec un sourire timide, de pages de lui pour lesquelles on lui déclarait son admiration : « Je crois que c'est assez vrai, c'est assez exact, cela peut être utile », mais simplement par modestie, comme à une femme à qui on dit que sa robe, ou sa fille, est ravissante, répond, pour la première : « Elle est commode », pour la seconde : « Elle a un bon caractère ». Mais l'instinct du constructeur était trop profond chez Bergotte pour qu'il ignorât que la seule preuve qu'il avait bâti utilement et selon la vérité, résidait dans la joie que son oeuvre lui avait donnée, à lui d'abord, et aux autres ensuite. Seulement bien des années plus tard, quand il n'eut plus de talent, chaque fois qu'il écrivit quelque chose dont il n'était pas content, pour ne pas l'effacer comme il aurait dû, pour le publier, il se répéta, à soi-même cette fois : « Malgré tout, c'est assez exact, cela n'est pas inutile à mon pays. » De sorte que la phrase murmurée jadis devant ses admirateurs par une ruse de sa modestie, le fut, à la fin, dans le secret de son coeur, par les inquiétudes de son orgueil. Et les mêmes mots qui avaient servi à Bergotte d'excuse superflue pour la valeur de ses premières oeuvres, lui devinrent comme une inefficace consolation de la médiocrité des dernières.

Une espèce de sévérité de goût qu'il avait, de volonté de n'écrire jamais que des choses dont il pût dire : « C'est doux », et qui l'avait fait passer tant d'années pour un artiste stérile, précieux, ciseleur de riens, était au contraire le secret de sa force, car l'habitude fait aussi bien le style de l'écrivain que le caractère de l'homme, et l'auteur qui s'est plusieurs fois contenté d'atteindre dans l'expression de sa pensée à un certain agrément, pose ainsi pour toujours les bornes de son talent, comme en cédant souvent au plaisir, à la paresse, à la peur de souffrir on dessine soi-même sur un caractère où la retouche finit par n'être plus possible, la figure de ses vices et les limites de sa vertu.

Si, pourtant, malgré tant de correspondances que je perçus dans la suite entre l'écrivain et l'homme, je n'avais pas cru au premier moment, chez Odette, que ce fût Bergotte, que ce fût l'auteur de tant de livres divins qui se trouvât devant moi, peut-être n'avais-je pas eu absolument tort, car lui-même (au vrai sens du mot) ne le « croyait » pas non plus. Il ne le croyait pas puisqu'il montrait un grand empressement envers des gens du monde (sans être d'ailleurs snob), envers des gens de lettres, des journalistes, qui lui étaient bien inférieurs. Certes, maintenant il avait appris par le suffrage des autres qu'il avait du génie, à côté de quoi la situation dans le monde et les positions officielles ne sont rien. Il avait appris qu'il avait du génie, mais il ne le croyait pas puisqu'il continuait à simuler la déférence envers des écrivains médiocres pour arriver à être prochainement académicien, alors que l'Académie ou le faubourg Saint-Germain n'ont pas plus à voir avec la part de l'Esprit éternel laquelle est l'auteur des livres de Bergotte qu'avec le principe de causalité ou l'idée de Dieu. Cela il le savait aussi, comme un kleptomane sait inutilement qu'il est mal de voler. Et l'homme à barbiche et à nez en colimaçon avait des ruses de gentleman voleur de fourchettes, pour se rapprocher du fauteuil académique espéré, de telle duchesse qui disposait de plusieurs voix dans les élections, mais de s'en rapprocher en tâchant qu'aucune personne qui eût estimé que c'était un vice de poursuivre un pareil but, pût voir son manège. Il n'y réussissait qu'à demi, on entendait alterner avec les propos du vrai Bergotte ceux du Bergotte égoïste, ambitieux et qui ne pensait qu'à parler de tels gens puissants, nobles ou riches, pour se faire valoir, lui qui dans ses livres, quand il était vraiment lui-même, avait si bien montré, pur comme celui d'une source, le charme des pauvres.

Quant à ces autres vices auxquels avait fait allusion Norpois, à cet amour à demi incestueux qu'on disait même compliqué d'indélicatesse en matière d'argent, s'ils contredisaient d'une façon choquante la tendance de ses derniers romans, pleins d'un souci si scrupuleux, si douloureux, du bien, que les moindres joies de leurs héros en étaient empoisonnées et que pour le lecteur même il s'en dégageait un sentiment d'angoisse à travers lequel l'existence la plus douce semblait difficile à supporter, ces vices ne prouvaient pas cependant, à supposer qu'on les imputât justement à Bergotte, que sa littérature fût mensongère, et tant de sensibilité, de la comédie. De même qu'en pathologie certains états d'apparence semblable sont dus, les uns à un excès, d'autres à une insuffisance de tension, de sécrétion, etc., de même il peut y avoir vice par hypersensibilité comme il y a vice par manque de sensibilité. Peut-être n'est-ce que dans des vies réellement vicieuses que le problème moral peut se poser avec toute sa force d'anxiété. Et à ce problème l'artiste donne une solution non pas dans le plan de sa vie individuelle, mais de ce qui est pour lui sa vraie vie, une solution générale, littéraire. Comme les grands docteurs de l'Église commencèrent souvent tout en étant bons par connaître les péchés de tous les hommes, et en tirèrent leur sainteté personnelle, souvent les grands artistes tout en étant mauvais se servent de leurs vices pour arriver à concevoir la règle morale de tous. Ce sont les vices (ou seulement les faiblesses et les ridicules) du milieu où ils vivaient, les propos inconséquents, la vie frivole et choquante de leur fille, les trahisons de leur femme ou leurs propres fautes, que les écrivains ont le plus souvent flétries dans leurs diatribes sans changer pour cela le train de leur ménage ou le mauvais ton qui règne dans leur foyer. Mais ce contraste frappait moins autrefois qu'au temps de Bergotte, parce que d'une part, au fur et à mesure que se corrompait la société, les notions de moralité allaient s'épurant, et que d'autre part le public s'était mis au courant plus qu'il n'avait encore fait jusque-là de la vie privée des écrivains ; et certains soirs au théâtre on se montrait l'auteur que j'avais tant admiré à Combray, assis au fond d'une loge dont la seule composition semblait un commentaire singulièrement risible ou poignant, un impudent démenti de la thèse qu'il venait de soutenir dans sa dernière oeuvre. Ce n'est pas ce que les uns ou les autres purent me dire qui me renseigna beaucoup sur la bonté ou la méchanceté de Bergotte. Tel de ses proches fournissait des preuves de sa dureté, tel inconnu citait un trait (touchant car il avait été évidemment destiné à rester caché) de sa sensibilité profonde. Il avait agi cruellement avec sa femme. Mais dans une auberge de village où il était venu passer la nuit, il était resté pour veiller une pauvresse qui avait tenté de se jeter à l'eau, et quand il avait été obligé de partir il avait laissé beaucoup d'argent à l'aubergiste pour qu'il ne chassât pas cette malheureuse et pour qu'il eût des attentions envers elle. Peut-être plus le grand écrivain se développa en Bergotte aux dépens de l'homme à barbiche, plus sa vie individuelle se noya dans le flot de toutes les vies qu'il imaginait et ne lui parut plus l'obliger à des devoirs effectifs, lesquels étaient remplacés pour lui par le devoir d'imaginer ces autres vies. Mais en même temps, parce qu'il imaginait les sentiments des autres aussi bien que s'ils avaient été les siens, quand l'occasion faisait qu'il avait à s'adresser à un malheureux, au moins d'une façon passagère, il le faisait en se plaçant non à son point de vue personnel, mais à celui même de l'être qui souffrait, point de vue d'où lui aurait fait horreur le langage de ceux qui continuent à penser à leurs petits intérêts devant la douleur d'autrui. De sorte qu'il a excité autour de lui des rancunes justifiées et des gratitudes ineffaçables.
