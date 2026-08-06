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
      "canonical_name": "M. Vinteuil",
      "surface_forms": [
        "M. Vinteuil",
        "père M. Vinteuil"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "M. Vinteuil",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.82,
      "evidence": "On disait: « Faut-il que ce pauvre M. Vinteuil soit aveuglé par la tendresse... permettre à sa fille... de faire vivre sous son toit une femme pareille. » ... « Il peut être sûr que ce n'est pas de musique qu'elle s'occupe... »; le docteur fait rire tout le monde avec: « Ah! sapristi on en fait une musique dans c'te boîte-là. »",
      "explanation": "The country speech ridicules and deprecates Mr. Vinteuil by linking him to the 'bad reputation' of his daughter's friend and saying he is blinded; the narrator reports this blame with ironic distance, highlighting the pleasant treachery of the doctor."
    }
  ],
  "status_effects": [
    {
      "character": "M. Vinteuil",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Locally, his reputation is undermined by the gossip that presents him as morally blind and compromised by his daughter's friend."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-306-p-310"
}

### Candidate characters

[
  "Bloch",
  "Gilberte",
  "duchesse de Guermantes",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Ma tante n'alla pas voir la haie d'épines roses, mais à tous moments je demandais à mes parents si elle n'irait pas, si autrefois elle allait souvent à Tansonville, tâchant de les faire parler des parents et grands-parents de Gilberte qui me semblaient grands comme des Dieux. Ce nom, devenu pour moi presque mythologique, de Swann, quand je causais avec mes parents, je languissais du besoin de le leur entendre dire, je n'osais pas le prononcer moi-même, mais je les entraînais sur des sujets qui avoisinaient Gilberte et sa famille, qui la concernaient, où je ne me sentais pas exilé trop loin d'elle ; et je contraignais tout d'un coup le père du narrateur, en feignant de croire par exemple que la charge de mon grand-père avait été déjà avant lui dans notre famille, ou que la haie d'épines roses que voulait voir ma tante Léonie se trouvait en terrain communal, à rectifier mon assertion, à me dire, comme malgré moi, comme de lui-même : « Mais non, cette charge-là était au père de Swann, cette haie fait partie du parc de Swann. » Alors j'étais obligé de reprendre ma respiration, tant, en se posant sur la place où il était toujours écrit en moi, pesait à m'étouffer ce nom qui, au moment où je l'entendais, me paraissait plus plein que tout autre, parce qu'il était lourd de toutes les fois où, d'avance, je l'avais mentalement proféré. Il me causait un plaisir que j'étais confus d'avoir osé réclamer à mes parents, car ce plaisir était si grand qu'il avait dû exiger d'eux pour qu'ils me le procurassent beaucoup de peine, et sans compensation, puisqu'il n'était pas un plaisir pour eux. Aussi je détournais la conversation par discrétion. Par scrupule aussi. Toutes les séductions singulières que je mettais dans ce nom de Swann, je les retrouvais en lui dès qu'ils le prononçaient. Il me semblait alors tout d'un coup que mes parents ne pouvaient pas ne pas les ressentir, qu'ils se trouvaient placés à mon point de vue, qu'ils apercevaient à leur tour, absolvaient, épousaient mes rêves, et j'étais malheureux comme si je les avais vaincus et dépravés.

### Passage

Cette année-là, quand, un peu plus tôt que d'habitude, mes parents eurent fixé le jour de rentrer à Paris, le matin du départ, comme on m'avait fait friser pour être photographié, coiffer avec précaution un chapeau que je n'avais encore jamais mis et revêtir une douillette de velours, après m'avoir cherché partout, ma mère me trouva en larmes dans le petit raidillon contigu à Tansonville, en train de dire adieu aux aubépines, entourant de mes bras les branches piquantes, et, comme une princesse de tragédie à qui pèseraient ces vains ornements, ingrat envers l'importune main qui en formant tous ces noeuds avait pris soin sur mon front d'assembler mes cheveux, foulant aux pieds mes papillotes arrachées et mon chapeau neuf. Ma mère ne fut pas touchée par mes larmes, mais elle ne put retenir un cri à la vue de la coiffe défoncée et de la douillette perdue. Je ne l'entendis pas : « Ô mes pauvres petites aubépines, disais-je en pleurant, ce n'est pas vous qui voudriez me faire du chagrin, me forcer à partir. Vous, vous ne m'avez jamais fait de peine ! Aussi je vous aimerai toujours. » Et, essuyant mes larmes, je leur promettais, quand je serais grand, de ne pas imiter la vie insensée des autres hommes et, même à Paris, les jours de printemps, au lieu d'aller faire des visites et écouter des niaiseries, de partir dans la campagne voir les premières aubépines.

Une fois dans les champs, on ne les quittait plus pendant tout le reste de la promenade qu'on faisait du côté de Méséglise. Ils étaient perpétuellement parcourus, comme par un chemineau invisible, par le vent qui était pour moi le génie particulier de Combray. Chaque année, le jour de notre arrivée, pour sentir que j'étais bien à Combray, je montais le retrouver qui courait dans les sayons et me faisait courir à sa suite. On avait toujours le vent à côté de soi du côté de Méséglise, sur cette plaine bombée où pendant des lieues il ne rencontre aucun accident de terrain. Je savais que Gilberte allait souvent à Laon passer quelques jours et, bien que ce fût à plusieurs lieues, la distance se trouvant compensée par l'absence de tout obstacle, quand, par les chauds après-midi, je voyais un même souffle, venu de l'extrême horizon, abaisser les blés les plus éloignés, se propager comme un flot sur toute l'immense étendue et venir se coucher, murmurant et tiède, parmi les sainfoins et les trèfles, à mes pieds, cette plaine qui nous était commune à tous deux semblait nous rapprocher, nous unir, je pensais que ce souffle avait passé auprès d'elle, que c'était quelque message d'elle qu'il me chuchotait sans que je pusse le comprendre, et je l'embrassais au passage. À gauche était un village qui s'appelait Champieu (Campus Pagani, selon le curé). Sur la droite, on apercevait par delà les blés les deux clochers ciselés et rustiques de Saint-André-des-Champs, eux-mêmes effilés, écailleux, imbriqués d'alvéoles, guillochés, jaunissants et grumeleux, comme deux épis.

À intervalles symétriques, au milieu de l'inimitable ornementation de leurs feuilles qu'on ne peut confondre avec la feuille d'aucun autre arbre fruitier, les pommiers ouvraient leurs larges pétales de satin blanc ou suspendaient les timides bouquets de leurs rougissants boutons. C'est du côté de Méséglise que j'ai remarqué pour la première fois l'ombre ronde que les pommiers font sur la terre ensoleillée, et aussi ces soies d'or impalpable que le couchant tisse obliquement sous les feuilles, et que je voyais mon père interrompre de sa canne sans les faire jamais dévier.

Parfois dans le ciel de l'après-midi passait la lune blanche comme une nuée, furtive, sans éclat, comme une actrice dont ce n'est pas l'heure de jouer et qui, de la salle, en toilette de ville, regarde un moment ses camarades, s'effaçant, ne voulant pas qu'on fasse attention à elle. J'aimais à retrouver son image dans des tableaux et dans des livres, mais ces oeuvres d'art étaient bien différentes – du moins pendant les premières années, avant que Bloch eût accoutumé mes yeux et ma pensée à des harmonies plus subtiles – de celles où la lune me paraîtrait belle aujourd'hui et où je ne l'eusse pas reconnue alors. C'était, par exemple, quelque roman de Saintine, un paysage de Gleyre où elle découpe nettement sur le ciel une faucille d'argent, de ces oeuvres naïvement incomplètes comme étaient mes propres impressions et que les soeurs de ma grand'mère s'indignaient de me voir aimer. Elles pensaient qu'on doit mettre devant les enfants, et qu'ils font preuve de goût en aimant d'abord les oeuvres que parvenu à la maturité, on admire définitivement. C'est sans doute qu'elles se figuraient les mérites esthétiques comme des objets matériels qu'un oeil ouvert ne peut faire autrement que de percevoir, sans avoir eu besoin d'en mûrir lentement des équivalents dans son propre coeur.

C'est du côté de Méséglise, à Montjouvain, maison située au bord d'une grande mare et adossée à un talus buissonneux que demeurait Vinteuil. Aussi croisait-on souvent sur la route sa fille, conduisant un buggy à toute allure. À partir d'une certaine année on ne la rencontra plus seule, mais avec une amie plus âgée, qui avait mauvaise réputation dans le pays et qui un jour s'installa définitivement à Montjouvain. On disait : « Faut-il que ce pauvre Vinteuil soit aveuglé par la tendresse pour ne pas s'apercevoir de ce qu'on raconte, et permettre à sa fille, lui qui se scandalise d'une parole déplacée, de faire vivre sous son toit une femme pareille. Il dit que c'est une femme supérieure, un grand coeur et qu'elle aurait eu des dispositions extraordinaires pour la musique si elle les avait cultivées. Il peut être sûr que ce n'est pas de musique qu'elle s'occupe avec sa fille. » Vinteuil le disait ; et il est en effet remarquable combien une personne excite toujours d'admiration pour ses qualités morales chez les parents de toute autre personne avec qui elle a des relations charnelles. L'amour physique, si injustement décrié, force tellement tout être à manifester jusqu'aux moindres parcelles qu'il possède de bonté, d'abandon de soi, qu'elles resplendissent jusqu'aux yeux de l'entourage immédiat. Le docteur Percepied à qui sa grosse voix et ses gros sourcils permettaient de tenir tant qu'il voulait le rôle de perfide dont il n'avait pas le physique, sans compromettre en rien sa réputation inébranlable et imméritée de bourru bienfaisant, savait faire rire aux larmes le curé et tout le monde en disant d'un ton rude : « Hé bien ! il paraît qu'elle fait de la musique avec son amie, Mlle Vinteuil. Ça a l'air de vous étonner. Moi je sais pas. C'est le père Vinteuil qui m'a encore dit ça hier. Après tout, elle a bien le droit d'aimer la musique, c'te fille. Moi je ne suis pas pour contrarier les vocations artistiques des enfants. Vinteuil non plus à ce qu'il paraît. Et puis lui aussi il fait de la musique avec l'amie de sa fille. Ah ! sapristi on en fait une musique dans c'te boîte-là. Mais qu'est-ce que vous avez à rire ; mais ils font trop de musique ces gens. L'autre jour j'ai rencontré le père Vinteuil près du cimetière. Il ne tenait pas sur ses jambes. »
