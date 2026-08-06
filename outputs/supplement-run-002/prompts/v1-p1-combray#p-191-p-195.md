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
        "Bergotte",
        "l'écrivain",
        "ses livres",
        "sa prose"
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
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "« je pleurai sur les pages de l'écrivain comme dans les bras d'un père retrouvé »; « je lisais, je chantais intérieurement sa prose »; « Plus que tout j'aimais sa philosophie, je m'étais donné à elle pour toujours ».",
      "explanation": "The narrator expresses intense admiration and devotion for Bergotte; his writings even validate intuitions of the narrator’s own, producing a local affective and intellectual elevation of Bergotte."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "Bergotte is locally magnified as an affective and philosophical guide, arousing veneration and confidence."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-191-p-195"
}

### Candidate characters

[
  "Bloch",
  "Françoise",
  "Legrandin",
  "Swann",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Je n'étais pas tout à fait le seul admirateur de Bergotte ; il était aussi l'écrivain préféré d'une amie de la mère du narrateur qui était très lettrée ; enfin pour lire son dernier livre paru, docteur Cottard du Boulbon faisait attendre ses malades ; et ce fut de son cabinet de consultation, et d'un parc voisin de Combray, que s'envolèrent quelques-unes des premières graines de cette prédilection pour Bergotte, espèce si rare alors, aujourd'hui universellement répandue, et dont on trouve partout en Europe, en Amérique, jusque dans le moindre village, la fleur idéale et commune. Ce que l'amie de la mère du narrateur et, paraît-il, docteur Cottard du Boulbon aimaient surtout dans les livres de Bergotte c'était, comme moi, ce même flux mélodique, ces expressions anciennes, quelques autres très simples et connues, mais pour lesquelles la place où il les mettait en lumière semblait révéler de sa part un goût particulier ; enfin, dans les passages tristes, une certaine brusquerie, un accent presque rauque. Et sans doute lui-même devait sentir que là étaient ses plus grands charmes. Car dans les livres qui suivirent, s'il avait rencontré quelque grande vérité, ou le nom d'une célèbre cathédrale, il interrompait son récit et dans une invocation, une apostrophe, une longue prière, il donnait un libre cours à ces effluves qui dans ses premiers ouvrages restaient intérieurs à sa prose, décelés seulement alors par les ondulations de la surface, plus douces peut-être encore, plus harmonieuses quand elles étaient ainsi voilées et qu'on n'aurait pu indiquer d'une manière précise où naissait, où expirait leur murmure. Ces morceaux auxquels il se complaisait étaient nos morceaux préférés. Pour moi, je les savais par coeur. J'étais déçu quand il reprenait le fil de son récit. Chaque fois qu'il parlait de quelque chose dont la beauté m'était restée jusque-là cachée, des forêts de pins, de la grêle, de Notre-Dame de Paris, d'Athalie ou de Phèdre, il faisait dans une image exploser cette beauté jusqu'à moi. Aussi sentant combien il y avait de parties de l'univers que ma perception infirme ne distinguerait pas s'il ne les rapprochait de moi, j'aurais voulu posséder une opinion de lui, une métaphore de lui, sur toutes choses, surtout sur celles que j'aurais l'occasion de voir moi-même, et entre celles-là, particulièrement sur d'anciens monuments français et certains paysages maritimes, parce que l'insistance avec laquelle il les citait dans ses livres prouvait qu'il les tenait pour riches de signification et de beauté. Malheureusement sur presque toutes choses j'ignorais son opinion. Je ne doutais pas qu'elle ne fût entièrement différente des miennes, puisqu'elle descendait d'un monde inconnu vers lequel je cherchais à m'élever : persuadé que mes pensées eussent paru pure ineptie à cet esprit parfait, j'avais tellement fait table rase de toutes, que quand par hasard il m'arriva d'en rencontrer, dans tel de ses livres, une que j'avais déjà eue moi-même, mon coeur se gonflait comme si un Dieu dans sa bonté me l'avait rendue, l'avait déclarée légitime et belle. Il arrivait parfois qu'une page de lui disait les mêmes choses que j'écrivais souvent la nuit à la grand-mère et à la mère du narrateur quand je ne pouvais pas dormir, si bien que cette page de Bergotte avait l'air d'un recueil d'épigraphes pour être placées en tête de mes lettres. Même plus tard, quand je commençai de composer un livre, certaines phrases dont la qualité ne suffit pas pour décider à le continuer, j'en retrouvai l'équivalent dans Bergotte. Mais ce n'était qu'alors, quand je les lisais dans son oeuvre, que je pouvais en jouir ; quand c'était moi qui les composais, préoccupé qu'elles reflétassent exactement ce que j'apercevais dans ma pensée, craignant de ne pas « faire ressemblant », j'avais bien le temps de me demander si ce que j'écrivais était agréable ! Mais en réalité il n'y avait que ce genre de phrases, ce genre d'idées que j'aimais vraiment. Mes efforts inquiets et mécontents étaient eux-mêmes une marque d'amour, d'amour sans plaisir mais profond. Aussi quand tout d'un coup je trouvais de telles phrases dans l'oeuvre d'un autre, c'est-à-dire sans plus avoir de scrupules, de sévérité, sans avoir à me tourmenter, je me laissais enfin aller avec délices au goût que j'avais pour elles, comme un cuisinier qui pour une fois où il n'a pas à faire la cuisine trouve enfin le temps d'être gourmand.

### Passage

Un jour, ayant rencontré dans un livre de Bergotte, à propos d'une vieille servante, une plaisanterie que le magnifique et solennel langage de l'écrivain rendait encore plus ironique, mais qui était la même que j'avais si souvent faite à ma grand'mère en parlant de Françoise, une autre fois que je vis qu'il ne jugeait pas indigne de figurer dans un de ces miroirs de la vérité qu'étaient ses ouvrages une remarque analogue à celle que j'avais eu l'occasion de faire sur notre ami Legrandin (remarques sur Françoise et Legrandin qui étaient certes de celles que j'eusse le plus délibérément sacrifiées à Bergotte, persuadé qu'il les trouverait sans intérêt), il me sembla soudain que mon humble vie et les royaumes du vrai n'étaient pas aussi séparés que j'avais cru, qu'ils coïncidaient même sur certains points, et de confiance et de joie je pleurai sur les pages de l'écrivain comme dans les bras d'un père retrouvé.

D'après ses livres j'imaginais Bergotte comme un vieillard faible et déçu qui avait perdu des enfants et ne s'était jamais consolé. Aussi je lisais, je chantais intérieurement sa prose, plus « dolce », plus « lento » peut-être qu'elle n'était écrite, et la phrase la plus simple s'adressait à moi avec une intonation attendrie. Plus que tout j'aimais sa philosophie, je m'étais donné à elle pour toujours. Elle me rendait impatient d'arriver à l'âge où j'entrerais au collège, dans la classe appelée Philosophie. Mais je ne voulais pas qu'on y fît autre chose que vivre uniquement par la pensée de Bergotte, et si l'on m'avait dit que les métaphysiciens auxquels je m'attacherais alors ne lui ressembleraient en rien, j'aurais ressenti le désespoir d'un amoureux qui veut aimer pour la vie et à qui on parle des autres maîtresses qu'il aura plus tard.

Un dimanche, pendant ma lecture au jardin, je fus dérangé par Swann qui venait voir mes parents.

– Qu'est-ce que vous lisez, on peut regarder ? Tiens, du Bergotte ? Qui donc vous a indiqué ses ouvrages ?

Je lui dis que c'était Bloch.
