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
      "canonical_name": "duc de Guermantes",
      "surface_forms": [
        "duc de Guermantes",
        "le vieux duc",
        "le duc"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "duc de Guermantes",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Cette liaison ... venait de faire perdre au duc de Guermantes ... la présidence du Jockey et un siège ... à l'Académie des Beaux-Arts ... Ainsi les deux frères ... étaient arrivés à la déconsidération ... Ces positions ... avaient perdu leur inviolabilité.",
      "explanation": "The narrator presents the duke’s affair as publicly discrediting him and eroding once ‘imprenable’ positions, with concrete losses of prestigious posts."
    }
  ],
  "status_effects": [
    {
      "character": "duc de Guermantes",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He suffers clear public discredit, with lost presidencies and the collapse of formerly inviolable social positions."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-101-p-105"
}

### Candidate characters

[
  "Albertine",
  "Gilberte",
  "Jupien",
  "M. Verdurin",
  "M. de Marsantes",
  "Mme de Villeparisis",
  "Morel",
  "Odette",
  "Robert de Saint-Loup",
  "Swann",
  "baron de Charlus",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "oncle Adolphe",
  "princesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Quand on pensait à l'âge que devait avoir maintenant Mme de comte de Forcheville, cela semblait, en effet, extraordinaire. Mais peut-être Odette avait-elle commencé la vie de femme galante très jeune. Et puis il y a des femmes qu'à chaque décade on retrouve en une nouvelle incarnation, ayant de nouvelles amours, parfois alors qu'on les croyait mortes, faisant le désespoir d'une jeune femme que pour elles abandonne son mari.

### Passage

La vie de la duchesse ne laissait pas, d'ailleurs, d'être très malheureuse et pour une raison qui, par ailleurs, avait pour effet de déclasser parallèlement la société que fréquentait duc de Guermantes . Celui-ci qui, depuis longtemps calmé par son âge avancé, et quoiqu'il fût encore robuste, avait cessé de tromper Mme de Guermantes, s'était épris de Mme de Forcheville sans qu'on sût bien les débuts de cette liaison.

Mais celle-ci avait pris des proportions telles que le vieillard, imitant, dans ce dernier amour, la manière de celles qu'il avait eues autrefois, séquestrait sa maîtresse au point que, si mon amour pour Albertine avait répété, avec de grandes variations, l'amour de Swann pour Odette, l'amour de duc de Guermantes  rappelait celui que j'avais eu pour Albertine. Il fallait qu'elle déjeunât, qu'elle dînât avec lui, il était toujours chez elle ; elle s'en parait auprès d'amis qui sans elle n'eussent jamais été en relation avec le duc de Guermantes et qui venaient là pour le connaître, un peu comme on va chez une cocotte pour connaître un souverain son amant. Certes, Mme de Forcheville était depuis longtemps devenue une femme du monde. Mais recommençant à être entretenue sur le tard, et par un si orgueilleux vieillard qui était tout de même chez elle le personnage important, elle se diminuait à chercher seulement à avoir les peignoirs qui lui plussent, la cuisine qu'il aimait, à flatter ses amis en leur disant qu'elle lui avait parlé d'eux, comme elle disait à mon grand-oncle qu'elle avait parlé de lui au Grand-Duc qui lui envoyait des cigarettes, en un mot elle tendait, malgré tout l'acquis de sa situation mondaine, et par la force de circonstances nouvelles, à redevenir, telle qu'elle était apparue à mon enfance, la dame en rose. Certes, il y avait bien des années que mon oncle Adolphe était mort. Mais la substitution autour de nous d'autres personnes aux anciennes nous empêche-t-elle de recommencer la même vie ? Ces circonstances nouvelles, elle s'y était prêtée sans doute par cupidité, mais aussi parce que, assez recherchée dans le monde quand elle avait une fille à marier, laissée de côté dès que Gilberte eut épousé Saint-Loup, elle sentit que le duc de Guermantes, qui eût tout fait pour elle, lui amènerait nombre de duchesses peut-être enchantées de jouer un tour à leur amie Mme de Guermantes, et peut-être enfin piquée au jeu par le mécontentement de la duchesse sur laquelle un sentiment féminin de rivalité la rendait heureuse de prévaloir. Des neveux fort difficiles du duc de Guermantes, les Courvoisier, Mme de Marsantes, la princesse de Trania, allaient chez Mme de Forcheville dans un espoir d'héritage, sans s'occuper de la peine que cela pouvait faire à Mme de Guermantes, dont Odette, piquée par ses dédains, disait tout le mal possible. Cette liaison avec Mme de Forcheville, liaison qui n'était qu'une imitation de ses liaisons plus anciennes, venait de faire perdre au duc de Guermantes, pour la deuxième fois, la possibilité de la présidence du Jockey et un siège de membre libre à l'Académie des Beaux-Arts, comme la vie de Charlus, publiquement associée à celle de Jupien, lui avait fait manquer la présidence de l'Union et celle aussi de la Société des amis du Vieux Paris. Ainsi les deux frères, si différents dans leurs goûts, étaient arrivés à la déconsidération à cause d'une même paresse, d'un même manque de volonté, lequel était sensible, mais agréablement, chez le duc de Guermantes leur grand-père, membre de l'Académie française, mais qui, chez les deux petits-fils, avait permis à un goût naturel et à un autre qui passe pour ne l'être pas, de les désocialiser.

Le vieux duc ne sortait plus, car il passait ses journées et ses soirées chez Odette. Mais aujourd'hui, comme elle-même s'était rendue à la matinée de la princesse de Guermantes, il était venu un instant pour la voir, malgré l'ennui de rencontrer sa femme. Je ne l'eusse sans doute pas reconnu, si la duchesse, quelques instants plus tôt, ne me l'eût clairement désigné en allant jusqu'à lui. Il n'était plus qu'une ruine, mais superbe, et plus encore qu'une ruine, cette belle chose romantique que peut être un rocher dans la tempête. Fouettée de toutes parts par les vagues de souffrance, de colère de souffrir, d'avancée montante de la mer qui la circonvenaient, sa figure, effritée comme un bloc, gardait le style, la cambrure que j'avais toujours admirés ; elle était rongée comme une de ces belles têtes antiques trop abîmées mais dont nous sommes trop heureux d'orner un cabinet de travail. Elle paraissait seulement appartenir à une époque plus ancienne qu'autrefois, non seulement à cause de ce qu'elle avait pris de rude et de rompu dans sa matière jadis plus brillante, mais parce que à l'expression de finesse et d'enjouement avait succédé une involontaire, une inconsciente expression, bâtie par la maladie, de lutte contre la mort, de résistance, de difficulté à vivre. Les artères ayant perdu toute souplesse avaient donné au visage jadis épanoui une dureté sculpturale. Et sans que le duc s'en doutât, il découvrait des aspects de nuque, de joue, de front, où l'être, comme obligé de se raccrocher avec acharnement à chaque minute, semblait bousculé dans une tragique rafale, pendant que les mèches blanches de sa chevelure moins épaisse venaient souffleter de leur écume le promontoire envahi du visage. Et comme ces reflets étranges, uniques, que seule l'approche de la tempête où tout va sombrer donne aux roches qui avaient été jusque-là d'une autre couleur, je compris que le gris plombé des joues raides et usées, le gris presque blanc et moutonnant des mèches soulevées, la faible lumière encore départie aux yeux qui voyaient à peine, étaient des teintes non pas irréelles, trop réelles au contraire, mais fantastiques et empruntées à la palette de l'éclairage, inimitable dans ses noirceurs effrayantes et prophétiques, de la vieillesse, de la proximité de la mort. Le duc ne resta que quelques instants, assez pour que je comprisse qu'Odette, toute à des soupirants plus jeunes, se moquait de lui. Mais, chose curieuse, lui qui jadis était presque ridicule quand il prenait l'allure d'un roi de théâtre avait pris un aspect véritablement grand, un peu comme son frère, à qui la vieillesse, en le désencombrant de tout l'accessoire, le faisait ressembler. Et comme son frère, lui, jadis orgueilleux, bien que d'une autre manière, semblait presque respectueux, quoique aussi d'une autre façon. Car il n'avait pas subi la déchéance de Charlus, réduit à saluer avec une politesse de malade oublieux ceux qu'il eût jadis dédaignés, mais il était très vieux, et quand il voulut passer la porte et descendre l'escalier pour sortir, la vieillesse, qui est tout de même l'état le plus misérable pour les hommes et qui les précipite de leur faîte le plus semblablement aux rois des tragédies grecques, la vieillesse, en le forçant à s'arrêter dans le chemin de croix que devient la vie des impotents menacés, à essuyer son front ruisselant, à tâtonner, en cherchant des yeux une marche qui se dérobait, parce qu'il aurait eu besoin pour ses pas mal assurés, pour ses yeux ennuagés, d'un appui, lui donnait à son insu l'air de l'implorer doucement et timidement des autres, la vieillesse l'avait fait encore plus qu'auguste, suppliant.

Ainsi, dans le faubourg Saint-Germain, ces positions en apparence imprenables du duc et de la Mme de Guermantes, du baron de Charlus avaient perdu leur inviolabilité, comme toutes choses changent en ce monde, par l'action d'un principe intérieur auquel on n'avait pas pensé : chez Charlus l'amour de Morel qui l'avait rendu esclave des Verdurin, puis le ramollissement ; chez Mme de Guermantes, un goût de nouveauté et d'art ; chez duc de Guermantes , un amour exclusif, comme il en avait déjà eu de pareils dans sa vie, que la faiblesse de l'âge rendait plus tyrannique et aux faiblesses duquel la sévérité du salon de la duchesse, où le duc ne paraissait plus et qui, d'ailleurs, ne fonctionnait plus guère, n'opposait plus son démenti, son rachat mondain. Ainsi change la figure des choses de ce monde, ainsi le centre des empires et le cadastre des fortunes, et la charte des situations, tout ce qui semblait définitif est-il perpétuellement remanié et les yeux d'un homme qui a vécu peuvent-ils contempler le changement le plus complet là où justement il lui paraissait le plus impossible.

Ne pouvant se passer d'Odette, toujours installé chez elle dans le même fauteuil d'où la vieillesse et la goutte le faisaient difficilement lever, duc de Guermantes  la laissait recevoir des amis qui étaient trop contents d'être présentés au duc, de lui laisser la parole, de l'entendre parler de la vieille société, de la Mme de Villeparisis, du duc de Chartres.
