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
      "canonical_name": "Albertine",
      "surface_forms": [
        "Albertine"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Albertine",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "« ce fut la seule image d'Albertine qui s'éleva de mon coeur et se mit à briller »; « la vivacité déjà grande de mon amour pour Albertine »",
      "explanation": "The narrator foregrounds Albertine as the sole luminous image and affirms the growing intensity of his love, locally elevating her in his evaluative focus."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Albertine's local standing is raised through the narrator's concentrated admiration and desire."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-461-p-465"
}

### Candidate characters

[
  "Andrée",
  "Elstir",
  "Mme Bontemps",
  "le narrateur"
]

### Prior local context (optional)

Nous allâmes retrouver les autres jeunes filles pour rentrer. Je savais maintenant que j'aimais Albertine ; mais hélas ! je ne me souciais pas de le lui apprendre. C'est que, depuis le temps des jeux aux Champs-Élysées, ma conception de l'amour était devenue différente, si les êtres auxquels s'attachaient successivement mon amour demeuraient presque identiques. D'une part l'aveu, la déclaration de ma tendresse à celle que j'aimais ne me semblait plus une des scènes capitales et nécessaires de l'amour ; ni celui-ci, une réalité extérieure mais seulement un plaisir subjectif. Et ce plaisir, je sentais qu'Albertine ferait d'autant plus ce qu'il fallait pour l'entretenir qu'elle ignorerait que je l'éprouvais.

### Passage

Pendant tout ce retour, l'image d'Albertine noyée dans la lumière qui émanait des autres jeunes filles ne fut pas seule à exister pour moi. Mais comme la lune, qui n'est qu'un petit nuage blanc d'une forme plus caractérisée et plus fixe pendant le jour, prend toute sa puissance dès que celui-ci s'est éteint, ainsi quand je fus rentré à l'hôtel ce fut la seule image d'Albertine qui s'éleva de mon coeur et se mit à briller. Ma chambre me semblait tout d'un coup nouvelle. Certes, il y avait bien longtemps qu'elle n'était plus la chambre ennemie du premier soir. Nous modifions inlassablement notre demeure autour de nous ; et, au fur et à mesure que l'habitude nous dispense de sentir, nous supprimons les éléments nocifs de couleur, de dimension et d'odeur qui objectivaient notre malaise. Ce n'était plus davantage la chambre, assez puissante encore sur ma sensibilité, non certes pour me faire souffrir, mais pour me donner de la joie, la cuve des beaux jours, semblable à une piscine à mi-hauteur de laquelle ils faisaient miroiter un azur mouillé de lumière, que recouvrait un moment, impalpable et blanche comme une émanation de la chaleur, une voile reflétée et fuyante ; ni la chambre purement esthétique des soirs picturaux ; c'était la chambre où j'étais depuis tant de jours que je ne la voyais plus. Or voici que je venais de recommencer à ouvrir les yeux sur elle, mais cette fois-ci de ce point de vue égoïste qui est celui de l'amour. Je songeais que la belle glace oblique, les élégantes bibliothèques vitrées donneraient à Albertine si elle venait me voir une bonne idée de moi. À la place d'un lieu de transition où je passais un instant avant de m'évader vers la plage ou vers Rivebelle, ma chambre me redevenait réelle et chère, se renouvelait, car j'en regardais et en appréciais chaque meuble avec les yeux d'Albertine.

Quelques jours après la partie de furet, comme nous étant laissés entraîner trop loin dans une promenade nous avions été fort heureux de trouver à Maineville deux petits « tonneaux » à deux places qui nous permettraient de revenir pour l'heure du dîner, la vivacité déjà grande de mon amour pour Albertine eut pour effet que ce fut successivement à Rosemonde et à Andrée que je proposai de monter avec moi, et pas une fois à Albertine ; ensuite que, tout en invitant de préférence Andrée ou Rosemonde, j'amenai tout le monde, par des considérations secondaires d'heure, de chemin et de manteaux, à décider comme contre mon gré que le plus pratique était que je prisse avec moi Albertine, à la compagnie de laquelle je feignis de me résigner tant bien que mal. Malheureusement l'amour tendant à l'assimilation complète d'un être, comme aucun n'est comestible par la seule conversation, Albertine eut beau être aussi gentille que possible pendant ce retour, quand je l'eus déposée chez elle, elle me laissa heureux, mais plus affamé d'elle encore que je n'étais au départ, et ne comptant les moments que nous venions de passer ensemble que comme un prélude, sans grande importance par lui-même, à ceux qui suivraient. Il avait pourtant ce premier charme qu'on ne retrouve pas. Je n'avais encore rien demandé à Albertine. Elle pouvait imaginer ce que je désirais, mais n'en étant pas sûre, supposer que je ne tendais qu'à des relations sans but précis auxquelles mon amie devait trouver ce vague délicieux, riche de surprises attendues, qui est le romanesque.

Dans la semaine qui suivit je ne cherchai guère à voir Albertine. Je faisais semblant de préférer Andrée. L'amour commence, on voudrait rester pour celle qu'on aime l'inconnu qu'elle peut aimer, mais on a besoin d'elle, on a besoin de toucher moins son corps que son attention, son coeur. On glisse dans une lettre une méchanceté qui forcera l'indifférente à vous demander une gentillesse, et l'amour, suivant une technique infaillible, resserre pour nous d'un mouvement alterné l'engrenage dans lequel on ne peut plus ni ne pas aimer, ni être aimé. Je donnais à Andrée les heures où les autres allaient à quelque matinée que je savais qu'Andrée me sacrifierait, par plaisir, et qu'elle m'eût sacrifiées même avec ennui, par élégance morale, pour ne pas donner aux autres ni à elle-même l'idée qu'elle attachait du prix à un plaisir relativement mondain. Je m'arrangeais ainsi à l'avoir chaque soir toute à moi, pensant non pas rendre Albertine jalouse, mais accroître à ses yeux mon prestige ou du moins ne pas le perdre en apprenant à Albertine que c'était elle et non Andrée que j'aimais. Je ne le disais pas non plus à Andrée de peur qu'elle le lui répétât. Quand je parlais d'Albertine avec Andrée, j'affectais une froideur dont Andrée fut peut-être moins dupe que moi de sa crédulité apparente. Elle faisait semblant de croire à mon indifférence pour Albertine, de désirer l'union la plus complète possible entre Albertine et moi. Il est probable qu'au contraire elle ne croyait pas à la première ni ne souhaitait la seconde. Pendant que je lui disais me soucier assez peu de son amie, je ne pensais qu'à une chose, tâcher d'entrer en relations avec Mme Bontemps qui était pour quelques jours près de Balbec et chez qui Albertine devait bientôt aller passer trois jours. Naturellement, je ne laissais pas voir ce désir à Andrée, et, quand je lui parlais de la famille d'Albertine, c'était de l'air le plus inattentif. Les réponses explicites d'Andrée ne paraissaient pas mettre en doute ma sincérité. Pourquoi donc lui échappa-t-il un de ces jours-là de me dire : « J'ai justement vu la tante à Albertine » ? Certes elle ne m'avait pas dit : « J'ai bien démêlé sous vos paroles, jetées comme par hasard, que vous ne pensiez qu'à vous lier avec la tante d'Albertine. » Mais c'est bien à la présence, dans l'esprit d'Andrée, d'une telle idée qu'elle trouvait plus poli de me cacher, que semblait se rattacher le mot « justement ». Il était de la famille de certains regards, de certains gestes, qui bien que n'ayant pas une forme logique, rationnelle, directement élaborée pour l'intelligence de celui qui écoute, lui parviennent cependant avec leur signification véritable, de même que la parole humaine, changée en électricité dans le téléphone, se refait parole pour être entendue. Afin d'effacer de l'esprit d'Andrée l'idée que je m'intéressais à Mme Bontemps, je ne parlai plus d'elle avec distraction seulement, mais avec bienveillance ; je dis avoir rencontré autrefois cette espèce de folle et que j'espérais bien que cela ne m'arriverait plus. Or je cherchais au contraire de toute façon à la rencontrer.

Je tâchai d'obtenir d'Elstir, mais sans dire à personne que je l'en avais sollicité, qu'il lui parlât de moi et me réunît avec elle. Il me promit de me la faire connaître, s'étonnant toutefois que je le souhaitasse, car il la jugeait une femme méprisable, intrigante et aussi inintéressante qu'intéressée. Pensant que, si je voyais Mme Bontemps, Andrée le saurait tôt ou tard, je crus qu'il valait mieux l'avertir. « Les choses qu'on cherche le plus à fuir sont celles qu'on arrive à ne pouvoir éviter, lui dis-je. Rien au monde ne peut m'ennuyer autant que de retrouver Mme Bontemps, et pourtant je n'y échapperai pas, Elstir doit m'inviter avec elle. – Je n'en ai jamais douté un seul instant », s'écria Andrée d'un ton amer, pendant que son regard grandi et altéré par le mécontentement se rattachait à je ne sais quoi d'invisible. Ces paroles d'Andrée ne constituaient pas l'exposé le plus ordonné d'une pensée qui peut se résumer ainsi : « Je sais bien que vous aimez Albertine et que vous faites des pieds et des mains pour vous rapprocher de sa famille. » Mais elles étaient les débris informes et reconstituables de cette pensée que j'avais fait exploser, en la heurtant, malgré Andrée. De même que le « justement », ces paroles n'avaient de signification qu'au second degré, c'est-à-dire qu'elles étaient celles qui (et non pas les affirmations directes) nous inspirent de l'estime ou de la méfiance à l'égard de quelqu'un, nous brouillent avec lui.

Puisque Andrée ne m'avait pas cru quand je lui disais que la famille d'Albertine m'était indifférente, c'est qu'elle pensait que j'aimais Albertine. Et probablement n'en était-elle pas heureuse.
