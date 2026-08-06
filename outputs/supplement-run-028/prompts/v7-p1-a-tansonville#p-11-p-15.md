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
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "Accent « volontairement tendre… voix d’alcoolique et des modulations d’acteur »; « flatté d’être aimé par Gilberte »; il « donnait… des détails » sur l’amour supposé de Morel, « exagérés sinon inventés », tandis que Morel lui « demandait chaque jour plus d’argent »; ses manières « devenaient les manières de baron de Charlus »; il « faisait mourir sa femme de jalousie en cherchant sans plaisir des maîtresses »; « Moi, je suis un soldat… je n’ai pas soupçon de ces choses-là. »",
      "explanation": "The narrator locally belittles Robert by showing him as vain, affected, and insincere, embellishing his relationship with Morel who exploits him, adopting imitated and ambiguous manners, denying his inclination while burdening his wife with a clumsy social strategy."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "He emerges locally diminished: vanity, concealment, theatrical posture, and marital conduct that causes Gilberte to suffer."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p1-a-tansonville#p-11-p-15"
}

### Candidate characters

[
  "Albertine",
  "Andrée",
  "Gilberte",
  "M. Vinteuil",
  "M. de Marsantes",
  "Morel",
  "Odette",
  "baron de Charlus",
  "le narrateur"
]

### Prior local context (optional)

Robert de Saint-Loup insistait pour que je restasse à Tansonville et laissa échapper une fois, bien qu'il ne cherchât visiblement plus à me faire plaisir, que ma venue avait été pour sa femme une joie telle qu'elle en était restée, à ce qu'elle lui avait dit, transportée de joie tout un soir, un soir où elle se sentait si triste que je l'avais, en arrivant à l'improviste, miraculeusement sauvée du désespoir, « peut-être du pire », ajouta-t-il. Il me demandait de tâcher de la persuader qu'il l'aimait, me disant que la femme qu'il aimait aussi, il l'aimait moins qu'elle et romprait bientôt. « Et pourtant », ajouta-t-il, avec une telle félinité et un tel besoin de confidence que je croyais par moments que le nom de Morel allait, malgré Robert de Saint-Loup, « sortir » comme le numéro d'une loterie, « j'avais de quoi être fier. Cette femme qui me donna tant de preuves de sa tendresse et que je vais sacrifier à Gilberte, jamais elle n'avait fait attention à un homme, elle se croyait elle-même incapable d'être amoureuse. Je suis le premier. Je savais qu'elle s'était refusée à tout le monde tellement que, quand j'ai reçu la lettre adorable où elle me disait qu'il ne pouvait y avoir de bonheur pour elle qu'avec moi, je n'en revenais pas. Évidemment, il y aurait de quoi me griser, si la pensée de voir cette pauvre petite Gilberte en larmes ne m'était pas intolérable. Ne trouves-tu pas qu'elle a quelque chose de Rachel ? », me disait-il. Et en effet j'avais été frappé d'une vague ressemblance qu'on pouvait à la rigueur trouver maintenant entre elles. Peut-être tenait-elle à une similitude réelle de quelques traits (dus par exemple à l'origine hébraïque pourtant si peu marquée chez Gilberte) à cause de laquelle Robert de Saint-Loup, quand sa famille avait voulu qu'il se mariât, s'était senti attiré vers Gilberte. Elle tenait aussi à ce que Gilberte, ayant surpris des photographies de Rachel, cherchait pour plaire à Robert de Saint-Loup à imiter certaines habitudes chères à l'actrice, comme d'avoir toujours des noeuds rouges dans les cheveux, un ruban de velours noir au bras, et se teignait les cheveux pour paraître brune. Puis sentant que ses chagrins lui donnaient mauvaise mine, elle essayait d'y remédier. Elle le faisait parfois sans mesure. Un jour où Robert de Saint-Loup devait venir le soir pour vingt-quatre heures à Tansonville, je fus stupéfait de la voir venir se mettre à table si étrangement différente de ce qu'elle était, non seulement autrefois, mais même les jours habituels, que je restai stupéfait comme si j'avais eu devant moi une actrice, une espèce de Théodora. Je sentais que malgré moi je la regardais trop fixement dans ma curiosité de savoir ce qu'elle avait de changé. Cette curiosité fut d'ailleurs bientôt satisfaite quand elle se moucha, car, malgré toutes les précautions qu'elle y mit, par toutes les couleurs qui restèrent sur le mouchoir, en faisant une riche palette, je vis qu'elle était complètement peinte. C'était cela qui lui faisait cette bouche sanglante et qu'elle s'efforçait de rendre rieuse en croyant que cela lui allait bien, tandis que l'heure du train qui s'approchait sans que Gilberte sût si son mari arrivait vraiment ou s'il n'enverrait pas une de ces dépêches dont duc de Guermantes  avait spirituellement fixé le modèle : « Impossible venir, mensonge suit », pâlissait ses joues et cernait ses yeux.

### Passage

« Ah ! vois-tu, me disait Saint-Loup – avec un accent volontairement tendre qui contrastait tant avec sa tendresse spontanée d'autrefois, avec une voix d'alcoolique et des modulations d'acteur – Gilberte heureuse, il n'y a rien que je ne donnerais pour cela. Elle a tant fait pour moi. Tu ne peux pas savoir. » Et ce qui était le plus déplaisant dans tout cela était encore l'amour-propre, car Saint-Loup était flatté d'être aimé par Gilberte, et, sans oser dire que c'était Morel qu'il aimait, donnait pourtant sur l'amour que le violoniste était censé avoir pour lui des détails qu'il savait bien exagérés sinon inventés de toute pièce, lui à qui Morel demandait chaque jour plus d'argent. Et c'était en me confiant Gilberte qu'il repartait pour Paris.

J'eus, du reste, l'occasion, pour anticiper un peu, puisque je suis encore à Tansonville, de l'y apercevoir une fois dans le monde, et de loin, où sa parole, malgré tout vivante et charmante, me permettait de retrouver le passé. Je fus frappé de voir combien il changeait. Il ressemblait de plus en plus à sa mère. Mais la manière de sveltesse hautaine qu'il avait héritée d'elle et qu'elle avait parfaite, chez lui, grâce à l'éducation la plus accomplie, s'exagérait, se figeait ; la pénétration du regard propre aux Guermantes lui donnait l'air d'inspecter tous les lieux au milieu desquels il passait, mais d'une façon quasi inconsciente, par une sorte d'habitude et de particularité animale ; même immobile, la couleur qui était la sienne plus que de tous les Guermantes, d'être seulement de l'ensoleillement d'une journée d'or devenue solide, lui donnait comme un plumage si étrange, faisait de lui une espèce si rare, si précieuse, qu'on aurait voulu la posséder pour une collection ornithologique ; mais quand, de plus, cette lumière changée en oiseau se mettait en mouvement, en action, quand par exemple je voyais Saint-Loup de Saint-Loup entrer dans une soirée où j'étais, il avait des redressements de sa tête si joyeusement et si fièrement huppée sous l'aigrette d'or de ses cheveux un peu déplumés, des mouvements de cou tellement plus souples, plus fiers et plus coquets que n'en ont les humains, que devant la curiosité et l'admiration moitié mondaine, moitié zoologique qu'il vous inspirait, on se demandait si c'était dans le faubourg Saint-Germain qu'on se trouvait ou au Jardin des Plantes et si on regardait un grand seigneur traverser un salon, ou se promener dans sa cage un merveilleux oiseau. Pour peu qu'on y mît un peu d'imagination, le ramage ne se prêtait pas moins à cette interprétation que le plumage. Il disait ce qu'il croyait grand siècle et par là imitait les manières des Guermantes. Mais un rien d'indéfinissable faisait qu'elles devenaient les manières de Charlus. « Je te quitte un instant, me dit-il, dans cette soirée où Mme de Marsantes était un peu plus loin. Je vais faire un doigt de cour à ma nièce. » Quant à cet amour dont il me parlait sans cesse, il n'était pas d'ailleurs que celui pour Morel, bien que ce fût le seul qui comptât pour lui. Quel que soit le genre d'amours d'un homme, on se trompe toujours sur le nombre des personnes avec qui il a des liaisons, parce qu'on interprète faussement des amitiés comme des liaisons, ce qui est une erreur par addition, mais aussi parce qu'on croit qu'une liaison prouvée en exclut une autre, ce qui est un autre genre d'erreur. Deux personnes peuvent dire : « la maîtresse de X..., je la connais », prononcer deux noms différents et ne se tromper ni l'une ni l'autre. Une femme qu'on aime suffit rarement à tous nos besoins et on la trompe avec une femme qu'on n'aime pas. Quant au genre d'amours que Saint-Loup avait hérité de Charlus, un mari qui y est enclin fait habituellement le bonheur de sa femme. C'est une loi générale à laquelle les Guermantes trouvaient le moyen de faire exception parce que ceux qui avaient ce goût voulaient faire croire qu'ils avaient, au contraire, celui des femmes. Ils s'affichaient avec l'une ou l'autre et désespéraient la leur. Les Courvoisier en usaient plus sagement. Le jeune vicomte de Courvoisier se croyait seul sur la terre, et depuis l'origine du monde, à être tenté par quelqu'un de son sexe. Supposant que ce penchant lui venait du diable, il lutta contre lui, épousa une femme ravissante, lui fit des enfants... Puis un de ses cousins lui enseigna que ce penchant est assez répandu, poussa la bonté jusqu'à le mener dans des lieux où il pouvait le satisfaire. M. de Courvoisier n'en aima que plus sa femme, redoubla de zèle prolifique et elle et lui étaient cités comme le meilleur ménage de Paris. On n'en disait point autant de celui de Saint-Loup parce que Saint-Loup au lieu de se contenter de l'inversion, faisait mourir sa femme de jalousie en cherchant sans plaisir des maîtresses !

Il est possible que Morel, étant excessivement noir, fût nécessaire à Saint-Loup comme l'ombre l'est au rayon de soleil. On imagine très bien dans cette famille si ancienne un grand seigneur blond, doré, intelligent, doué de tous les prestiges et recelant à fond de cale un goût secret, ignoré de tous, pour les nègres. Saint-Loup, d'ailleurs, ne laissait jamais la conversation toucher à ce genre d'amours qui était le sien. Si je disais un mot : « Oh ! je ne sais pas, répondait-il avec un détachement si profond qu'il en laissait tomber son monocle, je n'ai pas soupçon de ces choses-là. Si tu désires des renseignements là-dessus, mon cher, je te conseille de t'adresser ailleurs. Moi, je suis un soldat, un point c'est tout. Autant ces choses-là m'indiffèrent, autant je suis avec passion la guerre balkanique. Autrefois cela t'intéressait, l'histoire des batailles. Je te disais alors qu'on reverrait, même dans les conditions les plus différentes, les batailles typiques, par exemple le grand essai d'enveloppement par l'aile de la bataille d'Ulm. Eh bien ! si spéciales que soient ces guerres balkaniques, Lullé-Burgas c'est encore Ulm, l'enveloppement par l'aile. Voilà les sujets dont tu peux me parler. Mais pour le genre de choses auxquelles tu fais allusion, je m'y connais autant qu'en sanscrit. » Ces sujets que Saint-Loup dédaignait ainsi, Gilberte, au contraire, quand il était reparti, les abordait volontiers en causant avec moi. Non, certes, relativement à son mari car elle ignorait, ou feignait d'ignorer tout. Mais elle s'étendait volontiers sur eux en tant qu'ils concernaient les autres, soit qu'elle y vît une sorte d'excuse indirecte pour Saint-Loup, soit que celui-ci, partagé comme son oncle entre un silence sévère à l'égard de ces sujets et un besoin de s'épancher et de médire, l'eût instruite pour beaucoup. Entre tous, Charlus n'était pas épargné ; c'était sans doute que Saint-Loup, sans parler de Morel à Gilberte, ne pouvait s'empêcher, avec elle, de lui répéter, sous une forme ou sous une autre, ce que le violoniste lui avait appris. Et il poursuivait son ancien bienfaiteur de sa haine. Ces conversations, que Gilberte affectionnait, me permirent de lui demander si, dans un genre parallèle, Albertine, dont c'est par elle que j'avais entendu la première fois le nom, quand jadis elles étaient amies de cours, avait de ces goûts. Gilberte refusa de me donner ce renseignement. Au reste, il y avait longtemps qu'il eût cessé d'offrir quelque intérêt pour moi. Mais je continuais à m'en enquérir machinalement, comme un vieillard qui, ayant perdu la mémoire, demande de temps à autre des nouvelles du fils qu'il a perdu.

Un autre jour je revins à la charge et demandai encore à Gilberte si Albertine aimait les femmes. « Oh ! pas du tout. – Mais vous disiez autrefois qu'elle avait mauvais genre. – J'ai dit cela, moi ? vous devez vous tromper. En tout cas si je l'ai dit – mais vous faites erreur – je parlais au contraire d'amourettes avec des jeunes gens. À cet âge-là, du reste, cela n'allait probablement pas bien loin. »

Gilberte disait-elle cela pour me cacher qu'elle-même, selon ce qu'Albertine m'avait dit, aimait les femmes et avait fait à Albertine des propositions ? Ou bien (car les autres sont souvent plus renseignés sur notre vie que nous ne croyons) savait-elle que j'avais aimé, que j'avais été jaloux d'Albertine et (les autres pouvant savoir plus de vérité que nous ne croyons, mais l'étendre aussi trop loin et être dans l'erreur par des suppositions excessives, alors que nous les avions espérés dans l'erreur par l'absence de toute supposition) s'imaginait-elle que je l'étais encore et me mettait-elle sur les yeux, par bonté, ce bandeau qu'on a toujours tout prêt pour les jaloux ? En tout cas, les paroles de Gilberte, depuis « le mauvais genre » d'autrefois jusqu'au certificat de bonne vie et moeurs d'aujourd'hui, suivaient une marche inverse des affirmations d'Albertine qui avait fini presque par avouer des demi-rapports avec Gilberte. Albertine m'avait étonné en cela comme sur ce que m'avait dit Andrée, car pour toute cette petite bande, si j'avais d'abord cru, avant de la connaître, à sa perversité, je m'étais rendu compte de mes fausses suppositions, comme il arrive si souvent quand on trouve une honnête fille et presque ignorante des réalités de l'amour dans le milieu qu'on avait cru à tort le plus dépravé. Puis j'avais refait le chemin en sens contraire, reprenant pour vraies mes suppositions du début. Mais peut-être Albertine avait-elle voulu me dire cela pour avoir l'air plus expérimentée qu'elle n'était et pour m'éblouir, à Paris, du prestige de sa perversité comme la première fois, à Balbec, par celui de sa vertu. Et tout simplement, quand je lui avais parlé des femmes qui aimaient les femmes, pour ne pas avoir l'air de ne pas savoir ce que c'était, comme dans une conversation on prend un air entendu si on parle de Fourier ou de Tobolsk encore qu'on ne sache pas ce que c'est. Elle avait peut-être vécu près de l'amie de Mlle Vinteuil et d'Andrée, séparée par une cloison étanche d'elles qui croyaient qu'elle n'en était pas, ne s'était renseignée ensuite – comme une femme qui épouse un homme de lettres cherche à se cultiver – qu'afin de me complaire en se faisant capable de répondre à mes questions, jusqu'au jour où elle avait compris qu'elles étaient inspirées par la jalousie et où elle avait fait machine en arrière, à moins que ce ne fût Gilberte qui me mentît. L'idée me vint que c'était pour avoir appris d'elle, au cours d'un flirt qu'il aurait conduit dans le sens qui l'intéressait, qu'elle ne détestait pas les femmes, que Saint-Loup l'avait épousée, espérant des plaisirs qu'il n'avait pas dû trouver chez lui puisqu'il les prenait ailleurs. Aucune de ces hypothèses n'était absurde, car chez des femmes comme la fille d'Odette ou les jeunes filles de la petite bande il y a une telle diversité, un tel cumul de goûts alternants, si même ils ne sont pas simultanés, qu'elles passent aisément d'une liaison avec une femme à un grand amour pour un homme, si bien que définir le goût réel et dominant reste difficile. C'est ainsi qu'Albertine avait cherché à me plaire pour me décider à l'épouser, mais elle y avait renoncé elle-même à cause de mon caractère indécis et tracassier. C'était, en effet, sous cette forme trop simple que je jugeais mon aventure avec Albertine, maintenant que je ne voyais plus cette aventure que du dehors.
