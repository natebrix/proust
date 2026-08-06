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
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "mon oncle baron de Charlus",
        "baron de Charlus",
        "le baron de Guermantes"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
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
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "Regard « terrible et profond » lancé aux inconnus, feinte distraction et hauteur, poignée de main à deux doigts avec « Heue, heue, heue », « ne m'honora … pas d'une parole mais même d'un regard ».",
      "explanation": "The introductory scene shows him affected, haughty, and aggressively indifferent, which makes him appear socially discourteous and vaguely unsettling despite his prestige."
    },
    {
      "event_id": "E2",
      "source": "Robert de Saint-Loup",
      "target": "baron de Charlus",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.9,
      "evidence": "« il donnait le ton, … faisait la loi à toute la société »; tout était aussitôt imité par les snobs (boire au théâtre, pardessus de vigogne, dîner en veston, façons de manger, quatuors de Beethoven).",
      "explanation": "Robert presents Charlus as an arbiter of elegance and a model that society imitates, which raises him significantly in the social order."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.9,
      "explanation": "He is positioned as a trendsetter and imitated authority, which clearly enhances his local social position."
    },
    {
      "character": "baron de Charlus",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "His affected behavior and refusal to look/speak make him appear unpleasant and socially discordant in the scene."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-196-p-200"
}

### Candidate characters

[
  "M. de Marsantes",
  "Mme de Villeparisis",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Robert de Saint-Loup me parla de la jeunesse, depuis longtemps passée, de son oncle. Il amenait tous les jours des femmes dans une garçonnière qu'il avait en commun avec deux de ses amis, beaux comme lui, ce qui faisait qu'on les appelait « les trois Grâces ».

### Passage

– Un jour un des hommes qui est aujourd'hui des plus en vue dans le faubourg Saint-Germain, comme eût dit Balzac, mais qui dans une première période assez fâcheuse montrait des goûts bizarres, avait demandé à mon oncle de venir dans cette garçonnière. Mais à peine arrivé ce ne fut pas aux femmes, mais à mon oncle Charlus, qu'il se mit à faire une déclaration. Mon oncle fit semblant de ne pas comprendre, emmena sous un prétexte ses deux amis, ils revinrent, prirent le coupable, le déshabillèrent, le frappèrent jusqu'au sang, et par un froid de dix degrés au-dessous de zéro le jetèrent à coups de pieds dehors où il fut trouvé à demi mort, si bien que la justice fit une enquête à laquelle le malheureux eut toute la peine du monde à la faire renoncer. Mon oncle ne se livrerait plus aujourd'hui à une exécution aussi cruelle et tu n'imagines pas le nombre d'hommes du peuple, lui si hautain avec les gens du monde, qu'il prend en affection, qu'il protège, quitte à être payé d'ingratitude. Ce sera un domestique qui l'aura servi dans un hôtel et qu'il placera à Paris, ou un paysan à qui il fera apprendre un métier. C'est même le côté assez gentil qu'il y a chez lui, par contraste avec le côté mondain. » Saint-Loup appartenait, en effet, à ce genre de jeunes gens du monde, situés à une altitude où on ait pu faire pousser ces expressions : « Ce qu'il y a même d'assez gentil chez lui, son côté assez gentil », semences assez précieuses, produisant très vite une manière de concevoir les choses dans laquelle on se compte pour rien, et le « peuple » pour tout ; en somme tout le contraire de l'orgueil plébéien. « Il paraît qu'on ne peut se figurer comme il donnait le ton, comme il faisait la loi à toute la société dans sa jeunesse. Pour lui en toute circonstance il faisait ce qui lui paraissait le plus agréable, le plus commode, mais aussitôt c'était imité par les snobs. S'il avait eu soif au théâtre et s'était fait apporter à boire dans le fond de sa loge, les petits salons qu'il y avait derrière chacune se remplissaient, la semaine suivante, de rafraîchissements. Un été pluvieux où il avait un peu de rhumatisme, il s'était commandé un pardessus d'une vigogne souple mais chaude qui ne sert que pour faire des couvertures de voyage et dont il avait respecté les raies bleues et oranges. Les grands tailleurs se virent commander aussitôt par leurs clients des pardessus bleus et frangés, à longs poils. Si pour une raison quelconque il désirait ôter tout caractère de solennité à un dîner dans un château où il passait une journée, et pour marquer cette nuance n'avait pas apporté d'habits et s'était mis à table avec le veston de l'après-midi, la mode devenait de dîner à la campagne en veston. Que pour manger un gâteau il se servît, au lieu de sa cuiller, d'une fourchette ou d'un couvert de son invention commandé par lui à un orfèvre, ou de ses doigts, il n'était plus permis de faire autrement. Il avait eu envie de réentendre certains quatuors de Beethoven (car avec toutes ses idées saugrenues il est loin d'être bête, et est fort doué) et avait fait venir des artistes pour les jouer chaque semaine, pour lui et quelques amis. La grande élégance fut cette année-là de donner des réunions peu nombreuses où on entendait de la musique de chambre. Je crois d'ailleurs qu'il ne s'est pas ennuyé dans la vie. Beau comme il a été, il a dû avoir des femmes ! Je ne pourrais pas vous dire d'ailleurs exactement lesquelles parce qu'il est très discret. Mais je sais qu'il a bien trompé ma pauvre tante. Ce qui n'empêche pas qu'il était délicieux avec elle, qu'elle l'adorait, et qu'il l'a pleurée pendant des années. Quand il est à Paris, il va encore au cimetière presque chaque jour. »

Le lendemain du jour où Saint-Loup m'avait ainsi parlé de son oncle tout en l'attendant, vainement du reste, comme je passais seul devant le casino en rentrant à l'hôtel, j'eus la sensation d'être regardé par quelqu'un qui n'était pas loin de moi. Je tournai la tête et j'aperçus un homme d'une quarantaine d'années, très grand et assez gros, avec des moustaches très noires, et qui, tout en frappant nerveusement son pantalon avec une badine, fixait sur moi des yeux dilatés par l'attention. Par moments, ils étaient percés en tous sens par des regards d'une extrême activité comme en ont seuls devant une personne qu'ils ne connaissent pas des hommes à qui, pour un motif quelconque, elle inspire des pensées qui ne viendraient pas à tout autre – par exemple, des fous ou des espions. Il lança sur moi une suprême oeillade à la fois hardie, prudente, rapide et profonde, comme un dernier coup que l'on tire au moment de prendre la fuite, et après avoir regardé tout autour de lui, prenant soudain un air distrait et hautain, par un brusque revirement de toute sa personne il se tourna vers une affiche dans la lecture de laquelle il s'absorba, en fredonnant un air et en arrangeant la rose mousseuse qui pendait à sa boutonnière. Il sortit de sa poche un calepin sur lequel il eut l'air de prendre en note le titre du spectacle annoncé, tira deux ou trois fois sa montre, abaissa sur ses yeux un canotier de paille noire dont il prolongea le rebord avec sa main mise en visière comme pour voir si quelqu'un n'arrivait pas, fit le geste de mécontentement par lequel on croit faire voir qu'on a assez d'attendre, mais qu'on ne fait jamais quand on attend réellement, puis rejetant en arrière son chapeau et laissant voir une brosse coupée ras qui admettait cependant de chaque côté d'assez longues ailes de pigeon ondulées, il exhala le souffle bruyant des personnes qui ont non pas trop chaud mais le désir de montrer qu'elles ont trop chaud. J'eus l'idée d'un escroc d'hôtel qui, nous ayant peut-être déjà remarqués les jours précédents ma grand'mère et moi, et préparant quelque mauvais coup, venait de s'apercevoir que je l'avais surpris pendant qu'il m'épiait ; pour me donner le change, peut-être cherchait-il seulement par sa nouvelle attitude à exprimer la distraction et le détachement, mais c'était avec une exagération si agressive que son but semblait, au moins autant que de dissiper les soupçons que j'avais dû avoir, de venger une humiliation qu'à mon insu je lui eusse infligée, de me donner l'idée non pas tant qu'il ne m'avait pas vu, que celle que j'étais un objet de trop petite importance pour attirer l'attention. Il cambrait sa taille d'un air de bravade, pinçait les lèvres, relevait ses moustaches et dans son regard ajustait quelque chose d'indifférent, de dur, de presque insultant. Si bien que la singularité de son expression me le faisait prendre tantôt pour un voleur et tantôt pour un aliéné. Pourtant sa mise extrêmement soignée était beaucoup plus grave et beaucoup plus simple que celles de tous les baigneurs que je voyais à Balbec, et rassurante pour mon veston si souvent humilié par la blancheur éclatante et banale de leurs costumes de plage. Mais ma grand'mère venait à ma rencontre, nous fîmes un tour ensemble et je l'attendais, une heure après, devant l'hôtel où elle était rentrée un instant, quand je vis sortir Mme de Villeparisis avec Saint-Loup de Saint-Loup et l'inconnu qui m'avait regardé si fixement devant le casino. Avec la rapidité d'un éclair son regard me traversa, ainsi qu'au moment où je l'avais aperçu, et revint, comme s'il ne m'avait pas vu, se ranger, un peu bas, devant ses yeux, émoussé comme le regard neutre qui feint de ne rien voir au dehors et n'est capable de rien dire au dedans, le regard qui exprime seulement la satisfaction de sentir autour de soi les cils qu'il écarte de sa rondeur béate, le regard dévot et confit qu'ont certains hypocrites, le regard fat qu'ont certains sots. Je vis qu'il avait changé de costume. Celui qu'il portait était encore plus sombre ; et sans doute c'est que la véritable élégance est moins loin de la simplicité que la fausse ; mais il y avait autre chose : d'un peu près on sentait que si la couleur était presque entièrement absente de ces vêtements, ce n'était pas parce que celui qui l'en avait bannie y était indifférent, mais plutôt parce que pour une raison quelconque il se l'interdisait. Et la sobriété qu'ils laissaient paraître semblait de celles qui viennent de l'obéissance à un régime, plutôt que du manque de gourmandise. Un filet de vert sombre s'harmonisait dans le tissu du pantalon à la rayure des chaussettes avec un raffinement qui décelait la vivacité d'un goût maté partout ailleurs et à qui cette seule concession avait été faite par tolérance, tandis qu'une tache rouge sur la cravate était imperceptible comme une liberté qu'on n'ose prendre.

– Comment, allez-vous ? Je vous présente mon neveu, le baron de Guermantes, me dit Mme de Villeparisis, pendant que l'inconnu, sans me regarder, grommelant un vague : « Charmé » qu'il fit suivre de : « Heue, heue, heue » pour donner à son amabilité quelque chose de forcé, et repliant le petit doigt, l'index et le pouce, me tendait le troisième doigt et l'annulaire, dépourvus de toute bague, que je serrai sous son gant de Suède ; puis sans avoir levé les yeux sur moi il se détourna vers Mme de Villeparisis.

– Mon Dieu, est-ce que je perds la tête ? dit celle-ci, voilà que je t'appelle le baron de Guermantes. Je vous présente le baron de Charlus. Après tout, l'erreur n'est pas si grande, ajouta-t-elle, tu es bien un Guermantes tout de même.

Cependant ma grand'mère sortait, nous fîmes route ensemble. L'oncle de Saint-Loup ne m'honora non seulement pas d'une parole mais même d'un regard. S'il dévisageait les inconnus (et pendant cette courte promenade il lança deux ou trois fois son terrible et profond regard en coup de sonde sur des gens insignifiants et de la plus modeste extraction qui passaient), en revanche, il ne regardait à aucun moment, si j'en jugeais par moi, les personnes qu'il connaissait – comme un policier en mission secrète mais qui tient ses amis en dehors de sa surveillance professionnelle. Les laissant causer ensemble, ma grand'mère, Mme de Villeparisis et lui, je retins Saint-Loup en arrière :
