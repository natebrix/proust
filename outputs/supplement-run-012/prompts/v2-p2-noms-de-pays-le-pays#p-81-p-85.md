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
      "canonical_name": "le directeur",
      "surface_forms": [
        "le Directeur général",
        "Directeur général",
        "le directeur",
        "le propriétaire"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.95
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "le directeur",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« connu... pour un des premiers hôteliers de l'Europe »; « d'une impassibilité et d'une correction extraordinaires »; « véritable généralissime »",
      "explanation": "The narrator elevates the Director General as an eminent, authoritative figure who commands the room with disciplined control and international prestige."
    }
  ],
  "status_effects": [
    {
      "character": "le directeur",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He is framed as a top European hotelier and commanding presence, which markedly raises his local standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-81-p-85"
}

### Candidate characters

[
  "Aimé",
  "M. de Stermaria",
  "Mlle de Stermaria",
  "Mme de Villeparisis",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

– Nos amis communs, les de Cambremer, voulaient justement nous réunir, nos jours n'ont pas coïncidé, enfin je ne sais plus, dit le bâtonnier, qui comme beaucoup de menteurs s'imaginent qu'on ne cherchera pas à élucider un détail insignifiant qui suffit pourtant (si le hasard vous met en possession de l'humble réalité qui est en contradiction avec lui) pour dénoncer un caractère et inspirer à jamais la méfiance.

### Passage

Comme toujours, mais plus facilement pendant que son père s'était éloigné pour causer avec le bâtonnier, je regardais Mlle de Stermaria. Autant que la singularité hardie et toujours belle de ses attitudes, comme quand, les deux coudes posés sur la table, elle élevait son verre au-dessus de ses deux avant-bras, la sécheresse d'un regard vite épuisé, la dureté foncière, familiale, qu'on sentait, mal recouverte sous ses inflexions personnelles, au fond de sa voix, et qui avait choqué ma grand'mère, une sorte de cran d'arrêt atavique auquel elle revenait dès que dans un coup d'oeil ou une intonation elle avait achevé de donner sa pensée propre ; tout cela ramenait la pensée de celui qui la regardait vers la lignée qui lui avait légué cette insuffisance de sympathie humaine, des lacunes de sensibilité, un manque d'ampleur dans l'étoffe qui à tout moment faisait faute. Mais à certains regards qui passaient un instant sur le fond si vite à sec de sa prunelle et dans lesquels on sentait cette douceur presque humble que le goût prédominant des plaisirs des sens donne à la plus fière, laquelle bientôt ne reconnaît plus qu'un prestige, celui qu'a pour elle tout être qui peut les lui faire éprouver, fût-ce un comédien ou un saltimbanque pour lequel elle quittera peut-être un jour son mari ; à certaine teinte d'un rose sensuel et vif qui s'épanouissait dans ses joues pâles, pareille à celle qui mettait son incarnat au coeur des nymphéas blancs de la Vivonne, je croyais sentir qu'elle eût facilement permis que je vinsse chercher sur elle le goût de cette vie si poétique qu'elle menait en Bretagne, vie à laquelle, soit par trop d'habitude, soit par distinction innée, soit par dégoût de la pauvreté ou de l'avarice des siens, elle ne semblait pas trouver grand prix, mais que pourtant elle contenait enclose en son corps. Dans la chétive réserve de volonté qui lui avait été transmise et qui donnait à son expression quelque chose de lâche, peut-être n'eût-elle pas trouvé les ressources d'une résistance. Et surmonté d'une plume un peu démodée et prétentieuse, le feutre gris qu'elle portait invariablement à chaque repas me la rendait plus douce, non parce qu'il s'harmonisait avec son teint d'argent ou de rose, mais parce qu'en me la faisant supposer pauvre, il la rapprochait de moi. Obligée à une attitude de convention par la présence de son père, mais apportant déjà à la perception et au classement des êtres qui étaient devant elle des principes autres que lui, peut-être voyait-elle en moi non le rang insignifiant, mais le sexe et l'âge. Si un jour M. de Stermaria était sorti sans elle, surtout si Mme de Villeparisis en venant s'asseoir à notre table lui avait donné de nous une opinion qui m'eût enhardi à m'approcher d'elle, peut-être aurions-nous pu échanger quelques paroles, prendre un rendez-vous, nous lier davantage. Et, un mois où elle serait restée seule sans ses parents dans son château romanesque, peut-être aurions-nous pu nous promener seuls le soir tous deux dans le crépuscule où luiraient plus doucement au-dessus de l'eau assombrie les fleurs roses des bruyères, sous les chênes battus par le clapotement des vagues. Ensemble nous aurions parcouru cette île empreinte pour moi de tant de charme parce qu'elle avait enfermé la vie habituelle de Mlle de Stermaria et qu'elle reposait dans la mémoire de ses yeux. Car il me semblait que je ne l'aurais vraiment possédée que là, quand j'aurais traversé ces lieux qui l'enveloppaient de tant de souvenirs – voile que mon désir voulait arracher et de ceux que la nature interpose entre la femme et quelques êtres (dans la même intention qui lui fait, pour tous, mettre l'acte de la reproduction entre eux et le plus vif plaisir, et pour les insectes, placer devant le nectar le pollen qu'ils doivent emporter) afin que trompés par l'illusion de la posséder ainsi plus entière ils soient forcés de s'emparer d'abord des paysages au milieu desquels elle vit et qui, plus utiles pour leur imagination que le plaisir sensuel, n'eussent pas suffi pourtant, sans lui, à les attirer.

Mais je dus détourner mes regards de Mlle de Stermaria, car déjà, considérant sans doute que faire la connaissance d'une personnalité importante était un acte curieux et bref qui se suffisait à lui-même et qui pour développer tout l'intérêt qu'il comportait n'exigeait qu'une poignée de mains et un coup d'oeil pénétrant sans conversation immédiate ni relations ultérieures, son père avait pris congé du bâtonnier et retournait s'asseoir en face d'elle, en se frottant les mains comme un homme qui vient de faire une précieuse acquisition. Quant au bâtonnier, la première émotion de cette entrevue une fois passée, comme les autres jours, on l'entendait par moments s'adressant au maître d'hôtel :

– Mais moi je ne suis pas roi, Aimé ; allez donc près du roi... Dites, Premier, cela a l'air très bon ces petites truites-là, nous allons en demander à Aimé. Aimé, cela me semble tout à fait recommandable ce petit poisson que vous avez là-bas : vous allez nous apporter de cela, Aimé, et à discrétion.

Il répétait tout le temps le nom d'Aimé, ce qui faisait que quand il avait quelqu'un à dîner, son invité lui disait : « Je vois que vous êtes tout à fait bien dans la maison » et croyait devoir aussi prononcer constamment « Aimé » par cette disposition, où il entre à la fois de la timidité, de la vulgarité et de la sottise, qu'ont certaines personnes à croire qu'il est spirituel et élégant d'imiter à la lettre les gens avec qui elles se trouvent. Il le répétait sans cesse, mais avec un sourire, car il tenait à étaler à la fois ses bonnes relations avec le maître d'hôtel et sa supériorité sur lui. Et le maître d'hôtel lui aussi, chaque fois que revenait son nom, souriait d'un air attendri et fier, montrant qu'il ressentait l'honneur et comprenait la plaisanterie.

Si intimidants que fussent toujours pour moi les repas, dans ce vaste restaurant, habituellement comble, du Grand-Hôtel, ils le devenaient davantage encore quand arrivait pour quelques jours le propriétaire (ou directeur général élu par une société de commanditaires, je ne sais) non seulement de ce palace mais de sept ou huit autres, situés aux quatre coins de la France, et dans chacun desquels, faisant entre eux la navette, il venait passer, de temps en temps, une semaine. Alors, presque au commencement du dîner, apparaissait chaque soir, à l'entrée de la salle à manger, cet homme petit, à cheveux blancs, à nez rouge, d'une impassibilité et d'une correction extraordinaires, et qui était connu, paraît-il, à Londres aussi bien qu'à Monte-Carlo, pour un des premiers hôteliers de l'Europe. Une fois que j'étais sorti un instant au commencement du dîner, comme en rentrant je passai devant lui, il me salua, mais avec une froideur dont je ne pus démêler si la cause était la réserve de quelqu'un qui n'oublie pas ce qu'il est, ou le dédain pour un client sans importance. Devant ceux qui en avaient au contraire une très grande, le Directeur général s'inclinait avec autant de froideur mais plus profondément, les paupières abaissées par une sorte de respect pudique, comme s'il eût eu devant lui, à un enterrement, le père de la défunte ou le Saint-Sacrement. Sauf pour ces saluts glacés et rares, il ne faisait pas un mouvement, comme pour montrer que ses yeux étincelants qui semblaient lui sortir de la figure, voyaient tout, réglaient tout, assuraient dans « le Dîner au Grand-Hôtel » aussi bien le fini des détails que l'harmonie de l'ensemble. Il se sentait évidemment plus que metteur en scène, que chef d'orchestre, véritable généralissime. Jugeant qu'une contemplation portée à son maximum d'intensité lui suffisait pour s'assurer que tout était prêt, qu'aucune faute commise ne pouvait entraîner la déroute, et pour prendre enfin ses responsabilités, il s'abstenait non seulement de tout geste, même de bouger ses yeux pétrifiés par l'attention qui embrassaient et dirigeaient la totalité des opérations. Je sentais que les mouvements de ma cuiller eux-mêmes ne lui échappaient pas, et s'éclipsât-il dès après le potage, pour tout le dîner, la revue qu'il venait de passer m'avait coupé l'appétit. Le sien était fort bon, comme on pouvait le voir au déjeuner qu'il prenait comme un simple particulier, à la même table que tout le monde, dans la salle à manger. Sa table n'avait qu'une particularité, c'est qu'à côté, pendant qu'il mangeait, l'autre directeur, l'habituel, restait debout tout le temps à faire la conversation. Car étant le subordonné du Directeur général, il cherchait à le flatter et avait de lui une grande peur. La mienne était moindre pendant ces déjeuners, car perdu alors au milieu des clients, il mettait la discrétion d'un général assis dans un restaurant où se trouvent aussi des soldats à ne pas avoir l'air de s'occuper d'eux. Néanmoins quand le concierge, entouré de ses « chasseurs », m'annonçait : « Il repart demain matin pour Dinard. De là il va à Biarritz et après à Cannes », je respirais plus librement.
