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
      "canonical_name": "Legrandin",
      "surface_forms": [
        "Legrandin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.96
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Legrandin",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.86,
      "evidence": "« il était aux yeux de ma famille, qui le citait toujours en exemple, le type de l'homme d'élite, prenant la vie de la façon la plus noble et la plus délicate »; « d'une politesse raffinée, causeur comme nous n'en avions jamais entendu »",
      "explanation": "The narrator’s family sets him up as a model of the elite and admires his conversation and manners, which elevates him locally."
    },
    {
      "event_id": "E2",
      "source": "la grand-mère",
      "target": "Legrandin",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.72,
      "evidence": "« Ma la grand-mère lui reprochait seulement de parler un peu trop bien, un peu trop comme un livre »; « Elle s'étonnait aussi des tirades enflammées qu'il entamait souvent contre l'aristocratie, la vie mondaine, le snobisme »",
      "explanation": "The grandmother reproaches him for a language that is too literary and for emphatic diatribes, which slightly undermines his rhetorical credit despite the general admiration."
    }
  ],
  "status_effects": [
    {
      "character": "Legrandin",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "He is locally valued as a man of the elite and an example of nobility of life by the family."
    },
    {
      "character": "Legrandin",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.72,
      "explanation": "His manner of speech and his tirades are judged affected by the grandmother, which diminishes his rhetorical position."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-96-p-100"
}

### Candidate characters

[
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

On reconnaissait le clocher de Saint-Hilaire de bien loin, inscrivant sa figure inoubliable à l'horizon où Combray n'apparaissait pas encore ; quand du train qui, la semaine de Pâques, nous amenait de Paris, le père du narrateur l'apercevait qui filait tour à tour sur tous les sillons du ciel, faisant courir en tous sens son petit coq de fer, il nous disait : « Allons, prenez les couvertures, on est arrivé. » Et dans une des plus grandes promenades que nous faisions de Combray, il y avait un endroit où la route resserrée débouchait tout à coup sur un immense plateau fermé à l'horizon par des forêts déchiquetées que dépassait seul la fine pointe du clocher de Saint-Hilaire, mais si mince, si rose, qu'elle semblait seulement rayée sur le ciel par un ongle qui aurait voulu donner à ce paysage, à ce tableau rien que de nature, cette petite marque d'art, cette unique indication humaine. Quand on se rapprochait et qu'on pouvait apercevoir le reste de la tour carrée et à demi détruite qui, moins haute, subsistait à côté de lui, on était frappé surtout du ton rougeâtre et sombre des pierres ; et, par un matin brumeux d'automne, on aurait dit, s'élevant au-dessus du violet orageux des vignobles, une ruine de pourpre presque de la couleur de la vigne vierge.

### Passage

Souvent sur la place, quand nous rentrions, ma grand'mère me faisait arrêter pour le regarder. Des fenêtres de sa tour, placées deux par deux les unes au-dessus des autres, avec cette juste et originale proportion dans les distances qui ne donne pas de la beauté et de la dignité qu'aux visages humains, il lâchait, laissait tomber à intervalles réguliers des volées de corbeaux qui, pendant un moment, tournoyaient en criant, comme si les vieilles pierres qui les laissaient s'ébattre sans paraître les voir, devenues tout d'un coup inhabitables et dégageant un principe d'agitation infinie, les avait frappés et repoussés. Puis, après avoir rayé en tous sens le velours violet de l'air du soir, brusquement calmés ils revenaient s'absorber dans la tour, de néfaste redevenue propice, quelques-uns posés çà et là, ne semblant pas bouger, mais happant peut-être quelque insecte, sur la pointe d'un clocheton, comme une mouette arrêtée avec l'immobilité d'un pêcheur à la crête d'une vague. Sans trop savoir pourquoi, ma grand'mère trouvait au clocher de Saint-Hilaire cette absence de vulgarité, de prétention, de mesquinerie, qui lui faisait aimer et croire riches d'une influence bienfaisante la nature quand la main de l'homme ne l'avait pas, comme faisait le jardinier de ma grand'tante, rapetissée, et les oeuvres de génie. Et sans doute, toute partie de l'église qu'on apercevait la distinguait de tout autre édifice par une sorte de pensée qui lui était infuse, mais c'était dans son clocher qu'elle semblait prendre conscience d'elle-même, affirmer une existence individuelle et responsable. C'était lui qui parlait pour elle. Je crois surtout que, confusément, ma grand'mère trouvait au clocher de Combray ce qui pour elle avait le plus de prix au monde, l'air naturel et l'air distingué. Ignorante en architecture, elle disait : « Mes enfants, moquez-vous de moi si vous voulez, il n'est peut-être pas beau dans les règles, mais sa vieille figure bizarre me plaît. Je suis sûre que s'il jouait du piano, il ne jouerait pas sec. » Et en le regardant, en suivant des yeux la douce tension, l'inclinaison fervente de ses pentes de pierre qui se rapprochaient en s'élevant comme des mains jointes qui prient, elle s'unissait si bien à l'effusion de la flèche, que son regard semblait s'élancer avec elle ; et en même temps elle souriait amicalement aux vieilles pierres usées dont le couchant n'éclairait plus que le faîte et qui, à partir du moment où elles entraient dans cette zone ensoleillée, adoucies par la lumière, paraissaient tout d'un coup montées bien plus haut, lointaines, comme un chant repris « en voix de tête » une octave au-dessus.

C'était le clocher de Saint-Hilaire qui donnait à toutes les occupations, à toutes les heures, à tous les points de vue de la ville, leur figure, leur couronnement, leur consécration. De ma chambre, je ne pouvais apercevoir que sa base qui avait été recouverte d'ardoises ; mais quand, le dimanche, je les voyais, par une chaude matinée d'été, flamboyer comme un soleil noir, je me disais : « Mon Dieu ! neuf heures ! il faut se préparer pour aller à la grand'messe si je veux avoir le temps d'aller embrasser tante Léonie avant », et je savais exactement la couleur qu'avait le soleil sur la place, la chaleur et la poussière du marché, l'ombre que faisait le store du magasin où maman entrerait peut-être avant la messe, dans une odeur de toile écrue, faire emplette de quelque mouchoir que lui ferait montrer, en cambrant la taille, le patron qui, tout en se préparant à fermer, venait d'aller dans l'arrière-boutique passer sa veste du dimanche et se savonner les mains qu'il avait l'habitude, toutes les cinq minutes, même dans les circonstances les plus mélancoliques, de frotter l'une contre l'autre d'un air d'entreprise, de partie fine et de réussite.

Quand après la messe, on entrait dire à Théodore d'apporter une brioche plus grosse que d'habitude parce que nos cousins avaient profité du beau temps pour venir de Thiberzy déjeuner avec nous, on avait devant soi le clocher qui, doré et cuit lui-même comme une plus grande brioche bénie, avec des écailles et des égouttements gommeux de soleil, piquait sa pointe aiguë dans le ciel bleu. Et le soir, quand je rentrais de promenade et pensais au moment où il faudrait tout à l'heure dire bonsoir à ma mère et ne plus la voir, il était au contraire si doux, dans la journée finissante, qu'il avait l'air d'être posé et enfoncé comme un coussin de velours brun sur le ciel pâli qui avait cédé sous sa pression, s'était creusé légèrement pour lui faire sa place et refluait sur ses bords ; et les cris des oiseaux qui tournaient autour de lui semblaient accroître son silence, élancer encore sa flèche et lui donner quelque chose d'ineffable.

Même dans les courses qu'on avait à faire derrière l'église, là où on ne la voyait pas, tout semblait ordonné par rapport au clocher surgi ici ou là entre les maisons, peut-être plus émouvant encore quand il apparaissait ainsi sans l'église. Et certes, il y en a bien d'autres qui sont plus beaux vus de cette façon, et j'ai dans mon souvenir des vignettes de clochers dépassant les toits, qui ont un autre caractère d'art que celles que composaient les tristes rues de Combray. Je n'oublierai jamais dans une curieuse ville de Normandie voisine de Balbec, deux charmants hôtels du XVIIIe siècle, qui me sont à beaucoup d'égards chers et vénérables et entre lesquels, quand on la regarde du beau jardin qui descend des perrons vers la rivière, la flèche gothique d'une église qu'ils cachent s'élance, ayant l'air de terminer, de surmonter leurs façades, mais d'une matière si différente, si précieuse, si annelée, si rose, si vernie, qu'on voit bien qu'elle n'en fait pas plus partie que de deux beaux galets unis, entre lesquels elle est prise sur la plage, la flèche purpurine et crénelée de quelque coquillage fuselé en tourelle et glacé d'émail. Même à Paris, dans un des quartiers les plus laids de la ville, je sais une fenêtre où on voit après un premier, un second et même un troisième plan fait des toits amoncelés de plusieurs rues, une cloche violette, parfois rougeâtre, parfois aussi, dans les plus nobles « épreuves » qu'en tire l'atmosphère, d'un noir décanté de cendres, laquelle n'est autre que le dôme Saint-Augustin et qui donne à cette vue de Paris le caractère de certaines vues de Rome par Piranesi. Mais comme dans aucune de ces petites gravures, avec quelque goût que ma mémoire ait pu les exécuter, elle ne put mettre ce que j'avais perdu depuis longtemps, le sentiment qui nous fait non pas considérer une chose comme un spectacle, mais y croire comme en un être sans équivalent, aucune d'elles ne tient sous sa dépendance toute une partie profonde de ma vie, comme fait le souvenir de ces aspects du clocher de Combray dans les rues qui sont derrière l'église. Qu'on le vît à cinq heures, quand on allait chercher les lettres à la poste, à quelques maisons de soi, à gauche, surélevant brusquement d'une cime isolée la ligne de faîte des toits ; que si, au contraire, on voulait entrer demander des nouvelles de Mme Sazerat, on suivît des yeux cette ligne redevenue basse après la descente de son autre versant en sachant qu'il faudrait tourner à la deuxième rue après le clocher ; soit qu'encore, poussant plus loin, si on allait à la gare, on le vît obliquement, montrant de profil des arêtes et des surfaces nouvelles comme un solide surpris à un moment inconnu de sa révolution ; ou que, des bords de la Vivonne, l'abside musculeusement ramassée et remontée par la perspective semblât jaillir de l'effort que le clocher faisait pour lancer sa flèche au coeur du ciel ; c'était toujours à lui qu'il fallait revenir, toujours lui qui dominait tout, sommant les maisons d'un pinacle inattendu, levé devant moi comme le doigt de Dieu dont le corps eût été caché dans la foule des humains sans que je le confondisse pour cela avec elle. Et aujourd'hui encore si, dans une grande ville de province ou dans un quartier de Paris que je connais mal, un passant qui m'a « mis dans mon chemin » me montre au loin, comme un point de repère, tel beffroi d'hôpital, tel clocher de couvent levant la pointe de son bonnet ecclésiastique au coin d'une rue que je dois prendre, pour peu que ma mémoire puisse obscurément lui trouver quelque trait de ressemblance avec la figure chère et disparue, le passant, s'il se retourne pour s'assurer que je ne m'égare pas, peut, à son étonnement, m'apercevoir qui, oublieux de la promenade entreprise ou de la course obligée, reste là, devant le clocher, pendant des heures, immobile, essayant de me souvenir, sentant au fond de moi des terres reconquises sur l'oubli qui s'assèchent et se rebâtissent ; et sans doute alors, et plus anxieusement que tout à l'heure quand je lui demandais de me renseigner, je cherche encore mon chemin, je tourne une rue... mais... c'est dans mon coeur...

En rentrant de la messe, nous rencontrions souvent Legrandin qui, retenu à Paris par sa profession d'ingénieur, ne pouvait, en dehors des grandes vacances, venir à sa propriété de Combray que du samedi soir au lundi matin. C'était un de ces hommes qui, en dehors d'une carrière scientifique où ils ont d'ailleurs brillamment réussi, possèdent une culture toute différente, littéraire, artistique, que leur spécialisation professionnelle n'utilise pas et dont profite leur conversation. Plus lettrés que bien des littérateurs (nous ne savions pas à cette époque que Legrandin eût une certaine réputation comme écrivain et nous fûmes très étonnés de voir qu'un musicien célèbre avait composé une mélodie sur des vers de lui), doués de plus de « facilité » que bien des peintres, ils s'imaginent que la vie qu'ils mènent n'est pas celle qui leur aurait convenu et apportent à leurs occupations positives soit une insouciance mêlée de fantaisie, soit une application soutenue et hautaine, méprisante, amère et consciencieuse. Grand, avec une belle tournure, un visage pensif et fin aux longues moustaches blondes, au regard bleu et désenchanté, d'une politesse raffinée, causeur comme nous n'en avions jamais entendu, il était aux yeux de ma famille, qui le citait toujours en exemple, le type de l'homme d'élite, prenant la vie de la façon la plus noble et la plus délicate. Ma grand'mère lui reprochait seulement de parler un peu trop bien, un peu trop comme un livre, de ne pas avoir dans son langage le naturel qu'il y avait dans ses cravates lavallière toujours flottantes, dans son veston droit presque d'écolier. Elle s'étonnait aussi des tirades enflammées qu'il entamait souvent contre l'aristocratie, la vie mondaine, le snobisme, « certainement le péché auquel pense saint Paul quand il parle du péché pour lequel il n'y a pas de rémission. »
