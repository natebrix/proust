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
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Françoise",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« elle se montrait tout de même insupportable » ; « tout un atavisme de rapacité et de vulgarité provinciales » ; « prenait un visage de reine… le couloir… retentissait alors de propos… injurieux » ; « chemins détournés et absurdes qui me retardaient beaucoup »",
      "explanation": "The narrator belittles Françoise by describing her as proud, quarrelsome, and socially vulgar, obstructing her outings and attributing selfish intentions to the young girls."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "She is locally depreciated by the narrator for her pride, reproaches, and roughness, which tarnishes her image."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-431-p-435"
}

### Candidate characters

[
  "Albertine",
  "Andrée",
  "Bloch",
  "Elstir",
  "Mme de Villeparisis",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Comme Andrée était extrêmement riche, Albertine pauvre et orpheline, Andrée avec une grande générosité la faisait profiter de son luxe. Quant à ses sentiments pour Gisèle ils n'étaient pas tout à fait ceux que j'avais crus. On eut en effet bientôt des nouvelles de l'étudiante et, quand Albertine montra la lettre qu'elle en avait reçue, lettre destinée par Gisèle à donner des nouvelles de son voyage et de son arrivée à la petite bande en s'excusant de sa paresse de ne pas écrire encore aux autres, je fus surpris d'entendre Andrée, que je croyais brouillée à mort avec elle, dire : « Je lui écrirai demain, parce que si j'attends sa lettre d'abord, je peux attendre longtemps, elle est si négligente. » Et se tournant vers moi elle ajouta : « Vous ne la trouveriez pas très remarquable évidemment, mais c'est une si brave fille et puis j'ai vraiment une grande affection pour elle. » Je conclus que les brouilles d'Andrée ne duraient pas longtemps.

### Passage

Sauf ces jours de pluie, comme nous devions aller en bicyclette sur la falaise ou dans la campagne, une heure d'avance je cherchais à me faire beau et gémissais si Françoise n'avait pas bien préparé mes affaires. Or, même à Paris, elle redressait fièrement et rageusement sa taille que l'âge commençait à courber, pour peu qu'on la trouvât en faute, elle humble, elle modeste et charmante quand son amour-propre était flatté. Comme il était le grand ressort de sa vie, la satisfaction et la bonne humeur de Françoise étaient en proportion directe de la difficulté des choses qu'on lui demandait. Celles qu'elle avait à faire à Balbec étaient si aisées qu'elle montrait presque toujours un mécontentement qui était soudain centuplé et auquel s'alliait une ironique expression d'orgueil quand je me plaignais, au moment d'aller retrouver mes amies, que mon chapeau ne fût pas brossé, ou mes cravates en ordre. Elle qui pouvait se donner tant de peine sans trouver pour cela qu'elle eût rien fait, à la simple observation qu'un veston n'était pas à sa place, non seulement elle vantait avec quel soin elle l'avait « renfermé plutôt que non pas le laisser à la poussière », mais prononçant un éloge en règle de ses travaux, déplorait que ce ne fussent guère des vacances qu'elle prenait à Balbec, qu'on ne trouverait pas une seconde personne comme elle pour mener une telle vie. « Je ne comprends pas comment qu'on peut laisser ses affaires comme ça et allez-y voir si une autre saurait se retrouver dans ce pêle et mêle. Le diable lui-même y perdrait son latin. » Ou bien elle se contentait de prendre un visage de reine, me lançant des regards enflammés, et gardait un silence rompu aussitôt qu'elle avait fermé la porte et s'était engagée dans le couloir ; il retentissait alors de propos que je devinais injurieux, mais qui restaient aussi indistincts que ceux des personnages qui débitent leurs premières paroles derrière le portant avant d'être entrés en scène. D'ailleurs, quand je me préparais ainsi à sortir avec mes amies, même si rien ne manquait et si Françoise était de bonne humeur, elle se montrait tout de même insupportable. Car se servant de plaisanteries que dans mon besoin de parler de ces jeunes filles je lui avais faites sur elles, elle prenait un air de me révéler ce que j'aurais mieux su qu'elle si cela avait été exact, mais ce qui ne l'était pas car Françoise avait mal compris. Elle avait comme tout le monde son caractère propre ; une personne ne ressemble jamais à une voie droite, mais nous étonne de ses détours singuliers et inévitables dont les autres ne s'aperçoivent pas et par où il nous est pénible d'avoir à passer. Chaque fois que j'arrivais au point : « Chapeau pas en place », « nom d'Andrée ou d'Albertine », j'étais obligé par Françoise de m'égarer dans les chemins détournés et absurdes qui me retardaient beaucoup. Il en était de même quand je faisais préparer des sandwiches au chester et à la salade et acheter des tartes que je mangerais à l'heure du goûter, sur la falaise, avec ces jeunes filles, et qu'elles auraient bien pu payer à tour de rôle si elles n'avaient été aussi intéressées, déclarait Françoise, au secours de qui venait alors tout un atavisme de rapacité et de vulgarité provinciales, et pour laquelle on eût dit que l'âme divisée de la défunte Eulalie s'était incarnée, plus gracieusement qu'en Saint-Éloi, dans les corps charmants de mes amies de la petite bande. J'entendais ces accusations avec la rage de me sentir buter à un des endroits à partir desquels le chemin rustique et familier qu'était le caractère de Françoise devenait impraticable, pas pour longtemps heureusement. Puis le veston retrouvé et les sandwichs prêts, j'allais chercher Albertine, Andrée, Rosemonde, d'autres parfois, et, à pied ou en bicyclette, nous partions.

Autrefois j'eusse préféré que cette promenade eût lieu par le mauvais temps. Alors je cherchais à retrouver dans Balbec « le pays des Cimmériens », et de belles journées étaient une chose qui n'aurait pas dû exister là, une intrusion du vulgaire été des baigneurs dans cette antique région voilée par les brumes. Mais maintenant, tout ce que j'avais dédaigné, écarté de ma vue, non seulement les effets de soleil, mais même les régates, les courses de chevaux, je l'eusse recherché avec passion pour la même raison qu'autrefois je n'aurais voulu que des mers tempétueuses, et qui était qu'elles se rattachaient, les unes comme autrefois les autres, à une idée esthétique. C'est qu'avec mes amies nous étions quelquefois allés voir Elstir, et les jours où les jeunes filles étaient là, ce qu'il avait montré de préférence, c'était quelques croquis d'après de jolies yachtswomen ou bien une esquisse prise sur un hippodrome voisin de Balbec. J'avais d'abord timidement avoué à Elstir que je n'avais pas voulu aller aux réunions qui y avaient été données. « Vous avez eu tort, me dit-il, c'est si joli et si curieux aussi. D'abord cet être particulier, le jockey, sur lequel tant de regards sont fixés, et qui devant le paddock est là morne, grisâtre dans sa casaque éclatante, ne faisant qu'un avec le cheval caracolant qu'il ressaisit, comme ce serait intéressant de dégager ses mouvements professionnels, de montrer la tache brillante qu'il fait et que fait aussi la robe des chevaux, sur le champ de courses. Quelle transformation de toutes choses dans cette immensité lumineuse d'un champ de courses où on est surpris par tant d'ombres, de reflets, qu'on ne voit que là. Ce que les femmes peuvent y être jolies ! La première réunion surtout était ravissante, et il y avait des femmes d'une extrême élégance, dans une lumière humide, hollandaise, où l'on sentait monter dans le soleil même, le froid pénétrant de l'eau. Jamais je n'ai vu de femmes arrivant en voiture ou leurs jumelles aux yeux, dans une pareille lumière qui tient sans doute à l'humidité marine. Ah ! que j'aurais aimé la rendre ; je suis revenu de ces courses, fou, avec un tel désir de travailler ! » Puis il s'extasia plus encore sur les réunions du yachting que sur les courses de chevaux, et je compris que des régates, que des meetings sportifs où des femmes bien habillées baignent dans la glauque lumière d'un hippodrome marin, pouvaient être pour un artiste moderne motifs aussi intéressants que les fêtes qu'ils aimaient tant à décrire pour un Véronèse ou un Carpaccio. « Votre comparaison est d'autant plus exacte, me dit Elstir, qu'à cause de la ville où ils peignaient, ces fêtes étaient pour une part nautiques. Seulement, la beauté des embarcations de ce temps-là résidait le plus souvent dans leur lourdeur, dans leur complication. Il y avait des joutes sur l'eau, comme ici, données généralement en l'honneur de quelque ambassade pareille à celle que Carpaccio a représentée dans la Légende de Sainte Ursule. Les navires étaient massifs, construits comme des architectures, et semblaient presque amphibies comme de moindres Venises au milieu de l'autre, quand amarrés à l'aide de ponts volants, recouverts de satin cramoisi et de tapis persans ils portaient des femmes en brocart cerise ou en damas vert, tout près des balcons incrustés de marbres multicolores où d'autres femmes se penchaient pour regarder, dans leurs robes aux manches noires à crevés blancs serrés de perles ou ornés de guipures. On ne savait plus où finissait la terre, où commençait l'eau, qu'est-ce qui était encore le palais ou déjà le navire, la caravelle, la galéasse, le Bucentaure. » Albertine écoutait avec une attention passionnée ces détails de toilette, ces images de luxe que nous décrivait Elstir. « Oh ! je voudrais bien avoir les guipures dont vous me parlez, c'est si joli le point de Venise, s'écriait-elle ; d'ailleurs j'aimerais tant aller à Venise. »

– Vous pourrez peut-être bientôt, lui dit Elstir, contempler les étoffes merveilleuses qu'on portait là-bas. On ne les voyait plus que dans les tableaux des peintres vénitiens, ou alors très rarement dans les trésors des églises, parfois même il y en avait une qui passait dans une vente. Mais on dit qu'un artiste de Venise, Fortuny, a retrouvé le secret de leur fabrication et qu'avant quelques années les femmes pourront se promener, et surtout rester chez elles, dans des brocarts aussi magnifiques que ceux que Venise ornait, pour ses patriciennes, avec des dessins d'Orient. Mais je ne sais pas si j'aimerai beaucoup cela, si ce ne sera pas un peu trop costume anachronique, pour des femmes d'aujourd'hui, même paradant aux régates, car pour en revenir à nos bateaux modernes de plaisance, c'est tout le contraire que du temps de Venise, « Reine de l'Adriatique ». Le plus grand charme d'un yacht, de l'ameublement d'un yacht, des toilettes de yachting, est leur simplicité de choses de la mer, et j'aime tant la mer ! Je vous avoue que je préfère les modes d'aujourd'hui aux modes du temps de Véronèse et même de Carpaccio. Ce qu'il y a de joli dans nos yachts – et dans les yachts moyens surtout, je n'aime pas les énormes, trop navires, c'est comme pour les chapeaux, il y a une mesure à garder – c'est la chose unie, simple, claire, grise, qui par les temps voilés, bleuâtres, prend un flou crémeux. Il faut que la pièce où l'on se tient ait l'air d'un petit café. Les toilettes des femmes sur un yacht c'est la même chose ; ce qui est gracieux, ce sont ces toilettes légères, blanches et unies, en toile, en linon, en pékin, en coutil, qui au soleil et sur le bleu de la mer font un blanc aussi éclatant qu'une voile blanche. Il y a très peu de femmes du reste qui s'habillent bien, quelques-unes pourtant sont merveilleuses. Aux courses, Mlle Léa avait un petit chapeau blanc et une petite ombrelle blanche, c'était ravissant. Je ne sais pas ce que je donnerais pour avoir cette petite ombrelle. » J'aurais tant voulu savoir en quoi cette petite ombrelle différait des autres, et pour d'autres raisons, de coquetterie féminine, Albertine l'aurait voulu plus encore. Mais comme Françoise qui disait pour les soufflés : « C'est un tour de main », la différence était dans la coupe. « C'était, disait Elstir, tout petit, tout rond, comme un parasol chinois. » Je citai les ombrelles de certaines femmes, mais ce n'était pas cela du tout. Elstir trouvait toutes ces ombrelles affreuses. Homme d'un goût difficile et exquis, il faisait consister dans un rien, qui était tout, la différence entre ce que portaient les trois quarts des femmes et qui lui faisait horreur et une jolie chose qui le ravissait, et, au contraire de ce qui m'arrivait à moi pour qui tout luxe était stérilisant, exaltait son désir de peintre « pour tâcher de faire des choses aussi jolies ». « Tenez, voilà une petite qui a déjà compris comment étaient le chapeau et l'ombrelle, me dit Elstir en me montrant Albertine, dont les yeux brillaient de convoitise. – Comme j'aimerais être riche pour avoir un yacht, dit-elle au peintre. Je vous demanderais des conseils pour l'aménager. Quels beaux voyages je ferais ! Et comme ce serait joli d'aller aux régates de Cowes. Et une automobile ! Est-ce que vous trouvez que c'est joli, les modes des femmes pour les automobiles ? – Non, répondait Elstir, mais cela sera. D'ailleurs, il y a peu de couturiers, un ou deux, Callot, quoique donnant un peu trop dans la dentelle, Doucet, Cheruit, quelquefois Paquin. Le reste sont des horreurs. – Mais alors, il y a une différence immense entre une toilette de Callot et celle d'un couturier quelconque ? demandai-je à Albertine. – Mais énorme, mon petit bonhomme, me répondit-elle. Oh ! pardon. Seulement, hélas ! ce qui coûte trois cents francs ailleurs coûte deux mille francs chez eux. Mais cela ne se ressemble pas, cela a l'air pareil pour les gens qui n'y connaissent rien. – Parfaitement, répondit Elstir, sans aller pourtant jusqu'à dire que la différence soit aussi profonde qu'entre une statue de la cathédrale de Reims et de l'église Saint-Augustin... Tenez, à propos de cathédrales, dit-il en s'adressant spécialement à moi, parce que cela se référait à une causerie à laquelle ces jeunes filles n'avaient pas pris part et qui d'ailleurs ne les eût nullement intéressées, je vous parlais l'autre jour de l'église de Balbec comme d'une grande falaise, une grande levée des pierres du pays, mais inversement, me dit-il en me montrant une aquarelle, regardez ces falaises (c'est une esquisse prise tout près d'ici, aux Creuniers), regardez comme ces rochers puissamment et délicatement découpés font penser à une cathédrale. » En effet, on eût dit d'immenses arceaux roses. Mais peints par un jour torride, ils semblaient réduits en poussière, volatilisés par la chaleur, laquelle avait à demi bu la mer, presque passée, dans toute l'étendue de la toile, à l'état gazeux. Dans ce jour où la lumière avait comme détruit la réalité, celle-ci était concentrée dans des créatures sombres et transparentes qui par contraste donnaient une impression de vie plus saisissante, plus proche : les ombres. Altérées de fraîcheur, la plupart, désertant le large enflammé, s'étaient réfugiées au pied des rochers, à l'abri du soleil ; d'autres nageant lentement sur les eaux comme des dauphins s'attachaient aux flancs de barques en promenade dont elles élargissaient la coque, sur l'eau pâle, de leur corps verni et bleu. C'était peut-être la soif de fraîcheur communiquée par elles qui donnait le plus la sensation de la chaleur de ce jour et qui me fit m'écrier combien je regrettais de ne pas connaître les Creuniers. Albertine et Andrée assurèrent que j'avais dû y aller cent fois. En ce cas, c'était sans le savoir, ni me douter qu'un jour leur vue pourrait m'inspirer une telle soif de beauté, non pas précisément naturelle comme celle que j'avais cherchée jusqu'ici dans les falaises de Balbec, mais plutôt architecturale. Surtout moi qui, parti pour voir le royaume des tempêtes, ne trouvais jamais dans mes promenades avec Mme de Villeparisis où souvent nous ne l'apercevions que de loin, peint dans l'écartement des arbres, l'océan assez réel, assez liquide, assez vivant, donnant assez l'impression de lancer ses masses d'eau, et qui n'aurais aimé le voir immobile que sous un linceul hivernal de brume, je n'eusse guère pu croire que je rêverais maintenant d'une mer qui n'était plus qu'une vapeur blanchâtre ayant perdu la consistance et la couleur. Mais cette mer, Elstir, comme ceux qui rêvaient dans ces barques engourdies par la chaleur, en avait, jusqu'à une telle profondeur, goûté l'enchantement qu'il avait su rapporter, fixer sur sa toile, l'imperceptible reflux de l'eau, la pulsation d'une minute heureuse ; et on était soudain devenu si amoureux, en voyant ce portrait magique, qu'on ne pensait plus qu'à courir le monde pour retrouver la journée enfuie, dans sa grâce instantanée et dormante.

De sorte que si, avant ces visites chez Elstir, avant d'avoir vu une marine de lui où une jeune femme, en robe de barège ou de linon, dans un yacht arborant le drapeau américain, mit le « double » spirituel d'une robe de linon blanc et d'un drapeau dans mon imagination, qui aussitôt couva un désir insatiable de voir sur-le-champ des robes de linon blanc et des drapeaux près de la mer, comme si cela ne m'était jamais arrivé jusque-là, je m'étais toujours efforcé, devant la mer, d'expulser du champ de ma vision, aussi bien que les baigneurs du premier plan, les yachts aux voiles trop blanches comme un costume de plage, tout ce qui m'empêchait de me persuader que je contemplais le flot immémorial qui déroulait déjà sa même vie mystérieuse avant l'apparition de l'espèce humaine, et jusqu'aux jours radieux qui me semblaient revêtir de l'aspect banal de l'universel été de cette côte de brumes et de tempêtes, y marquer un simple temps d'arrêt, l'équivalent de ce qu'on appelle en musique une mesure pour rien ; maintenant c'était le mauvais temps qui me paraissait devenir quelque accident funeste, ne pouvant plus trouver de place dans le monde de la beauté ; je désirais vivement aller retrouver dans la réalité ce qui m'exaltait si fort et j'espérais que le temps serait assez favorable pour voir du haut de la falaise les mêmes ombres bleues que dans le tableau d'Elstir.

Le long de la route, je ne me faisais plus d'ailleurs un écran de mes mains comme dans ces jours où concevant la nature comme animée d'une vie antérieure à l'apparition de l'homme, et en opposition avec tous ces fastidieux perfectionnements de l'industrie qui m'avaient fait jusqu'ici bâiller d'ennui dans les expositions universelles ou chez les modistes, j'essayais de ne voir de la mer que la section où il n'y avait pas de bateau à vapeur, de façon à me la représenter comme immémoriale, encore contemporaine des âges où elle avait été séparée de la terre, à tout le moins contemporaine des premiers siècles de la Grèce, ce qui me permettait de me redire en toute vérité les vers du « père Leconte » chers à Bloch :
