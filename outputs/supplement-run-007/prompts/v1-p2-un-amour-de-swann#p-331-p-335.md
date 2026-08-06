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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "La jalousie inverse les sourires d’Odette et \"enrichir d'instruments nouveaux son supplice\"; scène de la visite manquée et du mensonge; Swann finit par \"prévoir dans son budget une disponibilité importante\" pour obtenir des renseignements.",
      "explanation": "The narrator presents Swann as emotionally overmastered by jealousy and dependence, reorganizing his life around suspicion and the need for information."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann’s local standing in affective terms declines as jealousy and Odette’s evasions dominate his thoughts and choices."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-331-p-335"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Saniette",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Il ne lui parla pas de cette mésaventure, lui-même n'y songeait plus. Mais, par moments, un mouvement de sa pensée venait en rencontrer le souvenir qu'elle n'avait pas aperçu, le heurtait, l'enfonçait plus avant et Swann avait ressenti une douleur brusque et profonde. Comme si ç'avait été une douleur physique, les pensées de Swann ne pouvaient pas l'amoindrir ; mais du moins la douleur physique, parce qu'elle est indépendante de la pensée, la pensée peut s'arrêter sur elle, constater qu'elle a diminué, qu'elle a momentanément cessé. Mais cette douleur-là, la pensée, rien qu'en se la rappelant, la recréait. Vouloir n'y pas penser, c'était y penser encore, en souffrir encore. Et quand, causant avec des amis, il oubliait son mal, tout d'un coup un mot qu'on lui disait le faisait changer de visage, comme un blessé dont un maladroit vient de toucher sans précaution le membre douloureux. Quand il quittait Odette, il était heureux, il se sentait calme, il se rappelait les sourires qu'elle avait eus, railleurs en parlant de tel ou tel autre, et tendres pour lui, la lourdeur de sa tête qu'elle avait détachée de son axe pour l'incliner, la laisser tomber, presque malgré elle, sur ses lèvres, comme elle avait fait la première fois en voiture, les regards mourants qu'elle lui avait jetés pendant qu'elle était dans ses bras, tout en contractant frileusement contre l'épaule sa tête inclinée.

### Passage

Mais aussitôt sa jalousie, comme si elle était l'ombre de son amour, se complétait du double de ce nouveau sourire qu'elle lui avait adressé le soir même – et qui, inverse maintenant, raillait Swann et se chargeait d'amour pour un autre – de cette inclinaison de sa tête mais renversée vers d'autres lèvres, et, données à un autre, toutes les marques de tendresse qu'elle avait eues pour lui. Et tous les souvenirs voluptueux qu'il emportait de chez elle étaient comme autant d'esquisses, de « projets » pareils à ceux que vous soumet un décorateur, et qui permettaient à Swann de se faire une idée des attitudes ardentes ou pâmées qu'elle pouvait avoir avec d'autres. De sorte qu'il en arrivait à regretter chaque plaisir qu'il goûtait près d'elle, chaque caresse inventée et dont il avait eu l'imprudence de lui signaler la douceur, chaque grâce qu'il lui découvrait, car il savait qu'un instant après, elles allaient enrichir d'instruments nouveaux son supplice.

Celui-ci était rendu plus cruel encore quand revenait à Swann le souvenir d'un bref regard qu'il avait surpris, il y avait quelques jours, et pour la première fois, dans les yeux d'Odette. C'était après dîner, chez les Verdurin. Soit que Forcheville sentant que Saniette, son beau-frère, n'était pas en faveur chez eux, eût voulu le prendre comme tête de Turc et briller devant eux à ses dépens, soit qu'il eût été irrité par un mot maladroit que celui-ci venait de lui dire, et qui, d'ailleurs, passa inaperçu pour les assistants qui ne savaient pas quelle allusion désobligeante il pouvait renfermer, bien contre le gré de celui qui le prononçait sans malice aucune, soit enfin qu'il cherchât depuis quelque temps une occasion de faire sortir de la maison quelqu'un qui le connaissait trop bien et qu'il savait trop délicat pour qu'il ne se sentît pas gêné à certains moments rien que de sa présence, Forcheville répondit à ce propos maladroit de Saniette avec une telle grossièreté, se mettant à l'insulter, s'enhardissant, au fur et à mesure qu'il vociférait, de l'effroi, de la douleur, des supplications de l'autre, que le malheureux, après avoir demandé à Mme Verdurin s'il devait rester, et n'ayant pas reçu de réponse, s'était retiré en balbutiant, les larmes aux yeux. Odette avait assisté impassible à cette scène, mais quand la porte se fut refermée sur Saniette, faisant descendre en quelque sorte de plusieurs crans l'expression habituelle de son visage, pour pouvoir se trouver dans la bassesse, de plain-pied avec Forcheville, elle avait brillanté ses prunelles d'un sourire sournois de félicitations pour l'audace qu'il avait eue, d'ironie pour celui qui en avait été victime ; elle lui avait jeté un regard de complicité dans le mal, qui voulait si bien dire : « voilà une exécution, ou je ne m'y connais pas. Avez-vous vu son air penaud, il en pleurait », que Forcheville, quand ses yeux rencontrèrent ce regard, dégrisé soudain de la colère ou de la simulation de colère dont il était encore chaud, sourit et répondit :

– Il n'avait qu'à être aimable, il serait encore ici, une bonne correction peut être utile à tout âge.

Un jour que Swann était sorti au milieu de l'après-midi pour faire une visite, n'ayant pas trouvé la personne qu'il voulait rencontrer, il eut l'idée d'entrer chez Odette à cette heure où il n'allait jamais chez elle, mais où il savait qu'elle était toujours à la maison à faire sa sieste ou à écrire des lettres avant l'heure du thé, et où il aurait plaisir à la voir un peu sans la déranger. Le concierge lui dit qu'il croyait qu'elle était là ; il sonna, crut entendre du bruit, entendre marcher, mais on n'ouvrit pas. Anxieux, irrité, il alla dans la petite rue où donnait l'autre face de l'hôtel, se mit devant la fenêtre de la chambre d'Odette ; les rideaux l'empêchaient de rien voir, il frappa avec force aux carreaux, appela ; personne n'ouvrit. Il vit que des voisins le regardaient. Il partit, pensant qu'après tout, il s'était peut-être trompé en croyant entendre des pas ; mais il en resta si préoccupé qu'il ne pouvait penser à autre chose. Une heure après, il revint. Il la trouva ; elle lui dit qu'elle était chez elle tantôt quand il avait sonné, mais dormait ; la sonnette l'avait éveillée, elle avait deviné que c'était Swann, elle avait couru après lui, mais il était déjà parti. Elle avait bien entendu frapper aux carreaux. Swann reconnut tout de suite dans ce dire un de ces fragments d'un fait exact que les menteurs pris de court se consolent de faire entrer dans la composition du fait faux qu'ils inventent, croyant y faire sa part et y dérober sa ressemblance à la Vérité. Certes quand Odette venait de faire quelque chose qu'elle ne voulait pas révéler, elle le cachait bien au fond d'elle-même. Mais dès qu'elle se trouvait en présence de celui à qui elle voulait mentir, un trouble la prenait, toutes ses idées s'effondraient, ses facultés d'invention et de raisonnement étaient paralysées, elle ne trouvait plus dans sa tête que le vide, il fallait pourtant dire quelque chose, et elle rencontrait à sa portée précisément la chose qu'elle avait voulu dissimuler et qui étant vraie, était seule restée là. Elle en détachait un petit morceau, sans importance par lui-même, se disant qu'après tout c'était mieux ainsi puisque c'était un détail véritable qui n'offrait pas les mêmes dangers qu'un détail faux. « Ça du moins, c'est vrai, se disait-elle, c'est toujours autant de gagné, il peut s'informer, il reconnaîtra que c'est vrai, ce n'est toujours pas ça qui me trahira. » Elle se trompait, c'était cela qui la trahissait, elle ne se rendait pas compte que ce détail vrai avait des angles qui ne pouvaient s'emboîter que dans les détails contigus du fait vrai dont elle l'avait arbitrairement détaché et qui, quels que fussent les détails inventés entre lesquels elle le placerait, révéleraient toujours par la matière excédante et les vides non remplis, que ce n'était pas d'entre ceux-là qu'il venait. « Elle avoue qu'elle m'avait entendu sonner, puis frapper, et qu'elle avait cru que c'était moi, qu'elle avait envie de me voir, se disait Swann. Mais cela ne s'arrange pas avec le fait qu'elle n'ait pas fait ouvrir. »

Mais il ne lui fit pas remarquer cette contradiction, car il pensait que, livrée à elle-même, Odette produirait peut-être quelque mensonge qui serait un faible indice de la vérité ; elle parlait ; il ne l'interrompait pas, il recueillait avec une piété avide et douloureuse ces mots qu'elle lui disait et qu'il sentait (justement, parce qu'elle la cachait derrière eux tout en lui parlant) garder vaguement, comme le voile sacré, l'empreinte, dessiner l'incertain modelé, de cette réalité infiniment précieuse et hélas introuvable : – ce qu'elle faisait tantôt à trois heures, quand il était venu – de laquelle il ne posséderait jamais que ces mensonges, illisibles et divins vestiges, et qui n'existait plus que dans le souvenir receleur de cet être qui la contemplait sans savoir l'apprécier, mais ne la lui livrerait pas. Certes il se doutait bien par moments qu'en elles-mêmes les actions quotidiennes d'Odette n'étaient pas passionnément intéressantes, et que les relations qu'elle pouvait avoir avec d'autres hommes n'exhalaient pas naturellement d'une façon universelle et pour tout être pensant une tristesse morbide, capable de donner la fièvre du suicide. Il se rendait compte alors que cet intérêt, cette tristesse n'existaient qu'en lui comme une maladie, et que quand celle-ci serait guérie, les actes d'Odette, les baisers qu'elle aurait pu donner redeviendraient inoffensifs comme ceux de tant d'autres femmes. Mais que la curiosité douloureuse que Swann y portait maintenant n'eût sa cause qu'en lui n'était pas pour lui faire trouver déraisonnable de considérer cette curiosité comme importante et de mettre tout en oeuvre pour lui donner satisfaction. C'est que Swann arrivait à un âge dont la philosophie – favorisée par celle de l'époque, par celle aussi du milieu où Swann avait beaucoup vécu, de cette coterie de la princesse des Laumes où il était convenu qu'on est intelligent dans la mesure où on doute de tout et où on ne trouvait de réel et d'incontestable que les goûts de chacun – n'est déjà plus celle de la jeunesse, mais une philosophie positive, presque médicale, d'hommes qui au lieu d'extérioriser les objets de leurs aspirations, essayent de dégager de leurs années déjà écoulées un résidu fixe d'habitudes, de passions qu'ils puissent considérer en eux comme caractéristiques et permanentes et auxquelles, délibérément, ils veilleront d'abord que le genre d'existence qu'ils adoptent puisse donner satisfaction. Swann trouvait sage de faire dans sa vie la part de la souffrance qu'il éprouvait à ignorer ce qu'avait fait Odette, aussi bien que la part de la recrudescence qu'un climat humide causait à son eczéma ; de prévoir dans son budget une disponibilité importante pour obtenir sur l'emploi des journées d'Odette des renseignements sans lesquels il se sentirait malheureux, aussi bien qu'il en réservait pour d'autres goûts dont il savait qu'il pouvait attendre du plaisir, au moins avant qu'il fût amoureux, comme celui des collections et de la bonne cuisine.
