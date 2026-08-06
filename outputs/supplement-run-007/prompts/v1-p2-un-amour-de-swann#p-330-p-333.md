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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "comte de Forcheville",
      "surface_forms": [
        "comte de Forcheville"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Saniette",
      "surface_forms": [
        "Saniette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "comte de Forcheville",
      "target": "Saniette",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.93,
      "evidence": "Forcheville « répondit … avec une telle grossièreté, se mettant à l’insulter… que le malheureux… s’était retiré… les larmes aux yeux »; Saniette « après avoir demandé à Mme Verdurin s’il devait rester, et n’ayant pas reçu de réponse, s’était retiré »; Odette « lui avait jeté un regard de complicité dans le mal ».",
      "explanation": "Forcheville publicly humiliates and effectively expels Saniette; Mme Verdurin’s silence and Odette’s complicit, congratulatory glance align the salon against Saniette. The narrator’s diction marks the act as base and cruel."
    }
  ],
  "status_effects": [
    {
      "character": "Saniette",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.95,
      "explanation": "He is humiliated and leaves in tears after being insulted, and receives no support from the hostess, indicating clear exclusion."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-330-p-333"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Swann"
]

### Prior local context (optional)

Il regarda. Devant lui, deux vieux messieurs étaient à la fenêtre, l'un tenant une lampe, et alors, il vit la chambre, une chambre inconnue. Ayant l'habitude, quand il venait chez Odette très tard, de reconnaître sa fenêtre à ce que c'était la seule éclairée entre les fenêtres toutes pareilles, il s'était trompé et avait frappé à la fenêtre suivante qui appartenait à la maison voisine. Il s'éloigna en s'excusant et rentra chez lui, heureux que la satisfaction de sa curiosité eût laissé leur amour intact et qu'après avoir simulé depuis si longtemps vis-à-vis d'Odette une sorte d'indifférence, il ne lui eût pas donné, par sa jalousie, cette preuve qu'il l'aimait trop, qui, entre deux amants, dispense, à tout jamais, d'aimer assez, celui qui la reçoit.

### Passage

Il ne lui parla pas de cette mésaventure, lui-même n'y songeait plus. Mais, par moments, un mouvement de sa pensée venait en rencontrer le souvenir qu'elle n'avait pas aperçu, le heurtait, l'enfonçait plus avant et Swann avait ressenti une douleur brusque et profonde. Comme si ç'avait été une douleur physique, les pensées de Swann ne pouvaient pas l'amoindrir ; mais du moins la douleur physique, parce qu'elle est indépendante de la pensée, la pensée peut s'arrêter sur elle, constater qu'elle a diminué, qu'elle a momentanément cessé. Mais cette douleur-là, la pensée, rien qu'en se la rappelant, la recréait. Vouloir n'y pas penser, c'était y penser encore, en souffrir encore. Et quand, causant avec des amis, il oubliait son mal, tout d'un coup un mot qu'on lui disait le faisait changer de visage, comme un blessé dont un maladroit vient de toucher sans précaution le membre douloureux. Quand il quittait Odette, il était heureux, il se sentait calme, il se rappelait les sourires qu'elle avait eus, railleurs en parlant de tel ou tel autre, et tendres pour lui, la lourdeur de sa tête qu'elle avait détachée de son axe pour l'incliner, la laisser tomber, presque malgré elle, sur ses lèvres, comme elle avait fait la première fois en voiture, les regards mourants qu'elle lui avait jetés pendant qu'elle était dans ses bras, tout en contractant frileusement contre l'épaule sa tête inclinée.

Mais aussitôt sa jalousie, comme si elle était l'ombre de son amour, se complétait du double de ce nouveau sourire qu'elle lui avait adressé le soir même – et qui, inverse maintenant, raillait Swann et se chargeait d'amour pour un autre – de cette inclinaison de sa tête mais renversée vers d'autres lèvres, et, données à un autre, toutes les marques de tendresse qu'elle avait eues pour lui. Et tous les souvenirs voluptueux qu'il emportait de chez elle étaient comme autant d'esquisses, de « projets » pareils à ceux que vous soumet un décorateur, et qui permettaient à Swann de se faire une idée des attitudes ardentes ou pâmées qu'elle pouvait avoir avec d'autres. De sorte qu'il en arrivait à regretter chaque plaisir qu'il goûtait près d'elle, chaque caresse inventée et dont il avait eu l'imprudence de lui signaler la douceur, chaque grâce qu'il lui découvrait, car il savait qu'un instant après, elles allaient enrichir d'instruments nouveaux son supplice.

Celui-ci était rendu plus cruel encore quand revenait à Swann le souvenir d'un bref regard qu'il avait surpris, il y avait quelques jours, et pour la première fois, dans les yeux d'Odette. C'était après dîner, chez les Verdurin. Soit que Forcheville sentant que Saniette, son beau-frère, n'était pas en faveur chez eux, eût voulu le prendre comme tête de Turc et briller devant eux à ses dépens, soit qu'il eût été irrité par un mot maladroit que celui-ci venait de lui dire, et qui, d'ailleurs, passa inaperçu pour les assistants qui ne savaient pas quelle allusion désobligeante il pouvait renfermer, bien contre le gré de celui qui le prononçait sans malice aucune, soit enfin qu'il cherchât depuis quelque temps une occasion de faire sortir de la maison quelqu'un qui le connaissait trop bien et qu'il savait trop délicat pour qu'il ne se sentît pas gêné à certains moments rien que de sa présence, Forcheville répondit à ce propos maladroit de Saniette avec une telle grossièreté, se mettant à l'insulter, s'enhardissant, au fur et à mesure qu'il vociférait, de l'effroi, de la douleur, des supplications de l'autre, que le malheureux, après avoir demandé à Mme Verdurin s'il devait rester, et n'ayant pas reçu de réponse, s'était retiré en balbutiant, les larmes aux yeux. Odette avait assisté impassible à cette scène, mais quand la porte se fut refermée sur Saniette, faisant descendre en quelque sorte de plusieurs crans l'expression habituelle de son visage, pour pouvoir se trouver dans la bassesse, de plain-pied avec Forcheville, elle avait brillanté ses prunelles d'un sourire sournois de félicitations pour l'audace qu'il avait eue, d'ironie pour celui qui en avait été victime ; elle lui avait jeté un regard de complicité dans le mal, qui voulait si bien dire : « voilà une exécution, ou je ne m'y connais pas. Avez-vous vu son air penaud, il en pleurait », que Forcheville, quand ses yeux rencontrèrent ce regard, dégrisé soudain de la colère ou de la simulation de colère dont il était encore chaud, sourit et répondit :

– Il n'avait qu'à être aimable, il serait encore ici, une bonne correction peut être utile à tout âge.
