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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "il"
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
      "confidence": 0.93,
      "evidence": "« Swann restait là, désolé, confus et pourtant heureux »; « sa jalousie… vorace »; « allait pouvoir commencer à s'inquiéter… »; « comme une pieuvre… s'attacha… »; « ne savait pas inventer ses souffrances ».",
      "explanation": "Jealousy seizes Swann, placing him in an emotionally weak and dependent position; the narrator diminishes him by casting him as prey to his own jealousy."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« En somme, c'était lui, Swann, l'homme à qui elle attachait de l'importance et pour qui elle avait congédié l'autre. » La formule pour Forcheville est « très froide », alors que, pour Swann, elle écrivait autrefois des mots tendres.",
      "explanation": "Locally, the narrative places Swann above Forcheville: Odette favors him, dismisses the other, and treats Forcheville more coldly, even deceives him."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": 1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.86,
      "explanation": "Swann is locally favored by Odette (he is the one for whom the other is dismissed), which includes him as a privileged recipient."
    },
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "His jealousy, now fueled, puts him in a position of distress and anxious surveillance."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-339-p-341"
}

### Candidate characters

[
  "Odette",
  "comte de Forcheville"
]

### Prior local context (optional)

On entendit la porte d'entrée se refermer et le bruit d'une voiture, comme si repartait une personne – celle probablement que Swann ne devait pas rencontrer – à qui on avait dit qu'Odette était sortie. Alors en songeant que rien qu'en venant à une heure où il n'en avait pas l'habitude, il s'était trouvé déranger tant de choses qu'elle ne voulait pas qu'il sût, il éprouva un sentiment de découragement, presque de détresse. Mais comme il aimait Odette, comme il avait l'habitude de tourner vers elle toutes ses pensées, la pitié qu'il eût pu s'inspirer à lui-même, ce fut pour elle qu'il la ressentit, et il murmura : « Pauvre chérie ! » Quand il la quitta, elle prit plusieurs lettres qu'elle avait sur sa table et lui demanda s'il ne pourrait pas les mettre à la poste. Il les emporta et, une fois rentré, s'aperçut qu'il avait gardé les lettres sur lui. Il retourna jusqu'à la poste, les tira de sa poche et avant de les jeter dans la boîte regarda les adresses. Elles étaient toutes pour des fournisseurs, sauf une pour comte de Forcheville. Il la tenait dans sa main. Il se disait : « Si je voyais ce qu'il y a dedans, je saurais comment elle l'appelle, comment elle lui parle, s'il y a quelque chose entre eux. Peut-être même qu'en ne la regardant pas, je commets une indélicatesse à l'égard d'Odette, car c'est la seule manière de me délivrer d'un soupçon peut-être calomnieux pour elle, destiné en tous cas à la faire souffrir et que rien ne pourrait plus détruire, une fois la lettre partie. »

### Passage

Il rentra chez lui en quittant la poste, mais il avait gardé sur lui cette dernière lettre. Il alluma une bougie et en approcha l'enveloppe qu'il n'avait pas osé ouvrir. D'abord il ne put rien lire, mais l'enveloppe était mince, et en la faisant adhérer à la carte dure qui y était incluse, il put à travers sa transparence, lire les derniers mots. C'était une formule finale très froide. Si, au lieu que ce fût lui qui regardât une lettre adressée à Forcheville, c'eût été Forcheville qui eût lu une lettre adressée à Swann, il aurait pu voir des mots autrement tendres. Il maintint immobile la carte qui dansait dans l'enveloppe plus grande qu'elle, puis, la faisant glisser avec le pouce, en amena successivement les différentes lignes sous la partie de l'enveloppe qui n'était pas doublée, la seule à travers laquelle on pouvait lire.

Malgré cela il ne distinguait pas bien. D'ailleurs cela ne faisait rien, car il en avait assez vu pour se rendre compte qu'il s'agissait d'un petit événement sans importance et qui ne touchait nullement à des relations amoureuses ; c'était quelque chose qui se rapportait à un oncle d'Odette. Swann avait bien lu au commencement de la ligne : « J'ai eu raison », mais ne comprenait pas ce qu'Odette avait eu raison de faire, quand soudain, un mot qu'il n'avait pas pu déchiffrer d'abord, apparut et éclaira le sens de la phrase tout entière : « J'ai eu raison d'ouvrir, c'était mon oncle. » D'ouvrir ! alors Forcheville était là tantôt quand Swann avait sonné et elle l'avait fait partir, d'où le bruit qu'il avait entendu.

Alors il lut toute la lettre ; à la fin elle s'excusait d'avoir agi aussi sans façon avec lui et lui disait qu'il avait oublié ses cigarettes chez elle, la même phrase qu'elle avait écrite à Swann une des premières fois qu'il était venu. Mais pour Swann elle avait ajouté : « puissiez-vous y avoir laissé votre coeur, je ne vous aurais pas laissé le reprendre ». Pour Forcheville rien de tel : aucune allusion qui pût faire supposer une intrigue entre eux. À vrai dire d'ailleurs, Forcheville était en tout ceci plus trompé que lui, puisque Odette lui écrivait pour lui faire croire que le visiteur était son oncle. En somme, c'était lui, Swann, l'homme à qui elle attachait de l'importance et pour qui elle avait congédié l'autre. Et pourtant, s'il n'y avait rien entre Odette et Forcheville, pourquoi n'avoir pas ouvert tout de suite, pourquoi avoir dit : « J'ai bien fait d'ouvrir, c'était mon oncle » ; si elle ne faisait rien de mal à ce moment-là, comment Forcheville pourrait-il même s'expliquer qu'elle eût pu ne pas ouvrir ? Swann restait là, désolé, confus et pourtant heureux, devant cette enveloppe qu'Odette lui avait remise sans crainte, tant était absolue la confiance qu'elle avait en sa délicatesse, mais à travers le vitrage transparent de laquelle se dévoilait à lui, avec le secret d'un incident qu'il n'aurait jamais cru possible de connaître, un peu de la vie d'Odette, comme dans une étroite section lumineuse pratiquée à même l'inconnu. Puis sa jalousie s'en réjouissait, comme si cette jalousie eût eu une vitalité indépendante, égoïste, vorace de tout ce qui la nourrirait, fût-ce aux dépens de lui-même. Maintenant elle avait un aliment et Swann allait pouvoir commencer à s'inquiéter chaque jour des visites qu'Odette avait reçues vers cinq heures, à chercher à apprendre où se trouvait Forcheville à cette heure-là. Car la tendresse de Swann continuait à garder le même caractère que lui avait imprimé dès le début à la fois l'ignorance où il était de l'emploi des journées d'Odette et la paresse cérébrale qui l'empêchait de suppléer à l'ignorance par l'imagination. Il ne fut pas jaloux d'abord de toute la vie d'Odette, mais des seuls moments où une circonstance, peut-être mal interprétée, l'avait amené à supposer qu'Odette avait pu le tromper. Sa jalousie, comme une pieuvre qui jette une première, puis une seconde, puis une troisième amarre, s'attacha solidement à ce moment de cinq heures du soir, puis à un autre, puis à un autre encore. Mais Swann ne savait pas inventer ses souffrances. Elles n'étaient que le souvenir, la perpétuation d'une souffrance qui lui était venue du dehors.
