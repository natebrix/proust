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
      "presence_confidence": 0.98
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
      "evidence": "« verve d'insincérité », « intonations menteuses », « ton factice »; puis, brusque revirement: « Je crois que j'ai trouvé le moyen de me faire inviter… »; enfin: « Swann ne fut pas invité. »",
      "explanation": "The narrator unmasks Swann’s invectives against the Verdurins as an insincere screen for his jealousy and his desire to be reinvited to be with Odette; he tries to obtain an invitation and fails."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "His outburst appears fabricated and self-interested, which diminishes him in the narrator’s eyes."
    },
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "Locally, Swann is excluded from the dinner at Chatou despite his attempt to get himself invited."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-367-p-369"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "docteur Cottard",
  "le narrateur",
  "le pianiste"
]

### Prior local context (optional)

Il entendit les plaisanteries que ferait Mme Verdurin après dîner, les plaisanteries qui, quel que fût l'ennuyeux qu'elles eussent pour cible, l'avaient toujours amusé parce qu'il voyait Odette en rire, en rire avec lui, presque en lui. Maintenant il sentait que c'était peut-être de lui qu'on allait faire rire Odette. « Quelle gaieté fétide ! disait-il en donnant à sa bouche une expression de dégoût si forte qu'il avait lui-même la sensation musculaire de sa grimace jusque dans son cou révulsé contre le col de sa chemise. Et comment une créature dont le visage est fait à l'image de Dieu peut-elle trouver matière à rire dans ces plaisanteries nauséabondes ? Toute narine un peu délicate se détournerait avec horreur pour ne pas se laisser offusquer par de tels relents. C'est vraiment incroyable de penser qu'un être humain peut ne pas comprendre qu'en se permettant un sourire à l'égard d'un semblable qui lui a tendu loyalement la main, il se dégrade jusqu'à une fange d'où il ne sera plus possible à la meilleure volonté du monde de jamais le relever. J'habite à trop de milliers de mètres d'altitude au-dessus des bas-fonds où clapotent et clabaudent de tels sales papotages, pour que je puisse être éclaboussé par les plaisanteries d'une M. Verdurin, s'écria-t-il, en relevant la tête, en redressant fièrement son corps en arrière. Dieu m'est témoin que j'ai sincèrement voulu tirer Odette de là, et l'élever dans une atmosphère plus noble et plus pure. Mais la patience humaine a des bornes, et la mienne est à bout », se dit-il, comme si cette mission d'arracher Odette à une atmosphère de sarcasmes datait de plus longtemps que de quelques minutes, et comme s'il ne se l'était pas donnée seulement depuis qu'il pensait que ces sarcasmes l'avaient peut-être lui-même pour objet et tentaient de détacher Odette de lui.

### Passage

Il voyait le pianiste prêt à jouer la sonate Clair de lune et les mines de Mme Verdurin s'effrayant du mal que la musique de Beethoven allait faire à ses nerfs : « Idiote, menteuse ! s'écria-t-il, et ça croit aimer l'Art ! ». Elle dira à Odette, après lui avoir insinué adroitement quelques mots louangeurs pour Forcheville, comme elle avait fait si souvent pour lui : « Vous allez faire une petite place à côté de vous à M. de Forcheville. » « Dans l'obscurité ! maquerelle, entremetteuse ! » « Entremetteuse », c'était le nom qu'il donnait aussi à la musique qui les convierait à se taire, à rêver ensemble, à se regarder, à se prendre la main. Il trouvait du bon à la sévérité contre les arts, de Platon, de Bossuet, et de la vieille éducation française.

En somme la vie qu'on menait chez les Verdurin et qu'il avait appelée si souvent « la vraie vie » lui semblait la pire de toutes, et leur petit noyau le dernier des milieux. « C'est vraiment, disait-il, ce qu'il y a de plus bas dans l'échelle sociale, le dernier cercle de Dante. Nul doute que le texte auguste ne se réfère aux Verdurin ! Au fond, comme les gens du monde dont on peut médire, mais qui tout de même sont autre chose que ces bandes de voyous, montrent leur profonde sagesse en refusant de les connaître, d'y salir même le bout de leurs doigts ! Quelle divination dans ce « Noli me tangere » du faubourg Saint-Germain. » Il avait quitté depuis bien longtemps les allées du Bois, il était presque arrivé chez lui, que, pas encore dégrisé de sa douleur et de la verve d'insincérité dont les intonations menteuses, la sonorité artificielle de sa propre voix lui versaient d'instant en instant plus abondamment l'ivresse, il continuait encore à pérorer tout haut dans le silence de la nuit : « Les gens du monde ont leurs défauts que personne ne reconnaît mieux que moi, mais enfin ce sont tout de même des gens avec qui certaines choses sont impossibles. Telle femme élégante que j'ai connue était loin d'être parfaite, mais enfin il y avait tout de même chez elle un fond de délicatesse, une loyauté dans les procédés qui l'auraient rendue, quoi qu'il arrivât, incapable d'une félonie et qui suffisent à mettre des abîmes entre elle et une mégère comme la Verdurin. Verdurin ! quel nom ! Ah ! on peut dire qu'ils sont complets, qu'ils sont beaux dans leur genre ! Dieu merci, il n'était que temps de ne plus condescendre à la promiscuité avec cette infamie, avec ces ordures. »

Mais, comme les vertus qu'il attribuait tantôt encore aux Verdurin, n'auraient pas suffi, même s'ils les avaient vraiment possédées, mais s'ils n'avaient pas favorisé et protégé son amour, à provoquer chez Swann cette ivresse où il s'attendrissait sur leur magnanimité et qui, même propagée à travers d'autres personnes, ne pouvait lui venir que d'Odette – de même, l'immoralité, eût-elle été réelle, qu'il trouvait aujourd'hui aux Verdurin aurait été impuissante, s'ils n'avaient pas invité Odette avec Forcheville et sans lui, à déchaîner son indignation et à lui faire flétrir « leur infamie ». Et sans doute la voix de Swann était plus clairvoyante que lui-même, quand elle se refusait à prononcer ces mots pleins de dégoût pour le milieu Verdurin et de la joie d'en avoir fini avec lui, autrement que sur un ton factice et comme s'ils étaient choisis plutôt pour assouvir sa colère que pour exprimer sa pensée. Celle-ci, en effet, pendant qu'il se livrait à ces invectives, était probablement, sans qu'il s'en aperçût, occupée d'un objet tout à fait différent, car une fois arrivé chez lui, à peine eut-il refermé la porte cochère, que brusquement il se frappa le front, et, la faisant rouvrir, ressortit en s'écriant d'une voix naturelle cette fois : « Je crois que j'ai trouvé le moyen de me faire inviter demain au dîner de Chatou ! » Mais le moyen devait être mauvais, car Swann ne fut pas invité : le docteur Cottard qui, appelé en province pour un cas grave, n'avait pas vu les Verdurin depuis plusieurs jours et n'avait pu aller à Chatou, dit, le lendemain de ce dîner, en se mettant à table chez eux :
