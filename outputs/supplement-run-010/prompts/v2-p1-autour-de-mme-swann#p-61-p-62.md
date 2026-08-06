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
  "Odette": {
    "aliases": [
      "Mme Swann",
      "Madame Swann",
      "la belle Madame Swann"
    ]
  },
  "Gilberte": {
    "aliases": [
      "Gilberte",
      "la fille de Odette"
    ]
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "l'Ambassadeur",
      "le marquis de Norpois"
    ]
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  },
  "le narrateur": {
    "aliases": [
      "je",
      "moi",
      "mon fils"
    ]
  },
  "le père du narrateur": {
    "aliases": [
      "mon père",
      "Monsieur votre père"
    ]
  },
  "la mère du narrateur": {
    "aliases": [
      "ma mère",
      "Madame"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "M. de Norpois"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
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
      "source": "Norpois",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.84,
      "evidence": "Swann aurait « un manque de réserve et de goût, presque de tact », parle « en véritable parvenu », montre « d'empressement auprès d'une société… fort mêlée »; sa tante refuse de recevoir Odette et mène une campagne d'exclusion.",
      "explanation": "Norpois belittles Swann by stressing his supposed vulgarity, his mixed acquaintances, and the hostility of his own milieu; the narrator frames these remarks as delivered mischievously."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "He is presented as socially lowered and of dubious taste, seeking a « mêlé » milieu and undergoing an orchestrated exclusion."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-61-p-62"
}

### Candidate characters

[
  "Odette",
  "le directeur"
]

### Prior local context (optional)

Ma mère réprima un frémissement, car d'une sensibilité plus prompte que le père du narrateur, elle s'alarmait pour lui de ce qui ne devait le contrarier qu'un instant après. Les désagréments qui lui arrivaient étaient perçus d'abord par elle comme ces mauvaises nouvelles de France qui sont connues plus tôt à l'étranger que chez nous. Mais curieuse de savoir quel genre de personnes les Swann pouvaient recevoir, elle s'enquit auprès de Norpois de celles qu'il y avait rencontrées.

### Passage

– Mon Dieu... c'est une maison où il me semble que vont surtout... des messieurs. Il y avait quelques hommes mariés, mais leurs femmes étaient souffrantes ce soir-là et n'étaient pas venues, répondit l'Ambassadeur avec une finesse voilée de bonhomie et en jetant autour de lui des regards dont la douceur et la discrétion faisaient mine de tempérer et exagéraient habilement la malice.

– Je dois ajouter, pour être tout à fait juste, qu'il y va cependant des femmes, mais... appartenant plutôt..., comment dirais-je, au monde républicain qu'à la société de Swann (il prononçait Svann). Qui sait ? Ce sera peut-être un jour un salon politique ou littéraire. Du reste, il semble qu'ils soient contents comme cela. Je trouve que Swann le montre un peu trop. Il nommait les gens chez qui lui et sa femme étaient invités pour la semaine suivante et de l'intimité desquels il n'y a pourtant pas lieu de s'enorgueillir, avec un manque de réserve et de goût, presque de tact, qui m'a étonné chez un homme aussi fin. Il répétait : « Nous n'avons pas un soir de libre », comme si ç'avait été une gloire, et en véritable parvenu, qu'il n'est pas cependant. Car Swann avait beaucoup d'amis et même d'amies, et sans trop m'avancer, ni vouloir commettre d'indiscrétion, je crois pouvoir dire que non pas toutes, ni même le plus grand nombre, mais l'une au moins, et qui est une fort grande dame, ne se serait peut-être pas montrée entièrement réfractaire à l'idée d'entrer en relations avec Madame Swann, auquel cas, vraisemblablement, plus d'un mouton de Panurge aurait suivi. Mais il semble qu'il n'y ait eu de la part de Swann aucune démarche esquissée en ce sens. Comment ? encore un pudding à la Nasselrode ! Ce ne sera pas de trop de la cure de Carlsbad pour me remettre d'un pareil festin de Lucullus... Peut-être Swann a-t-il senti qu'il y aurait trop de résistances à vaincre. Le mariage, cela est certain, n'a pas plu. On a parlé de la fortune de la femme, ce qui est une grosse bourde. Mais, enfin, tout cela n'a pas paru agréable. Et puis Swann a une tante excessivement riche et admirablement posée, femme d'un homme qui, financièrement parlant, est une puissance. Et non seulement elle a refusé de recevoir Odette, mais elle a mené une campagne en règle pour que ses amies et connaissances en fissent autant. Je n'entends pas par là qu'aucun Parisien de bonne compagnie ait manqué de respect à Madame Swann... Non ! cent fois non ! le mari était d'ailleurs homme à relever le gant. En tous cas, il y a une chose curieuse, c'est de voir combien Swann, qui connaît tant de monde et du plus choisi, montre d'empressement auprès d'une société dont le moins qu'on puisse dire est qu'elle est fort mêlée. Moi qui l'ai connu jadis, j'avoue que j'éprouvais autant de surprise que d'amusement à voir un homme aussi bien élevé, aussi à la mode dans les coteries les plus triées, remercier avec effusion le directeur du Cabinet du ministre des Postes d'être venu chez eux et lui demander si Madame Swann pourrait se permettre d'aller voir sa femme. Il doit pourtant se trouver dépaysé ; évidemment ce n'est plus le même monde. Mais je ne crois pas cependant que Swann soit malheureux. Il y a eu, il est vrai, dans les années qui précédèrent le mariage, d'assez vilaines manoeuvres de chantage de la part de la femme ; elle privait Swann de sa fille chaque fois qu'il lui refusait quelque chose. Le pauvre Swann, aussi naïf qu'il est pourtant raffiné, croyait chaque fois que l'enlèvement de sa fille était une coïncidence et ne voulait pas voir la réalité. Elle lui faisait d'ailleurs des scènes si continuelles qu'on pensait que le jour où elle serait arrivée à ses fins et se serait fait épouser, rien ne la retiendrait plus et que leur vie serait un enfer. Hé bien ! c'est le contraire qui est arrivé. On plaisante beaucoup la manière dont Swann parle de sa femme, on en fait même des gorges chaudes. On ne demandait certes pas que, plus ou moins conscient d'être... (vous savez le mot de Molière), il allât le proclamer urbi et orbi ; n'empêche qu'on le trouve exagéré quand il dit que sa femme est une excellente épouse. Or, ce n'est pas aussi faux qu'on le croit. À sa manière qui n'est pas celle que tous les maris préféreraient, – mais enfin, entre nous, il me semble difficile que Swann, qui la connaissait depuis longtemps et est loin d'être un maître-sot, ne sût pas à quoi s'en tenir, – il est indéniable qu'elle semble avoir de l'affection pour lui. Je ne dis pas qu'elle ne soit pas volage, et Swann lui-même ne se fait pas faute de l'être, à en croire les bonnes langues qui, vous pouvez le penser, vont leur train. Mais elle lui est reconnaissante de ce qu'il a fait pour elle, et, contrairement aux craintes éprouvées par tout le monde, elle paraît devenue d'une douceur d'ange.
