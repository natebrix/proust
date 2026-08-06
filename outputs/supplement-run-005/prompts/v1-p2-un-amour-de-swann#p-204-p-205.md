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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "comte de Forcheville",
      "surface_forms": [
        "comte de Forcheville",
        "Forcheville"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "comte de Forcheville",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.86,
      "evidence": "Le premier dîner « mit en lumière toutes ces différences, fit ressortir ses qualités et précipita la disgrâce de Swann »; Forcheville est « abasourdi, émerveillé » par les tirades du peintre « sans d'ailleurs les comprendre », et se « délecte » des plaisanteries.",
      "explanation": "The narrator contrasts Forcheville's ready conformity to the Verdurins' tastes with Swann's scrupled reserve, leading to Forcheville's favorable reception. The elevation is social within the clan and narrated with irony about Forcheville's snobbery and limited understanding."
    }
  ],
  "status_effects": [
    {
      "character": "comte de Forcheville",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Within the Verdurin circle, he gains favor as his reactions align with what the hosts and habitués value; the dinner highlights his 'qualities.'"
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-204-p-205"
}

### Candidate characters

[
  "Brichot",
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Saniette",
  "Swann",
  "docteur Cottard",
  "le peintre"
]

### Prior local context (optional)

Ainsi il n'y avait sans doute pas, dans tout le milieu M. Verdurin, un seul fidèle qui les aimât ou crût les aimer autant que Swann. Et pourtant, quand M. Verdurin avait dit que Swann ne lui revenait pas, non seulement il avait exprimé sa propre pensée, mais il avait deviné celle de sa femme. Sans doute Swann avait pour Odette une affection trop particulière et dont il avait négligé de faire de Mme Verdurin la confidente quotidienne ; sans doute la discrétion même avec laquelle il usait de l'hospitalité des M. Verdurin, s'abstenant souvent de venir dîner pour une raison qu'ils ne soupçonnaient pas et à la place de laquelle ils voyaient le désir de ne pas manquer une invitation chez des « ennuyeux », sans doute aussi, et malgré toutes les précautions qu'il avait prises pour la leur cacher, la découverte progressive qu'ils faisaient de sa brillante situation mondaine, tout cela contribuait à leur irritation contre lui. Mais la raison profonde en était autre. C'est qu'ils avaient très vite senti en lui un espace réservé, impénétrable, où il continuait à professer silencieusement pour lui-même que la princesse de Sagan n'était pas grotesque et que les plaisanteries de Cottard n'étaient pas drôles, enfin et bien que jamais il ne se départît de son amabilité et ne se révoltât contre leurs dogmes, une impossibilité de les lui imposer, de l'y convertir entièrement, comme ils n'en avaient jamais rencontré une pareille chez personne. Ils lui auraient pardonné de fréquenter des ennuyeux (auxquels d'ailleurs, dans le fond de son coeur, il préférait mille fois les M. Verdurin et tout le petit noyau) s'il avait consenti, pour le bon exemple, à les renier en présence des fidèles. Mais c'est une abjuration qu'ils comprirent qu'on ne pourrait pas lui arracher.

### Passage

Quelle différence avec un « nouveau » qu'Odette leur avait demandé d'inviter, quoiqu'elle ne l'eût rencontré que peu de fois, et sur lequel ils fondaient beaucoup d'espoir, le comte de Forcheville ! (Il se trouva qu'il était justement le beau-frère de Saniette, ce qui remplit d'étonnement les fidèles : le vieil archiviste avait des manières si humbles qu'ils l'avaient toujours cru d'un rang social inférieur au leur et ne s'attendaient pas à apprendre qu'il appartenait à un monde riche et relativement aristocratique.) Sans doute Forcheville était grossièrement snob, alors que Swann ne l'était pas ; sans doute il était bien loin de placer, comme lui, le milieu des Verdurin au-dessus de tous les autres. Mais il n'avait pas cette délicatesse de nature qui empêchait Swann de s'associer aux critiques trop manifestement fausses que dirigeait Mme Verdurin contre des gens qu'il connaissait. Quant aux tirades prétentieuses et vulgaires que le peintre lançait à certains jours, aux plaisanteries de commis voyageur que risquait Cottard et auxquelles Swann, qui les aimait l'un et l'autre, trouvait facilement des excuses mais n'avait pas le courage et l'hypocrisie d'applaudir, Forcheville était au contraire d'un niveau intellectuel qui lui permettait d'être abasourdi, émerveillé par les unes, sans d'ailleurs les comprendre, et de se délecter aux autres. Et justement le premier dîner chez les Verdurin auquel assista Forcheville mit en lumière toutes ces différences, fit ressortir ses qualités et précipita la disgrâce de Swann.

Il y avait, à ce dîner, en dehors des habitués, un professeur de la Sorbonne, Brichot, qui avait rencontré M. et Mme Verdurin aux eaux et, si ses fonctions universitaires et ses travaux d'érudition n'avaient pas rendu très rares ses moments de liberté, serait volontiers venu souvent chez eux. Car il avait cette curiosité, cette superstition de la vie, qui unie à un certain scepticisme relatif à l'objet de leurs études, donne dans n'importe quelle profession, à certains hommes intelligents, médecins qui ne croient pas à la médecine, professeurs de lycée qui ne croient pas au thème latin, la réputation d'esprits larges, brillants, et même supérieurs. Il affectait, chez Mme Verdurin, de chercher ses comparaisons dans ce qu'il y avait de plus actuel quand il parlait de philosophie et d'histoire, d'abord parce qu'il croyait qu'elles ne sont qu'une préparation à la vie et qu'il s'imaginait trouver en action dans le petit clan ce qu'il n'avait connu jusqu'ici que dans les livres, puis peut-être aussi parce que, s'étant vu inculquer autrefois, et ayant gardé à son insu, le respect de certains sujets, il croyait dépouiller l'universitaire en prenant avec eux des hardiesses qui, au contraire, ne lui paraissaient telles, que parce qu'il l'était resté.
