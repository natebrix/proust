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
      "Odette",
      "Odette de Crécy",
      "Mme de Crécy",
      "Mme de Crecy"
    ]
  },
  "Mme Verdurin": {
    "aliases": [
      "Mme Verdurin",
      "Madame Verdurin",
      "la Patronne"
    ]
  },
  "M. Verdurin": {
    "aliases": [
      "M. Verdurin",
      "Monsieur Verdurin"
    ]
  },
  "docteur Cottard": {
    "aliases": [
      "le docteur",
      "Cottard",
      "le docteur Cottard"
    ]
  },
  "le pianiste": {
    "aliases": [
      "le jeune artiste",
      "le jeune pianiste",
      "le pianiste"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "docteur Cottard",
      "surface_forms": [
        "docteur Cottard",
        "docteur",
        "docteur docteur Cottard"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "docteur Cottard",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« C'est un savant qui vit en dehors de l'existence pratique... il ne connaît pas... la valeur des choses »; ensuite, M. Verdurin « acheta pour trois cents francs une pierre reconstituée en laissant entendre qu'on pouvait difficilement en voir d'aussi belle ».",
      "explanation": "Mme Verdurin's judgment that Cottard lacks practical discernment is validated by the narrated episode where the Verdurins successfully downgrade his New Year's gift without his noticing. His anxious, clueless outburst at the name “Swann” further underlines his dependence on others' cues."
    }
  ],
  "status_effects": [
    {
      "character": "docteur Cottard",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He is locally portrayed as gullible and socially uncertain, easy to mislead about value and needing identification by association."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-45-p-46"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Swann",
  "le peintre"
]

### Prior local context (optional)

Comme le sens critique qu'il croyait exercer sur tout lui faisait complètement défaut, le raffinement de politesse qui consiste à affirmer à quelqu'un qu'on oblige, sans souhaiter d'en être cru, que c'est à lui qu'on a obligation, était peine perdue avec lui, il prenait tout au pied de la lettre. Quel que fût l'aveuglement de Mme Verdurin à son égard, elle avait fini, tout en continuant à le trouver très fin, par être agacée de voir que quand elle l'invitait dans une avant-scène à entendre Sarah Bernhardt, lui disant, pour plus de grâce : « Vous êtes trop aimable d'être venu, docteur, d'autant plus que je suis sûre que vous avez déjà souvent entendu Sarah Bernhardt, et puis nous sommes peut-être trop près de la scène », docteur Cottard qui était entré dans la loge avec un sourire qui attendait pour se préciser ou pour disparaître que quelqu'un d'autorisé le renseignât sur la valeur du spectacle, lui répondait : « En effet on est beaucoup trop près et on commence à être fatigué de Sarah Bernhardt. Mais vous m'avez exprimé le désir que je vienne. Pour moi vos désirs sont des ordres. Je suis trop heureux de vous rendre ce petit service. Que ne ferait-on pas pour vous être agréable, vous êtes si bonne ! » Et il ajoutait : « Sarah Bernhardt, c'est bien la Voix d'Or, n'est-ce pas ? On écrit souvent aussi qu'elle brûle les planches. C'est une expression bizarre, n'est-ce pas ? » dans l'espoir de commentaires qui ne venaient point.

### Passage

« Tu sais, avait dit Mme Verdurin à son mari, je crois que nous faisons fausse route quand par modestie nous déprécions ce que nous offrons au docteur. C'est un savant qui vit en dehors de l'existence pratique, il ne connaît pas par lui-même la valeur des choses et il s'en rapporte à ce que nous lui en disons. » – « Je n'avais pas osé te le dire, mais je l'avais remarqué », répondit M. Verdurin. Et au jour de l'an suivant, au lieu d'envoyer au docteur Cottard un rubis de trois mille francs en lui disant que c'était bien peu de chose, M. Verdurin acheta pour trois cents francs une pierre reconstituée en laissant entendre qu'on pouvait difficilement en voir d'aussi belle.

Quand Mme Verdurin avait annoncé qu'on aurait, dans la soirée, Swann : « Swann ? » s'était écrié le docteur d'un accent rendu brutal par la surprise, car la moindre nouvelle prenait toujours plus au dépourvu que quiconque cet homme qui se croyait perpétuellement préparé à tout. Et voyant qu'on ne lui répondait pas : « Swann ? Qui ça, Swann ! » hurla-t-il au comble d'une anxiété qui se détendit soudain quand Mme Verdurin eut dit : « Mais l'ami dont Odette nous avait parlé. » – « Ah ! bon, bon, ça va bien », répondit le docteur apaisé. Quant au peintre il se réjouissait de l'introduction de Swann chez Mme Verdurin, parce qu'il le supposait amoureux d'Odette et qu'il aimait à favoriser les liaisons. « Rien ne m'amuse comme de faire des mariages, confia-t-il, dans l'oreille, au docteur Cottard, j'en ai déjà réussi beaucoup, même entre femmes ! »
