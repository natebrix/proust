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
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.86,
      "evidence": "« Swann a ri d'un rire niais »; « je l'ai trouvé extrêmement bête »; « Il n'est pas franc, c'est un monsieur cauteleux… »; « ce n'est pas comme l'autre qui n'est jamais ni figue ni raisin »; « c'est le raté, le petit individu envieux »; éloge comparatif de Forcheville: « Voilà au moins un homme qui vous dit carrément sa façon de penser… il est toujours comte de Forcheville. »",
      "explanation": "In the end-of-evening review, the Verdurin couple and the voice of their circle disparage Swann (stupid, sly, a failure) while valuing Forcheville as frank and titled. The narrative subsequently signals an ironic distance."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Within the Verdurin clan, Swann is locally belittled and disadvantaged compared to Forcheville."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-296-p-304"
}

### Candidate characters

[
  "Brichot",
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "la mère du narrateur"
]

### Prior local context (optional)

– Qu'est-ce que c'est exactement que cette Mme Verdurin ? un demi-castor ? dit comte de Forcheville au peintre à qui il proposa de revenir avec lui.

### Passage

Odette le vit s'éloigner avec regret, elle n'osa pas ne pas revenir avec Swann, mais fut de mauvaise humeur en voiture, et quand il lui demanda s'il devait entrer chez elle, elle lui dit : « Bien entendu », en haussant les épaules avec impatience. Quand tous les invités furent partis, Mme Verdurin dit à son mari :

– As-tu remarqué comme Swann a ri d'un rire niais quand nous avons parlé de Mme La Trémoïlle ?

Elle avait remarqué que devant ce nom Swann et Forcheville avaient plusieurs fois supprimé la particule. Ne doutant pas que ce fût pour montrer qu'ils n'étaient pas intimidés par les titres, elle souhaitait d'imiter leur fierté, mais n'avait pas bien saisi par quelle forme grammaticale elle se traduisait. Aussi sa vicieuse façon de parler l'emportant sur son intransigeance républicaine, elle disait encore les de La Trémoïlle ou plutôt par une abréviation en usage dans les paroles des chansons de café-concert et les légendes des caricaturistes et qui dissimulait le de, les d'La Trémoïlle, mais elle se rattrapait en disant : « Madame La Trémoïlle. » « La Duchesse, comme dit Swann », ajouta-t-elle ironiquement avec un sourire qui prouvait qu'elle ne faisait que citer et ne prenait pas à son compte une dénomination aussi naïve et ridicule.

– Je te dirai que je l'ai trouvé extrêmement bête.

Et M. Verdurin lui répondit :

– Il n'est pas franc, c'est un monsieur cauteleux, toujours entre le zist et le zest. Il veut toujours ménager la chèvre et le chou. Quelle différence avec Forcheville ! Voilà au moins un homme qui vous dit carrément sa façon de penser. Ça vous plaît ou ça ne vous plaît pas. Ce n'est pas comme l'autre qui n'est jamais ni figue ni raisin. Du reste Odette a l'air de préférer joliment le Forcheville, et je lui donne raison. Et puis enfin, puisque Swann veut nous la faire à l'homme du monde, au champion des duchesses, au moins l'autre a son titre ; il est toujours comte de Forcheville, ajouta-t-il d'un air délicat, comme si, au courant de l'histoire de ce comté, il en soupesait minutieusement la valeur particulière.

– Je te dirai, dit Mme Verdurin, qu'il a cru devoir lancer contre Brichot quelques insinuations venimeuses et assez ridicules. Naturellement, comme il a vu que Brichot était aimé dans la maison, c'était une manière de nous atteindre, de bêcher notre dîner. On sent le bon petit camarade qui vous débinera en sortant.

– Mais je te l'ai dit, répondit M. Verdurin, c'est le raté, le petit individu envieux de tout ce qui est un peu grand.

En réalité il n'y avait pas un fidèle qui ne fût plus malveillant que Swann ; mais tous ils avaient la précaution d'assaisonner leurs médisances de plaisanteries connues, d'une petite pointe d'émotion et de cordialité ; tandis que la moindre réserve que se permettait Swann, dépouillée des formules de convention telles que : « Ce n'est pas du mal que nous disons » et auxquelles il dédaignait de s'abaisser, paraissait une perfidie. Il y a des auteurs originaux dont la moindre hardiesse révolte parce qu'ils n'ont pas d'abord flatté les goûts du public et ne lui ont pas servi les lieux communs auxquels il est habitué ; c'est de la même manière que Swann indignait M. Verdurin. Pour Swann comme pour eux, c'était la nouveauté de son langage qui faisait croire à la noirceur de ses intentions.
