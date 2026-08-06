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
      "Madame Swann"
    ]
  },
  "baron de Charlus": {
    "aliases": [
      "Charlus",
      "M. de Charlus",
      "le baron de Charlus",
      "le baron"
    ]
  },
  "Mme de Villeparisis": {
    "aliases": [
      "Mme de Villeparisis",
      "Madame de Villeparisis"
    ]
  },
  "Robert de Saint-Loup": {
    "aliases": [
      "Saint-Loup",
      "Robert de Saint-Loup"
    ]
  },
  "la grand-mère": {
    "aliases": [
      "ma grand'mère",
      "ma grand-mere",
      "la grand-mère du narrateur"
    ]
  },
  "le narrateur": {
    "aliases": [
      "je",
      "moi"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "Charlus",
        "mon oncle Charlus"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Robert de Saint-Loup",
      "target": "baron de Charlus",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.88,
      "evidence": "« Dame, ils n'ont pas cet air de race, de grand seigneur jusqu'au bout des ongles, qu'a mon oncle Charlus »",
      "explanation": "Saint-Loup explicitly praises Charlus’s aristocratic distinction, singling him out as possessing a superior 'air de race' compared to other Guermantes."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Being recognized by Saint-Loup as uniquely embodying noble distinction raises Charlus’s local social prestige."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-222"
}

### Candidate characters

[
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Puisqu'il refusait toute explication, j'essayai de m'en donner une, et le narrateur n'arrivai qu'à hésiter entre plusieurs dont aucune ne pouvait être la bonne. Peut-être ne se rappelait-il pas ou peut-être c'était le narrateur qui avais mal compris ce qu'il m'avait dit le matin... Plus probablement par orgueil ne voulait-il pas paraître avoir cherché à attirer des gens qu'il dédaignait, et préférait-il rejeter sur eux l'initiative de leur venue. Mais alors, s'il nous dédaignait, pourquoi avait-il tenu à ce que nous vinssions, ou plutôt à ce que la grand-mère du narrateur vînt, car de nous deux ce fut à elle seule qu'il adressa la parole pendant cette soirée et pas une seule fois à le narrateur. Causant avec la plus grande animation avec elle ainsi qu'avec Mme de Villeparisis, caché en quelque sorte derrière elles, comme il eût été au fond d'une loge, il se contentait seulement, détournant par moments le regard investigateur de ses yeux pénétrants, de l'attacher sur ma figure, avec le même sérieux, le même air de préoccupation, que si elle eût été un manuscrit difficile à déchiffrer.

### Passage

Sans doute s'il n'avait pas eu ces yeux, le visage de Charlus était semblable à celui de beaucoup de beaux hommes. Et quand Saint-Loup en me parlant d'autres Guermantes me dit plus tard : « Dame, ils n'ont pas cet air de race, de grand seigneur jusqu'au bout des ongles, qu'a mon oncle Charlus », en confirmant que l'air de race et la distinction aristocratiques n'étaient rien de mystérieux et de nouveau, mais qui consistaient en des éléments que j'avais reconnus sans difficulté et sans éprouver d'impression particulière, je devais sentir se dissiper une de mes illusions. Mais ce visage, auquel une légère couche de poudre donnait un peu l'aspect d'un visage de théâtre, Charlus avait beau en fermer hermétiquement l'expression, les yeux étaient comme une lézarde, comme une meurtrière que seule il n'avait pu boucher et par laquelle, selon le point où on était placé par rapport à lui, on se sentait brusquement croisé du reflet de quelque engin intérieur qui semblait n'avoir rien de rassurant, même pour celui qui, sans en être absolument maître, le porterait en soi, à l'état d'équilibre instable et toujours sur le point d'éclater ; et l'expression circonspecte et incessamment inquiète de ces yeux, avec toute la fatigue qui, autour d'eux, jusqu'à un cerne descendu très bas, en résultait pour le visage, si bien composé et arrangé qu'il fût, faisait penser à quelque incognito, à quelque déguisement d'un homme puissant en danger, ou seulement d'un individu dangereux, mais tragique. J'aurais voulu deviner quel était ce secret que ne portaient pas en eux les autres hommes et qui m'avait déjà rendu si énigmatique le regard de Charlus quand je l'avais vu le matin près du casino. Mais avec ce que je savais maintenant de sa parenté, je ne pouvais plus croire ni que ce fût celui d'un voleur, ni, d'après ce que j'entendais de sa conversation, que ce fût celui d'un fou. S'il était froid avec moi, alors qu'il était tellement aimable avec ma grand'mère, cela ne tenait peut-être pas à une antipathie personnelle, car d'une manière générale, autant il était bienveillant pour les femmes, des défauts de qui il parlait sans se départir, habituellement, d'une grande indulgence, autant il avait à l'égard des hommes, et particulièrement des jeunes gens, une haine d'une violence qui rappelait celle de certains misogynes pour les femmes. De deux ou trois « gigolos » qui étaient de la famille ou de l'intimité de Saint-Loup et dont celui-ci cita par hasard le nom, Charlus dit avec une expression presque féroce qui tranchait sur sa froideur habituelle : « Ce sont de petites canailles. » Je compris que ce qu'il reprochait surtout aux jeunes gens d'aujourd'hui, c'était d'être trop efféminés. « Ce sont de vraies femmes », disait-il avec mépris. Mais quelle vie n'eût pas semblé efféminée auprès de celle qu'il voulait que menât un homme et qu'il ne trouvait jamais assez énergique et virile ? (lui-même dans ses voyages à pied, après des heures de course, se jetait brûlant dans des rivières glacées.) Il n'admettait même pas qu'un homme portât une seule bague.
