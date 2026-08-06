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
        "le neveu"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "le narrateur",
      "surface_forms": [
        "je",
        "moi"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "baron de Charlus",
      "target": "le narrateur",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "« je ne pus pas attraper son regard »; « vis ses deux doigts tendus ... sans qu'il eût tourné les yeux ou interrompu la conversation »; plus tard, « Charlus ne me répondit pas davantage » et « le sourire de ceux qui de très haut jugent les caractères et les éducations ».",
      "explanation": "Charlus refuses to acknowledge the narrator beyond a minimal gesture and ignores his request for clarification, marking a haughty distancing."
    }
  ],
  "status_effects": [
    {
      "character": "le narrateur",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "Ignored by gaze and then by speech, he undergoes a clear and humiliating social exclusion."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-220"
}

### Candidate characters

[
  "Mme de Villeparisis",
  "la grand-mère"
]

### Prior local context (optional)

– Elle a un peu mal à la tête, la chaleur, cet orage. Il lui suffit d'un rien ; mais le narrateur crois que vous la verrez ce soir. Je lui ai conseillé de descendre. Cela ne peut lui faire que du bien.

### Passage

J'avais pensé qu'en nous invitant ainsi chez sa tante, que je ne doutais pas qu'il eût prévenue, Charlus eût voulu réparer l'impolitesse qu'il m'avait témoignée pendant la promenade du matin. Mais quand, arrivé dans le salon de Mme de Villeparisis, je voulus saluer le neveu de celle-ci, j'eus beau tourner autour de lui qui, d'une voix aiguë, racontait une histoire assez malveillante pour un de ses parents, je ne pus pas attraper son regard ; je me décidai à lui dire bonjour, et assez fort, pour l'avertir de ma présence, mais je compris qu'il l'avait remarquée, car avant même qu'aucun mot ne fût sorti de mes lèvres, au moment où je m'inclinais je vis ses deux doigts tendus pour que je les serrasse, sans qu'il eût tourné les yeux ou interrompu la conversation. Il m'avait évidemment vu, sans le laisser paraître, et je m'aperçus alors que ses yeux, qui n'étaient jamais fixés sur l'interlocuteur, se promenaient perpétuellement dans toutes les directions, comme ceux de certains animaux effrayés, ou ceux de ces marchands en plein air qui, tandis qu'ils débitent leur boniment et exhibent leur marchandise illicite, scrutent, sans pourtant tourner la tête, les différents points de l'horizon par où pourrait venir la police. Cependant j'étais un peu étonné de voir que Mme de Villeparisis, heureuse de nous voir venir, ne semblait pas s'y être attendue, je le fus plus encore d'entendre Charlus dire à ma grand'mère : « Ah ! c'est une très bonne idée que vous avez eue de venir, c'est charmant, n'est-ce pas, ma tante ? » Sans doute avait-il remarqué la surprise de celle-ci à notre entrée et pensait-il en homme habitué à donner le ton, le « la », qu'il lui suffisait pour changer cette surprise en joie d'indiquer qu'il en éprouvait lui-même, que c'était bien le sentiment que notre venue devait exciter. En quoi il calculait bien, car Mme de Villeparisis qui comptait fort son neveu et savait combien il était difficile de lui plaire, parut soudain avoir trouvé à ma grand'mère de nouvelles qualités et ne cessa de lui faire fête. Mais je ne pouvais comprendre que Charlus eût oublié en quelques heures l'invitation si brève, mais en apparence si intentionnelle, si préméditée qu'il m'avait adressée le matin même, et qu'il appelât « bonne idée » de ma grand'mère, une idée qui était toute de lui. Avec un scrupule de précision que je gardai jusqu'à l'âge où je compris que ce n'est pas en la lui demandant qu'on apprend la vérité sur l'intention qu'un homme a eue et que le risque d'un malentendu qui passera probablement inaperçu est moindre que celui d'une naïve insistance : « Mais monsieur, lui dis-je, vous vous rappelez bien, n'est-ce pas, que c'est vous qui m'avez demandé que nous vinssions ce soir ? » Aucun son, aucun mouvement ne trahirent que Charlus eût entendu ma question. Ce que voyant je la répétai comme les diplomates ou ces jeunes gens brouillés qui mettent une bonne volonté inlassable et vaine à obtenir des éclaircissements que l'adversaire est décidé à ne pas donner. Charlus ne me répondit pas davantage. Il me sembla voir flotter sur ses lèvres le sourire de ceux qui de très haut jugent les caractères et les éducations.
