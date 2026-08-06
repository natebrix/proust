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
      "canonical_name": "docteur Cottard",
      "surface_forms": [
        "docteur Cottard"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "docteur Cottard",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.78,
      "evidence": "Forcheville \"s'en amusa\" de la plaisanterie, et M. Verdurin \"ne marchanda pas sa gaieté\" en toussant d'hilarité; la même quinte recommence après le mot sur le \"duc d'Aumale\".",
      "explanation": "Laughter and coded signs of hilarity socially validate Cottard’s witticisms, including him as a purveyor of wit, even if the narrator underlines the recited and anxious side of his performance."
    }
  ],
  "status_effects": [
    {
      "character": "docteur Cottard",
      "dimension": "inclusion_exclusion",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.76,
      "explanation": "The laughing and repeated reception of his jokes integrates him favorably into the group’s sociability."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-268-p-273"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Cottard",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Saniette qui, depuis qu'il avait rendu précipitamment au maître d'hôtel son assiette encore pleine, s'était replongé dans un silence méditatif, en sortit enfin pour raconter en riant l'histoire d'un dîner qu'il avait fait avec le duc de La Trémoïlle et d'où il résultait que celui-ci ne savait pas que George Sand était le pseudonyme d'une femme. Swann, qui avait de la sympathie pour Saniette, crut devoir lui donner sur la culture du duc des détails montrant qu'une telle ignorance de la part de celui-ci était matériellement impossible ; mais tout d'un coup il s'arrêta, il venait de comprendre que Saniette n'avait pas besoin de ces preuves et savait que l'histoire était fausse pour la raison qu'il venait de l'inventer il y avait un moment. Cet excellent homme souffrait d'être trouvé si ennuyeux par les M. Verdurin ; et ayant conscience d'avoir été plus terne encore à ce dîner que d'habitude, il n'avait voulu le laisser finir sans avoir réussi à amuser. Il capitula si vite, eut l'air si malheureux de voir manqué l'effet sur lequel il avait compté et répondit d'un ton si lâche à Swann pour que celui-ci ne s'acharnât pas à une réfutation désormais inutile : « C'est bon, c'est bon ; en tous cas, même si je me trompe, ce n'est pas un crime, je pense » que Swann aurait voulu pouvoir dire que l'histoire était vraie et délicieuse. Le docteur qui les avait écoutés eut l'idée que c'était le cas de dire : « Se non è vero », mais il n'était pas assez sûr des mots et craignit de s'embrouiller.

### Passage

Après le dîner, Forcheville alla de lui-même vers le docteur.

– Elle n'a pas dû être mal, Mme Verdurin, et puis c'est une femme avec qui on peut causer, pour moi tout est là. Évidemment elle commence à avoir un peu de bouteille. Mais Mme de Crécy, voilà une petite femme qui a l'air intelligente, ah ! saperlipopette, on voit tout de suite qu'elle a l'oeil américain, celle-là ! Nous parlons de Mme de Crécy, dit-il à M. Verdurin qui s'approchait, la pipe à la bouche. Je me figure que comme corps de femme...

– J'aimerais mieux l'avoir dans mon lit que le tonnerre, dit précipitamment Cottard qui depuis quelques instants attendait en vain que Forcheville reprît haleine pour placer cette vieille plaisanterie dont il craignait que ne revînt pas l'à-propos si la conversation changeait de cours, et qu'il débita avec cet excès de spontanéité et d'assurance qui cherche à masquer la froideur et l'émoi inséparables d'une récitation. Forcheville la connaissait, il la comprit et s'en amusa. Quant à M. Verdurin, il ne marchanda pas sa gaieté, car il avait trouvé depuis peu pour la signifier un symbole autre que celui dont usait sa femme, mais aussi simple et aussi clair. À peine avait-il commencé à faire le mouvement de tête et d'épaules de quelqu'un qui s'esclaffle qu'aussitôt il se mettait à tousser comme si, en riant trop fort, il avait avalé la fumée de sa pipe. Et la gardant toujours au coin de sa bouche, il prolongeait indéfiniment le simulacre de suffocation et d'hilarité. Ainsi lui et Mme Verdurin, qui en face, écoutant le peintre qui lui racontait une histoire, fermait les yeux avant de précipiter son visage dans ses mains, avaient l'air de deux masques de théâtre qui figuraient différemment la gaieté.

M. Verdurin avait d'ailleurs fait sagement en ne retirant pas sa pipe de sa bouche, car Cottard qui avait besoin de s'éloigner un instant fit à mi-voix une plaisanterie qu'il avait apprise depuis peu et qu'il renouvelait chaque fois qu'il avait à aller au même endroit : « Il faut que j'aille entretenir un instant le duc d'Aumale », de sorte que la quinte de M. Verdurin recommença.

– Voyons, enlève donc ta pipe de ta bouche, tu vois bien que tu vas t'étouffer à te retenir de rire comme ça, lui dit Mme Verdurin qui venait offrir des liqueurs.

– Quel homme charmant que votre mari, il a de l'esprit comme quatre, déclara Forcheville à Mme Cottard. Merci madame. Un vieux troupier comme moi ça ne refuse jamais la goutte.
