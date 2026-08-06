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
      "evidence": "« on le voyait sauvage, fuyant la société des hommes »; « sa jalousie … altérait le caractère de Swann et changeait du tout au tout, aux yeux des autres, l’aspect même des signes extérieurs »",
      "explanation": "The narrator depicts Swann's jealousy transforming him into a misanthrope and altering how others perceive him, marking a clear local diminishment."
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
      "confidence": 0.88,
      "explanation": "Others’ view of him worsens as jealousy distorts his outward conduct."
    },
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "He withdraws from society and is seen as 'sauvage,' indicating sharp local exclusion."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-342"
}

### Candidate characters

[
  "Odette",
  "comte de Forcheville"
]

### Prior local context (optional)

Alors il lut toute la lettre ; à la fin elle s'excusait d'avoir agi aussi sans façon avec lui et lui disait qu'il avait oublié ses cigarettes chez elle, la même phrase qu'elle avait écrite à Swann une des premières fois qu'il était venu. Mais pour Swann elle avait ajouté : « puissiez-vous y avoir laissé votre coeur, je ne vous aurais pas laissé le reprendre ». Pour comte de Forcheville rien de tel : aucune allusion qui pût faire supposer une intrigue entre eux. À vrai dire d'ailleurs, comte de Forcheville était en tout ceci plus trompé que lui, puisque Odette lui écrivait pour lui faire croire que le visiteur était son oncle. En somme, c'était lui, Swann, l'homme à qui elle attachait de l'importance et pour qui elle avait congédié l'autre. Et pourtant, s'il n'y avait rien entre Odette et comte de Forcheville, pourquoi n'avoir pas ouvert tout de suite, pourquoi avoir dit : « J'ai bien fait d'ouvrir, c'était mon oncle » ; si elle ne faisait rien de mal à ce moment-là, comment comte de Forcheville pourrait-il même s'expliquer qu'elle eût pu ne pas ouvrir ? Swann restait là, désolé, confus et pourtant heureux, devant cette enveloppe qu'Odette lui avait remise sans crainte, tant était absolue la confiance qu'elle avait en sa délicatesse, mais à travers le vitrage transparent de laquelle se dévoilait à lui, avec le secret d'un incident qu'il n'aurait jamais cru possible de connaître, un peu de la vie d'Odette, comme dans une étroite section lumineuse pratiquée à même l'inconnu. Puis sa jalousie s'en réjouissait, comme si cette jalousie eût eu une vitalité indépendante, égoïste, vorace de tout ce qui la nourrirait, fût-ce aux dépens de lui-même. Maintenant elle avait un aliment et Swann allait pouvoir commencer à s'inquiéter chaque jour des visites qu'Odette avait reçues vers cinq heures, à chercher à apprendre où se trouvait comte de Forcheville à cette heure-là. Car la tendresse de Swann continuait à garder le même caractère que lui avait imprimé dès le début à la fois l'ignorance où il était de l'emploi des journées d'Odette et la paresse cérébrale qui l'empêchait de suppléer à l'ignorance par l'imagination. Il ne fut pas jaloux d'abord de toute la vie d'Odette, mais des seuls moments où une circonstance, peut-être mal interprétée, l'avait amené à supposer qu'Odette avait pu le tromper. Sa jalousie, comme une pieuvre qui jette une première, puis une seconde, puis une troisième amarre, s'attacha solidement à ce moment de cinq heures du soir, puis à un autre, puis à un autre encore. Mais Swann ne savait pas inventer ses souffrances. Elles n'étaient que le souvenir, la perpétuation d'une souffrance qui lui était venue du dehors.

### Passage

Mais là tout lui en apportait. Il voulut éloigner Odette de Forcheville, l'emmener quelques jours dans le Midi. Mais il croyait qu'elle était désirée par tous les hommes qui se trouvaient dans l'hôtel et qu'elle-même les désirait. Aussi lui qui jadis en voyage recherchait les gens nouveaux, les assemblées nombreuses, on le voyait sauvage, fuyant la société des hommes comme si elle l'eût cruellement blessé. Et comment n'aurait-il pas été misanthrope, quand dans tout homme il voyait un amant possible pour Odette ? Et ainsi sa jalousie, plus encore que n'avait fait le goût voluptueux et riant qu'il avait d'abord pour Odette, altérait le caractère de Swann et changeait du tout au tout, aux yeux des autres, l'aspect même des signes extérieurs par lesquels ce caractère se manifestait.
