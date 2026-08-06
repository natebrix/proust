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
  "le peintre": {
    "aliases": [
      "le peintre",
      "Biche"
    ]
  },
  "Remi": {
    "aliases": [
      "Rémi",
      "Remi",
      "le cocher"
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
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "« il n'était plus le même, et qu'il n'était plus seul, qu'un être nouveau était là avec lui... avec qui il allait être obligé d'user de ménagements comme avec un maître ou avec une maladie »; sa « déception et la torture » face à la vaine présence d’Odette; sa diversion absurde sur la « provision de bois » au moment décisif.",
      "explanation": "The narrator shows Swann overwhelmed by a new passionate state that dominates him like a master or an illness, altering his autonomy and making him clumsy and powerless in action."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "Swann is locally diminished, dominated by anxiety and emotional dependence, which governs his thoughts and gestures."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-159-p-162"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Remi",
  "Rémi",
  "le narrateur"
]

### Prior local context (optional)

– Mais cela ne l'empêche pas d'être charmante ; nous ne disons pas du mal d'elle, nous disons que ce n'est pas une vertu ni une intelligence. Au fond, dit-il au peintre, tenez-vous tant que ça à ce qu'elle soit vertueuse ? Elle serait peut-être beaucoup moins charmante, qui sait ?

### Passage

Sur le palier, Swann avait été rejoint par le maître d'hôtel qui ne se trouvait pas là au moment où il était arrivé et avait été chargé par Odette de lui dire – mais il y avait bien une heure déjà – au cas où il viendrait encore, qu'elle irait probablement prendre du chocolat chez Prévost avant de rentrer. Swann partit chez Prévost, mais à chaque pas sa voiture était arrêtée par d'autres ou par des gens qui traversaient, odieux obstacles qu'il eût été heureux de renverser si le procès-verbal de l'agent ne l'eût retardé plus encore que le passage du piéton. Il comptait le temps qu'il mettait, ajoutait quelques secondes à toutes les minutes pour être sûr de ne pas les avoir faites trop courtes, ce qui lui eût laissé croire plus grande qu'elle n'était en réalité sa chance d'arriver assez tôt et de trouver encore Odette. Et à un moment, comme un fiévreux qui vient de dormir et qui prend conscience de l'absurdité des rêvasseries qu'il ruminait sans se distinguer nettement d'elles, Swann tout d'un coup aperçut en lui l'étrangeté des pensées qu'il roulait depuis le moment où on lui avait dit chez les Verdurin qu'Odette était déjà partie, la nouveauté de la douleur au coeur dont il souffrait, mais qu'il constata seulement comme s'il venait de s'éveiller. Quoi ? toute cette agitation parce qu'il ne verrait Odette que demain, ce que précisément il avait souhaité, il y a une heure, en se rendant chez Mme Verdurin. Il fut bien obligé de constater que dans cette même voiture qui l'emmenait chez Prévost il n'était plus le même, et qu'il n'était plus seul, qu'un être nouveau était là avec lui, adhérent, amalgamé à lui, duquel il ne pourrait peut-être pas se débarrasser, avec qui il allait être obligé d'user de ménagements comme avec un maître ou avec une maladie. Et pourtant depuis un moment qu'il sentait qu'une nouvelle personne s'était ainsi ajoutée à lui, sa vie lui paraissait plus intéressante. C'est à peine s'il se disait que cette rencontre possible chez Prévost (de laquelle l'attente saccageait, dénudait à ce point les moments qui la précédaient qu'il ne trouvait plus une seule idée, un seul souvenir derrière lequel il pût faire reposer son esprit), il était probable pourtant, si elle avait lieu, qu'elle serait comme les autres, fort peu de chose. Comme chaque soir dès qu'il serait avec Odette, jetant furtivement sur son changeant visage un regard aussitôt détourné de peur qu'elle n'y vît l'avance d'un désir et ne crût plus à son désintéressement, il cesserait de pouvoir penser à elle, trop occupé à trouver des prétextes qui lui permissent de ne pas la quitter tout de suite et de s'assurer, sans avoir l'air d'y tenir, qu'il la retrouverait le lendemain chez les Verdurin : c'est-à-dire de prolonger pour l'instant et de renouveler un jour de plus la déception et la torture que lui apportait la vaine présence de cette femme qu'il approchait sans oser l'étreindre.

Elle n'était pas chez Prévost ; il voulut chercher dans tous les restaurants des boulevards. Pour gagner du temps, pendant qu'il visitait les uns, il envoya dans les autres son cocher Rémi (le doge Loredan de Rizzo) qu'il alla attendre ensuite – n'ayant rien trouvé lui-même – à l'endroit qu'il lui avait désigné. La voiture ne revenait pas et Swann se représentait le moment qui approchait, à la fois comme celui où Rémi lui dirait : « cette dame est là », et comme celui où Rémi lui dirait : « cette dame n'était dans aucun des cafés. » Et ainsi il voyait la fin de la soirée devant lui, une et pourtant alternative, précédée soit par la rencontre d'Odette qui abolirait son angoisse, soit par le renoncement forcé à la trouver ce soir, par l'acceptation de rentrer chez lui sans l'avoir vue.

Le cocher revint, mais, au moment où il s'arrêta devant Swann, celui-ci ne lui dit pas : « Avez-vous trouvé cette dame ? » mais : « Faites-moi donc penser demain à commander du bois, je crois que la provision doit commencer à s'épuiser. » Peut-être se disait-il que si Rémi avait trouvé Odette dans un café où elle l'attendait, la fin de la soirée néfaste était déjà anéantie par la réalisation commencée de la fin de soirée bienheureuse et qu'il n'avait pas besoin de se presser d'atteindre un bonheur capturé et en lieu sûr, qui ne s'échapperait plus. Mais aussi c'était par force d'inertie ; il avait dans l'âme le manque de souplesse que certains êtres ont dans le corps, ceux-là qui au moment d'éviter un choc, d'éloigner une flamme de leur habit, d'accomplir un mouvement urgent, prennent leur temps, commencent par rester une seconde dans la situation où ils étaient auparavant comme pour y trouver leur point d'appui, leur élan. Et sans doute si le cocher l'avait interrompu en lui disant : « Cette dame est là », il eut répondu : « Ah ! oui, c'est vrai, la course que je vous avais donnée, tiens je n'aurais pas cru », et aurait continué à lui parler provision de bois pour lui cacher l'émotion qu'il avait eue et se laisser à lui-même le temps de rompre avec l'inquiétude et de se donner au bonheur.

Mais le cocher revint lui dire qu'il ne l'avait trouvée nulle part, et ajouta son avis, en vieux serviteur :
