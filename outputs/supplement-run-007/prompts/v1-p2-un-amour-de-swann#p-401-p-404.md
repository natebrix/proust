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
  },
  "marquis de Forestelle": {
    "aliases": [
      "marquis de Forestelle",
      "M. de Forestelle",
      "Forestelle"
    ]
  },
  "baron de Charlus": {
    "aliases": [
      "baron de Charlus",
      "Charlus"
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
    },
    {
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Swann",
      "target": "Odette",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.8,
      "evidence": "« il recommençait à fabriquer de la tendresse, de la pitié pour Odette. Elle était redevenue l’Odette charmante et bonne »; « il revoyait de la bonté dans son sourire »; « goût pour les sensations que lui donnait la personne d’Odette »",
      "explanation": "Swann’s jealousy softens into renewed tenderness and aesthetic admiration for Odette, recasting her as charming and good. The narrator frames this shift as a recurrent byproduct of Swann’s ‘mal,’ creating an ironic distance."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.8,
      "explanation": "Within Swann’s view, Odette is re-elevated from suspected betrayer to a ‘charmante et bonne’ figure, regaining positive valuation."
    },
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "By replacing jealousy with tenderness and anticipating her grateful return, Swann gains local emotional relief and hope."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-401-p-404"
}

### Candidate characters

[
  "comte de Forcheville"
]

### Prior local context (optional)

Comme il avait dû lui faire de la peine ! Certes il trouvait des raisons valables à son ressentiment contre elle, mais elles n'auraient pas suffi à le lui faire éprouver s'il ne l'avait pas autant aimée. N'avait-il pas eu des griefs aussi graves contre d'autres femmes, auxquelles il eût néanmoins volontiers rendu service aujourd'hui, étant contre elles sans colère parce qu'il ne les aimait plus ? S'il devait jamais un jour se trouver dans le même état d'indifférence vis-à-vis d'Odette, il comprendrait que c'était sa jalousie seule qui lui avait fait trouver quelque chose d'atroce, d'impardonnable, à ce désir, au fond si naturel, provenant d'un peu d'enfantillage et aussi d'une certaine délicatesse d'âme, de pouvoir à son tour, puisqu'une occasion s'en présentait, rendre des politesses aux M. Verdurin, jouer à la maîtresse de maison.

### Passage

Il revenait à ce point de vue – opposé à celui de son amour et de sa jalousie, et auquel il se plaçait quelquefois par une sorte d'équité intellectuelle et pour faire la part des diverses probabilités – d'où il essayait de juger Odette comme s'il ne l'avait pas aimée, comme si elle était pour lui une femme comme les autres, comme si la vie d'Odette n'avait pas été, dès qu'il n'était plus là, différente, tramée en cachette de lui, ourdie contre lui.

Pourquoi croire qu'elle goûterait là-bas avec Forcheville ou avec d'autres des plaisirs enivrants qu'elle n'avait pas connus auprès de lui et que seule sa jalousie forgeait de toutes pièces ? À Bayreuth comme à Paris, s'il arrivait que Forcheville pensât à lui, ce n'eût pu être que comme à quelqu'un qui comptait beaucoup dans la vie d'Odette, à qui il était obligé de céder la place, quand ils se rencontraient chez elle. Si Forcheville et elle triomphaient d'être là-bas malgré lui, c'est lui qui l'aurait voulu en cherchant inutilement à l'empêcher d'y aller, tandis que s'il avait approuvé son projet, d'ailleurs défendable, elle aurait eu l'air d'être là-bas d'après son avis, elle s'y serait sentie envoyée, logée par lui, et le plaisir qu'elle aurait éprouvé à recevoir ces gens qui l'avaient tant reçue, c'est à Swann qu'elle en aurait su gré.

Et – au lieu qu'elle allait partir brouillée avec lui, sans l'avoir revu – s'il lui envoyait cet argent, s'il l'encourageait à ce voyage et s'occupait de le lui rendre agréable, elle allait accourir, heureuse, reconnaissante, et il aurait cette joie de la voir qu'il n'avait pas goûtée depuis près d'une semaine et que rien ne pouvait lui remplacer. Car sitôt que Swann pouvait se la représenter sans horreur, qu'il revoyait de la bonté dans son sourire, et que le désir de l'enlever à tout autre, n'était plus ajouté par la jalousie à son amour, cet amour redevenait surtout un goût pour les sensations que lui donnait la personne d'Odette, pour le plaisir qu'il avait à admirer comme un spectacle ou à interroger comme un phénomène le lever d'un de ses regards, la formation d'un de ses sourires, l'émission d'une intonation de sa voix. Et ce plaisir différent de tous les autres avait fini par créer en lui un besoin d'elle et qu'elle seule pouvait assouvir par sa présence ou ses lettres, presque aussi désintéressé, presque aussi artistique, aussi pervers, qu'un autre besoin qui caractérisait cette période nouvelle de la vie de Swann où à la sécheresse, à la dépression des années antérieures, avait succédé une sorte de trop-plein spirituel, sans qu'il sût davantage à quoi il devait cet enrichissement inespéré de sa vie intérieure qu'une personne de santé délicate qui à partir d'un certain moment se fortifie, engraisse, et semble pendant quelque temps s'acheminer vers une complète guérison – cet autre besoin qui se développait aussi en dehors du monde réel, c'était celui d'entendre, de connaître de la musique.

Ainsi, par le chimisme même de son mal, après qu'il avait fait de la jalousie avec son amour, il recommençait à fabriquer de la tendresse, de la pitié pour Odette. Elle était redevenue l'Odette charmante et bonne. Il avait des remords d'avoir été dur pour elle. Il voulait qu'elle vînt près de lui et, auparavant, il voulait lui avoir procuré quelque plaisir, pour voir la reconnaissance pétrir son visage et modeler son sourire.
