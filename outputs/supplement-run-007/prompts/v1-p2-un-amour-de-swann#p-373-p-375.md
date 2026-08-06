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
      "confidence": 0.9,
      "evidence": "Swann « lui tenait le même raisonnement, au même degré d'insincérité… » et « obéissait aussi au désir de la prendre par l'amour-propre »; son long laïus la traite d'« eau informe », « poisson sans mémoire »; Odette, lisant cela comme un discours d’un homme amoureux « il était inutile de leur obéir », sourit et craint surtout de « finir par manquer l’Ouverture ! »",
      "explanation": "The narrator exposes Swann’s insincerity and maneuver, then shows that his harangue fails: Odette interprets it as a sign of love and does not let herself be diverted from the Opéra-Comique. This combination lowers Swann locally and reveals his weak hold over her."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "rhetorical_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "His plea, unmasked as insincere and manipulative, does not win Odette’s assent; he clearly loses in persuasive power."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-373-p-375"
}

### Candidate characters

[
  "M. Verdurin",
  "Odette",
  "le narrateur"
]

### Prior local context (optional)

docteur Cottard à ces mots manifesta en même temps son étonnement et sa soumission, comme devant une vérité contraire à tout ce qu'il avait cru jusque-là, mais d'une évidence irrésistible ; et, baissant d'un air ému et peureux son nez dans son assiette, il se contenta de répondre : « Ah ! ah ! ah ! ah ! ah ! » en traversant à reculons, dans sa retraite repliée en bon ordre jusqu'au fond de lui-même, le long d'une gamme descendante, tout le registre de sa voix. Et il ne fut plus question de Swann chez les M. Verdurin.

### Passage

Alors ce salon qui avait réuni Swann et Odette devint un obstacle à leurs rendez-vous. Elle ne lui disait plus comme au premier temps de leur amour : « Nous nous verrons en tous cas demain soir, il y a un souper chez les Verdurin » mais : « Nous ne pourrons pas nous voir demain soir, il y a un souper chez les Verdurin. » Ou bien les Verdurin devaient l'emmener à l'Opéra-Comique voir « Une nuit de Cléopâtre » et Swann lisait dans les yeux d'Odette cet effroi qu'il lui demandât de n'y pas aller, que naguère il n'aurait pu se retenir de baiser au passage sur le visage de sa maîtresse, et qui maintenant l'exaspérait. « Ce n'est pas de la colère, pourtant, se disait-il à lui-même, que j'éprouve en voyant l'envie qu'elle a d'aller picorer dans cette musique stercoraire. C'est du chagrin, non pas certes pour moi, mais pour elle ; du chagrin de voir qu'après avoir vécu plus de six mois en contact quotidien avec moi, elle n'a pas su devenir assez une autre pour éliminer spontanément Victor Massé ! Surtout pour ne pas être arrivée à comprendre qu'il y a des soirs où un être d'une essence un peu délicate doit savoir renoncer à un plaisir, quand on le lui demande. Elle devrait savoir dire « je n'irai pas », ne fût-ce que par intelligence, puisque c'est sur sa réponse qu'on classera une fois pour toutes sa qualité d'âme. » Et s'étant persuadé à lui-même que c'était seulement en effet pour pouvoir porter un jugement plus favorable sur la valeur spirituelle d'Odette qu'il désirait que ce soir-là elle restât avec lui au lieu d'aller à l'Opéra-Comique, il lui tenait le même raisonnement, au même degré d'insincérité qu'à soi-même, et même, à un degré de plus, car alors il obéissait aussi au désir de la prendre par l'amour-propre.

– Je te jure, lui disait-il, quelques instants avant qu'elle partît pour le théâtre, qu'en te demandant de ne pas sortir, tous mes souhaits, si j'étais égoïste, seraient pour que tu me refuses, car j'ai mille choses à faire ce soir et je me trouverai moi-même pris au piège et bien ennuyé si contre toute attente tu me réponds que tu n'iras pas. Mais mes occupations, mes plaisirs, ne sont pas tout, je dois penser à toi. Il peut venir un jour où me voyant à jamais détaché de toi tu auras le droit de me reprocher de ne pas t'avoir avertie dans les minutes décisives où je sentais que j'allais porter sur toi un de ces jugements sévères auxquels l'amour ne résiste pas longtemps. Vois-tu, « Une nuit de Cléopâtre » (quel titre !) n'est rien dans la circonstance. Ce qu'il faut savoir, c'est si vraiment tu es cet être qui est au dernier rang de l'esprit, et même du charme, l'être méprisable qui n'est pas capable de renoncer à un plaisir. Alors, si tu es cela, comment pourrait-on t'aimer, car tu n'es même pas une personne, une créature définie, imparfaite, mais du moins perfectible ? Tu es une eau informe qui coule selon la pente qu'on lui offre, un poisson sans mémoire et sans réflexion qui tant qu'il vivra dans son aquarium se heurtera cent fois par jour contre le vitrage qu'il continuera à prendre pour de l'eau. Comprends-tu que ta réponse, je ne dis pas aura pour effet que je cesserai de t'aimer immédiatement, bien entendu, mais te rendra moins séduisante à mes yeux quand je comprendrai que tu n'es pas une personne, que tu es au-dessous de toutes les choses et ne sais te placer au-dessus d'aucune ? Évidemment j'aurais mieux aimé te demander comme une chose sans importance, de renoncer à « Une nuit de Cléopâtre » (puisque tu m'obliges à me souiller les lèvres de ce nom abject) dans l'espoir que tu irais cependant. Mais, décidé à tenir un tel compte, à tirer de telles conséquences de ta réponse, j'ai trouvé plus loyal de t'en prévenir.

Odette depuis un moment donnait des signes d'émotion et d'incertitude. À défaut du sens de ce discours, elle comprenait qu'il pouvait rentrer dans le genre commun des « laïus », et scènes de reproches ou de supplications dont l'habitude qu'elle avait des hommes lui permettait, sans s'attacher aux détails des mots, de conclure qu'ils ne les prononceraient pas s'ils n'étaient pas amoureux, que du moment qu'ils étaient amoureux, il était inutile de leur obéir, qu'ils ne le seraient que plus après. Aussi aurait-elle écouté Swann avec le plus grand calme si elle n'avait vu que l'heure passait et que pour peu qu'il parlât encore quelque temps, elle allait, comme elle le lui dit avec un sourire tendre, obstiné et confus, « finir par manquer l'Ouverture ! »
