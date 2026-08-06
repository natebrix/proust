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
  },
  "oncle Adolphe": {
    "aliases": [
      "mon oncle Adolphe",
      "oncle Adolphe",
      "Adolphe"
    ]
  },
  "marquise de Saint-Euverte": {
    "aliases": [
      "marquise de Saint-Euverte",
      "Mme de Saint-Euverte",
      "Saint-Euverte"
    ]
  },
  "général de Froberville": {
    "aliases": [
      "général de Froberville",
      "general de Froberville",
      "Froberville"
    ]
  },
  "marquis de Bréauté": {
    "aliases": [
      "marquis de Bréauté",
      "marquis de Breaute",
      "Bréauté",
      "Breaute"
    ]
  },
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
    ]
  },
  "marquise de Gallardon": {
    "aliases": [
      "marquise de Gallardon",
      "Mme de Gallardon",
      "Gallardon"
    ]
  },
  "duc de Guermantes": {
    "aliases": [
      "duc de Guermantes"
    ]
  },
  "princesse de Parme": {
    "aliases": [
      "princesse de Parme"
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
      "evidence": "« Ces mots ... percèrent douloureusement le coeur de Swann »; « Swann était déjà heureux comme s'il avait parlé d'Odette »; « Il souffrait de rester enfermé au milieu de ces gens dont la bêtise et les ridicules le frappaient... »; « il souffrait surtout ... de prolonger son exil dans ce lieu où Odette ne viendrait jamais... d'où elle était entièrement absente. »",
      "explanation": "The narrator presents Swann as emotionally trapped and isolated among indifferent, shallow company; his love appears unreal to others, and the renewed concert forces him to stay, intensifying his sense of exile from Odette."
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
      "confidence": 0.9,
      "explanation": "Swann is locally diminished by acute distress and a felt exile in a space indifferent to his love for Odette."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-518-p-529"
}

### Candidate characters

[
  "Mme de Cambremer",
  "Mme de Chaussepierre",
  "Odette",
  "général de Froberville"
]

### Prior local context (optional)

Swann refusa ; ayant prévenu baron de Charlus qu'en quittant de chez marquise de Saint-Euverte il rentrerait directement chez lui, il ne se souciait pas en allant chez la princesse de Parme de risquer de manquer un mot qu'il avait tout le temps espéré se voir remettre par un domestique pendant la soirée, et que peut-être il allait trouver chez son concierge. « Ce pauvre Swann, dit ce soir-là princesse des Laumes à son mari, il est toujours gentil, mais il a l'air bien malheureux. Vous le verrez, car il a promis de venir dîner un de ces jours. Je trouve ridicule au fond qu'un homme de son intelligence souffre pour une personne de ce genre et qui n'est même pas intéressante, car on la dit idiote », ajouta-t-elle avec la sagesse des gens non amoureux, qui trouvent qu'un homme d'esprit ne devrait être malheureux que pour une personne qui en valût la peine ; c'est à peu près comme s'étonner qu'on daigne souffrir du choléra par le fait d'un être aussi petit que le bacille virgule.

### Passage

Swann voulait partir, mais au moment où il allait enfin s'échapper, le général de Froberville lui demanda à connaître Mme de Cambremer et il fut obligé de rentrer avec lui dans le salon pour la chercher.

– Dites donc, Swann, j'aimerais mieux être le mari de cette femme-là que d'être massacré par les sauvages, qu'en dites-vous ?

Ces mots « massacré par les sauvages » percèrent douloureusement le coeur de Swann ; aussitôt il éprouva le besoin de continuer la conversation avec le général :

– Ah ! lui dit-il, il y a eu de bien belles vies qui ont fini de cette façon... Ainsi vous savez... ce navigateur dont Dumont d'Urville ramena les cendres, La Pérouse...(et Swann était déjà heureux comme s'il avait parlé d'Odette). C'est un beau caractère et qui m'intéresse beaucoup que celui de La Pérouse, ajouta-t-il d'un air mélancolique.

– Ah ! parfaitement, La Pérouse, dit le général. C'est un nom connu. Il a sa rue.

– Vous connaissez quelqu'un rue La Pérouse ? demanda Swann d'un air agité.

– Je ne connais que Mme de Chanlivault, la soeur de ce brave Chaussepierre. Elle nous a donné une jolie soirée de comédie l'autre jour. C'est un salon qui sera un jour très élégant, vous verrez !

– Ah ! elle demeure rue La Pérouse. C'est sympathique, c'est une jolie rue, si triste.

– Mais non ; c'est que vous n'y êtes pas allé depuis quelque temps ; ce n'est plus triste, cela commence à se construire, tout ce quartier-là.

Quand enfin Swann présenta M. de Froberville à la jeune Mme de Cambremer, comme c'était la première fois qu'elle entendait le nom du général, elle esquissa le sourire de joie et de surprise qu'elle aurait eu si on n'en avait jamais prononcé devant elle d'autre que celui-là, car ne connaissant pas les amis de sa nouvelle famille, à chaque personne qu'on lui amenait, elle croyait que c'était l'un d'eux, et pensant qu'elle faisait preuve de tact en ayant l'air d'en avoir tant entendu parler depuis qu'elle était mariée, elle tendait la main d'un air hésitant destiné à prouver la réserve apprise qu'elle avait à vaincre et la sympathie spontanée qui réussissait à en triompher. Aussi ses beaux-parents, qu'elle croyait encore les gens les plus brillants de France, déclaraient-ils qu'elle était un ange ; d'autant plus qu'ils préféraient paraître, en la faisant épouser à leur fils, avoir cédé à l'attrait plutôt de ses qualités que de sa grande fortune.

– On voit que vous êtes musicienne dans l'âme, madame, lui dit le général, en faisant inconsciemment allusion à l'incident de la bobèche.

Mais le concert recommença et Swann comprit qu'il ne pourrait pas s'en aller avant la fin de ce nouveau numéro du programme. Il souffrait de rester enfermé au milieu de ces gens dont la bêtise et les ridicules le frappaient d'autant plus douloureusement qu'ignorant son amour, incapables, s'ils l'avaient connu, de s'y intéresser et de faire autre chose que d'en sourire comme d'un enfantillage ou de le déplorer comme une folie, ils le lui faisaient apparaître sous l'aspect d'un état subjectif qui n'existait que pour lui, dont rien d'extérieur ne lui affirmait la réalité ; il souffrait surtout, et au point que même le son des instruments lui donnait envie de crier, de prolonger son exil dans ce lieu où Odette ne viendrait jamais, où personne, où rien ne la connaissait, d'où elle était entièrement absente.
