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
  "le peintre": {
    "aliases": [
      "le peintre",
      "le peintre favori"
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
      "canonical_name": "Mme Cottard",
      "surface_forms": [
        "Mme Cottard"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "comte de Forcheville",
      "surface_forms": [
        "comte de Forcheville"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    },
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
      "source": "comte de Forcheville",
      "target": "Mme Cottard",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Qui est cette dame ? elle a de l'esprit »; la narration souligne son rire « charmant » et « irrésistible » après son mot sur la « salade japonaise ».",
      "explanation": "Forcheville publicly credits Mme Cottard with wit in response to her timely allusion, and the narrator supports this positive impression."
    },
    {
      "event_id": "E2",
      "source": "Swann",
      "target": "Mme Cottard",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.78,
      "evidence": "« Pardonnez-moi... mais j'avoue que mon manque d'admiration est à peu près égal pour ces deux chefs-d'oeuvre. »",
      "explanation": "Swann’s ironic dismissal of the works Mme Cottard champions undercuts her cultural opinions and halts her attempt to engage him on popular theatre/novel taste."
    }
  ],
  "status_effects": [
    {
      "character": "Mme Cottard",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She gains local credit for esprit through Forcheville’s explicit admiration, reinforced by the narrator’s framing."
    },
    {
      "character": "Mme Cottard",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.78,
      "explanation": "Her conversational standing dips when Swann pointedly rejects her taste, prompting her to retreat into non-committal relativism."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-232-p-240"
}

### Candidate characters

[
  "Mme Verdurin",
  "Odette",
  "le narrateur"
]

### Prior local context (optional)

– Mais nous ne croyons pas que vous exagérez, nous voulons seulement que vous mangiez et que mon mari mange aussi ; redonnez de la sole normande à Monsieur, vous voyez bien que la sienne est froide. Nous ne sommes pas si pressés, vous servez comme s'il y avait le feu, attendez donc un peu pour donner la salade.

### Passage

Mme Cottard, qui était modeste et parlait peu, savait pourtant ne pas manquer d'assurance quand une heureuse inspiration lui avait fait trouver un mot juste. Elle sentait qu'elle aurait du succès, cela la mettait en confiance, et ce qu'elle en faisait était moins pour briller que pour être utile à la carrière de son mari. Aussi ne laissa-t-elle pas échapper le mot de salade que venait de prononcer Mme Verdurin.

– Ce n'est pas de la salade japonaise ? dit-elle à mi-voix en se tournant vers Odette.

Et ravie et confuse de l'à-propos et de la hardiesse qu'il y avait à faire ainsi une allusion discrète, mais claire, à la nouvelle et retentissante pièce de Dumas, elle éclata d'un rire charmant d'ingénue, peu bruyant, mais si irrésistible qu'elle resta quelques instants sans pouvoir le maîtriser. « Qui est cette dame ? elle a de l'esprit », dit Forcheville.

– Non, mais nous vous en ferons si vous venez tous dîner vendredi.

– Je vais vous paraître bien provinciale, monsieur, dit Mme Cottard à Swann, mais je n'ai pas encore vu cette fameuse Francillon dont tout le monde parle. Le docteur y est allé (je me rappelle même qu'il m'a dit avoir eu le très grand plaisir de passer la soirée avec vous) et j'avoue que je n'ai pas trouvé raisonnable qu'il louât des places pour y retourner avec moi. Évidemment, au Théâtre-Français, on ne regrette jamais sa soirée, c'est toujours si bien joué, mais comme nous avons des amis très aimables (Mme Cottard prononçait rarement un nom propre et se contentait de dire « des amis à nous », « une de mes amies », par « distinction », sur un ton factice, et avec l'air d'importance d'une personne qui ne nomme que qui elle veut) qui ont souvent des loges et ont la bonne idée de nous emmener à toutes les nouveautés qui en valent la peine, je suis toujours sûre de voir Francillon un peu plus tôt ou un peu plus tard, et de pouvoir me former une opinion. Je dois pourtant confesser que je me trouve assez sotte, car, dans tous les salons où je vais en visite, on ne parle naturellement que de cette malheureuse salade japonaise. On commence même à en être un peu fatigué, ajouta-t-elle en voyant que Swann n'avait pas l'air aussi intéressé qu'elle aurait cru par une si brûlante actualité. Il faut avouer pourtant que cela donne quelquefois prétexte à des idées assez amusantes. Ainsi j'ai une de mes amies qui est très originale, quoique très jolie femme, très entourée, très lancée, et qui prétend qu'elle a fait faire chez elle cette salade japonaise, mais en faisant mettre tout ce qu'Alexandre Dumas fils dit dans la pièce. Elle avait invité quelques amies à venir en manger. Malheureusement je n'étais pas des élues. Mais elle nous l'a raconté tantôt, à son jour ; il paraît que c'était détestable, elle nous a fait rire aux larmes. Mais vous savez, tout est dans la manière de raconter, dit-elle en voyant que Swann gardait un air grave.

Et supposant que c'était peut-être parce qu'il n'aimait pas Francillon :

– Du reste, je crois que j'aurai une déception. Je ne crois pas que cela vaille Serge Panine, l'idole de Mme de Crécy. Voilà au moins des sujets qui ont du fond, qui font réfléchir ; mais donner une recette de salade sur la scène du Théâtre-Français ! Tandis que Serge Panine ! Du reste, comme tout ce qui vient de la plume de Georges Ohnet, c'est toujours si bien écrit. Je ne sais pas si vous connaissez le Maître de Forges que je préférerais encore à Serge Panine.

– Pardonnez-moi, lui dit Swann d'un air ironique, mais j'avoue que mon manque d'admiration est à peu près égal pour ces deux chefs-d'oeuvre.

– Vraiment, qu'est-ce que vous leur reprochez ? Est-ce un parti pris ? Trouvez-vous peut-être que c'est un peu triste ? D'ailleurs, comme je dis toujours, il ne faut jamais discuter sur les romans ni sur les pièces de théâtre. Chacun a sa manière de voir et vous pouvez trouver détestable ce que j'aime le mieux.
