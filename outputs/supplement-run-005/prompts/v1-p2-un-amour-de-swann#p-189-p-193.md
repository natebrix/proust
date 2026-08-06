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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.78,
      "evidence": "Odette détaille les « endroits chics », vante le bal d’Herbinger « où il y avait tout ce qu’il y a de chic à Paris », avoue n’avoir pu obtenir d’invitation, puis se console en disant qu’« au fond j’aime autant ne pas y être allée » et qu’« c’est plutôt pour pouvoir dire qu’on était chez Herbinger ».",
      "explanation": "By letting Odette expose herself through her own discourse (envy, exclusion, then rationalization), the narrative presents her as socially kept at a distance from the « chic » that she elevates into a value, which diminishes her locally."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Her inability to enter the ball that she sets up as a reference and her subsequent rationalization signal a local exclusion from the « chic » that she values."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-189-p-193"
}

### Candidate characters

[
  "Swann",
  "le narrateur"
]

### Prior local context (optional)

Et en effet, elle trouvait Swann, intellectuellement, inférieur à ce qu'elle aurait cru. « Tu gardes toujours ton sang-froid, je ne peux te définir. » Elle s'émerveillait davantage de son indifférence à l'argent, de sa gentillesse pour chacun, de sa délicatesse. Et il arrive en effet souvent pour de plus grands que n'était Swann, pour un savant, pour un artiste, quand il n'est pas méconnu par ceux qui l'entourent, que celui de leurs sentiments qui prouve que la supériorité de son intelligence s'est imposée à eux, ce n'est pas leur admiration pour ses idées, car elles leur échappent, mais leur respect pour sa bonté. C'est aussi du respect qu'inspirait à Odette la situation qu'avait Swann dans le monde, mais elle ne désirait pas qu'il cherchât à l'y faire recevoir. Peut-être sentait-elle qu'il ne pourrait pas y réussir, et même craignait-elle que rien qu'en parlant d'elle il ne provoquât des révélations qu'elle redoutait. Toujours est-il qu'elle lui avait fait promettre de ne jamais prononcer son nom. La raison pour laquelle elle ne voulait pas aller dans le monde, lui avait-elle dit, était une brouille qu'elle avait eue autrefois avec une amie qui, pour se venger, avait ensuite dit du mal d'elle. Swann objectait : « Mais tout le monde n'a pas connu ton amie. » – « Mais si, ça fait la tache d'huile, le monde est si méchant. » D'une part Swann ne comprit pas cette histoire, mais d'autre part il savait que ces propositions : « Le monde est si méchant » et « un propos calomnieux fait la tache d'huile », sont généralement tenues pour vraies ; il devait y avoir des cas auxquels elles s'appliquaient. Celui d'Odette était-il l'un de ceux-là ? Il se le demandait, mais pas longtemps, car il était sujet, lui aussi, à cette lourdeur d'esprit qui s'appesantissait sur son père, quand il se posait un problème difficile. D'ailleurs, ce monde qui faisait si peur à Odette ne lui inspirait peut-être pas de grands désirs, car pour qu'elle se le représentât bien nettement, il était trop éloigné de celui qu'elle connaissait. Pourtant, tout en étant restée à certains égards vraiment simple (elle avait par exemple gardé pour amie une petite couturière retirée dont elle grimpait presque chaque jour l'escalier raide, obscur et fétide), elle avait soif de chic, mais ne s'en faisait pas la même idée que les gens du monde. Pour eux, le chic est une émanation de quelques personnes peu nombreuses qui le projettent jusqu'à un degré assez éloigné – et plus ou moins affaibli dans la mesure où l'on est distant du centre de leur intimité – dans le cercle de leurs amis ou des amis de leurs amis dont les noms forment une sorte de répertoire. Les gens du monde le possèdent dans leur mémoire, ils ont sur ces matières une érudition d'où ils ont extrait une sorte de goût, de tact, si bien que Swann par exemple, sans avoir besoin de faire appel à son savoir mondain, s'il lisait dans un journal les noms des personnes qui se trouvaient à un dîner pouvait dire immédiatement la nuance du chic de ce dîner, comme un lettré, à la simple lecture d'une phrase, apprécie exactement la qualité littéraire de son auteur. Mais Odette faisait partie des personnes (extrêmement nombreuses quoi qu'en pensent les gens du monde, et comme il y en a dans toutes les classes de la société) qui ne possèdent pas ces notions, imaginent un chic tout autre, qui revêt divers aspects selon le milieu auquel elles appartiennent, mais a pour caractère particulier – que ce soit celui dont rêvait Odette, ou celui devant lequel s'inclinait Mme Cottard – d'être directement accessible à tous. L'autre, celui des gens du monde, l'est à vrai dire aussi, mais il y faut quelque délai. Odette disait de quelqu'un :

### Passage

– Il ne va jamais que dans les endroits chics.

Et si Swann lui demandait ce qu'elle entendait par là, elle lui répondait avec un peu de mépris :

– Mais les endroits chics, parbleu ! Si, à ton âge, il faut t'apprendre ce que c'est que les endroits chics, que veux-tu que je te dise, moi ? par exemple, le dimanche matin, l'avenue de l'Impératrice, à cinq heures le tour du Lac, le jeudi l'Éden Théâtre, le vendredi l'Hippodrome, les bals...

– Mais quels bals ?

– Mais les bals qu'on donne à Paris, les bals chics, je veux dire. Tiens, Herbinger, tu sais, celui qui est chez un coulissier ? mais si, tu dois savoir, c'est un des hommes les plus lancés de Paris, ce grand jeune homme blond qui est tellement snob, il a toujours une fleur à la boutonnière, une raie dans le dos, des paletots clairs ; il est avec ce vieux tableau qu'il promène à toutes les premières. Eh bien ! il a donné un bal, l'autre soir, il y avait tout ce qu'il y a de chic à Paris. Ce que j'aurais aimé y aller ! mais il fallait présenter sa carte d'invitation à la porte et je n'avais pas pu en avoir. Au fond j'aime autant ne pas y être allée, c'était une tuerie, je n'aurais rien vu. C'est plutôt pour pouvoir dire qu'on était chez Herbinger. Et tu sais, moi, la gloriole ! Du reste, tu peux bien te dire que sur cent qui racontent qu'elles y étaient, il y a bien la moitié dont ça n'est pas vrai... Mais ça m'étonne que toi, un homme si « pschutt », tu n'y étais pas.
