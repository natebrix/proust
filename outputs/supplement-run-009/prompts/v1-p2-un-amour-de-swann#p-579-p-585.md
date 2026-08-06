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
  },
  "M. d'Orsan": {
    "aliases": [
      "M. d'Orsan",
      "d'Orsan",
      "Orsan"
    ]
  },
  "Rémi": {
    "aliases": [
      "Rémi",
      "Remi"
    ]
  },
  "comtesse de Monteriender": {
    "aliases": [
      "comtesse de Monteriender",
      "Mme de Monteriender",
      "Monteriender"
    ]
  },
  "Napoléon III": {
    "aliases": [
      "Napoléon III",
      "Napoleon III"
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
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme Cottard",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Mme Cottard, meilleur thérapeute que n'eût été son mari, avait greffé ... d'autres sentiments, normaux ceux-là ... hâteraient sa transformation définitive ... »",
      "explanation": "The narrator explicitly credits Mme Cottard with therapeutically redirecting Swann’s feelings toward a healthier, calmer affection, elevating her practical wisdom and beneficent influence."
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
      "explanation": "Narrator-endorsed depiction of her as an effective, benevolent ‘thérapeute’ raises her standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-579-p-585"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Swann",
  "comte de Forcheville",
  "le peintre"
]

### Prior local context (optional)

Ayant tenu ces propos que lui inspiraient la hauteur de son aigrette, le chiffre de son porte-cartes, le petit numéro tracé à l'encre dans ses gants par le teinturier et l'embarras de parler à Swann des M. Verdurin, Mme Cottard, voyant qu'on était encore loin du coin de la rue Bonaparte où le conducteur devait l'arrêter, écouta son coeur qui lui conseillait d'autres paroles.

### Passage

– Les oreilles ont dû vous tinter, monsieur, lui dit-elle, pendant le voyage que nous avons fait avec Mme Verdurin. On ne parlait que de vous.

Swann fut bien étonné, il supposait que son nom n'était jamais proféré devant les Verdurin.

– D'ailleurs, ajouta Mme Cottard, Mme de Crécy était là et c'est tout dire. Quand Odette est quelque part, elle ne peut jamais rester bien longtemps sans parler de vous. Et vous pensez que ce n'est pas en mal. Comment ! vous en doutez ? dit-elle, en voyant un geste sceptique de Swann.

Et emportée par la sincérité de sa conviction, ne mettant d'ailleurs aucune mauvaise pensée sous ce mot qu'elle prenait seulement dans le sens où on l'emploie pour parler de l'affection qui unit des amis :

– Mais elle vous adore ! Ah ! je crois qu'il ne faudrait pas dire ça de vous devant elle ! On serait bien arrangé ! À propos de tout, si on voyait un tableau par exemple elle disait : « Ah ! s'il était là, c'est lui qui saurait vous dire si c'est authentique ou non. Il n'y a personne comme lui pour ça. » Et à tout moment elle demandait : « Qu'est-ce qu'il peut faire en ce moment ? Si seulement il travaillait un peu ! C'est malheureux, un garçon si doué, qu'il soit si paresseux. (Vous me pardonnez, n'est-ce pas ?) En ce moment je le vois, il pense à nous, il se demande où nous sommes. » Elle a même eu un mot que j'ai trouvé bien joli ; M. Verdurin lui disait : « Mais comment pouvez-vous voir ce qu'il fait en ce moment puisque vous êtes à huit cents lieues de lui ? » Alors Odette lui a répondu : « Rien n'est impossible à l'oeil d'une amie. » Non je vous jure, je ne vous dis pas cela pour vous flatter, vous avez là une vraie amie comme on n'en a pas beaucoup. Je vous dirai du reste que si vous ne le savez pas, vous êtes le seul. Mme Verdurin me le disait encore le dernier jour (vous savez les veilles de départ on cause mieux) : « Je ne dis pas qu'Odette ne nous aime pas, mais tout ce que nous lui disons ne pèserait pas lourd auprès de ce que lui dirait Swann. » Oh ! mon Dieu, voilà que le conducteur m'arrête, en bavardant avec vous j'allais laisser passer la rue Bonaparte... me rendriez-vous le service de me dire si mon aigrette est droite ? »

Et Mme Cottard sortit de son manchon pour la tendre à Swann sa main gantée de blanc d'où s'échappa, avec une correspondance, une vision de haute vie qui remplit l'omnibus, mêlée à l'odeur du teinturier. Et Swann se sentit déborder de tendresse pour elle, autant que pour Mme Verdurin (et presque autant que pour Odette, car le sentiment qu'il éprouvait pour cette dernière n'étant plus mêlé de douleur, n'était plus guère de l'amour), tandis que de la plate-forme il la suivait de ses yeux attendris, qui enfilait courageusement la rue Bonaparte, l'aigrette haute, d'une main relevant sa jupe, de l'autre tenant son en-tout-cas et son porte-cartes dont elle laissait voir le chiffre, laissant baller devant elle son manchon.

Pour faire concurrence aux sentiments maladifs que Swann avait pour Odette, Mme Cottard, meilleur thérapeute que n'eût été son mari, avait greffé à côté d'eux d'autres sentiments, normaux ceux-là, de gratitude, d'amitié, des sentiments qui dans l'esprit de Swann rendraient Odette plus humaine (plus semblable aux autres femmes, parce que d'autres femmes aussi pouvaient les lui inspirer), hâteraient sa transformation définitive en cette Odette aimée d'affection paisible, qui l'avait ramené un soir après une fête chez le peintre boire un verre d'orangeade avec Forcheville et près de qui Swann avait entrevu qu'il pourrait vivre heureux.
