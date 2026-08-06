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
        "l'oncle de Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "Il n’en était pas moins vrai que l’idéal de Charlus était fort factice... jouant ainsi sur le mot par une équivoque qui le trompait lui-même et où résidait le mensonge de cette conception bâtarde",
      "explanation": "The narrator sharply lowers Charlus by diagnosing his ideal as self-deceiving and factitious, a confused blend of aristocracy, generosity, and art maintained through a misleading equivocation."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "il était permis de se féliciter qu'ils eussent fait défaut chez Charlus, lequel avait fait transporter chez lui une grande partie des admirables boiseries de l'hôtel Guermantes",
      "explanation": "The narrator grants Charlus a pragmatic credit for preserving aristocratic art and heritage, in contrast to Saint-Loup’s modernizing choices."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.91,
      "explanation": "Despite a brief concession for preserving heritage, the dominant effect is negative due to the narrator’s critique of his factitious, self-deceiving ideal."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-214-p-215"
}

### Candidate characters

[
  "Robert de Saint-Loup",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Je n'osais lui répondre qu'on en aurait éprouvé bien plus à Combray si j'avais eu l'air de ne pas le croire.

### Passage

Ma grand'mère fut enchantée de Charlus. Sans doute il attachait une extrême importance à toutes les questions de naissance et de situation mondaine, et ma grand'mère l'avait remarqué mais sans rien de cette sévérité où entrent d'habitude une secrète envie et l'irritation de voir un autre se réjouir d'avantages qu'on voudrait et qu'on ne peut posséder. Comme au contraire ma grand'mère, contente de son sort et ne regrettant nullement de ne pas vivre dans une société plus brillante, ne se servait que de son intelligence pour observer les travers de Charlus, elle parlait de l'oncle de Saint-Loup avec cette bienveillance détachée, souriante, presque sympathique, par laquelle nous récompensons l'objet de notre observation désintéressée du plaisir qu'elle nous procure, et d'autant plus que cette fois l'objet était un personnage dont elle trouvait que les prétentions sinon légitimes, du moins pittoresques, le faisaient assez vivement trancher sur les personnes qu'elle avait généralement l'occasion de voir. Mais c'était surtout en faveur de l'intelligence et de la sensibilité, qu'on devinait extrêmement vives chez Charlus, au contraire de tant de gens du monde dont se moquait Saint-Loup, que ma grand'mère lui avait si aisément pardonné son préjugé aristocratique. Celui-ci n'avait pourtant pas été sacrifié par l'oncle, comme par le neveu, à des qualités supérieures. Charlus l'avait plutôt concilié avec elles. Possédant, comme descendant des ducs de Nemours et des princes de Lamballe, des archives, des meubles, des tapisseries, des portraits faits pour ses aïeux par Raphaël, par Vélasquez, par Boucher, pouvant dire justement qu'il visitait un musée et une incomparable bibliothèque rien qu'en parcourant ses souvenirs de famille, il plaçait au contraire au rang d'où son neveu l'avait fait déchoir tout l'héritage de l'aristocratie. Peut-être aussi moins idéologue que Saint-Loup, se payant moins de mots, plus réaliste observateur des hommes, ne voulait-il pas négliger un élément essentiel de prestige à leurs yeux et qui, s'il donnait à son imagination des jouissances désintéressées, pouvait être souvent pour son activité utilitaire un adjuvant puissamment efficace. Le débat reste ouvert entre les hommes de cette sorte et ceux qui obéissent à l'idéal intérieur qui les pousse à se défaire de ces avantages pour chercher uniquement à le réaliser, semblables en cela aux peintres, aux écrivains qui renoncent à leur virtuosité, aux peuples artistes qui se modernisent, aux peuples guerriers prenant l'initiative du désarmement universel, aux gouvernements absolus qui se font démocratiques et abrogent de dures lois, bien souvent sans que la réalité récompense leur noble effort ; car les uns perdent leur talent, les autres leur prédominance séculaire ; le pacifisme multiplie quelquefois les guerres et l'indulgence la criminalité. Si les efforts de sincérité et d'émancipation de Saint-Loup ne pouvaient être trouvés que très nobles, à en juger par le résultat extérieur, il était permis de se féliciter qu'ils eussent fait défaut chez Charlus, lequel avait fait transporter chez lui une grande partie des admirables boiseries de l'hôtel Guermantes au lieu de les échanger, comme son neveu, contre un mobilier modern style, des Lebourg et des Guillaumin. Il n'en était pas moins vrai que l'idéal de Charlus était fort factice, et si cette épithète peut être rapprochée du mot idéal, tout autant mondain qu'artistique. À quelques femmes de grande beauté et de rare culture dont les aïeules avaient été deux siècles plus tôt mêlées à toute la gloire et à toute l'élégance de l'ancien régime, il trouvait une distinction qui le faisait pouvoir se plaire seulement avec elles, et sans doute l'admiration qu'il leur avait vouée était sincère, mais de nombreuses réminiscences d'histoire et d'art évoquées par leurs noms y entraient pour une grande part, comme des souvenirs de l'antiquité sont une des raisons du plaisir qu'un lettré trouve à lire une ode d'Horace peut-être inférieure à des poèmes de nos jours qui laisseraient ce même lettré indifférent. Chacune de ces femmes à côté d'une jolie bourgeoise était pour lui ce que sont à une toile contemporaine représentant une route ou une noce, ces tableaux anciens dont on sait l'histoire, depuis le Pape ou le Roi qui les commandèrent, en passant par tels personnages auprès de qui leur présence, par don, achat, prise ou héritage nous rappelle quelque événement, ou tout au moins quelque alliance d'un intérêt historique, par conséquent des connaissances que nous avons acquises, leur donne une nouvelle utilité, augmente le sentiment de la richesse des possessions de notre mémoire ou de notre érudition. Charlus se félicitait qu'un préjugé analogue au sien, en empêchant ces quelques grandes dames de frayer avec des femmes d'un sang moins pur, les offrît à son culte intactes, dans leur noblesse inaltérée, comme telle façade du XVIIIe siècle soutenue par ses colonnes plates de marbre rose et à laquelle les temps nouveaux n'ont rien changé.

Charlus célébrait la véritable noblesse d'esprit et de coeur de ces femmes, jouant ainsi sur le mot par une équivoque qui le trompait lui-même et où résidait le mensonge de cette conception bâtarde, de cet ambigu d'aristocratie, de générosité et d'art, mais aussi sa séduction, dangereuse pour des êtres comme ma grand'mère à qui le préjugé plus grossier mais plus innocent d'un noble qui ne regarde qu'aux quartiers et ne se soucie pas du reste eût semblé trop ridicule, mais qui était sans défense dès que quelque chose se présentait sous les dehors d'une supériorité spirituelle, au point qu'elle trouvait les princes enviables par-dessus tous les hommes parce qu'ils purent avoir un La Bruyère, un Fénelon comme précepteurs.
