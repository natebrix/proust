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
  },
  "Gilberte": {
    "aliases": [
      "Gilberte"
    ]
  },
  "Françoise": {
    "aliases": [
      "Françoise",
      "Francoise"
    ]
  },
  "la Berma": {
    "aliases": [
      "la Berma",
      "Berma"
    ]
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "le marquis de Norpois"
    ]
  },
  "la grand-mère": {
    "aliases": [
      "ma grand-mère",
      "grand-mère",
      "ma grand'mère",
      "grand'mère",
      "la grand-mère"
    ]
  },
  "M. de Stermaria": {
    "aliases": [
      "M. de Stermaria",
      "de Stermaria",
      "Stermaria"
    ]
  },
  "Aimé": {
    "aliases": [
      "Aimé",
      "Aime"
    ]
  },
  "Mlle de Stermaria": {
    "aliases": [
      "Mlle de Stermaria"
    ]
  },
  "marquis de Cambremer": {
    "aliases": [
      "marquis de Cambremer",
      "M. de Cambremer"
    ]
  },
  "princesse de Luxembourg": {
    "aliases": [
      "princesse de Luxembourg",
      "La princesse de Luxembourg"
    ]
  },
  "le père du narrateur": {
    "aliases": [
      "mon père",
      "votre père"
    ]
  },
  "Mme Blandais": {
    "aliases": [
      "Mme Blandais",
      "Madame Blandais"
    ]
  },
  "Mme Poncin": {
    "aliases": [
      "Mme Poncin",
      "Madame Poncin"
    ]
  },
  "Robert de Saint-Loup": {
    "aliases": [
      "Saint-Loup",
      "Robert de Saint-Loup",
      "marquis de Saint-Loup-en-Bray",
      "le neveu de Mme de Villeparisis"
    ]
  },
  "M. de Marsantes": {
    "aliases": [
      "M. de Marsantes",
      "Marsantes",
      "Saint-Loup de Saint-Loup"
    ]
  },
  "Bloch": {
    "aliases": [
      "Bloch",
      "Bloch fils"
    ]
  },
  "prince des Laumes": {
    "aliases": [
      "prince des Laumes"
    ]
  },
  "Bloch père": {
    "aliases": [
      "Bloch père"
    ]
  },
  "le directeur": {
    "aliases": [
      "le directeur",
      "directeur"
    ]
  },
  "Dreyfus": {
    "aliases": [
      "Dreyfus"
    ]
  },
  "jeune blonde de Rivebelle": {
    "aliases": [
      "jeune blonde",
      "jeune blonde à l'air triste"
    ]
  },
  "duchesse de Guermantes": {
    "aliases": [
      "duchesse de Guermantes",
      "Mme de Guermantes",
      "Madame de Guermantes",
      "la duchesse"
    ]
  },
  "Jupien": {
    "aliases": [
      "Jupien"
    ]
  },
  "princesse de Guermantes": {
    "aliases": [
      "princesse de Guermantes",
      "princesse de Guermantes-Bavière",
      "Mme de Guermantes-Bavière"
    ]
  },
  "duc de Châtellerault": {
    "aliases": [
      "duc de Châtellerault",
      "M. de Châtellerault",
      "Châtellerault"
    ]
  },
  "M. de Vaugoubert": {
    "aliases": [
      "M. de Vaugoubert",
      "Vaugoubert"
    ]
  },
  "Mme de Vaugoubert": {
    "aliases": [
      "Mme de Vaugoubert",
      "Madame de Vaugoubert"
    ]
  },
  "Albertine": {
    "aliases": [
      "Albertine"
    ]
  },
  "Andrée": {
    "aliases": [
      "Andrée",
      "Andree"
    ]
  },
  "Mme Bontemps": {
    "aliases": [
      "Mme Bontemps",
      "Madame Bontemps"
    ]
  },
  "Morel": {
    "aliases": [
      "Morel"
    ]
  },
  "Elstir": {
    "aliases": [
      "Elstir"
    ]
  },
  "prince de Léon": {
    "aliases": [
      "prince de Léon",
      "prince de Leon",
      "Léon",
      "Leon"
    ]
  },
  "marquis du Lau": {
    "aliases": [
      "marquis du Lau",
      "du Lau"
    ]
  },
  "Mme de Chaussepierre": {
    "aliases": [
      "Mme de Chaussepierre",
      "Madame de Chaussepierre",
      "Chaussepierre"
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
      "evidence": "« il courait chez elle et exigeait de la voir tous les jours suivants »; « contrairement au calcul de Swann, le consentement d'Odette avait tout changé en lui »; « l'amour de Swann en était arrivé à ce degré où le médecin… se demandent si priver un malade de son vice… est encore raisonnable ou même possible ».",
      "explanation": "The narrator depicts Swann’s failed attempts at self-imposed separation, his self-deception, and addiction-like compulsion, culminating in a clinical metaphor that frames his dependence as nearly incurable, reducing his autonomy and leverage."
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
      "explanation": "Swann’s local standing is weakened by his inability to sustain distance and by the narrator’s portrayal of his love as an overpowering, near-incurable dependency."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-406-p-410"
}

### Candidate characters

[
  "Odette"
]

### Prior local context (optional)

Aussi Odette, sûre de le voir venir après quelques jours, aussi tendre et soumis qu'avant, lui demander une réconciliation, prenait-elle l'habitude de ne plus craindre de lui déplaire et même de l'irriter et lui refusait-elle, quand cela lui était commode, les faveurs auxquelles il tenait le plus.

### Passage

Peut-être ne savait-elle pas combien il avait été sincère vis-à-vis d'elle pendant la brouille, quand il lui avait dit qu'il ne lui enverrait pas d'argent et chercherait à lui faire du mal. Peut-être ne savait-elle pas davantage combien il l'était, vis-à-vis sinon d'elle, du moins de lui-même, en d'autres cas où dans l'intérêt de l'avenir de leur liaison, pour montrer à Odette qu'il était capable de se passer d'elle, qu'une rupture restait toujours possible, il décidait de rester quelque temps sans aller chez elle.

Parfois c'était après quelques jours où elle ne lui avait pas causé de souci nouveau ; et comme, des visites prochaines qu'il lui ferait, il savait qu'il ne pouvait tirer nulle bien grande joie, mais plus probablement quelque chagrin qui mettrait fin au calme où il se trouvait, il lui écrivait qu'étant très occupé il ne pourrait la voir aucun des jours qu'il lui avait dit. Or une lettre d'elle, se croisant avec la sienne, le priait précisément de déplacer un rendez-vous. Il se demandait pourquoi ; ses soupçons, sa douleur le reprenaient. Il ne pouvait plus tenir, dans l'état nouveau d'agitation où il se trouvait, l'engagement qu'il avait pris dans l'état antérieur de calme relatif, il courait chez elle et exigeait de la voir tous les jours suivants. Et même si elle ne lui avait pas écrit la première, si elle répondait seulement, cela suffisait pour qu'il ne pût plus rester sans la voir. Car, contrairement au calcul de Swann, le consentement d'Odette avait tout changé en lui. Comme tous ceux qui possèdent une chose, pour savoir ce qui arriverait s'il cessait un moment de la posséder, il avait ôté cette chose de son esprit, en y laissant tout le reste dans le même état que quand elle était là. Or l'absence d'une chose, ce n'est pas que cela, ce n'est pas un simple manque partiel, c'est un bouleversement de tout le reste, c'est un état nouveau qu'on ne peut prévoir dans l'ancien.

Mais d'autres fois au contraire – Odette était sur le point de partir en voyage – c'était après quelque petite querelle dont il choisissait le prétexte, qu'il se résolvait à ne pas lui écrire et à ne pas la revoir avant son retour, donnant ainsi les apparences, et demandant le bénéfice d'une grande brouille, qu'elle croirait peut-être définitive, à une séparation dont la plus longue part était inévitable du fait du voyage et qu'il faisait commencer seulement un peu plus tôt. Déjà il se figurait Odette inquiète, affligée, de n'avoir reçu ni visite ni lettre et cette image, en calmant sa jalousie, lui rendait facile de se déshabituer de la voir. Sans doute, par moments, tout au bout de son esprit où sa résolution la refoulait grâce à toute la longueur interposée des trois semaines de séparation acceptée, c'était avec plaisir qu'il considérait l'idée qu'il reverrait Odette à son retour : mais c'était aussi avec si peu d'impatience, qu'il commençait à se demander s'il ne doublerait pas volontairement la durée d'une abstinence si facile. Elle ne datait encore que de trois jours, temps beaucoup moins long que celui qu'il avait souvent passé en ne voyant pas Odette, et sans l'avoir comme maintenant prémédité. Et pourtant voici qu'une légère contrariété ou un malaise physique – en l'incitant à considérer le moment présent comme un moment exceptionnel, en dehors de la règle, où la sagesse même admettrait d'accueillir l'apaisement qu'apporte un plaisir et de donner congé, jusqu'à la reprise utile de l'effort, à la volonté – suspendait l'action de celle-ci qui cessait d'exercer sa compression ; ou, moins que cela, le souvenir d'un renseignement qu'il avait oublié de demander à Odette, si elle avait décidé la couleur dont elle voulait faire repeindre sa voiture, ou, pour une certaine valeur de bourse, si c'était des actions ordinaires ou privilégiées qu'elle désirait acquérir (c'était très joli de lui montrer qu'il pouvait rester sans la voir, mais si après ça la peinture était à refaire ou si les actions ne donnaient pas de dividende, il serait bien avancé), voici que comme un caoutchouc tendu qu'on lâche ou comme l'air dans une machine pneumatique qu'on entr'ouvre, l'idée de la revoir, des lointains où elle était maintenue, revenait d'un bond dans le champ du présent et des possibilités immédiates.

Elle y revenait sans plus trouver de résistance, et d'ailleurs si irrésistible que Swann avait eu bien moins de peine à sentir s'approcher un à un les quinze jours qu'il devait rester séparé d'Odette, qu'il n'en avait à attendre les dix minutes que son cocher mettait pour atteler la voiture qui allait l'emmener chez elle et qu'il passait dans des transports d'impatience et de joie où il ressaisissait mille fois pour lui prodiguer sa tendresse, cette idée de la retrouver qui, par un retour si brusque, au moment où il la croyait si loin, était de nouveau près de lui dans sa plus proche conscience. C'est qu'elle ne trouvait plus pour lui faire obstacle le désir de chercher sans plus tarder à lui résister, qui n'existait plus chez Swann depuis que, s'étant prouvé à lui-même – il le croyait du moins – qu'il en était si aisément capable, il ne voyait plus aucun inconvénient à ajourner un essai de séparation qu'il était certain maintenant de mettre à exécution dès qu'il le voudrait. C'est aussi que cette idée de la revoir revenait parée pour lui d'une nouveauté, d'une séduction, douée d'une virulence que l'habitude avait émoussées, mais qui s'étaient retrempées dans cette privation non de trois jours mais de quinze (car la durée d'un renoncement doit se calculer, par anticipation, sur le terme assigné), et de ce qui jusque-là eût été un plaisir attendu qu'on sacrifie aisément, avait fait un bonheur inespéré contre lequel on est sans force. C'est enfin qu'elle y revenait embellie par l'ignorance où était Swann de ce qu'Odette avait pu penser, faire peut-être en voyant qu'il ne lui avait pas donné signe de vie, si bien que ce qu'il allait trouver c'était la révélation passionnante d'une Odette presque inconnue.

Mais elle, de même qu'elle avait cru que son refus d'argent n'était qu'une feinte, ne voyait qu'un prétexte dans le renseignement que Swann venait lui demander sur la voiture à repeindre ou la valeur à acheter. Car elle ne reconstituait pas les diverses phases de ces crises qu'il traversait et, dans l'idée qu'elle s'en faisait, elle omettait d'en comprendre le mécanisme, ne croyant qu'à ce qu'elle connaissait d'avance, à la nécessaire, à l'infaillible et toujours identique terminaison. Idée incomplète – d'autant plus profonde peut-être – si on la jugeait du point de vue de Swann qui eût sans doute trouvé qu'il était incompris d'Odette, comme un morphinomane ou un tuberculeux, persuadés qu'ils ont été arrêtés, l'un par un événement extérieur au moment où il allait se délivrer de son habitude invétérée, l'autre par une indisposition accidentelle au moment où il allait être enfin rétabli, se sentent incompris du médecin qui n'attache pas la même importance qu'eux à ces prétendues contingences, simples déguisements, selon lui, revêtus, pour redevenir sensibles à ses malades, par le vice et l'état morbide qui, en réalité, n'ont pas cessé de peser incurablement sur eux tandis qu'ils berçaient des rêves de sagesse ou de guérison. Et de fait, l'amour de Swann en était arrivé à ce degré où le médecin et, dans certaines affections, le chirurgien le plus audacieux, se demandent si priver un malade de son vice ou lui ôter son mal, est encore raisonnable ou même possible.
