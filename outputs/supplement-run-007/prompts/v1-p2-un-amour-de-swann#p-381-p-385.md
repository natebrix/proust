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
      "evidence": "Il se plongeait dans « le plus enivrant des romans d'amour, l'indicateur des chemins de fer »; « cette restriction apportée pour lui au droit universel de libre circulation, n'était qu'une des formes de cet esclavage »; « Il attendait toute la nuit... Odette était à Paris depuis midi; elle n'avait pas eu l'idée de l'en prévenir. »",
      "explanation": "The narrator presents Swann as emotionally enslaved and self-restricting, rationalizing travel timetables as 'authorization,' foregoing movement for Odette, and keeping a futile vigil while she returns without notifying him."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.85,
      "explanation": "Locally excluded from Odette’s movements and even from ordinary travel ('n'avait plus le droit de voyager'), and left uninformed of her return."
    },
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann is shown in a state of jealousy and dependence, culminating in an anxious all-night wait contrasted with Odette’s indifference."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-381-p-385"
}

### Candidate characters

[
  "M. Verdurin",
  "Odette",
  "Robert de Saint-Loup",
  "le narrateur",
  "marquis de Forestelle"
]

### Prior local context (optional)

Mais si elle n'y réussissait pas, M. Verdurin partait en campagne, trouvait un bureau de télégraphe ou un messager et s'informait de ceux des fidèles qui avaient quelqu'un à faire prévenir. Mais Odette le remerciait et disait qu'elle n'avait de dépêche à faire pour personne, car elle avait dit à Swann une fois pour toutes qu'en lui en envoyant une aux yeux de tous, elle se compromettrait. Parfois c'était pour plusieurs jours qu'elle s'absentait, les M. Verdurin l'emmenaient voir les tombeaux de Dreux, ou à Compiègne admirer, sur le conseil du le peintre, des couchers de soleil en forêt et on poussait jusqu'au château de Pierrefonds.

### Passage

– Penser qu'elle pourrait visiter de vrais monuments avec moi qui ai étudié l'architecture pendant dix ans et qui suis tout le temps supplié de mener à Beauvais ou à Saint-Loup-de-Naud des gens de la plus haute valeur et ne le ferais que pour elle, et qu'à la place elle va avec les dernières des brutes s'extasier successivement devant les déjections de Louis-Philippe et devant celles de Viollet-le-Duc ! Il me semble qu'il n'y a pas besoin d'être artiste pour cela et que, même sans flair particulièrement fin, on ne choisit pas d'aller villégiaturer dans des latrines pour être plus à portée de respirer des excréments.

Mais quand elle était partie pour Dreux ou pour Pierrefonds – hélas, sans lui permettre d'y aller, comme par hasard, de son côté, car « cela ferait un effet déplorable », disait-elle – il se plongeait dans le plus enivrant des romans d'amour, l'indicateur des chemins de fer, qui lui apprenait les moyens de la rejoindre, l'après-midi, le soir, ce matin même ! Le moyen ? presque davantage : l'autorisation. Car enfin l'indicateur et les trains eux-mêmes n'étaient pas faits pour des chiens. Si on faisait savoir au public, par voie d'imprimés, qu'à huit heures du matin partait un train qui arrivait à Pierrefonds à dix heures, c'est donc qu'aller à Pierrefonds était un acte licite, pour lequel la permission d'Odette était superflue ; et c'était aussi un acte qui pouvait avoir un tout autre motif que le désir de rencontrer Odette, puisque des gens qui ne la connaissaient pas l'accomplissaient chaque jour, en assez grand nombre pour que cela valût la peine de faire chauffer des locomotives.

En somme elle ne pouvait tout de même pas l'empêcher d'aller à Pierrefonds s'il en avait envie ! Or, justement, il sentait qu'il en avait envie, et que s'il n'avait pas connu Odette, certainement il y serait allé. Il y avait longtemps qu'il voulait se faire une idée plus précise des travaux de restauration de Viollet-le-Duc. Et par le temps qu'il faisait, il éprouvait l'impérieux désir d'une promenade dans la forêt de Compiègne.

Ce n'était vraiment pas de chance qu'elle lui défendît le seul endroit qui le tentait aujourd'hui. Aujourd'hui ! S'il y allait, malgré son interdiction, il pourrait la voir aujourd'hui même ! Mais, alors que, si elle eût retrouvé à Pierrefonds quelque indifférent, elle lui eût dit joyeusement : « Tiens, vous ici ! », et lui aurait demandé d'aller la voir à l'hôtel où elle était descendue avec les Verdurin, au contraire si elle l'y rencontrait, lui, Swann, elle serait froissée, elle se dirait qu'elle était suivie, elle l'aimerait moins, peut-être se détournerait-elle avec colère en l'apercevant. « Alors, je n'ai plus le droit de voyager ! » lui dirait-elle au retour, tandis qu'en somme c'était lui qui n'avait plus le droit de voyager !

Il avait eu un moment l'idée, pour pouvoir aller à Compiègne et à Pierrefonds sans avoir l'air que ce fût pour rencontrer Odette, de s'y faire emmener par un de ses amis, le marquis de Forestelle, qui avait un château dans le voisinage. Celui-ci, à qui il avait fait part de son projet sans lui en dire le motif, ne se sentait pas de joie et s'émerveillait que Swann, pour la première fois depuis quinze ans, consentît enfin à venir voir sa propriété et, puisqu'il ne voulait pas s'y arrêter, lui avait-il dit, lui promît du moins de faire ensemble des promenades et des excursions pendant plusieurs jours. Swann s'imaginait déjà là-bas avec M. de Forestelle. Même avant d'y voir Odette, même s'il ne réussissait pas à l'y voir, quel bonheur il aurait à mettre le pied sur cette terre où ne sachant pas l'endroit exact, à tel moment, de sa présence, il sentirait palpiter partout la possibilité de sa brusque apparition : dans la cour du château, devenu beau pour lui parce que c'était à cause d'elle qu'il était allé le voir ; dans toutes les rues de la ville, qui lui semblait romanesques ; sur chaque route de la forêt, rosée par un couchant profond et tendre ; asiles innombrables et alternatifs, où venait simultanément se réfugier, dans l'incertaine ubiquité de ses espérances, son coeur heureux, vagabond et multiplié. « Surtout, dirait-il à M. de Forestelle, prenons garde de ne pas tomber sur Odette et les Verdurin ; je viens d'apprendre qu'ils sont justement aujourd'hui à Pierrefonds. On a assez le temps de se voir à Paris, ce ne serait pas la peine de le quitter pour ne pas pouvoir faire un pas les uns sans les autres. » Et son ami ne comprendrait pas pourquoi une fois là-bas il changerait vingt fois de projets, inspecterait les salles à manger de tous les hôtels de Compiègne sans se décider à s'asseoir dans aucune de celles où pourtant on n'avait pas vu trace de Verdurin, ayant l'air de rechercher ce qu'il disait vouloir fuir et du reste le fuyant dès qu'il l'aurait trouvé, car s'il avait rencontré le petit groupe, il s'en serait écarté avec affectation, content d'avoir vu Odette et qu'elle l'eût vu, surtout qu'elle l'eût vu ne se souciant pas d'elle. Mais non, elle devinerait bien que c'était pour elle qu'il était là. Et quand M. de Forestelle venait le chercher pour partir, il lui disait : « Hélas ! non, je ne peux pas aller aujourd'hui à Pierrefonds, Odette y est justement. » Et Swann était heureux malgré tout de sentir que, si seul de tous les mortels il n'avait pas le droit en ce jour d'aller à Pierrefonds, c'était parce qu'il était en effet pour Odette quelqu'un de différent des autres, son amant, et que cette restriction apportée pour lui au droit universel de libre circulation, n'était qu'une des formes de cet esclavage, de cet amour qui lui était si cher. Décidément il valait mieux ne pas risquer de se brouiller avec elle, patienter, attendre son retour. Il passait ses journées penché sur une carte de la forêt de Compiègne comme si ç'avait été la carte du Tendre, s'entourait de photographies du château de Pierrefonds. Dès que venait le jour où il était possible qu'elle revînt, il rouvrait l'indicateur, calculait quel train elle avait dû prendre, et si elle s'était attardée, ceux qui lui restaient encore. Il ne sortait pas de peur de manquer une dépêche, ne se couchait pas, pour le cas où, revenue par le dernier train, elle aurait voulu lui faire la surprise de venir le voir au milieu de la nuit. Justement il entendait sonner à la porte cochère, il lui semblait qu'on tardait à ouvrir, il voulait éveiller le concierge, se mettait à la fenêtre pour appeler Odette si c'était elle, car malgré les recommandations qu'il était descendu faire plus de dix fois lui-même, on était capable de lui dire qu'il n'était pas là. C'était un domestique qui rentrait. Il remarquait le vol incessant des voitures qui passaient, auquel il n'avait jamais fait attention autrefois. Il écoutait chacune venir au loin, s'approcher, dépasser sa porte sans s'être arrêtée et porter plus loin un message qui n'était pas pour lui. Il attendait toute la nuit, bien inutilement, car les Verdurin ayant avancé leur retour, Odette était à Paris depuis midi ; elle n'avait pas eu l'idée de l'en prévenir ; ne sachant que faire, elle avait été passer sa soirée seule au théâtre et il y avait longtemps qu'elle était rentrée se coucher et dormait.
