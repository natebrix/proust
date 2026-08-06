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
      "canonical_name": "duc de Guermantes",
      "surface_forms": [
        "duc de Guermantes",
        "le duc"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "duc de Guermantes",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.91,
      "evidence": "Face à « la dame en rose » qui lui tenait tête, « le vieux fauve dompté … rentrait dans ses épaules … et reprenait son récit »; plus loin, « c'est en un risible Géronte que se change inévitablement Jupiter ».",
      "explanation": "The passage highlights the loss of authority and the jealous senility of the duke: he is tamed in public by Odette, blind to his infidelities, and belittled by images of a caged beast and a Jupiter degraded to Géronte."
    }
  ],
  "status_effects": [
    {
      "character": "duc de Guermantes",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "He notably loses local stature, reduced to a 'tamed beast' and compared to a Géronte, in front of onlookers."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-106-p-110"
}

### Candidate characters

[
  "Albertine",
  "Gilberte",
  "Odette",
  "Robert de Saint-Loup",
  "Swann",
  "baron de Charlus",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "le narrateur",
  "marquis de Bréauté"
]

### Prior local context (optional)

Ne pouvant se passer d'Odette, toujours installé chez elle dans le même fauteuil d'où la vieillesse et la goutte le faisaient difficilement lever, duc de Guermantes  la laissait recevoir des amis qui étaient trop contents d'être présentés au duc, de lui laisser la parole, de l'entendre parler de la vieille société, de la Mme de Villeparisis, du duc de Chartres.

### Passage

Par moments, sous le regard des tableaux anciens réunis par Swann dans un arrangement de « collectionneur » qui achevait le caractère démodé de cette scène, avec ce duc si « Restauration » et cette cocotte tellement « Second Empire », dans un des peignoirs qu'il aimait, la dame en rose l'interrompait d'une jacasserie : il s'arrêtait net, plantait sur elle un regard féroce. Peut-être s'était-il aperçu qu'elle aussi, comme la duchesse, disait quelquefois des bêtises ; peut-être, dans une hallucination de vieillard, croyait-il que c'était un trait d'esprit intempestif de Mme de Guermantes qui lui coupait la parole, et se croyait-il à l'hôtel de Guermantes, comme ces fauves enchaînés qui se figurent un instant être encore libres dans les déserts de l'Afrique. Levant brusquement la tête, de ses petits yeux jaunes qui avaient l'éclat d'yeux de fauves il fixait sur elle un de ces regards qui quelquefois chez Mme de Guermantes, quand celle-ci parlait trop, m'avaient fait trembler. Ainsi le duc regardait-il un instant l'audacieuse dame en rose. Mais celle-ci lui tenait tête, ne le quittait pas des yeux, et au bout de quelques instants qui semblaient longs aux spectateurs, le vieux fauve dompté, se rappelant qu'il était, non pas libre chez la duchesse, dans ce Sahara dont le paillasson du palier marquait l'entrée, mais chez Mme de Forcheville, dans la cage du Jardin des Plantes, rentrait dans ses épaules sa tête d'où pendait encore une épaisse crinière dont on n'aurait pu dire si elle était blonde ou blanche, et reprenait son récit. Il semblait n'avoir pas compris ce que Mme de Forcheville avait voulu dire et qui, d'ailleurs, généralement n'avait pas grand sens. Il lui permettait d'avoir des amis à dîner avec lui. Par une manie empruntée à ses anciennes amours, qui n'était pas pour étonner Odette, habituée à avoir eu la même de Swann, et qui me touchait moi, en me rappelant ma vie avec Albertine, il exigeait que ces personnes se retirassent de bonne heure afin qu'il pût dire bonsoir à Odette le dernier. Inutile de dire qu'à peine était-il parti, elle allait en rejoindre d'autres. Mais le duc ne s'en doutait pas ou préférait ne pas avoir l'air de s'en douter ; la vue des vieillards baisse, comme leur oreille devient plus dure, leur clairvoyance s'obscurcit, la fatigue même fait faire relâche à leur vigilance. Et à un certain âge c'est en un personnage de Molière – non pas même en l'olympien amant d'Alcmène mais en un risible Géronte – que se change inévitablement Jupiter. D'ailleurs, Odette trompait duc de Guermantes , et aussi le soignait, sans charme, sans grandeur. Elle était médiocre dans ce rôle comme dans tous les autres. Non pas que la vie ne lui en eût souvent donné de beaux, mais elle ne savait pas les jouer. En attendant, elle jouait celui de recluse. De fait, chaque fois que je voulus la voir dans la suite je n'y pus réussir, car duc de Guermantes , voulant à la fois concilier les exigences de son hygiène et de sa jalousie, ne lui permettait que les fêtes de jour, à condition encore que ce ne fussent pas des bals. Cette réclusion où elle était tenue, elle me l'avoua avec franchise, pour diverses raisons. La principale est qu'elle s'imaginait, bien que je n'eusse écrit que des articles ou publié que des études, que j'étais un auteur connu, ce qui lui faisait même naïvement dire, se rappelant le temps où j'allais avenue des Acacias pour la voir passer, et plus tard chez elle : « Ah ! si j'avais pu deviner que ce petit serait un jour un grand écrivain ! » Or, ayant entendu dire que les écrivains se plaisent auprès des femmes pour se documenter, se faire raconter des histoires d'amour, elle redevenait maintenant avec moi simple cocotte pour m'intéresser : « Tenez, une fois il y avait un homme qui s'était toqué de moi et que j'aimais éperdument aussi. Nous vivions d'une vie divine. Il avait un voyage à faire en Amérique, je devais y aller avec lui. La veille du départ, je trouvai que c'était plus beau de ne pas laisser diminuer un amour qui ne pourrait pas toujours rester à ce point. Nous eûmes une dernière soirée où il était persuadé que je partais, ce fut une nuit folle, j'avais près de lui des joies infinies et le désespoir de sentir que je ne le reverrais pas. Le matin j'étais allée donner mon billet à un voyageur que je ne connaissais pas. Il voulait au moins l'acheter. Je lui répondis : « Non, vous me rendez un tel service en me le prenant, je ne veux pas d'argent. » Puis c'était une autre histoire : « Un jour j'étais dans les Champs-Élysées, M. de Bréauté, que je n'avais vu qu'une fois, se mit à me regarder avec une telle insistance que je m'arrêtai et lui demandai pourquoi il se permettait de me regarder comme ça. Il me répondit : « Je vous regarde parce que vous avez un chapeau ridicule. » C'était vrai. C'était un petit chapeau avec des pensées, les modes de ce temps-là étaient affreuses. Mais j'étais en fureur, je lui dis : « Je ne vous permets pas de me parler ainsi. » Il se mit à pleuvoir. Je lui dis : « Je ne vous pardonnerais que si vous aviez une voiture. – Hé bien, justement j'en ai une et je vais vous accompagner. – Non, je veux bien de votre voiture, mais pas de vous. » Je montai dans la voiture, il partit sous la pluie. Mais le soir il arriva chez moi. Nous eûmes deux années d'un amour fou. » Elle reprit : « Venez prendre une fois le thé avec moi, je vous raconterai comment j'ai fait la connaissance de M. de Forcheville. Au fond, dit-elle d'un air mélancolique, j'ai passé ma vie cloîtrée parce que je n'ai eu de grands amours que pour des hommes qui étaient terriblement jaloux de moi. Je ne parle pas de M. de Forcheville, car, au fond, c'était un médiocre et je n'ai jamais pu aimer véritablement que des gens intelligents. Mais, voyez-vous, Swann était aussi jaloux que l'est ce pauvre duc ; pour celui-ci je me prive de tout parce que je sais qu'il n'est pas heureux chez lui. Pour Swann, c'était parce que je l'aimais follement, et je trouve qu'on peut bien sacrifier la danse, et le monde, et tout le reste à ce qui peut faire plaisir ou seulement éviter des soucis à un homme qu'on aime. Pauvre Swann, il était si intelligent, si séduisant, exactement le genre d'hommes que j'aimais. » Et c'était peut-être vrai. Il y avait eu un temps où Swann lui avait plu, justement celui où elle n'était pas « son genre ». À vrai dire, « son genre », même plus tard, elle ne l'avait jamais été. Il l'avait pourtant alors tant et si douloureusement aimée. Il était surpris plus tard de cette contradiction. Elle ne doit pas en être une si nous songeons combien est forte dans la vie des hommes la proportion des souffrances pour des femmes « qui n'étaient pas leur genre ». Peut-être cela tient-il à bien des causes ; d'abord, parce qu'elles ne sont pas votre genre on se laisse d'abord aimer sans aimer, par là on laisse prendre sur sa vie une habitude qui n'aurait pas eu lieu avec une femme qui eût été votre genre et qui, se sentant désirée, se fût disputée, ne nous aurait accordé que de rares rendez-vous, n'eût pas pris dans notre vie cette installation dans toutes nos heures qui plus tard, si l'amour vient et qu'elle vienne à nous manquer, pour une brouille, pour un voyage où on nous laisse sans nouvelles, ne nous arrache pas un seul lien mais mille. Ensuite, cette habitude est sentimentale parce qu'il n'y a pas grand désir physique à la base, et si l'amour naît, le cerveau travaille bien davantage : il y a un roman au lieu d'un besoin. Nous ne nous méfions pas des femmes qui ne sont pas notre genre, nous les laissons nous aimer, et si nous les aimons ensuite, nous les aimons cent fois plus que les autres, sans avoir même près d'elles la satisfaction du désir assouvi. Pour ces raisons et bien d'autres, le fait que nous ayons nos plus gros chagrins avec les femmes qui ne sont pas notre genre ne tient pas seulement à cette dérision du destin qui ne réalise notre bonheur que sous la forme qui nous plaît le moins. Une femme qui est notre genre est rarement dangereuse, car ou elle ne veut pas de nous, ou nous contente et nous quitte vite, ne s'installe pas dans notre vie, et ce qui est dangereux et procréateur de souffrances dans l'amour, ce n'est pas la femme elle-même, c'est sa présence de tous les jours, la curiosité de ce qu'elle fait à tous moments ; ce n'est pas la femme, c'est l'habitude. J'eus la lâcheté d'ajouter que ce qu'elle disait de Swann était gentil et noble de sa part, mais je savais combien c'était faux et que sa franchise se mêlait de mensonges. Je pensais avec effroi, au fur et à mesure qu'elle me racontait ses aventures, à tout ce que Swann avait ignoré, dont il aurait tant souffert parce qu'il avait fixé sa sensibilité sur cet être-là, et qu'il devinait à en être sûr, rien qu'à ses regards quand elle voyait un homme ou une femme inconnus et qui lui plaisaient. Au fond, elle le faisait seulement pour me donner ce qu'elle croyait des sujets de nouvelles ! Elle se trompait, non qu'elle n'eût de tout temps abondamment fourni les réserves de mon imagination, mais d'une façon bien plus involontaire et par un acte émané de moi-même, qui dégageait d'elle à son insu les lois de sa vie.

duc de Guermantes  ne gardait ses foudres que pour la duchesse ; sur les libres fréquentations de laquelle Mme de Forcheville ne manquait pas d'attirer l'attention irritée du duc. Aussi la duchesse était-elle fort malheureuse. Il est vrai que Charlus, à qui j'en avais parlé une fois, prétendait que les premiers torts n'avaient pas été du côté de son frère, que la légende de pureté de la duchesse était faite, en réalité, d'un nombre incalculable d'aventures habilement dissimulées. Je n'avais jamais entendu parler de cela. Pour presque tout le monde Mme de Guermantes était une femme toute différente. L'idée qu'elle avait été toujours irréprochable gouvernait les esprits. Entre ces deux idées je ne pouvais décider laquelle était conforme à la vérité, cette vérité que presque toujours les trois quarts des gens ignorent. Je me rappelais bien certains regards bleus et vagabonds de la Mme de Guermantes dans la nef de Combray, mais, vraiment, aucune des deux idées n'était réfutée par eux, et l'une et l'autre pouvaient leur donner un sens différent et aussi acceptable. Dans ma folie, enfant, je les avais pris un instant pour des regards d'amour adressés à moi. Depuis j'avais compris qu'ils n'étaient que des regards bienveillants d'une suzeraine, pareille à celle des vitraux de l'église, pour ses vassaux. Fallait-il maintenant croire que c'était ma première idée qui avait été la vraie, et que si, plus tard, jamais la duchesse ne m'avait parlé d'amour, c'est parce qu'elle avait craint de se compromettre avec un ami de sa tante et de son neveu plus qu'avec un enfant inconnu rencontré par hasard à Saint-Hilaire de Combray ?

* * *

La duchesse avait pu un instant être heureuse de sentir son passé plus consistant parce qu'il était partagé par moi, mais à quelques questions que je lui posai à nouveau sur le provincialisme de M. de Bréauté, que j'avais à l'époque peu distingué de M. de Sagan, ou de duc de Guermantes , elle reprit son point de vue de femme du monde, c'est-à-dire de contemptrice de la mondanité. Tout en me parlant, la duchesse me faisait visiter l'Hôtel. Dans des salons plus petits on trouvait des intimes qui, pour écouter la musique, avaient préféré s'isoler. Dans un petit salon Empire, où quelques rares habits noirs écoutaient assis sur un canapé, on voyait, à côté d'une Psyché supportée par une Minerve, une chaise longue, placée de façon rectiligne, mais à l'intérieur incurvée comme un berceau, et où une jeune femme était étendue. La mollesse de sa pose, que l'entrée de la duchesse ne lui fit même pas déranger, contrastait avec l'éclat merveilleux de sa robe Empire en une soierie nacarat devant laquelle les plus rouges fuchsias eussent pâli et sur le tissu nacré de laquelle des insignes et des fleurs semblaient avoir été enfoncés longtemps, car leur trace y restait en creux. Pour saluer la duchesse elle inclina légèrement sa belle tête brune. Bien qu'il fît grand jour, comme elle avait demandé qu'on fermât les grands rideaux, en vue de plus de recueillement pour la musique, on avait, pour ne pas se tordre les pieds, allumé sur un trépied une urne où s'irisait une faible lueur. En réponse à ma demande, la Mme de Guermantes me dit que c'était Mme de Sainte-Euverte. Alors je voulus savoir ce qu'elle était à la madame de Sainte-Euverte que j'avais connue. Mme de Guermantes me dit que c'était la femme d'un de ses petits-neveux, parut supporter l'idée qu'elle était née La Rochefoucauld, mais nia avoir elle-même connu des Sainte-Euverte. Je lui rappelai la soirée, que je n'avais sue, il est vrai, que par ouï-dire, où princesse des Laumes, elle avait retrouvé Swann. Mme de Guermantes m'affirma n'avoir jamais été à cette soirée. La duchesse avait toujours été un peu menteuse et l'était devenue davantage. Mme de Sainte-Euverte était pour elle un salon – d'ailleurs assez tombé avec le temps – qu'elle aimait à renier. Je n'insistai pas. « Non, qui vous avez pu entrevoir chez moi, parce qu'il avait de l'esprit, c'est le mari de celle dont vous parlez et avec qui je n'étais pas en relations. – Mais elle n'avait pas de mari. – Vous vous l'êtes figuré parce qu'ils étaient séparés, mais il était bien plus agréable qu'elle. » Je finis par comprendre qu'un homme énorme, extrêmement grand, extrêmement fort, avec des cheveux tout blancs, que je rencontrais un peu partout et dont je n'avais jamais su le nom était le mari de Mme de Sainte-Euverte. Il était mort l'an passé. Quant à la nièce, j'ignore si c'est à cause d'une maladie d'estomac, de nerfs, d'une phlébite, d'un accouchement prochain, récent ou manqué, qu'elle écoutait la musique étendue sans se bouger pour personne. Le plus probable est que, fière de ses belles soies rouges, elle pensait faire sur sa chaise longue un effet genre Récamier. Elle ne se rendait pas compte qu'elle donnait pour moi la naissance à un nouvel épanouissement de ce nom Sainte-Euverte, qui à tant d'intervalle marquait la distance et la continuité du Temps. C'est le Temps qu'elle berçait dans cette nacelle où fleurissaient le nom de Sainte-Euverte et le style Empire en soie de fuchsias rouges. Ce style Empire, Mme de Guermantes déclarait l'avoir toujours détesté ; cela voulait dire qu'elle le détestait maintenant, ce qui était vrai, car elle suivait la mode, bien qu'avec quelque retard. Sans compliquer en parlant de David qu'elle connaissait peu, toute jeune fille elle avait cru M. Ingres le plus ennuyeux des poncifs, puis, brusquement, le plus savoureux des maîtres de l'Art nouveau, jusqu'à détester Delacroix. Par quels degrés elle était revenue de ce culte à la réprobation importe peu, puisque ce sont là des nuances des goûts que le critique d'art reflète dix ans avant la conversation des femmes supérieures. Après avoir critiqué le style Empire, elle s'excusa de m'avoir parlé de gens aussi insignifiants que les Sainte-Euverte et de niaiseries comme le côté provincial de Bréauté, car elle était aussi loin de penser pourquoi cela m'intéressait que Mme de Sainte-Euverte de La Rochefoucauld, cherchant le bien de son estomac ou un effet ingresque, était loin de soupçonner que son nom m'avait ravi, celui de son mari, non celui plus glorieux de ses parents, et que je lui voyais comme une fonction dans cette pièce pleine d'attributs de bercer le temps.

« Mais comment puis-je vous parler de ces sottises, comment cela peut-il vous intéresser ? » s'écria la duchesse. Elle avait dit cette phrase à mi-voix et personne n'avait pu entendre ce qu'elle disait. Mais un jeune homme (qui devait m'intéresser dans la suite par un nom bien plus familier de moi autrefois que celui de Sainte-Euverte) se leva d'un air exaspéré et alla plus loin pour écouter avec plus de recueillement. Car c'était la sonate à Kreutzer qu'on jouait, mais, s'étant trompé sur le programme, il croyait que c'était un morceau de Ravel qu'on lui avait déclaré être beau comme du Palestrina, mais difficile à comprendre. Dans sa violence à changer de place, il heurta, à cause de la demi-obscurité, un bonheur du jour, ce qui n'alla pas sans faire tourner la tête à beaucoup de personnes pour qui cet exercice si simple de regarder derrière soi interrompait un peu le supplice d'écouter « religieusement » la sonate à Kreutzer. Et Mme de Guermantes et moi, causes de ce petit scandale, nous nous hâtâmes de changer de pièce. « Oui, comment ces riens-là peuvent-ils intéresser un homme de votre mérite ? C'est comme tout à l'heure, quand je vous voyais causer avec Gilberte de Saint-Loup. Ce n'est pas digne de vous. Pour moi c'est exactement rien, cette femme-là, ce n'est même pas une femme, c'est ce que je connais de plus factice et de plus bourgeois au monde (car, même à sa défense de l'actualité, la duchesse mêlait ses préjugés d'aristocrate). D'ailleurs devriez-vous venir dans des maisons comme ici ? Aujourd'hui, encore, je comprends parce qu'il y avait cette récitation de Rachel, ça peut vous intéresser. Mais si belle qu'elle ait été, elle ne donne pas devant ce public-là. Je vous ferai déjeuner seule avec elle. Alors vous verrez l'être que c'est. Mais elle est cent fois supérieure à tout ce qui est ici. Et après déjeuner elle vous dira du Verlaine. Vous m'en direz des nouvelles. » Elle me vanta surtout ses après-déjeuners, où il y avait tous les jours X et Y. Car elle en était arrivée à cette conception des femmes à « salons » qu'elle méprisait autrefois (bien qu'elle le niât aujourd'hui) et dont la grande supériorité, le signe d'élection selon elle, étaient d'avoir chez elle « tous les hommes ». Si je lui disais que telle grande dame à « salons » ne disait pas du bien, quand elle vivait, de Mme Howland, la duchesse éclatait de rire devant ma naïveté : « Naturellement, l'autre avait chez elle tous les hommes et celle-ci cherchait à les attirer. » Elle reprit : « Mais dans de grandes machines comme ici, non, ça me passe que vous veniez. À moins que ce ne soit pour faire des études... », ajouta-t-elle d'un air de doute, de méfiance, et sans trop s'aventurer, car elle ne savait pas très exactement en quoi consistait le genre d'opérations improbables auquel elle faisait allusion.
