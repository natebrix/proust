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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "Monsieur l'Ambassadeur"
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
      "source": "Norpois",
      "target": "Odette",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.8,
      "evidence": "« Tout à fait excellente ! » ... « Elle est tout à fait charmante ! »; et, au sujet du Prince, « le Prince semblait donner assez volontiers à entendre que son impression était en somme loin d'avoir été défavorable. »",
      "explanation": "Norpois offers strong personal praise of Odette and reports a likely favorable impression from a high-status Prince, which locally elevates her standing even as the narrator frames Norpois’s delivery as a practiced conversational performance."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Odette’s local standing rises through Norpois’s explicit admiration and the hinted favorable view from a princely figure, signaling potential elite acceptability."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-65-p-72"
}

### Candidate characters

[
  "Swann",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Parmi les gens qui trouvaient ce genre de mariage ridicule, gens qui pour eux-mêmes se demandaient : « Que pensera duc de Guermantes , que dira marquis de Bréauté, quand j'épouserai Mlle de Montmorency ? », parmi les gens ayant cette sorte d'idéal social, aurait figuré, vingt ans plus tôt, Swann lui-même. Swann qui s'était donné du mal pour être reçu au Jockey et avait compté dans ce temps-là faire un éclatant mariage qui eût achevé, en consolidant sa situation, de faire de lui un des hommes les plus en vue de Paris. Seulement, les images que représentent un tel mariage à l'intéressé ont, comme toutes les images, pour ne pas dépérir et s'effacer complètement, besoin d'être alimentées du dehors. Votre rêve le plus ardent est d'humilier l'homme qui vous a offensé. Mais si vous n'entendez plus jamais parler de lui, ayant changé de pays, votre ennemi finira par ne plus avoir pour vous aucune importance. Si on a perdu de vue pendant vingt ans toutes les personnes à cause desquelles on aurait aimé entrer au Jockey ou à l'Institut, la perspective d'être membre de l'un ou de l'autre de ces groupements ne tentera nullement. Or, tout autant qu'une retraite, qu'une maladie, qu'une conversion religieuse, une liaison prolongée substitue d'autres images aux anciennes. Il n'y eut pas de la part de Swann, quand il épousa Odette, renoncement aux ambitions mondaines car de ces ambitions-là, depuis longtemps Odette l'avait, au sens spirituel du mot, détaché. D'ailleurs, ne l'eût-il pas été qu'il n'en aurait eu que plus de mérite. C'est parce qu'ils impliquent le sacrifice d'une situation plus ou moins flatteuse à une douceur purement intime, que généralement les mariages infamants sont les plus estimables de tous (on ne peut en effet entendre par mariage infamant un mariage d'argent, n'y ayant point d'exemple d'un ménage où la femme ou bien le mari se soient vendus et qu'on n'ait fini par recevoir, ne fût-ce que par tradition et sur la foi de tant d'exemples et pour ne pas avoir deux poids et deux mesures). Peut-être, d'autre part, en artiste, sinon en corrompu, Swann eût-il en tous cas éprouvé une certaine volupté à accoupler à lui, dans un de ces croisements d'espèces comme en pratiquent les mendelistes ou comme en raconte la mythologie, un être de race différente, archiduchesse ou cocotte, à contracter une alliance royale ou à faire une mésalliance. Il n'y avait eu dans le monde qu'une seule personne dont il se fût préoccupé, chaque fois qu'il avait pensé à son mariage possible avec Odette, c'était, et non par snobisme, la princesse des Laumes. De celle-là, au contraire, Odette se souciait peu, pensant seulement aux personnes situées immédiatement au-dessus d'elle-même plutôt que dans un aussi vague empyrée. Mais quand Swann dans ses heures de rêverie voyait Odette devenue sa femme, il se représentait invariablement le moment où il l'amènerait, elle et surtout sa fille, chez la princesse des Laumes, devenue bientôt la princesse des Laumes par la mort de son beau-père. Il ne désirait pas les présenter ailleurs, mais il s'attendrissait quand il inventait, en énonçant les mots eux-mêmes, tout ce que la duchesse dirait de lui à Odette, et Odette à princesse des Laumes, la tendresse que celle-ci témoignerait à Gilberte, la gâtant, le rendant fier de sa fille. Il se jouait à lui-même la scène de la présentation avec la même précision dans le détail imaginaire qu'ont les gens qui examinent comment ils emploieraient, s'ils gagnaient, un lot dont ils fixent arbitrairement le chiffre. Dans la mesure où une image qui accompagne une de nos résolutions la motive, on peut dire que si Swann épousa Odette, ce fut pour la présenter elle et Gilberte, sans qu'il y eût personne là, au besoin sans que personne le sût jamais, à la princesse des Laumes. On verra comment cette seule ambition mondaine qu'il avait souhaitée pour sa femme et sa fille fut justement celle dont la réalisation se trouva lui être interdite, et par un veto si absolu que Swann mourut sans supposer que la duchesse pourrait jamais les connaître. On verra aussi qu'au contraire la princesse des Laumes se lia avec Odette et Gilberte après la mort de Swann. Et peut-être eût-il été sage – pour autant qu'il pouvait attacher de l'importance à si peu de chose – en ne se faisant pas une idée trop sombre de l'avenir à cet égard, et en réservant que la réunion souhaitée pourrait bien avoir lieu quand il ne serait plus là pour en jouir. Le travail de causalité qui finit par produire à peu près tous les effets possibles, et par conséquent aussi ceux qu'on avait cru l'être le moins, ce travail est parfois lent, rendu un peu plus lent encore par notre désir – qui en cherchant à l'accélérer l'entrave – par notre existence même, et n'aboutit que quand nous avons cessé de désirer, et quelquefois de vivre. Swann ne le savait-il pas par sa propre expérience, et n'était-ce pas déjà, dans sa vie – comme une préfiguration de ce qui devait arriver après sa mort – un bonheur après décès que ce mariage avec cette Odette qu'il avait passionnément aimée – si elle ne lui avait pas plu au premier abord – et qu'il avait épousée quand il ne l'aimait plus, quand l'être qui, en Swann, avait tant souhaité et tant désespéré de vivre toute sa vie avec Odette, quand cet être-là était mort ?

### Passage

Je me mis à parler du comte de Paris, à demander s'il n'était pas ami de Swann, car je craignais que la conversation se détournât de celui-ci. « Oui, en effet, répondit Norpois en se tournant vers moi et en fixant sur ma modeste personne le regard bleu où flottaient, comme dans leur élément vital, ses grandes facultés de travail et son esprit d'assimilation. Et, mon Dieu, ajouta-t-il en s'adressant de nouveau à mon père, je ne crois pas franchir les bornes du respect dont je fais profession pour le Prince (sans cependant entretenir avec lui des relations personnelles que rendrait difficiles ma situation, si peu officielle qu'elle soit) en vous citant ce fait assez piquant que, pas plus tard qu'il y a quatre ans, dans une petite gare de chemins de fer d'un des pays de l'Europe Centrale, le Prince eut l'occasion d'apercevoir Odette. Certes, aucun de ses familiers ne s'est permis de demander à Monseigneur comment il l'avait trouvée. Cela n'eût pas été séant. Mais quand par hasard la conversation amenait son nom, à de certains signes, imperceptibles si l'on veut, mais qui ne trompent pas, le Prince semblait donner assez volontiers à entendre que son impression était en somme loin d'avoir été défavorable.

– Mais il n'y aurait pas eu possibilité de la présenter au comte de Paris ? demanda mon père.

– Eh bien ! on ne sait pas ; avec les princes on ne sait jamais, répondit Norpois ; les plus glorieux, ceux qui savent le plus se faire rendre ce qu'on leur doit, sont aussi quelquefois ceux qui s'embarrassent le moins des décrets de l'opinion publique, même les plus justifiés, pour peu qu'il s'agisse de récompenser certains attachements. Or, il est certain que le comte de Paris a toujours agréé avec beaucoup de bienveillance le dévouement de Swann qui est, d'ailleurs, un garçon d'esprit s'il en fut.

– Et votre impression à vous, quelle a-t-elle été, Monsieur l'Ambassadeur ? demanda ma mère par politesse et par curiosité.

Avec une énergie de vieux connaisseur, qui tranchait sur la modération habituelle de ses propos :

– Tout à fait excellente ! répondit Norpois.

Et sachant que l'aveu d'une forte sensation produite par une femme rentre, à condition qu'on le fasse avec enjouement, dans une certaine forme particulièrement appréciée de l'esprit de conversation, il éclata d'un petit rire qui se prolongea pendant quelques instants, humectant les yeux bleus du vieux diplomate et faisant vibrer les ailes de son nez nervurées de fibrilles rouges.

– Elle est tout à fait charmante !
