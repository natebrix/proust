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
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« le bonheur, le bonheur par Gilberte »; « je l’aimais déjà tant que toutes les cinq minutes il me fallait la relire, l’embrasser. Alors, je connus mon bonheur. »",
      "explanation": "The narrator explicitly frames Gilberte as the source of his newly realized happiness and shows intense affection by rereading and kissing her letter, locally elevating her through his admiration."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Gilberte’s standing is locally enhanced because the narrator confers strong value and affection on her as the bearer of his 'bonheur'."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-121-p-124"
}

### Candidate characters

[
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Françoise s'approchait tous les jours de moi en me disant : « Monsieur a une mine ! Vous ne vous êtes pas regardé, on dirait un mort ! » Il est vrai que si j'avais eu un simple rhume, Françoise eût pris le même air funèbre. Ces déplorations tenaient plus à sa « classe » qu'à mon état de santé. Je ne démêlais pas alors si ce pessimisme était chez Françoise douloureux ou satisfait. Je conclus provisoirement qu'il était social et professionnel.

### Passage

Un jour, à l'heure du courrier, ma mère posa sur mon lit une lettre. Je l'ouvris distraitement puisqu'elle ne pouvait pas porter la seule signature qui m'eût rendu heureux, celle de Gilberte avec qui je n'avais pas de relations en dehors des Champs-Élysées. Or, au bas du papier, timbré d'un sceau d'argent représentant un chevalier casqué sous lequel se contournait cette devise : Per viam rectam, au-dessous d'une lettre, d'une grande écriture, et où presque toutes les phrases semblaient soulignées, simplement parce que la barre des t étant tracée non au travers d'eux, mais au-dessus, mettait un trait sous le mot correspondant de la ligne supérieure, ce fut justement la signature de Gilberte que je vis. Mais parce que je la savais impossible dans une lettre adressée à moi, cette vue, non accompagnée de croyance, ne me causa pas de joie. Pendant un instant elle ne fit que frapper d'irréalité tout ce qui m'entourait. Avec une vitesse vertigineuse, cette signature sans vraisemblance jouait aux quatre coins avec mon lit, ma cheminée, mon mur. Je voyais tout vaciller comme quelqu'un qui tombe de cheval et je me demandais s'il n'y avait pas une existence toute différente de celle que je connaissais, en contradiction avec elle, mais qui serait la vraie, et qui m'étant montrée tout d'un coup me remplissait de cette hésitation que les sculpteurs dépeignant le Jugement dernier ont donnée aux morts réveillés qui se trouvent au seuil de l'autre Monde. « Mon cher ami, disait la lettre, j'ai appris que vous aviez été très souffrant et que vous ne veniez plus aux Champs-Élysées. Moi je n'y vais guère non plus parce qu'il y a énormément de malades. Mais mes amies viennent goûter tous les lundis et vendredis à la maison. Maman me charge de vous dire que vous nous feriez très grand plaisir en venant aussi dès que vous serez rétabli, et nous pourrions reprendre à la maison nos bonnes causeries des Champs-Élysées. Adieu, mon cher ami, j'espère que vos parents vous permettront de venir très souvent goûter, et je vous envoie toutes mes amitiés. Gilberte. »

Tandis que je lisais ces mots, mon système nerveux recevait avec une diligence admirable la nouvelle qu'il m'arrivait un grand bonheur. Mais mon âme, c'est-à-dire moi-même, et en somme le principal intéressé, l'ignorait encore. Le bonheur, le bonheur par Gilberte, c'était une chose à laquelle j'avais constamment songé, une chose toute en pensées, c'était, comme disait Léonard, de la peinture, cosa mentale. Une feuille de papier couverte de caractères, la pensée ne s'assimile pas cela tout de suite. Mais dès que j'eus terminé la lettre, je pensai à elle, elle devint un objet de rêverie, elle devint, elle aussi, cosa mentale et je l'aimais déjà tant que toutes les cinq minutes il me fallait la relire, l'embrasser. Alors, je connus mon bonheur.

La vie est semée de ces miracles que peuvent toujours espérer les personnes qui aiment. Il est possible que celui-ci eût été provoqué artificiellement par ma mère qui, voyant que depuis quelque temps j'avais perdu tout coeur à vivre, avait peut-être fait demander à Gilberte de m'écrire, comme, au temps de mes premiers bains de mer, pour me donner du plaisir à plonger, ce que je détestais parce que cela me coupait la respiration, elle remettait en cachette à mon guide baigneur de merveilleuses boîtes en coquillages et des branches de corail que je croyais trouver moi-même au fond des eaux. D'ailleurs, pour tous les événements qui dans la vie et ses situations contrastées, se rapportent à l'amour, le mieux est de ne pas essayer de comprendre, puisque, dans ce qu'ils ont d'inexorable, comme d'inespéré, ils semblent régis par des lois plutôt magiques que rationnelles. Quand un multimillionnaire, homme malgré cela charmant, reçoit son congé d'une femme pauvre et sans agrément avec qui il vit, appelle à lui, dans son désespoir, toutes les puissances de l'or et fait jouer toutes les influences de la terre, sans réussir à se faire reprendre, mieux vaut devant l'invincible entêtement de sa maîtresse supposer que le Destin veut l'accabler et le faire mourir d'une maladie de coeur plutôt que de chercher une explication logique. Ces obstacles contre lesquels les amants ont à lutter et que leur imagination surexcitée par la souffrance cherche en vain à deviner, résident parfois dans quelque singularité de caractère de la femme qu'ils ne peuvent ramener à eux, dans sa bêtise, dans l'influence qu'ont prise sur elle et les craintes que lui ont suggérées des êtres que l'amant ne connaît pas, dans le genre de plaisirs qu'elle demande momentanément à la vie, plaisirs que son amant, ni la fortune de son amant ne peuvent lui offrir. En tous cas l'amant est mal placé pour connaître la nature des obstacles que la ruse de la femme lui cache et que son propre jugement faussé par l'amour l'empêche d'apprécier exactement. Ils ressemblent à ces tumeurs que le médecin finit par réduire mais sans en avoir connu l'origine. Comme elles ces obstacles restent mystérieux mais sont temporaires. Seulement ils durent généralement plus que l'amour. Et comme celui-ci n'est pas une passion désintéressée, l'amoureux qui n'aime plus ne cherche pas à savoir pourquoi la femme pauvre et légère, qu'il aimait, s'est obstinément refusée pendant des années à ce qu'il continuât à l'entretenir.

Or, le même mystère qui dérobe aux yeux souvent la cause des catastrophes, quand il s'agit de l'amour, entoure, tout aussi fréquemment la soudaineté de certaines solutions heureuses (telle que celle qui m'était apportée par la lettre de Gilberte). Solutions heureuses ou du moins qui paraissent l'être, car il n'y en a guère qui le soient réellement quand il s'agit d'un sentiment d'une telle sorte que toute satisfaction qu'on lui donne ne fait généralement que déplacer la douleur. Parfois pourtant une trêve est accordée et l'on a pendant quelque temps l'illusion d'être guéri.
