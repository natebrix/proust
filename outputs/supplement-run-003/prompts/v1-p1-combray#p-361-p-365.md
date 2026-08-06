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
      "canonical_name": "duchesse de Guermantes",
      "surface_forms": [
        "duchesse de Guermantes"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "duchesse de Guermantes",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« Qu'elle est belle ! Quelle noblesse ! Comme c'est bien une fière Guermantes, la descendante de Geneviève, que j'ai devant moi ! » … « l'infériorité proclamait trop sa suprématie » … « un sourire un peu timide de suzeraine »",
      "explanation": "The narrator isolates the duchess, surrounds her with images of hereditary nobility and social supremacy, and interprets her smile as a sovereign grace addressed to the people of Combray. These notes elevate her clearly above the assembly."
    }
  ],
  "status_effects": [
    {
      "character": "duchesse de Guermantes",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Locally, she is presented as socially supreme and benevolent, which increases her stature in the scene."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-361-p-365"
}

### Candidate characters

[
  "Geneviève",
  "Gilberte",
  "M. Vinteuil",
  "docteur Cottard",
  "le grand-père du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Mais en même temps, sur cette image que le nez proéminent, les yeux perçants, épinglaient dans ma vision (peut-être parce que c'était eux qui l'avaient d'abord atteinte, qui y avaient fait la première encoche, au moment où je n'avais pas encore le temps de songer que la femme qui apparaissait devant moi pouvait être duchesse de Guermantes), sur cette image toute récente, inchangeable, j'essayais d'appliquer l'idée : « C'est duchesse de Guermantes » sans parvenir qu'à la faire manoeuvrer en face de l'image, comme deux disques séparés par un intervalle. Mais cette duchesse de Guermantes à laquelle j'avais si souvent rêvé, maintenant que je voyais qu'elle existait effectivement en dehors de moi, en prit plus de puissance encore sur mon imagination qui, un moment paralysée au contact d'une réalité si différente de ce qu'elle attendait, se mit à réagir et à me dire : « Glorieux dès avant Charlemagne, les Guermantes avaient le droit de vie et de mort sur leurs vassaux ; la duchesse de Guermantes descend de Geneviève. Elle ne connaît, ni ne consentirait à connaître aucune des personnes qui sont ici. »

### Passage

Et – ô merveilleuse indépendance des regards humains, retenus au visage par une corde si lâche, si longue, si extensible qu'ils peuvent se promener seuls loin de lui – pendant que Mme de Guermantes était assise dans la chapelle au-dessus des tombes de ses morts, ses regards flânaient çà et là, montaient le long des piliers, s'arrêtaient même sur moi comme un rayon de soleil errant dans la nef, mais un rayon de soleil qui, au moment où je reçus sa caresse, me sembla conscient. Quant à Mme de Guermantes elle-même, comme elle restait immobile, assise comme une mère qui semble ne pas voir les audaces espiègles et les entreprises indiscrètes de ses enfants qui jouent et interpellent des personnes qu'elle ne connaît pas, il me fut impossible de savoir si elle approuvait ou blâmait, dans le désoeuvrement de son âme, le vagabondage de ses regards.

Je trouvais important qu'elle ne partît pas avant que j'eusse pu la regarder suffisamment, car je me rappelais que depuis des années je considérais sa vue comme éminemment désirable, et je ne détachais pas mes yeux d'elle, comme si chacun de mes regards eût pu matériellement emporter et mettre en réserve en moi le souvenir du nez proéminent, des joues rouges, de toutes ces particularités qui me semblaient autant de renseignements précieux, authentiques et singuliers sur son visage. Maintenant que me le faisaient trouver beau toutes les pensées que j'y rapportais – et peut-être surtout, forme de l'instinct de conservation des meilleures parties de nous-mêmes, ce désir qu'on a toujours de ne pas avoir été déçu – la replaçant (puisque c'était une seule personne qu'elle et cette Mme de Guermantes que j'avais évoquée jusque-là) hors du reste de l'humanité dans laquelle la vue pure et simple de son corps me l'avait fait un instant confondre, je m'irritais en entendant dire autour de moi : « Elle est mieux que Mme Sazerat, que Mlle Vinteuil », comme si elle leur eût été comparable. Et mes regards s'arrêtant à ses cheveux blonds, à ses yeux bleus, à l'attache de son cou et omettant les traits qui eussent pu me rappeler d'autres visages, je m'écriais devant ce croquis volontairement incomplet : « Qu'elle est belle ! Quelle noblesse ! Comme c'est bien une fière Guermantes, la descendante de Geneviève, que j'ai devant moi ! » Et l'attention avec laquelle j'éclairais son visage l'isolait tellement, qu'aujourd'hui si je repense à cette cérémonie, il m'est impossible de revoir une seule des personnes qui y assistaient sauf elle et le suisse qui répondit affirmativement quand je lui demandai si cette dame était bien Mme de Guermantes. Mais elle, je la revois, surtout au moment du défilé dans la sacristie qu'éclairait le soleil intermittent et chaud d'un jour de vent et d'orage, et dans laquelle Mme de Guermantes se trouvait au milieu de tous ces gens de Combray dont elle ne savait même pas les noms, mais dont l'infériorité proclamait trop sa suprématie pour qu'elle ne ressentît pas pour eux une sincère bienveillance, et auxquels du reste elle espérait imposer davantage encore à force de bonne grâce et de simplicité. Aussi, ne pouvant émettre ces regards volontaires, chargés d'une signification précise, qu'on adresse à quelqu'un qu'on connaît, mais seulement laisser ses pensées distraites s'échapper incessamment devant elle en un flot de lumière bleue qu'elle ne pouvait contenir, elle ne voulait pas qu'il pût gêner, paraître dédaigner ces petites gens qu'il rencontrait au passage, qu'il atteignait à tous moments. Je revois encore, au-dessus de sa cravate mauve, soyeuse et gonflée, le doux étonnement de ses yeux auxquels elle avait ajouté sans oser le destiner à personne, mais pour que tous pussent en prendre leur part, un sourire un peu timide de suzeraine qui a l'air de s'excuser auprès de ses vassaux et de les aimer. Ce sourire tomba sur moi qui ne la quittais pas des yeux. Alors me rappelant ce regard qu'elle avait laissé s'arrêter sur moi, pendant la messe, bleu comme un rayon de soleil qui aurait traversé le vitrail de Gilbert le Mauvais, je me dis : « Mais sans doute elle fait attention à moi. » Je crus que je lui plaisais, qu'elle penserait encore à moi quand elle aurait quitté l'église, qu'à cause de moi elle serait peut-être triste le soir à Guermantes. Et aussitôt je l'aimai, car s'il peut quelquefois suffire pour que nous aimions une femme qu'elle nous regarde avec mépris comme j'avais cru qu'avait fait Gilberte et que nous pensions qu'elle ne pourra jamais nous appartenir, quelquefois aussi il peut suffire qu'elle nous regarde avec bonté comme faisait Mme de Guermantes et que nous pensions qu'elle pourra nous appartenir. Ses yeux bleuissaient comme une pervenche impossible à cueillir et que pourtant elle m'eût dédiée ; et le soleil menacé par un nuage mais dardant encore de toute sa force sur la place et dans la sacristie, donnait une carnation de géranium aux tapis rouges qu'on y avait étendus par terre pour la solennité, et sur lesquels s'avançait en souriant Mme de Guermantes, et ajoutait à leur lainage un velouté rose, un épiderme de lumière, cette sorte de tendresse, de sérieuse douceur dans la pompe et dans la joie qui caractérisent certaines pages de Lohengrin, certaines peintures de Carpaccio, et qui font comprendre que Baudelaire ait pu appliquer au son de la trompette l'épithète de délicieux.

Combien depuis ce jour, dans mes promenades du côté de Guermantes, il me parut plus affligeant encore qu'auparavant de n'avoir pas de dispositions pour les lettres, et de devoir renoncer à être jamais un écrivain célèbre. Les regrets que j'en éprouvais, tandis que je restais seul à rêver un peu à l'écart, me faisaient tant souffrir, que pour ne plus les ressentir, de lui-même par une sorte d'inhibition devant la douleur, mon esprit s'arrêtait entièrement de penser aux vers, aux romans, à un avenir poétique sur lequel mon manque de talent m'interdisait de compter. Alors, bien en dehors de toutes ces préoccupations littéraires et ne s'y rattachant en rien, tout d'un coup un toit, un reflet de soleil sur une pierre, l'odeur d'un chemin me faisaient arrêter par un plaisir particulier qu'ils me donnaient, et aussi parce qu'ils avaient l'air de cacher au delà de ce que je voyais, quelque chose qu'ils m'invitaient à venir prendre et que malgré mes efforts je n'arrivais pas à découvrir. Comme je sentais que cela se trouvait en eux, je restais là, immobile, à regarder, à respirer, à tâcher d'aller avec ma pensée au delà de l'image ou de l'odeur. Et s'il me fallait rattraper mon grand-père, poursuivre ma route, je cherchais à les retrouver, en fermant les yeux ; je m'attachais à me rappeler exactement la ligne du toit, la nuance de la pierre qui, sans que je pusse comprendre pourquoi, m'avaient semblé pleines, prêtes à s'entr'ouvrir, à me livrer ce dont elles n'étaient qu'un couvercle. Certes ce n'était pas des impressions de ce genre qui pouvaient me rendre l'espérance que j'avais perdue de pouvoir être un jour écrivain et poète, car elles étaient toujours liées à un objet particulier dépourvu de valeur intellectuelle et ne se rapportant à aucune vérité abstraite. Mais du moins elles me donnaient un plaisir irraisonné, l'illusion d'une sorte de fécondité et par là me distrayaient de l'ennui, du sentiment de mon impuissance que j'avais éprouvés chaque fois que j'avais cherché un sujet philosophique pour une grande oeuvre littéraire. Mais le devoir de conscience était si ardu – que m'imposaient ces impressions de forme, de parfum ou de couleur – de tâcher d'apercevoir ce qui se cachait derrière elles, que je ne tardais pas à me chercher à moi-même des excuses qui me permissent de me dérober à ces efforts et de m'épargner cette fatigue. Par bonheur mes parents m'appelaient, je sentais que je n'avais pas présentement la tranquillité nécessaire pour poursuivre utilement ma recherche, et qu'il valait mieux n'y plus penser jusqu'à ce que je fusse rentré, et ne pas me fatiguer d'avance sans résultat. Alors je ne m'occupais plus de cette chose inconnue qui s'enveloppait d'une forme ou d'un parfum, bien tranquille puisque je la ramenais à la maison, protégée par le revêtement d'images sous lesquelles je la trouverais vivante, comme les poissons que, les jours où on m'avait laissé aller à la pêche, je rapportais dans mon panier, couverts par une couche d'herbe qui préservait leur fraîcheur. Une fois à la maison je songeais à autre chose et ainsi s'entassaient dans mon esprit (comme dans ma chambre les fleurs que j'avais cueillies dans mes promenades ou les objets qu'on m'avait donnés), une pierre où jouait un reflet, un toit, un son de cloche, une odeur de feuilles, bien des images différentes sous lesquelles il y a longtemps qu'est morte la réalité pressentie que je n'ai pas eu assez de volonté pour arriver à découvrir. Une fois pourtant – où notre promenade s'étant prolongée fort au delà de sa durée habituelle, nous avions été bien heureux de rencontrer à mi-chemin du retour, comme l'après-midi finissait, le docteur Percepied qui passait en voiture à bride abattue, nous avait reconnus et fait monter avec lui – j'eus une impression de ce genre et ne l'abandonnai pas sans un peu l'approfondir. On m'avait fait monter près du cocher, nous allions comme le vent parce que le docteur avait encore avant de rentrer à Combray à s'arrêter à Martinville-le-Sec chez un malade à la porte duquel il avait été convenu que nous l'attendrions. Au tournant d'un chemin j'éprouvai tout à coup ce plaisir spécial qui ne ressemblait à aucun autre, à apercevoir les deux clochers de Martinville, sur lesquels donnait le soleil couchant et que le mouvement de notre voiture et les lacets du chemin avaient l'air de faire changer de place, puis celui de Vieuxvicq qui, séparé d'eux par une colline et une vallée, et situé sur un plateau plus élevé dans le lointain, semblait pourtant tout voisin d'eux.

En constatant, en notant la forme de leur flèche, le déplacement de leurs lignes, l'ensoleillement de leur surface, je sentais que je n'allais pas au bout de mon impression, que quelque chose était derrière ce mouvement, derrière cette clarté, quelque chose qu'ils semblaient contenir et dérober à la fois.

Les clochers paraissaient si éloignés et nous avions l'air de si peu nous rapprocher d'eux, que je fus étonné quand, quelques instants après, nous nous arrêtâmes devant l'église de Martinville. Je ne savais pas la raison du plaisir que j'avais eu à les apercevoir à l'horizon et l'obligation de chercher à découvrir cette raison me semblait bien pénible ; j'avais envie de garder en réserve dans ma tête ces lignes remuantes au soleil et de n'y plus penser maintenant. Et il est probable que si je l'avais fait, les deux clochers seraient allés à jamais rejoindre tant d'arbres, de toits, de parfums, de sons, que j'avais distingués des autres à cause de ce plaisir obscur qu'ils m'avaient procuré et que je n'ai jamais approfondi. Je descendis causer avec mes parents en attendant le docteur. Puis nous repartîmes, je repris ma place sur le siège, je tournai la tête pour voir encore les clochers qu'un peu plus tard j'aperçus une dernière fois au tournant d'un chemin. Le cocher, qui ne semblait pas disposé à causer, ayant à peine répondu à mes propos, force me fut, faute d'autre compagnie, de me rabattre sur celle de moi-même et d'essayer de me rappeler mes clochers. Bientôt, leurs lignes et leurs surfaces ensoleillées, comme si elles avaient été une sorte d'écorce, se déchirèrent, un peu de ce qui m'était caché en elles m'apparut, j'eus une pensée qui n'existait pas pour moi l'instant avant, qui se formula en mots dans ma tête, et le plaisir que m'avait fait tout à l'heure éprouver leur vue s'en trouva tellement accru que, pris d'une sorte d'ivresse, je ne pus plus penser à autre chose. À ce moment et comme nous étions déjà loin de Martinville, en tournant la tête je les aperçus de nouveau, tout noirs cette fois, car le soleil était déjà couché. Par moments les tournants du chemin me les dérobaient, puis ils se montrèrent une dernière fois et enfin je ne les vis plus.
