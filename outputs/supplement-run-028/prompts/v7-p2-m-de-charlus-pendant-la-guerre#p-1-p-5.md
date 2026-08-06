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
      "canonical_name": "Elstir",
      "surface_forms": [
        "Elstir",
        "« Monsieur Tiche »"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Elstir",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« la prétentieuse vulgarité d'un Elstir à ses débuts » … « avant d'arriver … à un bon goût supérieur »; « quel est l'homme de génie qui n'a pas adopté les irritantes façons de parler … »",
      "explanation": "The narrator reframes Elstir’s earlier vulgarity as a stage en route to a 'bon goût supérieur,' explicitly grouping him with 'hommes de génie' and affirming his value despite prior social irritations."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Elstir’s standing rises as his earlier faults are reframed as part of a genius’s trajectory toward superior taste."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-1-p-5"
}

### Candidate characters

[
  "Bergotte",
  "Brichot",
  "M. Verdurin",
  "M. Vinteuil",
  "Swann",
  "duc de Guermantes",
  "duchesse de Guermantes",
  "la grand-mère",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

(none provided)

### Passage

Cette disposition-là, les pages de Goncourt que je lus me la firent regretter. Car peut-être j'aurais pu conclure d'elles que la vie apprend à rabaisser le prix de la lecture, et nous montre que ce que l'écrivain nous vante ne valait pas grand'chose ; mais je pouvais tout aussi bien en conclure que la lecture, au contraire, nous apprend à relever la valeur de la vie, valeur que nous n'avons pas su apprécier et dont nous nous rendons compte seulement par le livre combien elle était grande. À la rigueur, nous pouvons nous consoler de nous être peu plu dans la société d'un Vinteuil, d'un Bergotte, puisque le bourgeoisisme pudibond de l'un, les défauts insupportables de l'autre ne prouvent rien contre eux, puisque leur génie est manifesté par leurs oeuvres ; de même la prétentieuse vulgarité d'un Elstir à ses débuts. Ainsi le journal des Goncourt m'avait fait découvrir qu'Elstir n'était autre que le « Monsieur Tiche » qui avait tenu jadis de si exaspérants discours à Swann, chez les Verdurin. Mais quel est l'homme de génie qui n'a pas adopté les irritantes façons de parler des artistes de sa bande, avant d'arriver (comme c'était venu pour Elstir et comme cela arrive rarement) à un bon goût supérieur. Les lettres de Balzac, par exemple, ne sont-elles pas semées de termes vulgaires que Swann eût souffert mille morts d'employer ? Et cependant il est probable que Swann, si fin, si purgé de tout ridicule haïssable, eût été incapable d'écrire la Cousine Bette et le Curé de Tours. Que ce soit donc les Mémoires qui aient tort de donner du charme à leur société alors qu'elle nous a déplu est un problème de peu d'importance, puisque, même si c'est l'écrivain de Mémoires qui se trompe, cela ne prouve rien contre la valeur de la vie qui produit de tels génies et qui n'existait pas moins dans les oeuvres de Vinteuil, d'Elstir et de Bergotte.

Tout à l'autre extrémité de l'expérience, quand je voyais que les plus curieuses anecdotes, qui font la matière inépuisable, divertissement des soirées solitaires pour le lecteur, du journal des Goncourt, lui avaient été contées par ces convives que nous eussions à travers ces pages envié de connaître et qui ne m'avaient pas laissé à moi trace d'un souvenir intéressant, cela n'était pas trop inexplicable encore. Malgré la naïveté de Goncourt, qui concluait de l'intérêt de ces anecdotes à la distinction probable de l'homme qui les contait, il pouvait très bien se faire que des hommes médiocres eussent eu dans leur vie, ou entendu raconter, des choses curieuses et les contassent à leur tour. Goncourt savait écouter, comme il savait voir ; je ne le savais pas. D'ailleurs, tous ces faits auraient eu besoin d'être jugés un à un duc de Guermantes  ne m'avait certes pas donné l'impression de cet adorable modèle des grâces juvéniles que ma grand'mère eût tant voulu connaître et me proposait comme modèle inimitable d'après les Mémoires de Mme de Beausergent. Mais il faut songer que duc de Guermantes avait alors sept ans, que l'écrivain était sa tante, et que même les maris qui doivent divorcer quelques mois après vous font un grand éloge de leur femme. Une des plus jolies poésies de Sainte-Beuve est consacrée à l'apparition devant une fontaine d'une jeune enfant couronnée de tous les dons et de toutes les grâces, la jeune Mlle de Champlâtreux, qui ne devait pas avoir alors dix ans. Malgré toute la tendre vénération que le poète de génie qu'est la comtesse de Noailles portait à sa belle-mère, la duchesse de Noailles, née Champlâtreux, il est possible, si elle avait eu à en faire le portrait, que celui-ci eût contrasté assez vivement avec celui que Sainte-Beuve en traçait cinquante ans plus tôt.

Ce qui eût peut-être été plus troublant, c'était l'entre-deux, c'étaient ces gens desquels ce qu'on dit implique, chez eux, plus que la mémoire qui a su retenir une anecdote curieuse, sans que pourtant on ait, comme pour les Vinteuil, les Bergotte, le recours de les juger sur leur oeuvre ; ils n'en ont pas créé, ils en ont seulement – à notre grand étonnement à nous qui les trouvions si médiocres – inspiré. Passe encore que le salon qui, dans les musées, donnera la plus grande impression d'élégance, depuis les grandes peintures de la Renaissance, soit celui de la petite bourgeoise ridicule que j'eusse, si je ne l'avais pas connue, rêvé devant le tableau de pouvoir approcher dans la réalité, espérant apprendre d'elle les secrets les plus précieux que l'art du peintre, que sa toile ne me donnaient pas et de qui la pompeuse traîne de velours et de dentelles est un morceau de peinture comparable aux plus beaux du Titien. Si j'avais compris jadis que ce n'est pas le plus spirituel, le plus instruit, le mieux relationné des hommes, mais celui qui sait devenir miroir et peut refléter ainsi sa vie, fût-elle médiocre, qui devient un Bergotte (les contemporains le tinssent-ils pour moins homme d'esprit que Swann et moins savant que Brichot), on peut souvent à plus forte raison en dire autant des modèles de l'artiste. Dans l'éveil de l'amour de la beauté, chez l'artiste, qui peut tout peindre, de l'élégance où il pourra trouver de si beaux motifs, le modèle lui sera fourni par des gens un peu plus riches que lui, chez qui il trouvera ce qu'il n'a pas d'habitude dans son atelier d'homme de génie méconnu qui vend ses toiles cinquante francs, un salon avec des meubles recouverts de vieille soie, beaucoup de lampes, de belles fleurs, de beaux fruits, de belles robes – gens modestes relativement, ou qui le paraîtraient à des gens vraiment brillants (qui ne connaissent même pas leur existence), mais qui, à cause de cela, sont plus à portée de connaître l'artiste obscur, de l'apprécier, de l'inviter, de lui acheter ses toiles, que les gens de l'aristocratie qui se font peindre, comme le Pape et les chefs d'État, par les peintres académiciens. La poésie d'un élégant foyer et des belles toilettes de notre temps ne se trouvera-t-elle pas plutôt, pour la postérité, dans le salon de l'éditeur Charpentier par Renoir que dans le portrait de la princesse de Sagan ou de la comtesse de la Rochefoucauld par Cotte ou Chaplin ? Les artistes qui nous ont donné les plus grandes visions d'élégance en ont recueilli les éléments chez des gens qui étaient rarement les grands élégants de leur époque, lesquels se font rarement peindre par l'inconnu porteur d'une beauté qu'ils ne peuvent pas distinguer sur ses toiles, dissimulée qu'elle est par l'interposition d'un poncif de grâce surannée qui flotte dans l'oeil du public comme ces visions subjectives que le malade croit effectivement posées devant lui. Mais que ces modèles médiocres que j'avais connus eussent en outre inspiré, conseillé certains arrangements qui m'avaient enchanté, que la présence de tel d'entre eux dans les tableaux fût plus que celle d'un modèle, mais d'un ami qu'on veut faire figurer dans ses toiles, c'était à se demander si tous les gens que nous regrettons de ne pas avoir connus parce que Balzac les peignait dans ses livres ou les leur dédiait en hommage d'admiration, sur lesquels Sainte-Beuve ou Baudelaire firent leurs plus jolis vers, si, à plus forte raison, toutes les Récamier, toutes les Pompadour ne m'eussent pas paru d'insignifiantes personnes, soit par une infirmité de ma nature, ce qui me faisait alors enrager d'être malade et de ne pouvoir retourner voir tous les gens que j'avais méconnus, soit qu'elles ne dussent leur prestige qu'à une magie illusoire de la littérature, ce qui forçait à changer de dictionnaire pour lire et me consolait de devoir d'un jour à l'autre, à cause des progrès que faisait mon état maladif, rompre avec la société, renoncer au voyage, aux musées, pour aller me soigner dans une maison de santé. Peut-être, pourtant, ce côté mensonger, ce faux-jour n'existe-t-il dans les Mémoires que quand ils sont trop récents, trop près des réputations, qui plus tard s'anéantiront si vite, aussi bien intellectuelles que mondaines. (Et si l'érudition essaye alors de réagir contre cet ensevelissement, parvient-elle à détruire un sur mille de ces oublis qui vont s'entassant ?)

Ces idées, tendant, les unes à diminuer, les autres à accroître mon regret de ne pas avoir de dons pour la littérature, ne se présentèrent plus à ma pensée pendant les longues années que je passai à me soigner, loin de Paris, dans une maison de santé où, d'ailleurs, j'avais tout à fait renoncé au projet d'écrire, jusqu'à ce que celle-ci ne pût plus trouver de personnel médical, au commencement de 1916. Je rentrai alors dans un Paris bien différent de celui où j'étais déjà revenu une première fois, comme on le verra tout à l'heure, en août 1914, pour subir une visite médicale, après quoi j'avais rejoint ma maison de santé.

Chapitre II
