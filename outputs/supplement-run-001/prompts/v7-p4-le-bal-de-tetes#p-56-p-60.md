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
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "duchesse de Guermantes",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "« Et parce qu'il y avait au moins vingt ans qu'elle avait vu Bloch pour la première fois, duchesse de Guermantes eût juré qu'il était né dans son monde et avait été bercé sur les genoux de duchesse de Guermantes de Chartres quand il avait deux ans. »",
      "explanation": "As an example of how memory distorts over time, the narrator shows the duchesse confidently misremembering Bloch’s origins, which locally undermines her discernment and credibility."
    }
  ],
  "status_effects": [
    {
      "character": "duchesse de Guermantes",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Her reliability and social discernment are diminished by the narrator’s illustration of her faulty, self-serving memory."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-56-p-60"
}

### Candidate characters

[
  "Albertine",
  "Bergotte",
  "Bloch",
  "Elstir",
  "Françoise",
  "Gilberte",
  "Legrandin",
  "M. Verdurin",
  "Mme de Cambremer",
  "Morel",
  "Odette",
  "Robert de Saint-Loup",
  "Swann",
  "baron de Charlus",
  "docteur Cottard",
  "le narrateur",
  "oncle Adolphe"
]

### Prior local context (optional)

* * *

### Passage

Ainsi, à chacun des moments de sa durée, le nom de Guermantes, considéré comme un ensemble de tous les noms qu'il admettait en lui, autour de lui, subissait des déperditions, recrutait des éléments nouveaux, comme ces jardins où à tout moment des fleurs à peine en bouton et se préparant à remplacer celles qui se flétrissent déjà se confondent dans une masse qui semble pareille, sauf à ceux qui n'ont pas toujours vu les nouvelles venues et gardent dans leur souvenir l'image précise de celles qui ne sont plus.

Plus d'une des personnes que cette matinée réunissait, ou dont elle m'évoquait le souvenir, me donnait les aspects qu'elle avait tour à tour présentés pour moi, par les circonstances différentes, opposées, d'où elle avait, les unes après les autres, surgi devant moi, faisait ressortir les aspects variés de ma vie, les différences de perspective, comme un accident de terrain, de colline ou château, qui, apparaissant tantôt à droite, tantôt à gauche, semble d'abord dominer une forêt, ensuite sortir d'une vallée, et révéler ainsi au voyageur des changements d'orientation et des différences d'altitude dans la route qu'il suit. En remontant de plus en plus haut, je finissais par trouver des images d'une même personne séparées par un intervalle de temps si long, conservées par des « moi » si distincts, ayant elles-mêmes des significations si différentes, que je les omettais d'habitude quand je croyais embrasser le cours passé de mes relations avec elles, que j'avais même cessé de penser qu'elles étaient les mêmes que j'avais connues autrefois et qu'il me fallait le hasard d'un éclair d'attention pour les rattacher, comme à une étymologie, à cette signification primitive qu'elles avaient eue pour moi. Gilberte me jetait, de l'autre côté de la haie d'épines roses, un regard dont j'avais dû, d'ailleurs, rétrospectivement retoucher la signification, qui était du désir. L'amant de Odette, selon la chronique de Combray, me regardait derrière cette même haie d'un air dur qui n'avait pas non plus le sens que je lui avais donné alors, et ayant, d'ailleurs, tellement changé depuis, que je ne l'avais nullement reconnu à Balbec dans le Monsieur qui regardait une affiche, près du Casino, et dont il m'arrivait une fois tous les dix ans de me souvenir en me disant : « Mais c'était Charlus, déjà, comme c'est curieux. » Mme de Guermantes au mariage du Dr Percepied, Odette en rose chez mon grand-oncle, Mme de Cambremer, soeur de Legrandin, si élégante qu'il craignait que nous ne le priions de nous donner une recommandation pour elle, c'étaient, ainsi que tant d'autres concernant Swann, Saint-Loup, etc., autant d'images que je m'amusais parfois, quand je les retrouvais, à placer comme frontispice au seuil de mes relations avec ces différentes personnes, mais qui ne me semblaient, en effet, qu'une image, et non déposée en moi par l'être lui-même, auquel rien ne la reliait plus. Non seulement certaines gens ont de la mémoire et d'autres pas (sans aller jusqu'à l'oubli constant où vivent les ambassadeurs de Turquie), ce qui leur permet de trouver toujours – la nouvelle précédente s'étant évanouie au bout de huit jours, ou la suivante ayant le don de l'exorciser – de la place pour la nouvelle contraire qu'on leur dit. Mais même à égalité de mémoire, deux personnes ne se souviennent pas des mêmes choses. L'une aura prêté peu d'attention à un fait dont l'autre gardera grand remords, et, en revanche, aura saisi à la volée comme signe sympathique et caractéristique une parole que l'autre aura laissé échapper sans presque y penser. L'intérêt de ne pas s'être trompé quand on a émis un pronostic faux abrège la durée du souvenir de ce pronostic et permet d'affirmer très vite qu'on ne l'a pas émis. Enfin, un intérêt plus profond, plus désintéressé, diversifie les mémoires, si bien que le poète, qui a presque tout oublié des faits qu'on lui rappelle, retient une impression fugitive. De tout cela vient qu'après vingt ans d'absence on rencontre, au lieu de rancunes présumées, des pardons involontaires, inconscients, et, en revanche, tant de haines dont on ne peut s'expliquer (parce qu'on a oublié à son tour l'impression mauvaise qu'on a faite) la raison. L'histoire même des gens qu'on a le plus connus, on en a oublié les dates. Et parce qu'il y avait au moins vingt ans qu'elle avait vu Bloch pour la première fois, Mme de Guermantes eût juré qu'il était né dans son monde et avait été bercé sur les genoux de la duchesse de Chartres quand il avait deux ans.

Et combien de fois ces personnes étaient revenues devant moi, au cours de leur vie dont les diverses circonstances semblaient présenter les mêmes êtres, mais sous des formes et pour des fins variées ; et la diversité des points de ma vie par où avait passé le fil de celle de chacun de ces personnages avait fini par mêler ceux qui semblaient le plus éloignés, comme si la vie ne possédait qu'un nombre limité de fils pour exécuter les dessins les plus différents. Quoi de plus séparé, par exemple, dans mes passés divers, que mes visites à mon oncle Adolphe, que le neveu de Mme de Villeparisis cousine du Maréchal, que Legrandin et sa soeur, que l'ancien giletier ami de Françoise, dans la cour ! Et aujourd'hui tous ces fils différents s'étaient réunis pour faire la trame ici du ménage Saint-Loup, là jadis du jeune ménage Cambremer, pour ne pas parler de Morel et de tant d'autres dont la conjonction avait concouru à former une circonstance, si bien qu'il me semblait que la circonstance était l'unité complète et le personnage seulement une partie composante. Et ma vie était déjà assez longue pour qu'à plus d'un des êtres qu'elle m'offrait je trouvasse dans des régions opposées de mes souvenirs un autre être pour le compléter. Aux Elstir que je voyais ici en une place qui était un signe de la gloire maintenant acquise, je pouvais ajouter les plus anciens souvenirs des Verdurin, des Cottard, la conversation dans le restaurant de Rivebelle, la matinée où j'avais connu Albertine, et tant d'autres. Ainsi un amateur d'art à qui on montre le volet d'un retable se rappelle dans quelle église, dans quel musée, dans quelle collection particulière, les autres sont dispersés (de même qu'en suivant les catalogues des ventes ou en fréquentant les antiquaires, il finit par trouver l'objet jumeau de celui qu'il possède et qui fait avec lui la paire, il peut reconstituer dans sa tête la prédelle, l'autel tout entier). Comme un seau, montant le long d'un treuil, vient toucher la corde à diverses reprises et sur des côtés opposés, il n'y avait pas de personnage, presque pas même de choses ayant eu place dans ma vie, qui n'y eût joué tour à tour des rôles différents. Une simple relation mondaine, même un objet matériel, si je le retrouvais au bout de quelques années dans mon souvenir, je voyais que la vie n'avait pas cessé de tisser autour de lui des fils différents qui finissaient par le feutrer de ce beau velours pareil à celui qui, dans les vieux parcs, enveloppe une simple conduite d'eau d'un fourreau d'émeraude.

Ce n'était pas que l'aspect de ces personnes qui donnait l'idée de personnes de songe. Pour elles-mêmes la vie, déjà ensommeillée dans la jeunesse et l'amour, était de plus en plus devenue un songe. Elles avaient oublié jusqu'à leurs rancunes, leurs haines, et pour être certaines que c'était à la personne qui était là qu'elles n'adressaient plus la parole il y a dix ans, il eût fallu qu'elles se reportassent à un registre, mais qui était aussi vague qu'un rêve où on a été insulté on ne sait plus par qui. Tous ces songes formaient les apparences contrastées de la vie politique où on voyait dans un même ministère des gens qui s'étaient accusés de meurtre ou de trahison. Et ce songe devenait épais comme la mort chez certains vieillards, dans les jours qui suivaient celui où ils avaient fait l'amour. Pendant ces jours-là on ne pouvait plus rien demander au président de la République, il oubliait tout. Puis si on le laissait se reposer quelques jours, le souvenir des affaires publiques lui revenait, fortuit comme celui d'un rêve.

Parfois ce n'était pas en une seule image qu'apparaissait cet être si différent de celui que j'avais connu depuis. C'est pendant des années que Bergotte m'avait paru un doux vieillard divin, que je m'étais senti paralysé comme par une apparition devant le chapeau gris de Swann, le manteau violet de sa femme, le mystère dont le nom de sa race entourait la Mme de Guermantes jusque dans un salon : origines presque fabuleuses, charmante mythologie de relations devenues si banales ensuite, mais qu'elles prolongeaient dans le passé comme en plein ciel, avec un éclat pareil à celui que projette la queue étincelante d'une comète. Et même celles qui n'avaient pas commencé dans le mystère, comme mes relations avec Mme de Souvré, si sèches et si purement mondaines aujourd'hui, gardaient à leurs débuts leur premier sourire, plus calme, plus doux, et si onctueusement tracé dans la plénitude d'une après-midi au bord de la mer, d'une fin de journée de printemps à Paris, bruyante d'équipages, de poussière soulevée, et de soleil remué comme de l'eau. Et peut-être Mme de Souvré n'eût pas valu grand'chose si on l'eût détachée de ce cadre, comme ces monuments – la Salute par exemple – qui, sans grande beauté propre, font admirablement là où ils sont situés, mais elle faisait partie d'un lot de souvenirs que j'estimais à un certain prix, « l'un dans l'autre », sans me demander pour combien exactement la personne de Mme de Souvré y figurait.
