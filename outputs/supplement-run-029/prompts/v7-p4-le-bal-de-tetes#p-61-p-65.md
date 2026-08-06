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
      "canonical_name": "Bloch",
      "surface_forms": [
        "Bloch"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bloch",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.83,
      "evidence": "Bloch... ne put prendre part utilement à la discussion, car... il abordait [cette société] de biais.",
      "explanation": "In the death-notice debate with Mme de Cambremer, the narrator frames Bloch as unable to contribute usefully due to generational and social mismatch, lowering his local rhetorical standing."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Bloch",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.86,
      "evidence": "Legrandin méprisait Bloch autrefois... Il fut très aimable avec lui.",
      "explanation": "The narrator reports a reversal from past contempt to present amiability by Legrandin toward Bloch, locally repositioning Bloch from shunned to included."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch",
      "dimension": "inclusion_exclusion",
      "delta": 1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.84,
      "explanation": "Legrandin's shift to amiability brings Bloch into friendly social contact rather than prior avoidance."
    },
    {
      "character": "Bloch",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "Bloch is portrayed as socially and informationally out of depth in the discussion, weakening his local authority."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-61-p-65"
}

### Candidate characters

[
  "Albertine",
  "Bergotte",
  "Gilberte",
  "Legrandin",
  "Mlle de Stermaria",
  "Mme de Cambremer",
  "duchesse de Guermantes",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Parfois ce n'était pas en une seule image qu'apparaissait cet être si différent de celui que j'avais connu depuis. C'est pendant des années que Bergotte m'avait paru un doux vieillard divin, que je m'étais senti paralysé comme par une apparition devant le chapeau gris de Swann, le manteau violet de sa femme, le mystère dont le nom de sa race entourait la duchesse de Guermantes jusque dans un salon : origines presque fabuleuses, charmante mythologie de relations devenues si banales ensuite, mais qu'elles prolongeaient dans le passé comme en plein ciel, avec un éclat pareil à celui que projette la queue étincelante d'une comète. Et même celles qui n'avaient pas commencé dans le mystère, comme mes relations avec Mme de Souvré, si sèches et si purement mondaines aujourd'hui, gardaient à leurs débuts leur premier sourire, plus calme, plus doux, et si onctueusement tracé dans la plénitude d'une après-midi au bord de la mer, d'une fin de journée de printemps à Paris, bruyante d'équipages, de poussière soulevée, et de soleil remué comme de l'eau. Et peut-être Mme de Souvré n'eût pas valu grand'chose si on l'eût détachée de ce cadre, comme ces monuments – la Salute par exemple – qui, sans grande beauté propre, font admirablement là où ils sont situés, mais elle faisait partie d'un lot de souvenirs que j'estimais à un certain prix, « l'un dans l'autre », sans me demander pour combien exactement la personne de Mme de Souvré y figurait.

### Passage

Une chose me frappa plus encore chez tous ces êtres que les changements physiques, sociaux, qu'ils avaient subis, ce fut celui qui tenait à l'idée différente qu'ils avaient les uns des autres. Legrandin méprisait Bloch autrefois et ne lui adressait jamais la parole. Il fut très aimable avec lui. Ce n'était pas du tout à cause de la situation plus grande qu'avait prise Bloch, ce qui, dans ce cas, ne mériterait pas d'être noté, car les changements sociaux amènent forcément des changements respectifs de position entre ceux qui les ont subis. Non ; c'était que les gens – les gens, c'est-à-dire ce qu'ils sont pour nous – n'ont plus dans notre mémoire l'uniformité d'un tableau. Au gré de notre oubli, ils évoluent. Quelquefois nous allons jusqu'à les confondre avec d'autres : « Bloch, c'est quelqu'un qui venait à Combray », et en disant Bloch c'était moi qu'on voulait dire. Inversement, Mme Sazerat était persuadée que de moi était telle thèse historique sur Philippe II (laquelle était de Bloch). Sans aller jusqu'à ces interversions, on oublie les crasses que l'un vous a faites, ses défauts, la dernière fois où on s'est quitté sans se serrer la main et, en revanche, on s'en rappelle une plus ancienne, où on était bien ensemble. Et c'est à cette fois plus ancienne que les manières de Legrandin répondaient dans son amabilité avec Bloch, soit qu'il eût perdu la mémoire d'un certain passé, soit qu'il le jugeât prescrit, mélange de pardon, d'oubli, d'indifférence qui est aussi un effet du Temps. D'ailleurs, les souvenirs que nous avons les uns des autres, même dans l'amour, ne sont pas les mêmes. J'avais vu Albertine me rappeler à merveille telle parole que je lui avais dite dans nos premières rencontres et que j'avais complètement oubliée. D'un autre fait enfoncé à jamais dans ma tête comme un caillou elle n'avait aucun souvenir. Nos vies parallèles ressemblaient aux bords de ces allées où de distance en distance des vases de fleurs sont placés symétriquement, mais non en face les uns des autres. À plus forte raison est-il compréhensible que pour des gens qu'on connaît peu on se rappelle à peine qui ils sont, ou on s'en rappelle autre chose, mais de plus ancien, que ce qu'on en pensait autrefois, quelque chose qui est suggéré par les gens au milieu de qui on les retrouve, qui ne les connaissent que depuis peu, parés de qualités et d'une situation qu'ils n'avaient pas autrefois mais que l'oublieux accepte d'emblée.

Sans doute la vie, en mettant à plusieurs reprises ces personnes sur mon chemin, me les avait présentées dans des circonstances particulières qui, en les entourant de toutes parts, m'avaient rétréci la vue que j'avais eue d'elles, et m'avait empêché de connaître leur essence. Ces Guermantes mêmes, qui avaient été pour moi l'objet d'un si grand rêve, quand je m'étais approché d'abord de l'un d'eux, m'étaient apparus sous l'aspect, l'une d'une vieille amie de grand'mère, l'autre d'un monsieur qui m'avait regardé d'un air si désagréable à midi dans les jardins du casino. (Car il y a entre nous et les êtres un liséré de contingences, comme j'avais compris, dans mes lectures de Combray, qu'il y en a un de perception et qui empêche la mise en contact absolue de la réalité et de l'esprit.) De sorte que ce n'était jamais qu'après coup, en les rapportant à un nom, que leur connaissance était devenue pour moi la connaissance des Guermantes. Mais peut-être cela même me rendait-il la vie plus poétique de penser que la race mystérieuse aux yeux perçants, au bec d'oiseau, la race rose, dorée, inapprochable, s'était trouvée si souvent, si naturellement, par l'effet de circonstances aveugles et différentes, s'offrir à ma contemplation, à mon commerce, même à mon intimité, au point que, quand j'avais voulu connaître Mlle de Stermaria ou faire faire des robes à Albertine, c'était, comme aux plus serviables de mes amis, à des Guermantes que je m'étais adressé. Certes, cela m'ennuyait d'aller chez eux autant que chez les autres gens du monde que j'avais connus ensuite. Même, pour la Mme de Guermantes, comme pour certaines pages de Bergotte, son charme ne m'était visible qu'à distance et s'évanouissait quand j'étais près d'elle, car il résidait dans ma mémoire et dans mon imagination. Mais enfin, malgré tout, les Guermantes, comme Gilberte aussi, différaient des autres gens du monde en ce qu'ils plongeaient plus avant leurs racines dans un passé de ma vie où je rêvais davantage et croyais plus aux individus. Ce que je possédais avec ennui, en causant en ce moment avec l'une et avec l'autre, c'était du moins celles des imaginations de mon enfance que j'avais trouvées le plus belles et crues le plus inaccessibles, et je me consolais en confondant, comme un marchand qui s'embrouille dans ses livres, la valeur de leur possession avec le prix auquel les avait cotées mon désir.

Mais pour d'autres êtres, le passé de mes relations avec eux était gonflé de rêves plus ardents, formés sans espoir, où s'épanouissait si richement ma vie d'alors, dédiée à eux tout entière, que je pouvais à peine comprendre comment leur exaucement était ce mince, étroit et terne ruban d'une intimité indifférente et dédaignée où je ne pouvais plus rien retrouver de ce qui avait fait leur mystère, leur fièvre et leur douceur.

* * *

« Que devient la marquise d'Arpajon ? demanda Mme de Cambremer. – Mais elle est morte, répondit Bloch. – Vous confondez avec la comtesse d'Arpajon qui est morte l'année dernière. » La princesse de Malte se mêla à la discussion ; jeune veuve d'un vieux mari très riche et porteur d'un grand nom, elle était beaucoup demandée en mariage et en avait pris une grande assurance. « La marquise d'Arpajon est morte aussi il y a à peu près un an. – Ah ! un an, je vous réponds que non, répondit Mme de Cambremer, j'ai été à une soirée de musique chez elle il y a moins d'un an. » Bloch, pas plus que les « gigolos » du monde, ne put prendre part utilement à la discussion, car toutes ces morts de personnes âgées étaient à une distance d'eux trop grande, soit par la différence énorme des années, soit par la récente arrivée (de Bloch, par exemple) dans une société différente qu'il abordait de biais, au moment où elle déclinait, dans un crépuscule où le souvenir d'un passé qui ne lui était pas familier ne pouvait l'éclairer. Et pour les gens du même âge et du même milieu, la mort avait perdu de sa signification étrange. D'ailleurs, on faisait tous les jours prendre des nouvelles de tant de gens à l'article de la mort, et dont les uns s'étaient rétablis tandis que d'autres avaient « succombé », qu'on ne se souvenait plus au juste si telle personne qu'on n'avait jamais l'occasion de voir s'était sortie de sa fluxion de poitrine ou avait trépassé. La mort se multipliait et devenait plus incertaine dans ces régions âgées. À cette croisée de deux générations et de deux sociétés qui, en vertu de raisons différentes, mal placées pour distinguer la mort, la confondaient presque avec la vie, la première s'était mondanisée, était devenue un incident qui qualifiait plus ou moins une personne ; sans que le ton dont on parlait eût l'air de signifier que cet incident terminait tout pour elle, on disait : « mais vous oubliez, un tel est mort », comme on eût dit : « il est décoré » (l'adjectif était autre, quoique pas plus important), « il est de l'Académie », ou – et cela revenait au même puisque cela empêchait aussi d'assister aux fêtes – « il est allé passer l'hiver dans le Midi », « on lui a ordonné les montagnes ». Encore, pour des hommes connus, ce qu'ils laissaient en mourant aidait à se rappeler que leur existence était terminée. Mais pour les simples gens du monde très âgés, on s'embrouillait sur le fait qu'ils fussent morts ou non, non seulement parce qu'on connaissait mal ou qu'on avait oublié leur passé, mais parce qu'ils ne tenaient en quoi que ce soit à l'avenir. Et la difficulté qu'avait chacun de faire un triage entre les maladies, l'absence, la retraite à la campagne, la mort des vieilles gens du monde, consacrait, tout autant que l'indifférence des hésitants, l'insignifiance des défunts.
