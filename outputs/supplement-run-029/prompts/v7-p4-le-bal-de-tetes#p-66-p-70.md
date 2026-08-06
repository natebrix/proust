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
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte",
        "Gilberte de Robert de Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    },
    {
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup",
        "Robert"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Gilberte",
      "target": "Robert de Saint-Loup",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Gilberte en parlait sur un ton déférent, comme si c'eût été un être supérieur... »; « Je ne puis vous dire à quel point la moindre des choses qu'il me disait... me frappe maintenant. » Le narrateur ajoute que ses idées « s'étaient souvent... vérifiées par la dernière guerre ».",
      "explanation": "Gilberte expresses strong admiration for Robert's military intelligence and insight; the narrator corroborates by reminding that his theses were confirmed by events and critics."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Locally, Robert is strongly elevated as a clear-sighted and foresighted mind, thanks to Gilberte's deference and the retrospective validation of the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-66-p-70"
}

### Candidate characters

[
  "Elstir",
  "M. Verdurin",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

« Que devient la marquise d'Arpajon ? demanda Mme de Cambremer. – Mais elle est morte, répondit Bloch. – Vous confondez avec la comtesse d'Arpajon qui est morte l'année dernière. » La princesse de Malte se mêla à la discussion ; jeune veuve d'un vieux mari très riche et porteur d'un grand nom, elle était beaucoup demandée en mariage et en avait pris une grande assurance. « La marquise d'Arpajon est morte aussi il y a à peu près un an. – Ah ! un an, je vous réponds que non, répondit Mme de Cambremer, j'ai été à une soirée de musique chez elle il y a moins d'un an. » Bloch, pas plus que les « gigolos » du monde, ne put prendre part utilement à la discussion, car toutes ces morts de personnes âgées étaient à une distance d'eux trop grande, soit par la différence énorme des années, soit par la récente arrivée (de Bloch, par exemple) dans une société différente qu'il abordait de biais, au moment où elle déclinait, dans un crépuscule où le souvenir d'un passé qui ne lui était pas familier ne pouvait l'éclairer. Et pour les gens du même âge et du même milieu, la mort avait perdu de sa signification étrange. D'ailleurs, on faisait tous les jours prendre des nouvelles de tant de gens à l'article de la mort, et dont les uns s'étaient rétablis tandis que d'autres avaient « succombé », qu'on ne se souvenait plus au juste si telle personne qu'on n'avait jamais l'occasion de voir s'était sortie de sa fluxion de poitrine ou avait trépassé. La mort se multipliait et devenait plus incertaine dans ces régions âgées. À cette croisée de deux générations et de deux sociétés qui, en vertu de raisons différentes, mal placées pour distinguer la mort, la confondaient presque avec la vie, la première s'était mondanisée, était devenue un incident qui qualifiait plus ou moins une personne ; sans que le ton dont on parlait eût l'air de signifier que cet incident terminait tout pour elle, on disait : « mais vous oubliez, un tel est mort », comme on eût dit : « il est décoré » (l'adjectif était autre, quoique pas plus important), « il est de l'Académie », ou – et cela revenait au même puisque cela empêchait aussi d'assister aux fêtes – « il est allé passer l'hiver dans le Midi », « on lui a ordonné les montagnes ». Encore, pour des hommes connus, ce qu'ils laissaient en mourant aidait à se rappeler que leur existence était terminée. Mais pour les simples gens du monde très âgés, on s'embrouillait sur le fait qu'ils fussent morts ou non, non seulement parce qu'on connaissait mal ou qu'on avait oublié leur passé, mais parce qu'ils ne tenaient en quoi que ce soit à l'avenir. Et la difficulté qu'avait chacun de faire un triage entre les maladies, l'absence, la retraite à la campagne, la mort des vieilles gens du monde, consacrait, tout autant que l'indifférence des hésitants, l'insignifiance des défunts.

### Passage

« Mais si elle n'est pas morte, comment se fait-il qu'on ne la voie plus jamais, ni son mari non plus ? demanda une vieille fille qui aimait faire de l'esprit. – Mais je te dirai, reprit la mère, qui, quoique quinquagénaire, ne manquait pas une fête, que c'est parce qu'ils sont vieux, et qu'à cet âge-là on ne sort plus. » Il semblait qu'il y eût avant le cimetière toute une cité close des vieillards, aux lampes toujours allumées dans la brume. Mme de Sainte-Euverte trancha le débat en disant que la comtesse d'Arpajon était morte, il y avait un an, d'une longue maladie, mais que la marquise d'Arpajon était morte aussi depuis, très vite, « d'une façon tout à fait insignifiante », mort qui par là ressemblait à toutes ces vies, et par là aussi expliquait qu'elle eût passé inaperçue, excusait ceux qui confondaient. En entendant que Mme d'Arpajon était vraiment morte, la vieille fille jeta sur sa mère un regard alarmé, car elle craignait que d'apprendre la mort d'une de ses « contemporaines » ne la « frappât » ; elle croyait entendre d'avance parler de la mort de sa propre mère avec cette explication : « Elle avait été « très frappée » par la mort de Madame d'Arpajon. » Mais la mère, au contraire, se faisait à elle-même l'effet de l'avoir emporté dans un concours sur des concurrents de marque, chaque fois qu'une personne de son âge « disparaissait ». Leur mort était la seule manière dont elle prît encore agréablement conscience de sa propre vie. La vieille fille s'aperçut que sa mère, qui n'avait pas semblé fâchée de dire que Mme d'Arpajon était recluse dans les demeures d'où ne sortent plus guère les vieillards fatigués, l'avait été moins encore d'apprendre que la marquise était entrée dans la Cité d'après, celle d'où on ne sort plus. Cette constatation de l'indifférence de sa mère amusa l'esprit caustique de la vieille fille. Et pour faire rire ses amies, plus tard, elle fit un récit désopilant de la manière allègre, prétendait-elle, dont sa mère avait dit en se frottant les mains : « Mon Dieu, il est bien vrai que cette pauvre Madame d'Arpajon est morte. » Même pour ceux qui n'avaient pas besoin de cette mort pour se réjouir d'être vivants, elle les rendit heureux. Car toute mort est pour les autres une simplification d'existence, ôte le scrupule de se montrer reconnaissant, l'obligation de faire des visites. Toutefois, comme je l'ai dit, ce n'est pas ainsi que la mort de M. Verdurin avait été accueillie par Elstir.

Une dame sortit, car elle avait d'autres matinées et devait aller goûter avec deux reines. C'était cette grande cocotte du monde que j'avais connue autrefois, la princesse de Nassau. Mis à part le fait que sa taille avait diminué – ce qui lui donnait l'air, par sa tête située à une bien moindre hauteur qu'elle n'était autrefois, d'avoir ce qu'on appelle « un pied dans la tombe » – on aurait à peine pu dire qu'elle avait vieilli. Elle restait une Marie-Antoinette au nez autrichien, au regard délicieux, conservée, embaumée grâce à mille fards adorablement unis qui lui faisaient une figure lilas. Il flottait sur elle cette expression confuse et tendre d'être obligée de partir, de promettre tendrement de revenir, de s'esquiver discrètement, qui tenait à la foule des réunions d'élite où on l'attendait. Née presque sur les marches d'un trône, mariée trois fois, entretenue longtemps et richement par de grands banquiers, sans compter les mille fantaisies qu'elle s'était offertes, elle portait légèrement, comme ses yeux admirables et ronds, comme sa figure fardée et comme sa robe mauve, les souvenirs un peu embrouillés de ce passé innombrable. Comme elle passait devant moi en se sauvant « à l'anglaise », je la saluai. Elle me reconnut, elle me serra la main et fixa sur moi ses rondes prunelles mauves de l'air qui voulait dire : « Comme il y a longtemps que nous nous sommes vus, nous parlerons de cela une autre fois. » Elle me serrait la main avec force, ne se rappelant pas au juste si en voiture, un soir qu'elle me ramenait de chez la Mme de Guermantes, il y avait eu ou non une passade entre nous. À tout hasard, elle sembla faire allusion à ce qui n'avait pas été, chose qui ne lui était pas difficile puisqu'elle prenait un air de tendresse pour une tarte aux fraises et revêtait, si elle était obligée de partir avant la fin de la musique, l'attitude désespérée d'un abandon qui toutefois ne serait pas définitif. Incertaine, d'ailleurs, sur la passade avec moi, son serrement furtif ne s'attarda pas et elle ne me dit pas un mot. Elle me regarda seulement comme j'ai dit, d'une façon qui signifiait « qu'il y a longtemps ! » et où repassaient ses maris, les hommes qui l'avaient entretenue, deux guerres, et ses yeux stellaires, semblables à une horloge astronomique taillée dans une opale, marquèrent successivement toutes ces heures solennelles d'un passé si lointain, qu'elle retrouvait à tout moment quand elle voulait vous dire un bonjour qui était toujours une excuse. Puis m'ayant quitté, elle se mit à trotter vers la porte pour qu'on ne se dérangeât pas pour elle, pour me montrer que, si elle n'avait pas causé avec moi, c'est qu'elle était pressée, pour rattraper la minute perdue à me serrer la main afin d'être exacte chez la reine d'Espagne qui devait goûter seule avec elle. Même, près de la porte, je crus qu'elle allait prendre le pas de course. Elle courait, en effet, à son tombeau.

Pendant ce temps on entendait la princesse de Guermantes répéter d'un air exalté et d'une voix de ferraille que lui faisait son râtelier : « Oui, c'est cela, nous ferons clan ! nous ferons clan ! J'aime cette jeunesse si intelligente, si participante, ah ! quelle mugichienne vous êtes ! » Elle parlait, son gros monocle dans son oeil rond, mi-amusé, mi-s'excusant de ne pouvoir soutenir la gaîté longtemps, mais jusqu'au bout elle était décidée à « participer », à « faire clan ».

* * *

Je m'étais assis à côté de Gilberte de Saint-Loup. Nous parlâmes beaucoup de Saint-Loup, Gilberte en parlait sur un ton déférent, comme si c'eût été un être supérieur qu'elle tenait à me montrer qu'elle avait admiré et compris. Nous nous rappelâmes l'un à l'autre combien les idées qu'il exposait jadis sur l'art de la guerre (car il lui avait souvent redit à Tansonville les mêmes thèses que je lui avais entendu exposer à Doncières et plus tard) s'étaient souvent et, en somme, sur un grand nombre de points trouvées vérifiées par la dernière guerre. « Je ne puis vous dire à quel point la moindre des choses qu'il me disait à Doncières et aussi pendant la guerre me frappe maintenant. Les dernières paroles que j'ai entendues de lui, quand nous nous sommes quittés pour ne plus nous revoir, étaient qu'il attendait Hindenburg, général napoléonien, à un des types de la bataille napoléonienne, celle qui a pour but de séparer deux adversaires, peut-être, avait-il ajouté, les Anglais et nous. Or, à peine un an après la mort de Saint-Loup, un critique pour lequel il avait une profonde admiration et qui exerçait visiblement une grande influence sur ses idées militaires, M. Henry Bidou, disait que l'offensive d'Hindenburg en mars 1918, c'était « la bataille de séparation d'un adversaire massé contre deux adversaires en ligne, manoeuvre que l'Empereur a réussie en 1796 sur l'Apennin et qu'il a manquée en 1815 en Belgique ». Quelques instants auparavant, Saint-Loup comparait devant moi les batailles à des pièces où il n'est pas toujours facile de savoir ce qu'a voulu l'auteur, où lui-même a changé son plan en cours de route. Or, pour cette offensive allemande de 1918, sans doute, en l'interprétant de cette façon Saint-Loup ne serait pas d'accord avec M. Bidou. Mais d'autres critiques pensent que c'est le succès d'Hindenburg dans la direction d'Amiens, puis son arrêt forcé, son succès dans les Flandres, puis l'arrêt encore qui ont fait, accidentellement en somme, d'Amiens, puis de Boulogne, des buts qu'il ne s'était pas préalablement assignés. Et, chacun pouvant refaire une pièce à sa manière, il y en a qui voient dans cette offensive l'annonce d'une marche foudroyante sur Paris, d'autres des coups de boutoir désordonnés pour détruire l'armée anglaise. Et même si les ordres donnés par le chef s'opposent à telles ou telles conceptions, il restera toujours aux critiques le moyen de dire, comme Mounet-Sully à Coquelin qui l'assurait que le Misanthrope n'était pas la pièce triste, dramatique qu'il voulait jouer (car Molière, au témoignage des contemporains, en donnait une interprétation comique et y faisait rire) : « Hé bien, c'est que Molière se trompait. »
