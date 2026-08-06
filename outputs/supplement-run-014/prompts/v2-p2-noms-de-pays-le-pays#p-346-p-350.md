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
        "le peintre Elstir"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
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
      "confidence": 0.9,
      "evidence": "« C'est un ami de Swann, et un artiste très connu, de grande valeur » … « Il prodigua pour moi une amabilité… À côté de celle d'un grand artiste, l'amabilité d'un grand seigneur… a l'air d'un jeu d'acteur… Robert de Saint-Loup cherchait à plaire, Elstir aimait à donner, à se donner. »",
      "explanation": "The narrator elevates Elstir as a great artist and emphasizes his generous, authentic amability, contrasting it favorably with aristocratic charm."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He is framed as both artistically eminent and personally generous, raising his local standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-346-p-350"
}

### Candidate characters

[
  "Robert de Saint-Loup",
  "Swann",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Sans doute bien des fois, au passage de jolies jeunes filles, je m'étais fait la promesse de les revoir. D'habitude, elles ne reparaissent pas ; d'ailleurs la mémoire, qui oublie vite leur existence, retrouverait difficilement leurs traits ; nos yeux ne les reconnaîtraient peut-être pas, et déjà nous avons vu passer de nouvelles jeunes filles que nous ne reverrons pas non plus. Mais d'autres fois, et c'est ainsi que cela devait arriver pour la petite bande insolente, le hasard les ramène avec insistance devant nous. Il nous paraît alors beau, car nous discernons en lui comme un commencement d'organisation, d'effort, pour composer notre vie ; il nous rend facile, inévitable et quelquefois – après des interruptions qui ont pu faire espérer de cesser de nous souvenir – cruelle la fidélité des images à la possession desquelles nous nous croirons plus tard avoir été prédestinés, et que sans lui nous aurions pu, tout au début, oublier, comme tant d'autres, si aisément.

### Passage

Bientôt le séjour de Saint-Loup toucha à sa fin. Je n'avais pas revu ces jeunes filles sur la plage. Il restait trop peu de l'après-midi à Balbec pour pouvoir s'occuper d'elles et tâcher de faire, à mon intention, leur connaissance. Le soir il était plus libre et continuait à m'emmener souvent à Rivebelle. Il y a dans ces restaurants, comme dans les jardins publics et les trains, des gens enfermés dans une apparence ordinaire et dont le nom nous étonne, si l'ayant par hasard demandé, nous découvrons qu'ils sont non l'inoffensif premier venu que nous supposions, mais rien de moins que le ministre ou le duc dont nous avons si souvent entendu parler. Déjà deux ou trois fois dans le restaurant de Rivebelle, nous avions, Saint-Loup et moi, vu venir s'asseoir à une table, quand tout le monde commençait à partir, un homme de grande taille, très musclé, aux traits réguliers, à la barbe grisonnante, mais de qui le regard songeur restait fixé avec application dans le vide. Un soir que nous demandions au patron qui était ce dîneur obscur, isolé et retardataire : « Comment, vous ne connaissiez pas le célèbre peintre Elstir ? » nous dit-il. Swann avait une fois prononcé son nom devant moi, j'avais entièrement oublié à quel propos ; mais l'omission d'un souvenir, comme celui d'un membre de phrase dans une lecture, favorise parfois non l'incertitude, mais l'éclosion d'une certitude prématurée. « C'est un ami de Swann, et un artiste très connu, de grande valeur », dis-je à Saint-Loup. Aussitôt passa sur lui et sur moi, comme un frisson, la pensée qu'Elstir était un grand artiste, un homme célèbre, puis, que nous confondant avec les autres dîneurs, il ne se doutait pas de l'exaltation où nous jetait l'idée de son talent. Sans doute, qu'il ignorât notre admiration, et que nous connaissions Swann, ne nous eût pas été pénible si nous n'avions pas été aux bains de mer. Mais attardés à un âge où l'enthousiasme ne peut rester silencieux, et transportés dans une vie où l'incognito semble étouffant, nous écrivîmes une lettre signée de nos noms, où nous dévoilions à Elstir dans les deux dîneurs assis à quelques pas de lui deux amateurs passionnés de son talent, deux amis de son grand ami Swann, et où nous demandions à lui présenter nos hommages. Un garçon se chargea de porter cette missive à l'homme célèbre.

Célèbre, Elstir ne l'était peut-être pas encore à cette époque tout à fait autant que le prétendait le patron de l'établissement, et qu'il le fut d'ailleurs bien peu d'années plus tard. Mais il avait été un des premiers à habiter ce restaurant alors que ce n'était encore qu'une sorte de ferme et à y amener une colonie d'artistes (qui avaient du reste tous émigré ailleurs dès que la ferme où l'on mangeait en plein air sous un simple auvent était devenue un centre élégant ; Elstir lui-même ne revenait en ce moment à Rivebelle qu'à cause d'une absence de sa femme avec laquelle il habitait non loin de là). Mais un grand talent, même quand il n'est pas encore reconnu, provoque nécessairement quelques phénomènes d'admiration, tels que le patron de la ferme avait été à même d'en distinguer dans les questions de plus d'une Anglaise de passage, avide de renseignements sur la vie que menait Elstir, ou dans le nombre de lettres que celui-ci recevait de l'étranger. Alors le patron avait remarqué davantage qu'Elstir n'aimait pas être dérangé pendant qu'il travaillait, qu'il se relevait la nuit pour emmener un petit modèle poser nu au bord de la mer, quand il y avait clair de lune, et il s'était dit que tant de fatigues n'étaient pas perdues, ni l'admiration des touristes injustifiée, quand il avait dans un tableau d'Elstir reconnu une croix de bois qui était plantée à l'entrée de Rivebelle. « C'est bien elle, répétait-il avec stupéfaction. Il y a les quatre morceaux ! Ah ! aussi il s'en donne une peine ! »

Et il ne savait pas si un petit « lever de soleil sur la mer », qu'Elstir lui avait donné, ne valait pas une fortune.

Nous le vîmes lire notre lettre, la remettre dans sa poche, continuer à dîner, commencer à demander ses affaires, se lever pour partir, et nous étions tellement sûrs de l'avoir choqué par notre démarche que nous eussions souhaité maintenant (tout autant que nous l'avions redouté) de partir sans avoir été remarqués par lui. Nous ne pensions pas un seul instant à une chose qui aurait dû pourtant nous sembler la plus importante, c'est que notre enthousiasme pour Elstir, de la sincérité duquel nous n'aurions pas permis qu'on doutât et dont nous aurions pu, en effet, donner comme témoignage notre respiration entrecoupée par l'attente, notre désir de faire n'importe quoi de difficile ou d'héroïque pour le grand homme, n'était pas, comme nous nous le figurions, de l'admiration, puisque nous n'avions jamais rien vu d'Elstir ; notre sentiment pouvait avoir pour objet l'idée creuse de « un grand artiste », non pas une oeuvre qui nous était inconnue. C'était tout au plus de l'admiration à vide, le cadre nerveux, l'armature sentimentale d'une admiration sans contenu, c'est-à-dire quelque chose d'aussi indissolublement attaché à l'enfance que certains organes qui n'existent plus chez l'homme adulte ; nous étions encore des enfants. Elstir cependant allait arriver à la porte, quand tout à coup il fit un crochet et vint à nous. J'étais transporté d'une délicieuse épouvante comme je n'aurais pu en éprouver quelques années plus tard, parce que, en même temps que l'âge diminue la capacité, l'habitude du monde ôte toute idée de provoquer d'aussi étranges occasions de ressentir ce genre d'émotions.

Dans les quelques mots qu'Elstir vint nous dire, en s'asseyant à notre table, il ne me répondit jamais, les diverses fois où je lui parlai de Swann. Je commençai à croire qu'il ne le connaissait pas. Il ne m'en demanda pas moins d'aller le voir à son atelier de Balbec, invitation qu'il n'adressa pas à Saint-Loup, et que me valurent, ce que n'aurait peut-être pas fait la recommandation de Swann si Elstir eût été lié avec lui (car la part des sentiments désintéressés est plus grande qu'on ne croit dans la vie des hommes), quelques paroles qui lui firent penser que j'aimais les arts. Il prodigua pour moi une amabilité, qui était aussi supérieure à celle de Saint-Loup que celle-ci à l'affabilité d'un petit bourgeois. À côté de celle d'un grand artiste, l'amabilité d'un grand seigneur, si charmante soit-elle, a l'air d'un jeu d'acteur, d'une simulation. Saint-Loup cherchait à plaire, Elstir aimait à donner, à se donner. Tout ce qu'il possédait, idées, oeuvres, et le reste qu'il comptait pour bien moins, il l'eût donné avec joie à quelqu'un qui l'eût compris. Mais faute d'une société supportable, il vivait dans un isolement, avec une sauvagerie que les gens du monde appelaient de la pose et de la mauvaise éducation, les pouvoirs publics un mauvais esprit, ses voisins de la folie, sa famille de l'égoïsme et de l'orgueil.
