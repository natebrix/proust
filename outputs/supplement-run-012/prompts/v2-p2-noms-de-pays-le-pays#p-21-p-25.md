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
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère",
        "grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la grand-mère",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "Mais la grand-mère ... m'avait appris à en aimer les vraies beautés, qui sont tout autres.",
      "explanation": "The narrator credits his grandmother with guiding him to the 'true beauties' of Madame de Sévigné, distinguishing her discerning, heartfelt approach from superficial salon imitation. This frames her as an authority in taste and moral-aesthetic insight."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "She is locally elevated as a trusted mentor whose inner-directed love and judgment teach the narrator to value genuine artistic qualities."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-21-p-25"
}

### Candidate characters

[
  "Elstir",
  "la mère du narrateur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

– Tu devrais peut-être essayer de dormir un peu, et tourna les yeux vers la fenêtre dont nous avions baissé le rideau qui ne remplissait pas tout le cadre de la vitre, de sorte que le soleil pouvait glisser sur le chêne ciré de la portière et le drap de la banquette (comme une réclame beaucoup plus persuasive pour une vie mêlée à la nature que celles accrochées trop haut dans le wagon, par les soins de la Compagnie, et représentant des paysages dont je ne pouvais pas lire les noms) la même clarté tiède et dormante qui faisait la sieste dans les clairières.

### Passage

Mais quand ma grand'mère croyait que j'avais les yeux fermés, je la voyais par moments sous son voile à gros pois jeter un regard sur moi, puis le retirer, puis recommencer, comme quelqu'un qui cherche à s'efforcer, pour s'y habituer, à un exercice qui lui est pénible.

Alors je lui parlais, mais cela ne semblait pas lui être agréable. Et à moi pourtant ma propre voix me donnait du plaisir, et de même les mouvements les plus insensibles, les plus intérieurs de mon corps. Aussi je tâchais de les faire durer, je laissais chacune de mes inflexions s'attarder longtemps aux mots, je sentais chacun de mes regards se trouver bien là où il s'était posé et y rester au delà du temps habituel. « Allons, repose-toi, me dit ma grand'mère. Si tu ne peux pas dormir lis quelque chose. » Et elle me passa un volume de Madame de Sévigné que j'ouvris, pendant qu'elle-même s'absorbait dans les Mémoires de Madame de Beausergent. Elle ne voyageait jamais sans un tome de l'une et de l'autre. C'était ses deux auteurs de prédilection. Ne bougeant pas volontiers ma tête en ce moment et éprouvant un grand plaisir à garder une position une fois que je l'avais prise, je restai à tenir le volume de Madame de Sévigné sans l'ouvrir, et je n'abaissai pas sur lui mon regard qui n'avait devant lui que le store bleu de la fenêtre. Mais contempler ce store me paraissait admirable et je n'eusse pas pris la peine de répondre à qui eût voulu me détourner de ma contemplation. La couleur bleue du store me semblait, non peut-être par sa beauté mais par sa vivacité intense, effacer à tel point toutes les couleurs qui avaient été devant mes yeux depuis le jour de ma naissance jusqu'au moment où j'avais fini d'avaler ma boisson et où elle avait commencé de faire son effet, qu'à côté de ce bleu du store, elles étaient pour moi aussi ternes, aussi nulles, que peut l'être rétrospectivement l'obscurité où ils ont vécu pour les aveugles-nés qu'on opère sur le tard et qui voient enfin les couleurs. Un vieil employé vint nous demander nos billets. Les reflets argentés qu'avaient les boutons en métal de sa tunique ne laissèrent pas de me charmer. Je voulus lui demander de s'asseoir à côté de nous. Mais il passa dans un autre wagon, et je songeai avec nostalgie à la vie des cheminots, lesquels, passant tout leur temps en chemin de fer, ne devaient guère manquer un seul jour de voir ce vieil employé. Le plaisir que j'éprouvais à regarder le store bleu et à sentir que ma bouche était à demi ouverte commença enfin à diminuer. Je devins plus mobile ; je remuai un peu ; j'ouvris le volume que ma grand'mère m'avait tendu et je pus fixer mon attention sur les pages que je choisis çà et là. Tout en lisant je sentais grandir mon admiration pour Madame de Sévigné.

Il ne faut pas se laisser tromper par des particularités purement formelles qui tiennent à l'époque, à la vie de salon et qui font que certaines personnes croient qu'elles ont fait leur Sévigné quand elles ont dit : « Mandez-moi, ma bonne » ou « Ce comte me parut avoir bien de l'esprit », ou « faner est la plus jolie chose du monde ». Déjà Mme de Simiane s'imagine ressembler à sa grand'mère parce qu'elle écrit : « M. de la Boulie se porte à merveille, monsieur, et il est fort en état d'entendre des nouvelles de sa mort », ou « Oh ! mon cher marquis, que votre lettre me plaît ! Le moyen de ne pas y répondre », ou encore : « Il me semble, monsieur, que vous me devez une réponse et moi des tabatières de bergamote. Je m'en acquitte pour huit, il en viendra d'autres... ; jamais la terre n'en avait tant porté. C'est apparemment pour vous plaire. » Et elle écrit dans ce même genre la lettre sur la saignée, sur les citrons, etc., qu'elle se figure être des lettres de Madame de Sévigné. Mais ma grand'mère qui était venue à celle-ci par le dedans, par l'amour pour les siens, pour la nature, m'avait appris à en aimer les vraies beautés, qui sont tout autres. Elles devaient bientôt me frapper d'autant plus que Madame de Sévigné est une grande artiste de la même famille qu'un peintre que j'allais rencontrer à Balbec et qui eut une influence si profonde sur ma vision des choses, Elstir. Je me rendis compte à Balbec que c'est de la même façon que lui qu'elle nous présente les choses, dans l'ordre de nos perceptions, au lieu de les expliquer d'abord par leur cause. Mais déjà cet après-midi-là, dans ce wagon, en relisant la lettre où apparaît le clair de lune : « Je ne pus résister à la tentation, je mets toutes mes coiffes et casques qui n'étaient pas nécessaires, je vais dans ce mail dont l'air est bon comme celui de ma chambre, je trouve mille coquecigrues, des moines blancs et noirs, plusieurs religieuses grises et blanches, du linge jeté par-ci par-là, des hommes ensevelis tout droits contre des arbres, etc. », je fus ravi par ce que j'eusse appelé un peu plus tard (ne peint-elle pas les paysages de la même façon que lui les caractères ?) le côté Dostoïewski des Lettres de Madame de Sévigné.

Quand le soir, après avoir conduit ma grand'mère et être resté quelques heures chez son amie, j'eus repris seul le train, du moins je ne trouvai pas pénible la nuit qui vint ; c'est que je n'avais pas à la passer dans la prison d'une chambre dont l'ensommeillement me tiendrait éveillé ; j'étais entouré par la calmante activité de tous ces mouvements du train qui me tenaient compagnie, s'offraient à causer avec moi si je ne trouvais pas le sommeil, me berçaient de leurs bruits que j'accouplais comme le son des cloches à Combray, tantôt sur un rythme, tantôt sur un autre (entendant selon ma fantaisie d'abord quatre doubles croches égales, puis une double croche furieusement précipitée contre une noire) ; ils neutralisaient la force centrifuge de mon insomnie en exerçant sur elle des pressions contraires qui me maintenaient en équilibre et sur lesquelles mon immobilité et bientôt mon sommeil se sentirent portés avec la même impression rafraîchissante que m'aurait donnée le repos dû à la vigilance de forces puissantes au sein de la nature et de la vie, si j'avais pu pour un moment m'incarner en quelque poisson qui dort dans la mer, promené dans son assoupissement par les courants et la vague, ou en quelque aigle étendu sur le seul appui de la tempête.

Les levers de soleil sont un accompagnement des longs voyages en chemin de fer, comme les oeufs durs, les journaux illustrés, les jeux de cartes, les rivières où des barques s'évertuent sans avancer. À un moment où je dénombrais les pensées qui avaient rempli mon esprit pendant les minutes précédentes, pour me rendre compte si je venais ou non de dormir (et où l'incertitude même qui me faisait me poser la question était en train de me fournir une réponse affirmative), dans le carreau de la fenêtre, au-dessus d'un petit bois noir, je vis des nuages échancrés dont le doux duvet était d'un rose fixé, mort, qui ne changera plus, comme celui qui teint les plumes de l'aile qui l'a assimilé ou le pastel sur lequel l'a déposé la fantaisie du peintre. Mais je sentais qu'au contraire cette couleur n'était ni inertie, ni caprice, mais nécessité et vie. Bientôt s'amoncelèrent derrière elle des réserves de lumière. Elle s'aviva, le ciel devint d'un incarnat que je tâchais, en collant mes yeux à la vitre, de mieux voir, car je le sentais en rapport avec l'existence profonde de la nature, mais la ligne du chemin de fer ayant changé de direction, le train tourna, la scène matinale fut remplacée dans le cadre de la fenêtre par un village nocturne aux toits bleus de clair de lune, avec un lavoir encrassé de la nacre opaline de la nuit, sous un ciel encore semé de toutes ses étoiles, et je me désolais d'avoir perdu ma bande de ciel rose quand je l'aperçus de nouveau, mais rouge cette fois, dans la fenêtre d'en face qu'elle abandonna à un deuxième coude de la voie ferrée ; si bien que je passais mon temps à courir d'une fenêtre à l'autre pour rapprocher, pour rentoiler les fragments intermittents et opposites de mon beau matin écarlate et versatile et en avoir une vue totale et un tableau continu.
