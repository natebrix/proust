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
      "canonical_name": "Mme de Villeparisis",
      "surface_forms": [
        "Mme de Villeparisis",
        "marquise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme de Villeparisis",
      "type": "narrated_elevation",
      "polarity": "mixed",
      "narrative_stance": "ironized",
      "confidence": 0.76,
      "evidence": "« Nous fûmes étonnés… de voir combien elle était plus “libérale”… Elle défendait la République… “Oh ! la noblesse aujourd’hui, qu’est-ce que c’est !” “Pour moi, un homme qui ne travaille pas, ce n’est rien”, peut-être seulement parce qu’elle sentait ce qu’ils prenaient de piquant… »",
      "explanation": "The narrator highlights Mme de Villeparisis’s unexpectedly liberal, Republican-friendly remarks, which raise her standing in the eyes of the narrator and his grandmother, while hinting these poses may be delivered for their piquant effect."
    }
  ],
  "status_effects": [
    {
      "character": "Mme de Villeparisis",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.76,
      "explanation": "Locally she is appraised more favorably for her liberal, demotic stances, even if the narrator suggests some performative intent."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-121-p-125"
}

### Candidate characters

[
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

À côté des voitures, devant le porche où j'attendais, était planté comme un arbrisseau d'une espèce rare un jeune chasseur qui ne frappait pas moins les yeux par l'harmonie singulière de ses cheveux colorés, que par son épiderme de plante. À l'intérieur dans le hall qui correspondait au narthex ou église des Catéchumènes, des églises romanes, et où les personnes qui n'habitaient pas l'hôtel avaient le droit de passer, les camarades du groom « extérieur » ne travaillaient pas beaucoup plus que lui mais exécutaient du moins quelques mouvements. Il est probable que le matin ils aidaient au nettoyage. Mais l'après-midi ils restaient là seulement comme des choristes qui, même quand ils ne servent à rien, demeurent en scène pour ajouter à la figuration. Le Directeur général, celui qui me faisait si peur, comptait augmenter considérablement leur nombre l'année suivante, car il « voyait grand ». Et sa décision affligeait beaucoup le directeur de l'Hôtel, lequel trouvait que tous ces enfants n'étaient que des « faiseurs d'embarras » entendant par là qu'ils embarrassaient le passage et ne servaient à rien. Du moins entre le déjeuner et le dîner, entre les sorties et les rentrées des clients remplissaient-ils le vide de l'action comme ces élèves de Mme de Maintenon qui sous le costume de jeunes israélites font intermède chaque fois qu'Esther ou Joad s'en vont. Mais le chasseur du dehors, aux nuances précieuses, à la taille élancée et frêle, non loin duquel j'attendais que la marquise descendît, gardait une immobilité à laquelle s'ajoutait de la mélancolie, car ses frères aînés avaient quitté l'hôtel pour des destinées plus brillantes et il se sentait isolé sur cette terre étrangère. Enfin Mme de Villeparisis arrivait. S'occuper de sa voiture et l'y faire monter eût peut-être dû faire partie des fonctions du chasseur. Mais il savait qu'une personne qui amène ses gens avec soi se fait servir par eux, et d'habitude donne peu de pourboires dans un hôtel, que les nobles de l'ancien faubourg Saint-Germain agissent de même. Mme de Villeparisis appartenait à la fois à ces deux catégories. Le chasseur arborescent en concluait qu'il n'avait rien à attendre de la marquise ; en laissant le maître d'hôtel et la femme de chambre de celle-ci l'installer avec ses affaires, il rêvait tristement au sort envié de ses frères et conservait son immobilité végétale.

### Passage

Nous partions ; quelque temps après avoir contourné la station du chemin de fer nous entrions dans une route campagnarde qui me devint bientôt aussi familière que celles de Combray, depuis le coude où elle s'amorçait entre des clos charmants jusqu'au tournant où nous la quittions et qui avait de chaque côté des terres labourées. Au milieu d'elles, on voyait çà et là un pommier, privé il est vrai de ses fleurs et ne portant plus qu'un bouquet de pistils, mais qui suffisait à m'enchanter parce que je reconnaissais ces feuilles inimitables dont la large étendue, comme le tapis d'estrade d'une fête nuptiale maintenant terminée, avait été tout récemment foulée par la traîne de satin blanc des fleurs rougissantes.

Combien de fois à Paris, dans le mois de mai de l'année suivante, il m'arriva d'acheter une branche de pommier chez le fleuriste et de passer ensuite la nuit devant ses fleurs où s'épanouissait la même essence crémeuse qui poudrait encore de son écume les bourgeons des feuilles et entre les blanches corolles desquelles il semblait que ce fût le marchand qui, par générosité envers moi, par goût inventif aussi et contraste ingénieux eût ajouté de chaque côté, en surplus, un seyant bouton rose ; je les regardais, je les faisais poser sous ma lampe – si longtemps que j'étais souvent encore là quand l'aurore leur apportait la même rougeur qu'elle devait faire en même temps à Balbec – et je cherchais à les reporter sur cette route par l'imagination, à les multiplier, à les étendre dans le cadre préparé, sur la toile toute prête de ces clos dont je savais le dessin par coeur – et que j'aurais tant voulu, qu'un jour je devais revoir – au moment où avec la verve ravissante du génie, le printemps couvre leur canevas de ses couleurs.

Avant de monter en voiture, j'avais composé le tableau de mer que j'allais chercher, que j'espérais voir avec le « soleil rayonnant », et qu'à Balbec je n'apercevais que trop morcelé entre tant d'enclaves vulgaires et que mon rêve n'admettait pas, de baigneurs, de cabines, de yacht de plaisance. Mais quand, la voiture de Mme de Villeparisis étant parvenue au haut d'une côte, j'apercevais la mer entre les feuillages des arbres, alors sans doute de si loin disparaissaient ces détails contemporains qui l'avaient mise comme en dehors de la nature et de l'histoire, et je pouvais en regardant les flots m'efforcer de penser que c'était les mêmes que Leconte de Lisle nous peint dans l'Orestie quand « tel qu'un vol d'oiseaux carnassiers dans l'aurore » les guerriers chevelus de l'héroïque Hellas « de cent mille avirons battaient le flot sonore ». Mais en revanche je n'étais plus assez près de la mer qui ne me semblait pas vivante, mais figée, je ne sentais plus de puissance sous ses couleurs étendues comme celles d'une peinture entre les feuilles où elle apparaissait aussi inconsistante que le ciel, et seulement plus foncée que lui.

Mme de Villeparisis voyant que j'aimais les églises me promettait que nous irions voir une fois l'une, une fois l'autre, et surtout celle de Carqueville « toute cachée sous son vieux lierre », dit-elle avec un mouvement de la main qui semblait envelopper avec goût la façade absente dans un feuillage invisible et délicat. Mme de Villeparisis avait souvent, avec ce petit geste descriptif, un mot juste pour définir le charme et la particularité d'un monument, évitant toujours les termes techniques, mais ne pouvant dissimuler qu'elle savait très bien les choses dont elle parlait. Elle semblait chercher à s'en excuser sur ce qu'un des châteaux de son père, et où elle avait été élevée, étant situé dans une région où il y avait des églises du même style qu'autour de Balbec il eût été honteux qu'elle n'eût pas pris le goût de l'architecture, ce château étant d'ailleurs le plus bel exemplaire de celle de la Renaissance. Mais comme il était aussi un vrai musée, comme d'autre part Chopin et Liszt y avaient joué, Lamartine récité des vers, tous les artistes connus de tout un siècle écrit des pensées, des mélodies, fait des croquis sur l'album familial, Mme de Villeparisis ne donnait, par grâce, bonne éducation, modestie réelle, ou manque d'esprit philosophique, que cette origine purement matérielle à sa connaissance de tous les arts, et finissait par avoir l'air de considérer la peinture, la musique, la littérature et la philosophie comme l'apanage d'une jeune fille élevée de la façon la plus aristocratique dans un monument classé et illustre. On aurait dit qu'il n'y avait pas pour elle d'autres tableaux que ceux dont on a hérités. Elle fut contente que ma grand'mère aimât un collier qu'elle portait et qui dépassait de sa robe. Il était dans le portrait d'une bisaïeule à elle, par Titien, et qui n'était jamais sorti de la famille. Comme cela on était sûr que c'était un vrai. Elle ne voulait pas entendre parler des tableaux achetés on ne sait comment par un Crésus, elle était d'avance persuadée qu'ils étaient faux et n'avait aucun désir de les voir, nous savions qu'elle-même faisait des aquarelles de fleurs, et ma grand'mère qui les avait entendu vanter lui en parla. Mme de Villeparisis changea de conversation par modestie, mais sans montrer plus d'étonnement ni de plaisir qu'une artiste suffisamment connue à qui les compliments n'apprennent rien. Elle se contenta de dire que c'était un passe-temps charmant parce que si les fleurs nées du pinceau n'étaient pas fameuses, du moins les peindre vous faisait vivre dans la société des fleurs naturelles, de la beauté desquelles, surtout quand on était obligé de les regarder de plus près pour les imiter, on ne se lassait pas. Mais à Balbec Mme de Villeparisis se donnait congé pour laisser reposer ses yeux.

Nous fûmes étonnés, ma grand'mère et moi, de voir combien elle était plus « libérale » que même la plus grande partie de la bourgeoisie. Elle s'étonnait qu'on fût scandalisé des expulsions des jésuites, disant que cela s'était toujours fait, même sous la monarchie, même en Espagne. Elle défendait la République à laquelle elle ne reprochait son anticléricalisme que dans cette mesure : « Je trouverais tout aussi mauvais qu'on m'empêchât d'aller à la messe si j'en ai envie que d'être forcée d'y aller si je ne le veux pas », lançant même certains mots comme : « Oh ! la noblesse aujourd'hui, qu'est-ce que c'est ! » « Pour moi, un homme qui ne travaille pas, ce n'est rien », peut-être seulement parce qu'elle sentait ce qu'ils prenaient de piquant, de savoureux, de mémorable dans sa bouche.
