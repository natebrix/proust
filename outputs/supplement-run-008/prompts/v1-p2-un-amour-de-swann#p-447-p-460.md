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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "duchesse de Guermantes",
      "surface_forms": [
        "princesse des Laumes"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "marquise de Gallardon",
      "surface_forms": [
        "marquise de Gallardon"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "duchesse de Guermantes",
      "target": "marquise de Gallardon",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« La princesse des Laumes éclatant d’un rire … destiné à … montrer aux autres qu’elle se moquait de quelqu’un »; puis elle regarde « d’un air étonné et rieur » lorsqu’elle est appelée par son prénom et répond à l’invitation par des excuses conditionnelles.",
      "explanation": "The princess publicly ridicules Gallardon with a mocking laugh, marks distance regarding the use of the first name, then gives an evasive response to the invitation, which locally functions as a social keeping at a distance."
    }
  ],
  "status_effects": [
    {
      "character": "marquise de Gallardon",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Mocked and kept at a distance by the princess, she undergoes a clear loss of face in the scene."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-447-p-460"
}

### Candidate characters

[
  "Mme de Cambremer",
  "Swann",
  "le narrateur",
  "le pianiste",
  "marquise de Saint-Euverte"
]

### Prior local context (optional)

Le monocle du marquis de Forestelle était minuscule, n'avait aucune bordure et, obligeant à une crispation incessante et douloureuse l'oeil où il s'incrustait comme un cartilage superflu dont la présence est inexplicable et la matière recherchée, il donnait au visage du marquis une délicatesse mélancolique, et le faisait juger par les femmes comme capable de grands chagrins d'amour. Mais celui de M. de Saint-Candé, entouré d'un gigantesque anneau, comme Saturne, était le centre de gravité d'une figure qui s'ordonnait à tout moment par rapport à lui, dont le nez frémissant et rouge et la bouche lippue et sarcastique tâchaient par leurs grimaces d'être à la hauteur des feux roulants d'esprit dont étincelait le disque de verre, et se voyait préférer aux plus beaux regards du monde par des jeunes femmes snobs et dépravées qu'il faisait rêver de charmes artificiels et d'un raffinement de volupté ; et cependant, derrière le sien, M. de Palancy qui, avec sa grosse tête de carpe aux yeux ronds, se déplaçait lentement au milieu des fêtes en desserrant d'instant en instant ses mandibules comme pour chercher son orientation, avait l'air de transporter seulement avec lui un fragment accidentel, et peut-être purement symbolique, du vitrage de son aquarium, partie destinée à figurer le tout qui rappela à Swann, grand admirateur des Vices et des Vertus de Giotto à Padoue, cet Injuste à côté duquel un rameau feuillu évoque les forêts où se cache son repaire.

### Passage

Swann s'était avancé, sur l'insistance de Mme de Saint-Euverte et pour entendre un air d'Orphée qu'exécutait un flûtiste, s'était mis dans un coin où il avait malheureusement comme seule perspective deux dames déjà mûres assises l'une à côté de l'autre, la marquise de Cambremer et la vicomtesse de Franquetot, lesquelles, parce qu'elles étaient cousines, passaient leur temps dans les soirées, portant leurs sacs et suivies de leurs filles, à se chercher comme dans une gare et n'étaient tranquilles que quand elles avaient marqué, par leur éventail ou leur mouchoir, deux places voisines : Mme de Cambremer, comme elle avait très peu de relations, étant d'autant plus heureuse d'avoir une compagne, Mme de Franquetot, qui était au contraire très lancée, trouvait quelque chose d'élégant, d'original, à montrer à toutes ses belles connaissances qu'elle leur préférait une dame obscure avec qui elle avait en commun des souvenirs de jeunesse. Plein d'une mélancolique ironie, Swann les regardait écouter l'intermède de piano (« Saint François parlant aux oiseaux », de Liszt) qui avait succédé à l'air de flûte, et suivre le jeu vertigineux du virtuose, Mme de Franquetot anxieusement, les yeux éperdus comme si les touches sur lesquelles il courait avec agilité avaient été une suite de trapèzes d'où il pouvait tomber d'une hauteur de quatre-vingts mètres, et non sans lancer à sa voisine des regards d'étonnement, de dénégation qui signifiaient : « Ce n'est pas croyable, je n'aurais jamais pensé qu'un homme pût faire cela », Mme de Cambremer, en femme qui a reçu une forte éducation musicale, battant la mesure avec sa tête transformée en balancier de métronome dont l'amplitude et la rapidité d'oscillations d'une épaule à l'autre étaient devenues telles (avec cette espèce d'égarement et d'abandon du regard qu'ont les douleurs qui ne se connaissent plus ni ne cherchent à se maîtriser et disent : « Que voulez-vous ! ») qu'à tout moment elle accrochait avec ses solitaires les pattes de son corsage et était obligée de redresser les raisins noirs qu'elle avait dans les cheveux, sans cesser pour cela d'accélérer le mouvement. De l'autre côté de Mme de Franquetot, mais un peu en avant, était la marquise de Gallardon, occupée à sa pensée favorite, l'alliance qu'elle avait avec les Guermantes et d'où elle tirait pour le monde et pour elle-même beaucoup de gloire avec quelque honte, les plus brillants d'entre eux la tenant un peu à l'écart, peut-être parce qu'elle était ennuyeuse, ou parce qu'elle était méchante, ou parce qu'elle était d'une branche inférieure, ou peut-être sans aucune raison. Quand elle se trouvait auprès de quelqu'un qu'elle ne connaissait pas, comme en ce moment auprès de Mme de Franquetot, elle souffrait que la conscience qu'elle avait de sa parenté avec les Guermantes ne pût se manifester extérieurement en caractères visibles comme ceux qui, dans les mosaïques des églises byzantines, placés les uns au-dessous des autres, inscrivent en une colonne verticale, à côté d'un Saint Personnage, les mots qu'il est censé prononcer. Elle songeait en ce moment qu'elle n'avait jamais reçu une invitation ni une visite de sa jeune cousine la princesse des Laumes, depuis six ans que celle-ci était mariée. Cette pensée la remplissait de colère, mais aussi de fierté ; car, à force de dire aux personnes qui s'étonnaient de ne pas la voir chez Mme des Laumes, que c'est parce qu'elle aurait été exposée à y rencontrer la princesse Mathilde – ce que sa famille ultra-légitimiste ne lui aurait jamais pardonné – elle avait fini par croire que c'était en effet la raison pour laquelle elle n'allait pas chez sa jeune cousine. Elle se rappelait pourtant qu'elle avait demandé plusieurs fois à Mme des Laumes comment elle pourrait faire pour la rencontrer, mais ne se le rappelait que confusément et d'ailleurs neutralisait et au delà ce souvenir un peu humiliant en murmurant : « Ce n'est tout de même pas à moi à faire les premiers pas, j'ai vingt ans de plus qu'elle. » Grâce à la vertu de ces paroles intérieures, elle rejetait fièrement en arrière ses épaules détachées de son buste et sur lesquelles sa tête posée presque horizontalement faisait penser à la tête « rapportée » d'un orgueilleux faisan qu'on sert sur une table avec toutes ses plumes. Ce n'est pas qu'elle ne fût par nature courtaude, hommasse et boulotte ; mais les camouflets l'avaient redressée comme ces arbres qui, nés dans une mauvaise position au bord d'un précipice, sont forcés de croître en arrière pour garder leur équilibre. Obligée, pour se consoler de ne pas être tout à fait l'égale des autres Guermantes, de se dire sans cesse que c'était par intransigeance de principes et fierté qu'elle les voyait peu, cette pensée avait fini par modeler son corps et par lui enfanter une sorte de prestance qui passait aux yeux des bourgeoises pour un signe de race et troublait quelquefois d'un désir fugitif le regard fatigué des hommes de cercle. Si on avait fait subir à la conversation de Mme de Gallardon ces analyses qui en relevant la fréquence plus ou moins grande de chaque terme permettent de découvrir la clef d'un langage chiffré, on se fût rendu compte qu'aucune expression, même la plus usuelle, n'y revenait aussi souvent que « chez mes cousins de Guermantes », « chez ma tante de Guermantes », « la santé d'Elzéar de Guermantes », « la baignoire de ma cousine de Guermantes ». Quand on lui parlait d'un personnage illustre, elle répondait que, sans le connaître personnellement, elle l'avait rencontré mille fois chez sa tante de Guermantes, mais elle répondait cela d'un ton si glacial et d'une voix si sourde qu'il était clair que si elle ne le connaissait pas personnellement, c'était en vertu de tous les principes indéracinables et entêtés auxquels ses épaules touchaient en arrière, comme à ces échelles sur lesquelles les professeurs de gymnastique vous font étendre pour vous développer le thorax.

Or, la princesse des Laumes, qu'on ne se serait pas attendu à voir chez Mme de Saint-Euverte, venait précisément d'arriver. Pour montrer qu'elle ne cherchait pas à faire sentir dans un salon, où elle ne venait que par condescendance, la supériorité de son rang, elle était entrée en effaçant les épaules là même où il n'y avait aucune foule à fendre et personne à laisser passer, restant exprès dans le fond, de l'air d'y être à sa place, comme un roi qui fait la queue à la porte d'un théâtre tant que les autorités n'ont pas été prévenues qu'il est là ; et, bornant simplement son regard – pour ne pas avoir l'air de signaler sa présence et de réclamer des égards – à la considération d'un dessin du tapis ou de sa propre jupe, elle se tenait debout à l'endroit qui lui avait paru le plus modeste (et d'où elle savait bien qu'une exclamation ravie de Mme de Saint-Euverte allait la tirer dès que celle-ci l'aurait aperçue), à côté de Mme de Cambremer qui lui était inconnue. Elle observait la mimique de sa voisine mélomane, mais ne l'imitait pas. Ce n'est pas que, pour une fois qu'elle venait passer cinq minutes chez Mme de Saint-Euverte, la princesse des Laumes n'eût souhaité, pour que la politesse qu'elle lui faisait comptât double, se montrer le plus aimable possible. Mais par nature, elle avait horreur de ce qu'elle appelait « les exagérations » et tenait à montrer qu'elle « n'avait pas à » se livrer à des manifestations qui n'allaient pas avec le « genre » de la coterie où elle vivait, mais qui pourtant d'autre part ne laissaient pas de l'impressionner, à la faveur de cet esprit d'imitation voisin de la timidité que développe, chez les gens les plus sûrs d'eux-mêmes, l'ambiance d'un milieu nouveau, fût-il inférieur. Elle commençait à se demander si cette gesticulation n'était pas rendue nécessaire par le morceau qu'on jouait et qui ne rentrait peut-être pas dans le cadre de la musique qu'elle avait entendue jusqu'à ce jour, si s'abstenir n'était pas faire preuve d'incompréhension à l'égard de l'oeuvre et d'inconvenance vis-à-vis de la maîtresse de la maison : de sorte que pour exprimer par une « cote mal taillée » ses sentiments contradictoires, tantôt elle se contentait de remonter la bride de ses épaulettes ou d'assurer dans ses cheveux blonds les petites boules de corail ou d'émail rose, givrées de diamant, qui lui faisaient une coiffure simple et charmante, en examinant avec une froide curiosité sa fougueuse voisine, tantôt de son éventail elle battait pendant un instant la mesure, mais, pour ne pas abdiquer son indépendance, à contretemps.

Le pianiste ayant terminé le morceau de Liszt et ayant commencé un prélude de Chopin, Mme de Cambremer lança à Mme de Franquetot un sourire attendri de satisfaction compétente et d'allusion au passé. Elle avait appris dans sa jeunesse à caresser les phrases, au long col sinueux et démesuré, de Chopin, si libres, si flexibles, si tactiles, qui commencent par chercher et essayer leur place en dehors et bien loin de la direction de leur départ, bien loin du point où on avait pu espérer qu'atteindrait leur attouchement, et qui ne se jouent dans cet écart de fantaisie que pour revenir plus délibérément – d'un retour plus prémédité, avec plus de précision, comme sur un cristal qui résonnerait jusqu'à faire crier – vous frapper au coeur.

Vivant dans une famille provinciale qui avait peu de relations, n'allant guère au bal, elle s'était grisée dans la solitude de son manoir, à ralentir, à précipiter la danse de tous ces couples imaginaires, à les égrener comme des fleurs, à quitter un moment le bal pour entendre le vent souffler dans les sapins, au bord du lac, et à y voir tout d'un coup s'avancer, plus différent de tout ce qu'on a jamais rêvé que ne sont les amants de la terre, un mince jeune homme à la voix un peu chantante, étrangère et fausse, en gants blancs. Mais aujourd'hui la beauté démodée de cette musique semblait défraîchie. Privée depuis quelques années de l'estime des connaisseurs, elle avait perdu son honneur et son charme et ceux mêmes dont le goût est mauvais n'y trouvaient plus qu'un plaisir inavoué et médiocre. Mme de Cambremer jeta un regard furtif derrière elle. Elle savait que sa jeune bru (pleine de respect pour sa nouvelle famille, sauf en ce qui touchait les choses de l'esprit sur lesquelles, sachant jusqu'à l'harmonie et jusqu'au grec, elle avait des lumières spéciales) méprisait Chopin et souffrait quand elle en entendait jouer. Mais loin de la surveillance de cette wagnérienne qui était plus loin avec un groupe de personnes de son âge, Mme de Cambremer se laissait aller à des impressions délicieuses. La princesse des Laumes les éprouvait aussi. Sans être par nature douée pour la musique, elle avait reçu il y a quinze ans les leçons qu'un professeur de piano du faubourg Saint-Germain, femme de génie qui avait été à la fin de sa vie réduite à la misère, avait recommencé, à l'âge de soixante-dix ans, à donner aux filles et aux petites-filles de ses anciennes élèves. Elle était morte aujourd'hui. Mais sa méthode, son beau son, renaissaient parfois sous les doigts de ses élèves, même de celles qui étaient devenues pour le reste des personnes médiocres, avaient abandonné la musique et n'ouvraient presque plus jamais un piano. Aussi Mme des Laumes put-elle secouer la tête, en pleine connaissance de cause, avec une appréciation juste de la façon dont le pianiste jouait ce prélude qu'elle savait par coeur. La fin de la phrase commencée chanta d'elle-même sur ses lèvres. Et elle murmura « c'est toujours charmant », avec un double ch au commencement du mot qui était une marque de délicatesse et dont elle sentait ses lèvres si romanesquement froissées comme une belle fleur, qu'elle harmonisa instinctivement son regard avec elles en lui donnant à ce moment-là une sorte de sentimentalité et de vague. Cependant Mme de Gallardon était en train de se dire qu'il était fâcheux qu'elle n'eût que bien rarement l'occasion de rencontrer la princesse des Laumes, car elle souhaitait lui donner une leçon en ne répondant pas à son salut. Elle ne savait pas que sa cousine fût là. Un mouvement de tête de Mme de Franquetot la lui découvrit. Aussitôt elle se précipita vers elle en dérangeant tout le monde ; mais désireuse de garder un air hautain et glacial qui rappelât à tous qu'elle ne désirait pas avoir de relations avec une personne chez qui on pouvait se trouver nez à nez avec la princesse Mathilde, et au-devant de qui elle n'avait pas à aller car elle n'était pas « sa contemporaine », elle voulut pourtant compenser cet air de hauteur et de réserve par quelque propos qui justifiât sa démarche et forçât la princesse à engager la conversation ; aussi une fois arrivée près de sa cousine, Mme de Gallardon, avec un visage dur, une main tendue comme une carte forcée, lui dit : « Comment va ton mari ? » de la même voix soucieuse que si le prince avait été gravement malade. La princesse éclatant d'un rire qui lui était particulier et qui était destiné à la fois à montrer aux autres qu'elle se moquait de quelqu'un et aussi à se faire paraître plus jolie en concentrant les traits de son visage autour de sa bouche animée et de son regard brillant, lui répondit :

– Mais le mieux du monde !

Et elle rit encore. Cependant tout en redressant sa taille et refroidissant sa mine, inquiète encore pourtant de l'état du prince, Mme de Gallardon dit à sa cousine :

– Mme de Guermantes (ici Mme des Laumes regarda d'un air étonné et rieur un tiers invisible vis-à-vis duquel elle semblait tenir à attester qu'elle n'avait jamais autorisé Mme de Gallardon à l'appeler par son prénom), je tiendrais beaucoup à ce que tu viennes un moment demain soir chez moi entendre un quintette avec clarinette de Mozart. Je voudrais avoir ton appréciation.

Elle semblait non pas adresser une invitation, mais demander un service, et avoir besoin de l'avis de la princesse sur le quintette de Mozart, comme si ç'avait été un plat de la composition d'une nouvelle cuisinière sur les talents de laquelle il lui eût été précieux de recueillir l'opinion d'un gourmet.

– Mais je connais ce quintette, je peux te dire tout de suite... que je l'aime !

– Tu sais, mon mari n'est pas bien, son foie..., cela lui ferait grand plaisir de te voir, reprit Mme de Gallardon, faisant maintenant à la princesse une obligation de charité de paraître à sa soirée.

La princesse n'aimait pas à dire aux gens qu'elle ne voulait pas aller chez eux. Tous les jours elle écrivait son regret d'avoir été privée – par une visite inopinée de sa belle-mère, par une invitation de son beau-frère, par l'Opéra, par une partie de campagne – d'une soirée à laquelle elle n'aurait jamais songé à se rendre. Elle donnait ainsi à beaucoup de gens la joie de croire qu'elle était de leurs relations, qu'elle eût été volontiers chez eux, qu'elle n'avait été empêchée de le faire que par les contretemps princiers qu'ils étaient flattés de voir entrer en concurrence avec leur soirée. Puis faisant partie de cette spirituelle coterie des Guermantes où survivait quelque chose de l'esprit alerte, dépouillé de lieux communs et de sentiments convenus, qui descend de Mérimée – et a trouvé sa dernière expression dans le théâtre de Meilhac et Halévy – elle l'adaptait même aux rapports sociaux, le transposait jusque dans sa politesse qui s'efforçait d'être positive, précise, de se rapprocher de l'humble vérité. Elle ne développait pas longuement à une maîtresse de maison l'expression du désir qu'elle avait d'aller à sa soirée ; elle trouvait plus aimable de lui exposer quelques petits faits d'où dépendrait qu'il lui fût ou non possible de s'y rendre.

– Ecoute, je vais te dire, dit-elle à Mme de Gallardon, il faut demain soir que j'aille chez une amie qui m'a demandé mon jour depuis longtemps. Si elle nous emmène au théâtre, il n'y aura pas, avec la meilleure volonté, possibilité que j'aille chez toi ; mais si nous restons chez elle, comme je sais que nous serons seuls, je pourrai la quitter.

– Tiens, tu as vu ton ami Swann ?

– Mais non, cet amour de Swann, je ne savais pas qu'il fût là, je vais tâcher qu'il me voie.
