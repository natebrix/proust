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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "les Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« il la forcerait à m'y emmener » … « Swann supprima brusquement pour moi une de ces affreuses distances… J'éprouvai pour lui une tendresse… Car maître de sa fille, il me la donnait »",
      "explanation": "Swann promises to compel Gilberte to include the narrator in the private space, which immediately relieves the narrator and prompts an explicit tender admiration grounded in Swann’s paternal authority and generosity."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.87,
      "explanation": "Swann’s standing in the narrator’s eyes rises due to his benevolent control and readiness to grant access to Gilberte."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-176-p-180"
}

### Candidate characters

[
  "Gilberte",
  "M. Vinteuil",
  "Odette",
  "le narrateur"
]

### Prior local context (optional)

Il me montrait des acquisitions nouvelles qu'il avait faites et m'en expliquait l'intérêt, mais l'émotion, jointe au manque d'habitude d'être encore à jeun à cette heure-là, tout en agitant mon esprit y faisait le vide, de sorte que, capable de parler, je ne l'étais pas d'entendre. D'ailleurs les oeuvres que possédait Swann, il suffisait pour moi qu'elles fussent situées chez lui, y fissent partie de l'heure délicieuse qui précédait le déjeuner. La Joconde se serait trouvée là qu'elle ne m'eût pas fait plus de plaisir qu'une robe de chambre de Odette, ou ses flacons de sel.

### Passage

Je continuais à attendre, seul, ou avec Swann et souvent Gilberte, qui était venue nous tenir compagnie. L'arrivée de Odette, préparée par tant de majestueuses entrées, me paraissait devoir être quelque chose d'immense. J'épiais chaque craquement. Mais on ne trouve jamais aussi hauts qu'on les avait espérés une cathédrale, une vague dans la tempête, le bond d'un danseur ; après ces valets de pied en livrée, pareils aux figurants dont le cortège, au théâtre, prépare, et par là même diminue l'apparition finale de la reine, Odette entrant furtivement en petit paletot de loutre, sa voilette baissée sur un nez rougi par le froid, ne tenait pas les promesses prodiguées dans l'attente à mon imagination.

Mais si elle était restée toute la matinée chez elle, quand elle arrivait dans le salon, c'était vêtue d'un peignoir en crêpe de Chine de couleur claire qui me semblait plus élégant que toutes les robes.

Quelquefois les Swann se décidaient à rester à la maison tout l'après-midi. Et alors, comme on avait déjeuné si tard, je voyais bien vite sur le mur du jardinet décliner le soleil de ce jour qui m'avait paru devoir être différent des autres, et les domestiques avaient beau apporter des lampes de toutes les grandeurs et de toutes les formes, brûlant chacune sur l'autel consacré d'une console, d'un guéridon, d'une « encoignure » ou d'une petite table, comme pour la célébration d'un culte inconnu, rien d'extraordinaire ne naissait de la conversation, et je m'en allais déçu, comme on l'est souvent dès l'enfance après la messe de minuit.

Mais ce désappointement-là n'était guère que spirituel. Je rayonnais de joie dans cette maison où Gilberte, quand elle n'était pas encore avec nous, allait entrer, et me donnerait dans un instant, pour des heures, sa parole, son regard attentif et souriant tel que je l'avais vu pour la première fois à Combray. Tout au plus étais-je un peu jaloux en la voyant souvent disparaître dans de grandes chambres auxquelles on accédait par un escalier intérieur. Obligé de rester au salon, comme l'amoureux d'une actrice qui n'a que son fauteuil à l'orchestre et rêve avec inquiétude de ce qui se passe dans les coulisses, au foyer des artistes, je posai à Swann, au sujet de cette autre partie de la maison, des questions savamment voilées, mais sur un ton duquel je ne parvins pas à bannir quelque anxiété. Il m'expliqua que la pièce où allait Gilberte était la lingerie, s'offrit à me la montrer et me promit que chaque fois que Gilberte aurait à s'y rendre il la forcerait à m'y emmener. Par ces derniers mots et la détente qu'ils me procurèrent, Swann supprima brusquement pour moi une de ces affreuses distances intérieures au terme desquelles une femme que nous aimons nous apparaît si lointaine. À ce moment-là, j'éprouvai pour lui une tendresse que je crus plus profonde que ma tendresse pour Gilberte. Car maître de sa fille, il me la donnait et elle, elle se refusait parfois, je n'avais pas directement sur elle ce même empire qu'indirectement par Swann. Enfin elle, je l'aimais et ne pouvais par conséquent la voir sans ce trouble, sans ce désir de quelque chose de plus, qui ôte, auprès de l'être qu'on aime, la sensation d'aimer.

Au reste, le plus souvent, nous ne restions pas à la maison, nous allions nous promener. Parfois, avant d'aller s'habiller, Odette se mettait au piano. Ses belles mains, sortant des manches roses, ou blanches, souvent de couleurs très vives, de sa robe de chambre de crêpe de Chine, allongeaient leurs phalanges sur le piano avec cette même mélancolie qui était dans ses yeux et n'était pas dans son coeur. Ce fut un de ces jours-là qu'il lui arriva de me jouer la partie de la Sonate de Vinteuil où se trouve la petite phrase que Swann avait tant aimée. Mais souvent on n'entend rien, si c'est une musique un peu compliquée qu'on écoute pour la première fois. Et pourtant quand plus tard on m'eut joué deux ou trois fois cette Sonate, je me trouvai la connaître parfaitement. Aussi n'a-t-on pas tort de dire « entendre pour la première fois ». Si l'on n'avait vraiment, comme on l'a cru, rien distingué à la première audition, la deuxième, la troisième seraient autant de premières, et il n'y aurait pas de raison pour qu'on comprît quelque chose de plus à la dixième. Probablement ce qui fait défaut, la première fois, ce n'est pas la compréhension, mais la mémoire. Car la nôtre, relativement à la complexité des impressions auxquelles elle a à faire face pendant que nous écoutons, est infime, aussi brève que la mémoire d'un homme qui en dormant pense mille choses qu'il oublie aussitôt, ou d'un homme tombé à moitié en enfance qui ne se rappelle pas la minute d'après ce qu'on vient de lui dire. Ces impressions multiples, la mémoire n'est pas capable de nous en fournir immédiatement le souvenir. Mais celui-ci se forme en elle peu à peu et, à l'égard des oeuvres qu'on a entendues deux ou trois fois, on est comme le collégien qui a relu à plusieurs reprises avant de s'endormir une leçon qu'il croyait ne pas savoir et qui la récite par coeur le lendemain matin. Seulement je n'avais encore, jusqu'à ce jour, rien entendu de cette Sonate, et là où Swann et sa femme voyaient une phrase distincte, celle-ci était aussi loin de ma perception claire qu'un nom qu'on cherche à se rappeler et à la place duquel on ne trouve que du néant, un néant d'où une heure plus tard, sans qu'on y pense, s'élanceront d'elles-mêmes, en un seul bond, les syllabes d'abord vainement sollicitées. Et non seulement on ne retient pas tout de suite les oeuvres vraiment rares, mais même au sein de chacune de ces oeuvres-là, et cela m'arriva pour la Sonate de Vinteuil, ce sont les parties les moins précieuses qu'on perçoit d'abord. De sorte que je ne me trompais pas seulement en pensant que l'oeuvre ne me réservait plus rien (ce qui fit que je restai longtemps sans chercher à l'entendre) du moment que Odette m'en avait joué la phrase la plus fameuse (j'étais aussi stupide en cela que ceux qui n'espèrent plus éprouver de surprise devant Saint-Marc de Venise parce que la photographie leur a appris la forme de ses dômes). Mais bien plus, même quand j'eus écouté la Sonate d'un bout à l'autre, elle me resta presque tout entière invisible, comme un monument dont la distance ou la brume ne laissent apercevoir que de faibles parties. De là, la mélancolie qui s'attache à la connaissance de tels ouvrages, comme de tout ce qui se réalise dans le temps. Quand ce qui est le plus caché dans la Sonate de Vinteuil se découvrit à moi, déjà entraîné par l'habitude hors des prises de ma sensibilité, ce que j'avais distingué, préféré tout d'abord, commençait à m'échapper, à me fuir. Pour n'avoir pu aimer qu'en des temps successifs tout ce que m'apportait cette Sonate, je ne la possédai jamais tout entière : elle ressemblait à la vie. Mais, moins décevants que la vie, ces grands chefs-d'oeuvre ne commencent pas par nous donner ce qu'ils ont de meilleur. Dans la Sonate de Vinteuil, les beautés qu'on découvre le plus tôt sont aussi celles dont on se fatigue le plus vite et pour la même raison sans doute, qui est qu'elles diffèrent moins de ce qu'on connaissait déjà. Mais quand celles-là se sont éloignées, il nous reste à aimer telle phrase que son ordre trop nouveau pour offrir à notre esprit rien que confusion nous avait rendue indiscernable et gardée intacte ; alors elle devant qui nous passions tous les jours sans le savoir et qui s'était réservée, qui pour le pouvoir de sa seule beauté était devenue invisible et restée inconnue, elle vient à nous la dernière. Mais nous la quitterons aussi en dernier. Et nous l'aimerons plus longtemps que les autres, parce que nous aurons mis plus longtemps à l'aimer. Ce temps du reste qu'il faut à un individu – comme il me le fallut à moi à l'égard de cette Sonate – pour pénétrer une oeuvre un peu profonde, n'est que le raccourci et comme le symbole des années, des siècles parfois, qui s'écoulent avant que le public puisse aimer un chef-d'oeuvre vraiment nouveau. Aussi l'homme de génie pour s'épargner les méconnaissances de la foule se dit peut-être que les contemporains manquant du recul nécessaire, les oeuvres écrites pour la postérité ne devraient être lues que par elle, comme certaines peintures qu'on juge mal de trop près. Mais en réalité toute lâche précaution pour éviter les faux arguments est inutile, ils ne sont pas évitables. Ce qui est cause qu'une oeuvre de génie est difficilement admirée tout de suite, c'est que celui qui l'a écrite est extraordinaire, que peu de gens lui ressemblent. C'est son oeuvre elle-même qui, en fécondant les rares esprits capables de la comprendre, les fera croître et multiplier. Ce sont les quatuors de Beethoven (les quatuors XII, XIII, XIV et XV) qui ont mis cinquante ans à faire naître, à grossir le public des quatuors de Beethoven, réalisant ainsi comme tous les chefs-d'oeuvre un progrès sinon dans la valeur des artistes, du moins dans la société des esprits, largement composée aujourd'hui de ce qui était introuvable quand le chef-d'oeuvre parut, c'est-à-dire d'êtres capables de l'aimer. Ce qu'on appelle la postérité, c'est la postérité de l'oeuvre. Il faut que l'oeuvre (en ne tenant pas compte, pour simplifier, des génies qui à la même époque peuvent parallèlement préparer pour l'avenir un public meilleur dont d'autres génies que lui bénéficieront) crée elle-même sa postérité. Si donc l'oeuvre était tenue en réserve, n'était connue que de la postérité, celle-ci, pour cette oeuvre, ne serait pas la postérité mais une assemblée de contemporains ayant simplement vécu cinquante ans plus tard. Aussi faut-il que l'artiste – et c'est ce qu'avait fait Vinteuil – s'il veut que son oeuvre puisse suivre sa route, la lance, là où il y a assez de profondeur, en plein et lointain avenir. Et pourtant ce temps à venir, vraie perspective des chefs-d'oeuvre, si n'en pas tenir compte est l'erreur des mauvais juges, en tenir compte est parfois le dangereux scrupule des bons. Sans doute, il est aisé de s'imaginer, dans une illusion analogue à celle qui uniformise toutes choses à l'horizon, que toutes les révolutions qui ont eu lieu jusqu'ici dans la peinture ou la musique respectaient tout de même certaines règles et que ce qui est immédiatement devant nous, impressionnisme, recherche de la dissonance, emploi exclusif de la gamme chinoise, cubisme, futurisme, diffère outrageusement de ce qui a précédé. C'est que ce qui a précédé, on le considère sans tenir compte qu'une longue assimilation l'a converti pour nous en une matière variée sans doute, mais somme toute homogène, où Hugo voisine avec Molière. Songeons seulement aux choquants disparates que nous présenterait, si nous ne tenions pas compte du temps à venir et des changements qu'il amène, tel horoscope de notre propre âge mûr tiré devant nous durant notre adolescence. Seulement tous les horoscopes ne sont pas vrais, et être obligé pour une oeuvre d'art de faire entrer dans le total de sa beauté le facteur du temps mêle à notre jugement quelque chose d'aussi hasardeux et par là aussi dénué d'intérêt véritable, que toute prophétie dont la non-réalisation n'impliquera nullement la médiocrité d'esprit du prophète, car ce qui appelle à l'existence les possibles ou les en exclut n'est pas forcément de la compétence du génie ; on peut en avoir eu et ne pas avoir cru à l'avenir des chemins de fer, ni des avions, ou, tout en étant grand psychologue, à la fausseté d'une maîtresse ou d'un ami, dont de plus médiocres eussent prévu les trahisons.
