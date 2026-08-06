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
      "canonical_name": "Bloch père",
      "surface_forms": [
        "Bloch père",
        "père Bloch"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bloch père",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.91,
      "evidence": "Il vivait dans le monde des à peu près, où l'on salue dans le vide, où l'on juge dans le faux... L'inexactitude, l'incompétence, n'y diminuent pas l'assurance, au contraire... « Ce Bergotte est devenu illisible... quelle tartine ! »",
      "explanation": "The narrator presents Bloch père as incompetent but self-assured, judging without knowledge and giving himself the luxury of condemning Bergotte with arrogance; this depiction clearly aims to belittle him."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch père",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Locally discredited as a presumptuous and ill-informed judge, culminating in his disparagements of Bergotte."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-261-p-265"
}

### Candidate characters

[
  "Bergotte",
  "Bloch",
  "Robert de Saint-Loup",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Mourir la pâle Adriatique.

### Passage

Or, de quelqu'un qu'on admire de confiance, on recueille, on cite avec admiration, des choses très inférieures à celles que livré à son propre génie on refuserait avec sévérité, de même qu'un écrivain utilise dans un roman, sous prétexte qu'ils sont vrais, des « mots », des personnages, qui dans l'ensemble vivant font au contraire poids mort, partie médiocre. Les portraits de Saint Simon écrits par lui sans qu'il s'admire sans doute, sont admirables, les traits qu'il cite comme charmants de gens d'esprit qu'il a connus sont restés médiocres ou devenus incompréhensibles. Il eût dédaigné d'inventer ce qu'il rapporte comme si fin ou si coloré de Mme Cornuel ou de Louis XIV, fait qui du reste est à noter chez bien d'autres et comporte diverses interprétations dont il suffit en ce moment de retenir celle-ci : c'est que dans l'état d'esprit où l'on « observe », on est très au-dessous du niveau où l'on se trouve quand on crée.

Il y avait donc, enclavé en mon camarade Bloch, un père Bloch, qui retardait de quarante ans sur son fils, débitait des anecdotes saugrenues, et en riait autant au fond de mon ami que ne faisait le père Bloch extérieur et véritable, puisque au rire que ce dernier lâchait non sans répéter deux ou trois fois le dernier mot, pour que son public goûtât bien l'histoire, s'ajoutait le rire bruyant par lequel le fils ne manquait pas à table de saluer les histoires de son père. C'est ainsi qu'après avoir dit les choses les plus intelligentes, Bloch jeune, manifestant l'apport qu'il avait reçu de sa famille, nous racontait pour la trentième fois quelques-uns des mots que le père Bloch sortait seulement (en même temps que sa redingote) les jours solennels où Bloch jeune amenait quelqu'un qu'il valait la peine d'éblouir : un de ses professeurs, un « copain » qui avait tous les prix, ou, ce soir-là, Saint-Loup et moi. Par exemple : « Un critique militaire très fort, qui avait savamment déduit avec preuves à l'appui pour quelles raisons infaillibles dans la guerre russo-japonaise, les Japonais seraient battus et les Russes vainqueurs », ou bien : « C'est un homme éminent qui passe pour un grand financier dans les milieux politiques et pour un grand politique dans les milieux financiers. » Ces histoires étaient interchangeables avec une du baron de Rothschild et une de sir Rufus Israël, personnages mis en scène d'une manière équivoque qui pouvait donner à entendre que Bloch les avait personnellement connus.

J'y fus moi-même pris et à la manière dont Bloch père parla de Bergotte, je crus aussi que c'était un de ses vieux amis. Or, tous les gens célèbres, Bloch ne les connaissait que « sans les connaître », pour les avoir vus de loin au théâtre, sur les boulevards. Il s'imaginait du reste que sa propre figure, son nom, sa personnalité ne leur étaient pas inconnus et qu'en l'apercevant, ils étaient souvent obligés de retenir une furtive envie de le saluer. Les gens du monde, parce qu'ils connaissent les gens de talent original, qu'ils les reçoivent à dîner, ne les comprennent pas mieux pour cela. Mais quand on a un peu vécu dans le monde, la sottise de ses habitants vous fait trop souhaiter de vivre, trop supposer d'intelligence, dans les milieux obscurs où l'on ne connaît que « sans connaître ». J'allais m'en rendre compte en parlant de Bergotte. Bloch n'était pas le seul qui eût des succès chez lui. Mon camarade en avait davantage encore auprès de ses soeurs qu'il ne cessait d'interpeller sur un ton bougon, en enfonçant sa tête dans son assiette ; il les faisait ainsi rire aux larmes. Elles avaient d'ailleurs adopté la langue de leur frère qu'elles parlaient couramment, comme si elle eût été obligatoire et la seule dont pussent user des personnes intelligentes. Quand nous arrivâmes, l'aînée dit à une de ses cadettes : « Va prévenir notre père prudent et notre mère vénérable. – Chiennes, leur dit Bloch, je vous présente le cavalier Saint-Loup, aux javelots rapides, qui est venu pour quelques jours de Doncières aux demeures de pierre polie, féconde en chevaux. » Comme il était aussi vulgaire que lettré, le discours se terminait d'habitude par quelque plaisanterie moins homérique : « Voyons, fermez un peu vos peplos aux belles agrafes, qu'est-ce que c'est que ce chichi-là ? Après tout c'est pas mon père ! » Et les demoiselles Bloch s'écroulaient dans une tempête de rires. Je dis à leur frère combien de joies il m'avait données en me recommandant la lecture de Bergotte dont j'avais adoré les livres.

Bloch père qui ne connaissait Bergotte que de loin, et la vie de Bergotte que par les racontars du parterre, avait une manière tout aussi indirecte de prendre connaissance de ses oeuvres, à l'aide de jugements d'apparence littéraire. Il vivait dans le monde des à peu près, où l'on salue dans le vide, où l'on juge dans le faux. L'inexactitude, l'incompétence, n'y diminuent pas l'assurance, au contraire. C'est le miracle bienfaisant de l'amour-propre que peu de gens pouvant avoir les relations brillantes et les connaissances profondes, ceux auxquels elles font défaut se croient encore les mieux partagés parce que l'optique des gradins sociaux fait que tout rang semble le meilleur à celui qui l'occupe et qui voit moins favorisés que lui, mal lotis, à plaindre, les plus grands qu'il nomme et calomnie sans les connaître, juge et dédaigne sans les comprendre. Même dans les cas où la multiplication des faibles avantages personnels par l'amour-propre ne suffirait pas à assurer à chacun la dose de bonheur, supérieure à celle accordée aux autres, qui lui est nécessaire, l'envie est là pour combler la différence. Il est vrai que si l'envie s'exprime en phrases dédaigneuses, il faut traduire : « Je ne veux pas le connaître » par « je ne peux pas le connaître ». C'est le sens intellectuel. Mais le sens passionné est bien : « Je ne veux pas le connaître. » On sait que cela n'est pas vrai mais on ne le dit pas cependant par simple artifice, on le dit parce qu'on éprouve ainsi, et cela suffit pour supprimer la distance, c'est-à-dire pour le bonheur.

L'égocentrisme permettant de la sorte à chaque humain de voir l'univers étagé au-dessous de lui qui est roi, Bloch se donnait le luxe d'en être un impitoyable quand le matin en prenant son chocolat, voyant la signature de Bergotte au bas d'un article dans le journal à peine entr'ouvert, il lui accordait dédaigneusement une audience écourtée, prononçait sa sentence, et s'octroyait le confortable plaisir de répéter entre chaque gorgée du breuvage bouillant : « Ce Bergotte est devenu illisible. Ce que cet animal-là peut être embêtant. C'est à se désabonner. Comme c'est emberlificoté, quelle tartine ! » Et il reprenait une beurrée.
