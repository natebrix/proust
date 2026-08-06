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
        "Elstir"
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
      "confidence": 0.93,
      "evidence": "« L'effort qu'Elstir faisait pour se dépouiller [...] était d'autant plus admirable [...] il avait justement une intelligence exceptionnellement cultivée. » Puis, exposant le porche de Balbec, Elstir déroule « tout un gigantesque poème théologique », « c'est fou, c'est divin, c'est mille fois supérieur... »",
      "explanation": "The narrator explicitly exalts the artistic probity and culture of Elstir, and stages his interpretative authority when he illuminates the porch of Balbec with precision and fervor."
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
      "explanation": "In this passage, Elstir is presented as admirable and scholarly; his didactic and convincing discourse enhances his value in the eyes of the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-366-p-370"
}

### Candidate characters

[
  "le peintre",
  "le narrateur"
]

### Prior local context (optional)

Parfois à ma fenêtre, dans l'hôtel de Balbec, le matin quand Françoise défaisait les couvertures qui cachaient la lumière, le soir quand j'attendais le moment de partir avec Robert de Saint-Loup, il m'était arrivé, grâce à un effet de soleil, de prendre une partie plus sombre de la mer pour une côte éloignée, ou de regarder avec joie une zone bleue et fluide sans savoir si elle appartenait à la mer ou au ciel. Bien vite mon intelligence rétablissait entre les éléments la séparation que mon impression avait abolie. C'est ainsi qu'il m'arrivait à Paris, dans ma chambre, d'entendre une dispute, presque une émeute, jusqu'à ce que j'eusse rapporté à sa cause, par exemple une voiture dont le roulement approchait, ce bruit dont j'éliminais alors les vociférations aiguës et discordantes que mon oreille avait réellement entendues, mais que mon intelligence savait que des roues ne produisaient pas. Mais les rares moments où l'on voit la nature telle qu'elle est, poétiquement, c'était de ceux-là qu'était faite l'oeuvre d'Elstir. Une de ses métaphores les plus fréquentes dans les marines qu'il avait près de lui en ce moment était justement celle qui, comparant la terre à la mer, supprimait entre elles toute démarcation. C'était cette comparaison, tacitement et inlassablement répétée dans une même toile, qui y introduisait cette multiforme et puissante unité, cause, parfois non clairement aperçue par eux, de l'enthousiasme qu'excitait chez certains amateurs la peinture d'Elstir.

### Passage

C'est par exemple à une métaphore de ce genre – dans un tableau, représentant le port de Carquethuit, tableau qu'il avait terminé depuis peu de jours et que je regardai longuement – qu'Elstir avait préparé l'esprit du spectateur en n'employant pour la petite ville que des termes marins, et que des termes urbains pour la mer. Soit que les maisons cachassent une partie du port, un bassin de calfatage ou peut-être la mer même s'enfonçant en golfe dans les terres ainsi que cela arrivait constamment dans ce pays de Balbec, de l'autre côté de la pointe avancée où était construite la ville, les toits étaient dépassés (comme ils l'eussent été par des cheminées ou par des clochers) par des mâts, lesquels avaient l'air de faire des vaisseaux auxquels ils appartenaient quelque chose de citadin, de construit sur terre, impression qu'augmentaient d'autres bateaux, demeurés le long de la jetée, mais en rangs si pressés que les hommes y causaient d'un bâtiment à l'autre sans qu'on pût distinguer leur séparation et l'interstice de l'eau, et ainsi cette flottille de pêche avait moins l'air d'appartenir à la mer que, par exemple, les églises de Criquebec qui, au loin, entourées d'eau de tous côtés parce qu'on les voyait sans la ville, dans un poudroiement de soleil et de vagues, semblaient sortir des eaux, soufflées en albâtre ou en écume et, enfermées dans la ceinture d'un arc-en-ciel versicolore, former un tableau irréel et mystique. Dans le premier plan de la plage, le peintre avait su habituer les yeux à ne pas reconnaître de frontière fixe, de démarcation absolue, entre la terre et l'océan. Des hommes qui poussaient des bateaux à la mer couraient aussi bien dans les flots que sur le sable, lequel mouillé, réfléchissait déjà les coques comme s'il avait été de l'eau. La mer elle-même ne montait pas régulièrement, mais suivait les accidents de la grève, que la perspective déchiquetait encore davantage, si bien qu'un navire en pleine mer, à demi caché par les ouvrages avancés de l'arsenal, semblait voguer au milieu de la ville ; des femmes qui ramassaient des crevettes dans les rochers avaient l'air, parce qu'elles étaient entourées d'eau et à cause de la dépression qui, après la barrière circulaire des roches, abaissait la plage (des deux côtés les plus rapprochés de terre) au niveau de la mer, d'être dans une grotte marine surplombée de barques et de vagues, ouverte et protégée au milieu des flots écartés miraculeusement. Si tout le tableau donnait cette impression des ports où la mer entre dans la terre, où la terre est déjà marine, et la population amphibie, la force de l'élément marin éclatait partout ; et près des rochers, à l'entrée de la jetée, où la mer était agitée, on sentait aux efforts des matelots et à l'obliquité des barques couchées en angle aigu devant la calme verticalité de l'entrepôt, de l'église, des maisons de la ville, où les uns rentraient, d'où les autres partaient pour la pêche, qu'ils trottaient rudement sur l'eau comme sur un animal fougueux et rapide dont les soubresauts, sans leur adresse, les eût jetés à terre. Une bande de promeneurs sortait gaiement en une barque secouée comme une carriole ; un matelot joyeux, mais attentif aussi, la gouvernait comme avec des guides, menait la voile fougueuse, chacun se tenait bien à sa place pour ne pas faire trop de poids d'un côté et ne pas verser, et on courait ainsi par les champs ensoleillés dans les sites ombreux, dégringolant les pentes. C'était une belle matinée malgré l'orage qu'il avait fait. Et même on sentait encore les puissantes actions qu'avait à neutraliser le bel équilibre des barques immobiles, jouissant du soleil et de la fraîcheur, dans les parties où la mer était si calme que les reflets avaient presque plus de solidité et de réalité que les coques vaporisées par un effet de soleil et que la perspective faisait s'enjamber les unes les autres. Ou plutôt on n'aurait pas dit d'autres parties de la mer. Car entre ces parties, il y avait autant de différence qu'entre l'une d'elles et l'église sortant des eaux, et les bateaux derrière la ville. L'intelligence faisait ensuite un même élément de ce qui était, ici noir dans un effet d'orage, plus loin tout d'une couleur avec le ciel et aussi verni que lui, et là si blanc de soleil, de brume et d'écume, si compact, si terrien, si circonvenu de maisons, qu'on pensait à quelque chaussée de pierres ou à un champ de neige, sur lequel on était effrayé de voir un navire s'élever en pente raide et à sec comme une voiture qui s'ébroue en sortant d'un gué, mais qu'au bout d'un moment, en y voyant sur l'étendue haute et inégale du plateau solide des bateaux titubants, on comprenait, identique en tous ces aspects divers, être encore la mer.

Bien qu'on dise avec raison qu'il n'y a pas de progrès, pas de découvertes en art, mais seulement dans les sciences, et que chaque artiste recommençant pour son compte un effort individuel ne peut y être aidé ni entravé par les efforts de tout autre, il faut pourtant reconnaître que dans la mesure où l'art met en lumière certaines lois, une fois qu'une industrie les a vulgarisées, l'art antérieur perd rétrospectivement un peu de son originalité. Depuis les débuts d'Elstir, nous avons connu ce qu'on appelle « d'admirables » photographies de paysages et de villes. Si on cherche à préciser ce que les amateurs désignent dans ce cas par cette épithète, on verra qu'elle s'applique d'ordinaire à quelque image singulière d'une chose connue, image différente de celles que nous avons l'habitude de voir, singulière et pourtant vraie et qui à cause de cela est pour nous doublement saisissante parce qu'elle nous étonne, nous fait sortir de nos habitudes, et tout à la fois nous fait rentrer en nous-même en nous rappelant une impression. Par exemple telle de ces photographies « magnifiques » illustrera une loi de la perspective, nous montrera telle cathédrale que nous avons l'habitude de voir au milieu de la ville, prise au contraire d'un point choisi d'où elle aura l'air trente fois plus haute que les maisons et faisant éperon au bord du fleuve d'où elle est en réalité distante. Or, l'effort d'Elstir de ne pas exposer les choses telles qu'il savait qu'elles étaient, mais selon ces illusions optiques dont notre vision première est faite, l'avait précisément amené à mettre en lumière certaines de ces lois de perspective, plus frappantes alors, car l'art était le premier à les dévoiler. Un fleuve, à cause du tournant de son cours, un golfe à cause du rapprochement apparent des falaises, avaient l'air de creuser au milieu de la plaine ou des montagnes un lac absolument fermé de toutes parts. Dans un tableau pris de Balbec par une torride journée d'été, un rentrant de la mer semblait enfermé dans des murailles de granit rose, n'être pas la mer, laquelle commençait plus loin. La continuité de l'océan n'était suggérée que par des mouettes qui, tournoyant sur ce qui semblait au spectateur de la pierre, humaient au contraire l'humidité du flot. D'autres lois se dégageaient de cette même toile comme, au pied des immenses falaises, la grâce lilliputienne des voiles blanches sur le miroir bleu où elles semblaient des papillons endormis, et certains contrastes entre la profondeur des ombres et la pâleur de la lumière. Ces jeux des ombres, que la photographie a banalisés aussi, avaient intéressé Elstir au point qu'il s'était complu autrefois à peindre de véritables mirages, où un château coiffé d'une tour apparaissait comme un château circulaire complètement prolongé d'une tour à son faîte, et en bas d'une tour inverse, soit que la pureté extraordinaire d'un beau temps donnât à l'ombre qui se reflétait dans l'eau la dureté et l'éclat de la pierre, soit que les brumes du matin rendissent la pierre aussi vaporeuse que l'ombre. De même au delà de la mer, derrière une rangée de bois une autre mer commençait, rosée par le coucher du soleil et qui était le ciel. La lumière, inventant comme de nouveaux solides, poussait la coque du bateau qu'elle frappait, en retrait de celle qui était dans l'ombre, et disposait comme les degrés d'un escalier de cristal la surface matériellement plane, mais brisée par l'éclairage de la mer au matin. Un fleuve qui passe sous les ponts d'une ville était pris d'un point de vue tel qu'il apparaissait entièrement disloqué, étalé ici en lac, aminci là en filet, rompu ailleurs par l'interposition d'une colline couronnée de bois où le citadin va le soir respirer la fraîcheur du soir ; et le rythme même de cette ville bouleversée n'était assuré que par la verticale inflexible des clochers qui ne montaient pas, mais plutôt, selon le fil à plomb de la pesanteur marquant la cadence comme dans une marche triomphale, semblaient tenir en suspens au-dessous d'eux toute la masse plus confuse des maisons étagées dans la brume, le long du fleuve écrasé et décousu. Et (comme les premières oeuvres d'Elstir dataient de l'époque où on agrémentait les paysages par la présence d'un personnage) sur la falaise ou dans la montagne, le chemin, cette partie à demi humaine de la nature, subissait comme le fleuve ou l'océan les éclipses de la perspective. Et soit qu'une arête montagneuse, ou la brume d'une cascade, ou la mer, empêchât de suivre la continuité de la route, visible pour le promeneur mais non pour nous, le petit personnage humain en habits démodés perdu dans ces solitudes semblait souvent arrêté devant un abîme, le sentier qu'il suivait finissant là, tandis que, trois cents mètres plus haut dans ces bois de sapins, c'est d'un oeil attendri et d'un coeur rassuré que nous voyions reparaître la mince blancheur de son sable hospitalier au pas du voyageur, mais dont le versant de la montagne nous avait dérobé, contournant la cascade ou le golfe, les lacets intermédiaires.

L'effort qu'Elstir faisait pour se dépouiller en présence de la réalité de toutes les notions de son intelligence était d'autant plus admirable que cet homme qui, avant de peindre, se faisait ignorant, oubliait tout par probité, car ce qu'on sait n'est pas à soi, avait justement une intelligence exceptionnellement cultivée. Comme je lui avouais la déception que j'avais eue devant l'église de Balbec : « Comment, me dit-il, vous avez été déçu par ce porche ? mais c'est la plus belle Bible historiée que le peuple ait jamais pu lire. Cette Vierge et tous les bas-reliefs qui racontent sa vie, c'est l'expression la plus tendre, la plus inspirée de ce long poème d'adoration et de louanges que le moyen âge déroulera à la gloire de la Madone. Si vous saviez à côté de l'exactitude la plus minutieuse à traduire le texte saint, quelles trouvailles de délicatesse a eues le vieux sculpteur, que de profondes pensées, quelle délicieuse poésie !

« L'idée de ce grand voile dans lequel les Anges portent le corps de la Vierge, trop sacré pour qu'ils osent le toucher directement (je lui dis que le même sujet était traité à Saint-André-des-Champs ; il avait vu des photographies du porche de cette dernière église mais me fit remarquer que l'empressement de ces petits paysans qui courent tous à la fois autour de la Vierge était autre chose que la gravité des deux grands anges presque italiens, si élancés, si doux) ; l'ange qui emporte l'âme de la Vierge pour la réunir à son corps ; dans la rencontre de la Vierge et d'Élisabeth, le geste de cette dernière qui touche le sein de Marie et s'émerveille de le sentir gonflé ; et le bras bandé de la sage-femme qui n'avait pas voulu croire, sans toucher, à l'Immaculée-Conception ; et la ceinture jetée par la Vierge à saint Thomas pour lui donner la preuve de sa résurrection ; ce voile aussi que la Vierge arrache de son sein pour en voiler la nudité de son fils d'un côté de qui l'Église recueille le sang, la liqueur de l'Eucharistie, tandis que, de l'autre, la Synagogue, dont le règne est fini, a les yeux bandés, tient un sceptre à demi brisé et laisse échapper, avec sa couronne qui lui tombe de la tête, les tables de l'ancienne Loi ; et l'époux qui aidant, à l'heure du Jugement dernier, sa jeune femme à sortir du tombeau lui appuie la main contre son propre coeur pour la rassurer et lui prouver qu'il bat vraiment, est-ce aussi assez chouette comme idée, assez trouvé ? Et l'ange qui emporte le soleil et la lune devenus inutiles puisqu'il est dit que la Lumière de la Croix sera sept fois plus puissante que celle des astres ; et celui qui trempe sa main dans l'eau du bain de Jésus pour voir si elle est assez chaude ; et celui qui sort des nuées pour poser sa couronne sur le front de la Vierge, et tous ceux qui penchés du haut du ciel, entre les balustres de la Jérusalem céleste, lèvent les bras d'épouvante ou de joie à la vue des supplices des méchants et du bonheur des élus ! Car c'est tous les cercles du ciel, tout un gigantesque poème théologique et symbolique que vous avez là. C'est fou, c'est divin, c'est mille fois supérieur à tout ce que vous verrez en Italie où d'ailleurs ce tympan a été littéralement copié par des sculpteurs de bien moins de génie. Il n'y a pas eu d'époque où tout le monde a du génie, tout ça c'est des blagues, ça serait plus fort que l'âge d'or. Le type qui a sculpté cette façade-là, croyez bien qu'il était aussi fort, qu'il avait des idées aussi profondes que les gens de maintenant que vous admirez le plus. Je vous montrerais cela, si nous y allions ensemble. Il y a certaines paroles de l'office de l'Assomption qui ont été traduites avec une subtilité qu'un Redon n'a pas égalée. »

Cette vaste vision céleste dont il me parlait, ce gigantesque poème théologique que je comprenais avoir été écrit là, pourtant quand mes yeux pleins de désirs s'étaient ouverts devant la façade, ce n'est pas eux que j'avais vus. Je lui parlais de ces grandes statues de saints qui montées sur des échasses forment une sorte d'avenue.
