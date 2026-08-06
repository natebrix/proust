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
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "la grand-mère",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.84,
      "evidence": "« [elle] ouvrit subrepticement un carreau et fit envoler du même coup, avec les menus, les journaux, voiles et casquettes… »; « au milieu des invectives … réunissaient contre nous les touristes méprisants, dépeignés et furieux »",
      "explanation": "By opening the window for air, the grandmother provokes the public anger of the other guests; the text records this social blame while ironizing it, valuing her serenity and depreciating the crowd."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "She undergoes a brief collective ostracism in the dining room, which isolates the duo from the tourists."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-41-p-45"
}

### Candidate characters

[
  "Bergotte",
  "Françoise",
  "Gilberte",
  "Swann",
  "le narrateur"
]

### Prior local context (optional)

– Confondre les coups de mon pauvre chou avec d'autres, mais entre mille sa la grand-mère les reconnaîtrait ! Crois-tu donc qu'il y en ait d'autres au monde qui soient aussi bêtas, aussi fébriles, aussi partagés entre la crainte de me réveiller et de ne pas être compris. Mais quand même elle se contenterait d'un grattement on reconnaîtrait tout de suite sa petite souris, surtout quand elle est aussi unique et à plaindre que la mienne. Je l'entendais déjà depuis un moment qui hésitait, qui se remuait dans le lit, qui faisait tous ses manèges.

### Passage

Elle entr'ouvrait les persiennes ; à l'annexe en saillie de l'hôtel, le soleil était déjà installé sur les toits comme un couvreur matinal qui commence tôt son ouvrage et l'accomplit en silence pour ne pas réveiller la ville qui dort encore et de laquelle l'immobilité le fait paraître plus agile. Elle me disait l'heure, le temps qu'il ferait, que ce n'était pas la peine que j'allasse jusqu'à la fenêtre, qu'il y avait de la brume sur la mer, si la boulangerie était déjà ouverte, quelle était cette voiture qu'on entendait : tout cet insignifiant lever de rideau, ce négligeable introït du jour auquel personne n'assiste, petit morceau de vie qui n'était qu'à nous deux, que j'évoquerais volontiers dans la journée devant Françoise ou des étrangers en parlant du brouillard à couper au couteau qu'il y avait eu le matin à six heures, avec l'ostentation non d'un savoir acquis, mais d'une marque d'affection reçue par moi seul ; doux instant matinal qui s'ouvrait comme une symphonie par le dialogue rythmé de mes trois coups auquel la cloison pénétrée de tendresse et de joie, devenue harmonieuse, immatérielle, chantant comme les anges, répondait par trois autres coups, ardemment attendus, deux fois répétés, et où elle savait transporter l'âme de ma grand'mère tout entière et la promesse de sa venue, avec une allégresse d'annonciation et une fidélité musicale. Mais cette première nuit d'arrivée, quand ma grand'mère m'eût quitté, je recommençai à souffrir, comme j'avais déjà souffert à Paris au moment de quitter la maison. Peut-être cet effroi que j'avais – qu'ont tant d'autres – de coucher dans une chambre inconnue, peut-être cet effroi n'est-il que la forme la plus humble, obscure, organique, presque inconsciente, de ce grand refus désespéré qu'opposent les choses qui constituent le meilleur de notre vie présente à ce que nous revêtions mentalement de notre acceptation la formule d'un avenir où elles ne figurent pas ; refus qui était au fond de l'horreur que m'avait fait si souvent éprouver la pensée que mes parents mourraient un jour, que les nécessités de la vie pourraient m'obliger à vivre loin de Gilberte, ou simplement à me fixer définitivement dans un pays où je ne reverrais plus jamais mes amis ; refus qui était encore au fond de la difficulté que j'avais à penser à ma propre mort ou à une survie comme celle que Bergotte promettait aux hommes dans ses livres, dans laquelle je ne pourrais emporter mes souvenirs, mes défauts, mon caractère qui ne se résignaient pas à l'idée de ne plus être et ne voulaient pour moi ni du néant, ni d'une éternité où ils ne seraient plus.

Quand Swann m'avait dit à Paris, un jour que j'étais particulièrement souffrant : « Vous devriez partir pour ces délicieuses îles de l'Océanie, vous verrez que vous n'en reviendrez plus », j'aurais voulu lui répondre : « Mais alors je ne verrai plus votre fille, je vivrai au milieu de choses et de gens qu'elle n'a jamais vus. » Et pourtant ma raison me disait : « Qu'est-ce que cela peut faire, puisque tu n'en seras pas affligé ? Quand Monsieur Swann te dit que tu ne reviendras pas, il entend par là que tu ne voudras pas revenir, et puisque tu ne le voudras pas, c'est que, là-bas, tu seras heureux. » Car ma raison savait que l'habitude – l'habitude qui allait assumer maintenant l'entreprise de me faire aimer ce logis inconnu, de changer de place la glace, la nuance des rideaux, d'arrêter la pendule – se charge aussi bien de nous rendre chers les compagnons qui nous avaient déplu d'abord, de donner une autre forme aux visages, de rendre sympathique le son d'une voix, de modifier l'inclination des coeurs. Certes ces amitiés nouvelles pour des lieux et des gens ont pour trame l'oubli des anciennes ; mais justement ma raison pensait que je pouvais envisager sans terreur la perspective d'une vie où je serais à jamais séparé d'êtres dont je perdrais le souvenir, et c'est comme une consolation qu'elle offrait à mon coeur une promesse d'oubli qui ne faisait au contraire qu'affoler son désespoir. Ce n'est pas que notre coeur ne doive éprouver lui aussi, quand la séparation sera consommée, les effets analgésiques de l'habitude ; mais jusque-là il continuera de souffrir. Et la crainte d'un avenir où nous serons enlevés la vue et l'entretien de ceux que nous aimons et d'où nous tirons aujourd'hui notre plus chère joie, cette crainte, loin de se dissiper, s'accroît, si à la douleur d'une telle privation nous pensons que s'ajoutera ce qui pour nous semble actuellement plus cruel encore : ne pas la ressentir comme une douleur, y rester indifférent ; car alors notre moi serait changé, ce ne serait plus seulement le charme de nos parents, de notre maîtresse, de nos amis, qui ne serait plus autour de nous, mais notre affection pour eux ; elle aurait été si parfaitement arrachée de notre coeur dont elle est aujourd'hui une notable part, que nous pourrions nous plaire à cette vie séparée d'eux dont la pensée nous fait horreur aujourd'hui ; ce serait donc une vraie mort de nous-même, mort suivie, il est vrai, de résurrection, mais en un moi différent et jusqu'à l'amour duquel ne peuvent s'élever les parties de l'ancien moi condamnées à mourir. Ce sont elles – même les plus chétives, comme les obscurs attachements aux dimensions, à l'atmosphère d'une chambre – qui s'effarent et refusent, en des rébellions où il faut voir un mode secret, partiel, tangible et vrai de la résistance à la mort, de la longue résistance désespérée et quotidienne à la mort fragmentaire et successive telle qu'elle s'insère dans toute la durée de notre vie, détachant de nous à chaque moment des lambeaux de nous-même sur la mortification desquels des cellules nouvelles multiplieront. Et pour une nature nerveuse comme était la mienne, c'est-à-dire chez qui les intermédiaires, les nerfs, remplissent mal leurs fonctions, n'arrêtent pas dans sa route vers la conscience, mais y laissent au contraire parvenir, distincte, épuisante, innombrable et douloureuse, la plainte des plus humbles éléments du moi qui vont disparaître, l'anxieuse alarme que j'éprouvais sous ce plafond inconnu et trop haut n'était que la protestation d'une amitié qui survivait en moi pour un plafond familier et bas. Sans doute cette amitié disparaîtrait, une autre ayant pris sa place (alors la mort, puis une nouvelle vie auraient, sous le nom d'Habitude, accompli leur oeuvre double) ; mais jusqu'à son anéantissement, chaque soir elle souffrirait, et ce premier soir-là surtout, mise en présence d'un avenir déjà réalisé où il n'y avait plus de place pour elle, elle se révoltait, elle me torturait du cri de ses lamentations chaque fois que mes regards, ne pouvant se détourner de ce qui les blessait, essayaient de se poser au plafond inaccessible.

Mais le lendemain matin ! – après qu'un domestique fut venu m'éveiller et m'apporter de l'eau chaude, et pendant que je faisais ma toilette et essayais vainement de trouver les affaires dont j'avais besoin dans ma malle d'où je ne tirais, pêle-mêle, que celles qui ne pouvaient me servir à rien, quelle joie, pensant déjà au plaisir du déjeuner et de la promenade, de voir dans la fenêtre et dans toutes les vitrines des bibliothèques, comme dans les hublots d'une cabine de navire, la mer nue, sans ombrages et pourtant à l'ombre sur une moitié de son étendue que délimitait une ligne mince et mobile, et de suivre des yeux les flots qui s'élançaient l'un après l'autre comme des sauteurs sur un tremplin. À tous moments, tenant à la main la serviette raide et empesée où était écrit le nom de l'hôtel et avec laquelle je faisais d'inutiles efforts pour me sécher, je retournais près de la fenêtre jeter encore un regard sur ce vaste cirque éblouissant et montagneux et sur les sommets neigeux de ses vagues en pierre d'émeraude çà et là polie et translucide, lesquelles avec une placide violence et un froncement léonin laissaient s'accomplir et dévaler l'écoulement de leurs pentes auxquelles le soleil ajoutait un sourire sans visage. Fenêtre à laquelle je devais ensuite me mettre chaque matin comme au carreau d'une diligence dans laquelle on a dormi, pour voir si pendant la nuit s'est rapprochée ou éloignée une chaîne désirée – ici ces collines de la mer qui, avant de revenir vers nous en dansant, peuvent reculer si loin que souvent ce n'était qu'après une longue plaine sablonneuse que j'apercevais à une grande distance leurs premières ondulations, dans un lointain transparent, vaporeux et bleuâtre comme ces glaciers qu'on voit au fond des tableaux des primitifs toscans. D'autres fois, c'était tout près de moi que le soleil riait sur ces flots d'un vert aussi tendre que celui que conserve aux prairies alpestres (dans les montagnes où le soleil s'étale çà et là comme un géant qui en descendrait gaiement, par bonds inégaux, les pentes), moins l'humidité du sol que la liquide mobilité de la lumière. Au reste, dans cette brèche que la plage et les flots pratiquent au milieu du monde pour du reste y faire passer, pour y accumuler la lumière, c'est elle surtout, selon la direction d'où elle vient et que suit notre oeil, c'est elle qui déplace et situe les vallonnements de la mer. La diversité de l'éclairage ne modifie pas moins l'orientation d'un lieu, ne dresse pas moins devant nous de nouveaux buts qu'il nous donne le désir d'atteindre, que ne ferait un trajet longuement et effectivement parcouru en voyage. Quand, le matin, le soleil venait de derrière l'hôtel, découvrant devant moi les grèves illuminées jusqu'aux premiers contreforts de la mer, il semblait m'en montrer un autre versant et m'engager à poursuivre, sur la route tournante de ses rayons, un voyage immobile et varié à travers les plus beaux sites du paysage accidenté des heures. Et dès ce premier matin le soleil me désignait au loin, d'un doigt souriant, ces cimes bleues de la mer qui n'ont de nom sur aucune carte géographique, jusqu'à ce qu'étourdi de sa sublime promenade à la surface retentissante et chaotique de leurs crêtes et de leurs avalanches, il vînt se mettre à l'abri du vent dans ma chambre, se prélassant sur le lit défait et égrenant ses richesses sur le lavabo mouillé, dans la malle ouverte, où par sa splendeur même et son luxe déplacé, il ajoutait encore à l'impression du désordre. Hélas, le vent de mer, une heure plus tard, dans la grande salle à manger – tandis que nous déjeunions et que, de la gourde de cuir d'un citron, nous répandions quelques gouttes d'or sur deux soles qui bientôt laissèrent dans nos assiettes le panache de leurs arêtes, frisé comme une plume et sonore comme une cithare – il parut cruel à ma grand'mère de n'en pas sentir le souffle vivifiant à cause du châssis transparent mais clos qui, comme une vitrine, nous séparait de la plage tout en nous la laissant entièrement voir et dans lequel le ciel entrait si complètement que son azur avait l'air d'être la couleur des fenêtres et ses nuages blancs un défaut du verre. Me persuadant que j'étais « assis sur le môle » ou au fond du « boudoir » dont parle Beaudelaire, je me demandais si son « soleil rayonnant sur la mer » ce n'était pas – bien différent du rayon du soir, simple et superficiel comme un trait doré et tremblant – celui qui en ce moment brûlait la mer comme une topaze, la faisait fermenter, devenir blonde et laiteuse comme de la bière, écumante comme du lait, tandis que par moments s'y promenaient çà et là de grandes ombres bleues, que quelque dieu semblait s'amuser à déplacer en bougeant un miroir dans le ciel. Malheureusement ce n'était pas seulement par son aspect que différait de la « salle » de Combray donnant sur les maisons d'en face, cette salle à manger de Balbec, nue, emplie de soleil vert comme l'eau d'une piscine, et à quelques mètres de laquelle la marée pleine et le grand jour élevaient, comme devant la cité céleste, un rempart indestructible et mobile d'émeraude et d'or. À Combray, comme nous étions connus de tout le monde, je ne me souciais de personne. Dans la vie de bains de mer on ne connaît que ses voisins. Je n'étais pas encore assez âgé et j'étais resté trop sensible pour avoir renoncé au désir de plaire aux êtres et de les posséder. Je n'avais pas l'indifférence plus noble qu'aurait éprouvée un homme du monde à l'égard des personnes qui déjeunaient dans la salle à manger, ni des jeunes gens et des jeunes filles passant sur la digue, avec lesquels je souffrais de penser que je ne pourrais pas faire d'excursions, moins pourtant que si ma grand'mère, dédaigneuse des formes mondaines et ne s'occupant que de ma santé, leur avait adressé la demande, humiliante pour moi, de m'agréer comme compagnon de promenade. Soit qu'ils rentrassent vers quelque chalet inconnu, soit qu'ils en sortissent pour se rendre raquette en mains à un terrain de tennis, ou montassent sur des chevaux dont les sabots me piétinaient le coeur, je les regardais avec une curiosité passionnée, dans cet éclairage aveuglant de la plage où les proportions sociales sont changées, je suivais tous leurs mouvements à travers la transparence de cette grande baie vitrée qui laissait passer tant de lumière. Mais elle interceptait le vent et c'était un défaut à l'avis de ma grand'mère qui, ne pouvant supporter l'idée que je perdisse le bénéfice d'une heure d'air, ouvrit subrepticement un carreau et fit envoler du même coup, avec les menus, les journaux, voiles et casquettes de toutes les personnes qui étaient en train de déjeuner ; elle-même, soutenue par le souffle céleste, restait calme et souriante comme sainte Blandine, au milieu des invectives qui, augmentant mon impression d'isolement et de tristesse, réunissaient contre nous les touristes méprisants, dépeignés et furieux.

Pour une certaine partie – ce qui, à Balbec, donnait à la population, d'ordinaire banalement riche et cosmopolite, de ces sortes d'hôtels de grand luxe, un caractère régional assez accentué – ils se composaient de personnalités éminentes des principaux départements de cette partie de la France, d'un premier président de Caen, d'un bâtonnier de Cherbourg, d'un grand notaire du Mans qui, à l'époque des vacances, partant des points sur lesquels toute l'année ils étaient disséminés en tirailleurs ou comme des pions au jeu de dames, venaient se concentrer dans cet hôtel. Ils y conservaient toujours les mêmes chambres, et, avec leurs femmes qui avaient des prétentions à l'aristocratie, formaient un petit groupe, auquel s'étaient adjoints un grand avocat et un grand médecin de Paris qui le jour du départ leur disaient :

– Ah ! c'est vrai, vous ne prenez pas le même train que nous, vous êtes privilégiés, vous serez rendus pour le déjeuner.
