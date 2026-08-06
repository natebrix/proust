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
      "canonical_name": "duchesse de Guermantes",
      "surface_forms": [
        "duchesse de Guermantes"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Mme de Robert de Saint-Loup",
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "duchesse de Guermantes",
      "target": "Gilberte",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.79,
      "evidence": "« C'est une petite horreur... Elle n'a jamais aimé son mari... elle m'a même étonnée par un rare cynisme... Non, voyez-vous, c'est une cochonne. »",
      "explanation": "The duchess overwhelms Gilberte with insults and reproaches (lack of love, possible infidelity, lack of mourning), and uses it to justify her hostile behavior towards her. The narrator emphasizes the hateful motives and the cruelty of the duchess, which ironizes and questions the adherence to these judgments."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "Locally, Gilberte's image is lowered by the violent condemnation of a figure of social authority, even if the narrator reveals the bias and does not clearly endorse these remarks."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-111-p-115"
}

### Candidate characters

[
  "Albertine",
  "Elstir",
  "Legrandin",
  "M. Verdurin",
  "M. Vinteuil",
  "M. de Marsantes",
  "Mme Verdurin",
  "Morel",
  "Odette",
  "Robert de Saint-Loup",
  "Swann",
  "baron de Charlus",
  "la grand-mère",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

« Mais comment puis-je vous parler de ces sottises, comment cela peut-il vous intéresser ? » s'écria duchesse de Guermantes. Elle avait dit cette phrase à mi-voix et personne n'avait pu entendre ce qu'elle disait. Mais un jeune homme (qui devait m'intéresser dans la suite par un nom bien plus familier de moi autrefois que celui de Sainte-Euverte) se leva d'un air exaspéré et alla plus loin pour écouter avec plus de recueillement. Car c'était la sonate à Kreutzer qu'on jouait, mais, s'étant trompé sur le programme, il croyait que c'était un morceau de Ravel qu'on lui avait déclaré être beau comme du Palestrina, mais difficile à comprendre. Dans sa violence à changer de place, il heurta, à cause de la demi-obscurité, un bonheur du jour, ce qui n'alla pas sans faire tourner la tête à beaucoup de personnes pour qui cet exercice si simple de regarder derrière soi interrompait un peu le supplice d'écouter « religieusement » la sonate à Kreutzer. Et duchesse de Guermantes et moi, causes de ce petit scandale, nous nous hâtâmes de changer de pièce. « Oui, comment ces riens-là peuvent-ils intéresser un homme de votre mérite ? C'est comme tout à l'heure, quand je vous voyais causer avec Gilberte de Robert de Saint-Loup. Ce n'est pas digne de vous. Pour moi c'est exactement rien, cette femme-là, ce n'est même pas une femme, c'est ce que je connais de plus factice et de plus bourgeois au monde (car, même à sa défense de l'actualité, duchesse de Guermantes mêlait ses préjugés d'aristocrate). D'ailleurs devriez-vous venir dans des maisons comme ici ? Aujourd'hui, encore, je comprends parce qu'il y avait cette récitation de Rachel, ça peut vous intéresser. Mais si belle qu'elle ait été, elle ne donne pas devant ce public-là. Je vous ferai déjeuner seule avec elle. Alors vous verrez l'être que c'est. Mais elle est cent fois supérieure à tout ce qui est ici. Et après déjeuner elle vous dira du Verlaine. Vous m'en direz des nouvelles. » Elle me vanta surtout ses après-déjeuners, où il y avait tous les jours X et Y. Car elle en était arrivée à cette conception des femmes à « salons » qu'elle méprisait autrefois (bien qu'elle le niât aujourd'hui) et dont la grande supériorité, le signe d'élection selon elle, étaient d'avoir chez elle « tous les hommes ». Si je lui disais que telle grande dame à « salons » ne disait pas du bien, quand elle vivait, de Mme Howland, duchesse de Guermantes éclatait de rire devant ma naïveté : « Naturellement, l'autre avait chez elle tous les hommes et celle-ci cherchait à les attirer. » Elle reprit : « Mais dans de grandes machines comme ici, non, ça me passe que vous veniez. À moins que ce ne soit pour faire des études... », ajouta-t-elle d'un air de doute, de méfiance, et sans trop s'aventurer, car elle ne savait pas très exactement en quoi consistait le genre d'opérations improbables auquel elle faisait allusion.

### Passage

« Est-ce que vous ne croyez pas, dis-je à la duchesse, que ce soit pénible à Mme de Saint-Loup d'entendre ainsi, comme elle vient de le faire, l'ancienne maîtresse de son mari ? » Je vis se former dans le visage de Mme de Guermantes cette barre oblique qui relie par des raisonnements ce qu'on vient d'entendre à des pensées peu agréables. Raisonnements inexprimés, il est vrai, mais toutes les choses graves que nous disons ne reçoivent jamais de réponse ni verbale, ni écrite. Les sots seuls sollicitent en vain deux fois de suite une réponse à une lettre qu'ils ont eu le tort d'écrire et qui était une gaffe ; car à ces lettres-là il n'est jamais répondu que par des actes, et la correspondante qu'on croit inexacte vous dit Monsieur quand elle vous rencontre, au lieu de vous appeler par votre prénom. Mon allusion à la liaison de Saint-Loup avec Rachel n'avait rien de si grave et ne put mécontenter qu'une seconde Mme de Guermantes en lui rappelant que j'avais été l'ami de Saint-Loup, et peut-être son confident au sujet des déboires qu'avait procurés à Rachel sa soirée chez la duchesse. Mais celle-ci ne persista pas dans ses pensées, la barre orageuse se dissipa, et Mme de Guermantes me répondit à ma question relative à Mme de Saint-Loup : « Je vous dirai que je crois que ça lui est d'autant plus égal que Gilberte n'a jamais aimé son mari. C'est une petite horreur. Elle a aimé la situation, le nom, être ma nièce, sortir de sa fange, après quoi elle n'a pas eu d'autre idée que d'y rentrer. Je vous dirai que ça me faisait beaucoup de peine à cause du pauvre Saint-Loup, parce qu'il avait beau ne pas être un aigle, il s'en apercevait très bien, et d'un tas de choses. Il ne faut pas le dire parce qu'elle est malgré tout ma nièce, je n'ai pas la preuve positive qu'elle le trompait, mais il y a eu un tas d'histoires. Mais si, je vous dis que je le sais, avec un officier de Méséglise, Saint-Loup a voulu se battre. C'est pour tout ça que Saint-Loup s'est engagé. La guerre lui est apparue comme une délivrance de ses chagrins de famille ; si vous voulez ma pensée, il n'a pas été tué, il s'est fait tuer. Elle n'a eu aucune espèce de chagrin, elle m'a même étonnée par un rare cynisme dans l'affectation de son indifférence, ce qui m'a fait beaucoup de chagrin parce que j'aimais bien le pauvre Saint-Loup. Ça vous étonnera peut-être parce qu'on me connaît mal, mais il m'arrive encore de penser à lui. Je n'oublie personne. Il ne m'a jamais rien dit, mais il avait bien compris que je devinais tout. Mais, voyons, si elle avait aimé tant soit peu son mari, pourrait-elle supporter avec ce flegme de se trouver dans le même salon que la femme dont il a été l'amant éperdu pendant tant d'années, on peut dire toujours, car j'ai la certitude que ça n'a jamais cessé, même pendant la guerre. Mais elle lui sauterait à la gorge », s'écria la duchesse, oubliant qu'elle-même, en faisant inviter Rachel et en rendant possible la scène qu'elle jugeait inévitable si Gilberte eût aimé Saint-Loup, agissait cruellement. « Non, voyez-vous, conclut-elle, c'est une cochonne. » Une telle expression était rendue possible à Mme de Guermantes par la pente agréable qu'elle descendait, du milieu des Guermantes à la société des comédiennes, et aussi parce qu'elle greffait cela sur un genre XVIIIe siècle qu'elle jugeait plein de verdeur, enfin parce qu'elle se croyait tout permis. Mais cette expression lui était aussi dictée par la haine qu'elle éprouvait pour Gilberte, par un besoin de la frapper, à défaut de matériellement, en effigie. Et en même temps la duchesse pensait justifier par là toute la conduite qu'elle tenait à l'égard de Gilberte, ou plutôt contre elle, dans le monde, dans la famille, au point de vue même des intérêts et de la succession de Saint-Loup. Mais parfois les jugements qu'on porte reçoivent des faits qu'on ignore et qu'on n'eût pu supposer une justification apparente. Gilberte, qui tenait sans doute un peu de l'ascendance de sa mère (et c'est bien cette facilité que j'avais, sans m'en rendre compte, escomptée, en lui demandant de me faire connaître de très jeunes filles), tira, après réflexion, de la demande que j'avais faite, et sans doute pour que le profit ne sortît pas de la famille, une conclusion plus hardie que toutes celles que j'avais pu supposer et, revenant vers moi, me dit : « Si vous le permettez, je vais aller chercher ma fille pour vous la présenter. Elle est là-bas qui cause avec le petit Mortemart et d'autres bambins sans intérêt. Je suis sûre qu'elle sera une gentille amie pour vous. » Je lui demandai si Saint-Loup avait été content d'avoir une fille : « Oh ! il était tout fier d'elle. Mais, naturellement, je crois tout de même qu'étant donné ses goûts, dit naïvement Gilberte, il aurait préféré un garçon. » Cette fille, dont le nom et la fortune pouvaient faire espérer à sa mère qu'elle épouserait un prince royal et couronnerait toute l'oeuvre ascendante de Swann et de sa femme, choisit plus tard comme mari un homme de lettres obscur, car elle n'avait aucun snobisme, et fit redescendre cette famille plus bas que le niveau d'où elle était partie. Il fut alors extrêmement difficile de faire croire aux générations nouvelles que les parents de cet obscur ménage avaient eu une grande situation.

L'étonnement que me causèrent les paroles de Gilberte et le plaisir qu'elles me firent furent bien vite remplacés, tandis que Mme de Saint-Loup s'éloignait vers un autre salon, par cette idée du Temps passé, qu'elle aussi, à sa manière, me rendait, et sans même que je l'eusse vue, Mlle de Saint-Loup. Comme la plupart des êtres, d'ailleurs, n'était-elle pas comme sont dans les forêts les « étoiles » des carrefours où viennent converger des routes venues, pour notre vie aussi, des points les plus différents. Elles étaient nombreuses pour moi, celles qui aboutissaient à Mlle de Saint-Loup et qui rayonnaient autour d'elle. Et avant tout venaient aboutir à elle les deux grands « côtés » où j'avais fait tant de promenades et de rêves – par son père Saint-Loup de Saint-Loup le côté de Guermantes, par Gilberte sa mère le côté de Méséglise qui était le côté de chez Swann. L'un, par la mère de la jeune fille et les Champs-Élysées, me menait jusqu'à Swann, à mes soirs de Combray, au côté de Méséglise ; l'autre, par son père, à mes après-midi de Balbec où je le revoyais près de la mer ensoleillée. Déjà entre ces deux routes des transversales s'établissaient. Car ce Balbec réel où j'avais connu Saint-Loup, c'était en grande partie à cause de ce que Swann m'avait dit sur les églises, sur l'église persane surtout, que j'avais tant voulu y aller et, d'autre part, par Saint-Loup de Saint-Loup, neveu de la Mme de Guermantes, je rejoignais, à Combray encore, le côté de Guermantes. Mais à bien d'autres points de ma vie encore conduisait Mlle de Saint-Loup, à la Dame en rose, qui était sa grand'mère et que j'avais vue chez mon grand-oncle. Nouvelle transversale ici, car le valet de chambre de ce grand-oncle et qui m'avait introduit ce jour-là et qui plus tard m'avait, par le don d'une photographie, permis d'identifier la Dame en rose, était l'oncle du jeune homme que, non seulement Charlus, mais le père même de Mlle de Saint-Loup avait aimé, pour qui il avait rendu sa mère malheureuse. Et n'était-ce pas le grand-père de Mlle de Saint-Loup, Swann, qui m'avait le premier parlé de la musique de Vinteuil, de même que Gilberte m'avait la première parlé d'Albertine ? Or, c'est en parlant de la musique de Vinteuil à Albertine que j'avais découvert qui était sa grande amie et commencé avec elle cette vie qui l'avait conduite à la mort et m'avait causé tant de chagrins. C'était, du reste, aussi le père de Mlle de Saint-Loup qui était parti tâcher de faire revenir Albertine. Et même je revoyais toute ma vie mondaine, soit à Paris dans le salon des Swann ou des Guermantes, soit tout à l'opposé, à Balbec chez les Verdurin, faisant ainsi s'aligner, à côté des deux côtés de Combray, les Champs-Élysées et la belle terrasse de la Raspelière. D'ailleurs, quels êtres avons-nous connus qui, pour raconter notre amitié avec eux, ne nous obligent à les placer nécessairement dans tous les sites les plus différents de notre vie ? Une vie de Saint-Loup peinte par moi se déroulerait dans tous les décors et intéresserait toute ma vie, même les parties de cette vie où il fut étranger, comme ma grand'mère ou comme Albertine. D'ailleurs, si à l'opposé qu'ils fussent, les Verdurin tenaient à Odette par le passé de celle-ci, à Saint-Loup de Saint-Loup par Morel, et chez eux quel rôle n'avait pas joué la musique de Vinteuil. Enfin Swann avait aimé la soeur de Legrandin, lequel avait connu Charlus, dont le jeune Cambremer avait épousé la pupille. Certes, s'il s'agit uniquement de nos coeurs, le poète a eu raison de parler des fils mystérieux que la vie brise. Mais il est encore plus vrai qu'elle en tisse sans cesse entre les êtres, entre les événements, qu'elle entre-croise ces fils, qu'elle les redouble pour épaissir la trame, si bien qu'entre le moindre point de notre passé et tous les autres, un riche réseau de souvenirs ne laisse que le choix des communications. On peut dire qu'il n'y avait pas, si je cherchais à ne pas en user inconsciemment mais à me rappeler ce qu'elle avait été, une seule des choses qui nous servaient en ce moment qui n'avait été une chose vivante, et vivant d'une vie personnelle pour nous, transformée ensuite à notre usage en simple matière industrielle. Et ma présentation à Mlle de Saint-Loup allait avoir lieu chez Mme Verdurin devenue princesse de Guermantes ! Avec quel charme je repensais à tous nos voyages avec Albertine – dont j'allais demander à Mlle de Saint-Loup d'être un succédané – dans le petit tram, vers Doville, pour aller chez Mme Verdurin, cette même Mme Verdurin qui avait noué et rompu, avant mon amour pour Albertine, celui du grand-père et de la grand'mère de Mlle de Saint-Loup. Tout autour de nous étaient des tableaux de cet Elstir qui m'avait présenté à Albertine. Et pour mieux fondre tous mes passés, Mme Verdurin, tout comme Gilberte, avait épousé un Guermantes.

Nous ne pourrions pas raconter nos rapports avec un être, que nous avons même peu connu, sans faire se succéder les sites les plus différents de notre vie. Ainsi chaque individu – et j'étais moi-même un de ces individus – mesurait pour moi la durée par la révolution qu'il avait accomplie non seulement autour de soi-même, mais autour des autres, et notamment par les positions qu'il avait occupées successivement par rapport à moi.

Et sans doute tous ces plans différents, suivant lesquels le Temps, depuis que je venais de le ressaisir, dans cette fête, disposait ma vie, en me faisant songer que, dans un livre qui voudrait en raconter une, il faudrait user, par opposition à la psychologie plane dont on use d'ordinaire, d'une sorte de psychologie dans l'espace, ajoutaient une beauté nouvelle à ces résurrections que ma mémoire opérait tant que je songeais seul dans la bibliothèque, puisque la mémoire, en introduisant le passé dans le présent sans le modifier, tel qu'il était au moment où il était le présent, supprime précisément cette grande dimension du Temps suivant laquelle la vie se réalise.

Je vis Gilberte s'avancer. Moi, pour qui le mariage de Saint-Loup – les pensées qui m'occupaient alors et qui étaient les mêmes ce matin – était d'hier, je fus étonné de voir à côté d'elle une jeune fille d'environ seize ans, dont la taille élevée mesurait cette distance que je n'avais pas voulu voir.
