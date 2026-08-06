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
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "baron de Charlus"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Charlus, convalescent and dépendant de Jupien, « salua Mme de Sainte-Euverte avec le même respect que si elle avait été la reine de France »; le narrateur y voit « l’amour des grandeurs de la terre » rabaissé et « tout l’orgueil humain » proclamé périssable.",
      "explanation": "The narrator presents Charlus's illness and worldly humility as a social and symbolic fall, opposite to his past snobbery."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "His dependency, his emphasized humility towards a woman once deemed unworthy, and the narrator's commentary mark a clear symbolic and worldly drop in rank."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-1-p-5"
}

### Candidate characters

[
  "Albertine",
  "Andrée",
  "Françoise",
  "Gilberte",
  "Jupien",
  "Swann",
  "duchesse de Guermantes",
  "la Berma",
  "le narrateur",
  "marquis de Bréauté",
  "princesse de Guermantes"
]

### Prior local context (optional)

(none provided)

### Passage

La nouvelle maison de santé dans laquelle je me retirai alors ne me guérit pas plus que la première ; et un long temps s'écoula avant que je la quittasse. Durant le trajet en chemin de fer que je fis pour rentrer à Paris, la pensée de mon absence de dons littéraires, que j'avais cru découvrir jadis du côté de Guermantes, que j'avais reconnue avec plus de tristesse encore dans mes promenades quotidiennes avec Gilberte, avant de rentrer dîner, fort avant dans la nuit, à Tansonville, et qu'à la veille de quitter cette propriété j'avais à peu près identifiée, en lisant quelques pages du journal des Goncourt, à la vanité, au mensonge de la littérature, cette pensée, moins douloureuse peut-être, plus morne encore, si je lui donnais comme objet non ma propre infirmité à moi particulière, mais l'inexistence de l'idéal auquel j'avais cru, cette pensée qui ne m'était pas depuis bien longtemps revenue à l'esprit me frappa de nouveau et avec une force plus lamentable que jamais. C'était, je me le rappelle, à un arrêt du train en pleine campagne. Le soleil éclairait jusqu'à la moitié de leur tronc une ligne d'arbres qui suivait la voie du chemin de fer. « Arbres, pensai-je, vous n'avez plus rien à me dire, mon coeur refroidi ne vous entend plus. Je suis pourtant ici en pleine nature, eh bien, c'est avec froideur, avec ennui que mes yeux constatent la ligne qui sépare votre front lumineux de votre tronc d'ombre. Si jamais j'ai pu me croire poète, je sais maintenant que je ne le suis pas. Peut-être dans la nouvelle partie de ma vie desséchée qui s'ouvre, les hommes pourraient-ils m'inspirer ce que ne me dit plus la nature. Mais les années où j'aurais peut-être été capable de la chanter ne reviendront jamais. » Mais en me donnant cette consolation d'une observation humaine possible venant prendre la place d'une inspiration impossible, je savais que je cherchais seulement à me donner une consolation, et que je savais moi-même sans valeur. Si j'avais vraiment une âme d'artiste, quel plaisir n'éprouverais-je pas devant ce rideau d'arbres éclairé par le soleil couchant, devant ces petites fleurs du talus qui se haussaient presque jusqu'au marchepied du wagon, dont je pouvais compter les pétales et dont je me garderais bien de décrire la couleur comme feraient tant de bons lettrés, car peut-on espérer transmettre au lecteur un plaisir qu'on n'a pas ressenti ? Un peu plus tard, j'avais vu avec la même indifférence les lentilles d'or et d'orange dont le même soleil couchant criblait les fenêtres d'une maison ; et enfin, comme l'heure avait avancé, j'avais vu une autre maison qui semblait construite en une substance d'un rose assez étrange. Mais j'avais fait ces diverses constatations avec la même absolue indifférence que si, me promenant dans un jardin avec une dame, j'avais vu une feuille de verre et un peu plus loin un objet d'une matière analogue à l'albâtre dont la couleur inaccoutumée ne m'aurait pas tiré du plus languissant ennui et que si, par politesse pour la dame, pour dire quelque chose et pour montrer que j'avais remarqué cette couleur, j'avais désigné en passant le verre coloré et le morceau de stuc. De la même manière, par acquit de conscience, je me signalais à moi-même, comme à quelqu'un qui m'eût accompagné et qui eût été capable d'en tirer plus de plaisir que moi, les reflets du feu dans les vitres et la transparence rose de la maison. Mais le compagnon à qui j'avais fait constater ces effets curieux était d'une nature sans doute moins enthousiaste que beaucoup de gens bien disposés, qu'une telle vue ravit, car il avait pris connaissance de ces couleurs sans aucune espèce d'allégresse.

Ma longue absence de Paris n'avait pas empêché d'anciens amis à continuer, comme mon nom restait sur leurs listes, à m'envoyer fidèlement des invitations, et quand j'en trouvai, en rentrant – avec une pour un goûter donné par la Berma en l'honneur de sa fille et de son gendre – une autre pour une matinée qui devait avoir lieu le lendemain chez le prince de Guermantes, les tristes réflexions que j'avais faites dans le train ne furent pas un des moindres motifs qui me conseillèrent de m'y rendre. Ce n'était vraiment pas la peine de me priver de mener la vie de l'homme du monde, m'étais-je dit, puisque le fameux « travail » auquel depuis si longtemps j'espère chaque jour me mettre le lendemain, je ne suis pas ou plus fait pour lui, et que peut-être même il ne correspond à aucune réalité. À vrai dire, cette raison était toute négative et ôtait simplement leur valeur à celles qui auraient pu me détourner de ce concert mondain. Mais celle qui m'y fit aller fut ce nom de Guermantes, depuis assez longtemps sorti de mon esprit pour que, lu sur la carte d'invitation, il réveillât un rayon de mon attention, allât prélever au fond de ma mémoire une coupe de leur passé, accompagné de toutes les images de forêt domaniale ou de hautes fleurs qui l'escortaient alors, et pour qu'il reprît pour moi le charme et la signification que je lui trouvais à Combray quand passant, avant de rentrer, dans la rue de l'Oiseau, je voyais du dehors, comme une laque obscure, le vitrail de Gilbert le Mauvais, sire de Guermantes. Pour un moment les Guermantes m'avaient semblé de nouveau entièrement différents des gens du monde, incomparables avec eux, avec tout être vivant, fût-il souverain ; ils me réapparaissaient comme des êtres issus de la fécondation de cet air aigre et vertueux de cette sombre ville de Combray où s'était passée mon enfance et du passé qu'on y apercevait dans la petite rue, à la hauteur du vitrail. J'avais eu envie d'aller chez les Guermantes comme si cela avait dû me rapprocher de mon enfance et des profondeurs de ma mémoire où je l'apercevais. Et j'avais continué à relire l'invitation jusqu'au moment où, révoltées, les lettres qui composaient ce nom si familier et si mystérieux, comme celui même de Combray, eussent repris leur indépendance et eussent dessiné devant mes yeux fatigués comme un nom que je ne connaissais pas.

Maman allant justement à un petit thé chez Mme Sazerat, je n'eus aucun scrupule à me rendre à la matinée de la princesse de Guermantes. Je pris une voiture pour y aller, car le prince de Guermantes n'habitait plus son ancien hôtel mais un magnifique qu'il s'était fait construire avenue du Bois. C'est un des torts des gens du monde de ne pas comprendre que s'ils veulent que nous croyions en eux il faudrait d'abord qu'ils y crussent eux-mêmes, ou au moins qu'ils respectassent les éléments essentiels de notre croyance. Au temps où je croyais, même si je savais le contraire, que les Guermantes habitaient tel palais en vertu d'un droit héréditaire, pénétrer dans le palais du sorcier ou de la fée, faire s'ouvrir devant moi les portes qui ne cèdent pas tant qu'on n'a pas prononcé la formule magique, me semblait aussi malaisé que d'obtenir un entretien du sorcier ou de la fée eux-mêmes. Rien ne m'était plus facile que de me faire croire à moi-même que le vieux domestique engagé de la veille ou fourni par Potel et Chabot était fils, petit-fils, descendant de ceux qui servaient la famille bien avant la Révolution, et j'avais une bonne volonté infinie à appeler portrait d'ancêtre le portrait qui avait été acheté le mois précédent chez Bernheim jeune. Mais un charme ne se transvase pas, les souvenirs ne peuvent se diviser, et du prince de Guermantes, maintenant qu'il avait percé lui-même à jour les illusions de ma croyance en étant allé habiter avenue du Bois, il ne restait plus grand'chose. Les plafonds que j'avais craint de voir s'écrouler quand on avait annoncé mon nom et sous lesquels eût flotté encore pour moi beaucoup du charme et des craintes de jadis couvraient les soirées d'une Américaine sans intérêt pour moi. Naturellement, les choses n'ont pas en elles-mêmes de pouvoir, et puisque c'est nous qui le leur confions, quelque jeune collégien bourgeois devait en ce moment avoir devant l'hôtel de l'avenue du Bois les mêmes sentiments que moi jadis devant l'ancien hôtel du prince de Guermantes. C'était qu'il était encore à l'âge des croyances, mais je l'avais dépassé, et j'avais perdu ce privilège, comme après la première jeunesse on perd le pouvoir qu'ont les enfants de dissocier en fractions digérables le lait qu'ils ingèrent, ce qui force les adultes à prendre, pour plus de prudence, le lait par petites quantités, tandis que les enfants peuvent le téter indéfiniment sans reprendre haleine. Du moins, le changement de résidence du prince de Guermantes eut cela de bon pour moi que la voiture qui était venue me chercher pour me conduire et dans laquelle je faisais ces réflexions dut traverser les rues qui vont vers les Champs-Élysées. Elles étaient fort mal pavées à cette époque, mais, dès le moment où j'y entrai, je n'en fus pas moins détaché de mes pensées par une sensation d'une extrême douceur ; on eût dit que tout d'un coup la voiture roulait plus facilement, plus doucement, sans bruit, comme quand les grilles d'un parc s'étant ouvertes on glisse sur les allées couvertes d'un sable fin ou de feuilles mortes ; matériellement il n'en était rien, mais je sentais tout à coup la suppression des obstacles extérieurs comme s'il n'y avait plus eu pour moi d'effort d'adaptation ou d'attention, tels que nous en faisons, même sans nous en rendre compte, devant les choses nouvelles ; les rues par lesquelles je passais en ce moment étaient celles, oubliées depuis si longtemps, que je prenais jadis avec Françoise pour aller aux Champs-Élysées. Le sol de lui-même savait où il devait aller ; sa résistance était vaincue. Et comme un aviateur qui a jusque-là péniblement roulé à terre, « décolle » brusquement, je m'élevais lentement vers les hauteurs silencieuses du souvenir. Dans Paris, ces rues-là se détacheront toujours pour moi en une autre matière que les autres. Quand j'arrivai au coin de la rue Royale, où était jadis le marchand en plein vent des photographies aimées de Françoise, il me sembla que la voiture, entraînée par des centaines de tours anciens, ne pourrait pas faire autrement que de tourner d'elle-même. Je ne traversais pas les mêmes rues que les promeneurs qui étaient dehors ce jour-là, mais un passé glissant, triste et doux. Il était, d'ailleurs, fait de tant de passés différents qu'il m'était difficile de reconnaître la cause de ma mélancolie, si elle était due à ces marches au-devant de Gilberte et dans la crainte qu'elle ne vînt pas, à la proximité d'une certaine maison où on m'avait dit qu'Albertine était allée avec Andrée, à la signification philosophique que semble prendre un chemin qu'on a suivi mille fois avec une passion qui ne dure plus et qui n'a pas porté de fruit, comme celui où, après le déjeuner, je faisais des courses si hâtives, si fiévreuses, pour regarder, toutes fraîches encore de colle, l'affiche de Phèdre et celle du Domino noir. Arrivé aux Champs-Élysées, comme je n'étais pas très désireux d'entendre tout le concert qui était donné chez les Guermantes, je fis arrêter la voiture et j'allais m'apprêter à descendre pour faire quelques pas à pied quand je fus frappé par le spectacle d'une voiture qui était en train de s'arrêter aussi. Un homme, les yeux fixes, la taille voûtée, était plutôt posé qu'assis dans le fond, et faisait pour se tenir droit les efforts qu'aurait faits un enfant à qui on aurait recommandé d'être sage. Mais son chapeau de paille laissait voir une forêt indomptée de cheveux entièrement blancs, et une barbe blanche, comme celle que la neige fait aux statues des fleuves dans les jardins publics, coulait de son menton. C'était, à côté de Jupien qui se multipliait pour lui, Charlus convalescent d'une attaque d'apoplexie que j'avais ignorée (on m'avait seulement dit qu'il avait perdu la vue ; or il ne s'était agi que de troubles passagers, car il voyait de nouveau très clair) et qui, à moins que jusque-là il se fût teint et qu'on lui eût interdit de continuer à en prendre la fatigue, avait plutôt, comme en une sorte de précipité chimique, rendu visible et brillant tout le métal dont étaient saturées et que lançaient comme autant de geysers les mèches maintenant de pur argent de sa chevelure et de sa barbe, cependant qu'elle avait imposé au vieux prince déchu la majesté shakespearienne d'un roi Lear. Les yeux n'étaient pas restés en dehors de cette convulsion totale, de cette altération métallurgique de la tête. Mais, par un phénomène inverse, ils avaient perdu tout leur éclat. Mais le plus émouvant est qu'on sentait que cet éclat perdu était la fierté morale, et que par là la vie physique et même intellectuelle de Charlus survivait à l'orgueil aristocratique, qu'on avait pu croire un moment faire corps avec elles. Ainsi à ce moment, se rendant sans doute aussi chez le prince de Guermantes, passa en Victoria Mme de Sainte-Euverte, que le baron jadis ne trouvait pas assez chic pour lui. Jupien, qui prenait soin de lui comme d'un enfant, lui souffla à l'oreille que c'était une personne de connaissance, Mme de Sainte-Euverte. Et aussitôt, avec une peine infinie et toute l'application d'un malade qui veut se montrer capable de tous les mouvements qui lui sont encore difficiles, Charlus se découvrit, s'inclina, et salua Mme de Sainte-Euverte avec le même respect que si elle avait été la reine de France. Peut-être y avait-il dans la difficulté même que Charlus avait à faire un tel salut une raison pour lui de le faire, sachant qu'il toucherait davantage par un acte qui, douloureux pour un malade, devenait doublement méritoire de la part de celui qui le faisait et flatteur pour celle à qui il s'adressait, les malades exagérant la politesse, comme les rois. Peut-être aussi y avait-il encore dans les mouvements du baron cette incoordination consécutive aux troubles de la moelle et du cerveau, et ses gestes dépassaient-ils l'intention qu'il avait. Pour moi, j'y vis plutôt une sorte de douceur quasi physique, de détachement des réalités de la vie, si frappants chez ceux que la mort a déjà fait entrer dans son ombre. La mise à nu des gisements argentés de la chevelure décelait un changement moins profond que cette inconsciente humilité mondaine qui intervertissait tous les rapports sociaux, humiliait devant Mme de Sainte-Euverte, eût humilié – en montrant ce qu'il a de fragile – devant la dernière des Américaines (qui eût pu enfin s'offrir la politesse jusque-là inaccessible pour elle du baron) le snobisme qui semblait le plus fier. Car le baron vivait toujours, pensait toujours ; son intelligence n'était pas atteinte. Et plus que n'eût fait tel choeur de Sophocle sur l'orgueil abaissé d'Œdipe, plus que la mort même, et toute oraison funèbre sur la mort, le salut empressé et humble du baron à Mme de Sainte-Euverte proclamait ce qu'a de périssable l'amour des grandeurs de la terre et tout l'orgueil humain. Charlus, qui jusque-là n'eût pas consenti à dîner avec Mme de Sainte-Euverte, la saluait maintenant jusqu'à terre. Il saluait peut-être par ignorance du rang de la personne qu'il saluait (les articles du code social pouvant être emportés par une attaque comme toute autre partie de la mémoire), peut-être par une incoordination qui transposait dans le plan de l'humilité apparente l'incertitude – sans cela hautaine qu'il aurait eue – de l'identité de la dame qui passait. Il la salua enfin avec cette politesse des enfants venant timidement dire bonjour aux grandes personnes, sur l'appel de leur mère. Et un enfant, c'est, sans la fierté qu'ils ont, ce qu'il était devenu. Recevoir l'hommage de Charlus, pour Mme de Sainte-Euverte c'était tout le snobisme, comme ç'avait été tout le snobisme du baron de le lui refuser. Or cette nature inaccessible et précieuse qu'il avait réussi à faire croire à Mme de Sainte-Euverte être essentielle à lui-même, Charlus l'anéantit d'un seul coup par la timidité appliquée, le zèle peureux avec lequel il ôta son chapeau, d'où les torrents de sa chevelure d'argent ruisselèrent tout le temps qu'il laissa sa tête découverte par déférence, avec l'éloquence d'un Bossuet.

Quand Jupien eut aidé le baron à descendre et que j'eus salué celui-ci, il me parla très vite, d'une voix si imperceptible que je ne pus distinguer ce qu'il me disait, ce qui lui arracha, quand pour la troisième fois je le fis répéter, un geste d'impatience qui m'étonna par l'impassibilité qu'avait d'abord montrée le visage et qui était due sans doute à un reste de paralysie. Mais quand je fus arrivé à comprendre ces paroles sussurrées, je m'aperçus que le malade gardait absolument intacte son intelligence. Il y avait, d'ailleurs, deux Charlus, sans compter les autres. Des deux, l'intellectuel passait son temps à se plaindre qu'il allait à l'aphasie, qu'il prononçait constamment un mot, une lettre pour une autre. Mais dès qu'en effet il lui arrivait de le faire, l'autre Charlus, le subconscient, lequel voulait autant faire envie que l'autre pitié, arrêtait immédiatement, comme un chef d'orchestre dont les musiciens pataugent, la phrase commencée, et avec une ingéniosité infinie attachait ce qui venait ensuite au mot dit en réalité pour un autre, mais qu'il semblait avoir choisi. Même sa mémoire était intacte ; il mettait, du reste, une coquetterie, qui n'allait pas sans la fatigue d'une application des plus ardues, à faire sortir tel souvenir ancien, peu important, se rapportant à moi et qui me montrerait qu'il avait gardé ou recouvré toute sa netteté d'esprit. Sans bouger la tête ni les yeux, ni varier d'une seule inflexion son débit, il me dit, par exemple : « Voici un poteau où il y a une affiche pareille à celle devant laquelle j'étais la première fois que je vous vis à Avranches, non, je me trompe, à Balbec. » Et c'était, en effet, une réclame pour le même produit. J'avais à peine, au début, distingué ce qu'il disait, de même qu'on commence par ne voir goutte dans une chambre dont tous les rideaux sont clos. Mais, comme des yeux dans la pénombre, mes oreilles s'habituèrent bientôt à ce pianissimo. Je crois aussi qu'il s'était graduellement renforcé pendant que le baron parlait, soit que la faiblesse de sa voix provînt en partie d'une appréhension nerveuse qui se dissipait quand, distrait par un tiers, il ne pensait plus à elle ; soit qu'au contraire cette faiblesse correspondît à son état véritable et que la force momentanée avec laquelle il parlait dans la conversation fût provoquée par une excitation factice, passagère et plutôt funeste, qui faisait dire aux étrangers : « Il est déjà mieux, il ne faut pas qu'il pense à son mal », mais augmentait au contraire celui-ci qui ne tardait pas à reprendre. Quoi qu'il en soit, le baron à ce moment (et même en tenant compte de mon adaptation) jetait ses paroles plus fort, comme la marée, les jours de mauvais temps, ses petites vagues tordues. Et ce qui lui restait de sa récente attaque faisait entendre au fond de ses paroles comme un bruit de cailloux roulés. D'ailleurs, continuant à me parler du passé, sans doute pour bien me montrer qu'il n'avait pas perdu la mémoire, il l'évoquait d'une façon funèbre, mais sans tristesse. Il ne cessait d'énumérer tous les gens de sa famille ou de son monde qui n'étaient plus, moins, semblait-il, avec la tristesse qu'ils ne fussent plus en vie qu'avec la satisfaction de leur survivre. Il semblait en rappelant leur trépas prendre mieux conscience de son retour vers la santé. C'est avec une dureté presque triomphale qu'il répétait sur un ton uniforme, légèrement bégayant et aux sourdes résonances sépulcrales : « Hannibal de Bréauté, mort ! Antoine de Mouchy, mort ! Swann Swann, mort ! Adalbert de Montmorency, mort ! Baron de Talleyrand, mort ! Sosthène de Doudeauville, mort ! » Et chaque fois, ce mot « mort » semblait tomber sur ces défunts comme une pelletée de terre plus lourde, lancée par un fossoyeur qui tenait à les river plus profondément à la tombe.

La duchesse de Létourville, qui n'allait pas à la matinée de la princesse de Guermantes, parce qu'elle venait d'être longtemps malade, passa à ce moment à pied à côté de nous, et apercevant le baron, dont elle ignorait la récente attaque, s'arrêta pour lui dire bonjour. Mais la maladie qu'elle venait d'avoir faisait qu'elle ne comprenait pas mieux, mais supportait plus impatiemment, avec une mauvaise humeur nerveuse où il y avait peut-être beaucoup de pitié, la maladie des autres. Entendant le baron prononcer difficilement et à faux certains mots, lui voyant bouger difficilement le bras, elle jeta les yeux tour à tour sur Jupien et sur moi comme pour nous demander l'explication d'un phénomène aussi choquant. Comme nous ne lui dîmes rien, ce fut à Charlus lui-même qu'elle adressa un long regard plein de tristesse mais aussi de reproches. Elle avait l'air de lui faire grief d'être avec elle, dehors, dans une attitude aussi peu usuelle que s'il fût sorti sans cravate ou sans souliers. À une nouvelle faute de prononciation que commit le baron, la douleur et l'indignation de la duchesse augmentant ensemble, elle dit au baron : « Charlus ! » sur le ton interrogatif et exaspéré des gens trop nerveux qui ne peuvent supporter d'attendre une minute et, si on les fait entrer tout de suite en s'excusant d'achever sa toilette, vous disent amèrement, non pour s'excuser mais pour s'accuser : « Mais alors, je vous dérange ! », comme si c'était un crime de la part de celui qu'on dérange. Finalement, elle nous quitta d'un air de plus en plus navré en disant au baron : « Vous feriez mieux de rentrer. »
