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
      "canonical_name": "M. Vinteuil",
      "surface_forms": [
        "M. Vinteuil"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "M. Vinteuil",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "uncertain",
      "confidence": 0.62,
      "evidence": "« si M. Vinteuil avait pu assister à cette scène, il n'eût peut-être pas encore perdu sa foi dans le bon coeur de sa fille, et peut-être même n'eût-il pas eu en cela tout à fait tort. »",
      "explanation": "The narrator suggests that Vinteuil’s paternal faith would not have been wholly mistaken, which locally rehabilitates his judgment and memory despite the daughter's shocking act."
    }
  ],
  "status_effects": [
    {
      "character": "M. Vinteuil",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.62,
      "explanation": "Vinteuil is modestly elevated as the narrator frames his trust in his daughter as at least partially justified."
    }
  ],
  "ambiguities": [
    "The elevation is hypothetical and hedged by repeated 'peut-être'; the broader passage chiefly analyzes the daughter's mixed motives, which are not directly mappable to a named character in the alias list."
  ],
  "unit_id": "v1-p1-combray#p-346-p-350"
}

### Candidate characters

[
  "Françoise",
  "la grand-mère",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Je n'en entendis pas davantage, car Mlle M. Vinteuil, d'un air las, gauche, affairé, honnête et triste, vint fermer les volets et la fenêtre, mais je savais maintenant, pour toutes les souffrances que pendant sa vie M. Vinteuil avait supportées à cause de sa fille, ce qu'après la mort il avait reçu d'elle en salaire.

### Passage

Et pourtant j'ai pensé depuis que si Vinteuil avait pu assister à cette scène, il n'eût peut-être pas encore perdu sa foi dans le bon coeur de sa fille, et peut-être même n'eût-il pas eu en cela tout à fait tort. Certes, dans les habitudes de Mlle Vinteuil l'apparence du mal était si entière qu'on aurait eu de la peine à la rencontrer réalisée à ce degré de perfection ailleurs que chez une sadique ; c'est à la lumière de la rampe des théâtres du boulevard plutôt que sous la lampe d'une maison de campagne véritable qu'on peut voir une fille faire cracher une amie sur le portrait d'un père qui n'a vécu que pour elle ; et il n'y a guère que le sadisme qui donne un fondement dans la vie à l'esthétique du mélodrame. Dans la réalité, en dehors des cas de sadisme, une fille aurait peut-être des manquements aussi cruels que ceux de Mlle Vinteuil envers la mémoire et les volontés de son père mort, mais elle ne les résumerait pas expressément en un acte d'un symbolisme aussi rudimentaire et aussi naïf ; ce que sa conduite aurait de criminel serait plus voilé aux yeux des autres et même à ses yeux à elle qui ferait le mal sans se l'avouer. Mais, au-delà de l'apparence, dans le coeur de Mlle Vinteuil, le mal, au début du moins, ne fut sans doute pas sans mélange. Une sadique comme elle est l'artiste du mal, ce qu'une créature entièrement mauvaise ne pourrait être, car le mal ne lui serait pas extérieur, il lui semblerait tout naturel, ne se distinguerait même pas d'elle ; et la vertu, la mémoire des morts, la tendresse filiale, comme elle n'en aurait pas le culte, elle ne trouverait pas un plaisir sacrilège à les profaner. Les sadiques de l'espèce de Mlle Vinteuil sont des êtres si purement sentimentaux, si naturellement vertueux que même le plaisir sensuel leur paraît quelque chose de mauvais, le privilège des méchants. Et quand ils se concèdent à eux-mêmes de s'y livrer un moment, c'est dans la peau des méchants qu'ils tâchent d'entrer et de faire entrer leur complice, de façon à avoir eu un moment l'illusion de s'être évadés de leur âme scrupuleuse et tendre, dans le monde inhumain du plaisir. Et je comprenais combien elle l'eût désiré en voyant combien il lui était impossible d'y réussir. Au moment où elle se voulait si différente de son père, ce qu'elle me rappelait, c'était les façons de penser, de dire, du vieux professeur de piano. Bien plus que sa photographie, ce qu'elle profanait, ce qu'elle faisait servir à ses plaisirs mais qui restait entre eux et elle et l'empêchait de les goûter directement, c'était la ressemblance de son visage, les yeux bleus de sa mère à lui qu'il lui avait transmis comme un bijou de famille, ces gestes d'amabilité qui interposaient entre le vice de Mlle Vinteuil et elle une phraséologie, une mentalité qui n'était pas faite pour lui et l'empêchait de le connaître, comme quelque chose de très différent des nombreux devoirs de politesse auxquels elle se consacrait d'habitude. Ce n'est pas le mal qui lui donnait l'idée du plaisir, qui lui semblait agréable ; c'est le plaisir qui lui semblait malin. Et comme chaque fois qu'elle s'y adonnait il s'accompagnait pour elle de ces pensées mauvaises qui le reste du temps étaient absentes de son âme vertueuse, elle finissait par trouver au plaisir quelque chose de diabolique, par l'identifier au Mal. Peut-être Mlle Vinteuil sentait-elle que son amie n'était pas foncièrement mauvaise, et qu'elle n'était pas sincère au moment où elle lui tenait ces propos blasphématoires. Du moins avait-elle le plaisir d'embrasser sur son visage des sourires, des regards, feints peut-être, mais analogues dans leur expression vicieuse et basse à ceux qu'aurait eus non un être de bonté et de souffrance, mais un être de cruauté et de plaisir. Elle pouvait s'imaginer un instant qu'elle jouait vraiment les jeux qu'eût joués, avec une complice aussi dénaturée, une fille qui aurait ressenti en effet ces sentiments barbares à l'égard de la mémoire de son père. Peut-être n'eût-elle pas pensé que le mal fût un état si rare, si extraordinaire, si dépaysant, où il était si reposant d'émigrer, si elle avait su discerner en elle, comme en tout le monde, cette indifférence aux souffrances qu'on cause et qui, quelques autres noms qu'on lui donne, est la forme terrible et permanente de la cruauté.

S'il était assez simple d'aller du côté de Méséglise, c'était une autre affaire d'aller du côté de Guermantes, car la promenade était longue et l'on voulait être sûr du temps qu'il ferait. Quand on semblait entrer dans une série de beaux jours ; quand Françoise désespérée qu'il ne tombât pas une goutte d'eau pour les « pauvres récoltes », et ne voyant que de rares nuages blancs nageant à la surface calme et bleue du ciel s'écriait en gémissant : « Ne dirait-on pas qu'on voit ni plus ni moins des chiens de mer qui jouent en montrant là-haut leurs museaux ? Ah ! ils pensent bien à faire pleuvoir pour les pauvres laboureurs ! Et puis quand les blés seront poussés, alors la pluie se mettra à tomber tout à petit patapon, sans discontinuer, sans plus savoir sur quoi elle tombe que si c'était sur la mer » ; quand mon père avait reçu invariablement les mêmes réponses favorables du jardinier et du baromètre, alors on disait au dîner : « Demain s'il fait le même temps, nous irons du côté de Guermantes. » On partait tout de suite après déjeuner par la petite porte du jardin et on tombait dans la rue des Perchamps, étroite et formant un angle aigu, remplie de graminées au milieu desquelles deux ou trois guêpes passaient la journée à herboriser, aussi bizarre que son nom d'où me semblaient dériver ses particularités curieuses et sa personnalité revêche, et qu'on chercherait en vain dans le Combray d'aujourd'hui où sur son tracé ancien s'élève l'école. Mais ma rêverie (semblable à ces architectes élèves de Viollet-le-Duc, qui, croyant retrouver sous un jubé Renaissance et un autel du XVIIe siècle les traces d'un choeur roman, remettent tout l'édifice dans l'état où il devait être au VIIe siècle) ne laisse pas une pierre du bâtiment nouveau, reperce et « restitue » la rue des Perchamps. Elle a d'ailleurs pour ces reconstitutions des données plus précises que n'en ont généralement les restaurateurs : quelques images conservées par ma mémoire, les dernières peut-être qui existent encore actuellement, et destinées à être bientôt anéanties, de ce qu'était le Combray du temps de mon enfance ; et parce que c'est lui-même qui les a tracées en moi avant de disparaître, émouvantes – si on peut comparer un obscur portrait à ces effigies glorieuses dont ma grand'mère aimait à me donner des reproductions – comme ces gravures anciennes de la Cène ou ce tableau de Gentile Bellini, dans lesquels l'on voit en un état qui n'existe plus aujourd'hui le chef-d'oeuvre de Vinci et le portail de Saint-Marc.

On passait, rue de l'Oiseau, devant la vieille hôtellerie de l'Oiseau flesché dans la grande cour de laquelle entrèrent quelquefois au XVIIe siècle les carrosses des duchesses de Montpensier, de Guermantes et de Montmorency, quand elles avaient à venir à Combray pour quelque contestation avec leurs fermiers, pour une question d'hommage. On gagnait le mail entre les arbres duquel apparaissait le clocher de Saint-Hilaire. Et j'aurais voulu pouvoir m'asseoir là et rester toute la journée à lire en écoutant les cloches ; car il faisait si beau et si tranquille que, quand sonnait l'heure, on aurait dit non qu'elle rompait le calme du jour, mais qu'elle le débarrassait de ce qu'il contenait et que le clocher, avec l'exactitude indolente et soigneuse d'une personne qui n'a rien d'autre à faire, venait seulement – pour exprimer et laisser tomber les quelques gouttes d'or que la chaleur y avait lentement et naturellement amassées – de presser, au moment voulu, la plénitude du silence.

Le plus grand charme du côté de Guermantes, c'est qu'on y avait presque tout le temps à côté de soi le cours de la Vivonne. On la traversait une première fois, dix minutes après avoir quitté la maison, sur une passerelle dite le Pont-Vieux. Dès le lendemain de notre arrivée, le jour de Pâques, après le sermon s'il faisait beau temps, je courais jusque-là, voir dans ce désordre d'un matin de grande fête où quelques préparatifs somptueux font paraître plus sordides les ustensiles de ménage qui traînent encore, la rivière qui se promenait déjà en bleu ciel entre les terres encore noires et nues, accompagnée seulement d'une bande de coucous arrivés trop tôt et de primevères en avance, cependant que çà et là une violette au bec bleu laissait fléchir sa tige sous le poids de la goutte d'odeur qu'elle tenait dans son cornet. Le Pont-Vieux débouchait dans un sentier de halage qui à cet endroit se tapissait l'été du feuillage bleu d'un noisetier sous lequel un pêcheur en chapeau de paille avait pris racine. À Combray où je savais quelle individualité de maréchal ferrant ou de garçon épicier était dissimulée sous l'uniforme du suisse ou le surplis de l'enfant de choeur, ce pêcheur est la seule personne dont je n'aie jamais découvert l'identité. Il devait connaître mes parents, car il soulevait son chapeau quand nous passions ; je voulais alors demander son nom, mais on me faisait signe de me taire pour ne pas effrayer le poisson. Nous nous engagions dans le sentier de halage qui dominait le courant d'un talus de plusieurs pieds ; de l'autre côté la rive était basse, étendue en vastes prés jusqu'au village et jusqu'à la gare qui en était distante. Ils étaient semés des restes, à demi enfouis dans l'herbe, du château des anciens comtes de Combray qui au moyen âge avait de ce côté le cours de la Vivonne comme défense contre les attaques des sires de Guermantes et des abbés de Martinville. Ce n'étaient plus que quelques fragments de tours bossuant la prairie, à peine apparents, quelques créneaux d'où jadis l'arbalétrier lançait des pierres, d'où le guetteur surveillait Novepont, Clairefontaine, Martinville-le-Sec, Bailleau-l'Exempt, toutes terres vassales de Guermantes entre lesquelles Combray était enclavé, aujourd'hui au ras de l'herbe, dominés par les enfants de l'école des frères qui venaient là apprendre leurs leçons ou jouer aux récréations – passé presque descendu dans la terre, couché au bord de l'eau comme un promeneur qui prend le frais, mais me donnant fort à songer, me faisant ajouter dans le nom de Combray à la petite ville d'aujourd'hui une cité très différente, retenant mes pensées par son visage incompréhensible et d'autrefois qu'il cachait à demi sous les boutons d'or. Ils étaient fort nombreux à cet endroit qu'ils avaient choisi pour leurs jeux sur l'herbe, isolés, par couples, par troupes, jaunes comme un jaune d'oeuf, brillants d'autant plus, me semblait-il, que ne pouvant dériver vers aucune velléité de dégustation le plaisir que leur vue me causait, je l'accumulais dans leur surface dorée, jusqu'à ce qu'il devînt assez puissant pour produire de l'inutile beauté ; et cela dès ma plus petite enfance, quand du sentier de halage je tendais les bras vers eux sans pouvoir épeler complètement leur joli nom de Princes de contes de fées français, venus peut-être il y a bien des siècles d'Asie, mais apatriés pour toujours au village, contents du modeste horizon, aimant le soleil et le bord de l'eau, fidèles à la petite vue de la gare, gardant encore pourtant comme certaines de nos vieilles toiles peintes, dans leur simplicité populaire, un poétique éclat d'orient.

Je m'amusais à regarder les carafes que les gamins mettaient dans la Vivonne pour prendre les petits poissons, et qui, remplies par la rivière, où elles sont à leur tour encloses, à la fois « contenant » aux flancs transparents comme une eau durcie, et « contenu » plongé dans un plus grand contenant de cristal liquide et courant, évoquaient l'image de la fraîcheur d'une façon plus délicieuse et plus irritante qu'elles n'eussent fait sur une table servie, en ne la montrant qu'en fuite dans cette allitération perpétuelle entre l'eau sans consistance où les mains ne pouvaient la capter et le verre sans fluidité où le palais ne pourrait en jouir. Je me promettais de venir là plus tard avec des lignes ; j'obtenais qu'on tirât un peu de pain des provisions du goûter ; j'en jetais dans la Vivonne des boulettes qui semblaient suffire pour y provoquer un phénomène de sursaturation, car l'eau se solidifiait aussitôt autour d'elles en grappes ovoïdes de têtards inanitiés qu'elle tenait sans doute jusque-là en dissolution, invisibles, tout près d'être en voie de cristallisation.
