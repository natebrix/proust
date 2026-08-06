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
      "canonical_name": "Albertine",
      "surface_forms": [
        "Mlle Simonet"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.96
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Albertine",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.76,
      "evidence": "« cette insignifiance ... finissait par comprendre même Mlle Simonet et ses amies »; « la de moins en moins existante Mlle Simonet »",
      "explanation": "The narrator states that, under the effect of his exaltation, the importance of Albertine falls for him, which diminishes her local emotional place."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.76,
      "explanation": "In the eyes of the narrator, her emotional importance clearly decreases in this scene."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-331-p-335"
}

### Candidate characters

[
  "Robert de Saint-Loup",
  "jeune blonde de Rivebelle",
  "la grand-mère",
  "le narrateur",
  "princesse de Luxembourg"
]

### Prior local context (optional)

Le restaurant n'était pas fréquenté seulement par des demi-mondaines, mais aussi par des gens du monde le plus élégant, qui y venaient goûter vers cinq heures ou y donnaient de grands dîners. Les goûters avaient lieu dans une longue galerie vitrée, étroite, en forme de couloir qui, allant du vestibule à la salle à manger, longeait sur un côté le jardin, duquel elle n'était séparée, sauf en exceptant quelques colonnes de pierre, que par le vitrage qu'on ouvrait ici ou là. Il en résultait, outre de nombreux courants d'air, des coups de soleil brusques, intermittents, un éclairage éblouissant, empêchant presque de distinguer les goûteuses, ce qui faisait que, quand elles étaient là, empilées deux tables par deux tables dans toute la longueur de l'étroit goulot, comme elles chatoyaient à tous les mouvements qu'elles faisaient pour boire leur thé ou se saluer entre elles, on aurait dit un réservoir, une nasse où le pêcheur a entassé les éclatants poissons qu'il a pris, lesquels à moitié hors de l'eau et baignés de rayons miroitent aux regards en leur éclat changeant.

### Passage

Quelques heures plus tard, pendant le dîner qui, lui, était naturellement servi dans la salle à manger, on allumait les lumières, bien qu'il fît encore clair dehors, de sorte qu'on voyait devant soi, dans le jardin, à côté de pavillons éclairés par le crépuscule et qui semblaient les pâles spectres du soir, des charmilles dont la glauque verdure était traversée par les derniers rayons et qui, de la pièce éclairée par les lampes où on dînait, apparaissaient au delà du vitrage non plus, comme on aurait dit, des dames qui goûtaient à la fin de l'après-midi, le long du couloir bleuâtre et or, dans un filet étincelant et humide, mais comme les végétations d'un pâle et vert aquarium géant à la lumière surnaturelle. On se levait de table ; et si les convives, pendant le repas, tout en passant leur temps à regarder, à reconnaître, à se faire nommer les convives du dîner voisin, avaient été retenus dans une cohésion parfaite autour de leur propre table, la force attractive qui les faisait graviter autour de leur amphitryon d'un soir perdait de sa puissance, au moment où pour prendre le café ils se rendaient dans ce même couloir qui avait servi aux goûters ; il arrivait souvent qu'au moment du passage, tel dîner en marche abandonnait l'un ou plusieurs de ses corpuscules, qui ayant subi trop fortement l'attraction du dîner rival se détachaient un instant du leur, où ils étaient remplacés par des messieurs ou des dames qui étaient venus saluer des amis, avant de rejoindre, en disant : « Il faut que je me sauve retrouver M. X... dont je suis ce soir l'invité. » Et pendant un instant on aurait dit de deux bouquets séparés qui auraient interchangé quelques-unes de leurs fleurs. Puis le couloir lui-même se vidait. Souvent, comme il faisait même après dîner encore un peu jour, on n'allumait pas ce long corridor, et côtoyé par les arbres qui se penchaient au dehors de l'autre côté du vitrage, il avait l'air d'une allée dans un jardin boisé et ténébreux. Parfois dans l'ombre une dîneuse s'y attardait. En le traversant pour sortir, j'y distinguai un soir, assise au milieu d'un groupe inconnu, la belle princesse de Luxembourg. Je me découvris sans m'arrêter. Elle me reconnut, inclina la tête en souriant ; très au-dessus de ce salut, émanant de ce mouvement même, s'élevèrent mélodieusement quelques paroles à mon adresse, qui devaient être un bonsoir un peu long, non pour que je m'arrêtasse, mais seulement pour compléter le salut, pour en faire un salut parlé. Mais les paroles restèrent si indistinctes et le son que seul je perçus se prolongea si doucement et me sembla si musical, que ce fut comme si, dans la ramure assombrie des arbres, un rossignol se fût mis à chanter. Si par hasard, pour finir la soirée avec telle bande d'amis à lui que nous avions rencontrée, Saint-Loup décidait de nous rendre au Casino d'une plage voisine, et, partant avec eux, s'il me mettait seul dans une voiture, je recommandais au cocher d'aller à toute vitesse, afin que fussent moins longs les instants que je passerais sans avoir l'aide de personne pour me dispenser de fournir moi-même à ma sensibilité – en faisant machine en arrière et en sortant de la passivité où j'étais pris comme dans un engrenage – ces modifications que depuis mon arrivée à Rivebelle je recevais des autres. Le choc possible avec une voiture venant en sens inverse dans ces sentiers où il n'y avait de place que pour une seule et où il faisait nuit noire, l'instabilité du sol souvent éboulé de la falaise, la proximité de son versant à pic sur la mer, rien de tout cela ne trouvait en moi le petit effort qui eût été nécessaire pour amener la représentation et la crainte du danger jusqu'à ma raison. C'est que, pas plus que ce n'est le désir de devenir célèbre, mais l'habitude d'être laborieux, qui nous permet de produire une oeuvre, ce n'est l'allégresse du moment présent, mais les sages réflexions du passé, qui nous aident à préserver le futur. Or, si déjà arrivant à Rivebelle, j'avais jeté loin de moi ces béquilles du raisonnement, du contrôle de soi-même qui aident notre infirmité à suivre le droit chemin, et me trouvais en proie à une sorte d'ataxie morale, l'alcool, en tendant exceptionnellement mes nerfs, avait donné aux minutes actuelles, une qualité, un charme, qui n'avaient pas eu pour effet de me rendre plus apte ni même plus résolu à les défendre ; car en me les faisant préférer mille fois au reste de ma vie, mon exaltation les en isolait ; j'étais enfermé dans le présent comme les héros, comme les ivrognes ; momentanément éclipsé, mon passé ne projetait plus devant moi cette ombre de lui-même que nous appelons notre avenir ; plaçant le but de ma vie, non plus dans la réalisation des rêves de ce passé, mais dans la félicité de la minute présente, je ne voyais pas plus loin qu'elle. De sorte que, par une contradiction qui n'était qu'apparente, c'est au moment où j'éprouvais un plaisir exceptionnel, où je sentais que ma vie pouvait être heureuse, où elle aurait dû avoir à mes yeux plus de prix, c'est à ce moment que, délivré des soucis qu'elle avait pu m'inspirer jusque-là, je la livrais sans hésitation au hasard d'un accident. Je ne faisais, du reste, en somme, que concentrer dans une soirée l'incurie qui pour les autres hommes est diluée dans leur existence entière où journellement ils affrontent sans nécessité le risque d'un voyage en mer, d'une promenade en aéroplane ou en automobile, quand les attend à la maison l'être que leur mort briserait ou quand est encore lié à la fragilité de leur cerveau le livre dont la prochaine mise au jour est la seule raison de leur vie. Et de même dans le restaurant de Rivebelle, les soirs où nous y restions, si quelqu'un était venu dans l'intention de me tuer, comme je ne voyais plus que dans un lointain sans réalité ma grand-mère, ma vie à venir, mes livres à composer, comme j'adhérais tout entier à l'odeur de la femme qui était à la table voisine, à la politesse des maîtres d'hôtel, au contour de la valse qu'on jouait, que j'étais collé à la sensation présente, n'ayant pas plus d'extension qu'elle ni d'autre but que de ne pas en être séparé, je serais mort contre elle, je me serais laissé massacrer sans offrir de défense, sans bouger, abeille engourdie par la fumée du tabac, qui n'a plus le souci de préserver sa ruche.

Je dois du reste dire que cette insignifiance où tombaient les choses les plus graves, par contraste avec la violence de mon exaltation, finissait par comprendre même Mlle Simonet et ses amies. L'entreprise de les connaître me semblait maintenant facile mais indifférente, car ma sensation présente seule, grâce à son extraordinaire puissance, à la joie que provoquaient ses moindres modifications et même sa simple continuité, avait de l'importance pour moi ; tout le reste, parents, travail, plaisirs, jeunes filles de Balbec, ne pesait pas plus qu'un flocon d'écume dans un grand vent qui ne le laisse pas se poser, n'existait plus que relativement à cette puissance intérieure ; l'ivresse réalise pour quelques heures l'idéalisme subjectif, le phénoménisme pur ; tout n'est plus qu'apparences et n'existe plus qu'en fonction de notre sublime nous-même. Ce n'est pas, du reste, qu'un amour véritable, si nous en avons un, ne puisse subsister dans un semblable état. Mais nous sentons si bien, comme dans un milieu nouveau, que des pressions inconnues ont changé les dimensions de ce sentiment que nous ne pouvons pas le considérer pareillement. Ce même amour, nous le retrouvons bien, mais déplacé, ne pesant plus sur nous, satisfait de la sensation que lui accorde le présent et qui nous suffit, car de ce qui n'est pas actuel nous ne nous soucions pas. Malheureusement le coefficient qui change ainsi les valeurs ne les change que dans cette heure d'ivresse. Les personnes qui n'avaient plus d'importance et sur lesquelles nous soufflions comme sur des bulles de savon reprendront le lendemain leur densité ; il faudra essayer de nouveau de se remettre aux travaux qui ne signifiaient plus rien. Chose plus grave encore, cette mathématique du lendemain, la même que celle d'hier et avec les problèmes de laquelle nous nous retrouverons inexorablement aux prises, c'est celle qui nous régit même pendant ces heures-là, sauf pour nous-même. S'il se trouve près de nous une femme vertueuse ou hostile, cette chose si difficile la veille – à savoir, que nous arrivions à lui plaire – nous semble maintenant un million de fois plus aisée sans l'être devenue en rien, car ce n'est qu'à nos propres yeux, à nos propres yeux intérieurs que nous avons changé. Et elle est aussi mécontente à l'instant même que nous nous soyons permis une familiarité que nous le serons le lendemain d'avoir donné cent francs au chasseur, et pour la même raison qui pour nous a été seulement retardée : l'absence d'ivresse.

Je ne connaissais aucune des femmes qui étaient à Rivebelle, et qui, parce qu'elles faisaient partie de mon ivresse comme les reflets font partie du miroir, me paraissaient mille fois plus désirables que la de moins en moins existante Mlle Simonet. Une jeune blonde, seule, à l'air triste, sous son chapeau de paille piqué de fleurs des champs, me regarda un instant d'un air rêveur et me parut agréable. Puis ce fut le tour d'une autre, puis d'une troisième ; enfin d'une brune au teint éclatant. Presque toutes étaient connues, à défaut de moi, par Saint-Loup.

Avant qu'il eût fait la connaissance de sa maîtresse actuelle, il avait en effet tellement vécu dans le monde restreint de la noce, que de toutes les femmes qui dînaient ces soirs-là à Rivebelle et dont beaucoup s'y trouvaient par hasard, étant venues au bord de la mer, certaines pour retrouver leur amant, d'autres pour tâcher d'en trouver un, il n'y en avait guère qu'il ne connût pour avoir passé – lui-même ou tel de ses amis – au moins une nuit avec elles. Il ne les saluait pas si elles étaient avec un homme, et elles, tout en le regardant plus qu'un autre parce que l'indifférence qu'on lui savait pour toute femme qui n'était pas son actrice lui donnait aux yeux de celles-ci un prestige singulier, elles avaient l'air de ne pas le connaître. Et l'une chuchotait : « C'est le petit Saint-Loup. Il paraît qu'il aime toujours sa grue. C'est la grande amour. Quel joli garçon ! Moi je le trouve épatant ; et quel chic ! Il y a tout de même des femmes qui ont une sacrée veine. Et un chic type en tout. Je l'ai bien connu quand j'étais avec d'Orléans. C'était les deux inséparables. Il en faisait une noce à ce moment-là ! Mais ce n'est plus ça ; il ne lui fait pas de queues. Ah ! elle peut dire qu'elle en a une chance. Et je me demande qu'est-ce qu'il peut lui trouver. Il faut qu'il soit tout de même une fameuse truffe. Elle a des pieds comme des bateaux, des moustaches à l'américaine et des dessous sales ! Je crois qu'une petite ouvrière ne voudrait pas de ses pantalons. Regardez-moi un peu quels yeux il a, on se jetterait au feu pour un homme comme ça. Tiens, tais-toi, il m'a reconnue, il rit, oh ! il me connaissait bien. On n'a qu'à lui parler de moi. » Entre elles et lui je surprenais un regard d'intelligence. J'aurais voulu qu'il me présentât à ces femmes, pouvoir leur demander un rendez-vous et qu'elles me l'accordassent même si je n'avais pas pu l'accepter. Car sans cela leur visage resterait éternellement dépourvu dans ma mémoire, de cette partie de lui-même – et comme si elle était cachée par un voile – qui varie avec toutes les femmes, que nous ne pouvons imaginer chez l'une quand nous ne l'y avons pas vue, et qui apparaît seulement dans le regard qui s'adresse à nous et qui acquiesce à notre désir et nous promet qu'il sera satisfait. Et pourtant, même aussi réduit, leur visage était pour moi bien plus que celui des femmes que j'aurais su vertueuses et ne me semblait pas comme le leur, plat, sans dessous, composé d'une pièce unique et sans épaisseur. Sans doute il n'était pas pour moi ce qu'il devait être pour Saint-Loup qui par la mémoire, sous l'indifférence, pour lui transparente, des traits immobiles qui affectaient de ne pas le connaître ou sous la banalité du même salut que l'on eût adressé aussi bien à tout autre, se rappelait, voyait, entre des cheveux défaits, une bouche pâmée et des yeux mi-clos, tout un tableau silencieux comme ceux que les peintres, pour tromper le gros des visiteurs, revêtent d'une toile décente. Certes, pour moi au contraire qui sentais que rien de mon être n'avait pénétré en telle ou telle de ces femmes et n'y serait emporté dans les routes inconnues qu'elle suivrait pendant sa vie, ces visages restaient fermés. Mais c'était déjà assez de savoir qu'ils s'ouvraient pour qu'ils me semblassent d'un prix que je ne leur aurais pas trouvé s'ils n'avaient été que de belles médailles, au lieu de médaillons sous lesquels se cachaient des souvenirs d'amour. Quand à Saint-Loup, tenant à peine en place, quand il était assis, dissimulant sous un sourire d'homme de cour l'avidité d'agir en homme de guerre, à le bien regarder, je me rendais compte combien l'ossature énergique de son visage triangulaire devait être la même que celle de ses ancêtres, plus faite pour un ardent archer que pour un lettré délicat. Sous la peau fine, la construction hardie, l'architecture féodale apparaissaient. Sa tête faisait penser à ces tours d'antiques donjons dont les créneaux inutilisés restent visibles, mais qu'on a aménagées intérieurement en bibliothèque.

En rentrant à Balbec, de telle de ces inconnues à qui il m'avait présenté je me redisais sans m'arrêter une seconde et pourtant sans presque m'en apercevoir : « Quelle femme délicieuse ! » comme on chante un refrain. Certes, ces paroles étaient plutôt dictées par des dispositions nerveuses que par un jugement durable. Il n'en est pas moins vrai que si j'eusse eu mille francs sur moi et qu'il y eût encore des bijoutiers d'ouverts à cette heure-là, j'eusse acheté une bague à l'inconnue. Quand les heures de notre vie se déroulent ainsi que sur des plans trop différents, on se trouve donner trop de soi pour des personnes diverses qui le lendemain vous semblent sans intérêt. Mais on se sent responsable de ce qu'on leur a dit la veille et on veut y faire honneur.
