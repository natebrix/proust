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
      "canonical_name": "Andrée",
      "surface_forms": [
        "Andrée"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Andrée",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« ...elle me répondit que c'était impossible parce qu'elle avait trouvé sa mère assez mal... » / « ...elle avait accepté un pique-nique à dix lieues d'ici... » / « Bien que ce mensonge fût... je n'aurais pas dû continuer à fréquenter une personne qui en était capable. »",
      "explanation": "The narrator discovers, through Elstir, that Andrée lied to him about an impediment. He explicitly condemns this lie and draws a negative character judgment from it."
    }
  ],
  "status_effects": [
    {
      "character": "Andrée",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "His moral credit locally diminishes in the eyes of the narrator due to an explicit lie."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-416-p-420"
}

### Candidate characters

[
  "Albertine",
  "Elstir",
  "M. Verdurin",
  "Mme de Villeparisis",
  "Octave",
  "Robert de Saint-Loup",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

J'avais cru, il y avait quelques heures, qu'Albertine ne répondrait à mon salut que de loin. Nous venions de nous quitter en faisant le projet d'une excursion ensemble. Je me promis, quand je rencontrerais Albertine, d'être plus hardi avec elle, et je m'étais tracé d'avance le plan de tout ce que je lui dirais et même (maintenant que j'avais tout à fait l'impression qu'elle devait être légère) de tous les plaisirs que je lui demanderais. Mais l'esprit est influençable comme la plante, comme la cellule, comme les éléments chimiques, et le milieu qui le modifie si on l'y plonge, ce sont des circonstances, un cadre nouveau. Devenu différent par le fait de sa présence même, quand je me trouvai de nouveau avec Albertine, je lui dis tout autre chose que ce que j'avais projeté. Puis me souvenant de la tempe enflammée je me demandais si Albertine n'appréciait pas davantage une gentillesse qu'elle saurait être désintéressée. Enfin j'étais embarrassé devant certains de ses regards, de ses sourires. Ils pouvaient signifier moeurs faciles, mais aussi gaieté un peu bête d'une jeune fille sémillante mais ayant un fond d'honnêteté. Une même expression, de figure comme de langage, pouvant comporter diverses acceptions ; j'étais hésitant comme un élève devant les difficultés d'une version grecque.

### Passage

Cette fois-là nous rencontrâmes presque tout de suite la grande Andrée, celle qui avait sauté par-dessus le premier président ; Albertine dut me présenter. Son amie avait des yeux extraordinairement clairs, comme est dans un appartement à l'ombre l'entrée, par la porte ouverte, d'une chambre où donnent le soleil et le reflet verdâtre de la mer illuminée.

Cinq messieurs passèrent que je connaissais très bien de vue depuis que j'étais à Balbec. Je m'étais souvent demandé qui ils étaient. « Ce ne sont pas des gens très chics, me dit Albertine en ricanant d'un air de mépris. Le petit vieux, qui a des gants jaunes, il en a une touche, hein, il dégotte bien, c'est le dentiste de Balbec, c'est un brave type ; le gros, c'est le maire, pas le tout petit gros, celui-là vous devez l'avoir vu, c'est le professeur de danse, il est assez moche aussi, il ne peut pas nous souffrir parce que nous faisons trop de bruit au Casino, que nous démolissons ses chaises, que nous voulons danser sans tapis, aussi il ne nous a jamais donné le prix quoique il n'y a que nous qui sachions danser. Le dentiste est un brave homme, je lui aurais fait bonjour pour faire rager le maître de danse, mais je ne pouvais pas parce qu'il y a avec eux M. de Sainte-Croix, le conseiller général, un homme d'une très bonne famille qui s'est mis du côté des républicains, pour de l'argent ; aucune personne propre ne le salue plus. Il connaît mon oncle, à cause du gouvernement, mais le reste de ma famille lui a tourné le dos. Le maigre avec un imperméable, c'est le chef d'orchestre. Comment, vous ne le connaissez pas ! Il joue divinement. Vous n'avez pas été entendre Cavalleria Rusticana ? Ah ! je trouve ça idéal ! Il donne un concert ce soir, mais nous ne pouvons pas y aller parce que ça a lieu dans la salle de la Mairie. Au casino ça ne fait rien, mais dans la salle de la Mairie d'où on a enlevé le Christ, la mère d'Andrée tomberait en apoplexie si nous y allions. Vous me direz que le mari de ma tante est dans le gouvernement. Mais qu'est-ce que vous voulez ? Ma tante est ma tante. Ce n'est pas pour cela que je l'aime ! Elle n'a jamais eu qu'un désir, se débarrasser de moi. La personne qui m'a vraiment servi de mère, et qui a eu double mérite puisqu'elle ne m'est rien, c'est une amie que j'aime du reste comme une mère. Je vous montrerai sa photo. » Nous fûmes abordés un instant par le champion de golf et joueur de baccara, Octave. Je pensai avoir découvert un lien entre nous, car j'appris dans la conversation qu'il était un peu parent, et de plus assez aimé des Verdurin. Mais il parla avec dédain des fameux mercredis, et ajouta que M. Verdurin ignorait l'usage du smoking, ce qui rendait assez gênant de le rencontrer dans certains « music-halls » où on aurait tant aimé ne pas s'entendre crier : « Bonjour, galopin » par un monsieur en veston et en cravate noire de notaire de village. Puis Octave nous quitta, et bientôt après ce fut le tour d'Andrée, arrivée devant son chalet où elle entra sans que de toute la promenade elle m'eût dit un seul mot. Je regrettai d'autant plus son départ que tandis que je faisais remarquer à Albertine combien son amie avait été froide avec moi, et rapprochais en moi-même cette difficulté qu'Albertine semblait avoir à me lier avec ses amies de l'hostilité contre laquelle, pour exaucer mon souhait, paraissait s'être le premier jour heurté Elstir, passèrent des jeunes filles que je saluai, les demoiselles d'Ambresac, auxquelles Albertine dit aussi bonjour.

Je pensais que ma situation vis-à-vis d'Albertine allait en être améliorée. Elles étaient les filles d'une parente de Mme de Villeparisis et qui connaissait aussi Mme de Luxembourg. M. et Mme d'Ambresac qui avaient une petite villa à Balbec, et excessivement riches, menaient une vie des plus simples, étaient toujours habillés, le mari du même veston, la femme d'une robe sombre. Tous deux faisaient à ma grand'mère d'immenses saluts qui ne menaient à rien. Les filles, très jolies, s'habillaient avec plus d'élégance, mais une élégance de ville et non de plage. Dans leurs robes longues, sous leurs grands chapeaux, elles avaient l'air d'appartenir à une autre humanité qu'Albertine. Celle-ci savait très bien qui elles étaient. « Ah ! vous connaissez les petites d'Ambresac. Hé bien, vous connaissez des gens très chics. Du reste, ils sont très simples, ajouta-t-elle comme si c'était contradictoire. Elles sont très gentilles mais tellement bien élevées qu'on ne les laisse pas aller au Casino, surtout à cause de nous, parce que nous avons trop mauvais genre. Elles vous plaisent ? Dame, ça dépend. C'est tout à fait les petites oies blanches. Ça a peut-être son charme. Si vous aimez les petites oies blanches, vous êtes servi à souhait. Il paraît qu'elles peuvent plaire puisqu'il y en a déjà une de fiancée au marquis de Saint-Loup. Et cela fait beaucoup de peine à la cadette qui était amoureuse de ce jeune homme. Moi, rien que leur manière de parler du bout des lèvres m'énerve. Et puis elles s'habillent d'une manière ridicule. Elles vont jouer au golf en robes de soie. À leur âge elles sont mises plus prétentieusement que des femmes âgées qui savent s'habiller. Tenez Madame Elstir, voilà une femme élégante. » Je répondis qu'elle m'avait semblé vêtue avec beaucoup de simplicité. Albertine se mit à rire. « Elle est mise très simplement, en effet, mais elle s'habille à ravir et pour arriver à ce que vous trouvez de la simplicité, elle dépense un argent fou. » Les robes de Mme Elstir passaient inaperçues aux yeux de quelqu'un qui n'avait pas le goût sûr et sobre des choses de la toilette. Il me faisait défaut. Elstir le possédait au suprême degré, à ce que me dit Albertine. Je ne m'en étais pas douté ni que les choses élégantes mais simples qui emplissaient son atelier étaient des merveilles désirées par lui, qu'il avait suivies de vente en vente, connaissant toute leur histoire, jusqu'au jour où il avait gagné assez d'argent pour pouvoir les posséder. Mais là-dessus Albertine, aussi ignorante que moi, ne pouvait rien m'apprendre. Tandis que pour les toilettes, avertie par un instinct de coquette et peut-être par un regret de jeune fille pauvre qui goûte avec plus de désintéressement, de délicatesse, chez les riches, ce dont elle ne pourra se parer elle-même, elle sut me parler très bien des raffinements d'Elstir, si difficile qu'il trouvait toute femme mal habillée, et que mettant tout un monde dans une proportion, dans une nuance, il faisait faire pour sa femme à des prix fous des ombrelles, des chapeaux, des manteaux qu'il avait appris à Albertine à trouver charmants et qu'une personne sans goût n'eût pas plus remarqués que je n'avais fait. Du reste, Albertine qui avait fait un peu de peinture sans avoir d'ailleurs, elle l'avouait, aucune « disposition », éprouvait une grande admiration pour Elstir, et grâce à ce qu'il lui avait dit et montré, s'y connaissait en tableaux d'une façon qui contrastait fort avec son enthousiasme pour Cavalleria Rusticana. C'est qu'en réalité, bien que cela ne se vît guère encore, elle était très intelligente et dans les choses qu'elle disait, la bêtise n'était pas sienne, mais celle de son milieu et de son âge. Elstir avait eu sur elle une influence heureuse mais partielle. Toutes les formes de l'intelligence n'étaient pas arrivées chez Albertine au même degré de développement. Le goût de la peinture avait presque rattrapé celui de la toilette et de toutes les formes de l'élégance, mais n'avait pas été suivi par le goût de la musique qui restait fort en arrière.

Albertine avait beau savoir qui étaient les Ambresac, comme qui peut le plus ne peut pas forcément le moins, je ne la trouvai pas, après que j'eusse salué ces jeunes filles, plus disposée à me faire connaître ses amies. « Vous êtes bien bon d'attacher, de leur donner de l'importance. Ne faites pas attention à elles, ce n'est rien du tout. Qu'est-ce que ces petites gosses peuvent compter pour un homme de votre valeur. Andrée au moins est remarquablement intelligente. C'est une bonne petite fille, quoique parfaitement fantasque, mais les autres sont vraiment très stupides. » Après avoir quitté Albertine, je ressentis tout à coup beaucoup de chagrin que Saint-Loup m'eût caché ses fiançailles, et fît quelque chose d'aussi mal que se marier sans avoir rompu avec sa maîtresse. Peu de jours après pourtant, je fus présenté à Andrée et comme elle parla assez longtemps, j'en profitai pour lui dire que je voudrais bien la voir le lendemain, mais elle me répondit que c'était impossible parce qu'elle avait trouvé sa mère assez mal et ne voulait pas la laisser seule. Deux jours après, étant allé voir Elstir, il me dit la sympathie très grande qu'Andrée avait pour moi ; comme je lui répondais : « Mais c'est moi qui ai eu beaucoup de sympathie pour elle dès le premier jour, je lui avais demandé à la revoir le lendemain, mais elle ne pouvait pas. – Oui, je sais, elle me l'a raconté, me dit Elstir, elle l'a assez regretté, mais elle avait accepté un pique-nique à dix lieues d'ici où elle devait aller en break et elle ne pouvait plus se décommander. » Bien que ce mensonge fût, Andrée me connaissant si peu, fort insignifiant, je n'aurais pas dû continuer à fréquenter une personne qui en était capable. Car ce que les gens ont fait, ils le recommencent indéfiniment. Et qu'on aille voir chaque année un ami qui les premières fois n'a pu venir à votre rendez-vous, ou s'est enrhumé, on le retrouvera avec un autre rhume qu'il aura pris, on le manquera à un autre rendez-vous où il ne sera pas venu, pour une même raison permanente à la place de laquelle il croit voir des raisons variées, tirées des circonstances.

Un des matins qui suivirent celui où Andrée m'avait dit qu'elle était obligée de rester auprès de sa mère, je faisais quelques pas avec Albertine que j'avais aperçue, élevant au bout d'un cordonnet un attribut bizarre qui la faisait ressembler à l'« Idolâtrie » de Giotto ; il s'appelle d'ailleurs un « diabolo » et est tellement tombé en désuétude que devant le portrait d'une jeune fille en tenant un, les commentateurs de l'avenir pourront disserter comme devant telle figure allégorique de l'Arêna, sur ce qu'elle a dans la main. Au bout d'un moment, leur amie à l'air pauvre et dur, qui avait ricané le premier jour d'un air si méchant : « Il me fait de la peine ce pauvre vieux » en parlant du vieux monsieur effleuré par les pieds légers d'Andrée, vint dire à Albertine : « Bonjour, je vous dérange ? » Elle avait ôté son chapeau qui la gênait, et ses cheveux comme une variété végétale ravissante et inconnue reposaient sur son front dans la minutieuse délicatesse de leur foliation. Albertine, peut-être irritée de la voir tête nue, ne répondit rien, garda un silence glacial malgré lequel l'autre resta, tenue à distance de moi par Albertine qui s'arrangeait à certains instants pour être seule avec elle, à d'autres pour marcher avec moi, en la laissant derrière. Je fus obligé pour qu'elle me présentât de le lui demander devant l'autre. Alors au moment où Albertine me nomma, sur la figure et dans les yeux bleus de cette jeune fille à qui j'avais trouvé un air si cruel quand elle avait dit : « Ce pauvre vieux, y m'fait d'la peine », je vis passer et briller un sourire cordial, aimant, et elle me tendit la main. Ses cheveux étaient dorés, et ne l'étaient pas seuls ; car si ses joues étaient roses et ses yeux bleus, c'était comme le ciel encore empourpré du matin où partout pointe et brille l'or.
