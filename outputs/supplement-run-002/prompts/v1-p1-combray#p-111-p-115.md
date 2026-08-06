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
      "canonical_name": "oncle Adolphe",
      "surface_forms": [
        "oncle Adolphe",
        "mon oncle"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "oncle Adolphe",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.82,
      "evidence": "Il se plaignait en ronchonnant que je n'étais pas venu depuis longtemps, qu'on l'abandonnait… Mon oncle se plongeait alors dans une méditation… Enfin… mon oncle prononçait infailliblement: «Deux heures et quart», que le valet… répétait avec étonnement, mais sans discuter.",
      "explanation": "The narrator presents Adolphe as plaintive, dated, and predictable; the ritualized scene of «méditation» that always comes to an end at the same hour, and the valet’s feigned astonishment, make him slightly ridiculous."
    }
  ],
  "status_effects": [
    {
      "character": "oncle Adolphe",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.8,
      "explanation": "Locally, he is diminished by an ironic depiction of his manias and his dependence on a domestic staging."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-111-p-115"
}

### Candidate characters

[
  "le grand-père du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Enfin la mère du narrateur me disait : « Voyons, ne reste pas ici indéfiniment, monte dans ta chambre si tu as trop chaud dehors, mais va d'abord prendre l'air un instant pour ne pas lire en sortant de table. » J'allais m'asseoir près de la pompe et de son auge, souvent ornée, comme un fond gothique, d'une salamandre, qui sculptait sur la pierre fruste le relief mobile de son corps allégorique et fuselé, sur le banc sans dossier ombragé d'un lilas, dans ce petit coin du jardin qui s'ouvrait par une porte de service sur la rue du Saint-Esprit et de la terre peu soignée duquel s'élevait par deux degrés, en saillie de la maison, et comme une construction indépendante, l'arrière-cuisine. On apercevait son dallage rouge et luisant comme du porphyre. Elle avait moins l'air de l'antre de Françoise que d'un petit temple de Vénus. Elle regorgeait des offrandes du crémier, du fruitier, de la marchande de légumes, venus parfois de hameaux assez lointains pour lui dédier les prémices de leurs champs. Et son faîte était toujours couronné du roucoulement d'une colombe.

### Passage

Autrefois, je ne m'attardais pas dans le bois consacré qui l'entourait, car, avant de monter lire, j'entrais dans le petit cabinet de repos que mon oncle Adolphe, un frère de mon grand-père, ancien militaire qui avait pris sa retraite comme commandant, occupait au rez-de-chaussée, et qui, même quand les fenêtres ouvertes laissaient entrer la chaleur, sinon les rayons du soleil qui atteignaient rarement jusque-là, dégageait inépuisablement cette odeur obscure et fraîche, à la fois forestière et ancien régime, qui fait rêver longuement les narines quand on pénètre dans certains pavillons de chasse abandonnés. Mais depuis nombre d'années je n'entrais plus dans le cabinet de mon oncle Adolphe, ce dernier ne venant plus à Combray à cause d'une brouille qui était survenue entre lui et ma famille, par ma faute, dans les circonstances suivantes :

Une ou deux fois par mois, à Paris, on m'envoyait lui faire une visite, comme il finissait de déjeuner, en simple vareuse, servi par son domestique en veste de travail de coutil rayé violet et blanc. Il se plaignait en ronchonnant que je n'étais pas venu depuis longtemps, qu'on l'abandonnait ; il m'offrait un massepain ou une mandarine, nous traversions un salon dans lequel on ne s'arrêtait jamais, où on ne faisait jamais de feu, dont les murs étaient ornés de moulures dorées, les plafonds peints d'un bleu qui prétendait imiter le ciel et les meubles capitonnés en satin comme chez mes grands-parents, mais jaune ; puis nous passions dans ce qu'il appelait son cabinet de « travail » aux murs duquel étaient accrochées de ces gravures représentant sur fond noir une déesse charnue et rose conduisant un char, montée sur un globe, ou une étoile au front, qu'on aimait sous le second Empire parce qu'on leur trouvait un air pompéien, puis qu'on détesta, et qu'on recommence à aimer pour une seule et même raison, malgré les autres qu'on donne, et qui est qu'elles ont l'air second Empire. Et je restais avec mon oncle jusqu'à ce que son valet de chambre vînt lui demander, de la part du cocher, pour quelle heure celui-ci devait atteler. Mon oncle se plongeait alors dans une méditation qu'aurait craint de troubler d'un seul mouvement son valet de chambre émerveillé, et dont il attendait avec curiosité le résultat, toujours identique. Enfin, après une hésitation suprême, mon oncle prononçait infailliblement ces mots : « Deux heures et quart », que le valet de chambre répétait avec étonnement, mais sans discuter : « Deux heures et quart ? bien...je vais le dire... »

À cette époque j'avais l'amour du théâtre, amour platonique, car mes parents ne m'avaient encore jamais permis d'y aller, et je me représentais d'une façon si peu exacte les plaisirs qu'on y goûtait que je n'étais pas éloigné de croire que chaque spectateur regardait comme dans un stéréoscope un décor qui n'était que pour lui, quoique semblable au millier d'autres que regardait, chacun pour soi, le reste des spectateurs.

Tous les matins je courais jusqu'à la colonne Moriss pour voir les spectacles qu'elle annonçait. Rien n'était plus désintéressé et plus heureux que les rêves offerts à mon imagination par chaque pièce annoncée, et qui étaient conditionnés à la fois par les images inséparables des mots qui en composaient le titre et aussi de la couleur des affiches encore humides et boursouflées de colle sur lesquelles il se détachait. Si ce n'est une de ces oeuvres étranges comme le Testament de César Girodot et Oedipe-Roi lesquelles s'inscrivaient, non sur l'affiche verte de l'Opéra-Comique, mais sur l'affiche lie de vin de la Comédie-Française, rien ne me paraissait plus différent de l'aigrette étincelante et blanche des Diamants de la Couronne que le satin lisse et mystérieux du Domino Noir, et, mes parents m'ayant dit que quand j'irais pour la première fois au théâtre j'aurais à choisir entre ces deux pièces, cherchant à approfondir successivement le titre de l'une et le titre de l'autre, puisque c'était tout ce que je connaissais d'elles, pour tâcher de saisir en chacun le plaisir qu'il me promettait et de le comparer à celui que recélait l'autre, j'arrivais à me représenter avec tant de force, d'une part une pièce éblouissante et fière, de l'autre une pièce douce et veloutée, que j'étais aussi incapable de décider laquelle aurait ma préférence, que si, pour le dessert, on m'avait donné à opter entre du riz à l'Impératrice et de la crème au chocolat.

Toutes mes conversations avec mes camarades portaient sur ces acteurs dont l'art, bien qu'il me fût encore inconnu, était la première forme, entre toutes celles qu'il revêt, sous laquelle se laissait pressentir par moi l'Art. Entre la manière que l'un ou l'autre avait de débiter, de nuancer une tirade, les différences les plus minimes me semblaient avoir une importance incalculable. Et, d'après ce que l'on m'avait dit d'eux, je les classais par ordre de talent, dans des listes que je me récitais toute la journée, et qui avaient fini par durcir dans mon cerveau et par le gêner de leur inamovibilité.
