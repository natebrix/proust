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
        "Swann"
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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Swann s'engagea avec la tristesse de penser qu'Odette ne l'avait jamais gravi » … « le souvenir entrevu d'une boîte au lait vide sur un paillasson lui serra le coeur. »",
      "explanation": "In the midst of a sumptuous social setting, the narrator shows Swann dominated by sadness and the desire to be in the modest environment tied to Odette, which places him locally in a diminished emotional position."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "The passage insists on his pain and lack of Odette, despite the surrounding social pomp."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-441-p-445"
}

### Candidate characters

[
  "Odette",
  "général de Froberville",
  "marquis de Bréauté"
]

### Prior local context (optional)

À quelques pas, un grand gaillard en livrée rêvait, immobile, sculptural, inutile, comme ce guerrier purement décoratif qu'on voit dans les tableaux les plus tumultueux de Mantegna, songer, appuyé sur son bouclier, tandis qu'on se précipite et qu'on s'égorge à côté de lui ; détaché du groupe de ses camarades qui s'empressaient autour de Swann, il semblait aussi résolu à se désintéresser de cette scène, qu'il suivait vaguement de ses yeux glauques et cruels, que si ç'eût été le massacre des Innocents ou le martyre de saint Jacques. Il semblait précisément appartenir à cette race disparue – ou qui peut-être n'exista jamais que dans le retable de San Zeno et les fresques des Eremitani où Swann l'avait approchée et où elle rêve encore – issue de la fécondation d'une statue antique par quelque modèle padouan du Maître ou quelque saxon d'Albert Dürer. Et les mèches de ses cheveux roux crespelés par la nature, mais collés par la brillantine, étaient largement traitées comme elles sont dans la sculpture grecque qu'étudiait sans cesse le peintre de Mantoue, et qui, si dans la création elle ne figure que l'homme, sait du moins tirer de ses simples formes des richesses si variées et comme empruntées à toute la nature vivante, qu'une chevelure, par l'enroulement lisse et les becs aigus de ses boucles, ou dans la superposition du triple et fleurissant diadème de ses tresses, a l'air à la fois d'un paquet d'algues, d'une nichée de colombes, d'un bandeau de jacinthes et d'une torsade de serpents.

### Passage

D'autres encore, colossaux aussi, se tenaient sur les degrés d'un escalier monumental que leur présence décorative et leur immobilité marmoréenne auraient pu faire nommer comme celui du Palais Ducal : « l'Escalier des Géants » et dans lequel Swann s'engagea avec la tristesse de penser qu'Odette ne l'avait jamais gravi. Ah ! avec quelle joie au contraire il eût grimpé les étages noirs, malodorants et casse-cou de la petite couturière retirée, dans le « cinquième » de laquelle il aurait été si heureux de payer plus cher qu'une avant-scène hebdomadaire à l'Opéra le droit de passer la soirée quand Odette y venait, et même les autres jours, pour pouvoir parler d'elle, vivre avec les gens qu'elle avait l'habitude de voir quand il n'était pas là, et qui à cause de cela lui paraissaient recéler, de la vie de sa maîtresse, quelque chose de plus naturel, de plus inaccessible et de plus mystérieux. Tandis que dans cet escalier pestilentiel et désiré de l'ancienne couturière, comme il n'y en avait pas un second pour le service, on voyait le soir devant chaque porte une boîte au lait vide et sale préparée sur le paillasson, dans l'escalier magnifique et dédaigné que Swann montait à ce moment, d'un côté et de l'autre, à des hauteurs différentes, devant chaque anfractuosité que faisait dans le mur la fenêtre de la loge, ou la porte d'un appartement, représentant le service intérieur qu'ils dirigeaient et en faisant hommage aux invités, un concierge, un majordome, un argentier (braves gens qui vivaient le reste de la semaine un peu indépendants dans leur domaine, y dînaient chez eux comme de petits boutiquiers et seraient peut-être demain au service bourgeois d'un médecin ou d'un industriel), attentifs à ne pas manquer aux recommandations qu'on leur avait faites avant de leur laisser endosser la livrée éclatante qu'ils ne revêtaient qu'à de rares intervalles et dans laquelle ils ne se sentaient pas très à leur aise, se tenaient sous l'arcature de leur portail avec un éclat pompeux tempéré de bonhomie populaire, comme des saints dans leur niche ; et un énorme suisse, habillé comme à l'église, frappait les dalles de sa canne au passage de chaque arrivant. Parvenu en haut de l'escalier le long duquel l'avait suivi un domestique à face blême, avec une petite queue de cheveux noués d'un catogan derrière la tête, comme un sacristain de Goya ou un tabellion du répertoire, Swann passa devant un bureau où des valets, assis comme des notaires devant de grands registres, se levèrent et inscrivirent son nom. Il traversa alors un petit vestibule qui – tel que certaines pièces aménagées par leur propriétaire pour servir de cadre à une seule oeuvre d'art, dont elles tirent leur nom, et d'une nudité voulue, ne contiennent rien d'autre – exhibait à son entrée, comme quelque précieuse effigie de Benvenuto Cellini représentant un homme de guet, un jeune valet de pied, le corps légèrement fléchi en avant, dressant sur son hausse-col rouge une figure plus rouge encore d'où s'échappaient des torrents de feu, de timidité et de zèle, et qui, perçant les tapisseries d'Aubusson tendues devant le salon où on écoutait la musique, de son regard impétueux, vigilant, éperdu, avait l'air, avec une impassibilité militaire ou une foi surnaturelle – allégorie de l'alarme, incarnation de l'attente, commémoration du branle-bas – d'épier, ange ou vigie, d'une tour de donjon ou de cathédrale, l'apparition de l'ennemi ou l'heure du Jugement. Il ne restait plus à Swann qu'à pénétrer dans la salle du concert dont un huissier chargé de chaînes lui ouvrit les portes, en s'inclinant, comme il lui aurait remis les clefs d'une ville. Mais il pensait à la maison où il aurait pu se trouver en ce moment même, si Odette l'avait permis, et le souvenir entrevu d'une boîte au lait vide sur un paillasson lui serra le coeur.

Swann retrouva rapidement le sentiment de la laideur masculine, quand, au delà de la tenture de tapisserie, au spectacle des domestiques succéda celui des invités. Mais cette laideur même de visages, qu'il connaissait pourtant si bien, lui semblait neuve depuis que leurs traits – au lieu d'être pour lui des signes pratiquement utilisables à l'identification de telle personne qui lui avait représenté jusque-là un faisceau de plaisirs à poursuivre, d'ennuis à éviter, ou de politesses à rendre – reposaient, coordonnés seulement par des rapports esthétiques, dans l'autonomie de leurs lignes. Et en ces hommes, au milieu desquels Swann se trouva enserré, il n'était pas jusqu'aux monocles que beaucoup portaient (et qui, autrefois, auraient tout au plus permis à Swann de dire qu'ils portaient un monocle), qui, déliés maintenant de signifier une habitude, la même pour tous, ne lui apparussent chacun avec une sorte d'individualité. Peut-être parce qu'il ne regarda le général de Froberville et le marquis de Bréauté qui causaient dans l'entrée que comme deux personnages dans un tableau, alors qu'ils avaient été longtemps pour lui les amis utiles qui l'avaient présenté au Jockey et assisté dans des duels, le monocle du général, resté entre ses paupières comme un éclat d'obus dans sa figure vulgaire, balafrée et triomphale, au milieu du front qu'il éborgnait comme l'oeil unique du cyclope, apparut à Swann comme une blessure monstrueuse qu'il pouvait être glorieux d'avoir reçue, mais qu'il était indécent d'exhiber ; tandis que celui que M. de Bréauté ajoutait, en signe de festivité, aux gants gris perle, au « gibus », à la cravate blanche et substituait au binocle familier (comme faisait Swann lui-même) pour aller dans le monde, portait collé à son revers, comme une préparation d'histoire naturelle sous un microscope, un regard infinitésimal et grouillant d'amabilité, qui ne cessait de sourire à la hauteur des plafonds, à la beauté des fêtes, à l'intérêt des programmes et à la qualité des rafraîchissements.

– Tiens, vous voilà, mais il y a des éternités qu'on ne vous a vu, dit à Swann le général qui, remarquant ses traits tirés et en concluant que c'était peut-être une maladie grave qui l'éloignait du monde, ajouta : « Vous avez bonne mine, vous savez ! » pendant que M. de Bréauté demandait :

– Comment, vous, mon cher, qu'est-ce que vous pouvez bien faire ici ? à un romancier mondain qui venait d'installer au coin de son oeil un monocle, son seul organe d'investigation psychologique et d'impitoyable analyse, et répondit d'un air important et mystérieux, en roulant l'r :

– J'observe.
