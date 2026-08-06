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
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "le père du narrateur",
      "surface_forms": [
        "le père du narrateur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "le père du narrateur",
      "target": "la grand-mère",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.78,
      "evidence": "La grand-mère avait été obligée de renoncer à ce projet, sur la défense de le père du narrateur, qui savait... combien on pouvait pronostiquer de trains manqués, de bagages perdus, de maux de gorge et de contraventions.",
      "explanation": "The narrator's father opposes a practical veto to the grandmother's artistic project, forcing her to abandon it; this locally reduces her initiative and influence in organizing the trip."
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
      "confidence": 0.78,
      "explanation": "Her travel plan is dismissed and she yields the initiative, which excludes her from the dominant decision regarding the departure."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-6-p-10"
}

### Candidate characters

[
  "Legrandin",
  "Norpois",
  "docteur Cottard",
  "la Berma",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Malheureusement ces lieux merveilleux que sont les gares, d'où l'on part pour une destination éloignée, sont aussi des lieux tragiques, car si le miracle s'y accomplit grâce auquel les pays qui n'avaient encore d'existence que dans notre pensée vont être ceux au milieu desquels nous vivrons, pour cette raison même il faut renoncer au sortir de la salle d'attente à retrouver tout à l'heure la chambre familière où l'on était il y a un instant encore. Il faut laisser toute espérance de rentrer coucher chez soi, une fois qu'on s'est décidé à pénétrer dans l'antre empesté par où l'on accède au mystère, dans un de ces grands ateliers vitrés, comme celui de Saint-Lazare où j'allais chercher le train de Balbec, et qui déployait au-dessus de la ville éventrée un de ces immenses ciels crus et gros de menaces amoncelées de drame, pareils à certains ciels, d'une modernité presque parisienne, de Mantegna ou de Véronèse, et sous lequel ne pouvait s'accomplir que quelque acte terrible et solennel comme un départ en chemin de fer ou l'érection de la Croix.

### Passage

Tant que je m'étais contenté d'apercevoir du fond de mon lit de Paris l'église persane de Balbec au milieu des flocons de la tempête, aucune objection à ce voyage n'avait été faite par mon corps. Elles avaient commencé seulement quand il avait compris qu'il serait de la partie et que le soir de l'arrivée on me conduirait à « ma » chambre qui lui serait inconnue. Sa révolte était d'autant plus profonde que la veille même du départ j'avais appris que ma mère ne nous accompagnerait pas, mon père, retenu au ministère jusqu'au moment où il partirait pour l'Espagne avec Norpois, ayant préféré louer une maison dans les environs de Paris. D'ailleurs la contemplation de Balbec ne me semblait pas moins désirable parce qu'il fallait l'acheter au prix d'un mal qui au contraire me semblait figurer et garantir la réalité de l'impression que j'allais chercher, impression que n'aurait remplacée aucun spectacle prétendu équivalent, aucun « panorama » que j'eusse pu aller voir sans être empêché par cela même de rentrer dormir dans mon lit. Ce n'était pas la première fois que je sentais que ceux qui aiment et ceux qui ont du plaisir ne sont pas les mêmes. Je croyais désirer aussi profondément Balbec que le docteur qui me soignait et qui me dit, s'étonnant, le matin du départ, de mon air malheureux : « Je vous réponds que si je pouvais seulement trouver huit jours pour aller prendre le frais au bord de la mer, je ne me ferais pas prier. Vous allez avoir les courses, les régates, ce sera exquis. » Pour moi j'avais déjà appris et même bien avant d'aller entendre la Berma, que quelle que fût la chose que j'aimerais, elle ne serait jamais placée qu'au terme d'une poursuite douloureuse au cours de laquelle il me faudrait d'abord sacrifier mon plaisir à ce bien suprême, au lieu de l'y chercher.

Ma grand'mère concevait naturellement notre départ d'une façon un peu différente et, toujours aussi désireuse qu'autrefois de donner aux présents qu'on me faisait un caractère artistique, avait voulu pour m'offrir de ce voyage une « épreuve » en partie ancienne, que nous refissions moitié en chemin de fer, moitié en voiture le trajet qu'avait suivi Mme de Sévigné quand elle était allée de Paris à « L'Orient » en passant par Chaulnes et par « le Pont Audemer ». Mais ma grand'mère avait été obligée de renoncer à ce projet, sur la défense de mon père, qui savait, quand elle organisait un déplacement en vue de lui faire rendre tout le profit intellectuel qu'il pouvait comporter, combien on pouvait pronostiquer de trains manqués, de bagages perdus, de maux de gorge et de contraventions. Elle se réjouissait du moins à la pensée que jamais, au moment d'aller sur la plage, nous ne serions exposés à en être empêchés par la survenue de ce que sa chère Sévigné appelle une chienne de carrossée, puisque nous ne connaîtrions personne à Balbec, Legrandin ne nous ayant pas offert de lettre d'introduction pour sa soeur. (Abstention qui n'avait pas été appréciée de même par mes tantes Céline et Victoire, lesquelles, ayant connu jeune fille celle qu'elles n'avaient appelée jusqu'ici, pour marquer cette intimité d'autrefois, que « Renée de Cambremer », et possédant encore d'elle de ces cadeaux qui meublent une chambre et la conversation mais auxquels la réalité actuelle ne correspond pas, croyaient venger notre affront en ne prononçant plus jamais, chez Mme Legrandin mère, le nom de sa fille, et se bornant à se congratuler une fois sorties par des phrases comme : « Je n'ai pas fait allusion à qui tu sais », « je crois qu'on aura compris ».)

Donc nous partirions simplement de Paris par ce train de une heure vingt-deux que je m'étais plu trop longtemps à chercher dans l'indicateur des chemins de fer, où il me donnait chaque fois l'émotion, presque la bienheureuse illusion du départ, pour ne pas me figurer que je le connaissais. Comme la détermination dans notre imagination des traits d'un bonheur tient plutôt à l'identité des désirs qu'il nous inspire qu'à la précision des renseignements que nous avons sur lui, je croyais connaître celui-là dans ses détails, et je ne doutais pas que j'éprouverais dans le wagon un plaisir spécial quand la journée commencerait à fraîchir, que je contemplerais tel effet à l'approche d'une certaine station ; si bien que ce train, réveillant toujours en moi les images des mêmes villes que j'enveloppais dans la lumière de ces heures de l'après-midi qu'il traverse, me semblait différent de tous les autres trains ; et j'avais fini, comme on fait souvent pour un être qu'on n'a jamais vu mais dont on se plaît à s'imaginer qu'on a conquis l'amitié, par donner une physionomie particulière et immuable à ce voyageur artiste et blond qui m'aurait emmené sur sa route, et à qui j'aurais dit adieu au pied de la cathédrale de Saint-Lô, avant qu'il se fût éloigné vers le couchant.

Comme ma grand'mère ne pouvait se résoudre à aller « tout bêtement » à Balbec, elle s'arrêterait vingt-quatre heures chez une de ses amies, de chez laquelle je repartirais le soir même pour ne pas déranger, et aussi de façon à voir dans la journée du lendemain l'église de Balbec, qui, avions-nous appris, était assez éloignée de Balbec-Plage, et où je ne pourrais peut-être pas aller ensuite au début de mon traitement de bains. Et peut-être était-il moins pénible pour moi de sentir l'objet admirable de mon voyage placé avant la cruelle première nuit où j'entrerais dans une demeure nouvelle et accepterais d'y vivre. Mais il avait fallu d'abord quitter l'ancienne ; ma mère avait arrangé de s'installer ce jour-là même à Saint-Cloud, et elle avait pris, ou feint de prendre, toutes ses dispositions pour y aller directement après nous avoir conduits à la gare, sans avoir à repasser par la maison où elle craignait que je ne voulusse, au lieu de partir pour Balbec, rentrer avec elle. Et même sous le prétexte d'avoir beaucoup à faire dans la maison qu'elle venait de louer et d'être à court de temps, en réalité pour m'éviter la cruauté de ce genre d'adieux, elle avait décidé de ne pas rester avec nous jusqu'à ce départ du train où, dissimulée auparavant dans des allées et venues et des préparatifs qui n'engagent pas définitivement, une séparation apparaît brusquement impossible à souffrir, alors qu'elle n'est déjà plus possible à éviter, concentrée tout entière dans un instant immense de lucidité impuissante et suprême.

Pour la première fois je sentais qu'il était possible que ma mère vécût sans moi, autrement que pour moi, d'une autre vie. Elle allait habiter de son côté avec mon père à qui peut-être elle trouvait que ma mauvaise santé, ma nervosité, rendaient l'existence un peu compliquée et triste. Cette séparation me désolait davantage parce que je me disais qu'elle était probablement pour ma mère le terme des déceptions successives que je lui avais causées, qu'elle m'avait tues et après lesquelles elle avait compris la difficulté de vacances communes ; et peut-être aussi le premier essai d'une existence à laquelle elle commençait à se résigner pour l'avenir, au fur et à mesure que les années viendraient pour mon père et pour elle, d'une existence où je la verrais moins, où, ce qui même dans mes cauchemars ne m'était jamais apparu, elle serait déjà pour moi un peu étrangère, une dame qu'on verrait rentrer seule dans une maison où je ne serais pas, demandant au concierge s'il n'y avait pas de lettres de moi.
