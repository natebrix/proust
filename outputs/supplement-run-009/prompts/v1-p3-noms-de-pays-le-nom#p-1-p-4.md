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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« afin d'apprendre de lui si c'était le point le mieux choisi... il m'avait répondu: “Je crois bien que je connais Balbec ! L'église de Balbec... on dirait de l'art persan.” »",
      "explanation": "The narrator turns to Swann as an authority and records his learned response; Swann’s knowledge reframes Balbec for the narrator and is presented with evident respect."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Swann gains local authority as a knowledgeable guide whose remarks reshape the narrator’s imagination of Balbec."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-1-p-4"
}

### Candidate characters

[
  "Françoise",
  "Legrandin",
  "le narrateur"
]

### Prior local context (optional)

(none provided)

### Passage

Parmi les chambres dont j'évoquais le plus souvent l'image dans mes nuits d'insomnie, aucune ne ressemblait moins aux chambres de Combray, saupoudrées d'une atmosphère grenue, pollinisée, comestible et dévote, que celle du Grand-Hôtel de la Plage, à Balbec, dont les murs passés au ripolin contenaient, comme les parois polies d'une piscine où l'eau bleuit, un air pur, azuré et salin. Le tapissier bavarois qui avait été chargé de l'aménagement de cet hôtel avait varié la décoration des pièces et sur trois côtés fait courir le long des murs, dans celle que je me trouvai habiter, des bibliothèques basses, à vitrines en glace, dans lesquelles, selon la place qu'elles occupaient, et par un effet qu'il n'avait pas prévu, telle ou telle partie du tableau changeant de la mer se reflétait, déroulant une frise de claires marines, qu'interrompaient seuls les pleins de l'acajou. Si bien que toute la pièce avait l'air d'un de ces dortoirs modèles qu'on présente dans les expositions « modern style » du mobilier, où ils sont ornés d'oeuvres d'art qu'on a supposées capables de réjouir les yeux de celui qui couchera là, et auxquelles on a donné des sujets en rapport avec le genre de site où l'habitation doit se trouver.

Mais rien ne ressemblait moins non plus à ce Balbec réel que celui dont j'avais souvent rêvé, les jours de tempête, quand le vent était si fort que Françoise en me menant aux Champs-Élysées me recommandait de ne pas marcher trop près des murs pour ne pas recevoir de tuiles sur la tête, et parlait en gémissant des grands sinistres et naufrages annoncés par les journaux. Je n'avais pas de plus grand désir que de voir une tempête sur la mer, moins comme un beau spectacle que comme un moment dévoilé de la vie réelle de la nature ; ou plutôt il n'y avait pour moi de beaux spectacles que ceux que je savais qui n'étaient pas artificiellement combinés pour mon plaisir, mais étaient nécessaires, inchangeables – les beautés des paysages ou du grand art. Je n'étais curieux, je n'étais avide de connaître que ce que je croyais plus vrai que moi-même, ce qui avait pour moi le prix de me montrer un peu de la pensée d'un grand génie, ou de la force ou de la grâce de la nature telle qu'elle se manifeste livrée à elle-même, sans l'intervention des hommes. De même que le beau son de sa voix, isolément reproduit par le phonographe, ne nous consolerait pas d'avoir perdu notre mère, de même une tempête mécaniquement imitée m'aurait laissé aussi indifférent que les fontaines lumineuses de l'Exposition. Je voulais aussi, pour que la tempête fût absolument vraie, que le rivage lui-même fût un rivage naturel, non une digue récemment créée par une municipalité. D'ailleurs la nature, par tous les sentiments qu'elle éveillait en moi, me semblait ce qu'il y avait de plus opposé aux productions mécaniques des hommes. Moins elle portait leur empreinte et plus elle offrait d'espace à l'expansion de mon coeur. Or j'avais retenu le nom de Balbec que nous avait cité Legrandin, comme d'une plage toute proche de « ces côtes funèbres, fameuses par tant de naufrages qu'enveloppent six mois de l'année le linceul des brumes et l'écume des vagues ».

« On y sent encore sous ses pas, disait-il, bien plus qu'au Finistère lui-même (et quand bien même des hôtels s'y superposeraient maintenant sans pouvoir y modifier la plus antique ossature de la terre), on y sent la véritable fin de la terre française, européenne, de la Terre antique. Et c'est le dernier campement de pêcheurs, pareils à tous les pêcheurs qui ont vécu depuis le commencement du monde, en face du royaume éternel des brouillards de la mer et des ombres. » Un jour qu'à Combray j'avais parlé de cette plage de Balbec devant Swann afin d'apprendre de lui si c'était le point le mieux choisi pour voir les plus fortes tempêtes, il m'avait répondu : « Je crois bien que je connais Balbec ! L'église de Balbec, du XIIe et XIIIe siècle, encore à moitié romane, est peut-être le plus curieux échantillon du gothique normand, et si singulière ! on dirait de l'art persan. » Et ces lieux qui jusque-là ne m'avaient semblé que de la nature immémoriale, restée contemporaine des grands phénomènes géologiques – et tout aussi en dehors de l'histoire humaine que l'Océan ou la grande Ourse, avec ces sauvages pêcheurs pour qui, pas plus que pour les baleines, il n'y eut de moyen âge – ç'avait été un grand charme pour moi de les voir tout d'un coup entrés dans la série des siècles, ayant connu l'époque romane, et de savoir que le trèfle gothique était venu nervurer aussi ces rochers sauvages à l'heure voulue, comme ces plantes frêles mais vivaces qui, quand c'est le printemps, étoilent çà et là la neige des pôles. Et si le gothique apportait à ces lieux et à ces hommes une détermination qui leur manquait, eux aussi lui en conféraient une en retour. J'essayais de me représenter comment ces pêcheurs avaient vécu, le timide et insoupçonné essai de rapports sociaux qu'ils avaient tenté là, pendant le moyen âge, ramassés sur un point des côtes d'Enfer, aux pieds des falaises de la mort ; et le gothique me semblait plus vivant maintenant que, séparé des villes où je l'avais toujours imaginé jusque-là, je pouvais voir comment, dans un cas particulier, sur des rochers sauvages, il avait germé et fleuri en un fin clocher. On me mena voir des reproductions des plus célèbres statues de Balbec – les apôtres moutonnants et camus, la Vierge du porche, et de joie ma respiration s'arrêtait dans ma poitrine quand je pensais que je pourrais les voir se modeler en relief sur le brouillard éternel et salé. Alors, par les soirs orageux et doux de février – le vent, soufflant dans mon coeur, qu'il ne faisait pas trembler moins fort que la cheminée de ma chambre – le projet d'un voyage à Balbec mêlait en moi le désir de l'architecture gothique avec celui d'une tempête sur la mer.

J'aurais voulu prendre dès le lendemain le beau train généreux d'une heure vingt-deux dont je ne pouvais jamais sans que mon coeur palpitât lire, dans les réclames des compagnies de chemin de fer, dans les annonces de voyages circulaires, l'heure de départ : elle me semblait inciser à un point précis de l'après-midi une savoureuse entaille, une marque mystérieuse à partir de laquelle les heures déviées conduisaient bien encore au soir, au matin du lendemain, mais qu'on verrait, au lieu de Paris, dans l'une de ces villes par où le train passe et entre lesquelles il nous permettait de choisir ; car il s'arrêtait à Bayeux, à Coutances, à Vitré, à Questambert, à Pontorson, à Balbec, à Lannion, à Lamballe, à Benodet, à Pont-Aven, à Quimperlé, et s'avançait magnifiquement surchargé de noms qu'il m'offrait et entre lesquels je ne savais lequel j'aurais préféré, par impossibilité d'en sacrifier aucun. Mais sans même l'attendre, j'aurais pu en m'habillant à la hâte partir le soir même, si mes parents me l'avaient permis, et arriver à Balbec quand le petit jour se lèverait sur la mer furieuse, contre les écumes envolées de laquelle j'irais me réfugier dans l'église de style persan. Mais à l'approche des vacances de Pâques, quand mes parents m'eurent promis de me les faire passer une fois dans le nord de l'Italie, voilà qu'à ces rêves de tempête dont j'avais été rempli tout entier, ne souhaitant voir que des vagues accourant de partout, toujours plus haut, sur la côte la plus sauvage, près d'églises escarpées et rugueuses comme des falaises et dans les tours desquelles crieraient les oiseaux de mer, voilà que tout à coup les effaçant, leur ôtant tout charme, les excluant parce qu'il leur était opposé et n'aurait pu que les affaiblir, se substituait en moi le rêve contraire du printemps le plus diapré, non pas le printemps de Combray qui piquait encore aigrement avec toutes les aiguilles du givre, mais celui qui couvrait déjà de lys et d'anémones les champs de Fiesole et éblouissait Florence de fonds d'or pareils à ceux de l'Angelico. Dès lors, seuls les rayons, les parfums, les couleurs me semblaient avoir du prix ; car l'alternance des images avait amené en moi un changement de front du désir, et, aussi brusque que ceux qu'il y a parfois en musique, un complet changement de ton dans ma sensibilité. Puis il arriva qu'une simple variation atmosphérique suffit à provoquer en moi cette modulation sans qu'il y eût besoin d'attendre le retour d'une saison. Car souvent dans l'une on trouve égaré un jour d'une autre, qui nous y fait vivre, en évoque aussitôt, en fait désirer les plaisirs particuliers et interrompt les rêves que nous étions en train de faire, en plaçant, plus tôt ou plus tard qu'à son tour, ce feuillet détaché d'un autre chapitre, dans le calendrier interpolé du Bonheur. Mais bientôt, comme ces phénomènes naturels dont notre confort ou notre santé ne peuvent tirer qu'un bénéfice accidentel et assez mince jusqu'au jour où la science s'empare d'eux, et, les produisant à volonté, remet en nos mains la possibilité de leur apparition, soustraite à la tutelle et dispensée de l'agrément du hasard, de même la production de ces rêves d'Atlantique et d'Italie cessa d'être soumise uniquement aux changements des saisons et du temps. Je n'eus besoin pour les faire renaître que de prononcer ces noms : Balbec, Venise, Florence, dans l'intérieur desquels avait fini par s'accumuler le désir que m'avaient inspiré les lieux qu'ils désignaient. Même au printemps, trouver dans un livre le nom de Balbec suffisait à réveiller en moi le désir des tempêtes et du gothique normand ; même par un jour de tempête le nom de Florence ou de Venise me donnait le désir du soleil, des lys, du palais des Doges et de Sainte-Marie-des-Fleurs.
