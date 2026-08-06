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
        "Swann",
        "M.",
        "son père"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "uncertain",
      "confidence": 0.72,
      "evidence": "« son père avait haussé les épaules en disant: “Tout cela ne signifie rien, cela ne fait que prouver combien j’ai raison.” » … « il fallait que ces nobles sentiments, il ne les eût lui-même jamais ressentis, ce qui devait le rendre incapable de les comprendre chez les autres. »",
      "explanation": "After Swann dismisses the narrator’s heartfelt letter, the narrator portrays Swann as unjust and incapable of recognizing noble sentiments, locally lowering Swann’s moral/appraisal standing."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "Within this passage, Swann is framed negatively (dismissive, unable to grasp generous feelings), which locally diminishes how he is appraised."
    }
  ],
  "ambiguities": [
    "The narrator signals temporal distance from his younger indignation (« je n’en doutais pas alors »), which may imply an ironized or retrospective nuance to the blame."
  ],
  "unit_id": "v2-p1-autour-de-mme-swann#p-101-p-105"
}

### Candidate characters

[
  "Gilberte",
  "Norpois",
  "Odette",
  "la Berma",
  "le narrateur"
]

### Prior local context (optional)

Les visites finies (la grand-mère dispensait que nous en fissions chez elle, comme nous y dînions ce jour-là), je courus jusqu'aux Champs-Élysées porter à notre marchande, pour qu'elle la remît à la personne qui venait plusieurs fois par semaine de chez les Swann y chercher du pain d'épices, la lettre que dès le jour où mon amie m'avait fait tant de peine j'avais décidé de lui envoyer au nouvel an, et dans laquelle je lui disais que notre amitié ancienne disparaissait avec l'année finie, que j'oubliais mes griefs et mes déceptions et qu'à partir du 1er janvier, c'était une amitié neuve que nous allions bâtir, si solide que rien ne la détruirait, si merveilleuse que j'espérais que Gilberte mettrait quelque coquetterie à lui garder toute sa beauté et à m'avertir à temps, comme je promettais de le faire moi-même, aussitôt que surviendrait le moindre péril qui pourrait l'endommager. En rentrant, Françoise me fit arrêter, au coin de la rue Royale, devant un étalage en plein vent où elle choisit, pour ses propres étrennes, des photographies de Pie IX et de Raspail et où, pour ma part, j'en achetai une de la Berma. Les innombrables admirations qu'excitait l'artiste donnaient quelque chose d'un peu pauvre à ce visage unique qu'elle avait pour y répondre, immuable et précaire comme ce vêtement des personnes qui n'en ont pas de rechange, et où elle ne pouvait exhiber toujours que le petit pli au-dessus de la lèvre supérieure, le relèvement des sourcils, quelques autres particularités physiques toujours les mêmes qui, en somme, étaient à la merci d'une brûlure ou d'un choc. Ce visage, d'ailleurs, ne m'eût pas à lui seul semblé beau, mais il me donnait l'idée et, par conséquent, l'envie de l'embrasser à cause de tous les baisers qu'il avait dû supporter, et que, du fond de la « carte-album », il semblait appeler encore par ce regard coquettement tendre et ce sourire artificieusement ingénu. Car la Berma devait ressentir effectivement pour bien des jeunes hommes ces désirs qu'elle avouait sous le couvert du personnage de Phèdre, et dont tout, même le prestige de son nom qui ajoutait à sa beauté et prorogeait sa jeunesse, devait lui rendre l'assouvissement si facile. Le soir tombait, je m'arrêtai devant une colonne de théâtre où était affichée la représentation que la Berma donnait pour le 1er janvier. Il soufflait un vent humide et doux. C'était un temps que je connaissais ; j'eus la sensation et le pressentiment que le jour de l'an n'était pas un jour différent des autres, qu'il n'était pas le premier d'un monde nouveau où j'aurais pu, avec une chance encore intacte, refaire la connaissance de Gilberte comme au temps de la Création, comme s'il n'existait pas encore de passé, comme si eussent été anéanties, avec les indices qu'on aurait pu en tirer pour l'avenir, les déceptions qu'elle m'avait parfois causées : un nouveau monde où rien ne subsistât de l'ancien... rien qu'une chose : mon désir que Gilberte m'aimât. Je compris que si mon coeur souhaitait ce renouvellement autour de lui d'un univers qui ne l'avait pas satisfait, c'est que lui, mon coeur, n'avait pas changé, et je me dis qu'il n'y avait pas de raison pour que celui de Gilberte eût changé davantage ; je sentis que cette nouvelle amitié c'était la même, comme ne sont pas séparées des autres par un fossé les années nouvelles que notre désir, sans pouvoir les atteindre et les modifier, recouvre à leur insu d'un nom différent. J'avais beau dédier celle-ci à Gilberte, et comme on superpose une religion aux lois aveugles de la nature essayer d'imprimer au jour de l'an l'idée particulière que je m'étais faite de lui, c'était en vain ; je sentais qu'il ne savait pas qu'on l'appelât le jour de l'an, qu'il finissait dans le crépuscule d'une façon qui ne m'était pas nouvelle : dans le vent doux qui soufflait autour de la colonne d'affiches, j'avais reconnu, j'avais senti reparaître la matière éternelle et commune, l'humidité familière, l'ignorante fluidité des anciens jours.

### Passage

Je revins à la maison. Je venais de vivre le 1er janvier des hommes vieux qui diffèrent ce jour-là des jeunes, non parce qu'on ne leur donne plus d'étrennes, mais parce qu'ils ne croient plus au nouvel an. Des étrennes j'en avais reçu, mais non pas les seules qui m'eussent fait plaisir, et qui eussent été un mot de Gilberte. J'étais pourtant jeune encore tout de même puisque j'avais pu lui en écrire un par lequel j'espérais, en lui disant les rêves lointains de ma tendresse, en éveiller de pareils en elle. La tristesse des hommes qui ont vieilli c'est de ne pas même songer à écrire de telles lettres dont ils ont appris l'inefficacité.

Quand je fus couché, les bruits de la rue, qui se prolongeaient plus tard ce soir de fête, me tinrent éveillé. Je pensais à tous les gens qui finiraient leur nuit dans les plaisirs, à l'amant, à la troupe de débauchés peut-être, qui avaient dû aller chercher la Berma à la fin de cette représentation que j'avais vue annoncée pour le soir. Je ne pouvais même pas, pour calmer l'agitation que cette idée faisait naître en moi dans cette nuit d'insomnie, me dire que la Berma ne pensait peut-être pas à l'amour, puisque les vers qu'elle récitait, qu'elle avait longuement étudiés, lui rappelaient à tous moments qu'il est délicieux, comme elle le savait d'ailleurs si bien qu'elle en faisait apparaître les troubles bien connus – mais doués d'une violence nouvelle et d'une douceur insoupçonnée – à des spectateurs émerveillés dont chacun pourtant les avait ressentis par soi-même. Je rallumai ma bougie éteinte pour regarder encore une fois son visage. À la pensée qu'il était sans doute en ce moment caressé par ces hommes que je ne pouvais empêcher de donner à la Berma, et de recevoir d'elle, des joies surhumaines et vagues, j'éprouvais un émoi plus cruel qu'il n'était voluptueux, une nostalgie que vint aggraver le son du cor, comme on l'entend la nuit de la Mi-Carême, et souvent des autres fêtes, et qui, parce qu'il est alors sans poésie, est plus triste, sortant d'un mastroquet, que « le soir au fond des bois ». À ce moment-là, un mot de Gilberte n'eût peut-être pas été ce qu'il m'eût fallu. Nos désirs vont s'interférant, et dans la confusion de l'existence, il est rare qu'un bonheur vienne justement se poser sur le désir qui l'avait réclamé.

Je continuai à aller aux Champs-Élysées les jours de beau temps, par des rues dont les maisons élégantes et roses baignaient, parce que c'était le moment de la grande vogue des Expositions d'Aquarellistes, dans un ciel mobile et léger. Je mentirais en disant que dans ce temps-là les palais de Gabriel m'aient paru d'une plus grande beauté ni même d'une autre époque que les hôtels avoisinants. Je trouvais plus de style et aurais cru plus d'ancienneté sinon au Palais de l'Industrie, du moins à celui du Trocadéro. Plongée dans un sommeil agité, mon adolescence enveloppait d'un même rêve tout le quartier où elle le promenait, et je n'avais jamais songé qu'il pût y avoir un édifice du XVIIIe siècle dans la rue Royale, de même que j'aurais été étonné si j'avais appris que la Porte Saint-Martin et la Porte Saint-Denis, chefs-d'oeuvre du temps de Louis XIV, n'étaient pas contemporains des immeubles les plus récents de ces arrondissements sordides. Une seule fois un des palais de Gabriel me fit arrêter longuement ; c'est que, la nuit étant venue, ses colonnes dématérialisées par le clair de lune avaient l'air découpées dans du carton et, me rappelant un décor de l'opérette Orphée aux Enfers, me donnaient pour la première fois une impression de beauté.

Gilberte cependant ne revenait toujours pas aux Champs-Élysées. Et pourtant j'aurais eu besoin de la voir, car je ne me rappelais même pas sa figure. La manière chercheuse, anxieuse, exigeante que nous avons de regarder la personne que nous aimons, notre attente de la parole qui nous donnera ou nous ôtera l'espoir d'un rendez-vous pour le lendemain, et, jusqu'à ce que cette parole soit dite, notre imagination alternative, sinon simultanée, de la joie et du désespoir, tout cela rend notre attention en face de l'être aimé trop tremblante pour qu'elle puisse obtenir de lui une image bien nette.

Peut-être aussi cette activité de tous les sens à la fois, et qui essaye de connaître avec les regards seuls ce qui est au delà d'eux, est-elle trop indulgente aux mille formes, à toutes les saveurs, aux mouvements de la personne vivante que d'habitude, quand nous n'aimons pas, nous immobilisons. Le modèle chéri, au contraire, bouge ; on n'en a jamais que des photographies manquées. Je ne savais vraiment plus comment étaient faits les traits de Gilberte, sauf dans les moments divins où elle les dépliait pour moi : je ne me rappelais que son sourire. Et ne pouvant revoir ce visage bien-aimé, quelque effort que je fisse pour m'en souvenir, je m'irritais de trouver, dessinés dans ma mémoire avec une exactitude définitive, les visages inutiles et frappants de l'homme des chevaux de bois et de la marchande de sucre d'orge : ainsi ceux qui ont perdu un être aimé qu'ils ne revoient jamais en dormant s'exaspèrent de rencontrer sans cesse dans leurs rêves tant de gens insupportables et que c'est déjà trop d'avoir connus dans l'état de veille. Dans leur impuissance à se représenter l'objet de leur douleur, ils s'accusent presque de n'avoir pas de douleur. Et moi je n'étais pas loin de croire que, ne pouvant me rappeler les traits de Gilberte, je l'avais oubliée elle-même, je ne l'aimais plus. Enfin elle revint jouer presque tous les jours, mettant devant moi de nouvelles choses à désirer, à lui demander, pour le lendemain, faisant bien chaque jour, en ce sens-là, de ma tendresse une tendresse nouvelle. Mais une chose changea une fois de plus et brusquement la façon dont tous les après-midis vers deux heures se posait le problème de mon amour. Swann avait-il surpris la lettre que j'avais écrite à sa fille, ou Gilberte ne faisait-elle que m'avouer longtemps après, et afin que je fusse plus prudent, un état de choses déjà ancien ? Comme je lui disais combien j'admirais son père et sa mère, elle prit cet air vague, plein de réticences et de secret qu'elle avait quand on lui parlait de ce qu'elle avait à faire, de ses courses et de ses visites, et tout d'un coup finit par me dire : « Vous savez, ils ne vous gobent pas ! » et glissante comme une ondine – elle était ainsi – elle éclata de rire. Souvent son rire en désaccord avec ses paroles semblait, comme la musique, décrire dans un autre plan une surface invisible. M. et Odette ne demandaient pas à Gilberte de cesser de jouer avec moi, mais eussent autant aimé, pensait-elle, que cela n'eût pas commencé. Ils ne voyaient pas mes relations avec elle d'un oeil favorable, ne me croyaient pas d'une grande moralité et s'imaginaient que je ne pouvais exercer sur leur fille qu'une mauvaise influence. Ce genre de jeunes gens peu scrupuleux auxquels Swann me croyait ressembler, je me les représentais comme détestant les parents de la jeune fille qu'ils aiment, les flattant quand ils sont là, mais se moquant d'eux avec elle, la poussant à leur désobéir, et quand ils ont une fois conquis leur fille, les privant même de la voir. À ces traits (qui ne sont jamais ceux sous lesquels le plus grand misérable se voit lui-même), avec quelle violence mon coeur opposait ces sentiments dont il était animé à l'égard de Swann, si passionnés au contraire que je ne doutais pas que s'il les eût soupçonnés il ne se fût repenti de son jugement à mon égard comme d'une erreur judiciaire. Tout ce que je ressentais pour lui, j'osai le lui écrire dans une longue lettre que je confiai à Gilberte en la priant de la lui remettre. Elle y consentit. Hélas ! il voyait donc en moi un plus grand imposteur encore que je ne pensais ; ces sentiments que j'avais cru peindre, en seize pages, avec tant de vérité, il en avait donc douté ! La lettre que je lui écrivis, aussi ardente et aussi sincère que les paroles que j'avais dites à Norpois, n'eut pas plus de succès. Gilberte me raconta le lendemain, après m'avoir emmené à l'écart derrière un massif de lauriers, dans une petite allée où nous nous assîmes chacun sur une chaise, qu'en lisant la lettre, qu'elle me rapportait, son père avait haussé les épaules en disant : « Tout cela ne signifie rien, cela ne fait que prouver combien j'ai raison. » Moi qui savais la pureté de mes intentions, la bonté de mon âme, j'étais indigné que mes paroles n'eussent même pas effleuré l'absurde erreur de Swann. Car que ce fut une erreur, je n'en doutais pas alors. Je sentais que j'avais décrit avec tant d'exactitude certaines caractéristiques irrécusables de mes sentiments généreux que, pour que d'après elles Swann ne les eût pas aussitôt reconstitués, ne fût pas venu me demander pardon et avouer qu'il s'était trompé, il fallait que ces nobles sentiments, il ne les eût lui-même jamais ressentis, ce qui devait le rendre incapable de les comprendre chez les autres.
