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
      "canonical_name": "marquis de Cambremer",
      "surface_forms": [
        "le grand seigneur de la contrée",
        "le beau-frère de Legrandin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.86
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "marquis de Cambremer",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.85,
      "evidence": "« le grand seigneur de la contrée, lequel n'était autre que le beau-frère de Legrandin »; « par la garden-party hebdomadaire que sa femme et lui donnaient, dépeuplait l'hôtel... parce qu'un ou deux d'entre eux étaient invités... et parce que les autres... choisissaient ce jour-là pour faire une excursion ».",
      "explanation": "The text presents Legrandin's brother-in-law as a local authority whose invitations regulate hotel life, marking his effective social superiority despite an initial anecdote of ignorance by the staff."
    }
  ],
  "status_effects": [
    {
      "character": "marquis de Cambremer",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "He is placed at the center of the local hierarchy: his garden parties structure inclusion/exclusion and empty the hotel."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-56-p-60"
}

### Candidate characters

[
  "Legrandin",
  "M. de Stermaria",
  "Mlle de Stermaria",
  "le directeur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Dès lors, ayant placé entre elle d'une part, le personnel de l'hôtel et les fournisseurs de l'autre, ses domestiques qui recevaient à sa place le contact de cette humanité nouvelle et entretenaient autour de leur maîtresse l'atmosphère accoutumée, ayant mis ses préjugés entre elle et les baigneurs, insoucieuse de déplaire à des gens que ses amies n'auraient pas reçus, c'est dans son monde qu'elle continuait à vivre par la correspondance avec ses amies, par le souvenir, par la conscience intime qu'elle avait de sa situation, de la qualité de ses manières, de la compétence de sa politesse. Et tous les jours, quand elle descendait pour aller dans sa calèche faire une promenade, sa femme de chambre qui portait ses affaires derrière elle, son valet de pied qui la devançait semblaient comme ces sentinelles qui, aux portes d'une ambassade pavoisée aux couleurs du pays dont elle dépend, garantissent pour elle, au milieu d'un sol étranger, le privilège de son exterritorialité. Elle ne quitta pas sa chambre avant le milieu de l'après-midi, le jour de notre arrivée, et nous ne l'aperçûmes pas dans la salle à manger où le directeur, comme nous étions nouveaux venus, nous conduisit, sous sa protection, à l'heure du déjeuner, comme un gradé qui mène des bleus chez le caporal tailleur pour les faire habiller ; mais nous y vîmes, en revanche, au bout d'un instant un hobereau et sa fille, d'une obscure mais très ancienne famille de Bretagne, M. et Mlle de Stermaria, dont on nous avait fait donner la table, croyant qu'ils ne rentreraient que le soir. Venus seulement à Balbec pour retrouver des châtelains qu'ils connaissaient dans le voisinage, ils ne passaient dans la salle à manger de l'hôtel, entre les invitations acceptées au dehors et les visites rendues que le temps strictement nécessaire. C'était leur morgue qui les préservait de toute sympathie humaine, de tout intérêt pour les inconnus assis autour d'eux, et au milieu desquels M. de Stermaria gardait l'air glacial, pressé, distant, rude, pointilleux et malintentionné, qu'on a dans un buffet de chemin de fer au milieu de voyageurs qu'on n'a jamais vus, qu'on ne reverra pas, et avec qui on ne conçoit d'autres rapports que de défendre contre eux son poulet froid et son coin dans le wagon. À peine commencions-nous à déjeuner qu'on vint nous faire lever sur l'ordre de M. de Stermaria, lequel venait d'arriver et, sans le moindre geste d'excuse à notre adresse, pria à haute voix le maître d'hôtel de veiller à ce qu'une pareille erreur ne se renouvelât pas, car il lui était désagréable que « des gens qu'il ne connaissait pas » eussent pris sa table.

### Passage

Et certes dans le sentiment qui poussait une certaine actrice (plus connue d'ailleurs à cause de son élégance, de son esprit, de ses belles collections de porcelaine allemande que pour quelques rôles joués à l'Odéon), son amant, jeune homme très riche pour lequel elle s'était cultivée, et deux hommes très en vue de l'aristocratie, à faire dans la vie bande à part, à ne voyager qu'ensemble, à prendre à Balbec leur déjeuner, très tard, quand tout le monde avait fini, à passer la journée dans leur salon à jouer aux cartes, il n'entrait aucune malveillance, mais seulement les exigences du goût qu'ils avaient pour certaines formes spirituelles de conversation, pour certains raffinements de bonne chère, lequel leur faisait trouver plaisir à ne vivre, à ne prendre leurs repas qu'ensemble, et leur eût rendu insupportable la vie en commun avec des gens qui n'y avaient pas été initiés. Même devant une table servie, ou devant une table à jeu, chacun d'eux avait besoin de savoir que dans le convive ou le partenaire qui était assis en face de lui, reposaient en suspens et inutilisés un certain savoir qui permet de reconnaître la camelote dont tant de demeures parisiennes se parent comme d'un « moyen âge » ou d'une « Renaissance » authentiques et, en toutes choses, des critériums communs à eux pour distinguer le bon et le mauvais. Sans doute ce n'était plus, dans ces moments-là, que par quelque rare et drôle interjection jetée au milieu du silence du repas ou de la partie, ou par la robe charmante et nouvelle que la jeune actrice avait revêtue pour déjeuner ou faire un poker, que se manifestait l'existence spéciale dans laquelle ces amis voulaient partout rester plongés. Mais en les enveloppant ainsi d'habitudes qu'ils connaissaient à fond, elle suffisait à les protéger contre le mystère de la vie ambiante. Pendant de longs après-midi, la mer n'était suspendue en face d'eux que comme une toile d'une couleur agréable accrochée dans le boudoir d'un riche célibataire, et ce n'était que dans l'intervalle des coups qu'un des joueurs, n'ayant rien de mieux à faire, levait les yeux vers elle pour en tirer une indication sur le beau temps ou sur l'heure, et rappeler aux autres que le goûter attendait. Et le soir ils ne dînaient pas à l'hôtel où les sources électriques faisant sourdre à flots la lumière dans la grande salle à manger, celle-ci devenait comme un immense et merveilleux aquarium devant la paroi de verre duquel la population ouvrière de Balbec, les pêcheurs et aussi les familles de petits bourgeois, invisibles dans l'ombre, s'écrasaient au vitrage pour apercevoir, lentement balancée dans des remous d'or, la vie luxueuse de ces gens, aussi extraordinaire pour les pauvres que celle de poissons et de mollusques étranges (une grande question sociale, de savoir si la paroi de verre protègera toujours le festin des bêtes merveilleuses et si les gens obscurs qui regardent avidement dans la nuit ne viendront pas les cueillir dans leur aquarium et les manger). En attendant, peut-être parmi la foule arrêtée et confondue dans la nuit, y avait-il quelque écrivain, quelque amateur d'ichtyologie humaine, qui, regardant les mâchoires de vieux monstres féminins se refermer sur un morceau de nourriture engloutie, se complaisait à classer ceux-ci par race, par caractères innés et aussi par ces caractères acquis qui font qu'une vieille dame serbe dont l'appendice buccal est d'un grand poisson de mer, parce que depuis son enfance elle vit dans les eaux douces du faubourg Saint-Germain, mange la salade comme une La Rochefoucauld.

À cette heure-là on apercevait les trois hommes en smoking attendant la femme en retard, laquelle bientôt, en une robe presque chaque fois nouvelle et des écharpes choisies selon un goût particulier à son amant, après avoir, de son étage, sonné le lift, sortait de l'ascenseur comme d'une boîte de joujoux. Et tous les quatre qui trouvaient que le phénomène international du Palace, implanté à Balbec, y avait fait fleurir le luxe plus que la bonne cuisine, s'engouffraient dans une voiture, allaient dîner à une demi-lieue de là dans un petit restaurant réputé où ils avaient avec le cuisinier d'interminables conférences sur la composition du menu et la confection des plats. Pendant ce trajet la route bordée de pommiers qui part de Balbec n'était pour eux que la distance qu'il fallait franchir – peu distincte dans la nuit noire de celle qui séparait leurs domiciles parisiens du Café Anglais ou de la Tour d'Argent – avant d'arriver au petit restaurant élégant où, tandis que les amis du jeune homme riche l'enviaient d'avoir une maîtresse si bien habillée, les écharpes de celle-ci tendaient devant la petite société comme un voile parfumé et souple, mais qui la séparait du monde.

Malheureusement pour ma tranquillité, j'étais bien loin d'être comme tous ces gens. De beaucoup d'entre eux je me souciais ; j'aurais voulu ne pas être ignoré d'un homme au front déprimé, au regard fuyant entre les oeillères de ses préjugés et de son éducation, le grand seigneur de la contrée, lequel n'était autre que le beau-frère de Legrandin, qui venait quelquefois en visite à Balbec et, le dimanche, par la garden-party hebdomadaire que sa femme et lui donnaient, dépeuplait l'hôtel d'une partie de ses habitants parce qu'un ou deux d'entre eux étaient invités à ces fêtes, et parce que les autres, pour ne pas avoir l'air de ne pas l'être, choisissaient ce jour-là pour faire une excursion éloignée. Il avait, d'ailleurs, été le premier jour fort mal reçu à l'hôtel quand le personnel, frais débarqué de la Côte d'Azur, ne savait pas encore qui il était. Non seulement il n'était pas habillé en flanelle blanche, mais par vieille manière française et ignorance de la vie des Palaces, entrant dans un hall où il y avait des femmes, il avait ôté son chapeau dès la porte, ce qui avait fait que le directeur n'avait même pas touché le sien pour lui répondre, estimant que ce devait être quelqu'un de la plus humble extraction, ce qu'il appelait un homme « sortant de l'ordinaire ». Seule la femme du notaire s'était sentie attirée vers le nouveau venu qui fleurait toute la vulgarité gourmée des gens comme il faut, et elle avait déclaré, avec le fond de discernement infaillible et d'autorité sans réplique d'une personne pour qui la première société du Mans n'a pas de secrets, qu'on se sentait devant lui en présence d'un homme d'une haute distinction, parfaitement bien élevé et qui tranchait sur tout ce qu'on rencontrait à Balbec et qu'elle jugeait infréquentable tant qu'elle ne le fréquentait pas. Ce jugement favorable qu'elle avait porté sur le beau-frère de Legrandin tenait peut-être au terne aspect de quelqu'un qui n'avait rien d'intimidant, peut-être à ce qu'elle avait reconnu dans ce gentilhomme-fermier à allure de sacristain les signes maçonniques de son propre cléricalisme.

J'avais beau avoir appris que les jeunes gens qui montaient tous les jours à cheval devant l'hôtel étaient les fils du propriétaire véreux d'un magasin de nouveautés et que mon père n'eût jamais consenti à connaître, la « vie de bains de mer » les dressait, à mes yeux, en statues équestres de demi-dieux et le mieux que je pouvais espérer était qu'ils laissassent jamais tomber leurs regards sur le pauvre garçon que j'étais, qui ne quittait la salle à manger de l'hôtel que pour aller s'asseoir sur le sable. J'aurais voulu inspirer de la sympathie à l'aventurier même qui avait été roi d'une île déserte en Océanie, même au jeune tuberculeux dont j'aimais à supposer qu'il cachait sous ses dehors insolents une âme craintive et tendre qui eût peut-être prodigué pour moi seul des trésors d'affection. D'ailleurs (au contraire de ce qu'on dit d'habitude des relations de voyage), comme être vu avec certaines personnes peut vous ajouter, sur une plage où l'on retourne quelquefois, un coefficient sans équivalent dans la vraie vie mondaine, il n'y a rien, non pas qu'on tienne aussi à distance, mais qu'on cultive si soigneusement dans la vie de Paris, que les amitiés de bains de mer. Je me souciais de l'opinion que pouvaient avoir de moi toutes ces notabilités momentanées ou locales que ma disposition à me mettre à la place des gens et à recréer leur état d'esprit me faisait situer non à leur rang réel, à celui qu'ils auraient occupé à Paris par exemple et qui eût été fort bas, mais à celui qu'ils devaient croire le leur, et qui l'était à vrai dire à Balbec où l'absence de commune mesure leur donnait une sorte de supériorité relative et d'intérêt singulier. Hélas, d'aucune de ces personnes le mépris ne m'était aussi pénible que celui de M. de Stermaria.

Car j'avais remarqué sa fille dès son entrée, son joli visage pâle et presque bleuté, ce qu'il y avait de particulier dans le port de sa haute taille, dans sa démarche, et qui m'évoquait avec raison son hérédité, son éducation aristocratique et d'autant plus clairement que je savais son nom – comme ces thèmes expressifs inventés par des musiciens de génie et qui peignent splendidement le scintillement de la flamme, le bruissement du fleuve et la paix de la campagne, pour les auditeurs qui, en parcourant préalablement le livret, ont aiguillé leur imagination dans la bonne voie. La « race », en ajoutant aux charmes de Mlle de Stermaria l'idée de leur cause, les rendait plus intelligibles, plus complets. Elle les faisait aussi plus désirables, annonçant qu'ils étaient peu accessibles, comme un prix élevé ajoute à la valeur d'un objet qui nous a plu. Et la tige héréditaire donnait à ce teint composé de sucs choisis la saveur d'un fruit exotique ou d'un cru célèbre.
