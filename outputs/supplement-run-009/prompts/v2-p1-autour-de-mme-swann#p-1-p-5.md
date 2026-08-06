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
        "fils Swann",
        "Swann du Jockey"
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
      "evidence": "Swann, « avec son ostentation… un vulgaire esbrouffeur »; puis « il s'y montrait un autre homme… faire sonner bien haut que la femme d'un sous-chef de cabinet était venue rendre sa visite à Odette ».",
      "explanation": "The narrator validates the father's harsh view by showing Swann's altered behavior in his 'second life' with Odette: he boasts of modest connections and displays ostentation, contrasting with his earlier discreet elegance."
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
      "confidence": 0.86,
      "explanation": "Locally judged as ostentatious and vulgar compared to his former discreet self, lowering his immediate appraisal."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-1-p-5"
}

### Candidate characters

[
  "Gilberte",
  "M. Verdurin",
  "Mme Cottard",
  "Mme Verdurin",
  "Norpois",
  "Odette",
  "docteur Cottard",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur",
  "prince de Léon"
]

### Prior local context (optional)

(none provided)

### Passage

Ma mère, quand il fut question d'avoir pour la première fois Norpois à dîner, ayant exprimé le regret que le professeur Cottard fût en voyage et qu'elle-même eût entièrement cessé de fréquenter Swann, car l'un et l'autre eussent sans doute intéressé l'ancien ambassadeur, mon père répondit qu'un convive éminent, un savant illustre, comme Cottard, ne pouvait jamais mal faire dans un dîner, mais que Swann, avec son ostentation, avec sa manière de crier sur les toits ses moindres relations, était un vulgaire esbrouffeur que le marquis de Norpois eût sans doute trouvé, selon son expression, « puant ». Or cette réponse de mon père demande quelques mots d'explication, certaines personnes se souvenant peut-être d'un Cottard bien médiocre et d'un Swann poussant jusqu'à la plus extrême délicatesse, en matière mondaine, la modestie et la discrétion. Mais pour ce qui regarde celui-ci, il était arrivé qu'au « fils Swann » et aussi au Swann du Jockey, l'ancien ami de mes parents avait ajouté une personnalité nouvelle (et qui ne devait pas être la dernière), celle de mari d'Odette. Adaptant aux humbles ambitions de cette femme, l'instinct, le désir, l'industrie, qu'il avait toujours eus, il s'était ingénié à se bâtir, fort au-dessous de l'ancienne, une position nouvelle et appropriée à la compagne qui l'occuperait avec lui. Or il s'y montrait un autre homme. Puisque (tout en continuant à fréquenter seul ses amis personnels, à qui il ne voulait pas imposer Odette quand ils ne lui demandaient pas spontanément à la connaître) c'était une seconde vie qu'il commençait, en commun avec sa femme, au milieu d'êtres nouveaux, on eût encore compris que pour mesurer le rang de ceux-ci, et par conséquent le plaisir d'amour-propre qu'il pouvait éprouver à les recevoir, il se fût servi, comme un point de comparaison, non pas des gens les plus brillants qui formaient sa société avant son mariage, mais des relations antérieures d'Odette. Mais, même quand on savait que c'était avec d'inélégants fonctionnaires, avec des femmes tarées, parure des bals de ministères, qu'il désirait de se lier, on était étonné de l'entendre, lui qui autrefois et même encore aujourd'hui dissimulait si gracieusement une invitation de Twickenham ou de Buckingham Palace, faire sonner bien haut que la femme d'un sous-chef de cabinet était venue rendre sa visite à Odette. On dira peut-être que cela tenait à ce que la simplicité du Swann élégant n'avait été chez lui qu'une forme plus raffinée de la vanité et que, comme certains israélites, l'ancien ami de mes parents avait pu présenter tour à tour les états successifs par où avaient passé ceux de sa race, depuis le snobisme le plus naïf et la plus grossière goujaterie, jusqu'à la plus fine politesse. Mais la principale raison, et celle-là applicable à l'humanité en général, était que nos vertus elles-mêmes ne sont pas quelque chose de libre, de flottant, de quoi nous gardions la disponibilité permanente ; elles finissent par s'associer si étroitement dans notre esprit avec les actions à l'occasion desquelles nous nous sommes fait un devoir de les exercer, que si surgit pour nous une activité d'un autre ordre, elle nous prend au dépourvu et sans que nous ayons seulement l'idée qu'elle pourrait comporter la mise en oeuvre de ces mêmes vertus. Swann empressé avec ces nouvelles relations et les citant avec fierté, était comme ces grands artistes modestes ou généreux qui, s'ils se mettent à la fin de leur vie à se mêler de cuisine ou de jardinage, étalent une satisfaction naïve des louanges qu'on donne à leurs plats ou à leurs plates-bandes pour lesquels ils n'admettent pas la critique qu'ils acceptent aisément s'il s'agit de leurs chefs-d'oeuvre ; ou bien qui, donnant une de leurs toiles pour rien, ne peuvent en revanche sans mauvaise humeur perdre quarante sous aux dominos.

Quant au professeur Cottard, on le reverra, longuement, beaucoup plus loin, chez la Patronne, au château de la Raspelière. Qu'il suffise actuellement, à son égard, de faire observer ceci : pour Swann, à la rigueur le changement peut surprendre puisqu'il était accompli et non soupçonné de moi quand je voyais le père de Gilberte aux Champs-Élysées, où d'ailleurs ne m'adressant pas la parole il ne pouvait faire étalage devant moi de ses relations politiques (il est vrai que s'il l'eût fait, je ne me fusse peut-être pas aperçu tout de suite de sa vanité car l'idée qu'on s'est faite longtemps d'une personne bouche les yeux et les oreilles ; ma mère pendant trois ans ne distingua pas plus le fard qu'une de ses nièces se mettait aux lèvres que s'il eût été invisiblement dissous entièrement dans un liquide ; jusqu'au jour où une parcelle supplémentaire, ou bien quelque autre cause amena le phénomène appelé sursaturation ; tout le fard non aperçu cristallisa, et ma mère, devant cette débauche soudaine de couleurs déclara, comme on eût fait à Combray, que c'était une honte, et cessa presque toute relation avec sa nièce). Mais pour Cottard au contraire, l'époque où on l'a vu assister aux débuts de Swann chez les Verdurin était déjà assez lointaine ; or les honneurs, les titres officiels viennent avec les années ; deuxièmement, on peut être illettré, faire des calembours stupides, et posséder un don particulier qu'aucune culture générale ne remplace, comme le don du grand stratège ou du grand clinicien. Ce n'est pas seulement en effet comme un praticien obscur, devenu, à la longue, notoriété européenne, que ses confrères considéraient Cottard. Les plus intelligents d'entre les jeunes médecins déclarèrent – au moins pendant quelques années, car les modes changent étant nées elles-mêmes du besoin de changement – que si jamais ils tombaient malades, Cottard était le seul maître auquel ils confieraient leur peau. Sans doute ils préféraient le commerce de certains chefs plus lettrés, plus artistes, avec lesquels ils pouvaient parler de Nietzsche, de Wagner. Quand on faisait de la musique chez Mme Cottard, aux soirées où elle recevait, avec l'espoir qu'il devînt un jour doyen de la Faculté, les collègues et les élèves de son mari, celui-ci, au lieu d'écouter, préférait jouer aux cartes dans un salon voisin. Mais on vantait la promptitude, la profondeur, la sûreté de son coup d'oeil, de son diagnostic. En troisième lieu, en ce qui concerne l'ensemble de façons que le professeur Cottard montrait à un homme comme mon père, remarquons que la nature que nous faisons paraître dans la seconde partie de notre vie n'est pas toujours, si elle l'est souvent, notre nature première développée ou flétrie, grossie ou atténuée ; elle est quelquefois une nature inverse, un véritable vêtement retourné. Sauf chez les Verdurin qui s'étaient engoués de lui, l'air hésitant de Cottard, sa timidité, son amabilité excessives, lui avaient, dans sa jeunesse, valu de perpétuels brocards. Quel ami charitable lui conseilla l'air glacial ? L'importance de sa situation lui rendit plus aisé de le prendre. Partout, sinon chez les Verdurin où il redevenait instinctivement lui-même, il se rendit froid, volontiers silencieux, péremptoire quand il fallait parler, n'oubliant pas de dire des choses désagréables. Il put faire l'essai de cette nouvelle attitude devant des clients qui, ne l'ayant pas encore vu, n'étaient pas à même de faire des comparaisons, et eussent été bien étonnés d'apprendre qu'il n'était pas un homme d'une rudesse naturelle. C'est surtout à l'impassibilité qu'il s'efforçait, et même dans son service d'hôpital, quand il débitait quelques-uns de ces calembours qui faisaient rire tout le monde, du chef de clinique au plus récent externe, il le faisait toujours sans qu'un muscle bougeât dans sa figure d'ailleurs méconnaissable depuis qu'il avait rasé barbe et moustaches.

Disons pour finir qui était le marquis de Norpois. Il avait été ministre plénipotentiaire avant la guerre et ambassadeur au Seize Mai, et, malgré cela, au grand étonnement de beaucoup, chargé plusieurs fois, depuis, de représenter la France dans des missions extraordinaires – et même comme contrôleur de la Dette, en Égypte, où grâce à ses grandes capacités financières il avait rendu d'importants services – par des cabinets radicaux qu'un simple bourgeois réactionnaire se fût refusé à servir, et auxquels le passé de Norpois, ses attaches, ses opinions eussent dû le rendre suspect. Mais ces ministres avancés semblaient se rendre compte qu'ils montraient par une telle désignation quelle largeur d'esprit était la leur dès qu'il s'agissait des intérêts supérieurs de la France, se mettaient hors de pair des hommes politiques en méritant que le Journal des Débats lui-même les qualifiât d'hommes d'État, et bénéficiaient enfin du prestige qui s'attache à un nom aristocratique et de l'intérêt qu'éveille comme un coup de théâtre un choix inattendu. Et ils savaient aussi que ces avantages ils pouvaient, en faisant appel à Norpois, les recueillir sans avoir à craindre de celui-ci un manque de loyalisme politique contre lequel la naissance du marquis devait non pas les mettre en garde, mais les garantir. Et en cela le gouvernement de la République ne se trompait pas. C'est d'abord parce qu'une certaine aristocratie, élevée dès l'enfance à considérer son nom comme un avantage intérieur que rien ne peut lui enlever (et dont ses pairs, ou ceux qui sont de naissance plus haute encore, connaissent assez exactement la valeur), sait qu'elle peut s'éviter, car ils ne lui ajouteraient rien, les efforts que sans résultat ultérieur appréciable font tant de bourgeois pour ne professer que des opinions bien portées et ne fréquenter que des gens bien pensants. En revanche, soucieuse de se grandir aux yeux des familles princières ou ducales au-dessous desquelles elle est immédiatement située, cette aristocratie sait qu'elle ne le peut qu'en augmentant son nom de ce qu'il ne contenait pas, de ce qui fait qu'à nom égal, elle prévaudra : une influence politique, une réputation littéraire ou artistique, une grande fortune. Et les frais dont elle se dispense à l'égard de l'inutile hobereau recherché des bourgeois et de la stérile amitié duquel un prince ne lui saurait aucun gré, elle les prodiguera aux hommes politiques, fussent-ils francs-maçons, qui peuvent faire arriver dans les ambassades ou patronner dans les élections, aux artistes ou aux savants dont l'appui aide à « percer » dans la branche où ils priment, à tous ceux enfin qui sont en mesure de conférer une illustration nouvelle ou de faire réussir un riche mariage.

Mais en ce qui concernait Norpois, il y avait surtout que, dans une longue pratique de la diplomatie, il s'était imbu de cet esprit négatif, routinier, conservateur, dit « esprit de gouvernement » et qui est, en effet, celui de tous les gouvernements et, en particulier, sous tous les gouvernements, l'esprit des chancelleries. Il avait puisé dans la carrière l'aversion, la crainte et le mépris de ces procédés plus ou moins révolutionnaires, et à tout le moins incorrects, que sont les procédés des oppositions. Sauf chez quelques illettrés du peuple et du monde, pour qui la différence des genres est lettre morte, ce qui rapproche, ce n'est pas la communauté des opinions, c'est la consanguinité des esprits. Un académicien du genre de Legouvé et qui serait partisan des classiques, eût applaudi plus volontiers à l'éloge de Victor Hugo par Maxime Ducamp ou Mézières, qu'à celui de Boileau par Claudel. Un même nationalisme suffit à rapprocher Barrès de ses électeurs qui ne doivent pas faire grande différence entre lui et M. Georges Berry, mais non de ceux de ses collègues de l'Académie qui, ayant ses opinions politiques mais un autre genre d'esprit, lui préfèreront même des adversaires comme MM. Ribot et Deschanel, dont à leur tour de fidèles monarchistes se sentent beaucoup plus près que de Maurras et de Léon Daudet qui souhaitent cependant aussi le retour du Roi. Avare de ses mots non seulement par pli professionnel de prudence et de réserve, mais aussi parce qu'ils ont plus de prix, offrent plus de nuances aux yeux d'hommes dont les efforts de dix années pour rapprocher deux pays se résument, se traduisent – dans un discours, dans un protocole – par un simple adjectif, banal en apparence, mais où ils voient tout un monde. Norpois passait pour très froid, à la Commission, où il siégeait à côté de mon père, et où chacun félicitait celui-ci de l'amitié que lui témoignait l'ancien ambassadeur. Elle étonnait mon père tout le premier. Car étant généralement peu aimable, il avait l'habitude de n'être pas recherché en dehors du cercle de ses intimes et l'avouait avec simplicité. Il avait conscience qu'il y avait dans les avances du diplomate un effet de ce point de vue tout individuel où chacun se place pour décider de ses sympathies, et d'où toutes les qualités intellectuelles ou la sensibilité d'une personne ne seront pas auprès de l'un de nous qu'elle ennuie ou agace une aussi bonne recommandation que la rondeur et la gaieté d'une autre qui passerait, aux yeux de beaucoup, pour vide, frivole et nulle. « De Norpois m'a invité de nouveau à dîner ; c'est extraordinaire ; tout le monde en est stupéfait à la Commission où il n'a de relations privées avec personne. Je suis sûr qu'il va encore me raconter des choses palpitantes sur la guerre de 70. » Mon père savait que seul, peut-être, Norpois avait averti l'Empereur de la puissance grandissante et des intentions belliqueuses de la Prusse, et que Bismarck avait pour son intelligence une estime particulière. Dernièrement encore à l'Opéra, pendant le gala offert au roi Théodose, les journaux avaient remarqué l'entretien prolongé que le souverain avait accordé à Norpois. « Il faudra que je sache si cette visite du roi a vraiment de l'importance, nous dit mon père qui s'intéressait beaucoup à la politique étrangère. Je sais bien que le père Norpois est très boutonné, mais avec moi, il s'ouvre si gentiment. »

Quant à ma mère, peut-être l'Ambassadeur n'avait-il pas par lui-même le genre d'intelligence vers lequel elle se sentait le plus attirée. Et je dois dire que la conversation de Norpois était un répertoire si complet des formes surannées du langage particulières à une carrière, à une classe, et à un temps – un temps qui, pour cette carrière et cette classe-là, pourrait bien ne pas être tout à fait aboli – que je regrette parfois de n'avoir pas retenu purement et simplement les propos que je lui ai entendu tenir. J'aurais ainsi obtenu un effet de démodé, à aussi bon compte et de la même façon que cet acteur du Palais-Royal à qui on demandait où il pouvait trouver ses surprenants chapeaux et qui répondait : « Je ne trouve pas mes chapeaux. Je les garde. » En un mot, je crois que ma mère jugeait Norpois un peu « vieux jeu », ce qui était loin de lui sembler déplaisant au point de vue des manières, mais la charmait moins dans le domaine, sinon des idées – car celles de Norpois étaient fort modernes – mais des expressions. Seulement, elle sentait que c'était flatter délicatement son mari que de lui parler avec admiration du diplomate qui lui marquait une prédilection si rare. En fortifiant dans l'esprit de mon père la bonne opinion qu'il avait de Norpois, et par là en le conduisant à en prendre une bonne aussi de lui-même, elle avait conscience de remplir celui de ses devoirs qui consistait à rendre la vie agréable à son époux, comme elle faisait quand elle veillait à ce que la cuisine fût soignée et le service silencieux. Et comme elle était incapable de mentir à mon père, elle s'entraînait elle-même à admirer l'Ambassadeur pour pouvoir le louer avec sincérité. D'ailleurs, elle goûtait naturellement son air de bonté, sa politesse un peu désuète (et si cérémonieuse que quand, marchant en redressant sa haute taille, il apercevait ma mère qui passait en voiture, avant de lui envoyer un coup de chapeau, il jetait au loin un cigare à peine commencé) ; sa conversation si mesurée, où il parlait de lui-même le moins possible et tenait toujours compte de ce qui pouvait être agréable à l'interlocuteur, sa ponctualité tellement surprenante à répondre à une lettre que quand, venant de lui en envoyer une, mon père reconnaissait l'écriture de Norpois sur une enveloppe, son premier mouvement était de croire que par mauvaise chance leur correspondance s'était croisée : on eût dit qu'il existait, pour lui, à la poste, des levées supplémentaires et de luxe. Ma mère s'émerveillait qu'il fut si exact quoique si occupé, si aimable quoique si répandu, sans songer que les « quoique » sont toujours des « parce que » méconnus, et que (de même que les vieillards sont étonnants pour leur âge, les rois pleins de simplicité, et les provinciaux au courant de tout) c'était les mêmes habitudes qui permettaient à Norpois de satisfaire à tant d'occupations et d'être si ordonné dans ses réponses, de plaire dans le monde et d'être aimable avec nous. De plus, l'erreur de ma mère comme celle de toutes les personnes qui ont trop de modestie, venait de ce qu'elle mettait les choses qui la concernaient au-dessous, et par conséquent en dehors des autres. La réponse qu'elle trouvait que l'ami de mon père avait eu tant de mérite à nous adresser rapidement parce qu'il écrivait par jour beaucoup de lettres, elle l'exceptait de ce grand nombre de lettres dont ce n'était que l'une ; de même elle ne considérait pas qu'un dîner chez nous fût pour Norpois un des actes innombrables de sa vie sociale : elle ne songeait pas que l'Ambassadeur avait été habitué autrefois dans la diplomatie à considérer les dîners en ville comme faisant partie de ses fonctions, et à y déployer une grâce invétérée dont c'eût été trop lui demander de se départir par extraordinaire quand il venait dîner chez nous.
