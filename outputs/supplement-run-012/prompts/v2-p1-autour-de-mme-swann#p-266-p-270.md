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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« La personne du reste qui était le plus complètement dupe de l’illusion … c’était Odette. » … « elle avait l’air de trouver que je faisais bien des embarras, qu’il y avait un peu de sottise et de prétention dans mes paroles »",
      "explanation": "The narrator presents Odette as the most deceived by the illusion that proximity to their salon could substitute for work, and shows her misreading his refusals to visit as pretentious. This locally lowers her discernment."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Odette is framed as gullible and judgmentally off-base, which diminishes her local esteem in the narrator’s account."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-266-p-270"
}

### Candidate characters

[
  "Bergotte",
  "Gilberte",
  "Swann",
  "la grand-mère",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Alors, ce n'est pas encore pour ce soir que je vous unis à « Rachel quand du Seigneur » ? Comment dites-vous cela : « Rachel quand du Seigneur ! » Ah ! ça c'est très bien trouvé. Je vais vous fiancer. Vous verrez que vous ne le regretterez pas.

### Passage

Une fois je faillis me décider, mais elle était « sous presse », une autre fois entre les mains du « coiffeur », un vieux monsieur qui ne faisait rien d'autre aux femmes que verser de l'huile sur leurs cheveux déroulés et les peigner ensuite. Et je me lassai d'attendre, bien que quelques habituées fort humbles, soi-disant ouvrières, mais toujours sans travail, fussent venues me faire de la tisane et tenir avec moi une longue conversation à laquelle – malgré le sérieux des sujets traités – la nudité partielle ou complète de mes interlocutrices donnait une savoureuse simplicité. Je cessai du reste d'aller dans cette maison parce que, désireux de témoigner mes bons sentiments à la femme qui la tenait et avait besoin de meubles, je lui en donnai quelques-uns – notamment un grand canapé – que j'avais hérités de ma tante Léonie. Je ne les voyais jamais, car le manque de place avait empêché mes parents de les laisser entrer chez nous et ils étaient entassés dans un hangar. Mais dès que je les retrouvai dans la maison où ces femmes se servaient d'eux, toutes les vertus qu'on respirait dans la chambre de ma tante à Combray m'apparurent, suppliciées par le contact cruel auquel je les avais livrés sans défense ! J'aurais fait violer une morte que je n'aurais pas souffert davantage. Je ne retournai plus chez l'entremetteuse, car ils me semblaient vivre et me supplier, comme ces objets en apparence inanimés d'un conte persan, dans lesquels sont enfermées des âmes qui subissent un martyre et implorent leur délivrance. D'ailleurs, comme notre mémoire ne nous présente pas d'habitude nos souvenirs dans leur suite chronologique, mais comme un reflet où l'ordre des parties est renversé, je me rappelai seulement beaucoup plus tard que c'était sur ce même canapé que bien des années auparavant j'avais connu pour la première fois les plaisirs de l'amour avec une de mes petites cousines avec qui je ne savais où me mettre, et qui m'avait donné le conseil dangereux de profiter d'une heure où ma tante Léonie était levée.

Toute une autre partie des meubles, et surtout une magnifique argenterie ancienne de ma tante Léonie, je les vendis, malgré l'avis contraire de mes parents, pour pouvoir disposer de plus d'argent et envoyer plus de fleurs à Odette qui me disait en recevant d'immenses corbeilles d'orchidées : « Si j'étais Monsieur votre père, je vous ferais donner un conseil judiciaire. » Comment pouvais-je supposer qu'un jour je pourrais regretter tout particulièrement cette argenterie et placer certains plaisirs plus haut que celui, qui deviendrait peut-être absolument nul, de faire des politesses aux parents de Gilberte. C'est de même en vue de Gilberte et pour ne pas la quitter que j'avais décidé de ne pas entrer dans les ambassades. Ce n'est jamais qu'à cause d'un état d'esprit qui n'est pas destiné à durer qu'on prend des résolutions définitives. J'imaginais à peine que cette substance étrange qui résidait en Gilberte et rayonnait en ses parents, en sa maison, me rendant indifférent à tout le reste, cette substance pourrait être libérée, émigrer dans un autre être. Vraiment la même substance, et pourtant devant avoir sur moi de tout autres effets. Car la même maladie évolue ; et un délicieux poison n'est plus toléré de même quand, avec les années, a diminué la résistance du coeur.

Mes parents cependant auraient souhaité que l'intelligence que Bergotte m'avait reconnue se manifestât par quelque travail remarquable. Quand je ne connaissais pas les Swann je croyais que j'étais empêché de travailler par l'état d'agitation où me mettait l'impossibilité de voir librement Gilberte. Mais quand leur demeure me fut ouverte, à peine je m'étais assis à mon bureau de travail que je me levais et courais chez eux. Et une fois que je les avais quittés et que j'étais rentré à la maison, mon isolement n'était qu'apparent, ma pensée ne pouvait plus remonter le courant du flux de paroles par lequel je m'étais laissé machinalement entraîner pendant des heures. Seul, je continuais à fabriquer les propos qui eussent été capables de plaire aux Swann, et pour donner plus d'intérêt au jeu, je tenais la place de ces partenaires absents, je me posais à moi-même des questions fictives choisies de telle façon que mes traits brillants ne leur servissent que d'heureuse répartie. Silencieux, cet exercice était pourtant une conversation et non une méditation, ma solitude une vie de salon mentale où c'était non ma propre personne mais des interlocuteurs imaginaires qui gouvernaient mes paroles et où j'éprouvais à former, au lieu des pensées que je croyais vraies, celles qui me venaient sans peine, sans régression du dehors vers le dedans, ce genre de plaisir tout passif que trouve à rester tranquille quelqu'un qui est alourdi par une mauvaise digestion.

Si j'avais été moins décidé à me mettre définitivement au travail, j'aurais peut-être fait un effort pour commencer tout de suite. Mais puisque ma résolution était formelle, et qu'avant vingt-quatre heures, dans les cadres vides de la journée du lendemain où tout se plaçait si bien parce que je n'y étais pas encore, mes bonnes dispositions se réaliseraient aisément, il valait mieux ne pas choisir un soir où j'étais mal disposé pour un début auquel les jours suivants, hélas ! ne devaient pas se montrer plus propices. Mais j'étais raisonnable. De la part de qui avait attendu des années, il eût été puéril de ne pas supporter un retard de trois jours. Certain que le surlendemain j'aurais déjà écrit quelques pages, je ne disais plus un seul mot à mes parents de ma décision ; j'aimais mieux patienter quelques heures, et apporter à ma grand'mère consolée et convaincue, de l'ouvrage en train. Malheureusement le lendemain n'était pas cette journée extérieure et vaste que j'avais attendue dans la fièvre. Quand il était fini, ma paresse et ma lutte pénible contre certains obstacles internes avaient simplement duré vingt-quatre heures de plus. Et au bout de quelques jours, mes plans n'ayant pas été réalisés, je n'avais plus le même espoir qu'ils le seraient immédiatement, partant, plus autant de courage pour subordonner tout à cette réalisation : je recommençais à veiller, n'ayant plus pour m'obliger à me coucher de bonne heure un soir, la vision certaine de voir l'oeuvre commencée le lendemain matin. Il me fallait avant de reprendre mon élan quelques jours de détente, et la seule fois où ma grand'mère osa d'un ton doux et désenchanté formuler ce reproche : « Hé bien, ce travail, on n'en parle même plus ? » je lui en voulus, persuadé que, n'ayant pas su voir que mon parti était irrévocablement pris, elle venait d'en ajourner encore et pour longtemps peut-être, l'exécution, par l'énervement que son déni de justice me causait et sous l'empire duquel je ne voudrais pas commencer mon oeuvre. Elle sentit que son scepticisme venait de heurter à l'aveugle une volonté. Elle s'en excusa, me dit en m'embrassant : « Pardon, je ne dirai plus rien. » Et pour que je ne me décourageasse pas, m'assura que du jour où je serais bien portant, le travail viendrait tout seul par surcroît.

D'ailleurs, me disais-je, en passant ma vie chez les Swann ne fais-je pas comme Bergotte ? À mes parents il semblait presque que, tout en étant paresseux, je menais, puisque c'était dans le même salon qu'un grand écrivain, la vie la plus favorable au talent. Et pourtant, que quelqu'un puisse être dispensé de faire ce talent soi-même, par le dedans, et le reçoive d'autrui, est aussi impossible que se faire une bonne santé (malgré qu'on manque à toutes les règles de l'hygiène et qu'on commette les pires excès) rien qu'en dînant souvent en ville avec un médecin. La personne du reste qui était le plus complètement dupe de l'illusion qui m'abusait ainsi que mes parents, c'était Odette. Quand je lui disais que je ne pouvais pas venir, qu'il fallait que je restasse à travailler, elle avait l'air de trouver que je faisais bien des embarras, qu'il y avait un peu de sottise et de prétention dans mes paroles :
