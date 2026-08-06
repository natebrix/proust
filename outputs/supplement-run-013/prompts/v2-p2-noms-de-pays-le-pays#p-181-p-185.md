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
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup",
        "Robert"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« c'était Robert de Saint-Loup qui rougissait comme si ç'avait été lui le coupable »; « manqué de l'indulgence dont il débordait »; « la rougeur... il la sentit par anticipation... monter au sien »",
      "explanation": "The narrator highlights Robert's delicacy and indulgence, as he blushes and feels guilty on behalf of a friend who made a social faux pas, anticipating the other's shame."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "His empathy and delicacy in the face of others' blunders enhance his local estimation by the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-181-p-185"
}

### Candidate characters

[
  "Bloch",
  "le narrateur"
]

### Prior local context (optional)

Et pourtant elle était, dans une certaine mesure, leur condition. C'est parce qu'il était un gentilhomme que cette activité mentale, ces aspirations socialistes, qui lui faisaient rechercher de jeunes étudiants prétentieux et mal mis, avaient chez lui quelque chose de vraiment pur et désintéressé qu'elles n'avaient pas chez eux. Se croyant l'héritier d'une caste ignorante et égoïste, il cherchait sincèrement à ce qu'ils lui pardonnassent ces origines aristocratiques qui exerçaient sur eux, au contraire, une séduction et à cause desquelles ils le recherchaient, tout en simulant à son égard la froideur et même l'insolence. Il était ainsi amené à faire des avances à des gens dont mes parents, fidèles à la sociologie de Combray, eussent été stupéfaits qu'il ne se détournât pas. Un jour que nous étions assis sur le sable, Robert de Saint-Loup et moi, nous entendîmes d'une tente de toile contre laquelle nous étions, sortir des imprécations contre le fourmillement d'Israélites qui infestait Balbec. « On ne peut faire deux pas sans en rencontrer, disait la voix. Je ne suis pas par principe irréductiblement hostile à la nationalité juive, mais ici il y a pléthore. On n'entend que : « Dis donc Apraham, chai fu Chakop. » On se croirait rue d'Aboukir. » L'homme qui tonnait ainsi contre Israël sortit enfin de la tente, nous levâmes les yeux sur cet antisémite. C'était mon camarade Bloch. Robert de Saint-Loup me demanda immédiatement de rappeler à celui-ci qu'ils s'étaient rencontrés au Concours général où Bloch avait eu le prix d'honneur, puis dans une Université populaire.

### Passage

Tout au plus souriais-je parfois de retrouver chez Saint-Loup les leçons des Jésuites dans la gêne que la peur de froisser faisait naître chez lui, chaque fois que quelqu'un de ses amis intellectuels commettait une erreur mondaine, faisait une chose ridicule à laquelle lui, Saint-Loup, n'attachait aucune importance, mais dont il sentait que l'autre aurait rougi si l'on s'en était aperçu. Et c'était Saint-Loup qui rougissait comme si ç'avait été lui le coupable, par exemple le jour où Bloch lui promettait d'aller le voir à l'hôtel, ajouta :

– Comme je ne peux pas supporter d'attendre parmi le faux chic de ces grands caravansérails, et que les tziganes me feraient trouver mal, dites au « laïft » de les faire taire et de vous prévenir de suite.

Personnellement, je ne tenais pas beaucoup à ce que Bloch vînt à l'hôtel. Il était à Balbec, non pas seul, malheureusement, mais avec ses soeurs qui y avaient elles-mêmes beaucoup de parents et d'amis. Or cette colonie juive était plus pittoresque qu'agréable. Il en était de Balbec comme de certains pays, la Russie ou la Roumanie, où les cours de géographie nous enseignent que la population israélite n'y jouit point de la même faveur et n'y est pas parvenue au même degré d'assimilation qu'à Paris par exemple. Toujours ensemble, sans mélange d'aucun autre élément, quand les cousines et les oncles de Bloch, ou leurs coreligionnaires mâles ou femelles se rendaient au Casino, les unes pour le « bal », les autres bifurquant vers le baccarat, ils formaient un cortège homogène en soi et entièrement dissemblable des gens qui les regardaient passer et les retrouvaient là tous les ans sans jamais échanger un salut avec eux, que ce fût la société des Cambremer, le clan du premier président, ou des grands et petits bourgeois, ou même de simples grainetiers de Paris, dont les filles, belles, fières, moqueuses et françaises comme les statues de Reims, n'auraient pas voulu se mêler à cette horde de fillasses mal élevées, poussant le souci des modes de « bains de mer » jusqu'à toujours avoir l'air de revenir de pêcher la crevette ou d'être en train de danser le tango. Quant aux hommes, malgré l'éclat des smokings et des souliers vernis, l'exagération de leur type faisait penser à ces recherches dites « intelligentes » des peintres qui, ayant à illustrer les Évangiles ou les Mille et Une Nuits, pensent au pays où la scène se passe et donnent à saint Pierre ou à Ali-Baba précisément la figure qu'avait le plus gros « ponte » de Balbec. Bloch me présenta ses soeurs, auxquelles il fermait le bec avec la dernière brusquerie et qui riaient aux éclats des moindres boutades de leur frère, leur admiration et leur idole. De sorte qu'il est probable que ce milieu devait renfermer comme tout autre, peut-être plus que tout autre, beaucoup d'agréments, de qualités et de vertus. Mais pour les éprouver, il eût fallu y pénétrer. Or, il ne plaisait pas, il le sentait, il voyait là la preuve d'un antisémitisme contre lequel il faisait front en une phalange compacte et close où personne d'ailleurs ne songeait à se frayer un chemin.

Pour ce qui est de « laïft », cela avait d'autant moins lieu de me surprendre que quelques jours auparavant, Bloch m'ayant demandé pourquoi j'étais venu à Balbec (il lui semblait au contraire tout naturel que lui-même y fût) et si c'était « dans l'espoir de faire de belles connaissances », comme je lui avais dit que ce voyage répondait à un de mes plus anciens désirs, moins profond pourtant que celui d'aller à Venise, il avait répondu : « Oui, naturellement, pour boire des sorbets avec les belles madames, tout en faisant semblant de lire les Stones of Venaïce, de Lord John Ruskin, sombre raseur et l'un des plus barbifiants bonshommes qui soient. » Bloch croyait donc évidemment qu'en Angleterre, non seulement tous les individus du sexe mâle sont lords, mais encore que la lettre i s'y prononce toujours aï. Quant à Saint-Loup, il trouvait cette faute de prononciation d'autant moins grave qu'il y voyait surtout un manque de ces notions presque mondaines que mon nouvel ami méprisait autant qu'il les possédait. Mais la peur que Bloch, apprenant un jour qu'on dit Venice et que Ruskin n'était pas lord, crût rétrospectivement que Saint-Loup l'avait trouvé ridicule, fit que ce dernier se sentit coupable comme s'il avait manqué de l'indulgence dont il débordait, et que la rougeur qui colorerait sans doute un jour le visage de Bloch à la découverte de son erreur, il la sentit par anticipation et réversibilité monter au sien. Car il pensait bien que Bloch attachait plus d'importance que lui à cette faute. Ce que Bloch prouva quelque temps après, un jour qu'il m'entendit prononcer « lift », en interrompant :

– Ah ! on dit lift. Et d'un ton sec et hautain :
