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
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "baron de Charlus",
        "le baron"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "Jupien le décrit comme « resté coureur » et sous surveillance; anecdote où, aveugle, il est « avec un enfant qui n’avait pas dix ans »; « en proie presque chaque jour à des crises de dépression mentale » où il avoue sa germanophilie; l’entourage, « Jupien ou la duchesse de Guermantes », doit interrompre et donner une version « honorable »; Jupien conclut qu’il « n’est plus qu’un grand enfant ».",
      "explanation": "The passage locally lowers Charlus by exposing compromised sexual and political behaviors, a mental fragility, and a dependence on constant guardianship, confirmed by the regular intervention of close ones to cover him."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Revealed as inconsiderate, reckless, and mentally diminished, Charlus significantly loses local esteem and appears dependent on others."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-6-p-10"
}

### Candidate characters

[
  "Jupien",
  "duchesse de Guermantes",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

La duchesse de Létourville, qui n'allait pas à la matinée de la princesse de Guermantes, parce qu'elle venait d'être longtemps malade, passa à ce moment à pied à côté de nous, et apercevant le baron, dont elle ignorait la récente attaque, s'arrêta pour lui dire bonjour. Mais la maladie qu'elle venait d'avoir faisait qu'elle ne comprenait pas mieux, mais supportait plus impatiemment, avec une mauvaise humeur nerveuse où il y avait peut-être beaucoup de pitié, la maladie des autres. Entendant le baron prononcer difficilement et à faux certains mots, lui voyant bouger difficilement le bras, elle jeta les yeux tour à tour sur Jupien et sur moi comme pour nous demander l'explication d'un phénomène aussi choquant. Comme nous ne lui dîmes rien, ce fut à baron de Charlus lui-même qu'elle adressa un long regard plein de tristesse mais aussi de reproches. Elle avait l'air de lui faire grief d'être avec elle, dehors, dans une attitude aussi peu usuelle que s'il fût sorti sans cravate ou sans souliers. À une nouvelle faute de prononciation que commit le baron, la douleur et l'indignation de duchesse de Guermantes augmentant ensemble, elle dit au baron : « baron de Charlus ! » sur le ton interrogatif et exaspéré des gens trop nerveux qui ne peuvent supporter d'attendre une minute et, si on les fait entrer tout de suite en s'excusant d'achever sa toilette, vous disent amèrement, non pour s'excuser mais pour s'accuser : « Mais alors, je vous dérange ! », comme si c'était un crime de la part de celui qu'on dérange. Finalement, elle nous quitta d'un air de plus en plus navré en disant au baron : « Vous feriez mieux de rentrer. »

### Passage

Charlus demanda à s'asseoir sur un fauteuil pour se reposer pendant que Jupien et moi ferions quelques pas et tira péniblement de sa poche un livre qui me sembla être un livre de prières. Je n'étais pas fâché de pouvoir apprendre par Jupien bien des détails sur l'état de santé du baron. « Je suis content de causer avec vous, Monsieur, me dit Jupien, mais nous n'irons pas plus loin que le rond-point. Dieu merci, le baron va bien maintenant, mais je n'ose pas le laisser longtemps seul, il est toujours le même, il a trop bon coeur, il donnerait tout ce qu'il a aux autres, et puis ce n'est pas tout, il est resté coureur comme un jeune homme et je suis obligé d'ouvrir les yeux. – D'autant plus qu'il a retrouvé les siens, répondis-je ; on m'avait beaucoup attristé en me disant qu'il avait perdu la vue. – Sa paralysie s'était, en effet, portée là, il ne voyait absolument plus. Pensez que, pendant la cure qui lui a fait, du reste, tant de bien, il est resté plusieurs mois sans voir plus qu'un aveugle de naissance. – Cela devait au moins rendre inutile toute une partie de votre surveillance ? – Pas le moins du monde, à peine arrivé dans un hôtel, il me demandait comment était telle personne de service. Je l'assurais qu'il n'y avait que des horreurs. Mais il sentait bien que cela ne pouvait pas être universel, que je devais quelquefois mentir. Voyez-vous, ce petit polisson ! Et puis il avait une espèce de flair, d'après la voix peut-être, je ne sais pas. Alors il s'arrangeait pour m'envoyer faire d'urgence des courses. Un jour – vous m'excuserez de vous dire cela, mais vous êtes venu une fois par hasard dans le Temple de l'Impudeur, je n'ai rien à vous cacher (d'ailleurs, il avait toujours une satisfaction assez peu sympathique à faire étalage des secrets qu'il détenait) – je rentrais d'une de ces courses soi-disant pressées, d'autant plus vite que je me figurais bien qu'elle avait été arrangée à dessein, quand, au moment où j'approchais de la chambre du baron, j'entendis une voix qui disait : « Quoi ? – Comment, répondit le baron, c'était donc la première fois ? » J'entrai sans frapper, et quelle ne fut pas ma frayeur. Le baron, trompé par la voix qui était, en effet, plus forte qu'elle n'est d'habitude à cet âge-là (et à cette époque-là le baron était complètement aveugle), était, lui qui aimait plutôt autrefois les personnes mûres, avec un enfant qui n'avait pas dix ans.

On m'a raconté qu'à cette époque-là il était en proie presque chaque jour à des crises de dépression mentale, caractérisée non pas précisément par de la divagation, mais par la confession à haute voix – devant des tiers dont il oubliait la présence ou la sévérité – d'opinions qu'il avait l'habitude de cacher, sa germanophilie par exemple. Ainsi, longtemps après la fin de la guerre, il gémissait de la défaite des Allemands, parmi lesquels il se comptait, et disait orgueilleusement : « Et pourtant il ne se peut pas que nous ne prenions pas notre revanche, car nous avons prouvé que c'est nous qui étions capables de la plus grande résistance, et qui avions la meilleure organisation. » Ou bien ses confidences prenaient un autre ton, et il s'écriait rageusement : « Que Lord X ou le prince de X ne viennent pas redire ce qu'ils disaient hier, car je me suis tenu à quatre pour ne pas leur répondre : « Vous savez bien que vous en êtes au moins autant que moi. » Inutile d'ajouter que, quand Charlus faisait ainsi, dans les moments où, comme on dit, il n'était pas très « présent », des aveux germanophiles ou autres, les personnes de l'entourage qui se trouvaient là, que ce fût Jupien ou la Mme de Guermantes, avaient l'habitude d'interrompre les paroles imprudentes et d'en donner, pour les tiers moins intimes et plus indiscrets, une interprétation forcée mais honorable. « Mais mon Dieu ! s'écria Jupien, j'avais bien raison de vouloir que nous ne nous éloignions pas, le voilà qui a trouvé déjà le moyen d'entrer en conversation avec un garçon jardinier. Adieu, Monsieur, il vaut mieux que je vous quitte et que je ne laisse pas un instant seul mon malade qui n'est plus qu'un grand enfant. »

* * *

Je descendis de nouveau de voiture un peu avant d'arriver chez la princesse de Guermantes et je recommençai à penser à cette lassitude et à cet ennui avec lesquels j'avais essayé, la veille, de noter la ligne qui, dans une des campagnes réputées les plus belles de France, séparait sur les arbres l'ombre de la lumière. Certes, les conclusions intellectuelles que j'en avais tirées n'affectaient pas aujourd'hui aussi cruellement ma sensibilité. Elles restaient les mêmes. Mais comme chaque fois que je me trouvais arraché à mes habitudes, sorti à une autre heure, dans un lieu nouveau, j'éprouvais un vif plaisir.

Ce plaisir me semblait aujourd'hui un plaisir purement frivole, celui d'aller à une matinée chez Mme de Guermantes. Mais puisque je savais maintenant que je ne pouvais rien atteindre de plus que des plaisirs frivoles, à quoi bon me les refuser ? Je me redisais que je n'avais éprouvé en essayant cette description rien de cet enthousiasme qui n'est pas le seul mais qui est un premier critérium du talent. J'essayais maintenant de tirer de ma mémoire d'autres « instantanés », notamment des instantanés qu'elle avait pris à Venise, mais rien que ce mot me la rendait ennuyeuse comme une exposition de photographies, et je ne me sentais pas plus de goût, plus de talent, pour décrire maintenant ce que j'avais vu autrefois qu'hier ce que j'observais d'un oeil minutieux et morne, au moment même. Dans un instant tant d'amis que je n'avais pas vus depuis si longtemps allaient sans doute me demander de ne plus m'isoler ainsi, de leur consacrer mes journées. Je n'aurais aucune raison de le leur refuser, puisque j'avais maintenant la preuve que je n'étais plus bon à rien, que la littérature ne pouvait plus me causer aucune joie, soit par ma faute, étant trop peu doué, soit par la sienne, si elle était, en effet, moins chargée de réalité que je n'avais cru.
