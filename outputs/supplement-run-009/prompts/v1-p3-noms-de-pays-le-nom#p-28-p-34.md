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
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "uncertain",
      "confidence": 0.68,
      "evidence": "« Sa figure resplendit et ce fut en sautant de joie qu'elle me répondit : — Demain, comptez-y, mon bel ami, mais je ne viendrai pas ! ... »; « les mots où Gilberte avait laissé éclater sa joie de ne pas venir de longtemps »; « cette marque d'indifférence »",
      "explanation": "Gilberte is portrayed as joyfully refusing future meetings and displaying indifference, which locally casts her in a colder, less considerate light. The narrator later romanticizes even this indifference, softening but not erasing the negative framing."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.62,
      "explanation": "Within the passage, she is seen as openly indifferent and delighted not to come, which lowers her appraisal in the narrator's eyes despite his subsequent self-consoling romanticization."
    }
  ],
  "ambiguities": [
    "The chief harm falls on the unnamed narrator (not in the alias map); the snub is therefore encoded indirectly as a diminishment of Gilberte."
  ],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-28-p-34"
}

### Candidate characters

[
  "Françoise",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Il répondait poliment aux saluts des camarades de Gilberte, même au mien quoiqu'il fût brouillé avec ma famille, mais sans avoir l'air de me connaître. (Cela me rappela qu'il m'avait pourtant vu bien souvent à la campagne ; souvenir que j'avais gardé mais dans l'ombre, parce que depuis que j'avais revu Gilberte, pour moi Swann était surtout son père, et non plus le Swann de Combray ; comme les idées sur lesquelles j'embranchais maintenant son nom étaient différentes des idées dans le réseau desquelles il était autrefois compris et que je n'utilisais plus jamais quand j'avais à penser à lui, il était devenu un personnage nouveau ; je le rattachai pourtant par une ligne artificielle, secondaire et transversale à notre invité d'autrefois ; et comme rien n'avait plus pour moi de prix que dans la mesure où mon amour pouvait en profiter, ce fut avec un mouvement de honte et le regret de ne pouvoir les effacer que je retrouvai les années où, aux yeux de ce même Swann qui était en ce moment devant moi aux Champs-Élysées et à qui heureusement Gilberte n'avait peut-être pas dit mon nom, je m'étais si souvent le soir rendu ridicule en envoyant demander à la mère du narrateur de monter dans ma chambre me dire bonsoir, pendant qu'elle prenait le café avec lui, mon père et mes grands-parents à la table du jardin.) Il disait à Gilberte qu'il lui permettait de faire une partie, qu'il pouvait attendre un quart d'heure, et s'asseyant comme tout le monde sur une chaise de fer payait son ticket de cette main que Philippe VII avait si souvent retenue dans la sienne, tandis que nous commencions à jouer sur la pelouse, faisant envoler les pigeons, dont les beaux corps irisés qui ont la forme d'un coeur et sont comme les lilas du règne des oiseaux, venaient se réfugier comme en des lieux d'asile, tel sur le grand vase de pierre, à qui son bec en y disparaissant faisait faire le geste et assignait la destination d'offrir en abondance les fruits ou les graines qu'il avait l'air d'y picorer, tel autre sur le front de la statue, qu'il semblait surmonter d'un de ces objets en émail desquels la polychromie varie dans certaines oeuvres antiques la monotonie de la pierre, et d'un attribut qui, quand la déesse le porte, lui vaut une épithète particulière et en fait, comme pour une mortelle un prénom différent, une divinité nouvelle.

### Passage

Un de ces jours de soleil qui n'avait pas réalisé mes espérances, je n'eus pas le courage de cacher ma déception à Gilberte.

– J'avais justement beaucoup de choses à vous demander, lui dis-je. Je croyais que ce jour compterait beaucoup dans notre amitié. Et aussitôt arrivée, vous allez partir ! Tâchez de venir demain de bonne heure, que je puisse enfin vous parler.

Sa figure resplendit et ce fut en sautant de joie qu'elle me répondit :

– Demain, comptez-y, mon bel ami, mais je ne viendrai pas ! j'ai un grand goûter ; après-demain non plus, je vais chez une amie pour voir de ses fenêtres l'arrivée du roi Théodose, ce sera superbe, et le lendemain encore à Michel Strogoff et puis après, cela va être bientôt Noël et les vacances du jour de l'An. Peut-être on va m'emmener dans le midi. Ce que ce serait chic ! quoique cela me fera manquer un arbre de Noël ; en tous cas si je reste à Paris, je ne viendrai pas ici car j'irai faire des visites avec maman. Adieu, voilà papa qui m'appelle.

Je revins avec Françoise par les rues qui étaient encore pavoisées de soleil, comme au soir d'une fête qui est finie. Je ne pouvais pas traîner mes jambes.

– Ça n'est pas étonnant, dit Françoise, ce n'est pas un temps de saison, il fait trop chaud. Hélas ! mon Dieu, de partout il doit y avoir bien des pauvres malades, c'est à croire que là-haut aussi tout se détraque.

Je me redisais en étouffant mes sanglots les mots où Gilberte avait laissé éclater sa joie de ne pas venir de longtemps aux Champs-Élysées. Mais déjà le charme dont, par son simple fonctionnement, se remplissait mon esprit dès qu'il songeait à elle, la position particulière, unique – fût elle affligeante – où me plaçait inévitablement, par rapport à Gilberte, la contrainte interne d'un pli mental, avaient commencé à ajouter, même à cette marque d'indifférence, quelque chose de romanesque, et au milieu de mes larmes se formait un sourire qui n'était que l'ébauche timide d'un baiser. Et quand vint l'heure du courrier, je me dis ce soir-là comme tous les autres : « Je vais recevoir une lettre de Gilberte, elle va me dire enfin qu'elle n'a jamais cessé de m'aimer, et m'expliquera la raison mystérieuse pour laquelle elle a été forcée de me le cacher jusqu'ici, de faire semblant de pouvoir être heureuse sans me voir, la raison pour laquelle elle a pris l'apparence de la Gilberte simple camarade. »
