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
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Ce nom de Gilberte passa près de moi… formant… un petit nuage d’une couleur précieuse… »",
      "explanation": "The narrator poetically invests Gilberte’s name with aura and desirability, elevating her in his eyes as a figure surrounded by a rich, inaccessible world."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Locally, Gilberte is enhanced by the narrator’s rapt admiration and the aura surrounding her name."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-10-p-12"
}

### Candidate characters

[
  "Bergotte",
  "Françoise",
  "la Berma",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Mais je n'étais encore qu'en chemin vers le dernier degré de l'allégresse ; je l'atteignis enfin (ayant seulement alors la révélation que sur les rues clapotantes, rougies du reflet des fresques de Giorgione, ce n'était pas, comme j'avais, malgré tant d'avertissements, continué à l'imaginer, les hommes « majestueux et terribles comme la mer, portant leur armure aux reflets de bronze sous les plis de leur manteau sanglant » qui se promèneraient dans Venise la semaine prochaine, la veille de Pâques, mais que ce pourrait être moi, le personnage minuscule que, dans une grande photographie de Saint-Marc qu'on m'avait prêtée, l'illustrateur avait représenté, en chapeau melon, devant les proches), quand j'entendis mon père me dire : « Il doit faire encore froid sur le Grand Canal, tu ferais bien de mettre à tout hasard dans ta malle ton pardessus d'hiver et ton gros veston. » À ces mots je m'élevai à une sorte d'extase ; ce que j'avais cru jusque-là impossible, je me sentis vraiment pénétrer entre ces « rochers d'améthyste pareils à un récif de la mer des Indes » ; par une gymnastique suprême et au-dessus de mes forces, me dévêtant comme d'une carapace sans objet de l'air de ma chambre, qui m'entourait, je le remplaçai par des parties égales d'air vénitien, cette atmosphère marine, indicible et particulière comme celle des rêves que mon imagination avait enfermée dans le nom de Venise, je sentis s'opérer en moi une miraculeuse désincarnation ; elle se doubla aussitôt de la vague envie de vomir qu'on éprouve quand on vient de prendre un gros mal de gorge, et on dut me mettre au lit avec une fièvre si tenace, que docteur Cottard déclara qu'il fallait renoncer non seulement à me laisser partir maintenant à Florence et à Venise mais, même quand je serais entièrement rétabli, m'éviter, d'ici au moins un an, tout projet de voyage et toute cause d'agitation.

### Passage

Et hélas, il défendit aussi d'une façon absolue qu'on me laissât aller au théâtre entendre la Berma ; l'artiste sublime, à laquelle Bergotte trouvait du génie, m'aurait, en me faisant connaître quelque chose qui était peut-être aussi important et aussi beau, consolé de n'avoir pas été à Florence et à Venise, de n'aller pas à Balbec. On devait se contenter de m'envoyer chaque jour aux Champs-Élysées, sous la surveillance d'une personne qui m'empêcherait de me fatiguer et qui fut Françoise, entrée à notre service après la mort de ma tante Léonie. Aller aux Champs-Élysées me fut insupportable. Si seulement Bergotte les eût décrits dans un de ses livres, sans doute j'aurais désiré de les connaître, comme toutes les choses dont on avait commencé par mettre le « double » dans mon imagination. Elle les réchauffait, les faisait vivre, leur donnait une personnalité, et je voulais les retrouver dans la réalité ; mais dans ce jardin public rien ne se rattachait à mes rêves.

Un jour, comme je m'ennuyais à notre place familière, à côté des chevaux de bois, Françoise m'avait emmené en excursion – au delà de la frontière que gardent à intervalles égaux les petits bastions des marchandes de sucre d'orge – dans ces régions voisines mais étrangères où les visages sont inconnus, où passe la voiture aux chèvres ; puis elle était revenue prendre ses affaires sur sa chaise adossée à un massif de lauriers ; en l'attendant je foulais la grande pelouse chétive et rase, jaunie par le soleil, au bout de laquelle le bassin est dominé par une statue quand, de l'allée, s'adressant à une fillette à cheveux roux qui jouait au volant devant la vasque, une autre, en train de mettre son manteau et de serrer sa raquette, lui cria, d'une voix brève : « Adieu, Gilberte, je rentre, n'oublie pas que nous venons ce soir chez toi après dîner. » Ce nom de Gilberte passa près de moi, évoquant d'autant plus l'existence de celle qu'il désignait qu'il ne la nommait pas seulement comme un absent dont on parle, mais l'interpellait ; il passa ainsi près de moi, en action pour ainsi dire, avec une puissance qu'accroissait la courbe de son jet et l'approche de son but ; transportant à son bord, je le sentais, la connaissance, les notions qu'avait de celle à qui il était adressé, non pas moi, mais l'amie qui l'appelait, tout ce que, tandis qu'elle le prononçait, elle revoyait ou, du moins, possédait en sa mémoire, de leur intimité quotidienne, des visites qu'elles se faisaient l'une chez l'autre, de tout cet inconnu, encore plus inaccessible et plus douloureux pour moi d'être au contraire si familier et si maniable pour cette fille heureuse qui m'en frôlait, sans que j'y puisse pénétrer, et le jetait en plein air dans un cri ; laissant déjà flotter dans l'air l'émanation délicieuse qu'il avait fait se dégager, en les touchant avec précision, de quelques points invisibles de la vie de Gilberte, du soir qui allait venir, tel qu'il serait, après dîner, chez elle ; formant, passager céleste au milieu des enfants et des bonnes, un petit nuage d'une couleur précieuse, pareil à celui qui, bombé au-dessus d'un beau jardin du Poussin, reflète minutieusement comme un nuage d'opéra, plein de chevaux et de chars, quelque apparition de la vie des dieux ; jetant enfin, sur cette herbe pelée, à l'endroit où elle était un morceau à la fois de pelouse flétrie et un moment de l'après-midi de la blonde joueuse de volant (qui ne s'arrêta de le lancer et de le rattraper que quand une institutrice à plumet bleu l'eût appelée), une petite bande merveilleuse et couleur d'héliotrope, impalpable comme un reflet et superposée comme un tapis, sur lequel je ne pus me lasser de promener mes pas attardés, nostalgiques et profanateurs, tandis que Françoise me criait : « Allons, aboutonnez voir votre paletot et filons », et que je remarquais pour la première fois avec irritation qu'elle avait un langage vulgaire, et hélas, pas de plumet bleu à son chapeau.

Retournerait-elle seulement aux Champs-Élysées ? Le lendemain elle n'y était pas ; mais je l'y vis les jours suivants ; je tournais tout le temps autour de l'endroit où elle jouait avec ses amies, si bien qu'une fois où elles ne se trouvèrent pas en nombre pour leur partie de barres elle me fit demander si je voulais compléter leur camp, et je jouai désormais avec elle chaque fois qu'elle était là. Mais ce n'était pas tous les jours ; il y en avait où elle était empêchée de venir par ses cours, le catéchisme, un goûter, toute cette vie séparée de la mienne que par deux fois, condensée dans le nom de Gilberte, j'avais senti passer si douloureusement près de moi, dans le raidillon de Combray et sur la pelouse des Champs-Élysées. Ces jours-là, elle annonçait d'avance qu'on ne la verrait pas ; si c'était à cause de ses études, elle disait : « C'est rasant, je ne pourrai pas venir demain ; vous allez tous vous amuser sans moi », d'un air chagrin qui me consolait un peu ; mais en revanche quand elle était invitée à une matinée, et que, ne le sachant pas je lui demandais si elle viendrait jouer, elle me répondait : « J'espère bien que non ! J'espère bien que maman me laissera aller chez mon amie. » Du moins ces jours-là, je savais que je ne la verrais pas, tandis que d'autres fois, c'était à l'improviste que sa mère l'emmenait faire des courses avec elle, et le lendemain elle disait : « Ah ! oui, je suis sortie avec maman », comme une chose naturelle, et qui n'eût pas été pour quelqu'un le plus grand malheur possible. Il y avait aussi les jours de mauvais temps où son institutrice, qui pour elle-même craignait la pluie, ne voulait pas l'emmener aux Champs-Élysées.
