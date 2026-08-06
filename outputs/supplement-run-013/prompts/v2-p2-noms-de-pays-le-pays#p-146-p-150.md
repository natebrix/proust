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
      "canonical_name": "Mme de Villeparisis",
      "surface_forms": [
        "Mme de Villeparisis",
        "Madame de Villeparisis"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme de Villeparisis",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "Elle conteste les citations romantiques du narrateur (« Et vous trouvez cela beau ? … génial comme vous dites ? »), relate des anecdotes internes sur Chateaubriand devenant « une charge à la maison », et le narrateur confirme: « Mon père n’était pas sorcier, mais M. de Chateaubriand se contentait de servir toujours un même morceau tout préparé. »",
      "explanation": "Through her condescending tone and insider anecdotes, Mme de Villeparisis imposes a judgmental authority and takes precedence in the exchange; the narrator corroborates her point by validating Chateaubriand's predictability."
    }
  ],
  "status_effects": [
    {
      "character": "Mme de Villeparisis",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "She locally gains the discursive advantage by ridiculing romantic grandiloquence through internal evidence and a tone of authority confirmed by the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-146-p-150"
}

### Candidate characters

[
  "le narrateur",
  "le père du narrateur",
  "princesse de Luxembourg"
]

### Prior local context (optional)

Une fois que nous connûmes cette vieille route, pour changer, nous revînmes, à moins que nous ne l'eussions prise à l'aller, par une autre qui traversait les bois de Chantereine et de Canteloup. L'invisibilité des innombrables oiseaux qui s'y répondaient tout à côté de nous dans les arbres donnait la même impression de repos qu'on a les yeux fermés. Enchaîné à mon strapontin comme Prométhée sur son rocher, j'écoutais mes Océanides. Et, quand, par hasard, j'apercevais l'un de ces oiseaux qui passait d'une feuille sous une autre, il y avait si peu de lien apparent entre lui et ces chants que je ne croyais pas voir la cause de ceux-ci dans ce petit corps sautillant, étonné et sans regard.

### Passage

Cette route était pareille à bien d'autres de ce genre qu'on rencontre en France, montant en pente assez raide, puis redescendant sur une grande longueur. Au moment même, je ne lui trouvais pas un grand charme, j'étais seulement content de rentrer. Mais elle devint pour moi dans la suite une cause de joies en restant dans ma mémoire comme une amorce où toutes les routes semblables sur lesquelles je passerais plus tard au cours d'une promenade ou d'un voyage s'embrancheraient aussitôt sans solution de continuité et pourraient, grâce à elle, communiquer immédiatement avec mon coeur. Car dès que la voiture ou l'automobile s'engagerait dans une de ces routes qui auraient l'air d'être la continuation de celle que j'avais parcourue avec Mme de Villeparisis, ce à quoi ma conscience actuelle se trouverait immédiatement appuyée comme à mon passé le plus récent, ce serait (toutes les années intermédiaires se trouvant abolies) les impressions que j'avais eues par ces fins d'après-midi-là, en promenade près de Balbec, quand les feuilles sentaient bon, que la brume s'élevait et qu'au delà du prochain village on apercevrait entre les arbres le coucher du soleil comme s'il avait été quelque localité suivante, forestière, distante et qu'on n'atteindra pas le soir même. Raccordées à celles que j'éprouvais maintenant dans un autre pays, sur une route semblable, s'entourant de toutes les sensations accessoires de libre respiration, de curiosité, d'indolence, d'appétit, de gaieté qui leur étaient communes, excluant toutes les autres, ces impressions se renforceraient, prendraient la consistance d'un type particulier de plaisir, et presque d'un cadre d'existence que j'avais d'ailleurs rarement l'occasion de retrouver, mais dans lequel le réveil des souvenirs mettait au milieu de la réalité matériellement perçue une part assez grande de réalité évoquée, songée, insaisissable, pour me donner, au milieu de ces régions où je passais, plus qu'un sentiment esthétique, un désir fugitif mais exalté, d'y vivre désormais pour toujours. Que de fois, pour avoir simplement senti une odeur de feuillée, être assis sur un strapontin en face de Mme de Villeparisis, croiser la princesse de Luxembourg qui lui envoyait des bonjours de sa voiture, rentrer dîner au Grand-Hôtel, ne m'est-il pas apparu comme un de ces bonheurs ineffables que ni le présent ni l'avenir ne peuvent nous rendre et qu'on ne goûte qu'une fois dans la vie.

Souvent le jour était tombé avant que nous fussions de retour. Timidement je citais à Mme de Villeparisis en lui montrant la lune dans le ciel quelque belle expression de Chateaubriand, ou de Vigny, ou de Victor Hugo : « Elle répandait ce vieux secret de mélancolie » ou « pleurant comme Diane au bord de ses fontaines » ou « L'ombre était nuptiale, auguste et solennelle ».

– Et vous trouvez cela beau ? me demandait-elle, génial comme vous dites ? Je vous dirai que je suis toujours étonnée de voir qu'on prend maintenant au sérieux des choses que les amis de ces messieurs, tout en rendant pleine justice à leurs qualités, étaient les premiers à plaisanter. On ne prodiguait pas le nom de génie comme aujourd'hui, où si vous dites à un écrivain qu'il n'a que du talent il prend cela pour une injure. Vous me citez une grande phrase de M. de Chateaubriand sur le clair de lune. Vous allez voir que j'ai mes raisons pour y être réfractaire. M. de Chateaubriand venait bien souvent chez mon père. Il était du reste agréable quand on était seul parce qu'alors il était simple et amusant, mais dès qu'il y avait du monde, il se mettait à poser et devenait ridicule ; devant mon père, il prétendait avoir jeté sa démission à la face du roi et dirigé le conclave, oubliant que mon père avait été chargé par lui de supplier le roi de le reprendre, et l'avait entendu faire sur l'élection du pape les pronostics les plus insensés. Il fallait entendre sur ce fameux conclave M. de Blacas, qui était un autre homme que M. de Chateaubriand. Quant aux phrases de celui-ci sur le clair de lune elles étaient tout simplement devenues une charge à la maison. Chaque fois qu'il faisait clair de lune autour du château, s'il y avait quelque invité nouveau, on lui conseillait d'emmener M. de Chateaubriand prendre l'air après le dîner. Quand ils revenaient, mon père ne manquait pas de prendre à part l'invité : « M. de Chateaubriand a été bien éloquent ? – Oh ! oui. – Il vous a parlé du clair de lune. – Oui, comment savez-vous ? – Attendez, ne vous a-t-il pas dit, et il lui citait la phrase. – Oui, mais par quel mystère ? – Et il vous a parlé même du clair de lune dans la campagne romaine. – Mais vous êtes sorcier. » Mon père n'était pas sorcier, mais M. de Chateaubriand se contentait de servir toujours un même morceau tout préparé.

Au nom de Vigny elle se mit à rire.

– Celui qui disait : « Je suis le comte Alfred de Vigny. » On est comte ou on n'est pas comte, ça n'a aucune espèce d'importance.
