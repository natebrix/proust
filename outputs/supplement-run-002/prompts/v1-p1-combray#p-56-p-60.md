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
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Françoise",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "Elle est décrite comme « bonne si intelligente et active », « qui faisait tout bien, travaillant comme un cheval… sans bruit », « la seule… qui apportait vraiment bouillants »; les maîtres tiennent à ces serviteurs dont ils ont « éprouvé les capacités réelles »; elle reçoit la joie d’être comprise par la mère du narrateur.",
      "explanation": "The narrator strongly valorizes Françoise, highlighting her competence, her discretion, and the lasting trust she inspires in the masters, as well as the affective recognition she receives in this household."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "In this passage, she is presented as exceptionally capable and dignified, appreciated both for her work and for the relationship of trust and understanding that she maintains with the family."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-56-p-60"
}

### Candidate characters

[
  "Octave",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Je n'étais pas avec ma tante depuis cinq minutes, qu'elle me renvoyait par peur que je la fatigue. Elle tendait à mes lèvres son triste front pâle et fade sur lequel, à cette heure matinale, elle n'avait pas encore arrangé ses faux cheveux, et où les vertèbres transparaissaient comme les pointes d'une couronne d'épines ou les grains d'un rosaire, et elle me disait : « Allons, mon pauvre enfant, va-t'en, va te préparer pour la messe ; et si en bas tu rencontres Françoise, dis-lui de ne pas s'amuser trop longtemps avec vous, qu'elle monte bientôt voir si je n'ai besoin de rien. »

### Passage

Françoise, en effet, qui était depuis des années à son service et ne se doutait pas alors qu'elle entrerait un jour tout à fait au nôtre, délaissait un peu ma tante pendant les mois où nous étions là. Il y avait eu dans mon enfance, avant que nous allions à Combray, quand ma tante Léonie passait encore l'hiver à Paris chez sa mère, un temps où je connaissais si peu Françoise que, le 1er janvier, avant d'entrer chez ma grand'tante, ma mère me mettait dans la main une pièce de cinq francs et me disait : « Surtout ne te trompe pas de personne. Attends pour donner que tu m'entendes dire : « Bonjour Françoise » ; en même temps je te toucherai légèrement le bras. » À peine arrivions-nous dans l'obscure antichambre de ma tante que nous apercevions dans l'ombre, sous les tuyaux d'un bonnet éblouissant, raide et fragile comme s'il avait été de sucre filé, les remous concentriques d'un sourire de reconnaissance anticipé. C'était Françoise, immobile et debout dans l'encadrement de la petite porte du corridor comme une statue de sainte dans sa niche. Quand on était un peu habitué à ces ténèbres de chapelle, on distinguait sur son visage l'amour désintéressé de l'humanité, le respect attendri pour les hautes classes qu'exaltait dans les meilleures régions de son coeur l'espoir des étrennes. Maman me pinçait le bras avec violence et disait d'une voix forte : « Bonjour Françoise. » À ce signal mes doigts s'ouvraient et je lâchais la pièce qui trouvait pour la recevoir une main confuse, mais tendue. Mais depuis que nous allions à Combray je ne connaissais personne mieux que Françoise ; nous étions ses préférés, elle avait pour nous, au moins pendant les premières années, avec autant de considération que pour ma tante, un goût plus vif, parce que nous ajoutions, au prestige de faire partie de la famille (elle avait pour les liens invisibles que noue entre les membres d'une famille la circulation d'un même sang, autant de respect qu'un tragique grec), le charme de n'être pas ses maîtres habituels. Aussi, avec quelle joie elle nous recevait, nous plaignant de n'avoir pas encore plus beau temps, le jour de notre arrivée, la veille de Pâques, où souvent il faisait un vent glacial, quand maman lui demandait des nouvelles de sa fille et de ses neveux, si son petit-fils était gentil, ce qu'on comptait faire de lui, s'il ressemblerait à sa grand'mère.

Et quand il n'y avait plus de monde là, maman qui savait que Françoise pleurait encore ses parents morts depuis des années, lui parlait d'eux avec douceur, lui demandait mille détails sur ce qu'avait été leur vie.

Elle avait deviné que Françoise n'aimait pas son gendre et qu'il lui gâtait le plaisir qu'elle avait à être avec sa fille, avec qui elle ne causait pas aussi librement quand il était là. Aussi, quand Françoise allait les voir, à quelques lieues de Combray, maman lui disait en souriant : « N'est-ce pas Françoise, si Julien a été obligé de s'absenter et si vous avez Marguerite à vous toute seule pour toute la journée, vous serez désolée, mais vous vous ferez une raison ? » Et Françoise disait en riant : « Madame sait tout ; madame est pire que les rayons X (elle disait x avec une difficulté affectée et un sourire pour se railler elle-même, ignorante, d'employer ce terme savant), qu'on a fait venir pour Octave et qui voient ce que vous avez dans le coeur », et disparaissait, confuse qu'on s'occupât d'elle, peut-être pour qu'on ne la vît pas pleurer ; maman était la première personne qui lui donnât cette douce émotion de sentir que sa vie, ses bonheurs, ses chagrins de paysanne pouvaient présenter de l'intérêt, être un motif de joie ou de tristesse pour une autre qu'elle-même. Ma tante se résignait à se priver un peu d'elle pendant notre séjour, sachant combien ma mère appréciait le service de cette bonne si intelligente et active, qui était aussi belle dès cinq heures du matin dans sa cuisine, sous son bonnet dont le tuyautage éclatant et fixe avait l'air d'être en biscuit, que pour aller à la grand'messe ; qui faisait tout bien, travaillant comme un cheval, qu'elle fût bien portante ou non, mais sans bruit, sans avoir l'air de rien faire, la seule des bonnes de ma tante qui, quand maman demandait de l'eau chaude ou du café noir, les apportait vraiment bouillants ; elle était un de ces serviteurs qui, dans une maison, sont à la fois ceux qui déplaisent le plus au premier abord à un étranger, peut-être parce qu'ils ne prennent pas la peine de faire sa conquête et n'ont pas pour lui de prévenance, sachant très bien qu'ils n'ont aucun besoin de lui, qu'on cesserait de le recevoir plutôt que de les renvoyer ; et qui sont en revanche ceux à qui tiennent le plus les maîtres qui ont éprouvé leur capacités réelles, et ne se soucient pas de cet agrément superficiel, de ce bavardage servile qui fait favorablement impression à un visiteur, mais qui recouvre souvent une inéducable nullité.

Quand Françoise, après avoir veillé à ce que mes parents eussent tout ce qu'il leur fallait, remontait une première fois chez ma tante pour lui donner sa pepsine et lui demander ce qu'elle prendrait pour déjeuner, il était bien rare qu'il ne fallût pas donner déjà son avis ou fournir des explications sur quelque événement d'importance :

– Françoise, imaginez-vous que Mme Goupil est passée plus d'un quart d'heure en retard pour aller chercher sa soeur ; pour peu qu'elle s'attarde sur son chemin cela ne me surprendrait point qu'elle arrive après l'élévation.
