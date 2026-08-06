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
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.94,
      "evidence": "« elle nous présentait avec tant d’aisance, Gilberte et moi, … qu’il eût été difficile de dire … laquelle des deux était la grande dame »",
      "explanation": "The narrator elevates Odette by stressing her poise and ease when facing aristocratic passersby, making her indistinguishable from a grande dame."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "Odette's bearing and reception place her on equal footing with aristocratic ladies, boosting her local social standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-191-p-195"
}

### Candidate characters

[
  "Bloch",
  "Gilberte",
  "Mme Bontemps",
  "Napoléon III",
  "Swann",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Au Jardin d'Acclimatation, que j'étais fier, quand nous étions descendus de voiture, de m'avancer à côté de Odette ! Tandis que dans sa démarche nonchalante elle laissait flotter son manteau, je jetais sur elle des regards d'admiration auxquels elle répondait coquettement par un long sourire. Maintenant si nous rencontrions l'un ou l'autre des camarades, fille ou garçon, de Gilberte, qui nous saluait de loin, j'étais à mon tour regardé par eux comme un de ces êtres que j'avais enviés, un de ces amis de Gilberte qui connaissaient sa famille et étaient mêlés à l'autre partie de sa vie, celle qui ne se passait pas aux Champs-Élysées.

### Passage

Souvent dans les allées du Bois ou du Jardin d'Acclimatation nous croisions, nous étions salués par telle ou telle grande dame amie des Swann, qu'il lui arrivait de ne pas voir et que lui signalait sa femme. « Swann, vous ne voyez pas Mme de Montmorency ? » et Swann, avec le sourire amical dû à une longue familiarité, se découvrait pourtant largement avec une élégance qui n'était qu'à lui. Quelquefois la dame s'arrêtait, heureuse de faire à Odette une politesse qui ne tirait pas à conséquence et de laquelle on savait qu'elle ne chercherait pas à profiter ensuite, tant Swann l'avait habituée à rester sur la réserve. Elle n'en avait pas moins pris toutes les manières du monde, et si élégante et noble de port que fût la dame, Odette l'égalait toujours en cela ; arrêtée un moment auprès de l'amie que son mari venait de rencontrer, elle nous présentait avec tant d'aisance, Gilberte et moi, gardait tant de liberté et de calme dans son amabilité, qu'il eût été difficile de dire de la femme de Swann ou de l'aristocratique passante laquelle des deux était la grande dame. Le jour où nous étions allés voir les Cynghalais, comme nous revenions, nous aperçûmes, venant dans notre direction et suivie de deux autres qui semblaient l'escorter, une dame âgée, mais encore belle, enveloppée dans un manteau sombre et coiffée d'une petite capote attachée sous le cou par deux brides. « Ah ! voilà quelqu'un qui va vous intéresser », me dit Swann. La vieille dame maintenant à trois pas de nous souriait avec une douceur caressante. Swann se découvrit, Odette s'abaissa en une révérence et voulut baiser la main de la dame pareille à un portrait de Winterhalter qui la releva et l'embrassa. « Voyons, voulez-vous mettre votre chapeau, vous », dit-elle à Swann, d'une grosse voix un peu maussade, en amie familière. « Je vais vous présenter à Son Altesse Impériale », me dit Odette. Swann m'attira un moment à l'écart pendant que Odette causait du beau temps et des animaux nouvellement arrivés au Jardin d'Acclimatation, avec l'Altesse. « C'est la princesse Mathilde, me dit-il, vous savez, l'amie de Flaubert, de Sainte-Beuve, de Dumas. Songez, c'est la nièce de Napoléon Ier ! Elle a été demandée en mariage par Napoléon III et par l'empereur de Russie. Ce n'est pas intéressant ? Parlez-lui un peu. Mais je voudrais qu'elle ne nous fît pas rester une heure sur nos jambes. – J'ai rencontré Taine qui m'a dit que la Princesse était brouillée avec lui, dit Swann. – Il s'est conduit comme un cauchon, dit-elle d'une voix rude et en prononçant le mot comme si ç'avait été le nom de l'évêque contemporain de Jeanne d'Arc. Après l'article qu'il a écrit sur l'Empereur je lui ai laissé une carte avec P.P.C. » J'éprouvais la surprise qu'on a en ouvrant la correspondance de la duchesse d'Orléans, née princesse Palatine. Et, en effet, la princesse Mathilde, animée de sentiments si français, les éprouvait avec une honnête rudesse comme en avait l'Allemagne d'autrefois et qu'elle avait hérités sans doute de sa mère wurtemburgeoise. Sa franchise un peu fruste et presque masculine, elle l'adoucissait, dès qu'elle souriait, de langueur italienne. Et le tout était enveloppé dans une toilette tellement Second Empire que, bien que la princesse la portât seulement sans doute par attachement aux modes qu'elle avait aimées, elle semblait avoir eu l'intention de ne pas commettre une faute de couleur historique et de répondre à l'attente de ceux qui attendaient d'elle l'évocation d'une autre époque. Je soufflai à Swann de lui demander si elle avait connu Musset. « Très peu, Monsieur, répondit-elle d'un air qui faisait semblant d'être fâché, et, en effet, c'était par plaisanterie qu'elle disait Monsieur à Swann, étant fort intime avec lui. Je l'ai eu une fois à dîner. Je l'avais invité pour sept heures. À sept heures et demie, comme il n'était pas là, nous nous mîmes à table. Il arriva à huit heures, me salua, s'assied, ne desserre pas les dents, part après le dîner sans que j'aie entendu le son de sa voix. Il était ivre-mort. Cela ne m'a pas beaucoup encouragée à recommencer. » Nous étions un peu à l'écart, Swann et moi. « J'espère que cette petite séance ne va pas se prolonger, me dit-il, j'ai mal à la plante des pieds. Aussi je ne sais pas pourquoi ma femme alimente la conversation. Après cela c'est elle qui se plaindra d'être fatiguée et moi je ne peux plus supporter ces stations debout. » Odette en effet, qui tenait le renseignement de Mme Bontemps, était en train de dire à la princesse que le gouvernement, comprenant enfin sa goujaterie, avait décidé de lui envoyer une invitation pour assister dans les tribunes à la visite que le tsar Nicolas devait faire le surlendemain aux Invalides. Mais la princesse qui, malgré les apparences, malgré le genre de son entourage composé surtout d'artistes et d'hommes de lettres, était restée au fond, et chaque fois qu'elle avait à agir, nièce de Napoléon : « Oui, Madame, je l'ai reçue ce matin et je l'ai renvoyée au ministre qui doit l'avoir à l'heure qu'il est. Je lui ai dit que je n'avais pas besoin d'invitation pour aller aux Invalides. Si le gouvernement désire que j'y aille, ce ne sera pas dans une tribune, mais dans notre caveau, où est le tombeau de l'Empereur. Je n'ai pas besoin de carte pour cela. J'ai mes clefs. J'entre comme je veux. Le gouvernement n'a qu'à me faire savoir s'il désire que je vienne ou non. Mais si j'y vais, ce sera là ou pas du tout. » À ce moment nous fûmes salués, Odette et moi, par un jeune homme qui lui dit bonjour sans s'arrêter et que je ne savais pas qu'elle connût : Bloch. Sur une question que je lui posai, Odette me dit qu'il lui avait été présenté par Mme Bontemps, qu'il était attaché au Cabinet du ministre, ce que j'ignorais. Du reste, elle ne devait pas l'avoir vu souvent – ou bien elle n'avait pas voulu citer le nom, trouvé peut-être par elle peu « chic », de Bloch – car elle dit qu'il s'appelait M. Moreul. Je lui assurai qu'elle confondait, qu'il s'appelait Bloch. La princesse redressa une traîne qui se déroulait derrière elle et que Odette regardait avec admiration. « C'est justement une fourrure que l'empereur de Russie m'avait envoyée, dit la princesse et comme j'ai été le voir tantôt, je l'ai mise pour lui montrer que cela avait pu s'arranger en manteau. – Il paraît que le prince Louis s'est engagé dans l'armée russe, la princesse va être désolée de ne plus l'avoir près d'elle, dit Odette qui ne voyait pas les signes d'impatience de son mari. – Il avait besoin de cela ! Comme je lui ai dit : Ce n'est pas une raison parce que tu as eu un militaire dans ta famille », répondit la princesse, faisant, avec cette brusque simplicité, allusion à Napoléon Ier.

Swann ne tenait plus en place. « Madame, c'est moi qui vais faire l'Altesse et vous demander la permission de prendre congé, mais ma femme a été très souffrante et je ne veux pas qu'elle reste davantage immobile. » Odette refit la révérence et la princesse eut pour nous tous un divin sourire qu'elle sembla amener du passé, des grâces de sa jeunesse, des soirées de Compiègne et qui coula intact et doux sur le visage tout à l'heure grognon, puis elle s'éloigna suivie des deux dames d'honneur qui n'avaient fait, à la façon d'interprètes, de bonnes d'enfants, ou de gardes-malades que ponctuer notre conversation de phrases insignifiantes et d'explications inutiles. « Vous devriez aller écrire votre nom chez elle, un jour de cette semaine, me dit Odette ; on ne corne pas de bristol à toutes ces royalties, comme disent les Anglais, mais elle vous invitera si vous vous faites inscrire. »

Parfois dans ces derniers jours d'hiver, nous entrions avant d'aller nous promener dans quelqu'une des petites expositions qui s'ouvraient alors et où Swann, collectionneur de marque, était salué avec une particulière déférence par les marchands de tableaux chez qui elles avaient lieu. Et par ces temps encore froids, mes anciens désirs de partir pour le Midi et Venise étaient réveillés par ces salles où un printemps déjà avancé et un soleil ardent mettaient des reflets violacés sur les Alpilles roses et donnaient la transparence foncée de l'émeraude au Grand Canal. S'il faisait mauvais nous allions au concert ou au théâtre et goûter ensuite dans un « Thé ». Dès que Odette voulait me dire quelque chose qu'elle désirait que les personnes des tables voisines ou même les garçons qui servaient ne comprissent pas, elle me le disait en anglais comme si c'eût été un langage connu de nous deux seulement. Or tout le monde savait l'anglais, moi seul je ne l'avais pas encore appris et étais obligé de le dire à Odette pour qu'elle cessât de faire sur les personnes qui buvaient le thé ou sur celles qui l'apportaient des réflexions que je devinais désobligeantes sans que j'en comprisse, ni que l'individu visé en perdît un seul mot.

Une fois, à propos d'une matinée théâtrale, Gilberte me causa un étonnement profond. C'était justement le jour dont elle m'avait parlé d'avance et où tombait l'anniversaire de la mort de son grand-père. Nous devions, elle et moi, aller entendre avec son institutrice les fragments d'un opéra et Gilberte s'était habillée dans l'intention de se rendre à cette exécution musicale, gardant l'air d'indifférence qu'elle avait l'habitude de montrer pour la chose que nous devions faire, disant que ce pouvait être n'importe quoi pourvu que cela me plût et fût agréable à ses parents. Avant le déjeuner, sa mère nous prit à part pour lui dire que cela ennuyait son père de nous voir aller au concert ce jour-là. Je trouvai que c'était trop naturel. Gilberte resta impassible mais devint pâle d'une colère qu'elle ne put cacher, et ne dit plus un mot. Quand Swann revint, sa femme l'emmena à l'autre bout du salon et lui parla à l'oreille. Il appela Gilberte et la prit à part dans la pièce à côté. On entendit des éclats de voix. Je ne pouvais cependant pas croire que Gilberte, si soumise, si tendre, si sage, résistât à la demande de son père, un jour pareil et pour une cause si insignifiante. Enfin Swann sortit en lui disant :

– Tu sais ce que je t'ai dit. Maintenant, fais ce que tu voudras.
