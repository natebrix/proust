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
        "Robert",
        "le marquis"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.96
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.91,
      "evidence": "« il avait avec Gilberte des affectations de sensibleries poussées jusqu'à la comédie »; « il lui mentait tout le temps »; le “Monsieur du pays” dément le prétexte; puis Robert « sanglotait, s'inondait d'eau froide, parlait de sa mort prochaine » et « s'abattait sur le parquet ».",
      "explanation": "The narrator presents Robert as duplicitous and theatrical, whose lies are exposed and who resorts to a sentimental comedy deemed ridiculous."
    },
    {
      "event_id": "E2",
      "source": "Françoise",
      "target": "Robert de Saint-Loup",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.78,
      "evidence": "« les protecteurs sont ceux qui aiment, qui souffrent, qui pardonnent »; elle « leur donnait le beau rôle »; « De même estimait-elle plus Robert de Saint-Loup que Morel … car c'est un homme qui avait trop de coeur ».",
      "explanation": "Françoise values Robert as a warm-hearted protector and bestows him a higher esteem than Morel, attributing to him the 'beautiful role' in these relationships."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.78,
      "explanation": "He is locally enhanced by Françoise's admiration who recognizes his warmth and the nobility of the protector."
    },
    {
      "character": "Robert de Saint-Loup",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "His credibility and authority are diminished by the discovered duplicity and the melodramatic staging."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p1-a-tansonville#p-6-p-10"
}

### Candidate characters

[
  "Gilberte",
  "Jupien",
  "Legrandin",
  "M. de Marsantes",
  "Morel",
  "baron de Charlus",
  "duc de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Pour être complet il faudrait faire entrer en ligne de compte le désir, plus il vieillissait, de paraître jeune, et même l'impatience de ces hommes, toujours ennuyés, toujours blasés, que sont les gens trop intelligents pour la vie relativement oisive qu'ils mènent et où leurs facultés ne se réalisent pas. Sans doute l'oisiveté même de ceux-là peut se traduire par de la nonchalance. Mais, surtout depuis la faveur dont jouissent les exercices physiques, l'oisiveté a pris une forme sportive, même en dehors des heures de sport et qui se traduit par une vivacité fébrile qui croit ne pas laisser à l'ennui le temps ni la place de se développer.

### Passage

Devenant beaucoup plus sec, il ne faisait presque plus preuve vis-à-vis de ses amis, par exemple vis-à-vis de moi, d'aucune sensibilité. Et en revanche il avait avec Gilberte des affectations de sensibleries poussées jusqu'à la comédie, qui déplaisaient. Ce n'est pas qu'en réalité Gilberte lui fût indifférente. Non, Saint-Loup l'aimait. Mais il lui mentait tout le temps, et son esprit de duplicité, sinon le fond même de ses mensonges, était perpétuellement découvert. Et alors il ne croyait pouvoir s'en tirer qu'en exagérant dans des proportions ridicules la tristesse réelle qu'il avait de peiner Gilberte. Il arrivait à Tansonville obligé, disait-il, de repartir le lendemain matin pour une affaire avec un certain Monsieur du pays qui était censé l'attendre à Paris et qui, précisément rencontré dans la soirée près de Combray, dévoilait involontairement le mensonge au courant duquel Saint-Loup avait négligé de le mettre, en disant qu'il était venu dans le pays se reposer pour un mois et ne retournerait pas à Paris d'ici là. Saint-Loup rougissait, voyait le sourire mélancolique et fin de Gilberte, se dépêtrait – en l'insultant – du gaffeur, rentrait avant sa femme, lui faisait remettre un mot désespéré où il lui disait qu'il avait fait un mensonge pour ne pas lui faire de peine, pour qu'en le voyant repartir pour une raison qu'il ne pouvait pas lui dire elle ne crût pas qu'il ne l'aimait pas (et tout cela, bien qu'il l'écrivît comme un mensonge, était en somme vrai), puis faisait demander s'il pouvait entrer chez elle et là, moitié tristesse réelle, moitié énervement de cette vie, moitié simulation chaque jour plus audacieuse, sanglotait, s'inondait d'eau froide, parlait de sa mort prochaine, quelquefois s'abattait sur le parquet comme s'il se fût trouvé mal. Gilberte ne savait pas dans quelle mesure elle devait le croire, le supposait menteur à chaque cas particulier, et s'inquiétait de ce pressentiment d'une mort prochaine, mais pensait que d'une façon générale elle était aimée, qu'il avait peut-être une maladie qu'elle ne savait pas, et n'osait pas à cause de cela le contrarier et lui demander de renoncer à ses voyages. Je comprenais, du reste, d'autant moins pourquoi il se faisait que Morel fût reçu comme l'enfant de la maison partout où étaient les Saint-Loup, à Paris, à Tansonville.

Françoise, qui avait déjà vu tout ce que Charlus avait fait pour Jupien et tout ce que Saint-Loup de Saint-Loup faisait pour Morel, n'en concluait pas que c'était un trait qui reparaissait à certaines générations chez les Guermantes, mais plutôt – comme Legrandin aidait beaucoup Théodore – elle avait fini, elle personne si morale et si pleine de préjugés, par croire que c'était une coutume que son universalité rendait respectable. Elle disait toujours d'un jeune homme, que ce fût Morel ou Théodore : « Il a trouvé un Monsieur qui s'est toujours intéressé à lui et qui lui a bien aidé. » Et comme en pareil cas les protecteurs sont ceux qui aiment, qui souffrent, qui pardonnent, Françoise, entre eux et les mineurs qu'ils détournaient, n'hésitait pas à leur donner le beau rôle, à leur trouver « bien du coeur ». Elle blâmait sans hésiter Théodore qui avait joué bien des tours à Legrandin, et semblait pourtant ne pouvoir guère avoir de doutes sur la nature de leurs relations, car elle ajoutait : « Alors le petit a compris qu'il fallait y mettre du sien et y a dit : « Prenez-moi avec vous, je vous aimerai bien, je vous cajolerai bien », et ma foi ce Monsieur a tant de coeur que bien sûr que Théodore est sûr de trouver près de lui peut-être bien plus qu'il ne mérite, car c'est une tête brûlée, mais ce Monsieur est si bon que j'ai souvent dit à Jeannette (la fiancée de Théodore) : Petite, si jamais vous êtes dans la peine, allez vers ce Monsieur. Il coucherait plutôt par terre et vous donnerait son lit. Il a trop aimé le petit Théodore pour le mettre dehors, bien sûr qu'il ne l'abandonnera jamais. »

De même estimait-elle plus Saint-Loup que Morel et jugeait-elle que, malgré tous les coups que Morel avait faits, le marquis ne le laisserait jamais dans la peine, car c'est un homme qui avait trop de coeur, ou alors il faudrait qu'il lui soit arrivé à lui-même de grands revers.

C'est au cours d'un de ces entretiens, qu'ayant demandé le nom de famille de Théodore, qui vivait maintenant dans le Midi, je compris brusquement que c'était lui qui m'avait écrit pour mon article du Figaro cette lettre, d'une écriture populaire et d'un langage charmant, dont le nom du signataire m'était alors inconnu.

Saint-Loup insistait pour que je restasse à Tansonville et laissa échapper une fois, bien qu'il ne cherchât visiblement plus à me faire plaisir, que ma venue avait été pour sa femme une joie telle qu'elle en était restée, à ce qu'elle lui avait dit, transportée de joie tout un soir, un soir où elle se sentait si triste que je l'avais, en arrivant à l'improviste, miraculeusement sauvée du désespoir, « peut-être du pire », ajouta-t-il. Il me demandait de tâcher de la persuader qu'il l'aimait, me disant que la femme qu'il aimait aussi, il l'aimait moins qu'elle et romprait bientôt. « Et pourtant », ajouta-t-il, avec une telle félinité et un tel besoin de confidence que je croyais par moments que le nom de Morel allait, malgré Saint-Loup, « sortir » comme le numéro d'une loterie, « j'avais de quoi être fier. Cette femme qui me donna tant de preuves de sa tendresse et que je vais sacrifier à Gilberte, jamais elle n'avait fait attention à un homme, elle se croyait elle-même incapable d'être amoureuse. Je suis le premier. Je savais qu'elle s'était refusée à tout le monde tellement que, quand j'ai reçu la lettre adorable où elle me disait qu'il ne pouvait y avoir de bonheur pour elle qu'avec moi, je n'en revenais pas. Évidemment, il y aurait de quoi me griser, si la pensée de voir cette pauvre petite Gilberte en larmes ne m'était pas intolérable. Ne trouves-tu pas qu'elle a quelque chose de Rachel ? », me disait-il. Et en effet j'avais été frappé d'une vague ressemblance qu'on pouvait à la rigueur trouver maintenant entre elles. Peut-être tenait-elle à une similitude réelle de quelques traits (dus par exemple à l'origine hébraïque pourtant si peu marquée chez Gilberte) à cause de laquelle Saint-Loup, quand sa famille avait voulu qu'il se mariât, s'était senti attiré vers Gilberte. Elle tenait aussi à ce que Gilberte, ayant surpris des photographies de Rachel, cherchait pour plaire à Saint-Loup à imiter certaines habitudes chères à l'actrice, comme d'avoir toujours des noeuds rouges dans les cheveux, un ruban de velours noir au bras, et se teignait les cheveux pour paraître brune. Puis sentant que ses chagrins lui donnaient mauvaise mine, elle essayait d'y remédier. Elle le faisait parfois sans mesure. Un jour où Saint-Loup devait venir le soir pour vingt-quatre heures à Tansonville, je fus stupéfait de la voir venir se mettre à table si étrangement différente de ce qu'elle était, non seulement autrefois, mais même les jours habituels, que je restai stupéfait comme si j'avais eu devant moi une actrice, une espèce de Théodora. Je sentais que malgré moi je la regardais trop fixement dans ma curiosité de savoir ce qu'elle avait de changé. Cette curiosité fut d'ailleurs bientôt satisfaite quand elle se moucha, car, malgré toutes les précautions qu'elle y mit, par toutes les couleurs qui restèrent sur le mouchoir, en faisant une riche palette, je vis qu'elle était complètement peinte. C'était cela qui lui faisait cette bouche sanglante et qu'elle s'efforçait de rendre rieuse en croyant que cela lui allait bien, tandis que l'heure du train qui s'approchait sans que Gilberte sût si son mari arrivait vraiment ou s'il n'enverrait pas une de ces dépêches dont duc de Guermantes  avait spirituellement fixé le modèle : « Impossible venir, mensonge suit », pâlissait ses joues et cernait ses yeux.
