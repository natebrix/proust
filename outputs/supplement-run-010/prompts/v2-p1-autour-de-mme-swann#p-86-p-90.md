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
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.83,
      "evidence": "« une expression d'hésitation et de mécontentement »; le « regard ... à cet interlocuteur invisible »; Norpois pense à « quelque visée suspecte » et le narrateur comprend « qu'il ne la ferait jamais »; plus tard, Norpois « avait fait allusion ... 'vu le moment où j'allais lui baiser les mains' »",
      "explanation": "The narrator stages Norpois's suspicious withdrawal (failing to speak of Odette/Gilberte) and reveals his subsequent indiscretion in gossiping about the episode, which undermines his credit for kindness and discretion."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "He appears less benevolent and less discreet (implicit refusal to intercede, reported gossip), which diminishes his local esteem."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-86-p-90"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "Swann",
  "la Berma",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Ah ! mais je vais leur dire cela, elles seront très flattées.

### Passage

Pendant qu'il disait ces mots, Norpois était, pour quelques secondes encore, dans la situation de toutes les personnes qui, m'entendant parler de Swann comme d'un homme intelligent, de ses parents comme d'agents de change honorables, de sa maison comme d'une belle maison, croyaient que je parlerais aussi volontiers d'un autre homme aussi intelligent, d'autres agents de change aussi honorables, d'une autre maison aussi belle ; c'est le moment où un homme sain d'esprit qui cause avec un fou ne s'est pas encore aperçu que c'est un fou. Norpois savait qu'il n'y a rien que de naturel dans le plaisir de regarder les jolies femmes, qu'il est de bonne compagnie, dès que quelqu'un nous parle avec chaleur de l'une d'elles, de faire semblant de croire qu'il en est amoureux, de l'en plaisanter, et de lui promettre de seconder ses desseins. Mais en disant qu'il parlerait de moi à Gilberte et à sa mère (ce qui me permettrait, comme une divinité de l'Olympe qui a pris la fluidité d'un souffle ou plutôt l'aspect du vieillard dont Minerve emprunte les traits, de pénétrer moi-même, invisible, dans le salon de Odette, d'attirer son attention, d'occuper sa pensée, d'exciter sa reconnaissance pour mon admiration, de lui apparaître comme l'ami d'un homme important, de lui sembler à l'avenir digne d'être invité par elle et d'entrer dans l'intimité de sa famille), cet homme important qui allait user en ma faveur du grand prestige qu'il devait avoir aux yeux de Odette, m'inspira subitement une tendresse si grande que j'eus peine à me retenir de ne pas embrasser ses douces mains blanches et fripées, qui avaient l'air d'être restées trop longtemps dans l'eau. J'en ébauchai presque le geste que je me crus seul à avoir remarqué. Il est difficile en effet à chacun de nous de calculer exactement à quelle échelle ses paroles ou ses mouvements apparaissent à autrui ; par peur de nous exagérer notre importance et en grandissant dans des proportions énormes le champ sur lequel sont obligés de s'étendre les souvenirs des autres au cours de leur vie, nous nous imaginons que les parties accessoires de notre discours, de nos attitudes, pénètrent à peine dans la conscience, à plus forte raison ne demeurent pas dans la mémoire de ceux avec qui nous causons. C'est d'ailleurs à une supposition de ce genre qu'obéissent les criminels quand ils retouchent après coup un mot qu'ils ont dit et duquel ils pensent qu'on ne pourra confronter cette variante à aucune autre version. Mais il est bien possible que, même en ce qui concerne la vie millénaire de l'humanité, la philosophie du feuilletoniste selon laquelle tout est promis à l'oubli soit moins vraie qu'une philosophie contraire qui prédirait la conservation de toutes choses. Dans le même journal où le moraliste du « Premier Paris » nous dit d'un événement, d'un chef-d'oeuvre, à plus forte raison d'une chanteuse qui eut « son heure de célébrité » : « Qui se souviendra de tout cela dans dix ans ? », à la troisième page, le compte rendu de l'Académie des Inscriptions ne parle-t-il pas souvent d'un fait par lui-même moins important, d'un poème de peu de valeur, qui date de l'époque des Pharaons et qu'on connaît encore intégralement ? Peut-être n'en est-il pas tout à fait de même dans la courte vie humaine. Pourtant quelques années plus tard, dans une maison où Norpois, qui se trouvait en visite, me semblait le plus solide appui que j'y pusse rencontrer, parce qu'il était l'ami de mon père, indulgent, porté à nous vouloir du bien à tous, d'ailleurs habitué par sa profession et ses origines à la discrétion, quand, une fois l'Ambassadeur parti, on me raconta qu'il avait fait allusion à une soirée d'autrefois dans laquelle il avait « vu le moment où j'allais lui baiser les mains », je ne rougis pas seulement jusqu'aux oreilles, je fus stupéfait d'apprendre qu'étaient si différentes de ce que j'aurais cru, non seulement la façon dont Norpois parlait de moi, mais encore la composition de ses souvenirs ; ce « potin » m'éclaira sur les proportions inattendues de distraction et de présence d'esprit, de mémoire et d'oubli dont est fait l'esprit humain ; et, je fus aussi merveilleusement surpris que le jour où je lus pour la première fois, dans un livre de Maspero, qu'on savait exactement la liste des chasseurs qu'Assourbanipal invitait à ses battues, dix siècles avant Jésus-Christ.

– Oh ! Monsieur, dis-je à Norpois, quand il m'annonça qu'il ferait part à Gilberte et à sa mère de l'admiration que j'avais pour elles, si vous faisiez cela, si vous parliez de moi à Odette, ce ne serait pas assez de toute ma vie pour vous témoigner ma gratitude, et cette vie vous appartiendrait ! Mais je tiens à vous faire remarquer que je ne connais pas Odette et que je ne lui ai jamais été présenté.

J'avais ajouté ces derniers mots par scrupule et pour ne pas avoir l'air de m'être vanté d'une relation que je n'avais pas. Mais en les prononçant, je sentais qu'ils étaient déjà devenus inutiles, car dès le début de mon remerciement, d'une ardeur réfrigérante, j'avais vu passer sur le visage de l'Ambassadeur une expression d'hésitation et de mécontentement, et dans ses yeux ce regard vertical, étroit et oblique (comme, dans le dessin en perspective d'un solide, la ligne fuyante d'une de ses faces), regard qui s'adresse à cet interlocuteur invisible qu'on a en soi-même, au moment où on lui dit quelque chose que l'autre interlocuteur, le Monsieur avec qui on parlait jusqu'ici – moi dans la circonstance – ne doit pas entendre. Je me rendis compte aussitôt que ces phrases que j'avais prononcées et qui, faibles encore auprès de l'effusion reconnaissante dont j'étais envahi, m'avaient paru devoir toucher Norpois et achever de le décider à une intervention qui lui eût donné si peu de peine, et à moi tant de joie, étaient peut-être (entre toutes celles qu'eussent pu chercher diaboliquement des personnes qui m'eussent voulu du mal) les seules qui pussent avoir pour résultat de l'y faire renoncer. En les entendant en effet, de même qu'au moment où un inconnu, avec qui nous venions d'échanger agréablement des impressions que nous avions pu croire semblables sur des passants que nous nous accordions à trouver vulgaires, nous montre tout à coup l'abîme pathologique qui le sépare de nous en ajoutant négligemment tout en tâtant sa poche : « C'est malheureux que je n'aie pas mon revolver, il n'en serait pas resté un seul », Norpois qui savait que rien n'était moins précieux ni plus aisé que d'être recommandé à Odette et introduit chez elle, et qui vit que pour moi, au contraire, cela présentait un tel prix, par conséquent, sans doute, une grande difficulté, pensa que le désir, normal en apparence, que j'avais exprimé, devait dissimuler quelque pensée différente, quelque visée suspecte, quelque faute antérieure, à cause de quoi, dans la certitude de déplaire à Odette, personne n'avait jusqu'ici voulu se charger de lui transmettre une commission de ma part. Et je compris que cette commission, il ne la ferait jamais, qu'il pourrait voir Odette quotidiennement pendant des années, sans pour cela lui parler une seule fois de moi. Il lui demanda cependant quelques jours plus tard un renseignement que je désirais et chargea mon père de me le transmettre. Mais il n'avait pas cru devoir dire pour qui il le demandait. Elle n'apprendrait donc pas que je connaissais Norpois et que je souhaitais tant d'aller chez elle ; et ce fut peut-être un malheur moins grand que je ne croyais. Car la seconde de ces nouvelles n'eût probablement pas beaucoup ajouté à l'efficacité, d'ailleurs incertaine, de la première. Pour Odette, l'idée de sa propre vie et de sa demeure n'éveillant aucun trouble mystérieux, une personne qui la connaissait, qui allait chez elle, ne lui semblait pas un être fabuleux comme il le paraissait à moi qui aurais jeté dans les fenêtres de Swann une pierre si j'avais pu écrire sur elle que je connaissais Norpois : j'étais persuadé qu'un tel message, même transmis d'une façon aussi brutale, m'eût donné beaucoup plus de prestige aux yeux de la maîtresse de la maison qu'il ne l'eût indisposée contre moi. Mais, même si j'avais pu me rendre compte que la mission dont ne s'acquitta pas Norpois fût restée sans utilité, bien plus, qu'elle eût pu me nuire auprès des Swann, je n'aurais pas eu le courage, s'il s'était montré consentant, d'en décharger l'Ambassadeur et de renoncer à la volupté, si funestes qu'en pussent être les suites, que mon nom et ma personne se trouvassent ainsi un moment auprès de Gilberte, dans sa maison et sa vie inconnues.

Quand Norpois fut parti, mon père jeta un coup d'oeil sur le journal du soir ; je songeais de nouveau à la Berma. Le plaisir que j'avais eu à l'entendre exigeait d'autant plus d'être complété qu'il était loin d'égaler celui que je m'étais promis ; aussi s'assimilait-il immédiatement tout ce qui était susceptible de le nourrir, par exemple ces mérites que Norpois avait reconnus à la Berma et que mon esprit avait bus d'un seul trait comme un pré trop sec sur qui on verse de l'eau. Or mon père me passa le journal en me désignant un entrefilet conçu en ces termes : « La représentation de Phèdre qui a été donnée devant une salle enthousiaste où on remarquait les principales notabilités du monde des arts et de la critique a été pour Mme Berma, qui jouait le rôle de Phèdre, l'occasion d'un triomphe comme elle en a rarement connu de plus éclatant au cours de sa prestigieuse carrière. Nous reviendrons plus longuement sur cette représentation qui constitue un véritable événement théâtral ; disons seulement que les juges les plus autorisés s'accordaient à déclarer qu'une telle interprétation renouvelait entièrement le rôle de Phèdre, qui est un des plus beaux et des plus fouillés de Racine, et constituait la plus pure et la plus haute manifestation d'art à laquelle de notre temps il ait été donné d'assister. » Dès que mon esprit eut conçu cette idée nouvelle de « la plus pure et haute manifestation d'art », celle-ci se rapprocha du plaisir imparfait que j'avais éprouvé au théâtre, lui ajouta un peu de ce qui lui manquait et leur réunion forma quelque chose de si exaltant que je m'écriai : « Quelle grande artiste ! » Sans doute on peut trouver que je n'étais pas absolument sincère. Mais qu'on songe plutôt à tant d'écrivains qui, mécontents du morceau qu'ils viennent d'écrire, s'ils lisent un éloge du génie de Chateaubriand, ou évoquant tel grand artiste dont ils ont souhaité d'être l'égal, fredonnant par exemple en eux-mêmes telle phrase de Beethoven de laquelle ils comparent la tristesse à celle qu'ils ont voulu mettre dans leur prose, se remplissent tellement de cette idée de génie qu'ils l'ajoutent à leurs propres productions en repensant à elles, ne les voient plus telles qu'elles leur étaient apparues d'abord, et risquant un acte de foi dans la valeur de leur oeuvre se disent : « Après tout ! » sans se rendre compte que, dans le total qui détermine leur satisfaction finale, ils font entrer le souvenir de merveilleuses pages de Chateaubriand qu'ils assimilent aux leurs, mais enfin qu'ils n'ont point écrites ; qu'on se rappelle tant d'hommes qui croient en l'amour d'une maîtresse de qui ils ne connaissent que les trahisons ; tous ceux aussi qui espèrent alternativement soit une survie incompréhensible dès qu'ils pensent, maris inconsolables, à une femme qu'ils ont perdue et qu'ils aiment encore, artistes, à la gloire future de laquelle ils pourront jouir, soit un néant rassurant quand leur intelligence se reporte au contraire aux fautes que sans lui ils auraient à expier après leur mort ; qu'on pense encore aux touristes qu'exalte la beauté d'ensemble d'un voyage dont jour par jour ils n'ont éprouvé que de l'ennui, et qu'on dise, si dans la vie en commun que mènent les idées au sein de notre esprit, il est une seule de celles qui nous rendent le plus heureux qui n'ait été d'abord en véritable parasite demander à une idée étrangère et voisine le meilleur de la force qui lui manquait.

Ma mère ne parut pas très satisfaite que mon père ne songeât plus pour moi à la « carrière ». Je crois que, soucieuse avant tout qu'une règle d'existence disciplinât les caprices de mes nerfs, ce qu'elle regrettait, c'était moins de me voir renoncer à la diplomatie que m'adonner à la littérature. « Mais laisse donc, s'écria mon père, il faut avant tout prendre du plaisir à ce qu'on fait. Or, il n'est plus un enfant. Il sait bien maintenant ce qu'il aime, il est peu probable qu'il change, et il est capable de se rendre compte de ce qui le rendra heureux dans l'existence. » En attendant que, grâce à la liberté qu'elles m'octroyaient, je fusse, ou non, heureux dans l'existence, les paroles de mon père me firent ce soir-là bien de la peine. De tout temps ses gentillesses imprévues m'avaient, quand elles se produisaient, donné une telle envie d'embrasser au-dessus de sa barbe ses joues colorées que si je n'y cédais pas, c'était seulement par peur de lui déplaire. Aujourd'hui, comme un auteur s'effraye de voir ses propres rêveries qui lui paraissent sans grande valeur parce qu'il ne les sépare pas de lui-même, obliger un éditeur à choisir un papier, à employer des caractères peut-être trop beaux pour elles, je me demandais si mon désir d'écrire était quelque chose d'assez important pour que mon père dépensât à cause de cela tant de bonté. Mais surtout en parlant de mes goûts qui ne changeraient plus, de ce qui était destiné à rendre mon existence heureuse, il insinuait en moi deux terribles soupçons. Le premier, c'était que (alors que chaque jour je me considérais comme sur le seuil de ma vie encore intacte et qui ne débuterait que le lendemain matin) mon existence était déjà commencée, bien plus, que ce qui allait en suivre ne serait pas très différent de ce qui avait précédé. Le second soupçon, qui n'était à vrai dire qu'une autre forme du premier, c'est que je n'étais pas situé en dehors du Temps, mais soumis à ses lois, tout comme ces personnages de roman qui, à cause de cela, me jetaient dans une telle tristesse, quand je lisais leur vie, à Combray, au fond de ma guérite d'osier. Théoriquement on sait que la terre tourne, mais en fait on ne s'en aperçoit pas, le sol sur lequel on marche semble ne pas bouger et on vit tranquille. Il en est ainsi du Temps dans la vie. Et pour rendre sa fuite sensible, les romanciers sont obligés, en accélérant follement les battements de l'aiguille, de faire franchir au lecteur dix, vingt, trente ans, en deux minutes. Au haut d'une page on a quitté un amant plein d'espoir, au bas de la suivante on le retrouve octogénaire, accomplissant péniblement dans le préau d'un hospice sa promenade quotidienne, répondant à peine aux paroles qu'on lui adresse, ayant oublié le passé. En disant de moi : « Ce n'est plus un enfant, ses goûts ne changeront plus, etc. », mon père venait tout d'un coup de me faire apparaître à moi-même dans le Temps, et me causait le même genre de tristesse que si j'avais été non pas encore l'hospitalisé ramolli, mais ces héros dont l'auteur, sur un ton indifférent qui est particulièrement cruel, nous dit à la fin d'un livre : « Il quitte de moins en moins la campagne. Il a fini par s'y fixer définitivement, etc. »
