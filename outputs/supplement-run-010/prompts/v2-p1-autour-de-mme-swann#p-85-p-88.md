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
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "le marquis de Norpois"
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
      "confidence": 0.86,
      "evidence": "Après avoir promis: « je vais leur dire cela », Norpois montre « une expression d'hésitation et de mécontentement » et le regard tourné vers « l'interlocuteur invisible »; le narrateur « comprit que cette commission, il ne la ferait jamais »; plus tard, on rapporte qu'il a dit avoir « vu le moment où j'allais lui baiser les mains » et, interrogeant Odette, « il n'avait pas cru devoir dire pour qui il le demandait ».",
      "explanation": "The passage locally lowers Norpois: he prudently retracts a casual promise, suspects the narrator's desire, shows indiscretion by retailing the episode of the quasi-hand-kiss, and does not even dare to associate the narrator's name with his request to Odette."
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
      "confidence": 0.86,
      "explanation": "He appears less benevolent and less discreet than supposed: withdrawal of the promised intercession and later gossip tarnish his image."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-85-p-88"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "Swann",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Je préfère la figure de Gilberte, mais j'admire aussi énormément sa mère, je vais me promener au Bois rien que dans l'espoir de la voir passer.

### Passage

– Ah ! mais je vais leur dire cela, elles seront très flattées.

Pendant qu'il disait ces mots, Norpois était, pour quelques secondes encore, dans la situation de toutes les personnes qui, m'entendant parler de Swann comme d'un homme intelligent, de ses parents comme d'agents de change honorables, de sa maison comme d'une belle maison, croyaient que je parlerais aussi volontiers d'un autre homme aussi intelligent, d'autres agents de change aussi honorables, d'une autre maison aussi belle ; c'est le moment où un homme sain d'esprit qui cause avec un fou ne s'est pas encore aperçu que c'est un fou. Norpois savait qu'il n'y a rien que de naturel dans le plaisir de regarder les jolies femmes, qu'il est de bonne compagnie, dès que quelqu'un nous parle avec chaleur de l'une d'elles, de faire semblant de croire qu'il en est amoureux, de l'en plaisanter, et de lui promettre de seconder ses desseins. Mais en disant qu'il parlerait de moi à Gilberte et à sa mère (ce qui me permettrait, comme une divinité de l'Olympe qui a pris la fluidité d'un souffle ou plutôt l'aspect du vieillard dont Minerve emprunte les traits, de pénétrer moi-même, invisible, dans le salon de Odette, d'attirer son attention, d'occuper sa pensée, d'exciter sa reconnaissance pour mon admiration, de lui apparaître comme l'ami d'un homme important, de lui sembler à l'avenir digne d'être invité par elle et d'entrer dans l'intimité de sa famille), cet homme important qui allait user en ma faveur du grand prestige qu'il devait avoir aux yeux de Odette, m'inspira subitement une tendresse si grande que j'eus peine à me retenir de ne pas embrasser ses douces mains blanches et fripées, qui avaient l'air d'être restées trop longtemps dans l'eau. J'en ébauchai presque le geste que je me crus seul à avoir remarqué. Il est difficile en effet à chacun de nous de calculer exactement à quelle échelle ses paroles ou ses mouvements apparaissent à autrui ; par peur de nous exagérer notre importance et en grandissant dans des proportions énormes le champ sur lequel sont obligés de s'étendre les souvenirs des autres au cours de leur vie, nous nous imaginons que les parties accessoires de notre discours, de nos attitudes, pénètrent à peine dans la conscience, à plus forte raison ne demeurent pas dans la mémoire de ceux avec qui nous causons. C'est d'ailleurs à une supposition de ce genre qu'obéissent les criminels quand ils retouchent après coup un mot qu'ils ont dit et duquel ils pensent qu'on ne pourra confronter cette variante à aucune autre version. Mais il est bien possible que, même en ce qui concerne la vie millénaire de l'humanité, la philosophie du feuilletoniste selon laquelle tout est promis à l'oubli soit moins vraie qu'une philosophie contraire qui prédirait la conservation de toutes choses. Dans le même journal où le moraliste du « Premier Paris » nous dit d'un événement, d'un chef-d'oeuvre, à plus forte raison d'une chanteuse qui eut « son heure de célébrité » : « Qui se souviendra de tout cela dans dix ans ? », à la troisième page, le compte rendu de l'Académie des Inscriptions ne parle-t-il pas souvent d'un fait par lui-même moins important, d'un poème de peu de valeur, qui date de l'époque des Pharaons et qu'on connaît encore intégralement ? Peut-être n'en est-il pas tout à fait de même dans la courte vie humaine. Pourtant quelques années plus tard, dans une maison où Norpois, qui se trouvait en visite, me semblait le plus solide appui que j'y pusse rencontrer, parce qu'il était l'ami de mon père, indulgent, porté à nous vouloir du bien à tous, d'ailleurs habitué par sa profession et ses origines à la discrétion, quand, une fois l'Ambassadeur parti, on me raconta qu'il avait fait allusion à une soirée d'autrefois dans laquelle il avait « vu le moment où j'allais lui baiser les mains », je ne rougis pas seulement jusqu'aux oreilles, je fus stupéfait d'apprendre qu'étaient si différentes de ce que j'aurais cru, non seulement la façon dont Norpois parlait de moi, mais encore la composition de ses souvenirs ; ce « potin » m'éclaira sur les proportions inattendues de distraction et de présence d'esprit, de mémoire et d'oubli dont est fait l'esprit humain ; et, je fus aussi merveilleusement surpris que le jour où je lus pour la première fois, dans un livre de Maspero, qu'on savait exactement la liste des chasseurs qu'Assourbanipal invitait à ses battues, dix siècles avant Jésus-Christ.

– Oh ! Monsieur, dis-je à Norpois, quand il m'annonça qu'il ferait part à Gilberte et à sa mère de l'admiration que j'avais pour elles, si vous faisiez cela, si vous parliez de moi à Odette, ce ne serait pas assez de toute ma vie pour vous témoigner ma gratitude, et cette vie vous appartiendrait ! Mais je tiens à vous faire remarquer que je ne connais pas Odette et que je ne lui ai jamais été présenté.

J'avais ajouté ces derniers mots par scrupule et pour ne pas avoir l'air de m'être vanté d'une relation que je n'avais pas. Mais en les prononçant, je sentais qu'ils étaient déjà devenus inutiles, car dès le début de mon remerciement, d'une ardeur réfrigérante, j'avais vu passer sur le visage de l'Ambassadeur une expression d'hésitation et de mécontentement, et dans ses yeux ce regard vertical, étroit et oblique (comme, dans le dessin en perspective d'un solide, la ligne fuyante d'une de ses faces), regard qui s'adresse à cet interlocuteur invisible qu'on a en soi-même, au moment où on lui dit quelque chose que l'autre interlocuteur, le Monsieur avec qui on parlait jusqu'ici – moi dans la circonstance – ne doit pas entendre. Je me rendis compte aussitôt que ces phrases que j'avais prononcées et qui, faibles encore auprès de l'effusion reconnaissante dont j'étais envahi, m'avaient paru devoir toucher Norpois et achever de le décider à une intervention qui lui eût donné si peu de peine, et à moi tant de joie, étaient peut-être (entre toutes celles qu'eussent pu chercher diaboliquement des personnes qui m'eussent voulu du mal) les seules qui pussent avoir pour résultat de l'y faire renoncer. En les entendant en effet, de même qu'au moment où un inconnu, avec qui nous venions d'échanger agréablement des impressions que nous avions pu croire semblables sur des passants que nous nous accordions à trouver vulgaires, nous montre tout à coup l'abîme pathologique qui le sépare de nous en ajoutant négligemment tout en tâtant sa poche : « C'est malheureux que je n'aie pas mon revolver, il n'en serait pas resté un seul », Norpois qui savait que rien n'était moins précieux ni plus aisé que d'être recommandé à Odette et introduit chez elle, et qui vit que pour moi, au contraire, cela présentait un tel prix, par conséquent, sans doute, une grande difficulté, pensa que le désir, normal en apparence, que j'avais exprimé, devait dissimuler quelque pensée différente, quelque visée suspecte, quelque faute antérieure, à cause de quoi, dans la certitude de déplaire à Odette, personne n'avait jusqu'ici voulu se charger de lui transmettre une commission de ma part. Et je compris que cette commission, il ne la ferait jamais, qu'il pourrait voir Odette quotidiennement pendant des années, sans pour cela lui parler une seule fois de moi. Il lui demanda cependant quelques jours plus tard un renseignement que je désirais et chargea mon père de me le transmettre. Mais il n'avait pas cru devoir dire pour qui il le demandait. Elle n'apprendrait donc pas que je connaissais Norpois et que je souhaitais tant d'aller chez elle ; et ce fut peut-être un malheur moins grand que je ne croyais. Car la seconde de ces nouvelles n'eût probablement pas beaucoup ajouté à l'efficacité, d'ailleurs incertaine, de la première. Pour Odette, l'idée de sa propre vie et de sa demeure n'éveillant aucun trouble mystérieux, une personne qui la connaissait, qui allait chez elle, ne lui semblait pas un être fabuleux comme il le paraissait à moi qui aurais jeté dans les fenêtres de Swann une pierre si j'avais pu écrire sur elle que je connaissais Norpois : j'étais persuadé qu'un tel message, même transmis d'une façon aussi brutale, m'eût donné beaucoup plus de prestige aux yeux de la maîtresse de la maison qu'il ne l'eût indisposée contre moi. Mais, même si j'avais pu me rendre compte que la mission dont ne s'acquitta pas Norpois fût restée sans utilité, bien plus, qu'elle eût pu me nuire auprès des Swann, je n'aurais pas eu le courage, s'il s'était montré consentant, d'en décharger l'Ambassadeur et de renoncer à la volupté, si funestes qu'en pussent être les suites, que mon nom et ma personne se trouvassent ainsi un moment auprès de Gilberte, dans sa maison et sa vie inconnues.
