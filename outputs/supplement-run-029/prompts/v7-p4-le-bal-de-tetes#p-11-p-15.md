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
      "canonical_name": "Legrandin",
      "surface_forms": [
        "Legrandin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Legrandin",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« La suppression du rose … donnait à sa figure l'apparence grisâtre … Un dieu ! un revenant plutôt. Il avait perdu … de tenir des discours ingénieux. … on se disait que cette cause … c'était la vieillesse. »",
      "explanation": "The narrator portrays Legrandin as a pale, diminished 'ghost' of himself, no longer witty or lively, explicitly attributing this decline to old age."
    }
  ],
  "status_effects": [
    {
      "character": "Legrandin",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Legrandin's standing is clearly lowered as he is depicted as lifeless and bereft of his former charm and brilliance."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-11-p-15"
}

### Candidate characters

[
  "Bergotte",
  "Gilberte",
  "Mme de Cambremer",
  "Odette",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur",
  "marquis de Cambremer"
]

### Prior local context (optional)

En attendant la duchesse de Guermantes dire : « Comment, si j'ai connu le maréchal ? Mais j'ai connu des gens bien plus représentatifs, duchesse de Guermantes de Galliera, Pauline de Périgord, Mgr Dupanloup », je regrettais naïvement de ne pas avoir connu moi-même ceux qu'elle appelait un reste d'ancien régime. J'aurais dû penser qu'on appelle ancien régime ce dont on n'a pu connaître que la fin ; c'est ainsi que ce que nous apercevons à l'horizon prend une grandeur mystérieuse et nous semble se refermer sur un monde qu'on ne reverra plus ; cependant nous avançons, et c'est bientôt nous-même qui sommes à l'horizon pour les générations qui sont derrière nous ; cependant l'horizon recule, et le monde, qui semblait fini, recommence. « J'ai même pu voir, quand j'étais jeune fille, ajouta duchesse de Guermantes, duchesse de Guermantes de Dino. Dame, vous savez que je n'ai plus vingt-cinq ans. » Ces derniers mots me fâchèrent. Elle ne devrait pas dire cela, ce serait bon pour une vieille femme. « Quant à vous, reprit-elle, vous êtes toujours le même, vous n'avez pour ainsi dire pas changé », me dit duchesse de Guermantes, et cela me fit presque plus de peine que si elle m'avait parlé d'un changement, car cela prouvait, puisqu'il était extraordinaire qu'il s'en fût si peu produit, que bien du temps s'était écoulé. « Ami, me dit-elle, vous êtes étonnant, vous restez toujours jeune », expression si mélancolique puisqu'elle n'a de sens que si nous sommes, en fait sinon d'apparence, devenus vieux. Et elle me donna le dernier coup en ajoutant : « J'ai toujours regretté que vous ne vous soyez pas marié. Au fond, qui sait, c'est peut-être plus heureux. Vous auriez été d'âge à avoir des fils à la guerre, et s'ils avaient été tués, comme l'a été ce pauvre M. de Marsantes (je pense encore souvent à lui), sensible comme vous êtes, vous ne leur auriez pas survécu. » Et je pus me voir, comme dans la première glace véridique que j'eusse rencontrée dans les yeux de vieillards restés jeunes, à leur avis, comme je le croyais moi-même de moi, et qui, quand je me citais à eux, pour entendre un démenti, comme exemple de vieux, n'avaient pas dans leurs regards, qui me voyaient tel qu'ils ne se voyaient pas eux-mêmes et tel que je les voyais, une seule protestation. Car nous ne voyions pas notre propre aspect, nos propres âges, mais chacun, comme un miroir opposé, voyait celui de l'autre. Et sans doute, à découvrir qu'ils ont vieilli, bien des gens eussent été moins tristes que moi. Mais d'abord il en est de la vieillesse comme de la mort, quelques-uns les affrontent avec indifférence, non pas parce qu'ils ont plus de courage que les autres, mais parce qu'ils ont moins d'imagination. Puis un homme qui depuis son enfance vise une même idée, auquel sa paresse même et jusqu'à son état de santé, en lui faisant remettre sans cesse les réalisations, annule chaque soir le jour écoulé et perdu, si bien que la maladie qui hâte le vieillissement de son corps retarde celui de son esprit, est plus surpris et plus bouleversé de voir qu'il n'a cessé de vivre dans le Temps, que celui qui vit peu en soi-même, se règle sur le calendrier, et ne découvre pas d'un seul coup le total des années dont il a poursuivi quotidiennement l'addition. Mais une raison plus grave expliquait mon angoisse ; je découvrais cette action destructrice du Temps au moment même où je voulais entreprendre de rendre claires, d'intellectualiser dans une oeuvre d'art, des réalités extra-temporelles.

### Passage

Chez certains êtres le remplacement successif, mais accompli en mon absence, de chaque cellule par d'autres, avait amené un changement si complet, une si entière métamorphose que j'aurais pu dîner cent fois en face d'eux dans un restaurant sans me douter plus que je les avais connus autrefois que je n'aurais pu deviner la royauté d'un souverain incognito ou le vice d'un inconnu. La comparaison devient même insuffisante pour le cas où j'entendais leur nom, car on peut admettre qu'un inconnu assis en face de vous soit criminel ou roi, tandis qu'eux, je les avais connus, ou plutôt j'avais connu des personnes portant le même nom, mais si différentes que je ne pouvais croire que ce fussent les mêmes. Pourtant, comme j'aurais fait en partant de l'idée de souveraineté ou de vice qui ne tarde pas à donner à l'inconnu (avec qui on aurait fait si aisément, quand on avait encore les yeux bandés, la gaffe d'être insolent ou aimable), dans les mêmes traits de qui on discerne maintenant quelque chose de distingué ou de suspect, je m'appliquais à introduire dans le visage de l'inconnue, entièrement inconnue, l'idée qu'elle était Mme Sazerat, et je finissais par rétablir le sens autrefois connu de ce visage, mais qui serait resté vraiment aliéné pour moi, entièrement celui d'une autre femme ayant autant perdu tous les attributs humains que j'avais connus, qu'un homme devenu singe, si le nom et l'affirmation de l'identité ne m'avaient mis, malgré ce que le problème avait d'ardu, sur la voie de la solution. Parfois pourtant, l'ancienne image renaissait assez précise pour que je puisse essayer une confrontation ; et comme un témoin mis en présence d'un inculpé qu'il a vu, j'étais forcé, tant la différence était grande, de dire : « Non... je ne le reconnais pas. »

Une jeune femme me dit : « Voulez-vous que nous allions dîner tous les deux au restaurant ? » Comme je répondais : « Si vous ne trouvez pas compromettant de venir dîner seule avec un jeune homme », j'entendis que tout le monde autour de moi riait, et je m'empressai d'ajouter : « ou plutôt avec un vieil homme ». Je sentais que la phrase qui avait fait rire était de celles qu'aurait pu, en parlant de moi, dire ma mère, ma mère pour qui j'étais toujours un enfant. Or je m'apercevais que je me plaçais pour me juger au même point de vue qu'elle. Si j'avais fini par enregistrer comme elle certains changements qui s'étaient faits depuis ma première enfance, c'était tout de même des changements maintenant très anciens. J'en étais resté à celui qui faisait qu'on avait dit un temps, presque en prenant de l'avance sur le fait : « C'est maintenant presque un grand jeune homme. » Je le pensais encore, mais cette fois avec un immense retard. Je ne m'apercevais pas combien j'avais changé. Mais, au fait, eux, qui venaient de rire aux éclats, à quoi s'en apercevaient-ils ? Je n'avais pas un cheveu gris, ma moustache était noire. J'aurais voulu pouvoir leur demander à quoi se révélait l'évidence de la terrible chose. Et maintenant je comprenais ce qu'était la vieillesse – la vieillesse qui, de toutes les réalités, est peut-être celle dont nous gardons le plus longtemps dans la vie une notion purement abstraite, regardant les calendriers, datant nos lettres, voyant se marier nos amis, les enfants de nos amis, sans comprendre, soit par peur, soit par paresse, ce que cela signifie, jusqu'au jour où nous apercevons une silhouette inconnue, comme celle de M. d'Argencourt, laquelle nous apprend que nous vivons dans un nouveau monde ; jusqu'au jour où le petit-fils d'une de nos amies, jeune homme qu'instinctivement nous traiterions en camarade, sourit comme si nous nous moquions de lui, nous qui lui sommes apparu comme un grand-père ; je comprenais ce que signifiaient la mort, l'amour, les joies de l'esprit, l'utilité de la douleur, la vocation. Car si les noms avaient perdu pour moi de leur individualité, les mots me découvraient tout leur sens. La beauté des images est logée à l'arrière des choses, celle des idées à l'avant. De sorte que la première cesse de nous émerveiller quand on les a atteintes, mais qu'on ne comprend la seconde que quand on les a dépassées.

Or, à toutes ces idées, la cruelle découverte que je venais de faire relativement au Temps qui s'était écoulé ne pourrait que s'ajouter et me servir en ce qui concernait la matière même de mon livre. Puisque j'avais décidé qu'elle ne pouvait être uniquement constituée par les impressions véritablement pleines, celles qui sont en dehors du Temps, parmi les vérités avec lesquelles je comptais les sertir, celles qui se rapportent au Temps, au Temps dans lequel baignent et s'altèrent les hommes, les sociétés, les nations, tiendraient une place importante. Je n'aurais pas soin seulement de faire une place à ces altérations que subit l'aspect des êtres et dont j'avais de nouveaux exemples à chaque minute, car tout en songeant à mon oeuvre, assez définitivement mise en marche pour ne pas se laisser arrêter par des distractions passagères, je continuais à dire bonjour aux gens que je connaissais et à causer avec eux. Le vieillissement, d'ailleurs, ne se marquait pas pour tous d'une manière analogue. Je vis quelqu'un qui demandait mon nom, on me dit que c'était M. de Cambremer. Et alors, pour me montrer qu'il m'avait reconnu : « Est-ce que vous avez toujours vos étouffements ? » me demanda-t-il, et sur ma réponse affirmative : « Vous voyez que ça n'empêche pas la longévité », me dit-il, comme si j'étais décidément centenaire. Je lui parlais les yeux attachés sur deux ou trois traits que je pouvais faire rentrer par la pensée dans cette synthèse, pour le reste toute différente, de mes souvenirs, que j'appelais sa personne. Mais un instant il tourna à demi la tête. Et alors je vis qu'il était rendu méconnaissable par l'adjonction d'énormes poches rouges aux joues qui l'empêchaient d'ouvrir complètement la bouche et les yeux, si bien que je restais hébété, n'osant regarder cette sorte d'anthrax dont il me semblait plus convenable qu'il me parlât le premier. Mais comme, en malade courageux, il n'y faisait pas allusion et riait, j'avais peur d'avoir l'air de manquer de coeur en ne lui demandant pas, de tact en lui demandant ce qu'il avait. Mais « ils ne vous viennent pas plus rarement avec l'âge ? » me demanda-t-il, en continuant à parler de mes étouffements. Je lui dis que non. « Ah ! pourtant, ma soeur en a sensiblement moins qu'autrefois », me dit-il, d'un ton de contradiction comme si cela ne pouvait pas être autrement pour moi que pour sa soeur, et comme si l'âge était un de ces remèdes dont il n'admettait pas, quand ils avaient fait du bien à Mme de Gaucourt, qu'ils ne me fussent pas salutaires. Mme de Cambremer-Legrandin s'étant approchée, j'avais de plus en plus peur de paraître insensible en ne déplorant pas ce que je remarquais sur la figure de son mari et je n'osais pas cependant parler de ça le premier. « Vous êtes content de le voir ? me dit-elle. – Il va bien ? répliquai-je sur un ton incertain. – Mais comme vous voyez. » Elle ne s'était pas aperçue de ce mal qui offusquait ma vue et qui n'était autre qu'un des masques du Temps que celui-ci avait appliqué à la figure du marquis, mais peu à peu, et en l'épaississant si progressivement que la marquise n'en avait rien vu. Quand M. de Cambremer eut fini ses questions sur mes étouffements, ce fut mon tour de m'informer tout bas auprès de quelqu'un si la mère du marquis vivait encore. Elle vivait. Dans l'appréciation du temps écoulé, il n'y a que le premier pas qui coûte. On éprouve d'abord beaucoup de peine à se figurer que tant de temps ait passé et ensuite qu'il n'en ait pas passé davantage. On n'avait jamais songé que le XIIIe siècle fût si loin, et après on a peine à croire qu'il puisse subsister encore des églises du XIIIe siècle, lesquelles pourtant sont innombrables en France. En quelques instants s'était fait en moi ce travail plus lent qui se fait chez ceux qui, ayant eu peine à comprendre qu'une personne qu'ils ont connue jeune ait soixante ans, en ont plus encore, quinze ans après, à apprendre qu'elle vit encore et n'a pas plus de soixante-quinze ans. Je demandai à M. de Cambremer comment allait sa mère. « Elle est toujours admirable », me dit-il, usant d'un adjectif qui, par opposition aux tribus où on traite sans pitié les parents âgés, s'applique dans certaines familles aux vieillards chez qui l'usage des facultés les plus matérielles, comme d'entendre, d'aller à pied à la messe, et de supporter avec insensibilité les deuils, s'empreint, aux yeux de leurs enfants, d'une extraordinaire beauté morale.

Si certaines femmes avouaient leur vieillesse en se fardant, elle apparaissait, au contraire, par l'absence de fard chez certains hommes sur le visage desquels je ne l'avais jamais expressément remarquée, et qui tout de même me semblaient bien changés depuis que, découragés de chercher à plaire, ils en avaient cessé l'usage. Parmi eux était Legrandin. La suppression du rose, que je n'avais jamais soupçonné artificiel, de ses lèvres et de ses joues donnait à sa figure l'apparence grisâtre et à ses traits allongés et mornes la précision sculpturale et lapidaire de ceux d'un dieu égyptien. Un dieu ! un revenant plutôt. Il avait perdu non seulement le courage de se peindre, mais de sourire, de faire briller son regard, de tenir des discours ingénieux. On s'étonnait de le voir si pâle, abattu, ne prononçant que de rares paroles qui avaient l'insignifiance de celles que disent les morts qu'on évoque. On se demandait quelle cause l'empêchait d'être vif, éloquent, charmant, comme on se le demande devant « le double » insignifiant d'un homme brillant de son vivant et auquel un spirite pose pourtant des questions qui prêteraient aux développements charmeurs. Et on se disait que cette cause qui avait substitué au Legrandin coloré et rapide un pâle et triste fantôme de Legrandin, c'était la vieillesse. Chez certains même les cheveux n'avaient pas blanchi. Ainsi je reconnus, quand il vint dire un mot à son maître, le vieux valet de chambre du prince de Guermantes. Les poils bourrus qui hérissaient ses joues tout autant que son crâne étaient restés d'un roux tirant sur le rose et on ne pouvait le soupçonner de se teindre comme la Mme de Guermantes. Mais il n'en paraissait pas moins vieux. On sentait seulement qu'il existe chez les hommes comme, dans le règne végétal, les mousses, les lichens et tant d'autres, des espèces qui ne changent pas à l'approche de l'hiver.

Chez d'autres invités, dont le visage était intact, l'âge se marquait autrement ; ils semblaient seulement embarrassés quand ils avaient à marcher ; on croyait d'abord qu'ils avaient mal aux jambes, et ce n'est qu'ensuite qu'on comprenait que la vieillesse leur avait attaché ses semelles de plomb. Elle en embellissait d'autres, comme le prince d'Agrigente. À cet homme long, mince, au regard terne, aux cheveux qui semblaient devoir rester éternellement rougeâtres, avait succédé, par une métamorphose analogue à celle des insectes, un vieillard chez qui les cheveux rouges, trop longtemps vus, avaient été, comme un tapis de table qui a trop servi, remplacé par des cheveux blancs. Sa poitrine avait pris une corpulence inconnue, robuste, presque guerrière, et qui avait dû nécessiter un véritable éclatement de la frêle chrysalide que j'avais connue ; une gravité consciente d'elle-même baignait les yeux, où elle était teintée d'une bienveillance nouvelle qui s'inclinait vers chacun. Et comme, malgré tout, une certaine ressemblance subsistait entre le puissant prince actuel et le portrait que gardait mon souvenir, j'admirais la force de renouvellement original du temps qui, tout en respectant l'unité de l'être et les lois de la vie, sait changer ainsi le décor et introduire de hardis contrastes dans deux aspects successifs d'un même personnage, car, beaucoup de ces gens, on les identifiait immédiatement, mais comme d'assez mauvais portraits d'eux-mêmes réunis dans l'exposition où un artiste inexact et malveillant durcit les traits de l'un, enlève la fraîcheur du teint ou la légèreté de la taille à celle-ci, assombrit le regard de tel autre. Comparant ces images avec celles que j'avais sous les yeux de ma mémoire, j'aimais moins celles qui m'étaient montrées en dernier lieu. Comme souvent on trouve moins bonne et on refuse une des photographies entre lesquelles un ami vous a prié de choisir. À chaque personne et devant l'image qu'elle me montrait d'elle-même j'aurais voulu dire : « Non, pas celle-ci, vous êtes moins bien, ce n'est pas vous. » Je n'aurais pas osé ajouter : « Au lieu de votre beau nez droit on vous a fait le nez crochu de votre père que je ne vous ai jamais connu. » En effet, c'était un nez nouveau et familial. Bref, l'artiste le Temps avait « rendu » tous ces modèles de telle façon qu'ils étaient reconnaissables, mais ils n'étaient pas ressemblants, non parce qu'il les avait flattés, mais parce qu'il les avait vieillis. Cet artiste-là, du reste, travaille fort lentement. Ainsi cette réplique du visage d'Odette, dont, le jour où j'avais pour la première fois vu Bergotte, j'avais aperçu l'esquisse à peine ébauchée dans le visage de Gilberte, le temps l'avait enfin poussée jusqu'à la plus parfaite ressemblance, comme on le verra tout à l'heure, pareil à ces peintres qui gardent longtemps une oeuvre et la complètent année par année.
