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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.95,
      "evidence": "« brusquement précipité dans ce nouveau cercle de l'enfer »; « sa cruelle jalousie le replaçait … pour le faire frapper par l’aveu d’Odette »; « Son âme les charriait… comme des cadavres. Et elle en était empoisonnée. »",
      "explanation": "The narrator presents Swann as plunged into obsessive jealousy and recurring torment by Odette’s revelations, emphasizing helpless suffering and emotional collapse."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.95,
      "explanation": "He is rendered powerless and anguished by jealousy, repeatedly re-wounded by Odette’s words."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-566-p-570"
}

### Candidate characters

[
  "M. Verdurin",
  "Odette",
  "duchesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

– Tu es un misérable, tu te plais à me torturer, à me faire faire des mensonges que je dis afin que tu me laisses tranquille.

### Passage

Ce second coup porté à Swann était plus atroce encore que le premier. Jamais il n'avait supposé que ce fût une chose aussi récente, cachée à ses yeux qui n'avaient pas su la découvrir, non dans un passé qu'il n'avait pas connu, mais dans des soirs qu'il se rappelait si bien, qu'il avait vécus avec Odette, qu'il avait cru connus si bien par lui et qui maintenant prenaient rétrospectivement quelque chose de fourbe et d'atroce ; au milieu d'eux tout d'un coup se creusait cette ouverture béante, ce moment dans l'île du Bois. Odette sans être intelligente avait le charme du naturel. Elle avait raconté, elle avait mimé cette scène avec tant de simplicité que Swann haletant voyait tout : le bâillement d'Odette, le petit rocher. Il l'entendait répondre – gaiement, hélas ! : « Cette blague ! » Il sentait qu'elle ne dirait rien de plus ce soir, qu'il n'y avait aucune révélation nouvelle à attendre en ce moment ; elle se taisait ; il lui dit :

– Mon pauvre chéri, pardonne-moi, je sens que je te fais de la peine, c'est fini, je n'y pense plus.

Mais elle vit que ses yeux restaient fixés sur les choses qu'il ne savait pas et sur ce passé de leur amour, monotone et doux dans sa mémoire parce qu'il était vague, et que déchirait maintenant comme une blessure cette minute dans l'île du Bois, au clair de lune, après le dîner chez la princesse des Laumes. Mais il avait tellement pris l'habitude de trouver la vie intéressante – d'admirer les curieuses découvertes qu'on peut y faire – que tout en souffrant au point de croire qu'il ne pourrait pas supporter longtemps une pareille douleur, il se disait : « La vie est vraiment étonnante et réserve de belles surprises ; en somme le vice est quelque chose de plus répandu qu'on ne croit. Voilà une femme en qui j'avais confiance, qui a l'air si simple, si honnête, en tous cas, si même elle était légère, qui semblait bien normale et saine dans ses goûts : sur une dénonciation invraisemblable, je l'interroge et le peu qu'elle m'avoue révèle bien plus que ce qu'on eût pu soupçonner. » Mais il ne pouvait pas se borner à ces remarques désintéressées. Il cherchait à apprécier exactement la valeur de ce qu'elle lui avait raconté, afin de savoir s'il devait conclure que ces choses, elle les avait faites souvent, qu'elles se renouvelleraient. Il se répétait ces mots qu'elle avait dits : « Je voyais bien où elle voulait en venir », « Deux ou trois fois », « Cette blague ! », mais ils ne reparaissaient pas désarmés dans la mémoire de Swann, chacun d'eux tenait son couteau et lui en portait un nouveau coup. Pendant bien longtemps, comme un malade ne peut s'empêcher d'essayer à toute minute de faire le mouvement qui lui est douloureux, il se redisait ces mots : « Je suis bien ici », « Cette blague ! », mais la souffrance était si forte qu'il était obligé de s'arrêter. Il s'émerveillait que des actes que toujours il avait jugés si légèrement, si gaiement, maintenant fussent devenus pour lui graves comme une maladie dont on peut mourir. Il connaissait bien des femmes à qui il eût pu demander de surveiller Odette. Mais comment espérer qu'elles se placeraient au même point de vue que lui et ne resteraient pas à celui qui avait été si longtemps le sien, qui avait toujours guidé sa vie voluptueuse, ne lui diraient pas en riant : « Vilain jaloux qui veut priver les autres d'un plaisir. » Par quelle trappe soudainement abaissée (lui qui n'avait eu autrefois de son amour pour Odette que des plaisirs délicats) avait-il été brusquement précipité dans ce nouveau cercle de l'enfer d'où il n'apercevait pas comment il pourrait jamais sortir. Pauvre Odette ! il ne lui en voulait pas. Elle n'était qu'à demi coupable. Ne disait-on pas que c'était par sa propre mère qu'elle avait été livrée, presque enfant, à Nice, à un riche Anglais. Mais quelle vérité douloureuse prenait pour lui ces lignes du Journal d'un Poète d'Alfred de Vigny qu'il avait lues avec indifférence autrefois : « Quand on se sent pris d'amour pour une femme, on devrait se dire : Comment est-elle entourée ? Quelle a été sa vie ? Tout le bonheur de la vie est appuyé là-dessus. » Swann s'étonnait que de simples phrases épelées par sa pensée, comme « Cette blague ! », « Je voyais bien où elle voulait en venir » pussent lui faire si mal. Mais il comprenait que ce qu'il croyait de simples phrases n'était que les pièces de l'armature entre lesquelles tenait, pouvait lui être rendue, la souffrance qu'il avait éprouvée pendant le récit d'Odette. Car c'était bien cette souffrance-là qu'il éprouvait de nouveau. Il avait beau savoir maintenant – même, il eut beau, le temps passant, avoir un peu oublié, avoir pardonné – au moment où il se redisait ses mots, la souffrance ancienne le refaisait tel qu'il était avant qu'Odette ne parlât : ignorant, confiant ; sa cruelle jalousie le replaçait pour le faire frapper par l'aveu d'Odette dans la position de quelqu'un qui ne sait pas encore, et au bout de plusieurs mois cette vieille histoire le bouleversait toujours comme une révélation. Il admirait la terrible puissance recréatrice de sa mémoire. Ce n'est que de l'affaiblissement de cette génératrice dont la fécondité diminue avec l'âge qu'il pouvait espérer un apaisement à sa torture. Mais quand paraissait un peu épuisé le pouvoir qu'avait de le faire souffrir un des mots prononcés par Odette, alors un de ceux sur lesquels l'esprit de Swann s'était moins arrêté jusque-là, un mot presque nouveau venait relayer les autres et le frappait avec une vigueur intacte. La mémoire du soir où il avait dîné chez la princesse des Laumes lui était douloureuse, mais ce n'était que le centre de son mal. Celui-ci irradiait confusément à l'entour dans tous les jours avoisinants. Et à quelque point d'elle qu'il voulût toucher dans ses souvenirs, c'est la saison tout entière où les Verdurin avaient si souvent dîné dans l'île du Bois qui lui faisait mal. Si mal, que peu à peu les curiosités qu'excitait en lui sa jalousie furent neutralisées par la peur des tortures nouvelles qu'il s'infligerait en les satisfaisant. Il se rendait compte que toute la période de la vie d'Odette écoulée avant qu'elle ne le rencontrât, période qu'il n'avait jamais cherché à se représenter, n'était pas l'étendue abstraite qu'il voyait vaguement, mais avait été faite d'années particulières, remplie d'incidents concrets. Mais en les apprenant, il craignait que ce passé incolore, fluide et supportable, ne prît un corps tangible et immonde, un visage individuel et diabolique. Et il continuait à ne pas chercher à le concevoir non plus par paresse de penser, mais par peur de souffrir. Il espérait qu'un jour il finirait par pouvoir entendre le nom de l'île du Bois, de la princesse des Laumes, sans ressentir le déchirement ancien, et trouvait imprudent de provoquer Odette à lui fournir de nouvelles paroles, le nom d'endroits, de circonstances différentes qui, son mal à peine calmé, le feraient renaître sous une autre forme.

Mais souvent les choses qu'il ne connaissait pas, qu'il redoutait maintenant de connaître, c'est Odette elle-même qui les lui révélait spontanément, et sans s'en rendre compte ; en effet l'écart que le vice mettait entre la vie réelle d'Odette et la vie relativement innocente que Swann avait cru, et bien souvent croyait encore, que menait sa maîtresse, cet écart, Odette en ignorait l'étendue : un être vicieux, affectant toujours la même vertu devant les êtres de qui il ne veut pas que soient soupçonnés ses vices, n'a pas de contrôle pour se rendre compte combien ceux-ci, dont la croissance continue est insensible pour lui-même, l'entraînent peu à peu loin des façons de vivre normales. Dans leur cohabitation, au sein de l'esprit d'Odette, avec le souvenir des actions qu'elle cachait à Swann, d'autres peu à peu en recevaient le reflet, étaient contagionnées par elles, sans qu'elle pût leur trouver rien d'étrange, sans qu'elles détonassent dans le milieu particulier où elle les faisait vivre en elle ; mais si elle les racontait à Swann, il était épouvanté par la révélation de l'ambiance qu'elles trahissaient. Un jour il cherchait, sans blesser Odette, à lui demander si elle n'avait jamais été chez des entremetteuses. À vrai dire il était convaincu que non ; la lecture de la lettre anonyme en avait introduit la supposition dans son intelligence, mais d'une façon mécanique ; elle n'y avait rencontré aucune créance, mais en fait y était restée, et Swann, pour être débarrassé de la présence purement matérielle mais pourtant gênante du soupçon, souhaitait qu'Odette l'extirpât. « Oh ! non ! Ce n'est pas que je ne sois pas persécutée pour cela, ajouta-t-elle, en dévoilant dans un sourire une satisfaction de vanité qu'elle ne s'apercevait plus ne pas pouvoir paraître légitime à Swann. Il y en a une qui est encore restée plus de deux heures hier à m'attendre, elle me proposait n'importe quel prix. Il paraît qu'il y a un ambassadeur qui lui a dit : « Je me tue si vous ne me l'amenez pas. » On lui a dit que j'étais sortie, j'ai fini par aller moi-même lui parler pour qu'elle s'en aille. J'aurais voulu que tu voies comme je l'ai reçue, ma femme de chambre qui m'entendait de la pièce voisine m'a dit que je criais à tue-tête : « Mais puisque je vous dis que je ne veux pas ! C'est une idée comme ça, ça ne me plaît pas. Je pense que je suis libre de faire ce que je veux, tout de même ! Si j'avais besoin d'argent, je comprends... » Le concierge a ordre de ne plus la laisser entrer, il dira que je suis à la campagne. Ah ! j'aurais voulu que tu sois caché quelque part. Je crois que tu aurais été content, mon chéri. Elle a du bon, tout de même, tu vois, ta petite Odette, quoiqu'on la trouve si détestable. »

D'ailleurs ses aveux même, quand elle lui en faisait, de fautes qu'elle le supposait avoir découvertes, servaient plutôt pour Swann de point de départ à de nouveaux doutes qu'ils ne mettaient un terme aux anciens. Car ils n'étaient jamais exactement proportionnés à ceux-ci. Odette avait eu beau retrancher de sa confession tout l'essentiel, il restait dans l'accessoire quelque chose que Swann n'avait jamais imaginé, qui l'accablait de sa nouveauté et allait lui permettre de changer les termes du problème de sa jalousie. Et ces aveux il ne pouvait plus les oublier. Son âme les charriait, les rejetait, les berçait, comme des cadavres. Et elle en était empoisonnée.
