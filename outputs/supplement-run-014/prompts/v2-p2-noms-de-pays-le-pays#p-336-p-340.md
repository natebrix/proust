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
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.69,
      "evidence": "« le suprême bonheur eût été de rencontrer Legrandin avec lequel je venais de causer en rêve »; puis le narrateur interroge sa grand-mère sur « la famille Legrandin ».",
      "explanation": "The narrator expresses a strong desire to see Legrandin and prolongs this interest by questioning about him, elevating him locally. But this admiration is explicitly attributed to a dreamlike state and difficult digestion, which ironizes him and relativizes its significance."
    }
  ],
  "status_effects": [
    {
      "character": "Legrandin",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.66,
      "explanation": "Legrandin gains local esteem as the object of a 'supreme happiness' in meeting him, despite the dreamlike and ironized framework of this valorization."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-336-p-340"
}

### Candidate characters

[
  "Françoise",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

En rentrant à Balbec, de telle de ces inconnues à qui il m'avait présenté je me redisais sans m'arrêter une seconde et pourtant sans presque m'en apercevoir : « Quelle femme délicieuse ! » comme on chante un refrain. Certes, ces paroles étaient plutôt dictées par des dispositions nerveuses que par un jugement durable. Il n'en est pas moins vrai que si j'eusse eu mille francs sur moi et qu'il y eût encore des bijoutiers d'ouverts à cette heure-là, j'eusse acheté une bague à l'inconnue. Quand les heures de notre vie se déroulent ainsi que sur des plans trop différents, on se trouve donner trop de soi pour des personnes diverses qui le lendemain vous semblent sans intérêt. Mais on se sent responsable de ce qu'on leur a dit la veille et on veut y faire honneur.

### Passage

Comme ces soirs-là je rentrais plus tard, je retrouvais avec plaisir dans ma chambre qui n'était plus hostile le lit où, le jour de mon arrivée, j'avais cru qu'il me serait toujours impossible de me reposer et où maintenant mes membres si las cherchaient un soutien ; de sorte que successivement mes cuisses, mes hanches, mes épaules tâchaient d'adhérer en tous leurs points aux draps qui enveloppaient le matelas, comme si ma fatigue, pareille à un sculpteur, avait voulu prendre un moulage total d'un corps humain. Mais je ne pouvais m'endormir, je sentais approcher le matin ; le calme, la bonne santé n'étaient plus en moi. Dans ma détresse, il me semblait que jamais je ne les retrouverais plus. Il m'eût fallu dormir longtemps pour les rejoindre. Or, me fussé-je assoupi, que de toutes façons je serais réveillé deux heures après par le concert symphonique. Tout à coup je m'endormais, je tombais dans ce sommeil lourd où se dévoilent pour nous le retour à la jeunesse, la reprise des années passées, des sentiments perdus, la désincarnation, la transmigration des âmes, l'évocation des morts, les illusions de la folie, la régression vers les règnes les plus élémentaires de la nature (car on dit que nous voyons souvent des animaux en rêve, mais on oublie presque toujours que nous y sommes nous-mêmes un animal privé de cette raison qui projette sur les choses une clarté de certitude ; nous n'y offrons au contraire, au spectacle de la vie, qu'une vision douteuse et à chaque minute anéantie par l'oubli, la réalité précédente s'évanouissant devant celle qui lui succède comme une projection de lanterne magique devant la suivante quand on a changé le verre), tous ces mystères que nous croyons ne pas connaître et auxquels nous sommes en réalité initiés presque toutes les nuits ainsi qu'à l'autre grand mystère de l'anéantissement et de la résurrection. Rendue plus vagabonde par la digestion difficile du dîner de Rivebelle, l'illumination successive et errante de zones assombries de mon passé faisait de moi un être dont le suprême bonheur eût été de rencontrer Legrandin avec lequel je venais de causer en rêve.

Puis, même ma propre vie m'était entièrement cachée par un décor nouveau, comme celui planté tout au bord du plateau et devant lequel pendant que, derrière, on procède aux changements de tableaux, des acteurs donnent un divertissement. Celui où je tenais alors mon rôle était dans le goût des contes orientaux, je n'y savais rien de mon passé ni de moi-même, à cause de cet extrême rapprochement d'un décor interposé ; je n'étais qu'un personnage qui recevait la bastonnade et subissais des châtiments variés pour une faute que je n'apercevais pas mais qui était d'avoir bu trop de porto. Tout à coup je m'éveillais, je m'apercevais qu'à la faveur d'un long sommeil, je n'avais pas entendu le concert symphonique. C'était déjà l'après-midi ; je m'en assurais à ma montre, après quelques efforts pour me redresser, efforts infructueux d'abord et interrompus par des chutes sur l'oreiller, mais de ces chutes courtes qui suivent le sommeil comme les autres ivresses, que ce soit le vin qui les procure, ou une convalescence ; du reste avant même d'avoir regardé l'heure j'étais certain que midi était passé. Hier soir, je n'étais plus qu'un être vidé, sans poids, et comme il faut avoir été couché pour être capable de s'asseoir et avoir dormi pour l'être de se taire, je ne pouvais cesser de remuer ni de parler, je n'avais plus de consistance, de centre de gravité, j'étais lancé, il me semblait que j'aurais pu continuer ma morne course jusque dans la lune. Or, si en dormant mes yeux n'avaient pas vu l'heure, mon corps avait su la calculer, il avait mesuré le temps non pas sur un cadran superficiellement figuré, mais par la pesée progressive de toutes mes forces refaites que comme une puissante horloge il avait cran par cran laissé descendre de mon cerveau dans le reste de mon corps où elles entassaient maintenant jusque au-dessus de mes genoux l'abondance intacte de leurs provisions. S'il est vrai que la mer ait été autrefois notre milieu vital où il faille replonger notre sang pour retrouver nos forces, il en est de même de l'oubli, du néant mental ; on semble alors absent du temps pendant quelques heures ; mais les forces qui se sont rangées pendant ce temps-là sans être dépensées le mesurent par leur quantité aussi exactement que les poids de l'horloge ou les croulants monticules du sablier. On ne sort, d'ailleurs, pas plus aisément d'un tel sommeil que de la veille prolongée, tant toutes choses tendent à durer, et s'il est vrai que certains narcotiques font dormir, dormir longtemps est un narcotique plus puissant encore, après lequel on a bien de la peine à se réveiller. Pareil à un matelot qui voit bien le quai où amarrer sa barque, secouée cependant encore par les flots, j'avais bien l'idée de regarder l'heure et de me lever, mais mon corps était à tout instant rejeté dans le sommeil ; l'atterrissage était difficile, et avant de me mettre debout pour atteindre ma montre et confronter son heure avec celle qu'indiquait la richesse de matériaux dont disposaient mes jambes rompues, je retombais encore deux ou trois fois sur mon oreiller.

Enfin je voyais clairement : « deux heures de l'après-midi ! » je sonnais, mais aussitôt je rentrais dans un sommeil qui cette fois devait être infiniment plus long si j'en jugeais par le repos et la vision d'une immense nuit dépassée, que je trouvais au réveil. Pourtant comme celui-ci était causé par l'entrée de Françoise, entrée qu'avait elle-même motivée mon coup de sonnette, ce nouveau sommeil, qui me paraissait avoir dû être plus long que l'autre et avait amené en moi tant de bien-être et d'oubli, n'avait duré qu'une demi-minute.

Ma grand-mère ouvrait la porte de ma chambre, je lui posais mille questions sur la famille Legrandin.

Ce n'est pas assez dire que j'avais rejoint le calme et la santé, car c'était plus qu'une simple distance qui les avait la veille séparés de moi, j'avais eu toute la nuit à lutter contre un flot contraire, et puis je ne me retrouvais pas seulement auprès d'eux, ils étaient rentrés en moi. À des points précis et encore un peu douloureux de ma tête vide et qui serait un jour brisée, laissant mes idées s'échapper à jamais, celles-ci avaient une fois encore repris leur place, et retrouvé cette existence dont hélas ! jusqu'ici elles n'avaient pas su profiter.
