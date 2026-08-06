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
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.84,
      "evidence": "« ...maintenant tous pensaient… : “Il n'est pas positivement laid si vous voulez, mais il est ridicule ; ce monocle, ce toupet, ce sourire !”, réalisant… la démarcation… entre une tête d'amant de coeur et une tête de cocu. »",
      "explanation": "The social gaze reclassifies Swann from 'chic' to 'ridicule' and 'cocu,' lowering his image; the narrator notes this as a suggestible, programmatic shift in perception."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "He is locally diminished both in Odette's personal treatment and in the broader social view that now reads him as ridiculous/cuckolded."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-420-p-425"
}

### Candidate characters

[
  "Odette",
  "comte de Forcheville"
]

### Prior local context (optional)

Comme Odette ne lui donnait aucun renseignement sur ces choses si importantes qui l'occupaient tant chaque jour (bien qu'il eût assez vécu pour savoir qu'il n'y en a jamais d'autres que les plaisirs), il ne pouvait pas chercher longtemps de suite à les imaginer, son cerveau fonctionnait à vide ; alors il passait son doigt sur ses paupières fatiguées comme il aurait essuyé le verre de son lorgnon, et cessait entièrement de penser. Il surnageait pourtant à cet inconnu certaines occupations qui réapparaissaient de temps en temps, vaguement rattachées par elle à quelque obligation envers des parents éloignés ou des amis d'autrefois, qui, parce qu'ils étaient les seuls qu'elle lui citait souvent comme l'empêchant de le voir, paraissaient à Swann former le cadre fixe, nécessaire, de la vie d'Odette. À cause du ton dont elle lui disait de temps à autre « le jour où je vais avec mon amie à l'Hippodrome », si, s'étant senti malade et ayant pensé : « peut-être Odette voudrait bien passer chez moi », il se rappelait brusquement que c'était justement ce jour-là, il se disait : « Ah ! non, ce n'est pas la peine de lui demander de venir, j'aurais dû y penser plus tôt, c'est le jour où elle va avec son amie à l'Hippodrome. Réservons-nous pour ce qui est possible ; c'est inutile de s'user à proposer des choses inacceptables et refusées d'avance. » Et ce devoir qui incombait à Odette d'aller à l'Hippodrome et devant lequel Swann s'inclinait ainsi ne lui paraissait pas seulement inéluctable ; mais ce caractère de nécessité dont il était empreint semblait rendre plausible et légitime tout ce qui de près ou de loin se rapportait à lui. Si Odette dans la rue ayant reçu d'un passant un salut qui avait éveillé la jalousie de Swann, elle répondait aux questions de celui-ci en rattachant l'existence de l'inconnu à un des deux ou trois grands devoirs dont elle lui parlait, si, par exemple, elle disait : « C'est un monsieur qui était dans la loge de mon amie avec qui je vais à l'Hippodrome », cette explication calmait les soupçons de Swann, qui en effet trouvait inévitable que l'amie eût d'autre invités qu'Odette dans sa loge à l'Hippodrome, mais n'avait jamais cherché ou réussi à se les figurer. Ah ! comme il eût aimé la connaître, l'amie qui allait à l'Hippodrome, et qu'elle l'y emmenât avec Odette ! Comme il aurait donné toutes ses relations pour n'importe quelle personne qu'avait l'habitude de voir Odette, fût-ce une manucure ou une demoiselle de magasin. Il eût fait pour elles plus de frais que pour des reines. Ne lui auraient-elles pas fourni, dans ce qu'elles contenaient de la vie d'Odette, le seul calmant efficace pour ses souffrances ? Comme il aurait couru avec joie passer les journées chez telle de ces petites gens avec lesquelles Odette gardait des relations, soit par intérêt, soit par simplicité véritable. Comme il eût volontiers élu domicile à jamais au cinquième étage de telle maison sordide et enviée où Odette ne l'emmenait pas, et où, s'il y avait habité avec la petite couturière retirée dont il eût volontiers fait semblant d'être l'amant, il aurait presque chaque jour reçu sa visite. Dans ces quartiers presque populaires, quelle existence modeste, abjecte, mais douce, mais nourrie de calme et de bonheur, il eût accepté de vivre indéfiniment.

### Passage

Il arrivait encore parfois, quand, ayant rencontré Swann, elle voyait s'approcher d'elle quelqu'un qu'il ne connaissait pas, qu'il pût remarquer sur le visage d'Odette cette tristesse qu'elle avait eue le jour où il était venu pour la voir pendant que Forcheville était là. Mais c'était rare ; car les jours où malgré tout ce qu'elle avait à faire et la crainte de ce que penserait le monde, elle arrivait à voir Swann, ce qui dominait maintenant dans son attitude était l'assurance : grand contraste, peut-être revanche inconsciente ou réaction naturelle de l'émotion craintive qu'aux premiers temps où elle l'avait connu elle éprouvait auprès de lui, et même loin de lui, quand elle commençait une lettre par ces mots : « Mon ami, ma main tremble si fort que je peux à peine écrire » (elle le prétendait du moins, et un peu de cet émoi devait être sincère pour qu'elle désirât d'en feindre davantage). Swann lui plaisait alors. On ne tremble jamais que pour soi, que pour ceux qu'on aime. Quand notre bonheur n'est plus dans leurs mains, de quel calme, de quelle aisance, de quelle hardiesse on jouit auprès d'eux ! En lui parlant, en lui écrivant, elle n'avait plus de ces mots par lesquels elle cherchait à se donner l'illusion qu'il lui appartenait, faisant naître les occasions de dire « mon », « mien », quand il s'agissait de lui : « Vous êtes mon bien, c'est le parfum de notre amitié, je le garde », de lui parler de l'avenir, de la mort même, comme d'une seule chose pour eux deux. Dans ce temps-là, à tout de qu'il disait, elle répondait avec admiration : « Vous, vous ne serez jamais comme tout le monde » ; elle regardait sa longue tête un peu chauve, dont les gens qui connaissaient les succès de Swann pensaient : « Il n'est pas régulièrement beau si vous voulez, mais il est chic : ce toupet, ce monocle, ce sourire ! », et, plus curieuse peut-être de connaître ce qu'il était que désireuse d'être sa maîtresse, elle disait :

– Si je pouvais savoir ce qu'il y a dans cette tête-là !

Maintenant, à toutes les paroles de Swann elle répondait d'un ton parfois irrité, parfois indulgent :

– Ah ! tu ne seras donc jamais comme tout le monde !

Elle regardait cette tête qui n'était qu'un peu plus vieillie par le souci (mais dont maintenant tous pensaient, en vertu de cette même aptitude qui permet de découvrir les intentions d'un morceau symphonique dont on a lu le programme, et les ressemblances d'un enfant quand on connaît sa parenté : « Il n'est pas positivement laid si vous voulez, mais il est ridicule ; ce monocle, ce toupet, ce sourire ! », réalisant dans leur imagination suggestionnée la démarcation immatérielle qui sépare à quelques mois de distance une tête d'amant de coeur et une tête de cocu), elle disait :

– Ah ! si je pouvais changer, rendre raisonnable ce qu'il y a dans cette tête-là.
