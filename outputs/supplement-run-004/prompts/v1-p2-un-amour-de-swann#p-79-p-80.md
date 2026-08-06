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
  "Odette": {
    "aliases": [
      "Odette",
      "Odette de Crécy",
      "Mme de Crécy",
      "Mme de Crecy"
    ]
  },
  "Mme Verdurin": {
    "aliases": [
      "Mme Verdurin",
      "Madame Verdurin",
      "la Patronne"
    ]
  },
  "M. Verdurin": {
    "aliases": [
      "M. Verdurin",
      "Monsieur Verdurin"
    ]
  },
  "docteur Cottard": {
    "aliases": [
      "le docteur",
      "Cottard",
      "le docteur Cottard"
    ]
  },
  "le pianiste": {
    "aliases": [
      "le jeune artiste",
      "le jeune pianiste",
      "le pianiste",
      "le petit pianiste"
    ]
  },
  "le peintre": {
    "aliases": [
      "le peintre",
      "Biche"
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
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "« une sorte de rajeunissement »; « il se sentait de nouveau le désir et presque la force de consacrer sa vie »; puis, chez Mme Verdurin, « il la tenait… il pourrait l'avoir chez lui… apprendre son langage et son secret »",
      "explanation": "The narrator presents Swann's rediscovery of the « phrase » musicale as an inner revitalization that restores to him desire and a power of orientation, crowned by the identification of the work and the possibility of devoting himself to it."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann moves from a dryness and a withdrawal from ideals to a resumption of desire, meaning, and mastery thanks to the music that he can now name and find again."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-79-p-80"
}

### Candidate characters

[
  "M. Vinteuil",
  "Mme Verdurin",
  "le peintre",
  "le pianiste"
]

### Prior local context (optional)

D'un rythme lent elle le dirigeait ici d'abord, puis là, puis ailleurs, vers un bonheur noble, inintelligible et précis. Et tout d'un coup, au point où elle était arrivée et d'où il se préparait à la suivre, après une pause d'un instant, brusquement elle changeait de direction, et d'un mouvement nouveau, plus rapide, menu, mélancolique, incessant et doux, elle l'entraînait avec elle vers des perspectives inconnues. Puis elle disparut. Il souhaita passionnément la revoir une troisième fois. Et elle reparut en effet mais sans lui parler plus clairement, en lui causant même une volupté moins profonde. Mais rentré chez lui il eut besoin d'elle, il était comme un homme dans la vie de qui une passante qu'il a aperçue un moment vient de faire entrer l'image d'une beauté nouvelle qui donne à sa propre sensibilité une valeur plus grande, sans qu'il sache seulement s'il pourra revoir jamais celle qu'il aime déjà et dont il ignore jusqu'au nom.

### Passage

Même cet amour pour une phrase musicale sembla un instant devoir amorcer chez Swann la possibilité d'une sorte de rajeunissement. Depuis si longtemps il avait renoncé à appliquer sa vie à un but idéal et la bornait à la poursuite de satisfactions quotidiennes, qu'il croyait, sans jamais se le dire formellement, que cela ne changerait plus jusqu'à sa mort ; bien plus, ne se sentant plus d'idées élevées dans l'esprit, il avait cessé de croire à leur réalité, sans pouvoir non plus la nier tout à fait. Aussi avait-il pris l'habitude de se réfugier dans des pensées sans importance et qui lui permettaient de laisser de côté le fond des choses. De même qu'il ne se demandait pas s'il n'eût pas mieux fait de ne pas aller dans le monde, mais en revanche savait avec certitude que s'il avait accepté une invitation il devait s'y rendre, et que s'il ne faisait pas de visite après il lui fallait laisser des cartes, de même dans sa conversation il s'efforçait de ne jamais exprimer avec coeur une opinion intime sur les choses, mais de fournir des détails matériels qui valaient en quelque sorte par eux-mêmes et lui permettaient de ne pas donner sa mesure. Il était extrêmement précis pour une recette de cuisine, pour la date de la naissance ou de la mort d'un peintre, pour la nomenclature de ses oeuvres. Parfois, malgré tout, il se laissait aller à émettre un jugement sur une oeuvre, sur une manière de comprendre la vie, mais il donnait alors à ses paroles un ton ironique comme s'il n'adhérait pas tout entier à ce qu'il disait. Or, comme certains valétudinaires chez qui, tout d'un coup, un pays où ils sont arrivés, un régime différent, quelquefois une évolution organique, spontanée et mystérieuse, semblent amener une telle régression de leur mal qu'ils commencent à envisager la possibilité inespérée de commencer sur le tard une vie toute différente, Swann trouvait en lui, dans le souvenir de la phrase qu'il avait entendue, dans certaines sonates qu'il s'était fait jouer, pour voir s'il ne l'y découvrirait pas, la présence d'une de ces réalités invisibles auxquelles il avait cessé de croire et auxquelles, comme si la musique avait eu sur la sécheresse morale dont il souffrait une sorte d'influence élective, il se sentait de nouveau le désir et presque la force de consacrer sa vie. Mais n'étant pas arrivé à savoir de qui était l'oeuvre qu'il avait entendue, il n'avait pu se la procurer et avait fini par l'oublier. Il avait bien rencontré dans la semaine quelques personnes qui se trouvaient comme lui à cette soirée et les avait interrogées ; mais plusieurs étaient arrivées après la musique ou parties avant ; certaines pourtant étaient là pendant qu'on l'exécutait, mais étaient allées causer dans un autre salon, et d'autres restées à écouter n'avaient pas entendu plus que les premières. Quant aux maîtres de maison, ils savaient que c'était une oeuvre nouvelle que les artistes qu'ils avaient engagés avaient demandé à jouer ; ceux-ci étant partis en tournée, Swann ne put pas en savoir davantage. Il avait bien des amis musiciens, mais tout en se rappelant le plaisir spécial et intraduisible que lui avait fait la phrase, en voyant devant ses yeux les formes qu'elle dessinait, il était pourtant incapable de la leur chanter. Puis il cessa d'y penser.

Or, quelques minutes à peine après que le petit pianiste avait commencé de jouer chez Mme Verdurin, tout d'un coup après une note longuement tendue pendant deux mesures, il vit approcher, s'échappant de sous cette sonorité prolongée et tendue comme un rideau sonore pour cacher le mystère de son incubation, il reconnut, secrète, bruissante et divisée, la phrase aérienne et odorante qu'il aimait. Et elle était si particulière, elle avait un charme si individuel et qu'aucun autre n'aurait pu remplacer, que ce fut pour Swann comme s'il eût rencontré dans un salon ami une personne qu'il avait admirée dans la rue et désespérait de jamais retrouver. À la fin, elle s'éloigna, indicatrice, diligente, parmi les ramifications de son parfum, laissant sur le visage de Swann le reflet de son sourire. Mais maintenant il pouvait demander le nom de son inconnue (on lui dit que c'était l'andante de la sonate pour piano et violon de Vinteuil,) il la tenait, il pourrait l'avoir chez lui aussi souvent qu'il voudrait, essayer d'apprendre son langage et son secret.
