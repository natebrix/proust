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
      "Mme Swann",
      "Madame Swann",
      "la belle Madame Swann"
    ]
  },
  "Gilberte": {
    "aliases": [
      "Gilberte",
      "la fille de Odette"
    ]
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "l'Ambassadeur",
      "le marquis de Norpois"
    ]
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  },
  "le narrateur": {
    "aliases": [
      "je",
      "moi",
      "mon fils"
    ]
  },
  "le père du narrateur": {
    "aliases": [
      "mon père",
      "Monsieur votre père"
    ]
  },
  "la mère du narrateur": {
    "aliases": [
      "ma mère",
      "Madame"
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
        "M. de Norpois",
        "l'Ambassadeur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Norpois",
      "target": "Bergotte",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.92,
      "evidence": "Il me fut présenté ... désirait être invité ... il prétendait ne pas être invité sans sa compagne ... j’avoue qu’il y a un degré d’ignominie ... Bref, j’éludai la réponse ... la princesse revint à la charge, mais sans plus de succès.",
      "explanation": "Norpois describes having refused to invite Bergotte to the Embassy on the grounds of a moral judgment, excluding him from a prestigious social space while blaming him for hypocrisy and his private life."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "He is explicitly excluded from the Embassy's receptions by Norpois, a marked social sidelining."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-77-p-78"
}

### Candidate characters

[
  "Swann",
  "duchesse de Guermantes",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Mon Dieu, dit Norpois (qui m'inspira sur ma propre intelligence des doutes plus graves que ceux qui me déchiraient d'habitude, quand le narrateur vis que ce que le narrateur mettais mille et mille fois au-dessus de le narrateur-même, ce que le narrateur trouvais de plus élevé au monde, était pour lui tout en bas de l'échelle de ses admirations), le narrateur ne partage pas cette manière de voir. Bergotte est ce que j'appelle un joueur de flûte ; il faut reconnaître du reste qu'il en joue agréablement quoique avec bien du maniérisme, de l'afféterie. Mais enfin ce n'est que cela, et cela n'est pas grand'chose. Jamais on ne trouve dans ses ouvrages sans muscles ce qu'on pourrait nommer la charpente. Pas d'action – ou si peu – mais surtout pas de portée. Ses livres pèchent par la base ou plutôt il n'y a pas de base du tout. Dans un temps comme le nôtre où la complexité croissante de la vie laisse à peine le temps de lire, où la carte de l'Europe a subi des remaniements profonds et est à la veille d'en subir de plus grands encore peut-être, où tant de problèmes menaçants et nouveaux se posent partout, vous m'accorderez qu'on a le droit de demander à un écrivain d'être autre chose qu'un bel esprit qui nous fait oublier dans des discussions oiseuses et byzantines sur des mérites de pure forme, que nous pouvons être envahis d'un instant à l'autre par un double flot de Barbares, ceux du dehors et ceux du dedans. Je sais que c'est blasphémer contre la Sacro-Sainte École de ce que ces messieurs appellent l'Art pour l'Art, mais à notre époque il y a des tâches plus urgentes que d'agencer des mots d'une façon harmonieuse. Celle de Bergotte est parfois assez séduisante, le narrateur n'en disconviens pas, mais au total tout cela est bien mièvre, bien mince, et bien peu viril. Je comprends mieux maintenant, en me reportant à votre admiration tout à fait exagérée pour Bergotte, les quelques lignes que vous m'avez montrées tout à l'heure et sur lesquelles j'aurais mauvaise grâce à ne pas passer l'éponge, puisque vous avez dit vous-même, en toute simplicité, que ce n'était qu'un griffonnage d'enfant (le narrateur l'avais dit, en effet, mais le narrateur n'en pensais pas un mot). À tout péché miséricorde et surtout aux péchés de jeunesse. Après tout, d'autres que vous en ont de pareils sur la conscience, et vous n'êtes pas le seul qui se soit cru poète à son heure. Mais on voit dans ce que vous m'avez montré la mauvaise influence de Bergotte. Évidemment, le narrateur ne vous étonnerai pas en vous disant qu'il n'y avait là aucune de ses qualités, puisqu'il est passé maître dans l'art, tout superficiel du reste, d'un certain style dont à votre âge vous ne pouvez posséder même le rudiment. Mais c'est déjà le même défaut, ce contre-sens d'aligner des mots bien sonores en ne se souciant qu'ensuite du fond. C'est mettre la charrue avant les boeufs, même dans les livres de Bergotte. Toutes ces chinoiseries de forme, toutes ces subtilités de mandarin déliquescent me semblent bien vaines. Pour quelques feux d'artifice agréablement tirés par un écrivain, on crie de suite au chef-d'oeuvre. Les chefs-d'oeuvre ne sont pas si fréquents que cela ! Bergotte n'a pas à son actif, dans son bagage si le narrateur puis dire, un roman d'une envolée un peu haute, un de ces livres qu'on place dans le bon coin de sa bibliothèque. Je n'en vois pas un seul dans son oeuvre. Il n'empêche que chez lui l'oeuvre est infiniment supérieure à l'auteur. Ah ! voilà quelqu'un qui donne raison à l'homme d'esprit qui prétendait qu'on ne doit connaître les écrivains que par leurs livres. Impossible de voir un individu qui réponde moins aux siens, plus prétentieux, plus solennel, moins homme de bonne compagnie. Vulgaire par moments, parlant à d'autres comme un livre, et même pas comme un livre de lui, mais comme un livre ennuyeux, ce qu'au moins ne sont pas les siens, tel est ce Bergotte. C'est un esprit des plus confus, alambiqué, ce que nos pères appelaient un diseur de phébus et qui rend encore plus déplaisantes, par sa façon de les énoncer, les choses qu'il dit. Je ne sais si c'est Loménie ou Sainte-Beuve qui raconte que Vigny rebutait par le même travers. Mais Bergotte n'a jamais écrit Cinq-Mars, ni le Cachet rouge, où certaines pages sont de véritables morceaux d'anthologie.

### Passage

Atterré par ce que Norpois venait de me dire du fragment que je lui avais soumis, songeant d'autre part aux difficultés que j'éprouvais quand je voulais écrire un essai ou seulement me livrer à des réflexions sérieuses, je sentis une fois de plus ma nullité intellectuelle et que je n'étais pas né pour la littérature. Sans doute autrefois à Combray, certaines impressions fort humbles, ou une lecture de Bergotte, m'avaient mis dans un état de rêverie qui m'avait paru avoir une grande valeur. Mais cet état, mon poème en prose le reflétait : nul doute que Norpois n'en eût saisi et percé à jour tout de suite ce que j'y trouvais de beau seulement par un mirage entièrement trompeur, puisque l'Ambassadeur n'en était pas dupe. Il venait de m'apprendre au contraire quelle place infime était la mienne (quand j'étais jugé du dehors, objectivement, par le connaisseur le mieux disposé et le plus intelligent). Je me sentais consterné, réduit ; et mon esprit comme un fluide qui n'a de dimensions que celles du vase qu'on lui fournit, de même qu'il s'était dilaté jadis à remplir les capacités immenses du génie, contracté maintenant, tenait tout entier dans la médiocrité étroite où Norpois l'avait soudain enfermé et restreint.

– Notre mise en présence, à Bergotte et à moi, ajouta-t-il en se tournant vers mon père, ne laissait pas que d'être assez épineuse (ce qui après tout est aussi une manière d'être piquante). Bergotte, voilà quelques années de cela, fit un voyage à Vienne, pendant que j'y étais ambassadeur ; il me fut présenté par la princesse de Metternich, vint s'inscrire et désirait être invité. Or, étant à l'étranger représentant de la France, à qui en somme il fait honneur par ses écrits, dans une certaine mesure, disons, pour être exacts, dans une mesure bien faible, j'aurais passé sur la triste opinion que j'ai de sa vie privée. Mais il ne voyageait pas seul et bien plus il prétendait ne pas être invité sans sa compagne. Je crois ne pas être plus pudibond qu'un autre et, étant célibataire, je pouvais peut-être ouvrir un peu plus largement les portes de l'Ambassade que si j'eusse été marié et père de famille. Néanmoins, j'avoue qu'il y a un degré d'ignominie dont je ne saurais m'accommoder, et qui est rendu plus écoeurant encore par le ton plus que moral, tranchons le mot, moralisateur, que prend Bergotte dans ses livres où on ne voit qu'analyses perpétuelles et d'ailleurs, entre nous, un peu languissantes, de scrupules douloureux, de remords maladifs, et pour de simples peccadilles, de véritables prêchi-prêcha (on sait ce qu'en vaut l'aune) alors qu'il montre tant d'inconscience et de cynisme dans sa vie privée. Bref, j'éludai la réponse, la princesse revint à la charge, mais sans plus de succès. De sorte que je ne suppose pas que je doive être très en odeur de sainteté auprès du personnage, et je ne sais pas jusqu'à quel point il a apprécié l'attention de Swann de l'inviter en même temps que moi. À moins que ce ne soit lui qui l'ait demandé. On ne peut pas savoir, car au fond c'est un malade. C'est même sa seule excuse.
