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
    },
    {
      "canonical_name": "M. Verdurin",
      "surface_forms": [
        "M. Verdurin",
        "Monsieur Verdurin",
        "Verdurin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "M. Verdurin",
      "target": "Swann",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.95,
      "evidence": "« quand M. Verdurin avait dit que Swann ne lui revenait pas… il avait deviné celle de sa femme » ; « une impossibilité de les lui imposer… de l’y convertir entièrement » ; « c’est une abjuration qu’ils comprirent qu’on ne pourrait pas lui arracher. »",
      "explanation": "The Verdurins register dislike and resist fully accepting Swann because he maintains an independent inner reserve and will not abjure contrary tastes; this functions as a local exclusionary judgment."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "He is locally edged out in the Verdurin circle, as their voiced dislike and inability to ‘convert’ him mark a refusal to fully include him."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-200-p-203"
}

### Candidate characters

[
  "Mme Verdurin",
  "Odette",
  "docteur Cottard",
  "duchesse de Guermantes",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Sentant que souvent il ne pouvait pas réaliser ce qu'elle rêvait, il cherchait du moins à ce qu'elle se plût avec lui, à ne pas contrecarrer ces idées vulgaires, ce mauvais goût qu'elle avait en toutes choses, et qu'il aimait d'ailleurs comme tout ce qui venait d'elle, qui l'enchantaient même, car c'était autant de traits particuliers grâce auxquels l'essence de cette femme lui apparaissait, devenait visible. Aussi, quand elle avait l'air heureux parce qu'elle devait aller à la Reine Topaze, ou que son regard devenait sérieux, inquiet et volontaire, si elle avait peur de manquer la fête des fleurs ou simplement l'heure du thé, avec muffins et toasts, au « Thé de la Rue Royale » où elle croyait que l'assiduité était indispensable pour consacrer la réputation d'élégance d'une femme, Swann, transporté comme nous le sommes par le naturel d'un enfant ou par la vérité d'un portrait qui semble sur le point de parler, sentait si bien l'âme de sa maîtresse affleurer à son visage qu'il ne pouvait résister à venir l'y toucher avec ses lèvres. « Ah ! elle veut qu'on la mène à la fête des fleurs, la petite Odette, elle veut se faire admirer, eh bien, on l'y mènera, nous n'avons qu'à nous incliner. » Comme la vue de Swann était un peu basse, il dut se résigner à se servir de lunettes pour travailler chez lui, et à adopter, pour aller dans le monde, le monocle qui le défigurait moins. La première fois qu'elle lui en vit un dans l'oeil, elle ne put contenir sa joie : « Je trouve que pour un homme, il n'y a pas à dire, ça a beaucoup de chic ! Comme tu es bien ainsi ! tu as l'air d'un vrai gentleman. Il ne te manque qu'un titre ! » ajouta-t-elle, avec une nuance de regret. Il aimait qu'Odette fût ainsi, de même que s'il avait été épris d'une Bretonne, il aurait été heureux de la voir en coiffe et de lui entendre dire qu'elle croyait aux revenants. Jusque-là, comme beaucoup d'hommes chez qui leur goût pour les arts se développe indépendamment de la sensualité, une disparate bizarre avait existé entre les satisfactions qu'il accordait à l'un et à l'autre, jouissant, dans la compagnie de femmes de plus en plus grossières, des séductions d'oeuvres de plus en plus raffinées, emmenant une petite bonne dans une baignoire grillée à la représentation d'une pièce décadente qu'il avait envie d'entendre ou à une exposition de peinture impressionniste, et persuadé d'ailleurs qu'une femme du monde cultivée n'y eût pas compris davantage, mais n'aurait pas su se taire aussi gentiment. Mais, au contraire, depuis qu'il aimait Odette, sympathiser avec elle, tâcher de n'avoir qu'une âme à eux deux lui était si doux, qu'il cherchait à se plaire aux choses qu'elle aimait, et il trouvait un plaisir d'autant plus profond non seulement à imiter ses habitudes, mais à adopter ses opinions, que, comme elles n'avaient aucune racine dans sa propre intelligence, elles lui rappelaient seulement son amour, à cause duquel il les avait préférées. S'il retournait à Serge Panine, s'il recherchait les occasions d'aller voir conduire Olivier Métra, c'était pour la douceur d'être initié dans toutes les conceptions d'Odette, de se sentir de moitié dans tous ses goûts. Ce charme de le rapprocher d'elle, qu'avaient les ouvrages ou les lieux qu'elle aimait, lui semblait plus mystérieux que celui qui est intrinsèque à de plus beaux, mais qui ne la lui rappelaient pas. D'ailleurs, ayant laissé s'affaiblir les croyances intellectuelles de sa jeunesse, et son scepticisme d'homme du monde ayant à son insu pénétré jusqu'à elles, il pensait (ou du moins il avait si longtemps pensé cela qu'il le disait encore) que les objets de nos goûts n'ont pas en eux une valeur absolue, mais que tout est affaire d'époque, de classe, consiste en modes, dont les plus vulgaires valent celles qui passent pour les plus distinguées. Et comme il jugeait que l'importance attachée par Odette à avoir des cartes pour le vernissage n'était pas en soi quelque chose de plus ridicule que le plaisir qu'il avait autrefois à déjeuner chez le prince de Galles, de même, il ne pensait pas que l'admiration qu'elle professait pour Monte-Carlo ou pour le Righi fût plus déraisonnable que le goût qu'il avait, lui, pour la Hollande qu'elle se figurait laide et pour Versailles qu'elle trouvait triste. Aussi, se privait-il d'y aller, ayant plaisir à se dire que c'était pour elle, qu'il voulait ne sentir, n'aimer qu'avec elle.

### Passage

Comme tout ce qui environnait Odette et n'était en quelque sorte que le mode selon lequel il pouvait la voir, causer avec elle, il aimait la société des Verdurin. Là, comme au fond de tous les divertissements, repas, musique, jeux, soupers costumés, parties de campagne, parties de théâtre, même les rares « grandes soirées » données pour les « ennuyeux », il y avait la présence d'Odette, la vue d'Odette, la conversation avec Odette, dont les Verdurin faisaient à Swann, en l'invitant, le don inestimable ; il se plaisait mieux que partout ailleurs dans le « petit noyau », et cherchait à lui attribuer des mérites réels, car il s'imaginait ainsi que par goût il le fréquenterait toute sa vie. Or, n'osant pas se dire, par peur de ne pas le croire, qu'il aimerait toujours Odette, du moins en cherchant á supposer qu'il fréquenterait toujours les Verdurin (proposition qui, a priori, soulevait moins d'objections de principe de la part de son intelligence), il se voyait dans l'avenir continuant à rencontrer chaque soir Odette ; cela ne revenait peut-être pas tout à fait au même que l'aimer toujours, mais, pour le moment, pendant qu'il l'aimait, croire qu'il ne cesserait pas un jour de la voir, c'est tout ce qu'il demandait. « Quel charmant milieu, se disait-il. Comme c'est au fond la vraie vie qu'on mène là ! Comme on y est plus intelligent, plus artiste que dans le monde ! Comme Mme Verdurin, malgré de petites exagérations un peu risibles, a un amour sincère de la peinture, de la musique ! Quelle passion pour les oeuvres, quel désir de faire plaisir aux artistes ! Elle se fait une idée inexacte des gens du monde ; mais avec cela que le monde n'en a pas une plus fausse encore, des milieux artistes ! Peut-être n'ai-je pas de grands besoins intellectuels à assouvir dans la conversation, mais je me plais parfaitement bien avec Cottard, quoiqu'il fasse des calembours ineptes. Et quant au peintre, si sa prétention est déplaisante quand il cherche à étonner, en revanche c'est une des plus belles intelligences que j'aie connues. Et puis surtout, là, on se sent libre, on fait ce qu'on veut sans contrainte, sans cérémonie. Quelle dépense de bonne humeur il se fait par jour dans ce salon-là ! Décidément, sauf quelques rares exceptions, je n'irai plus jamais que dans ce milieu. C'est là que j'aurai de plus en plus mes habitudes et ma vie. »

Et comme les qualités qu'il croyait intrinsèques aux Verdurin n'étaient que le reflet sur eux de plaisirs qu'avait goûtés chez eux son amour pour Odette, ces qualités devenaient plus sérieuses, plus profondes, plus vitales, quand ces plaisirs l'étaient aussi. Comme Mme Verdurin donnait parfois à Swann ce qui seul pouvait constituer pour lui le bonheur ; comme, tel soir où il se sentait anxieux parce qu'Odette avait causé avec un invité plus qu'avec un autre, et où, irrité contre elle, il ne voulait pas prendre l'initiative de lui demander si elle reviendrait avec lui, Mme Verdurin lui apportait la paix et la joie en disant spontanément : « Odette, vous allez ramener Swann, n'est-ce pas » ? comme cet été qui venait et où il s'était d'abord demandé avec inquiétude si Odette ne s'absenterait pas sans lui, s'il pourrait continuer à la voir tous les jours, Mme Verdurin allait les inviter à le passer tous deux chez elle à la campagne – Swann laissant à son insu la reconnaissance et l'intérêt s'infiltrer dans son intelligence et influer sur ses idées, allait jusqu'à proclamer que Mme Verdurin était une grande âme. De quelques gens exquis ou éminents que tel de ses anciens camarades de l'école du Louvre lui parlât : « Je préfère cent fois les Verdurin », lui répondait-il. Et, avec une solennité qui était nouvelle chez lui : « Ce sont des êtres magnanimes, et la magnanimité est, au fond, la seule chose qui importe et qui distingue ici-bas. Vois-tu, il n'y a que deux classes d'êtres : les magnanimes et les autres ; et je suis arrivé à un âge où il faut prendre parti, décider une fois pour toutes qui on veut aimer et qui on veut dédaigner, se tenir à ceux qu'on aime et, pour réparer le temps qu'on a gâché avec les autres, ne plus les quitter jusqu'à sa mort. Eh bien ! ajoutait-il avec cette légère émotion qu'on éprouve quand, même sans bien s'en rendre compte, on dit une chose non parce qu'elle est vraie, mais parce qu'on a plaisir à la dire et qu'on l'écoute dans sa propre voix comme si elle venait d'ailleurs que de nous-mêmes, le sort en est jeté, j'ai choisi d'aimer les seuls coeurs magnanimes et de ne plus vivre que dans la magnanimité. Tu me demandes si Mme Verdurin est véritablement intelligente. Je t'assure qu'elle m'a donné les preuves d'une noblesse de coeur, d'une hauteur d'âme où, que veux-tu, on n'atteint pas sans une hauteur égale de pensée. Certes elle a la profonde intelligence des arts. Mais ce n'est peut-être pas là qu'elle est le plus admirable ; et telle petite action ingénieusement, exquisement bonne, qu'elle a accomplie pour moi, telle géniale attention, tel geste familièrement sublime, révèlent une compréhension plus profonde de l'existence que tous les traités de philosophie. »

Il aurait pourtant pu se dire qu'il y avait des anciens amis de ses parents aussi simples que les Verdurin, des camarades de sa jeunesse aussi épris d'art, qu'il connaissait d'autres êtres d'un grand coeur, et que, pourtant, depuis qu'il avait opté pour la simplicité, les arts et la magnanimité, il ne les voyait plus jamais. Mais ceux-là ne connaissaient pas Odette, et, s'ils l'avaient connue, ne se seraient pas souciés de la rapprocher de lui.

Ainsi il n'y avait sans doute pas, dans tout le milieu Verdurin, un seul fidèle qui les aimât ou crût les aimer autant que Swann. Et pourtant, quand M. Verdurin avait dit que Swann ne lui revenait pas, non seulement il avait exprimé sa propre pensée, mais il avait deviné celle de sa femme. Sans doute Swann avait pour Odette une affection trop particulière et dont il avait négligé de faire de Mme Verdurin la confidente quotidienne ; sans doute la discrétion même avec laquelle il usait de l'hospitalité des Verdurin, s'abstenant souvent de venir dîner pour une raison qu'ils ne soupçonnaient pas et à la place de laquelle ils voyaient le désir de ne pas manquer une invitation chez des « ennuyeux », sans doute aussi, et malgré toutes les précautions qu'il avait prises pour la leur cacher, la découverte progressive qu'ils faisaient de sa brillante situation mondaine, tout cela contribuait à leur irritation contre lui. Mais la raison profonde en était autre. C'est qu'ils avaient très vite senti en lui un espace réservé, impénétrable, où il continuait à professer silencieusement pour lui-même que la princesse de Sagan n'était pas grotesque et que les plaisanteries de Cottard n'étaient pas drôles, enfin et bien que jamais il ne se départît de son amabilité et ne se révoltât contre leurs dogmes, une impossibilité de les lui imposer, de l'y convertir entièrement, comme ils n'en avaient jamais rencontré une pareille chez personne. Ils lui auraient pardonné de fréquenter des ennuyeux (auxquels d'ailleurs, dans le fond de son coeur, il préférait mille fois les Verdurin et tout le petit noyau) s'il avait consenti, pour le bon exemple, à les renier en présence des fidèles. Mais c'est une abjuration qu'ils comprirent qu'on ne pourrait pas lui arracher.
