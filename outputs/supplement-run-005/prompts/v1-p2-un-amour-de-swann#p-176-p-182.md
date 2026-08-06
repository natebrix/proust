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
  "Remi": {
    "aliases": [
      "Rémi",
      "Remi",
      "le cocher"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "il"
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
      "confidence": 0.9,
      "evidence": "« ayant fini par la posséder ce soir-là, en commençant par arranger ses catleyas »; Odette sourit et hausse les épaules « comme pour dire… ça me plaît »; la métaphore « faire catleya » devint leur terme pour la possession physique.",
      "explanation": "The passage culminates in a successful intimacy between Swann and Odette and establishes a private code (« faire catleya »). Timidity and the pretext do not negate this local success."
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
      "explanation": "Swann obtains the fulfillment of a long-cherished desire and a shared intimate language, which places him locally in a strong affective position."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-176-p-182"
}

### Candidate characters

[
  "Odette",
  "le narrateur"
]

### Prior local context (optional)

– Surtout, ne me parlez pas, ne me répondez que par signes pour ne pas vous essouffler encore davantage. Cela ne vous gêne pas que je remette droites les fleurs de votre corsage qui ont été déplacées par le choc. J'ai peur que vous ne les perdiez, je voudrais les enfoncer un peu.

### Passage

Elle, qui n'avait pas été habituée à voir les hommes faire tant de façons avec elle, dit en souriant :

– Non, pas du tout, ça ne me gêne pas.

Mais lui, intimidé par sa réponse, peut-être aussi pour avoir l'air d'avoir été sincère quand il avait pris ce prétexte, ou même, commençant déjà à croire qu'il l'avait été, s'écria :

– Oh ! non, surtout, ne parlez pas, vous allez encore vous essouffler, vous pouvez bien me répondre par gestes, je vous comprendrai bien. Sincèrement je ne vous gêne pas ? Voyez, il y a un peu... je pense que c'est du pollen qui s'est répandu sur vous ; vous permettez que je l'essuie avec ma main ? Je ne vais pas trop fort, je ne suis pas trop brutal ? Je vous chatouille peut-être un peu ? mais c'est que je ne voudrais pas toucher le velours de la robe pour ne pas le friper. Mais, voyez-vous, il était vraiment nécessaire de les fixer, ils seraient tombés ; et comme cela, en les enfonçant un peu moi-même... Sérieusement, je ne vous suis pas désagréable ? Et en les respirant pour voir s'ils n'ont vraiment pas d'odeur non plus ? Je n'en ai jamais senti, je peux ? dites la vérité ?

Souriant, elle haussa légèrement les épaules, comme pour dire « vous êtes fou, vous voyez bien que ça me plaît ».

Il élevait son autre main le long de la joue d'Odette ; elle le regarda fixement, de l'air languissant et grave qu'ont les femmes du maître florentin avec lesquelles il lui avait trouvé de la ressemblance ; amenés au bord des paupières, ses yeux brillants, larges et minces, comme les leurs, semblaient prêts à se détacher ainsi que deux larmes. Elle fléchissait le cou comme on leur voit faire à toutes, dans les scènes païennes comme dans les tableaux religieux. Et, en une attitude qui sans doute lui était habituelle, qu'elle savait convenable à ces moments-là et qu'elle faisait attention à ne pas oublier de prendre, elle semblait avoir besoin de toute sa force pour retenir son visage, comme si une force invisible l'eût attiré vers Swann. Et ce fut Swann, qui, avant qu'elle le laissât tomber, comme malgré elle, sur ses lèvres, le retint un instant, à quelque distance, entre ses deux mains. Il avait voulu laisser à sa pensée le temps d'accourir, de reconnaître le rêve qu'elle avait si longtemps caressé et d'assister à sa réalisation, comme une parente qu'on appelle pour prendre sa part du succès d'un enfant qu'elle a beaucoup aimé. Peut-être aussi Swann attachait-il sur ce visage d'Odette non encore possédée, ni même encore embrassée par lui, qu'il voyait pour la dernière fois, ce regard avec lequel, un jour de départ, on voudrait emporter un paysage qu'on va quitter pour toujours.

Mais il était si timide avec elle, qu'ayant fini par la posséder ce soir-là, en commençant par arranger ses catleyas, soit crainte de la froisser, soit peur de paraître rétrospectivement avoir menti, soit manque d'audace pour formuler une exigence plus grande que celle-là (qu'il pouvait renouveler puisqu'elle n'avait pas fâché Odette la première fois), les jours suivants il usa du même prétexte. Si elle avait des catleyas à son corsage, il disait : « C'est malheureux, ce soir, les catleyas n'ont pas besoin d'être arrangés, ils n'ont pas été déplacés comme l'autre soir ; il me semble pourtant que celui-ci n'est pas très droit. Je peux voir s'ils ne sentent pas plus que les autres ? » Ou bien, si elle n'en avait pas : « Oh ! pas de catleyas ce soir, pas moyen de me livrer à mes petits arrangements. » De sorte que, pendant quelque temps, ne fut pas changé l'ordre qu'il avait suivi le premier soir, en débutant par des attouchements de doigts et de lèvres sur la gorge d'Odette, et que ce fut par eux encore que commençaient chaque fois ses caresses ; et, bien plus tard quand l'arrangement (ou le simulacre d'arrangement) des catleyas, fut depuis longtemps tombé en désuétude, la métaphore « faire catleya » devenue un simple vocable qu'ils employaient sans y penser quand ils voulaient signifier l'acte de la possession physique – où d'ailleurs l'on ne possède rien – survécut dans leur langage, où elle le commémorait, à cet usage oublié. Et peut-être cette manière particulière de dire « faire l'amour » ne signifiait-elle pas exactement la même chose que ses synonymes. On a beau être blasé sur les femmes, considérer la possession des plus différentes comme toujours la même et connue d'avance, elle devient au contraire un plaisir nouveau s'il s'agit de femmes assez difficiles – ou crues telles par nous – pour que nous soyons obligés de la faire naître de quelque épisode imprévu de nos relations avec elles, comme avait été la première fois pour Swann l'arrangement des catleyas. Il espérait en tremblant, ce soir-là (mais Odette, se disait-il, si elle était dupe de sa ruse, ne pouvait le deviner), que c'était la possession de cette femme qui allait sortir d'entre leurs larges pétales mauves ; et le plaisir qu'il éprouvait déjà et qu'Odette ne tolérait peut-être, pensait-il, que parce qu'elle ne l'avait pas reconnu, lui semblait, à cause de cela – comme il put paraître au premier homme qui le goûta parmi les fleurs du paradis terrestre – un plaisir qui n'avait pas existé jusque-là, qu'il cherchait à créer, un plaisir – ainsi que le nom spécial qu'il lui donna en garda la trace – entièrement particulier et nouveau.
