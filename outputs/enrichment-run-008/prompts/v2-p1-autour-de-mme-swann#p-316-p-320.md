You are annotating a French passage from Marcel Proust's *À la recherche du temps perdu* for **local appraisal events** and **character status effects**.

Your purpose is to generate structured literary-social annotations that can later be transformed into different notions of "winning" and "losing."
Do **not** reduce the passage to a single winner-loser verdict unless the evidence is overwhelmingly explicit.
Instead, identify the **dominant local evaluative acts** and their consequences for the named characters.

## Inputs

You will be given:

1. A French passage. The text is Proust's original, unaltered.
2. A **reference sheet** of known characters and their surface forms for this
   chapter, including a small reviewed set of individuated unnamed figures
   (e.g. l'amie de Mlle Vinteuil) and resolution notes for family references
   ("ma grand'mère", "maman").
3. Optionally, brief prior context from the immediately preceding window.

## Scope rules

* The reference sheet is **advisory context, not a constraint on who exists**.
  When a referent in the passage matches an entry, use that entry's canonical
  name.
* List **every named character materially involved** in the passage's
  evaluative or social dynamics — including characters **not** on the
  reference sheet. For those, use the passage's surface form verbatim as
  `canonical_name` and add `"resolution": "unresolved"` to their
  `characters_present` entry.
* Unnamed figures are included only when they appear on the reference sheet
  as reviewed entries; do not invent descriptor-identified characters beyond
  the sheet.
* Work primarily from the passage itself.
* Use prior context only for local disambiguation of references, not for broad plot inference.
* Do not invent motives, unstated events, or long-run arc interpretations.
* Prefer the **smallest sufficient reading** of the passage.

## What to detect

Track local shifts in how named characters are positioned through:

* praise
* blame
* admiration
* snub
* prestige by association
* discredit by association
* narrated elevation or diminishment
* inclusion in or exclusion from valued social space
* signs that another character depends on, yields to, or dismisses them

For this first pass, prefer **broad, stable categories** over fine-grained distinctions.
Do not split one local movement into multiple event labels unless the passage clearly stages them as distinct.

## Interpretive principles

A character may come out ahead or behind in several ways:

* directly, by being praised or insulted
* comparatively, by being favored over another
* socially, by being included, deferred to, or excluded
* rhetorically, by speaking with force, wit, authority, or discernment
* emotionally, by gaining leverage over another character
* associationally, by benefiting from or suffering through a linked person
* narratively, through a passage that appears to elevate or diminish them

Do **not** judge:

* morality
* factual correctness
* long-term importance
* whether the character "deserves" the treatment

Judge only the **local evaluative and social dynamics** of the supplied passage.

## Special caution for Proust

Proust often layers evaluation through:

* quoted speech
* free indirect style
* remembered perception
* irony
* narrator distance
* social codes that are reported rather than endorsed

For every evaluative event, distinguish:

* who is making the evaluation
* who is its target
* whether the passage appears to endorse, neutrally report, ironize, or leave uncertain that evaluation

## Special rule for consummation and renewal

In erotic or affective passages, distinguish between:

* the mode of attainment
* the attained local outcome

Timidity, dependence, awkwardness, compromise, or ironic framing may qualify how an outcome is reached, but they do **not** by themselves override a plainly successful local outcome.

If the passage culminates in any of the following, that dominant positive movement must be represented in the annotation:

* realized intimacy
* mutual receptivity
* successful consummation
* narrator-endorsed emotional renewal

Do **not** collapse such passages into pure diminishment merely because the path to that outcome is hesitant, dependent, socially compromising, or lightly ironized.
When both weakness and successful renewal coexist, preserve the dominant local success and mention the qualifying weakness only if it is central.
Do **not** treat mere reunion, temporary relief, or the end of uncertainty as equivalent to consummation or renewal.
Relief at finding the desired person again counts as dominant positive movement only if the passage itself goes on to show realized intimacy, mutual receptivity, or an explicitly narrator-endorsed affective transformation.
Weigh the passage as a whole rather than over-privileging its final sentence or final image.
Do **not** let a single upbeat note, however striking or late-placed, outweigh a passage whose dominant body is organized around anxious search, dependency, agitation, jealousy, or diminishment unless the passage is plainly structured so that the culmination redefines the whole local movement.
If most of the passage is governed by distress and only a brief closing note offers relief, prefer a mixed or negative reading unless the text clearly presents that closing turn as the main narrated point.

## Task

1. Identify only the named characters who are materially involved in the dominant local movement.
2. Extract only the **significant** appraisal or status-relevant events.
3. Record only the dominant local status effects for the characters involved.
4. Note ambiguity only when it materially changes how the event or status effect should be read.
5. Multiple characters may gain or lose simultaneously.
6. If there is no meaningful status movement, return empty `appraisal_events` and empty `status_effects`.
7. Prefer fewer, high-quality events over many trivial ones.
8. Default to **1 main event** for a passage.
9. Use **2 events** when the passage clearly contains two distinct, non-redundant local movements.
10. A dense social scene — several characters, several separately witnessed movements — may carry up to **4 events** when each grounds a distinct movement that the others do not cover. Never return more than **4 appraisal events**.
11. Do not add balancing or countervailing events unless they are central to the passage.
12. For one character: at most **one status effect per dimension**. A second effect in the same dimension is allowed only for clearly separate moments of the passage, each citing different `based_on_events`.
13. For one character, a single event grounds at most **one** effect among `general_appraisal`, `emotional_position`, and `rhetorical_position` — these are facets of the same contest; choose the dimension that best names the movement. The same event may additionally ground a `social_status` or `inclusion_exclusion` effect for that character when that dimension's own criterion is independently met.

## Output

Return valid JSON only.

Schema:

{
"characters_present": [
{
"canonical_name": "string",
"surface_forms": ["string"],
"presence_type": "explicit | implicit",
"presence_confidence": 0.0,
"resolution": "resolved | unresolved"
}
],
"appraisal_events": [
{
"event_id": "E1",
"source": "canonical character name | narrator | collective_social_voice | unknown",
"target": "canonical character name",
"type": "praise | blame | admiration | snub | prestige_association | discredit_association | narrated_elevation | narrated_diminishment | other",
"polarity": "positive | negative | mixed",
"narrative_stance": "endorsed | neutral_report | ironized | uncertain",
"confidence": 0.0,
"evidence": "brief quotation or paraphrase from the passage",
"explanation": "1-2 sentence explanation"
}
],
"status_effects": [
{
"character": "canonical character name",
"dimension": "general_appraisal | social_status | rhetorical_position | emotional_position | inclusion_exclusion",
"delta": -2,
"based_on_events": ["E1"],
"confidence": 0.0,
"explanation": "brief explanation"
}
],
"ambiguities": [
"string"
]
}

## Schema guidance

Use a reduced first-pass schema.

Do not add fields beyond the schema above.

### `characters_present`

Include only named characters who are either:

* explicitly mentioned in the passage
* clearly implicated by nearby reference in the passage or optional prior context

Do **not** include every discourse participant.
Include a character only if omitting them would distort the dominant local appraisal or status movement.
Peripheral parents, relatives, or bystanders should usually be omitted in this first pass.

### `appraisal_events`

Record only significant local evaluative or status-relevant events.

Prefer fewer, stronger events over exhaustive tagging.
Default to **1 event**.
Use **2 events** when the passage clearly contains two distinct movements that are both central.
A dense social scene may carry up to **4 events** when each grounds a distinct
witnessed movement that the others do not cover — a salon may snub one guest,
crown another, and watch a third fall in the same evening. This is not license
to fragment: every "bad reason to create an event" below still applies, and an
event that merely restates or elaborates another must not be added.
Never emit more than **4 events** total.

Good reasons to create an event:

* one character is praised, admired, deferred to, or favored
* one character is blamed, ridiculed, excluded, snubbed, or diminished
* a comparison clearly puts one character above or below another
* a character gains or loses rhetorical or emotional leverage
* association with another figure clearly raises or lowers local standing

Bad reasons to create a separate event:

* a subordinate nuance merely restates the same local movement
* a quoted phrase provides color but not a distinct appraisal act
* a possible counter-reading is present but not central
* a nearby character is involved in the scene but not in the main evaluative movement
* one sentence offers several pieces of evidence for the same local movement
* the same movement could be described with several nearby labels

Prefer these broad labels when possible:

* `narrated_diminishment`
* `narrated_elevation`
* `snub`
* `admiration`
* `blame`
* `prestige_association`
* `discredit_association`

Avoid finer-grained labels unless the passage truly demands them.
Choose the **single best label** for the dominant movement rather than enumerating nearby alternatives.

### `status_effects`

Use these dimensions only. Each is governed by its own criterion:

* `general_appraisal` — the passage's evaluative verdict on a character:
  praised, admired, ridiculed, or discredited in the narration or the scene.
* `rhetorical_position` — the upper hand in talk: wit that lands or misfires,
  an argument won or lost, a line that silences or is silenced.
* `emotional_position` — emotional leverage in a relation: who needs, who
  withholds, who suffers visibly, who controls.
* `social_status` — a movement or display of standing that is **witnessed
  inside the world of the passage**. Someone present — or society's reported
  voice — must register it: deference given or withheld, a reception or
  invitation that marks rank, a public snub, a reputation spoken of as risen
  or fallen. A private, unshared judgment of a character is
  `general_appraisal`, not `social_status`.
* `inclusion_exclusion` — a **boundary event**: an interior with an
  exterior, and a character shown crossing that line or barred at it —
  introduced or not introduced, greeted or cut, invited or left out, absorbed
  into the group or held at its edge. The boundary need not belong to a
  social set: the family table, a household, a bedroom door, a clan, a club,
  a box at the theatre, the circle of a conversation, the intimacy of
  tutoiement, an institution, a nation, a clandestine fraternity — all count,
  so long as the passage shows an inside, an outside, and this character's
  position across that line changing. Mere
  presence at a gathering is not inclusion, and absence is not exclusion;
  the boundary itself must be shown moving for this character.

Create status effects only when there is meaningful local movement.
Record **every distinct movement that meets its dimension's criterion**.
Most characters will still have 1 status effect, some 2; a dense social
scene may genuinely support more. There is no fixed cap, but distinctness
is strict:

* Two effects in the **same dimension** for one character are allowed only
  when they arise from clearly separate moments of the passage — different
  events, different witnesses, a different boundary — and could not honestly
  be summarized as one movement. Each must cite different `based_on_events`.
  When in doubt, summarize as one.
* Never restate one social fact as several effects.
* For one character, a single event grounds at most **one** effect among
  `general_appraisal`, `emotional_position`, and `rhetorical_position`:
  these are facets of the same contest, so choose the dimension that best
  names the movement rather than recording several facets of it.
* An event supports a dimension only when it meets that dimension's
  criterion, not when it merely brushes it.
* Do not collapse a real standing or belonging movement into
  `general_appraisal` to economize: a witticism that wins the room may be
  both `rhetorical_position` for the duel and `social_status` for the
  standing it visibly confers — record each on its own merits.

`delta` should reflect local movement in the passage:

Status delta:

* -2 = clearly diminished in this passage
* -1 = somewhat diminished
* 0 = do not use — if there is no clear movement, record no effect at all: null is not zero
* +1 = somewhat elevated
* +2 = clearly elevated

### `ambiguities`

Use this list to record uncertainty such as:

* ironic or layered narration
* uncertain evaluator
* ambiguous target
* uncertain reference resolution (who a surface form refers to)
* unclear endorsement by narrator or social voice

Default to an empty list.
Only add an ambiguity when it materially changes how the event or status effect should be read.
Do not use `ambiguities` for routine caveats or general interpretive hedging.
Usually return an empty list.

### Confidence

* 0.0 to 1.0
* Be conservative when irony, layered narration, or reference resolution makes interpretation unstable.

## Important rules

* Named characters, plus only the reviewed unnamed figures on the reference sheet.
* Use the reference sheet's canonical names where a referent matches; use the
  passage's surface form verbatim with `"resolution": "unresolved"` where it
  does not. Never silently drop a materially involved character.
* `"resolution": "resolved"` may be omitted (it is the default); always state
  `"resolution": "unresolved"` explicitly.
* If a surface form is ambiguous, mention that in "ambiguities."
* Do not infer broad character arcs.
* Do not force zero-sum logic.
* A single event may affect both source and target, but record that through separate status effects if needed.
* Ignore trivial mentions that do not meaningfully alter evaluation or status.
* Do not add a winner/loser verdict field.
* Do not add a summary object.
* If there is no meaningful status movement, keep `appraisal_events` and `status_effects` empty rather than inventing a weak event.
* When narrator framing clearly guides the judgment, do not default to `neutral_report`.
* Do not turn one dominant movement into a chain of micro-events.
* Do not add balancing positive and negative effects unless both are central to the passage.
* Do not annotate every evaluatively charged phrase.
* Multiple quotations can support one event.
* If you are unsure between several event labels, pick the broadest stable one.

## Compression examples

These examples illustrate the intended first-pass compression.

Example A:

If a passage reveals that Swann is far more socially eminent than the family realizes, do **not** split this into:

* one event for narrator elevation
* one event for aristocratic association
* one event for local under-recognition

Instead, prefer:

* `1` event: `narrated_elevation`
* `1` main status effect: `social_status +2`

Example B:

If a passage exposes Legrandin’s evasive, socially defensive performance, do **not** split this into:

* ridicule
* humiliation
* blame
* rhetorical weakness
* emotional vulnerability

Instead, prefer:

* `1` event: `narrated_diminishment`
* `1` or `2` status effects at most, only if both are central

Example C:

If a passage shows Swann reaching long-desired intimacy with Odette through awkward pretext, hesitation, or dependence, do **not** reduce it to:

* one event for timidity
* one event for dependence
* one event for social compromise

Instead, prefer:

* `1` central event that preserves the realized intimacy or renewal
* a positive `emotional_position` effect when that attainment is the dominant local outcome
* a secondary note about hesitation or compromise only if it is itself central
* do **not** promote mere reunion relief unless the passage clearly crosses into intimacy or explicit renewal
* do **not** let a brief closing uplift outweigh the dominant movement of the passage unless that culminating turn is clearly the passage's main point

Example D:

If a passage shows a guest turned away at a hostess's door and then dwells
on that same refusal — the guest's confusion, the footman's repeated
message, the slow walk back down the staircase — that is **one** boundary
event:

* `1` event: `snub`
* `1` status effect: `inclusion_exclusion -2`

But if the same passage later shows the assembled room deferring to a newly
arrived title, that is a separate witnessed movement, for a different
character, grounded in its own event — record it on its own merits.

## Positive examples

Use the following as models for the **level of compression and focality** expected in this task.

### Positive example 1: single narrator-led elevation

```json
{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": ["Swann", "M. Swann"],
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
      "confidence": 0.97,
      "evidence": "Swann is described as one of the most elegant and most sought-after men in the highest social world, though the family does not realize it.",
      "explanation": "The narrator sharply elevates Swann by contrasting the family's ignorance with his actual prestige, influence, and desirability in elite society."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": ["E1"],
      "confidence": 0.97,
      "explanation": "Within this passage, Swann's local standing rises clearly because he is framed as socially eminent far beyond what his hosts understand."
    }
  ],
  "ambiguities": []
}
```

### Positive example 2: single narrator-led diminishment

```json
{
  "characters_present": [
    {
      "canonical_name": "Legrandin",
      "surface_forms": ["Legrandin", "M. Legrandin"],
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
      "confidence": 0.95,
      "evidence": "At the name Guermantes, Legrandin's bodily reaction, evasive denial, and defensive explanation expose the social craving he claims to reject; the narrator explicitly concludes that he is a snob.",
      "explanation": "The passage locally lowers Legrandin by revealing a gap between his anti-snob rhetoric and his actual dependence on aristocratic approval."
    }
  ],
  "status_effects": [
    {
      "character": "Legrandin",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": ["E1"],
      "confidence": 0.95,
      "explanation": "Legrandin comes off worse because the narrator exposes him as insincere and socially compromised."
    },
    {
      "character": "Legrandin",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": ["E1"],
      "confidence": 0.88,
      "explanation": "His local social position is weakened because he appears dependent on aristocratic regard and anxious not to be associated with bourgeois friends."
    }
  ],
  "ambiguities": []
}
```

### Positive example 3: genuine two-movement mixed passage

```json
{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": ["Swann"],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "M. Vinteuil",
      "surface_forms": ["Vinteuil", "M. Vinteuil"],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "M. Vinteuil",
      "target": "Swann",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.94,
      "evidence": "Vinteuil calls Swann an exquisite man and speaks of him with enthusiastic veneration.",
      "explanation": "Locally, Swann is elevated by Vinteuil's explicit admiration and deference."
    },
    {
      "event_id": "E2",
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "Swann's marriage is treated as socially misplaced, and Vinteuil withholds sending his daughter to him.",
      "explanation": "The passage lowers Swann through the social discredit attached to his marriage, which overrides personal admiration and leads to a practical form of exclusion."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": ["E1"],
      "confidence": 0.9,
      "explanation": "Swann is locally praised as personally admirable and refined."
    },
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": ["E2"],
      "confidence": 0.92,
      "explanation": "His local standing is diminished because his marriage carries social stigma that affects how others treat him."
    }
  ],
  "ambiguities": [
    "The praise of Swann is explicit, but the passage also stresses the hypocrisy of those who admire him personally while condemning his marriage socially."
  ]
}
```

### Positive example 4: hesitant path, successful local consummation

```json
{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": ["Swann"],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Odette",
      "surface_forms": ["Odette"],
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
      "evidence": "Swann reaches the desired intimacy with Odette, and the passage treats the realized encounter as the decisive local development even though he arrives there through hesitation and pretext.",
      "explanation": "The dominant movement is successful consummation or affective fulfillment. Swann's awkwardness qualifies the mode of attainment, but it does not cancel the attained local outcome."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 2,
      "based_on_events": ["E1"],
      "confidence": 0.9,
      "explanation": "Swann comes out locally ahead because the passage culminates in realized intimacy rather than frustrated pursuit."
    }
  ],
  "ambiguities": []
}
```

## Inputs begin below

### Reference sheet

{
  "Robert de Saint-Loup": {
    "aliases": [
      "M. le marquis de Robert de Saint-Loup",
      "le neveu de Mme de Villeparisis",
      "marquis de Robert de Saint-Loup",
      "marquis de Saint-Loup-en-Bray",
      "marquis de Saint-Loup",
      "Robert de Saint-Loup",
      "M. le marquis",
      "Saint-Loup",
      "Robert",
      "Bobbey"
    ],
    "notes": ""
  },
  "princesse de Guermantes": {
    "aliases": [
      "princesse de Guermantes-Bavière",
      "Mme de Guermantes-Bavière",
      "princesse de Guermantes",
      "Marie-Gilbert",
      "la princesse"
    ],
    "notes": "Marie-Gilbert, princesse de Guermantes in her own right. Ruling 2026-08-11: her era ends with her death; the title later passes to Mme Verdurin by remarriage — a different person holding the same title, not a same-person merge (policy keep_separate). In the Le Temps retrouvé matinée, the bare form \"princesse de Guermantes\" is genuinely ambiguous between the two entities; the registry surfaces that as an ambiguous resolution (see mme-verdurin's chapter-scoped form of the same text) rather than guessing."
  },
  "duchesse de Guermantes": {
    "aliases": [
      "Madame duchesse de Guermantes",
      "Mme duchesse de Guermantes",
      "duchesse de Guermantes",
      "Madame de Guermantes",
      "princesse des Laumes",
      "Mme de Guermantes",
      "Mme des Laumes",
      "la duchesse",
      "princesse",
      "duchesse",
      "Oriane"
    ],
    "notes": "princesse des Laumes -> duchesse merge already reviewed and accepted upstream in Nathan's normalization plan."
  },
  "M. Grevy": {
    "aliases": [
      "le Président de la République",
      "le Chef de l'État",
      "M. Grevy",
      "M. Grévy"
    ],
    "notes": ""
  },
  "Gilberte": {
    "aliases": [
      "Mlle de comte de Forcheville",
      "Mme de Robert de Saint-Loup",
      "marquise de Saint-Loup",
      "Mlle de Forcheville",
      "Mlle d'Éporcheville",
      "la fille de Odette",
      "Mme de Saint-Loup",
      "Gilberte Swann",
      "Mlle Swann",
      "Gilberte"
    ],
    "notes": ""
  },
  "jeune blonde de Rivebelle": {
    "aliases": [
      "jeune blonde à l'air triste",
      "jeune blonde de Rivebelle",
      "jeune blonde"
    ],
    "notes": ""
  },
  "la grand-mère": {
    "aliases": [
      "la grand-mère du narrateur",
      "la grand-mère",
      "ma grand-mère",
      "ma grand'mère",
      "ma grand-mere",
      "Mme Amédée",
      "grand'mère",
      "grand-mère",
      "Grand'mère",
      "Bathilde"
    ],
    "notes": ""
  },
  "le grand-père du narrateur": {
    "aliases": [
      "le grand-père du narrateur",
      "mon grand-père",
      "mon grand-pere",
      "Amédée"
    ],
    "notes": ""
  },
  "Mme de Cambremer": {
    "aliases": [
      "Mme de Cambremer-Legrandin",
      "marquise de Cambremer",
      "Madame de Cambremer",
      "Mme de Cambremer"
    ],
    "notes": "Ruling 2026-08-11 (registry-audit v2 candidates): the audit candidate 'marquise de Cambremer' (count 11) is ambiguous between this entity (nee Legrandin, the marquis's wife) and his mother, the dowager marquise -- see the new la-marquise-douairiere-de-cambremer entity. The shared surface form is added to both entities and deliberately resolves as ambiguous rather than guessed, per registry semantics."
  },
  "princesse de Luxembourg": {
    "aliases": [
      "La princesse de Luxembourg",
      "princesse de Luxembourg",
      "Mme de Luxembourg"
    ],
    "notes": ""
  },
  "marquise de Saint-Euverte": {
    "aliases": [
      "marquise de Saint-Euverte",
      "Mme de Sainte-Euverte",
      "Mme de Saint-Euverte",
      "Saint-Euverte"
    ],
    "notes": ""
  },
  "M. de Marsantes": {
    "aliases": [
      "Saint-Loup de Saint-Loup",
      "M. de Marsantes",
      "Marsantes"
    ],
    "notes": "Robert's father, dead before the novel opens; posthumous-mention-only, NEVER a scene participant. Ruling 2026-08-11 closes the audit's 4-unit adjudication question without re-annotating those units: they belong to the legacy pipeline, and legacy units die with the legacy corpus (text rewriting is dead under prompt v2 regardless)."
  },
  "Rachel": {
    "aliases": [
      "Rachel quand du Seigneur",
      "Zézette",
      "Rachel"
    ],
    "notes": "MISSING from every legacy layer -> structurally invisible to the annotator under scope rule 17 despite co-lead scenes in V3 and the V7 Berma duel."
  },
  "comtesse de Monteriender": {
    "aliases": [
      "comtesse de Monteriender",
      "Mme de Monteriender",
      "Monteriender"
    ],
    "notes": ""
  },
  "Mme de Villeparisis": {
    "aliases": [
      "marquise de Villeparisis",
      "Madame de Villeparisis",
      "Mme de Villeparisis",
      "Madame la Marquise",
      "tante Villeparisis",
      "marquise"
    ],
    "notes": ""
  },
  "l'amie de Mlle Vinteuil": {
    "aliases": [
      "l'amie de Mlle Vinteuil"
    ],
    "notes": "Genuinely unnamed but individuated and consequential. Ruling 2026-08-11: admitted as an entity — reviewed case-by-case; descriptor-identified; mention-only."
  },
  "Mme de Chaussepierre": {
    "aliases": [
      "Madame de Chaussepierre",
      "Mme de Chaussepierre",
      "Chaussepierre"
    ],
    "notes": ""
  },
  "Mme de Montmorency": {
    "aliases": [
      "duchesse de Montmorency",
      "Mme de Montmorency"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'Mme de Montmorency' (count 11) and 'duchesse de Montmorency' (count 6) as one entity, same person."
  },
  "Albertine": {
    "aliases": [
      "Mademoiselle Albertine",
      "Albertine Simonet",
      "Mlle Albertine",
      "Mlle Simonet",
      "Albertine",
      "ALBERTINE"
    ],
    "notes": ""
  },
  "général de Froberville": {
    "aliases": [
      "général de Froberville",
      "general de Froberville",
      "Froberville"
    ],
    "notes": ""
  },
  "Norpois": {
    "aliases": [
      "Monsieur l'Ambassadeur",
      "le marquis de Norpois",
      "marquis de Norpois",
      "M. de Noirpois",
      "M. de Norpois",
      "l'Ambassadeur",
      "Norpois"
    ],
    "notes": ""
  },
  "princesse de Caprarola": {
    "aliases": [
      "princesse de Caprarola"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Fashionable rival hostess discussed in the Guermantes salon."
  },
  "colonel de Froberville": {
    "aliases": [
      "colonel de Froberville",
      "M. de Froberville"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'M. de Froberville' (count 6) and 'colonel de Froberville' (count 5). général de Froberville already exists as a confirmed entity; kept as a separate entity per the task ruling (the novel appears to distinguish the général from a colonel cousin) but linked with review:true since the generic honorific 'M. de Froberville' could in principle denote either man and the family relation is not independently verified here."
  },
  "Mme de Marsantes": {
    "aliases": [
      "comtesse de Marsantes",
      "Mme de Marsantes"
    ],
    "notes": "Robert's mother. MISSING from all legacy layers; her mentions risk mis-resolving to her dead husband."
  },
  "marquis de Forestelle": {
    "aliases": [
      "marquis de Forestelle",
      "M. de Forestelle",
      "Forestelle"
    ],
    "notes": ""
  },
  "marquise de Gallardon": {
    "aliases": [
      "marquise de Gallardon",
      "Mme de Gallardon",
      "Gallardon"
    ],
    "notes": ""
  },
  "Odette": {
    "aliases": [
      "la belle Madame Swann",
      "Mme de Forcheville",
      "Odette de Crécy",
      "la dame en rose",
      "Odette de Crecy",
      "Miss Sacripant",
      "Madame Swann",
      "Mme de Crécy",
      "Mme de Crecy",
      "Mme Swann",
      "Odette"
    ],
    "notes": ""
  },
  "capitaine de Borodino": {
    "aliases": [
      "capitaine de Borodino",
      "prince de Borodino",
      "M. de Borodino"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'capitaine de Borodino' (count 8) and 'M. de Borodino' (count 5) as one entity, same person -- a Doncieres regiment officer known by rank and by hereditary title. 'prince de Borodino' (his formal title, per task ruling) added as a third surface form though it did not itself surface as an audit candidate."
  },
  "la marquise douairière de Cambremer": {
    "aliases": [
      "marquise de Cambremer"
    ],
    "notes": "Registry-audit candidate 2026-08: 'marquise de Cambremer' (count 11) is ambiguous between the existing mme-de-cambremer entity (nee Legrandin, the marquis's wife) and the marquis's mother, the dowager marquise. No dowager entity previously existed; created here per task ruling. The shared surface form 'marquise de Cambremer' is also added to mme-de-cambremer and is deliberately left to resolve as ambiguous per registry semantics rather than guessed."
  },
  "prince de Guermantes": {
    "aliases": [
      "prince de Guermantes",
      "Gilbert"
    ],
    "notes": "MISSING from standings despite major scenes (Dreyfus confession, V7 matinée host)."
  },
  "princesse Sherbatoff": {
    "aliases": [
      "princesse Sherbatoff",
      "Mme Sherbatoff"
    ],
    "notes": "Faithful of the little clan; MISSING from all legacy layers."
  },
  "comte de Forcheville": {
    "aliases": [
      "comte de Forcheville",
      "M. de Forcheville",
      "Forcheville"
    ],
    "notes": ""
  },
  "duc de Châtellerault": {
    "aliases": [
      "duc de Châtellerault",
      "M. de Châtellerault",
      "Châtellerault"
    ],
    "notes": ""
  },
  "Geneviève": {
    "aliases": [
      "Geneviève de Brabant",
      "Geneviève"
    ],
    "notes": ""
  },
  "la mère du narrateur": {
    "aliases": [
      "la mère du narrateur",
      "ma mère",
      "Madame",
      "maman",
      "Maman"
    ],
    "notes": ""
  },
  "le père du narrateur": {
    "aliases": [
      "le père du narrateur",
      "Monsieur votre père",
      "votre père",
      "mon père"
    ],
    "notes": ""
  },
  "marquis de Cambremer": {
    "aliases": [
      "marquis de Cambremer",
      "M. de Cambremer",
      "Cancan"
    ],
    "notes": ""
  },
  "Mme de Vaugoubert": {
    "aliases": [
      "Madame de Vaugoubert",
      "Mme de Vaugoubert"
    ],
    "notes": ""
  },
  "baron de Charlus": {
    "aliases": [
      "le baron de Charlus",
      "baron de Charlus",
      "Baron de Charlus",
      "M. de Charlus",
      "Palamède",
      "le baron",
      "Monsieur",
      "Charlus",
      "Mémé"
    ],
    "notes": ""
  },
  "Mlle d'Éporcheville": {
    "aliases": [
      "Mlle d'Éporcheville"
    ],
    "notes": ""
  },
  "M. de Chateaubriand": {
    "aliases": [
      "M. de Chateaubriand"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 6; admitted below the count>=8 threshold per explicit task ruling). Referenced figure, association source; not a scene participant by default -- writer quoted/invoked as a literary touchstone."
  },
  "docteur Cottard": {
    "aliases": [
      "le docteur Cottard",
      "docteur Cottard",
      "Cottard",
      "docteur"
    ],
    "notes": ""
  },
  "la reine de Naples": {
    "aliases": [
      "la reine de Naples"
    ],
    "notes": ""
  },
  "Legrandin": {
    "aliases": [
      "comte de Méséglise",
      "M. Legrandin",
      "Legrandin"
    ],
    "notes": ""
  },
  "marquis de Bréauté": {
    "aliases": [
      "marquis de Bréauté",
      "marquis de Breaute",
      "M. de Bréauté",
      "Breaute",
      "Bréauté"
    ],
    "notes": "Ruling 2026-08-11 (registry-audit v2 candidates): 'M. de Bréauté' (audit candidate, count 53 -- the highest-count gap in the audit) is the same man under his common address form; attached here rather than as a new entity even though it is not among the consolidations explicitly enumerated in the batch ruling, to avoid registering a duplicate for an already-confirmed entity."
  },
  "princesse de Parme": {
    "aliases": [
      "princesse de Parme",
      "Mme de Parme"
    ],
    "notes": ""
  },
  "docteur du Boulbon": {
    "aliases": [
      "docteur du Boulbon"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 12). Nerve specialist consulted about the grandmother's illness; distinct from the existing docteur Cottard entity."
  },
  "Mlle de Saint-Loup": {
    "aliases": [
      "Mlle de Saint-Loup"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 11). Daughter of Robert de Saint-Loup and Gilberte; a real in-novel character (Le Temps retrouve), not a mention-only reference."
  },
  "princesse d'Épinay": {
    "aliases": [
      "princesse d'Épinay"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Guermantes-salon figure."
  },
  "princesse Mathilde": {
    "aliases": [
      "princesse Mathilde"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 9). Referenced figure, association source; not a scene participant by default -- real historical Bonapartist salonniere, discussed (e.g. by M. de Norpois) rather than staged."
  },
  "le peintre": {
    "aliases": [
      "le peintre favori",
      "peintre"
    ],
    "notes": "Ruling 2026-08-11: person-view merges into Elstir via same_person_links (policy person_view_merge); this name-view entity is kept separate by construction so the novel's revelation ('le peintre' IS Elstir) stays visible rather than being flattened."
  },
  "prince des Laumes": {
    "aliases": [
      "prince des Laumes",
      "M. des Laumes"
    ],
    "notes": "Era identity of the future duc de Guermantes (Un amour de Swann). Ruling 2026-08-11 keeps the era ledger split (name-view preserved by construction) with an explicit person_view_merge link to duc-de-guermantes for person-view aggregation."
  },
  "duc de Guermantes": {
    "aliases": [
      "duc de Guermantes",
      "M. de Guermantes",
      "Basin",
      "duc"
    ],
    "notes": "Ruling 2026-08-11: kept separate from prince-des-laumes (name-view); linked via same_person_links on that entity (policy person_view_merge, review false) for person-view aggregation."
  },
  "Bloch": {
    "aliases": [
      "Jacques du Rozier",
      "Monsieur Bloch",
      "Bloch fils",
      "M. Bloch",
      "Bloch"
    ],
    "notes": ""
  },
  "la jeune ouvriere": {
    "aliases": [
      "la jeune ouvriere",
      "la jeune ouvrière"
    ],
    "notes": ""
  },
  "le pianiste": {
    "aliases": [
      "le jeune pianiste",
      "le petit pianiste",
      "le jeune artiste",
      "le pianiste"
    ],
    "notes": ""
  },
  "M. Nissim Bernard": {
    "aliases": [
      "M. Nissim Bernard"
    ],
    "notes": ""
  },
  "M. Verdurin": {
    "aliases": [
      "Monsieur Verdurin",
      "M. Verdurin",
      "Verdurin"
    ],
    "notes": ""
  },
  "Mlle de Stermaria": {
    "aliases": [
      "Mlle de Stermaria"
    ],
    "notes": ""
  },
  "oncle Adolphe": {
    "aliases": [
      "mon oncle Adolphe",
      "oncle Adolphe",
      "Adolphe",
      "oncle"
    ],
    "notes": ""
  },
  "Mme de Franquetot": {
    "aliases": [
      "Mme de Franquetot"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 11). Guermantes-salon and Sainte-Euverte soiree figure."
  },
  "docteur Percepied": {
    "aliases": [
      "docteur Percepied"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 10). Combray family doctor; drives the narrator past the Martinville steeples."
  },
  "Mme de Sévigné": {
    "aliases": [
      "Madame de Sévigné",
      "Mme de Sévigné"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'Mme de Sévigné' (count 36) and 'Madame de Sévigné' (count 8). Referenced figure, association source; not a scene participant by default -- the grandmother's favorite letter-writer, quoted throughout as a touchstone."
  },
  "M. de Vaugoubert": {
    "aliases": [
      "M. de Vaugoubert",
      "Vaugoubert"
    ],
    "notes": ""
  },
  "Mme de Mortemart": {
    "aliases": [
      "Mme de Mortemart"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 11). Guermantes-salon figure."
  },
  "Céleste Albaret": {
    "aliases": [
      "Céleste Albaret",
      "Céleste"
    ],
    "notes": "MISSING from all legacy layers."
  },
  "tante Léonie": {
    "aliases": [
      "ma tante Léonie",
      "Madame Octave",
      "tante Léonie",
      "Mme Octave",
      "Léonie"
    ],
    "notes": "MISSING from all legacy layers as herself; root aliases.csv wrongly merged her forms of address into Octave."
  },
  "M. de Chevregny": {
    "aliases": [
      "M. de Chevregny"
    ],
    "notes": ""
  },
  "M. de Stermaria": {
    "aliases": [
      "M. de Stermaria",
      "de Stermaria",
      "Stermaria"
    ],
    "notes": ""
  },
  "Mme Blandais": {
    "aliases": [
      "Madame Blandais",
      "Mme Blandais"
    ],
    "notes": ""
  },
  "Mme Bontemps": {
    "aliases": [
      "Madame Bontemps",
      "Mme Bontemps"
    ],
    "notes": ""
  },
  "Mme Verdurin": {
    "aliases": [
      "Madame Verdurin",
      "Mme Verdurin"
    ],
    "notes": ""
  },
  "M. d'Argencourt": {
    "aliases": [
      "M. d'Argencourt"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 47); confirmed via A/B review. Faubourg Saint-Germain habitue, part of the Guermantes salon set."
  },
  "Mme de Valcourt": {
    "aliases": [
      "Mme de Valcourt"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 9). Guermantes-salon figure."
  },
  "duc de Chartres": {
    "aliases": [
      "duc de Chartres"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Named guest at Guermantes receptions; real Orleans-family royalty, but not among the mention-only reference figures explicitly named in this batch's ruling (unlike duc d'Aumale), so admitted as a regular entity per the count threshold -- flagged in the task report as a judgment call."
  },
  "Mme de Varambon": {
    "aliases": [
      "Mme de Varambon"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Princesse de Parme's lady-in-waiting, known for the running melon-metaphor joke."
  },
  "marquis du Lau": {
    "aliases": [
      "marquis du Lau",
      "du Lau"
    ],
    "notes": ""
  },
  "Mme Cottard": {
    "aliases": [
      "Madame Cottard",
      "Mme Cottard"
    ],
    "notes": ""
  },
  "prince de Léon": {
    "aliases": [
      "prince de Léon",
      "prince de Leon",
      "Leon",
      "Léon"
    ],
    "notes": ""
  },
  "Swann": {
    "aliases": [
      "Monsieur Swann",
      "Charles Swann",
      "M. Swann",
      "Charles",
      "Swann"
    ],
    "notes": ""
  },
  "Mme Putbus": {
    "aliases": [
      "baronne Putbus",
      "Mme Putbus"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'Mme Putbus' (count 18) and 'baronne Putbus' (count 8) as one entity, same person. Never appears on stage; famous only as the mistress of her lady's maid, an object of the narrator's and Saint-Loup's desire."
  },
  "duc de Brabant": {
    "aliases": [
      "duc de Brabant"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Named guest at Guermantes receptions; real Belgian-royalty title, but not among the mention-only reference figures explicitly named in this batch's ruling, so admitted as a regular entity per the count threshold -- flagged in the task report as a judgment call. Distinct from 'Geneviève de Brabant', the legendary magic-lantern figure already in the registry (genevieve entity)."
  },
  "Mlle Vinteuil": {
    "aliases": [
      "Mlle Vinteuil"
    ],
    "notes": "MISSING from all legacy layers despite Montjouvain and the posthumous transcription arc."
  },
  "Bloch père": {
    "aliases": [
      "Bloch le père",
      "Bloch père",
      "M. Bloch"
    ],
    "notes": ""
  },
  "Mme Poncin": {
    "aliases": [
      "Madame Poncin",
      "Mme Poncin"
    ],
    "notes": ""
  },
  "Morel": {
    "aliases": [
      "Charles Morel",
      "Charlie",
      "Morel"
    ],
    "notes": ""
  },
  "Mme d'Arpajon": {
    "aliases": [
      "Mme d'Arpajon"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 38). Guermantes-salon figure, briefly the duc de Guermantes' mistress."
  },
  "Mme de Surgis": {
    "aliases": [
      "Mme de Surgis"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 34). Mme de Surgis-le-Duc; mother of two sons Charlus courts at the Guermantes soiree."
  },
  "comtesse Molé": {
    "aliases": [
      "comtesse Molé",
      "Mme Molé"
    ],
    "notes": "Registry-audit candidate 2026-08: consolidates 'comtesse Molé' (count 21) and 'Mme Molé' (count 18) as one entity, same person under different address forms. Late-era society hostess praised by Charlus."
  },
  "Mme de Souvré": {
    "aliases": [
      "Mme de Souvré"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 19). Guermantes-salon figure noted for her noncommittal politeness."
  },
  "M. de Grouchy": {
    "aliases": [
      "M. de Grouchy"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Named guest of the Faubourg Saint-Germain set."
  },
  "la Berma": {
    "aliases": [
      "Mme la Berma",
      "la Berma",
      "Berma"
    ],
    "notes": ""
  },
  "le directeur": {
    "aliases": [
      "le directeur",
      "directeur"
    ],
    "notes": ""
  },
  "le narrateur": {
    "aliases": [
      "le narrateur",
      "mon fils",
      "Marcel",
      "moi",
      "je"
    ],
    "notes": "First-person narrator; never a rewrite target."
  },
  "M. Ski": {
    "aliases": [
      "Viradobetski",
      "M. Ski",
      "Ski"
    ],
    "notes": ""
  },
  "Napoléon III": {
    "aliases": [
      "Napoléon III",
      "Napoleon III"
    ],
    "notes": ""
  },
  "duc d'Aumale": {
    "aliases": [
      "duc d'Aumale"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 13). Referenced figure, association source; not a scene participant by default -- real historical Orleans prince, invoked as a point of social/artistic reference."
  },
  "Mme Sazerat": {
    "aliases": [
      "Mme Sazerat"
    ],
    "notes": ""
  },
  "Bergotte": {
    "aliases": [
      "M. Bergotte",
      "Bergotte"
    ],
    "notes": ""
  },
  "M. Vinteuil": {
    "aliases": [
      "M. Vinteuil",
      "Vinteuil"
    ],
    "notes": ""
  },
  "M. Bontemps": {
    "aliases": [
      "M. Bontemps"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 10). Albertine's uncle, a ministry official; distinct from the existing Mme Bontemps entity (his wife, Albertine's aunt/guardian)."
  },
  "Brichot": {
    "aliases": [
      "M. Brichot",
      "Brichot"
    ],
    "notes": ""
  },
  "M. d'Orsan": {
    "aliases": [
      "M. d'Orsan",
      "d'Orsan",
      "Orsan"
    ],
    "notes": ""
  },
  "Mme Goupil": {
    "aliases": [
      "Mme Goupil"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 16). Combray notable, watched from the church pew."
  },
  "Mme Blatin": {
    "aliases": [
      "Mme Blatin"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 8). Formidable Balbec-hotel acquaintance of the grandmother."
  },
  "Rosemonde": {
    "aliases": [
      "Rosemonde"
    ],
    "notes": "Petite bande member; missing from legacy layers."
  },
  "Rémi": {
    "aliases": [
      "le cocher",
      "Rémi",
      "Remi"
    ],
    "notes": "Swann's coachman; standings currently split him across a diacritic variant (Rémi 4 matches / Remi 2 matches)."
  },
  "Elstir": {
    "aliases": [
      "M. Elstir",
      "M. Biche",
      "Elstir",
      "Biche",
      "Tiche"
    ],
    "notes": ""
  },
  "Françoise": {
    "aliases": [
      "Françoise",
      "Francoise"
    ],
    "notes": ""
  },
  "Mme Leroi": {
    "aliases": [
      "Mme Leroi"
    ],
    "notes": "Registry-audit candidate 2026-08 (count 28). Arbiter of Faubourg Saint-Germain taste, frequently discussed in the Guermantes salon."
  },
  "Théodore": {
    "aliases": [
      "Théodore"
    ],
    "notes": ""
  },
  "Saniette": {
    "aliases": [
      "Saniette"
    ],
    "notes": ""
  },
  "Eulalie": {
    "aliases": [
      "Eulalie"
    ],
    "notes": ""
  },
  "Dreyfus": {
    "aliases": [
      "Dreyfus"
    ],
    "notes": ""
  },
  "Octave": {
    "aliases": [
      "Octave"
    ],
    "notes": "The Balbec dandy ('dans les choux'), later revealed a genius; must NOT absorb tante Léonie's 'Mme Octave'."
  },
  "Gisèle": {
    "aliases": [
      "Gisèle"
    ],
    "notes": "Petite bande member; missing from legacy layers."
  },
  "Andrée": {
    "aliases": [
      "Andrée",
      "Andree"
    ],
    "notes": ""
  },
  "Jupien": {
    "aliases": [
      "Jupien"
    ],
    "notes": ""
  },
  "Aimé": {
    "aliases": [
      "Aimé",
      "Aime"
    ],
    "notes": ""
  }
}

### Prior local context (optional)

En attendant ces réalisations après coup d’un rêve auquel je ne tiendrais plus, à force d’inventer, comme au temps où je connaissais à peine Gilberte, des paroles, des lettres, où elle implorait mon pardon, avouait n’avoir jamais aimé que moi et demandait à m’épouser, une série de douces images incessamment recréées finirent par prendre plus de place dans mon esprit que la vision de Gilberte et du jeune homme, laquelle n’était plus alimentée par rien. Je serais peut-être dès lors retourné chez Mme Swann sans un rêve que je fis et où un de mes amis, lequel n’était pourtant pas de ceux que je me connaissais, agissait envers moi avec la plus grande fausseté et croyait à la mienne. Brusquement réveillé par la souffrance que venait de me causer ce rêve et voyant qu’elle persistait, je repensai à lui, cherchai à me rappeler quel était l’ami que j’avais vu en dormant et dont le nom espagnol n’était déjà plus distinct. À la fois Joseph et Pharaon, je me mis à interpréter mon rêve. Je savais que dans beaucoup d’entre eux il ne faut tenir compte ni de l’apparence des personnes, lesquelles peuvent être déguisées et avoir interchangé leurs visages, comme ces saints mutilés des cathédrales que des archéologues ignorants ont refaits, en mettant sur le corps de l’un la tête de l’autre, et en mêlant les attributs et les noms. Ceux que les êtres portent dans un rêve peuvent nous abuser. La personne que nous aimons doit y être reconnue seulement à la force de la douleur éprouvée. La mienne m’apprit que, devenue pendant mon sommeil un jeune homme, la personne dont la fausseté récente me faisait encore mal était Gilberte. Je me rappelai alors que la dernière fois que je l’avais vue, le jour où sa mère l’avait empêchée d’aller à une matinée de danse, elle avait soit sincèrement, soit en le feignant, refusé, tout en riant d’une façon étrange, de croire à mes bonnes intentions pour elle. Par association, ce souvenir en ramena un autre dans ma mémoire. Longtemps auparavant, ç’avait été Swann qui n’avait pas voulu croire à ma sincérité, ni que je fusse un bon ami pour Gilberte. Inutilement je lui avais écrit, Gilberte m’avait rapporté ma lettre et me l’avait rendue avec le même rire incompréhensible. Elle ne me l’avait pas rendue tout de suite, je me rappelai toute la scène derrière le massif de lauriers. On devient moral dès qu’on est malheureux. L’antipathie actuelle de Gilberte pour moi me sembla comme un châtiment infligé par la vie à cause de la conduite que j’avais eue ce jour-là. Les châtiments, on croit les éviter, parce qu’on fait attention aux voitures en traversant, qu’on évite les dangers. Mais il en est d’internes. L’accident vient du côté auquel on ne songeait pas, du dedans, du cœur. Les mots de Gilberte : « Si vous voulez, continuons à lutter » me firent horreur. Je l’imaginai telle, chez elle peut-être, dans la lingerie, avec le jeune homme que j’avais vu l’accompagnant dans l’avenue des Champs-Élysées. Ainsi, autant que (il y avait quelque temps) de croire que j’étais tranquillement installé dans le bonheur, j’avais été insensé, maintenant que j’avais renoncé à être heureux, de tenir pour assuré que du moins j’étais devenu, je pourrais rester calme. Car tant que notre cœur enferme d’une façon permanente l’image d’un autre être, ce n’est pas seulement notre bonheur qui peut à tout moment être détruit ; quand ce bonheur est évanoui, quand nous avons souffert, puis que nous avons réussi à endormir notre souffrance, ce qui est aussi trompeur et précaire qu’avait été le bonheur même, c’est le calme. Le mien finit par revenir, car ce qui, modifiant notre état moral, nos désirs, est entré, à la faveur d’un rêve, dans notre esprit, cela aussi peu à peu se dissipe, la permanence et la durée ne sont promises à rien, pas même à la douleur. D’ailleurs, ceux qui souffrent par l’amour sont, comme on dit de certains malades, leur propre médecin. Comme il ne peut leur venir de consolation que de l’être qui cause leur douleur et que cette douleur est une émanation de lui, c’est en elle qu’ils finissent par trouver un remède. Elle le leur découvre elle-même à un moment donné, car au fur et à mesure qu’ils la retournent en eux, cette douleur leur montre un autre aspect de la personne regrettée, tantôt si haïssable qu’on n’a même plus le désir de la revoir parce qu’avant de se plaire avec elle il faudrait la faire souffrir, tantôt si douce que la douceur qu’on lui prête on lui en fait un mérite et on en tire une raison d’espérer. Mais la souffrance qui s’était renouvelée en moi eut beau finir par s’apaiser, je ne voulus plus retourner que rarement chez Mme Swann. C’est d’abord que chez ceux qui aiment et sont abandonnés, le sentiment d’attente — même d’attente inavouée — dans lequel ils vivent se transforme de lui-même, et bien qu’en apparence identique, fait succéder à un premier état, un second exactement contraire. Le premier était la suite, le reflet des incidents douloureux qui nous avaient bouleversés. L’attente de ce qui pourrait se produire est mêlée d’effroi, d’autant plus que nous désirons à ce moment-là, si rien de nouveau ne nous vient du côté de celle que nous aimons, agir nous-mêmes, et nous ne savons trop quel sera le succès d’une démarche après laquelle il ne sera peut-être plus possible d’en entamer d’autre. Mais bientôt, sans que nous nous en rendions compte, notre attente qui continue est déterminée, nous l’avons vu, non plus par le souvenir du passé que nous avons subi, mais par l’espérance d’un avenir imaginaire. Dès lors, elle est presque agréable. Puis la première, en durant un peu, nous a habitués à vivre dans l’expectative. La souffrance que nous avons éprouvée durant nos derniers rendez-vous survit encore en nous, mais déjà ensommeillée. Nous ne sommes pas trop pressés de la renouveler, d’autant plus que nous ne voyons pas bien ce que nous demanderions maintenant. La possession d’un peu plus de la femme que nous aimons ne ferait que nous rendre plus nécessaire ce que nous ne possédons pas, et qui resterait, malgré tout, nos besoins naissant de nos satisfactions, quelque chose d’irréductible.

### Passage

Enfin une dernière raison s’ajouta plus tard à celle-ci pour me faire cesser complètement mes visites à Mme Swann. Cette raison, plus tardive, n’était pas que j’eusse encore oublié Gilberte, mais de tâcher de l’oublier plus vite. Sans doute, depuis que ma grande souffrance était finie, mes visites chez Mme Swann étaient redevenues, pour ce qui me restait de tristesse, le calmant et la distraction qui m’avaient été si précieux au début. Mais la raison de l’efficacité du premier faisait l’inconvénient de la seconde, à savoir qu’à ces visites le souvenir de Gilberte était intimement mêlé. La distraction ne m’eût été utile que si elle eût mis en lutte avec un sentiment que la présence de Gilberte n’alimentait plus, des pensées, des intérêts, des passions où Gilberte ne fût entrée pour rien. Ces états de conscience auxquels l’être qu’on aime reste étranger occupent alors une place qui, si petite qu’elle soit d’abord, est autant de retranché à l’amour qui occupait l’âme tout entière. Il faut chercher à nourrir, à faire croître ces pensées, cependant que décline le sentiment qui n’est plus qu’un souvenir, de façon que les éléments nouveaux introduits dans l’esprit lui disputent, lui arrachent une part de plus en plus grande de l’âme, et finalement la lui dérobent toute. Je me rendais compte que c’était la seule manière de tuer un amour, et j’étais encore assez jeune, assez courageux pour entreprendre de le faire, pour assumer la plus cruelle des douleurs qui naît de la certitude que, quelque temps qu’on doive y mettre, on réussira. La raison que je donnais maintenant dans mes lettres à Gilberte, de mon refus de la voir, c’était une allusion à quelque mystérieux malentendu, parfaitement fictif, qu’il y aurait eu entre elle et moi et sur lequel j’avais espéré d’abord que Gilberte me demanderait des explications. Mais, en fait, jamais, dans les relations les plus insignifiantes de la vie, un éclaircissement n’est sollicité par un correspondant qui sait qu’une phrase obscure, mensongère, incriminatrice, est mise à dessein pour qu’il proteste, et qui est trop heureux de sentir par là qu’il possède — et de garder — la maîtrise de l’initiative des opérations. À plus forte raison en est-il de même dans des relations plus tendres, où l’amour a tant d’éloquence, l’indifférence si peu de curiosité. Gilberte n’ayant pas mis en doute ni cherché à connaître ce malentendu, il devint pour moi quelque chose de réel auquel je me référais dans chaque lettre. Et il y a dans ces situations prises à faux, dans l’affectation de la froideur, un sortilège qui vous y fait persévérer. À force d’écrire : « Depuis que nos cœurs sont désunis » pour que Gilberte me répondît : « Mais ils ne le sont pas, expliquons-nous », j’avais fini par me persuader qu’ils l’étaient. En répétant toujours : « La vie a pu changer pour nous, elle n’effacera pas le sentiment que nous eûmes », par désir de m’entendre dire enfin : « Mais il n’y a rien de changé, ce sentiment est plus fort que jamais », je vivais avec l’idée que la vie avait changé en effet, que nous garderions le souvenir du sentiment qui n’était plus, comme certains nerveux pour avoir simulé une maladie finissent par rester toujours malades. Maintenant chaque fois que j’avais à écrire à Gilberte, je me reportais à ce changement imaginé et dont l’existence, désormais tacitement reconnue par le silence qu’elle gardait à ce sujet dans ses réponses, subsisterait entre nous. Puis Gilberte cessa de s’en tenir à la prétérition. Elle-même adopta mon point de vue ; et comme dans les toasts officiels, où le chef d’État qui est reçu reprend peu à peu les mêmes expressions dont vient d’user le chef d’État qui le reçoit, chaque fois que j’écrivais à Gilberte : « La vie a pu nous séparer, le souvenir du temps où nous nous connûmes durera », elle ne manqua pas de répondre : « La vie a pu nous séparer, elle ne pourra nous faire oublier les bonnes heures qui nous seront toujours chères » (nous aurions été bien embarrassés de dire pourquoi « la vie » nous avait séparés, quel changement s’était produit). Je ne souffrais plus trop. Pourtant un jour où je lui disais dans une lettre que j’avais appris la mort de notre vieille marchande de sucre d’orge des Champs-Élysées, comme je venais d’écrire ces mots : « J’ai pensé que cela vous a fait de la peine, en moi cela a remué bien des souvenirs », je ne pus m’empêcher de fondre en larmes en voyant que je parlais au passé, et comme s’il s’agissait d’un mort déjà presque oublié, de cet amour auquel malgré moi je n’avais jamais cessé de penser comme étant vivant, pouvant du moins renaître. Rien de plus tendre que cette correspondance entre amis qui ne voulaient plus se voir. Les lettres de Gilberte avaient la délicatesse de celles que j’écrivais aux indifférents, et me donnaient les mêmes marques apparentes d’affection si douces pour moi à recevoir d’elle.

D’ailleurs peu à peu chaque refus de la voir me fit moins de peine. Et comme elle me devenait moins chère, mes souvenirs douloureux n’avaient plus assez de force pour détruire dans leur retour incessant la formation du plaisir que j’avais à penser à Florence, à Venise. Je regrettais à ces moments-là d’avoir renoncé à entrer dans la diplomatie et de m’être fait une existence sédentaire pour ne pas m’éloigner d’une jeune fille que je ne verrais plus et que j’avais déjà presque oubliée. On construit sa vie pour une personne et, quand enfin on peut l’y recevoir, cette personne ne vient pas, puis meurt pour vous et on vit prisonnier dans ce qui n’était destiné qu’à elle. Si Venise semblait à mes parents bien lointain et bien fiévreux pour moi, il était du moins facile d’aller sans fatigue s’installer à Balbec. Mais pour cela il eût fallu quitter Paris, renoncer à ces visites, grâce auxquelles, si rares qu’elles fussent, j’entendais quelquefois Mme Swann me parler de sa fille. Je commençais du reste à y trouver tel ou tel plaisir où Gilberte n’était pour rien.

Quand le printemps approcha, ramenant le froid, au temps des Saints de glace et des giboulées de la Semaine Sainte, comme Mme Swann trouvait qu’on gelait chez elle, il m’arrivait souvent de la voir recevant dans des fourrures, ses mains et ses épaules frileuses disparaissant sous le blanc et brillant tapis d’un immense manchon plat et d’un collet, tous deux d’hermine, qu’elle n’avait pas quittés en rentrant et qui avaient l’air des derniers carrés des neiges de l’hiver plus persistants que les autres, et que la chaleur du feu ni le progrès de la saison n’avaient réussi à fondre. Et la vérité totale de ces semaines glaciales mais déjà fleurissantes, était suggérée pour moi dans ce salon, où bientôt je n’irais plus, par d’autres blancheurs plus enivrantes, celles, par exemple, des « boules de neige » assemblant au sommet de leurs hautes tiges nues comme les arbustes linéaires des préraphaélites, leurs globes parcellés mais unis, blancs comme des anges annonciateurs et qu’entourait une odeur de citron. Car la châtelaine de Tansonville savait qu’avril, même glacé, n’est pas dépourvu de fleurs, que l’hiver, le printemps, l’été, ne sont pas séparés par des cloisons aussi hermétiques que tend à le croire le boulevardier qui jusqu’aux premières chaleurs s’imagine le monde comme renfermant seulement des maisons nues sous la pluie. Que Mme Swann se contentât des envois que lui faisait son jardinier de Combray, et que par l’intermédiaire de sa fleuriste « attitrée » elle ne comblât pas les lacunes d’une insuffisante évocation à l’aide d’emprunts faits à la précocité méditerranéenne, je suis loin de le prétendre et je ne m’en souciais pas. Il me suffisait pour avoir la nostalgie de la campagne, qu’à côté des névés du manchon que tenait Mme Swann, les boules de neige (qui n’avaient peut-être dans la pensée de la maîtresse de la maison d’autre but que de faire, sur les conseils de Bergotte, « symphonie en blanc majeur » avec son ameublement et sa toilette) me rappelassent que l’Enchantement du Vendredi Saint figure un miracle naturel auquel on pourrait assister tous les ans si l’on était plus sage, et aidées du parfum acide et capiteux de corolles d’autres espèces dont j’ignorais les noms et qui m’avait fait rester tant de fois en arrêt dans mes promenades de Combray, rendissent le salon de Mme Swann aussi virginal, aussi candidement fleuri sans aucune feuille, aussi surchargé d’odeurs authentiques, que le petit raidillon de Tansonville.

Mais c’était encore trop que celui-ci me fût rappelé. Son souvenir risquait d’entretenir le peu qui subsistait de mon amour pour Gilberte. Aussi, bien que je ne souffrisse plus du tout durant ces visites à Mme Swann, je les espaçai encore et cherchai à la voir le moins possible. Tout au plus, comme je continuais à ne pas quitter Paris, me concédai-je certaines promenades avec elle. Les beaux jours étaient enfin revenus, et la chaleur. Comme je savais qu’avant le déjeuner Mme Swann sortait pendant une heure et allait faire quelques pas avenue du Bois, près de l’Étoile, et de l’endroit qu’on appelait alors, à cause des gens qui venaient regarder les riches qu’ils ne connaissaient que de nom, « Club des Pannés », j’obtins de mes parents que le dimanche — car je n’étais pas libre en semaine à cette heure-là — je pourrais ne déjeuner que bien après eux, à une heure un quart, et aller faire un tour auparavant. Je n’y manquai jamais pendant ce mois de mai, Gilberte étant allée à la campagne chez des amies. J’arrivais à l’Arc de Triomphe vers midi. Je faisais le guet à l’entrée de l’avenue, ne perdant pas des yeux le coin de la petite rue par où Mme Swann, qui n’avait que quelques mètres à franchir, venait de chez elle. Comme c’était déjà l’heure où beaucoup de promeneurs rentraient déjeuner, ceux qui restaient étaient peu nombreux et, pour la plus grande part, des gens élégants. Tout d’un coup, sur le sable de l’allée, tardive, alentie et luxuriante comme la plus belle fleur et qui ne s’ouvrirait qu’à midi, Mme Swann apparaissait, épanouissant autour d’elle une toilette toujours différente mais que je me rappelle surtout mauve ; puis elle hissait et déployait sur un long pédoncule, au moment de sa plus complète irradiation, le pavillon de soie d’une large ombrelle de la même nuance que l’effeuillaison des pétales de sa robe. Toute une suite l’environnait ; Swann, quatre ou cinq hommes de club qui étaient venus la voir le matin chez elle ou qu’elle avait rencontrés : et leur noire ou grise agglomération obéissante, exécutant les mouvements presque mécaniques d’un cadre inerte autour d’Odette, donnait l’air à cette femme, qui seule avait de l’intensité dans les yeux, de regarder devant elle, d’entre tous ces hommes, comme d’une fenêtre dont elle se fût approchée, et la faisait surgir, frêle, sans crainte, dans la nudité de ses tendres couleurs, comme l’apparition d’un être d’une espèce différente, d’une race inconnue, et d’une puissance presque guerrière, grâce à quoi elle compensait à elle seule sa multiple escorte. Souriante, heureuse du beau temps, du soleil qui n’incommodait pas encore, ayant l’air d’assurance et de calme du créateur qui a accompli son œuvre et ne se soucie plus du reste, certaine que sa toilette — dussent des passants vulgaires ne pas l’apprécier — était la plus élégante de toutes, elle la portait pour soi-même et pour ses amis, naturellement, sans attention exagérée, mais aussi sans détachement complet ; n’empêchant pas les petits nœuds de son corsage et de sa jupe de flotter légèrement devant elle comme des créatures dont elle n’ignorait pas la présence et à qui elle permettait avec indulgence de se livrer à leurs jeux, selon leur rythme propre, pourvu qu’ils suivissent sa marche, et même sur son ombrelle mauve que souvent elle tenait encore fermée quand elle arrivait, elle laissait tomber par moment, comme sur un bouquet de violettes de Parme, son regard heureux et si doux que quand il ne s’attachait plus à ses amis, mais à un objet inanimé, il avait l’air de sourire encore. Elle réservait ainsi, elle faisait occuper à sa toilette cet intervalle d’élégance dont les hommes à qui Mme Swann parlait le plus en camarade respectaient l’espace et la nécessité, non sans une certaine déférence de profanes, un aveu de leur propre ignorance, et sur lequel ils reconnaissaient à leur amie comme à un malade sur les soins spéciaux qu’il doit prendre, ou comme à une mère sur l’éducation de ses enfants, compétence et juridiction. Non moins que par la cour qui l’entourait et ne semblait pas voir les passants, Mme Swann, à cause de l’heure tardive de son apparition, évoquait cet appartement où elle avait passé une matinée si longue et où il faudrait qu’elle rentrât bientôt déjeuner ; elle semblait en indiquer la proximité par la tranquillité flâneuse de sa promenade, pareille à celle qu’on fait à petits pas dans son jardin ; de cet appartement on aurait dit qu’elle portait encore autour d’elle l’ombre intérieure et fraîche. Mais, par tout cela même, sa vue ne me donnait que davantage la sensation du plein air et de la chaleur. D’autant plus que déjà persuadé qu’en vertu de la liturgie et des rites dans lesquels Mme Swann était profondément versée, sa toilette était unie à la saison et à l’heure par un lien nécessaire, unique, les fleurs de son inflexible chapeau de paille, les petits rubans de sa robe me semblaient naître du mois de mai plus naturellement encore que les fleurs des jardins et des bois ; et pour connaître le trouble nouveau de la saison, je ne levais pas les yeux plus haut que son ombrelle, ouverte et tendue comme un autre ciel plus proche, rond, clément, mobile et bleu. Car ces rites, s’ils étaient souverains, mettaient leur gloire, et par conséquent Mme Swann mettait la sienne à obéir avec condescendance au matin, au printemps, au soleil, lesquels ne me semblaient pas assez flattés qu’une femme si élégante voulût bien ne pas les ignorer et eût choisi à cause d’eux une robe d’une étoffe plus claire, plus légère, faisant penser, par son évasement au col et aux manches, à la moiteur du cou et des poignets, fît enfin pour eux tous les frais d’une grande dame qui s’étant gaiement abaissée à aller voir à la campagne des gens communs et que tout le monde, même le vulgaire, connaît, n’en a pas moins tenu à revêtir spécialement pour ce jour-là une toilette champêtre. Dès son arrivée, je saluais Mme Swann, elle m’arrêtait et me disait : « Good morning » en souriant. Nous faisions quelques pas. Et je comprenais que ces canons selon lesquels elle s’habillait, c’était pour elle-même qu’elle y obéissait, comme à une sagesse supérieure dont elle eût été la grande prêtresse : car s’il lui arrivait qu’ayant trop chaud, elle entr’ouvrît, ou même ôtât tout à fait et me donnât à porter sa jaquette qu’elle avait cru garder fermée, je découvrais dans la chemisette mille détails d’exécution qui avaient eu grande chance de rester inaperçus comme ces parties d’orchestre auxquelles le compositeur a donné tous ses soins, bien qu’elles ne doivent jamais arriver aux oreilles du public ; ou dans les manches de la jaquette pliée sur mon bras je voyais, je regardais longuement, par plaisir ou par amabilité, quelque détail exquis, une bande d’une teinte délicieuse, une satinette mauve habituellement cachée aux yeux de tous, mais aussi délicatement travaillée que les parties extérieures, comme ces sculptures gothiques d’une cathédrale dissimulées au revers d’une balustrade à quatre-vingts pieds de hauteur, aussi parfaites que les bas-reliefs du grand porche, mais que personne n’avait jamais vues avant qu’au hasard d’un voyage, un artiste n’eût obtenu de monter se promener en plein ciel, pour dominer toute la ville, entre les deux tours.

Ce qui augmentait cette impression que Mme Swann se promenait dans l’avenue du Bois comme dans l’allée d’un jardin à elle, c’était — pour ces gens qui ignoraient ses habitudes de « footing » — qu’elle fût venue à pied, sans voiture qui suivît, elle que, dès le mois de mai, on avait l’habitude de voir passer avec l’attelage le plus soigné, la livrée la mieux tenue de Paris, mollement et majestueusement assise comme une déesse, dans le tiède plein air d’une immense victoria à huit ressorts. À pied, Mme Swann avait l’air, surtout avec sa démarche que ralentissait la chaleur, d’avoir cédé à une curiosité, de commettre une élégante infraction aux règles du protocole, comme ces souverains qui sans consulter personne, accompagnés par l’admiration un peu scandalisée d’une suite qui n’ose formuler une critique, sortent de leur loge pendant un gala et visitent le foyer en se mêlant pendant quelques instants aux autres spectateurs. Ainsi, entre Mme Swann et la foule, celle-ci sentait ces barrières d’une certaine sorte de richesse, lesquelles lui semblent les plus infranchissables de toutes. Le faubourg Saint-Germain a bien aussi les siennes, mais moins parlantes aux yeux et à l’imagination des « pannés ». Ceux-ci, auprès d’une grande dame plus simple, plus facile à confondre avec une petite bourgeoise, moins éloignée du peuple, n’éprouveront pas ce sentiment de leur inégalité, presque de leur indignité, qu’ils ont devant une Mme Swann. Sans doute, ces sortes de femmes ne sont pas elles-mêmes frappées comme eux du brillant appareil dont elles sont entourées, elles n’y font plus attention, mais c’est à force d’y être habituées, c’est-à-dire d’avoir fini par le trouver d’autant plus naturel, d’autant plus nécessaire, par juger les autres êtres selon qu’ils sont plus ou moins initiés à ces habitudes du luxe : de sorte que (la grandeur qu’elles laissent éclater en elles, qu’elles découvrent chez les autres, étant toute matérielle, facile à constater, longue à acquérir, difficile à compenser), si ces femmes mettent un passant au rang le plus bas, c’est de la même manière qu’elles lui sont apparues au plus haut, à savoir immédiatement, à première vue, sans appel. Peut-être cette classe sociale particulière qui comptait alors des femmes comme lady Israels mêlée à celles de l’aristocratie et Mme Swann qui devait les fréquenter un jour, cette classe intermédiaire, inférieure au faubourg Saint-Germain, puisqu’elle le courtisait, mais supérieure à ce qui n’est pas du faubourg Saint-Germain, et qui avait ceci de particulier que, déjà dégagée du monde des riches, elle était la richesse encore ; mais la richesse devenue ductile, obéissant à une destination, à une pensée artistiques, l’argent malléable, poétiquement ciselé et qui sait sourire, peut-être cette classe, du moins avec le même caractère et le même charme, n’existe-t-elle plus. D’ailleurs, les femmes qui en faisaient partie n’auraient plus aujourd’hui ce qui était la première condition de leur règne, puisque avec l’âge elles ont, presque toutes, perdu leur beauté. Or, autant que du faîte de sa noble richesse, c’était du comble glorieux de son été mûr et si savoureux encore, que Mme Swann, majestueuse, souriante et bonne, s’avançant dans l’avenue du Bois, voyait comme Hypatie, sous la lente marche de ses pieds, rouler les mondes. Des jeunes gens qui passaient la regardaient anxieusement, incertains si leurs vagues relations avec elle (d’autant plus qu’ayant à peine été présentés une fois à Swann ils craignaient qu’il ne les reconnût pas) étaient suffisantes pour qu’ils se permissent de la saluer. Et ce n’était qu’en tremblant devant les conséquences, qu’ils s’y décidaient, se demandant si leur geste audacieusement provocateur et sacrilège, attentant à l’inviolable suprématie d’une caste, n’allait pas déchaîner des catastrophes ou faire descendre le châtiment d’un dieu. Il déclenchait seulement, comme un mouvement d’horlogerie, la gesticulation de petits personnages salueurs qui n’étaient autres que l’entourage d’Odette, à commencer par Swann, lequel soulevait son tube doublé de cuir vert, avec une grâce souriante, apprise dans le faubourg Saint-Germain, mais à laquelle ne s’alliait plus l’indifférence qu’il aurait eue autrefois. Elle était remplacée (comme s’il était dans une certaine mesure pénétré des préjugés d’Odette), à la fois par l’ennui d’avoir à répondre à quelqu’un d’assez mal habillé, et par la satisfaction que sa femme connût tant de monde, sentiment mixte qu’il traduisait en disant aux amis élégants qui l’accompagnaient : « Encore un ! Ma parole je me demande où Odette va chercher tous ces gens-là ! » Cependant, ayant répondu par un signe de tête au passant alarmé déjà hors de vue, mais dont le cœur battait encore, Mme Swann se tournait vers moi : « Alors, me disait-elle, c’est fini ? Vous ne viendrez plus jamais voir Gilberte ? Je suis contente d’être exceptée et que vous ne me « dropiez » pas tout à fait. J’aime vous voir, mais j’aimais aussi l’influence que vous aviez sur ma fille. Je crois qu’elle le regrette beaucoup aussi. Enfin, je ne veux pas vous tyranniser parce que vous n’auriez qu’à ne plus vouloir me voir non plus ! » « Odette, Sagan qui vous dit bonjour », faisait remarquer Swann à sa femme. Et, en effet, le prince faisant comme dans une apothéose de théâtre, de cirque, ou dans un tableau ancien, faire front à son cheval dans une magnifique apothéose, adressait à Odette un grand salut théâtral et comme allégorique où s’amplifiait toute la chevaleresque courtoisie du grand seigneur inclinant son respect devant la Femme, fût-elle incarnée en une femme que sa mère ou sa sœur ne pourraient pas fréquenter. D’ailleurs à tout moment, reconnue au fond de la transparence liquide et du vernis lumineux de l’ombre que versait sur elle son ombrelle, Mme Swann était saluée par les derniers cavaliers attardés, comme cinématographiés au galop sur l’ensoleillement blanc de l’avenue, hommes de cercle dont les noms, célèbres pour le public — Antoine de Castellane, Adalbert de Montmorency et tant d’autres — étaient pour Mme Swann des noms familiers d’amis. Et, comme la durée moyenne de la vie — la longévité relative — est beaucoup plus grande pour les souvenirs des sensations poétiques que pour ceux des souffrances du cœur, depuis si longtemps que se sont évanouis les chagrins que j’avais alors à cause de Gilberte, il leur a survécu le plaisir que j’éprouve, chaque fois que je veux lire, en une sorte de cadran solaire, les minutes qu’il y a entre midi un quart et une heure, au mois de mai, à me revoir causant ainsi avec Mme Swann, sous son ombrelle, comme sous le reflet d’un berceau de glycines.
