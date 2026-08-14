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

S’il pleuvait, bien que le mauvais temps n’effrayât pas Albertine qu’on voyait souvent, dans son caoutchouc, filer en bicyclette sous les averses, nous passions la journée dans le Casino où il m’eût paru ces jours-là impossible de ne pas aller. J’avais le plus grand mépris pour les demoiselles d’Ambresac qui n’y étaient jamais entrées. Et j’aidais volontiers mes amies à jouer de mauvais tours au professeur de danse. Nous subissions généralement quelques admonestations du tenancier ou des employés usurpant un pouvoir directorial, parce que mes amies, même Andrée qu’à cause de cela j’avais crue le premier jour une créature si dionysiaque et qui était au contraire frêle, intellectuelle, et cette année-là fort souffrante, mais qui obéissait malgré cela moins à l’état de santé qu’au génie de cet âge qui emporte tout et confond dans la gaîté les malades et les vigoureux, ne pouvaient pas aller au vestibule, à la salle des fêtes, sans prendre leur élan, sauter par-dessus toutes les chaises, revenir sur une glissade en gardant leur équilibre par un gracieux mouvement de bras, en chantant, mêlant tous les arts, dans cette première jeunesse, à la façon de ces poètes des anciens âges pour qui les genres ne sont pas encore séparés, et qui mêlent dans un poème épique les préceptes agricoles aux enseignements théologiques.

### Passage

Cette Andrée qui m’avait paru la plus froide le premier jour était infiniment plus délicate, plus affectueuse, plus fine qu’Albertine à qui elle montrait une tendresse caressante et douce de grande sœur. Elle venait au Casino s’asseoir à côté de moi et savait — au contraire d’Albertine — refuser un tour de valse ou même si j’étais fatigué renoncer à aller au Casino pour venir à l’hôtel. Elle exprimait son amitié pour moi, pour Albertine, avec des nuances qui prouvaient la plus délicieuse intelligence des choses du cœur, laquelle était peut-être due en partie à son état maladif. Elle avait toujours un sourire gai pour excuser l’enfantillage d’Albertine qui exprimait avec une violence naïve la tentation irrésistible qu’offraient pour elle des parties de plaisir auxquelles elle ne savait pas, comme Andrée, préférer résolument de causer avec moi… Quand l’heure d’aller à un goûter donné au golf approchait, si nous étions tous ensemble à ce moment-là, elle se préparait, puis venant à Andrée : « Hé bien, Andrée, qu’est-ce que tu attends pour venir ? tu sais que nous allons goûter au golf. — Non, je reste à causer avec lui, répondait Andrée en me désignant. — Mais tu sais que Madame Durieux t’a invitée, s’écriait Albertine, comme si l’intention d’Andrée de rester avec moi ne pouvait s’expliquer que par l’ignorance où elle devait être qu’elle avait été invitée. — Voyons, ma petite, ne sois pas tellement idiote », répondait Andrée. Albertine n’insistait pas de peur qu’on lui proposât de rester aussi. Elle secouait la tête : « Fais à ton idée, répondait-elle, comme on dit à un malade qui par plaisir se tue à petit feu, moi je me trotte, car je crois que ma montre retarde », et elle prenait ses jambes à son cou. « Elle est charmante, mais inouïe », disait Andrée en enveloppant son amie d’un sourire qui la caressait et la jugeait à la fois. Si, en ce goût du divertissement, Albertine avait quelque chose de la Gilberte des premiers temps, c’est qu’une certaine ressemblance existe, tout en évoluant, entre les femmes que nous aimons successivement, ressemblance qui tient à la fixité de notre tempérament parce que c’est lui qui les choisit, éliminant toutes celles qui ne nous seraient pas à la fois opposées et complémentaires, c’est-à-dire propres à satisfaire nos sens et à faire souffrir notre cœur. Elles sont, ces femmes, un produit de notre tempérament, une image, une projection renversée, un « négatif » de notre sensibilité. De sorte qu’un romancier pourrait, au cours de la vie de son héros, peindre presque exactement semblables ses successives amours et donner par là l’impression non de s’imiter lui-même mais de créer, puisqu’il y a moins de force dans une innovation artificielle que dans une répétition destinée à suggérer une vérité neuve. Encore devrait-il noter, dans le caractère de l’amoureux, un indice de variation qui s’accuse au fur et à mesure qu’on arrive dans de nouvelles régions, sous d’autres latitudes de la vie. Et peut-être exprimerait-il encore une vérité de plus si, peignant pour ses autres personnages des caractères, il s’abstenait d’en donner aucun à la femme aimée. Nous connaissons le caractère des indifférents, comment pourrions-nous saisir celui d’un être qui se confond avec notre vie, que bientôt nous ne séparerons plus de nous-même, sur les mobiles duquel nous ne cessons de faire d’anxieuses hypothèses, perpétuellement remaniées. S’élançant d’au delà de l’intelligence, notre curiosité de la femme que nous aimons dépasse dans sa course le caractère de cette femme, nous pourrions nous y arrêter que sans doute nous ne le voudrions pas. L’objet de notre inquiète investigation est plus essentiel que ces particularités de caractère, pareilles à ces petits losanges d’épiderme dont les combinaisons variées font l’originalité fleurie de la chair. Notre radiation intuitive les traverse et les images qu’elle nous rapporte ne sont point celles d’un visage particulier, mais représentent la morne et douloureuse universalité d’un squelette.

Comme Andrée était extrêmement riche, Albertine pauvre et orpheline, Andrée avec une grande générosité la faisait profiter de son luxe. Quant à ses sentiments pour Gisèle ils n’étaient pas tout à fait ceux que j’avais crus. On eut en effet bientôt des nouvelles de l’étudiante et, quand Albertine montra la lettre qu’elle en avait reçue, lettre destinée par Gisèle à donner des nouvelles de son voyage et de son arrivée à la petite bande en s’excusant de sa paresse de ne pas écrire encore aux autres, je fus surpris d’entendre Andrée, que je croyais brouillée à mort avec elle, dire : « Je lui écrirai demain, parce que si j’attends sa lettre d’abord, je peux attendre longtemps, elle est si négligente. » Et se tournant vers moi elle ajouta : « Vous ne la trouveriez pas très remarquable évidemment, mais c’est une si brave fille et puis j’ai vraiment une grande affection pour elle. » Je conclus que les brouilles d’Andrée ne duraient pas longtemps.

Sauf ces jours de pluie, comme nous devions aller en bicyclette sur la falaise ou dans la campagne, une heure d’avance je cherchais à me faire beau et gémissais si Françoise n’avait pas bien préparé mes affaires. Or, même à Paris, elle redressait fièrement et rageusement sa taille que l’âge commençait à courber, pour peu qu’on la trouvât en faute, elle humble, elle modeste et charmante quand son amour-propre était flatté. Comme il était le grand ressort de sa vie, la satisfaction et la bonne humeur de Françoise étaient en proportion directe de la difficulté des choses qu’on lui demandait. Celles qu’elle avait à faire à Balbec étaient si aisées qu’elle montrait presque toujours un mécontentement qui était soudain centuplé et auquel s’alliait une ironique expression d’orgueil quand je me plaignais, au moment d’aller retrouver mes amies, que mon chapeau ne fût pas brossé, ou mes cravates en ordre. Elle qui pouvait se donner tant de peine sans trouver pour cela qu’elle eût rien fait, à la simple observation qu’un veston n’était pas à sa place, non seulement elle vantait avec quel soin elle l’avait « renfermé plutôt que non pas le laisser à la poussière », mais prononçant un éloge en règle de ses travaux, déplorait que ce ne fussent guère des vacances qu’elle prenait à Balbec, qu’on ne trouverait pas une seconde personne comme elle pour mener une telle vie. « Je ne comprends pas comment qu’on peut laisser ses affaires comme ça et allez-y voir si une autre saurait se retrouver dans ce pêle et mêle. Le diable lui-même y perdrait son latin. » Ou bien elle se contentait de prendre un visage de reine, me lançant des regards enflammés, et gardait un silence rompu aussitôt qu’elle avait fermé la porte et s’était engagée dans le couloir ; il retentissait alors de propos que je devinais injurieux, mais qui restaient aussi indistincts que ceux des personnages qui débitent leurs premières paroles derrière le portant avant d’être entrés en scène. D’ailleurs, quand je me préparais ainsi à sortir avec mes amies, même si rien ne manquait et si Françoise était de bonne humeur, elle se montrait tout de même insupportable. Car se servant de plaisanteries que dans mon besoin de parler de ces jeunes filles je lui avais faites sur elles, elle prenait un air de me révéler ce que j’aurais mieux su qu’elle si cela avait été exact, mais ce qui ne l’était pas car Françoise avait mal compris. Elle avait comme tout le monde son caractère propre ; une personne ne ressemble jamais à une voie droite, mais nous étonne de ses détours singuliers et inévitables dont les autres ne s’aperçoivent pas et par où il nous est pénible d’avoir à passer. Chaque fois que j’arrivais au point : « Chapeau pas en place », « nom d’Andrée ou d’Albertine », j’étais obligé par Françoise de m’égarer dans des chemins détournés et absurdes qui me retardaient beaucoup. Il en était de même quand je faisais préparer des sandwiches au chester et à la salade et acheter des tartes que je mangerais à l’heure du goûter, sur la falaise, avec ces jeunes filles, et qu’elles auraient bien pu payer à tour de rôle si elles n’avaient été aussi intéressées, déclarait Françoise, au secours de qui venait alors tout un atavisme de rapacité et de vulgarité provinciales, et pour laquelle on eût dit que l’âme divisée de la défunte Eulalie s’était incarnée, plus gracieusement qu’en Saint-Éloi, dans les corps charmants de mes amies de la petite bande. J’entendais ces accusations avec la rage de me sentir buter à un des endroits à partir desquels le chemin rustique et familier qu’était le caractère de Françoise devenait impraticable, pas pour longtemps heureusement. Puis le veston retrouvé et les sandwiches prêts, j’allais chercher Albertine, Andrée, Rosemonde, d’autres parfois, et, à pied ou en bicyclette, nous partions.

Autrefois j’eusse préféré que cette promenade eût lieu par le mauvais temps. Alors je cherchais à retrouver dans Balbec « le pays des Cimmériens », et de belles journées étaient une chose qui n’aurait pas dû exister là, une intrusion du vulgaire été des baigneurs dans cette antique région voilée par les brumes. Mais maintenant, tout ce que j’avais dédaigné, écarté de ma vue, non seulement les effets de soleil, mais même les régates, les courses de chevaux, je l’eusse recherché avec passion pour la même raison qu’autrefois je n’aurais voulu que des mers tempétueuses, et qui était qu’elles se rattachaient, les unes comme autrefois les autres, à une idée esthétique. C’est qu’avec mes amies nous étions quelquefois allés voir Elstir, et les jours où les jeunes filles étaient là, ce qu’il avait montré de préférence, c’était quelques croquis d’après de jolies yachtswomen ou bien une esquisse prise sur un hippodrome voisin de Balbec. J’avais d’abord timidement avoué à Elstir que je n’avais pas voulu aller aux réunions qui y avaient été données. « Vous avez eu tort, me dit-il, c’est si joli et si curieux aussi. D’abord cet être particulier, le jockey, sur lequel tant de regards sont fixés, et qui devant le paddock est là morne, grisâtre dans sa casaque éclatante, ne faisant qu’un avec le cheval caracolant qu’il ressaisit, comme ce serait intéressant de dégager ses mouvements professionnels, de montrer la tache brillante qu’il fait et que fait aussi la robe des chevaux, sur le champ de courses. Quelle transformation de toutes choses dans cette immensité lumineuse d’un champ de courses où on est surpris par tant d’ombres, de reflets, qu’on ne voit que là. Ce que les femmes peuvent y être jolies ! La première réunion surtout était ravissante, et il y avait des femmes d’une extrême élégance, dans une lumière humide, hollandaise, où l’on sentait monter dans le soleil même, le froid pénétrant de l’eau. Jamais je n’ai vu de femmes arrivant en voiture ou leurs jumelles aux yeux, dans une pareille lumière qui tient sans doute à l’humidité marine. Ah ! que j’aurais aimé la rendre ; je suis revenu de ces courses, fou, avec un tel désir de travailler ! » Puis il s’extasia plus encore sur les réunions du yachting que sur les courses de chevaux, et je compris que des régates, que des meetings sportifs où des femmes bien habillées baignent dans la glauque lumière d’un hippodrome marin, pouvaient être pour un artiste moderne motifs aussi intéressants que les fêtes qu’ils aimaient tant à décrire pour un Véronèse ou un Carpaccio. « Votre comparaison est d’autant plus exacte, me dit Elstir, qu’à cause de la ville où ils peignaient, ces fêtes étaient pour une part nautiques. Seulement, la beauté des embarcations de ce temps-là résidait le plus souvent dans leur lourdeur, dans leur complication. Il y avait des joutes sur l’eau, comme ici, données généralement en l’honneur de quelque ambassade pareille à celle que Carpaccio a représentée dans la Légende de Sainte Ursule. Les navires étaient massifs, construits comme des architectures, et semblaient presque amphibies comme de moindres Venises au milieu de l’autre, quand amarrés à l’aide de ponts volants, recouverts de satin cramoisi et de tapis persans ils portaient des femmes en brocart cerise ou en damas vert, tout près des balcons incrustés de marbres multicolores où d’autres femmes se penchaient pour regarder, dans leurs robes aux manches noires à crevés blancs serrés de perles ou ornés de guipures. On ne savait plus où finissait la terre, où commençait l’eau, qu’est-ce qui était encore le palais ou déjà le navire, la caravelle, la galéasse, le Bucentaure. » Albertine écoutait avec une attention passionnée ces détails de toilette, ces images de luxe que nous décrivait Elstir. « Oh ! je voudrais bien avoir les guipures dont vous me parlez, c’est si joli le point de Venise, s’écriait-elle ; d’ailleurs j’aimerais tant aller à Venise ! »

— Vous pourrez peut-être bientôt, lui dit Elstir, contempler les étoffes merveilleuses qu’on portait là-bas. On ne les voyait plus que dans les tableaux des peintres vénitiens, ou alors très rarement dans les trésors des églises, parfois même il y en avait une qui passait dans une vente. Mais on dit qu’un artiste de Venise, Fortuny, a retrouvé le secret de leur fabrication et qu’avant quelques années les femmes pourront se promener, et surtout rester chez elles, dans des brocarts aussi magnifiques que ceux que Venise ornait, pour ses patriciennes, avec des dessins d’Orient. Mais je ne sais pas si j’aimerai beaucoup cela, si ce ne sera pas un peu trop costume anachronique, pour des femmes d’aujourd’hui, même paradant aux régates, car pour en revenir à nos bateaux modernes de plaisance, c’est tout le contraire que du temps de Venise, « Reine de l’Adriatique ». Le plus grand charme d’un yacht, de l’ameublement d’un yacht, des toilettes de yachting, est leur simplicité de choses de la mer, et j’aime tant la mer ! Je vous avoue que je préfère les modes d’aujourd’hui aux modes du temps de Véronèse et même de Carpaccio. Ce qu’il y a de joli dans nos yachts — et dans les yachts moyens surtout, je n’aime pas les énormes, trop navires, c’est comme pour les chapeaux, il y a une mesure à garder — c’est la chose unie, simple, claire, grise, qui par les temps voilés, bleuâtres, prend un flou crémeux. Il faut que la pièce où l’on se tient ait l’air d’un petit café. Les toilettes des femmes sur un yacht c’est la même chose ; ce qui est gracieux, ce sont ces toilettes légères, blanches et unies, en toile, en linon, en pékin, en coutil, qui au soleil et sur le bleu de la mer font un blanc aussi éclatant qu’une voile blanche. Il y a très peu de femmes du reste qui s’habillent bien, quelques-unes pourtant sont merveilleuses. Aux courses, Mlle Léa avait un petit chapeau blanc et une petite ombrelle blanche, c’était ravissant. Je ne sais pas ce que je donnerais pour avoir cette petite ombrelle. » J’aurais tant voulu savoir en quoi cette petite ombrelle différait des autres, et pour d’autres raisons, de coquetterie féminine, Albertine l’aurait voulu plus encore. Mais comme Françoise qui disait pour les soufflés : « C’est un tour de main », la différence était dans la coupe. « C’était, disait Elstir, tout petit, tout rond, comme un parasol chinois. » Je citai les ombrelles de certaines femmes, mais ce n’était pas cela du tout. Elstir trouvait toutes ces ombrelles affreuses. Homme d’un goût difficile et exquis, il faisait consister dans un rien, qui était tout, la différence entre ce que portaient les trois quarts des femmes et qui lui faisait horreur et une jolie chose qui le ravissait, et, au contraire de ce qui m’arrivait à moi pour qui tout luxe était stérilisant, exaltait son désir de peintre « pour tâcher de faire des choses aussi jolies ». « Tenez, voilà une petite qui a déjà compris comment étaient le chapeau et l’ombrelle, me dit Elstir en me montrant Albertine, dont les yeux brillaient de convoitise. — Comme j’aimerais être riche pour avoir un yacht, dit-elle au peintre. Je vous demanderais des conseils pour l’aménager. Quels beaux voyages je ferais ! Et comme ce serait joli d’aller aux régates de Cowes. Et une automobile ! Est-ce que vous trouvez que c’est joli, les modes des femmes pour les automobiles ? — Non, répondait Elstir, mais cela sera. D’ailleurs, il y a peu de couturiers, un ou deux, Callot, quoique donnant un peu trop dans la dentelle, Doucet, Cheruit, quelquefois Paquin. Le reste sont des horreurs. — Mais alors, il y a une différence immense entre une toilette de Callot et celle d’un couturier quelconque ? demandai-je à Albertine. — Mais énorme, mon petit bonhomme, me répondit-elle. Oh ! pardon. Seulement, hélas ! ce qui coûte trois cents francs ailleurs coûte deux mille francs chez eux. Mais cela ne se ressemble pas, cela a l’air pareil pour les gens qui n’y connaissent rien. — Parfaitement, répondit Elstir, sans aller pourtant jusqu’à dire que la différence soit aussi profonde qu’entre une statue de la cathédrale de Reims et de l’église Saint-Augustin… Tenez, à propos de cathédrales, dit-il en s’adressant spécialement à moi, parce que cela se référait à une causerie à laquelle ces jeunes filles n’avaient pas pris part et qui d’ailleurs ne les eût nullement intéressées, je vous parlais l’autre jour de l’église de Balbec comme d’une grande falaise, une grande levée des pierres du pays, mais inversement, me dit-il en me montrant une aquarelle, regardez ces falaises (c’est une esquisse prise tout près d’ici, aux Creuniers), regardez comme ces rochers puissamment et délicatement découpés font penser à une cathédrale. » En effet, on eût dit d’immenses arceaux roses. Mais peints par un jour torride, ils semblaient réduits en poussière, volatilisés par la chaleur, laquelle avait à demi bu la mer, presque passée, dans toute l’étendue de la toile, à l’état gazeux. Dans ce jour où la lumière avait comme détruit la réalité, celle-ci était concentrée dans des créatures sombres et transparentes qui par contraste donnaient une impression de vie plus saisissante, plus proche : les ombres. Altérées de fraîcheur, la plupart, désertant le large enflammé, s’étaient réfugiées au pied des rochers, à l’abri du soleil ; d’autres nageant lentement sur les eaux comme des dauphins s’attachaient aux flancs de barques en promenade dont elles élargissaient la coque, sur l’eau pâle, de leur corps verni et bleu. C’était peut-être la soif de fraîcheur communiquée par elles qui donnait le plus la sensation de la chaleur de ce jour et qui me fit m’écrier combien je regrettais de ne pas connaître les Creuniers. Albertine et Andrée assurèrent que j’avais dû y aller cent fois. En ce cas, c’était sans le savoir, ni me douter qu’un jour leur vue pourrait m’inspirer une telle soif de beauté, non pas précisément naturelle comme celle que j’avais cherchée jusqu’ici dans les falaises de Balbec, mais plutôt architecturale. Surtout moi qui, parti pour voir le royaume des tempêtes, ne trouvais jamais dans mes promenades avec Mme de Villeparisis où souvent nous ne l’apercevions que de loin, peint dans l’écartement des arbres, l’océan assez réel, assez liquide, assez vivant, donnant assez l’impression de lancer ses masses d’eau, et qui n’aurais aimé le voir immobile que sous un linceul hivernal de brume, je n’eusse guère pu croire que je rêverais maintenant d’une mer qui n’était plus qu’une vapeur blanchâtre ayant perdu la consistance et la couleur. Mais cette mer, Elstir, comme ceux qui rêvaient dans ces barques engourdies par la chaleur, en avait, jusqu’à une telle profondeur, goûté l’enchantement qu’il avait su rapporter, fixer sur sa toile, l’imperceptible reflux de l’eau, la pulsation d’une minute heureuse ; et on était soudain devenu si amoureux, en voyant ce portrait magique, qu’on ne pensait plus qu’à courir le monde pour retrouver la journée enfuie, dans sa grâce instantanée et dormante.
