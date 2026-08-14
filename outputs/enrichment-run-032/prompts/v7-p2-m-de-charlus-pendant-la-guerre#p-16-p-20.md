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

À l’heure du dîner les restaurants étaient pleins et si, passant dans la rue, je voyais un pauvre permissionnaire, échappé pour six jours au risque permanent de la mort, et prêt à repartir pour les tranchées, arrêter un instant ses yeux devant les vitrines illuminées, je souffrais comme à l’hôtel de Balbec quand les pêcheurs nous regardaient dîner, mais je souffrais davantage parce que je savais que la misère du soldat est plus grande que celle du pauvre, les réunissant toutes, et plus touchante encore parce qu’elle est plus résignée, plus noble, et que c’est d’un hochement de tête philosophe, sans haine, que, prêt à repartir pour la guerre, il disait en voyant se bousculer les embusqués retenant leurs tables : « On ne dirait pas que c’est la guerre ici. » Puis à 9 h. ½, alors que personne n’avait encore eu le temps de finir de dîner, à cause des ordonnances de police on éteignait brusquement toutes les lumières et la nouvelle bousculade des embusqués arrachant leurs pardessus aux chasseurs du restaurant où j’avais dîné avec Saint-Loup un soir de perme avait lieu à 9 h. 35 dans une mystérieuse pénombre de chambre où l’on montre la lanterne magique, ou de salle de spectacle servant à exhiber les films d’un de ces cinémas vers lesquels allaient se précipiter dîneurs et dîneuses. Mais après cette heure-là, pour ceux qui, comme moi, le soir dont je parle, étaient restés à dîner chez eux, et sortaient pour aller voir des amis, Paris était, au moins dans certains quartiers, encore plus noir que n’était le Combray de mon enfance ; les visites qu’on se faisait prenaient un air de visites de voisins de campagne. Ah ! si Albertine avait vécu, qu’il eût été doux, les soirs où j’aurais dîné en ville, de lui donner rendez-vous dehors, sous les arcades. D’abord, je n’aurais rien vu, j’aurais eu l’émotion de croire qu’elle avait manqué au rendez-vous, quand tout à coup j’eusse vu se détacher du mur noir une de ses chères robes grises, ses yeux souriants qui m’auraient aperçu, et nous aurions pu nous promener enlacés sans que personne nous distinguât, nous dérangeât et rentrer ensuite à la maison. Hélas, j’étais seul et je me faisais l’effet d’aller faire une visite de voisin à la campagne, de ces visites comme Swann venait nous en faire après le dîner, sans rencontrer plus de passants dans l’obscurité de Tansonville, par ce petit chemin de halage, jusqu’à la rue du Saint-Esprit, que je n’en rencontrais maintenant dans les rues devenues de sinueux chemins rustiques de la rue Clotilde à la rue Bonaparte. D’ailleurs, comme ces fragments de paysage, que le temps qu’il fait modifie, n’étaient plus contrariés par un cadre devenu nuisible, les soirs où le vent chassait un grain glacial je me croyais bien plus au bord de la mer furieuse, dont j’avais jadis tant rêvé, que je ne m’y étais senti à Balbec ; et même d’autres éléments de nature qui n’existaient pas jusque-là à Paris faisaient croire qu’on venait, descendant du train, d’arriver pour les vacances, en pleine campagne : par exemple le contraste de lumière et d’ombre qu’on avait à côté de soi par terre les soirs de clair de lune. Celui-ci donnait de ces effets que les villes ne connaissent pas, même en plein hiver ; ses rayons s’étalaient sur la neige qu’aucun travailleur ne déblayait plus, boulevard Haussmann, comme ils eussent fait sur un glacier des Alpes. Les silhouettes des arbres se reflétaient nettes et pures sur cette neige d’or bleuté, avec la délicatesse qu’elles ont dans certaines peintures japonaises ou dans certains fonds de Raphaël ; elles étaient allongées à terre au pied de l’arbre lui-même, comme on les voit souvent dans la nature au soleil couchant, quand celui-ci inonde et rend réfléchissantes les prairies où des arbres s’élèvent à intervalles réguliers. Mais, par un raffinement d’une délicatesse délicieuse, la prairie sur laquelle se développaient ces ombres d’arbres, légères comme des âmes, était une prairie paradisiaque, non pas verte mais d’un blanc si éclatant, à cause du clair de lune qui rayonnait sur la neige de jade, qu’on aurait dit que cette prairie était tissée seulement avec des pétales de poiriers en fleurs. Et sur les places, les divinités des fontaines publiques tenant en main un jet de glace avaient l’air de statues d’une matière double pour l’exécution desquelles l’artiste avait voulu marier exclusivement le bronze au cristal. Par ces jours exceptionnels, toutes les maisons étaient noires. Mais au printemps, au contraire, parfois de temps à autre, bravant les règlements de la police, un hôtel particulier, ou seulement un étage d’un hôtel, ou même seulement une chambre d’un étage, n’ayant pas fermé ses volets apparaissait, ayant l’air de se soutenir toute seule sur d’impalpables ténèbres, comme une projection purement lumineuse, comme une apparition sans consistance. Et la femme qu’en levant les yeux bien haut on distinguait dans cette pénombre dorée prenait, dans cette nuit où l’on était perdu et où elle-même semblait recluse, le charme mystérieux et voilé d’une vision d’Orient. Puis on passait et rien n’interrompait plus l’hygiénique et monotone piétinement rythmique dans l’obscurité.

### Passage

Je songeais que je n’avais revu depuis bien longtemps aucune des personnes dont il a été question dans cet ouvrage. En 1914, pendant les deux mois que j’avais passés à Paris, j’avais aperçu M. de Charlus et vu Bloch et Saint-Loup, ce dernier seulement deux fois. La seconde fois était certainement celle où il s’était le plus montré lui-même ; il avait effacé toutes les impressions peu agréables de manque de sincérité qu’il m’avait produites pendant le séjour à Tansonville que je viens de rapporter et j’avais reconnu en lui toutes les belles qualités d’autrefois. La première fois que je l’avais vu après la déclaration de guerre, c’est-à-dire au début de la semaine qui suivit, tandis que Bloch faisait montre des sentiments les plus chauvins, Saint-Loup n’avait pas assez d’ironie pour lui-même qui ne reprenait pas de service et j’avais été presque choqué de la violence de son ton. Saint-Loup revenait de Balbec. « Non, s’écria-t-il avec force et gaîté, tous ceux qui ne se battent pas, quelque raison qu’ils donnent, c’est qu’ils n’ont pas envie d’être tués, c’est par peur. » Et avec le même geste d’affirmation plus énergique encore que celui avec lequel il avait souligné la peur des autres, il ajouta : « Et moi, si je ne reprends pas de service, c’est tout bonnement par peur, na. » J’avais déjà remarqué chez différentes personnes que l’affectation des sentiments louables n’est pas la seule couverture des mauvais, mais qu’une plus nouvelle est l’exhibition de ces mauvais, de sorte qu’on n’ait pas l’air au moins de s’en cacher. De plus, chez Saint-Loup cette tendance était fortifiée par son habitude, quand il avait commis une indiscrétion, fait une gaffe, et qu’on aurait pu les lui reprocher, de les proclamer en disant que c’était exprès. Habitude qui, je crois bien, devait lui venir de quelque professeur à l’École de Guerre dans l’intimité de qui il avait vécu et pour qui il professait une grande admiration. Je n’eus donc aucun embarras pour interpréter cette boutade comme la ratification verbale d’un sentiment que Saint-Loup aimait mieux proclamer, puisqu’il avait dicté sa conduite et son abstention dans la guerre qui commençait. « Est-ce que tu as entendu dire, demanda-t-il en me quittant, que ma tante Oriane divorcerait ? Personnellement je n’en sais absolument rien. On dit cela de temps en temps et je l’ai entendu annoncer si souvent que j’attendrai que ce soit fait pour le croire. J’ajoute que ce serait très compréhensible ; mon oncle est un homme charmant, non seulement dans le monde, mais pour ses amis, pour ses parents. Même, d’une façon, il a beaucoup plus de cœur que ma tante qui est une sainte, mais qui le lui fait terriblement sentir. Seulement c’est un mari terrible, qui n’a jamais cessé de tromper sa femme, de l’insulter, de la brutaliser, de la priver d’argent. Ce serait si naturel qu’elle le quitte que c’est une raison pour que ce soit vrai, mais aussi pour que cela ne le soit pas parce que c’en est une pour qu’on en ait l’idée et qu’on le dise. Et puis du moment qu’elle l’a supporté si longtemps… Maintenant je sais bien qu’il y a tant de choses qu’on annonce à tort, qu’on dément, et puis qui plus tard deviennent vraies. » Cela me fit penser à lui demander s’il avait jamais été question, avant son mariage avec Gilberte, qu’il épousât Mlle de Guermantes. Il sursauta et m’assura que non, que ce n’était qu’un de ces bruits du monde, qui naissent de temps à autre on ne sait pourquoi, s’évanouissent de même et dont la fausseté ne rend pas ceux qui ont cru en eux plus prudents, dès que naît un bruit nouveau de fiançailles, de divorce, ou un bruit politique, pour y ajouter foi et le colporter. Quarante-huit heures n’étaient pas passées que certains faits que j’appris me prouvèrent que je m’étais absolument trompé dans l’interprétation des paroles de Robert : « Tous ceux qui ne sont pas au front, c’est qu’ils ont peur. » Saint-Loup avait dit cela pour briller dans la conversation, pour faire de l’originalité psychologique, tant qu’il n’était pas sûr que son engagement serait accepté. Mais il faisait pendant ce temps-là des pieds et des mains pour qu’il le fût, étant en cela moins original, au sens qu’il croyait qu’il fallait donner à ce mot, mais plus profondément français de Saint-André-des-Champs, plus en conformité avec tout ce qu’il y avait à ce moment-là de meilleur chez les Français de Saint-André-des-Champs, seigneurs, bourgeois et serfs respectueux des seigneurs ou révoltés contre les seigneurs, deux divisions également françaises de la même famille, sous-embranchement Françoise et sous-embranchement Sauton, d’où deux flèches se dirigeaient à nouveau dans une même direction, qui était la frontière. Bloch avait été enchanté d’entendre l’aveu de la lâcheté d’un nationaliste (qui l’était d’ailleurs si peu) et, comme Saint-Loup avait demandé si lui-même devait partir, avait pris une figure de grand-prêtre pour répondre : « Myope. » Mais Bloch avait complètement changé d’avis sur la guerre quelques jours après où il vint me voir affolé. Quoique « myope », il avait été reconnu bon pour le service. Je le ramenais chez lui quand nous rencontrâmes Saint-Loup qui avait rendez-vous, pour être présenté au Ministère de la Guerre à un colonel, avec un ancien officier, « M. de Cambremer », me dit-il. « Ah ! c’est vrai, mais c’est d’une ancienne connaissance que je te parle. Tu connais aussi bien que moi Cancan. » Je lui répondis que je le connaissais en effet et sa femme aussi, que je ne les appréciais qu’à demi. Mais j’étais tellement habitué, depuis que je les avais vus pour la première fois, à considérer la femme comme une personne malgré tout remarquable, connaissant à fond Schopenhauer et ayant accès, en somme, dans un milieu intellectuel qui était fermé à son grossier époux, que je fus d’abord étonné d’entendre Saint-Loup répondre : « Sa femme est idiote, je te l’abandonne. Mais lui est un excellent homme qui était doué et qui est resté fort agréable. » Par l’« idiotie » de la femme, Saint-Loup entendait sans doute le désir éperdu de celle-ci de fréquenter le grand monde, ce que le grand monde juge le plus sévèrement. Par les qualités du mari, sans doute quelque chose de celles que lui reconnaissait sa nièce quand elle le trouvait le mieux de la famille. Lui, du moins, ne se souciait pas de duchesses, mais à vrai dire c’est là une « intelligence » qui diffère autant de celle qui caractérise les penseurs, que « l’intelligence » reconnue par le public à tel homme riche « d’avoir su faire sa fortune ». Mais les paroles de Saint-Loup ne me déplaisaient pas en ce qu’elles rappelaient que la prétention avoisine la bêtise et que la simplicité a un goût un peu caché mais agréable. Je n’avais pas eu, il est vrai, l’occasion de savourer celle de M. de Cambremer. Mais c’est justement ce qui fait qu’un être est tant d’êtres différents selon les personnes qui le jugent, en dehors même des différences de jugement. De Cambremer je n’avais connu que l’écorce. Et sa saveur, qui m’était attestée par d’autres, m’était inconnue. Bloch nous quitta devant sa porte, débordant d’amertume contre Saint-Loup, lui disant qu’eux autres, « beaux fils galonnés », paradant dans les États-Majors, ne risquaient rien, et que lui, simple soldat de 2e classe, n’avait pas envie de se faire « trouer la peau » pour Guillaume. « Il paraît qu’il est gravement malade, l’Empereur Guillaume », répondit Saint-Loup. Bloch qui, comme tous les gens qui tiennent de près à la Bourse, accueillait avec une facilité particulière les nouvelles sensationnelles, ajouta : « On dit même beaucoup qu’il est mort. » À la Bourse tout souverain malade, que ce soit Edouard VII ou Guillaume II, est mort, toute ville sur le point d’être assiégée est prise. « On ne le cache, ajouta Bloch, que pour ne pas déprimer l’opinion chez les Boches. Mais il est mort dans la nuit d’hier. Mon père le tient d’une source de tout premier ordre. » Les sources de tout premier ordre étaient les seules dont tînt compte M. Bloch le père, alors que, par la chance qu’il avait, grâce à de « hautes relations », d’être en communication avec elles, il en recevait la nouvelle encore secrète que l’Extérieure allait monter ou la de Beers fléchir. D’ailleurs, si à ce moment précis se produisait une hausse sur la de Beers, ou des « offres » sur l’Extérieure, si le marché de la première était « ferme » et « actif », celui de la seconde « hésitant », « faible », et qu’on s’y tînt « sur la réserve », la source de premier ordre n’en restait pas moins une source de premier ordre. Aussi Bloch nous annonça-t-il la mort du Kaiser d’un air mystérieux et important, mais aussi rageur. Il était surtout particulièrement exaspéré d’entendre Robert dire : « l’Empereur Guillaume ». Je crois que sous le couperet de la guillotine Saint-Loup et M. de Guermantes n’auraient pas pu dire autrement. Deux hommes du monde restant seuls vivants dans une île déserte, où ils n’auraient à faire preuve de bonnes façons pour personne, se reconnaîtraient à ces traces d’éducation, comme deux latinistes citeraient correctement du Virgile. Saint-Loup n’eût jamais pu, même torturé par les Allemands, dire autrement que « l’Empereur Guillaume ». Et ce savoir-vivre est malgré tout l’indice de grandes entraves pour l’esprit. Celui qui ne sait pas les rejeter reste un homme du monde. Cette élégante médiocrité est d’ailleurs délicieuse — surtout avec tout ce qui s’y allie de générosité cachée et d’héroïsme inexprimé — à côté de la vulgarité de Bloch, à la fois pleutre et fanfaron, qui criait à Saint-Loup : « Tu ne pourrais pas dire « Guillaume » tout court ? C’est ça, tu as la frousse, déjà ici tu te mets à plat ventre devant lui ! Ah ! ça nous fera de beaux soldats à la frontière, ils lécheront les bottes des Boches. Vous êtes des galonnés qui savez parader dans un carrousel. Un point, c’est tout. » « Ce pauvre Bloch veut absolument que je ne fasse que parader », me dit Saint-Loup en souriant, quand nous eûmes quitté notre camarade. Et je sentais bien que parader n’était pas du tout ce que désirait Robert, bien que je ne me rendisse pas compte alors de ses intentions aussi exactement que je le fis plus tard quand, la cavalerie restant inactive, il obtint de servir comme officier d’infanterie, puis de chasseurs à pied, et enfin quand vint la suite qu’on lira plus loin. Mais du patriotisme de Robert, Bloch ne se rendit pas compte, simplement parce que Robert ne l’exprimait nullement. Si Bloch nous avait fait des professions de foi méchamment antimilitaristes une fois qu’il avait été reconnu « bon », il avait eu préalablement les déclarations les plus chauvines quand il se croyait réformé pour myopie. Mais ces déclarations, Saint-Loup eût été incapable de les faire ; d’abord par une espèce de délicatesse morale qui empêche d’exprimer les sentiments trop profonds et qu’on trouve tout naturels. Ma mère autrefois non seulement n’eût pas hésité une seconde à mourir pour ma grand’mère, mais aurait horriblement souffert si on l’avait empêchée de le faire. Néanmoins, il m’est impossible d’imaginer rétrospectivement dans sa bouche une phrase telle que : « Je donnerais ma vie pour ma mère. » Aussi tacite était, dans son amour de la France, Robert qu’en ce moment je trouvais beaucoup plus Saint-Loup (autant que je pouvais me représenter son père) que Guermantes. Il eût été préservé aussi d’exprimer ces sentiments-là par la qualité en quelque sorte morale de son intelligence. Il y a chez les travailleurs intelligents et vraiment sérieux une certaine aversion pour ceux qui mettent en littérature ce qu’ils font, le font valoir. Nous n’avions été ensemble ni au lycée, ni à la Sorbonne, mais nous avions séparément suivi certains cours des mêmes maîtres, et je me rappelle le sourire de Saint-Loup en parlant de ceux qui, tout en faisant un cours remarquable, voulaient se faire passer pour des hommes de génie en donnant un nom ambitieux à leurs théories. Pour peu que nous en parlions, Robert riait de bon cœur. Naturellement notre prédilection n’allait pas d’instinct aux Cottard ou aux Brichot, mais enfin nous avions une certaine considération pour les gens qui savaient à fond le grec ou la médecine et ne se croyaient pas autorisés pour cela à faire les charlatans. De même que toutes les actions de maman reposaient jadis sur le sentiment qu’elle eût donné sa vie pour sa mère, comme elle ne s’était jamais formulé ce sentiment à elle-même, en tout cas elle eût trouvé non pas seulement inutile et ridicule, mais choquant et honteux de l’exprimer aux autres ; de même il m’était impossible d’imaginer Saint-Loup (me parlant de son équipement, des courses qu’il avait à faire, de nos chances de victoire, du peu de valeur de l’armée russe, de ce que ferait l’Angleterre) prononçant une des phrases les plus éloquentes que peut dire le Ministre le plus sympathique aux députés debout et enthousiastes. Je ne peux cependant pas dire que, dans ce côté négatif qui l’empêchait d’exprimer les beaux sentiments qu’il ressentait, il n’y avait pas un effet de l’« esprit des Guermantes », comme on en a vu tant d’exemples chez Swann. Car si je le trouvais Saint-Loup surtout, il restait Guermantes aussi et par là, parmi les nombreux mobiles qui excitaient son courage, il y en avait qui n’étaient pas les mêmes que ceux de ses amis de Doncières, ces jeunes gens épris de leur métier avec qui j’avais dîné chaque soir et dont tant se firent tuer à la bataille de la Marne ou ailleurs en entraînant leurs hommes. Les jeunes socialistes qu’il pouvait y avoir à Doncières quand j’y étais, mais que je ne connaissais pas parce qu’ils ne fréquentaient pas le milieu de Saint-Loup, purent se rendre compte que les officiers de ce milieu n’étaient nullement des « aristos » dans l’acception hautainement fière et bassement jouisseuse que le « populo », les officiers sortis des rangs, les francs-maçons donnaient à ce surnom. Et pareillement d’ailleurs, ce même patriotisme, les officiers nobles le rencontrèrent pleinement chez les socialistes que je les avais entendu accuser, pendant que j’étais à Doncières, en pleine affaire Dreyfus, d’être des sans-patrie. Le patriotisme des militaires, aussi sincère, aussi profond, avait pris une forme définie qu’ils croyaient intangible et sur laquelle ils s’indignaient de voir jeter « l’opprobre », tandis que les patriotes en quelque sorte inconscients, indépendants, sans religion patriotique définie, qu’étaient les radicaux-socialistes, n’avaient pas su comprendre quelle réalité profonde vivait dans ce qu’ils croyaient de vaines et haineuses formules. Sans doute Saint-Loup comme eux s’était habitué à développer en lui, comme la partie la plus vraie de lui-même, la recherche et la conception des meilleures manœuvres en vue des plus grands succès stratégiques et tactiques, de sorte que, pour lui comme pour eux, la vie de son corps était quelque chose de relativement peu important qui pouvait être facilement sacrifié à cette partie intérieure, véritable noyau vital chez eux, autour duquel l’existence personnelle n’avait de valeur que comme un épiderme protecteur. Je parlai à Saint-Loup de son ami le directeur du Grand Hôtel de Balbec qui, paraît-il, avait prétendu qu’il y avait eu au début de la guerre dans certains régiments français des défections, qu’il appelait des « défectuosités », et avait accusé de les avoir provoquée ce qu’il appelait le « militariste prussien », disant d’ailleurs en riant à propos de son frère : « Il est dans les tranchées, ils sont à trente mètres des Boches ! » jusqu’à ce qu’ayant appris qu’il l’était lui-même on l’eût mis dans un camp de concentration. « À propos de Balbec, te rappelles-tu l’ancien liftier de l’hôtel ? » me dit en me quittant Saint-Loup sur le ton de quelqu’un qui n’avait pas trop l’air de savoir qui c’était et qui comptait sur moi pour l’éclairer. « Il s’engage et m’a écrit pour le faire entrer dans l’aviation. » Sans doute le liftier était-il las de monter dans la cage captive de l’ascenseur, et les hauteurs de l’escalier du Grand Hôtel ne lui suffisaient plus. Il allait « prendre ses galons » autrement que comme concierge, car notre destin n’est pas toujours ce que nous avions cru. « Je vais sûrement appuyer sa demande, me dit Saint-Loup. Je le disais encore à Gilberte ce matin, jamais nous n’aurons assez d’avions. C’est avec cela qu’on verra ce que prépare l’adversaire. C’est cela qui lui enlèvera le bénéfice le plus grand d’une attaque, celui de la surprise, l’armée la meilleure sera peut-être celle qui aura les meilleurs yeux. Eh bien, et la pauvre Françoise a-t-elle réussi à faire réformer son neveu ? » Mais Françoise, qui avait fait depuis longtemps tous ses efforts pour que son neveu fût réformé et qui, quand on lui avait proposé une recommandation, par la voie des Guermantes, pour le général de Saint-Joseph, avait répondu d’un ton désespéré : « Oh ! non, ça ne servirait à rien, il n’y a rien à faire avec ce vieux bonhomme-là, c’est tout ce qu’il y a de pis, il est patriotique », Françoise, dès qu’il avait été question de la guerre, et quelque douleur qu’elle en éprouvât, trouvait qu’on ne devait pas abandonner les « pauvres Russes », puisqu’on était « alliancé ». Le maître d’hôtel, persuadé d’ailleurs que la guerre ne durerait que dix jours et se terminerait par la victoire éclatante de la France, n’aurait pas osé, par peur d’être démenti par les événements, et n’aurait même pas eu assez d’imagination pour prédire une guerre longue et indécise. Mais cette victoire complète et immédiate, il tâchait au moins d’en extraire d’avance tout ce qui pouvait faire souffrir Françoise. « Ça pourrait bien faire du vilain, parce qu’il paraît qu’il y en a beaucoup qui ne veulent pas marcher, des gars de seize ans qui pleurent. » Il tâchait aussi pour la « vexer » de lui dire des choses désagréables, c’est ce qu’il appelait « lui jeter un pépin, lui lancer une apostrophe, lui envoyer un calembour ». « De seize ans, Vierge Marie », disait Françoise, et un instant méfiante : « On disait pourtant qu’on ne les prenait qu’après vingt ans, c’est encore des enfants. — Naturellement les journaux ont ordre de ne pas dire cela. Du reste, c’est toute la jeunesse qui sera en avant, il n’en reviendra pas lourd. D’un côté, ça fera du bon, une bonne saignée, là, c’est utile de temps en temps, ça fera marcher le commerce. Ah ! dame, s’il y a des gosses trop tendres qui ont une hésitation, on les fusille immédiatement, douze balles dans la peau, vlan ! D’un côté, il faut ça. Et puis, les officiers, qu’est-ce que ça peut leur faire ? Ils touchent leurs pesetas, c’est tout ce qu’ils demandent. » Françoise pâlissait tellement pendant chacune de ces conversations qu’on craignait que le maître d’hôtel ne la fît mourir d’une maladie de cœur. Elle ne perdait pas ses défauts pour cela. Quand une jeune fille venait me voir, si mal aux jambes qu’eût la vieille servante, m’arrivait-il de sortir un instant de ma chambre, je la voyais au haut d’une échelle, dans la penderie, en train, disait-elle, de chercher quelque paletot à moi pour voir si les mites ne s’y mettaient pas, en réalité pour nous écouter. Elle gardait malgré toutes mes critiques sa manière insidieuse de poser des questions d’une façon indirecte pour laquelle elle avait utilisé depuis quelque temps un certain « parce que sans doute ». N’osant pas me dire : « Est-ce que cette dame a un hôtel ? » elle me disait, les yeux timidement levés comme ceux d’un bon chien : « Parce que sans doute cette dame a un hôtel particulier… », évitant l’interrogation flagrante, moins pour être polie que pour ne pas sembler curieuse. Enfin, comme les domestiques que nous aimons le plus — surtout s’ils ne nous rendent presque plus les services et les égards de leur emploi — restent, hélas, des domestiques et marquent plus nettement les limites (que nous voudrions effacer) de leur caste au fur et à mesure qu’ils croient le plus pénétrer la nôtre, Françoise avait souvent à mon endroit (pour me piquer, eût dit le maître d’hôtel) de ces propos étranges qu’une personne du monde n’aurait pas ; avec une joie aussi dissimulée mais aussi profonde que si c’eût été une maladie grave, si j’avais chaud et que la sueur — je n’y prenais pas garde — perlât à mon front : « Mais vous êtes en nage », me disait-elle, étonnée comme devant un phénomène étrange, souriant un peu avec le mépris que cause quelque chose d’indécent, « vous sortez, mais vous avez oublié de mettre votre cravate », prenant pourtant la voix préoccupée qui est chargée d’inquiéter quelqu’un sur son état. On aurait dit que moi seul dans l’univers avais jamais été en nage. Car dans son humilité, dans sa tendre admiration pour des êtres qui lui étaient infiniment inférieurs, elle adoptait leur vilain tour de langage. Sa fille s’étant plaint d’elle à moi et m’ayant dit (je ne sais de qui elle l’avait appris) : « Elle a toujours quelque chose à dire, que je ferme mal les portes, et patati patali et patata patala », Françoise crut sans doute que son incomplète éducation seule l’avait privée jusqu’ici de ce bel usage. Et sur ses lèvres où j’avais vu fleurir jadis le français le plus pur, j’entendis plusieurs fois par jour : « Et patati patali et patata patala ». Il est du reste curieux combien non seulement les expressions mais les pensées varient peu chez une même personne. Le maître d’hôtel ayant pris l’habitude de déclarer que M. Poincaré était mal intentionné, pas pour l’argent, mais parce qu’il avait voulu absolument la guerre, il redisait cela sept à huit fois par jour devant le même auditoire habituel et toujours aussi intéressé. Pas un mot n’était modifié, pas un geste, une intonation. Bien que cela ne durât que deux minutes, c’était invariable, comme une représentation. Ses fautes de français corrompaient le langage de Françoise tout autant que les fautes de sa fille.

Elle ne dormait plus, ne mangeait plus, se faisait lire les communiqués, auxquels elle ne comprenait rien, par le maître d’hôtel qui n’y comprenait guère davantage, et chez qui le désir de tourmenter Françoise était souvent dominé par une allégresse patriotique ; il disait avec un rire sympathique, en parlant des Allemands : « Ça doit chauffer, notre vieux Joffre est en train de leur tirer des plans sur la comète. » Françoise ne comprenait pas trop de quelle comète il s’agissait, mais n’en sentait pas moins que cette phrase faisait partie des aimables et originales extravagances auxquelles une personne bien élevée doit répondre avec bonne humeur, par urbanité, et haussant gaiement les épaules d’un air de dire : « Il est bien toujours le même », elle tempérait ses larmes d’un sourire. Au moins était-elle heureuse que son nouveau garçon boucher qui, malgré son métier, était assez craintif (il avait cependant commencé dans les abattoirs) ne fût pas d’âge à partir. Sans quoi elle eût été capable d’aller trouver le Ministre de la Guerre.

Le maître d’hôtel n’eût pu imaginer que les communiqués ne fussent pas excellents et qu’on ne se rapprochât pas de Berlin, puisqu’il lisait : « Nous avons repoussé, avec de fortes pertes pour l’ennemi, etc. », actions qu’il célébrait comme de nouvelles victoires. J’étais cependant effrayé de la rapidité avec laquelle le théâtre de ces victoires se rapprochait de Paris, et je fus même étonné que le maître d’hôtel, ayant vu dans un communiqué qu’une action avait eu lieu près de Lens, n’eût pas été inquiet en voyant dans le journal du lendemain que ses suites avaient tourné à notre avantage à Jouy-le-Vicomte, dont nous tenions solidement les abords. Le maître d’hôtel savait, connaissait pourtant bien le nom, Jouy-le-Vicomte, qui n’était pas tellement éloigné de Combray. Mais on lit les journaux comme on aime, un bandeau sur les yeux. On ne cherche pas à comprendre les faits. On écoute les douces paroles du rédacteur en chef, comme on écoute les paroles de sa maîtresse. On est battu et content parce qu’on ne se croit pas battu, mais vainqueur.

Je n’étais pas, du reste, demeuré longtemps à Paris et j’avais regagné assez vite ma maison de santé. Bien qu’en principe le docteur nous traitât par l’isolement, on m’y avait remis à deux époques différentes une lettre de Gilberte et une lettre de Robert. Gilberte m’écrivait (c’était à peu près en septembre 1914) que, quelque désir qu’elle eût de rester à Paris pour avoir plus facilement des nouvelles de Robert, les raids perpétuels de taubes au-dessus de Paris lui avaient causé une telle épouvante, surtout pour sa petite fille, qu’elle s’était enfuie de Paris par le dernier train qui partait encore pour Combray, que le train n’était même pas allé à Combray et que ce n’était que grâce à la charrette d’un paysan sur laquelle elle avait fait dix heures d’un trajet atroce, qu’elle avait pu gagner Tansonville ! « Et là, imaginez-vous ce qui attendait votre vieille amie, m’écrivait en finissant Gilberte. J’étais partie de Paris pour fuir les avions allemands, me figurant qu’à Tansonville je serais à l’abri de tout. Je n’y étais pas depuis deux jours que vous n’imaginerez jamais ce qui arrivait : les Allemands qui envahissaient la région après avoir battu nos troupes près de La Fère, et un état-major allemand suivi d’un régiment qui se présentait à la porte de Tansonville, et que j’étais obligée d’héberger, et pas moyen de fuir, plus un train, rien. » L’état-major allemand s’était-il bien conduit, ou fallait-il voir dans la lettre de Gilberte un effet par contagion de l’esprit des Guermantes, lesquels étaient de souche bavaroise, apparentée à la plus haute aristocratie d’Allemagne, mais Gilberte ne tarissait pas sur la parfaite éducation de l’état-major, et même des soldats qui lui avaient seulement demandé « la permission de cueillir un des ne-m’oubliez-pas qui poussaient auprès de l’étang », bonne éducation qu’elle opposait à la violence désordonnée des fuyards français, qui avaient traversé la propriété en saccageant tout, avant l’arrivée des généraux allemands. En tout cas, si la lettre de Gilberte était par certains côtés imprégnée de l’esprit des Guermantes — d’autres diraient de l’internationalisme juif, ce qui n’aurait probablement pas été juste, comme on verra — la lettre que je reçus pas mal de mois plus tard de Robert était, elle, beaucoup plus Saint-Loup que Guermantes, reflétant de plus toute la culture libérale qu’il avait acquise, et, en somme, entièrement sympathique. Malheureusement il ne me parlait pas de stratégie comme dans ses conversations de Doncières et ne me disait pas dans quelle mesure il estimait que la guerre confirmât ou infirmât les principes qu’il m’avait alors exposés. Tout au plus me dit-il que depuis 1914 s’étaient en réalité succédé plusieurs guerres, les enseignements de chacune influant sur la conduite de la suivante. Et, par exemple, la théorie de la « percée » avait été complétée par cette thèse qu’il fallait avant de percer bouleverser entièrement par l’artillerie le terrain occupé par l’adversaire. Mais ensuite on avait constaté qu’au contraire ce bouleversement rendait impossible l’avance de l’infanterie et de l’artillerie dans des terrains dont des milliers de trous d’obus avaient fait autant d’obstacles. « La guerre, disait-il, n’échappe pas aux lois de notre vieil Hegel. Elle est en état de perpétuel devenir. » C’était peu auprès de ce que j’aurais voulu savoir. Mais ce qui me fâchait davantage encore c’est qu’il n’avait plus le droit de me citer de noms de généraux. Et d’ailleurs, par le peu que me disait le journal, ce n’était pas ceux dont j’étais à Doncières si préoccupé de savoir lesquels montreraient le plus de valeur dans une guerre, qui conduisaient celle-ci. Geslin de Bourgogne, Galliffet, Négrier étaient morts. Pau avait quitté le service actif presque au début de la guerre. De Joffre, de Foch, de Castelnau, de Pétain, nous n’avions jamais parlé. « Mon petit, m’écrivait Robert, si tu voyais tout ce monde, surtout les gens du peuple, les ouvriers, les petits commerçants, qui ne se doutaient pas de ce qu’ils recelaient en eux d’héroïsme et seraient morts dans leur lit sans l’avoir soupçonné, courir sous les balles pour secourir un camarade, pour emporter un chef blessé, et, frappés eux-mêmes, sourire au moment où ils vont mourir parce que le médecin-chef leur apprend que la tranchée a été reprise aux Allemands, je t’assure, mon cher petit, que cela donne une belle idée du Français et que ça fait comprendre les époques historiques qui nous paraissaient un peu extraordinaires dans nos classes. L’époque est tellement belle que tu trouverais comme moi que les mots ne sont plus rien. Au contact d’une telle grandeur, le mot « poilu » est devenu pour moi quelque chose dont je ne sens pas plus s’il a pu contenir d’abord une allusion ou une plaisanterie que quand nous lisons « chouans » par exemple. Mais je sais « poilu » déjà prêt pour de grands poètes, comme les mots déluge, ou Christ, ou barbares qui étaient déjà pétris de grandeur avant que s’en fussent servis Hugo, Vigny, ou les autres. Je dis que le peuple est ce qu’il y a de mieux, mais tout le monde est bien. Le pauvre Vaugoubert, le fils de l’ambassadeur, a été sept fois blessé avant d’être tué, et chaque fois qu’il revenait d’une expédition sans avoir écopé, il avait l’air de s’excuser et de dire que ce n’était pas sa faute. C’était un être charmant. Nous nous étions beaucoup liés, les pauvres parents ont eu la permission de venir à l’enterrement, à condition de ne pas être en deuil et de ne rester que cinq minutes à cause du bombardement. La mère, un grand cheval que tu connais peut-être, pouvait avoir beaucoup de chagrin, on ne distinguait rien. Mais le pauvre père était dans un tel état que je t’assure que moi, qui ai fini par devenir tout à fait insensible à force de prendre l’habitude de voir la tête du camarade, qui est en train de me parler, subitement labourée par une torpille ou même détachée du tronc, je ne pouvais pas me contenir en voyant l’effondrement du pauvre Vaugoubert qui n’était plus qu’une espèce de loque. Le Général avait beau lui dire que c’était pour la France, que son fils s’était conduit en héros, cela ne faisait que redoubler les sanglots du pauvre homme qui ne pouvait pas se détacher du corps de son fils. Enfin, et c’est pour cela qu’il faut se dire qu’« ils ne passeront pas », tous ces gens-là, comme mon pauvre valet de chambre, comme Vaugoubert, ont empêché les Allemands de passer. Tu trouves peut-être que nous n’avançons pas beaucoup, mais il ne faut pas raisonner, une armée se sent victorieuse par une impression intime, comme un mourant se sent foutu. Or nous savons que nous aurons la victoire et nous la voulons pour dicter la paix juste, je ne veux pas dire seulement pour nous, vraiment juste, juste pour les Français, juste pour les Allemands. »

De même que les héros d’un esprit médiocre et banal écrivant des poèmes pendant leur convalescence se plaçaient pour décrire la guerre non au niveau des événements, qui en eux-mêmes ne sont rien, mais de la banale esthétique, dont ils avaient suivi les règles jusque-là, parlant, comme ils eussent fait dix ans plus tôt, de la « sanglante aurore », du « vol frémissant de la victoire », etc., Saint-Loup, lui, beaucoup plus intelligent et artiste, restait intelligent et artiste, et notait avec goût pour moi des paysages pendant qu’il était immobilisé à la lisière d’une forêt marécageuse, mais comme si ç’avait été pour une chasse au canard. Pour me faire comprendre certaines oppositions d’ombre et de lumière qui avaient été « l’enchantement de sa matinée », il me citait certains tableaux que nous aimions l’un et l’autre et ne craignait pas de faire allusion à une page de Romain Rolland, voire de Nietzsche, avec cette indépendance des gens du front qui n’avaient pas la même peur de prononcer un nom allemand que ceux de l’arrière, et même avec cette pointe de coquetterie à citer un ennemi que mettait, par exemple, le colonel du Paty de Clam, dans la salle des témoins de l’affaire Zola, à réciter en passant devant Pierre Quillard, poète dreyfusard de la plus extrême violence et que, d’ailleurs, il ne connaissait pas, des vers de son drame symboliste : La Fille aux mains coupées. Saint-Loup me parlait-il d’une mélodie de Schumann, il n’en donnait le titre qu’en allemand et ne prenait aucune circonlocution pour me dire que quand, à l’aube, il avait entendu un premier gazouillement à la lisière d’une forêt, il avait été enivré comme si lui avait parlé l’oiseau de ce « sublime Siegfried » qu’il espérait bien entendre après la guerre.
