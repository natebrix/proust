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

Je dis avec humilité à Robert combien on sentait peu la guerre à Paris, il me dit que même à Paris c’était quelquefois « assez inouï ». Il faisait allusion à un raid de zeppelins qu’il y avait eu la veille et il me demanda si j’avais bien vu, mais comme il m’eût parlé autrefois de quelque spectacle d’une grande beauté esthétique. Encore au front comprend-on qu’il y ait une sorte de coquetterie à dire : « C’est merveilleux, quel rose ! et ce vert pâle ! », au moment où on peut à tout instant être tué, mais ceci n’existait pas chez Saint-Loup, à Paris, à propos d’un raid insignifiant. Je lui parlai de la beauté des avions qui montaient dans la nuit. « Et peut-être encore plus de ceux qui descendent, me dit-il. Je reconnais que c’est très beau le moment où ils montent, où ils vont faire constellation et obéissent en cela à des lois tout aussi précises que celles qui régissent les constellations, car ce qui te semble un spectacle est le ralliement des escadrilles, les commandements qu’on leur donne, leur départ en chasse, etc. Mais est-ce que tu n’aimes pas mieux le moment où, définitivement assimilés aux étoiles, ils s’en détachent pour partir en chasse ou rentrer après la berloque, le moment où ils « font apocalypse », même les étoiles ne gardant plus leur place. Et ces sirènes, était-ce assez wagnérien, ce qui, du reste, était bien naturel pour saluer l’arrivée des Allemands, ça faisait très hymne national, très Wacht am Rhein, avec le Kronprinz et les princesses dans la loge impériale ; c’était à se demander si c’était bien des aviateurs et pas plutôt des Walkyries qui montaient. » Il semblait avoir plaisir à cette assimilation des aviateurs et des Walkyries et l’expliquait, d’ailleurs, par des raisons purement musicales : « Dame, c’est que la musique des sirènes était d’une Chevauchée. Il faut décidément l’arrivée des Allemands pour qu’on puisse entendre du Wagner à Paris. » À certains points de vue la comparaison n’était pas fausse. La ville semblait une masse informe et noire qui tout d’un coup passait des profondeurs de la nuit dans la lumière et dans le ciel où un à un les aviateurs s’élevaient à l’appel déchirant des sirènes, cependant que d’un mouvement plus lent, mais plus insidieux, plus alarmant, car ce regard faisait penser à l’objet invisible encore et peut-être déjà proche qu’il cherchait, les projecteurs se remuaient sans cesse, flairaient l’ennemi, le cernaient dans leurs lumières jusqu’au moment où les avions aiguillés bondiraient en chasse pour le saisir. Et escadrille après escadrille chaque aviateur s’élançait ainsi de la ville, transporté maintenant dans le ciel, pareil à une Walkyrie. Pourtant des coins de la terre, au ras des maisons, s’éclairaient et je dis à Saint-Loup que s’il avait été à la maison la veille, il aurait pu, tout en contemplant l’apocalypse dans le ciel, voir sur la terre, comme dans l’enterrement du comte d’Orgaz du Greco où ces différents plans sont parallèles, un vrai vaudeville joué par des personnages en chemise de nuit, lesquels, à cause de leurs noms célèbres, eussent mérité d’être envoyés à quelque successeur de ce Ferrari dont les notes mondaines nous avaient si souvent amusés, Saint-Loup et moi, que nous nous amusions pour nous-mêmes à en inventer. Et c’est ce que nous aurions fait encore ce jour-là comme s’il n’y avait pas la guerre, bien que sur un sujet fort « guerre » : la peur des Zeppelins — reconnu : la duchesse de Guermantes superbe en chemise de nuit, le duc de Guermantes inénarrable en pyjama rose et peignoir de bain, etc., etc. « Je suis sûr, me dit-il, que dans tous les grands hôtels on a dû voir les juives américaines en chemise, serrant sur leur sein décati le collier de perles qui leur permettra d’épouser un duc décavé. L’hôtel Ritz, ces soirs-là, doit ressembler à l’Hôtel du libre échange. »

### Passage

Je demandai à Saint-Loup si cette guerre avait confirmé ce que nous disions des guerres passées à Doncières. Je lui rappelai des propos que lui-même avait oubliés, par exemple sur les pastiches des batailles par les généraux à venir. « La feinte, lui disais-je, n’est plus guère possible dans ces opérations qu’on prépare d’avance avec de telles accumulations d’artillerie. Et ce que tu m’as dit depuis sur les reconnaissances par les avions, qu’évidemment tu ne pouvais pas prévoir, empêche l’emploi des ruses napoléoniennes. — Comme tu te trompes, me répondit-il, cette guerre, évidemment, est nouvelle par rapport aux autres et se compose elle-même de guerres successives, dont la dernière est une innovation par rapport à celle qui l’a précédée. Il faut s’adapter à une formule nouvelle de l’ennemi pour se défendre contre elle, et alors lui-même recommence à innover, mais, comme en toute chose humaine, les vieux trucs prennent toujours. Pas plus tard qu’hier au soir, le plus intelligent des critiques militaires écrivait : « Quand les Allemands ont voulu délivrer la Prusse orientale, ils ont commencé l’opération par une puissante démonstration fort au sud contre Varsovie, sacrifiant dix mille hommes pour tromper l’ennemi. Quand ils ont créé, au début de 1915, la masse de manœuvre de l’archiduc Eugène pour dégager la Hongrie menacée, ils ont répandu le bruit que cette masse était destinée à une opération contre la Serbie. C’est ainsi qu’en 1800 l’armée qui allait opérer contre l’Italie était essentiellement qualifiée d’armée de réserve et semblait destinée non à passer les Alpes, mais à appuyer les armées engagées sur les théâtres septentrionaux. La ruse d’Hindenburg attaquant Varsovie pour masquer l’attaque véritable sur les lacs de Mazurie est imitée d’un plan de Napoléon de 1812. » Tu vois que M. Bidou reproduit presque les paroles que tu me rappelles et que j’avais oubliées. Et comme la guerre n’est pas finie, ces ruses-là se reproduiront encore et réussiront, car on ne perce rien à jour, ce qui a pris une fois a pris parce que c’était bon et prendra toujours. » Et en effet, bien longtemps après cette conversation avec Saint-Loup, pendant que les regards des Alliés étaient fixés sur Pétrograd, contre laquelle capitale on croyait que les Allemands commençaient leur marche, ils préparaient la plus puissante offensive contre l’Italie. Saint-Loup me cita bien d’autres exemples de pastiches militaires, ou, si l’on croit qu’il n’y a pas un art mais une science militaire, d’application de lois permanentes. « Je ne veux pas dire, il y aurait contradiction dans les mots, ajouta Saint-Loup, que l’art de la guerre soit une science. Et s’il y a une science de la guerre, il y a diversité, dispute et contradiction entre les savants. Diversité projetée pour une part dans la catégorie du temps. Ceci est assez rassurant, car, pour autant que cela est, cela n’indique pas forcément erreur mais vérité qui évolue. » Il devait me dire plus tard : « Vois dans cette guerre l’évolution des idées sur la possibilité de la percée, par exemple. On y croit d’abord, puis on vient à la doctrine de l’invulnérabilité des fronts, puis à celle de la percée possible, mais dangereuse, de la nécessité de ne pas faire un pas en avant sans que l’objectif soit d’abord détruit (un journaliste péremptoire écrira que prétendre le contraire est la plus grande sottise qu’on puisse dire), puis, au contraire, à celle d’avancer avec une très faible préparation d’artillerie, puis on en vient à faire remonter l’invulnérabilité des fronts à la guerre de 1870 et à prétendre que c’est une idée fausse pour la guerre actuelle, donc une idée d’une vérité relative. Fausse dans la guerre actuelle à cause de l’accroissement des masses et du perfectionnement des engins (voir Bidou du 2 juillet 1918), accroissement qui d’abord avait fait croire que la prochaine guerre serait très courte, puis très longue, et enfin a fait croire de nouveau à la possibilité des décisions victorieuses. Bidou cite les Alliés sur la Somme, les Allemands vers Paris en 1918. De même à chaque conquête des Allemands on dit : le terrain n’est rien, les villes ne sont rien, ce qu’il faut c’est détruire la force militaire de l’adversaire. Puis les Allemands à leur tour adoptent cette théorie en 1918 et alors Bidou explique curieusement (2 juillet 1918) comment certains points vitaux, certains espaces essentiels s’ils sont conquis décident de la victoire. C’est, d’ailleurs, une tournure de son esprit. Il a montré comment si la Russie était bouchée sur mer elle serait défaite et qu’une armée enfermée dans une sorte de camp d’emprisonnement est destinée à périr. »

Il faut dire pourtant que si la guerre n’avait pas modifié le caractère de Saint-Loup, son intelligence, conduite par une évolution où l’hérédité entrait pour une grande part, avait pris un brillant que je ne lui avais jamais vu. Quelle distance entre le jeune blondin qui jadis était courtisé par les femmes chic ou aspirait à le devenir, et le discoureur, le doctrinaire qui ne cessait de jouer avec les mots ! À une autre génération, sur une autre tige, comme un acteur qui reprend le rôle joué jadis par Bressant ou Delaunay, il était comme un successeur — rose, blond et doré, alors que l’autre était mi-partie très noir et tout blanc — de M. de Charlus. Il avait beau ne pas s’entendre avec son oncle sur la guerre, s’étant rangé dans cette fraction de l’aristocratie qui faisait passer la France avant tout tandis que M. de Charlus était au fond défaitiste, il pouvait montrer à celui qui n’avait pas vu le « créateur du rôle » comment on pouvait exceller dans l’emploi de raisonneur. « Il paraît que Hindenbourg c’est une révélation, lui dis-je. — Une vieille révélation, me répondit-il du « tac au tac », ou une future révélation. » Il aurait fallu, au lieu de ménager l’ennemi, laisser faire Mangin, abattre l’Autriche et l’Allemagne et européaniser la Turquie au lieu de montégriniser la France. « Mais nous aurons l’aide des États-Unis, lui dis-je. — En attendant, je ne vois ici que le spectacle des États désunis. Pourquoi ne pas faire des concessions plus larges à l’Italie par la peur de déchristianiser la France ? — Si ton oncle Charlus t’entendait ! lui dis-je. Au fond tu ne serais pas fâché qu’on offense encore un peu plus le Pape, et lui pense avec désespoir au mal qu’on peut faire au trône de François-Joseph. Il se dit, d’ailleurs, en cela dans la tradition de Talleyrand et du Congrès de Vienne. — L’ère du Congrès de Vienne est révolue, me répondit-il ; à la diplomatie secrète il faut opposer la diplomatie concrète. Mon oncle est au fond un monarchiste impénitent à qui on ferait avaler des carpes comme Mme Molé ou des escarpes comme Arthur Meyer, pourvu que carpes et escarpes fussent à la Chambord. Par haine du drapeau tricolore, je crois qu’il se rangerait plutôt sous le torchon du Bonnet rouge, qu’il prendrait de bonne foi pour le Drapeau blanc. » Certes, ce n’était que des mots et Saint-Loup était loin d’avoir l’originalité quelquefois profonde de son oncle. Mais il était aussi affable et charmant de caractère que l’autre était soupçonneux et jaloux. Et il était resté charmant et rose comme à Balbec, sous tous ses cheveux d’or. La seule chose où son oncle ne l’eût pas dépassé était cet état d’esprit du faubourg Saint-Germain dont sont empreints ceux qui croient s’en être le plus détachés et qui leur donne à la fois ce respect des hommes intelligents pas nés (qui ne fleurit vraiment que dans la noblesse et rend les révolutions si injustes) et cette niaise satisfaction de soi. De par ce mélange d’humilité et d’orgueil, de curiosité d’esprit acquise et d’autorité innée, M. de Charlus et Saint-Loup, par des chemins différents et avec des opinions opposées, étaient devenus, à une génération d’intervalle, des intellectuels que toute idée nouvelle intéresse et des causeurs de qui aucun interrupteur ne peut obtenir le silence. De sorte qu’une personne un peu médiocre pouvait les trouver l’un et l’autre, selon la disposition où elle se trouvait, éblouissants ou raseurs.

Tout en me rappelant la visite de Saint-Loup j’avais marché, puis, pour aller chez Mme Verdurin, fait un long crochet ; j’étais presque au pont des Invalides. Les lumières, assez peu nombreuses (à cause des gothas), étaient allumées un peu trop tôt, car le changement d’heure avait été fait un peu trop tôt, quand la nuit venait encore assez vite, mais stabilisé pour toute la belle saison (comme les calorifères sont allumés et éteints à partir d’une certaine date), et au-dessus de la ville nocturnement éclairée, dans toute une partie du ciel — du ciel ignorant de l’heure d’été et de l’heure d’hiver, et qui ne daignait pas savoir que 8 h. ½ était devenu 9 h. ½ — dans toute une partie du ciel bleuâtre il continuait à faire un peu jour. Dans toute la partie de la ville que dominent les tours du Trocadéro, le ciel avait l’air d’une immense mer nuance de turquoise qui se retire, laissant déjà émerger toute une ligne légère de rochers noirs, peut-être même de simples filets de pêcheurs alignés les uns auprès des autres, et qui étaient de petits nuages. Mer en ce moment couleur turquoise et qui emporte avec elle, sans qu’ils s’en aperçoivent, les hommes entraînés dans l’immense révolution de la terre, de la terre sur laquelle ils sont assez fous pour continuer leurs révolutions à eux, et leurs vaines guerres, comme celle qui ensanglantait en ce moment la France. Du reste, à force de regarder le ciel paresseux et trop beau, qui ne trouvait pas digne de lui de changer son horaire et au-dessus de la ville allumée prolongeait mollement, en ces tons bleuâtres, sa journée qui s’attardait, le vertige prenait : ce n’était plus une mer étendue, mais une gradation verticale de bleus glaciers. Et les tours du Trocadéro qui semblaient si proches des degrés de turquoise devaient en être extrêmement éloignées, comme ces deux tours de certaines villes de Suisse qu’on croirait dans le lointain voisines avec la pente des cimes. Je revins sur mes pas, mais une fois quitté le pont des Invalides, il ne faisait plus jour dans le ciel, il n’y avait même guère de lumières dans la ville, et butant çà et là contre des poubelles, prenant un chemin pour un autre, je me trouvai sans m’en douter, en suivant machinalement un dédale de rues obscures, arrivé sur les boulevards. Là, l’impression d’Orient que je venais d’avoir se renouvela et, d’autre part, à l’évocation du Paris du Directoire succéda celle du Paris de 1815. Comme en 1815 c’était le défilé le plus disparate des uniformes des troupes alliées ; et, parmi elles, des Africains en jupe-culotte rouge, des Hindous enturbannés de blanc suffisaient pour que de ce Paris où je me promenais je fisse toute une imaginaire cité exotique, dans un Orient à la fois minutieusement exact en ce qui concernait les costumes et la couleur des visages, arbitrairement chimérique en ce qui concernait le décor, comme de la ville où il vivait, Carpaccio fit une Jérusalem ou une Constantinople en y assemblant une foule dont la merveilleuse bigarrure n’était pas plus colorée que celle-ci. Marchant derrière deux zouaves qui ne semblaient guère se préoccuper de lui, j’aperçus un homme gras et gros, en feutre mou, en longue houppelande et sur la figure mauve duquel j’hésitai si je devais mettre le nom d’un acteur ou d’un peintre également connus pour d’innombrables scandales sodomistes. J’étais certain en tout cas que je ne connaissais pas le promeneur, aussi fus-je bien surpris, quand ses regards rencontrèrent les miens, de voir qu’il avait l’air gêné et fit exprès de s’arrêter et de venir à moi comme un homme qui veut montrer que vous ne le surprenez nullement en train de se livrer à une occupation qu’il eût préféré laisser secrète. Une seconde je me demandai qui me disait bonjour : c’était M. de Charlus. On peut dire que pour lui l’évolution de son mal ou la révolution de son vice était à ce point extrême où la petite personnalité primitive de l’individu, ses qualités ancestrales, sont entièrement interceptées par le passage en face d’elles du défaut ou du mal générique dont ils sont accompagnés. M. de Charlus était arrivé aussi loin qu’il était possible de soi-même, ou plutôt il était lui-même si parfaitement masqué par ce qu’il était devenu et qui n’appartenait pas à lui seul, mais à beaucoup d’autres invertis, qu’à la première minute je l’avais pris pour un autre d’entre eux, derrière ces zouaves, en plein boulevard, pour un autre d’entre eux qui n’était pas M. de Charlus, qui n’était pas un grand seigneur, qui n’était pas un homme d’imagination et d’esprit et qui n’avait pour toute ressemblance avec le baron que cet air commun à eux tous, et qui maintenant chez lui, au moins avant qu’on se fût appliqué à bien regarder, couvrait tout. C’est ainsi qu’ayant voulu aller chez Mme Verdurin j’avais rencontré M. de Charlus. Et certes, je ne l’eusse pas comme autrefois trouvé chez elle ; leur brouille n’avait fait que s’aggraver et Mme Verdurin se servait même des événements présents pour le discréditer davantage. Ayant dit depuis longtemps qu’elle le trouvait usé, fini, plus démodé dans ses prétendues audaces que les plus pompiers, elle résumait maintenant cette condamnation et dégoûtait de lui toutes les imaginations en disant qu’il était « avant-guerre ». La guerre avait mis entre lui et le présent, selon le petit clan, une coupure qui le reculait dans le passé le plus mort. D’ailleurs — et ceci s’adressait plutôt au monde politique, qui était moins informé — elle le représentait comme aussi « toc », aussi « à côté » comme situation mondaine que comme valeur intellectuelle. « Il ne voit personne, personne ne le reçoit », disait-elle à M. Bontemps, qu’elle persuadait aisément. Il y avait d’ailleurs du vrai dans ces paroles. La situation de M. de Charlus avait changé. Se souciant de moins en moins du monde, s’étant brouillé par caractère quinteux et ayant, par conscience de sa valeur sociale, dédaigné de se réconcilier avec la plupart des personnes qui étaient la fleur de la société, il vivait dans un isolement relatif qui n’avait pas, comme celui où était morte Mme de Villeparisis, l’ostracisme de l’aristocratie pour cause, mais qui aux yeux du public paraissait pire pour deux raisons. La mauvaise réputation, maintenant connue, de M. de Charlus faisait croire aux gens peu renseignés que c’était pour cela que ne le fréquentaient point les gens que de son propre chef il refusait de fréquenter. De sorte que ce qui était l’effet de son humeur atrabilaire semblait celui du mépris des personnes à l’égard de qui elle s’exerçait. D’autre part, Mme de Villeparisis avait eu un grand rempart : la famille. Mais M. de Charlus avait multiplié entre elle et lui les brouilles. Elle lui avait, d’ailleurs — surtout côté vieux faubourg, côté Courvoisier — semblé inintéressante. Et il ne se doutait guère, lui qui avait fait vers l’art, par opposition aux Courvoisier, des pointes si hardies, que ce qui eût intéressé le plus en lui un Bergotte, par exemple, c’était sa parenté avec tout ce vieux faubourg, c’eût été le pouvoir de décrire la vie quasi provinciale menée par ses cousines de la rue de la Chaise, à la place du Palais-Bourbon et à la rue Garancière. Point de vue moins transcendant et plus pratique, Mme Verdurin affectait de croire qu’il n’était pas Français. « Quelle est sa nationalité exacte, est-ce qu’il n’est pas Autrichien ? demandait innocemment M. Verdurin. — Mais non, pas du tout, répondait la comtesse Molé, dont le premier mouvement obéissait plutôt au bon sens qu’à la rancune. — Mais non, il est Prussien, disait la Patronne, mais je vous le dis, je le sais, il nous l’a assez répété qu’il était membre héréditaire de la Chambre des Seigneurs de Prusse et Durchlaucht. — Pourtant la reine de Naples m’avait dit… — Vous savez que c’est une affreuse espionne, s’écriait Mme Verdurin qui n’avait pas oublié l’attitude que la souveraine déchue avait eue un soir chez elle. Je le sais et d’une façon précise, elle ne vivait que de ça. Si nous avions un gouvernement plus énergique, tout ça devrait être dans un camp de concentration. Et allez donc ! En tout cas, vous ferez bien de ne pas recevoir ce joli monde, parce que je sais que le Ministre de l’Intérieur a l’œil sur eux, votre hôtel serait surveillé. Rien ne m’enlèvera de l’idée que pendant deux ans Charlus n’a pas cessé d’espionner chez moi. » Et pensant probablement qu’on pouvait avoir un doute sur l’intérêt que pouvaient présenter pour le gouvernement allemand les rapports les plus circonstanciés sur l’organisation du petit clan, Mme Verdurin, d’un air doux et perspicace, en personne qui sait que la valeur de ce qu’elle dit ne paraîtra que plus précieuse si elle n’enfle pas la voix pour le dire : « Je vous dirai que dès le premier jour j’ai dit à mon mari : Ça ne me va pas, la façon dont cet homme s’est introduit chez moi. Ça a quelque chose de louche. Nous avions une propriété au fond d’une baie, sur un point très élevé. Il était sûrement chargé par les Allemands de préparer là une base pour leurs sous-marins. Il y avait des choses qui m’étonnaient et que maintenant je comprends. Ainsi au début il ne pouvait pas venir par le train avec les autres habitués. Moi je lui avais très gentiment proposé une chambre dans le château. Hé bien, non, il avait préféré habiter Doncières où il y avait énormément de troupe. Tout ça sentait l’espionnage à plein nez. » Pour la première des accusations dirigées contre le baron de Charlus, celle d’être passé de mode, les gens du monde ne donnaient que trop aisément raison à Mme Verdurin. En fait, ils étaient ingrats, car M. de Charlus était en quelque sorte leur poète, celui qui avait su dégager dans la mondanité ambiante une sorte de poésie où il entrait de l’histoire, de la beauté, du pittoresque, du comique, de la frivole élégance. Mais les gens du monde, incapables de comprendre cette poésie, n’en voyant aucune dans leur vie, la cherchaient ailleurs et mettaient à mille pieds au-dessus de M. de Charlus des hommes qui lui étaient infiniment inférieurs, mais qui prétendaient mépriser le monde et, en revanche, professaient des théories de sociologie et d’économie politique. M. de Charlus s’enchantait à raconter des mots involontairement lyriques, et à décrire les toilettes savamment gracieuses de la duchesse de X…, la traitant de femme sublime, ce qui le faisait considérer comme une espèce d’imbécile par des femmes du monde qui trouvaient la duchesse de X… une sotte sans intérêt, que les robes sont faites pour être portées mais sans qu’on ait l’air d’y faire aucune attention, et qui, elles, plus intelligentes, couraient à la Sorbonne ou à la Chambre, si Deschanel devait parler. Bref, les gens du monde s’étaient désengoués de M. de Charlus, non pas pour avoir trop pénétré, mais sans avoir pénétré jamais sa rare valeur intellectuelle. On le trouvait « avant-guerre », démodé, car ceux-là mêmes qui sont le plus incapables de juger les mérites sont ceux qui pour les classer adoptent le plus l’ordre de la mode ; ils n’ont pas épuisé, pas même effleuré les hommes de mérite qu’il y avait dans une génération, et maintenant il faut les condamner tous en bloc car voici l’étiquette d’une génération nouvelle, qu’on ne comprendra pas davantage. Quant à la deuxième accusation, celle de germanisme, l’esprit juste-milieu des gens du monde la leur faisait repousser, mais elle avait trouvé un interprète inlassable et particulièrement cruel en Morel qui, ayant su garder dans les journaux, et même dans le monde, la place que M. de Charlus avait, en prenant, les deux fois, autant de peine, réussi à lui faire obtenir, mais non pas ensuite à lui faire retirer, poursuivait le baron d’une haine implacable ; c’était non seulement cruel de la part de Morel, mais doublement coupable, car quelles qu’eussent été ses relations exactes avec le baron, il avait connu de lui ce qu’il cachait à tant de gens, sa profonde bonté. M. de Charlus avait été avec le violoniste d’une telle générosité, d’une telle délicatesse, lui avait montré de tels scrupules de ne pas manquer à sa parole, qu’en le quittant l’idée que Charlie avait emportée de lui n’était nullement l’idée d’un homme vicieux (tout au plus considérait-il le vice du baron comme une maladie) mais de l’homme ayant le plus d’idées élevées qu’il eût jamais connu, un homme d’une sensibilité extraordinaire, une manière de saint. Il le niait si peu que, même brouillé avec lui, il disait sincèrement à des parents : « Vous pouvez lui confier votre fils, il ne peut avoir sur lui que la meilleure influence. » Aussi quand il cherchait par ses articles à le faire souffrir, dans sa pensée ce qu’il bafouait en lui ce n’était pas le vice, c’était la vertu. Un peu avant la guerre, de petites chroniques, transparentes pour ce qu’on appelait les initiés, avaient commencé à faire le plus grand tort à M. de Charlus. De l’une intitulée : « Les mésaventures d’une douairière en us, les vieux jours de la Baronne », Mme Verdurin avait acheté cinquante exemplaires pour pouvoir la prêter à ses connaissances, et M. Verdurin, déclarant que Voltaire même n’écrivait pas mieux, en donnait lecture à haute voix. Depuis la guerre le ton avait changé. L’inversion du baron n’était pas seule dénoncée, mais aussi sa prétendue nationalité germanique : « Frau Bosch », « Frau von den Bosch » étaient les surnoms habituels de M. de Charlus. Un morceau d’un caractère poétique avait ce titre emprunté à certains airs de danse dans Beethoven : « Une Allemande ». Enfin deux nouvelles : « Oncle d’Amérique et Tante de Francfort » et « Gaillard d’arrière » lues en épreuves dans le petit clan, avaient fait la joie de Brichot lui-même qui s’était écrié : « Pourvu que très haute et très puissante Anastasie ne nous caviarde pas ! » Les articles eux-mêmes étaient plus fins que ces titres ridicules. Leur style dérivait de Bergotte mais d’une façon à laquelle seul peut-être j’étais sensible, et voici pourquoi. Les écrits de Bergotte n’avaient nullement influé sur Morel. La fécondation s’était faite d’une façon toute particulière et si rare que c’est à cause de cela seulement que je la rapporte ici. J’ai indiqué en son temps la manière si spéciale que Bergotte avait, quand il parlait, de choisir ses mots, de les prononcer. Morel, qui l’avait longtemps rencontré, avait fait de lui alors des « imitations », où il contrefaisait parfaitement sa voix, usant des mêmes mots qu’il eût pris. Or maintenant, Morel pour écrire transcrivait des conversations à la Bergotte, mais sans leur faire subir cette transposition qui en eût fait du Bergotte écrit. Peu de personnes ayant causé avec Bergotte, on ne reconnaissait pas le ton, qui différait du style. Cette fécondation orale est si rare que j’ai voulu la citer ici. Elle ne produit, d’ailleurs, que des fleurs stériles.

Morel qui était au bureau de la presse et dont personne ne connaissait la situation irrégulière affectait de trouver, son sang français bouillant dans ses veines comme le jus des raisins de Combray, que c’était peu de chose que d’être dans un bureau pendant la guerre et feignait de vouloir s’engager (alors qu’il n’avait qu’à rejoindre) pendant que Mme Verdurin faisait tout ce qu’elle pouvait pour lui persuader de rester à Paris. Certes, elle était indignée que M. de Cambremer, à son âge, fût dans un état-major, et de tout homme qui n’allait pas chez elle elle disait : « Où est-ce qu’il a encore trouvé le moyen de se cacher celui-là ? », et si on affirmait que celui-là était en première ligne depuis le premier jour, répondait sans scrupule de mentir ou peut-être par habitude de se tromper : « Mais pas du tout, il n’a pas bougé de Paris, il fait quelque chose d’à peu près aussi dangereux que de promener un ministre, c’est moi qui vous le dis, je vous en réponds, je le sais par quelqu’un qui l’a vu » ; mais pour les fidèles ce n’était pas la même chose, elle ne voulait pas les laisser partir, considérant la guerre comme une grande « ennuyeuse » qui les faisait la lâcher ; aussi faisait-elle toutes les démarches pour qu’ils restassent, ce qui lui donnerait le double plaisir de les avoir à dîner et, quand ils n’étaient pas encore arrivés ou déjà partis, de flétrir leur inaction. Encore fallait-il que le fidèle se prêtât à cet embusquage, et elle était désolée de voir Morel feindre de vouloir s’y montrer récalcitrant ; aussi lui disait-elle : « Mais si, vous servez dans ce bureau, et plus qu’au front. Ce qu’il faut, c’est d’être utile, faire vraiment partie de la guerre, en être. Il y a ceux qui en sont et les embusqués. Eh bien, vous, vous en êtes, et, soyez tranquille, tout le monde le sait, personne ne vous jette la pierre. » Telle dans des circonstances différentes, quand pourtant les hommes n’étaient pas aussi rares et qu’elle n’était pas obligée comme maintenant d’avoir surtout des femmes, si l’un d’eux perdait sa mère, elle n’hésitait pas à lui persuader qu’il pouvait sans inconvénient continuer à venir à ses réceptions. « Le chagrin se porte dans le cœur. Vous voudriez aller au bal (elle n’en donnait pas), je serais la première à vous le déconseiller, mais ici, à mes petits mercredis ou dans une baignoire, personne ne s’en étonnera. On sait bien que vous avez du chagrin… » Maintenant les hommes étaient plus rares, les deuils plus fréquents, inutiles même à les empêcher d’aller dans le monde, la guerre suffisait. Elle voulait leur persuader qu’ils étaient plus utiles à la France en restant à Paris, comme elle leur eût assuré autrefois que le défunt eût été plus heureux de les voir se distraire. Malgré tout elle avait peu d’hommes, peut-être regrettait-elle parfois d’avoir consommé avec M. de Charlus une rupture sur laquelle il n’y avait plus à revenir.

Mais si M. de Charlus et Mme Verdurin ne se fréquentaient plus, chacun — avec quelques petites différences sans grande importance — continuait, comme si rien n’avait changé, Mme Verdurin à recevoir, M. de Charlus à aller à ses plaisirs : par exemple, chez Mme Verdurin, Cottard assistait maintenant aux réceptions dans un uniforme de colonel de « l’île du Rêve », assez semblable à celui d’un amiral haïtien et sur le drap duquel un large ruban bleu ciel rappelait celui des « Enfants de Marie » ; quant à M. de Charlus, se trouvant dans une ville d’où les hommes déjà faits, qui avaient été jusqu’ici son goût, avaient disparu, il faisait comme certains Français, amateurs de femmes en France et vivant aux colonies : il avait, par nécessité d’abord, pris l’habitude et ensuite le goût des petits garçons.
