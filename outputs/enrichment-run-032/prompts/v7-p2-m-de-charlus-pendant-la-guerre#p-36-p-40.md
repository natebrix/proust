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

Mais enfin, je ne peux que supposer ce que j’aurais fait si je n’avais pas été acteur, si je n’avais pas été une partie de l’acteur France, comme dans mes querelles avec Albertine, où mon regard triste et ma gorge oppressée étaient une partie de mon individu passionnément intéressé à ma cause, je ne pouvais arriver au détachement. Celui de M. de Charlus était complet. Or, dès lors qu’il n’était plus qu’un spectateur, tout devait le porter à être germanophile, du moment que, n’étant pas véritablement français, il vivait en France. Il était très fin, les sots sont en tous pays les plus nombreux ; nul doute que, vivant en Allemagne, les sots d’Allemagne défendant avec sottise et passion une cause injuste ne l’eussent irrité ; mais vivant en France, les sots français défendant avec sottise et passion une cause juste ne l’irritaient pas moins. La logique de la passion, fût-elle au service du meilleur droit, n’est jamais irréfutable pour celui qui n’est pas passionné. M. de Charlus relevait avec finesse chaque faux raisonnement des patriotes. La satisfaction que cause à un imbécile son bon droit et la certitude du succès vous laissent particulièrement irrité. M. de Charlus l’était par l’optimisme triomphant de gens qui ne connaissaient pas comme lui l’Allemagne et sa force, qui croyaient chaque mois à un écrasement pour le mois suivant, et au bout d’un an n’étaient pas moins assurés dans un nouveau pronostic, comme s’ils n’en avaient pas porté, avec tout autant d’assurance, d’aussi faux, mais qu’ils avaient oubliés disant, si on le leur rappelait, que « ce n’était pas la même chose ». Or, M. de Charlus, qui avait certaines profondeurs dans l’esprit, n’eût peut-être pas compris en Art que le « ce n’est pas la même chose » opposé par les détracteurs de Monet à ceux qui leur disent « on a dit la même chose pour Delacroix », répondait à la même tournure d’esprit. Enfin M. de Charlus était pitoyable, l’idée d’un vaincu lui faisait mal, il était toujours pour le faible, il ne lisait pas les chroniques judiciaires pour ne pas avoir à souffrir dans sa chair des angoisses du condamné et de l’impossibilité d’assassiner le juge, le bourreau, et la foule ravie de voir que « justice est faite ». Il était certain, en tout cas, que la France ne pouvait plus être vaincue, et, en revanche, il savait que les Allemands souffraient de la famine, seraient obligés un jour ou l’autre de se rendre à merci. Cette idée elle aussi lui était rendue plus désagréable par ce fait qu’il vivait en France. Ses souvenirs de l’Allemagne étaient malgré tout lointains, tandis que les Français qui parlaient de l’écrasement de l’Allemagne avec une joie qui lui déplaisait, c’étaient des gens dont les défauts lui étaient connus, la figure antipathique. Dans ces cas-là on plaint plus ceux qu’on ne connaît pas, ceux qu’on imagine, que ceux qui sont tout près de nous dans la vulgarité de la vie quotidienne, à moins alors d’être tout à fait ceux-là, de ne faire qu’une chair avec eux ; le patriotisme fait ce miracle, on est pour son pays comme on est pour soi-même dans une querelle amoureuse. Aussi la guerre était-elle pour M. de Charlus une culture extraordinairement féconde de ces haines qui chez lui naissaient en un instant, avaient une durée très courte mais pendant laquelle il se fût livré à toutes les violences. En lisant les journaux, l’air de triomphe des chroniqueurs présentant chaque jour l’Allemagne à bas : « La Bête aux abois, réduite à l’impuissance », alors que le contraire n’était que trop vrai, l’enivrait de rage par leur sottise allègre et féroce. Les journaux étaient en partie rédigés à ce moment-là par des gens connus qui trouvaient là une manière de « reprendre du service », par des Brichot, par des Norpois, par des Legrandin. M. de Charlus rêvait de les rencontrer, de les accabler des plus amers sarcasmes. Toujours particulièrement instruit des tares sexuelles, il les connaissait chez quelques-uns qui, pensant qu’elles étaient ignorées chez eux, se complaisaient à les dénoncer chez les souverains des « Empires de proie », chez Wagner, etc. Il brûlait de se trouver face à face avec eux, de leur mettre le nez dans leur propre vice devant tout le monde et de laisser ces insulteurs d’un vaincu, déshonorés et pantelants. M. de Charlus enfin avait encore des raisons plus particulières d’être ce germanophile. L’une était qu’homme du monde, il avait beaucoup vécu parmi les gens du monde, parmi les gens honorables, parmi les hommes d’honneur, de ces gens qui ne serreront pas la main à une fripouille, il connaissait leur délicatesse et leur dureté ; il les savait insensibles aux larmes d’un homme qu’ils font chasser d’un cercle ou avec qui ils refusent de se battre, dût leur acte de « propreté morale » amener la mort de la mère de la brebis galeuse. Malgré lui, quelque admiration qu’il eût pour l’Angleterre, cette Angleterre impeccable, incapable de mensonge, empêchant le blé et le lait d’entrer en Allemagne, c’était un peu cette nation d’hommes d’honneur, de témoins patentés, d’arbitres en affaires d’honneur ; tandis qu’il savait que des gens tarés, des fripouilles comme certains personnages de Dostoïewski peuvent être meilleurs, et je n’ai jamais pu comprendre pourquoi il leur identifiait les Allemands, le mensonge et la ruse ne leur suffisant pas pour faire préjuger un bon cœur qu’il ne semble pas que les Allemands aient montré. Enfin, un dernier trait complétera cette germanophilie de M. de Charlus : il la devait, et par une réaction très bizarre, à son « charlisme ». Il trouvait les Allemands fort laids, peut-être parce qu’ils étaient un peu trop près de son sang ; il était fou des Marocains, mais surtout des Anglo-Saxons en qui il voyait comme des statues vivantes de Phidias. Or, chez lui, le plaisir n’allait pas sans une certaine idée cruelle dont je ne savais pas encore à ce moment-là toute la force ; l’homme qu’il aimait lui apparaissait comme un délicieux bourreau. Il eût cru, en prenant parti contre les Allemands, agir comme il n’agissait que dans les heures de volupté, c’est-à-dire en sens contraire de sa nature pitoyable, c’est-à-dire enflammée pour le mal séduisant et écrasant la vertueuse laideur. Il en fut encore ainsi au moment du meurtre de Raspoutine, meurtre auquel on fut surpris, d’ailleurs, de trouver un si fort cachet de couleur russe, dans un souper à la Dostoïewski (impression qui eût été encore bien plus forte si le public n’avait pas ignoré de tout cela ce que savait parfaitement M. de Charlus), parce que la vie nous déçoit tellement que nous finissons par croire que la littérature n’a aucun rapport avec elle et que nous sommes stupéfaits de voir que les précieuses idées que les livres nous ont montrées s’étalent, sans peur de s’abîmer, gratuitement, naturellement, en pleine vie quotidienne et, par exemple, qu’un souper, un meurtre, événement russe, ont quelque chose de russe.

### Passage

La guerre se prolongeait indéfiniment et ceux qui avaient annoncé de source sûre, il y avait déjà plusieurs années, que les pourparlers de paix étaient commencés, spécifiant les clauses du traité, ne prenaient pas la peine, quand ils causaient avec vous, de s’excuser de leurs fausses nouvelles. Ils les avaient oubliées et étaient prêts à en propager sincèrement d’autres, qu’ils oublieraient aussi vite. C’était l’époque où il y avait continuellement des raids de gothas ; l’air grésillait perpétuellement d’une vibration vigilante et sonore d’aéroplanes français. Mais parfois retentissait la sirène comme un appel déchirant de Walkyrie — seule musique allemande qu’on eût entendue depuis la guerre — jusqu’à l’heure où les pompiers annonçaient que l’alerte était finie tandis qu’à côté d’eux la berloque, comme un invisible gamin, commentait à intervalles réguliers la bonne nouvelle et jetait en l’air son cri de joie.

M. de Charlus était étonné de voir que même des gens comme Brichot qui avant la guerre avaient été militaristes, reprochant surtout à la France de ne pas l’être assez, ne se contentaient pas de reprocher les excès de son militarisme à l’Allemagne, mais même son admiration de l’armée. Sans doute ils changeaient d’avis dès qu’il s’agissait de ralentir la guerre contre l’Allemagne et dénonçaient avec raison les pacifistes. Mais, par exemple, Brichot, ayant accepté, malgré ses yeux, de rendre compte dans des conférences de certains ouvrages parus chez les neutres, exaltait le roman d’un Suisse où sont raillés comme semence de militarisme deux enfants tombant d’une admiration symbolique à la vue d’un dragon. Cette raillerie avait de quoi déplaire pour d’autres raisons à M. de Charlus, lequel estimait qu’un dragon peut être quelque chose de fort beau. Mais surtout il ne comprenait pas l’admiration de Brichot, sinon pour le livre, que le baron n’avait pas lu, du moins pour son esprit, si différent de celui qui animait Brichot avant la guerre. Alors tout ce que faisait un militaire était bien, fût-ce les irrégularités du général de Boisdeffre, les travestissements et machinations du colonel du Paty de Clam, le faux du colonel Henry. Par quelle volte-face extraordinaire (et qui n’était en réalité qu’une autre face de la même passion fort noble, la passion patriotique, obligée, de militariste qu’elle était quand elle luttait contre le dreyfusisme, lequel était de tendances antimilitaristes, à se faire presque antimilitariste puisque c’était maintenant contre la Germanie sur-militariste qu’elle luttait) Brichot s’écriait-il : « Oh ! le spectacle bien mirifique et digne d’attirer la jeunesse d’un siècle tout de brutalité, ne connaissant que le culte de la force : un dragon ! On peut juger de ce que sera la vile soldatesque d’une génération élevée dans le culte de ces manifestations de force brutale ! » « Voyons, me dit M. de Charlus, vous connaissez Brichot et Cambremer. Chaque fois que je les vois ils me parlent de l’extraordinaire manque de psychologie de l’Allemagne. Entre nous, croyez-vous que jusqu’ici ils avaient eu grand souci de la psychologie, et que même maintenant ils soient capables d’en faire preuve ? Mais croyez bien que je n’exagère pas. Qu’il s’agisse du plus grand Allemand, de Nietzsche, de Gœthe, vous entendrez Brichot dire : « Avec l’habituel manque de psychologie qui caractérise la race teutonne ». Il y a évidemment dans la guerre des choses qui me font plus de peine. Mais avouez que c’est énervant. Norpois est plus fin, je le reconnais, bien qu’il n’ait pas cessé de se tromper depuis le commencement. Mais qu’est-ce que ça veut dire que ces articles qui excitent l’enthousiasme universel ? Mon cher Monsieur, vous savez aussi bien que moi ce que vaut Brichot, que j’aime beaucoup, même depuis le schisme qui m’a séparé de sa petite église, à cause de quoi je le vois beaucoup moins. Mais enfin j’ai une certaine considération pour ce régent de collège, beau parleur et fort instruit, et j’avoue que c’est fort touchant qu’à son âge, et diminué comme il est, car il l’est très sensiblement depuis quelques années, il se soit remis, comme il dit, à servir. Mais enfin la bonne intention est une chose, le talent en est une autre, et Brichot n’a jamais eu de talent. J’avoue que je partage son admiration pour certaines grandeurs de la guerre actuelle. Tout au plus est-il étrange qu’un partisan aveugle de l’Antiquité comme Brichot, qui n’avait pas assez de sarcasmes pour Zola trouvant plus de poésie dans un ménage d’ouvriers, dans la mine, que dans les palais historiques, ou pour Goncourt mettant Diderot au-dessus d’Homère et Watteau au-dessus de Raphaël, ne cesse de nous répéter que les Thermopyles, qu’Austerlitz même, ce n’était rien à côté de Vauquois. Cette fois, du reste, le public, qui avait résisté aux modernistes de la littérature et de l’art, suit ceux de la guerre, parce que c’est une mode adoptée de penser ainsi et puis que les petits esprits sont écrasés non par la beauté, mais par l’énormité de l’action. On n’écrit plus Kolossal qu’avec un K, mais, au fond, ce devant quoi on s’agenouille c’est bien du colossal.

« C’est, du reste, une étrange chose, ajouta M. de Charlus de la petite voix pointue qu’il prenait par moments. J’entends des gens qui ont l’air très heureux toute la journée, qui prennent d’excellents cocktails, déclarer qu’ils ne pourront aller jusqu’au bout de la guerre, que leur cœur n’aura pas la force, qu’ils ne peuvent pas penser à autre chose, qu’ils mourront tout d’un coup, et le plus extraordinaire, c’est que cela arrive en effet. Comme c’est curieux ! Est-ce une question d’alimentation, parce qu’ils n’ingéreront plus que des choses mal préparées, ou parce que pour prouver leur zèle ils s’attellent à des besognes vaines mais qui détruisent le régime qui les conservait ? Mais enfin j’enregistre un nombre étonnant de ces étranges morts prématurées, prématurées au moins au gré du défunt. Je ne sais plus ce que je vous disais, que Brichot et Norpois admiraient cette guerre, mais quelle singulière manière d’en parler ! D’abord avez-vous remarqué ce pullulement d’expressions nouvelles qu’emploie Norpois qui, quand elles ont fini par s’user à force d’être employées tous les jours — car vraiment il est infatigable, et je crois que c’est la mort de ma tante Villeparisis qui lui a donné une seconde jeunesse, — sont immédiatement remplacées par d’autres lieux communs ? Autrefois je me rappelle que vous vous amusiez à noter ces modes de langage qui apparaissaient, se maintenaient, puis disparaissaient : celui qui sème le vent récolte la tempête ; les chiens aboient, la caravane passe ; faites-moi de bonne politique et je vous ferai de bonnes finances, disait le baron Louis ; il y a des symptômes qu’il serait exagéré de prendre au tragique mais qu’il convient de prendre au sérieux ; travailler pour le roi de Prusse (celle-là a d’ailleurs ressuscité, ce qui était infaillible). Hé bien, depuis, hélas, que j’en ai vu mourir ! Nous avons eu : le chiffon de papier, les empires de proie, la fameuse kultur qui consiste à assassiner des femmes et des enfants sans défense, la victoire appartient, comme disent les Japonais, à celui qui sait souffrir un quart d’heure de plus que l’autre, les Germano-Touraniens, la barbarie scientifique — si nous voulons gagner la guerre, selon la forte expression de M. Lloyd George — enfin ça ne se compte plus, et le mordant des troupes, et le cran des troupes. Même la syntaxe de l’excellent Norpois subit du fait de la guerre une altération aussi profonde que la fabrication du pain ou la rapidité des transports. Avez-vous remarqué que l’excellent homme, tenant à proclamer ses désirs comme une vérité sur le point d’être réalisée, n’ose pas tout de même employer le futur pur et simple, qui risquerait d’être contredit par les événements, mais a adopté comme signe de ce temps le verbe savoir ? » J’avouai à M. de Charlus que je ne comprenais pas bien ce qu’il voulait dire. Il me faut noter ici que le duc de Guermantes ne partageait nullement le pessimisme de son frère. Il était, de plus, aussi anglophile que M. de Charlus était anglophobe. Enfin il tenait M. Caillaux pour un traître qui méritait mille fois d’être fusillé. Quand son frère lui demandait des preuves de cette trahison, M. de Guermantes répondait que s’il ne fallait condamner que les gens qui signent un papier où ils déclarent « j’ai trahi » on ne punirait jamais le crime de trahison. Mais pour le cas où je n’aurais pas l’occasion d’y revenir, je noterai aussi que, deux ans plus tard, le duc de Guermantes, animé du plus pur anticaillautisme, rencontra un attaché militaire anglais et sa femme, couple remarquablement lettré avec lequel il se lia, comme au temps de l’affaire Dreyfus avec les trois dames charmantes ; que dès le premier jour il eut la stupéfaction, parlant de Caillaux dont il estimait la condamnation certaine et le crime patent, d’entendre le couple charmant et lettré dire : « Mais il sera probablement acquitté, il n’y a absolument rien contre lui. » M. de Guermantes essaya d’alléguer que M. de Norpois, dans sa déposition, avait dit en regardant Caillaux atterré : « Monsieur Caillaux, vous êtes le Giolitti de la France. » Mais le couple charmant avait souri, tourné M. de Norpois en ridicule, cité des preuves de son gâtisme et conclu qu’il avait dit cela devant M. Caillaux atterré, disait le Figaro, mais probablement, en réalité, devant M. Caillaux narquois. Les opinions du duc de Guermantes n’avaient pas tardé à changer. Attribuer ce changement à l’influence d’une Anglaise n’est pas aussi extraordinaire que cela eût pu paraître si on l’eût prophétisé même en 1919, où les Anglais n’appelaient les Allemands que les Huns et réclamaient une féroce condamnation contre les coupables. Leur opinion à eux aussi devait changer et toute décision être approuvée par eux qui pouvait contrister la France et venir en aide à l’Allemagne. Pour revenir à M. de Charlus : « Mais si, répondit-il à l’aveu que je ne le comprenais pas : « savoir », dans les articles de Norpois, est le signe du futur, c’est-à-dire le signe des désirs de Norpois et des désirs de nous tous d’ailleurs, ajouta-t-il, peut-être sans une complète sincérité, vous comprenez bien que si « savoir » n’était pas devenu le simple signe du futur, on comprendrait à la rigueur que le sujet de ce verbe pût être un pays, par exemple chaque fois que Norpois dit : « L’Amérique ne saurait rester indifférente à ces violations répétées du droit », « La monarchie bicéphale ne saurait manquer de venir à résipiscence ». Il est clair que de telles phrases expriment les désirs de Norpois (comme les miens, comme les vôtres), mais enfin, là le verbe peut encore garder malgré tout son sens ancien, car un pays peut « savoir », l’Amérique peut « savoir », la monarchie « bicéphale » elle-même peut « savoir » (malgré l’éternel manque de psychologie), mais le doute n’est plus possible quand Norpois écrit : « Ces dévastations systématiques ne sauraient persuader aux neutres », « La région des lacs ne saurait manquer de tomber à bref délai aux mains des alliés », « Les résultats de ces élections neutralistes ne sauraient refléter l’opinion de la grande majorité du pays. » Or il est certain que ces dévastations, ces régions et ces résultats de votes sont des choses inanimées qui ne peuvent pas « savoir ». Par cette formule Norpois adresse simplement aux neutres l’injonction (à laquelle j’ai le regret de constater qu’ils ne semblent pas obéir) de sortir de la neutralité ou aux régions des lacs de ne plus appartenir aux « Boches » (M. de Charlus mettait à prononcer le mot « boche » le même genre de hardiesse que jadis dans le train de Balbec à parler des hommes dont le goût n’est pas pour les femmes). D’ailleurs, avez-vous remarqué avec quelles ruses Norpois a toujours commencé, dès 1914, ses articles aux neutres ? Il commence par déclarer que, certes, la France n’a pas à s’immiscer dans la politique de l’Italie ou de la Roumanie ou de la Bulgarie, etc. C’est à ces puissances seules qu’il convient de décider en toute indépendance et en ne consultant que l’intérêt national si elles doivent ou non sortir de la neutralité. Mais si ces premières déclarations de l’article (ce qu’on eût appelé autrefois l’exorde) sont si remarquables et désintéressées, le morceau suivant l’est généralement beaucoup moins. Toutefois, en continuant, dit en substance Norpois, « il est bien clair que seules tireront un bénéfice matériel de la lutte les nations qui se seront rangées du côté du Droit et de la Justice. On ne peut attendre que les alliés récompensent, en leur octroyant leurs territoires d’où s’élève depuis des siècles la plainte de leurs frères opprimés, les peuples qui, suivant la politique de moindre effort, n’auront pas mis leur épée au service des alliés ». Ce premier pas fait vers un conseil d’intervention, rien n’arrête plus Norpois, ce n’est plus seulement le principe mais l’époque de l’intervention sur lesquels il donne des conseils de moins en moins déguisés. « Certes, dit-il en faisant ce qu’il appellerait lui-même le bon apôtre, c’est à l’Italie, à la Roumanie seules de décider de l’heure opportune et de la forme sous laquelle il leur conviendra d’intervenir. Elles ne peuvent pourtant ignorer qu’à trop tergiverser elles risquent de laisser passer l’heure. Déjà les sabots des cavaliers russes font frémir la Germanie traquée d’une indicible épouvante. Il est bien évident que les peuples qui n’auront fait que voler au secours de la victoire, dont on voit déjà l’aube resplendissante, n’auront nullement droit à cette même récompense qu’ils peuvent encore en se hâtant, etc. » C’est comme au théâtre quand on dit : « Les dernières places qui restent ne tarderont pas à être enlevées. Avis aux retardataires. » Raisonnement d’autant plus stupide que Norpois le refait tous les six mois, et dit périodiquement à la Roumanie : « L’heure est venue pour la Roumanie de savoir si elle veut ou non réaliser ses aspirations nationales. Qu’elle attende encore, il risque d’être trop tard. » Or, depuis deux ans qu’il le dit, non seulement le « trop tard » n’est pas encore venu, mais on ne cesse de grossir les offres qu’on fait à la Roumanie. De même il invite la France, etc., à intervenir en Grèce en tant que puissance protectrice parce que le traité qui liait la Grèce à la Serbie n’a pas été tenu. Or, de bonne foi, si la France n’était pas en guerre et ne souhaitait pas le concours ou la neutralité bienveillante de la Grèce, aurait-elle l’idée d’intervenir en tant que puissance protectrice, et le sentiment moral qui la pousse à se révolter parce que la Grèce n’a pas tenu ses engagements avec la Serbie ne se tait-il pas aussi dès qu’il s’agit de violation tout aussi flagrante de la Roumanie et de l’Italie qui, avec raison, je le crois, comme la Grèce aussi, n’ont pas rempli leurs devoirs, moins impératifs et étendus qu’on ne dit, d’alliés de l’Allemagne. La vérité c’est que les gens voient tout par leur journal, et comment pourraient-ils faire autrement puisqu’ils ne connaissent pas personnellement les gens ni les événements dont il s’agit ? Au temps de l’affaire qui passionnait si bizarrement à une époque dont il est convenu de dire que nous sommes séparés par des siècles, car les philosophes de la guerre ont accrédité que tout lien est rompu avec le passé, j’étais choqué de voir des gens de ma famille accorder toute leur estime à des anticléricaux, anciens communards que leur journal leur avait présentés comme antidreyfusards, et honnir un général bien né et catholique mais révisionniste. Je ne le suis pas moins de voir tous les Français exécrer l’Empereur François-Joseph qu’ils vénéraient, avec raison, je peux vous le dire, moi qui l’ai beaucoup connu et qu’il veut bien traiter en cousin. Ah ! je ne lui ai pas écrit depuis la guerre, ajouta-t-il comme avouant hardiment une faute qu’il savait très bien qu’on ne pouvait blâmer. Si, la première année, et une seule fois. Mais qu’est-ce que vous voulez, cela ne change rien à mon respect pour lui, mais j’ai ici beaucoup de jeunes parents qui se battent dans nos lignes et qui trouveraient, je le sais, fort mauvais que j’entretienne une correspondance suivie avec le chef d’une nation en guerre avec nous. Que voulez-vous ? me critique qui voudra, ajouta-t-il, comme s’exposant hardiment à mes reproches, je n’ai pas voulu qu’une lettre signée Charlus arrivât en ce moment à Vienne. La plus grande critique que j’adresserais au vieux souverain, c’est qu’un seigneur de son rang, chef d’une des maisons les plus anciennes et les plus illustres d’Europe, se soit laissé mener par ce petit hobereau, fort intelligent d’ailleurs, mais enfin par un simple parvenu comme Guillaume de Hohenzollern. Ce n’est pas une des anomalies les moins choquantes de cette guerre. » Et comme, dès qu’il se replaçait au point de vue nobiliaire, qui pour lui au fond dominait tout, M. de Charlus arrivait à d’extraordinaires enfantillages, il me dit du même ton qu’il m’eût parlé de la Marne ou de Verdun qu’il y avait des choses capitales et fort curieuses que ne devrait pas omettre celui qui écrirait l’histoire de cette guerre. « Ainsi, me dit-il, par exemple, tout le monde est si ignorant que personne n’a fait remarquer cette chose si marquante : le grand maître de l’ordre de Malte, qui est un pur boche, n’en continue pas moins de vivre à Rome où il jouit, en tant que grand maître de notre ordre, du privilège de l’exterritorialité. C’est intéressant », ajouta-t-il d’un air de me dire : « Vous voyez que vous n’avez pas perdu votre soirée en me rencontrant. » Je le remerciai et il prit l’air modeste de quelqu’un qui n’exige pas de salaire. « Qu’est-ce que j’étais donc en train de vous dire ? Ah ! oui, que les gens haïssaient maintenant François-Joseph, d’après leur journal. Pour le roi Constantin de Grèce et le tzar de Bulgarie, le public a oscillé, à diverses reprises, entre l’aversion et la sympathie, parce qu’on disait tour à tour qu’ils se mettaient du côté de l’Entente ou de ce que Norpois appelle les Empires centraux. C’est comme quand il nous répète à tout moment que « l’heure de Venizelos va sonner ». Je ne doute pas que M. Venizelos soit un homme d’État plein de capacité, mais qui nous dit que les Grecs désirent tant que cela Venizelos ? Il voulait, nous dit-on, que la Grèce tînt ses engagements envers la Serbie. Encore faudrait-il savoir quels étaient ces engagements et s’ils étaient plus étendus que ceux que l’Italie et la Roumanie ont cru pouvoir violer. Nous avons de la façon dont la Grèce exécute ses traités et respecte sa constitution un souci que nous n’aurions certainement pas si ce n’était pas notre intérêt. Qu’il n’y ait pas eu la guerre, croyez-vous que les puissances « garantes » auraient même fait attention à la dissolution des Chambres ? Je vois simplement qu’on retire un à un ses appuis au Roi de Grèce pour pouvoir le jeter dehors ou l’enfermer le jour où il n’aura plus d’armée pour le défendre. Je vous disais que le public ne juge le Roi de Grèce et le Roi des Bulgares que d’après les journaux. Et comment pourraient-ils penser sur eux autrement que par le journal puisqu’ils ne les connaissent pas ? Moi je les ai vus énormément, j’ai beaucoup connu, quand il était diadoque, Constantin de Grèce, qui était une pure merveille. J’ai toujours pensé que l’Empereur Nicolas avait eu un énorme sentiment pour lui. En tout bien tout honneur, bien entendu. La princesse Christian en parlait ouvertement, mais c’est une gale. Quant au tzar des Bulgares, c’est une fine coquine, une vraie affiche, mais très intelligent, un homme remarquable. Il m’aime beaucoup. »

M. de Charlus, qui pouvait être si agréable, devenait odieux quand il abordait ces sujets. Il y apportait la satisfaction qui agace déjà chez un malade qui vous fait tout le temps valoir sa bonne santé. J’ai souvent pensé que, dans le tortillard de Balbec, les fidèles qui souhaitaient tant les aveux devant lesquels il se dérobait n’auraient peut-être pas pu supporter cette espèce d’ostentation d’une manie et, mal à l’aise, respirant mal comme dans une chambre de malade ou devant un morphinomane qui tirerait devant vous sa seringue, ce fussent eux qui eussent mis fin aux confidences qu’ils croyaient désirer. De plus, on était agacé d’entendre accuser tout le monde, et probablement bien souvent sans aucune espèce de preuve, par quelqu’un qui s’omettait lui-même de la catégorie spéciale à laquelle on savait pourtant qu’il appartenait et où il rangeait si volontiers les autres. Enfin, lui si intelligent, s’était fait à cet égard une petite philosophie étroite (à la base de laquelle il y avait peut-être un rien des curiosités que Swann trouvait dans « la vie ») expliquant tout par ces causes spéciales et où, comme chaque fois qu’on verse dans son défaut, il était non seulement au-dessous de lui-même mais exceptionnellement satisfait de lui. C’est ainsi que lui si grave, si noble, eut le sourire le plus niais pour achever la phrase que voici : « Comme il y a de fortes présomptions du même genre que pour Ferdinand de Cobourg à l’égard de l’Empereur Guillaume, cela pourrait être la cause pour laquelle le tzar Ferdinand s’est mis du côté des « Empires de proie ». Dame, au fond, c’est très compréhensible, on est indulgent pour une sœur, on ne lui refuse rien. Je trouve que ce serait très joli comme explication de l’alliance de la Bulgarie avec l’Allemagne. » Et de cette explication stupide M. de Charlus rit longuement comme s’il l’avait vraiment trouvée très ingénieuse alors que, même si elle avait reposé sur des faits vrais, elle était aussi puérile que les réflexions que M. de Charlus faisait sur la guerre quand il la jugeait en tant que féodal ou que chevalier de Saint-Jean de Jérusalem. Il finit par une remarque juste : « Ce qui est étonnant, dit-il, c’est que ce public qui ne juge ainsi des hommes et des choses de la guerre que par les journaux est persuadé qu’il juge par lui-même. » En cela M. de Charlus avait raison. On m’a raconté qu’il fallait voir les moments de silence et d’hésitation qu’avait Mme de Forcheville, pareils à ceux qui sont nécessaires, non pas même seulement à l’énonciation, mais à la formation d’une opinion personnelle, avant de dire, sur le ton d’un sentiment intime : « Non, je ne crois pas qu’ils prendront Varsovie » ; « Je n’ai pas l’impression qu’on puisse passer un second hiver » ; « Ce que je ne voudrais pas, c’est une paix boiteuse » ; « Ce qui me fait peur, si vous voulez que je vous le dise, c’est la Chambre » ; « Si, j’estime tout de même qu’on pourrait percer. » Et pour dire cela Odette prenait un air mièvre qu’elle poussait à l’extrême quand elle disait : « Je ne dis pas que les armées allemandes ne se battent pas bien, mais il leur manque ce qu’on appelle le cran. » Pour prononcer « le cran » (et même simplement pour le « mordant ») elle faisait avec sa main le geste de pétrissage et avec ses yeux le clignement des rapins employant un terme d’atelier. Son langage à elle était pourtant plus encore qu’autrefois la trace de son admiration pour les Anglais, qu’elle n’était plus obligée de se contenter d’appeler comme autrefois nos voisins d’outre-Manche, ou tout au plus nos amis les Anglais, mais nos loyaux alliés ! Inutile de dire qu’elle ne se faisait pas faute de citer à tout propos l’expression de « fair play » pour montrer les Anglais trouvant les Allemands des joueurs incorrects, et « ce qu’il faut c’est gagner la guerre », comme disent nos braves alliés. Tout au plus associait-elle assez maladroitement le nom de son gendre à tout ce qui touchait les soldats anglais et au plaisir qu’il trouvait à vivre dans l’intimité des Australiens aussi bien que des Écossais, des Néo-Zélandais et des Canadiens. « Mon gendre Saint-Loup connaît maintenant l’argot de tous les braves « tommies », il sait se faire entendre de ceux des plus lointains « dominions » et, aussi bien qu’avec le général commandant la base, fraternise avec le plus humble « private ».

Que cette parenthèse sur Mme de Forcheville m’autorise, tandis que je descends les boulevards côte à côte avec M. de Charlus, à une autre plus longue encore, mais utile pour décrire cette époque, sur les rapports de Mme Verdurin avec Brichot. En effet, si le pauvre Brichot était, ainsi que Norpois, jugé sans indulgence par M. de Charlus (parce que celui-ci était à la fois très fin et plus ou moins inconsciemment germanophile), il était encore bien plus maltraité par les Verdurin. Sans doute ceux-ci étaient chauvins, ce qui eût dû les faire se plaire aux articles de Brichot, lesquels d’autre part n’étaient pas inférieurs à bien des écrits où se délectait Mme Verdurin. Mais d’abord on se rappelle peut-être que, déjà à la Raspelière, Brichot était devenu pour les Verdurin du grand homme qu’il leur avait paru être autrefois, sinon une tête de Turc comme Saniette, du moins l’objet de leurs railleries à peine déguisées. Du moins restait-il, à ce moment-là, un fidèle entre les fidèles, ce qui lui assurait une part des avantages prévus tacitement par les statuts à tous les membres fondateurs associés du petit groupe. Mais au fur et à mesure que, à la faveur de la guerre peut-être, ou par la rapide cristallisation d’une élégance si longtemps retardée, mais dont tous les éléments nécessaires et restés invisibles saturaient depuis longtemps le salon des Verdurin, celui-ci s’était ouvert à un monde nouveau et que les fidèles, appâts d’abord de ce monde nouveau, avaient fini par être de moins en moins invités, un phénomène parallèle se produisait pour Brichot. Malgré la Sorbonne, malgré l’Institut, sa notoriété n’avait pas jusqu’à la guerre dépassé les limites du salon Verdurin. Mais quand il se mit à écrire, presque quotidiennement, des articles parés de ce faux brillant qu’on l’a vu si souvent dépenser sans compter pour les fidèles, riches, d’autre part, d’une érudition fort réelle, et qu’en vrai sorbonien il ne cherchait pas à dissimuler de quelques formes plaisantes qu’il l’entourât, le « grand monde » fut littéralement ébloui. Pour une fois, d’ailleurs, il donnait sa faveur à quelqu’un qui était loin d’être une nullité et qui pouvait retenir l’attention par la fertilité de son intelligence et les ressources de sa mémoire. Et pendant que trois duchesses allaient passer la soirée chez Mme Verdurin, trois autres se disputaient l’honneur d’avoir chez elles à dîner le grand homme, lequel acceptait chez l’une, se sentant d’autant plus libre que Mme Verdurin, exaspérée du succès que ses articles rencontraient auprès du faubourg Saint-Germain, avait soin de ne jamais avoir Brichot chez elle quand il devait s’y trouver quelque personne brillante qu’il ne connaissait pas encore et qui se hâterait de l’attirer. Ce fut ainsi que le journalisme, dans lequel Brichot se contentait, en somme, de donner tardivement, avec honneur et en échange d’émoluments superbes, ce qu’il avait gaspillé toute sa vie gratis et incognito dans le salon des Verdurin (car ses articles ne lui coûtaient pas plus de peine, tant il était disert et savant, que ses causeries) eût conduit, et parut même un moment conduire Brichot à une gloire incontestée, s’il n’y avait pas eu Mme Verdurin. Certes, les articles de Brichot étaient loin d’être aussi remarquables que le croyaient les gens du monde. La vulgarité de l’homme apparaissait à tout instant sous le pédantisme du lettré. Et à côté d’images qui ne voulaient rien dire du tout (les Allemands ne pourront plus regarder en face la statue de Beethoven ; Schiller a dû frémir dans son tombeau ; l’encre qui avait paraphé la neutralité de la Belgique était à peine séchée ; Lénine parle, mais autant en emporte le vent de la steppe), c’étaient des trivialités telles que : « Vingt mille prisonniers, c’est un chiffre » ; « Notre commandement saura ouvrir l’œil et le bon » ; « Nous voulons vaincre, un point c’est tout. » Mais, mêlés à tout cela, tant de savoir, tant d’intelligence, de si justes raisonnements. Or, Mme Verdurin ne commençait jamais un article de Brichot sans la satisfaction préalable de penser qu’elle allait y trouver des choses ridicules, et le lisait avec l’attention la plus soutenue pour être certaine de ne les pas laisser échapper. Or, il était malheureusement certain qu’il y en avait quelques-unes. On n’attendait même pas de les avoir trouvées. La citation la plus heureuse d’un auteur vraiment peu connu, au moins dans l’œuvre à laquelle Brichot se reportait, était incriminée comme preuve du pédantisme le plus insoutenable et Mme Verdurin attendait avec impatience l’heure du dîner pour déchaîner les éclats de rire de ses convives. « Hé bien, qu’est-ce que vous avez dit du Brichot de ce soir ? J’ai pensé à vous en lisant la citation de Cuvier. Ma parole, je crois qu’il devient fou. — Je ne l’ai pas encore lu, disait un fidèle. — Comment, vous ne l’avez pas encore lu ? Mais vous ne savez pas les délices que vous vous refusez. C’est-à-dire que c’est d’un ridicule à mourir. » Et contente au fond que quelqu’un n’eût pas encore lu le Brichot pour avoir l’occasion d’en mettre elle-même en lumière les ridicules, Mme Verdurin disait au maître d’hôtel d’apporter le Temps et faisait elle-même la lecture à haute voix, en faisant sonner avec emphase les phrases les plus simples. Après le dîner, pendant toute la soirée ; cette campagne anti-brichotiste continuait, mais avec de fausses réserves. « Je ne le dis pas trop haut parce que j’ai peur que là-bas, disait-elle en montrant la comtesse Molé, on n’admire assez cela. Les gens du monde sont plus naïfs qu’on ne croit. » Mme Molé, à qui on tâchait de faire entendre, en parlant assez fort, qu’on parlait d’elle, tout en s’efforçant de lui montrer par des baissements de voix, qu’on n’aurait pas voulu être entendu d’elle, reniait lâchement Brichot qu’elle égalait en réalité à Michelet. Elle donnait raison à Mme Verdurin, et pour terminer pourtant par quelque chose qui lui paraissait incontestable, disait : « Ce qu’on ne peut pas lui retirer, c’est que c’est bien écrit. — Vous trouvez ça bien écrit, vous ? disait Mme Verdurin, moi je trouve ça écrit comme par un cochon », audace qui faisait rire les gens du monde, d’autant plus que Mme Verdurin, effarouchée elle-même par le mot de cochon, l’avait prononcé en le chuchotant la main rabattue sur les lèvres. Sa rage contre Brichot croissait d’autant plus que celui-ci étalait naïvement la satisfaction de son succès, malgré les accès de mauvaise humeur que provoquait chez lui la censure, chaque fois que, comme il le disait avec son habitude d’employer les mots nouveaux pour montrer qu’il n’était pas trop universitaire, elle avait « caviardé » une partie de son article. Devant lui Mme Verdurin ne laissait pas trop voir, sauf par une maussaderie qui eût averti un homme plus perspicace, le peu de cas qu’elle faisait de ce qu’il écrivait. Elle lui reprocha seulement une fois d’écrire si souvent « je ». Et il avait, en effet, l’habitude de l’écrire continuellement, d’abord parce que, par habitude de professeur, il se servait constamment d’expressions comme « j’accorde que », « je veux bien que l’énorme développement des fronts nécessite », etc., mais surtout parce que, ancien antidreyfusard militant qui flairait la préparation germanique bien longtemps avant la guerre, il s’était trouvé écrire très souvent : « J’ai dénoncé dès 1897 » ; « j’ai signalé en 1901 » ; « j’ai averti dans ma petite brochure aujourd’hui rarissime (habent sua fata libelli) », et ensuite l’habitude lui était restée. Il rougit fortement de l’observation de Mme Verdurin, qui lui fut faite d’un ton aigre. « Vous avez raison, Madame, quelqu’un qui n’aimait pas plus les jésuites que M. Combes, encore qu’il n’ait pas eu de préface de notre doux maître en scepticisme délicieux, Anatole France, qui fut si je ne me trompe mon adversaire… avant le Déluge, a dit que le moi est toujours haïssable. » À partir de ce moment Brichot remplaça je par on, mais on n’empêchait pas le lecteur de voir que l’auteur parlait de lui et permit à l’auteur de ne plus cesser de parler de lui, de commenter la moindre de ses phrases, de faire un article sur une seule négation, toujours à l’abri de on. Par exemple, Brichot avait-il dit, fût-ce dans un autre article, que les armées allemandes avaient perdu de leur valeur, il commençait ainsi : « On ne camoufle pas ici la vérité. On a dit que les armées allemandes avaient perdu de leur valeur. On n’a pas dit qu’elles n’avaient plus une grande valeur. Encore moins écrira-t-on qu’elles n’ont plus aucune valeur. On ne dira pas non plus que le terrain gagné, s’il n’est pas, etc. » Bref, rien qu’à énoncer tout ce qu’il ne dirait pas, à rappeler tout ce qu’il avait dit il y avait quelques années, et ce que Clausewitz, Ovide, Apollonius de Tyane avaient dit il y avait plus ou moins de siècles, Brichot aurait pu constituer aisément la matière d’un fort volume. Il est à regretter qu’il n’en ait pas publié, car ces articles si nourris sont maintenant difficiles à retrouver. Le faubourg Saint-Germain, chapitré par Mme Verdurin, commença par rire de Brichot chez elle, mais continua, une fois sorti du petit clan, à admirer Brichot. Puis se moquer de lui devint une mode comme ç’avait été de l’admirer, et celles mêmes qu’il continuait d’intéresser en secret, dès le temps qu’elles lisaient son article, s’arrêtaient et riaient dès qu’elles n’étaient plus seules, pour ne pas avoir l’air moins fines que les autres. Jamais on ne parla tant de Brichot qu’à cette époque dans le petit clan, mais par dérision. On prenait comme critérium de l’intelligence de tout nouveau ce qu’il pensait des articles de Brichot ; s’il répondait mal la première fois, on ne se faisait pas faute de lui apprendre à quoi l’on reconnaît que les gens sont intelligents.
