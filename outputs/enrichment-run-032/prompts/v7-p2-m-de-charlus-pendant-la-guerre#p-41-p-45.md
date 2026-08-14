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

Que cette parenthèse sur Mme de Forcheville m’autorise, tandis que je descends les boulevards côte à côte avec M. de Charlus, à une autre plus longue encore, mais utile pour décrire cette époque, sur les rapports de Mme Verdurin avec Brichot. En effet, si le pauvre Brichot était, ainsi que Norpois, jugé sans indulgence par M. de Charlus (parce que celui-ci était à la fois très fin et plus ou moins inconsciemment germanophile), il était encore bien plus maltraité par les Verdurin. Sans doute ceux-ci étaient chauvins, ce qui eût dû les faire se plaire aux articles de Brichot, lesquels d’autre part n’étaient pas inférieurs à bien des écrits où se délectait Mme Verdurin. Mais d’abord on se rappelle peut-être que, déjà à la Raspelière, Brichot était devenu pour les Verdurin du grand homme qu’il leur avait paru être autrefois, sinon une tête de Turc comme Saniette, du moins l’objet de leurs railleries à peine déguisées. Du moins restait-il, à ce moment-là, un fidèle entre les fidèles, ce qui lui assurait une part des avantages prévus tacitement par les statuts à tous les membres fondateurs associés du petit groupe. Mais au fur et à mesure que, à la faveur de la guerre peut-être, ou par la rapide cristallisation d’une élégance si longtemps retardée, mais dont tous les éléments nécessaires et restés invisibles saturaient depuis longtemps le salon des Verdurin, celui-ci s’était ouvert à un monde nouveau et que les fidèles, appâts d’abord de ce monde nouveau, avaient fini par être de moins en moins invités, un phénomène parallèle se produisait pour Brichot. Malgré la Sorbonne, malgré l’Institut, sa notoriété n’avait pas jusqu’à la guerre dépassé les limites du salon Verdurin. Mais quand il se mit à écrire, presque quotidiennement, des articles parés de ce faux brillant qu’on l’a vu si souvent dépenser sans compter pour les fidèles, riches, d’autre part, d’une érudition fort réelle, et qu’en vrai sorbonien il ne cherchait pas à dissimuler de quelques formes plaisantes qu’il l’entourât, le « grand monde » fut littéralement ébloui. Pour une fois, d’ailleurs, il donnait sa faveur à quelqu’un qui était loin d’être une nullité et qui pouvait retenir l’attention par la fertilité de son intelligence et les ressources de sa mémoire. Et pendant que trois duchesses allaient passer la soirée chez Mme Verdurin, trois autres se disputaient l’honneur d’avoir chez elles à dîner le grand homme, lequel acceptait chez l’une, se sentant d’autant plus libre que Mme Verdurin, exaspérée du succès que ses articles rencontraient auprès du faubourg Saint-Germain, avait soin de ne jamais avoir Brichot chez elle quand il devait s’y trouver quelque personne brillante qu’il ne connaissait pas encore et qui se hâterait de l’attirer. Ce fut ainsi que le journalisme, dans lequel Brichot se contentait, en somme, de donner tardivement, avec honneur et en échange d’émoluments superbes, ce qu’il avait gaspillé toute sa vie gratis et incognito dans le salon des Verdurin (car ses articles ne lui coûtaient pas plus de peine, tant il était disert et savant, que ses causeries) eût conduit, et parut même un moment conduire Brichot à une gloire incontestée, s’il n’y avait pas eu Mme Verdurin. Certes, les articles de Brichot étaient loin d’être aussi remarquables que le croyaient les gens du monde. La vulgarité de l’homme apparaissait à tout instant sous le pédantisme du lettré. Et à côté d’images qui ne voulaient rien dire du tout (les Allemands ne pourront plus regarder en face la statue de Beethoven ; Schiller a dû frémir dans son tombeau ; l’encre qui avait paraphé la neutralité de la Belgique était à peine séchée ; Lénine parle, mais autant en emporte le vent de la steppe), c’étaient des trivialités telles que : « Vingt mille prisonniers, c’est un chiffre » ; « Notre commandement saura ouvrir l’œil et le bon » ; « Nous voulons vaincre, un point c’est tout. » Mais, mêlés à tout cela, tant de savoir, tant d’intelligence, de si justes raisonnements. Or, Mme Verdurin ne commençait jamais un article de Brichot sans la satisfaction préalable de penser qu’elle allait y trouver des choses ridicules, et le lisait avec l’attention la plus soutenue pour être certaine de ne les pas laisser échapper. Or, il était malheureusement certain qu’il y en avait quelques-unes. On n’attendait même pas de les avoir trouvées. La citation la plus heureuse d’un auteur vraiment peu connu, au moins dans l’œuvre à laquelle Brichot se reportait, était incriminée comme preuve du pédantisme le plus insoutenable et Mme Verdurin attendait avec impatience l’heure du dîner pour déchaîner les éclats de rire de ses convives. « Hé bien, qu’est-ce que vous avez dit du Brichot de ce soir ? J’ai pensé à vous en lisant la citation de Cuvier. Ma parole, je crois qu’il devient fou. — Je ne l’ai pas encore lu, disait un fidèle. — Comment, vous ne l’avez pas encore lu ? Mais vous ne savez pas les délices que vous vous refusez. C’est-à-dire que c’est d’un ridicule à mourir. » Et contente au fond que quelqu’un n’eût pas encore lu le Brichot pour avoir l’occasion d’en mettre elle-même en lumière les ridicules, Mme Verdurin disait au maître d’hôtel d’apporter le Temps et faisait elle-même la lecture à haute voix, en faisant sonner avec emphase les phrases les plus simples. Après le dîner, pendant toute la soirée ; cette campagne anti-brichotiste continuait, mais avec de fausses réserves. « Je ne le dis pas trop haut parce que j’ai peur que là-bas, disait-elle en montrant la comtesse Molé, on n’admire assez cela. Les gens du monde sont plus naïfs qu’on ne croit. » Mme Molé, à qui on tâchait de faire entendre, en parlant assez fort, qu’on parlait d’elle, tout en s’efforçant de lui montrer par des baissements de voix, qu’on n’aurait pas voulu être entendu d’elle, reniait lâchement Brichot qu’elle égalait en réalité à Michelet. Elle donnait raison à Mme Verdurin, et pour terminer pourtant par quelque chose qui lui paraissait incontestable, disait : « Ce qu’on ne peut pas lui retirer, c’est que c’est bien écrit. — Vous trouvez ça bien écrit, vous ? disait Mme Verdurin, moi je trouve ça écrit comme par un cochon », audace qui faisait rire les gens du monde, d’autant plus que Mme Verdurin, effarouchée elle-même par le mot de cochon, l’avait prononcé en le chuchotant la main rabattue sur les lèvres. Sa rage contre Brichot croissait d’autant plus que celui-ci étalait naïvement la satisfaction de son succès, malgré les accès de mauvaise humeur que provoquait chez lui la censure, chaque fois que, comme il le disait avec son habitude d’employer les mots nouveaux pour montrer qu’il n’était pas trop universitaire, elle avait « caviardé » une partie de son article. Devant lui Mme Verdurin ne laissait pas trop voir, sauf par une maussaderie qui eût averti un homme plus perspicace, le peu de cas qu’elle faisait de ce qu’il écrivait. Elle lui reprocha seulement une fois d’écrire si souvent « je ». Et il avait, en effet, l’habitude de l’écrire continuellement, d’abord parce que, par habitude de professeur, il se servait constamment d’expressions comme « j’accorde que », « je veux bien que l’énorme développement des fronts nécessite », etc., mais surtout parce que, ancien antidreyfusard militant qui flairait la préparation germanique bien longtemps avant la guerre, il s’était trouvé écrire très souvent : « J’ai dénoncé dès 1897 » ; « j’ai signalé en 1901 » ; « j’ai averti dans ma petite brochure aujourd’hui rarissime (habent sua fata libelli) », et ensuite l’habitude lui était restée. Il rougit fortement de l’observation de Mme Verdurin, qui lui fut faite d’un ton aigre. « Vous avez raison, Madame, quelqu’un qui n’aimait pas plus les jésuites que M. Combes, encore qu’il n’ait pas eu de préface de notre doux maître en scepticisme délicieux, Anatole France, qui fut si je ne me trompe mon adversaire… avant le Déluge, a dit que le moi est toujours haïssable. » À partir de ce moment Brichot remplaça je par on, mais on n’empêchait pas le lecteur de voir que l’auteur parlait de lui et permit à l’auteur de ne plus cesser de parler de lui, de commenter la moindre de ses phrases, de faire un article sur une seule négation, toujours à l’abri de on. Par exemple, Brichot avait-il dit, fût-ce dans un autre article, que les armées allemandes avaient perdu de leur valeur, il commençait ainsi : « On ne camoufle pas ici la vérité. On a dit que les armées allemandes avaient perdu de leur valeur. On n’a pas dit qu’elles n’avaient plus une grande valeur. Encore moins écrira-t-on qu’elles n’ont plus aucune valeur. On ne dira pas non plus que le terrain gagné, s’il n’est pas, etc. » Bref, rien qu’à énoncer tout ce qu’il ne dirait pas, à rappeler tout ce qu’il avait dit il y avait quelques années, et ce que Clausewitz, Ovide, Apollonius de Tyane avaient dit il y avait plus ou moins de siècles, Brichot aurait pu constituer aisément la matière d’un fort volume. Il est à regretter qu’il n’en ait pas publié, car ces articles si nourris sont maintenant difficiles à retrouver. Le faubourg Saint-Germain, chapitré par Mme Verdurin, commença par rire de Brichot chez elle, mais continua, une fois sorti du petit clan, à admirer Brichot. Puis se moquer de lui devint une mode comme ç’avait été de l’admirer, et celles mêmes qu’il continuait d’intéresser en secret, dès le temps qu’elles lisaient son article, s’arrêtaient et riaient dès qu’elles n’étaient plus seules, pour ne pas avoir l’air moins fines que les autres. Jamais on ne parla tant de Brichot qu’à cette époque dans le petit clan, mais par dérision. On prenait comme critérium de l’intelligence de tout nouveau ce qu’il pensait des articles de Brichot ; s’il répondait mal la première fois, on ne se faisait pas faute de lui apprendre à quoi l’on reconnaît que les gens sont intelligents.

### Passage

« Enfin, mon pauvre ami, continua M. de Charlus, tout cela est épouvantable et nous avons plus que d’ennuyeux articles à déplorer. On parle de vandalisme, de statues détruites. Mais est-ce que la destruction de tant de merveilleux jeunes gens, qui étaient des statues polychromes incomparables, n’est pas du vandalisme aussi ? Est-ce qu’une ville qui n’aura plus de beaux hommes ne sera pas comme une ville dont toute la statuaire aurait été brisée ? Quel plaisir puis-je avoir à aller dîner au restaurant quand j’y suis servi par de vieux bouffons moussus qui ressemblent au Père Didon, si ce n’est pas par des femmes en cornette qui me font croire que je suis entré au bouillon Duval. Parfaitement, mon cher, et je crois que j’ai le droit de parler ainsi parce que le Beau est tout de même le Beau dans une matière vivante. Le grand plaisir d’être servi par des êtres rachitiques, portant binocle, dont le cas d’exemption se lit sur le visage ! Contrairement à ce qui arrivait toujours jadis, si l’on veut reposer ses yeux sur quelqu’un de bien dans un restaurant, il ne faut plus regarder parmi les garçons qui servent mais parmi les clients qui consomment. Mais on pouvait revoir un servant, bien qu’ils changeassent souvent, mais allez donc savoir qui est et quand reviendra ce lieutenant anglais qui vient pour la première fois et sera peut-être tué demain. Quand Auguste de Pologne, comme raconte le charmant Morand, l’auteur délicieux de Clarisse, échangea un de ses régiments contre une collection de potiches chinoises, il fit à mon avis une mauvaise affaire. Pensez que tous ces grands valets de pied qui avaient deux mètres de haut et qui ornaient les escaliers monumentaux de nos plus belles amies ont tous été tués, engagés pour la plupart parce qu’on leur répétait que la guerre durerait deux mois. Ah ! ils ne savaient pas comme moi la force de l’Allemagne, la vertu de la race prussienne, dit-il en s’oubliant — et puis, remarquant qu’il avait trop laissé voir son point de vue — ce n’est pas tant l’Allemagne que je crains pour la France que la guerre elle-même. Les gens de l’arrière s’imaginent que la guerre est seulement un gigantesque match de boxe auquel ils assistent de loin, grâce aux journaux. Mais cela n’a aucun rapport. C’est une maladie qui quand elle semble conjurée sur un point reprend sur un autre. Aujourd’hui Noyon sera délivré, demain on n’aura plus ni pain ni chocolat, après-demain celui qui se croyait tranquille et accepterait au besoin une balle qu’il n’imagine pas s’affolera parce qu’il lira dans les journaux que sa classe est rappelée. Quant aux monuments, un chef-d’œuvre unique comme Reims par la qualité n’est pas tellement ce dont la disparition m’épouvante, c’est surtout de voir anéantis une telle quantité d’ensembles qui rendaient le moindre village de France instructif et charmant. » Je pensai aussitôt à Combray et qu’autrefois j’aurais cru me diminuer aux yeux de Mme de Guermantes en avouant la petite situation que ma famille occupait à Combray. Je me demandai si elle n’avait pas été révélée aux Guermantes et à M. de Charlus, soit par Legrandin, ou Swann, ou Saint-Loup, ou Morel. Mais cette prétérition même était moins pénible pour moi que des explications rétrospectives. Je souhaitai seulement que M. de Charlus ne parlât pas de Combray. « Je ne veux pas dire de mal des Américains, Monsieur, continua-t-il, il paraît qu’ils sont inépuisablement généreux, et comme il n’y a pas eu de chef d’orchestre dans cette guerre, que chacun est entré dans la danse longtemps après l’autre, et que les Américains ont commencé quand nous étions quasiment finis, ils peuvent avoir une ardeur que quatre ans de guerre ont pu calmer chez nous. Même avant la guerre ils aimaient notre pays, notre art, ils payaient fort cher nos chefs-d’œuvre. Beaucoup sont chez eux maintenant. Mais précisément cet art déraciné, comme dirait M. Barrès, est tout le contraire de ce qui faisait l’agrément délicieux de la France. Le château expliquait l’église qui, elle-même, parce qu’elle avait été un lieu de pèlerinage, expliquait la chanson de geste. Je n’ai pas à surfaire l’illustration de mes origines et de mes alliances, et d’ailleurs ce n’est pas de cela qu’il s’agit. Mais dernièrement j’ai eu à régler une question d’intérêts, et, malgré un certain refroidissement qu’il y a entre le ménage et moi, à aller faire une visite à ma nièce Saint-Loup qui habite à Combray. Combray n’était qu’une toute petite ville comme il y en a tant. Mais nos ancêtres étaient représentés en donateurs dans certains vitraux, dans d’autres étaient inscrites nos armoiries. Nous y avions notre chapelle, nos tombeaux. Cette église a été détruite par les Français et par les Anglais parce qu’elle servait d’observatoire aux Allemands. Tout ce mélange d’histoire survivante et d’art, qui était la France, se détruit, et ce n’est pas fini. Et, bien entendu, je n’ai pas le ridicule de comparer, pour des raisons de famille, la destruction de l’église de Combray à celle de la cathédrale de Reims, qui était comme le miracle d’une cathédrale gothique retrouvant naturellement la pureté de la statuaire antique, ou de celle d’Amiens. Je ne sais si le bras levé de Saint Firmin est aujourd’hui brisé. Dans ce cas la plus haute affirmation de la foi et de l’énergie a disparu de ce monde. — Son symbole, Monsieur, lui répondis-je. Et j’adore autant que vous certains symboles. Mais il serait absurde de sacrifier au symbole la réalité qu’il symbolise. Les cathédrales doivent être adorées jusqu’au jour où, pour les préserver, il faudrait renier les vérités qu’elles enseignent. Le bras levé de Saint Firmin dans un geste de commandement presque militaire disait : Que nous soyons brisés si l’honneur l’exige. Ne sacrifiez pas des hommes à des pierres dont la beauté vient justement d’avoir un moment fixé des vérités humaines. — Je comprends ce que vous voulez dire, me répondit M. de Charlus, et M. Barrès, qui nous a fait, hélas, trop faire de pèlerinages à la statue de Strasbourg et au tombeau de M. Déroulède, a été touchant et gracieux quand il a écrit que la cathédrale de Reims elle-même nous était moins chère que la vie de nos fantassins. Assertion qui rend assez ridicule la colère de nos journaux contre le général allemand qui commandait là-bas et qui disait que la cathédrale de Reims lui était moins précieuse que celle d’un soldat allemand. C’est, du reste, ce qui est exaspérant et navrant, c’est que chaque pays dit la même chose. Les raisons pour lesquelles les associations industrielles de l’Allemagne déclarent la possession de Belfort indispensable à préserver leur nation contre nos idées de revanche sont les mêmes que celles de Barrès exigeant Mayence pour nous protéger contre les velléités d’invasion des Boches. Pourquoi la restitution de l’Alsace-Lorraine a-t-elle paru à la France un motif insuffisant pour faire la guerre, un motif suffisant pour la continuer, pour la redéclarer à nouveau chaque année ? Vous avez l’air de croire que la victoire est désormais promise à la France, je le souhaite de tout mon cœur, vous n’en doutez pas, mais enfin, depuis qu’à tort ou à raison les Alliés se croient sûrs de vaincre (pour ma part je serais naturellement enchanté de cette solution, mais je vois surtout beaucoup de victoires sur le papier, de victoires à la Pyrrhus, avec un coût qui ne nous est pas dit) et que les Boches ne se croient plus sûrs de vaincre, on voit l’Allemagne chercher à hâter la paix, la France à prolonger la guerre, la France qui est la France juste et a raison de faire entendre des paroles de justice, mais est aussi la douce France et devrait faire entendre des paroles de pitié, fût-ce seulement pour ses propres enfants et pour qu’à chaque printemps les fleurs qui renaîtront aient autre chose à éclairer que des tombes. Soyez franc, mon cher ami, vous-même m’aviez fait une théorie sur les choses qui n’existent que grâce à une création perpétuellement recommencée. La création du monde n’a pas eu lieu une fois pour toutes, me disiez-vous, elle a nécessairement lieu tous les jours. Hé bien, si vous êtes de bonne foi, vous ne pouvez pas excepter la guerre de cette théorie. Notre excellent Norpois a beau écrire — en sortant un des accessoires de rhétorique qui lui sont aussi chers que « l’aube de la victoire » et le « Général Hiver » : — « Maintenant que l’Allemagne a voulu la guerre », « Les dés en sont jetés », la vérité c’est que chaque matin on déclare à nouveau la guerre. Donc celui qui veut la continuer est aussi coupable que celui qui l’a commencée, plus peut-être car ce premier n’en prévoyait peut-être pas toutes les horreurs. Or rien ne dit qu’une guerre aussi prolongée, même si elle doit avoir une issue victorieuse, ne soit pas sans péril. Il est difficile de parler de choses qui n’ont point de précédent et des répercussions sur l’organisme d’une opération qu’on tente pour la première fois. Généralement, il est vrai, ces nouveautés dont on s’alarme se passent fort bien. Les républicains les plus sages pensaient qu’il était fou de faire la séparation de l’Église. Elle a passé comme une lettre à la poste. Dreyfus a été réhabilité, Picquart ministre de la guerre, sans qu’on crie ouf. Pourtant que ne peut-on pas craindre d’un surmenage pareil à celui d’une guerre ininterrompue pendant plusieurs années ! Que feront les hommes au retour ? seront-ils las ? la fatigue les aura-t-elle rompus ou affolés ? Tout cela pourrait mal tourner, sinon pour la France, au moins pour le gouvernement, peut-être même pour la forme du gouvernement. Vous m’avez fait lire autrefois l’admirable Aimée de Coigny de Maurras. Je serais fort surpris que quelque Aimée de Coigny n’attendît pas du développement de la guerre que fait la République ce qu’en 1812 Aimée de Coigny attendit de la guerre que faisait l’Empire. Si l’Aimée actuelle existe, ses espérances se réaliseront-elles ? Je ne le désire pas. Pour en revenir à la guerre elle-même, le premier qui l’a commencée est-il l’empereur Guillaume ? J’en doute fort. Et si c’est lui, qu’a-t-il fait autre chose que Napoléon par exemple, chose que moi je trouve abominable mais que je m’étonne de voir inspirer tant d’horreurs aux thuriféraires de Napoléon, aux gens qui, le jour de la déclaration de guerre, se sont écriés comme le général X. : « J’attendais ce jour-là depuis quarante ans. C’est le plus beau jour de ma vie. » Dieu sait si personne a protesté avec plus de force que moi quand on a fait dans la société une place disproportionnée aux nationalistes, aux militaires, quand tout ami des arts était accusé de s’occuper de choses funestes à la patrie, toute civilisation qui n’était pas belliqueuse étant délétère. C’est à peine si un homme du monde authentique comptait auprès d’un général. Une folle faillit me présenter à M. Syveton. Vous me direz que ce que je m’efforçais de maintenir n’était que les règles mondaines. Mais, malgré leur frivolité apparente, elles eussent peut-être empêché bien des excès. J’ai toujours honoré ceux qui défendent la grammaire, ou la logique. On se rend compte cinquante ans après qu’ils ont conjuré de grands périls. Or nos nationalistes sont les plus germanophobes, les plus jusqu’auboutistes des hommes… Mais après quinze ans leur philosophie a changé entièrement. En fait, ils poussent bien à la continuation de la guerre. Mais ce n’est que pour exterminer une race belliqueuse et par amour de la paix. Car une civilisation guerrière, ce qu’ils trouvaient si beau il y a quinze ans, leur fait horreur ; non seulement ils reprochent à la Prusse d’avoir fait prédominer chez elle l’élément militaire, mais en tout temps ils pensent que les civilisations militaires furent destructrices de tout ce qu’ils trouvent maintenant précieux, non seulement les arts, mais même la galanterie. Il suffit qu’un de leurs critiques se soit converti au nationalisme pour qu’il soit devenu du même coup un ami de la paix… Il est persuadé que, dans toutes les civilisations guerrières, la femme avait un rôle humilié et bas. On n’ose lui répondre que les « Dames » des chevaliers au moyen âge et la Béatrice de Dante étaient peut-être placées sur un trône aussi élevé que les héroïnes de M. Becque. Je m’attends un de ces jours à me voir placé à table après un révolutionnaire russe ou simplement après un de nos généraux faisant la guerre par horreur de la guerre et pour punir un peuple de cultiver un idéal qu’eux-mêmes jugeaient le seul tonifiant il y a quinze ans. Le malheureux Tzar était encore honoré il y a quelques mois parce qu’il avait réuni la conférence de La Haye. Mais maintenant qu’on salue la Russie libre, on oublie le titre qui permettait de la glorifier. Ainsi tourne la Roue du Monde. Et pourtant l’Allemagne emploie tellement les mêmes expressions que la France que c’est à croire qu’elle la cite, elle ne se lasse pas de dire qu’elle « lutte pour l’existence ». Quand je lis : « nous luttons contre un ennemi implacable et cruel jusqu’à ce que nous ayons obtenu une paix qui nous garantisse l’avenir de toute agression et pour que le sang de nos braves soldats n’ait pas coulé en vain », ou bien : « qui n’est pas pour nous est contre nous », je ne sais pas si cette phrase est de l’Empereur Guillaume ou de M. Poincaré, car ils l’ont, à quelques variantes près, prononcée vingt fois l’un et l’autre, bien qu’à vrai dire je doive confesser que l’Empereur ait été en ce cas l’imitateur du Président de la République. La France n’aurait peut-être pas tenu tant à prolonger la guerre si elle était restée faible, mais surtout l’Allemagne n’aurait peut-être pas été si pressée de la finir si elle n’avait pas cessé d’être forte. D’être aussi forte, car forte, vous verrez qu’elle l’est encore. » Il avait pris l’habitude de crier très fort en parlant, par nervosité, par recherche d’issue pour des impressions dont il fallait — n’ayant jamais cultivé aucun art — qu’il se débarrassât, comme un aviateur de ses bombes, fût-ce en plein champ, là où ses paroles n’atteignaient personne, et surtout dans le monde où elles tombaient au hasard et où il était écouté par snobisme, de confiance et, tant il tyrannisait les auditeurs, on peut dire de force et même par crainte. Sur les boulevards cette harangue était de plus une marque de mépris à l’égard des passants pour qui il ne baissait pas plus la voix qu’il n’eût dévié son chemin. Mais elle y détonnait, y étonnait et surtout rendait intelligibles à des gens qui se retournaient des propos qui eussent pu nous faire prendre pour des défaitistes. Je le fis remarquer à M. de Charlus sans réussir qu’à exciter son hilarité. « Avouez que ce serait bien drôle, dit-il. Après tout, ajouta-t-il, on ne sait jamais, chacun de nous risque chaque soir d’être le fait divers du lendemain. En somme, pourquoi ne serais-je pas fusillé dans les fossés de Vincennes ? La même chose est bien arrivée à mon grand-oncle le duc d’Enghien. La soif du sang noble affole une certaine populace qui en cela se montre plus raffinée que les lions. Vous savez que pour ces animaux il suffirait pour qu’ils se jetassent sur elle que Mme Verdurin eût une écorchure sur son nez. Sur ce que dans ma jeunesse on eût appelé son pif ! » Et il se mit à rire à gorge déployée comme si nous avions été seuls dans un salon. Par moments, voyant des individus assez louches extraits de l’ombre par le passage de M. de Charlus se conglomérer à quelque distance de lui, je me demandais si je lui serais plus agréable en le laissant seul ou en ne le quittant pas. Tel celui qui a rencontré un vieillard sujet à de fréquentes crises épileptiformes et qui voit, par l’incohérence de la démarche, l’imminence probable d’un accès se demande si sa compagnie est plutôt désirée comme celle d’un soutien, ou redoutée comme celle d’un témoin à qui on voudrait cacher la crise et dont la présence seule peut-être, quand le calme absolu réussirait à l’écarter, suffira à la hâter. Mais la possibilité de l’événement duquel on ne sait si l’on doit s’écarter ou non est révélée, chez le malade, par les circuits qu’il fait comme un homme ivre. Tandis que pour M. de Charlus les diverses positions divergentes, signe d’un incident possible dont je n’étais pas bien sûr s’il souhaitait ou redoutait que ma présence l’empêchât de se produire, étaient, par une ingénieuse mise en scène, occupées non par le baron lui-même, qui marchait fort droit, mais par tout un cercle de figurants. Tout de même, je crois qu’il préférait éviter la rencontre, car il m’entraîna dans une rue de traverse, plus obscure que le boulevard et où celui-ci ne cessait de déverser des soldats de toute arme et de toute nation, influx juvénile, compensateur et consolant, pour M. de Charlus, de ce reflux de tous les hommes à la frontière qui avait fait frénétiquement le vide dans Paris aux premiers temps de la mobilisation. M. de Charlus ne cessait pas d’admirer les brillants uniformes qui passaient devant nous et qui faisaient de Paris une ville aussi cosmopolite qu’un port, aussi irréelle qu’un décor de peintre qui n’a dressé quelques architectures que pour avoir un prétexte à grouper les costumes les plus variés et les plus chatoyants. Il gardait tout son respect et toute son affection à de grandes dames accusées de défaitisme, comme jadis à celles qui avaient été accusées de dreyfusisme. Il regrettait seulement qu’en s’abaissant à faire de la politique elles eussent donné prise « aux polémiques des journalistes ». Pour lui, à leur égard, rien n’était changé. Car sa frivolité était si systématique, que la naissance unie à la beauté et à d’autres prestiges était la chose durable — et la guerre, comme l’affaire Dreyfus, des modes vulgaires et fugitives. Eût-on fusillé la duchesse de Guermantes pour essai de paix séparée avec l’Autriche qu’il l’eût considérée comme toujours aussi noble et pas plus dégradée que ne nous apparaît aujourd’hui Marie-Antoinette d’avoir été condamnée à la décapitation. En parlant à ce moment-là, M. de Charlus, noble comme une espèce de Saint-Vallier ou de Saint-Mégrin, était droit, rigide, solennel, parlait gravement, ne faisait pour un moment aucune des manières où se révèlent ceux de sa sorte. Et pourtant, pourquoi ne peut-il y en avoir aucun dont la voix soit jamais absolument juste ?… Même en ce moment où elle approchait le plus du grave, elle était fausse encore et aurait eu besoin de l’accordeur. D’ailleurs, M. de Charlus ne savait littéralement où donner de la tête et il la levait souvent avec le regret de ne pas avoir une jumelle qui, d’ailleurs, ne lui eût pas servi à grand’chose, car en plus grand nombre que d’habitude, à cause du raid de zeppelins de l’avant-veille qui avait réveillé la vigilance des pouvoirs publics, il y avait des militaires jusque dans le ciel. Les aéroplanes que j’avais vus quelques heures plus tôt faire, comme des insectes, des taches brunes sur le soir bleu passaient maintenant dans la nuit qu’approfondissait encore l’extinction partielle des réverbères comme de lumineux brûlots. La plus grande impression de beauté que nous faisaient éprouver ces étoiles humaines et filantes était peut-être surtout de faire regarder le ciel vers lequel on lève peu les yeux d’habitude dans ce Paris dont, en 1914, j’avais vu la beauté presque sans défense attendre la menace de l’ennemi qui se rapprochait. Il y avait certes, maintenant comme alors, la splendeur antique inchangée d’une lune cruellement, mystérieusement sereine, qui versait aux monuments encore intacts l’inutile beauté de sa lumière, mais comme en 1914, et plus qu’en 1914, il y avait aussi autre chose, des lumières différentes et des feux intermittents, que soit de ces aéroplanes, soit des projecteurs de la Tour Eiffel on savait dirigés par une volonté intelligente, par une vigilance amie qui donnait ce même genre d’émotion, inspirait cette même sorte de reconnaissance et de calme que j’avais éprouvés dans la chambre de Saint-Loup, dans la cellule de ce cloître militaire où s’exerçaient, avant qu’ils consommassent un jour, sans une hésitation, en pleine jeunesse, leur sacrifice, tant de cœurs fervents et disciplinés.

Après le raid de l’avant-veille, où le ciel avait été plus mouvementé que la terre, il s’était calmé comme la mer après une tempête. Mais comme la mer après une tempête il n’avait pas encore repris son apaisement absolu. Des aéroplanes montaient encore comme des fusées rejoindre les étoiles et des projecteurs promenaient lentement, dans le ciel sectionné, comme une pâle poussière d’astres, d’errantes voies lactées. Cependant les aéroplanes venaient s’insérer au milieu des constellations et on aurait pu se croire dans un autre hémisphère en effet, en voyant ces « étoiles nouvelles ». M. de Charlus me dit son admiration pour ces aviateurs, et comme il ne pouvait pas plus s’empêcher de donner libre cours à sa germanophilie qu’à ses autres penchants tout en niant l’une comme les autres : « D’ailleurs j’ajoute que j’admire autant les Allemands qui montent dans des gothas. Et sur des zeppelins, pensez le courage qu’il faut. Mais ce sont des héros tout simplement. Qu’est-ce que ça peut faire que ce soit sur des civils qu’ils lancent leurs bombes puisque ces batteries tirent sur eux ? Est-ce que vous avez peur des gothas et du canon ? » J’avouai que non et peut-être je me trompais. Sans doute ma paresse m’ayant donné l’habitude, pour mon travail, de le remettre jour par jour au lendemain, je me figurais qu’il pouvait en être de même pour la mort. Comment aurait-on peur d’un canon dont on est persuadé qu’il ne vous frappera pas ce jour-là ? D’ailleurs formées isolément, ces idées de bombes lancées, de mort possible n’ajoutèrent pour moi rien de tragique à l’image que je me faisais du passage des aéronefs allemands jusqu’à ce que j’eusse vu de l’un d’eux ballotté, segmenté à mes regards par les flots de brume d’un ciel agité, d’un aéroplane que, bien que je le susse meurtrier, je n’imaginais que stellaire et céleste, j’eusse vu un soir le geste de la bombe lancée vers nous. Car la réalité originale d’un danger n’est perçue que de cette chose nouvelle, irréductible à ce qu’on sait déjà, qui s’appelle une impression et qui est souvent, comme ce fut le cas là, résumée par une ligne, une ligne qui découvrait une intention, une ligne où il y avait la puissance latente d’un accomplissement qui la déformait, tandis que sur le pont de la Concorde, autour de l’aéroplane menaçant et tragique, et comme si s’étaient reflétées dans les nuages les fontaines des Champs-Élysées, de la place de la Concorde et des Tuileries, les jets d’eau lumineux des projecteurs s’infléchissaient dans le ciel, lignes pleines d’intentions aussi, d’intentions prévoyantes et protectrices, d’hommes puissants et sages auxquels, comme la nuit au quartier de Doncières, j’étais reconnaissant que leur force daignât prendre, avec cette précision si belle, la peine de veiller sur nous.

La nuit était aussi belle qu’en 1914, comme Paris était aussi menacé. Le clair de lune semblait comme un doux magnésium continu permettant de prendre une dernière fois des images nocturnes de ces beaux ensembles comme la place Vendôme, la place de la Concorde, auxquels l’effroi que j’avais des obus qui allaient peut-être les détruire donnait, par contraste, dans leur beauté encore intacte, une sorte de plénitude, comme si elles se tendaient en avant, offrant aux coups leurs architectures sans défense. « Vous n’avez pas peur, répéta M. de Charlus. Les Parisiens ne se rendent pas compte. On me dit que Mme Verdurin donne des réunions tous les jours. Je ne le sais que par les on-dit, moi je ne sais absolument rien d’eux, j’ai entièrement rompu », ajouta-t-il en baissant non seulement les yeux comme si avait passé un télégraphiste, mais aussi la tête, les épaules, et en levant le bras avec le geste qui signifie sinon « je m’en lave les mains », du moins « je ne peux rien vous dire » (bien que je ne lui demandasse rien). « Je sais que Morel y va toujours beaucoup », me dit-il (c’était la première fois qu’il m’en reparlait). « On prétend qu’il regrette beaucoup le passé, qu’il désire se rapprocher de moi », ajouta-t-il, faisant preuve à la fois de cette même crédulité d’homme du faubourg qui dit : « On dit beaucoup que la France cause plus que jamais avec l’Allemagne et que les pourparlers sont même engagés » et de l’amoureux que les pires rebuffades n’ont pas persuadé. « En tout cas, s’il le veut il n’a qu’à le dire, je suis plus vieux que lui, ce n’est pas à moi à faire les premiers pas. » Et sans doute il était bien inutile de le dire tant c’était évident. Mais, de plus, ce n’était même pas sincère, et c’est pour cela qu’on était si gêné pour M. de Charlus, car on sentait qu’en disant que ce n’était pas à lui de faire les premiers pas, il en faisait au contraire un et attendait que j’offrisse de me charger du rapprochement. Certes, je connaissais cette naïve ou feinte crédulité des gens qui aiment quelqu’un, ou simplement ne sont pas reçus chez quelqu’un, et imputent à ce quelqu’un un désir qu’il n’a pourtant pas manifesté, malgré des sollicitations fastidieuses.

Malheureusement, dès le lendemain, disons-le tout de suite, M. de Charlus se trouva dans la rue face à face avec Morel ; celui-ci, pour exciter sa jalousie, le prit par le bras, lui raconta des histoires plus ou moins vraies et quand M. de Charlus éperdu, ayant besoin que Morel restât cette soirée auprès de lui, le supplia de ne pas aller ailleurs, l’autre, apercevant un camarade, dit adieu à M. de Charlus qui, de colère, espérant que cette menace que, bien entendu, il semblait ne devoir exécuter jamais, ferait rester Morel, lui dit : « Prends garde, je me vengerai », et Morel, riant, partit en tapotant sur le cou et en enlaçant par la taille son camarade étonné.

À l’accent soudain tremblant avec lequel M. de Charlus avait, en me parlant de Morel, scandé ses paroles, au regard trouble qui vacillait au fond de ses yeux, j’eus l’impression qu’il y avait autre chose qu’une banale insistance. Je ne me trompais pas et je dirai tout de suite les deux faits qui me le prouvèrent rétrospectivement (j’anticipe de beaucoup d’années pour le second de ces faits, postérieur à la mort de M. de Charlus. Or elle ne devait se produire que bien plus tard, et nous aurons l’occasion de le revoir plusieurs fois, bien différent de ce que nous l’avons connu, et en particulier la dernière fois, à une époque où il avait entièrement oublié Morel). Quant au premier de ces faits, il se produisit deux ans seulement après le soir où je descendais ainsi les boulevards avec M. de Charlus. Donc environ deux ans après cette soirée, je rencontrai Morel. Je pensai aussitôt à M. de Charlus, au plaisir qu’il aurait à revoir le violoniste, et j’insistai auprès de lui pour qu’il allât le voir, fût-ce une fois. « Il a été bon pour vous, dis-je à Morel. Il est déjà vieux, il peut mourir, il faut liquider les vieilles querelles et effacer les traces de la brouille. » Morel parut entièrement de mon avis quant à un apaisement désirable, mais il n’en refusa pas moins catégoriquement de faire même une seule visite à M. de Charlus. « Vous avez tort, lui dis-je. Est-ce par entêtement, par paresse, par méchanceté, par amour-propre mal placé, par vertu (soyez sûr qu’elle ne sera pas attaquée), par coquetterie ? » Alors le violoniste, tordant son visage pour un aveu qui lui coûtait sans doute extrêmement, me répondit en frissonnant : « Non, ce n’est pour rien de tout cela, la vertu je m’en fous ; la méchanceté, au contraire je commence à le plaindre ; ce n’est pas par coquetterie, elle serait inutile ; ce n’est pas par paresse, il y a des journées entières où je reste à me tourner les pouces, non, ce n’est à cause de rien de tout cela ; c’est, ne le dites jamais à personne et je suis fou de vous le dire, c’est, c’est… c’est… par peur ! » Il se mit à trembler de tous ses membres. Je lui avouai que je ne le comprenais pas. « Non, ne me demandez pas, n’en parlons plus, vous ne le connaissez pas comme moi, je peux dire que vous ne le connaissez pas du tout. — Mais quel tort peut-il vous faire ? il cherchera, d’ailleurs, d’autant moins à vous en faire qu’il n’y aura plus de rancune entre vous. Et puis, au fond, vous savez qu’il est très bon. — Parbleu si, je le sais qu’il est bon ! Et la délicatesse et la droiture. Mais laissez-moi, ne m’en parlez plus, je vous en supplie, c’est honteux à dire, j’ai peur ! » Le second fait date d’après la mort de M. de Charlus. On m’apporta quelques souvenirs qu’il m’avait laissés et une lettre à triple enveloppe, écrite au moins dix ans avant sa mort. Mais il avait été gravement malade, avait pris ses dispositions, puis s’était rétabli avant de tomber plus tard dans l’état où nous le verrons le jour d’une matinée chez la princesse de Guermantes — et la lettre, restée dans un coffre avec les objets qu’il léguait à quelques amis, était restée là sept ans, sept ans pendant lesquels il avait entièrement oublié Morel. La lettre, tracée d’une écriture fine et ferme, était ainsi conçue : « Mon cher ami, les voies de la Providence sont inconnues. Parfois c’est du défaut d’un être médiocre qu’elle use pour empêcher de faillir la suréminence d’un juste. Vous connaissez Morel, d’où il est sorti, à quel faîte j’ai voulu l’élever, autant dire à mon niveau. Vous savez qu’il a préféré retourner non pas à la poussière et à la cendre d’où tout homme, c’est-à-dire le véritable phœnix, peut renaître, mais à la boue où rampe la vipère. Il s’est laissé choir, ce qui m’a préservé de déchoir. Vous savez que mes armes contiennent la devise même de Notre-Seigneur : « Inculcabis super leonem et aspidem » avec un homme représenté comme ayant à la plante de ses pieds, comme support héraldique, un lion et un serpent. Or si j’ai pu fouler ainsi le propre lion que je suis, c’est grâce au serpent et à sa prudence, qu’on appelle trop légèrement parfois un défaut, car la profonde sagesse de l’Évangile en fait une vertu, au moins une vertu pour les autres. Notre serpent aux sifflements jadis harmonieusement modulés, quand il avait un charmeur — fort charmé, du reste — n’était pas seulement musical et reptile, il avait jusqu’à la lâcheté cette vertu que je tiens maintenant pour divine, la Prudence. C’est cette divine prudence qui l’a fait résister aux appels que je lui ai fait transmettre de revenir me voir, et je n’aurai de paix en ce monde et d’espoir de pardon dans l’autre que si je vous en fais l’aveu. C’est lui qui a été en cela l’instrument de la Sagesse divine, car, je l’avais résolu, il ne serait pas sorti de chez moi vivant. Il fallait que l’un de nous deux disparût. J’étais décidé à le tuer. Dieu lui a conseillé la prudence pour me préserver d’un crime. Je ne doute pas que l’intercession de l’Archange Michel, mon saint patron, n’ait joué là un grand rôle et je le prie de me pardonner de l’avoir tant négligé pendant plusieurs années et d’avoir si mal répondu aux innombrables bontés qu’il m’a témoignées, tout spécialement dans ma lutte contre le mal. Je dois à ce serviteur, je le dis dans la plénitude de ma foi et de mon intelligence, que le Père céleste ait inspiré à Morel de ne pas venir. Aussi, c’est moi maintenant qui me meurs. Votre fidèlement dévoué, Semper idem, P. G. Charlus. » Alors je compris la peur de Morel ; certes il y avait dans cette lettre bien de l’orgueil et de la littérature. Mais l’aveu était vrai. Et Morel savait mieux que moi que le « côté presque fou » que Mme de Guermantes trouvait chez son beau-frère ne se bornait pas, comme je l’avais cru jusque-là, à ces dehors momentanés de rage superficielle et inopérante.
