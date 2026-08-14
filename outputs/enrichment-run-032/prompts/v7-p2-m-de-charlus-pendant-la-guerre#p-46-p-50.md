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

À l’accent soudain tremblant avec lequel M. de Charlus avait, en me parlant de Morel, scandé ses paroles, au regard trouble qui vacillait au fond de ses yeux, j’eus l’impression qu’il y avait autre chose qu’une banale insistance. Je ne me trompais pas et je dirai tout de suite les deux faits qui me le prouvèrent rétrospectivement (j’anticipe de beaucoup d’années pour le second de ces faits, postérieur à la mort de M. de Charlus. Or elle ne devait se produire que bien plus tard, et nous aurons l’occasion de le revoir plusieurs fois, bien différent de ce que nous l’avons connu, et en particulier la dernière fois, à une époque où il avait entièrement oublié Morel). Quant au premier de ces faits, il se produisit deux ans seulement après le soir où je descendais ainsi les boulevards avec M. de Charlus. Donc environ deux ans après cette soirée, je rencontrai Morel. Je pensai aussitôt à M. de Charlus, au plaisir qu’il aurait à revoir le violoniste, et j’insistai auprès de lui pour qu’il allât le voir, fût-ce une fois. « Il a été bon pour vous, dis-je à Morel. Il est déjà vieux, il peut mourir, il faut liquider les vieilles querelles et effacer les traces de la brouille. » Morel parut entièrement de mon avis quant à un apaisement désirable, mais il n’en refusa pas moins catégoriquement de faire même une seule visite à M. de Charlus. « Vous avez tort, lui dis-je. Est-ce par entêtement, par paresse, par méchanceté, par amour-propre mal placé, par vertu (soyez sûr qu’elle ne sera pas attaquée), par coquetterie ? » Alors le violoniste, tordant son visage pour un aveu qui lui coûtait sans doute extrêmement, me répondit en frissonnant : « Non, ce n’est pour rien de tout cela, la vertu je m’en fous ; la méchanceté, au contraire je commence à le plaindre ; ce n’est pas par coquetterie, elle serait inutile ; ce n’est pas par paresse, il y a des journées entières où je reste à me tourner les pouces, non, ce n’est à cause de rien de tout cela ; c’est, ne le dites jamais à personne et je suis fou de vous le dire, c’est, c’est… c’est… par peur ! » Il se mit à trembler de tous ses membres. Je lui avouai que je ne le comprenais pas. « Non, ne me demandez pas, n’en parlons plus, vous ne le connaissez pas comme moi, je peux dire que vous ne le connaissez pas du tout. — Mais quel tort peut-il vous faire ? il cherchera, d’ailleurs, d’autant moins à vous en faire qu’il n’y aura plus de rancune entre vous. Et puis, au fond, vous savez qu’il est très bon. — Parbleu si, je le sais qu’il est bon ! Et la délicatesse et la droiture. Mais laissez-moi, ne m’en parlez plus, je vous en supplie, c’est honteux à dire, j’ai peur ! » Le second fait date d’après la mort de M. de Charlus. On m’apporta quelques souvenirs qu’il m’avait laissés et une lettre à triple enveloppe, écrite au moins dix ans avant sa mort. Mais il avait été gravement malade, avait pris ses dispositions, puis s’était rétabli avant de tomber plus tard dans l’état où nous le verrons le jour d’une matinée chez la princesse de Guermantes — et la lettre, restée dans un coffre avec les objets qu’il léguait à quelques amis, était restée là sept ans, sept ans pendant lesquels il avait entièrement oublié Morel. La lettre, tracée d’une écriture fine et ferme, était ainsi conçue : « Mon cher ami, les voies de la Providence sont inconnues. Parfois c’est du défaut d’un être médiocre qu’elle use pour empêcher de faillir la suréminence d’un juste. Vous connaissez Morel, d’où il est sorti, à quel faîte j’ai voulu l’élever, autant dire à mon niveau. Vous savez qu’il a préféré retourner non pas à la poussière et à la cendre d’où tout homme, c’est-à-dire le véritable phœnix, peut renaître, mais à la boue où rampe la vipère. Il s’est laissé choir, ce qui m’a préservé de déchoir. Vous savez que mes armes contiennent la devise même de Notre-Seigneur : « Inculcabis super leonem et aspidem » avec un homme représenté comme ayant à la plante de ses pieds, comme support héraldique, un lion et un serpent. Or si j’ai pu fouler ainsi le propre lion que je suis, c’est grâce au serpent et à sa prudence, qu’on appelle trop légèrement parfois un défaut, car la profonde sagesse de l’Évangile en fait une vertu, au moins une vertu pour les autres. Notre serpent aux sifflements jadis harmonieusement modulés, quand il avait un charmeur — fort charmé, du reste — n’était pas seulement musical et reptile, il avait jusqu’à la lâcheté cette vertu que je tiens maintenant pour divine, la Prudence. C’est cette divine prudence qui l’a fait résister aux appels que je lui ai fait transmettre de revenir me voir, et je n’aurai de paix en ce monde et d’espoir de pardon dans l’autre que si je vous en fais l’aveu. C’est lui qui a été en cela l’instrument de la Sagesse divine, car, je l’avais résolu, il ne serait pas sorti de chez moi vivant. Il fallait que l’un de nous deux disparût. J’étais décidé à le tuer. Dieu lui a conseillé la prudence pour me préserver d’un crime. Je ne doute pas que l’intercession de l’Archange Michel, mon saint patron, n’ait joué là un grand rôle et je le prie de me pardonner de l’avoir tant négligé pendant plusieurs années et d’avoir si mal répondu aux innombrables bontés qu’il m’a témoignées, tout spécialement dans ma lutte contre le mal. Je dois à ce serviteur, je le dis dans la plénitude de ma foi et de mon intelligence, que le Père céleste ait inspiré à Morel de ne pas venir. Aussi, c’est moi maintenant qui me meurs. Votre fidèlement dévoué, Semper idem, P. G. Charlus. » Alors je compris la peur de Morel ; certes il y avait dans cette lettre bien de l’orgueil et de la littérature. Mais l’aveu était vrai. Et Morel savait mieux que moi que le « côté presque fou » que Mme de Guermantes trouvait chez son beau-frère ne se bornait pas, comme je l’avais cru jusque-là, à ces dehors momentanés de rage superficielle et inopérante.

### Passage

Mais il faut revenir en arrière. Je descends les boulevards à côté de M. de Charlus, lequel vient de me prendre comme vague intermédiaire pour des ouvertures de paix entre lui et Morel. Voyant que je ne lui répondais pas, il continua ainsi : « Je ne sais pas, du reste, pourquoi il ne joue pas, on ne fait plus de musique sous prétexte que c’est la guerre, mais on danse, on dîne en ville. Les fêtes remplissent ce qui sera peut-être, si les Allemands avancent encore, les derniers jours de notre Pompéi. Pour peu que la lave de quelque Vésuve allemand (leurs pièces de marine ne sont pas moins terribles qu’un volcan) vienne les surprendre à leur toilette et éternise leur geste en l’interrompant, les enfants s’instruiront plus tard en regardant dans les livres de classes illustrés Mme Molé qui allait mettre une dernière couche de fard avant d’aller dîner chez une belle-sœur, ou Sosthène de Guermantes finissant de peindre ses faux sourcils ; ce sera matière à cours pour les Brichot de l’avenir ; la frivolité d’une époque quand dix siècles ont passé sur elle est digne de la plus grave érudition, surtout si elle a été conservée intacte par une éruption volcanique ou des matières analogues à la lave projetées par bombardement. Quels documents pour l’histoire future, quand les gaz asphyxiants analogues à ceux qu’émettait le Vésuve et des écroulements comme ceux qui ensevelirent Pompéi garderont intactes toutes les dernières imprudentes qui n’ont pas fait encore filer pour Bayonne leurs tableaux et leurs statues. D’ailleurs, n’est-ce pas déjà, depuis un an, Pompéi par fragments, chaque soir, que ces gens se sauvant dans les caves, non pas pour en rapporter quelque vieille bouteille de Mouton Rothschild ou de Saint-Émilion, mais pour cacher avec eux ce qu’ils ont de plus précieux, comme les prêtres d’Herculanum surpris par la mort au moment où ils emportaient les vases sacrés. C’est toujours l’attachement à l’objet qui amène la mort du possesseur. Paris, lui, ne fut pas, comme Herculanum, fondé par Hercule. Mais que de ressemblances s’imposent ! et cette lucidité qui nous est donnée n’est pas que de notre époque, chacune l’a possédée. Si je pense que nous pouvons avoir demain le sort des villes du Vésuve, celles-ci sentaient qu’elles étaient menacées du sort des villes maudites de la Bible. On a retrouvé sur les murs d’une des maisons de Pompéi cette inscription révélatrice : « Sodoma, Gomora. » Je ne sais si ce fut ce nom de Sodome et les idées qu’il éveilla en lui, soit celle du bombardement, qui firent que M. de Charlus leva un instant les yeux au ciel, mais il les ramena bientôt sur la terre. « J’admire tous les héros de cette guerre, dit-il. Tenez, mon cher, les soldats anglais que j’ai un peu légèrement considérés au début de la guerre comme de simples joueurs de football assez présomptueux pour se mesurer avec des professionnels — et quels professionnels ! — hé bien, rien qu’esthétiquement ce sont des athlètes de la Grèce, vous entendez bien, de la Grèce, mon cher, ce sont les jeunes gens de Platon, ou plutôt des Spartiates. J’ai un ami qui est allé à Rouen où ils ont leur camp, il a vu des merveilles, de pures merveilles dont on n’a pas idée. Ce n’est plus Rouen, c’est une autre ville. Évidemment il y a aussi l’ancien Rouen, avec les Saints émaciés de la cathédrale. Bien entendu, c’est beau aussi, mais c’est autre chose. Et nos poilus ! je ne peux pas vous dire quelle saveur je trouve en nos poilus, aux petits Parigots, tenez, comme celui qui passe là, avec son air dessalé, sa mine éveillée et drôle. Il m’arrive souvent de les arrêter, de faire un brin de causette avec eux, quelle finesse, quel bon sens ! et les gars de province, comme ils sont amusants et gentils avec leur roulement d’r et leur jargon patoiseur !… Moi, j’ai toujours beaucoup vécu à la campagne, couché dans les fermes, je sais leur parler, mais notre admiration pour les Français ne doit pas nous faire déprécier nos ennemis, ce serait nous diminuer nous-mêmes. Et vous ne savez pas quel soldat est le soldat allemand, vous ne l’avez pas vu comme moi défiler au pas de parade, au pas de l’oie, « unter den Linden ». En revenant à l’idéal de virilité qu’il m’avait esquissé à Balbec et qui avec le temps avait pris chez lui une forme philosophique, usant, d’ailleurs, de raisonnements absurdes, qui par moments, même quand il venait d’être supérieur, laissaient voir la trame trop mince du simple homme du monde, bien qu’homme du monde intelligent : « Voyez-vous, me dit-il, le superbe gaillard qu’est le soldat boche est un être fort, sain, ne pensant qu’à la grandeur de son pays, « Deutschland über alles », ce qui n’est pas si bête, et tandis qu’ils se préparaient virilement, nous nous sommes abîmés dans le dilettantisme. » Ce mot signifiait probablement pour M. de Charlus quelque chose d’analogue à la littérature, car aussitôt se rappelant sans doute que j’aimais les lettres et avais eu un moment l’intention de m’y adonner, il me tapa sur l’épaule (profitant du geste pour s’y appuyer jusqu’à me faire aussi mal qu’autrefois, quand je faisais mon service militaire, le recul contre l’omoplate du « 76 »), il me dit comme pour adoucir le reproche : « Oui, nous nous sommes abîmés dans le dilettantisme, nous tous, vous aussi, rappelez-vous, vous pouvez faire comme moi votre mea culpa, nous avons été trop dilettantes. » Par surprise du reproche, manque d’esprit de repartie, déférence envers mon interlocuteur et attendrissement pour son amicale bonté, je répondis comme si, ainsi qu’il m’y invitait, j’avais aussi à me frapper la poitrine, ce qui était parfaitement stupide car je n’avais pas l’ombre de dilettantisme à me reprocher. « Allons, me dit-il, je vous quitte (le groupe qui l’avait escorté de loin ayant fini par nous abandonner). Je m’en vais me coucher comme un très vieux Monsieur, d’autant plus qu’il paraît que la guerre a changé toutes nos habitudes, un de ces aphorismes qu’affectionne Norpois. » Je savais, du reste, qu’en rentrant chez lui M. de Charlus ne cessait pas pour cela d’être au milieu des soldats, car il avait transformé son hôtel en hôpital militaire, cédant du reste, je le crois, aux besoins bien moins de son imagination que de son bon cœur.

Il faisait une nuit transparente et sans un souffle. J’imaginais que la Seine coulant entre ses ponts circulaires, faits de leur plateau et de son reflet, devait ressembler au Bosphore. Et symbole soit de cette invasion que prédisait le défaitisme de M. de Charlus, soit de la coopération de nos frères musulmans avec les armées de la France, la lune étroite et recourbée comme un sequin semblait mettre le ciel parisien sous le signe oriental du croissant. Pour un instant encore il resta en arrêt devant un Sénégalais en me disant adieu et en me serrant la main à me la broyer, ce qui est une particularité allemande chez les gens qui sentent comme le baron, et en continuant pendant quelque temps à me la malaxer, eût dit jadis Cottard, comme si M. de Charlus avait voulu rendre à mes articulations une souplesse qu’elles n’avaient point perdue. Chez certains aveugles, le toucher supplée dans une certaine mesure à la vue. Je ne sais trop de quel sens il prenait la place ici. Il croyait peut-être seulement me serrer la main comme il crut sans doute ne faire que voir le Sénégalais qui passait dans l’ombre et ne daigna pas s’apercevoir qu’il était admiré. Mais, dans ces deux cas, le baron se trompait, il péchait par excès de contact et de regards. « Est-ce que tout l’Orient de Decamps, de Fromentin, d’Ingres, de Delacroix n’est pas là dedans ? me dit-il, encore immobilisé par le passage du Sénégalais. Vous savez, moi, je ne m’intéresse jamais aux choses et aux êtres qu’en peintre, en philosophe. D’ailleurs je suis trop vieux. Mais quel malheur, pour compléter le tableau, que l’un de nous deux ne soit pas une odalisque. » Ce ne fut pas l’Orient de Decamps, ni même de Delacroix qui commença de hanter mon imagination quand le baron m’eut quitté, mais le vieil Orient de ces Mille et une Nuits que j’avais tant aimées, et, me perdant peu à peu dans le lacis de ces rues noires, je pensais au calife Haroun Al Raschid en quête d’aventures dans les quartiers perdus de Bagdad. D’autre part, la chaleur du temps et de la marche m’avait donné soif, mais depuis longtemps tous les bars étaient fermés, et à cause de la pénurie d’essence les rares taxis que je rencontrais, conduits par des Levantins ou des Nègres, ne prenaient même pas la peine de répondre à mes signes. Le seul endroit où j’aurais pu me faire servir à boire et reprendre des forces pour rentrer chez moi eût été un hôtel. Mais dans la rue assez éloignée du centre où j’étais parvenu, tous, depuis que sur Paris les gothas lançaient leurs bombes, avaient fermé. Il en était de même de presque toutes les boutiques de commerçants, lesquels, faute d’employés ou eux-mêmes pris de peur, avaient fui à la campagne et laissé sur la porte un avertissement habituel écrit à la main et annonçant leur réouverture pour une époque éloignée et, d’ailleurs, problématique. Les autres établissements qui avaient pu survivre encore annonçaient de la même manière qu’ils n’ouvraient que deux fois par semaine. On sentait que la misère, l’abandon, la peur habitaient tout ce quartier. Je n’en fus que plus surpris de voir qu’entre ces maisons délaissées il y en avait une où la vie au contraire semblait avoir vaincu l’effroi, la faillite, et entretenait l’activité et la richesse. Derrière les volets clos de chaque fenêtre la lumière, tamisée à cause des ordonnances de police, décelait pourtant un insouci complet de l’économie. Et à tout instant la porte s’ouvrait pour laisser entrer ou sortir quelque visiteur nouveau. C’était un hôtel par qui la jalousie de tous les commerçants voisins (à cause de l’argent que ses propriétaires devaient gagner) devait être excitée ; et ma curiosité le fut aussi quand je vis sortir rapidement, à une quinzaine de mètres de moi, c’est-à-dire trop loin pour que dans l’obscurité profonde je pusse le reconnaître, un officier.

Quelque chose pourtant me frappa qui n’était pas sa figure que je ne voyais pas, ni son uniforme dissimulé dans une grande houppelande, mais la disproportion extraordinaire entre le nombre de points différents par où passa son corps et le petit nombre de secondes pendant lesquelles cette sortie, qui avait l’air de la sortie tentée par un assiégé, s’exécuta. De sorte que je pensai, si je ne le reconnus pas formellement — je ne dirai pas même à la tournure ni à la sveltesse, ni à l’allure, ni à la vélocité de Saint-Loup — mais à l’espèce d’ubiquité qui lui était si spéciale. Le militaire capable d’occuper en si peu de temps tant de positions différentes dans l’espace avait disparu, sans m’avoir aperçu, dans une rue de traverse, et je restais à me demander si je devais ou non entrer dans cet hôtel dont l’apparence modeste me fit fortement douter que ce fût Saint-Loup qui en fût sorti. Je me rappelai involontairement que Saint-Loup avait été injustement mêlé à une affaire d’espionnage parce qu’on avait trouvé son nom dans les lettres saisies sur un officier allemand. Pleine justice lui avait d’ailleurs été rendue par l’autorité militaire. Mais malgré moi je rapprochai ce fait de ce que je voyais. Cet hôtel servait-il de lieu de rendez-vous à des espions ? L’officier avait depuis un moment disparu quand je vis entrer de simples soldats de plusieurs armes, ce qui ajouta encore à la force de ma supposition. J’avais, d’autre part, extrêmement soif. « Il est probable que je pourrai trouver à boire ici », me dis-je, et j’en profitai pour tâcher d’assouvir, malgré l’inquiétude qui s’y mêlait, ma curiosité. Je ne pense donc pas que ce fut la curiosité de cette rencontre qui me décida à monter le petit escalier de quelques marches au bout duquel la porte d’une espèce de vestibule était ouverte, sans doute à cause de la chaleur. Je crus d’abord que, cette curiosité, je ne pourrais la satisfaire, car je vis plusieurs personnes venir demander une chambre, à qui on répondit qu’il n’y en avait plus une seule. Mais je compris ensuite qu’elles n’avaient évidemment contre elles que de ne pas faire partie du nid d’espionnage, car un simple marin s’étant présenté un moment après on se hâta de lui donner le no 28. Je pus apercevoir sans être vu, grâce à l’obscurité, quelques militaires et deux ouvriers qui causaient tranquillement dans une petite pièce étouffée, prétentieusement ornée de portraits en couleurs de femmes découpés dans des magazines et des revues illustrées. Ces gens causaient tranquillement, en train d’exposer des idées patriotiques : « Qu’est-ce que tu veux, on fera comme les camarades », disait l’un. « Ah ! pour sûr que je pense bien ne pas être tué », répondait à un vœu que je n’avais pas entendu, un autre qui, à ce que je compris, repartait le lendemain pour un poste dangereux. « Par exemple, à vingt-deux ans, en n’ayant encore fait que six mois, ce serait fort », criait-il avec un ton où perçait encore plus que le désir de vivre longtemps la conscience de raisonner juste, et comme si le fait de n’avoir que vingt-deux ans devait lui donner plus de chances de ne pas être tué, et que ce dût être une chose impossible qu’il le fût. « À Paris c’est épatant, disait un autre ; on ne dirait pas qu’il y a la guerre. Et toi, Julot, tu t’engages toujours ? — Pour sûr que je m’engage, j’ai envie d’aller y taper un peu dans le tas à tous ces sales Boches. — Mais Joffre, c’est un homme qui couche avec les femmes des Ministres, c’est pas un homme qui a fait quelque chose. — C’est malheureux d’entendre des choses pareilles, dit un aviateur un peu plus âgé en se tournant vers l’ouvrier qui venait de faire entendre cette proposition ; je vous conseillerais pas de causer comme ça en première ligne, les poilus vous auraient vite expédié. » La banalité de ces conversations ne me donnait pas grande envie d’en entendre davantage, et j’allais entrer ou redescendre quand je fus tiré de mon indifférence en entendant ces phrases qui me firent frémir : « C’est épatant, le patron qui ne revient pas, dame, à cette heure-ci je ne sais pas trop où il trouvera des chaînes. — Mais puisque l’autre est déjà attaché. — Il est attaché bien sûr, il est attaché et il ne l’est pas, moi je serais attaché comme ça que je pourrais me détacher. — Mais le cadenas est fermé. — C’est entendu qu’il est fermé, mais ça peut s’ouvrir à la rigueur. Ce qu’il y a, c’est que les chaînes ne sont pas assez longues. Tu vas pas m’expliquer à moi ce que c’est, j’y ai tapé dessus hier pendant toute la nuit que le sang m’en coulait sur les mains. — C’est toi qui taperas ce soir. — Non, c’est pas moi, c’est Maurice. Mais ça sera moi dimanche, le patron me l’a promis. » Je compris maintenant pourquoi on avait eu besoin des bras solides du marin. Si on avait éloigné de paisibles bourgeois, ce n’était donc pas qu’un nid d’espions que cet hôtel. Un crime atroce allait y être consommé, si on n’arrivait pas à temps pour le découvrir et faire arrêter les coupables. Tout cela pourtant, dans cette nuit paisible et menacée, gardait une apparence de rêve, de conte, et c’est à la fois avec une fierté de justicier et une volupté de poète que j’entrai délibérément dans l’hôtel. Je touchai légèrement mon chapeau et les personnes présentes, sans se déranger, répondirent plus ou moins poliment à mon salut. « Est-ce que vous pourriez me dire à qui il faut m’adresser ? Je voudrais avoir une chambre et qu’on m’y monte à boire. — Attendez une minute, le patron est sorti. — Mais il y a le chef là-haut, insinua un des causeurs. — Mais tu sais bien qu’on ne peut pas le déranger. — Croyez-vous qu’on me donnera une chambre ? — J’crois. — Le 43 doit être libre », dit le jeune homme qui était sûr de ne pas être tué parce qu’il avait vingt-deux ans. Et il se poussa légèrement sur le sofa pour me faire place. « Si on ouvrait un peu la fenêtre, il y a une fumée ici », dit l’aviateur ; et en effet chacun avait sa pipe ou sa cigarette. « Oui, mais alors, fermez d’abord les volets, vous savez bien qu’il est défendu d’avoir de la lumière à cause des Zeppelins. — Il n’en viendra plus de Zeppelins. Les journaux ont même fait allusion sur ce qu’ils avaient été tous descendus. — Il n’en viendra plus, il n’en viendra plus, qu’est-ce que tu en sais ? Quand tu auras comme moi quinze mois de front et que tu auras abattu ton cinquième avion boche, tu pourras en causer. Faut pas croire les journaux. Ils sont allés hier sur Compiègne, ils ont tué une mère de famille avec ses deux enfants. — Une mère de famille avec ses deux enfants », dit avec des yeux ardents et un air de profonde pitié le jeune homme qui espérait bien ne pas être tué et qui avait, du reste, une figure énergique, ouverte et des plus sympathiques. « On n’a pas de nouvelles du grand Julot. Sa marraine n’a pas reçu de lettre de lui depuis huit jours et c’est la première fois qu’il reste si longtemps sans lui en donner. — Qui est sa marraine ? — C’est la dame qui tient le chalet de nécessité un peu plus bas que l’Olympia. — Ils couchent ensemble ? — Qu’est-ce que tu dis là ; c’est une femme mariée, tout ce qu’il y a de sérieuse. Elle lui envoie de l’argent toutes les semaines parce qu’elle a bon cœur. Ah ! c’est une chic femme. — Alors tu le connais, le grand Julot ? — Si je le connais ! reprit avec chaleur le jeune homme de vingt-deux ans. C’est un de mes meilleurs amis intimes. Il n’y en a pas beaucoup que j’estime comme lui, et bon camarade, toujours prêt à rendre service, ah ! tu parles que ce serait un rude malheur s’il lui était arrivé quelque chose. » Quelqu’un proposa une partie de dés et à la hâte fébrile avec laquelle le jeune homme de vingt-deux ans retournait les dés et criait les résultats, les yeux hors de la tête, il était aisé de voir qu’il avait un tempérament de joueur. Je ne saisis pas bien ce que quelqu’un lui dit ensuite, mais il s’écria d’un ton de profonde pitié : « Julot, un maquereau ! C’est-à-dire qu’il dit qu’il est un maquereau. Mais il n’est pas foutu de l’être. Moi je l’ai vu payer sa femme, oui, la payer. C’est-à-dire que je ne dis pas que Jeanne l’Algérienne ne lui donnait pas quelque chose, mais elle ne lui donnait pas plus de cinq francs, une femme qui était en maison, qui gagnait plus de cinquante francs par jour. Se faire donner que cinq francs ! il faut qu’un homme soit trop bête. Et maintenant qu’elle est sur le front, elle a une vie dure, je veux bien, mais elle gagne ce qu’elle veut ; eh bien, elle ne lui envoie rien. Ah ! un maquereau, Julot ? Il y en a beaucoup qui pourraient se dire maquereaux à ce compte-là. Non seulement ce n’est pas un maquereau, mais à mon avis c’est même un imbécile. » Le plus vieux de la bande, et que le patron avait sans doute, à cause de son âge, chargé de lui faire garder une certaine tenue, n’entendit, étant allé un moment jusqu’aux cabinets, que la fin de la conversation. Mais il ne put s’empêcher de me regarder et parut visiblement contrarié de l’effet qu’elle avait dû produire sur moi. Sans s’adresser spécialement au jeune homme de vingt-deux ans qui venait pourtant d’exposer cette théorie de l’amour vénal, il dit, d’une façon générale : « Vous causez trop et trop fort, la fenêtre est ouverte, il y a des gens qui dorment à cette heure-ci. Vous savez que si le patron rentrait et vous entendait causer comme ça, il ne serait pas content. » Précisément en ce moment on entendit la porte s’ouvrir et tout le monde se tut croyant que c’était le patron, mais ce n’était qu’un chauffeur d’auto étranger auquel tout le monde fit grand accueil. Mais en voyant une chaîne de montre superbe qui s’étalait sur la veste du chauffeur, le jeune homme de vingt-deux ans lui lança un coup d’œil interrogatif et rieur, suivi d’un froncement de sourcil et d’un clignement d’œil sévère dirigé de mon côté. Et je compris que le premier regard voulait dire : « Qu’est-ce que ça ? tu l’as volée ? Toutes mes félicitations. » Et le second : « Ne dis rien à cause de ce type que nous ne connaissons pas. » Tout à coup le patron entra, chargé de plusieurs mètres de grosses chaînes capables d’attacher plusieurs forçats, suant, et dit : « J’en ai une charge, si vous tous vous n’étiez pas si fainéants, je ne devrais pas être obligé d’y aller moi-même. » Je lui dis que je demandais une chambre. « Pour quelques heures seulement, je n’ai pas trouvé de voiture et je suis un peu malade. Mais je voudrais qu’on me monte à boire. — Pierrot, va à la cave chercher du cassis et dis qu’on mette en état le numéro 43. Voilà le 7 qui sonne. Ils disent qu’ils sont malades. Malades, je t’en fiche, c’est des gens à prendre de la coco, ils ont l’air à moitié piqués, il faut les foutre dehors. A-t-on mis une paire de draps au 22 ? Bon ! voilà le 7 qui sonne encore, cours-y voir. Allons, Maurice, qu’est-ce que tu fais là, tu sais bien qu’on t’attend, monte au 14 bis. Et plus vite que ça. » Et Maurice sortit rapidement, suivant le patron qui, un peu ennuyé que j’eusse vu ses chaînes, disparut en les emportant. « Comment que tu viens si tard ? » demanda le jeune homme de vingt-deux ans au chauffeur. « Comment, si tard, je suis d’une heure en avance. Mais il fait trop chaud marcher. J’ai rendez-vous qu’à minuit. — Pour qui donc est-ce que tu viens ? — Pour Pamela la charmeuse », dit le chauffeur oriental dont le rire découvrit les belles dents blanches. « Ah ! » dit le jeune homme de vingt-deux ans. Bientôt on me fit monter dans la chambre 43, mais l’atmosphère était si désagréable et ma curiosité si grande que, mon « cassis » bu, je redescendis l’escalier, puis, pris d’une autre idée, je remontai et dépassai l’étage de la chambre 43, allai jusqu’en haut. Tout à coup, d’une chambre qui était isolée au bout d’un couloir me semblèrent venir des plaintes étouffées. Je marchai vivement dans cette direction et appliquai mon oreille à la porte. « Je vous en supplie, grâce, grâce, pitié, détachez-moi, ne me frappez pas si fort, disait une voix. Je vous baise les pieds, je m’humilie, je ne recommencerai pas. Ayez pitié. — Non, crapule, répondit une autre voix, et puisque tu gueules et que tu te traînes à genoux, on va t’attacher sur le lit, pas de pitié », et j’entendis le bruit du claquement d’un martinet, probablement aiguisé de clous car il fut suivi de cris de douleur. Alors je m’aperçus qu’il y avait dans cette chambre un œil-de-bœuf latéral dont on avait oublié de tirer le rideau ; cheminant à pas de loup dans l’ombre, je me glissai jusqu’à cet œil-de-bœuf, et là, enchaîné sur un lit comme Prométhée sur son rocher, recevant les coups d’un martinet en effet planté de clous que lui infligeait Maurice, je vis, déjà tout en sang, et couvert d’ecchymoses qui prouvaient que le supplice n’avait pas lieu pour la première fois, je vis devant moi M. de Charlus. Tout à coup la porte s’ouvrit et quelqu’un entra qui heureusement ne me vit pas, c’était Jupien. Il s’approcha du baron avec un air de respect et un sourire d’intelligence : « Hé bien, vous n’avez pas besoin de moi ? » Le baron pria Jupien de faire sortir un moment Maurice. Jupien le mit dehors avec la plus grande désinvolture. « On ne peut pas nous entendre ? » dit le baron à Jupien, qui lui affirma que non. Le baron savait que Jupien, intelligent comme un homme de lettres, n’avait nullement l’esprit pratique, parlait toujours, devant les intéressés, avec des sous-entendus qui ne trompaient personne et des surnoms que tout le monde connaissait. « Une seconde », interrompit Jupien qui avait entendu une sonnette retentir à la chambre no 3. C’était un député de l’Action Libérale qui sortait. Jupien n’avait pas besoin de voir le tableau car il connaissait son coup de sonnette, le député venant, en effet, tous les jours après déjeuner. Il avait été obligé ce jour-là de changer ses heures, car il avait marié sa fille à midi à Saint-Pierre de Chaillot. Il était donc venu le soir, mais tenait à partir de bonne heure à cause de sa femme, vite inquiète quand il rentrait tard, surtout par ces temps de bombardement. Jupien tenait à accompagner sa sortie pour témoigner de la déférence qu’il portait à la qualité d’honorable, sans aucun intérêt personnel d’ailleurs. Car bien que ce député, répudiant les exagérations de l’Action Française (il eût, d’ailleurs, été incapable de comprendre une ligne de Charles Maurras ou de Léon Daudet), fût bien avec les ministres, flattés d’être invités à ses chasses, Jupien n’aurait pas osé lui demander le moindre appui dans ses démêlés avec la police. Il savait que, s’il s’était risqué à parler de cela au législateur fortuné et froussard, il n’aurait pas évité la plus inoffensive des « descentes » mais eût instantanément perdu le plus généreux de ses clients. Après avoir reconduit jusqu’à la porte le député, qui avait rabattu son chapeau sur ses yeux, relevé son col et, glissant rapidement comme il faisait dans ses programmes électoraux, croyait cacher son visage, Jupien remonta près de M. de Charlus à qui il dit : « C’était Monsieur Eugène. » Chez Jupien, comme dans les maisons de santé, on n’appelait les gens que par leur prénom tout en ayant soin d’ajouter à l’oreille, pour satisfaire la curiosité des habitués ou augmenter le prestige de la maison, leur nom véritable. Quelquefois cependant Jupien ignorait la personnalité vraie de ses clients, s’imaginait et disait que c’était tel boursier, tel noble, tel artiste, erreurs passagères et charmantes pour ceux qu’on nommait à tort, et finissait par se résigner à ignorer toujours qui était Monsieur Victor. Jupien avait aussi l’habitude, pour plaire au baron, de faire l’inverse de ce qui est de mise dans certaines réunions. « Je vais vous présenter Monsieur Lebrun » (à l’oreille : « Il se fait appeler M. Lebrun mais en réalité c’est le grand-duc de Russie »). Inversement, Jupien sentait que ce n’était pas encore assez de présenter à M. de Charlus un garçon laitier. Il lui murmurait en clignant de l’œil : « Il est garçon laitier, mais, au fond, c’est surtout un des plus dangereux apaches de Belleville » (il fallait voir le ton grivois dont Jupien disait « apache »). Et comme si ces références ne suffisaient pas, il tâchait d’ajouter quelques « citations ». « Il a été condamné plusieurs fois pour vol et cambriolage de villas, il a été à Fresnes pour s’être battu (même air grivois) avec des passants qu’il a à moitié estropiés et il a été au bat’ d’Af. Il a tué son sergent. »

Le baron en voulait même légèrement à Jupien, car il savait que dans cette maison, qu’il avait chargé son factotum d’acheter pour lui et de faire gérer par un sous-ordre, tout le monde, par les maladresses de l’oncle de Mlle d’Oloron, feu Mme de Cambremer, connaissait plus ou moins sa personnalité et son nom (beaucoup seulement croyaient que c’était un surnom et, le prononçant mal, l’avaient déformé, de sorte que la sauvegarde du baron avait été leur propre bêtise et non la discrétion de Jupien). Mais il trouvait plus simple de se laisser rassurer par ses assurances, et tranquillisé de savoir qu’on ne pouvait les entendre, le baron lui dit : « Je ne voulais pas parler devant ce petit, qui est très gentil et fait de son mieux. Mais je ne le trouve pas assez brutal. Sa figure me plaît, mais il m’appelle « crapule » comme si c’était une leçon apprise. — Oh ! non, personne ne lui a rien dit, répondit Jupien sans s’apercevoir de l’invraisemblance de cette assertion. Il a, du reste, été compromis dans le meurtre d’une concierge de la Villette. — Ah ! cela c’est assez intéressant, dit le baron avec un sourire. — Mais j’ai justement là le tueur de bœufs, l’homme des abattoirs qui lui ressemble ; il a passé par hasard. Voulez-vous en essayer ? — Ah ! oui, volontiers. » Je vis entrer l’homme des abattoirs, il ressemblait, en effet, un peu à « Maurice », mais, chose plus curieuse, tous deux avaient quelque chose d’un type que personnellement je n’avais jamais dégagé, mais qu’à ce moment je me rendis très bien compte exister dans la figure de Morel, sinon dans la figure de Morel telle que je l’avais toujours vue, du moins dans un certain visage que des yeux aimants voyant Morel autrement que moi auraient pu composer avec ses traits. Dès que je me fus fait intérieurement, avec des traits empruntés à mes souvenirs de Morel, cette maquette de ce qu’il pouvait représenter à un autre, je me rendis compte que ces deux jeunes gens, dont l’un était un garçon bijoutier et l’autre un employé d’hôtel, étaient de vagues succédanés de Morel. Fallait-il en conclure que M. de Charlus, au moins en une certaine forme de ses amours, était toujours fidèle à un même type et que le désir qui lui avait fait choisir l’un après l’autre ces deux jeunes gens était le même que celui qui lui avait fait arrêter Morel sur le quai de la gare de Doncières ; que tous trois ressemblaient un peu à l’éphèbe dont la forme, intaillée dans le saphir qu’étaient les yeux de M. de Charlus, donnait à son regard ce quelque chose de si particulier qui m’avait effrayé le premier jour à Balbec ? Ou que son amour pour Morel ayant modifié le type qu’il cherchait, pour se consoler de son absence il cherchait des hommes qui lui ressemblassent ? Une supposition que je fis aussi fut que peut-être il n’avait jamais existé entre Morel et lui, malgré les apparences, que des relations d’amitié, et que M. de Charlus faisait venir chez Jupien des jeunes gens qui ressemblassent assez à Morel pour qu’il pût avoir auprès d’eux l’illusion de prendre du plaisir avec lui. Il est vrai qu’en songeant à tout ce que M. de Charlus a fait pour Morel, cette supposition eût semblé peu probable si l’on ne savait que l’amour nous pousse non seulement aux plus grands sacrifices pour l’être que nous aimons, mais parfois jusqu’au sacrifice de notre désir lui-même qui, d’ailleurs, est d’autant moins facilement exaucé que l’être que nous aimons sent que nous aimons davantage. Ce qui enlève aussi à une telle supposition l’invraisemblance qu’elle semble avoir au premier abord (bien qu’elle ne corresponde sans doute pas à la réalité) est dans le tempérament nerveux, dans le caractère profondément passionné de M. de Charlus, pareil en cela à celui de Saint-Loup, et qui avait pu jouer au début de ses relations avec Morel le même rôle, et plus décent, et négatif, qu’au début des relations de son neveu avec Rachel. Les relations avec une femme qu’on aime (et cela peut s’étendre à l’amour pour un jeune homme) peuvent rester platoniques pour une autre raison que la vertu de la femme ou que la nature peu sensuelle de l’amour qu’elle inspire. Cette raison peut être que l’amoureux, trop impatient par l’excès même de son amour, ne sait pas attendre avec une feinte suffisante d’indifférence le moment où il obtiendra ce qu’il désire. Tout le temps il revient à la charge, il ne cesse d’écrire à celle qu’il aime, il cherche tout le temps à la voir, elle le lui refuse, il est désespéré. Dès lors elle a compris que si elle lui accorde sa compagnie, son amitié, ces biens paraîtront déjà tellement considérables à celui qui a cru en être privé qu’elle peut se dispenser de donner davantage et profiter d’un moment où il ne peut plus supporter de ne pas la voir, où il veut à tout prix terminer la guerre, en lui imposant une paix qui aura pour première condition le platonisme des relations. D’ailleurs, pendant tout le temps qui a précédé ce traité, l’amoureux tout le temps anxieux, sans cesse à l’affût d’une lettre, d’un regard, a cessé de penser à la possession physique dont le désir l’avait tourmenté d’abord mais qui s’est usé dans l’attente et a fait place à des besoins d’un autre ordre, plus douloureux d’ailleurs s’ils ne sont pas satisfaits. Alors le plaisir qu’on avait le premier jour espéré des caresses, on le reçoit plus tard tout dénaturé sous la forme de paroles amicales, de promesses de présence qui, après les effets de l’incertitude, quelquefois simplement après un regard embrumé de tous les brouillards de la froideur et qui recule si loin la personne qu’on croit qu’on ne la reverra jamais, amènent de délicieuses détentes. Les femmes devinent tout cela et savent qu’elles peuvent s’offrir le luxe de ne se donner jamais à ceux dont elles sentent, s’ils ont été trop nerveux pour le leur cacher les premiers jours, l’inguérissable désir qu’ils ont d’elles. La femme est trop heureuse que, sans rien donner, elle reçoive beaucoup plus qu’elle n’a d’habitude quand elle se donne. Les grands nerveux croient ainsi à la vertu de leur idole. Et l’auréole qu’ils mettent autour d’elle est aussi un produit, mais, comme on voit, fort indirect, de leur excessif amour. Il existe alors chez la femme ce qui existe à l’état inconscient chez les médicaments à leur insu rusés, comme sont les soporifiques, la morphine. Ce n’est pas à ceux à qui ils donnent le plaisir du sommeil ou un véritable bien-être qu’ils sont absolument nécessaires. Ce n’est pas par ceux-là qu’ils seraient achetés à prix d’or, échangés contre tout ce que le malade possède, c’est par ces autres malades (d’ailleurs peut-être les mêmes, mais, à quelques années de distance, devenus autres) que le médicament ne fait pas dormir, à qui il ne cause aucune volupté, mais qui, tant qu’ils ne l’ont pas, sont en proie à une agitation qu’ils veulent faire cesser à tout prix, fût-ce en se donnant la mort. Pour M. de Charlus, dont le cas, en somme, avec cette légère différenciation due à la similitude du sexe, rentre dans les lois générales de l’amour, il avait beau appartenir à une famille plus ancienne que les Capétiens, être riche, être vainement recherché par une société élégante, et Morel n’être rien, il aurait eu beau dire à Morel, comme il m’avait dit à moi-même : « Je suis prince, je veux votre bien », encore était-ce Morel qui avait le dessus s’il ne voulait pas se rendre. Et pour qu’il ne le voulût pas, il suffisait peut-être qu’il se sentît aimé. L’horreur que les grands ont pour les snobs qui veulent à toute force se lier avec eux, l’homme viril l’a pour l’inverti, la femme pour tout homme trop amoureux. M. de Charlus non seulement avait tous les avantages, mais en eût proposé d’immenses à Morel. Mais il est possible que tout cela se fût brisé contre une volonté. Il en eût été dans ce cas de M. de Charlus comme de ces Allemands, auxquels il appartenait, du reste, par ses origines, et qui, dans la guerre qui se déroulait à ce moment, étaient bien, comme le baron le répétait un peu trop volontiers, vainqueurs sur tous les fronts. Mais à quoi leur servait leur victoire, puisque après chacune ils trouvaient les Alliés plus résolus à leur refuser la seule chose qu’eux, les Allemands, eussent souhaité d’obtenir, la paix et la réconciliation ? Ainsi Napoléon entrait en Russie et demandait magnanimement aux autorités de venir vers lui. Mais personne ne se présentait.

Je descendis et rentrai dans la petite antichambre où Maurice, incertain si on le rappellerait et à qui Jupien avait à tout hasard dit d’attendre, était en train de faire une partie de cartes avec un de ses camarades. On était très agité d’une croix de guerre qui avait été trouvée par terre, et on ne savait pas qui l’avait perdue, à qui la renvoyer pour éviter au titulaire un ennui. Puis on parla de la bonté d’un officier qui s’était fait tuer pour tâcher de sauver son ordonnance. « Il y a tout de même du bon monde chez les riches. Moi je me ferais tuer avec plaisir pour un type comme ça », dit Maurice, qui, évidemment, n’accomplissait ses terribles fustigations sur le baron que par une habitude mécanique, les effets d’une éducation négligée, le besoin d’argent et un certain penchant à le gagner d’une façon qui était censée donner moins de mal que le travail et en donnait peut-être davantage. Mais, ainsi que l’avait craint M. de Charlus, c’était peut-être un très bon cœur et c’était, paraît-il, un garçon d’une admirable bravoure. Il avait presque les larmes aux yeux en parlant de la mort de cet officier et le jeune homme de vingt-deux ans n’était pas moins ému. « Ah ! oui, ce sont de chic types. Des malheureux comme nous encore, ça n’a pas grand’chose à perdre, mais un Monsieur qui a des tas de larbins, qui peut aller prendre son apéro tous les jours à 6 heures, c’est vraiment chouette. On peut charrier tant qu’on veut, mais quand on voit des types comme ça mourir, ça fait vraiment quelque chose. Le bon Dieu ne devrait pas permettre que des riches comme ça meurent ; d’abord ils sont trop utiles à l’ouvrier. Rien qu’à cause d’une mort comme ça faudra tuer tous les Boches jusqu’au dernier ; et ce qu’ils ont fait à Louvain, et couper des poignets de petits enfants ; non, je ne sais pas, moi je ne suis pas meilleur qu’un autre, mais je me laisserais envoyer des pruneaux dans la gueule plutôt que d’obéir à des barbares comme ça ; car c’est pas des hommes, c’est des vrais barbares, tu ne diras pas le contraire. » Tous ces garçons étaient, en somme, patriotes. Un seul, légèrement blessé au bras, ne fut pas à la hauteur des autres car il dit, comme il devait bientôt repartir : « Dame, ça n’a pas été la bonne blessure » (celle qui fait réformer), comme Mme Swann disait jadis : « J’ai trouvé le moyen d’attraper la fâcheuse influenza. » La porte se rouvrit sur le chauffeur qui était allé un instant prendre l’air. « Comment, c’est déjà fini ? ça n’a pas été long », dit-il en apercevant Maurice qu’il croyait en train de frapper celui qu’on avait surnommé, par allusion à un journal qui paraissait à cette époque : « l’Homme enchaîné ». « Ce n’est pas long pour toi qui es allé prendre l’air, répondit Maurice, froissé qu’on vît qu’il avait déplu là-haut. Mais si tu étais obligé de taper à tour de bras comme moi, par cette chaleur ! Si c’était pas les cinquante francs qu’il donne… — Et puis, c’est un homme qui cause bien ; on sent qu’il a de l’instruction. Dit-il que ce sera bientôt fini ? — Il dit qu’on ne pourra pas les avoir, que ça finira sans que personne ait le dessus. — Bon sang de bon sang, mais c’est donc un Boche… — Je vous ai dit que vous causiez trop haut, dit le plus vieux aux autres en m’apercevant. Vous avez fini avec la chambre ? — Ah ! ta gueule, tu n’es pas le maître ici. — Oui, j’ai fini, et je venais pour payer. — Il vaut mieux que vous payiez au patron. Maurice, va donc le chercher. — Mais je ne veux pas vous déranger. — Ça ne me dérange pas. » Maurice monta et revint en me disant : « Le patron descend. » Je lui donnai deux francs pour son dérangement. Il rougit de plaisir. « Ah ! merci bien. Je les enverrai à mon frère qui est prisonnier. Non, il n’est pas malheureux, ça dépend beaucoup des camps. » Pendant ce temps, deux clients très élégants, en habit et cravate blanche sous leur pardessus — deux Russes, me sembla-t-il à leur très léger accent — se tenaient sur le seuil et délibéraient s’ils devaient entrer. C’était visiblement la première fois qu’ils venaient là, on avait dû leur indiquer l’endroit et ils semblaient partagés entre le désir, la tentation et une extrême frousse. L’un des deux — un beau jeune homme — répétait toutes les deux minutes à l’autre, avec un sourire mi-interrogateur, mi-destiné à persuader : « Quoi ! Après tout on s’en fiche. » Mais il avait beau vouloir dire par là qu’après tout on se fichait des conséquences, il est probable qu’il ne s’en fichait pas tant que cela, car cette parole n’était suivie d’aucun mouvement pour entrer, mais d’un nouveau regard vers l’autre, suivi du même sourire et du même « après tout, on s’en fiche ». C’était, ce « après tout on s’en fiche ! », un exemplaire entre mille de ce magnifique langage, si différent de celui que nous parlons d’habitude, et où l’émotion fait dévier ce que nous voulions dire et épanouir à la place une phrase tout autre, émergée d’un lac inconnu où vivent des expressions sans rapport avec la pensée, et qui par cela même la révèlent. Je me souviens qu’une fois Albertine, comme Françoise, que nous n’avions pas entendue, entrait au moment où mon amie était toute nue contre moi, dit malgré elle, voulant me prévenir : « Tiens, voilà la belle Françoise. » Françoise, qui n’y voyait pas très clair et ne faisait que traverser la pièce assez loin de nous, ne se fût sans doute aperçue de rien. Mais les mots si anormaux de « belle Françoise », qu’Albertine n’avait jamais prononcés de sa vie, montrèrent d’eux-mêmes leur origine ; elle les sentit cueillis au hasard par l’émotion, n’eut pas besoin de regarder rien pour comprendre tout et s’en alla en murmurant dans son patois le mot de « poutana ». Une autre fois, bien plus tard, quand Bloch devenu père de famille eut marié une de ses filles à un catholique, un monsieur mal élevé dit à celle-ci qu’il croyait avoir entendu dire qu’elle était fille d’un juif et lui en demanda le nom. La jeune femme, qui avait été Mlle Bloch depuis sa naissance, répondit en prononçant Bloch à l’allemande, comme eût fait le duc de Guermantes, c’est-à-dire en prononçant le ch non pas comme un c ou un k mais avec le rh germanique.
