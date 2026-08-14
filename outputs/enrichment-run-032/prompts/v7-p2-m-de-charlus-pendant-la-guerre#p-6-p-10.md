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

Un des premiers soirs dès mon nouveau retour à Paris en 1916, ayant envie d’entendre parler de la seule chose qui m’intéressait alors, la guerre, je sortis, après le dîner, pour aller voir Mme Verdurin, car elle était, avec Mme Bontemps, une des reines de ce Paris de la guerre qui faisait penser au Directoire. Comme par l’ensemencement d’une petite quantité de levure, en apparence de génération spontanée, des jeunes femmes allaient tout le jour coiffées de hauts turbans cylindriques comme aurait pu l’être une contemporaine de Mme Tallien. Par civisme, ayant des tuniques égyptiennes droites, sombres, très « guerre », sur des jupes très courtes, elles chaussaient des lanières rappelant le cothurne selon Talma, ou de hautes guêtres rappelant celles de nos chers combattants ; c’est, disaient-elles, parce qu’elles n’oubliaient pas qu’elles devaient réjouir les yeux de ces combattants qu’elles se paraient encore, non seulement de toilettes « floues », mais encore de bijoux évoquant les armées par leur thème décoratif, si même leur matière ne venait pas des armées, n’avait pas été travaillée aux armées ; au lieu d’ornements égyptiens rappelant la campagne d’Égypte, c’étaient des bagues ou des bracelets faits avec des fragments d’obus ou des ceintures de 75, des allume-cigarettes composés de deux sous anglais, auxquels un militaire était arrivé à donner, dans sa cagna, une patine si belle que le profil de la reine Victoria y avait l’air tracé par Pisanello ; c’est encore parce qu’elles y pensaient sans cesse, disaient-elles, qu’elles portaient à peine le deuil quand l’un des leurs tombait, sous le prétexte qu’il était « mêlé de fierté », ce qui permettait un bonnet de crêpe anglais blanc (du plus gracieux effet et autorisant tous les espoirs), dans l’invincible certitude du triomphe définitif, et permettait ainsi de remplacer le cachemire d’autrefois par le satin et la mousseline de soie, et même de garder ses perles, « tout en observant le tact et la correction qu’il est inutile de rappeler à des Françaises ».

### Passage

Le Louvre, tous les musées étaient fermés, et quand on lisait en tête d’un article de journal : « Une exposition sensationnelle », on pouvait être sûr qu’il s’agissait d’une exposition non de tableaux, mais de robes, de robes destinées, d’ailleurs, à éveiller « ces délicates joies d’art dont les Parisiennes étaient depuis trop longtemps sevrées ». C’est ainsi que l’élégance et le plaisir avaient repris ; l’élégance, à défaut des arts, cherchait à s’excuser comme ceux-ci en 1793, année où les artistes exposant au Salon révolutionnaire proclamaient que ce serait à tort qu’il paraîtrait « étrange à d’austères républicains que nous nous occupions des arts quand l’Europe coalisée assiège le territoire de la liberté ». Ainsi faisaient en 1916 les couturiers qui, d’ailleurs, avec une orgueilleuse conscience d’artistes, avouaient que « chercher du nouveau, s’écarter de la banalité, préparer la victoire, dégager pour les générations d’après la guerre une formule nouvelle du beau, telle était l’ambition qui les tourmentait, la chimère qu’ils poursuivaient, ainsi qu’on pouvait s’en rendre compte en venant visiter leurs salons délicieusement installés rue de la …, où effacer par une note lumineuse et gaie les lourdes tristesses de l’heure semble être le mot d’ordre, avec la discrétion toutefois qu’imposent les circonstances. Les tristesses de l’heure, il est vrai, pourraient avoir raison des énergies féminines si nous n’avions tant de hauts exemples de courage et d’endurance à méditer. Aussi en pensant à nos combattants qui au fond de leur tranchée rêvent de plus de confort et de coquetterie pour la chère absente laissée au foyer, ne cesserons-nous pas d’apporter toujours plus de recherche dans la création de robes répondant aux nécessités du moment. La vogue, cela se conçoit, est surtout aux maisons anglaises, donc alliées, et on raffole cette année de la robe-tonneau dont le joli abandon nous donne à toutes un amusant petit cachet de rare distinction. Ce sera même une des plus heureuses conséquences de cette triste guerre, ajoutait le charmant chroniqueur (en attendant la reprise des provinces perdues, le réveil du sentiment national), ce sera même une des plus heureuses conséquences de cette guerre que d’avoir obtenu de jolis résultats en fait de toilette, sans luxe inconsidéré et de mauvais aloi, avec très peu de chose, d’avoir créé de la coquetterie avec des riens. À la robe du grand couturier éditée à plusieurs exemplaires on préfère en ce moment les robes faites chez soi, parce qu’affirmant l’esprit, le goût et les tendances indiscutables de chacun. » Quant à la charité, en pensant à toutes les misères nées de l’invasion, à tant de mutilés, il était bien naturel qu’elle fût obligée de se faire « plus ingénieuse encore », ce qui obligeait les dames à hauts turbans à passer la fin de l’après-midi dans les thés autour d’une table de bridge, en commentant les nouvelles du « front », tandis qu’à la porte les attendaient leurs automobiles ayant sur le siège un beau militaire qui bavardait avec le chasseur. Ce n’était pas, du reste, seulement les coiffures surmontant les visages de leur étrange cylindre qui étaient nouvelles. Les visages l’étaient aussi. Les dames à nouveaux chapeaux étaient des jeunes femmes venues on ne savait trop d’où et qui étaient la fleur de l’élégance, les unes depuis six mois, les autres depuis deux ans, les autres depuis quatre. Ces différences avaient, d’ailleurs, pour elles autant d’importance qu’au temps où j’avais débuté dans le monde en avaient entre deux familles comme les Guermantes et les La Rochefoucauld trois ou quatre siècles d’ancienneté prouvée. La dame qui connaissait les Guermantes depuis 1914 regardait comme une parvenue celle qu’on présentait chez eux en 1916, lui faisait un bonjour de douairière, la dévisageait de son face-à-main et avouait dans une moue qu’on ne savait même pas au juste si cette dame était ou non mariée. « Tout cela est assez nauséabond », concluait la dame de 1914, qui eût voulu que le cycle des nouvelles admissions s’arrêtât après elle. Ces personnes nouvelles, que les jeunes gens trouvaient fort anciennes, et que d’ailleurs certains vieillards qui n’avaient pas été que dans le grand monde croyaient bien reconnaître pour ne pas être si nouvelles que cela, n’offraient pas seulement à la société les divertissements de conversation politique et de musique dans l’intimité qui lui convenaient ; il fallait encore que ce fussent elles qui les offrissent, car pour que les choses paraissent nouvelles, même si elles sont anciennes, et même si elles sont nouvelles, il faut en art, comme en médecine, comme en mondanité, des noms nouveaux (ils étaient d’ailleurs nouveaux en certaines choses). Ainsi Mme Verdurin était allée à Venise pendant la guerre, mais comme ces gens qui veulent éviter de parler chagrin et sentiment, quand elle disait que c’était épatant, ce qu’elle admirait ce n’était ni Venise, ni Saint-Marc, ni les palais, tout ce qui m’avait tant plu et dont elle faisait bon marché, mais l’effet des projecteurs dans le ciel, des projecteurs sur lesquels elle donnait des renseignements appuyés de chiffres. (Ainsi d’âge en âge renaît un certain réalisme en réaction contre l’art admiré jusque-là.) Le salon Sainte-Euverte était une étiquette défraîchie, sous laquelle la présence des plus grands artistes, des ministres les plus influents, n’eût attiré personne. On courait, au contraire, pour écouter un mot prononcé par le secrétaire des uns ou le sous-chef de cabinet des autres, chez les nouvelles dames à turban, dont l’invasion ailée et jacassante emplissait Paris. Les dames du Premier Directoire avaient une reine qui était jeune et belle et s’appelait Madame Tallien. Celles du second en avaient deux qui étaient vieilles et laides et qui s’appelaient Mme Verdurin et Mme Bontemps. Qui eût pu tenir rigueur à Mme Bontemps que son mari eût joué un rôle, âprement critiqué par l’Écho de Paris, dans l’affaire Dreyfus ? Toute la Chambre étant à un certain moment devenue révisionniste, c’était forcément parmi d’anciens révisionnistes, comme parmi d’anciens socialistes, qu’on avait été obligé de recruter le parti de l’Ordre social, de la Tolérance religieuse, de la Préparation militaire. On aurait détesté autrefois M. Bontemps parce que les antipatriotes avaient alors le nom de dreyfusards. Mais bientôt ce nom avait été oublié et remplacé par celui d’adversaire de la loi de trois ans. M. Bontemps était, au contraire, un des auteurs de cette loi, c’était donc un patriote. Dans le monde (et ce phénomène social n’est, d’ailleurs, qu’une application d’une loi psychologique bien plus générale), les nouveautés coupables ou non n’excitent l’horreur que tant qu’elles ne sont pas assimilées et entourées d’éléments rassurants. Il en était du dreyfusisme comme du mariage de Saint-Loup avec la fille d’Odette, mariage qui avait d’abord fait crier. Maintenant qu’on voyait chez les Saint-Loup tous les gens « qu’on connaissait », Gilberte aurait pu avoir les mœurs d’Odette elle-même que, malgré cela, on y serait « allé » et qu’on eût approuvé Gilberte de blâmer comme une douairière des nouveautés morales non assimilées. Le dreyfusisme était maintenant intégré dans une série de choses respectables et habituelles. Quant à se demander ce qu’il valait en soi, personne n’y songeait, pas plus pour l’admettre maintenant qu’autrefois pour le condamner. Il n’était plus « shocking ». C’était tout ce qu’il fallait. À peine se rappelait-on qu’il l’avait été, comme on ne sait plus au bout de quelque temps si le père d’une jeune fille fut un voleur ou non. Au besoin, on peut dire : « Non, c’est du beau-frère, ou d’un homonyme que vous parlez, mais contre celui-là il n’y a jamais eu rien à dire. » De même il y avait certainement eu dreyfusisme et dreyfusisme, et celui qui allait chez la duchesse de Montmorency et faisait passer la loi de trois ans ne pouvait être mauvais. En tout cas, à tout péché miséricorde. Cet oubli qui était octroyé au dreyfusisme l’était a fortiori aux dreyfusards. Il n’y avait plus qu’eux, du reste, dans la politique, puisque tous à un moment l’avaient été s’il voulaient être du Gouvernement, même ceux qui représentaient le contraire de ce que le dreyfusisme, dans sa choquante nouveauté, avait incarné (au temps où Saint-Loup était sur une mauvaise pente) : l’antipatriotisme, l’irréligion, l’anarchie, etc. Ainsi le dreyfusisme de M. Bontemps, invisible et contemplatif comme celui de tous les hommes politiques, ne se voyait pas plus que les os sous la peau. Personne ne se fût rappelé qu’il avait été dreyfusard, car les gens du monde sont distraits et oublieux, parce qu’aussi il y avait de cela un temps fort long, et qu’ils affectaient de croire plus long, car c’était une des idées les plus à la mode de dire que l’avant-guerre était séparé de la guerre par quelque chose d’aussi profond, simulant autant de durée qu’une période géologique, et Brichot lui-même, ce nationaliste, quand il faisait allusion à l’affaire Dreyfus disait : « Dans ces temps préhistoriques ». À vrai dire, ce changement profond opéré par la guerre était en raison inverse de la valeur des esprits touchés, du moins à partir d’un certain degré, car, tout en bas, les purs sots, les purs gens de plaisir ne s’occupaient pas qu’il y eût la guerre. Mais tout en haut, ceux qui se sont fait une vie intérieure ambiante ont peu d’égard à l’importance des événements. Ce qui modifie profondément pour eux l’ordre des pensées, c’est bien plutôt quelque chose qui semble en soi n’avoir aucune importance et qui renverse pour eux l’ordre du temps en les faisant contemporains d’un autre temps de leur vie. Un chant d’oiseau dans le parc de Montboissier, ou une brise chargée de l’odeur de réséda, sont évidemment des événements de moindre conséquence que les plus grandes dates de la Révolution et de l’Empire. Ils ont cependant inspiré à Chateaubriand, dans les Mémoires d’Outre-tombe, des pages d’une valeur infiniment plus grande.

M. Bontemps ne voulait pas entendre parler de paix avant que l’Allemagne eût été réduite au même morcellement qu’au moyen âge, la déchéance de la maison de Hohenzollern prononcée, Guillaume ayant reçu douze balles dans la peau. En un mot, il était ce que Brichot appelait un « Jusquauboutiste », c’était le meilleur brevet de civisme qu’on pouvait lui donner. Sans doute, les trois premiers jours, Mme Bontemps avait été un peu dépaysée au milieu des personnes qui avaient demandé à Mme Verdurin à la connaître, et ce fut d’un ton légèrement aigre que Mme Verdurin répondit : « Le comte, ma chère », à Mme Bontemps qui lui disait : « C’est bien le duc d’Haussonville que vous venez de me présenter », soit par entière ignorance et absence de toute association entre le nom Haussonville et un titre quelconque, soit, au contraire, par excessive instruction et association d’idées avec le « Parti des Ducs », dont on lui avait dit que M. d’Haussonville était un des membres à l’Académie. À partir du quatrième jour elle avait commencé d’être solidement installée dans le faubourg Saint-Germain. Quelquefois encore on voyait autour d’elle les fragments inconnus d’un monde qu’on ne connaissait pas et qui n’étonnaient pas plus que des débris de coquille autour du poussin, ceux qui savaient l’œuf d’où Mme Bontemps était sortie. Mais dès le quinzième jour, elle les avait secoués, et avant la fin du premier mois, quand elle disait : « Je vais chez les Lévi », tout le monde comprenait, sans qu’elle eût besoin de préciser, qu’il s’agissait des Lévis-Mirepoix, et pas une duchesse ne se serait couchée sans avoir appris de Mme Bontemps ou de Mme Verdurin, au moins par téléphone, ce qu’il y avait dans le communiqué du soir, ce qu’on y avait omis, où on en était avec la Grèce, quelle offensive on préparait, en un mot tout ce que le public ne saurait que le lendemain ou plus tard, et dont on avait ainsi comme une sorte de répétition des couturières. Dans la conversation, Mme Verdurin, pour communiquer les nouvelles, disait : « nous » en parlant de la France. « Hé bien, voici : nous exigeons du roi de Grèce qu’il se retire du Péloponèse, etc. ; nous lui envoyons, etc. » Et dans tous ses récits revenait tout le temps le G.Q.G. (j’ai téléphoné au G.Q.G.), abréviation qu’elle avait à prononcer le même plaisir qu’avaient naguère les femmes qui ne connaissaient pas le prince d’Agrigente à demander en souriant, quand on parlait de lui et pour montrer qu’elles étaient au courant : « Grigri ? », un plaisir qui dans les époques peu troublées n’est connu que par les mondains, mais que dans ces grandes crises le peuple même connaît. Notre maître d’hôtel, par exemple, si on parlait du roi de Grèce, était capable, grâce aux journaux, de dire comme Guillaume II : « Tino », tandis que jusque-là sa familiarité avec les rois était restée plus vulgaire, ayant été inventée par lui, comme quand jadis, pour parler du Roi d’Espagne, il disait : « Fonfonse ». On peut remarquer, d’ailleurs, qu’au fur et à mesure qu’augmenta le nombre des gens brillants qui firent des avances à Mme Verdurin, le nombre de ceux qu’elle appelait les « ennuyeux » diminua. Par une sorte de transformation magique, tout ennuyeux qui était venu lui faire une visite et avait sollicité une invitation devenait subitement quelqu’un d’agréable, d’intelligent. Bref, au bout d’un an le nombre des ennuyeux était réduit dans une proportion tellement forte, que la « peur et l’impossibilité de s’ennuyer », qui avait tenu une si grande place dans la conversation et joué un si grand rôle dans la vie de Mme Verdurin, avait presque entièrement disparu. On eût dit que sur le tard cette impossibilité de s’ennuyer (qu’autrefois, d’ailleurs, elle assurait ne pas avoir éprouvée dans sa prime jeunesse) la faisait moins souffrir, comme certaines migraines, certains asthmes nerveux qui perdent de leur force quand on vieillit. Et l’effroi de s’ennuyer eût sans doute entièrement abandonné Mme Verdurin, faute d’ennuyeux, si elle n’avait, dans une faible mesure, remplacé ceux qui ne l’étaient plus par d’autres recrutés parmi les anciens fidèles. Du reste, pour en finir avec les duchesses qui fréquentaient maintenant chez Mme Verdurin, elles venaient y chercher, sans qu’elles s’en doutassent, exactement la même chose que les dreyfusards autrefois, c’est-à-dire un plaisir mondain composé de telle manière que sa dégustation assouvît les curiosités politiques et rassasiât le besoin de commenter entre soi les incidents lus dans les journaux. Mme Verdurin disait : « Vous viendrez à 5 heures parler de la guerre », comme autrefois « parler de l’affaire », et dans l’intervalle : « Vous viendrez entendre Morel ». Or Morel n’aurait pas dû être là, pour la raison qu’il n’était nullement réformé. Simplement il n’avait pas rejoint et était déserteur, mais personne ne le savait. Une autre étoile du salon était « dans les choux », qui malgré ses goûts sportifs s’était fait réformer. Il était devenu tellement pour moi l’auteur d’une œuvre admirable à laquelle je pensais constamment que ce n’est que par hasard, quand j’établissais un courant transversal entre deux séries de souvenirs, que je songeais qu’il était celui qui avait amené le départ d’Albertine de chez moi. Et encore ce courant transversal aboutissait, en ce qui concernait ces reliques de souvenirs d’Albertine, à une voie s’arrêtant en pleine friche à plusieurs années de distance. Car je ne pensais plus jamais à elle. C’était une voie non fréquentée de souvenirs, une ligne que je n’empruntais plus. Tandis que les œuvres de « dans les choux » étaient récentes et cette ligne de souvenirs perpétuellement fréquentée et utilisée par mon esprit.

Je dois, du reste, dire que la connaissance du mari d’Andrée n’était ni très facile ni très agréable à faire, et que l’amitié qu’on lui vouait était promise à bien des déceptions. Il était, en effet, à ce moment déjà fort malade et s’épargnait les fatigues autres que celles qui lui paraissaient devoir peut-être lui donner du plaisir. Or il ne classait parmi celles-là que les rendez-vous avec des gens qu’il ne connaissait pas encore et que son ardente imagination lui représentait sans doute comme ayant une chance d’être différents des autres. Mais pour ceux qu’il connaissait déjà, il savait trop bien comment ils étaient, comment ils seraient, ils ne lui paraissaient plus valoir la peine d’une fatigue dangereuse pour lui et peut-être mortelle. C’était, en somme, un très mauvais ami. Et peut-être dans son goût pour des gens nouveaux se retrouvait-il quelque chose de l’audace frénétique qu’il portait jadis, à Balbec, aux sports, au jeu, à tous les excès de table. Quant à Mme Verdurin, elle voulait à chaque fois me faire faire la connaissance d’Andrée, ne pouvant admettre que je l’eusse connue depuis longtemps. D’ailleurs Andrée venait rarement avec son mari, mais elle était pour moi une amie admirable et sincère. Fidèle à l’esthétique de son mari, qui était en réaction contre les Ballets russes, elle disait du marquis de Polignac : « Il a sa maison décorée par Bakst ; comment peut-on dormir là dedans, j’aimerais mieux Dubufe. »

D’ailleurs les Verdurin, par le progrès fatal de l’esthétisme, qui finit par se manger la queue, disaient ne pas pouvoir supporter le modern style (de plus c’était munichois) ni les appartements blancs et n’aimaient plus que les vieux meubles français dans un décor sombre.

On fut très étonné à cette époque, où Mme Verdurin pouvait avoir chez elle qui elle voulait, de lui voir faire indirectement des avances à une personne qu’elle avait complètement perdue de vue, Odette. On trouvait qu’elle ne pourrait rien ajouter au brillant milieu qu’était devenu le petit groupe. Mais une séparation prolongée, en même temps qu’elle apaise les rancunes, réveille quelquefois l’amitié. Et puis le phénomène qui amène non seulement les mourants à ne prononcer que des noms autrefois familiers, mais les vieillards à se complaire dans leurs souvenirs d’enfance, ce phénomène a son équivalent social. Pour réussir dans l’entreprise de faire revenir Odette chez elle, Mme Verdurin n’employa pas, bien entendu, les « ultras », mais les habitués moins fidèles qui avaient gardé un pied dans l’un et l’autre salon. Elle leur disait : « Je ne sais pas pourquoi on ne la voit plus ici. Elle est peut-être brouillée, moi pas. En somme, qu’est-ce que je lui ai fait ? C’est chez moi qu’elle a connu ses deux maris. Si elle veut revenir, qu’elle sache que les portes lui sont ouvertes. » Ces paroles, qui auraient dû coûter à la fierté de la Patronne si elles ne lui avaient pas été dictées par son imagination, furent redites, mais sans succès. Mme Verdurin attendit Odette sans la voir venir, jusqu’à ce que des événements qu’on verra plus loin amenassent pour de tout autres raisons ce que n’avait pu l’ambassade pourtant zélée des lâcheurs. Tant il est peu de réussites faciles, et d’échecs définitifs.
