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
  "Mme Verdurin": {
    "aliases": [
      "princesse de Guermantes",
      "Madame Verdurin",
      "Mme Verdurin"
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

[none]

### Passage

La nouvelle maison de santé dans laquelle je me retirai alors ne me guérit pas plus que la première ; et un long temps s’écoula avant que je la quittasse. Durant le trajet en chemin de fer que je fis pour rentrer à Paris, la pensée de mon absence de dons littéraires, que j’avais cru découvrir jadis du côté de Guermantes, que j’avais reconnue avec plus de tristesse encore dans mes promenades quotidiennes avec Gilberte, avant de rentrer dîner, fort avant dans la nuit, à Tansonville, et qu’à la veille de quitter cette propriété j’avais à peu près identifiée, en lisant quelques pages du journal des Goncourt, à la vanité, au mensonge de la littérature, cette pensée, moins douloureuse peut-être, plus morne encore, si je lui donnais comme objet non ma propre infirmité à moi particulière, mais l’inexistence de l’idéal auquel j’avais cru, cette pensée qui ne m’était pas depuis bien longtemps revenue à l’esprit me frappa de nouveau et avec une force plus lamentable que jamais. C’était, je me le rappelle, à un arrêt du train en pleine campagne. Le soleil éclairait jusqu’à la moitié de leur tronc une ligne d’arbres qui suivait la voie du chemin de fer. « Arbres, pensai-je, vous n’avez plus rien à me dire, mon cœur refroidi ne vous entend plus. Je suis pourtant ici en pleine nature, eh bien, c’est avec froideur, avec ennui que mes yeux constatent la ligne qui sépare votre front lumineux de votre tronc d’ombre. Si jamais j’ai pu me croire poète, je sais maintenant que je ne le suis pas. Peut-être dans la nouvelle partie de ma vie desséchée qui s’ouvre, les hommes pourraient-ils m’inspirer ce que ne me dit plus la nature. Mais les années où j’aurais peut-être été capable de la chanter ne reviendront jamais. » Mais en me donnant cette consolation d’une observation humaine possible venant prendre la place d’une inspiration impossible, je savais que je cherchais seulement à me donner une consolation, et que je savais moi-même sans valeur. Si j’avais vraiment une âme d’artiste, quel plaisir n’éprouverais-je pas devant ce rideau d’arbres éclairé par le soleil couchant, devant ces petites fleurs du talus qui se haussaient presque jusqu’au marchepied du wagon, dont je pouvais compter les pétales et dont je me garderais bien de décrire la couleur comme feraient tant de bons lettrés, car peut-on espérer transmettre au lecteur un plaisir qu’on n’a pas ressenti ? Un peu plus tard, j’avais vu avec la même indifférence les lentilles d’or et d’orange dont le même soleil couchant criblait les fenêtres d’une maison ; et enfin, comme l’heure avait avancé, j’avais vu une autre maison qui semblait construite en une substance d’un rose assez étrange. Mais j’avais fait ces diverses constatations avec la même absolue indifférence que si, me promenant dans un jardin avec une dame, j’avais vu une feuille de verre et un peu plus loin un objet d’une matière analogue à l’albâtre dont la couleur inaccoutumée ne m’aurait pas tiré du plus languissant ennui et que si, par politesse pour la dame, pour dire quelque chose et pour montrer que j’avais remarqué cette couleur, j’avais désigné en passant le verre coloré et le morceau de stuc. De la même manière, par acquit de conscience, je me signalais à moi-même, comme à quelqu’un qui m’eût accompagné et qui eût été capable d’en tirer plus de plaisir que moi, les reflets du feu dans les vitres et la transparence rose de la maison. Mais le compagnon à qui j’avais fait constater ces effets curieux était d’une nature sans doute moins enthousiaste que beaucoup de gens bien disposés, qu’une telle vue ravit, car il avait pris connaissance de ces couleurs sans aucune espèce d’allégresse.

Ma longue absence de Paris n’avait pas empêché d’anciens amis à continuer, comme mon nom restait sur leurs listes, à m’envoyer fidèlement des invitations, et quand j’en trouvai, en rentrant — avec une pour un goûter donné par la Berma en l’honneur de sa fille et de son gendre — une autre pour une matinée qui devait avoir lieu le lendemain chez le prince de Guermantes, les tristes réflexions que j’avais faites dans le train ne furent pas un des moindres motifs qui me conseillèrent de m’y rendre. Ce n’était vraiment pas la peine de me priver de mener la vie de l’homme du monde, m’étais-je dit, puisque le fameux « travail » auquel depuis si longtemps j’espère chaque jour me mettre le lendemain, je ne suis pas ou plus fait pour lui, et que peut-être même il ne correspond à aucune réalité. À vrai dire, cette raison était toute négative et ôtait simplement leur valeur à celles qui auraient pu me détourner de ce concert mondain. Mais celle qui m’y fit aller fut ce nom de Guermantes, depuis assez longtemps sorti de mon esprit pour que, lu sur la carte d’invitation, il réveillât un rayon de mon attention, allât prélever au fond de ma mémoire une coupe de leur passé, accompagné de toutes les images de forêt domaniale ou de hautes fleurs qui l’escortaient alors, et pour qu’il reprît pour moi le charme et la signification que je lui trouvais à Combray quand passant, avant de rentrer, dans la rue de l’Oiseau, je voyais du dehors, comme une laque obscure, le vitrail de Gilbert le Mauvais, sire de Guermantes. Pour un moment les Guermantes m’avaient semblé de nouveau entièrement différents des gens du monde, incomparables avec eux, avec tout être vivant, fût-il souverain ; ils me réapparaissaient comme des êtres issus de la fécondation de cet air aigre et vertueux de cette sombre ville de Combray où s’était passée mon enfance et du passé qu’on y apercevait dans la petite rue, à la hauteur du vitrail. J’avais eu envie d’aller chez les Guermantes comme si cela avait dû me rapprocher de mon enfance et des profondeurs de ma mémoire où je l’apercevais. Et j’avais continué à relire l’invitation jusqu’au moment où, révoltées, les lettres qui composaient ce nom si familier et si mystérieux, comme celui même de Combray, eussent repris leur indépendance et eussent dessiné devant mes yeux fatigués comme un nom que je ne connaissais pas.

Maman allant justement à un petit thé chez Mme Sazerat, je n’eus aucun scrupule à me rendre à la matinée de la princesse de Guermantes. Je pris une voiture pour y aller, car le prince de Guermantes n’habitait plus son ancien hôtel mais un magnifique qu’il s’était fait construire avenue du Bois. C’est un des torts des gens du monde de ne pas comprendre que s’ils veulent que nous croyions en eux il faudrait d’abord qu’ils y crussent eux-mêmes, ou au moins qu’ils respectassent les éléments essentiels de notre croyance. Au temps où je croyais, même si je savais le contraire, que les Guermantes habitaient tel palais en vertu d’un droit héréditaire, pénétrer dans le palais du sorcier ou de la fée, faire s’ouvrir devant moi les portes qui ne cèdent pas tant qu’on n’a pas prononcé la formule magique, me semblait aussi malaisé que d’obtenir un entretien du sorcier ou de la fée eux-mêmes. Rien ne m’était plus facile que de me faire croire à moi-même que le vieux domestique engagé de la veille ou fourni par Potel et Chabot était fils, petit-fils, descendant de ceux qui servaient la famille bien avant la Révolution, et j’avais une bonne volonté infinie à appeler portrait d’ancêtre le portrait qui avait été acheté le mois précédent chez Bernheim jeune. Mais un charme ne se transvase pas, les souvenirs ne peuvent se diviser, et du prince de Guermantes, maintenant qu’il avait percé lui-même à jour les illusions de ma croyance en étant allé habiter avenue du Bois, il ne restait plus grand’chose. Les plafonds que j’avais craint de voir s’écrouler quand on avait annoncé mon nom et sous lesquels eût flotté encore pour moi beaucoup du charme et des craintes de jadis couvraient les soirées d’une Américaine sans intérêt pour moi. Naturellement, les choses n’ont pas en elles-mêmes de pouvoir, et puisque c’est nous qui le leur confions, quelque jeune collégien bourgeois devait en ce moment avoir devant l’hôtel de l’avenue du Bois les mêmes sentiments que moi jadis devant l’ancien hôtel du prince de Guermantes. C’était qu’il était encore à l’âge des croyances, mais je l’avais dépassé, et j’avais perdu ce privilège, comme après la première jeunesse on perd le pouvoir qu’ont les enfants de dissocier en fractions digérables le lait qu’ils ingèrent, ce qui force les adultes à prendre, pour plus de prudence, le lait par petites quantités, tandis que les enfants peuvent le téter indéfiniment sans reprendre haleine. Du moins, le changement de résidence du prince de Guermantes eut cela de bon pour moi que la voiture qui était venue me chercher pour me conduire et dans laquelle je faisais ces réflexions dut traverser les rues qui vont vers les Champs-Élysées. Elles étaient fort mal pavées à cette époque, mais, dès le moment où j’y entrai, je n’en fus pas moins détaché de mes pensées par une sensation d’une extrême douceur ; on eût dit que tout d’un coup la voiture roulait plus facilement, plus doucement, sans bruit, comme quand les grilles d’un parc s’étant ouvertes on glisse sur les allées couvertes d’un sable fin ou de feuilles mortes ; matériellement il n’en était rien, mais je sentais tout à coup la suppression des obstacles extérieurs comme s’il n’y avait plus eu pour moi d’effort d’adaptation ou d’attention, tels que nous en faisons, même sans nous en rendre compte, devant les choses nouvelles ; les rues par lesquelles je passais en ce moment étaient celles, oubliées depuis si longtemps, que je prenais jadis avec Françoise pour aller aux Champs-Élysées. Le sol de lui-même savait où il devait aller ; sa résistance était vaincue. Et comme un aviateur qui a jusque-là péniblement roulé à terre, « décolle » brusquement, je m’élevais lentement vers les hauteurs silencieuses du souvenir. Dans Paris, ces rues-là se détacheront toujours pour moi en une autre matière que les autres. Quand j’arrivai au coin de la rue Royale, où était jadis le marchand en plein vent des photographies aimées de Françoise, il me sembla que la voiture, entraînée par des centaines de tours anciens, ne pourrait pas faire autrement que de tourner d’elle-même. Je ne traversais pas les mêmes rues que les promeneurs qui étaient dehors ce jour-là, mais un passé glissant, triste et doux. Il était, d’ailleurs, fait de tant de passés différents qu’il m’était difficile de reconnaître la cause de ma mélancolie, si elle était due à ces marches au-devant de Gilberte et dans la crainte qu’elle ne vînt pas, à la proximité d’une certaine maison où on m’avait dit qu’Albertine était allée avec Andrée, à la signification philosophique que semble prendre un chemin qu’on a suivi mille fois avec une passion qui ne dure plus et qui n’a pas porté de fruit, comme celui où, après le déjeuner, je faisais des courses si hâtives, si fiévreuses, pour regarder, toutes fraîches encore de colle, l’affiche de Phèdre et celle du Domino noir. Arrivé aux Champs-Élysées, comme je n’étais pas très désireux d’entendre tout le concert qui était donné chez les Guermantes, je fis arrêter la voiture et j’allais m’apprêter à descendre pour faire quelques pas à pied quand je fus frappé par le spectacle d’une voiture qui était en train de s’arrêter aussi. Un homme, les yeux fixes, la taille voûtée, était plutôt posé qu’assis dans le fond, et faisait pour se tenir droit les efforts qu’aurait faits un enfant à qui on aurait recommandé d’être sage. Mais son chapeau de paille laissait voir une forêt indomptée de cheveux entièrement blancs, et une barbe blanche, comme celle que la neige fait aux statues des fleuves dans les jardins publics, coulait de son menton. C’était, à côté de Jupien qui se multipliait pour lui, M. de Charlus convalescent d’une attaque d’apoplexie que j’avais ignorée (on m’avait seulement dit qu’il avait perdu la vue ; or il ne s’était agi que de troubles passagers, car il voyait de nouveau très clair) et qui, à moins que jusque-là il se fût teint et qu’on lui eût interdit de continuer à en prendre la fatigue, avait plutôt, comme en une sorte de précipité chimique, rendu visible et brillant tout le métal dont étaient saturées et que lançaient comme autant de geysers les mèches maintenant de pur argent de sa chevelure et de sa barbe, cependant qu’elle avait imposé au vieux prince déchu la majesté shakespearienne d’un roi Lear. Les yeux n’étaient pas restés en dehors de cette convulsion totale, de cette altération métallurgique de la tête. Mais, par un phénomène inverse, ils avaient perdu tout leur éclat. Mais le plus émouvant est qu’on sentait que cet éclat perdu était la fierté morale, et que par là la vie physique et même intellectuelle de M. de Charlus survivait à l’orgueil aristocratique, qu’on avait pu croire un moment faire corps avec elles. Ainsi à ce moment, se rendant sans doute aussi chez le prince de Guermantes, passa en Victoria Mme de Sainte-Euverte, que le baron jadis ne trouvait pas assez chic pour lui. Jupien, qui prenait soin de lui comme d’un enfant, lui souffla à l’oreille que c’était une personne de connaissance, Mme de Sainte-Euverte. Et aussitôt, avec une peine infinie et toute l’application d’un malade qui veut se montrer capable de tous les mouvements qui lui sont encore difficiles, M. de Charlus se découvrit, s’inclina, et salua Mme de Sainte-Euverte avec le même respect que si elle avait été la reine de France. Peut-être y avait-il dans la difficulté même que M. de Charlus avait à faire un tel salut une raison pour lui de le faire, sachant qu’il toucherait davantage par un acte qui, douloureux pour un malade, devenait doublement méritoire de la part de celui qui le faisait et flatteur pour celle à qui il s’adressait, les malades exagérant la politesse, comme les rois. Peut-être aussi y avait-il encore dans les mouvements du baron cette incoordination consécutive aux troubles de la moelle et du cerveau, et ses gestes dépassaient-ils l’intention qu’il avait. Pour moi, j’y vis plutôt une sorte de douceur quasi physique, de détachement des réalités de la vie, si frappants chez ceux que la mort a déjà fait entrer dans son ombre. La mise à nu des gisements argentés de la chevelure décelait un changement moins profond que cette inconsciente humilité mondaine qui intervertissait tous les rapports sociaux, humiliait devant Mme de Sainte-Euverte, eût humilié — en montrant ce qu’il a de fragile — devant la dernière des Américaines (qui eût pu enfin s’offrir la politesse jusque-là inaccessible pour elle du baron) le snobisme qui semblait le plus fier. Car le baron vivait toujours, pensait toujours ; son intelligence n’était pas atteinte. Et plus que n’eût fait tel chœur de Sophocle sur l’orgueil abaissé d’Œdipe, plus que la mort même, et toute oraison funèbre sur la mort, le salut empressé et humble du baron à Mme de Sainte-Euverte proclamait ce qu’a de périssable l’amour des grandeurs de la terre et tout l’orgueil humain. M. de Charlus, qui jusque-là n’eût pas consenti à dîner avec Mme de Sainte-Euverte, la saluait maintenant jusqu’à terre. Il saluait peut-être par ignorance du rang de la personne qu’il saluait (les articles du code social pouvant être emportés par une attaque comme toute autre partie de la mémoire), peut-être par une incoordination qui transposait dans le plan de l’humilité apparente l’incertitude — sans cela hautaine qu’il aurait eue — de l’identité de la dame qui passait. Il la salua enfin avec cette politesse des enfants venant timidement dire bonjour aux grandes personnes, sur l’appel de leur mère. Et un enfant, c’est, sans la fierté qu’ils ont, ce qu’il était devenu. Recevoir l’hommage de M. de Charlus, pour Mme de Sainte-Euverte c’était tout le snobisme, comme ç’avait été tout le snobisme du baron de le lui refuser. Or cette nature inaccessible et précieuse qu’il avait réussi à faire croire à Mme de Sainte-Euverte être essentielle à lui-même, M. de Charlus l’anéantit d’un seul coup par la timidité appliquée, le zèle peureux avec lequel il ôta son chapeau, d’où les torrents de sa chevelure d’argent ruisselèrent tout le temps qu’il laissa sa tête découverte par déférence, avec l’éloquence d’un Bossuet. Quand Jupien eut aidé le baron à descendre et que j’eus salué celui-ci, il me parla très vite, d’une voix si imperceptible que je ne pus distinguer ce qu’il me disait, ce qui lui arracha, quand pour la troisième fois je le fis répéter, un geste d’impatience qui m’étonna par l’impassibilité qu’avait d’abord montrée le visage et qui était due sans doute à un reste de paralysie. Mais quand je fus arrivé à comprendre ces paroles sussurrées, je m’aperçus que le malade gardait absolument intacte son intelligence. Il y avait, d’ailleurs, deux M. de Charlus, sans compter les autres. Des deux, l’intellectuel passait son temps à se plaindre qu’il allait à l’aphasie, qu’il prononçait constamment un mot, une lettre pour une autre. Mais dès qu’en effet il lui arrivait de le faire, l’autre M. de Charlus, le subconscient, lequel voulait autant faire envie que l’autre pitié, arrêtait immédiatement, comme un chef d’orchestre dont les musiciens pataugent, la phrase commencée, et avec une ingéniosité infinie attachait ce qui venait ensuite au mot dit en réalité pour un autre, mais qu’il semblait avoir choisi. Même sa mémoire était intacte ; il mettait, du reste, une coquetterie, qui n’allait pas sans la fatigue d’une application des plus ardues, à faire sortir tel souvenir ancien, peu important, se rapportant à moi et qui me montrerait qu’il avait gardé ou recouvré toute sa netteté d’esprit. Sans bouger la tête ni les yeux, ni varier d’une seule inflexion son débit, il me dit, par exemple : « Voici un poteau où il y a une affiche pareille à celle devant laquelle j’étais la première fois que je vous vis à Avranches, non, je me trompe, à Balbec. » Et c’était, en effet, une réclame pour le même produit. J’avais à peine, au début, distingué ce qu’il disait, de même qu’on commence par ne voir goutte dans une chambre dont tous les rideaux sont clos. Mais, comme des yeux dans la pénombre, mes oreilles s’habituèrent bientôt à ce pianissimo. Je crois aussi qu’il s’était graduellement renforcé pendant que le baron parlait, soit que la faiblesse de sa voix provînt en partie d’une appréhension nerveuse qui se dissipait quand, distrait par un tiers, il ne pensait plus à elle ; soit qu’au contraire cette faiblesse correspondît à son état véritable et que la force momentanée avec laquelle il parlait dans la conversation fût provoquée par une excitation factice, passagère et plutôt funeste, qui faisait dire aux étrangers : « Il est déjà mieux, il ne faut pas qu’il pense à son mal », mais augmentait au contraire celui-ci qui ne tardait pas à reprendre. Quoi qu’il en soit, le baron à ce moment (et même en tenant compte de mon adaptation) jetait ses paroles plus fort, comme la marée, les jours de mauvais temps, ses petites vagues tordues. Et ce qui lui restait de sa récente attaque faisait entendre au fond de ses paroles comme un bruit de cailloux roulés. D’ailleurs, continuant à me parler du passé, sans doute pour bien me montrer qu’il n’avait pas perdu la mémoire, il l’évoquait d’une façon funèbre, mais sans tristesse. Il ne cessait d’énumérer tous les gens de sa famille ou de son monde qui n’étaient plus, moins, semblait-il, avec la tristesse qu’ils ne fussent plus en vie qu’avec la satisfaction de leur survivre. Il semblait en rappelant leur trépas prendre mieux conscience de son retour vers la santé. C’est avec une dureté presque triomphale qu’il répétait sur un ton uniforme, légèrement bégayant et aux sourdes résonances sépulcrales : « Hannibal de Bréauté, mort ! Antoine de Mouchy, mort ! Charles Swann, mort ! Adalbert de Montmorency, mort ! Baron de Talleyrand, mort ! Sosthène de Doudeauville, mort ! » Et chaque fois, ce mot « mort » semblait tomber sur ces défunts comme une pelletée de terre plus lourde, lancée par un fossoyeur qui tenait à les river plus profondément à la tombe.

La duchesse de Létourville, qui n’allait pas à la matinée de la princesse de Guermantes, parce qu’elle venait d’être longtemps malade, passa à ce moment à pied à côté de nous, et apercevant le baron, dont elle ignorait la récente attaque, s’arrêta pour lui dire bonjour. Mais la maladie qu’elle venait d’avoir faisait qu’elle ne comprenait pas mieux, mais supportait plus impatiemment, avec une mauvaise humeur nerveuse où il y avait peut-être beaucoup de pitié, la maladie des autres. Entendant le baron prononcer difficilement et à faux certains mots, lui voyant bouger difficilement le bras, elle jeta les yeux tour à tour sur Jupien et sur moi comme pour nous demander l’explication d’un phénomène aussi choquant. Comme nous ne lui dîmes rien, ce fut à M. de Charlus lui-même qu’elle adressa un long regard plein de tristesse mais aussi de reproches. Elle avait l’air de lui faire grief d’être avec elle, dehors, dans une attitude aussi peu usuelle que s’il fût sorti sans cravate ou sans souliers. À une nouvelle faute de prononciation que commit le baron, la douleur et l’indignation de la duchesse augmentant ensemble, elle dit au baron : « Palamède ! » sur le ton interrogatif et exaspéré des gens trop nerveux qui ne peuvent supporter d’attendre une minute et, si on les fait entrer tout de suite en s’excusant d’achever sa toilette, vous disent amèrement, non pour s’excuser mais pour s’accuser : « Mais alors, je vous dérange ! », comme si c’était un crime de la part de celui qu’on dérange. Finalement, elle nous quitta d’un air de plus en plus navré en disant au baron : « Vous feriez mieux de rentrer. »

M. de Charlus demanda à s’asseoir sur un fauteuil pour se reposer pendant que Jupien et moi ferions quelques pas et tira péniblement de sa poche un livre qui me sembla être un livre de prières. Je n’étais pas fâché de pouvoir apprendre par Jupien bien des détails sur l’état de santé du baron. « Je suis content de causer avec vous, Monsieur, me dit Jupien, mais nous n’irons pas plus loin que le rond-point. Dieu merci, le baron va bien maintenant, mais je n’ose pas le laisser longtemps seul, il est toujours le même, il a trop bon cœur, il donnerait tout ce qu’il a aux autres, et puis ce n’est pas tout, il est resté coureur comme un jeune homme et je suis obligé d’ouvrir les yeux. — D’autant plus qu’il a retrouvé les siens, répondis-je ; on m’avait beaucoup attristé en me disant qu’il avait perdu la vue. — Sa paralysie s’était, en effet, portée là, il ne voyait absolument plus. Pensez que, pendant la cure qui lui a fait, du reste, tant de bien, il est resté plusieurs mois sans voir plus qu’un aveugle de naissance. — Cela devait au moins rendre inutile toute une partie de votre surveillance ? — Pas le moins du monde, à peine arrivé dans un hôtel, il me demandait comment était telle personne de service. Je l’assurais qu’il n’y avait que des horreurs. Mais il sentait bien que cela ne pouvait pas être universel, que je devais quelquefois mentir. Voyez-vous, ce petit polisson ! Et puis il avait une espèce de flair, d’après la voix peut-être, je ne sais pas. Alors il s’arrangeait pour m’envoyer faire d’urgence des courses. Un jour — vous m’excuserez de vous dire cela, mais vous êtes venu une fois par hasard dans le Temple de l’Impudeur, je n’ai rien à vous cacher (d’ailleurs, il avait toujours une satisfaction assez peu sympathique à faire étalage des secrets qu’il détenait) — je rentrais d’une de ces courses soi-disant pressées, d’autant plus vite que je me figurais bien qu’elle avait été arrangée à dessein, quand, au moment où j’approchais de la chambre du baron, j’entendis une voix qui disait : « Quoi ? — Comment, répondit le baron, c’était donc la première fois ? » J’entrai sans frapper, et quelle ne fut pas ma frayeur. Le baron, trompé par la voix qui était, en effet, plus forte qu’elle n’est d’habitude à cet âge-là (et à cette époque-là le baron était complètement aveugle), était, lui qui aimait plutôt autrefois les personnes mûres, avec un enfant qui n’avait pas dix ans. »
