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

Détendus ou brisés, les ressorts de la machine refoulante ne fonctionnaient plus, mille corps étrangers y pénétraient, lui ôtaient toute homogénéité, toute tenue, toute couleur. Le faubourg Saint-Germain, comme une douairière gâteuse, ne répondait que par des sourires timides à des domestiques insolents qui envahissaient ses salons, buvaient son orangeade et lui présentaient leurs maîtresses. Encore la sensation du temps écoulé et de l’anéantissement d’une partie de mon passé disparu m’était-elle donnée moins vivement encore par la destruction de cet ensemble cohérent (qu’avait été le salon Guermantes) d’éléments dont mille nuances, mille raisons expliquaient la présence, la fréquence, la coordination, qu’expliquée par l’anéantissement même de la connaissance des mille raisons, des mille nuances qui faisaient que tel qui s’y trouvait encore maintenant y était tout naturellement indiqué et à sa place, tandis que tel autre qui l’y coudoyait y présentait une nouveauté suspecte. Cette ignorance n’était pas que du monde, mais de la politique, de tout. Car la mémoire dure moins que la vie chez les individus, et, d’ailleurs, de très jeunes, qui n’avaient jamais eu les souvenirs abolis chez les autres, faisant maintenant partie du monde, et très légitimement, même au sens nobiliaire, les débuts étant oubliés ou ignorés, on prenait les gens — au point d’élévation ou de chute — où ils se trouvaient, croyant qu’il en avait toujours été ainsi, et que la princesse de Guermantes et Bloch avaient toujours eu la plus grande situation, que Clemenceau et Viviani avaient toujours été conservateurs. Et comme certains faits ont plus de durée, le souvenir exécré de l’Affaire Dreyfus persistant vaguement chez eux, grâce à ce que leur avaient dit leurs pères, si on leur disait que Clemenceau avait été dreyfusard, ils disaient : « Pas possible, vous confondez, il est juste de l’autre côté. » Des ministres tarés et d’anciennes filles publiques étaient tenus pour des parangons de vertu. Quelqu’un ayant demandé à un jeune homme de la plus grande famille s’il n’y avait pas eu quelque chose à dire sur la mère de Gilberte, le jeune seigneur répondit qu’en effet, dans la première partie de son existence, elle avait épousé un aventurier du nom de Swann, mais qu’ensuite elle avait épousé un des hommes les plus en vue de la société, le comte de Forcheville. Sans doute quelques personnes encore dans ce salon, la duchesse de Guermantes par exemple, eussent souri de cette assertion (qui, niant l’élégance de Swann, me paraissait monstrueuse, alors que moi-même jadis, à Combray, j’avais cru avec ma grand’tante que Swann ne pouvait connaître des « princesses ») et aussi des femmes qui eussent pu se trouver là mais qui ne sortaient plus guère, les duchesses de Montmorency, de Mouchy, de Sagan, qui avaient été les amies intimes de Swann et n’avaient jamais aperçu ce Forcheville, non reçu dans le monde au temps où elles y allaient encore. Mais précisément c’est que la société d’alors, de même que les visages aujourd’hui modifiés et les cheveux blonds remplacés par des cheveux blancs, n’existait plus que dans la mémoire d’êtres dont le nombre diminuait tous les jours. Bloch, pendant la guerre, avait cessé de « sortir », de fréquenter ses anciens milieux d’autrefois où il faisait piètre figure. En revanche, il n’avait cessé de publier de ces ouvrages dont je m’efforçais aujourd’hui, pour ne pas être entravé par elle, de détruire l’absurde sophistique, ouvrages sans originalité, mais qui donnaient aux jeunes gens et à beaucoup de femmes du monde l’impression d’une hauteur intellectuelle peu commune, d’une sorte de génie. Ce fut donc après une scission complète entre son ancienne mondanité et la nouvelle que, dans une société reconstituée, il avait fait, pour une phase nouvelle de sa vie, honorée, glorieuse, une apparition de grand homme. Les jeunes gens ignoraient naturellement qu’il fît à cet âge-là des débuts dans la société, d’autant que le peu de noms qu’il avait retenus dans la fréquentation de Saint-Loup lui permettaient de donner à son prestige actuel une sorte de recul indéfini. En tout cas il paraissait un de ces hommes de talent qui à toute époque ont fleuri dans le grand monde et on ne pensait pas qu’il eût jamais vécu ailleurs.

### Passage

Dès que j’eus fini de parler au prince de Guermantes, Bloch se saisit de moi et me présenta à une jeune femme qui avait beaucoup entendu parler de moi par la duchesse de Guermantes. Si les gens des nouvelles générations tenaient la duchesse de Guermantes pour peu de chose parce qu’elle connaissait des actrices, etc., les dames — aujourd’hui vieilles — de la famille la considéraient toujours comme un personnage extraordinaire, d’une part parce qu’elles savaient exactement sa naissance, sa primauté héraldique, ses intimités avec ce que Mme de Forcheville eût appelé des « royalties », mais encore parce qu’elle dédaignait de venir dans la famille, s’y ennuyait et qu’on savait qu’on n’y pouvait jamais compter sur elle. Ses relations théâtrales et politiques, d’ailleurs mal sues, ne faisaient qu’augmenter sa rareté, donc son prestige. De sorte que, tandis que dans le monde politique et artistique on la tenait pour une créature mal définie, une sorte de défroquée du faubourg Saint-Germain qui fréquente les sous-secrétaires d’État et les étoiles, dans ce même faubourg Saint-Germain, si on donnait une belle soirée, on disait : « Est-ce même la peine d’inviter Marie Sosthènes ? elle ne viendra pas. Enfin pour la forme, mais il ne faut pas se faire d’illusions. » Et si, vers 10 h. ½, dans une toilette éclatante, paraissant, de ses yeux durs pour elles, mépriser toutes ses cousines, entrait Marie Sosthènes qui s’arrêtait sur le seuil avec une sorte de majestueux dédain, et si elle restait une heure, c’était une plus grande fête pour la vieille grande dame qui donnait la soirée qu’autrefois pour un directeur de théâtre que Sarah Bernhardt, qui avait vaguement promis un concours sur lequel on ne comptait pas, fût venue et eût, avec une complaisance et une simplicité infinies, récité, au lieu du morceau promis, vingt autres. La présence de Marie Sosthènes, à laquelle les chefs de cabinet parlaient de haut en bas et qui n’en continuait pas moins (l’esprit mène ainsi le monde) à chercher à en connaître de plus en plus, venait de classer la soirée de la douairière, où il n’y avait pourtant que des femmes excessivement chic, en dehors et au-dessus de toutes les autres soirées de douairières de la même « season » (comme aurait encore dit Mme de Forcheville), mais pour lesquelles soirées ne s’était pas dérangée Marie Sosthènes qui était une des femmes les plus élégantes du jour. Le nom de la jeune femme à laquelle Bloch m’avait présenté m’était entièrement inconnu, et celui des différents Guermantes ne devait pas lui être très familier, car elle demanda à une Américaine à quel titre Mme de Saint-Loup avait l’air si intime avec toute la plus brillante société qui se trouvait là. Or, cette Américaine était mariée au comte de Furcy, parent obscur des Forcheville et pour lequel ils représentaient ce qu’il y a de plus brillant au monde. Aussi répondit-elle tout naturellement : « Quand ce ne serait que parce qu’elle est née Forcheville. C’est ce qu’il y a de plus grand. » Encore Mme de Furcy, tout en croyant naïvement le nom de Forcheville supérieur à celui de Saint-Loup, savait-elle du moins ce qu’était ce dernier. Mais la charmante amie de Bloch et de la duchesse de Guermantes l’ignorait absolument et, étant assez étourdie, répondit de bonne foi à une jeune fille qui lui demandait comment Mme de Saint-Loup était parente du maître de la maison, le prince de Guermantes : « Par les Forcheville », renseignement que la jeune fille communiqua, comme si elle l’avait possédé de tout temps, à une de ses amies, laquelle, ayant mauvais caractère et étant nerveuse, devint rouge comme un coq la première fois qu’un monsieur lui dit que ce n’était pas par les Forcheville que Gilberte tenait aux Guermantes, de sorte que le monsieur crut qu’il s’était trompé, adopta l’erreur et ne tarda pas à la propager. Les dîners, les fêtes mondaines, étaient pour l’Américaine une sorte d’École Berlitz. Elle entendait les noms et les répétait sans avoir connu préalablement leur valeur, leur portée exacte. On expliqua à quelqu’un qui demandait si Tansonville venait à Gilberte de son père M. de Forcheville, que cela ne venait pas du tout par là, que c’était une terre de la famille de son mari, que Tansonville était voisin de Guermantes, appartenait à Mme de Marsantes, mais étant très hypothéqué, avait été racheté, en dot, par Gilberte. Enfin un vieux de la vieille, ayant évoqué Swann ami des Sagan et des Mouchy, et l’Américaine amie de Bloch ayant demandé comment je l’avais connu, déclara que je l’avais connu chez Mme de Guermantes, ne se doutant pas du voisin de campagne, jeune ami de mon grand-père, qu’il représentait pour moi. Des méprises de ce genre ont été commises par les hommes les plus fameux et passent pour particulièrement graves dans toute société conservatrice. Saint-Simon, voulant montrer que Louis XIV était d’une ignorance qui « le fit tomber quelquefois, en public, dans les absurdités les plus grossières », ne donne de cette ignorance que deux exemples, à savoir que le Roi, ne sachant pas que Rénel était de la famille de Clermont-Gallerande ni Saint-Hérem de celle de Montmorin, les traita en hommes de peu. Du moins, en ce qui concerne Saint-Hérem, avons-nous la consolation de savoir que le Roi ne mourut pas dans l’erreur, car il fut détrompé « fort tard » par M. de la Rochefoucauld. « Encore, ajoute Saint-Simon avec un peu de pitié, lui fallut-il expliquer quelles étaient ces maisons que leur nom ne lui apprenait pas. » Cet oubli si vivace qui recouvre si rapidement le passé le plus récent, cette ignorance si envahissante, créent par contre-coup une valeur d’érudition à un petit savoir d’autant plus précieux qu’il est peu répandu, s’appliquant à la généalogie des gens, à leurs vraies situations, à la raison d’amour, d’argent ou autre pour quoi ils se sont alliés à telle famille, ou mésalliés, savoir prisé dans toutes les sociétés où règne un esprit conservateur, savoir que mon grand-père possédait au plus haut degré, concernant la bourgeoisie de Combray et de Paris, savoir que Saint-Simon prisait tant que, au moment où il célèbre la merveilleuse intelligence du prince de Conti, avant même de parler des sciences, ou plutôt comme si c’était la première des sciences, il le loue d’avoir été « un très bel esprit, lumineux, juste, exact, étendu, d’une lecture infinie, qui n’oubliait rien, qui connaissait les généalogies, leurs chimères et leurs réalités, d’une politesse distinguée selon le rang, le mérite, rendant tout ce que les princes du sang doivent et qu’ils ne rendent plus. Il s’en expliquait même et, sur leurs usurpations, l’histoire des livres et des conversations lui fournissait de quoi placer ce qu’il trouvait de plus obligeant sur la naissance, les emplois, etc. » Moins brillant, pour tout ce qui avait trait à la bourgeoisie de Combray et de Paris, mon grand-père ne le savait pas avec moins d’exactitude et ne le savourait pas avec moins de gourmandise. Ces gourmets-là, ces amateurs-là étaient déjà devenus peu nombreux qui savaient que Gilberte n’était pas Forcheville, ni Mme de Cambremer Méséglise, ni la plus jeune une Valintonais. Peu nombreux, peut-être même pas recrutés dans la plus haute aristocratie (ce ne sont pas forcément les dévots, ni même les catholiques, qui sont le plus savants concernant la Légende Dorée ou les vitraux du xiiie siècle), mais souvent dans une aristocratie secondaire, plus friande de ce qu’elle n’approche guère et qu’elle a d’autant plus le loisir d’étudier qu’elle le fréquente moins, se retrouvant avec plaisir, faisant la connaissance les uns des autres, donnant de succulents dîners de corps, comme la société des bibliophiles ou des amis de Reims, dîners où on déguste des généalogies. Les femmes n’y sont pas admises, mais les maris rentrent en disant à la leur : « J’ai fait un dîner intéressant. Il y avait un M. de la Raspelière qui nous a tenus sous le charme en nous expliquant que cette Mme de Saint-Loup qui a cette jolie fille n’est pas du tout née Forcheville. C’est tout un roman. »

L’amie de Bloch et de la duchesse de Guermantes n’était pas seulement élégante et charmante, elle était intelligente aussi, et la conversation avec elle était agréable, mais m’était rendue difficile parce que ce n’était pas seulement le nom de mon interlocutrice qui était nouveau pour moi, mais celui d’un grand nombre de personnes dont elle me parla et qui formaient actuellement le fond de la société. Il est vrai que, d’autre part, comme elle voulait m’entendre raconter des histoires, beaucoup de ceux que je lui citai ne lui dirent absolument rien, ils étaient tous tombés dans l’oubli, du moins ceux qui n’avaient brillé que de l’éclat individuel d’une personne et n’étaient pas le nom générique et permanent de quelque célèbre famille aristocratique (dont la jeune femme savait rarement le titre exact, supposant des naissances inexactes sur un nom qu’elle avait entendu de travers la veille dans un dîner), et elle ne les avait pour la plupart jamais entendu prononcer, n’ayant commencé à aller dans le monde (non seulement parce qu’elle était encore jeune, mais parce qu’elle habitait depuis peu la France et n’avait pas été reçue tout de suite) que quelques années après que je m’en étais moi-même retiré. De sorte que, si nous avions en commun un même vocabulaire de mots, pour les noms, celui de chacun de nous était différent. Je ne sais comment le nom de Mme Leroi tomba de mes lèvres et, par hasard, mon interlocutrice, grâce à quelque vieil ami, galant auprès d’elle, de Mme de Guermantes, en avait entendu parler. Mais inexactement comme je le vis au ton dédaigneux dont cette jeune femme snob me répondit : « Si, je sais qui est Mme Leroi, une vieille amie de Bergotte » d’un ton qui voulait dire « une personne que je n’aurais jamais voulu faire venir chez moi ». Je compris très bien que le vieil ami de Mme de Guermantes, en parfait homme du monde imbu de l’esprit des Guermantes, dont un des traits était de ne pas avoir l’air d’attacher d’importance aux fréquentations aristocratiques, avait trouvé trop bête et trop anti-Guermantes de dire : « Mme Leroi, qui fréquentait toutes les altesses, toutes les duchesses » et il avait préféré dire : « Elle était assez drôle. Elle a répondu un jour à Bergotte ceci. » Seulement, pour les gens qui ne savent pas, ces renseignements par la conversation équivalent à ceux que donne la Presse aux gens du peuple et qui croient alternativement, selon leur journal, que M. Loubet et M. Reinach sont des voleurs ou de grands citoyens. Pour mon interlocutrice, Mme Leroi avait été une espèce de Mme Verdurin première manière, avec moins d’éclat et dont le petit clan eût été limité au seul Bergotte… Cette jeune femme est, d’ailleurs, une des dernières qui, par un pur hasard, ait entendu le nom de Mme Leroi. Aujourd’hui personne ne sait plus qui c’est, ce qui est, du reste, parfaitement juste. Son nom ne figure même pas dans l’index des mémoires posthumes de Mme de Villeparisis, de laquelle Mme Leroi occupa tant l’esprit. La marquise n’a, d’ailleurs, pas parlé de Mme Leroi, moins parce que celle-ci, de son vivant, avait été peu aimable pour elle, que parce que personne ne pouvait s’intéresser à elle après sa mort, et ce silence est dicté moins par la rancune mondaine de la femme que par le tact littéraire de l’écrivain. Ma conversation avec l’élégante amie de Bloch fut charmante, car cette jeune femme était intelligente, mais cette différence entre nos deux vocabulaires la rendait malaisée et en même temps instructive. Nous avons beau savoir que les années passent, que la jeunesse fait place à la vieillesse, que les fortunes et les trônes les plus solides s’écroulent, que la célébrité est passagère, notre manière de prendre connaissance et, pour ainsi dire, de prendre le cliché de cet univers mouvant, entraîné par le Temps, l’immobilise au contraire. De sorte que nous voyons toujours jeunes les gens que nous avons connus jeunes, que ceux que nous avons connus vieux nous les parons rétrospectivement dans le passé des vertus de la vieillesse, que nous nous fions sans réserve au crédit d’un milliardaire et à l’appui d’un souverain, sachant par le raisonnement, mais ne croyant pas effectivement, qu’ils pourront être demain des fugitifs dénués de pouvoir. Dans un champ plus restreint et de mondanité pure, comme dans un problème plus simple qui initie à des difficultés plus complexes mais de même ordre, l’inintelligibilité qui résultait, dans notre conversation avec la jeune femme, du fait que nous avions vécu dans un certain monde à vingt-cinq ans de distance, me donnait l’impression et aurait pu fortifier chez moi le sens de l’histoire. Du reste, il faut bien dire que cette ignorance des situations réelles, qui tous les dix ans fait surgir les élus dans leur apparence actuelle et comme si le passé n’existait pas, qui empêche, pour une Américaine fraîchement débarquée, de voir que M. de Charlus avait eu la plus grande situation de Paris à une époque où Bloch n’en avait aucune, et que Swann qui faisait tant de frais pour M. Bontemps avait été traité avec la plus grande amitié par le prince de Galles, cette ignorance n’existe pas seulement chez les nouveaux venus, mais chez ceux qui ont fréquenté toujours des sociétés voisines, et cette ignorance, chez ces derniers comme chez les autres, est aussi un effet (mais cette fois s’exerçant sur l’individu et non sur la courbe sociale) du Temps. Sans doute, nous avons beau changer de milieu, de genre de vie, notre mémoire, en retenant le fil de notre personnalité identique, attache à elle, aux époques successives, le souvenir des sociétés où nous avons vécu, fût-ce quarante ans plus tôt. Bloch, chez le prince de Guermantes, savait parfaitement l’humble milieu juif où il avait vécu à dix-huit ans, et Swann, quand il n’aima plus Mme Swann mais une femme qui servait le thé chez ce même Colombin où Mme Swann avait cru quelque temps qu’il était chic d’aller, comme au thé de la rue Royale, Swann savait très bien sa valeur mondaine, se rappelant Twickenham, n’avait aucun doute sur les raisons pour lesquelles il allait plutôt chez Colombin que chez la duchesse de Broglie, et savait parfaitement qu’eût-il été lui-même mille fois moins « chic », cela ne l’eût pas empêché davantage d’aller chez Colombin ou à l’hôtel Ritz, puisque tout le monde peut y aller en payant. Sans doute les amis de Bloch ou de Swann se rappelaient eux aussi la petite société juive ou les invitations à Twickenham, et ainsi les amis, comme des « moi » un peu moins distincts de Swann et de Bloch, ne séparaient pas, dans leur mémoire, du Bloch élégant d’aujourd’hui le Bloch sordide d’autrefois, du Swann de chez Colombin des derniers jours le Swann de Buckingham Palace. Mais ces amis étaient, en quelque sorte, dans la vie, les voisins de Swann ; la leur s’était développée sur une ligne assez voisine pour que leur mémoire pût être assez pleine de lui ; mais chez d’autres plus éloignés de Swann, à une distance plus grande de lui, non pas précisément socialement, mais d’intimité, qui avait fait la connaissance plus vague et les rencontres très rares, les souvenirs moins nombreux avaient rendu les notions plus flottantes. Or, chez des étrangers de ce genre, au bout de trente ans on ne se rappelle plus rien de précis qui puisse prolonger dans le passé et changer de valeur l’être qu’on a sous les yeux. J’avais entendu, dans les dernières années de la vie de Swann, des gens du monde pourtant, à qui on parlait de lui, dire et comme si ç’avait été son titre de notoriété : « Vous parlez du Swann de chez Colombin ? » J’entendais maintenant des gens qui auraient pourtant dû savoir, dire en parlant de Bloch : « Le Bloch-Guermantes ? Le familier des Guermantes ? » Ces erreurs qui scindent une vie et en isolant le présent font de l’homme dont on parle un autre homme, un homme différent, une création de la veille, un homme qui n’est que la condensation de ses habitudes actuelles (alors que lui porte en lui-même la continuité de sa vie qui le relie au passé), ces erreurs dépendent bien aussi du Temps, mais elles sont non un phénomène social, mais un phénomène de mémoire. J’eus dans l’instant même un exemple, d’une variété assez différente, il est vrai, mais d’autant plus frappante, de ces oublis qui modifient pour nous l’aspect des êtres. Un jeune neveu de Mme de Guermantes, le marquis de Villemandois, avait été jadis pour moi d’une insolence obstinée qui m’avait conduit par représailles à adopter à son égard une attitude si insultante que nous étions devenus tacitement comme deux ennemis. Pendant que j’étais en train de réfléchir sur le temps, à cette matinée chez la princesse de Guermantes, il se fit présenter à moi en disant qu’il croyait que j’avais connu de ses parents, qu’il avait lu des articles de moi et désirait faire ou refaire ma connaissance. Il est vrai de dire qu’avec l’âge il était devenu, comme beaucoup, d’impertinent sérieux, qu’il n’avait plus la même arrogance et que, d’autre part, on parlait de moi, pour de bien minces articles cependant, dans le milieu qu’il fréquentait. Mais ces raisons de sa cordialité et de ses avances ne furent qu’accessoires. La principale, ou du moins celle qui permit aux autres d’entrer en jeu, c’est que, ou ayant une plus mauvaise mémoire que moi, ou ayant attaché une attention moins soutenue à mes ripostes que je n’avais fait autrefois à ses attaques, parce que j’étais alors pour lui un bien plus petit personnage qu’il n’était pour moi, il avait entièrement oublié notre inimitié. Mon nom lui rappelait tout au plus qu’il avait dû me voir, ou quelqu’un des miens, chez une de ses tantes… Et ne sachant pas au juste s’il se faisait présenter ou représenter, il se hâta de me parler de sa tante, chez qui il ne doutait pas qu’il avait dû me rencontrer, se rappelant qu’on y parlait souvent de moi, mais non de nos querelles. Un nom, c’est tout ce qui reste bien souvent pour nous d’un être, non pas même quand il est mort, mais de son vivant. Et nos notions actuelles sur lui sont si vagues ou si bizarres, et correspondent si peu à celles que nous avons eues de lui, que nous avons entièrement oublié que nous avons failli nous battre en duel avec lui, mais que nous nous rappelons qu’il portait, enfant, d’étranges guêtres jaunes aux Champs-Élysées, dans lesquels par contre, malgré que nous le lui assurions, il n’a aucun souvenir d’avoir joué avec nous. Bloch était entré en sautant comme une hyène. Je pensais : « Il vient dans des salons où il n’eût pas pénétré il y a vingt ans. » Mais il avait aussi vingt ans de plus. Il était plus près de la mort. À quoi cela l’avançait-il ? De près, dans la translucidité d’un visage où, de plus loin et mal éclairé, je ne voyais que la jeunesse gaie (soit qu’elle y survécût, soit que je l’y évoquasse), se tenait le visage presque effrayant, tout anxieux, d’un vieux Shylock attendant, tout grimé dans la coulisse, le moment d’entrer en scène, récitant déjà les premiers vers à mi-voix. Dans dix ans, dans ces salons où leur veulerie l’aurait imposé, il entrerait en béquillant, devenu maître, trouvant une corvée d’être obligé d’aller chez les La Trémoïlle. À quoi cela l’avançait-il ?

Des changements produits dans la société je pouvais d’autant plus extraire des vérités importantes et dignes de cimenter une partie de mon œuvre qu’ils n’étaient nullement, comme j’aurais pu être au premier moment tenté de le croire, particuliers à notre époque. Au temps où moi-même, à peine parvenu, j’étais entré, plus nouveau que ne l’était Bloch lui-même aujourd’hui, dans le milieu des Guermantes, j’avais dû y contempler, comme faisant partie intégrante de ce milieu, des éléments absolument différents, agrégés depuis peu et qui paraissaient étrangement nouveaux à de plus anciens dont je ne les différenciais pas et qui eux-mêmes, crus, par les ducs d’alors, membres de tout temps du faubourg, y avaient, eux, ou leurs pères, ou leurs grands-pères, été jadis des parvenus. Si bien que ce n’était pas la qualité d’hommes du grand monde qui rendait cette société si brillante, mais le fait d’avoir été assimilés plus ou moins complètement par cette société qui faisait, de gens qui cinquante ans plus tard paraissaient tous pareils, des gens du grand monde. Même dans le passé où je reculais le nom de Guermantes pour lui donner toute sa grandeur, et avec raison du reste, car sous Louis XIV les Guermantes, quasi royaux, faisaient plus grande figure qu’aujourd’hui, le phénomène que je remarquais en ce moment se produisait de même. Ne les avait-on pas vus alors s’allier à la famille Colbert par exemple, laquelle aujourd’hui, il est vrai, nous paraît très noble puisque épouser une Colbert semble un grand parti pour un La Rochefoucauld. Mais ce n’est pas parce que les Colbert, simples bourgeois alors, étaient nobles, que les Guermantes s’allièrent avec eux, c’est parce que les Guermantes s’allièrent avec eux qu’ils devinrent nobles. Si le nom d’Haussonville s’éteint avec le représentant actuel de cette maison, il tirera peut-être son illustration de descendre de Mme de Staël, alors qu’avant la Révolution, M. d’Haussonville, un des premiers seigneurs du royaume, tirait vanité auprès de M. de Broglie de ne pas connaître le père de Mme de Staël et de ne pas pouvoir plus le présenter que M. de Broglie ne pouvait le présenter lui-même, ne se doutant guère que leurs fils épouseraient un jour l’un la fille, l’autre la petite-fille de l’auteur de Corinne. Je me rendais compte, d’après ce que me disait la duchesse de Guermantes, que j’aurais pu faire dans ce monde la figure d’homme élégant non titré, mais qu’on croit volontiers affilié de tout temps à l’aristocratie, que Swann y avait faite autrefois, et avant lui M. Lebrun, M. Ampère, tous ces amis de la duchesse de Broglie, qui elle-même était au début fort peu du grand monde. Les premières fois que j’avais dîné chez Mme de Guermantes, combien n’avais-je pas dû choquer des hommes comme M. de Beauserfeuil, moins par ma présence que par des remarques témoignant que j’étais entièrement ignorant des souvenirs qui constituaient son passé et donnaient sa forme à l’usage qu’il avait de la société. Bloch un jour, quand, devenu très vieux, il aurait une mémoire assez ancienne du salon Guermantes tel qu’il se présentait à ce moment à ses yeux, éprouverait le même étonnement, la même mauvaise humeur en présence de certaines intrusions et de certaines ignorances. Et, d’autre part, il aurait sans doute contracté et dispenserait autour de lui ces qualités de tact et de discrétion que j’avais crues le privilège d’hommes comme M. de Norpois, et qui se reforment et s’incarnent dans ceux qui nous paraissent entre tous les exclure. D’ailleurs, le cas qui s’était présenté pour moi d’être admis dans la société des Guermantes m’avait paru quelque chose d’exceptionnel. Mais si je sortais de moi et du milieu qui m’entourait immédiatement, je voyais que ce phénomène social n’était pas aussi isolé qu’il m’avait paru d’abord et que du bassin de Combray où j’étais né, assez nombreux, en somme, étaient les jets d’eau qui symétriquement à moi s’étaient élevés au-dessus de la même masse liquide qui les avait alimentés. Sans doute les circonstances ayant toujours quelque chose de particulier et les caractères d’individuel, c’était de façons toutes différentes que Legrandin (par l’étrange mariage de son neveu) à son tour avait pénétré dans ce milieu, que la fille d’Odette s’y était apparentée, que Swann lui-même, et moi enfin y étions venus. Pour moi qui avais passé enfermé dans ma vie et la voyant du dedans, celle de Legrandin me semblait n’avoir aucun rapport et avoir suivi un chemin opposé, de même que celui qui suit le cours d’une rivière dans sa vallée profonde ne voit pas qu’une rivière divergente, malgré les écarts de son cours, se jette dans le même fleuve. Mais à vol d’oiseau, comme fait le statisticien qui néglige la raison sentimentale, les imprudences évitables qui ont conduit telle personne à la mort, et compte seulement le nombre de personnes qui meurent par an, on voyait que plusieurs personnes, parties d’un même milieu dont la peinture a occupé le début de ce récit, étaient parvenues dans un autre tout différent, et il est probable que, comme il se fait par an à Paris un nombre moyen de mariages, tout autre milieu bourgeois cultivé et riche eût fourni une proportion à peu près égale de gens comme Swann, comme Legrandin, comme moi et comme Bloch, qu’on retrouverait se jetant dans l’océan du « grand monde ». Et, d’ailleurs, ils s’y reconnaissaient, car si le jeune comte de Cambremer émerveillait tout le monde par sa distinction, sa grâce, sa sobre élégance, je reconnaissais en elles — en même temps que dans son beau regard et dans son désir ardent de parvenir — ce qui caractérisait déjà son oncle Legrandin, c’est-à-dire un vieil ami fort bourgeois, quoique de tournure aristocratique, de mes parents.

La bonté, simple maturation qui a fini par sucrer des natures plus primitivement acides que celle de Bloch, est aussi répandue que ce sentiment de la justice qui fait que, si notre cause est bonne, nous ne devons pas plus redouter un juge prévenu qu’un juge ami. Et les petits-enfants de Bloch seraient bons et discrets presque de naissance. Bloch n’en était peut-être pas encore là. Mais je remarquai que lui, qui jadis feignait de se croire obligé à faire deux heures de chemin de fer pour aller voir quelqu’un qui ne le lui avait guère demandé, maintenant qu’il recevait beaucoup d’invitations, non seulement à déjeuner et à dîner, mais à venir passer quinze jours ici, quinze jours là, en refusait beaucoup et sans le dire, sans se vanter de les avoir reçues, de les avoir refusées. La discrétion, discrétion dans les actions, dans les paroles, lui était venue avec la situation sociale et l’âge, avec une sorte d’âge social, si l’on peut dire. Sans doute Bloch était jadis indiscret autant qu’incapable de bienveillance et de conseils. Mais certains défauts, certaines qualités sont moins attachés à tel individu, à tel autre, qu’à tel ou tel moment de l’existence considéré au point de vue social. Ils sont presque extérieurs aux individus, lesquels passent dans leur lumière comme sous des solstices variés, préexistants, généraux, inévitables. Les médecins qui cherchent à se rendre compte si tel médicament diminue ou augmente l’acidité de l’estomac, active ou ralentit ses sécrétions, obtiennent des résultats différents, non pas selon l’estomac sur les sécrétions duquel ils prélèvent un peu de suc gastrique, mais selon qu’ils le lui empruntent à un moment plus ou moins avancé de l’ingestion du remède.

Ainsi, à chacun des moments de sa durée, le nom de Guermantes, considéré comme un ensemble de tous les noms qu’il admettait en lui, autour de lui, subissait des déperditions, recrutait des éléments nouveaux, comme ces jardins où à tout moment des fleurs à peine en bouton et se préparant à remplacer celles qui se flétrissent déjà se confondent dans une masse qui semble pareille, sauf à ceux qui n’ont pas toujours vu les nouvelles venues et gardent dans leur souvenir l’image précise de celles qui ne sont plus.
