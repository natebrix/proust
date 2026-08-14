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

Dans toute cette conversation, Gilberte m’avait parlé de Robert avec une déférence qui semblait plus s’adresser à mon ancien ami qu’à son époux défunt. Elle avait l’air de me dire : « Je sais combien vous l’admiriez. Croyez bien que j’ai su comprendre l’être supérieur qu’il était. » Et pourtant, l’amour que certainement elle n’avait plus pour son souvenir était peut-être encore la cause lointaine de particularités de sa vie actuelle. Ainsi Gilberte avait maintenant pour amie inséparable Andrée. Quoique celle-ci commençât, surtout à la faveur du talent de son mari et de sa propre intelligence, à pénétrer non pas, certes, dans le milieu des Guermantes, mais dans un monde infiniment plus élégant que celui qu’elle fréquentait jadis, on fut étonné que la marquise de Saint-Loup condescendît à devenir sa meilleure amie. Le fait sembla être un signe, chez Gilberte, de son penchant pour ce qu’elle croyait une existence artistique, et pour une véritable déchéance sociale. Cette explication peut être la vraie. Une autre pourtant vint à mon esprit, toujours fort pénétré de ce fait que les images que nous voyons assemblées quelque part sont généralement le reflet, ou d’une façon quelconque l’effet, d’un premier groupement, assez différent quoique symétrique, d’autres images extrêmement éloignées du second. Je pensais que si on voyait tous les soirs ensemble Andrée, son mari et Gilberte, c’était peut-être parce que, tant d’années auparavant, on avait pu voir le futur mari d’Andrée vivant avec Rachel, puis la quittant pour Andrée. Il est probable que Gilberte alors, dans le monde trop distant, trop élevé, où elle vivait, n’en avait rien su. Mais elle avait dû l’apprendre plus tard, quand Andrée avait monté et qu’elle-même avait descendu assez pour qu’elles pussent s’apercevoir. Alors avait dû exercer sur elle un grand prestige de la femme pour laquelle Rachel avait été quittée par l’homme, pourtant séduisant sans doute, qu’elle avait préféré à Robert.

### Passage

Ainsi peut-être la vue d’Andrée rappelait à Gilberte le roman de jeunesse qu’avait été son amour pour Robert, et lui inspirait aussi un grand respect pour Andrée, de laquelle était toujours amoureux un homme tant aimé par cette Rachel que Gilberte sentait avoir été plus aimée de Saint-Loup qu’elle ne l’avait été elle-même. Peut-être, au contraire, ces souvenirs ne jouaient-ils aucun rôle dans la prédilection de Gilberte pour ce ménage artiste et fallait-il y voir simplement — comme chez beaucoup — l’épanouissement des goûts, habituellement inséparables chez les femmes du monde, de s’instruire et de s’encanailler. Peut-être Gilberte avait-elle oublié Robert autant que moi Albertine, et si même elle savait que c’était Rachel que l’artiste avait quittée pour Andrée, ne pensait-elle jamais, quand elle les voyait, à ce fait qui n’avait jamais joué aucun rôle dans son goût pour eux. On n’aurait pu décider si mon explication première n’était pas seulement possible, mais était vraie, que grâce au témoignage des intéressés, seul recours qui reste en pareil cas, s’ils pouvaient apporter dans leurs confidences de la clairvoyance et de la sincérité. Or la première s’y rencontre rarement et la seconde jamais.

« Mais comment venez-vous dans des matinées si nombreuses ? me demanda Gilberte. Vous retrouver dans une grande tuerie comme cela, ce n’est pas ainsi que je vous schématisais. Certes, je m’attendais à vous voir partout ailleurs qu’à un des grands tralalas de ma tante, puisque tante il y a », ajouta-t-elle d’un air fin, car étant Mme de Saint-Loup depuis un peu plus longtemps que Mme Verdurin n’était entrée dans la famille, elle se considérait comme une Guermantes de tout temps et atteinte par la mésalliance que son oncle avait faite en épousant Mme Verdurin, qu’il est vrai elle avait entendu railler mille fois devant elle, dans la famille, tandis que, naturellement, ce n’était que hors de sa présence qu’on avait parlé de la mésalliance qu’avait faite Saint-Loup en l’épousant. Elle affectait, d’ailleurs, d’autant plus de dédain pour cette tante mauvais teint que la princesse de Guermantes, par l’espèce de perversion qui pousse les gens intelligents à s’évader du chic habituel, par le besoin aussi de souvenirs qu’ont les gens âgés, pour tâcher de donner un passé à son élégance nouvelle aimait à dire, en parlant de Gilberte : « Je vous dirai que ce n’est pas pour moi une relation nouvelle, j’ai énormément connu la mère de cette petite ; tenez, c’était une grande amie à ma cousine Marsantes. C’est chez moi qu’elle a connu le père de Gilberte. Quant au pauvre Saint-Loup, je connaissais d’avance toute sa famille, son propre oncle était mon intime autrefois à la Raspelière. » « Vous voyez que les Verdurin n’étaient pas du tout des bohèmes, me disaient les gens qui entendaient parler ainsi la princesse de Guermantes, c’étaient des amis de tout temps de la famille de Mme de Saint-Loup. » J’étais peut-être seul à savoir par mon grand-père qu’en effet les Verdurin n’étaient pas des bohèmes. Mais ce n’était pas précisément parce qu’ils avaient connu Odette. Mais on arrange aisément les récits du passé que personne ne connaît plus, comme ceux des voyages dans les pays où personne n’est jamais allé. « Enfin, conclut Gilberte, puisque vous sortez quelquefois de votre Tour d’Ivoire, des petites réunions intimes chez moi, où j’inviterais des esprits sympathiques, ne vous conviendraient-elles pas mieux ? Ces grandes machines comme ici sont bien peu faites pour vous. Je vous voyais causer avec ma tante Oriane, qui a toutes les qualités qu’on voudra, mais à qui nous ne ferons pas tort, n’est-ce pas, en déclarant qu’elle n’appartient pas à l’élite pensante. » Je ne pouvais mettre Gilberte au courant des pensées que j’avais depuis une heure, mais je crus que, sur un point de pure distraction, elle pourrait servir mes plaisirs, lesquels, en effet, ne me semblaient pas devoir être de parler littérature avec la duchesse de Guermantes plus qu’avec Mme de Saint-Loup. Certes, j’avais l’intention de recommencer dès demain, bien qu’avec un but cette fois, à vivre dans la solitude. Même chez moi je ne laisserais pas les gens venir me voir dans mes instants de travail, car le devoir de faire mon œuvre primait celui d’être poli, ou même bon. Ils insisteraient sans doute. Ceux qui ne m’avaient pas vu depuis si longtemps, venaient de me retrouver et me jugeaient guéri. Ils insisteraient, venant quand le labeur de leur journée, de leur vie, serait fini ou interrompu, et ayant alors le même besoin de moi que j’avais eu autrefois de Saint-Loup, et cela parce que, comme je m’en étais aperçu à Combray quand mes parents me faisaient des reproches au moment où je venais de prendre à leur insu les plus louables résolutions, les cadrans intérieurs qui sont départis aux hommes ne sont pas tous réglés à la même heure, l’un sonne celle du repos en même temps que l’autre celle du travail, l’un celle du châtiment par le juge quand chez le coupable celle du repentir et du perfectionnement intérieur est sonnée depuis longtemps. Mais j’aurais le courage de répondre à ceux qui viendraient me voir ou me feraient chercher que j’avais, pour des choses essentielles au courant desquelles il fallait que je fusse mis sans retard, un rendez-vous urgent, capital, avec moi-même. Et pourtant, bien qu’il y ait peu de rapport entre notre moi véritable et l’autre, à cause de l’homonymat et du corps commun aux deux, l’abnégation qui vous fait faire le sacrifice des devoirs plus faciles, même des plaisirs, paraît aux autres de l’égoïsme. Et d’ailleurs, n’était-ce pas pour m’occuper d’eux que je vivrais loin de ceux qui se plaindraient de ne pas me voir, pour m’occuper d’eux plus à fond que je n’aurais pu le faire avec eux, pour chercher à les révéler à eux-mêmes, à les réaliser ? À quoi eût servi que, pendant des années encore, j’eusse perdu des soirées à faire glisser sur l’écho à peine expiré de leurs paroles le son tout aussi vain des miennes, pour le stérile plaisir d’un contact mondain qui exclut toute pénétration ? Ne valait-il pas mieux que ces gestes qu’ils faisaient, ces paroles qu’ils disaient, leur vie, leur nature, j’essayasse d’en décrire la courbe et d’en dégager la loi ? Malheureusement, j’aurais à lutter contre cette habitude de se mettre à la place des autres qui, si elle favorise la conception d’une œuvre, en retarde l’exécution. Car, par une politesse supérieure, elle pousse à sacrifier aux autres non seulement son plaisir, mais son devoir, quand, se mettant à la place des autres, le devoir quel qu’il soit, fût-ce, pour quelqu’un qui ne peut rendre aucun service au front, de rester à l’arrière s’il est utile, paraîtra comme, ce qu’il n’est pas en réalité, notre plaisir. Et bien loin de me croire malheureux de cette vie sans amis, sans causerie, comme il est arrivé aux plus grands de le croire, je me rendais compte que les forces d’exaltation qui se dépensent dans l’amitié sont une sorte de porte-à-faux visant une amitié particulière qui ne mène à rien et se détournent d’une vérité vers laquelle elles étaient capables de nous conduire. Mais enfin, quand des intervalles de repos et de société me seraient nécessaires, je sentais que, plutôt que les conversations intellectuelles que les gens du monde croient utiles aux écrivains, de légères amours avec des jeunes filles en fleurs seraient un aliment choisi que je pourrais à la rigueur permettre à mon imagination semblable au cheval fameux qu’on ne nourrissait que de roses ! Ce que tout d’un coup je souhaitais de nouveau, c’est ce dont j’avais rêvé à Balbec, quand, sans les connaître encore, j’avais vu passer devant la mer Albertine, Andrée et leurs amies. Mais hélas ! je ne pouvais plus chercher à retrouver celles que justement en ce moment je désirais si fort. L’action des années qui avait transformé tous les êtres que j’avais vus aujourd’hui, et Gilberte elle-même, avait certainement fait de toutes celles qui survivaient, comme elle eût fait d’Albertine si elle n’avait pas péri, des femmes trop différentes de ce que je me rappelais. Je souffrais d’être obligé de moi-même à atteindre celles-là, car le temps qui change les êtres ne modifie pas l’image que nous avons gardée d’eux. Rien n’est plus douloureux que cette opposition entre l’altération des êtres et la fixité du souvenir, quand nous comprenons que ce qui a gardé tant de fraîcheur dans notre mémoire n’en peut plus avoir dans la vie, que nous ne pouvons, au dehors, nous rapprocher de ce qui nous paraît si beau au-dedans de nous, de ce qui excite en nous un désir, pourtant si individuel, de le revoir. Ce violent désir que la mémoire excitait en moi pour ces jeunes filles vues jadis, je sentais que je ne pourrais espérer l’assouvir qu’à condition de le chercher dans un être du même âge, c’est-à-dire dans un autre être. J’avais pu souvent soupçonner que ce qui semble unique dans une personne qu’on désire ne lui appartient pas. Mais le temps écoulé m’en donnait une preuve plus complète, puisque, après vingt ans, spontanément, je voulais chercher, au lieu des filles que j’avais connues, celles possédant maintenant la jeunesse que les autres avaient alors. D’ailleurs, ce n’est pas seulement le réveil de nos désirs charnels qui ne correspond à aucune réalité parce qu’il ne tient pas compte du temps perdu. Il m’arrivait parfois de souhaiter que par un miracle vinssent auprès de moi, restées vivantes contrairement à ce que j’avais cru, ma grand’mère, Albertine. Je croyais les voir, mon cœur s’élançait vers elles. J’oubliais seulement une chose, c’est que, si elles vivaient en effet, Albertine aurait à peu près maintenant l’aspect que m’avait présenté à Balbec Mme Cottard, et que ma grand’mère, ayant plus de quatre-vingt-quinze ans, ne me montrerait rien du beau visage calme et souriant avec lequel je l’imaginais encore maintenant, aussi arbitrairement qu’on donne une barbe à Dieu le Père, ou qu’on représentait, au xviie siècle, les héros d’Homère avec un accoutrement de gentilshommes et sans tenir compte de leur antiquité. Je regardai Gilberte et je ne pensai pas : « Je voudrais la revoir », mais je lui dis qu’elle me ferait toujours plaisir en m’invitant avec des jeunes filles, sans que j’eusse, d’ailleurs, à leur rien demander que de faire renaître en moi les rêveries, les tristesses d’autrefois, peut-être, un jour improbable, un chaste baiser. Comme Elstir aimait à voir incarnée devant lui, dans sa femme, la beauté vénitienne, qu’il avait si souvent peinte dans ses œuvres, je me donnais l’excuse d’être attiré, par un certain égoïsme esthétique, vers les belles femmes qui pouvaient me causer de la souffrance, et j’avais un certain sentiment d’idolâtrie pour les futures Gilberte, les futures duchesses de Guermantes, les futures Albertine que je pourrais rencontrer, et qui, me semblait-il, pourraient m’inspirer, comme un sculpteur qui se promène au milieu de beaux marbres antiques. J’aurais dû pourtant penser qu’antérieur à chacune était mon sentiment du mystère où elles baignaient et qu’ainsi, plutôt que de demander à Gilberte de me faire connaître des jeunes filles, j’aurais mieux fait d’aller dans ces lieux où rien ne nous rattache à elles, où entre elles et soi on sent quelque chose d’infranchissable, où, à deux pas, sur la plage, allant au bain, on se sent séparé d’elles par l’impossible. C’est ainsi que mon sentiment du mystère avait pu s’appliquer successivement à Gilberte, à la duchesse de Guermantes, à Albertine, à tant d’autres. Sans doute l’inconnu et presque l’inconnaissable était devenu le commun, le familier, indifférent ou douloureux, mais retenant de ce qu’il avait été un certain charme. Et, à vrai dire, comme dans ces calendriers que le facteur nous apporte pour avoir ses étrennes, il n’était pas une de mes années qui n’ait eu à son frontispice, ou intercalée dans ses jours, l’image d’une femme que j’y avais désirée ; image souvent d’autant plus arbitraire que parfois je n’avais pas vu cette femme, quand c’était, par exemple, la femme de chambre de Mme Putbus, Mlle d’Orgeville, ou telle jeune fille dont j’avais vu le nom dans le compte rendu mondain d’un journal, parmi l’essaim des charmantes valseuses. Je la devinais belle, m’éprenais d’elle, et lui composais un corps idéal dominant de toute sa hauteur un paysage de la province où j’avais lu, dans l’Annuaire des Châteaux, que se trouvaient les propriétés de sa famille. Pour les femmes que j’avais connues, ce paysage était au moins double. Chacune s’élevait, à un point différent de ma vie, dressée comme une divinité protectrice et locale, d’abord au milieu d’un de ces paysages rêvés dont la juxtaposition quadrillait ma vie et où je m’étais attaché à l’imaginer ; ensuite, vue du côté du souvenir entourée des sites où je l’avais connue et qu’elle me rappelait, y restant attachée, car si notre vie est vagabonde notre mémoire est sédentaire, et nous avons beau nous élancer sans trêve, nos souvenirs, eux, rivés aux lieux dont nous nous détachons, continuent à y continuer leur vie casanière, comme ces amis momentanés que le voyageur s’était faits dans une ville et qu’il est obligé d’abandonner quand il la quitte, parce que c’est là qu’eux, qui ne partent pas, finiront leur journée et leur vie comme s’il était là encore, au pied de l’église, devant la porte et sous les arbres du cours. Si bien que l’ombre de Gilberte s’allongeait, non seulement devant une église de l’Île-de-France où je l’avais imaginée, mais aussi sur l’allée d’un parc, du côté de Méséglise, celle de Mme de Guermantes dans un chemin humide où montaient en quenouilles des grappes violettes et rougeâtres, ou sur l’or matinal d’un trottoir parisien. Et cette seconde personne, celle née non du désir, mais du souvenir, n’était, pour chacune de ces femmes, unique. Car, chacune, je l’avais connue à diverses reprises, en des temps différents où elle était une autre pour moi, où moi-même j’étais autre, baignant dans des rêves d’une autre couleur. Or la loi qui avait gouverné les rêves de chaque année maintenant assemblés autour d’eux les souvenirs d’une femme que j’y avais connue, tout ce qui se rapportait, par exemple, à la duchesse de Guermantes au temps de mon enfance, était concentré, par une force attractive, autour de Combray, et tout ce qui avait trait à la duchesse de Guermantes qui allait tout à l’heure m’inviter à déjeuner, autour d’un sensitif tout différent ; il y avait plusieurs duchesses de Guermantes, comme il y avait eu, depuis la dame en rose, plusieurs Mmes Swann, séparées par l’éther incolore des années, et de l’une à l’autre desquelles je ne pouvais pas plus sauter que si j’avais eu à quitter une planète pour aller dans une autre planète que l’éther en sépare. Non seulement séparée, mais différente, parée des rêves que j’avais eus dans des temps si différents, comme d’une flore particulière, qu’on ne retrouvera pas dans une autre planète ; au point qu’après avoir pensé que je n’irais déjeuner ni chez Mme de Forcheville, ni chez Mme de Guermantes, je ne pouvais me dire, tant cela m’eût transporté dans un monde autre, que l’une n’était pas une personne différente de la duchesse de Guermantes qui descendait de Geneviève de Brabant, et l’autre de la Dame en rose, que parce qu’en moi un homme instruit me l’affirmait avec la même autorité qu’un savant qui m’eût affirmé qu’une voie lactée de nébuleuses était due à la segmentation d’une seule et même étoile. Telle Gilberte, à qui je demandais pourtant, sans m’en rendre compte, de me permettre d’avoir des amies comme elle avait été autrefois, n’était plus pour moi que Mme de Saint-Loup. Je ne songeais plus en la voyant au rôle qu’avait eu jadis dans mon amour, oublié lui aussi par elle, mon admiration pour Bergotte, pour Bergotte redevenu pour moi simplement l’auteur de ses livres, sans que je me rappelasse (que dans des souvenirs rares et entièrement séparés) l’émoi d’avoir été présenté à l’homme, la déception, l’étonnement de sa conversation, dans le salon aux fourrures blanches, plein de violettes, où on apportait si tôt, sur tant de consoles différentes, tant de lampes. Tous les souvenirs qui composaient la première mademoiselle Swann étaient, en effet, retranchés de la Gilberte actuelle, retenus bien loin par les forces d’attraction d’un autre univers, autour d’une phrase de Bergotte avec laquelle ils faisaient corps et baignés d’un parfum d’aubépine. La fragmentaire Gilberte d’aujourd’hui écouta ma requête en souriant. Puis, en se mettant à y réfléchir, elle prit un air sérieux en ayant l’air de chercher dans sa tête. Et j’en fus heureux car cela l’empêcha de faire attention à un groupe qui se trouvait non loin de nous et dont la vue n’eût pu certes lui être agréable. On y remarquait la duchesse de Guermantes en grande conversation avec une affreuse vieille femme que je regardais sans pouvoir du tout deviner qui elle était : je n’en savais absolument rien. « Comme c’est drôle de voir ici Rachel », me dit à l’oreille Bloch qui passait à ce moment. Ce nom magique rompit aussitôt l’enchantement qui avait donné à la maîtresse de Saint-Loup la forme inconnue de cette immonde vieille, et je la reconnus alors parfaitement. De même, j’ai dit ailleurs que dès qu’on me nommait les hommes dont je ne pouvais reconnaître les visages l’enchantement cessait, et que je les reconnaissais. Pourtant il y en eut un que, même nommé, je ne pus reconnaître, et je crus à un homonyme, car il n’avait aucune espèce de rapport avec celui que non seulement j’avais connu autrefois mais que j’avais retrouvé il y a quelques années. C’était pourtant lui, blanchi seulement et engraissé, mais il avait rasé ses moustaches et cela avait suffi pour lui faire perdre sa personnalité. Pour en revenir à Rachel, c’était bien avec elle, devenue une actrice célèbre et qui allait, au cours de cette matinée, réciter des vers de Musset et de La Fontaine, que la tante de Gilberte, la duchesse de Guermantes, causait en ce moment. Or la vue de Rachel ne pouvait en tout cas être bien agréable à Gilberte, et je fus d’autant plus ennuyé d’apprendre qu’elle allait réciter des vers et de constater son intimité avec la duchesse. Celle-ci, consciente depuis trop longtemps d’occuper la première situation de Paris (ne se rendant pas compte qu’une telle situation n’existe que dans les esprits qui y croient et que beaucoup de nouvelles personnes, si elles ne la voyaient nulle part, si elles ne lisaient son nom dans le compte rendu d’aucune fête élégante, croiraient, en effet, qu’elle n’occupait aucune situation), ne voyait plus, qu’en visites aussi rares et aussi espacées qu’elle pouvait, le faubourg Saint-Germain qui, disait-elle, « l’ennuyait à mourir », et, en revanche, se passait la fantaisie de déjeuner avec telle ou telle actrice qu’elle trouvait délicieuse.

La duchesse hésitait encore, par peur d’une scène de M. de Guermantes, devant Balthy et Mistinguett, qu’elle trouvait adorables, mais avait décidément Rachel pour amie. Les nouvelles générations en concluaient que la duchesse de Guermantes, malgré son nom, devait être quelque demi-castor qui n’avait jamais été tout à fait du gratin. Il est vrai que, pour quelques souverains dont l’intimité lui était disputée par deux autres grandes dames, Mme de Guermantes se donnait encore la peine de les avoir à déjeuner. Mais, d’une part, ils viennent rarement, connaissent des gens de peu, et la duchesse, par la superstition des Guermantes à l’égard du vieux protocole (car à la fois les gens bien élevés l’assommaient et elle tenait à la bonne éducation), faisait mettre : « Sa Majesté a ordonné à la duchesse de Guermantes », « a daigné », etc. Et les nouvelles couches, ignorantes de ces formules, en concluaient que la position de la duchesse était d’autant plus basse. Au point de vue de Mme de Guermantes, cette intimité avec Rachel pouvait signifier que nous nous étions trompés quand nous croyions Mme de Guermantes hypocrite et menteuse dans ses condamnations de l’élégance, quand nous croyions qu’au moment où elle refusait d’aller chez Mme de Sainte-Euverte, ce n’était pas au nom de l’intelligence mais du snobisme qu’elle agissait ainsi, ne la trouvant bête que parce que la marquise laissait voir qu’elle était snob, n’ayant pas encore atteint son but. Mais cette intimité avec Rachel pouvait signifier aussi que l’intelligence était, en réalité, chez la duchesse, médiocre, insatisfaite et désireuse sur le tard, quand elle était fatiguée du monde, de réalisations, par ignorance totale des véritables réalités intellectuelles et une pointe de cet esprit de fantaisie qui fait à des dames très bien, qui se disent : « comme ce sera amusant », finir leur soirée d’une façon à vrai dire assommante, en puisant la force d’aller réveiller quelqu’un, à qui finalement on ne sait que dire, près du lit de qui on reste un moment dans son manteau de soirée, après quoi, ayant constaté qu’il est fort tard, on finit par aller se coucher.

Il faut ajouter qu’une vive antipathie qu’avait depuis peu pour Gilberte la versatile duchesse pouvait lui faire prendre un certain plaisir à recevoir Rachel, ce qui lui permettait, en plus, de proclamer une des maximes des Guermantes, à savoir qu’ils étaient trop nombreux pour épouser les querelles (presque pour prendre le deuil) les uns des autres, indépendance de « je n’ai pas à » qu’avait renforcée la politique qu’on avait dû adopter à l’égard de M. de Charlus, lequel, si on l’avait suivi, vous eût brouillé avec tout le monde. Quant à Rachel, si elle s’était, en réalité, donné une grande peine pour se lier avec la duchesse de Guermantes (peine que la duchesse n’avait pas su démêler sous des dédains affectés, des impolitesses voulues, qui l’avaient piquée au jeu et lui avaient donné grande idée d’une actrice si peu snob), sans doute cela tenait, d’une façon générale, à la fascination que les gens du monde exercent à partir d’un certain moment sur les bohèmes les plus endurcis, parallèle à celle que ces bohèmes exercent eux-mêmes sur les gens du monde, double reflux qui correspond à ce qu’est, dans l’ordre politique, la curiosité réciproque et le désir de faire alliance entre peuples qui se sont combattus. Mais le désir de Rachel pouvait avoir une raison plus particulière. C’est chez Mme de Guermantes, c’est de Mme de Guermantes, qu’elle avait reçu jadis sa plus terrible avanie. Rachel l’avait peu à peu non pas oubliée mais pardonnée, mais le prestige singulier qu’en avait reçu à ses yeux la duchesse ne devait s’effacer jamais. L’entretien, de l’attention duquel je désirais détourner Gilberte, fut, du reste, interrompu, car la maîtresse de maison vint chercher Rachel dont c’était le moment de réciter et qui bientôt, ayant quitté la duchesse, parut sur l’estrade.

Or, pendant ce temps, avait lieu à l’autre bout de Paris un spectacle bien différent. La Berma avait convié quelques personnes à venir prendre le thé pour fêter son fils et sa belle-fille. Mais les invités ne se pressaient pas d’arriver. Ayant appris que Rachel récitait des vers chez la princesse de Guermantes (ce qui scandalisait fort la Berma, grande artiste pour laquelle Rachel était restée une grue qu’on laissait figurer dans les pièces où elle-même, la Berma, jouait le premier rôle — parce que Saint-Loup lui payait ses toilettes pour la scène — scandale d’autant plus grand que la nouvelle avait couru dans Paris que les invitations étaient au nom de la princesse de Guermantes, mais que c’était Rachel qui, en réalité, recevait chez la princesse), la Berma avait récrit avec insistance à quelques fidèles pour qu’ils ne manquassent pas à son goûter, car elle les savait aussi amis de la princesse de Guermantes qu’ils avaient connue Verdurin. Or, les heures passaient et personne n’arrivait chez la Berma. Bloch, à qui on avait demandé s’il voulait y venir, avait répondu naïvement : « Non, j’aime mieux aller chez la princesse de Guermantes. » Hélas ! c’est ce qu’au fond de soi chacun avait décidé. La Berma, atteinte d’une maladie mortelle qui la forçait à fréquenter peu le monde, avait vu son état s’aggraver quand, pour subvenir aux besoins de luxe de sa fille, besoins que son gendre, souffrant et paresseux, ne pouvait satisfaire, elle s’était remise à jouer. Elle savait qu’elle abrégeait ses jours, mais voulait faire plaisir à sa fille à qui elle rapportait de gros cachets, à son gendre qu’elle détestait mais flattait, car, le sachant adoré par sa fille, elle craignait, si elle le mécontentait, qu’il la privât, par méchanceté, de voir celle-ci. La fille de la Berma, qui n’était cependant pas positivement cruelle et était aimée en secret par le médecin qui soignait sa mère, s’était laissé persuader que ces représentations de Phèdre n’étaient pas bien dangereuses pour la malade. Elle avait en quelque sorte forcé le médecin à le lui dire, n’ayant retenu que cela de ce qu’il lui avait répondu, et parmi des objections dont elle ne tenait pas compte ; en effet, le médecin avait dit ne pas voir grand inconvénient aux représentations de la Berma ; il l’avait dit parce qu’il sentait qu’il ferait ainsi plaisir à la jeune femme qu’il aimait, peut-être aussi par ignorance, parce qu’aussi il savait de toutes façons la maladie inguérissable, et qu’on se résigne volontiers à abréger le martyre des malades quand ce qui est destiné à l’abréger nous profite à nous-même, peut-être aussi par la bête conception que cela faisait plaisir à la Berma et devait donc lui faire du bien, bête conception qui lui parut justifiée quand, ayant reçu une loge des enfants de la Berma et ayant pour cela lâché tous ses malades, il l’avait trouvée aussi extraordinaire de vie sur la scène qu’elle semblait moribonde à la ville. Et, en effet, nos habitudes nous permettent dans une large mesure, permettent même à nos organismes, de s’accommoder d’une existence qui semblerait au premier abord ne pas être possible. Qui n’a vu un vieux maître de manège cardiaque faire toutes les acrobaties auxquelles on n’aurait pu croire que son cœur résisterait une minute ? La Berma n’était pas une moins vieille habituée de la scène, aux exigences de laquelle ses organes étaient si parfaitement adaptés qu’elle pouvait donner, en se dépensant avec une prudence indiscernable pour le public, l’illusion d’une bonne santé troublée seulement par un mal purement nerveux et imaginaire. Après la scène de la déclaration à Hippolyte, la Berma avait beau sentir l’épouvantable nuit qu’elle allait passer, ses admirateurs l’applaudissaient à toute force, la déclarant plus belle que jamais. Elle rentrait dans d’horribles souffrances mais heureuse d’apporter à sa fille les billets bleus, que, par une gaminerie de vieille enfant de la balle, elle avait l’habitude de serrer dans ses bas, d’où elle les sortait avec fierté, espérant un sourire, un baiser. Malheureusement, ces billets ne faisaient que permettre au gendre et à la fille de nouveaux embellissements de leur hôtel, contigu à celui de leur mère, d’où d’incessants coups de marteau qui interrompaient le sommeil dont la grande tragédienne aurait eu tant besoin. Selon les variations de la mode, et pour se conformer au goût de M. de X. ou de Y., qu’ils espéraient recevoir, ils modifiaient chaque pièce. Et la Berma, sentant que le sommeil, qui seul aurait calmé sa souffrance, s’était enfui, se résignait à ne pas se rendormir, non sans un secret mépris pour ces élégances qui avançaient sa mort, rendaient atroces ses derniers jours. C’est sans doute un peu à cause de cela qu’elle les méprisait, vengeance naturelle contre ce qui nous fait mal et que nous sommes impuissants à empêcher. Mais c’est aussi parce qu’ayant conscience du génie qui était en elle, ayant appris dès son plus jeune âge l’insignifiance de tous ces décrets de la mode, elle était quant à elle restée fidèle à la tradition qu’elle avait toujours respectée, dont elle était l’incarnation, qui lui faisait juger les choses et les gens comme trente ans auparavant, et, par exemple, juger Rachel non comme l’actrice à la mode qu’elle était devenue, mais comme la petite grue qu’elle avait connue. La Berma n’était pas, du reste, meilleure que sa fille, c’est en elle que sa fille avait puisé, par l’hérédité et par la contagion de l’exemple, qu’une admiration trop naturelle rendait plus efficace, son égoïsme, son impitoyable raillerie, son inconsciente cruauté. Seulement, tout cela la Berma l’avait immolé à sa fille et s’en était ainsi délivrée. D’ailleurs, la fille de la Berma n’eût-elle pas eu sans cesse des ouvriers chez elle, qu’elle eût fatigué sa mère, comme les forces attractives féroces et légères de la jeunesse fatiguent la vieillesse, la maladie, qui se surmènent à vouloir les suivre. Tous les jours c’était un déjeuner nouveau, et on eût trouvé la Berma égoïste d’en priver sa fille, même de ne pas assister au déjeuner où on comptait, pour attirer bien difficilement quelques relations récentes et qui se faisaient tirer l’oreille, sur la présence prestigieuse de la mère illustre. On la « promettait » à ces mêmes relations pour une fête au dehors, afin de leur faire « une politesse ». Et la pauvre mère, gravement occupée dans son tête-à-tête avec la mort installée en elle, était obligée de se lever de bonne heure, de sortir. Bien plus, comme, à la même époque, Réjane, dans tout l’éblouissement de son talent, donna à l’étranger des représentations qui eurent un succès énorme, le gendre trouva que la Berma ne devait pas se laisser éclipser, voulut que la famille ramassât la même profusion de gloire, et força la Berma à des tournées où on était obligé de la piquer à la morphine, ce qui pouvait la faire mourir à cause de l’état de ses reins. Ce même attrait de l’élégance, du prestige social, de la vie, avait, le jour de la fête chez la princesse de Guermantes, fait pompe aspirante et avait amené là-bas, avec la force d’une machine pneumatique, même les plus fidèles habitués de la Berma, où, par contre et en conséquence, il y avait vide absolu et mort. Un seul jeune homme, qui n’était pas certain que la fête chez la Berma ne fût, elle aussi, brillante, était venu. Quand la Berma vit l’heure passer et comprit que tout le monde la lâchait, elle fit servir le goûter et on s’assit autour de la table, mais comme pour un repas funéraire. Rien dans la figure de la Berma ne rappelait plus celle dont la photographie m’avait, un soir de mi-carême, tant troublé. La Berma avait, comme dit le peuple, la mort sur le visage. Cette fois c’était bien d’un marbre de l’Erechtéion qu’elle avait l’air. Ses artères durcies étant déjà à demi pétrifiées, on voyait de longs rubans sculpturaux parcourir les joues, avec une rigidité minérale. Les yeux mourants vivaient relativement, par contraste avec ce terrible masque ossifié, et brillaient faiblement comme un serpent endormi au milieu des pierres. Cependant le jeune homme, qui s’était mis à la table par politesse, regardait sans cesse l’heure, attiré qu’il était par la brillante fête chez les Guermantes. La Berma n’avait pas un mot de reproche à l’adresse des amis qui l’avaient lâchée et qui espéraient naïvement qu’elle ignorerait qu’ils étaient allés chez les Guermantes. Elle murmura seulement : « Une Rachel donnant une fête chez la princesse de Guermantes, il faut venir à Paris pour voir de ces choses-là. » Et elle mangeait silencieusement, et avec une lenteur solennelle, des gâteaux défendus, ayant l’air d’obéir à des rites funèbres. Le « goûter » était d’autant plus triste que le gendre était furieux que Rachel, que lui et sa femme connaissaient très bien, ne les eût pas invités. Son crève-cœur fut d’autant plus grand que le jeune homme invité lui avait dit connaître assez bien Rachel pour que, s’il partait tout de suite chez les Guermantes, il pût lui demander d’inviter ainsi, à la dernière heure, le couple frivole. Mais la fille de la Berma savait trop à quel niveau infime sa mère situait Rachel, et qu’elle l’eût tuée de désespoir en sollicitant de l’ancienne grue une invitation. Aussi avait-elle dit au jeune homme et à son mari que c’était chose impossible. Mais elle se vengeait en prenant pendant ce goûter des petites mines exprimant le désir des plaisirs, l’ennui d’être privée d’eux par cette gêneuse qu’était sa mère. Celle-ci faisait semblant de ne pas voir les moues de sa fille et adressait de temps en temps, d’une voix mourante, une parole aimable au jeune homme, le seul invité qui fût venu. Mais bientôt la chasse d’air qui emportait tout vers les Guermantes, et qui m’y avait entraîné moi-même, fut la plus forte, il se leva et partit, laissant Phèdre ou la mort, on ne savait trop laquelle des deux c’était, achever de manger, avec sa fille et son gendre, les gâteaux funéraires.
