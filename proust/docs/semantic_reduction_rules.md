# Semantic Reduction Rules

These rules capture the remaining interpretive decisions that generic compression scores do not handle well enough.

They are intended for the reducer stage that turns richer model outputs into the first-pass annotation schema.

## Purpose

The reducer should not only make outputs smaller.

It should preserve the **same dominant reading** a careful human reviewer would prefer for the first-pass benchmark.

## Rule 1: Local snubs are usually not rank-collapse events

When the surviving event is a local `snub`, prefer:

- `general_appraisal -1`
- `inclusion_exclusion -1`

Do **not** default to `social_status -2` unless the passage is explicitly about rank, title, or access to a higher social stratum.

Reason:

- many Combray passages show condescension, underestimation, or exclusion without implying a literal collapse in rank
- the reviewed benchmark treats these as local slight plus withholding of honorary inclusion

Canonical example:

- `v1-p1-combray#p-21-p-22`

## Rule 2: Ironized self-elevation should not survive narrator-led exposure

If a character’s positive self-presentation is:

- directed at themselves
- ironized
- and overshadowed by a stronger narrator-led negative exposure

drop the positive self-elevation event.

Reason:

- these passages are usually not genuinely mixed
- the self-praise is evidence of the exposure, not a second movement worth preserving

Canonical example:

- `v1-p1-combray#p-274-p-275`

## Rule 3: Preserve explicit mixed targeting on the focal character

If the raw output contains both:

- an explicit positive appraisal of a focal character
- and an explicit negative social appraisal of that same focal character

preserve that shared focal target through reduction whenever possible.

Reason:

- some passages are genuinely mixed, and the tension between positive and negative positioning is the point
- the reducer should not re-center the passage on a source character just because narrator irony is present

Canonical example:

- `v1-p1-combray#p-312-p-313`

Preferred shape in that case:

- positive event on `Swann`
- negative event on `Swann`
- status effects centered on `Swann`

## Rule 4: Broad narrator-led movements absorb subordinate cues

If a passage already has a strong narrator-led:

- `narrated_elevation`
- or `narrated_diminishment`

subordinate same-direction cues such as prestige markers, ridicule, or blame should usually be treated as evidence for that broader event rather than separate survivors.

Canonical examples:

- `v1-p1-combray#p-17`
- `v1-p1-combray#p-274-p-275`

## Rule 5: Exact benchmark equality is not the immediate goal

The reducer is succeeding if it moves raw model output toward:

- the correct focal character
- the correct event count
- the correct dominant polarity structure
- the correct status dimensions

Exact equality on evidence wording, confidence, or surface-form selection can remain a later concern.
