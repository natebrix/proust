# run-003 vs run-002: recurring model failure modes

`run-003` is operationally successful but semantically too loose against the reviewed baseline in `run-002`.

## High-level summary

- Shared units: `10`
- Exact matches: `0`
- All `10` units contain more appraisal events than the reviewed baseline.
- `7` of `10` units contain more status effects than the reviewed baseline.
- `3` of `10` units add extra characters not present in the reviewed baseline.

## Recurring failure modes

### 1. Event over-segmentation

The model repeatedly breaks one reviewed local movement into several smaller events.

Examples:

- `v1-p1-combray#p-17`
  - Reviewed: one `narrated_elevation`
  - Automated: splits into elevation, prestige association, and local diminishment
- `v1-p1-combray#p-21-p-22`
  - Reviewed: one `snub`
  - Automated: splits into exclusion, contempt, diminishment, and compensating elevation
- `v1-p1-combray#p-274-p-275`
  - Reviewed: one `narrated_diminishment`
  - Automated: splits into ridicule, blame, humiliation, social blame, and a small praise counterweight

Effect:

- The output becomes analytically busy.
- Status effects are then computed from micro-events rather than from the passage’s dominant local movement.

### 2. Over-reading mixed or countervailing signals

The model often turns subordinate nuances into standalone positive or balancing events.

Examples:

- `v1-p1-combray#p-20`
  - The reviewed reading treats the aunt’s reframing as the main movement.
  - The automated reading adds an initial positive `prestige_association` from the report of dining with a princess.
- `v1-p1-combray#p-25-p-26`
  - The reviewed reading keeps one positive and one negative movement.
  - The automated reading adds praise, preference, diminishment, maternal solicitude, and a separate snub of the mother.
- `v1-p1-combray#p-312-p-313`
  - The reviewed reading keeps one positive admiration event and one negative marriage-based discredit.
  - The automated reading adds deference, multiple blame events, a face-saving inclusion for Vinteuil, and Swann’s dependence as a separate emotional event.

Effect:

- The model drifts from “main local appraisal/status movement” toward “everything evaluatively mentionable.”

### 3. Scope creep in characters_present

The model sometimes includes nearby figures who are textually present but not central to the focal local movement.

Examples:

- `v1-p1-combray#p-23-p-24`: adds `la mère du narrateur`
- `v1-p1-combray#p-25-p-26`: adds `la mère du narrateur`
- `v1-p1-combray#p-278-p-279`: adds `la mère du narrateur`

Effect:

- The annotation begins to track discourse participants rather than focal social-literary targets.
- This increases downstream event and status clutter.

### 4. Narrative stance drift toward `neutral_report` or `ironized`

The model frequently weakens reviewed `endorsed` judgments into `neutral_report`, or introduces irony where the reviewed reading treats the narrator’s local conclusion as stable.

Examples:

- `v1-p1-combray#p-17`
  - Reviewed `endorsed`
  - Automated `neutral_report`
- `v1-p1-combray#p-20`
  - Reviewed `ironized` for the aunt’s lowering move
  - Automated splits the passage and treats the positive association as `neutral_report`
- `v1-p1-combray#p-278-p-279`
  - Reviewed `endorsed`
  - Automated `ironized`

Effect:

- Narrator-led appraisal gets flattened into descriptive reporting.
- Strong local conclusions become hedged.

### 5. Category drift toward fine-grained but less stable event labels

The model prefers narrower or flashier labels even when the reviewed baseline intentionally uses broader first-pass types.

Examples:

- `narrated_diminishment` -> `ridicule`, `humiliation`, `blame`
- `snub` -> `exclusion`, `contempt`
- `admiration` + `discredit_association` -> several layered `admiration` / `deference` / `blame` events

Effect:

- The output becomes harder to compare across passages.
- The first-pass schema loses its intentionally coarse calibration.

### 6. Status-effect proliferation

The model often emits several dimensions per passage, including dimensions not used in the reviewed baseline.

Frequent extra dimensions:

- `rhetorical_position`
- `emotional_position`
- extra `inclusion_exclusion`
- zero-sum or balancing effects not needed for the first pass

Examples:

- `v1-p1-combray#p-17`: adds `general_appraisal` and `inclusion_exclusion`
- `v1-p1-combray#p-274-p-275`: adds `rhetorical_position`, `emotional_position`, `inclusion_exclusion`
- `v1-p1-combray#p-312-p-313`: adds `emotional_position` for Swann and two effects for Vinteuil

Effect:

- The model treats status dimensions as an invitation to exhaust possibilities.
- Reviewed “dominant effect only” discipline is lost.

### 7. Ambiguity inflation

The model often emits long ambiguity lists even when the reviewed baseline treats the passage as stable enough for a clean first-pass annotation.

Examples:

- `v1-p1-combray#p-17`
- `v1-p1-combray#p-274-p-275`
- `v1-p1-combray#p-312-p-313`

Effect:

- Output quality looks more cautious than it actually is.
- Many ambiguities are interpretive caveats rather than true blockers.

## Prompt implications for run-004

The next prompt revision should push much harder on these constraints:

1. Prefer the smallest sufficient reading.
   - Default to `1` main event per unit.
   - Use `2` only when the passage clearly contains two distinct, non-redundant movements.

2. Annotate the dominant local movement, not every evaluative trace.
   - Do not split one narrated movement into event fragments like ridicule + blame + humiliation unless the passage clearly stages them separately.

3. Keep `characters_present` focal.
   - Include only characters materially involved in the dominant appraisal/status movement.
   - Do not add peripheral parents/family members unless the event centrally targets or sources them.

4. Use broad first-pass labels.
   - Prefer `narrated_diminishment`, `narrated_elevation`, `snub`, `admiration`, `blame`, `prestige_association`, `discredit_association`.
   - Avoid finer-grained labels unless the passage truly demands them.

5. Emit only the dominant status effects.
   - Usually `1`.
   - Occasionally `2`.
   - Avoid balancing every positive with a negative unless both are central to the passage.

6. Use `ambiguities` sparingly.
   - Empty list by default.
   - Add an ambiguity only when it materially changes how the event or status effect should be read.

7. Treat narrator framing as decisive when it clearly is.
   - Do not default to `neutral_report` when the narrator is obviously steering the appraisal.

## Recommended evaluation focus for the next automated run

When reviewing `run-004`, check these first:

- event count per unit
- whether extra peripheral characters appear
- whether narrator-led passages are still weakened into `neutral_report`
- whether status effects stay limited to dominant dimensions
- whether `ambiguities` remain mostly empty
