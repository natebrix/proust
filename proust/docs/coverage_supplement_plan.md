# Coverage Supplement Plan

This document defines the additive coverage pass that widens per-unit character coverage over the accepted canonicalized corpus.

Its purpose is to fix, at the source, the match starvation identified in the character ELO assessment: `1684` accepted units currently yield only `276` pairwise `advantage` matches because the first-pass prompt deliberately scored only the focal characters of each unit, and the narrator was never a scored participant.

## Reviewed decisions

These were decided explicitly before work began (2026-08-05):

1. **The narrator becomes a scored participant.** `le narrateur` (the in-scene self, not the narrating voice) may receive status effects in supplement annotations. This is a deliberate extension of the original named-characters-only ontology. The voice/participant distinction is defined in the supplement prompt.
2. **Annotator: Claude Sonnet subagents.** The first pass was annotated by `gpt-5` through the OpenAI automation pipeline. Supplements are produced by Claude Sonnet agents. Because supplement scores and first-pass scores meet inside the same unit (which is exactly where ELO matches are made), a calibration batch quantifies cross-model directional agreement before the mass pass.
3. **Scope: all flagged units.** Every unit the coverage audit flags with at least one material candidate enters the queue. Coverage is complete, not sampled.

## Core principles

* **The accepted corpus stays byte-for-byte untouched.** No accepted annotation JSON, run directory, or current aggregate artifact is modified.
* **Supplements are a separate artifact family**: `outputs/supplement-run-*`, schema version `annotation_supplement_v1`, same run-directory layout (`run.json`, `units/`, `prompts/`, `raw/`, `annotations/`) so existing status/score/report tooling applies.
* **Supplements only add characters.** A supplement annotation may never score a character already scored in the accepted annotation for the same unit. A validator rule enforces zero overlap.
* **Merging happens at analysis time**, behind an explicit option on the aggregate builders. Default behavior of all existing surfaces is unchanged; supplemented surfaces are new artifacts, clearly labeled.

## Pipeline

1. **Coverage audit** (`coverage_audit_v1`, mechanical, no LLM)
   - `python -m proust coverage-audit --discover-runs outputs`
   - surface-form matching of run alias maps (translated to accepted canonical identities) against unit `raw_text`, plus a first-person-marker heuristic for narrator presence
   - artifacts: `outputs/coverage-audit-current.{json,md}`
   - produces the supplement work queue and projected match gains
2. **Supplement contract**
   - prompt: `proust/prompts/supplement_prompt.md` — first-pass schema and interpretive rules, focality inverted: score only listed candidate characters, empty results valid and expected, narrator voice/participant ruleset
   - supplement event ids use the `S` prefix to remain distinguishable from first-pass `E` ids
3. **Pilot** — ~20 units spanning chapter types (salon scene, intimate two-person, third-person Swann material, essayistic), human-reviewed before anything scales
4. **Calibration** — Sonnet blind-rescores ~30 already-scored characters (first-pass prompt, no accepted annotation shown); compare directional agreement of `advantage` net scores against accepted gpt-5 values. Gate: proceed to mass pass only if report-level directional agreement is high; record the number here when known.
5. **Mass pass** — batched Sonnet annotation over the full queue, structured-output enforced, mechanical validate/score/report per batch with the existing review gates (parse errors, validation errors, cross-lens sign flips, mixed-unit threshold), resumable via on-disk manifests
6. **Merge + ELO v2** — supplemented overlay surface behind an explicit include-supplements option; regenerate character ELO on the merged surface as a new artifact; before/after diff on match counts, characters clearing ~30 matches, and rating stability of the current top and bottom cohorts; minimum-match filters added to presentation tables

## Review rule for supplement batches

Reuse the production review discipline:

1. validate and score each batch mechanically
2. read the three lens reports first
3. inspect units only on genuinely surprising signals

Supplement-specific stop conditions:

* any overlap between supplement characters and accepted characters in the same unit
* supplement events that reverse the direction of an accepted event for the same interaction
* narrator scored in clearly essayistic or third-person material
* candidate acceptance rate far above expectation (most candidates are peripheral; a high acceptance rate signals materiality drift, since the audit list is a mechanical screen, not a quota)

## Known risks

* **Cross-model provenance.** First-pass and supplement scores come from different annotator models. Mitigations: identical schema and scoring weights, calibration gate, and provenance retained by keeping supplements in their own artifact family.
* **Ontology change.** Scoring the narrator changes the character universe of merged surfaces. Merged artifacts must be labeled as supplemented; unmerged artifacts remain the default.
* **Anchoring.** Supplement annotators see the accepted annotation. This is intentional (prevents contradiction and duplication) but may bias supplements toward the accepted framing. The calibration batch, which hides the accepted annotation, bounds this.
* **Audit recall limits.** Surface matching misses pronoun-only presence; the narrator heuristic over-triggers on quoted first-person speech. The audit is a screen; the annotator judges materiality; acceptance-rate monitoring catches systematic drift.

## Ambiguity normalization rule

The production reducer keeps at most one ambiguity, and only when an event carries an
`uncertain` stance; the accepted corpus is post-reduction. The scorer charges every
character in a unit `0.4` per ambiguity flag. Supplement annotations are therefore
normalized with the same rule (`normalize_supplement_ambiguities` in
`proust/supplement.py`) before validation and writing. Full structural reduction is
deliberately NOT applied to supplements: the reducer's dominant-movement compression
assumes it is picking the unit's focal movement, which the accepted annotation already
owns; applying it would drop legitimate secondary-participant events.

Calibration evidence (2026-08-06, 15 units / 30 characters, Sonnet blind rescore vs
accepted gpt-5, advantage lens): before normalization the raw comparison showed a
mechanical negative skew (mean signed delta ≈ -0.4-per-flag artifacts, 5 apparent sign
flips). After applying the same ambiguity rule to both sides: direction agreement
`19/30`, hard sign flips `2` (both on genuinely contested passages: Norpois's
demolition of Bergotte in `v2-p1#p-241-p-250`, and the Charlus-Jupien encounter in
`v4-p1#p-6-p-10`, which is the consummation-vs-diminishment edge case the first-pass
prompt itself legislates), mean signed bias `-0.045` (effectively unbiased), mean
`|net delta|` `0.720`.

## Current status

- decisions reviewed and recorded (this document)
- supplement prompt written: `proust/prompts/supplement_prompt.md`
- coverage audit complete: `outputs/coverage-audit-current.{json,md}` — `1161` units,
  `1116` flagged, `5454` candidate additions, `738` narrator-candidate units after
  quote-aware narrator counting
- supplement run machinery complete: `proust/supplement.py`, `prepare-supplements` CLI
- pilot complete: `outputs/supplement-run-001` (20 units, 20 valid annotations,
  0 validation rejections after the known-source validator fix, 0 direction reversals,
  6 empty supplements, narrator scored in 8 units and correctly absent from
  third-person material), review surface: `outputs/supplement-pilot-review.md`
- calibration complete: gate PASSED with the caveats recorded above
- mass pass complete (2026-08-06): `1124` supplement units across
  `supplement-run-001` .. `supplement-run-029`, all 28 mass batches gated clean;
  gate trips during the pass hardened the writer three times (alias-roster event
  sources, accent-variant name reconciliation via the extended merge table,
  untracked one-off sources mapped to `unknown`) plus one out-of-band single-unit
  recovery; candidate acceptance held at `0.27`-`0.29` throughout, `0` direction
  reversals corpus-wide, `0` narrator-in-third-person violations
- merge + ELO v2 complete: supplemented artifacts generated as
  `character-elo-advantage-supplemented-current.{json,md}`,
  `character-elo-advantage-timeline-supplemented-current.{json,md}`, and
  `character-elo-supplement-diff-current.{json,md}`; match count `276 -> 1801`,
  character count `60 -> 70`, `24` characters newly clear the 30-match
  reliability threshold; unsupplemented default artifacts unchanged
- merge + supplemented ELO surfaces implemented: `build_chapter_overlay_data`,
  `build_character_elo`, and `build_character_elo_timeline` all accept an optional
  `supplement_run_dirs` parameter (see "Merge semantics" below); `elo-supplement-diff`
  CLI command added

## Merge semantics

Implemented in `proust/app_exports.py` (`discover_supplement_run_dirs`,
`build_chapter_overlay_data`, `build_character_elo`, `build_character_elo_timeline`,
`build_character_elo_supplement_diff`) and wired into `proust/cli.py`.

* `discover_supplement_run_dirs(outputs_dir="outputs")` returns the sorted list of
  `outputs/supplement-run-*` directories that have a `run.json`. Unlike
  `discover_annotation_run_dirs`, it does not raise if the outputs directory is
  missing or if no supplement runs are found yet -- it returns `[]`, since "no
  supplements exist yet" is a normal, non-error state during the mass pass.
* `build_chapter_overlay_data(run_dirs, supplement_run_dirs=None)` scores each
  supplement run with the exact same per-lens `runner.build_outcome_report` path used
  for accepted `run_dirs`. Supplement duplicate units resolve with the same
  "latest reviewed run wins" rule as accepted runs, scoped to the supplement family
  only (a supplement run never contests an accepted run's precedence, and vice versa).
  For each unit that already has an accepted row: supplement characters absent from
  that unit's accepted roster are appended with `"provenance": "supplement"`; accepted
  rows are never mutated and carry no `provenance` key (its absence means accepted).
  A name collision (a supplement run scoring a character the accepted annotation
  already scored for that unit -- which the supplement write-time validator should
  already prevent) is skipped and counted, never applied. When
  `supplement_run_dirs` is passed and non-empty, the returned dataset gains
  `supplement_run_count`, `supplement_runs` (the winning run ids), `supplemented_unit_count`
  (units that actually gained at least one supplement row), and
  `supplement_collision_count`. The default call (`supplement_run_dirs` omitted or
  `None`) is byte-identical to the pre-supplement behavior.
* `build_character_elo` / `build_character_elo_timeline` take the same
  `supplement_run_dirs` parameter and pass it straight through to the overlay
  builder; when supplements are included, the top-level result gains
  `"supplemented": true` and `"supplement_runs"`.
* `build_character_elo` also takes `min_match_count` (default `10`), a presentation
  filter applied only to `top_rated_characters`, `lowest_rated_characters`, and
  `largest_rank_mismatches` -- never to the full `characters` table, and never to
  `highest_match_count_characters`. This applies to the non-supplemented path too: it
  fixes a pre-existing defect where characters with 0 pairwise matches (e.g. scored
  alone in every unit they appear in) could still show up in the mismatch table.
* `build_character_elo_supplement_diff(baseline_analysis, supplemented_analysis,
  clearing_threshold=30)` computes match/character/draw-rate before-and-after,
  characters newly clearing `clearing_threshold` matches, and the top 15 rating
  movers. A character with no baseline row (e.g. `le narrateur` in units where it was
  never scored pre-supplement) reports `elo_before: null` and `delta: null`; it is
  still ranked among the movers using its movement from the shared initial rating,
  since it did move, just without a prior elo of its own.
* New artifact names when supplemented: `outputs/character-elo-advantage-supplemented-current.{json,md}`
  and `outputs/character-elo-advantage-timeline-supplemented-current.{json,md}`. The CLI's
  `--include-supplements` flag switches the default `--output`/`--markdown-output` paths to
  these names (an explicit `--output`/`--markdown-output` always wins); it never touches the
  unsuffixed `-current.*` files. `elo-supplement-diff` computes both the baseline and
  supplemented analyses in one invocation and writes the supplemented ELO, supplemented ELO
  timeline, and `outputs/character-elo-supplement-diff-current.{json,md}` artifacts -- it
  never writes the unsuffixed baseline files, since those already exist and must stay
  untouched.
