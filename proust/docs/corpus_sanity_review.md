# Corpus Sanity Review

This document records the first corpus-level sanity review over the accepted exploratory annotation corpus.

It is meant to answer one question:

- does the accepted corpus still look literarily and operationally sane once the runs are aggregated?

The full machine-readable review artifact for this pass is:

- [corpus-review-001.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-001.json:1)

## Reviewed corpus

This review covered the currently accepted larger-scale corpus:

- the accepted `v3-p1` scaling sequence through `run-209`
- the completed widened-aperture `v4-p2` sequence through `run-240`

Operationally, that means:

- `36` accepted runs
- `186` declared units
- `186` valid annotations

## High-level result

The corpus passes this first sanity review.

Current judgment:

- no hidden large-scale distortion became visible only after aggregation
- the three lenses disagree occasionally, but not in a way that suggests systemic instability
- the strongest recurring surfaces remain text-plausible rather than obviously pathological
- the main watchpoint is not corpus corruption but focal narrowness:
  - many runs average close to one scored character per unit

That is a real thing to keep watching, but it is not yet strong evidence that the stack is failing.

## Headline metrics

Aggregate annotation shape:

- event polarity counts:
  - positive: `64`
  - negative: `114`
  - mixed: `2`
- most common event types:
  - `narrated_diminishment`: `86`
  - `narrated_elevation`: `44`
  - `admiration`: `18`
  - `snub`: `17`

Cross-lens stability:

- comparable unit-character entries across all three lenses: `196`
- label disagreement count: `8`
- label disagreement rate: `0.041`
- direction disagreement count: `6`
- direction disagreement rate: `0.031`
- positive-versus-negative sign-flip cases: `0`

Surface narrowness:

- average run-level average of scored characters per scored unit: `1.089`
- runs at or below `1.0` characters per scored unit: `27` of `36`
- runs at or below `1.25` characters per scored unit: `32` of `36`
- highest average characters per scored unit: `1.667` in:
  - `run-209`
  - `run-219`

## What looks sane

### 1. Lens disagreement stays bounded

The disagreement cases are mostly threshold or mixedness cases, not outright reversals.

Representative examples:

- `run-211`, `M. de Vaugoubert`:
  - `neutral` in local
  - `mixed` in prestige and inclusion
- `run-238`, `baron de Charlus`:
  - `neutral` in local
  - `mixed` in prestige and inclusion
- `run-231`, `Robert de Saint-Loup`:
  - `win` in local
  - `mixed` in prestige and inclusion

What did **not** appear:

- cases where one lens read a character as a clear win and another read the same case as a clear loss

That is the most important cross-lens sanity signal in this review.

### 2. Extreme units look plausible rather than noisy

Strong positive units are dominated by figures and scenes that are not surprising at the literary level, including:

- `duchesse de Guermantes`
- `baron de Charlus`
- `Odette`
- `docteur Cottard`
- `Robert de Saint-Loup`

Strong negative units are likewise plausible and socially legible, including:

- `marquise de Saint-Euverte`
- `Mme de Villeparisis`
- `le père du narrateur`
- `baron de Charlus`
- `Mme de Cambremer`

Nothing in the extremes suggests random target assignment or meaningless score inflation.

### 3. Corpus-level character surfaces still read like literary-social material

Top positive and negative character totals by lens are not interchangeable.

Examples:

- `duchesse de Guermantes` is strongly positive across the corpus
- `baron de Charlus` is strongly negative in aggregate despite having real positive spikes
- `Swann`, `le directeur`, `Mme de Cambremer`, and `marquise de Saint-Euverte` read as notably negative in ways that match the accepted runs
- volatility concentrates in characters where oscillation is actually plausible:
  - `baron de Charlus`
  - `Mme de Villeparisis`
  - `Odette`
  - `duc de Guermantes`
  - `Robert de Saint-Loup`

That pattern looks like literary variability, not mechanical instability.

## Main watchpoint

The main thing this review surfaced is:

- the corpus is often very focal and narrow

Many accepted runs have roughly one scored character per unit.

This may still be acceptable because:

- many units were deliberately chosen or accepted as dominant-movement windows
- the report surfaces remain interpretable
- the narrowness does not coincide with large cross-lens disagreement or obvious garbage extremes

But it does matter because it could mean:

- the stack is systematically under-registering secondary characters
- some passage types are being compressed too aggressively into one focal target
- future broad social scenes may stress the current reduction discipline differently

Current judgment on this issue:

- watch closely
- do not treat it as a blocker yet

## Conclusion

The accepted corpus currently passes a first corpus-level sanity check.

The main conclusions are:

- the corpus looks usable enough to justify the next validation phase
- the hidden-risk picture is better than feared:
  - no corpus-level sign-flip problem
  - low disagreement across lenses
  - plausible extremes
- the main remaining concern is bounded and intelligible:
  - focal narrowness, not generalized corruption

## Implication for the checklist

This review completes the checklist item:

- corpus-level sanity proof

It does **not** by itself justify full-corpus automation.

The remaining major evidence still to gather is:

- additional contrasting terrain-transfer proof
- an adverse-case stress pack
- an unattended production-style dry run

## Final Production-Corpus Review

The final aggregation review over the accepted contiguous production pass is now complete.

Machine-readable artifact:

- [corpus-review-final.json](/Users/nathan_brixius/dev/proust/outputs/corpus-review-final.json:1)

Reviewed corpus:

- `run-285` through `run-556`, including only output runs with annotation JSON files
- `136` accepted production output runs
- `1000` declared units
- `1000` valid annotations

Headline metrics:

- event polarity counts:
  - positive: `393`
  - negative: `596`
  - mixed: `22`
- cross-lens comparable entries: `1165`
- label disagreement rate: `0.065`
- direction disagreement rate: `0.045`
- positive-versus-negative sign-flip examples: `0`

Final judgment:

- the completed production corpus passes the corpus-level sanity check
- the aggregate positive and negative character surfaces remain literarily plausible
- cross-lens disagreement remains bounded and does not show systemic inversion
- focal narrowness remains the main interpretation caveat, not evidence of corpus corruption
