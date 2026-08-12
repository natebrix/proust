# Scoring v2 Design

Status: design pass for review (2026-08-12). Decision context: the v1 scoring
formula predates the foundation corpus. Its hand-tuned event/status weight
tables were calibrated informally against a sparse closed-world corpus, it
double-counts (events and the status effects derived from them both add to the
score), and its ambiguity penalty — dormant on legacy data — misfires
structurally on open-world rosters. Backwards compatibility is explicitly
waived: the foundation corpus takes the principled path.

## What scoring must produce

1. **Comparisons** — per unit, per lens, an ordering (with ties) over the
   participating characters. This is the primary object: the rating layer
   (WHR) consumes only comparisons.
2. **Unit readings** — a per-character, per-lens movement with a magnitude and
   a label, for reader-facing passage views.
3. **Profiles** — per-character aggregates for dossiers, standings, chapters.

## Principles

- **The annotator already judged; scoring should not re-judge.** Status
  effects carry the annotator's calibrated delta (−2..+2), dimension, and
  confidence. Events are the *evidence lineage* for effects (`based_on_events`)
  — counting them again as additive terms double-counts one reading.
- **Uncertainty weighs, never subtracts.** Ambiguity notes and uncertain
  stances reduce how much a comparison counts; they never push a score in a
  direction. (This retires the v1 ambiguity penalty.)
- **Lenses should be orthogonal by construction.** v1's blended weight tables
  made every lens partly every other lens; divergence between lenses — the
  project's best insights (Odette, Villeparisis) — is sharpest when the lenses
  partition the annotation's dimensions.
- **Frequency must not masquerade as strength.** v1 profile sums grew with
  appearance count. Aggregates separate intensity (mean movement per
  appearance) from standing (rating), never summing raw nets.

## The formula

### 1. Unit movement

For character c in unit u under lens ℓ:

    m(c, u, ℓ) = Σ over status effects e of c in u:
                   W_ℓ(dimension(e)) · delta(e) · confidence(e)

The proposed lens projection W is a **partition** of the five dimensions:

| dimension | lens |
| --- | --- |
| social_status | prestige |
| inclusion_exclusion | inclusion |
| general_appraisal | advantage |
| emotional_position | advantage |
| rhetorical_position | advantage |

(Within advantage the three dimensions carry weights 1.0 / 0.8 / 0.6
respectively — the situational core outranks its rhetorical edge.) Events
contribute no additive term.

### 2. Comparisons

Within unit u, lens ℓ, for each unordered pair (a, b) of scored characters:

    outcome: a beats b if m_a − m_b > τ; b beats a if < −τ; else draw
    τ = 0.25  (same tie-band spirit as v1; deltas are integer-scaled, so the
               scale is unchanged)

    weight:  w(a, b, u) = ρ(u) · min(κ_a, κ_b)
    ρ(u) = max(0.5, 0.8^A)  where A = number of ambiguity notes in u
    κ_c  = mean confidence of c's effects in u, additionally multiplied by
           0.7 if any supporting event for c carries stance "uncertain";
           κ_c = presence_confidence for zero-effect characters

A comparison is a weighted game: uncertain readings count for less, uniformly
for both characters — direction is never touched.

### 3. Ratings

WHR per lens over the weighted comparisons (Wiener prior over narrative time,
smoothed + filtered modes, exactly the existing machinery extended to accept
game weights; a draw of weight w contributes w/2 to each side, generalizing
the current half-win convention). Standings rank by conservative rating
(rating − band) among non-provisional characters, as now. ELO and Glicko-2
remain as predictive baselines in validation reports only.

### 4. Display layer

- **Unit label** per character×lens: positive (m > τ), negative (m < −τ),
  mixed **only** when the character has both positive and negative effects in
  that lens within the unit (genuine internal conflict — never a penalty
  artifact), else neutral.
- **Dossier lens cards**: rating rank + percentile (standing), mean |m| per
  appearing unit (intensity), signed mean m (direction), dominant dimension
  (trivial for prestige/inclusion; for advantage, the largest-|contribution|
  of its three dimensions), band.
- **Archetype**: sign triple of (rating − 1500) per lens — consistent with
  the standings the reader sees, not with frequency-confounded sums.

### 5. Aggregation keys

Aggregation keys on registry `entity_id` with the era ledger, enabling
person-view and name-view standings (registry design step 4). The reader-
facing person/name toggle is UI scope, deferred; the data carries both from
day one.

## Validation before adoption (the gate)

v2 cannot be validated by agreement with v1 (v1 is what it replaces). The
gate is coherence, stability, and face validity, reported side by side:

1. **Lens orthogonality**: cross-lens rating correlations should fall vs v1
   (sharper lenses = more information per lens).
2. **Stability**: bootstrap over units — rating rank variance for the
   non-provisional set; v2 should be no less stable than v1.
3. **Predictive sanity**: WHR-filtered one-step log-loss on v2 comparisons,
   with ELO/Glicko baselines, reported (not gated — comparisons differ from
   v1's so cross-formula numbers are not comparable).
4. **Literary panel** (fixed before running, judged after): the duchesse's
   standing among the corpus elite; Rachel ranked; Bloch's inclusion near the
   bottom; Odette's prestige > her inclusion; Charlus's trajectory declining
   across the late volumes; the narrator mid-table with a tight band; Saniette
   last or near it; l'amie de Mlle Vinteuil present.

Adoption is a reviewed decision on that report.

## Explicitly retired from v1

- Event-type weight tables and stance multipliers as score terms (stance
  survives as an uncertainty signal in κ; events survive as evidence lineage).
- The ambiguity penalty (uncertainty now weighs comparisons).
- Score-sum profiles and sum-based archetypes.
- The win/loss ±0.75 thresholds (labels now come from τ and effect-sign
  structure).

## Deferred, recorded

- Davidson-style tie modeling in WHR (half-win draws retained for now).
- Person/name-view UI toggle.
- Editorial rewrite and app sync: only after the adoption gate.
