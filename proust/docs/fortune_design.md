# Fortune arcs design

Status: **PROTOTYPE** (2026-09-27; person view and rating scale 2026-09-28). Code in `proust/fortune.py`, build in
`scripts/build_fortune.py`, artifacts in `outputs/fortune/`.

## Why a second score

Scoring v2's WHR answers a relative question: who comes out ahead of whom
inside a passage. A relative score moves only as far as a character's
opponents allow. Its drift parameter is also chosen by one-step prediction,
which rewards ratings that stand still, and it landed on the smallest
candidate (w2 = 5) in every lens. The result is that Charlus's fall shows up
as about 37 Elo, while his raw per-passage movements go from +0.60 (V2) to
−0.98 (V5) and −0.95 (V7).

Fortune answers the other question: what is happening to this character
across the novel. It needs no opponents.

## Model

Local level (random walk plus noise) per character and lens:

    x(t') = x(t) + Normal(0, q · (t' − t))
    y(u)  = x(t_u) + Normal(0, sigma² / w(u))

- `y(u)`: the character's scoring v2 movement in passage u. Only lens
  participants are observed; presence without an effect is silence.
- `t_u`: midpoint of the passage's word span, in 10,000-word units.
- `w(u)`: κ · ρ, the same confidence and ambiguity weights v2 uses for
  comparisons. Doubt makes an observation noisier, never smaller.
- The `overall` lens sums the three v2 lenses' movements.
- Prior: mean = the lens-wide mean movement, sd = 0.5.

Kalman filter plus RTS smoother. `q` and `sigma²` are chosen per lens by
pooled marginal likelihood over every character, never per character.

## Person view

`PersonKeyer` keys each annotation name to a registry entity, applying
`person_view_merge` links (le peintre → Elstir, prince des Laumes → duc de
Guermantes). Unresolved names key on themselves (`name:<name>`) and never pool.
Two names for one person in the same passage become one observation (movements
add, weights average).

It adds one rule the registry intends but `Registry.resolve` does not apply:
if another entity has a `chapters:`-scoped form with the same text in this
chapter, the name is ambiguous there and stays on its name key.
`Registry.resolve` answers exact annotation names first, so in the Matinée and
the Bal de têtes "princesse de Guermantes" always resolves to Marie-Gilbert,
contrary to the ruling in `characters.yaml`. Fortune does not change
`resolve` (scoring v2's person view depends on it); it works around it.

`REVIEWED_UNIT_RESOLUTIONS` settles ambiguous passages one at a time, each
with its reason. There is one today: Bal de têtes p. 61–65 ("nous ferons
clan!", dentures) → Mme Verdurin. It is the only late passage that uses the
bare title.

Both views are built: `fortune-<lens>-person.json` and
`fortune-<lens>-name.json`. The report reads the person view.

## Rating scale

`fortune_rating(level) = 1500 + k · level`, with k from the fitted passage
noise: two characters' next passages differ by Normal(a − b, 2σ²), so
P(A's passage is better) = Φ((a − b)/√(2σ²)). Matching that to Elo's
logistic (Φ(z) ≈ logistic(1.702 z)) gives k = (400/ln 10) · 1.702 / √(2σ²):
about 220 points per unit in overall and inclusion (σ² = 0.9) and 250 in
advantage and prestige (σ² = 0.7). A gap of D points then means what it means
in Elo. 1500 is a level of 0: the passages leave the character where they
were. Scales differ by lens, so compare ratings within a lens.

## Checks

- Arc evidence: log-likelihood of the fitted model minus one with q → 0.
  overall 32.4 (person view; 32.8 name view), advantage 11.9, prestige 7.7,
  inclusion 0.5 (no arcs visible).
- Order permutation test per character: shuffle the outcomes across the
  character's appearance times (1000 shuffles); p = share of shuffles with an equal or bigger
  fall (or rise). About 40 characters are tested per lens, so values near
  0.05 are expected by chance; values near 0.01 are the dependable ones.
- `tests/test_fortune.py` checks the smoother and likelihood against a dense
  Gaussian posterior.

## First findings (person view, overall lens unless noted)

- Charlus: the largest fall in every lens that sees arcs; 1627 ± 68
  (Jeunes Filles) → 1149 ± 73 (Matinée), 479 points, p = 0.001.
- Clear by the order test (p ≤ 0.01, move ≥ 0.25): falls of Charlus,
  Albertine, the princesse de Guermantes (Marie-Gilbert alone) and
  Mme Cottard; rises of M. Vinteuil (posthumous vindication) and
  Mme Bontemps. The princesse de Parme's rise is non-random (p = 0.001) but
  small (94 points). Advantage adds Gilberte's fall (p = 0.002); prestige
  keeps Charlus and the duchesse.
- Borderline (p 0.02–0.05): falls of Swann, the duchesse, the narrator;
  rises of Morel and the grandmother. la Berma sits at 0.05.
- Mme Verdurin: the rise is not supported (p = 0.12) even with her Bal de
  têtes passage restored. Her shape is: peak in wartime Paris (1509), then
  every Bal de têtes passage negative — the title arrives as mockery.
- The narrator's vocation in L'Adoration perpétuelle is invisible: the two
  revelation passages have empty annotations, because the schema has no
  entry for an inner revelation.

## Open questions

- `characters.yaml` has `prince-de-guermantes` (proposed, overlay) and
  `prince-de-guermantes-2` (confirmed, from annotations): one person, two
  entities. Harmless for fortune today (one resolves every time), but worth
  merging.
- Whether `Registry.resolve` itself should honor chapter-scoped overlaps.
  That would change scoring v2's person view, so it is a separate decision.
- Tie-in with WHR: fortune falling while head-to-head standing holds would
  mark a character sinking with their whole milieu.
