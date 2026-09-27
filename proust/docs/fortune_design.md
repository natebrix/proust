# Fortune arcs design

Status: **PROTOTYPE** (2026-09-27). Code in `proust/fortune.py`, build in
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

## Checks

- Arc evidence: log-likelihood of the fitted model minus one with q → 0.
  overall 32.8, advantage 11.9, prestige 7.7, inclusion 0.5 (no arcs visible).
- Order permutation test per character: shuffle the outcomes across the
  character's appearance times; p = share of shuffles with an equal or bigger
  fall (or rise). About 40 characters are tested per lens, so values near
  0.05 are expected by chance; values near 0.01 are the dependable ones.
- `tests/test_fortune.py` checks the smoother and likelihood against a dense
  Gaussian posterior.

## First findings (overall lens, enrichment corpus)

- Charlus: the largest fall in every lens that sees arcs; +0.58 ± 0.31
  (Jeunes Filles) → −1.59 ± 0.33 (Matinée), p = 0.003.
- Clear by the order test (p ≤ 0.01): falls of Charlus and Albertine;
  rises of M. Vinteuil (posthumous vindication), Mme Bontemps and the
  princesse de Parme.
- Borderline (p 0.02–0.05): falls of Swann, the duchesse, the narrator, the
  duc, la Berma; rises of Morel and the grandmother.
- Not supported: Mme Verdurin's rise (p = 0.13), partly because the name view
  credits her late passages to "princesse de Guermantes".
- The narrator's vocation in L'Adoration perpétuelle is invisible: the two
  revelation passages have empty annotations, because the schema has no
  entry for an inner revelation.

## Open questions

- Person view (and a Verdurin-as-princesse era mapping) so arcs follow people
  across titles.
- Whether to show fortune on an Elo-like display scale for the "fun" framing.
- Tie-in with WHR: fortune falling while head-to-head standing holds would
  mark a character sinking with their whole milieu.
