# Scoring v2 sample walkthrough

Per unit: the annotator's effects verbatim, the v2 movement arithmetic, labels, and weighted comparisons. v1 nets shown for contrast where the scorer exposes them.

## v1-p1-combray#p-101-p-105
*The Legrandin garden scene: basic arithmetic check*  (run: foundation-run-001)

- ambiguity notes: 1 -> comparison weight factor rho = 0.80

**Effects as annotated:**
- Legrandin: general_appraisal -1 (confidence 0.8) — Legrandin's local standing is somewhat lowered by the exposed gap between his own family's noble marriage and 
- le narrateur: general_appraisal +1 (confidence 0.72) — The narrator is locally elevated by Legrandin's direct, warm praise of his soul and artistic sensibility.

**advantage**: Legrandin = -0.80 [negative], la grand-mère = +0.00 [neutral], le narrateur = +0.72 [positive]
  - Legrandin vs la grand-mère: la grand-mère wins (m -0.80 vs +0.00), weight 0.64
  - Legrandin vs le narrateur: le narrateur wins (m -0.80 vs +0.72), weight 0.58
  - la grand-mère vs le narrateur: le narrateur wins (m +0.00 vs +0.72), weight 0.58
**prestige**: Legrandin = +0.00 [neutral], la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 3 draws among characters with no relative movement)
**inclusion**: Legrandin = +0.00 [neutral], la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 3 draws among characters with no relative movement)

## v3-p2#p-91-p-95
*The grandmother's death: advantage internal weights, solemn passage*  (run: foundation-run-017)

- ambiguity notes: 1 -> comparison weight factor rho = 0.80

**Effects as annotated:**
- la grand-mère: general_appraisal +2 (confidence 0.85) — The passage elevates the grandmother locally by stripping away the marks of illness and age in the narrator's 
- le narrateur: emotional_position +2 (confidence 0.78) — The narrator locally gains emotional renewal, describing himself as reborn with existence restored to him, rat

**advantage**: la grand-mère = +1.70 [positive], le narrateur = +1.25 [positive]
  - la grand-mère vs le narrateur: la grand-mère wins (m +1.70 vs +1.25), weight 0.62
**prestige**: la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 1 draws among characters with no relative movement)
**inclusion**: la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 1 draws among characters with no relative movement)

## v2-p2-noms-de-pays-le-pays#p-101-p-105
*Villeparisis at Balbec: the adjudicated prestige/inclusion divergence*  (run: foundation-run-009)

- ambiguity notes: 1 -> comparison weight factor rho = 0.80

**Effects as annotated:**
- Mme de Villeparisis: social_status +1 (confidence 0.78) — The princesse de Luxembourg's visit authenticates her rank in the hotel's eyes and silences the campaign to ex
- Mme de Villeparisis: inclusion_exclusion -1 (confidence 0.72) — For most of the passage she is the object of hostile scrutiny by the hotel's bourgeois group, held outside the

**advantage**: Mme de Villeparisis = +0.00 [neutral], princesse de Luxembourg = +0.00 [neutral]
  - (+ 1 draws among characters with no relative movement)
**prestige**: Mme de Villeparisis = +0.78 [positive], princesse de Luxembourg = +0.00 [neutral]
  - Mme de Villeparisis vs princesse de Luxembourg: Mme de Villeparisis wins (m +0.78 vs +0.00), weight 0.62
**inclusion**: Mme de Villeparisis = -0.72 [negative], princesse de Luxembourg = +0.00 [neutral]
  - Mme de Villeparisis vs princesse de Luxembourg: princesse de Luxembourg wins (m -0.72 vs +0.00), weight 0.58

## v4-p2#p-406-p-410
*The 17-character Verdurin salon: ambiguity note weighs, never pushes*  (run: foundation-run-024)

- ambiguity notes: 1 -> comparison weight factor rho = 0.80

**Effects as annotated:**
- Mme de Cambremer: social_status -2 (confidence 0.88) — The chemist's smile with which she anticipated introducing Charlus to Mme Féré turns into near-collapse when M
- marquis de Cambremer: social_status -1 (confidence 0.8) — He shares the rebuff and his retaliation — attending la Raspelière alone as though a princess of the blood con
- baron de Charlus: social_status +1 (confidence 0.75) — Both parties defer to him: the Cambremer's belated climb-down « fit sourire M. de Charlus en lui montrant son 
- Morel: general_appraisal -1 (confidence 0.78) — He wins the exchange practically but is lowered in the telling: he receives Charlus's crude genealogy « pieuse
- le narrateur: general_appraisal -1 (confidence 0.85) — In Bloch's eyes he is fixed as a snob who behaves differently before « des gens nés », and loses both the frie

**advantage**: Albertine = +0.00 [neutral], Bloch = +0.00 [neutral], Bloch père = +0.00 [neutral], Brichot = +0.00 [neutral], Legrandin = +0.00 [neutral], M. de Chevregny = +0.00 [neutral], M. de Crécy = +0.00 [neutral], Mme Féré = +0.00 [neutral], Mme Verdurin = +0.00 [neutral], Mme de Cambremer = +0.00 [neutral], Morel = -0.78 [negative], Robert de Saint-Loup = +0.00 [neutral], baron de Charlus = +0.00 [neutral], docteur Cottard = +0.00 [neutral], la marquise douairière de Cambremer = +0.00 [neutral], le narrateur = -0.85 [negative], marquis de Cambremer = +0.00 [neutral]
  - Albertine vs Morel: Albertine wins (m +0.00 vs -0.78), weight 0.62
  - Albertine vs le narrateur: Albertine wins (m +0.00 vs -0.85), weight 0.68
  - Bloch vs Morel: Bloch wins (m +0.00 vs -0.78), weight 0.62
  - Bloch vs le narrateur: Bloch wins (m +0.00 vs -0.85), weight 0.68
  - Bloch père vs Morel: Bloch père wins (m +0.00 vs -0.78), weight 0.62
  - Bloch père vs le narrateur: Bloch père wins (m +0.00 vs -0.85), weight 0.62
  - Brichot vs Morel: Brichot wins (m +0.00 vs -0.78), weight 0.62
  - Brichot vs le narrateur: Brichot wins (m +0.00 vs -0.85), weight 0.68
  - Legrandin vs Morel: Legrandin wins (m +0.00 vs -0.78), weight 0.62
  - Legrandin vs le narrateur: Legrandin wins (m +0.00 vs -0.85), weight 0.68
  - M. de Chevregny vs Morel: M. de Chevregny wins (m +0.00 vs -0.78), weight 0.62
  - M. de Chevregny vs le narrateur: M. de Chevregny wins (m +0.00 vs -0.85), weight 0.68
  - M. de Crécy vs Morel: M. de Crécy wins (m +0.00 vs -0.78), weight 0.62
  - M. de Crécy vs le narrateur: M. de Crécy wins (m +0.00 vs -0.85), weight 0.68
  - Mme Féré vs Morel: Mme Féré wins (m +0.00 vs -0.78), weight 0.62
  - Mme Féré vs le narrateur: Mme Féré wins (m +0.00 vs -0.85), weight 0.68
  - Mme Verdurin vs Morel: Mme Verdurin wins (m +0.00 vs -0.78), weight 0.62
  - Mme Verdurin vs le narrateur: Mme Verdurin wins (m +0.00 vs -0.85), weight 0.68
  - Mme de Cambremer vs Morel: Mme de Cambremer wins (m +0.00 vs -0.78), weight 0.62
  - Mme de Cambremer vs le narrateur: Mme de Cambremer wins (m +0.00 vs -0.85), weight 0.68
  - Morel vs Robert de Saint-Loup: Robert de Saint-Loup wins (m -0.78 vs +0.00), weight 0.62
  - Morel vs baron de Charlus: baron de Charlus wins (m -0.78 vs +0.00), weight 0.62
  - Morel vs docteur Cottard: docteur Cottard wins (m -0.78 vs +0.00), weight 0.62
  - Morel vs la marquise douairière de Cambremer: la marquise douairière de Cambremer wins (m -0.78 vs +0.00), weight 0.62
  - Morel vs marquis de Cambremer: marquis de Cambremer wins (m -0.78 vs +0.00), weight 0.62
  - Robert de Saint-Loup vs le narrateur: Robert de Saint-Loup wins (m +0.00 vs -0.85), weight 0.68
  - baron de Charlus vs le narrateur: baron de Charlus wins (m +0.00 vs -0.85), weight 0.68
  - docteur Cottard vs le narrateur: docteur Cottard wins (m +0.00 vs -0.85), weight 0.68
  - la marquise douairière de Cambremer vs le narrateur: la marquise douairière de Cambremer wins (m +0.00 vs -0.85), weight 0.68
  - le narrateur vs marquis de Cambremer: marquis de Cambremer wins (m -0.85 vs +0.00), weight 0.68
  - (+ 106 draws among characters with no relative movement)
**prestige**: Albertine = +0.00 [neutral], Bloch = +0.00 [neutral], Bloch père = +0.00 [neutral], Brichot = +0.00 [neutral], Legrandin = +0.00 [neutral], M. de Chevregny = +0.00 [neutral], M. de Crécy = +0.00 [neutral], Mme Féré = +0.00 [neutral], Mme Verdurin = +0.00 [neutral], Mme de Cambremer = -1.76 [negative], Morel = +0.00 [neutral], Robert de Saint-Loup = +0.00 [neutral], baron de Charlus = +0.75 [positive], docteur Cottard = +0.00 [neutral], la marquise douairière de Cambremer = +0.00 [neutral], le narrateur = +0.00 [neutral], marquis de Cambremer = -0.80 [negative]
  - Albertine vs Mme de Cambremer: Albertine wins (m +0.00 vs -1.76), weight 0.70
  - Albertine vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Albertine vs marquis de Cambremer: Albertine wins (m +0.00 vs -0.80), weight 0.64
  - Bloch vs Mme de Cambremer: Bloch wins (m +0.00 vs -1.76), weight 0.70
  - Bloch vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Bloch vs marquis de Cambremer: Bloch wins (m +0.00 vs -0.80), weight 0.64
  - Bloch père vs Mme de Cambremer: Bloch père wins (m +0.00 vs -1.76), weight 0.62
  - Bloch père vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Bloch père vs marquis de Cambremer: Bloch père wins (m +0.00 vs -0.80), weight 0.62
  - Brichot vs Mme de Cambremer: Brichot wins (m +0.00 vs -1.76), weight 0.70
  - Brichot vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Brichot vs marquis de Cambremer: Brichot wins (m +0.00 vs -0.80), weight 0.64
  - Legrandin vs Mme de Cambremer: Legrandin wins (m +0.00 vs -1.76), weight 0.70
  - Legrandin vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Legrandin vs marquis de Cambremer: Legrandin wins (m +0.00 vs -0.80), weight 0.64
  - M. de Chevregny vs Mme de Cambremer: M. de Chevregny wins (m +0.00 vs -1.76), weight 0.68
  - M. de Chevregny vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - M. de Chevregny vs marquis de Cambremer: M. de Chevregny wins (m +0.00 vs -0.80), weight 0.64
  - M. de Crécy vs Mme de Cambremer: M. de Crécy wins (m +0.00 vs -1.76), weight 0.70
  - M. de Crécy vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - M. de Crécy vs marquis de Cambremer: M. de Crécy wins (m +0.00 vs -0.80), weight 0.64
  - Mme Féré vs Mme de Cambremer: Mme Féré wins (m +0.00 vs -1.76), weight 0.70
  - Mme Féré vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Mme Féré vs marquis de Cambremer: Mme Féré wins (m +0.00 vs -0.80), weight 0.64
  - Mme Verdurin vs Mme de Cambremer: Mme Verdurin wins (m +0.00 vs -1.76), weight 0.70
  - Mme Verdurin vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Mme Verdurin vs marquis de Cambremer: Mme Verdurin wins (m +0.00 vs -0.80), weight 0.64
  - Mme de Cambremer vs Morel: Morel wins (m -1.76 vs +0.00), weight 0.70
  - Mme de Cambremer vs Robert de Saint-Loup: Robert de Saint-Loup wins (m -1.76 vs +0.00), weight 0.70
  - Mme de Cambremer vs baron de Charlus: baron de Charlus wins (m -1.76 vs +0.75), weight 0.60
  - Mme de Cambremer vs docteur Cottard: docteur Cottard wins (m -1.76 vs +0.00), weight 0.70
  - Mme de Cambremer vs la marquise douairière de Cambremer: la marquise douairière de Cambremer wins (m -1.76 vs +0.00), weight 0.70
  - Mme de Cambremer vs le narrateur: le narrateur wins (m -1.76 vs +0.00), weight 0.70
  - Mme de Cambremer vs marquis de Cambremer: marquis de Cambremer wins (m -1.76 vs -0.80), weight 0.64
  - Morel vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Morel vs marquis de Cambremer: Morel wins (m +0.00 vs -0.80), weight 0.64
  - Robert de Saint-Loup vs baron de Charlus: baron de Charlus wins (m +0.00 vs +0.75), weight 0.60
  - Robert de Saint-Loup vs marquis de Cambremer: Robert de Saint-Loup wins (m +0.00 vs -0.80), weight 0.64
  - baron de Charlus vs docteur Cottard: baron de Charlus wins (m +0.75 vs +0.00), weight 0.60
  - baron de Charlus vs la marquise douairière de Cambremer: baron de Charlus wins (m +0.75 vs +0.00), weight 0.60
  - baron de Charlus vs le narrateur: baron de Charlus wins (m +0.75 vs +0.00), weight 0.60
  - baron de Charlus vs marquis de Cambremer: baron de Charlus wins (m +0.75 vs -0.80), weight 0.60
  - docteur Cottard vs marquis de Cambremer: docteur Cottard wins (m +0.00 vs -0.80), weight 0.64
  - la marquise douairière de Cambremer vs marquis de Cambremer: la marquise douairière de Cambremer wins (m +0.00 vs -0.80), weight 0.64
  - le narrateur vs marquis de Cambremer: le narrateur wins (m +0.00 vs -0.80), weight 0.64
  - (+ 91 draws among characters with no relative movement)
**inclusion**: Albertine = +0.00 [neutral], Bloch = +0.00 [neutral], Bloch père = +0.00 [neutral], Brichot = +0.00 [neutral], Legrandin = +0.00 [neutral], M. de Chevregny = +0.00 [neutral], M. de Crécy = +0.00 [neutral], Mme Féré = +0.00 [neutral], Mme Verdurin = +0.00 [neutral], Mme de Cambremer = +0.00 [neutral], Morel = +0.00 [neutral], Robert de Saint-Loup = +0.00 [neutral], baron de Charlus = +0.00 [neutral], docteur Cottard = +0.00 [neutral], la marquise douairière de Cambremer = +0.00 [neutral], le narrateur = +0.00 [neutral], marquis de Cambremer = +0.00 [neutral]
  - (+ 136 draws among characters with no relative movement)

## v7-p4-le-bal-de-tetes#p-66-p-70
*Search hit for Rachel & Berma*  (run: foundation-run-034)

- ambiguity notes: 2 -> comparison weight factor rho = 0.64

**Effects as annotated:**
- Rachel: social_status +2 (confidence 0.93) — Rachel moves from remembered kept woman and bit-player to celebrated actress, friend of the duchesse, and de f
- la Berma: social_status -2 (confidence 0.95) — Her guests defect en masse to the Guermantes matinée, leaving her house empty; the world's verdict inverts the
- la Berma: emotional_position -1 (confidence 0.88) — Dying, exploited by her daughter and son-in-law, and publicly deserted, she is left with only a murmured, powe
- duchesse de Guermantes: social_status -1 (confidence 0.78) — Her intimacy with Rachel becomes evidence against her: the new generations conclude that despite her name she 
- Gilberte: social_status -1 (confidence 0.7) — Rachel's elevation reaches Gilberte as a snub by proxy: her aunt's « vive antipathie » makes receiving her lat

**advantage**: Bloch = +0.00 [neutral], Gilberte = +0.00 [neutral], Mme Verdurin = +0.00 [neutral], Rachel = +0.00 [neutral], duchesse de Guermantes = +0.00 [neutral], la Berma = -0.70 [negative], le narrateur = +0.00 [neutral]
  - Bloch vs la Berma: Bloch wins (m +0.00 vs -0.70), weight 0.56
  - Gilberte vs la Berma: Gilberte wins (m +0.00 vs -0.70), weight 0.56
  - Mme Verdurin vs la Berma: Mme Verdurin wins (m +0.00 vs -0.70), weight 0.56
  - Rachel vs la Berma: Rachel wins (m +0.00 vs -0.70), weight 0.56
  - duchesse de Guermantes vs la Berma: duchesse de Guermantes wins (m +0.00 vs -0.70), weight 0.56
  - la Berma vs le narrateur: le narrateur wins (m -0.70 vs +0.00), weight 0.56
  - (+ 15 draws among characters with no relative movement)
**prestige**: Bloch = +0.00 [neutral], Gilberte = -0.70 [negative], Mme Verdurin = +0.00 [neutral], Rachel = +1.86 [positive], duchesse de Guermantes = -0.78 [negative], la Berma = -1.90 [negative], le narrateur = +0.00 [neutral]
  - Bloch vs Gilberte: Bloch wins (m +0.00 vs -0.70), weight 0.45
  - Bloch vs Rachel: Rachel wins (m +0.00 vs +1.86), weight 0.60
  - Bloch vs duchesse de Guermantes: Bloch wins (m +0.00 vs -0.78), weight 0.50
  - Bloch vs la Berma: Bloch wins (m +0.00 vs -1.90), weight 0.61
  - Gilberte vs Mme Verdurin: Mme Verdurin wins (m -0.70 vs +0.00), weight 0.45
  - Gilberte vs Rachel: Rachel wins (m -0.70 vs +1.86), weight 0.45
  - Gilberte vs la Berma: Gilberte wins (m -0.70 vs -1.90), weight 0.45
  - Gilberte vs le narrateur: le narrateur wins (m -0.70 vs +0.00), weight 0.45
  - Mme Verdurin vs Rachel: Rachel wins (m +0.00 vs +1.86), weight 0.58
  - Mme Verdurin vs duchesse de Guermantes: Mme Verdurin wins (m +0.00 vs -0.78), weight 0.50
  - Mme Verdurin vs la Berma: Mme Verdurin wins (m +0.00 vs -1.90), weight 0.58
  - Rachel vs duchesse de Guermantes: Rachel wins (m +1.86 vs -0.78), weight 0.50
  - Rachel vs la Berma: Rachel wins (m +1.86 vs -1.90), weight 0.60
  - Rachel vs le narrateur: Rachel wins (m +1.86 vs +0.00), weight 0.60
  - duchesse de Guermantes vs la Berma: duchesse de Guermantes wins (m -0.78 vs -1.90), weight 0.50
  - duchesse de Guermantes vs le narrateur: le narrateur wins (m -0.78 vs +0.00), weight 0.50
  - la Berma vs le narrateur: le narrateur wins (m -1.90 vs +0.00), weight 0.61
  - (+ 4 draws among characters with no relative movement)
**inclusion**: Bloch = +0.00 [neutral], Gilberte = +0.00 [neutral], Mme Verdurin = +0.00 [neutral], Rachel = +0.00 [neutral], duchesse de Guermantes = +0.00 [neutral], la Berma = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 21 draws among characters with no relative movement)

## v2-p1-autour-de-mme-swann#p-111-p-115
*Search hit for Norpois & Bergotte*  (run: foundation-run-007)

- ambiguity notes: 2 -> comparison weight factor rho = 0.64

**Effects as annotated:**
- la grand-mère: general_appraisal +2 (confidence 0.93) — She comes out clearly elevated: her love is shown as costly, concealed, and unrewarded, and the narration endo
- la grand-mère: emotional_position -1 (confidence 0.8) — The same scene shows her losing ground emotionally — no longer 'maître de ses émotions qu’autrefois', red-face
- le narrateur: general_appraisal -1 (confidence 0.87) — He is locally lowered by the narration's exposure of his exacted pity and his unjust reproach of indifference.
- le narrateur: emotional_position +1 (confidence 0.7) — Within the dyad he nevertheless gains leverage: his grandmother yields to his signalled suffering, insisting u

**advantage**: Bergotte = +0.00 [neutral], Gilberte = +0.00 [neutral], Norpois = +0.00 [neutral], la grand-mère = +1.22 [positive], la mère du narrateur = +0.00 [neutral], le narrateur = -0.31 [negative]
  - Bergotte vs la grand-mère: la grand-mère wins (m +0.00 vs +1.22), weight 0.52
  - Bergotte vs le narrateur: Bergotte wins (m +0.00 vs -0.31), weight 0.50
  - Gilberte vs la grand-mère: la grand-mère wins (m +0.00 vs +1.22), weight 0.55
  - Gilberte vs le narrateur: Gilberte wins (m +0.00 vs -0.31), weight 0.50
  - Norpois vs la grand-mère: la grand-mère wins (m +0.00 vs +1.22), weight 0.55
  - Norpois vs le narrateur: Norpois wins (m +0.00 vs -0.31), weight 0.50
  - la grand-mère vs la mère du narrateur: la grand-mère wins (m +1.22 vs +0.00), weight 0.55
  - la grand-mère vs le narrateur: la grand-mère wins (m +1.22 vs -0.31), weight 0.50
  - la mère du narrateur vs le narrateur: la mère du narrateur wins (m +0.00 vs -0.31), weight 0.50
  - (+ 6 draws among characters with no relative movement)
**prestige**: Bergotte = +0.00 [neutral], Gilberte = +0.00 [neutral], Norpois = +0.00 [neutral], la grand-mère = +0.00 [neutral], la mère du narrateur = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 15 draws among characters with no relative movement)
**inclusion**: Bergotte = +0.00 [neutral], Gilberte = +0.00 [neutral], Norpois = +0.00 [neutral], la grand-mère = +0.00 [neutral], la mère du narrateur = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 15 draws among characters with no relative movement)

## v1-p1-combray#p-101-p-105
*Zero-effect bystander case (la grand-mère)*  (run: foundation-run-001)

- ambiguity notes: 1 -> comparison weight factor rho = 0.80

**Effects as annotated:**
- Legrandin: general_appraisal -1 (confidence 0.8) — Legrandin's local standing is somewhat lowered by the exposed gap between his own family's noble marriage and 
- le narrateur: general_appraisal +1 (confidence 0.72) — The narrator is locally elevated by Legrandin's direct, warm praise of his soul and artistic sensibility.

**advantage**: Legrandin = -0.80 [negative], la grand-mère = +0.00 [neutral], le narrateur = +0.72 [positive]
  - Legrandin vs la grand-mère: la grand-mère wins (m -0.80 vs +0.00), weight 0.64
  - Legrandin vs le narrateur: le narrateur wins (m -0.80 vs +0.72), weight 0.58
  - la grand-mère vs le narrateur: le narrateur wins (m +0.00 vs +0.72), weight 0.58
**prestige**: Legrandin = +0.00 [neutral], la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 3 draws among characters with no relative movement)
**inclusion**: Legrandin = +0.00 [neutral], la grand-mère = +0.00 [neutral], le narrateur = +0.00 [neutral]
  - (+ 3 draws among characters with no relative movement)

