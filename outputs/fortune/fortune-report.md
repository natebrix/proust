# Fortune arcs

Each character's own arc: a smoothed level of their scoring v2 movements, passage by
passage, from `proust/fortune.py`. A level of +0.5 means that around this point of the novel
the passages involving the character tend to leave them half a step up. Levels carry ± one
posterior standard deviation. Drift and noise are chosen once per lens by pooled marginal
likelihood; nothing is tuned per character.

`order p` asks whether the ORDER of a character's passages made the arc: the share of
random reshufflings of their outcomes in time that give an equal or bigger move. Around
40 characters are tested per lens, so a handful under 0.05 are expected by chance; lean on
the ones near 0.01.

`arc evidence` is how much better (in log-likelihood) the fitted model explains the
movements than one in which no character's fortune ever changes. Several points is strong
evidence that arcs are real; under one point means the lens cannot see arcs.

| lens | observations | characters | q | sigma² | arc evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 1874 | 170 | 0.016 | 0.9 | 32.8 |
| advantage | 1610 | 153 | 0.004 | 0.7 | 11.9 |
| prestige | 356 | 79 | 0.008 | 0.7 | 7.7 |
| inclusion | 200 | 52 | 0.004 | 0.9 | 0.5 |

## overall

Characters with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 89 | 0.003 | +0.58 ± 0.31 → -1.59 ± 0.33 (2.17) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| la Berma | 13 | 0.030 | +0.71 ± 0.38 → -0.91 ± 0.52 (1.62) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Temps retrouvé — IV. Le Bal de têtes |
| Swann | 147 | 0.027 | -0.13 ± 0.26 → -1.43 ± 0.35 (1.30) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Saniette | 11 | 0.196 | -0.84 ± 0.36 → -2.05 ± 0.57 (1.20) · Du Côté de Chez Swann — II. Un amour de Swann → La Prisonnière |
| Albertine | 103 | 0.007 | +0.41 ± 0.27 → -0.75 ± 0.24 (1.16) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → La Prisonnière |
| duchesse de Guermantes | 132 | 0.020 | +0.62 ± 0.27 → -0.48 ± 0.37 (1.10) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 182 | 0.033 | +0.41 ± 0.20 → -0.61 ± 0.24 (1.02) · Le Côté de Guermantes — II → La Prisonnière |
| duc de Guermantes | 63 | 0.023 | -0.54 ± 0.31 → -1.54 ± 0.45 (1.00) · Le Côté de Guermantes — I → La Prisonnière |
| marquise de Saint-Euverte | 8 | 0.538 | -0.68 ± 0.41 → -1.58 ± 0.40 (0.90) · Du Côté de Chez Swann — II. Un amour de Swann → Sodome et Gomorrhe — II |
| princesse de Guermantes | 16 | 0.063 | +0.33 ± 0.34 → -0.53 ± 0.75 (0.86) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| M. Vinteuil | 8 | 0.007 | -0.23 ± 0.33 → +1.89 ± 0.82 (2.12) · Du Côté de Chez Swann — I. Combray → La Prisonnière |
| Mme Bontemps | 8 | 0.007 | -0.33 ± 0.38 → +0.91 ± 0.62 (1.24) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| la grand-mère | 36 | 0.033 | -0.16 ± 0.32 → +0.82 ± 0.48 (0.98) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Morel | 33 | 0.020 | -0.91 ± 0.28 → -0.01 ± 0.46 (0.90) · La Prisonnière → Le Temps retrouvé — IV. Le Bal de têtes |
| marquise de Saint-Euverte | 8 | 0.036 | -1.58 ± 0.40 → -0.70 ± 0.76 (0.88) · Sodome et Gomorrhe — II → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| Robert de Saint-Loup | 105 | 0.073 | -0.41 ± 0.18 → +0.38 ± 0.44 (0.79) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| Jupien | 9 | 0.525 | +0.41 ± 0.37 → +1.18 ± 0.42 (0.77) · Le Côté de Guermantes — I → Sodome et Gomorrhe — I |
| Mme Verdurin | 48 | 0.100 | -0.59 ± 0.34 → +0.08 ± 0.39 (0.67) · Sodome et Gomorrhe — II → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Andrée | 19 | 0.246 | -0.36 ± 0.39 → +0.28 ± 0.63 (0.64) · La Prisonnière → Le Temps retrouvé — IV. Le Bal de têtes |
| Françoise | 39 | 0.429 | +0.02 ± 0.32 → +0.64 ± 0.34 (0.62) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |


## advantage

Characters with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 79 | 0.003 | +0.20 ± 0.24 → -1.09 ± 0.25 (1.28) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — III. Matinée chez la princesse de Guermantes. L'Adoration perpétuelle |
| Gilberte | 30 | 0.003 | +0.30 ± 0.24 → -0.50 ± 0.32 (0.80) · Du Côté de Chez Swann — III. Noms de pays : le nom → Le Temps retrouvé — IV. Le Bal de têtes |
| Albertine | 100 | 0.007 | +0.15 ± 0.20 → -0.59 ± 0.22 (0.75) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Albertine disparue — III |
| Odette | 68 | 0.027 | -0.04 ± 0.15 → -0.65 ± 0.32 (0.61) · Du Côté de Chez Swann — II. Un amour de Swann → Albertine disparue — IV |
| duchesse de Guermantes | 111 | 0.040 | +0.22 ± 0.16 → -0.32 ± 0.30 (0.54) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| Saniette | 10 | 0.239 | -0.69 ± 0.32 → -1.09 ± 0.33 (0.40) · Du Côté de Chez Swann — II. Un amour de Swann → Sodome et Gomorrhe — II |
| Andrée | 18 | 0.110 | +0.04 ± 0.26 → -0.31 ± 0.30 (0.35) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → La Prisonnière |
| Swann | 132 | 0.339 | -0.34 ± 0.19 → -0.68 ± 0.35 (0.34) · Le Côté de Guermantes — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Norpois | 34 | 0.103 | -0.01 ± 0.22 → -0.34 ± 0.32 (0.34) · Le Côté de Guermantes — I → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Mme de Marsantes | 13 | 0.046 | -0.54 ± 0.31 → -0.88 ± 0.48 (0.34) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Albertine disparue — IV |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| M. Vinteuil | 8 | 0.010 | -0.16 ± 0.30 → +0.47 ± 0.54 (0.62) · Du Côté de Chez Swann — I. Combray → La Prisonnière |
| la grand-mère | 31 | 0.053 | -0.04 ± 0.26 → +0.57 ± 0.32 (0.60) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Robert de Saint-Loup | 97 | 0.053 | -0.31 ± 0.13 → +0.15 ± 0.30 (0.47) · Le Côté de Guermantes — I → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 134 | 0.432 | -0.39 ± 0.17 → -0.12 ± 0.19 (0.27) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Sodome et Gomorrhe — II |
| le père du narrateur | 9 | 0.146 | -0.01 ± 0.34 → +0.26 ± 0.46 (0.27) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| M. Verdurin | 13 | 0.043 | -0.48 ± 0.28 → -0.23 ± 0.45 (0.25) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Françoise | 37 | 0.611 | +0.06 ± 0.25 → +0.28 ± 0.23 (0.22) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |
| docteur Cottard | 29 | 0.332 | -0.28 ± 0.20 → -0.06 ± 0.45 (0.22) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — I. A Tansonville |
| Morel | 27 | 0.066 | -1.07 ± 0.22 → -0.87 ± 0.30 (0.21) · La Prisonnière → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| la mère du narrateur | 10 | 0.515 | +0.47 ± 0.29 → +0.68 ± 0.40 (0.21) · Du Côté de Chez Swann — I. Combray → Le Côté de Guermantes — II |


## prestige

Characters with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| baron de Charlus | 28 | 0.007 | +0.60 ± 0.32 → -0.57 ± 0.39 (1.17) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Temps retrouvé — IV. Le Bal de têtes |
| duchesse de Guermantes | 42 | 0.007 | +1.14 ± 0.20 → +0.19 ± 0.39 (0.95) · Le Côté de Guermantes — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Swann | 32 | 0.106 | +0.33 ± 0.24 → -0.31 ± 0.63 (0.65) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Robert de Saint-Loup | 18 | 0.106 | +0.60 ± 0.31 → +0.09 ± 0.44 (0.51) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Albertine disparue — IV |
| le narrateur | 22 | 0.076 | +0.72 ± 0.28 → +0.23 ± 0.49 (0.50) · Le Côté de Guermantes — II → Albertine disparue — I |
| Mme de Villeparisis | 16 | 0.253 | +0.18 ± 0.30 → -0.03 ± 0.38 (0.21) · À l'Ombre des Jeunes Filles en Fleurs — II. Noms de pays : le pays → Le Côté de Guermantes — II |
| Odette | 18 | 0.684 | +0.98 ± 0.40 → +0.86 ± 0.58 (0.13) · Sodome et Gomorrhe — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Gilberte | 8 | 0.492 | +0.52 ± 0.37 → +0.41 ± 0.44 (0.12) · Albertine disparue — II → Le Temps retrouvé — IV. Le Bal de têtes |
| Mme Verdurin | 17 | 0.774 | +0.63 ± 0.37 → +0.54 ± 0.33 (0.09) · Sodome et Gomorrhe — II → La Prisonnière |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Odette | 18 | 0.449 | +0.58 ± 0.32 → +0.98 ± 0.40 (0.40) · Du Côté de Chez Swann — I. Combray → Sodome et Gomorrhe — II |
| Mme Verdurin | 17 | 0.352 | +0.43 ± 0.32 → +0.81 ± 0.43 (0.39) · Du Côté de Chez Swann — II. Un amour de Swann → Le Temps retrouvé — IV. Le Bal de têtes |
| duchesse de Guermantes | 42 | 0.631 | +0.82 ± 0.32 → +1.14 ± 0.20 (0.32) · Du Côté de Chez Swann — I. Combray → Le Côté de Guermantes — II |
| le narrateur | 22 | 0.605 | +0.55 ± 0.31 → +0.72 ± 0.28 (0.17) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Le Côté de Guermantes — II |
| Gilberte | 8 | 0.767 | +0.47 ± 0.40 → +0.52 ± 0.37 (0.05) · Du Côté de Chez Swann — III. Noms de pays : le nom → Albertine disparue — II |
| Robert de Saint-Loup | 18 | 0.917 | +0.09 ± 0.44 → +0.12 ± 0.47 (0.03) · Albertine disparue — IV → Le Temps retrouvé — II. M. de Charlus pendant la guerre |
| Mme de Villeparisis | 16 | 0.751 | +0.00 ± 0.27 → +0.02 ± 0.28 (0.02) · Le Côté de Guermantes — I → Le Côté de Guermantes — I |
| Swann | 32 | 0.987 | +0.28 ± 0.22 → +0.29 ± 0.21 (0.01) · Du Côté de Chez Swann — I. Combray → Du Côté de Chez Swann — II. Un amour de Swann |
| baron de Charlus | 28 | 0.997 | +0.43 ± 0.28 → +0.43 ± 0.27 (0.00) · Sodome et Gomorrhe — II → Sodome et Gomorrhe — II |


## inclusion

Characters with at least 8 appearances.

### Biggest falls

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Swann | 28 | 0.076 | -0.58 ± 0.25 → -1.08 ± 0.39 (0.50) · Du Côté de Chez Swann — I. Combray → Albertine disparue — II |
| Odette | 9 | 0.233 | -0.84 ± 0.32 → -1.27 ± 0.57 (0.42) · Du Côté de Chez Swann — I. Combray → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 63 | 0.160 | +0.32 ± 0.20 → +0.08 ± 0.33 (0.24) · À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann → Sodome et Gomorrhe — II |
| Bloch | 13 | 0.987 | -0.73 ± 0.31 → -0.73 ± 0.31 (0.00) · Le Côté de Guermantes — I → Le Côté de Guermantes — I |

### Biggest rises

| character | n | order p | move |
| --- | ---: | ---: | --- |
| Bloch | 13 | 0.057 | -0.79 ± 0.32 → -0.40 ± 0.49 (0.39) · Du Côté de Chez Swann — I. Combray → Le Temps retrouvé — IV. Le Bal de têtes |
| le narrateur | 63 | 0.824 | +0.19 ± 0.27 → +0.32 ± 0.20 (0.14) · Du Côté de Chez Swann — I. Combray → À l'Ombre des Jeunes Filles en Fleurs — I. Autour de Mme Swann |

