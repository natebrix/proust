"""Pure Glicko-2 rating math.

This module implements the Glicko-2 rating system exactly as specified in
Mark Glickman's "Example of the Glicko-2 System"
(http://www.glicko.net/glicko/glicko2.pdf), including the illustrative
Newton/regula-falsi bracket procedure used to solve for the updated
volatility (Step 5 of the paper).

This module has no knowledge of the corpus, characters, chapters, or units
it may be used to rate. It only knows about players, ratings, rating
deviations (RD), volatilities, and match outcomes. Corpus-specific assembly
of matches into rating periods lives in app_exports.py.

Two scales are used throughout, matching the paper:

- the "rating scale", the familiar Elo-like scale (default initial rating
  1500, default initial RD 350)
- the "Glicko-2 scale", the internal scale the paper's algorithm actually
  operates on (mu, phi), related to the rating scale by a fixed constant

Both `rate_player` (rating scale) and `rate_player_glicko2_scale` (internal
scale) are exposed, since callers who are themselves batching many players
through a shared rating period may prefer to work in the internal scale
throughout and only convert at the edges.
"""

import math

# The fixed conversion constant between the rating scale and the Glicko-2
# scale, as given in the paper.
GLICKO2_SCALE = 173.7178

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_INITIAL_RD = 350.0
DEFAULT_INITIAL_VOLATILITY = 0.06
DEFAULT_TAU = 0.5
DEFAULT_CONVERGENCE_EPSILON = 1e-6


def to_glicko2_scale(rating, rd):
    """Convert a (rating, RD) pair on the rating scale to (mu, phi) on the Glicko-2 scale."""
    mu = (rating - DEFAULT_INITIAL_RATING) / GLICKO2_SCALE
    phi = rd / GLICKO2_SCALE
    return mu, phi


def from_glicko2_scale(mu, phi):
    """Convert a (mu, phi) pair on the Glicko-2 scale back to (rating, RD) on the rating scale."""
    rating = GLICKO2_SCALE * mu + DEFAULT_INITIAL_RATING
    rd = GLICKO2_SCALE * phi
    return rating, rd


def g(phi):
    """The paper's g(phi) function: down-weights an opponent's influence as their RD grows."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def expected_score(mu, mu_j, phi_j):
    """The paper's E(mu, mu_j, phi_j): expected score for a player against opponent j."""
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def _volatility_update_function(phi, sigma, v, delta, tau):
    a = math.log(sigma ** 2)

    def f(x):
        ex = math.exp(x)
        numerator = ex * (delta ** 2 - phi ** 2 - v - ex)
        denominator = 2.0 * (phi ** 2 + v + ex) ** 2
        return numerator / denominator - (x - a) / (tau ** 2)

    return f, a


def _solve_new_volatility(phi, sigma, v, delta, tau, convergence_epsilon):
    """Step 5 of the paper: the illustrative Newton/regula-falsi bracket to solve for sigma'."""
    f, a = _volatility_update_function(phi, sigma, v, delta, tau)

    capital_a = a
    if delta ** 2 > phi ** 2 + v:
        capital_b = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        capital_b = a - k * tau

    f_a = f(capital_a)
    f_b = f(capital_b)

    while abs(capital_b - capital_a) > convergence_epsilon:
        capital_c = capital_a + (capital_a - capital_b) * f_a / (f_b - f_a)
        f_c = f(capital_c)
        if f_c * f_b < 0:
            capital_a = capital_b
            f_a = f_b
        else:
            f_a = f_a / 2.0
        capital_b = capital_c
        f_b = f_c

    return math.exp(capital_a / 2.0)


def rate_player_glicko2_scale(mu, phi, sigma, opponents, tau=DEFAULT_TAU, convergence_epsilon=DEFAULT_CONVERGENCE_EPSILON):
    """Apply one rating period's worth of results to a single player, on the Glicko-2 scale.

    `opponents` is an iterable of `(mu_j, phi_j, score)` triples, one per
    match this period, where `mu_j`/`phi_j` are the OPPONENT's pre-period
    rating/RD on the Glicko-2 scale (never the player's own mid-period
    values -- all matches in a rating period are resolved against
    pre-period opponent state, per the paper's batch semantics) and `score`
    is the observed result for the player being rated (1.0 win, 0.5 draw,
    0.0 loss).

    If `opponents` is empty (the player had no games this period), only the
    RD grows (Step 6's phi* treated as the final phi'); rating and
    volatility are unchanged, per the paper's guidance for inactive players.

    Returns `(mu_prime, phi_prime, sigma_prime)`.
    """
    opponents = list(opponents)
    if not opponents:
        phi_prime = math.sqrt(phi ** 2 + sigma ** 2)
        return mu, phi_prime, sigma

    variance_terms = []
    delta_terms = []
    for mu_j, phi_j, score in opponents:
        g_j = g(phi_j)
        e_j = expected_score(mu, mu_j, phi_j)
        variance_terms.append(g_j ** 2 * e_j * (1.0 - e_j))
        delta_terms.append(g_j * (score - e_j))

    v = 1.0 / sum(variance_terms)
    delta = v * sum(delta_terms)

    sigma_prime = _solve_new_volatility(phi, sigma, v, delta, tau, convergence_epsilon)

    phi_star = math.sqrt(phi ** 2 + sigma_prime ** 2)
    phi_prime = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_prime = mu + phi_prime ** 2 * sum(delta_terms)

    return mu_prime, phi_prime, sigma_prime


def rate_player(rating, rd, volatility, opponents, tau=DEFAULT_TAU, convergence_epsilon=DEFAULT_CONVERGENCE_EPSILON):
    """Apply one rating period's worth of results to a single player, on the rating scale.

    `opponents` is an iterable of `(opponent_rating, opponent_rd, score)`
    triples on the rating scale, using each opponent's pre-period rating
    and RD. Returns `(rating_prime, rd_prime, sigma_prime)` on the rating
    scale.
    """
    mu, phi = to_glicko2_scale(rating, rd)
    scaled_opponents = [(*to_glicko2_scale(opponent_rating, opponent_rd), score) for opponent_rating, opponent_rd, score in opponents]
    mu_prime, phi_prime, sigma_prime = rate_player_glicko2_scale(
        mu, phi, volatility, scaled_opponents, tau=tau, convergence_epsilon=convergence_epsilon
    )
    rating_prime, rd_prime = from_glicko2_scale(mu_prime, phi_prime)
    return rating_prime, rd_prime, sigma_prime


__all__ = [
    "DEFAULT_CONVERGENCE_EPSILON",
    "DEFAULT_INITIAL_RATING",
    "DEFAULT_INITIAL_RD",
    "DEFAULT_INITIAL_VOLATILITY",
    "DEFAULT_TAU",
    "GLICKO2_SCALE",
    "expected_score",
    "from_glicko2_scale",
    "g",
    "rate_player",
    "rate_player_glicko2_scale",
    "to_glicko2_scale",
]
