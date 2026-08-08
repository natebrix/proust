"""Pure Whole-History Rating (WHR) math.

This module implements Remi Coulom's "Whole-History Rating: A Bayesian
Rating System for Players of Time-Varying Strength" (Computers and Games
2008). Unlike ELO (sequential, fixed step) or Glicko-2 (batched into
rating periods, each period a forward update), WHR estimates the entire
rating TRAJECTORY of every player at once, as the maximum a posteriori
point of a Bayesian model in which

- game results follow Bradley-Terry: player i beats player j with
  probability gamma_i / (gamma_i + gamma_j)
- a player's rating follows a Wiener process (Brownian motion) in time:
  the increment between two consecutive appearances separated by `dt` is
  Normal(0, w2 * dt) on the natural log-gamma scale

Like glicko2.py, this module has no knowledge of the corpus, characters,
chapters, or units it may be used to rate. It only knows about players,
integer time points, and game results. Corpus-specific assembly of
matches lives in character_whr.py.

Two scales are used:

- the internal scale, natural log-gamma (`r = ln gamma`), which is what
  every recurrence below actually operates on
- the familiar Elo-like display scale, `rating = 1500 + (400 / ln 10) * r`

The conversion constant `400 / ln 10 = 173.7178...` is the same constant
Glicko-2 uses between its rating scale and its internal scale, so a WHR
band and a Glicko-2 RD are directly comparable in Elo points.

Three structural facts make the fit cheap and exact:

1. Holding all other players fixed, one player's log-posterior is
   strictly concave in their own trajectory, so Newton's method on that
   trajectory converges.
2. The Hessian of that trajectory is TRIDIAGONAL -- the Bradley-Terry
   terms only touch the diagonal, and the Wiener prior only couples
   consecutive nodes -- so a Newton step costs O(nodes) via the standard
   tridiagonal LU (Thomas algorithm). Dense matrices are never formed.
3. The diagonal of the inverse of a tridiagonal matrix also has an
   O(nodes) forward/backward recurrence, which is where the per-node
   posterior variance (and hence the uncertainty band) comes from.

Coordinate ascent over players, using 1-3 for each player in turn, is
Coulom's fitting procedure and is what `fit` does.
"""

from collections import defaultdict
import math

# rating = 1500 + ELO_PER_LOG_GAMMA * log_gamma
ELO_PER_LOG_GAMMA = 400.0 / math.log(10.0)

DEFAULT_INITIAL_RATING = 1500.0
# Prior standard deviation, in Elo points, on a player's rating at their
# FIRST node. This is the Wiener process's initial distribution; it plays
# the same role as Glicko-2's initial RD of 350 (and uses the same value),
# it pins the otherwise free additive constant of a Bradley-Terry model,
# and it keeps an undefeated player's rating finite.
DEFAULT_INITIAL_RD = 350.0
DEFAULT_MAX_SWEEPS = 200
DEFAULT_TOLERANCE = 1e-6
DEFAULT_NEWTON_ITERATIONS = 20
DEFAULT_NEWTON_TOLERANCE = 1e-10

# A draw is modeled as two half-weight games: half a win and half a loss
# for each side. Recorded in artifacts as this identifier.
DRAW_MODEL = "half_win_half_loss"


def to_rating(log_gamma):
    """Convert an internal log-gamma value to the Elo-like display scale."""
    return DEFAULT_INITIAL_RATING + ELO_PER_LOG_GAMMA * log_gamma


def from_rating(rating):
    """Convert an Elo-like display rating to the internal log-gamma scale."""
    return (rating - DEFAULT_INITIAL_RATING) / ELO_PER_LOG_GAMMA


def to_rating_span(log_gamma_span):
    """Convert a log-gamma DIFFERENCE (a band, a standard deviation) to Elo points."""
    return ELO_PER_LOG_GAMMA * log_gamma_span


def w2_from_elo(w2_elo):
    """Convert a Wiener variance rate given in Elo^2 per time unit to log-gamma^2 per time unit."""
    return w2_elo / (ELO_PER_LOG_GAMMA ** 2)


def variance_from_rd(rd_elo):
    """Convert a standard deviation in Elo points to a variance on the log-gamma scale."""
    return (rd_elo / ELO_PER_LOG_GAMMA) ** 2


def logistic(x):
    """Numerically stable 1 / (1 + exp(-x)); never exponentiates a raw rating."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def win_probability(log_gamma_a, log_gamma_b):
    """Bradley-Terry gamma_a / (gamma_a + gamma_b), computed in log space."""
    return logistic(log_gamma_a - log_gamma_b)


def solve_symmetric_tridiagonal(diagonal, off_diagonal, right_hand_side):
    """Solve a symmetric tridiagonal system by the Thomas algorithm (LU without pivoting).

    `diagonal` has length n, `off_diagonal` length n-1 (entry k is both
    A[k][k+1] and A[k+1][k]). The matrices this module builds are
    positive definite, so every pivot must be positive; a non-positive
    pivot means the caller handed us something that is not, and we raise
    rather than silently producing a meaningless step.
    """
    n = len(diagonal)
    if n == 0:
        return []
    pivots = [0.0] * n
    forward = [0.0] * n
    pivots[0] = diagonal[0]
    if pivots[0] <= 0.0:
        raise ValueError("Non-positive pivot in tridiagonal solve at index 0.")
    forward[0] = right_hand_side[0]
    for index in range(1, n):
        multiplier = off_diagonal[index - 1] / pivots[index - 1]
        pivots[index] = diagonal[index] - multiplier * off_diagonal[index - 1]
        if pivots[index] <= 0.0:
            raise ValueError(f"Non-positive pivot in tridiagonal solve at index {index}.")
        forward[index] = right_hand_side[index] - multiplier * forward[index - 1]

    solution = [0.0] * n
    solution[n - 1] = forward[n - 1] / pivots[n - 1]
    for index in range(n - 2, -1, -1):
        solution[index] = (forward[index] - off_diagonal[index] * solution[index + 1]) / pivots[index]
    return solution


def tridiagonal_inverse_diagonal(diagonal, off_diagonal):
    """Diagonal entries of the inverse of a symmetric positive definite tridiagonal matrix.

    Eliminating everything BEFORE row i gives the forward pivot
    `u[i] = d[i] - e[i-1]^2 / u[i-1]`; eliminating everything AFTER row i
    gives the backward pivot `v[i] = d[i] - e[i]^2 / v[i+1]`. The Schur
    complement of row i against the whole rest of the matrix is therefore
    `u[i] + v[i] - d[i]`, and the inverse's diagonal entry is its
    reciprocal. This is Coulom's appendix recurrence written in the form
    that cannot overflow (no continuant determinants are ever formed).
    """
    n = len(diagonal)
    if n == 0:
        return []
    forward_pivots = [0.0] * n
    forward_pivots[0] = diagonal[0]
    if forward_pivots[0] <= 0.0:
        raise ValueError("Non-positive pivot in tridiagonal inverse recurrence at index 0.")
    for index in range(1, n):
        forward_pivots[index] = diagonal[index] - off_diagonal[index - 1] ** 2 / forward_pivots[index - 1]
        if forward_pivots[index] <= 0.0:
            raise ValueError(f"Non-positive forward pivot in tridiagonal inverse recurrence at index {index}.")

    backward_pivots = [0.0] * n
    backward_pivots[n - 1] = diagonal[n - 1]
    if backward_pivots[n - 1] <= 0.0:
        raise ValueError(f"Non-positive backward pivot in tridiagonal inverse recurrence at index {n - 1}.")
    for index in range(n - 2, -1, -1):
        backward_pivots[index] = diagonal[index] - off_diagonal[index] ** 2 / backward_pivots[index + 1]
        if backward_pivots[index] <= 0.0:
            raise ValueError(f"Non-positive backward pivot in tridiagonal inverse recurrence at index {index}.")

    return [1.0 / (forward_pivots[i] + backward_pivots[i] - diagonal[i]) for i in range(n)]


def expand_draws(games):
    """Rewrite `(player_a, player_b, time, score_a)` games as weighted decisive games.

    A win becomes one full-weight game; a loss becomes one full-weight
    game with the players swapped; a draw becomes TWO half-weight games,
    one in each direction. That is the draw model this module commits to,
    and it is what `DRAW_MODEL` names.
    """
    expanded = []
    for player_a, player_b, time, score_a in games:
        if player_a == player_b:
            raise ValueError(f'Player "{player_a}" cannot play itself at time {time}.')
        if score_a == 1.0:
            expanded.append((player_a, player_b, time, 1.0))
        elif score_a == 0.0:
            expanded.append((player_b, player_a, time, 1.0))
        elif score_a == 0.5:
            expanded.append((player_a, player_b, time, 0.5))
            expanded.append((player_b, player_a, time, 0.5))
        else:
            raise ValueError(f"Unsupported score {score_a!r}; expected 1.0, 0.5, or 0.0.")
    return expanded


def _build_players(games, warm_start=None):
    """Build the per-player node structure a coordinate-ascent sweep operates on.

    A player gets exactly one node per distinct time at which they play.
    Each node carries the games played at that time as
    `(opponent_name, opponent_node_index, weight, result)`, so a sweep can
    read an opponent's CURRENT log-gamma at the right node without
    re-searching.
    """
    decisive = expand_draws(games)

    times_by_player = defaultdict(set)
    for winner, loser, time, _weight in decisive:
        times_by_player[winner].add(time)
        times_by_player[loser].add(time)

    players = {}
    for name in sorted(times_by_player):
        times = sorted(times_by_player[name])
        seed = (warm_start or {}).get(name) or {}
        log_gammas = []
        carried = 0.0
        for time in times:
            if time in seed:
                carried = seed[time]
            log_gammas.append(carried)
        players[name] = {
            "times": times,
            "node_index_by_time": {time: index for index, time in enumerate(times)},
            "log_gammas": log_gammas,
            "games": [[] for _ in times],
        }

    for winner, loser, time, weight in decisive:
        winner_player = players[winner]
        loser_player = players[loser]
        winner_index = winner_player["node_index_by_time"][time]
        loser_index = loser_player["node_index_by_time"][time]
        winner_player["games"][winner_index].append((loser, loser_index, weight, 1.0))
        loser_player["games"][loser_index].append((winner, winner_index, weight, 0.0))

    return players


def _opponents_by_player(players):
    """Map each player to the set of players they face at any time."""
    opponents = {}
    for name, player in players.items():
        faced = set()
        for node_games in player["games"]:
            for opponent_name, _opponent_index, _weight, _result in node_games:
                faced.add(opponent_name)
        opponents[name] = faced
    return opponents


def _player_system(player, players, w2, initial_variance):
    """Assemble the gradient and the (negated) tridiagonal Hessian for one player.

    Returned `diagonal`/`off_diagonal` describe -H, which is positive
    definite, so the Newton step for maximization is the solution of
    (-H) delta = gradient.
    """
    times = player["times"]
    log_gammas = player["log_gammas"]
    node_count = len(times)

    gradient = [0.0] * node_count
    diagonal = [0.0] * node_count
    off_diagonal = [0.0] * (node_count - 1)

    for index in range(node_count):
        own = log_gammas[index]
        for opponent_name, opponent_index, weight, result in player["games"][index]:
            opponent = players[opponent_name]["log_gammas"][opponent_index]
            probability = logistic(own - opponent)
            gradient[index] += weight * (result - probability)
            diagonal[index] += weight * probability * (1.0 - probability)

    if initial_variance is not None:
        # Prior on the trajectory's starting point: r_0 ~ Normal(0, initial_variance).
        gradient[0] -= log_gammas[0] / initial_variance
        diagonal[0] += 1.0 / initial_variance

    for index in range(node_count - 1):
        precision = 1.0 / (w2 * (times[index + 1] - times[index]))
        difference = log_gammas[index + 1] - log_gammas[index]
        gradient[index] += precision * difference
        gradient[index + 1] -= precision * difference
        diagonal[index] += precision
        diagonal[index + 1] += precision
        off_diagonal[index] = -precision

    return gradient, diagonal, off_diagonal


def log_posterior(players, w2, initial_variance):
    """Log of the unnormalized posterior over all players' trajectories.

    Used by the tests to check gradients by finite differences; the fit
    itself never needs it.
    """
    total = 0.0
    for name in sorted(players):
        player = players[name]
        times = player["times"]
        log_gammas = player["log_gammas"]
        for index in range(len(times)):
            own = log_gammas[index]
            for opponent_name, opponent_index, weight, result in player["games"][index]:
                opponent = players[opponent_name]["log_gammas"][opponent_index]
                probability = logistic(own - opponent)
                # Every game appears once for the winner and once for the
                # loser, so each side contributes half of its log term and
                # the pair sums to the full Bradley-Terry log-likelihood.
                if result == 1.0:
                    total += 0.5 * weight * math.log(probability)
                else:
                    total += 0.5 * weight * math.log(1.0 - probability)
        if initial_variance is not None:
            total -= log_gammas[0] ** 2 / (2.0 * initial_variance)
        for index in range(len(times) - 1):
            variance = w2 * (times[index + 1] - times[index])
            total -= (log_gammas[index + 1] - log_gammas[index]) ** 2 / (2.0 * variance)
    return total


def _recenter(players, names):
    """Maximize the posterior along the one direction plain coordinate ascent cannot see.

    Adding the same constant to every rating of every player leaves both
    the Bradley-Terry likelihood and the Wiener increments untouched --
    only the prior on each player's first node notices. Coordinate ascent
    therefore crawls along that direction (each player alone has almost no
    incentive to move), which is what makes an otherwise converged fit
    take hundreds of sweeps. Treating the shift as its own block has a
    closed form: the optimum puts the MEAN first-node rating at zero.

    With a flat initial prior (`initial_rd=None`) the shift is a genuine
    gauge freedom rather than a slow mode, and this same step fixes the
    gauge, which is what makes that case deterministic.

    Returns the magnitude of the shift applied.
    """
    if not names:
        return 0.0
    shift = -sum(players[name]["log_gammas"][0] for name in names) / len(names)
    if shift == 0.0:
        return 0.0
    for name in names:
        log_gammas = players[name]["log_gammas"]
        for index in range(len(log_gammas)):
            log_gammas[index] += shift
    return abs(shift)


def _update_player(player, players, w2, initial_variance, newton_iterations, newton_tolerance):
    """Run Newton's method on one player's whole trajectory; return how far it moved."""
    before = list(player["log_gammas"])
    for _iteration in range(newton_iterations):
        gradient, diagonal, off_diagonal = _player_system(player, players, w2, initial_variance)
        step = solve_symmetric_tridiagonal(diagonal, off_diagonal, gradient)
        for index in range(len(step)):
            player["log_gammas"][index] += step[index]
        if max(abs(value) for value in step) < newton_tolerance:
            break
    return max(abs(player["log_gammas"][index] - before[index]) for index in range(len(before)))


def fit(
    games,
    w2_elo,
    initial_rd=DEFAULT_INITIAL_RD,
    max_sweeps=DEFAULT_MAX_SWEEPS,
    tolerance=DEFAULT_TOLERANCE,
    newton_iterations=DEFAULT_NEWTON_ITERATIONS,
    newton_tolerance=DEFAULT_NEWTON_TOLERANCE,
    warm_start=None,
):
    """Fit whole-history rating trajectories to `games` by coordinate ascent.

    `games` is an iterable of `(player_a, player_b, time, score_a)` where
    `time` is an integer narrative index supplied by the caller and
    `score_a` is 1.0, 0.5, or 0.0 from player_a's point of view.

    `w2_elo` is the Wiener variance rate in Elo^2 per unit of time; it is
    converted internally to the log-gamma scale. `initial_rd` is the
    prior standard deviation, in Elo points, of a player's first node;
    pass None for a flat (improper) prior, in which case the fit is only
    determined up to an additive constant and every first node must carry
    games.

    `warm_start` is an optional `{player: {time: log_gamma}}` mapping used
    to seed the ascent; nodes with no seed inherit the player's most
    recent seeded value, which is what makes the filtered (online) fits
    cheap.

    Returns a dict with `players` mapping each player to a list of node
    dicts (`time`, `log_gamma`, `variance`, `rating`, `band`), plus
    convergence statistics. `band` is `2 * sqrt(variance)` in Elo points,
    an approximate 95% interval for that node conditional on the other
    players' trajectories.
    """
    if w2_elo <= 0.0:
        raise ValueError("w2_elo must be positive; a zero Wiener rate would freeze every trajectory.")
    w2 = w2_from_elo(w2_elo)
    initial_variance = None if initial_rd is None else variance_from_rd(initial_rd)

    players = _build_players(games, warm_start=warm_start)
    names = sorted(players)
    opponents_by_player = _opponents_by_player(players)

    # Coordinate ascent with an active set. A sweep updates every player
    # currently in the active set; a player whose trajectory actually
    # moved re-activates itself and everyone it has ever played, since
    # those are the only players whose optima it can have disturbed. From
    # a cold start the first sweep activates everyone, so this is
    # ordinary coordinate ascent; from a warm start (the filtered fits,
    # where only the newest time's players have new evidence) it skips
    # the overwhelming majority of already-converged trajectories without
    # changing the fixed point it converges to.
    sweep_count = 0
    max_change = 0.0
    converged = not names
    active = list(names)
    while sweep_count < max_sweeps and active:
        sweep_count += 1
        max_change = 0.0
        reactivated = set()
        for name in active:
            change = _update_player(
                players[name], players, w2, initial_variance, newton_iterations, newton_tolerance
            )
            if change > max_change:
                max_change = change
            if change >= tolerance:
                reactivated.add(name)
                reactivated.update(opponents_by_player[name])
        shift = _recenter(players, names)
        if shift > max_change:
            max_change = shift
        if shift >= tolerance:
            reactivated.update(names)
        active = sorted(reactivated)
        if not active:
            converged = True

    result_players = {}
    for name in names:
        player = players[name]
        _gradient, diagonal, off_diagonal = _player_system(player, players, w2, initial_variance)
        variances = tridiagonal_inverse_diagonal(diagonal, off_diagonal)
        nodes = []
        for index, time in enumerate(player["times"]):
            log_gamma = player["log_gammas"][index]
            variance = variances[index]
            nodes.append(
                {
                    "time": time,
                    "log_gamma": log_gamma,
                    "variance": variance,
                    "rating": to_rating(log_gamma),
                    "band": to_rating_span(2.0 * math.sqrt(variance)),
                }
            )
        result_players[name] = nodes

    return {
        "players": result_players,
        "player_count": len(names),
        "sweep_count": sweep_count,
        "max_log_gamma_change": max_change,
        "converged": converged,
        "w2_elo": w2_elo,
        "initial_rd": initial_rd,
        "draw_model": DRAW_MODEL,
    }


def warm_start_from(result):
    """Turn a `fit` result into the `{player: {time: log_gamma}}` seed for the next fit."""
    return {
        name: {node["time"]: node["log_gamma"] for node in nodes}
        for name, nodes in result["players"].items()
    }


__all__ = [
    "DEFAULT_INITIAL_RATING",
    "DEFAULT_INITIAL_RD",
    "DEFAULT_MAX_SWEEPS",
    "DEFAULT_NEWTON_ITERATIONS",
    "DEFAULT_NEWTON_TOLERANCE",
    "DEFAULT_TOLERANCE",
    "DRAW_MODEL",
    "ELO_PER_LOG_GAMMA",
    "expand_draws",
    "fit",
    "from_rating",
    "log_posterior",
    "logistic",
    "solve_symmetric_tridiagonal",
    "to_rating",
    "to_rating_span",
    "tridiagonal_inverse_diagonal",
    "variance_from_rd",
    "w2_from_elo",
    "warm_start_from",
    "win_probability",
]
