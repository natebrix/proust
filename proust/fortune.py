"""Fortune: each character's own arc, as a smoothed level of their movements.

The rating layer (`whr.py` via scoring v2) answers a RELATIVE question --
who comes out ahead of whom inside a passage -- and a relative score can
only move as far as the characters around it let it. Fortune answers the
other question: what is happening TO this character, passage by passage,
across the novel. It needs no opponents.

The model is the local-level (random-walk-plus-noise) state space model:

    level:        x(t') = x(t) + Normal(0, q * (t' - t))
    observation:  y(u)  = x(t_u) + Normal(0, sigma2 / w(u))

where `y(u)` is the character's scoring v2 movement in passage u under one
lens, `t_u` is the passage's narrative position, and `w(u)` is the
passage's evidence weight: the character's confidence kappa times the
unit's ambiguity discount rho, exactly the quantities scoring v2 already
uses to weigh comparisons. Uncertainty therefore still weighs and never
subtracts: a doubtful reading is a noisier observation, not a smaller one.

The level x is read in movement units: +0.5 means that, around this point
of the novel, the passages that involve this character tend to leave them
half a step up. A Kalman filter gives the level from the past only; the
Rauch-Tung-Striebel smoother gives it from the whole novel, which is the
arc a reader who has finished the book would draw.

`q` (how fast fortune may drift) and `sigma2` (how noisy one passage is)
are not tuned per character or against any expected arc. They are chosen
once per lens by maximizing the marginal likelihood pooled over every
character, so the data decide how much arc there is.

This module is pure: plain lists and dicts in, plain data out.
"""

from collections import defaultdict
import math

from . import scoring_v2 as v2

# The three scoring v2 lenses plus their sum: a character's whole movement
# in a passage, whichever dimension carried it.
OVERALL_LENS = "overall"
FORTUNE_LENS_ORDER = (OVERALL_LENS,) + tuple(v2.SCORING_V2_LENS_ORDER)
# Prior on a character's level before their first appearance: centred on
# the lens-wide mean movement (supplied by the caller), with this standard
# deviation in movement units. Wide enough that one strong passage moves
# it, narrow enough that a single appearance cannot claim a +-2 fortune.
DEFAULT_PRIOR_SD = 0.5

# Narrative time is measured in units of this many words of the novel.
WORDS_PER_TIME_UNIT = 10_000

DEFAULT_Q_GRID = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
DEFAULT_SIGMA2_GRID = (0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9, 1.2)

MIN_WEIGHT = 0.05


def unit_time(corpus_position):
    """A passage's narrative time: the midpoint of its word span, in 10k-word units.

    Words rather than passage counts, so a long passage is a long stretch
    of the novel and the drift between two appearances grows with how much
    of the book lies between them.
    """
    start = corpus_position["cumulative_word_count"]
    end = corpus_position["cumulative_word_count_end"]
    return round((start + end) / 2.0 / WORDS_PER_TIME_UNIT, 4)


def unit_observations(annotation, lens):
    """{character: (movement, weight)} for the characters this lens saw move.

    Only lens participants -- characters with at least one effect in the
    lens -- are observed. A character present but untouched by the lens is
    silence, not a zero: the same vacuous-pair reasoning scoring v2 applies
    to comparisons. For `overall`, movements add across the three lenses
    and the weight uses the mean of the per-lens confidences.
    """
    lenses = v2.SCORING_V2_LENS_ORDER if lens == OVERALL_LENS else (v2.require_known_lens(lens),)
    rho = v2.ambiguity_weight(annotation)
    movement = defaultdict(float)
    confidences = defaultdict(list)
    for each in lenses:
        movements = v2.unit_movements(annotation, each)
        lens_confidences = v2.unit_confidences(annotation, each)
        participants = {
            effect.get("character")
            for effect in annotation.get("status_effects") or []
            if isinstance(effect, dict) and v2.lens_weight(each, effect.get("dimension")) > 0
        }
        for character in participants & set(movements):
            movement[character] += movements[character]
            confidences[character].append(lens_confidences[character])
    return {
        character: (round(movement[character], 12), rho * sum(values) / len(values))
        for character, values in confidences.items()
    }


# ---------------------------------------------------------------------------
# Person view: follow a person across names and titles.
# ---------------------------------------------------------------------------

NAME_KEY_PREFIX = "name:"


class PersonKeyer:
    """Maps an annotation name in a passage to the person it refers to.

    Keys are registry entity ids (after `person_view_merge` links, so "le
    peintre" is Elstir and the prince des Laumes is the duc de Guermantes)
    or, for names the registry cannot place, `name:<name>`, so an
    unresolved or ambiguous name never pools with anyone. Resolution is the
    registry's own, passage-aware (`Registry.resolve` with `unit_id`), so
    chapter-scoped rulings and `unit_rulings` in characters.yaml apply here
    exactly as they do in scoring v2's person view.

    `ruled` and `ambiguous` record which registry rulings the build leaned
    on, for the report.
    """

    def __init__(self, registry):
        self.registry = registry
        self.merge_map = v2.person_view_merge_map(registry)
        self.ruled = set()  # (unit_id, name)
        self.ambiguous = {}  # (chapter_id, name) -> candidate entity ids

    def key(self, name, unit_id=None, chapter_id=None):
        if name in v2.NON_CHARACTER_NAMES:
            return NAME_KEY_PREFIX + name
        resolution = self.registry.resolve(name, chapter_id=chapter_id, unit_id=unit_id)
        if resolution.status == "ambiguous":
            self.ambiguous[(chapter_id, name)] = resolution.candidates
        if resolution.status != "resolved":
            return NAME_KEY_PREFIX + name
        if (unit_id, name) in self.registry.unit_rulings:
            self.ruled.add((unit_id, name))
        return self.merge_map.get(resolution.entity_id, resolution.entity_id)

    def display(self, key):
        if key.startswith(NAME_KEY_PREFIX):
            return key[len(NAME_KEY_PREFIX):]
        entity = self.registry.entities.get(key)
        return entity.display_name if entity else key


def character_series(units, lens, keyer=None):
    """{character: [(time, movement, weight), ...]} in narrative order.

    `units` are `scoring_v2_build.load_scored_units` rows (each carrying
    `annotation`, `corpus_position`, `unit_id` and `chapter_id`). Without a
    `keyer` characters are annotation names (the name view). With a
    `PersonKeyer` they are person keys, and two names for one person in the
    same passage become one observation: their movements add, since both
    are movements of that person, and the weight is their mean.
    """
    series = defaultdict(list)
    for unit in sorted(units, key=lambda row: row["time"]):
        time = unit_time(unit["corpus_position"])
        merged = defaultdict(list)
        for character, (movement, weight) in unit_observations(unit["annotation"], lens).items():
            key = character if keyer is None else keyer.key(
                character, unit_id=unit.get("unit_id"), chapter_id=unit.get("chapter_id")
            )
            merged[key].append((movement, weight))
        for key, parts in merged.items():
            movement = sum(value for value, _weight in parts)
            weight = sum(w for _value, w in parts) / len(parts)
            series[key].append((time, movement, weight))
    return dict(series)


# ---------------------------------------------------------------------------
# Elo-style display scale.
# ---------------------------------------------------------------------------

FORTUNE_RATING_CENTER = 1500.0

# Phi(z) is within 0.01 of logistic(1.702 z) everywhere.
PROBIT_TO_LOGIT = 1.702


def elo_points_per_level(sigma2):
    """Display points per unit of fortune level, derived from the fitted noise.

    Two characters at levels a and b each have their next passage drawn
    around their level with variance sigma2, so A's passage beats B's with
    probability Phi((a - b) / sqrt(2 sigma2)). Matching that to Elo's
    1 / (1 + 10^(-D/400)) fixes the scale: a gap of D points means what it
    means in chess, "wins about this often", with passages as the games.
    """
    return (400.0 / math.log(10.0)) * PROBIT_TO_LOGIT / math.sqrt(2.0 * sigma2)


def fortune_rating(level, sigma2):
    """Level on the Elo-style scale. 1500 is a level of 0: passages leave you where you were."""
    return FORTUNE_RATING_CENTER + elo_points_per_level(sigma2) * level


def filter_and_smooth(observations, q, sigma2, prior_mean=0.0, prior_sd=DEFAULT_PRIOR_SD):
    """Kalman filter and RTS smoother for one character.

    `observations` is a list of `(time, value, weight)` sorted by time.
    Several observations at the same time are applied sequentially at one
    node. Returns `(nodes, log_likelihood)` where each node is
    `{time, filtered, filtered_var, smoothed, smoothed_var, n}` and the
    log-likelihood is the sum of the one-step predictive log densities --
    the marginal likelihood of this character's movements under (q, sigma2).
    """
    if not observations:
        return [], 0.0

    grouped = []
    for time, value, weight in observations:
        if grouped and grouped[-1][0] == time:
            grouped[-1][1].append((value, weight))
        else:
            grouped.append((time, [(value, weight)]))

    mean = prior_mean
    var = prior_sd ** 2
    previous_time = None
    log_likelihood = 0.0
    forward = []  # (time, predicted_mean, predicted_var, filtered_mean, filtered_var, n)
    for time, values in grouped:
        if previous_time is not None:
            var += q * (time - previous_time)
        predicted_mean, predicted_var = mean, var
        for value, weight in values:
            noise = sigma2 / max(weight, MIN_WEIGHT)
            innovation_var = var + noise
            residual = value - mean
            log_likelihood += -0.5 * (math.log(2.0 * math.pi * innovation_var) + residual ** 2 / innovation_var)
            gain = var / innovation_var
            mean += gain * residual
            var *= 1.0 - gain
        forward.append((time, predicted_mean, predicted_var, mean, var, len(values)))
        previous_time = time

    smoothed_mean = forward[-1][3]
    smoothed_var = forward[-1][4]
    backward = [(smoothed_mean, smoothed_var)]
    for index in range(len(forward) - 2, -1, -1):
        _time, _pm, _pv, filtered_mean, filtered_var, _n = forward[index]
        _next_time, next_predicted_mean, next_predicted_var, _fm, _fv, _nn = forward[index + 1]
        gain = filtered_var / next_predicted_var
        smoothed_mean = filtered_mean + gain * (smoothed_mean - next_predicted_mean)
        smoothed_var = filtered_var + gain ** 2 * (smoothed_var - next_predicted_var)
        backward.append((smoothed_mean, smoothed_var))
    backward.reverse()

    nodes = [
        {
            "time": time,
            "filtered": filtered_mean,
            "filtered_var": filtered_var,
            "smoothed": backward[index][0],
            "smoothed_var": backward[index][1],
            "n": count,
        }
        for index, (time, _pm, _pv, filtered_mean, filtered_var, count) in enumerate(forward)
    ]
    return nodes, log_likelihood


def pooled_log_likelihood(series_by_character, q, sigma2, prior_mean, prior_sd=DEFAULT_PRIOR_SD):
    """Sum of every character's marginal log-likelihood under one (q, sigma2)."""
    return sum(
        filter_and_smooth(series, q, sigma2, prior_mean=prior_mean, prior_sd=prior_sd)[1]
        for series in series_by_character.values()
    )


def select_hyperparameters(
    series_by_character,
    prior_mean,
    q_grid=DEFAULT_Q_GRID,
    sigma2_grid=DEFAULT_SIGMA2_GRID,
    prior_sd=DEFAULT_PRIOR_SD,
):
    """Grid-search (q, sigma2) by pooled marginal likelihood.

    Returns `(best_q, best_sigma2, table)` where `table` lists every grid
    point's log-likelihood, and also the log-likelihood of q -> 0 (a flat,
    arc-free level) at the chosen sigma2, so a report can say how much the
    data prefer arcs to no arcs at all.
    """
    table = []
    for q in q_grid:
        for sigma2 in sigma2_grid:
            table.append(
                {
                    "q": q,
                    "sigma2": sigma2,
                    "log_likelihood": pooled_log_likelihood(
                        series_by_character, q, sigma2, prior_mean, prior_sd=prior_sd
                    ),
                }
            )
    best = max(table, key=lambda row: row["log_likelihood"])
    return best["q"], best["sigma2"], table


def arc_summary(nodes):
    """Numbers a reader can hold onto: start, end, peak, trough, biggest fall and rise.

    `biggest_fall` is the largest drop from any earlier smoothed level to
    any later one (a peak-to-trough drawdown); `biggest_rise` is its mirror.
    `sd` values are the smoothed posterior standard deviations at the
    corresponding nodes, so every claim carries its own uncertainty.
    """
    if not nodes:
        return None
    levels = [node["smoothed"] for node in nodes]
    sds = [math.sqrt(node["smoothed_var"]) for node in nodes]

    def extreme(pick):
        index = pick(range(len(levels)), key=lambda i: levels[i])
        return {"time": nodes[index]["time"], "level": levels[index], "sd": sds[index]}

    fall = {"size": 0.0}
    rise = {"size": 0.0}
    high_index = low_index = 0
    for index in range(1, len(levels)):
        if levels[index] > levels[high_index]:
            high_index = index
        if levels[index] < levels[low_index]:
            low_index = index
        drop = levels[high_index] - levels[index]
        if drop > fall["size"]:
            fall = {"size": drop, "from_time": nodes[high_index]["time"], "to_time": nodes[index]["time"],
                    "from_level": levels[high_index], "to_level": levels[index]}
        gain = levels[index] - levels[low_index]
        if gain > rise["size"]:
            rise = {"size": gain, "from_time": nodes[low_index]["time"], "to_time": nodes[index]["time"],
                    "from_level": levels[low_index], "to_level": levels[index]}

    return {
        "start": {"time": nodes[0]["time"], "level": levels[0], "sd": sds[0]},
        "end": {"time": nodes[-1]["time"], "level": levels[-1], "sd": sds[-1]},
        "peak": extreme(max),
        "trough": extreme(min),
        "biggest_fall": fall,
        "biggest_rise": rise,
        "net": levels[-1] - levels[0],
    }


def order_permutation_p_values(observations, q, sigma2, prior_mean, rng, samples=300, prior_sd=DEFAULT_PRIOR_SD):
    """How often shuffling a character's outcomes in time gives an arc as big as theirs.

    Keeps the character's appearance times and their multiset of
    (movement, weight) outcomes, and permutes which outcome lands at which
    time. A real arc depends on ORDER -- the good passages early, the bad
    ones late -- so a fall that shuffled orders rarely match is a fall
    the novel's sequence put there, not a large number the character's
    volatility would produce anyway. Returns `{"fall": p, "rise": p}` with
    the usual +1 correction, so p is never zero. `rng` is a
    `random.Random`, passed in so builds are reproducible.
    """
    def moves(rows):
        nodes, _ = filter_and_smooth(rows, q, sigma2, prior_mean=prior_mean, prior_sd=prior_sd)
        summary = arc_summary(nodes)
        return summary["biggest_fall"]["size"], summary["biggest_rise"]["size"]

    observed_fall, observed_rise = moves(observations)
    times = [time for time, _value, _weight in observations]
    outcomes = [(value, weight) for _time, value, weight in observations]
    fall_hits = rise_hits = 0
    for _sample in range(samples):
        rng.shuffle(outcomes)
        fall, rise = moves([(time, value, weight) for time, (value, weight) in zip(times, outcomes)])
        fall_hits += fall >= observed_fall
        rise_hits += rise >= observed_rise
    return {"fall": (fall_hits + 1) / (samples + 1), "rise": (rise_hits + 1) / (samples + 1)}


__all__ = [
    "DEFAULT_PRIOR_SD",
    "DEFAULT_Q_GRID",
    "DEFAULT_SIGMA2_GRID",
    "FORTUNE_LENS_ORDER",
    "FORTUNE_RATING_CENTER",
    "NAME_KEY_PREFIX",
    "PersonKeyer",
    "OVERALL_LENS",
    "WORDS_PER_TIME_UNIT",
    "arc_summary",
    "character_series",
    "elo_points_per_level",
    "fortune_rating",
    "filter_and_smooth",
    "order_permutation_p_values",
    "pooled_log_likelihood",
    "select_hyperparameters",
    "unit_observations",
    "unit_time",
]
