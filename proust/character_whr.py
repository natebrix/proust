"""Whole-History Rating over the ISLT character corpus.

`proust/whr.py` holds the corpus-agnostic math; this module supplies the
corpus. It reuses the match definition every other rating surface here
uses -- `iter_character_lens_matches`, i.e. within each unit every pair
of co-scored characters compared on one lens's net score, with
differences of at most `epsilon` counted as draws -- and gives each match
a narrative TIME: the unit's `cumulative_unit_index`, the same corpus
position index the ELO timeline plots against.

Two rating surfaces come out of the same model:

- **smoothed**: one MAP fit over the whole corpus. Every point of a
  character's trajectory is informed by the entire novel, before and
  after. This is Coulom's WHR proper, and it is the right thing to read
  when asking "what was this character's standing in volume 3".
- **filtered**: one fit per match-bearing time, using only matches up to
  that time. Each point sees only the past, which is what makes it
  comparable to ELO and Glicko-2 (both strictly forward) and what makes
  it usable for honest one-step-ahead prediction.

The gap between the two is itself informative: where filtered and
smoothed diverge, the novel later revised its own verdict on a character.

Because `w2` (how fast a character's standing is allowed to drift) is a
modeling choice rather than a fact, it is chosen by out-of-sample
prediction rather than asserted: several candidates are scored by
sequential one-step-ahead log-loss, alongside the existing sequential
ELO and Glicko-2 surfaces, and the winner becomes the default.
"""

from collections import defaultdict
import math
from time import perf_counter

from . import glicko2
from . import runner
from . import whr
from .app_exports import (
    _build_corpus_position_index,
    _elo_expected_score,
    _require_known_scoring_lens,
    build_chapter_overlay_data,
    iter_character_lens_matches,
)

DEFAULT_W2_ELO_CANDIDATES = (5.0, 15.0, 35.0, 60.0)
# Mirrors Glicko-2's "provisional means RD > 100": a band is 2 sigma, so
# a band above 200 Elo is the same statement about the same quantity.
DEFAULT_BAND_PROVISIONAL_THRESHOLD = 200.0
DEFAULT_ELO_K_FACTOR = 24.0
DEFAULT_ELO_INITIAL_RATING = 1500.0
TIME_AXIS = "cumulative_unit_index"

# Predicted probabilities are pushed inside the open unit interval before
# taking a log. No system compared here can actually produce 0 or 1, so
# this only guards against underflow, never against a real prediction.
PROBABILITY_FLOOR = 1e-15


def _character_whr_version(lens):
    return f"character_whr_{lens}_v1"


# ---------------------------------------------------------------------------
# Predictive scoring, shared by all three systems being compared.
# ---------------------------------------------------------------------------


def _new_scores():
    return {"log_loss_total": 0.0, "brier_total": 0.0, "match_count": 0}


def _accumulate_prediction(scores, probability_a, observed_a):
    """Score one prediction, using WHR's half-win draw convention for every system.

    A draw is scored as half a win plus half a loss, exactly as the WHR
    likelihood models it. For log-loss that is the usual cross-entropy
    against a target of 0.5; for Brier it adds a constant 0.25 per draw
    relative to scoring against a target of 0.5, which is identical for
    every system and so does not affect the comparison.
    """
    probability_a = min(max(probability_a, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
    if observed_a == 0.5:
        scores["log_loss_total"] += -0.5 * math.log(probability_a) - 0.5 * math.log(1.0 - probability_a)
        scores["brier_total"] += 0.5 * (probability_a - 1.0) ** 2 + 0.5 * probability_a ** 2
    elif observed_a == 1.0:
        scores["log_loss_total"] += -math.log(probability_a)
        scores["brier_total"] += (probability_a - 1.0) ** 2
    else:
        scores["log_loss_total"] += -math.log(1.0 - probability_a)
        scores["brier_total"] += probability_a ** 2
    scores["match_count"] += 1


def _finish_scores(scores, system, description):
    count = scores["match_count"]
    return {
        "system": system,
        "description": description,
        "match_count": count,
        "log_loss": round(scores["log_loss_total"] / count, 6) if count else None,
        "brier": round(scores["brier_total"] / count, 6) if count else None,
    }


def _evaluate_sequential_elo(matches, k_factor, initial_rating):
    """One-step-ahead scores for the existing sequential ELO surface, two ways.

    The first reproduces `build_character_elo` exactly -- same initial
    rating, same K, same strictly sequential order -- and records the
    expected score it computes immediately before each update. That is
    genuinely the information sequential ELO has, but it includes the
    OTHER pairings of the same unit, which are strongly correlated with
    the pairing being predicted (every pairing in a unit is driven by the
    same net scores). So a second variant is scored alongside it, in
    which predictions are frozen at the unit boundary while the updates
    stay sequential; that is the like-for-like comparison against filtered
    WHR and chapter-period Glicko-2, both of which are frozen at their own
    boundaries.

    Returns `(sequential_evaluation, unit_frozen_evaluation)`.
    """
    ratings = {}
    sequential_scores = _new_scores()
    unit_frozen_scores = _new_scores()
    frozen_ratings = {}
    frozen_time = None

    for match in matches:
        character_a = match["character_a"]
        character_b = match["character_b"]
        ratings.setdefault(character_a, float(initial_rating))
        ratings.setdefault(character_b, float(initial_rating))

        if match["time"] != frozen_time:
            frozen_time = match["time"]
            frozen_ratings = dict(ratings)
        _accumulate_prediction(
            unit_frozen_scores,
            _elo_expected_score(
                frozen_ratings.get(character_a, float(initial_rating)),
                frozen_ratings.get(character_b, float(initial_rating)),
            ),
            match["observed_a"],
        )

        rating_a = ratings[character_a]
        rating_b = ratings[character_b]
        expected_a = _elo_expected_score(rating_a, rating_b)
        expected_b = _elo_expected_score(rating_b, rating_a)
        _accumulate_prediction(sequential_scores, expected_a, match["observed_a"])
        ratings[character_a] = rating_a + k_factor * (match["observed_a"] - expected_a)
        ratings[character_b] = rating_b + k_factor * (match["observed_b"] - expected_b)

    sequential_evaluation = _finish_scores(
        sequential_scores,
        "elo_sequential",
        f"sequential ELO, K={k_factor:g}, expected score from the pre-match ratings",
    )
    unit_frozen_evaluation = _finish_scores(
        unit_frozen_scores,
        "elo_unit_frozen",
        f"sequential ELO, K={k_factor:g}, expected score frozen at the unit boundary",
    )
    return sequential_evaluation, unit_frozen_evaluation


def _evaluate_glicko2(
    chapter_characters, matches_by_chapter, tau, initial_rating, initial_rd, initial_volatility, convergence_epsilon
):
    """One-step-ahead scores for the existing chapter-period Glicko-2 surface.

    Glicko-2's own prediction for a match is `E(mu, mu_j, phi_j)` against
    the opponent's PRE-period state, which for `build_character_glicko2`
    means the state frozen at the chapter boundary. That is exactly the
    information the system has before the chapter is played, so it is
    scored here without any peeking.

    `chapter_characters` is the canonical-order list of
    `(chapter_id, characters appearing in that chapter)` pairs, which is
    what `build_character_glicko2` uses to decide who is tracked (and
    therefore whose RD grows) in each period.
    """
    state = {}
    scores = _new_scores()

    for chapter_id, characters in chapter_characters:
        for character in characters:
            state.setdefault(
                character,
                {
                    "rating": float(initial_rating),
                    "rd": float(initial_rd),
                    "volatility": float(initial_volatility),
                },
            )
        pre_period_state = {name: dict(values) for name, values in state.items()}

        opponents_by_character = defaultdict(list)
        for match in matches_by_chapter.get(chapter_id, []):
            character_a = match["character_a"]
            character_b = match["character_b"]
            own = pre_period_state[character_a]
            opponent_a = pre_period_state[character_b]
            opponent_b = own
            mu_a, _phi_a = glicko2.to_glicko2_scale(own["rating"], own["rd"])
            mu_b, phi_b = glicko2.to_glicko2_scale(opponent_a["rating"], opponent_a["rd"])
            _accumulate_prediction(scores, glicko2.expected_score(mu_a, mu_b, phi_b), match["observed_a"])
            opponents_by_character[character_a].append((opponent_a["rating"], opponent_a["rd"], match["observed_a"]))
            opponents_by_character[character_b].append((opponent_b["rating"], opponent_b["rd"], match["observed_b"]))

        new_state = {}
        for character, values in state.items():
            rating_prime, rd_prime, volatility_prime = glicko2.rate_player(
                values["rating"],
                values["rd"],
                values["volatility"],
                opponents_by_character.get(character, []),
                tau=tau,
                convergence_epsilon=convergence_epsilon,
            )
            new_state[character] = {"rating": rating_prime, "rd": rd_prime, "volatility": volatility_prime}
        state = new_state

    return _finish_scores(
        scores,
        "glicko2_chapter_period",
        "Glicko-2 E(mu, mu_j, phi_j) against opponents' state frozen at the chapter boundary",
    )


# ---------------------------------------------------------------------------
# The two WHR surfaces.
# ---------------------------------------------------------------------------


def _filtered_pass(times, games_by_time, matches_by_time, w2_elo, initial_rd, max_sweeps, tolerance):
    """Fit WHR repeatedly on growing prefixes of narrative time.

    At each match-bearing time, every match at that time is first
    predicted from the fit that has seen only strictly earlier matches
    (a character never seen before predicts from the 1500 prior), and
    only then is the time's evidence folded in. Each fit warm-starts from
    the previous one, which is what keeps this affordable.
    """
    scores = _new_scores()
    deflated_scores = _new_scores()
    trajectories = defaultdict(list)
    prior = (0.0, whr.variance_from_rd(initial_rd) if initial_rd is not None else 0.0)
    latest_node = {}
    warm_start = None
    games_so_far = []
    sweep_total = 0
    max_sweep_count = 0
    non_converged_fits = 0

    started = perf_counter()
    for time_point in times:
        for match in matches_by_time[time_point]:
            log_gamma_a, variance_a = latest_node.get(match["character_a"], prior)
            log_gamma_b, variance_b = latest_node.get(match["character_b"], prior)
            _accumulate_prediction(scores, whr.win_probability(log_gamma_a, log_gamma_b), match["observed_a"])
            # The plug-in prediction above ignores how uncertain the two
            # ratings are. Deflating the rating difference by Glicko's
            # g() against the summed posterior variance is the standard
            # logistic-normal approximation to integrating the
            # Bradley-Terry probability over that uncertainty, and is the
            # like-for-like counterpart of Glicko-2's own E().
            deflation = 1.0 / math.sqrt(1.0 + 3.0 * (variance_a + variance_b) / math.pi ** 2)
            _accumulate_prediction(
                deflated_scores, whr.logistic(deflation * (log_gamma_a - log_gamma_b)), match["observed_a"]
            )

        games_so_far.extend(games_by_time[time_point])
        result = whr.fit(
            games_so_far,
            w2_elo,
            initial_rd=initial_rd,
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            warm_start=warm_start,
        )
        warm_start = whr.warm_start_from(result)
        sweep_total += result["sweep_count"]
        max_sweep_count = max(max_sweep_count, result["sweep_count"])
        if not result["converged"]:
            non_converged_fits += 1

        for character, nodes in result["players"].items():
            node = nodes[-1]
            latest_node[character] = (node["log_gamma"], node["variance"])
            if node["time"] == time_point:
                trajectories[character].append(
                    [time_point, round(node["rating"], 1), round(node["band"], 1)]
                )

    return {
        "scores": scores,
        "deflated_scores": deflated_scores,
        "trajectories": dict(trajectories),
        "latest_node": latest_node,
        "fit_count": len(times),
        "sweep_total": sweep_total,
        "max_sweep_count": max_sweep_count,
        "non_converged_fits": non_converged_fits,
        "wall_clock_seconds": round(perf_counter() - started, 3),
    }


def _trajectory_summary(points):
    """First / last / lowest / highest point of a trajectory, for the Markdown view."""
    if not points:
        return None
    lowest = min(points, key=lambda point: (point[1], point[0]))
    highest = max(points, key=lambda point: (point[1], -point[0]))
    return {
        "point_count": len(points),
        "first": points[0],
        "last": points[-1],
        "min": lowest,
        "max": highest,
    }


def build_character_whr(
    run_dirs,
    lens="advantage",
    supplement_run_dirs=None,
    w2_elo=None,
    epsilon=0.25,
    mode="both",
    w2_elo_candidates=DEFAULT_W2_ELO_CANDIDATES,
    band_provisional_threshold=DEFAULT_BAND_PROVISIONAL_THRESHOLD,
    initial_rd=whr.DEFAULT_INITIAL_RD,
    max_sweeps=whr.DEFAULT_MAX_SWEEPS,
    tolerance=whr.DEFAULT_TOLERANCE,
    elo_k_factor=DEFAULT_ELO_K_FACTOR,
    glicko2_tau=glicko2.DEFAULT_TAU,
):
    """Build a lens Whole-History Rating surface: trajectories, bands, and a referee.

    `w2_elo` is the Wiener drift rate in Elo^2 per unit of narrative time.
    Left as None it is CHOSEN, not asserted: every candidate in
    `w2_elo_candidates` is scored by sequential one-step-ahead log-loss
    and the best one becomes the artifact's rating. All candidates' scores
    are kept in the artifact, next to the same scores for the existing
    sequential ELO and chapter-period Glicko-2 surfaces, so the claim that
    this surface predicts better (or does not) is checkable.

    `mode` is "smoothed", "filtered", or "both" and controls only which
    trajectories are written; the filtered pass always runs, because the
    predictive evaluation depends on it.
    """
    _require_known_scoring_lens(lens)
    if not run_dirs:
        raise ValueError("At least one run directory is required for character WHR.")
    if mode not in ("smoothed", "filtered", "both"):
        raise ValueError(f'Unknown mode "{mode}". Expected one of: smoothed, filtered, both.')

    review = runner.build_corpus_sanity_review(run_dirs)
    overlay_dataset = build_chapter_overlay_data(run_dirs, supplement_run_dirs=supplement_run_dirs)
    units_by_chapter = {chapter["chapterId"]: chapter["units"] for chapter in overlay_dataset["chapters"]}
    corpus_positions = _build_corpus_position_index(units_by_chapter)

    diagnostics = {}
    chapter_characters = []
    for chapter in overlay_dataset["chapters"]:
        appearing = []
        seen_in_chapter = set()
        for unit in chapter["units"]:
            for character_row in unit["characters"]:
                character = character_row["character"]
                if character not in seen_in_chapter:
                    seen_in_chapter.add(character)
                    appearing.append(character)
                row = diagnostics.setdefault(
                    character,
                    {
                        "character": character,
                        "match_count": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "draw_count": 0,
                        "unit_count": 0,
                        "net_scores": [],
                    },
                )
                row["unit_count"] += 1
                row["net_scores"].append(character_row[lens]["netScore"])
        chapter_characters.append((chapter["chapterId"], appearing))

    matches = []
    for match in iter_character_lens_matches(overlay_dataset, lens, epsilon):
        position = corpus_positions[match["unit_id"]]
        match["time"] = position["cumulative_unit_index"]
        matches.append(match)

        character_a = match["character_a"]
        character_b = match["character_b"]
        diagnostics[character_a]["match_count"] += 1
        diagnostics[character_b]["match_count"] += 1
        if match["observed_a"] == 0.5:
            diagnostics[character_a]["draw_count"] += 1
            diagnostics[character_b]["draw_count"] += 1
        elif match["observed_a"] == 1.0:
            diagnostics[character_a]["win_count"] += 1
            diagnostics[character_b]["loss_count"] += 1
        else:
            diagnostics[character_a]["loss_count"] += 1
            diagnostics[character_b]["win_count"] += 1

    if not matches:
        raise ValueError("No pairwise character matches were found; WHR needs at least one match.")

    games_by_time = defaultdict(list)
    matches_by_time = defaultdict(list)
    matches_by_chapter = defaultdict(list)
    for match in matches:
        games_by_time[match["time"]].append(
            (match["character_a"], match["character_b"], match["time"], match["observed_a"])
        )
        matches_by_time[match["time"]].append(match)
        matches_by_chapter[match["chapter_id"]].append(match)
    times = sorted(matches_by_time)

    games = [game for time_point in times for game in games_by_time[time_point]]
    draw_count = sum(1 for match in matches if match["observed_a"] == 0.5)

    elo_evaluation, elo_unit_frozen_evaluation = _evaluate_sequential_elo(
        matches, elo_k_factor, DEFAULT_ELO_INITIAL_RATING
    )
    glicko2_evaluation = _evaluate_glicko2(
        chapter_characters,
        matches_by_chapter,
        glicko2_tau,
        glicko2.DEFAULT_INITIAL_RATING,
        glicko2.DEFAULT_INITIAL_RD,
        glicko2.DEFAULT_INITIAL_VOLATILITY,
        glicko2.DEFAULT_CONVERGENCE_EPSILON,
    )

    candidates = [float(w2_elo)] if w2_elo is not None else [float(value) for value in w2_elo_candidates]
    candidate_evaluations = []
    passes_by_candidate = {}
    for candidate in candidates:
        filtered = _filtered_pass(
            times, games_by_time, matches_by_time, candidate, initial_rd, max_sweeps, tolerance
        )
        passes_by_candidate[candidate] = filtered
        evaluation = _finish_scores(
            filtered["scores"],
            "whr_filtered",
            f"filtered WHR at w2={candidate:g} Elo^2 per unit, previous node's rating",
        )
        evaluation["w2_elo"] = candidate
        evaluation["wall_clock_seconds"] = filtered["wall_clock_seconds"]
        candidate_evaluations.append(evaluation)

    best_evaluation = min(candidate_evaluations, key=lambda item: (item["log_loss"], item["w2_elo"]))
    chosen_w2_elo = best_evaluation["w2_elo"]
    filtered_pass = passes_by_candidate[chosen_w2_elo]
    deflated_evaluation = _finish_scores(
        filtered_pass["deflated_scores"],
        "whr_filtered_deflated",
        f"filtered WHR at w2={chosen_w2_elo:g}, previous node's rating deflated by its posterior variance",
    )
    deflated_evaluation["w2_elo"] = chosen_w2_elo

    smoothed_started = perf_counter()
    smoothed = whr.fit(
        games, chosen_w2_elo, initial_rd=initial_rd, max_sweeps=max_sweeps, tolerance=tolerance
    )
    smoothed_wall_clock = round(perf_counter() - smoothed_started, 3)

    smoothed_trajectories = {
        character: [[node["time"], round(node["rating"], 1), round(node["band"], 1)] for node in nodes]
        for character, nodes in smoothed["players"].items()
    }
    filtered_trajectories = filtered_pass["trajectories"]

    characters = []
    for character in sorted(diagnostics):
        row = diagnostics[character]
        smoothed_points = smoothed_trajectories.get(character, [])
        filtered_points = filtered_trajectories.get(character, [])
        if smoothed_points:
            rating = smoothed_points[-1][1]
            band = smoothed_points[-1][2]
        else:
            # A character who is never co-scored with anyone plays no
            # matches and so has no node; their posterior is the prior.
            rating = round(whr.DEFAULT_INITIAL_RATING, 1)
            band = round(2.0 * initial_rd, 1) if initial_rd is not None else None
        mean_net_score = (
            round(sum(row["net_scores"]) / len(row["net_scores"]), 3) if row["net_scores"] else 0.0
        )
        character_row = {
            "character": character,
            "rating": rating,
            "band": band,
            "sigma": round(band / 2.0, 1) if band is not None else None,
            "conservative_rating": round(rating - band, 1) if band is not None else None,
            "match_count": row["match_count"],
            "win_count": row["win_count"],
            "loss_count": row["loss_count"],
            "draw_count": row["draw_count"],
            "unit_count": row["unit_count"],
            "node_count": len(smoothed_points),
            "mean_net_score": mean_net_score,
            "provisional": band is None or band > band_provisional_threshold,
            "first_time": smoothed_points[0][0] if smoothed_points else None,
            "last_time": smoothed_points[-1][0] if smoothed_points else None,
            "smoothed_summary": _trajectory_summary(smoothed_points),
        }
        if mode in ("smoothed", "both"):
            character_row["smoothed_trajectory"] = smoothed_points
        if mode in ("filtered", "both"):
            character_row["filtered_trajectory"] = filtered_points
            character_row["filtered_summary"] = _trajectory_summary(filtered_points)
        characters.append(character_row)

    characters.sort(
        key=lambda item: (
            -(item["conservative_rating"] if item["conservative_rating"] is not None else -1e9),
            item["character"],
        )
    )
    non_provisional = [row for row in characters if not row["provisional"]]
    provisional = [row for row in characters if row["provisional"]]

    time_index = []
    for time_point in times:
        unit_id = matches_by_time[time_point][0]["unit_id"]
        position = corpus_positions[unit_id]
        time_index.append(
            {
                "time": time_point,
                "unit_id": unit_id,
                "chapter_id": position["chapter_id"],
                "chapter_index": position["chapter_index"],
                "volume_number": position["volume_number"],
                "unit_index_within_chapter": position["unit_index_within_chapter"],
            }
        )

    predictive_comparison = [
        best_evaluation,
        deflated_evaluation,
        elo_evaluation,
        elo_unit_frozen_evaluation,
        glicko2_evaluation,
    ]

    result = {
        "character_whr_version": _character_whr_version(lens),
        "lens": lens,
        "source_review_version": review["corpus_review_version"],
        "duplicate_resolution": overlay_dataset["duplicate_resolution"],
        "mode": mode,
        "time_axis": TIME_AXIS,
        "w2_elo": chosen_w2_elo,
        "w2_elo_selected_by": "sequential_one_step_ahead_log_loss" if w2_elo is None else "caller",
        "w2_elo_candidates": candidates,
        "epsilon": epsilon,
        "draw_model": whr.DRAW_MODEL,
        "initial_rating": whr.DEFAULT_INITIAL_RATING,
        "initial_rd": initial_rd,
        "band_provisional_threshold": band_provisional_threshold,
        "conservative_rating_rule": "rating_minus_band",
        "character_count": len(characters),
        "match_count": len(matches),
        "draw_rate": round(draw_count / len(matches), 3),
        "time_point_count": len(times),
        "node_count": sum(len(points) for points in smoothed_trajectories.values()),
        "convergence": {
            "tolerance": tolerance,
            "max_sweeps": max_sweeps,
            "smoothed_sweep_count": smoothed["sweep_count"],
            "smoothed_converged": smoothed["converged"],
            "smoothed_max_log_gamma_change": smoothed["max_log_gamma_change"],
            "filtered_fit_count": filtered_pass["fit_count"],
            "filtered_sweep_total": filtered_pass["sweep_total"],
            "filtered_max_sweep_count": filtered_pass["max_sweep_count"],
            "filtered_non_converged_fits": filtered_pass["non_converged_fits"],
        },
        "wall_clock_seconds": {
            "smoothed": smoothed_wall_clock,
            "filtered": filtered_pass["wall_clock_seconds"],
            "filtered_all_candidates": round(
                sum(item["wall_clock_seconds"] for item in candidate_evaluations), 3
            ),
        },
        "predictive_evaluation": {
            "protocol": (
                "sequential one-step-ahead over all matches in narrative order; each match is "
                "predicted from prior information only, and draws are scored as half a win plus "
                "half a loss for every system. Systems freeze at different boundaries: filtered "
                "WHR at the unit, Glicko-2 at the chapter, and sequential ELO at the individual "
                "match -- so elo_sequential alone can see the other pairings of the unit it is "
                "predicting, which are driven by the same net scores. elo_unit_frozen is the "
                "like-for-like row"
            ),
            "comparison": predictive_comparison,
            "whr_candidates": candidate_evaluations,
        },
        "top_rated_characters": non_provisional[:10],
        "bottom_rated_characters": sorted(
            non_provisional, key=lambda item: (item["conservative_rating"], item["character"])
        )[:10],
        "provisional_characters": sorted(provisional, key=lambda item: (-item["rating"], item["character"])),
        "times": time_index,
        "characters": characters,
    }

    if supplement_run_dirs:
        result["supplemented"] = True
        result["supplement_runs"] = overlay_dataset.get("supplement_runs", [])

    return result


__all__ = [
    "DEFAULT_BAND_PROVISIONAL_THRESHOLD",
    "DEFAULT_ELO_K_FACTOR",
    "DEFAULT_W2_ELO_CANDIDATES",
    "TIME_AXIS",
    "build_character_whr",
]
