"""Scoring v2 over the foundation corpus: comparisons, ratings, artifacts.

`proust/scoring_v2.py` is the formula and knows nothing outside one
annotation. This module supplies the corpus: it reads the foundation runs,
gives every unit its narrative time (`cumulative_unit_index`, the same axis
the v1 rating surfaces use), turns the units into weighted comparisons, and
fits Whole-History Rating over them with `proust/whr.py`'s weighted games.

Two keyings come off one comparison, so both are fitted:

- the **name view** keys on the annotation's canonical name, which is what
  a reader sees on the page ("le peintre" is not yet Elstir);
- the **person view** keys on registry entity ids with `person_view_merge`
  links applied, which is what a biographer counts ("le peintre" IS
  Elstir, the prince des Laumes IS the duc de Guermantes).

Nothing here writes outside `outputs/scoring-v2/`: v2 is staged for the
adoption gate described in the design doc, not adopted.
"""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

from . import glicko2
from . import runner
from . import scoring_v2 as v2
from . import whr
from .app_exports import (
    _build_corpus_position_index,
    _chapter_id_from_unit_id,
    _paragraph_range_from_unit_id,
    _run_id_sort_key,
    foundation_corpus_label,
)
from .character_whr import (
    DEFAULT_BAND_PROVISIONAL_THRESHOLD,
    DEFAULT_ELO_INITIAL_RATING,
    DEFAULT_ELO_K_FACTOR,
    DEFAULT_W2_ELO_CANDIDATES,
    TIME_AXIS,
    TIMELINE_MODES,
    _evaluate_glicko2,
    _evaluate_sequential_elo,
    _filtered_pass,
    _finish_scores,
    _trajectory_summary,
)
from .editorial import CHARACTER_PAGE_PILOT_EDITORIAL
from .registry import Registry

SCORING_V2_BUILD_VERSION = "scoring_v2_build_v1"
VIEWS = ("name", "person")
DEFAULT_OUTPUT_DIR = "outputs/scoring-v2"


# ---------------------------------------------------------------------------
# The corpus: reviewed annotations in narrative order.
# ---------------------------------------------------------------------------


def load_scored_units(run_dirs):
    """Every reviewed annotation of the corpus, in narrative order, with its time.

    A unit annotated by more than one run resolves latest-run-wins, the
    same rule `build_chapter_overlay_data` applies, and the narrative time
    is `_build_corpus_position_index`'s `cumulative_unit_index` over the
    units that survive it -- so v2 sits on exactly the corpus axis the v1
    surfaces sit on and the two are directly comparable.
    """
    preferred_run_by_unit = {}
    annotation_dir_by_run = {}
    for run_dir in run_dirs:
        status = runner.get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        annotation_dir_by_run[run_id] = Path(status["manifest"]["directories"]["annotations"])
        for unit in status["units"]:
            if unit["review_state"] != "reviewed":
                continue
            unit_id = unit["unit_id"]
            incumbent = preferred_run_by_unit.get(unit_id)
            if incumbent is None or _run_id_sort_key(run_id) > _run_id_sort_key(incumbent):
                preferred_run_by_unit[unit_id] = run_id

    if not preferred_run_by_unit:
        raise ValueError("No reviewed annotations were found in the given run directories.")

    units_by_chapter = defaultdict(list)
    for unit_id in preferred_run_by_unit:
        paragraph_start, paragraph_end = _paragraph_range_from_unit_id(unit_id)
        units_by_chapter[_chapter_id_from_unit_id(unit_id)].append(
            {"unitId": unit_id, "paragraphStart": paragraph_start, "paragraphEnd": paragraph_end}
        )
    for rows in units_by_chapter.values():
        rows.sort(key=lambda row: (row["paragraphStart"], row["paragraphEnd"], row["unitId"]))

    positions = _build_corpus_position_index(units_by_chapter)

    units = []
    for unit_id, run_id in preferred_run_by_unit.items():
        annotation_path = annotation_dir_by_run[run_id] / f"{unit_id}.json"
        position = positions[unit_id]
        units.append(
            {
                "unit_id": unit_id,
                "run_id": run_id,
                "chapter_id": position["chapter_id"],
                "chapter_index": position["chapter_index"],
                "volume_number": position["volume_number"],
                "unit_index_within_chapter": position["unit_index_within_chapter"],
                "time": position["cumulative_unit_index"],
                "annotation": json.loads(annotation_path.read_text()),
            }
        )
    units.sort(key=lambda unit: unit["time"])
    return units


# ---------------------------------------------------------------------------
# Comparisons and readings.
# ---------------------------------------------------------------------------


def build_comparisons(units, lens, registry=None, merge_map=None):
    """Every v2 comparison of the corpus for one lens, in narrative order.

    Each row is `scoring_v2.unit_comparisons`'s, plus the unit's narrative
    time and chapter id. Both keyings ride along on every row; picking one
    is `view_matches`'s job.
    """
    v2.require_known_lens(lens)
    if merge_map is None and registry is not None:
        merge_map = v2.person_view_merge_map(registry)

    comparisons = []
    for unit in units:
        for row in v2.unit_comparisons(
            unit["annotation"],
            lens,
            unit_id=unit["unit_id"],
            registry=registry,
            merge_map=merge_map,
            chapter_id=unit["chapter_id"],
        ):
            row["time"] = unit["time"]
            row["chapter_id"] = unit["chapter_id"]
            comparisons.append(row)
    return comparisons


def build_readings(units, lens, registry=None, merge_map=None):
    """Per unit, per character: the movement and the label a reader would see."""
    v2.require_known_lens(lens)
    if merge_map is None and registry is not None:
        merge_map = v2.person_view_merge_map(registry)

    readings = []
    for unit in units:
        movements = v2.unit_movements(unit["annotation"], lens)
        labels = v2.unit_labels(unit["annotation"], lens)
        confidences = v2.unit_confidences(unit["annotation"], lens)
        for character in sorted(movements):
            readings.append(
                {
                    "unit_id": unit["unit_id"],
                    "chapter_id": unit["chapter_id"],
                    "time": unit["time"],
                    "character": character,
                    "person": v2.person_view_key(
                        character, registry=registry, merge_map=merge_map, chapter_id=unit["chapter_id"]
                    ),
                    "movement": round(movements[character], 6),
                    "label": labels[character],
                    "confidence": round(confidences[character], 6),
                }
            )
    return readings


def view_keys(view):
    if view == "name":
        return "character_a", "character_b"
    if view == "person":
        return "person_a", "person_b"
    raise ValueError(f'Unknown view "{view}". Expected one of: {", ".join(VIEWS)}.')


def view_matches(comparisons, view):
    """Comparisons re-keyed for one view, with person-view self-pairings dropped.

    A unit naming both "le peintre" and Elstir compares two names that the
    person view holds to be one man; the rating layer cannot play a player
    against itself, and the comparison carries no information about anyone
    else, so it is dropped here and counted in `dropped_self_pairings`.
    """
    key_a, key_b = view_keys(view)
    matches = []
    dropped = 0
    for row in comparisons:
        player_a = row[key_a]
        player_b = row[key_b]
        if player_a == player_b:
            dropped += 1
            continue
        matches.append(
            {
                "unit_id": row["unit_id"],
                "chapter_id": row["chapter_id"],
                "time": row["time"],
                "character_a": player_a,
                "character_b": player_b,
                "observed_a": row["observed_a"],
                "observed_b": row["observed_b"],
                "weight": row["weight"],
                "movement_a": row["movement_a"],
                "movement_b": row["movement_b"],
            }
        )
    return matches, dropped


def view_reading_key(reading, view):
    return reading["character"] if view == "name" else reading["person"]


# ---------------------------------------------------------------------------
# The rating fit, per lens, per view.
# ---------------------------------------------------------------------------


def _chapter_characters(readings, view):
    """(chapter_id, characters appearing) in canonical order, for the Glicko baseline."""
    ordered = []
    seen_by_chapter = {}
    for reading in sorted(readings, key=lambda row: (row["time"], row["character"])):
        chapter_id = reading["chapter_id"]
        if chapter_id not in seen_by_chapter:
            seen_by_chapter[chapter_id] = set()
            ordered.append((chapter_id, []))
        character = view_reading_key(reading, view)
        if character not in seen_by_chapter[chapter_id]:
            seen_by_chapter[chapter_id].add(character)
            ordered[-1][1].append(character)
    return ordered


def fit_view(
    matches,
    readings,
    view,
    lens,
    w2_elo=None,
    w2_elo_candidates=DEFAULT_W2_ELO_CANDIDATES,
    band_provisional_threshold=DEFAULT_BAND_PROVISIONAL_THRESHOLD,
    initial_rd=whr.DEFAULT_INITIAL_RD,
    max_sweeps=whr.DEFAULT_MAX_SWEEPS,
    tolerance=whr.DEFAULT_TOLERANCE,
    elo_k_factor=DEFAULT_ELO_K_FACTOR,
    glicko2_tau=glicko2.DEFAULT_TAU,
    weight_floor=0.0,
):
    """Fit smoothed and filtered WHR over one view's weighted comparisons.

    The shape of the returned artifact deliberately mirrors
    `character_whr.build_character_whr`, so the two can be diffed row for
    row, but every game here carries its v2 weight and the predictive
    numbers are computed on v2 comparisons.

    `w2_elo` left None is chosen the same way v1 chooses it: each
    candidate is scored by one-step-ahead filtered log-loss and the best
    wins. Predictions are scored UNWEIGHTED (one comparison, one
    prediction), because the ELO and Glicko-2 baselines have no notion of
    a game weight and a weighted loss would not be comparable to theirs.
    """
    v2.require_known_lens(lens)
    scored = [match for match in matches if match["weight"] > weight_floor]
    zero_weight_count = len(matches) - len(scored)
    if not scored:
        # Under the vacuous-pair rule an effect-sparse lens can legitimately
        # yield zero comparisons (e.g. tiny corpora): every character is
        # unrated-provisional rather than this being an error.
        return {
            "scoring_v2_ratings_version": f"scoring_v2_{lens}_{view}_view_v1",
            "scoring_version": v2.SCORING_V2_VERSION,
            "lens": lens,
            "view": view,
            "time_axis": TIME_AXIS,
            "w2_elo": w2_elo,
            "w2_elo_selected_by": "no_comparisons",
            "w2_elo_candidates": [],
            "tie_band": v2.TIE_BAND,
            "draw_model": whr.DRAW_MODEL,
            "initial_rating": whr.DEFAULT_INITIAL_RATING,
            "initial_rd": initial_rd,
            "band_provisional_threshold": band_provisional_threshold,
            "conservative_rating_rule": "rating_minus_band",
            "character_count": 0,
            "non_provisional_count": 0,
            "comparison_count": 0,
            "zero_weight_comparison_count": zero_weight_count,
            "weight_total": 0.0,
            "mean_weight": 0.0,
            "draw_rate": 0.0,
            "time_point_count": 0,
            "node_count": 0,
            "convergence": None,
            "wall_clock_seconds": None,
            "characters": [],
        }

    diagnostics = {}
    for reading in readings:
        character = view_reading_key(reading, view)
        row = diagnostics.setdefault(
            character,
            {
                "character": character,
                "match_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "draw_count": 0,
                "unit_count": 0,
                "weight_total": 0.0,
                "movements": [],
                "labels": {label: 0 for label in ("positive", "negative", "mixed", "neutral")},
            },
        )
        row["unit_count"] += 1
        row["movements"].append(reading["movement"])
        row["labels"][reading["label"]] += 1

    for match in scored:
        for side, other in (("a", "b"), ("b", "a")):
            character = match[f"character_{side}"]
            row = diagnostics[character]
            row["match_count"] += 1
            row["weight_total"] += match["weight"]
            observed = match[f"observed_{side}"]
            if observed == 0.5:
                row["draw_count"] += 1
            elif observed == 1.0:
                row["win_count"] += 1
            else:
                row["loss_count"] += 1

    games_by_time = defaultdict(list)
    matches_by_time = defaultdict(list)
    matches_by_chapter = defaultdict(list)
    for match in scored:
        games_by_time[match["time"]].append(
            (match["character_a"], match["character_b"], match["time"], match["observed_a"], match["weight"])
        )
        matches_by_time[match["time"]].append(match)
        matches_by_chapter[match["chapter_id"]].append(match)
    times = sorted(matches_by_time)
    games = [game for time_point in times for game in games_by_time[time_point]]

    elo_evaluation, elo_unit_frozen_evaluation = _evaluate_sequential_elo(
        scored, elo_k_factor, DEFAULT_ELO_INITIAL_RATING
    )
    glicko2_evaluation = _evaluate_glicko2(
        _chapter_characters(readings, view),
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
            f"filtered weighted WHR at w2={candidate:g} Elo^2 per unit, previous node's rating",
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
        f"filtered weighted WHR at w2={chosen_w2_elo:g}, rating deflated by its posterior variance",
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
            rating = round(whr.DEFAULT_INITIAL_RATING, 1)
            band = round(2.0 * initial_rd, 1) if initial_rd is not None else None
        movements = row["movements"]
        characters.append(
            {
                "character": character,
                "rating": rating,
                "band": band,
                "conservative_rating": round(rating - band, 1) if band is not None else None,
                "provisional": band is None or band > band_provisional_threshold,
                "match_count": row["match_count"],
                "win_count": row["win_count"],
                "loss_count": row["loss_count"],
                "draw_count": row["draw_count"],
                "unit_count": row["unit_count"],
                "weight_total": round(row["weight_total"], 3),
                "mean_weight": round(row["weight_total"] / row["match_count"], 4) if row["match_count"] else 0.0,
                "mean_movement": round(sum(movements) / len(movements), 4) if movements else 0.0,
                "mean_absolute_movement": (
                    round(sum(abs(value) for value in movements) / len(movements), 4) if movements else 0.0
                ),
                "labels": dict(row["labels"]),
                "node_count": len(smoothed_points),
                "first_time": smoothed_points[0][0] if smoothed_points else None,
                "last_time": smoothed_points[-1][0] if smoothed_points else None,
                "smoothed_summary": _trajectory_summary(smoothed_points),
                "filtered_summary": _trajectory_summary(filtered_points),
                "smoothed_trajectory": smoothed_points,
                "filtered_trajectory": filtered_points,
            }
        )

    characters.sort(
        key=lambda item: (
            -(item["conservative_rating"] if item["conservative_rating"] is not None else -1e9),
            item["character"],
        )
    )
    for rank, row in enumerate(
        [item for item in characters if not item["provisional"]], start=1
    ):
        row["rank"] = rank
    for row in characters:
        row.setdefault("rank", None)

    non_provisional = [row for row in characters if not row["provisional"]]
    provisional = [row for row in characters if row["provisional"]]
    draw_count = sum(1 for match in scored if match["observed_a"] == 0.5)

    return {
        "scoring_v2_ratings_version": f"scoring_v2_{lens}_{view}_view_v1",
        "scoring_version": v2.SCORING_V2_VERSION,
        "lens": lens,
        "view": view,
        "time_axis": TIME_AXIS,
        "w2_elo": chosen_w2_elo,
        "w2_elo_selected_by": "one_step_ahead_log_loss_on_v2_comparisons" if w2_elo is None else "caller",
        "w2_elo_candidates": candidates,
        "tie_band": v2.TIE_BAND,
        "draw_model": whr.DRAW_MODEL,
        "initial_rating": whr.DEFAULT_INITIAL_RATING,
        "initial_rd": initial_rd,
        "band_provisional_threshold": band_provisional_threshold,
        "conservative_rating_rule": "rating_minus_band",
        "character_count": len(characters),
        "non_provisional_count": len(non_provisional),
        "comparison_count": len(scored),
        "zero_weight_comparison_count": zero_weight_count,
        "weight_total": round(sum(match["weight"] for match in scored), 3),
        "mean_weight": round(sum(match["weight"] for match in scored) / len(scored), 4),
        "draw_rate": round(draw_count / len(scored), 3),
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
                "one-step-ahead over the v2 comparisons in narrative order, each predicted from "
                "prior information only and scored UNWEIGHTED (one comparison, one prediction) so "
                "the weighted WHR rows and the unweighted ELO/Glicko-2 baselines are comparable; "
                "draws score as half a win plus half a loss for every system"
            ),
            "comparison": [
                best_evaluation,
                deflated_evaluation,
                elo_evaluation,
                elo_unit_frozen_evaluation,
                glicko2_evaluation,
            ],
            "whr_candidates": candidate_evaluations,
        },
        "top_rated_characters": non_provisional[:15],
        "bottom_rated_characters": sorted(
            non_provisional, key=lambda item: (item["conservative_rating"], item["character"])
        )[:5],
        "provisional_characters": sorted(provisional, key=lambda item: (-item["rating"], item["character"])),
        "characters": characters,
    }


# ---------------------------------------------------------------------------
# Timelines and the corpus summary.
# ---------------------------------------------------------------------------


def build_timeline(ratings, units, readings, view, target_characters=None):
    """Trajectory nodes joined to corpus positions, for the tracked characters."""
    position_by_time = {
        unit["time"]: {
            "unit_id": unit["unit_id"],
            "chapter_id": unit["chapter_id"],
            "chapter_index": unit["chapter_index"],
            "volume_number": unit["volume_number"],
            "unit_index_within_chapter": unit["unit_index_within_chapter"],
            "cumulative_unit_index": unit["time"],
        }
        for unit in units
    }
    reading_by_key = {
        (reading["time"], view_reading_key(reading, view)): reading for reading in readings
    }
    rows_by_character = {row["character"]: row for row in ratings["characters"]}
    selected = [
        character
        for character in (target_characters or sorted(rows_by_character))
        if character in rows_by_character
    ]

    points = []
    for character in selected:
        row = rows_by_character[character]
        for mode in TIMELINE_MODES:
            for time_point, rating, band in row.get(f"{mode}_trajectory", []):
                position = position_by_time.get(time_point)
                if position is None:
                    raise ValueError(
                        f'{mode} trajectory node at time {time_point} for "{character}" has no '
                        "corpus position; the units and the fit have drifted apart."
                    )
                reading = reading_by_key.get((time_point, character))
                points.append(
                    {
                        "character": character,
                        "mode": mode,
                        "rating": rating,
                        "band": band,
                        "movement": reading["movement"] if reading else None,
                        "label": reading["label"] if reading else None,
                        "corpus_position": dict(position),
                    }
                )
    points.sort(key=lambda point: (point["character"], point["mode"], point["corpus_position"]["cumulative_unit_index"]))

    return {
        "scoring_v2_timeline_version": f"scoring_v2_timeline_{ratings['lens']}_{view}_view_v1",
        "lens": ratings["lens"],
        "view": view,
        "time_axis": TIME_AXIS,
        "modes": list(TIMELINE_MODES),
        "w2_elo": ratings["w2_elo"],
        "tracked_character_count": len(selected),
        "tracked_characters": selected,
        "point_count": len(points),
        "points": points,
    }


def build_corpus_summary(ratings_by_lens, readings_by_lens, view="name"):
    """Per character, per lens: intensity, direction, labels, standing.

    Intensity (mean |m| per appearing unit) and direction (signed mean m)
    are per-appearance means, never sums, so a character cannot climb by
    appearing often -- v1's profile sums did exactly that. Standing is the
    rating, its band, and the conservative rank among the non-provisional
    set.
    """
    lenses = list(v2.SCORING_V2_LENS_ORDER)
    rows = {}
    for lens in lenses:
        rating_rows = {row["character"]: row for row in ratings_by_lens[lens]["characters"]}
        appearances = defaultdict(int)
        movements = defaultdict(list)
        for reading in readings_by_lens[lens]:
            key = view_reading_key(reading, view)
            appearances[key] += 1
            movements[key].append(reading["movement"])
        # every character with readings gets a lens entry; a character the
        # lens never rated (vacuous-pair rule: no comparisons touched them)
        # still reports appearances and movement, with null standing
        for character in set(appearances) | set(rating_rows):
            rating_row = rating_rows.get(character)
            entry = rows.setdefault(character, {"character": character, "lenses": {}})
            character_movements = movements.get(character, [])
            mean_movement = (
                round(sum(character_movements) / len(character_movements), 4)
                if character_movements
                else 0.0
            )
            mean_absolute = (
                round(sum(abs(m) for m in character_movements) / len(character_movements), 4)
                if character_movements
                else 0.0
            )
            entry["lenses"][lens] = {
                "appearances": appearances.get(character, 0),
                "mean_movement": rating_row["mean_movement"] if rating_row else mean_movement,
                "mean_absolute_movement": (
                    rating_row["mean_absolute_movement"] if rating_row else mean_absolute
                ),
                "labels": rating_row["labels"] if rating_row else None,
                "rating": rating_row["rating"] if rating_row else None,
                "band": rating_row["band"] if rating_row else None,
                "conservative_rating": rating_row["conservative_rating"] if rating_row else None,
                "rank": rating_row["rank"] if rating_row else None,
                "provisional": rating_row["provisional"] if rating_row else True,
                "comparison_count": rating_row["match_count"] if rating_row else 0,
            }

    summary_rows = sorted(
        rows.values(),
        key=lambda row: (
            -(row["lenses"].get("advantage", {}).get("conservative_rating") or -1e9),
            row["character"],
        ),
    )
    for row in summary_rows:
        signs = []
        for lens in lenses:
            # a character can be unrated in a lens that never saw them move
            # (vacuous-pair rule); that lens contributes a neutral sign
            rating = (row["lenses"].get(lens) or {}).get("rating")
            if rating is None or rating == whr.DEFAULT_INITIAL_RATING:
                signs.append(0)
            else:
                signs.append(1 if rating > whr.DEFAULT_INITIAL_RATING else -1)
        row["archetype_signs"] = {lens: sign for lens, sign in zip(lenses, signs)}

    return {
        "scoring_v2_corpus_summary_version": "scoring_v2_corpus_summary_v1",
        "scoring_version": v2.SCORING_V2_VERSION,
        "view": view,
        "lenses": lenses,
        "character_count": len(summary_rows),
        "characters": summary_rows,
    }


# ---------------------------------------------------------------------------
# The whole build.
# ---------------------------------------------------------------------------


def build_scoring_v2(
    run_dirs,
    w2_elo=None,
    w2_elo_candidates=DEFAULT_W2_ELO_CANDIDATES,
    views=VIEWS,
    lenses=v2.SCORING_V2_LENS_ORDER,
    registry=None,
    target_characters=None,
    progress=None,
):
    """Comparisons, ratings, timelines, and a corpus summary for every lens and view."""
    started = perf_counter()
    registry = registry or Registry.load()
    merge_map = v2.person_view_merge_map(registry)
    units = load_scored_units(run_dirs)

    comparisons_by_lens = {}
    readings_by_lens = {}
    ratings = {}
    timelines = {}
    dropped_self_pairings = {}

    tracked = list(target_characters or CHARACTER_PAGE_PILOT_EDITORIAL.keys())

    for lens in lenses:
        comparisons_by_lens[lens] = build_comparisons(units, lens, registry=registry, merge_map=merge_map)
        readings_by_lens[lens] = build_readings(units, lens, registry=registry, merge_map=merge_map)
        for view in views:
            matches, dropped = view_matches(comparisons_by_lens[lens], view)
            dropped_self_pairings[(lens, view)] = dropped
            if progress:
                progress(f"fitting {lens} / {view} view: {len(matches)} comparisons")
            ratings[(lens, view)] = fit_view(
                matches,
                readings_by_lens[lens],
                view,
                lens,
                w2_elo=w2_elo,
                w2_elo_candidates=w2_elo_candidates,
            )
            ratings[(lens, view)]["dropped_self_pairings"] = dropped
            view_tracked = [
                character if view == "name" else v2.person_view_key(character, registry=registry, merge_map=merge_map)
                for character in tracked
            ]
            timelines[(lens, view)] = build_timeline(
                ratings[(lens, view)], units, readings_by_lens[lens], view, target_characters=view_tracked
            )
            if progress:
                progress(
                    f"  w2={ratings[(lens, view)]['w2_elo']:g} "
                    f"log_loss={ratings[(lens, view)]['predictive_evaluation']['comparison'][0]['log_loss']}"
                )

    summary = build_corpus_summary(
        {lens: ratings[(lens, "name")] for lens in lenses},
        readings_by_lens,
        view="name",
    )

    manifest = {
        "scoring_v2_build_version": SCORING_V2_BUILD_VERSION,
        "scoring_version": v2.SCORING_V2_VERSION,
        "corpus": foundation_corpus_label(run_dirs),
        "run_count": len(list(run_dirs)),
        "unit_count": len(units),
        "time_point_count": len({unit["time"] for unit in units}),
        "lenses": list(lenses),
        "views": list(views),
        "tie_band": v2.TIE_BAND,
        "lens_dimension_weights": v2.LENS_DIMENSION_WEIGHTS,
        "ambiguity_decay": v2.AMBIGUITY_DECAY,
        "ambiguity_floor": v2.AMBIGUITY_FLOOR,
        "uncertain_stance_multiplier": v2.UNCERTAIN_STANCE_MULTIPLIER,
        "person_view_merges": merge_map,
        "comparison_counts": {lens: len(comparisons_by_lens[lens]) for lens in lenses},
        "dropped_self_pairings": {
            f"{lens}/{view}": dropped_self_pairings[(lens, view)] for lens, view in dropped_self_pairings
        },
        "w2_elo_selected": {
            f"{lens}/{view}": ratings[(lens, view)]["w2_elo"] for lens, view in ratings
        },
        "wall_clock_seconds": round(perf_counter() - started, 3),
    }

    return {
        "manifest": manifest,
        "units": units,
        "comparisons": comparisons_by_lens,
        "readings": readings_by_lens,
        "ratings": ratings,
        "timelines": timelines,
        "corpus_summary": summary,
    }


# ---------------------------------------------------------------------------
# Staging.
# ---------------------------------------------------------------------------


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


COMPARISON_COLUMNS = (
    "unit_id",
    "time",
    "chapter_id",
    "character_a",
    "character_b",
    "person_a",
    "person_b",
    "movement_a",
    "movement_b",
    "observed_a",
    "weight",
)


def render_corpus_summary_markdown(summary, top=25):
    lines = [
        "# Scoring v2 corpus summary (staged)",
        "",
        f"Characters: {summary['character_count']}. View: {summary['view']}. "
        f"Lenses: {', '.join(summary['lenses'])}.",
        "",
        "Intensity is the mean |m| per appearing unit; direction is the signed mean m. "
        "Neither is a sum, so appearing often cannot raise either one.",
        "",
    ]
    for lens in summary["lenses"]:
        ranked = sorted(
            (row for row in summary["characters"] if row["lenses"][lens]["rank"] is not None),
            key=lambda row: row["lenses"][lens]["rank"],
        )
        lines.extend(
            [
                f"## {lens} (top {top}, non-provisional)",
                "",
                "| rank | character | rating | band | appearances | mean m | mean abs m | +/-/mixed/neutral |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in ranked[:top]:
            cell = row["lenses"][lens]
            labels = cell["labels"]
            lines.append(
                f"| {cell['rank']} | {row['character']} | {cell['rating']:.1f} | {cell['band']:.1f} | "
                f"{cell['appearances']} | {cell['mean_movement']:+.3f} | {cell['mean_absolute_movement']:.3f} | "
                f"{labels['positive']}/{labels['negative']}/{labels['mixed']}/{labels['neutral']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_scoring_v2_artifacts(build, output_dir=DEFAULT_OUTPUT_DIR):
    """Stage every v2 artifact under `output_dir` and nowhere else."""
    output_path = Path(output_dir)
    written = []

    written.append(_write_json(output_path / "scoring-v2-build-manifest.json", build["manifest"]))

    for (lens, view), ratings in build["ratings"].items():
        written.append(_write_json(output_path / f"scoring-v2-{lens}-{view}-view-ratings.json", ratings))
    for (lens, view), timeline in build["timelines"].items():
        written.append(_write_json(output_path / f"scoring-v2-{lens}-{view}-view-timeline.json", timeline))

    for lens, comparisons in build["comparisons"].items():
        written.append(
            _write_json(
                output_path / f"scoring-v2-{lens}-comparisons.json",
                {
                    "scoring_v2_comparisons_version": f"scoring_v2_comparisons_{lens}_v1",
                    "lens": lens,
                    "comparison_count": len(comparisons),
                    "columns": list(COMPARISON_COLUMNS),
                    "rows": [[row[column] for column in COMPARISON_COLUMNS] for row in comparisons],
                },
            )
        )

    written.append(_write_json(output_path / "scoring-v2-corpus-summary.json", build["corpus_summary"]))
    summary_markdown = output_path / "scoring-v2-corpus-summary.md"
    summary_markdown.write_text(render_corpus_summary_markdown(build["corpus_summary"]))
    written.append(summary_markdown)

    return [str(path) for path in written]


__all__ = [
    "COMPARISON_COLUMNS",
    "DEFAULT_OUTPUT_DIR",
    "SCORING_V2_BUILD_VERSION",
    "VIEWS",
    "build_comparisons",
    "build_corpus_summary",
    "build_readings",
    "build_scoring_v2",
    "build_timeline",
    "fit_view",
    "load_scored_units",
    "render_corpus_summary_markdown",
    "view_keys",
    "view_matches",
    "view_reading_key",
    "write_scoring_v2_artifacts",
]
