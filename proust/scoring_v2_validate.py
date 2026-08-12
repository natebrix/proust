"""The scoring v2 adoption gate: coherence, stability, prediction, face validity.

v2 cannot be validated by agreement with v1 -- v1 is what it replaces -- so
the design doc fixes four questions instead, and this module answers them
with numbers, side by side:

1. **Orthogonality.** Cross-lens rating correlations should FALL against
   v1: v1's blended weight tables made every lens partly every other lens,
   while v2's lens projection is a partition, so each lens should now carry
   its own information.
2. **Stability.** Bootstrap over units: v2's rank standard deviation over
   the non-provisional set should be no worse than v1's on the same corpus.
   The v1 formula is reimplemented here (reading `proust/scoring.py`'s
   weight tables and `runner`'s term functions, both read-only) purely so
   the two formulas can be resampled identically.
3. **Prediction.** Filtered one-step log-loss and Brier on v2 comparisons,
   with sequential-ELO and chapter-Glicko-2 baselines on the SAME
   comparisons. Baselines have no notion of a game weight, so they ignore
   it, and for comparability every system here is scored unweighted.
4. **Face validity.** The literary panel the design doc pre-registered,
   each claim operationalized in one sentence before the numbers are read.

Nothing here is a pass/fail gate on its own; adoption is a reviewed
decision on this report.
"""

from collections import defaultdict
import json
import math
import random
from pathlib import Path
from time import perf_counter

from . import runner
from . import scoring
from . import scoring_v2 as v2
from . import scoring_v2_build as build_module
from . import whr
from .registry import Registry

DEFAULT_BOOTSTRAP_SAMPLES = 50
BOOTSTRAP_SEED = 20260812
V1_EPSILON = 0.25
V1_BAND_PROVISIONAL_THRESHOLD = 200.0

NARRATOR = "le narrateur"
DUCHESSE = "duchesse de Guermantes"
PANEL_CHARACTERS = {
    "duchesse": DUCHESSE,
    "rachel": "Rachel",
    "bloch": "Bloch",
    "odette": "Odette",
    "charlus": "baron de Charlus",
    "narrator": NARRATOR,
    "saniette": "Saniette",
    "amie": "l'amie de Mlle Vinteuil",
}

# Pre-registered operationalizations: each claim in the design doc, turned
# into one checkable number BEFORE the ratings were read.
PANEL_CLAIMS = {
    "duchesse": (
        "the duchesse de Guermantes's standing among the corpus elite: non-provisional and "
        "ranked in the top 10% of the non-provisional set in at least one lens"
    ),
    "rachel": (
        "Rachel ranked: present in the corpus, playing comparisons, and non-provisional in at "
        "least one lens (the closed-world corpus could not see her at all)"
    ),
    "bloch": "Bloch's inclusion near the bottom: bottom quartile of the non-provisional inclusion set",
    "odette": "Odette's prestige above her inclusion: prestige rating > inclusion rating",
    "charlus": (
        "Charlus's trajectory declining across the late volumes: mean smoothed advantage rating "
        "over volumes 5-7 below the mean over volumes 1-4"
    ),
    "narrator": (
        "the narrator mid-table with a tight band: advantage rank in the middle third of the "
        "non-provisional set, band below its median"
    ),
    "saniette": "Saniette last or near it: bottom 10% of the non-provisional advantage set",
    "amie": "l'amie de Mlle Vinteuil present: appears in at least one scored unit and plays comparisons",
}


# ---------------------------------------------------------------------------
# Rank statistics.
# ---------------------------------------------------------------------------


def average_ranks(values):
    """Ranks 1..n with ties averaged, the form Spearman's rho is defined on."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        mean_rank = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = mean_rank
        position = end + 1
    return ranks


def spearman(values_a, values_b):
    """Spearman rank correlation, ties averaged; None when it is undefined."""
    if len(values_a) != len(values_b):
        raise ValueError("Spearman needs two equally long sequences.")
    if len(values_a) < 3:
        return None
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    variance_a = sum((a - mean_a) ** 2 for a in ranks_a)
    variance_b = sum((b - mean_b) ** 2 for b in ranks_b)
    if variance_a <= 0.0 or variance_b <= 0.0:
        return None
    return covariance / math.sqrt(variance_a * variance_b)


def standard_deviation(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# The v1 formula, reimplemented for resampling only.
# ---------------------------------------------------------------------------


def v1_unit_net_scores(annotation, lens):
    """v1's per-character net score for one unit: events + effects - ambiguity penalty.

    This is `runner._score_run_outcomes`'s arithmetic for a single
    annotation, using the same weight tables from `proust/scoring.py`. It
    exists so the bootstrap can resample both formulas over the same units;
    v1 itself is not modified or re-run.
    """
    config = scoring.SCORING_LENS_CONFIGS[lens]
    penalty = len(annotation.get("ambiguities") or []) * config["ambiguity_penalty"]
    scores = {
        row["canonical_name"]: 0.0
        for row in annotation.get("characters_present") or []
        if isinstance(row, dict) and isinstance(row.get("canonical_name"), str)
    }
    for event in annotation.get("appraisal_events") or []:
        target = event.get("target")
        if target in scores:
            scores[target] += runner._outcome_event_score(event, config)
    for effect in annotation.get("status_effects") or []:
        character = effect.get("character")
        if character in scores:
            scores[character] += runner._outcome_status_score(effect, config)
    return {character: value - penalty for character, value in scores.items()}


def v1_matches(units, lens, epsilon=V1_EPSILON):
    """v1's within-unit pairwise matches, unweighted, on the name view."""
    from itertools import combinations

    matches = []
    for unit in units:
        scores = v1_unit_net_scores(unit["annotation"], lens)
        for character_a, character_b in combinations(sorted(scores), 2):
            difference = scores[character_a] - scores[character_b]
            if difference > epsilon:
                observed_a = 1.0
            elif difference < -epsilon:
                observed_a = 0.0
            else:
                observed_a = 0.5
            matches.append(
                {
                    "time": unit["time"],
                    "character_a": character_a,
                    "character_b": character_b,
                    "observed_a": observed_a,
                    "weight": 1.0,
                }
            )
    return matches


def v2_matches(units, lens, registry=None, merge_map=None, view="name"):
    comparisons = build_module.build_comparisons(units, lens, registry=registry, merge_map=merge_map)
    matches, _dropped = build_module.view_matches(comparisons, view)
    return matches


# ---------------------------------------------------------------------------
# Bootstrap stability.
# ---------------------------------------------------------------------------


def _conservative_ratings(matches, w2_elo, weighted, initial_rd=whr.DEFAULT_INITIAL_RD):
    """Smoothed fit of one match set; character -> (rating - band)."""
    games = []
    for match in matches:
        if weighted:
            if match["weight"] <= 0.0:
                continue
            games.append(
                (match["character_a"], match["character_b"], match["time"], match["observed_a"], match["weight"])
            )
        else:
            games.append((match["character_a"], match["character_b"], match["time"], match["observed_a"]))
    result = whr.fit(games, w2_elo, initial_rd=initial_rd)
    return {
        character: nodes[-1]["rating"] - nodes[-1]["band"]
        for character, nodes in result["players"].items()
    }


def bootstrap_stability(
    units,
    lens,
    reference_characters,
    w2_by_formula,
    registry=None,
    merge_map=None,
    samples=DEFAULT_BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
    progress=None,
):
    """Resample units with replacement; how far do the standings move?

    Both formulas see the SAME resampled unit lists (one shared random
    stream, drawn once per sample), so any difference in rank standard
    deviation is a difference between the formulas rather than between two
    bootstraps. Ranks are taken over `reference_characters` -- the
    characters both formulas rate non-provisionally on the full corpus --
    so the two are ranking the same field.
    """
    generator = random.Random(seed)
    ordered_reference = sorted(reference_characters)
    ranks_by_formula = {
        formula: defaultdict(list) for formula in ("v2", "v1")
    }

    for sample_index in range(samples):
        drawn = [units[generator.randrange(len(units))] for _ in range(len(units))]
        for formula in ("v2", "v1"):
            if formula == "v2":
                matches = v2_matches(drawn, lens, registry=registry, merge_map=merge_map)
            else:
                matches = v1_matches(drawn, lens)
            ratings = _conservative_ratings(matches, w2_by_formula[formula], weighted=(formula == "v2"))
            present = [character for character in ordered_reference if character in ratings]
            values = [-ratings[character] for character in present]
            for character, rank in zip(present, average_ranks(values)):
                ranks_by_formula[formula][character].append(rank)
        if progress and (sample_index + 1) % 10 == 0:
            progress(f"  bootstrap {lens}: {sample_index + 1}/{samples} samples")

    summary = {}
    for formula, ranks in ranks_by_formula.items():
        deviations = {
            character: standard_deviation(values)
            for character, values in ranks.items()
            if len(values) >= 2
        }
        ordered = sorted(deviations.values())
        summary[formula] = {
            "character_count": len(deviations),
            "mean_rank_stddev": round(sum(deviations.values()) / len(deviations), 3) if deviations else None,
            "median_rank_stddev": round(ordered[len(ordered) // 2], 3) if ordered else None,
            "max_rank_stddev": round(max(deviations.values()), 3) if deviations else None,
            "least_stable": sorted(deviations.items(), key=lambda item: -item[1])[:5],
        }
    summary["sample_count"] = samples
    summary["reference_character_count"] = len(ordered_reference)
    return summary


# ---------------------------------------------------------------------------
# Reading the staged artifacts and v1's current artifacts.
# ---------------------------------------------------------------------------


def load_staged_ratings(
    output_dir=build_module.DEFAULT_OUTPUT_DIR,
    lenses=v2.SCORING_V2_LENS_ORDER,
    views=build_module.VIEWS,
):
    staged = {}
    for lens in lenses:
        for view in views:
            path = Path(output_dir) / f"scoring-v2-{lens}-{view}-view-ratings.json"
            staged[(lens, view)] = json.loads(path.read_text())
    return staged


def load_v1_ratings(outputs_dir="outputs", lenses=v2.SCORING_V2_LENS_ORDER):
    v1 = {}
    for lens in lenses:
        path = Path(outputs_dir) / f"character-whr-{lens}-current.json"
        v1[lens] = json.loads(path.read_text())
    return v1


def _rating_map(artifact, non_provisional_only=False):
    return {
        row["character"]: row["rating"]
        for row in artifact["characters"]
        if not (non_provisional_only and row["provisional"])
    }


def cross_lens_correlations(rating_by_lens, lenses=v2.SCORING_V2_LENS_ORDER):
    """Spearman rho between every pair of lenses, over their shared characters."""
    pairs = {}
    for index, lens_a in enumerate(lenses):
        for lens_b in lenses[index + 1 :]:
            shared = sorted(set(rating_by_lens[lens_a]) & set(rating_by_lens[lens_b]))
            rho = spearman(
                [rating_by_lens[lens_a][character] for character in shared],
                [rating_by_lens[lens_b][character] for character in shared],
            )
            pairs[f"{lens_a} vs {lens_b}"] = {
                "spearman": round(rho, 4) if rho is not None else None,
                "character_count": len(shared),
            }
    values = [entry["spearman"] for entry in pairs.values() if entry["spearman"] is not None]
    pairs["mean_absolute"] = {
        "spearman": round(sum(abs(value) for value in values) / len(values), 4) if values else None,
        "character_count": None,
    }
    return pairs


def frequency_confounding(staged, v1_artifacts, lenses=v2.SCORING_V2_LENS_ORDER):
    """Does the standing track how often a character is compared?

    The design's fourth principle is that frequency must not masquerade as
    strength. Rank is by conservative rating (rating - band) and a band
    narrows with evidence, so a character compared often is pushed up the
    table by that alone. This measures the effect directly, for both
    formulas, over each one's own non-provisional set.
    """
    rows = {}
    for lens in lenses:
        entry = {}
        for label, artifact in (("v2", staged[(lens, "name")]), ("v1", v1_artifacts[lens])):
            ranked = [row for row in artifact["characters"] if not row["provisional"]]
            counts = [row["match_count"] for row in ranked]
            entry[label] = {
                "character_count": len(ranked),
                "conservative_vs_comparisons": _rounded(
                    spearman([row["conservative_rating"] for row in ranked], counts)
                ),
                "rating_vs_comparisons": _rounded(spearman([row["rating"] for row in ranked], counts)),
                "band_vs_comparisons": _rounded(spearman([row["band"] for row in ranked], counts)),
                "rating_spread": round(
                    max(row["rating"] for row in ranked) - min(row["rating"] for row in ranked), 1
                ),
                "band_spread": round(
                    max(row["band"] for row in ranked) - min(row["band"] for row in ranked), 1
                ),
            }
        rows[lens] = entry
    return rows


def _rounded(value, digits=4):
    return round(value, digits) if value is not None else None


# ---------------------------------------------------------------------------
# The literary panel.
# ---------------------------------------------------------------------------


def _rows_by_character(artifact):
    return {row["character"]: row for row in artifact["characters"]}


def _non_provisional(artifact):
    return [row for row in artifact["characters"] if not row["provisional"]]


def _percentile_of_rank(row, artifact):
    ranked = _non_provisional(artifact)
    if row is None or row.get("rank") is None or not ranked:
        return None
    return round(row["rank"] / len(ranked), 4)


def evaluate_panel(staged, units, readings_by_lens):
    """Every pre-registered claim, with the numbers that decide it."""
    name_view = {lens: staged[(lens, "name")] for lens in v2.SCORING_V2_LENS_ORDER}
    rows = {lens: _rows_by_character(name_view[lens]) for lens in name_view}
    volume_by_time = {unit["time"]: unit["volume_number"] for unit in units}
    results = {}

    def row_for(lens, character):
        return rows[lens].get(character)

    # 1. The duchesse among the elite.
    duchesse_detail = {}
    passed = False
    for lens in v2.SCORING_V2_LENS_ORDER:
        row = row_for(lens, DUCHESSE)
        percentile = _percentile_of_rank(row, name_view[lens])
        duchesse_detail[lens] = {
            "rating": row["rating"] if row else None,
            "band": row["band"] if row else None,
            "rank": row["rank"] if row else None,
            "non_provisional_count": len(_non_provisional(name_view[lens])),
            "rank_percentile": percentile,
        }
        if row and not row["provisional"] and percentile is not None and percentile <= 0.10:
            passed = True
    results["duchesse"] = {"passed": passed, "detail": duchesse_detail}

    # 2. Rachel ranked.
    rachel_detail = {}
    passed = False
    for lens in v2.SCORING_V2_LENS_ORDER:
        row = row_for(lens, "Rachel")
        rachel_detail[lens] = {
            "rating": row["rating"] if row else None,
            "band": row["band"] if row else None,
            "rank": row["rank"] if row else None,
            "unit_count": row["unit_count"] if row else 0,
            "comparison_count": row["match_count"] if row else 0,
            "provisional": row["provisional"] if row else None,
        }
        if row and not row["provisional"]:
            passed = True
    results["rachel"] = {"passed": passed, "detail": rachel_detail}

    # 3. Bloch's inclusion near the bottom.
    row = row_for("inclusion", "Bloch")
    percentile = _percentile_of_rank(row, name_view["inclusion"])
    results["bloch"] = {
        "passed": bool(row and not row["provisional"] and percentile is not None and percentile >= 0.75),
        "detail": {
            "inclusion": {
                "rating": row["rating"] if row else None,
                "rank": row["rank"] if row else None,
                "rank_percentile": percentile,
                "non_provisional_count": len(_non_provisional(name_view["inclusion"])),
                "provisional": row["provisional"] if row else None,
            }
        },
    }

    # 4. Odette's prestige above her inclusion.
    prestige_row = row_for("prestige", "Odette")
    inclusion_row = row_for("inclusion", "Odette")
    results["odette"] = {
        "passed": bool(prestige_row and inclusion_row and prestige_row["rating"] > inclusion_row["rating"]),
        "detail": {
            "prestige": {
                "rating": prestige_row["rating"] if prestige_row else None,
                "rank": prestige_row["rank"] if prestige_row else None,
                "mean_movement": prestige_row["mean_movement"] if prestige_row else None,
            },
            "inclusion": {
                "rating": inclusion_row["rating"] if inclusion_row else None,
                "rank": inclusion_row["rank"] if inclusion_row else None,
                "mean_movement": inclusion_row["mean_movement"] if inclusion_row else None,
            },
        },
    }

    # 5. Charlus declining across the late volumes.
    charlus_detail = {}
    for lens in v2.SCORING_V2_LENS_ORDER:
        row = row_for(lens, PANEL_CHARACTERS["charlus"])
        early = [
            rating
            for time_point, rating, _band in (row["smoothed_trajectory"] if row else [])
            if volume_by_time.get(time_point, 0) <= 4
        ]
        late = [
            rating
            for time_point, rating, _band in (row["smoothed_trajectory"] if row else [])
            if volume_by_time.get(time_point, 0) >= 5
        ]
        trajectory = row["smoothed_trajectory"] if row else []
        charlus_detail[lens] = {
            "first_rating": trajectory[0][1] if trajectory else None,
            "last_rating": trajectory[-1][1] if trajectory else None,
            "early_volume_mean": round(sum(early) / len(early), 1) if early else None,
            "late_volume_mean": round(sum(late) / len(late), 1) if late else None,
            "early_node_count": len(early),
            "late_node_count": len(late),
            "rating": row["rating"] if row else None,
            "rank": row["rank"] if row else None,
        }
    advantage = charlus_detail["advantage"]
    results["charlus"] = {
        "passed": bool(
            advantage["early_volume_mean"] is not None
            and advantage["late_volume_mean"] is not None
            and advantage["late_volume_mean"] < advantage["early_volume_mean"]
        ),
        "detail": charlus_detail,
    }

    # 6. The narrator mid-table with a tight band.
    row = row_for("advantage", NARRATOR)
    ranked = _non_provisional(name_view["advantage"])
    bands = sorted(item["band"] for item in ranked)
    median_band = bands[len(bands) // 2] if bands else None
    percentile = _percentile_of_rank(row, name_view["advantage"])
    results["narrator"] = {
        "passed": bool(
            row
            and not row["provisional"]
            and percentile is not None
            and 1 / 3 <= percentile <= 2 / 3
            and median_band is not None
            and row["band"] < median_band
        ),
        "detail": {
            "advantage": {
                "rating": row["rating"] if row else None,
                "band": row["band"] if row else None,
                "rank": row["rank"] if row else None,
                "rank_percentile": percentile,
                "median_band": median_band,
                "unit_count": row["unit_count"] if row else 0,
                "comparison_count": row["match_count"] if row else 0,
            },
            "prestige": {
                "rating": row_for("prestige", NARRATOR)["rating"] if row_for("prestige", NARRATOR) else None,
                "rank": row_for("prestige", NARRATOR)["rank"] if row_for("prestige", NARRATOR) else None,
            },
            "inclusion": {
                "rating": row_for("inclusion", NARRATOR)["rating"] if row_for("inclusion", NARRATOR) else None,
                "rank": row_for("inclusion", NARRATOR)["rank"] if row_for("inclusion", NARRATOR) else None,
            },
        },
    }

    # 7. Saniette last or near it.
    row = row_for("advantage", "Saniette")
    percentile = _percentile_of_rank(row, name_view["advantage"])
    results["saniette"] = {
        "passed": bool(row and not row["provisional"] and percentile is not None and percentile >= 0.90),
        "detail": {
            "advantage": {
                "rating": row["rating"] if row else None,
                "band": row["band"] if row else None,
                "rank": row["rank"] if row else None,
                "rank_percentile": percentile,
                "provisional": row["provisional"] if row else None,
                "non_provisional_count": len(ranked),
                "mean_movement": row["mean_movement"] if row else None,
            }
        },
    }

    # 8. l'amie de Mlle Vinteuil present.
    amie = PANEL_CHARACTERS["amie"]
    unit_count = len({reading["unit_id"] for reading in readings_by_lens["advantage"] if reading["character"] == amie})
    row = row_for("advantage", amie)
    results["amie"] = {
        "passed": bool(unit_count > 0 and row and row["match_count"] > 0),
        "detail": {
            "advantage": {
                "unit_count": unit_count,
                "comparison_count": row["match_count"] if row else 0,
                "rating": row["rating"] if row else None,
                "band": row["band"] if row else None,
                "rank": row["rank"] if row else None,
                "provisional": row["provisional"] if row else None,
            }
        },
    }

    for key, result in results.items():
        result["claim"] = PANEL_CLAIMS[key]
    return results


# ---------------------------------------------------------------------------
# Person-view deltas.
# ---------------------------------------------------------------------------


def person_view_deltas(staged, registry, merge_map):
    """Which standings rows merge in the person view, and how the ranks shift."""
    deltas = {}
    for lens in v2.SCORING_V2_LENS_ORDER:
        name_rows = _rows_by_character(staged[(lens, "name")])
        person_rows = _rows_by_character(staged[(lens, "person")])
        merged = []
        for source, target in sorted(merge_map.items()):
            source_names = registry.entities[source].annotation_names or (registry.entities[source].display_name,)
            target_names = registry.entities[target].annotation_names or (registry.entities[target].display_name,)
            merged.append(
                {
                    "merged_entity": target,
                    "from": source,
                    "name_view_rows": [
                        {
                            "character": name,
                            "rating": name_rows[name]["rating"],
                            "band": name_rows[name]["band"],
                            "rank": name_rows[name]["rank"],
                            "unit_count": name_rows[name]["unit_count"],
                        }
                        for name in list(source_names) + list(target_names)
                        if name in name_rows
                    ],
                    "person_view_row": (
                        {
                            "character": target,
                            "rating": person_rows[target]["rating"],
                            "band": person_rows[target]["band"],
                            "rank": person_rows[target]["rank"],
                            "unit_count": person_rows[target]["unit_count"],
                        }
                        if target in person_rows
                        else None
                    ),
                }
            )

        shifts = []
        for character, row in name_rows.items():
            if row["rank"] is None:
                continue
            person_key = v2.person_view_key(character, registry=registry, merge_map=merge_map)
            person_row = person_rows.get(person_key)
            if person_row is None or person_row["rank"] is None:
                continue
            shifts.append(
                {
                    "character": character,
                    "person": person_key,
                    "name_rank": row["rank"],
                    "person_rank": person_row["rank"],
                    "rank_shift": row["rank"] - person_row["rank"],
                    "rating_shift": round(person_row["rating"] - row["rating"], 1),
                }
            )
        shifts.sort(key=lambda item: (-abs(item["rank_shift"]), item["character"]))
        ranked_names = [row for row in staged[(lens, "name")]["characters"] if not row["provisional"]]
        ranked_persons = [row for row in staged[(lens, "person")]["characters"] if not row["provisional"]]
        deltas[lens] = {
            "merged": merged,
            "largest_shifts": shifts[:10],
            "mean_absolute_rank_shift": (
                round(sum(abs(item["rank_shift"]) for item in shifts) / len(shifts), 3) if shifts else None
            ),
            "name_view_character_count": staged[(lens, "name")]["character_count"],
            "person_view_character_count": staged[(lens, "person")]["character_count"],
            "name_view_non_provisional": len(ranked_names),
            "person_view_non_provisional": len(ranked_persons),
            "dropped_self_pairings": staged[(lens, "person")].get("dropped_self_pairings"),
        }
    return deltas


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


def build_validation_report(
    run_dirs,
    output_dir=build_module.DEFAULT_OUTPUT_DIR,
    outputs_dir="outputs",
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    registry=None,
    progress=None,
):
    started = perf_counter()
    registry = registry or Registry.load()
    merge_map = v2.person_view_merge_map(registry)
    staged = load_staged_ratings(output_dir)
    manifest = json.loads((Path(output_dir) / "scoring-v2-build-manifest.json").read_text())
    v1_artifacts = load_v1_ratings(outputs_dir)
    units = build_module.load_scored_units(run_dirs)
    readings_by_lens = {
        lens: build_module.build_readings(units, lens, registry=registry, merge_map=merge_map)
        for lens in v2.SCORING_V2_LENS_ORDER
    }

    if progress:
        progress("orthogonality")
    orthogonality = {
        "v2_all_rated": cross_lens_correlations(
            {lens: _rating_map(staged[(lens, "name")]) for lens in v2.SCORING_V2_LENS_ORDER}
        ),
        "v2_non_provisional": cross_lens_correlations(
            {lens: _rating_map(staged[(lens, "name")], True) for lens in v2.SCORING_V2_LENS_ORDER}
        ),
        "v1_all_rated": cross_lens_correlations(
            {lens: _rating_map(v1_artifacts[lens]) for lens in v2.SCORING_V2_LENS_ORDER}
        ),
        "v1_non_provisional": cross_lens_correlations(
            {lens: _rating_map(v1_artifacts[lens], True) for lens in v2.SCORING_V2_LENS_ORDER}
        ),
    }

    if progress:
        progress("bootstrap stability")
    stability = {}
    for lens in v2.SCORING_V2_LENS_ORDER:
        v2_non_provisional = {row["character"] for row in _non_provisional(staged[(lens, "name")])}
        v1_non_provisional = {
            row["character"] for row in v1_artifacts[lens]["characters"] if not row["provisional"]
        }
        reference = v2_non_provisional & v1_non_provisional
        if progress:
            progress(f"  {lens}: {len(reference)} shared non-provisional characters")
        stability[lens] = bootstrap_stability(
            units,
            lens,
            reference,
            {"v2": staged[(lens, "name")]["w2_elo"], "v1": v1_artifacts[lens]["w2_elo"]},
            registry=registry,
            merge_map=merge_map,
            samples=bootstrap_samples,
            progress=progress,
        )
        stability[lens]["v2_non_provisional_count"] = len(v2_non_provisional)
        stability[lens]["v1_non_provisional_count"] = len(v1_non_provisional)

    predictive = {
        f"{lens}/{view}": {
            "w2_elo": staged[(lens, view)]["w2_elo"],
            "w2_candidates": staged[(lens, view)]["predictive_evaluation"]["whr_candidates"],
            "comparison": staged[(lens, view)]["predictive_evaluation"]["comparison"],
            "draw_rate": staged[(lens, view)]["draw_rate"],
            "comparison_count": staged[(lens, view)]["comparison_count"],
            "mean_weight": staged[(lens, view)]["mean_weight"],
        }
        for lens in v2.SCORING_V2_LENS_ORDER
        for view in build_module.VIEWS
    }

    frequency = frequency_confounding(staged, v1_artifacts)

    if progress:
        progress("panel")
    panel = evaluate_panel(staged, units, readings_by_lens)
    deltas = person_view_deltas(staged, registry, merge_map)

    headline = {}
    for lens in v2.SCORING_V2_LENS_ORDER:
        ranked = _non_provisional(staged[(lens, "name")])
        headline[lens] = {
            "top": ranked[:15],
            "bottom": sorted(ranked, key=lambda row: row["conservative_rating"])[:5],
            "non_provisional_count": len(ranked),
            "character_count": staged[(lens, "name")]["character_count"],
        }

    return {
        "scoring_v2_validation_version": "scoring_v2_validation_v1",
        "build_manifest": manifest,
        "unit_count": len(units),
        "bootstrap_samples": bootstrap_samples,
        "orthogonality": orthogonality,
        "stability": stability,
        "frequency_confounding": frequency,
        "predictive": predictive,
        "panel": panel,
        "person_view": deltas,
        "headline": headline,
        "wall_clock_seconds": round(perf_counter() - started, 1),
    }


def _cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _format_detail_table(detail):
    """Render a panel claim's detail dict as one row per lens, generically."""
    columns = []
    for values in detail.values():
        for key in values:
            if key not in columns:
                columns.append(key)
    lines = [
        "| lens | " + " | ".join(column.replace("_", " ") for column in columns) + " |",
        "| --- | " + " | ".join("---:" for _column in columns) + " |",
    ]
    for lens, values in detail.items():
        lines.append("| " + lens + " | " + " | ".join(_cell(values.get(column)) for column in columns) + " |")
    return lines


def _format_correlation_table(correlations, title):
    lines = [f"| pair | {title} |", "| --- | ---: |"]
    for pair, entry in correlations.items():
        if pair == "mean_absolute":
            continue
        rho = entry["spearman"]
        value = f"{rho:+.4f}" if rho is not None else "-"
        lines.append(f"| {pair} | {value} (n={entry['character_count']}) |")
    mean_absolute = correlations["mean_absolute"]["spearman"]
    lines.append(
        f"| **mean abs** | **{mean_absolute:.4f}** |" if mean_absolute is not None else "| **mean abs** | - |"
    )
    return lines


def render_validation_report_markdown(report):
    manifest = report["build_manifest"]
    lines = [
        "# Scoring v2 validation report (staged, pre-adoption)",
        "",
        f"Corpus: {manifest['corpus']}, {manifest['run_count']} runs, {manifest['unit_count']} reviewed units, "
        f"{manifest['time_point_count']} narrative time points. "
        f"Comparisons per lens: {manifest['comparison_counts']}.",
        "",
        "Formula: `proust/scoring_v2.py`, exactly as specified in "
        "`proust/docs/scoring_v2_design.md`. Ratings: weighted WHR "
        "(`proust/whr.py`), smoothed and filtered, on the "
        "`cumulative_unit_index` narrative axis. Everything here is staged under "
        "`outputs/scoring-v2/`; adoption is a separate reviewed decision.",
        "",
        f"w2 selected per lens/view: "
        + ", ".join(f"{key} = {value:g}" for key, value in sorted(manifest["w2_elo_selected"].items())),
        "",
        "## 1. Lens orthogonality",
        "",
        "The design predicts cross-lens rating correlations should FALL against v1: "
        "v1's weight tables blended every dimension into every lens, v2's projection partitions them.",
        "",
    ]
    lines.extend(_format_correlation_table(report["orthogonality"]["v2_all_rated"], "v2 Spearman (all rated)"))
    lines.append("")
    lines.extend(_format_correlation_table(report["orthogonality"]["v1_all_rated"], "v1 Spearman (all rated)"))
    lines.append("")
    lines.extend(
        _format_correlation_table(
            report["orthogonality"]["v2_non_provisional"], "v2 Spearman (non-provisional)"
        )
    )
    lines.append("")
    lines.extend(
        _format_correlation_table(
            report["orthogonality"]["v1_non_provisional"], "v1 Spearman (non-provisional)"
        )
    )
    v2_mean = report["orthogonality"]["v2_all_rated"]["mean_absolute"]["spearman"]
    v1_mean = report["orthogonality"]["v1_all_rated"]["mean_absolute"]["spearman"]
    lines.extend(
        [
            "",
            f"**Verdict**: mean |rho| {_cell(v1_mean)} (v1) -> {_cell(v2_mean)} (v2): "
            + (
                "prediction held."
                if v1_mean is not None and v2_mean is not None and v2_mean < v1_mean
                else "prediction FAILED."
            ),
            "",
            "## 2. Bootstrap stability",
            "",
            f"{report['bootstrap_samples']} unit-level resamples with replacement, both formulas scored on the "
            "same drawn corpora; ranks are taken over the characters both formulas rate non-provisionally "
            "on the full corpus. Lower rank standard deviation is more stable.",
            "",
            "| lens | field | v2 mean sd | v1 mean sd | v2 median sd | v1 median sd | v2 non-prov | v1 non-prov |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lens, entry in report["stability"].items():
        lines.append(
            f"| {lens} | {entry['reference_character_count']} | {entry['v2']['mean_rank_stddev']} | "
            f"{entry['v1']['mean_rank_stddev']} | {entry['v2']['median_rank_stddev']} | "
            f"{entry['v1']['median_rank_stddev']} | {entry['v2_non_provisional_count']} | "
            f"{entry['v1_non_provisional_count']} |"
        )

    lines.extend(
        [
            "",
            "### 2b. Frequency confounding",
            "",
            "The design's fourth principle is that frequency must not masquerade as strength. "
            "Ratings are no longer sums, so nothing accumulates with appearances -- but the "
            "standings rank by rating MINUS band, and a band narrows with evidence. Where a "
            "lens's ratings are tightly packed and its bands are not, the ranking is mostly a "
            "comparison count. Spearman rho against comparison count, over each formula's own "
            "non-provisional set:",
            "",
            "| lens | formula | conservative vs count | rating vs count | band vs count | rating spread | band spread |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lens, entry in report["frequency_confounding"].items():
        for formula in ("v2", "v1"):
            values = entry[formula]
            lines.append(
                f"| {lens} | {formula} | {_cell(values['conservative_vs_comparisons'])} | "
                f"{_cell(values['rating_vs_comparisons'])} | {_cell(values['band_vs_comparisons'])} | "
                f"{values['rating_spread']:.1f} | {values['band_spread']:.1f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Predictive sanity",
            "",
            "One-step-ahead over the v2 comparisons in narrative order. Every system is scored "
            "UNWEIGHTED (one comparison, one prediction): ELO and Glicko-2 have no notion of a game "
            "weight, so a weighted loss would not be comparable to theirs. The WHR fits themselves "
            "DO use the weights. Cross-formula comparison against v1's own numbers is meaningless "
            "here -- the comparisons differ -- so only the within-v2 ordering is informative.",
            "",
            "| lens/view | system | log-loss | Brier | comparisons |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for key, entry in report["predictive"].items():
        for row in entry["comparison"]:
            lines.append(
                f"| {key} | {row['system']} | {row['log_loss']:.6f} | {row['brier']:.6f} | {row['match_count']} |"
            )

    lines.extend(["", "### w2 selection", "", "| lens/view | w2 | log-loss |", "| --- | ---: | ---: |"])
    for key, entry in report["predictive"].items():
        for candidate in entry["w2_candidates"]:
            marker = " **(selected)**" if candidate["w2_elo"] == entry["w2_elo"] else ""
            lines.append(f"| {key} | {candidate['w2_elo']:g}{marker} | {candidate['log_loss']:.6f} |")

    passed = sum(1 for result in report["panel"].values() if result["passed"])
    lines.extend(
        [
            "",
            "## 4. Literary panel (pre-registered)",
            "",
            "Each claim comes from the design doc; each operationalization was fixed before the "
            "ratings were read. Name view, and the standings referred to are the non-provisional set.",
            "",
            f"**{passed}/{len(report['panel'])} claims pass.**",
            "",
            "| claim | verdict |",
            "| --- | --- |",
        ]
    )
    for result in report["panel"].values():
        lines.append(f"| {result['claim']} | {'PASS' if result['passed'] else 'FAIL'} |")
    lines.append("")
    for key, result in report["panel"].items():
        lines.extend([f"### {key} — {'PASS' if result['passed'] else 'FAIL'}", "", result["claim"], ""])
        lines.extend(_format_detail_table(result["detail"]))
        lines.append("")

    lines.extend(["## 5. Headline standings (name view, non-provisional)", ""])
    for lens, entry in report["headline"].items():
        lines.extend(
            [
                f"### {lens} — top 15 of {entry['non_provisional_count']} non-provisional "
                f"({entry['character_count']} rated)",
                "",
                "| rank | character | rating | band | conservative | units | comparisons | mean m | mean abs m |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in entry["top"]:
            lines.append(
                f"| {row['rank']} | {row['character']} | {row['rating']:.1f} | {row['band']:.1f} | "
                f"{row['conservative_rating']:.1f} | {row['unit_count']} | {row['match_count']} | "
                f"{row['mean_movement']:+.3f} | {row['mean_absolute_movement']:.3f} |"
            )
        lines.extend(["", f"Bottom 5, {lens}:", "", "| rank | character | rating | band | conservative |", "| ---: | --- | ---: | ---: | ---: |"])
        for row in entry["bottom"]:
            lines.append(
                f"| {row['rank']} | {row['character']} | {row['rating']:.1f} | {row['band']:.1f} | "
                f"{row['conservative_rating']:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. Person view",
            "",
            "The person view aggregates on registry entity ids with `person_view_merge` links "
            "applied, so the two era names of one man become one player; `keep_separate` links "
            "(the post-V7 princesse de Guermantes, who is Mme Verdurin holding a dead woman's "
            "title) never merge.",
            "",
            "| lens | merged | name-view rows | person-view row | mean abs rank shift | self-pairings dropped |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for lens, entry in report["person_view"].items():
        for merged in entry["merged"]:
            name_rows = "; ".join(
                f"{row['character']} r={row['rating']:.0f} rank={_cell(row['rank'])} units={row['unit_count']}"
                for row in merged["name_view_rows"]
            )
            person_row = merged["person_view_row"]
            person_text = (
                f"{person_row['character']} r={person_row['rating']:.0f} "
                f"rank={_cell(person_row['rank'])} units={person_row['unit_count']}"
                if person_row
                else "-"
            )
            lines.append(
                f"| {lens} | {merged['from']} -> {merged['merged_entity']} | {name_rows} | {person_text} | "
                f"{entry['mean_absolute_rank_shift']} | {entry['dropped_self_pairings']} |"
            )
    lines.append("")
    lines.extend(
        [
            "Largest rank shifts between the two views (name-view rank minus person-view rank; "
            "the two views rank different fields, so a shift is not by itself a finding):",
            "",
            "| lens | character | person key | name rank | person rank | shift | rating shift |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for lens, entry in report["person_view"].items():
        for shift in entry["largest_shifts"][:5]:
            lines.append(
                f"| {lens} | {shift['character']} | {shift['person']} | {shift['name_rank']} | "
                f"{shift['person_rank']} | {shift['rank_shift']:+d} | {shift['rating_shift']:+.1f} |"
            )
    lines.append("")

    lines.extend(
        [
            "## 7. Reading notes: where the implementation had to choose",
            "",
            "The design doc leaves four points open; each was resolved once, in code, and is "
            "recorded here so the review can overrule it.",
            "",
            "1. **kappa is scoped to the lens.** \"The mean confidence of c's effects in u\" is read "
            "as the effects that MOVE c in this lens. A character with only a `social_status` "
            "effect is therefore a zero-effect character under advantage and falls back to "
            "presence confidence there, while carrying that effect's confidence under prestige. "
            "The alternative (pooling all five dimensions into one kappa) would let a lens's "
            "weights be set by evidence that lens is defined not to see.",
            "2. **Label precedence.** A movement past the tie band names itself first; the "
            "sign-conflict test decides only within the band. So a character with a big positive "
            "movement and one small negative effect reads positive, not mixed. Mixed still "
            "REQUIRES a genuine sign conflict, which is the clause the doc makes binding.",
            "3. **Predictive scores are unweighted.** The WHR fits use the weights; the scoring of "
            "predictions does not, because ELO and Glicko-2 have no weight to use and a weighted "
            "loss would not be comparable to theirs.",
            "4. **w2 is selected per lens AND per view**, independently, by the same one-step-ahead "
            "log-loss rule v1 uses.",
            "",
            "Deferred, as the design doc says: dossier lens cards (dominant dimension, percentile), "
            "the archetype rewrite, and the person/name UI toggle -- all app-facing, all after the "
            "adoption gate. The corpus summary carries the sign triple the archetype would use.",
            "",
            f"Wall clock: {report['wall_clock_seconds']} s for the validation battery; "
            f"{report['build_manifest']['wall_clock_seconds']} s for the build it reads.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_validation_report(report, output_dir=build_module.DEFAULT_OUTPUT_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "validation-report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown_path = output_path / "validation-report.md"
    markdown_path.write_text(render_validation_report_markdown(report))
    return [str(json_path), str(markdown_path)]


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "PANEL_CLAIMS",
    "average_ranks",
    "bootstrap_stability",
    "build_validation_report",
    "cross_lens_correlations",
    "evaluate_panel",
    "load_staged_ratings",
    "load_v1_ratings",
    "person_view_deltas",
    "render_validation_report_markdown",
    "spearman",
    "standard_deviation",
    "v1_matches",
    "v1_unit_net_scores",
    "v2_matches",
    "write_validation_report",
]
