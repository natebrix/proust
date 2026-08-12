from collections import defaultdict
import json
import math
import random

import proust as pn
import proust.runner as pr
from proust import whr
from proust.annotation import AnnotationUnitSpec


# ---------------------------------------------------------------------------
# Pure WHR math (proust/whr.py), no corpus involved.
# ---------------------------------------------------------------------------


def _static_bradley_terry_difference(win_count, loss_count, draw_count):
    """Maximum-likelihood rating difference for a two-player Bradley-Terry model.

    With only two players the whole likelihood depends on a single number,
    the log-gamma difference `d`, so the MLE can be found by plain 1-D
    search without reference to anything in whr.py. Draws count as half a
    win plus half a loss, matching the module's draw model.
    """
    effective_wins = win_count + 0.5 * draw_count
    effective_losses = loss_count + 0.5 * draw_count

    def log_likelihood(difference):
        probability = 1.0 / (1.0 + math.exp(-difference))
        return effective_wins * math.log(probability) + effective_losses * math.log(1.0 - probability)

    low, high = -20.0, 20.0
    for _iteration in range(400):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if log_likelihood(first) < log_likelihood(second):
            low = first
        else:
            high = second
    return (low + high) / 2.0


def test_static_reduction_matches_bradley_terry_maximum_likelihood():
    # With w2 driven to zero the Wiener prior glues every node of a
    # player together, so a whole-history fit must collapse to the static
    # Bradley-Terry MLE. The initial-node prior is switched off so the
    # comparison is against the pure likelihood.
    games = []
    for time in (0, 5, 40, 41, 100):
        for _repeat in range(12):
            games.append(("A", "B", time, 1.0))
        for _repeat in range(5):
            games.append(("A", "B", time, 0.0))
        for _repeat in range(3):
            games.append(("A", "B", time, 0.5))

    result = whr.fit(games, 1e-8, initial_rd=None, tolerance=1e-10)

    expected = _static_bradley_terry_difference(12 * 5, 5 * 5, 3 * 5)
    nodes_a = result["players"]["A"]
    nodes_b = result["players"]["B"]
    assert len(nodes_a) == 5
    assert len(nodes_b) == 5
    for node_a, node_b in zip(nodes_a, nodes_b):
        assert abs((node_a["log_gamma"] - node_b["log_gamma"]) - expected) < 1e-3
    # Every node of a player is the same rating: that is what "static" means.
    for nodes in (nodes_a, nodes_b):
        assert max(node["log_gamma"] for node in nodes) - min(node["log_gamma"] for node in nodes) < 1e-3


def test_symmetric_results_give_equal_ratings_and_equal_bands():
    games = [
        ("A", "B", 0, 1.0),
        ("A", "B", 0, 0.0),
        ("A", "B", 10, 1.0),
        ("A", "B", 10, 0.0),
        ("A", "B", 20, 0.5),
    ]

    result = whr.fit(games, 15.0)

    for node_a, node_b in zip(result["players"]["A"], result["players"]["B"]):
        assert abs(node_a["log_gamma"] - node_b["log_gamma"]) < 1e-6
        assert abs(node_a["band"] - node_b["band"]) < 1e-6
    # Symmetric evidence about a prior centred on 1500 must leave both at 1500.
    for node in result["players"]["A"]:
        assert abs(node["rating"] - 1500.0) < 1e-6


def test_single_win_over_anchor_raises_rating_and_leaves_zero_gradient():
    # "Anchor" is held steady by a long history of exactly balanced
    # results against "ballast"; the player beats it exactly once.
    games = []
    for time in range(0, 40, 2):
        games.append(("anchor", "ballast", time, 1.0))
        games.append(("anchor", "ballast", time, 0.0))
    games.append(("player", "anchor", 20, 1.0))

    result = whr.fit(games, 15.0, tolerance=1e-10)

    assert result["converged"]
    assert result["players"]["player"][0]["rating"] > 1500.0
    # The fit's gauge puts the MEAN first-node rating at 1500, so a single
    # unbeaten newcomer pushes the rest of the field slightly below it;
    # what the single win has to buy is a clear margin over the anchor.
    assert result["players"]["player"][0]["rating"] - result["players"]["anchor"][0]["rating"] > 100.0
    # The anchor's one loss costs it a little against its balanced ballast,
    # but only a little: it is carrying forty other results.
    ballast_gap = result["players"]["ballast"][0]["rating"] - result["players"]["anchor"][0]["rating"]
    assert 0.0 < ballast_gap < 10.0

    # At a maximum the gradient of the log-posterior vanishes. Checking it
    # by finite differences (rather than against the analytic gradient the
    # solver itself uses) is what catches a sign error in the gradient or
    # the Hessian: a solver with a flipped sign would converge to a
    # perfectly self-consistent wrong point.
    players = whr._build_players(games)
    for name, nodes in result["players"].items():
        for index, node in enumerate(nodes):
            players[name]["log_gammas"][index] = node["log_gamma"]
    w2 = whr.w2_from_elo(15.0)
    initial_variance = whr.variance_from_rd(whr.DEFAULT_INITIAL_RD)
    step = 1e-5
    for name in players:
        for index in range(len(players[name]["log_gammas"])):
            original = players[name]["log_gammas"][index]
            players[name]["log_gammas"][index] = original + step
            up = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original - step
            down = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original
            assert abs((up - down) / (2.0 * step)) < 1e-3


def test_wiener_rate_controls_how_far_apart_distant_nodes_may_drift():
    # The same player wins everything early and loses everything late. A
    # large w2 lets the two nodes follow their local evidence; a small w2
    # forces them to move together.
    games = []
    for _repeat in range(10):
        games.append(("player", "opponent", 0, 1.0))
        games.append(("player", "opponent", 100, 0.0))

    loose = whr.fit(games, 400.0)
    tight = whr.fit(games, 0.01)

    loose_nodes = loose["players"]["player"]
    tight_nodes = tight["players"]["player"]
    loose_gap = loose_nodes[0]["log_gamma"] - loose_nodes[1]["log_gamma"]
    tight_gap = tight_nodes[0]["log_gamma"] - tight_nodes[1]["log_gamma"]

    assert loose_gap > 1.0
    assert abs(tight_gap) < 0.05
    assert loose_gap > tight_gap * 10.0
    # Under a loose prior the early node follows its wins upward and the
    # late node follows its losses downward.
    assert loose_nodes[0]["rating"] > 1500.0 > loose_nodes[1]["rating"]


def test_more_games_at_a_node_narrows_its_band():
    def band_for(repeat_count):
        games = []
        for _repeat in range(repeat_count):
            games.append(("player", "opponent", 0, 1.0))
            games.append(("player", "opponent", 0, 0.0))
        return whr.fit(games, 15.0)["players"]["player"][0]["band"]

    assert band_for(2) > band_for(10) > band_for(60)


def test_long_gap_widens_the_later_node_band_under_filtered_fitting():
    # Filtered fitting means the later node is estimated with no future
    # evidence at all, so it leans on the link back to the earlier node.
    # Stretch that link and the later node's band must widen.
    def band_after_gap(gap):
        games = []
        for _repeat in range(30):
            games.append(("player", "opponent", 0, 1.0))
            games.append(("player", "opponent", 0, 0.0))
        games.append(("player", "opponent", gap, 1.0))
        games.append(("player", "opponent", gap, 0.0))
        result = whr.fit(games, 15.0)
        return result["players"]["player"][-1]["band"]

    assert band_after_gap(500) > band_after_gap(50) > band_after_gap(2)


def test_analytic_gradient_matches_finite_differences_away_from_the_optimum():
    random.seed(20260807)
    names = ["p1", "p2", "p3", "p4"]
    times = [0, 7, 30]
    games = []
    for _game in range(12):
        first, second = random.sample(names, 2)
        games.append((first, second, random.choice(times), random.choice([1.0, 0.5, 0.0])))

    players = whr._build_players(games)
    for name in players:
        for index in range(len(players[name]["log_gammas"])):
            players[name]["log_gammas"][index] = random.uniform(-1.5, 1.5)

    w2 = whr.w2_from_elo(15.0)
    initial_variance = whr.variance_from_rd(whr.DEFAULT_INITIAL_RD)
    step = 1e-6
    for name in sorted(players):
        gradient, _diagonal, _off_diagonal = whr._player_system(players[name], players, w2, initial_variance)
        for index in range(len(gradient)):
            original = players[name]["log_gammas"][index]
            players[name]["log_gammas"][index] = original + step
            up = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original - step
            down = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original
            numeric = (up - down) / (2.0 * step)
            assert abs(numeric - gradient[index]) < 1e-4


def test_fit_is_deterministic():
    games = [
        ("A", "B", 0, 1.0),
        ("B", "C", 0, 0.5),
        ("A", "C", 3, 0.0),
        ("A", "B", 9, 1.0),
        ("B", "C", 9, 1.0),
    ]

    first = whr.fit(games, 15.0)
    second = whr.fit(games, 15.0)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_draw_against_a_stronger_player_raises_your_rating():
    # "strong" earns a high rating off a long run of wins over "weak".
    # A newcomer who only ever draws with "strong" should land above 1500,
    # and above a newcomer who only ever draws with "weak".
    games = []
    for time in range(0, 60, 2):
        games.append(("strong", "weak", time, 1.0))
    for time in (61, 63, 65):
        games.append(("newcomer_high", "strong", time, 0.5))
        games.append(("newcomer_low", "weak", time, 0.5))

    result = whr.fit(games, 15.0)

    strong = result["players"]["strong"][-1]["rating"]
    weak = result["players"]["weak"][-1]["rating"]
    assert strong > 1500.0 > weak

    high = result["players"]["newcomer_high"][-1]["rating"]
    low = result["players"]["newcomer_low"][-1]["rating"]
    assert high > 1500.0
    assert high > low
    assert low < 1500.0


def test_filtered_fit_at_the_final_time_equals_the_smoothed_fit():
    games = [
        ("A", "B", 0, 1.0),
        ("B", "C", 0, 0.5),
        ("A", "C", 5, 0.0),
        ("A", "B", 12, 1.0),
        ("B", "C", 12, 1.0),
        ("A", "C", 20, 1.0),
    ]

    smoothed = whr.fit(games, 15.0, tolerance=1e-10)

    warm_start = None
    filtered = None
    for limit in sorted({game[2] for game in games}):
        prefix = [game for game in games if game[2] <= limit]
        filtered = whr.fit(games=prefix, w2_elo=15.0, tolerance=1e-10, warm_start=warm_start)
        warm_start = whr.warm_start_from(filtered)

    # The last filtered fit saw exactly the same games as the smoothed
    # fit, so it must BE the smoothed fit.
    for name, nodes in smoothed["players"].items():
        for smoothed_node, filtered_node in zip(nodes, filtered["players"][name]):
            assert smoothed_node["time"] == filtered_node["time"]
            assert abs(smoothed_node["log_gamma"] - filtered_node["log_gamma"]) < 1e-6
            assert abs(smoothed_node["band"] - filtered_node["band"]) < 1e-6


def test_tridiagonal_inverse_diagonal_matches_a_direct_inverse():
    diagonal = [4.0, 5.0, 6.0, 3.5]
    off_diagonal = [-1.0, -2.0, -1.5]

    computed = whr.tridiagonal_inverse_diagonal(diagonal, off_diagonal)

    # Solve A x = e_i column by column with the same tridiagonal solver
    # and read the i-th entry; that is the inverse's diagonal by definition.
    for index in range(len(diagonal)):
        basis = [0.0] * len(diagonal)
        basis[index] = 1.0
        column = whr.solve_symmetric_tridiagonal(diagonal, off_diagonal, basis)
        assert abs(column[index] - computed[index]) < 1e-12


def test_tridiagonal_solve_raises_on_a_non_positive_pivot():
    try:
        whr.solve_symmetric_tridiagonal([1.0, 1.0], [2.0], [1.0, 1.0])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pivot" in str(exc)


def test_fit_rejects_a_non_positive_wiener_rate():
    try:
        whr.fit([("A", "B", 0, 1.0)], 0.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "w2_elo" in str(exc)


def test_expand_draws_splits_a_weighted_draw_evenly_between_the_sides():
    expanded = whr.expand_draws(
        [
            ("A", "B", 0, 1.0, 0.8),
            ("A", "B", 0, 0.0, 0.8),
            ("A", "B", 0, 0.5, 0.8),
            ("A", "B", 0, 1.0),
        ]
    )

    assert expanded == [
        ("A", "B", 0, 0.8),
        ("B", "A", 0, 0.8),
        ("A", "B", 0, 0.4),
        ("B", "A", 0, 0.4),
        ("A", "B", 0, 1.0),
    ]


def test_two_half_weight_games_equal_one_full_weight_game():
    # A weight is evidence, not a count: staking 0.5 twice on the same
    # reading is staking 1.0 once. If this failed, weighted comparisons
    # would not compose and no weighted fit could be interpreted.
    halved = []
    whole = []
    for game in (
        ("A", "B", 0, 1.0),
        ("B", "C", 0, 0.5),
        ("A", "C", 6, 0.0),
        ("A", "B", 14, 1.0),
        ("B", "C", 14, 0.5),
    ):
        player_a, player_b, time, score_a = game
        halved.append((player_a, player_b, time, score_a, 0.5))
        halved.append((player_a, player_b, time, score_a, 0.5))
        whole.append((player_a, player_b, time, score_a, 1.0))

    from_halves = whr.fit(halved, 15.0, tolerance=1e-12)
    from_wholes = whr.fit(whole, 15.0, tolerance=1e-12)

    for name, nodes in from_wholes["players"].items():
        for whole_node, half_node in zip(nodes, from_halves["players"][name]):
            assert whole_node["time"] == half_node["time"]
            assert abs(whole_node["log_gamma"] - half_node["log_gamma"]) < 1e-9
            assert abs(whole_node["band"] - half_node["band"]) < 1e-7


def test_analytic_gradient_matches_finite_differences_with_mixed_weights():
    # The same check as the unweighted case, but with every game carrying
    # its own weight and evaluated well away from the optimum, which is
    # where a weight dropped from the gradient (or from the Hessian, via
    # a Newton step that no longer matches its own gradient) shows up.
    random.seed(20260812)
    names = ["p1", "p2", "p3", "p4"]
    times = [0, 7, 30]
    games = []
    for _game in range(16):
        first, second = random.sample(names, 2)
        games.append(
            (
                first,
                second,
                random.choice(times),
                random.choice([1.0, 0.5, 0.0]),
                random.choice([0.15, 0.5, 0.83, 1.0]),
            )
        )

    players = whr._build_players(games)
    for name in players:
        for index in range(len(players[name]["log_gammas"])):
            players[name]["log_gammas"][index] = random.uniform(-1.5, 1.5)

    w2 = whr.w2_from_elo(15.0)
    initial_variance = whr.variance_from_rd(whr.DEFAULT_INITIAL_RD)
    step = 1e-6
    for name in sorted(players):
        gradient, _diagonal, _off_diagonal = whr._player_system(players[name], players, w2, initial_variance)
        for index in range(len(gradient)):
            original = players[name]["log_gammas"][index]
            players[name]["log_gammas"][index] = original + step
            up = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original - step
            down = whr.log_posterior(players, w2, initial_variance)
            players[name]["log_gammas"][index] = original
            numeric = (up - down) / (2.0 * step)
            assert abs(numeric - gradient[index]) < 1e-4


# Pinned from the unweighted implementation as it stood before game
# weights existed (commit 7751542). Weight 1.0 everywhere must reproduce
# it to the last digit, which is what makes the weighted generalization a
# generalization rather than a rewrite.
UNWEIGHTED_REGRESSION_GAMES = [
    ("Charlus", "Swann", 0, 1.0),
    ("Odette", "Swann", 0, 0.0),
    ("Charlus", "Odette", 0, 0.5),
    ("Swann", "Odette", 4, 1.0),
    ("Charlus", "Swann", 4, 0.0),
    ("Odette", "Charlus", 11, 1.0),
    ("Swann", "Charlus", 11, 0.5),
    ("Odette", "Swann", 11, 1.0),
    ("Charlus", "Odette", 30, 0.0),
    ("Swann", "Odette", 30, 0.5),
]
UNWEIGHTED_REGRESSION_LOG_GAMMAS = {
    "Charlus": [
        (0, -0.409085215531),
        (4, -0.410871253130),
        (11, -0.412778418794),
        (30, -0.416066760902),
    ],
    "Odette": [
        (0, 0.202101646499),
        (4, 0.203486990227),
        (11, 0.207647637058),
        (30, 0.210922337540),
    ],
    "Swann": [
        (0, 0.206983569032),
        (4, 0.207384262903),
        (11, 0.205130781736),
        (30, 0.205144423362),
    ],
}
UNWEIGHTED_REGRESSION_BANDS = {
    "Charlus": [274.001047, 273.945219, 274.062729, 275.548537],
    "Odette": [250.700636, 250.582585, 250.591762, 251.732411],
    "Swann": [250.486691, 250.367758, 250.573425, 252.247744],
}


def test_unweighted_fit_reproduces_the_pinned_pre_weight_fixture():
    implicit = whr.fit(UNWEIGHTED_REGRESSION_GAMES, 15.0, tolerance=1e-10)
    explicit = whr.fit(
        [game + (1.0,) for game in UNWEIGHTED_REGRESSION_GAMES], 15.0, tolerance=1e-10
    )

    assert json.dumps(implicit, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    for name, expected_nodes in UNWEIGHTED_REGRESSION_LOG_GAMMAS.items():
        nodes = implicit["players"][name]
        assert len(nodes) == len(expected_nodes)
        for node, (time, log_gamma) in zip(nodes, expected_nodes):
            assert node["time"] == time
            assert abs(node["log_gamma"] - log_gamma) < 1e-9
        for node, band in zip(nodes, UNWEIGHTED_REGRESSION_BANDS[name]):
            assert abs(node["band"] - band) < 1e-5


def test_fit_rejects_a_non_positive_game_weight():
    for weight in (0.0, -0.5):
        try:
            whr.fit([("A", "B", 0, 1.0, weight)], 15.0)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "weight" in str(exc)


def test_fit_rejects_a_malformed_game_tuple():
    try:
        whr.fit([("A", "B", 0)], 15.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "player_a" in str(exc)


def test_a_lighter_game_moves_a_rating_less():
    # Two identical head-to-head histories, one staked at full weight and
    # one at a tenth of it: uncertainty must weigh, and only weigh.
    def gap_at(weight):
        games = [("winner", "loser", time, 1.0, weight) for time in range(0, 20, 2)]
        result = whr.fit(games, 15.0, tolerance=1e-10)
        return result["players"]["winner"][-1]["rating"] - result["players"]["loser"][-1]["rating"]

    assert gap_at(1.0) > gap_at(0.4) > gap_at(0.1) > 0.0


def test_rating_scale_round_trips():
    assert abs(whr.to_rating(0.0) - 1500.0) < 1e-12
    assert abs(whr.from_rating(whr.to_rating(1.25)) - 1.25) < 1e-12
    # One log-gamma point is a factor of e in odds, which on the Elo scale
    # is 400 / ln 10 points.
    assert abs(whr.to_rating_span(1.0) - 400.0 / math.log(10.0)) < 1e-12
    assert abs(whr.win_probability(0.0, 0.0) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# The corpus surface (proust/character_whr.py).
# ---------------------------------------------------------------------------


def _annotation(unit_id, characters):
    characters_present = []
    appraisal_events = []
    status_effects = []
    for index, item in enumerate(characters, start=1):
        character = item["character"]
        delta = item.get("delta", 1)
        dimension = item.get("dimension", "social_status")
        event_id = f"E{index}"
        characters_present.append(
            {
                "canonical_name": character,
                "surface_forms": [character],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        )
        appraisal_events.append(
            {
                "event_id": event_id,
                "source": "narrator",
                "target": character,
                "type": "admiration" if delta >= 0 else "narrated_diminishment",
                "polarity": "positive" if delta >= 0 else "negative",
                "narrative_stance": "endorsed",
                "confidence": 1.0,
                "evidence": "x",
                "explanation": "x",
            }
        )
        status_effects.append(
            {
                "character": character,
                "dimension": dimension,
                "delta": delta,
                "based_on_events": [event_id],
                "confidence": 1.0,
                "explanation": "x",
            }
        )
    return {
        "unit_id": unit_id,
        "characters_present": characters_present,
        "appraisal_events": appraisal_events,
        "status_effects": status_effects,
        "ambiguities": [],
    }


def _make_synthetic_corpus(tmp_path):
    # Two chapters, two units, so the corpus spans more than one point of
    # narrative time and every character has a trajectory rather than a
    # single node.
    run_dir = tmp_path / "run-001"
    pn.prepare_annotation_run(
        run_dir,
        unit_specs=[
            AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=17, paragraph_end=None, notes=""),
            AnnotationUnitSpec(
                chapter_id="v1-p2-un-amour-de-swann", paragraph_start=1, paragraph_end=None, notes=""
            ),
        ],
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [
                {"character": "Swann", "delta": 1},
                {"character": "Odette", "delta": -1},
                {"character": "Albertine", "delta": -1},
            ],
        ),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p2-un-amour-de-swann#p-1",
        _annotation(
            "v1-p2-un-amour-de-swann#p-1",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    return run_dir


def test_build_character_whr_produces_a_coherent_artifact(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    analysis = pr.build_character_whr([run_dir])

    assert analysis["character_whr_version"] == "character_whr_advantage_v1"
    assert analysis["lens"] == "advantage"
    assert analysis["mode"] == "both"
    assert analysis["time_axis"] == "cumulative_unit_index"
    assert analysis["draw_model"] == "half_win_half_loss"
    assert analysis["conservative_rating_rule"] == "rating_minus_band"
    assert analysis["epsilon"] == 0.25
    # Combray contributes 3 matches, Un Amour de Swann 1 more.
    assert analysis["match_count"] == 4
    assert analysis["character_count"] == 3
    assert analysis["time_point_count"] == 2

    # w2 is chosen by prediction, not asserted, and every candidate's
    # score is kept.
    assert analysis["w2_elo_selected_by"] == "sequential_one_step_ahead_log_loss"
    assert analysis["w2_elo"] in analysis["w2_elo_candidates"]
    assert len(analysis["predictive_evaluation"]["whr_candidates"]) == len(analysis["w2_elo_candidates"])
    systems = {row["system"] for row in analysis["predictive_evaluation"]["comparison"]}
    assert {"whr_filtered", "elo_sequential", "glicko2_chapter_period"} <= systems
    for row in analysis["predictive_evaluation"]["comparison"]:
        assert row["match_count"] == analysis["match_count"]
        assert row["log_loss"] > 0.0
        assert 0.0 <= row["brier"] <= 1.0

    rows = {row["character"]: row for row in analysis["characters"]}
    assert set(rows) == {"Swann", "Odette", "Albertine"}

    assert rows["Swann"]["rating"] > 1500.0
    assert rows["Swann"]["rating"] > rows["Odette"]["rating"]
    assert rows["Swann"]["match_count"] == 3
    assert rows["Swann"]["win_count"] == 3
    assert rows["Odette"]["draw_count"] == 1
    assert rows["Albertine"]["draw_count"] == 1

    for row in analysis["characters"]:
        assert row["band"] > 0.0
        assert abs(row["conservative_rating"] - (row["rating"] - row["band"])) < 0.05
        assert row["provisional"] == (row["band"] > analysis["band_provisional_threshold"])
        # A character plays at every time they appear with a co-scored
        # character, and each such time is one trajectory point.
        assert row["node_count"] == len(row["smoothed_trajectory"])
        assert len(row["filtered_trajectory"]) == row["node_count"]
        assert [point[0] for point in row["smoothed_trajectory"]] == [
            point[0] for point in row["filtered_trajectory"]
        ]
        assert row["smoothed_summary"]["point_count"] == row["node_count"]
        assert row["first_time"] == row["smoothed_trajectory"][0][0]
        assert row["last_time"] == row["smoothed_trajectory"][-1][0]

    # Trajectory times are real narrative positions, described in `times`.
    known_times = {entry["time"] for entry in analysis["times"]}
    for row in analysis["characters"]:
        for point in row["smoothed_trajectory"]:
            assert point[0] in known_times

    for row in analysis["top_rated_characters"] + analysis["bottom_rated_characters"]:
        assert not row["provisional"]
    for row in analysis["provisional_characters"]:
        assert row["provisional"]


def test_build_character_whr_mode_controls_which_trajectories_are_written(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    smoothed_only = pr.build_character_whr([run_dir], mode="smoothed")
    filtered_only = pr.build_character_whr([run_dir], mode="filtered")

    for row in smoothed_only["characters"]:
        assert "smoothed_trajectory" in row
        assert "filtered_trajectory" not in row
    for row in filtered_only["characters"]:
        assert "filtered_trajectory" in row
        assert "smoothed_trajectory" not in row
    # The headline rating is the smoothed one either way.
    smoothed_rows = {row["character"]: row["rating"] for row in smoothed_only["characters"]}
    filtered_rows = {row["character"]: row["rating"] for row in filtered_only["characters"]}
    assert smoothed_rows == filtered_rows


def test_build_character_whr_is_deterministic(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    first = pr.build_character_whr([run_dir], w2_elo=15.0)
    second = pr.build_character_whr([run_dir], w2_elo=15.0)

    first.pop("wall_clock_seconds")
    second.pop("wall_clock_seconds")
    for analysis in (first, second):
        for row in analysis["predictive_evaluation"]["whr_candidates"]:
            row.pop("wall_clock_seconds")
        for row in analysis["predictive_evaluation"]["comparison"]:
            row.pop("wall_clock_seconds", None)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_character_whr_with_supplements_marks_metadata_and_gains_matches(tmp_path):
    accepted_run = tmp_path / "run-001"
    supplement_run = tmp_path / "supplement-run-001"
    pn.prepare_annotation_run(accepted_run)
    pn.prepare_annotation_run(supplement_run)
    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    pn.write_annotation_result(
        supplement_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "le narrateur", "delta": 2}]),
    )

    baseline = pr.build_character_whr([accepted_run], w2_elo=15.0)
    supplemented = pr.build_character_whr(
        [accepted_run], supplement_run_dirs=[supplement_run], w2_elo=15.0
    )

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]
    assert supplemented["match_count"] == baseline["match_count"] + 2

    rows = {row["character"]: row for row in supplemented["characters"]}
    assert "le narrateur" in rows
    assert rows["le narrateur"]["match_count"] == 2
    assert rows["le narrateur"]["win_count"] == 2
    assert rows["le narrateur"]["rating"] > rows["Swann"]["rating"]


def test_build_character_whr_rejects_unknown_lens_and_mode(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    try:
        pr.build_character_whr([run_dir], lens="notalens")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "notalens" in str(exc)

    try:
        pr.build_character_whr([run_dir], mode="notamode")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "notamode" in str(exc)


def test_render_character_whr_markdown_covers_the_required_sections(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    analysis = pr.build_character_whr([run_dir], w2_elo=15.0)

    markdown = pr.render_character_whr_markdown(analysis)

    assert markdown.startswith("# Character Whole-History Rating\n")
    assert "## Predictive Comparison" in markdown
    assert "### w2 Selection" in markdown
    assert "## Final Standings" in markdown
    assert "## Provisional Characters" in markdown
    assert "## Trajectory Summaries" in markdown
    assert "±" in markdown


def test_main_character_whr_lens_aware_default_paths(tmp_path, capsys, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        ["character-whr", "--discover-runs", str(outputs_dir), "--lens", "prestige", "--w2", "15"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-whr-prestige-current.json"
    assert payload["markdown_output"] == "outputs/character-whr-prestige-current.md"
    assert payload["w2_elo"] == 15.0
    assert (tmp_path / "outputs" / "character-whr-prestige-current.json").exists()
    assert (tmp_path / "outputs" / "character-whr-prestige-current.md").exists()

    exit_code_supplemented = pr.main(
        [
            "character-whr",
            "--discover-runs",
            str(outputs_dir),
            "--include-supplements",
            "--supplement-outputs-dir",
            str(outputs_dir),
            "--w2",
            "15",
        ]
    )
    payload_supplemented = json.loads(capsys.readouterr().out)
    assert exit_code_supplemented == 0
    assert payload_supplemented["json_output"] == "outputs/character-whr-advantage-supplemented-current.json"


# ---------------------------------------------------------------------------
# The app-facing WHR timeline (build_character_whr_timeline).
# ---------------------------------------------------------------------------


def test_build_character_whr_timeline_joins_every_node_to_a_corpus_position(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    target_characters = ["Swann", "Odette", "Albertine"]

    timeline = pn.build_character_whr_timeline(
        [run_dir], target_characters=target_characters, w2_elo=15.0
    )

    assert timeline["character_whr_timeline_version"] == "character_whr_timeline_advantage_v1"
    assert timeline["lens"] == "advantage"
    assert timeline["time_axis"] == "cumulative_unit_index"
    assert timeline["modes"] == ["smoothed", "filtered"]
    assert timeline["tracked_characters"] == target_characters
    assert timeline["tracked_character_count"] == 3
    assert timeline["point_count"] == len(timeline["points"])
    assert timeline["point_count"] > 0

    for point in timeline["points"]:
        assert point["character"] in target_characters
        assert point["mode"] in ("smoothed", "filtered")
        assert isinstance(point["rating"], float)
        assert isinstance(point["band"], float)
        assert isinstance(point["net_score"], float)
        assert point["label"] in ("win", "loss", "mixed", "neutral")
        assert point["unit_character_count"] >= 2
        position = point["corpus_position"]
        assert position["unit_id"]
        assert position["chapter_id"]
        assert position["cumulative_unit_index"] > 0
        assert position["cumulative_word_count"] >= 0

    # Every character with any nodes has points in BOTH modes, in equal
    # numbers -- filtered and smoothed trajectories share the same times.
    modes_by_character = defaultdict(set)
    counts_by_character_mode = defaultdict(lambda: defaultdict(int))
    for point in timeline["points"]:
        modes_by_character[point["character"]].add(point["mode"])
        counts_by_character_mode[point["character"]][point["mode"]] += 1
    for character, modes in modes_by_character.items():
        assert modes == {"smoothed", "filtered"}
        assert (
            counts_by_character_mode[character]["smoothed"]
            == counts_by_character_mode[character]["filtered"]
        )

    # Points are sorted by (character, mode, cumulative_unit_index).
    sort_keys = [
        (point["character"], point["mode"], point["corpus_position"]["cumulative_unit_index"])
        for point in timeline["points"]
    ]
    assert sort_keys == sorted(sort_keys)

    # The per-character summary agrees with what the points actually contain.
    for row in timeline["characters"]:
        character = row["character"]
        assert row["smoothed_point_count"] == counts_by_character_mode[character]["smoothed"]
        assert row["filtered_point_count"] == counts_by_character_mode[character]["filtered"]
        assert row["node_count"] == row["smoothed_point_count"]


def test_build_character_whr_timeline_filters_to_target_characters(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    timeline = pn.build_character_whr_timeline([run_dir], target_characters=["Swann"], w2_elo=15.0)

    assert timeline["tracked_characters"] == ["Swann"]
    assert timeline["tracked_character_count"] == 1
    assert {point["character"] for point in timeline["points"]} == {"Swann"}
    assert [row["character"] for row in timeline["characters"]] == ["Swann"]


def test_build_character_whr_timeline_keeps_a_tracked_character_absent_from_the_corpus(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    timeline = pn.build_character_whr_timeline(
        [run_dir], target_characters=["Swann", "Nobody"], w2_elo=15.0
    )

    assert {point["character"] for point in timeline["points"]} == {"Swann"}
    rows = {row["character"]: row for row in timeline["characters"]}
    assert rows["Nobody"]["node_count"] == 0
    assert rows["Nobody"]["smoothed_point_count"] == 0
    assert rows["Nobody"]["filtered_point_count"] == 0
    assert rows["Nobody"]["final_rating"] == 1500.0


def test_build_character_whr_timeline_reuses_a_precomputed_whr_analysis(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    whr_analysis = pn.build_character_whr([run_dir], w2_elo=15.0)

    timeline = pn.build_character_whr_timeline(
        [run_dir], target_characters=["Swann"], whr_analysis=whr_analysis
    )

    assert timeline["w2_elo"] == 15.0
    assert timeline["point_count"] > 0


def test_build_character_whr_timeline_rejects_a_whr_analysis_that_is_not_mode_both(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    whr_analysis = pn.build_character_whr([run_dir], w2_elo=15.0, mode="smoothed")

    try:
        pn.build_character_whr_timeline([run_dir], whr_analysis=whr_analysis)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "mode" in str(exc)


def test_build_character_whr_timeline_raises_when_a_node_cannot_be_resolved(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    whr_analysis = pn.build_character_whr([run_dir], w2_elo=15.0)
    rows = {row["character"]: row for row in whr_analysis["characters"]}
    # Corrupt one node's time to a narrative position the corpus position
    # index -- built fresh from the same run_dirs -- cannot know about.
    rows["Swann"]["smoothed_trajectory"][0][0] = 999999

    try:
        pn.build_character_whr_timeline(
            [run_dir], target_characters=["Swann"], whr_analysis=whr_analysis
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "999999" in str(exc)
        assert "corpus position" in str(exc)


def test_render_character_whr_timeline_markdown_covers_the_required_sections(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    timeline = pn.build_character_whr_timeline(
        [run_dir], target_characters=["Swann", "Odette", "Albertine"], w2_elo=15.0
    )

    markdown = pn.render_character_whr_timeline_markdown(timeline)

    assert markdown.startswith("# Character WHR Timeline\n")
    assert "## Character Coverage" in markdown
    assert "±" in markdown
    # No full point dump: the JSON artifact carries the points, not the Markdown.
    assert "Cumulative Unit" not in markdown


def test_main_character_whr_timeline_lens_aware_default_paths(tmp_path, capsys, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        ["character-whr-timeline", "--discover-runs", str(outputs_dir), "--lens", "prestige", "--w2", "15"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-whr-prestige-timeline-current.json"
    assert payload["markdown_output"] == "outputs/character-whr-prestige-timeline-current.md"
    assert payload["w2_elo"] == 15.0
    # "Swann" and "Odette" are both in the character-page pilot set, so
    # the default tracked set (no --character flag on this command) picks
    # them up without any extra configuration.
    assert payload["point_count"] > 0
    assert (tmp_path / "outputs" / "character-whr-prestige-timeline-current.json").exists()
    assert (tmp_path / "outputs" / "character-whr-prestige-timeline-current.md").exists()

    exit_code_supplemented = pr.main(
        [
            "character-whr-timeline",
            "--discover-runs",
            str(outputs_dir),
            "--include-supplements",
            "--supplement-outputs-dir",
            str(outputs_dir),
            "--w2",
            "15",
        ]
    )
    payload_supplemented = json.loads(capsys.readouterr().out)
    assert exit_code_supplemented == 0
    assert (
        payload_supplemented["json_output"]
        == "outputs/character-whr-advantage-timeline-supplemented-current.json"
    )
