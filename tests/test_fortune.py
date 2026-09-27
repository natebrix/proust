import math
import random

import numpy as np

from proust import fortune


def _dense_posterior(observations, q, sigma2, prior_mean, prior_sd):
    """The local-level model solved by brute force: one joint Gaussian over every node.

    Independent of fortune.py's recursions, so it checks both the smoother
    and the marginal likelihood.
    """
    times = sorted({time for time, _value, _weight in observations})
    index = {time: i for i, time in enumerate(times)}
    n = len(times)
    prior_cov = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            prior_cov[i, j] = prior_sd ** 2 + q * (min(times[i], times[j]) - times[0])
    design = np.zeros((len(observations), n))
    noise = np.zeros(len(observations))
    values = np.zeros(len(observations))
    for row, (time, value, weight) in enumerate(observations):
        design[row, index[time]] = 1.0
        noise[row] = sigma2 / max(weight, fortune.MIN_WEIGHT)
        values[row] = value

    observation_cov = design @ prior_cov @ design.T + np.diag(noise)
    residual = values - prior_mean
    sign, log_det = np.linalg.slogdet(observation_cov)
    assert sign > 0
    log_likelihood = -0.5 * (
        len(values) * math.log(2.0 * math.pi) + log_det + residual @ np.linalg.solve(observation_cov, residual)
    )

    gain = prior_cov @ design.T @ np.linalg.inv(observation_cov)
    mean = prior_mean + gain @ residual
    cov = prior_cov - gain @ design @ prior_cov
    return times, mean, np.diag(cov), log_likelihood


def test_smoother_and_likelihood_match_dense_gaussian():
    rng = random.Random(3)
    observations = []
    time = 0.0
    for _ in range(12):
        time += rng.choice([0.0, 0.4, 1.3, 5.0])
        observations.append((round(time, 3), rng.uniform(-2, 2), rng.uniform(0.2, 1.0)))
    q, sigma2, prior_mean, prior_sd = 0.05, 0.6, -0.2, 0.5

    nodes, log_likelihood = fortune.filter_and_smooth(
        observations, q, sigma2, prior_mean=prior_mean, prior_sd=prior_sd
    )
    times, mean, var, dense_log_likelihood = _dense_posterior(observations, q, sigma2, prior_mean, prior_sd)

    assert [node["time"] for node in nodes] == times
    for node, expected_mean, expected_var in zip(nodes, mean, var):
        assert math.isclose(node["smoothed"], expected_mean, abs_tol=1e-9)
        assert math.isclose(node["smoothed_var"], expected_var, abs_tol=1e-9)
    assert math.isclose(log_likelihood, dense_log_likelihood, abs_tol=1e-9)
    assert nodes[-1]["smoothed"] == nodes[-1]["filtered"]


def test_hyperparameters_see_a_step_and_ignore_noise():
    rng = random.Random(11)
    stepped = {
        f"c{k}": [(float(t), (1.0 if t < 20 else -1.0) + rng.gauss(0, 0.3), 1.0) for t in range(40)]
        for k in range(6)
    }
    flat = {f"c{k}": [(float(t), rng.gauss(0, 0.8), 1.0) for t in range(40)] for k in range(6)}

    q_step, _sigma2, _table = fortune.select_hyperparameters(stepped, 0.0)
    q_flat, _sigma2, _table = fortune.select_hyperparameters(flat, 0.0)
    assert q_step > q_flat
    assert q_flat == min(fortune.DEFAULT_Q_GRID)


def test_arc_summary_finds_drawdown_after_peak():
    nodes = [
        {"time": float(t), "smoothed": level, "smoothed_var": 0.01, "filtered": level, "filtered_var": 0.01, "n": 1}
        for t, level in enumerate([0.0, 0.8, 0.2, 1.0, -1.5, -1.0])
    ]
    summary = fortune.arc_summary(nodes)
    assert math.isclose(summary["biggest_fall"]["size"], 2.5)
    assert summary["biggest_fall"]["from_time"] == 3.0
    assert summary["biggest_fall"]["to_time"] == 4.0
    assert math.isclose(summary["biggest_rise"]["size"], 1.0)
    assert summary["peak"]["time"] == 3.0 and summary["trough"]["time"] == 4.0


def test_order_permutation_separates_a_fall_from_its_shuffle():
    declining = [(float(t), 1.0 - t / 10.0, 1.0) for t in range(20)]
    p = fortune.order_permutation_p_values(declining, 0.05, 0.3, 0.0, random.Random(1), samples=200)
    assert p["fall"] < 0.02
    assert p["rise"] > 0.5


def test_unit_observations_only_lens_participants_and_overall_sums():
    annotation = {
        "characters_present": [
            {"canonical_name": "Charlus", "presence_confidence": 1.0},
            {"canonical_name": "Morel", "presence_confidence": 1.0},
            {"canonical_name": "Jupien", "presence_confidence": 1.0},
        ],
        "appraisal_events": [{"event_id": "E1", "narrative_stance": "endorsed"}],
        "status_effects": [
            {"character": "Charlus", "dimension": "social_status", "delta": -2, "confidence": 0.5,
             "based_on_events": ["E1"]},
            {"character": "Charlus", "dimension": "general_appraisal", "delta": -1, "confidence": 1.0,
             "based_on_events": ["E1"]},
            {"character": "Morel", "dimension": "general_appraisal", "delta": 1, "confidence": 1.0,
             "based_on_events": ["E1"]},
        ],
        "ambiguities": ["one"],
    }
    prestige = fortune.unit_observations(annotation, "prestige")
    assert set(prestige) == {"Charlus"}
    assert math.isclose(prestige["Charlus"][0], -1.0)
    assert math.isclose(prestige["Charlus"][1], 0.8 * 0.5)

    overall = fortune.unit_observations(annotation, fortune.OVERALL_LENS)
    assert set(overall) == {"Charlus", "Morel"}  # Jupien was present but untouched
    assert math.isclose(overall["Charlus"][0], -2.0)
    assert math.isclose(overall["Charlus"][1], 0.8 * (0.5 + 1.0) / 2)
    assert math.isclose(overall["Morel"][0], 1.0)
