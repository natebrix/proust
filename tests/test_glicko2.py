import json
import math

import proust as pn
import proust.runner as pr
from proust import glicko2 as g2
from proust.annotation import AnnotationUnitSpec


# ---------------------------------------------------------------------------
# Pure Glicko-2 math (proust/glicko2.py), no corpus involved.
# ---------------------------------------------------------------------------


def test_rate_player_matches_glickman_paper_worked_example():
    # "Example of the Glicko-2 System" (glicko.net/glicko/glicko2.pdf):
    # a player rated 1500 with RD 200 and volatility 0.06 plays three games
    # in one rating period against opponents (1400, 30) win, (1550, 100)
    # loss, (1700, 300) loss, with tau = 0.5. The paper works this through
    # by hand to r' ~= 1464.06, RD' ~= 151.52, sigma' ~= 0.05999.
    rating, rd, volatility = 1500.0, 200.0, 0.06
    opponents = [
        (1400.0, 30.0, 1.0),
        (1550.0, 100.0, 0.0),
        (1700.0, 300.0, 0.0),
    ]

    rating_prime, rd_prime, volatility_prime = g2.rate_player(rating, rd, volatility, opponents, tau=0.5)

    assert abs(rating_prime - 1464.06) < 0.01
    assert abs(rd_prime - 151.52) < 0.01
    assert abs(volatility_prime - 0.05999) < 0.0001


def test_rate_player_glicko2_scale_matches_rating_scale_wrapper():
    rating, rd, volatility = 1500.0, 200.0, 0.06
    opponents = [
        (1400.0, 30.0, 1.0),
        (1550.0, 100.0, 0.0),
        (1700.0, 300.0, 0.0),
    ]

    rating_prime, rd_prime, volatility_prime = g2.rate_player(rating, rd, volatility, opponents, tau=0.5)

    mu, phi = g2.to_glicko2_scale(rating, rd)
    scaled_opponents = [(*g2.to_glicko2_scale(r, d), s) for r, d, s in opponents]
    mu_prime, phi_prime, sigma_prime = g2.rate_player_glicko2_scale(mu, phi, volatility, scaled_opponents, tau=0.5)
    rating_prime_2, rd_prime_2 = g2.from_glicko2_scale(mu_prime, phi_prime)

    assert abs(rating_prime - rating_prime_2) < 1e-9
    assert abs(rd_prime - rd_prime_2) < 1e-9
    assert abs(volatility_prime - sigma_prime) < 1e-9


def test_rate_player_with_zero_games_raises_rd_only():
    rating, rd, volatility = 1500.0, 200.0, 0.06

    rating_prime, rd_prime, volatility_prime = g2.rate_player(rating, rd, volatility, [], tau=0.5)

    assert rating_prime == rating
    assert volatility_prime == volatility
    assert rd_prime > rd
    # phi' = sqrt(phi^2 + sigma^2) on the Glicko-2 scale.
    _mu, phi = g2.to_glicko2_scale(rating, rd)
    expected_phi_prime = math.sqrt(phi ** 2 + volatility ** 2)
    _expected_rating, expected_rd_prime = g2.from_glicko2_scale(0.0, expected_phi_prime)
    assert abs(rd_prime - expected_rd_prime) < 1e-9


def test_rate_player_repeated_zero_games_keeps_raising_rd_toward_a_ceiling():
    rating, rd, volatility = 1500.0, 50.0, 0.06

    for _ in range(10):
        rating, rd, volatility = g2.rate_player(rating, rd, volatility, [], tau=0.5)

    assert rating == 1500.0
    assert volatility == 0.06
    assert rd > 50.0


# ---------------------------------------------------------------------------
# Corpus-level builder (proust/app_exports.py::build_character_glicko2).
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
    # Two chapters (two rating periods): Combray gives Swann two wins over
    # Odette and Albertine (three-way scene forces a draw between Odette and
    # Albertine, whose deltas are equal); Un Amour de Swann gives Swann a
    # rematch against Odette, so period 2 uses period 1's post-update
    # ratings as the pre-period opponent state.
    #
    # A default prepared run only declares units within v1-p1-combray (the
    # three STARTER_UNITS), so a second chapter's unit must be requested
    # explicitly via unit_specs to exercise the multi-period batch path.
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


def test_build_character_glicko2_produces_coherent_artifact(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    analysis = pr.build_character_glicko2([run_dir])

    assert analysis["character_glicko2_version"] == "character_glicko2_advantage_v1"
    assert analysis["lens"] == "advantage"
    assert analysis["rating_period_rule"] == "canonical_chapter"
    assert analysis["tau"] == 0.5
    assert analysis["epsilon"] == 0.25
    assert analysis["initial_rating"] == 1500.0
    assert analysis["initial_rd"] == 350.0
    assert analysis["initial_volatility"] == 0.06
    assert analysis["rd_provisional_threshold"] == 100.0
    # Combray contributes 3 matches (Swann-Odette, Swann-Albertine,
    # Odette-Albertine); Un Amour de Swann contributes 1 more.
    assert analysis["match_count"] == 4
    assert analysis["character_count"] == 3
    assert analysis["period_count"] > 0

    rows = {row["character"]: row for row in analysis["characters"]}
    assert set(rows) == {"Swann", "Odette", "Albertine"}

    # Swann won every match he played; his rating should end up above the
    # initial rating, and above both Odette's and Albertine's.
    assert rows["Swann"]["rating"] > 1500.0
    assert rows["Swann"]["rating"] > rows["Odette"]["rating"]
    assert rows["Swann"]["rating"] > rows["Albertine"]["rating"]
    # Swann plays Odette and Albertine in Combray, then Odette again in Un
    # Amour de Swann: 3 matches, all wins. Odette plays Swann twice (losses)
    # and draws Albertine once in Combray: 3 matches. Albertine plays Swann
    # once (loss) and draws Odette once in Combray: 2 matches.
    assert rows["Swann"]["match_count"] == 3
    assert rows["Swann"]["win_count"] == 3
    assert rows["Odette"]["match_count"] == 3
    assert rows["Odette"]["loss_count"] == 2
    assert rows["Odette"]["draw_count"] == 1
    assert rows["Albertine"]["match_count"] == 2
    assert rows["Albertine"]["loss_count"] == 1
    assert rows["Albertine"]["draw_count"] == 1

    # conservative_rating = rating - 2*rd for every row (computed from
    # unrounded internal state, so it may differ from the rounded rating/rd
    # columns by a tenth of a point).
    for row in analysis["characters"]:
        assert abs(row["conservative_rating"] - (row["rating"] - 2 * row["rd"])) < 0.15
        assert isinstance(row["provisional"], bool)
        assert row["provisional"] == (row["rd"] > analysis["rd_provisional_threshold"])
        # Every row must carry an ELO cross-reference for the divergence
        # diagnostic.
        assert "elo_rank" in row
        assert "glicko_rank" in row

    # Summary lists are well-formed and only ever contain non-provisional
    # characters (this fixture's characters all have very few matches, so
    # they may all be provisional -- the invariant to check is the filter,
    # not that the lists are non-empty).
    for row in analysis["top_rated_characters"] + analysis["bottom_rated_characters"]:
        assert not row["provisional"]
    for row in analysis["largest_glicko_elo_rank_divergences"]:
        assert not row["provisional"]
    for row in analysis["provisional_characters"]:
        assert row["provisional"]


def test_build_character_glicko2_draws_are_counted(tmp_path):
    # A draw is scored when both characters' net scores are within epsilon
    # of each other (default epsilon = 0.25). Equal deltas at -1 apiece
    # produce a net score difference of 0, well inside the draw band.
    run_dir = _make_synthetic_corpus(tmp_path)
    analysis = pr.build_character_glicko2([run_dir])

    rows = {row["character"]: row for row in analysis["characters"]}
    assert rows["Odette"]["draw_count"] == 1
    assert rows["Albertine"]["draw_count"] == 1
    assert analysis["draw_rate"] > 0.0


def test_build_character_glicko2_is_deterministic(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    first = pr.build_character_glicko2([run_dir])
    second = pr.build_character_glicko2([run_dir])

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_character_glicko2_zero_match_character_has_rising_rd_and_unchanged_rating(tmp_path):
    # A character who appears solo in a unit (no co-scored character) never
    # gets a pairwise match, but must still be tracked from the period they
    # first appear in; their RD should rise afterward through later periods
    # of inactivity while their rating and volatility hold steady at the
    # initial values.
    run_dir = tmp_path / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Lonely", "delta": 1}]),
    )

    analysis = pr.build_character_glicko2([run_dir])

    lonely = next(row for row in analysis["characters"] if row["character"] == "Lonely")
    assert lonely["match_count"] == 0
    assert lonely["rating"] == 1500.0
    assert lonely["volatility"] == 0.06
    assert lonely["rd"] > 350.0
    assert lonely["unit_count"] == 1


def test_build_character_glicko2_with_supplements_marks_metadata_and_gains_matches(tmp_path):
    accepted_run = tmp_path / "run-001"
    supplement_run = tmp_path / "supplement-run-001"
    pn.prepare_annotation_run(accepted_run)
    pn.prepare_annotation_run(supplement_run)
    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Swann", "delta": 1}]),
    )
    pn.write_annotation_result(
        supplement_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "le narrateur", "delta": -1}]),
    )

    baseline = pr.build_character_glicko2([accepted_run])
    supplemented = pr.build_character_glicko2([accepted_run], supplement_run_dirs=[supplement_run])

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]
    assert supplemented["match_count"] == baseline["match_count"] + 1

    rows = {row["character"]: row for row in supplemented["characters"]}
    assert "le narrateur" in rows
    assert rows["le narrateur"]["match_count"] == 1


def test_build_character_glicko2_rejects_unknown_lens(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)

    try:
        pr.build_character_glicko2([run_dir], lens="notalens")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "notalens" in str(exc)


def test_render_character_glicko2_markdown_renders_ratings_as_confidence_band(tmp_path):
    run_dir = _make_synthetic_corpus(tmp_path)
    analysis = pr.build_character_glicko2([run_dir])

    markdown = pr.render_character_glicko2_markdown(analysis)

    assert markdown.startswith("# Character Glicko-2\n")
    assert "±" in markdown


def test_main_character_glicko2_lens_aware_default_paths(tmp_path, capsys, monkeypatch):
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
        [
            "character-glicko2",
            "--discover-runs",
            str(outputs_dir),
            "--lens",
            "prestige",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-glicko2-prestige-current.json"
    assert payload["markdown_output"] == "outputs/character-glicko2-prestige-current.md"
    assert (tmp_path / "outputs" / "character-glicko2-prestige-current.json").exists()

    exit_code_supplemented = pr.main(
        [
            "character-glicko2",
            "--discover-runs",
            str(outputs_dir),
            "--include-supplements",
            "--supplement-outputs-dir",
            str(outputs_dir),
        ]
    )
    payload_supplemented = json.loads(capsys.readouterr().out)
    assert exit_code_supplemented == 0
    assert payload_supplemented["json_output"] == "outputs/character-glicko2-advantage-supplemented-current.json"
