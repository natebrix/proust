"""Tests for the scoring v2 promotion (proust/scoring_v2_promote.py).

Promotion is the step that makes v2 the current analysis surface, so what
these tests pin is the presentation policy the adoption asked for rather
than the formula (tests/test_scoring_v2.py pins that): the ranked set and
the not-yet-rankable set are two sections and never one table, a trajectory
node reaches the app with the corpus position it was fitted at, and a
dossier page's profile block is v2 numbers with a null rank wherever the
evidence is too thin for a placement.
"""

import json

import pytest

from proust import scoring_v2, scoring_v2_build, scoring_v2_promote


# ---------------------------------------------------------------- fixtures


def _character(name, presence_confidence=0.9):
    return {
        "canonical_name": name,
        "surface_forms": [name],
        "presence_type": "explicit",
        "presence_confidence": presence_confidence,
    }


def _effect(character, dimension, delta, confidence=1.0, explanation="explanation"):
    return {
        "character": character,
        "dimension": dimension,
        "delta": delta,
        "based_on_events": ["E1"],
        "confidence": confidence,
        "explanation": explanation,
    }


def _annotation(characters, effects=(), unit_id="v1-p1-combray#p-1-p-5"):
    return {
        "unit_id": unit_id,
        "characters_present": list(characters),
        "appraisal_events": [],
        "status_effects": list(effects),
        "ambiguities": [],
    }


def _unit(time, annotation, chapter_id="v1-p1-combray"):
    """A loaded unit, with the full corpus position load_scored_units attaches."""
    return {
        "unit_id": annotation["unit_id"],
        "run_id": "foundation-run-001",
        "chapter_id": chapter_id,
        "chapter_index": 1,
        "volume_number": 1,
        "unit_index_within_chapter": time,
        "time": time,
        "corpus_position": {
            "volume_number": 1,
            "chapter_id": chapter_id,
            "chapter_title": "Combray",
            "chapter_index": 1,
            "unit_id": annotation["unit_id"],
            "unit_index_within_chapter": time,
            "cumulative_unit_index": time,
            "paragraph_start": time,
            "paragraph_end": time + 4,
            "cumulative_paragraph_index": time,
            "cumulative_paragraph_index_end": time + 4,
            "cumulative_word_count": 100 * time,
            "cumulative_word_count_end": 100 * time + 400,
        },
        "annotation": annotation,
    }


def _standing_row(character, rating, band, provisional=False, mean_movement=0.5, labels=None):
    return {
        "character": character,
        "rating": rating,
        "band": band,
        "conservative_rating": round(rating - band, 1),
        "provisional": provisional,
        "match_count": 40,
        "win_count": 20,
        "loss_count": 10,
        "draw_count": 10,
        "unit_count": 30,
        "weight_total": 30.0,
        "mean_weight": 0.75,
        "mean_movement": mean_movement,
        "mean_absolute_movement": abs(mean_movement),
        "labels": labels or {"positive": 10, "negative": 5, "mixed": 1, "neutral": 14},
        "node_count": 3,
        "first_time": 1,
        "last_time": 3,
        "smoothed_summary": {"point_count": 3, "first": [1, rating, band], "last": [3, rating, band]},
        "filtered_summary": {"point_count": 3, "first": [1, rating, band], "last": [3, rating, band]},
        "smoothed_trajectory": [[1, rating, band], [2, rating, band], [3, rating, band]],
        "filtered_trajectory": [[1, rating, band], [2, rating, band], [3, rating, band]],
        "rank": None,
    }


def _ratings(characters, lens="advantage", view="name"):
    return {
        "scoring_v2_ratings_version": f"scoring_v2_{lens}_{view}_view_v1",
        "scoring_version": scoring_v2.SCORING_V2_VERSION,
        "lens": lens,
        "view": view,
        "time_axis": "cumulative_unit_index",
        "w2_elo": 15.0,
        "w2_elo_selected_by": "caller",
        "tie_band": scoring_v2.TIE_BAND,
        "draw_model": "half_win_half_loss",
        "initial_rating": 1500.0,
        "initial_rd": 200.0,
        "band_provisional_threshold": 200.0,
        "conservative_rating_rule": "rating_minus_band",
        "character_count": len(characters),
        "non_provisional_count": len([row for row in characters if not row["provisional"]]),
        "comparison_count": 120,
        "weight_total": 90.0,
        "mean_weight": 0.75,
        "draw_rate": 0.1,
        "time_point_count": 3,
        "node_count": 3 * len(characters),
        "convergence": {"tolerance": 1e-6, "max_sweeps": 200},
        "predictive_evaluation": {"protocol": "one-step-ahead", "comparison": [], "whr_candidates": []},
        "dropped_self_pairings": 0,
        "characters": characters,
    }


# ---------------------------------------------------------------- standings


def test_standings_split_the_ranked_set_from_the_insufficient_evidence_set():
    ratings = _ratings(
        [
            _standing_row("Swann", 1620.0, 68.0),
            _standing_row("le narrateur", 1552.0, 77.0),
            _standing_row("Saniette", 1300.0, 90.0),
            _standing_row("Cottard", 1580.0, 410.0, provisional=True),
            _standing_row("Mme de Cambremer", 1490.0, 380.0, provisional=True),
        ]
    )

    standings = scoring_v2_promote.build_character_standings(ratings, corpus="foundation")

    assert standings["ranked_count"] == 3
    assert standings["insufficient_evidence_count"] == 2
    assert standings["character_count"] == 5
    # ranked by conservative rating, which is not the rating order: the
    # narrator's wider band costs him nothing here, but it is what the
    # ordering is made of.
    assert [row["character"] for row in standings["ranked"]] == ["Swann", "le narrateur", "Saniette"]
    assert [row["rank"] for row in standings["ranked"]] == [1, 2, 3]
    # the second section is ordered by rating and carries no rank at all
    assert [row["character"] for row in standings["insufficient_evidence"]] == [
        "Cottard",
        "Mme de Cambremer",
    ]
    assert all(row["rank"] is None for row in standings["insufficient_evidence"])
    assert standings["corpus"] == "foundation"


def test_standings_leave_the_trajectories_in_the_staged_fit():
    # The point-by-point series is large and already on disk; republishing
    # it would put the same numbers in two places that can drift.
    ratings = _ratings([_standing_row("Swann", 1620.0, 68.0)])

    standings = scoring_v2_promote.build_character_standings(ratings)

    row = standings["ranked"][0]
    assert "smoothed_trajectory" not in row and "filtered_trajectory" not in row
    assert row["smoothed_summary"]["point_count"] == 3
    assert standings["trajectory_source"].endswith("scoring-v2-advantage-name-view-ratings.json")
    # and the staged fit itself is untouched
    assert ratings["characters"][0]["smoothed_trajectory"]


def test_standings_rank_densely_so_a_tie_shares_one_rank():
    ratings = _ratings(
        [
            _standing_row("Swann", 1600.0, 100.0),
            _standing_row("Bloch", 1550.0, 50.0),
            _standing_row("Saniette", 1400.0, 100.0),
        ]
    )

    standings = scoring_v2_promote.build_character_standings(ratings)

    # Swann and Bloch both sit at a conservative 1500.0
    assert [(row["character"], row["rank"]) for row in standings["ranked"]] == [
        ("Bloch", 1),
        ("Swann", 1),
        ("Saniette", 2),
    ]


def test_standings_markdown_frames_the_second_section_as_missing_evidence():
    ratings = _ratings(
        [
            _standing_row("le narrateur", 1552.0, 77.0),
            _standing_row("Cottard", 1580.0, 410.0, provisional=True),
        ]
    )

    markdown = scoring_v2_promote.render_character_standings_markdown(
        scoring_v2_promote.build_character_standings(ratings)
    )

    assert "## Ranked" in markdown
    assert "## Insufficient comparative evidence" in markdown
    assert "1552 ± 77" in markdown
    assert "NOT THE BOTTOM OF THE TABLE ABOVE" in markdown
    ranked_section, insufficient_section = markdown.split("## Insufficient comparative evidence")
    assert "Cottard" not in ranked_section
    assert "Cottard" in insufficient_section


# ------------------------------------------------------- journey timelines


def _timeline_corpus():
    units = []
    for index in range(1, 4):
        units.append(
            _unit(
                index,
                _annotation(
                    [_character("Swann"), _character("Odette")],
                    effects=[
                        _effect("Swann", "general_appraisal", -2, explanation=f"Swann loses, {index}"),
                        _effect("Odette", "general_appraisal", 1),
                    ],
                    unit_id=f"v1-p1-combray#p-{index}-p-{index + 4}",
                ),
            )
        )
    return units


def test_journey_timeline_joins_every_node_to_its_full_corpus_position():
    units = _timeline_corpus()
    readings = scoring_v2_build.build_readings(units, "advantage")
    ratings = _ratings([_standing_row("Swann", 1400.0, 80.0)])

    timeline = scoring_v2_promote.build_character_journey_timeline(
        ratings, units, readings, target_characters=["Swann"]
    )

    assert timeline["tracked_characters"] == ["Swann"]
    # three nodes in each of the two modes
    assert timeline["point_count"] == 6
    assert timeline["characters"][0]["smoothed_point_count"] == 3
    assert timeline["characters"][0]["filtered_point_count"] == 3

    point = timeline["points"][0]
    assert point["mode"] == "filtered"
    assert point["rating"] == 1400.0 and point["band"] == 80.0
    assert point["label"] == "negative" and point["movement"] < 0
    assert point["unit_character_count"] == 2
    # the whole position record, not the fit's reduced one: an app places a
    # point on a reading axis with the paragraph and word offsets.
    assert point["corpus_position"]["paragraph_start"] == 1
    assert point["corpus_position"]["paragraph_end"] == 5
    assert point["corpus_position"]["cumulative_word_count"] == 100
    assert point["corpus_position"]["chapter_title"] == "Combray"
    # and each mode's points run forward through the corpus
    smoothed = [p for p in timeline["points"] if p["mode"] == "smoothed"]
    times = [p["corpus_position"]["cumulative_unit_index"] for p in smoothed]
    assert times == sorted(times)


def test_journey_timeline_raises_when_the_fit_and_the_corpus_have_drifted_apart():
    units = _timeline_corpus()
    readings = scoring_v2_build.build_readings(units, "advantage")
    ratings = _ratings([_standing_row("Swann", 1400.0, 80.0)])
    ratings["characters"][0]["smoothed_trajectory"] = [[99, 1400.0, 80.0]]

    with pytest.raises(ValueError, match="no corpus position"):
        scoring_v2_promote.build_character_journey_timeline(
            ratings, units, readings, target_characters=["Swann"]
        )


def test_journey_timeline_requires_units_carrying_their_corpus_position():
    units = _timeline_corpus()
    readings = scoring_v2_build.build_readings(units, "advantage")
    for unit in units:
        del unit["corpus_position"]

    with pytest.raises(ValueError, match="carries no corpus_position"):
        scoring_v2_promote.build_character_journey_timeline(
            _ratings([_standing_row("Swann", 1400.0, 80.0)]), units, readings, target_characters=["Swann"]
        )


# ------------------------------------------------------------------- pages


def _pages_inputs():
    units = []
    for index in range(1, 4):
        # the third unit is the one that moves the narrator hardest
        delta = -2 if index == 3 else -1
        units.append(
            _unit(
                index,
                _annotation(
                    [_character("le narrateur"), _character("Swann")],
                    effects=[
                        _effect(
                            "le narrateur",
                            "general_appraisal",
                            delta,
                            explanation=f"The narrator is snubbed, unit {index}",
                        ),
                        _effect("le narrateur", "social_status", 1),
                        _effect("Swann", "general_appraisal", 1),
                    ],
                    unit_id=f"v1-p1-combray#p-{index}-p-{index + 4}",
                ),
            )
        )
    units.append(
        _unit(
            4,
            _annotation(
                [_character("le narrateur"), _character("Swann")],
                effects=[_effect("le narrateur", "general_appraisal", 1)],
                unit_id="v1-p2-un-amour-de-swann#p-1-p-5",
            ),
            chapter_id="v1-p2-un-amour-de-swann",
        )
    )

    lenses = list(scoring_v2.SCORING_V2_LENS_ORDER)
    readings_by_lens = {lens: scoring_v2_build.build_readings(units, lens) for lens in lenses}
    ratings_by_lens = {
        "advantage": _ratings(
            [
                _standing_row("le narrateur", 1368.7, 87.2, mean_movement=-0.3045),
                _standing_row("Swann", 1620.0, 68.0, mean_movement=0.4),
            ],
            lens="advantage",
        ),
        "prestige": _ratings(
            [
                _standing_row("le narrateur", 1702.0, 170.7, mean_movement=0.0259),
                _standing_row("Swann", 1480.0, 410.0, provisional=True),
            ],
            lens="prestige",
        ),
        "inclusion": _ratings(
            [
                _standing_row("le narrateur", 1520.1, 101.3, mean_movement=0.0514),
                _standing_row("Swann", 1500.0, 420.0, provisional=True),
            ],
            lens="inclusion",
        ),
    }
    summary = scoring_v2_build.build_corpus_summary(ratings_by_lens, readings_by_lens)
    standings_by_lens = {
        lens: scoring_v2_promote.build_character_standings(ratings_by_lens[lens]) for lens in lenses
    }
    return units, readings_by_lens, summary, standings_by_lens


def test_pages_profile_carries_the_v2_lens_block():
    units, readings_by_lens, summary, standings = _pages_inputs()

    pages = scoring_v2_promote.build_character_pages_v2(
        units, readings_by_lens, summary, standings, target_characters=["le narrateur", "Swann"]
    )

    assert pages["character_pages_version"] == "character_pages_v2"
    page = next(row for row in pages["pages"] if row["character"] == "le narrateur")
    # the page machinery's key names survive the rescoring
    assert set(page) == {
        "character",
        "slug",
        "portrait",
        "profile",
        "editorial",
        "top_chapters",
        "reading_path",
        "notable_units",
    }
    assert page["slug"] == "le-narrateur"
    assert page["profile"]["archetype_signs"] == {"advantage": -1, "prestige": 1, "inclusion": 1}
    assert page["profile"]["annotation_unit_count"] == 4

    advantage = page["profile"]["lens_scores"]["advantage"]
    assert set(advantage) == {
        "rating",
        "band",
        "conservative_rating",
        "rank",
        "non_provisional_count",
        "provisional",
        "appearances",
        "mean_movement",
        "mean_absolute_movement",
        "labels",
        "comparison_count",
    }
    assert advantage["rating"] == 1368.7 and advantage["band"] == 87.2
    assert advantage["rank"] == 2 and advantage["non_provisional_count"] == 2
    assert advantage["provisional"] is False
    assert advantage["appearances"] == 4


def test_pages_report_no_rank_where_the_evidence_is_insufficient():
    units, readings_by_lens, summary, standings = _pages_inputs()

    pages = scoring_v2_promote.build_character_pages_v2(
        units, readings_by_lens, summary, standings, target_characters=["Swann"]
    )

    prestige = pages["pages"][0]["profile"]["lens_scores"]["prestige"]
    assert prestige["provisional"] is True
    # a wide band is missing evidence, so there is no placement to report
    assert prestige["rank"] is None
    assert prestige["non_provisional_count"] == 1


def test_pages_notable_units_are_the_biggest_v2_movements_with_the_annotator_reason():
    units, readings_by_lens, summary, standings = _pages_inputs()

    pages = scoring_v2_promote.build_character_pages_v2(
        units, readings_by_lens, summary, standings, target_characters=["le narrateur"]
    )

    notable = pages["pages"][0]["notable_units"]
    assert notable[0]["unit_id"] == "v1-p1-combray#p-3-p-7"
    assert notable[0]["label"] == "The narrator is snubbed, unit 3"
    assert notable[0]["movements"]["advantage"] == -2.0
    assert notable[0]["reader_link"].endswith("/v1-p1-combray#p-3")
    scores = [row["max_absolute_movement"] for row in notable]
    assert scores == sorted(scores, reverse=True)


def test_pages_top_chapters_rank_by_absolute_movement():
    units, readings_by_lens, summary, standings = _pages_inputs()

    pages = scoring_v2_promote.build_character_pages_v2(
        units, readings_by_lens, summary, standings, target_characters=["le narrateur"]
    )

    chapters = pages["pages"][0]["top_chapters"]
    assert [row["chapter_id"] for row in chapters] == [
        "v1-p1-combray",
        "v1-p2-un-amour-de-swann",
    ]
    assert chapters[0]["unit_count"] == 3
    assert chapters[0]["advantage"]["absolute_movement"] == 4.0
    assert chapters[0]["chapter_title"].endswith("Combray")


def test_pages_markdown_documents_the_new_profile_shape():
    units, readings_by_lens, summary, standings = _pages_inputs()
    pages = scoring_v2_promote.build_character_pages_v2(
        units, readings_by_lens, summary, standings, target_characters=["le narrateur"]
    )

    markdown = scoring_v2_promote.render_character_pages_v2_markdown(pages)

    assert "## Profile shape" in markdown
    assert "`rank`, `non_provisional_count`" in markdown
    assert "1369 ± 87" in markdown
    assert "2 of 2" in markdown


# ------------------------------------------------------------------ naming


def test_promoted_artifact_names_are_the_current_family(tmp_path):
    assert scoring_v2_promote.standings_paths("advantage", outputs_dir=tmp_path)[0].endswith(
        "character-standings-advantage-current.json"
    )
    assert scoring_v2_promote.standings_paths(
        "advantage", view="person", outputs_dir=tmp_path
    )[0].endswith("character-standings-advantage-person-view-current.json")
    assert scoring_v2_promote.journey_timeline_paths("prestige", outputs_dir=tmp_path)[1].endswith(
        "character-journey-prestige-timeline-current.md"
    )
    assert scoring_v2_promote.character_pages_paths(outputs_dir=tmp_path)[0].endswith(
        "character-pages-current.json"
    )


def test_promotion_refuses_to_run_without_the_staged_fits(tmp_path):
    with pytest.raises(ValueError, match="scripts/build_scoring_v2.py"):
        scoring_v2_promote.read_staged_ratings("advantage", "name", staged_dir=tmp_path)


def test_standings_artifacts_are_written_where_they_are_asked_for(tmp_path):
    standings = scoring_v2_promote.build_character_standings(
        _ratings([_standing_row("Swann", 1620.0, 68.0)])
    )
    json_output = tmp_path / "character-standings-advantage-current.json"
    markdown_output = tmp_path / "character-standings-advantage-current.md"

    written = scoring_v2_promote.write_character_standings_artifacts(
        standings, json_output=json_output, markdown_output=markdown_output
    )

    assert len(written) == 2
    payload = json.loads(json_output.read_text())
    assert payload["ranked"][0]["character"] == "Swann"
    assert markdown_output.read_text().startswith("# Character Standings — advantage")
