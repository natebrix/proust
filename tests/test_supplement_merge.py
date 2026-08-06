import json

import proust as pn
import proust.app_exports as pa
import proust.runner as pr


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


def _make_accepted_and_supplement_runs(tmp_path):
    accepted_run = tmp_path / "run-001"
    supplement_run = tmp_path / "supplement-run-001"
    pn.prepare_annotation_run(accepted_run)
    pn.prepare_annotation_run(supplement_run)
    return accepted_run, supplement_run


def test_build_chapter_overlay_data_merges_supplement_characters_without_touching_accepted(tmp_path):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)

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

    baseline = pr.build_chapter_overlay_data([accepted_run])
    dataset = pr.build_chapter_overlay_data([accepted_run], supplement_run_dirs=[supplement_run])

    baseline_chapter = next(c for c in baseline["chapters"] if c["chapterId"] == "v1-p1-combray")
    baseline_unit = next(u for u in baseline_chapter["units"] if u["unitId"] == "v1-p1-combray#p-17")
    assert [row["character"] for row in baseline_unit["characters"]] == ["Swann"]
    assert "provenance" not in baseline_unit["characters"][0]
    assert "supplement_run_count" not in baseline

    chapter = next(c for c in dataset["chapters"] if c["chapterId"] == "v1-p1-combray")
    unit = next(u for u in chapter["units"] if u["unitId"] == "v1-p1-combray#p-17")
    rows_by_character = {row["character"]: row for row in unit["characters"]}
    assert set(rows_by_character) == {"Swann", "le narrateur"}
    assert "provenance" not in rows_by_character["Swann"]
    assert rows_by_character["le narrateur"]["provenance"] == "supplement"
    # The accepted row's own data is untouched by the merge.
    assert rows_by_character["Swann"]["advantage"] == baseline_unit["characters"][0]["advantage"]

    assert dataset["supplement_run_count"] == 1
    assert dataset["supplement_runs"] == ["supplement-run-001"]
    assert dataset["supplemented_unit_count"] == 1
    assert dataset["supplement_collision_count"] == 0


def test_build_chapter_overlay_data_skips_colliding_supplement_character_and_counts_it(tmp_path):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)

    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Swann", "delta": 1}]),
    )
    pn.write_annotation_result(
        supplement_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Swann", "delta": -1}]),
    )

    dataset = pr.build_chapter_overlay_data([accepted_run], supplement_run_dirs=[supplement_run])

    chapter = next(c for c in dataset["chapters"] if c["chapterId"] == "v1-p1-combray")
    unit = next(u for u in chapter["units"] if u["unitId"] == "v1-p1-combray#p-17")
    assert [row["character"] for row in unit["characters"]] == ["Swann"]
    # Accepted delta was +1 (a "win"); the colliding supplement delta of -1
    # must never overwrite it.
    assert unit["characters"][0]["advantage"]["label"] == "win"
    assert dataset["supplement_collision_count"] == 1
    assert dataset["supplemented_unit_count"] == 0


def test_build_chapter_overlay_data_default_call_matches_explicit_none(tmp_path):
    accepted_run, _supplement_run = _make_accepted_and_supplement_runs(tmp_path)
    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Swann", "delta": 1}]),
    )

    default_call = pr.build_chapter_overlay_data([accepted_run])
    explicit_none_call = pr.build_chapter_overlay_data([accepted_run], supplement_run_dirs=None)

    assert json.dumps(default_call, sort_keys=True) == json.dumps(explicit_none_call, sort_keys=True)
    assert "supplement_run_count" not in default_call


def test_build_character_elo_with_supplements_gains_matches_and_marks_metadata(tmp_path):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)
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

    baseline = pr.build_character_elo([accepted_run])
    supplemented = pr.build_character_elo([accepted_run], supplement_run_dirs=[supplement_run])

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]
    # The new "le narrateur" row pairs with the pre-existing "Swann" row in
    # the same unit, producing exactly one additional pairwise match.
    assert supplemented["match_count"] == baseline["match_count"] + 1

    rows = {row["character"]: row for row in supplemented["characters"]}
    assert "le narrateur" in rows
    assert rows["le narrateur"]["match_count"] == 1


def test_build_character_elo_min_match_count_filters_summary_tables_not_full_table(tmp_path):
    run_dir = tmp_path / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Lonely", "delta": 1}]),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        _annotation(
            "v1-p1-combray#p-274-p-275",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )

    analysis = pr.build_character_elo([run_dir], min_match_count=1)

    assert analysis["min_match_count"] == 1
    lonely_row = next(row for row in analysis["characters"] if row["character"] == "Lonely")
    assert lonely_row["match_count"] == 0
    assert lonely_row["elo"] == 1500.0
    assert not any(row["character"] == "Lonely" for row in analysis["top_rated_characters"])
    assert not any(row["character"] == "Lonely" for row in analysis["lowest_rated_characters"])
    assert not any(row["character"] == "Lonely" for row in analysis["largest_rank_mismatches"])
    # The full characters table is never filtered.
    assert any(row["character"] == "Lonely" for row in analysis["characters"])


def test_build_character_elo_timeline_with_supplements_marks_metadata(tmp_path):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)
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

    baseline = pr.build_character_elo_timeline([accepted_run], target_characters=["Swann", "le narrateur"])
    supplemented = pr.build_character_elo_timeline(
        [accepted_run],
        target_characters=["Swann", "le narrateur"],
        supplement_run_dirs=[supplement_run],
    )

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]
    supplemented_rows = {row["character"]: row for row in supplemented["characters"]}
    assert supplemented_rows["le narrateur"]["point_count"] == 1


def test_discover_supplement_run_dirs_finds_supplement_run_directories_with_manifests(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    run_a = outputs_dir / "supplement-run-001"
    run_b = outputs_dir / "supplement-run-002"
    incomplete_run = outputs_dir / "supplement-run-003"
    accepted_run = outputs_dir / "run-001"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    incomplete_run.mkdir()
    pn.prepare_annotation_run(accepted_run)

    found = pa.discover_supplement_run_dirs(str(outputs_dir))

    assert [path.name for path in found] == ["supplement-run-001", "supplement-run-002"]


def test_discover_supplement_run_dirs_returns_empty_list_when_outputs_dir_missing(tmp_path):
    assert pa.discover_supplement_run_dirs(str(tmp_path / "does-not-exist")) == []


def test_build_character_elo_supplement_diff_reports_before_after(tmp_path):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)
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

    baseline = pr.build_character_elo([accepted_run])
    supplemented = pr.build_character_elo([accepted_run], supplement_run_dirs=[supplement_run])

    diff = pr.build_character_elo_supplement_diff(baseline, supplemented)

    assert diff["character_elo_supplement_diff_version"] == "character_elo_supplement_diff_v1"
    assert diff["match_count"]["before"] == baseline["match_count"]
    assert diff["match_count"]["after"] == supplemented["match_count"]
    assert diff["character_count"]["before"] == baseline["character_count"]
    assert diff["character_count"]["after"] == supplemented["character_count"]

    movers_by_character = {row["character"]: row for row in diff["top_rating_movers"]}
    assert "le narrateur" in movers_by_character
    assert movers_by_character["le narrateur"]["elo_before"] is None
    assert movers_by_character["le narrateur"]["delta"] is None
    assert movers_by_character["le narrateur"]["match_count_before"] == 0
    assert movers_by_character["le narrateur"]["match_count_after"] == 1

    baseline_swann_elo = next(row for row in baseline["characters"] if row["character"] == "Swann")["elo"]
    assert movers_by_character["Swann"]["elo_before"] == baseline_swann_elo

    assert pr.render_character_elo_supplement_diff_markdown(diff).startswith("# Character ELO Supplement Diff\n")
