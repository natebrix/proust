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


def _make_two_unit_run(tmp_path):
    run_dir = tmp_path / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        _annotation(
            "v1-p1-combray#p-274-p-275",
            [{"character": "Swann", "delta": 1}, {"character": "Albertine", "delta": -1}],
        ),
    )
    return run_dir


def test_build_character_elo_prestige_lens_builds_from_synthetic_fixtures(tmp_path):
    run_dir = _make_two_unit_run(tmp_path)

    analysis = pr.build_character_elo([run_dir], lens="prestige")

    assert analysis["character_elo_version"] == "character_elo_prestige_v1"
    assert analysis["lens"] == "prestige"
    assert analysis["match_count"] == 2
    rows = {row["character"]: row for row in analysis["characters"]}
    # Prestige weights are nonzero and same-signed as advantage for the
    # admiration/narrated_diminishment events used by the fixture, so Swann
    # (positive delta in both units) still comes out ahead.
    assert rows["Swann"]["elo"] > rows["Odette"]["elo"]
    assert rows["Swann"]["elo"] > rows["Albertine"]["elo"]
    # Every row carries the lens-generic field; the advantage-only alias must
    # not leak into a non-advantage lens.
    for row in analysis["characters"]:
        assert "mean_net_score" in row
        assert "mean_advantage_net_score" not in row

    timeline = pr.build_character_elo_timeline(
        [run_dir],
        lens="prestige",
        target_characters=["Swann", "Odette"],
    )
    assert timeline["character_elo_timeline_version"] == "character_elo_timeline_prestige_v1"
    assert timeline["lens"] == "prestige"
    for point in timeline["points"]:
        assert "net_score" in point
        assert "label" in point
        assert "advantage_net_score" not in point
        assert "advantage_label" not in point

    assert pr.render_character_elo_markdown(analysis).startswith("# Character ELO\n")
    assert "Mean Prestige" in pr.render_character_elo_markdown(analysis)
    assert pr.render_character_elo_timeline_markdown(timeline).startswith("# Character ELO Timeline\n")


def test_build_character_elo_advantage_version_strings_and_alias_unchanged(tmp_path):
    run_dir = _make_two_unit_run(tmp_path)

    analysis = pr.build_character_elo([run_dir])
    timeline = pr.build_character_elo_timeline([run_dir], target_characters=["Swann", "Odette"])

    assert analysis["character_elo_version"] == "character_elo_advantage_v1"
    assert timeline["character_elo_timeline_version"] == "character_elo_timeline_v1"

    for row in analysis["characters"]:
        assert row["mean_advantage_net_score"] == row["mean_net_score"]
    for point in timeline["points"]:
        assert point["advantage_net_score"] == point["net_score"]
        assert point["advantage_label"] == point["label"]

    markdown = pr.render_character_elo_markdown(analysis)
    assert "Mean Advantage" in markdown
    assert "Mean Prestige" not in markdown
    timeline_markdown = pr.render_character_elo_timeline_markdown(timeline)
    assert "| Character | ELO | Advantage | Label |" in timeline_markdown.replace("  ", " ")


def test_build_character_elo_rejects_unknown_lens(tmp_path):
    run_dir = _make_two_unit_run(tmp_path)

    try:
        pr.build_character_elo([run_dir], lens="notalens")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "notalens" in str(exc)

    try:
        pr.build_character_elo_timeline([run_dir], lens="notalens")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "notalens" in str(exc)


def test_main_character_elo_lens_aware_default_paths(tmp_path, capsys, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    pn.prepare_annotation_run(run_a)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _annotation(
            "v1-p1-combray#p-17",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        [
            "character-elo",
            "--discover-runs",
            str(outputs_dir),
            "--lens",
            "prestige",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-elo-prestige-current.json"
    assert payload["markdown_output"] == "outputs/character-elo-prestige-current.md"
    assert (tmp_path / "outputs" / "character-elo-prestige-current.json").exists()

    # Advantage default paths are unaffected by the new lens-aware logic.
    exit_code_advantage = pr.main(
        [
            "character-elo",
            "--discover-runs",
            str(outputs_dir),
        ]
    )
    payload_advantage = json.loads(capsys.readouterr().out)
    assert exit_code_advantage == 0
    assert payload_advantage["json_output"] == "outputs/character-elo-advantage-current.json"


def test_main_elo_supplement_diff_lens_aware_default_paths(tmp_path, capsys, monkeypatch):
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path / "outputs")
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
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        [
            "elo-supplement-diff",
            "--discover-runs",
            str(tmp_path / "outputs"),
            "--supplement-outputs-dir",
            str(tmp_path / "outputs"),
            "--lens",
            "prestige",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["elo_output"] == "outputs/character-elo-prestige-supplemented-current.json"
    assert payload["elo_timeline_output"] == "outputs/character-elo-prestige-timeline-supplemented-current.json"
    assert payload["diff_output"] == "outputs/character-elo-prestige-supplement-diff-current.json"
    assert (tmp_path / "outputs" / "character-elo-prestige-supplement-diff-current.json").exists()


def _make_pilot_supplement_scenario(tmp_path):
    # A small fixture used across the profile-card/chapter-analysis/pages
    # supplement tests below: "Swann" and "Odette" are accepted-scored in
    # two units; "le narrateur" is entirely absent from the accepted
    # annotation (mirroring the pre-supplement production state) and only
    # appears via the supplement run, in a unit the accepted annotation
    # already covers (unit A, additive alongside Swann) and in a unit the
    # accepted annotation never touched at all (unit C, which also grows
    # Swann's own unit count). All three units share the same chapter
    # ("v1-p1-combray") so chapter-level aggregation can be checked too.
    accepted_run, supplement_run = _make_accepted_and_supplement_runs(tmp_path)
    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "Swann", "delta": 1}]),
    )
    pn.write_annotation_result(
        accepted_run,
        "v1-p1-combray#p-274-p-275",
        _annotation(
            "v1-p1-combray#p-274-p-275",
            [{"character": "Swann", "delta": 1}, {"character": "Odette", "delta": -1}],
        ),
    )
    pn.write_annotation_result(
        supplement_run,
        "v1-p1-combray#p-17",
        _annotation("v1-p1-combray#p-17", [{"character": "le narrateur", "delta": -1}]),
    )
    pn.write_annotation_result(
        supplement_run,
        "v1-p1-combray#p-312-p-313",
        _annotation(
            "v1-p1-combray#p-312-p-313",
            [{"character": "le narrateur", "delta": 1}, {"character": "Swann", "delta": 1}],
        ),
    )
    return accepted_run, supplement_run


def test_build_corpus_sanity_review_merges_supplement_characters_without_touching_accepted(tmp_path):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path)

    baseline = pr.build_corpus_sanity_review([accepted_run])
    review = pr.build_corpus_sanity_review([accepted_run], supplement_run_dirs=[supplement_run])

    assert "supplemented" not in baseline
    assert review["supplemented"] is True
    assert review["supplement_runs"] == ["supplement-run-001"]

    baseline_totals = {row["character"]: row for row in baseline["lens_reviews"]["advantage"]["character_totals"]}
    merged_totals = {row["character"]: row for row in review["lens_reviews"]["advantage"]["character_totals"]}

    assert "le narrateur" not in baseline_totals
    assert merged_totals["le narrateur"]["unit_count"] == 2
    # Swann gains one unit (unit C) from the supplement without losing the
    # two units already accepted.
    assert baseline_totals["Swann"]["unit_count"] == 2
    assert merged_totals["Swann"]["unit_count"] == 3
    # A character with no supplement contribution is untouched.
    assert merged_totals["Odette"] == baseline_totals["Odette"]
    # Run-level accepted-only accounting (never fed by supplements) is
    # unaffected by the merge.
    assert review["run_count"] == baseline["run_count"]
    assert review["declared_unit_count"] == baseline["declared_unit_count"]
    assert review["run_ids"] == baseline["run_ids"]


def test_build_corpus_sanity_review_default_call_matches_explicit_none(tmp_path):
    accepted_run, _supplement_run = _make_pilot_supplement_scenario(tmp_path)

    default_call = pr.build_corpus_sanity_review([accepted_run])
    explicit_none_call = pr.build_corpus_sanity_review([accepted_run], supplement_run_dirs=None)

    assert json.dumps(default_call, sort_keys=True) == json.dumps(explicit_none_call, sort_keys=True)
    assert "supplemented" not in default_call


def test_build_character_profile_cards_with_supplements_includes_new_characters_and_higher_counts(tmp_path):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path)

    baseline = pr.build_character_profile_cards([accepted_run])
    supplemented = pr.build_character_profile_cards([accepted_run], supplement_run_dirs=[supplement_run])

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]

    baseline_cards = {row["character"]: row for row in baseline["cards"]}
    supplemented_cards = {row["character"]: row for row in supplemented["cards"]}

    assert "le narrateur" not in baseline_cards
    assert supplemented_cards["le narrateur"]["annotation_unit_count"] == 2
    assert baseline_cards["Swann"]["annotation_unit_count"] == 2
    assert supplemented_cards["Swann"]["annotation_unit_count"] == 3
    # Odette's own totals are unaffected by the merge (her rank/percentile
    # can still shift because the population gained "le narrateur").
    assert supplemented_cards["Odette"]["annotation_unit_count"] == baseline_cards["Odette"]["annotation_unit_count"]
    assert supplemented_cards["Odette"]["lens_scores"]["advantage"]["net_score"] == (
        baseline_cards["Odette"]["lens_scores"]["advantage"]["net_score"]
    )


def test_build_character_profile_cards_default_call_matches_explicit_none(tmp_path):
    accepted_run, _supplement_run = _make_pilot_supplement_scenario(tmp_path)

    default_call = pr.build_character_profile_cards([accepted_run])
    explicit_none_call = pr.build_character_profile_cards([accepted_run], supplement_run_dirs=None)

    assert json.dumps(default_call, sort_keys=True) == json.dumps(explicit_none_call, sort_keys=True)
    assert "supplemented" not in default_call


def test_build_character_chapter_analysis_with_supplements_merges_chapter_rows_and_marks_metadata(tmp_path):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path)

    baseline = pr.build_character_chapter_analysis([accepted_run], target_characters=["Swann"])
    supplemented = pr.build_character_chapter_analysis(
        [accepted_run],
        target_characters=["Swann", "le narrateur"],
        supplement_run_dirs=[supplement_run],
    )

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]

    baseline_swann_chapters = {
        row["chapter_id"]: row
        for row in next(row for row in baseline["characters"] if row["character"] == "Swann")["chapters"]
    }
    supplemented_swann_chapters = {
        row["chapter_id"]: row
        for row in next(row for row in supplemented["characters"] if row["character"] == "Swann")["chapters"]
    }
    supplemented_narrator_chapters = {
        row["chapter_id"]: row
        for row in next(row for row in supplemented["characters"] if row["character"] == "le narrateur")["chapters"]
    }

    baseline_swann_unit_count = max(
        baseline_swann_chapters["v1-p1-combray"]["advantage"]["unit_count"],
        baseline_swann_chapters["v1-p1-combray"]["prestige"]["unit_count"],
        baseline_swann_chapters["v1-p1-combray"]["inclusion"]["unit_count"],
    )
    supplemented_swann_unit_count = max(
        supplemented_swann_chapters["v1-p1-combray"]["advantage"]["unit_count"],
        supplemented_swann_chapters["v1-p1-combray"]["prestige"]["unit_count"],
        supplemented_swann_chapters["v1-p1-combray"]["inclusion"]["unit_count"],
    )
    narrator_unit_count = max(
        supplemented_narrator_chapters["v1-p1-combray"]["advantage"]["unit_count"],
        supplemented_narrator_chapters["v1-p1-combray"]["prestige"]["unit_count"],
        supplemented_narrator_chapters["v1-p1-combray"]["inclusion"]["unit_count"],
    )

    assert baseline_swann_unit_count == 2
    assert supplemented_swann_unit_count == 3
    assert narrator_unit_count == 2


def test_build_character_chapter_analysis_default_call_matches_explicit_none(tmp_path):
    accepted_run, _supplement_run = _make_pilot_supplement_scenario(tmp_path)

    default_call = pr.build_character_chapter_analysis([accepted_run], target_characters=["Swann"])
    explicit_none_call = pr.build_character_chapter_analysis(
        [accepted_run], target_characters=["Swann"], supplement_run_dirs=None
    )

    assert json.dumps(default_call, sort_keys=True) == json.dumps(explicit_none_call, sort_keys=True)
    assert "supplemented" not in default_call


def test_build_character_pages_with_supplements_marks_supplemented_and_gains_unit_count(tmp_path):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path)

    baseline = pr.build_character_pages([accepted_run], target_characters=["Swann"])
    supplemented = pr.build_character_pages(
        [accepted_run],
        target_characters=["Swann", "le narrateur"],
        supplement_run_dirs=[supplement_run],
    )

    assert "supplemented" not in baseline
    assert supplemented["supplemented"] is True
    assert supplemented["supplement_runs"] == ["supplement-run-001"]

    baseline_swann = next(page for page in baseline["pages"] if page["character"] == "Swann")
    supplemented_swann = next(page for page in supplemented["pages"] if page["character"] == "Swann")
    supplemented_narrator = next(page for page in supplemented["pages"] if page["character"] == "le narrateur")

    assert baseline_swann["profile"]["annotation_unit_count"] == 2
    assert supplemented_swann["profile"]["annotation_unit_count"] == 3
    assert supplemented_narrator["profile"]["annotation_unit_count"] == 2
    assert supplemented_narrator["slug"] == "le-narrateur"
    # notable_units draws from the supplement-merged overlay dataset, so a
    # brand-new supplement-only character still gets notable unit entries.
    assert len(supplemented_narrator["notable_units"]) > 0


def test_build_character_pages_default_call_matches_explicit_none(tmp_path):
    accepted_run, _supplement_run = _make_pilot_supplement_scenario(tmp_path)

    default_call = pr.build_character_pages([accepted_run], target_characters=["Swann"])
    explicit_none_call = pr.build_character_pages(
        [accepted_run], target_characters=["Swann"], supplement_run_dirs=None
    )

    assert json.dumps(default_call, sort_keys=True) == json.dumps(explicit_none_call, sort_keys=True)
    assert "supplemented" not in default_call


def test_main_character_profile_cards_supplemented_default_paths(tmp_path, capsys, monkeypatch):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path / "outputs")
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        [
            "character-profile-cards",
            "--discover-runs",
            str(tmp_path / "outputs"),
            "--include-supplements",
            "--supplement-outputs-dir",
            str(tmp_path / "outputs"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-profile-cards-supplemented-current.json"
    assert payload["markdown_output"] == "outputs/character-profile-cards-supplemented-current.md"
    assert (tmp_path / "outputs" / "character-profile-cards-supplemented-current.json").exists()

    # Without --include-supplements there is no default path at all -- the
    # analysis is printed to stdout, exactly as before this feature existed.
    exit_code_unsupplemented = pr.main(
        [
            "character-profile-cards",
            "--discover-runs",
            str(tmp_path / "outputs"),
        ]
    )
    payload_unsupplemented = json.loads(capsys.readouterr().out)
    assert exit_code_unsupplemented == 0
    assert "cards" in payload_unsupplemented
    assert not (tmp_path / "outputs" / "character-profile-cards-current.json").exists()


def test_main_character_chapter_analysis_supplemented_default_paths(tmp_path, capsys, monkeypatch):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path / "outputs")
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        [
            "character-chapter-analysis",
            "--discover-runs",
            str(tmp_path / "outputs"),
            "--character",
            "Swann",
            "--character",
            "le narrateur",
            "--include-supplements",
            "--supplement-outputs-dir",
            str(tmp_path / "outputs"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-chapter-cross-lens-supplemented-current.json"
    assert payload["markdown_output"] == "outputs/character-chapter-cross-lens-supplemented-current.md"
    assert (tmp_path / "outputs" / "character-chapter-cross-lens-supplemented-current.json").exists()


def test_main_character_pages_supplemented_default_paths(tmp_path, capsys, monkeypatch):
    accepted_run, supplement_run = _make_pilot_supplement_scenario(tmp_path / "outputs")
    monkeypatch.chdir(tmp_path)

    exit_code = pr.main(
        [
            "character-pages",
            "--discover-runs",
            str(tmp_path / "outputs"),
            "--character",
            "Swann",
            "--character",
            "le narrateur",
            "--include-supplements",
            "--supplement-outputs-dir",
            str(tmp_path / "outputs"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["json_output"] == "outputs/character-pages-supplemented-current.json"
    assert payload["markdown_output"] == "outputs/character-pages-supplemented-current.md"
    analysis = json.loads((tmp_path / "outputs" / "character-pages-supplemented-current.json").read_text())
    narrator_page = next(page for page in analysis["pages"] if page["character"] == "le narrateur")
    assert narrator_page["profile"]["annotation_unit_count"] == 2
