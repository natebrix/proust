import json
from pathlib import Path
from urllib import request as urllib_request

import proust as pn
import proust.runner as pr
import pytest


def _minimal_annotation(unit_id, character="Swann", dimension="social_status", delta=1):
    return {
        "unit_id": unit_id,
        "characters_present": [
            {
                "canonical_name": character,
                "surface_forms": [character],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [
            {
                "event_id": "E1",
                "source": "narrator",
                "target": character,
                "type": "admiration" if delta >= 0 else "narrated_diminishment",
                "polarity": "positive" if delta >= 0 else "negative",
                "narrative_stance": "endorsed",
                "confidence": 1.0,
                "evidence": "x",
                "explanation": "x",
            }
        ],
        "status_effects": [
            {
                "character": character,
                "dimension": dimension,
                "delta": delta,
                "based_on_events": ["E1"],
                "confidence": 1.0,
                "explanation": "x",
            }
        ],
        "ambiguities": [],
    }


def _multi_character_annotation(unit_id, characters):
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


def test_prepare_annotation_run_writes_expected_directory_shape(tmp_path):
    run_dir = tmp_path / "run-001"
    manifest = pn.prepare_annotation_run(run_dir, notes="starter batch")

    assert manifest.run_id == "run-001"
    assert (run_dir / "run.json").exists()
    assert (run_dir / "units").is_dir()
    assert (run_dir / "prompts").is_dir()
    assert (run_dir / "raw").is_dir()
    assert (run_dir / "annotations").is_dir()

    run_data = json.loads((run_dir / "run.json").read_text())
    assert run_data["run_id"] == "run-001"
    assert run_data["notes"] == "starter batch"
    assert run_data["unit_ids"] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]

    unit_path = run_dir / "units" / "v1-p1-combray#p-17.json"
    prompt_path = run_dir / "prompts" / "v1-p1-combray#p-17.txt"
    assert unit_path.exists()
    assert prompt_path.exists()

    unit = json.loads(unit_path.read_text())
    assert unit["reader_urls"]["fr-original"].endswith("/v1-p1-combray#p-17")
    assert "Swann" in unit["preprocessed_text"]

    prompt_text = prompt_path.read_text()
    assert "{{PASSAGE}}" not in prompt_text
    assert unit["preprocessed_text"] in prompt_text


def test_write_raw_response_and_annotation_result_persist_by_unit_id(tmp_path):
    run_dir = tmp_path / "run-002"
    pn.prepare_annotation_run(run_dir)

    raw_path = pn.write_raw_response(run_dir, "v1-p1-combray#p-17", '{"ok": true}')
    annotation_path = pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {"unit_id": "v1-p1-combray#p-17", "appraisal_events": [], "status_effects": [], "ambiguities": []},
    )

    assert raw_path.read_text() == '{"ok": true}'
    assert json.loads(annotation_path.read_text())["unit_id"] == "v1-p1-combray#p-17"


def test_validate_annotation_result_accepts_reviewed_benchmark_example():
    annotation_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "run-016"
        / "annotations"
        / "v2-p2-noms-de-pays-le-pays#p-211-p-213.json"
    )
    annotation = json.loads(annotation_path.read_text())

    assert pn.validate_annotation_result(annotation, expected_unit_id="v2-p2-noms-de-pays-le-pays#p-211-p-213") == []


def test_validate_annotation_result_rejects_unknown_event_reference():
    annotation = {
        "unit_id": "v1-p1-combray#p-17",
        "characters_present": [
            {
                "canonical_name": "Swann",
                "surface_forms": ["Swann"],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [
            {
                "event_id": "E1",
                "source": "narrator",
                "target": "Swann",
                "type": "admiration",
                "polarity": "positive",
                "narrative_stance": "endorsed",
                "confidence": 0.9,
                "evidence": "test evidence",
                "explanation": "test explanation",
            }
        ],
        "status_effects": [
            {
                "character": "Swann",
                "dimension": "social_status",
                "delta": 1,
                "based_on_events": ["E2"],
                "confidence": 0.9,
                "explanation": "test explanation",
            }
        ],
        "ambiguities": [],
    }

    errors = pn.validate_annotation_result(annotation, expected_unit_id="v1-p1-combray#p-17")

    assert errors == ["status_effects[0].based_on_events references unknown event ids: E2"]


def test_get_run_status_summarizes_benchmark_readiness(tmp_path):
    run_dir = tmp_path / "run-003"
    pn.prepare_annotation_run(run_dir)
    reviewed_annotation = {
        "unit_id": "v1-p1-combray#p-17",
        "characters_present": [
            {
                "canonical_name": "Swann",
                "surface_forms": ["Swann"],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [],
        "status_effects": [],
        "ambiguities": [],
    }
    for unit_id in [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]:
        pn.write_annotation_result(run_dir, unit_id, reviewed_annotation | {"unit_id": unit_id})

    status = pn.get_run_status(run_dir)

    assert status["summary"]["unit_count"] == 3
    assert status["summary"]["annotation_file_count"] == 3
    assert status["summary"]["valid_annotation_count"] == 3
    assert status["summary"]["reviewed_unit_count"] == 3
    assert status["summary"]["pending_unit_count"] == 0
    assert status["summary"]["benchmark_ready"] is True
    assert all(unit["review_state"] == "reviewed" for unit in status["units"])


def test_summarize_run_annotations_aggregates_events_and_status_effects(tmp_path):
    run_dir = tmp_path / "run-summary"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "narrator",
                    "target": "Swann",
                    "type": "admiration",
                    "polarity": "positive",
                    "narrative_stance": "endorsed",
                    "confidence": 0.9,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "social_status",
                    "delta": 2,
                    "based_on_events": ["E1"],
                    "confidence": 0.9,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )

    summary = pn.summarize_run_annotations(run_dir)

    assert summary["run_id"] == "run-summary"
    assert summary["unit_count"] == 3
    assert summary["valid_annotation_count"] == 1
    assert summary["event_type_counts"] == {"admiration": 1}
    assert summary["event_polarity_counts"]["positive"] == 1
    assert summary["event_source_counts"] == {"narrator": 1}
    assert summary["event_target_counts"] == {"Swann": 1}
    assert summary["status_dimension_totals"] == {"social_status": 2}
    assert summary["character_status_totals"] == {"Swann": {"social_status": 2}}


def test_score_run_advantage_outcomes_combines_event_and_status_signals(tmp_path):
    run_dir = tmp_path / "run-score"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "narrator",
                    "target": "Swann",
                    "type": "admiration",
                    "polarity": "positive",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "social_status",
                    "delta": 1,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        {
            "unit_id": "v1-p1-combray#p-274-p-275",
            "characters_present": [
                {
                    "canonical_name": "Legrandin",
                    "surface_forms": ["Legrandin"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "narrator",
                    "target": "Legrandin",
                    "type": "narrated_elevation",
                    "polarity": "positive",
                    "narrative_stance": "uncertain",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [],
            "ambiguities": ["unstable advantage uplift"],
        },
    )

    summary = pn.score_run_advantage_outcomes(run_dir)

    assert summary["run_id"] == "run-score"
    assert summary["scoring_version"] == "advantage_outcome_v1"
    assert summary["scored_unit_count"] == 2
    swann = next(unit for unit in summary["units"] if unit["unit_id"] == "v1-p1-combray#p-17")["characters"]["Swann"]
    legrandin = next(unit for unit in summary["units"] if unit["unit_id"] == "v1-p1-combray#p-274-p-275")["characters"][
        "Legrandin"
    ]
    assert swann["event_score"] == 0.9
    assert swann["status_score"] == 1.3
    assert swann["ambiguity_penalty"] == 0.0
    assert swann["net_score"] == 2.2
    assert swann["label"] == "win"
    assert legrandin["event_score"] == 0.5
    assert legrandin["status_score"] == 0.0
    assert legrandin["ambiguity_penalty"] == 0.4
    assert legrandin["net_score"] == 0.1
    assert legrandin["label"] == "neutral"


def test_main_score_reports_json_summary(tmp_path, capsys):
    run_dir = tmp_path / "run-score-cli"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [],
            "appraisal_events": [],
            "status_effects": [],
            "ambiguities": [],
        },
    )

    assert pr.main(["score", "--run", str(run_dir)]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["run_id"] == "run-score-cli"
    assert output["scoring_version"] == "advantage_outcome_v1"


def test_build_outcome_report_summarizes_characters_and_timeline(tmp_path):
    run_dir = tmp_path / "run-report"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "narrator",
                    "target": "Swann",
                    "type": "admiration",
                    "polarity": "positive",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "social_status",
                    "delta": 1,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        {
            "unit_id": "v1-p1-combray#p-274-p-275",
            "characters_present": [
                {
                    "canonical_name": "Legrandin",
                    "surface_forms": ["Legrandin"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "narrator",
                    "target": "Legrandin",
                    "type": "narrated_diminishment",
                    "polarity": "negative",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Legrandin",
                    "dimension": "general_appraisal",
                    "delta": -1,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )

    report = pn.build_outcome_report(run_dir)

    assert report["run_id"] == "run-report"
    assert report["report_version"] == "outcome_report_v1"
    assert report["scoring_version"] == "advantage_outcome_v1"
    assert report["character_count"] == 2
    assert [entry["character"] for entry in report["character_summaries"]] == ["Swann", "Legrandin"]
    assert report["character_summaries"][0]["top_win"]["unit_id"] == "v1-p1-combray#p-17"
    assert report["character_summaries"][1]["top_loss"]["unit_id"] == "v1-p1-combray#p-274-p-275"
    assert [entry["unit_id"] for entry in report["timeline"]] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
    ]
    assert report["top_wins"][0]["character"] == "Swann"
    assert report["top_losses"][0]["character"] == "Legrandin"
    assert report["mixed_units"] == []


def test_discover_annotation_run_dirs_finds_valid_annotated_runs(tmp_path):
    outputs_dir = tmp_path / "outputs"
    annotated_run = outputs_dir / "run-002"
    empty_run = outputs_dir / "run-001"
    pn.prepare_annotation_run(empty_run)
    pn.prepare_annotation_run(annotated_run)
    pn.write_annotation_result(
        annotated_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17"),
    )

    discovered = pr.discover_annotation_run_dirs(outputs_dir)

    assert discovered == [annotated_run]


def test_render_corpus_review_markdown_includes_headline_sections(tmp_path):
    run_dir = tmp_path / "run-review"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17"),
    )
    review = pr.build_corpus_sanity_review([run_dir])

    markdown = pr.render_corpus_review_markdown(review)

    assert "# Corpus Review" in markdown
    assert "Valid annotation count: `1`" in markdown
    assert "## Lens Reviews" in markdown
    assert "## Cross-Lens Summary" in markdown
    assert "| social_status | +1 |" in markdown


def test_build_outcome_report_can_apply_reviewed_character_normalization(tmp_path):
    run_dir = tmp_path / "run-normalized-report"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    report = pr.build_outcome_report(
        run_dir,
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )

    assert report["character_normalization"]["applied"] is True
    assert report["character_count"] == 1
    assert [entry["character"] for entry in report["character_summaries"]] == ["baron de Charlus"]
    assert [entry["character"] for entry in report["timeline"]] == [
        "baron de Charlus",
        "baron de Charlus",
    ]
    assert report["character_summaries"][0]["unit_count"] == 2


def test_build_corpus_sanity_review_can_apply_character_normalization_and_diff(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="baron de Charlus"),
    )

    before_review = pr.build_corpus_sanity_review([first_run, second_run])
    after_review = pr.build_corpus_sanity_review(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )
    diff = pr.build_corpus_review_normalization_diff(before_review, after_review)

    assert before_review["lens_reviews"]["advantage"]["character_count"] == 2
    assert after_review["character_normalization"]["applied"] is True
    assert after_review["lens_reviews"]["advantage"]["character_count"] == 1
    assert after_review["lens_reviews"]["advantage"]["top_positive_characters"][0]["character"] == "baron de Charlus"
    assert diff["lens_diffs"]["advantage"]["character_count_before"] == 2
    assert diff["lens_diffs"]["advantage"]["character_count_after"] == 1
    assert diff["lens_diffs"]["advantage"]["normalized_characters"][0]["character"] == "baron de Charlus"
    assert diff["lens_diffs"]["advantage"]["normalized_characters"][0]["merged_from"] == ["Charlus"]


def test_main_corpus_review_can_discover_and_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "run-001"
    json_output = tmp_path / "corpus-review.json"
    markdown_output = tmp_path / "corpus-review.md"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17"),
    )

    exit_code = pr.main(
        [
            "corpus-review",
            "--discover-runs",
            str(outputs_dir),
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    review = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["run_count"] == 1
    assert payload["valid_annotation_count"] == 1
    assert review["run_count"] == 1
    assert markdown_output.read_text().startswith("# Corpus Review\n")


def test_main_corpus_review_can_write_normalized_artifacts_and_diff(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    json_output = tmp_path / "corpus-review-normalized.json"
    markdown_output = tmp_path / "corpus-review-normalized.md"
    diff_output = tmp_path / "corpus-review-normalization-diff.md"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "corpus-review",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--normalization-diff-output",
            str(diff_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    review = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["normalization_diff_output"] == str(diff_output)
    assert review["character_normalization"]["applied"] is True
    assert review["lens_reviews"]["advantage"]["character_count"] == 1
    assert markdown_output.read_text().startswith("# Corpus Review\n")
    assert diff_output.read_text().startswith("# Corpus Review Normalization Diff\n")


def test_build_character_cross_lens_analysis_uses_normalized_review(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="baron de Charlus"),
    )

    review = pr.build_corpus_sanity_review(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )
    analysis = pr.build_character_cross_lens_analysis(review)

    assert analysis["character_normalization"]["applied"] is True
    assert analysis["character_count"] == 1
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert analysis["characters"][0]["lens_scores"]["advantage"]["unit_count"] == 2
    assert analysis["characters"][0]["lens_scores"]["advantage"]["percentile"] == 100
    assert "baron de Charlus" in pr.render_character_cross_lens_analysis_markdown(analysis)


def test_main_character_analysis_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    json_output = tmp_path / "character-analysis.json"
    markdown_output = tmp_path / "character-analysis.md"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "character-analysis",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    analysis = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["character_count"] == 1
    assert analysis["character_normalization"]["applied"] is True
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert markdown_output.read_text().startswith("# Character Cross-Lens Analysis\n")


def test_build_character_chapter_analysis_groups_scores_by_chapter(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    analysis = pr.build_character_chapter_analysis(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
        target_characters=["baron de Charlus"],
    )

    assert analysis["selected_character_count"] == 1
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert analysis["characters"][0]["chapters"][0]["chapter_id"] == "v1-p1-combray"
    assert analysis["characters"][0]["chapters"][0]["advantage"]["unit_count"] == 2
    assert "v1-p1-combray" in pr.render_character_chapter_analysis_markdown(analysis)


def test_main_character_chapter_analysis_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    json_output = tmp_path / "character-chapter-analysis.json"
    markdown_output = tmp_path / "character-chapter-analysis.md"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "character-chapter-analysis",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--character",
            "baron de Charlus",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    analysis = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["selected_character_count"] == 1
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert markdown_output.read_text().startswith("# Character Chapter Analysis\n")


def test_build_character_annotation_counts_sorts_by_unit_count(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-312-p-313",
        _minimal_annotation("v1-p1-combray#p-312-p-313", character="Swann"),
    )

    review = pr.build_corpus_sanity_review(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )
    analysis = pr.build_character_annotation_counts(review)

    assert analysis["character_count"] == 2
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert analysis["characters"][0]["annotation_unit_count"] == 2
    assert analysis["characters"][1]["character"] == "Swann"
    assert "Annotation Units" in pr.render_character_annotation_counts_markdown(analysis)


def test_main_character_annotation_counts_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    json_output = tmp_path / "character-annotation-counts.json"
    markdown_output = tmp_path / "character-annotation-counts.md"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "character-annotation-counts",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    analysis = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["character_count"] == 1
    assert analysis["characters"][0]["character"] == "baron de Charlus"
    assert markdown_output.read_text().startswith("# Character Annotation Counts\n")


def test_build_character_profile_cards_uses_existing_derived_fields(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-312-p-313",
        _minimal_annotation("v1-p1-combray#p-312-p-313", character="Swann"),
    )

    analysis = pr.build_character_profile_cards(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )

    assert analysis["character_count"] == 2
    assert analysis["cards"][0]["character"] == "baron de Charlus"
    assert analysis["cards"][0]["annotation_unit_count"] == 2
    assert analysis["cards"][0]["lens_scores"]["advantage"]["percentile"] == 100
    assert analysis["cards"][0]["top_chapters"][0]["chapter_id"] == "v1-p1-combray"
    assert "Top chapters" in pr.render_character_profile_cards_markdown(analysis)


def test_main_character_profile_cards_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    json_output = tmp_path / "character-profile-cards.json"
    markdown_output = tmp_path / "character-profile-cards.md"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "character-profile-cards",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    analysis = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["character_count"] == 1
    assert analysis["cards"][0]["character"] == "baron de Charlus"
    assert markdown_output.read_text().startswith("# Character Profile Cards\n")


def test_build_chapter_overlay_data_exports_normalized_chapter_units(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-312-p-313",
        _minimal_annotation("v1-p1-combray#p-312-p-313", character="Swann", delta=-1),
    )

    dataset = pr.build_chapter_overlay_data(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )

    assert dataset["chapter_overlay_version"] == "chapter_overlay_v2"
    assert dataset["chapter_count"] == len(pr.CANONICAL_CHAPTER_SPECS)
    assert dataset["character_normalization"]["applied"] is True

    manifest_row = next(row for row in dataset["manifest"]["chapters"] if row["chapterId"] == "v1-p1-combray")
    assert manifest_row["unitCount"] == 3
    assert manifest_row["characterCount"] == 2

    chapter = next(row for row in dataset["chapters"] if row["chapterId"] == "v1-p1-combray")
    assert chapter["title"] == "Du Côté de Chez Swann — I. Combray"
    assert chapter["summary"].startswith("This chapter contains 3 annotated units")
    assert [unit["unitId"] for unit in chapter["units"]] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]
    assert chapter["units"][0]["paragraphStart"] == 17
    assert chapter["units"][1]["paragraphEnd"] == 275
    assert chapter["units"][0]["dominantCharacter"] == "baron de Charlus"
    assert chapter["units"][0]["characters"][0]["character"] == "baron de Charlus"
    assert chapter["units"][0]["characters"][0]["advantage"]["label"] == "win"
    assert "baron de Charlus gains social status" in chapter["units"][0]["summary"]
    assert chapter["units"][2]["characters"][0]["character"] == "Swann"
    assert chapter["units"][2]["characters"][0]["advantage"]["label"] == "loss"
    assert "Swann loses social status" in chapter["units"][2]["summary"]


def test_build_chapter_overlay_data_prefers_latest_reviewed_run_for_duplicate_unit_ids(tmp_path):
    first_run = tmp_path / "run-001"
    second_run = tmp_path / "run-002"
    pn.prepare_annotation_run(first_run)
    pn.prepare_annotation_run(second_run)
    pn.write_annotation_result(
        first_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Swann"),
    )
    pn.write_annotation_result(
        second_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )

    dataset = pr.build_chapter_overlay_data(
        [first_run, second_run],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
    )

    chapter = next(row for row in dataset["chapters"] if row["chapterId"] == "v1-p1-combray")
    assert len(chapter["units"]) == 1
    assert chapter["units"][0]["characters"][0]["character"] == "baron de Charlus"
    assert "baron de Charlus" in chapter["units"][0]["summary"]
    assert dataset["manifest"]["duplicate_resolution"] == "latest_reviewed_run_wins"


def test_main_chapter_overlays_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    run_b = outputs_dir / "run-002"
    output_dir = tmp_path / "chapter-overlays"
    pn.prepare_annotation_run(run_a)
    pn.prepare_annotation_run(run_b)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        run_b,
        "v1-p1-combray#p-274-p-275",
        _minimal_annotation("v1-p1-combray#p-274-p-275", character="baron de Charlus"),
    )

    exit_code = pr.main(
        [
            "chapter-overlays",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    chapter = json.loads((output_dir / "chapters" / "v1-p1-combray.json").read_text())
    assert exit_code == 0
    assert payload["chapter_count"] == len(pr.CANONICAL_CHAPTER_SPECS)
    assert payload["character_normalization_applied"] is True
    assert manifest["chapter_overlay_version"] == "chapter_overlay_v2"
    assert chapter["chapterId"] == "v1-p1-combray"
    assert "summary" in chapter
    assert "summary" in chapter["units"][0]
    assert chapter["units"][0]["characters"][0]["character"] == "baron de Charlus"


def test_build_character_pages_pilot_uses_profile_portraits_and_editorial(tmp_path):
    run_dir = tmp_path / "run-001"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        _multi_character_annotation(
            "v1-p1-combray#p-17",
            [
                {"character": "Odette", "delta": -1},
                {"character": "Saint-Loup", "delta": 1},
            ],
        ),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        _multi_character_annotation(
            "v1-p1-combray#p-274-p-275",
            [
                {"character": "Swann", "delta": -1},
                {"character": "Albertine", "delta": -1},
            ],
        ),
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-312-p-313",
        _minimal_annotation("v1-p1-combray#p-312-p-313", character="Charlus", delta=-1),
    )

    analysis = pr.build_character_pages(
        [run_dir],
        character_name_map=pr.REVIEWED_CHARACTER_NORMALIZATION_MAP,
        target_characters=["Odette", "Robert de Saint-Loup", "Swann", "Albertine", "baron de Charlus"],
    )

    assert analysis["character_pages_version"] == "character_pages_v1"
    assert analysis["character_count"] == 5
    odette = next(page for page in analysis["pages"] if page["character"] == "Odette")
    assert odette["portrait"]["default"]
    assert odette["editorial"]["primary_pattern"] == "prestige_positive_inclusion_negative"
    assert odette["profile"]["lens_scores"]["prestige"]["percentile"] is not None
    assert odette["top_chapters"][0]["reader_link"].startswith("/projects/islt/fr-original/")
    assert odette["reading_path"][0]["chapter_id"] == "v2-p1-autour-de-mme-swann"
    saint_loup = next(page for page in analysis["pages"] if page["character"] == "Robert de Saint-Loup")
    assert saint_loup["portrait"]["default"]
    assert saint_loup["slug"] == "robert-de-saint-loup"
    assert "core prestige/inclusion split" in saint_loup["reading_path"][0]["label"].lower()
    assert "Why interesting" in pr.render_character_pages_markdown(analysis)


def test_main_character_pages_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_a = outputs_dir / "run-001"
    json_output = tmp_path / "character-pages.json"
    markdown_output = tmp_path / "character-pages.md"
    pn.prepare_annotation_run(run_a)
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-17",
        _multi_character_annotation(
            "v1-p1-combray#p-17",
            [
                {"character": "Odette", "delta": -1},
                {"character": "Saint-Loup", "delta": 1},
            ],
        ),
    )
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-274-p-275",
        _multi_character_annotation(
            "v1-p1-combray#p-274-p-275",
            [
                {"character": "Swann", "delta": -1},
                {"character": "Albertine", "delta": -1},
            ],
        ),
    )
    pn.write_annotation_result(
        run_a,
        "v1-p1-combray#p-312-p-313",
        _minimal_annotation("v1-p1-combray#p-312-p-313", character="Charlus", delta=-1),
    )

    exit_code = pr.main(
        [
            "character-pages",
            "--discover-runs",
            str(outputs_dir),
            "--reviewed-character-normalization",
            "--character",
            "Odette",
            "--character",
            "Robert de Saint-Loup",
            "--character",
            "Swann",
            "--character",
            "Albertine",
            "--character",
            "baron de Charlus",
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    analysis = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["character_count"] == 5
    assert analysis["pages"][0]["portrait"]["default"]
    assert markdown_output.read_text().startswith("# Character Pages\n")


def test_build_character_alias_audit_reports_candidate_merge_groups(tmp_path):
    outputs_dir = tmp_path / "outputs"
    aliases_csv = tmp_path / "aliases.csv"
    aliases_csv.write_text("Charlus,baron de Charlus\n")
    charlus_run = outputs_dir / "run-001"
    baron_run = outputs_dir / "run-002"
    pn.prepare_annotation_run(
        charlus_run,
        alias_map={"Charlus": {"aliases": ["Charlus", "M. de Charlus"]}},
    )
    pn.prepare_annotation_run(
        baron_run,
        alias_map={"baron de Charlus": {"aliases": ["baron de Charlus", "Charlus"]}},
    )
    pn.write_annotation_result(
        charlus_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Charlus"),
    )
    pn.write_annotation_result(
        baron_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="baron de Charlus"),
    )

    audit = pr.build_character_alias_audit(outputs_dir=outputs_dir, aliases_csv=aliases_csv)

    assert audit["annotation_name_count"] == 2
    assert audit["candidate_merge_group_count"] == 1
    group = audit["candidate_merge_groups"][0]
    assert group["preferred_name_by_usage"] == "baron de Charlus"
    assert {name["name"] for name in group["names"]} == {"Charlus", "baron de Charlus"}
    assert any("aliases_csv" in edge["sources"] for edge in group["alias_edges"])


def test_main_character_alias_audit_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    aliases_csv = tmp_path / "aliases.csv"
    json_output = tmp_path / "character-alias-audit.json"
    markdown_output = tmp_path / "character-alias-audit.md"
    aliases_csv.write_text("Mme Swann,Odette\n")
    mme_run = outputs_dir / "run-001"
    odette_run = outputs_dir / "run-002"
    pn.prepare_annotation_run(mme_run)
    pn.prepare_annotation_run(odette_run)
    pn.write_annotation_result(
        mme_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Mme Swann"),
    )
    pn.write_annotation_result(
        odette_run,
        "v1-p1-combray#p-17",
        _minimal_annotation("v1-p1-combray#p-17", character="Odette"),
    )

    exit_code = pr.main(
        [
            "character-alias-audit",
            "--outputs-dir",
            str(outputs_dir),
            "--aliases-csv",
            str(aliases_csv),
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    audit = json.loads(json_output.read_text())
    assert exit_code == 0
    assert payload["candidate_merge_group_count"] == 1
    assert audit["candidate_merge_groups"][0]["preferred_name_by_usage"] == "Odette"
    assert markdown_output.read_text().startswith("# Character Alias Audit\n")


def test_main_report_outputs_json(tmp_path, capsys):
    run_dir = tmp_path / "run-report-cli"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [],
            "appraisal_events": [],
            "status_effects": [],
            "ambiguities": [],
        },
    )

    assert pr.main(["report", "--run", str(run_dir)]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["run_id"] == "run-report-cli"
    assert output["report_version"] == "outcome_report_v1"


def test_score_run_prestige_outcomes_reweights_prestige_and_inclusion(tmp_path):
    run_dir = tmp_path / "run-prestige"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "collective_social_voice",
                    "target": "Swann",
                    "type": "prestige_association",
                    "polarity": "positive",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "social_status",
                    "delta": 1,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        {
            "unit_id": "v1-p1-combray#p-274-p-275",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "collective_social_voice",
                    "target": "Swann",
                    "type": "snub",
                    "polarity": "negative",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "inclusion_exclusion",
                    "delta": -2,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )

    local_summary = pn.score_run_advantage_outcomes(run_dir)
    prestige_summary = pn.score_run_prestige_outcomes(run_dir)

    local_positive = next(unit for unit in local_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-17")["characters"][
        "Swann"
    ]
    prestige_positive = next(
        unit for unit in prestige_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-17"
    )["characters"]["Swann"]
    local_negative = next(
        unit for unit in local_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-274-p-275"
    )["characters"]["Swann"]
    prestige_negative = next(
        unit for unit in prestige_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-274-p-275"
    )["characters"]["Swann"]

    assert prestige_summary["scoring_version"] == "prestige_outcome_v1"
    assert prestige_positive["net_score"] > local_positive["net_score"]
    assert prestige_negative["net_score"] > local_negative["net_score"]


def test_main_report_accepts_prestige_lens(tmp_path, capsys):
    run_dir = tmp_path / "run-report-prestige-cli"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [],
            "appraisal_events": [],
            "status_effects": [],
            "ambiguities": [],
        },
    )

    assert pr.main(["report", "--run", str(run_dir), "--lens", "prestige"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["report_version"] == "outcome_report_v1"
    assert output["scoring_version"] == "prestige_outcome_v1"
    assert output["lens"] == "prestige"


def test_score_run_inclusion_outcomes_reweights_exclusion_and_prestige(tmp_path):
    run_dir = tmp_path / "run-inclusion"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "collective_social_voice",
                    "target": "Swann",
                    "type": "prestige_association",
                    "polarity": "positive",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "social_status",
                    "delta": 1,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-274-p-275",
        {
            "unit_id": "v1-p1-combray#p-274-p-275",
            "characters_present": [
                {
                    "canonical_name": "Swann",
                    "surface_forms": ["Swann"],
                    "presence_type": "explicit",
                    "presence_confidence": 0.99,
                }
            ],
            "appraisal_events": [
                {
                    "event_id": "E1",
                    "source": "collective_social_voice",
                    "target": "Swann",
                    "type": "snub",
                    "polarity": "negative",
                    "narrative_stance": "endorsed",
                    "confidence": 1.0,
                    "evidence": "x",
                    "explanation": "x",
                }
            ],
            "status_effects": [
                {
                    "character": "Swann",
                    "dimension": "inclusion_exclusion",
                    "delta": -2,
                    "based_on_events": ["E1"],
                    "confidence": 1.0,
                    "explanation": "x",
                }
            ],
            "ambiguities": [],
        },
    )

    local_summary = pn.score_run_advantage_outcomes(run_dir)
    inclusion_summary = pn.score_run_inclusion_outcomes(run_dir)

    local_positive = next(unit for unit in local_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-17")["characters"][
        "Swann"
    ]
    inclusion_positive = next(
        unit for unit in inclusion_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-17"
    )["characters"]["Swann"]
    local_negative = next(
        unit for unit in local_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-274-p-275"
    )["characters"]["Swann"]
    inclusion_negative = next(
        unit for unit in inclusion_summary["units"] if unit["unit_id"] == "v1-p1-combray#p-274-p-275"
    )["characters"]["Swann"]

    assert inclusion_summary["scoring_version"] == "inclusion_outcome_v1"
    assert inclusion_positive["net_score"] < local_positive["net_score"]
    assert inclusion_negative["net_score"] < local_negative["net_score"]


def test_main_report_accepts_inclusion_lens(tmp_path, capsys):
    run_dir = tmp_path / "run-report-inclusion-cli"
    pn.prepare_annotation_run(run_dir)
    pn.write_annotation_result(
        run_dir,
        "v1-p1-combray#p-17",
        {
            "unit_id": "v1-p1-combray#p-17",
            "characters_present": [],
            "appraisal_events": [],
            "status_effects": [],
            "ambiguities": [],
        },
    )

    assert pr.main(["report", "--run", str(run_dir), "--lens", "inclusion"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["report_version"] == "outcome_report_v1"
    assert output["scoring_version"] == "inclusion_outcome_v1"
    assert output["lens"] == "inclusion"


def test_mark_run_as_benchmark_writes_validation_metadata(tmp_path):
    run_dir = tmp_path / "run-004"
    pn.prepare_annotation_run(run_dir)
    for unit_id in [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]:
        pn.write_annotation_result(
            run_dir,
            unit_id,
            {
                "unit_id": unit_id,
                "characters_present": [],
                "appraisal_events": [],
                "status_effects": [],
                "ambiguities": [],
            },
        )

    benchmark = pn.mark_run_as_benchmark(run_dir, label="starter gold set")
    run_data = json.loads((run_dir / "run.json").read_text())

    assert benchmark["label"] == "starter gold set"
    assert benchmark["status"] == "reviewed"
    assert benchmark["benchmark_ready"] is True
    assert run_data["benchmark"]["reviewed_unit_ids"] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]


def test_compare_run_to_benchmark_reports_exact_matches_and_differences(tmp_path):
    benchmark_dir = tmp_path / "run-benchmark"
    candidate_dir = tmp_path / "run-candidate"
    pn.prepare_annotation_run(benchmark_dir)
    pn.prepare_annotation_run(candidate_dir)

    shared_exact = {
        "unit_id": "v1-p1-combray#p-17",
        "characters_present": [
            {
                "canonical_name": "Swann",
                "surface_forms": ["Swann"],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [],
        "status_effects": [],
        "ambiguities": [],
    }
    benchmark_only_shape = {
        "unit_id": "v1-p1-combray#p-274-p-275",
        "characters_present": [
            {
                "canonical_name": "Legrandin",
                "surface_forms": ["Legrandin"],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [],
        "status_effects": [],
        "ambiguities": [],
    }
    candidate_diff = {
        "unit_id": "v1-p1-combray#p-274-p-275",
        "characters_present": [
            {
                "canonical_name": "Legrandin",
                "surface_forms": ["Legrandin"],
                "presence_type": "explicit",
                "presence_confidence": 0.99,
            }
        ],
        "appraisal_events": [
            {
                "event_id": "E1",
                "source": "narrator",
                "target": "Legrandin",
                "type": "blame",
                "polarity": "negative",
                "narrative_stance": "endorsed",
                "confidence": 0.8,
                "evidence": "test evidence",
                "explanation": "test explanation",
            }
        ],
        "status_effects": [
            {
                "character": "Legrandin",
                "dimension": "general_appraisal",
                "delta": -1,
                "based_on_events": ["E1"],
                "confidence": 0.8,
                "explanation": "test explanation",
            }
        ],
        "ambiguities": [],
    }

    pn.write_annotation_result(benchmark_dir, "v1-p1-combray#p-17", shared_exact)
    pn.write_annotation_result(candidate_dir, "v1-p1-combray#p-17", shared_exact)
    pn.write_annotation_result(benchmark_dir, "v1-p1-combray#p-274-p-275", benchmark_only_shape)
    pn.write_annotation_result(candidate_dir, "v1-p1-combray#p-274-p-275", candidate_diff)

    comparison = pn.compare_run_to_benchmark(candidate_dir, benchmark_dir)

    assert comparison["summary"]["shared_unit_count"] == 3
    assert comparison["summary"]["exact_match_count"] == 1
    assert comparison["summary"]["differing_annotation_count"] == 1
    assert comparison["summary"]["missing_annotation_count"] == 1
    assert comparison["summary"]["all_shared_annotations_match"] is False
    assert comparison["shared_unit_ids"] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]
    assert comparison["units"] == [
        {
            "unit_id": "v1-p1-combray#p-17",
            "run_annotation_exists": True,
            "benchmark_annotation_exists": True,
            "annotation_exact_match": True,
        },
        {
            "unit_id": "v1-p1-combray#p-274-p-275",
            "run_annotation_exists": True,
            "benchmark_annotation_exists": True,
            "annotation_exact_match": False,
        },
        {
            "unit_id": "v1-p1-combray#p-312-p-313",
            "run_annotation_exists": False,
            "benchmark_annotation_exists": False,
            "annotation_exact_match": False,
        },
    ]


def test_get_run_status_raises_clear_error_for_missing_run_manifest(tmp_path):
    missing_run_dir = tmp_path / "missing-run"

    with pytest.raises(pr.RunManifestNotFoundError) as exc_info:
        pn.get_run_status(missing_run_dir)

    assert str(exc_info.value) == (
        f'Run directory "{missing_run_dir}" does not contain a run.json manifest at '
        f'"{missing_run_dir / "run.json"}".'
    )


def test_wait_for_automation_completion_returns_immediately_for_finished_run(tmp_path):
    run_dir = tmp_path / "run-finished"
    pn.prepare_annotation_run(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["automation"] = {
        "requested_unit_count": 3,
        "completed_unit_count": 3,
        "successful_annotation_count": 3,
        "parse_error_count": 0,
        "validation_error_count": 0,
        "in_progress": False,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    waited = pr.wait_for_automation_completion(run_dir, poll_interval=0.01)

    assert waited["requested_unit_count"] == 3
    assert waited["completed_unit_count"] == 3
    assert waited["successful_annotation_count"] == 3
    assert waited["in_progress"] is False


def test_wait_for_automation_completion_times_out_for_stuck_run(tmp_path):
    run_dir = tmp_path / "run-stuck"
    pn.prepare_annotation_run(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["automation"] = {
        "requested_unit_count": 3,
        "completed_unit_count": 1,
        "successful_annotation_count": 1,
        "parse_error_count": 0,
        "validation_error_count": 0,
        "in_progress": True,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    with pytest.raises(TimeoutError):
        pr.wait_for_automation_completion(run_dir, poll_interval=0.01, timeout=0.02)


def test_main_wait_can_reduce_and_report_finished_run(tmp_path, capsys):
    run_dir = tmp_path / "run-wait"
    pn.prepare_annotation_run(run_dir)
    for unit_id in json.loads((run_dir / "run.json").read_text())["unit_ids"]:
        pn.write_raw_response(
            run_dir,
            unit_id,
            json.dumps(
                {
                    "unit_id": unit_id,
                    "characters_present": [],
                    "appraisal_events": [],
                    "status_effects": [],
                    "ambiguities": [],
                }
            ),
        )
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["automation"] = {
        "requested_unit_count": 3,
        "completed_unit_count": 3,
        "successful_annotation_count": 3,
        "parse_error_count": 0,
        "validation_error_count": 0,
        "in_progress": False,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    exit_code = pr.main(["wait", "--run", str(run_dir), "--quiet", "--reduce", "--report"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["wait"]["completed_unit_count"] == 3
    assert payload["reprocess"]["run"] == str(run_dir)
    assert sorted(payload["reports"]) == ["advantage", "inclusion", "prestige"]


def test_main_compare_reports_missing_run_manifest_without_traceback(tmp_path, capsys):
    benchmark_dir = tmp_path / "run-benchmark"
    pn.prepare_annotation_run(benchmark_dir)

    with pytest.raises(SystemExit) as exc_info:
        pr.main(
            [
                "compare",
                "--run",
                str(tmp_path / "run-candidate"),
                "--benchmark",
                str(benchmark_dir),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Run directory" in captured.err
    assert "run-candidate/run.json" in captured.err
    assert "Traceback" not in captured.err


def test_prepare_annotation_run_from_existing_copies_units_and_prompts(tmp_path):
    source_run = tmp_path / "run-source"
    target_run = tmp_path / "run-target"
    pn.prepare_annotation_run(source_run, notes="source batch")

    manifest = pn.prepare_annotation_run_from_existing(source_run, target_run)
    target_data = json.loads((target_run / "run.json").read_text())

    assert manifest.run_id == "run-target"
    assert target_data["derived_from"]["run_id"] == "run-source"
    assert target_data["unit_ids"] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]
    assert (target_run / "units" / "v1-p1-combray#p-17.json").exists()
    assert (target_run / "prompts" / "v1-p1-combray#p-17.txt").exists()
    assert list((target_run / "annotations").glob("*.json")) == []
    assert list((target_run / "raw").glob("*.txt")) == []


def test_parse_annotation_response_text_accepts_fenced_json():
    raw_text = """```json
    {"unit_id":"v1-p1-combray#p-17","characters_present":[],"appraisal_events":[],"status_effects":[],"ambiguities":[]}
    ```"""

    parsed = pn.parse_annotation_response_text(raw_text)

    assert parsed["unit_id"] == "v1-p1-combray#p-17"


def test_run_annotation_requests_records_raw_and_valid_annotations(tmp_path):
    run_dir = tmp_path / "run-auto"
    pn.prepare_annotation_run(run_dir)

    def fake_requester(prompt_text, unit_payload, model):
        assert unit_payload["unit_id"].startswith("v1-p1-combray#p-")
        assert model == "gpt-test"
        assert unit_payload["preprocessed_text"] in prompt_text
        return json.dumps(
            {
                "unit_id": unit_payload["unit_id"],
                "characters_present": [],
                "appraisal_events": [],
                "status_effects": [],
                "ambiguities": [],
            }
        )

    automation = pr.run_annotation_requests(run_dir, fake_requester, model="gpt-test", limit=2)
    run_data = json.loads((run_dir / "run.json").read_text())

    assert automation["provider"] == "openai"
    assert automation["model"] == "gpt-test"
    assert automation["requested_unit_count"] == 2
    assert automation["successful_annotation_count"] == 2
    assert automation["parse_error_count"] == 0
    assert automation["validation_error_count"] == 0
    assert len(automation["results"]) == 2
    assert run_data["automation"]["requested_unit_count"] == 2
    assert len(list((run_dir / "raw").glob("*.txt"))) == 2
    assert len(list((run_dir / "annotations").glob("*.json"))) == 2


def test_openai_responses_request_retries_remote_disconnect(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"unit_id":"v1-p1-combray#p-17","characters_present":[],"appraisal_events":[],"status_effects":[],"ambiguities":[]}',
                                }
                            ],
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        del request, timeout
        calls["count"] += 1
        if calls["count"] == 1:
            raise pr.RemoteDisconnected("Remote end closed connection without response")
        return FakeResponse()

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)

    raw_text = pr._openai_responses_request("prompt", {}, "gpt-5", api_key="test-key", max_attempts=2)

    assert calls["count"] == 2
    assert '"unit_id":"v1-p1-combray#p-17"' in raw_text


def test_parse_annotation_response_text_injects_unit_id_and_normalizes_plus_delta():
    raw_text = """{
      "characters_present": [],
      "appraisal_events": [],
      "status_effects": [{"character":"Swann","dimension":"social_status","delta": +2,"based_on_events":["E1"],"confidence":0.9,"explanation":"x"}],
      "ambiguities": []
    }"""

    parsed = pn.parse_annotation_response_text(raw_text, expected_unit_id="v1-p1-combray#p-17")

    assert parsed["unit_id"] == "v1-p1-combray#p-17"
    assert parsed["status_effects"][0]["delta"] == 2


def test_reprocess_raw_annotations_salvages_schema_drift_from_saved_raw(tmp_path):
    run_dir = tmp_path / "run-reprocess"
    pn.prepare_annotation_run(run_dir)
    pn.write_raw_response(
        run_dir,
        "v1-p1-combray#p-17",
        """{
          "characters_present": [],
          "appraisal_events": [],
          "status_effects": [],
          "ambiguities": []
        }""",
    )

    results = pn.reprocess_raw_annotations(run_dir)

    assert results == [
        {
            "unit_id": "v1-p1-combray#p-17",
            "annotation_written": True,
            "parse_error": None,
            "validation_errors": [],
        }
    ]
    assert json.loads((run_dir / "annotations" / "v1-p1-combray#p-17.json").read_text())["unit_id"] == (
        "v1-p1-combray#p-17"
    )


def test_reduce_annotation_result_compresses_events_and_status_effects():
    raw_annotation = {
        "unit_id": "v1-p1-combray#p-310-p-311",
        "characters_present": [
            {
                "canonical_name": "M. Vinteuil",
                "surface_forms": ["M. Vinteuil", "Vinteuil"],
                "presence_type": "explicit",
                "presence_confidence": 0.98,
            },
            {
                "canonical_name": "Swann",
                "surface_forms": ["Swann"],
                "presence_type": "explicit",
                "presence_confidence": 0.98,
            },
        ],
        "appraisal_events": [
            {
                "event_id": "E1",
                "source": "collective_social_voice",
                "target": "M. Vinteuil",
                "type": "ridicule",
                "polarity": "negative",
                "narrative_stance": "neutral_report",
                "confidence": 0.8,
                "evidence": "x",
                "explanation": "x",
            },
            {
                "event_id": "E2",
                "source": "narrator",
                "target": "M. Vinteuil",
                "type": "narrated_diminishment",
                "polarity": "negative",
                "narrative_stance": "endorsed",
                "confidence": 0.84,
                "evidence": "x",
                "explanation": "x",
            },
            {
                "event_id": "E3",
                "source": "Swann",
                "target": "M. Vinteuil",
                "type": "inclusion",
                "polarity": "positive",
                "narrative_stance": "neutral_report",
                "confidence": 0.82,
                "evidence": "x",
                "explanation": "x",
            },
            {
                "event_id": "E4",
                "source": "narrator",
                "target": "Swann",
                "type": "blame",
                "polarity": "negative",
                "narrative_stance": "endorsed",
                "confidence": 0.7,
                "evidence": "x",
                "explanation": "x",
            },
        ],
        "status_effects": [
            {
                "character": "M. Vinteuil",
                "dimension": "general_appraisal",
                "delta": -2,
                "based_on_events": ["E1"],
                "confidence": 0.85,
                "explanation": "x",
            },
            {
                "character": "M. Vinteuil",
                "dimension": "social_status",
                "delta": -2,
                "based_on_events": ["E2"],
                "confidence": 0.84,
                "explanation": "x",
            },
            {
                "character": "M. Vinteuil",
                "dimension": "emotional_position",
                "delta": -2,
                "based_on_events": ["E2"],
                "confidence": 0.83,
                "explanation": "x",
            },
            {
                "character": "Swann",
                "dimension": "social_status",
                "delta": 1,
                "based_on_events": ["E3"],
                "confidence": 0.73,
                "explanation": "x",
            },
            {
                "character": "Swann",
                "dimension": "general_appraisal",
                "delta": -1,
                "based_on_events": ["E4"],
                "confidence": 0.68,
                "explanation": "x",
            },
        ],
        "ambiguities": [
            "Routine hedge that should be dropped."
        ],
    }

    reduced = pn.reduce_annotation_result(raw_annotation)

    assert len(reduced["appraisal_events"]) == 2
    assert reduced["appraisal_events"][0]["type"] == "narrated_diminishment"
    assert [event["event_id"] for event in reduced["appraisal_events"]] == ["E1", "E2"]
    assert len(reduced["status_effects"]) <= 4
    assert "M. Vinteuil" in {effect["character"] for effect in reduced["status_effects"]}
    assert all(len(effect["based_on_events"]) >= 1 for effect in reduced["status_effects"])
    assert reduced["ambiguities"] == []
    assert pn.validate_annotation_result(reduced, expected_unit_id="v1-p1-combray#p-310-p-311") == []


def test_reprocess_raw_annotations_with_reduce_salvages_invalid_event_types(tmp_path):
    run_dir = tmp_path / "run-reduce"
    pn.prepare_annotation_run(run_dir)
    pn.write_raw_response(
        run_dir,
        "v1-p1-combray#p-17",
        """{
          "characters_present": [
            {
              "canonical_name": "Swann",
              "surface_forms": ["Swann"],
              "presence_type": "explicit",
              "presence_confidence": 0.99
            }
          ],
          "appraisal_events": [
            {
              "event_id": "E9",
              "source": "narrator",
              "target": "Swann",
              "type": "ridicule",
              "polarity": "negative",
              "narrative_stance": "endorsed",
              "confidence": 0.9,
              "evidence": "x",
              "explanation": "x"
            }
          ],
          "status_effects": [
            {
              "character": "Swann",
              "dimension": "general_appraisal",
              "delta": -1,
              "based_on_events": ["E9"],
              "confidence": 0.9,
              "explanation": "x"
            }
          ],
          "ambiguities": ["routine hedge"]
        }""",
    )

    results = pn.reprocess_raw_annotations(run_dir, reduce=True)

    assert results == [
        {
            "unit_id": "v1-p1-combray#p-17",
            "annotation_written": True,
            "parse_error": None,
            "validation_errors": [],
        }
    ]
    reduced = json.loads((run_dir / "annotations" / "v1-p1-combray#p-17.json").read_text())
    assert reduced["appraisal_events"][0]["type"] == "blame"
    assert reduced["appraisal_events"][0]["event_id"] == "E1"
    assert reduced["ambiguities"] == []
