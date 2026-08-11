"""Tests for the foundation re-annotation machinery (proust/foundation.py and
scripts/process_foundation_batch.py).

The foundation pass re-annotates the whole corpus from scratch with prompt v2
plus a per-chapter registry reference sheet; these tests pin the window
derivation, the run preparation contract (v2 prompt fully rendered, registry
hash recorded), the resolution-aware validator, and the batch gates.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from proust import foundation
from proust.annotation import AnnotationUnitSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = REPO_ROOT / "scripts" / "process_foundation_batch.py"

CHAPTER_ID = "v4-p1"  # smallest canonical chapter (21 paragraphs)


# ------------------------------------------------------------- unit windows


def test_windows_are_consecutive_and_the_tail_keeps_the_remainder():
    specs = foundation.derive_foundation_unit_specs(CHAPTER_ID, window=5)
    spans = [(spec.paragraph_start, spec.paragraph_end) for spec in specs]

    assert spans == [(1, 5), (6, 10), (11, 15), (16, 20), (21, 21)]
    assert all(spec.chapter_id == CHAPTER_ID for spec in specs)


def test_unit_ids_follow_the_existing_convention():
    unit_ids = [foundation.foundation_unit_id(spec) for spec in foundation.derive_foundation_unit_specs(CHAPTER_ID)]

    assert unit_ids[0] == f"{CHAPTER_ID}#p-1-p-5"
    # A single-paragraph tail window uses the short "#p-{n}" form.
    assert unit_ids[-1] == f"{CHAPTER_ID}#p-21"


def test_unit_ids_round_trip_through_the_parser():
    for spec in foundation.derive_foundation_unit_specs(CHAPTER_ID):
        parsed = foundation.parse_foundation_unit_id(foundation.foundation_unit_id(spec))
        assert (parsed.chapter_id, parsed.paragraph_start, parsed.paragraph_end) == (
            spec.chapter_id,
            spec.paragraph_start,
            spec.paragraph_end,
        )


def test_malformed_unit_id_is_rejected():
    with pytest.raises(ValueError):
        foundation.parse_foundation_unit_id("v4-p1-p-3")


def test_empty_paragraphs_are_trimmed_from_window_edges(monkeypatch):
    # p-5 and p-6 blank: the first window shrinks to p-1..p-4 and the second
    # starts at p-7; a window of only blank paragraphs disappears entirely.
    texts = ["a", "b", "c", "d", "  ", "", "g", "h", "i", "j", "", "", "", "", "", "p"]
    monkeypatch.setattr(foundation, "_chapter_paragraph_texts", lambda chapter_id: texts)

    spans = [
        (spec.paragraph_start, spec.paragraph_end)
        for spec in foundation.derive_foundation_unit_specs(CHAPTER_ID, window=5)
    ]

    assert spans == [(1, 4), (7, 10), (16, 16)]


def test_interior_empty_paragraphs_are_kept(monkeypatch):
    # Only edges are trimmed: an interior blank cannot be dropped without
    # splitting one window into two non-contiguous spans.
    texts = ["a", "b", "", "d", "e"]
    monkeypatch.setattr(foundation, "_chapter_paragraph_texts", lambda chapter_id: texts)

    specs = foundation.derive_foundation_unit_specs(CHAPTER_ID, window=5)

    assert [(spec.paragraph_start, spec.paragraph_end) for spec in specs] == [(1, 5)]


def test_window_size_must_be_positive():
    with pytest.raises(ValueError):
        foundation.derive_foundation_unit_specs(CHAPTER_ID, window=0)


def test_whole_corpus_windows_cover_every_chapter():
    specs = foundation.derive_foundation_unit_specs()
    chapter_ids = []
    for spec in specs:
        if spec.chapter_id not in chapter_ids:
            chapter_ids.append(spec.chapter_id)

    assert chapter_ids == foundation.canonical_chapter_ids()


# ---------------------------------------------------------------- preparing


@pytest.fixture(scope="module")
def prepared_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("foundation-run") / "foundation-run-test"
    specs = foundation.derive_foundation_unit_specs(CHAPTER_ID)[:2]
    manifest = foundation.prepare_foundation_run(run_dir, CHAPTER_ID, specs, notes="test run")
    return run_dir, manifest


def test_prepare_records_run_type_prompt_version_and_registry_hash(prepared_run):
    run_dir, manifest = prepared_run
    stored = json.loads((run_dir / "run.json").read_text())

    assert manifest["run_type"] == "foundation"
    assert manifest["prompt_version"] == "v2"
    assert manifest["chapter_id"] == CHAPTER_ID
    assert stored["registry"]["content_sha256"] == foundation.registry_content_hash()
    assert stored["registry"]["path"] == "characters.yaml"
    # Portable paths only -- nothing machine-specific baked into run.json.
    assert stored["directories"] == {name: name for name in ("units", "prompts", "raw", "annotations")}
    assert not Path(stored["prompt_path"]).is_absolute()


def test_prepare_renders_v2_prompts_with_no_unfilled_placeholders(prepared_run):
    run_dir, manifest = prepared_run
    reference_sheet = json.loads((run_dir / "reference-sheet.json").read_text())

    assert reference_sheet
    for unit_id in manifest["unit_ids"]:
        prompt = (run_dir / "prompts" / f"{unit_id}.txt").read_text()
        assert "{{" not in prompt
        assert "### Reference sheet" in prompt
        assert '"resolution": "unresolved"' in prompt
        # The reference sheet really is injected, and the passage is the
        # unit's unaltered raw text.
        assert next(iter(reference_sheet)) in prompt
        unit = json.loads((run_dir / "units" / f"{unit_id}.json").read_text())
        assert unit["raw_text"] in prompt
        assert unit["alias_map"] == {}
        assert unit["preprocessed_text"] == unit["raw_text"]


def test_prepare_rejects_units_from_another_chapter(tmp_path):
    specs = [AnnotationUnitSpec(chapter_id="v5", paragraph_start=1, paragraph_end=5)]

    with pytest.raises(ValueError, match="scoped to chapter"):
        foundation.prepare_foundation_run(tmp_path / "run", CHAPTER_ID, specs)


# -------------------------------------------------------------------- plan


def test_plan_batches_runs_within_chapters():
    plan = foundation.build_foundation_plan(window=5, run_size=40)
    run_ids = [run["runId"] for run in plan["runs"]]

    assert plan["unit_count"] == sum(run["unitCount"] for run in plan["runs"])
    assert plan["run_count"] == len(plan["runs"])
    assert run_ids == [f"foundation-run-{index:03d}" for index in range(1, len(run_ids) + 1)]
    assert all(run["unitCount"] <= 40 for run in plan["runs"])
    # Canonical order, and no run straddles a chapter boundary (one run
    # renders exactly one chapter's reference sheet).
    assert [run["chapterId"] for run in plan["runs"]] == sorted(
        [run["chapterId"] for run in plan["runs"]],
        key=lambda chapter_id: foundation.canonical_chapter_ids().index(chapter_id),
    )
    assert plan["unit_count"] == len(foundation.derive_foundation_unit_specs())
    assert sum(volume["unitCount"] for volume in plan["volumes"]) == plan["unit_count"]


# --------------------------------------------------------------- validation


def _annotation(unit_id, characters=None, ambiguities=None, stance="endorsed"):
    characters = characters or [{"canonical_name": "Swann"}]
    primary = characters[0]["canonical_name"]
    return {
        "unit_id": unit_id,
        "characters_present": [
            {
                "canonical_name": character["canonical_name"],
                "surface_forms": [character["canonical_name"]],
                "presence_type": "explicit",
                "presence_confidence": 0.9,
                **({"resolution": character["resolution"]} if "resolution" in character else {}),
            }
            for character in characters
        ],
        "appraisal_events": [
            {
                "event_id": "E1",
                "source": "narrator",
                "target": primary,
                "type": "narrated_elevation",
                "polarity": "positive",
                "narrative_stance": stance,
                "confidence": 0.9,
                "evidence": "x",
                "explanation": "x",
            }
        ],
        "status_effects": [
            {
                "character": primary,
                "dimension": "social_status",
                "delta": 1,
                "based_on_events": ["E1"],
                "confidence": 0.9,
                "explanation": "x",
            }
        ],
        "ambiguities": list(ambiguities or []),
    }


UNIT_ID = "v4-p1#p-1-p-5"


def test_validator_accepts_resolution_fields():
    annotation = _annotation(
        UNIT_ID,
        characters=[
            {"canonical_name": "Swann", "resolution": "resolved"},
            {"canonical_name": "la marquise de Plassans", "resolution": "unresolved"},
        ],
    )

    assert foundation.validate_foundation_result(annotation, expected_unit_id=UNIT_ID) == []
    summary = foundation.resolution_summary(annotation, unit_id=UNIT_ID)
    assert summary["character_count"] == 2
    assert summary["unresolved_count"] == 1
    assert summary["unresolved_names"] == ["la marquise de Plassans"]


def test_validator_accepts_annotations_without_resolution():
    assert foundation.validate_foundation_result(_annotation(UNIT_ID), expected_unit_id=UNIT_ID) == []


def test_validator_rejects_a_garbage_resolution_value():
    annotation = _annotation(UNIT_ID, characters=[{"canonical_name": "Swann", "resolution": "maybe"}])

    errors = foundation.validate_foundation_result(annotation, expected_unit_id=UNIT_ID)

    assert any("resolution must be one of" in error for error in errors)


def test_validator_still_enforces_the_v1_schema():
    annotation = _annotation(UNIT_ID)
    annotation["characters_present"][0]["nickname"] = "le cygne"
    annotation["appraisal_events"][0]["polarity"] = "ambivalent"

    errors = foundation.validate_foundation_result(annotation, expected_unit_id=UNIT_ID)

    assert any("unexpected keys: nickname" in error for error in errors)
    assert any("polarity" in error for error in errors)
    assert foundation.validate_foundation_result("not an annotation") == ["annotation must be a JSON object."]


def test_writer_strips_resolution_and_keeps_a_resolution_record(tmp_path):
    annotation = _annotation(
        UNIT_ID,
        characters=[
            {"canonical_name": "Swann"},
            {"canonical_name": "la marquise de Plassans", "resolution": "unresolved"},
        ],
    )

    path = foundation.write_foundation_result(tmp_path, UNIT_ID, annotation)
    written = json.loads(path.read_text())
    record = json.loads((tmp_path / "resolutions" / f"{UNIT_ID}.json").read_text())

    # annotations/ stay v1-shaped so every downstream consumer keeps working.
    assert all("resolution" not in character for character in written["characters_present"])
    assert record["unresolved_names"] == ["la marquise de Plassans"]


def test_writer_raises_with_joined_errors(tmp_path):
    annotation = _annotation(UNIT_ID, characters=[{"canonical_name": "Swann", "resolution": "maybe"}])

    with pytest.raises(ValueError) as excinfo:
        foundation.write_foundation_result(tmp_path, UNIT_ID, annotation)

    assert "resolution must be one of" in str(excinfo.value)
    assert not (tmp_path / "annotations" / f"{UNIT_ID}.json").exists()


# -------------------------------------------------------------- batch gates


def _synthetic_run(tmp_path, raw_by_unit, chapter_id=CHAPTER_ID):
    run_dir = tmp_path / "foundation-run-999"
    for name in ("units", "prompts", "raw", "annotations"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    for unit_id, raw in raw_by_unit.items():
        (run_dir / "raw" / f"{unit_id}.json").write_text(
            raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "run_type": "foundation",
                "chapter_id": chapter_id,
                "prompt_path": "proust/prompts/prompt_v2.md",
                "unit_ids": list(raw_by_unit),
                "directories": {name: name for name in ("units", "prompts", "raw", "annotations")},
                "alias_map": {},
            }
        )
    )
    return run_dir


def _run_batch(run_dir):
    result = subprocess.run(
        [sys.executable, str(BATCH_SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_batch_writes_annotations_and_passes_a_clean_run(tmp_path):
    units = {
        f"{CHAPTER_ID}#p-1-p-5": _annotation(f"{CHAPTER_ID}#p-1-p-5"),
        f"{CHAPTER_ID}#p-6-p-10": _annotation(f"{CHAPTER_ID}#p-6-p-10"),
    }
    run_dir = _synthetic_run(tmp_path, units)

    report = _run_batch(run_dir)

    assert report["status"] == "ok"
    assert report["reasons"] == []
    assert report["written"] == 2
    assert report["escalate"] == []
    assert set(report["mixed_counts"]) == {"advantage", "prestige", "inclusion"}
    assert (run_dir / "annotations" / f"{CHAPTER_ID}#p-1-p-5.json").exists()
    assert json.loads((run_dir / "gate-report.json").read_text())["status"] == "ok"


def test_batch_gate_trips_on_validation_failure_and_missing_raw(tmp_path):
    good_unit = f"{CHAPTER_ID}#p-1-p-5"
    bad_unit = f"{CHAPTER_ID}#p-6-p-10"
    broken = _annotation(bad_unit)
    broken["appraisal_events"][0]["polarity"] = "ambivalent"
    run_dir = _synthetic_run(tmp_path, {good_unit: _annotation(good_unit), bad_unit: broken})
    # A third unit whose annotator output never landed.
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unit_ids"].append(f"{CHAPTER_ID}#p-11-p-15")
    manifest_path.write_text(json.dumps(manifest))

    report = _run_batch(run_dir)

    assert report["status"] == "gate_tripped"
    assert bad_unit in report["validation_failures"]
    assert report["missing_raw"] == [f"{CHAPTER_ID}#p-11-p-15"]
    assert any("validation failures" in reason for reason in report["reasons"])
    assert any("missing raw output" in reason for reason in report["reasons"])


def test_batch_gate_trips_on_unparseable_raw_output(tmp_path):
    unit_id = f"{CHAPTER_ID}#p-1-p-5"
    run_dir = _synthetic_run(tmp_path, {unit_id: "{not json"})

    report = _run_batch(run_dir)

    assert report["status"] == "gate_tripped"
    assert "not valid JSON" in report["validation_failures"][unit_id]


def test_batch_gate_trips_on_the_unresolved_rate(tmp_path):
    units = {}
    for start in (1, 6, 11, 16):
        unit_id = f"{CHAPTER_ID}#p-{start}-p-{start + 4}"
        units[unit_id] = _annotation(
            unit_id,
            characters=[
                {"canonical_name": "Swann"},
                {"canonical_name": "la marquise de Plassans", "resolution": "unresolved"},
            ],
        )
    run_dir = _synthetic_run(tmp_path, units)

    report = _run_batch(run_dir)

    assert report["status"] == "gate_tripped"
    assert report["unresolved"]["unresolved_rate"] == 0.5
    assert any("unresolved rate" in reason for reason in report["reasons"])
    # The triage inventory names the gap and where to find it.
    assert report["unresolved"]["names"]["la marquise de Plassans"]["count"] == 4
    assert len(report["unresolved"]["names"]["la marquise de Plassans"]["units"]) == 4


def test_batch_escalation_list_is_advisory_and_lists_hard_units(tmp_path):
    clean_unit = f"{CHAPTER_ID}#p-1-p-5"
    ambiguous_unit = f"{CHAPTER_ID}#p-6-p-10"
    uncertain_unit = f"{CHAPTER_ID}#p-11-p-15"
    units = {
        clean_unit: _annotation(clean_unit, ambiguities=["one caveat"]),
        ambiguous_unit: _annotation(ambiguous_unit, ambiguities=["a", "b"]),
        uncertain_unit: _annotation(uncertain_unit, stance="uncertain"),
    }
    run_dir = _synthetic_run(tmp_path, units)

    report = _run_batch(run_dir)

    escalated = {entry["unit_id"]: entry["reasons"] for entry in report["escalate"]}
    assert set(escalated) == {ambiguous_unit, uncertain_unit}
    assert escalated[ambiguous_unit] == ["2 ambiguities"]
    assert escalated[uncertain_unit] == ["uncertain narrative stance"]
    # Escalation alone must not fail the batch.
    assert report["status"] == "ok"


def test_batch_gate_trips_on_the_narrator_guard(tmp_path):
    units = {}
    for start in (1, 6, 11, 16):
        unit_id = f"v1-p2-un-amour-de-swann#p-{start}-p-{start + 4}"
        units[unit_id] = _annotation(unit_id, characters=[{"canonical_name": "le narrateur"}])
    run_dir = _synthetic_run(tmp_path, units, chapter_id="v1-p2-un-amour-de-swann")

    report = _run_batch(run_dir)

    assert report["status"] == "gate_tripped"
    assert len(report["narrator_v1p2_units"]) == 4
    assert any("narrator scored" in reason for reason in report["reasons"])
