"""Tests for the prompt-v2 A/B staging scripts (scripts/prepare_ab_run.py,
scripts/compare_ab_run.py). Loaded via importlib like tests/test_wikisource_build.py
does for the sibling migration scripts, since scripts/ isn't a package."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: prepare_ab_run.py's frozen dataclass needs
    # sys.modules[cls.__module__] to resolve its `str | None` annotations
    # (from __future__ import annotations defers them to strings).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_ab_run")
compare = _load("compare_ab_run")


# ------------------------------------------------------------- span bridge


def test_bridge_span_one_to_one_is_identity():
    pairings = [
        {"kind": "one_to_one", "old_ids": ["p-1"], "new_ids": ["p-1"]},
        {"kind": "one_to_one", "old_ids": ["p-2"], "new_ids": ["p-2"]},
        {"kind": "one_to_one", "old_ids": ["p-3"], "new_ids": ["p-3"]},
    ]
    assert prepare.bridge_span(pairings, 1, 3) == (1, 3)


def test_bridge_span_merge_and_split_shift_the_range():
    # old p-2,p-3 merged into new p-2; old p-4 split into new p-4,p-5.
    pairings = [
        {"kind": "one_to_one", "old_ids": ["p-1"], "new_ids": ["p-1"]},
        {"kind": "merge", "old_ids": ["p-2", "p-3"], "new_ids": ["p-2"]},
        {"kind": "split", "old_ids": ["p-4"], "new_ids": ["p-4", "p-5"]},
        {"kind": "one_to_one", "old_ids": ["p-5"], "new_ids": ["p-6"]},
    ]
    assert prepare.bridge_span(pairings, 1, 5) == (1, 6)
    assert prepare.bridge_span(pairings, 2, 4) == (2, 5)


def test_bridge_span_old_only_borrows_from_neighbours():
    # old p-2 is a dropped section-break with no new counterpart. A span
    # START anchored there borrows the FOLLOWING mapped paragraph's start; a
    # span END anchored there borrows the PRECEDING mapped paragraph's end.
    pairings = [
        {"kind": "one_to_one", "old_ids": ["p-1"], "new_ids": ["p-1"]},
        {"kind": "old_only", "old_ids": ["p-2"], "new_ids": []},
        {"kind": "one_to_one", "old_ids": ["p-3"], "new_ids": ["p-2"]},
    ]
    assert prepare.bridge_span(pairings, 2, 3) == (2, 2)
    assert prepare.bridge_span(pairings, 1, 2) == (1, 1)


def test_bridge_span_unmapped_raises():
    pairings = [{"kind": "old_only", "old_ids": ["p-1"], "new_ids": []}]
    with pytest.raises(ValueError):
        prepare.bridge_span(pairings, 1, 1)


# ------------------------------------------------------------- v2 rendering


def test_render_prompt_v2_input_injects_reference_sheet_and_clears_placeholders():
    unit_payload = {"raw_text": "Le passage original de Proust.", "prior_context": ""}
    reference_sheet = {"Swann": {"aliases": ["Swann", "M. Swann"], "notes": ""}}
    template = "REF:{{REFERENCE_SHEET}}\nCTX:{{PRIOR_CONTEXT}}\nTXT:{{PASSAGE}}"

    rendered = prepare.render_prompt_v2_input(unit_payload, reference_sheet, prompt_template=template)

    assert "{{REFERENCE_SHEET}}" not in rendered
    assert "{{PRIOR_CONTEXT}}" not in rendered
    assert "{{PASSAGE}}" not in rendered
    assert '"Swann"' in rendered
    assert "Le passage original de Proust." in rendered
    assert "CTX:[none]" in rendered  # empty prior_context defaults to [none]


def test_render_prompt_v2_input_uses_raw_text_not_preprocessed():
    unit_payload = {"raw_text": "RAW TEXT", "preprocessed_text": "REWRITTEN TEXT", "prior_context": "ctx"}
    rendered = prepare.render_prompt_v2_input(unit_payload, {}, prompt_template="{{PASSAGE}}")
    assert rendered == "RAW TEXT"


def test_render_prompt_v2_input_with_real_template_leaves_no_placeholders():
    import proust as pn

    unit = pn.build_annotation_unit("v1-p1-combray", 17, prior_context_paragraphs=1)
    reference_sheet = {"Swann": {"aliases": ["Swann"], "notes": ""}}

    rendered = prepare.render_prompt_v2_input(unit, reference_sheet)

    assert "{{REFERENCE_SHEET}}" not in rendered
    assert "{{PRIOR_CONTEXT}}" not in rendered
    assert "{{PASSAGE}}" not in rendered
    assert '"Swann"' in rendered
    assert unit["raw_text"] in rendered


# ---------------------------------------------------------- compare: direction


def test_direction_uses_a_quarter_point_band():
    assert compare.direction(0.3) == "positive"
    assert compare.direction(-0.3) == "negative"
    assert compare.direction(0.1) == "neutral"
    assert compare.direction(-0.1) == "neutral"
    assert compare.direction(0.25) == "neutral"  # band edge is exclusive
    assert compare.direction(0.26) == "positive"


def _annotation(characters, events=None, effects=None, ambiguities=None):
    return {
        "characters_present": [{"canonical_name": c} for c in characters],
        "appraisal_events": events or [],
        "status_effects": effects or [],
        "ambiguities": ambiguities or [],
    }


def test_score_annotation_by_character_positive_elevation():
    lens_config = compare._resolve_scoring_lens("advantage")
    annotation = _annotation(
        ["Swann"],
        events=[
            {
                "target": "Swann",
                "type": "narrated_elevation",
                "polarity": "positive",
                "narrative_stance": "endorsed",
                "confidence": 0.9,
            }
        ],
        effects=[{"character": "Swann", "dimension": "social_status", "delta": 2, "confidence": 0.9}],
    )
    scores = compare.score_annotation_by_character(annotation, lens_config)
    assert scores["Swann"] > 0


def test_score_annotation_by_character_empty_annotation_is_empty_dict():
    lens_config = compare._resolve_scoring_lens("advantage")
    assert compare.score_annotation_by_character(None, lens_config) == {}
    assert compare.score_annotation_by_character(_annotation([]), lens_config) == {}


def test_direction_agreement_counts_shared_characters_only():
    net_a = {"Swann": 1.0, "Odette": -1.0}
    net_b = {"Swann": 0.9, "Legrandin": -1.0}
    result = compare.direction_agreement(net_a, net_b)
    assert result["compared"] == 1  # only Swann is shared
    assert result["agree"] == 1
    assert result["characters"]["Swann"]["agree"] is True


def test_direction_agreement_detects_disagreement():
    net_a = {"Swann": 1.0}
    net_b = {"Swann": -1.0}
    result = compare.direction_agreement(net_a, net_b)
    assert result["compared"] == 1
    assert result["agree"] == 0
    assert result["characters"]["Swann"] == {"left": "positive", "right": "negative", "agree": False}


# --------------------------------------------------------- compare: unresolved


def _synthetic_registry():
    from proust.registry import Entity, Registry, SurfaceForm, normalize_text

    registry = Registry()
    entity = Entity(id="rachel", display_name="Rachel")
    registry.entities["rachel"] = entity
    form = SurfaceForm(form="Rachel", entity_id="rachel", scope="global")
    registry.forms.append(form)
    registry._by_norm_form.setdefault(normalize_text("Rachel"), []).append(form)
    return registry


def test_classify_unresolved_exact_registry_match_is_model_error():
    registry = _synthetic_registry()
    assert compare.classify_unresolved(registry, "Rachel") == "registry_miss_model_error"


def test_classify_unresolved_near_variant_is_possible_registry_gap():
    registry = _synthetic_registry()
    assert compare.classify_unresolved(registry, "la petite Rachel") == "possible_registry_gap"


def test_classify_unresolved_no_match_is_legitimate_off_sheet():
    registry = _synthetic_registry()
    assert compare.classify_unresolved(registry, "un inconnu quelconque") == "legitimate_off_sheet"


# ------------------------------------------------------------- compare: unit


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_compare_unit_detects_b_only_open_world_discovery(tmp_path):
    entry = {
        "legacy_unit_id": "v7-p4-le-bal-de-tetes#p-96-p-100",
        "new_unit_id": "v7-p4-le-bal-de-tetes#p-84-p-87",
        "chapter_id": "v7-p4-le-bal-de-tetes",
        "notes": "synthetic",
    }
    accepted = _annotation(["la Berma"])
    annotation_a = _annotation(["la Berma"])
    annotation_b = _annotation(["la Berma", "Rachel"])
    annotation_b["characters_present"][1]["resolution"] = "unresolved"

    _write_json(tmp_path / "accepted" / f"{entry['legacy_unit_id']}.json", accepted)
    _write_json(tmp_path / "annotations-a" / f"{entry['new_unit_id']}.json", annotation_a)
    _write_json(tmp_path / "annotations-b" / f"{entry['new_unit_id']}.json", annotation_b)

    report = compare.compare_unit(entry, tmp_path, _synthetic_registry())

    assert report["present"] == {"accepted": True, "a": True, "b": True}
    b_only_names = [c["canonical_name"] for c in report["b_only_discoveries"]]
    assert b_only_names == ["Rachel"]
    assert report["b_only_discoveries"][0]["resolution"] == "unresolved"
    assert report["unresolved_in_b"] == [
        {"canonical_name": "Rachel", "classification": "registry_miss_model_error"}
    ]
    assert report["missing_vs_accepted"] == {"a": [], "b": []}


def test_compare_unit_tolerates_missing_annotation_files(tmp_path):
    entry = {
        "legacy_unit_id": "v1-p1-combray#p-111-p-115",
        "new_unit_id": "v1-p1-combray#p-111-p-115",
        "chapter_id": "v1-p1-combray",
        "notes": "synthetic, nothing filled in yet",
    }
    # No files written at all -- everything should report as pending, not crash.
    report = compare.compare_unit(entry, tmp_path, _synthetic_registry())

    assert report["present"] == {"accepted": False, "a": False, "b": False}
    assert report["characters"] == {"accepted": [], "a": [], "b": []}
    assert report["b_only_discoveries"] == []
    assert report["missing_vs_accepted"] == {"a": None, "b": None}
    for lens_report in report["direction_by_lens"].values():
        for pair_report in lens_report.values():
            assert pair_report["compared"] == 0


def test_compare_unit_missing_vs_accepted_lists_dropped_characters(tmp_path):
    entry = {
        "legacy_unit_id": "x#p-1",
        "new_unit_id": "x#p-1",
        "chapter_id": "x",
        "notes": "synthetic",
    }
    accepted = _annotation(["Swann", "Odette"])
    annotation_a = _annotation(["Swann"])  # dropped Odette

    _write_json(tmp_path / "accepted" / f"{entry['legacy_unit_id']}.json", accepted)
    _write_json(tmp_path / "annotations-a" / f"{entry['new_unit_id']}.json", annotation_a)

    report = compare.compare_unit(entry, tmp_path, _synthetic_registry())

    assert report["missing_vs_accepted"]["a"] == ["Odette"]
    assert report["missing_vs_accepted"]["b"] is None  # B never landed


# --------------------------------------------------------- compare: aggregate


def test_build_aggregates_counts_open_world_and_unresolved(tmp_path):
    entry_1 = {
        "legacy_unit_id": "a#p-1",
        "new_unit_id": "a#p-1",
        "chapter_id": "a",
        "notes": "",
    }
    entry_2 = {
        "legacy_unit_id": "b#p-1",
        "new_unit_id": "b#p-1",
        "chapter_id": "b",
        "notes": "",
    }
    accepted_1 = _annotation(["Swann"])
    a_1 = _annotation(["Swann"])
    b_1 = _annotation(["Swann", "Rachel"])
    b_1["characters_present"][1]["resolution"] = "unresolved"

    _write_json(tmp_path / "accepted" / f"{entry_1['legacy_unit_id']}.json", accepted_1)
    _write_json(tmp_path / "annotations-a" / f"{entry_1['new_unit_id']}.json", a_1)
    _write_json(tmp_path / "annotations-b" / f"{entry_1['new_unit_id']}.json", b_1)
    # entry_2 left entirely pending.

    registry = _synthetic_registry()
    reports = [
        compare.compare_unit(entry_1, tmp_path, registry),
        compare.compare_unit(entry_2, tmp_path, registry),
    ]
    aggregates = compare.build_aggregates(reports)

    assert aggregates["unit_count"] == 2
    assert aggregates["units_with_accepted"] == 1
    assert aggregates["units_with_annotation_a"] == 1
    assert aggregates["units_with_annotation_b"] == 1
    assert aggregates["units_fully_complete"] == 1
    assert aggregates["open_world_discovery_count"] == 1
    assert aggregates["open_world_discovery_names"] == ["Rachel"]
    assert aggregates["unresolved_counts"]["total"] == 1
    assert aggregates["unresolved_counts"]["registry_miss_model_error"] == 1


def test_compare_ab_run_end_to_end_on_synthetic_manifest(tmp_path):
    entry = {
        "legacy_unit_id": "x#p-1",
        "new_unit_id": "x#p-1",
        "chapter_id": "x",
        "notes": "synthetic end-to-end",
    }
    manifest = {"units": [entry]}
    _write_json(tmp_path / "manifest.json", manifest)
    _write_json(tmp_path / "accepted" / f"{entry['legacy_unit_id']}.json", _annotation(["Swann"]))

    unit_reports, aggregates = compare.compare_ab_run(tmp_path)

    assert len(unit_reports) == 1
    assert aggregates["units_with_accepted"] == 1
    assert aggregates["units_with_annotation_a"] == 0

    report_text = compare.render_report(unit_reports, aggregates)
    assert "x#p-1" in report_text
    assert "pending" in report_text
