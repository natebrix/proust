import json

import proust as pn
import proust.supplement as ps


UNIT_ID = "v1-p1-combray#p-17"
SOURCE_RUN_ID = "run-556"

ACCEPTED_RAW_TEXT = (
    "Charlus parla longuement avec Swann. Le narrateur les observait en silence."
)
ACCEPTED_ALIAS_MAP = {
    # Pre-canonicalization key ("Charlus") that translate_canonical_name must
    # translate to the accepted identity ("baron de Charlus") -- see
    # coverage.REVIEWED_CANONICAL_MERGES.
    "Charlus": {"aliases": ["Charlus", "le baron"]},
    "Swann": {"aliases": ["Swann"]},
}


def _accepted_annotation(character="baron de Charlus"):
    return {
        "unit_id": UNIT_ID,
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
                "character": character,
                "dimension": "social_status",
                "delta": 1,
                "based_on_events": ["E1"],
                "confidence": 0.9,
                "explanation": "x",
            }
        ],
        "ambiguities": [],
    }


def _empty_supplement_annotation():
    return {
        "unit_id": UNIT_ID,
        "characters_present": [],
        "appraisal_events": [],
        "status_effects": [],
        "ambiguities": [],
    }


def _supplement_annotation_for(character, event_id="S1"):
    return {
        "unit_id": UNIT_ID,
        "characters_present": [
            {
                "canonical_name": character,
                "surface_forms": [character],
                "presence_type": "explicit",
                "presence_confidence": 0.8,
            }
        ],
        "appraisal_events": [
            {
                "event_id": event_id,
                "source": "narrator",
                "target": character,
                "type": "admiration",
                "polarity": "positive",
                "narrative_stance": "endorsed",
                "confidence": 0.8,
                "evidence": "x",
                "explanation": "x",
            }
        ],
        "status_effects": [
            {
                "character": character,
                "dimension": "social_status",
                "delta": 1,
                "based_on_events": [event_id],
                "confidence": 0.8,
                "explanation": "x",
            }
        ],
        "ambiguities": [],
    }


def _prepare_accepted_source_run(source_run_dir):
    unit_specs = [pn.AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=17)]
    pn.prepare_annotation_run(source_run_dir, run_id=SOURCE_RUN_ID, unit_specs=unit_specs)

    unit_path = source_run_dir / "units" / f"{UNIT_ID}.json"
    payload = json.loads(unit_path.read_text())
    payload["raw_text"] = ACCEPTED_RAW_TEXT
    payload["alias_map"] = ACCEPTED_ALIAS_MAP
    payload["prior_context"] = "Combray vu de loin."
    unit_path.write_text(json.dumps(payload, ensure_ascii=False))

    pn.write_annotation_result(source_run_dir, UNIT_ID, _accepted_annotation())


def _write_audit(audit_path, source_run_dir_name=SOURCE_RUN_ID):
    audit = {
        "coverage_audit_version": "coverage_audit_v1",
        "units": [
            {
                "unit_id": UNIT_ID,
                "chapter_id": "v1-p1-combray",
                "source_run": source_run_dir_name,
                "scored_characters": ["baron de Charlus"],
                "candidate_additions": [
                    {"character": "Swann", "matched_surface_forms": ["Swann"], "occurrence_count": 1}
                ],
                "narrator_first_person_count": 0,
                "narrator_candidate": True,
                "projected_new_matches_without_narrator": 1,
                "projected_new_matches_with_narrator": 3,
            }
        ],
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def test_prepare_supplement_run_builds_translated_units_prompts_and_manifest(tmp_path):
    outputs_dir = tmp_path / "outputs"
    source_run_dir = outputs_dir / SOURCE_RUN_ID
    _prepare_accepted_source_run(source_run_dir)

    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path)

    output_dir = tmp_path / "supplement-run-001"
    manifest = ps.prepare_supplement_run(
        audit_path,
        [UNIT_ID],
        output_dir,
        notes="pilot batch",
        source_outputs_dir=outputs_dir,
    )

    assert manifest["run_id"] == "supplement-run-001"
    assert manifest["run_type"] == "supplement"
    assert manifest["schema_version"] == "annotation_supplement_v1"
    assert manifest["unit_ids"] == [UNIT_ID]
    assert manifest["notes"] == "pilot batch"
    assert manifest["derived_from"]["source_run_ids"] == [SOURCE_RUN_ID]
    # Portable directory paths (deliverable 1's convention), not baked
    # absolute paths.
    assert manifest["directories"] == {
        "units": "units",
        "prompts": "prompts",
        "raw": "raw",
        "annotations": "annotations",
    }

    manifest_on_disk = json.loads((output_dir / "run.json").read_text())
    assert manifest_on_disk == manifest

    unit_payload = json.loads((output_dir / "units" / f"{UNIT_ID}.json").read_text())
    assert unit_payload["raw_text"] == ACCEPTED_RAW_TEXT
    assert unit_payload["supplement_of_run"] == SOURCE_RUN_ID
    assert unit_payload["candidate_characters"] == ["Swann", "le narrateur"]
    assert unit_payload["accepted_annotation"] == _accepted_annotation()
    # "Charlus" is a pre-canonicalization alias_map key; it must be
    # translated to the accepted identity "baron de Charlus" using
    # coverage.translate_canonical_name, and its aliases preserved.
    assert "Charlus" not in unit_payload["alias_map"]
    assert unit_payload["alias_map"]["baron de Charlus"]["aliases"] == ["Charlus", "le baron"]
    assert unit_payload["alias_map"]["Swann"]["aliases"] == ["Swann"]

    prompt_text = (output_dir / "prompts" / f"{UNIT_ID}.md").read_text()
    assert ACCEPTED_RAW_TEXT in prompt_text
    assert "Combray vu de loin." in prompt_text
    assert json.dumps(["Swann", "le narrateur"], ensure_ascii=False, indent=2) in prompt_text
    assert '"baron de Charlus"' in prompt_text
    assert "{{PASSAGE}}" not in prompt_text
    assert "{{ALIAS_MAP}}" not in prompt_text
    assert "{{ACCEPTED_ANNOTATION}}" not in prompt_text
    assert "{{CANDIDATES}}" not in prompt_text
    assert "{{PRIOR_CONTEXT}}" not in prompt_text


def test_prepare_supplement_run_prior_context_defaults_when_absent(tmp_path):
    outputs_dir = tmp_path / "outputs"
    source_run_dir = outputs_dir / SOURCE_RUN_ID
    _prepare_accepted_source_run(source_run_dir)

    unit_path = source_run_dir / "units" / f"{UNIT_ID}.json"
    payload = json.loads(unit_path.read_text())
    payload["prior_context"] = ""
    unit_path.write_text(json.dumps(payload, ensure_ascii=False))

    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path)

    output_dir = tmp_path / "supplement-run-002"
    ps.prepare_supplement_run(audit_path, [UNIT_ID], output_dir, source_outputs_dir=outputs_dir)

    prompt_text = (output_dir / "prompts" / f"{UNIT_ID}.md").read_text()
    assert "(none provided)" in prompt_text


def test_prepare_supplement_run_rejects_unit_ids_missing_from_audit(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _prepare_accepted_source_run(outputs_dir / SOURCE_RUN_ID)

    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path)

    try:
        ps.prepare_supplement_run(
            audit_path,
            [UNIT_ID, "does-not-exist#p-1"],
            tmp_path / "supplement-run-003",
            source_outputs_dir=outputs_dir,
        )
    except ValueError as exc:
        assert "does-not-exist#p-1" in str(exc)
    else:
        raise AssertionError("expected ValueError for a unit id missing from the audit")


def test_validate_supplement_result_rejects_accepted_character_overlap():
    annotation = _supplement_annotation_for("baron de Charlus")

    errors = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert any("overlap" in error for error in errors)


def test_validate_supplement_result_rejects_non_candidate_target():
    annotation = _supplement_annotation_for("Legrandin")

    errors = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert any("candidate characters" in error for error in errors)


def test_validate_supplement_result_enforces_s_prefixed_event_ids():
    annotation = _supplement_annotation_for("Swann", event_id="E1")

    errors = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert any('"S" prefix' in error for error in errors)


def test_validate_supplement_result_accepts_accepted_character_as_event_source():
    annotation = _supplement_annotation_for("Swann")
    annotation["appraisal_events"][0]["source"] = "baron de Charlus"

    errors = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert errors == []


def test_validate_supplement_result_accepts_unscored_candidate_as_event_source():
    annotation = _supplement_annotation_for("Swann")
    annotation["appraisal_events"][0]["source"] = "le narrateur"

    errors = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert errors == []


def test_validate_supplement_result_accepts_alias_roster_character_as_event_source():
    annotation = _supplement_annotation_for("Swann")
    annotation["appraisal_events"][0]["source"] = "Mme Verdurin"

    with_roster = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
        roster_characters=["Mme Verdurin", "docteur Cottard"],
    )
    without_roster = ps.validate_supplement_result(
        annotation,
        expected_unit_id=UNIT_ID,
        accepted_characters=["baron de Charlus"],
        candidate_characters=["Swann", "le narrateur"],
    )

    assert with_roster == []
    assert any("characters_present" in error for error in without_roster)


def test_canonicalize_supplement_names_reconciles_accent_variants():
    annotation = _supplement_annotation_for("Remi")

    result = ps.canonicalize_supplement_names(annotation, ["Rémi", "Swann"])

    assert result["status_effects"][0]["character"] == "Rémi"
    assert result["appraisal_events"][0]["target"] == "Rémi"
    assert result["characters_present"][0]["canonical_name"] == "Rémi"


def test_canonicalize_supplement_names_leaves_unknown_and_ambiguous_names():
    annotation = _supplement_annotation_for("la jeune fille inconnue")

    result = ps.canonicalize_supplement_names(annotation, ["Rémi", "Swann"])

    assert result["status_effects"][0]["character"] == "la jeune fille inconnue"


def test_canonicalize_supplement_names_maps_untracked_sources_to_unknown():
    annotation = _supplement_annotation_for("Swann")
    annotation["appraisal_events"][0]["source"] = "roi Théodose"

    result = ps.canonicalize_supplement_names(annotation, ["Swann", "Norpois"])

    assert result["appraisal_events"][0]["source"] == "unknown"
    assert result["appraisal_events"][0]["target"] == "Swann"


def test_normalize_supplement_ambiguities_follows_reduction_rule():
    annotation = _supplement_annotation_for("Swann")
    annotation["ambiguities"] = ["first note", "second note"]

    normalized = ps.normalize_supplement_ambiguities(annotation)
    assert normalized["ambiguities"] == []

    annotation["appraisal_events"][0]["narrative_stance"] = "uncertain"
    normalized = ps.normalize_supplement_ambiguities(annotation)
    assert normalized["ambiguities"] == ["first note"]


def test_validate_supplement_result_accepts_valid_and_empty_supplements():
    valid_annotation = _supplement_annotation_for("Swann")
    empty_annotation = _empty_supplement_annotation()

    assert (
        ps.validate_supplement_result(
            valid_annotation,
            expected_unit_id=UNIT_ID,
            accepted_characters=["baron de Charlus"],
            candidate_characters=["Swann", "le narrateur"],
        )
        == []
    )
    assert (
        ps.validate_supplement_result(
            empty_annotation,
            expected_unit_id=UNIT_ID,
            accepted_characters=["baron de Charlus"],
            candidate_characters=["Swann", "le narrateur"],
        )
        == []
    )


def test_write_supplement_result_writes_valid_annotation_and_rejects_invalid_ones(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _prepare_accepted_source_run(outputs_dir / SOURCE_RUN_ID)

    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path)

    output_dir = tmp_path / "supplement-run-004"
    ps.prepare_supplement_run(audit_path, [UNIT_ID], output_dir, source_outputs_dir=outputs_dir)

    valid_annotation = _supplement_annotation_for("Swann")
    annotation_path = ps.write_supplement_result(output_dir, UNIT_ID, valid_annotation)

    assert annotation_path == output_dir / "annotations" / f"{UNIT_ID}.json"
    assert json.loads(annotation_path.read_text()) == valid_annotation

    overlapping_annotation = _supplement_annotation_for("baron de Charlus")
    try:
        ps.write_supplement_result(output_dir, UNIT_ID, overlapping_annotation)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError for a supplement that rescoreds an accepted character")


def test_main_prepare_supplements_writes_run_from_units_file(tmp_path, capsys, monkeypatch):
    import proust.runner as pr

    # The "prepare-supplements" CLI command locates source runs under the
    # default (repo-relative) "outputs" directory, matching the rest of the
    # CLI's --discover-runs default; run from tmp_path so that resolves to
    # our synthetic accepted run instead of the real corpus.
    monkeypatch.chdir(tmp_path)

    outputs_dir = tmp_path / "outputs"
    _prepare_accepted_source_run(outputs_dir / SOURCE_RUN_ID)

    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path)

    units_file = tmp_path / "units.txt"
    units_file.write_text(f"{UNIT_ID}\n")

    output_dir = tmp_path / "supplement-run-cli"
    exit_code = pr.main(
        [
            "prepare-supplements",
            "--audit",
            str(audit_path),
            "--output",
            str(output_dir),
            "--units-file",
            str(units_file),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["unit_count"] == 1
    assert (output_dir / "run.json").exists()
