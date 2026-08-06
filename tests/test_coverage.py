import json

import proust as pn
import proust.coverage as pc
import proust.runner as pr


def _multi_character_annotation(unit_id, characters):
    # Mirrors tests/test_runner.py's _multi_character_annotation helper.
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


UNIT_1 = "v1-p1-combray#p-17"
UNIT_2 = "v1-p1-combray#p-274-p-275"
UNIT_3 = "v1-p1-combray#p-312-p-313"

# Unit 1: "Charlus" is present under its pre-canonicalization surface form and
# is already scored as "baron de Charlus" (must be excluded as a candidate).
# "Swann" is present three times and unscored (must be a candidate, with the
# right occurrence count). Four first-person markers push it over a
# narrator_min_first_person=2 threshold.
UNIT_1_RAW_TEXT = (
    "Charlus parla longuement avec Swann. "
    "Le petit Marcel se souvenait de Charlus. "
    "J'avais pensé que Charlus ne viendrait pas. "
    "Je me demandais si Swann m'aimait bien. "
    "Mon ami Swann partit."
)
UNIT_1_ALIAS_MAP = {
    "Charlus": {"aliases": ["Charlus"]},
    "Swann": {"aliases": ["Swann"]},
}

# Unit 2: no candidate characters present in the text, and no first-person
# markers -- this is the "nothing to flag" control case.
UNIT_2_RAW_TEXT = "Legrandin salua poliment. Il n'y avait personne d'autre dans le salon ce jour-la."

# Unit 3: heavy first-person narration, but "le narrateur" is already scored,
# so narrator_candidate must stay False even though the marker count clears
# the threshold.
UNIT_3_RAW_TEXT = "Je regardais ma mere. Moi je pensais a mes soucis."


def _prepare_fixture_run(run_dir):
    unit_specs = [
        pn.AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=17),
        pn.AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=274, paragraph_end=275),
        pn.AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=312, paragraph_end=313),
    ]
    pn.prepare_annotation_run(run_dir, unit_specs=unit_specs)

    overrides = {
        UNIT_1: (UNIT_1_RAW_TEXT, UNIT_1_ALIAS_MAP),
        UNIT_2: (UNIT_2_RAW_TEXT, {}),
        UNIT_3: (UNIT_3_RAW_TEXT, {}),
    }
    for unit_id, (raw_text, alias_map) in overrides.items():
        unit_path = run_dir / "units" / f"{unit_id}.json"
        payload = json.loads(unit_path.read_text())
        payload["raw_text"] = raw_text
        payload["alias_map"] = alias_map
        unit_path.write_text(json.dumps(payload, ensure_ascii=False))

    pn.write_annotation_result(run_dir, UNIT_1, _multi_character_annotation(UNIT_1, [{"character": "baron de Charlus"}]))
    pn.write_annotation_result(run_dir, UNIT_2, _multi_character_annotation(UNIT_2, [{"character": "Legrandin"}]))
    pn.write_annotation_result(run_dir, UNIT_3, _multi_character_annotation(UNIT_3, [{"character": "le narrateur"}]))


def test_canonicalize_translates_reviewed_merges_and_passes_through_unknown_names():
    assert pc._canonicalize("Charlus") == "baron de Charlus"
    assert pc._canonicalize("Saint-Loup") == "Robert de Saint-Loup"
    assert pc._canonicalize("princesse des Laumes") == "duchesse de Guermantes"
    assert pc._canonicalize("Someone Never Merged") == "Someone Never Merged"


def test_build_surface_index_excludes_short_forms_and_self_maps_canonical_names(tmp_path):
    run_dir = tmp_path / "run-001"
    units_dir = run_dir / "units"
    units_dir.mkdir(parents=True)
    (units_dir / "u1.json").write_text(
        json.dumps(
            {
                "alias_map": {
                    "Charlus": {"aliases": ["Charlus", "le baron", "Mo"]},
                }
            }
        )
    )

    index = pc._build_surface_index([run_dir])

    assert index["Charlus"] == "baron de Charlus"
    assert index["le baron"] == "baron de Charlus"
    assert "Mo" not in index
    # The self-identity finalization pass makes the accepted name itself a
    # matchable surface form, taking priority over any conflicting data.
    assert index["baron de Charlus"] == "baron de Charlus"


def test_find_candidate_additions_prefers_longest_surface_form_first():
    surface_index = {
        "Mme de Cambremer": "Mme de Cambremer",
        "Cambremer": "marquis de Cambremer",
    }
    raw_text = "Mme de Cambremer arriva. Cambremer resta seul."

    candidates = pc._find_candidate_additions(raw_text, surface_index, set())
    by_character = {row["character"]: row for row in candidates}

    assert by_character["Mme de Cambremer"]["occurrence_count"] == 1
    assert by_character["Mme de Cambremer"]["matched_surface_forms"] == ["Mme de Cambremer"]
    # The "Cambremer" inside "Mme de Cambremer" is claimed by the longer
    # match and must not also be attributed to "marquis de Cambremer"; only
    # the standalone second occurrence should count.
    assert by_character["marquis de Cambremer"]["occurrence_count"] == 1
    assert by_character["marquis de Cambremer"]["matched_surface_forms"] == ["Cambremer"]


def test_find_candidate_additions_excludes_already_scored_characters():
    surface_index = {"Odette": "Odette", "Swann": "Swann"}
    raw_text = "Odette et Swann se promenaient."

    candidates = pc._find_candidate_additions(raw_text, surface_index, {"Odette"})

    assert [row["character"] for row in candidates] == ["Swann"]


def test_count_narrator_markers_counts_all_marker_forms():
    text = "J'aime ma mere. Moi, je pense a mes soucis avec mon pere. Il me voit."

    assert pc._count_narrator_markers(text) == 7


def test_strip_quoted_spans_removes_guillemet_quotations():
    text = 'Mme Verdurin disait : « Je vous aime, moi, mon ami. » Le narrateur se taisait.'

    stripped = pc._strip_quoted_spans(text)

    assert "Je vous aime" not in stripped
    assert "moi, mon ami" not in stripped
    assert "Mme Verdurin disait" in stripped
    assert "Le narrateur se taisait." in stripped


def test_strip_quoted_spans_handles_guillemet_span_across_a_paragraph_break():
    text = "Avant « voici\n\nune longue citation » apres."

    stripped = pc._strip_quoted_spans(text)

    assert "voici" not in stripped
    assert "citation" not in stripped
    assert "Avant" in stripped
    assert "apres." in stripped


def test_strip_quoted_spans_drops_dash_initiated_dialogue_paragraphs():
    text = (
        "Je pensais a autre chose.\n\n"
        "– Je t'aime, moi, dit-elle. Mon coeur est a toi.\n\n"
        "— Ne dis pas cela, moi je le crois si peu.\n\n"
        "Mes pensees restaient ailleurs."
    )

    stripped = pc._strip_quoted_spans(text)

    assert "Je t'aime" not in stripped
    assert "Ne dis pas cela" not in stripped
    assert "Je pensais a autre chose." in stripped
    assert "Mes pensees restaient ailleurs." in stripped


def test_count_narrator_markers_excludes_quoted_dialogue():
    # Regression case for the "Un amour de Swann" over-triggering bug:
    # quoted dialogue is full of first-person markers that belong to the
    # speaking character, not the narrating voice, and must not count.
    text = (
        "Swann observait la scene sans se meler a la conversation.\n\n"
        "« Je t'aime, moi, mon ami », dit Odette en riant.\n\n"
        "– Mon coeur est a toi, repondit Swann.\n\n"
        "Odette sourit sans rien ajouter."
    )

    assert pc._count_narrator_markers(text) == 0


def test_projected_new_matches_uses_combinatorial_growth():
    assert pc._projected_new_matches(1, 2) == 1
    assert pc._projected_new_matches(2, 4) == 5
    assert pc._projected_new_matches(3, 3) == 0


def test_build_coverage_audit_flags_unscored_characters_and_narrator_presence(tmp_path):
    run_dir = tmp_path / "run-001"
    _prepare_fixture_run(run_dir)

    audit = pc.build_coverage_audit([run_dir], narrator_min_first_person=2)

    assert audit["coverage_audit_version"] == "coverage_audit_v1"
    assert audit["metadata"]["unit_count"] == 3
    assert audit["metadata"]["duplicate_resolution"] == "latest_reviewed_run_wins"

    units_by_id = {row["unit_id"]: row for row in audit["units"]}

    unit_1 = units_by_id[UNIT_1]
    assert unit_1["source_run"] == "run-001"
    assert unit_1["scored_characters"] == ["baron de Charlus"]
    assert len(unit_1["candidate_additions"]) == 1
    candidate = unit_1["candidate_additions"][0]
    assert candidate["character"] == "Swann"
    assert candidate["occurrence_count"] == 3
    assert candidate["matched_surface_forms"] == ["Swann"]
    assert unit_1["narrator_first_person_count"] == 4
    assert unit_1["narrator_candidate"] is True
    assert unit_1["projected_new_matches_without_narrator"] == 1
    assert unit_1["projected_new_matches_with_narrator"] == 3

    unit_2 = units_by_id[UNIT_2]
    assert unit_2["candidate_additions"] == []
    assert unit_2["narrator_candidate"] is False
    assert unit_2["projected_new_matches_without_narrator"] == 0
    assert unit_2["projected_new_matches_with_narrator"] == 0

    unit_3 = units_by_id[UNIT_3]
    assert unit_3["scored_characters"] == ["le narrateur"]
    assert unit_3["narrator_first_person_count"] == 5
    # Already scored as "le narrateur" -> not a narrator candidate even
    # though the marker count clears the threshold.
    assert unit_3["narrator_candidate"] is False
    assert unit_3["candidate_additions"] == []

    assert audit["metadata"]["flagged_unit_count"] == 1
    assert audit["metadata"]["candidate_addition_count"] == 1
    assert audit["metadata"]["narrator_candidate_unit_count"] == 1

    corpus_totals = audit["summary"]["corpus_totals"]
    assert corpus_totals["projected_new_matches_without_narrator"] == 1
    assert corpus_totals["projected_new_matches_with_narrator"] == 3

    characters_by_name = {row["character"]: row for row in audit["summary"]["characters"]}
    assert set(characters_by_name) == {"baron de Charlus", "Swann", "le narrateur"}
    for row in characters_by_name.values():
        assert row["flagged_unit_count"] == 1
        assert row["projected_new_matches_without_narrator"] == 1
        assert row["projected_new_matches_with_narrator"] == 3


def test_build_coverage_audit_narrator_candidate_ignores_quoted_dialogue(tmp_path):
    run_dir = tmp_path / "run-101"
    unit_specs = [pn.AnnotationUnitSpec(chapter_id="v1-p1-combray", paragraph_start=17)]
    pn.prepare_annotation_run(run_dir, unit_specs=unit_specs)

    unit_id = "v1-p1-combray#p-17"
    quoted_raw_text = (
        "Le narrateur observait la scene sans se meler a la conversation.\n\n"
        "– Je t'aime, moi, dit Swann a Odette. Mon coeur est a toi.\n\n"
        "Odette sourit sans repondre."
    )
    unit_path = run_dir / "units" / f"{unit_id}.json"
    payload = json.loads(unit_path.read_text())
    payload["raw_text"] = quoted_raw_text
    payload["alias_map"] = {}
    unit_path.write_text(json.dumps(payload, ensure_ascii=False))

    pn.write_annotation_result(run_dir, unit_id, _multi_character_annotation(unit_id, [{"character": "Swann"}]))

    audit = pc.build_coverage_audit([run_dir], narrator_min_first_person=2)

    assert audit["metadata"]["unit_count"] == 1
    unit_row = audit["units"][0]
    assert unit_row["unit_id"] == unit_id
    assert unit_row["source_run"] == "run-101"
    # Without quote-stripping, the dash-initiated dialogue line alone would
    # clear narrator_min_first_person=2 ("je", "moi", "mon"); stripped, the
    # surrounding narration contributes zero first-person markers.
    assert unit_row["narrator_first_person_count"] == 0
    assert unit_row["narrator_candidate"] is False


def test_build_coverage_audit_requires_run_dirs():
    try:
        pc.build_coverage_audit([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an empty run_dirs list")


def test_main_coverage_audit_can_write_artifacts(tmp_path, capsys):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "run-001"
    _prepare_fixture_run(run_dir)

    json_output = tmp_path / "coverage-audit.json"
    markdown_output = tmp_path / "coverage-audit.md"

    exit_code = pr.main(
        [
            "coverage-audit",
            "--discover-runs",
            str(outputs_dir),
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    payload = json.loads(json_output.read_text())
    markdown = markdown_output.read_text()

    assert exit_code == 0
    assert summary["unit_count"] == 3
    assert summary["flagged_unit_count"] == 1
    assert summary["candidate_addition_count"] == 1
    assert summary["narrator_candidate_unit_count"] == 1
    assert payload["coverage_audit_version"] == "coverage_audit_v1"
    assert markdown.startswith("# Coverage Audit")
    assert "Swann" in markdown
