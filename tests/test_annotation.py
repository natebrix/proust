import proust as pn


def test_build_annotation_unit_uses_canonical_coordinates_and_reader_urls():
    unit = pn.build_annotation_unit("v1-p1-combray", 17, prior_context_paragraphs=1)

    assert unit["unit_id"] == "v1-p1-combray#p-17"
    assert unit["chapter_id"] == "v1-p1-combray"
    assert unit["paragraph_start"] == 17
    assert unit["paragraph_end"] == 17
    assert unit["reader_urls"] == {
        "fr-original": "https://nathanbrixius.com/projects/islt/fr-original/v1-p1-combray#p-17",
        "en-moncrieff": "https://nathanbrixius.com/projects/islt/en-moncrieff/v1-p1-combray#p-17",
    }
    assert "Swann, le fils" in unit["raw_text"]
    assert "Swann" in unit["preprocessed_text"]
    assert unit["prior_context"]
    assert unit["source_chapter_range"] == {"start": "001", "end": "043"}


def test_build_annotation_unit_aliases_do_not_corrupt_other_words():
    unit = pn.build_annotation_unit(
        "v2-p2-noms-de-pays-le-pays",
        214,
        215,
        alias_map={
            "Charlus": {
                "aliases": ["Charlus", "l'oncle de Saint-Loup"],
                "notes": "",
            },
            "la grand-mère du narrateur": {
                "aliases": ["ma grand'mère"],
                "notes": "",
            },
            "le narrateur": {
                "aliases": ["je", "moi"],
                "notes": "",
            },
        },
    )

    assert "le narrateurns" not in unit["preprocessed_text"]
    assert "oble narrateurt" not in unit["preprocessed_text"]
    assert "moins idéologue" in unit["preprocessed_text"]
    assert "l'objet de notre observation" in unit["preprocessed_text"]


def test_build_starter_units_returns_three_prompt_ready_units():
    units = pn.build_starter_units()

    assert [unit["unit_id"] for unit in units] == [
        "v1-p1-combray#p-17",
        "v1-p1-combray#p-274-p-275",
        "v1-p1-combray#p-312-p-313",
    ]
    assert "Swann" in units[0]["alias_map"]
    assert units[1]["notes"] == "Legrandin's reaction to Guermantes exposes snobbery."


def test_render_prompt_input_injects_alias_map_context_and_passage():
    unit = pn.build_annotation_unit("v1-p1-combray", 17, prior_context_paragraphs=1)
    prompt = pn.render_prompt_input(unit)

    assert "{{ALIAS_MAP}}" not in prompt
    assert "{{PRIOR_CONTEXT}}" not in prompt
    assert "{{PASSAGE}}" not in prompt
    assert '"Swann"' in prompt
    assert unit["preprocessed_text"] in prompt
    assert unit["prior_context"] in prompt
