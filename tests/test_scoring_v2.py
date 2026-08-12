"""Tests for the scoring v2 formula (proust/scoring_v2.py).

Every test here pins one clause of proust/docs/scoring_v2_design.md: the
dimension partition and its advantage-internal weights, movement as
delta x confidence with no event term, mixed labels only on genuine sign
conflict, uncertainty entering the weight and never the direction, and the
two entity keyings (name view and person view) coming off one comparison.
"""

import pytest

from proust import scoring_v2, scoring_v2_build
from proust.registry import REGISTRY_PATH, Registry


@pytest.fixture(scope="module")
def registry():
    return Registry.load(REGISTRY_PATH)


def _character(name, presence_confidence=0.9):
    return {
        "canonical_name": name,
        "surface_forms": [name],
        "presence_type": "explicit",
        "presence_confidence": presence_confidence,
    }


def _event(event_id, target, stance="endorsed", source="collective_social_voice"):
    return {
        "event_id": event_id,
        "source": source,
        "target": target,
        "type": "admiration",
        "polarity": "positive",
        "narrative_stance": stance,
        "confidence": 0.9,
        "evidence": "evidence",
        "explanation": "explanation",
    }


def _effect(character, dimension, delta, confidence=1.0, based_on_events=("E1",)):
    return {
        "character": character,
        "dimension": dimension,
        "delta": delta,
        "based_on_events": list(based_on_events),
        "confidence": confidence,
        "explanation": "explanation",
    }


def _annotation(characters, events=(), effects=(), ambiguities=(), unit_id="v1-p1-combray#p-1-p-5"):
    return {
        "unit_id": unit_id,
        "characters_present": list(characters),
        "appraisal_events": list(events),
        "status_effects": list(effects),
        "ambiguities": list(ambiguities),
    }


# ---------------------------------------------------------------- movements


def test_the_lens_projection_is_a_partition_of_the_dimensions():
    # No dimension may feed two lenses: that is what makes the lenses
    # orthogonal by construction rather than by hope.
    claimed = [
        dimension
        for weights in scoring_v2.LENS_DIMENSION_WEIGHTS.values()
        for dimension in weights
    ]
    assert sorted(claimed) == sorted(set(claimed))
    assert set(claimed) == {
        "social_status",
        "inclusion_exclusion",
        "general_appraisal",
        "emotional_position",
        "rhetorical_position",
    }


def test_movement_is_delta_times_confidence_times_the_dimension_weight():
    annotation = _annotation(
        [_character("Swann")],
        events=[_event("E1", "Swann")],
        effects=[
            _effect("Swann", "general_appraisal", 2, confidence=0.5),
            _effect("Swann", "emotional_position", 1, confidence=0.5),
            _effect("Swann", "rhetorical_position", -1, confidence=1.0),
        ],
    )

    # 1.0*2*0.5 + 0.8*1*0.5 - 0.6*1*1.0
    assert scoring_v2.unit_movements(annotation, "advantage")["Swann"] == pytest.approx(0.8)
    # Those three dimensions belong to advantage alone.
    assert scoring_v2.unit_movements(annotation, "prestige")["Swann"] == 0.0
    assert scoring_v2.unit_movements(annotation, "inclusion")["Swann"] == 0.0


def test_prestige_and_inclusion_read_their_own_dimension_at_full_weight():
    annotation = _annotation(
        [_character("Odette")],
        events=[_event("E1", "Odette")],
        effects=[
            _effect("Odette", "social_status", 2, confidence=0.8),
            _effect("Odette", "inclusion_exclusion", -1, confidence=0.6),
        ],
    )

    assert scoring_v2.unit_movements(annotation, "prestige")["Odette"] == pytest.approx(1.6)
    assert scoring_v2.unit_movements(annotation, "inclusion")["Odette"] == pytest.approx(-0.6)
    assert scoring_v2.unit_movements(annotation, "advantage")["Odette"] == 0.0


def test_events_contribute_no_movement_of_their_own():
    # v1 added an event term on top of the effects the events support,
    # counting one reading twice. A unit with events but no effects must
    # now move nobody.
    annotation = _annotation(
        [_character("Bloch"), _character("Saniette")],
        events=[_event("E1", "Bloch"), _event("E2", "Saniette")],
    )

    for lens in scoring_v2.SCORING_V2_LENS_ORDER:
        assert scoring_v2.unit_movements(annotation, lens) == {"Bloch": 0.0, "Saniette": 0.0}


def test_a_character_present_without_effects_still_gets_a_zero_movement():
    annotation = _annotation(
        [_character("Swann"), _character("Françoise")],
        events=[_event("E1", "Swann")],
        effects=[_effect("Swann", "general_appraisal", 1)],
    )

    movements = scoring_v2.unit_movements(annotation, "advantage")
    assert movements == {"Swann": 1.0, "Françoise": 0.0}


def test_an_effect_naming_an_absent_character_is_skipped():
    annotation = _annotation(
        [_character("Swann")],
        events=[_event("E1", "Swann")],
        effects=[
            _effect("Swann", "general_appraisal", 1),
            _effect("un inconnu", "general_appraisal", 2),
        ],
    )

    assert scoring_v2.unit_movements(annotation, "advantage") == {"Swann": 1.0}


def test_unknown_lenses_are_rejected():
    with pytest.raises(ValueError, match="advantage"):
        scoring_v2.unit_movements(_annotation([]), "charm")


# ------------------------------------------------------------------- labels


def test_labels_follow_the_movement_past_the_tie_band():
    annotation = _annotation(
        [_character("up"), _character("down"), _character("still")],
        events=[_event("E1", "up"), _event("E2", "down")],
        effects=[
            _effect("up", "general_appraisal", 1, confidence=0.9),
            _effect("down", "general_appraisal", -1, confidence=0.9, based_on_events=("E2",)),
        ],
    )

    labels = scoring_v2.unit_labels(annotation, "advantage")
    assert labels == {"up": "positive", "down": "negative", "still": "neutral"}


def test_mixed_requires_a_genuine_sign_conflict_within_the_lens():
    # Two effects that cancel each other inside advantage: the character
    # is pulled both ways at once, which is exactly what mixed means.
    conflicted = _annotation(
        [_character("oncle Adolphe")],
        events=[_event("E1", "oncle Adolphe")],
        effects=[
            _effect("oncle Adolphe", "general_appraisal", 1, confidence=0.8),
            _effect("oncle Adolphe", "emotional_position", -1, confidence=1.0),
        ],
    )

    assert scoring_v2.unit_movements(conflicted, "advantage")["oncle Adolphe"] == pytest.approx(0.0)
    assert scoring_v2.unit_labels(conflicted, "advantage")["oncle Adolphe"] == "mixed"


def test_a_sign_conflict_across_lenses_is_not_mixed_in_either_lens():
    # The conflict must be INSIDE the lens. A character raised in status
    # and shut out socially is positive in prestige and negative in
    # inclusion -- two clean readings, not one muddled one.
    annotation = _annotation(
        [_character("Odette")],
        events=[_event("E1", "Odette")],
        effects=[
            _effect("Odette", "social_status", 2, confidence=0.9),
            _effect("Odette", "inclusion_exclusion", -2, confidence=0.9),
        ],
    )

    assert scoring_v2.unit_labels(annotation, "prestige")["Odette"] == "positive"
    assert scoring_v2.unit_labels(annotation, "inclusion")["Odette"] == "negative"


def test_ambiguity_notes_never_manufacture_a_label():
    # In v1 an ambiguity penalty could push an untouched character's net
    # score below the loss threshold. Notes now weigh comparisons only.
    quiet = _annotation([_character("Saniette")])
    hedged = _annotation([_character("Saniette")], ambiguities=["a", "b", "c", "d"])

    for lens in scoring_v2.SCORING_V2_LENS_ORDER:
        assert scoring_v2.unit_labels(quiet, lens) == scoring_v2.unit_labels(hedged, lens)
        assert scoring_v2.unit_labels(hedged, lens)["Saniette"] == "neutral"


def test_a_conflict_that_still_clears_the_band_is_labelled_by_its_direction():
    annotation = _annotation(
        [_character("Charlus")],
        events=[_event("E1", "Charlus")],
        effects=[
            _effect("Charlus", "general_appraisal", 2, confidence=1.0),
            _effect("Charlus", "rhetorical_position", -1, confidence=0.5),
        ],
    )

    assert scoring_v2.unit_movements(annotation, "advantage")["Charlus"] == pytest.approx(1.7)
    assert scoring_v2.unit_labels(annotation, "advantage")["Charlus"] == "positive"


# -------------------------------------------------------------- comparisons


def test_comparison_direction_comes_from_the_movements_alone():
    annotation = _annotation(
        [_character("Swann"), _character("Bloch"), _character("Cottard")],
        events=[_event("E1", "Swann"), _event("E2", "Bloch")],
        effects=[
            _effect("Swann", "general_appraisal", 1, confidence=1.0),
            _effect("Bloch", "general_appraisal", -1, confidence=1.0, based_on_events=("E2",)),
        ],
    )

    outcomes = {
        (row["character_a"], row["character_b"]): row["observed_a"]
        for row in scoring_v2.unit_comparisons(annotation, "advantage")
    }
    assert outcomes[("Bloch", "Swann")] == 0.0
    assert outcomes[("Bloch", "Cottard")] == 0.0
    assert outcomes[("Cottard", "Swann")] == 0.0


def test_movements_inside_the_tie_band_are_drawn():
    annotation = _annotation(
        [_character("a"), _character("b")],
        events=[_event("E1", "a")],
        effects=[_effect("a", "general_appraisal", 1, confidence=0.2)],
    )

    row = scoring_v2.unit_comparisons(annotation, "advantage")[0]
    assert row["movement_a"] == pytest.approx(0.2)
    assert row["observed_a"] == 0.5 and row["observed_b"] == 0.5


def test_the_weight_is_the_ambiguity_discount_times_the_weaker_reading():
    annotation = _annotation(
        [_character("a", presence_confidence=0.95), _character("b")],
        events=[_event("E1", "a"), _event("E2", "b")],
        effects=[
            _effect("a", "general_appraisal", 1, confidence=0.8),
            _effect("b", "general_appraisal", 2, confidence=0.6, based_on_events=("E2",)),
        ],
        ambiguities=["one note"],
    )

    row = scoring_v2.unit_comparisons(annotation, "advantage")[0]
    assert row["ambiguity_weight"] == pytest.approx(0.8)
    assert row["confidence_a"] == pytest.approx(0.8)
    assert row["confidence_b"] == pytest.approx(0.6)
    assert row["weight"] == pytest.approx(0.8 * 0.6)


def test_the_ambiguity_discount_decays_to_a_floor():
    assert scoring_v2.ambiguity_weight(_annotation([])) == 1.0
    assert scoring_v2.ambiguity_weight(_annotation([], ambiguities=["a"])) == pytest.approx(0.8)
    assert scoring_v2.ambiguity_weight(_annotation([], ambiguities=["a", "b"])) == pytest.approx(0.64)
    # 0.8**3 = 0.512 is still above the floor; 0.8**4 = 0.4096 is not.
    assert scoring_v2.ambiguity_weight(_annotation([], ambiguities=list("abc"))) == pytest.approx(0.512)
    assert scoring_v2.ambiguity_weight(_annotation([], ambiguities=list("abcdefgh"))) == 0.5


def test_an_uncertain_supporting_event_discounts_only_the_weight():
    endorsed = _annotation(
        [_character("a"), _character("b")],
        events=[_event("E1", "a")],
        effects=[_effect("a", "general_appraisal", 1, confidence=1.0)],
    )
    uncertain = _annotation(
        [_character("a"), _character("b")],
        events=[_event("E1", "a", stance="uncertain")],
        effects=[_effect("a", "general_appraisal", 1, confidence=1.0)],
    )

    endorsed_row = scoring_v2.unit_comparisons(endorsed, "advantage")[0]
    uncertain_row = scoring_v2.unit_comparisons(uncertain, "advantage")[0]

    assert endorsed_row["confidence_a"] == pytest.approx(1.0)
    assert uncertain_row["confidence_a"] == pytest.approx(0.7)
    # Direction and magnitude of the reading are untouched.
    assert uncertain_row["movement_a"] == endorsed_row["movement_a"]
    assert uncertain_row["observed_a"] == endorsed_row["observed_a"]


def test_a_zero_effect_character_falls_back_to_presence_confidence():
    annotation = _annotation(
        [_character("Swann", presence_confidence=0.99), _character("la dame en rose", presence_confidence=0.4)],
        events=[_event("E1", "Swann")],
        effects=[_effect("Swann", "general_appraisal", 2, confidence=0.9)],
    )

    confidences = scoring_v2.unit_confidences(annotation, "advantage")
    assert confidences["Swann"] == pytest.approx(0.9)
    assert confidences["la dame en rose"] == pytest.approx(0.4)
    row = scoring_v2.unit_comparisons(annotation, "advantage")[0]
    assert row["weight"] == pytest.approx(0.4)


def test_vacuous_pair_rule_scopes_comparisons_to_lens_participants():
    # Amendment (2026-08-12): a pair where NEITHER character has an effect
    # in the lens is vacuous and emits nothing; a bystander still compares
    # against anyone who moved.
    annotation = _annotation(
        [_character("a"), _character("b"), _character("c")],
        events=[_event("E1", "a")],
        effects=[_effect("a", "general_appraisal", 2)],
    )

    # prestige saw nobody move: no comparisons at all
    assert scoring_v2.unit_comparisons(annotation, "prestige") == []
    # advantage saw "a" move: a-b and a-c stay, b-c is vacuous
    advantage = scoring_v2.unit_comparisons(annotation, "advantage")
    pairs = {(row["character_a"], row["character_b"]) for row in advantage}
    assert pairs == {("a", "b"), ("a", "c")}


def test_comparisons_carry_the_unit_id_and_a_stable_pair_order():
    annotation = _annotation(
        [_character("Zola"), _character("Bloch"), _character("Swann")],
        events=[_event("E1", "Bloch"), _event("E2", "Swann")],
        effects=[
            _effect("Bloch", "social_status", 1),
            _effect("Swann", "social_status", -1),
        ],
        unit_id="v2-p2-noms-de-pays#p-10-p-14",
    )

    rows = scoring_v2.unit_comparisons(annotation, "prestige")
    assert [(row["character_a"], row["character_b"]) for row in rows] == [
        ("Bloch", "Swann"),
        ("Bloch", "Zola"),
        ("Swann", "Zola"),
    ]
    assert {row["unit_id"] for row in rows} == {"v2-p2-noms-de-pays#p-10-p-14"}


# ---------------------------------------------------------------- keying


def test_person_view_merges_a_pre_revelation_identity(registry):
    merge_map = scoring_v2.person_view_merge_map(registry)

    # "le peintre" of the Verdurin salon IS Elstir; the person view says so.
    assert merge_map["le-peintre"] == "elstir"
    assert scoring_v2.person_view_key("le peintre", registry=registry) == "elstir"
    assert scoring_v2.person_view_key("Elstir", registry=registry) == "elstir"
    # The name view keeps the novel's own withholding intact.
    assert scoring_v2.name_view_key("le peintre") == "le peintre"


def test_person_view_merges_a_pre_succession_title(registry):
    assert scoring_v2.person_view_key("prince des Laumes", registry=registry) == "duc-de-guermantes"
    assert scoring_v2.person_view_key("duc de Guermantes", registry=registry) == "duc-de-guermantes"


def test_keep_separate_links_never_merge(registry):
    # Mme Verdurin holds the princesse de Guermantes title after the
    # original princesse dies: one title, two people. Merging them would
    # fuse two women into a single rating.
    merge_map = scoring_v2.person_view_merge_map(registry)
    assert "princesse-de-guermantes" not in merge_map
    assert scoring_v2.person_view_key("princesse de Guermantes", registry=registry) == (
        "princesse-de-guermantes"
    )
    assert scoring_v2.person_view_key("Mme Verdurin", registry=registry) == "mme-verdurin"


def test_an_unresolved_name_keys_on_itself(registry):
    assert scoring_v2.person_view_key("un personnage inconnu", registry=registry) == (
        "un personnage inconnu"
    )
    assert scoring_v2.person_view_key("narrator", registry=registry) == "narrator"


def test_comparisons_carry_both_keyings(registry):
    annotation = _annotation(
        [_character("le peintre"), _character("Swann"), _character("un inconnu")],
        events=[_event("E1", "le peintre"), _event("E2", "Swann")],
        effects=[
            _effect("le peintre", "social_status", 2),
            _effect("Swann", "social_status", 1),
        ],
    )

    rows = {
        (row["character_a"], row["character_b"]): row
        for row in scoring_v2.unit_comparisons(annotation, "prestige", registry=registry)
    }
    painter_row = rows[("Swann", "le peintre")]
    assert painter_row["person_a"] == "swann"
    assert painter_row["person_b"] == "elstir"
    assert rows[("Swann", "un inconnu")]["person_b"] == "un inconnu"


def test_a_person_view_self_pairing_is_visible_rather_than_dropped(registry):
    # Both era names in one unit: the person view collapses them onto one
    # entity, which the rating layer cannot play against itself. The
    # comparison still reports the collapse instead of hiding it.
    annotation = _annotation(
        [_character("le peintre"), _character("Elstir")],
        events=[_event("E1", "Elstir")],
        effects=[_effect("Elstir", "social_status", 1)],
    )

    row = scoring_v2.unit_comparisons(annotation, "prestige", registry=registry)[0]
    assert row["character_a"] != row["character_b"]
    assert row["person_a"] == row["person_b"] == "elstir"


def test_person_keys_fall_back_to_names_without_a_registry():
    annotation = _annotation(
        [_character("le peintre"), _character("Swann")],
        events=[_event("E1", "le peintre")],
        effects=[_effect("le peintre", "social_status", 1)],
    )

    row = scoring_v2.unit_comparisons(annotation, "prestige")[0]
    assert row["person_a"] == row["character_a"]
    assert row["person_b"] == row["character_b"]


# ------------------------------------------------------- corpus assembly


def _unit(time, annotation, chapter_id="v1-p1-combray"):
    return {
        "unit_id": annotation["unit_id"],
        "chapter_id": chapter_id,
        "chapter_index": 1,
        "volume_number": 1,
        "unit_index_within_chapter": time,
        "time": time,
        "annotation": annotation,
    }


def _corpus():
    units = []
    for index in range(1, 7):
        unit_id = f"v1-p1-combray#p-{index}-p-{index}"
        units.append(
            _unit(
                index,
                _annotation(
                    [_character("Swann"), _character("Bloch"), _character("Saniette")],
                    events=[_event("E1", "Swann"), _event("E2", "Saniette")],
                    effects=[
                        _effect("Swann", "general_appraisal", 2, confidence=0.9),
                        _effect("Saniette", "general_appraisal", -2, confidence=0.9, based_on_events=("E2",)),
                    ],
                    unit_id=unit_id,
                ),
            )
        )
    return units


def test_view_matches_drops_person_view_self_pairings(registry):
    annotation = _annotation(
        [_character("le peintre"), _character("Elstir"), _character("Swann")],
        events=[_event("E1", "Elstir")],
        effects=[_effect("Elstir", "social_status", 2)],
        unit_id="v1-p2-un-amour-de-swann#p-1-p-5",
    )
    units = [_unit(1, annotation, chapter_id="v1-p2-un-amour-de-swann")]

    comparisons = scoring_v2_build.build_comparisons(units, "prestige", registry=registry)
    name_matches, name_dropped = scoring_v2_build.view_matches(comparisons, "name")
    person_matches, person_dropped = scoring_v2_build.view_matches(comparisons, "person")

    # the "le peintre" / Swann pair is vacuous (only Elstir moved) and is
    # not emitted; the two Elstir pairings survive.
    assert len(comparisons) == 2
    assert name_dropped == 0 and len(name_matches) == 2
    # "le peintre" vs Elstir is one man against himself once the person
    # view merges them.
    assert person_dropped == 1 and len(person_matches) == 1
    assert all(match["character_a"] != match["character_b"] for match in person_matches)


def test_fit_view_ranks_the_corpus_and_carries_the_v2_weights(registry):
    units = _corpus()
    comparisons = scoring_v2_build.build_comparisons(units, "advantage", registry=registry)
    matches, _dropped = scoring_v2_build.view_matches(comparisons, "name")
    readings = scoring_v2_build.build_readings(units, "advantage", registry=registry)

    ratings = scoring_v2_build.fit_view(matches, readings, "name", "advantage", w2_elo=15.0)

    order = [row["character"] for row in ratings["characters"]]
    assert order[0] == "Swann" and order[-1] == "Saniette"
    assert ratings["comparison_count"] == len(matches)
    # Every comparison here is weighted by the annotator's confidence, so
    # the mean weight must be below the unweighted 1.0 the games would
    # otherwise carry.
    assert 0.0 < ratings["mean_weight"] < 1.0
    assert ratings["characters"][0]["labels"]["positive"] == 6
    assert ratings["characters"][-1]["labels"]["negative"] == 6


def test_staging_writes_everything_under_one_directory(tmp_path, registry):
    # v2 is staged, not adopted: a build must never touch an artifact
    # outside its own output directory.
    units = _corpus()
    comparisons = scoring_v2_build.build_comparisons(units, "advantage", registry=registry)
    matches, _dropped = scoring_v2_build.view_matches(comparisons, "name")
    readings = scoring_v2_build.build_readings(units, "advantage", registry=registry)
    ratings = scoring_v2_build.fit_view(matches, readings, "name", "advantage", w2_elo=15.0)
    build = {
        "manifest": {"corpus": "foundation", "lenses": ["advantage"], "views": ["name"]},
        "comparisons": {"advantage": comparisons},
        "ratings": {("advantage", "name"): ratings},
        "timelines": {
            ("advantage", "name"): scoring_v2_build.build_timeline(ratings, units, readings, "name")
        },
        "corpus_summary": scoring_v2_build.build_corpus_summary(
            {lens: ratings for lens in scoring_v2.SCORING_V2_LENS_ORDER},
            {lens: readings for lens in scoring_v2.SCORING_V2_LENS_ORDER},
        ),
    }

    staged = tmp_path / "scoring-v2"
    written = scoring_v2_build.write_scoring_v2_artifacts(build, output_dir=staged)

    assert all(path.startswith(str(staged)) for path in written)
    assert (staged / "scoring-v2-advantage-name-view-ratings.json").exists()
    assert (staged / "scoring-v2-advantage-comparisons.json").exists()
    assert (staged / "scoring-v2-corpus-summary.md").exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["scoring-v2"]


def test_corpus_summary_reports_intensity_per_appearance_not_sums(registry):
    units = _corpus()
    ratings_by_lens = {}
    readings_by_lens = {}
    for lens in scoring_v2.SCORING_V2_LENS_ORDER:
        comparisons = scoring_v2_build.build_comparisons(units, lens, registry=registry)
        matches, _dropped = scoring_v2_build.view_matches(comparisons, "name")
        readings_by_lens[lens] = scoring_v2_build.build_readings(units, lens, registry=registry)
        ratings_by_lens[lens] = scoring_v2_build.fit_view(
            matches, readings_by_lens[lens], "name", lens, w2_elo=15.0
        )

    summary = scoring_v2_build.build_corpus_summary(ratings_by_lens, readings_by_lens)
    rows = {row["character"]: row for row in summary["characters"]}

    swann = rows["Swann"]["lenses"]["advantage"]
    assert swann["appearances"] == 6
    # Six identical units: the mean is one unit's movement, not six of them.
    assert swann["mean_movement"] == pytest.approx(1.8)
    assert swann["mean_absolute_movement"] == pytest.approx(1.8)
    assert rows["Bloch"]["lenses"]["prestige"]["mean_movement"] == 0.0


def test_vacuous_pairs_are_skipped_but_baselines_and_cancelling_movers_stay():
    from proust import scoring_v2

    annotation = {
        "unit_id": "v1-p1-combray#p-1-p-5",
        "characters_present": [
            {"canonical_name": n, "surface_forms": [n], "presence_type": "explicit",
             "presence_confidence": 0.9}
            for n in ("Swann", "Odette", "docteur Cottard", "Mme Cottard")
        ],
        "appraisal_events": [
            {"event_id": "E1", "source": "narrator", "target": "Swann",
             "type": "narrated_elevation", "polarity": "positive", "narrative_stance": "endorsed",
             "confidence": 0.9, "evidence": "x", "explanation": "x"},
        ],
        "status_effects": [
            # Swann moves in advantage
            {"character": "Swann", "dimension": "general_appraisal", "delta": 1,
             "based_on_events": ["E1"], "confidence": 0.9, "explanation": "x"},
            # Odette's advantage effects cancel to zero movement but she participated
            {"character": "Odette", "dimension": "emotional_position", "delta": 1,
             "based_on_events": ["E1"], "confidence": 0.5, "explanation": "x"},
            {"character": "Odette", "dimension": "emotional_position", "delta": -1,
             "based_on_events": ["E1"], "confidence": 0.5, "explanation": "x"},
        ],
        "ambiguities": [],
    }

    comps = scoring_v2.unit_comparisons(annotation, "advantage")
    pairs = {(c["character_a"], c["character_b"]) for c in comps}
    # Cottard vs Mme Cottard: neither moved in advantage -> vacuous, skipped
    assert ("Mme Cottard", "docteur Cottard") not in pairs
    # bystander vs mover stays
    assert ("Swann", "docteur Cottard") in pairs
    # cancelled-to-zero participant stays comparable, even against a bystander
    assert ("Mme Cottard", "Odette") in pairs
    # prestige: nobody has prestige effects -> no comparisons at all
    assert scoring_v2.unit_comparisons(annotation, "prestige") == []
