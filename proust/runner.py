import argparse
from collections import defaultdict
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from .annotation import (
    DEFAULT_STARTER_ALIAS_MAP,
    PROMPT_PATH,
    STARTER_UNITS,
    build_annotation_unit,
    load_prompt_template,
    render_prompt_input,
)

ANNOTATION_TOP_LEVEL_KEYS = {
    "unit_id",
    "characters_present",
    "appraisal_events",
    "status_effects",
    "ambiguities",
}

CHARACTER_PRESENT_KEYS = {
    "canonical_name",
    "surface_forms",
    "presence_type",
    "presence_confidence",
}

APPRAISAL_EVENT_KEYS = {
    "event_id",
    "source",
    "target",
    "type",
    "polarity",
    "narrative_stance",
    "confidence",
    "evidence",
    "explanation",
}

STATUS_EFFECT_KEYS = {
    "character",
    "dimension",
    "delta",
    "based_on_events",
    "confidence",
    "explanation",
}

ALLOWED_PRESENCE_TYPES = {"explicit", "implicit"}
ALLOWED_EVENT_TYPES = {
    "praise",
    "blame",
    "admiration",
    "contempt",
    "ridicule",
    "preference",
    "favorable_comparison",
    "unfavorable_comparison",
    "deference",
    "snub",
    "exclusion",
    "humiliation",
    "prestige_association",
    "discredit_association",
    "rhetorical_authority",
    "emotional_leverage",
    "narrated_elevation",
    "narrated_diminishment",
    "other",
}
ALLOWED_POLARITIES = {"positive", "negative", "mixed"}
ALLOWED_NARRATIVE_STANCES = {"endorsed", "neutral_report", "ironized", "uncertain"}
ALLOWED_STATUS_DIMENSIONS = {
    "general_appraisal",
    "social_status",
    "rhetorical_position",
    "emotional_position",
    "inclusion_exclusion",
}
ALLOWED_STATUS_DELTAS = {-2, -1, 0, 1, 2}
ALLOWED_EVENT_SOURCES = {"narrator", "collective_social_voice", "unknown"}

EVENT_TYPE_REDUCTION_MAP = {
    "praise": "admiration",
    "contempt": "blame",
    "ridicule": "blame",
    "preference": "admiration",
    "favorable_comparison": "admiration",
    "unfavorable_comparison": "blame",
    "deference": "admiration",
    "exclusion": "snub",
    "humiliation": "narrated_diminishment",
    "rhetorical_authority": "other",
    "emotional_leverage": "other",
    "inclusion": "prestige_association",
}
PREFERRED_EVENT_TYPE_ORDER = {
    "narrated_diminishment": 0,
    "narrated_elevation": 1,
    "discredit_association": 2,
    "prestige_association": 3,
    "snub": 4,
    "blame": 5,
    "admiration": 6,
    "praise": 7,
    "other": 8,
}
PREFERRED_STATUS_DIMENSION_ORDER = {
    "social_status": 0,
    "general_appraisal": 1,
    "inclusion_exclusion": 2,
    "rhetorical_position": 3,
    "emotional_position": 4,
}
LOCAL_OUTCOME_EVENT_WEIGHTS = {
    "narrated_elevation": 1.0,
    "prestige_association": 1.0,
    "admiration": 0.9,
    "snub": 1.1,
    "discredit_association": 1.0,
    "narrated_diminishment": 1.0,
    "blame": 0.9,
    "other": 0.6,
}
LOCAL_OUTCOME_STATUS_WEIGHTS = {
    "social_status": 1.3,
    "inclusion_exclusion": 1.2,
    "general_appraisal": 1.0,
    "rhetorical_position": 0.8,
    "emotional_position": 0.6,
}
LOCAL_OUTCOME_STANCE_MULTIPLIERS = {
    "endorsed": 1.0,
    "neutral_report": 0.9,
    "ironized": 0.7,
    "uncertain": 0.5,
}
LOCAL_OUTCOME_LABEL_THRESHOLDS = {
    "win": 0.75,
    "loss": -0.75,
}
LOCAL_OUTCOME_AMBIGUITY_PENALTY = 0.4
PRESTIGE_OUTCOME_EVENT_WEIGHTS = {
    "narrated_elevation": 0.9,
    "prestige_association": 1.4,
    "admiration": 0.7,
    "snub": 0.6,
    "discredit_association": 1.2,
    "narrated_diminishment": 0.8,
    "blame": 0.7,
    "other": 0.5,
}
PRESTIGE_OUTCOME_STATUS_WEIGHTS = {
    "social_status": 1.6,
    "inclusion_exclusion": 0.6,
    "general_appraisal": 0.8,
    "rhetorical_position": 0.5,
    "emotional_position": 0.4,
}
INCLUSION_OUTCOME_EVENT_WEIGHTS = {
    "narrated_elevation": 0.7,
    "prestige_association": 0.6,
    "admiration": 0.6,
    "snub": 1.5,
    "discredit_association": 0.8,
    "narrated_diminishment": 0.9,
    "blame": 0.7,
    "other": 0.5,
}
INCLUSION_OUTCOME_STATUS_WEIGHTS = {
    "social_status": 0.7,
    "inclusion_exclusion": 1.7,
    "general_appraisal": 0.8,
    "rhetorical_position": 0.6,
    "emotional_position": 0.5,
}
SCORING_LENS_CONFIGS = {
    "local": {
        "scoring_version": "local_outcome_v1",
        "event_weights": LOCAL_OUTCOME_EVENT_WEIGHTS,
        "status_weights": LOCAL_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": LOCAL_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": LOCAL_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": LOCAL_OUTCOME_AMBIGUITY_PENALTY,
    },
    "prestige": {
        "scoring_version": "prestige_outcome_v1",
        "event_weights": PRESTIGE_OUTCOME_EVENT_WEIGHTS,
        "status_weights": PRESTIGE_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": LOCAL_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": LOCAL_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": LOCAL_OUTCOME_AMBIGUITY_PENALTY,
    },
    "inclusion": {
        "scoring_version": "inclusion_outcome_v1",
        "event_weights": INCLUSION_OUTCOME_EVENT_WEIGHTS,
        "status_weights": INCLUSION_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": LOCAL_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": LOCAL_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": LOCAL_OUTCOME_AMBIGUITY_PENALTY,
    },
}


class RunManifestNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class AnnotationRunManifest:
    run_id: str
    created_at: str
    prompt_path: str
    unit_ids: list[str]
    directories: dict[str, str]
    alias_map: dict
    notes: str = ""
    derived_from: dict | None = None
    automation: dict | None = None
    benchmark: dict | None = None


def _unit_filename(unit_id):
    return f"{unit_id}.json"


def _prompt_filename(unit_id):
    return f"{unit_id}.txt"


def _raw_filename(unit_id):
    return f"{unit_id}.txt"


def _annotation_filename(unit_id):
    return f"{unit_id}.json"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_run_manifest(run_dir):
    run_path = Path(run_dir)
    manifest_path = run_path / "run.json"
    if not manifest_path.exists():
        raise RunManifestNotFoundError(
            f'Run directory "{run_path}" does not contain a run.json manifest at "{manifest_path}".'
        )
    return _read_json(manifest_path)


def _ensure_run_directories(run_dir):
    directories = {
        "units": run_dir / "units",
        "prompts": run_dir / "prompts",
        "raw": run_dir / "raw",
        "annotations": run_dir / "annotations",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _copy_run_file_tree(source_dir, destination_dir, suffix):
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(source_path.glob(f"*{suffix}")):
        shutil.copy2(source_file, destination_path / source_file.name)


def _write_run_manifest(run_dir, manifest):
    (Path(run_dir) / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _check_confidence(value, field_name, errors):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{field_name} must be a number between 0.0 and 1.0.")
        return
    if not 0.0 <= float(value) <= 1.0:
        errors.append(f"{field_name} must be between 0.0 and 1.0.")


def _check_string_list(value, field_name, errors):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field_name} must be a list of strings.")


def validate_annotation_result(annotation, expected_unit_id=None):
    errors = []

    if not isinstance(annotation, dict):
        return ["annotation must be a JSON object."]

    annotation_keys = set(annotation)
    missing_top_level = sorted(ANNOTATION_TOP_LEVEL_KEYS - annotation_keys)
    extra_top_level = sorted(annotation_keys - ANNOTATION_TOP_LEVEL_KEYS)
    if missing_top_level:
        errors.append(f"missing top-level keys: {', '.join(missing_top_level)}")
    if extra_top_level:
        errors.append(f"unexpected top-level keys: {', '.join(extra_top_level)}")

    unit_id = annotation.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id:
        errors.append("unit_id must be a non-empty string.")
    elif expected_unit_id and unit_id != expected_unit_id:
        errors.append(f'unit_id "{unit_id}" does not match expected unit id "{expected_unit_id}".')

    characters_present = annotation.get("characters_present")
    character_names = set()
    if not isinstance(characters_present, list):
        errors.append("characters_present must be a list.")
    else:
        for index, character in enumerate(characters_present):
            prefix = f"characters_present[{index}]"
            if not isinstance(character, dict):
                errors.append(f"{prefix} must be an object.")
                continue
            keys = set(character)
            missing_keys = sorted(CHARACTER_PRESENT_KEYS - keys)
            extra_keys = sorted(keys - CHARACTER_PRESENT_KEYS)
            if missing_keys:
                errors.append(f"{prefix} missing keys: {', '.join(missing_keys)}")
            if extra_keys:
                errors.append(f"{prefix} unexpected keys: {', '.join(extra_keys)}")

            canonical_name = character.get("canonical_name")
            if not isinstance(canonical_name, str) or not canonical_name:
                errors.append(f"{prefix}.canonical_name must be a non-empty string.")
            else:
                character_names.add(canonical_name)

            _check_string_list(character.get("surface_forms"), f"{prefix}.surface_forms", errors)

            presence_type = character.get("presence_type")
            if presence_type not in ALLOWED_PRESENCE_TYPES:
                errors.append(
                    f"{prefix}.presence_type must be one of: {', '.join(sorted(ALLOWED_PRESENCE_TYPES))}."
                )
            _check_confidence(character.get("presence_confidence"), f"{prefix}.presence_confidence", errors)

    appraisal_events = annotation.get("appraisal_events")
    event_ids = set()
    if not isinstance(appraisal_events, list):
        errors.append("appraisal_events must be a list.")
    else:
        for index, event in enumerate(appraisal_events):
            prefix = f"appraisal_events[{index}]"
            if not isinstance(event, dict):
                errors.append(f"{prefix} must be an object.")
                continue
            keys = set(event)
            missing_keys = sorted(APPRAISAL_EVENT_KEYS - keys)
            extra_keys = sorted(keys - APPRAISAL_EVENT_KEYS)
            if missing_keys:
                errors.append(f"{prefix} missing keys: {', '.join(missing_keys)}")
            if extra_keys:
                errors.append(f"{prefix} unexpected keys: {', '.join(extra_keys)}")

            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                errors.append(f"{prefix}.event_id must be a non-empty string.")
            elif event_id in event_ids:
                errors.append(f'{prefix}.event_id "{event_id}" is duplicated.')
            else:
                event_ids.add(event_id)

            source = event.get("source")
            if not isinstance(source, str) or not source:
                errors.append(f"{prefix}.source must be a non-empty string.")
            elif source not in ALLOWED_EVENT_SOURCES and source not in character_names:
                errors.append(
                    f"{prefix}.source must be narrator, collective_social_voice, unknown, or a character in characters_present."
                )

            target = event.get("target")
            if not isinstance(target, str) or not target:
                errors.append(f"{prefix}.target must be a non-empty string.")
            elif target not in character_names:
                errors.append(f'{prefix}.target "{target}" must appear in characters_present.')

            event_type = event.get("type")
            if event_type not in ALLOWED_EVENT_TYPES:
                errors.append(f"{prefix}.type must be one of the prompt schema event types.")

            polarity = event.get("polarity")
            if polarity not in ALLOWED_POLARITIES:
                errors.append(f"{prefix}.polarity must be one of: {', '.join(sorted(ALLOWED_POLARITIES))}.")

            narrative_stance = event.get("narrative_stance")
            if narrative_stance not in ALLOWED_NARRATIVE_STANCES:
                errors.append(
                    f"{prefix}.narrative_stance must be one of: {', '.join(sorted(ALLOWED_NARRATIVE_STANCES))}."
                )

            _check_confidence(event.get("confidence"), f"{prefix}.confidence", errors)

            for field_name in ("evidence", "explanation"):
                value = event.get(field_name)
                if not isinstance(value, str) or not value:
                    errors.append(f"{prefix}.{field_name} must be a non-empty string.")

    status_effects = annotation.get("status_effects")
    if not isinstance(status_effects, list):
        errors.append("status_effects must be a list.")
    else:
        for index, effect in enumerate(status_effects):
            prefix = f"status_effects[{index}]"
            if not isinstance(effect, dict):
                errors.append(f"{prefix} must be an object.")
                continue
            keys = set(effect)
            missing_keys = sorted(STATUS_EFFECT_KEYS - keys)
            extra_keys = sorted(keys - STATUS_EFFECT_KEYS)
            if missing_keys:
                errors.append(f"{prefix} missing keys: {', '.join(missing_keys)}")
            if extra_keys:
                errors.append(f"{prefix} unexpected keys: {', '.join(extra_keys)}")

            character = effect.get("character")
            if not isinstance(character, str) or not character:
                errors.append(f"{prefix}.character must be a non-empty string.")
            elif character not in character_names:
                errors.append(f'{prefix}.character "{character}" must appear in characters_present.')

            dimension = effect.get("dimension")
            if dimension not in ALLOWED_STATUS_DIMENSIONS:
                errors.append(f"{prefix}.dimension must be one of the prompt schema status dimensions.")

            delta = effect.get("delta")
            if delta not in ALLOWED_STATUS_DELTAS:
                errors.append(f"{prefix}.delta must be one of: -2, -1, 0, 1, 2.")

            based_on_events = effect.get("based_on_events")
            if not isinstance(based_on_events, list) or not based_on_events:
                errors.append(f"{prefix}.based_on_events must be a non-empty list of event ids.")
            elif not all(isinstance(event_id, str) for event_id in based_on_events):
                errors.append(f"{prefix}.based_on_events must be a non-empty list of event ids.")
            else:
                unknown_event_ids = sorted(set(based_on_events) - event_ids)
                if unknown_event_ids:
                    errors.append(
                        f"{prefix}.based_on_events references unknown event ids: {', '.join(unknown_event_ids)}"
                    )

            _check_confidence(effect.get("confidence"), f"{prefix}.confidence", errors)

            explanation = effect.get("explanation")
            if not isinstance(explanation, str) or not explanation:
                errors.append(f"{prefix}.explanation must be a non-empty string.")

    ambiguities = annotation.get("ambiguities")
    if not isinstance(ambiguities, list) or not all(isinstance(item, str) for item in ambiguities):
        errors.append("ambiguities must be a list of strings.")

    return errors


def _strip_code_fence(text):
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_annotation_response_text(raw_text, expected_unit_id=None):
    normalized_text = _strip_code_fence(raw_text)
    normalized_text = re.sub(r'(:\s*)\+(\d+)', r"\1\2", normalized_text)
    annotation = json.loads(normalized_text)
    if expected_unit_id and isinstance(annotation, dict) and "unit_id" not in annotation:
        annotation["unit_id"] = expected_unit_id
    return annotation


def _reduced_event_type(event_type):
    return EVENT_TYPE_REDUCTION_MAP.get(event_type, event_type)


def _event_priority_score(event):
    score = float(event.get("confidence", 0.0))
    source = event.get("source")
    stance = event.get("narrative_stance")
    event_type = _reduced_event_type(event.get("type"))

    if source == "narrator":
        score += 0.35
    elif source == "collective_social_voice":
        score += 0.15

    if stance == "endorsed":
        score += 0.3
    elif stance == "ironized":
        score += 0.15
    elif stance == "uncertain":
        score -= 0.05

    score -= 0.02 * PREFERRED_EVENT_TYPE_ORDER.get(event_type, 9)
    return score


def _is_narrator_only_positive(event):
    return event.get("polarity") == "positive" and event.get("source") == "narrator"


def _select_dominant_negative(events):
    social_negative_types = {"snub", "discredit_association"}
    social_negatives = [
        event
        for event in events
        if event.get("polarity") == "negative"
        and event.get("source") != "narrator"
        and event.get("type") in social_negative_types
    ]
    if social_negatives:
        return max(social_negatives, key=_event_priority_score)

    narrated_negatives = [
        event
        for event in events
        if event.get("polarity") == "negative" and event.get("type") == "narrated_diminishment"
    ]
    if narrated_negatives:
        return max(narrated_negatives, key=_event_priority_score)

    negatives = [event for event in events if event.get("polarity") == "negative"]
    if negatives:
        return max(negatives, key=_event_priority_score)
    return None


def _select_dominant_positive(events):
    direct_positive_types = {"admiration", "prestige_association"}
    direct_positives = [
        event
        for event in events
        if event.get("polarity") == "positive"
        and event.get("source") != "narrator"
        and event.get("source") != event.get("target")
        and event.get("type") in direct_positive_types
    ]
    if direct_positives:
        return max(direct_positives, key=_event_priority_score)

    narrated_positives = [
        event
        for event in events
        if event.get("polarity") == "positive" and event.get("type") == "narrated_elevation"
    ]
    if narrated_positives:
        return max(narrated_positives, key=_event_priority_score)

    positives = [event for event in events if event.get("polarity") == "positive"]
    if positives:
        return max(positives, key=_event_priority_score)
    return None


def _raw_target_polarities(events):
    target_to_polarities = {}
    for event in events:
        target = event.get("target")
        polarity = event.get("polarity")
        if not target or polarity not in {"positive", "negative"}:
            continue
        target_to_polarities.setdefault(target, set()).add(polarity)
    return target_to_polarities


def _has_mixed_target_pair(events, target):
    polarities = set()
    for event in events:
        if event.get("target") != target:
            continue
        if event.get("source") == target:
            continue
        polarity = event.get("polarity")
        if polarity in {"positive", "negative"}:
            polarities.add(polarity)
    return polarities == {"positive", "negative"}


def _status_priority_score(effect):
    score = abs(int(effect.get("delta", 0))) + float(effect.get("confidence", 0.0))
    score -= 0.02 * PREFERRED_STATUS_DIMENSION_ORDER.get(effect.get("dimension"), 9)
    return score


def reduce_annotation_result(annotation, expected_unit_id=None):
    reduced = json.loads(json.dumps(annotation))
    if expected_unit_id and "unit_id" not in reduced:
        reduced["unit_id"] = expected_unit_id

    characters_present = reduced.get("characters_present") or []
    events = reduced.get("appraisal_events") or []
    effects = reduced.get("status_effects") or []

    normalized_events = []
    for event in events:
        normalized_event = dict(event)
        normalized_event["type"] = _reduced_event_type(normalized_event.get("type"))
        normalized_events.append(normalized_event)

    raw_target_polarities = _raw_target_polarities(normalized_events)

    deduped_events = {}
    for event in normalized_events:
        key = (
            event.get("target"),
            event.get("source"),
            event.get("type"),
            event.get("polarity"),
        )
        existing = deduped_events.get(key)
        if existing is None or _event_priority_score(event) > _event_priority_score(existing):
            deduped_events[key] = event

    events_by_target = {}
    for event in deduped_events.values():
        events_by_target.setdefault(event.get("target"), []).append(event)

    candidate_pairs = []
    for target, target_events in events_by_target.items():
        dominant_negative = _select_dominant_negative(target_events)
        dominant_positive = _select_dominant_positive(target_events)

        selected_for_target = []
        if dominant_negative and dominant_positive:
            if _is_narrator_only_positive(dominant_positive) and dominant_negative.get("source") != "narrator":
                selected_for_target = [dominant_negative]
            else:
                selected_for_target = [dominant_negative, dominant_positive]
        elif dominant_negative:
            selected_for_target = [dominant_negative]
        elif dominant_positive:
            selected_for_target = [dominant_positive]

        if not selected_for_target:
            continue

        combined_score = sum(_event_priority_score(event) for event in selected_for_target)
        if (
            len(selected_for_target) == 2
            and {event.get("polarity") for event in selected_for_target} == {"positive", "negative"}
            and any(event.get("source") not in ALLOWED_EVENT_SOURCES for event in selected_for_target)
        ):
            combined_score += 0.4
        candidate_pairs.append((combined_score, target, selected_for_target))

    if candidate_pairs:
        mixed_targets = {
            target
            for target, polarities in raw_target_polarities.items()
            if polarities == {"positive", "negative"}
        }
        if mixed_targets:
            candidate_pairs = [
                (
                    score + (0.6 if target in mixed_targets else 0.0),
                    target,
                    events,
                )
                for score, target, events in candidate_pairs
            ]
        _, _, selected_events = max(candidate_pairs, key=lambda item: item[0])
    else:
        selected_events = sorted(
            deduped_events.values(),
            key=lambda event: _event_priority_score(event),
            reverse=True,
        )[:2]

    selected_events = sorted(selected_events, key=lambda event: _event_priority_score(event), reverse=True)

    if len(selected_events) == 2:
        positive_events = [event for event in selected_events if event.get("polarity") == "positive"]
        negative_events = [event for event in selected_events if event.get("polarity") == "negative"]
        if (
            len(positive_events) == 1
            and len(negative_events) == 1
            and positive_events[0].get("source") == positive_events[0].get("target")
            and positive_events[0].get("narrative_stance") == "ironized"
            and negative_events[0].get("source") == "narrator"
        ):
            selected_events = negative_events

    if (
        len(selected_events) == 2
        and selected_events[0].get("target") == selected_events[1].get("target")
        and selected_events[0].get("polarity") == selected_events[1].get("polarity") == "positive"
        and {selected_events[0].get("type"), selected_events[1].get("type")} == {"narrated_elevation", "prestige_association"}
    ):
        selected_events = [
            max(selected_events, key=lambda event: (event.get("type") == "narrated_elevation", _event_priority_score(event)))
        ]

    event_id_map = {}
    for index, event in enumerate(selected_events, start=1):
        original_event_id = event.get("event_id")
        new_event_id = f"E{index}"
        event_id_map[original_event_id] = new_event_id
        event["event_id"] = new_event_id

    selected_character_names = set()
    for event in selected_events:
        selected_character_names.add(event["target"])
        if event.get("source") not in ALLOWED_EVENT_SOURCES:
            selected_character_names.add(event["source"])

    filtered_characters = [
        character
        for character in characters_present
        if character.get("canonical_name") in selected_character_names
    ]

    if not filtered_characters and characters_present:
        filtered_characters = characters_present[:1]

    character_names = {character.get("canonical_name") for character in filtered_characters}
    selected_events = [
        event
        for event in selected_events
        if event.get("target") in character_names
        and (event.get("source") in ALLOWED_EVENT_SOURCES or event.get("source") in character_names)
    ]

    valid_event_ids = {event["event_id"] for event in selected_events}
    filtered_effects = []
    for effect in effects:
        if effect.get("character") not in character_names:
            continue
        based_on_events = [event_id_map.get(event_id) for event_id in effect.get("based_on_events", [])]
        based_on_events = [event_id for event_id in based_on_events if event_id in valid_event_ids]
        if not based_on_events:
            continue
        normalized_effect = dict(effect)
        normalized_effect["based_on_events"] = based_on_events
        filtered_effects.append(normalized_effect)

    best_effects_by_key = {}
    for effect in filtered_effects:
        key = (effect.get("character"), effect.get("dimension"))
        existing = best_effects_by_key.get(key)
        if existing is None or _status_priority_score(effect) > _status_priority_score(existing):
            best_effects_by_key[key] = effect

    per_character_effects = {}
    for effect in best_effects_by_key.values():
        per_character_effects.setdefault(effect["character"], []).append(effect)

    selected_effects = []
    for character, character_effects in per_character_effects.items():
        del character
        chosen_effects = sorted(
            character_effects,
            key=lambda effect: _status_priority_score(effect),
            reverse=True,
        )[:2]
        selected_effects.extend(chosen_effects)

    selected_effects = sorted(
        selected_effects,
        key=lambda effect: (
            effect["character"],
            PREFERRED_STATUS_DIMENSION_ORDER.get(effect["dimension"], 9),
            -_status_priority_score(effect),
        ),
    )

    focal_target = None
    if selected_events:
        targets = {event["target"] for event in selected_events}
        if len(targets) == 1:
            focal_target = next(iter(targets))

    if len(selected_events) == 1:
        event = selected_events[0]
        focal_target = event["target"]
        effects_for_target = [effect for effect in selected_effects if effect["character"] == focal_target]

        if event["type"] == "snub":
            preferred_dimensions = ["general_appraisal", "inclusion_exclusion"]
        elif event["type"] == "narrated_elevation":
            preferred_dimensions = ["social_status"]
        elif event["type"] == "narrated_diminishment":
            preferred_dimensions = ["general_appraisal", "social_status"]
        else:
            preferred_dimensions = []

            if preferred_dimensions:
                chosen = []
                for dimension in preferred_dimensions:
                    matches = [effect for effect in effects_for_target if effect["dimension"] == dimension]
                    if matches:
                        chosen.append(max(matches, key=_status_priority_score))
                if event["type"] == "snub" and len(chosen) < 2:
                    existing_dimensions = {effect["dimension"] for effect in chosen}
                    fallback_dimensions = ["general_appraisal", "inclusion_exclusion"]
                    for dimension in fallback_dimensions:
                        if dimension in existing_dimensions:
                            continue
                        chosen.append(
                            {
                                "character": focal_target,
                                "dimension": dimension,
                                "delta": -1,
                                "based_on_events": [event["event_id"]],
                                "confidence": float(event.get("confidence", 0.0)),
                                "explanation": event.get("explanation", ""),
                            }
                        )
                        if len(chosen) == 2:
                            break

                selected_effects = chosen or effects_for_target[:2]

            if event["type"] == "snub":
                normalized_snub_effects = []
                for dimension in ("general_appraisal", "inclusion_exclusion"):
                    matching_effect = next((effect for effect in selected_effects if effect["dimension"] == dimension), None)
                    if matching_effect is None:
                        matching_effect = {
                            "character": focal_target,
                            "dimension": dimension,
                            "delta": -1,
                            "based_on_events": [event["event_id"]],
                            "confidence": float(event.get("confidence", 0.0)),
                            "explanation": event.get("explanation", ""),
                        }
                    if matching_effect["delta"] < -1:
                        matching_effect["delta"] = -1
                    if matching_effect["delta"] > -1:
                        matching_effect["delta"] = -1
                    normalized_snub_effects.append(matching_effect)
                selected_effects = normalized_snub_effects
            if event["type"] == "narrated_elevation":
                for effect in selected_effects:
                    if effect["dimension"] == "social_status" and effect["delta"] < 2:
                        effect["delta"] = max(effect["delta"], 1)

    elif len(selected_events) == 2 and focal_target is not None:
        polarities = {event["polarity"] for event in selected_events}
        if polarities == {"positive", "negative"}:
            selected_effects = [effect for effect in selected_effects if effect["character"] == focal_target]

    mixed_targets = [target for target in raw_target_polarities if _has_mixed_target_pair(normalized_events, target)]
    if mixed_targets and not any(event["target"] in mixed_targets for event in selected_events):
        best_target = mixed_targets[0]
        target_events = events_by_target.get(best_target, [])
        dominant_negative = _select_dominant_negative(target_events)
        dominant_positive = _select_dominant_positive(target_events)
        forced_events = [event for event in [dominant_negative, dominant_positive] if event is not None]
        if len(forced_events) == 2:
            selected_events = sorted(forced_events, key=lambda event: _event_priority_score(event), reverse=True)
            event_id_map = {}
            for index, event in enumerate(selected_events, start=1):
                original_event_id = event.get("event_id")
                new_event_id = f"E{index}"
                event_id_map[original_event_id] = new_event_id
                event["event_id"] = new_event_id

            selected_character_names = {best_target}
            for event in selected_events:
                if event.get("source") not in ALLOWED_EVENT_SOURCES:
                    selected_character_names.add(event["source"])

            filtered_characters = [
                character
                for character in characters_present
                if character.get("canonical_name") in selected_character_names
            ]
            character_names = {character.get("canonical_name") for character in filtered_characters}
            selected_effects = []
            for effect in effects:
                if effect.get("character") != best_target:
                    continue
                based_on_events = [event_id_map.get(event_id) for event_id in effect.get("based_on_events", [])]
                based_on_events = [event_id for event_id in based_on_events if event_id in {"E1", "E2"}]
                if not based_on_events:
                    continue
                normalized_effect = dict(effect)
                normalized_effect["based_on_events"] = based_on_events
                selected_effects.append(normalized_effect)

    kept_ambiguities = []
    if any(event.get("narrative_stance") == "uncertain" for event in selected_events):
        ambiguities = reduced.get("ambiguities") or []
        if ambiguities:
            kept_ambiguities = [ambiguities[0]]

    reduced["characters_present"] = filtered_characters
    reduced["appraisal_events"] = selected_events
    reduced["status_effects"] = selected_effects
    reduced["ambiguities"] = kept_ambiguities
    return reduced


def extract_response_output_text(response_payload):
    if isinstance(response_payload.get("output_text"), str) and response_payload["output_text"]:
        return response_payload["output_text"]

    text_chunks = []
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                text_chunks.append(content_item["text"])

    return "\n".join(chunk for chunk in text_chunks if chunk).strip()


def get_run_status(run_dir):
    run_path = Path(run_dir)
    manifest = _read_run_manifest(run_path)
    directories = {name: Path(path) for name, path in manifest["directories"].items()}
    unit_statuses = []

    for unit_id in manifest["unit_ids"]:
        unit_path = directories["units"] / _unit_filename(unit_id)
        prompt_path = directories["prompts"] / _prompt_filename(unit_id)
        raw_path = directories["raw"] / _raw_filename(unit_id)
        annotation_path = directories["annotations"] / _annotation_filename(unit_id)

        annotation_errors = []
        if annotation_path.exists():
            annotation_errors = validate_annotation_result(
                _read_json(annotation_path),
                expected_unit_id=unit_id,
            )

        unit_statuses.append(
            {
                "unit_id": unit_id,
                "unit_exists": unit_path.exists(),
                "prompt_exists": prompt_path.exists(),
                "raw_exists": raw_path.exists(),
                "annotation_exists": annotation_path.exists(),
                "annotation_valid": annotation_path.exists() and not annotation_errors,
                "annotation_errors": annotation_errors,
                "review_state": "reviewed"
                if annotation_path.exists() and not annotation_errors
                else "pending",
            }
        )

    summary = {
        "run_id": manifest["run_id"],
        "unit_count": len(unit_statuses),
        "unit_file_count": sum(1 for status in unit_statuses if status["unit_exists"]),
        "prompt_file_count": sum(1 for status in unit_statuses if status["prompt_exists"]),
        "raw_file_count": sum(1 for status in unit_statuses if status["raw_exists"]),
        "annotation_file_count": sum(1 for status in unit_statuses if status["annotation_exists"]),
        "valid_annotation_count": sum(1 for status in unit_statuses if status["annotation_valid"]),
        "reviewed_unit_count": sum(1 for status in unit_statuses if status["review_state"] == "reviewed"),
        "pending_unit_count": sum(1 for status in unit_statuses if status["review_state"] != "reviewed"),
        "benchmark_ready": all(
            status["unit_exists"] and status["prompt_exists"] and status["annotation_valid"]
            for status in unit_statuses
        ),
    }
    return {"manifest": manifest, "summary": summary, "units": unit_statuses}


def summarize_run_annotations(run_dir):
    status = get_run_status(run_dir)
    manifest = status["manifest"]
    annotation_dir = Path(manifest["directories"]["annotations"])
    summary = {
        "run_id": manifest["run_id"],
        "unit_count": len(manifest["unit_ids"]),
        "valid_annotation_count": 0,
        "event_type_counts": {},
        "event_polarity_counts": {"positive": 0, "negative": 0, "mixed": 0, "neutral": 0},
        "event_source_counts": {},
        "event_target_counts": {},
        "status_dimension_totals": {},
        "character_status_totals": {},
    }

    for unit_id in manifest["unit_ids"]:
        annotation_path = annotation_dir / _annotation_filename(unit_id)
        if not annotation_path.exists():
            continue

        annotation = _read_json(annotation_path)
        errors = validate_annotation_result(annotation, expected_unit_id=unit_id)
        if errors:
            continue

        summary["valid_annotation_count"] += 1

        for event in annotation["appraisal_events"]:
            event_type = event["type"]
            polarity = event["polarity"]
            source = event["source"]
            target = event["target"]

            summary["event_type_counts"][event_type] = summary["event_type_counts"].get(event_type, 0) + 1
            summary["event_polarity_counts"][polarity] = summary["event_polarity_counts"].get(polarity, 0) + 1
            summary["event_source_counts"][source] = summary["event_source_counts"].get(source, 0) + 1
            summary["event_target_counts"][target] = summary["event_target_counts"].get(target, 0) + 1

        for effect in annotation["status_effects"]:
            character = effect["character"]
            dimension = effect["dimension"]
            delta = effect["delta"]

            summary["status_dimension_totals"][dimension] = (
                summary["status_dimension_totals"].get(dimension, 0) + delta
            )
            character_totals = summary["character_status_totals"].setdefault(character, {})
            character_totals[dimension] = character_totals.get(dimension, 0) + delta

    return summary


def wait_for_automation_completion(run_dir, poll_interval=5.0, timeout=None, progress_stream=None):
    run_path = Path(run_dir)
    start_time = time.time()
    last_progress = None

    while True:
        manifest = _read_run_manifest(run_path)
        automation = manifest.get("automation") or {}
        requested = automation.get("requested_unit_count")
        completed = automation.get("completed_unit_count", 0)
        successful = automation.get("successful_annotation_count", 0)
        parse_errors = automation.get("parse_error_count", 0)
        validation_errors = automation.get("validation_error_count", 0)
        in_progress = automation.get("in_progress", False)

        progress = (
            requested,
            completed,
            successful,
            parse_errors,
            validation_errors,
            in_progress,
        )
        if progress_stream is not None and progress != last_progress:
            progress_stream.write(
                json.dumps(
                    {
                        "run": str(run_path),
                        "requested_unit_count": requested,
                        "completed_unit_count": completed,
                        "successful_annotation_count": successful,
                        "parse_error_count": parse_errors,
                        "validation_error_count": validation_errors,
                        "in_progress": in_progress,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            progress_stream.flush()
            last_progress = progress

        if not in_progress:
            return {
                "run": str(run_path),
                "requested_unit_count": requested,
                "completed_unit_count": completed,
                "successful_annotation_count": successful,
                "parse_error_count": parse_errors,
                "validation_error_count": validation_errors,
                "in_progress": in_progress,
                "wait_seconds": round(time.time() - start_time, 3),
            }

        if timeout is not None and (time.time() - start_time) >= timeout:
            raise TimeoutError(f"Timed out waiting for automation to finish for run {run_path}.")

        time.sleep(poll_interval)


def build_cross_lens_review_summary(reports):
    cross_lens_entries = {}
    for lens, report in reports.items():
        for entry in report["timeline"]:
            cross_lens_entries.setdefault(
                (entry["unit_id"], entry["character"]),
                {"unit_id": entry["unit_id"], "character": entry["character"], "lenses": {}},
            )["lenses"][lens] = {
                "label": entry["label"],
                "net_score": entry["net_score"],
                "dominant_status_dimension": entry["dominant_status_dimension"],
            }

    comparable_entries = []
    for entry in cross_lens_entries.values():
        if len(entry["lenses"]) != len(SCORING_LENS_CONFIGS):
            continue
        labels = {lens: lens_entry["label"] for lens, lens_entry in entry["lenses"].items()}
        directions = {lens: _label_direction(lens_entry["label"]) for lens, lens_entry in entry["lenses"].items()}
        net_scores = {lens: lens_entry["net_score"] for lens, lens_entry in entry["lenses"].items()}
        label_set = set(labels.values())
        direction_set = set(directions.values())
        comparable_entries.append(
            {
                "unit_id": entry["unit_id"],
                "character": entry["character"],
                "labels": labels,
                "directions": directions,
                "net_scores": net_scores,
                "label_disagreement": len(label_set) > 1,
                "direction_disagreement": len(direction_set) > 1,
            }
        )

    label_disagreements = [entry for entry in comparable_entries if entry["label_disagreement"]]
    direction_disagreements = [entry for entry in comparable_entries if entry["direction_disagreement"]]
    sign_flip_examples = sorted(
        [
            entry
            for entry in comparable_entries
            if "positive" in entry["directions"].values() and "negative" in entry["directions"].values()
        ],
        key=lambda item: (
            -max(item["net_scores"].values()) + min(item["net_scores"].values()),
            item["unit_id"],
            item["character"],
        ),
    )[:10]

    return {
        "comparable_entry_count": len(comparable_entries),
        "label_disagreement_count": len(label_disagreements),
        "direction_disagreement_count": len(direction_disagreements),
        "sign_flip_count": len(sign_flip_examples),
        "sign_flip_examples": sign_flip_examples,
    }


def build_run_review_gate(wait_result, reports, max_mixed_units_per_lens=3):
    review_issues = []
    if wait_result.get("automation_error"):
        review_issues.append(f'automation_error={wait_result["automation_error"]}')
    if wait_result.get("parse_error_count", 0) > 0:
        review_issues.append(f'parse_error_count={wait_result["parse_error_count"]}')
    if wait_result.get("validation_error_count", 0) > 0:
        review_issues.append(f'validation_error_count={wait_result["validation_error_count"]}')

    mixed_unit_counts = {lens: len(report["mixed_units"]) for lens, report in reports.items()}
    for lens, count in sorted(mixed_unit_counts.items()):
        if count > max_mixed_units_per_lens:
            review_issues.append(f"{lens}_mixed_units={count}")

    cross_lens_summary = build_cross_lens_review_summary(reports)
    if cross_lens_summary["sign_flip_count"] > 0:
        review_issues.append(f'cross_lens_sign_flips={cross_lens_summary["sign_flip_count"]}')

    return {
        "ok": not review_issues,
        "review_issue_count": len(review_issues),
        "review_issues": review_issues,
        "mixed_unit_counts": mixed_unit_counts,
        "cross_lens_summary": cross_lens_summary,
        "max_mixed_units_per_lens": max_mixed_units_per_lens,
    }


def run_automated_batch(
    source_run_dir,
    output_dir,
    model="gpt-5",
    overwrite=False,
    limit=None,
    poll_interval=5.0,
    timeout=None,
    progress_stream=None,
    max_mixed_units_per_lens=3,
):
    automation = run_openai_annotation(
        source_run_dir,
        output_dir,
        model=model,
        overwrite=overwrite,
        limit=limit,
    )
    waited = wait_for_automation_completion(
        output_dir,
        poll_interval=poll_interval,
        timeout=timeout,
        progress_stream=progress_stream,
    )
    waited["automation_error"] = automation.get("error")
    reprocess_results = reprocess_raw_annotations(output_dir, overwrite=True, reduce=True)
    reports = {
        lens: build_outcome_report(output_dir, lens=lens)
        for lens in sorted(SCORING_LENS_CONFIGS)
    }
    review_gate = build_run_review_gate(
        waited,
        reports,
        max_mixed_units_per_lens=max_mixed_units_per_lens,
    )
    return {
        "automation": automation,
        "wait": waited,
        "reprocess": {"run": output_dir, "results": reprocess_results},
        "reports": reports,
        "review_gate": review_gate,
    }


def _resolve_scoring_lens(lens):
    try:
        return SCORING_LENS_CONFIGS[lens]
    except KeyError as exc:
        raise ValueError(f'Unknown scoring lens "{lens}". Expected one of: {", ".join(sorted(SCORING_LENS_CONFIGS))}.') from exc


def _outcome_event_score(event, lens_config):
    polarity = event.get("polarity")
    if polarity == "positive":
        polarity_sign = 1.0
    elif polarity == "negative":
        polarity_sign = -1.0
    else:
        polarity_sign = 0.0

    event_weights = lens_config["event_weights"]
    stance_multipliers = lens_config["stance_multipliers"]
    weight = event_weights.get(event.get("type"), event_weights["other"])
    stance_multiplier = stance_multipliers.get(
        event.get("narrative_stance"),
        stance_multipliers["neutral_report"],
    )
    confidence = float(event.get("confidence", 0.0))
    return polarity_sign * weight * stance_multiplier * confidence


def _outcome_status_score(effect, lens_config):
    dimension_weight = lens_config["status_weights"].get(effect.get("dimension"), 1.0)
    confidence = float(effect.get("confidence", 0.0))
    return int(effect.get("delta", 0)) * dimension_weight * confidence


def _outcome_label(net_score, lens_config):
    label_thresholds = lens_config["label_thresholds"]
    if net_score >= label_thresholds["win"]:
        return "win"
    if net_score <= label_thresholds["loss"]:
        return "loss"
    if abs(net_score) < 0.25:
        return "neutral"
    return "mixed"


def _score_run_outcomes(run_dir, lens="local"):
    lens_config = _resolve_scoring_lens(lens)
    status = get_run_status(run_dir)
    manifest = status["manifest"]
    annotation_dir = Path(manifest["directories"]["annotations"])
    summary = {
        "run_id": manifest["run_id"],
        "scoring_version": lens_config["scoring_version"],
        "lens": lens,
        "weights": {
            "event_type": lens_config["event_weights"],
            "status_dimension": lens_config["status_weights"],
            "narrative_stance": lens_config["stance_multipliers"],
            "label_thresholds": lens_config["label_thresholds"],
            "ambiguity_penalty_per_flag": lens_config["ambiguity_penalty"],
        },
        "unit_count": len(manifest["unit_ids"]),
        "scored_unit_count": 0,
        "character_totals": {},
        "units": [],
    }

    for unit_id in manifest["unit_ids"]:
        annotation_path = annotation_dir / _annotation_filename(unit_id)
        if not annotation_path.exists():
            continue

        annotation = _read_json(annotation_path)
        errors = validate_annotation_result(annotation, expected_unit_id=unit_id)
        if errors:
            continue
        ambiguity_penalty = len(annotation["ambiguities"]) * lens_config["ambiguity_penalty"]

        character_scores = {}
        for character in annotation["characters_present"]:
            character_scores[character["canonical_name"]] = {
                "event_score": 0.0,
                "status_score": 0.0,
                "net_score": 0.0,
                "positive_event_count": 0,
                "negative_event_count": 0,
                "event_types": {},
                "status_dimensions": {},
            }

        for event in annotation["appraisal_events"]:
            target = event["target"]
            if target not in character_scores:
                continue
            event_score = _outcome_event_score(event, lens_config)
            target_scores = character_scores[target]
            target_scores["event_score"] += event_score
            target_scores["event_types"][event["type"]] = target_scores["event_types"].get(event["type"], 0) + 1
            if event["polarity"] == "positive":
                target_scores["positive_event_count"] += 1
            elif event["polarity"] == "negative":
                target_scores["negative_event_count"] += 1

        for effect in annotation["status_effects"]:
            character = effect["character"]
            if character not in character_scores:
                continue
            status_score = _outcome_status_score(effect, lens_config)
            character_scores[character]["status_score"] += status_score
            dimension = effect["dimension"]
            character_scores[character]["status_dimensions"][dimension] = (
                character_scores[character]["status_dimensions"].get(dimension, 0) + int(effect["delta"])
            )

        for character, scores in character_scores.items():
            scores["event_score"] = round(scores["event_score"], 3)
            scores["status_score"] = round(scores["status_score"], 3)
            scores["ambiguity_penalty"] = round(ambiguity_penalty, 3)
            scores["net_score"] = round(scores["event_score"] + scores["status_score"] - ambiguity_penalty, 3)
            scores["label"] = _outcome_label(scores["net_score"], lens_config)

            totals = summary["character_totals"].setdefault(
                character,
                {
                    "event_score": 0.0,
                    "status_score": 0.0,
                    "net_score": 0.0,
                    "unit_labels": {"win": 0, "loss": 0, "mixed": 0, "neutral": 0},
                    "status_dimensions": {},
                },
            )
            totals["event_score"] += scores["event_score"]
            totals["status_score"] += scores["status_score"]
            totals["net_score"] += scores["net_score"]
            totals["unit_labels"][scores["label"]] += 1
            for dimension, delta_total in scores["status_dimensions"].items():
                totals["status_dimensions"][dimension] = totals["status_dimensions"].get(dimension, 0) + delta_total

        summary["units"].append(
            {
                "unit_id": unit_id,
                "ambiguity_count": len(annotation["ambiguities"]),
                "characters": character_scores,
            }
        )
        summary["scored_unit_count"] += 1

    for totals in summary["character_totals"].values():
        totals["event_score"] = round(totals["event_score"], 3)
        totals["status_score"] = round(totals["status_score"], 3)
        totals["net_score"] = round(totals["net_score"], 3)

    return summary


def score_run_local_outcomes(run_dir):
    return _score_run_outcomes(run_dir, lens="local")


def score_run_prestige_outcomes(run_dir):
    return _score_run_outcomes(run_dir, lens="prestige")


def score_run_inclusion_outcomes(run_dir):
    return _score_run_outcomes(run_dir, lens="inclusion")


def _sorted_status_dimensions(status_dimensions):
    return sorted(
        status_dimensions.items(),
        key=lambda item: (
            -abs(item[1]),
            PREFERRED_STATUS_DIMENSION_ORDER.get(item[0], 9),
            item[0],
        ),
    )


def _dominant_status_dimension(status_dimensions):
    if not status_dimensions:
        return None
    return _sorted_status_dimensions(status_dimensions)[0][0]


def _build_unit_outcome_entry(unit_id, character, scores):
    dominant_dimension = _dominant_status_dimension(scores["status_dimensions"])
    return {
        "unit_id": unit_id,
        "character": character,
        "label": scores["label"],
        "net_score": scores["net_score"],
        "event_score": scores["event_score"],
        "status_score": scores["status_score"],
        "ambiguity_penalty": scores["ambiguity_penalty"],
        "dominant_status_dimension": dominant_dimension,
        "status_dimensions": dict(_sorted_status_dimensions(scores["status_dimensions"])),
        "event_types": dict(sorted(scores["event_types"].items())),
        "positive_event_count": scores["positive_event_count"],
        "negative_event_count": scores["negative_event_count"],
    }


def build_outcome_report(run_dir, lens="local"):
    score_summary = _score_run_outcomes(run_dir, lens=lens)
    units = []
    character_summaries = {}

    for unit in score_summary["units"]:
        unit_id = unit["unit_id"]
        for character, scores in unit["characters"].items():
            entry = _build_unit_outcome_entry(unit_id, character, scores)
            units.append(entry)

            character_summary = character_summaries.setdefault(
                character,
                {
                    "character": character,
                    "net_score": 0.0,
                    "event_score": 0.0,
                    "status_score": 0.0,
                    "unit_count": 0,
                    "labels": {"win": 0, "loss": 0, "mixed": 0, "neutral": 0},
                    "status_dimensions": {},
                    "top_win": None,
                    "top_loss": None,
                },
            )
            character_summary["net_score"] += entry["net_score"]
            character_summary["event_score"] += entry["event_score"]
            character_summary["status_score"] += entry["status_score"]
            character_summary["unit_count"] += 1
            character_summary["labels"][entry["label"]] += 1
            for dimension, delta_total in entry["status_dimensions"].items():
                character_summary["status_dimensions"][dimension] = (
                    character_summary["status_dimensions"].get(dimension, 0) + delta_total
                )

            if character_summary["top_win"] is None or entry["net_score"] > character_summary["top_win"]["net_score"]:
                character_summary["top_win"] = {"unit_id": unit_id, "net_score": entry["net_score"]}
            if (
                character_summary["top_loss"] is None
                or entry["net_score"] < character_summary["top_loss"]["net_score"]
            ):
                character_summary["top_loss"] = {"unit_id": unit_id, "net_score": entry["net_score"]}

    for summary in character_summaries.values():
        summary["net_score"] = round(summary["net_score"], 3)
        summary["event_score"] = round(summary["event_score"], 3)
        summary["status_score"] = round(summary["status_score"], 3)
        summary["status_dimensions"] = dict(_sorted_status_dimensions(summary["status_dimensions"]))
        summary["dominant_status_dimension"] = _dominant_status_dimension(summary["status_dimensions"])

    sorted_character_summaries = sorted(
        character_summaries.values(),
        key=lambda item: (-item["net_score"], item["character"]),
    )
    top_wins = sorted(units, key=lambda item: (-item["net_score"], item["unit_id"], item["character"]))[:5]
    top_losses = sorted(units, key=lambda item: (item["net_score"], item["unit_id"], item["character"]))[:5]
    mixed_units = [
        entry
        for entry in sorted(units, key=lambda item: (item["unit_id"], item["character"]))
        if entry["label"] == "mixed"
    ]

    return {
        "run_id": score_summary["run_id"],
        "report_version": "outcome_report_v1",
        "scoring_version": score_summary["scoring_version"],
        "lens": lens,
        "scored_unit_count": score_summary["scored_unit_count"],
        "character_count": len(sorted_character_summaries),
        "character_summaries": sorted_character_summaries,
        "timeline": sorted(units, key=lambda item: (item["unit_id"], item["character"])),
        "top_wins": top_wins,
        "top_losses": top_losses,
        "mixed_units": mixed_units,
    }


def _label_direction(label):
    if label == "win":
        return "positive"
    if label == "loss":
        return "negative"
    return "non_directional"


def build_corpus_sanity_review(run_dirs):
    if not run_dirs:
        raise ValueError("At least one run directory is required for a corpus sanity review.")

    run_statuses = []
    run_reports = {}

    for run_dir in run_dirs:
        status = get_run_status(run_dir)
        manifest = status["manifest"]
        run_id = manifest["run_id"]
        run_statuses.append(status)
        run_reports[run_id] = {
            lens: build_outcome_report(run_dir, lens=lens)
            for lens in sorted(SCORING_LENS_CONFIGS)
        }

    run_statuses.sort(key=lambda item: item["manifest"]["run_id"])
    run_ids = [status["manifest"]["run_id"] for status in run_statuses]

    run_surface_rows = []
    aggregate_event_type_counts = {}
    aggregate_event_polarity_counts = {"positive": 0, "negative": 0, "mixed": 0, "neutral": 0}
    aggregate_status_dimension_totals = {}
    lens_character_totals = {lens: {} for lens in sorted(SCORING_LENS_CONFIGS)}
    lens_unit_entries = {lens: [] for lens in sorted(SCORING_LENS_CONFIGS)}
    lens_character_entries = {lens: defaultdict(list) for lens in sorted(SCORING_LENS_CONFIGS)}
    cross_lens_entries = {}

    total_declared_unit_count = 0
    total_valid_annotation_count = 0

    for status in run_statuses:
        manifest = status["manifest"]
        run_id = manifest["run_id"]
        total_declared_unit_count += len(manifest["unit_ids"])

        raw_summary = summarize_run_annotations(Path(manifest["directories"]["annotations"]).parent)
        total_valid_annotation_count += raw_summary["valid_annotation_count"]

        for event_type, count in raw_summary["event_type_counts"].items():
            aggregate_event_type_counts[event_type] = aggregate_event_type_counts.get(event_type, 0) + count
        for polarity, count in raw_summary["event_polarity_counts"].items():
            aggregate_event_polarity_counts[polarity] = aggregate_event_polarity_counts.get(polarity, 0) + count
        for dimension, total in raw_summary["status_dimension_totals"].items():
            aggregate_status_dimension_totals[dimension] = aggregate_status_dimension_totals.get(dimension, 0) + total

        local_report = run_reports[run_id]["local"]
        unit_character_counts = defaultdict(int)
        for entry in local_report["timeline"]:
            unit_character_counts[entry["unit_id"]] += 1

        unit_count = len(manifest["unit_ids"])
        scored_unit_count = local_report["scored_unit_count"]
        unique_character_count = local_report["character_count"]
        single_character_unit_count = sum(1 for count in unit_character_counts.values() if count == 1)
        zero_character_unit_count = unit_count - len(unit_character_counts)
        avg_characters_per_scored_unit = (
            round(sum(unit_character_counts.values()) / len(unit_character_counts), 3)
            if unit_character_counts
            else 0.0
        )
        run_surface_rows.append(
            {
                "run_id": run_id,
                "unit_count": unit_count,
                "scored_unit_count": scored_unit_count,
                "unique_character_count": unique_character_count,
                "avg_characters_per_scored_unit": avg_characters_per_scored_unit,
                "single_character_unit_count": single_character_unit_count,
                "zero_character_unit_count": zero_character_unit_count,
            }
        )

        for lens, report in run_reports[run_id].items():
            for character_summary in report["character_summaries"]:
                existing = lens_character_totals[lens].setdefault(
                    character_summary["character"],
                    {
                        "character": character_summary["character"],
                        "net_score": 0.0,
                        "event_score": 0.0,
                        "status_score": 0.0,
                        "unit_count": 0,
                        "labels": {"win": 0, "loss": 0, "mixed": 0, "neutral": 0},
                        "status_dimensions": {},
                    },
                )
                existing["net_score"] += character_summary["net_score"]
                existing["event_score"] += character_summary["event_score"]
                existing["status_score"] += character_summary["status_score"]
                existing["unit_count"] += character_summary["unit_count"]
                for label, count in character_summary["labels"].items():
                    existing["labels"][label] += count
                for dimension, total in character_summary["status_dimensions"].items():
                    existing["status_dimensions"][dimension] = existing["status_dimensions"].get(dimension, 0) + total

            for entry in report["timeline"]:
                corpus_entry = dict(entry)
                corpus_entry["run_id"] = run_id
                lens_unit_entries[lens].append(corpus_entry)
                lens_character_entries[lens][entry["character"]].append(corpus_entry)
                cross_lens_entries.setdefault(
                    (run_id, entry["unit_id"], entry["character"]),
                    {"run_id": run_id, "unit_id": entry["unit_id"], "character": entry["character"], "lenses": {}},
                )["lenses"][lens] = {
                    "label": entry["label"],
                    "net_score": entry["net_score"],
                    "dominant_status_dimension": entry["dominant_status_dimension"],
                }

    lens_reviews = {}
    for lens in sorted(SCORING_LENS_CONFIGS):
        label_counts = {"win": 0, "loss": 0, "mixed": 0, "neutral": 0}
        for entry in lens_unit_entries[lens]:
            label_counts[entry["label"]] += 1

        top_positive_characters = sorted(
            (
                {
                    "character": totals["character"],
                    "net_score": round(totals["net_score"], 3),
                    "unit_count": totals["unit_count"],
                    "labels": totals["labels"],
                    "dominant_status_dimension": _dominant_status_dimension(totals["status_dimensions"]),
                }
                for totals in lens_character_totals[lens].values()
            ),
            key=lambda item: (-item["net_score"], item["character"]),
        )[:10]
        top_negative_characters = sorted(
            (
                {
                    "character": totals["character"],
                    "net_score": round(totals["net_score"], 3),
                    "unit_count": totals["unit_count"],
                    "labels": totals["labels"],
                    "dominant_status_dimension": _dominant_status_dimension(totals["status_dimensions"]),
                }
                for totals in lens_character_totals[lens].values()
            ),
            key=lambda item: (item["net_score"], item["character"]),
        )[:10]

        volatility_rows = []
        for character, entries in lens_character_entries[lens].items():
            scores = [entry["net_score"] for entry in entries]
            volatility_rows.append(
                {
                    "character": character,
                    "unit_count": len(entries),
                    "min_score": round(min(scores), 3),
                    "max_score": round(max(scores), 3),
                    "score_span": round(max(scores) - min(scores), 3),
                    "mean_score": round(sum(scores) / len(scores), 3),
                }
            )
        most_volatile_characters = sorted(
            [row for row in volatility_rows if row["unit_count"] >= 2],
            key=lambda item: (-item["score_span"], -item["unit_count"], item["character"]),
        )[:10]

        extreme_positive_units = sorted(
            lens_unit_entries[lens],
            key=lambda item: (-item["net_score"], item["run_id"], item["unit_id"], item["character"]),
        )[:10]
        extreme_negative_units = sorted(
            lens_unit_entries[lens],
            key=lambda item: (item["net_score"], item["run_id"], item["unit_id"], item["character"]),
        )[:10]

        lens_reviews[lens] = {
            "entry_count": len(lens_unit_entries[lens]),
            "character_count": len(lens_character_totals[lens]),
            "label_counts": label_counts,
            "top_positive_characters": top_positive_characters,
            "top_negative_characters": top_negative_characters,
            "most_volatile_characters": most_volatile_characters,
            "extreme_positive_units": extreme_positive_units,
            "extreme_negative_units": extreme_negative_units,
        }

    comparable_entries = []
    for entry in cross_lens_entries.values():
        if len(entry["lenses"]) != len(SCORING_LENS_CONFIGS):
            continue
        labels = {lens: lens_entry["label"] for lens, lens_entry in entry["lenses"].items()}
        directions = {lens: _label_direction(lens_entry["label"]) for lens, lens_entry in entry["lenses"].items()}
        net_scores = {lens: lens_entry["net_score"] for lens, lens_entry in entry["lenses"].items()}
        label_set = set(labels.values())
        direction_set = set(directions.values())
        comparable_entries.append(
            {
                "run_id": entry["run_id"],
                "unit_id": entry["unit_id"],
                "character": entry["character"],
                "labels": labels,
                "directions": directions,
                "net_scores": net_scores,
                "label_disagreement": len(label_set) > 1,
                "direction_disagreement": len(direction_set) > 1,
            }
        )

    label_disagreements = [entry for entry in comparable_entries if entry["label_disagreement"]]
    direction_disagreements = [entry for entry in comparable_entries if entry["direction_disagreement"]]

    label_disagreement_examples = sorted(
        label_disagreements,
        key=lambda item: (
            max(item["net_scores"].values()) - min(item["net_scores"].values()),
            item["run_id"],
            item["unit_id"],
            item["character"],
        ),
        reverse=True,
    )[:10]
    direction_disagreement_examples = sorted(
        direction_disagreements,
        key=lambda item: (
            max(item["net_scores"].values()) - min(item["net_scores"].values()),
            item["run_id"],
            item["unit_id"],
            item["character"],
        ),
        reverse=True,
    )[:10]
    sign_flip_examples = sorted(
        [
            entry
            for entry in comparable_entries
            if "positive" in entry["directions"].values() and "negative" in entry["directions"].values()
        ],
        key=lambda item: (
            -max(item["net_scores"].values()) + min(item["net_scores"].values()),
            item["run_id"],
            item["unit_id"],
            item["character"],
        ),
    )[:10]

    narrow_surface_runs = sorted(
        run_surface_rows,
        key=lambda item: (
            item["avg_characters_per_scored_unit"],
            item["unique_character_count"],
            item["run_id"],
        ),
    )[:10]

    return {
        "corpus_review_version": "corpus_sanity_review_v1",
        "run_count": len(run_statuses),
        "run_ids": run_ids,
        "declared_unit_count": total_declared_unit_count,
        "valid_annotation_count": total_valid_annotation_count,
        "aggregate_annotation_summary": {
            "event_type_counts": dict(sorted(aggregate_event_type_counts.items())),
            "event_polarity_counts": aggregate_event_polarity_counts,
            "status_dimension_totals": dict(_sorted_status_dimensions(aggregate_status_dimension_totals)),
        },
        "run_surface_summaries": sorted(run_surface_rows, key=lambda item: item["run_id"]),
        "narrow_surface_runs": narrow_surface_runs,
        "lens_reviews": lens_reviews,
        "cross_lens_summary": {
            "comparable_entry_count": len(comparable_entries),
            "label_disagreement_count": len(label_disagreements),
            "label_disagreement_rate": round(
                len(label_disagreements) / len(comparable_entries), 3
            )
            if comparable_entries
            else 0.0,
            "direction_disagreement_count": len(direction_disagreements),
            "direction_disagreement_rate": round(
                len(direction_disagreements) / len(comparable_entries), 3
            )
            if comparable_entries
            else 0.0,
            "label_disagreement_examples": label_disagreement_examples,
            "direction_disagreement_examples": direction_disagreement_examples,
            "sign_flip_examples": sign_flip_examples,
        },
    }


def discover_annotation_run_dirs(outputs_dir="outputs"):
    output_path = Path(outputs_dir)
    if not output_path.exists():
        raise ValueError(f'Outputs directory "{output_path}" does not exist.')

    run_dirs = []
    for run_dir in sorted(output_path.glob("run-*")):
        if not run_dir.is_dir() or not (run_dir / "run.json").exists():
            continue
        annotation_dir = run_dir / "annotations"
        if not annotation_dir.exists() or not any(annotation_dir.glob("*.json")):
            continue
        status = get_run_status(run_dir)
        if status["summary"]["valid_annotation_count"] > 0:
            run_dirs.append(run_dir)

    if not run_dirs:
        raise ValueError(f'No annotated run directories found under "{output_path}".')

    return run_dirs


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_signed_number(value):
    if isinstance(value, float):
        value = round(value, 3)
    if isinstance(value, (int, float)) and value > 0:
        return f"+{value}"
    return str(value)


def render_corpus_review_markdown(review):
    annotation_summary = review["aggregate_annotation_summary"]
    cross_lens_summary = review["cross_lens_summary"]
    lines = [
        "# Corpus Review",
        "",
        f"- Review version: `{review['corpus_review_version']}`",
        f"- Run count: `{review['run_count']}`",
        f"- Declared unit count: `{review['declared_unit_count']}`",
        f"- Valid annotation count: `{review['valid_annotation_count']}`",
        "",
        "## Aggregate Annotation Summary",
        "",
        "### Event Polarity Counts",
        "",
        _markdown_table(
            ["Polarity", "Count"],
            annotation_summary["event_polarity_counts"].items(),
        ),
        "",
        "### Status Dimension Totals",
        "",
        _markdown_table(
            ["Dimension", "Total"],
            [
                (dimension, _format_signed_number(total))
                for dimension, total in annotation_summary["status_dimension_totals"].items()
            ],
        ),
        "",
        "### Event Type Counts",
        "",
        _markdown_table(
            ["Event Type", "Count"],
            annotation_summary["event_type_counts"].items(),
        ),
        "",
        "## Run Surface",
        "",
        _markdown_table(
            [
                "Run",
                "Units",
                "Scored Units",
                "Characters",
                "Avg Characters/Scored Unit",
                "Zero-character Units",
            ],
            [
                (
                    row["run_id"],
                    row["unit_count"],
                    row["scored_unit_count"],
                    row["unique_character_count"],
                    row["avg_characters_per_scored_unit"],
                    row["zero_character_unit_count"],
                )
                for row in review["run_surface_summaries"][:25]
            ],
        ),
        "",
    ]

    if len(review["run_surface_summaries"]) > 25:
        lines.extend(
            [
                f"_Showing first 25 of {len(review['run_surface_summaries'])} run surface rows._",
                "",
            ]
        )

    lines.extend(
        [
            "### Narrowest Surface Runs",
            "",
            _markdown_table(
                ["Run", "Units", "Characters", "Avg Characters/Scored Unit", "Zero-character Units"],
                [
                    (
                        row["run_id"],
                        row["unit_count"],
                        row["unique_character_count"],
                        row["avg_characters_per_scored_unit"],
                        row["zero_character_unit_count"],
                    )
                    for row in review["narrow_surface_runs"]
                ],
            ),
            "",
            "## Lens Reviews",
            "",
        ]
    )

    for lens, lens_review in review["lens_reviews"].items():
        lines.extend(
            [
                f"### {lens}",
                "",
                f"- Entry count: `{lens_review['entry_count']}`",
                f"- Character count: `{lens_review['character_count']}`",
                "",
                "Label counts:",
                "",
                _markdown_table(["Label", "Count"], lens_review["label_counts"].items()),
                "",
                "Top positive characters:",
                "",
                _markdown_table(
                    ["Character", "Net Score", "Units", "Dominant Dimension"],
                    [
                        (
                            row["character"],
                            _format_signed_number(row["net_score"]),
                            row["unit_count"],
                            row["dominant_status_dimension"],
                        )
                        for row in lens_review["top_positive_characters"]
                    ],
                ),
                "",
                "Top negative characters:",
                "",
                _markdown_table(
                    ["Character", "Net Score", "Units", "Dominant Dimension"],
                    [
                        (
                            row["character"],
                            _format_signed_number(row["net_score"]),
                            row["unit_count"],
                            row["dominant_status_dimension"],
                        )
                        for row in lens_review["top_negative_characters"]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-Lens Summary",
            "",
            f"- Comparable entries: `{cross_lens_summary['comparable_entry_count']}`",
            f"- Label disagreement count: `{cross_lens_summary['label_disagreement_count']}`",
            f"- Label disagreement rate: `{cross_lens_summary['label_disagreement_rate']}`",
            f"- Direction disagreement count: `{cross_lens_summary['direction_disagreement_count']}`",
            f"- Direction disagreement rate: `{cross_lens_summary['direction_disagreement_rate']}`",
            f"- Sign-flip examples: `{len(cross_lens_summary['sign_flip_examples'])}`",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def write_corpus_review_artifacts(review, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_corpus_review_markdown(review))


def prepare_annotation_run_from_existing(
    source_run_dir,
    output_dir,
    run_id=None,
    notes="",
):
    source_status = get_run_status(source_run_dir)
    source_manifest = source_status["manifest"]
    source_run_path = Path(source_run_dir)
    output_path = Path(output_dir)
    resolved_run_id = run_id or output_path.name
    directories = _ensure_run_directories(output_path)

    _copy_run_file_tree(source_run_path / "units", directories["units"], ".json")
    _copy_run_file_tree(source_run_path / "prompts", directories["prompts"], ".txt")

    manifest = AnnotationRunManifest(
        run_id=resolved_run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        prompt_path=source_manifest["prompt_path"],
        unit_ids=list(source_manifest["unit_ids"]),
        directories={name: str(path.resolve()) for name, path in directories.items()},
        alias_map=source_manifest["alias_map"],
        notes=notes or f'automated run derived from {source_manifest["run_id"]}',
        derived_from={
            "run_id": source_manifest["run_id"],
            "run_path": str(source_run_path.resolve()),
        },
    )
    _write_run_manifest(output_path, asdict(manifest))
    return manifest


def run_annotation_requests(
    run_dir,
    requester,
    model,
    overwrite=False,
    limit=None,
):
    run_path = Path(run_dir)
    status = get_run_status(run_path)
    manifest = status["manifest"]
    directories = {name: Path(path) for name, path in manifest["directories"].items()}
    selected_units = []

    for unit_status in status["units"]:
        if not overwrite and unit_status["annotation_exists"]:
            continue
        selected_units.append(unit_status["unit_id"])

    if limit is not None:
        selected_units = selected_units[:limit]

    results = []
    successful_annotations = 0
    parse_error_count = 0
    validation_error_count = 0
    requested_at = datetime.now(timezone.utc).isoformat()

    def write_automation_state(*, in_progress, error_message=None, failed_at=None, completed_at=None):
        refreshed_manifest = _read_run_manifest(run_path)
        refreshed_manifest["automation"] = {
            "provider": "openai",
            "model": model,
            "requested_at": requested_at,
            "requested_unit_count": len(selected_units),
            "successful_annotation_count": successful_annotations,
            "parse_error_count": parse_error_count,
            "validation_error_count": validation_error_count,
            "overwrite": overwrite,
            "limit": limit,
            "completed_unit_count": len(results),
            "results": results,
            "in_progress": in_progress,
        }
        if error_message is not None:
            refreshed_manifest["automation"]["error"] = error_message
        if failed_at is not None:
            refreshed_manifest["automation"]["failed_at"] = failed_at
        if completed_at is not None:
            refreshed_manifest["automation"]["completed_at"] = completed_at
        _write_run_manifest(run_path, refreshed_manifest)
        return refreshed_manifest["automation"]

    write_automation_state(in_progress=True)

    try:
        for unit_id in selected_units:
            unit_payload = _read_json(directories["units"] / _unit_filename(unit_id))
            prompt_text = (directories["prompts"] / _prompt_filename(unit_id)).read_text()
            raw_text = requester(prompt_text, unit_payload, model)
            write_raw_response(run_path, unit_id, raw_text)

            parse_error = None
            validation_errors = []
            annotation_written = False

            try:
                annotation = parse_annotation_response_text(raw_text, expected_unit_id=unit_id)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                parse_error_count += 1
            else:
                validation_errors = validate_annotation_result(annotation, expected_unit_id=unit_id)
                if validation_errors:
                    validation_error_count += 1
                else:
                    write_annotation_result(run_path, unit_id, annotation)
                    annotation_written = True
                    successful_annotations += 1

            results.append(
                {
                    "unit_id": unit_id,
                    "annotation_written": annotation_written,
                    "parse_error": parse_error,
                    "validation_errors": validation_errors,
                }
            )

            write_automation_state(in_progress=True)
    except Exception as exc:
        return write_automation_state(
            in_progress=False,
            error_message=str(exc),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )

    return write_automation_state(
        in_progress=False,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def reprocess_raw_annotations(run_dir, overwrite=False, reduce=False):
    run_path = Path(run_dir)
    status = get_run_status(run_path)
    manifest = status["manifest"]
    raw_dir = Path(manifest["directories"]["raw"])
    annotation_dir = Path(manifest["directories"]["annotations"])
    results = []

    for unit_status in status["units"]:
        unit_id = unit_status["unit_id"]
        raw_path = raw_dir / _raw_filename(unit_id)
        annotation_path = annotation_dir / _annotation_filename(unit_id)
        if not raw_path.exists():
            continue
        if annotation_path.exists() and not overwrite:
            continue

        parse_error = None
        validation_errors = []
        annotation_written = False

        try:
            annotation = parse_annotation_response_text(raw_path.read_text(), expected_unit_id=unit_id)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
        else:
            if reduce:
                annotation = reduce_annotation_result(annotation, expected_unit_id=unit_id)
            validation_errors = validate_annotation_result(annotation, expected_unit_id=unit_id)
            if not validation_errors:
                write_annotation_result(run_path, unit_id, annotation)
                annotation_written = True

        results.append(
            {
                "unit_id": unit_id,
                "annotation_written": annotation_written,
                "parse_error": parse_error,
                "validation_errors": validation_errors,
            }
        )

    refreshed_manifest = _read_run_manifest(run_path)
    if refreshed_manifest.get("automation") is None:
        refreshed_manifest["automation"] = {}
    refreshed_manifest["automation"]["reprocessed_at"] = datetime.now(timezone.utc).isoformat()
    refreshed_manifest["automation"]["reprocess_reduce"] = reduce
    refreshed_manifest["automation"]["reprocess_results"] = results
    _write_run_manifest(run_path, refreshed_manifest)
    return results


def reduce_run_annotations(run_dir, overwrite=False):
    return reprocess_raw_annotations(run_dir, overwrite=overwrite, reduce=True)


def _openai_responses_request(prompt_text, unit_payload, model, api_key=None, timeout=180, max_attempts=4):
    del unit_payload
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    request_body = {
        "model": model,
        "input": prompt_text,
    }
    request = urllib_request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    transient_http_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            is_retryable = exc.code in transient_http_statuses and attempt < max_attempts
            if not is_retryable:
                raise RuntimeError(f"OpenAI API request failed with status {exc.code}: {body}") from exc
            time.sleep(min(2 ** (attempt - 1), 8))
        except (urllib_error.URLError, TimeoutError, RemoteDisconnected) as exc:
            if attempt >= max_attempts:
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(f"OpenAI API request failed: {reason}") from exc
            time.sleep(min(2 ** (attempt - 1), 8))

    raw_text = extract_response_output_text(response_payload)
    if not raw_text:
        raise RuntimeError("OpenAI API response did not contain output text.")
    return raw_text


def run_openai_annotation(
    source_run_dir,
    output_dir,
    model="gpt-5",
    overwrite=False,
    limit=None,
    api_key=None,
):
    output_path = Path(output_dir)
    if not (output_path / "run.json").exists():
        prepare_annotation_run_from_existing(
            source_run_dir,
            output_path,
            notes=f'automated run derived from {Path(source_run_dir).name}',
        )

    return run_annotation_requests(
        output_path,
        requester=lambda prompt_text, unit_payload, active_model: _openai_responses_request(
            prompt_text,
            unit_payload,
            active_model,
            api_key=api_key,
        ),
        model=model,
        overwrite=overwrite,
        limit=limit,
    )


def compare_run_to_benchmark(run_dir, benchmark_run_dir):
    run_status = get_run_status(run_dir)
    benchmark_status = get_run_status(benchmark_run_dir)
    run_manifest = run_status["manifest"]
    benchmark_manifest = benchmark_status["manifest"]
    run_annotation_dir = Path(run_manifest["directories"]["annotations"])
    benchmark_annotation_dir = Path(benchmark_manifest["directories"]["annotations"])

    run_unit_ids = set(run_manifest["unit_ids"])
    benchmark_unit_ids = set(benchmark_manifest["unit_ids"])
    shared_unit_ids = sorted(run_unit_ids & benchmark_unit_ids)
    benchmark_only_unit_ids = sorted(benchmark_unit_ids - run_unit_ids)
    run_only_unit_ids = sorted(run_unit_ids - benchmark_unit_ids)

    per_unit = []
    exact_match_count = 0
    differing_annotation_count = 0
    missing_annotation_count = 0

    for unit_id in shared_unit_ids:
        run_annotation_path = run_annotation_dir / _annotation_filename(unit_id)
        benchmark_annotation_path = benchmark_annotation_dir / _annotation_filename(unit_id)
        run_annotation_exists = run_annotation_path.exists()
        benchmark_annotation_exists = benchmark_annotation_path.exists()
        annotations_equal = False

        if run_annotation_exists and benchmark_annotation_exists:
            annotations_equal = _read_json(run_annotation_path) == _read_json(benchmark_annotation_path)

        if run_annotation_exists and benchmark_annotation_exists and annotations_equal:
            exact_match_count += 1
        elif not run_annotation_exists or not benchmark_annotation_exists:
            missing_annotation_count += 1
        else:
            differing_annotation_count += 1

        per_unit.append(
            {
                "unit_id": unit_id,
                "run_annotation_exists": run_annotation_exists,
                "benchmark_annotation_exists": benchmark_annotation_exists,
                "annotation_exact_match": annotations_equal,
            }
        )

    summary = {
        "run_id": run_manifest["run_id"],
        "benchmark_run_id": benchmark_manifest["run_id"],
        "shared_unit_count": len(shared_unit_ids),
        "benchmark_only_unit_count": len(benchmark_only_unit_ids),
        "run_only_unit_count": len(run_only_unit_ids),
        "exact_match_count": exact_match_count,
        "differing_annotation_count": differing_annotation_count,
        "missing_annotation_count": missing_annotation_count,
        "all_shared_annotations_match": (
            len(shared_unit_ids) > 0 and differing_annotation_count == 0 and missing_annotation_count == 0
        ),
    }
    return {
        "run": run_status,
        "benchmark": benchmark_status,
        "summary": summary,
        "shared_unit_ids": shared_unit_ids,
        "benchmark_only_unit_ids": benchmark_only_unit_ids,
        "run_only_unit_ids": run_only_unit_ids,
        "units": per_unit,
    }


def mark_run_as_benchmark(run_dir, label="reviewed benchmark"):
    run_path = Path(run_dir)
    status = get_run_status(run_path)
    manifest = status["manifest"]
    summary = status["summary"]
    manifest["benchmark"] = {
        "label": label,
        "status": "reviewed" if summary["benchmark_ready"] else "incomplete",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_unit_ids": [
            unit_status["unit_id"]
            for unit_status in status["units"]
            if unit_status["review_state"] == "reviewed"
        ],
        "valid_annotation_count": summary["valid_annotation_count"],
        "pending_unit_count": summary["pending_unit_count"],
        "benchmark_ready": summary["benchmark_ready"],
    }
    (run_path / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest["benchmark"]


def prepare_annotation_run(
    output_dir,
    run_id=None,
    unit_specs=None,
    alias_map=None,
    prompt_path=None,
    notes="",
):
    output_path = Path(output_dir)
    resolved_run_id = run_id or output_path.name
    run_dir = output_path
    directories = _ensure_run_directories(run_dir)
    active_alias_map = alias_map or DEFAULT_STARTER_ALIAS_MAP
    prompt_template_path = Path(prompt_path) if prompt_path else PROMPT_PATH
    prompt_template = load_prompt_template(prompt_template_path)
    selected_unit_specs = list(unit_specs or STARTER_UNITS)
    units = []

    for unit_spec in selected_unit_specs:
        unit = build_annotation_unit(
            unit_spec.chapter_id,
            unit_spec.paragraph_start,
            paragraph_end=unit_spec.paragraph_end,
            prior_context_paragraphs=1,
            alias_map=active_alias_map,
        )
        if unit_spec.notes:
            unit["notes"] = unit_spec.notes
        units.append(unit)

        (directories["units"] / _unit_filename(unit["unit_id"])).write_text(
            json.dumps(unit, ensure_ascii=False, indent=2) + "\n"
        )
        (directories["prompts"] / _prompt_filename(unit["unit_id"])).write_text(
            render_prompt_input(unit, prompt_template=prompt_template)
        )

    manifest = AnnotationRunManifest(
        run_id=resolved_run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        prompt_path=str(Path(prompt_template_path).resolve()),
        unit_ids=[unit["unit_id"] for unit in units],
        directories={name: str(path.resolve()) for name, path in directories.items()},
        alias_map=active_alias_map,
        notes=notes,
    )
    (run_dir / "run.json").write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n")
    return manifest


def write_raw_response(run_dir, unit_id, raw_text):
    raw_path = Path(run_dir) / "raw" / _raw_filename(unit_id)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text)
    return raw_path


def write_annotation_result(run_dir, unit_id, annotation):
    annotation_path = Path(run_dir) / "annotations" / _annotation_filename(unit_id)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n")
    return annotation_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare an ISLT annotation run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare an annotation run directory.")
    prepare_parser.add_argument("--output", required=True, help="Run directory to create.")
    prepare_parser.add_argument("--run-id", help="Optional run identifier. Defaults to the output directory name.")
    prepare_parser.add_argument("--notes", default="", help="Optional note stored in run.json.")
    prepare_parser.add_argument("--prompt", dest="prompt_path", help="Optional prompt template path.")

    status_parser = subparsers.add_parser("status", help="Summarize and validate an annotation run.")
    status_parser.add_argument("--run", required=True, help="Run directory to inspect.")
    status_parser.add_argument(
        "--write-benchmark",
        action="store_true",
        help="Persist benchmark validation metadata back into run.json.",
    )
    status_parser.add_argument(
        "--label",
        default="reviewed benchmark",
        help="Benchmark label written into run.json when --write-benchmark is used.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare a run against a reviewed benchmark.")
    compare_parser.add_argument("--run", required=True, help="Run directory to inspect.")
    compare_parser.add_argument("--benchmark", required=True, help="Benchmark run directory.")

    summary_parser = subparsers.add_parser("summary", help="Aggregate validated annotations in a run.")
    summary_parser.add_argument("--run", required=True, help="Run directory to summarize.")

    score_parser = subparsers.add_parser(
        "score",
        help='Compute a lightweight local "winning"/"losing" transformation for a run.',
    )
    score_parser.add_argument("--run", required=True, help="Run directory to score.")
    score_parser.add_argument("--lens", default="local", choices=sorted(SCORING_LENS_CONFIGS), help="Scoring lens.")

    report_parser = subparsers.add_parser(
        "report",
        help="Build a compact downstream outcome report from local outcome scores.",
    )
    report_parser.add_argument("--run", required=True, help="Run directory to report on.")
    report_parser.add_argument("--lens", default="local", choices=sorted(SCORING_LENS_CONFIGS), help="Scoring lens.")

    corpus_review_parser = subparsers.add_parser(
        "corpus-review",
        help="Aggregate multiple runs into a corpus-level sanity review.",
    )
    corpus_review_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    corpus_review_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    corpus_review_parser.add_argument("--output", help="Optional JSON output path.")
    corpus_review_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    automate_parser = subparsers.add_parser("automate", help="Run prompts in a prepared source run through OpenAI.")
    automate_parser.add_argument("--source-run", required=True, help="Reviewed or candidate source run directory.")
    automate_parser.add_argument("--output", required=True, help="Output run directory for automated results.")
    automate_parser.add_argument("--model", default="gpt-5", help="OpenAI model to use.")
    automate_parser.add_argument("--limit", type=int, help="Optional maximum number of units to request.")
    automate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-request units even if annotations already exist in the output run.",
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run a prepared source batch end-to-end: automate, wait, reduce, report, and review-gate.",
    )
    batch_parser.add_argument("--source-run", required=True, help="Reviewed or candidate source run directory.")
    batch_parser.add_argument("--output", required=True, help="Output run directory for automated results.")
    batch_parser.add_argument("--model", default="gpt-5", help="OpenAI model to use.")
    batch_parser.add_argument("--limit", type=int, help="Optional maximum number of units to request.")
    batch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-request units even if annotations already exist in the output run.",
    )
    batch_parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between manifest checks.",
    )
    batch_parser.add_argument(
        "--timeout",
        type=float,
        help="Optional maximum number of seconds to wait before failing.",
    )
    batch_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress incremental progress output while waiting.",
    )
    batch_parser.add_argument(
        "--max-mixed-units-per-lens",
        type=int,
        default=3,
        help="Review-gate threshold for mixed units in a single lens.",
    )

    wait_parser = subparsers.add_parser(
        "wait",
        help="Wait for an automated run to finish and optionally post-process it.",
    )
    wait_parser.add_argument("--run", required=True, help="Automated run directory to monitor.")
    wait_parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between manifest checks.",
    )
    wait_parser.add_argument(
        "--timeout",
        type=float,
        help="Optional maximum number of seconds to wait before failing.",
    )
    wait_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress incremental progress output while waiting.",
    )
    wait_parser.add_argument(
        "--reduce",
        action="store_true",
        help="Run reducer-based reprocessing after automation completes.",
    )
    wait_parser.add_argument(
        "--report",
        action="store_true",
        help="Build local, prestige, and inclusion reports after completion.",
    )

    reprocess_parser = subparsers.add_parser("reprocess", help="Re-parse saved raw outputs into annotations.")
    reprocess_parser.add_argument("--run", required=True, help="Run directory to reprocess.")
    reprocess_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite annotations even if they already exist.",
    )
    reprocess_parser.add_argument(
        "--reduce",
        action="store_true",
        help="Apply the first-pass reducer before validation and writing annotations.",
    )
    args = parser.parse_args(argv)

    if args.command == "prepare":
        prepare_annotation_run(
            args.output,
            run_id=args.run_id,
            prompt_path=args.prompt_path,
            notes=args.notes,
        )
        return 0

    if args.command == "compare":
        try:
            comparison = compare_run_to_benchmark(args.run, args.benchmark)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "summary":
        try:
            summary = summarize_run_annotations(args.run)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score":
        try:
            summary = _score_run_outcomes(args.run, lens=args.lens)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report":
        try:
            report = build_outcome_report(args.run, lens=args.lens)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "corpus-review":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            review = build_corpus_sanity_review(runs)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_corpus_review_artifacts(
            review,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "run_count": review["run_count"],
                        "declared_unit_count": review["declared_unit_count"],
                        "valid_annotation_count": review["valid_annotation_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0

    if args.command == "automate":
        try:
            automation = run_openai_annotation(
                args.source_run,
                args.output,
                model=args.model,
                overwrite=args.overwrite,
                limit=args.limit,
            )
        except (RunManifestNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(automation, ensure_ascii=False, indent=2))
        return 0

    if args.command == "batch":
        try:
            progress_stream = None if args.quiet else sys.stderr
            result = run_automated_batch(
                args.source_run,
                args.output,
                model=args.model,
                overwrite=args.overwrite,
                limit=args.limit,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                progress_stream=progress_stream,
                max_mixed_units_per_lens=args.max_mixed_units_per_lens,
            )
        except (RunManifestNotFoundError, RuntimeError, TimeoutError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["review_gate"]["ok"] else 2

    if args.command == "wait":
        try:
            progress_stream = None if args.quiet else sys.stderr
            waited = wait_for_automation_completion(
                args.run,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                progress_stream=progress_stream,
            )
            result = {"wait": waited}
            if args.reduce:
                reprocess_results = reprocess_raw_annotations(args.run, overwrite=True, reduce=True)
                result["reprocess"] = {"run": args.run, "results": reprocess_results}
            if args.report:
                result["reports"] = {
                    lens: build_outcome_report(args.run, lens=lens)
                    for lens in sorted(SCORING_LENS_CONFIGS)
                }
        except (RunManifestNotFoundError, RuntimeError, TimeoutError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "reprocess":
        try:
            results = reprocess_raw_annotations(args.run, overwrite=args.overwrite, reduce=args.reduce)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps({"run": args.run, "results": results}, ensure_ascii=False, indent=2))
        return 0

    try:
        status = get_run_status(args.run)
    except RunManifestNotFoundError as exc:
        parser.error(str(exc))
    print(json.dumps(status["summary"], ensure_ascii=False, indent=2))
    if args.write_benchmark:
        mark_run_as_benchmark(args.run, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
