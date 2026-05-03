ADVANTAGE_OUTCOME_EVENT_WEIGHTS = {
    "narrated_elevation": 1.0,
    "prestige_association": 1.0,
    "admiration": 0.9,
    "snub": 1.1,
    "discredit_association": 1.0,
    "narrated_diminishment": 1.0,
    "blame": 0.9,
    "other": 0.6,
}

ADVANTAGE_OUTCOME_STATUS_WEIGHTS = {
    "social_status": 1.3,
    "inclusion_exclusion": 1.2,
    "general_appraisal": 1.0,
    "rhetorical_position": 0.8,
    "emotional_position": 0.6,
}

ADVANTAGE_OUTCOME_STANCE_MULTIPLIERS = {
    "endorsed": 1.0,
    "neutral_report": 0.9,
    "ironized": 0.7,
    "uncertain": 0.5,
}

ADVANTAGE_OUTCOME_LABEL_THRESHOLDS = {
    "win": 0.75,
    "loss": -0.75,
}

ADVANTAGE_OUTCOME_AMBIGUITY_PENALTY = 0.4

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
    "advantage": {
        "scoring_version": "advantage_outcome_v1",
        "event_weights": ADVANTAGE_OUTCOME_EVENT_WEIGHTS,
        "status_weights": ADVANTAGE_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": ADVANTAGE_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": ADVANTAGE_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": ADVANTAGE_OUTCOME_AMBIGUITY_PENALTY,
    },
    "prestige": {
        "scoring_version": "prestige_outcome_v1",
        "event_weights": PRESTIGE_OUTCOME_EVENT_WEIGHTS,
        "status_weights": PRESTIGE_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": ADVANTAGE_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": ADVANTAGE_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": ADVANTAGE_OUTCOME_AMBIGUITY_PENALTY,
    },
    "inclusion": {
        "scoring_version": "inclusion_outcome_v1",
        "event_weights": INCLUSION_OUTCOME_EVENT_WEIGHTS,
        "status_weights": INCLUSION_OUTCOME_STATUS_WEIGHTS,
        "stance_multipliers": ADVANTAGE_OUTCOME_STANCE_MULTIPLIERS,
        "label_thresholds": ADVANTAGE_OUTCOME_LABEL_THRESHOLDS,
        "ambiguity_penalty": ADVANTAGE_OUTCOME_AMBIGUITY_PENALTY,
    },
}

SCORING_LENS_ORDER = ("advantage", "prestige", "inclusion")


__all__ = [
    "ADVANTAGE_OUTCOME_AMBIGUITY_PENALTY",
    "ADVANTAGE_OUTCOME_EVENT_WEIGHTS",
    "ADVANTAGE_OUTCOME_LABEL_THRESHOLDS",
    "ADVANTAGE_OUTCOME_STANCE_MULTIPLIERS",
    "ADVANTAGE_OUTCOME_STATUS_WEIGHTS",
    "INCLUSION_OUTCOME_EVENT_WEIGHTS",
    "INCLUSION_OUTCOME_STATUS_WEIGHTS",
    "PRESTIGE_OUTCOME_EVENT_WEIGHTS",
    "PRESTIGE_OUTCOME_STATUS_WEIGHTS",
    "SCORING_LENS_CONFIGS",
    "SCORING_LENS_ORDER",
]
