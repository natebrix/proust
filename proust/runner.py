import argparse
import csv
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
import unicodedata
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
from .export import CANONICAL_CHAPTER_SPECS
from .paths import ALIASES_CSV

ISLT_PORTRAITS_DIR = Path("/Users/nathan_brixius/dev/brixius-web/public/projects/islt/portraits")
ISLT_READER_BASE_PATH = "/projects/islt/fr-original"
PORTRAIT_STYLES = (
    "vermeer-proustian",
    "tarot-marseille-belle-epoque",
    "elstir",
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

REVIEWED_CHARACTER_NORMALIZATION_MAP = {
    "Saint-Loup": "Robert de Saint-Loup",
    "princesse des Laumes": "duchesse de Guermantes",
    "Charlus": "baron de Charlus",
    "Mme Swann": "Odette",
    "la grand-mère du narrateur": "la grand-mère",
    "Vinteuil": "M. Vinteuil",
    "Mme de Saint-Euverte": "marquise de Saint-Euverte",
}

CHARACTER_PORTRAIT_SLUGS = {
    "Albertine": "albertine",
    "Odette": "odette",
    "Robert de Saint-Loup": "saint-loup",
    "Swann": "swann",
    "baron de Charlus": "charlus",
}

CHARACTER_PAGE_PILOT_EDITORIAL = {
    "Odette": {
        "dek": "Prestige-positive but inclusion-negative, with her sharpest gains and reversals concentrated in a few high-pressure chapters.",
        "summary": "Odette is one of the clearest cross-lens split figures in the corpus: she rises strongly in prestige while remaining far more unstable in belonging and immediate advantage.",
        "why_interesting": [
            "Her prestige and inclusion readings diverge much more sharply than her raw frequency alone would predict.",
            "Her profile is driven by a few concentrated chapter zones rather than a flat corpus-wide pattern.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Prestige ascent around Mme Swann"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Negative counterweight in Swann's love"},
            {"chapter_id": "v3-p1", "label": "Later reversals in Guermantes-adjacent society"},
        ],
    },
    "Robert de Saint-Loup": {
        "dek": "One of the most frequent figures in the corpus, but also one of the most sharply split across prestige and inclusion.",
        "summary": "Robert de Saint-Loup combines very high annotation frequency with one of the largest lens spreads in the corpus, especially where aristocratic polish and emotional belonging pull apart.",
        "why_interesting": [
            "He is not just volatile; he is structurally split across lenses in a way that makes him central to the project's interpretive payoff.",
            "His strongest divergence is heavily chapter-located, especially in v3-p1.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Core prestige/inclusion split"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Later stabilizing wartime treatment"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Earlier positive social energy"},
        ],
    },
    "Swann": {
        "dek": "The most annotated figure in the corpus and one of its most consistently negative, especially in emotionally charged social passages.",
        "summary": "Swann dominates the corpus by sheer annotation footprint, and his aggregate profile remains broadly and repeatedly negative across all three lenses.",
        "why_interesting": [
            "He is the clearest example of frequency and stability combining into a durable corpus-wide shape.",
            "His chapter drivers reveal how strongly one major narrative zone can define a character's aggregate outcome.",
        ],
        "primary_pattern": "consistently_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "The main negative core of the profile"},
            {"chapter_id": "v4-p2", "label": "Later social afterlife and residue"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Secondary shaping around Mme Swann"},
        ],
    },
    "Albertine": {
        "dek": "Highly central and highly negative, with much of her profile concentrated in the captivity and aftermath volumes.",
        "summary": "Albertine is one of the largest and most persistently negative figures in the corpus, with her strongest shaping concentrated in the prison and disappearance chapters.",
        "why_interesting": [
            "Her aggregate pattern is less about lens disagreement than about strong repeated negative accumulation.",
            "She helps distinguish cross-lens split figures from characters whose importance lies in concentrated directional force.",
        ],
        "primary_pattern": "broadly_negative",
        "reading_path": [
            {"chapter_id": "v5", "label": "The main captivity-zone accumulation"},
            {"chapter_id": "v6-p1", "label": "Aftermath and disappearance"},
            {"chapter_id": "v4-p2", "label": "Earlier social framing before concentration in v5"},
        ],
    },
    "baron de Charlus": {
        "dek": "A major recurrent figure whose treatment is strongly negative overall but distributed across several distinct social terrains.",
        "summary": "baron de Charlus is a highly annotated and highly volatile figure whose negative aggregate treatment is spread across salon, sexual, and wartime configurations rather than one single narrative block.",
        "why_interesting": [
            "He is central both to corpus-wide negativity and to some of the project's richest later-terrain dynamics.",
            "His profile is broad enough to test whether the analysis stays coherent across very different social worlds.",
        ],
        "primary_pattern": "volatile_negative",
        "reading_path": [
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime concentration"},
            {"chapter_id": "v4-p2", "label": "Salon and sexual volatility"},
            {"chapter_id": "v5", "label": "Continuation of negative concentration"},
        ],
    },
    "duchesse de Guermantes": {
        "dek": "The clearest aggregate winner in the corpus, combining immediate advantage, prestige, and inclusion at the very top of all three lenses.",
        "summary": "duchesse de Guermantes is the strongest uniformly positive figure in the current corpus surface, with her social command and symbolic force holding across every lens rather than depending on a narrow chapter exception.",
        "why_interesting": [
            "She is the cleanest example of a character whose authority remains legible whether the lens tracks immediate advantage, rank, or belonging.",
            "Her profile shows how a major aristocratic figure can dominate the aggregate surface without needing volatility or cross-lens disagreement to become interesting.",
        ],
        "primary_pattern": "uniformly_ascendant",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "First major Guermantes concentration"},
            {"chapter_id": "v3-p2", "label": "Sustained supremacy in the salon world"},
            {"chapter_id": "v5", "label": "Later reinforcement of the same social position"},
        ],
    },
    "Mme de Villeparisis": {
        "dek": "A socially legible figure whose prestige often holds better than her immediate advantage or belonging.",
        "summary": "Mme de Villeparisis is one of the clearest moderate split figures in the corpus: she remains comparatively strong in prestige while advantage and inclusion drift downward or oscillate by chapter.",
        "why_interesting": [
            "She helps separate durable social standing from warmer forms of acceptance or local dominance.",
            "Her profile is chapter-shaped rather than monotone, especially across Balbec and Guermantes material.",
        ],
        "primary_pattern": "prestige_positive_advantage_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Prestige-bearing Guermantes presence"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec counterweight and social cooling"},
            {"chapter_id": "v3-p2", "label": "Sharper local reversals after the initial rise"},
        ],
    },
    "Françoise": {
        "dek": "A frequent figure whose aggregate treatment is mostly negative, but with a few striking pockets of reversal and support.",
        "summary": "Françoise accumulates as a broadly negative figure across the corpus, though her profile is not flat: a small number of chapters briefly reverse the trend before the longer downward pull returns.",
        "why_interesting": [
            "She shows that a highly familiar household figure can remain structurally disadvantaged even while recurring constantly across volumes.",
            "Her chapter profile includes one especially visible positive pocket that makes the overall negativity more interpretable.",
        ],
        "primary_pattern": "broadly_negative_with_reversals",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "Core negative household concentration"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "The strongest temporary upward reversal"},
            {"chapter_id": "v3-p1", "label": "Return to pressure in Guermantes society"},
        ],
    },
    "Mme Verdurin": {
        "dek": "A substantial recurring presence whose aggregate treatment stays negative across all three lenses, especially in Swann-centered and later wartime material.",
        "summary": "Mme Verdurin is one of the clearest broadly negative salon figures in the corpus, with losses in advantage, prestige, and inclusion all reinforcing rather than offsetting one another.",
        "why_interesting": [
            "She is a useful contrast case to aristocratic figures whose rank can stay high even when scenes turn against them.",
            "Her chapter drivers show how one salon world can define a long-lasting social profile that still echoes later in the novel.",
        ],
        "primary_pattern": "consistently_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Main salon-world concentration"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Later wartime continuation"},
            {"chapter_id": "v4-p2", "label": "Intermediate pressure in the later social field"},
        ],
    },
    "Gilberte": {
        "dek": "A strongly prestige-positive figure whose advantage remains high while inclusion stays much more mixed and contingent.",
        "summary": "Gilberte is a compact but revealing cross-lens figure: she scores very well in prestige and immediate advantage, yet her inclusion profile remains markedly less secure.",
        "why_interesting": [
            "She is one of the best smaller-footprint examples of how social elevation and emotional incorporation can diverge.",
            "Her strongest shaping chapters are distributed across early Mme Swann material and later retrospective zones rather than one single dominant block.",
        ],
        "primary_pattern": "advantage_prestige_positive_inclusion_mixed",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Primary prestige and advantage concentration"},
            {"chapter_id": "v6-p2", "label": "Later inclusion strain and uneven return"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "Retrospective late counterweight"},
        ],
    },
    "Norpois": {
        "dek": "One of the corpus's clearest rhetorical winners, with very high advantage and prestige and comparatively little cross-lens instability.",
        "summary": "Norpois is a strongly positive figure across all three lenses, driven less by intimacy than by durable rhetorical authority and socially legible judgment.",
        "why_interesting": [
            "He is a useful example of a character whose strength comes from voice, sanction, and interpretive authority more than warmth.",
            "His profile shows how the project captures positive hierarchy without needing romantic or emotional centrality.",
        ],
        "primary_pattern": "rhetorically_ascendant",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Main field of diplomatic and rhetorical authority"},
            {"chapter_id": "v3-p1", "label": "Guermantes reinforcement of social weight"},
            {"chapter_id": "v6-p3", "label": "Later counterpressure against the earlier rise"},
        ],
    },
    "la grand-mère": {
        "dek": "A central emotional figure whose aggregate treatment is strikingly negative, especially in belonging and general appraisal.",
        "summary": "la grand-mère accumulates as one of the corpus's more strongly negative recurring figures, with the harshest pressure falling on inclusion and broad valuation rather than on a narrow prestige story alone.",
        "why_interesting": [
            "She shows how importance in the novel does not guarantee aggregate social or emotional advantage in the annotation surface.",
            "Her chapter spread distinguishes Balbec tenderness from later Guermantes and illness-weighted decline.",
        ],
        "primary_pattern": "emotionally_central_but_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Strong Guermantes-zone deterioration"},
            {"chapter_id": "v3-p2", "label": "Continuation of decline and pressure"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec countertexture with limited prestige recovery"},
        ],
    },
    "Bloch": {
        "dek": "A heavily annotated figure whose treatment is almost uniformly and intensely negative across all three lenses.",
        "summary": "Bloch is one of the clearest aggregate negative cases in the corpus, with repeated losses in advantage, prestige, and inclusion reinforcing each other rather than splitting apart.",
        "why_interesting": [
            "He is a high-frequency example of durable corpus-wide diminishment rather than a character defined by one isolated late collapse.",
            "His profile is especially useful for testing whether repeated comic or social discredit remains stable across volumes.",
        ],
        "primary_pattern": "uniformly_fallen",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Main concentration of ridicule and social diminishment"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec reinforcement of the same pattern"},
            {"chapter_id": "v4-p2", "label": "Later persistence in a changed social field"},
        ],
    },
    "duc de Guermantes": {
        "dek": "A major aristocratic figure whose aggregate profile is surprisingly negative across advantage, prestige, and inclusion alike.",
        "summary": "duc de Guermantes is one of the project's most revealing reversals of expectation: despite formal rank, his annotation surface is broadly negative across all three lenses.",
        "why_interesting": [
            "He helps separate nominal social title from actual passage-level advantage and valuation.",
            "His profile is a strong reminder that the corpus tracks enacted social force, not merely inherited station.",
        ],
        "primary_pattern": "rank_without_advantage",
        "reading_path": [
            {"chapter_id": "v3-p2", "label": "Main concentration of aggregate decline"},
            {"chapter_id": "v3-p1", "label": "Earlier Guermantes-stage weakening"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "Late retrospective continuation"},
        ],
    },
    "docteur Cottard": {
        "dek": "A recurrent social presence whose aggregate treatment stays modestly negative, with occasional prestige or advantage recoveries that never fully stabilize.",
        "summary": "docteur Cottard is a mid-tier negative figure whose profile is shaped by one strong Swann-world concentration, then complicated by smaller later recoveries and uneven prestige moments.",
        "why_interesting": [
            "He is useful as a moderate case: neither overwhelmingly central nor trivial, but persistent enough to show how local reversals can coexist with an overall downward pattern.",
            "His profile also distinguishes prestige blips from broader social and inclusion weakness.",
        ],
        "primary_pattern": "moderately_negative_with_recoveries",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Primary negative concentration"},
            {"chapter_id": "v4-p2", "label": "Later mixed prestige recovery"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Smaller positive countercurrent"},
        ],
    },
    "la mère du narrateur": {
        "dek": "A strongly positive figure whose gains are grounded less in public rank than in steady rhetorical and relational authority.",
        "summary": "la mère du narrateur is a quietly high-performing figure across all three lenses, with especially strong advantage and inclusion values driven by stable interpretive and familial force.",
        "why_interesting": [
            "She offers a model of positive social power that does not depend on aristocratic prestige or theatrical display.",
            "Her chapter distribution includes one notable Mme Swann-era counterweight, which makes the overall positivity more informative.",
        ],
        "primary_pattern": "quietly_ascendant",
        "reading_path": [
            {"chapter_id": "v6-p3", "label": "Strongest concentrated positive treatment"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Main negative counterweight"},
            {"chapter_id": "v1-p1-combray", "label": "Early domestic base of authority"},
        ],
    },
    "Bergotte": {
        "dek": "A highly positive literary figure whose advantage and prestige remain near the top of the corpus, with inclusion somewhat softer but still strong.",
        "summary": "Bergotte is one of the corpus's clearest positive symbolic figures, with his literary authority translating into very high advantage and prestige across several distinct narrative zones.",
        "why_interesting": [
            "He provides a clean case of aesthetic or intellectual distinction remaining socially effective across lenses.",
            "His profile is also useful because one large chapter block softens his inclusion without undoing the larger positive pattern.",
        ],
        "primary_pattern": "symbolically_ascendant",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Strong Guermantes-era literary authority"},
            {"chapter_id": "v1-p1-combray", "label": "Early concentration of esteem"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Longer mixed field with softened inclusion"},
        ],
    },
    "Legrandin": {
        "dek": "A character of recurrent social and rhetorical diminishment, with especially weak advantage and prestige despite occasional brief reversals.",
        "summary": "Legrandin is a broadly negative figure whose profile is shaped by repeated discredit and awkward self-positioning, even though a few isolated units briefly interrupt the downward pattern.",
        "why_interesting": [
            "He is a good example of how rhetorical self-fashioning can fail across more than one lens at once.",
            "His profile connects early Combray material to later Guermantes treatment without turning into a single monotone block.",
        ],
        "primary_pattern": "self_undermining_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Strongest later concentration of diminishment"},
            {"chapter_id": "v1-p1-combray", "label": "Early formation of the negative pattern"},
            {"chapter_id": "v6-p4", "label": "Brief late reversal against the broader trend"},
        ],
    },
    "Mme de Cambremer": {
        "dek": "A lower-frequency but consistently negative figure whose aggregate treatment remains weak across all three lenses.",
        "summary": "Mme de Cambremer is a compact but stable negative case: she does not dominate the corpus by volume, but what is there reads overwhelmingly downward in advantage, prestige, and inclusion.",
        "why_interesting": [
            "She helps test whether the analysis stays coherent on mid-sized characters whose signal is distributed across a few compact chapter zones.",
            "Her profile also shows how a small positive Balbec pocket can exist without changing the overall direction.",
        ],
        "primary_pattern": "compact_consistent_negative",
        "reading_path": [
            {"chapter_id": "v4-p2", "label": "Strongest later concentration of loss"},
            {"chapter_id": "v3-p1", "label": "Earlier Guermantes pressure"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Limited Balbec counterweight"},
        ],
    },
    "M. Vinteuil": {
        "dek": "A relatively small-footprint but strongly positive figure, especially in inclusion, whose profile mixes early negativity with later recovery and elevation.",
        "summary": "M. Vinteuil is one of the more surprising positive figures in the corpus: despite some strongly negative early material, his aggregate treatment ends up decisively positive, especially in inclusion.",
        "why_interesting": [
            "He is a good example of how the corpus can register rehabilitation or retrospective elevation rather than just cumulative damage.",
            "His profile is sharply chapter-shaped, which makes him useful for showing how aggregate positivity can arise from uneven terrain.",
        ],
        "primary_pattern": "rehabilitated_positive",
        "reading_path": [
            {"chapter_id": "v5", "label": "Main late positive recovery"},
            {"chapter_id": "v1-p1-combray", "label": "Early negative counterweight"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Intermediate positive reinforcement"},
        ],
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


def _score_run_outcomes(run_dir, lens="advantage"):
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


def score_run_advantage_outcomes(run_dir):
    return _score_run_outcomes(run_dir, lens="advantage")


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


def _normalize_character_name(character, character_name_map=None):
    if not character_name_map:
        return character
    return character_name_map.get(character, character)


def _normalize_character_name_map(character_name_map):
    if not character_name_map:
        return {}

    normalized_map = {}
    for source, target in character_name_map.items():
        clean_source = _clean_character_name(source)
        clean_target = _clean_character_name(target)
        if not clean_source or not clean_target or clean_source == clean_target:
            continue
        normalized_map[clean_source] = clean_target
    return normalized_map


def _merge_unit_character_scores(score_maps, lens_config):
    merged = {
        "event_score": 0.0,
        "status_score": 0.0,
        "ambiguity_penalty": 0.0,
        "status_dimensions": {},
        "event_types": {},
        "positive_event_count": 0,
        "negative_event_count": 0,
    }

    for scores in score_maps:
        merged["event_score"] += scores["event_score"]
        merged["status_score"] += scores["status_score"]
        merged["ambiguity_penalty"] = max(merged["ambiguity_penalty"], scores["ambiguity_penalty"])
        merged["positive_event_count"] += scores["positive_event_count"]
        merged["negative_event_count"] += scores["negative_event_count"]
        for dimension, delta_total in scores["status_dimensions"].items():
            merged["status_dimensions"][dimension] = merged["status_dimensions"].get(dimension, 0) + delta_total
        for event_type, count in scores["event_types"].items():
            merged["event_types"][event_type] = merged["event_types"].get(event_type, 0) + count

    merged["event_score"] = round(merged["event_score"], 3)
    merged["status_score"] = round(merged["status_score"], 3)
    merged["ambiguity_penalty"] = round(merged["ambiguity_penalty"], 3)
    merged["net_score"] = round(
        merged["event_score"] + merged["status_score"] - merged["ambiguity_penalty"],
        3,
    )
    merged["label"] = _outcome_label(merged["net_score"], lens_config)
    return merged


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


def build_outcome_report(run_dir, lens="advantage", character_name_map=None):
    score_summary = _score_run_outcomes(run_dir, lens=lens)
    lens_config = SCORING_LENS_CONFIGS[lens]
    character_name_map = _normalize_character_name_map(character_name_map)
    units = []
    character_summaries = {}

    for unit in score_summary["units"]:
        unit_id = unit["unit_id"]
        normalized_unit_scores = defaultdict(list)
        for character, scores in unit["characters"].items():
            normalized_character = _normalize_character_name(character, character_name_map)
            normalized_unit_scores[normalized_character].append(scores)

        for character, score_maps in normalized_unit_scores.items():
            scores = (
                score_maps[0]
                if len(score_maps) == 1
                else _merge_unit_character_scores(score_maps, lens_config)
            )
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
        "character_normalization": {
            "applied": bool(character_name_map),
            "map": dict(sorted(character_name_map.items())),
        },
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


def build_corpus_sanity_review(run_dirs, character_name_map=None):
    if not run_dirs:
        raise ValueError("At least one run directory is required for a corpus sanity review.")

    character_name_map = _normalize_character_name_map(character_name_map)
    run_statuses = []
    run_reports = {}

    for run_dir in run_dirs:
        status = get_run_status(run_dir)
        manifest = status["manifest"]
        run_id = manifest["run_id"]
        run_statuses.append(status)
        run_reports[run_id] = {
            lens: build_outcome_report(run_dir, lens=lens, character_name_map=character_name_map)
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

        advantage_report = run_reports[run_id]["advantage"]
        unit_character_counts = defaultdict(int)
        for entry in advantage_report["timeline"]:
            unit_character_counts[entry["unit_id"]] += 1

        unit_count = len(manifest["unit_ids"])
        scored_unit_count = advantage_report["scored_unit_count"]
        unique_character_count = advantage_report["character_count"]
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
        character_totals = sorted(
            (
                {
                    "character": totals["character"],
                    "net_score": round(totals["net_score"], 3),
                    "event_score": round(totals["event_score"], 3),
                    "status_score": round(totals["status_score"], 3),
                    "unit_count": totals["unit_count"],
                    "labels": totals["labels"],
                    "dominant_status_dimension": _dominant_status_dimension(totals["status_dimensions"]),
                }
                for totals in lens_character_totals[lens].values()
            ),
            key=lambda item: (-item["net_score"], item["character"]),
        )

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
        character_volatility = sorted(volatility_rows, key=lambda item: item["character"])

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
            "character_totals": character_totals,
            "character_volatility": character_volatility,
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
        "character_normalization": {
            "applied": bool(character_name_map),
            "map": dict(sorted(character_name_map.items())),
        },
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


def _character_ranks(character_totals, reverse=True):
    ordered = sorted(
        character_totals,
        key=lambda item: ((-item["net_score"]) if reverse else item["net_score"], item["character"]),
    )
    return {row["character"]: index + 1 for index, row in enumerate(ordered)}


def _character_totals_by_name(character_totals):
    return {row["character"]: row for row in character_totals}


def _character_volatility_by_name(character_volatility):
    return {row["character"]: row for row in character_volatility}


def _rank_to_percentile(rank, population_size):
    if rank is None or population_size <= 0:
        return None
    if population_size == 1:
        return 100
    return round(((population_size - rank) / (population_size - 1)) * 100)


def build_character_cross_lens_analysis(review):
    lenses = sorted(SCORING_LENS_CONFIGS)
    missing_lenses = [lens for lens in lenses if lens not in review["lens_reviews"]]
    if missing_lenses:
        raise ValueError(f"Review is missing lens reviews for: {', '.join(missing_lenses)}")

    rank_maps = {
        lens: _character_ranks(review["lens_reviews"][lens]["character_totals"], reverse=True)
        for lens in lenses
    }
    totals_by_lens = {
        lens: _character_totals_by_name(review["lens_reviews"][lens]["character_totals"])
        for lens in lenses
    }
    volatility_by_lens = {
        lens: _character_volatility_by_name(review["lens_reviews"][lens]["character_volatility"])
        for lens in lenses
    }
    population_sizes = {
        lens: len(review["lens_reviews"][lens]["character_totals"])
        for lens in lenses
    }

    all_characters = sorted(
        {
            character
            for lens in lenses
            for character in totals_by_lens[lens]
        }
    )

    character_rows = []
    for character in all_characters:
        lens_rows = {}
        ranks = []
        unit_counts = []
        score_spans = []
        for lens in lenses:
            total = totals_by_lens[lens].get(character)
            volatility = volatility_by_lens[lens].get(character)
            rank = rank_maps[lens].get(character)
            if rank is not None:
                ranks.append(rank)
            if total is not None:
                unit_counts.append(total["unit_count"])
            if volatility is not None:
                score_spans.append(volatility["score_span"])
            lens_rows[lens] = {
                "net_score": total["net_score"] if total else 0.0,
                "rank": rank,
                "percentile": _rank_to_percentile(rank, population_sizes[lens]),
                "unit_count": total["unit_count"] if total else 0,
                "dominant_status_dimension": total["dominant_status_dimension"] if total else None,
                "score_span": volatility["score_span"] if volatility else 0.0,
                "mean_score": volatility["mean_score"] if volatility else 0.0,
            }

        character_rows.append(
            {
                "character": character,
                "lens_scores": lens_rows,
                "rank_spread": (max(ranks) - min(ranks)) if ranks else 0,
                "max_unit_count": max(unit_counts) if unit_counts else 0,
                "max_score_span": max(score_spans) if score_spans else 0.0,
            }
        )

    top_rank_spread = sorted(
        [row for row in character_rows if row["max_unit_count"] >= 2],
        key=lambda item: (-item["rank_spread"], -item["max_unit_count"], item["character"]),
    )[:15]
    top_volatility = sorted(
        [row for row in character_rows if row["max_unit_count"] >= 2],
        key=lambda item: (-item["max_score_span"], -item["max_unit_count"], item["character"]),
    )[:15]

    return {
        "character_cross_lens_analysis_version": "character_cross_lens_analysis_v1",
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "character_count": len(character_rows),
        "characters": sorted(
            character_rows,
            key=lambda item: (
                -item["lens_scores"]["advantage"]["net_score"],
                item["character"],
            ),
        ),
        "top_rank_spread_characters": top_rank_spread,
        "top_volatile_characters": top_volatility,
        "top_positive_by_lens": {
            lens: review["lens_reviews"][lens]["top_positive_characters"]
            for lens in lenses
        },
        "top_negative_by_lens": {
            lens: review["lens_reviews"][lens]["top_negative_characters"]
            for lens in lenses
        },
    }


def _chapter_id_from_unit_id(unit_id):
    return unit_id.split("#", 1)[0]


def _run_id_sort_key(run_id):
    match = re.search(r"(\d+)$", run_id)
    return (int(match.group(1)), run_id) if match else (-1, run_id)


def _paragraph_range_from_unit_id(unit_id):
    _, paragraph_spec = unit_id.split("#", 1)
    matches = [int(value) for value in re.findall(r"p-(\d+)", paragraph_spec)]
    if not matches:
        raise ValueError(f"Unit id does not contain a paragraph range: {unit_id}")
    if len(matches) == 1:
        return matches[0], matches[0]
    return matches[0], matches[-1]


def _overlay_character_sort_key(character_row):
    return (
        -max(
            abs(character_row["advantage"]["netScore"]),
            abs(character_row["prestige"]["netScore"]),
            abs(character_row["inclusion"]["netScore"]),
        ),
        character_row["character"],
    )


def _overlay_dominant_character(characters):
    if not characters:
        return None

    def _key(row):
        return (
            abs(row["advantage"]["netScore"]),
            abs(row["prestige"]["netScore"]),
            abs(row["inclusion"]["netScore"]),
            row["character"],
        )

    return max(characters, key=_key)["character"]


OVERLAY_DIMENSION_PHRASES = {
    "general_appraisal": "overall standing",
    "social_status": "social status",
    "rhetorical_position": "rhetorical authority",
    "emotional_position": "emotional standing",
    "inclusion_exclusion": "inclusion standing",
}


def _natural_join(items):
    values = [item for item in items if item]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _overlay_dimension_phrase(dimension):
    return OVERLAY_DIMENSION_PHRASES.get(dimension, "standing")


def _overlay_lens_group_phrase(lenses):
    if set(lenses) == set(sorted(SCORING_LENS_CONFIGS)):
        return "across all three lenses"
    return f"in {_natural_join(lenses)}"


def _build_overlay_character_summary(character_row):
    dimension_phrase = _overlay_dimension_phrase(character_row.get("dominantStatusDimension"))
    labels_by_lens = {lens: character_row[lens]["label"] for lens in sorted(SCORING_LENS_CONFIGS)}
    grouped_lenses = defaultdict(list)
    for lens, label in labels_by_lens.items():
        grouped_lenses[label].append(lens)

    if len(grouped_lenses) == 1:
        only_label = next(iter(grouped_lenses))
        if only_label == "win":
            return f"{character_row['character']} gains {dimension_phrase} across all three lenses."
        if only_label == "loss":
            return f"{character_row['character']} loses {dimension_phrase} across all three lenses."
        if only_label == "mixed":
            return f"{character_row['character']} shows mixed {dimension_phrase} across all three lenses."
        return f"{character_row['character']} remains neutral across all three lenses."

    clauses = []
    if grouped_lenses.get("win"):
        clauses.append(
            f"gains {dimension_phrase} {_overlay_lens_group_phrase(grouped_lenses['win'])}"
        )
    if grouped_lenses.get("loss"):
        clauses.append(
            f"loses {dimension_phrase} {_overlay_lens_group_phrase(grouped_lenses['loss'])}"
        )
    if grouped_lenses.get("mixed"):
        clauses.append(
            f"shows mixed {dimension_phrase} {_overlay_lens_group_phrase(grouped_lenses['mixed'])}"
        )
    if not clauses:
        clauses.append("remains neutral")

    return f"{character_row['character']} " + "; ".join(clauses) + "."


def _build_overlay_unit_summary(character_rows):
    if not character_rows:
        return "No character scoring data is available for this unit."

    active_rows = [
        row
        for row in character_rows
        if any(row[lens]["label"] != "neutral" for lens in sorted(SCORING_LENS_CONFIGS))
    ]
    source_rows = active_rows[:2] if active_rows else character_rows[:1]
    return " ".join(_build_overlay_character_summary(row) for row in source_rows)


def _chapter_lens_mode(label_counts):
    if label_counts["loss"] > max(label_counts["win"], label_counts["mixed"], label_counts["neutral"]):
        return "loss-heavy"
    if label_counts["win"] > max(label_counts["loss"], label_counts["mixed"], label_counts["neutral"]):
        return "win-heavy"
    if label_counts["mixed"] > max(label_counts["win"], label_counts["loss"], label_counts["neutral"]):
        return "mixed"
    return "balanced"


def _build_overlay_chapter_summary(units):
    if not units:
        return "No reviewed annotation units are currently available for this chapter."

    dominant_character_counts = defaultdict(int)
    lens_label_counts = {lens: {"win": 0, "loss": 0, "mixed": 0, "neutral": 0} for lens in sorted(SCORING_LENS_CONFIGS)}
    for unit in units:
        if unit["dominantCharacter"]:
            dominant_character_counts[unit["dominantCharacter"]] += 1
        for character_row in unit["characters"]:
            for lens in sorted(SCORING_LENS_CONFIGS):
                lens_label_counts[lens][character_row[lens]["label"]] += 1

    top_characters = [
        character
        for character, _count in sorted(
            dominant_character_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    ]
    chapter_focus = (
        f"centered on {_natural_join(top_characters)}"
        if top_characters
        else "without a single dominant character focus"
    )
    lens_modes = [
        f"{lens} {_chapter_lens_mode(lens_label_counts[lens])}"
        for lens in sorted(SCORING_LENS_CONFIGS)
    ]
    return (
        f"This chapter contains {len(units)} annotated units, {chapter_focus}. "
        f"Overall it is {_natural_join(lens_modes)}."
    )


def _slugify_text(value):
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower()).strip("-")
    return cleaned or "item"


def _reader_chapter_link(chapter_id, paragraph_start=None):
    link = f"{ISLT_READER_BASE_PATH}/{chapter_id}"
    if paragraph_start is not None:
        link += f"#p-{paragraph_start}"
    return link


def _chapter_title_map():
    return {chapter.id: chapter.title for chapter in CANONICAL_CHAPTER_SPECS}


def _discover_character_portraits(character):
    portrait_slug = CHARACTER_PORTRAIT_SLUGS.get(character, _slugify_text(character))
    if not ISLT_PORTRAITS_DIR.exists():
        return {"default": None, "variants": []}

    variants = []
    for path in sorted(ISLT_PORTRAITS_DIR.glob(f"{portrait_slug}-*.png")):
        stem = path.stem
        prefix = f"{portrait_slug}-"
        if not stem.startswith(prefix):
            continue
        body = stem[len(prefix) :]
        match = re.match(r"(.+)-(\d{8}|[0-9a-f]{8})-(\d{4,})$", body)
        if not match:
            continue
        descriptor = match.group(1)
        variant = "default"
        style = None
        for candidate in PORTRAIT_STYLES:
            suffix = f"-{candidate}"
            if descriptor == candidate:
                style = candidate
                variant = "default"
                break
            if descriptor.endswith(suffix):
                style = candidate
                variant = descriptor[: -len(suffix)]
                break
        if style is None:
            descriptor_parts = descriptor.split("-", 1)
            variant = descriptor_parts[0]
            style = descriptor_parts[1] if len(descriptor_parts) > 1 else "unknown"
        variants.append(
            {
                "variant": variant,
                "style": style,
                "src": f"/projects/islt/portraits/{path.name}",
            }
        )

    default_src = None
    preferred_default = next(
        (
            row["src"]
            for row in variants
            if row["variant"] == "default" and row["style"] == "vermeer-proustian"
        ),
        None,
    )
    if preferred_default:
        default_src = preferred_default
    elif variants:
        default_src = variants[0]["src"]

    return {"default": default_src, "variants": variants}


def _build_character_page_notable_units(character, overlay_dataset, limit=3):
    rows = []
    for chapter in overlay_dataset["chapters"]:
        for unit in chapter["units"]:
            for character_row in unit["characters"]:
                if character_row["character"] != character:
                    continue
                rows.append(
                    {
                        "unit_id": unit["unitId"],
                        "chapter_id": chapter["chapterId"],
                        "paragraph_start": unit["paragraphStart"],
                        "summary": unit["summary"],
                        "max_abs_score": max(
                            abs(character_row["advantage"]["netScore"]),
                            abs(character_row["prestige"]["netScore"]),
                            abs(character_row["inclusion"]["netScore"]),
                        ),
                    }
                )

    rows.sort(key=lambda item: (-item["max_abs_score"], item["unit_id"]))
    return [
        {
            "unit_id": row["unit_id"],
            "label": row["summary"],
            "reader_link": _reader_chapter_link(row["chapter_id"], paragraph_start=row["paragraph_start"]),
        }
        for row in rows[:limit]
    ]


def build_character_pages(run_dirs, character_name_map=None, target_characters=None, top_chapter_limit=5):
    selected_characters = list(target_characters or CHARACTER_PAGE_PILOT_EDITORIAL.keys())
    review = build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    profile_cards = build_character_profile_cards(
        run_dirs,
        character_name_map=character_name_map,
        top_chapter_limit=top_chapter_limit,
    )
    chapter_analysis = build_character_chapter_analysis(
        run_dirs,
        character_name_map=character_name_map,
        target_characters=selected_characters,
    )
    overlay_dataset = build_chapter_overlay_data(run_dirs, character_name_map=character_name_map)
    chapter_titles = _chapter_title_map()

    cards_by_character = {row["character"]: row for row in profile_cards["cards"]}
    chapter_rows_by_character = {row["character"]: row for row in chapter_analysis["characters"]}

    pages = []
    for character in selected_characters:
        if character not in cards_by_character:
            raise ValueError(f"Character page target not found in derived profile data: {character}")
        if character not in CHARACTER_PAGE_PILOT_EDITORIAL:
            raise ValueError(f"Character page editorial data is missing for: {character}")

        card = cards_by_character[character]
        chapter_rows = chapter_rows_by_character.get(character, {}).get("chapters", [])
        top_chapters = sorted(
            chapter_rows,
            key=lambda item: max(
                abs(item["advantage"]["net_score"]),
                abs(item["prestige"]["net_score"]),
                abs(item["inclusion"]["net_score"]),
            ),
            reverse=True,
        )[:top_chapter_limit]
        editorial = CHARACTER_PAGE_PILOT_EDITORIAL[character]

        pages.append(
            {
                "character": character,
                "slug": _slugify_text(character),
                "portrait": _discover_character_portraits(character),
                "profile": {
                    "annotation_unit_count": card["annotation_unit_count"],
                    "rank_spread": card["rank_spread"],
                    "max_score_span": card["max_score_span"],
                    "selected_by": card["selected_by"],
                    "lens_scores": card["lens_scores"],
                },
                "editorial": {
                    "dek": editorial["dek"],
                    "summary": editorial["summary"],
                    "why_interesting": editorial["why_interesting"],
                    "primary_pattern": editorial["primary_pattern"],
                },
                "top_chapters": [
                    {
                        "chapter_id": row["chapter_id"],
                        "chapter_title": chapter_titles.get(row["chapter_id"], row["chapter_id"]),
                        "advantage": row["advantage"],
                        "prestige": row["prestige"],
                        "inclusion": row["inclusion"],
                        "reader_link": _reader_chapter_link(row["chapter_id"]),
                    }
                    for row in top_chapters
                ],
                "reading_path": [
                    {
                        "chapter_id": row["chapter_id"],
                        "label": row["label"],
                        "reader_link": _reader_chapter_link(row["chapter_id"]),
                    }
                    for row in editorial["reading_path"]
                ],
                "notable_units": _build_character_page_notable_units(character, overlay_dataset),
            }
        )

    pages.sort(
        key=lambda item: (
            -item["profile"]["annotation_unit_count"],
            -item["profile"]["rank_spread"],
            item["character"],
        )
    )
    return {
        "character_pages_version": "character_pages_v1",
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "character_count": len(pages),
        "pages": pages,
    }


def build_chapter_overlay_data(run_dirs, character_name_map=None):
    if not run_dirs:
        raise ValueError("At least one run directory is required for chapter overlay export.")

    character_name_map = _normalize_character_name_map(character_name_map)
    timeline_by_lens = {lens: [] for lens in sorted(SCORING_LENS_CONFIGS)}
    preferred_run_by_unit = {}

    for run_dir in run_dirs:
        status = get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for unit in status["units"]:
            unit_id = unit["unit_id"]
            if unit["review_state"] != "reviewed":
                continue
            existing_run_id = preferred_run_by_unit.get(unit_id)
            if existing_run_id is None or _run_id_sort_key(run_id) > _run_id_sort_key(existing_run_id):
                preferred_run_by_unit[unit_id] = run_id

    for run_dir in run_dirs:
        status = get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for lens in sorted(SCORING_LENS_CONFIGS):
            report = build_outcome_report(run_dir, lens=lens, character_name_map=character_name_map)
            timeline_by_lens[lens].extend(
                entry for entry in report["timeline"] if preferred_run_by_unit.get(entry["unit_id"]) == run_id
            )

    unit_rows = defaultdict(lambda: {"characters": {}})
    for lens, entries in timeline_by_lens.items():
        for entry in entries:
            unit_id = entry["unit_id"]
            chapter_id = _chapter_id_from_unit_id(unit_id)
            paragraph_start, paragraph_end = _paragraph_range_from_unit_id(unit_id)
            chapter_bucket = unit_rows[chapter_id]
            chapter_bucket["characters"].setdefault(unit_id, {})
            character_bucket = chapter_bucket["characters"][unit_id].setdefault(
                entry["character"],
                {
                    "character": entry["character"],
                    "dominantStatusDimension": entry["dominant_status_dimension"],
                },
            )
            character_bucket[lens] = {
                "netScore": entry["net_score"],
                "label": entry["label"],
            }
            chapter_bucket.setdefault("unit_meta", {})[unit_id] = {
                "paragraphStart": paragraph_start,
                "paragraphEnd": paragraph_end,
            }

    chapters = []
    manifest_rows = []
    for chapter in CANONICAL_CHAPTER_SPECS:
        chapter_unit_map = unit_rows.get(chapter.id, {})
        unit_meta = chapter_unit_map.get("unit_meta", {})
        units = []
        for unit_id, characters in sorted(
            chapter_unit_map.get("characters", {}).items(),
            key=lambda item: (
                unit_meta[item[0]]["paragraphStart"],
                unit_meta[item[0]]["paragraphEnd"],
                item[0],
            ),
        ):
            character_rows = []
            for character in characters.values():
                character_row = {
                    "character": character["character"],
                    "dominantStatusDimension": character["dominantStatusDimension"],
                }
                for lens in sorted(SCORING_LENS_CONFIGS):
                    character_row[lens] = character.get(lens, {"netScore": 0.0, "label": "neutral"})
                character_rows.append(character_row)

            character_rows.sort(key=_overlay_character_sort_key)
            units.append(
                {
                    "unitId": unit_id,
                    "paragraphStart": unit_meta[unit_id]["paragraphStart"],
                    "paragraphEnd": unit_meta[unit_id]["paragraphEnd"],
                    "dominantCharacter": _overlay_dominant_character(character_rows),
                    "characters": character_rows,
                    "summary": _build_overlay_unit_summary(character_rows),
                }
            )

        chapter_payload = {
            "chapter_overlay_version": "chapter_overlay_v2",
            "chapterId": chapter.id,
            "chapterNumber": chapter.number,
            "title": chapter.title,
            "volumeNumber": chapter.volume_number,
            "volumeTitle": chapter.volume_title,
            "partNumber": chapter.part_number,
            "partTitle": chapter.part_title,
            "sectionTitle": chapter.section_title,
            "characterNormalizationApplied": bool(character_name_map),
            "summary": _build_overlay_chapter_summary(units),
            "units": units,
        }
        chapters.append(chapter_payload)
        manifest_rows.append(
            {
                "chapterId": chapter.id,
                "title": chapter.title,
                "path": f"chapters/{chapter.id}.json",
                "unitCount": len(units),
                "characterCount": len({row["character"] for unit in units for row in unit["characters"]}),
            }
        )

    return {
        "chapter_overlay_version": "chapter_overlay_v2",
        "source_review_version": "corpus_sanity_review_v1",
        "character_normalization": {
            "applied": bool(character_name_map),
            "map": dict(sorted(character_name_map.items())),
        },
        "chapter_count": len(chapters),
        "duplicate_resolution": "latest_reviewed_run_wins",
        "chapters": chapters,
        "manifest": {
            "chapter_overlay_version": "chapter_overlay_v2",
            "source_review_version": "corpus_sanity_review_v1",
            "character_normalization": {
                "applied": bool(character_name_map),
                "map": dict(sorted(character_name_map.items())),
            },
            "chapter_count": len(chapters),
            "duplicate_resolution": "latest_reviewed_run_wins",
            "chapters": manifest_rows,
        },
    }


def build_character_chapter_analysis(
    run_dirs,
    character_name_map=None,
    target_characters=None,
    top_rank_spread_limit=10,
    top_volatile_limit=10,
):
    review = build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    cross_lens = build_character_cross_lens_analysis(review)

    if target_characters:
        selected_characters = list(dict.fromkeys(target_characters))
    else:
        selected_characters = []
        seen = set()
        for row in cross_lens["top_rank_spread_characters"][:top_rank_spread_limit]:
            character = row["character"]
            if character not in seen:
                seen.add(character)
                selected_characters.append(character)
        for row in cross_lens["top_volatile_characters"][:top_volatile_limit]:
            character = row["character"]
            if character not in seen:
                seen.add(character)
                selected_characters.append(character)

    if not selected_characters:
        raise ValueError("At least one target character is required for chapter analysis.")

    chapter_order = [chapter.id for chapter in CANONICAL_CHAPTER_SPECS]
    chapter_positions = {chapter_id: index for index, chapter_id in enumerate(chapter_order)}
    lens_reports = {lens: [] for lens in sorted(SCORING_LENS_CONFIGS)}
    for run_dir in run_dirs:
        for lens in sorted(SCORING_LENS_CONFIGS):
            lens_reports[lens].append(build_outcome_report(run_dir, lens=lens, character_name_map=character_name_map))

    selected_set = set(selected_characters)
    chapter_rows = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"net_score": 0.0, "unit_count": 0})))

    for lens, reports in lens_reports.items():
        for report in reports:
            for entry in report["timeline"]:
                character = entry["character"]
                if character not in selected_set:
                    continue
                chapter_id = _chapter_id_from_unit_id(entry["unit_id"])
                chapter_row = chapter_rows[character][chapter_id][lens]
                chapter_row["net_score"] += entry["net_score"]
                chapter_row["unit_count"] += 1

    selected_by = {}
    for row in cross_lens["top_rank_spread_characters"][:top_rank_spread_limit]:
        selected_by.setdefault(row["character"], []).append("rank_spread")
    for row in cross_lens["top_volatile_characters"][:top_volatile_limit]:
        selected_by.setdefault(row["character"], []).append("volatility")

    characters = []
    for character in selected_characters:
        source_row = next(row for row in cross_lens["characters"] if row["character"] == character)
        chapters = []
        for chapter_id, chapter_lenses in sorted(
            chapter_rows.get(character, {}).items(),
            key=lambda item: (chapter_positions.get(item[0], 999), item[0]),
        ):
            chapters.append(
                {
                    "chapter_id": chapter_id,
                    "advantage": {
                        "net_score": round(chapter_lenses["advantage"]["net_score"], 3),
                        "unit_count": chapter_lenses["advantage"]["unit_count"],
                    },
                    "prestige": {
                        "net_score": round(chapter_lenses["prestige"]["net_score"], 3),
                        "unit_count": chapter_lenses["prestige"]["unit_count"],
                    },
                    "inclusion": {
                        "net_score": round(chapter_lenses["inclusion"]["net_score"], 3),
                        "unit_count": chapter_lenses["inclusion"]["unit_count"],
                    },
                }
            )

        characters.append(
            {
                "character": character,
                "selected_by": selected_by.get(character, []),
                "cross_lens_summary": source_row,
                "chapters": chapters,
            }
        )

    return {
        "character_chapter_analysis_version": "character_chapter_analysis_v1",
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "source_review_version": review["corpus_review_version"],
        "selected_character_count": len(characters),
        "selected_characters": selected_characters,
        "characters": characters,
    }


def build_character_annotation_counts(review):
    advantage_totals = review["lens_reviews"]["advantage"]["character_totals"]
    rows = []
    for row in advantage_totals:
        rows.append(
            {
                "character": row["character"],
                "annotation_unit_count": row["unit_count"],
                "advantage_net_score": row["net_score"],
                "prestige_net_score": next(
                    item["net_score"]
                    for item in review["lens_reviews"]["prestige"]["character_totals"]
                    if item["character"] == row["character"]
                ),
                "inclusion_net_score": next(
                    item["net_score"]
                    for item in review["lens_reviews"]["inclusion"]["character_totals"]
                    if item["character"] == row["character"]
                ),
            }
        )

    rows.sort(key=lambda item: (-item["annotation_unit_count"], item["character"]))
    return {
        "character_annotation_counts_version": "character_annotation_counts_v1",
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "character_count": len(rows),
        "characters": rows,
    }


def build_character_profile_cards(run_dirs, character_name_map=None, top_chapter_limit=5):
    review = build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    cross_lens = build_character_cross_lens_analysis(review)
    chapter_analysis = build_character_chapter_analysis(
        run_dirs,
        character_name_map=character_name_map,
        target_characters=[row["character"] for row in cross_lens["characters"]],
    )
    annotation_counts = build_character_annotation_counts(review)

    chapter_rows_by_character = {row["character"]: row for row in chapter_analysis["characters"]}
    counts_by_character = {row["character"]: row for row in annotation_counts["characters"]}

    cards = []
    for row in cross_lens["characters"]:
        character = row["character"]
        chapter_rows = chapter_rows_by_character.get(character, {}).get("chapters", [])
        top_chapters = sorted(
            chapter_rows,
            key=lambda item: max(
                abs(item["advantage"]["net_score"]),
                abs(item["prestige"]["net_score"]),
                abs(item["inclusion"]["net_score"]),
            ),
            reverse=True,
        )[:top_chapter_limit]

        cards.append(
            {
                "character": character,
                "annotation_unit_count": counts_by_character[character]["annotation_unit_count"],
                "rank_spread": row["rank_spread"],
                "max_score_span": row["max_score_span"],
                "selected_by": chapter_rows_by_character.get(character, {}).get("selected_by", []),
                "lens_scores": row["lens_scores"],
                "top_chapters": top_chapters,
            }
        )

    cards.sort(
        key=lambda item: (-item["annotation_unit_count"], -item["rank_spread"], item["character"]),
    )
    return {
        "character_profile_cards_version": "character_profile_cards_v1",
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "character_count": len(cards),
        "cards": cards,
    }


def build_corpus_review_normalization_diff(before_review, after_review):
    after_map = after_review.get("character_normalization", {}).get("map", {})
    if not after_map:
        raise ValueError("A normalized corpus review is required to build a normalization diff.")

    normalized_targets = sorted(set(after_map.values()))
    lens_diffs = {}
    for lens in sorted(SCORING_LENS_CONFIGS):
        before_lens = before_review["lens_reviews"][lens]
        after_lens = after_review["lens_reviews"][lens]
        before_totals = _character_totals_by_name(before_lens["character_totals"])
        after_totals = _character_totals_by_name(after_lens["character_totals"])
        positive_ranks_before = _character_ranks(before_lens["character_totals"], reverse=True)
        positive_ranks_after = _character_ranks(after_lens["character_totals"], reverse=True)
        negative_ranks_before = _character_ranks(before_lens["character_totals"], reverse=False)
        negative_ranks_after = _character_ranks(after_lens["character_totals"], reverse=False)

        normalized_character_rows = []
        for target in normalized_targets:
            sources = sorted(source for source, normalized in after_map.items() if normalized == target)
            before_net_score = round(
                sum(before_totals.get(name, {}).get("net_score", 0.0) for name in [target, *sources]),
                3,
            )
            before_unit_count = sum(before_totals.get(name, {}).get("unit_count", 0) for name in [target, *sources])
            after_row = after_totals.get(target)
            if not after_row and before_unit_count == 0:
                continue

            normalized_character_rows.append(
                {
                    "character": target,
                    "merged_from": sources,
                    "net_score_before": before_net_score,
                    "net_score_after": after_row["net_score"] if after_row else 0.0,
                    "unit_count_before": before_unit_count,
                    "unit_count_after": after_row["unit_count"] if after_row else 0,
                    "positive_rank_before": positive_ranks_before.get(target),
                    "positive_rank_after": positive_ranks_after.get(target),
                    "negative_rank_before": negative_ranks_before.get(target),
                    "negative_rank_after": negative_ranks_after.get(target),
                }
            )

        lens_diffs[lens] = {
            "character_count_before": before_lens["character_count"],
            "character_count_after": after_lens["character_count"],
            "top_positive_before": before_lens["top_positive_characters"],
            "top_positive_after": after_lens["top_positive_characters"],
            "top_negative_before": before_lens["top_negative_characters"],
            "top_negative_after": after_lens["top_negative_characters"],
            "normalized_characters": normalized_character_rows,
        }

    before_cross_lens = before_review["cross_lens_summary"]
    after_cross_lens = after_review["cross_lens_summary"]
    return {
        "normalization_diff_version": "corpus_review_normalization_diff_v1",
        "character_normalization_map": dict(sorted(after_map.items())),
        "lens_diffs": lens_diffs,
        "cross_lens_summary_diff": {
            "comparable_entry_count_before": before_cross_lens["comparable_entry_count"],
            "comparable_entry_count_after": after_cross_lens["comparable_entry_count"],
            "label_disagreement_count_before": before_cross_lens["label_disagreement_count"],
            "label_disagreement_count_after": after_cross_lens["label_disagreement_count"],
            "direction_disagreement_count_before": before_cross_lens["direction_disagreement_count"],
            "direction_disagreement_count_after": after_cross_lens["direction_disagreement_count"],
            "sign_flip_count_before": len(before_cross_lens["sign_flip_examples"]),
            "sign_flip_count_after": len(after_cross_lens["sign_flip_examples"]),
        },
        "before_review_version": before_review["corpus_review_version"],
        "after_review_version": after_review["corpus_review_version"],
    }


def render_corpus_review_normalization_diff_markdown(diff):
    lines = [
        "# Corpus Review Normalization Diff",
        "",
        f"- Diff version: `{diff['normalization_diff_version']}`",
        f"- Reviewed merges: `{len(diff['character_normalization_map'])}`",
        "",
        "## Character Map",
        "",
        _markdown_table(
            ["Source Name", "Normalized Name"],
            diff["character_normalization_map"].items(),
        ),
        "",
        "## Lens Diffs",
        "",
    ]

    for lens, lens_diff in diff["lens_diffs"].items():
        lines.extend(
            [
                f"### {lens}",
                "",
                f"- Character count: `{lens_diff['character_count_before']}` -> `{lens_diff['character_count_after']}`",
                "",
                "Normalized character movement:",
                "",
                _markdown_table(
                    [
                        "Character",
                        "Merged From",
                        "Net Before",
                        "Net After",
                        "Units Before",
                        "Units After",
                        "Positive Rank",
                        "Negative Rank",
                    ],
                    [
                        (
                            row["character"],
                            ", ".join(row["merged_from"]),
                            _format_signed_number(row["net_score_before"]),
                            _format_signed_number(row["net_score_after"]),
                            row["unit_count_before"],
                            row["unit_count_after"],
                            f"{row['positive_rank_before']} -> {row['positive_rank_after']}",
                            f"{row['negative_rank_before']} -> {row['negative_rank_after']}",
                        )
                        for row in lens_diff["normalized_characters"]
                    ],
                ),
                "",
                "Top positive characters:",
                "",
                _markdown_table(
                    ["Before", "After"],
                    [
                        (
                            row_before["character"] if index < len(lens_diff["top_positive_before"]) else "",
                            row_after["character"] if index < len(lens_diff["top_positive_after"]) else "",
                        )
                        for index, (row_before, row_after) in enumerate(
                            zip(
                                lens_diff["top_positive_before"] + [{}] * 10,
                                lens_diff["top_positive_after"] + [{}] * 10,
                            )
                        )
                        if index < 10
                    ],
                ),
                "",
                "Top negative characters:",
                "",
                _markdown_table(
                    ["Before", "After"],
                    [
                        (
                            row_before["character"] if index < len(lens_diff["top_negative_before"]) else "",
                            row_after["character"] if index < len(lens_diff["top_negative_after"]) else "",
                        )
                        for index, (row_before, row_after) in enumerate(
                            zip(
                                lens_diff["top_negative_before"] + [{}] * 10,
                                lens_diff["top_negative_after"] + [{}] * 10,
                            )
                        )
                        if index < 10
                    ],
                ),
                "",
            ]
        )

    cross_lens_diff = diff["cross_lens_summary_diff"]
    lines.extend(
        [
            "## Cross-Lens Summary Diff",
            "",
            f"- Comparable entries: `{cross_lens_diff['comparable_entry_count_before']}` -> `{cross_lens_diff['comparable_entry_count_after']}`",
            f"- Label disagreements: `{cross_lens_diff['label_disagreement_count_before']}` -> `{cross_lens_diff['label_disagreement_count_after']}`",
            f"- Direction disagreements: `{cross_lens_diff['direction_disagreement_count_before']}` -> `{cross_lens_diff['direction_disagreement_count_after']}`",
            f"- Sign-flip examples: `{cross_lens_diff['sign_flip_count_before']}` -> `{cross_lens_diff['sign_flip_count_after']}`",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def write_corpus_review_normalization_diff_artifacts(diff, markdown_output=None):
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_corpus_review_normalization_diff_markdown(diff))


def render_character_cross_lens_analysis_markdown(analysis):
    lines = [
        "# Character Cross-Lens Analysis",
        "",
        f"- Analysis version: `{analysis['character_cross_lens_analysis_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
        "## Top Positive By Lens",
        "",
    ]

    for lens in sorted(SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                _markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            _format_signed_number(row["net_score"]),
                            row["unit_count"],
                        )
                        for row in analysis["top_positive_by_lens"][lens]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Top Negative By Lens",
            "",
        ]
    )

    for lens in sorted(SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                _markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            _format_signed_number(row["net_score"]),
                            row["unit_count"],
                        )
                        for row in analysis["top_negative_by_lens"][lens]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Largest Cross-Lens Rank Spread",
            "",
            _markdown_table(
                ["Character", "Advantage Rank", "Prestige Rank", "Inclusion Rank", "Rank Spread", "Max Units"],
                [
                    (
                        row["character"],
                        row["lens_scores"]["advantage"]["rank"],
                        row["lens_scores"]["prestige"]["rank"],
                        row["lens_scores"]["inclusion"]["rank"],
                        row["rank_spread"],
                        row["max_unit_count"],
                    )
                    for row in analysis["top_rank_spread_characters"]
                ],
            ),
            "",
            "## Highest Volatility",
            "",
            _markdown_table(
                ["Character", "Advantage Span", "Prestige Span", "Inclusion Span", "Max Span", "Max Units"],
                [
                    (
                        row["character"],
                        _format_signed_number(row["lens_scores"]["advantage"]["score_span"]),
                        _format_signed_number(row["lens_scores"]["prestige"]["score_span"]),
                        _format_signed_number(row["lens_scores"]["inclusion"]["score_span"]),
                        _format_signed_number(row["max_score_span"]),
                        row["max_unit_count"],
                    )
                    for row in analysis["top_volatile_characters"]
                ],
            ),
            "",
            "## Character Table",
            "",
            _markdown_table(
                [
                    "Character",
                    "Advantage",
                    "Prestige",
                    "Inclusion",
                    "Advantage Rank",
                    "Prestige Rank",
                    "Inclusion Rank",
                    "Max Units",
                    "Max Span",
                ],
                [
                    (
                        row["character"],
                        _format_signed_number(row["lens_scores"]["advantage"]["net_score"]),
                        _format_signed_number(row["lens_scores"]["prestige"]["net_score"]),
                        _format_signed_number(row["lens_scores"]["inclusion"]["net_score"]),
                        row["lens_scores"]["advantage"]["rank"],
                        row["lens_scores"]["prestige"]["rank"],
                        row["lens_scores"]["inclusion"]["rank"],
                        row["max_unit_count"],
                        _format_signed_number(row["max_score_span"]),
                    )
                    for row in analysis["characters"][:40]
                ],
            ),
            "",
        ]
    )

    if len(analysis["characters"]) > 40:
        lines.extend(
            [
                f"_Showing first 40 of {len(analysis['characters'])} character rows._",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_character_cross_lens_analysis_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_cross_lens_analysis_markdown(analysis))


def render_character_chapter_analysis_markdown(analysis):
    lines = [
        "# Character Chapter Analysis",
        "",
        f"- Analysis version: `{analysis['character_chapter_analysis_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Selected character count: `{analysis['selected_character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for character_row in analysis["characters"]:
        summary = character_row["cross_lens_summary"]
        lines.extend(
            [
                f"## {character_row['character']}",
                "",
                f"- Selected by: `{', '.join(character_row['selected_by']) or 'manual'}`",
                f"- Advantage / Prestige / Inclusion ranks: `{summary['lens_scores']['advantage']['rank']}` / `{summary['lens_scores']['prestige']['rank']}` / `{summary['lens_scores']['inclusion']['rank']}`",
                f"- Rank spread: `{summary['rank_spread']}`",
                f"- Max units: `{summary['max_unit_count']}`",
                f"- Max score span: `{_format_signed_number(summary['max_score_span'])}`",
                "",
                _markdown_table(
                    [
                        "Chapter",
                        "Advantage",
                        "Prestige",
                        "Inclusion",
                        "Advantage Units",
                        "Prestige Units",
                        "Inclusion Units",
                    ],
                    [
                        (
                            row["chapter_id"],
                            _format_signed_number(row["advantage"]["net_score"]),
                            _format_signed_number(row["prestige"]["net_score"]),
                            _format_signed_number(row["inclusion"]["net_score"]),
                            row["advantage"]["unit_count"],
                            row["prestige"]["unit_count"],
                            row["inclusion"]["unit_count"],
                        )
                        for row in character_row["chapters"]
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_character_chapter_analysis_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_chapter_analysis_markdown(analysis))


def render_character_annotation_counts_markdown(analysis):
    lines = [
        "# Character Annotation Counts",
        "",
        f"- Analysis version: `{analysis['character_annotation_counts_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
        _markdown_table(
            ["Character", "Annotation Units", "Advantage", "Prestige", "Inclusion"],
            [
                (
                    row["character"],
                    row["annotation_unit_count"],
                    _format_signed_number(row["advantage_net_score"]),
                    _format_signed_number(row["prestige_net_score"]),
                    _format_signed_number(row["inclusion_net_score"]),
                )
                for row in analysis["characters"]
            ],
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_character_annotation_counts_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_annotation_counts_markdown(analysis))


def render_character_profile_cards_markdown(analysis):
    lines = [
        "# Character Profile Cards",
        "",
        f"- Analysis version: `{analysis['character_profile_cards_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for card in analysis["cards"][:20]:
        lines.extend(
            [
                f"## {card['character']}",
                "",
                f"- Annotation units: `{card['annotation_unit_count']}`",
                f"- Rank spread: `{card['rank_spread']}`",
                f"- Max score span: `{_format_signed_number(card['max_score_span'])}`",
                f"- Selected by: `{', '.join(card['selected_by']) or 'none'}`",
                "",
                _markdown_table(
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            _format_signed_number(card["lens_scores"][lens]["net_score"]),
                            (
                                f"{card['lens_scores'][lens]['percentile']}th"
                                if card["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            card["lens_scores"][lens]["rank"],
                            card["lens_scores"][lens]["unit_count"],
                            card["lens_scores"][lens]["dominant_status_dimension"],
                            _format_signed_number(card["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                _markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            _format_signed_number(row["advantage"]["net_score"]),
                            _format_signed_number(row["prestige"]["net_score"]),
                            _format_signed_number(row["inclusion"]["net_score"]),
                        )
                        for row in card["top_chapters"]
                    ],
                ),
                "",
            ]
        )

    if len(analysis["cards"]) > 20:
        lines.extend([f"_Showing first 20 of {len(analysis['cards'])} cards._", "",])

    return "\n".join(lines).rstrip() + "\n"


def write_character_profile_cards_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_profile_cards_markdown(analysis))


def write_chapter_overlay_artifacts(dataset, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(dataset["manifest"], ensure_ascii=False, indent=2) + "\n")

    chapters_dir = output_path / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    for chapter in dataset["chapters"]:
        chapter_path = chapters_dir / f"{chapter['chapterId']}.json"
        chapter_path.write_text(json.dumps(chapter, ensure_ascii=False, indent=2) + "\n")


def render_character_pages_markdown(analysis):
    lines = [
        "# Character Pages",
        "",
        f"- Analysis version: `{analysis['character_pages_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for page in analysis["pages"]:
        lines.extend(
            [
                f"## {page['character']}",
                "",
                f"- Slug: `{page['slug']}`",
                f"- Portrait default: `{page['portrait']['default'] or 'none'}`",
                f"- Annotation units: `{page['profile']['annotation_unit_count']}`",
                f"- Rank spread: `{page['profile']['rank_spread']}`",
                f"- Max score span: `{_format_signed_number(page['profile']['max_score_span'])}`",
                f"- Pattern: `{page['editorial']['primary_pattern']}`",
                "",
                page["editorial"]["dek"],
                "",
                page["editorial"]["summary"],
                "",
                "Why interesting:",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in page["editorial"]["why_interesting"])
        lines.extend(
            [
                "",
                _markdown_table(
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            _format_signed_number(page["profile"]["lens_scores"][lens]["net_score"]),
                            (
                                f"{page['profile']['lens_scores'][lens]['percentile']}th"
                                if page["profile"]["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            page["profile"]["lens_scores"][lens]["rank"],
                            page["profile"]["lens_scores"][lens]["unit_count"],
                            page["profile"]["lens_scores"][lens]["dominant_status_dimension"],
                            _format_signed_number(page["profile"]["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                _markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            _format_signed_number(row["advantage"]["net_score"]),
                            _format_signed_number(row["prestige"]["net_score"]),
                            _format_signed_number(row["inclusion"]["net_score"]),
                        )
                        for row in page["top_chapters"]
                    ],
                ),
                "",
                "Reading path:",
                "",
            ]
        )
        lines.extend(f"- {row['label']}: `{row['reader_link']}`" for row in page["reading_path"])
        lines.extend(
            [
                "",
                "Notable units:",
                "",
            ]
        )
        lines.extend(f"- {row['label']}: `{row['reader_link']}`" for row in page["notable_units"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_character_pages_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_pages_markdown(analysis))


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
        f"- Character normalization applied: `{review['character_normalization']['applied']}`",
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


def _clean_character_name(value):
    return value.strip() if isinstance(value, str) else ""


def _read_alias_csv_pairs(path=ALIASES_CSV):
    alias_path = Path(path)
    if not alias_path.exists():
        raise ValueError(f'Alias CSV "{alias_path}" does not exist.')

    pairs = []
    with alias_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            alias = _clean_character_name(row[0])
            canonical = _clean_character_name(row[1])
            if alias and canonical and alias != canonical:
                pairs.append({"alias": alias, "canonical": canonical})
    return pairs


def _record_character_usage(usage, name, role, path, unit_id):
    clean_name = _clean_character_name(name)
    if not clean_name:
        return
    entry = usage.setdefault(
        clean_name,
        {
            "total_references": 0,
            "roles": {
                "characters_present": 0,
                "event_source": 0,
                "event_target": 0,
                "status_effect": 0,
            },
            "examples": [],
        },
    )
    entry["total_references"] += 1
    entry["roles"][role] += 1
    if len(entry["examples"]) < 5:
        entry["examples"].append({"path": str(path), "unit_id": unit_id})


def _collect_annotation_character_usage(outputs_dir):
    usage = {}
    for annotation_path in sorted(Path(outputs_dir).glob("run-*/annotations/*.json")):
        annotation = _read_json(annotation_path)
        unit_id = annotation.get("unit_id")
        for character in annotation.get("characters_present", []):
            _record_character_usage(
                usage,
                character.get("canonical_name"),
                "characters_present",
                annotation_path,
                unit_id,
            )
        for event in annotation.get("appraisal_events", []):
            source = event.get("source")
            if source not in ALLOWED_EVENT_SOURCES:
                _record_character_usage(usage, source, "event_source", annotation_path, unit_id)
            _record_character_usage(usage, event.get("target"), "event_target", annotation_path, unit_id)
        for effect in annotation.get("status_effects", []):
            _record_character_usage(
                usage,
                effect.get("character"),
                "status_effect",
                annotation_path,
                unit_id,
            )
    return usage


def _collect_run_alias_evidence(outputs_dir):
    canonical_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    pair_examples = defaultdict(list)
    for manifest_path in sorted(Path(outputs_dir).glob("run-*/run.json")):
        manifest = _read_json(manifest_path)
        for canonical, entry in (manifest.get("alias_map") or {}).items():
            clean_canonical = _clean_character_name(canonical)
            if not clean_canonical:
                continue
            canonical_counts[clean_canonical] += 1
            for alias in entry.get("aliases", []):
                clean_alias = _clean_character_name(alias)
                if not clean_alias or clean_alias == clean_canonical:
                    continue
                key = (clean_alias, clean_canonical)
                pair_counts[key] += 1
                if len(pair_examples[key]) < 5:
                    pair_examples[key].append(manifest_path.parent.name)
    return canonical_counts, pair_counts, pair_examples


def _connected_alias_components(edges):
    graph = defaultdict(set)
    for left, right in edges:
        if left == right:
            continue
        graph[left].add(right)
        graph[right].add(left)

    seen = set()
    components = []
    for node in sorted(graph):
        if node in seen:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(graph[current] - seen)
        components.append(component)
    return components


def build_character_alias_audit(outputs_dir="outputs", aliases_csv=ALIASES_CSV):
    output_path = Path(outputs_dir)
    if not output_path.exists():
        raise ValueError(f'Outputs directory "{output_path}" does not exist.')

    annotation_usage = _collect_annotation_character_usage(output_path)
    csv_pairs = _read_alias_csv_pairs(aliases_csv)
    run_canonical_counts, run_pair_counts, run_pair_examples = _collect_run_alias_evidence(output_path)

    csv_edges = {(pair["alias"], pair["canonical"]) for pair in csv_pairs}
    csv_canonical_counts = defaultdict(int)
    for pair in csv_pairs:
        csv_canonical_counts[pair["canonical"]] += 1
    run_edges = set(run_pair_counts)
    components = _connected_alias_components(csv_edges | run_edges)

    candidate_groups = []
    for component in components:
        annotation_names = sorted(name for name in component if name in annotation_usage)
        if len(annotation_names) < 2:
            continue

        preferred_name = sorted(
            annotation_names,
            key=lambda name: (
                -annotation_usage[name]["total_references"],
                -csv_canonical_counts.get(name, 0),
                -run_canonical_counts.get(name, 0),
                name.lower(),
            ),
        )[0]
        names = [
            {
                "name": name,
                "total_references": annotation_usage[name]["total_references"],
                "roles": annotation_usage[name]["roles"],
                "run_manifest_canonical_count": run_canonical_counts.get(name, 0),
                "examples": annotation_usage[name]["examples"],
            }
            for name in sorted(
                annotation_names,
                key=lambda item: (-annotation_usage[item]["total_references"], item.lower()),
            )
        ]

        alias_edges = []
        for left, right in sorted(csv_edges | run_edges):
            if left not in component or right not in component:
                continue
            sources = []
            if (left, right) in csv_edges:
                sources.append("aliases_csv")
            run_count = run_pair_counts.get((left, right), 0)
            if run_count:
                sources.append("run_alias_maps")
            alias_edges.append(
                {
                    "alias": left,
                    "canonical": right,
                    "sources": sources,
                    "run_manifest_count": run_count,
                    "run_examples": run_pair_examples.get((left, right), []),
                }
            )

        candidate_groups.append(
            {
                "preferred_name_by_usage": preferred_name,
                "annotation_name_count": len(annotation_names),
                "total_annotation_references": sum(
                    annotation_usage[name]["total_references"] for name in annotation_names
                ),
                "names": names,
                "alias_edges": alias_edges,
            }
        )

    candidate_groups.sort(
        key=lambda group: (
            -group["total_annotation_references"],
            group["preferred_name_by_usage"].lower(),
        )
    )

    annotation_names = {
        name: {
            "total_references": usage["total_references"],
            "roles": usage["roles"],
        }
        for name, usage in sorted(
            annotation_usage.items(),
            key=lambda item: (-item[1]["total_references"], item[0].lower()),
        )
    }

    return {
        "character_alias_audit_version": "character_alias_audit_v1",
        "outputs_dir": str(output_path),
        "aliases_csv": str(Path(aliases_csv)),
        "annotation_name_count": len(annotation_usage),
        "run_manifest_canonical_name_count": len(run_canonical_counts),
        "csv_alias_pair_count": len(csv_pairs),
        "run_manifest_alias_pair_count": len(run_pair_counts),
        "candidate_merge_group_count": len(candidate_groups),
        "candidate_merge_groups": candidate_groups,
        "annotation_names": annotation_names,
    }


def render_character_alias_audit_markdown(audit):
    lines = [
        "# Character Alias Audit",
        "",
        f"- Audit version: `{audit['character_alias_audit_version']}`",
        f"- Outputs directory: `{audit['outputs_dir']}`",
        f"- Alias CSV: `{audit['aliases_csv']}`",
        f"- Annotation names: `{audit['annotation_name_count']}`",
        f"- Run-manifest canonical names: `{audit['run_manifest_canonical_name_count']}`",
        f"- CSV alias pairs: `{audit['csv_alias_pair_count']}`",
        f"- Run-manifest alias pairs: `{audit['run_manifest_alias_pair_count']}`",
        f"- Candidate merge groups: `{audit['candidate_merge_group_count']}`",
        "",
        "## Candidate Merge Groups",
        "",
    ]

    if audit["candidate_merge_groups"]:
        lines.append(
            _markdown_table(
                ["Preferred Name", "Names", "References"],
                [
                    (
                        group["preferred_name_by_usage"],
                        ", ".join(name["name"] for name in group["names"]),
                        group["total_annotation_references"],
                    )
                    for group in audit["candidate_merge_groups"]
                ],
            )
        )
        lines.append("")
    else:
        lines.extend(["No candidate merge groups found.", ""])

    for group in audit["candidate_merge_groups"]:
        lines.extend(
            [
                f"### {group['preferred_name_by_usage']}",
                "",
                _markdown_table(
                    [
                        "Name",
                        "References",
                        "Character Present",
                        "Event Source",
                        "Event Target",
                        "Status Effect",
                        "Run Canonical Count",
                    ],
                    [
                        (
                            name["name"],
                            name["total_references"],
                            name["roles"]["characters_present"],
                            name["roles"]["event_source"],
                            name["roles"]["event_target"],
                            name["roles"]["status_effect"],
                            name["run_manifest_canonical_count"],
                        )
                        for name in group["names"]
                    ],
                ),
                "",
                "Alias evidence:",
                "",
                _markdown_table(
                    ["Alias", "Canonical", "Sources", "Run Count", "Run Examples"],
                    [
                        (
                            edge["alias"],
                            edge["canonical"],
                            ", ".join(edge["sources"]),
                            edge["run_manifest_count"],
                            ", ".join(edge["run_examples"]),
                        )
                        for edge in group["alias_edges"]
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_character_alias_audit_artifacts(audit, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_alias_audit_markdown(audit))


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
        help='Compute a lightweight advantage "winning"/"losing" transformation for a run.',
    )
    score_parser.add_argument("--run", required=True, help="Run directory to score.")
    score_parser.add_argument("--lens", default="advantage", choices=sorted(SCORING_LENS_CONFIGS), help="Scoring lens.")

    report_parser = subparsers.add_parser(
        "report",
        help="Build a compact downstream outcome report from advantage outcome scores.",
    )
    report_parser.add_argument("--run", required=True, help="Run directory to report on.")
    report_parser.add_argument("--lens", default="advantage", choices=sorted(SCORING_LENS_CONFIGS), help="Scoring lens.")
    report_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )

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
    corpus_review_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    corpus_review_parser.add_argument("--output", help="Optional JSON output path.")
    corpus_review_parser.add_argument("--markdown-output", help="Optional Markdown output path.")
    corpus_review_parser.add_argument(
        "--normalization-diff-output",
        help="Optional Markdown output path for a diff between unnormalized and normalized corpus reviews.",
    )

    character_analysis_parser = subparsers.add_parser(
        "character-analysis",
        help="Build a per-character cross-lens downstream analysis from the corpus review surface.",
    )
    character_analysis_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    character_analysis_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_analysis_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_analysis_parser.add_argument("--output", help="Optional JSON output path.")
    character_analysis_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_chapter_parser = subparsers.add_parser(
        "character-chapter-analysis",
        help="Build a chapter-by-chapter cross-lens analysis for the highest-information characters.",
    )
    character_chapter_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    character_chapter_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_chapter_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_chapter_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional character to include. Repeat for multiple characters. Defaults to the union of top rank-spread and top volatile figures.",
    )
    character_chapter_parser.add_argument("--output", help="Optional JSON output path.")
    character_chapter_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_counts_parser = subparsers.add_parser(
        "character-annotation-counts",
        help="Build a normalized character list sorted by annotation unit count.",
    )
    character_counts_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    character_counts_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_counts_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_counts_parser.add_argument("--output", help="Optional JSON output path.")
    character_counts_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_cards_parser = subparsers.add_parser(
        "character-profile-cards",
        help="Build app-facing cross-lens character profile cards.",
    )
    character_cards_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    character_cards_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_cards_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_cards_parser.add_argument("--output", help="Optional JSON output path.")
    character_cards_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_pages_parser = subparsers.add_parser(
        "character-pages",
        help="Build pilot app-facing character pages from existing analysis artifacts.",
    )
    character_pages_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    character_pages_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_pages_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_pages_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional pilot character to include. Repeat for multiple characters.",
    )
    character_pages_parser.add_argument("--output", help="Optional JSON output path.")
    character_pages_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    chapter_overlay_parser = subparsers.add_parser(
        "chapter-overlays",
        help="Export chapter-keyed app overlay JSON from the accepted normalized corpus surface.",
    )
    chapter_overlay_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run directory to include. Repeat for multiple runs.",
    )
    chapter_overlay_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    chapter_overlay_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    chapter_overlay_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest.json and chapter JSON files should be written.",
    )

    alias_audit_parser = subparsers.add_parser(
        "character-alias-audit",
        help="Audit possible duplicate character names using aliases and annotation outputs.",
    )
    alias_audit_parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Outputs directory containing run-* annotation outputs.",
    )
    alias_audit_parser.add_argument(
        "--aliases-csv",
        default=str(ALIASES_CSV),
        help="Two-column alias CSV to use as audit evidence.",
    )
    alias_audit_parser.add_argument("--output", help="Optional JSON output path.")
    alias_audit_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

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
        help="Build advantage, prestige, and inclusion reports after completion.",
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
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            report = build_outcome_report(args.run, lens=args.lens, character_name_map=character_name_map)
        except RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "corpus-review":
        if args.normalization_diff_output and not args.reviewed_character_normalization:
            parser.error("--normalization-diff-output requires --reviewed-character-normalization")
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            baseline_review = None
            if args.normalization_diff_output:
                baseline_review = build_corpus_sanity_review(runs)
            review = build_corpus_sanity_review(runs, character_name_map=character_name_map)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_corpus_review_artifacts(
            review,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.normalization_diff_output:
            if baseline_review is None:
                baseline_review = build_corpus_sanity_review(runs)
            diff = build_corpus_review_normalization_diff(baseline_review, review)
            write_corpus_review_normalization_diff_artifacts(
                diff,
                markdown_output=args.normalization_diff_output,
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
                        "normalization_diff_output": args.normalization_diff_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-alias-audit":
        try:
            audit = build_character_alias_audit(
                outputs_dir=args.outputs_dir,
                aliases_csv=args.aliases_csv,
            )
        except ValueError as exc:
            parser.error(str(exc))
        write_character_alias_audit_artifacts(
            audit,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "annotation_name_count": audit["annotation_name_count"],
                        "candidate_merge_group_count": audit["candidate_merge_group_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-analysis":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            review = build_corpus_sanity_review(runs, character_name_map=character_name_map)
            analysis = build_character_cross_lens_analysis(review)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_character_cross_lens_analysis_artifacts(
            analysis,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "character_count": analysis["character_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-chapter-analysis":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            analysis = build_character_chapter_analysis(
                runs,
                character_name_map=character_name_map,
                target_characters=args.characters,
            )
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_character_chapter_analysis_artifacts(
            analysis,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "selected_character_count": analysis["selected_character_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-annotation-counts":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            review = build_corpus_sanity_review(runs, character_name_map=character_name_map)
            analysis = build_character_annotation_counts(review)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_character_annotation_counts_artifacts(
            analysis,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "character_count": analysis["character_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-profile-cards":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            analysis = build_character_profile_cards(runs, character_name_map=character_name_map)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_character_profile_cards_artifacts(
            analysis,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "character_count": analysis["character_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chapter-overlays":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            dataset = build_chapter_overlay_data(runs, character_name_map=character_name_map)
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_chapter_overlay_artifacts(dataset, args.output_dir)
        print(
            json.dumps(
                {
                    "chapter_count": dataset["chapter_count"],
                    "output_dir": args.output_dir,
                    "character_normalization_applied": dataset["character_normalization"]["applied"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "character-pages":
        try:
            runs = list(args.runs or [])
            if args.discover_runs:
                runs.extend(discover_annotation_run_dirs(args.discover_runs))
            character_name_map = (
                REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None
            )
            analysis = build_character_pages(
                runs,
                character_name_map=character_name_map,
                target_characters=args.characters,
            )
        except (RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_character_pages_artifacts(
            analysis,
            json_output=args.output,
            markdown_output=args.markdown_output,
        )
        if args.output or args.markdown_output:
            print(
                json.dumps(
                    {
                        "character_count": analysis["character_count"],
                        "json_output": args.output,
                        "markdown_output": args.markdown_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
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
