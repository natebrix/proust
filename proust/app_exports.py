from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import median
import unicodedata

from . import runner
from .app_config import ISLT_PORTRAITS_DIR, ISLT_READER_BASE_PATH, PORTRAIT_STYLES
from .character_data import normalize_character_name_map
from .editorial import CHAPTER_SUMMARY_EDITORIAL, CHARACTER_PAGE_PILOT_EDITORIAL, CHARACTER_PORTRAIT_SLUGS
from .paths import ISLT_EDITIONS_DIR
from .reporting_utils import character_ranks, character_totals_by_name
from .scoring import SCORING_LENS_CONFIGS, SCORING_LENS_ORDER


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
        lens: character_ranks(review["lens_reviews"][lens]["character_totals"], reverse=True)
        for lens in lenses
    }
    totals_by_lens = {
        lens: character_totals_by_name(review["lens_reviews"][lens]["character_totals"])
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
    lens_label_counts = {
        lens: {"win": 0, "loss": 0, "mixed": 0, "neutral": 0}
        for lens in sorted(SCORING_LENS_CONFIGS)
    }
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


def _chapter_density_direction(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "balanced"


def _chapter_character_signature(row):
    parts = []
    for lens in SCORING_LENS_ORDER:
        direction = _chapter_density_direction(row[lens]["net_score"])
        if direction == "balanced":
            parts.append(f"{lens} mixed")
        else:
            parts.append(f"{lens} {direction}")
    return ", ".join(parts)


def _chapter_tonal_archetype_label(intensity_flags):
    key = "".join("1" if intensity_flags[lens] else "0" for lens in SCORING_LENS_ORDER)
    return {
        "000": "Diffuse",
        "001": "Intimate",
        "010": "Ceremonial",
        "011": "Social",
        "100": "Confrontational",
        "101": "Volatile",
        "110": "Competitive",
        "111": "Totalizing",
    }[key]


def _chapter_impact_mass(row):
    return round(
        sum(abs(row[lens]["net_score"]) for lens in SCORING_LENS_ORDER),
        3,
    )


def _unit_impact_mass(unit):
    return round(
        sum(
            abs(character_row[lens]["netScore"])
            for character_row in unit["characters"]
            for lens in SCORING_LENS_ORDER
        ),
        3,
    )


def _build_chapter_distinguishing_passages(chapter_overlay_row, limit=5):
    passages = []
    for unit in chapter_overlay_row.get("units", []):
        lens_totals = {
            lens: round(
                sum(character_row[lens]["netScore"] for character_row in unit["characters"]),
                3,
            )
            for lens in SCORING_LENS_ORDER
        }
        passages.append(
            {
                "unit_id": unit["unitId"],
                "paragraph_start": unit["paragraphStart"],
                "paragraph_end": unit["paragraphEnd"],
                "reader_link": _reader_chapter_link(
                    chapter_overlay_row["chapterId"],
                    paragraph_start=unit["paragraphStart"],
                ),
                "summary": unit["summary"],
                "dominant_character": unit["dominantCharacter"],
                "impact_mass": _unit_impact_mass(unit),
                "lens_signature": {
                    lens: _chapter_density_direction(value)
                    for lens, value in lens_totals.items()
                },
            }
        )

    passages.sort(
        key=lambda item: (
            -item["impact_mass"],
            item["paragraph_start"],
            item["paragraph_end"],
            item["unit_id"],
        )
    )
    return passages[:limit]


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
    return {chapter.id: chapter.title for chapter in runner.CANONICAL_CHAPTER_SPECS}


def _validate_chapter_summary_editorial():
    canonical_ids = {chapter.id for chapter in runner.CANONICAL_CHAPTER_SPECS}
    editorial_ids = set(CHAPTER_SUMMARY_EDITORIAL)
    missing = sorted(canonical_ids - editorial_ids)
    extra = sorted(editorial_ids - canonical_ids)
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if extra:
            problems.append(f"extra: {', '.join(extra)}")
        raise ValueError("Chapter summary editorial coverage does not match canonical chapters: " + "; ".join(problems))


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
        body = stem[len(prefix):]
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
    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
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
                    "subheading": editorial["subheading"],
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


def build_chapter_summary_export(run_dirs, character_name_map=None, top_character_limit=8, top_passage_limit=5):
    _validate_chapter_summary_editorial()
    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    cross_lens = build_character_cross_lens_analysis(review)
    chapter_analysis = build_character_chapter_analysis(
        run_dirs,
        character_name_map=character_name_map,
        target_characters=[row["character"] for row in cross_lens["characters"]],
    )
    overlay_dataset = build_chapter_overlay_data(run_dirs, character_name_map=character_name_map)
    chapter_titles = _chapter_title_map()
    chapter_manifest_rows = {row["chapterId"]: row for row in overlay_dataset["manifest"]["chapters"]}
    overlay_rows = {row["chapterId"]: row for row in overlay_dataset["chapters"]}

    chapter_character_rows = defaultdict(list)
    for character_row in chapter_analysis["characters"]:
        summary = character_row["cross_lens_summary"]
        for row in character_row["chapters"]:
            chapter_character_rows[row["chapter_id"]].append(
                {
                    "character": character_row["character"],
                    "unit_count": max(
                        row["advantage"]["unit_count"],
                        row["prestige"]["unit_count"],
                        row["inclusion"]["unit_count"],
                    ),
                    "advantage": {"net_score": row["advantage"]["net_score"]},
                    "prestige": {"net_score": row["prestige"]["net_score"]},
                    "inclusion": {"net_score": row["inclusion"]["net_score"]},
                    "rank_spread": summary["rank_spread"],
                    "max_score_span": summary["max_score_span"],
                }
            )

    chapter_payloads = []
    chapters = []
    for chapter in runner.CANONICAL_CHAPTER_SPECS:
        rows = chapter_character_rows.get(chapter.id, [])
        if rows:
            for lens in SCORING_LENS_ORDER:
                ordered = sorted(rows, key=lambda item: (-item[lens]["net_score"], item["character"]))
                for rank, row in enumerate(ordered, start=1):
                    row.setdefault("chapter_ranks", {})
                    row["chapter_ranks"][lens] = rank

        top_character_rows = sorted(
            rows,
            key=lambda item: (
                -_chapter_impact_mass(item),
                -item["unit_count"],
                item["character"],
            ),
        )[:top_character_limit]
        top_characters = [
            {
                "character": row["character"],
                "unit_count": row["unit_count"],
                "impact_mass": _chapter_impact_mass(row),
                "dominant_lens": max(
                    SCORING_LENS_ORDER,
                    key=lambda lens: (abs(row[lens]["net_score"]), lens),
                ),
                "lens_signature": _chapter_character_signature(row),
                "advantage": {"net_score": row["advantage"]["net_score"]},
                "prestige": {"net_score": row["prestige"]["net_score"]},
                "inclusion": {"net_score": row["inclusion"]["net_score"]},
            }
            for row in top_character_rows
        ]

        strongest_split_character = None
        if rows:
            for row in rows:
                chapter_ranks = row.get("chapter_ranks", {})
                if chapter_ranks:
                    row["chapter_rank_spread"] = max(chapter_ranks.values()) - min(chapter_ranks.values())
                else:
                    row["chapter_rank_spread"] = 0
            strongest = max(
                rows,
                key=lambda item: (
                    item["chapter_rank_spread"],
                    item["unit_count"],
                    max(
                        abs(item["advantage"]["net_score"]),
                        abs(item["prestige"]["net_score"]),
                        abs(item["inclusion"]["net_score"]),
                    ),
                    item["character"],
                ),
            )
            strongest_split_character = {
                "character": strongest["character"],
                "rank_spread": strongest["chapter_rank_spread"],
                "unit_count": strongest["unit_count"],
            }

        unit_count = chapter_manifest_rows.get(chapter.id, {}).get("unitCount", 0)
        lens_totals = {
            lens: round(sum(row[lens]["net_score"] for row in rows), 3)
            for lens in SCORING_LENS_ORDER
        }
        intensity_totals = {
            lens: round(sum(abs(row[lens]["net_score"]) for row in rows), 3)
            for lens in SCORING_LENS_ORDER
        }
        lens_profile = {
            lens: {
                "net_score_total": lens_totals[lens],
                "signed_density": round(lens_totals[lens] / unit_count, 3) if unit_count else 0.0,
                "intensity_density": round(intensity_totals[lens] / unit_count, 3) if unit_count else 0.0,
                "direction": _chapter_density_direction(lens_totals[lens]),
            }
            for lens in SCORING_LENS_ORDER
        }
        distinguishing_passages = _build_chapter_distinguishing_passages(
            overlay_rows.get(chapter.id, {}),
            limit=top_passage_limit,
        )
        chapter_payloads.append(
            {
                "chapter_id": chapter.id,
                "chapter_title": chapter_titles.get(chapter.id, chapter.id),
                "reader_link": _reader_chapter_link(chapter.id),
                "unit_count": unit_count,
                "top_characters": top_characters,
                "strongest_split_character": strongest_split_character,
                "lens_profile": lens_profile,
                "distinguishing_passages": distinguishing_passages,
            }
        )

    chapter_population_size = len(chapter_payloads)
    intensity_medians = {
        lens: round(
            median(
                chapter["lens_profile"][lens]["intensity_density"]
                for chapter in chapter_payloads
            ),
            3,
        )
        if chapter_payloads
        else 0.0
        for lens in SCORING_LENS_ORDER
    }
    signed_density_ranks = {
        lens: {
            chapter["chapter_id"]: index + 1
            for index, chapter in enumerate(
                sorted(
                    chapter_payloads,
                    key=lambda item: (
                        -item["lens_profile"][lens]["signed_density"],
                        item["chapter_id"],
                    ),
                )
            )
        }
        for lens in SCORING_LENS_ORDER
    }

    for chapter in chapter_payloads:
        intensity_flags = {}
        for lens in SCORING_LENS_ORDER:
            lens_row = chapter["lens_profile"][lens]
            lens_row["chapter_rank"] = signed_density_ranks[lens][chapter["chapter_id"]]
            lens_row["chapter_percentile"] = _rank_to_percentile(
                lens_row["chapter_rank"],
                chapter_population_size,
            )
            lens_row["intensity_above_median"] = lens_row["intensity_density"] > intensity_medians[lens]
            intensity_flags[lens] = lens_row["intensity_above_median"]

        chapter["tonal_archetype"] = {
            "label": _chapter_tonal_archetype_label(intensity_flags),
            "intense_lenses": [
                lens for lens in SCORING_LENS_ORDER if intensity_flags[lens]
            ],
            "intensity_signature": intensity_flags,
        }
        chapter["summary"] = CHAPTER_SUMMARY_EDITORIAL[chapter["chapter_id"]]
        chapters.append(chapter)

    return {
        "chapter_summary_export_version": "chapter_summary_export_v2",
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "intensity_medians": intensity_medians,
        "chapter_count": len(chapters),
        "chapters": chapters,
    }


def build_chapter_overlay_data(run_dirs, character_name_map=None):
    if not run_dirs:
        raise ValueError("At least one run directory is required for chapter overlay export.")

    character_name_map = normalize_character_name_map(character_name_map)
    timeline_by_lens = {lens: [] for lens in sorted(SCORING_LENS_CONFIGS)}
    preferred_run_by_unit = {}

    for run_dir in run_dirs:
        status = runner.get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for unit in status["units"]:
            unit_id = unit["unit_id"]
            if unit["review_state"] != "reviewed":
                continue
            existing_run_id = preferred_run_by_unit.get(unit_id)
            if existing_run_id is None or _run_id_sort_key(run_id) > _run_id_sort_key(existing_run_id):
                preferred_run_by_unit[unit_id] = run_id

    for run_dir in run_dirs:
        status = runner.get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for lens in sorted(SCORING_LENS_CONFIGS):
            report = runner.build_outcome_report(run_dir, lens=lens, character_name_map=character_name_map)
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
    for chapter in runner.CANONICAL_CHAPTER_SPECS:
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
    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
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

    chapter_order = [chapter.id for chapter in runner.CANONICAL_CHAPTER_SPECS]
    chapter_positions = {chapter_id: index for index, chapter_id in enumerate(chapter_order)}
    preferred_run_by_unit = {}
    for run_dir in run_dirs:
        status = runner.get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for unit in status["units"]:
            if unit["review_state"] != "reviewed":
                continue
            unit_id = unit["unit_id"]
            existing_run_id = preferred_run_by_unit.get(unit_id)
            if existing_run_id is None or _run_id_sort_key(run_id) > _run_id_sort_key(existing_run_id):
                preferred_run_by_unit[unit_id] = run_id
    lens_reports = {lens: [] for lens in sorted(SCORING_LENS_CONFIGS)}
    for run_dir in run_dirs:
        status = runner.get_run_status(run_dir)
        run_id = status["manifest"]["run_id"]
        for lens in sorted(SCORING_LENS_CONFIGS):
            lens_reports[lens].append(
                (
                    run_id,
                    runner.build_outcome_report(run_dir, lens=lens, character_name_map=character_name_map),
                )
            )

    selected_set = set(selected_characters)
    chapter_rows = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"net_score": 0.0, "unit_count": 0})))

    for lens, reports in lens_reports.items():
        for run_id, report in reports:
            for entry in report["timeline"]:
                if preferred_run_by_unit.get(entry["unit_id"]) != run_id:
                    continue
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


def _elo_expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_observed_scores(score_a, score_b, epsilon):
    difference = score_a - score_b
    if difference > epsilon:
        return 1.0, 0.0
    if difference < -epsilon:
        return 0.0, 1.0
    return 0.5, 0.5


def _word_count(text):
    return len(re.findall(r"\S+", text or ""))


def _load_canonical_chapter_json_silent(chapter_id, edition="fr-original"):
    chapter_path = ISLT_EDITIONS_DIR / edition / "chapters" / f"{chapter_id}.json"
    return json.loads(chapter_path.read_text())


def _build_corpus_position_index(units_by_chapter):
    chapter_order = [chapter.id for chapter in runner.CANONICAL_CHAPTER_SPECS]
    chapter_titles = _chapter_title_map()
    chapter_index = {}
    cumulative_unit_index = 0
    cumulative_paragraph_offset = 0
    cumulative_word_offset = 0

    for chapter_number, chapter_id in enumerate(chapter_order, start=1):
        chapter_json = _load_canonical_chapter_json_silent(chapter_id)
        paragraphs = chapter_json["paragraphs"]
        chapter_spec = next(chapter for chapter in runner.CANONICAL_CHAPTER_SPECS if chapter.id == chapter_id)
        paragraph_word_counts = [_word_count(paragraph["text"]) for paragraph in paragraphs]
        cumulative_words_by_paragraph_end = []
        cumulative_words_by_paragraph_start = []
        running_word_count = cumulative_word_offset
        for count in paragraph_word_counts:
            cumulative_words_by_paragraph_start.append(running_word_count)
            running_word_count += count
            cumulative_words_by_paragraph_end.append(running_word_count)

        unit_rows = units_by_chapter.get(chapter_id, [])
        for unit_index_within_chapter, unit in enumerate(unit_rows, start=1):
            cumulative_unit_index += 1
            paragraph_start = unit["paragraphStart"]
            paragraph_end = unit["paragraphEnd"]
            chapter_index[unit["unitId"]] = {
                "volume_number": chapter_spec.volume_number,
                "chapter_id": chapter_id,
                "chapter_title": chapter_titles.get(chapter_id, chapter_id),
                "chapter_index": chapter_number,
                "unit_id": unit["unitId"],
                "unit_index_within_chapter": unit_index_within_chapter,
                "cumulative_unit_index": cumulative_unit_index,
                "paragraph_start": paragraph_start,
                "paragraph_end": paragraph_end,
                "cumulative_paragraph_index": cumulative_paragraph_offset + paragraph_start,
                "cumulative_paragraph_index_end": cumulative_paragraph_offset + paragraph_end,
                "cumulative_word_count": cumulative_words_by_paragraph_start[paragraph_start - 1],
                "cumulative_word_count_end": cumulative_words_by_paragraph_end[paragraph_end - 1],
            }

        cumulative_paragraph_offset += len(paragraphs)
        cumulative_word_offset = running_word_count

    return chapter_index


def build_character_elo(
    run_dirs,
    character_name_map=None,
    lens="advantage",
    initial_rating=1500.0,
    k_factor=24.0,
    epsilon=0.25,
):
    if lens != "advantage":
        raise ValueError("character_elo_v1 currently supports only the advantage lens.")
    if not run_dirs:
        raise ValueError("At least one run directory is required for character ELO.")

    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    overlay_dataset = build_chapter_overlay_data(run_dirs, character_name_map=character_name_map)

    ratings = {}
    diagnostics = {}
    unit_order_rows = []
    draw_count = 0
    match_count = 0

    for chapter in overlay_dataset["chapters"]:
        for unit in chapter["units"]:
            for character_row in unit["characters"]:
                character = character_row["character"]
                ratings.setdefault(character, float(initial_rating))
                diagnostics.setdefault(
                    character,
                    {
                        "character": character,
                        "match_count": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "draw_count": 0,
                        "unit_count": 0,
                        "net_scores": [],
                        "top_positive_unit": None,
                        "top_negative_unit": None,
                    },
                )
                net_score = character_row[lens]["netScore"]
                diagnostics[character]["unit_count"] += 1
                diagnostics[character]["net_scores"].append(net_score)
                positive_unit = diagnostics[character]["top_positive_unit"]
                if positive_unit is None or net_score > positive_unit["net_score"]:
                    diagnostics[character]["top_positive_unit"] = {
                        "unit_id": unit["unitId"],
                        "net_score": net_score,
                    }
                negative_unit = diagnostics[character]["top_negative_unit"]
                if negative_unit is None or net_score < negative_unit["net_score"]:
                    diagnostics[character]["top_negative_unit"] = {
                        "unit_id": unit["unitId"],
                        "net_score": net_score,
                    }

            ordered_rows = sorted(unit["characters"], key=lambda row: row["character"])
            unit_order_rows.append((chapter["chapterId"], unit, ordered_rows))

    for _chapter_id, unit, ordered_rows in unit_order_rows:
        for row_a, row_b in combinations(ordered_rows, 2):
            character_a = row_a["character"]
            character_b = row_b["character"]
            score_a = row_a[lens]["netScore"]
            score_b = row_b[lens]["netScore"]
            observed_a, observed_b = _elo_observed_scores(score_a, score_b, epsilon)

            rating_a = ratings[character_a]
            rating_b = ratings[character_b]
            expected_a = _elo_expected_score(rating_a, rating_b)
            expected_b = _elo_expected_score(rating_b, rating_a)
            ratings[character_a] = rating_a + k_factor * (observed_a - expected_a)
            ratings[character_b] = rating_b + k_factor * (observed_b - expected_b)

            match_count += 1
            if observed_a == observed_b == 0.5:
                draw_count += 1
                diagnostics[character_a]["draw_count"] += 1
                diagnostics[character_b]["draw_count"] += 1
            elif observed_a > observed_b:
                diagnostics[character_a]["win_count"] += 1
                diagnostics[character_b]["loss_count"] += 1
            else:
                diagnostics[character_a]["loss_count"] += 1
                diagnostics[character_b]["win_count"] += 1

            diagnostics[character_a]["match_count"] += 1
            diagnostics[character_b]["match_count"] += 1

    characters = []
    for character, row in diagnostics.items():
        top_positive_unit = row["top_positive_unit"]
        top_negative_unit = row["top_negative_unit"]
        characters.append(
            {
                "character": character,
                "elo": round(ratings[character], 3),
                "match_count": row["match_count"],
                "win_count": row["win_count"],
                "loss_count": row["loss_count"],
                "draw_count": row["draw_count"],
                "unit_count": row["unit_count"],
                "mean_advantage_net_score": round(sum(row["net_scores"]) / len(row["net_scores"]), 3)
                if row["net_scores"]
                else 0.0,
                "top_positive_unit": top_positive_unit,
                "top_negative_unit": top_negative_unit,
            }
        )

    score_rank = {
        row["character"]: rank
        for rank, row in enumerate(
            sorted(
                characters,
                key=lambda item: (-item["mean_advantage_net_score"], item["character"]),
            ),
            start=1,
        )
    }
    elo_rank = {
        row["character"]: rank
        for rank, row in enumerate(
            sorted(
                characters,
                key=lambda item: (-item["elo"], item["character"]),
            ),
            start=1,
        )
    }

    for row in characters:
        row["elo_rank"] = elo_rank[row["character"]]
        row["mean_score_rank"] = score_rank[row["character"]]
        row["elo_rank_minus_mean_score_rank"] = row["elo_rank"] - row["mean_score_rank"]

    characters.sort(key=lambda item: (-item["elo"], item["character"]))
    return {
        "character_elo_version": "character_elo_advantage_v1",
        "lens": lens,
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "duplicate_resolution": overlay_dataset["duplicate_resolution"],
        "initial_rating": initial_rating,
        "k_factor": k_factor,
        "epsilon": epsilon,
        "ordering_rule": "canonical_chapter_then_paragraph_then_unit_then_pair",
        "character_count": len(characters),
        "match_count": match_count,
        "draw_rate": round(draw_count / match_count, 3) if match_count else 0.0,
        "top_rated_characters": characters[:10],
        "lowest_rated_characters": sorted(characters, key=lambda item: (item["elo"], item["character"]))[:10],
        "highest_match_count_characters": sorted(
            characters,
            key=lambda item: (-item["match_count"], -item["unit_count"], item["character"]),
        )[:10],
        "largest_rank_mismatches": sorted(
            characters,
            key=lambda item: (-abs(item["elo_rank_minus_mean_score_rank"]), item["character"]),
        )[:10],
        "characters": characters,
    }


def build_character_elo_timeline(
    run_dirs,
    character_name_map=None,
    lens="advantage",
    target_characters=None,
    initial_rating=1500.0,
    k_factor=24.0,
    epsilon=0.25,
):
    if lens != "advantage":
        raise ValueError("character_elo_timeline_v1 currently supports only the advantage lens.")
    if not run_dirs:
        raise ValueError("At least one run directory is required for character ELO timeline export.")

    selected_characters = list(target_characters or CHARACTER_PAGE_PILOT_EDITORIAL.keys())
    selected_set = set(selected_characters)
    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
    overlay_dataset = build_chapter_overlay_data(run_dirs, character_name_map=character_name_map)

    units_by_chapter = {chapter["chapterId"]: chapter["units"] for chapter in overlay_dataset["chapters"]}
    corpus_positions = _build_corpus_position_index(units_by_chapter)
    ratings = {character: float(initial_rating) for character in selected_characters}
    points = []

    for chapter in overlay_dataset["chapters"]:
        for unit in chapter["units"]:
            ordered_rows = sorted(unit["characters"], key=lambda row: row["character"])
            for row_a, row_b in combinations(ordered_rows, 2):
                character_a = row_a["character"]
                character_b = row_b["character"]
                score_a = row_a[lens]["netScore"]
                score_b = row_b[lens]["netScore"]
                observed_a, observed_b = _elo_observed_scores(score_a, score_b, epsilon)

                ratings.setdefault(character_a, float(initial_rating))
                ratings.setdefault(character_b, float(initial_rating))
                rating_a = ratings[character_a]
                rating_b = ratings[character_b]
                expected_a = _elo_expected_score(rating_a, rating_b)
                expected_b = _elo_expected_score(rating_b, rating_a)
                ratings[character_a] = rating_a + k_factor * (observed_a - expected_a)
                ratings[character_b] = rating_b + k_factor * (observed_b - expected_b)

            position = corpus_positions[unit["unitId"]]
            for character_row in ordered_rows:
                character = character_row["character"]
                if character not in selected_set:
                    continue
                ratings.setdefault(character, float(initial_rating))
                points.append(
                    {
                        "character": character,
                        "elo": round(ratings[character], 3),
                        "advantage_net_score": character_row[lens]["netScore"],
                        "advantage_label": character_row[lens]["label"],
                        "unit_character_count": len(ordered_rows),
                        "corpus_position": dict(position),
                    }
                )

    point_counts = defaultdict(int)
    latest_points = {}
    for point in points:
        character = point["character"]
        point_counts[character] += 1
        latest_points[character] = point

    characters = []
    for character in selected_characters:
        latest = latest_points.get(character)
        characters.append(
            {
                "character": character,
                "point_count": point_counts.get(character, 0),
                "final_elo": latest["elo"] if latest else round(initial_rating, 3),
                "latest_corpus_position": latest["corpus_position"] if latest else None,
            }
        )

    characters.sort(key=lambda item: (-item["point_count"], -item["final_elo"], item["character"]))
    return {
        "character_elo_timeline_version": "character_elo_timeline_v1",
        "lens": lens,
        "source_review_version": review["corpus_review_version"],
        "character_normalization": review.get("character_normalization", {"applied": False, "map": {}}),
        "duplicate_resolution": overlay_dataset["duplicate_resolution"],
        "initial_rating": initial_rating,
        "k_factor": k_factor,
        "epsilon": epsilon,
        "timeline_type": "sparse_by_character_participation",
        "tracked_character_count": len(selected_characters),
        "tracked_characters": selected_characters,
        "point_count": len(points),
        "characters": characters,
        "points": points,
    }


def build_character_profile_cards(run_dirs, character_name_map=None, top_chapter_limit=5):
    review = runner.build_corpus_sanity_review(run_dirs, character_name_map=character_name_map)
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
        before_totals = character_totals_by_name(before_lens["character_totals"])
        after_totals = character_totals_by_name(after_lens["character_totals"])
        positive_ranks_before = character_ranks(before_lens["character_totals"], reverse=True)
        positive_ranks_after = character_ranks(after_lens["character_totals"], reverse=True)
        negative_ranks_before = character_ranks(before_lens["character_totals"], reverse=False)
        negative_ranks_after = character_ranks(after_lens["character_totals"], reverse=False)

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
