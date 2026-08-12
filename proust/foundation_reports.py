"""The two cutover reports the foundation rebuild owes a human reader.

Neither report changes any surface. Both exist because the foundation
corpus (prompt v2, open world, authoritative Wikisource text) replaced the
legacy corpus underneath prose and registry decisions that were made
against the superseded numbers:

- `build_foundation_unresolved_triage` inventories every name prompt v2
  named but could not resolve against the registry reference sheet, with
  a suggested disposition per name. It is the worklist for the next
  `characters.yaml` pass.
- `build_foundation_editorial_discrepancies` re-checks the corpus claims
  embedded in `CHARACTER_PAGE_PILOT_EDITORIAL` against the rebuilt
  surfaces and reports each claim that no longer holds. It deliberately
  does NOT rewrite the editorial: what a page should say instead is a
  human judgement, and the pages ship with the existing text until that
  judgement is made.
"""

from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import unicodedata

from .editorial import CHARACTER_PAGE_PILOT_EDITORIAL
from .registry import Registry
from .reporting_utils import markdown_table
from .scoring import SCORING_LENS_ORDER


UNRESOLVED_TRIAGE_VERSION = "foundation_unresolved_triage_v1"
EDITORIAL_DISCREPANCIES_VERSION = "foundation_editorial_discrepancies_v1"

# A name this close to a name the registry already knows is a typographic
# variant (an apostrophe, an accent, a dropped particle), not a discovery.
NEAR_DUPLICATE_RATIO = 0.88
# A name unresolved in this many distinct units is a recurring figure, so
# the registry not knowing it is a gap rather than a walk-on.
RECURRING_UNIT_COUNT = 2


def _write_artifacts(payload, markdown, json_output=None, markdown_output=None):
    for path, text in ((json_output, None), (markdown_output, markdown)):
        if not path:
            continue
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if text is None:
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            output_path.write_text(text)


def _fold(value):
    """Accent- and punctuation-insensitive key, for near-duplicate detection."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(character for character in decomposed if not unicodedata.combining(character))
    return "".join(character for character in stripped.lower() if character.isalnum() or character == " ").strip()


# ---------------------------------------------------------------- triage


def _registry_known_names(registry):
    names = set()
    for entity in registry.entities.values():
        names.add(entity.display_name)
        names.update(entity.annotation_names or [])
    names.update(form.form for form in registry.forms)
    return {name for name in names if name}


def _closest_known_name(name, known_folded):
    folded = _fold(name)
    best_name = None
    best_ratio = 0.0
    for candidate_folded, candidate in known_folded.items():
        ratio = SequenceMatcher(None, folded, candidate_folded).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = candidate
    return best_name, round(best_ratio, 3)


def build_foundation_unresolved_triage(run_dirs, registry=None):
    """Inventory every unresolved name across the foundation runs' resolutions/ sidecars.

    `write_foundation_result` keeps a per-unit resolution record next to
    each annotation precisely so this survives the v1-shaped annotation
    storage. Each distinct name gets a count, the units it appeared in,
    and one of three suggested dispositions:

    - `possible-error`: the name is a near-duplicate of a name the registry
      already knows, so it is a typographic variant to normalize rather
      than an entity to admit.
    - `likely-registry-gap`: unresolved in more than one unit, i.e. a
      recurring figure the registry does not carry.
    - `one-off-legitimate`: a single-unit walk-on, which prompt v2's open
      world is expected to produce and the registry need not carry.
    """
    registry = registry or Registry.load()
    known_names = _registry_known_names(registry)
    known_folded = {_fold(name): name for name in known_names if _fold(name)}

    names = defaultdict(lambda: {"unit_ids": [], "chapter_ids": set(), "run_ids": set()})
    unit_count = 0
    units_with_unresolved = 0
    unresolved_entry_count = 0
    character_entry_count = 0

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        run_id = run_path.name
        for record_path in sorted((run_path / "resolutions").glob("*.json")):
            record = json.loads(record_path.read_text())
            unit_count += 1
            character_entry_count += record.get("character_count", 0)
            unresolved_names = record.get("unresolved_names") or []
            if unresolved_names:
                units_with_unresolved += 1
            for name in unresolved_names:
                unresolved_entry_count += 1
                row = names[name]
                row["unit_ids"].append(record["unit_id"])
                row["chapter_ids"].add(str(record["unit_id"]).split("#", 1)[0])
                row["run_ids"].add(run_id)

    # Two spellings of one name ("prince d'Agrigente" / "prince d’Agrigente")
    # are one figure for triage purposes, so recurrence is counted over the
    # accent- and punctuation-insensitive group, not over the surface form.
    group_units = defaultdict(set)
    group_names = defaultdict(set)
    for name, row in names.items():
        group_units[_fold(name)].update(row["unit_ids"])
        group_names[_fold(name)].add(name)

    rows = []
    for name, row in names.items():
        closest_name, closest_ratio = _closest_known_name(name, known_folded)
        folded = _fold(name)
        group_unit_count = len(group_units[folded])
        if closest_ratio >= NEAR_DUPLICATE_RATIO:
            disposition = "possible-error"
        elif group_unit_count >= RECURRING_UNIT_COUNT:
            disposition = "likely-registry-gap"
        else:
            disposition = "one-off-legitimate"
        rows.append(
            {
                "name": name,
                "unresolved_count": len(row["unit_ids"]),
                "unit_count": len(set(row["unit_ids"])),
                "group_unit_count": group_unit_count,
                "variant_names": sorted(group_names[folded] - {name}),
                "unit_ids": sorted(set(row["unit_ids"])),
                "chapter_ids": sorted(row["chapter_ids"]),
                "run_ids": sorted(row["run_ids"]),
                "closest_registry_name": closest_name,
                "closest_registry_ratio": closest_ratio,
                "disposition": disposition,
            }
        )

    rows.sort(key=lambda item: (-item["unresolved_count"], item["name"]))
    disposition_counts = defaultdict(int)
    for row in rows:
        disposition_counts[row["disposition"]] += 1

    return {
        "foundation_unresolved_triage_version": UNRESOLVED_TRIAGE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(list(run_dirs)),
        "unit_count": unit_count,
        "character_entry_count": character_entry_count,
        "unresolved_entry_count": unresolved_entry_count,
        "unresolved_unit_count": units_with_unresolved,
        "distinct_name_count": len(rows),
        "near_duplicate_ratio": NEAR_DUPLICATE_RATIO,
        "recurring_unit_count": RECURRING_UNIT_COUNT,
        "disposition_counts": dict(disposition_counts),
        "names": rows,
    }


def render_foundation_unresolved_triage_markdown(report):
    lines = [
        "# Foundation Unresolved-Name Triage",
        "",
        f"- Report version: `{report['foundation_unresolved_triage_version']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Foundation runs: `{report['run_count']}`, units: `{report['unit_count']}`",
        f"- Character entries: `{report['character_entry_count']}`",
        f"- Unresolved entries: `{report['unresolved_entry_count']}` "
        f"across `{report['unresolved_unit_count']}` units",
        f"- Distinct unresolved names: `{report['distinct_name_count']}`",
        "",
        "## Dispositions",
        "",
        "Suggested, not applied. `possible-error` means the name is within a "
        f"{report['near_duplicate_ratio']} similarity of a name the registry already knows "
        "(accent-, case-, and punctuation-insensitive), so it reads as a typographic variant to "
        "normalize rather than an entity to admit. `likely-registry-gap` means the name went "
        f"unresolved in at least {report['recurring_unit_count']} distinct units, i.e. a recurring "
        "figure `characters.yaml` does not carry. `one-off-legitimate` is a single-unit walk-on, "
        "which prompt v2's open world is supposed to produce.",
        "",
        markdown_table(
            ["Disposition", "Names"],
            sorted(report["disposition_counts"].items(), key=lambda item: (-item[1], item[0])),
        ),
        "",
        "## Unresolved names",
        "",
        markdown_table(
            ["Name", "Entries", "Units", "Spelling variants", "Chapters", "Closest registry name", "Ratio", "Disposition"],
            [
                (
                    row["name"],
                    row["unresolved_count"],
                    row["unit_count"],
                    ", ".join(row["variant_names"]),
                    ", ".join(row["chapter_ids"]),
                    row["closest_registry_name"] or "",
                    row["closest_registry_ratio"],
                    row["disposition"],
                )
                for row in report["names"]
            ],
        ),
        "",
        "## Unit references",
        "",
    ]
    for row in report["names"]:
        lines.append(f"- `{row['name']}`: {', '.join(row['unit_ids'])}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_foundation_unresolved_triage_artifacts(report, json_output=None, markdown_output=None):
    _write_artifacts(
        report,
        render_foundation_unresolved_triage_markdown(report),
        json_output=json_output,
        markdown_output=markdown_output,
    )


# ------------------------------------------------------- editorial claims

# The checkable corpus claims embedded in CHARACTER_PAGE_PILOT_EDITORIAL,
# transcribed by hand from the prose that carries them. Every claim names
# the field it comes from, so a reader can find and rewrite the sentence.
# Check kinds:
#   lens_signs           every named lens's net score has the named sign
#   lens_ranks_max       every named lens's rank is at least this good
#   lens_rank_better     one lens ranks better than each of the others
#   unit_count_rank_max  the character is among the N most-scored
#   unit_count_rank_min  the character is NOT among the N most-scored
#   rank_spread_rank_max the character is among the N widest lens spreads
#   score_span_rank_max  the character is among the N most volatile
#   mean_abs_max         every named lens's mean score is this close to zero
#   extreme_lens         the named lens is the most positive / most negative
EDITORIAL_CLAIM_CHECKS = {
    "le narrateur": [
        {
            "claim": "first in advantage and inclusion among the novel's central figures",
            "field": "subheading",
            "check": {"kind": "lens_ranks_max", "ranks": {"advantage": 1, "inclusion": 1}},
        },
        {
            "claim": "average scores that stay stubbornly near zero",
            "field": "subheading",
            "check": {"kind": "mean_abs_max", "lenses": ["advantage", "prestige", "inclusion"], "max_abs": 0.2},
        },
        {
            "claim": "he meets more of the cast, more often, than any other figure",
            "field": "summary",
            "check": {"kind": "unit_count_rank_max", "max_rank": 1},
        },
    ],
    "Odette": [
        {
            "claim": "Prestige-positive but inclusion-negative",
            "field": "subheading",
            "check": {"kind": "lens_signs", "signs": {"prestige": "positive", "inclusion": "negative"}},
        },
    ],
    "Robert de Saint-Loup": [
        {
            "claim": "shows one of the largest lens spreads in the novel",
            "field": "summary",
            "check": {"kind": "rank_spread_rank_max", "max_rank": 15},
        },
        {
            "claim": "prestige often holds even where belonging and immediate advantage give way",
            "field": "subheading",
            "check": {"kind": "lens_rank_better", "lens": "prestige", "than": ["advantage", "inclusion"]},
        },
    ],
    "Swann": [
        {
            "claim": "The most-scored figure in the novel",
            "field": "subheading",
            "check": {"kind": "unit_count_rank_max", "max_rank": 1},
        },
        {
            "claim": "broadly and repeatedly negative across all three lenses",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "Albertine": [
        {
            "claim": "one of the largest and most persistently negative figures in the novel",
            "field": "summary",
            "check": {"kind": "unit_count_rank_max", "max_rank": 5},
        },
        {
            "claim": "persistently negative",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "baron de Charlus": [
        {
            "claim": "A highly volatile major figure",
            "field": "subheading",
            "check": {"kind": "score_span_rank_max", "max_rank": 15},
        },
        {
            "claim": "whose negative scores are spread across salon, sexual, and wartime terrains",
            "field": "subheading",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "duchesse de Guermantes": [
        {
            "claim": "The novel's clearest uniformly positive great-world figure",
            "field": "subheading",
            "check": {"kind": "lens_ranks_max", "ranks": {"advantage": 1, "prestige": 1, "inclusion": 1}},
        },
        {
            "claim": "command and symbolic force holding across all three lenses",
            "field": "subheading",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "positive", "prestige": "positive", "inclusion": "positive"},
            },
        },
    ],
    "Mme de Villeparisis": [
        {
            "claim": "relatively strong in prestige while advantage and inclusion drift downward",
            "field": "subheading",
            "check": {"kind": "lens_rank_better", "lens": "prestige", "than": ["advantage", "inclusion"]},
        },
        {
            "claim": "advantage and inclusion drift downward",
            "field": "subheading",
            "check": {"kind": "lens_signs", "signs": {"advantage": "negative", "inclusion": "negative"}},
        },
    ],
    "Françoise": [
        {
            "claim": "accumulates as a broadly negative figure across the book",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "Mme Verdurin": [
        {
            "claim": "losses in advantage, prestige, and inclusion all reinforcing rather than offsetting one another",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "Gilberte": [
        {
            "claim": "she scores very well in prestige and immediate advantage, yet her inclusion profile remains markedly less secure",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "positive", "prestige": "positive", "inclusion": "negative"},
            },
        },
    ],
    "Norpois": [
        {
            "claim": "a strongly positive figure across all three lenses",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "positive", "prestige": "positive", "inclusion": "positive"},
            },
        },
    ],
    "la grand-mère": [
        {
            "claim": "the harshest pressure falling on inclusion and broad valuation",
            "field": "summary",
            "check": {"kind": "extreme_lens", "lens": "inclusion", "direction": "negative"},
        },
        {
            "claim": "one of the book's more strongly negative recurring figures",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "Bloch": [
        {
            "claim": "repeated losses in advantage, prestige, and inclusion reinforcing each other",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "duc de Guermantes": [
        {
            "claim": "despite formal rank, his scores are broadly negative across all three lenses",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "docteur Cottard": [
        {
            "claim": "A mid-tier negative figure",
            "field": "subheading",
            "check": {"kind": "lens_percentile_band", "lens": "advantage", "low": 20, "high": 80},
        },
    ],
    "la mère du narrateur": [
        {
            "claim": "quietly high-performing figure across all three lenses",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "positive", "prestige": "positive", "inclusion": "positive"},
            },
        },
        {
            "claim": "especially strong advantage and inclusion",
            "field": "summary",
            "check": {"kind": "lens_rank_better", "lens": "advantage", "than": ["prestige"]},
        },
    ],
    "Bergotte": [
        {
            "claim": "his literary authority translating into very high advantage and prestige",
            "field": "summary",
            "check": {"kind": "lens_signs", "signs": {"advantage": "positive", "prestige": "positive"}},
        },
        {
            "claim": "one of the novel's clearest positive symbolic figures",
            "field": "summary",
            "check": {"kind": "lens_ranks_max", "ranks": {"advantage": 10}},
        },
    ],
    "Legrandin": [
        {
            "claim": "A broadly negative recurring figure marked by repeated self-positioning failures",
            "field": "subheading",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
    ],
    "Mme de Cambremer": [
        {
            "claim": "strongly downward wherever she appears",
            "field": "subheading",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "negative", "prestige": "negative", "inclusion": "negative"},
            },
        },
        {
            "claim": "she doesn't appear as often as the novel's biggest characters",
            "field": "summary",
            "check": {"kind": "unit_count_rank_min", "min_rank": 20},
        },
    ],
    "M. Vinteuil": [
        {
            "claim": "his scores end up decisively positive, especially in inclusion",
            "field": "summary",
            "check": {
                "kind": "lens_signs",
                "signs": {"advantage": "positive", "prestige": "positive", "inclusion": "positive"},
            },
        },
        {
            "claim": "especially in inclusion",
            "field": "summary",
            "check": {"kind": "extreme_lens", "lens": "inclusion", "direction": "positive"},
        },
    ],
}


def _cards_by_character(cards):
    return {row["character"]: row for row in cards["cards"]}


def _unit_count_ranks(cards):
    ordered = sorted(
        cards["cards"],
        key=lambda row: (-row["annotation_unit_count"], row["character"]),
    )
    return {row["character"]: index for index, row in enumerate(ordered, start=1)}


def _metric_ranks(cards, key):
    ordered = sorted(cards["cards"], key=lambda row: (-row[key], row["character"]))
    return {row["character"]: index for index, row in enumerate(ordered, start=1)}


def _sign(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _evaluate_claim(check, character, cards, context):
    """(holds, rendered value) for one claim against one profile-cards surface."""
    row = _cards_by_character(cards).get(character)
    if row is None:
        return False, "absent from the corpus"
    lens_scores = row["lens_scores"]
    kind = check["kind"]

    if kind == "lens_signs":
        holds = all(_sign(lens_scores[lens]["net_score"]) == sign for lens, sign in check["signs"].items())
        value = ", ".join(
            f"{lens} {lens_scores[lens]['net_score']:+.1f}" for lens in check["signs"]
        )
        return holds, value

    if kind == "lens_ranks_max":
        population = len(cards["cards"])
        holds = all(lens_scores[lens]["rank"] <= max_rank for lens, max_rank in check["ranks"].items())
        value = ", ".join(f"{lens} rank {lens_scores[lens]['rank']}/{population}" for lens in check["ranks"])
        return holds, value

    if kind == "lens_rank_better":
        lens = check["lens"]
        holds = all(lens_scores[lens]["rank"] < lens_scores[other]["rank"] for other in check["than"])
        value = ", ".join(
            f"{name} rank {lens_scores[name]['rank']}" for name in [lens] + list(check["than"])
        )
        return holds, value

    if kind in ("unit_count_rank_max", "unit_count_rank_min"):
        rank = context["unit_count_ranks"][character]
        population = len(cards["cards"])
        if kind == "unit_count_rank_max":
            holds = rank <= check["max_rank"]
        else:
            holds = rank > check["min_rank"]
        return holds, f"{row['annotation_unit_count']} units, rank {rank}/{population}"

    if kind == "rank_spread_rank_max":
        rank = context["rank_spread_ranks"][character]
        return rank <= check["max_rank"], f"rank spread {row['rank_spread']}, rank {rank}"

    if kind == "score_span_rank_max":
        rank = context["score_span_ranks"][character]
        return rank <= check["max_rank"], f"max score span {row['max_score_span']}, rank {rank}"

    if kind == "mean_abs_max":
        means = {lens: lens_scores[lens]["mean_score"] for lens in check["lenses"]}
        holds = all(abs(value) <= check["max_abs"] for value in means.values())
        return holds, ", ".join(f"{lens} mean {value:+.3f}" for lens, value in means.items())

    if kind == "extreme_lens":
        if check["direction"] == "negative":
            extreme = min(SCORING_LENS_ORDER, key=lambda lens: lens_scores[lens]["net_score"])
        else:
            extreme = max(SCORING_LENS_ORDER, key=lambda lens: lens_scores[lens]["net_score"])
        holds = extreme == check["lens"] and _sign(lens_scores[extreme]["net_score"]) == check["direction"]
        value = ", ".join(f"{lens} {lens_scores[lens]['net_score']:+.1f}" for lens in SCORING_LENS_ORDER)
        return holds, value

    if kind == "lens_percentile_band":
        percentile = lens_scores[check["lens"]]["percentile"]
        holds = check["low"] <= percentile <= check["high"]
        return holds, f"{check['lens']} percentile {percentile}"

    raise ValueError(f'Unknown editorial claim check kind "{kind}".')


def _claim_context(cards):
    return {
        "unit_count_ranks": _unit_count_ranks(cards),
        "rank_spread_ranks": _metric_ranks(cards, "rank_spread"),
        "score_span_ranks": _metric_ranks(cards, "max_score_span"),
    }


def _reading_path_direction(label):
    lowered = label.lower()
    if "negative" in lowered:
        return "negative"
    if "positive" in lowered:
        return "positive"
    return None


def build_foundation_editorial_discrepancies(
    foundation_cards,
    baseline_cards,
    foundation_chapter_analysis=None,
    foundation_whr=None,
    baseline_whr=None,
    ambiguity_statistics=None,
):
    """Re-check the pilot editorial's corpus claims against the rebuilt surfaces.

    `foundation_cards` / `baseline_cards` are profile-card analyses (the
    surface that carries every per-character rank, unit count, lens score,
    spread, and span the editorial prose asserts) built from the foundation
    corpus and from the superseded corpus the prose was written against.
    Each claim is evaluated on both, and every claim that no longer holds is
    reported with the value it had then and the value it has now.
    """
    foundation_context = _claim_context(foundation_cards)
    baseline_context = _claim_context(baseline_cards)

    claims = []
    for character, character_claims in EDITORIAL_CLAIM_CHECKS.items():
        for entry in character_claims:
            baseline_holds, baseline_value = _evaluate_claim(
                entry["check"], character, baseline_cards, baseline_context
            )
            foundation_holds, foundation_value = _evaluate_claim(
                entry["check"], character, foundation_cards, foundation_context
            )
            claims.append(
                {
                    "character": character,
                    "claim": entry["claim"],
                    "field": entry["field"],
                    "check_kind": entry["check"]["kind"],
                    "held_before": baseline_holds,
                    "holds_now": foundation_holds,
                    "old_value": baseline_value,
                    "new_value": foundation_value,
                }
            )

    reading_path_claims = []
    chapter_rows = {}
    if foundation_chapter_analysis:
        chapter_rows = {
            row["character"]: {chapter["chapter_id"]: chapter for chapter in row["chapters"]}
            for row in foundation_chapter_analysis["characters"]
        }
    for character, editorial in CHARACTER_PAGE_PILOT_EDITORIAL.items():
        for step in editorial["reading_path"]:
            chapter_id = step["chapter_id"]
            direction = _reading_path_direction(step["label"])
            chapter = chapter_rows.get(character, {}).get(chapter_id)
            if chapter is None:
                new_value = "no scored units in this chapter"
                holds = False
            else:
                new_value = ", ".join(
                    f"{lens} {chapter[lens]['net_score']:+.1f}" for lens in SCORING_LENS_ORDER
                )
                holds = direction is None or _sign(chapter["advantage"]["net_score"]) == direction
            reading_path_claims.append(
                {
                    "character": character,
                    "chapter_id": chapter_id,
                    "label": step["label"],
                    "asserted_direction": direction,
                    "holds_now": holds,
                    "new_value": new_value,
                }
            )

    baseline_cards_by_character = _cards_by_character(baseline_cards)
    foundation_cards_by_character = _cards_by_character(foundation_cards)
    baseline_whr_rows = (
        {row["character"]: row for row in baseline_whr["characters"]} if baseline_whr else {}
    )
    foundation_whr_rows = (
        {row["character"]: row for row in foundation_whr["characters"]} if foundation_whr else {}
    )

    character_deltas = []
    for character in CHARACTER_PAGE_PILOT_EDITORIAL:
        before = baseline_cards_by_character.get(character)
        after = foundation_cards_by_character.get(character)
        character_deltas.append(
            {
                "character": character,
                "unit_count_before": before["annotation_unit_count"] if before else 0,
                "unit_count_after": after["annotation_unit_count"] if after else 0,
                "population_before": len(baseline_cards["cards"]),
                "population_after": len(foundation_cards["cards"]),
                "lens_ranks_before": {
                    lens: before["lens_scores"][lens]["rank"] for lens in SCORING_LENS_ORDER
                }
                if before
                else None,
                "lens_ranks_after": {
                    lens: after["lens_scores"][lens]["rank"] for lens in SCORING_LENS_ORDER
                }
                if after
                else None,
                "lens_net_scores_before": {
                    lens: before["lens_scores"][lens]["net_score"] for lens in SCORING_LENS_ORDER
                }
                if before
                else None,
                "lens_net_scores_after": {
                    lens: after["lens_scores"][lens]["net_score"] for lens in SCORING_LENS_ORDER
                }
                if after
                else None,
                "match_count_before": baseline_whr_rows.get(character, {}).get("match_count"),
                "match_count_after": foundation_whr_rows.get(character, {}).get("match_count"),
            }
        )

    discrepant_claims = [row for row in claims if not row["holds_now"]]
    discrepant_reading_path = [row for row in reading_path_claims if not row["holds_now"]]

    return {
        "foundation_editorial_discrepancies_version": EDITORIAL_DISCREPANCIES_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "character_count": len(CHARACTER_PAGE_PILOT_EDITORIAL),
        "claim_count": len(claims),
        "discrepant_claim_count": len(discrepant_claims),
        "reading_path_claim_count": len(reading_path_claims),
        "discrepant_reading_path_count": len(discrepant_reading_path),
        "claims": claims,
        "reading_path_claims": reading_path_claims,
        "character_deltas": character_deltas,
        "ambiguity_statistics": ambiguity_statistics,
    }


def build_ambiguity_statistics(run_dirs, baseline_run_dirs=None):
    """Per-unit ambiguity-flag density, which the scoring config turns into a per-character penalty.

    Every character scored in a unit loses `ambiguity_penalty` per flag on
    that unit, so a corpus that flags more ambiguity moves every recurring
    character's net score down by a near-constant amount without any change
    to the scoring config. This is measurement for the reader of the
    editorial report, not an argument for changing the config.
    """

    def measure(dirs):
        unit_count = 0
        ambiguity_total = 0
        character_total = 0
        for run_dir in dirs or []:
            for annotation_path in sorted((Path(run_dir) / "annotations").glob("*.json")):
                annotation = json.loads(annotation_path.read_text())
                if not isinstance(annotation, dict) or "ambiguities" not in annotation:
                    continue
                unit_count += 1
                ambiguity_total += len(annotation["ambiguities"] or [])
                character_total += len(annotation.get("characters_present") or [])
        if not unit_count:
            return None
        return {
            "unit_count": unit_count,
            "mean_ambiguities_per_unit": round(ambiguity_total / unit_count, 3),
            "mean_characters_per_unit": round(character_total / unit_count, 3),
        }

    return {"foundation": measure(run_dirs), "baseline": measure(baseline_run_dirs)}


def render_foundation_editorial_discrepancies_markdown(report):
    lines = [
        "# Foundation Editorial Discrepancies",
        "",
        f"- Report version: `{report['foundation_editorial_discrepancies_version']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Pilot characters: `{report['character_count']}`",
        f"- Prose claims checked: `{report['claim_count']}`, no longer holding: "
        f"`{report['discrepant_claim_count']}`",
        f"- Reading-path steps checked: `{report['reading_path_claim_count']}`, no longer holding: "
        f"`{report['discrepant_reading_path_count']}`",
        "",
        "The character pages ship with the existing `CHARACTER_PAGE_PILOT_EDITORIAL` text. This "
        "report exists so the prose can be rewritten with human judgement rather than silently "
        "patched: it lists each corpus claim the editorial makes, the value that claim had under "
        "the superseded corpus it was written against, and the value it has under the foundation "
        "corpus. Nothing here was applied to any artifact.",
        "",
        "## Claims that no longer hold",
        "",
        markdown_table(
            ["Character", "Claim", "Field", "Old-implied value", "New value"],
            [
                (row["character"], row["claim"], row["field"], row["old_value"], row["new_value"])
                for row in report["claims"]
                if not row["holds_now"]
            ],
        ),
        "",
        "## Claims that still hold",
        "",
        markdown_table(
            ["Character", "Claim", "Old-implied value", "New value"],
            [
                (row["character"], row["claim"], row["old_value"], row["new_value"])
                for row in report["claims"]
                if row["holds_now"]
            ],
        ),
        "",
        "## Reading-path steps whose stated direction no longer holds",
        "",
        "A reading-path label that names a direction (\"Primary negative concentration\") is a "
        "claim about that chapter's scores for that character. Steps with no direction in the "
        "label are checked only for the character having scored units in the chapter at all.",
        "",
        markdown_table(
            ["Character", "Chapter", "Label", "Asserted", "New value"],
            [
                (
                    row["character"],
                    row["chapter_id"],
                    row["label"],
                    row["asserted_direction"] or "(none)",
                    row["new_value"],
                )
                for row in report["reading_path_claims"]
                if not row["holds_now"]
            ],
        ),
        "",
        "## Per-character corpus deltas",
        "",
        markdown_table(
            ["Character", "Units before", "Units after", "Ranks before", "Ranks after", "Matches before", "Matches after"],
            [
                (
                    row["character"],
                    row["unit_count_before"],
                    row["unit_count_after"],
                    _format_ranks(row["lens_ranks_before"], row["population_before"]),
                    _format_ranks(row["lens_ranks_after"], row["population_after"]),
                    row["match_count_before"] if row["match_count_before"] is not None else "",
                    row["match_count_after"] if row["match_count_after"] is not None else "",
                )
                for row in report["character_deltas"]
            ],
        ),
        "",
    ]

    statistics = report.get("ambiguity_statistics") or {}
    if statistics.get("foundation"):
        foundation = statistics["foundation"]
        baseline = statistics.get("baseline")
        lines.extend(
            [
                "## Corpus-level context",
                "",
                "The scoring config is unchanged (v1 weights, thresholds, and ambiguity penalty), so "
                "every movement above comes from the corpus. One corpus-level difference explains a "
                "large share of it: the ambiguity penalty is subtracted from EVERY character scored "
                "in a unit, once per ambiguity flag on that unit, and prompt v2 raises ambiguities far "
                "more often than the legacy prompt did.",
                "",
                markdown_table(
                    ["Corpus", "Units", "Mean ambiguity flags per unit", "Mean characters per unit"],
                    [
                        row
                        for row in [
                            (
                                "foundation",
                                foundation["unit_count"],
                                foundation["mean_ambiguities_per_unit"],
                                foundation["mean_characters_per_unit"],
                            ),
                            (
                                "legacy (superseded)",
                                baseline["unit_count"],
                                baseline["mean_ambiguities_per_unit"],
                                baseline["mean_characters_per_unit"],
                            )
                            if baseline
                            else None,
                        ]
                        if row
                    ],
                ),
                "",
                "Reading that as an editorial matter rather than a scoring one: a recurring character "
                "now carries a near-constant per-unit deduction that a walk-on barely feels, so "
                "sum-of-net-score standings tilt against exactly the figures the pilot pages are "
                "about. Whether that is the right model is the separately flagged ambiguity-penalty "
                "decision; it was deliberately NOT touched by this rebuild.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _format_ranks(ranks, population):
    if not ranks:
        return ""
    return ", ".join(f"{lens[:3]} {ranks[lens]}" for lens in SCORING_LENS_ORDER) + f" / {population}"


def write_foundation_editorial_discrepancies_artifacts(report, json_output=None, markdown_output=None):
    _write_artifacts(
        report,
        render_foundation_editorial_discrepancies_markdown(report),
        json_output=json_output,
        markdown_output=markdown_output,
    )


__all__ = [
    "EDITORIAL_CLAIM_CHECKS",
    "build_ambiguity_statistics",
    "build_foundation_editorial_discrepancies",
    "build_foundation_unresolved_triage",
    "render_foundation_editorial_discrepancies_markdown",
    "render_foundation_unresolved_triage_markdown",
    "write_foundation_editorial_discrepancies_artifacts",
    "write_foundation_unresolved_triage_artifacts",
]
