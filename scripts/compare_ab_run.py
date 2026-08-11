"""Compare a prepared A/B run (scripts/prepare_ab_run.py) once annotations land.

Reads outputs/ab-run-001/manifest.json and, for each unit, whatever of
accepted/<legacy_unit_id>.json, annotations-a/<new_unit_id>.json, and
annotations-b/<new_unit_id>.json currently exist -- annotation happens in a
separate orchestrated run, so this script is designed to run repeatedly
against a partially-filled directory and report "pending" for what's missing
rather than failing.

For each unit with at least one side present it reports:

  - characters_present in accepted vs A vs B (B entries carry a resolution
    flag: "resolved" | "unresolved").
  - B-only discoveries: characters B lists that neither accepted nor A do --
    the open-world gain the registry design doc is aiming for (Rachel,
    l'amie de Mlle Vinteuil, dame-en-rose attribution, etc.).
  - missing-vs-accepted: accepted characters absent from A / from B.
  - per-lens (advantage / prestige / inclusion) net-score direction
    agreement for A-vs-accepted, B-vs-accepted, A-vs-B, scored with the same
    weights proust.runner._score_run_outcomes uses (event score + status
    score - ambiguity penalty), reduced to direction = sign(net_score) with
    a 0.25 neutral band.

Aggregates across all units: open-world discovery count, direction agreement
rates per lens/pair, and unresolved-name counts split into a heuristic
"legitimate off-sheet name" vs "possible registry gap" classification (see
classify_unresolved) -- final adjudication of that split is a human call, so
the report flags it as advisory, not a verdict.

Usage:  python scripts/compare_ab_run.py [--dir outputs/ab-run-001] [--out outputs/ab-run-001/ab-report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from proust.registry import Registry, normalize_text  # noqa: E402
from proust.runner import _outcome_event_score, _outcome_status_score, _resolve_scoring_lens  # noqa: E402
from proust.scoring import SCORING_LENS_ORDER  # noqa: E402

DEFAULT_AB_RUN_DIR = REPO / "outputs" / "ab-run-001"
DIRECTION_BAND = 0.25


# --------------------------------------------------------------------- I/O


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path):
    return _read_json(path) if path.exists() else None


# ------------------------------------------------------------------ scoring


def direction(net_score: float, band: float = DIRECTION_BAND) -> str:
    if net_score > band:
        return "positive"
    if net_score < -band:
        return "negative"
    return "neutral"


def score_annotation_by_character(annotation: dict | None, lens_config: dict) -> dict:
    """canonical_name -> net_score, mirroring the per-unit inner loop of
    proust.runner._score_run_outcomes (event_score + status_score -
    ambiguity_penalty), without needing a full run directory on disk."""
    if not annotation:
        return {}
    ambiguity_penalty = len(annotation.get("ambiguities") or []) * lens_config["ambiguity_penalty"]

    accum = {}
    for character in annotation.get("characters_present") or []:
        name = character.get("canonical_name")
        if name is not None:
            accum[name] = {"event_score": 0.0, "status_score": 0.0}

    for event in annotation.get("appraisal_events") or []:
        target = event.get("target")
        if target in accum:
            accum[target]["event_score"] += _outcome_event_score(event, lens_config)

    for effect in annotation.get("status_effects") or []:
        character = effect.get("character")
        if character in accum:
            accum[character]["status_score"] += _outcome_status_score(effect, lens_config)

    return {
        name: round(scores["event_score"] + scores["status_score"] - ambiguity_penalty, 3)
        for name, scores in accum.items()
    }


def direction_agreement(net_a: dict, net_b: dict, band: float = DIRECTION_BAND) -> dict:
    """Compare direction for characters present on BOTH sides."""
    shared = sorted(set(net_a) & set(net_b))
    characters = {}
    agree_count = 0
    for name in shared:
        dir_a = direction(net_a[name], band)
        dir_b = direction(net_b[name], band)
        agree = dir_a == dir_b
        agree_count += agree
        characters[name] = {"left": dir_a, "right": dir_b, "agree": agree}
    return {"compared": len(shared), "agree": agree_count, "characters": characters}


# ------------------------------------------------------------- characters


def characters_present_names(annotation: dict | None) -> list:
    if not annotation:
        return []
    return [c.get("canonical_name") for c in (annotation.get("characters_present") or []) if c.get("canonical_name")]


def characters_present_with_resolution(annotation: dict | None) -> list:
    if not annotation:
        return []
    out = []
    for c in annotation.get("characters_present") or []:
        name = c.get("canonical_name")
        if name is None:
            continue
        out.append({"canonical_name": name, "resolution": c.get("resolution", "resolved")})
    return out


def classify_unresolved(registry: Registry, surface_form: str) -> str:
    """Heuristic triage for a name arm B marked resolution:"unresolved".

    - "registry_miss_model_error": the registry actually has a form that
      resolves (or is ambiguous for) this exact surface -- the model should
      have matched the reference sheet; not a registry gap.
    - "possible_registry_gap": no exact registry form matches, but a
      substring-level match against a known form suggests this may be a
      spelling/case variant the registry should also carry.
    - "legitimate_off_sheet": no registry evidence either way -- consistent
      with a genuinely minor/one-off named figure the registry, by design,
      does not catalog.

    This is advisory triage, not adjudication: see
    proust/docs/character_registry_design.md's "Decision queue for Nathan".
    """
    resolution = registry.resolve(surface_form)
    if resolution.status in ("resolved", "ambiguous"):
        return "registry_miss_model_error"
    norm = normalize_text(surface_form.strip()).lower()
    if norm:
        for form in registry.forms:
            candidate = normalize_text(form.form).lower()
            if candidate and (candidate in norm or norm in candidate):
                return "possible_registry_gap"
    return "legitimate_off_sheet"


# ------------------------------------------------------------------- unit


def compare_unit(entry: dict, ab_run_dir: Path, registry: Registry) -> dict:
    legacy_unit_id = entry["legacy_unit_id"]
    new_unit_id = entry["new_unit_id"]

    accepted = _load_optional(ab_run_dir / "accepted" / f"{legacy_unit_id}.json")
    annotation_a = _load_optional(ab_run_dir / "annotations-a" / f"{new_unit_id}.json")
    annotation_b = _load_optional(ab_run_dir / "annotations-b" / f"{new_unit_id}.json")

    names_accepted = set(characters_present_names(accepted))
    names_a = set(characters_present_names(annotation_a))
    b_entries = characters_present_with_resolution(annotation_b)
    names_b = {c["canonical_name"] for c in b_entries}

    b_only = sorted(names_b - names_accepted - names_a)
    b_only_entries = [c for c in b_entries if c["canonical_name"] in b_only]

    unresolved_entries = [c for c in b_entries if c.get("resolution") == "unresolved"]
    unresolved_report = [
        {
            "canonical_name": c["canonical_name"],
            "classification": classify_unresolved(registry, c["canonical_name"]),
        }
        for c in unresolved_entries
    ]

    direction_by_lens = {}
    for lens in SCORING_LENS_ORDER:
        lens_config = _resolve_scoring_lens(lens)
        net_accepted = score_annotation_by_character(accepted, lens_config)
        net_a = score_annotation_by_character(annotation_a, lens_config)
        net_b = score_annotation_by_character(annotation_b, lens_config)
        direction_by_lens[lens] = {
            "a_vs_accepted": direction_agreement(net_a, net_accepted),
            "b_vs_accepted": direction_agreement(net_b, net_accepted),
            "a_vs_b": direction_agreement(net_a, net_b),
        }

    return {
        "legacy_unit_id": legacy_unit_id,
        "new_unit_id": new_unit_id,
        "chapter_id": entry["chapter_id"],
        "notes": entry["notes"],
        "present": {
            "accepted": accepted is not None,
            "a": annotation_a is not None,
            "b": annotation_b is not None,
        },
        "characters": {
            "accepted": sorted(names_accepted),
            "a": sorted(names_a),
            "b": b_entries,
        },
        "b_only_discoveries": b_only_entries,
        "missing_vs_accepted": {
            "a": sorted(names_accepted - names_a) if annotation_a is not None else None,
            "b": sorted(names_accepted - names_b) if annotation_b is not None else None,
        },
        "unresolved_in_b": unresolved_report,
        "direction_by_lens": direction_by_lens,
    }


# ---------------------------------------------------------------- aggregate


def build_aggregates(unit_reports: list) -> dict:
    units_with = {side: sum(1 for u in unit_reports if u["present"][side]) for side in ("accepted", "a", "b")}
    units_complete = sum(1 for u in unit_reports if all(u["present"].values()))

    open_world_names = set()
    for u in unit_reports:
        for c in u["b_only_discoveries"]:
            open_world_names.add(c["canonical_name"])
    open_world_total = sum(len(u["b_only_discoveries"]) for u in unit_reports)

    unresolved_counts = {"total": 0, "registry_miss_model_error": 0, "possible_registry_gap": 0, "legitimate_off_sheet": 0}
    for u in unit_reports:
        for entry in u["unresolved_in_b"]:
            unresolved_counts["total"] += 1
            unresolved_counts[entry["classification"]] += 1

    direction_rates = {}
    for lens in SCORING_LENS_ORDER:
        direction_rates[lens] = {}
        for pair in ("a_vs_accepted", "b_vs_accepted", "a_vs_b"):
            compared = sum(u["direction_by_lens"][lens][pair]["compared"] for u in unit_reports)
            agree = sum(u["direction_by_lens"][lens][pair]["agree"] for u in unit_reports)
            rate = round(agree / compared, 3) if compared else None
            direction_rates[lens][pair] = {"compared": compared, "agree": agree, "rate": rate}

    return {
        "unit_count": len(unit_reports),
        "units_with_accepted": units_with["accepted"],
        "units_with_annotation_a": units_with["a"],
        "units_with_annotation_b": units_with["b"],
        "units_fully_complete": units_complete,
        "open_world_discovery_count": open_world_total,
        "open_world_discovery_names": sorted(open_world_names),
        "unresolved_counts": unresolved_counts,
        "direction_agreement_rates": direction_rates,
    }


# ------------------------------------------------------------------- report


def _status_cell(present: bool) -> str:
    return "yes" if present else "pending"


def _fmt_rate(rate_entry: dict) -> str:
    compared = rate_entry["compared"]
    if not compared:
        return "n/a (0 comparable)"
    rate = rate_entry.get("rate")
    if rate is None:
        rate = rate_entry["agree"] / compared
    return f"{rate:.0%} ({rate_entry['agree']}/{compared})"


def render_report(unit_reports: list, aggregates: dict) -> str:
    lines = []
    lines.append("# Prompt-v2 A/B report")
    lines.append("")
    lines.append(
        "Arm A = prompt v1 + legacy alias map (lifted from the accepted run). "
        "Arm B = prompt v2 + registry reference sheet. Both on the current "
        "(Wikisource) canonical text; `accepted` is the legacy annotation, "
        "kept as a reference point, not ground truth. Direction = "
        f"sign(net_score) with a {DIRECTION_BAND} neutral band; net_score "
        "reuses proust.runner's scoring weights per lens."
    )
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    lines.append(f"- Units: {aggregates['unit_count']}")
    lines.append(
        f"- Present: accepted={aggregates['units_with_accepted']}, "
        f"A={aggregates['units_with_annotation_a']}, B={aggregates['units_with_annotation_b']}, "
        f"fully complete={aggregates['units_fully_complete']}"
    )
    lines.append(
        f"- Open-world discoveries (B-only, not in accepted or A): "
        f"{aggregates['open_world_discovery_count']} instances, "
        f"{len(aggregates['open_world_discovery_names'])} distinct names: "
        f"{', '.join(aggregates['open_world_discovery_names']) or '(none yet)'}"
    )
    unresolved = aggregates["unresolved_counts"]
    lines.append(
        f"- Unresolved names in B: {unresolved['total']} total -- "
        f"legitimate_off_sheet={unresolved['legitimate_off_sheet']}, "
        f"possible_registry_gap={unresolved['possible_registry_gap']}, "
        f"registry_miss_model_error={unresolved['registry_miss_model_error']} "
        "(heuristic triage; final call is human, see the design doc's decision queue)"
    )
    lines.append("")
    lines.append("### Direction agreement rates")
    lines.append("")
    lines.append("| lens | A vs accepted | B vs accepted | A vs B |")
    lines.append("| --- | --- | --- | --- |")
    for lens in SCORING_LENS_ORDER:
        rates = aggregates["direction_agreement_rates"][lens]
        lines.append(
            f"| {lens} | {_fmt_rate(rates['a_vs_accepted'])} | {_fmt_rate(rates['b_vs_accepted'])} "
            f"| {_fmt_rate(rates['a_vs_b'])} |"
        )
    lines.append("")
    lines.append("## Units")
    lines.append("")
    for u in unit_reports:
        lines.append(f"### {u['legacy_unit_id']} -> {u['new_unit_id']}")
        lines.append("")
        lines.append(f"{u['notes']}")
        lines.append("")
        lines.append(
            f"present: accepted={_status_cell(u['present']['accepted'])}, "
            f"A={_status_cell(u['present']['a'])}, B={_status_cell(u['present']['b'])}"
        )
        lines.append("")
        lines.append(f"- characters_present (accepted): {', '.join(u['characters']['accepted']) or '(none/pending)'}")
        lines.append(f"- characters_present (A): {', '.join(u['characters']['a']) or '(none/pending)'}")
        b_display = ", ".join(
            f"{c['canonical_name']}[{c['resolution']}]" for c in u["characters"]["b"]
        )
        lines.append(f"- characters_present (B): {b_display or '(none/pending)'}")
        if u["b_only_discoveries"]:
            names = ", ".join(f"{c['canonical_name']}[{c['resolution']}]" for c in u["b_only_discoveries"])
            lines.append(f"- B-only discoveries: {names}")
        missing_a = u["missing_vs_accepted"]["a"]
        missing_b = u["missing_vs_accepted"]["b"]
        if missing_a:
            lines.append(f"- missing vs accepted (A): {', '.join(missing_a)}")
        if missing_b:
            lines.append(f"- missing vs accepted (B): {', '.join(missing_b)}")
        if u["unresolved_in_b"]:
            names = ", ".join(f"{c['canonical_name']} ({c['classification']})" for c in u["unresolved_in_b"])
            lines.append(f"- unresolved in B: {names}")
        for lens in SCORING_LENS_ORDER:
            rates = u["direction_by_lens"][lens]
            lines.append(
                f"- direction agreement [{lens}]: "
                f"A-vs-accepted={_fmt_rate(rates['a_vs_accepted'])}, "
                f"B-vs-accepted={_fmt_rate(rates['b_vs_accepted'])}, "
                f"A-vs-B={_fmt_rate(rates['a_vs_b'])}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main


def compare_ab_run(ab_run_dir: Path = DEFAULT_AB_RUN_DIR) -> tuple:
    manifest = _read_json(ab_run_dir / "manifest.json")
    registry = Registry.load()
    unit_reports = [compare_unit(entry, ab_run_dir, registry) for entry in manifest["units"]]
    aggregates = build_aggregates(unit_reports)
    return unit_reports, aggregates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_AB_RUN_DIR), help="ab-run directory.")
    parser.add_argument("--out", default=None, help="Report path (default: <dir>/ab-report.md).")
    args = parser.parse_args(argv)

    ab_run_dir = Path(args.dir)
    out_path = Path(args.out) if args.out else ab_run_dir / "ab-report.md"

    unit_reports, aggregates = compare_ab_run(ab_run_dir)
    report = render_report(unit_reports, aggregates)
    out_path.write_text(report, encoding="utf-8")

    print(f"Wrote {out_path}")
    print(
        f"present: accepted={aggregates['units_with_accepted']}, "
        f"A={aggregates['units_with_annotation_a']}, B={aggregates['units_with_annotation_b']}, "
        f"fully_complete={aggregates['units_fully_complete']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
