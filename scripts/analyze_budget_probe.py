"""Analyze the budget-probe run (outputs/budget-probe-001/).

Compares probe annotations (dimension criteria + uncapped distinctness-based
budget) against the foundation baseline on the same units. Answers three
questions ahead of the enrichment design note:

1. Rates: do prestige (social_status) and inclusion (inclusion_exclusion)
   effects rise on social-dense units?
2. Demand effect: do the control units stay flat? (The decisive gate.)
3. Budget: how often does a character legitimately carry more than one
   effect per lens family in a single unit — the calibration question for
   the one-per-family rule.

Usage: python scripts/analyze_budget_probe.py [--probe outputs/budget-probe-001]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAMILY = {
    "social_status": "prestige",
    "inclusion_exclusion": "inclusion",
    "general_appraisal": "advantage",
    "emotional_position": "advantage",
    "rhetorical_position": "advantage",
}


def load_run(probe_dir):
    manifest = json.loads((probe_dir / "run.json").read_text())
    units = []
    for entry in manifest["units"]:
        uid = entry["unit_id"]
        fn = uid + ".json"
        probe_path = probe_dir / "annotations" / fn
        base_path = probe_dir / "baseline" / fn
        units.append(
            {
                "unit_id": uid,
                "group": entry["group"],
                "probe": json.loads(probe_path.read_text()) if probe_path.exists() else None,
                "baseline": json.loads(base_path.read_text()) if base_path.exists() else None,
            }
        )
    return units


def effect_rows(annotation):
    for effect in (annotation or {}).get("status_effects") or []:
        dim = effect.get("dimension")
        if dim in FAMILY:
            yield effect["character"], dim, FAMILY[dim], effect


def summarize(units, arm):
    per_group = defaultdict(lambda: Counter())
    for unit in units:
        annotation = unit[arm]
        if annotation is None:
            continue
        counts = Counter()
        for _char, dim, _fam, _e in effect_rows(annotation):
            counts[dim] += 1
        per_group[unit["group"]]["units"] += 1
        for dim, n in counts.items():
            per_group[unit["group"]][dim] += n
        per_group[unit["group"]]["total_effects"] += sum(counts.values())
    return per_group


def multi_family_cases(units):
    """Character-unit pairs carrying >1 effect in one lens family (probe arm)."""
    cases = []
    for unit in units:
        annotation = unit["probe"]
        if annotation is None:
            continue
        fam_effects = defaultdict(list)
        for char, dim, fam, effect in effect_rows(annotation):
            fam_effects[(char, fam)].append((dim, effect))
        for (char, fam), effects in sorted(fam_effects.items()):
            if len(effects) > 1:
                cases.append(
                    {
                        "unit_id": unit["unit_id"],
                        "group": unit["group"],
                        "character": char,
                        "family": fam,
                        "effects": [
                            {
                                "dimension": dim,
                                "delta": e.get("delta"),
                                "confidence": e.get("confidence"),
                                "based_on_events": e.get("based_on_events"),
                                "explanation": e.get("explanation"),
                            }
                            for dim, e in effects
                        ],
                        "distinct_event_lineage": len(
                            {tuple(sorted(e.get("based_on_events") or [])) for _d, e in effects}
                        )
                        == len(effects),
                    }
                )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="outputs/budget-probe-001")
    parser.add_argument("--json", action="store_true", help="emit full JSON report")
    args = parser.parse_args()
    probe_dir = REPO / args.probe

    units = load_run(probe_dir)
    missing = [u["unit_id"] for u in units if u["probe"] is None]
    if missing:
        print(f"NOTE: {len(missing)} unit(s) not yet annotated: {missing}")

    report = {"groups": {}, "multi_family_cases": multi_family_cases(units)}
    base = summarize(units, "baseline")
    probe = summarize(units, "probe")
    dims = ["general_appraisal", "rhetorical_position", "emotional_position", "social_status", "inclusion_exclusion"]

    for group in ("social", "control_silent", "control_intimate"):
        rows = {}
        for dim in dims + ["total_effects"]:
            rows[dim] = {"baseline": base[group].get(dim, 0), "probe": probe[group].get(dim, 0)}
        report["groups"][group] = {"units": probe[group].get("units", 0), "effects": rows}

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    for group, payload in report["groups"].items():
        print(f"\n== {group} ({payload['units']} units annotated) ==")
        print(f"   {'dimension':22s} {'baseline':>8s} {'probe':>8s}")
        for dim, row in payload["effects"].items():
            print(f"   {dim:22s} {row['baseline']:8d} {row['probe']:8d}")

    cases = report["multi_family_cases"]
    print(f"\n== >1 effect per lens family per character-unit (probe arm): {len(cases)} case(s) ==")
    for case in cases:
        lineage = "distinct events" if case["distinct_event_lineage"] else "SHARED events (suspect restatement)"
        print(f"\n  {case['unit_id']} [{case['group']}] — {case['character']} / {case['family']} ({lineage})")
        for e in case["effects"]:
            print(f"    {e['dimension']} delta={e['delta']} conf={e['confidence']} events={e['based_on_events']}")
            print(f"      {e['explanation']}")


if __name__ == "__main__":
    main()
