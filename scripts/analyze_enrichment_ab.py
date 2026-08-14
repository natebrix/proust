"""Analyze the enrichment A/B run (outputs/enrichment-ab-001/).

Evaluates prompt v2.1 against the adoption gates in
proust/docs/enrichment_design.md:

1. Control flatness (hard): no prestige/inclusion effects on silent controls;
   no rise on intimate controls beyond variance-arm re-roll noise.
2. Distinctness discipline (hard): no same-advantage-family pair for one
   character citing the same single event; same-dimension seconds cite
   distinct events.
3. Coverage direction (target): more characters carry prestige/inclusion
   effects on social units than baseline; new effects listed for spot review.
4. Sign consistency (advisory): agreement on shared character-unit-family
   observations, benchmarked against the variance arm's self-agreement.

Usage: python scripts/analyze_enrichment_ab.py [--run outputs/enrichment-ab-001]
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
ADVANTAGE_DIMS = {d for d, f in FAMILY.items() if f == "advantage"}


def load(run_dir):
    manifest = json.loads((run_dir / "run.json").read_text())
    units = []
    for entry in manifest["units"]:
        uid = entry["unit_id"]
        fn = uid + ".json"

        def read(sub):
            p = run_dir / sub / fn
            return json.loads(p.read_text()) if p.exists() else None

        units.append(
            {
                "unit_id": uid,
                "group": entry["group"],
                "variance_arm": entry.get("variance_arm", False),
                "baseline": read("baseline"),
                "b": read("annotations-b"),
                "var": read("annotations-var"),
            }
        )
    return units


def effects(annotation):
    for e in (annotation or {}).get("status_effects") or []:
        if e.get("dimension") in FAMILY:
            yield e


def fam_signs(annotation):
    """(character, family) -> summed delta sign for one annotation."""
    sums = defaultdict(float)
    for e in effects(annotation):
        sums[(e["character"], FAMILY[e["dimension"]])] += e.get("delta") or 0
    return {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in sums.items()}


def gate1_control_flatness(units):
    violations = []
    intimate = {"baseline": 0, "b": 0}
    for u in units:
        if u["group"] == "control_silent":
            for e in effects(u["b"]):
                if FAMILY[e["dimension"]] != "advantage":
                    violations.append((u["unit_id"], e["character"], e["dimension"], e.get("explanation")))
        if u["group"] == "control_intimate":
            for arm in ("baseline", "b"):
                intimate[arm] += sum(1 for e in effects(u[arm]) if FAMILY[e["dimension"]] != "advantage")
    return {"silent_violations": violations, "intimate_prestige_inclusion": intimate}


def gate2_distinctness(units):
    violations = []
    for u in units:
        per_char = defaultdict(list)
        for e in effects(u["b"]):
            per_char[e["character"]].append(e)
        for char, char_effects in per_char.items():
            by_dim = Counter(e["dimension"] for e in char_effects)
            for dim, n in by_dim.items():
                if n > 1:
                    lineages = [tuple(sorted(e.get("based_on_events") or [])) for e in char_effects if e["dimension"] == dim]
                    if len(set(lineages)) < n:
                        violations.append((u["unit_id"], char, f"same-dimension {dim} with shared events"))
            adv = [e for e in char_effects if e["dimension"] in ADVANTAGE_DIMS]
            event_use = Counter()
            for e in adv:
                for ev in e.get("based_on_events") or []:
                    event_use[ev] += 1
            for ev, n in event_use.items():
                if n > 1:
                    violations.append((u["unit_id"], char, f"event {ev} grounds {n} advantage-family effects"))
    return violations


def gate3_coverage(units):
    out = {}
    for fam in ("prestige", "inclusion"):
        base_chars, b_chars, new = set(), set(), []
        for u in units:
            if u["group"] != "social":
                continue
            for arm, bag in (("baseline", base_chars), ("b", b_chars)):
                for e in effects(u[arm]):
                    if FAMILY[e["dimension"]] == fam:
                        bag.add((u["unit_id"], e["character"]))
                        if arm == "b" and (u["unit_id"], e["character"]) not in {
                            (u["unit_id"], x["character"]) for x in effects(u["baseline"]) if FAMILY[x["dimension"]] == fam
                        }:
                            new.append({"unit_id": u["unit_id"], "character": e["character"],
                                        "dimension": e["dimension"], "delta": e.get("delta"),
                                        "explanation": e.get("explanation")})
        out[fam] = {"baseline_character_units": len(base_chars), "b_character_units": len(b_chars), "new_in_b": new}
    return out


def gate4_signs(units):
    def agreement(arm_x, arm_y, subset):
        shared, agree = 0, 0
        for u in subset:
            sx, sy = fam_signs(u[arm_x]), fam_signs(u[arm_y])
            for key in set(sx) & set(sy):
                shared += 1
                if sx[key] == sy[key]:
                    agree += 1
        return shared, agree

    var_units = [u for u in units if u["variance_arm"] and u["var"] is not None]
    shared_v, agree_v = agreement("baseline", "var", var_units)
    social = [u for u in units if u["group"] == "social"]
    shared_b, agree_b = agreement("baseline", "b", social)
    flips = []
    for u in social:
        sx, sy = fam_signs(u["baseline"]), fam_signs(u["b"])
        for key in set(sx) & set(sy):
            if sx[key] != sy[key] and 0 not in (sx[key], sy[key]):
                flips.append((u["unit_id"], key[0], key[1], sx[key], sy[key]))
    return {
        "variance_arm": {"shared": shared_v, "agree": agree_v},
        "b_vs_baseline": {"shared": shared_b, "agree": agree_b},
        "hard_flips": flips,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/enrichment-ab-001")
    args = parser.parse_args()
    units = load(REPO / args.run)

    missing_b = [u["unit_id"] for u in units if u["b"] is None]
    missing_v = [u["unit_id"] for u in units if u["variance_arm"] and u["var"] is None]
    if missing_b or missing_v:
        print(f"NOTE: missing arm B: {missing_b}; missing variance: {missing_v}")

    g1 = gate1_control_flatness(units)
    print("== Gate 1: control flatness (hard) ==")
    print(f"  silent-control prestige/inclusion effects: {len(g1['silent_violations'])}")
    for v in g1["silent_violations"]:
        print(f"    VIOLATION {v}")
    ip = g1["intimate_prestige_inclusion"]
    print(f"  intimate-control prestige/inclusion effects: baseline {ip['baseline']}, arm B {ip['b']}")

    g2 = gate2_distinctness(units)
    print("\n== Gate 2: distinctness discipline (hard) ==")
    print(f"  violations: {len(g2)}")
    for v in g2:
        print(f"    VIOLATION {v}")

    g3 = gate3_coverage(units)
    print("\n== Gate 3: coverage direction (target) ==")
    for fam, row in g3.items():
        print(f"  {fam}: character-units weighed — baseline {row['baseline_character_units']}, arm B {row['b_character_units']}")
        for n in row["new_in_b"]:
            print(f"    NEW {n['unit_id']} — {n['character']} {n['dimension']} {n['delta']:+d}")
            print(f"        {n['explanation'][:140]}")

    g4 = gate4_signs(units)
    print("\n== Gate 4: sign consistency (advisory) ==")
    v, b = g4["variance_arm"], g4["b_vs_baseline"]
    v_rate = f"{v['agree']}/{v['shared']}" if v["shared"] else "n/a"
    b_rate = f"{b['agree']}/{b['shared']}" if b["shared"] else "n/a"
    print(f"  variance arm self-agreement (re-roll noise floor): {v_rate}")
    print(f"  arm B vs baseline agreement: {b_rate}")
    print(f"  hard sign flips (nonzero to opposite nonzero): {len(g4['hard_flips'])}")
    for f in g4["hard_flips"]:
        print(f"    FLIP {f}")


if __name__ == "__main__":
    main()
