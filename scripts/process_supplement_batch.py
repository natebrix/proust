"""Write, score, and gate one supplement batch run.

Usage: python3 process_supplement_batch.py <run_dir>

Reads annotator output from <run_dir>/raw/<unit_id>.json, writes validated
annotations via write_supplement_result (which normalizes ambiguities), scores
all three lenses, applies the review gates from coverage_supplement_plan.md,
and prints a single JSON gate report to stdout (last line).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/nathanbrixius/dev/proust")
from proust import runner, supplement  # noqa: E402

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "run.json").read_text())
unit_ids = manifest["unit_ids"]

report = {
    "run": run_dir.name,
    "status": "ok",
    "reasons": [],
    "unit_count": len(unit_ids),
    "written": 0,
    "already_written": 0,
    "missing_raw": [],
    "validation_failures": {},
    "direction_reversals": [],
    "acceptance": {},
    "narrator_v1p2_units": [],
    "mixed_counts": {},
}

candidates_offered = 0
characters_scored = 0
nonempty_units = 0

for unit_id in unit_ids:
    ann_path = run_dir / "annotations" / f"{unit_id}.json"
    raw_path = run_dir / "raw" / f"{unit_id}.json"
    payload = json.loads((run_dir / "units" / f"{unit_id}.json").read_text())
    candidates_offered += len(payload.get("candidate_characters") or [])

    if ann_path.exists():
        report["already_written"] += 1
        ann = json.loads(ann_path.read_text())
    else:
        if not raw_path.exists():
            report["missing_raw"].append(unit_id)
            continue
        try:
            ann = json.loads(raw_path.read_text())
        except json.JSONDecodeError as exc:
            report["validation_failures"][unit_id] = f"raw output is not valid JSON: {exc}"
            continue
        keep = {"characters_present", "appraisal_events", "status_effects", "ambiguities"}
        ann = {k: ann.get(k) for k in keep}
        ann["unit_id"] = unit_id
        try:
            supplement.write_supplement_result(run_dir, unit_id, ann)
            report["written"] += 1
            ann = json.loads(ann_path.read_text())
        except ValueError as exc:
            report["validation_failures"][unit_id] = str(exc)[:500]
            continue

    scored = {s["character"] for s in ann.get("status_effects") or []}
    characters_scored += len(scored)
    if scored:
        nonempty_units += 1

    accepted = payload.get("accepted_annotation") or {}
    acc_dir = {}
    for ev in accepted.get("appraisal_events") or []:
        acc_dir[(ev.get("source"), ev.get("target"))] = ev.get("polarity")
    for ev in ann.get("appraisal_events") or []:
        key = (ev.get("source"), ev.get("target"))
        if key in acc_dir and {acc_dir[key], ev.get("polarity")} == {"positive", "negative"}:
            report["direction_reversals"].append({"unit": unit_id, "pair": list(key)})

    if unit_id.startswith("v1-p2-un-amour-de-swann") and "le narrateur" in scored:
        report["narrator_v1p2_units"].append(unit_id)

report["acceptance"] = {
    "candidates_offered": candidates_offered,
    "characters_scored": characters_scored,
    "candidate_acceptance_rate": round(characters_scored / candidates_offered, 3) if candidates_offered else 0.0,
    "nonempty_unit_rate": round(nonempty_units / len(unit_ids), 3) if unit_ids else 0.0,
}

for lens in ("advantage", "prestige", "inclusion"):
    rep = runner.build_outcome_report(run_dir, lens=lens)
    report["mixed_counts"][lens] = len(rep["mixed_units"])

# gates
if report["validation_failures"]:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"{len(report['validation_failures'])} validation failures")
if report["direction_reversals"]:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"{len(report['direction_reversals'])} direction reversals")
if report["missing_raw"]:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"{len(report['missing_raw'])} units missing raw output")
if report["acceptance"]["candidate_acceptance_rate"] > 0.5:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"candidate acceptance {report['acceptance']['candidate_acceptance_rate']} > 0.5 (materiality drift)")
v1p2_in_batch = [u for u in unit_ids if u.startswith("v1-p2-un-amour-de-swann")]
if v1p2_in_batch and len(report["narrator_v1p2_units"]) / len(v1p2_in_batch) > 0.3:
    report["status"] = "gate_tripped"
    report["reasons"].append("narrator scored in >30% of third-person v1-p2 units")
mixed_limit = max(3, round(15 * len(unit_ids) / 40))
for lens, count in report["mixed_counts"].items():
    if count > mixed_limit:
        report["status"] = "gate_tripped"
        report["reasons"].append(f"{count} mixed units in {lens} > {mixed_limit}")

(run_dir / "gate-report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report))
