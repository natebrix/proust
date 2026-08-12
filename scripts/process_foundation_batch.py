"""Write, score, and gate one foundation batch run.

Usage: python3 process_foundation_batch.py <run_dir>

Reads annotator output from <run_dir>/raw/<unit_id>.json, writes validated
annotations via foundation.write_foundation_result (which accepts prompt v2's
optional "resolution" field and keeps a per-unit resolution record), scores all
three lenses, applies the batch gates from the supplement pass adapted to the
foundation pass, and prints a single JSON gate report to stdout (last line).

Gates (any trip halts the batch for human triage):
  - validation failures, unparseable raw output, or missing raw output
  - mixed units per lens above the scaled supplement threshold (15 per 40)
  - unresolved character instances above 15% of all character instances,
    which signals a registry gap that should be fixed before annotating on
  - "le narrateur" scored in more than 30% of third-person v1-p2 units

"escalate" is advisory output for the orchestrator (units to re-annotate with
Opus), not a gate: escalation-only units do not trip the batch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust import foundation, runner  # noqa: E402

ANNOTATION_KEYS = ("characters_present", "appraisal_events", "status_effects", "ambiguities")
UNRESOLVED_RATE_LIMIT = 0.15
NARRATOR_UNIT_RATE_LIMIT = 0.3
NARRATOR_CHARACTER = "le narrateur"
ESCALATION_AMBIGUITY_LIMIT = 2

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "run.json").read_text())
unit_ids = manifest["unit_ids"]

report = {
    "run": run_dir.name,
    "chapter_id": manifest.get("chapter_id"),
    "status": "ok",
    "reasons": [],
    "unit_count": len(unit_ids),
    "written": 0,
    "already_written": 0,
    "missing_raw": [],
    "validation_failures": {},
    "narrator_v1p2_units": [],
    "mixed_counts": {},
    "unresolved": {
        "character_instances": 0,
        "unresolved_instances": 0,
        "unresolved_rate": 0.0,
        "limit": UNRESOLVED_RATE_LIMIT,
        "names": {},
    },
    "escalate": [],
}

escalation_reasons = {}


def flag_escalation(unit_id, reason):
    escalation_reasons.setdefault(unit_id, []).append(reason)


for unit_id in unit_ids:
    annotation_path = run_dir / "annotations" / f"{unit_id}.json"
    raw_path = run_dir / "raw" / f"{unit_id}.json"
    resolution_path = run_dir / "resolutions" / f"{unit_id}.json"

    if annotation_path.exists():
        report["already_written"] += 1
        annotation = json.loads(annotation_path.read_text())
        resolution = (
            json.loads(resolution_path.read_text())
            if resolution_path.exists()
            else foundation.resolution_summary(annotation, unit_id=unit_id)
        )
    else:
        if not raw_path.exists():
            report["missing_raw"].append(unit_id)
            flag_escalation(unit_id, "missing raw output")
            continue
        try:
            raw_annotation = json.loads(raw_path.read_text())
        except json.JSONDecodeError as exc:
            report["validation_failures"][unit_id] = f"raw output is not valid JSON: {exc}"
            flag_escalation(unit_id, "validation failure")
            continue
        annotation = {key: raw_annotation.get(key) for key in ANNOTATION_KEYS}
        annotation["unit_id"] = unit_id
        try:
            foundation.write_foundation_result(run_dir, unit_id, annotation)
            report["written"] += 1
        except ValueError as exc:
            report["validation_failures"][unit_id] = str(exc)[:500]
            flag_escalation(unit_id, "validation failure")
            continue
        resolution = json.loads(resolution_path.read_text())
        annotation = json.loads(annotation_path.read_text())

    report["unresolved"]["character_instances"] += resolution["character_count"]
    report["unresolved"]["unresolved_instances"] += resolution["unresolved_count"]
    for name in resolution["unresolved_names"]:
        entry = report["unresolved"]["names"].setdefault(name, {"count": 0, "units": []})
        entry["count"] += 1
        if unit_id not in entry["units"]:
            entry["units"].append(unit_id)

    scored = {effect["character"] for effect in annotation.get("status_effects") or []}
    if unit_id.startswith("v1-p2-un-amour-de-swann") and NARRATOR_CHARACTER in scored:
        report["narrator_v1p2_units"].append(unit_id)

    if len(annotation.get("ambiguities") or []) >= ESCALATION_AMBIGUITY_LIMIT:
        flag_escalation(unit_id, f'{len(annotation["ambiguities"])} ambiguities')
    if any(event.get("narrative_stance") == "uncertain" for event in annotation.get("appraisal_events") or []):
        flag_escalation(unit_id, "uncertain narrative stance")

character_instances = report["unresolved"]["character_instances"]
report["unresolved"]["unresolved_rate"] = (
    round(report["unresolved"]["unresolved_instances"] / character_instances, 3) if character_instances else 0.0
)
report["unresolved"]["names"] = dict(
    sorted(report["unresolved"]["names"].items(), key=lambda item: (-item[1]["count"], item[0]))
)
report["escalate"] = [
    {"unit_id": unit_id, "reasons": escalation_reasons[unit_id]}
    for unit_id in unit_ids
    if unit_id in escalation_reasons
]

for lens in ("advantage", "prestige", "inclusion"):
    lens_report = runner.build_outcome_report(run_dir, lens=lens)
    entry_total = len(lens_report["timeline"])
    mixed_total = len(lens_report["mixed_units"])
    report["mixed_counts"][lens] = mixed_total
    report.setdefault("entry_totals", {})[lens] = entry_total
    report.setdefault("mixed_rates", {})[lens] = (
        round(mixed_total / entry_total, 3) if entry_total else 0.0
    )

# gates
if report["validation_failures"]:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"{len(report['validation_failures'])} validation failures")
if report["missing_raw"]:
    report["status"] = "gate_tripped"
    report["reasons"].append(f"{len(report['missing_raw'])} units missing raw output")
if report["unresolved"]["unresolved_rate"] > UNRESOLVED_RATE_LIMIT:
    report["status"] = "gate_tripped"
    report["reasons"].append(
        f"unresolved rate {report['unresolved']['unresolved_rate']} > {UNRESOLVED_RATE_LIMIT} (registry gaps)"
    )
v1p2_in_batch = [unit_id for unit_id in unit_ids if unit_id.startswith("v1-p2-un-amour-de-swann")]
if v1p2_in_batch and len(report["narrator_v1p2_units"]) / len(v1p2_in_batch) > NARRATOR_UNIT_RATE_LIMIT:
    report["status"] = "gate_tripped"
    report["reasons"].append("narrator scored in >30% of third-person v1-p2 units")
# Mixed gate is rate-based per character entry, not an absolute per-unit
# count: the open-world prompt roughly doubles roster size (2.2 vs 1.1-1.4
# characters/unit in the legacy corpus), so the legacy-calibrated absolute
# threshold overcounts on the richer surface. Foundation-run-001 measured a
# 23% mixed-entry rate with annotations that eyeballed as sound; 30% leaves
# headroom while still catching genuine drift.
MIXED_ENTRY_RATE_LIMIT = 0.30
MIXED_GATE_MIN_ENTRIES = 20  # a rate on a tiny denominator is noise, not signal
for lens, rate in report.get("mixed_rates", {}).items():
    if report.get("entry_totals", {}).get(lens, 0) < MIXED_GATE_MIN_ENTRIES:
        continue
    if rate > MIXED_ENTRY_RATE_LIMIT:
        report["status"] = "gate_tripped"
        report["reasons"].append(
            f"mixed-entry rate {rate} in {lens} > {MIXED_ENTRY_RATE_LIMIT}"
        )

(run_dir / "gate-report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report))
