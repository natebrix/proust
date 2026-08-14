"""Gate the whole enrichment pass: ingest, score, and gate all 34 runs.

For each outputs/enrichment-run-* directory this runs the foundation batch
processor (ingest raw -> validated annotations + resolutions, score, gates)
followed by the distinctness-discipline gate, and writes a pass-level
summary to outputs/enrichment-gate-summary.json.

Usage: python3 scripts/gate_enrichment_pass.py [--runs enrichment-run-007 ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from check_enrichment_distinctness import check_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None)
    arguments = parser.parse_args()

    if arguments.runs:
        run_dirs = [REPO / "outputs" / name for name in arguments.runs]
    else:
        run_dirs = sorted((REPO / "outputs").glob("enrichment-run-*"))

    summary = {"runs": {}, "tripped": [], "escalate": [], "totals": {
        "units": 0, "written": 0, "already_written": 0,
        "validation_failures": 0, "missing_raw": 0,
        "sign_flips_unadjudicated": 0, "distinctness_violations": 0,
    }}
    for run_dir in run_dirs:
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "process_foundation_batch.py"), str(run_dir)],
            capture_output=True, text=True, cwd=REPO,
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            summary["runs"][run_dir.name] = {"status": "processor_error", "stderr": proc.stderr[-500:]}
            summary["tripped"].append(run_dir.name)
            continue
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        distinctness = check_run(run_dir)
        entry = {
            "status": report["status"],
            "reasons": report["reasons"],
            "unit_count": report["unit_count"],
            "written": report["written"],
            "already_written": report["already_written"],
            "validation_failures": len(report["validation_failures"]),
            "missing_raw": len(report["missing_raw"]),
            "unresolved_rate": report["unresolved"]["unresolved_rate"],
            "mixed_rates": report.get("mixed_rates", {}),
            "unadjudicated_sign_flips": report.get("unadjudicated_sign_flips", []),
            "escalate": report["escalate"],
            "distinctness": {
                "status": distinctness["status"],
                "violations": distinctness["violations"],
            },
        }
        if distinctness["status"] != "ok":
            entry["status"] = "gate_tripped"
            entry["reasons"] = entry["reasons"] + [
                f"{distinctness['violation_count']} distinctness violations"
            ]
        summary["runs"][run_dir.name] = entry
        if entry["status"] != "ok":
            summary["tripped"].append(run_dir.name)
        for row in report["escalate"]:
            summary["escalate"].append({"run": run_dir.name, **row})
        t = summary["totals"]
        t["units"] += report["unit_count"]
        t["written"] += report["written"]
        t["already_written"] += report["already_written"]
        t["validation_failures"] += len(report["validation_failures"])
        t["missing_raw"] += len(report["missing_raw"])
        t["sign_flips_unadjudicated"] += len(entry["unadjudicated_sign_flips"])
        t["distinctness_violations"] += distinctness["violation_count"]
        print(f"{run_dir.name}: {entry['status']}"
              + (f" ({'; '.join(entry['reasons'])})" if entry["reasons"] else ""),
              flush=True)

    out = REPO / "outputs" / "enrichment-gate-summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"tripped": summary["tripped"], "totals": summary["totals"],
                      "escalation_count": len(summary["escalate"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
