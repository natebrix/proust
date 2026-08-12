"""Apply Opus escalation results to foundation runs.

For every raw-opus/<unit_id>.json in the given runs: validate and write it
through write_foundation_result (registry reconciliation included). On
success the Opus annotation replaces the Sonnet one in annotations/ and the
replacement is recorded in escalations.json (per run) with both provenances;
on validation failure the Sonnet annotation stands and the failure is
recorded. Prints a summary; exits nonzero only on unexpected errors.

Usage: python scripts/apply_escalations.py outputs/foundation-run-001 [...]
       python scripts/apply_escalations.py --glob 'outputs/foundation-run-*'
"""
import glob as globlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proust.foundation import write_foundation_result  # noqa: E402


def apply_run(run_dir):
    run_path = Path(run_dir)
    opus_dir = run_path / "raw-opus"
    if not opus_dir.exists():
        return None
    record = {"applied": [], "kept_sonnet": []}
    for opus_file in sorted(opus_dir.glob("*.json")):
        unit_id = opus_file.stem
        try:
            annotation = json.loads(opus_file.read_text())
        except json.JSONDecodeError as exc:
            record["kept_sonnet"].append({"unit_id": unit_id, "reason": f"opus JSON invalid: {exc}"})
            continue
        try:
            write_foundation_result(run_path, unit_id, annotation)
            record["applied"].append(unit_id)
        except ValueError as exc:
            record["kept_sonnet"].append({"unit_id": unit_id, "reason": str(exc)[:300]})
    (run_path / "escalations.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv):
    if argv and argv[0] == "--glob":
        run_dirs = sorted(globlib.glob(argv[1]))
    else:
        run_dirs = argv
    applied = kept = 0
    for run_dir in run_dirs:
        record = apply_run(run_dir)
        if record is None:
            continue
        applied += len(record["applied"])
        kept += len(record["kept_sonnet"])
        if record["kept_sonnet"]:
            for row in record["kept_sonnet"]:
                print(f"KEPT SONNET {run_dir} {row['unit_id']}: {row['reason'][:160]}")
    print(f"applied={applied} kept_sonnet={kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
