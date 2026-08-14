"""Distinctness-discipline gate for enrichment runs (hard gate 2 of
proust/docs/enrichment_design.md).

Checks every annotation in <run_dir>/annotations/ for the two prompt-v2.1
budget rules:

  1. Same-dimension multiples for one character must cite distinct
     based_on_events lineages.
  2. Within the advantage family (general_appraisal, emotional_position,
     rhetorical_position), a single event grounds at most one effect per
     character.

Usage: python3 scripts/check_enrichment_distinctness.py <run_dir> [...]
Prints one JSON report line per run dir; any violation => status gate_tripped.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ADVANTAGE_DIMS = {"general_appraisal", "emotional_position", "rhetorical_position"}


def check_run(run_dir):
    violations = []
    for path in sorted((run_dir / "annotations").glob("*.json")):
        annotation = json.loads(path.read_text())
        per_char = defaultdict(list)
        for effect in annotation.get("status_effects") or []:
            per_char[effect.get("character")].append(effect)
        for char, effects in per_char.items():
            by_dim = Counter(e.get("dimension") for e in effects)
            for dim, n in by_dim.items():
                if n > 1:
                    lineages = [
                        tuple(sorted(e.get("based_on_events") or []))
                        for e in effects
                        if e.get("dimension") == dim
                    ]
                    if len(set(lineages)) < n:
                        violations.append(
                            {"unit_id": path.stem, "character": char,
                             "rule": f"same-dimension {dim} with shared event lineage"}
                        )
            event_use = Counter()
            for e in effects:
                if e.get("dimension") in ADVANTAGE_DIMS:
                    for ev in e.get("based_on_events") or []:
                        event_use[ev] += 1
            for ev, n in event_use.items():
                if n > 1:
                    violations.append(
                        {"unit_id": path.stem, "character": char,
                         "rule": f"event {ev} grounds {n} advantage-family effects"}
                    )
    return {
        "run": run_dir.name,
        "status": "gate_tripped" if violations else "ok",
        "violation_count": len(violations),
        "violations": violations,
    }


def main():
    for arg in sys.argv[1:]:
        print(json.dumps(check_run(Path(arg)), ensure_ascii=False))


if __name__ == "__main__":
    main()
