"""Build the staged scoring v2 surfaces and their validation report.

Usage:
    python3 scripts/build_scoring_v2.py [--outputs-dir outputs]
                                        [--output-dir outputs/scoring-v2]
                                        [--stage build|validate|both]

Everything this writes lands under --output-dir (outputs/scoring-v2 by
default): v2 is staged for the adoption gate in
proust/docs/scoring_v2_design.md, and no artifact outside that directory is
touched. `--stage validate` re-reads the staged artifacts, so the validation
battery can be re-run without repeating the fits.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust import scoring_v2_build, scoring_v2_validate  # noqa: E402
from proust.app_exports import discover_foundation_run_dirs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-dir", default=scoring_v2_build.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=("build", "validate", "both"), default="both")
    parser.add_argument("--bootstrap-samples", type=int, default=scoring_v2_validate.DEFAULT_BOOTSTRAP_SAMPLES)
    arguments = parser.parse_args()

    started = perf_counter()
    run_dirs = discover_foundation_run_dirs(arguments.outputs_dir)
    print(f"foundation runs: {len(run_dirs)}", flush=True)

    if arguments.stage in ("build", "both"):
        build = scoring_v2_build.build_scoring_v2(
            run_dirs, progress=lambda message: print(message, flush=True)
        )
        written = scoring_v2_build.write_scoring_v2_artifacts(build, output_dir=arguments.output_dir)
        print(json.dumps({"written": written, "manifest": build["manifest"]}, ensure_ascii=False), flush=True)

    if arguments.stage in ("validate", "both"):
        report = scoring_v2_validate.build_validation_report(
            run_dirs,
            output_dir=arguments.output_dir,
            outputs_dir=arguments.outputs_dir,
            bootstrap_samples=arguments.bootstrap_samples,
            progress=lambda message: print(message, flush=True),
        )
        written = scoring_v2_validate.write_validation_report(report, output_dir=arguments.output_dir)
        print(json.dumps({"written": written}, ensure_ascii=False), flush=True)

    print(f"total wall clock seconds: {round(perf_counter() - started, 1)}", flush=True)


if __name__ == "__main__":
    main()
