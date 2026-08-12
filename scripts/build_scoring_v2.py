"""Build the staged scoring v2 surfaces, validate them, and promote them.

Usage:
    python3 scripts/build_scoring_v2.py [--outputs-dir outputs]
                                        [--output-dir outputs/scoring-v2]
                                        [--stage build|validate|promote|both|all]

`build` and `validate` write only under --output-dir (outputs/scoring-v2 by
default): that directory is the fit store, and the validation battery
re-reads it rather than repeating the fits. `promote` reads those staged
fits and writes the current surfaces under --outputs-dir -- standings,
journey timelines, character pages -- which is the step that makes v2 the
project's analysis surface (see the adoption record in
proust/docs/scoring_v2_design.md). `both` is build plus validate; `all`
adds promote.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust import scoring_v2_build, scoring_v2_promote, scoring_v2_validate  # noqa: E402
from proust.app_exports import discover_foundation_run_dirs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-dir", default=scoring_v2_build.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage", choices=("build", "validate", "promote", "both", "all"), default="both"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=scoring_v2_validate.DEFAULT_BOOTSTRAP_SAMPLES)
    arguments = parser.parse_args()

    started = perf_counter()
    run_dirs = discover_foundation_run_dirs(arguments.outputs_dir)
    print(f"foundation runs: {len(run_dirs)}", flush=True)

    if arguments.stage in ("build", "both", "all"):
        build = scoring_v2_build.build_scoring_v2(
            run_dirs, progress=lambda message: print(message, flush=True)
        )
        written = scoring_v2_build.write_scoring_v2_artifacts(build, output_dir=arguments.output_dir)
        print(json.dumps({"written": written, "manifest": build["manifest"]}, ensure_ascii=False), flush=True)

    if arguments.stage in ("validate", "both", "all"):
        report = scoring_v2_validate.build_validation_report(
            run_dirs,
            output_dir=arguments.output_dir,
            outputs_dir=arguments.outputs_dir,
            bootstrap_samples=arguments.bootstrap_samples,
            progress=lambda message: print(message, flush=True),
        )
        written = scoring_v2_validate.write_validation_report(report, output_dir=arguments.output_dir)
        print(json.dumps({"written": written}, ensure_ascii=False), flush=True)

    if arguments.stage in ("promote", "all"):
        promotion = scoring_v2_promote.promote_scoring_v2(
            run_dirs,
            staged_dir=arguments.output_dir,
            outputs_dir=arguments.outputs_dir,
            progress=lambda message: print(message, flush=True),
        )
        print(json.dumps({"written": promotion["written"]}, ensure_ascii=False), flush=True)

    print(f"total wall clock seconds: {round(perf_counter() - started, 1)}", flush=True)


if __name__ == "__main__":
    main()
