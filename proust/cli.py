import argparse
import json
import sys

from . import runner as core


def _collect_runs(args):
    runs = list(args.runs or [])
    if getattr(args, "discover_runs", None):
        runs.extend(core.discover_annotation_run_dirs(args.discover_runs))
    return runs


def _character_name_map(args):
    return core.REVIEWED_CHARACTER_NORMALIZATION_MAP if args.reviewed_character_normalization else None


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare an ISLT annotation run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare an annotation run directory.")
    prepare_parser.add_argument("--output", required=True, help="Run directory to create.")
    prepare_parser.add_argument("--run-id", help="Optional run identifier. Defaults to the output directory name.")
    prepare_parser.add_argument("--notes", default="", help="Optional note stored in run.json.")
    prepare_parser.add_argument("--prompt", dest="prompt_path", help="Optional prompt template path.")

    status_parser = subparsers.add_parser("status", help="Summarize and validate an annotation run.")
    status_parser.add_argument("--run", required=True, help="Run directory to inspect.")
    status_parser.add_argument(
        "--write-benchmark",
        action="store_true",
        help="Persist benchmark validation metadata back into run.json.",
    )
    status_parser.add_argument(
        "--label",
        default="reviewed benchmark",
        help="Benchmark label written into run.json when --write-benchmark is used.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare a run against a reviewed benchmark.")
    compare_parser.add_argument("--run", required=True, help="Run directory to inspect.")
    compare_parser.add_argument("--benchmark", required=True, help="Benchmark run directory.")

    summary_parser = subparsers.add_parser("summary", help="Aggregate validated annotations in a run.")
    summary_parser.add_argument("--run", required=True, help="Run directory to summarize.")

    score_parser = subparsers.add_parser(
        "score",
        help='Compute a lightweight advantage "winning"/"losing" transformation for a run.',
    )
    score_parser.add_argument("--run", required=True, help="Run directory to score.")
    score_parser.add_argument("--lens", default="advantage", choices=sorted(core.SCORING_LENS_CONFIGS), help="Scoring lens.")

    report_parser = subparsers.add_parser(
        "report",
        help="Build a compact downstream outcome report from advantage outcome scores.",
    )
    report_parser.add_argument("--run", required=True, help="Run directory to report on.")
    report_parser.add_argument("--lens", default="advantage", choices=sorted(core.SCORING_LENS_CONFIGS), help="Scoring lens.")
    report_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )

    corpus_review_parser = subparsers.add_parser(
        "corpus-review",
        help="Aggregate multiple runs into a corpus-level sanity review.",
    )
    corpus_review_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    corpus_review_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    corpus_review_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    corpus_review_parser.add_argument("--output", help="Optional JSON output path.")
    corpus_review_parser.add_argument("--markdown-output", help="Optional Markdown output path.")
    corpus_review_parser.add_argument(
        "--normalization-diff-output",
        help="Optional Markdown output path for a diff between unnormalized and normalized corpus reviews.",
    )

    character_analysis_parser = subparsers.add_parser(
        "character-analysis",
        help="Build a per-character cross-lens downstream analysis from the corpus review surface.",
    )
    character_analysis_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_analysis_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_analysis_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_analysis_parser.add_argument("--output", help="Optional JSON output path.")
    character_analysis_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_chapter_parser = subparsers.add_parser(
        "character-chapter-analysis",
        help="Build a chapter-by-chapter cross-lens analysis for the highest-information characters.",
    )
    character_chapter_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_chapter_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_chapter_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_chapter_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional character to include. Repeat for multiple characters. Defaults to the union of top rank-spread and top volatile figures.",
    )
    character_chapter_parser.add_argument("--output", help="Optional JSON output path.")
    character_chapter_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_counts_parser = subparsers.add_parser(
        "character-annotation-counts",
        help="Build a normalized character list sorted by annotation unit count.",
    )
    character_counts_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_counts_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_counts_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_counts_parser.add_argument("--output", help="Optional JSON output path.")
    character_counts_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_elo_parser = subparsers.add_parser(
        "character-elo",
        help="Build an advantage-lens ELO ranking from pairwise within-unit character comparisons.",
    )
    character_elo_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_elo_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_elo_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_elo_parser.add_argument("--output", help="Optional JSON output path.")
    character_elo_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_elo_timeline_parser = subparsers.add_parser(
        "character-elo-timeline",
        help="Build sparse per-unit ELO timeline points for the tracked character set.",
    )
    character_elo_timeline_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_elo_timeline_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_elo_timeline_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_elo_timeline_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional tracked character to include. Repeat for multiple characters.",
    )
    character_elo_timeline_parser.add_argument("--output", help="Optional JSON output path.")
    character_elo_timeline_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_cards_parser = subparsers.add_parser(
        "character-profile-cards",
        help="Build app-facing cross-lens character profile cards.",
    )
    character_cards_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_cards_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_cards_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_cards_parser.add_argument("--output", help="Optional JSON output path.")
    character_cards_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_pages_parser = subparsers.add_parser(
        "character-pages",
        help="Build pilot app-facing character pages from existing analysis artifacts.",
    )
    character_pages_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_pages_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_pages_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    character_pages_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional pilot character to include. Repeat for multiple characters.",
    )
    character_pages_parser.add_argument("--output", help="Optional JSON output path.")
    character_pages_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    chapter_summaries_parser = subparsers.add_parser(
        "chapter-summaries",
        help="Build chapter-keyed app-facing chapter summary export data.",
    )
    chapter_summaries_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    chapter_summaries_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    chapter_summaries_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    chapter_summaries_parser.add_argument("--output", help="Optional JSON output path.")
    chapter_summaries_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    chapter_overlay_parser = subparsers.add_parser(
        "chapter-overlays",
        help="Export chapter-keyed app overlay JSON from the accepted normalized corpus surface.",
    )
    chapter_overlay_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    chapter_overlay_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    chapter_overlay_parser.add_argument(
        "--reviewed-character-normalization",
        action="store_true",
        help="Apply the reviewed explicit aggregate-layer character normalization map.",
    )
    chapter_overlay_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest.json and chapter JSON files should be written.",
    )

    alias_audit_parser = subparsers.add_parser(
        "character-alias-audit",
        help="Audit possible duplicate character names using aliases and annotation outputs.",
    )
    alias_audit_parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Outputs directory containing run-* annotation outputs.",
    )
    alias_audit_parser.add_argument(
        "--aliases-csv",
        default=str(core.ALIASES_CSV),
        help="Two-column alias CSV to use as audit evidence.",
    )
    alias_audit_parser.add_argument("--output", help="Optional JSON output path.")
    alias_audit_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    automate_parser = subparsers.add_parser("automate", help="Run prompts in a prepared source run through OpenAI.")
    automate_parser.add_argument("--source-run", required=True, help="Reviewed or candidate source run directory.")
    automate_parser.add_argument("--output", required=True, help="Output run directory for automated results.")
    automate_parser.add_argument("--model", default="gpt-5", help="OpenAI model to use.")
    automate_parser.add_argument("--limit", type=int, help="Optional maximum number of units to request.")
    automate_parser.add_argument("--overwrite", action="store_true", help="Re-request units even if annotations already exist in the output run.")

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run a prepared source batch end-to-end: automate, wait, reduce, report, and review-gate.",
    )
    batch_parser.add_argument("--source-run", required=True, help="Reviewed or candidate source run directory.")
    batch_parser.add_argument("--output", required=True, help="Output run directory for automated results.")
    batch_parser.add_argument("--model", default="gpt-5", help="OpenAI model to use.")
    batch_parser.add_argument("--limit", type=int, help="Optional maximum number of units to request.")
    batch_parser.add_argument("--overwrite", action="store_true", help="Re-request units even if annotations already exist in the output run.")
    batch_parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to wait between manifest checks.")
    batch_parser.add_argument("--timeout", type=float, help="Optional maximum number of seconds to wait before failing.")
    batch_parser.add_argument("--quiet", action="store_true", help="Suppress incremental progress output while waiting.")
    batch_parser.add_argument("--max-mixed-units-per-lens", type=int, default=3, help="Review-gate threshold for mixed units in a single lens.")

    wait_parser = subparsers.add_parser(
        "wait",
        help="Wait for an automated run to finish and optionally post-process it.",
    )
    wait_parser.add_argument("--run", required=True, help="Automated run directory to monitor.")
    wait_parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to wait between manifest checks.")
    wait_parser.add_argument("--timeout", type=float, help="Optional maximum number of seconds to wait before failing.")
    wait_parser.add_argument("--quiet", action="store_true", help="Suppress incremental progress output while waiting.")
    wait_parser.add_argument("--reduce", action="store_true", help="Run reducer-based reprocessing after automation completes.")
    wait_parser.add_argument("--report", action="store_true", help="Build advantage, prestige, and inclusion reports after completion.")

    reprocess_parser = subparsers.add_parser("reprocess", help="Re-parse saved raw outputs into annotations.")
    reprocess_parser.add_argument("--run", required=True, help="Run directory to reprocess.")
    reprocess_parser.add_argument("--overwrite", action="store_true", help="Rewrite annotations even if they already exist.")
    reprocess_parser.add_argument("--reduce", action="store_true", help="Apply the first-pass reducer before validation and writing annotations.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        core.prepare_annotation_run(
            args.output,
            run_id=args.run_id,
            prompt_path=args.prompt_path,
            notes=args.notes,
        )
        return 0

    if args.command == "compare":
        try:
            comparison = core.compare_run_to_benchmark(args.run, args.benchmark)
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "summary":
        try:
            summary = core.summarize_run_annotations(args.run)
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score":
        try:
            summary = core._score_run_outcomes(args.run, lens=args.lens)
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report":
        try:
            report = core.build_outcome_report(args.run, lens=args.lens, character_name_map=_character_name_map(args))
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "corpus-review":
        if args.normalization_diff_output and not args.reviewed_character_normalization:
            parser.error("--normalization-diff-output requires --reviewed-character-normalization")
        try:
            runs = _collect_runs(args)
            baseline_review = None
            if args.normalization_diff_output:
                baseline_review = core.build_corpus_sanity_review(runs)
            review = core.build_corpus_sanity_review(runs, character_name_map=_character_name_map(args))
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_corpus_review_artifacts(review, json_output=args.output, markdown_output=args.markdown_output)
        if args.normalization_diff_output:
            if baseline_review is None:
                baseline_review = core.build_corpus_sanity_review(runs)
            diff = core.build_corpus_review_normalization_diff(baseline_review, review)
            core.write_corpus_review_normalization_diff_artifacts(diff, markdown_output=args.normalization_diff_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "run_count": review["run_count"],
                "declared_unit_count": review["declared_unit_count"],
                "valid_annotation_count": review["valid_annotation_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
                "normalization_diff_output": args.normalization_diff_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-alias-audit":
        try:
            audit = core.build_character_alias_audit(outputs_dir=args.outputs_dir, aliases_csv=args.aliases_csv)
        except ValueError as exc:
            parser.error(str(exc))
        core.write_character_alias_audit_artifacts(audit, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "annotation_name_count": audit["annotation_name_count"],
                "candidate_merge_group_count": audit["candidate_merge_group_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-analysis":
        try:
            review = core.build_corpus_sanity_review(_collect_runs(args), character_name_map=_character_name_map(args))
            analysis = core.build_character_cross_lens_analysis(review)
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_cross_lens_analysis_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-chapter-analysis":
        try:
            analysis = core.build_character_chapter_analysis(
                _collect_runs(args),
                character_name_map=_character_name_map(args),
                target_characters=args.characters,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_chapter_analysis_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "selected_character_count": analysis["selected_character_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-annotation-counts":
        try:
            review = core.build_corpus_sanity_review(_collect_runs(args), character_name_map=_character_name_map(args))
            analysis = core.build_character_annotation_counts(review)
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_annotation_counts_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-elo":
        try:
            analysis = core.build_character_elo(
                _collect_runs(args),
                character_name_map=_character_name_map(args),
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_elo_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "match_count": analysis["match_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-elo-timeline":
        try:
            analysis = core.build_character_elo_timeline(
                _collect_runs(args),
                character_name_map=_character_name_map(args),
                target_characters=args.characters,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_elo_timeline_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "tracked_character_count": analysis["tracked_character_count"],
                "point_count": analysis["point_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-profile-cards":
        try:
            analysis = core.build_character_profile_cards(_collect_runs(args), character_name_map=_character_name_map(args))
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_profile_cards_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chapter-overlays":
        try:
            dataset = core.build_chapter_overlay_data(_collect_runs(args), character_name_map=_character_name_map(args))
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_chapter_overlay_artifacts(dataset, args.output_dir)
        print(json.dumps({
            "chapter_count": dataset["chapter_count"],
            "output_dir": args.output_dir,
            "character_normalization_applied": dataset["character_normalization"]["applied"],
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-pages":
        try:
            analysis = core.build_character_pages(
                _collect_runs(args),
                character_name_map=_character_name_map(args),
                target_characters=args.characters,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_character_pages_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chapter-summaries":
        try:
            analysis = core.build_chapter_summary_export(_collect_runs(args), character_name_map=_character_name_map(args))
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_chapter_summary_export_artifacts(analysis, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "chapter_count": analysis["chapter_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "automate":
        try:
            automation = core.run_openai_annotation(
                args.source_run,
                args.output,
                model=args.model,
                overwrite=args.overwrite,
                limit=args.limit,
            )
        except (core.RunManifestNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(automation, ensure_ascii=False, indent=2))
        return 0

    if args.command == "batch":
        try:
            progress_stream = None if args.quiet else sys.stderr
            result = core.run_automated_batch(
                args.source_run,
                args.output,
                model=args.model,
                overwrite=args.overwrite,
                limit=args.limit,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                progress_stream=progress_stream,
                max_mixed_units_per_lens=args.max_mixed_units_per_lens,
            )
        except (core.RunManifestNotFoundError, RuntimeError, TimeoutError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["review_gate"]["ok"] else 2

    if args.command == "wait":
        try:
            progress_stream = None if args.quiet else sys.stderr
            waited = core.wait_for_automation_completion(
                args.run,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                progress_stream=progress_stream,
            )
            result = {"wait": waited}
            if args.reduce:
                reprocess_results = core.reprocess_raw_annotations(args.run, overwrite=True, reduce=True)
                result["reprocess"] = {"run": args.run, "results": reprocess_results}
            if args.report:
                result["reports"] = {
                    lens: core.build_outcome_report(args.run, lens=lens)
                    for lens in sorted(core.SCORING_LENS_CONFIGS)
                }
        except (core.RunManifestNotFoundError, RuntimeError, TimeoutError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "reprocess":
        try:
            results = core.reprocess_raw_annotations(args.run, overwrite=args.overwrite, reduce=args.reduce)
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps({"run": args.run, "results": results}, ensure_ascii=False, indent=2))
        return 0

    try:
        status = core.get_run_status(args.run)
    except core.RunManifestNotFoundError as exc:
        parser.error(str(exc))
    print(json.dumps(status["summary"], ensure_ascii=False, indent=2))
    if args.write_benchmark:
        core.mark_run_as_benchmark(args.run, label=args.label)
    return 0
