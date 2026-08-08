import argparse
import json
import sys
from pathlib import Path

from . import coverage
from . import runner as core
from . import supplement
from .export_artifacts import write_coverage_audit_artifacts


def _character_elo_default_paths(lens, include_supplements):
    suffix = "-supplemented" if include_supplements else ""
    base = f"outputs/character-elo-{lens}{suffix}-current"
    return f"{base}.json", f"{base}.md"


def _character_elo_timeline_default_paths(lens, include_supplements):
    suffix = "-supplemented" if include_supplements else ""
    base = f"outputs/character-elo-{lens}-timeline{suffix}-current"
    return f"{base}.json", f"{base}.md"


def _character_elo_supplement_diff_default_paths(lens):
    if lens == "advantage":
        base = "outputs/character-elo-supplement-diff-current"
    else:
        base = f"outputs/character-elo-{lens}-supplement-diff-current"
    return f"{base}.json", f"{base}.md"


def _character_glicko2_default_paths(lens, include_supplements):
    suffix = "-supplemented" if include_supplements else ""
    base = f"outputs/character-glicko2-{lens}{suffix}-current"
    return f"{base}.json", f"{base}.md"


def _character_whr_default_paths(lens, include_supplements):
    suffix = "-supplemented" if include_supplements else ""
    base = f"outputs/character-whr-{lens}{suffix}-current"
    return f"{base}.json", f"{base}.md"


def _supplemented_default_paths(base_name, include_supplements):
    # Unlike the character-elo family, these builders have no default output
    # path at all when unsupplemented (an omitted --output/--markdown-output
    # simply prints the analysis to stdout, preserving existing behavior
    # exactly); --include-supplements is the only thing that turns on a
    # default path, and it always points at the -supplemented- artifact
    # names so the unsuffixed -current.* files are never touched by this
    # flag.
    if not include_supplements:
        return None, None
    base = f"outputs/{base_name}-supplemented-current"
    return f"{base}.json", f"{base}.md"


def _collect_runs(args):
    runs = list(args.runs or [])
    if getattr(args, "discover_runs", None):
        runs.extend(core.discover_annotation_run_dirs(args.discover_runs))
    deduped = []
    seen = set()
    for run in runs:
        key = str(run)
        if key not in seen:
            seen.add(key)
            deduped.append(run)
    return deduped


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
    corpus_review_parser.add_argument("--output", help="Optional JSON output path.")
    corpus_review_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    coverage_audit_parser = subparsers.add_parser(
        "coverage-audit",
        help="Find named characters present in accepted unit passages but not scored in the annotation, plus a narrator-presence heuristic.",
    )
    coverage_audit_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    coverage_audit_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    coverage_audit_parser.add_argument(
        "--narrator-min-first-person",
        type=int,
        default=2,
        help="Minimum first-person marker count for a unit to be flagged as a narrator candidate.",
    )
    coverage_audit_parser.add_argument("--output", default="outputs/coverage-audit-current.json", help="Optional JSON output path.")
    coverage_audit_parser.add_argument("--markdown-output", default="outputs/coverage-audit-current.md", help="Optional Markdown output path.")

    prepare_supplements_parser = subparsers.add_parser(
        "prepare-supplements",
        help="Prepare a supplement annotation run from coverage-audit-flagged candidate units.",
    )
    prepare_supplements_parser.add_argument(
        "--audit",
        default="outputs/coverage-audit-current.json",
        help="Coverage audit JSON path.",
    )
    prepare_supplements_parser.add_argument("--output", required=True, help="Supplement run directory to create.")
    prepare_supplements_units_group = prepare_supplements_parser.add_mutually_exclusive_group(required=True)
    prepare_supplements_units_group.add_argument("--units", help="Comma-separated unit ids to include.")
    prepare_supplements_units_group.add_argument(
        "--units-file",
        help="Path to a newline-delimited file of unit ids to include.",
    )
    prepare_supplements_parser.add_argument("--notes", default="", help="Optional note stored in run.json.")

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
        "--character",
        dest="characters",
        action="append",
        help="Optional character to include. Repeat for multiple characters. Defaults to the union of top rank-spread and top volatile figures.",
    )
    character_chapter_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the corpus review before computing the chapter analysis.",
    )
    character_chapter_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_chapter_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-chapter-cross-lens-supplemented-current.json with --include-supplements.")
    character_chapter_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-chapter-cross-lens-supplemented-current.md with --include-supplements.")

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
    character_counts_parser.add_argument("--output", help="Optional JSON output path.")
    character_counts_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    character_elo_parser = subparsers.add_parser(
        "character-elo",
        help="Build a lens ELO ranking from pairwise within-unit character comparisons.",
    )
    character_elo_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_elo_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_elo_parser.add_argument(
        "--lens",
        default="advantage",
        choices=list(core.SCORING_LENS_ORDER),
        help="Scoring lens the ELO ranking is computed over. Defaults to advantage.",
    )
    character_elo_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the overlay before computing ELO.",
    )
    character_elo_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_elo_parser.add_argument(
        "--min-match-count",
        type=int,
        default=10,
        help="Minimum pairwise match count required for a character to appear in the top/lowest/mismatch summary tables.",
    )
    character_elo_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-elo-advantage-current.json for the advantage lens (outputs/character-elo-<lens>-current.json for other lenses), or the -supplemented- variant with --include-supplements.")
    character_elo_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-elo-advantage-current.md for the advantage lens (outputs/character-elo-<lens>-current.md for other lenses), or the -supplemented- variant with --include-supplements.")

    character_glicko2_parser = subparsers.add_parser(
        "character-glicko2",
        help="Build a lens Glicko-2 rating surface from the same pairwise within-unit character comparisons as character-elo, batched by canonical chapter.",
    )
    character_glicko2_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_glicko2_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_glicko2_parser.add_argument(
        "--lens",
        default="advantage",
        choices=list(core.SCORING_LENS_ORDER),
        help="Scoring lens the Glicko-2 rating is computed over. Defaults to advantage.",
    )
    character_glicko2_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the overlay before computing Glicko-2.",
    )
    character_glicko2_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_glicko2_parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Glicko-2 system constant that constrains volatility over time. Defaults to 0.5.",
    )
    character_glicko2_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-glicko2-advantage-current.json for the advantage lens (outputs/character-glicko2-<lens>-current.json for other lenses), or the -supplemented- variant with --include-supplements.")
    character_glicko2_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-glicko2-advantage-current.md for the advantage lens (outputs/character-glicko2-<lens>-current.md for other lenses), or the -supplemented- variant with --include-supplements.")

    character_whr_parser = subparsers.add_parser(
        "character-whr",
        help="Build a lens Whole-History Rating surface: per-character rating trajectories with uncertainty bands, smoothed and/or filtered, from the same pairwise within-unit character comparisons as character-elo.",
    )
    character_whr_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    character_whr_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    character_whr_parser.add_argument(
        "--lens",
        default="advantage",
        choices=list(core.SCORING_LENS_ORDER),
        help="Scoring lens the rating is computed over. Defaults to advantage.",
    )
    character_whr_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the overlay before computing WHR.",
    )
    character_whr_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_whr_parser.add_argument(
        "--w2",
        type=float,
        help="Wiener drift rate in Elo^2 per unit of narrative time. Omitted, it is selected by sequential one-step-ahead log loss over the candidate set.",
    )
    character_whr_parser.add_argument(
        "--mode",
        default="both",
        choices=["smoothed", "filtered", "both"],
        help="Which trajectories to write. Defaults to both.",
    )
    character_whr_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-whr-<lens>-current.json, or the -supplemented- variant with --include-supplements.")
    character_whr_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-whr-<lens>-current.md, or the -supplemented- variant with --include-supplements.")

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
        "--lens",
        default="advantage",
        choices=list(core.SCORING_LENS_ORDER),
        help="Scoring lens the ELO timeline is computed over. Defaults to advantage.",
    )
    character_elo_timeline_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional tracked character to include. Repeat for multiple characters.",
    )
    character_elo_timeline_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the overlay before computing the ELO timeline.",
    )
    character_elo_timeline_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_elo_timeline_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-elo-advantage-timeline-current.json for the advantage lens (outputs/character-elo-<lens>-timeline-current.json for other lenses), or the -supplemented- variant with --include-supplements.")
    character_elo_timeline_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-elo-advantage-timeline-current.md for the advantage lens (outputs/character-elo-<lens>-timeline-current.md for other lenses), or the -supplemented- variant with --include-supplements.")

    elo_supplement_diff_parser = subparsers.add_parser(
        "elo-supplement-diff",
        help="Compute baseline-vs-supplemented character ELO in one invocation and write the supplemented ELO, supplemented ELO timeline, and before/after diff artifacts.",
    )
    elo_supplement_diff_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    elo_supplement_diff_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
    )
    elo_supplement_diff_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from. Defaults to outputs.",
    )
    elo_supplement_diff_parser.add_argument(
        "--lens",
        default="advantage",
        choices=list(core.SCORING_LENS_ORDER),
        help="Scoring lens the ELO diff is computed over. Defaults to advantage.",
    )
    elo_supplement_diff_parser.add_argument(
        "--character",
        dest="characters",
        action="append",
        help="Optional tracked character to include in the supplemented ELO timeline artifact. Repeat for multiple characters.",
    )
    elo_supplement_diff_parser.add_argument(
        "--min-match-count",
        type=int,
        default=10,
        help="Minimum pairwise match count required for a character to appear in the ELO summary tables (applied to both baseline and supplemented).",
    )
    elo_supplement_diff_parser.add_argument(
        "--clearing-threshold",
        type=int,
        default=30,
        help="Match-count threshold used for the diff's 'newly clearing' diagnostic.",
    )
    elo_supplement_diff_parser.add_argument(
        "--elo-output",
        help="JSON output path for the supplemented character-elo artifact. Defaults to outputs/character-elo-advantage-supplemented-current.json for the advantage lens (outputs/character-elo-<lens>-supplemented-current.json for other lenses).",
    )
    elo_supplement_diff_parser.add_argument(
        "--elo-markdown-output",
        help="Markdown output path for the supplemented character-elo artifact. Defaults to outputs/character-elo-advantage-supplemented-current.md for the advantage lens (outputs/character-elo-<lens>-supplemented-current.md for other lenses).",
    )
    elo_supplement_diff_parser.add_argument(
        "--elo-timeline-output",
        help="JSON output path for the supplemented character-elo-timeline artifact. Defaults to outputs/character-elo-advantage-timeline-supplemented-current.json for the advantage lens (outputs/character-elo-<lens>-timeline-supplemented-current.json for other lenses).",
    )
    elo_supplement_diff_parser.add_argument(
        "--elo-timeline-markdown-output",
        help="Markdown output path for the supplemented character-elo-timeline artifact. Defaults to outputs/character-elo-advantage-timeline-supplemented-current.md for the advantage lens (outputs/character-elo-<lens>-timeline-supplemented-current.md for other lenses).",
    )
    elo_supplement_diff_parser.add_argument(
        "--diff-output",
        help="JSON output path for the before/after diff artifact. Defaults to outputs/character-elo-supplement-diff-current.json for the advantage lens (outputs/character-elo-<lens>-supplement-diff-current.json for other lenses).",
    )
    elo_supplement_diff_parser.add_argument(
        "--diff-markdown-output",
        help="Markdown output path for the before/after diff artifact. Defaults to outputs/character-elo-supplement-diff-current.md for the advantage lens (outputs/character-elo-<lens>-supplement-diff-current.md for other lenses).",
    )

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
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the corpus review before computing profile cards.",
    )
    character_cards_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_cards_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-profile-cards-supplemented-current.json with --include-supplements.")
    character_cards_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-profile-cards-supplemented-current.md with --include-supplements.")

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
        "--character",
        dest="characters",
        action="append",
        help="Optional pilot character to include. Repeat for multiple characters.",
    )
    character_pages_parser.add_argument(
        "--include-supplements",
        action="store_true",
        help="Merge outputs/supplement-run-* annotations into the corpus review and overlay before computing character pages.",
    )
    character_pages_parser.add_argument(
        "--supplement-outputs-dir",
        default="outputs",
        help="Outputs directory to discover supplement-run-* directories from when --include-supplements is set. Defaults to outputs.",
    )
    character_pages_parser.add_argument("--output", help="Optional JSON output path. Defaults to outputs/character-pages-supplemented-current.json with --include-supplements.")
    character_pages_parser.add_argument("--markdown-output", help="Optional Markdown output path. Defaults to outputs/character-pages-supplemented-current.md with --include-supplements.")

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
    chapter_summaries_parser.add_argument("--output", help="Optional JSON output path.")
    chapter_summaries_parser.add_argument("--markdown-output", help="Optional Markdown output path.")

    chapter_overlay_parser = subparsers.add_parser(
        "chapter-overlays",
        help="Export chapter-keyed app overlay JSON from the accepted canonicalized corpus surface.",
    )
    chapter_overlay_parser.add_argument("--run", dest="runs", action="append", help="Run directory to include. Repeat for multiple runs.")
    chapter_overlay_parser.add_argument(
        "--discover-runs",
        nargs="?",
        const="outputs",
        help="Discover annotated run directories under this outputs directory. Defaults to outputs.",
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
            report = core.build_outcome_report(args.run, lens=args.lens)
        except core.RunManifestNotFoundError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "corpus-review":
        try:
            runs = _collect_runs(args)
            review = core.build_corpus_sanity_review(runs)
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_corpus_review_artifacts(review, json_output=args.output, markdown_output=args.markdown_output)
        if args.output or args.markdown_output:
            print(json.dumps({
                "run_count": review["run_count"],
                "declared_unit_count": review["declared_unit_count"],
                "valid_annotation_count": review["valid_annotation_count"],
                "json_output": args.output,
                "markdown_output": args.markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0

    if args.command == "coverage-audit":
        try:
            runs = _collect_runs(args)
            audit = coverage.build_coverage_audit(runs, narrator_min_first_person=args.narrator_min_first_person)
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        write_coverage_audit_artifacts(audit, json_output=args.output, markdown_output=args.markdown_output)
        print(json.dumps({
            "unit_count": audit["metadata"]["unit_count"],
            "flagged_unit_count": audit["metadata"]["flagged_unit_count"],
            "candidate_addition_count": audit["metadata"]["candidate_addition_count"],
            "narrator_candidate_unit_count": audit["metadata"]["narrator_candidate_unit_count"],
            "json_output": args.output,
            "markdown_output": args.markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-supplements":
        if args.units:
            unit_ids = [item.strip() for item in args.units.split(",") if item.strip()]
        else:
            unit_ids = [line.strip() for line in Path(args.units_file).read_text().splitlines() if line.strip()]
        try:
            manifest = supplement.prepare_supplement_run(
                args.audit,
                unit_ids,
                args.output,
                notes=args.notes,
            )
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            "run_id": manifest["run_id"],
            "unit_count": len(manifest["unit_ids"]),
            "output": args.output,
        }, ensure_ascii=False, indent=2))
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
            review = core.build_corpus_sanity_review(_collect_runs(args))
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
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_chapter_analysis(
                runs,
                target_characters=args.characters,
                supplement_run_dirs=supplement_run_dirs,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _supplemented_default_paths(
            "character-chapter-cross-lens", args.include_supplements
        )
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_chapter_analysis_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        if json_output or markdown_output:
            print(json.dumps({
                "selected_character_count": analysis["selected_character_count"],
                "json_output": json_output,
                "markdown_output": markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-annotation-counts":
        try:
            review = core.build_corpus_sanity_review(_collect_runs(args))
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
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_elo(
                runs,
                lens=args.lens,
                supplement_run_dirs=supplement_run_dirs,
                min_match_count=args.min_match_count,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _character_elo_default_paths(args.lens, args.include_supplements)
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_elo_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        print(json.dumps({
            "character_count": analysis["character_count"],
            "match_count": analysis["match_count"],
            "json_output": json_output,
            "markdown_output": markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-glicko2":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_glicko2(
                runs,
                lens=args.lens,
                supplement_run_dirs=supplement_run_dirs,
                tau=args.tau,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _character_glicko2_default_paths(args.lens, args.include_supplements)
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_glicko2_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        print(json.dumps({
            "character_count": analysis["character_count"],
            "match_count": analysis["match_count"],
            "period_count": analysis["period_count"],
            "json_output": json_output,
            "markdown_output": markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-whr":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_whr(
                runs,
                lens=args.lens,
                supplement_run_dirs=supplement_run_dirs,
                w2_elo=args.w2,
                mode=args.mode,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _character_whr_default_paths(args.lens, args.include_supplements)
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_whr_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        print(json.dumps({
            "character_count": analysis["character_count"],
            "match_count": analysis["match_count"],
            "time_point_count": analysis["time_point_count"],
            "w2_elo": analysis["w2_elo"],
            "json_output": json_output,
            "markdown_output": markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-elo-timeline":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_elo_timeline(
                runs,
                lens=args.lens,
                target_characters=args.characters,
                supplement_run_dirs=supplement_run_dirs,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _character_elo_timeline_default_paths(args.lens, args.include_supplements)
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_elo_timeline_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        print(json.dumps({
            "tracked_character_count": analysis["tracked_character_count"],
            "point_count": analysis["point_count"],
            "json_output": json_output,
            "markdown_output": markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "elo-supplement-diff":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = core.discover_supplement_run_dirs(args.supplement_outputs_dir)
            baseline = core.build_character_elo(runs, lens=args.lens, min_match_count=args.min_match_count)
            supplemented = core.build_character_elo(
                runs,
                lens=args.lens,
                supplement_run_dirs=supplement_run_dirs,
                min_match_count=args.min_match_count,
            )
            supplemented_timeline = core.build_character_elo_timeline(
                runs,
                lens=args.lens,
                target_characters=args.characters,
                supplement_run_dirs=supplement_run_dirs,
            )
            diff = core.build_character_elo_supplement_diff(
                baseline,
                supplemented,
                clearing_threshold=args.clearing_threshold,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_elo_json, default_elo_markdown = _character_elo_default_paths(args.lens, True)
        default_timeline_json, default_timeline_markdown = _character_elo_timeline_default_paths(args.lens, True)
        default_diff_json, default_diff_markdown = _character_elo_supplement_diff_default_paths(args.lens)
        elo_output = args.elo_output or default_elo_json
        elo_markdown_output = args.elo_markdown_output or default_elo_markdown
        elo_timeline_output = args.elo_timeline_output or default_timeline_json
        elo_timeline_markdown_output = args.elo_timeline_markdown_output or default_timeline_markdown
        diff_output = args.diff_output or default_diff_json
        diff_markdown_output = args.diff_markdown_output or default_diff_markdown
        core.write_character_elo_artifacts(supplemented, json_output=elo_output, markdown_output=elo_markdown_output)
        core.write_character_elo_timeline_artifacts(
            supplemented_timeline,
            json_output=elo_timeline_output,
            markdown_output=elo_timeline_markdown_output,
        )
        core.write_character_elo_supplement_diff_artifacts(diff, json_output=diff_output, markdown_output=diff_markdown_output)
        print(json.dumps({
            "match_count_before": diff["match_count"]["before"],
            "match_count_after": diff["match_count"]["after"],
            "newly_clearing_threshold_count": diff["newly_clearing_threshold_count"],
            "elo_output": elo_output,
            "elo_markdown_output": elo_markdown_output,
            "elo_timeline_output": elo_timeline_output,
            "elo_timeline_markdown_output": elo_timeline_markdown_output,
            "diff_output": diff_output,
            "diff_markdown_output": diff_markdown_output,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-profile-cards":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_profile_cards(runs, supplement_run_dirs=supplement_run_dirs)
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _supplemented_default_paths(
            "character-profile-cards", args.include_supplements
        )
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_profile_cards_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        if json_output or markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": json_output,
                "markdown_output": markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chapter-overlays":
        try:
            dataset = core.build_chapter_overlay_data(_collect_runs(args))
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        core.write_chapter_overlay_artifacts(dataset, args.output_dir)
        print(json.dumps({
            "chapter_count": dataset["chapter_count"],
            "output_dir": args.output_dir,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "character-pages":
        try:
            runs = _collect_runs(args)
            supplement_run_dirs = (
                core.discover_supplement_run_dirs(args.supplement_outputs_dir) if args.include_supplements else None
            )
            analysis = core.build_character_pages(
                runs,
                target_characters=args.characters,
                supplement_run_dirs=supplement_run_dirs,
            )
        except (core.RunManifestNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        default_json_output, default_markdown_output = _supplemented_default_paths(
            "character-pages", args.include_supplements
        )
        json_output = args.output or default_json_output
        markdown_output = args.markdown_output or default_markdown_output
        core.write_character_pages_artifacts(analysis, json_output=json_output, markdown_output=markdown_output)
        if json_output or markdown_output:
            print(json.dumps({
                "character_count": analysis["character_count"],
                "json_output": json_output,
                "markdown_output": markdown_output,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chapter-summaries":
        try:
            analysis = core.build_chapter_summary_export(_collect_runs(args))
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
