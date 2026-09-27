"""Build the fortune arcs: each character's own trajectory across the novel.

Usage:
    python3 scripts/build_fortune.py [--outputs-dir outputs]
                                     [--output-dir outputs/fortune]
                                     [--corpus foundation|enrichment]
                                     [--min-appearances 8]

Writes `fortune-<lens>.json` for the overall lens and each scoring v2 lens,
plus `fortune-report.md`. The model is `proust/fortune.py`; nothing here is
tuned per character. See that module's docstring for what a level means.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust import fortune  # noqa: E402
from proust import scoring_v2_build  # noqa: E402
from proust.app_exports import (  # noqa: E402
    discover_enrichment_run_dirs,
    discover_foundation_run_dirs,
)

# Order-permutation samples per listed character (see
# `fortune.order_permutation_p_values`), and the seed that makes them
# reproducible.
PERMUTATION_SAMPLES = 300
PERMUTATION_SEED = 20260927


def chapter_marks(units):
    """(time, volume, chapter_id, chapter_title) at each chapter's first passage."""
    marks = []
    seen = set()
    for unit in sorted(units, key=lambda row: row["time"]):
        position = unit["corpus_position"]
        if position["chapter_id"] in seen:
            continue
        seen.add(position["chapter_id"])
        marks.append(
            {
                "time": fortune.unit_time(position),
                "volume": position["volume_number"],
                "chapter_id": position["chapter_id"],
                "chapter_title": position["chapter_title"],
            }
        )
    return marks


def locate(time, marks):
    current = marks[0]
    for mark in marks:
        if mark["time"] <= time:
            current = mark
    return current


def _move(summary, key, marks, p_value=None):
    move = summary[key]
    if not move["size"]:
        return None
    from_sd = summary["_by_time"][move["from_time"]]["sd"]
    to_sd = summary["_by_time"][move["to_time"]]["sd"]
    return {
        "size": round(move["size"], 3),
        "from": {
            "time": move["from_time"],
            "level": round(move["from_level"], 3),
            "sd": round(from_sd, 3),
            "chapter_title": locate(move["from_time"], marks)["chapter_title"],
        },
        "to": {
            "time": move["to_time"],
            "level": round(move["to_level"], 3),
            "sd": round(to_sd, 3),
            "chapter_title": locate(move["to_time"], marks)["chapter_title"],
        },
        "order_p": None if p_value is None else round(p_value, 4),
    }


def build_lens(units, lens, marks, min_appearances, samples=PERMUTATION_SAMPLES):
    series = fortune.character_series(units, lens)
    values = [value for rows in series.values() for _time, value, _weight in rows]
    prior_mean = sum(values) / len(values)
    q, sigma2, table = fortune.select_hyperparameters(series, prior_mean)
    best = max(row["log_likelihood"] for row in table)
    flat = fortune.pooled_log_likelihood(series, 1e-9, sigma2, prior_mean)

    rng = random.Random(f"{PERMUTATION_SEED}:{lens}")
    characters = []
    for name in sorted(series):
        listed = len(series[name]) >= min_appearances
        p_values = (
            fortune.order_permutation_p_values(series[name], q, sigma2, prior_mean, rng, samples=samples)
            if listed and samples
            else {"fall": None, "rise": None}
        )
        nodes, _log_likelihood = fortune.filter_and_smooth(series[name], q, sigma2, prior_mean=prior_mean)
        summary = fortune.arc_summary(nodes)
        summary["_by_time"] = {node["time"]: {"sd": math.sqrt(node["smoothed_var"])} for node in nodes}
        row = {
            "character": name,
            "appearances": len(series[name]),
            "listed": listed,
            "start": {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["start"].items()},
            "end": {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["end"].items()},
            "peak": {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["peak"].items()},
            "trough": {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["trough"].items()},
            "biggest_fall": _move(summary, "biggest_fall", marks, p_values["fall"]),
            "biggest_rise": _move(summary, "biggest_rise", marks, p_values["rise"]),
            "trajectory": [
                [node["time"], round(node["smoothed"], 3), round(math.sqrt(node["smoothed_var"]), 3),
                 round(node["filtered"], 3)]
                for node in nodes
            ],
            "observations": [[time, round(value, 3), round(weight, 3)] for time, value, weight in series[name]],
        }
        characters.append(row)

    return {
        "fortune_version": "fortune_v1",
        "lens": lens,
        "time_axis": f"word midpoint / {fortune.WORDS_PER_TIME_UNIT}",
        "prior_mean": round(prior_mean, 4),
        "prior_sd": fortune.DEFAULT_PRIOR_SD,
        "q": q,
        "sigma2": sigma2,
        "selected_by": "pooled_marginal_likelihood",
        "log_likelihood": round(best, 2),
        "log_likelihood_no_arcs": round(flat, 2),
        "arc_evidence": round(best - flat, 2),
        "grid": [{**row, "log_likelihood": round(row["log_likelihood"], 2)} for row in table],
        "min_appearances": min_appearances,
        "permutation_samples": samples,
        "chapter_marks": marks,
        "characters": characters,
    }


def _fmt_move(move):
    return (
        f"{move['from']['level']:+.2f} ± {move['from']['sd']:.2f} → {move['to']['level']:+.2f} ± "
        f"{move['to']['sd']:.2f} ({move['size']:.2f}) · "
        f"{move['from']['chapter_title']} → {move['to']['chapter_title']}"
    )


def render_report(results, top=10):
    lines = [
        "# Fortune arcs",
        "",
        "Each character's own arc: a smoothed level of their scoring v2 movements, passage by",
        "passage, from `proust/fortune.py`. A level of +0.5 means that around this point of the novel",
        "the passages involving the character tend to leave them half a step up. Levels carry ± one",
        "posterior standard deviation. Drift and noise are chosen once per lens by pooled marginal",
        "likelihood; nothing is tuned per character.",
        "",
        "`order p` asks whether the ORDER of a character's passages made the arc: the share of",
        "random reshufflings of their outcomes in time that give an equal or bigger move. Around",
        "40 characters are tested per lens, so a handful under 0.05 are expected by chance; lean on",
        "the ones near 0.01.",
        "",
        "`arc evidence` is how much better (in log-likelihood) the fitted model explains the",
        "movements than one in which no character's fortune ever changes. Several points is strong",
        "evidence that arcs are real; under one point means the lens cannot see arcs.",
        "",
        "| lens | observations | characters | q | sigma² | arc evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lens, result in results.items():
        lines.append(
            f"| {lens} | {sum(c['appearances'] for c in result['characters'])} | {len(result['characters'])} "
            f"| {result['q']:g} | {result['sigma2']:g} | {result['arc_evidence']:.1f} |"
        )
    for lens, result in results.items():
        listed = [row for row in result["characters"] if row["listed"]]
        lines += ["", f"## {lens}", "", f"Characters with at least {result['min_appearances']} appearances.", ""]
        for label, key in (("Biggest falls", "biggest_fall"), ("Biggest rises", "biggest_rise")):
            ranked = sorted(
                (row for row in listed if row[key]), key=lambda row: -row[key]["size"]
            )[:top]
            lines += [f"### {label}", "", "| character | n | order p | move |", "| --- | ---: | ---: | --- |"]
            lines += [
                f"| {row['character']} | {row['appearances']} | {row[key]['order_p']:.3f} | {_fmt_move(row[key])} |"
                for row in ranked
            ]
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/fortune")
    parser.add_argument("--corpus", choices=("foundation", "enrichment"), default="enrichment")
    parser.add_argument("--min-appearances", type=int, default=8)
    parser.add_argument("--permutation-samples", type=int, default=PERMUTATION_SAMPLES)
    arguments = parser.parse_args()

    if arguments.corpus == "foundation":
        run_dirs = discover_foundation_run_dirs(arguments.outputs_dir)
    else:
        run_dirs = discover_enrichment_run_dirs(arguments.outputs_dir)
    units = scoring_v2_build.load_scored_units(run_dirs)
    marks = chapter_marks(units)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lens in fortune.FORTUNE_LENS_ORDER:
        results[lens] = build_lens(
            units, lens, marks, arguments.min_appearances, samples=arguments.permutation_samples
        )
        results[lens]["corpus"] = arguments.corpus
        path = output_dir / f"fortune-{lens}.json"
        path.write_text(json.dumps(results[lens], ensure_ascii=False, indent=1) + "\n")
        print(f"{lens}: q={results[lens]['q']:g} sigma2={results[lens]['sigma2']:g} "
              f"arc evidence={results[lens]['arc_evidence']:.1f} -> {path}", flush=True)
    (output_dir / "fortune-report.md").write_text(render_report(results))


if __name__ == "__main__":
    main()
