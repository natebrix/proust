"""Build the fortune arcs: each character's own trajectory across the novel.

Usage:
    python3 scripts/build_fortune.py [--outputs-dir outputs]
                                     [--output-dir outputs/fortune]
                                     [--corpus foundation|enrichment]
                                     [--min-appearances 8]
                                     [--permutation-samples 300]

Writes `fortune-<lens>-<view>.json` for every lens (overall plus the three
scoring v2 lenses) in both views -- `person` (registry entities, merged
names, registry unit rulings) and `name` (annotation names as written) --
plus `fortune-report.md`, which reads the person view. The model is
`proust/fortune.py`; nothing here is tuned per character.
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
    _reader_chapter_link,
    _slugify_text,
    discover_enrichment_run_dirs,
    discover_foundation_run_dirs,
)
from proust.registry import Registry  # noqa: E402

VIEWS = ("person", "name")

# Order-permutation samples per listed character (see
# `fortune.order_permutation_p_values`), and the seed that makes them
# reproducible.
PERMUTATION_SAMPLES = 1000
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


def _point(time, level, sd, sigma2, marks):
    return {
        "time": time,
        "level": round(level, 3),
        "sd": round(sd, 3),
        "rating": round(fortune.fortune_rating(level, sigma2)),
        "rating_sd": round(fortune.elo_points_per_level(sigma2) * sd),
        "chapter_title": locate(time, marks)["chapter_title"],
    }


def _move(move, sd_by_time, sigma2, marks, p_value):
    if not move["size"]:
        return None
    return {
        "size": round(move["size"], 3),
        "rating_size": round(fortune.elo_points_per_level(sigma2) * move["size"]),
        "from": _point(move["from_time"], move["from_level"], sd_by_time[move["from_time"]], sigma2, marks),
        "to": _point(move["to_time"], move["to_level"], sd_by_time[move["to_time"]], sigma2, marks),
        "order_p": None if p_value is None else round(p_value, 4),
    }


def build_lens(units, lens, marks, min_appearances, keyer=None, view="name", samples=PERMUTATION_SAMPLES):
    series = fortune.character_series(units, lens, keyer=keyer)
    values = [value for rows in series.values() for _time, value, _weight in rows]
    prior_mean = sum(values) / len(values)
    q, sigma2, table = fortune.select_hyperparameters(series, prior_mean)
    best = max(row["log_likelihood"] for row in table)
    flat = fortune.pooled_log_likelihood(series, 1e-9, sigma2, prior_mean)
    points = fortune.elo_points_per_level(sigma2)

    rng = random.Random(f"{PERMUTATION_SEED}:{lens}:{view}")
    characters = []
    for key in sorted(series):
        rows = series[key]
        listed = len(rows) >= min_appearances
        p_values = (
            fortune.order_permutation_p_values(rows, q, sigma2, prior_mean, rng, samples=samples)
            if listed and samples
            else {"fall": None, "rise": None}
        )
        nodes, _log_likelihood = fortune.filter_and_smooth(rows, q, sigma2, prior_mean=prior_mean)
        summary = fortune.arc_summary(nodes)
        sd_by_time = {node["time"]: math.sqrt(node["smoothed_var"]) for node in nodes}
        characters.append(
            {
                "key": key,
                "character": keyer.display(key) if keyer else key,
                "appearances": len(rows),
                "listed": listed,
                **{
                    part: _point(summary[part]["time"], summary[part]["level"], summary[part]["sd"], sigma2, marks)
                    for part in ("start", "end", "peak", "trough")
                },
                "biggest_fall": _move(summary["biggest_fall"], sd_by_time, sigma2, marks, p_values["fall"]),
                "biggest_rise": _move(summary["biggest_rise"], sd_by_time, sigma2, marks, p_values["rise"]),
                "trajectory": [
                    [node["time"], round(node["smoothed"], 3), round(math.sqrt(node["smoothed_var"]), 3),
                     round(node["filtered"], 3)]
                    for node in nodes
                ],
                "observations": [[time, round(value, 3), round(weight, 3)] for time, value, weight in rows],
            }
        )

    return {
        "fortune_version": "fortune_v2",
        "lens": lens,
        "view": view,
        "time_axis": f"word midpoint / {fortune.WORDS_PER_TIME_UNIT}",
        "prior_mean": round(prior_mean, 4),
        "prior_sd": fortune.DEFAULT_PRIOR_SD,
        "q": q,
        "sigma2": sigma2,
        "selected_by": "pooled_marginal_likelihood",
        "rating_center": fortune.FORTUNE_RATING_CENTER,
        "rating_points_per_level": round(points, 2),
        "log_likelihood": round(best, 2),
        "log_likelihood_no_arcs": round(flat, 2),
        "arc_evidence": round(best - flat, 2),
        "grid": [{**row, "log_likelihood": round(row["log_likelihood"], 2)} for row in table],
        "min_appearances": min_appearances,
        "permutation_samples": samples,
        "chapter_marks": marks,
        "characters": characters,
    }


# ---------------------------------------------------------------------------
# App-facing export: what the islt character pages read.
# ---------------------------------------------------------------------------

APP_EXPORT_NAME = "character-fortune-current.json"
APP_EXPORT_VERSION = "character_fortune_v1"

# A lens "shows arcs" when the fitted model beats a no-arc model by at least
# this many log-likelihood points; below it the app should not draw lines.
ARC_EVIDENCE_TO_SHOW = 2.0

# The rule the design doc and the report call a clear move.
CLEAR_MAX_ORDER_P = 0.01
CLEAR_MIN_LEVEL_CHANGE = 0.25


def _app_point(point, x_of):
    """A start/end/peak/move endpoint, reusing the ratings the build computed."""
    return {
        "x": x_of(point["time"]),
        "rating": point["rating"],
        "sd": point["rating_sd"],
        "chapter_title": point["chapter_title"],
    }


def _app_move(move, x_of):
    if not move:
        return None
    return {
        "points": move["rating_size"],
        "from": _app_point(move["from"], x_of),
        "to": _app_point(move["to"], x_of),
        "order_p": move["order_p"],
        "clear": bool(
            move["order_p"] is not None
            and move["order_p"] <= CLEAR_MAX_ORDER_P
            and move["size"] >= CLEAR_MIN_LEVEL_CHANGE
        ),
    }


def build_app_export(results, units, marks, keyer, page_slugs, corpus):
    """One small file for the app: person view, precomputed ratings, positions, links.

    x is the fraction of the novel's words before a point (0 to 1), so the
    app never needs word counts or chapter lengths. Ratings and their
    standard deviations are already on the Elo-style scale. Only characters
    with enough passages for an arc in the overall lens are included.
    """
    total_words = max(unit["corpus_position"]["cumulative_word_count_end"] for unit in units)

    def x_of(time):
        return round(time * fortune.WORDS_PER_TIME_UNIT / total_words, 5)

    unit_at_time = {fortune.unit_time(unit["corpus_position"]): unit for unit in units}

    volumes = []
    for mark in marks:
        if not volumes or volumes[-1]["volume"] != mark["volume"]:
            volumes.append(
                {"volume": mark["volume"], "title": mark["chapter_title"].split(" — ")[0], "x": x_of(mark["time"])}
            )
    chapters = [
        {"chapter_id": mark["chapter_id"], "chapter_title": mark["chapter_title"], "volume": mark["volume"],
         "x": x_of(mark["time"])}
        for mark in marks
    ]

    lenses = {}
    for lens, views in results.items():
        result = views["person"]
        lenses[lens] = {
            "points_per_level": result["rating_points_per_level"],
            "arc_evidence": result["arc_evidence"],
            "shows_arcs": result["arc_evidence"] >= ARC_EVIDENCE_TO_SHOW,
        }

    overall = {row["key"]: row for row in results[fortune.OVERALL_LENS]["person"]["characters"] if row["listed"]}
    characters = []
    for key in sorted(overall, key=lambda k: -overall[k]["appearances"]):
        name = overall[key]["character"]
        entry = {
            "character": name,
            "key": key,
            "slug": page_slugs.get(name, _slugify_text(name)),
            "has_character_page": name in page_slugs,
            "lenses": {},
        }
        for lens, views in results.items():
            result = views["person"]
            row = next((r for r in result["characters"] if r["key"] == key and r["listed"]), None)
            if row is None or not lenses[lens]["shows_arcs"]:
                continue
            passages = []
            for time, movement, _weight in row["observations"]:
                unit = unit_at_time[time]
                position = unit["corpus_position"]
                passages.append(
                    {
                        "x": x_of(time),
                        "outcome": round(fortune.FORTUNE_RATING_CENTER + result["rating_points_per_level"] * movement),
                        "unit_id": unit["unit_id"],
                        "chapter_title": position["chapter_title"],
                        "reader_link": _reader_chapter_link(position["chapter_id"], position["paragraph_start"]),
                    }
                )
            entry["lenses"][lens] = {
                "passages_count": row["appearances"],
                "line": [
                    [x_of(time), round(fortune.FORTUNE_RATING_CENTER + result["rating_points_per_level"] * level),
                     round(result["rating_points_per_level"] * sd)]
                    for time, level, sd, _filtered in row["trajectory"]
                ],
                "passages": passages,
                "start": _app_point(row["start"], x_of),
                "end": _app_point(row["end"], x_of),
                "biggest_fall": _app_move(row["biggest_fall"], x_of),
                "biggest_rise": _app_move(row["biggest_rise"], x_of),
            }
        characters.append(entry)

    return {
        "character_fortune_version": APP_EXPORT_VERSION,
        "corpus": corpus,
        "view": "person",
        "x_axis": "fraction of the novel's words before the point (0 to 1)",
        "rating_center": fortune.FORTUNE_RATING_CENTER,
        "clear_move_rule": {"max_order_p": CLEAR_MAX_ORDER_P, "min_level_change": CLEAR_MIN_LEVEL_CHANGE},
        "lenses": lenses,
        "volumes": volumes,
        "chapters": chapters,
        "characters": characters,
    }


def _fmt_point(point):
    return f"{point['rating']} ± {point['rating_sd']}"


def _fmt_move(move):
    return (
        f"{_fmt_point(move['from'])} → {_fmt_point(move['to'])} ({move['rating_size']} pts) · "
        f"{move['from']['chapter_title']} → {move['to']['chapter_title']}"
    )


def _fmt_p(p):
    return "–" if p is None else f"{p:.3f}"


def render_report(results, keyer_notes, top=10):
    person = {lens: views["person"] for lens, views in results.items()}
    lines = [
        "# Fortune arcs",
        "",
        "Each character's own arc: a smoothed level of their scoring v2 movements, passage by",
        "passage, from `proust/fortune.py`, in the PERSON view (registry entities; names and titles",
        "of one person pooled). Drift and noise are chosen once per lens by pooled marginal",
        "likelihood; nothing is tuned per character.",
        "",
        "Ratings are Elo-style: 1500 is a level of 0 (passages leave the character where they were),",
        "and the points-per-level factor comes from the fitted passage noise, so a gap of D points",
        "means one character's next passage goes better than the other's about as often as a",
        "D-point Elo favourite wins (100 points ≈ 64%, 200 ≈ 76%, 400 ≈ 91%). `±` is one posterior",
        "standard deviation.",
        "",
        "`order p` asks whether the ORDER of a character's passages made the arc: the share of",
        "random reshufflings of their outcomes in time that give an equal or bigger move. Around",
        "40 characters are tested per lens, so a handful under 0.05 are expected by chance; lean on",
        "the ones near 0.01.",
        "",
        "`arc evidence` is how much better (in log-likelihood) the fitted model explains the",
        "movements than one in which no character's fortune ever changes.",
        "",
        "| lens | observations | people | q | sigma² | points per level | arc evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lens, result in person.items():
        lines.append(
            f"| {lens} | {sum(c['appearances'] for c in result['characters'])} | {len(result['characters'])} "
            f"| {result['q']:g} | {result['sigma2']:g} | {result['rating_points_per_level']:.0f} "
            f"| {result['arc_evidence']:.1f} |"
        )
    lines += ["", "## Person view rulings", ""] + [f"- {note}" for note in keyer_notes]

    for lens, result in person.items():
        listed = [row for row in result["characters"] if row["listed"]]
        lines += ["", f"## {lens}", "", f"People with at least {result['min_appearances']} appearances.", ""]
        for label, key in (("Biggest falls", "biggest_fall"), ("Biggest rises", "biggest_rise")):
            ranked = sorted((row for row in listed if row[key]), key=lambda row: -row[key]["size"])[:top]
            lines += [f"### {label}", "", "| character | n | order p | move |", "| --- | ---: | ---: | --- |"]
            lines += [
                f"| {row['character']} | {row['appearances']} | {_fmt_p(row[key]['order_p'])} | {_fmt_move(row[key])} |"
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
    keyer = fortune.PersonKeyer(Registry.load())

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lens in fortune.FORTUNE_LENS_ORDER:
        results[lens] = {}
        for view in VIEWS:
            result = build_lens(
                units,
                lens,
                marks,
                arguments.min_appearances,
                keyer=keyer if view == "person" else None,
                view=view,
                samples=arguments.permutation_samples,
            )
            result["corpus"] = arguments.corpus
            results[lens][view] = result
            path = output_dir / f"fortune-{lens}-{view}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
            print(
                f"{lens}/{view}: q={result['q']:g} sigma2={result['sigma2']:g} "
                f"{result['rating_points_per_level']:.0f} pts/level "
                f"arc evidence={result['arc_evidence']:.1f} -> {path}",
                flush=True,
            )

    keyer_notes = [
        f"merged by `person_view_merge`: "
        + ", ".join(f"{keyer.display(source)} → {keyer.display(target)}" for source, target in sorted(keyer.merge_map.items())),
    ]
    keyer_notes += [
        f"registry unit ruling: \"{name}\" in `{unit_id}` → "
        f"{keyer.display(keyer.registry.unit_rulings[(unit_id, name)].entity_id)}"
        for unit_id, name in sorted(keyer.ruled)
    ]
    keyer_notes += [
        f"left on the name (registry resolves it as ambiguous in `{chapter_id or 'any chapter'}`: "
        f"{', '.join(candidates)}): \"{name}\""
        for (chapter_id, name), candidates in sorted(keyer.ambiguous.items(), key=lambda item: (item[0][0] or "", item[0][1]))
    ] or ["no names left ambiguous by the registry"]
    (output_dir / "fortune-report.md").write_text(render_report(results, keyer_notes))

    pages_path = Path(arguments.outputs_dir) / "character-pages-current.json"
    page_slugs = (
        {page["character"]: page["slug"] for page in json.loads(pages_path.read_text())["pages"]}
        if pages_path.exists()
        else {}
    )
    app_export = build_app_export(results, units, marks, keyer, page_slugs, arguments.corpus)
    app_path = Path(arguments.outputs_dir) / APP_EXPORT_NAME
    app_path.write_text(json.dumps(app_export, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"app export: {len(app_export['characters'])} characters -> {app_path}", flush=True)


if __name__ == "__main__":
    main()
