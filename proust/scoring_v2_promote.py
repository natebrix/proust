"""Promotion: the staged scoring v2 fits become the current analysis surface.

`proust/scoring_v2.py` is the formula, `proust/scoring_v2_build.py` fits it
over the foundation corpus and stages the result under `outputs/scoring-v2/`,
and this module publishes those staged fits as the `-current` artifacts the
project and the `islt` app read. v2 was adopted (see the adoption record in
`proust/docs/scoring_v2_design.md`), so promotion is a rendering and joining
step, never a re-fit: every number here is read back from the staged
artifacts, which keeps the current surface and the validated one identical
by construction.

Three surfaces come out of it.

1. **Standings** (`character-standings-{lens}-current.*`). The name view's
   final ratings, split into the characters the corpus compared often
   enough to rank and the ones it did not. That split is the honest
   presentation policy the adoption asked for: a wide band is missing
   evidence, not a low placement, and rendering the two in one ordered
   table would say the opposite. The person view promotes alongside as
   JSON.

2. **Journey timelines** (`character-journey-{lens}-timeline-current.*`).
   Smoothed and filtered trajectory nodes for the pilot editorial cast,
   every node joined to its corpus position exactly as
   `character_whr.build_character_whr_timeline` joined the v1 ones, so the
   app port is mechanical.

3. **Dossier pages** (`character-pages-current.*`). The v1 page machinery
   -- portraits, editorial, reading paths -- over a v2 profile: per-lens
   rating, band, rank, movement means, and the archetype signs, with
   notable units and chapters chosen by v2 movement rather than v1 net
   score.

The full point-by-point trajectories of every character stay in the staged
fits; the standings carry the per-character summaries and leave the series
to the journey timelines and to `outputs/scoring-v2/`, so no number is
written to disk twice and nothing can drift between the two copies.
"""

from collections import defaultdict
import json
from pathlib import Path

from . import scoring_v2 as v2
from . import scoring_v2_build
from .app_exports import (
    _chapter_title_map,
    _discover_character_portraits,
    _reader_chapter_link,
    _slugify_text,
)
from .character_whr import TIME_AXIS, TIMELINE_MODES
from .editorial import CHARACTER_PAGE_PILOT_EDITORIAL
from .registry import Registry
from .reporting_utils import format_signed_number, markdown_table

DEFAULT_STAGED_DIR = scoring_v2_build.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUTS_DIR = "outputs"
PROMOTED_VIEW = "name"

# The trajectories are the one thing the standings deliberately do not
# republish: they are large, they are already staged, and the journey
# timelines carry the app-facing slice of them.
STANDINGS_OMITTED_ROW_KEYS = ("smoothed_trajectory", "filtered_trajectory")

RANK_RULE = "dense_rank_by_conservative_rating"


# ---------------------------------------------------------------------------
# Reading the staged fits.
# ---------------------------------------------------------------------------


def staged_ratings_path(lens, view, staged_dir=DEFAULT_STAGED_DIR):
    return Path(staged_dir) / f"scoring-v2-{lens}-{view}-view-ratings.json"


def read_staged_ratings(lens, view, staged_dir=DEFAULT_STAGED_DIR):
    """One staged fit, or a message naming the build that would produce it."""
    v2.require_known_lens(lens)
    path = staged_ratings_path(lens, view, staged_dir=staged_dir)
    if not path.exists():
        raise ValueError(
            f"No staged scoring v2 fit at {path}. Run `python3 scripts/build_scoring_v2.py "
            "--stage build` before promoting."
        )
    return json.loads(path.read_text())


def read_staged_json(name, staged_dir=DEFAULT_STAGED_DIR):
    path = Path(staged_dir) / name
    if not path.exists():
        raise ValueError(
            f"No staged scoring v2 artifact at {path}. Run `python3 scripts/build_scoring_v2.py "
            "--stage build` before promoting."
        )
    return json.loads(path.read_text())


def assign_dense_ranks(rows, value_of):
    """Dense rank over rows already sorted best-first: ties share, no gaps."""
    rank = 0
    previous = None
    for row in rows:
        value = value_of(row)
        if previous is None or value != previous:
            rank += 1
            previous = value
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# 1. Standings.
# ---------------------------------------------------------------------------


def build_character_standings(ratings, corpus=None):
    """The promoted standings for one lens and view: ranked, and not yet rankable.

    Every row is the staged fit's row minus its point-by-point
    trajectories, and the two sections partition the fit's characters:
    `ranked` are the non-provisional ones ordered by conservative rating
    (`rating - band`) and densely ranked on it, `insufficient_evidence`
    are the provisional ones ordered by rating with no rank at all.

    The second section is not the bottom of the first. A provisional
    character's band is wider than the threshold because the corpus
    compared them too rarely, so their rating is where the fit currently
    sits and not a claim about where they stand; ordering them into the
    same table would turn missing evidence into a low placement.
    """
    lens = ratings["lens"]
    view = ratings["view"]
    rows = [
        {key: value for key, value in row.items() if key not in STANDINGS_OMITTED_ROW_KEYS}
        for row in ratings["characters"]
    ]

    ranked = sorted(
        (row for row in rows if not row["provisional"]),
        key=lambda row: (-row["conservative_rating"], row["character"]),
    )
    assign_dense_ranks(ranked, lambda row: row["conservative_rating"])

    insufficient = sorted(
        (row for row in rows if row["provisional"]),
        key=lambda row: (-row["rating"], row["character"]),
    )
    for row in insufficient:
        row["rank"] = None

    standings = {
        "character_standings_version": f"character_standings_{lens}_{view}_view_v2",
        "scoring_version": ratings["scoring_version"],
        "source_fit_version": ratings["scoring_v2_ratings_version"],
        "trajectory_source": str(staged_ratings_path(lens, view)),
        "lens": lens,
        "view": view,
        "time_axis": ratings["time_axis"],
        "w2_elo": ratings["w2_elo"],
        "w2_elo_selected_by": ratings["w2_elo_selected_by"],
        "tie_band": ratings["tie_band"],
        "draw_model": ratings["draw_model"],
        "initial_rating": ratings["initial_rating"],
        "initial_rd": ratings["initial_rd"],
        "band_provisional_threshold": ratings["band_provisional_threshold"],
        "conservative_rating_rule": ratings["conservative_rating_rule"],
        "rank_rule": RANK_RULE,
        "character_count": len(rows),
        "ranked_count": len(ranked),
        "insufficient_evidence_count": len(insufficient),
        "comparison_count": ratings["comparison_count"],
        "weight_total": ratings["weight_total"],
        "mean_weight": ratings["mean_weight"],
        "draw_rate": ratings["draw_rate"],
        "time_point_count": ratings["time_point_count"],
        "node_count": ratings["node_count"],
        "convergence": ratings["convergence"],
        "predictive_evaluation": ratings["predictive_evaluation"],
        "ranked": ranked,
        "insufficient_evidence": insufficient,
    }
    if ratings.get("dropped_self_pairings") is not None:
        standings["dropped_self_pairings"] = ratings["dropped_self_pairings"]
    if corpus:
        standings["corpus"] = corpus
    return standings


def format_standing(row):
    """`1552 ± 77`: the rating and the band it is worth to two significant places."""
    if row["band"] is None:
        return f"{row['rating']:.0f}"
    return f"{row['rating']:.0f} ± {row['band']:.0f}"


def render_character_standings_markdown(standings):
    lines = [
        f"# Character Standings — {standings['lens']} (scoring v2)",
        "",
        f"- Standings version: `{standings['character_standings_version']}`",
        f"- Scoring version: `{standings['scoring_version']}`",
        f"- Source fit: `{standings['source_fit_version']}` (`{standings['trajectory_source']}`)",
        f"- Lens / view: `{standings['lens']}` / `{standings['view']}`",
        f"- Time axis: `{standings['time_axis']}`",
        f"- Characters: `{standings['character_count']}` "
        f"(`{standings['ranked_count']}` ranked, "
        f"`{standings['insufficient_evidence_count']}` without sufficient evidence)",
        f"- Comparisons: `{standings['comparison_count']}` "
        f"(mean weight `{standings['mean_weight']}`, draw rate `{standings['draw_rate']}`)",
        f"- w2: `{standings['w2_elo']}` Elo² per unit of narrative time "
        f"(selected by `{standings['w2_elo_selected_by']}`)",
        f"- Provisional band threshold: `{standings['band_provisional_threshold']}` Elo",
        f"- Rank rule: `{standings['rank_rule']}`",
    ]
    if standings.get("corpus"):
        lines.append(f"- Corpus: `{standings['corpus']}`")

    lines.extend(
        [
            "",
            "Ratings read `1552 ± 77`: the rating, and the band that is `2*sigma` from the node's "
            "posterior variance -- an approximate 95% interval conditional on the other characters' "
            "trajectories. The ranked listing sorts by the conservative rating `rating - band`, so a "
            "character has to be both high and well-measured to place.",
            "",
            "The point-by-point trajectories behind these standings are not repeated here; they live "
            f"in `{standings['trajectory_source']}` and, for the pilot cast, in the "
            "`character-journey-*-timeline-current` artifacts.",
            "",
            "## Ranked",
            "",
            f"The `{standings['ranked_count']}` characters the corpus compared often enough for the "
            f"rating to mean something (band at or under `{standings['band_provisional_threshold']}` "
            "Elo), by conservative rating, densely ranked.",
            "",
            markdown_table(
                ["Rank", "Character", "Rating", "Conservative", "Comparisons", "W-L-D", "Units", "Mean m", "Mean abs m"],
                [
                    (
                        row["rank"],
                        row["character"],
                        format_standing(row),
                        row["conservative_rating"],
                        row["match_count"],
                        f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                        row["unit_count"],
                        format_signed_number(row["mean_movement"]),
                        row["mean_absolute_movement"],
                    )
                    for row in standings["ranked"]
                ],
            ),
            "",
            "## Insufficient comparative evidence",
            "",
            f"The `{standings['insufficient_evidence_count']}` characters whose band is still wider "
            f"than `{standings['band_provisional_threshold']}` Elo. THIS IS NOT THE BOTTOM OF THE "
            "TABLE ABOVE. These characters were not compared often enough for a standing to exist: "
            "the rating shown is where the fit currently sits, and it is listed here only so the "
            "reader can see who is unmeasured and how thin the evidence is. Sorted by rating, which "
            "is an ordering of the fit's current guesses and not of the characters.",
            "",
            markdown_table(
                ["Character", "Rating", "Band", "Comparisons", "Units", "Mean m", "Mean abs m"],
                [
                    (
                        row["character"],
                        format_standing(row),
                        row["band"],
                        row["match_count"],
                        row["unit_count"],
                        format_signed_number(row["mean_movement"]),
                        row["mean_absolute_movement"],
                    )
                    for row in standings["insufficient_evidence"]
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_character_standings_artifacts(standings, json_output=None, markdown_output=None):
    written = []
    if json_output:
        written.append(_write_text(json_output, json.dumps(standings, ensure_ascii=False, indent=2) + "\n"))
    if markdown_output:
        written.append(_write_text(markdown_output, render_character_standings_markdown(standings)))
    return written


# ---------------------------------------------------------------------------
# 2. App-shaped journey timelines.
# ---------------------------------------------------------------------------


def _corpus_positions_by_time(units):
    positions = {}
    for unit in units:
        position = unit.get("corpus_position")
        if position is None:
            raise ValueError(
                f'Unit "{unit["unit_id"]}" carries no corpus_position; load it with '
                "scoring_v2_build.load_scored_units so the journey timeline can join on it."
            )
        positions[unit["time"]] = position
    return positions


def build_character_journey_timeline(
    ratings,
    units,
    readings,
    view=PROMOTED_VIEW,
    target_characters=None,
    corpus=None,
):
    """v2 trajectories joined to corpus positions, one point per node.

    A trajectory node's time IS a `cumulative_unit_index`, so the join is
    the same inversion `build_character_whr_timeline` performed on the v1
    fits: `cumulative_unit_index -> the unit's full corpus position`. A
    node whose time has no position raises, because that means the fit and
    the corpus it was fitted on have drifted apart.

    Each point carries the WHR timeline's shape -- `mode`, `rating`,
    `band`, `corpus_position` -- so the app's existing timeline component
    ports across mechanically, plus the v2 reading of the same unit
    (`movement`, `label`) in place of v1's net score.
    """
    positions_by_time = _corpus_positions_by_time(units)
    character_counts_by_time = {
        unit["time"]: len(unit["annotation"].get("characters_present") or []) for unit in units
    }
    reading_by_key = {
        (reading["time"], scoring_v2_build.view_reading_key(reading, view)): reading
        for reading in readings
    }
    rows_by_character = {row["character"]: row for row in ratings["characters"]}
    selected = [
        character
        for character in (target_characters or CHARACTER_PAGE_PILOT_EDITORIAL.keys())
        if character in rows_by_character
    ]

    points = []
    point_counts = defaultdict(lambda: defaultdict(int))
    for character in selected:
        row = rows_by_character[character]
        for mode in TIMELINE_MODES:
            for time_point, rating, band in row.get(f"{mode}_trajectory", []):
                position = positions_by_time.get(time_point)
                if position is None:
                    raise ValueError(
                        f'{mode} trajectory node at time {time_point} for "{character}" has no '
                        "corpus position; the staged fit and the corpus have drifted apart."
                    )
                reading = reading_by_key.get((time_point, character))
                points.append(
                    {
                        "character": character,
                        "mode": mode,
                        "rating": rating,
                        "band": band,
                        "movement": reading["movement"] if reading else None,
                        "label": reading["label"] if reading else None,
                        "unit_character_count": character_counts_by_time.get(time_point),
                        "corpus_position": dict(position),
                    }
                )
                point_counts[character][mode] += 1

    points.sort(
        key=lambda point: (
            point["character"],
            point["mode"],
            point["corpus_position"]["cumulative_unit_index"],
        )
    )

    characters = []
    for character in selected:
        row = rows_by_character[character]
        characters.append(
            {
                "character": character,
                "node_count": row["node_count"],
                "smoothed_point_count": point_counts[character]["smoothed"],
                "filtered_point_count": point_counts[character]["filtered"],
                "final_rating": row["rating"],
                "final_band": row["band"],
                "provisional": row["provisional"],
            }
        )

    timeline = {
        "character_journey_timeline_version": (
            f"character_journey_timeline_{ratings['lens']}_{view}_view_v2"
        ),
        "scoring_version": ratings["scoring_version"],
        "source_fit_version": ratings["scoring_v2_ratings_version"],
        "lens": ratings["lens"],
        "view": view,
        "time_axis": TIME_AXIS,
        "modes": list(TIMELINE_MODES),
        "w2_elo": ratings["w2_elo"],
        "band_provisional_threshold": ratings["band_provisional_threshold"],
        "tracked_character_count": len(selected),
        "tracked_characters": selected,
        "point_count": len(points),
        "characters": characters,
        "points": points,
    }
    if corpus:
        timeline["corpus"] = corpus
    return timeline


def _format_journey_rating(row):
    if row["final_band"] is None:
        return f"{row['final_rating']:.1f}"
    return f"{row['final_rating']:.1f} ± {row['final_band']:.1f}"


def render_character_journey_timeline_markdown(timeline):
    lines = [
        f"# Character Journey Timeline — {timeline['lens']} (scoring v2)",
        "",
        f"- Timeline version: `{timeline['character_journey_timeline_version']}`",
        f"- Scoring version: `{timeline['scoring_version']}`",
        f"- Source fit: `{timeline['source_fit_version']}`",
        f"- Lens / view: `{timeline['lens']}` / `{timeline['view']}`",
        f"- Modes: `{', '.join(timeline['modes'])}`",
        f"- Time axis: `{timeline['time_axis']}`",
        f"- w2: `{timeline['w2_elo']}` Elo² per unit of narrative time",
        f"- Tracked character count: `{timeline['tracked_character_count']}`",
        f"- Point count: `{timeline['point_count']}`",
    ]
    if timeline.get("corpus"):
        lines.append(f"- Corpus: `{timeline['corpus']}`")

    lines.extend(
        [
            "",
            "Each point is one trajectory node joined to the corpus position of the unit it was "
            "fitted at: chapter, unit index, paragraph range, and the cumulative paragraph and word "
            "offsets an app needs to place it on a reading axis. The point also carries the v2 "
            "reading of that same unit (`movement`, `label`), so a chart can show what the character "
            "did in the passage that moved the rating. The full series lives in the JSON artifact; "
            "the table below is coverage only.",
            "",
            markdown_table(
                ["Character", "Nodes", "Smoothed Points", "Filtered Points", "Final Rating", "Ranked"],
                [
                    (
                        row["character"],
                        row["node_count"],
                        row["smoothed_point_count"],
                        row["filtered_point_count"],
                        _format_journey_rating(row),
                        "no" if row["provisional"] else "yes",
                    )
                    for row in timeline["characters"]
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_character_journey_timeline_artifacts(timeline, json_output=None, markdown_output=None):
    written = []
    if json_output:
        written.append(_write_text(json_output, json.dumps(timeline, ensure_ascii=False, indent=2) + "\n"))
    if markdown_output:
        written.append(_write_text(markdown_output, render_character_journey_timeline_markdown(timeline)))
    return written


# ---------------------------------------------------------------------------
# 3. Dossier pages on v2.
# ---------------------------------------------------------------------------


def _readings_by_character(readings_by_lens, lenses):
    """character -> one `{lens: reading}` per unit they appear in.

    A character present in a unit has a reading in every lens (a lens that
    saw nothing about them still reports a zero movement), so this is the
    per-appearance record both page derivations read across lenses.
    """
    by_unit = defaultdict(dict)
    for lens in lenses:
        for reading in readings_by_lens[lens]:
            by_unit[(reading["character"], reading["unit_id"])][lens] = reading

    index = defaultdict(list)
    for (character, _unit_id), readings in by_unit.items():
        index[character].append(readings)
    return index


def build_character_page_top_chapters(character_readings, lenses, chapter_titles, limit=5):
    """The chapters where this character moved most, by absolute v2 movement.

    Selection is by mass -- the total |movement| the chapter put on the
    character in its strongest lens -- because the question a page asks is
    "where does this character's story happen?", and a chapter earns that
    by moving them a lot in total. The per-lens means ride along so the
    direction of the mass is visible next to it.
    """
    by_chapter = {}
    for readings in character_readings:
        for lens, reading in readings.items():
            row = by_chapter.setdefault(
                reading["chapter_id"],
                {"chapter_id": reading["chapter_id"], "unit_ids": set(), "movements": {name: [] for name in lenses}},
            )
            row["unit_ids"].add(reading["unit_id"])
            row["movements"][lens].append(reading["movement"])

    rows = []
    for row in by_chapter.values():
        lens_cells = {}
        for lens in lenses:
            movements = row["movements"][lens]
            lens_cells[lens] = {
                "mean_movement": round(sum(movements) / len(movements), 4) if movements else 0.0,
                "absolute_movement": round(sum(abs(value) for value in movements), 4),
            }
        rows.append(
            {
                "chapter_id": row["chapter_id"],
                "chapter_title": chapter_titles.get(row["chapter_id"], row["chapter_id"]),
                "unit_count": len(row["unit_ids"]),
                "max_absolute_movement": round(
                    max(cell["absolute_movement"] for cell in lens_cells.values()), 4
                ),
                "reader_link": _reader_chapter_link(row["chapter_id"]),
                **lens_cells,
            }
        )

    rows.sort(key=lambda row: (-row["max_absolute_movement"], row["chapter_id"]))
    return rows[:limit]


def _notable_unit_label(annotation, character, lenses):
    """The annotator's own reason for the biggest thing that happened here.

    The status effect with the largest weighted contribution in any lens
    carries the explanation a reader wants next to the passage link. A
    character present without an effect has no such sentence, so the unit
    is named positionally instead.
    """
    best_explanation = None
    best_magnitude = 0.0
    for effect in annotation.get("status_effects") or []:
        if not isinstance(effect, dict) or effect.get("character") != character:
            continue
        magnitude = max(abs(v2.effect_movement(effect, lens)) for lens in lenses)
        explanation = effect.get("explanation")
        if explanation and magnitude > best_magnitude:
            best_magnitude = magnitude
            best_explanation = explanation
    return best_explanation


def build_character_page_notable_units(character, character_readings, units_by_id, lenses, limit=3):
    """The units that moved this character most, with the annotator's reason."""
    rows = []
    for readings in character_readings:
        any_reading = next(iter(readings.values()))
        unit_id = any_reading["unit_id"]
        movements = {lens: readings[lens]["movement"] for lens in lenses if lens in readings}
        labels = {lens: readings[lens]["label"] for lens in lenses if lens in readings}
        rows.append(
            {
                "unit_id": unit_id,
                "chapter_id": any_reading["chapter_id"],
                "max_absolute_movement": round(max(abs(value) for value in movements.values()), 4),
                "movements": movements,
                "labels": labels,
            }
        )

    rows.sort(key=lambda row: (-row["max_absolute_movement"], row["unit_id"]))

    notable = []
    for row in rows[:limit]:
        unit = units_by_id[row["unit_id"]]
        position = unit["corpus_position"]
        label = _notable_unit_label(unit["annotation"], character, lenses)
        if label is None:
            label = (
                f"{position['chapter_title']}, paragraphs "
                f"{position['paragraph_start']}-{position['paragraph_end']}"
            )
        notable.append(
            {
                "unit_id": row["unit_id"],
                "chapter_id": row["chapter_id"],
                "label": label,
                "movements": row["movements"],
                "labels": row["labels"],
                "max_absolute_movement": row["max_absolute_movement"],
                "reader_link": _reader_chapter_link(
                    row["chapter_id"], paragraph_start=position["paragraph_start"]
                ),
            }
        )
    return notable


def build_character_pages_v2(
    units,
    readings_by_lens,
    corpus_summary,
    standings_by_lens,
    target_characters=None,
    top_chapter_limit=5,
    notable_unit_limit=3,
    corpus=None,
):
    """The pilot dossier pages, scored by v2.

    The page machinery is v1's and stays that way: portraits are
    discovered the same way, the editorial block is read from
    `editorial.CHARACTER_PAGE_PILOT_EDITORIAL` AT BUILD TIME (so an
    editorial rewrite regenerates the pages without touching this code),
    and reading paths come straight off that editorial.

    What changed is the profile. `profile.lens_scores[lens]` no longer
    carries v1 net scores, ranks, and percentiles; it carries the v2
    standing (rating, band, conservative rating, dense rank out of the
    lens's ranked set, and whether the character is ranked at all)
    alongside the per-appearance movement means and label counts. A
    character who is provisional in a lens reports `rank: null` and
    `provisional: true` rather than a number that would read as a
    placement.
    """
    lenses = list(corpus_summary["lenses"])
    selected_characters = list(target_characters or CHARACTER_PAGE_PILOT_EDITORIAL.keys())
    summary_by_character = {row["character"]: row for row in corpus_summary["characters"]}
    units_by_id = {unit["unit_id"]: unit for unit in units}
    chapter_titles = _chapter_title_map()
    ranked_counts = {lens: standings_by_lens[lens]["ranked_count"] for lens in lenses}
    ranks_by_lens = {
        lens: {row["character"]: row["rank"] for row in standings_by_lens[lens]["ranked"]}
        for lens in lenses
    }
    readings_index = _readings_by_character(readings_by_lens, lenses)

    pages = []
    for character in selected_characters:
        if character not in summary_by_character:
            raise ValueError(f"Character page target has no scoring v2 readings: {character}")
        if character not in CHARACTER_PAGE_PILOT_EDITORIAL:
            raise ValueError(f"Character page editorial data is missing for: {character}")

        summary_row = summary_by_character[character]
        editorial = CHARACTER_PAGE_PILOT_EDITORIAL[character]
        character_readings = readings_index[character]

        lens_scores = {}
        for lens in lenses:
            cell = summary_row["lenses"][lens]
            lens_scores[lens] = {
                "rating": cell["rating"],
                "band": cell["band"],
                "conservative_rating": cell["conservative_rating"],
                "rank": ranks_by_lens[lens].get(character),
                "non_provisional_count": ranked_counts[lens],
                "provisional": cell["provisional"],
                "appearances": cell["appearances"],
                "mean_movement": cell["mean_movement"],
                "mean_absolute_movement": cell["mean_absolute_movement"],
                "labels": cell["labels"],
                "comparison_count": cell["comparison_count"],
            }

        pages.append(
            {
                "character": character,
                "slug": _slugify_text(character),
                "portrait": _discover_character_portraits(character),
                "profile": {
                    "annotation_unit_count": max(
                        lens_scores[lens]["appearances"] for lens in lenses
                    ),
                    "archetype_signs": summary_row["archetype_signs"],
                    "lens_scores": lens_scores,
                },
                "editorial": {
                    "subheading": editorial["subheading"],
                    "summary": editorial["summary"],
                    "why_interesting": editorial["why_interesting"],
                    "primary_pattern": editorial["primary_pattern"],
                },
                "top_chapters": build_character_page_top_chapters(
                    character_readings, lenses, chapter_titles, limit=top_chapter_limit
                ),
                "reading_path": [
                    {
                        "chapter_id": row["chapter_id"],
                        "label": row["label"],
                        "reader_link": _reader_chapter_link(row["chapter_id"]),
                    }
                    for row in editorial["reading_path"]
                ],
                "notable_units": build_character_page_notable_units(
                    character, character_readings, units_by_id, lenses, limit=notable_unit_limit
                ),
            }
        )

    pages.sort(key=lambda page: (-page["profile"]["annotation_unit_count"], page["character"]))

    result = {
        "character_pages_version": "character_pages_v2",
        "scoring_version": corpus_summary["scoring_version"],
        "source_summary_version": corpus_summary["scoring_v2_corpus_summary_version"],
        "view": corpus_summary["view"],
        "lenses": lenses,
        "character_count": len(pages),
        "pages": pages,
    }
    if corpus:
        result["corpus"] = corpus
    return result


def _format_page_standing(cell):
    if cell["rating"] is None:
        return "unrated"
    if cell["band"] is None:
        return f"{cell['rating']:.0f}"
    return f"{cell['rating']:.0f} ± {cell['band']:.0f}"


def _format_archetype_signs(signs):
    return ", ".join(f"{lens} {sign:+d}" for lens, sign in signs.items())


def _format_page_rank(cell):
    if cell["rank"] is None:
        return "insufficient evidence"
    return f"{cell['rank']} of {cell['non_provisional_count']}"


def render_character_pages_v2_markdown(analysis):
    lines = [
        "# Character Pages (scoring v2)",
        "",
        f"- Analysis version: `{analysis['character_pages_version']}`",
        f"- Scoring version: `{analysis['scoring_version']}`",
        f"- Source corpus summary: `{analysis['source_summary_version']}`",
        f"- View: `{analysis['view']}`",
        f"- Character count: `{analysis['character_count']}`",
    ]
    if analysis.get("corpus"):
        lines.append(f"- Corpus: `{analysis['corpus']}`")

    lines.extend(
        [
            "",
            "## Profile shape",
            "",
            "`profile.lens_scores[lens]` is scoring v2 and no longer carries v1 net scores, "
            "percentiles, or score spans. Its keys are:",
            "",
            "- `rating`, `band`, `conservative_rating`: the weighted-WHR standing at the "
            "character's last node, the `2*sigma` band around it, and `rating - band`",
            "- `rank`, `non_provisional_count`: dense rank by conservative rating among the lens's "
            "ranked characters, and how many characters that set holds. `rank` is `null` whenever "
            "`provisional` is true -- a wide band is missing evidence, not a low placement",
            "- `provisional`: true when the band still exceeds the fit's threshold",
            "- `appearances`: annotated units the character is present in (lens-independent)",
            "- `mean_movement`, `mean_absolute_movement`: direction and intensity per appearing "
            "unit. Both are means, never sums, so appearing often cannot raise either",
            "- `labels`: positive / negative / mixed / neutral unit counts in this lens",
            "- `comparison_count`: weighted comparisons the character took part in",
            "",
            "`profile.archetype_signs` gives the sign of each lens's rating against the initial "
            "rating: the three-way signature the lens-polarity archetypes are read from. "
            "`top_chapters` and `notable_units` are selected by v2 absolute movement, and a notable "
            "unit's label is the annotator's own explanation of the largest effect in it.",
            "",
        ]
    )

    for page in analysis["pages"]:
        lines.extend(
            [
                f"## {page['character']}",
                "",
                f"- Slug: `{page['slug']}`",
                f"- Portrait default: `{page['portrait']['default'] or 'none'}`",
                f"- Annotation units: `{page['profile']['annotation_unit_count']}`",
                f"- Archetype signs: `{_format_archetype_signs(page['profile']['archetype_signs'])}`",
                f"- Pattern: `{page['editorial']['primary_pattern']}`",
                "",
                page["editorial"]["subheading"],
                "",
                page["editorial"]["summary"],
                "",
                "Why interesting:",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in page["editorial"]["why_interesting"])
        lines.extend(
            [
                "",
                markdown_table(
                    ["Lens", "Standing", "Conservative", "Rank", "Appearances", "Mean m", "Mean abs m", "+/-/mixed/neutral"],
                    [
                        (
                            lens,
                            _format_page_standing(page["profile"]["lens_scores"][lens]),
                            page["profile"]["lens_scores"][lens]["conservative_rating"],
                            _format_page_rank(page["profile"]["lens_scores"][lens]),
                            page["profile"]["lens_scores"][lens]["appearances"],
                            format_signed_number(page["profile"]["lens_scores"][lens]["mean_movement"]),
                            page["profile"]["lens_scores"][lens]["mean_absolute_movement"],
                            "/".join(
                                str(page["profile"]["lens_scores"][lens]["labels"][label])
                                for label in ("positive", "negative", "mixed", "neutral")
                            ),
                        )
                        for lens in analysis["lenses"]
                    ],
                ),
                "",
                "Top chapters (by absolute movement):",
                "",
                markdown_table(
                    ["Chapter", "Units", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            row["unit_count"],
                            format_signed_number(row["advantage"]["mean_movement"]),
                            format_signed_number(row["prestige"]["mean_movement"]),
                            format_signed_number(row["inclusion"]["mean_movement"]),
                        )
                        for row in page["top_chapters"]
                    ],
                ),
                "",
                "Reading path:",
                "",
            ]
        )
        lines.extend(f"- {row['label']}: `{row['reader_link']}`" for row in page["reading_path"])
        lines.extend(["", "Notable units:", ""])
        lines.extend(f"- {row['label']}: `{row['reader_link']}`" for row in page["notable_units"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_character_pages_v2_artifacts(analysis, json_output=None, markdown_output=None):
    written = []
    if json_output:
        written.append(_write_text(json_output, json.dumps(analysis, ensure_ascii=False, indent=2) + "\n"))
    if markdown_output:
        written.append(_write_text(markdown_output, render_character_pages_v2_markdown(analysis)))
    return written


# ---------------------------------------------------------------------------
# The whole promotion.
# ---------------------------------------------------------------------------


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def standings_paths(lens, view=PROMOTED_VIEW, outputs_dir=DEFAULT_OUTPUTS_DIR):
    suffix = "" if view == PROMOTED_VIEW else f"-{view}-view"
    base = Path(outputs_dir) / f"character-standings-{lens}{suffix}-current"
    return f"{base}.json", f"{base}.md"


def journey_timeline_paths(lens, outputs_dir=DEFAULT_OUTPUTS_DIR):
    base = Path(outputs_dir) / f"character-journey-{lens}-timeline-current"
    return f"{base}.json", f"{base}.md"


def character_pages_paths(outputs_dir=DEFAULT_OUTPUTS_DIR):
    base = Path(outputs_dir) / "character-pages-current"
    return f"{base}.json", f"{base}.md"


def promote_scoring_v2(
    run_dirs,
    staged_dir=DEFAULT_STAGED_DIR,
    outputs_dir=DEFAULT_OUTPUTS_DIR,
    lenses=v2.SCORING_V2_LENS_ORDER,
    registry=None,
    target_characters=None,
    progress=None,
):
    """Rebuild every promoted v2 artifact from the staged fits, deterministically.

    Nothing is re-fitted: the ratings are read back from `staged_dir` and
    only the corpus-side joins (positions, readings, annotations) are
    recomputed from `run_dirs`, which is cheap and carries no randomness.
    Running this twice over the same staged fits and the same corpus
    produces byte-identical artifacts.
    """
    lenses = list(lenses)
    run_dirs = list(run_dirs)
    manifest = read_staged_json("scoring-v2-build-manifest.json", staged_dir=staged_dir)
    corpus_summary = read_staged_json("scoring-v2-corpus-summary.json", staged_dir=staged_dir)
    corpus = manifest.get("corpus")

    registry = registry or Registry.load()
    merge_map = v2.person_view_merge_map(registry)
    if progress:
        progress(f"loading {len(run_dirs)} foundation runs")
    units = scoring_v2_build.load_scored_units(run_dirs)
    if len(units) != manifest["unit_count"]:
        raise ValueError(
            f"The corpus has {len(units)} reviewed units but the staged fits were built over "
            f"{manifest['unit_count']}. Re-run `scripts/build_scoring_v2.py --stage build` before "
            "promoting."
        )

    readings_by_lens = {}
    standings_by_lens = {}
    written = []
    for lens in lenses:
        readings_by_lens[lens] = scoring_v2_build.build_readings(
            units, lens, registry=registry, merge_map=merge_map
        )
        for view in scoring_v2_build.VIEWS:
            ratings = read_staged_ratings(lens, view, staged_dir=staged_dir)
            standings = build_character_standings(ratings, corpus=corpus)
            json_output, markdown_output = standings_paths(lens, view=view, outputs_dir=outputs_dir)
            written.extend(
                write_character_standings_artifacts(
                    standings,
                    json_output=json_output,
                    markdown_output=markdown_output if view == PROMOTED_VIEW else None,
                )
            )
            if view != PROMOTED_VIEW:
                continue

            standings_by_lens[lens] = standings
            timeline = build_character_journey_timeline(
                ratings,
                units,
                readings_by_lens[lens],
                view=view,
                target_characters=target_characters,
                corpus=corpus,
            )
            timeline_json, timeline_markdown = journey_timeline_paths(lens, outputs_dir=outputs_dir)
            written.extend(
                write_character_journey_timeline_artifacts(
                    timeline, json_output=timeline_json, markdown_output=timeline_markdown
                )
            )
            if progress:
                progress(
                    f"{lens}: {standings['ranked_count']} ranked, "
                    f"{standings['insufficient_evidence_count']} without sufficient evidence, "
                    f"{timeline['point_count']} timeline points"
                )

    pages = build_character_pages_v2(
        units,
        readings_by_lens,
        corpus_summary,
        standings_by_lens,
        target_characters=target_characters,
        corpus=corpus,
    )
    pages_json, pages_markdown = character_pages_paths(outputs_dir=outputs_dir)
    written.extend(
        write_character_pages_v2_artifacts(pages, json_output=pages_json, markdown_output=pages_markdown)
    )
    if progress:
        progress(f"pages: {pages['character_count']} characters")

    return {"written": written, "standings": standings_by_lens, "pages": pages}


__all__ = [
    "DEFAULT_OUTPUTS_DIR",
    "DEFAULT_STAGED_DIR",
    "PROMOTED_VIEW",
    "RANK_RULE",
    "assign_dense_ranks",
    "build_character_journey_timeline",
    "build_character_page_notable_units",
    "build_character_page_top_chapters",
    "build_character_pages_v2",
    "build_character_standings",
    "character_pages_paths",
    "format_standing",
    "journey_timeline_paths",
    "promote_scoring_v2",
    "read_staged_json",
    "read_staged_ratings",
    "render_character_journey_timeline_markdown",
    "render_character_pages_v2_markdown",
    "render_character_standings_markdown",
    "staged_ratings_path",
    "standings_paths",
    "write_character_journey_timeline_artifacts",
    "write_character_pages_v2_artifacts",
    "write_character_standings_artifacts",
]
