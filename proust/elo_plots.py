import json
import os
from pathlib import Path
import tempfile

_cache_root = Path(tempfile.gettempdir()) / "proust-matplotlib-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_config_dir = _cache_root / "mplconfig"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_COLORS = [
    "#1f4b99",
    "#c04b2f",
    "#2a7f62",
    "#7a4bb7",
    "#8a6d1d",
    "#555555",
]


def _volume_starts(points, x_axis):
    starts = {}
    for row in points:
        volume_number = row["corpus_position"]["volume_number"]
        starts.setdefault(volume_number, row["corpus_position"][x_axis])
    return starts


def _draw_volume_markers(axis, volume_starts):
    for volume_number, x_start in sorted(volume_starts.items()):
        axis.axvline(x_start, color="#d0d0d0", linewidth=0.8, linestyle=":", zorder=0)
        axis.text(
            x_start,
            axis.get_ylim()[1],
            f"V{volume_number}",
            fontsize=8,
            color="#666666",
            ha="left",
            va="bottom",
        )


def load_character_elo_timeline(path):
    return json.loads(Path(path).read_text())


def get_character_elo_points(timeline, character):
    points = [row for row in timeline["points"] if row["character"] == character]
    points.sort(
        key=lambda row: (
            row["corpus_position"]["cumulative_unit_index"],
            row["corpus_position"]["chapter_id"],
            row["corpus_position"]["unit_id"],
        )
    )
    return points


def plot_character_elo_timeline(
    timeline,
    character,
    output_path,
    x_axis="cumulative_unit_index",
    title=None,
):
    points = get_character_elo_points(timeline, character)
    if not points:
        raise ValueError(f"No ELO timeline points found for character: {character}")

    x_values = [row["corpus_position"][x_axis] for row in points]
    y_values = [row["elo"] for row in points]
    volume_starts = _volume_starts(points, x_axis)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(x_values, y_values, color=LINE_COLORS[0], linewidth=1.8, marker="o", markersize=2.2)
    axis.axhline(1500, color="#888888", linewidth=1, linestyle="--", alpha=0.8)

    for label in ("win", "loss", "mixed", "neutral"):
        label_points = [
            (row["corpus_position"][x_axis], row["elo"])
            for row in points
            if row["advantage_label"] == label
        ]
        if not label_points:
            continue
        xs, ys = zip(*label_points)
        axis.scatter(
            xs,
            ys,
            s=18,
            alpha=0.8,
            label=label,
            color={
                "win": "#2e8b57",
                "loss": "#b03a2e",
                "mixed": "#b8860b",
                "neutral": "#6c757d",
            }[label],
        )

    _draw_volume_markers(axis, volume_starts)

    axis.set_title(title or f"{character} Advantage ELO Timeline")
    axis.set_xlabel(x_axis.replace("_", " ").title())
    axis.set_ylabel("ELO")
    axis.legend(loc="best", frameon=False, ncols=4)
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_character_elo_comparison(
    timeline,
    characters,
    output_path,
    x_axis="cumulative_unit_index",
    title=None,
):
    if not characters:
        raise ValueError("At least one character is required for an ELO comparison plot.")

    series = []
    for character in characters:
        points = get_character_elo_points(timeline, character)
        if not points:
            raise ValueError(f"No ELO timeline points found for character: {character}")
        series.append((character, points))

    all_points = [point for _character, points in series for point in points]
    volume_starts = _volume_starts(all_points, x_axis)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.axhline(1500, color="#888888", linewidth=1, linestyle="--", alpha=0.8)

    for index, (character, points) in enumerate(series):
        x_values = [row["corpus_position"][x_axis] for row in points]
        y_values = [row["elo"] for row in points]
        axis.plot(
            x_values,
            y_values,
            color=LINE_COLORS[index % len(LINE_COLORS)],
            linewidth=2.0,
            marker="o",
            markersize=2.0,
            label=character,
        )

    _draw_volume_markers(axis, volume_starts)

    axis.set_title(title or "Character Advantage ELO Comparison")
    axis.set_xlabel(x_axis.replace("_", " ").title())
    axis.set_ylabel("ELO")
    axis.legend(loc="best", frameon=False)
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
