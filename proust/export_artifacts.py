import json
from pathlib import Path

from . import runner as legacy


def render_corpus_review_normalization_diff_markdown(diff):
    lines = [
        "# Corpus Review Normalization Diff",
        "",
        f"- Diff version: `{diff['normalization_diff_version']}`",
        f"- Reviewed merges: `{len(diff['character_normalization_map'])}`",
        "",
        "## Character Map",
        "",
        legacy._markdown_table(
            ["Source Name", "Normalized Name"],
            diff["character_normalization_map"].items(),
        ),
        "",
        "## Lens Diffs",
        "",
    ]

    for lens, lens_diff in diff["lens_diffs"].items():
        lines.extend(
            [
                f"### {lens}",
                "",
                f"- Character count: `{lens_diff['character_count_before']}` -> `{lens_diff['character_count_after']}`",
                "",
                "Normalized character movement:",
                "",
                legacy._markdown_table(
                    [
                        "Character",
                        "Merged From",
                        "Net Before",
                        "Net After",
                        "Units Before",
                        "Units After",
                        "Positive Rank",
                        "Negative Rank",
                    ],
                    [
                        (
                            row["character"],
                            ", ".join(row["merged_from"]),
                            legacy._format_signed_number(row["net_score_before"]),
                            legacy._format_signed_number(row["net_score_after"]),
                            row["unit_count_before"],
                            row["unit_count_after"],
                            f"{row['positive_rank_before']} -> {row['positive_rank_after']}",
                            f"{row['negative_rank_before']} -> {row['negative_rank_after']}",
                        )
                        for row in lens_diff["normalized_characters"]
                    ],
                ),
                "",
                "Top positive characters:",
                "",
                legacy._markdown_table(
                    ["Before", "After"],
                    [
                        (
                            row_before["character"] if index < len(lens_diff["top_positive_before"]) else "",
                            row_after["character"] if index < len(lens_diff["top_positive_after"]) else "",
                        )
                        for index, (row_before, row_after) in enumerate(
                            zip(
                                lens_diff["top_positive_before"] + [{}] * 10,
                                lens_diff["top_positive_after"] + [{}] * 10,
                            )
                        )
                        if index < 10
                    ],
                ),
                "",
                "Top negative characters:",
                "",
                legacy._markdown_table(
                    ["Before", "After"],
                    [
                        (
                            row_before["character"] if index < len(lens_diff["top_negative_before"]) else "",
                            row_after["character"] if index < len(lens_diff["top_negative_after"]) else "",
                        )
                        for index, (row_before, row_after) in enumerate(
                            zip(
                                lens_diff["top_negative_before"] + [{}] * 10,
                                lens_diff["top_negative_after"] + [{}] * 10,
                            )
                        )
                        if index < 10
                    ],
                ),
                "",
            ]
        )

    cross_lens_diff = diff["cross_lens_summary_diff"]
    lines.extend(
        [
            "## Cross-Lens Summary Diff",
            "",
            f"- Comparable entries: `{cross_lens_diff['comparable_entry_count_before']}` -> `{cross_lens_diff['comparable_entry_count_after']}`",
            f"- Label disagreements: `{cross_lens_diff['label_disagreement_count_before']}` -> `{cross_lens_diff['label_disagreement_count_after']}`",
            f"- Direction disagreements: `{cross_lens_diff['direction_disagreement_count_before']}` -> `{cross_lens_diff['direction_disagreement_count_after']}`",
            f"- Sign-flip examples: `{cross_lens_diff['sign_flip_count_before']}` -> `{cross_lens_diff['sign_flip_count_after']}`",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def write_corpus_review_normalization_diff_artifacts(diff, markdown_output=None):
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_corpus_review_normalization_diff_markdown(diff))


def render_character_cross_lens_analysis_markdown(analysis):
    lines = [
        "# Character Cross-Lens Analysis",
        "",
        f"- Analysis version: `{analysis['character_cross_lens_analysis_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
        "## Top Positive By Lens",
        "",
    ]

    for lens in sorted(legacy.SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                legacy._markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            legacy._format_signed_number(row["net_score"]),
                            row["unit_count"],
                        )
                        for row in analysis["top_positive_by_lens"][lens]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Top Negative By Lens",
            "",
        ]
    )

    for lens in sorted(legacy.SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                legacy._markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            legacy._format_signed_number(row["net_score"]),
                            row["unit_count"],
                        )
                        for row in analysis["top_negative_by_lens"][lens]
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Largest Cross-Lens Rank Spread",
            "",
            legacy._markdown_table(
                ["Character", "Advantage Rank", "Prestige Rank", "Inclusion Rank", "Rank Spread", "Max Units"],
                [
                    (
                        row["character"],
                        row["lens_scores"]["advantage"]["rank"],
                        row["lens_scores"]["prestige"]["rank"],
                        row["lens_scores"]["inclusion"]["rank"],
                        row["rank_spread"],
                        row["max_unit_count"],
                    )
                    for row in analysis["top_rank_spread_characters"]
                ],
            ),
            "",
            "## Highest Volatility",
            "",
            legacy._markdown_table(
                ["Character", "Advantage Span", "Prestige Span", "Inclusion Span", "Max Span", "Max Units"],
                [
                    (
                        row["character"],
                        legacy._format_signed_number(row["lens_scores"]["advantage"]["score_span"]),
                        legacy._format_signed_number(row["lens_scores"]["prestige"]["score_span"]),
                        legacy._format_signed_number(row["lens_scores"]["inclusion"]["score_span"]),
                        legacy._format_signed_number(row["max_score_span"]),
                        row["max_unit_count"],
                    )
                    for row in analysis["top_volatile_characters"]
                ],
            ),
            "",
            "## Character Table",
            "",
            legacy._markdown_table(
                [
                    "Character",
                    "Advantage",
                    "Prestige",
                    "Inclusion",
                    "Advantage Rank",
                    "Prestige Rank",
                    "Inclusion Rank",
                    "Max Units",
                    "Max Span",
                ],
                [
                    (
                        row["character"],
                        legacy._format_signed_number(row["lens_scores"]["advantage"]["net_score"]),
                        legacy._format_signed_number(row["lens_scores"]["prestige"]["net_score"]),
                        legacy._format_signed_number(row["lens_scores"]["inclusion"]["net_score"]),
                        row["lens_scores"]["advantage"]["rank"],
                        row["lens_scores"]["prestige"]["rank"],
                        row["lens_scores"]["inclusion"]["rank"],
                        row["max_unit_count"],
                        legacy._format_signed_number(row["max_score_span"]),
                    )
                    for row in analysis["characters"][:40]
                ],
            ),
            "",
        ]
    )

    if len(analysis["characters"]) > 40:
        lines.extend(
            [
                f"_Showing first 40 of {len(analysis['characters'])} character rows._",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_character_cross_lens_analysis_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_cross_lens_analysis_markdown(analysis))


def render_character_chapter_analysis_markdown(analysis):
    lines = [
        "# Character Chapter Analysis",
        "",
        f"- Analysis version: `{analysis['character_chapter_analysis_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Selected character count: `{analysis['selected_character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for character_row in analysis["characters"]:
        summary = character_row["cross_lens_summary"]
        lines.extend(
            [
                f"## {character_row['character']}",
                "",
                f"- Selected by: `{', '.join(character_row['selected_by']) or 'manual'}`",
                f"- Advantage / Prestige / Inclusion ranks: `{summary['lens_scores']['advantage']['rank']}` / `{summary['lens_scores']['prestige']['rank']}` / `{summary['lens_scores']['inclusion']['rank']}`",
                f"- Rank spread: `{summary['rank_spread']}`",
                f"- Max units: `{summary['max_unit_count']}`",
                f"- Max score span: `{legacy._format_signed_number(summary['max_score_span'])}`",
                "",
                legacy._markdown_table(
                    [
                        "Chapter",
                        "Advantage",
                        "Prestige",
                        "Inclusion",
                        "Advantage Units",
                        "Prestige Units",
                        "Inclusion Units",
                    ],
                    [
                        (
                            row["chapter_id"],
                            legacy._format_signed_number(row["advantage"]["net_score"]),
                            legacy._format_signed_number(row["prestige"]["net_score"]),
                            legacy._format_signed_number(row["inclusion"]["net_score"]),
                            row["advantage"]["unit_count"],
                            row["prestige"]["unit_count"],
                            row["inclusion"]["unit_count"],
                        )
                        for row in character_row["chapters"]
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_character_chapter_analysis_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_chapter_analysis_markdown(analysis))


def render_character_annotation_counts_markdown(analysis):
    lines = [
        "# Character Annotation Counts",
        "",
        f"- Analysis version: `{analysis['character_annotation_counts_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
        legacy._markdown_table(
            ["Character", "Annotation Units", "Advantage", "Prestige", "Inclusion"],
            [
                (
                    row["character"],
                    row["annotation_unit_count"],
                    legacy._format_signed_number(row["advantage_net_score"]),
                    legacy._format_signed_number(row["prestige_net_score"]),
                    legacy._format_signed_number(row["inclusion_net_score"]),
                )
                for row in analysis["characters"]
            ],
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_character_annotation_counts_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_annotation_counts_markdown(analysis))


def render_character_elo_markdown(analysis):
    lines = [
        "# Character ELO",
        "",
        f"- Analysis version: `{analysis['character_elo_version']}`",
        f"- Lens: `{analysis['lens']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Match count: `{analysis['match_count']}`",
        f"- Draw rate: `{analysis['draw_rate']}`",
        f"- Initial rating: `{analysis['initial_rating']}`",
        f"- K factor: `{analysis['k_factor']}`",
        f"- Epsilon: `{analysis['epsilon']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
        "## Top Rated Characters",
        "",
        legacy._markdown_table(
            ["Character", "ELO", "Matches", "W-L-D", "Units", "Mean Advantage"],
            [
                (
                    row["character"],
                    row["elo"],
                    row["match_count"],
                    f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                    row["unit_count"],
                    legacy._format_signed_number(row["mean_advantage_net_score"]),
                )
                for row in analysis["top_rated_characters"]
            ],
        ),
        "",
        "## Lowest Rated Characters",
        "",
        legacy._markdown_table(
            ["Character", "ELO", "Matches", "W-L-D", "Units", "Mean Advantage"],
            [
                (
                    row["character"],
                    row["elo"],
                    row["match_count"],
                    f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                    row["unit_count"],
                    legacy._format_signed_number(row["mean_advantage_net_score"]),
                )
                for row in analysis["lowest_rated_characters"]
            ],
        ),
        "",
        "## Largest Rank Mismatches",
        "",
        legacy._markdown_table(
            ["Character", "ELO Rank", "Mean Score Rank", "Delta", "ELO", "Mean Advantage"],
            [
                (
                    row["character"],
                    row["elo_rank"],
                    row["mean_score_rank"],
                    row["elo_rank_minus_mean_score_rank"],
                    row["elo"],
                    legacy._format_signed_number(row["mean_advantage_net_score"]),
                )
                for row in analysis["largest_rank_mismatches"]
            ],
        ),
        "",
        "## Character Table",
        "",
        legacy._markdown_table(
            [
                "Character",
                "ELO",
                "ELO Rank",
                "Matches",
                "W-L-D",
                "Units",
                "Mean Advantage",
                "Top Positive Unit",
                "Top Negative Unit",
            ],
            [
                (
                    row["character"],
                    row["elo"],
                    row["elo_rank"],
                    row["match_count"],
                    f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                    row["unit_count"],
                    legacy._format_signed_number(row["mean_advantage_net_score"]),
                    (
                        f"{row['top_positive_unit']['unit_id']} ({legacy._format_signed_number(row['top_positive_unit']['net_score'])})"
                        if row["top_positive_unit"]
                        else ""
                    ),
                    (
                        f"{row['top_negative_unit']['unit_id']} ({legacy._format_signed_number(row['top_negative_unit']['net_score'])})"
                        if row["top_negative_unit"]
                        else ""
                    ),
                )
                for row in analysis["characters"]
            ],
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_character_elo_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_elo_markdown(analysis))


def render_character_elo_timeline_markdown(analysis):
    lines = [
        "# Character ELO Timeline",
        "",
        f"- Analysis version: `{analysis['character_elo_timeline_version']}`",
        f"- Lens: `{analysis['lens']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Timeline type: `{analysis['timeline_type']}`",
        f"- Tracked character count: `{analysis['tracked_character_count']}`",
        f"- Point count: `{analysis['point_count']}`",
        f"- Initial rating: `{analysis['initial_rating']}`",
        f"- K factor: `{analysis['k_factor']}`",
        f"- Epsilon: `{analysis['epsilon']}`",
        "",
        "## Character Coverage",
        "",
        legacy._markdown_table(
            ["Character", "Points", "Final ELO", "Latest Unit"],
            [
                (
                    row["character"],
                    row["point_count"],
                    row["final_elo"],
                    (
                        row["latest_corpus_position"]["unit_id"]
                        if row["latest_corpus_position"]
                        else ""
                    ),
                )
                for row in analysis["characters"]
            ],
        ),
        "",
        "## Sample Points",
        "",
        legacy._markdown_table(
            [
                "Character",
                "ELO",
                "Advantage",
                "Label",
                "Chapter",
                "Unit",
                "Cumulative Unit",
                "Cumulative Words",
            ],
            [
                (
                    row["character"],
                    row["elo"],
                    legacy._format_signed_number(row["advantage_net_score"]),
                    row["advantage_label"],
                    row["corpus_position"]["chapter_id"],
                    row["corpus_position"]["unit_id"],
                    row["corpus_position"]["cumulative_unit_index"],
                    row["corpus_position"]["cumulative_word_count"],
                )
                for row in analysis["points"][:40]
            ],
        ),
        "",
    ]
    if len(analysis["points"]) > 40:
        lines.extend([f"_Showing first 40 of {len(analysis['points'])} timeline points._", "",])
    return "\n".join(lines).rstrip() + "\n"


def write_character_elo_timeline_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_elo_timeline_markdown(analysis))


def render_character_profile_cards_markdown(analysis):
    lines = [
        "# Character Profile Cards",
        "",
        f"- Analysis version: `{analysis['character_profile_cards_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for card in analysis["cards"][:20]:
        lines.extend(
            [
                f"## {card['character']}",
                "",
                f"- Annotation units: `{card['annotation_unit_count']}`",
                f"- Rank spread: `{card['rank_spread']}`",
                f"- Max score span: `{legacy._format_signed_number(card['max_score_span'])}`",
                f"- Selected by: `{', '.join(card['selected_by']) or 'none'}`",
                "",
                legacy._markdown_table(
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            legacy._format_signed_number(card["lens_scores"][lens]["net_score"]),
                            (
                                legacy._format_ordinal(card["lens_scores"][lens]["percentile"])
                                if card["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            card["lens_scores"][lens]["rank"],
                            card["lens_scores"][lens]["unit_count"],
                            card["lens_scores"][lens]["dominant_status_dimension"],
                            legacy._format_signed_number(card["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(legacy.SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                legacy._markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            legacy._format_signed_number(row["advantage"]["net_score"]),
                            legacy._format_signed_number(row["prestige"]["net_score"]),
                            legacy._format_signed_number(row["inclusion"]["net_score"]),
                        )
                        for row in card["top_chapters"]
                    ],
                ),
                "",
            ]
        )

    if len(analysis["cards"]) > 20:
        lines.extend([f"_Showing first 20 of {len(analysis['cards'])} cards._", "",])

    return "\n".join(lines).rstrip() + "\n"


def write_character_profile_cards_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_profile_cards_markdown(analysis))


def write_chapter_overlay_artifacts(dataset, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(dataset["manifest"], ensure_ascii=False, indent=2) + "\n")

    chapters_dir = output_path / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    for chapter in dataset["chapters"]:
        chapter_path = chapters_dir / f"{chapter['chapterId']}.json"
        chapter_path.write_text(json.dumps(chapter, ensure_ascii=False, indent=2) + "\n")


def render_character_pages_markdown(analysis):
    lines = [
        "# Character Pages",
        "",
        f"- Analysis version: `{analysis['character_pages_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        "",
    ]

    for page in analysis["pages"]:
        lines.extend(
            [
                f"## {page['character']}",
                "",
                f"- Slug: `{page['slug']}`",
                f"- Portrait default: `{page['portrait']['default'] or 'none'}`",
                f"- Annotation units: `{page['profile']['annotation_unit_count']}`",
                f"- Rank spread: `{page['profile']['rank_spread']}`",
                f"- Max score span: `{legacy._format_signed_number(page['profile']['max_score_span'])}`",
                f"- Pattern: `{page['editorial']['primary_pattern']}`",
                "",
                page["editorial"]["dek"],
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
                legacy._markdown_table(
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            legacy._format_signed_number(page["profile"]["lens_scores"][lens]["net_score"]),
                            (
                                legacy._format_ordinal(page["profile"]["lens_scores"][lens]["percentile"])
                                if page["profile"]["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            page["profile"]["lens_scores"][lens]["rank"],
                            page["profile"]["lens_scores"][lens]["unit_count"],
                            page["profile"]["lens_scores"][lens]["dominant_status_dimension"],
                            legacy._format_signed_number(page["profile"]["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(legacy.SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                legacy._markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            legacy._format_signed_number(row["advantage"]["net_score"]),
                            legacy._format_signed_number(row["prestige"]["net_score"]),
                            legacy._format_signed_number(row["inclusion"]["net_score"]),
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
        lines.extend(
            [
                "",
                "Notable units:",
                "",
            ]
        )
        lines.extend(f"- {row['label']}: `{row['reader_link']}`" for row in page["notable_units"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_character_pages_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_pages_markdown(analysis))


def render_chapter_summary_export_markdown(analysis):
    lines = [
        "# Chapter Summary Export",
        "",
        f"- Analysis version: `{analysis['chapter_summary_export_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Chapter count: `{analysis['chapter_count']}`",
        f"- Character normalization applied: `{analysis['character_normalization']['applied']}`",
        f"- Intensity medians: `advantage={analysis['intensity_medians']['advantage']}`, `prestige={analysis['intensity_medians']['prestige']}`, `inclusion={analysis['intensity_medians']['inclusion']}`",
        "",
    ]

    for chapter in analysis["chapters"]:
        strongest_split = chapter.get("strongest_split_character")
        lines.extend(
            [
                f"## {chapter['chapter_id']}",
                "",
                f"- Title: `{chapter['chapter_title']}`",
                f"- Unit count: `{chapter['unit_count']}`",
                f"- Reader link: `{chapter['reader_link']}`",
                f"- Tonal archetype: `{chapter['tonal_archetype']['label']}`",
                f"- Strongest split: `{strongest_split['character'] if strongest_split else 'none'}`",
                "",
                chapter["summary"] or "No summary available.",
                "",
                legacy._markdown_table(
                    ["Lens", "Direction", "Signed Density", "Intensity Density", "Chapter Rank"],
                    [
                        (
                            lens,
                            chapter["lens_profile"][lens]["direction"],
                            legacy._format_signed_number(chapter["lens_profile"][lens]["signed_density"]),
                            chapter["lens_profile"][lens]["intensity_density"],
                            f"{chapter['lens_profile'][lens]['chapter_rank']} ({legacy._format_ordinal(chapter['lens_profile'][lens]['chapter_percentile'])})",
                        )
                        for lens in legacy.SCORING_LENS_ORDER
                    ],
                ),
                "",
                legacy._markdown_table(
                    ["Character", "Units", "Impact Mass", "Dominant Lens", "Signature"],
                    [
                        (
                            row["character"],
                            row["unit_count"],
                            row["impact_mass"],
                            row["dominant_lens"],
                            row["lens_signature"],
                        )
                        for row in chapter["top_characters"]
                    ],
                ),
                "",
                legacy._markdown_table(
                    ["Passage", "Impact Mass", "Dominant Character", "Lens Signature", "Summary"],
                    [
                        (
                            f"{row['unit_id']} ({row['paragraph_start']}-{row['paragraph_end']})",
                            row["impact_mass"],
                            row["dominant_character"] or "none",
                            ", ".join(
                                f"{lens} {row['lens_signature'][lens]}"
                                for lens in legacy.SCORING_LENS_ORDER
                            ),
                            row["summary"],
                        )
                        for row in chapter["distinguishing_passages"]
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_chapter_summary_export_artifacts(analysis, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_chapter_summary_export_markdown(analysis))
