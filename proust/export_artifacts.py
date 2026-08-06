from collections import defaultdict
import json
from pathlib import Path

from .reporting_utils import (
    format_ordinal,
    format_signed_number,
    markdown_table,
)
from .scoring import SCORING_LENS_CONFIGS, SCORING_LENS_ORDER


def render_character_cross_lens_analysis_markdown(analysis):
    lines = [
        "# Character Cross-Lens Analysis",
        "",
        f"- Analysis version: `{analysis['character_cross_lens_analysis_version']}`",
        f"- Source review version: `{analysis['source_review_version']}`",
        f"- Character count: `{analysis['character_count']}`",
        "",
        "## Top Positive By Lens",
        "",
    ]

    for lens in sorted(SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            format_signed_number(row["net_score"]),
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

    for lens in sorted(SCORING_LENS_CONFIGS):
        lines.extend(
            [
                f"### {lens}",
                "",
                markdown_table(
                    ["Character", "Net Score", "Units"],
                    [
                        (
                            row["character"],
                            format_signed_number(row["net_score"]),
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
            markdown_table(
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
            markdown_table(
                ["Character", "Advantage Span", "Prestige Span", "Inclusion Span", "Max Span", "Max Units"],
                [
                    (
                        row["character"],
                        format_signed_number(row["lens_scores"]["advantage"]["score_span"]),
                        format_signed_number(row["lens_scores"]["prestige"]["score_span"]),
                        format_signed_number(row["lens_scores"]["inclusion"]["score_span"]),
                        format_signed_number(row["max_score_span"]),
                        row["max_unit_count"],
                    )
                    for row in analysis["top_volatile_characters"]
                ],
            ),
            "",
            "## Character Table",
            "",
            markdown_table(
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
                        format_signed_number(row["lens_scores"]["advantage"]["net_score"]),
                        format_signed_number(row["lens_scores"]["prestige"]["net_score"]),
                        format_signed_number(row["lens_scores"]["inclusion"]["net_score"]),
                        row["lens_scores"]["advantage"]["rank"],
                        row["lens_scores"]["prestige"]["rank"],
                        row["lens_scores"]["inclusion"]["rank"],
                        row["max_unit_count"],
                        format_signed_number(row["max_score_span"]),
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
                f"- Max score span: `{format_signed_number(summary['max_score_span'])}`",
                "",
                markdown_table(
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
                            format_signed_number(row["advantage"]["net_score"]),
                            format_signed_number(row["prestige"]["net_score"]),
                            format_signed_number(row["inclusion"]["net_score"]),
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
        "",
        markdown_table(
            ["Character", "Annotation Units", "Advantage", "Prestige", "Inclusion"],
            [
                (
                    row["character"],
                    row["annotation_unit_count"],
                    format_signed_number(row["advantage_net_score"]),
                    format_signed_number(row["prestige_net_score"]),
                    format_signed_number(row["inclusion_net_score"]),
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
    mean_column_label = f"Mean {analysis['lens'].capitalize()}"
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
        f"- Min match count (summary tables): `{analysis.get('min_match_count', 0)}`",
    ]
    if analysis.get("supplemented"):
        lines.append(f"- Supplemented: `true` (runs: {', '.join(analysis.get('supplement_runs', [])) or 'none'})")
    lines.extend(
        [
            "",
            "## Top Rated Characters",
        "",
        markdown_table(
            ["Character", "ELO", "Matches", "W-L-D", "Units", mean_column_label],
            [
                (
                    row["character"],
                    row["elo"],
                    row["match_count"],
                    f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                    row["unit_count"],
                    format_signed_number(row["mean_net_score"]),
                )
                for row in analysis["top_rated_characters"]
            ],
        ),
        "",
        "## Lowest Rated Characters",
        "",
        markdown_table(
            ["Character", "ELO", "Matches", "W-L-D", "Units", mean_column_label],
            [
                (
                    row["character"],
                    row["elo"],
                    row["match_count"],
                    f"{row['win_count']}-{row['loss_count']}-{row['draw_count']}",
                    row["unit_count"],
                    format_signed_number(row["mean_net_score"]),
                )
                for row in analysis["lowest_rated_characters"]
            ],
        ),
        "",
        "## Largest Rank Mismatches",
        "",
        markdown_table(
            ["Character", "ELO Rank", "Mean Score Rank", "Delta", "ELO", mean_column_label],
            [
                (
                    row["character"],
                    row["elo_rank"],
                    row["mean_score_rank"],
                    row["elo_rank_minus_mean_score_rank"],
                    row["elo"],
                    format_signed_number(row["mean_net_score"]),
                )
                for row in analysis["largest_rank_mismatches"]
            ],
        ),
        "",
        "## Character Table",
        "",
        markdown_table(
            [
                "Character",
                "ELO",
                "ELO Rank",
                "Matches",
                "W-L-D",
                "Units",
                mean_column_label,
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
                    format_signed_number(row["mean_net_score"]),
                    (
                        f"{row['top_positive_unit']['unit_id']} ({format_signed_number(row['top_positive_unit']['net_score'])})"
                        if row["top_positive_unit"]
                        else ""
                    ),
                    (
                        f"{row['top_negative_unit']['unit_id']} ({format_signed_number(row['top_negative_unit']['net_score'])})"
                        if row["top_negative_unit"]
                        else ""
                    ),
                )
                for row in analysis["characters"]
            ],
        ),
        "",
        ]
    )
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
        markdown_table(
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
        markdown_table(
            [
                "Character",
                "ELO",
                analysis["lens"].capitalize(),
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
                    format_signed_number(row["net_score"]),
                    row["label"],
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
        "",
    ]

    for card in analysis["cards"][:20]:
        lines.extend(
            [
                f"## {card['character']}",
                "",
                f"- Annotation units: `{card['annotation_unit_count']}`",
                f"- Rank spread: `{card['rank_spread']}`",
                f"- Max score span: `{format_signed_number(card['max_score_span'])}`",
                f"- Selected by: `{', '.join(card['selected_by']) or 'none'}`",
                "",
                markdown_table(
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            format_signed_number(card["lens_scores"][lens]["net_score"]),
                            (
                                format_ordinal(card["lens_scores"][lens]["percentile"])
                                if card["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            card["lens_scores"][lens]["rank"],
                            card["lens_scores"][lens]["unit_count"],
                            card["lens_scores"][lens]["dominant_status_dimension"],
                            format_signed_number(card["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            format_signed_number(row["advantage"]["net_score"]),
                            format_signed_number(row["prestige"]["net_score"]),
                            format_signed_number(row["inclusion"]["net_score"]),
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
                f"- Max score span: `{format_signed_number(page['profile']['max_score_span'])}`",
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
                    ["Lens", "Net Score", "Percentile", "Rank", "Units", "Dominant Dimension", "Score Span"],
                    [
                        (
                            lens,
                            format_signed_number(page["profile"]["lens_scores"][lens]["net_score"]),
                            (
                                format_ordinal(page["profile"]["lens_scores"][lens]["percentile"])
                                if page["profile"]["lens_scores"][lens]["percentile"] is not None
                                else ""
                            ),
                            page["profile"]["lens_scores"][lens]["rank"],
                            page["profile"]["lens_scores"][lens]["unit_count"],
                            page["profile"]["lens_scores"][lens]["dominant_status_dimension"],
                            format_signed_number(page["profile"]["lens_scores"][lens]["score_span"]),
                        )
                        for lens in sorted(SCORING_LENS_CONFIGS)
                    ],
                ),
                "",
                "Top chapters:",
                "",
                markdown_table(
                    ["Chapter", "Advantage", "Prestige", "Inclusion"],
                    [
                        (
                            row["chapter_id"],
                            format_signed_number(row["advantage"]["net_score"]),
                            format_signed_number(row["prestige"]["net_score"]),
                            format_signed_number(row["inclusion"]["net_score"]),
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
                markdown_table(
                    ["Lens", "Direction", "Signed Density", "Intensity Density", "Chapter Rank"],
                    [
                        (
                            lens,
                            chapter["lens_profile"][lens]["direction"],
                            format_signed_number(chapter["lens_profile"][lens]["signed_density"]),
                            chapter["lens_profile"][lens]["intensity_density"],
                            f"{chapter['lens_profile'][lens]['chapter_rank']} ({format_ordinal(chapter['lens_profile'][lens]['chapter_percentile'])})",
                        )
                        for lens in SCORING_LENS_ORDER
                    ],
                ),
                "",
                markdown_table(
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
                markdown_table(
                    ["Passage", "Impact Mass", "Dominant Character", "Lens Signature", "Summary"],
                    [
                        (
                            f"{row['unit_id']} ({row['paragraph_start']}-{row['paragraph_end']})",
                            row["impact_mass"],
                            row["dominant_character"] or "none",
                            ", ".join(
                                f"{lens} {row['lens_signature'][lens]}"
                                for lens in SCORING_LENS_ORDER
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


def render_coverage_audit_markdown(audit):
    metadata = audit["metadata"]
    corpus_totals = audit["summary"]["corpus_totals"]

    narrator_distribution = defaultdict(int)
    for unit in audit["units"]:
        narrator_distribution[unit["narrator_first_person_count"]] += 1

    lines = [
        "# Coverage Audit",
        "",
        f"- Audit version: `{audit['coverage_audit_version']}`",
        f"- Duplicate resolution: `{metadata['duplicate_resolution']}`",
        f"- Unit count: `{metadata['unit_count']}`",
        f"- Flagged unit count: `{metadata['flagged_unit_count']}`",
        f"- Candidate addition count: `{metadata['candidate_addition_count']}`",
        f"- Narrator candidate unit count: `{metadata['narrator_candidate_unit_count']}`",
        f"- Narrator min first-person threshold: `{metadata['params']['narrator_min_first_person']}`",
        f"- Projected new matches (without narrator): `{corpus_totals['projected_new_matches_without_narrator']}`",
        f"- Projected new matches (with narrator): `{corpus_totals['projected_new_matches_with_narrator']}`",
        "",
        "## Top 20 Characters By Projected Match Gain",
        "",
        markdown_table(
            ["Character", "Flagged Units", "Projected Gain (No Narrator)", "Projected Gain (With Narrator)"],
            [
                (
                    row["character"],
                    row["flagged_unit_count"],
                    row["projected_new_matches_without_narrator"],
                    row["projected_new_matches_with_narrator"],
                )
                for row in audit["summary"]["characters"][:20]
            ],
        ),
        "",
        "## Top 20 Flagged Units By Projected Match Gain",
        "",
        markdown_table(
            ["Unit", "Chapter", "Scored", "Candidates", "Narrator Candidate", "Projected Gain (With Narrator)"],
            [
                (
                    row["unit_id"],
                    row["chapter_id"],
                    len(row["scored_characters"]),
                    ", ".join(candidate["character"] for candidate in row["candidate_additions"]),
                    "yes" if row["narrator_candidate"] else "no",
                    row["projected_new_matches_with_narrator"],
                )
                for row in sorted(
                    audit["units"],
                    key=lambda row: (-row["projected_new_matches_with_narrator"], row["unit_id"]),
                )[:20]
            ],
        ),
        "",
        "## Narrator Summary",
        "",
        f"- Units flagged as narrator candidates: `{metadata['narrator_candidate_unit_count']}`",
        "",
        markdown_table(
            ["First-Person Marker Count", "Unit Count"],
            [(count, narrator_distribution[count]) for count in sorted(narrator_distribution)],
        ),
        "",
        "## Counts By Chapter",
        "",
        markdown_table(
            ["Chapter", "Units", "Flagged Units", "Narrator Candidate Units", "Candidate Additions"],
            [
                (
                    row["chapter_id"],
                    row["unit_count"],
                    row["flagged_unit_count"],
                    row["narrator_candidate_unit_count"],
                    row["candidate_addition_count"],
                )
                for row in audit["summary"]["chapters"]
            ],
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_coverage_audit_artifacts(audit, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_coverage_audit_markdown(audit))


def render_character_elo_supplement_diff_markdown(diff):
    lines = [
        "# Character ELO Supplement Diff",
        "",
        f"- Analysis version: `{diff['character_elo_supplement_diff_version']}`",
        f"- Clearing threshold: `{diff['clearing_threshold']}`",
        f"- Match count: `{diff['match_count']['before']}` -> `{diff['match_count']['after']}`",
        f"- Character count: `{diff['character_count']['before']}` -> `{diff['character_count']['after']}`",
        f"- Draw rate: `{diff['draw_rate']['before']}` -> `{diff['draw_rate']['after']}`",
        f"- Characters newly clearing threshold: `{diff['newly_clearing_threshold_count']}`",
        "",
        "## Characters Newly Clearing Threshold",
        "",
        markdown_table(
            ["Character", "Matches Before", "Matches After"],
            [
                (row["character"], row["match_count_before"], row["match_count_after"])
                for row in diff["characters_newly_clearing_threshold"]
            ],
        ),
        "",
        "## Top Rating Movers",
        "",
        markdown_table(
            ["Character", "ELO Before", "ELO After", "Delta", "Matches Before", "Matches After"],
            [
                (
                    row["character"],
                    row["elo_before"] if row["elo_before"] is not None else "new",
                    row["elo_after"],
                    format_signed_number(row["delta"]) if row["delta"] is not None else "new",
                    row["match_count_before"],
                    row["match_count_after"],
                )
                for row in diff["top_rating_movers"]
            ],
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_character_elo_supplement_diff_artifacts(diff, json_output=None, markdown_output=None):
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n")
    if markdown_output:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_character_elo_supplement_diff_markdown(diff))
