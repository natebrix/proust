def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_signed_number(value):
    if isinstance(value, float):
        value = round(value, 3)
    if isinstance(value, (int, float)) and value > 0:
        return f"+{value}"
    return str(value)


def format_ordinal(value):
    if value is None:
        return ""
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def character_ranks(character_totals, reverse=True):
    ordered = sorted(
        character_totals,
        key=lambda item: ((-item["net_score"]) if reverse else item["net_score"], item["character"]),
    )
    return {row["character"]: index + 1 for index, row in enumerate(ordered)}


def character_totals_by_name(character_totals):
    return {row["character"]: row for row in character_totals}


__all__ = [
    "character_ranks",
    "character_totals_by_name",
    "format_ordinal",
    "format_signed_number",
    "markdown_table",
]
