"""Check the character-page editorial's rank claims against the promoted standings.

Usage:
    python3 scripts/check_editorial_claims.py [--outputs-dir outputs]

Every dossier in `proust.editorial.CHARACTER_PAGE_PILOT_EDITORIAL` is read
sentence by sentence and checked against the promoted name-view standings
(`character-standings-{lens}-current.json`):

- "Nth of M" and "first/last of M": N must be the character's rank and M
  the size of the ranked set, in the lens named nearest the claim
- "unranked" and "too ... to rank": the character must be unranked in that
  lens
- "ranked in all three registers" / "one of the eight": the character must be
  ranked in every lens
- spelled-out ranked-set sizes are flagged: write them as digits so this
  check can read them

A lens is named by its word or its register: advantage / scene(s) /
scene-level, prestige / standing, inclusion / belonging. Claims about the
past are written without "of M" ("3rd-of-8", "from 7th to 20th") and are not
checked. Exits non-zero when any claim fails.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust.editorial import CHARACTER_PAGE_PILOT_EDITORIAL  # noqa: E402

LENSES = ("advantage", "prestige", "inclusion")
LENS_WORDS = {
    "advantage": r"advantage|scene-level|scenes?",
    "prestige": r"prestige|standing",
    "inclusion": r"inclusion|belonging",
}
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
RANK_CLAIM = re.compile(
    r"\b(?:(\d+)(?:st|nd|rd|th)|(" + "|".join(ORDINAL_WORDS) + r"|last))\s+of\s+(?:the\s+)?(\d+)\b",
    re.IGNORECASE,
)
UNRANKED_CLAIM = re.compile(r"\bunranked\b|\btoo\s+(?:\w+\s+){0,3}to\s+rank\b|\bno longer supports a rank\b", re.IGNORECASE)
ALL_THREE_CLAIM = re.compile(r"ranked in (?:all three registers|every register)|one of the eight", re.IGNORECASE)
SPELLED_TOTAL = re.compile(r"\b(?:twenty-two|forty-one|thirty-one|fourteen)\b", re.IGNORECASE)


def load_standings(outputs_dir):
    standings = {}
    for lens in LENSES:
        data = json.loads((Path(outputs_dir) / f"character-standings-{lens}-current.json").read_text())
        standings[lens] = {
            "size": data["ranked_count"],
            "rank": {row["character"]: row["rank"] for row in data["ranked"]},
        }
    return standings


def sentences(text):
    return [part for part in re.split(r"(?<=[.;!?])\s+", text) if part.strip()]


def _lens_matches(text):
    for lens, words in LENS_WORDS.items():
        for match in re.finditer(r"\b(?:" + words + r")\b", text, re.IGNORECASE):
            yield match.start(), lens


def nearest_lens(sentence, start, end):
    """The lens a claim at sentence[start:end] is about.

    A lens word right after the claim wins ("4th of 14 in prestige"), then
    the closest one before it ("Prestige: 6th of 14", "advantage (10th of
    31)"), then the closest one after it anywhere in the sentence.
    """
    after = sentence[end:end + 30]
    following = sorted(_lens_matches(after))
    if following and not re.search(r"[,;()]", after[: following[0][0]]):
        return following[0][1]
    before = [(position, lens) for position, lens in _lens_matches(sentence) if position < start]
    if before:
        return max(before)[1]
    later = [(position, lens) for position, lens in _lens_matches(sentence) if position >= end]
    return min(later)[1] if later else None


def texts(entry):
    yield "subheading", entry["subheading"]
    yield "summary", entry["summary"]
    for index, item in enumerate(entry["why_interesting"]):
        yield f"why_interesting[{index}]", item


def check(editorial, standings):
    problems = []
    checked = 0
    for character, entry in editorial.items():
        ranked_everywhere = all(character in standings[lens]["rank"] for lens in LENSES)
        for field, text in texts(entry):
            for sentence in sentences(text):
                for match in RANK_CLAIM.finditer(sentence):
                    checked += 1
                    lens = nearest_lens(sentence, match.start(), match.end())
                    size = int(match.group(3))
                    if lens is None:
                        problems.append(f"{character} / {field}: no lens named near {match.group(0)!r}")
                        continue
                    actual_rank = standings[lens]["rank"].get(character)
                    actual_size = standings[lens]["size"]
                    word = (match.group(2) or "").lower()
                    claimed = actual_size if word == "last" else ORDINAL_WORDS.get(word) or int(match.group(1))
                    if actual_rank is None:
                        problems.append(
                            f"{character} / {field}: claims {match.group(0)!r} in {lens}, but is unranked there"
                        )
                    elif (claimed, size) != (actual_rank, actual_size):
                        problems.append(
                            f"{character} / {field}: claims {match.group(0)!r} in {lens}; "
                            f"standings say {actual_rank} of {actual_size}"
                        )
                for match in UNRANKED_CLAIM.finditer(sentence):
                    checked += 1
                    lens = nearest_lens(sentence, match.start(), match.end())
                    if lens and character in standings[lens]["rank"]:
                        problems.append(
                            f"{character} / {field}: says {match.group(0)!r} for {lens}, "
                            f"but is ranked {standings[lens]['rank'][character]} there"
                        )
                if ALL_THREE_CLAIM.search(sentence):
                    checked += 1
                    if not ranked_everywhere:
                        problems.append(f"{character} / {field}: claims a rank in all three registers")
                for match in SPELLED_TOTAL.finditer(sentence):
                    problems.append(f"{character} / {field}: spelled-out total {match.group(0)!r}; use digits")
    return checked, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    arguments = parser.parse_args()
    checked, problems = check(CHARACTER_PAGE_PILOT_EDITORIAL, load_standings(arguments.outputs_dir))
    for problem in problems:
        print(problem)
    print(f"{checked} claims checked, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
