"""The character-page editorial must agree with the promoted standings."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location("check_editorial_claims", ROOT / "scripts" / "check_editorial_claims.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_rank_claim_in_the_editorial_matches_the_promoted_standings():
    checker = _checker()
    checked, problems = checker.check(
        checker.CHARACTER_PAGE_PILOT_EDITORIAL, checker.load_standings(ROOT / "outputs")
    )
    assert checked > 100
    assert problems == []


def test_the_checker_catches_a_wrong_rank_and_a_wrong_unranked_claim():
    checker = _checker()
    standings = {
        "advantage": {"size": 31, "rank": {"X": 5}},
        "prestige": {"size": 14, "rank": {}},
        "inclusion": {"size": 8, "rank": {}},
    }
    entry = {
        "subheading": "X is 6th of 31 in scene-level advantage.",
        "summary": "Prestige is 2nd of 14. Advantage is unranked.",
        "why_interesting": ["Ranked in all three registers."],
    }
    _checked, problems = checker.check({"X": entry}, standings)
    assert len(problems) == 4
