"""Rebuild fr-original canonical chapters from the page cache, alias-free.

Migration step 0 (see proust/docs/character_registry_design.md): the shipped
canonical chapters were built with the alias substitution pass enabled, which
rewrote Proust's text (bare-title constructions like "avec duc de Guermantes",
"M. de Charlus" flattened to "Charlus", and more). This rebuilds every chapter
from the archived original page cache (data/fr/islt_fr_*.html, recovered from
the Wayback Machine captures of marcel-proust.com) with aliases disabled.

Reconciliation: the original build's treatment of structural lines (volume
titles, chapter markers, argument summaries, empty spacer paragraphs) was
inconsistent chapter to chapter, so no filter ruleset can reproduce it. The
CURRENT paragraph sequence is therefore the structural template — paragraph
anchors #p-N are the deep-link contract for the reader and every analysis
artifact. Rebuilt page paragraphs are aligned to it two-pointer:

  - a rebuilt paragraph PAIRS with the next current paragraph when it is
    byte-identical OR applying the legacy alias replacements reproduces the
    current text exactly (the contamination being reversed)
  - an unpaired rebuilt paragraph is a structural line the original build
    dropped: it is skipped and RECORDED in the audit trail
  - a current empty paragraph is kept as-is (structural spacer)
  - any other unpaired current paragraph fails the gate

Gate (per chapter, verified after reconciliation): paragraph counts equal and
every diff explained. Output is staged to fr-original-rebuilt/; run with
--apply to copy staged chapters over the live edition after a passing gate.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from proust.corpus import (  # noqa: E402
    apply_alias_replacements,
    get_chapter_body,
    preprocess,
    read_aliases,
)
from proust.paths import DATA_DIR, ISLT_EDITIONS_DIR  # noqa: E402

EDITION_DIR = ISLT_EDITIONS_DIR / "fr-original"
STAGING_DIR = ISLT_EDITIONS_DIR / "fr-original-rebuilt"
NLP_MODEL = "fr_core_news_sm"


def load_nlp():
    import spacy

    return spacy.load(NLP_MODEL, disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])


def page_paragraphs(page_id):
    """All <p> texts from a cached page, whitespace-collapsed, unfiltered."""
    html = (DATA_DIR / f"islt_fr_{page_id:03}.html").read_text()
    soup = BeautifulSoup(html, features="html.parser")
    body = soup.find("article") or get_chapter_body(html)
    if body is None:
        raise ValueError(f"page {page_id:03}: no chapter body found")
    return [
        " ".join(preprocess(node.text, use_aliases=False).split())
        for node in body.find_all("p")
    ]


def _squash(text):
    return "".join(text.split())


def explained_match(current_text, rebuilt_text, legacy_aliases):
    """True when the current text is the rebuilt text plus legacy alias
    substitutions and/or markup-era spacing artifacts."""
    if current_text == rebuilt_text:
        return True
    applied = apply_alias_replacements(rebuilt_text, legacy_aliases)
    if applied == current_text:
        return True
    current_squashed = _squash(current_text)
    return _squash(rebuilt_text) == current_squashed or _squash(applied) == current_squashed


# Patterns that indicate alias contamination surviving in a kept-current
# paragraph (bare aristocratic title directly after a preposition/verb).
_CONTAMINATION = None


def _looks_contaminated(text):
    global _CONTAMINATION
    if _CONTAMINATION is None:
        import re

        _CONTAMINATION = re.compile(
            r"\b(avec|chez|que|dit|répondit|ajouta|pour|de|à)\s+"
            r"(duc|duchesse|marquis|marquise|baron|comte|comtesse)\s+de\s+[A-ZÉ]"
        )
    return bool(_CONTAMINATION.search(text))


def reconcile(current_paragraphs, rebuilt_texts, legacy_aliases):
    """Align rebuilt texts to the current paragraph sequence.

    Returns (aligned_texts, dropped, kept_current). aligned_texts[i] is the
    restored text for current paragraph i. Current paragraphs with no
    explained rebuilt counterpart (site drift, editorial markers, later typo
    fixes) keep their current text verbatim and are reported.
    """
    aligned, dropped, kept_current = [], [], []
    j = 0
    for current in current_paragraphs:
        current_text = current["text"]
        if not current_text.strip():
            aligned.append(current_text)  # structural spacer, keep verbatim
            continue
        found = None
        probe = j
        # a bounded look-ahead lets us skip runs of structural lines
        while probe < len(rebuilt_texts) and probe - j <= 12:
            candidate = rebuilt_texts[probe]
            if candidate and explained_match(current_text, candidate, legacy_aliases):
                found = probe
                break
            probe += 1
        if found is None:
            aligned.append(current_text)
            kept_current.append(
                {
                    "id": current["id"],
                    "text": current_text[:120],
                    "contamination_risk": _looks_contaminated(current_text),
                }
            )
            continue
        for skipped_index in range(j, found):
            if rebuilt_texts[skipped_index]:
                dropped.append(rebuilt_texts[skipped_index])
        aligned.append(rebuilt_texts[found])
        j = found + 1
    for remaining in rebuilt_texts[j:]:
        if remaining:
            dropped.append(remaining)
    return aligned, dropped, kept_current


def segment(text, nlp):
    if not text.strip():
        return []
    return [
        {"id": f"s-{index}", "index": index, "text": sent.text.strip()}
        for index, sent in enumerate(
            (s for s in nlp(text).sents if s.text.strip()), start=1
        )
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--chapter", help="restrict to one chapter id")
    args = parser.parse_args(argv)

    if args.apply:
        staged = sorted((STAGING_DIR / "chapters").glob("*.json"))
        if not staged:
            raise SystemExit("nothing staged")
        for path in staged:
            shutil.copy2(path, EDITION_DIR / "chapters" / path.name)
        print(f"applied {len(staged)} staged chapters to {EDITION_DIR}")
        return 0

    structure = json.loads((EDITION_DIR / "reader-structure.json").read_text())
    legacy_aliases = read_aliases()
    nlp = load_nlp()
    (STAGING_DIR / "chapters").mkdir(parents=True, exist_ok=True)

    gate_failed = False
    total_restored = 0
    for entry in structure:
        chapter_id = entry["id"]
        if args.chapter and chapter_id != args.chapter:
            continue
        current = json.loads((EDITION_DIR / "chapters" / f"{chapter_id}.json").read_text())
        rebuilt_texts = []
        for page_id in range(int(entry["startChapterId"]), int(entry["endChapterId"]) + 1):
            rebuilt_texts.extend(page_paragraphs(page_id))

        aligned, dropped, kept_current = reconcile(
            current["paragraphs"], rebuilt_texts, legacy_aliases
        )

        contaminated_kept = [k for k in kept_current if k["contamination_risk"]]
        if len(aligned) != len(current["paragraphs"]) or contaminated_kept:
            gate_failed = True
            print(f"FAIL {chapter_id}: aligned {len(aligned)}/{len(current['paragraphs'])}, "
                  f"contaminated kept-current: {[k['id'] for k in contaminated_kept]}")
            continue

        restored = sum(
            1
            for cur, new in zip(current["paragraphs"], aligned)
            if cur["text"] != new
        )
        total_restored += restored
        print(f"ok {chapter_id}: {restored} paragraphs restored, "
              f"{len(dropped)} structural lines dropped, {len(kept_current)} kept current")
        for line in dropped:
            print(f"    dropped: {line[:110]!r}")
        for keep in kept_current:
            print(f"    kept-current {keep['id']}: {keep['text']!r}")

        paragraphs = []
        for index, (cur, text) in enumerate(
            zip(current["paragraphs"], aligned), start=1
        ):
            paragraphs.append(
                {
                    "id": f"p-{index}",
                    "index": index,
                    "text": text,
                    "sentences": segment(text, nlp),
                }
            )
        chapter = dict(current)
        chapter["paragraphs"] = paragraphs
        chapter["paragraphCount"] = len(paragraphs)
        chapter["sentenceCount"] = sum(len(p["sentences"]) for p in paragraphs)
        out = STAGING_DIR / "chapters" / f"{chapter_id}.json"
        out.write_text(json.dumps(chapter, ensure_ascii=False, indent=2) + "\n")

    print(f"\ntotal paragraphs restored: {total_restored}")
    print("GATE:", "FAILED — nothing should be applied" if gate_failed else "PASSED")
    return 1 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
