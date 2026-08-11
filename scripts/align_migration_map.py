"""Align the legacy fr-original paragraphs to the staged Wikisource chapters.

Phase B of proust/docs/source_migration_plan.md. The two transcriptions are not
the same text: typography differs throughout (straight vs curly apostrophes,
ellipses, dashes), the legacy fan-site transcription paragraphs differently in
places, carries its own typos, and the posthumous volumes carry audiobook
collation artifacts. So the alignment is by SIMILARITY, not equality.

Per chapter this produces an ordered list of pairings between old paragraph ids
and new paragraph ids:

  - ``one_to_one``  old p-N  -> new p-M
  - ``split``       old p-N  -> new p-M..p-M+k (the legacy text ran them together)
  - ``merge``       old p-N..p-N+k -> new p-M   (the legacy text broke them apart)
  - ``old_only``    a legacy paragraph with no counterpart; annotated when it is
                    a known legacy artifact or corresponds to Wikisource
                    apparatus we deliberately stripped (headings, arguments,
                    section-break glyphs)
  - ``new_only``    text Wikisource has that the legacy source lacked

Outputs outputs/source-migration-map.json (the bridge phase D remaps through)
and outputs/source-migration-divergence.md (the review surface). Gates are
reported and enforced with the exit status; nothing is applied here.
"""
import argparse
import difflib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proust.paths import ISLT_EDITIONS_DIR  # noqa: E402

OLD_EDITION_DIR = ISLT_EDITIONS_DIR / "fr-original"
NEW_EDITION_DIR = ISLT_EDITIONS_DIR / "fr-original-ws"
MAPPING_PATH = REPO_ROOT / "outputs" / "wikisource-mapping.json"
MAP_PATH = REPO_ROOT / "outputs" / "source-migration-map.json"
REPORT_PATH = REPO_ROOT / "outputs" / "source-migration-divergence.md"

PAIR_THRESHOLD = 0.60
REVIEW_THRESHOLD = 0.75
EXCERPT_THRESHOLD = 0.99  # pairings whose texts are quoted in the map
WEAKEST_LISTED = 5
NEW_ONLY_MAX_RATE = 0.02
MIN_MEDIAN_SIMILARITY = 0.95

CHAR_VARIANTS = {
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...",
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
}

# Legacy-side paragraphs that are known artifacts of the fan-site transcription
# rather than Proust's text. Expected to have no Wikisource counterpart.
LEGACY_ARTIFACTS = (
    ("site_navigation", re.compile(r"Retour à la première page")),
    ("colophon", re.compile(r"^FIN du roman", re.IGNORECASE)),
    ("editorial_marker", re.compile(r"\[-*\s*Ajout Gallimard")),
    ("editorial_marker", re.compile(r"\[L['’]édition sonore Thélème")),
    ("editorial_marker", re.compile(r"^\s*\[-{2,}")),
)
# The legacy transcription's own section-break lines, the counterpart of the ⁂
# glyphs stripped from the Wikisource side.
SECTION_BREAK_CHARS = set("*-–—⁂·. ")


def normalize(text):
    """Comparison form: NFC, unified typography, lowercased, whitespace collapsed."""
    folded = unicodedata.normalize("NFC", text)
    folded = "".join(CHAR_VARIANTS.get(char, char) for char in folded)
    return " ".join(folded.lower().split())


def tokens(text):
    return re.findall(r"[\w']+", normalize(text))


def similarity(left, right):
    """Token-level similarity of two paragraph texts, in [0, 1]."""
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return difflib.SequenceMatcher(None, left_tokens, right_tokens).ratio()


def word_divergence(left, right):
    """Fraction of word tokens that differ between two paragraph texts."""
    left_tokens, right_tokens = tokens(left), tokens(right)
    total = max(len(left_tokens), len(right_tokens))
    if not total:
        return 0.0
    matcher = difflib.SequenceMatcher(None, left_tokens, right_tokens)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (total - matched) / total


def annotate_old_only(text, apparatus_keys):
    """Why a legacy paragraph has no counterpart, or None if unexplained."""
    stripped = text.strip()
    if not stripped:
        return "structural_spacer"
    if set(stripped) <= SECTION_BREAK_CHARS:
        return "section_break"
    for label, pattern in LEGACY_ARTIFACTS:
        if pattern.search(text):
            return label
    key = normalize(text).strip(" .")
    if key and any(key in apparatus for apparatus in apparatus_keys):
        return "wikisource_apparatus"
    return None


def propagate_block_annotations(pairings):
    """Spread an artifact annotation over the run of paragraphs it governs.

    The legacy edition's collation artifacts are not single paragraphs: an
    ``[----Ajout Gallimard----]`` marker introduces a stretch of text taken
    from a later Gallimard edition, and the closing colophon runs over several
    lines. Where an unmatched run contains a recognised artifact, the whole run
    is that artifact — recorded as ``<label>_block`` so review can see the
    extent rather than a lone marker.
    """
    index = 0
    while index < len(pairings):
        if pairings[index]["kind"] != "old_only":
            index += 1
            continue
        end = index
        while end < len(pairings) and pairings[end]["kind"] == "old_only":
            end += 1
        run = pairings[index:end]
        labels = [
            pairing["annotation"]
            for pairing in run
            if pairing.get("annotation") in {"editorial_marker", "colophon", "site_navigation"}
        ]
        if labels:
            for pairing in run:
                if not pairing.get("annotation"):
                    pairing["annotation"] = f"{labels[0]}_block"
            for neighbour in (index - 1, end):
                if 0 <= neighbour < len(pairings) and pairings[neighbour]["kind"] == "new_only":
                    pairings[neighbour]["annotation"] = f"{labels[0]}_block_counterpart"
        index = end
    return pairings


# --- alignment -------------------------------------------------------------


def anchor_blocks(old_texts, new_texts):
    """Equal runs of the two paragraph sequences, matched on a coarse key.

    SequenceMatcher over whole paragraphs needs hashable, comparable items, and
    the two transcriptions are never byte-equal. The coarse key (first and last
    words plus rounded length) is equal for the overwhelming majority of
    corresponding paragraphs, which gives the alignment its skeleton; the gaps
    between anchors are then resolved by similarity.
    """
    def key(text):
        words = tokens(text)
        if not words:
            return ("", "", 0)
        return (" ".join(words[:6]), " ".join(words[-4:]), len(words) // 8)

    old_keys = [key(text) for text in old_texts]
    new_keys = [key(text) for text in new_texts]
    matcher = difflib.SequenceMatcher(None, old_keys, new_keys, autojunk=False)
    return matcher.get_matching_blocks()


MAX_SPAN = 4


def best_pairing(old_slice, new_slice, i, j, max_old_span=MAX_SPAN, max_new_span=MAX_SPAN):
    """Best (score, old span, new span) for a pairing starting at (i, j).

    The span caps let a caller ask the narrower question "does this paragraph
    pair here on its own", which is what keeps a merge from quietly absorbing
    an interpolated paragraph and scoring well in spite of it.
    """
    if i >= len(old_slice) or j >= len(new_slice):
        return (0.0, 1, 1)
    candidates = [(similarity(old_slice[i], new_slice[j]), 1, 1)]
    for span in range(2, MAX_SPAN + 1):
        if span <= max_new_span and j + span <= len(new_slice):
            candidates.append(
                (similarity(old_slice[i], " ".join(new_slice[j:j + span])), 1, span)
            )
        if span <= max_old_span and i + span <= len(old_slice):
            candidates.append(
                (similarity(" ".join(old_slice[i:i + span]), new_slice[j]), span, 1)
            )
    return max(candidates)


def pair_gap(old_slice, new_slice, old_offset, new_offset):
    """Pairings for a stretch where the coarse anchors disagree.

    Greedy two-pointer over the gap: at each step take the best of a 1:1
    pairing, a split (one old paragraph covering several new ones) and a merge
    (several old covering one new) — unless dropping the paragraph under one of
    the pointers lets the other side pair better on its own, which is what
    happens at an interpolated legacy paragraph (an editorial marker, a chapter
    heading) and at Wikisource text the legacy source never had.
    """
    pairings = []
    i = j = 0
    while i < len(old_slice) and j < len(new_slice):
        score, old_span, new_span = best_pairing(old_slice, new_slice, i, j)
        skip_old = best_pairing(old_slice, new_slice, i + 1, j, max_new_span=1)[0]
        skip_new = best_pairing(old_slice, new_slice, i, j + 1, max_old_span=1)[0]
        if score >= PAIR_THRESHOLD and score >= skip_old and score >= skip_new:
            pairings.append(
                (
                    [old_offset + i + k for k in range(old_span)],
                    [new_offset + j + k for k in range(new_span)],
                    score,
                )
            )
            i += old_span
            j += new_span
        elif skip_old >= skip_new:
            pairings.append(([old_offset + i], [], 0.0))
            i += 1
        else:
            pairings.append(([], [new_offset + j], 0.0))
            j += 1
    for k in range(i, len(old_slice)):
        pairings.append(([old_offset + k], [], 0.0))
    for k in range(j, len(new_slice)):
        pairings.append(([], [new_offset + k], 0.0))
    return pairings


def align(old_texts, new_texts):
    """Ordered (old indices, new indices, similarity) triples for one chapter."""
    pairings = []
    old_cursor = new_cursor = 0
    for block in anchor_blocks(old_texts, new_texts):
        if block.a > old_cursor or block.b > new_cursor:
            pairings.extend(
                pair_gap(
                    old_texts[old_cursor:block.a],
                    new_texts[new_cursor:block.b],
                    old_cursor,
                    new_cursor,
                )
            )
        for offset in range(block.size):
            old_index, new_index = block.a + offset, block.b + offset
            pairings.append(
                (
                    [old_index],
                    [new_index],
                    similarity(old_texts[old_index], new_texts[new_index]),
                )
            )
        old_cursor, new_cursor = block.a + block.size, block.b + block.size
    return pairings


def pairing_kind(old_indices, new_indices):
    if old_indices and new_indices:
        if len(old_indices) == 1 and len(new_indices) == 1:
            return "one_to_one"
        return "split" if len(old_indices) == 1 else "merge"
    return "old_only" if old_indices else "new_only"


# --- reporting -------------------------------------------------------------


def excerpt(text, length=100):
    text = " ".join(text.split())
    return text[:length] + ("…" if len(text) > length else "")


def chapter_record(chapter_id, old_chapter, new_chapter, apparatus):
    old_texts = [p["text"] for p in old_chapter["paragraphs"]]
    new_texts = [p["text"] for p in new_chapter["paragraphs"]]
    apparatus_keys = [normalize(text) for text in apparatus]

    pairings = []
    for old_indices, new_indices, score in align(old_texts, new_texts):
        kind = pairing_kind(old_indices, new_indices)
        record = {
            "kind": kind,
            "old_ids": [f"p-{index + 1}" for index in old_indices],
            "new_ids": [f"p-{index + 1}" for index in new_indices],
            "similarity": round(score, 4),
        }
        if kind == "old_only":
            record["annotation"] = annotate_old_only(
                old_texts[old_indices[0]], apparatus_keys
            )
            record["text"] = excerpt(old_texts[old_indices[0]])
        elif kind == "new_only":
            record["text"] = excerpt(new_texts[new_indices[0]])
        elif score < EXCERPT_THRESHOLD:
            record["old_text"] = excerpt(" ".join(old_texts[i] for i in old_indices))
            record["new_text"] = excerpt(" ".join(new_texts[i] for i in new_indices))
        pairings.append(record)
    propagate_block_annotations(pairings)

    kinds = {kind: 0 for kind in ("one_to_one", "split", "merge", "old_only", "new_only")}
    for pairing in pairings:
        kinds[pairing["kind"]] += 1
    one_to_one = [p for p in pairings if p["kind"] == "one_to_one"]
    scores = sorted(p["similarity"] for p in one_to_one)
    divergences = [
        word_divergence(old_texts[int(p["old_ids"][0][2:]) - 1], new_texts[int(p["new_ids"][0][2:]) - 1])
        for p in one_to_one
    ]
    unexplained = [
        p for p in pairings if p["kind"] == "old_only" and not p.get("annotation")
    ]
    new_only = [p for p in pairings if p["kind"] == "new_only"]
    stats = {
        "oldParagraphCount": len(old_texts),
        "newParagraphCount": len(new_texts),
        "kinds": kinds,
        "medianSimilarity": round(statistics.median(scores), 4) if scores else 0.0,
        "p10Similarity": round(percentile(scores, 0.10), 4) if scores else 0.0,
        "minSimilarity": round(min(scores), 4) if scores else 0.0,
        "meanWordDivergence": round(statistics.fmean(divergences), 4) if divergences else 0.0,
        "unexplainedOldOnly": len(unexplained),
        "newOnly": len(new_only),
        "newOnlyRate": round(len(new_only) / len(new_texts), 4) if new_texts else 0.0,
        "reviewPairings": sum(
            1
            for p in pairings
            if p["kind"] in ("one_to_one", "split", "merge")
            and p["similarity"] < REVIEW_THRESHOLD
        ),
    }
    return {"chapterId": chapter_id, "stats": stats, "pairings": pairings}


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def gate_failures(record):
    stats = record["stats"]
    failures = []
    if stats["unexplainedOldOnly"]:
        failures.append(f"{stats['unexplainedOldOnly']} unexplained old_only paragraphs")
    if stats["newOnlyRate"] > NEW_ONLY_MAX_RATE:
        failures.append(
            f"new_only rate {stats['newOnlyRate']:.2%} > {NEW_ONLY_MAX_RATE:.0%}"
        )
    if stats["medianSimilarity"] < MIN_MEDIAN_SIMILARITY:
        failures.append(
            f"median 1:1 similarity {stats['medianSimilarity']:.3f} < {MIN_MEDIAN_SIMILARITY}"
        )
    return failures


def write_report(records, path):
    lines = [
        "# Source migration divergence report",
        "",
        "Legacy `fr-original` (marcel-proust.com transcription) aligned to the",
        "staged `fr-original-ws` chapters built from the pinned French Wikisource",
        "revisions (`outputs/wikisource-mapping.json`, `data/wikisource/manifest.json`).",
        "Similarity is a token-level `difflib` ratio on typography-normalized text",
        "(NFC, unified apostrophes/quotes/dashes/ellipses, lowercased); it is not an",
        "equality test, because the two transcriptions genuinely differ.",
        "",
        f"Gates: every old paragraph paired or annotated; new_only < {NEW_ONLY_MAX_RATE:.0%}"
        f" per chapter; median 1:1 similarity >= {MIN_MEDIAN_SIMILARITY}.",
        "",
        "## Summary",
        "",
        "| chapter | old | new | 1:1 | split | merge | old_only | new_only | median | p10 | min | word div | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        stats = record["stats"]
        failures = gate_failures(record)
        lines.append(
            "| {id} | {old} | {new} | {one} | {split} | {merge} | {old_only} | {new_only} "
            "| {median:.3f} | {p10:.3f} | {minimum:.3f} | {div:.2%} | {gate} |".format(
                id=record["chapterId"],
                old=stats["oldParagraphCount"],
                new=stats["newParagraphCount"],
                one=stats["kinds"]["one_to_one"],
                split=stats["kinds"]["split"],
                merge=stats["kinds"]["merge"],
                old_only=stats["kinds"]["old_only"],
                new_only=stats["kinds"]["new_only"],
                median=stats["medianSimilarity"],
                p10=stats["p10Similarity"],
                minimum=stats["minSimilarity"],
                div=stats["meanWordDivergence"],
                gate="PASS" if not failures else "FAIL: " + "; ".join(failures),
            )
        )

    lines += ["", "## Per chapter", ""]
    for record in records:
        stats = record["stats"]
        lines += [
            f"### {record['chapterId']}",
            "",
            f"- paragraphs: old {stats['oldParagraphCount']}, new {stats['newParagraphCount']}",
            "- pairings: "
            + ", ".join(f"{kind} {count}" for kind, count in stats["kinds"].items()),
            f"- 1:1 similarity: median {stats['medianSimilarity']:.4f}, "
            f"p10 {stats['p10Similarity']:.4f}, min {stats['minSimilarity']:.4f}",
            f"- word-level divergence on 1:1 pairs: {stats['meanWordDivergence']:.2%}",
            "",
        ]
        old_only = [p for p in record["pairings"] if p["kind"] == "old_only"]
        if old_only:
            lines.append(f"**old_only ({len(old_only)})** — legacy paragraphs with no counterpart:")
            lines.append("")
            for pairing in old_only:
                annotation = pairing.get("annotation") or "UNEXPLAINED"
                lines.append(
                    f"- `{pairing['old_ids'][0]}` [{annotation}] {pairing['text']!r}"
                )
            lines.append("")
        new_only = [p for p in record["pairings"] if p["kind"] == "new_only"]
        if new_only:
            lines.append(
                f"**new_only ({len(new_only)})** — Wikisource text the legacy source lacked:"
            )
            lines.append("")
            for pairing in new_only:
                annotation = pairing.get("annotation")
                label = f" [{annotation}]" if annotation else ""
                lines.append(f"- `{pairing['new_ids'][0]}`{label} {pairing['text']!r}")
            lines.append("")
        paired = [
            p
            for p in record["pairings"]
            if p["kind"] in ("one_to_one", "split", "merge") and "old_text" in p
        ]
        weak = [p for p in paired if p["similarity"] < REVIEW_THRESHOLD]
        if weak:
            heading = f"**pairings below {REVIEW_THRESHOLD} ({len(weak)})** — review:"
        else:
            weak = sorted(paired, key=lambda p: p["similarity"])[:WEAKEST_LISTED]
            heading = (
                f"**weakest pairings ({len(weak)})** — none below {REVIEW_THRESHOLD}, "
                "these are the least similar:"
            )
        if weak:
            lines.append(heading)
            lines.append("")
            for pairing in weak:
                lines.append(
                    f"- {pairing['kind']} `{','.join(pairing['old_ids'])}` -> "
                    f"`{','.join(pairing['new_ids'])}` ({pairing['similarity']:.3f})"
                )
                lines.append(f"  - old: {pairing['old_text']!r}")
                lines.append(f"  - new: {pairing['new_text']!r}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", help="restrict to one canonical chapter id")
    args = parser.parse_args(argv)

    mapping = json.loads(MAPPING_PATH.read_text())
    apparatus_by_chapter = json.loads(
        (NEW_EDITION_DIR / "stripped-apparatus.json").read_text()
    )

    records = []
    for entry in mapping["chapters"]:
        chapter_id = entry["canonical_id"]
        if args.chapter and chapter_id != args.chapter:
            continue
        old_chapter = json.loads(
            (OLD_EDITION_DIR / "chapters" / f"{chapter_id}.json").read_text()
        )
        new_chapter = json.loads(
            (NEW_EDITION_DIR / "chapters" / f"{chapter_id}.json").read_text()
        )
        record = chapter_record(
            chapter_id, old_chapter, new_chapter, apparatus_by_chapter.get(chapter_id, [])
        )
        records.append(record)
        stats = record["stats"]
        failures = gate_failures(record)
        print(
            f"{chapter_id:52s} old={stats['oldParagraphCount']:4d} new={stats['newParagraphCount']:4d} "
            f"1:1={stats['kinds']['one_to_one']:4d} split={stats['kinds']['split']:3d} "
            f"merge={stats['kinds']['merge']:3d} old_only={stats['kinds']['old_only']:3d} "
            f"new_only={stats['kinds']['new_only']:3d} median={stats['medianSimilarity']:.3f} "
            + ("PASS" if not failures else "FAIL " + "; ".join(failures))
        )

    payload = {
        "generated_from": {
            "old_edition": "fr-original",
            "new_edition": "fr-original-ws",
            "mapping": str(MAPPING_PATH.relative_to(REPO_ROOT)),
        },
        "thresholds": {
            "pair": PAIR_THRESHOLD,
            "review": REVIEW_THRESHOLD,
            "newOnlyMaxRate": NEW_ONLY_MAX_RATE,
            "minMedianSimilarity": MIN_MEDIAN_SIMILARITY,
        },
        "chapters": records,
    }
    MAP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_report(records, REPORT_PATH)

    failed = [r["chapterId"] for r in records if gate_failures(r)]
    print(f"\nwrote {MAP_PATH.name} and {REPORT_PATH.name}")
    print("GATES:", "PASSED" if not failed else f"FAILED for {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
