"""Phase D of proust/docs/source_migration_plan.md: remap paragraph references.

The fr-original chapters were just replaced with the Wikisource text (the
canonical chapter JSONs under data/islt/editions/fr-original/chapters), which
renumbered paragraphs throughout. Every app-facing artifact still generated
against the old (fan-site) numbering, so this script rewrites those paragraph
references in place using outputs/source-migration-map.json, the old->new
bridge produced by scripts/align_migration_map.py.

This is a display-bridge patch, not a re-annotation: unit ids and unit_id
strings (e.g. "v1-p2-un-amour-de-swann#p-343-p-348") are legacy-span
identifiers and are left exactly as they are. Only the paragraph NUMBERS used
to point a reader at the text move: reader_link "#p-<n>" anchors, and the
paragraph_start/paragraph_end integers that go with them.

Lookup construction
--------------------
Per chapter, the migration map's ``pairings`` list is an ordered walk over
old paragraph 1..oldParagraphCount; each pairing carries the (possibly empty)
run of new paragraph ids it corresponds to:

  - one_to_one / split / merge: old id(s) -> new id(s); a span anchored on
    any of those old ids starts at min(new ids) and ends at max(new ids).
  - old_only: an old paragraph with no Wikisource counterpart (stripped
    apparatus, a legacy transcription artifact). It has no new id of its
    own, so it borrows one from a neighbouring mapped paragraph:
      * as a span START, the NEAREST FOLLOWING mapped paragraph's start
        (don't start a passage earlier than intended);
      * as a span END, the NEAREST PRECEDING mapped paragraph's end.
    If that preferred neighbour doesn't exist (an old_only run at the very
    edge of the chapter), we fall back to the other direction rather than
    leaving the reference unresolved.

This gives, per chapter, a dict old_paragraph_index -> {"start": new_index,
"end": new_index} covering every old paragraph 1..oldParagraphCount.

Fields NOT touched (see the plan and the migration task): unit_id / unit_id
strings anywhere; cumulative_unit_index and the cumulative_word_count /
cumulative_word_count_end fields in the WHR timelines, which are legacy-scale
bookkeeping used only to place points on a chart x-axis, not reader
pointers. cumulative_paragraph_index / cumulative_paragraph_index_end are the
same kind of cross-corpus running total and are likewise left alone -- only
the per-chapter paragraph_start/paragraph_end are remapped.

Usage: python scripts/remap_artifact_paragraphs.py
Exits 1 (after printing every reference it could not resolve) if any
reference is unresolved; 0 otherwise.
"""
import bisect
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"

MIGRATION_MAP_PATH = OUTPUTS_DIR / "source-migration-map.json"
CHARACTER_PAGES_JSON = OUTPUTS_DIR / "character-pages-supplemented-current.json"
CHARACTER_PAGES_MD = OUTPUTS_DIR / "character-pages-supplemented-current.md"
CHAPTER_SUMMARIES_JSON = OUTPUTS_DIR / "chapter-summaries-current.json"
WHR_TIMELINE_JSONS = {
    lens: OUTPUTS_DIR / f"character-whr-{lens}-timeline-supplemented-current.json"
    for lens in ("advantage", "prestige", "inclusion")
}

PARAGRAPH_ID_RE = re.compile(r"^p-(\d+)$")
READER_LINK_RE = re.compile(r"^(?P<prefix>.*/fr-original/(?P<chapter>[^/#]+))#p-(?P<num>\d+)$")


def paragraph_number(paragraph_id):
    match = PARAGRAPH_ID_RE.match(paragraph_id)
    if not match:
        raise ValueError(f"unexpected paragraph id {paragraph_id!r} in migration map")
    return int(match.group(1))


def _nearest_at_or_after(sorted_ids, value):
    index = bisect.bisect_left(sorted_ids, value)
    return sorted_ids[index] if index < len(sorted_ids) else None


def _nearest_before(sorted_ids, value):
    index = bisect.bisect_left(sorted_ids, value)
    return sorted_ids[index - 1] if index > 0 else None


def build_chapter_lookup(chapter_record):
    """old paragraph index -> {"start": new index, "end": new index}."""
    direct_start = {}
    direct_end = {}
    all_old_ids = []
    for pairing in chapter_record["pairings"]:
        old_ids = [paragraph_number(pid) for pid in pairing["old_ids"]]
        new_ids = [paragraph_number(pid) for pid in pairing["new_ids"]]
        all_old_ids.extend(old_ids)
        if not new_ids:
            continue
        start, end = min(new_ids), max(new_ids)
        for old_index in old_ids:
            direct_start[old_index] = start
            direct_end[old_index] = end

    mapped_ids = sorted(direct_start)
    lookup = {}
    for old_index in sorted(all_old_ids):
        if old_index in direct_start:
            lookup[old_index] = {"start": direct_start[old_index], "end": direct_end[old_index]}
            continue
        # old_only: prefer following for a span start, preceding for a span
        # end; fall back to the other direction if that neighbour is off
        # the edge of the chapter.
        following = _nearest_at_or_after(mapped_ids, old_index)
        preceding = _nearest_before(mapped_ids, old_index)
        start = direct_start[following] if following is not None else (
            direct_end[preceding] if preceding is not None else None
        )
        end = direct_end[preceding] if preceding is not None else (
            direct_start[following] if following is not None else None
        )
        lookup[old_index] = {"start": start, "end": end}
    return lookup


def build_lookups(migration_map):
    return {
        chapter["chapterId"]: build_chapter_lookup(chapter)
        for chapter in migration_map["chapters"]
    }


class Remapper:
    """Applies the per-chapter lookups and tallies what happened."""

    def __init__(self, lookups):
        self.lookups = lookups
        self.counts = {}  # artifact -> {"resolved": n, "changed": n}
        self.unresolved = []  # list of human-readable descriptions
        self.link_rewrites = {}  # old reader_link -> new reader_link (for md patching)

    def _tally(self, artifact, changed):
        bucket = self.counts.setdefault(artifact, {"resolved": 0, "changed": 0})
        bucket["resolved"] += 1
        if changed:
            bucket["changed"] += 1

    def resolve(self, artifact, chapter_id, old_number, field, context):
        chapter_lookup = self.lookups.get(chapter_id)
        entry = chapter_lookup.get(old_number) if chapter_lookup else None
        new_number = entry.get(field) if entry else None
        if new_number is None:
            self.unresolved.append(
                f"{artifact}: chapter={chapter_id} p-{old_number} ({field}) -- {context}"
            )
            return None
        self._tally(artifact, changed=new_number != old_number)
        return new_number

    def remap_reader_link(self, artifact, reader_link, field, context):
        if not reader_link or "#p-" not in reader_link:
            return reader_link
        match = READER_LINK_RE.match(reader_link)
        if not match:
            self.unresolved.append(
                f"{artifact}: unparsable reader_link {reader_link!r} -- {context}"
            )
            return reader_link
        chapter_id = match.group("chapter")
        old_number = int(match.group("num"))
        new_number = self.resolve(artifact, chapter_id, old_number, field, context)
        if new_number is None:
            return reader_link
        new_link = f"{match.group('prefix')}#p-{new_number}"
        if new_link != reader_link:
            self.link_rewrites[reader_link] = new_link
        return new_link


# --- per-artifact remapping --------------------------------------------------


def remap_character_pages(remapper):
    if not CHARACTER_PAGES_JSON.exists():
        return
    data = json.loads(CHARACTER_PAGES_JSON.read_text(encoding="utf-8"))
    artifact = "character-pages-supplemented-current.json"
    for page in data.get("pages", []):
        character = page.get("character", "?")
        for field_name in ("notable_units", "reading_path", "top_chapters"):
            for row in page.get(field_name, []):
                context = f"{field_name} character={character} unit_id={row.get('unit_id')}"
                row["reader_link"] = remapper.remap_reader_link(
                    artifact, row.get("reader_link", ""), "start", context
                )
    CHARACTER_PAGES_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if CHARACTER_PAGES_MD.exists() and remapper.link_rewrites:
        text = CHARACTER_PAGES_MD.read_text(encoding="utf-8")
        # reader_link is always rendered backtick-wrapped on its own
        # ("- {label}: `{reader_link}`"). A single re.sub pass over the
        # ORIGINAL text (rather than chained str.replace calls) is required:
        # chaining would let one link's NEW value collide with a different
        # link's OLD value and get double-rewritten.
        any_link_re = re.compile(r"`(/projects/islt/fr-original/[^`]+#p-\d+)`")
        text = any_link_re.sub(
            lambda m: f"`{remapper.link_rewrites.get(m.group(1), m.group(1))}`", text
        )
        CHARACTER_PAGES_MD.write_text(text, encoding="utf-8")


def remap_chapter_summaries(remapper):
    if not CHAPTER_SUMMARIES_JSON.exists():
        return
    data = json.loads(CHAPTER_SUMMARIES_JSON.read_text(encoding="utf-8"))
    artifact = "chapter-summaries-current.json"
    for chapter in data.get("chapters", []):
        chapter_id = chapter["chapter_id"]
        for passage in chapter.get("distinguishing_passages", []):
            context = f"distinguishing_passages unit_id={passage.get('unit_id')}"
            old_start = passage["paragraph_start"]
            old_end = passage["paragraph_end"]
            new_start = remapper.resolve(artifact, chapter_id, old_start, "start", context)
            new_end = remapper.resolve(artifact, chapter_id, old_end, "end", context)
            if new_start is not None:
                passage["paragraph_start"] = new_start
            if new_end is not None:
                passage["paragraph_end"] = new_end
            passage["reader_link"] = remapper.remap_reader_link(
                artifact, passage.get("reader_link", ""), "start", context
            )
    CHAPTER_SUMMARIES_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def remap_whr_timelines(remapper):
    for lens, path in WHR_TIMELINE_JSONS.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        artifact = path.name
        for point in data.get("points", []):
            corpus_position = point["corpus_position"]
            chapter_id = corpus_position["chapter_id"]
            context = f"points character={point.get('character')} unit_id={corpus_position.get('unit_id')}"
            old_start = corpus_position["paragraph_start"]
            old_end = corpus_position["paragraph_end"]
            new_start = remapper.resolve(artifact, chapter_id, old_start, "start", context)
            new_end = remapper.resolve(artifact, chapter_id, old_end, "end", context)
            if new_start is not None:
                corpus_position["paragraph_start"] = new_start
            if new_end is not None:
                corpus_position["paragraph_end"] = new_end
            # cumulative_unit_index, cumulative_paragraph_index(_end) and
            # cumulative_word_count(_end) are legacy-scale running totals
            # used only as chart x-axis coordinates -- left untouched.
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    migration_map = json.loads(MIGRATION_MAP_PATH.read_text(encoding="utf-8"))
    lookups = build_lookups(migration_map)
    remapper = Remapper(lookups)

    remap_character_pages(remapper)
    remap_chapter_summaries(remapper)
    remap_whr_timelines(remapper)

    print("Remap counts (resolved references, of which numerically changed):")
    total_resolved = total_changed = 0
    for artifact, bucket in remapper.counts.items():
        print(f"  {artifact}: resolved={bucket['resolved']} changed={bucket['changed']}")
        total_resolved += bucket["resolved"]
        total_changed += bucket["changed"]
    print(f"  TOTAL: resolved={total_resolved} changed={total_changed}")

    print(f"\nUnresolved references: {len(remapper.unresolved)}")
    for line in remapper.unresolved:
        print(f"  UNRESOLVED {line}")

    return 1 if remapper.unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
