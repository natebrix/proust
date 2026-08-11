"""Build staged canonical chapters from the cached Wikisource revisions.

Phase B of proust/docs/source_migration_plan.md. Reads the pinned page cache
(data/wikisource/pages, see fetch_wikisource_pages.py), extracts Proust's body
text from the rendered HTML, slices it at the mapping's chapter boundary
anchors, segments sentences with fr_core_news_sm, and writes chapters in the
existing edition schema to data/islt/editions/fr-original-ws/chapters/.

STAGED ONLY: nothing here touches the live fr-original edition. The staged
chapters are reviewed through the alignment report (align_migration_map.py)
before phase D applies them.

Extraction rules. The rendered HTML carries Wikisource apparatus that is not
part of the text: the header/footer/navigation templates (``div.ws-noexport``),
the transcluded page-number markers (``span.pagenum``), footnote markers and
the footnote list (``sup.reference`` / ``ol.references`` — Albertine disparue
chapitre II ends with a Robert Proust editorial note that is emphatically not
Proust's text), the per-chapter argument summaries (``div.alineanegatif``), the
chapter headings, and the ⁂ section-break glyphs. All of that is stripped, and
what was stripped is recorded per chapter so the alignment step can explain
old-side paragraphs that corresponded to it. Everything else is kept verbatim
apart from whitespace collapse: Wikisource's apostrophes (U+2019), guillemets,
accents and ellipses are the new edition's typography.
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bs4 import BeautifulSoup, NavigableString, Tag  # noqa: E402
from bs4.element import PreformattedString  # noqa: E402

from proust.paths import ISLT_EDITIONS_DIR  # noqa: E402

MAPPING_PATH = REPO_ROOT / "outputs" / "wikisource-mapping.json"
PAGES_DIR = REPO_ROOT / "data" / "wikisource" / "pages"
MANIFEST_PATH = REPO_ROOT / "data" / "wikisource" / "manifest.json"
SOURCE_EDITION_DIR = ISLT_EDITIONS_DIR / "fr-original"
STAGING_DIR = ISLT_EDITIONS_DIR / "fr-original-ws"
NLP_MODEL = "fr_core_news_sm"

# Nodes that are apparatus, not text. Removed before any block is collected.
APPARATUS_SELECTORS = (
    "div.ws-noexport",  # header, chapter navigation, footer templates
    "style",
    "script",
    "link",
    "sup.reference",  # footnote markers in the body
    "ol.references",  # the footnote list itself
    "span.pagenum",
    "span.ws-pagenum",
    "div.toc",
    "#toc",
)
# Apparatus whose text is recorded (so old-side counterparts are explainable).
RECORDED_APPARATUS_SELECTORS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div.alineanegatif",  # the edition's chapter argument summaries
)
SECTION_BREAK_CHARS = set("⁂*✳❋❉·.  \n")
BLOCK_TAGS = {
    "p", "div", "dd", "dt", "dl", "center", "blockquote", "li", "td", "th",
    "table", "tr", "ul", "ol", "section", "figure", "figcaption", "pre",
}
# Section titles the edition sets in full capitals outside a heading tag
# ("PREMIÈRE APPARITION DES HOMMES-FEMMES…", "LES INTERMITTENCES DU CŒUR").
# Proust's prose never runs uppercase, so the test is unambiguous.
CAPS_HEADING_MAX_CHARS = 250


def load_nlp():
    import spacy

    return spacy.load(NLP_MODEL, disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])


def collapse(text):
    """Whitespace-collapse a block's text; typography is otherwise untouched."""
    return " ".join(text.split())


def is_section_break(text):
    stripped = text.strip()
    return bool(stripped) and set(stripped) <= SECTION_BREAK_CHARS


def is_caps_heading(text):
    letters = [char for char in text if char.isalpha()]
    return (
        len(letters) >= 3
        and len(text) <= CAPS_HEADING_MAX_CHARS
        and text == text.upper()
    )


def content_root(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div", class_="mw-parser-output") or soup


def in_verse(node):
    """True inside a verse block, where a <br> ends a line, not a soft wrap."""
    chain = [node] + [parent for parent in node.parents if isinstance(parent, Tag)]
    return any(
        {"poem", "verse"} & set(element.get("class") or [])
        for element in chain
        if isinstance(element, Tag)
    )


def is_text(node):
    """A real text node: comments and other preformatted strings are not."""
    return isinstance(node, NavigableString) and not isinstance(node, PreformattedString)


def node_lines(nodes):
    """Text of a run of nodes, <br> as line separator, whitespace-collapsed."""
    parts = []
    for node in nodes:
        if not isinstance(node, Tag):
            if is_text(node):
                parts.append(str(node))
            continue
        for descendant in [node] + list(node.descendants):
            if is_text(descendant):
                parts.append(str(descendant))
            elif isinstance(descendant, Tag) and descendant.name == "br":
                parts.append("\n")
    return [collapse(line) for line in "".join(parts).split("\n")]


def block_texts(nodes, verse):
    """Paragraph texts for one block node (or one run of loose inline nodes).

    Verse blocks keep one line per paragraph (the way the source sets them and
    the way the legacy edition stored them); prose joins across the soft line
    breaks that page transclusion leaves behind.
    """
    lines = [line for line in node_lines(nodes) if line]
    if not lines:
        return []
    if verse and len(lines) > 1:
        return lines
    return [" ".join(lines)]


def collect_blocks(node, out):
    """Walk the content tree, emitting one text per block-level leaf.

    Loose inline content between block children is emitted too: Wikisource's
    verse blocks interleave <p> lines with bare italic lines (La Prisonnière
    chapitre 1 sets one line of the Esther quotation that way), and dropping
    them would silently lose text.
    """
    pending = []
    for child in node.children:
        if isinstance(child, Tag) and child.name in BLOCK_TAGS:
            out.extend(block_texts(pending, in_verse(node)))
            pending = []
            if child.name != "p" and child.find("p") is not None:
                collect_blocks(child, out)
            else:
                out.extend(block_texts([child], in_verse(child)))
            continue
        pending.append(child)
    out.extend(block_texts(pending, in_verse(node)))
    return out


def page_paragraphs(html):
    """(body paragraphs, stripped apparatus texts) for one cached page."""
    root = content_root(html)
    stripped = []
    for selector in RECORDED_APPARATUS_SELECTORS:
        for node in root.select(selector):
            text = collapse(node.get_text(""))
            if text:
                stripped.append(text)
            node.decompose()
    for selector in APPARATUS_SELECTORS:
        for node in root.select(selector):
            node.decompose()
    paragraphs = []
    for text in collect_blocks(root, []):
        if not text:
            continue
        if is_section_break(text) or is_caps_heading(text):
            stripped.append(text)
            continue
        paragraphs.append(text)
    return paragraphs, stripped


# --- boundary anchors ------------------------------------------------------

QUOTE_VARIANTS = {
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...",
}
# Residue allowed between an anchor match and the paragraph edge when deciding
# that the anchor sits at that edge (a closing guillemet, final punctuation).
EDGE_RESIDUE_CHARS = set(" .,;:!?«»\"'…-—–")


def fold(text):
    """Comparison form of a text, with a map back to the original offsets.

    Apostrophe, quote, dash and ellipsis variants are unified and case is
    dropped, so an anchor phrase quoted with one typography finds text set with
    another. Returns (folded_text, offsets) where offsets[i] is the index in
    ``text`` that produced folded character i.
    """
    folded, offsets = [], []
    for index, char in enumerate(unicodedata.normalize("NFC", text)):
        replacement = QUOTE_VARIANTS.get(char, char).lower()
        folded.append(replacement)
        offsets.extend([index] * len(replacement))
    return "".join(folded), offsets


def anchor_key(text):
    """Comparison form of an anchor phrase (leading/trailing ellipsis dropped)."""
    return fold(collapse(text))[0].strip(". ")


def find_anchor(paragraphs, anchor, kind, chapter_id):
    """(paragraph index, start offset, end offset) of the one anchor match.

    Offsets are into the paragraph's own text, so a boundary that falls inside
    a Wikisource paragraph (their paragraphing is not always ours) can be cut
    at exactly the anchor.
    """
    needle = anchor_key(anchor)
    if not needle:
        raise ValueError(f"{chapter_id}: empty {kind} anchor")
    hits = []
    for index, text in enumerate(paragraphs):
        folded, offsets = fold(text)
        position = folded.find(needle)
        if position < 0:
            continue
        if folded.find(needle, position + 1) >= 0:
            raise ValueError(
                f"{chapter_id}: {kind} anchor matches twice in one paragraph: "
                f"{anchor[:80]!r}"
            )
        hits.append((index, offsets[position], offsets[position + len(needle) - 1] + 1))
    if len(hits) != 1:
        raise ValueError(
            f"{chapter_id}: {kind} anchor matched {len(hits)} paragraphs "
            f"(need exactly 1): {anchor[:80]!r}"
        )
    return hits[0]


def _edge_residue(text):
    return set(text) <= EDGE_RESIDUE_CHARS


def slice_page(paragraphs, from_anchor, to_anchor, chapter_id):
    """The paragraph run for one chapter on one page, anchors inclusive."""
    paragraphs = list(paragraphs)
    start, end = 0, len(paragraphs)
    if from_anchor:
        index, begin, _ = find_anchor(paragraphs, from_anchor, "from", chapter_id)
        start = index
        if begin and not _edge_residue(paragraphs[index][:begin]):
            paragraphs[index] = paragraphs[index][begin:].lstrip()
    if to_anchor:
        index, _, finish = find_anchor(paragraphs, to_anchor, "to", chapter_id)
        end = index + 1
        tail = paragraphs[index][finish:]
        if tail and not _edge_residue(tail):
            paragraphs[index] = paragraphs[index][:finish] + _trailing_punctuation(tail)
    if end <= start:
        raise ValueError(f"{chapter_id}: empty slice ({start}:{end})")
    return paragraphs[start:end]


def _trailing_punctuation(tail):
    """The closing punctuation an anchor phrase stops short of (' »')."""
    keep = 0
    while keep < len(tail) and tail[keep] in EDGE_RESIDUE_CHARS:
        keep += 1
    return tail[:keep].rstrip()


# --- chapter assembly ------------------------------------------------------


def segment(text, nlp):
    if not text.strip():
        return []
    return [
        {"id": f"s-{index}", "index": index, "text": sentence.text.strip()}
        for index, sentence in enumerate(
            (s for s in nlp(text).sents if s.text.strip()), start=1
        )
    ]


def chapter_paragraphs(chapter_entry, pages_by_key):
    """Body paragraphs and stripped apparatus for one canonical chapter."""
    chapter_id = chapter_entry["canonical_id"]
    paragraphs, stripped = [], []
    for page in chapter_entry["pages"]:
        html = pages_by_key[(page["title"], page["revid"])]
        page_body, page_stripped = page_paragraphs(html)
        stripped.extend(page_stripped)
        paragraphs.extend(
            slice_page(
                page_body, page.get("from_anchor"), page.get("to_anchor"), chapter_id
            )
        )
    return paragraphs, stripped


def build_chapter(current, texts, nlp):
    paragraphs = [
        {
            "id": f"p-{index}",
            "index": index,
            "text": text,
            "sentences": segment(text, nlp),
        }
        for index, text in enumerate(texts, start=1)
    ]
    chapter = dict(current)
    chapter["paragraphs"] = paragraphs
    chapter["paragraphCount"] = len(paragraphs)
    chapter["sentenceCount"] = sum(len(p["sentences"]) for p in paragraphs)
    return chapter


def load_pages():
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        (entry["title"], entry["revid"]): (PAGES_DIR / f"{entry['slug']}.html").read_text()
        for entry in manifest["pages"]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", help="restrict to one canonical chapter id")
    parser.add_argument(
        "--dry-run", action="store_true", help="report counts, write nothing"
    )
    args = parser.parse_args(argv)

    mapping = json.loads(MAPPING_PATH.read_text())
    pages_by_key = load_pages()
    nlp = None if args.dry_run else load_nlp()
    chapters_dir = STAGING_DIR / "chapters"
    if not args.dry_run:
        chapters_dir.mkdir(parents=True, exist_ok=True)

    apparatus = {}
    total = 0
    for entry in mapping["chapters"]:
        chapter_id = entry["canonical_id"]
        if args.chapter and chapter_id != args.chapter:
            continue
        texts, stripped = chapter_paragraphs(entry, pages_by_key)
        current_path = SOURCE_EDITION_DIR / "chapters" / f"{chapter_id}.json"
        current = json.loads(current_path.read_text())
        apparatus[chapter_id] = stripped
        total += len(texts)
        print(
            f"{chapter_id:52s} old={current['paragraphCount']:4d} "
            f"new={len(texts):4d} stripped-apparatus={len(stripped)}"
        )
        if args.dry_run:
            continue
        chapter = build_chapter(current, texts, nlp)
        (chapters_dir / f"{chapter_id}.json").write_text(
            json.dumps(chapter, ensure_ascii=False, indent=2) + "\n"
        )

    print(f"\ntotal staged paragraphs: {total}")
    if not args.dry_run:
        (STAGING_DIR / "stripped-apparatus.json").write_text(
            json.dumps(apparatus, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"staged to {STAGING_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
