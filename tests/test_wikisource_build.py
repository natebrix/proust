"""Tests for the Wikisource source-migration machinery (phase B)."""
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_wikisource_chapters")
align = _load("align_migration_map")


PAGE_HTML = """
<div class="mw-parser-output">
  <div class="ws-noexport" id="headertemplate">
    <div class="headertemplate">Marcel Proust — navigation ◄ Chapitre I Chapitre III ►</div>
  </div>
  <style>.headertemplate{color:red}</style>
  <p><span><span class="pagenum ws-pagenum" id="175" title="Page:Proust.djvu/175"></span></span></p>
  <h2 class="tmp">CHAPITRE II<span class="sc">Mademoiselle de Forcheville</span></h2>
  <div class="alineanegatif">Brusque revirement vers Albertine.</div>
  <p><br/></p>
  <p>Ce n’était pas que je n’aimasse encore Albertine<sup class="reference" id="cite_ref-1"><a href="#cite_note-1">[1]</a></sup>,
     mais déjà pas de la même façon.</p>
  <div style="text-align:center;">⁂</div>
  <p>Parfois la lecture d’un roman<span><span class="pagenum ws-pagenum" id="176"></span></span> un peu triste
     me ramenait brusquement en arrière.</p>
  <div style="display:table;"><div class="poem">
    <p>« <i>Quel est donc ce mystère</i><br/>
    <i>Je n’y puis rien comprendre.</i> »</p>
    <i>Rien ne met à l’abri de cet ordre fatal,</i><br/>
  </div></div>
  <div style="margin-left:0em;">ou :</div>
  <p>Et pourtant, si l’on ne peut pas revenir à l’indifférence d’où on était parti.</p>
  <ol class="references">
    <li id="cite_note-1"><span class="reference-text">Anecdote racontée par Mme de Guermantes.
      (Note du Dr Robert Proust.)</span></li>
  </ol>
</div>
"""


# --- extraction ------------------------------------------------------------


def test_page_paragraphs_strips_apparatus_and_keeps_body():
    paragraphs, stripped = build.page_paragraphs(PAGE_HTML)

    assert paragraphs == [
        "Ce n’était pas que je n’aimasse encore Albertine, mais déjà pas de la même façon.",
        "Parfois la lecture d’un roman un peu triste me ramenait brusquement en arrière.",
        "« Quel est donc ce mystère",
        "Je n’y puis rien comprendre. »",
        "Rien ne met à l’abri de cet ordre fatal,",
        "ou :",
        "Et pourtant, si l’on ne peut pas revenir à l’indifférence d’où on était parti.",
    ]
    body = " ".join(paragraphs)
    assert "Robert Proust" not in body  # the editorial footnote is not Proust
    assert "[1]" not in body  # nor is its marker
    assert "navigation" not in body
    assert "⁂" not in body


def test_page_paragraphs_records_stripped_apparatus():
    _, stripped = build.page_paragraphs(PAGE_HTML)

    assert "CHAPITRE IIMademoiselle de Forcheville" in stripped
    assert "Brusque revirement vers Albertine." in stripped
    assert "⁂" in stripped


def test_caps_headings_are_apparatus_but_prose_is_not():
    assert build.is_caps_heading("LES INTERMITTENCES DU CŒUR")
    assert not build.is_caps_heading("Les intermittences du cœur")
    assert not build.is_caps_heading("« I »")


def test_verse_lines_stay_separate_but_prose_joins_soft_breaks():
    paragraphs, _ = build.page_paragraphs(PAGE_HTML)

    assert "« Quel est donc ce mystère" in paragraphs
    assert "Je n’y puis rien comprendre. »" in paragraphs
    assert all("\n" not in text for text in paragraphs)


# --- boundary anchors ------------------------------------------------------

PARAGRAPHS = [
    "Premier paragraphe de la page.",
    "Deuxième paragraphe, où commence notre chapitre suivant.",
    "Troisième paragraphe de la page.",
]


def test_anchor_matches_across_apostrophe_and_quote_variants():
    paragraphs = ["Il n’y avait rien « là ».", "Autre chose — vraiment…"]

    index, start, end = build.find_anchor(
        paragraphs, "Il n'y avait rien \" là \"", "from", "v0"
    )

    assert (index, start) == (0, 0)
    assert paragraphs[0][start:end].startswith("Il n’y avait rien")


def test_anchor_must_match_exactly_once():
    with pytest.raises(ValueError, match="matched 0 paragraphs"):
        build.find_anchor(PARAGRAPHS, "phrase absente", "from", "v0")
    with pytest.raises(ValueError, match="matched 2 paragraphs"):
        build.find_anchor(PARAGRAPHS, "paragraphe de la page", "from", "v0")


def test_slice_page_keeps_anchor_paragraphs_at_the_edges():
    sliced = build.slice_page(PARAGRAPHS, "Deuxième paragraphe", None, "v0")
    assert sliced == PARAGRAPHS[1:]

    sliced = build.slice_page(PARAGRAPHS, None, "commence notre chapitre suivant", "v0")
    assert sliced == PARAGRAPHS[:2]


def test_slice_page_cuts_inside_a_paragraph_when_the_boundary_falls_there():
    """Le Temps retrouvé/I sets our v7-p1|v7-p2 boundary mid-paragraph."""
    paragraphs = [
        "Début du chapitre.",
        "Fin de la première partie ! » Cette disposition-là, les pages de Goncourt. Suite.",
    ]

    head = build.slice_page(paragraphs, None, "Fin de la première partie !", "v7-p1")
    tail = build.slice_page(paragraphs, "Cette disposition-là", None, "v7-p2")

    assert head == ["Début du chapitre.", "Fin de la première partie ! »"]
    assert tail == ["Cette disposition-là, les pages de Goncourt. Suite."]


def test_slice_page_rejects_an_empty_slice():
    with pytest.raises(ValueError, match="empty slice"):
        build.slice_page(PARAGRAPHS, "Troisième", "Premier", "v0")


# --- alignment -------------------------------------------------------------


def _kinds(old_texts, new_texts):
    return [
        (align.pairing_kind(old, new), old, new)
        for old, new, _ in align.align(old_texts, new_texts)
    ]


def test_alignment_pairs_typography_variants_one_to_one():
    old = ["C'est ainsi qu'il parlait, disait-elle.", "Un second paragraphe entier."]
    new = ["C’est ainsi qu’il parlait, disait-elle.", "Un second paragraphe entier."]

    pairings = align.align(old, new)

    assert [align.pairing_kind(o, n) for o, n, _ in pairings] == ["one_to_one"] * 2
    assert all(score == 1.0 for _, _, score in pairings)


def test_alignment_detects_split_merge_old_only_and_new_only():
    old = [
        "Une phrase commune à tout le monde et parfaitement identique des deux côtés.",
        "Une seule vieille phrase suivie d'une autre vieille phrase dans le même bloc.",
        "[----Ajout Gallimard---- un passage que Wikisource ignore complètement ----]",
        "Un fragment coupé en deux,",
        "et la suite du fragment coupé.",
        "Une dernière phrase commune aux deux transcriptions du même paragraphe.",
    ]
    new = [
        "Une phrase commune à tout le monde et parfaitement identique des deux côtés.",
        "Une seule vieille phrase",
        "suivie d’une autre vieille phrase dans le même bloc.",
        "Un fragment coupé en deux, et la suite du fragment coupé.",
        "Un paragraphe que la transcription ancienne avait purement et simplement perdu.",
        "Une dernière phrase commune aux deux transcriptions du même paragraphe.",
    ]

    kinds = _kinds(old, new)

    assert ("split", [1], [1, 2]) in kinds
    assert ("merge", [3, 4], [3]) in kinds
    assert ("old_only", [2], []) in kinds
    assert ("new_only", [], [4]) in kinds
    assert kinds[0] == ("one_to_one", [0], [0])
    assert kinds[-1] == ("one_to_one", [5], [5])


def test_similarity_ignores_typography_but_not_words():
    assert align.similarity("C'est l'été...", "C’est l’été…") == 1.0
    assert align.similarity("il partit demain", "il partit hier") < 1.0
    assert align.similarity("rien de commun ici", "") == 0.0


# --- annotation ------------------------------------------------------------


def test_known_legacy_artifacts_are_annotated():
    assert align.annotate_old_only("--> Retour à la première page : 001", []) == "site_navigation"
    assert align.annotate_old_only("FIN du roman A LA RECHERCHE DU TEMPS PERDU de", []) == "colophon"
    assert align.annotate_old_only("[----Ajout Gallimard---- Et vous", []) == "editorial_marker"
    assert align.annotate_old_only("[L'édition sonore Thélème reprend ici", []) == "editorial_marker"
    assert align.annotate_old_only("* * *", []) == "section_break"
    assert align.annotate_old_only("   ", []) == "structural_spacer"
    assert align.annotate_old_only("Chapitre deuxième", ["chapitre deuxième les verdurin"]) == (
        "wikisource_apparatus"
    )
    assert align.annotate_old_only("Un paragraphe de Proust bien réel.", []) is None


def test_artifact_annotation_covers_the_whole_unmatched_run():
    pairings = [
        {"kind": "one_to_one", "old_ids": ["p-1"], "new_ids": ["p-1"], "similarity": 1.0},
        {"kind": "new_only", "old_ids": [], "new_ids": ["p-2"], "similarity": 0.0},
        {"kind": "old_only", "old_ids": ["p-2"], "new_ids": [], "similarity": 0.0,
         "annotation": None},
        {"kind": "old_only", "old_ids": ["p-3"], "new_ids": [], "similarity": 0.0,
         "annotation": "editorial_marker"},
        {"kind": "old_only", "old_ids": ["p-4"], "new_ids": [], "similarity": 0.0,
         "annotation": None},
    ]

    align.propagate_block_annotations(pairings)

    assert pairings[2]["annotation"] == "editorial_marker_block"
    assert pairings[4]["annotation"] == "editorial_marker_block"
    assert pairings[1]["annotation"] == "editorial_marker_block_counterpart"


# --- gates -----------------------------------------------------------------


def _record(**stats):
    base = {
        "oldParagraphCount": 100,
        "newParagraphCount": 100,
        "kinds": {"one_to_one": 100, "split": 0, "merge": 0, "old_only": 0, "new_only": 0},
        "medianSimilarity": 1.0,
        "unexplainedOldOnly": 0,
        "newOnly": 0,
        "newOnlyRate": 0.0,
    }
    base.update(stats)
    return {"chapterId": "v0", "stats": base, "pairings": []}


def test_gates_pass_on_a_clean_chapter():
    assert align.gate_failures(_record()) == []


def test_gate_fails_on_unexplained_old_only():
    failures = align.gate_failures(_record(unexplainedOldOnly=2))
    assert failures and "unexplained old_only" in failures[0]


def test_gate_fails_when_new_only_exceeds_the_rate():
    failures = align.gate_failures(_record(newOnly=3, newOnlyRate=0.03))
    assert failures and "new_only rate" in failures[0]


def test_gate_fails_on_low_median_similarity():
    failures = align.gate_failures(_record(medianSimilarity=0.94))
    assert failures and "median 1:1 similarity" in failures[0]


def test_chapter_record_reports_counts_and_stats():
    old_chapter = {
        "paragraphs": [
            {"id": "p-1", "text": "Une phrase parfaitement commune aux deux transcriptions."},
            {"id": "p-2", "text": "* * *"},
            {"id": "p-3", "text": "Une autre phrase commune aux deux transcriptions du texte."},
        ]
    }
    new_chapter = {
        "paragraphs": [
            {"id": "p-1", "text": "Une phrase parfaitement commune aux deux transcriptions."},
            {"id": "p-2", "text": "Une autre phrase commune aux deux transcriptions du texte."},
        ]
    }

    record = align.chapter_record("v0", old_chapter, new_chapter, ["⁂"])

    assert record["stats"]["oldParagraphCount"] == 3
    assert record["stats"]["newParagraphCount"] == 2
    assert record["stats"]["kinds"]["one_to_one"] == 2
    assert record["stats"]["kinds"]["old_only"] == 1
    assert record["stats"]["unexplainedOldOnly"] == 0
    assert record["stats"]["medianSimilarity"] == 1.0
    assert align.gate_failures(record) == []
