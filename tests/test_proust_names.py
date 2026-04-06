from types import SimpleNamespace

import pandas as pd

import proust as pn
from proust import create_session
from proust import corpus as corpus_module


class FakeToken:
    def __init__(self, text, lemma=None, is_alpha=True, is_stop=False):
        self.text = text
        self.lemma_ = lemma or text
        self.is_alpha = is_alpha
        self.is_stop = is_stop

    def __len__(self):
        return len(self.text)


class FakePipeline:
    def __call__(self, text):
        sentences = [segment.strip() for segment in text.split(".") if segment.strip()]
        return SimpleNamespace(
            sents=[SimpleNamespace(text=segment + ".") for segment in sentences],
            ents=[],
        )


def test_preprocess_applies_punctuation_fix_and_aliases(monkeypatch):
    monkeypatch.setattr(corpus_module, "get_aliases", lambda: {"M. Swann": "Swann"})
    text = "M. Swann ; – arrive – ; vite"
    assert pn.preprocess(text) == "Swann ; arrive ; vite"


def test_preprocess_can_skip_aliases(monkeypatch):
    monkeypatch.setattr(corpus_module, "get_aliases", lambda: {"M. Swann": "Swann"})
    assert pn.preprocess("M. Swann", use_aliases=False) == "M. Swann"


def test_entity_helpers_and_top_entities():
    et = pd.DataFrame(
        [
            {"chapter": 0, "paragraph": 0, "name": "Swann"},
            {"chapter": 0, "paragraph": 1, "name": "Swann"},
            {"chapter": 1, "paragraph": 0, "name": "Odette"},
            {"chapter": 1, "paragraph": 1, "name": "Swann"},
        ]
    )

    refs = pn.get_references_for_names(et, ["Swann"])
    assert refs["name"].tolist() == ["Swann", "Swann", "Swann"]

    counts = pn.entity_count(et)
    assert counts.loc["Swann", "count"] == 3
    assert counts.loc["Odette", "count"] == 1

    top = pn.top_entities(et, 1)
    assert top.index.tolist() == ["Swann"]


def test_get_ref_count_by_chapter_pivots_counts():
    et = pd.DataFrame(
        [
            {"chapter": 0, "paragraph": 0, "name": "Swann"},
            {"chapter": 0, "paragraph": 1, "name": "Swann"},
            {"chapter": 1, "paragraph": 0, "name": "Swann"},
            {"chapter": 1, "paragraph": 1, "name": "Odette"},
        ]
    )
    rc = pn.get_ref_count_by_chapter(et, ["Swann", "Odette"])
    assert rc.to_dict("records") == [
        {"chapter": 0, "Odette": 0.0, "Swann": 2.0},
        {"chapter": 1, "Odette": 1.0, "Swann": 1.0},
    ]


def test_smooth_ref_count_preserves_chapter_column():
    rc = pd.DataFrame(
        [
            {"chapter": 0, "Swann": 0.0},
            {"chapter": 1, "Swann": 2.0},
            {"chapter": 2, "Swann": 4.0},
        ]
    )
    smoothed = pn.smooth_ref_count(rc, window=2)
    assert smoothed["chapter"].tolist() == [0, 1, 2]
    assert smoothed["Swann"].tolist() == [0.0, 1.0, 3.0]


def test_volume_column_uses_volume_boundaries():
    df = pd.DataFrame({"chapter": [0, 2, 3, 17]})
    assert pn.volume_column(df).tolist() == [1, 1, 2, 7]


def test_volume_column_accepts_custom_boundaries():
    df = pd.DataFrame({"chapter": [0, 0, 1, 2]})
    assert pn.volume_column(df, starts=[0, 1, 3]).tolist() == [1, 1, 2, 2]


def test_flatten_islt_joins_chapters_and_paragraphs():
    islt = [["A", "B"], ["C"]]
    assert pn.flatten_islt(islt) == "A \nB\nC"


def test_summary_stats_counts_words_and_lemmas():
    doc = [
        FakeToken("Bonjour", lemma="bonjour"),
        FakeToken("le", lemma="le", is_stop=True),
        FakeToken("monde", lemma="monde"),
        FakeToken("!", lemma="!", is_alpha=False),
    ]
    df, word_freq, word_freq_no_stop, lemma_freq = pn.summary_stats(doc)
    stats = dict(df.values.tolist())

    assert stats == {
        "characters": 15,
        "words": 3,
        "words (no stop)": 2,
        "lemmas": 2,
        "unique words": 3,
        "unique words (no stop)": 2,
        "unique lemma": 2,
    }
    assert word_freq["bonjour"] == 1
    assert word_freq_no_stop["monde"] == 1
    assert lemma_freq["bonjour"] == 1


def test_join_and_aggregate_sentiment_helpers():
    et = pd.DataFrame(
        [
            {"chapter": 0, "paragraph": 0, "name": "Swann", "volume": 1},
            {"chapter": 0, "paragraph": 1, "name": "Swann", "volume": 1},
            {"chapter": 1, "paragraph": 0, "name": "Odette", "volume": 1},
        ]
    )
    sentiment = pd.DataFrame(
        [
            {"chapter": 0, "paragraph": 0, "polarity": 0.5, "subjectivity": 0.2},
            {"chapter": 0, "paragraph": 1, "polarity": 0.1, "subjectivity": 0.8},
            {"chapter": 1, "paragraph": 0, "polarity": -0.2, "subjectivity": 0.4},
        ]
    )

    joined = pn.join_sentiment(et, sentiment)
    assert list(joined.columns) == ["chapter", "paragraph", "name", "volume", "polarity", "subjectivity"]

    by_name = pn.get_sentiment_by_name(et, sentiment)
    assert by_name.to_dict("records") == [
        {"name": "Swann", "count": 2, "polarity": 0.3, "subjectivity": 0.5},
        {"name": "Odette", "count": 1, "polarity": -0.2, "subjectivity": 0.4},
    ]

    by_name_volume = pn.get_sentiment_by_name_volume(et, sentiment, ["Swann"])
    assert by_name_volume.to_dict("records") == [
        {"name": "Swann", "volume": 1, "count": 2, "polarity": 0.3, "subjectivity": 0.5}
    ]


def test_get_polarity_rank_filters_by_min_count():
    s_by_n = pd.DataFrame(
        [
            {"name": "A", "count": 50, "polarity": -0.1},
            {"name": "B", "count": 100, "polarity": 0.0},
            {"name": "C", "count": 200, "polarity": 0.4},
        ]
    )
    ranked = pn.get_polarity_rank(s_by_n, min_count=100)
    assert ranked.index.tolist() == [1, 2]
    assert ranked.tolist() == [0.5, 1.0]


def test_session_uses_explicit_aliases_and_nlp():
    session = create_session(
        aliases={"M. Swann": "Swann"},
        nlp=FakePipeline(),
    )

    assert session.preprocess("M. Swann ; – arrive") == "Swann ; arrive"
    assert [sent.text for sent in session.get_sentences("Un. Deux.")] == ["Un.", "Deux."]


def test_canonical_structure_and_chapter_loading_smoke():
    structure = pn.get_canonical_structure()
    assert len(structure) == 18
    assert structure[0]["id"] == "v1-p1-combray"
    assert structure[-1]["id"] == "v7-p4-le-bal-de-tetes"

    chapter = pn.get_canonical_chapter("v1-p1-combray")
    assert chapter["chapterId"] == "v1-p1-combray"
    assert chapter["volumeNumber"] == 1
    assert chapter["partTitle"] == "Combray"


def test_get_canonical_chapters_returns_18_canonical_units():
    chapters = pn.get_canonical_chapters(use_aliases=False)
    assert len(chapters) == 18
    assert chapters[0][0].startswith("Longtemps, je me suis couché de bonne heure.")


def test_get_proust_chapters_defaults_to_canonical_units():
    chapters = pn.get_proust_chapters(1, 1, use_aliases=False)
    assert len(chapters) == 1
    assert chapters[0][0].startswith("Longtemps, je me suis couché de bonne heure.")


def test_session_exposes_canonical_dataset():
    session = create_session(aliases={}, nlp=FakePipeline())
    chapters = session.get_canonical_chapters(use_aliases=False)
    assert len(chapters) == 18
    assert session.get_canonical_structure()[0]["volumeNumber"] == 1


def test_session_defaults_to_repo_supported_model():
    assert create_session().model == "fr_core_news_sm"
