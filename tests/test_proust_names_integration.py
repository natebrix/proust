import pytest


pytest.importorskip("spacy")
pytest.importorskip("spacytextblob")

import proust as pn


@pytest.fixture(scope="module")
def live_nlp():
    original_nlp = pn.get_loaded_nlp()
    try:
        nlp = pn.load_spacy("fr_core_news_sm")
    except Exception as exc:
        pytest.skip(f"live spaCy pipeline unavailable: {exc}")
    pn.set_nlp(nlp)
    yield nlp
    pn.set_nlp(original_nlp)


def test_load_spacy_builds_live_pipeline(live_nlp):
    assert live_nlp.pipe_names == [
        "tok2vec",
        "morphologizer",
        "proust_sentence_start",
        "proust_proper_name",
        "parser",
        "attribute_ruler",
        "lemmatizer",
        "ner",
        "merge_entities",
        "spacytextblob",
    ]


def test_live_pipeline_extracts_sentences_and_entities(live_nlp):
    chapters = [["Swann rencontre Odette.", "Swann revient chez Odette."]]

    assert [sent.text for sent in pn.get_sentences("Swann rencontre Odette. Puis il revient.")] == [
        "Swann rencontre Odette.",
        "Puis il revient.",
    ]

    assert pn.entity_table(chapters).to_dict("records") == [
        {"chapter": 0, "paragraph": 0, "name": "Swann"},
        {"chapter": 0, "paragraph": 1, "name": "Swann"},
        {"chapter": 0, "paragraph": 1, "name": "Odette"},
    ]


def test_live_pipeline_word_freq_and_sentiment(live_nlp):
    chapters = [["Swann rencontre Odette.", "Swann revient chez Odette."]]

    assert pn.word_freq_table(chapters, {"swann", "odette"}).to_dict("records") == [
        {"chapter": 0, "paragraph": 0, "name": "swann"},
        {"chapter": 0, "paragraph": 0, "name": "odette"},
        {"chapter": 0, "paragraph": 1, "name": "swann"},
        {"chapter": 0, "paragraph": 1, "name": "odette"},
    ]

    assert pn.get_sentiment(chapters).to_dict("records") == [
        {"chapter": 0, "paragraph": 0, "polarity": 0.0, "subjectivity": 0.0, "assessed": 0, "length": 23},
        {"chapter": 0, "paragraph": 1, "polarity": 0.0, "subjectivity": 0.0, "assessed": 0, "length": 26},
    ]


def test_file_backed_chapter_loading_smoke(live_nlp):
    chapters = pn.get_proust_chapters(1, 1, source="file")
    assert len(chapters) == 1
    assert len(chapters[0]) == 11
    assert chapters[0][0] == "Du Côté de Chez Swann - Première partie"
    assert chapters[0][1] == "Combray"


def test_get_islt_nlp_flattens_small_text(live_nlp):
    doc = pn.get_islt_nlp(live_nlp, [["Swann rencontre Odette."], ["Odette repond."]])
    assert "Swann rencontre Odette." in doc.text
    assert "Odette repond." in doc.text


def test_session_api_uses_live_pipeline(live_nlp):
    session = pn.ProustSession(model="fr_core_news_sm", aliases={"M. Swann": "Swann"}, nlp=live_nlp)
    assert session.preprocess("M. Swann revient.") == "Swann revient."
    assert [sent.text for sent in session.get_sentences("Swann revient. Odette attend.")] == [
        "Swann revient.",
        "Odette attend.",
    ]
