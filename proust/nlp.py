try:
    import spacy
    from spacy.language import Language
except ModuleNotFoundError:
    spacy = None
    Language = None

try:
    from spacytextblob.spacytextblob import SpacyTextBlob
except ModuleNotFoundError:
    SpacyTextBlob = None

from .config import conjunction, entity_pos, honorific, proper_name_exceptions
from .state import get_loaded_nlp, set_nlp


def _identity(value):
    return value


if Language is not None:
    component = Language.component
else:
    def component(_name):
        return _identity


def _require_spacy():
    if spacy is None:
        raise ModuleNotFoundError("spaCy is required for this operation")
    if SpacyTextBlob is None:
        raise ModuleNotFoundError("spacytextblob is required for this operation")


def _ensure_spacy_extensions():
    _require_spacy()
    spacy.tokens.token.Token.set_extension("family", default="", force=True)


@component("proust_sentence_start")
def proust_sentence_start(doc):
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text == ";" and index + 1 < length:
            doc[index + 1].sent_start = False
    return doc


@component("proust_proper_name")
def proust_proper_name(doc):
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text in honorific:
            if index + 3 < length and doc[index + 1].text in conjunction and doc[index + 2].text in honorific:
                token._.family = "next"
            else:
                token._.family = "skip"

        if token.pos_ == "PROPN":
            case = None
            if token.text in proper_name_exceptions:
                case = "EXCEPTION"
            elif len(token.text) and not token.text[0].isupper():
                case = "IS_LOWERCASE"
            if case:
                token.pos_ = "NOUN"
    return doc


def get_family_name(entity, token):
    del entity
    return " ".join(token.text.split(" ")[1:]) if " " in token.text else token.text


def is_entity_pos(entity):
    return entity.root.pos_ in entity_pos


def canonicalize_entity(entity, index, entities_):
    if entity.root._.family == "next":
        return entity.text + " " + get_family_name(entity, entities_[index + 1])
    if entity.root._.family == "skip":
        return entity.text
    return entity.text


def entities(text):
    items = [entity for entity in text.ents if is_entity_pos(entity)]
    return [canonicalize_entity(entity, index, items) for index, entity in enumerate(items)]


def words_in_list(text, words):
    return [token.text.lower() for token in text if token.text.lower() in words]


def load_spacy(model="fr_core_news_lg"):
    _require_spacy()
    _ensure_spacy_extensions()
    print(f"loading spacy model {model}")
    nlp = spacy.load(model)
    nlp.add_pipe("proust_sentence_start", before="parser")
    nlp.add_pipe("proust_proper_name", before="parser")
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("spacytextblob")
    return nlp


def get_nlp(model="fr_core_news_lg"):
    nlp = get_loaded_nlp()
    if nlp is None:
        nlp = load_spacy(model)
        set_nlp(nlp)
    return nlp


def get_doc_sentiment(doc):
    sentiment = getattr(doc._, "sentiment", None)
    if sentiment is not None:
        return sentiment
    blob = getattr(doc._, "blob", None)
    if blob is None:
        raise AttributeError("No sentiment extension found on spaCy doc")
    return blob


def sentiment_assessment_count(sentiment):
    assessments = getattr(sentiment, "assessments", None)
    if assessments is not None:
        return len(assessments)
    sentiment_assessments = getattr(sentiment, "sentiment_assessments", None)
    if sentiment_assessments is not None:
        return len(sentiment_assessments.assessments)
    return 0
