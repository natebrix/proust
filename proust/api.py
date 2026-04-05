from dataclasses import dataclass

from . import analytics, corpus, nlp as nlp_module


@dataclass
class ProustSession:
    model: str = "fr_core_news_sm"
    aliases: dict | None = None
    nlp: object | None = None
    default_source: str = "file"

    def load_aliases(self):
        self.aliases = corpus.read_aliases()
        return self.aliases

    def get_aliases(self):
        if self.aliases is None:
            self.aliases = corpus.read_aliases()
        return self.aliases

    def load_nlp(self):
        self.nlp = nlp_module.load_spacy(self.model)
        return self.nlp

    def get_nlp(self):
        if self.nlp is None:
            self.nlp = self.load_nlp()
        return self.nlp

    def preprocess(self, text, use_aliases=True):
        aliases = self.get_aliases() if use_aliases else None
        return corpus.preprocess(text, use_aliases=use_aliases, aliases=aliases)

    def get_sentences(self, text):
        return corpus.get_sentences(text, nlp=self.get_nlp())

    def get_proust_page(self, id, source=None):
        return corpus.get_proust_page(id, source=source or self.default_source)

    def get_proust_pages(self, id_start=1, id_end=486, source=None):
        return corpus.get_proust_pages(id_start=id_start, id_end=id_end, source=source or self.default_source)

    def get_proust_chapters(self, id_start=1, id_end=None, source=None, use_aliases=True, by_paragraph=True):
        aliases = self.get_aliases() if use_aliases else None
        return corpus.get_proust_chapters(
            id_start=id_start,
            id_end=id_end,
            source=source or self.default_source,
            use_aliases=use_aliases,
            by_paragraph=by_paragraph,
            aliases=aliases,
        )

    def get_canonical_structure(self, edition="fr-original"):
        return corpus.get_canonical_structure(edition=edition)

    def get_canonical_chapter(self, chapter_id, edition="fr-original"):
        return corpus.get_canonical_chapter(chapter_id, edition=edition)

    def get_canonical_chapters(self, id_start=1, id_end=None, edition="fr-original", use_aliases=True):
        aliases = self.get_aliases() if use_aliases else None
        return corpus.get_canonical_chapters(
            id_start=id_start,
            id_end=id_end,
            edition=edition,
            use_aliases=use_aliases,
            aliases=aliases,
        )

    def get_paragraphs(self, chapter):
        return corpus.get_paragraphs(chapter, nlp=self.get_nlp(), aliases=self.get_aliases())

    def entity_table(self, chapters):
        return analytics.entity_table(chapters, nlp=self.get_nlp())

    def word_freq_table(self, chapters, words):
        return analytics.word_freq_table(chapters, words, nlp=self.get_nlp())

    def get_proust_names(self, source=None, use_aliases=True):
        aliases = self.get_aliases() if use_aliases else None
        return analytics.get_proust_names(
            source=source or self.default_source,
            use_aliases=use_aliases,
            aliases=aliases,
            nlp=self.get_nlp(),
        )

    def get_sentiment(self, chapters):
        return analytics.get_sentiment(chapters, nlp=self.get_nlp())


def create_session(model="fr_core_news_sm", aliases=None, nlp=None, default_source="file"):
    return ProustSession(model=model, aliases=aliases, nlp=nlp, default_source=default_source)
