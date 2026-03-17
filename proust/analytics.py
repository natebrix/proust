import bisect
from collections import Counter

import numpy as np
import pandas as pd

from .config import short_labels, volume_starts
from .corpus import get_proust_chapters
from .nlp import entities, get_doc_sentiment, get_nlp, sentiment_assessment_count, words_in_list


def entity_table(chapters):
    print(f"Finding entities within {len(chapters)} chapters.")
    nlp = get_nlp()
    rows = []
    for chapter_index, chapter in enumerate(chapters):
        for paragraph_index, paragraph in enumerate(chapter):
            rows.extend([[chapter_index, paragraph_index, entity] for entity in entities(nlp(paragraph))])
    return pd.DataFrame(rows, columns=["chapter", "paragraph", "name"])


def word_freq_table(chapters, words):
    print(f"Finding word frequency within {len(chapters)} chapters.")
    nlp = get_nlp()
    rows = []
    for chapter_index, chapter in enumerate(chapters):
        for paragraph_index, paragraph in enumerate(chapter):
            rows.extend([[chapter_index, paragraph_index, word] for word in words_in_list(nlp(paragraph), words)])
    return pd.DataFrame(rows, columns=["chapter", "paragraph", "name"])


def entity_count(entity_table_):
    return entity_table_.groupby("name").count().drop("paragraph", axis=1).rename(columns={"chapter": "count"})


def top_entities(entity_table_, count):
    return entity_count(entity_table_).sort_values("count", ascending=False).head(count)


def get_proust_names():
    chapters = get_proust_chapters()
    return chapters, entity_table(chapters)


def get_references_for_names(entity_table_, names):
    keep = pd.DataFrame(names, columns=["name"])
    return entity_table_.join(keep.set_index("name"), on="name", how="inner")


def get_ref_count_by_chapter(entity_table_, names):
    filtered = get_references_for_names(entity_table_, names)
    counts = filtered.groupby(["name", "chapter"]).count().reset_index().rename({"paragraph": "count"}, axis=1)
    return pd.pivot_table(counts, index="chapter", columns="name", values="count").reset_index().fillna(0)


def smooth_ref_count(ref_count, window=3):
    smoothed = ref_count.rolling(window).mean().fillna(ref_count)
    smoothed["chapter"] = ref_count["chapter"]
    return smoothed


def volume_column(df, chapter="chapter", volume="volume"):
    del volume
    return df[chapter].apply(lambda current_chapter: bisect.bisect_left(volume_starts, current_chapter + 1))


def name_frequency_plot(df, labels=short_labels, starts=volume_starts[:7], transform=None, color_map="Blues", norm=None):
    import matplotlib.pyplot as plt

    xy = np.array(df)
    if transform:
        xy[:, 1:] = transform(xy[:, 1:])
    row_count = xy.shape[1] - 1
    plt.rcParams["figure.figsize"] = 8, row_count
    fig, axs = plt.subplots(nrows=row_count, sharex=True)
    del fig

    x = xy[:, 0]
    extent = [x[0] - (x[1] - x[0]) / 2.0, x[-1] + (x[1] - x[0]) / 2.0, 0, 1]
    for ax_index, ax in enumerate(axs):
        column_index = ax_index + 1
        y = xy[:, column_index]
        ax.imshow(y[np.newaxis, :], cmap=color_map, aspect="auto", extent=extent, norm=norm)
        ax.set_yticks([])
        ax.set_ylabel(df.columns[column_index])
        ax.set_xlim(extent[0], extent[1])
        ax.set_xticks(starts)
        ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.show()


def flatten_islt(islt):
    print("Flattening ISLT chapters.")
    return "\n".join([" \n".join(chapter) for chapter in islt])


def get_islt_nlp(nlp, islt):
    flat_islt = flatten_islt(islt)
    nlp.max_length = len(flat_islt) + 1
    print("Getting spaCy object for ISLT; please wait.")
    return nlp(flat_islt)


def summary_stats(doc):
    stats = []
    stats += [("characters", sum(len(item) for item in doc))]

    words = [token.text.lower() for token in doc if token.is_alpha]
    stats += [("words", len(words))]

    tokens_without_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
    words_without_stop_words = [token.text.lower() for token in tokens_without_stop_words]
    stats += [("words (no stop)", len(words_without_stop_words))]

    lemmas = [token.lemma_.lower() for token in tokens_without_stop_words]
    stats += [("lemmas", len(lemmas))]

    word_freq = Counter(words)
    stats += [("unique words", len(word_freq))]

    word_freq_no_stop = Counter(words_without_stop_words)
    stats += [("unique words (no stop)", len(word_freq_no_stop))]

    lemma_freq = Counter(lemmas)
    stats += [("unique lemma", len(lemma_freq))]

    return pd.DataFrame(stats), word_freq, word_freq_no_stop, lemma_freq


def get_sentiment(islt):
    print(f"getting sentiment for {len(islt)} chapters, please wait...")
    nlp = get_nlp()
    sentiments = [[(len(paragraph), get_doc_sentiment(nlp(paragraph))) for paragraph in chapter] for chapter in islt]
    print("extracting polarity and subjectivity")
    rows = [
        (
            chapter_index,
            paragraph_index,
            sentiment.polarity,
            sentiment.subjectivity,
            sentiment_assessment_count(sentiment),
            paragraph_length,
        )
        for chapter_index, chapter in enumerate(sentiments)
        for paragraph_index, (paragraph_length, sentiment) in enumerate(chapter)
    ]
    return pd.DataFrame(rows, columns=["chapter", "paragraph", "polarity", "subjectivity", "assessed", "length"])


def join_sentiment(entity_table_, sentiment):
    return entity_table_.join(sentiment.set_index(["chapter", "paragraph"]), on=["chapter", "paragraph"])


def agg_sentiment(entity_sentiment, group):
    return entity_sentiment.groupby(group).agg({"chapter": "count", "polarity": "mean", "subjectivity": "mean"}).rename(
        columns={"chapter": "count"}
    ).reset_index()


def get_sentiment_by_name(entity_table_, sentiment):
    return agg_sentiment(join_sentiment(entity_table_, sentiment), "name").sort_values("count", ascending=False)


def get_sentiment_by_volume(entity_table_, sentiment):
    return agg_sentiment(join_sentiment(entity_table_, sentiment), "volume").sort_values("volume")


def get_sentiment_by_name_volume(entity_table_, sentiment, names):
    entity_names = get_references_for_names(entity_table_, names)
    return agg_sentiment(join_sentiment(entity_names, sentiment), ["name", "volume"]).sort_values(["name", "volume"])


def get_polarity_rank(sentiment_by_name, min_count=100):
    return sentiment_by_name.query(f"count >= {min_count}").polarity.rank(pct=True).sort_values()
