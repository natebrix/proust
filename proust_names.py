# proust_names.py / Jan 2021 / Nathan Brixius @natebrix
#
# Counts occurrences of proper names in «À la recherche du temps perdu» by
# Marcel Proust.
#
# *** Consider this work in progress. ****
#
# *** what this is:
# - code to read the original French text
# - spacy code to preprocess
# - spacy code to extract entities
#
# I am not an expert in NLP / spacy, so buyer beware.
#
# *** packages you'll need:
#   bs4, pandas, spacy, matplotlib, spacytextblob
# also, you will need the largest french-language spacy model:
# python -m spacy download fr_core_news_lg
#
#
# *** how to run me:
# 1) islt, et = get_proust_names()
# 2) names = top_entities(et, 5).index
# 3) rc = get_ref_count_by_chapter(et, names)
# 4) name_frequency_plot(rc, volume_starts)

# moving average of sentiment over time
# plot sentiment as color bar

from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.pipeline import merge_entities, merge_noun_chunks
from urllib.request import urlopen

# I encountered situations where long sentences were broken up by spacy
# at semicolons.
def proust_sentence_start(doc):
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text == ';' and index+1 < length:
            doc[index+1].sent_start = False
    return doc

entity_pos = set(['NOUN', 'PROPN'])
proper_name_exceptions = set(['Êtes', 'Oh', 'Ah', 'Suave', 'Viens']) # todo text file
honorific = set(['M.', 'Mme', 'Mlle', 'Monsieur', 'Madame']) 
conjunction = set(['et'])

def read_aliases():
    aliases = pd.read_csv('aliases.csv', header=None, index_col=0, squeeze=True).to_dict()
    print(f'{len(aliases)} aliases read from aliases.csv')
    return aliases

aliases = read_aliases()

# family name for entity if indirectly determined;
# e.g. for the "M." in "M. et Mlle Verdurin"
spacy.tokens.token.Token.set_extension('family', default='', force=True) 

# This is a bit wonky but sort of works. Annotate tokens so that proper names
# can be extracted in certain corner cases.
#
# e.g. M. et Mme. Foobar should expand to M. Foobar and Mme. Foobar.
def proust_proper_name(doc, verbose=False):
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text in honorific:
            # We sometimes have pairs of names like this:
            # 0   M.
            # 1   et
            # 2   Mme Verdurin
            # (note token 2 has both the honorific and the family name.)
            #
            # We want to note this for entity extraction.
            if index+3 < length and doc[index+1].text in conjunction and doc[index+2].text in honorific:
                token._.family = 'next'
            else: # e.g. M. de Charlus
                token._.family = 'skip'
        
        if token.pos_ == 'PROPN':
            case = None
            if token.text in proper_name_exceptions:
                case = 'EXCEPTION'
            elif len(token.text) and not token.text[0].isupper():
                case = 'IS_LOWERCASE'
            if case:
                token.pos_ = 'NOUN'
                if verbose:
                    print(f'Changed {token} from proper noun to noun for reason {case}')
    return doc

def get_family_name(e, token):
    return ' '.join(token.text.split(' ')[1:]) if token.text.find(' ') >= 0 else token.text


def is_entity_pos(e):
    return e.root.pos_ in entity_pos


def canonicalize_entity(e, i, ents):
    if e.root._.family == 'next':
        return e.text + ' ' + get_family_name(e, ents[i+1])
    elif e.root._.family == 'skip':
        return e.text # no longer using this; should go back and fix
    else:
        return e.text

# Get a list of all entity names in the given text.    
def entities(text):
    ents = [e for e in text.ents if is_entity_pos(e)]
    return [canonicalize_entity(e, i, ents) for i, e in enumerate(ents)]


def words_in_list(text, words):
    return [e.text.lower() for i, e in enumerate(text) if e.text.lower() in words]


def load_spacy(model='fr_core_news_lg'):
    print(f'loading spacy model {model}')
    spacy_text_blob = SpacyTextBlob()
    nlp = spacy.load(model)
    nlp.add_pipe(proust_sentence_start, before='parser')
    nlp.add_pipe(proust_proper_name, before='parser')
    nlp.add_pipe(merge_entities) # ['Saint', 'Loup'] -> ['Saint Loup']
    nlp.add_pipe(spacy_text_blob) # for sentiment
    return nlp

nlp = load_spacy()

# get a single HTML page of ISLT, either from 'file' or 'web'.
def get_proust_page(id, source='file'):
    if source == 'file':
        file = f'data/fr/islt_fr_{id:03}.html'
        print(f'Loading page from file: {file}')
        return open(file).read()
    elif source == 'web':
        url = f'https://marcel-proust.com/marcelproust/{id:03}'
        print(f'Retreiving page from web: {url}')
        return urlopen(url).read()
    else:
        raise ValueError(f'Invalid source "{source}". Valid sources = file, web.')


# write the contents of ISLT pages to files.
def write_proust_pages(start=1, end=486, source='web'):
    for id in range(start, end+1):
        page = get_proust_page(id, source)
        file_name = f'data/fr/islt_fr_{id:03}.html'
        print(f'Writing file {file_name}')
        with open(file_name, 'wb') as text_file:
            text_file.write(page)

            
def get_chapter_info(page):
    soup = BeautifulSoup(page)
    title = soup.body.find('h1').text 
    # re.split('\[(.*?)\]', t)
    # this will be either length 3 or length 1. If length 3, have 'book' header
    return title


def get_sentences(text):
    return [sent for sent in nlp(text).sents]


# hacks to help with parsing
def preprocess(text, use_aliases=True):
    t = text.replace("; –", ";").replace("– ;", ";")
    if use_aliases:
        for a in aliases:
            t = t.replace(a, aliases[a])
    return t


def get_chapter_body(html):
    soup = BeautifulSoup(html)
    chapter = soup.body.find('div', attrs={'class':'field-item'})
    return chapter


def get_proust_pages(id_start=1, id_end=486, source='file'):
    return [get_proust_page(id, source) for id in range(id_start, id_end + 1)]


def get_proust_chapters(id_start=1, id_end=486, source='file', use_aliases=True, by_paragraph=True):
    # todo use by_paragraph
    return [[preprocess(p.text, use_aliases) for p in get_chapter_body(get_proust_page(id, source)).find_all('p')] for id in range(id_start, id_end + 1)]


def get_paragraphs(chapter):
    paragraphs = chapter.find_all('p')
    # todo get the 'book' and 'chapter'
    data = [[par_num, sent_num, sentence.text] for par_num, paragraph in enumerate(paragraphs) for sent_num, sentence in enumerate(get_sentences(preprocess(paragraph.text)))]
    return pd.DataFrame(data, columns=['paragraph', 'sentence', 'text'])


# get all occurrences of entities within ISLT
def entity_table(chapters):
    print(f'Finding entities within {len(chapters)} chapters.')
    rows = []
    for index_chapter, chapter in enumerate(chapters):
        for index_para, para in enumerate(chapter):
            rows.extend([[index_chapter, index_para, e] for e in entities(nlp(para))])
    names = pd.DataFrame(rows, columns=['chapter', 'paragraph', 'name'])
    #names['name_pure'] = names['name_core'].apply(lambda n: n.replace('!', ' ').replace('–', ' ').strip())
    #names['name'] = names['name_pure'].apply(lambda n: aliases[n] if n in aliases else n)
    return names

def word_freq_table(chapters, words):
    print(f'Finding word frequency within {len(chapters)} chapters.')
    rows = []
    for index_chapter, chapter in enumerate(chapters):
        for index_para, para in enumerate(chapter):
            rows.extend([[index_chapter, index_para, e] for e in words_in_list(nlp(para), words)])
    names = pd.DataFrame(rows, columns=['chapter', 'paragraph', 'name'])
    return names

# get entity counts
def entity_count(et):
    return et.groupby('name').count().drop('paragraph',axis=1).rename(columns={'chapter':'count'})


def top_entities(et, count):
    return entity_count(et).sort_values('count', ascending=False).head(count)


# run write_proust_pages() before you run me (to download the text)
def get_proust_names():
    islt = get_proust_chapters()
    et = entity_table(islt)
    return islt, et


def get_references_for_names(et, names):
    keep = pd.DataFrame(names, columns=['name'])
    return et.join(keep.set_index('name'), on='name', how='inner')


# Get the number of references to each name by chapter, given an entity table
def get_ref_count_by_chapter(et, names):
    et2 = get_references_for_names(et, names)
    et3 = et2.groupby(['name', 'chapter']).count().reset_index().rename({'paragraph':'count'}, axis=1)
    return pd.pivot_table(et3, index='chapter', columns='name', values='count').reset_index().fillna(0)


# smooth a ref_count dataframe for visualization
def smooth_ref_count(rc, window=3):
    rcs = rc.rolling(window).mean().fillna(rc) 
    rcs['chapter'] = rc['chapter']
    return rcs


# the first chapter of each of the seven volumes, plus a sentinel
volume_starts = [0, 102, 182, 265, 334, 390, 432, 486]
short_labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']


def volume_column(df, chapter='chapter', volume='volume'):
    return df[chapter].apply(lambda c: bisect.bisect_left(volume_starts, c+1))

    
# Create a plot of frequency of character references by chapter
# df is assumed to be a data frame whose first column is chapter
# and the remaining columns are reference counts for different characters.
# You can get such a table by calling get_ref_count_by_chapter
#
# smoothing a bit looks nice. use smooth_ref_count for that.
def name_frequency_plot(df, labels=short_labels, starts=volume_starts[:7], transform=None, color_map='Blues', norm=None):
    xy = np.array(df)
    if transform:
        xy[:, 1:] = transform(xy[:, 1:]) 
    row_count = xy.shape[1] - 1
    plt.rcParams["figure.figsize"] = 8,row_count

    # thanks stackoverflow:
    #  https://stackoverflow.com/questions/45841786/creating-a-1d-heat-map-from-a-line-graph
    fig, axs = plt.subplots(nrows=row_count, sharex=True)

    x = xy[:, 0]
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    for ax_i, ax in enumerate(axs):
        i = ax_i + 1
        y = xy[:, i]
        ax.imshow(y[np.newaxis,:], cmap=color_map, aspect="auto", extent=extent, norm=norm)
        ax.set_yticks([])
        ax.set_ylabel(df.columns[i])
        ax.set_xlim(extent[0], extent[1])
        ax.set_xticks(starts)
        ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.show()

    
# flatten the chapters of ISLT into a single string.
def flatten_islt(islt):
    print('Flattening ISLT chapters.')
    all_c = [" \n".join(c) for c in islt]
    all_islt = "\n".join([c for c in all_c])
    return all_islt


# return the entire text of ISLT as a single spacy object.
def get_islt_nlp(nlp, islt):
    flat_islt = flatten_islt(islt)
    nlp.max_length = len(flat_islt) + 1 # haha
    print('Getting spaCy object for ISLT; please wait.')
    return nlp(flat_islt)


# return summary statistics for doc.
def summary_stats(doc):
    stats = []

    stats += [('characters', sum([len(i) for i in doc]))]
    
    words = [tok.text.lower() for tok in doc if tok.is_alpha]
    stats += [('words', len(words))]

    toks_no_stop = [tok for tok in doc if tok.is_alpha and not tok.is_stop]
    words_no_stop = [tok.text.lower() for tok in toks_no_stop]
    stats += [('words (no stop)', len(words_no_stop))]
    
    lemmas = [tok.lemma_.lower() for tok in toks_no_stop]
    stats += [('lemmas', len(lemmas))]

    word_freq = Counter(words)
    stats += [('unique words', len(word_freq))]

    word_freq_no_stop = Counter(words_no_stop)
    stats += [('unique words (no stop)', len(word_freq_no_stop))]

    lemma_freq = Counter(lemmas)
    stats += [('unique lemma', len(lemma_freq))]

    df = pd.DataFrame(stats)
    return df, word_freq, word_freq_no_stop, lemma_freq


# uses spacytextblob to get sentiment information for each paragraph
# This takes awhile to run...
def get_sentiment(islt):
    print(f'getting sentiment for {len(islt)} chapters, please wait...')
    ss = [[(len(para), nlp(para)._.sentiment) for para in chap] for chap in islt]
    print(f'extracting polarity and subjectivity')
    ss2 = [(c, p, s[1].polarity, s[1].subjectivity, len(s[1].assessments), s[0]) for c, ss_c in enumerate(ss) for p, s in enumerate(ss_c)]
    return pd.DataFrame(ss2, columns=['chapter', 'paragraph', 'polarity', 'subjectivity', 'assessed', 'length'])


# joins entity and sentiment tables
def join_sentiment(et, sentiment):
    return et.join(sentiment.set_index(['chapter', 'paragraph']), on=['chapter', 'paragraph'])


# aggregates per-chapter sentiment by the specified key(s). The chapter count is given as the
# 'count' column
def agg_sentiment(ets, group):
    return ets.groupby(group).agg({'chapter': 'count', 'polarity': 'mean', 'subjectivity':'mean'}).rename(columns={'chapter':'count'}).reset_index()


# not all text has tokens with identified sentiment. Here's how to filter those out:
# sentiment.query('assessed!=0')


# get per-chapter sentiment by name
def get_sentiment_by_name(et, sentiment):
    ets = join_sentiment(et, sentiment)
    return agg_sentiment(ets, 'name').sort_values('count', ascending=False)


# get sentiment by volume
def get_sentiment_by_volume(et, sentiment):
    # restrict et to names
    ets = join_sentiment(et, sentiment)
    return agg_sentiment(ets, 'volume').sort_values('volume')

# get sentiment by name-volume pair
def get_sentiment_by_name_volume(et, sentiment, names):
    # restrict et to names
    et_name = get_references_for_names(et, names)
    ets = join_sentiment(et_name, sentiment)
    return agg_sentiment(ets, ['name', 'volume']).sort_values(['name', 'volume'])


def get_polarity_rank(s_by_n, min_count=100):
    s_top = s_by_n.query(f'count >= {min_count}')
    return s_top.polarity.rank(pct=True).sort_values()

