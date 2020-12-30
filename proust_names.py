# proust_names.py / Dec 2020 / Nathan Brixius @natebrix
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
#   bs4, pandas, spacy, matplotlib
# also, you will need the largest french-language spacy model:
# python -m spacy download fr_core_news_lg
#
#
# *** how to run me:
# 1) one time only: time: write_proust_pages()
# 2) islt, et = get_proust_names()
# 3) rc = get_ref_count_by_chapter(et, ['Albertine', 'Charlus'])
# 4) name_frequency_plot(rc, volume_starts)

from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
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

    
def entities(text):
    ents = [e for e in text.ents if is_entity_pos(e)]
    return [canonicalize_entity(e, i, ents) for i, e in enumerate(ents)]


def load_spacy(model='fr_core_news_lg'):
    print(f'loading spacy model {model}')
    nlp = spacy.load(model)
    nlp.add_pipe(proust_sentence_start, before='parser')
    nlp.add_pipe(proust_proper_name, before='parser')
    nlp.add_pipe(merge_entities) # ['Saint', 'Loup'] -> ['Saint Loup']
    return nlp

nlp = load_spacy()

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
def preprocess(text):
    t = text.replace("; –", ";").replace("– ;", ";")
    for a in aliases:
        t = t.replace(a, aliases[a])
    return t


def get_chapter_body(html):
    soup = BeautifulSoup(html)
    chapter = soup.body.find('div', attrs={'class':'field-item'})
    return chapter


def get_proust_pages(id_start=1, id_end=486, source='file'):
    return [get_proust_page(id, source) for id in range(id_start, id_end + 1)]


def get_proust_chapters(id_start=1, id_end=486, source='file', by_paragraph=True):
    # todo use by_paragraph
    return [[preprocess(p.text) for p in get_chapter_body(get_proust_page(id, source)).find_all('p')] for id in range(id_start, id_end + 1)]


def get_paragraphs(chapter):
    paragraphs = chapter.find_all('p')
    # todo get the 'book' and 'chapter'
    data = [[par_num, sent_num, sentence.text] for par_num, paragraph in enumerate(paragraphs) for sent_num, sentence in enumerate(get_sentences(preprocess(paragraph.text)))]
    return pd.DataFrame(data, columns=['paragraph', 'sentence', 'text'])


# get all occurrences of entities within ISLT
def entity_table(chapters):
    rows = []
    for index_chapter, chapter in enumerate(chapters):
        for index_para, para in enumerate(chapter):
            rows.extend([[index_chapter, index_para, e] for e in entities(nlp(para))])
    names = pd.DataFrame(rows, columns=['chapter', 'paragraph', 'name_core'])
    names['name_pure'] = names['name_core'].apply(lambda n: n.replace('!', ' ').replace('–', ' ').strip())
    names['name'] = names['name_pure'].apply(lambda n: aliases[n] if n in aliases else n)
    return names


# get entity counts
def entity_count(et):
    return et.groupby('name').count()


# run write_proust_pages() before you run me (to download the text)
def get_proust_names():
    islt = get_proust_chapters()
    et = entity_table(islt)
    return islt, et


# Get the number of references to each name by chapter, given an entity table
def get_ref_count_by_chapter(et, names):
    keep = pd.DataFrame(names, columns=['name'])
    et2 = et.join(keep.set_index('name'), on='name', how='inner')
    et3 = et2.groupby(['name', 'chapter']).count().reset_index()
    return pd.pivot_table(et3, values='name_core', index=['chapter'], columns=['name']).reset_index().fillna(0)


# the first chapter of each of the seven volumes
volume_starts = [0, 102, 182, 265, 334, 390, 432]

# Create a plot of frequency of character references by chapter
# df is assumed to be a data frame whose first column is chapter
# and the remaining columns are reference counts for different characters.
# You can get such a table by calling get_ref_count_by_chapter
def name_frequency_plot(df, starts):
    xy = np.array(df)
    xy[:, 1:] = np.sqrt(xy[:, 1:]) # looks better...
    row_count = xy.shape[1] - 1
    plt.rcParams["figure.figsize"] = 8,5

    # thanks stackoverflow:
    #  https://stackoverflow.com/questions/45841786/creating-a-1d-heat-map-from-a-line-graph
    fig, axs = plt.subplots(nrows=row_count, sharex=True)

    x = xy[:, 0]
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    for ax_i, ax in enumerate(axs):
        i = ax_i + 1
        y = xy[:, i]
        ax.imshow(y[np.newaxis,:], cmap="YlGnBu", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_ylabel(df.columns[i])
        ax.set_xlim(extent[0], extent[1])
        ax.set_xticks(starts)
        ax.set_xticklabels(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'])
    plt.tight_layout()
    plt.show()

def count_unique_words(islt):
    all_c = [" ".join(c) for c in islt]
    all_islt = " ".join([c for c in all_c])
    nlp.max_length= len(all_islt)+1 # haha
    # word_freq_2 = Counter([tok.text.lower() for tok in nlp(all_islt) if tok.is_alpha and not tok.is_stop])
    word_freq = Counter([tok.text.lower() for tok in nlp(all_islt)]) 
    #cb = nlp(all_islt).count_by(ORTH) # get a cup of coffee
    return word_freq
