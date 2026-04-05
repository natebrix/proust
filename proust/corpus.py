import json
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup

from .nlp import get_nlp
from .paths import ALIASES_CSV, DATA_DIR, ISLT_EDITIONS_DIR, REPO_ROOT
from .state import get_loaded_aliases
from .state import set_aliases


def read_aliases(path=ALIASES_CSV):
    aliases = pd.read_csv(path, header=None, index_col=0).iloc[:, 0].to_dict()
    print(f"{len(aliases)} aliases read from aliases.csv")
    return aliases


def read_json(path):
    return json.loads(path.read_text())


def get_aliases():
    aliases = get_loaded_aliases()
    if aliases is None:
        aliases = read_aliases()
        set_aliases(aliases)
    return aliases


def get_proust_page(id, source="file"):
    if source == "file":
        file_path = DATA_DIR / f"islt_fr_{id:03}.html"
        print(f"Loading page from file: {file_path.relative_to(DATA_DIR.parent.parent)}")
        return file_path.read_text()
    if source == "web":
        url = f"https://marcel-proust.com/marcelproust/{id:03}"
        print(f"Retreiving page from web: {url}")
        return urlopen(url).read()
    raise ValueError(f'Invalid source "{source}". Valid sources = file, web.')


def write_proust_pages(start=1, end=486, source="web"):
    for id in range(start, end + 1):
        page = get_proust_page(id, source)
        file_path = DATA_DIR / f"islt_fr_{id:03}.html"
        print(f"Writing file {file_path.relative_to(DATA_DIR.parent.parent)}")
        with file_path.open("wb") as text_file:
            text_file.write(page)


def get_chapter_info(page):
    soup = BeautifulSoup(page, features="html.parser")
    return soup.body.find("h1").text


def get_sentences(text, nlp=None):
    pipeline = nlp or get_nlp()
    return [sent for sent in pipeline(text).sents]


def preprocess(text, use_aliases=True, aliases=None):
    text = text.replace("; –", ";").replace("– ;", ";")
    if use_aliases:
        replacements = aliases if aliases is not None else get_aliases()
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
    return text


def get_chapter_body(html):
    soup = BeautifulSoup(html, features="html.parser")
    return soup.body.find("div", attrs={"class": "field-item"})


def get_proust_pages(id_start=1, id_end=486, source="file"):
    return [get_proust_page(id, source) for id in range(id_start, id_end + 1)]


def get_proust_chapters(id_start=1, id_end=None, source="file", use_aliases=True, by_paragraph=True, aliases=None):
    del by_paragraph
    if source in {"canonical", "file"}:
        return get_canonical_chapters(
            id_start=id_start,
            id_end=id_end,
            use_aliases=use_aliases,
            aliases=aliases,
        )

    if id_end is None:
        id_end = 486

    if source not in {"legacy-file", "web"}:
        raise ValueError(f'Invalid source "{source}". Valid sources = canonical, file, legacy-file, web.')

    return [
        [
            preprocess(paragraph.text, use_aliases, aliases=aliases)
            for paragraph in get_chapter_body(get_proust_page(id, "file" if source == "legacy-file" else source)).find_all("p")
        ]
        for id in range(id_start, id_end + 1)
    ]


def get_paragraphs(chapter, nlp=None, aliases=None):
    paragraphs = chapter.find_all("p")
    rows = [
        [paragraph_number, sentence_number, sentence.text]
        for paragraph_number, paragraph in enumerate(paragraphs)
        for sentence_number, sentence in enumerate(get_sentences(preprocess(paragraph.text, aliases=aliases), nlp=nlp))
    ]
    return pd.DataFrame(rows, columns=["paragraph", "sentence", "text"])


def get_canonical_structure(edition="fr-original"):
    structure_path = ISLT_EDITIONS_DIR / edition / "canonical-structure.json"
    print(f"Loading canonical structure from file: {structure_path.relative_to(REPO_ROOT)}")
    return read_json(structure_path)


def get_canonical_chapter(chapter_id, edition="fr-original"):
    chapter_path = ISLT_EDITIONS_DIR / edition / "chapters" / f"{chapter_id}.json"
    print(f"Loading canonical chapter from file: {chapter_path.relative_to(REPO_ROOT)}")
    return read_json(chapter_path)


def get_canonical_chapter_ids(edition="fr-original"):
    return [chapter["id"] for chapter in get_canonical_structure(edition=edition)]


def get_canonical_chapters(id_start=1, id_end=None, edition="fr-original", use_aliases=True, aliases=None):
    chapter_ids = get_canonical_chapter_ids(edition=edition)

    if id_end is None:
        id_end = len(chapter_ids)

    selected_ids = chapter_ids[id_start - 1:id_end]

    return [
        [
            preprocess(paragraph["text"], use_aliases=use_aliases, aliases=aliases)
            for paragraph in get_canonical_chapter(chapter_id, edition=edition)["paragraphs"]
        ]
        for chapter_id in selected_ids
    ]
