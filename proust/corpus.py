import json

import pandas as pd

from .nlp import get_nlp
from .paths import ALIASES_CSV, ISLT_EDITIONS_DIR, REPO_ROOT
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


def get_proust_chapters(id_start=1, id_end=None, use_aliases=True, aliases=None):
    return get_canonical_chapters(
        id_start=id_start,
        id_end=id_end,
        use_aliases=use_aliases,
        aliases=aliases,
    )
