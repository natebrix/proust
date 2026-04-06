# proust
Text analytics on Proust's In Search of Lost Time. You can read my posts on Proust here: https://nathanbrixius.wordpress.com/category/proust/

This repository contains code to:
- Work with a canonical 18-part French text of «À la recherche du temps perdu» by Marcel Proust
- preprocess the French text with spaCy
- count occurrences of proper names
- compute simple sentiment summaries
- produce a small amount of visualization

I am not an expert in NLP / spaCy, so buyer beware.

## Environment

Core dependencies:
- `pandas`
- `matplotlib`
- `spacy`
- `spacytextblob`

French model:
- repo-local `.venv/` currently uses `fr_core_news_sm`
- install it with `python -m spacy download fr_core_news_sm`
- `fr_core_news_lg` is still supported if you want the larger model explicitly

For local development in this repo, there is also a repo-local `.venv/` used by the integration tests.

## API

The entrypoint is the `proust` package with an explicit session object:

```python
from proust import create_session

session = create_session(model="fr_core_news_sm")

chapters = session.get_proust_chapters(1, 10)
entity_table = session.entity_table(chapters)
top = entity_table.groupby("name").count().sort_values("chapter", ascending=False).head(10)
```

A more complete flow:

```python
from proust import create_session, get_ref_count_by_chapter, smooth_ref_count, name_frequency_plot

session = create_session(model="fr_core_news_sm")

islt, entity_table = session.get_proust_names()
names = entity_table.groupby("name").count().sort_values("chapter", ascending=False).head(5).index
ref_count = get_ref_count_by_chapter(entity_table, names)
name_frequency_plot(smooth_ref_count(ref_count))
```

For sentiment:

```python
from proust import create_session

session = create_session(model="fr_core_news_sm")
chapters = session.get_proust_chapters(1, 5)
sentiment = session.get_sentiment(chapters)
```

`get_proust_chapters()` and `get_proust_names()` use the canonical 18-part French structure:

```python
from proust import create_session, canonical_volume_starts, volume_column

session = create_session(model="fr_core_news_sm")

chapters = session.get_canonical_chapters()
structure = session.get_canonical_structure()
chapter = session.get_canonical_chapter("v1-p1-combray")
```

The canonical dataset lives under `data/islt/editions/fr-original/`.
