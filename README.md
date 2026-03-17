# proust
Text analytics on Proust's In Search of Lost Time. You can read my posts on Proust here: https://nathanbrixius.wordpress.com/category/proust/

This repository contains code to:
- Download the full public domain French text of «À la recherche du temps perdu» by Marcel Proust
- preprocess the French text with spaCy
- count occurrences of proper names
- compute simple sentiment summaries
- produce a small amount of visualization

I am not an expert in NLP / spaCy, so buyer beware.

## Environment

Core dependencies:
- `beautifulsoup4`
- `pandas`
- `matplotlib`
- `spacy`
- `spacytextblob`

French model:
- `python -m spacy download fr_core_news_lg`

For local development in this repo, there is also a repo-local `.venv/` used by the integration tests.

## API

The entrypoint is the `proust` package with an explicit session object:

```python
from proust import create_session

session = create_session(model="fr_core_news_lg")

chapters = session.get_proust_chapters(1, 10)
entity_table = session.entity_table(chapters)
top = entity_table.groupby("name").count().sort_values("chapter", ascending=False).head(10)
```

A more complete flow:

```python
from proust import create_session, get_ref_count_by_chapter, smooth_ref_count, name_frequency_plot

session = create_session(model="fr_core_news_lg")

islt, entity_table = session.get_proust_names()
names = entity_table.groupby("name").count().sort_values("chapter", ascending=False).head(5).index
ref_count = get_ref_count_by_chapter(entity_table, names)
name_frequency_plot(smooth_ref_count(ref_count))
```

For sentiment:

```python
from proust import create_session

session = create_session(model="fr_core_news_lg")
chapters = session.get_proust_chapters(1, 5)
sentiment = session.get_sentiment(chapters)
```
