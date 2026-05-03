# proust

This repository is now primarily an annotation and downstream-analysis project for *À la recherche du temps perdu* / *In Search of Lost Time*.

The current center of gravity is not generic NLP exploration. It is:

- a canonicalized accepted annotation corpus over the full French text
- corpus-level social-outcome analysis across three lenses:
  - `advantage`: immediate scene-level gain or loss
  - `prestige`: change in visible rank, distinction, or standing
  - `inclusion`: change in belonging, acceptance, or incorporation
- app-facing export layers for the separate `islt` web reader
- experimental derived ranking surfaces such as character ELO on the `advantage` lens

Legacy name-frequency and sentiment utilities still exist in the repo, but they are secondary to the annotation pipeline and the derived artifact surface.

## What The Project Does

The annotation pipeline produces structured literary-social annotations for passage units in the French text. Those annotations are then reduced into aggregate surfaces that answer questions like:

- which characters repeatedly come out ahead or behind?
- which figures split most sharply across `advantage`, `prestige`, and `inclusion`?
- which chapters drive those splits?
- which passages distinguish a chapter or character most strongly?

The accepted corpus is now source-canonicalized for the reviewed same-person naming cases. Current analysis and app-facing exports run directly on that canonicalized source corpus.

## Current Project State

The shortest current handoff docs are:

- [proust/docs/current_state.md](proust/docs/current_state.md)
- [proust/docs/outputs_guide.md](proust/docs/outputs_guide.md)
- [proust/docs/annotation_plan.md](proust/docs/annotation_plan.md)

The project is in post-production canonicalized corpus analysis:

- the canonical full-corpus annotation pass is complete
- the accepted annotation corpus is stable by default
- current work is mainly downstream analysis and app-facing data export

## Main Output Families

The most important artifact families under `outputs/` are:

- `run-*`
  - granular source/output run directories with units, prompts, raw model output, and reduced annotations
- `corpus-review-current.*`
  - default corpus-wide aggregate review surface
- `character-cross-lens-current.*`
  - per-character cross-lens analysis
- `character-chapter-cross-lens-current.*`
  - chapter-by-chapter character analysis
- `character-profile-cards-current.*`
  - app-facing compact character profiles
- `character-pages-current.*`
  - app-facing richer character-page data
- `chapter-overlays-current/`
  - paragraph-range overlay data for the `islt` reader
- `chapter-summaries-current.*`
  - chapter-centered framing and summary data
- `character-elo-advantage-current.*`
  - corpus-wide `advantage`-lens character ELO
- `character-elo-advantage-timeline-current.*`
  - sparse per-unit ELO timeline points for tracked characters

For the full map, see [proust/docs/outputs_guide.md](proust/docs/outputs_guide.md).

## Recommended Reading Order

If you are re-entering the repo and want the shortest useful path:

1. [proust/docs/current_state.md](proust/docs/current_state.md)
2. [outputs/corpus-review-current.md](outputs/corpus-review-current.md)
3. [outputs/character-cross-lens-current.md](outputs/character-cross-lens-current.md)
4. [outputs/character-chapter-cross-lens-current.md](outputs/character-chapter-cross-lens-current.md)

## Core Commands

The CLI entrypoint is:

```bash
python -m proust <command> ...
```

The most important current commands are:

```bash
python -m proust prepare --output outputs/run-999
python -m proust automate --source-run outputs/run-999 --output outputs/run-1000
python -m proust corpus-review --discover-runs outputs
python -m proust character-analysis --discover-runs outputs
python -m proust character-chapter-analysis --discover-runs outputs
python -m proust character-profile-cards --discover-runs outputs
python -m proust character-pages --discover-runs outputs
python -m proust chapter-overlays --discover-runs outputs --output-dir outputs/chapter-overlays-current
python -m proust chapter-summaries --discover-runs outputs
python -m proust character-elo --discover-runs outputs
python -m proust character-elo-timeline --discover-runs outputs
```

Operational note:

- `prepare` only scaffolds a run
- production batch prep still depends on explicit unit specs and the runner workflow described in the docs

For run mechanics, see:

- [proust/docs/annotation_runner.md](proust/docs/annotation_runner.md)
- [proust/docs/full_corpus_runbook.md](proust/docs/full_corpus_runbook.md)

## App-Facing Data Products

This repo now exports data for the separate `islt` app in `/Users/nathan_brixius/dev/brixius-web/app/projects/islt`.

The main app-facing layers are:

- character profile cards
- character pages
- chapter overlays
- chapter summaries

The main handoff docs for that work are:

- [proust/docs/islt_app_integration_ideas.md](proust/docs/islt_app_integration_ideas.md)
- [proust/docs/islt_character_pages_handoff.md](proust/docs/islt_character_pages_handoff.md)
- [proust/docs/islt_chapter_summaries_handoff.md](proust/docs/islt_chapter_summaries_handoff.md)
- [proust/docs/islt_character_elo_handoff.md](proust/docs/islt_character_elo_handoff.md)

## ELO

The project currently computes an ELO-style derived ranking only for the `advantage` lens.

That is intentional. `advantage` maps most directly onto the pairwise “who came out ahead in the scene?” interpretation required by ELO.

Current artifacts:

- [outputs/character-elo-advantage-current.json](outputs/character-elo-advantage-current.json)
- [outputs/character-elo-advantage-timeline-current.json](outputs/character-elo-advantage-timeline-current.json)

The design rationale is documented in:

- [proust/docs/character_elo_plan.md](proust/docs/character_elo_plan.md)

## Legacy NLP Utilities

The repo still contains the older text-analytics surface:

- canonical chapter/text access
- name counting
- sentiment helpers
- simple plotting utilities

Those APIs are still usable from the `proust` package, but they are no longer the best description of the repository as a whole.

Example:

```python
from proust import create_session

session = create_session(model="fr_core_news_sm")
chapters = session.get_canonical_chapters()
structure = session.get_canonical_structure()
```

The canonical reader dataset lives under:

- `data/islt/editions/fr-original/`

## Environment

Core dependencies include:

- `pandas`
- `matplotlib`
- `spacy`
- `spacytextblob`

French model:

- repo-local `.venv/` currently uses `fr_core_news_sm`
- install it with:

```bash
python -m spacy download fr_core_news_sm
```

The larger `fr_core_news_lg` model is still supported if you want it explicitly.

## Tests

The main regression suite for the current annotation/analysis pipeline is:

```bash
pytest tests/test_runner.py -q
```

## Notes

This repo has gone through a real phase change. Earlier README text that described it mostly as a small NLP sandbox is now incomplete. The durable project model is:

- stable accepted annotation corpus
- aggregate literary-social analysis
- rendering-oriented data exports
- targeted experimental derived surfaces built on top of that corpus
