# Annotation Unit

## Purpose

Define the exact text unit that will be sent to the prompt during the first annotation phase.

The unit should be:

- deterministic
- small enough for close reading
- large enough to capture a coherent local appraisal or status movement
- easy to trace back to the source text

## First-pass unit choice

Use a **canonical chapter paragraph window**.

That means each annotation unit is a contiguous span of paragraphs inside one canonical Proust chapter such as `v1-p1-combray`.

This is the right first unit because:

- it matches the reader and blog coordinate system
- paragraph boundaries are easy to extract deterministically
- the window can stay small while still preserving local context
- provenance is simple and auditable

## Recommended window size

Default:

- `1` to `2` contiguous paragraphs

Allowed for first-pass experiments:

- expand to `3` paragraphs if the evaluative move clearly spills across the break

Avoid larger windows until the prompt behavior is stable.

## Required fields

Each annotation unit should carry at least:

```json
{
  "unit_id": "v1-p1-combray#p-27-p-28",
  "source_text": "islt_fr",
  "chapter_id": "v1-p1-combray",
  "paragraph_start": 27,
  "paragraph_end": 28,
  "raw_text": "original paragraph text joined in order",
  "preprocessed_text": "text after punctuation cleanup and alias replacement policy",
  "reader_urls": {
    "fr-original": "https://nathanbrixius.com/projects/islt/fr-original/v1-p1-combray#p-27",
    "en-moncrieff": "https://nathanbrixius.com/projects/islt/en-moncrieff/v1-p1-combray#p-27"
  },
  "prior_context": "optional immediately preceding window text",
  "notes": "optional human note"
}
```

## Field conventions

### `unit_id`

Use a stable, readable id:

`{chapter_id}#p-{start}` for a single paragraph

or

`{chapter_id}#p-{start}-p-{end}` for a span

Examples:

- `v1-p1-combray#p-27`
- `v1-p1-combray#p-28-p-29`
- `v1-p1-combray#p-74-p-75`

### `raw_text`

Join paragraphs in source order with a blank line between them.

### `preprocessed_text`

For the first pass:

- keep punctuation cleanup
- apply alias normalization only where the alias map clearly supports it
- do not aggressively rewrite the passage

### `prior_context`

Optional.

If included, use only the immediately preceding window from the same local scene.

Keep it short.

Its purpose is disambiguation, not broader interpretation.

## Granularity rule

Choose units based on **one coherent local social or evaluative move**.

Good units:

- a family’s demeaning or elevating treatment of Swann
- Legrandin’s evasive self-positioning around aristocratic names
- a passage where one character is clearly favored, slighted, admired, or diminished

Bad units:

- long descriptive passages with no named-character dynamics
- windows that require multiple pages of prior plot to interpret
- mechanically fixed slices that cut one social move in half

## Mixed vs Too Broad

Not every unit should collapse to a single event.

Some passages are genuinely **mixed**: they hold two opposed or distinct local movements together, and the tension between them is the point.

Other passages are simply **too broad**: they contain adjacent but separable movements that would annotate more cleanly as smaller windows.

Use this rule of thumb:

- keep one unit when the movements are tightly interdependent
- split the unit when the movements are merely sequential

Keep a larger unit when:

- the opposed movements concern the same focal character
- they belong to the same rhetorical or social setup
- separating them would flatten the local contradiction the passage is staging

Split into smaller units when:

- the movements target different focal characters
- one movement is mostly complete before the next begins
- each smaller span would yield a cleaner single-movement annotation

Examples:

- `v1-p1-combray#p-312-p-313` is a defensible mixed unit because Swann is simultaneously admired and socially diminished, and that doubleness is the point of the passage.
- `v1-p1-combray#p-21-p-22` should remain a single dominant-movement unit, because the narrator’s counterpoint mainly sharpens the family’s slight rather than creating an independent second event.
- `v1-p1-combray#p-274-p-275` should remain a single dominant-movement unit, because the narrator’s various signs of exposure all belong to one Legrandin diminishment.

## Output relationship

Each annotation output should preserve the originating canonical `unit_id`.

That gives a stable bridge from:

- source text
- prompt input
- raw model output
- normalized annotation
- later scoring layers
- reader URLs in French and English
