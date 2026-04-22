# Initial ISLT Subset

## Goal

Start with a tiny, manually inspectable subset that exercises the annotation schema on clearly social material.

The first subset should:

- include named characters already easy to alias
- contain explicit local appraisal or status positioning
- avoid needing large amounts of plot context
- cover more than one kind of social effect

## Chosen starter units

### 1. `v1-p1-combray#p-17`

Source:

- French reader: <https://nathanbrixius.com/projects/islt/fr-original/v1-p1-combray#p-17>
- English reader: <https://nathanbrixius.com/projects/islt/en-moncrieff/v1-p1-combray#p-17>

Why this unit:

- compact
- centered on `Swann`
- explicit narrated elevation through social prestige
- good first test of `narrated_elevation` and `social_status`

Core dynamic:

- the narrator reveals that the family unknowingly hosted a highly prized social figure

### 2. `v1-p1-combray#p-274-p-275`

Source:

- French reader: <https://nathanbrixius.com/projects/islt/fr-original/v1-p1-combray#p-274>
- English reader: <https://nathanbrixius.com/projects/islt/en-moncrieff/v1-p1-combray#p-274>

Why this unit:

- centered on `Legrandin`
- strong local evidence of disavowal, embarrassment, and social aspiration
- good test of irony, indirect evaluation, and `narrative_stance`

Core dynamic:

- Legrandin’s reaction to the name `Guermantes` exposes the snobbery he verbally condemns

### 3. `v1-p1-combray#p-312-p-313`

Source:

- French reader: <https://nathanbrixius.com/projects/islt/fr-original/v1-p1-combray#p-312>
- English reader: <https://nathanbrixius.com/projects/islt/en-moncrieff/v1-p1-combray#p-312>

Why this unit:

- centered on `Swann` and `M. Vinteuil`
- combines praise with social blame
- good test of mixed local positioning and multi-event annotation

Core dynamic:

- Swann is praised as personally exquisite while simultaneously diminished for his marriage

## Why this subset first

This trio covers three different mechanisms:

1. narrated prestige
2. exposed snobbery and evasive self-positioning
3. split appraisal in which a character is elevated in one respect and lowered in another

That is enough variation to stress-test the schema without creating a large evaluation burden.

## Suggested first alias map for this subset

```json
{
  "Swann": {
    "aliases": ["Swann", "M. Swann", "Charles Swann"],
    "notes": "Charles Swann"
  },
  "Legrandin": {
    "aliases": ["Legrandin", "M. Legrandin"],
    "notes": ""
  },
  "Mme de Villeparisis": {
    "aliases": ["Mme de Villeparisis", "Madame de Villeparisis"],
    "notes": ""
  },
  "Mme de Cambremer": {
    "aliases": ["Mme de Cambremer", "Madame de Cambremer"],
    "notes": "Legrandin's sister"
  },
  "M. Vinteuil": {
    "aliases": ["M. Vinteuil", "Vinteuil"],
    "notes": ""
  },
  "la mère du narrateur": {
    "aliases": ["maman", "ma mère", "ma mère"],
    "notes": ""
  }
}
```

## Practical note

The first actual run does not need all three units at once.

If we want one single calibration example, start with:

- `v1-p1-combray#p-17`

It is short, named, and socially explicit.
