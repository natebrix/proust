# Run-007 Subset

## Goal

`run-007` is the first modest-scale deployment subset for the current v1 stack.

It is meant to answer a practical question:

- can the current prompt plus reducer produce stable enough structure on a larger socially dense slice of `v1-p1-combray` to support exploratory downstream analysis?

This run is not designed as a new reviewed benchmark.

It is designed as a larger canonical working set.

## Shape of the batch

Total units: `20`

The batch stays inside `v1-p1-combray` and concentrates on three local social clusters:

- the Swann misrecognition cluster
- the Legrandin exposure and Balbec evasion cluster
- the Vinteuil shame and Swann comparison cluster

This keeps the run coherent enough for interpretation while being broad enough to test the pipeline beyond the 10-unit reviewed benchmark.

## Included units

### Swann cluster

- `v1-p1-combray#p-16`
- `v1-p1-combray#p-17`
- `v1-p1-combray#p-18`
- `v1-p1-combray#p-19`
- `v1-p1-combray#p-20`
- `v1-p1-combray#p-21-p-22`
- `v1-p1-combray#p-23-p-24`
- `v1-p1-combray#p-25-p-26`
- `v1-p1-combray#p-27-p-28`
- `v1-p1-combray#p-29-p-30`
- `v1-p1-combray#p-33`

Why this block:

- it extends beyond the first benchmarked Swann windows without leaving the same social field
- it tests narrated prestige, family misrecognition, awkward gratitude, class filtering, and post-marital local diminishment
- it also gives us repeated Swann-centered structure for downstream aggregation

### Legrandin cluster

- `v1-p1-combray#p-270`
- `v1-p1-combray#p-271-p-273`
- `v1-p1-combray#p-274-p-275`
- `v1-p1-combray#p-276-p-277`
- `v1-p1-combray#p-278-p-279`
- `v1-p1-combray#p-280-p-282`
- `v1-p1-combray#p-283-p-285`

Why this block:

- it gives a compact but varied Legrandin dossier
- it includes public abasement, retrospective interpretation, and repeated evasive maneuvers around Balbec and aristocratic access
- it is a good stress test for whether the reducer can keep one dominant movement where the prose proliferates surface cues

### Vinteuil cluster

- `v1-p1-combray#p-310-p-311`
- `v1-p1-combray#p-312-p-313`

Why this block:

- it preserves the existing Vinteuil benchmark terrain
- it keeps one hard mixed case in the larger run
- it gives us a small check on whether the automated stack still handles shame, pity, and socially voiced appraisal without collapsing entirely into the wrong focal target

## Selection rules

This subset deliberately favors passages that are:

- locally socially legible
- still interpretable under the current conservative alias map
- likely to produce recurring event and status patterns rather than one-off edge cases

This subset deliberately avoids:

- very broad family-only scenes that would force a larger alias expansion first
- scenic or reflective passages with little local social movement
- units whose dominant payoff falls outside the current first-pass schema

## Practical use

The intended workflow for `run-007` is:

1. prepare the run directory from these canonical units
2. automate with the current prompt
3. reprocess with `--reduce`
4. inspect the lightweight summary before deciding whether a larger run or a prompt/reducer change is the better next investment
