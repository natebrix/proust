# Character Alias Canonicalization Plan

This document records the current decision about same-character naming splits in the annotation corpus.

It supersedes the earlier conservative plan that treated normalization primarily as a downstream aggregate-layer feature.

The project now prefers a cleaner model:

- accepted annotation JSON should represent the project's stable canonical understanding of who each character is
- same-person identity splits should not remain in source annotations once they are reviewed and judged safe
- surface nuance should remain in `surface_forms`, evidence, explanation, and local context, not in duplicate identity keys

This plan is based on:

- [character-alias-audit-current.md](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.md:1)
- [character-alias-audit-current.json](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.json:1)
- the root [aliases.csv](/Users/nathan_brixius/dev/proust/aliases.csv:1)
- run-level alias maps stored in `outputs/run-*/run.json`
- accepted annotation character names found in `outputs/run-*/annotations/*.json`

## Canonical Principle

Canonical character names should represent stable person identity inside the accepted annotation corpus.

If the same person appears under:

- a title variant
- a married name
- a shorter social form
- an expanded personal form

the accepted annotation surface should normally use a single reviewed canonical identity.

Naming nuance should remain visible through:

- `surface_forms`
- evidence text
- explanation text
- event type
- status dimension
- local passage context

It should not normally survive as a split identity in accepted annotation JSON.

## Decision Rule

A merge is safe for upstream canonicalization when all of the following are true:

- the two names clearly refer to the same person
- the distinction is one of naming or title, not person identity
- the aggregate interpretation already treats them as one person
- keeping them split upstream adds noise rather than preserving meaningful ambiguity

If a merge fails any of those conditions, it should stay out of the source rewrite set.

## Reviewed Map Audit

The current reviewed map is:

```json
{
  "Saint-Loup": "Robert de Saint-Loup",
  "princesse des Laumes": "duchesse de Guermantes",
  "Charlus": "baron de Charlus",
  "Mme Swann": "Odette",
  "la grand-mère du narrateur": "la grand-mère",
  "Vinteuil": "M. Vinteuil",
  "Mme de Saint-Euverte": "marquise de Saint-Euverte"
}
```

After review, the current judgment is:

- all seven entries are safe for upstream canonicalization
- none of the current reviewed entries need to remain downstream-only
- none of the current reviewed entries require a third `needs discussion` bucket before rewrite

## Entry-by-Entry Decisions

### Safe To Canonicalize Upstream Now

| Source Name | Canonical Name | Reason |
| --- | --- | --- |
| `Saint-Loup` | `Robert de Saint-Loup` | Pure naming consistency. The audit shows overwhelmingly dominant later usage of `Robert de Saint-Loup`, and the split does not preserve a distinct person-level meaning. |
| `princesse des Laumes` | `duchesse de Guermantes` | Same person. The title-stage distinction is socially meaningful, but it belongs in evidence and local interpretation rather than in separate corpus identity keys. |
| `Charlus` | `baron de Charlus` | Pure naming consistency. The current split is an identity artifact, not a substantive distinction. |
| `Mme Swann` | `Odette` | Same person. The married-name/social-position nuance is real, but it is better preserved in passage context than as a separate aggregate character identity. |
| `la grand-mère du narrateur` | `la grand-mère` | Same person. The longer form is descriptive, not identity-bearing in a way that justifies a split. |
| `Vinteuil` | `M. Vinteuil` | Same person in the current annotation set. The shorter form does not preserve a different character identity. |
| `Mme de Saint-Euverte` | `marquise de Saint-Euverte` | Same person. This is a title-form consistency issue, not a person distinction. |

## Why The Earlier Downstream-Only Phase Was Still Useful

The earlier aggregate-layer normalization was still the right transitional step because it let the project:

- test the explicit reviewed merges without rewriting accepted annotations immediately
- compare normalized and unnormalized aggregate outputs directly
- confirm that the merges cleaned identity noise without introducing interpretive surprises

That staging phase did its job.

Now that the reviewed map has been tested and accepted, keeping the split identities upstream creates more conceptual clutter than value.

## Rewrite Scope

The upstream rewrite should touch accepted annotation JSON wherever a reviewed source-side character identity appears as a canonical character key.

That includes:

- `characters_present[].canonical_name`
- `appraisal_events[].source` when the source is a character name
- `appraisal_events[].target` when the target is a character name
- `status_effects[].character`

It should not rewrite:

- evidence text
- explanation text
- `surface_forms`
- historical prompt inputs
- raw model responses

Those fields should continue to preserve local wording and social nuance.

## Run Metadata

Run-level alias maps may still remain useful for prompt/reference hygiene even after accepted annotations are canonicalized upstream.

But the accepted annotation payload should stop relying on a downstream-only identity merge layer for these seven reviewed cases.

## Migration Sequence

1. Treat the seven reviewed entries above as the canonical source rewrite set.
2. Rewrite accepted annotation JSON for those entries only.
3. Regenerate all current downstream artifacts from the rewritten corpus.
4. Compare those regenerated artifacts against the historical normalized aggregate surface.
5. If they match in all material respects, remove the now-redundant downstream normalization layer from normal project use.

Status:

- completed
- the regenerated current artifacts matched the historical normalized comparison surface in all material respects

## Verification Standard

The rewrite should be accepted only if the regenerated source-canonicalized corpus reproduces the current normalized surfaces in all important ways:

- same aggregate character counts
- same top positive and negative characters by lens
- same cross-lens character profiles
- same character pages
- same chapter overlays and chapter summaries
- same ELO outputs up to expected deterministic identity renaming

That equivalence now holds, so the downstream normalization layer should be treated as historical scaffolding rather than an active conceptual layer.

## End State

The desired end state is:

- accepted annotation JSON already uses stable canonical identities for these reviewed same-person cases
- aggregate exports no longer need an optional character-normalization flag for them
- alias handling remains useful for prompt/reference hygiene, but not as a compensating layer for accepted annotation identity splits

That is the cleaner and more readable model for the project going forward.
