# Character Alias Normalization Plan

This document records the first reviewed plan for character-name normalization after the full annotation pass.

It is based on:

- [character-alias-audit-current.md](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.md:1)
- [character-alias-audit-current.json](/Users/nathan_brixius/dev/proust/outputs/character-alias-audit-current.json:1)
- the root [aliases.csv](/Users/nathan_brixius/dev/proust/aliases.csv:1)
- run-level alias maps stored in `outputs/run-*/run.json`
- annotation character names found in `outputs/run-*/annotations/*.json`

## Principle

Canonical character names should represent stable person identity.

If the same person appears under different titles, married names, abbreviated names, or local forms, aggregation should treat those names as one character unless there is evidence that the name refers to a different person.

Socially meaningful title or naming differences should remain visible through:

- `surface_forms`
- evidence text
- event type
- status dimension
- status delta
- explanation text
- local passage context

They should not normally be represented by splitting the same person into multiple aggregate character keys.

## Scope

This plan is for aggregate-layer normalization first.

It should not rewrite annotation JSON yet.

The next implementation should produce normalized aggregate artifacts beside the unnormalized ones, so the two surfaces can be compared directly.

## Reviewed Merge Decisions

### Merge Now

| Source Name | Normalized Name | Rationale |
| --- | --- | --- |
| `Saint-Loup` | `Robert de Saint-Loup` | Same person. Most annotations and later run manifests already use `Robert de Saint-Loup`; early uses of `Saint-Loup` should roll into that identity. |
| `princesse des Laumes` | `duchesse de Guermantes` | Same person. The title-stage distinction is socially meaningful, but it belongs in evidence and status effects, not in separate character identity keys. |
| `Charlus` | `baron de Charlus` | Same person. Later run manifests overwhelmingly use `baron de Charlus` as canonical with `Charlus` as an alias. |
| `Mme Swann` | `Odette` | Same person. Married-name/social-position nuance should remain in evidence and explanation, while aggregate identity should be stable. |
| `la grand-mère du narrateur` | `la grand-mère` | Same person. Later run manifests overwhelmingly use `la grand-mère`. |
| `Vinteuil` | `M. Vinteuil` | Same person in the current annotation set. Later run manifests overwhelmingly use `M. Vinteuil`. |
| `Mme de Saint-Euverte` | `marquise de Saint-Euverte` | Same person. Later run manifests use `marquise de Saint-Euverte` as canonical. |

## Implementation Recommendation

Add an optional aggregate-layer character normalization map.

The map should be applied when building scored reports and corpus reviews, not during validation and not by mutating source annotations.

Initial map:

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

Suggested generated artifacts:

- `outputs/corpus-review-current-normalized.json`
- `outputs/corpus-review-current-normalized.md`
- `outputs/corpus-review-normalization-diff.md`

The diff should summarize:

- character-count change
- changed top positive and negative character rankings
- per-lens score movement for normalized characters
- cross-lens disagreement changes

## Guardrails

Do not normalize by loose string matching.

Use only reviewed explicit mappings.

Do not automatically collapse bare title aliases such as `princesse`, `baron`, `duc`, or `la duchesse` unless they are already resolved within a reviewed run-level alias map and have become an annotation character key. Bare titles are useful prompt aliases but risky corpus-level identity keys.

Do not rewrite historical annotation files as the first step. If aggregate-layer normalization proves stable and desirable, source annotation rewriting can be considered separately.
