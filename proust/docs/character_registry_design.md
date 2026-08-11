# Character Registry Design

Status: proposal + working implementation (2026-08). Companion artifacts:
`characters.yaml`, `proust/registry.py`, `scripts/build_character_registry.py`,
`scripts/character_registry_audit.py`, `tests/test_character_registry.py`,
`outputs/character-registry-audit-v2.md`.

## Why

The audit found four coupled problems in the previous alias system:

1. **Three alias layers, no owner.** Root `aliases.csv` (manual, pre-LLM),
   per-run `alias_map` in `run.json` (grown ad hoc to 52–119 entries), and the
   reviewed downstream merge map. They disagree and none is authoritative.
2. **Closed-world roster.** Prompt scope rule 17 ("Use only named characters
   that appear in the alias map") made the alias map a roster: anyone unmapped
   was structurally invisible. Rachel: **44 latest annotated units mention her
   187 times; she appears in `characters_present` zero times** — including her
   bal-de-têtes triumph over la Berma, whose side of the duel *was* scored.
   Same class: Mlle Vinteuil, Mme de Marsantes, the prince de Guermantes,
   Céleste, princesse Sherbatoff, the dame-en-rose units of Odette.
3. **Dangerous rewrite rules.** The substitution *machinery*
   (`apply_alias_replacements`) is sound; the *rows* were not: context-free
   descriptor rules ("la duchesse" live in 404 runs, bare "princesse" in runs
   067–071, "le docteur"-class, "Marsantes"-class). Result: 630 annotated unit
   files carry severe substitutions (descriptor/mangled), 556 carry
   name-variant flattening (e.g. "Mme de Guermantes" → "duchesse de
   Guermantes", erasing Proust's deliberate sociolinguistic variation), 515
   carry family possessive normalization ("ma grand'mère" → "la grand-mère").
4. **The canonical edition itself is contaminated.** The rewrite pass ran
   upstream of `data/islt/editions/fr-original/`: 69 broken-grammar bare-title
   hits across 8 chapter files ("…si je ne vais pas **avec duc de
   Guermantes** chez cette princesse d'Iéna…", v1-p2). The reader on the site
   is displaying altered Proust. Rebuild requires the local `islt_fr_*.html`
   page cache (not in the repo) with `preprocess(use_aliases=False)`.

Also: "uncertain alias resolution" is an enumerated ambiguity type, so alias
gaps leak straight into the −0.4 scoring penalty.

## The registry

`characters.yaml` is the single source of truth. Everything else — the legacy
CSV (`outputs/aliases.generated.csv`), per-chapter prompt reference sheets —
is generated from it. One entity per person-or-era, with provenance per form.

Surface-form fields:

- `scope`: `global` | `mention_only` | `context:<tag>` | `chapters:<ids>`
- `rewrite`: `allow` | `never`. Load-time validation rejects `allow` on
  descriptor/article-led forms; `rewrite_map()` additionally refuses any alias
  embedded (word-bounded) inside a longer form of a *different* entity — the
  invariant that makes "Mme de **M. de** Marsantes" impossible.
- `scan`: whether the string is usable as mention *evidence*. Descriptor
  phrases and the model's coreference records ("elle", "votre ami", "la
  petite") are reference knowledge, never evidence.

Identity over time is explicit: `eras` on an entity, and `same_person_links`
with a `policy` (`person_view_merge` | `keep_separate`) and `review` flag for
the interpretive cases — le peintre ↔ Elstir, prince des Laumes ↔ duc,
princesse de Guermantes ↔ Mme Verdurin. This makes person-view and name-view
standings both computable instead of accidental.

Resolution (`Registry.resolve`) returns `resolved` / `ambiguous` (e.g.
"M. Bloch" → père or fils) / `unresolved` — never a silent drop. Unresolved
and ambiguous names are a triage queue, not a void.

## Prompt v2 (replaces scope rules 17–19)

> - You are given a **reference sheet** of known characters and their surface
>   forms for this chapter. When a referent in the passage matches an entry,
>   use that canonical name.
> - List **every named character materially involved** in the passage's
>   evaluative or social dynamics — including characters **not** on the
>   reference sheet. For those, use the passage's surface form verbatim as
>   `canonical_name` and add `"resolution": "unresolved"`.
> - The passage text is Proust's original; the reference sheet is advisory
>   context, not a constraint on who exists.

Open decision (variant B): whether to also admit *individuated unnamed*
figures (l'amie de Mlle Vinteuil) as descriptor-identified entities with
`resolution: unresolved`. Default for now: named characters only.

## Migration sequence

0. Rebuild `fr-original` chapters from the HTML page cache with
   `use_aliases=False` (typographic cleanup only); regenerate reader data.
1. Adopt `characters.yaml` + `proust/registry.py` (no pipeline behavior
   change; tests green).
2. Switch annotation input to raw text + reference sheet; prompt v2. A/B a
   handful of units against v1 to measure drift before committing.
3. Targeted re-annotation sized by the audit: units with severe
   substitutions, plus strong exclusion-gap units (Rachel's chapters, the
   dame-en-rose units, Marsantes scenes).
4. Key downstream aggregation on `entity_id` with a name-era ledger;
   person-view and name-view standings become a toggle.
5. Retire per-run alias-map growth: `run.json` records the registry content
   hash it ran under.

## Decision queue for Nathan

All `status: review` / `review_questions` entries in `characters.yaml`, chiefly:
prince-des-Laumes merge (precedent: accepted princesse merge), le peintre ↔
Elstir policy, princesse de Guermantes era split vs person key, Octave 4-0-0
record adjudication, M. de Marsantes four-unit adjudication (see audit §D),
admission policy for individuated unnamed figures, and whether family
possessive normalization ("ma grand'mère" → "la grand-mère") should survive
in any form.
