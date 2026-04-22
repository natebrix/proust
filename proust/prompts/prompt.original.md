You are annotating a French passage from Marcel Proust's *À la recherche du temps perdu* for **local appraisal events** and **character status effects**.

Your purpose is to generate structured literary-social annotations that can later be transformed into different notions of "winning" and "losing."
Do **not** reduce the passage to a single winner-loser verdict unless the evidence is overwhelmingly explicit.
Instead, identify evaluative acts and their consequences for the named characters.

## Inputs

You will be given:

1. A French passage.
2. An alias map for named characters.
3. Optionally, brief prior context from the immediately preceding window.

## Scope rules

* Use **only named characters** that appear in the alias map.
* Resolve references to the **canonical character name** using the alias map.
* Ignore unnamed figures unless they are clearly mapped to a named character.
* Work primarily from the passage itself.
* Use prior context only for local disambiguation of references, not for broad plot inference.
* Do not invent motives, unstated events, or long-run arc interpretations.

## What to detect

Track local shifts in how named characters are positioned through:

* praise
* blame
* admiration
* contempt
* ridicule
* preference
* favorable or unfavorable comparison
* deference
* snub
* exclusion
* humiliation
* prestige by association
* discredit by association
* rhetorical authority
* emotional leverage
* narrated elevation or diminishment
* inclusion in or exclusion from valued social space
* signs that another character depends on, yields to, fears, imitates, or dismisses them

## Interpretive principles

A character may come out ahead or behind in several ways:

* directly, by being praised or insulted
* comparatively, by being favored over another
* socially, by being included, deferred to, or excluded
* rhetorically, by speaking with force, wit, authority, or discernment
* emotionally, by gaining leverage over another character
* associationally, by benefiting from or suffering through a linked person
* narratively, through a passage that appears to elevate or diminish them

Do **not** judge:

* morality
* factual correctness
* long-term importance
* whether the character "deserves" the treatment

Judge only the **local evaluative and social dynamics** of the supplied passage.

## Special caution for Proust

Proust often layers evaluation through:

* quoted speech
* free indirect style
* remembered perception
* irony
* narrator distance
* social codes that are reported rather than endorsed

For every evaluative event, distinguish:

* who is making the evaluation
* who is its target
* whether the passage appears to endorse, neutrally report, ironize, or leave uncertain that evaluation

## Task

1. Identify all named characters present or clearly implicated.
2. Extract only the **significant** appraisal or status-relevant events.
3. Record local status effects for the characters involved.
4. Note ambiguity from irony, free indirect style, uncertain source, or uncertain alias resolution.
5. Multiple characters may gain or lose simultaneously.
6. If there is no meaningful status movement, say so.
7. Prefer fewer, high-quality events over many trivial ones.

## Output

Return valid JSON only.

Schema:

{
"characters_present": [
{
"canonical_name": "string",
"surface_forms": ["string"],
"presence_type": "explicit | implicit",
"presence_confidence": 0.0
}
],
"appraisal_events": [
{
"event_id": "E1",
"source": "canonical character name | narrator | collective_social_voice | unknown",
"target": "canonical character name",
"type": "praise | blame | admiration | contempt | ridicule | preference | favorable_comparison | unfavorable_comparison | deference | snub | exclusion | humiliation | prestige_association | discredit_association | rhetorical_authority | emotional_leverage | narrated_elevation | narrated_diminishment | other",
"evidence_mode": "quoted_speech | narration | comparison | action | reaction | mixed",
"polarity": "positive | negative | mixed",
"directness": "direct | indirect | implicit",
"narrative_stance": "endorsed | neutral_report | ironized | uncertain",
"intensity": 1,
"confidence": 0.0,
"evidence": "brief quotation or paraphrase from the passage",
"explanation": "1-2 sentence explanation"
}
],
"status_effects": [
{
"character": "canonical character name",
"dimension": "general_appraisal | social_status | rhetorical_position | emotional_position | prestige | intimacy | exclusion",
"delta": -2,
"based_on_events": ["E1"],
"confidence": 0.0,
"explanation": "brief explanation"
}
],
"window_summary": {
"salient_dynamics": [
"brief sentence",
"brief sentence"
],
"characters_ahead": [
{
"character": "canonical character name",
"basis": "brief explanation"
}
],
"characters_behind": [
{
"character": "canonical character name",
"basis": "brief explanation"
}
],
"overall_confidence": 0.0
},
"ambiguities": [
"string"
]
}

## Scoring guidance

Intensity:

* 1 = weak or slight
* 2 = moderate and clear
* 3 = strong or emphatic

Status delta:

* -2 = clearly diminished in this passage
* -1 = somewhat diminished
* 0 = neutral, mixed, or no clear movement
* +1 = somewhat elevated
* +2 = clearly elevated

Confidence:

* 0.0 to 1.0
* Be conservative when irony, layered narration, or reference resolution makes interpretation unstable.

## Important rules

* Named characters only.
* Always use canonical names from the alias map.
* If a surface form is ambiguous, mention that in "ambiguities."
* Do not infer broad character arcs.
* Do not force zero-sum logic.
* A single event may affect both source and target, but record that through separate status effects if needed.
* Ignore trivial mentions that do not meaningfully alter evaluation or status.

## Inputs begin below

### Alias map

{{ALIAS_MAP}}

### Prior local context (optional)

{{PRIOR_CONTEXT}}

### Passage

{{PASSAGE}}
