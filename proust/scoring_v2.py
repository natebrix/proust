"""Scoring v2: one annotation to movements, labels, and weighted comparisons.

The formula is specified in `proust/docs/scoring_v2_design.md`; this module
is that specification and nothing else. It is pure: every function takes an
annotation dict (the shape `runner.validate_annotation_result` accepts) and
returns plain data, with no corpus, no run directories, and no I/O.

Four ideas carry the whole thing.

1. **The annotator already judged.** Only `status_effects` move a
   character: each carries a calibrated delta in -2..+2, a dimension, and a
   confidence. `appraisal_events` are the evidence lineage those effects
   point back at through `based_on_events`, so scoring them again as
   additive terms would count one reading twice. Events survive here only
   as an uncertainty signal (a "uncertain" narrative stance).

2. **The lenses partition the dimensions.** `social_status` is prestige,
   `inclusion_exclusion` is inclusion, and the remaining three are
   advantage, weighted 1.0 / 0.8 / 0.6 so the situational core outranks
   its rhetorical edge. No dimension feeds two lenses, which is what makes
   divergence between lenses (Odette's prestige against her inclusion)
   readable as information rather than as leakage.

3. **Uncertainty weighs, never subtracts.** Ambiguity notes and uncertain
   stances lower the WEIGHT of a comparison -- how much of it the rating
   layer is asked to believe -- and never its direction. Both characters
   in a pair are affected identically, so no sign can come from doubt.

4. **Mixed means conflicted, not middling.** A character is labelled mixed
   only when they carry both a positive and a negative effect in the same
   lens in the same unit: genuine internal conflict, never the residue of
   a penalty.
"""

from itertools import combinations

# The lens projection W: a partition of the five status dimensions, with
# internal weights only inside advantage.
LENS_DIMENSION_WEIGHTS = {
    "advantage": {
        "general_appraisal": 1.0,
        "emotional_position": 0.8,
        "rhetorical_position": 0.6,
    },
    "prestige": {"social_status": 1.0},
    "inclusion": {"inclusion_exclusion": 1.0},
}

SCORING_V2_LENS_ORDER = ("advantage", "prestige", "inclusion")
SCORING_V2_VERSION = "scoring_v2"

# Movements within this band of each other are a draw, and a movement
# within it of zero is neutral. Deltas stay integer-scaled, so this is v1's
# tie band on an unchanged scale.
TIE_BAND = 0.25

# rho(u) = max(AMBIGUITY_FLOOR, AMBIGUITY_DECAY ** ambiguity_count): each
# ambiguity note the annotator raised about a unit discounts every
# comparison drawn from it, down to a floor -- a heavily hedged unit is
# still evidence, just half of it.
AMBIGUITY_DECAY = 0.8
AMBIGUITY_FLOOR = 0.5

# A supporting event the annotator marked "uncertain" discounts the
# character's side of every comparison in the unit.
UNCERTAIN_STANCE = "uncertain"
UNCERTAIN_STANCE_MULTIPLIER = 0.7

# Personal identities the registry does not carry as entities.
NON_CHARACTER_NAMES = ("narrator", "collective_social_voice", "unknown")

PERSON_VIEW_MERGE_POLICY = "person_view_merge"

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_MIXED = "mixed"
LABEL_NEUTRAL = "neutral"


def require_known_lens(lens):
    """Raise unless `lens` is one of the three v2 lenses."""
    if lens not in LENS_DIMENSION_WEIGHTS:
        raise ValueError(
            f'Unknown scoring v2 lens "{lens}". Expected one of: '
            f'{", ".join(sorted(LENS_DIMENSION_WEIGHTS))}.'
        )
    return lens


def lens_weight(lens, dimension):
    """W_lens(dimension): 0.0 for a dimension this lens does not own."""
    require_known_lens(lens)
    return LENS_DIMENSION_WEIGHTS[lens].get(dimension, 0.0)


def _characters_present(annotation):
    rows = annotation.get("characters_present") or []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("canonical_name"), str)]


def _lens_effects(annotation, lens):
    """Every status effect that this lens gives a non-zero weight, by character.

    Effects naming a character who is not in `characters_present` are
    skipped, exactly as v1 skipped them: the annotation schema forbids
    that combination, and scoring is not the place to repair it.
    """
    present = {row["canonical_name"] for row in _characters_present(annotation)}
    by_character = {name: [] for name in present}
    for effect in annotation.get("status_effects") or []:
        if not isinstance(effect, dict):
            continue
        character = effect.get("character")
        if character not in by_character:
            continue
        if lens_weight(lens, effect.get("dimension")) == 0.0:
            continue
        by_character[character].append(effect)
    return by_character


def effect_movement(effect, lens):
    """One effect's contribution: W_lens(dimension) * delta * confidence."""
    weight = lens_weight(lens, effect.get("dimension"))
    if weight == 0.0:
        return 0.0
    return weight * int(effect.get("delta", 0)) * float(effect.get("confidence", 0.0))


def unit_movements(annotation, lens):
    """m(c, u, lens) for every character present in the unit.

    The sum of `effect_movement` over the character's effects in this
    lens. A character with no effect in this lens moves 0.0 and is still
    returned: they were present, they simply held still, and holding still
    is an outcome the comparisons need.
    """
    require_known_lens(lens)
    return {
        character: round(sum(effect_movement(effect, lens) for effect in effects), 12)
        for character, effects in _lens_effects(annotation, lens).items()
    }


def unit_labels(annotation, lens):
    """positive / negative / mixed / neutral for every character in the unit.

    A movement past the tie band in either direction names itself. Only
    inside the band does the effect structure get a say, and then only to
    distinguish a character pulled both ways at once (mixed) from one who
    was barely touched (neutral). Mixed therefore requires a genuine sign
    conflict within the lens -- it can never be manufactured by
    subtracting an uncertainty penalty, which is how v1 produced it.
    """
    require_known_lens(lens)
    movements = unit_movements(annotation, lens)
    effects_by_character = _lens_effects(annotation, lens)
    labels = {}
    for character, movement in movements.items():
        if movement > TIE_BAND:
            labels[character] = LABEL_POSITIVE
            continue
        if movement < -TIE_BAND:
            labels[character] = LABEL_NEGATIVE
            continue
        contributions = [effect_movement(effect, lens) for effect in effects_by_character[character]]
        conflicted = any(value > 0.0 for value in contributions) and any(
            value < 0.0 for value in contributions
        )
        labels[character] = LABEL_MIXED if conflicted else LABEL_NEUTRAL
    return labels


def ambiguity_weight(annotation):
    """rho(u): how much of a unit's evidence its ambiguity notes leave standing."""
    ambiguities = annotation.get("ambiguities") or []
    return max(AMBIGUITY_FLOOR, AMBIGUITY_DECAY ** len(ambiguities))


def unit_confidences(annotation, lens):
    """kappa(c, u, lens): how much a character's reading in this unit is believed.

    The mean confidence of the character's effects in this lens, cut to
    70% when any event those effects are based on was narrated with an
    uncertain stance. A character with no effect in this lens has no
    reading to be confident about beyond having been there at all, so
    their presence confidence stands in -- which is what lets a
    zero-movement character be compared at all.
    """
    require_known_lens(lens)
    effects_by_character = _lens_effects(annotation, lens)
    events_by_id = {
        event.get("event_id"): event
        for event in annotation.get("appraisal_events") or []
        if isinstance(event, dict)
    }
    presence_confidence = {
        row["canonical_name"]: float(row.get("presence_confidence", 0.0))
        for row in _characters_present(annotation)
    }

    confidences = {}
    for character, effects in effects_by_character.items():
        if not effects:
            confidences[character] = presence_confidence.get(character, 0.0)
            continue
        mean_confidence = sum(float(effect.get("confidence", 0.0)) for effect in effects) / len(effects)
        supporting_ids = {
            event_id
            for effect in effects
            for event_id in (effect.get("based_on_events") or [])
        }
        uncertain = any(
            events_by_id.get(event_id, {}).get("narrative_stance") == UNCERTAIN_STANCE
            for event_id in supporting_ids
        )
        confidences[character] = mean_confidence * (UNCERTAIN_STANCE_MULTIPLIER if uncertain else 1.0)
    return confidences


def comparison_outcome(movement_a, movement_b, tie_band=TIE_BAND):
    """(observed_a, observed_b) from two movements, with a tie band."""
    difference = movement_a - movement_b
    if difference > tie_band:
        return 1.0, 0.0
    if difference < -tie_band:
        return 0.0, 1.0
    return 0.5, 0.5


# ---------------------------------------------------------------------------
# Entity keying: the same comparison, read two ways.
# ---------------------------------------------------------------------------


def person_view_merge_map(registry):
    """entity_id -> the entity it aggregates into in the person view.

    Only `same_person_links` whose policy is `person_view_merge` merge:
    "le peintre" becomes Elstir and the prince des Laumes becomes the duc
    de Guermantes, because those are one person under two names. A
    `keep_separate` link records the opposite finding -- the post-V7
    princesse de Guermantes is Mme Verdurin holding a dead woman's title,
    two people sharing one name -- and must never merge. Chains are
    followed to their end so the map is idempotent.
    """
    direct = {}
    for entity in registry.entities.values():
        for link in entity.same_person_links or ():
            if not isinstance(link, dict):
                continue
            if link.get("policy") != PERSON_VIEW_MERGE_POLICY:
                continue
            target = link.get("entity")
            if target and target in registry.entities:
                direct[entity.id] = target

    merged = {}
    for entity_id in direct:
        seen = [entity_id]
        current = entity_id
        while current in direct:
            current = direct[current]
            if current in seen:
                raise ValueError(
                    "characters.yaml has a person_view_merge cycle: " + " -> ".join(seen + [current])
                )
            seen.append(current)
        merged[entity_id] = current
    return merged


def name_view_key(name):
    """The name view's key: the annotation's canonical name, unchanged.

    Foundation annotations are reconciled through the registry at ingest
    (`foundation.reconcile_foundation_names`), so a canonical name here is
    already the registry's canonical annotation name for that entity.
    """
    return name


def person_view_key(name, registry=None, merge_map=None, chapter_id=None):
    """The person view's key: the merged entity id, or the name if unresolved.

    Open-world names the registry cannot resolve (and names it resolves
    ambiguously) key on themselves, so nothing is ever silently dropped or
    pooled; they simply behave like the name view for that character.
    """
    if registry is None:
        return name
    if name in NON_CHARACTER_NAMES:
        return name
    resolution = registry.resolve(name, chapter_id=chapter_id)
    if resolution.status != "resolved":
        return name
    entity_id = resolution.entity_id
    if merge_map is None:
        merge_map = person_view_merge_map(registry)
    return merge_map.get(entity_id, entity_id)


def unit_comparisons(
    annotation,
    lens,
    unit_id=None,
    registry=None,
    merge_map=None,
    chapter_id=None,
    tie_band=TIE_BAND,
):
    """Every within-unit pairwise comparison this lens supports, weighted.

    For each unordered pair of characters present in the unit (ordered by
    name, so the pair list is stable), the outcome is the sign of their
    movement difference outside `tie_band` and a draw inside it, and the
    weight is

        w(a, b, u) = rho(u) * min(kappa_a, kappa_b)

    -- the unit's ambiguity discount times the less confident of the two
    readings. Direction comes only from the movements; every uncertainty
    signal is in the weight, applied to both characters at once.

    Each comparison carries BOTH keyings: `character_a`/`character_b` are
    the name view (what the annotation called them) and
    `person_a`/`person_b` are the person view (registry entity ids with
    `person_view_merge` links applied). A pair whose two names are one
    person collapses in the person view; that is visible here as
    `person_a == person_b` and is left for the caller to drop, since it is
    a fact about the pair rather than an error.
    """
    require_known_lens(lens)
    movements = unit_movements(annotation, lens)
    confidences = unit_confidences(annotation, lens)
    rho = ambiguity_weight(annotation)
    resolved_unit_id = unit_id or annotation.get("unit_id")
    if merge_map is None and registry is not None:
        merge_map = person_view_merge_map(registry)

    keys = {
        name: person_view_key(name, registry=registry, merge_map=merge_map, chapter_id=chapter_id)
        for name in movements
    }

    comparisons = []
    for character_a, character_b in combinations(sorted(movements), 2):
        movement_a = movements[character_a]
        movement_b = movements[character_b]
        observed_a, observed_b = comparison_outcome(movement_a, movement_b, tie_band=tie_band)
        confidence_a = confidences[character_a]
        confidence_b = confidences[character_b]
        comparisons.append(
            {
                "unit_id": resolved_unit_id,
                "lens": lens,
                "character_a": name_view_key(character_a),
                "character_b": name_view_key(character_b),
                "person_a": keys[character_a],
                "person_b": keys[character_b],
                "movement_a": movement_a,
                "movement_b": movement_b,
                "observed_a": observed_a,
                "observed_b": observed_b,
                "confidence_a": round(confidence_a, 6),
                "confidence_b": round(confidence_b, 6),
                "ambiguity_weight": round(rho, 6),
                "weight": round(rho * min(confidence_a, confidence_b), 6),
            }
        )
    return comparisons


__all__ = [
    "AMBIGUITY_DECAY",
    "AMBIGUITY_FLOOR",
    "LABEL_MIXED",
    "LABEL_NEGATIVE",
    "LABEL_NEUTRAL",
    "LABEL_POSITIVE",
    "LENS_DIMENSION_WEIGHTS",
    "PERSON_VIEW_MERGE_POLICY",
    "SCORING_V2_LENS_ORDER",
    "SCORING_V2_VERSION",
    "TIE_BAND",
    "UNCERTAIN_STANCE",
    "UNCERTAIN_STANCE_MULTIPLIER",
    "ambiguity_weight",
    "comparison_outcome",
    "effect_movement",
    "lens_weight",
    "name_view_key",
    "person_view_key",
    "person_view_merge_map",
    "require_known_lens",
    "unit_comparisons",
    "unit_confidences",
    "unit_labels",
    "unit_movements",
]
