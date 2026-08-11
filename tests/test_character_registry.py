"""Regression tests for the character registry.

Each test pins a real failure found in the corpus audit (2026-08):
descriptor-rule corruption (docteur Cottard inserted into a brothel simile),
nested-honorific mangling (Mme de M. de Marsantes), the Mme Octave / Octave
chimera from the original aliases.csv, and the closed-world exclusion that
made Rachel invisible to the annotator.
"""

import pytest

from proust.corpus import apply_alias_replacements
from proust.registry import REGISTRY_PATH, Registry


@pytest.fixture(scope="module")
def registry():
    return Registry.load(REGISTRY_PATH)


# --------------------------------------------------------------- load safety

def test_registry_loads_and_validates(registry):
    assert len(registry.entities) > 60
    assert "rachel" in registry.entities


def test_rewrite_map_contains_no_descriptor_rules(registry):
    """Descriptor phrases must never be rewrite keys — this is the invariant
    that makes docteur-Cottard-class corruption structurally impossible."""
    for alias in registry.rewrite_map():
        lead = alias.split()[0].lower()
        assert lead not in {
            "le", "la", "les", "l'", "un", "une", "ma", "mon", "mes",
            "sa", "son", "ses", "notre", "votre", "leur",
        }, f"descriptor rewrite rule leaked: {alias!r}"


# ------------------------------------------------- historical corruption cases

BROTHEL_SIMILE = (
    "elle se tenait devant le docteur qui allait l'ausculter, et à qui elle "
    "demandait s'il fallait tout retirer"
)

MARSANTES_SENTENCE = (
    "Mme de Marsantes s'assit auprès de Robert et lui parla doucement."
)


def test_brothel_simile_survives_rewrite(registry):
    """'le docteur' in a generic simile must never become docteur Cottard."""
    rewritten = apply_alias_replacements(BROTHEL_SIMILE, registry.rewrite_map())
    assert "Cottard" not in rewritten
    assert "le docteur" in rewritten


def test_marsantes_is_not_mangled(registry):
    """Historical rule 'Marsantes' -> 'M. de Marsantes' produced
    'Mme de M. de Marsantes'. The registry map must leave her name intact."""
    rewritten = apply_alias_replacements(MARSANTES_SENTENCE, registry.rewrite_map())
    assert "Mme de M. de Marsantes" not in rewritten
    assert rewritten.startswith("Mme de Marsantes")


def test_bare_princesse_never_rewrites(registry):
    """runs 067-071 carried 'princesse' -> 'princesse des Laumes'; a bare
    'princesse' must never be a rewrite key again."""
    sentence = "la princesse de Parme sourit à la princesse de Luxembourg"
    assert apply_alias_replacements(sentence, registry.rewrite_map()) == sentence


# ----------------------------------------------------------------- resolution

def test_mme_octave_is_tante_leonie_not_octave(registry):
    assert registry.resolve("Mme Octave").entity_id == "tante-leonie"
    assert registry.resolve("Madame Octave").entity_id == "tante-leonie"
    assert registry.resolve("Octave").entity_id == "octave"


def test_rachel_resolves(registry):
    assert registry.resolve("Rachel").entity_id == "rachel"
    assert registry.resolve("Rachel quand du Seigneur").entity_id == "rachel"
    assert registry.resolve("Zézette").entity_id == "rachel"


def test_laumes_era_names_resolve_to_the_era_entity(registry):
    """The Laumes era stays a distinct entity (mirroring the corpus) with an
    explicit same-person link to the duc awaiting Nathan's merge decision."""
    assert registry.resolve("prince des Laumes").entity_id == "prince-des-laumes"
    assert registry.resolve("M. des Laumes").entity_id == "prince-des-laumes"
    links = registry.entities["prince-des-laumes"].same_person_links
    assert links and links[0]["entity"] == "duc-de-guermantes"


def test_ambiguous_surface_is_flagged_not_guessed(registry):
    resolution = registry.resolve("M. Bloch")
    assert resolution.status == "ambiguous"
    assert set(resolution.candidates) == {"bloch", "bloch-pere"}


def test_unknown_surface_is_unresolved_not_dropped(registry):
    resolution = registry.resolve("Mme de Nulle-Part")
    assert resolution.status == "unresolved"
    assert resolution.entity_id is None


def test_annotation_canonicals_resolve(registry):
    """Every canonical name in accepted annotations must resolve losslessly."""
    for name in ("le narrateur", "duchesse de Guermantes", "la grand-mère",
                 "M. Vinteuil", "baron de Charlus", "la Berma", "le peintre"):
        assert registry.resolve(name).status == "resolved", name


# ------------------------------------------------------------------- scanning

def test_scanner_finds_rachel_at_the_bal_de_tetes(registry):
    scanner = registry.compile_scanner()
    text = (
        "La Berma pouvait mourir : Rachel triomphait, et Rachel récitait "
        "des vers devant la duchesse."
    )
    found = scanner.scan(text)
    assert found.get("rachel") and len(found["rachel"]) == 2
    assert found.get("la-berma")


def test_scanner_ignores_pure_descriptors(registry):
    scanner = registry.compile_scanner()
    found = scanner.scan("le docteur entra, suivi de la duchesse et du peintre.")
    assert "docteur-cottard" not in found
    assert "duchesse-de-guermantes" not in found
