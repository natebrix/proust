from .paths import ALIASES_CSV


REVIEWED_CHARACTER_NORMALIZATION_MAP = {
    "Saint-Loup": "Robert de Saint-Loup",
    "princesse des Laumes": "duchesse de Guermantes",
    "Charlus": "baron de Charlus",
    "Mme Swann": "Odette",
    "la grand-mère du narrateur": "la grand-mère",
    "Vinteuil": "M. Vinteuil",
    "Mme de Saint-Euverte": "marquise de Saint-Euverte",
}


def clean_character_name(value):
    return value.strip() if isinstance(value, str) else ""


def normalize_character_name(character, character_name_map=None):
    if not character_name_map:
        return character
    return character_name_map.get(character, character)


def normalize_character_name_map(character_name_map):
    if not character_name_map:
        return {}

    normalized_map = {}
    for source, target in character_name_map.items():
        clean_source = clean_character_name(source)
        clean_target = clean_character_name(target)
        if not clean_source or not clean_target or clean_source == clean_target:
            continue
        normalized_map[clean_source] = clean_target
    return normalized_map


__all__ = [
    "ALIASES_CSV",
    "REVIEWED_CHARACTER_NORMALIZATION_MAP",
    "clean_character_name",
    "normalize_character_name",
    "normalize_character_name_map",
]
