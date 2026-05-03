from .paths import ALIASES_CSV


def clean_character_name(value):
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "ALIASES_CSV",
    "clean_character_name",
]
