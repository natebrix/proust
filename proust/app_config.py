from pathlib import Path


# Derived from the home directory rather than a baked absolute path: a stale
# username in this constant once silently emptied every character's portrait
# variants on regeneration.
ISLT_PORTRAITS_DIR = Path.home() / "dev" / "brixius-web" / "public" / "projects" / "islt" / "portraits"
ISLT_READER_BASE_PATH = "/projects/islt/fr-original"
PORTRAIT_STYLES = (
    "vermeer-proustian",
    "tarot-marseille-belle-epoque",
    "elstir",
)


__all__ = [
    "ISLT_PORTRAITS_DIR",
    "ISLT_READER_BASE_PATH",
    "PORTRAIT_STYLES",
]
