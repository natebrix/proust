from pathlib import Path


ISLT_PORTRAITS_DIR = Path("/Users/nathan_brixius/dev/brixius-web/public/projects/islt/portraits")
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
