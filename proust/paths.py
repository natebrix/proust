from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fr"
ISLT_DATA_DIR = REPO_ROOT / "data" / "islt"
ISLT_EDITIONS_DIR = ISLT_DATA_DIR / "editions"
ALIASES_CSV = REPO_ROOT / "aliases.csv"
