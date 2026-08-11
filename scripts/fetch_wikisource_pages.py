"""Fetch the pinned French Wikisource revisions for the source migration.

Phase B of proust/docs/source_migration_plan.md. Every page consumed by the
migration is pinned to an exact revision id in outputs/wikisource-mapping.json;
this fetches each one BY REVID (never "latest", so a refetch can never silently
change the text) through the MediaWiki parse API, caches the rendered HTML under
data/wikisource/pages/<slug>.html, and records provenance (title, revid,
timestamp, sha256 of the cached HTML) in data/wikisource/manifest.json.

Throttled and resumable: a page whose cache file exists and whose manifest entry
matches the pinned revid and the file's current sha256 is skipped.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_URL = "https://fr.wikisource.org/w/api.php"
MAPPING_PATH = REPO_ROOT / "outputs" / "wikisource-mapping.json"
WIKISOURCE_DIR = REPO_ROOT / "data" / "wikisource"
PAGES_DIR = WIKISOURCE_DIR / "pages"
MANIFEST_PATH = WIKISOURCE_DIR / "manifest.json"

DELAY_SECONDS = 1.0
USER_AGENT = (
    "proust-islt-source-migration/1.0 "
    "(https://github.com/natebrix; personal research; polite, throttled; "
    "pinned revisions only)"
)


def slugify(title):
    """Filesystem-safe, stable, collision-free-enough slug for a page title."""
    folded = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("’", "'").replace("'", "")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", folded).strip("-").lower()
    return slug


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pinned_pages(mapping):
    """Unique (title, revid, timestamp) triples in first-seen mapping order."""
    seen, pages = set(), []
    for chapter in mapping["chapters"]:
        for page in chapter["pages"]:
            key = (page["title"], page["revid"])
            if key in seen:
                continue
            seen.add(key)
            pages.append(
                {
                    "title": page["title"],
                    "revid": int(page["revid"]),
                    "timestamp": page.get("timestamp"),
                    "slug": slugify(page["title"]),
                }
            )
    return pages


def http_get(url, timeout=60):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_rendered_html(revid):
    """Rendered HTML for one pinned revision (action=parse&oldid=...)."""
    url = API_URL + "?" + urlencode(
        {
            "action": "parse",
            "oldid": str(revid),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )
    payload = json.loads(http_get(url))
    if "error" in payload:
        raise RuntimeError(f"revid {revid}: API error {payload['error']}")
    parse = payload["parse"]
    text = parse["text"]
    if isinstance(text, dict):  # formatversion=1 shape
        text = text["*"]
    if int(parse.get("revid", revid)) != int(revid):
        raise RuntimeError(
            f"revid mismatch: asked {revid}, API returned {parse.get('revid')}"
        )
    return text


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"source": API_URL, "fetched_at": None, "pages": []}


def is_cached(entry, page):
    """True when the cached file still matches the pinned revid and its hash."""
    if entry is None or entry.get("revid") != page["revid"]:
        return False
    path = PAGES_DIR / f"{page['slug']}.html"
    if not path.exists():
        return False
    return sha256_text(path.read_text()) == entry.get("sha256")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch cached pages")
    args = parser.parse_args(argv)

    mapping = json.loads(MAPPING_PATH.read_text())
    pages = pinned_pages(mapping)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    by_key = {(e["title"], e["revid"]): e for e in manifest.get("pages", [])}

    entries, fetched, skipped = [], 0, 0
    for page in pages:
        existing = by_key.get((page["title"], page["revid"]))
        if not args.force and is_cached(existing, page):
            entries.append(existing)
            skipped += 1
            continue
        html = fetch_rendered_html(page["revid"])
        (PAGES_DIR / f"{page['slug']}.html").write_text(html)
        entries.append(
            {
                "title": page["title"],
                "revid": page["revid"],
                "timestamp": page["timestamp"],
                "slug": page["slug"],
                "sha256": sha256_text(html),
                "bytes": len(html.encode("utf-8")),
            }
        )
        fetched += 1
        print(f"fetched {page['title']} (revid {page['revid']})")
        time.sleep(DELAY_SECONDS)

    slugs = [e["slug"] for e in entries]
    if len(set(slugs)) != len(slugs):
        raise SystemExit(f"slug collision among {len(slugs)} pages")

    manifest = {
        "source": API_URL,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mapping": str(MAPPING_PATH.relative_to(REPO_ROOT)),
        "note": (
            "Pinned revisions only: pages are fetched by oldid, never by latest. "
            "Text updates are deliberate re-pins in the mapping, never silent refetches."
        ),
        "pages": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"done: {len(entries)} pages ({fetched} fetched, {skipped} cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
