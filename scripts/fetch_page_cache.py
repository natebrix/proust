"""Rebuild the islt_fr_*.html page cache in data/fr.

The original cache was lost with a machine change, and marcel-proust.com has
since been redesigned (different markup, repartitioned pages), so the live
site cannot reproduce the original canonical build. The Wayback Machine holds
complete captures of the original site (all 486 pages, 2017-2018): this
fetches the archived original bytes (the ``id_`` raw mode, no archive chrome),
throttled and resumable.
"""
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proust.paths import DATA_DIR  # noqa: E402

DELAY_SECONDS = 1.0
USER_AGENT = "proust-islt-cache-rebuild/1.0 (personal research; polite, throttled)"
CDX_URL = (
    "http://web.archive.org/cdx/search/cdx"
    "?url=marcel-proust.com/marcelproust/*&output=json"
    "&filter=statuscode:200&collapse=urlkey&fl=timestamp,original"
)


def http_get(url, timeout=60):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_capture_index():
    rows = json.loads(http_get(CDX_URL))
    captures = {}
    for timestamp, original in rows[1:]:
        page = original.rstrip("/").rsplit("/", 1)[-1]
        if page.isdigit() and len(page) == 3:
            captures[page] = (timestamp, original)
    return captures


def main(start=1, end=486):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    captures = load_capture_index()
    print(f"capture index: {len(captures)} archived pages")
    fetched = skipped = failed = 0
    for page_id in range(start, end + 1):
        key = f"{page_id:03}"
        target = DATA_DIR / f"islt_fr_{key}.html"
        marker = b"field-item"  # original Drupal markup
        if target.exists() and marker in target.read_bytes():
            skipped += 1
            continue
        if key not in captures:
            print(f"NO CAPTURE for {key}")
            failed += 1
            continue
        timestamp, original = captures[key]
        url = f"http://web.archive.org/web/{timestamp}id_/{original}"
        try:
            body = http_get(url)
        except Exception as first_error:
            time.sleep(10)
            try:
                body = http_get(url)
            except Exception:
                print(f"FAILED {key}: {first_error}")
                failed += 1
                continue
        target.write_bytes(body)
        fetched += 1
        if fetched % 25 == 0:
            print(f"fetched {fetched} (at page {key})")
        time.sleep(DELAY_SECONDS)
    print(f"done: fetched={fetched} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
