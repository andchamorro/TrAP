"""Sync `.zenodo.json` onto the live Zenodo record for TrAP.

The published record's metadata drifted from the repository: it still carried the
pre-2026-09-09 manuscript title, named BSD as the licence where `LICENSE` is
Apache 2.0, left `version` empty, and carried a stray keyword. `.zenodo.json` is
the single place that metadata is now written; this script pushes it.

    python scripts/zenodo_sync.py                 # diff only, touches nothing
    python scripts/zenodo_sync.py --apply         # edit the published record's metadata
    python scripts/zenodo_sync.py --new-version   # open a new-version draft instead

Reads ZENODO_TOKEN from the environment or from `.env` at the repo root. The
sandbox token (SAND_ZENODO_TOKEN) is never used. `--apply` edits metadata in
place and republishes the same DOI; it does not mint a new one. `--new-version`
only opens the draft — uploading the archive and publishing stay manual, because
publishing is irreversible.
"""

import argparse
import json
import os
import sys
from urllib import error, request

from trap.config.config import PROJ_ROOT

API = "https://zenodo.org/api"
CONCEPT_RECID = "22757448"
METADATA = PROJ_ROOT / ".zenodo.json"


def token():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        env = PROJ_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ZENODO_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip("'\"")
    if not tok:
        sys.exit("ZENODO_TOKEN not set (environment or .env)")
    return tok


def call(url, tok, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        sys.exit(f"{method} {url} -> {exc.code}\n{exc.read().decode()[:2000]}")


def latest(tok):
    """The newest version of the concept record, resolved through its DOI."""
    rec = call(f"{API}/records/{CONCEPT_RECID}/versions/latest", tok)
    return str(rec["id"])


def diff(live, wanted):
    rows = []
    for key in sorted(set(live) | set(wanted)):
        if key not in wanted:
            continue  # the file only governs the keys it names
        a, b = live.get(key), wanted[key]
        if isinstance(a, dict) and "id" in a and isinstance(b, str):
            a = a["id"]  # license comes back as {"id": ...}
        if key == "license" and isinstance(a, str) and isinstance(b, str):
            # Zenodo stores its own licence ids, so "apache-2.0" comes back as
            # "apache2.0". Compare without the separator or this diff never
            # settles.
            if a.replace("-", "") == b.replace("-", ""):
                continue
        if a != b:
            rows.append((key, a, b))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="write the metadata")
    group.add_argument("--new-version", action="store_true", help="open a new-version draft")
    args = ap.parse_args()

    tok = token()
    wanted = json.loads(METADATA.read_text())
    recid = latest(tok)
    dep = call(f"{API}/deposit/depositions/{recid}", tok)
    live = dep["metadata"]

    print(f"record     {recid}  (concept {CONCEPT_RECID})")
    print(f"doi        {dep.get('doi')}")
    print(f"state      {dep.get('state')}  submitted={dep.get('submitted')}\n")

    rows = diff(live, wanted)
    if not rows:
        print("metadata already matches .zenodo.json")
    for key, was, now in rows:
        print(f"~ {key}")
        print(f"    live:   {json.dumps(was, ensure_ascii=False)[:300]}")
        print(f"    wanted: {json.dumps(now, ensure_ascii=False)[:300]}")

    if args.new_version:
        draft = call(dep["links"]["newversion"], tok, "POST")
        url = draft["links"]["latest_draft"]
        print(f"\nnew-version draft opened: {url}")
        print("upload the archive and publish by hand; this script will not publish.")
        return

    if not args.apply:
        print("\n(dry run; pass --apply to write)")
        return

    call(dep["links"]["edit"], tok, "POST")  # unlock the published record
    merged = dict(live)
    merged.update(wanted)
    call(f"{API}/deposit/depositions/{recid}", tok, "PUT", {"metadata": merged})
    call(dep["links"]["publish"], tok, "POST")
    print(f"\napplied; same DOI {dep.get('doi')} now carries the updated metadata")


if __name__ == "__main__":
    main()
