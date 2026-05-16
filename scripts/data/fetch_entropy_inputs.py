"""Deterministically fetch the corpora for the entropy/redundancy analysis.

Remotely gathers the inputs consumed by ``python -m trap.analysis`` and records
their exact bytes in a version-controllable lockfile so the whole analysis is
reproducible from scratch:

* **GENCODE v48 transcripts** — downloaded from the pinned EBI release URL (same
  source as ``scripts/data/fetch_references.sh``), streamed to disk while hashing.
* **RepeatMasker LINE-1 corpus** — verified if already built by
  ``fetch_references.sh`` (which needs the genome + RepeatMasker GFF); this script
  does not re-download the multi-GB genome, it only records the corpus checksum
  and tells you how to build it if missing.

The lockfile (``data/external/entropy_inputs.lock.json``) pins URL, GENCODE
release, byte size, and SHA-256 for every input.  Re-running is idempotent: an
input whose checksum already matches the lockfile is skipped.  Pass
``--expected-transcripts-sha256`` in CI to *assert* the downloaded bytes.

Usage::

    python scripts/data/fetch_entropy_inputs.py
    python scripts/data/fetch_entropy_inputs.py --release 48 --force
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request

from loguru import logger
import typer

from trap.config.config import EXTERNAL_DATA_DIR

app = typer.Typer(help="Fetch + lock the entropy/redundancy analysis corpora.")

EBI_BASE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human"


def _sha256_and_size(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """Return ``(hex_sha256, size_bytes)`` for an existing file."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """Stream *url* to *dest* while hashing; return ``(sha256, size)``.

    Downloads to a ``.part`` file and renames on success so an interrupted run
    never leaves a truncated file that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "trap-fetch/1.0"})
    h = hashlib.sha256()
    size = 0
    logger.log("STAGE", f"[fetch] {url} -> {dest}")
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(chunk)
            if not block:
                break
            out.write(block)
            h.update(block)
            size += len(block)
    tmp.rename(dest)
    return h.hexdigest(), size


def _record(
    lock: dict,
    key: str,
    *,
    path: Path,
    sha256: str,
    size: int,
    url: Optional[str],
    downloaded: bool,
) -> None:
    """Add/replace one input entry in the lockfile dict."""
    lock["inputs"][key] = {
        "path": str(path),
        "url": url,
        "sha256": sha256,
        "bytes": size,
        "downloaded": downloaded,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.command()
def main(
    release: int = typer.Option(48, help="GENCODE release"),
    out_dir: Path = typer.Option(EXTERNAL_DATA_DIR, help="Destination directory"),
    l1_corpus: Optional[Path] = typer.Option(
        None, help="LINE-1 corpus FASTA (default: GCF_000001405.40 v<release> from fetch_references.sh)"
    ),
    expected_transcripts_sha256: Optional[str] = typer.Option(
        None, help="If set, assert the transcripts SHA-256 matches"
    ),
    force: bool = typer.Option(False, help="Re-download even if the checksum matches the lock"),
) -> None:
    """Fetch + checksum the entropy/redundancy corpora and write the lockfile."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "entropy_inputs.lock.json"
    lock: dict = {
        "gencode_release": release,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "inputs": {},
    }
    prior = json.loads(lock_path.read_text()).get("inputs", {}) if lock_path.exists() else {}

    def _checksum(path: Path, key: str) -> tuple[str, int]:
        """Reuse the locked checksum when the file size is unchanged."""
        recorded = prior.get(key, {})
        if recorded.get("sha256") and recorded.get("bytes") == path.stat().st_size:
            logger.log("STAGE", f"[fetch] checksum unchanged, reusing lock: {path}")
            return recorded["sha256"], int(recorded["bytes"])
        return _sha256_and_size(path)

    # --- GENCODE transcripts (remote) ---
    transcripts = out_dir / f"gencode.v{release}.transcripts.fa.gz"
    url = f"{EBI_BASE}/release_{release}/gencode.v{release}.transcripts.fa.gz"
    if transcripts.exists() and not force:
        sha256, size = _checksum(transcripts, "gencode_v48")
        logger.log("STAGE", f"[fetch] exists, skipping download: {transcripts}")
        downloaded = False
    else:
        sha256, size = _download(url, transcripts)
        downloaded = True
    if expected_transcripts_sha256 and sha256 != expected_transcripts_sha256:
        raise typer.Exit(
            f"transcripts SHA-256 mismatch: got {sha256}, expected {expected_transcripts_sha256}"
        )
    _record(lock, "gencode_v48", path=transcripts, sha256=sha256, size=size, url=url, downloaded=downloaded)

    # --- LINE-1 corpus (built by fetch_references.sh; verify only) ---
    if l1_corpus is None:
        l1_corpus = out_dir / f"GCF_000001405.40_GRCh38.p14_rm.LINE1.gencode.v{release}.fa"
    if l1_corpus.exists():
        sha256, size = _checksum(l1_corpus, "l1_repeatmasker")
        _record(lock, "l1_repeatmasker", path=l1_corpus, sha256=sha256, size=size, url=None, downloaded=False)
        logger.log("STAGE", f"[fetch] L1 corpus present: {l1_corpus} ({size:,} B)")
    else:
        logger.warning(
            f"L1 corpus missing at {l1_corpus}. Build it with "
            f"`bash scripts/data/fetch_references.sh` (needs the genome + RepeatMasker GFF)."
        )
        _record(lock, "l1_repeatmasker", path=l1_corpus, sha256="", size=0, url=None, downloaded=False)

    lock_path.write_text(json.dumps(lock, indent=2))
    logger.success(f"wrote lockfile {lock_path}")


if __name__ == "__main__":
    app()
