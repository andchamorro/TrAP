"""Per-run manifest writer (DR-7).

Every CLI entry point calls ``write()`` on exit so that every output
directory carries a ``manifest.json`` recording the git commit, seed,
input paths, and throughput metrics.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Optional[Path], chunk: int = 1 << 20) -> Optional[str]:
    """Return hex SHA-256 of *path*, or ``None`` if path is absent.

    Args:
        path: File to hash; ``None`` is accepted and returns ``None``.
        chunk: Read chunk size in bytes.

    Returns:
        Hex digest string, or ``None``.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def write(out_dir: Path, **kwargs: Any) -> Path:
    """Write ``manifest.json`` to *out_dir* and return its path.

    Standard header fields (``trap_version``, ``git_commit``,
    ``timestamp_utc``) are written first; all ``**kwargs`` are merged on
    top.  Nested dicts are supported.

    Args:
        out_dir: Directory to write the manifest (created if needed).
        **kwargs: Arbitrary manifest fields.

    Returns:
        Path to the written ``manifest.json``.

    Example::

        write(
            output_path,
            seed=3469,
            k=17,
            model={"path": "...", "sha256": sha256_file(model_path)},
            throughput={"reads_per_s": 210_000, "total_reads": 10_000_000},
        )
    """
    try:
        import trap
        version: str = getattr(trap, "__version__", "0.0.1")
    except Exception:
        version = "0.0.1"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "trap_version": version,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(kwargs)
    path = out_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
