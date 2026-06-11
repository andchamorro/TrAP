"""Lazy alignment file reader (DR-2).

Previously ``SAMDataset`` lived in ``predict.py``; moved here so that the
preprocessing and inference paths can both import it without pulling in the
full inference dependency chain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pysam


class SAMDataset:
    """Lazy wrapper around a BAM/SAM/CRAM alignment file.

    Sequences are indexed on first access and served from memory on
    subsequent accesses.  The previous implementation materialised every
    alignment into a dict at ``__init__`` time; this version streams the
    file once via pysam.

    Args:
        file_path: Path to the alignment file (``.bam``, ``.sam``,
            ``.cram``).
    """

    _FORMAT_MODES: Dict[str, str] = {".bam": "rb", ".sam": "r", ".cram": "rc"}

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self._ids: List[str] = []
        self._queries: Dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for record in self._stream():
            if record.query_sequence:
                self._queries[record.query_name] = record.query_sequence
        self._ids = list(self._queries.keys())
        self._loaded = True

    def _stream(self) -> Iterator:
        suffix = self.file_path.suffix
        mode = self._FORMAT_MODES.get(suffix)
        if mode is None:
            raise ValueError(f"Unsupported alignment format: {suffix!r}")
        # check_sq=False + fetch(until_eof=True) streams reads from files with no
        # @SQ headers (e.g. unmapped-only SAM/BAM); pysam >=0.22 refuses to iterate
        # such files otherwise. We only need query name/sequence, not references.
        af = pysam.AlignmentFile(str(self.file_path), mode, check_sq=False)
        return af.fetch(until_eof=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def queries(self) -> Dict[str, str]:
        """Dict mapping query name → sequence (lazy-loaded)."""
        self._ensure_loaded()
        return self._queries

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._ids)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        self._ensure_loaded()
        for key in self._ids:
            yield key, self._queries[key]

    def __getitem__(self, index):
        self._ensure_loaded()
        if isinstance(index, slice):
            keys = self._ids[index]
            return keys, [self._queries[k] for k in keys]
        if isinstance(index, int):
            if index < 0:
                index += len(self._ids)
            if not (0 <= index < len(self._ids)):
                raise IndexError(f"SAMDataset index {index} out of range")
            key = self._ids[index]
            return key, self._queries[key]
        raise TypeError(f"Invalid index type: {type(index)!r}")
