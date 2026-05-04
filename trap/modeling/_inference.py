"""Shared inference-loop helpers (cleanup #14).

These small pieces were duplicated across the quantification commands in
``predict.py`` and ``quantify.py`` (device/dtype selection, the Parquet score
schema, and the softmax→table conversion). Centralising them removes the
copy-paste and lets ``quantify.py`` stop importing a private symbol from the
heavy ``predict`` module.
"""

from __future__ import annotations

import contextlib
from typing import List, Sequence

import numpy as np
import pyarrow as pa
import torch


def resolve_device() -> str:
    """Return the best available device string (``cuda`` / ``mps`` / ``cpu``)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx(device: str, enabled: bool = True):
    """Return a ``torch.autocast`` context for *device*.

    dtype selection:
      cuda  → bfloat16  (A100 has native bf16 tensor-core support)
      mps   → float16   (Apple Silicon; bf16 not fully supported on all chips)
      cpu   → disabled  (bf16 on CPU is slower than fp32 for small batches)
    """
    if not enabled or device == "cpu":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    return torch.autocast(device_type=device, dtype=dtype)


def build_score_schema(labels: Sequence[str]) -> pa.Schema:
    """Parquet schema for class scores: ``id`` (string) + one float32 per label."""
    return pa.schema([("id", pa.string())] + [(lbl, pa.float32()) for lbl in labels])


def scores_table(ids: List[str], probs: np.ndarray, labels: Sequence[str]) -> pa.Table:
    """Build a score table from read ``ids`` and a ``(n, n_labels)`` prob matrix."""
    return pa.table(
        {"id": pa.array(ids, type=pa.string())}
        | {lbl: pa.array(probs[:, i], type=pa.float32()) for i, lbl in enumerate(labels)}
    )
