"""Global seed initialisation for reproducible runs.

Call ``set_global_seed`` once at the top of every entry point (CLI command,
SLURM job, training script) before any random draws so that results are
identical across re-runs with the same seed.

Device coverage
---------------
CUDA        torch.cuda.manual_seed_all
MPS         torch.mps.manual_seed  (Apple Silicon, used for local testing)
CPU         torch.manual_seed
numpy       np.random.seed  (legacy global RNG only)
stdlib      random.seed + PYTHONHASHSEED

Important
---------
``np.random.seed`` seeds only numpy's *legacy* global RNG. It does **not**
affect ``np.random.default_rng()`` ``Generator`` objects — those draw from OS
entropy unless explicitly seeded. Code that uses a ``Generator`` for
reproducibility must build it with ``seeded_rng(seed)`` (or
``np.random.default_rng(seed)``) and thread it through, rather than relying on
``set_global_seed``.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed all random-number generators in one call.

    Args:
        seed: Integer seed value.  The same seed on the same hardware and
            library versions reproduces bit-identical results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def seeded_rng(seed: int) -> np.random.Generator:
    """Return a freshly seeded numpy ``Generator`` for reproducible draws.

    Use this (and pass the result explicitly) wherever a ``Generator`` is
    needed — ``set_global_seed`` does not cover ``default_rng`` generators.

    Args:
        seed: Integer seed value.

    Returns:
        ``np.random.Generator`` seeded with *seed*.
    """
    return np.random.default_rng(seed)
