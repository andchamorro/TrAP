"""Fixtures for the training / tuning workflow validation suite.

Kept here (not in the top-level conftest) so they are scoped to ``tests/modeling``
and reusable across the modeling tests without polluting unrelated suites.
"""

from __future__ import annotations

import pytest

from tests.modeling import workflow_data as wd

SEED = 3469  # canonical project seed (mirrors config/training/*.json)


@pytest.fixture
def size() -> wd.SizeConfig:
    """Dataset/model size — scalable via TRAP_TEST_DATASET_SIZE (default 'tiny')."""
    return wd.SizeConfig.from_env()


@pytest.fixture
def synthetic_classification(size) -> "wd.DatasetDict":
    """Deterministic, class-separable classification DatasetDict (train/eval/test)."""
    return wd.build_synthetic_dataset(size, seed=SEED)


@pytest.fixture
def synthetic_mlm(size) -> "wd.DatasetDict":
    """Deterministic tiny MLM DatasetDict (train/eval) with pre-computed labels."""
    return wd.build_synthetic_mlm_dataset(size, seed=SEED)


@pytest.fixture
def workflow_dataset(size):
    """Preferred real-data sample, else synthetic. Returns ``(DatasetDict, source)``."""
    return wd.build_dataset(size, seed=SEED, prefer_real=True)
