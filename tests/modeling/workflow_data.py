"""Reusable data engine + assertions for the training / tuning workflow suite.

This module is **not** collected by pytest (no ``test_`` prefix). It is imported
by ``test_training_workflow.py`` and is safe to reuse from other test modules.

Design / assumptions
---------------------
* **Two data sources, one schema.** Validation tests prefer a *small, seeded,
  stratified sample of the real processed classification dataset* (so schema and
  preprocessing realism are exercised); if none is on disk (fresh clone / CI),
  an equivalent **synthetic** dataset is generated with the identical column set
  and dtypes (:data:`MODEL_COLUMNS`).
* **Learning-dynamics tests always use synthetic data.** The synthetic generator
  builds *class-separable* reads (each label draws tokens from a disjoint band),
  so a tiny 1-layer ALBERT provably converges in a few epochs — making
  convergence / metric-consistency assertions deterministic and fast. Real data
  is not guaranteed to converge on a tiny model, so it is reserved for the
  structural (schema / leakage / bounds) checks.
* **Determinism.** Every generator takes an explicit ``seed`` and uses a local
  ``numpy`` ``Generator`` (never the global RNG). Same seed → bit-identical rows
  (guarded by :func:`fingerprint`). Splits are transcript-level (reusing the
  production splitter) so there is no read-level leakage by construction.
* **Configurable size / complexity hooks.** :class:`SizeConfig` controls every
  dimension; :meth:`SizeConfig.from_env` reads ``TRAP_TEST_DATASET_SIZE`` so the
  same tests can be scaled up locally (``small`` / ``medium``) without code
  changes while CI stays on ``tiny``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
import numpy as np

from trap.utils.preprocessing_sequences import _transcript_level_split

logger = logging.getLogger("trap.tests.workflow")

# Label set + special-token ids mirror the real tokenized dataset / albert_config.
LABELS = ["L1HS", "L1PA2", "NEGATIVE"]
N_SPECIAL = 5
PAD_ID, CLS_ID, SEP_ID, UNK_ID, MASK_ID = range(N_SPECIAL)

# Canonical model-facing schema. A drift here (column added/removed/retyped by
# preprocessing) must fail loudly — see ``assert_model_schema``. ``transcript_id``
# is an auxiliary column present only in synthetic data (used for leakage tests).
MODEL_COLUMNS = ("label", "input_ids", "token_type_ids", "attention_mask")


# ---------------------------------------------------------------------------
# Size / complexity configuration
# ---------------------------------------------------------------------------

_PRESETS = {
    # name:   (n_transcripts, reads_per_transcript, seq_len, vocab_size)
    "tiny": (24, 4, 16, 64),  # CI default: ~96 reads, < a few seconds
    "small": (60, 6, 24, 128),  # local sanity at higher signal
    "medium": (160, 8, 48, 256),  # stress / profiling; still CPU-friendly
}


@dataclass(frozen=True)
class SizeConfig:
    """Every knob that controls generated dataset size and model footprint."""

    n_transcripts: int = 24
    reads_per_transcript: int = 4
    seq_len: int = 16
    vocab_size: int = 64
    test_split: float = 0.25
    val_split: float = 0.25

    @classmethod
    def from_preset(cls, name: str) -> "SizeConfig":
        if name not in _PRESETS:
            raise ValueError(f"unknown size preset {name!r}; choose from {sorted(_PRESETS)}")
        nt, rpt, sl, vs = _PRESETS[name]
        return cls(n_transcripts=nt, reads_per_transcript=rpt, seq_len=sl, vocab_size=vs)

    @classmethod
    def from_env(cls, default: str = "tiny") -> "SizeConfig":
        """Read ``TRAP_TEST_DATASET_SIZE`` (preset name); fall back to *default*."""
        return cls.from_preset(os.environ.get("TRAP_TEST_DATASET_SIZE", default))

    @property
    def n_reads(self) -> int:
        return self.n_transcripts * self.reads_per_transcript


# ---------------------------------------------------------------------------
# Synthetic generators (class-separable → learnable)
# ---------------------------------------------------------------------------


def _label_bands(vocab_size: int, n_labels: int) -> list[tuple[int, int]]:
    """Disjoint ``[lo, hi)`` token bands, one per label, over the non-special ids."""
    span = (vocab_size - N_SPECIAL) // n_labels
    if span < 1:
        raise ValueError(
            f"vocab_size={vocab_size} too small for {n_labels} labels + {N_SPECIAL} specials"
        )
    return [(N_SPECIAL + i * span, N_SPECIAL + (i + 1) * span) for i in range(n_labels)]


def _features() -> Features:
    return Features(
        {
            "label": ClassLabel(names=LABELS),
            "input_ids": Sequence(Value("int32")),
            "token_type_ids": Sequence(Value("int8")),
            "attention_mask": Sequence(Value("int8")),
            "transcript_id": Value("string"),
        }
    )


def build_synthetic_dataset(size: SizeConfig, seed: int) -> DatasetDict:
    """Class-separable classification dataset with transcript-level splits.

    Each transcript (and all its reads) belongs to one label; the read body is
    drawn from that label's disjoint token band, so the classes are linearly
    separable and a tiny model converges quickly. Returns a ``DatasetDict`` with
    ``train`` / ``eval`` / ``test`` splits sharing no transcript id.
    """
    rng = np.random.default_rng(seed)
    bands = _label_bands(size.vocab_size, len(LABELS))
    body_len = size.seq_len - 2  # reserve CLS + SEP
    if body_len < 1:
        raise ValueError(f"seq_len={size.seq_len} too small (need >= 3)")

    rows = []
    for t in range(size.n_transcripts):
        label_idx = t % len(LABELS)
        lo, hi = bands[label_idx]
        for _ in range(size.reads_per_transcript):
            body = rng.integers(lo, hi, size=body_len).tolist()
            ids = [CLS_ID] + body + [SEP_ID]
            rows.append(
                {
                    "label": label_idx,
                    "input_ids": ids,
                    "token_type_ids": [0] * size.seq_len,
                    "attention_mask": [1] * size.seq_len,
                    "transcript_id": f"T{t:05d}",
                }
            )
    ds = Dataset.from_list(rows, features=_features())
    return _transcript_level_split(ds, size.test_split, size.val_split, seed)


def build_synthetic_mlm_dataset(
    size: SizeConfig, seed: int, mask_rate: float = 0.15
) -> DatasetDict:
    """Tiny MLM dataset: deterministic masking → ``input_ids`` + ``labels``.

    Non-masked positions get label ``-100`` (ignored); masked positions keep the
    original id as the target and have their input replaced by ``[MASK]``.
    Compatible with ``transformers.default_data_collator``.
    """
    rng = np.random.default_rng(seed)
    body_len = size.seq_len - 2
    rows = []
    for t in range(size.n_transcripts):
        lo, hi = _label_bands(size.vocab_size, len(LABELS))[t % len(LABELS)]
        for _ in range(size.reads_per_transcript):
            body = rng.integers(lo, hi, size=body_len).tolist()
            ids = [CLS_ID] + body + [SEP_ID]
            labels = [-100] * size.seq_len
            for pos in range(1, size.seq_len - 1):  # never mask CLS/SEP
                if rng.random() < mask_rate:
                    labels[pos] = ids[pos]
                    ids[pos] = MASK_ID
            rows.append(
                {
                    "input_ids": ids,
                    "token_type_ids": [0] * size.seq_len,
                    "attention_mask": [1] * size.seq_len,
                    "labels": labels,
                }
            )
    ds = Dataset.from_list(rows)
    half = max(1, len(ds) // 5)
    return DatasetDict(
        {
            "train": ds.select(range(len(ds) - half)),
            "eval": ds.select(range(len(ds) - half, len(ds))),
        }
    )


# ---------------------------------------------------------------------------
# Real-data discovery + seeded stratified sampling
# ---------------------------------------------------------------------------


def discover_real_classification_dataset(
    processed_root: str | os.PathLike = "data/processed",
) -> Optional[Path]:
    """Return the first on-disk ``*/classification/tokenized`` DatasetDict, or None."""
    root = Path(processed_root)
    if not root.exists():
        return None
    for cand in sorted(root.glob("*/classification/tokenized")):
        if (cand / "dataset_dict.json").exists():
            return cand
    return None


def _stratified_indices(
    labels: list[int], n_per_class: int, rng: np.random.Generator
) -> list[int]:
    """Deterministic stratified row indices: up to *n_per_class* per label."""
    labels_arr = np.asarray(labels)
    picked: list[int] = []
    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        take = min(n_per_class, len(cls_idx))
        picked.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return sorted(int(i) for i in picked)


def sample_real_classification_dataset(path: Path, size: SizeConfig, seed: int) -> DatasetDict:
    """Seeded, stratified, column-normalised subset of a real tokenized dataset.

    Caps each split to roughly ``size.n_reads`` rows (stratified by label). Keeps
    only :data:`MODEL_COLUMNS`. Sample is reproducible for a fixed ``seed``.
    """
    dd = load_from_disk(str(path))
    rng = np.random.default_rng(seed)
    out: dict[str, Dataset] = {}
    for split in ("train", "eval", "test"):
        if split not in dd:
            continue
        ds = dd[split]
        n_per_class = max(1, size.n_reads // max(1, len(set(ds["label"]))))
        idx = _stratified_indices(ds["label"], n_per_class, rng)
        sub = ds.select(idx)
        keep = [c for c in MODEL_COLUMNS if c in sub.column_names]
        out[split] = sub.remove_columns([c for c in sub.column_names if c not in keep])
    return DatasetDict(out)


def build_dataset(
    size: SizeConfig, seed: int, prefer_real: bool = True
) -> tuple[DatasetDict, str]:
    """Prefer a real-data sample; fall back to synthetic. Returns ``(dd, source)``."""
    if prefer_real:
        real = discover_real_classification_dataset()
        if real is not None:
            logger.info("workflow data: sampling real dataset at %s", real)
            return sample_real_classification_dataset(real, size, seed), "real"
    logger.info("workflow data: generating synthetic dataset (seed=%d)", seed)
    return build_synthetic_dataset(size, seed), "synthetic"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def max_token_id(dd: DatasetDict) -> int:
    """Largest token id present across all splits (for embedding-size guards)."""
    hi = 0
    for split in dd.values():
        for ids in split["input_ids"]:
            if ids:
                hi = max(hi, max(ids))
    return hi


def tiny_albert_config(dd: DatasetDict, n_labels: int = len(LABELS)):
    """A minimal, dropout-free ALBERT config sized to *dd* (deterministic + fast)."""
    from transformers import AlbertConfig

    vocab = max_token_id(dd) + 1
    seq_len = len(dd["train"][0]["input_ids"])
    return AlbertConfig(
        vocab_size=vocab,
        embedding_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=max(seq_len, 8),
        type_vocab_size=2,
        num_labels=n_labels,
        pad_token_id=PAD_ID,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Assertions + reproducibility helpers
# ---------------------------------------------------------------------------


def assert_model_schema(ds: Dataset) -> None:
    """Fail loudly if the model-facing columns drift in name or dtype."""
    missing = [c for c in MODEL_COLUMNS if c not in ds.column_names]
    assert not missing, f"schema drift: missing columns {missing} (have {ds.column_names})"
    assert isinstance(ds.features["label"], ClassLabel), "label must stay a ClassLabel"
    for col in ("input_ids", "token_type_ids", "attention_mask"):
        feat = ds.features[col]
        assert isinstance(feat, Sequence), f"{col} must be a Sequence, got {feat}"


def assert_no_split_leakage(dd: DatasetDict, key: str = "transcript_id") -> None:
    """No transcript id (or, if absent, no identical read) shared across splits."""
    names = list(dd.keys())
    if all(key in dd[n].column_names for n in names):
        sets = {n: set(dd[n][key]) for n in names}
    else:  # real data has no transcript_id → fall back to content fingerprints
        sets = {n: {tuple(x) for x in dd[n]["input_ids"]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            assert not overlap, f"leakage: {len(overlap)} shared items between {a!r} and {b!r}"


def assert_ids_within_vocab(dd: DatasetDict, vocab_size: int) -> None:
    """Every token id must be < vocab_size (guards the embedding-gather OOB bug)."""
    hi = max_token_id(dd)
    assert hi < vocab_size, f"token id {hi} >= vocab_size {vocab_size} → embedding OOB"


def assert_lengths_consistent(dd: DatasetDict) -> None:
    """``input_ids`` / ``attention_mask`` / ``token_type_ids`` lengths must match per row."""
    for name, split in dd.items():
        ex = split[0]
        cols = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in ex]
        lengths = {c: len(ex[c]) for c in cols}
        assert len(set(lengths.values())) == 1, f"{name}: inconsistent field lengths {lengths}"


def fingerprint(dd: DatasetDict) -> str:
    """Stable sha256 over (label, input_ids) of every split — for reproducibility."""
    h = hashlib.sha256()
    for name in sorted(dd.keys()):
        split = dd[name]
        h.update(name.encode())
        labels = split["label"] if "label" in split.column_names else [None] * len(split)
        for lab, ids in zip(labels, split["input_ids"]):
            h.update(repr((lab, list(ids))).encode())
    return h.hexdigest()


def write_debug_artifact(path: Path, dd: DatasetDict, size: SizeConfig, source: str) -> Path:
    """Dump a small JSON fingerprint of the run for post-hoc debugging."""
    payload = {
        "source": source,
        "size": asdict(size),
        "fingerprint": fingerprint(dd),
        "splits": {n: len(dd[n]) for n in dd},
        "label_counts": {
            n: {str(k): int(v) for k, v in zip(*np.unique(dd[n]["label"], return_counts=True))}
            for n in dd
            if "label" in dd[n].column_names
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


__all__ = [
    "LABELS",
    "MODEL_COLUMNS",
    "SizeConfig",
    "build_synthetic_dataset",
    "build_synthetic_mlm_dataset",
    "build_dataset",
    "discover_real_classification_dataset",
    "sample_real_classification_dataset",
    "tiny_albert_config",
    "max_token_id",
    "assert_model_schema",
    "assert_no_split_leakage",
    "assert_ids_within_vocab",
    "assert_lengths_consistent",
    "fingerprint",
    "write_debug_artifact",
    "replace",
]
