"""k -> classification ablation: does a longer k-mer help the task?

Reviewer 3 noted that an entropy plateau alone does not establish an *optimal*
k-mer length for LINE-1 classification.  This module measures, directly, whether
increasing k improves the separability of the three task classes
(``L1HS`` / ``L1PA`` / ``NEGATIVE``).

For each k the labelled reads are turned into hashed k-mer frequency vectors and
a multinomial logistic-regression classifier is scored with stratified
cross-validation (macro-F1 and accuracy).  A linear model on k-mer frequencies
is a deliberately lightweight, fully deterministic proxy for the full ALBERT
pipeline: it isolates the contribution of k-mer *resolution* from model capacity
and trains in seconds, so the whole ablation curve is cheap and reproducible.
The output table is designed to be overlaid on the entropy curve in the notebook;
genuine ALBERT accuracies can be dropped into the same schema if available.

Read labels are parsed from TrAP record ids (``...|LABEL-...``), mirroring
``scripts/diagnostics/kmer_overlap.py``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from loguru import logger
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import normalize

from trap.analysis.kmer_counting import _forward_codes
from trap.utils.canonical_kmer import canonical_codes, splitmix64

# task_label / TASK_CLASSES now live in trap.utils.labels (stdlib-only, shared with
# the classification preprocessing 3-class collapse); re-exported here for callers.
from trap.utils.labels import TASK_CLASSES, task_label  # noqa: F401
from trap.utils.seeding import seeded_rng
from trap.utils.sequence import standardize


def load_labeled_reads(
    reads_path: Union[str, Path],
    max_per_class: int = 5000,
    seed: int = 3469,
) -> Tuple[List[str], List[str]]:
    """Reservoir-sample standardised reads grouped by task class.

    Args:
        reads_path: FASTQ with TrAP labelled headers.
        max_per_class: Cap on reads kept per task class (balances classes and
            bounds runtime).
        seed: Reservoir-sampling seed.

    Returns:
        ``(sequences, labels)`` restricted to the three task classes.
    """
    rng = seeded_rng(seed)
    buckets: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, int] = defaultdict(int)
    read_id = None
    with open(reads_path) as handle:
        for i, line in enumerate(handle):
            phase = i % 4
            if phase == 0:
                read_id = line[1:].strip()
            elif phase == 1:
                label = task_label(read_id)
                if label == "OTHER":
                    continue
                seq = standardize(line.strip())
                if not seq:
                    continue
                seen[label] += 1
                bucket = buckets[label]
                if len(bucket) < max_per_class:
                    bucket.append(seq)
                else:
                    j = int(rng.integers(0, seen[label]))
                    if j < max_per_class:
                        bucket[j] = seq

    sequences: List[str] = []
    labels: List[str] = []
    for label, seqs in buckets.items():
        sequences.extend(seqs)
        labels.extend([label] * len(seqs))
    logger.log(
        "STAGE",
        "Loaded reads per class: "
        + ", ".join(f"{lbl}={len(buckets[lbl]):,}" for lbl in sorted(buckets)),
    )
    return sequences, labels


def featurize(
    sequences: Sequence[str],
    k: int,
    n_features: int,
    canonical: bool = False,
) -> sparse.csr_matrix:
    """Hashed k-mer frequency matrix (L1-normalised rows).

    The hashing trick keeps the feature dimension fixed across all k, so memory
    is bounded and the classifier is comparable between k values.

    Args:
        sequences: Standardised ACTG reads.
        k: K-mer length.
        n_features: Hash buckets (feature-space dimension).
        canonical: Collapse reverse complements before hashing.

    Returns:
        ``(n_reads, n_features)`` sparse CSR matrix of per-read k-mer
        frequencies.
    """
    encode = canonical_codes if canonical else _forward_codes
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for r, seq in enumerate(sequences):
        codes = encode(seq, k)
        if codes.size == 0:
            continue
        buckets = (splitmix64(codes) % np.uint64(n_features)).astype(np.int64)
        uniq, counts = np.unique(buckets, return_counts=True)
        rows.append(np.full(uniq.size, r, dtype=np.int64))
        cols.append(uniq)
        vals.append(counts.astype(np.float64))

    n_reads = len(sequences)
    if not rows:
        return sparse.csr_matrix((n_reads, n_features), dtype=np.float64)
    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_reads, n_features),
    )
    return normalize(matrix, norm="l1", copy=False)


def ablation(
    reads_path: Union[str, Path],
    k_values: Sequence[int],
    max_per_class: int = 5000,
    n_features: int = 1 << 18,
    canonical: bool = False,
    n_splits: int = 5,
    seed: int = 3469,
) -> List[Dict[str, object]]:
    """Cross-validated macro-F1 / accuracy of an L1 classifier per k.

    Args:
        reads_path: FASTQ with TrAP labelled headers.
        k_values: K-mer lengths to evaluate.
        max_per_class: Reads kept per class (balances + bounds runtime).
        n_features: Hash buckets for featurisation.
        canonical: Collapse reverse complements.
        n_splits: Stratified CV folds (clamped to the smallest class count).
        seed: Master seed for sampling, folds, and the solver.

    Returns:
        One tidy dict row per k with mean/std macro-F1 and accuracy.
    """
    sequences, labels = load_labeled_reads(reads_path, max_per_class, seed)
    if len(set(labels)) < 2:
        raise ValueError(
            f"need >= 2 task classes for the ablation; found {sorted(set(labels))}"
        )

    y = np.array(labels)
    _, class_counts = np.unique(y, return_counts=True)
    folds = max(2, min(n_splits, int(class_counts.min())))
    if folds < n_splits:
        logger.warning(f"reducing CV folds {n_splits} -> {folds} (smallest class limits it)")

    rows: List[Dict[str, object]] = []
    for k in k_values:
        features = featurize(sequences, k, n_features, canonical=canonical)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
        scores = cross_validate(
            classifier,
            features,
            y,
            cv=splitter,
            scoring=("accuracy", "f1_macro"),
            n_jobs=1,
        )
        rows.append(
            {
                "k": int(k),
                "n_reads": int(len(sequences)),
                "n_classes": int(len(set(labels))),
                "n_features": int(n_features),
                "canonical": bool(canonical),
                "cv_folds": int(folds),
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "macro_f1_mean": float(scores["test_f1_macro"].mean()),
                "macro_f1_std": float(scores["test_f1_macro"].std()),
            }
        )
        logger.log(
            "STAGE",
            f"k={k}: macro-F1={rows[-1]['macro_f1_mean']:.3f} "
            f"acc={rows[-1]['accuracy_mean']:.3f}",
        )
    return rows
