"""Label analysis utilities for genomic classification datasets (DR-2).

Previously ``analyze_labels`` lived in ``preprocessing_sequences.py``.
Centralising it here lets R/Python notebooks import it directly.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict

from loguru import logger
import numpy as np

# The classification task collapses the ~130 RepeatMasker LINE-1 subfamilies into
# three biologically meaningful classes: human-specific (L1HS), other primate L1
# (L1PA/L1P*), and reads with no LINE-1 overlap (NEGATIVE). Ancient mammalian-wide
# subfamilies (L1M*, L1ME*, …) map to OTHER and are dropped from the 3-class task.
TASK_CLASSES = ("L1HS", "L1PA", "NEGATIVE")

# Priority for resolving a read that overlaps more than one subfamily to a single
# label (highest first): the youngest/most-specific class wins (L1HS beats L1PA).
# A label not listed here (e.g. OTHER) ranks lowest, so an OTHER+real overlap keeps
# the real class and an OTHER-only fragment is dropped by the OTHER filter.
TASK_PRIORITY = ("L1HS", "L1PA", "NEGATIVE")


def task_label(read_id: str) -> str:
    """Map a TrAP record id to a task class, or ``"OTHER"``.

    Collapses the raw RepeatMasker subfamily label (parsed from the record id)
    into the three task classes. ``OTHER`` covers LINE-1 subfamilies that are
    neither L1HS nor L1PA*; callers building the 3-class dataset drop these.

    Args:
        read_id: FASTQ/FASTA header (without the leading ``@``).

    Returns:
        One of ``L1HS``, ``L1PA``, ``NEGATIVE``, or ``OTHER``.
    """
    parts = read_id.split("|")
    raw = parts[-1].split("-")[0] if len(parts) >= 2 else read_id
    upper = raw.upper()
    if upper == "NEGATIVE":
        return "NEGATIVE"
    if upper.startswith("L1HS"):
        return "L1HS"
    if upper.startswith("L1PA") or upper.startswith("L1P"):
        return "L1PA"
    return "OTHER"


def analyze_labels(transcripts_dataset) -> Dict[str, int]:
    """Log and return per-class read counts from a split HuggingFace dataset.

    Compares each non-training split to the training set and warns when a
    label appears in validation or test but not in training.

    Args:
        transcripts_dataset: ``DatasetDict`` with at least a ``"train"``
            split containing a ``ClassLabel`` ``"label"`` column.

    Returns:
        Dict mapping label name → count (training split only).
    """
    int2str = transcripts_dataset["train"].features["label"].int2str
    labels = [int2str(i) for i in transcripts_dataset["train"]["label"]]

    for split in ("eval", "test"):
        if split not in transcripts_dataset:
            continue
        split_int2str = transcripts_dataset[split].features["label"].int2str
        split_labels = [split_int2str(i) for i in transcripts_dataset[split]["label"]]
        diff = set(split_labels) - set(labels)
        if diff:
            logger.warning(f"Labels {diff!r} appear in {split!r} split but not in training set")

    label_counts = Counter(labels)
    if not label_counts:
        logger.warning("Training split has no labelled reads; skipping label histogram.")
        return {}

    quartiles = np.quantile(list(label_counts.values()), [0.25, 0.5, 0.75])
    logger.info(f"Label Q1/Q2/Q3: {quartiles[0]:.0f} / {quartiles[1]:.0f} / {quartiles[2]:.0f}")

    max_count = max(label_counts.values())
    scale = 40 / max_count
    lines = ["\nLabels histogram:"]
    for label, count in sorted(label_counts.items()):
        bar = "#" * int(count * scale)
        lines.append(f"  {label:<12} {bar} ({count})")
    logger.info("\n".join(lines))

    return dict(label_counts)
