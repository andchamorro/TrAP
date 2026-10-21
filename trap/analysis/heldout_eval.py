"""Held-out split evaluation for the reported classifier (Reviewer 3, Comment 3).

Why this exists
---------------
``trap.modeling.train.classification`` builds its ``Trainer`` with
``eval_dataset=lm_datasets["test"]`` (the per-epoch monitoring split) and then
calls ``trainer.evaluate(eval_dataset=lm_datasets["eval"])`` once training ends
to score the held-out split.  That second call does not do what it reads like:
``classification_final.json`` sets ``dataloader_persistent_workers``, and
``WeightedLossTrainer.get_eval_dataloader`` keys its dataloader cache on the
literal string ``"eval"`` whenever a ``Dataset`` *object* is passed rather than
a split name.  The per-epoch evaluations have already cached a loader under that
key, so the final call replays the monitoring split and the held-out split is
never scored.  ``eval_results.json`` shows the signature of this: ``eval_f1``
and ``eval_loss`` are bit-identical to the best per-epoch entry in
``trainer_state.json``, while ``eval_samples`` carries the *other* split's row
count because ``train.py`` overwrites it with ``len(lm_datasets["eval"])``.

This module scores a named split once, from a finished checkpoint, with the
cache disabled, and writes the per-class table plus the confusion matrix that
Reviewer 3 asked for.  It trains nothing.

Metric definitions are imported from ``trap.modeling.train`` so the numbers are
comparable with the reported training-time metrics rather than merely similar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
import numpy as np
import pandas as pd


def _load_split(preprocessing_name: str, split: str):
    """Load one split of the tokenized classification dataset."""
    from trap.modeling import train as train_mod

    lm_datasets, label2id, id2label = train_mod.load_classification_data(preprocessing_name)
    if split not in lm_datasets:
        raise KeyError(f"split {split!r} not in {sorted(lm_datasets)}")
    return lm_datasets[split], id2label


def _precision_kwargs() -> Tuple[Dict[str, bool], str]:
    """Pick the fastest eval precision the current device actually supports.

    ``TrainingArguments(bf16=True)`` *raises* on a card without bf16 rather than
    falling back, so this cannot be keyed on ``cuda.is_available()`` alone: Ampere
    and newer (A100) have bf16, Turing (T4, RTX 6000) does not. Those cards get
    fp16 instead, whose 10-bit mantissa is finer than bf16's 7 — and since this is
    a forward pass only, bf16's wider exponent range buys nothing here.

    Returns:
        ``(TrainingArguments kwargs, label)``; the label is recorded in the run
        summary and manifest so a number can always be traced to its precision.
    """
    import torch

    if not torch.cuda.is_available():
        return {}, "fp32"
    if torch.cuda.is_bf16_supported():
        return {"bf16": True}, "bf16"
    return {"fp16": True}, "fp16"


def _predict_logits(
    model_path: Path, dataset, batch_size: int, num_workers: int
) -> Tuple[np.ndarray, str]:
    """Return the raw logits for *dataset* under the checkpoint at *model_path*.

    A fresh ``Trainer`` is built per call with ``dataloader_persistent_workers``
    off, so no eval-dataloader cache can shadow the split being scored.

    Returns:
        ``(logits, precision_label)``.
    """
    import torch  # noqa: F401  (imported for parity with _precision_kwargs)
    from transformers import (
        AlbertForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from trap.modeling import train as train_mod

    model = AlbertForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = train_mod.load_tuned_tokenizer(
        str(model_path), model.config.max_position_embeddings
    )
    precision_kwargs, precision = _precision_kwargs()
    args = TrainingArguments(
        output_dir=str(model_path.parent / "heldout_tmp"),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        report_to="none",
        **precision_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    logger.info(
        f"[heldout] scoring {dataset.num_rows:,} rows from {model_path} "
        f"at {precision}, batch {batch_size}"
    )
    return trainer.predict(dataset).predictions, precision


def _metric_frames(
    logits: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Build the per-class table, the confusion matrix and the macro summary."""
    from trap.modeling.train import make_compute_classification_metrics

    metrics = make_compute_classification_metrics(id2label=id2label)((logits, labels))
    names = [id2label[i] for i in sorted(id2label)]

    per_class = pd.DataFrame(
        [
            {
                "Class": n,
                "F1": metrics[f"f1_{n}"],
                "Precision": metrics[f"precision_{n}"],
                "Recall": metrics[f"recall_{n}"],
                "PR-AUC": metrics[f"pr_auc_{n}"],
                "ROC-AUC": metrics[f"roc_auc_{n}"],
                "N": int(metrics[f"support_{n}"]),
            }
            for n in names
        ]
    )
    per_class.loc[len(per_class)] = {
        "Class": "Macro avg",
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "PR-AUC": metrics["pr_auc"],
        "ROC-AUC": metrics["roc_auc"],
        "N": int(per_class["N"].sum()),
    }

    confusion = pd.DataFrame(
        [
            {"true": t, **{f"pred_{p}": int(metrics[f"cm_{t}_as_{p}"]) for p in names}}
            for t in names
        ]
    )
    summary = {k: v for k, v in metrics.items() if not k.startswith("cm_")}
    return per_class, confusion, summary
