"""Exploration — linear baseline on hashed k-mer counts (Reviewer 3, Comment 4).

Reviewer 3 asks whether the gain of AlbertSalmon over Salmon is attributable to
the transformer rather than to read pre-filtering in general, and names simpler
read classifiers as the missing comparison: exact k-mer matching, logistic
regression or gradient boosting on k-mer counts.

This script answers the cheapest and most direct of those: multinomial logistic
regression on the *same* hashed canonical 17-mer features the transformer sees,
trained on the *same* split, scored on the *same* held-out split.  Nothing is
re-tokenized and nothing is re-simulated, so the only difference between this
baseline and the reported model is the function mapping k-mer counts to a class.

Design notes
------------
* Features are bag-of-k-mer counts over the 65,536 hash buckets.  Token ids
  0-4 are the special tokens ([CLS], <pad>, [SEP], <unk>, [MASK]) and are
  dropped: <pad> in particular would leak read length into the features.
* A linear model on counts discards k-mer *order*, which is precisely the
  information the transformer can use and this baseline cannot.  That is the
  point of the comparison, not a defect of it.
* ``SGDClassifier(loss="log_loss")`` with ``partial_fit`` keeps memory bounded:
  the full training split has ~134 k-mers per read, so a dense or even a single
  sparse matrix over all reads would not fit comfortably.
* Batches are *interleaved*, not contiguous: batch ``b`` is rows
  ``b, b+N, b+2N, ...``, so every batch spans the whole split and sees every
  class.  Preprocessing writes the classes in blocks, so a contiguous batch is
  nearly single-class and ``partial_fit`` just overwrites the previous step --
  the first smoke run collapsed exactly this way, predicting L1PA for 99.6% of
  reads and NEGATIVE for 0.01%.  Interleaving fixes the class mixing while
  keeping the read pattern regular, which a global ``Dataset.shuffle()`` would
  not: that turns every batch into a scattered gather from a memory-mapped
  Arrow file on Lustre.
* Nothing here imports torch, transformers or ``trap.modeling``.  On a cold
  Lustre cache that import chain took over 22 minutes and timed the job out
  before a single read was seen; the dataset is loaded with ``datasets``
  directly instead.  ``trap.utils.seeding`` is avoided for the same reason (it
  imports torch to seed it).

Usage:
    python scripts/python/explore_logreg_kmer_baseline.py OUT_DIR [EPOCHS] [BATCH]

``MAX_TRAIN`` / ``MAX_EVAL`` cap the rows used, for a smoke run.  A capped run
writes ``capped: true`` into its manifest so its numbers can never be mistaken
for the full result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time

print("[logreg] importing numpy/scipy/sklearn/datasets ...", flush=True)

from datasets import load_from_disk
from loguru import logger
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import normalize

from trap.config import manifest
import trap.config.config  # noqa: F401  (STAGE level + tqdm-loguru)
from trap.config.config import PROCESSED_DATA_DIR

N_HASH = 65536
FIRST_KMER_ID = 5  # 0-4 are [CLS], <pad>, [SEP], <unk>, [MASK]
SEED = 3469  # the seed of the model of record, so the comparison shares it
ALPHA = 1e-5
NORM = "l2"


def batch_to_csr(id_lists) -> csr_matrix:
    """Bag-of-k-mer count matrix for one batch of tokenized reads."""
    kept = []
    lengths = np.empty(len(id_lists), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        a = np.asarray(ids, dtype=np.int64)
        a = a[a >= FIRST_KMER_ID] - FIRST_KMER_ID
        kept.append(a)
        lengths[i] = a.size
    indices = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
    indptr = np.zeros(len(id_lists) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    m = csr_matrix(
        (np.ones(indices.size, dtype=np.float32), indices, indptr),
        shape=(len(id_lists), N_HASH),
    )
    m.sum_duplicates()  # duplicate ids become counts
    return m


def iter_batches(ds, batch_size: int, rng: random.Random | None = None):
    """Yield (input_ids, labels) batches that each span the whole split.

    Batch ``b`` is rows ``b, b+N, b+2N, ...`` for ``N`` batches, so every batch
    carries the full class mixture even though the classes are written to disk
    in blocks.  Indices within a batch stay ascending, so the gather is a
    strided read rather than a random one.  When *rng* is given, the order in
    which the batches are visited is permuted.
    """
    n = len(ds)
    n_batches = max(1, n // batch_size)
    order = list(range(n_batches))
    if rng is not None:
        rng.shuffle(order)
    for b in order:
        idx = list(range(b, n, n_batches))
        chunk = ds[idx]
        yield chunk["input_ids"], np.asarray(chunk["label"], dtype=np.int64)


def load_tokenized(preprocessing_name: str):
    """Load the tokenized classification DatasetDict and its label maps.

    Mirrors ``trap.modeling.train.load_classification_data`` without importing
    it, so no torch/transformers import is triggered.
    """
    path = Path(PROCESSED_DATA_DIR) / preprocessing_name / "classification" / "tokenized"
    lm_datasets = load_from_disk(str(path))
    class_labels = lm_datasets["train"].features["label"]
    label2id = {name: class_labels.str2int(name) for name in class_labels.names}
    return lm_datasets, label2id, {v: k for k, v in label2id.items()}


def main() -> None:
    out_dir = Path(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000

    random.seed(SEED)
    np.random.seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_name = "gencode.v48.k17.salmon.seqlabel/l1hs_l1pa2"
    t_load = time.time()
    lm_datasets, label2id, id2label = load_tokenized(preprocessing_name)
    logger.info(f"[logreg] dataset loaded in {time.time() - t_load:.0f}s")
    logger.log("STAGE", f"[logreg] splits: { {k: len(v) for k, v in lm_datasets.items()} }")
    logger.info(f"[logreg] labels: {label2id}")

    train_ds = lm_datasets["train"]
    eval_ds = lm_datasets["eval"]
    max_train = int(os.environ.get("MAX_TRAIN", 0))
    max_eval = int(os.environ.get("MAX_EVAL", 0))
    capped = bool(max_train or max_eval)
    if max_train:
        # Stride, not a head slice: the classes are written in blocks, so the
        # first N rows are nearly single-class. Striding keeps reads ordered.
        step = max(1, len(train_ds) // max_train)
        train_ds = train_ds.select(range(0, len(train_ds), step))
        logger.warning(f"[logreg] MAX_TRAIN -> {len(train_ds):,} rows (stride {step}) — SMOKE RUN")
    if max_eval:
        step = max(1, len(eval_ds) // max_eval)
        eval_ds = eval_ds.select(range(0, len(eval_ds), step))
        logger.warning(f"[logreg] MAX_EVAL -> {len(eval_ds):,} rows (stride {step}) — SMOKE RUN")
    classes = np.array(sorted(label2id.values()), dtype=np.int64)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=ALPHA,
        random_state=SEED,
        learning_rate="optimal",
        tol=None,
    )

    t0 = time.time()
    for epoch in range(epochs):
        rng = random.Random(SEED + epoch)
        hits = total = 0
        seen = 0
        for ids, y in iter_batches(train_ds, batch_size, rng=rng):
            x = normalize(batch_to_csr(ids), norm=NORM, copy=False)
            if seen:  # running train accuracy, to catch a collapse early
                hits += int((clf.predict(x) == y).sum())
                total += len(y)
            clf.partial_fit(x, y, classes=classes)
            seen += len(y)
            if seen % (batch_size * 20) == 0:
                logger.info(
                    f"[logreg] epoch {epoch + 1}/{epochs}  {seen:,}/{len(train_ds):,}"
                    f"  {time.time() - t0:.0f}s"
                )
        acc = hits / total if total else float("nan")
        logger.log(
            "STAGE",
            f"[logreg] epoch {epoch + 1} done ({time.time() - t0:.0f}s), "
            f"running train accuracy {acc:.4f}",
        )

    # ---- score the held-out split ---------------------------------------
    y_true, y_pred, proba = [], [], []
    for ids, y in iter_batches(eval_ds, batch_size):
        x = normalize(batch_to_csr(ids), norm=NORM, copy=False)
        p = clf.predict_proba(x)
        proba.append(p.astype(np.float32))
        y_pred.append(p.argmax(axis=1))
        y_true.append(y)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    proba = np.vstack(proba)
    logger.log("STAGE", f"[logreg] scored {len(y_true):,} held-out rows")

    names = [id2label[i] for i in classes]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
    }
    for i, name in enumerate(names):
        summary[f"precision_{name}"] = float(prec[i])
        summary[f"recall_{name}"] = float(rec[i])
        summary[f"f1_{name}"] = float(f1[i])
        summary[f"support_{name}"] = int(sup[i])

    (out_dir / "logreg_summary.eval.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out_dir / "logreg_per_class.eval.csv").open("w") as fh:
        fh.write("Class,F1,Precision,Recall,N\n")
        for i, name in enumerate(names):
            fh.write(f"{name},{f1[i]},{prec[i]},{rec[i]},{sup[i]}\n")
        fh.write(
            f"Macro avg,{summary['f1']},{prec.mean()},{rec.mean()},{int(sup.sum())}\n"
        )

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with (out_dir / "logreg_confusion.eval.csv").open("w") as fh:
        fh.write("true," + ",".join(f"pred_{n}" for n in names) + "\n")
        for i, name in enumerate(names):
            fh.write(name + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")

    manifest.write(
        out_dir,
        filename="manifest.logreg_kmer_baseline.json",
        command="explore_logreg_kmer_baseline",
        seed=SEED,
        capped=capped,
        params={
            "preprocessing_name": preprocessing_name,
            "split": "eval",
            "n_hash": N_HASH,
            "first_kmer_id": FIRST_KMER_ID,
            "features": "bag-of-k-mer counts, order discarded",
            "normalization": NORM,
            "model": "SGDClassifier(loss=log_loss)",
            "alpha": ALPHA,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_train": len(train_ds),
            "n_eval": len(eval_ds),
        },
    )

    logger.success(f"[logreg] macro F1 {summary['f1']:.4f}  accuracy {summary['accuracy']:.4f}")
    for name in names:
        logger.success(
            f"[logreg]   {name}: F1 {summary[f'f1_{name}']:.4f}"
            f"  P {summary[f'precision_{name}']:.4f}"
            f"  R {summary[f'recall_{name}']:.4f}"
        )
    logger.success(f"[logreg] -> {out_dir}")


if __name__ == "__main__":
    main()
