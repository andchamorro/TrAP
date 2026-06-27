"""Track 1 — decision-boundary & calibration optimisation for the L1 classifier.

Implements the cheapest macro-F1 lift in the L1PA-vs-NEGATIVE specificity plan
(``.trap/plans/l1pa-negative-specificity-data-centric.md`` §2): the model emits
3-way probabilities and the *rule* mapping them to a label is tunable, without
any retrain.

Two stages, deliberately decoupled so the analytical heart is CPU-only and
fully testable:

  * ``dump-logits`` (GPU / Grace) — run a trained classifier over the held-out
    eval split of the tokenized classification dataset and save per-read
    ``(logits, label)`` to an ``.npz``. Reusable inference infra.
  * ``optimize`` (CPU) — consume the ``.npz``; fit temperature scaling (§2.1),
    optimise per-class weights for macro-F1 (§2.2, the cost-sensitive boundary),
    sweep the NEGATIVE-score cutoff that ``postprocessing.py`` consumes, and
    report the macro-F1 / L1PA-precision / L1PA-recall trade vs the argmax
    baseline (§2.3). Emits an operating-point JSON.

The ``optimize`` step splits the dumped eval predictions into a *fit* half (for
temperature + weights) and a held-out *report* half (the scorecard), so the
reported lift is not optimistic.

Methodology anchor: Tran et al. 2025 §3.3 (calibration) and §3.4 (evaluation);
the cost-sensitive boundary is the practical form of the Bayes-optimal decision
region (plan Track 6).

Usage::

    # GPU (Grace): dump eval logits for the Salmon classifier
    python -m trap.analysis.boundary_calibration dump-logits \
        --model-path models/albert.l1hs_l1pa2.v48.k17.salmon/final \
        --tokenizer-path models/tokenizer.gencode.v48.k17.salmon \
        --preprocessing-name gencode.v48.k17.salmon/l1hs_l1pa2 \
        --max-position-embeddings 280 \
        --out results/eval_logits.salmon.npz

    # CPU: calibrate + optimise the boundary, emit the operating point
    python -m trap.analysis.boundary_calibration optimize \
        --logits results/eval_logits.salmon.npz \
        --out results/boundary_operating_point.salmon.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import typer

from trap.config.config import RESULTS_DIR
from trap.utils.seeding import set_global_seed

app = typer.Typer(
    add_completion=False, help="Decision-boundary & calibration optimisation (Track 1)."
)

BASELINE_MACRO_F1 = 0.664  # Salmon argmax deploy baseline (plan §0).


# --------------------------------------------------------------------------- #
# Pure numpy core (CPU, fully testable)                                        #
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits`` after dividing by ``temperature``.

    Args:
        logits: ``(N, C)`` real-valued class logits.
        temperature: Scalar > 0; ``T>1`` softens, ``T<1`` sharpens.

    Returns:
        ``(N, C)`` probabilities summing to 1 per row.
    """
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean NLL of ``labels`` under the temperature-scaled softmax of ``logits``."""
    probs = softmax(logits, temperature)
    n = labels.shape[0]
    p_true = probs[np.arange(n), labels]
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit the single temperature scalar that minimises NLL (Guo et al. 2017).

    Args:
        logits: ``(N, C)`` eval logits.
        labels: ``(N,)`` integer class ids.

    Returns:
        The optimal temperature ``T`` in ``[0.05, 10]``.
    """
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda t: negative_log_likelihood(logits, labels, t),
        bounds=(0.05, 10.0),
        method="bounded",
    )
    return float(res.x)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error on the top-1 confidence (equal-width bins).

    Args:
        probs: ``(N, C)`` probabilities.
        labels: ``(N,)`` integer class ids.
        n_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        ECE — the support-weighted mean ``|confidence - accuracy|`` over bins.
    """
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = labels.shape[0]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def predict_weighted(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted-argmax decision rule: ``argmax_c weights[c] * probs[:, c]``.

    Equal weights reproduce the plain argmax baseline. Raising a class weight
    lowers that class's effective decision threshold (more recall, less
    precision); this is the cost-sensitive boundary of plan §2.2.
    """
    return np.asarray(probs * np.asarray(weights)[None, :]).argmax(axis=1)


def macro_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    """Unweighted mean per-class F1 (matches the training selection metric)."""
    f1s = []
    for c in range(num_classes):
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def optimize_class_weights(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    neg_index: int,
    grid: Optional[np.ndarray] = None,
    refine: bool = True,
) -> Tuple[np.ndarray, float]:
    """Grid-search per-class weights maximising macro-F1 under ``predict_weighted``.

    The NEGATIVE weight is pinned to 1.0 (the rule is invariant to global
    scaling), so only the L1HS and L1PA weights are free — a cheap 2-D search.
    A coarse log-spaced grid is optionally refined around the best cell.

    Args:
        probs: ``(N, C)`` calibrated probabilities.
        labels: ``(N,)`` integer class ids.
        num_classes: ``C``.
        neg_index: Index of the NEGATIVE class (pinned weight).
        grid: Candidate weight multipliers; defaults to log-spaced ``[0.25, 4]``.
        refine: If true, a second finer grid is searched around the best coarse
            weights.

    Returns:
        ``(weights, macro_f1)`` — the best length-``C`` weight vector and its
        macro-F1 on ``(probs, labels)``.
    """
    if grid is None:
        grid = np.geomspace(0.25, 4.0, 25)
    free = [c for c in range(num_classes) if c != neg_index]

    def _search(candidates_per_class) -> Tuple[np.ndarray, float]:
        best_w = np.ones(num_classes)
        best_f1 = -1.0
        # Cartesian product over the free classes' candidate weights.
        from itertools import product

        for combo in product(*candidates_per_class):
            w = np.ones(num_classes)
            for c, val in zip(free, combo):
                w[c] = val
            f1 = macro_f1(labels, predict_weighted(probs, w), num_classes)
            if f1 > best_f1:
                best_f1, best_w = f1, w
        return best_w, best_f1

    best_w, best_f1 = _search([grid for _ in free])

    if refine:
        fine = []
        for c in free:
            center = best_w[c]
            fine.append(np.geomspace(center / 1.6, center * 1.6, 17))
        refined_w, refined_f1 = _search(fine)
        if refined_f1 >= best_f1:
            best_w, best_f1 = refined_w, refined_f1

    return best_w, best_f1


def sweep_negative_threshold(
    probs: np.ndarray, labels: np.ndarray, neg_index: int, n_steps: int = 200
) -> Tuple[float, float]:
    """Best NEGATIVE-score cutoff for the binary L1-vs-NEGATIVE decision.

    Maps the 3-class problem to the single cutoff that ``postprocessing.py``
    consumes: a read is *positive* (L1HS or L1PA) iff ``P(NEGATIVE) < tau``.
    The cutoff maximising binary macro-F1 (positive vs NEGATIVE) is returned so
    the downstream read filter inherits the calibrated operating point.

    Returns:
        ``(tau, binary_macro_f1)``.
    """
    p_neg = probs[:, neg_index]
    truth_pos = (labels != neg_index).astype(int)
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.0, 1.0, n_steps + 1):
        pred_pos = (p_neg < tau).astype(int)
        f1 = macro_f1(truth_pos, pred_pos, 2)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1


def scorecard(labels: np.ndarray, predictions: np.ndarray, classes: List[str]) -> Dict[str, float]:
    """Per-class P/R/F1 + macro-F1 + L1PA<->NEGATIVE confusion cells.

    Returns a flat dict (the single comparable scorecard of plan §8) with keys
    ``macro_f1``, ``precision_<CLASS>``/``recall_<CLASS>``/``f1_<CLASS>``,
    ``support_<CLASS>`` and ``cm_<TRUE>_as_<PRED>``.
    """
    num_classes = len(classes)
    out: Dict[str, float] = {"macro_f1": macro_f1(labels, predictions, num_classes)}
    for c, name in enumerate(classes):
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"precision_{name}"] = precision
        out[f"recall_{name}"] = recall
        out[f"f1_{name}"] = f1
        out[f"support_{name}"] = int((labels == c).sum())
    for t, tname in enumerate(classes):
        for p, pname in enumerate(classes):
            out[f"cm_{tname}_as_{pname}"] = int(((labels == t) & (predictions == p)).sum())
    return out


# --------------------------------------------------------------------------- #
# Stage A — dump eval logits (GPU)                                             #
# --------------------------------------------------------------------------- #
@app.command("dump-logits")
def dump_logits(
    model_path: Path = typer.Option(..., help="Trained classifier dir (…/final)."),
    tokenizer_path: Path = typer.Option(..., help="Tokenizer dir (SPM or Salmon)."),
    preprocessing_name: str = typer.Option(
        ..., help="Tokenized classification dataset name under data/processed/."
    ),
    split: str = typer.Option("eval", help="Dataset split to score (falls back to test/train)."),
    max_position_embeddings: int = typer.Option(280, help="Pad/truncate length."),
    batch_size: int = typer.Option(512),
    max_reads: int = typer.Option(
        400_000,
        help="Cap on reads scored (shuffled subsample). Calibration + the 2-D "
        "weight grid are stable well below the full multi-million-read split; "
        "0 = score the whole split.",
    ),
    num_workers: int = typer.Option(4, help="DataLoader workers feeding the GPU."),
    out: Path = typer.Option(RESULTS_DIR / "eval_logits.npz"),
    seed: int = typer.Option(42),
):
    """Run the classifier over a held-out split; save ``(logits, labels)`` to npz.

    Reuses the already-tokenized classification ``DatasetDict`` saved by
    preprocessing (``…/classification/tokenized``) so tokenisation matches
    training exactly. Output npz holds ``logits`` ``(N, C)``, ``labels`` ``(N,)``
    and ``classes`` (class names from ``model.config.id2label``).

    The full L1 eval split is multi-million reads; scoring it all in fp32 blew
    the 1h wall. So forward runs under bf16 autocast and ``--max-reads`` caps a
    shuffled subsample (class proportions preserved, so rare L1HS keeps support).
    """
    import os

    from datasets import load_from_disk
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AlbertForSequenceClassification, DataCollatorWithPadding

    from trap.config.config import PROCESSED_DATA_DIR
    from trap.loaders.tokenizer import load_kmer_tokenizer
    from trap.modeling._inference import autocast_ctx, resolve_device

    set_global_seed(seed)
    device = resolve_device()
    tokenizer = load_kmer_tokenizer(str(tokenizer_path), max_position_embeddings)
    model = AlbertForSequenceClassification.from_pretrained(str(model_path)).to(device).eval()
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]

    ds = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "classification", "tokenized")
    )
    if split not in ds:
        fallback = next((s for s in ("eval", "test", "train") if s in ds), None)
        if fallback is None:
            raise typer.BadParameter(f"split {split!r} not in dataset (have {list(ds)})")
        logger.warning(f"split {split!r} absent; falling back to {fallback!r}")
        split = fallback
    ds = ds[split]
    keep = [
        c
        for c in ("input_ids", "attention_mask", "token_type_ids", "label")
        if c in ds.column_names
    ]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    full_n = len(ds)
    if max_reads and full_n > max_reads:
        ds = ds.shuffle(seed=seed).select(range(max_reads))
    logger.info(
        f"device={device} classes={classes} split={split} "
        f"n={len(ds):,} (of {full_n:,}) batch={batch_size}"
    )

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    n_batches = (len(ds) + batch_size - 1) // batch_size
    with torch.inference_mode(), autocast_ctx(device):
        for batch in tqdm(loader, total=n_batches, desc=f"scoring {split}"):
            labels = batch.pop("labels", batch.pop("label", None))
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inp).logits.float().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(np.asarray(labels))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=logits, labels=labels, classes=np.asarray(classes))
    logger.success(f"Wrote {logits.shape[0]:,} eval predictions -> {out}")


# --------------------------------------------------------------------------- #
# Stage B — calibrate + optimise the boundary (CPU)                           #
# --------------------------------------------------------------------------- #
def _format_scorecard(tag: str, card: Dict[str, float], classes: List[str]) -> str:
    """Render a scorecard dict as a compact log block."""
    lines = [f"  [{tag}] macro_f1={card['macro_f1']:.4f}"]
    for name in classes:
        lines.append(
            f"    {name:<9} P={card[f'precision_{name}']:.3f} "
            f"R={card[f'recall_{name}']:.3f} F1={card[f'f1_{name}']:.3f} "
            f"n={card[f'support_{name}']}"
        )
    return "\n".join(lines)


def _load_logits_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load an ``(logits, labels, classes)`` npz written by ``dump-logits``."""
    data = np.load(path, allow_pickle=True)
    return (
        data["logits"].astype(np.float64),
        data["labels"].astype(np.int64),
        [str(c) for c in data["classes"]],
    )


@app.command("optimize")
def optimize(
    logits: Path = typer.Option(
        ..., help="npz from dump-logits used to FIT temperature + weights (the eval/val split)."
    ),
    report_logits: Path = typer.Option(
        None,
        help="Second npz to REPORT the scorecard on (the held-out test split). "
        "When omitted, --logits is split 50/50 into fit/report halves.",
    ),
    out: Path = typer.Option(RESULTS_DIR / "boundary_operating_point.json"),
    beta: float = typer.Option(
        1.0, help="Fβ weight on L1PA recall in the weight objective (β>1 favours recall)."
    ),
    n_bins: int = typer.Option(15, help="ECE confidence bins."),
    seed: int = typer.Option(42),
):
    """Calibrate, optimise the boundary, and emit the operating-point JSON.

    Calibration and the class weights are *fit* on one set of predictions and
    the scorecard is *reported* on a disjoint set, so the lift over the argmax
    baseline is honest. Pass ``--report-logits`` to fit on the validation split
    (``--logits``) and report on the held-out test split — the deployment-faithful
    design that reproduces the manuscript test-set baseline. With no
    ``--report-logits``, ``--logits`` is split 50/50 instead.
    """
    set_global_seed(seed)
    fit_logits, fit_labels, classes = _load_logits_npz(logits)
    num_classes = len(classes)
    neg_index = classes.index("NEGATIVE") if "NEGATIVE" in classes else num_classes - 1
    pa_index = classes.index("L1PA") if "L1PA" in classes else None

    if report_logits is not None:
        rep_logits, rep_labels, rep_classes = _load_logits_npz(report_logits)
        if rep_classes != classes:
            raise typer.BadParameter(f"class mismatch: fit {classes} vs report {rep_classes}")
        report_source = str(report_logits)
        logger.info(
            f"fit on {fit_labels.shape[0]:,} ({logits.name}); "
            f"report on {rep_labels.shape[0]:,} ({report_logits.name}); classes={classes}"
        )
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(fit_labels.shape[0])
        half = fit_labels.shape[0] // 2
        fit_idx, rep_idx = perm[:half], perm[half:]
        rep_logits, rep_labels = fit_logits[rep_idx], fit_labels[rep_idx]
        fit_logits, fit_labels = fit_logits[fit_idx], fit_labels[fit_idx]
        report_source = f"{logits.name} (50/50 self-split)"
        logger.info(
            f"loaded {fit_labels.shape[0] + rep_labels.shape[0]:,} predictions "
            f"(50/50 fit/report); classes={classes}"
        )

    # --- §2.1 temperature scaling -------------------------------------------
    temperature = fit_temperature(fit_logits, fit_labels)
    rep_probs_raw = softmax(rep_logits, 1.0)
    rep_probs_cal = softmax(rep_logits, temperature)
    ece_before = expected_calibration_error(rep_probs_raw, rep_labels, n_bins)
    ece_after = expected_calibration_error(rep_probs_cal, rep_labels, n_bins)
    logger.info(f"temperature T={temperature:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")

    # --- §2.2 cost-sensitive weights (on calibrated fit probs) ---------------
    fit_probs_cal = softmax(fit_logits, temperature)
    weights, fit_f1 = optimize_class_weights(
        _beta_reweighted(fit_probs_cal, pa_index, beta),
        fit_labels,
        num_classes,
        neg_index,
    )
    logger.info(
        f"class weights={dict(zip(classes, np.round(weights, 4)))} (fit macro-F1 {fit_f1:.4f})"
    )

    # --- scorecards on the held-out report half ------------------------------
    baseline_preds = rep_probs_cal.argmax(axis=1)
    tuned_preds = predict_weighted(rep_probs_cal, weights)
    baseline_card = scorecard(rep_labels, baseline_preds, classes)
    tuned_card = scorecard(rep_labels, tuned_preds, classes)

    neg_tau, neg_tau_f1 = sweep_negative_threshold(fit_probs_cal, fit_labels, neg_index)

    logger.info(
        "\nScorecard on %s (vs Salmon argmax baseline %.3f):\n%s\n%s"
        % (
            report_source,
            BASELINE_MACRO_F1,
            _format_scorecard("argmax (calibrated)", baseline_card, classes),
            _format_scorecard("weighted (tuned)", tuned_card, classes),
        )
    )
    if pa_index is not None:
        logger.info(
            f"L1PA precision {baseline_card['precision_L1PA']:.3f} -> "
            f"{tuned_card['precision_L1PA']:.3f}; L1PA recall "
            f"{baseline_card['recall_L1PA']:.3f} -> {tuned_card['recall_L1PA']:.3f}"
        )

    operating_point = {
        "baseline_macro_f1": BASELINE_MACRO_F1,
        "rule": "weighted_argmax",
        "report_source": report_source,
        "classes": classes,
        "temperature": temperature,
        "class_weights": {c: float(w) for c, w in zip(classes, weights)},
        "negative_threshold": neg_tau,
        "negative_threshold_binary_macro_f1": neg_tau_f1,
        "beta": beta,
        "seed": seed,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "report_argmax": baseline_card,
        "report_weighted": tuned_card,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(operating_point, fh, indent=2)
    logger.success(
        f"macro-F1 {baseline_card['macro_f1']:.4f} (argmax) -> "
        f"{tuned_card['macro_f1']:.4f} (tuned); operating point -> {out}"
    )


def _beta_reweighted(probs: np.ndarray, pa_index: Optional[int], beta: float) -> np.ndarray:
    """Pass-through hook for an Fβ-on-L1PA objective.

    The weight optimiser maximises macro-F1 on the supplied probabilities; an
    ``Fβ>1`` preference for L1PA recall is expressed by nudging the L1PA column
    up by ``beta`` before the search, which biases the chosen weights toward
    recovering L1PA reads lost to NEGATIVE (plan §2.2). ``beta==1`` is a no-op.
    """
    if pa_index is None or beta == 1.0:
        return probs
    scaled = probs.copy()
    scaled[:, pa_index] *= beta
    return scaled / scaled.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    app()
