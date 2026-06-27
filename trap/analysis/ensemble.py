"""Track 3 — Salmon⊕SPM soft-vote ensemble for the L1PA↔NEGATIVE boundary.

Track 4 (`boundary_mining`) found that Salmon's and SPM's L1PA-recall errors are
substantially **decorrelated**: ~56% of Salmon's L1PA→NEGATIVE leak (≈20% of all
L1PA reads) is *recovered* by SPM. A single-model decision rule (Track 1) could
not exploit that; an ensemble can (Tran 2025 §3.1.3, soft voting).

This module answers the open question — *does SPM's L1PA-recall recovery beat its
~2× NEGATIVE→L1PA false-positive cost in macro-F1?* — in the same two-stage shape
as Track 1:

  * ``dump-dual`` (GPU) — score reads of KNOWN raw subfamily with ≥2 classifiers
    **in lockstep** (every model sees the identical read, with ids), and save the
    per-read probability matrix of each model + the true label to an ``.npz``.
  * ``optimize`` (CPU) — calibrate each model (temperature), **resample to the
    deployment class priors** (so the precision-sensitive macro-F1 is honest, not
    inflated by the per-subfamily sampling), sweep the soft-vote weight ``α`` to
    maximise macro-F1, and report the ensemble scorecard vs each single model.

The lockstep scoring (not the per-tokenizer ``boundary_calibration`` npz) is
required because the two tokenizers' processed datasets do not share row order
and dropped their read ids — only same-read scoring keeps the two probability
rows aligned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import typer

from trap.analysis.boundary_calibration import (
    expected_calibration_error,
    fit_temperature,
    macro_f1,
    optimize_class_weights,
    predict_weighted,
    scorecard,
    softmax,
)
from trap.config.config import RESULTS_DIR
from trap.utils.seeding import set_global_seed

app = typer.Typer(add_completion=False, help="Salmon⊕SPM soft-vote ensemble (Track 3).")

# Deployment class proportions (plan §0: ~89% NEGATIVE, ~10% L1PA, ~0.3% L1HS).
DEFAULT_PRIORS = {"NEGATIVE": 0.893, "L1PA": 0.104, "L1HS": 0.003}
BASELINE_MACRO_F1 = 0.664


# --------------------------------------------------------------------------- #
# Pure core (CPU, testable)                                                   #
# --------------------------------------------------------------------------- #
def soft_vote(probs_a: np.ndarray, probs_b: np.ndarray, alpha: float) -> np.ndarray:
    """Convex soft-vote ``α·probs_a + (1-α)·probs_b`` (rows stay normalised)."""
    return alpha * np.asarray(probs_a) + (1.0 - alpha) * np.asarray(probs_b)


def ensemble_soft_labels(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    temperature_a: float,
    temperature_b: float,
    alpha: float,
) -> np.ndarray:
    """Calibrated soft-vote teacher targets from two models' raw logits.

    Applies each model's operating-point temperature, then the convex soft-vote —
    i.e. the exact ensemble distribution the distillation student should mimic.

    Returns:
        ``(N, C)`` probability rows (the distillation soft labels).
    """
    return soft_vote(softmax(logits_a, temperature_a), softmax(logits_b, temperature_b), alpha)


def optimize_ensemble_weight(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Sweep the soft-vote weight ``α`` to maximise macro-F1 of ``argmax``.

    ``α=1`` is model A alone, ``α=0`` is model B alone; the optimum trades model
    A's precision against model B's recall recovery.

    Returns:
        ``(alpha, macro_f1)`` at the best grid point.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)
    best_alpha, best_f1 = 1.0, -1.0
    for alpha in grid:
        preds = soft_vote(probs_a, probs_b, alpha).argmax(axis=1)
        f1 = macro_f1(labels, preds, num_classes)
        if f1 > best_f1:
            best_f1, best_alpha = f1, float(alpha)
    return best_alpha, best_f1


def resample_indices_to_priors(
    labels: np.ndarray,
    classes: List[str],
    priors: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices of a subsample whose class proportions match ``priors``.

    Takes the largest subsample of the available reads that hits the target
    fractions (the class limiting ``count_c / prior_c`` sets the total), so a
    per-subfamily-capped sample is rebalanced to the deployment mix before the
    precision-sensitive scorecard is computed.
    """
    total_prior = sum(priors.get(c, 0.0) for c in classes)
    norm = {c: priors.get(c, 0.0) / total_prior for c in classes}
    counts = {c: int((labels == i).sum()) for i, c in enumerate(classes)}
    scale = min(counts[c] / norm[c] for c in classes if norm[c] > 0)
    picked = []
    for i, c in enumerate(classes):
        target = int(round(norm[c] * scale))
        idx = np.where(labels == i)[0]
        if len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
        picked.append(idx)
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Stage A — dump dual-model probabilities (GPU)                               #
# --------------------------------------------------------------------------- #
@app.command("dump-dual")
def dump_dual(
    model: List[str] = typer.Option(
        ..., "--model", help="Repeatable 'name,model_dir,tokenizer_dir,pad[,k]' (≥2)."
    ),
    r1: Path = typer.Option(..., help="L1 R1 FASTQ (raw subfamily in the read ids)."),
    r2: Path = typer.Option(..., help="L1 R2 FASTQ."),
    per_subfamily: int = typer.Option(8000, help="Max reads per L1 subfamily."),
    negative_cap: int = typer.Option(
        200000, help="Max NEGATIVE reads (large, so resampling can hit ~89% NEG)."
    ),
    max_scan: int = typer.Option(20_000_000),
    batch_size: int = typer.Option(256),
    out: Path = typer.Option(RESULTS_DIR / "dual_probs.npz"),
):
    """Score reads with ≥2 classifiers in lockstep; save per-model probs + labels."""
    from trap.analysis._l1_inference import load_l1_classifier
    from trap.analysis.boundary_mining import _read_labeled_pairs, parse_model_spec

    specs = [parse_model_spec(s) for s in model]
    names = [s[0] for s in specs]
    if len(names) < 2 or len(set(names)) != len(names):
        raise typer.BadParameter(f"need ≥2 uniquely-named models (got {names})")

    classifiers, classes = [], None
    for name, model_dir, tok_dir, pad, k in specs:
        clf = load_l1_classifier(model_dir, tok_dir, k, pad)
        if classes is None:
            classes = clf.classes
        elif clf.classes != classes:
            raise typer.BadParameter(f"class mismatch: {name} {clf.classes} != {classes}")
        logger.info(f"loaded {name}: classes={clf.classes} salmon={clf.is_salmon} pad={pad}")
        classifiers.append((name, clf))

    reads = _read_labeled_pairs(str(r1), str(r2), per_subfamily, negative_cap, max_scan)
    keep = [(rid, s1, s2, sf, lab) for (rid, s1, s2, sf, lab) in reads if lab != "OTHER"]
    logger.info(f"scoring {len(keep):,} 3-class read pairs (of {len(reads):,} sampled)")
    b1 = [r[1] for r in keep]
    b2 = [r[2] for r in keep]
    label_idx = {c: i for i, c in enumerate(classes)}
    labels = np.asarray([label_idx[r[4]] for r in keep], dtype=np.int64)

    arrays = {"labels": labels, "classes": np.asarray(classes), "names": np.asarray(names)}
    for name, clf in classifiers:
        chunks = [
            clf.predict_probs(b1[i : i + batch_size], b2[i : i + batch_size])
            for i in range(0, len(keep), batch_size)
        ]
        arrays[f"probs_{name}"] = (
            np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(classes)))
        )
        logger.info(f"scored {name}: {arrays[f'probs_{name}'].shape}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    logger.success(f"Wrote dual probs ({len(keep):,} reads, models={names}) -> {out}")


# --------------------------------------------------------------------------- #
# Stage B — optimise the ensemble weight (CPU)                                #
# --------------------------------------------------------------------------- #
def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Recover logits (up to a constant) from probabilities for temperature fit."""
    return np.log(np.clip(probs, 1e-12, 1.0))


def _card_line(tag: str, card: Dict[str, float], classes: List[str]) -> str:
    cells = "  ".join(
        f"{c}:P{card[f'precision_{c}']:.3f}/R{card[f'recall_{c}']:.3f}" for c in classes
    )
    return f"  [{tag:<22}] macroF1={card['macro_f1']:.4f}   {cells}"


@app.command("optimize")
def optimize(
    dual: Path = typer.Option(..., help="npz from dump-dual."),
    out: Path = typer.Option(RESULTS_DIR / "ensemble_operating_point.json"),
    tune_weights: bool = typer.Option(
        True, help="Also tune per-class weights on the ensemble (after α)."
    ),
    seed: int = typer.Option(42),
):
    """Calibrate, resample to deployment priors, tune α, and score the ensemble."""
    set_global_seed(seed)
    data = np.load(dual, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]
    names = [str(n) for n in data["names"]]
    labels = data["labels"].astype(np.int64)
    num_classes = len(classes)
    a_name, b_name = names[0], names[1]
    probs = {n: data[f"probs_{n}"].astype(np.float64) for n in names}
    logger.info(f"loaded {labels.shape[0]:,} reads; models={names}; classes={classes}")

    rng = np.random.default_rng(seed)
    bal = resample_indices_to_priors(labels, classes, DEFAULT_PRIORS, rng)
    labels_b = labels[bal]
    perm = rng.permutation(labels_b.shape[0])
    half = labels_b.shape[0] // 2
    fit_i, rep_i = perm[:half], perm[half:]
    logger.info(
        f"resampled to deployment priors: {labels_b.shape[0]:,} reads "
        f"({dict((c, int((labels_b == i).sum())) for i, c in enumerate(classes))})"
    )

    # Per-model temperature on the fit split (from recovered logits).
    temps, cal = {}, {}
    for n in names:
        p = probs[n][bal]
        t = fit_temperature(_probs_to_logits(p[fit_i]), labels_b[fit_i])
        temps[n] = t
        cal[n] = softmax(_probs_to_logits(p), t)
        ece = expected_calibration_error(cal[n][rep_i], labels_b[rep_i])
        logger.info(f"  {n}: T={t:.3f} ECE(report)={ece:.4f}")

    pa_fit, pb_fit = cal[a_name][fit_i], cal[b_name][fit_i]
    pa_rep, pb_rep = cal[a_name][rep_i], cal[b_name][rep_i]
    alpha, alpha_fit_f1 = optimize_ensemble_weight(pa_fit, pb_fit, labels_b[fit_i], num_classes)
    logger.info(f"best soft-vote α={alpha:.3f} ({a_name} weight); fit macro-F1={alpha_fit_f1:.4f}")

    ens_rep = soft_vote(pa_rep, pb_rep, alpha)
    cards = {
        a_name: scorecard(labels_b[rep_i], pa_rep.argmax(1), classes),
        b_name: scorecard(labels_b[rep_i], pb_rep.argmax(1), classes),
        "ensemble": scorecard(labels_b[rep_i], ens_rep.argmax(1), classes),
    }

    weights = None
    if tune_weights:
        neg_index = classes.index("NEGATIVE") if "NEGATIVE" in classes else num_classes - 1
        ens_fit = soft_vote(pa_fit, pb_fit, alpha)
        weights, _ = optimize_class_weights(ens_fit, labels_b[fit_i], num_classes, neg_index)
        cards["ensemble+weights"] = scorecard(
            labels_b[rep_i], predict_weighted(ens_rep, weights), classes
        )

    logger.info(
        "\nDeployment-prior scorecard (report half; baseline argmax 0.664):\n"
        + "\n".join(_card_line(tag, cards[tag], classes) for tag in cards)
    )

    operating_point = {
        "baseline_macro_f1": BASELINE_MACRO_F1,
        "models": names,
        "rule": "soft_vote",
        "alpha": alpha,
        "alpha_weight_on": a_name,
        "temperatures": temps,
        "class_weights": (
            {c: float(w) for c, w in zip(classes, weights)} if weights is not None else None
        ),
        "priors": DEFAULT_PRIORS,
        "classes": classes,
        "seed": seed,
        "report": cards,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(operating_point, fh, indent=2)
    best = max(cards, key=lambda k: cards[k]["macro_f1"])
    logger.success(
        f"best={best} macro-F1={cards[best]['macro_f1']:.4f} "
        f"(vs {a_name} {cards[a_name]['macro_f1']:.4f}); -> {out}"
    )


if __name__ == "__main__":
    app()
