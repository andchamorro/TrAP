"""Aggregate-quantification scorecard — TrAP's production-relevant metric.

Per-read macro-F1 (Track 1/3/4) is the wrong yardstick for TrAP's actual
deliverable: **quantification** — how much of each class (L1HS / L1PA / NEGATIVE)
is in a sample. Per-read errors partially cancel when reads are aggregated, so a
0.70-macro-F1 classifier can still recover class *abundance* well, especially
with calibrated *expected* (soft) counts.

This measures that directly, two ways:

  * **Single-pool recovery** at deployment priors — estimated vs true class
    fractions (hard argmax counts and soft expected counts) for each method.
  * **Composition recovery** — over many bootstrap pseudo-samples with random
    class compositions, the agreement between estimated and true per-class
    abundance (Pearson r, Lin's concordance CCC, MAE, bias). This is the standard
    quantifier-validation scatter (cf. Salmon/RSEM), reduced to scalar scores.

Soft (expected) counts from *calibrated* probabilities are typically the
least-biased abundance estimator — a strong, honest manuscript point.

Inputs (reuses existing artifacts, no GPU):
  * ``--dual dual_probs.npz --operating-point …json`` → methods
    ``salmon`` / ``spm`` / ``ensemble`` (all temperature-calibrated).
  * ``--logits eval_logits.<model>.npz`` → a single model's calibrated probs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
import numpy as np
import typer

from trap.analysis.boundary_calibration import softmax
from trap.analysis.ensemble import (
    DEFAULT_PRIORS,
    _probs_to_logits,
    resample_indices_to_priors,
    soft_vote,
)
from trap.config.config import RESULTS_DIR
from trap.utils.seeding import set_global_seed

app = typer.Typer(add_completion=False, help="Aggregate-quantification scorecard.")


@app.callback()
def _main() -> None:
    """Keep the ``scorecard`` subcommand name (Typer collapses single-command apps)."""


# --------------------------------------------------------------------------- #
# Pure core (CPU, testable)                                                   #
# --------------------------------------------------------------------------- #
def soft_counts(probs: np.ndarray) -> np.ndarray:
    """Expected per-class counts — the column sum of the probability matrix."""
    return np.asarray(probs).sum(axis=0)


def hard_counts(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class counts from the argmax (hard) prediction."""
    return np.bincount(np.asarray(probs).argmax(axis=1), minlength=num_classes).astype(float)


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement *and* bias)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mt, mp = yt.mean(), yp.mean()
    cov = ((yt - mt) * (yp - mp)).mean()
    denom = yt.var() + yp.var() + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else 1.0


def bootstrap_compositions(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_samples: int,
    rng: np.random.Generator,
    soft: bool = True,
    sample_size: int = 20000,
    concentration: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """True and estimated per-class fractions over random-composition pseudo-samples.

    Each pseudo-sample draws a class composition from ``Dirichlet(concentration)``,
    samples reads (with replacement) to match it, and estimates class fractions
    from those reads' predictions (soft expected counts or hard argmax counts).

    Returns:
        ``(true_fracs, est_fracs)``, each ``(n_samples, num_classes)``.
    """
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    present = [c for c in range(num_classes) if len(idx_by_class[c]) > 0]
    true_fracs = np.zeros((n_samples, num_classes))
    est_fracs = np.zeros((n_samples, num_classes))
    for b in range(n_samples):
        comp = rng.dirichlet(np.full(len(present), concentration))
        counts = rng.multinomial(sample_size, comp)
        picks = np.concatenate(
            [rng.choice(idx_by_class[c], n, replace=True) for c, n in zip(present, counts) if n]
        )
        sub = probs[picks]
        for c, n in zip(present, counts):
            true_fracs[b, c] = n / sample_size
        est = soft_counts(sub) if soft else hard_counts(sub, num_classes)
        est_fracs[b] = est / est.sum()
    return true_fracs, est_fracs


def recovery_metrics(
    true_fracs: np.ndarray, est_fracs: np.ndarray, classes: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-class abundance-agreement metrics across pseudo-samples."""
    out = {}
    for c, name in enumerate(classes):
        yt, yp = true_fracs[:, c], est_fracs[:, c]
        pearson = (
            float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else float("nan")
        )
        out[name] = {
            "pearson": pearson,
            "ccc": concordance_correlation(yt, yp),
            "mae": float(np.abs(yp - yt).mean()),
            "bias": float((yp - yt).mean()),
        }
    return out


def collapse_l1_vs_negative(
    probs: np.ndarray, labels: np.ndarray, classes: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collapse the 3-class problem to the production filter's binary ``L1`` vs ``NEGATIVE``.

    The classifier is deployed only as a NEGATIVE filter (``postprocessing.py``)
    feeding salmon, so its production-relevant view is binary: ``P(L1)=1-P(NEG)``.
    """
    neg = classes.index("NEGATIVE")
    p_neg = np.asarray(probs)[:, neg]
    probs2 = np.column_stack([1.0 - p_neg, p_neg])  # [L1, NEGATIVE]
    labels2 = np.where(labels == neg, 1, 0)
    return probs2, labels2, ["L1", "NEGATIVE"]


def filter_purity_yield(
    p_negative: np.ndarray, is_l1: np.ndarray, thresholds: np.ndarray
) -> List[Dict[str, float]]:
    """Yield/purity of the ``P(NEGATIVE) < τ`` filter that selects reads for salmon.

    For each τ: ``yield`` = fraction of true L1 reads kept (sensitivity), ``purity``
    = fraction of kept reads that are truly L1 (precision of salmon's input),
    ``kept_frac`` = fraction of all reads passed downstream.
    """
    is_l1 = np.asarray(is_l1).astype(bool)
    n_l1 = int(is_l1.sum())
    rows = []
    for tau in thresholds:
        kept = np.asarray(p_negative) < tau
        n_kept = int(kept.sum())
        rows.append(
            {
                "threshold": float(tau),
                "yield": float((kept & is_l1).sum() / n_l1) if n_l1 else float("nan"),
                "purity": float((kept & is_l1).sum() / n_kept) if n_kept else float("nan"),
                "kept_frac": float(kept.mean()),
            }
        )
    return rows


def yield_at_purity(purity_yield_rows: List[Dict[str, float]], target_purity: float) -> float:
    """Best (max) yield achievable at or above ``target_purity`` — the frontier value.

    The right way to compare filters across models: τ is not comparable between
    models, but "how much L1 can I keep while salmon's input is ≥X% pure" is.
    Returns ``nan`` when the model cannot reach ``target_purity`` at any threshold.
    """
    feasible = [r["yield"] for r in purity_yield_rows if r["purity"] >= target_purity]
    return max(feasible) if feasible else float("nan")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _methods_from_dual(
    dual: Path, operating_point: Path
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Build per-model calibrated probabilities + the soft-vote ensemble from dump-dual.

    Handles **any** number of models in the npz (e.g. salmon, spm, *and* a distilled
    student): each is temperature-calibrated (the operating point's temperature when
    known, else T=1), and the ``ensemble`` method is the soft-vote of the two
    operating-point members. So a 3-model dump yields salmon / spm / distilled /
    ensemble, all on the *same* reads — the apples-to-apples filter comparison.
    """
    point = json.loads(Path(operating_point).read_text())
    data = np.load(dual, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]
    names = [str(n) for n in data["names"]]
    labels = data["labels"].astype(np.int64)
    temps, alpha = point["temperatures"], point["alpha"]
    methods = {
        n: softmax(_probs_to_logits(data[f"probs_{n}"].astype(np.float64)), temps.get(n, 1.0))
        for n in names
    }
    ens_a, ens_b = point["models"][0], point["models"][1]
    if ens_a in methods and ens_b in methods:
        methods["ensemble"] = soft_vote(methods[ens_a], methods[ens_b], alpha)
    return methods, labels, classes


def _methods_from_logits(
    logits_npz: Path, temperature: float
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Build one model's (optionally temperature-calibrated) probability matrix."""
    data = np.load(logits_npz, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]
    labels = data["labels"].astype(np.int64)
    logits = data["logits"].astype(np.float64)
    methods = {"model": softmax(logits, temperature)}
    return methods, labels, classes


@app.command()
def scorecard(
    dual: Path = typer.Option(
        None, help="dump-dual npz (with --operating-point) for the ensemble."
    ),
    operating_point: Path = typer.Option(None, help="ensemble_operating_point.json."),
    logits: Path = typer.Option(None, help="Single-model logits npz (alternative to --dual)."),
    temperature: float = typer.Option(1.0, help="Temperature for the --logits single-model path."),
    n_boot: int = typer.Option(500, help="Bootstrap pseudo-samples for composition recovery."),
    sample_size: int = typer.Option(20000, help="Reads per pseudo-sample."),
    seed: int = typer.Option(42),
    out: Path = typer.Option(RESULTS_DIR / "quantification_scorecard.json"),
):
    """Score how well each method recovers class *abundance* (not per-read labels)."""
    set_global_seed(seed)
    if dual is not None:
        methods, labels, classes = _methods_from_dual(dual, operating_point)
    elif logits is not None:
        methods, labels, classes = _methods_from_logits(logits, temperature)
    else:
        raise typer.BadParameter("provide --dual (+ --operating-point) or --logits.")
    num_classes = len(classes)
    logger.info(f"methods={list(methods)} classes={classes} reads={labels.shape[0]:,}")

    rng = np.random.default_rng(seed)
    # Single-pool recovery at deployment priors.
    pool = resample_indices_to_priors(labels, classes, DEFAULT_PRIORS, rng)
    true_frac = np.bincount(labels[pool], minlength=num_classes) / pool.size

    report: Dict[str, dict] = {"classes": classes, "true_fraction_priors": true_frac.tolist()}
    lines = [f"\nSingle-pool class fractions at deployment priors (n={pool.size:,}):"]
    lines.append(f"  {'method':<12} {'estimator':<6} " + " ".join(f"{c:>10}" for c in classes))
    lines.append(f"  {'TRUE':<12} {'':<6} " + " ".join(f"{x:>10.4f}" for x in true_frac))
    for name, probs in methods.items():
        sub = probs[pool]
        soft_f = soft_counts(sub) / sub.shape[0]
        hard_f = hard_counts(sub, num_classes) / sub.shape[0]
        report[name] = {"soft_fraction": soft_f.tolist(), "hard_fraction": hard_f.tolist()}
        lines.append(f"  {name:<12} {'soft':<6} " + " ".join(f"{x:>10.4f}" for x in soft_f))
        lines.append(f"  {name:<12} {'hard':<6} " + " ".join(f"{x:>10.4f}" for x in hard_f))
    logger.info("\n".join(lines))

    # 3-class argmax metrics at deployment priors (per-class P/R/F1 + macro-F1) — the
    # apples-to-apples classification score for the B4 head-to-head.
    from trap.analysis.boundary_calibration import macro_f1

    y_true = labels[pool]
    clines = ["\n3-class metrics at deployment priors (argmax):"]
    for name, probs in methods.items():
        y_pred = probs[pool].argmax(axis=1)
        per = {}
        for c, cname in enumerate(classes):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            per[cname] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        mf1 = macro_f1(y_true, y_pred, num_classes)
        report[name]["classification"] = {"macro_f1": round(mf1, 4), "per_class": per}
        clines.append(
            f"  {name:<12} macroF1={mf1:.4f}  "
            + "  ".join(f"{c}:F1={per[c]['f1']:.3f}" for c in classes)
        )
    logger.info("\n".join(clines))

    # Composition recovery (abundance agreement across random compositions).
    comp_lines = [f"\nComposition recovery over {n_boot} pseudo-samples (CCC | Pearson | bias):"]
    for name, probs in methods.items():
        report.setdefault(name, {})
        for est_name, soft in (("soft", True), ("hard", False)):
            tf, ef = bootstrap_compositions(
                probs, labels, num_classes, n_boot, np.random.default_rng(seed), soft, sample_size
            )
            metrics = recovery_metrics(tf, ef, classes)
            report[name][f"composition_{est_name}"] = metrics
            cells = "  ".join(
                f"{c}:CCC{metrics[c]['ccc']:.3f}/r{metrics[c]['pearson']:.3f}/b{metrics[c]['bias']:+.3f}"
                for c in classes
            )
            comp_lines.append(f"  {name:<10} {est_name:<5} {cells}")
    logger.info("\n".join(comp_lines))

    # --- Production view: the binary L1-vs-NEGATIVE FILTER feeding salmon ------
    # postprocessing.py keeps reads with P(NEGATIVE) < τ; salmon then quantifies
    # the survivors. The classifier's production-relevant metrics are filter
    # yield/purity (salmon's input quality) and how well the kept (soft-L1) count
    # tracks true L1 abundance — NOT 3-class read-counting.
    if "NEGATIVE" in classes:
        neg = classes.index("NEGATIVE")
        taus = np.round(np.arange(0.02, 0.96, 0.02), 2)  # fine sweep for the frontier
        purity_targets = [0.70, 0.80, 0.90]
        report["filter"] = {}
        # Frontier comparison: yield achievable at each purity target (τ-agnostic).
        flines = [
            "\nProduction filter (P(NEGATIVE)<τ → salmon), deployment priors —",
            "  FRONTIER: max yield at purity ≥ target (— = unreachable):",
            f"  {'method':<12} {'maxPurity':>9}  "
            + "  ".join(f"yld@P{int(p * 100)}" for p in purity_targets),
        ]
        for name, probs in methods.items():
            sub = probs[pool]
            is_l1 = labels[pool] != neg
            py = filter_purity_yield(sub[:, neg], is_l1, taus)
            frontier = {
                f"yield_at_p{int(p * 100)}": yield_at_purity(py, p) for p in purity_targets
            }
            probs2, labels2, classes2 = collapse_l1_vs_negative(probs, labels, classes)
            tf, ef = bootstrap_compositions(
                probs2, labels2, 2, n_boot, np.random.default_rng(seed), True, sample_size
            )
            l1_rec = recovery_metrics(tf, ef, classes2)["L1"]
            report["filter"][name] = {
                "purity_yield": py,
                "frontier": frontier,
                "max_purity": max(r["purity"] for r in py),
                "l1_abundance_soft": l1_rec,
            }
            cells = "  ".join(
                f"{frontier[f'yield_at_p{int(p * 100)}']:.3f}".replace("nan", "  — ")
                for p in purity_targets
            )
            flines.append(f"  {name:<12} {report['filter'][name]['max_purity']:>9.3f}  {cells}")
        flines.append(
            "  L1-abundance recovery (soft): "
            + "  ".join(
                f"{name}:CCC{report['filter'][name]['l1_abundance_soft']['ccc']:.3f}"
                for name in methods
            )
        )
        logger.info("\n".join(flines))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.success(f"Wrote quantification scorecard -> {out}")


if __name__ == "__main__":
    app()
