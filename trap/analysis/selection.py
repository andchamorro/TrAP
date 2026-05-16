"""Quantitative k-mer length selection criterion.

Replaces "the entropy curve plateaus by eye" with three explicit, reproducible
gates that together answer the reviewers:

1. **Saturation gate (R1)** — the bias-corrected entropy at a given k must be
   stable as the corpus grows: the relative change between the ~50% and 100%
   subsample is below ``epsilon``.  A k whose entropy is still climbing with N is
   in the under-sampled regime, so any "plateau" there is a sampling artefact.
2. **Marginal-gain gate (R1/R2)** — diminishing returns begin at the first k
   whose per-step information gain ``dH_k = H_k - H_{k-1}`` (bias-corrected)
   falls below ``tau`` bits.
3. **Task gate (R3)** — the smallest k whose downstream macro-F1 is
   statistically indistinguishable from the best (within one CV std).

The recommendation is reported with every gate value and a plain-language
verdict, and is explicit about whether the diminishing-returns point lies inside
the reliably-sampled regime — so the result can corroborate *or* revise the
manuscript's k=16 plateau / k=17 operational choice rather than assuming it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ENTROPY_COL = "h_miller_madow"


def _closest_to_half(fracs: List[float]) -> Optional[float]:
    """Return the subsample fraction nearest 0.5 among fractions ``< 1.0``."""
    below = [f for f in fracs if f < 1.0]
    if not below:
        return None
    return min(below, key=lambda f: abs(f - 0.5))


def _select_one(
    spectrum: pd.DataFrame,
    ablation: Optional[pd.DataFrame],
    epsilon: float,
    tau: float,
    entropy_col: str,
) -> Dict[str, object]:
    """Apply the three gates to a single corpus' spectrum (+ shared ablation)."""
    full = spectrum[spectrum["subsample_frac"] == 1.0]
    full_h = full.groupby("k")[entropy_col].mean().sort_index()
    ks = full_h.index.to_numpy()
    h = full_h.to_numpy()
    d_h = np.diff(h, prepend=h[:1])  # dH[0] := 0 (no predecessor)

    # --- Gate 1: saturation ---
    half_frac = _closest_to_half(sorted(spectrum["subsample_frac"].unique()))
    if half_frac is None:
        rel_change = np.full(ks.shape, np.nan)
        saturation_pass = np.ones(ks.shape, dtype=bool)
    else:
        half_h = (
            spectrum[spectrum["subsample_frac"] == half_frac]
            .groupby("k")[entropy_col]
            .mean()
            .reindex(ks)
            .to_numpy()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_change = np.abs(h - half_h) / np.where(h != 0, h, np.nan)
        saturation_pass = rel_change < epsilon

    saturation_safe_k = int(ks[saturation_pass].max()) if saturation_pass.any() else None

    # --- Gate 2: marginal information gain ---
    diminishing = (d_h < tau) & (ks > int(ks.min()))
    diminishing_returns_k = int(ks[diminishing].min()) if diminishing.any() else None

    # --- Gate 3: downstream task ---
    task_plateau_k = None
    if ablation is not None and not ablation.empty:
        task = ablation.sort_values("k")
        f1 = task["macro_f1_mean"].to_numpy()
        f1_std = task["macro_f1_std"].to_numpy()
        task_ks = task["k"].to_numpy()
        best_idx = int(np.argmax(f1))
        threshold = f1[best_idx] - f1_std[best_idx]
        good = task_ks[f1 >= threshold]
        task_plateau_k = int(good.min()) if good.size else int(task_ks[best_idx])

    # --- Compose recommendation ---
    within_reliable = (
        diminishing_returns_k is not None
        and saturation_safe_k is not None
        and diminishing_returns_k <= saturation_safe_k
    )
    if within_reliable:
        recommended_k = diminishing_returns_k
        verdict = (
            f"Diminishing returns at k={diminishing_returns_k} lies within the "
            f"reliably-sampled regime (entropy stable up to k={saturation_safe_k}); "
            "the plateau is real."
        )
    else:
        candidates = [k for k in (saturation_safe_k, task_plateau_k) if k is not None]
        recommended_k = max(candidates) if candidates else diminishing_returns_k
        verdict = (
            f"Marginal-gain plateau (k={diminishing_returns_k}) falls in the "
            f"under-sampled regime (entropy reliable only to k={saturation_safe_k}); "
            "the apparent entropy plateau is partly a sampling artefact. "
            f"Recommendation defers to the saturation limit and the task gate "
            f"(macro-F1 plateau at k={task_plateau_k})."
        )

    corroborates_manuscript = recommended_k is not None and 15 <= recommended_k <= 18

    per_k = [
        {
            "k": int(k),
            "h_bias_corrected": float(hh),
            "marginal_gain": float(dd),
            "marginal_gain_below_tau": bool(dd < tau and k > ks.min()),
            "saturation_rel_change": (None if np.isnan(rc) else float(rc)),
            "saturation_pass": bool(sp),
        }
        for k, hh, dd, rc, sp in zip(ks, h, d_h, rel_change, saturation_pass)
    ]

    return {
        "saturation_safe_k_max": saturation_safe_k,
        "diminishing_returns_k": diminishing_returns_k,
        "diminishing_returns_within_reliable_regime": bool(within_reliable),
        "task_plateau_k": task_plateau_k,
        "recommended_k": recommended_k,
        "corroborates_manuscript_k16_17": bool(corroborates_manuscript),
        "verdict": verdict,
        "gates": {"saturation_epsilon": epsilon, "marginal_gain_tau": tau},
        "per_k": per_k,
    }


def compute_selection(
    spectrum: pd.DataFrame,
    ablation: Optional[pd.DataFrame] = None,
    epsilon: float = 0.01,
    tau: float = 0.05,
    entropy_col: str = ENTROPY_COL,
) -> Dict[str, object]:
    """Run the k-selection criterion for every corpus in *spectrum*.

    Args:
        spectrum: Tidy spectrum table (one row per corpus/k/frac/replicate).
        ablation: Optional k -> macro-F1 table (shared across corpora).
        epsilon: Saturation-gate threshold (relative entropy change).
        tau: Marginal-gain threshold in bits.
        entropy_col: Which (bias-corrected) entropy column to use.

    Returns:
        Mapping ``corpus -> selection report`` plus an ``overall`` summary that
        lists the per-corpus recommended k.
    """
    report: Dict[str, object] = {}
    for corpus, group in spectrum.groupby("corpus"):
        report[str(corpus)] = _select_one(group, ablation, epsilon, tau, entropy_col)

    report["overall"] = {
        "recommended_k_by_corpus": {
            corpus: rep["recommended_k"]
            for corpus, rep in report.items()
            if corpus != "overall"
        },
        "entropy_col": entropy_col,
    }
    return report
