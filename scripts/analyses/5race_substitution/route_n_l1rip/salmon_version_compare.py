"""Is the latest salmon interchangeable with the pinned one, on THIS reference?

Compares three `quant.sf` files over the same target set:

  A  `v1_10_3`        the pinned build, and the one whose index reproduces the
                      published HEK293 target set exactly
  B  `v2_default`     salmon 2.x as shipped: chain pruning off
                      (`--orphanChainSubThresh`/`--postMergeChainSubThresh` = 0.0)
  C  `v2_cpp_parity`  salmon 2.x with 0.95/0.9, which upstream says reproduces C++
                      mapping counts exactly

Upstream reports per-transcript Pearson ~0.999 between the C++ and Rust
implementations.  That is a statement about ordinary transcriptomes.  This index is
10,666 LINE-1 elements roughly 99 % identical to one another, which is exactly the
regime where "align every candidate" versus "prune sub-threshold chains" should bite
hardest, so the claim is checked here rather than inherited.

What is reported, and why each column is needed:

* **Pearson r on log10(TPM + 1)** and **Spearman rho** over the union of expressed
  loci -- the headline agreement.
* **the same two over `NumReads`**, because TPM divides by effective length and two
  builds could agree on TPM while disagreeing on how many reads they assigned.
* **max and median absolute relative difference**, because a correlation of 0.999 is
  compatible with one locus being wrong by a factor of ten, and in a benchmark whose
  restricted fits turn on single high-leverage loci that is the failure that matters.
* **detection disagreement**: loci expressed in one arm and not the other, counted
  both ways.  The matched locus set is built from loci with finite estimates in every
  method, so a build that detects a different set changes the set the whole benchmark
  regresses on -- independently of how well the shared loci agree.
* **the largest disagreeing loci by name**, so a difference can be chased rather than
  just noted.

Exit status is always 0: this is a measurement, not a gate.  The decision about
whether to adopt 2.x belongs to whoever reads the numbers.

Run (python -I, paths as arguments):

    python -I salmon_version_compare.py A.quant.sf B.quant.sf C.quant.sf out.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

#: Labels for the three arms, in the order they are passed.
ARMS = ("v1_10_3", "v2_default", "v2_cpp_parity")

#: A locus counts as expressed in an arm when it is assigned more than this.
EXPRESSED_READS = 0.0

#: How many of the worst-disagreeing loci to name.
N_WORST = 10


def load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    return df[["Name", "TPM", "NumReads", "EffectiveLength"]].rename(
        columns={
            "TPM": f"tpm_{label}",
            "NumReads": f"reads_{label}",
            "EffectiveLength": f"efflen_{label}",
        }
    )


def agreement(a: np.ndarray, b: np.ndarray, logged: bool) -> dict:
    """Correlations and relative-difference summaries for one pair of arms."""
    if logged:
        x, y = np.log10(a + 1.0), np.log10(b + 1.0)
    else:
        x, y = a, b
    both = (a > EXPRESSED_READS) | (b > EXPRESSED_READS)
    out: dict[str, float | int] = {"n_compared": int(both.sum())}
    if both.sum() < 3:
        return out
    out["pearson_r"] = float(pearsonr(x[both], y[both]).statistic)
    out["spearman_rho"] = float(spearmanr(x[both], y[both]).statistic)
    denom = np.maximum(np.maximum(a[both], b[both]), 1e-9)
    rel = np.abs(a[both] - b[both]) / denom
    out["median_abs_rel_diff"] = float(np.median(rel))
    out["p95_abs_rel_diff"] = float(np.percentile(rel, 95))
    out["max_abs_rel_diff"] = float(rel.max())
    out["n_a_only"] = int(((a > EXPRESSED_READS) & (b <= EXPRESSED_READS)).sum())
    out["n_b_only"] = int(((a <= EXPRESSED_READS) & (b > EXPRESSED_READS)).sum())
    return out


def main(paths: list[Path], out_csv: Path) -> None:
    frames = [load(p, label) for p, label in zip(paths, ARMS)]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Name", how="outer")
    merged = merged.fillna(0.0)
    print(f"targets: {len(merged):,} across the three arms")
    for label in ARMS:
        n = int((merged[f"reads_{label}"] > EXPRESSED_READS).sum())
        print(
            f"  {label:<14} {n:>6,} loci with reads, "
            f"{merged[f'reads_{label}'].sum():>14,.0f} reads assigned"
        )

    rows = []
    pairs = (("v1_10_3", "v2_default"), ("v1_10_3", "v2_cpp_parity"),
             ("v2_default", "v2_cpp_parity"))
    for left, right in pairs:
        for measure, logged in (("tpm", True), ("reads", False)):
            stats = agreement(
                merged[f"{measure}_{left}"].to_numpy(float),
                merged[f"{measure}_{right}"].to_numpy(float),
                logged=logged,
            )
            rows.append({"left": left, "right": right, "measure": measure,
                         "log10_scale": logged, **stats})
    report = pd.DataFrame(rows)
    report.to_csv(out_csv, index=False)

    print("\nagreement (TPM on log10, reads raw):")
    cols = ["left", "right", "measure", "n_compared", "pearson_r", "spearman_rho",
            "median_abs_rel_diff", "max_abs_rel_diff", "n_a_only", "n_b_only"]
    print(report[[c for c in cols if c in report.columns]].to_string(index=False))

    # Effective length is the other way two builds can disagree while both being
    # "right": TPM divides by it, so a changed efflen moves TPM with reads unchanged.
    for label in ARMS[1:]:
        d = (merged[f"efflen_{label}"] - merged["efflen_v1_10_3"]).abs()
        shared = (merged[f"reads_{label}"] > 0) & (merged["reads_v1_10_3"] > 0)
        if shared.sum():
            print(
                f"\neffective length vs v1_10_3, {label}: "
                f"median |diff| = {d[shared].median():.3f} bp, max = {d[shared].max():.1f} bp"
            )

    worst = merged.assign(
        rel=lambda f: (f["reads_v2_default"] - f["reads_v1_10_3"]).abs()
        / np.maximum(np.maximum(f["reads_v2_default"], f["reads_v1_10_3"]), 1e-9)
    )
    worst = worst[(worst["reads_v1_10_3"] > 0) | (worst["reads_v2_default"] > 0)]
    worst = worst.nlargest(N_WORST, "rel")
    print(f"\n{N_WORST} largest read-assignment disagreements, v1_10_3 vs v2_default:")
    print(
        worst[["Name", "reads_v1_10_3", "reads_v2_default", "reads_v2_cpp_parity", "rel"]]
        .to_string(index=False, max_colwidth=60)
    )
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:4]], Path(sys.argv[4]))
