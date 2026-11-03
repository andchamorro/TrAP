#!/usr/bin/env python3
"""Route N blocker 1: is rho = 0.41 a property of the protocol, or of the depth?

THE PROBLEM WITH THE NUMBER AS FIRST REPORTED
``l1rip_calibration_report.py`` computes replicate agreement over the whole SAF --
all 122,626 intervals -- of which only 1,486 carry a read in both libraries.  A
Spearman correlation on a vector that is ~98 % ties at zero is dominated by the tie
structure: it measures "do the same loci get ANY read", not "do the two libraries
agree on abundance".  That is not the quantity that bounds a locus-level claim, so
rho = 0.41 cannot be read as "this reference is 41 % reproducible".

WHAT THIS SCRIPT ADDS, AND WHY EACH PIECE IS NEEDED
1. FLOOR-RESTRICTED AGREEMENT.  Agreement recomputed on locus sets defined by the
   POOLED count (a + b >= floor), so the set is chosen independently of how the two
   libraries split -- no selection bias toward agreement.
2. A MULTINOMIAL CEILING.  The highest rho two PERFECT replicates could reach at
   these depths, from counting noise alone: take the pooled profile as truth, draw
   two independent multinomial samples at the observed depths, recompute rho on the
   same fixed locus set.  Observed vs ceiling is the whole question.  (Mildly
   optimistic: the pooled profile is itself noisy, which spreads the truth a little
   and makes ranking easier.  Stated rather than hidden.)
3. AN EMPIRICAL YARDSTICK FROM A DIFFERENT PROTOCOL.  PRJNA1176011 contains two
   pairs of same-BioSample Iso-Seq runs -- SRR31072049/50 (SAMN44378914) and
   SRR31072047/48 (SAMN44378913).  Those are technical replicates with zero
   biological difference, on the same annotation, so they answer "what rho do
   long-read technical replicates achieve here at all?"  Thinned to route N's depth
   they answer it at the SAME depth, which is the comparison that matters.
4. RELIABILITY OF THE PUBLISHED QUANTITY.  Whatever bounds the benchmark is not
   rho over 122k mostly-empty rows; it is the precision of the numbers that would
   be reported.  Young-family share is given three ways: read-level binomial (wrong
   -- reads cluster by locus), locus-cluster bootstrap (right), and the observed
   between-library spread (which contains real prep differences as well as noise).

Nothing here re-defines a pre-registered criterion.  The verdict stays where it is;
this measures how much of blocker 1 is real.

    python scripts/analyses/5race_substitution/route_n_l1rip/l1rip_replicate_power.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.stats import spearmanr

app = typer.Typer(add_completion=False, help=__doc__)

_FAMILY = r"([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)"
L1_TARGETS = ("L1HS", "L1PA2")
FULL_LENGTH_BP = 6_000
RUNS = ("Library_8", "Library_10")
#: BAM column order in counts.l1em.txt -> library label.
L1RIP_BAM_ORDER = ("Library_8", "Library_10")
#: Same-BioSample Iso-Seq technical pairs, plus one cross-BioSample pair for contrast.
ISOSEQ_PAIRS = (
    ("SRR31072049", "SRR31072050", "same BioSample SAMN44378914 (technical)"),
    ("SRR31072047", "SRR31072048", "same BioSample SAMN44378913 (technical)"),
    ("SRR31072049", "SRR31072047", "different BioSample (biological + technical)"),
)
FLOORS = (0, 2, 5, 10, 20, 50)
N_SIM = 400
SEED = 3469


def _featurecounts(path: Path, labels: tuple[str, ...]) -> pd.DataFrame:
    """featureCounts table -> Geneid + one integer column per BAM, in BAM order."""
    frame = pd.read_csv(path, sep="\t", comment="#")
    bam_cols = [c for c in frame.columns if c.endswith(".bam")]
    if len(bam_cols) != len(labels):
        raise ValueError(f"{path}: {len(bam_cols)} BAM columns, {len(labels)} labels")
    out = frame[["Geneid"]].copy()
    for label, col in zip(labels, bam_cols):
        out[label] = frame[col].round().astype(np.int64)
    return out


def concordance(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement between two count vectors, on whatever set was handed in."""
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"n_loci": int(a.size), "spearman_rho": np.nan, "pearson_r2_log10": np.nan,
                "ccc_log10": np.nan, "detected_both": int(((a > 0) & (b > 0)).sum())}
    la, lb = np.log10(1.0 + a), np.log10(1.0 + b)
    # Lin's concordance: penalises scale and location shift, unlike Pearson.
    ccc = (2 * np.cov(la, lb)[0, 1]) / (la.var() + lb.var() + (la.mean() - lb.mean()) ** 2)
    return {
        "n_loci": int(a.size),
        "detected_both": int(((a > 0) & (b > 0)).sum()),
        "spearman_rho": float(spearmanr(a, b).statistic),
        "pearson_r2_log10": float(np.corrcoef(la, lb)[0, 1] ** 2),
        "ccc_log10": float(ccc),
    }


def null_ceiling(a: np.ndarray, b: np.ndarray, mask: np.ndarray, rng, n_sim=N_SIM) -> dict:
    """rho two perfect replicates would reach at these depths, counting noise only."""
    pooled = (a + b).astype(float)
    if pooled.sum() <= 0:
        return {"ceiling_rho_mean": np.nan, "ceiling_rho_p05": np.nan, "ceiling_rho_p95": np.nan}
    p = pooled / pooled.sum()
    na, nb = int(a.sum()), int(b.sum())
    rhos = []
    for _ in range(n_sim):
        sa = rng.multinomial(na, p)[mask].astype(float)
        sb = rng.multinomial(nb, p)[mask].astype(float)
        if np.ptp(sa) and np.ptp(sb):
            rhos.append(spearmanr(sa, sb).statistic)
    rhos = np.asarray(rhos, dtype=float)
    return {
        "ceiling_rho_mean": float(np.nanmean(rhos)),
        "ceiling_rho_p05": float(np.nanpercentile(rhos, 5)),
        "ceiling_rho_p95": float(np.nanpercentile(rhos, 95)),
        "n_sim": int(np.isfinite(rhos).sum()),
    }


def sweep(a: np.ndarray, b: np.ndarray, label: str, rng, subset: np.ndarray | None = None) -> list[dict]:
    """Observed agreement and its ceiling, over pooled-count floors."""
    base = np.ones(a.size, dtype=bool) if subset is None else subset
    rows = []
    pooled = a + b
    for floor in FLOORS:
        mask = base & (pooled >= max(floor, 1)) if floor else base
        if mask.sum() < 3:
            continue
        row = {"comparison": label, "pooled_floor": floor,
               "depth_left": int(a.sum()), "depth_right": int(b.sum())}
        row.update(concordance(a[mask].astype(float), b[mask].astype(float)))
        row.update(null_ceiling(a, b, mask, rng))
        row["rho_as_pct_of_ceiling"] = (
            100 * row["spearman_rho"] / row["ceiling_rho_mean"]
            if np.isfinite(row.get("ceiling_rho_mean", np.nan)) and row["ceiling_rho_mean"] > 0
            else np.nan
        )
        rows.append(row)
    return rows


def thin(counts: np.ndarray, target: int, rng) -> np.ndarray:
    """Subsample a count vector down to `target` reads (multinomial thinning)."""
    total = counts.sum()
    if total <= target:
        return counts.astype(np.int64)
    return rng.multinomial(target, counts / total)


def share_reliability(counts: np.ndarray, flag: np.ndarray, rng, n_boot=2_000) -> dict:
    """A share of total signal, with read-level and locus-cluster uncertainty.

    Reads are not independent -- they arrive in locus-sized clumps -- so the
    binomial interval is the wrong one and is reported only to show how wrong.
    """
    n = counts.sum()
    if n <= 0:
        return {}
    point = float(100 * counts[flag].sum() / n)
    binom_se = float(100 * np.sqrt((point / 100) * (1 - point / 100) / n))
    idx = np.arange(counts.size)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(idx, size=idx.size, replace=True)
        c, f = counts[draw], flag[draw]
        if c.sum() > 0:
            boots.append(100 * c[f].sum() / c.sum())
    boots = np.asarray(boots)
    return {
        "point_pct": point,
        "binomial_se_pct": binom_se,
        "cluster_boot_p02.5_pct": float(np.percentile(boots, 2.5)),
        "cluster_boot_p97.5_pct": float(np.percentile(boots, 97.5)),
        "cluster_boot_se_pct": float(boots.std(ddof=1)),
        "reads": int(n),
    }


@app.command()
def main(
    fulllength_counts: Path = typer.Option(Path("results/l1rip_calibration/l1rip_fulllength_counts.csv")),
    l1rip_counts: Path = typer.Option(Path("results/l1rip_calibration/counts.l1em.txt")),
    isoseq_counts: Path = typer.Option(Path("results/isoseq_calibration/counts.l1em.isoseq_hek293.txt")),
    out_dir: Path = typer.Option(Path("results/l1rip_calibration")),
):
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = pd.read_csv(fulllength_counts)
    overlap = _featurecounts(l1rip_counts, L1RIP_BAM_ORDER)
    joined = fl.merge(overlap.rename(columns={r: f"{r}_overlap" for r in L1RIP_BAM_ORDER}),
                      on="Geneid", how="left")
    for run in RUNS:
        joined[f"{run}_overlap"] = joined[f"{run}_overlap"].fillna(0).astype(np.int64)

    young = joined["target"].isin(L1_TARGETS).to_numpy()
    full_len = (joined["element_bp"] >= FULL_LENGTH_BP).to_numpy()
    logger.info(f"{joined.shape[0]:,} intervals; {young.sum():,} young; "
                f"{(young & full_len).sum():,} full-length young")

    # ---- 1 + 2: floor-restricted agreement and its counting-noise ceiling ------
    rows: list[dict] = []
    for measure in ("overlap", "five_prime", "body_covered", "full_length"):
        a = joined[f"Library_8_{measure}"].to_numpy(dtype=np.int64)
        b = joined[f"Library_10_{measure}"].to_numpy(dtype=np.int64)
        rows += sweep(a, b, f"route N replicates ({measure})", rng)
    # the set a benchmark would actually regress on
    a = joined["Library_8_five_prime"].to_numpy(dtype=np.int64)
    b = joined["Library_10_five_prime"].to_numpy(dtype=np.int64)
    rows += sweep(a, b, "route N replicates (five_prime, full-length young only)", rng,
                  subset=young & full_len)

    # ---- 3: Iso-Seq technical replicates, raw and thinned to route N's depth ---
    iso = _featurecounts(isoseq_counts, ("SRR31072049", "SRR31072050", "SRR31072047", "SRR31072048"))
    target_depth = int((a.sum() + b.sum()) / 2)
    for left, right, note in ISOSEQ_PAIRS:
        x = iso[left].to_numpy(dtype=np.int64)
        y = iso[right].to_numpy(dtype=np.int64)
        rows += sweep(x, y, f"Iso-Seq {left} vs {right} -- {note}", rng)
        rows += sweep(thin(x, target_depth, rng), thin(y, target_depth, rng),
                      f"Iso-Seq {left} vs {right} -- thinned to {target_depth:,} reads", rng)
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "l1rip_replicate_power.csv", index=False)

    # ---- 4: how precise are the numbers that would be published? --------------
    rel = []
    pooled_fp = (a + b).astype(np.int64)
    for name, flag in (("young share of 5'-restricted signal", young),
                       ("full-length-element share of 5'-restricted signal", full_len),
                       ("young AND full-length share", young & full_len)):
        rel.append({"quantity": name, "arm": "pooled (both libraries)",
                    **share_reliability(pooled_fp, flag, rng)})
        for run, col in zip(RUNS, (a, b)):
            rel.append({"quantity": name, "arm": run, **share_reliability(col, flag, rng)})
    reliability = pd.DataFrame(rel)
    reliability.to_csv(out_dir / "l1rip_share_reliability.csv", index=False)

    # ---- how many loci actually carry the signal ------------------------------
    order = np.sort(pooled_fp)[::-1]
    csum = np.cumsum(order) / order.sum()
    conc_rows = [{"fraction_of_signal": f,
                  "loci_needed": int(np.searchsorted(csum, f) + 1)}
                 for f in (0.5, 0.8, 0.9, 0.95, 0.99)]
    conc_rows.append({"fraction_of_signal": "loci with >=10 pooled reads",
                      "loci_needed": int((pooled_fp >= 10).sum())})
    pd.DataFrame(conc_rows).to_csv(out_dir / "l1rip_signal_concentration.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        cols = ["comparison", "pooled_floor", "n_loci", "spearman_rho",
                "ceiling_rho_mean", "rho_as_pct_of_ceiling", "ccc_log10"]
        print("\n== agreement vs counting-noise ceiling ==")
        print(agreement[cols].to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
        print("\n== reliability of the published shares ==")
        print(reliability.to_string(index=False, float_format=lambda v: f"{v:0.2f}"))
        print("\n== signal concentration (pooled five_prime) ==")
        print(pd.DataFrame(conc_rows).to_string(index=False))
    logger.success(f"wrote 3 CSVs to {out_dir}")


if __name__ == "__main__":
    app()
