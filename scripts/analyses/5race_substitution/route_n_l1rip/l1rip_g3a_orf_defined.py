"""G3a for route N: does the ORF-defined restriction leave a usable locus set?

The counts and the 5'-restricted profile exist for route N, but the L1Base2
restriction — the gate that decides whether a *restricted* regression is even
possible — was not evaluated.  This script closes that gap from artifacts that
already exist, so it costs no credits.

It answers three questions, in the order the plan asks them.

1. **How many loci survive each restriction, and with how much signal?**  Per
   track (FLI, ORF2, FLnI) and per composite (``union``, ``orf_defined``), with
   the > 6 kb body floor and per-track flank convention already registered in
   :data:`trap.analysis.robustness.L1BASE_TRACKS`.  Reported for every measure,
   not only the headline one, because route N's whole-interval column is
   passenger-dominated and would flatter the gate.
2. **Is the surviving set a result or a single locus?**  ``leverage_share_pct``
   accompanies every row, as §5 requires.
3. **Would the paired bootstrap be licensed?**  The pre-registered stop condition
   is ~50 matched loci; below that, report descriptive concordance and decline the
   bootstrap, because a locus-cluster bootstrap on fewer resamples its own noise.

The reference measure is ``five_prime`` — the plan's §5 decision point, since the
whole-interval column sits at 14.0 % young share against 50.3 % for this one.
Counts are pooled over the two ORF1p-RIP libraries; the pooled-vs-per-library
difference is itself reported, because §11 established that the two libraries
differ in composition beyond counting noise.

Run (python -I, paths as arguments — the inputs are generated data):

    python -I scripts/analyses/5race_substitution/route_n_l1rip/l1rip_g3a_orf_defined.py \
        results/l1rip_calibration/l1rip_loci.csv results/l1rip_calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Measures carried through every table, headline first.
MEASURES = ("five_prime", "overlap", "body_covered", "full_length")

#: Libraries pooled for the reference side.
LIBRARIES = ("Library_8", "Library_10")

#: Restrictions, in the plan's reporting order.  `None` is the unrestricted set.
RESTRICTIONS = ("none", "FLI", "ORF2", "orf_defined", "FLnI", "union")

#: Which restrictions are annotation-independent in the sense R2-1 means.
ORTHOGONAL = {"FLI": True, "ORF2": True, "orf_defined": True, "FLnI": False, "union": False}

#: Reference-side activity floor, as a percentage of the library total.
RACE5_MIN_PCT = 1.0

#: Depth floor from the G3b amendment: a locus-level claim needs real reads.
MIN_LOCUS_READS = 10

#: Below this many matched loci, §11 says decline the paired bootstrap.
BOOTSTRAP_MIN_LOCI = 50

#: HEK293 5'RACE reference, for the side-by-side the plan asks for
#: ("How many overlapping loci clear 1 %").  Source: plan §5.
HEK293_ABOVE_1PCT = {
    "FLI": (1, 140),
    "ORF2": (0, 74),
    "orf_defined": (1, 214),
    "FLnI": (8, 1117),
    "union": (9, 1259),
}


def mask_for(loci: pd.DataFrame, restriction: str) -> np.ndarray:
    """Boolean row mask selecting *restriction*."""
    if restriction == "none":
        return np.ones(len(loci), dtype=bool)
    return loci[f"l1base_{restriction}"].to_numpy(dtype=bool)


def leverage_share(values: np.ndarray) -> float:
    """Percentage of a set's signal carried by its single largest locus."""
    total = values.sum()
    return float("nan") if total <= 0 else float(100.0 * values.max() / total)


def gate_rows(loci: pd.DataFrame) -> list[dict]:
    """One row per (measure, restriction): survival, activity and leverage."""
    rows = []
    for measure in MEASURES:
        pooled = sum(loci[f"{lib}_{measure}"].to_numpy(dtype=float) for lib in LIBRARIES)
        library_total = pooled.sum()
        pct = 100.0 * pooled / library_total if library_total > 0 else pooled * 0.0
        for restriction in RESTRICTIONS:
            mask = mask_for(loci, restriction)
            inside = pooled[mask]
            inside_pct = pct[mask]
            detected = inside > 0
            rows.append(
                {
                    "measure": measure,
                    "restriction": restriction,
                    "orthogonal": ORTHOGONAL.get(restriction, False),
                    "n_loci_in_restriction": int(mask.sum()),
                    "library_reads": int(library_total),
                    "reads_in_restriction": int(inside.sum()),
                    "pct_of_library_in_restriction": round(
                        100.0 * inside.sum() / library_total, 3
                    )
                    if library_total > 0
                    else float("nan"),
                    "n_detected": int(detected.sum()),
                    "n_ge10_reads": int((inside >= MIN_LOCUS_READS).sum()),
                    "n_above_1pct": int((inside_pct > RACE5_MIN_PCT).sum()),
                    "n_above_1pct_and_ge10": int(
                        ((inside_pct > RACE5_MIN_PCT) & (inside >= MIN_LOCUS_READS)).sum()
                    ),
                    "leverage_share_pct": round(leverage_share(inside), 3),
                    "top3_share_pct": round(
                        100.0 * np.sort(inside)[-3:].sum() / inside.sum(), 3
                    )
                    if inside.sum() > 0
                    else float("nan"),
                }
            )
    return rows


def dynamic_range(loci: pd.DataFrame, measure: str) -> list[dict]:
    """Per-restriction dynamic range over the detected loci of *measure*."""
    pooled = sum(loci[f"{lib}_{measure}"].to_numpy(dtype=float) for lib in LIBRARIES)
    total = pooled.sum()
    rows = []
    for restriction in RESTRICTIONS:
        inside = pooled[mask_for(loci, restriction)]
        detected = inside[inside > 0]
        if not len(detected):
            continue
        pct = 100.0 * detected / total
        rows.append(
            {
                "measure": measure,
                "restriction": restriction,
                "n_detected": len(detected),
                "reads_min": int(detected.min()),
                "reads_q1": float(np.percentile(detected, 25)),
                "reads_median": float(np.median(detected)),
                "reads_q3": float(np.percentile(detected, 75)),
                "reads_p90": float(np.percentile(detected, 90)),
                "reads_p99": float(np.percentile(detected, 99)),
                "reads_max": int(detected.max()),
                "pct_max": round(float(pct.max()), 3),
                "top1_share_pct": round(leverage_share(detected), 3),
                "top3_share_pct": round(
                    100.0 * np.sort(detected)[-3:].sum() / detected.sum(), 3
                ),
                "log10_range": round(
                    float(np.log10(detected.max()) - np.log10(detected.min())), 3
                ),
            }
        )
    return rows


def matched_set_sizes(loci: pd.DataFrame, measure: str = "five_prime") -> list[dict]:
    """Candidate matched-locus sets, against §11's bootstrap floor of 50.

    The query side has not run, so the common set is not yet known; these are
    the *upper bounds* a four-method intersection would be taken from.  Both
    pooled and per-library columns are given, since §11 showed the libraries
    are not interchangeable.
    """
    rows = []
    full_length = loci["element_bp"].to_numpy(dtype=float) >= 6000
    young = loci["target"].isin(("L1HS", "L1PA2")).to_numpy(dtype=bool)
    arms = {lib: loci[f"{lib}_{measure}"].to_numpy(dtype=float) for lib in LIBRARIES}
    arms["pooled"] = sum(arms.values())
    for restriction in RESTRICTIONS:
        base = mask_for(loci, restriction)
        for extra_name, extra in (
            ("detected", np.ones(len(loci), dtype=bool)),
            ("full_length_body", full_length),
            ("full_length_young", full_length & young),
        ):
            for arm, values in arms.items():
                mask = base & extra & (values > 0)
                n_detected = int(mask.sum())
                n_ge10 = int((base & extra & (values >= MIN_LOCUS_READS)).sum())
                rows.append(
                    {
                        "measure": measure,
                        "restriction": restriction,
                        "stratum": extra_name,
                        "arm": arm,
                        "n_detected": n_detected,
                        "n_ge10_reads": n_ge10,
                        "bootstrap_licensed_at_floor10": bool(n_ge10 >= BOOTSTRAP_MIN_LOCI),
                    }
                )
    return rows


def main(loci_csv: Path, out_dir: Path) -> None:
    loci = pd.read_csv(loci_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = pd.DataFrame(gate_rows(loci))
    gates.to_csv(out_dir / "l1rip_g3a_restrictions.csv", index=False)

    ranges = pd.DataFrame(
        [row for measure in ("five_prime", "overlap") for row in dynamic_range(loci, measure)]
    )
    ranges.to_csv(out_dir / "l1rip_g3a_dynamic_range.csv", index=False)

    matched = pd.DataFrame(matched_set_sizes(loci))
    matched.to_csv(out_dir / "l1rip_g3a_matched_sets.csv", index=False)

    headline = gates[gates["measure"] == "five_prime"]
    print("=== G3a, route N reference side (pooled ORF1p RIP, five_prime measure) ===")
    print(
        headline[
            [
                "restriction",
                "orthogonal",
                "n_loci_in_restriction",
                "n_detected",
                "n_ge10_reads",
                "n_above_1pct",
                "reads_in_restriction",
                "pct_of_library_in_restriction",
                "leverage_share_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== 'loci clearing 1 %' against the HEK293 5'RACE reference ===")
    for restriction, (hek_active, hek_loci) in HEK293_ABOVE_1PCT.items():
        row = headline[headline["restriction"] == restriction].iloc[0]
        print(
            f"  {restriction:<12} route N {int(row['n_above_1pct']):>3} of "
            f"{int(row['n_loci_in_restriction']):>5}   |   HEK293 5'RACE "
            f"{hek_active:>3} of {hek_loci:>5}"
        )

    print("\n=== Dynamic range, five_prime, detected loci ===")
    print(
        ranges[ranges["measure"] == "five_prime"][
            [
                "restriction",
                "n_detected",
                "reads_median",
                "reads_p90",
                "reads_p99",
                "reads_max",
                "top1_share_pct",
                "top3_share_pct",
                "log10_range",
            ]
        ].to_string(index=False)
    )

    print(f"\n=== Matched-set upper bounds vs the §11 floor of {BOOTSTRAP_MIN_LOCI} ===")
    pooled = matched[(matched["arm"] == "pooled") & (matched["measure"] == "five_prime")]
    print(
        pooled[
            ["restriction", "stratum", "n_detected", "n_ge10_reads", "bootstrap_licensed_at_floor10"]
        ].to_string(index=False)
    )

    orf = pooled[(pooled["restriction"] == "orf_defined") & (pooled["stratum"] == "detected")]
    n_ge10 = int(orf["n_ge10_reads"].iloc[0])
    print(
        f"\nG3a verdict on orf_defined: {int(orf['n_detected'].iloc[0])} loci detected, "
        f"{n_ge10} with >= {MIN_LOCUS_READS} reads -> "
        f"{'bootstrap licensed' if n_ge10 >= BOOTSTRAP_MIN_LOCI else 'DESCRIPTIVE ONLY (below the floor)'}"
    )
    print(f"\nwrote 3 CSVs to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
