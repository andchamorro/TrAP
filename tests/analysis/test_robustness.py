"""Tests for the 5'RACE concordance robustness analysis.

The fixture is a miniature of the real benchmark: four methods, two unit
scales (TPM for the Salmon family, RPM for the EM family), a locus set that is
ragged because the EM methods only emit estimates where they detect signal, and
one high-leverage locus.  That is enough to pin the behaviours the revision
depends on — above all that the published pooled-unit cut is *not* scale
neutral, and that the paired bootstrap refuses a locus set it cannot pair on.
"""

import numpy as np
import pandas as pd
import pytest

from trap.analysis.robustness import (
    DOMINANT_LOCUS,
    LIKE_FOR_LIKE_DEFINITIONS,
    LOCUS_SET_DEFINITIONS,
    MAX_RESIDUAL_LOCUS,
    METHODS,
    bootstrap_delta_r2,
    dynamic_range_summary,
    filter_definition_grid,
    l1base_overlap_counts,
    leave_one_out,
    load_locus_table,
    locus_subset,
    paired_wins,
    pearson_r2,
    spearman_rho,
)

pytestmark = pytest.mark.unit


# region -> (race5, {method: value or None})
# AlbertSalmon/Salmon report TPM and estimate every locus; AlbertEM/L1EM report
# RPM on the small subset they detect.  L7 is reference-undetected (race5 = 0)
# but present for all four methods, so it separates F1 from F5.
_LOCI = {
    "chr1:100-200": (
        12.0,
        {"AlbertSalmon": 1000.0, "Salmon": 900.0, "AlbertEM": 5.0, "L1EM": 4.0},
    ),
    "chr1:300-400": (3.0, {"AlbertSalmon": 500.0, "Salmon": 480.0, "AlbertEM": 3.0, "L1EM": 2.6}),
    "chr1:500-600": (0.0, {"AlbertSalmon": 200.0, "Salmon": 260.0, "AlbertEM": 1.0, "L1EM": None}),
    "chr1:700-800": (0.5, {"AlbertSalmon": 50.0, "Salmon": 70.0, "AlbertEM": None, "L1EM": None}),
    "chr1:900-1000": (2.0, {"AlbertSalmon": 300.0, "Salmon": 240.0, "AlbertEM": 2.0, "L1EM": 1.4}),
    "chr1:1100-1200": (0.0, {"AlbertSalmon": 5.0, "Salmon": 8.0, "AlbertEM": None, "L1EM": None}),
    "chr1:1300-1400": (0.0, {"AlbertSalmon": 40.0, "Salmon": 30.0, "AlbertEM": 0.5, "L1EM": 0.4}),
    # Five more fully-estimated loci, so the matched set is large enough for a
    # bootstrap that does not degenerate on repeated draws.
    "chr1:1500-1600": (
        1.5,
        {"AlbertSalmon": 260.0, "Salmon": 210.0, "AlbertEM": 1.8, "L1EM": 1.2},
    ),
    "chr1:1700-1800": (
        0.8,
        {"AlbertSalmon": 120.0, "Salmon": 150.0, "AlbertEM": 0.9, "L1EM": 0.7},
    ),
    "chr1:1900-2000": (
        4.0,
        {"AlbertSalmon": 620.0, "Salmon": 400.0, "AlbertEM": 3.4, "L1EM": 2.2},
    ),
    "chr1:2100-2200": (0.2, {"AlbertSalmon": 80.0, "Salmon": 110.0, "AlbertEM": 0.6, "L1EM": 0.9}),
    "chr1:2300-2400": (
        6.0,
        {"AlbertSalmon": 780.0, "Salmon": 520.0, "AlbertEM": 4.1, "L1EM": 3.0},
    ),
}

_COMMON = {
    "chr1:100-200",
    "chr1:300-400",
    "chr1:900-1000",
    "chr1:1300-1400",
    "chr1:1500-1600",
    "chr1:1700-1800",
    "chr1:1900-2000",
    "chr1:2100-2200",
    "chr1:2300-2400",
}
_COMMON_DETECTED = _COMMON - {"chr1:1300-1400"}


def _tidy_frame() -> pd.DataFrame:
    """Tidy (long) table in the schema written by the SRR aggregator."""
    rows = []
    for region, (race5, values) in _LOCI.items():
        chrom, span = region.split(":")
        start, end = (int(x) for x in span.split("-"))
        for method in METHODS:
            rows.append(
                {
                    "region": region,
                    "chrom": chrom,
                    "chromStart": start,
                    "chromEnd": end,
                    "target": "L1HS_dup",
                    "category": 1,
                    "strand": "+",
                    "race5": race5,
                    "method": method,
                    "value": values[method],
                }
            )
    # Rows that must be filtered out: wrong target family, and a right-family
    # locus with category 0 (not part of the benchmark).
    for target, category in (("L1PA3", 1), ("L1HS_dup", 0)):
        for method in METHODS:
            rows.append(
                {
                    "region": "chr2:10-20",
                    "chrom": "chr2",
                    "chromStart": 10,
                    "chromEnd": 20,
                    "target": target,
                    "category": category,
                    "strand": "+",
                    "race5": 9.0,
                    "method": method,
                    "value": 99.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "l1em_salmon_fpm.csv"
    _tidy_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def wide(table_path):
    return load_locus_table(table_path)


# ── Statistics ────────────────────────────────────────────────────────────────


def test_pearson_r2_perfect_and_degenerate():
    assert pearson_r2([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    # A perfect negative relationship is still R² = 1; direction is reported by
    # Spearman, not by R².
    assert pearson_r2([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(1.0)
    assert np.isnan(pearson_r2([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert np.isnan(pearson_r2([1.0, 2.0], [1.0, 2.0]))


def test_spearman_rho_is_rank_based():
    # Monotone but strongly non-linear: rho = 1 while R² < 1.
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 4.0, 100.0]
    assert spearman_rho(x, y) == pytest.approx(1.0)
    assert pearson_r2(x, y) < 1.0


# ── Input table ───────────────────────────────────────────────────────────────


def test_load_locus_table_shape_and_derived_columns(wide):
    assert len(wide) == len(_LOCI)  # the L1PA3 / category-0 rows are excluded
    assert set(wide.loc[wide["in_common_set"], "region"]) == _COMMON
    for method in METHODS:
        pct = wide[f"{method}_pct"].dropna()
        assert pct.sum() == pytest.approx(100.0)
    assert wide["chromStart"].notna().all()


def test_load_locus_table_rejects_missing_method(tmp_path):
    frame = _tidy_frame()
    frame = frame[frame["method"] != "L1EM"]
    path = tmp_path / "partial.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="L1EM"):
        load_locus_table(path)


# ── Locus-set definitions ─────────────────────────────────────────────────────


def test_published_filter_is_not_unit_neutral(wide):
    """The pooled `value < 10` arm bites on the TPM side and not the RPM side.

    This is the asymmetry the revision discloses: the cut was calibrated on RPM
    output, so applying it to a column that pools both scales removes a large
    fraction of TPM loci and essentially no RPM loci.
    """
    tpm_removed = {
        m: 1 - len(locus_subset(wide, "F2_legacy_published", m)) / int(wide[m].notna().sum())
        for m in ("AlbertSalmon", "Salmon")
    }
    rpm_removed = {
        m: 1 - len(locus_subset(wide, "F2_legacy_published", m)) / int(wide[m].notna().sum())
        for m in ("AlbertEM", "L1EM")
    }
    assert min(tpm_removed.values()) > 0.0
    assert max(rpm_removed.values()) == 0.0


def test_unit_aware_definition_differs_only_on_the_tpm_side(wide):
    for method in ("AlbertEM", "L1EM"):
        assert set(locus_subset(wide, "F6_legacy_unit_aware", method)["region"]) == set(
            locus_subset(wide, "F2_legacy_published", method)["region"]
        )
    # On the TPM side F6 keeps only the reference-active arm, so it is a subset.
    published = set(locus_subset(wide, "F2_legacy_published", "AlbertSalmon")["region"])
    unit_aware = set(locus_subset(wide, "F6_legacy_unit_aware", "AlbertSalmon")["region"])
    assert unit_aware < published


def test_primary_locus_set_is_matched_across_methods(wide):
    sets = {m: set(locus_subset(wide, "F5_common_detected", m)["region"]) for m in METHODS}
    assert set.union(*sets.values()) == _COMMON_DETECTED
    assert all(s == _COMMON_DETECTED for s in sets.values())


def test_f1_keeps_reference_undetected_loci_that_f5_drops(wide):
    f1 = set(locus_subset(wide, "F1_common", "AlbertSalmon")["region"])
    f5 = set(locus_subset(wide, "F5_common_detected", "AlbertSalmon")["region"])
    assert f1 == _COMMON
    assert f5 == _COMMON_DETECTED
    assert f1 - f5 == {"chr1:1300-1400"}


def test_ragged_definitions_are_not_matched_across_methods(wide):
    """F0 and F4 score each method on its own detected subset."""
    for locus_set in ("F0_all_finite", "F4_support_floor"):
        assert locus_set not in LIKE_FOR_LIKE_DEFINITIONS
        sizes = {m: len(locus_subset(wide, locus_set, m)) for m in METHODS}
        assert len(set(sizes.values())) > 1


def test_locus_subset_rejects_unknown_definition(wide):
    with pytest.raises(KeyError, match="unknown locus set"):
        locus_subset(wide, "F9_nope", "AlbertSalmon")


# ── Grid ──────────────────────────────────────────────────────────────────────


def test_filter_definition_grid_is_complete_and_consistent(wide):
    grid = filter_definition_grid(wide)
    assert len(grid) == len(LOCUS_SET_DEFINITIONS) * len(METHODS)
    assert set(grid["filter_def"]) == set(LOCUS_SET_DEFINITIONS)
    for _, row in grid.iterrows():
        finite_n = int(wide[row["method"]].notna().sum())
        assert row["n"] + row["n_removed"] == finite_n
        assert row["frac_removed"] == pytest.approx(row["n_removed"] / finite_n)
    assert set(grid["unit"]) == {"TPM", "RPM"}
    assert grid["note"].notna().all()


def test_paired_wins_reports_losses_and_honours_like_for_like(wide):
    grid = filter_definition_grid(wide)
    everything = paired_wins(grid, like_for_like_only=False)
    matched = paired_wins(grid, like_for_like_only=True)
    assert set(everything["n_definitions"]) == {len(LOCUS_SET_DEFINITIONS)}
    assert set(matched["n_definitions"]) == {len(LIKE_FOR_LIKE_DEFINITIONS)}
    for frame in (everything, matched):
        assert (frame["n_won"] <= frame["n_definitions"]).all()
        for _, row in frame.iterrows():
            lost = [d for d in row["definitions_lost"].split(",") if d]
            assert len(lost) == row["n_definitions"] - row["n_won"]


# ── Leverage ──────────────────────────────────────────────────────────────────


def test_leave_one_out_drops_the_named_loci(wide):
    # Relabel the fixture's leverage locus and circled locus to the real names
    # so the default variants have something to drop.
    renamed = wide.copy()
    renamed["region"] = renamed["region"].replace(
        {"chr1:100-200": DOMINANT_LOCUS, "chr1:300-400": MAX_RESIDUAL_LOCUS}
    )
    out = leave_one_out(renamed)
    sizes = out.set_index(["variant", "method"])["n"]
    for method in METHODS:
        full = sizes[("full", method)]
        assert sizes[("drop_dominant", method)] == full - 1
        assert sizes[("drop_circled", method)] == full - 1
        assert sizes[("drop_both", method)] == full - 2


def test_leave_one_out_accepts_custom_variants(wide):
    out = leave_one_out(wide, variants={"full": (), "drop_one": ("chr1:900-1000",)})
    assert set(out["variant"]) == {"full", "drop_one"}
    full_n = out.loc[out["variant"] == "full", "n"].unique()
    dropped_n = out.loc[out["variant"] == "drop_one", "n"].unique()
    assert dropped_n[0] == full_n[0] - 1


# ── Bootstrap ─────────────────────────────────────────────────────────────────


def test_bootstrap_delta_r2_is_seed_deterministic(wide):
    first = bootstrap_delta_r2(wide, n_boot=200, seed=3469)
    second = bootstrap_delta_r2(wide, n_boot=200, seed=3469)
    pd.testing.assert_frame_equal(first, second)
    other = bootstrap_delta_r2(wide, n_boot=200, seed=11)
    assert not np.allclose(first["ci_lo"], other["ci_lo"])


def test_bootstrap_delta_r2_interval_brackets_the_point_estimate(wide):
    out = bootstrap_delta_r2(wide, n_boot=400, seed=3469)
    assert (out["ci_lo"] <= out["ci_hi"]).all()
    assert out["p_le_zero"].between(0.0, 1.0).all()
    assert (out["n"] == len(_COMMON_DETECTED)).all()
    # The observed ΔR² is a difference of fits on the same matched set, so it
    # must be the negation of the reversed contrast.
    forward = out.set_index("contrast").loc["AlbertEM-AlbertSalmon", "delta_r2"]
    reverse = bootstrap_delta_r2(
        wide, contrasts=[("AlbertSalmon", "AlbertEM")], n_boot=10, seed=3469
    )
    assert reverse["delta_r2"].iloc[0] == pytest.approx(-forward)


def test_bootstrap_delta_r2_refuses_an_unmatched_locus_set(wide):
    with pytest.raises(ValueError, match="not method-invariant"):
        bootstrap_delta_r2(wide, locus_set="F0_all_finite", n_boot=10)


# ── Descriptive tables ────────────────────────────────────────────────────────


def test_l1base_overlap_counts_use_half_open_intervals(tmp_path, wide):
    bed = tmp_path / "l1base.bed"
    bed.write_text(
        "\n".join(
            [
                "chr1\t150\t250\tUID-1",  # overlaps chr1:100-200
                "chr1\t200\t300\tUID-2",  # abuts chr1:100-200, overlaps nothing
                "chr3\t100\t200\tUID-3",  # chromosome absent from the benchmark
            ]
        )
        + "\n"
    )
    counts = l1base_overlap_counts(wide, bed).set_index("stratum")["n"]
    assert counts["all_benchmark_loci"] == len(_LOCI)
    assert counts["l1base_overlap"] == 1
    assert counts["l1base_overlap_race5_gt_1pct"] == 1
    assert counts["l1base_overlap_in_common_set"] == 1
    assert counts["l1base_overlap_estimated_by_L1EM"] == 1


def test_dynamic_range_summary_separates_the_two_unit_scales(wide):
    out = dynamic_range_summary(wide).set_index("method")
    assert out.loc["AlbertSalmon", "unit"] == "TPM"
    assert out.loc["L1EM", "unit"] == "RPM"
    assert out.loc["AlbertSalmon", "n_loci_estimated"] == len(_LOCI)
    assert out.loc["L1EM", "frac_of_benchmark"] < 1.0
    # The filter removes TPM loci and no RPM loci.
    assert out.loc["AlbertSalmon", "frac_removed_by_filter"] > 0.0
    assert out.loc["L1EM", "frac_removed_by_filter"] == 0.0
    assert out.loc["AlbertSalmon", "value_max"] > out.loc["L1EM", "value_max"]
