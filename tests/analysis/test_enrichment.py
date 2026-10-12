"""Tests for domain and ORF0 enrichment of top-attribution k-mers.

The fixture is a synthetic attribution table with a planted signal: k-mers in
ORF2_C are given high attribution and k-mers in the ORF0 footprint low
attribution, so the direction of every statistic is known in advance.  The
reproduction guard (:func:`verify_against_published`) is tested both ways,
because its whole purpose is to fail loudly.
"""

import numpy as np
import pandas as pd
import pytest

from trap.analysis._l1_align import L1_3_SUBREGIONS, l1_region, l1_subregions
from trap.analysis.enrichment import (
    DOMAIN_ORDER,
    domain_enrichment,
    load_attribution,
    subregion_enrichment,
    verify_against_published,
)

pytestmark = pytest.mark.unit

_ORF0_NAME, _ORF0_START, _ORF0_END, _ORF0_STRAND = L1_3_SUBREGIONS[0]


def _attribution_frame(seed: int = 3469) -> pd.DataFrame:
    """Attributions with high signal in ORF2_C and low signal inside ORF0.

    Every one of the eight domains is populated, and the 5'UTR outside ORF0
    carries a few high-attribution k-mers, so the parent-region universe has a
    non-empty top set — as it does in the real data (365 of 5,337).  A universe
    with no top k-mers at all makes the enrichment ratio undefined, which the
    module returns as nan and which is tested separately.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # 5'UTR inside the ORF0 footprint: depleted.
    for position in np.linspace(_ORF0_START, _ORF0_END, 200):
        rows.append((float(position), rng.normal(-0.01, 0.001)))
    # 5'UTR outside ORF0: background, with a handful of high values.
    for i, position in enumerate(np.linspace(_ORF0_END + 50, 900, 200)):
        high = i % 20 == 0
        rows.append((float(position), rng.normal(0.08 if high else 0.0, 0.001)))
    # ORF2_C: strongly enriched.
    for position in np.linspace(4400, 5800, 400):
        rows.append((float(position), rng.normal(0.05, 0.001)))
    # Every remaining domain at background level, so no row is empty.
    for lo, hi in (
        (1000, 1900),
        (1930, 1985),
        (2000, 2700),
        (2800, 3300),
        (3500, 4300),
        (5850, 6050),
    ):
        for position in np.linspace(lo, hi, 60):
            rows.append((float(position), rng.normal(0.0, 0.001)))

    frame = pd.DataFrame(rows, columns=["l1_pos", "attribution"])
    frame["domain"] = [l1_region(p) for p in frame["l1_pos"]]
    frame["label"] = "L1HS"
    # Rows that must be filtered out by load_attribution.
    extra = pd.DataFrame(
        [
            {"l1_pos": 500.0, "attribution": 9.0, "domain": "5UTR", "label": "NEGATIVE"},
            {"l1_pos": np.nan, "attribution": 9.0, "domain": "unaligned", "label": "L1HS"},
            {"l1_pos": 7000.0, "attribution": 9.0, "domain": "outside", "label": "L1HS"},
        ]
    )
    return pd.concat([frame, extra], ignore_index=True)


@pytest.fixture
def attribution_path(tmp_path):
    path = tmp_path / "per_kmer_attribution.csv"
    _attribution_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def attribution(attribution_path):
    return load_attribution(attribution_path)


# ── Region map ────────────────────────────────────────────────────────────────


def test_orf0_is_nested_inside_the_5utr():
    assert l1_region(_ORF0_START) == "5UTR"
    assert l1_region(_ORF0_END) == "5UTR"
    assert l1_subregions((_ORF0_START + _ORF0_END) / 2) == (_ORF0_NAME,)
    assert l1_subregions(_ORF0_START - 1) == ()
    assert l1_subregions(_ORF0_END + 1) == ()
    # ORF0 is an antisense feature, which is what makes it unresolvable under
    # canonical k-mers.
    assert _ORF0_STRAND == "antisense"


def test_orf0_length_is_a_whole_number_of_codons_plus_stop():
    span = _ORF0_END - _ORF0_START + 1
    assert span % 3 == 0
    assert span // 3 - 1 == 71  # 71 codons + stop, as published for ORF0


# ── Loading ───────────────────────────────────────────────────────────────────


def test_load_attribution_applies_the_published_filter(attribution):
    assert set(attribution["label"]) == {"L1HS"}
    assert "unaligned" not in set(attribution["domain"])
    assert "outside" not in set(attribution["domain"])
    assert attribution["l1_pos"].notna().all()
    # The top set is the top decile of the retained subset.
    assert int(attribution["is_top"].sum()) == pytest.approx(len(attribution) * 0.1, rel=0.05)


# ── Domain table ──────────────────────────────────────────────────────────────


def test_domain_enrichment_finds_the_planted_signal(attribution):
    table = domain_enrichment(attribution).set_index("domain")
    assert list(table.index) == list(DOMAIN_ORDER)
    assert table.loc["ORF2_C", "enrichment"] > 1
    assert table.loc["ORF2_C", "p_enriched"] < 1e-4
    assert table.loc["ORF2_C", "sig"] == "***"
    assert table.loc["5UTR", "enrichment"] < 1
    assert table.loc["5UTR", "sig"] == "ns"
    # Every domain is populated in the fixture, so every row is finite.
    assert (table["n"] > 0).all()
    assert table["enrichment"].notna().all()


def test_domain_enrichment_columns_are_internally_consistent(attribution):
    table = domain_enrichment(attribution)
    total = len(attribution)
    total_top = int(attribution["is_top"].sum())
    for _, row in table.iterrows():
        assert row["bg_frac"] == pytest.approx(row["n"] / total)
        assert row["top_frac"] == pytest.approx(row["n_top"] / total_top)
        if row["n"]:
            assert row["enrichment"] == pytest.approx(row["top_frac"] / row["bg_frac"])
            assert row["expected_top"] == pytest.approx(row["n"] * total_top / total)
        assert min(row["p_enriched"], row["p_depleted"]) <= row["p_two_sided"]
        assert row["or_ci_lo"] <= row["or_ci_hi"]


# ── Subregion table ───────────────────────────────────────────────────────────


def test_subregion_enrichment_tests_both_universes(attribution):
    table = subregion_enrichment(attribution)
    assert set(table["universe"]) == {"consensus", "parent_region"}
    assert set(table["domain"]) == {_ORF0_NAME}
    for _, row in table.iterrows():
        assert row["parent_region"] == "5UTR"
        assert row["start"] == _ORF0_START and row["end"] == _ORF0_END
        # The planted depletion must be detected against the whole consensus.
        assert row["enrichment"] < 1


def test_subregion_enrichment_flags_antisense_as_unresolvable(attribution):
    table = subregion_enrichment(attribution)
    assert not table["strand_resolved"].any()
    sense = subregion_enrichment(attribution, subregions=(("FAKE", 4400, 4600, "sense"),))
    assert sense["strand_resolved"].all()


def test_subregion_enrichment_parent_universe_is_smaller(attribution):
    table = subregion_enrichment(attribution).set_index("universe")
    # Same numerator, different denominator: the parent-region test is the
    # targeted one and must use the 5'UTR rather than the whole consensus.
    assert table.loc["consensus", "n"] == table.loc["parent_region", "n"]
    assert table.loc["consensus", "bg_frac"] < table.loc["parent_region", "bg_frac"]


def test_subregion_enrichment_handles_a_universe_with_no_top_kmers(attribution):
    """A parent region whose top set is empty gives nan, not a false zero."""
    flat = attribution.copy()
    flat["is_top"] = False
    table = subregion_enrichment(flat)
    assert table["enrichment"].isna().all()
    assert (table["n"] > 0).all()


def test_subregion_enrichment_skips_a_feature_with_no_kmers(attribution):
    # A valid interval inside the 3'UTR that the fixture leaves unpopulated.
    table = subregion_enrichment(attribution, subregions=(("EMPTY", 6060, 6080, "sense"),))
    assert table.empty


# ── Reproduction guard ────────────────────────────────────────────────────────


def test_verify_against_published_passes_on_a_faithful_table(attribution, tmp_path):
    table = domain_enrichment(attribution)
    published = tmp_path / "domain_enrichment.csv"
    table.rename(columns={"p_enriched": "p_value"})[
        ["domain", "n", "n_top", "enrichment", "p_value"]
    ].to_csv(published, index=False)
    report = verify_against_published(table, published)
    assert report["matches"].all()


def test_verify_against_published_fails_on_a_perturbed_table(attribution, tmp_path):
    table = domain_enrichment(attribution)
    published = tmp_path / "domain_enrichment.csv"
    perturbed = table.rename(columns={"p_enriched": "p_value"})[
        ["domain", "n", "n_top", "enrichment", "p_value"]
    ].copy()
    perturbed.loc[perturbed["domain"] == "ORF2_C", "enrichment"] *= 1.5
    perturbed.to_csv(published, index=False)
    report = verify_against_published(table, published)
    assert not report["matches"].all()
    assert (report.loc[report["domain"] == "ORF2_C", "note"] == "MISMATCH").all()


def test_verify_against_published_flags_a_missing_domain(attribution, tmp_path):
    table = domain_enrichment(attribution)
    published = tmp_path / "domain_enrichment.csv"
    subset = table.rename(columns={"p_enriched": "p_value"})[
        ["domain", "n", "n_top", "enrichment", "p_value"]
    ]
    subset[subset["domain"] != "ORF1"].to_csv(published, index=False)
    report = verify_against_published(table, published).set_index("domain")
    assert not report.loc["ORF1", "matches"]
    assert report.loc["ORF1", "note"] == "absent from published"
