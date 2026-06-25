"""Tests for the minimap2/PAF alignment core (pure parsing + region mapping)."""

import pytest

from trap.analysis._l1_align import L1_3_REGIONS, l1_region, parse_paf, parse_paf_line

pytestmark = pytest.mark.unit


def test_l1_regions_cover_the_element_in_order():
    # Regions are ordered and contiguous over L1.3 (~6 kb).
    prev_hi = 0
    for _name, lo, hi in L1_3_REGIONS:
        assert lo == prev_hi + 1
        assert hi >= lo
        prev_hi = hi
    assert L1_3_REGIONS[0][1] == 1
    assert prev_hi >= 6059  # spans the full element


def test_l1_region_assignment():
    assert l1_region(500) == "5UTR"
    assert l1_region(1500) == "ORF1"
    assert l1_region(2200) == "ORF2_EN"
    assert l1_region(3800) == "ORF2_RT"
    assert l1_region(5900) == "3UTR"
    assert l1_region(99999) == "outside"


def _paf(qname, qlen, qs, qe, strand, ts, te, nmatch, alen):
    return "\t".join(
        str(x) for x in [qname, qlen, qs, qe, strand, "L19088.1", 6059, ts, te, nmatch, alen, 60]
    )


def test_parse_paf_line_identity_and_region():
    line = _paf("read7", 150, 0, 150, "+", 3700, 3850, 142, 150)
    qname, aln = parse_paf_line(line)
    assert qname == "read7"
    assert aln.identity == pytest.approx(142 / 150)
    assert aln.aligned_frac == pytest.approx(1.0)
    assert aln.region == "ORF2_RT"  # midpoint ~3775
    assert aln.strand == "+"
    assert aln.target_name == "L19088.1"  # which reference it aligned to (for subfamily)


def test_parse_paf_line_malformed_returns_none():
    assert parse_paf_line("not\ta\tpaf") is None
    assert parse_paf_line("") is None


def test_parse_paf_keeps_best_alignment_per_read():
    weak = _paf("r", 150, 0, 60, "+", 100, 160, 50, 60)  # short, low coverage
    strong = _paf("r", 150, 0, 150, "+", 2000, 2150, 145, 150)  # full, high identity
    best = parse_paf(weak + "\n" + strong)
    assert set(best) == {"r"}
    assert best["r"].region == "ORF2_EN"  # the strong (ORF2) alignment wins
    assert best["r"].identity == pytest.approx(145 / 150)
