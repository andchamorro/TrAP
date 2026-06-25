"""Tests for the Track-B1 sequence-relabelling core (CPU-only, no minimap2)."""

import pytest

from trap.analysis._l1_align import L1Alignment
from trap.analysis.relabel_by_sequence import (
    _best,
    sequence_label,
    subfamily_from_header,
)

pytestmark = pytest.mark.unit


def _aln(identity, aligned_frac, target):
    return L1Alignment(identity, aligned_frac, 1000.0, "ORF1", "+", target)


def test_subfamily_from_header_variants():
    assert subfamily_from_header("L1PA3#LINE/L1") == "L1PA3"
    assert subfamily_from_header("L1HS") == "L1HS"
    assert subfamily_from_header("L1HS L1 human") == "L1HS"
    assert subfamily_from_header("") == ""


def test_sequence_label_collapses_subfamily():
    assert sequence_label(_aln(0.95, 0.9, "L1HS#LINE/L1"), 0.8, 0.5) == ("L1HS", "L1HS")
    assert sequence_label(_aln(0.92, 0.8, "L1PA3"), 0.8, 0.5) == ("L1PA", "L1PA3")
    # Ancient L1M aligns (it IS L1 sequence) but collapses to OTHER for the 3-class.
    assert sequence_label(_aln(0.85, 0.7, "L1ME4a"), 0.8, 0.5) == ("OTHER", "L1ME4a")


def test_sequence_label_below_threshold_is_negative():
    assert sequence_label(_aln(0.70, 0.9, "L1HS"), 0.80, 0.5) == ("NEGATIVE", "NEGATIVE")  # low id
    assert sequence_label(_aln(0.95, 0.3, "L1HS"), 0.80, 0.5) == (
        "NEGATIVE",
        "NEGATIVE",
    )  # low cov
    assert sequence_label(None, 0.80, 0.5) == ("NEGATIVE", "NEGATIVE")  # unaligned


def test_best_picks_higher_identity_times_coverage():
    weak = _aln(0.99, 0.30, "L1PA8")  # high id, low cov -> 0.297
    strong = _aln(0.90, 0.90, "L1HS")  # 0.81
    assert _best(weak, strong).target_name == "L1HS"
    assert _best(None, weak) is weak
    assert _best(strong, None) is strong
    assert _best(None, None) is None
