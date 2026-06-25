"""Tests for the RepeatMasker .out parsing + strict-label core (Track B2)."""

import pytest

from trap.utils.rmout import (
    best_task_class,
    is_l1,
    load_chrom_map,
    parse_rmout_line,
    resolve_fragment_label,
)

pytestmark = pytest.mark.unit

# A real-shaped RepeatMasker .out data row (whitespace-delimited).
_ROW = "  2331  12.5  2.1  1.3  NC_000001.11  10469  10618  (248945372)  +  L1PA3  LINE/L1  1  150  (5900)  1"
_HEADER = "   SW   perc perc perc  query  position in query  matching  repeat"


def test_parse_rmout_line_coords_and_family():
    rec = parse_rmout_line(_ROW)
    assert rec is not None
    chrom, start, end, family, div, strand = rec
    assert chrom == "NC_000001.11"
    assert start == 10468 and end == 10618  # 1-based inclusive 10469 → 0-based 10468
    assert family == "L1PA3"
    assert div == pytest.approx(12.5)
    assert strand == "+"


def test_parse_rmout_strand_c_is_minus_and_header_skipped():
    rev = _ROW.replace("  +  L1PA3", "  C  L1PA3")
    assert parse_rmout_line(rev)[5] == "-"
    assert parse_rmout_line(_HEADER) is None
    assert parse_rmout_line("") is None


def test_is_l1():
    assert is_l1("L1PA3", "LINE/L1")
    assert is_l1("L1HS", "")  # by name
    assert is_l1("L1ME4a", "")
    assert not is_l1("AluY", "SINE/Alu")


def test_best_task_class_priority():
    assert best_task_class(["L1PA3", "L1HS"]) == "L1HS"  # L1HS outranks L1PA
    assert best_task_class(["L1PA3", "L1PA8"]) == "L1PA"
    assert best_task_class(["L1ME4a", "L1MB8"]) == "OTHER"  # ancient only
    assert best_task_class(["L1PA3", "L1ME4a"]) == "L1PA"  # young wins over ancient


def test_emit_labeled_filters_and_tags(tmp_path):
    from trap.utils.rmout import emit_labeled

    r1 = tmp_path / "all_R1.fq"
    r2 = tmp_path / "all_R2.fq"
    # 3 fragments; only frag1 (L1PA) and frag3 (NEGATIVE) are labelled.
    r1.write_text("@frag1/1\nAAAA\n+\nIIII\n@frag2/1\nCCCC\n+\nIIII\n@frag3/1\nGGGG\n+\nIIII\n")
    r2.write_text("@frag1/2\nTTTT\n+\nIIII\n@frag2/2\nGGGG\n+\nIIII\n@frag3/2\nCCCC\n+\nIIII\n")
    labels = tmp_path / "labels.tsv"
    labels.write_text("frag1\tL1PA\nfrag3\tNEGATIVE\n")
    o1, o2 = tmp_path / "seq_R1.fq", tmp_path / "seq_R2.fq"
    emit_labeled(r1_in=r1, r2_in=r2, labels=labels, r1_out=o1, r2_out=o2)

    out1 = o1.read_text().splitlines()
    assert out1[0] == "@frag1/1|L1PA"  # tag appended, mate marker preserved
    assert "@frag2/1|" not in o1.read_text()  # unlabelled fragment dropped
    assert o2.read_text().splitlines()[0] == "@frag1/2|L1PA"
    assert o1.read_text().count("@frag") == 2  # frag1 + frag3 kept


def test_load_chrom_map(tmp_path):
    # NCBI assembly_report.txt: col 6 = RefSeq-Accn, col 9 = UCSC-style-name.
    report = tmp_path / "assembly_report.txt"
    report.write_text(
        "# Assembly name: GRCh38.p14\n"
        "# Sequence-Name\tRole\tMol\tType\tGenBank\tRel\tRefSeq-Accn\tUnit\tLen\tUCSC-name\n"
        "1\tassembled\t1\tChromosome\tCM000663.2\t=\tNC_000001.11\tPrimary\t248956422\tchr1\n"
        "6\tassembled\t6\tChromosome\tCM000668.2\t=\tNC_000006.12\tPrimary\t170805979\tchr6\n"
        "HSCHR1\tunlocalized\t1\tChromosome\tKI270706.1\t=\tNT_187361.1\tUnit\t175055\tna\n"
    )
    cmap = load_chrom_map(str(report))
    assert cmap == {"NC_000001.11": "chr1", "NC_000006.12": "chr6"}  # 'na' UCSC skipped


def test_resolve_fragment_label():
    # ≥fraction L1 overlap → the L1 class
    assert resolve_fragment_label(["L1PA3"], any_overlap=True) == "L1PA"
    assert resolve_fragment_label(["L1HS"], any_overlap=True) == "L1HS"
    # ancient-only strict hit → OTHER (caller drops)
    assert resolve_fragment_label(["L1ME4a"], any_overlap=True) == "OTHER"
    # no overlap at all → NEGATIVE
    assert resolve_fragment_label([], any_overlap=False) == "NEGATIVE"
    # partial overlap only (any but not strict) → DROP (ambiguous boundary)
    assert resolve_fragment_label([], any_overlap=True) == "DROP"
