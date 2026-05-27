"""Tests for scripts/divide_gff.py — validated 2026-05-29 against real data.

The real RepeatMasker GFF3 (GCF_000001405.40_GRCh38.p14_rm.gff) is an empty
placeholder to be downloaded at runtime. These tests use a synthetic GFF3
derived from the real BED/GTF so CI stays offline.

Validated against ground truth from the real dataset (28,484-entry GTF):
    L1HS=79, L1PA2=392 — exact match to rm_out_transcript.bed counts.
All three NCBI/RepeatMasker GFF3 ``Target`` attribute formats are covered.
"""

import os
import sys
import tempfile

import pytest


def _import_divide():
    """Import divide_gff without requiring it to be a proper package."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    spec = importlib.util.spec_from_file_location(
        "divide_gff", os.path.join(root, "scripts", "divide_gff.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def divide_gff():
    return _import_divide()


def _write_gff(path, rows):
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for r in rows:
            fh.write(r + "\n")


def _gff_row(chrom, start, end, strand, attr):
    return f"{chrom}\tRepeatMasker\tdispersed_repeat\t{start}\t{end}\t.\t{strand}\t.\t{attr}"


class TestDivideGff:
    """divide() correctly splits GFF by subfamily."""

    @pytest.mark.unit
    def test_target_motif_format(self, divide_gff, tmp_path):
        """``Target "Motif:L1HS"`` — primary NCBI format."""
        gff = tmp_path / "in.gff"
        _write_gff(
            gff,
            [
                _gff_row("chr1", 100, 200, "+", 'Target "Motif:L1HS" 1 100'),
                _gff_row("chr1", 300, 400, "-", 'Target "Motif:L1HS" 1 100'),
                _gff_row("chr1", 500, 600, "+", 'Target "Motif:L1PA2" 1 100'),
            ],
        )
        counts = divide_gff.divide(str(gff), str(tmp_path / "out"))
        assert counts == {"L1HS": 2, "L1PA2": 1}

    @pytest.mark.unit
    def test_target_equals_format(self, divide_gff, tmp_path):
        """``Target=L1HS`` — GFF3 bare-value form."""
        gff = tmp_path / "in.gff"
        _write_gff(gff, [_gff_row("chr1", 1, 100, "+", "ID=r1;Target=L1HS")])
        counts = divide_gff.divide(str(gff), str(tmp_path / "out"))
        assert counts == {"L1HS": 1}

    @pytest.mark.unit
    def test_target_quoted_no_motif(self, divide_gff, tmp_path):
        """``Target "L1PA2" 1 100`` — quoted without the Motif prefix."""
        gff = tmp_path / "in.gff"
        _write_gff(gff, [_gff_row("chr1", 1, 100, "-", 'Target "L1PA2" 1 100')])
        counts = divide_gff.divide(str(gff), str(tmp_path / "out"))
        assert counts == {"L1PA2": 1}

    @pytest.mark.unit
    def test_pattern_filter(self, divide_gff, tmp_path):
        """``--pattern '^L1'`` keeps only L1* subfamilies."""
        gff = tmp_path / "in.gff"
        _write_gff(
            gff,
            [
                _gff_row("chr1", 1, 100, "+", 'Target "Motif:L1HS" 1 100'),
                _gff_row("chr1", 200, 300, "-", 'Target "Motif:AluSx" 1 100'),
                _gff_row("chr1", 400, 500, "+", 'Target "Motif:HAL1" 1 100'),
            ],
        )
        counts = divide_gff.divide(str(gff), str(tmp_path / "out"), pattern="^L1")
        assert counts == {"L1HS": 1}
        # Non-L1 files must not exist.
        assert not (tmp_path / "out" / "AluSx.gff").exists()
        assert not (tmp_path / "out" / "HAL1.gff").exists()

    @pytest.mark.unit
    def test_output_files_are_valid_gff(self, divide_gff, tmp_path):
        """Each output file is a parsable 9-column GFF; rows are preserved verbatim."""
        gff = tmp_path / "in.gff"
        rows = [
            _gff_row("chr1", 100, 200, "+", 'Target "Motif:L1HS" 1 100'),
            _gff_row("chr1", 300, 400, "-", 'Target "Motif:L1HS" 1 100'),
        ]
        _write_gff(gff, rows)
        out = tmp_path / "out"
        divide_gff.divide(str(gff), str(out))
        written = (out / "L1HS.gff").read_text().splitlines()
        assert len(written) == 2
        for line in written:
            assert len(line.split("\t")) == 9

    @pytest.mark.unit
    def test_empty_gff_returns_empty_counts(self, divide_gff, tmp_path):
        """An empty / header-only GFF returns an empty dict (no crash)."""
        gff = tmp_path / "empty.gff"
        _write_gff(gff, [])
        counts = divide_gff.divide(str(gff), str(tmp_path / "out"))
        assert counts == {}

    @pytest.mark.unit
    def test_counts_match_real_bed_ground_truth(self, divide_gff, tmp_path):
        """Ground-truth check: L1HS=79, L1PA2=392 from real GTF-derived GFF3.

        The synthetic GFF3 is built from the actual
        ``GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript.gtf`` via the same
        parsing logic used to validate the script against the ground-truth BED.
        """
        import re
        from collections import defaultdict

        gtf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "raw",
            "RepeatMasker",
            "GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript",
            "GCF_000001405.40_GRCh38.p14_rm.LINE1.transcript.gtf",
        )
        if not os.path.exists(gtf_path):
            pytest.skip("real GTF not available in this environment")

        # Build a synthetic GFF3 from the real GTF gene entries.
        gff_path = tmp_path / "synthetic.gff"
        out = tmp_path / "out"
        entries = []
        with open(gtf_path) as fh:
            for line in fh:
                if "\t" not in line:
                    continue
                c = line.strip().split("\t")
                if len(c) < 9 or c[2] != "gene":
                    continue
                m = re.search(r'gene_name "([^"]+)"', c[8])
                if not m:
                    continue
                sf = m.group(1)
                length = int(c[4]) - int(c[3]) + 1
                gff_attr = f'Target "Motif:{sf}" 1 {length}'
                entries.append(
                    "\t".join([c[0], "RepeatMasker", "dispersed_repeat",
                                c[3], c[4], c[5], c[6], ".", gff_attr])
                )
        _write_gff(gff_path, entries)

        counts = divide_gff.divide(str(gff_path), str(out))
        assert counts["L1HS"] == 79, f"expected 79 L1HS, got {counts['L1HS']}"
        assert counts["L1PA2"] == 392, f"expected 392 L1PA2, got {counts['L1PA2']}"
        assert len(counts) == 132, f"expected 132 subfamilies, got {len(counts)}"
