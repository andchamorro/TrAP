"""Tests for the k -> classification ablation."""

import pytest

from trap.analysis import ablation as A

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "read_id, expected",
    [
        ("t1|L1HS-3", "L1HS"),
        ("t2|L1PA2-7", "L1PA"),
        ("t3|L1P5-1", "L1PA"),
        ("t4|NEGATIVE-9", "NEGATIVE"),
        ("t5|L1ME3-1", "OTHER"),
        ("no_label", "OTHER"),
    ],
)
def test_task_label(read_id, expected):
    assert A.task_label(read_id) == expected


def _write_fastq(tmp_path, reads_per_class=6, length=40):
    """Three compositionally-distinct classes so a linear model can separate."""
    motifs = {"L1HS": "AC", "L1PA": "AG", "NEGATIVE": "AT"}
    lines = []
    for label, motif in motifs.items():
        seq = (motif * length)[:length]
        for i in range(reads_per_class):
            lines += [f"@r{i}|{label}-{i}", seq, "+", "I" * length]
    path = tmp_path / "reads.fq"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_labeled_reads_keeps_task_classes(tmp_path):
    path = _write_fastq(tmp_path)
    seqs, labels = A.load_labeled_reads(path, max_per_class=10, seed=1)
    assert set(labels) == {"L1HS", "L1PA", "NEGATIVE"}
    assert len(seqs) == len(labels) == 18


def test_featurize_shape_and_normalisation(tmp_path):
    path = _write_fastq(tmp_path)
    seqs, _ = A.load_labeled_reads(path, max_per_class=10, seed=1)
    matrix = A.featurize(seqs, k=3, n_features=256)
    assert matrix.shape == (len(seqs), 256)
    row_sums = matrix.sum(axis=1)
    assert row_sums.min() == pytest.approx(1.0)  # L1-normalised


def test_ablation_returns_row_per_k_and_is_deterministic(tmp_path):
    path = _write_fastq(tmp_path)
    kwargs = dict(k_values=[2, 3], max_per_class=10, n_features=256, n_splits=3, seed=5)
    first = A.ablation(path, **kwargs)
    second = A.ablation(path, **kwargs)
    assert [r["k"] for r in first] == [2, 3]
    assert first == second  # full determinism
    for row in first:
        assert 0.0 <= row["macro_f1_mean"] <= 1.0


def test_ablation_separable_classes_score_well(tmp_path):
    path = _write_fastq(tmp_path)
    rows = A.ablation(path, k_values=[3], max_per_class=10, n_features=256, n_splits=3, seed=5)
    assert rows[0]["macro_f1_mean"] > 0.8  # distinct compositions are easy
