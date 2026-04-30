"""Tests for trap.utils.preprocessing_sequences — dataset_loader splits."""

import textwrap
from pathlib import Path

import pytest

from trap.utils.labels import TASK_PRIORITY, task_label
from trap.utils.preprocessing_sequences import (
    _extract_label,
    _extract_transcript_id,
    _fragment_key,
    _resolve_fragment_labels,
    _transcript_level_split,
    dataset_loader,
)

# ---------------------------------------------------------------------------
# FASTA fixture content
# 10 reads, 3 classes, 10 unique transcript IDs.
# ---------------------------------------------------------------------------

_FASTA_CONTENT = textwrap.dedent("""\
    >T001|L1HS-r1
    ACTGACTGACTGACTGACTGACTGACTGACTG
    >T002|L1HS-r2
    CCCCGGGGTTTTAAAACCCCGGGGTTTTAAAA
    >T003|L1PA-r1
    GGGGCCCCTTTTAAAACCCCGGGGTTTTAAAA
    >T004|L1PA-r2
    TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    >T005|NEGATIVE-r1
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    >T006|L1HS-r3
    ACTGACTGACTGACTGACTGACTGACTGACTG
    >T007|L1PA-r3
    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    >T008|NEGATIVE-r2
    GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
    >T009|L1HS-r4
    TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    >T010|NEGATIVE-r3
    ACTGACTGACTGACTGACTGACTGACTGACTG
""")


@pytest.fixture
def small_fasta(tmp_path):
    p = tmp_path / "small.fasta"
    p.write_text(_FASTA_CONTENT)
    return p


# ---------------------------------------------------------------------------
# Label / transcript-ID extraction utilities
# ---------------------------------------------------------------------------


class TestLabelExtraction:
    @pytest.mark.unit
    def test_extract_label_pipe_format(self):
        assert _extract_label("T001|L1HS-r1") == "L1HS"

    @pytest.mark.unit
    def test_extract_label_no_pipe_returns_full(self):
        assert _extract_label("plain_id") == "plain_id"

    @pytest.mark.unit
    def test_extract_label_multiple_pipes(self):
        assert _extract_label("a|b|NEGATIVE-x") == "NEGATIVE"

    @pytest.mark.unit
    def test_extract_transcript_id(self):
        assert _extract_transcript_id("T001|L1HS-r1") == "T001"

    @pytest.mark.unit
    def test_extract_transcript_id_no_pipe(self):
        assert _extract_transcript_id("plain_id") == "plain_id"


# ---------------------------------------------------------------------------
# _transcript_level_split
# ---------------------------------------------------------------------------


class TestTranscriptLevelSplit:
    @pytest.mark.integration
    def test_no_overlap_between_splits(self, small_fasta):
        from datasets import Dataset

        rows = [
            {"sequence": "ACTG", "label": "L1HS", "transcript_id": f"T{i:03d}"}
            for i in range(20)
        ]
        ds = Dataset.from_list(rows)
        ds = ds.class_encode_column("label")
        splits = _transcript_level_split(ds, test_split=0.2, val_split=None)
        train_tids = set(splits["train"]["transcript_id"])
        test_tids = set(splits["test"]["transcript_id"])
        assert train_tids.isdisjoint(test_tids)

    @pytest.mark.integration
    def test_val_split_produces_eval_key(self):
        from datasets import Dataset

        rows = [
            {"sequence": "ACTG", "label": "L1HS", "transcript_id": f"T{i:03d}"}
            for i in range(30)
        ]
        ds = Dataset.from_list(rows)
        ds = ds.class_encode_column("label")
        splits = _transcript_level_split(ds, test_split=0.2, val_split=0.1)
        assert "eval" in splits
        assert "train" in splits
        assert "test" in splits

    @pytest.mark.unit
    def test_raises_if_train_empty(self):
        from datasets import Dataset

        rows = [
            {"sequence": "ACTG", "label": "L1HS", "transcript_id": "T001"},
            {"sequence": "ACTG", "label": "L1HS", "transcript_id": "T002"},
        ]
        ds = Dataset.from_list(rows)
        ds = ds.class_encode_column("label")
        with pytest.raises(ValueError):
            _transcript_level_split(ds, test_split=0.9, val_split=0.1)


# ---------------------------------------------------------------------------
# dataset_loader — val_split=None (regression for Bug 2)
# ---------------------------------------------------------------------------


class TestDatasetLoader:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_default_val_split_does_not_crash(self, small_fasta):
        result = dataset_loader(
            builder=str(small_fasta),
            file_format="fasta",
            test_split=0.2,
            val_split=None,
            split_strategy="transcript-level",
        )
        assert "train" in result
        assert "test" in result
        assert "eval" not in result

    @pytest.mark.integration
    @pytest.mark.slow
    def test_val_split_produces_eval(self, small_fasta):
        result = dataset_loader(
            builder=str(small_fasta),
            file_format="fasta",
            test_split=0.1,
            val_split=0.1,
            split_strategy="transcript-level",
        )
        assert "eval" in result

    @pytest.mark.integration
    @pytest.mark.slow
    def test_read_level_val_split_none_does_not_crash(self, small_fasta):
        # test_split=0.3 ensures test set has >= 1 sample per class (3 classes, 10 records)
        result = dataset_loader(
            builder=str(small_fasta),
            file_format="fasta",
            test_split=0.3,
            val_split=None,
            split_strategy="read-level",
        )
        assert "train" in result
        assert "test" in result

    @pytest.mark.integration
    @pytest.mark.slow
    def test_no_transcript_leakage(self, small_fasta):
        result = dataset_loader(
            builder=str(small_fasta),
            file_format="fasta",
            test_split=0.2,
            val_split=None,
            split_strategy="transcript-level",
        )
        train_ids = set(result["train"]["transcript_id"])
        test_ids = set(result["test"]["transcript_id"])
        assert train_ids.isdisjoint(test_ids)


# ---------------------------------------------------------------------------
# Multi-subfamily fragment dedup (priority resolution)
# ---------------------------------------------------------------------------


class TestFragmentKey:
    """A fragment key collapses both mates and all per-subfamily labelled copies."""

    @pytest.mark.unit
    def test_collapses_mate_and_label(self):
        a = _fragment_key("ENST1|gene|-45/1|L1HS")
        b = _fragment_key("ENST1|gene|-45/2|L1PA")  # other mate, other label
        c = _fragment_key("ENST1|gene|-45/1|L1PA2")  # duplicate copy
        assert a == b == c == "ENST1|gene|-45"


class TestResolveFragmentLabels:
    """Priority resolution keeps the highest-ranked label per fragment."""

    @pytest.mark.unit
    def test_priority_and_unknown_rank_lowest(self):
        reads = [
            ("seq", "ENST1|g|-45/1|L1PA2"),  # -> L1PA
            ("seq", "ENST1|g|-45/1|L1HS"),  # -> L1HS (wins)
            ("seq", "ENST2|g|-46/1|L1MA8"),  # -> OTHER (only copy)
            ("seq", "ENST2|g|-46/1|L1HS"),  # -> L1HS (beats OTHER)
            ("seq", "ENST3|g|-47/1|NEGATIVE"),
        ]
        resolved = _resolve_fragment_labels(reads, task_label, TASK_PRIORITY)
        assert resolved["ENST1|g|-45"] == "L1HS"
        assert resolved["ENST2|g|-46"] == "L1HS"
        assert resolved["ENST3|g|-47"] == "NEGATIVE"


class TestDedupInDatasetLoader:
    """A read overlapping two subfamilies yields one example at the priority label."""

    def _write_paired_fq(self, tmp_path, rows):
        r1 = tmp_path / "r1.fq"
        r2 = tmp_path / "r2.fq"
        seq = "ACTGACTGACTGACTGACTG"
        r1.write_text("".join(f"@{rid}/1|{lbl}\n{seq}\n+\n{'I' * len(seq)}\n" for rid, lbl in rows))
        r2.write_text("".join(f"@{rid}/2|{lbl}\n{seq}\n+\n{'I' * len(seq)}\n" for rid, lbl in rows))
        return r1, r2

    @pytest.mark.integration
    def test_multi_subfamily_read_deduped_to_priority(self, tmp_path):
        # Fragment -45 appears under both L1PA2 and L1HS; must collapse to one L1HS.
        # Extra fragments from distinct transcript IDs (ENST4–6) ensure the
        # stratified split has ≥2 transcripts per class (needed for test_split=0.4).
        rows = [
            ("ENST1|g|-45", "L1PA2"),   # conflict copy 1
            ("ENST1|g|-45", "L1HS"),    # conflict copy 2 → resolves to L1HS
            ("ENST2|g|-46", "L1PA2"),   # L1PA
            ("ENST3|g|-47", "NEGATIVE"),
            ("ENST4|g|-48", "L1HS"),    # second L1HS transcript (ENST4)
            ("ENST5|g|-49", "L1PA2"),   # second L1PA transcript (ENST5)
            ("ENST6|g|-50", "NEGATIVE"),  # second NEGATIVE transcript (ENST6)
        ]
        r1, r2 = self._write_paired_fq(tmp_path, rows)
        ds = dataset_loader(
            builder=str(r1),
            pair=str(r2),
            file_format="fastq",
            test_split=0.4,
            split_strategy="transcript-level",
            num_proc=1,
            label_fn=task_label,
            drop_labels=frozenset({"OTHER"}),
            dedup_priority=TASK_PRIORITY,
        )
        labels = []
        for split in ds:
            names = ds[split].features["label"].names
            labels += [names[i] for i in ds[split]["label"]]
        # 7 input rows → 6 examples (frag -45 deduped from 2 copies to 1)
        assert len(labels) == 6
        assert set(labels) == {"L1HS", "L1PA", "NEGATIVE"}  # all classes present
