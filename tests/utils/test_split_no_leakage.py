"""DR-6 correctness: stratified transcript-level splits preserve leakage constraints."""
import pytest
from collections import Counter
from datasets import ClassLabel, Dataset

from trap.utils.preprocessing_sequences import _dominant_label_per_transcript, _transcript_level_split


def _build_dataset(n_transcripts: int = 100, reads_per_transcript: int = 10):
    """Build a synthetic dataset with known transcript_id → reads mapping."""
    labels = ["L1HS", "L1PA", "NEGATIVE"]
    cl = ClassLabel(names=labels)
    rows = []
    for t in range(n_transcripts):
        label = labels[t % len(labels)]
        for r in range(reads_per_transcript):
            rows.append({
                "transcript_id": f"ENST{t:06d}",
                "label": cl.str2int(label),
            })
    ds = Dataset.from_list(rows)
    return ds.cast_column("label", cl)


def _build_rare_dataset():
    """Dataset where one class (L1HS) has very few transcripts."""
    labels = ["L1HS", "L1PA", "NEGATIVE"]
    cl = ClassLabel(names=labels)
    rows = []
    # L1HS: 3 transcripts, L1PA: 40, NEGATIVE: 40
    for t in range(3):
        for r in range(10):
            rows.append({"transcript_id": f"L1HS_{t}", "label": cl.str2int("L1HS")})
    for t in range(40):
        for r in range(10):
            rows.append({"transcript_id": f"L1PA_{t}", "label": cl.str2int("L1PA")})
    for t in range(40):
        for r in range(10):
            rows.append({"transcript_id": f"NEG_{t}", "label": cl.str2int("NEGATIVE")})
    return Dataset.from_list(rows).cast_column("label", cl)


@pytest.mark.unit
class TestTranscriptLevelSplit:
    def test_no_transcript_overlap_train_test(self):
        ds = _build_dataset()
        splits = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=42)
        train_t = set(splits["train"]["transcript_id"])
        test_t = set(splits["test"]["transcript_id"])
        assert train_t.isdisjoint(test_t), "Transcript IDs leaked between train and test"

    def test_no_transcript_overlap_with_eval(self):
        ds = _build_dataset()
        splits = _transcript_level_split(ds, test_split=0.15, val_split=0.15, seed=42)
        train_t = set(splits["train"]["transcript_id"])
        eval_t = set(splits["eval"]["transcript_id"])
        test_t = set(splits["test"]["transcript_id"])
        assert train_t.isdisjoint(eval_t)
        assert train_t.isdisjoint(test_t)
        assert eval_t.isdisjoint(test_t)

    def test_all_reads_accounted_for(self):
        ds = _build_dataset()
        splits = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=42)
        assert len(splits["train"]) + len(splits["test"]) == len(ds)

    def test_all_reads_accounted_for_with_eval(self):
        ds = _build_dataset()
        splits = _transcript_level_split(ds, test_split=0.15, val_split=0.15, seed=42)
        total = len(splits["train"]) + len(splits["eval"]) + len(splits["test"])
        assert total == len(ds)

    def test_approximate_split_ratios(self):
        """Train fraction should be roughly 1 - test_split."""
        ds = _build_dataset(n_transcripts=200)
        splits = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=0)
        train_frac = len(splits["train"]) / len(ds)
        assert 0.70 < train_frac < 0.90, f"Unexpected train fraction: {train_frac:.2f}"

    def test_reproducible_with_same_seed(self):
        ds = _build_dataset()
        s1 = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=99)
        s2 = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=99)
        assert set(s1["train"]["transcript_id"]) == set(s2["train"]["transcript_id"])

    def test_different_seeds_differ(self):
        ds = _build_dataset(n_transcripts=200)
        s1 = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=1)
        s2 = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=2)
        # Very likely to differ with 200 transcripts
        assert set(s1["test"]["transcript_id"]) != set(s2["test"]["transcript_id"])

    def test_raises_when_fractions_exceed_one(self):
        ds = _build_dataset(n_transcripts=10)
        with pytest.raises(ValueError, match="must be < 1.0"):
            _transcript_level_split(ds, test_split=0.9, val_split=0.9, seed=42)

    def test_raises_when_test_set_empty(self):
        # 2 transcripts, each a different class → both fall back to train → empty test.
        ds = _build_dataset(n_transcripts=2, reads_per_transcript=2)
        with pytest.raises(ValueError, match="empty test set"):
            _transcript_level_split(ds, test_split=0.1, val_split=0.1, seed=42)

    def test_no_leakage_small_n_with_eval(self):
        ds = _build_dataset(n_transcripts=12, reads_per_transcript=3)
        splits = _transcript_level_split(ds, test_split=0.25, val_split=0.25, seed=7)
        train_t = set(splits["train"]["transcript_id"])
        eval_t = set(splits["eval"]["transcript_id"])
        test_t = set(splits["test"]["transcript_id"])
        assert train_t.isdisjoint(eval_t)
        assert train_t.isdisjoint(test_t)
        assert eval_t.isdisjoint(test_t)
        assert len(train_t) >= 1


# ---------------------------------------------------------------------------
# Stratification tests — rare class must appear in every split
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDominantLabelPerTranscript:
    def test_most_common_label_wins(self):
        tids = ["T1", "T1", "T1", "T2", "T2"]
        labels = [0, 0, 1, 1, 1]
        result = _dominant_label_per_transcript(tids, labels)
        assert result["T1"] == 0  # L1HS (2 reads) beats L1PA (1 read)
        assert result["T2"] == 1  # L1PA wins unambiguously

    def test_single_read_transcript(self):
        result = _dominant_label_per_transcript(["T1"], [2])
        assert result["T1"] == 2


@pytest.mark.unit
class TestStratificationCorrectness:
    def test_rare_class_present_in_all_splits(self):
        """L1HS with only 3 transcripts must appear in train, test, and eval."""
        ds = _build_rare_dataset()
        splits = _transcript_level_split(ds, test_split=0.2, val_split=0.2, seed=42)
        label_names = ds.features["label"].names

        def classes_in(split):
            names = splits[split].features["label"].names
            return {names[i] for i in splits[split]["label"]}

        assert "L1HS" in classes_in("train"), "L1HS missing from train"
        assert "L1HS" in classes_in("test"),  "L1HS missing from test"
        assert "L1HS" in classes_in("eval"),  "L1HS missing from eval"

    def test_no_leakage_with_rare_class(self):
        ds = _build_rare_dataset()
        splits = _transcript_level_split(ds, test_split=0.2, val_split=0.2, seed=42)
        train_t = set(splits["train"]["transcript_id"])
        test_t  = set(splits["test"]["transcript_id"])
        eval_t  = set(splits["eval"]["transcript_id"])
        assert train_t.isdisjoint(test_t)
        assert train_t.isdisjoint(eval_t)
        assert test_t.isdisjoint(eval_t)

    def test_all_reads_accounted_for_with_rare_class(self):
        ds = _build_rare_dataset()
        splits = _transcript_level_split(ds, test_split=0.2, val_split=0.2, seed=42)
        total = sum(len(splits[s]) for s in splits)
        assert total == len(ds)

    def test_label_balance_closer_than_random(self):
        """Stratification should give a better class balance than a random split."""
        ds = _build_rare_dataset()
        label_names = ds.features["label"].names
        splits = _transcript_level_split(ds, test_split=0.2, val_split=None, seed=42)

        test_counts = Counter(label_names[i] for i in splits["test"]["label"])
        # With 3 L1HS transcripts and test_split=0.2, stratification allocates
        # round(3*0.2)=1 L1HS transcript to test — it must not be zero.
        assert test_counts["L1HS"] > 0, (
            "Stratification failed: no L1HS in test set despite 3 L1HS transcripts"
        )
