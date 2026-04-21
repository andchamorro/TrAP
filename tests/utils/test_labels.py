"""Tests for trap.utils.labels (DR-2)."""
import pytest
from datasets import Dataset, DatasetDict, ClassLabel

from trap.utils.labels import analyze_labels


def _make_dataset(train_labels, test_labels):
    """Build a minimal DatasetDict for testing."""
    all_labels = sorted(set(train_labels) | set(test_labels))
    cl = ClassLabel(names=all_labels)

    def encode(labels):
        return [cl.str2int(l) for l in labels]

    train = Dataset.from_dict({"label": encode(train_labels)})
    train = train.cast_column("label", cl)
    test = Dataset.from_dict({"label": encode(test_labels)})
    test = test.cast_column("label", cl)
    return DatasetDict({"train": train, "test": test})


@pytest.mark.unit
class TestAnalyzeLabels:
    def test_returns_counts_dict(self):
        ds = _make_dataset(
            ["L1HS", "L1HS", "NEGATIVE", "L1PA"],
            ["L1HS", "NEGATIVE"],
        )
        counts = analyze_labels(ds)
        assert counts["L1HS"] == 2
        assert counts["NEGATIVE"] == 1
        assert counts["L1PA"] == 1

    def test_warns_on_unseen_test_label(self, capsys):
        # loguru routes through tqdm.write → stdout; caplog only catches stdlib logging.
        ds = _make_dataset(["L1HS", "NEGATIVE"], ["L1HS", "L1PA"])
        analyze_labels(ds)
        out = capsys.readouterr().out
        assert "L1PA" in out

    def test_no_warn_when_labels_match(self, capsys):
        ds = _make_dataset(["L1HS", "NEGATIVE"], ["L1HS", "NEGATIVE"])
        analyze_labels(ds)
        out = capsys.readouterr().out
        assert "appear in" not in out

    def test_empty_train_split_returns_empty(self, capsys):
        # An empty training split (0 rows, ClassLabel preserved, as produced by a
        # degenerate filter) must not crash on max()/np.quantile.
        base = _make_dataset(["L1HS", "NEGATIVE"], ["L1HS"])
        ds = DatasetDict({"train": base["train"].select([]), "test": base["test"]})
        counts = analyze_labels(ds)
        assert counts == {}
        assert "no labelled reads" in capsys.readouterr().out
