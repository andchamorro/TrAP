"""Tests for trap.modeling.train helpers (Phase-3 reproducibility wiring)."""

import json

import numpy as np
import pytest


class TestComputeMaskingMetrics:
    """``compute_masking_metrics`` correctly computes masked-token accuracy.

    Regression guard for the metrics inconsistency bug where
    ``train_result.metrics["train_loss"]`` is a running mean from step 0
    (includes early high-loss epochs) while ``eval_loss`` is from the final
    checkpoint — they are not directly comparable for publication.
    Perplexity is now derived from ``eval_loss`` via ``_PerplexityLogCallback``.
    """

    @pytest.mark.unit
    def test_partial_accuracy(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import compute_masking_metrics

        # 4 positions: positions 1 and 3 are masked (-100 = ignored)
        # position 0 (ignored): argmax=0 — must not count
        # position 1 (masked):  argmax=1, label=1 -> correct
        # position 2 (ignored): argmax=2 — must not count
        # position 3 (masked):  argmax=2, label=1 -> wrong
        logits = np.array([[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9], [0.3, 0.3, 0.4]]])
        labels = np.array([[-100, 1, -100, 1]])
        result = compute_masking_metrics((logits, labels))
        assert abs(result["accuracy"] - 0.5) < 1e-6

    @pytest.mark.unit
    def test_all_correct(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import compute_masking_metrics

        logits = np.array([[[0.9, 0.1], [0.1, 0.9]]])
        labels = np.array([[0, 1]])
        result = compute_masking_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    @pytest.mark.unit
    def test_no_masked_tokens_returns_zero(self):
        """No masked positions → accuracy 0.0 (not a division-by-zero crash)."""
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import compute_masking_metrics

        logits = np.array([[[0.9, 0.1], [0.1, 0.9]]])
        labels = np.array([[-100, -100]])
        result = compute_masking_metrics((logits, labels))
        assert result["accuracy"] == 0.0

    @pytest.mark.unit
    def test_only_masked_positions_count(self):
        """Non-masked positions (label == -100) must never affect the accuracy."""
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import compute_masking_metrics

        # position 0: not masked; position 1: masked and correct
        logits = np.array([[[0.1, 0.9], [0.9, 0.1]]])  # argmax = [1, 0]
        labels = np.array([[-100, 0]])  # position 1 label is 0 → correct
        result = compute_masking_metrics((logits, labels))
        assert result["accuracy"] == 1.0

    @pytest.mark.unit
    def test_accepts_pre_reduced_predictions(self):
        """With the preprocess hook, predictions arrive already argmax'd (2-D)."""
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import compute_masking_metrics

        # predictions already reduced to token ids (same ndim as labels)
        # masked positions are 1 and 3: pred[1]=1==label 1 (correct);
        # pred[3]=2!=label 1 (wrong) → accuracy 0.5
        predictions = np.array([[0, 1, 2, 2]])
        labels = np.array([[-100, 1, -100, 1]])
        result = compute_masking_metrics((predictions, labels))
        assert abs(result["accuracy"] - 0.5) < 1e-6


class TestPerplexityLogCallback:
    """``_PerplexityLogCallback`` appends perplexity for every evaluation loss logged."""

    @pytest.mark.unit
    def test_adds_eval_perplexity(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import _PerplexityLogCallback

        cb = _PerplexityLogCallback()
        logs = {"eval_loss": 2.0, "eval_accuracy": 0.9}
        cb.on_log(None, None, None, logs=logs)
        assert "eval_perplexity" in logs
        assert abs(logs["eval_perplexity"] - np.exp(2.0)) < 1e-6

    @pytest.mark.unit
    def test_adds_final_train_perplexity(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import _PerplexityLogCallback

        cb = _PerplexityLogCallback()
        logs = {"final_train_loss": 1.5}
        cb.on_log(None, None, None, logs=logs)
        assert abs(logs["final_train_perplexity"] - np.exp(1.5)) < 1e-6

    @pytest.mark.unit
    def test_skips_train_loss(self):
        """Running-mean train_loss must not produce a perplexity entry."""
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import _PerplexityLogCallback

        cb = _PerplexityLogCallback()
        logs = {"train_loss": 3.0, "eval_loss": 2.0}
        cb.on_log(None, None, None, logs=logs)
        assert "train_perplexity" not in logs
        assert "eval_perplexity" in logs

    @pytest.mark.unit
    def test_noop_on_empty_logs(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import _PerplexityLogCallback

        cb = _PerplexityLogCallback()
        cb.on_log(None, None, None, logs=None)  # must not raise
        cb.on_log(None, None, None, logs={})


class TestComputeClassificationMetrics:
    """``make_compute_classification_metrics`` returns accuracy/f1/precision/recall/roc_auc."""

    @pytest.mark.unit
    def test_binary_roc_auc_perfect(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        pytest.importorskip("sklearn", reason="scikit-learn not installed")
        from trap.modeling.train import make_compute_classification_metrics

        compute_metrics = make_compute_classification_metrics()
        # perfect binary classifier: high logit for correct class
        logits = np.array([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 1, 0, 1])
        result = compute_metrics((logits, labels))
        assert result["roc_auc"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_binary_roc_auc_random(self):
        pytest.importorskip("accelerate", reason="accelerate not installed")
        pytest.importorskip("sklearn", reason="scikit-learn not installed")
        from trap.modeling.train import make_compute_classification_metrics

        compute_metrics = make_compute_classification_metrics()
        # uninformative classifier: equal logits → roc_auc ≈ 0.5
        logits = np.zeros((10, 2))
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = compute_metrics((logits, labels))
        assert 0.0 <= result["roc_auc"] <= 1.0

    @pytest.mark.unit
    def test_roc_auc_nan_on_single_class(self):
        """sklearn raises ValueError when only one class is present; must return nan."""
        pytest.importorskip("accelerate", reason="accelerate not installed")
        pytest.importorskip("sklearn", reason="scikit-learn not installed")
        from trap.modeling.train import make_compute_classification_metrics

        compute_metrics = make_compute_classification_metrics()
        logits = np.array([[10.0, -10.0], [10.0, -10.0]])
        labels = np.array([0, 0])  # single class in batch
        result = compute_metrics((logits, labels))
        assert np.isnan(result["roc_auc"])


class TestMlmPreprocessLogitsForMetrics:
    """``mlm_preprocess_logits_for_metrics`` collapses the vocab axis on-device."""

    @pytest.mark.unit
    def test_argmax_over_vocab_axis(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import mlm_preprocess_logits_for_metrics

        # [batch=1, seq_len=3, vocab=4]; argmax over vocab → [1, 3]
        logits = torch.tensor([[[0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, 0.7], [0.5, 0.1, 0.1, 0.1]]])
        labels = torch.tensor([[1, 3, 0]])
        reduced = mlm_preprocess_logits_for_metrics(logits, labels)
        assert reduced.shape == labels.shape
        assert reduced.tolist() == [[1, 3, 0]]

    @pytest.mark.unit
    def test_unwraps_tuple_output(self):
        """Some models return a tuple; the hook must take the first element."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("accelerate", reason="accelerate not installed")
        from trap.modeling.train import mlm_preprocess_logits_for_metrics

        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
        reduced = mlm_preprocess_logits_for_metrics((logits, None), torch.tensor([[1, 0]]))
        assert reduced.tolist() == [[1, 0]]


class TestReadConfigDict:
    """``_read_config_dict`` accepts both JSON and YAML trainer configs."""

    @pytest.mark.unit
    def test_reads_json(self, tmp_path):
        from trap.modeling.train import _read_config_dict

        p = tmp_path / "c.json"
        p.write_text(json.dumps({"seed": 7, "num_train_epochs": 3}))
        cfg = _read_config_dict(p)
        assert cfg == {"seed": 7, "num_train_epochs": 3}

    @pytest.mark.unit
    def test_reads_yaml(self, tmp_path):
        from trap.modeling.train import _read_config_dict

        p = tmp_path / "c.yaml"
        p.write_text("seed: 7\nnum_train_epochs: 3\nbf16: true\n")
        cfg = _read_config_dict(p)
        assert cfg == {"seed": 7, "num_train_epochs": 3, "bf16": True}

    @pytest.mark.unit
    def test_shipped_training_configs_load(self):
        """The committed config/training/*.json parse and carry seed 3469."""
        from trap.config.config import CONFIG_DIR
        from trap.modeling.train import _read_config_dict

        for name in ("mlm.json", "classification_final.json"):
            cfg = _read_config_dict(CONFIG_DIR / "training" / name)
            assert cfg["seed"] == 3469
            assert cfg["bf16"] is True


class TestAssertDatasetIdsInRange:
    """``assert_dataset_ids_in_range`` is the fail-fast guard for the CUDA gather
    OOB: a token id >= model vocab (or label >= num_labels) otherwise surfaces
    only as an async assertion + 30-min NCCL watchdog timeout deep in training.
    """

    @pytest.mark.unit
    def test_in_range_passes(self):
        from datasets import Dataset, DatasetDict

        from trap.modeling.train import assert_dataset_ids_in_range

        dd = DatasetDict({"train": Dataset.from_dict({"input_ids": [[0, 5, 100], [4, 99, 2]]})})
        assert_dataset_ids_in_range(dd, vocab_size=101, dataset_name="ok")

    @pytest.mark.unit
    def test_token_id_at_or_above_vocab_raises(self):
        from datasets import Dataset, DatasetDict

        from trap.modeling.train import assert_dataset_ids_in_range

        dd = DatasetDict({"train": Dataset.from_dict({"input_ids": [[0, 100]]})})
        with pytest.raises(ValueError, match="max input_id=100 >= model vocab_size=100"):
            assert_dataset_ids_in_range(dd, vocab_size=100, dataset_name="bad")

    @pytest.mark.unit
    def test_negative_token_id_raises(self):
        from datasets import Dataset, DatasetDict

        from trap.modeling.train import assert_dataset_ids_in_range

        dd = DatasetDict({"train": Dataset.from_dict({"input_ids": [[0, -1, 3]]})})
        with pytest.raises(ValueError, match="negative input_id"):
            assert_dataset_ids_in_range(dd, vocab_size=10, dataset_name="neg")

    @pytest.mark.unit
    def test_label_out_of_range_raises(self):
        from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

        from trap.modeling.train import assert_dataset_ids_in_range

        feats = Features(
            {"input_ids": Sequence(Value("int64")), "label": ClassLabel(names=["A", "B"])}
        )
        dd = DatasetDict(
            {"train": Dataset.from_dict({"input_ids": [[0, 1]], "label": [1]}, features=feats)}
        )
        with pytest.raises(ValueError, match="max label=1 >= num_labels=1"):
            assert_dataset_ids_in_range(dd, vocab_size=10, num_labels=1, dataset_name="bad-cls")
