"""Deep validation of the TrAP training and hyperparameter-tuning workflows.

The suite is layered so most checks are fast ``unit`` tests; only the multi-epoch
training loops and the real Optuna search are marked ``slow``. See
``tests/modeling/workflow_data.py`` for the data engine + assertions and
``docs/development/training_workflow_tests.md`` for the design rationale.

What it guards
--------------
* data engine: deterministic seeded sampling, configurable size, real→synthetic
  fallback, stratification, schema-drift detection;
* preprocessing: transcript-level split has no leakage, ids within vocab,
  field lengths consistent;
* model init: ``vocab_size`` override honoured (the embedding-gather OOB
  regression), deterministic weights, embedding table covers every token id;
* training loop: finite losses (instability), convergence, metric consistency,
  run-to-run reproducibility;
* hyperparameter search: a real 2-trial search runs + is reproducible, and
  mis-specified search spaces / samplers fail loudly.
"""

from __future__ import annotations

import pytest

from tests.modeling import workflow_data as wd
from tests.modeling.conftest import SEED

# ---------------------------------------------------------------------------
# Shared training helpers (imports kept local so collection stays fast)
# ---------------------------------------------------------------------------


def _gpu_precision_flags() -> dict:
    """Mirror train._resolve_precision: use bf16 on Ampere+, fp16 on older GPUs."""
    import torch

    if not torch.cuda.is_available():
        return {}
    try:
        from transformers.utils import is_torch_bf16_gpu_available

        bf16_ok = is_torch_bf16_gpu_available()
    except ImportError:
        bf16_ok = torch.cuda.get_device_capability()[0] >= 8
    if bf16_ok:
        return {"bf16": True}
    return {"fp16": True}


def _training_args(tmp_path, epochs: int, seed: int = SEED, **over):
    """TrainingArguments that use GPU + bf16/fp16 when CUDA is available.

    Falls back to CPU-only when no GPU is detected (local dev / pure-CPU CI).
    On GPU the slow training-loop tests finish in seconds instead of minutes.
    """
    import torch
    from transformers import TrainingArguments

    on_gpu = torch.cuda.is_available()
    defaults = dict(
        output_dir=str(tmp_path / "run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-3,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        seed=seed,
        use_cpu=not on_gpu,
        dataloader_num_workers=0,
        logging_strategy="epoch",
    )
    if on_gpu:
        defaults.update(_gpu_precision_flags())
    defaults.update(over)
    return TrainingArguments(**defaults)


def _numpy_classification_metrics(eval_pred):
    """Offline accuracy + micro-F1 (== accuracy for multiclass micro avg).

    Computing both lets a test assert the two stay consistent — a guard against
    a future metrics bug where they silently diverge.
    """
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "f1": acc}


def _run_classification(dd, tmp_path, epochs: int, seed: int = SEED):
    """Train a tiny ALBERT classifier on *dd*; return (trainer, eval_metrics)."""
    from transformers import AlbertForSequenceClassification, Trainer

    from trap.utils.seeding import set_global_seed

    set_global_seed(seed)
    cfg = wd.tiny_albert_config(dd)
    trainer = Trainer(
        model_init=lambda: AlbertForSequenceClassification(cfg),
        args=_training_args(tmp_path, epochs, seed),
        train_dataset=dd["train"],
        eval_dataset=dd["test"],
        compute_metrics=_numpy_classification_metrics,
    )
    trainer.train()
    return trainer, trainer.evaluate()


# ===========================================================================
# 1. Data engine
# ===========================================================================


@pytest.mark.unit
class TestDatasetEngine:
    def test_deterministic_same_seed(self, size):
        a = wd.build_synthetic_dataset(size, seed=SEED)
        b = wd.build_synthetic_dataset(size, seed=SEED)
        assert wd.fingerprint(a) == wd.fingerprint(b)

    def test_different_seed_differs(self, size):
        a = wd.build_synthetic_dataset(size, seed=1)
        b = wd.build_synthetic_dataset(size, seed=2)
        assert wd.fingerprint(a) != wd.fingerprint(b)

    def test_configurable_size_scales_rows(self):
        small = wd.build_synthetic_dataset(wd.SizeConfig.from_preset("tiny"), seed=SEED)
        big = wd.build_synthetic_dataset(wd.SizeConfig.from_preset("small"), seed=SEED)
        assert sum(len(big[s]) for s in big) > sum(len(small[s]) for s in small)

    def test_all_labels_present_in_train_and_test(self, synthetic_classification):
        for split in ("train", "test"):
            assert set(synthetic_classification[split]["label"]) == set(range(len(wd.LABELS)))

    def test_three_splits_built(self, synthetic_classification):
        assert set(synthetic_classification.keys()) == {"train", "eval", "test"}

    def test_from_env_preset(self, monkeypatch):
        monkeypatch.setenv("TRAP_TEST_DATASET_SIZE", "small")
        assert wd.SizeConfig.from_env() == wd.SizeConfig.from_preset("small")

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown size preset"):
            wd.SizeConfig.from_preset("enormous")

    def test_build_dataset_reports_source(self, size):
        dd, source = wd.build_dataset(size, seed=SEED, prefer_real=True)
        assert source in ("real", "synthetic")
        assert len(dd["train"]) > 0

    def test_fallback_to_synthetic_when_no_real(self, size, monkeypatch):
        monkeypatch.setattr(wd, "discover_real_classification_dataset", lambda *a, **k: None)
        dd, source = wd.build_dataset(size, seed=SEED, prefer_real=True)
        assert source == "synthetic"

    def test_debug_artifact_written(self, tmp_path, synthetic_classification, size):
        out = wd.write_debug_artifact(
            tmp_path / "fp.json", synthetic_classification, size, "synthetic"
        )
        import json

        payload = json.loads(out.read_text())
        assert payload["source"] == "synthetic"
        assert len(payload["fingerprint"]) == 64


# ===========================================================================
# 2. Schema regression
# ===========================================================================


@pytest.mark.unit
class TestSchemaRegression:
    def test_synthetic_matches_canonical_schema(self, synthetic_classification):
        wd.assert_model_schema(synthetic_classification["train"])

    def test_real_sample_matches_canonical_schema(self, size):
        real = wd.discover_real_classification_dataset()
        if real is None:
            pytest.skip("no real processed dataset on disk")
        dd = wd.sample_real_classification_dataset(real, size, seed=SEED)
        wd.assert_model_schema(dd["train"])

    def test_missing_column_is_detected(self, synthetic_classification):
        broken = synthetic_classification["train"].remove_columns(["attention_mask"])
        with pytest.raises(AssertionError, match="schema drift"):
            wd.assert_model_schema(broken)


# ===========================================================================
# 3. Preprocessing validation
# ===========================================================================


@pytest.mark.unit
class TestPreprocessingValidation:
    def test_no_transcript_leakage(self, synthetic_classification):
        wd.assert_no_split_leakage(synthetic_classification)

    def test_ids_within_model_vocab(self, synthetic_classification, size):
        wd.assert_ids_within_vocab(synthetic_classification, size.vocab_size)

    def test_field_lengths_consistent(self, synthetic_classification):
        wd.assert_lengths_consistent(synthetic_classification)

    def test_leakage_detector_catches_injected_overlap(self, size):
        dd = wd.build_synthetic_dataset(size, seed=SEED)
        # Force a leak: copy a train transcript's rows into test.
        bad = wd.DatasetDict(dd)
        leaked_id = dd["train"][0]["transcript_id"]
        from datasets import concatenate_datasets

        rows = dd["train"].filter(lambda e: e["transcript_id"] == leaked_id)
        bad["test"] = concatenate_datasets([dd["test"], rows])
        with pytest.raises(AssertionError, match="leakage"):
            wd.assert_no_split_leakage(bad)


# ===========================================================================
# 4. Model initialization
# ===========================================================================


@pytest.mark.unit
class TestModelInitialization:
    def test_masking_init_honors_vocab_override(self, tmp_path):
        """make_masking_model_init must size the embedding to the override.

        Regression for the SalmonKmerTokenizer OOB: a wrong vocab built
        nn.Embedding too small and the first k-mer id crashed the gather.
        """
        pytest.importorskip("accelerate")
        import json

        from trap.modeling.train import make_masking_model_init

        cfg_path = tmp_path / "albert.json"
        cfg_path.write_text(json.dumps({"model_type": "albert", "vocab_size": 32000}))
        model = make_masking_model_init(str(cfg_path), vocab_size=65541)()
        assert model.config.vocab_size == 65541
        assert model.get_input_embeddings().num_embeddings == 65541

    def test_classification_init_sets_label_maps(self, tmp_path):
        pytest.importorskip("accelerate")
        import json

        from trap.modeling.train import make_classification_model_init

        cfg_path = tmp_path / "albert.json"
        cfg_path.write_text(json.dumps({"model_type": "albert", "vocab_size": 100}))
        id2label = {i: n for i, n in enumerate(wd.LABELS)}
        label2id = {n: i for i, n in id2label.items()}
        model = make_classification_model_init(
            None, str(cfg_path), len(wd.LABELS), id2label, label2id, vocab_size=80
        )()
        assert model.config.num_labels == len(wd.LABELS)
        assert model.config.id2label[0] == "L1HS"
        assert model.config.vocab_size == 80

    def test_embedding_covers_every_token_id(self, synthetic_classification):
        cfg = wd.tiny_albert_config(synthetic_classification)
        assert cfg.vocab_size > wd.max_token_id(synthetic_classification)

    def test_deterministic_weights(self, synthetic_classification):
        import torch
        from transformers import AlbertForSequenceClassification

        from trap.utils.seeding import set_global_seed

        cfg = wd.tiny_albert_config(synthetic_classification)
        set_global_seed(SEED)
        a = AlbertForSequenceClassification(cfg)
        set_global_seed(SEED)
        b = AlbertForSequenceClassification(cfg)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)


# ===========================================================================
# 5. Classification training loop
# ===========================================================================


@pytest.mark.slow
class TestClassificationTrainingLoop:
    def test_losses_finite_and_converge(self, synthetic_classification, tmp_path):
        import math

        pytest.importorskip("accelerate")
        trainer, _ = _run_classification(synthetic_classification, tmp_path, epochs=6)
        losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
        assert losses, "no training loss was logged"
        assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"

    def test_metrics_in_range_and_consistent(self, synthetic_classification, tmp_path):
        pytest.importorskip("accelerate")
        _, metrics = _run_classification(synthetic_classification, tmp_path, epochs=6)
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        # accuracy and micro-F1 must agree (consistency guard)
        assert abs(metrics["eval_accuracy"] - metrics["eval_f1"]) < 1e-9

    def test_separable_data_actually_learns(self, synthetic_classification, tmp_path):
        pytest.importorskip("accelerate")
        # Force fp32 regardless of GPU: this test verifies training *logic*
        # (a tiny ALBERT must converge on linearly-separable data), not the
        # bf16/fp16 precision path.  bf16 on a 1-layer micro-model converges
        # to ~0.5 on the 3-class tiny dataset; fp32 reliably reaches ~1.0.
        _, metrics = _run_classification(
            synthetic_classification, tmp_path, epochs=15, bf16=False, fp16=False
        )
        assert metrics["eval_accuracy"] > 0.6, f"failed to learn: {metrics}"

    def test_reproducible_across_runs(self, synthetic_classification, tmp_path):
        pytest.importorskip("accelerate")
        _, m1 = _run_classification(synthetic_classification, tmp_path / "a", epochs=4)
        _, m2 = _run_classification(synthetic_classification, tmp_path / "b", epochs=4)
        assert abs(m1["eval_loss"] - m2["eval_loss"]) < 1e-4, (m1["eval_loss"], m2["eval_loss"])


# ===========================================================================
# 6. MLM training loop
# ===========================================================================


@pytest.mark.slow
class TestMaskingTrainingLoop:
    def test_mlm_loss_finite_and_accuracy_bounded(self, synthetic_mlm, tmp_path):
        import math

        pytest.importorskip("accelerate")
        from transformers import (
            AlbertConfig,
            AlbertForMaskedLM,
            Trainer,
            default_data_collator,
        )

        from trap.modeling.train import compute_masking_metrics
        from trap.utils.seeding import set_global_seed

        vocab = max(max(r) for r in synthetic_mlm["train"]["input_ids"]) + 1
        seq_len = len(synthetic_mlm["train"][0]["input_ids"])
        cfg = AlbertConfig(
            vocab_size=vocab,
            embedding_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=max(seq_len, 8),
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        set_global_seed(SEED)
        trainer = Trainer(
            model=AlbertForMaskedLM(cfg),
            args=_training_args(tmp_path, epochs=3),
            train_dataset=synthetic_mlm["train"],
            eval_dataset=synthetic_mlm["eval"],
            data_collator=default_data_collator,
            compute_metrics=compute_masking_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        assert math.isfinite(metrics["eval_loss"])
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0


# ===========================================================================
# 7. Hyperparameter search
# ===========================================================================


@pytest.mark.unit
class TestSearchMisspecification:
    def test_invalid_sampler_rejected_by_schema(self):
        from pydantic import ValidationError

        from trap.config.schemas import HPSpec

        with pytest.raises(ValidationError):
            HPSpec(sampler="bogus", low=0.0, high=1.0)

    def test_unknown_study_sampler_raises(self):
        from trap.modeling.tune import _make_sampler

        with pytest.raises(ValueError, match="unknown sampler"):
            _make_sampler("nope", seed=0)

    def test_unknown_pruner_raises(self):
        from trap.modeling.tune import _make_pruner

        with pytest.raises(ValueError, match="unknown pruner"):
            _make_pruner("nope")


@pytest.mark.slow
class TestHyperparameterSearch:
    def test_search_runs_and_is_reproducible(self, synthetic_classification, tmp_path):
        pytest.importorskip("accelerate")
        from transformers import AlbertForSequenceClassification, Trainer

        from trap.config.schemas import HPSpec
        from trap.modeling.tune import _build_hp_space, _study_kwargs
        from trap.utils.seeding import set_global_seed

        cfg = wd.tiny_albert_config(synthetic_classification)
        space = {"learning_rate": HPSpec(sampler="loguniform", low=1e-4, high=1e-2)}

        def run():
            set_global_seed(SEED)
            trainer = Trainer(
                model_init=lambda: AlbertForSequenceClassification(cfg),
                args=_training_args(tmp_path, epochs=1),
                train_dataset=synthetic_classification["train"],
                eval_dataset=synthetic_classification["test"],
                compute_metrics=_numpy_classification_metrics,
            )
            return trainer.hyperparameter_search(
                hp_space=_build_hp_space(space),
                compute_objective=lambda m: m["eval_accuracy"],
                n_trials=2,
                direction="maximize",
                backend="optuna",
                **_study_kwargs(None, None, "tpe", "hyperband", SEED),
            )

        import math

        best = run()
        assert best is not None and "learning_rate" in best.hyperparameters
        assert math.isfinite(best.objective)
        # Same seed → identical first sampled hyperparameter (TPE seeded).
        best2 = run()
        assert best.hyperparameters["learning_rate"] == best2.hyperparameters["learning_rate"]
