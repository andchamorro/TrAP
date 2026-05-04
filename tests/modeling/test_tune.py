"""Tests for trap.modeling.tune (Optuna hyperparameter search wiring)."""

import json

import pytest


class _RecordingTrial:
    """Stand-in Optuna trial that records which ``suggest_*`` calls fire."""

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, log=False, step=None):
        self.calls.append(("float", name, {"log": log}))
        return low

    def suggest_int(self, name, low, high, step=1, log=False):
        self.calls.append(("int", name, {"step": step, "log": log}))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, {"choices": list(choices)}))
        return choices[0]


class TestBuildHpSpace:
    """``_build_hp_space`` maps each HPSpec sampler to the right suggest call."""

    @pytest.mark.unit
    def test_each_sampler_maps_to_expected_call(self):
        from trap.config.schemas import HPSpec
        from trap.modeling.tune import _build_hp_space

        space = {
            "learning_rate": HPSpec(sampler="loguniform", low=1e-5, high=1e-3),
            "weight_decay": HPSpec(sampler="uniform", low=0.0, high=0.3),
            "n_layers": HPSpec(sampler="int", low=1, high=4, step=1),
            "warmup_ratio": HPSpec(sampler="choice", values=[0.0, 0.1, 0.2]),
            "batch": HPSpec(sampler="categorical", values=[8, 16]),
        }
        trial = _RecordingTrial()
        params = _build_hp_space(space)(trial)

        kinds = {name: (kind, meta) for kind, name, meta in trial.calls}
        assert kinds["learning_rate"] == ("float", {"log": True})
        assert kinds["weight_decay"] == ("float", {"log": False})
        assert kinds["n_layers"][0] == "int"
        assert kinds["warmup_ratio"][0] == "categorical"
        assert kinds["batch"][0] == "categorical"
        # Every searched key ends up in the returned override dict.
        assert set(params) == set(space)


class TestTuneSchema:
    """The shipped tuning YAMLs validate and carry the canonical seed."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name,metric,direction",
        [
            ("classification_optuna.yaml", "eval_f1", "maximize"),
            ("mlm_optuna.yaml", "eval_loss", "minimize"),
        ],
    )
    def test_shipped_configs_load(self, name, metric, direction):
        from trap.config.config import CONFIG_DIR
        from trap.config.schemas import TuneSearchSchema, load_config

        cfg = load_config(CONFIG_DIR / "tuning" / name, TuneSearchSchema)
        assert cfg.seed == 3469
        assert cfg.metric == metric
        assert cfg.direction == direction
        assert cfg.pruner == "hyperband"
        assert cfg.search_space  # non-empty


class TestMakeStudy:
    """``_make_study`` builds an in-memory study with the requested sampler/pruner."""

    @pytest.mark.unit
    def test_in_memory_tpe_hyperband(self):
        from optuna.pruners import HyperbandPruner
        from optuna.samplers import TPESampler

        from trap.modeling.tune import _make_study

        study = _make_study(None, None, "tpe", "hyperband", 3469, "maximize")
        assert isinstance(study.sampler, TPESampler)
        assert isinstance(study.pruner, HyperbandPruner)
        assert study.direction.name == "MAXIMIZE"


class TestFinalize:
    """``finalize`` reads a shared journal study and writes a tuned config + manifest."""

    @pytest.mark.unit
    def test_finalize_writes_tuned_config(self, tmp_path):
        from typer.testing import CliRunner

        from trap.config.config import CONFIG_DIR
        from trap.modeling.tune import _make_study, app

        journal = tmp_path / "study.journal"
        study = _make_study("t_cls", journal, "tpe", "hyperband", 3469, "maximize")

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            trial.suggest_float("weight_decay", 0.0, 0.3)
            return lr

        study.optimize(objective, n_trials=3)

        out = tmp_path / "classification_final.tuned.json"
        result = CliRunner().invoke(
            app,
            [
                "finalize",
                "--search-config",
                str(CONFIG_DIR / "tuning" / "classification_optuna.yaml"),
                "--study-name",
                "t_cls",
                "--storage",
                str(journal),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        merged = json.loads(out.read_text())
        # Best params merged over the base config; base keys preserved.
        assert "learning_rate" in merged and "weight_decay" in merged
        assert merged["seed"] == 3469  # from base classification_final.json
        assert (tmp_path / "manifest.json").exists()


@pytest.mark.slow
class TestEndToEndSearch:
    """A real (tiny) 2-trial search proves the Optuna<->HF Trainer wiring."""

    def test_classification_search_runs(self, tmp_path):
        import datasets
        import numpy as np
        from transformers import (
            AlbertConfig,
            AlbertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )

        from trap.config.schemas import HPSpec
        from trap.modeling.tune import _build_hp_space, _study_kwargs

        rng = np.random.default_rng(0)

        def mk(n):
            return datasets.Dataset.from_dict(
                {
                    "input_ids": [list(rng.integers(5, 25, size=8)) for _ in range(n)],
                    "attention_mask": [[1] * 8 for _ in range(n)],
                    "label": [int(i % 2) for i in range(n)],
                }
            )

        ds = datasets.DatasetDict({"train": mk(16), "test": mk(8)})
        cfg = AlbertConfig(
            vocab_size=30,
            embedding_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            num_labels=2,
        )

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return {"f1": float((preds == p.label_ids).mean())}

        args = TrainingArguments(
            output_dir=str(tmp_path / "hpo"),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            seed=3469,
        )
        trainer = Trainer(
            model_init=lambda: AlbertForSequenceClassification(cfg),
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            compute_metrics=compute_metrics,
        )
        space = {"learning_rate": HPSpec(sampler="loguniform", low=1e-5, high=1e-3)}
        best = trainer.hyperparameter_search(
            hp_space=_build_hp_space(space),
            compute_objective=lambda m: m["eval_f1"],
            n_trials=2,
            direction="maximize",
            backend="optuna",
            **_study_kwargs(None, None, "tpe", "hyperband", 3469),
        )
        assert best is not None
        assert "learning_rate" in best.hyperparameters
