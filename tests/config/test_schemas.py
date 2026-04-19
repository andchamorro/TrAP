"""Tests for trap.config.schemas (DR-4)."""
import json

import pytest

from trap.config.schemas import (
    AlbertConfigSchema,
    ARTConfigSchema,
    DatasetBuildSchema,
    QuantifyConfigSchema,
    STARConfigSchema,
    TokenizerConfigSchema,
    TrainerConfigSchema,
    dump_config,
    load_config,
)


@pytest.mark.unit
class TestAlbertConfigSchema:
    def test_defaults(self):
        cfg = AlbertConfigSchema()
        assert cfg.max_position_embeddings == 1280
        assert cfg.vocab_size == 32000
        assert cfg.num_labels == 3

    def test_override(self):
        cfg = AlbertConfigSchema(max_position_embeddings=512, vocab_size=10000)
        assert cfg.max_position_embeddings == 512


@pytest.mark.unit
class TestTrainerConfigSchema:
    def test_pbt_defaults(self):
        cfg = TrainerConfigSchema()
        assert abs(cfg.learning_rate - 8.64491338167939e-05) < 1e-12
        assert abs(cfg.weight_decay - 0.17959754525911098) < 1e-12
        assert cfg.seed == 3469

    def test_bf16_default_false(self):
        assert TrainerConfigSchema().bf16 is False

    def test_uses_new_eval_strategy_name(self):
        # transformers >=4.46 renamed evaluation_strategy -> eval_strategy.
        dumped = TrainerConfigSchema().model_dump()
        assert dumped["eval_strategy"] == "epoch"
        assert "evaluation_strategy" not in dumped


@pytest.mark.unit
class TestQuantifyConfigSchema:
    def test_defaults(self):
        cfg = QuantifyConfigSchema()
        assert cfg.batch_size == 64
        assert cfg.dtype == "bfloat16"
        assert cfg.checkpoint_every == 1000


@pytest.mark.unit
class TestDatasetBuildSchema:
    def test_transcript_level_default(self):
        cfg = DatasetBuildSchema()
        assert cfg.split_strategy == "transcript-level"
        assert cfg.k == 17

    def test_nested_art(self):
        cfg = DatasetBuildSchema()
        assert cfg.art.coverage == 5
        assert cfg.art.frag_mean == 500

    def test_nested_star(self):
        cfg = DatasetBuildSchema()
        assert cfg.star.outFilterMultimapNmax == 100


@pytest.mark.unit
class TestLoadDump:
    def test_round_trip_json(self, tmp_path):
        cfg = AlbertConfigSchema(hidden_size=128, num_labels=2)
        path = tmp_path / "cfg.json"
        dump_config(cfg, path)
        loaded = load_config(path, AlbertConfigSchema)
        assert loaded.hidden_size == 128
        assert loaded.num_labels == 2

    def test_load_creates_parent_dirs(self, tmp_path):
        cfg = QuantifyConfigSchema(batch_size=32)
        path = tmp_path / "deep" / "nested" / "cfg.json"
        dump_config(cfg, path)
        assert path.exists()
