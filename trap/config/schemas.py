"""Typed configuration schemas (DR-4).

Pydantic v2 models for ALBERT, trainer, tokenizer, dataset, and quantify
configs.  Loaders write ``resolved_config.json`` next to every run output.
Supports both JSON and YAML sources via ``load_config`` / ``dump_config``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError("pydantic >=2 is required: pip install pydantic") from exc


# ---------------------------------------------------------------------------
# ALBERT
# ---------------------------------------------------------------------------


class AlbertConfigSchema(BaseModel):
    """ALBERT model hyperparameters (mirrors ``transformers.AlbertConfig``)."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_hidden_groups: int = 1
    inner_group_num: int = 1
    max_position_embeddings: int = 1280
    vocab_size: int = 32000
    pad_token_id: int = 1
    type_vocab_size: int = 2
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    classifier_dropout_prob: float = 0.1
    num_labels: int = 3
    model_type: str = "albert"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainerConfigSchema(BaseModel):
    """HuggingFace ``TrainingArguments`` fields used by TrAP.

    PBT-derived final values for classification:
    ``lr=8.64491338167939e-05``, ``weight_decay=0.17959754525911098``,
    ``warmup_steps`` (replace ``warmup_ratio``), ``per_device_train_batch_size=16``.
    """

    output_dir: str = "./models/output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 8.64491338167939e-05
    weight_decay: float = 0.17959754525911098
    warmup_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    seed: int = 3469
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4
    save_strategy: str = "epoch"
    # transformers >=4.46 renamed evaluation_strategy -> eval_strategy (the old
    # name is removed in 4.57); matches config/trainer_config_base_*.json.
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    report_to: str = "none"
    verbosity: Literal["off", "normal", "detailed"] = "off"


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------


class HPSpec(BaseModel):
    """One hyperparameter's search range, mapped to an Optuna ``suggest_*`` call.

    ``sampler`` selects the Optuna suggestion:

    - ``loguniform`` / ``uniform`` -> ``trial.suggest_float`` (``log=True`` for the
      former); needs ``low`` and ``high``.
    - ``int`` -> ``trial.suggest_int``; needs ``low`` and ``high`` (optional
      ``step``/``log``).
    - ``categorical`` / ``choice`` -> ``trial.suggest_categorical``; needs ``values``.
    """

    sampler: Literal["loguniform", "uniform", "int", "categorical", "choice"]
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    values: Optional[list] = None


class TuneSearchSchema(BaseModel):
    """Optuna hyperparameter-search recipe (``config/tuning/*.yaml``).

    Replaces the retired Ray-Tune ``pbt_search_space.yaml`` for the clean re-do.
    ``base_trainer_config`` is the JSON ``TrainingArguments`` file the winning
    trial's params are merged into (see ``trap.modeling.tune.finalize``).
    """

    metric: str = "eval_f1"
    direction: Literal["minimize", "maximize"] = "maximize"
    n_trials: int = 16
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: Literal["hyperband", "median", "none"] = "hyperband"
    seed: int = 3469
    # Cap epochs per trial (cheap proxy schedule for expensive MLM pretraining).
    max_trial_epochs: Optional[float] = None
    base_trainer_config: str = "config/training/classification_final.json"
    search_space: dict[str, HPSpec] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TokenizerConfigSchema(BaseModel):
    """SentencePiece / WordPiece tokenizer training configuration."""

    algorithm: Literal["unigram", "wordpiece"] = "unigram"
    k: int = 17
    vocab_size: int = 32000
    batch_size: int = 1024
    corpus: Optional[str] = None
    extra_corpus: Optional[str] = None
    out: str = "./models"
    name: str = "tokenizer.gencode.v48.k17.32k"
    max_training_chars: Optional[int] = None
    max_sentence_length: int = 500000
    verbosity: Literal["off", "normal", "detailed"] = "off"


# ---------------------------------------------------------------------------
# ART / STAR sub-schemas
# ---------------------------------------------------------------------------


class ARTConfigSchema(BaseModel):
    """ART Illumina simulation parameters (§7 of reviewer-response export)."""

    coverage: int = 5
    read_length: int = 150
    frag_mean: int = 500
    frag_sd: int = 10
    ss: str = "MSv3"
    paired: bool = True


class STARConfigSchema(BaseModel):
    """STAR alignment parameters (§8 of reviewer-response export)."""

    outFilterMultimapNmax: int = 100
    winAnchorMultimapNmax: int = 100
    version: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------


class DatasetBuildSchema(BaseModel):
    """End-to-end dataset construction parameters."""

    k: int = 17
    max_position_embeddings: int = 1280
    test_split: float = 0.1
    val_split: Optional[float] = None
    class_threshold: Optional[int] = None
    num_proc: int = 32
    # MLM grouping block size; None → defaults to max_position_embeddings.
    # Never set below max_position_embeddings or position embeddings above
    # chunk_size are never trained (see preprocessing_sequences.masking).
    chunk_size: Optional[int] = None
    padding: str = "max_length"
    split_strategy: Literal["read-level", "transcript-level"] = "transcript-level"
    art: ARTConfigSchema = Field(default_factory=ARTConfigSchema)
    star: STARConfigSchema = Field(default_factory=STARConfigSchema)
    verbosity: Literal["off", "normal", "detailed"] = "off"


# ---------------------------------------------------------------------------
# Quantify
# ---------------------------------------------------------------------------


class QuantifyConfigSchema(BaseModel):
    """Parameters for the streaming ``quantify run`` command."""

    batch_size: int = 64
    num_workers: int = 4
    padding: str = "longest"
    pad_to_multiple_of: int = 64
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile_model: bool = True
    checkpoint_every: int = 1000
    seed: int = 3469
    verbosity: Literal["off", "normal", "detailed"] = "off"


# ---------------------------------------------------------------------------
# Entropy / redundancy k-mer analysis
# ---------------------------------------------------------------------------


class KmerSpectrumConfigSchema(BaseModel):
    """Parameters for the entropy/redundancy k-mer spectrum + ablation.

    Drives ``python -m trap.analysis``.  ``corpora`` maps a short label to a
    FASTA/FASTQ path; one spectrum is produced per corpus.  Selection gates
    implement the quantitative k criterion (saturation + marginal gain + task).
    """

    corpora: Dict[str, str] = Field(
        default_factory=lambda: {
            "gencode_v48": "data/external/gencode.v48.transcripts.fa.gz",
            "l1_repeatmasker": "data/external/l1.fa",
        }
    )
    file_format: str = "fasta"
    k_min: int = 2
    k_max: int = 20
    subsample_fracs: List[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    replicates: int = 3
    bootstrap: int = 200
    alpha: float = 0.05
    canonical: bool = False
    chunk_size: int = 2000
    seed: int = 3469

    # Ablation (k -> classification).
    ablation_reads: str = "data/external/l1hs_l1pa2_negative.5x_sample_R1.fq"
    ablation_max_per_class: int = 5000
    ablation_n_features: int = 262144
    ablation_n_splits: int = 5

    # Selection gates.
    saturation_epsilon: float = 0.01
    marginal_gain_tau: float = 0.05

    # Parallelism: worker threads for the k-level loop in kmer_spectrum().
    # null → auto-detect from SLURM_CPUS_PER_TASK, else 1.
    # Pass -1 to use all CPUs reported by os.cpu_count().
    n_jobs: Optional[int] = None

    # Jellyfish initial hash-table size (-s flag).  "1G" handles GENCODE v48 at
    # k≤20 without resize passes; reduce to "500M" on memory-constrained nodes.
    jf_hash_size: str = "1G"

    output_dir: str = "results/entropy_redundancy"
    verbosity: Literal["off", "normal", "detailed"] = "off"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_config(path: Path, schema_cls):
    """Load a JSON or YAML config file into a Pydantic schema.

    Args:
        path: ``.json`` or ``.yaml``/``.yml`` file path.
        schema_cls: Pydantic model class to validate against.

    Returns:
        Validated schema instance.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    return schema_cls(**data)


def dump_config(obj, path: Path) -> None:
    """Write a Pydantic schema to JSON.

    Args:
        obj: Pydantic model instance.
        path: Destination ``.json`` path (parent directories created).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(obj.model_dump_json(indent=2))
