import json
import os
from pathlib import Path
from typing import Optional, Union

from accelerate.test_utils.testing import get_backend
from datasets import load_from_disk
import evaluate
from loguru import logger
import numpy as np
from si_prefix import si_format
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
from transformers import (
    TrainingArguments,
    default_data_collator,
)
from transformers import Trainer as Trainer, TrainerCallback
import typer

# from trap.modeling.albert import AlbertConfig, AlbertForMaskedLM, AlbertModel
from trap.config import manifest as manifest_mod
from trap.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from trap.config.verbosity import set_verbosity
from trap.loaders.tokenizer import WholeKmerMaskingDataCollator, load_kmer_tokenizer
from trap.utils.io import try_mkdir
from trap.utils.seeding import set_global_seed

app = typer.Typer()


def _read_config_dict(path: Path) -> dict:
    """Load a trainer config from JSON or YAML into a plain dict.

    Returns the raw dict (not a pydantic schema) so every HuggingFace
    ``TrainingArguments`` field present in the file is preserved.

    Args:
        path: ``.json`` or ``.yaml``/``.yml`` config file.

    Returns:
        Parsed config dict.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_precision(cfg: dict) -> dict:
    """Downgrade bf16→fp16 when the allocated GPU doesn't support bfloat16.

    TrainingArguments raises ValueError at construction if bf16=True on a
    pre-Ampere GPU (e.g. V100/Turing). This detects support at runtime and
    falls back so the stage doesn't crash when --gres lands on the wrong node.

    Args:
        cfg: Raw ``TrainingArguments`` keyword dict (mutated copy returned).

    Returns:
        Updated config dict with ``bf16``/``fp16`` set to a supported value.
    """
    if not cfg.get("bf16"):
        return cfg
    try:
        from transformers.utils import is_torch_bf16_gpu_available

        supported = is_torch_bf16_gpu_available()
    except ImportError:
        supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if supported:
        return cfg
    cfg = dict(cfg)
    cfg["bf16"] = False
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        logger.warning(
            f"bf16 not supported on this GPU (compute capability {cap[0]}.{cap[1]}); "
            "falling back to fp16. Request gpu:a100 via --gres to use bf16."
        )
        cfg["fp16"] = True
    else:
        logger.warning("bf16 requested but no Ampere+ GPU detected; training in fp32.")
    return cfg


def load_tuned_tokenizer(tokenizer_path, max_position: int) -> PreTrainedTokenizerFast:
    """Load a fast tokenizer and bake the paired ``[CLS]/[SEP]`` template.

    Shared by ``masking``/``classification``/``distiller`` and ``trap.modeling.tune``
    so the post-processor and ``model_max_length`` are set in exactly one place.

    Args:
        tokenizer_path: Directory of the saved fast tokenizer.
        max_position: Model ``max_position_embeddings`` (sets ``model_max_length``).

    Returns:
        Configured ``PreTrainedTokenizerFast``.
    """
    return load_kmer_tokenizer(tokenizer_path, max_position)


def make_compute_classification_metrics():
    """Return a ``compute_metrics`` callable (accuracy + macro f1/precision/recall/roc_auc).

    F1/precision/recall are **macro**-averaged (unweighted mean over classes) so
    the minority retroelement classes (L1HS ~0.3%, L1PA ~10% of the train split)
    count as much as the NEGATIVE majority (~89%). Micro-F1 equals accuracy for
    single-label multiclass, so selecting on it (``metric_for_best_model="f1"``)
    would reward a NEGATIVE-collapsed classifier; ``accuracy`` below still reports
    that overall/micro view. Matches the macro-F1 acceptance target (plan §7.4).
    """
    from scipy.special import softmax as scipy_softmax
    from sklearn.metrics import roc_auc_score

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    logger.info("Using accuracy, f1, precision, recall, and roc_auc as classification scores")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        probs = scipy_softmax(logits, axis=1)
        num_classes = probs.shape[1]
        try:
            if num_classes == 2:
                roc_auc = roc_auc_score(labels, probs[:, 1])
            else:
                roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = float("nan")
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
            "precision": precision.compute(
                predictions=predictions, references=labels, average="macro", zero_division=0
            )["precision"],
            "recall": recall.compute(predictions=predictions, references=labels, average="macro")[
                "recall"
            ],
            "roc_auc": roc_auc,
        }

    return compute_metrics


def load_classification_data(preprocessing_name: str):
    """Load the tokenized classification dataset and derive its label maps.

    Returns:
        ``(lm_datasets, label2id, id2label)`` where ``lm_datasets`` is the
        ``DatasetDict`` saved by preprocessing under ``classification/tokenized``.
    """
    lm_datasets = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "classification", "tokenized")
    )
    # transcript_id is kept on disk for the stage-21 leakage check but is a string
    # column the padding collator cannot tensorise (remove_unused_columns=False),
    # so drop it (and any read-id columns) before training/tuning consume it.
    drop = [
        c for c in ("transcript_id", "id", "read_id") if c in lm_datasets["train"].column_names
    ]
    if drop:
        lm_datasets = lm_datasets.remove_columns(drop)
    class_labels = lm_datasets["train"].features["label"]
    label2id = {label: class_labels.str2int(label) for label in class_labels.names}
    id2label = {class_labels.str2int(label): label for label in class_labels.names}
    return lm_datasets, label2id, id2label


def assert_dataset_ids_in_range(lm_datasets, *, vocab_size, num_labels=None, dataset_name=""):
    """Fail fast (synchronously, with a clear message) on an out-of-range id.

    A token id ``>= vocab_size`` (or a label ``>= num_labels``) triggers an
    *asynchronous* CUDA gather assertion deep in the embedding/loss lookup
    (``vectorized gather kernel index out of bounds``). Under DDP that surfaces
    only as a 30-minute NCCL watchdog timeout with the true cause invisible —
    and only after burning hours of compute. Checking the Arrow columns up front
    (cheap: ``pyarrow.compute`` over the memory-mapped table, no Python
    materialisation) turns that into an immediate error naming the offending id,
    the model vocab, and the split, so the usual culprit — a dataset tokenized
    with a different vocab than the model is sized from (e.g. mismatched
    ``n_hash``/``num_target`` in ``salmon_kmer_config.json``) — is obvious.

    Args:
        lm_datasets: ``DatasetDict`` (or split→Dataset mapping) to validate.
        vocab_size: Model input-embedding size; valid ids are ``[0, vocab_size)``.
        num_labels: When set, also require ``label`` in ``[0, num_labels)``.
        dataset_name: Label used in messages for context.
    """
    import pyarrow.compute as pc

    name = dataset_name or "dataset"
    for split, ds in lm_datasets.items():
        if "input_ids" in ds.column_names:
            try:
                flat = pc.list_flatten(ds.data.column("input_ids"))
                max_id = pc.max(flat).as_py()
                min_id = pc.min(flat).as_py()
            except Exception as exc:  # pragma: no cover - pyarrow version drift
                logger.warning(f"[id-check] {name}:{split}: input_id scan skipped ({exc})")
            else:
                if max_id is not None and max_id >= vocab_size:
                    raise ValueError(
                        f"[id-check] {name} split {split!r}: max input_id={max_id} >= "
                        f"model vocab_size={vocab_size}. The tokenized dataset and the "
                        "model embedding disagree — rebuild the dataset with the SAME "
                        "tokenizer the model is sized from (verify n_hash / num_target in "
                        "salmon_kmer_config.json). Left unchecked this fails mid-training "
                        "as a CUDA gather OOB / NCCL timeout."
                    )
                if min_id is not None and min_id < 0:
                    raise ValueError(
                        f"[id-check] {name} split {split!r}: negative input_id={min_id} "
                        "(non-ACGT bases not stripped before tokenisation?)."
                    )
        if num_labels is not None and "label" in ds.column_names:
            try:
                max_lbl = pc.max(ds.data.column("label")).as_py()
            except Exception as exc:  # pragma: no cover - pyarrow version drift
                logger.warning(f"[id-check] {name}:{split}: label scan skipped ({exc})")
            else:
                if max_lbl is not None and max_lbl >= num_labels:
                    raise ValueError(
                        f"[id-check] {name} split {split!r}: max label={max_lbl} >= "
                        f"num_labels={num_labels}."
                    )
    logger.info(f"[id-check] {name}: token ids within [0, {vocab_size}); labels in range.")


def make_classification_model_init(
    pretrained_model_path,
    albert_config_path,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    vocab_size: Optional[int] = None,
):
    """Build a fresh-model factory for ``AlbertForSequenceClassification``.

    Returns a zero-arg ``model_init`` so the same builder serves both a single
    training run and ``Trainer.hyperparameter_search`` (which re-inits the model
    per trial). Loads from the pretrained MLM checkpoint when present, else from
    the ALBERT JSON config.
    """

    def model_init():
        final = MODELS_DIR / pretrained_model_path / "final" if pretrained_model_path else None
        if final is not None and Path(final).exists():
            logger.info("Set model from pretrained ...")
            return AlbertForSequenceClassification.from_pretrained(
                str(final), num_labels=num_labels, id2label=id2label, label2id=label2id
            )
        logger.info("Set model from config ...")
        albert_config = AlbertConfig.from_json_file(albert_config_path)
        albert_config.num_labels = num_labels
        albert_config.id2label = id2label
        albert_config.label2id = label2id
        if vocab_size is not None:
            albert_config.vocab_size = vocab_size
        return AlbertForSequenceClassification(albert_config)

    return model_init


def make_masking_model_init(albert_config_path, vocab_size: Optional[int] = None):
    """Build a fresh-model factory for ``AlbertForMaskedLM`` from an ALBERT config.

    Args:
        albert_config_path: Path to the ALBERT JSON config.
        vocab_size: Override ``vocab_size`` in the config (required when the
            tokenizer vocab differs from the JSON default, e.g. SalmonKmerTokenizer).
    """

    def model_init():
        albert_config = AlbertConfig.from_json_file(albert_config_path)
        if vocab_size is not None:
            albert_config.vocab_size = vocab_size
        return AlbertForMaskedLM(albert_config)

    return model_init


def load_masking_data(preprocessing_name: str, tokenizer, num_workers: int, debug: bool = False):
    """Load the chunked MLM dataset and build its masked ``eval`` split.

    Returns:
        ``(lm_datasets, data_collator)`` ready for ``MaskingTrainer``; the
        ``eval`` split is pre-masked once via ``insert_random_mask`` for a
        deterministic evaluation loss.
    """
    lm_datasets = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "masking", "grouped")
    )
    if debug:
        logger.debug("Downsampling pretokenized dataset")
        train_size = 1_000
        test_size = int(0.1 * train_size)
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )

    data_collator = WholeKmerMaskingDataCollator(tokenizer)

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    lm_datasets["eval"] = lm_datasets["test"].map(
        insert_random_mask,
        batched=True,
        num_proc=num_workers,
        remove_columns=lm_datasets["test"].column_names,
        keep_in_memory=True,  # avoids Lustre cache-file race when concurrent workers share the dataset path
    )
    lm_datasets["eval"] = lm_datasets["eval"].rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_token_type_ids": "token_type_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    return lm_datasets, data_collator


debug_mode = False


def debug_callback(debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")):
    """
    Callback function to handle the debug flag.
    """
    global debug_mode
    if debug:
        typer.echo("Debug mode enabled")
        debug_mode = True


class _PerplexityLogCallback(TrainerCallback):
    """Append ``*_perplexity = exp(*_loss)`` for every evaluation loss key logged.

    Fires on every ``on_log`` event (periodic eval, final-train eval, etc.).
    Skips ``train_loss`` because that is a running mean from step 0 and is not
    comparable to checkpoint-evaluated losses.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for key, value in list(logs.items()):
            if key.endswith("_loss") and key != "train_loss" and isinstance(value, float):
                logs[key.replace("_loss", "_perplexity")] = float(np.exp(value))


def compute_masking_metrics(eval_pred) -> dict:
    """Compute masked-token accuracy for MLM evaluation.

    Provides the training-comparable metric that ``train_result.metrics``
    cannot supply: ``train_result.metrics["train_loss"]`` is the running mean
    from step 0 (includes early bad epochs) while ``eval_loss`` comes from the
    final checkpoint — they are not directly comparable.  By passing this
    function as ``compute_metrics`` to ``MaskingTrainer``, each evaluation call
    (including the post-training ``evaluate`` on a training sample) reports
    per-token accuracy from the *same* checkpoint.  Perplexity is derived from
    ``eval_loss`` by ``_PerplexityLogCallback`` (``exp(eval_loss)``).

    Labels are -100 for non-masked positions (ignored) and the true token id
    for masked positions, matching ``DataCollatorForLanguageModeling`` / the
    TrAP ``WholeKmerMaskingDataCollator`` convention.

    Accepts either raw logits ``[batch, seq_len, vocab_size]`` (no preprocessing
    hook) or already-reduced predicted token ids ``[batch, seq_len]`` (when
    ``mlm_preprocess_logits_for_metrics`` is wired into the trainer). The vocab
    axis is collapsed here only if it is still present.
    """
    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    # With the preprocess hook, predictions already match labels' shape; without
    # it the trainer hands us raw logits with a trailing vocab axis to argmax.
    if predictions.ndim == labels.ndim + 1:
        predictions = np.argmax(predictions, axis=-1)
    mask = labels != -100
    n_masked = mask.sum()
    if n_masked == 0:
        return {"accuracy": 0.0}
    accuracy = float((predictions[mask] == labels[mask]).sum() / n_masked)
    return {"accuracy": accuracy}


def mlm_preprocess_logits_for_metrics(logits, labels):
    """Reduce MLM logits to predicted token ids on-device, before accumulation.

    The HuggingFace ``Trainer`` concatenates every eval batch's model output into
    one tensor before handing it to ``compute_metrics``. Raw MLM logits are
    ``[batch, seq_len, vocab_size]``; with the salmon k-mer vocab (~65k) and 1280
    positions, accumulating them across the eval set needs tens of GiB and OOMs
    the GPU inside ``evaluation_loop``'s ``torch.cat`` (observed: a 20 GiB
    allocation on a 40 GB A100 at the first epoch's eval). Taking the argmax over
    the vocab axis here collapses each batch to ``[batch, seq_len]`` token ids — a
    ``vocab_size``-fold memory reduction — which is all ``compute_masking_metrics``
    needs (it only compares predicted vs. true ids). The HPO path in ``tune.py``
    instead sets ``prediction_loss_only`` because it only tracks ``eval_loss``;
    final training keeps the accuracy metric, hence this hook.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)


class MaskingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": default_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)


def compute_class_weights(labels, num_labels: int, scheme: str = "balanced") -> np.ndarray:
    """Per-class loss weights to counter class imbalance.

    ``"balanced"`` reproduces sklearn's heuristic ``w_c = N / (num_labels *
    count_c)``: rarer classes get proportionally larger weights, and the weighted
    average weight is 1, so the loss scale (and the smoke gate's ``ln(num_labels)``
    baseline) is preserved. Classes absent from ``labels`` get weight 0 (no
    samples ever back-propagate through them).

    Args:
        labels: Integer label per training example.
        num_labels: Number of classes (length of the returned vector).
        scheme: Currently only ``"balanced"``.

    Returns:
        ``float32`` array of length ``num_labels``.
    """
    if scheme != "balanced":
        raise ValueError(f"Unknown class_weighting scheme {scheme!r} (expected 'balanced').")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_labels).astype(
        np.float64
    )
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float64)
    nz = counts > 0
    weights[nz] = total / (num_labels * counts[nz])
    return weights.astype(np.float32)


class WeightedLossTrainer(Trainer):
    """``Trainer`` with class-weighted cross-entropy for sequence classification.

    Identical to the default loss except for the per-class ``weight`` vector, so
    the severe L1 imbalance (NEGATIVE ~89% vs L1HS ~0.3%) does not collapse the
    classifier onto the majority class. Pass ``class_weights`` (computed once from
    the train-split frequencies via :func:`compute_class_weights`); ``None``
    falls back to unweighted loss.

    Args:
        class_weights: Length-``num_labels`` weight vector, or ``None``.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), weight=weight)
        return (loss, outputs) if return_outputs else loss


class DistillationTrainer(Trainer):
    """Trainer to compress a SetFit model with knowledge distillation.

    Args:
        teacher_model:
            The teacher model to mimic.
        student_model:
            The model to train. If not provided, a `model_init` must be passed.
        args (`TrainingArguments`, *optional*):
            The training arguments to use.
    """

    _REQUIRED_COLUMNS = {"text"}

    def __init__(
        self,
        teacher_model=None,
        student_model=None,
        temperature=None,
        lambda_param=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device, _, _ = (
            get_backend()
        )  # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        student_output = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = student_output[self.args.past_index]

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature**2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        return (loss, student_output) if return_outputs else loss


@app.command()
def masking(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(
        default=..., help="Path to the pretrained tokenizer"
    ),
    trainer_config_path: Path = typer.Option(
        CONFIG_DIR / "trainer_config_base_uncased.json", help="Path to the trainer config"
    ),
    albert_config_path: Path = typer.Option(
        CONFIG_DIR / "albert_config_base_uncased.json", help="Path to the Albert config"
    ),
    builder: str = typer.Option(None, help="Path to the genome dataset"),
    k: int = typer.Option(18, help="K-mer size"),
    test_split: float = typer.Option(0.1, help="Test split ratio"),
    chunk_size: int = typer.Option(128, help="Chunk size for grouping texts"),
    preprocessing_name: str = typer.Option(
        "gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"
    ),
    num_workers: int = typer.Option(16, help="Number of workers"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
    # -----------------------------------------
):
    import time as _time

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[train:masking] model={model_name!r} preprocessing={preprocessing_name!r}",
    )
    debug_callback(debug)
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")

    logger.info("Training ALBERT model...")
    _trainer_cfg = _resolve_precision(_read_config_dict(trainer_config_path))
    set_global_seed(_trainer_cfg.get("seed", 3469))
    trainer_args = TrainingArguments(**_trainer_cfg)
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers
    albert_config = AlbertConfig.from_json_file(albert_config_path)
    logger.info("Loading pretrained tokenizer")
    tokenizer = load_tuned_tokenizer(
        pretrained_tokenizer_path, albert_config.max_position_embeddings
    )
    # The embedding table must match the tokenizer vocabulary. Salmon k-mer
    # tokenizers size their vocab from the k-mer hash/index, not the static JSON.
    albert_config.vocab_size = len(tokenizer)
    model = AlbertForMaskedLM(albert_config)

    model_num_parameters = model.num_parameters() / 1_000_000
    logger.log("STAGE", f"[train:masking] model parameters: {round(model_num_parameters)}M")
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")
    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    # preprocessing_sequences writes the chunked MLM dataset under masking/grouped
    lm_datasets, data_collator = load_masking_data(
        preprocessing_name, tokenizer, num_workers, debug=debug_mode
    )
    assert_dataset_ids_in_range(
        lm_datasets, vocab_size=model.config.vocab_size, dataset_name=preprocessing_name
    )
    logger.log(
        "STAGE",
        f"[train:masking] dataset — train={len(lm_datasets['train']):,} "
        f"eval={len(lm_datasets['eval']):,}",
    )

    logger.info("Preparing your data for training")

    device, _, _ = get_backend()
    logger.log("STAGE", f"[train:masking] device={device}")
    logger.info(f"'Training in device {device}'")

    try_mkdir(trainer_args.output_dir)
    trainer = MaskingTrainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_masking_metrics,
        preprocess_logits_for_metrics=mlm_preprocess_logits_for_metrics,
        callbacks=[_PerplexityLogCallback()],
    )
    logger.log("STAGE", "[train:masking] training started")
    train_result = trainer.train()
    logger.success("Modeling training complete.")

    # train_result.metrics["train_loss"] is the running mean from step 0 and
    # is NOT comparable to eval_loss (which is from the final checkpoint).
    # Evaluate the final checkpoint on a fixed-masked training sample so that
    # train / val / test perplexity are all measured the same way.
    _FINAL_TRAIN_EVAL_SIZE = 2_000
    train_eval_size = min(_FINAL_TRAIN_EVAL_SIZE, len(lm_datasets["train"]))
    train_eval_ds = lm_datasets["train"].select(range(train_eval_size))

    def _insert_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked.items()}

    train_eval_ds = train_eval_ds.map(
        _insert_mask, batched=True, remove_columns=train_eval_ds.column_names
    )
    train_eval_ds = train_eval_ds.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_token_type_ids": "token_type_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    final_train_metrics = trainer.evaluate(
        eval_dataset=train_eval_ds, metric_key_prefix="final_train"
    )
    final_train_metrics["final_train_samples"] = train_eval_size
    if "final_train_loss" in final_train_metrics:
        final_train_metrics["final_train_perplexity"] = float(
            np.exp(final_train_metrics["final_train_loss"])
        )
    trainer.log_metrics("final_train", final_train_metrics)
    trainer.save_metrics("final_train", final_train_metrics)
    _ftl = final_train_metrics.get("final_train_loss", "?")
    _ftp = final_train_metrics.get("final_train_perplexity", "?")
    _fta = final_train_metrics.get("final_train_accuracy", "?")
    logger.info(
        f"Final-checkpoint train metrics (n={train_eval_size}): "
        f"loss={_ftl:.4f}  perplexity={_ftp:.2f}  accuracy={_fta:.4f}"
        if isinstance(_ftl, float) and isinstance(_ftp, float) and isinstance(_fta, float)
        else f"Final-checkpoint train metrics (n={train_eval_size}): "
        f"loss={_ftl}  perplexity={_ftp}  accuracy={_fta}"
    )

    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_datasets["train"])
    logger.success("Saving model.")
    trainer.save_model(
        os.path.join(MODELS_DIR, model_name, "final")
    )  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        manifest_mod.write(
            os.path.join(MODELS_DIR, model_name, "final"),
            seed=getattr(trainer_args, "seed", 3469),
            k=k,
            model={"name": model_name, "objective": "mlm"},
            throughput={"train_runtime_s": metrics.get("train_runtime")},
        )
    _elapsed = _time.perf_counter() - _t0
    _loss = metrics.get("train_loss", "?")
    _loss_str = f"{_loss:.4f}" if isinstance(_loss, float) else str(_loss)
    logger.log(
        "STAGE",
        f"[train:masking] done — loss={_loss_str} elapsed={_elapsed:.0f} s → "
        f"{MODELS_DIR}/{model_name}/final/",
    )
    logger.success("Train model done.")
    # -----------------------------------------
    pass


@app.command()
def classification(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(
        default=None, help="Path to the pretrained tokenizer"
    ),
    pretrained_model_path: Path = typer.Option(default=None, help="Path to the pretrained model"),
    trainer_config_path: Path = typer.Option(
        CONFIG_DIR / "trainer_config_base_repeatmasker.json", help="Path to the trainer config"
    ),
    albert_config_path: Path = typer.Option(
        CONFIG_DIR / "albert_config_base_uncased.json", help="Path to the Albert config"
    ),
    builder: str = typer.Option(None, help="Path to the genome dataset"),
    k: int = typer.Option(18, help="K-mer size"),
    test_split: float = typer.Option(0.1, help="Test split ratio"),
    chunk_size: int = typer.Option(128, help="Chunk size for grouping texts"),
    preprocessing_name: str = typer.Option(
        "gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"
    ),
    num_workers: int = typer.Option(16, help="Number of workers"),
    do_eval: bool = typer.Option(False, help="Enable evaluation mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
    # -----------------------------------------
):
    import time as _time

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[train:classification] model={model_name!r} preprocessing={preprocessing_name!r}",
    )
    debug_callback(debug)
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")

    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    lm_datasets, label2id, id2label = load_classification_data(preprocessing_name)
    num_labels = len(id2label)
    if debug_mode:
        logger.debug("Downsampling pretokenized dataset")
        train_size = 1_000
        test_size = int(0.1 * train_size)
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        if do_eval:
            logger.debug("Downsampling pretokenized eval dataset")
            eval_size = int(0.1 * train_size)
            train_eval_split = lm_datasets["train"].train_test_split(test_size=eval_size, seed=42)
            lm_datasets["eval"] = train_eval_split["test"]

    logger.log(
        "STAGE",
        f"[train:classification] dataset — train={len(lm_datasets['train']):,} "
        f"test={len(lm_datasets['test']):,} labels={list(id2label.values())}",
    )
    logger.info("Loading pretrained tokenizer")
    # Prefer the tokenizer saved alongside the pretrained model checkpoint.
    final_dir = MODELS_DIR / pretrained_model_path / "final" if pretrained_model_path else None
    tokenizer_src = (
        str(final_dir)
        if final_dir is not None and Path(final_dir).exists()
        else pretrained_tokenizer_path
    )
    _max_pos = AlbertConfig.from_json_file(albert_config_path).max_position_embeddings
    tokenizer = load_tuned_tokenizer(tokenizer_src, _max_pos)

    logger.info("Set training model...")
    # vocab_size is used only by the from-config branch; the from-pretrained
    # branch inherits the (matching) vocab from the MLM checkpoint.
    model = make_classification_model_init(
        pretrained_model_path,
        albert_config_path,
        num_labels,
        id2label,
        label2id,
        vocab_size=len(tokenizer),
    )()

    model_num_parameters = model.num_parameters() / 1_000_000
    logger.log(
        "STAGE",
        f"[train:classification] model parameters: {round(model_num_parameters)}M",
    )
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    # The embedding table is sized from len(tokenizer); a mismatch means token
    # ids emitted by the tokenizer can index past the table → CUDA gather OOB.
    # The from-pretrained branch inherits the MLM checkpoint's vocab, which must
    # also match the tokenizer that produced the dataset.
    n_emb = model.get_input_embeddings().num_embeddings
    if n_emb != len(tokenizer):
        raise ValueError(
            f"Embedding table ({n_emb}) != len(tokenizer) ({len(tokenizer)}). "
            "The classifier's vocab must match the tokenizer that produced the "
            "tokenized dataset; check albert_config vocab_size / the pretrained "
            "checkpoint."
        )
    logger.info(f"[vocab-check] embedding table matches tokenizer: {n_emb} == {len(tokenizer)}")

    assert_dataset_ids_in_range(
        lm_datasets,
        vocab_size=model.config.vocab_size,
        num_labels=num_labels,
        dataset_name=preprocessing_name,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    _trainer_cfg = _resolve_precision(_read_config_dict(trainer_config_path))
    # class_weighting is a TrAP-only key, not a TrainingArguments field; pop it
    # before constructing TrainingArguments (which rejects unknown kwargs).
    class_weighting = _trainer_cfg.pop("class_weighting", None)
    set_global_seed(_trainer_cfg.get("seed", 3469))
    trainer_args = TrainingArguments(**_trainer_cfg)
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers

    logger.info("Preparing your data for training")

    device, _, _ = get_backend()
    logger.log("STAGE", f"[train:classification] device={device}")
    logger.info(f"'Training in device {device}'")

    compute_metrics = make_compute_classification_metrics()

    # Class-weighted loss to counter the NEGATIVE-dominated split (plan §6/§7.4).
    # WeightedLossTrainer with class_weights=None is the unweighted default.
    class_weights = None
    if class_weighting:
        class_weights = compute_class_weights(
            lm_datasets["train"]["label"], num_labels, scheme=class_weighting
        )
        logger.log(
            "STAGE",
            f"[train:classification] class_weighting={class_weighting!r} weights="
            + ", ".join(f"{id2label[i]}={w:.3g}" for i, w in enumerate(class_weights)),
        )

    try_mkdir(trainer_args.output_dir)
    trainer = WeightedLossTrainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    logger.log("STAGE", "[train:classification] training started")
    train_result = trainer.train()
    logger.success("Modeling training complete.")
    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_datasets["train"])
    logger.success("Saving model.")
    trainer.save_model(
        os.path.join(MODELS_DIR, model_name, "final")
    )  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        manifest_mod.write(
            os.path.join(MODELS_DIR, model_name, "final"),
            seed=getattr(trainer_args, "seed", 3469),
            k=k,
            model={
                "name": model_name,
                "objective": "classification",
                # Effective vocab actually used (override of the albert_config
                # default); records the real 65k salmon vocab, not the JSON label.
                "vocab_size": int(model.config.vocab_size),
                "tokenizer_len": len(tokenizer),
                "pretrained_from": str(pretrained_model_path) if pretrained_model_path else None,
                "class_weighting": class_weighting,
            },
            throughput={"train_runtime_s": metrics.get("train_runtime")},
        )
    _elapsed = _time.perf_counter() - _t0
    _f1 = metrics.get("train_f1", metrics.get("eval_f1", "?"))
    _f1_str = f"{_f1:.4f}" if isinstance(_f1, float) else str(_f1)
    logger.log(
        "STAGE",
        f"[train:classification] done — f1={_f1_str} elapsed={_elapsed:.0f} s → "
        f"{MODELS_DIR}/{model_name}/final/",
    )
    logger.success("Train model done.")
    # Evaluation
    if do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=lm_datasets["eval"])
        metrics["eval_samples"] = len(lm_datasets["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    # -----------------------------------------
    pass


@app.command()
def distiller(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_model_name: Path = typer.Option(
        default=None, help="Path to the pretrained teacher model"
    ),
    pretrained_student_path: Path = typer.Option(
        default=None, help="Path to the pretrained student model"
    ),
    trainer_config_path: Path = typer.Option(
        CONFIG_DIR / "trainer_config_base_repeatmasker.json", help="Path to the trainer config"
    ),
    student_config_path: Path = typer.Option(
        CONFIG_DIR / "little_albert_config_base_uncased.json", help="Path to the Albert config"
    ),
    preprocessing_name: str = typer.Option(
        "gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"
    ),
    num_workers: int = typer.Option(16, help="Number of workers"),
    do_eval: bool = typer.Option(False, help="Enable evaluation mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    verbosity: str = typer.Option(
        "off",
        "--verbosity",
        help="Log verbosity: off (default), normal, detailed.",
        envvar="TRAP_VERBOSITY",
    ),
    # -----------------------------------------
):
    """Distil a classification model via knowledge distillation."""
    import time as _time

    set_verbosity(verbosity)
    _t0 = _time.perf_counter()
    logger.log(
        "STAGE",
        f"[train:distiller] model={model_name!r} teacher={pretrained_model_name!r}",
    )
    debug_callback(debug)
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")

    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    lm_datasets = load_from_disk(
        os.path.join(PROCESSED_DATA_DIR, preprocessing_name, "classification", "tokenized")
    )
    if debug_mode:
        logger.debug("Downsampling pretokenized dataset")
        train_size = 1_000
        test_size = int(0.1 * train_size)
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        if do_eval:
            logger.debug("Downsampling pretokenized eval dataset")
            eval_size = int(0.1 * train_size)
            train_eval_split = lm_datasets["train"].train_test_split(test_size=eval_size, seed=42)
            lm_datasets["eval"] = train_eval_split["test"]
            pass

    logger.info("Set training model...")
    # Extract the ClassLabel feature
    class_labels = lm_datasets["train"].features["label"]
    # Create label2id and id2label mappings
    label2id = {label: class_labels.str2int(label) for label in class_labels.names}
    id2label = {class_labels.str2int(label): label for label in class_labels.names}

    logger.info("Set model from pretrained ...")
    teacher_model = AlbertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, "final")
    )

    if (
        pretrained_student_path is not None
        and os.path.join(MODELS_DIR, pretrained_student_path, "final").exists()
    ):
        logger.info("Set model from pretrained ...")
        student_model = AlbertForSequenceClassification.from_pretrained(
            os.path.join(MODELS_DIR, pretrained_student_path, "final"),
            num_labels=len(class_labels.names),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        logger.info("Set model from config ...")
        albert_config = AlbertConfig.from_json_file(student_config_path)
        albert_config.num_labels = len(class_labels.names)
        albert_config.id2label = id2label
        albert_config.label2id = label2id
        student_model = AlbertForSequenceClassification(albert_config)
        pass

    teacher_model_num_parameters = si_format(
        teacher_model.num_parameters(), precision=0, format_str="{value}{prefix}"
    ).upper()
    student_model_num_parameters = si_format(
        student_model.num_parameters(), precision=0, format_str="{value}{prefix}"
    ).upper()
    logger.info(f"'Teacher Genomics model number of parameters: {teacher_model_num_parameters}'")
    logger.info(f"'Student Genomics model number of parameters: {student_model_num_parameters}'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer ... ")
    logger.info("Set tokenizer from pretrained ...")
    tokenizer = load_tuned_tokenizer(
        os.path.join(MODELS_DIR, pretrained_model_name, "final"),
        teacher_model.config.max_position_embeddings,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    _trainer_cfg = _resolve_precision(_read_config_dict(trainer_config_path))
    set_global_seed(_trainer_cfg.get("seed", 3469))
    trainer_args = TrainingArguments(**_trainer_cfg)
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers

    logger.info("Preparing your data for training")
    device, _, _ = get_backend()
    logger.info(f"'Training in device {device}'")

    compute_metrics = make_compute_classification_metrics()

    try_mkdir(trainer_args.output_dir)
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=5,
        lambda_param=0.5,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    logger.success("Modeling training complete.")
    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_datasets["train"])
    logger.success("Saving model.")
    trainer.save_model(
        os.path.join(MODELS_DIR, model_name, "final")
    )  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        manifest_mod.write(
            os.path.join(MODELS_DIR, model_name, "final"),
            seed=getattr(trainer_args, "seed", 3469),
            model={"name": model_name, "objective": "distillation"},
            throughput={"train_runtime_s": metrics.get("train_runtime")},
        )
    _elapsed_d = _time.perf_counter() - _t0
    logger.log(
        "STAGE",
        f"[train:distiller] done — elapsed={_elapsed_d:.0f} s → "
        f"{MODELS_DIR}/{model_name}/final/",
    )
    logger.success("Train model done.")
    # Evaluation
    if do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=lm_datasets["eval"])
        metrics["eval_samples"] = len(lm_datasets["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    # -----------------------------------------
    pass


if __name__ == "__main__":
    app()
