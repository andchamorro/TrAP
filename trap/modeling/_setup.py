"""Model + tokenizer loading helpers shared across training and inference (DR-2).

Consolidates the repeated ``from_pretrained`` + ``post_processor`` boilerplate
that previously appeared in ``train.py``, ``predict.py``, and
``preprocessing_sequences.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from tokenizers import processors
from transformers import AlbertForSequenceClassification, PreTrainedTokenizerFast

from trap.config.config import MODELS_DIR


def load_model_and_tokenizer(
    model_path: Path,
    *,
    attn_implementation: str = "sdpa",
    model_max_length: Optional[int] = None,
    local_files_only: bool = True,
) -> Tuple[AlbertForSequenceClassification, PreTrainedTokenizerFast]:
    """Load a saved ALBERT classifier and its paired tokenizer.

    Appends ``/final`` when *model_path* does not already end with
    ``"final"``.  Path resolution order:
    1. Absolute path as given.
    2. ``MODELS_DIR / model_path / final``.

    Args:
        model_path: Model directory relative to ``MODELS_DIR``, or an
            absolute path.
        attn_implementation: Attention kernel; ``"sdpa"`` uses PyTorch's
            fused scaled-dot-product attention (~15% faster).
        model_max_length: Override the tokenizer's ``model_max_length``.
            Defaults to ``model.config.max_position_embeddings``.
        local_files_only: Passed through to HuggingFace ``from_pretrained``.

    Returns:
        ``(model, tokenizer)`` — tokenizer ``model_max_length`` is always
        synced to ``model.config.max_position_embeddings``.
    """
    final_path = Path(model_path)
    if not final_path.is_absolute():
        final_path = MODELS_DIR / model_path
    if final_path.name != "final":
        final_path = final_path / "final"

    logger.info(f"Loading model from {final_path}")
    model = AlbertForSequenceClassification.from_pretrained(
        str(final_path),
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

    logger.info(f"Loading tokenizer from {final_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(final_path),
        local_files_only=local_files_only,
    )
    # Safety net: bake TemplateProcessing for tokenizers saved before QW-10.
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
            ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
        ],
    )
    tokenizer.model_max_length = model_max_length or model.config.max_position_embeddings
    return model, tokenizer
