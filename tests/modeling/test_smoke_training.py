"""Smoke tests that the training stacks can actually *learn*.

Regression guard for the failure where MLM pretraining ran for 35 h but the
model never left the uniform-random baseline (loss pinned at ``ln(vocab)``,
masked-token accuracy frozen across all epochs). These tests train a tiny model
for a few hundred steps on a trivially learnable dataset using the *real*
collator / model classes, and assert the loss drops decisively below the
uniform baseline. They are CPU-only (fp32) and complement the on-GPU SLURM
gates (``24_mlm_smoke``/``34_classification_smoke``) which validate the real
configs on real data.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _tiny_albert_config(vocab_size, num_labels=None):
    from transformers import AlbertConfig

    kwargs = dict(
        vocab_size=vocab_size,
        embedding_size=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        num_hidden_groups=1,
        inner_group_num=1,
    )
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    return AlbertConfig(**kwargs)


@pytest.mark.slow
@pytest.mark.integration
class TestMlmSmokeTraining:
    """A tiny ``AlbertForMaskedLM`` must drive MLM loss below ``ln(vocab)``."""

    def test_mlm_training_reduces_loss(self):
        from types import SimpleNamespace

        from transformers import AlbertForMaskedLM

        from trap.loaders.tokenizer import WholeKmerMaskingDataCollator

        torch.manual_seed(3469)
        vocab, seq_len, mask_id = 24, 12, 1
        canonical = list(range(2, 2 + seq_len))  # fixed, trivially memorizable

        collator = WholeKmerMaskingDataCollator(SimpleNamespace(mask_token_id=mask_id))

        def make_batch(n=16):
            # Fresh copies every call: the collator masks in place.
            feats = [
                {
                    "input_ids": list(canonical),
                    "attention_mask": [1] * seq_len,
                    "token_type_ids": [0] * seq_len,
                    "word_ids": list(range(seq_len)),
                    "labels": list(canonical),
                }
                for _ in range(n)
            ]
            return collator(feats)

        model = AlbertForMaskedLM(_tiny_albert_config(vocab))
        model.train()

        def mean_loss(steps=5):
            losses = []
            with torch.no_grad():
                for _ in range(steps):
                    losses.append(model(**make_batch()).loss.item())
            return sum(losses) / len(losses)

        uniform = math.log(vocab)
        initial = mean_loss()
        assert initial > uniform * 0.7, f"untrained loss {initial:.3f} should be near ln(V)={uniform:.3f}"

        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        for _ in range(200):
            opt.zero_grad()
            loss = model(**make_batch()).loss
            loss.backward()
            opt.step()

        final = mean_loss()
        assert final < uniform - 1.0, (
            f"MLM did not learn: final loss {final:.3f} not below ln(V)-1={uniform - 1.0:.3f}"
        )
        assert final < initial * 0.5, f"final {final:.3f} should be << initial {initial:.3f}"


@pytest.mark.slow
@pytest.mark.integration
class TestMaskingTrainerSmoke:
    """Drive the *real* ``MaskingTrainer`` (HF Trainer path) on CPU/fp32.

    The manual-loop test above exercises the model class but not the Trainer.
    This one runs the actual ``MaskingTrainer`` + ``WholeKmerMaskingDataCollator``
    + ``compute_masking_metrics`` + ``mlm_preprocess_logits_for_metrics`` stack
    so a regression in the Trainer machinery (or the installed transformers
    version) that freezes training is caught without a GPU.
    """

    def test_masking_trainer_reduces_loss(self, tmp_path):
        from types import SimpleNamespace

        from datasets import Dataset
        from transformers import AlbertForMaskedLM, TrainingArguments

        from trap.loaders.tokenizer import WholeKmerMaskingDataCollator
        from trap.modeling.train import (
            MaskingTrainer,
            compute_masking_metrics,
            mlm_preprocess_logits_for_metrics,
        )

        torch.manual_seed(3469)
        vocab, seq_len, mask_id, n_rows = 24, 12, 1, 128
        canonical = list(range(2, 2 + seq_len))
        rows = {
            "input_ids": [list(canonical) for _ in range(n_rows)],
            "attention_mask": [[1] * seq_len for _ in range(n_rows)],
            "token_type_ids": [[0] * seq_len for _ in range(n_rows)],
            "word_ids": [list(range(seq_len)) for _ in range(n_rows)],
            "labels": [list(canonical) for _ in range(n_rows)],
        }
        train_ds = Dataset.from_dict(rows)
        collator = WholeKmerMaskingDataCollator(SimpleNamespace(mask_token_id=mask_id))
        model = AlbertForMaskedLM(_tiny_albert_config(vocab))

        args = TrainingArguments(
            output_dir=str(tmp_path),
            num_train_epochs=30,
            per_device_train_batch_size=16,
            learning_rate=5e-3,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            logging_steps=10_000,
            use_cpu=True,
            seed=3469,
        )
        trainer = MaskingTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
            compute_metrics=compute_masking_metrics,
            preprocess_logits_for_metrics=mlm_preprocess_logits_for_metrics,
        )
        trainer.train()

        # Check the trained model directly (train_loss is a running mean from step 0).
        model.eval()
        uniform = math.log(vocab)
        losses = []
        with torch.no_grad():
            for _ in range(5):
                feats = [
                    {
                        "input_ids": list(canonical),
                        "attention_mask": [1] * seq_len,
                        "token_type_ids": [0] * seq_len,
                        "word_ids": list(range(seq_len)),
                        "labels": list(canonical),
                    }
                    for _ in range(16)
                ]
                losses.append(model(**collator(feats)).loss.item())
        final = sum(losses) / len(losses)
        assert final < uniform - 1.0, (
            f"MaskingTrainer did not learn: final loss {final:.3f} not below "
            f"ln(V)-1={uniform - 1.0:.3f} — the Trainer path is frozen"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestClassificationSmokeTraining:
    """A tiny ``AlbertForSequenceClassification`` must learn a separable task."""

    def test_classification_training_reduces_loss(self):
        from transformers import AlbertForSequenceClassification

        torch.manual_seed(3469)
        vocab, seq_len, num_labels = 24, 12, 2
        # Label is fully determined by the token used: class 0 -> all 2s, class 1 -> all 3s.
        rows = [([2] * seq_len, 0), ([3] * seq_len, 1)] * 8
        input_ids = torch.tensor([r[0] for r in rows])
        labels = torch.tensor([r[1] for r in rows])
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(len(rows), seq_len, dtype=torch.long),
            "token_type_ids": torch.zeros(len(rows), seq_len, dtype=torch.long),
            "labels": labels,
        }

        model = AlbertForSequenceClassification(_tiny_albert_config(vocab, num_labels=num_labels))
        model.train()

        uniform = math.log(num_labels)
        with torch.no_grad():
            initial = model(**batch).loss.item()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(150):
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(**batch)
        final = out.loss.item()
        acc = (out.logits.argmax(-1) == labels).float().mean().item()

        assert final < uniform - 0.2, (
            f"classifier did not learn: final loss {final:.3f} not below "
            f"ln(C)-0.2={uniform - 0.2:.3f}"
        )
        assert final < initial * 0.7, f"final {final:.3f} should be << initial {initial:.3f}"
        assert acc == pytest.approx(1.0), f"separable task not solved: acc={acc:.3f}"
