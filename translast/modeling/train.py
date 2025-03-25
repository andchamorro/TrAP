import os
import io
import typer
import json
from loguru import logger
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from transformers import Trainer as Trainer, AlbertForMaskedLM, AlbertConfig, TrainingArguments
from transformers import PreTrainedTokenizerFast, default_data_collator
from tokenizers import processors
from datasets import load_from_disk

# from translast.modeling.albert import AlbertConfig, AlbertForMaskedLM, AlbertModel
from translast.loaders.tokenizer import WholeKmerMaskingDataCollator
from translast.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

debug_mode = False
def debug_callback(debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")):
    """
    Callback function to handle the debug flag.
    """
    global debug_mode
    if debug:
        typer.echo("Debug mode enabled")
        debug_mode = True

def try_mkdir(dir_name):
    # Save the tokenizer
    try:
        os.makedirs(dir_name)
    except FileExistsError:
            # directory already exists
            pass
class MaskingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
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
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
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

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(default=..., help="Path to the pretrained tokenizer"),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_fsdp.json", help="Path to the trainer config"),
    albert_config_path: Path = typer.Option(CONFIG_DIR / "albert_config.json", help="Path to the Albert config"),
    fsdp_config: str = typer.Option(None, help="Config to be used with fsdp (Pytorch Distributed Parallel Training)."),
    builder: str = typer.Option(None, help="Path to the genome dataset"),
    k: int = typer.Option(18, help="K-mer size"),
    test_split: float = typer.Option(0.1, help="Test split ratio"),
    chunk_size: int = typer.Option(128, help="Chunk size for grouping texts"),
    preprocessing_name: str = typer.Option("gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"),
    num_workers: int = typer.Option(16, help="Number of workers"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
    # -----------------------------------------
):
    debug_callback(debug)
    logger.info("Training ALBERT model...")
    with open(trainer_config_path, 'r') as trainer_config_file:
        trainer_config = json.load(trainer_config_file)
        if fsdp_config is not None:
            trainer_config["fsdp_config"] = fsdp_config
            pass

    trainer_args = TrainingArguments(**trainer_config)
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers
    model = AlbertForMaskedLM(AlbertConfig.from_json_file(albert_config_path))

    model_num_parameters = model.num_parameters() / 1_000_000
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = pretrained_tokenizer_path,
                                                        local_files_only=True)
    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
    tokenizer.model_max_length = model.config.max_position_embeddings
    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    lm_datasets = load_from_disk(os.path.join(PROCESSED_DATA_DIR, preprocessing_name))
    if debug_mode:
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
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
        
    lm_datasets["eval"] = lm_datasets["test"].map(
        insert_random_mask,
        batched=True, num_proc=num_workers,
        remove_columns=lm_datasets["test"].column_names,
    )
        
    lm_datasets["eval"] = lm_datasets["eval"].rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_token_type_ids": "token_type_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    
    logger.info("Preparing your data for training")
 
    device = torch.device(device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"'Training in device {device.type}'")

    try_mkdir(trainer_args.output_dir)
    trainer = MaskingTrainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.success("Modeling training complete.")
    # -----------------------------------------
    pass

if __name__ == "__main__":
    app()
