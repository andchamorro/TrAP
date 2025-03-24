import os
import math
import numpy as np
import typer
import json
from loguru import logger
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import PreTrainedTokenizerFast, default_data_collator, get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tokenizers import processors
from datasets import load_from_disk

from translast.modeling.albert import AlbertConfig, AlbertForMaskedLM, AlbertModel
from translast.loaders.tokenizer import WholeKmerMaskingDataCollator
from translast.config.config import MODELS_CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR

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
class TrainerConfig:
    def __init__(self, seed: int = 3469,
                 num_train_batches: int = 32,
                 num_eval_batches: int = 4,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 num_train_epochs: int = 25,
                 warmup: float = 0.1, 
                 save_steps: int = 100,
                 total_steps: int = 1000,
                 save_frequency: int = 500,
                 max_train_samples: int = None,
                 evaluation_strategy: str = "epoch",
                 save_strategy: str = "epoch",
                 per_device_train_batch_size: int = 16,
                 per_device_eval_batch_size: int = 16,
                 data_parallel: bool = False):
        
        self.seed = seed
        self.num_train_batches = num_train_batches
        self.num_eval_batches = num_eval_batches
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.warmup = warmup
        self.save_steps = save_steps
        self.total_steps = total_steps
        self.save_frequency = save_frequency
        self.max_train_samples = max_train_samples
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_parallel = data_parallel

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class Trainer:
    """
    Training Helper Class

    This class provides helper functions to train and evaluate a PyTorch model. It handles the training loop, evaluation loop,
    model saving, and loading. The optimizer used is `torch.optim.Adam`.

    Args:
        config (TrainerConfig): Configuration object containing training parameters.
        model (nn.Module): The PyTorch model to be trained.
        data_iter (iter): Iterator to load data.
        device (str): Device name (e.g., 'cpu' or 'cuda').

    Methods:
        train(loss_function, model_file=None, data_parallel=False):
            Runs the training loop.

        eval(evaluate, model_file, data_parallel=True):
            Runs the evaluation loop.

        load(model_file):
            Loads a saved model or pretrained transformer.

        save(i):
            Saves the current model state.
    """

    def __init__(self, config, model, train_dataloader, eval_dataloader, processing_class, device):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config.num_training_steps = self.config.num_train_epochs * len(train_dataloader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.processing_class=processing_class
        self.device = device
        self.accelerator = Accelerator(project_config = ProjectConfiguration(project_dir = self.config.output_dir, total_limit=5))
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=config.num_training_steps,
            )
        self.global_step = 0

    def train(self, data_parallel=False):
        """
        Train Loop

        This method runs the training loop for the model. It handles loading the model, training, saving checkpoints,
        and handling data parallelism.

        Args:
            loss_function (function): Function to calculate the loss.
            model_file (str, optional): Path to a saved model file to load. Defaults to None.
            data_parallel (bool, optional): Whether to use Data Parallelism with Multi-GPU. Defaults to False.
        """
        self.model = self.model.to(self.device)
        if data_parallel:  # If Multi-GPU
            self.model = nn.DataParallel(self.model)
        self.model, self.optimizer, self.train_dataloader , self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader , self.eval_dataloader
        )
        # Save the starting state
        self.save_checkpoint()

        self.model.to(self.accelerator.device)

        for e in range(self.config.num_train_epochs):
            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            progress_bar = tqdm(self.train_dataloader, desc='Iter (loss=X.XXX)')
            
            self.model.train()  # train mode
            for train_step, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = outputs['loss']
                self.accelerator.backward(loss)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                loss_sum += loss.item()

                self.global_step += 1
        
            # Evaluation
            self.model.eval()
            losses = []
            for eval_step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                loss = outputs['loss']
                losses.append(self.accelerator.gather(loss.repeat(self.config.batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(self.eval_dataloader)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            
            logger.info('Epoch %d/%d : Average Loss %5.3f Perplexity: %5.3f' % (e + 1, self.config.num_train_epochs, loss_sum / (train_step + 1), perplexity))
            # Save and upload
            self.accelerator.wait_for_everyone()
            self.save_checkpoint()

    def eval(self, evaluate, eval_dataloader, data_parallel=True):
        """
        Evaluation Loop

        This method runs the evaluation loop for the model. It handles loading the model, evaluating, and handling data parallelism.

        Args:
            evaluate (function): Function to evaluate the model.
            model_file (str): Path to a saved model file to load.
            data_parallel (bool, optional): Whether to use Data Parallelism with Multi-GPU. Defaults to True.

        Returns:
            list: A list of prediction results.
        """
        self.model.eval()  # evaluation mode
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = []  # prediction results
        iter_tqdm = tqdm(eval_dataloader, desc='Iteraction (loss=X.XXX)')
        for batch in iter_tqdm:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # Not calule the gradient
                accuracy, result = evaluate(model, batch)
            results.append(result)
            iter_tqdm.set_description('Iteraction (acc=%5.3f)' % accuracy)
        return results
    
    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            # Save model checkpoint
            checkpoint_folder = f"checkpoint-{self.global_step}"

            output_dir = os.path.join(self.config.output_dir, checkpoint_folder)
            self.save_model(output_dir)

            # Save the state
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "AdamW"))
            self.accelerator.save_state(os.path.join(output_dir, "accelerator"))


    def save_model(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save current model
        """
        output_dir = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        
        # Good practice: save your training arguments together with the trained model
        torch.save(self.config, os.path.join(output_dir, "training_config.bin"))

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(default=..., help="Path to the pretrained tokenizer"),
    trainer_config_path: Path = typer.Option(MODELS_CONFIG_DIR / "trainer_config.json", help="Path to the trainer config"),
    albert_config_path: Path = typer.Option(MODELS_CONFIG_DIR / "albert_config.json", help="Path to the Albert config"),
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
    trainer_config = TrainerConfig.from_json(trainer_config_path)
    albert_config = AlbertConfig.from_json(albert_config_path)
    
    trainer_config.chunk_size = albert_config.max_position_embeddings
    model = AlbertForMaskedLM(AlbertModel(albert_config))

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
    tokenizer.model_max_length = albert_config.max_position_embeddings
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

    def insert_random_mask(batch, data_collator):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    logger.info("Insert Random Mask to the evaluation dataset")
    eval_dataset = lm_datasets["test"].map(
        lambda examples: insert_random_mask(examples, data_collator),
        batched=True, num_proc=num_workers,
        remove_columns=lm_datasets["test"].column_names,)
    
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    
    logger.info("Preparing your data for training")
    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=trainer_config.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=trainer_config.per_device_eval_batch_size,
        collate_fn=default_data_collator
    )

    device = torch.device(device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"'Training in device {device.type}'")

    trainer_config.output_dir = os.path.join(MODELS_DIR, model_name)
    try_mkdir(trainer_config.output_dir)
    trainer = Trainer(
        config=trainer_config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        processing_class=tokenizer,
        device=device
    )
    trainer.train()
    logger.success("Modeling training complete.")
    # -----------------------------------------
    pass

if __name__ == "__main__":
    app()
