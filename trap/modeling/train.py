import os
import typer
import json
from si_prefix import si_format
from loguru import logger
from typing import Optional, Union, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from transformers import Trainer as Trainer, AlbertForMaskedLM, AlbertForSequenceClassification, AlbertConfig, TrainingArguments
from transformers import PreTrainedTokenizerFast, default_data_collator, DataCollatorWithPadding
from tokenizers import processors
from datasets import load_from_disk
from accelerate.test_utils.testing import get_backend
import evaluate

# from trap.modeling.albert import AlbertConfig, AlbertForMaskedLM, AlbertModel
from trap.loaders.tokenizer import WholeKmerMaskingDataCollator
from trap.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR

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
        *args, **kwargs
    ) -> None:
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
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
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        return (loss, student_output) if return_outputs else loss

@app.command()
def masking(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(default=..., help="Path to the pretrained tokenizer"),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_base_uncased.json", help="Path to the trainer config"),
    albert_config_path: Path = typer.Option(CONFIG_DIR / "albert_config_base_uncased.json", help="Path to the Albert config"),
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
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")
    
    logger.info("Training ALBERT model...")
    trainer_args = TrainingArguments(**json.load(open(trainer_config_path, 'r')))
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
    lm_datasets = load_from_disk(os.path.join(PROCESSED_DATA_DIR, preprocessing_name, 'masking'))
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
 
    device, _, _ = get_backend()
    logger.info(f"'Training in device {device}'")

    try_mkdir(trainer_args.output_dir)
    trainer = MaskingTrainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    logger.success("Modeling training complete.")
    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_datasets["train"])
    logger.success("Saving model.")
    trainer.save_model(os.path.join(MODELS_DIR, model_name, 'final')) # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.success("Train model done.")
    # -----------------------------------------
    pass

@app.command()
def classification(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_name: str = typer.Argument(help="Name of the model will be saved"),
    pretrained_tokenizer_path: Path = typer.Option(default=None, help="Path to the pretrained tokenizer"),
    pretrained_model_path: Path = typer.Option(default=None, help="Path to the pretrained model"),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_base_repeatmasker.json", help="Path to the trainer config"),
    albert_config_path: Path = typer.Option(CONFIG_DIR / "albert_config_base_uncased.json", help="Path to the Albert config"),
    builder: str = typer.Option(None, help="Path to the genome dataset"),
    k: int = typer.Option(18, help="K-mer size"),
    test_split: float = typer.Option(0.1, help="Test split ratio"),
    chunk_size: int = typer.Option(128, help="Chunk size for grouping texts"),
    preprocessing_name: str = typer.Option("gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"),
    num_workers: int = typer.Option(16, help="Number of workers"),
    do_eval: bool = typer.Option(False, help="Enable evaluation mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
    # -----------------------------------------
):
    debug_callback(debug)
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")
    
    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    lm_datasets = load_from_disk(os.path.join(PROCESSED_DATA_DIR, preprocessing_name, 'classification'))
    if debug_mode:
        logger.debug("Downsampling pretokenized dataset")
        train_size = 1_000
        test_size = int(0.1 * train_size)
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42)
        if do_eval:
            logger.debug("Downsampling pretokenized eval dataset")
            eval_size = int(0.1 * train_size)
            train_eval_split = lm_datasets['train'].train_test_split(
                test_size=eval_size, seed=42)
            lm_datasets['eval'] = train_eval_split['test']
            pass
        
    logger.info("Set training model...")
    # Extract the ClassLabel feature
    class_labels = lm_datasets['train'].features['label']
    # Create label2id and id2label mappings
    label2id = {label: class_labels.str2int(label) for label in class_labels.names}
    id2label = {class_labels.str2int(label): label for label in class_labels.names}

    if pretrained_model_path is not None and os.path.join(MODELS_DIR, pretrained_model_path, 'final').exists():
        logger.info("Set model from pretrained ...")
        model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_path, 'final'),
                                                            num_labels=len(class_labels.names),
                                                            id2label=id2label, label2id=label2id)
    else:
        logger.info("Set model from config ...")
        albert_config = AlbertConfig.from_json_file(albert_config_path)
        albert_config.num_labels=len(class_labels.names)
        albert_config.id2label=id2label
        albert_config.label2id=label2id
        model = AlbertForSequenceClassification(albert_config)
        pass

    model_num_parameters = model.num_parameters() / 1_000_000
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer")
    if pretrained_model_path is not None and os.path.join(MODELS_DIR, pretrained_model_path, 'final').exists():
        logger.info("Set tokenizer from pretrained ...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = os.path.join(MODELS_DIR, pretrained_model_path, 'final'),
                                                        local_files_only=True)
    else:
        logger.info("Set tokenizer from config ...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = pretrained_tokenizer_path,
                                                        local_files_only=True)
        pass
    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
    tokenizer.model_max_length = model.config.max_position_embeddings

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer_args = TrainingArguments(**json.load(open(trainer_config_path, 'r')))
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers
    
    logger.info("Preparing your data for training")
 
    device, _, _ = get_backend()
    logger.info(f"'Training in device {device}'")
    
    # Load individual metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    logger.info("Using accuracy, f1, precision, and recall as classification scores")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # Compute individual metrics
        accuracy_result = accuracy.compute(predictions=predictions, references=labels)
        f1_result = f1.compute(predictions=predictions, references=labels, average="micro")
        precision_result = precision.compute(predictions=predictions, references=labels, average="micro")
        recall_result = recall.compute(predictions=predictions, references=labels, average="micro")
        
        # Combine metrics into a single dictionary
        combined_metrics = {
            "accuracy": accuracy_result["accuracy"],
            "f1": f1_result["f1"],
            "precision": precision_result["precision"],
            "recall": recall_result["recall"]
        }
        
        return combined_metrics

    try_mkdir(trainer_args.output_dir)
    trainer = Trainer(
        model=model,
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
    trainer.save_model(os.path.join(MODELS_DIR, model_name, 'final')) # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
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
    pretrained_model_name: Path = typer.Option(default=None, help="Path to the pretrained teacher model"),
    pretrained_student_path: Path = typer.Option(default=None, help="Path to the pretrained student model"),
    trainer_config_path: Path = typer.Option(CONFIG_DIR / "trainer_config_base_repeatmasker.json", help="Path to the trainer config"),
    student_config_path: Path = typer.Option(CONFIG_DIR / "little_albert_config_base_uncased.json", help="Path to the Albert config"),
    preprocessing_name: str = typer.Option("gencode.v47.transcripts.k18.skipn.nocompress", help="Path to save the processed dataset"),
    num_workers: int = typer.Option(16, help="Number of workers"),
    do_eval: bool = typer.Option(False, help="Enable evaluation mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
    # -----------------------------------------
):
    """
    docstring
    """

    debug_callback(debug)
    if not trainer_config_path.exists():
        # Try in the CONFIG_DIR
        if (CONFIG_DIR / trainer_config_path).exists():
            trainer_config_path = CONFIG_DIR / trainer_config_path
        else:
            logger.error("Path to the trainer config not exist.")
    
    # TODO: arg.load_from_cache:
    logger.info("Loading pretokenized dataset")
    lm_datasets = load_from_disk(os.path.join(PROCESSED_DATA_DIR, preprocessing_name, 'classification'))
    if debug_mode:
        logger.debug("Downsampling pretokenized dataset")
        train_size = 1_000
        test_size = int(0.1 * train_size)
        lm_datasets = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42)
        if do_eval:
            logger.debug("Downsampling pretokenized eval dataset")
            eval_size = int(0.1 * train_size)
            train_eval_split = lm_datasets['train'].train_test_split(
                test_size=eval_size, seed=42)
            lm_datasets['eval'] = train_eval_split['test']
            pass
        
    logger.info("Set training model...")
    # Extract the ClassLabel feature
    class_labels = lm_datasets['train'].features['label']
    # Create label2id and id2label mappings
    label2id = {label: class_labels.str2int(label) for label in class_labels.names}
    id2label = {class_labels.str2int(label): label for label in class_labels.names}
        
    logger.info("Set model from pretrained ...")
    teacher_model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_name, 'final'))
    
    if pretrained_student_path is not None and os.path.join(MODELS_DIR, pretrained_student_path, 'final').exists():
        logger.info("Set model from pretrained ...")
        student_model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_student_path, 'final'),
                                                            num_labels=len(class_labels.names),
                                                            id2label=id2label, label2id=label2id)
    else:
        logger.info("Set model from config ...")
        albert_config = AlbertConfig.from_json_file(student_config_path)
        albert_config.num_labels=len(class_labels.names)
        albert_config.id2label=id2label
        albert_config.label2id=label2id
        student_model = AlbertForSequenceClassification(albert_config)
        pass

    teacher_model_num_parameters = si_format(teacher_model.num_parameters(), precision=0, format_str=u'{value}{prefix}').upper()
    student_model_num_parameters = si_format(student_model.num_parameters(), precision=0, format_str=u'{value}{prefix}').upper()
    logger.info(f"'Teacher Genomics model number of parameters: {teacher_model_num_parameters}'")
    logger.info(f"'Student Genomics model number of parameters: {student_model_num_parameters}'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer ... ")
    logger.info("Set tokenizer from pretrained ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = os.path.join(MODELS_DIR, pretrained_model_name, 'final'),
                                                        local_files_only=True)

    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )
    tokenizer.model_max_length = teacher_model.config.max_position_embeddings

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer_args = TrainingArguments(**json.load(open(trainer_config_path, 'r')))
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers
    
    logger.info("Preparing your data for training")
    device, _, _ = get_backend()
    logger.info(f"'Training in device {device}'")
    
    # Load individual metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    logger.info("Using accuracy, f1, precision, and recall as classification scores")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # Compute individual metrics
        accuracy_result = accuracy.compute(predictions=predictions, references=labels)
        f1_result = f1.compute(predictions=predictions, references=labels, average="micro")
        precision_result = precision.compute(predictions=predictions, references=labels, average="micro")
        recall_result = recall.compute(predictions=predictions, references=labels, average="micro")
        
        # Combine metrics into a single dictionary
        combined_metrics = {
            "accuracy": accuracy_result["accuracy"],
            "f1": f1_result["f1"],
            "precision": precision_result["precision"],
            "recall": recall_result["recall"]
        }
        
        return combined_metrics

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
    trainer.save_model(os.path.join(MODELS_DIR, model_name, 'final')) # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
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
