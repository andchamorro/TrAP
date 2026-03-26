import os
import time
import typer
import json
from typing import Dict, List
from pathlib import Path

import numpy as np
import pickle
import pysam
from Bio import SeqIO
from tqdm import tqdm
from loguru import logger

import torch
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers import pipeline, BitsAndBytesConfig
from transformers.pipelines import TextClassificationPipeline
from transformers.pipelines.base import GenericTensor
from tokenizers import processors
from datasets import load_from_disk, Dataset

from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object

# from trap.modeling.albert import AlbertConfig, AlbertForMaskedLM, AlbertModel
from trap.config.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from trap.loaders.dataset import GenomeDataset
from trap.utils.io import genome_file_handle, try_mkdir
from trap.utils.kmer import kmer_split

START_TIME = time.strftime("%Y%m%d_%H%M%S")
# DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

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

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

class SAMDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        self._ids, self.queries = self._load_alignments()
        self._index = 0  # Initialize the index for iteration

    def _load_alignments(self):
        queries = {}
        alignments = self.alignments()
        for record in alignments:
            queries[record.query_name] = record.query_sequence
        return list(queries.keys()), queries
    
    def alignments(self):
        if self.file_path.suffix == '.bam':
            return pysam.AlignmentFile(self.file_path, "rb")
        elif self.file_path.suffix == '.sam':
            return pysam.AlignmentFile(self.file_path, "r")
        elif self.file_path.suffix == '.cram':
            return pysam.AlignmentFile(self.file_path, "rc")
        else :
            raise ValueError("Unsupported file type")
    
    def __len__(self):
        return len(self._ids)
    
    def __iter__(self):
        self._index = 0  # Reset the index for a new iteration
        return self
    
    def __next__(self):
        if self._index < len(self._ids):
            key = self._ids[self._index]
            self._index += 1
            return key, self.queries[key]
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._ids[index], [self.queries[key] for key in self._ids[index]]
        elif isinstance(index, int):
            if index < 0:
                index += len(self._ids)
            if index >= len(self._ids) or index < 0:
                raise IndexError("The index is out of range.")
            key = self._ids[index]
            return key, self.queries[key]
        else:
            raise TypeError("Invalid argument type.")

def break_long_read(long_read, read_length=150, mean_fragment_size=500, std_fragment_size=10, coverage=5):
    if len(long_read) < mean_fragment_size + std_fragment_size:
        return [{'forward': long_read[:read_length], 'reverse': long_read[read_length::-1]}]
    # Calculate the number of fragments needed to achieve the desired coverage
    num_fragments = int(len(long_read) * coverage / mean_fragment_size)
    
    # Generate fragment sizes based on the mean and standard deviation
    fragment_sizes = np.random.normal(mean_fragment_size, std_fragment_size, num_fragments).astype(int)
    
    # Initialize an empty list to store the short reads
    short_reads = []
    
    # Generate short reads from the long read
    for fragment_size in fragment_sizes:
        try:
            forward_start = np.random.randint(0, len(long_read) - fragment_size + 1)
        except ValueError:
            forward_start = 0
        forward_end = forward_start + read_length
        forward = long_read[forward_start:forward_end]

        insert_size = fragment_size - (read_length * 2)
        reverse_start = forward_end + insert_size
        reverse_end = reverse_start + read_length
        if reverse_end > len(long_read):
            # we use random insert when the modelled template length distribution
            # is too large
            reverse_end = np.random.randint(read_length, len(long_read))
            reverse_start = reverse_end - read_length
        reverse = long_read[reverse_start:reverse_end]
        short_reads.append({'forward': forward, 'reverse': reverse})
    
    return short_reads

class TokenizedTextClassificationPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return inputs

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
    import evaluate
    from transformers import Trainer, AlbertForSequenceClassification, TrainingArguments

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
        logger.debug("Downsampling pretokenized eval dataset")
        eval_size = int(0.1 * train_size)
        train_eval_split = lm_datasets['train'].train_test_split(
            test_size=eval_size, seed=42)
        lm_datasets['eval'] = train_eval_split['test']
        pass
        
    logger.info("Set training model...")

    logger.info("Set model from pretrained ...")
    model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_path, 'final'))
    model_num_parameters = model.num_parameters() / 1_000_000
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer")
    logger.info("Set tokenizer from pretrained ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = os.path.join(MODELS_DIR, pretrained_model_path, 'final'),
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

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer_args = TrainingArguments(**json.load(open(trainer_config_path, 'r')))
    trainer_args.output_dir = os.path.join(MODELS_DIR, model_name)
    trainer_args.dataloader_num_workers = num_workers
    
    logger.info("Preparing your data for training")
 
    device = torch.device(device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"'Training in device {device.type}'")
    
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
def processing_dataset(
    input_file: Path = typer.Option(None, help="Path to the reads dataset"),
    pair_file: Path = typer.Option(None, help="Path to the reads dataset"),
    output_path: Path = typer.Option(None, help="Path to the output dataset"),
    k: int = typer.Option(18, help="K-mer size"),
    save_processing: bool = typer.Option(False, help="Padding when tokenize"),
    pretrained_tokenizer_name: Path = typer.Option(None, help="Tokenize the reads dataset"),
    padding: bool = typer.Option(True, help="Padding when tokenize"),
    is_long: bool = typer.Option(False, "--is-long", "-l", help="Declarate if the reads dataset is a long-read sequencing"),
    num_workers: int = typer.Option(16, help="Number of workers")
):
    break_fn = break_long_read if is_long else lambda x: [x]
    standardization = GenomeDataset._standardization
    if ''.join(input_file.suffixes) in ['.fq.gz', '.fastq.gz', '.fq.bgz', '.fastq.bgz', '.fq', '.fastq']:
        if pair_file is not None:
            def generator_from_iterator():
                with genome_file_handle(input_file) as r1_handle, genome_file_handle(pair_file) as r2_handle:
                    for r1, r2 in zip(SeqIO.parse(r1_handle, "fastq"), SeqIO.parse(r2_handle, "fastq")):
                        yield {'pair': {'text': kmer_split(k, standardization(str(r1.seq))), 'text_pair': kmer_split(k, standardization(str(r2.seq)))},
                           'id': str(r1.id)}
        else:
            # TODO: pass to just genome_file_handle
            raw_reads = GenomeDataset(input_file, "fastq")
            def generator_from_iterator():
                for seq, rev, id in raw_reads:
                    for pair in break_fn(seq):
                        yield {'pair': {'text': kmer_split(k, pair['forward']), 'text_pair': kmer_split(k, pair['reverse'])},
                               'id': id}

    elif ''.join(input_file.suffixes) in ['.sam', '.bam', '.cram']:
        # TODO: pass to just sam_file_handle
        raw_alignments = SAMDataset(input_file)
        break_fn = break_long_read if is_long else lambda x: x
        def generator_from_iterator():
            for key, seq in raw_alignments.queries.items():
                for pair in break_fn(seq):
                    yield {'pair': {'text': kmer_split(k, pair['forward']), 'text_pair': kmer_split(k, pair['reverse'])},
                           'id': key}
    else:
        raise ValueError("Unsupported file type")

    logger.info("Create datasets from generator")
    raw_dataset = Dataset.from_generator(generator_from_iterator, num_proc=num_workers)
    logger.info(f"Datasets create size {len(raw_dataset)}")
    try_mkdir(os.path.join(output_path, 'short_reads'))
    if save_processing or pretrained_tokenizer_name is None:
        raw_dataset.save_to_disk(os.path.join(output_path, 'short_reads'), num_proc=num_workers)
        logger.info(f"Preprocessing datasets saved in {os.path.join(output_path, 'short_reads')}")
        pass
    if pretrained_tokenizer_name is not None:
        logger.info("Loading pretrained tokenizer")
        logger.info("Set tokenizer from pretrained ...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path = os.path.join(MODELS_DIR, pretrained_tokenizer_name, 'final'),
                                                            local_files_only=True)

        tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS]:0 $A:0 [SEP]:0",
                pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]")),
                    ("[SEP]", tokenizer.convert_tokens_to_ids("[SEP]")),
                ],
            )


        def tokenize_function(examples):
            result = tokenizer(examples["pair"]["text"], examples["pair"]["text_pair"], padding=padding, truncation=True, verbose=False)
            return result
        
        tokenized_datasets = raw_dataset.map(
            tokenize_function, batched=False, remove_columns=['pair', 'id'], load_from_cache_file=False, num_proc=num_workers,
            desc="Running tokenizer on dataset"
        )
        logger.info("Datasets tokenized")
        try_mkdir(os.path.join(output_path, 'short_reads', 'tokenized'))
        tokenized_datasets.save_to_disk(os.path.join(output_path, 'short_reads', 'tokenized'))
        logger.info(f"Tokenized datasets saved in {os.path.join(output_path, 'short_reads', 'tokenized')}")
    pass

@app.command()
def quantify(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pretrained_model_name: Path = typer.Option(default=None, help="Path to the pretrained model"),
    output_path: Path = typer.Option(None, help="Path to the output dataset"),
    batch_size: int = typer.Option(16, help="Chunk size for grouping texts"),
    num_shards: int = typer.Option(None, help="Chunk size for grouping texts"),
    shards_index: int = typer.Option(0, help="Chunk size for grouping texts"),
    is_tokenized: bool = typer.Option(False, help="Enable debug mode"),
    bitsandbytes: str = typer.Option(None, help="Enable debug mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
    # -----------------------------------------
):
    # Imports
    from transformers import AlbertForSequenceClassification
    from transformers.pipelines.pt_utils import KeyDataset

    debug_callback(debug)
    quantization_config = None
    if bitsandbytes is not None:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if (bitsandbytes == '4bit') else BitsAndBytesConfig(load_in_8bit=True)
        
    logger.info("Set model from pretrained ...")
    model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_name, 'final'), 
                                                                attn_implementation="sdpa")
    model_num_parameters = model.num_parameters() / 1_000_000
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading pretrained tokenizer")
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
    tokenizer.model_max_length = model.config.max_position_embeddings
        
    # TODO: arg.load_from_cache:
    logger.info("Loading and processing dataset")
    if is_tokenized:
        logger.info("Loading tokenized dataset")
        processed_dataset = load_from_disk(os.path.join(output_path, 'short_reads', 'tokenized'))
        if num_shards is not None:
            logger.info(f"Shard dataset: shard index {shards_index}/{num_shards}")
            processed_dataset = processed_dataset.shard(num_shards=num_shards, index=shards_index)
        processed_dataset.set_format(type='torch')
    else:
        logger.info("Preprocessed tokenized dataset")
        processed_dataset = load_from_disk(os.path.join(output_path, 'short_reads'))
    if debug_mode:
        logger.debug("Downsampling pretokenized dataset")
        dataset_size = 1_000
        processed_dataset = processed_dataset.select(np.random.randint(len(processed_dataset), size=dataset_size))
    pass

    distributed_state = PartialState()
    # Create a classification pipeline
    classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, 
                          top_k=None, torch_dtype=torch.bfloat16, 
                          device = distributed_state.device, pipeline_class=TokenizedTextClassificationPipeline,
                          model_kwargs={"quantization_config": quantization_config})
    logger.info("Compile the model ...")
    classifier.model = torch.compile(classifier.model, backend="inductor", mode="max-autotune", fullgraph=True)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        if not os.path.exists(output_path):
            try_mkdir(output_path)
            logger.info(f"Directory '{output_path}' created successfully.")
        else:
            logger.info(f"Directory '{output_path}' already exists.")
    
    distributed_state.wait_for_everyone()
    with distributed_state.split_between_processes(processed_dataset) as datasets:
        results_bar = tqdm(classifier(datasets.iter(batch_size=1), batch_size=batch_size, truncation=True, padding=True), desc=f"classifying {distributed_state.device}", total=len(datasets))
        class_scores = []
        for step, results in enumerate(results_bar):
            class_scores.append(results)

    distributed_state.wait_for_everyone()
    class_scores = gather_object(class_scores)

    if distributed_state.is_main_process:
        with open(os.path.join(output_path, f'class_scores_{shards_index}.pkl'), 'wb') as f:
            pickle.dump(class_scores, f)
        logger.info(f"Quantification finished. Saved in '{output_path}'")

    pass

@app.command()
def cpu_quantify(
    pretrained_model_name: Path = typer.Option(default=None, help="Path to the pretrained model"),
    output_path: Path = typer.Option(None, help="Path to the output dataset"),
    batch_size: int = typer.Option(16, help="Chunk size for grouping texts"),
    num_shards: int = typer.Option(None, help="Number of shards"),
    shards_index: int = typer.Option(0, help="Shard index"),
    is_tokenized: bool = typer.Option(False, help="Use tokenized dataset"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
):
    from transformers import AlbertForSequenceClassification
    debug_callback(debug)

    logger.info("Loading ONNX model ...")
    model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_name, 'final'), 
                                                                attn_implementation="sdpa")
    model_num_parameters = model.num_parameters() / 1_000_000
    logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
    logger.info("Original ALBERT number of parameters: 11M")
    logger.info("Original BERT number of parameters: 110M")

    logger.info("Loading tokenizer ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, 'final'),
        local_files_only=True
    )

    tokenizer.model_max_length = model.config.max_position_embeddings

    logger.info("Loading dataset ...")
    if is_tokenized:
        processed_dataset = load_from_disk(os.path.join(output_path, 'short_reads', 'tokenized'))
        if num_shards is not None:
            processed_dataset = processed_dataset.shard(num_shards=num_shards, index=shards_index)
        processed_dataset.set_format(type='torch')
    else:
        processed_dataset = load_from_disk(os.path.join(output_path, 'short_reads'))

    if debug:
        logger.debug("Downsampling dataset for debug mode")
        processed_dataset = processed_dataset.select(np.random.randint(len(processed_dataset), size=1000))

    distributed_state = PartialState(cpu=True)

    logger.info("Creating ONNX pipeline ...")
    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        pipeline_class=TokenizedTextClassificationPipeline,
        top_k=None, 
        torch_dtype=torch.bfloat16,
        device=-1,  # CPU
    )

    if distributed_state.is_main_process:
        try_mkdir(output_path)

    distributed_state.wait_for_everyone()
    with distributed_state.split_between_processes(processed_dataset) as datasets:
        results_bar = tqdm(
            classifier(datasets.iter(batch_size=1), batch_size=batch_size, truncation=True, padding=True),
            desc=f"classifying on CPU", total=len(datasets))
        class_scores = [results for results in results_bar]

    distributed_state.wait_for_everyone()
    class_scores = gather_object(class_scores)

    if distributed_state.is_main_process:
        with open(os.path.join(output_path, f'class_scores_{shards_index}.pkl'), 'wb') as f:
            pickle.dump(class_scores, f)
        logger.info(f"Quantification finished. Saved in '{output_path}'")

@app.command()
def convert_model_to_onnx(
    pretrained_model_name: Path = typer.Option(default=None, help="Path to the pretrained model"),
    opset: int = 14,
    use_auth_token: bool = False
):
    """
    Converts a Hugging Face Transformers model to ONNX format using Optimum.

    Args:
        model_name_or_path (str): Path or model ID of the pretrained model.
        output_dir (str): Directory to save the ONNX model.
        opset (int): ONNX opset version.
        use_auth_token (bool): Whether to use Hugging Face auth token (for private models).
    """
    from optimum.exporters.onnx import main_export

    output_dir = os.path.join(MODELS_DIR, pretrained_model_name, 'onnx')
    try_mkdir(output_dir)

    logger.info("Loading tokenizer ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(MODELS_DIR, pretrained_model_name, 'final'),
        local_files_only=True
    )

    # Run the export
    main_export(
        model_name_or_path=os.path.join(MODELS_DIR, pretrained_model_name, 'final'),
        output=output_dir,
        task="text-classification",
        opset=opset,
        tokenizer=tokenizer,
        trust_remote_code=True,
        use_auth_token=use_auth_token
    )

    print(f"Model successfully exported to ONNX at: {output_dir}")

@app.command()
def accelerate_quantify(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    pretrained_model_name: Path = typer.Option(default=None, help="Path to the pretrained model"),
    output_path: Path = typer.Option(None, help="Path to the output dataset"),
    batch_size: int = typer.Option(16, help="Chunk size for grouping texts"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
    # -----------------------------------------
):
    # Imports
    from transformers import AlbertForSequenceClassification

    accelerator = Accelerator()
    with accelerator.main_process_first():
        debug_callback(debug)
        
        logger.info("Set model from pretrained ...")
        model = AlbertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, pretrained_model_name, 'final'), 
                                                                attn_implementation="sdpa")
        model_num_parameters = model.num_parameters() / 1_000_000
        logger.info(f"'Custom Genomics AlBERT number of parameters: {round(model_num_parameters)}M'")
        logger.info("Original ALBERT number of parameters: 11M")
        logger.info("Original BERT number of parameters: 110M")
        model = torch.compile(model, backend="inductor", mode="max-autotune", fullgraph=True)
        # TODO: arg.load_from_cache:
        logger.info("Loading and processing dataset")
        processed_dataset = load_from_disk(os.path.join(output_path, 'short_reads', 'tokenized'))
        if debug_mode:
            logger.debug("Downsampling pretokenized dataset")
            dataset_size = 1_000
            processed_dataset = processed_dataset.select(np.random.randint(len(processed_dataset), size=dataset_size))
            pass

        logger.info("Loading pretrained tokenizer")
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

        data_collator = DataCollatorWithPadding(tokenizer)
    # Prepare model and dataloader with accelerator
    model, processed_dataset = accelerator.prepare(model, processed_dataset)
    # Collect logits scores
    class_scores = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(processed_dataset.iter(batch_size=batch_size), desc=f"classifying {accelerator.process_index}", total=len(processed_dataset)):
            batch = data_collator(batch)
            logits = model(**batch).logits
            class_scores.append(logits.cpu().numpy())

    # Gather results
    accelerator.wait_for_everyone()
    class_scores = accelerator.gather(class_scores)

    with accelerator.main_process_first():
        with open(os.path.join(output_path, 'accelerate_class_scores.pkl'), 'wb') as f:
            pickle.dump(class_scores, f)
        logger.info(f"Quantification finished. Saved in '{output_path}'")
    pass

if __name__ == "__main__":
    app()