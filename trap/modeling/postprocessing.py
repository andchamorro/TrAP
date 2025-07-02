import os
import gzip
import pickle
from collections import defaultdict
from pathlib import Path

import typer
from Bio import bgzf, SeqIO
from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Filter read IDs based on class scores from trap quantification.")

def sanitize_paths(*args):
    """
    Convert input arguments to pathlib.Path objects if possible.
    """
    paths = []
    for arg in args:
        if isinstance(arg, Path) or arg is None:
            paths.append(arg)
        else:
            try:
                paths.append(Path(arg))
            except Exception as e:
                raise ValueError(f"Invalid path argument: {arg}") from e
    return paths

def genome_file_handle(file_path):
    """
    Open FASTQ file with appropriate decompression based on extension.
    """
    file_path, = sanitize_paths(file_path)
    if file_path.suffix == '.gz':
        return gzip.open(file_path, 'rt')
    elif file_path.suffix == '.bgz':
        return bgzf.open(file_path, 'rt')
    else:
        return open(file_path, 'rt')

def try_mkdir(path):
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)

def processing_ids_from_fastq(input_file: Path, num_workers: int = 4):
    """
    Extract read IDs from a FASTQ file using BioPython.
    """
    input_file, = sanitize_paths(input_file)

    def generator_from_iterator():
        with genome_file_handle(input_file) as r1_handle:
            for r1 in SeqIO.parse(r1_handle, "fastq"):
                yield {'id': str(r1.id)}

    logger.info("Creating dataset from FASTQ file")
    return Dataset.from_generator(generator_from_iterator, num_proc=num_workers)

def processing_ids_from_list(id_file: Path):
    """
    Load read IDs from a plain text file.
    """
    id_file, = sanitize_paths(id_file)
    with open(id_file, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({'id': ids})

@app.command()
def filter_ids(
    fastq: Path = typer.Option(None, help="Input FASTQ file (compressed or uncompressed)"),
    id_list: Path = typer.Option(None, help="Text file with list of read IDs"),
    output_path: Path = typer.Option(..., help="Path to directory containing class_scores.pkl"),
    threshold: float = typer.Option(0.5, help="Threshold for filtering NEGATIVE class score"),
    output_filtered_ids: Path = typer.Option(None, help="Optional output file to save filtered IDs"),
    num_workers: int = typer.Option(4, help="Number of workers for dataset processing")
):
    """
    Filter read IDs based on NEGATIVE class score from trap quantification.
    """
    if not fastq and not id_list:
        typer.echo("Error: You must provide either --fastq or --id-list.")
        raise typer.Exit(code=1)

    if fastq:
        processed_ids = processing_ids_from_fastq(fastq, num_workers=num_workers)
    else:
        processed_ids = processing_ids_from_list(id_list)

    output_path, = sanitize_paths(output_path)
    class_scores_file = output_path / 'class_scores.pkl'

    if not class_scores_file.exists():
        typer.echo(f"Error: class_scores.pkl not found in {output_path}")
        raise typer.Exit(code=1)

    with open(class_scores_file, 'rb') as f:
        scores = pickle.load(f)

    class_scores = Dataset.from_dict({"scores": tqdm(scores, desc="Loading class scores")})

    if len(processed_ids) != len(class_scores):
        raise ValueError(f"Length mismatch: {len(processed_ids)} IDs vs {len(class_scores)} class scores")

    class_scores_id = concatenate_datasets([processed_ids, class_scores], axis=1)

    def get_scores(examples):
        results = defaultdict(list)
        results['id'] = examples['id']
        for example in examples['scores']:
            for e in example:
                results[e['label']].append(e['score'])
        return results

    class_scores_id = class_scores_id.map(get_scores, remove_columns=['id', 'scores'], batched=True, num_proc=num_workers)

    df = class_scores_id.to_pandas().set_index('id')
    filtered = df[df['NEGATIVE'] < threshold]

    if output_filtered_ids:
        with open(output_filtered_ids, 'w') as f:
            f.write('\n'.join(filtered.index))
        typer.echo(f"Filtered IDs saved to {output_filtered_ids}")
    else:
        typer.echo("\n".join(filtered.index))

if __name__ == "__main__":
    app()