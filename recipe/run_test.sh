#!/usr/bin/env bash

# Advanced tests for TrAP conda package
# This script runs after the basic import tests

set -euo pipefail

echo "Running advanced TrAP tests..."

# Test 1: Import all submodules
echo "Testing submodule imports..."
python -c "
from trap.config import config
from trap.loaders import dataset, tokenizer
from trap.modeling import train, predict, postprocessing
from trap.utils import io, kmer, dna2bit
print('✓ All submodules imported successfully')
"

# Test 2: Verify utility functions are available
echo "Testing utility functions..."
python -c "
from trap.utils.io import try_mkdir, genome_file_handle
from trap.utils.kmer import kmer_split, seq_to_encoded
print('✓ Utility functions accessible')
"

# Test 3: Verify CLI entry points exist
echo "Testing CLI entry points..."
command -v trap-train >/dev/null 2>&1 || { echo "ERROR: trap-train not found"; exit 1; }
command -v trap-predict >/dev/null 2>&1 || { echo "ERROR: trap-predict not found"; exit 1; }
command -v trap-postprocess >/dev/null 2>&1 || { echo "ERROR: trap-postprocess not found"; exit 1; }
echo "✓ All CLI commands available"

# Test 4: Check dependencies are importable
echo "Testing critical dependencies..."
python -c "
import torch
import transformers
import datasets
import accelerate
from Bio import SeqIO
import pysam
import sentencepiece
print('✓ All critical dependencies available')
"

# Test 5: Basic functionality test
echo "Testing basic functionality..."
python -c "
from trap.utils.kmer import kmer_split
seq = 'ACTGACTGACTG'
result = kmer_split(3, seq)
assert len(result.split()) == 10, 'kmer_split failed'
print('✓ Basic functionality works')
"

echo "All advanced tests passed!"
