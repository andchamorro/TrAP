#!/bin/bash
#SBATCH --job-name=trap_00_env_check
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Pre-flight environment check — run before any pipeline stage.
# Verifies: correct conda env, package versions, and cross-stage
# standardization consistency (GenomeDataset vs sequence.standardize).
#
# Usage:
#   sbatch scripts/slurm/00_env_check.slurm
#
# Exit code 0 = all checks passed.  Any non-zero exit blocks downstream
# afterok dependencies.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_cpu_modules
activate_trap

python - <<'EOF'
import sys
import importlib

# --- package versions -------------------------------------------------------
import trap
print(f"trap.__version__ = {getattr(trap, '__version__', 'unknown')}")

_required = [
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
]
_optional = ["polars"]

for pkg in _required:
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg}: {m.__version__}")
    except ImportError as e:
        print(f"  ERROR: required package '{pkg}' not found: {e}", file=sys.stderr)
        sys.exit(1)

for pkg in _optional:
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg}: {m.__version__}  (optional, present)")
    except ImportError:
        print(f"  {pkg}: not installed  (optional, missing)")

# --- cross-stage standardization consistency --------------------------------
from trap.loaders.dataset import GenomeDataset
from trap.utils.sequence import standardize

_test_inputs = ["ACTGnN", "actgnnn", "NNNN", "", "ACGT", "ACXTYGRN"]

for s in _test_inputs:
    ds_out = GenomeDataset._standardization(s)
    seq_out = standardize(s)
    if ds_out != seq_out:
        print(
            f"  ERROR: standardization MISMATCH on {s!r}: "
            f"dataset={ds_out!r} vs sequence={seq_out!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if any(c not in "ACTG" for c in seq_out):
        print(
            f"  ERROR: non-ACTG char in cleaned output for {s!r}: {seq_out!r}",
            file=sys.stderr,
        )
        sys.exit(1)

print("  standardization: OK (ACTG-only, both paths agree)")
print("[00_env_check] ALL CHECKS PASSED")
EOF

echo "[00_env_check] PASS"
