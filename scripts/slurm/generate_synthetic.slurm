#!/bin/bash
#SBATCH --job-name=trap_generate_synthetic
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Generate the synthetic L1-insertion abundance benchmark (scripts/sh/generate_synthetic_dataset.sh).
# CPU job (ART + bedtools + salmon-index + BioPython). Resumable: cells whose reads
# already exist are skipped (SKIP_EXISTING=1), so a re-submit after a timeout continues.
#
#   sbatch scripts/slurm/generate_synthetic.slurm
#
# Options (env, pass-through to the generator):
#   POWERS="5 6 7 8 9 10 11 12 13"      insertion levels (2^power)
#   DELPROBS="0.000 0.025 0.050 0.075 0.100"   deletion probabilities
#   FCOV=5                              ART coverage
#   SEED=3469                           determinism
#   L1_MIN_LEN=5500  L1_MAX_LEN=6500    full-length L1 element filter (bp)
#   CHR=chr1                            chromosome
#   OUTPUT_DIR=data/ref/GRCh38.p14.genome.chr1.withdel
#   CHR1_TRANSCRIPTS=<fa>              skip the GENCODE_GTF chr-subset step
#   GENCODE_GTF=<gtf(.gz)>            gencode GTF for the chr-transcript subset
#   SKIP_EXISTING=1                    resume (set 0 to force-regenerate all)
#   KEEP_FASTA=0                       keep the (large) modified-transcript FASTAs
#   DRY_RUN=1                          print the resolved config + grid, generate nothing
#
#   e.g. quick subset:  POWERS="8" DELPROBS="0.025 0.100" sbatch scripts/slurm/generate_synthetic.slurm

export TRAP_VERBOSITY="${TRAP_VERBOSITY:-detailed}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/_common.sh"
load_bio_modules
activate_trap
module load "${SEQKIT_MODULE:-SeqKit/2.9.0}" 2>/dev/null || echo "[gen] seqkit module not loaded — must be on PATH"
[[ -n "${ART_MODULE:-}" ]] && { module load "${ART_MODULE}" 2>/dev/null || true; }

# Tool sanity (generic — names only).
for t in art_illumina seqkit bedtools samtools salmon python; do
    command -v "${t}" >/dev/null || echo "[gen] WARNING: ${t} not on PATH"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[gen] DRY_RUN — resolved config:"
    echo "  POWERS='${POWERS:-5 6 7 8 9 10 11 12 13}'  DELPROBS='${DELPROBS:-0.000 0.025 0.050 0.075 0.100}'"
    echo "  FCOV=${FCOV:-5}  SEED=${SEED:-3469}  CHR=${CHR:-chr1}  SKIP_EXISTING=${SKIP_EXISTING:-1}"
    echo "  L1_MIN_LEN=${L1_MIN_LEN:-5500}  L1_MAX_LEN=${L1_MAX_LEN:-6500}"
    echo "  OUTPUT_DIR=${OUTPUT_DIR:-data/ref/GRCh38.p14.genome.chr1.withdel}"
    exit 0
fi

bash "${REPO_ROOT:-$(pwd)}/scripts/sh/generate_synthetic_dataset.sh"
