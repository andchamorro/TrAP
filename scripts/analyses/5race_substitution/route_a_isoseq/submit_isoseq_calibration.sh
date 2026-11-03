#!/usr/bin/env bash
# Submit an Iso-Seq calibration job with the charge account applied from config.
#
# WHY A WRAPPER RATHER THAN A #SBATCH DIRECTIVE
# The account number is sensitive and must never enter git, so no tracked script
# may carry `#SBATCH --account=<value>`.  It is read from the git-ignored
# config/hpc/grace.yaml (where `slurm.account` already lives, beside
# `slurm.mail_user`) and exported as SBATCH_ACCOUNT, which sbatch honours as the
# default for --account.  Tracked files reference the variable, never the value.
# Convention adopted from Sertorius scripts/hpc/submit.sh.
#
#   bash scripts/analyses/5race_substitution/route_a_isoseq/submit_isoseq_calibration.sh \
#        scripts/analyses/5race_substitution/route_a_isoseq/isoseq_calibration_align_count.slurm
#   bash scripts/analyses/5race_substitution/route_a_isoseq/submit_isoseq_calibration.sh --array=0-3 \
#        scripts/analyses/5race_substitution/route_a_isoseq/isoseq_calibration_fetch.slurm
set -euo pipefail

HPC_CONFIG="${HPC_CONFIG:-config/hpc/grace.yaml}"
[ -s "$HPC_CONFIG" ] || { echo "[hpc] missing $HPC_CONFIG" >&2; exit 1; }

# Minimal scalar lookup under the `slurm:` block; avoids a YAML dependency on a
# login node, the same reason hpc_config.py keeps a stdlib fallback.
_yaml_slurm_value() {
    awk -v key="$1" '
        /^slurm:/ {inblock=1; next}
        /^[^[:space:]#]/ {inblock=0}
        inblock && $1 == key":" {
            sub(/^[[:space:]]*[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")
            gsub(/^"|"$/, "")
            print; exit
        }' "$HPC_CONFIG"
}

ACCOUNT="$(_yaml_slurm_value account)"
if [ -z "$ACCOUNT" ]; then
    echo "[hpc] no slurm.account in $HPC_CONFIG." >&2
    echo "[hpc]   Submitting without it would silently charge the default allocation." >&2
    exit 1
fi
export SBATCH_ACCOUNT="$ACCOUNT"
# Masked so the number never reaches a terminal transcript or a pasted log.
echo "[hpc] charging account: ${ACCOUNT:0:4}…${ACCOUNT: -3}" >&2

# Mail MUST be passed as CLI flags: sbatch reads SBATCH_ACCOUNT from the
# environment but NOT SBATCH_MAIL_USER/SBATCH_MAIL_TYPE, so exporting those
# silently sends no notification at all (established in Sertorius).
MAIL_ARGS=()
MAIL_USER="$(_yaml_slurm_value mail_user)"
if [ -n "$MAIL_USER" ]; then
    MAIL_ARGS=(--mail-user="$MAIL_USER" --mail-type="$(_yaml_slurm_value mail_type || true)")
    [ -n "${MAIL_ARGS[1]#--mail-type=}" ] || MAIL_ARGS[1]="--mail-type=END,FAIL"
    echo "[hpc] job mail: on (address from $HPC_CONFIG)" >&2
    echo "[hpc]   verify with: scontrol show job <id> | grep -i mail" >&2
fi

exec sbatch ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} "$@"
