#!/bin/bash
# Shared resolver for the synthetic L1-insertion benchmark's canonical directories.
#
# The two synthetic reference trees are named by EXPERIMENT TYPE (derived from SIM_MODEL),
# not by generation parameters:
#
#   SIM_MODEL=transcript  (model 2)  experiment "l1-transcript-pool"  standalone L1 pool, no host background
#   SIM_MODEL=insert      (model 1)  experiment "l1-host-insert"      full-length L1 spliced into host chr1
#
# The layout groups both trees under data/ref/synthetic/ and mirrors results under
# results/synthetic_validation/<experiment>/. The l1source (l1base / rm) stays a trailing
# qualifier so both sources coexist; withdel/chr1 are fixed for this benchmark and are
# recorded in the manifest instead of the directory name.
#
# Every synthetic script sources this file and calls experiment_dir / experiment_results_dir
# so the mapping lives in ONE place (CLAUDE.md: do not duplicate). Sourcing idiom:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_synthetic_paths.sh"
#   OUTPUT_DIR="${OUTPUT_DIR:-$(experiment_dir "${SIM_MODEL}" "${L1_SOURCE}" "${CHR}")}"

# experiment_token SIM_MODEL -> canonical experiment token.
experiment_token() {
    case "${1:-transcript}" in
        insert)     printf 'l1-host-insert' ;;
        transcript) printf 'l1-transcript-pool' ;;
        *) echo "[synthetic_paths] ERROR: unknown SIM_MODEL='${1:-}' (expected transcript|insert)" >&2; return 1 ;;
    esac
}

# experiment_dir SIM_MODEL L1_SOURCE CHR -> reference directory (relative to repo root).
# Falls back to the pre-refactor directory name when the new tree is absent, so existing
# Grace trees keep working until they are moved (see docs/guides/synthetic_validation.md).
experiment_dir() {
    local sim_model="${1:-transcript}" l1_source="${2:-l1base}" chr="${3:-chr1}"
    local token; token="$(experiment_token "${sim_model}")" || return 1
    local new="data/ref/synthetic/${token}.${l1_source}"
    local mtag=""; [[ "${sim_model}" == "insert" ]] && mtag=".insert"
    local legacy="data/ref/GRCh38.p14.genome.${chr}.withdel.${l1_source}${mtag}"
    if [[ ! -d "${new}" && -d "${legacy}" ]]; then printf '%s' "${legacy}"; else printf '%s' "${new}"; fi
}

# experiment_results_dir SIM_MODEL -> results/synthetic_validation/<experiment token>.
experiment_results_dir() {
    local token; token="$(experiment_token "${1:-transcript}")" || return 1
    printf 'results/synthetic_validation/%s' "${token}"
}
