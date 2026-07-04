#!/bin/bash
# Shared helper: make an L1EM repo quantify against OUR synthetic L1 annotation.
#
# L1EM/run_L1EM.sh (and run_MLL1EM.sh) HARDCODE their reference to
# $L1EM_PATH/annotation/L1EM.400.{bed,fa} plus the bwa index built from it — they IGNORE any
# BED passed on the command line. So the loci reads are quantified against are whatever lives
# there. If that is L1EM's stock 400-consensus set (or an index built from the wrong BED), no
# candidate alignment matches the synthetic UID L1 → full_counts.txt is empty (README: it only
# lists loci "with any aligned read pairs"), or is named by subfamily and never joins.
#
# This installs our BED as that annotation and rebuilds the fasta+index when it changed, so
# full_counts.txt reports the synthetic loci (UID107.1.chr1:...) that join the ground truth.
#
# The BED MUST be in L1EM's `family.category.locus.strand` form (col4 e.g.
# `UID107.1.chr1:71514617-71519522.+`) — NOT the generator's hsflil1_8438.bed, whose col4 is
# `UID-107` (that one builds the salmon index / ground truth). Keep them as separate files:
# the L1EM annotation lives at data/ref/l1base/hsflil1_8438.l1em.bed (canonical copy in
# .trap/L1EM/annotation/hsflil1_8438.bed). The format check below catches a wrong-BED mixup,
# which is what produced the earlier empty full_counts.txt.
ensure_l1em_reference() {
    local bed="$1" l1em_path="$2" genome="$3" tag="${4:-l1em}"
    local abs_bed abs_genome ann_bed c4
    abs_bed="$(realpath "${bed}")"; abs_genome="$(realpath "${genome}")"
    ann_bed="${l1em_path}/annotation/L1EM.400.bed"
    c4="$(awk -F'\t' 'NR==1{print $4; exit}' "${abs_bed}")"
    case "${c4}" in
        *.*) : ;;   # family.category.locus.strand → good
        *) echo "[${tag}] ERROR: ${bed} col4='${c4}' is not L1EM format (need" \
                "family.category.locus.strand, e.g. UID107.1.chr1:71514617-71519522.+)." \
                "This looks like the generator's UID-NNN BED — set L1EM_BED to the L1EM" \
                "annotation (.trap/L1EM/annotation/hsflil1_8438.bed). See .trap/L1EM/README.md." >&2
           return 1 ;;
    esac
    if [[ ! -f "${ann_bed}" ]] || ! cmp -s "${abs_bed}" "${ann_bed}"; then
        echo "[${tag}] installing L1EM annotation from ${bed} + rebuilding bwa index"
        cp "${abs_bed}" "${ann_bed}" || return 1
        ( cd "${l1em_path}" && bash generate_L1EM_fasta_and_index.sh "${abs_genome}" ) || {
            echo "[${tag}] ERROR: L1EM reference (re)build failed — is the genome bwa-indexed?" >&2
            return 1
        }
    fi
}
