#!/usr/bin/env bash
# Route N, QUERY side -- fetch the three VM-Cub-1 DMSO replicates with SRA Tools,
# ON THE LOGIN NODE.
#
# WHY THIS IS NOT A SLURM JOB AND NOT wget
# Two standing rules for this project, both adopted after the 2026-10-15 fetch went
# wrong:
#   1. Use the specialised tool when there is one.  `prefetch` verifies the archive it
#      downloads against the SRA's own internal checksums and resumes a partial
#      transfer from its own recorded state, instead of trusting whatever byte length
#      is on disk the way `wget -c` does.  That exact `wget -c` behaviour corrupted all
#      three mate-1 files earlier today: each ended at precisely ENA's declared byte
#      count and each had the wrong md5, so a size check would have passed all three.
#   2. Network transfers belong on the host (login) node, not on compute nodes.  The
#      download is a network wait, so paying for reserved cores while it happens is
#      waste -- the cancelled-and-restarted wget arrays cost ~20 SU to transfer nothing
#      usable.  Run here, the transfer costs **zero SU**.
# Everything CPU-bound -- vdb-validate, fasterq-dump, the depth-match -- stays in
# query_dump_match.slurm.  Not because the login node lacks cpus: it has 48 and full
# affinity.  `nproc` prints 1 here only because Grace exports OMP_NUM_THREADS=1 and GNU
# nproc honours it -- do not read that as a cgroup limit, and do not size anything off
# it.  CPU work belongs in a job as a matter of policy.
#
# PARALLELISM IS CAPPED AT 8 PROCESSES.  Three concurrent prefetches, nothing else.
#
# No charge account is involved: nothing is submitted.
#
#   bash scripts/analyses/5race_substitution/route_n_l1rip/query_prefetch.sh            # foreground
#   setsid nohup bash scripts/analyses/5race_substitution/route_n_l1rip/query_prefetch.sh > logs/prefetch.log 2>&1 &
set -uo pipefail

SRA_BIN=/scratch/user/achamorro/.local/sratoolkit.3.4.1-ubuntu64/bin
export PATH="$SRA_BIN:$PATH"

WORK=/scratch/user/achamorro/analysis/5race_route_n_query
CACHE="$WORK/sra_cache"          # untrusted: downloaded archives, nothing run from here
mkdir -p "$CACHE"

ACCS=(SRR14758878 SRR14758879 SRR14758880)

# Hard cap on concurrency, per the project rule.  Three accessions is already under it;
# the cap is written down so a longer list cannot quietly exceed it.
MAX_PARALLEL=3

echo "[$(date)] prefetch $(prefetch --version 2>&1 | tr -d '\n') on $(hostname), nproc=$(nproc)"

fetch_one() {
    local acc="$1"
    local dir="$CACHE/$acc"
    if [[ -s "$dir/$acc.sra" ]]; then
        echo "[$(date)] $acc : already present, re-verifying"
    else
        echo "[$(date)] $acc : prefetch -> $dir"
    fi
    # --verify yes is the default and is what makes this different from wget: prefetch
    # checks the archive against the SRA's own checksums and says so.  --max-size is
    # raised because these runs are ~4-5 GB each.
    # NO -p: the progress flag emits a percentage per 0.01 %, which turned a 4 GB
    # transfer into a 188 KB log of nothing.  Progress is read off the growing
    # .sra.tmp instead (`du -sh` on the cache dir).
    prefetch --verify yes --max-size 50G -O "$CACHE" "$acc" \
        || { echo "[$(date)] $acc : prefetch FAILED" >&2; return 1; }
    echo "[$(date)] $acc : prefetch reported success"
    ls -la "$dir"
}

running=0
pids=()
accs_running=()
for acc in "${ACCS[@]}"; do
    fetch_one "$acc" > "$CACHE/$acc.prefetch.log" 2>&1 &
    pids+=($!)
    accs_running+=("$acc")
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "[$(date)] ${accs_running[$idx]} : FAILED (see $CACHE/${accs_running[$idx]}.prefetch.log)" >&2
        fail=1
    fi
done

echo
echo "=== per-accession prefetch logs ==="
for acc in "${ACCS[@]}"; do
    echo "--- $acc"
    tail -4 "$CACHE/$acc.prefetch.log" 2>/dev/null
done

echo
echo "=== archives on disk ==="
for acc in "${ACCS[@]}"; do
    f="$CACHE/$acc/$acc.sra"
    if [[ -s "$f" ]]; then
        printf '%s\t%s bytes\n' "$acc" "$(stat -c %s "$f")"
    else
        printf '%s\tMISSING\n' "$acc"; fail=1
    fi
done

if (( fail )); then
    echo "[$(date)] one or more accessions did not land -- do NOT submit query_dump_match.slurm yet" >&2
    exit 1
fi
echo "[$(date)] all three archives present. Next: query_dump_match.slurm (vdb-validate -> fasterq-dump -> depth-match)"
