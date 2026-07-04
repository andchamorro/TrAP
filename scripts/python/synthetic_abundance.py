#!/usr/bin/env python3
"""Build the synthetic-benchmark abundance table for the R/ggplot analysis (5.02).

For each grid cell (insertion level × del_prob) it pairs, per L1 element:
  * Simulated  = ground-truth abundance — the per-element copy count from a model-2
    transcript-pool counts TSV, or the insertion count from a model-1 BED.
  * AlbertSalmon_seqlabel = salmon NumReads from the seqlabel filter→salmon quant.sf.

The element key (an L1Base ``UID`` or an RM ``<subfamily>.<flag>::chrom:start-end``) is
shared by the ground truth, the salmon index, and quant.sf, so the join is 1:1. A subfamily
token is emitted too, for aggregation in the R step.

Emits a long CSV: ``power, del_prob, l1_id, subfamily, simulated, albertsalmon_seqlabel``.

``Simulated`` is the **exact** injected copy count (a strict improvement over 4.08, which
re-estimated it from the reads BAM). Method values are **raw** estimator outputs by
default; pass ``--normalize`` to instead apply 4.08's per-element normalization
(``0.5*count / (effective_length*coverage / insert_size_average)``) to the count-based
methods (salmon/TEtranscripts/HTseq), using ``bedtools coverage`` of the genome STAR BAM
over the genomic L1 reference BED (``<refdir>/_inputs/l1_fulllength.bed`` — coordinate-
consistent with the BAM, unlike the transcript-coordinate per-cell insertion BED) and
``samtools stats`` for the insert size. Use ``--normalize`` when the
**per-element** figure must match 4.08's methodology; the raw default suffices for the
exact ground truth and the total-level comparison.

``abundance`` (and ``albertsalmon_seqlabel``) is **always raw**; ``--normalize`` *adds* a
parallel ``abundance_norm`` (``albertsalmon_seqlabel_norm``) column rather than replacing
it, so one normalized run serves both the raw total-abundance figures and the normalized
per-element figure. ``--normalize`` also emits ``simulated_reads`` — the empirical 4.08
``Simulated`` reference (``0.25 * bedtools_count / expected_reads`` per element) — which is
the x-axis the legacy per-element regression grid (``synthetic_salmon_l1em_teht_some``) is
plotted against (vs the exact integer copy count in ``simulated``, used for the totals).

  python scripts/python/synthetic_abundance.py                     # raw (default)
  python scripts/python/synthetic_abundance.py --normalize         # legacy 4.08 per-element abundance
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter


def load_simulated_counts(tsv_path: str) -> dict:
    """Per-element copy count from a model-2 counts TSV (``l1_id\\tcount``)."""
    counts = {}
    with open(tsv_path) as fh:
        next(fh, None)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                counts[f[0]] = int(f[1])
    return counts


def load_simulated_bed(bed_path: str) -> Counter:
    """Per-element insertion count from a model-1 insertion BED (col4 = element key)."""
    counts: Counter = Counter()
    with open(bed_path) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                counts[fields[3]] += 1
    return counts


def load_salmon(quant_sf: str) -> dict:
    """``{element_key: NumReads}`` from a salmon quant.sf."""
    est = {}
    with open(quant_sf) as fh:
        next(fh, None)  # header: Name Length EffectiveLength TPM NumReads
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 5:
                est[f[0]] = float(f[4])
    return est


def _norm_uid(key: str) -> str:
    """Normalize an alignment-method id to the ground-truth UID key (``UID107``)."""
    return key.split(":", 1)[0].split(".", 1)[0].replace("UID-", "UID")


def load_l1em(full_counts: str) -> dict:
    """``{uid: abundance}`` from an L1EM / AlbertEM full_counts.txt.

    Abundance = sum of the sense categories (only + 3prunon + passive_sense +
    passive_antisense), grouped by family (first token of the name column), per 4.08.
    """
    est: dict = {}
    with open(full_counts) as fh:
        header = next(fh, "").rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        cats = [c for c in ("only", "3prunon", "passive_sense", "passive_antisense") if c in idx]
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if not f or not cats or len(f) <= max(idx[c] for c in cats):
                continue
            uid = _norm_uid(f[0])
            est[uid] = est.get(uid, 0.0) + sum(float(f[idx[c]]) for c in cats)
    return est


def load_tecount(cnt_table: str) -> dict:
    """``{uid: count}`` from a TEcount .cntTable (LINE-1 UID features only)."""
    est: dict = {}
    with open(cnt_table) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0].startswith("UID-"):
                est[_norm_uid(f[0])] = est.get(_norm_uid(f[0]), 0.0) + float(f[1])
    return est


def load_htseq(counts_csv: str) -> dict:
    """``{uid: count}`` from an htseq-count table (drops the ``__`` summary rows)."""
    est: dict = {}
    with open(counts_csv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and not f[0].startswith("__"):
                est[_norm_uid(f[0])] = float(f[1])
    return est


# --- optional legacy 4.08 per-element normalization ---------------------------
# 4.08 turned a method's raw read count into an abundance via
#   abundance = k * count / (effective_length * coverage / insert_size_average)
# with the effective_length/coverage taken (method-independently) from bedtools coverage
# of the STAR BAM over the insertion BED, and insert_size_average from samtools stats.
# Only the count-based methods were normalized; L1EM/AlbertEM (EM/FPM) were left raw.
NORMALIZE_METHODS = {"AlbertSalmon_seqlabel", "Salmon", "TEtranscripts", "HTseq"}
NORMALIZE_K = 0.5  # 4.08 estimate_abundance() constant for method columns


def _check_tools(*tools):
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"[abundance] --normalize needs {', '.join(missing)} on PATH")


def fragment_insert_size(bam: str, default: float = 500.0) -> float:
    """``insert size average`` from ``samtools stats`` (4.08 get_fragment_stats)."""
    try:
        out = subprocess.run(["samtools", "stats", bam], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return default
    for line in out.splitlines():
        if line.startswith("SN") and "insert size average:" in line:
            try:
                return float(line.split("\t")[2]) or default
            except (IndexError, ValueError):
                break
    return default


def _bam_genome_file(bam: str, path: str) -> None:
    """Write a bedtools genome file (``chrom\\tlength``, header order) from the BAM @SQ lines."""
    hdr = subprocess.run(["samtools", "view", "-H", bam],
                         capture_output=True, text=True, check=True).stdout
    with open(path, "w") as fo:
        for ln in hdr.splitlines():
            if ln.startswith("@SQ"):
                sn = length = None
                for tok in ln.split("\t"):
                    if tok.startswith("SN:"):
                        sn = tok[3:]
                    elif tok.startswith("LN:"):
                        length = tok[3:]
                if sn and length:
                    fo.write(f"{sn}\t{length}\n")


def coverage_features(bed: str, bam: str) -> dict:
    """``{element_key: (effective_length, coverage)}`` via bedtools coverage of the BAM
    over the **genomic L1 reference BED** (4.08 abundance_line1_reads), per element key.

    The BED must be in the BAM's coordinate system: the model-1 per-cell insertion BED is
    in transcript coordinates (col1 = ENST…), but the STAR BAM is genome-aligned, so the
    coverage reference is the genomic L1 element BED (``<refdir>/_inputs/l1_fulllength.bed``),
    whose col4 is the element key (``UID-NNN`` normalised to ``UIDNNN`` to match salmon /
    the ground truth; RM ``subfamily.flag::coords`` kept as-is).

    Uses the ``-sorted`` sweeping algorithm with a genome file taken from the BAM header:
    the default (unsorted) algorithm loads the whole BAM into an interval tree and OOMs on
    a full STAR BAM. A clean 4-column BED (chrom,start,end,name), sorted into the BAM's
    chromosome order, is written so the appended coverage columns land at fixed indices.
    """
    with tempfile.TemporaryDirectory() as td:
        raw4 = os.path.join(td, "elements.raw.bed4")
        bed4 = os.path.join(td, "elements.bed4")
        genome = os.path.join(td, "genome.txt")
        with open(bed) as fi, open(raw4, "w") as fo:
            for ln in fi:
                c = ln.rstrip("\n").split("\t")
                if len(c) >= 4:
                    fo.write("\t".join([c[0], c[1], c[2], c[3].replace("UID-", "UID")]) + "\n")
        _bam_genome_file(bam, genome)
        # sort -a into the BAM's chromosome order so -sorted can co-sweep both.
        with open(bed4, "w") as fo:
            subprocess.run(["bedtools", "sort", "-i", raw4, "-g", genome], stdout=fo, check=True)
        base_cmd = ["bedtools", "coverage", "-sorted", "-g", genome, "-a", bed4, "-b", bam]
        cov = subprocess.run(base_cmd, capture_output=True, text=True, check=True).stdout
        men = subprocess.run(base_cmd + ["-mean"], capture_output=True, text=True, check=True).stdout
    # coverage default appends: [count, covered_bases, length, fraction] → count = col[4],
    # covered_bases (effective length) = col[5]. The raw count is kept so the empirical 4.08
    # "Simulated" reference (abundance_line1_reads) can be reconstructed downstream.
    eff: dict = {}
    for ln in cov.splitlines():
        c = ln.split("\t")
        if len(c) >= 7:
            e = eff.setdefault(c[3], [0, 0, 0]); e[0] += int(c[4]); e[1] += int(c[5]); e[2] += 1
    # -mean appends a single mean-depth column → col[4].
    dep: dict = {}
    for ln in men.splitlines():
        c = ln.split("\t")
        if len(c) >= 5:
            d = dep.setdefault(c[3], [0.0, 0]); d[0] += float(c[4]); d[1] += 1
    feats = {}
    for k, (cnt, s, n) in eff.items():
        efflen = s / max(n, 1)
        coverage = dep[k][0] / max(dep[k][1], 1) if k in dep else 0.0
        feats[k] = (cnt, efflen, coverage)
    return feats


def _expected_reads(bundle, key: str) -> float:
    """4.08 denominator: ``(effLen * coverage) / insert_size`` for an element (0 if absent)."""
    insert_size, feats = bundle
    _cnt, efflen, coverage = feats.get(key, (0.0, 0.0, 0.0))
    return (efflen * coverage) / insert_size if insert_size else 0.0


def _normalize(raw: float, bundle, key: str, k: float = NORMALIZE_K) -> float:
    """Estimate_abundance: ``k * raw / (effLen * coverage / insert_size)``."""
    den = _expected_reads(bundle, key)
    return k * raw / (den if den > 0 else 1.0)


# 4.08 abundance_line1_reads() used k=0.25 on the bedtools read count of the *element* to
# define the "Simulated" reference the per-element regression is plotted against.
SIMULATED_READS_K = 0.25


def legacy_simulated_reads(bundle, key: str) -> float:
    """Empirical 4.08 ``Simulated`` = ``0.25 * bedtools_count / expected_reads`` per element."""
    insert_size, feats = bundle
    cnt = feats.get(key, (0.0, 0.0, 0.0))[0]
    den = _expected_reads(bundle, key)
    return SIMULATED_READS_K * cnt / (den if den > 0 else 1.0)


# --- on-disk feature cache ----------------------------------------------------
# coverage_features() runs two bedtools passes over a whole-cell STAR BAM — the slow
# step. The (insert_size, per-element effLen/coverage) result depends only on the BAM +
# insertion BED, so it is cached next to the BAM under the reference tree and reused across
# runs. The SLURM array (submit_synthetic_abundance.sh) fills these in parallel; the
# aggregation then just reads them.
CACHE_NAME = "coverage_features.json"


def l1_reference_bed(refdir: str) -> str:
    """The genomic L1 element BED the salmon index was built from (shared by all cells).

    Coordinate-consistent with the genome STAR BAM (unlike the transcript-coordinate
    per-cell insertion BED), and its col4 keys join the ground truth / salmon quant.
    """
    return f"{refdir}/_inputs/l1_fulllength.bed"


def cell_paths(refdir: str, chrom: str, power: str, dp: str, base: str):
    """``(l1_reference_bed, star_bam, cache_json)`` for a grid cell (BED is shared)."""
    bam = f"{refdir}/star/{base}/Aligned.sortedByCoord.out.bam"
    return l1_reference_bed(refdir), bam, f"{refdir}/star/{base}/{CACHE_NAME}"


# Cache schema version. v2 stores per-element [count, effLen, coverage] (v1 lacked the read
# count, so the empirical 4.08 Simulated reference can't be rebuilt) — v1 caches are treated
# as stale and recomputed from the (unchanged) BAM.
CACHE_VERSION = 2


def load_feature_cache(cache: str, *deps: str):
    """Return ``(insert_size, features)`` from a fresh v2 cache, or None if absent/stale."""
    if not os.path.isfile(cache):
        return None
    try:
        cmt = os.path.getmtime(cache)
        if any(os.path.isfile(d) and cmt < os.path.getmtime(d) for d in deps):
            return None  # a dependency (BAM/BED) is newer → recompute
        with open(cache) as fh:
            d = json.load(fh)
        if d.get("v") != CACHE_VERSION:
            return None  # pre-count cache → recompute so Simulated can be reconstructed
        return float(d["insert_size"]), {k: (v[0], v[1], v[2]) for k, v in d["features"].items()}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def save_feature_cache(cache: str, bundle) -> None:
    insert_size, feats = bundle
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump({"v": CACHE_VERSION, "insert_size": insert_size,
                   "features": {k: [int(cnt), round(e, 4), round(c, 6)]
                                for k, (cnt, e, c) in feats.items()}}, fh)
    os.replace(tmp, cache)  # atomic — safe under the SLURM array


def compute_cell_features(refdir, chrom, power, dp, base, insert_default=500.0,
                          use_cache=True, refresh=False):
    """``(insert_size, features) | None`` for a cell, reading/writing the on-disk cache.

    None (no crash) when the STAR BAM or insertion BED is missing — model-1/insert only.
    """
    bed, bam, cache = cell_paths(refdir, chrom, power, dp, base)
    if not (os.path.isfile(bed) and os.path.isfile(bam)):
        return None
    if use_cache and not refresh:
        got = load_feature_cache(cache, bam, bed)
        if got is not None:
            return got
    bundle = (fragment_insert_size(bam, insert_default), coverage_features(bed, bam))
    if use_cache:
        try:
            save_feature_cache(cache, bundle)
        except OSError as exc:
            print(f"[abundance] WARN: could not write cache {cache}: {exc}")
    return bundle


def make_feature_getter(refdir: str, chrom: str, insert_default: float,
                        use_cache: bool = True, refresh: bool = False):
    """Lazy, in-memory-cached ``(power, dp, base) -> (insert_size, features) | None``,
    backed by the on-disk cache (model-1/insert only; None + one warning otherwise)."""
    mem: dict = {}
    warned: set = set()

    def get(power, dp, base):
        ckey = (power, dp)
        if ckey not in mem:
            bundle = compute_cell_features(refdir, chrom, power, dp, base,
                                           insert_default, use_cache, refresh)
            if bundle is None and ckey not in warned:
                bed, bam, _ = cell_paths(refdir, chrom, power, dp, base)
                miss = "L1 reference BED (_inputs/l1_fulllength.bed)" if not os.path.isfile(bed) else "STAR BAM"
                print(f"[abundance] WARN: --normalize needs {miss} for {power}/{dp} "
                      "(genomic L1 BED + star BAM) — leaving those methods raw")
                warned.add(ckey)
            mem[ckey] = bundle
        return mem[ckey]

    return get


def discover_cells(refdir: str, chrom: str):
    """Sorted ``(power, dp)`` grid cells that have a ground-truth BED or counts TSV."""
    cells = set()
    for pat in ("bed", "counts.tsv"):
        for p in glob.glob(f"{refdir}/GRCh38.p14.{chrom}.insert_level_*_delprob_*.{pat}"):
            m = re.search(r"insert_level_(\d+)_delprob_([0-9]+\.[0-9]+)", os.path.basename(p))
            if m:
                cells.add((m.group(1), m.group(2)))
    return sorted(cells)


# method name → (subdir, filename template relative to <refdir>/<subdir>/<base>/, loader).
# {base} = ...pair.<fcov>x, {proj} = ...insert_level_P_delprob_D (no .pair suffix).
METHODS = {
    "AlbertSalmon_seqlabel": ("salmon/filtered_seqlabel", "quant.sf", load_salmon),
    "Salmon": ("salmon/unfiltered", "quant.sf", load_salmon),
    "L1EM": ("L1EM", "full_counts.txt", load_l1em),
    "AlbertEM": ("MLEM", "full_counts.txt", load_l1em),
    "TEtranscripts": ("TEtranscripts", "{proj}.cntTable", load_tecount),
    "HTseq": ("HTseq", "htseq_counts.csv", load_htseq),
}


def subfamily(key: str) -> str:
    """Subfamily token from an element key.

    ``L1PA6.1::chr1:100-6100`` → ``L1PA6`` (RM source). A bare L1Base ``UID107`` has
    no subfamily encoded, so it returns itself (per-element == per-subfamily there).
    """
    return key.split("::", 1)[0].rsplit(".", 1)[0]


# Canonical synthetic directory resolver — mirror of scripts/sh/_synthetic_paths.sh so the
# SIM_MODEL → experiment-token mapping lives in one conceptual place (CLAUDE.md: do not
# duplicate). The two must stay in sync.
EXPERIMENT_TOKEN = {"transcript": "l1-transcript-pool", "insert": "l1-host-insert"}


def experiment_token(sim_model: str) -> str:
    """Canonical experiment token from SIM_MODEL (``transcript`` / ``insert``)."""
    try:
        return EXPERIMENT_TOKEN[sim_model]
    except KeyError:
        raise SystemExit(
            f"[abundance] ERROR: unknown SIM_MODEL='{sim_model}' (expected transcript|insert)"
        )


def experiment_dir(sim_model: str, l1_source: str, chrom: str) -> str:
    """Reference directory for an experiment, with a pre-refactor fallback.

    Returns the legacy ``GRCh38.p14.genome.<chr>.withdel.<src>[.insert]`` path only when the
    new ``data/ref/synthetic/<token>.<src>`` tree is absent, so unmoved Grace trees keep
    working (see docs/guides/synthetic_validation.md).
    """
    new = f"data/ref/synthetic/{experiment_token(sim_model)}.{l1_source}"
    mtag = ".insert" if sim_model == "insert" else ""
    legacy = f"data/ref/GRCh38.p14.genome.{chrom}.withdel.{l1_source}{mtag}"
    return legacy if (not os.path.isdir(new) and os.path.isdir(legacy)) else new


def experiment_results_dir(sim_model: str) -> str:
    """``results/synthetic_validation/<experiment token>`` for an experiment."""
    return f"results/synthetic_validation/{experiment_token(sim_model)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    _sim = os.environ.get("SIM_MODEL", "transcript")
    _src = os.environ.get("L1_SOURCE", "l1base")
    _chr = os.environ.get("CHR", "chr1")
    _rd = experiment_results_dir(_sim)
    ap.add_argument("--refdir",
                    default=os.environ.get("REFDIR", experiment_dir(_sim, _src, _chr)))
    ap.add_argument("--chr", default=_chr)
    ap.add_argument("--fcov", default="5")
    ap.add_argument("--out", default=f"{_rd}/abundance.csv")
    ap.add_argument("--methods-out", default=f"{_rd}/abundance_methods.csv",
                    help="long table of every baseline method that has outputs.")
    ap.add_argument("--normalize", action="store_true",
                    default=os.environ.get("NORMALIZE", "0") == "1",
                    help="apply the legacy 4.08 per-element normalization "
                         "(0.5*count / (effLen*coverage / insert_size)) to the count-based methods "
                         f"({', '.join(sorted(NORMALIZE_METHODS))}); needs samtools+bedtools and the "
                         "model-1/insert STAR BAMs + genomic insertion BEDs. Off by default (raw counts).")
    ap.add_argument("--insert-size-default", type=float, default=500.0,
                    help="fallback insert size when samtools stats yields none.")
    ap.add_argument("--cache-only", action="store_true",
                    help="only (re)build the per-cell coverage caches under "
                         "<refdir>/star/<base>/, then exit — the parallel SLURM-array prep step. "
                         "Restrict to one cell with --only-power/--only-del-prob.")
    ap.add_argument("--refresh-cache", action="store_true",
                    default=os.environ.get("REFRESH_CACHE", "0") == "1",
                    help="recompute coverage caches even when present and fresh.")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read or write the on-disk coverage cache.")
    ap.add_argument("--only-power", default=os.environ.get("ONLY_POWER"),
                    help="restrict --cache-only to one insertion level (SLURM array single cell).")
    ap.add_argument("--only-del-prob", default=os.environ.get("ONLY_DELPROB"),
                    help="restrict --cache-only to one deletion probability.")
    args = ap.parse_args()

    # --- cache-only: build per-cell coverage caches in parallel, then exit -----
    if args.cache_only:
        _check_tools("samtools", "bedtools")
        built = missing = 0
        for power, dp in discover_cells(args.refdir, args.chr):
            if args.only_power and power != str(args.only_power):
                continue
            if args.only_del_prob and dp != str(args.only_del_prob):
                continue
            base = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}.pair.{args.fcov}x"
            bundle = compute_cell_features(args.refdir, args.chr, power, dp, base,
                                           args.insert_size_default,
                                           use_cache=not args.no_cache, refresh=args.refresh_cache)
            if bundle is None:
                missing += 1
                print(f"[abundance] cache-only: {power}/{dp} skipped (no STAR BAM / insertion BED)")
            else:
                built += 1
                print(f"[abundance] cache-only: {power}/{dp} -> {cell_paths(args.refdir, args.chr, power, dp, base)[2]}")
        print(f"[abundance] cache-only done: {built} built/verified, {missing} skipped")
        return

    feat_get = None
    if args.normalize:
        _check_tools("samtools", "bedtools")
        feat_get = make_feature_getter(args.refdir, args.chr, args.insert_size_default,
                                       use_cache=not args.no_cache, refresh=args.refresh_cache)
        print("[abundance] --normalize ON: legacy per-element effLen×coverage×insert-size "
              f"normalization for {sorted(NORMALIZE_METHODS)}")

    rows, cells = [], 0
    # Iterate over the validation outputs; pair each with its ground truth — a model-2
    # counts TSV (preferred) or a model-1 insertion BED.
    qdirs = sorted(glob.glob(f"{args.refdir}/salmon/filtered_seqlabel/*/"))
    for qdir in qdirs:
        base = os.path.basename(qdir.rstrip("/"))
        # Anchor del_prob as digits.digits so the trailing '.pair.5x' isn't captured.
        m = re.search(r"insert_level_(\d+)_delprob_([0-9]+\.[0-9]+)", base)
        quant_sf = os.path.join(qdir, "quant.sf")
        if not m or not os.path.isfile(quant_sf):
            continue
        power, dp = m.group(1), m.group(2)
        gt = f"{args.refdir}/GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}"
        if os.path.isfile(f"{gt}.counts.tsv"):
            sim = load_simulated_counts(f"{gt}.counts.tsv")
        elif os.path.isfile(f"{gt}.bed"):
            sim = load_simulated_bed(f"{gt}.bed")
        else:
            print(f"[abundance] WARN: no ground truth for {base} — skipping")
            continue
        est = load_salmon(quant_sf)
        cells += 1
        # AlbertSalmon_seqlabel is a count-based method → normalize when requested.
        bundle = feat_get(power, dp, base) if feat_get else None
        # Element set = the salmon index targets (quant.sf); Simulated is 0 where the
        # element was never expressed, so the regression includes true negatives.
        # `abundance` is always RAW; `--normalize` adds a separate `_norm` column so the
        # total-abundance figures (raw) and the per-element figure (normalized) coexist.
        for key, num_reads in est.items():
            norm = _normalize(num_reads, bundle, key) if bundle else num_reads
            rows.append((power, dp, key, subfamily(key), sim.get(key, 0), num_reads, norm))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["power", "del_prob", "l1_id", "subfamily", "simulated", "albertsalmon_seqlabel"]
        w.writerow(hdr + (["albertsalmon_seqlabel_norm"] if args.normalize else []))
        w.writerows(rows if args.normalize else [r[:6] for r in rows])
    print(f"[abundance] wrote {len(rows):,} rows from {cells} cells -> {args.out}")

    # --- multi-method long table (for the method comparison) ---------------
    gt_map = {}  # (power, dp) -> ground-truth path; counts.tsv (model 2) wins over bed.
    for pat in ("counts.tsv", "bed"):
        for p in sorted(glob.glob(f"{args.refdir}/GRCh38.p14.{args.chr}.insert_level_*_delprob_*.{pat}")):
            m = re.search(r"insert_level_(\d+)_delprob_([0-9]+\.[0-9]+)", os.path.basename(p))
            if m:
                gt_map.setdefault((m.group(1), m.group(2)), p)

    mrows, mcells = [], 0
    for (power, dp), gt_path in sorted(gt_map.items()):
        sim = load_simulated_counts(gt_path) if gt_path.endswith(".counts.tsv") else load_simulated_bed(gt_path)
        base = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}.pair.{args.fcov}x"
        proj = f"GRCh38.p14.{args.chr}.insert_level_{power}_delprob_{dp}"
        bundle = feat_get(power, dp, base) if feat_get else None
        present = False
        for method, (subdir, fname, loader) in METHODS.items():
            path = f"{args.refdir}/{subdir}/{base}/{fname.format(proj=proj)}"
            if not os.path.isfile(path):
                continue
            try:
                vals = loader(path)
            except Exception as exc:  # noqa: BLE001 — a bad file skips one method, not the run
                print(f"[abundance] WARN: {method} parse failed for {base}: {exc}")
                continue
            # A 0-byte / header-only output (e.g. an L1EM/AlbertEM run that found no L1
            # reads to quantify) parses to {} → skip it instead of emitting a phantom
            # all-zero column that only gets dropped downstream.
            if not vals:
                print(f"[abundance] WARN: {method} produced no counts for {base} "
                      f"(empty {os.path.basename(path)}) — skipping")
                continue
            present = True
            do_norm = bundle is not None and method in NORMALIZE_METHODS
            for uid in sim:
                raw = vals.get(uid, 0)
                norm = _normalize(raw, bundle, uid) if do_norm else raw
                # Empirical 4.08 "Simulated" reference (bedtools abundance of the element);
                # method-independent, blank when features are unavailable (raw mode).
                sreads = legacy_simulated_reads(bundle, uid) if bundle else ""
                mrows.append((power, dp, method, uid, sim[uid], raw, norm, sreads))
        mcells += present

    if mrows:
        with open(args.methods_out, "w", newline="") as fh:
            w = csv.writer(fh)
            hdr = ["power", "del_prob", "method", "l1_id", "simulated", "abundance"]
            w.writerow(hdr + (["abundance_norm", "simulated_reads"] if args.normalize else []))
            w.writerows(mrows if args.normalize else [r[:6] for r in mrows])
        seen = sorted(set(r[2] for r in mrows))
        print(f"[abundance] wrote {len(mrows):,} method rows from {mcells} cells "
              f"(methods={seen}) -> {args.methods_out}")
    else:
        print("[abundance] no baseline-method outputs found (run scripts/sh/run_*.sh) — methods table skipped")


if __name__ == "__main__":
    main()
