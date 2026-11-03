"""Derive ``L1HS-L1PA_FPM.txt`` from L1EM's own outputs, matching the published file.

The Figure-11 aggregator reads each EM arm from ``L1HS-L1PA_FPM.txt`` with columns
``family.category.locus.strand``, ``only``, ``3prunon``.  **No script in the repo
produces that file** -- stock L1EM writes ``full_counts.txt`` (all five transcript
types, in read PAIRS) and ``filter_L1HS_FPM.txt`` (L1HS only, FPM, heavily
filtered).  The HEK293 file came from the legacy notebooks workspace, so the
transformation has to be reconstructed before the VM-Cub-1 arm can be comparable.

WHAT THE PUBLISHED FILE ACTUALLY IS, established by measuring it rather than by
reading the stock utility:

* 738 rows, every family either ``L1HS`` or ``L1PA*`` -- so it is
  ``filtered_and_normalized_l1hs.py`` with the family test widened from
  ``'L1HS' in name`` to L1HS **or** L1PA, which is what the filename says.
* minimum ``only + 3prunon`` = 0.00039111 FPM.  Against SRR3997504's 41.4 M pairs
  that is 0.016 read pairs, so the stock ``only+runon > 100`` floor is **not**
  applied, and neither is the ``> 3x runthrough`` test -- either would have cut
  that row.  The operative filter is ``only + 3prunon > 0``.
* FPM normalisation is by proper pairs in the ORIGINAL bam (``baminfo.txt`` line 2),
  not by the pairs L1EM actually consumed (line 3), which is what
  ``filtered_and_normalized_l1hs.py`` divides by.

So this reimplements that, and prints how many rows each of the stock filters
would have removed, so the reconstruction is visible instead of asserted.

**The 3/4 rule.**  A second file is written beside the primary one.  McKerrow & Fenyo
(2020) §2.5 states that L1EM does not report ``only``/``run-on`` transcripts when they
make up less than three quarters of the expression at that locus, "because some
fraction of 'passive' transcription can be mislabeled as 'only' or 'run-on'".  The
published HEK293 file does **not** apply it, and the primary output here does not
either -- changing the convention would silently move the published y-axis.  The rule
is worth evaluating anyway, and costs nothing, because ``full_counts.txt`` carries
all five transcript types.  So it is emitted as a
**declared sensitivity arm**: ``<out>.three_quarter.txt``, same format, same
normalisation, with the rule applied.  Note what the rule does to the number: it turns
``only``/``runon`` from an estimate into a thresholded call, so the two files are not
interchangeable and the sensitivity arm must never be presented as a correction of the
primary.  (Knowledge carried over from the sibling RobaleTE/Sertorius project, which
documents the rule and deliberately does not apply it either.)

Run (python -I, paths as arguments):

    python -I scripts/analyses/5race_substitution/route_n_l1rip/l1hs_l1pa_fpm.py <l1em_out_dir> <out_file>

where ``<l1em_out_dir>`` holds ``full_counts.txt`` and ``baminfo.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

#: Families kept, by prefix of the first dot-field of the locus name.
KEEP_PREFIXES = ("L1HS", "L1PA")

#: The stock utility's floors, reported but NOT applied (see module docstring).
STOCK_PAIR_FLOOR = 100
STOCK_RUNTHROUGH_MULTIPLE = 3

#: McKerrow §2.5's rule: report proper transcription only where it is at least this
#: fraction of the locus's total expression.  Applied in the sensitivity arm only.
THREE_QUARTER_FRACTION = 0.75

#: The transcript types that count as PROPER (promoter-driven) transcription.  Everything
#: else in ``full_counts.txt`` is the denominator of the 3/4 rule.
PROPER_TYPES = ("only", "3prunon")

#: ``full_counts.txt`` has a DIFFERENT SCHEMA depending on which L1EM driver ran, so the
#: columns are read off the file rather than assumed:
#:   stranded   (run_L1EM.sh)             only, 3prunon, passive_sense,
#:                                        passive_antisense, antisense
#:   unstranded (run_L1EM_unstranded.sh)  only, 3prunon, passive
#: The unstranded model cannot separate sense from antisense -- that is the whole reason
#: it exists -- so it merges them into one `passive` column and drops `antisense`.
#: Hardcoding the five stranded names would KeyError on an unstranded run, which is
#: exactly the library this route has.

#: McKerrow's own validity floor: §3 reports r^2 = 0.83 against simulation
#: specifically at loci with at least this many assigned read pairs.  Reported here so
#: the matched set can be read against L1EM's own documented reliability, which is a
#: stricter bar than the G3b depth floor of 10 reads.
MCKERROW_RELIABLE_PAIRS = 100


def main(l1em_dir: Path, out_path: Path) -> None:
    counts_path = l1em_dir / "full_counts.txt"
    baminfo_path = l1em_dir / "baminfo.txt"
    for path in (counts_path, baminfo_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    baminfo = [line.strip() for line in baminfo_path.read_text().splitlines() if line.strip()]
    median_insert = float(baminfo[0])
    proper_pairs = float(baminfo[1])
    pairs_to_em = float(baminfo[2])
    print(
        f"baminfo: median_insert={median_insert:.0f} "
        f"proper_pairs_in_original_bam={proper_pairs:,.0f} pairs_passed_to_EM={pairs_to_em:,.0f}"
    )

    counts = pd.read_csv(counts_path, sep="\t")
    name_col = counts.columns[0]
    family = counts[name_col].str.split(".", n=1).str[0]
    keep_family = family.str.startswith(KEEP_PREFIXES)

    types = tuple(c for c in counts.columns if c != name_col)
    missing = tuple(t for t in PROPER_TYPES if t not in types)
    if missing:
        raise SystemExit(f"{counts_path} lacks the proper-transcription columns {missing}")
    improper = tuple(t for t in types if t not in PROPER_TYPES)
    schema = "unstranded" if "passive" in types else "stranded"
    print(f"full_counts schema: {schema} -- columns {types}")
    runthrough = sum(counts[t] for t in improper)
    proper = sum(counts[t] for t in PROPER_TYPES)

    kept = counts[keep_family & (proper > 0)].copy()
    scale = 1e6 / proper_pairs
    out = pd.DataFrame(
        {
            "family.category.locus.strand": kept[name_col],
            "only": kept["only"] * scale,
            "3prunon": kept["3prunon"] * scale,
        }
    )
    out.to_csv(out_path, sep="\t", index=False)

    # --- declared sensitivity arm: McKerrow §2.5's 3/4 rule ---------------------
    total_expr = proper + runthrough
    passes_rule = proper >= THREE_QUARTER_FRACTION * total_expr
    kept_rule = counts[keep_family & (proper > 0) & passes_rule].copy()
    rule_out = pd.DataFrame(
        {
            "family.category.locus.strand": kept_rule[name_col],
            "only": kept_rule["only"] * scale,
            "3prunon": kept_rule["3prunon"] * scale,
        }
    )
    rule_path = out_path.with_suffix(".three_quarter.txt")
    rule_out.to_csv(rule_path, sep="\t", index=False)

    # What the stock filters would have done, so the choice is auditable.
    in_family = counts[keep_family]
    n_pair_floor = int((sum(in_family[t] for t in PROPER_TYPES) > STOCK_PAIR_FLOOR).sum())
    rt = sum(in_family[t] for t in improper)
    in_proper = sum(in_family[t] for t in PROPER_TYPES)
    n_both = int(
        ((in_proper > STOCK_PAIR_FLOOR) & (in_proper > STOCK_RUNTHROUGH_MULTIPLE * rt)).sum()
    )
    print(
        f"loci: {len(counts):,} quantified, {int(keep_family.sum()):,} L1HS/L1PA, "
        f"{len(out):,} with only+3prunon > 0  -> written"
    )
    print(
        f"  for reference, the stock floors would leave {n_pair_floor:,} "
        f"(>{STOCK_PAIR_FLOOR} pairs) and {n_both:,} (also >{STOCK_RUNTHROUGH_MULTIPLE}x runthrough); "
        "neither is applied, to match the published HEK293 file"
    )
    if len(out):
        total = out["only"] + out["3prunon"]
        print(
            f"  FPM only+3prunon: min={total.min():.8f} median={total.median():.6f} "
            f"max={total.max():.4f}"
        )
        n_reliable = int((proper[keep_family & (proper > 0)] >= MCKERROW_RELIABLE_PAIRS).sum())
        print(
            f"  of those, {n_reliable:,} carry >= {MCKERROW_RELIABLE_PAIRS} proper read pairs "
            "-- McKerrow's own validity floor (r^2 = 0.83 vs simulation holds there, not below)"
        )
    print(
        f"  3/4 rule (sensitivity arm, NOT applied to the primary): {len(rule_out):,} of "
        f"{len(out):,} loci keep their proper-transcription estimate "
        f"({100.0 * len(rule_out) / max(len(out), 1):.1f} %)"
    )
    print(f"wrote {out_path}")
    print(f"wrote {rule_path}  (declared sensitivity, do not substitute for the primary)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
