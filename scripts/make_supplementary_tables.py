"""Emit the manuscript's supplementary tables from the named results artifacts.

Table S1 (synthetic insertion benchmark, per stratum) and Table S2 (class
imbalance and the propagation of the L1HS recall deficit) are the two tables the
manuscript names in its Supplementary Materials section. Every cell here is read
from a file under ``results/``; nothing is typed by hand, so re-running the
analysis and re-running this script is the only way the tables change.

    python scripts/make_supplementary_tables.py [OUTPUT.md]

Default output is ``manuscript/desk/draft/supplementary.md``. Build the Word
version with ``make docx DOCX_NAME=supplementary`` from ``manuscript/``.
"""

import csv
import io
import json
from pathlib import Path
import sys

from trap.config.config import PROJ_ROOT

ROOT = PROJ_ROOT
SYN = ROOT / "results" / "synthetic" / "chr1"
HPO = ROOT / "results" / "hpo_classification"
OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else (ROOT / "manuscript" / "desk" / "draft" / "supplementary.md")
)

# Manuscript names, in the order the manuscript uses.
LABEL = {
    "AlbertSalmon": "ALBERT with Salmon",
    "Salmon": "Salmon",
    "AlbertEM": "ALBERT with EM",
    "L1EM": "L1EM",
    "TEtranscripts": "TEtranscripts",
    "HTseq": "HTSeq",
}
ORDER = list(LABEL)


def rows(path):
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wide(path):
    """Read a method-by-stratum matrix whose first column is unlabelled."""
    with io.open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)[1:]
        data = {r[0]: r[1:] for r in reader if r}
    return header, data


def f3(x):
    return f"{float(x):.3f}"


def f4(x):
    return f"{float(x):.4f}"


def table(header, body, align=None):
    align = align or (["left"] + ["center"] * (len(header) - 1))
    sep = {"left": ":---", "center": ":---:", "right": "---:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


doc = []
A = doc.append

# ── front matter ──────────────────────────────────────────────────────────────
A("""---
title: "Supplementary Materials: A Transformers Analysis Pipeline (TrAP) to Evaluate LINE-1 RNA Sequence Content"
author:
- name: Andres D. Chamorro-Parejo
- name: Kenneth S. Ramos
---
""")

# ── Table S1 ──────────────────────────────────────────────────────────────────
ins_hdr, ins = wide(SYN / "by_insertion_rate.csv")
del_hdr, dele = wide(SYN / "by_del_probability.csv")
summary = rows(SYN / "synthetic_strata_summary.csv")
contrasts = rows(SYN / "synthetic_strata_contrasts.csv")
gradients = rows(SYN / "synthetic_strata_gradients.csv")
syn_manifest = json.loads((SYN / "manifest.synthetic_strata.json").read_text())

summ = {(r["axis"], r["method"]): r for r in summary}

A("# Table S1. Per-stratum concordance on the synthetic insertion benchmark\n")
A(
    "**Table S1a. Coefficient of determination by insertion level.** Per-stratum "
    "Pearson $R^{2}$ between each method's estimate and the known insertion level, "
    "for the nine insertion-level strata of the synthetic chromosome 1 benchmark. "
    "Column headings give the insertion level as the exponent $n$ of $2^{n}$. The "
    "final two columns give the mean $\\pm$ standard deviation and the range across "
    "the nine strata. Strata won counts the strata in which the method had the "
    "highest $R^{2}$ of the six.\n"
)
head = ["Method"] + [f"$2^{{{h}}}$" for h in ins_hdr] + ["Mean $\\pm$ SD", "Range", "Strata won"]
body = []
for m in ORDER:
    s = summ[("insertion_level", m)]
    body.append(
        [LABEL[m]]
        + [f3(v) for v in ins[m]]
        + [
            f"{float(s['mean_r2']):.3f} $\\pm$ {float(s['sd_r2']):.3f}",
            f"{f3(s['min_r2'])}--{f3(s['max_r2'])}",
            s["n_top"],
        ]
    )
A(table(head, body) + "\n")

A(
    "**Table S1b. Coefficient of determination by deletion probability.** As Table "
    "S1a, for the five deletion-probability strata. Reported to four decimal "
    "places because no method moves by more than 0.014 across the whole axis.\n"
)
head = ["Method"] + [h for h in del_hdr] + ["Mean $\\pm$ SD", "Range", "Strata won"]
body = []
for m in ORDER:
    s = summ[("deletion_probability", m)]
    body.append(
        [LABEL[m]]
        + [f4(v) for v in dele[m]]
        + [
            f"{float(s['mean_r2']):.4f} $\\pm$ {float(s['sd_r2']):.4f}",
            f"{f4(s['min_r2'])}--{f4(s['max_r2'])}",
            s["n_top"],
        ]
    )
A(table(head, body) + "\n")

A(
    "**Table S1c. Paired comparisons across strata.** Each row pairs ALBERT with "
    "Salmon against one comparator, or pairs a TrAP configuration against the "
    "quantifier it wraps, stratum by stratum. $\\Delta R^{2}$ is the mean paired "
    "difference across strata, with the paired bootstrap 95% confidence interval "
    "(5,000 resamples, seed 3469). *p* is the two-sided paired Wilcoxon "
    "signed-rank test; the floor is the smallest two-sided *p* attainable at that "
    "number of strata, so a *p* equal to its floor reflects complete directional "
    "consistency rather than the magnitude of the differences. Because the "
    "resampling unit is a design stratum rather than an independent replicate, "
    "these intervals describe consistency across the synthetic grid rather than "
    "inferential uncertainty across independent experiments.\n"
)
head = [
    "Axis",
    "Comparison",
    "Strata (n)",
    "Strata won",
    "Stratum lost",
    "Mean $\\Delta R^{2}$",
    "Min $\\Delta R^{2}$",
    "95% CI",
    "Wilcoxon *p*",
    "*p* floor",
]
axis_label = {"insertion_level": "Insertion level", "deletion_probability": "Deletion probability"}
body = []
for r in contrasts:
    if r["family"] == "within_quantifier" and r["contrast"] == "AlbertSalmon-Salmon":
        continue  # identical to the reference row above it
    a, b = r["contrast"].split("-")
    name = f"{LABEL[a]} vs {LABEL[b]}"
    if r["family"] == "within_quantifier":
        name += " (within quantifier)"
    lost = r["strata_lost"] or "--"
    if lost != "--" and r["axis"] == "insertion_level":
        lost = f"$2^{{{lost}}}$"
    body.append(
        [
            axis_label[r["axis"]],
            name,
            r["n_strata"],
            r["n_won"],
            lost,
            f"+{float(r['mean_delta_r2']):.3f}",
            f"{float(r['min_delta_r2']):+.3f}",
            f"{float(r['ci_lo']):.3f} to {float(r['ci_hi']):.3f}",
            f"{float(r['wilcoxon_p']):.4f}",
            f"{float(r['p_floor']):.4f}",
        ]
    )
A(table(head, body) + "\n")

A(
    "**Table S1d. Sensitivity to insertion load.** $R^{2}$ at the lowest and "
    "highest insertion-level stratum, and at the lowest and highest deletion "
    "probability, with the difference between them.\n"
)
head = [
    "Axis",
    "Method",
    "First stratum",
    "Last stratum",
    "$R^{2}$ first",
    "$R^{2}$ last",
    "Decline",
]
body = []
for axis in ("insertion_level", "deletion_probability"):
    for m in ORDER:
        r = next(g for g in gradients if g["axis"] == axis and g["method"] == m)
        first = (
            f"$2^{{{r['first_stratum']}}}$" if axis == "insertion_level" else r["first_stratum"]
        )
        last = f"$2^{{{r['last_stratum']}}}$" if axis == "insertion_level" else r["last_stratum"]
        body.append(
            [
                axis_label[axis],
                LABEL[m],
                first,
                last,
                f3(r["r2_first"]),
                f3(r["r2_last"]),
                f"{float(r['drop']):+.3f}",
            ]
        )
A(table(head, body) + "\n")

A(f"""*Provenance.* Table S1 is generated from `results/synthetic/chr1/` by
`python -m trap.analysis synthetic-strata` (run manifest
`manifest.synthetic_strata.json`: TrAP commit `{syn_manifest['git_commit']}`,
seed {syn_manifest['seed']}, {syn_manifest['params']['n_boot']} bootstrap
resamples, reference method {syn_manifest['params']['reference']},
{syn_manifest['timestamp_utc'][:10]}). Panels S1a and S1b are
`by_insertion_rate.csv` and `by_del_probability.csv`; S1c is
`synthetic_strata_contrasts.csv`; S1d is `synthetic_strata_gradients.csv`; the
summary columns are `synthetic_strata_summary.csv`.
""")

# ── Table S2 ──────────────────────────────────────────────────────────────────
weights = rows(HPO / "imbalance_class_weights.csv")
ratios = rows(HPO / "imbalance_ratios.csv")
prop = rows(HPO / "imbalance_propagation_factors.csv")
sens = rows(HPO / "imbalance_race5_sensitivity.csv")
paired = rows(HPO / "imbalance_paired_contrasts.csv")
hpo_manifest = json.loads((HPO / "manifest.imbalance.json").read_text())

A(
    "# Table S2. Class imbalance, the training class weights, and the propagation of the L1HS recall deficit\n"
)

A(
    "**Table S2a. Training-split class frequencies and the resulting weight "
    "vector.** Weights are inverse-frequency balanced, $w_{c} = N/(k \\cdot n_{c})$ "
    "for $k = 3$ classes and $N$ training reads, and were used in the weighted "
    "cross-entropy objective. The frequency-weighted mean of the three weights is "
    "1.0000, so the objective raises the relative contribution of the minority "
    "L1HS class without changing the overall loss scale.\n"
)
head = ["Class", "Training reads", "Frequency", "Weight $w_{c}$"]
body = [
    [r["class"], f"{int(r['n_train']):,}", f"{float(r['frequency']):.6f}", f4(r["weight"])]
    for r in weights
]
total = sum(int(r["n_train"]) for r in weights)
wmean = sum(float(r["frequency"]) * float(r["weight"]) for r in weights)
body.append(
    ["**Total**", f"**{total:,}**", "1.000000", f"**{wmean:.4f}** (frequency-weighted mean)"]
)
A(table(head, body) + "\n")

A(
    "**Table S2b. L1HS-to-L1PA read ratio by split.** The imbalance is a property "
    "of the corpus and is reproduced in every split, so it is not an artifact of "
    "the transcript-level partition.\n"
)
head = ["Split", "L1HS reads", "L1PA reads", "L1PA : L1HS", "L1HS frequency"]
body = [
    [
        r["split"],
        f"{int(r['n_L1HS']):,}",
        f"{int(r['n_L1PA']):,}",
        f"{float(r['ratio']):.1f} : 1",
        f"{float(r['minority_frequency']):.6f}",
    ]
    for r in ratios
]
A(table(head, body) + "\n")

A(
    "**Table S2c. Propagation of the L1HS recall deficit into abundance.** Reads "
    "assigned to the NEGATIVE class are discarded before quantification, so the "
    "factor that scales the retained L1HS signal is the ratio of precision to "
    "recall and not recall alone. If $T$ is the true number of L1HS reads and $A$ "
    "the number assigned to L1HS, then $T/A = precision/recall$. The deficit "
    "carried into the abundance estimates is therefore substantially smaller than "
    "recall alone would imply, though it is not removed.\n"
)
head = ["Correction", "Recall", "Precision", "Factor", "Implied L1HS underestimate", "Basis"]
basis = {
    "recall_only": "1/recall; ignores the contamination that precision measures",
    "net": "precision/recall; the factor that recovers $T$ from $A$",
}
body = [
    [
        r["correction"].replace("_", " "),
        r["recall"],
        r["precision"],
        f"{float(r['factor']):.3f}",
        f"{float(r['implied_underestimate']) * 100:.1f}%",
        basis[r["correction"]],
    ]
    for r in prop
]
A(table(head, body) + "\n")

A(
    "**Table S2d. Downstream sensitivity of the 5'RACE comparison to the "
    "correction.** The factor was applied to the L1HS estimates of the two "
    "classifier-dependent configurations only, on the pre-specified matched set of "
    "156 loci, and the locus-level fit recomputed. Comparators that do not use the "
    "classifier are unaffected by construction. The corrected columns are reported "
    "as a sensitivity analysis and not as an abundance correction, because "
    "aggregate classification metrics do not establish that misclassification is "
    "uniform across loci.\n"
)
head = [
    "Correction",
    "Factor",
    "Method",
    "Classifier-dependent",
    "Corrected loci (n)",
    "$R^{2}$",
    "$\\Delta R^{2}$ vs published",
]
body = []
for r in sens:
    body.append(
        [
            r["correction"].replace("_", " "),
            f"{float(r['factor']):.3f}",
            LABEL[r["method"]],
            "yes" if r["classifier_dependent"] == "True" else "no",
            r["n_corrected_loci"],
            f3(r["pearson_r2"]),
            f"{float(r['delta_vs_published']):+.3f}",
        ]
    )
A(table(head, body) + "\n")

A(
    "**Table S2e. Paired within-quantifier margins under each correction.** Both "
    "margins widen under either correction, so the locus-level advantage is not an "
    "artifact of the uncorrected L1HS recall deficit.\n"
)
head = ["Correction", "Comparison", "$\\Delta R^{2}$", "TrAP configuration higher"]
body = []
rank = {"published": 0, "recall_only": 1, "net": 2}
for r in sorted(paired, key=lambda r: rank[r["correction"]]):
    a, b = r["contrast"].split("-")
    body.append(
        [
            r["correction"].replace("_", " "),
            f"{LABEL[a]} vs {LABEL[b]}",
            f"{float(r['delta_r2']):+.3f}",
            "yes" if r["trap_wins"] == "True" else "no",
        ]
    )
A(table(head, body) + "\n")

A(f"""*Provenance.* Table S2 is generated from `results/hpo_classification/` by
`python -m trap.analysis imbalance` (run manifest `manifest.imbalance.json`:
TrAP commit `{hpo_manifest['git_commit']}`, seed {hpo_manifest['seed']},
minority class {hpo_manifest['params']['minority']},
{hpo_manifest['timestamp_utc'][:10]}). Panel S2a is
`imbalance_class_weights.csv`; S2b is `imbalance_ratios.csv`; S2c is
`imbalance_propagation_factors.csv`; S2d is
`imbalance_race5_sensitivity.csv`; S2e is `imbalance_paired_contrasts.csv`.
""")

OUT.write_text("\n".join(doc), encoding="utf-8")
print(f"wrote {OUT}  ({len(chr(10).join(doc).splitlines())} lines)")
