"""Emit the manuscript's supplementary tables from the named results artifacts.

Tables S1-S4 cover the synthetic insertion benchmark stratum by stratum; tables
S5-S9 cover class imbalance and the propagation of the L1HS recall deficit;
tables S10-S11 cover the chromosome-length adjustment of the Figure 13
correlations. Each is a separate numbered supplementary table, cited
individually in the main text.
Every cell here is read from a file under ``results/``; nothing is typed by hand,
so re-running the analysis and re-running this script is the only way the tables
change.

Captions are emitted as pandoc table captions carrying the literal S-number and
no crossref identifier, so nothing is auto-renumbered as "Table 1". They state
the method in prose; the generating command and run manifest are recorded in the
manifests themselves and are deliberately not printed into a published file.

    python scripts/make_supplementary_tables.py [OUTPUT.md]

Default output is ``manuscript/desk/draft/supplementary.md``. Build the Word
version against the MDPI template with ``make mdpi MDPI_NAME=supplementary``
from ``manuscript/``.
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
NC = ROOT / "results" / "srr4099955"
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


def pval(x):
    """A p value at four decimals, or in the manuscript's scientific notation.

    Four decimals round the smallest coefficients here to 0.0000, which reads as
    an exact zero rather than as "too small to print".
    """
    x = float(x)
    if x >= 1e-4:
        return f"{x:.4f}"
    exponent = 0
    while x < 1:
        x *= 10
        exponent -= 1
    return f"${x:.1f}\\times10^{{{exponent}}}$"


def table(header, body, align=None):
    align = align or (["left"] + ["center"] * (len(header) - 1))
    sep = {"left": ":---", "center": ":---:", "right": "---:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def caption(number, title, text):
    """A pandoc table caption: literal "Table SN.", bold, then the prose."""
    return f": **Table S{number}.** {title} {text}"


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
# Quoted in the Table S3 caption; read from the manifest so the two cannot drift.
SEED = syn_manifest["seed"]
N_BOOT = int(syn_manifest["params"]["n_boot"])

summ = {(r["axis"], r["method"]): r for r in summary}

cap = caption(
    1,
    "Coefficient of determination by insertion level.",
    "Per-stratum Pearson $R^{2}$ between each method's estimate and the known "
    "insertion level, for the nine insertion-level strata of the synthetic "
    "chromosome 1 benchmark described in the Methods. All six methods were scored "
    "on the same simulated libraries, and each stratum was fitted separately. "
    "Column headings give the insertion level as the exponent $n$ of $2^{n}$. The "
    "final two columns give the mean $\\pm$ standard deviation and the range across "
    "the nine strata. Strata won counts the strata in which the method had the "
    "highest $R^{2}$ of the six.",
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
A(cap + "\n")

cap = caption(
    2,
    "Coefficient of determination by deletion probability.",
    "As Table S1, for the five deletion-probability strata of the same benchmark, "
    "spanning 0 to 0.1 in 0.025 increments. Reported to four decimal places "
    "because no method moves by more than 0.014 across the whole axis.",
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
A(cap + "\n")

cap = caption(
    3,
    "Paired comparisons across strata.",
    "Each row pairs ALBERT with Salmon against one comparator, or pairs a TrAP "
    "configuration against the quantifier it wraps, stratum by stratum, using the "
    "per-stratum values of Tables S1 and S2. $\\Delta R^{2}$ is the mean paired "
    f"difference across strata, with the paired bootstrap 95% confidence interval "
    f"({N_BOOT:,} resamples, seed {SEED}). *p* is the two-sided paired Wilcoxon "
    "signed-rank test; the floor is the smallest two-sided *p* attainable at that "
    "number of strata, so a *p* equal to its floor reflects complete directional "
    "consistency rather than the magnitude of the differences. Because the "
    "resampling unit is a design stratum rather than an independent replicate, "
    "these intervals describe consistency across the synthetic grid rather than "
    "inferential uncertainty across independent experiments.",
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
A(cap + "\n")

cap = caption(
    4,
    "Sensitivity to insertion load.",
    "$R^{2}$ at the lowest and highest insertion-level stratum, and at the lowest "
    "and highest deletion probability, with the difference between them. Values "
    "are the end points of the two axes tabulated in Tables S1 and S2.",
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
A(cap + "\n")

# ── Table S2 ──────────────────────────────────────────────────────────────────
weights = rows(HPO / "imbalance_class_weights.csv")
ratios = rows(HPO / "imbalance_ratios.csv")
prop = rows(HPO / "imbalance_propagation_factors.csv")
sens = rows(HPO / "imbalance_race5_sensitivity.csv")
paired = rows(HPO / "imbalance_paired_contrasts.csv")
hpo_manifest = json.loads((HPO / "manifest.imbalance.json").read_text())
assert hpo_manifest["seed"] == SEED, "the two analyses were run under different seeds"
assert hpo_manifest["params"]["minority"] == "L1HS", "Tables S5-S9 assume L1HS is the minority"

cap = caption(
    5,
    "Training-split class frequencies and the resulting weight vector.",
    "Counts are the reads of the transcript-level training split described in the "
    "Methods. Weights are inverse-frequency balanced, $w_{c} = N/(k \\cdot n_{c})$ "
    "for $k = 3$ classes and $N$ training reads, and were used in the weighted "
    "cross-entropy objective. The frequency-weighted mean of the three weights is "
    "1.0000, so the objective raises the relative contribution of the minority "
    "L1HS class without changing the overall loss scale.",
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
A(cap + "\n")

cap = caption(
    6,
    "L1HS-to-L1PA read ratio by split.",
    "Read counts of the two LINE-1 classes in each of the three transcript-level "
    "splits. The imbalance is a property of the corpus and is reproduced in every "
    "split, so it is not an artifact of the transcript-level partition.",
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
A(cap + "\n")

cap = caption(
    7,
    "Propagation of the L1HS recall deficit into abundance.",
    "Recall and precision are the held-out read-level values for the L1HS class. "
    "Reads assigned to the NEGATIVE class are discarded before quantification, so "
    "the factor that scales the retained L1HS signal is the ratio of precision to "
    "recall and not recall alone. If $T$ is the true number of L1HS reads and $A$ "
    "the number assigned to L1HS, then $T/A = precision/recall$. The deficit "
    "carried into the abundance estimates is therefore substantially smaller than "
    "recall alone would imply, though it is not removed.",
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
A(cap + "\n")

cap = caption(
    8,
    "Downstream sensitivity of the 5'RACE comparison to the correction.",
    "Each factor of Table S7 was applied to the L1HS estimates of the two "
    "classifier-dependent configurations only, on the pre-specified matched set of "
    "156 loci, and the locus-level fit recomputed. Comparators that do not use the "
    "classifier are unaffected by construction. The corrected rows are reported "
    "as a sensitivity analysis and not as an abundance correction, because "
    "aggregate classification metrics do not establish that misclassification is "
    "uniform across loci.",
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
A(cap + "\n")

cap = caption(
    9,
    "Paired within-quantifier margins under each correction.",
    "Each margin pairs a TrAP configuration against the quantifier it wraps on the "
    "matched set of Table S8. Both margins widen under either correction, so the "
    "locus-level advantage is not an artifact of the uncorrected L1HS recall "
    "deficit.",
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
A(cap + "\n")

# ── Tables S10-S11 ────────────────────────────────────────────────────────────
chrlen = rows(NC / "negative_control_chrom_length.csv")
chrlen_manifest = json.loads((NC / "manifest.chrom_length.json").read_text())
# Quoted in both captions, so the assembly cannot drift from the file that was read.
ASSEMBLY = chrlen_manifest["params"]["assembly"]
N_CHROM = int(chrlen[0]["n_chrom"])

cap = caption(
    10,
    "Chromosome-level correlations before and after adjustment for chromosome length.",
    "Each row pairs one classifier output class with one alignment class, over the "
    f"{N_CHROM} assembled chromosomes of the {ASSEMBLY} primary assembly. The "
    "Pearson columns reproduce the coefficients plotted in Figure 13; the partial "
    "columns repeat them after regressing both axes on chromosome length and "
    "correlating the residuals. Chromosome length is reported as a third covariate "
    "alongside the two in the main text, and is the one that shifts the "
    "NEGATIVE-class association furthest. It is not a clean confounder, because "
    "longer chromosomes carry proportionally more LINE-1 sequence, so the "
    "adjustment withdraws part of the signal under study together with the "
    "nuisance it targets; the partial coefficients are therefore a lower bound "
    "rather than a corrected estimate. The L1HS diagonal, on which the "
    "concordance claim rests, is unchanged.",
)
head = [
    "Classifier class",
    "Alignment class",
    "Pearson $r$",
    "$p$",
    "Partial $r$ given length",
    "$p$",
    "Change in $r$",
]
body = [
    [
        r["classifier_class"],
        r["alignment_class"],
        f"{float(r['pearson_r']):+.3f}",
        pval(r["pearson_p"]),
        f"{float(r['partial_r_given_chrom_length']):+.3f}",
        pval(r["partial_p_given_chrom_length"]),
        f"{float(r['delta_r']):+.3f}",
    ]
    for r in chrlen
]
A(table(head, body) + "\n")
A(cap + "\n")

cap = caption(
    11,
    "Dependence of each chromosome-level axis on chromosome length.",
    "Pearson correlation between each aggregated axis and the length of the "
    f"chromosome it was aggregated over, across the same {N_CHROM} chromosomes. "
    "This is the quantity that governs Table S10: a pairing moves under the "
    "adjustment only when at least one of its two axes is itself length "
    "dependent. The NEGATIVE-class read count and both L1PA2 axes scale with "
    "chromosome length, whereas neither L1HS axis does, which is why the "
    "adjustment leaves the L1HS concordance untouched while shrinking the "
    "NEGATIVE-class coefficients.",
)
head = ["Axis", "Class", "Pearson $r$ with chromosome length", "$p$"]
seen = []
body = []
for key, axis in (("classifier", "Classifier read count"), ("alignment", "Alignment signal")):
    for r in chrlen:
        cls = r[f"{key}_class"]
        if (axis, cls) in seen:
            continue
        seen.append((axis, cls))
        body.append(
            [
                axis,
                cls,
                f"{float(r[f'{key}_r_vs_length']):+.3f}",
                pval(r[f"{key}_p_vs_length"]),
            ]
        )
A(table(head, body) + "\n")
A(cap + "\n")

OUT.write_text("\n".join(doc), encoding="utf-8")
print(f"wrote {OUT}  ({len(chr(10).join(doc).splitlines())} lines)")
