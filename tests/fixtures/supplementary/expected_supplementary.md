---
title: "Supplementary Materials: A Transformers Analysis Pipeline (TrAP) to Evaluate LINE-1 RNA Sequence Content"
author:
- name: Andres D. Chamorro-Parejo
- name: Kenneth S. Ramos
---

| Method | $2^{5}$ | $2^{6}$ | $2^{7}$ | $2^{8}$ | $2^{9}$ | $2^{10}$ | $2^{11}$ | $2^{12}$ | $2^{13}$ | Mean $\pm$ SD | Range | Strata won |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ALBERT with Salmon | 1.018 | 1.007 | 1.039 | 1.050 | 1.055 | 1.041 | 0.946 | 0.889 | 0.913 | 1.025 $\pm$ 0.051 | 0.877--1.018 | 9 |
| Salmon | 0.813 | 0.894 | 0.975 | 1.007 | 0.908 | 0.832 | 0.733 | 0.528 | 0.331 | 0.779 $\pm$ 0.220 | 0.336--1.007 | 0 |
| ALBERT with EM | 0.561 | 0.504 | 0.468 | 0.519 | 0.402 | 0.486 | 0.382 | 0.243 | 0.222 | 0.408 $\pm$ 0.121 | 0.214--0.554 | 0 |
| L1EM | 0.541 | 0.464 | 0.452 | 0.516 | 0.410 | 0.471 | 0.358 | 0.200 | 0.192 | 0.419 $\pm$ 0.124 | 0.192--0.556 | 0 |
| TEtranscripts | 0.964 | 0.876 | 0.952 | 0.922 | 0.879 | 0.727 | 0.601 | 0.378 | 0.320 | 0.706 $\pm$ 0.236 | 0.316--0.976 | 0 |
| HTSeq | 0.924 | 0.957 | 0.910 | 0.804 | 0.749 | 0.605 | 0.487 | 0.299 | 0.266 | 0.656 $\pm$ 0.263 | 0.263--0.957 | 0 |

: **Table S1.** Coefficient of determination by insertion level. Per-stratum Pearson $R^{2}$ between each method's estimate and the known insertion level, for the nine insertion-level strata of the synthetic chromosome 1 benchmark described in the Methods. All six methods were scored on the same simulated libraries, and each stratum was fitted separately. Column headings give the insertion level as the exponent $n$ of $2^{n}$. The final two columns give the mean $\pm$ standard deviation and the range across the nine strata. Strata won counts the strata in which the method had the highest $R^{2}$ of the six.

| Method | 0.000 | 0.025 | 0.050 | 0.075 | 0.100 | Mean $\pm$ SD | Range | Strata won |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ALBERT with Salmon | 1.0895 | 1.0077 | 1.0212 | 1.0351 | 1.0485 | 1.0214 $\pm$ 0.0001 | 1.0484--1.0624 | 5 |
| Salmon | 1.0183 | 1.0303 | 1.0435 | 0.9631 | 0.9778 | 1.0431 $\pm$ 0.0011 | 0.9761--0.9922 | 0 |
| ALBERT with EM | 0.9034 | 0.9116 | 0.9272 | 0.9393 | 0.9428 | 0.9249 $\pm$ 0.0036 | 0.9428--0.8798 | 0 |
| L1EM | 0.8701 | 0.8766 | 0.8929 | 0.9041 | 0.9023 | 0.8893 $\pm$ 0.0061 | 0.9023--0.9289 | 0 |
| TEtranscripts | 1.0293 | 1.0451 | 0.9651 | 0.9791 | 0.9964 | 1.0056 $\pm$ 0.0025 | 1.0293--1.0489 | 0 |
| HTSeq | 0.9475 | 0.9639 | 0.9720 | 0.9865 | 0.9114 | 0.9735 $\pm$ 0.0018 | 0.9105--0.9268 | 0 |

: **Table S2.** Coefficient of determination by deletion probability. As Table S1, for the five deletion-probability strata of the same benchmark, spanning 0 to 0.1 in 0.025 increments. Reported to four decimal places because no method moves by more than 0.014 across the whole axis.

| Axis | Comparison | Strata (n) | Strata won | Stratum lost | Mean $\Delta R^{2}$ | Min $\Delta R^{2}$ | 95% CI | Wilcoxon *p* | *p* floor |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Insertion level | ALBERT with Salmon vs ALBERT with EM | 9 | 9 | -- | +0.557 | +0.485 | 0.528 to 0.631 | 0.0042 | 0.0042 |
| Insertion level | ALBERT with Salmon vs HTSeq | 9 | 9 | -- | +0.348 | +0.084 | 0.201 to 0.469 | 0.0041 | 0.0042 |
| Insertion level | ALBERT with Salmon vs L1EM | 9 | 9 | -- | +0.612 | +0.501 | 0.525 to 0.633 | 0.0041 | 0.0041 |
| Insertion level | ALBERT with Salmon vs Salmon | 9 | 9 | -- | +0.223 | +0.069 | 0.134 to 0.324 | 0.0040 | 0.0041 |
| Insertion level | ALBERT with Salmon vs TEtranscripts | 9 | 9 | -- | +0.265 | +0.069 | 0.160 to 0.410 | 0.0040 | 0.0040 |
| Insertion level | ALBERT with EM vs L1EM (within quantifier) | 9 | 8 | $2^{5}$ | +0.016 | -0.016 | 0.006 to 0.026 | 0.0211 | 0.0043 |
| Deletion probability | ALBERT with Salmon vs ALBERT with EM | 5 | 5 | -- | +0.130 | +0.130 | 0.132 to 0.139 | 0.0668 | 0.0676 |
| Deletion probability | ALBERT with Salmon vs HTSeq | 5 | 5 | -- | +0.104 | +0.093 | 0.096 to 0.100 | 0.0659 | 0.0668 |
| Deletion probability | ALBERT with Salmon vs L1EM | 5 | 5 | -- | +0.151 | +0.149 | 0.138 to 0.149 | 0.0651 | 0.0659 |
| Deletion probability | ALBERT with Salmon vs Salmon | 5 | 5 | -- | +0.045 | +0.045 | 0.046 to 0.044 | 0.0642 | 0.0651 |
| Deletion probability | ALBERT with Salmon vs TEtranscripts | 5 | 5 | -- | +0.043 | +0.040 | 0.042 to 0.046 | 0.0634 | 0.0642 |
| Deletion probability | ALBERT with EM vs L1EM (within quantifier) | 5 | 5 | -- | +0.012 | +0.010 | 0.011 to 0.015 | 0.0676 | 0.0685 |

: **Table S3.** Paired comparisons across strata. Each row pairs ALBERT with Salmon against one comparator, or pairs a TrAP configuration against the quantifier it wraps, stratum by stratum, using the per-stratum values of Tables S1 and S2. $\Delta R^{2}$ is the mean paired difference across strata, with the paired bootstrap 95% confidence interval (5,000 resamples, seed 3469). *p* is the two-sided paired Wilcoxon signed-rank test; the floor is the smallest two-sided *p* attainable at that number of strata, so a *p* equal to its floor reflects complete directional consistency rather than the magnitude of the differences. Because the resampling unit is a design stratum rather than an independent replicate, these intervals describe consistency across the synthetic grid rather than inferential uncertainty across independent experiments.

| Axis | Method | First stratum | Last stratum | $R^{2}$ first | $R^{2}$ last | Decline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Insertion level | ALBERT with Salmon | $2^{5}$ | $2^{13}$ | 1.059 | 0.949 | +0.125 |
| Insertion level | Salmon | $2^{5}$ | $2^{13}$ | 0.803 | 0.327 | +0.492 |
| Insertion level | ALBERT with EM | $2^{5}$ | $2^{13}$ | 0.526 | 0.208 | +0.329 |
| Insertion level | L1EM | $2^{5}$ | $2^{13}$ | 0.578 | 0.204 | +0.352 |
| Insertion level | TEtranscripts | $2^{5}$ | $2^{13}$ | 1.014 | 0.308 | +0.639 |
| Insertion level | HTSeq | $2^{5}$ | $2^{13}$ | 0.888 | 0.256 | +0.653 |
| Deletion probability | ALBERT with Salmon | 0.000e+00 | 0.10548 | 1.062 | 1.076 | +0.000 |
| Deletion probability | Salmon | 0.000e+00 | 0.10137 | 0.979 | 0.991 | +0.001 |
| Deletion probability | ALBERT with EM | 0.000e+00 | 0.10685 | 0.939 | 0.943 | +0.008 |
| Deletion probability | L1EM | 0.000e+00 | 0.10274 | 0.894 | 0.891 | +0.015 |
| Deletion probability | TEtranscripts | 0.000e+00 | 0.10959 | 0.964 | 0.983 | -0.006 |
| Deletion probability | HTSeq | 0.000e+00 | 0.10411 | 0.947 | 0.961 | -0.001 |

: **Table S4.** Sensitivity to insertion load. $R^{2}$ at the lowest and highest insertion-level stratum, and at the lowest and highest deletion probability, with the difference between them. Values are the end points of the two axes tabulated in Tables S1 and S2.

| Class | Training reads | Frequency | Weight $w_{c}$ |
|:---|:---:|:---:|:---:|
| L1HS | 60,639 | 0.013897 | 27.7361 |
| L1PA | 2,439,877 | 0.573493 | 0.6457 |
| NEGATIVE | 2,161,900 | 0.476392 | 0.7484 |
| **Total** | **4,662,416** | 1.000000 | **1.1123** (frequency-weighted mean) |

: **Table S5.** Training-split class frequencies and the resulting weight vector. Counts are the reads of the transcript-level training split described in the Methods. Weights are inverse-frequency balanced, $w_{c} = N/(k \cdot n_{c})$ for $k = 3$ classes and $N$ training reads, and were used in the weighted cross-entropy objective. The frequency-weighted mean of the three weights is 1.0000, so the objective raises the relative contribution of the minority L1HS class without changing the overall loss scale.

| Split | L1HS reads | L1PA reads | L1PA : L1HS | L1HS frequency |
|:---|:---:|:---:|:---:|:---:|
| Training | 60,639 | 2,439,877 | 42.4 : 1 | 0.013897 |
| Validation | 19,641 | 811,233 | 44.7 : 1 | 0.013850 |
| Test (held-out) | 19,499 | 814,814 | 42.4 : 1 | 0.012886 |
| Total | 99,779 | 4,065,924 | 42.4 : 1 | 0.013543 |

: **Table S6.** L1HS-to-L1PA read ratio by split. Read counts of the two LINE-1 classes in each of the three transcript-level splits. The imbalance is a property of the corpus and is reproduced in every split, so it is not an artifact of the transcript-level partition.

| Correction | Recall | Precision | Factor | Implied L1HS underestimate | Basis |
|:---|:---:|:---:|:---:|:---:|:---:|
| recall only | 0.666744 | 0.782431 | 1.756 | 38.1% | 1/recall; ignores the contamination that precision measures |
| net | 0.641098 | 0.752715 | 1.222 | 14.6% | precision/recall; the factor that recovers $T$ from $A$ |

: **Table S7.** Propagation of the L1HS recall deficit into abundance. Recall and precision are the held-out read-level values for the L1HS class. Reads assigned to the NEGATIVE class are discarded before quantification, so the factor that scales the retained L1HS signal is the ratio of precision to recall and not recall alone. If $T$ is the true number of L1HS reads and $A$ the number assigned to L1HS, then $T/A = precision/recall$. The deficit carried into the abundance estimates is therefore substantially smaller than recall alone would imply, though it is not removed.

| Correction | Factor | Method | Classifier-dependent | Corrected loci (n) | $R^{2}$ | $\Delta R^{2}$ vs published |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| published | 1.082 | ALBERT with Salmon | yes | 0 | 0.946 | +0.000 |
| published | 1.027 | Salmon | no | 0 | 0.815 | +0.000 |
| published | 1.069 | ALBERT with EM | yes | 0 | 0.945 | +0.000 |
| published | 1.014 | L1EM | no | 0 | 0.864 | +0.000 |
| recall only | 1.690 | ALBERT with Salmon | yes | 38 | 0.956 | +0.033 |
| recall only | 1.756 | Salmon | no | 0 | 0.794 | +0.000 |
| recall only | 1.668 | ALBERT with EM | yes | 38 | 0.960 | +0.039 |
| recall only | 1.734 | L1EM | no | 0 | 0.921 | +0.000 |
| net | 1.190 | ALBERT with Salmon | yes | 38 | 0.917 | +0.019 |
| net | 1.238 | Salmon | no | 0 | 0.847 | +0.000 |
| net | 1.175 | ALBERT with EM | yes | 38 | 0.917 | +0.021 |
| net | 1.222 | L1EM | no | 0 | 0.898 | +0.000 |

: **Table S8.** Downstream sensitivity of the 5'RACE comparison to the correction. Each factor of Table S7 was applied to the L1HS estimates of the two classifier-dependent configurations only, on the pre-specified matched set of 156 loci, and the locus-level fit recomputed. Comparators that do not use the classifier are unaffected by construction. The corrected rows are reported as a sensitivity analysis and not as an abundance correction, because aggregate classification metrics do not establish that misclassification is uniform across loci.

| Correction | Comparison | $\Delta R^{2}$ | TrAP configuration higher |
|:---|:---:|:---:|:---:|
| published | ALBERT with Salmon vs Salmon | +0.083 | yes |
| published | ALBERT with EM vs L1EM | +0.034 | yes |
| recall only | ALBERT with Salmon vs Salmon | +0.117 | yes |
| recall only | ALBERT with EM vs L1EM | +0.074 | yes |
| net | ALBERT with Salmon vs Salmon | +0.108 | yes |
| net | ALBERT with EM vs L1EM | +0.053 | yes |

: **Table S9.** Paired within-quantifier margins under each correction. Each margin pairs a TrAP configuration against the quantifier it wraps on the matched set of Table S8. Both margins widen under either correction, so the locus-level advantage is not an artifact of the uncorrected L1HS recall deficit.

| Classifier class | Alignment class | Pearson $r$ | $p$ | Partial $r$ given length | $p$ | Change in $r$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| L1HS | L1HS | +1.034 | $4.1\times10^{-13}$ | +0.970 | $2.9\times10^{-13}$ | +0.001 |
| L1HS | L1PA2 | -0.032 | 0.9093 | +0.050 | 0.8677 | +0.085 |
| L1HS | NEGATIVE | -0.041 | 0.9013 | +0.081 | 0.7854 | +0.126 |
| L1PA2 | L1HS | +0.291 | 0.2134 | +0.497 | 0.0262 | +0.186 |
| L1PA2 | L1PA2 | +0.910 | $5.1\times10^{-7}$ | +0.689 | 0.0003 | -0.169 |
| L1PA2 | NEGATIVE | +0.775 | $2.2\times10^{-5}$ | +0.407 | 0.0666 | -0.398 |
| NEGATIVE | L1HS | -0.013 | 1.0205 | +0.101 | 0.7277 | +0.107 |
| NEGATIVE | L1PA2 | +0.796 | $4.6\times10^{-5}$ | +0.414 | 0.0489 | -0.341 |
| NEGATIVE | NEGATIVE | +0.815 | $2.2\times10^{-6}$ | +0.161 | 0.4963 | -0.694 |

: **Table S10.** Chromosome-level correlations before and after adjustment for chromosome length. Each row pairs one classifier output class with one alignment class, over the 24 assembled chromosomes of the GRCh38.p14 (GCA_000001405.29) primary assembly. The Pearson columns reproduce the coefficients plotted in Figure 13; the partial columns repeat them after regressing both axes on chromosome length and correlating the residuals. Chromosome length is reported as a third covariate alongside the two in the main text, and is the one that shifts the NEGATIVE-class association furthest. It is not a clean confounder, because longer chromosomes carry proportionally more LINE-1 sequence, so the adjustment withdraws part of the signal under study together with the nuisance it targets; the partial coefficients are therefore a lower bound rather than a corrected estimate. The L1HS diagonal, on which the concordance claim rests, is unchanged.

| Axis | Class | Pearson $r$ with chromosome length | $p$ |
|:---|:---:|:---:|:---:|
| Classifier read count | L1HS | -0.105 | 0.6887 |
| Classifier read count | L1PA2 | +0.764 | $4.7\times10^{-5}$ |
| Classifier read count | NEGATIVE | +0.963 | $1.6\times10^{-11}$ |
| Alignment signal | L1HS | -0.052 | 0.9038 |
| Alignment signal | L1PA2 | +0.682 | 0.0003 |
| Alignment signal | NEGATIVE | +0.860 | $7.1\times10^{-7}$ |

: **Table S11.** Dependence of each chromosome-level axis on chromosome length. Pearson correlation between each aggregated axis and the length of the chromosome it was aggregated over, across the same 24 chromosomes. This is the quantity that governs Table S10: a pairing moves under the adjustment only when at least one of its two axes is itself length dependent. The NEGATIVE-class read count and both L1PA2 axes scale with chromosome length, whereas neither L1HS axis does, which is why the adjustment leaves the L1HS concordance untouched while shrinking the NEGATIVE-class coefficients.
