"""Information-theoretic k-mer analysis for k-mer length selection.

This subpackage replaces the ad-hoc entropy/redundancy analysis behind the
manuscript's Supplementary Figures 1-2 with a rigorous, reproducible, and
extensible implementation that addresses the peer-review concerns:

* **Finite-sample bias** (Reviewer 1) — plug-in Shannon entropy underestimates
  the true entropy, and a "plateau" can be an artefact of sampling saturation.
  :mod:`trap.analysis.entropy` provides the Miller-Madow and Chao-Shen
  bias-corrected estimators plus bootstrap confidence intervals, and
  :mod:`trap.analysis.kmer_counting` adds rarefaction (subsampling) curves so a
  true plateau can be distinguished from a sampling artefact.
* **Operational definitions** (Reviewers 1-2) — entropy, redundancy, and the
  empirical k-mer distribution are defined as code, not prose.
* **Beyond the plateau** (Reviewer 3) — :mod:`trap.analysis.ablation` measures
  whether longer k-mers actually improve downstream L1HS/L1PA/NEGATIVE
  separability.

The public estimator API lives in :mod:`trap.analysis.entropy`.
"""

from trap.analysis.entropy import (
    bootstrap_ci,
    chao_shen_entropy,
    entropy_summary,
    good_turing_coverage,
    kl_divergence_uniform,
    miller_madow_entropy,
    n_singletons,
    redundancy,
    shannon_entropy_mle,
)

__all__ = [
    "bootstrap_ci",
    "chao_shen_entropy",
    "entropy_summary",
    "good_turing_coverage",
    "kl_divergence_uniform",
    "miller_madow_entropy",
    "n_singletons",
    "redundancy",
    "shannon_entropy_mle",
]
