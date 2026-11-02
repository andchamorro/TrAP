# archive/scripts/draft/

Draft drivers for the synthetic-insertion series (insertion level 5–13 ×
deletion probability 0–0.1), written before the stage wrappers existed. Moved
here from `scripts/draft/`, where `2f02e59` had parked them.

| File | Successor |
|---|---|
| `run_mlem_mut.sh`, `run_mlem.sh` | `scripts/sh/run_albertem.sh` (AlbertEM), which states it is a refactor of `run_mlem_mut.sh` |
| `run_salmon_mut.sh`, `run_mlsalmon_mut.sh` | `scripts/sh/run_salmon.sh`, driven by `scripts/slurm/63_run_baseline.slurm` |
| `run_l1em_mut.sh` | `scripts/sh/run_l1em.sh`, driven by the same stage |

The reported synthetic benchmark (Figures 9–10, Tables 4 and S2–S5) was produced
by the successors, not by these files.
