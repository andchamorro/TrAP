# archive/scripts/sh/

Alignment-and-count wrappers from the early exploration of aligners for the
read-count axis: Bowtie 1, Bowtie 2, HISAT2 and STAR, each followed by
featureCounts. None is behind a reported result. The reported comparisons use
`scripts/sh/align_star.sh` (label alignment), `scripts/sh/bwaAndCount.sh`
(chromosome-level axis) and `scripts/sh/bwamemAndCount.sh` (the 5'RACE reference
of Figure 11), which stay in `scripts/sh/`.

Moved from `scripts/sh/`; they arrived there in `2f02e59`.
