TrAP — Transformers Analysis Pipeline
======================================

**TrAP** treats DNA as language to detect and quantify **LINE-1 (L1)** retroelements
in short- and long-read RNA-seq data. Reads are split into k-mers, tokenized with
SentencePiece, and classified by a custom **ALBERT** transformer that learns the
nucleotide distribution of L1 sequences directly — no reference alignment required.

.. code-block:: bash

   # Full pipeline on Grace HPRC (afterok chain)
   bash scripts/slurm/submit_pipeline.sh

   # Quick local tokenizer training
   python -m trap.loaders.tokenizer train \
       --corpus data/external/gencode.v48.transcripts.fa.gz \
       --out models --name tokenizer.gencode.v48.k17.32k \
       --k 17 --vocab-size 32000

.. note::

   This documentation mirrors the methodology in
   *A Transformers Analysis Pipeline to Evaluate Genome LINE-1 Sequence Content*
   (Chamorro-Parejo et al.). Terms such as ALBERT, MLM + SOP, k-mer entropy, and
   5′RACE validation are used here with the same meaning as in the manuscript.

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/installation
   guides/pipeline
   guides/verbosity
   guides/hpc
   guides/hyperparameter_tuning
   guides/reproduction

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Develop & Debug

   development/testing
   development/training_workflow_tests
   development/tokenizer_debug
   development/contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
