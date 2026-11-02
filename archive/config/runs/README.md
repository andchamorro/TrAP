# archive/config/runs/

`spm_k8.yaml` (`10d9f6c`) is the k=8 fallback for the experimental SentencePiece
run, meant to be used only if the k=17 run (`config/runs/spm.yaml`) failed its
token-fragmentation gate. Nothing selects it automatically. It `extends:
default`, which resolves against `config/runs/`, so it no longer loads from
here; to rerun it, copy it back to `config/runs/`.
