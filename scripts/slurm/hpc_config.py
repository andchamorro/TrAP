#!/usr/bin/env python3
"""Read an HPC module config YAML and emit shell-safe output.

Compatible with Python 3.6+ (system Python on RHEL 8 / Grace before
Anaconda is loaded).  Uses stdlib only — no third-party imports required.

Supports ``extends: <base>`` for single-level config inheritance.

Modes
-----
  --vars FILE
      Print KEY='VALUE' shell assignments for all config entries.
      Skips variables already set in the environment (preserves env overrides).

  --check-cmds GROUP FILE
      Print space-separated command names required for GROUP.
      Used by check_required_commands() in _common.sh.

Exit codes: 0 success, 1 config not found or parse error.
"""
# No 'from __future__ import annotations' — that requires Python 3.7+.
# This file must parse and run on Python 3.6 (RHEL 8 system Python).

import argparse
import os
import shlex
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_inline(s):
    """Strip an inline YAML comment and surrounding quotes from a value.

    Handles:
        value  # comment     ->  value
        "value"              ->  value
        ""                   ->  (empty string)
    """
    # Strip inline comment: first occurrence of ' #' that is not inside quotes
    idx = s.find(" #")
    if idx >= 0:
        s = s[:idx]
    s = s.strip()
    # Strip matching surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    return s


# ---------------------------------------------------------------------------
# Minimal YAML parser (stdlib only, handles our 3-level structure)
# ---------------------------------------------------------------------------


def _parse_simple(path):
    """Parse a simple 3-level YAML (indent 0/2/4) without PyYAML.

    Handles scalars at any level, lists at indent 4, and inline comments.
    Sufficient for config/hpc/*.yaml — do not use for arbitrary YAML.
    """
    root = {}
    section = None
    subsection = None

    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            # Skip blank lines and full-line comments
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            content = line.lstrip()

            if indent == 0:
                if content.endswith(":"):
                    section = content[:-1].strip()
                    root.setdefault(section, {})
                    subsection = None
                elif ":" in content:
                    k, _, v = content.partition(":")
                    root[k.strip()] = _strip_inline(v)
                    section = None
                    subsection = None

            elif indent == 2 and section is not None:
                if content.startswith("- "):
                    item = _strip_inline(content[2:])
                    if not isinstance(root.get(section), list):
                        root[section] = []
                    root[section].append(item)
                elif ":" in content:
                    k, _, v = content.partition(":")
                    k = k.strip()
                    v = _strip_inline(v)
                    sec = root.get(section)
                    if isinstance(sec, dict):
                        sec[k] = v if v else {}
                        subsection = k

            elif indent == 4 and section is not None and subsection is not None:
                if content.startswith("- "):
                    item = _strip_inline(content[2:])
                    sec = root.get(section)
                    if isinstance(sec, dict):
                        if not isinstance(sec.get(subsection), list):
                            sec[subsection] = []
                        sec[subsection].append(item)
                elif ":" in content:
                    k, _, v = content.partition(":")
                    k = k.strip()
                    v = _strip_inline(v)
                    sec = root.get(section)
                    if isinstance(sec, dict):
                        sub = sec.get(subsection)
                        if not isinstance(sub, dict):
                            sec[subsection] = {}
                            sub = sec[subsection]
                        sub[k] = v

    return root


def _load_raw(path):
    """Load YAML using PyYAML when available, else fall back to _parse_simple."""
    try:
        import yaml  # type: ignore

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    return _parse_simple(path)


def _deep_merge(base, override):
    """Merge override into base recursively for dicts; lists replace (not extend)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load(path):
    """Load a YAML config, resolving a single ``extends: <name>`` if present."""
    cfg = _load_raw(path)
    base_name = cfg.pop("extends", None)
    if base_name:
        base_path = os.path.join(
            os.path.dirname(os.path.abspath(path)),
            "{0}.yaml".format(base_name),
        )
        if os.path.isfile(base_path):
            base_cfg = _load_raw(base_path)
            cfg = _deep_merge(base_cfg, cfg)
        else:
            print(
                "[hpc_config] WARNING: base config not found: {0}".format(base_path),
                file=sys.stderr,
            )
    return cfg


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------


def _get_shell_vars(cfg):
    """Return the shell variable mapping derived from the loaded config."""
    mods = cfg.get("modules", {})
    gpu = cfg.get("gpu", {})
    conda = cfg.get("conda", {})
    slurm = cfg.get("slurm", {})

    def _join(key):
        val = mods.get(key, [])
        if isinstance(val, list):
            return " ".join(v for v in val if v)
        return str(val)

    return {
        "ANACONDA_MODULE": str(mods.get("anaconda", "")),
        "GCC_MODULE": str(mods.get("gcc", "")),
        "JELLYFISH_MODULE": str(mods.get("jellyfish", "")),
        "SEQKIT_MODULE": str(mods.get("seqkit", "")),
        "CUDA_MODULES": _join("cuda"),
        "BIO_MODULES": _join("bio"),
        "CONDA_ENV": str(conda.get("env", "trap")),
        "GPUS_PER_NODE": str(gpu.get("gpus_per_node", 2)),
        "MIXED_PRECISION": str(gpu.get("mixed_precision", "bf16")),
        "SLURM_ACCOUNT": str(slurm.get("account", "")),
        "SLURM_MAIL_USER": str(slurm.get("mail_user", "")),
        "SLURM_MAIL_TYPE": str(slurm.get("mail_type", "END,FAIL")),
        "SLURM_PARTITION_GPU": str(slurm.get("partition_gpu", "gpu")),
        "SLURM_GPU_GRES": str(gpu.get("gres", "")),
    }


def _get_required_commands(cfg, group):
    return cfg.get("required_commands", {}).get(group, [])


def _get_run_shell_vars(cfg):
    """Return the shell variable mapping for a pipeline run config.

    Handles the flat + one-level-nested structure of config/runs/*.yaml.
    """
    tok = cfg.get("tokenizer", {}) if isinstance(cfg.get("tokenizer"), dict) else {}

    pairs = []

    # Tokenizer
    if tok.get("algorithm"):
        pairs.append(("TOKENIZER_ALGORITHM", str(tok["algorithm"])))
    if tok.get("name"):
        pairs.append(("TOKENIZER_NAME", str(tok["name"])))
    if tok.get("num_threads") is not None:
        pairs.append(("TOKENIZER_NUM_THREADS", str(tok["num_threads"])))
    if tok.get("max_training_chars") is not None:
        pairs.append(("TOKENIZER_MAX_TRAINING_CHARS", str(tok["max_training_chars"])))

    # Scalar hyperparameters
    for yaml_key, env_key in (
        ("k", "K"),
        ("vocab", "VOCAB"),
        ("max_position", "MAX_POSITION"),
        ("seed", "SEED"),
    ):
        if yaml_key in cfg:
            pairs.append((env_key, str(cfg[yaml_key])))

    # Output names
    for yaml_key, env_key in (
        ("mlm_processing_name", "MLM_PROCESSING_NAME"),
        ("processing_name", "PROCESSING_NAME"),
        ("model_mlm", "MODEL_MLM"),
        ("model_cls", "MODEL_CLS"),
    ):
        if yaml_key in cfg:
            pairs.append((env_key, str(cfg[yaml_key])))

    # Input data paths
    for yaml_key, env_key in (
        ("gencode_fasta", "GENCODE_FASTA"),
        ("l1_r1", "L1_R1"),
        ("l1_r2", "L1_R2"),
    ):
        if yaml_key in cfg:
            pairs.append((env_key, str(cfg[yaml_key])))

    # Config file references
    for yaml_key, env_key in (
        ("albert_config", "ALBERT_CONFIG"),
        ("mlm_trainer_config", "MLM_TRAINER_CONFIG"),
        ("cls_trainer_config", "CLS_TRAINER_CONFIG"),
        ("tokenizer_config", "TOKENIZER_CONFIG"),
    ):
        if yaml_key in cfg:
            pairs.append((env_key, str(cfg[yaml_key])))

    # Per-stage SLURM resources: resources.<stage_num>.{time,cpus,mem}
    # Emitted as SLURM_TIME_<N>, SLURM_CPUS_<N>, SLURM_MEM_<N>.
    # submit_pipeline.sh reads these and passes them as sbatch CLI flags,
    # which override the #SBATCH directives hardcoded in each .slurm file.
    resources = cfg.get("resources", {})
    if isinstance(resources, dict):
        for stage_num, stage_res in resources.items():
            if not isinstance(stage_res, dict):
                continue
            for res_key, res_suffix in (
                ("time", "TIME"),
                ("cpus", "CPUS"),
                ("mem", "MEM"),
            ):
                val = stage_res.get(res_key, "")
                if val:
                    pairs.append(("SLURM_{0}_{1}".format(res_suffix, stage_num), str(val)))

    return dict(pairs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--vars", metavar="FILE", help="Emit shell KEY=VALUE assignments (skips env-set vars)"
    )
    ap.add_argument(
        "--run-vars",
        metavar="FILE",
        help="Emit run-config KEY=VALUE assignments (skips env-set vars)",
    )
    ap.add_argument(
        "--check-cmds",
        nargs=2,
        metavar=("GROUP", "FILE"),
        help="Emit required command names for GROUP",
    )
    args = ap.parse_args(argv)

    if args.vars:
        if not os.path.isfile(args.vars):
            print("[hpc_config] ERROR: config not found: {0}".format(args.vars), file=sys.stderr)
            return 1
        try:
            cfg = _load(args.vars)
        except Exception as exc:
            print(
                "[hpc_config] ERROR: failed to parse {0}: {1}".format(args.vars, exc),
                file=sys.stderr,
            )
            return 1
        for key, val in _get_shell_vars(cfg).items():
            if key not in os.environ and val:
                print("{0}={1}".format(key, shlex.quote(val)))
        return 0

    if args.run_vars:
        if not os.path.isfile(args.run_vars):
            print(
                "[hpc_config] ERROR: run config not found: {0}".format(args.run_vars),
                file=sys.stderr,
            )
            return 1
        try:
            cfg = _load(args.run_vars)
        except Exception as exc:
            print(
                "[hpc_config] ERROR: failed to parse {0}: {1}".format(args.run_vars, exc),
                file=sys.stderr,
            )
            return 1
        for key, val in _get_run_shell_vars(cfg).items():
            if key not in os.environ and val:
                print("{0}={1}".format(key, shlex.quote(val)))
        return 0

    if args.check_cmds:
        group, path = args.check_cmds
        if not os.path.isfile(path):
            print("[hpc_config] ERROR: config not found: {0}".format(path), file=sys.stderr)
            return 1
        try:
            cfg = _load(path)
        except Exception as exc:
            print("[hpc_config] ERROR: {0}".format(exc), file=sys.stderr)
            return 1
        cmds = _get_required_commands(cfg, group)
        if not cmds:
            print(
                "[hpc_config] WARNING: no required_commands for group '{0}'".format(group),
                file=sys.stderr,
            )
        print(" ".join(cmds))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
