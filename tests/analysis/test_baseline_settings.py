"""Drift guard for the reported baseline settings (R2-6).

`config/baselines/baseline_settings.yaml` is quoted directly in the response
letter and will become a supplementary table, so a silent edit to any pipeline
script would make a published statement false.  These tests assert that every
flag the YAML claims is still present in the script it names.

The guard is not hypothetical: transcribing these settings by hand produced a
real error, in which the training-label alignment's
``--outFilterMultimapNmax 100 --winAnchorMultimapNmax 100`` was attributed to
the TEtranscripts input, which actually received ``--outFilterMultimapNmax
1000 --outSAMprimaryFlag AllBestScore``.  The two alignments are therefore
tested separately and asserted to differ.
"""

from pathlib import Path

import pytest
import yaml

from trap.config.config import PROJ_ROOT

pytestmark = pytest.mark.unit

SETTINGS_PATH = PROJ_ROOT / "config" / "baselines" / "baseline_settings.yaml"


@pytest.fixture(scope="module")
def settings() -> dict:
    if not SETTINGS_PATH.exists():  # pragma: no cover - file is committed
        pytest.skip(f"{SETTINGS_PATH} not present")
    return yaml.safe_load(SETTINGS_PATH.read_text())


def _scripts_of(entry: dict) -> list[Path]:
    """Every script path an entry names, as absolute paths."""
    named = []
    for key in ("script", "index_script"):
        if entry.get(key):
            named.append(entry[key])
    named.extend(entry.get("scripts", []))
    return [PROJ_ROOT / name for name in named]


def _entries_with_flags(settings: dict):
    """Yield ``(label, entry)`` for every entry carrying ``assert_flags``."""
    yield "training_label_alignment", settings["training_label_alignment"]
    for benchmark in ("synthetic_benchmark", "hek293_5race_benchmark"):
        section = settings.get(benchmark, {})
        for key, value in section.items():
            if isinstance(value, dict) and value.get("assert_flags"):
                yield f"{benchmark}.{key}", value
            if key == "methods":
                for method in value:
                    if method.get("assert_flags"):
                        yield f"{benchmark}.{method['name']}", method


def test_settings_file_parses(settings):
    assert settings["modules"]["star"] == "2.7.11b"
    assert settings["synthetic_benchmark"]["shared_alignment"]["tool"] == "STAR"


def test_every_named_script_exists(settings):
    missing = []
    for label, entry in _entries_with_flags(settings):
        for script in _scripts_of(entry):
            if not script.exists():
                missing.append(f"{label}: {script.relative_to(PROJ_ROOT)}")
    assert not missing, f"scripts named in the settings file are absent: {missing}"


def test_asserted_flags_are_present_in_their_scripts(settings):
    """Each claimed flag must appear verbatim in at least one named script."""
    failures = []
    for label, entry in _entries_with_flags(settings):
        scripts = _scripts_of(entry)
        if not scripts:
            failures.append(f"{label}: declares assert_flags but names no script")
            continue
        blobs = [script.read_text() for script in scripts if script.exists()]
        for flag in entry["assert_flags"]:
            if not any(flag in blob for blob in blobs):
                failures.append(f"{label}: {flag!r} not found in {[str(s) for s in scripts]}")
    assert not failures, "reported settings no longer match the scripts:\n" + "\n".join(failures)


def test_the_two_star_alignments_are_distinct(settings):
    """The label alignment and the benchmark alignment must not be conflated."""
    label = settings["training_label_alignment"]["key_parameters"]
    bench = settings["synthetic_benchmark"]["shared_alignment"]["key_parameters"]
    assert label["outFilterMultimapNmax"] == 100
    assert bench["outFilterMultimapNmax"] == 1000
    assert label["outFilterMultimapNmax"] != bench["outFilterMultimapNmax"]
    # winAnchorMultimapNmax is set for labelling only; asserting its absence
    # from the benchmark alignment keeps the response from re-acquiring it.
    assert "winAnchorMultimapNmax" in label
    assert "winAnchorMultimapNmax" not in bench
    align_script = (
        PROJ_ROOT / settings["synthetic_benchmark"]["shared_alignment"]["script"]
    ).read_text()
    assert "winAnchorMultimapNmax" not in align_script


def test_every_alignment_dependent_method_shares_one_bam(settings):
    """R2-6's fairness premise: no comparator gets its own alignment."""
    methods = settings["synthetic_benchmark"]["methods"]
    alignment_dependent = {"HTSeq", "TEtranscripts", "L1EM"}
    named = {m["name"] for m in methods}
    assert alignment_dependent <= named
    # None of them may declare their own aligner command.
    for method in methods:
        if method["name"] in alignment_dependent:
            assert "align_command" not in method
            assert "index_command" not in method or method["name"] == "Salmon"


def test_multimapping_disclosure_is_present_for_every_method(settings):
    """Every comparator must state how multi-mappers were handled."""
    for method in settings["synthetic_benchmark"]["methods"]:
        assert method.get("multimapping"), f"{method['name']} lacks a multimapping statement"


def test_htseq_default_is_recorded_as_discarding_multimappers(settings):
    """The concession must stay in the file, since the response depends on it."""
    htseq = next(m for m in settings["synthetic_benchmark"]["methods"] if m["name"] == "HTSeq")
    assert "none" in str(htseq["key_parameters"]["nonunique"])
    assert "NOT retained" in htseq["multimapping"]
    # And the sensitivity run that would answer it must remain listed.
    runs = {r["tool"]: r for r in settings["sensitivity_runs"]}
    assert runs["HTSeq"]["change"] == "--nonunique all"


def test_tetranscripts_uses_its_own_te_recommendation(settings):
    """TEtranscripts needs no change; the rebuttal depends on --mode multi."""
    te = next(
        m for m in settings["synthetic_benchmark"]["methods"] if m["name"] == "TEtranscripts"
    )
    assert te["key_parameters"]["mode"] == "multi"
    assert "RETAINED" in te["multimapping"]
    script = (PROJ_ROOT / te["script"]).read_text()
    assert 'TE_MODE="${TE_MODE:-multi}"' in script


def test_l1em_runs_upstream_defaults(settings):
    """The L1EM rebuttal depends on using the tool's own recommended settings."""
    l1em = next(m for m in settings["synthetic_benchmark"]["methods"] if m["name"] == "L1EM")
    assert l1em["key_parameters"]["flanking_bp"] == 400
    assert l1em["key_parameters"]["EM_threshold"] == pytest.approx(1e-7)
    parameters = PROJ_ROOT / l1em["parameters_file"]
    if parameters.exists():
        text = parameters.read_text()
        assert "realignNM=3" in text
        assert "max_start2start_len=500" in text
        assert "EM_threshold=1e-7" in text


def test_versions_are_marked_verify_rather_than_guessed(settings):
    """Unknown versions must be explicit, not invented."""
    for method in settings["synthetic_benchmark"]["methods"]:
        if "version" in method:
            assert method["version"], f"{method['name']} has an empty version field"
    # The dev-env Salmon pin is not a real release and must not be quoted.
    dev_env = PROJ_ROOT / "envs" / "environment-dev.yml"
    if dev_env.exists() and "salmon=2.1.2" in dev_env.read_text():
        salmon = next(
            m for m in settings["synthetic_benchmark"]["methods"] if m["name"] == "Salmon"
        )
        assert salmon["version"] == "VERIFY"
