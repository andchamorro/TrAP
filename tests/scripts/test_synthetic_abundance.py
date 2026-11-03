"""Tests for the pure functions of ``scripts/python/synthetic_abundance.py``.

The script joins ground-truth copy counts to each method's estimate by element key, so
the loaders and key normalisation are what decide whether the join is 1:1.  The BAM and
bedtools steps are not exercised here; they need external tools and a genome.
"""

import importlib.util
import os
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def sa():
    spec = importlib.util.spec_from_file_location(
        "synthetic_abundance", os.path.join(ROOT, "scripts", "python", "synthetic_abundance.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, text):
    path.write_text(text)
    return str(path)


@pytest.mark.unit
class TestKeys:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("UID107", "UID107"),
            ("UID-107", "UID107"),
            ("UID107:chr1:100-200", "UID107"),
            ("UID-107.1", "UID107"),
        ],
    )
    def test_norm_uid(self, sa, raw, expected):
        assert sa._norm_uid(raw) == expected

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("L1PA6.1::chr1:100-6100", "L1PA6"),
            ("L1HS.0::chr2:5-9", "L1HS"),
            ("UID107", "UID107"),  # a bare L1Base UID encodes no subfamily
        ],
    )
    def test_subfamily(self, sa, key, expected):
        assert sa.subfamily(key) == expected


@pytest.mark.unit
class TestLoaders:
    def test_simulated_counts_skips_header_and_short_rows(self, sa, tmp_path):
        p = _write(tmp_path / "c.tsv", "l1_id\tcount\nUID1\t3\nUID2\t0\nbad\n")
        assert sa.load_simulated_counts(p) == {"UID1": 3, "UID2": 0}

    def test_simulated_bed_counts_insertions_per_element(self, sa, tmp_path):
        p = _write(tmp_path / "i.bed", "chr1\t1\t2\tUID1\nchr1\t5\t6\tUID1\nchr2\t1\t2\tUID2\n")
        assert dict(sa.load_simulated_bed(p)) == {"UID1": 2, "UID2": 1}

    def test_salmon_reads_numreads_column(self, sa, tmp_path):
        p = _write(
            tmp_path / "quant.sf",
            "Name\tLength\tEffectiveLength\tTPM\tNumReads\nUID1\t6000\t5800\t10.5\t42.5\n",
        )
        assert sa.load_salmon(p) == {"UID1": 42.5}

    def test_l1em_sums_sense_categories_and_groups_by_uid(self, sa, tmp_path):
        p = _write(
            tmp_path / "full_counts.txt",
            "name\tonly\t3prunon\tpassive_sense\tpassive_antisense\trunon\n"
            "UID-1:a\t1\t2\t3\t4\t100\n"
            "UID-1:b\t10\t0\t0\t0\t100\n"
            "UID-2:a\t0.5\t0\t0\t0\t100\n",
        )
        # runon is not a sense category, so the 100s must not leak in
        assert sa.load_l1em(p) == {"UID1": 20.0, "UID2": 0.5}

    def test_tecount_keeps_only_uid_features_and_skips_comments(self, sa, tmp_path):
        p = _write(tmp_path / "t.cntTable", "#header\nUID-3\t7\nAluSx\t99\n")
        assert sa.load_tecount(p) == {"UID3": 7.0}

    def test_htseq_drops_summary_rows(self, sa, tmp_path):
        p = _write(tmp_path / "h.csv", "UID-4\t5\n__no_feature\t1000\n__ambiguous\t3\n")
        assert sa.load_htseq(p) == {"UID4": 5.0}


@pytest.mark.unit
class TestNormalisation:
    BUNDLE = (500.0, {"UID1": (40.0, 5000.0, 2.0), "UID2": (7.0, 0.0, 0.0)})

    def test_expected_reads_is_efflen_times_coverage_over_insert(self, sa):
        assert sa._expected_reads(self.BUNDLE, "UID1") == pytest.approx(5000.0 * 2.0 / 500.0)

    def test_normalize_divides_by_expected_reads(self, sa):
        # k * raw / (effLen * coverage / insert) = 0.5 * 100 / 20
        assert sa._normalize(100.0, self.BUNDLE, "UID1") == pytest.approx(2.5)

    def test_zero_denominator_falls_back_to_one_instead_of_dividing_by_zero(self, sa):
        assert sa._normalize(100.0, self.BUNDLE, "UID2") == pytest.approx(50.0)

    def test_absent_element_is_treated_like_zero_coverage(self, sa):
        assert sa._normalize(100.0, self.BUNDLE, "UIDX") == pytest.approx(50.0)

    def test_zero_insert_size_does_not_raise(self, sa):
        assert sa._expected_reads((0.0, {"UID1": (1.0, 1.0, 1.0)}), "UID1") == 0.0

    def test_legacy_simulated_reads(self, sa):
        # 0.25 * count / expected_reads = 0.25 * 40 / 20
        assert sa.legacy_simulated_reads(self.BUNDLE, "UID1") == pytest.approx(0.5)


@pytest.mark.unit
class TestExperimentPaths:
    def test_token_maps_both_models(self, sa):
        assert sa.experiment_token("transcript") == "l1-transcript-pool"
        assert sa.experiment_token("insert") == "l1-host-insert"

    def test_unknown_model_exits_with_a_message(self, sa):
        with pytest.raises(SystemExit, match="unknown SIM_MODEL"):
            sa.experiment_token("bogus")

    def test_results_dir(self, sa):
        assert sa.experiment_results_dir("insert") == (
            "results/synthetic_validation/l1-host-insert"
        )

    def test_new_layout_is_used_when_neither_tree_exists(self, sa, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert sa.experiment_dir("transcript", "l1base", "chr1") == (
            "data/ref/synthetic/l1-transcript-pool.l1base"
        )

    def test_legacy_tree_is_used_only_when_the_new_one_is_absent(self, sa, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        legacy = "data/ref/GRCh38.p14.genome.chr1.withdel.l1base.insert"
        os.makedirs(legacy)
        assert sa.experiment_dir("insert", "l1base", "chr1") == legacy
        os.makedirs("data/ref/synthetic/l1-host-insert.l1base")
        assert sa.experiment_dir("insert", "l1base", "chr1") == (
            "data/ref/synthetic/l1-host-insert.l1base"
        )


@pytest.mark.unit
class TestDiscovery:
    def test_cells_are_sorted_and_deduplicated_across_bed_and_counts(self, sa, tmp_path):
        d = tmp_path
        for name in (
            "GRCh38.p14.chr1.insert_level_10_delprob_0.10.bed",
            "GRCh38.p14.chr1.insert_level_10_delprob_0.10.counts.tsv",
            "GRCh38.p14.chr1.insert_level_5_delprob_0.00.counts.tsv",
            "GRCh38.p14.chr2.insert_level_99_delprob_0.50.bed",  # other chromosome
        ):
            (d / name).write_text("")
        assert sa.discover_cells(str(d), "chr1") == [("10", "0.10"), ("5", "0.00")]

    def test_find_method_file_takes_the_first_candidate_that_exists(self, sa, tmp_path):
        (tmp_path / "salmon/art/base").mkdir(parents=True)
        (tmp_path / "salmon/art/base/quant.sf").write_text("")
        got = sa.find_method_file(
            str(tmp_path), ["salmon/unfiltered", "salmon/art"], "base", "quant.sf"
        )
        assert got.endswith("salmon/art/base/quant.sf")
        assert sa.find_method_file(str(tmp_path), ["nope"], "base", "quant.sf") is None

    def test_env_override_replaces_the_default_subdirs(self, sa, monkeypatch):
        monkeypatch.setenv("X_SUBDIRS", os.pathsep.join(["a", "b"]))
        assert sa._subdir_candidates("X_SUBDIRS", ["default"]) == ["a", "b"]
        monkeypatch.delenv("X_SUBDIRS")
        assert sa._subdir_candidates("X_SUBDIRS", ["default"]) == ["default"]


@pytest.mark.unit
class TestFeatureCache:
    BUNDLE = (480.0, {"UID1": (12, 5000.12345, 2.123456789)})

    def test_round_trip_rounds_floats_as_documented(self, sa, tmp_path):
        cache = str(tmp_path / "sub" / sa.CACHE_NAME)
        sa.save_feature_cache(cache, self.BUNDLE)
        insert_size, feats = sa.load_feature_cache(cache)
        assert insert_size == 480.0
        assert feats["UID1"] == (12, 5000.1235, 2.123457)

    def test_missing_cache_is_none(self, sa, tmp_path):
        assert sa.load_feature_cache(str(tmp_path / "absent.json")) is None

    def test_corrupt_cache_is_none_rather_than_an_exception(self, sa, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{not json")
        assert sa.load_feature_cache(str(p)) is None

    def test_stale_version_forces_a_recompute(self, sa, tmp_path):
        p = tmp_path / "c.json"
        p.write_text('{"v": -1, "insert_size": 1, "features": {}}')
        assert sa.load_feature_cache(str(p)) is None

    def test_cache_older_than_its_dependency_is_stale(self, sa, tmp_path):
        cache = str(tmp_path / sa.CACHE_NAME)
        sa.save_feature_cache(cache, self.BUNDLE)
        dep = tmp_path / "reads.bam"
        dep.write_text("")
        future = time.time() + 100
        os.utime(dep, (future, future))
        assert sa.load_feature_cache(cache, str(dep)) is None
        assert sa.load_feature_cache(cache) is not None
