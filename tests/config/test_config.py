"""Tests for trap.config.config — project path constants."""

from pathlib import Path

import pytest

from trap.config.config import (
    CONFIG_DIR,
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


class TestProjectPaths:
    """Verify all project path constants are consistent."""

    @pytest.mark.unit
    def test_proj_root_is_path(self):
        assert isinstance(PROJ_ROOT, Path)

    @pytest.mark.unit
    def test_proj_root_exists(self):
        assert PROJ_ROOT.exists()

    @pytest.mark.unit
    def test_proj_root_contains_trap_package(self):
        assert (PROJ_ROOT / "trap").is_dir()

    @pytest.mark.unit
    def test_data_dir_under_proj_root(self):
        assert DATA_DIR == PROJ_ROOT / "data"

    @pytest.mark.unit
    def test_raw_data_dir_under_data(self):
        assert RAW_DATA_DIR == DATA_DIR / "raw"

    @pytest.mark.unit
    def test_processed_data_dir_under_data(self):
        assert PROCESSED_DATA_DIR == DATA_DIR / "processed"

    @pytest.mark.unit
    def test_interim_data_dir_under_data(self):
        assert INTERIM_DATA_DIR == DATA_DIR / "interim"

    @pytest.mark.unit
    def test_external_data_dir_under_data(self):
        assert EXTERNAL_DATA_DIR == DATA_DIR / "external"

    @pytest.mark.unit
    def test_models_dir_under_proj_root(self):
        assert MODELS_DIR == PROJ_ROOT / "models"

    @pytest.mark.unit
    def test_config_dir_under_proj_root(self):
        assert CONFIG_DIR == PROJ_ROOT / "config"

    @pytest.mark.unit
    def test_reports_dir_under_proj_root(self):
        assert REPORTS_DIR == PROJ_ROOT / "reports"

    @pytest.mark.unit
    def test_figures_dir_under_reports(self):
        assert FIGURES_DIR == REPORTS_DIR / "figures"

    @pytest.mark.unit
    def test_all_paths_are_absolute(self):
        for p in [
            PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
            INTERIM_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR,
            CONFIG_DIR, REPORTS_DIR, FIGURES_DIR,
        ]:
            assert p.is_absolute(), f"{p} is not absolute"
