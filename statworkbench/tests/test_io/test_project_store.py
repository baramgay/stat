"""Tests for project_store module."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import FileReadError, FileWriteError
from statworkbench.core.project import Project
from statworkbench.core.typing import MeasureType
from statworkbench.core.variable import VariableMeta
from statworkbench.io.project_store import (
    DATA_CSV,
    DATA_PARQUET,
    MANIFEST_NAME,
    META_DATASET,
    META_PROJECT,
    META_VARIABLES,
    SCHEMA_VERSION,
    ProjectStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset for project testing."""
    df = pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "age": [45, 52, 38, 61, 29],
        "sex": ["M", "F", "M", "F", "M"],
        "systolic_bp": [120, 135, 128, 142, 118],
        "treatment": ["Drug", "Placebo", "Drug", "Placebo", "Drug"],
    })
    return Dataset(data=df, name="ClinicalTrial", source_info={"format": "test"})


@pytest.fixture
def sample_project(sample_dataset: Dataset) -> Project:
    """Create a sample project with a dataset."""
    proj = Project(
        name="TestProject",
        dataset=sample_dataset,
        settings={"decimal_places": 2, "confidence_level": 0.95},
    )
    return proj


@pytest.fixture
def sample_project_with_measures(sample_dataset: Dataset) -> Project:
    """Create a project with custom variable measures."""
    sample_dataset.variables["age"].measure = MeasureType.SCALE
    sample_dataset.variables["sex"].measure = MeasureType.NOMINAL
    sample_dataset.variables["systolic_bp"].measure = MeasureType.SCALE
    sample_dataset.variables["treatment"].measure = MeasureType.NOMINAL
    sample_dataset.variables["patient_id"].measure = MeasureType.NOMINAL

    proj = Project(
        name="ProjectWithMeasures",
        dataset=sample_dataset,
    )
    return proj


# ---------------------------------------------------------------------------
# save_project tests
# ---------------------------------------------------------------------------


class TestSaveProject:
    def test_saves_to_swb_file(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "test_project.swb")
        ProjectStore.save_project(sample_project, save_path)
        assert Path(save_path).exists()

    def test_auto_adds_swb_extension(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "no_extension")
        ProjectStore.save_project(sample_project, save_path)
        assert Path(save_path + ".swb").exists()

    def test_creates_valid_zip(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "valid_zip.swb")
        ProjectStore.save_project(sample_project, save_path)

        with zipfile.ZipFile(save_path, "r") as zf:
            names = zf.namelist()
            assert MANIFEST_NAME in names
            assert DATA_CSV in names
            assert META_VARIABLES in names
            assert META_DATASET in names
            assert META_PROJECT in names

    def test_manifest_content(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "manifest_test.swb")
        ProjectStore.save_project(sample_project, save_path)

        with zipfile.ZipFile(save_path, "r") as zf:
            manifest = json.loads(zf.read(MANIFEST_NAME))
            assert manifest["schema_version"] == SCHEMA_VERSION
            assert manifest["application"] == "StatWorkbench"
            assert DATA_CSV in manifest["entries"]

    def test_csv_backup_matches_data(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "csv_backup_test.swb")
        ProjectStore.save_project(sample_project, save_path)

        with zipfile.ZipFile(save_path, "r") as zf:
            csv_content = zf.read(DATA_CSV).decode("utf-8")
            df_from_csv = pd.read_csv(pd.io.common.StringIO(csv_content))

        pd.testing.assert_frame_equal(
            df_from_csv.sort_index(axis=1),
            sample_project.dataset.data.sort_index(axis=1),
        )

    def test_variables_json(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "vars_test.swb")
        ProjectStore.save_project(sample_project, save_path)

        with zipfile.ZipFile(save_path, "r") as zf:
            vars_dict = json.loads(zf.read(META_VARIABLES))

        assert len(vars_dict) == 5
        assert "patient_id" in vars_dict
        assert "age" in vars_dict
        assert vars_dict["age"]["storage_type"] == "integer"

    def test_project_json(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "proj_json_test.swb")
        ProjectStore.save_project(sample_project, save_path)

        with zipfile.ZipFile(save_path, "r") as zf:
            proj_dict = json.loads(zf.read(META_PROJECT))

        assert proj_dict["name"] == "TestProject"
        assert proj_dict["schema_version"] == SCHEMA_VERSION
        assert proj_dict["settings"]["decimal_places"] == 2

    def test_raises_when_no_dataset(self, tmp_path: Path) -> None:
        proj = Project(name="EmptyProject", dataset=None)
        save_path = str(tmp_path / "empty.swb")
        with pytest.raises((FileReadError, FileWriteError)) as exc_info:
            ProjectStore.save_project(proj, save_path)
        assert "데이터셋이 없습니다" in str(exc_info.value)

    def test_creates_parent_directories(self, sample_project: Project, tmp_path: Path) -> None:
        deep_path = str(tmp_path / "a" / "b" / "c" / "deep.swb")
        ProjectStore.save_project(sample_project, deep_path)
        assert Path(deep_path).exists()


# ---------------------------------------------------------------------------
# load_project tests
# ---------------------------------------------------------------------------


class TestLoadProject:
    def test_roundtrip(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "roundtrip.swb")
        ProjectStore.save_project(sample_project, save_path)

        loaded = ProjectStore.load_project(save_path)
        assert loaded.name == sample_project.name
        assert loaded.dataset is not None
        assert loaded.dataset.n_rows == sample_project.dataset.n_rows
        assert loaded.dataset.n_vars == sample_project.dataset.n_vars
        assert list(loaded.dataset.data.columns) == list(
            sample_project.dataset.data.columns
        )

    def test_data_integrity(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "integrity.swb")
        ProjectStore.save_project(sample_project, save_path)

        loaded = ProjectStore.load_project(save_path)
        assert loaded.dataset is not None
        pd.testing.assert_frame_equal(
            loaded.dataset.data.sort_index(axis=1),
            sample_project.dataset.data.sort_index(axis=1),
        )

    def test_variable_metadata_preserved(
        self, sample_project_with_measures: Project, tmp_path: Path
    ) -> None:
        proj = sample_project_with_measures
        save_path = str(tmp_path / "measures.swb")
        ProjectStore.save_project(proj, save_path)

        loaded = ProjectStore.load_project(save_path)
        assert loaded.dataset is not None
        vars_loaded = loaded.dataset.variables

        assert vars_loaded["age"].measure == MeasureType.SCALE
        assert vars_loaded["sex"].measure == MeasureType.NOMINAL
        assert vars_loaded["treatment"].measure == MeasureType.NOMINAL

    def test_dataset_name_preserved(self, sample_project: Project, tmp_path: Path) -> None:
        save_path = str(tmp_path / "name_preserve.swb")
        ProjectStore.save_project(sample_project, save_path)

        loaded = ProjectStore.load_project(save_path)
        assert loaded.dataset is not None
        assert loaded.dataset.name == "ClinicalTrial"

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileReadError) as exc_info:
            ProjectStore.load_project("/nonexistent/project.swb")
        assert "존재하지 않습니다" in str(exc_info.value)

    def test_bad_zip_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "not_a_zip.swb"
        bad_file.write_text("this is not a zip file")
        with pytest.raises(FileReadError) as exc_info:
            ProjectStore.load_project(str(bad_file))
        assert "손상된" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestProjectStoreIntegration:
    def test_multiple_save_load_cycles(self, sample_project: Project, tmp_path: Path) -> None:
        """Multiple save/load cycles should preserve data exactly."""
        path1 = str(tmp_path / "cycle1.swb")
        path2 = str(tmp_path / "cycle2.swb")

        ProjectStore.save_project(sample_project, path1)
        loaded1 = ProjectStore.load_project(path1)

        ProjectStore.save_project(loaded1, path2)
        loaded2 = ProjectStore.load_project(path2)

        assert loaded2.dataset is not None
        pd.testing.assert_frame_equal(
            loaded2.dataset.data.sort_index(axis=1),
            sample_project.dataset.data.sort_index(axis=1),
        )
        assert loaded2.name == sample_project.name

    def test_dataset_with_special_characters(self, tmp_path: Path) -> None:
        """Test saving/loading data with special characters."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie<>"],
            "description": ["Line1\nLine2", "Tab\there", "Quote\"test"],
            "value": [1.5, 2.5, 3.5],
        })
        ds = Dataset(data=df, name="SpecialChars")
        proj = Project(name="SpecialProject", dataset=ds)

        save_path = str(tmp_path / "special.swb")
        ProjectStore.save_project(proj, save_path)
        loaded = ProjectStore.load_project(save_path)

        assert loaded.dataset is not None
        pd.testing.assert_frame_equal(
            loaded.dataset.data.sort_index(axis=1),
            df.sort_index(axis=1),
        )

    def test_numeric_dataframe(self, tmp_path: Path) -> None:
        """Test with purely numeric DataFrame."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": [100, 200, 300, 400, 500],
        })
        ds = Dataset(data=df, name="NumericData")
        proj = Project(name="NumericProject", dataset=ds)

        save_path = str(tmp_path / "numeric.swb")
        ProjectStore.save_project(proj, save_path)
        loaded = ProjectStore.load_project(save_path)

        assert loaded.dataset is not None
        assert loaded.dataset.n_rows == 5
        assert loaded.dataset.n_vars == 3
        assert list(loaded.dataset.data.columns) == ["A", "B", "C"]
