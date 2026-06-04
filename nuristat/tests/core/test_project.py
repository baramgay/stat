"""Tests for project save/load using ZIP-based ``.swb`` archives.

Covers save, load, round-trip identity, manifest contents, and
error handling for corrupted or missing files.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nuristat.core.audit import AuditLog
from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import ProjectError
from nuristat.core.project import SCHEMA_VERSION, load, save
from nuristat.core.typing import MeasureType, Role, StorageType
from nuristat.core.variable import VariableMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_dataset() -> Dataset:
    """Create a sample dataset with varied metadata."""
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, 40, 45],
        "gender": ["M", "F", "F", "M", "F"],
        "bp_systolic": [120, 130, 125, 140, 135],
        "bp_diastolic": [80, 85, 82, 90, 88],
        "treatment": ["A", "B", "A", "B", "A"],
        "response": [1.2, 3.4, 2.1, 4.5, 3.0],
        "enrolled": [True, False, True, True, False],
        "enroll_date": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-01"]),
    })
    ds = Dataset(
        data=df,
        name="clinical_trial",
        description="Phase II clinical trial dataset for testing save/load.",
    )
    # Customize some metadata
    ds.update_variable_meta("patient_id", label="Patient ID", role=Role.ID)
    ds.update_variable_meta("age", label="Age (years)", unit="years", role=Role.INPUT, allowed_min=0, allowed_max=120)
    ds.update_variable_meta("gender", label="Gender", measure=MeasureType.NOMINAL, value_labels={"M": "Male", "F": "Female"})
    ds.update_variable_meta("bp_systolic", label="Systolic BP", unit="mmHg", role=Role.INPUT)
    ds.update_variable_meta("bp_diastolic", label="Diastolic BP", unit="mmHg", role=Role.INPUT)
    ds.update_variable_meta("treatment", label="Treatment Group", measure=MeasureType.NOMINAL, role=Role.SPLIT)
    ds.update_variable_meta("response", label="Response Score", unit="points", role=Role.TARGET)
    ds.update_variable_meta("enrolled", label="Currently Enrolled", measure=MeasureType.BINARY, role=Role.NONE)
    ds.update_variable_meta("enroll_date", label="Enrollment Date", role=Role.NONE)
    return ds


@pytest.fixture
def sample_dataset() -> Dataset:
    return _sample_dataset()


@pytest.fixture
def tmp_swb_path(tmp_path: Path) -> Path:
    return tmp_path / "test_project.swb"


# ---------------------------------------------------------------------------
# Save tests
# ---------------------------------------------------------------------------

class TestSave:
    """Tests for the save function."""

    def test_creates_file(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """save() creates the .swb file."""
        save(sample_dataset, tmp_swb_path)
        assert tmp_swb_path.exists()

    def test_file_is_valid_zip(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The saved file is a valid ZIP archive."""
        save(sample_dataset, tmp_swb_path)
        assert zipfile.is_zipfile(tmp_swb_path)

    def test_contains_manifest(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The archive contains manifest.json."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            assert "manifest.json" in zf.namelist()

    def test_contains_active_parquet(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The archive contains data/active.parquet."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            assert "data/active.parquet" in zf.namelist()

    def test_contains_backup_csv(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The archive contains data/backup.csv."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            assert "data/backup.csv" in zf.namelist()

    def test_contains_variables_json(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The archive contains metadata/variables.json."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            assert "metadata/variables.json" in zf.namelist()

    def test_contains_dataset_json(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The archive contains metadata/dataset.json."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            assert "metadata/dataset.json" in zf.namelist()

    def test_manifest_schema_version(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """manifest.json contains the correct schema version."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == SCHEMA_VERSION

    def test_manifest_dataset_info(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """manifest.json contains dataset dimensions."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["dataset_name"] == "clinical_trial"
        assert manifest["n_rows"] == 5
        assert manifest["n_cols"] == 9

    def test_manifest_has_timestamp(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """manifest.json contains a save timestamp."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert "saved_at" in manifest
        assert "T" in manifest["saved_at"]

    def test_manifest_with_audit_log(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Audit log entries are included in the manifest."""
        audit = AuditLog()
        audit.append("variable_rename", {"old": "x", "new": "age"})
        audit.append("analysis_run", {"procedure": "t_test"})
        save(sample_dataset, tmp_swb_path, audit=audit)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert "audit_log" in manifest
        assert len(manifest["audit_log"]) == 2
        assert manifest["audit_log"][0]["action"] == "variable_rename"

    def test_active_parquet_readable(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The Parquet file inside the archive is readable."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            import io
            df = pd.read_parquet(io.BytesIO(zf.read("data/active.parquet")))
        assert df.shape == (5, 9)
        assert list(df.columns) == list(sample_dataset.data.columns)

    def test_backup_csv_readable(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """The CSV backup inside the archive is readable."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            import io
            df = pd.read_csv(io.StringIO(zf.read("data/backup.csv").decode("utf-8")))
        assert df.shape[0] == 5

    def test_variables_json_contains_all_vars(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """variables.json contains metadata for all variables."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            variables = json.loads(zf.read("metadata/variables.json").decode("utf-8"))
        assert set(variables.keys()) == set(sample_dataset.data.columns)

    def test_variables_json_structure(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Each variable metadata has the expected fields."""
        save(sample_dataset, tmp_swb_path)
        with zipfile.ZipFile(tmp_swb_path, "r") as zf:
            variables = json.loads(zf.read("metadata/variables.json").decode("utf-8"))
        age_meta = variables["age"]
        assert age_meta["name"] == "age"
        assert age_meta["label"] == "Age (years)"
        assert age_meta["unit"] == "years"
        assert age_meta["role"] == "input"

    def test_creates_parent_directories(self, sample_dataset: Dataset, tmp_path: Path) -> None:
        """save() creates parent directories if they don't exist."""
        nested = tmp_path / "sub" / "dir" / "nested.swb"
        save(sample_dataset, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------

class TestLoad:
    """Tests for the load function."""

    def test_loads_dataset(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """load() returns a Dataset."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert isinstance(loaded, Dataset)

    def test_load_shape(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset has the same shape."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.shape == sample_dataset.shape

    def test_load_columns(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset has the same columns."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert list(loaded.data.columns) == list(sample_dataset.data.columns)

    def test_load_data_values(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves data values."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        pd.testing.assert_frame_equal(
            loaded.data.reset_index(drop=True),
            sample_dataset.data.reset_index(drop=True),
            check_dtype=False,
        )

    def test_load_dataset_name(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves the name."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.name == sample_dataset.name

    def test_load_dataset_description(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves the description."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.description == sample_dataset.description

    def test_load_variable_count(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset has the same number of variables."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.n_cols == sample_dataset.n_cols

    def test_load_variable_names(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset has the same variable names."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert set(loaded.variables.keys()) == set(sample_dataset.variables.keys())

    def test_load_variable_labels(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves variable labels."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["age"].label == "Age (years)"
        assert loaded.variables["bp_systolic"].label == "Systolic BP"

    def test_load_variable_roles(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves variable roles."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["patient_id"].role == Role.ID
        assert loaded.variables["age"].role == Role.INPUT
        assert loaded.variables["response"].role == Role.TARGET

    def test_load_variable_units(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves variable units."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["age"].unit == "years"
        assert loaded.variables["bp_systolic"].unit == "mmHg"

    def test_load_value_labels(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset preserves value labels."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["gender"].value_labels == {"M": "Male", "F": "Female"}

    def test_load_storage_types(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset restores storage types."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["age"].storage_type == StorageType.INTEGER
        assert loaded.variables["gender"].storage_type == StorageType.STRING

    def test_load_measure_types(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset restores measure types."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["gender"].measure == MeasureType.NOMINAL

    def test_load_allowed_range(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset restores allowed_min and allowed_max."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.variables["age"].allowed_min == 0
        assert loaded.variables["age"].allowed_max == 120

    def test_load_not_dirty(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Loaded dataset is not marked as dirty."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert not loaded.is_dirty

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises ProjectError."""
        with pytest.raises(ProjectError):
            load(tmp_path / "does_not_exist.swb")


# ---------------------------------------------------------------------------
# Corruption / error tests
# ---------------------------------------------------------------------------

class TestCorruption:
    """Tests for handling corrupted or invalid project files."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Loading a non-existent path raises ProjectError."""
        with pytest.raises(ProjectError):
            load(tmp_path / "missing.swb")

    def test_not_a_zip_file(self, tmp_path: Path) -> None:
        """Loading a non-ZIP file raises ProjectError."""
        bad = tmp_path / "not_a_zip.swb"
        bad.write_text("this is not a zip file")
        with pytest.raises(ProjectError):
            load(bad)

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Archive missing manifest.json raises ProjectError."""
        bad = tmp_path / "no_manifest.swb"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("data/active.parquet", b"fake")
        with pytest.raises(ProjectError):
            load(bad)

    def test_missing_parquet(self, tmp_path: Path) -> None:
        """Archive missing active.parquet raises ProjectError."""
        bad = tmp_path / "no_parquet.swb"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": SCHEMA_VERSION}))
        with pytest.raises(ProjectError):
            load(bad)

    def test_unsupported_schema_version(self, sample_dataset: Dataset, tmp_path: Path) -> None:
        """Archive with unsupported schema version raises ProjectError."""
        bad = tmp_path / "bad_version.swb"
        import io
        parquet_bytes = sample_dataset.data.to_parquet(index=False)
        variables_dict = {
            name: meta.to_dict()
            for name, meta in sample_dataset.variables.items()
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": "99.0"}))
            zf.writestr("data/active.parquet", parquet_bytes)
            zf.writestr("metadata/variables.json", json.dumps(variables_dict))
            zf.writestr("metadata/dataset.json", json.dumps({
                "name": "x",
                "shape": [5, 9],
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }))
        with pytest.raises(ProjectError) as exc_info:
            load(bad)
        assert "99.0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Comprehensive round-trip tests."""

    def test_full_round_trip_identity(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """A full save/load cycle produces an identical dataset."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)

        # Data
        assert loaded.shape == sample_dataset.shape
        assert list(loaded.data.columns) == list(sample_dataset.data.columns)
        pd.testing.assert_frame_equal(
            loaded.data.reset_index(drop=True),
            sample_dataset.data.reset_index(drop=True),
            check_dtype=False,
        )

        # Metadata
        assert loaded.name == sample_dataset.name
        assert loaded.description == sample_dataset.description
        assert loaded.n_cols == sample_dataset.n_cols

        # Per-variable metadata
        for name in sample_dataset.data.columns:
            orig = sample_dataset.variables[name]
            rest = loaded.variables[name]
            assert rest.name == orig.name
            assert rest.label == orig.label
            assert rest.storage_type == orig.storage_type
            assert rest.measure == orig.measure
            assert rest.role == orig.role
            assert rest.unit == orig.unit
            assert rest.value_labels == orig.value_labels
            assert rest.allowed_min == orig.allowed_min
            assert rest.allowed_max == orig.allowed_max

    def test_round_trip_timestamps(self, sample_dataset: Dataset, tmp_swb_path: Path) -> None:
        """Timestamps are preserved across round-trips."""
        save(sample_dataset, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.created_at == sample_dataset.created_at
        assert loaded.updated_at == sample_dataset.updated_at

    def test_multiple_save_load_cycles(self, sample_dataset: Dataset, tmp_path: Path) -> None:
        """Multiple consecutive save/load cycles are stable."""
        path1 = tmp_path / "cycle1.swb"
        path2 = tmp_path / "cycle2.swb"
        save(sample_dataset, path1)
        loaded1 = load(path1)
        save(loaded1, path2)
        loaded2 = load(path2)
        assert loaded2.shape == sample_dataset.shape
        assert loaded2.name == sample_dataset.name
        pd.testing.assert_frame_equal(
            loaded2.data.reset_index(drop=True),
            sample_dataset.data.reset_index(drop=True),
            check_dtype=False,
        )

    def test_empty_dataset_round_trip(self, tmp_swb_path: Path) -> None:
        """An empty dataset survives round-trip."""
        ds = Dataset(
            data=pd.DataFrame(),
            name="empty",
            description="No data",
        )
        save(ds, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.shape == (0, 0)
        assert loaded.name == "empty"
        assert loaded.n_cols == 0

    def test_single_column_round_trip(self, tmp_swb_path: Path) -> None:
        """A single-column dataset survives round-trip."""
        ds = Dataset(
            data=pd.DataFrame({"x": [1, 2, 3, 4, 5]}),
            name="single",
        )
        save(ds, tmp_swb_path)
        loaded = load(tmp_swb_path)
        assert loaded.shape == (5, 1)
        assert "x" in loaded.data.columns
        assert loaded.variables["x"].storage_type == StorageType.INTEGER
