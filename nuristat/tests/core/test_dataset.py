"""Tests for :class:`Dataset`.

Covers construction, variable rename, data property with dirty tracking,
add/remove variables, and serialization round-trips.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import DatasetError
from nuristat.core.typing import MeasureType, Role, StorageType
from nuristat.core.variable import VariableMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "gender": ["M", "F", "F", "M", "F"],
        "bp": [120.5, 130.2, 125.0, 140.8, 135.5],
    })


def _sample_dataset() -> Dataset:
    return Dataset(data=_sample_df(), name="clinical")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Tests related to creating Dataset instances."""

    def test_from_dataframe(self) -> None:
        """Construction from a DataFrame auto-creates metadata."""
        df = _sample_df()
        ds = Dataset(data=df)
        assert ds.n_rows == 5
        assert ds.n_cols == 3
        assert set(ds.variables.keys()) == {"age", "gender", "bp"}

    def test_shape_property(self) -> None:
        """shape returns (rows, columns)."""
        ds = _sample_dataset()
        assert ds.shape == (5, 3)

    def test_n_rows_and_n_cols(self) -> None:
        """n_rows and n_cols reflect the DataFrame shape."""
        ds = _sample_dataset()
        assert ds.n_rows == 5
        assert ds.n_cols == 3

    def test_len(self) -> None:
        """len(dataset) returns the row count."""
        ds = _sample_dataset()
        assert len(ds) == 5

    def test_name_and_description(self) -> None:
        """name and description are stored correctly."""
        ds = Dataset(data=_sample_df(), name="my_data", description="A test dataset")
        assert ds.name == "my_data"
        assert ds.description == "A test dataset"

    def test_is_dirty_after_construction(self) -> None:
        """A freshly constructed dataset is not dirty."""
        ds = _sample_dataset()
        assert not ds.is_dirty

    def test_infer_storage_types(self) -> None:
        """Storage types are inferred correctly from the DataFrame."""
        ds = _sample_dataset()
        assert ds.variables["age"].storage_type == StorageType.INTEGER
        assert ds.variables["gender"].storage_type == StorageType.STRING
        assert ds.variables["bp"].storage_type == StorageType.FLOAT

    def test_infer_measure_types(self) -> None:
        """Measure types are inferred correctly from the DataFrame."""
        ds = _sample_dataset()
        assert ds.variables["age"].measure == MeasureType.SCALE
        assert ds.variables["gender"].measure == MeasureType.BINARY
        assert ds.variables["bp"].measure == MeasureType.SCALE

    def test_repr(self) -> None:
        """__repr__ includes name and shape."""
        ds = _sample_dataset()
        r = repr(ds)
        assert "Dataset" in r
        assert "clinical" in r
        assert "5" in r
        assert "3" in r


# ---------------------------------------------------------------------------
# Variable rename
# ---------------------------------------------------------------------------

class TestRenameVariable:
    """Tests for rename_variable."""

    def test_rename_updates_dataframe(self) -> None:
        """Renaming a variable updates the DataFrame column."""
        ds = _sample_dataset()
        ds.rename_variable("age", "patient_age")
        assert "patient_age" in ds.data.columns
        assert "age" not in ds.data.columns

    def test_rename_updates_metadata_keys(self) -> None:
        """Renaming a variable updates the variables dict keys."""
        ds = _sample_dataset()
        ds.rename_variable("age", "patient_age")
        assert "patient_age" in ds.variables
        assert "age" not in ds.variables

    def test_rename_updates_meta_name(self) -> None:
        """Renaming a variable updates the VariableMeta.name field."""
        ds = _sample_dataset()
        ds.rename_variable("age", "patient_age")
        assert ds.variables["patient_age"].name == "patient_age"

    def test_rename_whitespace_to_underscore(self) -> None:
        """Whitespace in the new name is replaced with underscores."""
        ds = _sample_dataset()
        ds.rename_variable("age", "patient age")
        assert "patient_age" in ds.data.columns

    def test_rename_same_name_noop(self) -> None:
        """Renaming to the same name is a no-op."""
        ds = _sample_dataset()
        ds.rename_variable("age", "age")
        assert "age" in ds.data.columns
        assert ds.n_cols == 3

    def test_rename_missing_old_raises(self) -> None:
        """Renaming a non-existent variable raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.rename_variable("missing", "new_name")

    def test_rename_duplicate_new_raises(self) -> None:
        """Renaming to an existing name raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.rename_variable("age", "gender")

    def test_rename_marks_dirty(self) -> None:
        """Renaming a variable marks the dataset as dirty."""
        ds = _sample_dataset()
        assert not ds.is_dirty
        ds.rename_variable("age", "patient_age")
        assert ds.is_dirty


# ---------------------------------------------------------------------------
# Data property
# ---------------------------------------------------------------------------

class TestDataProperty:
    """Tests for the data property and its setter."""

    def test_get_data(self) -> None:
        """data returns the underlying DataFrame."""
        ds = _sample_dataset()
        df = ds.data
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 3)

    def test_set_data_with_new_column(self) -> None:
        """Assigning a new DataFrame with extra columns creates metadata."""
        ds = _sample_dataset()
        new_df = _sample_df().copy()
        new_df["new_col"] = [1, 2, 3, 4, 5]
        ds.data = new_df
        assert "new_col" in ds.variables
        assert ds.n_cols == 4

    def test_set_data_preserves_existing_meta(self) -> None:
        """Existing variable metadata is preserved when columns match."""
        ds = _sample_dataset()
        ds.update_variable_meta("age", label="Patient Age", role=Role.INPUT)
        new_df = _sample_df().copy()
        new_df["age"] = new_df["age"] + 1
        ds.data = new_df
        assert ds.variables["age"].label == "Patient Age"
        assert ds.variables["age"].role == Role.INPUT

    def test_set_data_marks_dirty(self) -> None:
        """Setting data marks the dataset as dirty."""
        ds = _sample_dataset()
        assert not ds.is_dirty
        ds.data = _sample_df()
        assert ds.is_dirty

    def test_set_data_non_dataframe_raises(self) -> None:
        """Setting data to a non-DataFrame raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.data = "not a dataframe"

    def test_set_data_same_columns_skips_meta_resync_loop(self) -> None:
        """컬럼 집합이 그대로면 변수 meta 재동기화 루프(O(cols) 순회) 자체를 건너뛴다(P1-4)."""
        ds = _sample_dataset()

        class CountingDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.contains_calls = 0

            def __contains__(self, key) -> bool:
                self.contains_calls += 1
                return super().__contains__(key)

        counting_vars = CountingDict(ds._variables)
        ds._variables = counting_vars

        new_df = _sample_df().copy()
        new_df["age"] = new_df["age"] + 1   # 값만 변경, 컬럼 동일
        ds.data = new_df

        assert counting_vars.contains_calls == 0
        assert ds.n_cols == 3
        assert ds.is_dirty

    def test_set_data_different_columns_still_resyncs(self) -> None:
        """컬럼 집합이 다르면 기존처럼 추가/삭제 diff를 반영한다(P1-4 회귀 방지)."""
        ds = _sample_dataset()
        new_df = _sample_df().copy()
        del new_df["age"]
        new_df["new_col"] = [1, 2, 3, 4, 5]
        ds.data = new_df
        assert "age" not in ds.variables
        assert "new_col" in ds.variables
        assert ds.n_cols == 3


# ---------------------------------------------------------------------------
# Add / remove variables
# ---------------------------------------------------------------------------

class TestAddRemoveVariable:
    """Tests for add_variable and remove_variable."""

    def test_add_variable(self) -> None:
        """Adding a variable creates a new column and metadata."""
        ds = _sample_dataset()
        ds.add_variable("height", data=pd.Series([170, 165, 180, 175, 160]))
        assert "height" in ds.data.columns
        assert "height" in ds.variables
        assert ds.n_cols == 4

    def test_add_variable_without_data(self) -> None:
        """Adding a variable without data creates a NaN column."""
        ds = _sample_dataset()
        ds.add_variable("notes")
        assert "notes" in ds.data.columns
        assert ds.data["notes"].isna().all()

    def test_add_duplicate_raises(self) -> None:
        """Adding a variable with an existing name raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.add_variable("age", data=pd.Series([1, 2, 3, 4, 5]))

    def test_add_variable_marks_dirty(self) -> None:
        """Adding a variable marks the dataset as dirty."""
        ds = _sample_dataset()
        ds.add_variable("height", data=pd.Series([170, 165, 180, 175, 160]))
        assert ds.is_dirty

    def test_remove_variable(self) -> None:
        """Removing a variable deletes the column and metadata."""
        ds = _sample_dataset()
        ds.remove_variable("age")
        assert "age" not in ds.data.columns
        assert "age" not in ds.variables
        assert ds.n_cols == 2

    def test_remove_missing_raises(self) -> None:
        """Removing a non-existent variable raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.remove_variable("missing")

    def test_remove_variable_marks_dirty(self) -> None:
        """Removing a variable marks the dataset as dirty."""
        ds = _sample_dataset()
        ds.remove_variable("age")
        assert ds.is_dirty


# ---------------------------------------------------------------------------
# Get / update variable metadata
# ---------------------------------------------------------------------------

class TestVariableMetaAccess:
    """Tests for get_variable and update_variable_meta."""

    def test_get_variable(self) -> None:
        """get_variable returns the correct VariableMeta."""
        ds = _sample_dataset()
        meta = ds.get_variable("age")
        assert isinstance(meta, VariableMeta)
        assert meta.name == "age"

    def test_get_missing_variable_raises(self) -> None:
        """get_variable for a non-existent name raises DatasetError."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.get_variable("missing")

    def test_update_variable_meta(self) -> None:
        """update_variable_meta changes fields on the metadata."""
        ds = _sample_dataset()
        ds.update_variable_meta("age", label="Age in years", unit="years")
        assert ds.variables["age"].label == "Age in years"
        assert ds.variables["age"].unit == "years"

    def test_update_variable_meta_marks_dirty(self) -> None:
        """update_variable_meta marks the dataset as dirty."""
        ds = _sample_dataset()
        ds.update_variable_meta("age", label="Age in years")
        assert ds.is_dirty

    def test_update_missing_variable_raises(self) -> None:
        """update_variable_meta for a non-existent variable raises."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.update_variable_meta("missing", label="X")

    def test_update_invalid_field_raises(self) -> None:
        """update_variable_meta with an invalid field name raises."""
        ds = _sample_dataset()
        with pytest.raises(DatasetError):
            ds.update_variable_meta("age", nonexistent_field="value")


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for to_dict / from_dict round-trips."""

    def test_round_trip(self) -> None:
        """Dataset survives a to_dict -> from_dict cycle."""
        original = _sample_dataset()
        d = original.to_dict()
        restored = Dataset.from_dict(d)

        assert restored.name == original.name
        assert restored.n_rows == original.n_rows
        assert restored.n_cols == original.n_cols
        assert set(restored.data.columns) == set(original.data.columns)

    def test_round_trip_preserves_metadata(self) -> None:
        """Variable metadata survives a round-trip."""
        original = _sample_dataset()
        original.update_variable_meta("age", label="Patient Age")
        d = original.to_dict()
        restored = Dataset.from_dict(d)
        assert restored.variables["age"].label == "Patient Age"

    def test_serialization_data_is_dict(self) -> None:
        """to_dict["data"] is a dictionary of lists."""
        ds = _sample_dataset()
        d = ds.to_dict()
        assert isinstance(d["data"], dict)
        assert "age" in d["data"]
        assert len(d["data"]["age"]) == 5

    def test_serialization_includes_variables(self) -> None:
        """to_dict["variables"] is a nested dict."""
        ds = _sample_dataset()
        d = ds.to_dict()
        assert isinstance(d["variables"], dict)
        assert "age" in d["variables"]
        assert d["variables"]["age"]["name"] == "age"

    def test_from_dict_sets_timestamps(self) -> None:
        """from_dict restores created_at and updated_at."""
        ds = _sample_dataset()
        d = ds.to_dict()
        restored = Dataset.from_dict(d)
        assert isinstance(restored.created_at, type(ds.created_at))
        assert isinstance(restored.updated_at, type(ds.updated_at))

    def test_from_dict_clears_dirty(self) -> None:
        """from_dict produces a non-dirty dataset."""
        ds = _sample_dataset()
        ds.rename_variable("age", "patient_age")  # make dirty
        d = ds.to_dict()
        restored = Dataset.from_dict(d)
        assert not restored.is_dirty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and error-handling tests."""

    def test_empty_dataframe(self) -> None:
        """Construction with an empty DataFrame is allowed."""
        df = pd.DataFrame()
        ds = Dataset(data=df)
        assert ds.n_rows == 0
        assert ds.n_cols == 0
        assert ds.variables == {}

    def test_single_column(self) -> None:
        """Construction with a single-column DataFrame works."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = Dataset(data=df)
        assert ds.n_cols == 1
        assert "x" in ds.variables

    def test_nan_values_preserved(self) -> None:
        """NaN values in the DataFrame are preserved through round-trip."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        ds = Dataset(data=df)
        d = ds.to_dict()
        restored = Dataset.from_dict(d)
        assert restored.data["x"].isna().sum() == 1

    def test_boolean_column(self) -> None:
        """Boolean columns are inferred correctly."""
        df = pd.DataFrame({"flag": [True, False, True]})
        ds = Dataset(data=df)
        assert ds.variables["flag"].storage_type == StorageType.BOOLEAN
        assert ds.variables["flag"].measure == MeasureType.BINARY

    def test_datetime_column(self) -> None:
        """Datetime columns are inferred correctly."""
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])})
        ds = Dataset(data=df)
        assert ds.variables["dt"].storage_type == StorageType.DATETIME
        assert ds.variables["dt"].measure == MeasureType.DATE_TIME
