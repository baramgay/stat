"""Tests for UI models (DataFrameTableModel, VariableTableModel)."""

from __future__ import annotations

import pytest
import pandas as pd

from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, Role


# Skip PySide6-dependent tests if not available
pytest.importorskip("PySide6", reason="PySide6 not installed")

from PySide6.QtCore import Qt, QModelIndex  # type: ignore[import-untyped]
from statworkbench.ui.models.dataframe_table_model import DataFrameTableModel
from statworkbench.ui.models.variable_table_model import VariableTableModel


class TestDataFrameTableModel:
    """Tests for DataFrameTableModel."""

    def test_empty_dataframe(self) -> None:
        """Test model with empty DataFrame."""
        df = pd.DataFrame()
        model = DataFrameTableModel(df)
        assert model.rowCount() == 0
        assert model.columnCount() == 0

    def test_row_count(self) -> None:
        """Test row count matches DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        model = DataFrameTableModel(df)
        assert model.rowCount() == 3

    def test_column_count(self) -> None:
        """Test column count matches DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        model = DataFrameTableModel(df)
        assert model.columnCount() == 3

    def test_data_display(self) -> None:
        """Test data() returns correct display values."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        model = DataFrameTableModel(df)

        index = model.index(0, 0)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == "1"

        index = model.index(1, 1)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == "y"

    def test_header_data(self) -> None:
        """Test horizontal header returns column names."""
        df = pd.DataFrame({"age": [25, 30], "sex": ["M", "F"]})
        model = DataFrameTableModel(df)

        header = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        assert header == "age"

        header = model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        assert header == "sex"

    def test_vertical_header(self) -> None:
        """Test vertical header returns 1-based row numbers."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        model = DataFrameTableModel(df)

        header = model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole)
        assert header == "1"

        header = model.headerData(2, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole)
        assert header == "3"

    def test_set_dataframe(self) -> None:
        """Test replacing the DataFrame."""
        df1 = pd.DataFrame({"a": [1, 2]})
        model = DataFrameTableModel(df1)
        assert model.rowCount() == 2

        df2 = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        model.set_dataframe(df2)
        assert model.rowCount() == 3
        assert model.columnCount() == 2

    def test_get_dataframe(self) -> None:
        """Test get_dataframe returns the DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        model = DataFrameTableModel(df)
        result = model.get_dataframe()
        assert result is df
        assert len(result) == 2

    def test_data_with_nan(self) -> None:
        """Test NaN values display as empty string."""
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        model = DataFrameTableModel(df)

        index = model.index(1, 0)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == ""

    def test_add_row(self) -> None:
        """Test adding a row."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        model = DataFrameTableModel(df)
        model.add_row({"a": 10, "b": 20})
        assert model.rowCount() == 3

    def test_remove_row(self) -> None:
        """Test removing a row."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        model = DataFrameTableModel(df)
        assert model.remove_row(1)
        assert model.rowCount() == 2

    def test_flags(self) -> None:
        """Test item flags."""
        df = pd.DataFrame({"a": [1]})
        model = DataFrameTableModel(df)
        index = model.index(0, 0)
        flags = model.flags(index)
        assert flags & Qt.ItemFlag.ItemIsEnabled
        assert flags & Qt.ItemFlag.ItemIsSelectable
        assert flags & Qt.ItemFlag.ItemIsEditable


class TestVariableTableModel:
    """Tests for VariableTableModel."""

    @pytest.fixture
    def sample_vars(self) -> list[VariableMeta]:
        """Create sample variables for testing."""
        return [
            VariableMeta(
                name="age",
                label="Age in years",
                storage_type=StorageType.INTEGER,
                measure=MeasureType.SCALE,
                role=Role.INPUT,
                width=8,
                decimals=0,
                unit="years",
            ),
            VariableMeta(
                name="sex",
                label="Sex",
                storage_type=StorageType.STRING,
                measure=MeasureType.BINARY,
                role=Role.NONE,
                width=4,
                decimals=0,
                value_labels={0: "Male", 1: "Female"},
            ),
            VariableMeta(
                name="treatment",
                label="Treatment group",
                storage_type=StorageType.STRING,
                measure=MeasureType.NOMINAL,
                role=Role.NONE,
                missing_values=[99, -99],
            ),
        ]

    def test_row_count(self, sample_vars: list[VariableMeta]) -> None:
        """Test row count matches variable list."""
        model = VariableTableModel(sample_vars)
        assert model.rowCount() == 3

    def test_column_count(self, sample_vars: list[VariableMeta]) -> None:
        """Test column count matches COLUMNS."""
        model = VariableTableModel(sample_vars)
        assert model.columnCount() == 11

    def test_data_name(self, sample_vars: list[VariableMeta]) -> None:
        """Test Name column display."""
        model = VariableTableModel(sample_vars)
        index = model.index(0, VariableTableModel.COL_NAME)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == "age"

    def test_data_label(self, sample_vars: list[VariableMeta]) -> None:
        """Test Label column display."""
        model = VariableTableModel(sample_vars)
        index = model.index(0, VariableTableModel.COL_LABEL)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == "Age in years"

    def test_data_measure(self, sample_vars: list[VariableMeta]) -> None:
        """Test Measure column display."""
        model = VariableTableModel(sample_vars)
        index = model.index(1, VariableTableModel.COL_MEASURE)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value == "binary"

    def test_data_values_with_labels(self, sample_vars: list[VariableMeta]) -> None:
        """Test Values column shows label count."""
        model = VariableTableModel(sample_vars)
        index = model.index(1, VariableTableModel.COL_VALUES)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert "2 items" in value

    def test_data_missing_with_rules(self, sample_vars: list[VariableMeta]) -> None:
        """Test Missing column shows rule count."""
        model = VariableTableModel(sample_vars)
        index = model.index(2, VariableTableModel.COL_MISSING)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert "2" in value and "rule" in value.lower()

    def test_header_data(self, sample_vars: list[VariableMeta]) -> None:
        """Test horizontal headers."""
        model = VariableTableModel(sample_vars)
        header = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        assert header == "Name"

    def test_add_variable(self, sample_vars: list[VariableMeta]) -> None:
        """Test adding a variable."""
        model = VariableTableModel(sample_vars)
        new_var = VariableMeta(name="bmi", label="Body Mass Index")
        model.add_variable(new_var)
        assert model.rowCount() == 4
        assert model.get_variable(3).name == "bmi"

    def test_remove_variable(self, sample_vars: list[VariableMeta]) -> None:
        """Test removing a variable."""
        model = VariableTableModel(sample_vars)
        model.remove_variable(1)
        assert model.rowCount() == 2
        assert model.get_variable(0).name == "age"
        assert model.get_variable(1).name == "treatment"

    def test_get_variable(self, sample_vars: list[VariableMeta]) -> None:
        """Test getting a variable by row."""
        model = VariableTableModel(sample_vars)
        var = model.get_variable(0)
        assert var.name == "age"
        assert var.measure == MeasureType.SCALE

    def test_get_variable_names(self, sample_vars: list[VariableMeta]) -> None:
        """Test getting all variable names."""
        model = VariableTableModel(sample_vars)
        names = model.get_variable_names()
        assert names == ["age", "sex", "treatment"]

    def test_get_variables_by_measure(self, sample_vars: list[VariableMeta]) -> None:
        """Test filtering variables by measure."""
        model = VariableTableModel(sample_vars)
        scale_vars = model.get_variables_by_measure([MeasureType.SCALE])
        assert len(scale_vars) == 1
        assert scale_vars[0].name == "age"

    def test_update_variable(self, sample_vars: list[VariableMeta]) -> None:
        """Test updating a variable."""
        model = VariableTableModel(sample_vars)
        updated = VariableMeta(name="AGE", label="Age", measure=MeasureType.SCALE)
        model.update_variable(0, updated)
        assert model.get_variable(0).name == "AGE"

    def test_clear(self, sample_vars: list[VariableMeta]) -> None:
        """Test clearing all variables."""
        model = VariableTableModel(sample_vars)
        model.clear()
        assert model.rowCount() == 0

    def test_flags(self, sample_vars: list[VariableMeta]) -> None:
        """Test item flags."""
        model = VariableTableModel(sample_vars)
        index = model.index(0, 0)
        flags = model.flags(index)
        assert flags & Qt.ItemFlag.ItemIsEnabled
        assert flags & Qt.ItemFlag.ItemIsEditable
