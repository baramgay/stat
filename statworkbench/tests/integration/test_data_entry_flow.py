"""Integration test: Data entry flow (SPSS-style).

Tests the complete flow from empty grid to data entry to variable creation.
"""

import pytest
import pandas as pd
from PySide6.QtCore import Qt

from statworkbench.ui.models.spss_grid_model import SPSSGridModel
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import StorageType, MeasureType


class TestDataEntryFlow:
    """Test SPSS-style data entry flow."""

    def test_empty_grid_has_no_variables(self):
        """Empty grid should have no variables."""
        model = SPSSGridModel()
        assert len(model.get_dataframe().columns) == 0
        assert len(model.get_variables()) == 0

    def test_first_data_entry_creates_var00001(self):
        """First data entry should create VAR00001."""
        model = SPSSGridModel()
        
        # Simulate entering "10" at row 0, col 0
        index = model.index(0, 0)
        result = model.setData(index, "10", Qt.ItemDataRole.EditRole)
        
        assert result is True
        df = model.get_dataframe()
        assert len(df.columns) == 1
        assert df.columns[0] == "VAR00001"
        assert df.iloc[0, 0] == 10

    def test_second_column_creates_var00002(self):
        """Second column should create VAR00002."""
        model = SPSSGridModel()
        
        # Enter data in col 0 and col 1
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), "20", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df.columns) == 2
        assert df.columns[0] == "VAR00001"
        assert df.columns[1] == "VAR00002"

    def test_variable_metadata_auto_created(self):
        """Variable metadata should be auto-created on data entry."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        
        variables = model.get_variables()
        assert "VAR00001" in variables
        
        var = variables["VAR00001"]
        assert var.name == "VAR00001"
        assert var.storage_type == StorageType.INTEGER
        # Single value defaults to SCALE (can be changed by user later)
        assert var.measure == MeasureType.SCALE

    def test_float_data_creates_scale_measure(self):
        """Float data should create SCALE measure."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "3.14", Qt.ItemDataRole.EditRole)
        
        variables = model.get_variables()
        var = variables["VAR00001"]
        assert var.storage_type == StorageType.FLOAT
        assert var.measure == MeasureType.SCALE

    def test_string_data_creates_nominal_measure(self):
        """String data should create NOMINAL measure."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "Male", Qt.ItemDataRole.EditRole)
        
        variables = model.get_variables()
        var = variables["VAR00001"]
        assert var.storage_type == StorageType.STRING
        assert var.measure == MeasureType.NOMINAL

    def test_multiple_values_ordinal_measure(self):
        """Multiple integer values with few unique should create ORDINAL."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "2", Qt.ItemDataRole.EditRole)
        model.setData(model.index(2, 0), "3", Qt.ItemDataRole.EditRole)
        
        variables = model.get_variables()
        var = variables["VAR00001"]
        assert var.measure == MeasureType.ORDINAL

    def test_dataframe_with_existing_variables(self):
        """Model should accept existing variables dict."""
        df = pd.DataFrame({"Age": [25, 30, 35], "Gender": ["M", "F", "M"]})
        vars_dict = {
            "Age": VariableMeta(name="Age", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
            "Gender": VariableMeta(name="Gender", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL),
        }
        
        model = SPSSGridModel(df, vars_dict)
        
        assert len(model.get_dataframe().columns) == 2
        variables = model.get_variables()
        assert "Age" in variables
        assert variables["Age"].measure == MeasureType.SCALE
        assert variables["Gender"].measure == MeasureType.NOMINAL

    def test_header_rename_updates_metadata(self):
        """Renaming header should update variable metadata."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        
        # Rename VAR00001 to Age
        result = model.setHeaderData(0, Qt.Orientation.Horizontal, "Age", Qt.ItemDataRole.EditRole)
        assert result is True
        
        variables = model.get_variables()
        assert "Age" in variables
        assert "VAR00001" not in variables
        assert variables["Age"].name == "Age"

    def test_multiple_rows_data_entry(self):
        """Entering data in multiple rows should work."""
        model = SPSSGridModel()
        
        # Enter data in a column
        for row in range(5):
            model.setData(model.index(row, 0), str(row * 10), Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 5
        assert df.iloc[4, 0] == 40

    def test_enter_key_navigation_simulation(self):
        """Simulate Enter key moving to next row."""
        model = SPSSGridModel()
        
        # Enter data at row 0
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        
        # Simulate moving to row 1 (as Enter key would do)
        model.setData(model.index(1, 0), "20", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 2
        assert df.iloc[0, 0] == 10
        assert df.iloc[1, 0] == 20

    def test_empty_string_clears_cell(self):
        """Entering empty string should clear cell but keep column."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        
        # Verify data exists
        df = model.get_dataframe()
        assert len(df.columns) == 1
        assert df.iloc[0, 0] == 10
        
        # Clear the cell
        model.setData(model.index(0, 0), "", Qt.ItemDataRole.EditRole)
        
        # Column should still exist
        df = model.get_dataframe()
        assert len(df.columns) == 1
        # But cell should be empty/NA
        assert len(df) == 0 or pd.isna(df.iloc[0, 0])
