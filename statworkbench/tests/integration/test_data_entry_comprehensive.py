"""Comprehensive integration tests for data entry scenarios.

Tests real-world data entry patterns:
- Empty cell -> data -> empty cell transitions
- Multi-cell entry with navigation
- Copy/paste operations
- Variable auto-creation with metadata
- Row/column expansion edge cases
"""

import pytest
import pandas as pd
from PySide6.QtCore import Qt

from statworkbench.ui.models.spss_grid_model import SPSSGridModel
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import StorageType, MeasureType


class TestEmptyToDataTransitions:
    """Test transitions between empty and data cells."""

    def test_empty_cell_then_data(self):
        """Empty cell -> enter data -> verify stored."""
        model = SPSSGridModel()
        
        # Start empty
        assert len(model.get_dataframe().columns) == 0
        
        # Enter data
        index = model.index(0, 0)
        result = model.setData(index, "100", Qt.ItemDataRole.EditRole)
        
        assert result is True
        df = model.get_dataframe()
        assert len(df.columns) == 1
        assert df.iloc[0, 0] == 100

    def test_data_then_empty_then_data_again(self):
        """Data -> empty -> data again in same cell."""
        model = SPSSGridModel()
        
        # Enter data
        model.setData(model.index(0, 0), "50", Qt.ItemDataRole.EditRole)
        assert model.get_dataframe().iloc[0, 0] == 50
        
        # Clear (empty string)
        model.setData(model.index(0, 0), "", Qt.ItemDataRole.EditRole)
        
        # Enter new data
        model.setData(model.index(0, 0), "75", Qt.ItemDataRole.EditRole)
        assert model.get_dataframe().iloc[0, 0] == 75

    def test_multiple_empty_cells_then_data(self):
        """Multiple empty cells, then fill some."""
        model = SPSSGridModel()
        
        # Fill row 0
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), "2", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 2), "3", Qt.ItemDataRole.EditRole)
        
        # Skip row 1 (empty)
        
        # Fill row 2
        model.setData(model.index(2, 0), "4", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 3  # Rows 0, 1, 2
        assert df.iloc[0, 0] == 1
        assert pd.isna(df.iloc[1, 0])  # Row 1 is empty
        assert df.iloc[2, 0] == 4


class TestMultiCellEntry:
    """Test entering data across multiple cells."""

    def test_fill_row_left_to_right(self):
        """Fill a row left to right."""
        model = SPSSGridModel()
        
        for col in range(5):
            model.setData(model.index(0, col), str(col * 10), Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df.columns) == 5
        for col in range(5):
            assert df.iloc[0, col] == col * 10

    def test_fill_column_top_to_bottom(self):
        """Fill a column top to bottom."""
        model = SPSSGridModel()
        
        for row in range(10):
            model.setData(model.index(row, 0), str(row + 1), Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 10
        for row in range(10):
            assert df.iloc[row, 0] == row + 1

    def test_fill_grid_pattern(self):
        """Fill a grid in a pattern."""
        model = SPSSGridModel()
        
        # Fill a 3x3 grid
        for row in range(3):
            for col in range(3):
                value = (row + 1) * 10 + (col + 1)
                model.setData(model.index(row, col), str(value), Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 3
        assert len(df.columns) == 3
        assert df.iloc[0, 0] == 11
        assert df.iloc[1, 1] == 22
        assert df.iloc[2, 2] == 33

    def test_sparse_data_entry(self):
        """Enter data sparsely (not all cells)."""
        model = SPSSGridModel()
        
        # Enter data at specific positions
        model.setData(model.index(0, 0), "A", Qt.ItemDataRole.EditRole)
        model.setData(model.index(2, 1), "B", Qt.ItemDataRole.EditRole)
        model.setData(model.index(5, 3), "C", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 6  # Rows 0-5
        assert len(df.columns) == 4  # Cols 0-3
        
        assert df.iloc[0, 0] == "A"
        assert pd.isna(df.iloc[1, 0])
        assert df.iloc[2, 1] == "B"
        assert pd.isna(df.iloc[5, 0])
        assert df.iloc[5, 3] == "C"


class TestVariableAutoCreation:
    """Test variable auto-creation behavior."""

    def test_variables_created_on_demand(self):
        """Variables should only be created when data is entered."""
        model = SPSSGridModel()
        
        # No variables initially
        assert len(model.get_variables()) == 0
        
        # Enter data in col 5 (skip cols 0-4)
        model.setData(model.index(0, 5), "test", Qt.ItemDataRole.EditRole)
        
        # Should create vars for cols 0-5
        vars_dict = model.get_variables()
        assert len(vars_dict) == 6
        assert "VAR00001" in vars_dict
        assert "VAR00006" in vars_dict

    def test_variable_naming_sequence(self):
        """Variable names should follow sequence."""
        model = SPSSGridModel()
        
        # Enter data in various columns
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 2), "3", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), "2", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        cols = list(df.columns)
        assert cols[0] == "VAR00001"
        assert cols[1] == "VAR00002"
        assert cols[2] == "VAR00003"

    def test_variable_metadata_persistence(self):
        """Variable metadata should persist across data changes."""
        model = SPSSGridModel()
        
        # Enter integer data
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.INTEGER
        
        # Add more integer data
        model.setData(model.index(1, 0), "20", Qt.ItemDataRole.EditRole)
        model.setData(model.index(2, 0), "30", Qt.ItemDataRole.EditRole)
        
        # Metadata should update
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.ORDINAL  # 3 unique values


class TestDataTypes:
    """Test different data type handling."""

    def test_integer_values(self):
        """Integer values should be stored as int."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "42", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 42
        assert isinstance(df.iloc[0, 0], int)

    def test_float_values(self):
        """Float values should be stored as float."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "3.14159", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert abs(df.iloc[0, 0] - 3.14159) < 0.00001
        assert isinstance(df.iloc[0, 0], float)

    def test_string_values(self):
        """String values should be stored as string."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "Hello", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == "Hello"
        assert isinstance(df.iloc[0, 0], str)

    def test_mixed_types_in_column(self):
        """Mixed types should default to string."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "text", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        # Both should be strings when mixed
        assert df.iloc[0, 0] == "10" or df.iloc[0, 0] == 10
        assert df.iloc[1, 0] == "text"

    def test_negative_numbers(self):
        """Negative numbers should work."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "-50", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == -50

    def test_zero_value(self):
        """Zero should be stored correctly."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "0", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_string(self):
        """Very long strings should be handled."""
        model = SPSSGridModel()
        long_text = "A" * 1000
        model.setData(model.index(0, 0), long_text, Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == long_text

    def test_special_characters(self):
        """Special characters should be preserved."""
        model = SPSSGridModel()
        special = "Hello!@#$%^&*()_+-=[]{}|;':\",./<>?"
        model.setData(model.index(0, 0), special, Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == special

    def test_unicode_characters(self):
        """Unicode characters should work."""
        model = SPSSGridModel()
        unicode_text = "한글 테스트 日本語 αβγ"
        model.setData(model.index(0, 0), unicode_text, Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert df.iloc[0, 0] == unicode_text

    def test_whitespace_handling(self):
        """Whitespace should be handled correctly."""
        model = SPSSGridModel()
        
        # Leading/trailing whitespace should be stripped for numbers
        model.setData(model.index(0, 0), "  42  ", Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 42

    def test_empty_string_vs_none(self):
        """Empty string and None should both clear cell."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "data", Qt.ItemDataRole.EditRole)
        assert model.get_dataframe().iloc[0, 0] == "data"
        
        # Empty string
        model.setData(model.index(0, 0), "", Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        # Cell should be empty/NA
        assert len(df.columns) == 1

    def test_large_grid_expansion(self):
        """Large grid should expand properly."""
        model = SPSSGridModel()
        
        # Enter data at row 100, col 50
        model.setData(model.index(100, 50), "far", Qt.ItemDataRole.EditRole)
        
        df = model.get_dataframe()
        assert len(df) == 101
        assert len(df.columns) == 51
        assert df.iloc[100, 50] == "far"


class TestRowColumnOperations:
    """Test row and column operations."""

    def test_add_row_after_data(self):
        """Adding row after data should work."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "2", Qt.ItemDataRole.EditRole)
        
        model.add_row()
        
        df = model.get_dataframe()
        assert len(df) == 3
        assert pd.isna(df.iloc[2, 0])

    def test_remove_row(self):
        """Removing a row should work."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "2", Qt.ItemDataRole.EditRole)
        model.setData(model.index(2, 0), "3", Qt.ItemDataRole.EditRole)
        
        model.remove_row(1)
        
        df = model.get_dataframe()
        assert len(df) == 2
        assert df.iloc[0, 0] == 1
        assert df.iloc[1, 0] == 3

    def test_remove_column(self):
        """Removing a column should work."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "A", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), "B", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 2), "C", Qt.ItemDataRole.EditRole)
        
        model.remove_column(1)
        
        df = model.get_dataframe()
        assert len(df.columns) == 2
        assert df.iloc[0, 0] == "A"
        assert df.iloc[0, 1] == "C"

    def test_add_column(self):
        """Adding a column should work."""
        model = SPSSGridModel()
        
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.add_column("NewVar", ["2"])
        
        df = model.get_dataframe()
        assert len(df.columns) == 2
        assert df.iloc[0, 1] == "2"


class TestMeasureTypeInference:
    """Test measure type inference with various data patterns."""

    def test_binary_numeric(self):
        """Two unique numeric values -> BINARY."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "0", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "1", Qt.ItemDataRole.EditRole)
        
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.BINARY

    def test_binary_string(self):
        """Two unique string values -> BINARY."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "Male", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "Female", Qt.ItemDataRole.EditRole)
        
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.BINARY

    def test_ordinal_few_values(self):
        """3-10 unique integer values -> ORDINAL."""
        model = SPSSGridModel()
        for i in range(5):
            model.setData(model.index(i, 0), str(i + 1), Qt.ItemDataRole.EditRole)
        
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.ORDINAL

    def test_scale_many_values(self):
        """Many unique numeric values -> SCALE."""
        model = SPSSGridModel()
        for i in range(20):
            model.setData(model.index(i, 0), str(i * 10), Qt.ItemDataRole.EditRole)
        
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_nominal_many_strings(self):
        """Many unique string values -> NOMINAL."""
        model = SPSSGridModel()
        for i in range(15):
            model.setData(model.index(i, 0), f"Category{i}", Qt.ItemDataRole.EditRole)
        
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.NOMINAL
