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

    def test_multiple_values_scale_measure(self):
        """Multiple integer values -> SCALE (SPSS 호환, 고유값 수 무관)."""
        model = SPSSGridModel()
        model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "2", Qt.ItemDataRole.EditRole)
        model.setData(model.index(2, 0), "3", Qt.ItemDataRole.EditRole)

        variables = model.get_variables()
        var = variables["VAR00001"]
        assert var.measure == MeasureType.SCALE

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


class TestBatchUpdate:
    """대량 편집 배치 업데이트(붙여넣기·채우기 최적화) 검증."""

    def _count_signals(self, model):
        counter = {"n": 0}
        model.data_changed.connect(lambda: counter.__setitem__("n", counter["n"] + 1))
        return counter

    def test_batch_emits_data_changed_once(self):
        """배치 블록 내 다수 setData → data_changed는 1회만 방출."""
        model = SPSSGridModel()
        counter = self._count_signals(model)
        with model.batch_update():
            for r in range(10):
                for c in range(4):
                    model.setData(model.index(r, c), str(r * 4 + c), Qt.ItemDataRole.EditRole)
        assert counter["n"] == 1, f"data_changed가 {counter['n']}회 방출됨 (1회 기대)"

    def test_batch_result_matches_per_cell(self):
        """배치 입력 결과가 셀별 입력 결과와 동일."""
        block = [[str(r * 3 + c) for c in range(3)] for r in range(8)]

        m_cell = SPSSGridModel()
        for r, row in enumerate(block):
            for c, v in enumerate(row):
                m_cell.setData(m_cell.index(r, c), v, Qt.ItemDataRole.EditRole)

        m_batch = SPSSGridModel()
        with m_batch.batch_update():
            for r, row in enumerate(block):
                for c, v in enumerate(row):
                    m_batch.setData(m_batch.index(r, c), v, Qt.ItemDataRole.EditRole)

        df_cell = m_cell.get_dataframe()
        df_batch = m_batch.get_dataframe()
        assert df_cell.shape == df_batch.shape
        assert (df_cell.values == df_batch.values).all()
        assert list(df_cell.dtypes) == list(df_batch.dtypes)

    def test_batch_dirty_region_covers_all_cells(self):
        """비순차 입력 순서에서도 더티 영역이 전체를 포함."""
        model = SPSSGridModel()
        received = {}

        def on_changed(top_left, bottom_right, roles):
            received["tl"] = (top_left.row(), top_left.column())
            received["br"] = (bottom_right.row(), bottom_right.column())

        model.dataChanged.connect(on_changed)
        with model.batch_update():
            # 일부러 역순·교차로 입력
            model.setData(model.index(5, 3), "1", Qt.ItemDataRole.EditRole)
            model.setData(model.index(2, 1), "2", Qt.ItemDataRole.EditRole)
            model.setData(model.index(7, 0), "3", Qt.ItemDataRole.EditRole)
        assert received["tl"] == (2, 0), f"top_left={received.get('tl')}"
        assert received["br"] == (7, 3), f"bottom_right={received.get('br')}"

    def test_nested_batch_flushes_once_at_outer_exit(self):
        """중첩 배치는 가장 바깥 블록 종료 시에만 1회 방출."""
        model = SPSSGridModel()
        counter = self._count_signals(model)
        with model.batch_update():
            model.setData(model.index(0, 0), "1", Qt.ItemDataRole.EditRole)
            with model.batch_update():
                model.setData(model.index(1, 0), "2", Qt.ItemDataRole.EditRole)
            # 내부 블록 종료로는 방출되지 않아야 함
            assert counter["n"] == 0
        assert counter["n"] == 1

    def test_empty_batch_no_emit(self):
        """변경 없는 빈 배치 블록은 신호를 방출하지 않음."""
        model = SPSSGridModel()
        counter = self._count_signals(model)
        with model.batch_update():
            pass
        assert counter["n"] == 0

    def test_batch_rejects_invalid_numeric_like_per_cell(self):
        """배치 모드에서도 숫자형 변수에 문자 입력은 거부."""
        model = SPSSGridModel()
        # 먼저 숫자형으로 확정
        model.setData(model.index(0, 0), "10", Qt.ItemDataRole.EditRole)
        with model.batch_update():
            ok = model.setData(model.index(1, 0), "abc", Qt.ItemDataRole.EditRole)
        assert ok is False
        df = model.get_dataframe()
        # 거부되어 1행만 존재
        assert df.iloc[0, 0] == 10
