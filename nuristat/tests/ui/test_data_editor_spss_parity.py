"""데이터 편집기 SPSS 편의성 동등성 검증.

단순 반복 입력 테스트가 아니라, SPSS 데이터 편집기의 실제 편의 기능
(실행 취소/다시 실행, 위치 삽입, 잘라내기, 케이스 이동)이 갖추어졌는지
실사용 시나리오로 검증한다.

담당 에이전트: ux-designer/frontend, tester-unit
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.models.spss_grid_model import SPSSGridModel


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _make_model() -> SPSSGridModel:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    variables = {
        "a": VariableMeta(name="a", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "b": VariableMeta(name="b", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
    }
    return SPSSGridModel(df, variables)


# ── 실행 취소 / 다시 실행 ─────────────────────────────────────────────────

class TestUndoRedo:
    def test_undo_single_cell_edit(self):
        m = _make_model()
        assert not m.can_undo()
        m.setData(m.index(0, 0), "99", Qt.ItemDataRole.EditRole)
        assert m._dataframe.iloc[0, 0] == 99
        assert m.can_undo()
        assert m.undo() is True
        assert m._dataframe.iloc[0, 0] == 1   # 원복

    def test_redo_after_undo(self):
        m = _make_model()
        m.setData(m.index(0, 0), "99", Qt.ItemDataRole.EditRole)
        m.undo()
        assert m.can_redo()
        assert m.redo() is True
        assert m._dataframe.iloc[0, 0] == 99

    def test_new_edit_clears_redo(self):
        m = _make_model()
        m.setData(m.index(0, 0), "99", Qt.ItemDataRole.EditRole)
        m.undo()
        assert m.can_redo()
        m.setData(m.index(0, 1), "77", Qt.ItemDataRole.EditRole)
        assert not m.can_redo()   # 새 편집이 redo 무효화

    def test_undo_empty_returns_false(self):
        m = _make_model()
        assert m.undo() is False

    def test_batch_paste_is_single_undo(self):
        """대량 붙여넣기는 1회 실행 취소로 전부 되돌려진다."""
        m = _make_model()
        with m.batch_update():
            for r in range(3):
                m.setData(m.index(r, 0), str(r + 100), Qt.ItemDataRole.EditRole)
        assert list(m._dataframe["a"]) == [100, 101, 102]
        assert len(m._undo_stack) == 1   # 셀 3개지만 스냅샷 1개
        m.undo()
        assert list(m._dataframe["a"]) == [1, 2, 3]

    def test_undo_remove_row(self):
        m = _make_model()
        m.remove_row(1)
        assert list(m._dataframe["a"]) == [1, 3]
        m.undo()
        assert list(m._dataframe["a"]) == [1, 2, 3]

    def test_undo_remove_column(self):
        m = _make_model()
        m.remove_column(1)
        assert "b" not in m._dataframe.columns
        m.undo()
        assert "b" in m._dataframe.columns

    def test_undo_sort(self):
        m = _make_model()
        m.sort_by_column(0, ascending=False)
        assert list(m._dataframe["a"]) == [3, 2, 1]
        m.undo()
        assert list(m._dataframe["a"]) == [1, 2, 3]

    def test_undo_rename(self):
        m = _make_model()
        m.setHeaderData(0, Qt.Orientation.Horizontal, "age")
        assert "age" in m._dataframe.columns
        m.undo()
        assert "a" in m._dataframe.columns and "age" not in m._dataframe.columns

    def test_undo_limit_capped(self):
        m = _make_model()
        m._undo_limit = 5
        for i in range(10):
            m.setData(m.index(0, 0), str(i), Qt.ItemDataRole.EditRole)
        assert len(m._undo_stack) == 5


# ── 위치 삽입 (SPSS 케이스/변수 삽입) ──────────────────────────────────────

class TestInsert:
    def test_insert_row_at_pushes_existing_down(self):
        m = _make_model()
        assert m.insert_row_at(1) is True
        # 1번 위치에 빈 행 → [1, NA, 2, 3]
        col = list(m._dataframe["a"])
        assert col[0] == 1
        assert pd.isna(col[1])
        assert col[2] == 2 and col[3] == 3

    def test_insert_row_undoable(self):
        m = _make_model()
        m.insert_row_at(0)
        assert len(m._dataframe) == 4
        m.undo()
        assert len(m._dataframe) == 3

    def test_insert_column_at_shifts_right(self):
        m = _make_model()
        name = m.insert_column_at(1)
        cols = list(m._dataframe.columns)
        assert cols[0] == "a"
        assert cols[1] == name      # 삽입된 새 변수
        assert cols[2] == "b"

    def test_insert_column_creates_metadata(self):
        m = _make_model()
        name = m.insert_column_at(0)
        assert name in m._variables
        assert m._variables[name].storage_type == StorageType.STRING

    def test_insert_column_undoable(self):
        m = _make_model()
        m.insert_column_at(0)
        assert len(m._dataframe.columns) == 3
        m.undo()
        assert len(m._dataframe.columns) == 2


# ── DataView 통합 (잘라내기·케이스 이동·삽입 위임) ─────────────────────────

class TestDataViewIntegration:
    def _make_view(self):
        from nuristat.ui.data_view import DataView
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        variables = {
            "a": VariableMeta(name="a", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
            "b": VariableMeta(name="b", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        }
        ds = Dataset(df, "t", variables)
        v = DataView()
        v.set_dataset(ds)
        return v

    def test_name_box_editable_for_goto(self):
        v = self._make_view()
        assert not v.name_box.isReadOnly()

    def test_name_box_goto_jumps_to_case(self):
        v = self._make_view()
        v.name_box.setText("3")
        v._name_box_goto()
        assert v.table.currentIndex().row() == 2   # 3번 케이스 = 인덱스 2

    def test_goto_clamps_out_of_range(self):
        v = self._make_view()
        v.name_box.setText("99999")
        v._name_box_goto()
        assert v.table.currentIndex().row() == v._model.rowCount() - 1

    def test_cut_copies_and_clears(self):
        v = self._make_view()
        idx = v._model.index(0, 0)
        v.table.setCurrentIndex(idx)
        v.table.selectionModel().select(idx, v.table.selectionModel().SelectionFlag.Select)
        v.cut_selection()
        assert QApplication.clipboard().text().strip() == "1"
        assert pd.isna(v._model._dataframe.iloc[0, 0])

    def test_view_undo_redo_delegates_to_model(self):
        v = self._make_view()
        v._model.setData(v._model.index(0, 0), "55", Qt.ItemDataRole.EditRole)
        v.undo()
        assert v._model._dataframe.iloc[0, 0] == 1
        v.redo()
        assert v._model._dataframe.iloc[0, 0] == 55

    def test_insert_row_via_view(self):
        v = self._make_view()
        v.table.setCurrentIndex(v._model.index(1, 0))
        v._insert_row()
        assert len(v._model._dataframe) == 4
        assert pd.isna(v._model._dataframe.iloc[1, 0])
