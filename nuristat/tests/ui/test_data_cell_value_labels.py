"""데이터 셀 값 라벨 드롭다운 + column_width 직렬화 검증 (SPSS 편의).

범주형 변수(값 라벨 보유)는 데이터 셀에서 '코드 = 라벨' 드롭다운으로 선택,
일반 변수는 라인에디트로 타이핑 — SPSS 데이터 입력 동작과 동일.

담당 에이전트: ux-designer/frontend, tester-unit
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QStyleOptionViewItem,
    QTableView,
)

from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.delegates.cell_delegate import CellDelegate
from nuristat.ui.models.spss_grid_model import SPSSGridModel


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def setup():
    df = pd.DataFrame({"sex": [0, 1, 0], "score": [10, 20, 30]})
    variables = {
        "sex": VariableMeta(name="sex", storage_type=StorageType.INTEGER,
                            measure=MeasureType.NOMINAL, value_labels={0: "남", 1: "여"}),
        "score": VariableMeta(name="score", storage_type=StorageType.INTEGER,
                              measure=MeasureType.SCALE),
    }
    model = SPSSGridModel(df, variables)
    table = QTableView()
    table.setModel(model)
    delegate = CellDelegate(table)
    table.setItemDelegate(delegate)
    return model, table, delegate


class TestModelValueLabelsAccessor:
    def test_labeled_col_returns_dict(self, setup):
        model, _, _ = setup
        assert model.value_labels_for_col(0) == {0: "남", 1: "여"}

    def test_unlabeled_col_returns_none(self, setup):
        model, _, _ = setup
        assert model.value_labels_for_col(1) is None

    def test_out_of_range_returns_none(self, setup):
        model, _, _ = setup
        assert model.value_labels_for_col(99) is None


class TestCellEditorType:
    def test_labeled_col_uses_combobox(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(0, 0))
        assert isinstance(ed, QComboBox)

    def test_unlabeled_col_uses_lineedit(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(0, 1))
        assert isinstance(ed, QLineEdit)

    def test_combo_items_show_code_and_label(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(0, 0))
        items = [(ed.itemText(i), ed.itemData(i)) for i in range(ed.count())]
        assert items == [("0 = 남", "0"), ("1 = 여", "1")]


class TestCellEditorBehavior:
    def test_set_editor_data_selects_current_code(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(0, 0))
        delegate._editor_initialized = False
        delegate.setEditorData(ed, model.index(0, 0))   # 코드 0
        assert ed.currentIndex() == 0

    def test_set_model_data_stores_code_not_label(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(2, 0))
        ed.setCurrentIndex(1)   # "1 = 여"
        delegate.setModelData(ed, model, model.index(2, 0))
        assert str(model._dataframe.iloc[2, 0]) == "1"

    def test_free_entry_code_kept(self, setup):
        """라벨에 없는 코드를 직접 입력하면 그대로 저장(SPSS 자유 입력)."""
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(2, 0))
        ed.setEditText("9")
        delegate.setModelData(ed, model, model.index(2, 0))
        assert str(model._dataframe.iloc[2, 0]) == "9"

    def test_lineedit_path_unchanged(self, setup):
        model, table, delegate = setup
        ed = delegate.createEditor(table, QStyleOptionViewItem(), model.index(0, 1))
        ed.setText("42")
        delegate.setModelData(ed, model, model.index(0, 1))
        assert str(model._dataframe.iloc[0, 1]) == "42"


class TestColumnWidthSerialization:
    def test_field_default(self):
        v = VariableMeta(name="x")
        assert v.column_width == 8

    def test_roundtrip_preserves_column_width(self):
        v = VariableMeta(name="x", width=10, column_width=15)
        restored = VariableMeta.from_dict(v.to_dict())
        assert restored.column_width == 15
        assert restored.width == 10

    def test_legacy_dict_without_column_width(self):
        """예전 저장 파일(column_width 없음)도 width로 보정돼 로드."""
        legacy = VariableMeta(name="x", width=12).to_dict()
        del legacy["column_width"]
        restored = VariableMeta.from_dict(legacy)
        assert restored.column_width == 12
