"""기본 기능 견고성 — 데이터 입력·변수 설정 다양한 엣지케이스 검증.

사용자에게 가장 중요한 기본 기능(셀 입력, 변수 속성 설정)이 다양한 입력에서
문제 없이 동작하는지 폭넓게 검증한다.

- 데이터 입력(SPSSGridModel): 음수·소수·지수·한글·특수문자·결측·덮어쓰기·
  타입승격·붙여넣기·정렬·행열조작 등
- 변수 설정(VariablePropertiesModel): 전 유형/측정/역할·경계값·이름규칙·
  타입변경 연쇄·데이터셋 동기화 등
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_app = QApplication.instance() or QApplication(sys.argv)

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, Role, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.models.spss_grid_model import SPSSGridModel
from nuristat.ui.variable_view import VariablePropertiesModel

EDIT = Qt.ItemDataRole.EditRole
DISPLAY = Qt.ItemDataRole.DisplayRole


def _set(m, r, c, v):
    return m.setData(m.index(r, c), v, EDIT)


# ════════════════════════════════════════════════════════════════════
# 데이터 입력 (SPSSGridModel) — 다양한 값 입력
# ════════════════════════════════════════════════════════════════════

class TestNumericEntryVariety:
    """숫자 입력의 다양한 형태."""

    @pytest.mark.parametrize("text,expected", [
        ("0", 0), ("42", 42), ("-7", -7), ("1000000", 1000000),
        ("3.14", 3.14), ("-2.5", -2.5), ("0.001", 0.001),
        ("1e3", 1000.0), ("1.5e-2", 0.015), ("100.0", 100.0),
    ])
    def test_numeric_forms(self, text, expected):
        m = SPSSGridModel()
        assert _set(m, 0, 0, text)
        val = m.get_dataframe().iloc[0, 0]
        assert abs(float(val) - expected) < 1e-9, f"{text} → {val}"

    def test_negative_then_positive_accumulate_independent(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "-5")
        _set(m, 1, 0, "10")
        df = m.get_dataframe()
        assert df.iloc[0, 0] == -5 and df.iloc[1, 0] == 10

    def test_int_then_float_promotes_to_float(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "5")
        _set(m, 1, 0, "5.5")
        var = m.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.FLOAT

    def test_leading_zero_preserved_as_number(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "007")
        assert m.get_dataframe().iloc[0, 0] == 7


class TestTextEntryVariety:
    """문자 입력의 다양한 형태 (한글·특수문자·길이)."""

    @pytest.mark.parametrize("text", [
        "사과", "서울특별시", "abc", "Hello World", "a-b_c",
        "특수!@#$%", "  공백포함  ".strip(), "긴문자열" * 20, "123abc", "true",
    ])
    def test_string_values(self, text):
        m = SPSSGridModel()
        assert _set(m, 0, 0, text)
        # 문자열은 NOMINAL/STRING로 저장
        var = m.get_variables()["VAR00001"]
        assert var.measure == MeasureType.NOMINAL
        assert str(m.get_dataframe().iloc[0, 0]) == text

    def test_korean_in_multiple_cells(self):
        m = SPSSGridModel()
        vals = ["남성", "여성", "남성", "기타"]
        for i, v in enumerate(vals):
            _set(m, i, 0, v)
        assert m.get_dataframe().iloc[:, 0].tolist() == vals


class TestEntryEdgeCases:
    """입력 경계 — 결측·덮어쓰기·지우기·거부."""

    def test_dot_becomes_missing(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "5")       # 숫자열 확정
        _set(m, 1, 0, ".")       # SPSS 결측 기호
        df = m.get_dataframe()
        assert df.iloc[0, 0] == 5
        assert pd.isna(df.iloc[1, 0])

    def test_empty_string_clears_cell(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "9")
        _set(m, 0, 0, "")
        df = m.get_dataframe()
        assert len(df) == 0 or pd.isna(df.iloc[0, 0])

    def test_overwrite_cell_value(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        _set(m, 0, 0, "2")
        assert m.get_dataframe().iloc[0, 0] == 2

    def test_numeric_column_rejects_text(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "10")       # 숫자열 확정
        assert _set(m, 1, 0, "abc") is False

    def test_string_column_accepts_number_as_text(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "hello")    # 문자열 확정
        # 문자열 열에 숫자 입력 — 거부되지 않음
        assert _set(m, 1, 0, "123") is not False


class TestPasteRobustness:
    """배치(붙여넣기) 다양한 형태."""

    def test_paste_rectangular_block(self):
        m = SPSSGridModel()
        block = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
        with m.batch_update():
            for r, row in enumerate(block):
                for c, v in enumerate(row):
                    _set(m, r, c, v)
        df = m.get_dataframe()
        assert df.shape == (3, 3)
        assert df.iloc[2, 2] == 9

    def test_paste_with_missing_cells(self):
        m = SPSSGridModel()
        with m.batch_update():
            _set(m, 0, 0, "1")
            _set(m, 0, 1, "")    # 빈 셀
            _set(m, 0, 2, "3")
        df = m.get_dataframe()
        assert df.iloc[0, 0] == 1 and df.iloc[0, 2] == 3

    def test_large_batch_integrity(self):
        m = SPSSGridModel()
        with m.batch_update():
            for r in range(100):
                _set(m, r, 0, str(r))
        df = m.get_dataframe()
        assert len(df) == 100
        assert df.iloc[:, 0].tolist() == list(range(100))

    def test_batch_signal_emitted_once(self):
        m = SPSSGridModel()
        cnt = {"n": 0}
        m.data_changed.connect(lambda: cnt.__setitem__("n", cnt["n"] + 1))
        with m.batch_update():
            for r in range(10):
                _set(m, r, 0, str(r))
        assert cnt["n"] == 1


class TestGridStructureOps:
    """행/열 조작·정렬·헤더 다양한 케이스."""

    def test_sort_numeric_preserves_data(self):
        m = SPSSGridModel()
        for r, v in enumerate(["3", "1", "2", "5", "4"]):
            _set(m, r, 0, v)
        m.sort_by_column(0, ascending=True)
        assert m.get_dataframe().iloc[:, 0].tolist() == [1, 2, 3, 4, 5]

    def test_sort_descending(self):
        m = SPSSGridModel()
        for r, v in enumerate(["1", "3", "2"]):
            _set(m, r, 0, v)
        m.sort_by_column(0, ascending=False)
        assert m.get_dataframe().iloc[:, 0].tolist() == [3, 2, 1]

    def test_header_rename_basic(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        assert m.setHeaderData(0, Qt.Orientation.Horizontal, "나이", EDIT)
        assert "나이" in m.get_variables()

    def test_header_rename_duplicate_rejected(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        _set(m, 0, 1, "2")
        assert m.setHeaderData(0, Qt.Orientation.Horizontal, "VAR00002", EDIT) is False

    def test_header_rename_empty_rejected(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        assert m.setHeaderData(0, Qt.Orientation.Horizontal, "   ", EDIT) is False

    def test_header_rename_korean_special(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        assert m.setHeaderData(0, Qt.Orientation.Horizontal, "소득_2024", EDIT)
        assert "소득_2024" in m.get_variables()

    def test_remove_then_readd_column(self):
        m = SPSSGridModel()
        _set(m, 0, 0, "1")
        _set(m, 0, 1, "2")
        m.remove_column(0)
        assert "VAR00001" not in m.get_variables()
        m.add_column("new", [10])
        assert "new" in m.get_variables()

    def test_add_remove_row_integrity(self):
        m = SPSSGridModel()
        for r in range(5):
            _set(m, r, 0, str(r))
        m.remove_row(2)
        assert len(m.get_dataframe()) == 4

    def test_value_labels_display_after_entry(self):
        df = pd.DataFrame({"g": [0, 1, 0]})
        var = VariableMeta(name="g", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)
        var.value_labels = {"0": "남", "1": "여"}
        m = SPSSGridModel(df, {"g": var})
        m.show_value_labels = True
        assert m.data(m.index(0, 0), DISPLAY) == "남"
        assert m.data(m.index(1, 0), DISPLAY) == "여"


# ════════════════════════════════════════════════════════════════════
# 변수 설정 (VariablePropertiesModel) — 전 속성 다양한 케이스
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def var_model():
    df = pd.DataFrame({"v1": [1, 2, 3], "v2": ["a", "b", "c"]})
    variables = {
        "v1": VariableMeta(name="v1", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "v2": VariableMeta(name="v2", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL),
    }
    ds = Dataset(df, "t", variables)
    return VariablePropertiesModel(ds), ds


class TestVariableTypeSetting:
    """유형(Type) 설정 — 전 유형 + 연쇄 효과."""

    @pytest.mark.parametrize("display,expected", [
        ("숫자형", StorageType.FLOAT),
        ("정수형", StorageType.INTEGER),
        ("문자열", StorageType.STRING),
        ("날짜/시간", StorageType.DATETIME),
        ("논리형", StorageType.BOOLEAN),
        ("범주형", StorageType.CATEGORICAL),
    ])
    def test_set_each_type(self, var_model, display, expected):
        m, _ = var_model
        assert m.setData(m.index(0, 1), display, EDIT)
        assert m._variables[0].storage_type == expected

    def test_type_to_string_adjusts_measure(self, var_model):
        m, _ = var_model
        # v1(SCALE) → 문자열 변경 시 measure가 NOMINAL로 조정
        m.setData(m.index(0, 1), "문자열", EDIT)
        assert m._variables[0].measure == MeasureType.NOMINAL


class TestVariableMeasureRole:
    """측정·역할 설정 — 전 항목."""

    @pytest.mark.parametrize("display,expected", [
        ("척도", MeasureType.SCALE), ("순서형", MeasureType.ORDINAL),
        ("명목형", MeasureType.NOMINAL), ("이분형", MeasureType.BINARY),
    ])
    def test_set_each_measure(self, var_model, display, expected):
        m, _ = var_model
        assert m.setData(m.index(0, 9), display, EDIT)
        assert m._variables[0].measure == expected

    @pytest.mark.parametrize("display,expected", [
        ("입력", Role.INPUT), ("목표", Role.TARGET), ("가중치", Role.WEIGHT),
        ("ID", Role.ID), ("분리", Role.SPLIT), ("빈도", Role.FREQUENCY), ("없음", Role.NONE),
    ])
    def test_set_each_role(self, var_model, display, expected):
        m, _ = var_model
        assert m.setData(m.index(0, 10), display, EDIT)
        assert m._variables[0].role == expected


class TestVariableNumericProps:
    """너비·소수 등 숫자 속성 경계값."""

    def test_width_normal(self, var_model):
        m, _ = var_model
        assert m.setData(m.index(0, 2), "15", EDIT)
        assert m._variables[0].width == 15

    def test_width_zero_clamped(self, var_model):
        m, _ = var_model
        m.setData(m.index(0, 2), "0", EDIT)
        assert m._variables[0].width >= 1

    def test_width_invalid_rejected(self, var_model):
        m, _ = var_model
        assert m.setData(m.index(0, 2), "abc", EDIT) is False

    def test_decimals_set(self, var_model):
        m, _ = var_model
        assert m.setData(m.index(0, 3), "4", EDIT)
        assert m._variables[0].decimals == 4

    def test_decimals_negative_clamped(self, var_model):
        m, _ = var_model
        m.setData(m.index(0, 3), "-2", EDIT)
        assert m._variables[0].decimals >= 0


class TestVariableNameLabel:
    """이름·라벨 다양한 입력."""

    def test_rename_syncs_dataset_and_data(self, var_model):
        m, ds = var_model
        m.setData(m.index(0, 0), "age", EDIT)
        assert "age" in ds.variables and "age" in ds.data.columns
        assert "v1" not in ds.variables

    def test_rename_korean(self, var_model):
        m, ds = var_model
        m.setData(m.index(0, 0), "나이", EDIT)
        assert "나이" in ds.variables

    def test_label_unicode_and_long(self, var_model):
        m, _ = var_model
        long_label = "응답자의 만 나이(세) " * 5
        m.setData(m.index(0, 4), long_label, EDIT)
        assert m._variables[0].label == long_label

    def test_align_options(self, var_model):
        m, _ = var_model
        m.setData(m.index(0, 8), "왼쪽", EDIT)
        assert m._variables[0].align == "left"
        m.setData(m.index(0, 8), "가운데", EDIT)
        assert m._variables[0].align == "center"


class TestVariableRowOps:
    """변수 추가·삭제·이동."""

    def test_add_variable(self, var_model):
        m, ds = var_model
        m.add_variable("v3")
        assert m.rowCount() == 3 and "v3" in ds.variables

    def test_remove_variable_syncs_data(self, var_model):
        m, ds = var_model
        m.remove_variable(0)
        assert "v1" not in ds.variables and "v1" not in ds.data.columns

    def test_move_variable_reorders_data(self, var_model):
        m, ds = var_model
        m.move_variable(0, 1)
        assert [v.name for v in m._variables] == ["v2", "v1"]
        assert list(ds.data.columns) == ["v2", "v1"]

    def test_move_out_of_range_rejected(self, var_model):
        m, _ = var_model
        assert m.move_variable(0, 99) is False

    def test_remove_all_variables_one_by_one(self, var_model):
        m, ds = var_model
        m.remove_variable(0)
        m.remove_variable(0)
        assert m.rowCount() == 0 and len(ds.variables) == 0
