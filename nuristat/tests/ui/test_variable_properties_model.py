"""변수 설정 모델(VariablePropertiesModel) 추가 테스트.

SPSS Variable View의 11개 속성 표시·편집·행 조작·데이터셋 동기화 검증.
ui/* 는 커버리지 집계 제외 대상이나 회귀 방지를 위한 동작 검증으로 가치가 있다.
"""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, Role, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.variable_view import VariablePropertiesModel


@pytest.fixture
def dataset() -> Dataset:
    df = pd.DataFrame({"age": [20, 30, 40], "city": ["A", "B", "C"]})
    variables = {
        "age": VariableMeta(name="age", label="나이",
                            storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "city": VariableMeta(name="city", label="도시",
                             storage_type=StorageType.STRING, measure=MeasureType.NOMINAL),
    }
    return Dataset(df, "test", variables)


@pytest.fixture
def model(dataset) -> VariablePropertiesModel:
    return VariablePropertiesModel(dataset)


def _disp(model, row, col):
    return model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)


def _set(model, row, col, value):
    return model.setData(model.index(row, col), value, Qt.ItemDataRole.EditRole)


# ─────────────────────────────────────────────────────────────
# 1. 표시(DisplayRole)
# ─────────────────────────────────────────────────────────────

class TestDisplay:

    def test_row_and_column_count(self, model):
        assert model.rowCount() == 2
        assert model.columnCount() == 11

    def test_name_display(self, model):
        assert _disp(model, 0, 0) == "age"
        assert _disp(model, 1, 0) == "city"

    def test_type_display(self, model):
        assert _disp(model, 0, 1) == "정수형"
        assert _disp(model, 1, 1) == "문자열"

    def test_width_decimals_label_display(self, model):
        assert _disp(model, 0, 4) == "나이"   # label
        # width/decimals 는 기본값 문자열
        assert _disp(model, 0, 2).isdigit()
        assert _disp(model, 0, 3).isdigit()

    def test_values_missing_display_default(self, model):
        assert _disp(model, 0, 5) == "없음"   # value_labels 없음
        assert _disp(model, 0, 6) == "없음"   # missing_values 없음

    def test_measure_display(self, model):
        assert _disp(model, 0, 9) == "척도"     # SCALE
        assert _disp(model, 1, 9) == "명목형"   # NOMINAL

    def test_role_display_default(self, model):
        # 기본 역할 표시 (입력 등)
        assert _disp(model, 0, 10) in ("입력", "없음")

    def test_invalid_index_returns_none(self, model):
        assert model.data(model.index(99, 0), Qt.ItemDataRole.DisplayRole) is None


# ─────────────────────────────────────────────────────────────
# 2. 편집(setData) — 컬럼별
# ─────────────────────────────────────────────────────────────

class TestEditing:

    def test_edit_name_syncs_dataset(self, model, dataset):
        assert _set(model, 0, 0, "years") is True
        assert "years" in dataset.variables
        assert "age" not in dataset.variables
        assert "years" in dataset.data.columns
        assert "age" not in dataset.data.columns

    def test_edit_type_to_string_resets_measure(self, model):
        # age(정수/SCALE) → 문자열 변경 시 measure가 NOMINAL로 조정
        assert _set(model, 0, 1, "문자열") is True
        var = model._variables[0]
        assert var.storage_type == StorageType.STRING
        assert var.measure == MeasureType.NOMINAL

    def test_edit_type_to_integer(self, model):
        assert _set(model, 1, 1, "정수형") is True
        assert model._variables[1].storage_type == StorageType.INTEGER

    def test_edit_width(self, model):
        assert _set(model, 0, 2, "12") is True
        assert model._variables[0].width == 12

    def test_edit_width_invalid_returns_false(self, model):
        assert _set(model, 0, 2, "abc") is False

    def test_edit_decimals(self, model):
        assert _set(model, 0, 3, "3") is True
        assert model._variables[0].decimals == 3

    def test_edit_label(self, model):
        assert _set(model, 0, 4, "연령") is True
        assert model._variables[0].label == "연령"

    def test_edit_columns_width(self, model):
        assert _set(model, 0, 7, "15") is True
        assert _disp(model, 0, 7) == "15"

    def test_edit_align(self, model):
        assert _set(model, 0, 8, "왼쪽") is True
        assert model._variables[0].align == "left"

    def test_edit_measure(self, model):
        assert _set(model, 0, 9, "순서형") is True
        assert model._variables[0].measure == MeasureType.ORDINAL

    def test_edit_role(self, model):
        assert _set(model, 0, 10, "목표") is True
        assert model._variables[0].role == Role.TARGET

    def test_edit_invalid_role_via_unknown_returns_true_noop(self, model):
        # 매핑에 없는 값 → 변경 없이 통과 (예외 없음)
        before = model._variables[0].role
        assert _set(model, 0, 10, "존재하지않는역할") is True
        assert model._variables[0].role == before

    def test_setdata_invalid_role_argument(self, model):
        assert model.setData(model.index(0, 0), "x", Qt.ItemDataRole.DisplayRole) is False


# ─────────────────────────────────────────────────────────────
# 3. 행 조작 (추가·삭제·이동)
# ─────────────────────────────────────────────────────────────

class TestRowOps:

    def test_add_variable(self, model, dataset):
        n = model.rowCount()
        model.add_variable("newvar")
        assert model.rowCount() == n + 1
        assert "newvar" in dataset.variables
        assert "newvar" in dataset.data.columns

    def test_remove_variable(self, model, dataset):
        assert model.remove_variable(0) is True
        assert "age" not in dataset.variables
        assert "age" not in dataset.data.columns
        assert model.rowCount() == 1

    def test_remove_variable_out_of_range(self, model):
        assert model.remove_variable(99) is False

    def test_move_variable_reorders(self, model, dataset):
        assert model.move_variable(0, 1) is True
        # age가 뒤로, city가 앞으로
        names = [v.name for v in model._variables]
        assert names == ["city", "age"]
        assert list(dataset.data.columns) == ["city", "age"]

    def test_move_variable_out_of_range(self, model):
        assert model.move_variable(0, 99) is False


# ─────────────────────────────────────────────────────────────
# 4. 데이터셋 없음 / 경계
# ─────────────────────────────────────────────────────────────

class TestBoundary:

    def test_empty_model_no_dataset(self):
        m = VariablePropertiesModel(None)
        assert m.rowCount() == 0
        m.add_variable("x")  # 데이터셋 없으면 무시 (예외 없음)
        assert m.rowCount() == 0

    def test_set_dataset_updates_rows(self, dataset):
        m = VariablePropertiesModel(None)
        assert m.rowCount() == 0
        m.set_dataset(dataset)
        assert m.rowCount() == 2
