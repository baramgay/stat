"""데이터 입력 최적화·견고성 추가 테스트.

대상: SPSSGridModel
  - batch_update() 배치 편집 견고성 (예외 안전·구조 변경·메타데이터)
  - 데이터 입력 엣지케이스 (행/열 추가·삭제·정렬·값 라벨)
  - 변수 메타데이터 자동 감지·승격·보호 규칙

ui/* 는 커버리지 집계 제외 대상이나, 회귀 방지를 위한 동작 검증으로 가치가 있다.
"""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.models.spss_grid_model import SPSSGridModel


def _edit(model, r, c, value):
    return model.setData(model.index(r, c), value, Qt.ItemDataRole.EditRole)


def _signal_counter(model):
    box = {"n": 0}
    model.data_changed.connect(lambda: box.__setitem__("n", box["n"] + 1))
    return box


# ─────────────────────────────────────────────────────────────
# 1. 배치 업데이트 견고성
# ─────────────────────────────────────────────────────────────

class TestBatchRobustness:

    def test_batch_flushes_even_on_exception(self):
        """배치 블록 안에서 예외가 나가도 try/finally로 데이터·신호가 flush됨."""
        model = SPSSGridModel()
        counter = _signal_counter(model)
        with pytest.raises(RuntimeError):
            with model.batch_update():
                _edit(model, 0, 0, "10")
                _edit(model, 1, 0, "20")
                raise RuntimeError("중단")
        # 예외 전 입력은 보존되고 신호는 1회 방출
        assert counter["n"] == 1
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 10
        assert df.iloc[1, 0] == 20

    def test_batch_creates_columns_structurally(self):
        """배치 중 신규 열 생성(구조 변경)이 정상 동작."""
        model = SPSSGridModel()
        with model.batch_update():
            _edit(model, 0, 0, "1")
            _edit(model, 0, 2, "3")  # 중간 열 건너뛰어 VAR00001~VAR00003 생성
        df = model.get_dataframe()
        assert len(df.columns) == 3
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 2] == 3

    def test_batch_extends_rows(self):
        """배치 중 행 확장이 정상 동작."""
        model = SPSSGridModel()
        with model.batch_update():
            for r in range(50):
                _edit(model, r, 0, str(r))
        df = model.get_dataframe()
        assert len(df) == 50
        assert df.iloc[49, 0] == 49

    def test_batch_promotes_storage_type_once(self):
        """배치 중 정수→실수 입력 시 storage_type이 FLOAT로 승격."""
        model = SPSSGridModel()
        with model.batch_update():
            _edit(model, 0, 0, "5")     # INTEGER
            _edit(model, 1, 0, "5.5")   # FLOAT 승격
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.FLOAT

    def test_batch_then_single_edit_consistent(self):
        """배치 후 단일 편집이 정상적으로 신호 1회 방출."""
        model = SPSSGridModel()
        with model.batch_update():
            _edit(model, 0, 0, "1")
        counter = _signal_counter(model)
        _edit(model, 1, 0, "2")  # 배치 밖 단일 편집
        assert counter["n"] == 1


# ─────────────────────────────────────────────────────────────
# 2. 행/열 조작 엣지케이스
# ─────────────────────────────────────────────────────────────

class TestRowColumnOps:

    def test_remove_row_updates_last_data_row(self):
        """행 삭제 후 마지막 데이터 행이 갱신됨."""
        model = SPSSGridModel()
        for r in range(5):
            _edit(model, r, 0, str(r))
        assert len(model.get_dataframe()) == 5
        model.remove_row(4)
        assert len(model.get_dataframe()) == 4

    def test_remove_row_out_of_range_returns_false(self):
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        assert model.remove_row(999) is False

    def test_add_column_with_values(self):
        """값과 함께 열 추가."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        _edit(model, 1, 0, "2")
        model.add_column("score", [10, 20])
        df = model.get_dataframe()
        assert "score" in df.columns
        assert df["score"].tolist() == [10, 20]

    def test_remove_column_deletes_variable_meta(self):
        """열 삭제 시 변수 메타데이터도 제거."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        _edit(model, 0, 1, "2")
        assert "VAR00001" in model.get_variables()
        model.remove_column(0)
        assert "VAR00001" not in model.get_variables()
        assert "VAR00002" in model.get_variables()

    def test_remove_column_out_of_range(self):
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        assert model.remove_column(999) is False

    def test_add_row_with_values(self):
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        model.add_row({"VAR00001": 99})
        df = model.get_dataframe()
        assert 99 in df["VAR00001"].tolist()


# ─────────────────────────────────────────────────────────────
# 3. 정렬·값 라벨
# ─────────────────────────────────────────────────────────────

class TestSortAndLabels:

    def test_sort_by_column_ascending(self):
        model = SPSSGridModel()
        for r, v in enumerate(["3", "1", "2"]):
            _edit(model, r, 0, v)
        model.sort_by_column(0, ascending=True)
        df = model.get_dataframe()
        assert df.iloc[:, 0].tolist() == [1, 2, 3]

    def test_sort_by_column_descending(self):
        model = SPSSGridModel()
        for r, v in enumerate(["1", "3", "2"]):
            _edit(model, r, 0, v)
        model.sort_by_column(0, ascending=False)
        df = model.get_dataframe()
        assert df.iloc[:, 0].tolist() == [3, 2, 1]

    def test_sort_out_of_range_column_noop(self):
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        model.sort_by_column(999)  # 예외 없이 무시
        assert model.get_dataframe().iloc[0, 0] == 1

    def test_toggle_value_labels(self):
        model = SPSSGridModel()
        assert model.show_value_labels is False
        assert model.toggle_value_labels() is True
        assert model.show_value_labels is True
        assert model.toggle_value_labels() is False

    def test_value_label_display(self):
        """값 라벨 모드에서 라벨 텍스트가 표시됨."""
        df = pd.DataFrame({"gender": [0, 1, 0]})
        var = VariableMeta(name="gender", label="성별", storage_type=StorageType.INTEGER,
                           measure=MeasureType.NOMINAL)
        var.value_labels = {"0": "남", "1": "여"}
        model = SPSSGridModel(df, {"gender": var})
        model.show_value_labels = True
        # DisplayRole에서 라벨 반환
        disp = model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole)
        assert disp == "남"


# ─────────────────────────────────────────────────────────────
# 4. 변수 메타데이터 자동 감지·보호
# ─────────────────────────────────────────────────────────────

class TestVariableMetadata:

    def test_numeric_first_entry_sets_scale(self):
        """숫자 첫 입력 → SCALE 자동 감지."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "10")
        assert model.get_variables()["VAR00001"].measure == MeasureType.SCALE

    def test_string_first_entry_sets_nominal(self):
        """문자 첫 입력 → NOMINAL 자동 감지."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "apple")
        assert model.get_variables()["VAR00001"].measure == MeasureType.NOMINAL

    def test_numeric_column_rejects_string(self):
        """숫자형 확정 변수에 문자 입력 거부 (SPSS 호환)."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "10")
        ok = _edit(model, 1, 0, "hello")
        assert ok is False

    def test_mark_measure_initialized_protects_user_setting(self):
        """사용자가 설정한 측정 척도는 이후 데이터 입력으로 덮어쓰이지 않음."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")  # 첫 입력 → SCALE, 초기화 완료
        var = model.get_variables()["VAR00001"]
        var.measure = MeasureType.ORDINAL  # 사용자가 변경
        model.mark_measure_initialized("VAR00001")
        _edit(model, 1, 0, "2")  # 추가 입력
        assert model.get_variables()["VAR00001"].measure == MeasureType.ORDINAL

    def test_decimals_tracked_from_input(self):
        """실수 입력 시 소수 자릿수가 추적됨."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1.5")
        _edit(model, 1, 0, "2.345")
        var = model.get_variables()["VAR00001"]
        assert var.decimals >= 3

    def test_rename_header_updates_variable(self):
        """헤더(변수명) 변경 시 메타데이터 키도 갱신."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        ok = model.setHeaderData(0, Qt.Orientation.Horizontal, "age", Qt.ItemDataRole.EditRole)
        assert ok is True
        assert "age" in model.get_variables()
        assert "VAR00001" not in model.get_variables()

    def test_rename_to_existing_name_rejected(self):
        """이미 존재하는 변수명으로 변경 거부."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        _edit(model, 0, 1, "2")
        ok = model.setHeaderData(0, Qt.Orientation.Horizontal, "VAR00002", Qt.ItemDataRole.EditRole)
        assert ok is False


# ─────────────────────────────────────────────────────────────
# 5. get_dataframe dtype 정합
# ─────────────────────────────────────────────────────────────

class TestDataFrameOutput:

    def test_numeric_column_coerced_to_numeric_dtype(self):
        """object로 저장된 숫자 열이 출력 시 수치형 dtype으로 변환."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "10")
        _edit(model, 1, 0, "20")
        df = model.get_dataframe()
        assert pd.api.types.is_numeric_dtype(df["VAR00001"])

    def test_empty_model_returns_empty_dataframe(self):
        model = SPSSGridModel()
        assert model.get_dataframe().empty

    def test_get_dataframe_trims_trailing_empty_rows(self):
        """마지막 데이터 행 이후의 빈 행은 출력에서 제외."""
        model = SPSSGridModel()
        _edit(model, 0, 0, "1")
        _edit(model, 2, 0, "3")  # 행 1은 비어있음
        df = model.get_dataframe()
        # 0,1,2 행 (마지막 데이터=2) 까지만
        assert len(df) == 3
        assert df.iloc[0, 0] == 1
        assert pd.isna(df.iloc[1, 0])
        assert df.iloc[2, 0] == 3
