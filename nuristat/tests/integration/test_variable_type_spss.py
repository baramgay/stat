"""변수 유형 자동 감지 및 데이터 입력 SPSS 호환 종합 테스트.

검증 항목:
1. 변수 유형 자동 감지 (숫자→SCALE, 문자→NOMINAL) - SPSS 29 규칙
2. 저장 유형 감지 (정수→INTEGER, 소수→FLOAT, 문자→STRING)
3. 결측치 처리 ("." → pd.NA)
4. 다양한 입력 패턴 (0/1, 리커트, Likert, 연속형, 범주형 텍스트)
5. 첫 번째 입력 후 모델 구조 일관성 (beginResetModel 없음)
6. 다중 열 생성 시 독립적 유형 감지
7. 값 재입력 시 유형 유지
8. SPSS 결측 기호 처리

SPSS 29 기준:
  - 숫자 변수: 기본 측정 척도 = SCALE (고유값 수 무관)
  - 문자 변수: 기본 측정 척도 = NOMINAL
  - BINARY/ORDINAL은 사용자가 Variable View에서 직접 지정
"""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nuristat.ui.models.spss_grid_model import (
    SPSSGridModel,
    infer_measure_type,
    infer_storage_type,
)
from nuristat.core.typing import MeasureType, StorageType


def _enter(model: SPSSGridModel, row: int, col: int, val: str) -> bool:
    return model.setData(model.index(row, col), val, Qt.ItemDataRole.EditRole)


# ──────────────────────────────────────────────────────────────
# 1. infer_storage_type 단위 테스트
# ──────────────────────────────────────────────────────────────

class TestInferStorageType:
    """infer_storage_type 함수 단위 테스트."""

    def test_python_int(self):
        assert infer_storage_type(5) == StorageType.INTEGER

    def test_python_bool_as_integer(self):
        assert infer_storage_type(True) == StorageType.INTEGER

    def test_python_float(self):
        assert infer_storage_type(3.14) == StorageType.FLOAT

    def test_string_integer(self):
        assert infer_storage_type("42") == StorageType.INTEGER

    def test_string_negative_integer(self):
        assert infer_storage_type("-10") == StorageType.INTEGER

    def test_string_float(self):
        assert infer_storage_type("3.14") == StorageType.FLOAT

    def test_string_float_with_only_dot(self):
        assert infer_storage_type("1.0") == StorageType.FLOAT

    def test_string_text(self):
        assert infer_storage_type("hello") == StorageType.STRING

    def test_string_mixed(self):
        assert infer_storage_type("12a") == StorageType.STRING

    def test_empty_string(self):
        assert infer_storage_type("") == StorageType.STRING


# ──────────────────────────────────────────────────────────────
# 2. infer_measure_type 단위 테스트 — SPSS 규칙
# ──────────────────────────────────────────────────────────────

class TestInferMeasureTypeSPSS:
    """infer_measure_type 함수 SPSS 29 호환 단위 테스트.

    SPSS 29 기준:
      숫자형 → SCALE (고유값 수 무관)
      문자형 → NOMINAL
    """

    def _series(self, vals):
        return pd.Series(vals)

    def test_empty_series(self):
        assert infer_measure_type(self._series([])) == MeasureType.NOMINAL

    def test_single_integer(self):
        assert infer_measure_type(self._series([1])) == MeasureType.SCALE

    def test_two_integers_not_binary(self):
        """0/1 두 고유값 → SCALE (SPSS: 자동 BINARY 없음)."""
        assert infer_measure_type(self._series([0, 1])) == MeasureType.SCALE

    def test_five_integers_not_ordinal(self):
        """5개 고유값 → SCALE (SPSS: 자동 ORDINAL 없음)."""
        assert infer_measure_type(self._series([1, 2, 3, 4, 5])) == MeasureType.SCALE

    def test_ten_integers(self):
        """10개 고유값 → SCALE."""
        assert infer_measure_type(self._series(list(range(10)))) == MeasureType.SCALE

    def test_twenty_integers(self):
        """20개 고유값 → SCALE."""
        assert infer_measure_type(self._series(list(range(20)))) == MeasureType.SCALE

    def test_float_values(self):
        """실수 → SCALE."""
        assert infer_measure_type(self._series([1.1, 2.2, 3.3])) == MeasureType.SCALE

    def test_numeric_strings(self):
        """숫자 문자열 → SCALE."""
        assert infer_measure_type(self._series(["1", "2", "3"])) == MeasureType.SCALE

    def test_single_string(self):
        """문자 1개 → NOMINAL."""
        assert infer_measure_type(self._series(["Male"])) == MeasureType.NOMINAL

    def test_two_strings_not_binary(self):
        """문자 2개 고유값 → NOMINAL (SPSS: 자동 BINARY 없음)."""
        assert infer_measure_type(self._series(["Male", "Female"])) == MeasureType.NOMINAL

    def test_many_strings(self):
        """문자 다수 → NOMINAL."""
        assert infer_measure_type(self._series([f"cat{i}" for i in range(15)])) == MeasureType.NOMINAL

    def test_with_na_values_numeric(self):
        """결측치 포함 숫자 → SCALE."""
        s = pd.Series([1, 2, pd.NA, 3])
        assert infer_measure_type(s) == MeasureType.SCALE

    def test_with_na_values_string(self):
        """결측치 포함 문자 → NOMINAL."""
        s = pd.Series(["A", "B", pd.NA, "C"])
        assert infer_measure_type(s) == MeasureType.NOMINAL

    def test_likert_5point(self):
        """리커트 5점 척도 (1-5 정수) → SCALE (SPSS 기본)."""
        s = pd.Series([1, 2, 3, 4, 5, 3, 2, 4, 1, 5])
        assert infer_measure_type(s) == MeasureType.SCALE


# ──────────────────────────────────────────────────────────────
# 3. SPSSGridModel 변수 생성 및 유형 감지
# ──────────────────────────────────────────────────────────────

class TestVariableCreationSPSS:
    """SPSSGridModel 데이터 입력 시 변수 생성 및 유형 SPSS 호환 테스트."""

    # -- 숫자형 입력 --

    def test_integer_input_creates_scale_variable(self):
        """정수 입력 → StorageType.INTEGER, MeasureType.SCALE."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "5")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.SCALE

    def test_float_input_creates_scale_variable(self):
        """소수 입력 → StorageType.FLOAT, MeasureType.SCALE."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "3.14")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.FLOAT
        assert var.measure == MeasureType.SCALE

    def test_zero_one_is_scale_not_binary(self):
        """0/1 입력 → SCALE (이전: BINARY). SPSS 호환."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "0")
        _enter(model, 1, 0, "1")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.SCALE

    def test_likert_3values_is_scale_not_ordinal(self):
        """리커트 3개 → SCALE (이전: ORDINAL). SPSS 호환."""
        model = SPSSGridModel()
        for v, r in [("1", 0), ("2", 1), ("3", 2)]:
            _enter(model, r, 0, v)
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_likert_5values_is_scale_not_ordinal(self):
        """리커트 5점 → SCALE. SPSS 호환."""
        model = SPSSGridModel()
        for i in range(10):
            _enter(model, i, 0, str((i % 5) + 1))
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_continuous_many_values_is_scale(self):
        """연속형 다수 값 → SCALE."""
        model = SPSSGridModel()
        for i in range(30):
            _enter(model, i, 0, str(i * 2 + 1))
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_negative_integers_is_scale(self):
        """음수 정수 → StorageType.INTEGER, MeasureType.SCALE."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "-5")
        _enter(model, 1, 0, "0")
        _enter(model, 2, 0, "5")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.SCALE

    # -- 문자형 입력 --

    def test_string_input_creates_nominal_variable(self):
        """문자 입력 → StorageType.STRING, MeasureType.NOMINAL."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "Male")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.STRING
        assert var.measure == MeasureType.NOMINAL

    def test_two_string_values_is_nominal_not_binary(self):
        """문자 2개 → NOMINAL (이전: BINARY). SPSS 호환."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "Yes")
        _enter(model, 1, 0, "No")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.NOMINAL

    def test_many_string_values_is_nominal(self):
        """문자 다수 → NOMINAL."""
        model = SPSSGridModel()
        cities = ["Seoul", "Busan", "Incheon", "Daegu", "Gwangju"]
        for i, c in enumerate(cities):
            _enter(model, i, 0, c)
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.NOMINAL

    # -- 결측치 처리 --

    def test_dot_is_missing_numeric(self):
        """숫자 열에 "." 입력 → 결측치(pd.NA)."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")
        _enter(model, 1, 0, ".")
        _enter(model, 2, 0, "20")
        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0])
        assert df.iloc[0, 0] == 10
        assert df.iloc[2, 0] == 20

    def test_empty_string_is_missing(self):
        """빈 문자열 → 결측치."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "5")
        _enter(model, 1, 0, "")
        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0])

    def test_missing_then_data_keeps_type(self):
        """결측 후 데이터 입력 → 변수 유형 유지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1")
        _enter(model, 1, 0, ".")
        _enter(model, 2, 0, "3")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    # -- 다중 열 독립 유형 --

    def test_two_columns_independent_types(self):
        """두 열 → 각각 독립적으로 유형 감지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")    # 숫자
        _enter(model, 0, 1, "Male")  # 문자
        v1 = model.get_variables()["VAR00001"]
        v2 = model.get_variables()["VAR00002"]
        assert v1.storage_type == StorageType.INTEGER
        assert v1.measure == MeasureType.SCALE
        assert v2.storage_type == StorageType.STRING
        assert v2.measure == MeasureType.NOMINAL

    def test_three_columns_mixed_types(self):
        """세 열: 정수/실수/문자 → 각각 올바른 유형."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "100")
        _enter(model, 0, 1, "3.14")
        _enter(model, 0, 2, "Group_A")
        v = model.get_variables()
        assert v["VAR00001"].storage_type == StorageType.INTEGER
        assert v["VAR00002"].storage_type == StorageType.FLOAT
        assert v["VAR00003"].storage_type == StorageType.STRING
        assert v["VAR00001"].measure == MeasureType.SCALE
        assert v["VAR00002"].measure == MeasureType.SCALE
        assert v["VAR00003"].measure == MeasureType.NOMINAL


# ──────────────────────────────────────────────────────────────
# 4. 모델 구조 안정성 (beginResetModel 제거 검증)
# ──────────────────────────────────────────────────────────────

class TestModelStructureStability:
    """첫 번째 입력 후 모델 구조 안정성 테스트.

    beginResetModel 호출이 제거됨으로써 뷰가 리셋되지 않아야 함.
    """

    def test_row_count_unchanged_after_first_entry(self):
        """첫 번째 값 입력 후 rowCount는 DEFAULT_ROWS 유지."""
        model = SPSSGridModel()
        before = model.rowCount()
        _enter(model, 0, 0, "1")
        after = model.rowCount()
        assert after >= before  # 줄어들지 않아야 함
        assert after >= model.DEFAULT_ROWS

    def test_column_count_unchanged_within_default(self):
        """컬럼 수가 DEFAULT_COLS 범위 내면 columnCount 변화 없음."""
        model = SPSSGridModel()
        before = model.columnCount()
        _enter(model, 0, 0, "1")
        after = model.columnCount()
        assert after == before  # 가상 그리드 크기 변화 없음

    def test_second_column_does_not_reset_first(self):
        """두 번째 열 생성 시 첫 번째 열 데이터 보존."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "42")
        _enter(model, 0, 1, "hello")
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 42
        assert df.iloc[0, 1] == "hello"

    def test_sequential_rows_in_single_column(self):
        """단일 열에 순차 입력 → 데이터 모두 보존."""
        model = SPSSGridModel()
        for i in range(10):
            _enter(model, i, 0, str(i + 1))
        df = model.get_dataframe()
        assert len(df) == 10
        for i in range(10):
            assert df.iloc[i, 0] == i + 1

    def test_setdata_returns_true_for_valid_entry(self):
        """유효한 입력 → setData returns True."""
        model = SPSSGridModel()
        result = _enter(model, 0, 0, "5")
        assert result is True

    def test_setdata_returns_true_for_first_column_creation(self):
        """첫 열 생성 포함 입력 → setData returns True."""
        model = SPSSGridModel()
        result = _enter(model, 5, 3, "99")
        assert result is True

    def test_variable_names_sequential(self):
        """변수명은 VAR00001, VAR00002, ... 순서."""
        model = SPSSGridModel()
        for col in range(5):
            _enter(model, 0, col, str(col + 1))
        names = list(model.get_variables().keys())
        for i, name in enumerate(names):
            assert name == f"VAR{i+1:05d}"


# ──────────────────────────────────────────────────────────────
# 5. 복합 입력 시나리오 (SPSS 실제 사용 패턴)
# ──────────────────────────────────────────────────────────────

class TestComplexInputScenarios:
    """SPSS 실제 사용 패턴 복합 테스트."""

    def test_survey_data_pattern(self):
        """설문 데이터 패턴: ID(정수) + 응답(리커트1-5) + 성별(M/F).

        SPSS 29 동작:
          ID → INTEGER + SCALE
          score → INTEGER + SCALE (리커트 5점이라도)
          gender → STRING + NOMINAL
        """
        model = SPSSGridModel()
        ids     = [1, 2, 3, 4, 5]
        scores  = [3, 4, 5, 2, 4]
        genders = ["M", "F", "M", "F", "M"]

        for row in range(5):
            _enter(model, row, 0, str(ids[row]))
            _enter(model, row, 1, str(scores[row]))
            _enter(model, row, 2, genders[row])

        vars_ = model.get_variables()
        assert vars_["VAR00001"].storage_type == StorageType.INTEGER
        assert vars_["VAR00001"].measure == MeasureType.SCALE
        assert vars_["VAR00002"].storage_type == StorageType.INTEGER
        assert vars_["VAR00002"].measure == MeasureType.SCALE
        assert vars_["VAR00003"].storage_type == StorageType.STRING
        assert vars_["VAR00003"].measure == MeasureType.NOMINAL

    def test_experimental_data_pattern(self):
        """실험 데이터: 처치(0/1) + 결과(연속) + 결측 포함.

        SPSS 29: 처치변수(0/1)라도 SCALE로 설정 (사용자가 NOMINAL로 변경 가능)
        """
        model = SPSSGridModel()
        treatments = ["0", "1", "0", "1", ".", "0"]
        outcomes   = ["12.5", "18.3", ".", "20.1", "15.6", "11.9"]

        for r, (t, o) in enumerate(zip(treatments, outcomes)):
            _enter(model, r, 0, t)
            _enter(model, r, 1, o)

        v = model.get_variables()
        assert v["VAR00001"].measure == MeasureType.SCALE
        assert v["VAR00002"].storage_type == StorageType.FLOAT
        assert v["VAR00002"].measure == MeasureType.SCALE

        df = model.get_dataframe()
        assert pd.isna(df.iloc[4, 0])  # 5번째 처치 결측
        assert pd.isna(df.iloc[2, 1])  # 3번째 결과 결측

    def test_all_same_value_is_scale(self):
        """동일 값만 반복 → SCALE."""
        model = SPSSGridModel()
        for i in range(10):
            _enter(model, i, 0, "5")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_large_integer_values_is_scale(self):
        """큰 정수 → INTEGER + SCALE."""
        model = SPSSGridModel()
        for v in ["1000000", "2000000", "3000000"]:
            _enter(model, 0, 0, v)
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.SCALE

    def test_decimal_precision_tracked(self):
        """소수점 자릿수 추적 → decimals 속성 설정."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1.123")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.FLOAT
        assert var.decimals >= 3

    def test_overwrite_value_keeps_type(self):
        """값 덮어쓰기 → 변수 유형 유지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")
        _enter(model, 1, 0, "20")
        _enter(model, 0, 0, "99")  # 덮어쓰기
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 99

    def test_spss_missing_dot_in_string_column(self):
        """문자 열에 "." → SPSS 결측 처리."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "Korea")
        _enter(model, 1, 0, ".")
        _enter(model, 2, 0, "Japan")
        df = model.get_dataframe()
        # "." in string context: parsed as pd.NA
        assert pd.isna(df.iloc[1, 0])

    def test_numeric_then_missing_then_numeric(self):
        """숫자-결측-숫자 패턴 → 전체 SCALE."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")
        _enter(model, 1, 0, ".")
        _enter(model, 2, 0, "30")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 10
        assert pd.isna(df.iloc[1, 0])
        assert df.iloc[2, 0] == 30

    def test_add_column_after_data_entry(self):
        """데이터 입력 후 열 추가 → 기존 데이터 보존."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "100")
        _enter(model, 1, 0, "200")
        model.add_column("extra")
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 100
        assert df.iloc[1, 0] == 200
        assert "extra" in df.columns

    def test_column_count_grows_beyond_default(self):
        """DEFAULT_COLS(100) 초과 시 열 수 증가 처리."""
        model = SPSSGridModel()
        # 90번째 열부터 DEFAULT 넘어가는 시점 테스트
        for col in range(95):
            _enter(model, 0, col, str(col))
        assert model.columnCount() >= 95
        vars_ = model.get_variables()
        assert len(vars_) == 95


# ──────────────────────────────────────────────────────────────
# 6. 사용자 설정 측정 척도 보존 테스트
# ──────────────────────────────────────────────────────────────

class TestUserDefinedMeasurePreservation:
    """사용자가 Variable View에서 설정한 측정 척도가 재입력 시 유지되는지 검증.

    SPSS 29 동작:
      - 자동 감지: 첫 입력 시 숫자→SCALE, 문자→NOMINAL
      - 사용자 설정 후에는 추가 데이터 입력으로 덮어씌워지지 않음
    """

    def test_user_set_ordinal_preserved_on_more_data(self):
        """사용자가 ORDINAL로 설정한 후 추가 입력 → ORDINAL 유지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1")  # 첫 입력: 자동 SCALE
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

        # 사용자가 Variable View에서 ORDINAL로 변경
        var.measure = MeasureType.ORDINAL

        # 추가 데이터 입력 → ORDINAL 유지되어야 함
        _enter(model, 1, 0, "2")
        _enter(model, 2, 0, "3")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.ORDINAL, (
            "사용자 설정 ORDINAL이 추가 입력으로 덮어씌워지면 안 됨"
        )

    def test_user_set_nominal_on_numeric_preserved(self):
        """사용자가 숫자 변수를 NOMINAL로 설정 → 추가 입력에도 NOMINAL 유지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "0")
        _enter(model, 1, 0, "1")

        # 사용자가 0/1 변수를 NOMINAL로 수동 설정
        var = model.get_variables()["VAR00001"]
        var.measure = MeasureType.NOMINAL

        # 추가 입력
        _enter(model, 2, 0, "0")
        _enter(model, 3, 0, "1")

        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.NOMINAL, (
            "사용자 설정 NOMINAL이 추가 입력으로 SCALE로 바뀌면 안 됨"
        )

    def test_user_set_binary_preserved(self):
        """사용자가 BINARY로 설정 → 추가 입력에도 유지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")
        var = model.get_variables()["VAR00001"]
        var.measure = MeasureType.BINARY

        _enter(model, 1, 0, "20")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.BINARY

    def test_first_entry_still_auto_detects(self):
        """첫 번째 입력은 여전히 자동 감지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "42")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.SCALE

    def test_new_string_var_auto_nominal(self):
        """문자형 변수는 첫 입력에 NOMINAL 자동 감지."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "Apple")
        var = model.get_variables()["VAR00001"]
        assert var.storage_type == StorageType.STRING
        assert var.measure == MeasureType.NOMINAL

        # 사용자가 BINARY로 설정
        var.measure = MeasureType.BINARY

        # 추가 입력 → BINARY 유지 (재감지 없음)
        _enter(model, 1, 0, "Banana")
        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.BINARY


# ──────────────────────────────────────────────────────────────
# 7. 네비게이션 동작 SPSS 호환 테스트
# ──────────────────────────────────────────────────────────────

class TestNavigationDoesNotCreateVariables:
    """네비게이션(Tab/화살표)만으로 변수가 생성되면 안 됨 (SPSS 호환)."""

    def test_navigate_beyond_columns_no_var_created(self):
        """실제 데이터 열 범위를 벗어나도 네비게이션만으로 변수 생성 없음."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1")  # VAR00001 생성
        assert len(model.get_variables()) == 1

        # 데이터 없이 col=5에 접근 (가상 열)
        # _navigate 코드는 _create_variable_at_col을 호출하지 않음
        # 모델 레벨에서 확인: setData 없이 get_variables 체크
        vars_before = len(model.get_variables())

        # 가상 셀 index 접근 (데이터 없음)
        idx = model.index(0, 5)
        assert idx.isValid()
        assert len(model.get_variables()) == vars_before  # 변수 생성 없음

    def test_only_setdata_creates_variable(self):
        """setData 호출 시에만 변수가 생성됨."""
        model = SPSSGridModel()
        assert len(model.get_variables()) == 0

        # 데이터 입력 → 변수 생성
        _enter(model, 0, 3, "99")  # col 3에 직접 입력
        assert len(model.get_variables()) == 4  # VAR00001 ~ VAR00004

    def test_empty_virtual_cell_data_returns_empty(self):
        """가상 열의 빈 셀은 EditRole로 빈 문자열 반환."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1")  # 1개 열만 있는 상태

        # col 5 (가상 열) - 데이터 없음
        idx = model.index(0, 5)
        val = model.data(idx, Qt.ItemDataRole.EditRole)
        assert val == "" or val is None


# ──────────────────────────────────────────────────────────────
# 8. 숫자형 변수 입력 유효성 검증 (SPSS 호환)
# ──────────────────────────────────────────────────────────────

class TestNumericColumnStringRejection:
    """숫자형 변수에 문자 입력 시 거부 — SPSS 동작 검증."""

    def test_string_rejected_in_integer_column(self):
        """정수형 컬럼에 문자열 입력 → setData False 반환."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "5")  # INTEGER로 확정

        result = model.setData(model.index(1, 0), "abc", Qt.ItemDataRole.EditRole)
        assert result is False, "숫자형 열에 문자 입력은 거부되어야 함"

        df = model.get_dataframe()
        assert len(df) == 1, "거부된 입력으로 행이 추가되면 안 됨"

    def test_string_rejected_in_float_column(self):
        """소수형 컬럼에 문자열 입력 → setData False 반환."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "3.14")  # FLOAT으로 확정

        result = model.setData(model.index(1, 0), "hello", Qt.ItemDataRole.EditRole)
        assert result is False

    def test_dot_allowed_in_numeric_column(self):
        """숫자형 열에 "." (SPSS 결측 기호) → 허용."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")

        result = model.setData(model.index(1, 0), ".", Qt.ItemDataRole.EditRole)
        assert result is True, '"." 는 결측 기호이므로 허용되어야 함'

        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0])

    def test_empty_string_allowed_in_numeric_column(self):
        """숫자형 열에 빈 문자열 → 허용 (결측 처리)."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "10")

        result = model.setData(model.index(1, 0), "", Qt.ItemDataRole.EditRole)
        assert result is True

    def test_first_entry_to_new_column_string_allowed(self):
        """새 열에는 타입 미확정이므로 문자 입력 허용."""
        model = SPSSGridModel()
        result = model.setData(model.index(0, 0), "hello", Qt.ItemDataRole.EditRole)
        assert result is True, "신규 변수에는 문자 입력 허용"

    def test_numeric_string_allowed_in_integer_column(self):
        """정수형 열에 숫자 문자열 → 허용."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "5")
        result = model.setData(model.index(1, 0), "10", Qt.ItemDataRole.EditRole)
        assert result is True

    def test_numeric_string_allowed_in_float_column(self):
        """소수형 열에 숫자 문자열 → 허용."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1.5")
        result = model.setData(model.index(1, 0), "2.5", Qt.ItemDataRole.EditRole)
        assert result is True


# ──────────────────────────────────────────────────────────────
# 9. mark_measure_initialized API 테스트
# ──────────────────────────────────────────────────────────────

class TestMarkMeasureInitialized:
    """Variable View에서 측정 척도 변경 후 모델에 초기화 완료 표시."""

    def test_mark_before_first_data_prevents_overwrite(self):
        """데이터 입력 전 mark_measure_initialized 호출 → 자동 감지 스킵."""
        model = SPSSGridModel()

        # 변수를 미리 생성 (add_column으로 Variable View 역할 시뮬레이션)
        model.add_column("custom_var")
        var = model.get_variables()["custom_var"]
        var.measure = MeasureType.ORDINAL

        # 초기화 완료 표시 (Variable View에서 measure 설정했음을 알림)
        model.mark_measure_initialized("custom_var")

        # 데이터 입력
        model.setData(model.index(0, 0), "3", Qt.ItemDataRole.EditRole)

        var = model.get_variables()["custom_var"]
        assert var.measure == MeasureType.ORDINAL, (
            "mark_measure_initialized 후 자동 감지가 사용자 설정을 덮어써선 안 됨"
        )

    def test_without_mark_auto_detects(self):
        """mark 없이 데이터 입력 → 자동 감지 작동."""
        model = SPSSGridModel()
        model.add_column("auto_var")

        # mark 없이 바로 데이터 입력 → 자동 감지
        model.setData(model.index(0, 0), "42", Qt.ItemDataRole.EditRole)

        var = model.get_variables()["auto_var"]
        assert var.measure == MeasureType.SCALE, "mark 없이 입력 시 SCALE 자동 감지"


# ──────────────────────────────────────────────────────────────
# 10. sort_by_column 테스트
# ──────────────────────────────────────────────────────────────

class TestSortByColumn:
    """sort_by_column: beginResetModel 없이 정렬 후 데이터 정확성 검증."""

    def test_sort_ascending(self):
        """오름차순 정렬."""
        model = SPSSGridModel()
        for i, v in enumerate(["30", "10", "20"]):
            _enter(model, i, 0, v)

        model.sort_by_column(0, ascending=True)
        df = model.get_dataframe()
        assert list(df.iloc[:, 0]) == [10, 20, 30]

    def test_sort_descending(self):
        """내림차순 정렬."""
        model = SPSSGridModel()
        for i, v in enumerate(["30", "10", "20"]):
            _enter(model, i, 0, v)

        model.sort_by_column(0, ascending=False)
        df = model.get_dataframe()
        assert list(df.iloc[:, 0]) == [30, 20, 10]

    def test_sort_out_of_bounds_col(self):
        """존재하지 않는 열 정렬 → 조용히 무시."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "5")
        model.sort_by_column(99, ascending=True)  # 오류 없이 종료
        df = model.get_dataframe()
        assert df.iloc[0, 0] == 5


# ──────────────────────────────────────────────────────────────
# 11. ORDINAL + FLOAT 호환성 검증
# ──────────────────────────────────────────────────────────────

class TestOrdinalFloatCompatibility:
    """SPSS 호환: ORDINAL 변수에 소수값 허용."""

    def test_ordinal_with_float_data_no_error(self):
        """ORDINAL 설정 변수에 소수 입력 → 오류 없이 저장."""
        model = SPSSGridModel()
        _enter(model, 0, 0, "1")
        var = model.get_variables()["VAR00001"]
        var.measure = MeasureType.ORDINAL

        # 소수 입력 — storage_type이 FLOAT으로 바뀌어도 ORDINAL 유지
        result = model.setData(model.index(1, 0), "2.5", Qt.ItemDataRole.EditRole)
        assert result is True

        var = model.get_variables()["VAR00001"]
        assert var.measure == MeasureType.ORDINAL
        assert var.storage_type == StorageType.FLOAT

    def test_ordinal_float_validation_passes(self):
        """ORDINAL + FLOAT 조합이 validate_measure_storage_compatibility 통과."""
        from nuristat.core.validation import validate_measure_storage_compatibility
        result = validate_measure_storage_compatibility(MeasureType.ORDINAL, StorageType.FLOAT)
        assert result is True
