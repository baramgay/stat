"""core/variable.py 커버리지 보강 테스트.

미커버 라인:
  194  : is_numeric — StorageType.FLOAT인 경우 True 반환
  202  : is_categorical — MeasureType.BINARY인 경우 True 반환
  213-232: validate_value — allowed_min/max 검사, NaN 처리, 형변환 실패 처리
"""

from __future__ import annotations

import math

import pytest

from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.core.variable import VariableMeta


@pytest.fixture
def float_var():
    v = VariableMeta(name="weight")
    v.storage_type = StorageType.FLOAT
    return v


@pytest.fixture
def binary_var():
    v = VariableMeta(name="sex")
    v.measure = MeasureType.BINARY
    return v


@pytest.fixture
def bounded_var():
    v = VariableMeta(name="score")
    v.storage_type = StorageType.FLOAT
    v.allowed_min = 0.0
    v.allowed_max = 100.0
    return v


# ──────────────────────────────────────────────────────────────
# is_numeric — FLOAT 분기 (line 194)
# ──────────────────────────────────────────────────────────────

class TestIsNumericFloat:

    def test_float_storage_is_numeric(self, float_var):
        """StorageType.FLOAT → is_numeric == True."""
        assert float_var.is_numeric is True

    def test_string_storage_not_numeric(self):
        v = VariableMeta(name="name")
        v.storage_type = StorageType.STRING
        assert v.is_numeric is False


# ──────────────────────────────────────────────────────────────
# is_categorical — BINARY 분기 (line 202)
# ──────────────────────────────────────────────────────────────

class TestIsCategoricalBinary:

    def test_binary_is_categorical(self, binary_var):
        """MeasureType.BINARY → is_categorical == True."""
        assert binary_var.is_categorical is True

    def test_scale_not_categorical(self):
        v = VariableMeta(name="bp")
        v.measure = MeasureType.SCALE
        assert v.is_categorical is False

    def test_nominal_is_categorical(self):
        v = VariableMeta(name="group")
        v.measure = MeasureType.NOMINAL
        assert v.is_categorical is True

    def test_ordinal_is_categorical(self):
        v = VariableMeta(name="rank")
        v.measure = MeasureType.ORDINAL
        assert v.is_categorical is True


# ──────────────────────────────────────────────────────────────
# validate_value — 경계 검사 (lines 213-232)
# ──────────────────────────────────────────────────────────────

class TestValidateValue:

    def test_none_returns_no_warnings(self, bounded_var):
        """None → 빈 경고 목록."""
        assert bounded_var.validate_value(None) == []

    def test_nan_returns_no_warnings(self, bounded_var):
        """NaN float → 빈 경고 목록 (결측으로 처리)."""
        assert bounded_var.validate_value(float("nan")) == []

    def test_value_within_range_no_warnings(self, bounded_var):
        """유효 범위 내 값 → 경고 없음."""
        warnings = bounded_var.validate_value(50.0)
        assert warnings == []

    def test_value_below_min_warns(self, bounded_var):
        """allowed_min 미만 값 → 경고 포함."""
        warnings = bounded_var.validate_value(-1.0)
        assert len(warnings) == 1
        assert "below minimum" in warnings[0]

    def test_value_above_max_warns(self, bounded_var):
        """allowed_max 초과 값 → 경고 포함."""
        warnings = bounded_var.validate_value(101.0)
        assert len(warnings) == 1
        assert "above maximum" in warnings[0]

    def test_value_at_min_boundary_no_warning(self, bounded_var):
        """경계값(min) → 경고 없음."""
        warnings = bounded_var.validate_value(0.0)
        assert warnings == []

    def test_value_at_max_boundary_no_warning(self, bounded_var):
        """경계값(max) → 경고 없음."""
        warnings = bounded_var.validate_value(100.0)
        assert warnings == []

    def test_both_violations_two_warnings(self):
        """min과 max 모두 위반 불가 (단일 값이라 동시 위반 없음) — max만 위반."""
        v = VariableMeta(name="x")
        v.allowed_min = 0.0
        v.allowed_max = 10.0
        warnings = v.validate_value(15.0)
        assert len(warnings) == 1
        assert "above maximum" in warnings[0]

    def test_non_numeric_string_skips_min_check(self):
        """문자열 값은 float 변환 실패 → min 경고 건너뜀 (ValueError 처리)."""
        v = VariableMeta(name="cat")
        v.allowed_min = 0.0
        warnings = v.validate_value("abc")
        assert warnings == []

    def test_non_numeric_string_skips_max_check(self):
        """문자열 값은 float 변환 실패 → max 경고 건너뜀 (ValueError 처리)."""
        v = VariableMeta(name="cat")
        v.allowed_max = 100.0
        warnings = v.validate_value("abc")
        assert warnings == []

    def test_no_bounds_set_no_warnings(self):
        """min/max 미설정 → 어떤 값도 경고 없음."""
        v = VariableMeta(name="free")
        assert v.validate_value(9999.9) == []
        assert v.validate_value(-9999.9) == []

    def test_warning_contains_value_and_bound(self, bounded_var):
        """경고 메시지에 값과 경계값이 포함됨."""
        warnings = bounded_var.validate_value(200.0)
        assert "200" in warnings[0]
        assert "100" in warnings[0]
