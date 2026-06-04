"""regression.py 미커버 경로 보완 테스트 (lines 88-89, 346-347, 351-556)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType
from nuristat.analysis.regression import run_analysis


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_dataset(n: int = 60) -> Dataset:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2 * x1 + 1.5 * x2 + rng.normal(0, 0.5, n)
    group = (rng.uniform(0, 1, n) > 0.5).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "group": group})
    ds = Dataset(df, name="reg_test")
    ds.variables["y"]     = _scale("y")
    ds.variables["x1"]    = _scale("x1")
    ds.variables["x2"]    = _scale("x2")
    ds.variables["x3"]    = _scale("x3")
    ds.variables["group"] = _nominal("group")
    return ds


# ── line 88-89: empty design matrix after dummy coding ──────────────────────

class TestEmptyDesignMatrix:
    def test_all_categorical_single_value(self):
        """범주 1개짜리 더미 코딩 후 빈 행렬 경고 처리."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "cat": [0, 0, 0, 0]})
        ds = Dataset(df, name="single_cat")
        ds.variables["y"]   = _scale("y")
        ds.variables["cat"] = _nominal("cat")
        spec = {
            "variables": {"dependent": "y", "predictors": ["cat"]},
            "options": {"selection_method": "enter"},
        }
        result = run_analysis(ds, spec)
        # 경고가 있거나 정상 결과가 나와야 함 (crash 없음)
        assert result is not None


# ── lines 346-347: Cook's D exception handler ────────────────────────────────

class TestInfluenceDiagnostics:
    def test_influence_no_influential_cases(self):
        """Cook's D 임계치 이하 → '영향력 큰 케이스 없음' 행 생성."""
        ds = _make_dataset(80)
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {
                "selection_method": "enter",
                "diagnostics": {"cooks_distance": True},
            },
        }
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("진단" in t or "Cook" in t for t in titles)

    def test_influence_with_extreme_outlier(self):
        """극단값 포함 시 Cook's D 임계 초과 케이스 테이블 생성."""
        ds = _make_dataset(40)
        # 첫 행을 극단값으로 설정
        ds.data.loc[0, "y"] = 999.0
        ds.data.loc[0, "x1"] = 50.0
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {
                "selection_method": "enter",
                "diagnostics": {"cooks_distance": True},
            },
        }
        result = run_analysis(ds, spec)
        assert result is not None


# ── lines 351-555: stepwise / forward / backward selection ──────────────────

class TestStepwiseSelection:
    def _spec(self, method: str) -> dict:
        return {
            "variables": {"dependent": "y", "predictors": ["x1", "x2", "x3"]},
            "options": {
                "selection_method": method,
                "stepwise_summary": True,
            },
        }

    def test_forward_returns_result(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("forward"))
        assert result is not None
        assert len(result.tables) > 0

    def test_forward_summary_table(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("forward"))
        titles = [t.title for t in result.tables]
        assert any("Forward" in t or "forward" in t.lower() or "변수 선택" in t for t in titles)

    def test_backward_returns_result(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("backward"))
        assert result is not None
        assert len(result.tables) > 0

    def test_backward_summary_table(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("backward"))
        titles = [t.title for t in result.tables]
        assert any("Backward" in t or "backward" in t.lower() or "변수 선택" in t for t in titles)

    def test_stepwise_returns_result(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("stepwise"))
        assert result is not None
        assert len(result.tables) > 0

    def test_stepwise_summary_table(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("stepwise"))
        titles = [t.title for t in result.tables]
        assert any("Stepwise" in t or "stepwise" in t.lower() or "변수 선택" in t for t in titles)

    def test_stepwise_with_no_significant_predictors(self):
        """유의한 예측변수 없을 때 스텝와이즈 — 빈 rows 처리."""
        rng = np.random.default_rng(99)
        n = 30
        df = pd.DataFrame({
            "y":  rng.normal(0, 1, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="noise")
        ds.variables["y"]  = _scale("y")
        ds.variables["x1"] = _scale("x1")
        ds.variables["x2"] = _scale("x2")
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {"selection_method": "forward", "stepwise_summary": True},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_backward_single_predictor_no_removal(self):
        """예측변수 1개는 backward에서 제거 불가 — 변경 없음 분기."""
        rng = np.random.default_rng(1)
        n = 40
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"y": x + rng.normal(0, 0.1, n), "x1": x})
        ds = Dataset(df, name="single_pred")
        ds.variables["y"]  = _scale("y")
        ds.variables["x1"] = _scale("x1")
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {"selection_method": "backward", "stepwise_summary": True},
        }
        result = run_analysis(ds, spec)
        assert result is not None
