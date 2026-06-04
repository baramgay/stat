"""logistic_regression.py 커버리지 보강 테스트.

미커버 라인:
  24-25   : sklearn ImportError (모듈 재로드 없이 불가 — 생략)
  77-78   : 유효 관측치 없음 → 경고 + return
  130     : len(unique_vals) > 20 NOMINAL → 더미 미생성
  155-157 : binary Logit 적합 실패 → 경고 + return
  244-245 : cm.shape != (2,2) → cm_df 분기
  261-265 : sklearn 미설치 → _manual_classification_table 호출
  314-319 : manual AUC 계산 (pos > 0 and neg > 0)
  364-365 : Hosmer-Lemeshow 예외 → 경고
  382-384 : multinomial 모델 적합 실패 → 경고 + return
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.logistic_regression import run_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_dataset():
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"y": y, "x": x})
    ds = Dataset(df, "BinData")
    ds.variables["y"].measure = MeasureType.BINARY
    ds.variables["x"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def multinomial_dataset():
    rng = np.random.default_rng(7)
    n = 90
    x = rng.normal(0, 1, n)
    y = pd.cut(x, bins=3, labels=["A", "B", "C"])
    df = pd.DataFrame({"y": y.astype(str), "x": x})
    ds = Dataset(df, "MultiData")
    ds.variables["y"].measure = MeasureType.NOMINAL
    ds.variables["x"].measure = MeasureType.SCALE
    return ds


# ---------------------------------------------------------------------------
# Lines 77-78: 유효 관측치 없음
# ---------------------------------------------------------------------------

class TestNoValidObservations:

    def test_all_nan_warns(self):
        df = pd.DataFrame({"y": [np.nan, np.nan, np.nan], "x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "Empty")
        ds.variables["y"].measure = MeasureType.BINARY
        ds.variables["x"].measure = MeasureType.SCALE
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        result = run_analysis(ds, spec)
        assert any("유효한 관측치" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Line 130: len(unique_vals) > 20 NOMINAL predictor
# ---------------------------------------------------------------------------

class TestHighCardinalityPredictor:

    def test_high_cardinality_nominal_used_as_numeric(self):
        """NOMINAL predictor에 25개 범주 → 더미 미생성, 수치 변환(130)."""
        rng = np.random.default_rng(0)
        n = 100
        cats = list(range(25))
        df = pd.DataFrame({
            "y": (rng.normal(0, 1, n) > 0).astype(int),
            "cat": rng.choice(cats, n),
        })
        ds = Dataset(df, "HighCat")
        ds.variables["y"].measure = MeasureType.BINARY
        ds.variables["cat"].measure = MeasureType.NOMINAL
        spec = {"variables": {"dependent": "y", "predictors": ["cat"]}}
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Lines 155-157: binary Logit 적합 실패
# ---------------------------------------------------------------------------

class TestBinaryLogitFitFailure:

    def test_logit_fit_exception_adds_warning(self, binary_dataset):
        """sm.Logit(...).fit() 예외 → 경고 추가 후 return."""
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        mock_fitted = MagicMock()
        mock_fitted.fit.side_effect = Exception("logit fail")

        with patch("nuristat.analysis.logistic_regression.sm.Logit",
                   return_value=mock_fitted):
            result = run_analysis(binary_dataset, spec)

        assert any("모델 적합 실패" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 261-265, 314-319: sklearn 미설치 → manual classification + AUC
# ---------------------------------------------------------------------------

class TestSklearnNotAvailable:

    def test_manual_classification_and_auc(self, binary_dataset):
        """_SKLEARN_AVAILABLE=False → _manual_classification_table + 수동 AUC."""
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        with patch("nuristat.analysis.logistic_regression._SKLEARN_AVAILABLE", False):
            result = run_analysis(binary_dataset, spec)
        # 분류표 생성 확인
        cls_tbl = next((t for t in result.tables if "분류표" in t.title), None)
        assert cls_tbl is not None


# ---------------------------------------------------------------------------
# Lines 364-365: Hosmer-Lemeshow 예외
# ---------------------------------------------------------------------------

class TestHosmerLemeshowException:

    def test_hl_exception_adds_warning(self, binary_dataset):
        """pd.qcut 예외 → Hosmer-Lemeshow 경고 추가."""
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        with patch("nuristat.analysis.logistic_regression.pd.qcut",
                   side_effect=Exception("qcut fail")):
            result = run_analysis(binary_dataset, spec)
        assert any("Hosmer-Lemeshow" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 382-384: multinomial 모델 적합 실패
# ---------------------------------------------------------------------------

class TestMultinomialFitFailure:

    def test_mnlogit_fit_exception_adds_warning(self, multinomial_dataset):
        """MNLogit fit 예외 → 경고 추가 후 return."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x"]},
            "options": {"method": "multinomial"},
        }
        mock_fitted = MagicMock()
        mock_fitted.fit.side_effect = Exception("mnlogit fail")

        with patch("nuristat.analysis.logistic_regression.sm.MNLogit",
                   return_value=mock_fitted):
            result = run_analysis(multinomial_dataset, spec)
        assert any("다항 로지스틱 모델 적합 실패" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullLogisticRegression:

    def test_binary_full(self, binary_dataset):
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.tables) >= 3

    def test_multinomial_full(self, multinomial_dataset):
        spec = {
            "variables": {"dependent": "y", "predictors": ["x"]},
            "options": {"method": "multinomial"},
        }
        result = run_analysis(multinomial_dataset, spec)
        assert len(result.tables) >= 2
