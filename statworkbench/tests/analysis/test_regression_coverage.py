"""regression.py 커버리지 보강 테스트.

미커버 라인:
  38      : string missing_policy → MissingPolicy 변환
  67-68   : 유효 관측치 없음 → 경고 + return
  143-153 : ANOVA 테이블 rows 생성 루프 (sm.OLS 수식 없이는 except 처리됨)
  164     : conf_int DataFrame .iloc 경로
  219-220 : VIF 예외 → N/A row
  313     : predictor not in df.columns → continue (_build_design_matrix 직접 호출)
  332     : ref_col not in dummies.columns → iloc[:, 1:]
  341     : len(unique_vals) > 20 → 더미 미생성
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.regression import run_analysis, _build_design_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_dataset():
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 * x1 + 1.5 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    ds = Dataset(df, name="RegData")
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["x1"].measure = MeasureType.SCALE
    ds.variables["x2"].measure = MeasureType.SCALE
    return ds


# ---------------------------------------------------------------------------
# Line 38: string missing_policy
# ---------------------------------------------------------------------------

class TestStringMissingPolicy:

    def test_string_listwise(self, reg_dataset):
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
            "missing_policy": "listwise",
        }
        result = run_analysis(reg_dataset, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Lines 67-68: 유효 관측치 없음
# ---------------------------------------------------------------------------

class TestNoValidObservations:

    def test_all_nan_returns_warning(self):
        df = pd.DataFrame({"y": [np.nan, np.nan, np.nan], "x1": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "EmptyData")
        ds.variables["y"].measure = MeasureType.SCALE
        ds.variables["x1"].measure = MeasureType.SCALE
        spec = {"variables": {"dependent": "y", "independent": ["x1"]}}
        result = run_analysis(ds, spec)
        assert any("No valid" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 143-153: ANOVA 테이블 rows 루프 + line 164: conf_int.iloc
# ---------------------------------------------------------------------------

class TestANOVATableAndCoefficients:

    def test_coef_table_generated(self, reg_dataset):
        """계수표 생성 → conf_int.iloc 경로(164) 실행."""
        spec = {"variables": {"dependent": "y", "independent": ["x1"]}}
        result = run_analysis(reg_dataset, spec)
        coef_tbl = next((t for t in result.tables if "Coefficients" in t.title), None)
        assert coef_tbl is not None


# ---------------------------------------------------------------------------
# Lines 219-220: VIF 예외 → N/A
# ---------------------------------------------------------------------------

class TestVIFException:

    def test_vif_exception_produces_na_row(self, reg_dataset):
        """variance_inflation_factor 예외 → 'N/A' row."""
        spec = {"variables": {"dependent": "y", "independent": ["x1", "x2"]}}
        with patch(
            "statworkbench.analysis.regression.variance_inflation_factor",
            side_effect=Exception("vif fail"),
        ):
            result = run_analysis(reg_dataset, spec)
        vif_tbl = next((t for t in result.tables if "VIF" in t.title), None)
        assert vif_tbl is not None
        assert "N/A" in vif_tbl.dataframe["VIF"].values


# ---------------------------------------------------------------------------
# Line 313: predictor not in df.columns → continue (_build_design_matrix 직접)
# ---------------------------------------------------------------------------

class TestPredictorNotInDF:

    def test_missing_predictor_skipped(self, reg_dataset):
        """_build_design_matrix: ghost 컬럼 → continue(313) 후 남은 predictor로 진행."""
        df = reg_dataset.data[["x1"]].copy()  # x2 없음
        X_df, dummy_info = _build_design_matrix(df, ["x1", "x2"], reg_dataset, {})
        # x2는 df에 없어서 continue → x1만 포함
        assert "x1" in X_df.columns
        assert "x2" not in X_df.columns


# ---------------------------------------------------------------------------
# Line 332: ref_col not in dummies.columns → iloc[:, 1:]
# ---------------------------------------------------------------------------

class TestCategoricalRefColMissing:

    def test_nonexistent_ref_col_uses_first_dummy(self):
        """reference_category가 실제 데이터에 없음 → iloc[:, 1:] 경로(332)."""
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame({
            "y": rng.normal(0, 1, n),
            "grp": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        ds = Dataset(df, "CatData")
        ds.variables["y"].measure = MeasureType.SCALE
        ds.variables["grp"].measure = MeasureType.NOMINAL
        spec = {
            "variables": {"dependent": "y", "independent": ["grp"]},
            "options": {"reference_category": {"grp": "Z"}},  # Z는 존재하지 않음
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Line 341: len(unique_vals) > 20 → dummy 미생성, 수치 그대로 사용
# ---------------------------------------------------------------------------

class TestManyCategories:

    def test_high_cardinality_categorical_as_numeric(self):
        """unique값 > 20인 NOMINAL → 더미 생성 없이 원본 사용(341)."""
        rng = np.random.default_rng(0)
        n = 100
        # 25개 범주
        cats = [str(i) for i in range(25)]
        df = pd.DataFrame({
            "y": rng.normal(0, 1, n),
            "cat": rng.choice(cats, n),
        })
        ds = Dataset(df, "HighCat")
        ds.variables["y"].measure = MeasureType.SCALE
        ds.variables["cat"].measure = MeasureType.NOMINAL
        spec = {"variables": {"dependent": "y", "independent": ["cat"]}}
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullRegression:

    def test_full_run_tables_count(self, reg_dataset):
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
        }
        result = run_analysis(reg_dataset, spec)
        # Case Summary, Model Summary, ANOVA, Coefficients, VIF, DW, Residuals
        assert len(result.tables) >= 5
