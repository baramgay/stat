"""explore.py 미커버 라인 보강 테스트.

대상 라인:
  91-92   : n==1 → se_mean=0.0, ci_lower=ci_upper=mean
  151     : _val(None) → return float("nan")
  330     : var not in df_full.columns → continue
  398     : else → missing_policy = missing_policy_str (enum 직접 전달)
  418     : factor 변수 없음 → ValueError
  457     : 그룹 내 arr 비어 있음 → warnings.append
  518     : norm_rows 비어 있음 (factor 그룹 없음) → norm_df=pd.DataFrame()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.explore import (
    _compute_explore_stats,
    _build_descriptives_rows,
    _build_case_processing_summary,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Lines 91-92: n==1 → se_mean=0.0, ci_lower=ci_upper=mean
# ---------------------------------------------------------------------------

class TestComputeExploreStatsNOne:

    def test_n_one_gives_zero_se(self):
        """n=1 → se_mean=0.0, CI 상·하한 = mean (91-92)."""
        arr = np.array([42.0])
        s = _compute_explore_stats(arr)
        assert s["se_mean"] == 0.0
        assert s["ci_lower"] == 42.0
        assert s["ci_upper"] == 42.0


# ---------------------------------------------------------------------------
# Line 151: _val(None) → return float("nan")
# ---------------------------------------------------------------------------

class TestBuildDescriptivesRowsNone:

    def test_none_value_returns_nan(self):
        """s["mean"]=None → _val(None) → float("nan") (151)."""
        nan = float("nan")
        s = {
            "mean": None,  # triggers line 151
            "ci_lower": nan, "ci_upper": nan,
            "trimmed_mean": nan, "median": nan,
            "sd": nan, "variance": nan,
            "min": nan, "max": nan, "range": nan, "iqr": nan,
            "skewness": nan, "skewness_se": nan,
            "kurtosis": nan, "kurtosis_se": nan,
        }
        rows = _build_descriptives_rows("x", s)
        mean_row = next(r for r in rows if r["통계량"] == "평균")
        import math
        assert math.isnan(mean_row["값"])


# ---------------------------------------------------------------------------
# Line 330: var not in df_full.columns → continue
# ---------------------------------------------------------------------------

class TestBuildCaseProcessingSummaryContinue:

    def test_var_not_in_df_full_skipped(self):
        """df_full에 없는 변수 → continue(330)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "Test")
        df_full = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        # "ghost" 변수는 df_full에 없음 → line 330
        tbl = _build_case_processing_summary(
            ds, target_vars=["x", "ghost"], df=df_full, df_full=df_full
        )
        # "ghost"는 건너뛰고 "x" 행만 생성
        assert len(tbl.dataframe) == 1
        assert tbl.dataframe["변수"].iloc[0] == "x"


# ---------------------------------------------------------------------------
# Line 398: else → missing_policy = missing_policy_str (enum 직접 전달)
# ---------------------------------------------------------------------------

class TestMissingPolicyStringConversion:

    def test_string_missing_policy_converted_to_enum(self):
        """missing_policy를 문자열로 전달 → MissingPolicy() 변환 실행(398)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"]},
            "missing_policy": "listwise",  # 문자열 → line 398 if 브랜치
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Line 418: factor 변수 없음 → ValueError
# ---------------------------------------------------------------------------

class TestFactorVarNotFound:

    def test_missing_factor_var_raises(self):
        """존재하지 않는 factor_var → warnings 반환."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"], "factor": "no_such_col"},
        }
        result = run_analysis(ds, spec)
        assert result.warnings, "존재하지 않는 factor 변수 → warnings 반환 기대"


# ---------------------------------------------------------------------------
# Line 457: 그룹 내 arr 비어 있음 → warnings.append
# ---------------------------------------------------------------------------

class TestGroupEmptyArrWarning:

    def test_group_with_all_nan_dep_var_adds_warning(self):
        """그룹 A의 dep_var 전부 NaN → warnings 추가(457)."""
        df = pd.DataFrame({
            "x":   [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
            "grp": ["A",    "A",    "A",    "B", "B", "B"],
        })
        ds = Dataset(df, "EmptyGroup")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["grp"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"target": ["x"], "factor": "grp"},
        }
        result = run_analysis(ds, spec)
        assert any("유효한 데이터 없음" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Line 518: factor 그룹이 전혀 없음 (factor_var 전체 NaN)
#           → norm_rows=[] → norm_df=pd.DataFrame()
# ---------------------------------------------------------------------------

class TestNormRowsEmptyWhenNoGroups:

    def test_all_nan_factor_gives_empty_norm_df(self):
        """factor_var 전부 NaN → groups=[] → norm_rows=[] → line 518."""
        df = pd.DataFrame({
            "x":   [1.0, 2.0, 3.0],
            "grp": [np.nan, np.nan, np.nan],
        })
        ds = Dataset(df, "NoGroups")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["grp"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"target": ["x"], "factor": "grp"},
            "options": {"normality": True},
        }
        result = run_analysis(ds, spec)
        norm_table = next(
            (t for t in result.tables if "Normality" in t.title), None
        )
        assert norm_table is not None
        # norm_rows가 비어 있어서 빈 DataFrame 이어야 함
        assert norm_table.dataframe.empty
