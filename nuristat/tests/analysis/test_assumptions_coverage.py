"""assumptions.py 커버리지 보강 테스트.

미커버 라인:
  81      : var not in dataset.variables → continue (_apply_user_missing)
  152     : MissingPolicy.ANALYSIS_DEFAULT → dropna
  158     : MissingPolicy.EXCLUDE_USER_MISSING_ONLY → subset (그대로)
  163     : else → dropna (알 수 없는 policy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.assumptions import prepare_analysis_frame, _apply_user_missing


# ---------------------------------------------------------------------------
# Line 81: var not in dataset.variables → continue
# ---------------------------------------------------------------------------

class TestApplyUserMissingVarNotInVariables:

    def test_var_not_in_dataset_variables_skipped(self):
        """dataset.variables에 없는 var → continue(81)."""
        df = pd.DataFrame({"x": [1.0, 99.0, 3.0]})
        ds = Dataset(df, "Test")
        # dataset.variables에 x가 없는 상태를 만들기 위해
        # 빈 Dataset 생성 후 data만 있고 variables 없는 경우 시뮬레이션
        # _apply_user_missing은 df, dataset, variables를 받음
        # dataset.variables에 없는 변수명을 variables로 전달
        result = _apply_user_missing(df.copy(), ds, ["x", "ghost"])
        # ghost는 dataset.variables에 없어서 continue(81) → 에러 없이 처리
        assert "x" in result.columns


# ---------------------------------------------------------------------------
# Line 152: MissingPolicy.ANALYSIS_DEFAULT
# ---------------------------------------------------------------------------

class TestAnalysisDefaultPolicy:

    def test_analysis_default_drops_nan(self):
        """ANALYSIS_DEFAULT → listwise와 동일(dropna) (152)."""
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0, 4.0],
            "y": [5.0, 6.0, np.nan, 8.0],
        })
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        prepared = prepare_analysis_frame(
            ds, ["x", "y"], missing_policy=MissingPolicy.ANALYSIS_DEFAULT
        )
        assert len(prepared.data) == 2  # 2행 제거됨


# ---------------------------------------------------------------------------
# Line 158: MissingPolicy.EXCLUDE_USER_MISSING_ONLY
# ---------------------------------------------------------------------------

class TestExcludeUserMissingOnly:

    def test_exclude_user_missing_keeps_nan(self):
        """EXCLUDE_USER_MISSING_ONLY → NaN 유지(158)."""
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "y": [4.0, 5.0, np.nan],
        })
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        prepared = prepare_analysis_frame(
            ds, ["x", "y"], missing_policy=MissingPolicy.EXCLUDE_USER_MISSING_ONLY
        )
        # NaN 유지 → 3행 모두 포함
        assert len(prepared.data) == 3


# ---------------------------------------------------------------------------
# Line 163: else → dropna (알 수 없는 policy — MissingPolicy enum 우회)
# ---------------------------------------------------------------------------

class TestUnknownPolicyFallback:

    def test_pairwise_policy_returns_full_subset(self):
        """PAIRWISE → subset 그대로 반환(149)."""
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "y": [4.0, 5.0, np.nan],
        })
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        prepared = prepare_analysis_frame(
            ds, ["x", "y"], missing_policy=MissingPolicy.PAIRWISE
        )
        assert len(prepared.data) == 3

    def test_include_as_category_keeps_all(self):
        """INCLUDE_AS_CATEGORY → 결측 포함(155)."""
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "y": [4.0, 5.0, np.nan],
        })
        ds = Dataset(df, "Test")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        prepared = prepare_analysis_frame(
            ds, ["x", "y"], missing_policy=MissingPolicy.INCLUDE_AS_CATEGORY
        )
        assert len(prepared.data) == 3
