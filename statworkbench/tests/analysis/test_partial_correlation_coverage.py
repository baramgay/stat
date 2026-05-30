"""partial_correlation.py 커버리지 보강 테스트.

미커버 라인:
  72      : denom == 0 → mat[i,j] = 0.0 (_partial_corr_matrix 직접 호출)
  84      : df <= 0 → return np.nan (_calc_pvalue 직접)
  88      : denom == 0 → unreachable (clip 때문), 생략
  138-139 : except Exception: pass — unreachable, 생략
  154-155 : n_after < 4 → 경고 + return
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.partial_correlation import (
    run_analysis,
    _partial_corr_matrix,
    _calc_pvalue,
)


# ---------------------------------------------------------------------------
# Line 72: denom == 0 in _partial_corr_matrix → mat[i,j] = 0.0
# ---------------------------------------------------------------------------

class TestPartialCorrDenomZero:

    def test_denom_zero_gives_zero_corr(self):
        """Ri 대각이 0이면 denom=0 → mat[i,j]=0.0 (line 72)."""
        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({
            "v1": rng.normal(0, 1, n),
            "v2": rng.normal(0, 1, n),
            "ctrl": rng.normal(0, 1, n),
        })

        # np.linalg.inv를 패치해서 대각이 0인 역행렬 반환
        fake_inv = np.zeros((3, 3))  # 모든 대각 = 0

        with patch("statworkbench.analysis.partial_correlation.np.linalg.inv",
                   return_value=fake_inv):
            result = _partial_corr_matrix(df, ["v1", "v2"], ["ctrl"])

        # denom=0 → mat[0,1] = 0
        assert result.loc["v1", "v2"] == 0.0


# ---------------------------------------------------------------------------
# Line 84: df <= 0 → return np.nan (_calc_pvalue 직접)
# ---------------------------------------------------------------------------

class TestCalcPvalueNegDF:

    def test_df_zero_returns_nan(self):
        """df=0 → return np.nan (line 84)."""
        result = _calc_pvalue(0.5, 0)
        assert np.isnan(result)

    def test_df_negative_returns_nan(self):
        """df=-1 → return np.nan (line 84)."""
        result = _calc_pvalue(0.5, -1)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Lines 154-155: n_after < 4 → 경고 + return
# ---------------------------------------------------------------------------

class TestTooFewValidCases:

    def test_3_rows_gives_warning(self):
        """3개 행(< 4) → 유효 케이스 부족 경고."""
        df = pd.DataFrame({
            "v1": [1.0, 2.0, 3.0],
            "v2": [4.0, 5.0, 6.0],
        })
        ds = Dataset(df, "FewRows")
        ds.variables["v1"].measure = MeasureType.SCALE
        ds.variables["v2"].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["v1", "v2"], "controlling": []}}
        result = run_analysis(ds, spec)
        assert any("케이스" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestNormalPartialCorr:

    def test_with_controlling_variable(self):
        rng = np.random.default_rng(42)
        n = 50
        z = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "x": z + rng.normal(0, 0.3, n),
            "y": z + rng.normal(0, 0.3, n),
            "z": z,
        })
        ds = Dataset(df, "PCorrData")
        for col in ["x", "y", "z"]:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["x", "y"], "controlling": ["z"]}}
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 3
