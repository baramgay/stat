"""chi_square_gof.py 미커버 라인 보강 테스트.

대상 라인:
  75-76   : n_after==0 → 경고 + 조기 반환
  104-107 : ratio_sum==0 → 경고 + 균등 분포로 대체
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.chi_square_gof import run_analysis


# ---------------------------------------------------------------------------
# Lines 75-76: n_after==0 → 경고 + return
# ---------------------------------------------------------------------------

class TestNAfterZero:

    def test_all_nan_data_returns_early_with_warning(self):
        """모든 값이 NaN → listwise 제거 후 n_after=0 → lines 75-76."""
        df = pd.DataFrame({"cat": [np.nan, np.nan, np.nan]})
        ds = Dataset(df, "AllNaN")
        ds.variables["cat"].measure = MeasureType.NOMINAL

        spec = {"variables": {"target": ["cat"]}}
        result = run_analysis(ds, spec)
        assert any("유효한 케이스" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 104-107: ratio_sum==0 → 균등 분포로 대체 경고
# ---------------------------------------------------------------------------

class TestRatioSumZero:

    def test_all_zero_expected_ratios_warns_and_uses_uniform(self):
        """관측 범주에 대한 expected_ratios 합이 0 → lines 104-107 실행."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B", "C", "C"]})
        ds = Dataset(df, "RatioZero")
        ds.variables["cat"].measure = MeasureType.NOMINAL

        # 범주 A, B, C 모두 비율 0 → ratio_sum = 0
        spec = {
            "variables": {
                "target": ["cat"],
                "expected_ratios": {"A": 0.0, "B": 0.0, "C": 0.0},
            }
        }
        result = run_analysis(ds, spec)
        assert any("기대 비율 합계가 0" in w for w in result.warnings)
        assert len(result.tables) > 0
