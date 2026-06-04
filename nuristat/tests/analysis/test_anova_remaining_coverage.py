"""anova.py 미커버 라인 보강 테스트.

대상 라인:
  211     : F_welch = np.nan (denominator == 0)
  221     : df2 = np.nan (denom_df == 0)
  226     : p_welch = np.nan (F_welch 또는 df2 가 NaN)
  323-324 : except Exception in _run_scheffe
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.anova import run_analysis


# ---------------------------------------------------------------------------
# Lines 211, 221, 226: Welch ANOVA NaN branches
# 그룹 내 분산이 0 → denominator=0 → F_welch=NaN, df2=NaN, p_welch=NaN
# ---------------------------------------------------------------------------

class TestWelchANOVANaNBranches:

    def _make_zero_variance_ds(self) -> Dataset:
        """그룹 내 값이 모두 동일한 데이터셋 (within-group variance = 0)."""
        df = pd.DataFrame({
            "score": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        ds = Dataset(df, "WelchZeroVar")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["group"].measure = MeasureType.NOMINAL
        return ds

    def test_welch_nan_when_zero_variance(self):
        """그룹 내 분산=0 → denominator=0 → lines 211, 221, 226 실행."""
        ds = self._make_zero_variance_ds()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": True, "post_hoc": []},
        }
        result = run_analysis(ds, spec)
        # 에러 없이 완료 + Welch ANOVA 테이블 포함
        table_titles = [t.title for t in result.tables]
        assert any("Welch" in t for t in table_titles)

    def test_welch_result_contains_nan_f(self):
        """Welch F=NaN → 테이블 F 셀이 '' 또는 'nan'."""
        ds = self._make_zero_variance_ds()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": True, "post_hoc": []},
        }
        result = run_analysis(ds, spec)
        welch_table = next(
            (t for t in result.tables if "Welch" in t.title), None
        )
        assert welch_table is not None
        # F 값이 NaN이면 포맷 후 'nan' 또는 빈 문자열
        f_val = str(welch_table.dataframe["F"].iloc[0])
        assert "nan" in f_val.lower() or f_val == ""


# ---------------------------------------------------------------------------
# Lines 323-324: except Exception in _run_scheffe
# stats.f.ppf 를 패치하여 예외 발생 → warnings 에 기록
# ---------------------------------------------------------------------------

class TestScheffeException:

    def test_scheffe_exception_adds_warning(self):
        """_run_scheffe 내 예외 → result.warnings 에 경고(323-324)."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "score": np.concatenate([
                rng.normal(10, 1, 20),
                rng.normal(20, 1, 20),
            ]),
            "group": ["A"] * 20 + ["B"] * 20,
        })
        ds = Dataset(df, "ScheffeErr")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["group"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": False, "post_hoc": ["scheffe"]},
            "confidence_level": 0.95,
        }

        with patch(
            "nuristat.analysis.anova.stats.f.ppf",
            side_effect=RuntimeError("ppf fail"),
        ):
            result = run_analysis(ds, spec)

        assert any("Scheffe" in w for w in result.warnings)
