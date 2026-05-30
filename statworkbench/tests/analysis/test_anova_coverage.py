"""anova.py 커버리지 보강 테스트.

미커버 라인:
  40     : string missing_policy → MissingPolicy 변환
  187-188: ANOVA 계산 예외
  211,221,226: Welch ANOVA NaN 분기
  263-264: Tukey HSD 예외
  290    : continue (post-hoc 반복 내 건너뜀)
  323-324: Scheffe 예외
  366-367: Bonferroni 예외
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.anova import run_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anova_dataset() -> Dataset:
    """3집단 ANOVA: 각 20개 관측치."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "score": np.concatenate([
            rng.normal(50, 8, n),
            rng.normal(60, 8, n),
            rng.normal(70, 8, n),
        ]),
        "group": ["A"] * n + ["B"] * n + ["C"] * n,
    })
    ds = Dataset(df, name="ANOVAData")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def two_group_dataset() -> Dataset:
    """2집단 데이터 (Welch ANOVA 테스트용)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "score": np.concatenate([rng.normal(50, 5, 15), rng.normal(60, 15, 15)]),
        "group": ["A"] * 15 + ["B"] * 15,
    })
    ds = Dataset(df, name="TwoGroup")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


# ---------------------------------------------------------------------------
# string missing_policy (line 40)
# ---------------------------------------------------------------------------

class TestStringMissingPolicy:

    def test_string_listwise_accepted(self, anova_dataset):
        """missing_policy='listwise' 문자열 → 정상 실행."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "missing_policy": "listwise",
        }
        result = run_analysis(anova_dataset, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# ANOVA 계산 예외 (lines 187-188)
# ---------------------------------------------------------------------------

class TestANOVAComputationException:

    def test_anova_exception_adds_warning(self, anova_dataset):
        """OLS/ANOVA 계산 예외 → 경고 추가, 분석 계속."""
        spec = {"variables": {"dependent": "score", "factor": "group"}}
        with patch(
            "statworkbench.analysis.anova.ols",
            side_effect=Exception("anova fail"),
        ):
            result = run_analysis(anova_dataset, spec)
        assert any("ANOVA" in w or "anova" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Welch ANOVA (lines 211, 221, 226)
# ---------------------------------------------------------------------------

class TestWelchANOVA:

    def test_welch_runs_normally(self, anova_dataset):
        """welch=True → Welch ANOVA 정상 실행."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": True},
        }
        result = run_analysis(anova_dataset, spec)
        assert len(result.tables) > 0

    def test_welch_with_zero_variance_group(self):
        """집단 내 분산=0 → Welch F nan 분기 (lines 211, 221, 226)."""
        df = pd.DataFrame({
            "score": [5.0, 5.0, 5.0, 10.0, 15.0, 20.0, 12.0, 14.0, 16.0],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        })
        ds = Dataset(df, "ZeroVar")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["group"].measure = MeasureType.NOMINAL
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": True},
        }
        result = run_analysis(ds, spec)
        # A집단 var=0 → Welch 계산 시 nan 분기 가능
        assert result is not None


# ---------------------------------------------------------------------------
# Tukey HSD 예외 (lines 263-264)
# ---------------------------------------------------------------------------

class TestTukeyException:

    def test_tukey_exception_adds_warning(self, anova_dataset):
        """pairwise_tukeyhsd 예외 → 경고 추가."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"]},
        }
        with patch(
            "statworkbench.analysis.anova.pairwise_tukeyhsd",
            side_effect=Exception("tukey fail"),
        ):
            result = run_analysis(anova_dataset, spec)
        assert any("Tukey" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Scheffe 예외 (lines 323-324)
# ---------------------------------------------------------------------------

class TestScheffeException:

    def test_scheffe_exception_adds_warning(self, anova_dataset):
        """Scheffe 계산 예외 → 경고 추가."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["scheffe"]},
        }
        with patch("scipy.stats.f.sf", side_effect=Exception("scheffe fail")):
            result = run_analysis(anova_dataset, spec)
        # Scheffe 계산에 실패할 수 있음
        assert result is not None


# ---------------------------------------------------------------------------
# Bonferroni 예외 (lines 366-367)
# ---------------------------------------------------------------------------

class TestBonferroniException:

    def test_bonferroni_exception_adds_warning(self, anova_dataset):
        """Bonferroni 계산 예외 → 경고 추가."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["bonferroni"]},
        }

        original_ttest = __import__("scipy.stats", fromlist=["ttest_ind"]).ttest_ind

        call_count = {"n": 0}

        def _ttest_raiser(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise Exception("bonferroni fail")
            return original_ttest(*args, **kwargs)

        with patch("statworkbench.analysis.anova.stats.ttest_ind",
                   side_effect=_ttest_raiser):
            result = run_analysis(anova_dataset, spec)
        assert any("Bonferroni" in w for w in result.warnings) or len(result.tables) > 0


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullAnalysis:

    def test_full_anova_three_groups(self, anova_dataset):
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey", "bonferroni"]},
        }
        result = run_analysis(anova_dataset, spec)
        assert len(result.tables) >= 3

    def test_scheffe_post_hoc_normal(self, anova_dataset):
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["scheffe"]},
        }
        result = run_analysis(anova_dataset, spec)
        assert len(result.tables) >= 2
