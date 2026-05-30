"""crosstab.py 커버리지 보강 테스트.

미커버 라인:
  21, 24, 33 : _compute_cramers_v/_compute_phi NaN 반환 (n==0, min_dim==0)
  56         : string missing_policy → MissingPolicy 변환
  118-119    : grand_total == 0 → No valid data 경고
  218-219    : Pearson chi-square 예외
  232-233    : Likelihood Ratio 예외
  247-248    : Continuity Correction 예외 (2x2)
  262-263    : Fisher's Exact 예외 (2x2)
  274-275    : Cramer's V 예외 (silent pass)
  286-287    : Phi 예외 (silent pass)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.crosstab import (
    run_analysis,
    _compute_cramers_v,
    _compute_phi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_crosstab_dataset() -> Dataset:
    """2×2 교차표 데이터 (50 관측치)."""
    rng = np.random.default_rng(0)
    n = 50
    a = rng.choice(["yes", "no"], n)
    b = rng.choice(["M", "F"], n)
    df = pd.DataFrame({"treat": a, "sex": b})
    ds = Dataset(df, name="CrossData")
    ds.variables["treat"].measure = MeasureType.BINARY
    ds.variables["sex"].measure = MeasureType.BINARY
    return ds


@pytest.fixture
def three_cat_dataset() -> Dataset:
    """3×3 교차표."""
    rng = np.random.default_rng(1)
    n = 90
    a = rng.choice(["A", "B", "C"], n)
    b = rng.choice(["X", "Y", "Z"], n)
    df = pd.DataFrame({"row": a, "col": b})
    return Dataset(df, name="3x3Data")


@pytest.fixture
def all_nan_dataset() -> Dataset:
    """모든 행이 NaN → grand_total == 0."""
    df = pd.DataFrame({"row": [None, None, None], "col": [None, None, None]})
    return Dataset(df, name="AllNaN")


# ---------------------------------------------------------------------------
# _compute_cramers_v / _compute_phi NaN 반환 (lines 21, 24, 33)
# ---------------------------------------------------------------------------

class TestSafeDivNaN:

    def test_cramers_v_zero_n(self):
        """빈 분할표 (n=0) → NaN."""
        contingency = np.array([[0, 0], [0, 0]])
        result = _compute_cramers_v(contingency)
        assert np.isnan(result)

    def test_cramers_v_single_row(self):
        """1행 분할표 → min_dim=0 → NaN."""
        contingency = np.array([[3, 4, 5]])
        result = _compute_cramers_v(contingency)
        assert np.isnan(result)

    def test_phi_zero_n(self):
        """빈 2×2 분할표 → NaN."""
        contingency = np.array([[0, 0], [0, 0]])
        result = _compute_phi(contingency)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# string missing_policy (line 56)
# ---------------------------------------------------------------------------

class TestStringMissingPolicy:

    def test_string_listwise_accepted(self, binary_crosstab_dataset):
        spec = {
            "variables": {"row": "treat", "column": "sex"},
            "missing_policy": "listwise",
        }
        result = run_analysis(binary_crosstab_dataset, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# grand_total == 0 → 경고 (lines 118-119)
# ---------------------------------------------------------------------------

class TestNoValidData:

    def test_all_nan_rows_warns(self, all_nan_dataset):
        """모든 행 NaN → listwise 제거 후 grand_total=0 → 경고."""
        spec = {"variables": {"row": "row", "column": "col"}}
        result = run_analysis(all_nan_dataset, spec)
        assert any("No valid data" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Pearson chi-square 예외 (lines 218-219)
# ---------------------------------------------------------------------------

class TestPearsonChiSquareException:

    def test_chi2_exception_adds_warning(self, binary_crosstab_dataset):
        spec = {"variables": {"row": "treat", "column": "sex"}}
        with patch(
            "statworkbench.analysis.crosstab.stats.chi2_contingency",
            side_effect=Exception("chi2 fail"),
        ):
            result = run_analysis(binary_crosstab_dataset, spec)
        assert any("Pearson Chi-Square" in w or "chi2" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Likelihood Ratio 예외 (lines 232-233)
# ---------------------------------------------------------------------------

class TestLikelihoodRatioException:

    def test_lr_exception_adds_warning(self, binary_crosstab_dataset):
        spec = {"variables": {"row": "treat", "column": "sex"}}
        call_count = {"n": 0}
        original_chi2 = __import__("scipy.stats", fromlist=["chi2_contingency"]).chi2_contingency

        def _chi2_with_lr_fail(*args, **kwargs):
            call_count["n"] += 1
            if kwargs.get("lambda_") == "log-likelihood":
                raise Exception("LR fail")
            return original_chi2(*args, **kwargs)

        with patch("statworkbench.analysis.crosstab.stats.chi2_contingency",
                   side_effect=_chi2_with_lr_fail):
            result = run_analysis(binary_crosstab_dataset, spec)
        assert any("Likelihood Ratio" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Fisher's Exact 예외 (lines 262-263)
# ---------------------------------------------------------------------------

class TestFisherExactException:

    def test_fisher_exact_exception_adds_warning(self, binary_crosstab_dataset):
        spec = {"variables": {"row": "treat", "column": "sex"}}
        with patch(
            "statworkbench.analysis.crosstab.stats.fisher_exact",
            side_effect=Exception("fisher fail"),
        ):
            result = run_analysis(binary_crosstab_dataset, spec)
        assert any("Fisher" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Cramer's V 예외 — silent pass (lines 274-275)
# ---------------------------------------------------------------------------

class TestCramersVException:

    def test_cramers_v_exception_silent(self, binary_crosstab_dataset):
        """_compute_cramers_v 예외 → pass (경고 없음, 분석 완료)."""
        spec = {"variables": {"row": "treat", "column": "sex"}}
        with patch(
            "statworkbench.analysis.crosstab._compute_cramers_v",
            side_effect=Exception("cramers fail"),
        ):
            result = run_analysis(binary_crosstab_dataset, spec)
        # 분석 완료, cramers 관련 경고 없음
        assert not any("Cramer" in w for w in result.warnings)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Phi 예외 — silent pass (lines 286-287)
# ---------------------------------------------------------------------------

class TestPhiException:

    def test_phi_exception_silent(self, binary_crosstab_dataset):
        """_compute_phi 예외 → pass (경고 없음, 분석 완료)."""
        spec = {"variables": {"row": "treat", "column": "sex"}}
        with patch(
            "statworkbench.analysis.crosstab._compute_phi",
            side_effect=Exception("phi fail"),
        ):
            result = run_analysis(binary_crosstab_dataset, spec)
        assert not any("Phi" in w for w in result.warnings)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullAnalysis:

    def test_2x2_produces_fisher(self, binary_crosstab_dataset):
        spec = {"variables": {"row": "treat", "column": "sex"}}
        result = run_analysis(binary_crosstab_dataset, spec)
        chi_table = next((t for t in result.tables if "Chi-Square" in t.title), None)
        assert chi_table is not None
        tests = chi_table.dataframe["Test"].tolist()
        assert "Fisher's Exact Test" in tests

    def test_3x3_no_fisher(self, three_cat_dataset):
        spec = {"variables": {"row": "row", "column": "col"}}
        result = run_analysis(three_cat_dataset, spec)
        chi_table = next((t for t in result.tables if "Chi-Square" in t.title), None)
        assert chi_table is not None
        tests = chi_table.dataframe["Test"].tolist()
        assert "Fisher's Exact Test" not in tests
