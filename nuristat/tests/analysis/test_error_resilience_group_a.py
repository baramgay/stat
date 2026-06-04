"""Error-resilience and edge-case tests for Group A analysis modules.

Every test calls run_analysis(dataset, spec) and asserts either:
  - isinstance(result, AnalysisResult)   — graceful handling with possible warnings
  - pytest.raises(...)                   — documented expected exception only

No mocking. Real data, real function calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
src_path = str(_PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset

import nuristat.analysis.descriptive as mod_descriptive
import nuristat.analysis.frequencies as mod_frequencies
import nuristat.analysis.ttests as mod_ttests
import nuristat.analysis.anova as mod_anova
import nuristat.analysis.correlation as mod_correlation
import nuristat.analysis.regression as mod_regression
import nuristat.analysis.explore as mod_explore
import nuristat.analysis.crosstab as mod_crosstab


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_ds(df: pd.DataFrame) -> Dataset:
    """Wrap a DataFrame in a Dataset with automatic metadata inference."""
    return Dataset(data=df, name="test")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptiveResilience:

    # --- empty / minimal data ---

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"x": pd.Series([], dtype=float)}),
        pd.DataFrame({"x": [1.0], "group": ["A"]}),
        pd.DataFrame({"x": [1.0, 2.0], "group": ["A", "B"]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- all-missing values ---

    def test_all_missing(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "group": ["A", "B", "A"]})
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- constant column (zero variance) ---

    def test_constant_column(self):
        df = pd.DataFrame({"x": [5.0] * 20, "group": ["A"] * 10 + ["B"] * 10})
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- extreme values ---

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"x": [1e15, -1e15, 1e15, -1e15, 0.0] * 4}),
        pd.DataFrame({"x": [np.inf, -np.inf, 1.0, 2.0, 3.0]}),
    ])
    def test_extreme_values(self, df):
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- wrong spec ---

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"scale": []}},
        {"variables": {"scale": ["nonexistent_col"]}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- Korean column names ---

    def test_korean_column_names(self):
        df = pd.DataFrame({"점수": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "집단": ["A", "B", "A", "B", "A"]})
        ds = make_ds(df)
        # After Dataset init, columns may be renamed; use actual column names
        score_col = ds.data.columns[0]
        spec = {"variables": {"scale": [score_col]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- mixed types ---

    def test_mixed_type_column(self):
        df = pd.DataFrame({"x": ["a", 1.0, None, True, 2.5]})
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"]}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- large dataset (stress test) ---

    def test_large_dataset(self):
        rng = np.random.default_rng(0)
        n = 10_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B", "C"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"], "group": "group"}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- Unicode/special chars in data values ---

    def test_unicode_group_values(self):
        df = pd.DataFrame({
            "group": ["서울", "부산", "대구",
                      "서울", "부산"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds = make_ds(df)
        group_col = ds.data.columns[0]
        score_col = ds.data.columns[1]
        spec = {"variables": {"scale": [score_col], "group": group_col}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- grouped: single group only ---

    def test_single_group_only(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "group": ["A"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"scale": ["x"], "group": "group"}}
        result = mod_descriptive.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FREQUENCIES
# ─────────────────────────────────────────────────────────────────────────────

class TestFrequenciesResilience:

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"cat": pd.Series([], dtype=str)}),
        pd.DataFrame({"cat": ["A"]}),
        pd.DataFrame({"cat": ["A", "B"]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"target": ["cat"]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"cat": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["cat"]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"target": []}},
        {"variables": {"target": ["nonexistent"]}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"cat": ["A", "B", "A"]})
        ds = make_ds(df)
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_korean_values(self):
        df = pd.DataFrame({
            "region": ["서울", "부산", "대구",
                       "서울", "부산"],
        })
        ds = make_ds(df)
        region_col = ds.data.columns[0]
        spec = {"variables": {"target": [region_col]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_column(self):
        df = pd.DataFrame({"cat": ["A"] * 20})
        ds = make_ds(df)
        spec = {"variables": {"target": ["cat"]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_mixed_types(self):
        df = pd.DataFrame({"x": ["a", 1.0, None, True, 2.5]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_include_missing_option(self):
        df = pd.DataFrame({"cat": ["A", None, "B", None, "A"]})
        ds = make_ds(df)
        spec = {
            "variables": {"target": ["cat"]},
            "options": {"include_missing": True, "show_cumulative": True},
        }
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(1)
        n = 10_000
        df = pd.DataFrame({"cat": rng.choice(["A", "B", "C", "D"], n)})
        ds = make_ds(df)
        spec = {"variables": {"target": ["cat"]}}
        result = mod_frequencies.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 3. T-TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTtestsResilience:

    # --- empty / minimal ---

    def test_empty_data_independent(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float),
                           "group": pd.Series([], dtype=str)})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_single_row(self):
        df = pd.DataFrame({"x": [1.0], "group": ["A"]})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing_dep(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "group": ["A", "B", "A"]})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_column_independent(self):
        df = pd.DataFrame({"x": [5.0] * 20, "group": ["A"] * 10 + ["B"] * 10})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- single group (not 2 groups) ---

    def test_single_group_only(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "group": ["A"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_three_groups(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "A", "B", "B", "C", "C"],
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    # --- wrong spec ---

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"dependent": "", "group": ""}},
        {"variables": {"dependent": "nonexistent", "group": "group"}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": ["A", "B", "A"]})
        ds = make_ds(df)
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- paired t-test edge cases ---

    def test_paired_empty(self):
        df = pd.DataFrame({"pre": pd.Series([], dtype=float),
                           "post": pd.Series([], dtype=float)})
        ds = make_ds(df)
        spec = {"variables": {"paired": ["pre", "post"]}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_paired_all_missing(self):
        df = pd.DataFrame({"pre": [np.nan, np.nan], "post": [np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"paired": ["pre", "post"]}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_paired_constant(self):
        df = pd.DataFrame({"pre": [5.0] * 10, "post": [5.0] * 10})
        ds = make_ds(df)
        spec = {"variables": {"paired": ["pre", "post"]}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- extreme values ---

    def test_extreme_values(self):
        df = pd.DataFrame({
            "x": [1e15, -1e15, 1e15, -1e15, 0.0] * 2,
            "group": ["A"] * 5 + ["B"] * 5,
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- large dataset ---

    def test_large_dataset(self):
        rng = np.random.default_rng(2)
        n = 10_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "group": "group"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- Korean column names in data values ---

    def test_unicode_group_values(self):
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "region": ["서울", "서울", "서울",
                       "부산", "부산", "부산"],
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "score", "group": "region"}}
        result = mod_ttests.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANOVA
# ─────────────────────────────────────────────────────────────────────────────

class TestAnovaResilience:

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"x": pd.Series([], dtype=float),
                      "group": pd.Series([], dtype=str)}),
        pd.DataFrame({"x": [1.0], "group": ["A"]}),
        pd.DataFrame({"x": [1.0, 2.0], "group": ["A", "B"]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "group": ["A", "B", "A"]})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_column(self):
        df = pd.DataFrame({"x": [5.0] * 20, "group": ["A"] * 7 + ["B"] * 7 + ["C"] * 6})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_single_group_only(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "group": ["A"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"dependent": "x", "factor": "nonexistent"}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": ["A", "B", "A"]})
        ds = make_ds(df)
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_extreme_values(self):
        df = pd.DataFrame({
            "x": [1e15, -1e15, 1e15, -1e15, 0.0] * 4,
            "group": ["A"] * 7 + ["B"] * 7 + ["C"] * 6,
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_unicode_group_values(self):
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "region": ["서울"] * 3 + ["부산"] * 3 + ["대구"] * 3,
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "score", "factor": "region"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(3)
        n = 10_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B", "C"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_two_groups_only(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "x", "factor": "group"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationResilience:

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"x": pd.Series([], dtype=float),
                      "y": pd.Series([], dtype=float)}),
        pd.DataFrame({"x": [1.0], "y": [1.0]}),
        pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan],
                           "y": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_column(self):
        df = pd.DataFrame({"x": [5.0] * 20, "y": list(range(20))})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- only one variable (< 2 required) ---

    def test_single_variable(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_no_variables(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": []}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"target": ["nonexistent", "also_nonexistent"]}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        ds = make_ds(df)
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_extreme_values(self):
        df = pd.DataFrame({
            "x": [1e15, -1e15, 1e15, -1e15, 0.0] * 4,
            "y": list(range(20)),
        })
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_infinity_values(self):
        df = pd.DataFrame({"x": [np.inf, -np.inf, 1.0, 2.0, 3.0],
                           "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_all_methods(self, method):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30), "y": rng.normal(0, 1, 30)})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}, "options": {"method": method}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_unknown_method(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "y": [2.0, 4.0, 3.0, 5.0, 1.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]},
                "options": {"method": "totally_unknown"}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_pairwise_option(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0],
                           "y": [2.0, np.nan, 3.0, 5.0, 1.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]},
                "options": {"pairwise": True}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(5)
        n = 10_000
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x", "y"]}}
        result = mod_correlation.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionResilience:

    def test_empty_data(self):
        df = pd.DataFrame({"y": pd.Series([], dtype=float),
                           "x": pd.Series([], dtype=float)})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_single_row(self):
        df = pd.DataFrame({"y": [1.0], "x": [2.0]})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"y": [np.nan, np.nan, np.nan],
                           "x": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_predictor(self):
        df = pd.DataFrame({"y": list(range(20)), "x": [5.0] * 20})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_outcome(self):
        df = pd.DataFrame({"y": [5.0] * 20, "x": list(range(20))})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"dependent": "", "independent": []}},
        {"variables": {"dependent": "nonexistent", "independent": ["x"]}},
        {"variables": {"dependent": "y", "independent": ["nonexistent"]}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
        ds = make_ds(df)
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_extreme_values(self):
        df = pd.DataFrame({
            "y": [1e15, -1e15, 1e15, -1e15, 0.0] * 4,
            "x": list(range(20)),
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_perfect_multicollinearity(self):
        rng = np.random.default_rng(6)
        x1 = rng.normal(0, 1, 50)
        df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x1": x1, "x2": x1 * 2})
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x1", "x2"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_more_predictors_than_obs(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0],
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "x3": [7.0, 8.0, 9.0],
            "x4": [10.0, 11.0, 12.0],
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y",
                               "independent": ["x1", "x2", "x3", "x4"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(7)
        n = 10_000
        df = pd.DataFrame({
            "y": rng.normal(0, 1, n),
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B", "C"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_predictors_alias_compatibility(self):
        """Test that both 'predictors' and 'independent' keys work."""
        rng = np.random.default_rng(8)
        df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})
        ds = make_ds(df)
        # 'predictors' key
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}}
        result = mod_regression.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPLORE
# ─────────────────────────────────────────────────────────────────────────────

class TestExploreResilience:

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"x": pd.Series([], dtype=float)}),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0, 2.0]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_column(self):
        df = pd.DataFrame({"x": [5.0] * 20})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_extreme_values(self):
        df = pd.DataFrame({"x": [1e15, -1e15, 1e15, -1e15, 0.0] * 4})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_infinity_values(self):
        df = pd.DataFrame({"x": [np.inf, -np.inf, 1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    # --- nonexistent variable raises ValueError (documented behavior) ---

    def test_nonexistent_variable_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["nonexistent"]}}
        result = mod_explore.run_analysis(ds, spec)
        assert result.warnings, "존재하지 않는 변수 → warnings 반환 기대"

    def test_nonexistent_factor_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"], "factor": "nonexistent"}}
        result = mod_explore.run_analysis(ds, spec)
        assert result.warnings, "존재하지 않는 factor 변수 → warnings 반환 기대"

    # --- empty target list returns gracefully ---

    def test_empty_target_list(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {"variables": {"target": []}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_empty_spec(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = make_ds(df)
        spec = {}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_with_factor_single_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "group": ["A"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"], "factor": "group"}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_with_factor_all_missing_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"], "factor": "group"}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_korean_column_names(self):
        df = pd.DataFrame({"점수": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = make_ds(df)
        score_col = ds.data.columns[0]
        spec = {"variables": {"target": [score_col]}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(9)
        n = 10_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B", "C"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"target": ["x"], "factor": "group"}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_unicode_group_values(self):
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "region": ["서울", "부산", "대구",
                       "서울", "부산"],
        })
        ds = make_ds(df)
        spec = {"variables": {"target": ["score"], "factor": "region"}}
        result = mod_explore.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSSTAB
# ─────────────────────────────────────────────────────────────────────────────

class TestCrosstabResilience:

    @pytest.mark.parametrize("df", [
        pd.DataFrame({"row": pd.Series([], dtype=str),
                      "col": pd.Series([], dtype=str)}),
        pd.DataFrame({"row": ["A"], "col": ["X"]}),
        pd.DataFrame({"row": ["A", "B"], "col": ["X", "Y"]}),
    ])
    def test_minimal_data(self, df):
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_all_missing(self):
        df = pd.DataFrame({"row": [np.nan, np.nan, np.nan],
                           "col": [np.nan, np.nan, np.nan]})
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_row_variable(self):
        df = pd.DataFrame({"row": ["A"] * 10,
                           "col": ["X", "Y"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_single_cell(self):
        df = pd.DataFrame({"row": ["A"] * 5, "col": ["X"] * 5})
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("spec", [
        {},
        {"variables": {}},
        {"variables": {"row": "", "column": ""}},
        {"variables": {"row": "nonexistent", "column": "col"}},
        {"variables": {"row": "row", "column": "nonexistent"}},
    ])
    def test_wrong_spec(self, spec):
        df = pd.DataFrame({"row": ["A", "B", "A"], "col": ["X", "Y", "X"]})
        ds = make_ds(df)
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_korean_values(self):
        df = pd.DataFrame({
            "region": ["서울", "부산", "대구",
                       "서울", "부산"],
            "gender": ["남", "여", "남", "여", "남"],
        })
        ds = make_ds(df)
        region_col = ds.data.columns[0]
        gender_col = ds.data.columns[1]
        spec = {"variables": {"row": region_col, "column": gender_col}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_large_dataset(self):
        rng = np.random.default_rng(10)
        n = 10_000
        df = pd.DataFrame({
            "row": rng.choice(["A", "B", "C", "D"], n),
            "col": rng.choice(["X", "Y", "Z"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_with_layer(self):
        rng = np.random.default_rng(11)
        n = 60
        df = pd.DataFrame({
            "row": rng.choice(["A", "B"], n),
            "col": rng.choice(["X", "Y"], n),
            "layer": rng.choice(["L1", "L2"], n),
        })
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col", "layer": "layer"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_mixed_types_in_column(self):
        df = pd.DataFrame({"row": ["a", 1.0, None, True, "b"],
                           "col": ["X", "Y", "X", "Y", "X"]})
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_numeric_categories(self):
        df = pd.DataFrame({
            "row": [1, 2, 1, 2, 1, 2],
            "col": [10, 20, 10, 20, 10, 20],
        })
        ds = make_ds(df)
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
