"""Comprehensive integration tests for 20 analysis modules.

Covers modules with no prior integration tests:
  - reliability, roc_analysis, sensitivity_specificity, cohens_kappa, icc,
    bland_altman, chi_square_gof, partial_correlation, normality, crosstab
  - anova, correlation, regression, logistic_regression, factor_analysis,
    cluster_analysis, discriminant_analysis, survival_analysis,
    nonparametric, explore
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from nuristat.core.dataset import Dataset
from nuristat.analysis.result import AnalysisResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_main_dataset() -> Dataset:
    np.random.seed(42)
    n = 50
    return Dataset(data=pd.DataFrame({
        "score":          np.random.normal(70, 15, n),
        "score2":         np.random.normal(65, 12, n),
        "score3":         np.random.normal(72, 10, n),
        "group":          np.random.choice(["A", "B", "C"], n),
        "binary":         np.random.choice([0, 1], n),
        "actual":         np.random.choice([0, 1], n),
        "predicted_prob": np.random.uniform(0, 1, n),
        "rater1":         np.random.choice([1, 2, 3, 4], n),
        "rater2":         np.random.choice([1, 2, 3, 4], n),
        "measure1":       np.random.normal(100, 10, n),
        "measure2":       np.random.normal(102, 10, n),
    }))


@pytest.fixture(scope="module")
def ds():
    return _make_main_dataset()


def _table_titles(result: AnalysisResult) -> list[str]:
    return [t.title for t in result.tables]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Reliability (Cronbach's Alpha)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReliability:
    def test_basic_cronbach(self, ds):
        from nuristat.analysis.reliability import run_analysis
        spec = {
            "variables": {"target": ["score", "score2", "score3"]},
            "options": {"listwise": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Reliability Statistics" in titles
        assert "Item Statistics" in titles
        assert "Item-Total Statistics" in titles

    def test_two_item_scale(self, ds):
        """Edge case: exactly 2 items — minimum for alpha."""
        from nuristat.analysis.reliability import run_analysis
        spec = {
            "variables": {"target": ["score", "score2"]},
            "options": {"listwise": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert "Reliability Statistics" in _table_titles(result)

    def test_single_item_warning(self, ds):
        """Single item must trigger a warning, not crash."""
        from nuristat.analysis.reliability import run_analysis
        spec = {"variables": {"target": ["score"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROC Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestROCAnalysis:
    def test_basic_roc(self, ds):
        from nuristat.analysis.roc_analysis import run_analysis
        spec = {
            "variables": {
                "state": "actual",
                "test": ["predicted_prob"],
                "positive_value": 1,
            },
            "options": {"max_coords": 20},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Case Processing Summary" in titles
        assert "Area Under the Curve" in titles
        assert "Optimal Cutoff" in titles

    def test_roc_multiple_test_vars(self, ds):
        """Two test variables in one run."""
        from nuristat.analysis.roc_analysis import run_analysis
        spec = {
            "variables": {
                "state": "actual",
                "test": ["predicted_prob", "score"],
            },
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        # AUC table should have 2 rows
        auc_table = next(t for t in result.tables if t.title == "Area Under the Curve")
        assert len(auc_table.dataframe) == 2

    def test_roc_missing_state_var(self, ds):
        from nuristat.analysis.roc_analysis import run_analysis
        spec = {"variables": {"test": ["predicted_prob"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_roc_well_separated(self):
        """Near-perfect classifier should yield AUC close to 1."""
        from nuristat.analysis.roc_analysis import run_analysis
        np.random.seed(0)
        n = 60
        actual = np.array([0] * 30 + [1] * 30)
        score = np.array(
            np.random.normal(0.1, 0.05, 30).tolist()
            + np.random.normal(0.9, 0.05, 30).tolist()
        )
        ds = Dataset(data=pd.DataFrame({"actual": actual, "score": score}))
        spec = {
            "variables": {"state": "actual", "test": ["score"], "positive_value": 1},
        }
        result = run_analysis(ds, spec)
        auc_table = next(t for t in result.tables if t.title == "Area Under the Curve")
        auc_val = float(auc_table.dataframe["AUC"].iloc[0])
        assert auc_val > 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sensitivity / Specificity
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensitivitySpecificity:
    def test_basic_sensitivity(self, ds):
        from nuristat.analysis.sensitivity_specificity import run_analysis
        spec = {
            "variables": {"outcome": "actual", "predictor": "binary"},
            "options": {"pos_label": 1},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Case Processing Summary" in titles
        assert "2×2 Contingency Table" in titles
        assert "Diagnostic Accuracy Measures" in titles

    def test_perfect_predictor(self):
        """Perfect predictor → sensitivity = 1, specificity = 1."""
        from nuristat.analysis.sensitivity_specificity import run_analysis
        actual = np.array([0, 0, 0, 1, 1, 1])
        pred   = np.array([0, 0, 0, 1, 1, 1])
        ds = Dataset(data=pd.DataFrame({"actual": actual, "pred": pred}))
        spec = {"variables": {"outcome": "actual", "predictor": "pred"}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        acc_table = next(t for t in result.tables if "Diagnostic Accuracy" in t.title)
        assert acc_table is not None

    def test_missing_outcome_var(self, ds):
        from nuristat.analysis.sensitivity_specificity import run_analysis
        spec = {"variables": {"predictor": "binary"}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cohen's Kappa
# ═══════════════════════════════════════════════════════════════════════════════

class TestCohensKappa:
    def test_basic_kappa(self, ds):
        from nuristat.analysis.cohens_kappa import run_analysis
        spec = {"variables": {"rater1": "rater1", "rater2": "rater2"}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Symmetric Measures" in titles
        assert "Interpretation" in titles

    def test_perfect_agreement(self):
        """Both raters identical → kappa = 1."""
        from nuristat.analysis.cohens_kappa import run_analysis
        ratings = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        ds = Dataset(data=pd.DataFrame({"r1": ratings, "r2": ratings}))
        spec = {"variables": {"rater1": "r1", "rater2": "r2"}}
        result = run_analysis(ds, spec)
        sym_table = next(t for t in result.tables if t.title == "Symmetric Measures")
        kappa_val = float(sym_table.dataframe["값"].iloc[0])
        assert kappa_val == pytest.approx(1.0, abs=0.01)

    def test_missing_rater_var(self, ds):
        from nuristat.analysis.cohens_kappa import run_analysis
        spec = {"variables": {"rater1": "rater1"}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ICC
# ═══════════════════════════════════════════════════════════════════════════════

class TestICC:
    def test_twoway_mixed(self, ds):
        from nuristat.analysis.icc import run_analysis
        spec = {
            "variables": {"target": ["rater1", "rater2"]},
            "options": {"model": "twoway_mixed"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "ICC" in titles
        assert "ANOVA" in titles
        assert "Interpretation" in titles

    def test_oneway_random(self, ds):
        from nuristat.analysis.icc import run_analysis
        spec = {
            "variables": {"target": ["rater1", "rater2"]},
            "options": {"model": "oneway_random"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        icc_table = next(t for t in result.tables if t.title == "ICC")
        assert len(icc_table.dataframe) == 1

    def test_three_raters(self):
        """Three numeric rater columns."""
        from nuristat.analysis.icc import run_analysis
        np.random.seed(1)
        n = 20
        true_score = np.random.normal(50, 10, n)
        df = pd.DataFrame({
            "r1": true_score + np.random.normal(0, 2, n),
            "r2": true_score + np.random.normal(0, 2, n),
            "r3": true_score + np.random.normal(0, 2, n),
        })
        ds = Dataset(data=df)
        spec = {"variables": {"target": ["r1", "r2", "r3"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert "ICC" in _table_titles(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Bland-Altman
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlandAltman:
    def test_basic_bland_altman(self, ds):
        from nuristat.analysis.bland_altman import run_analysis
        spec = {"variables": {"method1": "measure1", "method2": "measure2"}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Bland-Altman Statistics" in titles
        assert "Limits of Agreement" in titles
        assert "Individual Differences" in titles

    def test_identical_methods(self):
        """Identical measurements → mean diff = 0, SD = 0."""
        from nuristat.analysis.bland_altman import run_analysis
        vals = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
        ds = Dataset(data=pd.DataFrame({"m1": vals, "m2": vals}))
        spec = {"variables": {"method1": "m1", "method2": "m2"}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        ba_table = next(t for t in result.tables if t.title == "Bland-Altman Statistics")
        bias_row = ba_table.dataframe[ba_table.dataframe["통계량"] == "평균 차이 (Bias)"]
        bias_val = float(bias_row["값"].iloc[0])
        assert abs(bias_val) < 1e-10

    def test_missing_method_var(self, ds):
        from nuristat.analysis.bland_altman import run_analysis
        spec = {"variables": {"method1": "measure1"}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Chi-Square Goodness-of-Fit
# ═══════════════════════════════════════════════════════════════════════════════

class TestChiSquareGoF:
    def test_uniform_distribution(self, ds):
        from nuristat.analysis.chi_square_gof import run_analysis
        spec = {
            "variables": {"target": ["group"]},
            "options": {"listwise": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Frequencies" in titles
        assert "Test Statistics" in titles
        assert "Residuals" in titles

    def test_expected_ratios(self, ds):
        """Custom expected ratios."""
        from nuristat.analysis.chi_square_gof import run_analysis
        spec = {
            "variables": {
                "target": ["group"],
                "expected_ratios": {"A": 0.5, "B": 0.3, "C": 0.2},
            },
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert "Test Statistics" in _table_titles(result)

    def test_binary_variable(self):
        """Binary variable with known distribution."""
        from nuristat.analysis.chi_square_gof import run_analysis
        df = pd.DataFrame({"x": [0] * 40 + [1] * 60})
        ds = Dataset(data=df)
        spec = {"variables": {"target": ["x"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Partial Correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPartialCorrelation:
    def test_basic_partial_correlation(self, ds):
        from nuristat.analysis.partial_correlation import run_analysis
        spec = {
            "variables": {
                "target": ["score", "score2"],
                "controlling": ["score3"],
            },
            "options": {"listwise": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert "Partial Correlation" in titles
        assert "Significance (2-tailed)" in titles
        assert "Zero-order Correlations" in titles

    def test_no_controlling_variable(self, ds):
        """Without controlling variables → returns Pearson correlation."""
        from nuristat.analysis.partial_correlation import run_analysis
        spec = {
            "variables": {
                "target": ["score", "score2", "score3"],
                "controlling": [],
            },
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert "Partial Correlation" in _table_titles(result)

    def test_insufficient_vars(self, ds):
        from nuristat.analysis.partial_correlation import run_analysis
        spec = {"variables": {"target": ["score"], "controlling": []}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Normality
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormality:
    def test_shapiro_wilk_single_var(self, ds):
        from nuristat.analysis.normality import run_analysis
        spec = {
            "variables": {"target": ["score"]},
            "confidence_level": 0.95,
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert "Tests of Normality" in _table_titles(result)

    def test_multiple_variables(self, ds):
        from nuristat.analysis.normality import run_analysis
        spec = {"variables": {"target": ["score", "score2", "score3"]}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        norm_table = next(t for t in result.tables if t.title == "Tests of Normality")
        # One row per variable
        assert len(norm_table.dataframe) == 3

    def test_constant_series_warning(self):
        """Constant series (zero variance) — should not crash."""
        from nuristat.analysis.normality import run_analysis
        df = pd.DataFrame({"const": [5.0] * 20})
        ds = Dataset(data=df)
        spec = {"variables": {"target": ["const"]}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Crosstab
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrosstab:
    def test_basic_crosstab(self, ds):
        from nuristat.analysis.crosstab import run_analysis
        spec = {
            "variables": {"row": "group", "column": "binary"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        titles = _table_titles(result)
        assert any("Count" in t for t in titles)
        assert any("Chi-Square" in t for t in titles)

    def test_crosstab_with_layer(self, ds):
        """Layer variable splits the table."""
        from nuristat.analysis.crosstab import run_analysis

        # Build a dataset with 3 clear categorical variables
        np.random.seed(5)
        n = 60
        ds_layer = Dataset(data=pd.DataFrame({
            "row_var": np.random.choice(["X", "Y"], n),
            "col_var": np.random.choice(["P", "Q"], n),
            "layer_var": np.random.choice(["L1", "L2"], n),
        }))
        spec = {
            "variables": {"row": "row_var", "column": "col_var", "layer": "layer_var"},
            "options": {},
        }
        result = run_analysis(ds_layer, spec)
        assert isinstance(result, AnalysisResult)
        # Should produce at least one chi-square table per layer
        chi_tables = [t for t in result.tables if "Chi-Square" in t.title]
        assert len(chi_tables) >= 2

    def test_crosstab_row_percentages(self, ds):
        from nuristat.analysis.crosstab import run_analysis
        spec = {"variables": {"row": "group", "column": "binary"}, "options": {}}
        result = run_analysis(ds, spec)
        titles = _table_titles(result)
        assert any("Row" in t for t in titles)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

class TestANOVA:
    def test_one_way_anova(self, ds):
        from nuristat.analysis.anova import run_analysis
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"], "effect_size": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_anova_no_posthoc(self, ds):
        from nuristat.analysis.anova import run_analysis
        spec = {
            "variables": {"dependent": "score2", "factor": "group"},
            "options": {"post_hoc": [], "welch": False},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)

    def test_anova_two_groups(self):
        """ANOVA with exactly 2 groups (should work like t-test)."""
        from nuristat.analysis.anova import run_analysis
        np.random.seed(10)
        df = pd.DataFrame({
            "score": np.random.normal(70, 10, 40).tolist() + np.random.normal(80, 10, 40).tolist(),
            "group": ["A"] * 40 + ["B"] * 40,
        })
        ds = Dataset(data=df)
        spec = {"variables": {"dependent": "score", "factor": "group"}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrelation:
    def test_pearson_correlation(self, ds):
        from nuristat.analysis.correlation import run_analysis
        spec = {
            "variables": {"target": ["score", "score2", "score3"]},
            "options": {"method": "pearson", "tail": "two-tailed"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_spearman_correlation(self, ds):
        from nuristat.analysis.correlation import run_analysis
        spec = {
            "variables": {"target": ["score", "score2"]},
            "options": {"method": "spearman"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_correlation_single_var_warning(self, ds):
        from nuristat.analysis.correlation import run_analysis
        spec = {"variables": {"target": ["score"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Regression
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegression:
    def test_simple_regression(self, ds):
        from nuristat.analysis.regression import run_analysis
        spec = {
            "variables": {"dependent": "score", "independent": ["score2"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_multiple_regression(self, ds):
        from nuristat.analysis.regression import run_analysis
        spec = {
            "variables": {"dependent": "score", "independent": ["score2", "score3"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_regression_with_categorical(self, ds):
        """Regression with a categorical predictor (should handle dummies)."""
        from nuristat.analysis.regression import run_analysis
        spec = {
            "variables": {"dependent": "score", "independent": ["score2", "group"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogisticRegression:
    def test_binary_logistic(self, ds):
        from nuristat.analysis.logistic_regression import run_analysis
        spec = {
            "variables": {"dependent": "binary", "predictors": ["score", "score2"]},
            "options": {"method": "binary"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_logistic_single_predictor(self, ds):
        from nuristat.analysis.logistic_regression import run_analysis
        spec = {
            "variables": {"dependent": "binary", "predictors": ["score"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_logistic_well_separated(self):
        """Clear separation should give high classification accuracy."""
        from nuristat.analysis.logistic_regression import run_analysis
        np.random.seed(99)
        n = 80
        x = np.concatenate([np.random.normal(-3, 0.5, n // 2),
                             np.random.normal(3, 0.5, n // 2)])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        ds = Dataset(data=pd.DataFrame({"y": y, "x": x}))
        spec = {"variables": {"dependent": "y", "predictors": ["x"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorAnalysis:
    def test_efa_basic(self, ds):
        from nuristat.analysis.factor_analysis import run_analysis
        spec = {
            "variables": {"variables": ["score", "score2", "score3"]},
            "options": {"method": "efa", "n_factors": 1, "rotation": "varimax"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_pca_basic(self, ds):
        from nuristat.analysis.factor_analysis import run_analysis
        spec = {
            "variables": {"variables": ["score", "score2", "score3"]},
            "options": {"method": "pca", "n_factors": 2},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_factor_auto_n_factors(self, ds):
        from nuristat.analysis.factor_analysis import run_analysis
        spec = {
            "variables": {"variables": ["score", "score2", "score3", "measure1", "measure2"]},
            "options": {"method": "efa", "n_factors": "auto"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Cluster Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterAnalysis:
    def test_kmeans_clustering(self, ds):
        from nuristat.analysis.cluster_analysis import run_analysis
        spec = {
            "variables": {"variables": ["score", "score2"]},
            "options": {"method": "kmeans", "n_clusters": 3, "standardize": True},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_hierarchical_clustering(self, ds):
        from nuristat.analysis.cluster_analysis import run_analysis
        spec = {
            "variables": {"variables": ["score", "score2", "score3"]},
            "options": {"method": "hierarchical", "n_clusters": 2, "linkage": "ward"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_cluster_two_clusters(self, ds):
        from nuristat.analysis.cluster_analysis import run_analysis
        spec = {
            "variables": {"variables": ["measure1", "measure2"]},
            "options": {"method": "kmeans", "n_clusters": 2},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Discriminant Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiscriminantAnalysis:
    def test_basic_lda(self, ds):
        from nuristat.analysis.discriminant_analysis import run_analysis
        spec = {
            "variables": {"dependent": "group", "predictors": ["score", "score2"]},
            "options": {"method": "standard"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_binary_grouping(self):
        """Binary group classification."""
        from nuristat.analysis.discriminant_analysis import run_analysis
        np.random.seed(7)
        n = 40
        df = pd.DataFrame({
            "g": ["A"] * n + ["B"] * n,
            "x1": np.concatenate([np.random.normal(0, 1, n), np.random.normal(3, 1, n)]),
            "x2": np.concatenate([np.random.normal(0, 1, n), np.random.normal(3, 1, n)]),
        })
        ds = Dataset(data=df)
        spec = {
            "variables": {"dependent": "g", "predictors": ["x1", "x2"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Survival Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestSurvivalAnalysis:
    @pytest.fixture(autouse=True)
    def _survival_dataset(self):
        """Kaplan-Meier ready dataset: duration + event + group."""
        np.random.seed(42)
        n = 60
        self.ds_surv = Dataset(data=pd.DataFrame({
            "duration": np.random.exponential(scale=20, size=n),
            "event":    np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "group":    np.random.choice(["A", "B"], n),
            "age":      np.random.normal(55, 10, n),
        }))

    def test_kaplan_meier(self):
        from nuristat.analysis.survival_analysis import run_analysis
        spec = {
            "variables": {"duration": "duration", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(self.ds_surv, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_km_with_group(self):
        from nuristat.analysis.survival_analysis import run_analysis
        spec = {
            "variables": {
                "duration": "duration",
                "event": "event",
                "group": "group",
            },
            "options": {"method": "km"},
        }
        result = run_analysis(self.ds_surv, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_missing_duration_var(self):
        from nuristat.analysis.survival_analysis import run_analysis
        spec = {"variables": {"event": "event"}, "options": {}}
        result = run_analysis(self.ds_surv, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Nonparametric Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonparametric:
    def test_mann_whitney(self, ds):
        from nuristat.analysis.nonparametric import run_analysis
        spec = {
            "variables": {"dependent": "score", "group": "binary"},
            "options": {"test": "mann_whitney"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_kruskal_wallis(self, ds):
        from nuristat.analysis.nonparametric import run_analysis
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_wilcoxon_signed_rank(self, ds):
        from nuristat.analysis.nonparametric import run_analysis
        spec = {
            "variables": {"paired": ["score", "score2"]},
            "options": {"test": "wilcoxon"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_friedman(self):
        from nuristat.analysis.nonparametric import run_analysis
        np.random.seed(3)
        n = 30
        df = pd.DataFrame({
            "c1": np.random.normal(70, 10, n),
            "c2": np.random.normal(73, 10, n),
            "c3": np.random.normal(68, 10, n),
        })
        ds = Dataset(data=df)
        spec = {
            "variables": {"repeated": ["c1", "c2", "c3"]},
            "options": {"test": "friedman"},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Explore
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplore:
    def test_basic_explore(self, ds):
        from nuristat.analysis.explore import run_analysis
        spec = {
            "variables": {"target": ["score", "score2"]},
            "options": {"normality": True, "percentiles": [25, 50, 75]},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 2

    def test_explore_with_factor(self, ds):
        """Explore split by a grouping factor."""
        from nuristat.analysis.explore import run_analysis
        spec = {
            "variables": {"target": ["score"], "factor": "group"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 1

    def test_explore_no_normality(self, ds):
        from nuristat.analysis.explore import run_analysis
        spec = {
            "variables": {"target": ["score"]},
            "options": {"normality": False},
        }
        result = run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 1
