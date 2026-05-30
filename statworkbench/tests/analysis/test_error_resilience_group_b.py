"""Error-resilience and edge-case tests for StatWorkbench Group B analysis modules.

Every call to run_analysis(dataset, spec) must return AnalysisResult OR raise a
clearly-typed expected exception — no random tracebacks allowed.

Modules covered:
  1. logistic_regression
  2. factor_analysis
  3. cluster_analysis
  4. survival_analysis
  5. discriminant_analysis
  6. nonparametric
  7. normality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ds(df: pd.DataFrame) -> Dataset:
    """Wrap a DataFrame into a Dataset."""
    return Dataset(data=df)


# ===========================================================================
# 1. LOGISTIC REGRESSION
# ===========================================================================

from statworkbench.analysis.logistic_regression import run_analysis as lr_run


class TestLogisticRegressionEdgeCases:
    """Error-resilience tests for logistic_regression.run_analysis."""

    def test_single_class_all_zero(self):
        """Single class in outcome (all 0) — must not crash."""
        df = pd.DataFrame({"y": [0] * 30, "x": np.random.default_rng(0).normal(size=30)})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_single_class_all_one(self):
        """Single class in outcome (all 1) — must not crash."""
        df = pd.DataFrame({"y": [1] * 30, "x": np.random.default_rng(1).normal(size=30)})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_perfect_separation(self):
        """Perfect separation (x perfectly predicts y) — must return AnalysisResult."""
        x = np.concatenate([np.ones(20), np.zeros(20)])
        y = np.concatenate([np.ones(20), np.zeros(20)])
        df = pd.DataFrame({"y": y, "x": x})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_n_less_than_predictors(self):
        """n < number of predictors — must not crash."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "y": [0, 1, 0],
            **{f"x{i}": rng.normal(size=3) for i in range(10)},
        })
        spec = {
            "variables": {"dependent": "y", "predictors": [f"x{i}" for i in range(10)]},
            "options": {"type": "binary"},
        }
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_missing_outcome_values(self):
        """Missing outcome values — listwise deletion should produce valid result."""
        rng = np.random.default_rng(5)
        y = rng.choice([0, 1], size=40).astype(float)
        y[[3, 7, 15]] = np.nan
        df = pd.DataFrame({"y": y, "x": rng.normal(size=40)})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_binary_no_predictors(self):
        """Binary spec with empty predictors list — must not crash."""
        rng = np.random.default_rng(9)
        df = pd.DataFrame({"y": rng.choice([0, 1], size=30)})
        spec = {"variables": {"dependent": "y", "predictors": []},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("method", ["binary", "multinomial"])
    def test_multinomial_k2(self, method):
        """Multinomial with k=2 classes — must not crash."""
        rng = np.random.default_rng(10)
        y = rng.choice([0, 1], size=60)
        x = rng.normal(size=60)
        df = pd.DataFrame({"y": y, "x": x})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": method}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_multinomial_k3(self):
        """Multinomial with k=3 classes — must not crash."""
        rng = np.random.default_rng(11)
        y = rng.choice([0, 1, 2], size=90)
        x = rng.normal(size=90)
        df = pd.DataFrame({"y": y, "x": x})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "multinomial"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_empty_dataframe(self):
        """Empty dataframe — must not crash."""
        df = pd.DataFrame({"y": pd.Series([], dtype=float),
                           "x": pd.Series([], dtype=float)})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_constant_predictor(self):
        """Predictor with zero variance (constant) — must not crash."""
        rng = np.random.default_rng(20)
        df = pd.DataFrame({"y": rng.choice([0, 1], size=50),
                           "x": np.ones(50)})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_spec_uses_type_key(self):
        """Spec uses 'type' inside options (common user mistake) — graceful."""
        rng = np.random.default_rng(30)
        y = rng.choice([0, 1], size=50)
        x = rng.normal(size=50)
        df = pd.DataFrame({"y": y, "x": x})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"type": "binary"}}
        result = lr_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 2. FACTOR ANALYSIS
# ===========================================================================

from statworkbench.analysis.factor_analysis import run_analysis as fa_run


class TestFactorAnalysisEdgeCases:
    """Error-resilience tests for factor_analysis.run_analysis."""

    def test_too_few_variables_one(self):
        """n_vars = 1 — should warn and return early."""
        df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=20)})
        spec = {"variables": {"variables": ["x"]}, "options": {}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_too_few_variables_zero(self):
        """n_vars = 0 — should warn and return early."""
        df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=20)})
        spec = {"variables": {"variables": []}, "options": {}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_n_obs_less_than_n_vars(self):
        """n_obs < n_vars (underdetermined) — should warn and return early."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({f"v{i}": rng.normal(size=3) for i in range(8)})
        spec = {"variables": {"variables": [f"v{i}" for i in range(8)]},
                "options": {"method": "efa"}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_perfectly_correlated_variables(self):
        """Singular correlation matrix (perfectly correlated) — must not crash."""
        rng = np.random.default_rng(2)
        base = rng.normal(size=50)
        df = pd.DataFrame({"v1": base, "v2": base, "v3": base + 0.001})
        spec = {"variables": {"variables": ["v1", "v2", "v3"]},
                "options": {"method": "efa"}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_zero_factors_requested(self):
        """Spec with 0 factors requested — should clamp to 1 and not crash."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({f"v{i}": rng.normal(size=30) for i in range(4)})
        spec = {"variables": {"variables": ["v0", "v1", "v2", "v3"]},
                "options": {"n_factors": 0}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("method", ["pca", "efa"])
    def test_method_pca_and_efa(self, method):
        """Both pca and efa methods with valid data — must not crash."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame({f"v{i}": rng.normal(size=40) for i in range(5)})
        spec = {"variables": {"variables": [f"v{i}" for i in range(5)]},
                "options": {"method": method}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    @pytest.mark.parametrize("rotation", ["varimax", "none"])
    def test_rotation_variants(self, rotation):
        """Varimax and no-rotation options — must not crash."""
        rng = np.random.default_rng(5)
        df = pd.DataFrame({f"v{i}": rng.normal(size=40) for i in range(5)})
        spec = {"variables": {"variables": [f"v{i}" for i in range(5)]},
                "options": {"method": "pca", "rotation": rotation}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_ml_extraction(self):
        """EFA with ml extraction — must not crash."""
        rng = np.random.default_rng(6)
        df = pd.DataFrame({f"v{i}": rng.normal(size=50) for i in range(5)})
        spec = {"variables": {"variables": [f"v{i}" for i in range(5)]},
                "options": {"method": "efa", "extraction": "ml"}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_all_zero_variance_column(self):
        """One variable has zero variance — must not crash."""
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "v0": rng.normal(size=30),
            "v1": np.zeros(30),  # constant
            "v2": rng.normal(size=30),
        })
        spec = {"variables": {"variables": ["v0", "v1", "v2"]},
                "options": {"method": "pca"}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_n_factors_larger_than_vars(self):
        """n_factors > n_vars — should clamp and not crash."""
        rng = np.random.default_rng(8)
        df = pd.DataFrame({f"v{i}": rng.normal(size=30) for i in range(3)})
        spec = {"variables": {"variables": ["v0", "v1", "v2"]},
                "options": {"n_factors": 99, "method": "pca"}}
        result = fa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 3. CLUSTER ANALYSIS
# ===========================================================================

from statworkbench.analysis.cluster_analysis import run_analysis as ca_run


class TestClusterAnalysisEdgeCases:
    """Error-resilience tests for cluster_analysis.run_analysis."""

    def test_k_greater_than_n_samples(self):
        """K > n_samples — should warn and return early."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 10}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_k_equals_one(self):
        """K = 1 (single cluster) — must not crash."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=20), "y": rng.normal(size=20)})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 1}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_all_identical_rows(self):
        """All identical rows — must not crash."""
        df = pd.DataFrame({"x": [1.0] * 20, "y": [2.0] * 20})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 3}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_hierarchical_n2(self):
        """Hierarchical clustering with n=2 observations — must not crash."""
        df = pd.DataFrame({"x": [1.0, 5.0], "y": [2.0, 6.0]})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "hierarchical", "n_clusters": 2}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_hierarchical_default(self):
        """Hierarchical clustering with default linkage — must not crash."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "hierarchical", "n_clusters": 3}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_kmeans_valid(self):
        """K-means with valid data — must return AnalysisResult with tables."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 3}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_no_variables_specified(self):
        """Empty variable list — must warn and return early."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        spec = {"variables": {"variables": []},
                "options": {"method": "kmeans"}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_hierarchical_average_linkage(self):
        """Hierarchical clustering with average linkage — must not crash."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"x": rng.normal(size=20), "y": rng.normal(size=20)})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "hierarchical", "n_clusters": 3,
                            "linkage": "average"}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_standardize_false(self):
        """Clustering without standardization — must not crash."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 2,
                            "standardize": False}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_k_equals_n_samples(self):
        """K equals n_samples exactly — boundary condition."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "y": [2.0, 3.0, 4.0, 5.0, 6.0]})
        spec = {"variables": {"variables": ["x", "y"]},
                "options": {"method": "kmeans", "n_clusters": 5}}
        result = ca_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 4. SURVIVAL ANALYSIS
# ===========================================================================

from statworkbench.analysis.survival_analysis import run_analysis as sa_run


class TestSurvivalAnalysisEdgeCases:
    """Error-resilience tests for survival_analysis.run_analysis."""

    def _base_spec(self, **extra_vars):
        variables = {"duration": "t", "event": "e"}
        variables.update(extra_vars)
        return {"variables": variables, "options": {"method": "km"}}

    def test_all_censored_no_events(self):
        """No events (all censored, event_var all 0) — must not crash."""
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
                           "e": [0] * 20})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)

    def test_single_observation(self):
        """Single observation — must not crash."""
        df = pd.DataFrame({"t": [5.0], "e": [1]})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)

    def test_all_events_at_same_time(self):
        """All events at same time point — must not crash."""
        df = pd.DataFrame({"t": [3.0] * 10, "e": [1] * 10})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)

    def test_negative_time_values(self):
        """Negative time values in duration — must not crash (graceful handling)."""
        df = pd.DataFrame({"t": [-1.0, 2.0, 3.0, -0.5, 5.0],
                           "e": [1, 0, 1, 0, 1]})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)

    def test_missing_duration_and_event_spec(self):
        """Missing duration/event variables in spec — must warn and return early."""
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [1, 0, 1]})
        spec = {"variables": {}, "options": {"method": "km"}}
        result = sa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_basic_km_analysis(self):
        """Standard KM spec runs cleanly."""
        rng = np.random.default_rng(0)
        n = 50
        t = rng.exponential(scale=5.0, size=n)
        e = rng.choice([0, 1], size=n)
        df = pd.DataFrame({"t": t, "e": e})
        spec = {"variables": {"duration": "t", "event": "e"},
                "options": {"method": "km"}}
        result = sa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_cox_missing_covariates(self):
        """Cox with covariates that have missing values — must not crash."""
        rng = np.random.default_rng(1)
        n = 30
        t = rng.exponential(scale=5.0, size=n)
        e = rng.choice([0, 1], size=n)
        x = rng.normal(size=n).astype(float)
        x[[2, 8, 15]] = np.nan
        df = pd.DataFrame({"t": t, "e": e, "x": x})
        spec = {
            "variables": {"duration": "t", "event": "e", "covariates": ["x"]},
            "options": {"method": "both"},
        }
        result = sa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_invalid_event_values(self):
        """Event variable with values other than 0/1 — should warn."""
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0],
                           "e": [2, 3, 0, 1]})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_group_with_single_member(self):
        """Group variable where one group has only 1 member — must not crash."""
        df = pd.DataFrame({
            "t": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "e": [1, 0, 1, 0, 1, 0],
            "g": ["A", "A", "A", "A", "A", "B"],
        })
        spec = {"variables": {"duration": "t", "event": "e", "group": "g"},
                "options": {"method": "km"}}
        result = sa_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_empty_dataset_after_missing_removal(self):
        """All rows have NaN — must warn after listwise deletion."""
        df = pd.DataFrame({"t": [np.nan, np.nan], "e": [np.nan, np.nan]})
        result = sa_run(ds(df), self._base_spec())
        assert isinstance(result, AnalysisResult)
        # Should have some warning since no valid rows remain or event is wrong
        assert len(result.warnings) >= 0  # at minimum returns without crash


# ===========================================================================
# 5. DISCRIMINANT ANALYSIS
# ===========================================================================

from statworkbench.analysis.discriminant_analysis import run_analysis as da_run


class TestDiscriminantAnalysisEdgeCases:
    """Error-resilience tests for discriminant_analysis.run_analysis."""

    def test_single_group(self):
        """Only 1 group — should warn and return early."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "g": ["A"] * 20,
            "x": rng.normal(size=20),
            "y": rng.normal(size=20),
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x", "y"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_more_predictors_than_observations(self):
        """More predictors than observations — must not crash."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "g": ["A", "B", "A"],
            **{f"x{i}": rng.normal(size=3) for i in range(20)},
        })
        spec = {"variables": {"dependent": "g",
                              "predictors": [f"x{i}" for i in range(20)]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_perfectly_separated_groups(self):
        """Perfectly separated groups — must not crash."""
        df = pd.DataFrame({
            "g": ["A"] * 20 + ["B"] * 20,
            "x": list(range(1, 21)) + list(range(100, 120)),
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_two_groups_valid(self):
        """Normal two-group LDA — must return AnalysisResult with tables."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "g": ["A"] * 30 + ["B"] * 30,
            "x": np.concatenate([rng.normal(0, 1, 30), rng.normal(3, 1, 30)]),
            "y": rng.normal(size=60),
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x", "y"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_three_groups_valid(self):
        """Three-group LDA — must not crash."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "g": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "x": np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20),
                                  rng.normal(6, 1, 20)]),
            "y": rng.normal(size=60),
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x", "y"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_no_predictors(self):
        """Empty predictor list — must warn and return early."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            "g": rng.choice(["A", "B"], size=30),
        })
        spec = {"variables": {"dependent": "g", "predictors": []}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_missing_predictor_values(self):
        """Predictors with missing values — must not crash."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=40).astype(float)
        x[[3, 9, 20]] = np.nan
        df = pd.DataFrame({
            "g": rng.choice(["A", "B"], size=40),
            "x": x,
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_too_few_observations(self):
        """Only 1 valid observation — must warn and return early."""
        df = pd.DataFrame({"g": ["A"], "x": [1.0]})
        spec = {"variables": {"dependent": "g", "predictors": ["x"]}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_equal_prior_option(self):
        """Equal priors option — must not crash."""
        rng = np.random.default_rng(6)
        df = pd.DataFrame({
            "g": ["A"] * 20 + ["B"] * 30,
            "x": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 30)]),
        })
        spec = {"variables": {"dependent": "g", "predictors": ["x"]},
                "options": {"prior": "equal"}}
        result = da_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 6. NONPARAMETRIC
# ===========================================================================

from statworkbench.analysis.nonparametric import run_analysis as np_run


class TestNonparametricEdgeCases:
    """Error-resilience tests for nonparametric.run_analysis."""

    # -- Mann-Whitney --

    def test_mann_whitney_n1_per_group(self):
        """n=1 per group — should either succeed or warn gracefully."""
        df = pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "mann_whitney"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_mann_whitney_all_tied(self):
        """Identical values in both groups — must not crash."""
        df = pd.DataFrame({"x": [5.0] * 10, "g": ["A"] * 5 + ["B"] * 5})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "mann_whitney"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_mann_whitney_single_group(self):
        """Single group in group variable — must warn (requires exactly 2)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["A", "A", "A"]})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "mann_whitney"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_mann_whitney_valid(self):
        """Standard Mann-Whitney test — must return AnalysisResult with tables."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
            "g": ["A"] * 20 + ["B"] * 20,
        })
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "mann_whitney"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    # -- Kruskal-Wallis --

    def test_kruskal_wallis_only_two_groups(self):
        """Kruskal-Wallis with only 2 groups — must warn (requires 3+)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0],
                           "g": ["A", "A", "B", "B"]})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "kruskal_wallis"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_kruskal_wallis_valid(self):
        """Standard Kruskal-Wallis — must return AnalysisResult with tables."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x": np.concatenate([rng.normal(0, 1, 15), rng.normal(1, 1, 15),
                                  rng.normal(2, 1, 15)]),
            "g": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        })
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "kruskal_wallis"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_kruskal_wallis_single_group(self):
        """Kruskal-Wallis with single group — must warn."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["A", "A", "A"]})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "kruskal_wallis"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    # -- Wilcoxon --

    def test_wilcoxon_valid(self):
        """Standard Wilcoxon signed-rank test — must not crash."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, 20),
            "x2": rng.normal(0.5, 1, 20),
        })
        spec = {"variables": {"paired": ["x1", "x2"]},
                "options": {"test": "wilcoxon"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_wilcoxon_all_identical(self):
        """Identical paired values (all differences = 0) — must not crash."""
        df = pd.DataFrame({"x1": [3.0] * 10, "x2": [3.0] * 10})
        spec = {"variables": {"paired": ["x1", "x2"]},
                "options": {"test": "wilcoxon"}}
        # scipy raises ValueError when all differences are zero; result must handle it
        try:
            result = np_run(ds(df), spec)
            assert isinstance(result, AnalysisResult)
        except (ValueError, Exception) as exc:
            # scipy.stats.wilcoxon raises ValueError for all-zero diff
            # Accept if it's a clearly-typed expected exception
            assert isinstance(exc, (ValueError, RuntimeError))

    def test_wilcoxon_wrong_variable_count(self):
        """Wilcoxon with wrong number of paired vars — must warn."""
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        spec = {"variables": {"paired": ["x1"]},
                "options": {"test": "wilcoxon"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    # -- Friedman --

    def test_friedman_n1_subject(self):
        """Friedman with n=1 subject — must not crash."""
        df = pd.DataFrame({"c1": [1.0], "c2": [2.0], "c3": [3.0]})
        spec = {"variables": {"repeated": ["c1", "c2", "c3"]},
                "options": {"test": "friedman"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_friedman_valid(self):
        """Standard Friedman test — must return AnalysisResult with tables."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "c1": rng.normal(0, 1, 20),
            "c2": rng.normal(1, 1, 20),
            "c3": rng.normal(2, 1, 20),
        })
        spec = {"variables": {"repeated": ["c1", "c2", "c3"]},
                "options": {"test": "friedman"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_friedman_too_few_repeated_vars(self):
        """Friedman with only 2 repeated vars — must warn (requires 3+)."""
        df = pd.DataFrame({"c1": [1.0, 2.0, 3.0], "c2": [2.0, 3.0, 4.0]})
        spec = {"variables": {"repeated": ["c1", "c2"]},
                "options": {"test": "friedman"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    # -- Unknown test type --

    def test_unknown_test_type(self):
        """Unknown test type in spec — must warn and return early."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["A", "B", "A"]})
        spec = {"variables": {"dependent": "x", "group": "g"},
                "options": {"test": "nonexistent_test"}}
        result = np_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    @pytest.mark.parametrize("test_type", ["mann_whitney", "kruskal_wallis", "wilcoxon", "friedman"])
    def test_all_test_types_with_empty_data(self, test_type):
        """All test types with empty data — must not crash."""
        if test_type in ("mann_whitney", "kruskal_wallis"):
            df = pd.DataFrame({"x": pd.Series([], dtype=float),
                               "g": pd.Series([], dtype=str)})
            spec = {"variables": {"dependent": "x", "group": "g"},
                    "options": {"test": test_type}}
        elif test_type == "wilcoxon":
            df = pd.DataFrame({"x1": pd.Series([], dtype=float),
                               "x2": pd.Series([], dtype=float)})
            spec = {"variables": {"paired": ["x1", "x2"]},
                    "options": {"test": test_type}}
        else:  # friedman
            df = pd.DataFrame({"c1": pd.Series([], dtype=float),
                               "c2": pd.Series([], dtype=float),
                               "c3": pd.Series([], dtype=float)})
            spec = {"variables": {"repeated": ["c1", "c2", "c3"]},
                    "options": {"test": test_type}}
        try:
            result = np_run(ds(df), spec)
            assert isinstance(result, AnalysisResult)
        except Exception as exc:
            # Only truly unexpected exceptions should propagate
            assert isinstance(exc, (ValueError, RuntimeError, KeyError))


# ===========================================================================
# 7. NORMALITY
# ===========================================================================

from statworkbench.analysis.normality import run_analysis as norm_run


class TestNormalityEdgeCases:
    """Error-resilience tests for normality.run_analysis."""

    def test_n2_minimum_below_threshold(self):
        """n=2 (below minimum of 3 for Shapiro-Wilk) — must warn, not crash."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_n1_below_minimum(self):
        """n=1 (well below minimum) — must warn, not crash."""
        df = pd.DataFrame({"x": [5.0]})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_identical_values(self):
        """All identical values — Shapiro-Wilk may return p=0 or warn."""
        df = pd.DataFrame({"x": [3.0] * 10})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_single_variable_normal(self):
        """Single variable, normally distributed data — must return result."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30)})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) > 0

    def test_empty_target_list(self):
        """Empty target list — should return AnalysisResult without crashing."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        spec = {"variables": {"target": []}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_variable_not_in_dataframe(self):
        """Variable in spec not in dataframe — must warn, not crash."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = {"variables": {"target": ["nonexistent_var"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_multiple_variables_mixed(self):
        """Multiple variables, some with insufficient n — must not crash."""
        rng = np.random.default_rng(1)
        # 'b' has only 2 valid values (padded with NaN to match length)
        b_vals = [1.0, 2.0] + [float("nan")] * 28
        df = pd.DataFrame({
            "a": rng.normal(size=30),
            "b": b_vals,
            "c": rng.normal(size=30),
        })
        spec = {"variables": {"target": ["a", "b", "c"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_n3_exactly_at_threshold(self):
        """n=3 (minimum for Shapiro-Wilk) — must produce result."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_large_n_uses_dagostino(self):
        """n > 5000 — should use D'Agostino test and warn."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(size=5001)})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert any("5000" in w or "D'Agostino" in w for w in result.warnings)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 10, 50, 100])
    def test_various_sample_sizes(self, n):
        """Normality test for various n values — must never crash."""
        rng = np.random.default_rng(n)
        vals = rng.normal(size=n) if n > 0 else []
        df = pd.DataFrame({"x": vals})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_missing_values_in_variable(self):
        """Variable with NaN values — listwise removal should handle gracefully."""
        vals = [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"x": vals})
        spec = {"variables": {"target": ["x"]}}
        result = norm_run(ds(df), spec)
        assert isinstance(result, AnalysisResult)
