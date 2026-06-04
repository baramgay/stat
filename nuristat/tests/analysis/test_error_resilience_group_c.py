"""Error-resilience and edge-case tests for Group C analysis modules.

Covers:
  - roc_analysis.run_analysis
  - sensitivity_specificity.run_analysis
  - cohens_kappa.run_analysis
  - icc.run_analysis
  - bland_altman.run_analysis
  - reliability.run_analysis
  - partial_correlation.run_analysis
  - chi_square_gof.run_analysis

Contract: every call must either return AnalysisResult OR raise a typed
NuriStatError subclass.  No bare exceptions allowed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import NuriStatError


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ds(df: pd.DataFrame, name: str = "test") -> Dataset:
    """Wrap a DataFrame in a Dataset."""
    return Dataset(df, name=name)


# ===========================================================================
# 1. ROC Analysis
# ===========================================================================

class TestROCEdgeCases:

    def _spec(self, state: str, test: list[str], positive_value=1, max_coords: int = 20) -> dict:
        return {
            "variables": {
                "state": state,
                "test": test,
                "positive_value": positive_value,
            },
            "options": {"max_coords": max_coords},
        }

    def test_all_predictions_same_value(self):
        """All predictions = 0.5 — no discrimination.  Should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({
            "label": [0, 0, 1, 1, 0, 1],
            "score": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        })
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)

    def test_predicted_prob_all_zero(self):
        """All predictions = 0.0."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({
            "label": [0, 0, 1, 1, 0, 1],
            "score": [0.0] * 6,
        })
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)

    def test_predicted_prob_all_one(self):
        """All predictions = 1.0."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({
            "label": [0, 0, 1, 1, 0, 1],
            "score": [1.0] * 6,
        })
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)

    def test_n_equals_2(self):
        """Minimum sample size n=2."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8]})
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)

    def test_outcome_single_class(self):
        """Outcome has only one class — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [1, 1, 1, 1], "score": [0.1, 0.4, 0.7, 0.9]})
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_non_binary_outcome_3_classes(self):
        """Outcome has 3+ classes — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({
            "label": [0, 1, 2, 0, 1, 2],
            "score": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
        })
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_missing_state_variable(self):
        """state not in spec — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [0, 1], "score": [0.3, 0.7]})
        spec = {"variables": {"state": "", "test": ["score"]}}
        result = run_analysis(_ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_missing_test_variables(self):
        """test list empty — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [0, 1], "score": [0.3, 0.7]})
        spec = {"variables": {"state": "label", "test": []}}
        result = run_analysis(_ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_column_not_found(self):
        """Non-existent column — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [0, 1], "score": [0.3, 0.7]})
        result = run_analysis(_ds(df), self._spec("label", ["nonexistent"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_data_missing_after_dropna(self):
        """All data NaN after drop — should return result with warning."""
        from nuristat.analysis.roc_analysis import run_analysis

        df = pd.DataFrame({"label": [np.nan, np.nan], "score": [np.nan, np.nan]})
        result = run_analysis(_ds(df), self._spec("label", ["score"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ===========================================================================
# 2. Sensitivity / Specificity
# ===========================================================================

class TestSensSpecEdgeCases:

    def _spec(self, outcome: str, predictor: str, pos_label: int = 1) -> dict:
        return {
            "variables": {"outcome": outcome, "predictor": predictor},
            "options": {"pos_label": pos_label},
        }

    def test_all_true_positives(self):
        """TP=N, TN=0 — sensitivity=1, specificity undefined."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [1, 1, 1, 1, 0, 0],
            "predictor": [1, 1, 1, 1, 1, 1],   # all predicted positive
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_all_true_negatives(self):
        """TP=0, TN=N."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [1, 1, 0, 0, 0, 0],
            "predictor": [0, 0, 0, 0, 0, 0],   # all predicted negative
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_all_false_positives_zero_sensitivity(self):
        """FP=N, TP=0 — sensitivity=0."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [0, 0, 0, 0, 0, 1, 1],
            "predictor": [1, 1, 1, 1, 1, 0, 0],
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_all_false_negatives_zero_specificity(self):
        """FN=N, TN=0 — specificity=0."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [1, 1, 1, 0, 0],
            "predictor": [0, 0, 0, 1, 1],
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_n_equals_1(self):
        """n=1 case."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({"outcome": [1], "predictor": [1]})
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_mismatched_column_lengths_with_nan(self):
        """Extra NaN rows — listwise drops them, should handle cleanly."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [1, 0, 1, 0, np.nan, np.nan],
            "predictor": [1, 0, 0, 1, 1,      np.nan],
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)

    def test_missing_outcome_variable(self):
        """outcome variable not specified."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({"outcome": [1, 0], "predictor": [1, 0]})
        spec = {"variables": {"outcome": "", "predictor": "predictor"}, "options": {}}
        result = run_analysis(_ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_nonbinary_outcome(self):
        """outcome with 3+ unique values."""
        from nuristat.analysis.sensitivity_specificity import run_analysis

        df = pd.DataFrame({
            "outcome":   [0, 1, 2, 0, 1],
            "predictor": [0, 1, 1, 0, 1],
        })
        result = run_analysis(_ds(df), self._spec("outcome", "predictor"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ===========================================================================
# 3. Cohen's Kappa
# ===========================================================================

class TestCohensKappaEdgeCases:

    def _spec(self, r1: str, r2: str) -> dict:
        return {"variables": {"rater1": r1, "rater2": r2}}

    def test_perfect_agreement(self):
        """All ratings match — kappa=1."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({"r1": ["A", "B", "A", "B"], "r2": ["A", "B", "A", "B"]})
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)

    def test_zero_agreement(self):
        """All ratings differ — kappa near -1 or 0."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({"r1": ["A", "A", "B", "B"], "r2": ["B", "B", "A", "A"]})
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)

    def test_single_category(self):
        """Only category 'A' in both raters — should handle gracefully."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({"r1": ["A", "A", "A"], "r2": ["A", "A", "A"]})
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)
        # Should warn or return with note about single category
        assert len(result.warnings) > 0

    def test_three_plus_categories(self):
        """3 categories — kappa still computable."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({
            "r1": ["A", "B", "C", "A", "B", "C", "A"],
            "r2": ["A", "B", "C", "B", "A", "C", "A"],
        })
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)

    def test_missing_values_in_rater_columns(self):
        """NaN in rater columns — listwise drop."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({
            "r1": ["A", "B", np.nan, "A", "B"],
            "r2": ["A", np.nan, "B", "A", "B"],
        })
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)

    def test_missing_rater_variables(self):
        """rater1/rater2 not specified."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({"r1": ["A", "B"], "r2": ["A", "B"]})
        spec = {"variables": {"rater1": "", "rater2": ""}}
        result = run_analysis(_ds(df), spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_column_not_found(self):
        """Rater column does not exist in dataset."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({"r1": ["A", "B"], "r2": ["A", "B"]})
        result = run_analysis(_ds(df), self._spec("r1", "nonexistent"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_too_few_valid_cases(self):
        """After dropna, fewer than 2 valid cases."""
        from nuristat.analysis.cohens_kappa import run_analysis

        df = pd.DataFrame({
            "r1": [np.nan, np.nan, "A"],
            "r2": [np.nan, "A",    np.nan],
        })
        result = run_analysis(_ds(df), self._spec("r1", "r2"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ===========================================================================
# 4. ICC
# ===========================================================================

class TestICCEdgeCases:

    def _spec(self, target: list[str], model: str = "twoway_mixed") -> dict:
        return {"variables": {"target": target}, "options": {"model": model}}

    def test_perfect_agreement(self):
        """All raters give identical scores — ICC=1."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "r2": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"]))
        assert isinstance(result, AnalysisResult)

    def test_maximal_disagreement(self):
        """Raters give opposite extremes."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "r2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"]))
        assert isinstance(result, AnalysisResult)

    def test_n_equals_2_subjects(self):
        """Minimum subjects n=2."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({"r1": [1.0, 2.0], "r2": [1.5, 2.5]})
        result = run_analysis(_ds(df), self._spec(["r1", "r2"]))
        assert isinstance(result, AnalysisResult)

    def test_single_rater_should_handle_gracefully(self):
        """Only 1 rater — insufficient for ICC, should warn."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({"r1": [1.0, 2.0, 3.0]})
        result = run_analysis(_ds(df), self._spec(["r1"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_negative_values(self):
        """Negative values in rater scores."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [-3.0, -1.0, 0.0, 1.0, 3.0],
            "r2": [-2.5, -0.8, 0.2, 1.2, 2.8],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"]))
        assert isinstance(result, AnalysisResult)

    def test_oneway_random_model(self):
        """ICC(1,1) oneway_random model."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [4.0, 3.0, 5.0, 2.0, 4.0],
            "r2": [4.5, 3.5, 4.5, 2.5, 4.0],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"], model="oneway_random"))
        assert isinstance(result, AnalysisResult)

    def test_twoway_random_model(self):
        """ICC(2,1) twoway_random model."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [4.0, 3.0, 5.0, 2.0, 4.0],
            "r2": [4.5, 3.5, 4.5, 2.5, 4.0],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"], model="twoway_random"))
        assert isinstance(result, AnalysisResult)

    def test_missing_variables(self):
        """No target variables specified."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({"r1": [1.0, 2.0], "r2": [1.5, 2.5]})
        result = run_analysis(_ds(df), {"variables": {"target": []}, "options": {}})
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_too_few_valid_cases(self):
        """All rows NaN after dropna."""
        from nuristat.analysis.icc import run_analysis

        df = pd.DataFrame({
            "r1": [np.nan, np.nan],
            "r2": [np.nan, np.nan],
        })
        result = run_analysis(_ds(df), self._spec(["r1", "r2"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ===========================================================================
# 5. Bland-Altman
# ===========================================================================

class TestBlandAltmanEdgeCases:

    def _spec(self, m1: str, m2: str) -> dict:
        return {"variables": {"method1": m1, "method2": m2}}

    def test_identical_methods_zero_diff(self):
        """method1 == method2 everywhere — bias=0."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({"m1": [1.0, 2.0, 3.0, 4.0, 5.0], "m2": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = run_analysis(_ds(df), self._spec("m1", "m2"))
        assert isinstance(result, AnalysisResult)

    def test_large_systematic_bias(self):
        """method1 consistently 10 higher than method2."""
        from nuristat.analysis.bland_altman import run_analysis

        vals = np.arange(1.0, 11.0)
        df = pd.DataFrame({"m1": vals + 10.0, "m2": vals})
        result = run_analysis(_ds(df), self._spec("m1", "m2"))
        assert isinstance(result, AnalysisResult)

    def test_n_equals_2(self):
        """Minimum n=2."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [1.2, 1.8]})
        result = run_analysis(_ds(df), self._spec("m1", "m2"))
        assert isinstance(result, AnalysisResult)

    def test_one_column_all_nan(self):
        """method2 all NaN — should warn, no valid cases."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({
            "m1": [1.0, 2.0, 3.0],
            "m2": [np.nan, np.nan, np.nan],
        })
        result = run_analysis(_ds(df), self._spec("m1", "m2"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_missing_variable_spec(self):
        """method1 / method2 not specified."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [1.2, 1.8]})
        result = run_analysis(_ds(df), {"variables": {}})
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_column_not_found(self):
        """Non-existent column in spec."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [1.2, 1.8]})
        result = run_analysis(_ds(df), self._spec("m1", "nonexistent"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_n_equals_1_after_dropna(self):
        """Only 1 valid pair after dropna — should warn."""
        from nuristat.analysis.bland_altman import run_analysis

        df = pd.DataFrame({
            "m1": [1.0, np.nan, np.nan],
            "m2": [1.2, np.nan, np.nan],
        })
        result = run_analysis(_ds(df), self._spec("m1", "m2"))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0


# ===========================================================================
# 6. Reliability (Cronbach's Alpha)
# ===========================================================================

class TestReliabilityEdgeCases:

    def _spec(self, target: list[str]) -> dict:
        return {"variables": {"target": target}, "options": {"listwise": True}}

    def test_two_items_minimum(self):
        """2 items — minimum allowed."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({"q1": [1, 2, 3, 4, 5], "q2": [2, 3, 4, 5, 1]})
        result = run_analysis(_ds(df), self._spec(["q1", "q2"]))
        assert isinstance(result, AnalysisResult)

    def test_one_item_below_minimum(self):
        """1 item — below minimum, should warn."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({"q1": [1, 2, 3, 4, 5]})
        result = run_analysis(_ds(df), self._spec(["q1"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_items_perfectly_correlated(self):
        """All items identical — alpha near 1."""
        from nuristat.analysis.reliability import run_analysis

        v = [1, 2, 3, 4, 5, 4, 3, 2, 1, 5]
        df = pd.DataFrame({"q1": v, "q2": v, "q3": v})
        result = run_analysis(_ds(df), self._spec(["q1", "q2", "q3"]))
        assert isinstance(result, AnalysisResult)

    def test_all_items_uncorrelated(self):
        """Uncorrelated random items — alpha near 0 or negative."""
        from nuristat.analysis.reliability import run_analysis

        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "q1": rng.integers(1, 6, n),
            "q2": rng.integers(1, 6, n),
            "q3": rng.integers(1, 6, n),
        })
        result = run_analysis(_ds(df), self._spec(["q1", "q2", "q3"]))
        assert isinstance(result, AnalysisResult)

    def test_missing_values_in_items(self):
        """NaN in item columns — listwise drop."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({
            "q1": [1, 2, np.nan, 4, 5],
            "q2": [2, np.nan, 3, 4, 5],
            "q3": [3, 2, 4, np.nan, 5],
        })
        result = run_analysis(_ds(df), self._spec(["q1", "q2", "q3"]))
        assert isinstance(result, AnalysisResult)

    def test_n_equals_2_subjects(self):
        """Only 2 subjects."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({"q1": [1, 5], "q2": [2, 4]})
        result = run_analysis(_ds(df), self._spec(["q1", "q2"]))
        assert isinstance(result, AnalysisResult)

    def test_column_not_found(self):
        """One item column does not exist."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({"q1": [1, 2, 3], "q2": [2, 3, 4]})
        result = run_analysis(_ds(df), self._spec(["q1", "nonexistent"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_missing_after_listwise(self):
        """All rows dropped after listwise — should warn."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({
            "q1": [np.nan, np.nan, np.nan],
            "q2": [1.0, np.nan, np.nan],
        })
        result = run_analysis(_ds(df), self._spec(["q1", "q2"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_constant_item_zero_variance(self):
        """One item is constant (zero variance) — alpha may be NaN, no crash."""
        from nuristat.analysis.reliability import run_analysis

        df = pd.DataFrame({
            "q1": [3, 3, 3, 3, 3],
            "q2": [1, 2, 3, 4, 5],
        })
        result = run_analysis(_ds(df), self._spec(["q1", "q2"]))
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 7. Partial Correlation
# ===========================================================================

class TestPartialCorrelationEdgeCases:

    def _spec(self, target: list[str], controlling: list[str] | None = None) -> dict:
        return {
            "variables": {
                "target": target,
                "controlling": controlling or [],
            },
            "options": {"listwise": True},
        }

    def test_controlling_zero_variables(self):
        """Controlling for 0 variables — returns Pearson."""
        from nuristat.analysis.partial_correlation import run_analysis

        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        result = run_analysis(_ds(df), self._spec(["x", "y"]))
        assert isinstance(result, AnalysisResult)

    def test_controlling_all_variables_overconstrained(self):
        """Controlling for variables that make matrix singular."""
        from nuristat.analysis.partial_correlation import run_analysis

        rng = np.random.default_rng(2)
        n = 20
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        z = x + y  # perfectly collinear with x+y
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        result = run_analysis(_ds(df), self._spec(["x", "y"], controlling=["z"]))
        assert isinstance(result, AnalysisResult)

    def test_n_less_than_4(self):
        """n < 4 — insufficient for partial correlation, should warn."""
        from nuristat.analysis.partial_correlation import run_analysis

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 3.0, 4.0],
            "z": [3.0, 4.0, 5.0],
        })
        result = run_analysis(_ds(df), self._spec(["x", "y"], controlling=["z"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_missing_values(self):
        """NaN in variables — listwise drop."""
        from nuristat.analysis.partial_correlation import run_analysis

        rng = np.random.default_rng(3)
        n = 30
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "z": rng.normal(0, 1, n),
        })
        df.loc[0, "x"] = np.nan
        df.loc[5, "z"] = np.nan
        result = run_analysis(_ds(df), self._spec(["x", "y"], controlling=["z"]))
        assert isinstance(result, AnalysisResult)

    def test_too_few_target_variables(self):
        """Only 1 target variable — should warn."""
        from nuristat.analysis.partial_correlation import run_analysis

        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "z": [2, 3, 4, 5, 6]})
        result = run_analysis(_ds(df), self._spec(["x"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_column_not_found(self):
        """Non-existent column in spec."""
        from nuristat.analysis.partial_correlation import run_analysis

        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        result = run_analysis(_ds(df), self._spec(["x", "nonexistent"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_n_equals_4_boundary(self):
        """n=4 is the exact minimum — should succeed."""
        from nuristat.analysis.partial_correlation import run_analysis

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 3.0, 5.0, 4.0],
        })
        result = run_analysis(_ds(df), self._spec(["x", "y"]))
        assert isinstance(result, AnalysisResult)


# ===========================================================================
# 8. Chi-Square GOF
# ===========================================================================

class TestChiSquareGOFEdgeCases:

    def _spec(
        self,
        target: list[str],
        expected_ratios: dict | None = None,
    ) -> dict:
        spec: dict = {
            "variables": {"target": target},
            "options": {},
        }
        if expected_ratios is not None:
            spec["variables"]["expected_ratios"] = expected_ratios
        return spec

    def test_expected_frequencies_dont_sum_correctly(self):
        """expected_ratios with non-summing values — should normalize gracefully."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "A", "B", "B", "B", "C", "C"]})
        # ratios don't sum to 1 — should be normalized
        result = run_analysis(
            _ds(df),
            self._spec(["cat"], expected_ratios={"A": 0.3, "B": 0.4, "C": 0.5}),
        )
        assert isinstance(result, AnalysisResult)

    def test_single_category(self):
        """Only one unique value — should warn, no test performed."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "A", "A", "A"]})
        result = run_analysis(_ds(df), self._spec(["cat"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_observed_frequency_zero_for_some_categories(self):
        """Observed count = 0 for a category via expected_ratios mapping."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "A", "B", "B", "B"]})
        # C is expected but not observed — chi-square should handle
        result = run_analysis(
            _ds(df),
            self._spec(["cat"], expected_ratios={"A": 0.33, "B": 0.33, "C": 0.34}),
        )
        assert isinstance(result, AnalysisResult)

    def test_n_equals_1(self):
        """n=1 single observation."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A"]})
        result = run_analysis(_ds(df), self._spec(["cat"]))
        assert isinstance(result, AnalysisResult)

    def test_no_target_variables(self):
        """Empty target list — should warn."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "B", "C"]})
        result = run_analysis(_ds(df), self._spec([]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_column_not_found(self):
        """Non-existent target column."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "B", "A"]})
        result = run_analysis(_ds(df), self._spec(["nonexistent"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_missing_values(self):
        """All values NaN — no valid cases after dropna."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": [np.nan, np.nan, np.nan]})
        result = run_analysis(_ds(df), self._spec(["cat"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_expected_ratio_sum_zero(self):
        """All expected_ratios = 0 — should fall back to uniform."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
        result = run_analysis(
            _ds(df),
            self._spec(["cat"], expected_ratios={"A": 0.0, "B": 0.0}),
        )
        assert isinstance(result, AnalysisResult)

    def test_normal_two_categories(self):
        """Normal fair case with two categories."""
        from nuristat.analysis.chi_square_gof import run_analysis

        df = pd.DataFrame({"cat": ["A"] * 30 + ["B"] * 20})
        result = run_analysis(_ds(df), self._spec(["cat"]))
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 1
