"""Tests for sensitivity_specificity analysis module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.sensitivity_specificity import (
    _compute_2x2,
    _diagnostic_metrics,
    run_analysis,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_dataset() -> Dataset:
    """Standard 2×2 dataset: 60% sensitivity, 80% specificity."""
    df = pd.DataFrame({
        "disease": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "test":    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    })
    ds = Dataset(df, name="DiagTest")
    ds.variables["disease"].measure = MeasureType.BINARY
    ds.variables["test"].measure = MeasureType.BINARY
    return ds


@pytest.fixture
def perfect_dataset() -> Dataset:
    """Perfect classifier: all TP and TN, no errors."""
    df = pd.DataFrame({
        "disease": [1, 1, 1, 0, 0, 0],
        "test":    [1, 1, 1, 0, 0, 0],
    })
    return Dataset(df, name="Perfect")


@pytest.fixture
def all_wrong_dataset() -> Dataset:
    """Worst classifier: all FP and FN."""
    df = pd.DataFrame({
        "disease": [1, 1, 1, 0, 0, 0],
        "test":    [0, 0, 0, 1, 1, 1],
    })
    return Dataset(df, name="AllWrong")


@pytest.fixture
def missing_dataset() -> Dataset:
    """Dataset with NaN rows that should be excluded."""
    df = pd.DataFrame({
        "disease": [1, 1, None, 0, 0],
        "test":    [1, 0,    1, 0, 0],
    })
    return Dataset(df, name="WithMissing")


# ---------------------------------------------------------------------------
# _compute_2x2
# ---------------------------------------------------------------------------

class TestCompute2x2:

    def test_basic_cells(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])
        cells = _compute_2x2(y_true, y_pred)
        assert cells["TP"] == 2
        assert cells["FN"] == 1
        assert cells["FP"] == 1
        assert cells["TN"] == 2

    def test_perfect_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        cells = _compute_2x2(y_true, y_pred)
        assert cells["TP"] == 2
        assert cells["FP"] == 0
        assert cells["FN"] == 0
        assert cells["TN"] == 2

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        cells = _compute_2x2(y_true, y_pred)
        assert cells["TP"] == 0
        assert cells["FP"] == 2
        assert cells["FN"] == 2
        assert cells["TN"] == 0

    def test_custom_pos_label(self):
        y_true = np.array([2, 2, 1, 1])
        y_pred = np.array([2, 1, 2, 1])
        cells = _compute_2x2(y_true, y_pred, pos_label=2)
        assert cells["TP"] == 1
        assert cells["FP"] == 1
        assert cells["FN"] == 1
        assert cells["TN"] == 1

    def test_cell_sum_equals_n(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        cells = _compute_2x2(y_true, y_pred)
        assert cells["TP"] + cells["FP"] + cells["FN"] + cells["TN"] == len(y_true)


# ---------------------------------------------------------------------------
# _diagnostic_metrics
# ---------------------------------------------------------------------------

class TestDiagnosticMetrics:

    @pytest.fixture
    def balanced_cells(self):
        # TP=3, FP=1, FN=2, TN=4 → n=10
        return {"TP": 3, "FP": 1, "FN": 2, "TN": 4}

    @pytest.fixture
    def perfect_cells(self):
        return {"TP": 5, "FP": 0, "FN": 0, "TN": 5}

    @pytest.fixture
    def zero_cells(self):
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    def test_sensitivity_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert abs(m["sensitivity"] - 3 / 5) < 1e-10

    def test_specificity_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert abs(m["specificity"] - 4 / 5) < 1e-10

    def test_ppv_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert abs(m["ppv"] - 3 / 4) < 1e-10

    def test_npv_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert abs(m["npv"] - 4 / 6) < 1e-10

    def test_accuracy_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert abs(m["accuracy"] - 7 / 10) < 1e-10

    def test_f1_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        expected = 2 * 3 / (2 * 3 + 1 + 2)
        assert abs(m["f1"] - expected) < 1e-10

    def test_youden_j(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        expected = 3 / 5 + 4 / 5 - 1.0
        assert abs(m["youden_j"] - expected) < 1e-10

    def test_mcc_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        tp, fp, fn, tn = 3, 1, 2, 4
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        expected = (tp * tn - fp * fn) / denom
        assert abs(m["mcc"] - expected) < 1e-10

    def test_kappa_value(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert not math.isnan(m["kappa"])
        assert -1.0 <= m["kappa"] <= 1.0

    def test_ci_bounds_in_range(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        for key in ["sensitivity_ci", "specificity_ci", "ppv_ci", "npv_ci", "accuracy_ci"]:
            lo, hi = m[key]
            assert 0.0 <= lo <= hi <= 1.0, f"{key} CI out of range: [{lo}, {hi}]"

    def test_perfect_classifier_sensitivity_1(self, perfect_cells):
        m = _diagnostic_metrics(perfect_cells)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["accuracy"] == 1.0

    def test_perfect_odds_ratio_inf(self, perfect_cells):
        m = _diagnostic_metrics(perfect_cells)
        assert math.isinf(m["odds_ratio"])

    def test_zero_n_returns_nan(self, zero_cells):
        m = _diagnostic_metrics(zero_cells)
        assert math.isnan(m["sensitivity"])
        assert math.isnan(m["specificity"])
        assert math.isnan(m["accuracy"])

    def test_lr_positive_finite_for_normal(self, balanced_cells):
        m = _diagnostic_metrics(balanced_cells)
        assert math.isfinite(m["lr_pos"])
        assert math.isfinite(m["lr_neg"])


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:

    def test_valid_spec_returns_empty(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        errors = validate(binary_dataset, spec)
        assert errors == []

    def test_missing_outcome_error(self, binary_dataset):
        spec = {"variables": {"predictor": "test"}}
        errors = validate(binary_dataset, spec)
        assert any("outcome" in e for e in errors)

    def test_missing_predictor_error(self, binary_dataset):
        spec = {"variables": {"outcome": "disease"}}
        errors = validate(binary_dataset, spec)
        assert any("predictor" in e for e in errors)

    def test_nonexistent_column_error(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "nonexistent"}}
        errors = validate(binary_dataset, spec)
        assert any("nonexistent" in e for e in errors)

    def test_both_missing_two_errors(self, binary_dataset):
        spec = {"variables": {}}
        errors = validate(binary_dataset, spec)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# run_analysis()
# ---------------------------------------------------------------------------

class TestRunAnalysis:

    def test_returns_analysis_result(self, binary_dataset):
        from statworkbench.analysis.result import AnalysisResult
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        assert isinstance(result, AnalysisResult)

    def test_five_tables_produced(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.tables) == 5

    def test_table_titles(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        titles = [t.title for t in result.tables]
        assert "Case Processing Summary" in titles
        assert "2×2 Contingency Table" in titles
        assert "Diagnostic Accuracy Measures" in titles
        assert "Likelihood Ratios" in titles
        assert "Agreement Statistics" in titles

    def test_case_processing_summary_n(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        cps = result.tables[0].dataframe
        total_row = cps[cps[""] == "Total"]
        assert int(total_row["N"].values[0]) == 10

    def test_sensitivity_matches_known(self, binary_dataset):
        """Fixture: TP=3, FN=2 → sensitivity=0.6."""
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        acc_table = next(t for t in result.tables if t.title == "Diagnostic Accuracy Measures")
        sens_row = acc_table.dataframe[acc_table.dataframe["Measure"].str.contains("Sensitivity")]
        sens_val = float(sens_row["Value"].values[0])
        assert abs(sens_val - 0.6) < 0.001

    def test_specificity_matches_known(self, binary_dataset):
        """Fixture: TN=4, FP=1 → specificity=0.8."""
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        acc_table = next(t for t in result.tables if t.title == "Diagnostic Accuracy Measures")
        spec_row = acc_table.dataframe[acc_table.dataframe["Measure"].str.contains("Specificity")]
        spec_val = float(spec_row["Value"].values[0])
        assert abs(spec_val - 0.8) < 0.001

    def test_missing_outcome_var_warning(self, binary_dataset):
        spec = {"variables": {"predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_missing_predictor_var_warning(self, binary_dataset):
        spec = {"variables": {"outcome": "disease"}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.warnings) > 0

    def test_nonexistent_column_warning(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "ghost"}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.warnings) > 0

    def test_missing_rows_excluded(self, missing_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(missing_dataset, spec)
        cps = result.tables[0].dataframe
        excluded_row = cps[cps[""] == "Excluded"]
        assert int(excluded_row["N"].values[0]) == 1

    def test_perfect_classifier_no_warnings(self, perfect_dataset):
        perfect_dataset.data.columns = ["disease", "test"]
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(perfect_dataset, spec)
        assert len(result.warnings) == 0
        assert len(result.tables) == 5

    def test_all_wrong_classifier_no_crash(self, all_wrong_dataset):
        all_wrong_dataset.data.columns = ["disease", "test"]
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(all_wrong_dataset, spec)
        assert len(result.tables) == 5

    def test_custom_confidence_level(self, binary_dataset):
        spec = {
            "variables": {"outcome": "disease", "predictor": "test"},
            "confidence_level": 0.99,
        }
        result = run_analysis(binary_dataset, spec)
        acc_table = next(t for t in result.tables if t.title == "Diagnostic Accuracy Measures")
        assert "99% CI Lower" in acc_table.dataframe.columns

    def test_multi_category_outcome_warning(self):
        """3-category outcome → warning, no tables."""
        df = pd.DataFrame({
            "outcome": [0, 1, 2, 0, 1, 2],
            "test":    [0, 1, 1, 0, 0, 1],
        })
        ds = Dataset(df, name="MultiCat")
        spec = {"variables": {"outcome": "outcome", "predictor": "test"}}
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0

    def test_all_nan_returns_warning(self):
        df = pd.DataFrame({
            "outcome": [None, None, None],
            "test":    [None, None, None],
        })
        ds = Dataset(df, name="AllNaN")
        spec = {"variables": {"outcome": "outcome", "predictor": "test"}}
        result = run_analysis(ds, spec)
        assert any("유효한" in w for w in result.warnings)

    def test_result_notes_not_empty(self, binary_dataset):
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = run_analysis(binary_dataset, spec)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:

    def test_sensitivity_specificity_in_registry(self):
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        assert "sensitivity_specificity" in registry

    def test_sensitivity_specificity_is_implemented(self):
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        plugin = registry.get("sensitivity_specificity")
        assert plugin.implemented is True

    def test_sensitivity_specificity_not_in_planned(self):
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        planned_ids = [p.id for p in registry.list_planned()]
        assert "sensitivity_specificity" not in planned_ids

    def test_no_planned_analyses(self):
        """v3.7.0: 모든 분석이 구현됨 — planned 목록 비어 있어야 함."""
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        planned_ids = {p.id for p in registry.list_planned()}
        assert planned_ids == set(), f"미구현 분석이 남아 있음: {planned_ids}"

    def test_plugin_category(self):
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        plugin = registry.get("sensitivity_specificity")
        assert plugin.category == "Diagnostic Tests"

    def test_plugin_run_delegates_to_module(self, binary_dataset):
        from statworkbench.analysis.registry import AnalysisRegistry
        registry = AnalysisRegistry()
        plugin = registry.get("sensitivity_specificity")
        spec = {"variables": {"outcome": "disease", "predictor": "test"}}
        result = plugin.run(binary_dataset, spec)
        assert len(result.tables) == 5
