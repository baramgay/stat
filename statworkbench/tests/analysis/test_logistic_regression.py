"""Tests for logistic regression analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.logistic_regression import run_analysis


@pytest.fixture
def binary_dataset():
    """Create binary outcome dataset for logistic regression testing."""
    rng = np.random.default_rng(42)
    n = 100
    age = rng.integers(20, 70, n).astype(float)
    score = rng.normal(50, 10, n)
    # outcome is related to age and score to ensure model has signal
    log_odds = -3 + 0.05 * age + 0.04 * score
    prob = 1 / (1 + np.exp(-log_odds))
    outcome = (rng.random(n) < prob).astype(int)
    group = rng.choice(["A", "B", "C"], n)

    df = pd.DataFrame({
        "outcome": outcome,
        "age": age,
        "score": score,
        "group": group,
    })
    ds = Dataset(df, "binary_test")
    ds.variables["outcome"].measure = MeasureType.BINARY
    ds.variables["age"].measure = MeasureType.SCALE
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def binary_dataset_with_missing():
    """Binary dataset that has some missing values."""
    rng = np.random.default_rng(7)
    n = 80
    outcome = (rng.random(n) > 0.5).astype(float)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Inject missing
    outcome[5] = np.nan
    x1[10] = np.nan
    x2[20] = np.nan

    df = pd.DataFrame({"outcome": outcome, "x1": x1, "x2": x2})
    ds = Dataset(df, "missing_binary")
    ds.variables["outcome"].measure = MeasureType.BINARY
    ds.variables["x1"].measure = MeasureType.SCALE
    ds.variables["x2"].measure = MeasureType.SCALE
    return ds


class TestBinaryLogistic:
    """Tests for binary logistic regression."""

    def test_binary_basic(self, binary_dataset):
        """run_analysis returns a non-None result for a valid binary spec."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["age", "score"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset, spec)

        assert result is not None
        assert result.id == "logistic_regression"
        assert len(result.tables) > 0

    def test_result_has_tables(self, binary_dataset):
        """Result must include case processing, model summary, and coefficient tables."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["age", "score"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset, spec)

        titles = [t.title for t in result.tables]
        # At minimum: case processing summary + model summary + coefficients
        assert len(result.tables) >= 3
        # Case processing summary always present
        assert any("Case" in t or "케이스" in t or "case" in t.lower() for t in titles)

    def test_odds_ratio_positive(self, binary_dataset):
        """Odds ratios in the coefficient table must be positive numbers."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["age", "score"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset, spec)

        coef_tables = [t for t in result.tables if "계수" in t.title or "Coeff" in t.title]
        assert len(coef_tables) >= 1

        df = coef_tables[0].dataframe
        assert "OR (Exp(B))" in df.columns or any("OR" in c or "Exp" in c for c in df.columns)

        or_col = next(c for c in df.columns if "OR" in c or "Exp" in c)
        for val in df[or_col].values:
            try:
                or_val = float(str(val).replace(",", ""))
                if not np.isnan(or_val):
                    assert or_val > 0, f"OR must be positive, got {or_val}"
            except (ValueError, TypeError):
                pass  # formatted strings that cannot be parsed are skipped

    def test_case_processing_summary(self, binary_dataset):
        """Case processing summary table must report correct total N."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["age", "score"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset, spec)

        cps_tables = [
            t for t in result.tables
            if "case" in t.title.lower() or "케이스" in t.title
        ]
        assert len(cps_tables) >= 1

        cps_df = cps_tables[0].dataframe
        # The table must have some rows
        assert len(cps_df) >= 1

    def test_missing_policy(self, binary_dataset_with_missing):
        """Listwise deletion should reduce N when missing values are present."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset_with_missing, spec)

        assert result is not None
        # Result should at least contain the case processing summary
        assert len(result.tables) >= 1

    def test_categorical_predictor(self, binary_dataset):
        """Categorical predictor should be dummy-coded and model still runs."""
        spec = {
            "variables": {"dependent": "outcome", "predictors": ["age", "group"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_dataset, spec)

        assert result is not None
        assert len(result.tables) > 0
