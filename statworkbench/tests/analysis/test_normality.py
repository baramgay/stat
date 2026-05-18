"""Tests for normality test analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.normality import run_analysis


@pytest.fixture
def normal_dataset():
    """Create a dataset with normally distributed data."""
    np.random.seed(42)
    df = pd.DataFrame({
        "normal": np.random.normal(100, 15, 50),
        "skewed": np.random.exponential(2, 50),
    })
    ds = Dataset(df, name="NormTest")
    ds.variables["normal"].measure = MeasureType.SCALE
    ds.variables["skewed"].measure = MeasureType.SCALE
    return ds


class TestNormality:
    """Test normality analysis."""

    def test_normal_data(self, normal_dataset):
        """Test that normal data passes normality test."""
        spec = {
            "variables": {"target": ["normal"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(normal_dataset, spec)

        assert result.id == "normality_test"
        assert len(result.tables) >= 2

        norm_table = result.tables[1]
        row = norm_table.dataframe.iloc[0]
        assert row["N"] == 50
        assert "Shapiro" in str(row["Interpretation"]) or "D'Agostino" in str(row["Interpretation"])

    def test_skewed_data(self, normal_dataset):
        """Test that skewed data fails normality test."""
        spec = {
            "variables": {"target": ["skewed"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(normal_dataset, spec)

        norm_table = result.tables[1]
        row = norm_table.dataframe.iloc[0]
        # Exponential data should have low p-value
        p_str = str(row["p-value"])
        assert "*" in p_str or "<" in p_str

    def test_small_sample(self):
        """Test with very small sample."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
        })
        ds = Dataset(df, name="Small")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        # Should have a warning about low power
        assert len(result.warnings) >= 1

    def test_with_missing(self):
        """Test with missing data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        ds = Dataset(df, name="Missing")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        norm_table = result.tables[1]
        row = norm_table.dataframe.iloc[0]
        assert row["N"] == 7  # 8 - 1 NaN

    def test_multiple_variables(self, normal_dataset):
        """Test normality test with multiple variables."""
        spec = {
            "variables": {"target": ["normal", "skewed"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(normal_dataset, spec)

        norm_table = result.tables[1]
        df = norm_table.dataframe
        assert len(df) == 2
        assert set(df["Variable"]) == {"normal", "skewed"}

    def test_too_small_warning(self):
        """Test that N < 3 gives appropriate warning."""
        df = pd.DataFrame({
            "x": [1.0, 2.0],
        })
        ds = Dataset(df, name="Tiny")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        norm_table = result.tables[1]
        row = norm_table.dataframe.iloc[0]
        assert "Insufficient" in str(row["Interpretation"])
