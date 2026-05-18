"""Tests for correlation analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.correlation import run_analysis


@pytest.fixture
def correlation_dataset():
    """Create dataset for correlation testing."""
    np.random.seed(42)
    n = 50
    x = np.random.normal(50, 10, n)
    y = x * 0.8 + np.random.normal(0, 5, n)
    z = x * 0.3 + np.random.normal(0, 8, n)
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
    })
    ds = Dataset(df, name="CorrTest")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["z"].measure = MeasureType.SCALE
    return ds


class TestCorrelation:
    """Test correlation analysis."""

    def test_basic_pearson(self, correlation_dataset):
        """Test basic Pearson correlation."""
        spec = {
            "variables": {"target": ["x", "y", "z"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        assert result.id == "correlation"
        # Should have: CPS + corr matrix + p matrix + N matrix + pairwise
        assert len(result.tables) >= 5

    def test_correlation_matrix(self, correlation_dataset):
        """Test correlation matrix values."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        corr_table = [t for t in result.tables if "Correlation Matrix" in t.title][0]
        df = corr_table.dataframe
        r_xy = df.loc["x", "y"]
        assert abs(r_xy) > 0.5  # Strong positive correlation

    def test_vs_scipy(self, correlation_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        x = correlation_dataset.data["x"].values
        y = correlation_dataset.data["y"].values
        r_scipy, p_scipy = stats.pearsonr(x, y)

        corr_table = [t for t in result.tables if "Correlation Matrix" in t.title][0]
        df = corr_table.dataframe
        r_result = df.loc["x", "y"]
        assert abs(r_result - r_scipy) < 0.001

    def test_spearman(self, correlation_dataset):
        """Test Spearman correlation."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "spearman", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        corr_table = [t for t in result.tables if "Correlation Matrix" in t.title][0]
        df = corr_table.dataframe
        r_spearman = df.loc["x", "y"]

        x = correlation_dataset.data["x"].values
        y = correlation_dataset.data["y"].values
        r_scipy, _ = stats.spearmanr(x, y)
        assert abs(r_spearman - r_scipy) < 0.001

    def test_kendall(self, correlation_dataset):
        """Test Kendall correlation."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "kendall", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        corr_table = [t for t in result.tables if "Correlation Matrix" in t.title][0]
        df = corr_table.dataframe
        r_kendall = df.loc["x", "y"]

        x = correlation_dataset.data["x"].values
        y = correlation_dataset.data["y"].values
        r_scipy, _ = stats.kendalltau(x, y)
        assert abs(r_kendall - r_scipy) < 0.001

    def test_pairwise_missing(self):
        """Test pairwise deletion with missing data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, np.nan, 5.0],
            "y": [2.0, np.nan, 4.0, 5.0, 6.0],
            "z": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds = Dataset(df, name="MissingCorr")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE
        ds.variables["z"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x", "y", "z"]},
            "options": {"method": "pearson", "pairwise": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.PAIRWISE,
        }
        result = run_analysis(ds, spec)

        n_table = [t for t in result.tables if "N Matrix" in t.title][0]
        df_n = n_table.dataframe
        # x-y pair: rows 0, 2, 4 have both values -> N=3
        assert df_n.loc["x", "y"] == 3
        # x-z pair: all rows have both -> N=4 (excluding row with x=nan)
        assert df_n.loc["x", "z"] == 4

    def test_significance_flags(self, correlation_dataset):
        """Test significance flagging."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "flag_significant": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        pairwise_table = [t for t in result.tables if "Pairwise" in t.title][0]
        df = pairwise_table.dataframe
        assert "Significance" in df.columns
        assert "***" in str(df["Significance"].values) or "**" in str(df["Significance"].values)

    def test_ci_for_pearson(self, correlation_dataset):
        """Test that CI is computed for Pearson correlation."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        pairwise_table = [t for t in result.tables if "Pairwise" in t.title][0]
        df = pairwise_table.dataframe
        assert "CI" in df.columns
        ci_val = str(df.iloc[0]["CI"])
        assert "[" in ci_val and "]" in ci_val

    def test_no_ci_for_spearman(self, correlation_dataset):
        """Test that CI is not computed for Spearman."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "spearman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(correlation_dataset, spec)

        pairwise_table = [t for t in result.tables if "Pairwise" in t.title][0]
        df = pairwise_table.dataframe
        ci_val = str(df.iloc[0]["CI"])
        assert ci_val == "" or ci_val == "nan"

    def test_small_sample(self):
        """Test with very small sample."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
        })
        ds = Dataset(df, name="SmallCorr")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        corr_table = [t for t in result.tables if "Correlation Matrix" in t.title][0]
        df_corr = corr_table.dataframe
        # Perfect correlation
        assert abs(df_corr.loc["x", "y"] - 1.0) < 0.001

    def test_single_variable_warning(self):
        """Test warning when only one variable provided."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
        })
        ds = Dataset(df, name="Single")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x"]},
            "options": {"method": "pearson"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        assert len(result.warnings) >= 1
        assert "at least 2" in str(result.warnings[0]).lower()
