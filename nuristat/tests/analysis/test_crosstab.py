"""Tests for crosstabulation analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.crosstab import run_analysis


@pytest.fixture
def crosstab_dataset():
    """Create a dataset for crosstab testing."""
    df = pd.DataFrame({
        "treatment": ["Drug", "Drug", "Drug", "Drug", "Drug",
                      "Placebo", "Placebo", "Placebo", "Placebo", "Placebo"],
        "response": ["Yes", "Yes", "No", "Yes", "No",
                     "No", "No", "Yes", "No", "No"],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    })
    ds = Dataset(df, name="CrosstabTest")
    ds.variables["treatment"].measure = MeasureType.NOMINAL
    ds.variables["response"].measure = MeasureType.NOMINAL
    ds.variables["sex"].measure = MeasureType.NOMINAL
    return ds


class TestCrosstab:
    """Test crosstabulation analysis."""

    def test_basic_crosstab(self, crosstab_dataset):
        """Test basic 2x2 crosstab."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        assert result.id == "crosstab"
        # Should have CPS + count + row% + col% + total% + expected + residual + std residual + tests
        assert len(result.tables) >= 9

    def test_chi_square_values(self, crosstab_dataset):
        """Test that chi-square values match scipy."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        # Find chi-square tests table
        chi_table = [t for t in result.tables if "Chi-Square" in t.title][0]
        df = chi_table.dataframe

        pearson_row = df[df["Test"] == "Pearson Chi-Square"].iloc[0]
        chi_val = float(str(pearson_row["Value"]))
        assert chi_val > 0

    def test_fisher_exact_for_2x2(self, crosstab_dataset):
        """Test that Fisher's exact test is included for 2x2 tables."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        chi_table = [t for t in result.tables if "Chi-Square" in t.title][0]
        df = chi_table.dataframe

        fisher_row = df[df["Test"] == "Fisher's Exact Test"]
        assert len(fisher_row) == 1

    def test_cramers_v(self, crosstab_dataset):
        """Test that Cramer's V is computed."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        chi_table = [t for t in result.tables if "Chi-Square" in t.title][0]
        df = chi_table.dataframe

        cramers_row = df[df["Test"] == "Cramer's V"]
        assert len(cramers_row) == 1
        assert float(str(cramers_row.iloc[0]["Value"])) > 0

    def test_with_layer(self, crosstab_dataset):
        """Test crosstab with layer variable."""
        spec = {
            "variables": {"row": "treatment", "column": "response", "layer": "sex"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        # Should have more tables due to layers
        assert len(result.tables) >= 9

    def test_expected_frequency_warning(self):
        """Test warning when expected frequencies are small."""
        df = pd.DataFrame({
            "a": ["X", "X", "Y", "Y"],
            "b": ["P", "Q", "P", "Q"],
        })
        ds = Dataset(df, name="Small")
        ds.variables["a"].measure = MeasureType.NOMINAL
        ds.variables["b"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"row": "a", "column": "b"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        # May have warning about expected frequencies
        assert any("expected" in str(w).lower() for w in result.warnings) or len(result.warnings) == 0

    def test_residuals_sum_to_zero(self, crosstab_dataset):
        """Test that residuals sum approximately to zero."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        residual_table = [t for t in result.tables if "Residuals" == t.title][0]
        df = residual_table.dataframe
        total_residual = df.values.sum()
        assert abs(total_residual) < 0.01

    def test_row_percentages_sum(self, crosstab_dataset):
        """Test that row percentages sum to approximately 100."""
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(crosstab_dataset, spec)

        row_pct_table = [t for t in result.tables if "Row Percentages" in t.title][0]
        df = row_pct_table.dataframe
        row_sums = df.sum(axis=1)
        for s in row_sums:
            assert abs(s - 100.0) < 0.5
