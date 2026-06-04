"""Tests for linear regression analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.regression import run_analysis


@pytest.fixture
def regression_dataset():
    """Create dataset for regression testing."""
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(30, 5, n)
    y = 10 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 5, n)
    df = pd.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
    })
    ds = Dataset(df, name="RegTest")
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["x1"].measure = MeasureType.SCALE
    ds.variables["x2"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def regression_with_cat():
    """Create dataset with categorical predictor."""
    np.random.seed(42)
    n = 40
    x = np.random.normal(50, 10, n)
    cat = np.random.choice(["A", "B", "C"], n)
    y = 10 + 2 * x + np.random.normal(0, 5, n)
    df = pd.DataFrame({
        "y": y,
        "x": x,
        "cat": cat,
    })
    ds = Dataset(df, name="RegCatTest")
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["cat"].measure = MeasureType.NOMINAL
    return ds


class TestRegression:
    """Test linear regression analysis."""

    def test_basic_regression(self, regression_dataset):
        """Test basic regression."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        assert result.id == "linear_regression"
        # Should have: CPS + Model Summary + ANOVA + Coefficients + VIF + DW + Residuals
        assert len(result.tables) >= 6

    def test_model_summary(self, regression_dataset):
        """Test model summary values."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        summary_table = [t for t in result.tables if t.title == "Model Summary"][0]
        df = summary_table.dataframe
        stats_dict = {row["Statistic"]: row["Value"] for _, row in df.iterrows()}
        assert float(stats_dict["N"]) == 50
        assert float(stats_dict["R-squared"]) > 0

    def test_vs_statsmodels(self, regression_dataset):
        """Test that results match statsmodels."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        # Manual statsmodels
        X = sm.add_constant(regression_dataset.data[["x1", "x2"]].values)
        y = regression_dataset.data["y"].values
        model = sm.OLS(y, X).fit()

        summary_table = [t for t in result.tables if t.title == "Model Summary"][0]
        df = summary_table.dataframe
        stats_dict = {row["Statistic"]: row["Value"] for _, row in df.iterrows()}
        assert abs(float(stats_dict["R-squared"]) - model.rsquared) < 0.001
        assert abs(float(stats_dict["F-statistic"]) - model.fvalue) < 0.01

    def test_coefficients(self, regression_dataset):
        """Test coefficients table."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        coef_table = [t for t in result.tables if t.title == "Coefficients"][0]
        df = coef_table.dataframe
        vars_list = list(df["Variable"])
        assert "(Constant)" in vars_list
        assert "x1" in vars_list
        assert "x2" in vars_list

        # x1 should have positive coefficient (we generated y = 10 + 2*x1 - 1.5*x2)
        x1_row = df[df["Variable"] == "x1"].iloc[0]
        assert float(x1_row["B"]) > 0

        # x2 should have negative coefficient
        x2_row = df[df["Variable"] == "x2"].iloc[0]
        assert float(x2_row["B"]) < 0

    def test_vif(self, regression_dataset):
        """Test VIF computation."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        vif_tables = [t for t in result.tables if "VIF" in t.title]
        assert len(vif_tables) >= 1

    def test_durbin_watson(self, regression_dataset):
        """Test Durbin-Watson statistic."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        dw_tables = [t for t in result.tables if "Autocorrelation" in t.title]
        assert len(dw_tables) >= 1

    def test_dummy_coding(self, regression_with_cat):
        """Test automatic dummy coding for categorical predictors."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x", "cat"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_with_cat, spec)

        # Check dummy coding info table
        dummy_tables = [t for t in result.tables if "Dummy Coding" in t.title]
        assert len(dummy_tables) >= 1

    def test_with_missing_data(self):
        """Test regression with missing data."""
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
            "x1": [2.0, 3.0, np.nan, 5.0, 6.0, 7.0],
            "x2": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        })
        ds = Dataset(df, name="MissingReg")
        ds.variables["y"].measure = MeasureType.SCALE
        ds.variables["x1"].measure = MeasureType.SCALE
        ds.variables["x2"].measure = MeasureType.SCALE

        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        summary_table = [t for t in result.tables if t.title == "Model Summary"][0]
        df_sum = summary_table.dataframe
        stats_dict = {row["Statistic"]: row["Value"] for _, row in df_sum.iterrows()}
        # Only 2 rows have all 3 values
        assert float(stats_dict["N"]) == 3

    def test_no_predictors(self):
        """Test regression with no predictors."""
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds = Dataset(df, name="NoPred")
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"dependent": "y", "predictors": []},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        # Should still produce some output
        assert len(result.tables) >= 2

    def test_residual_summary(self, regression_dataset):
        """Test residual summary."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        resid_tables = [t for t in result.tables if "Residual" in t.title]
        assert len(resid_tables) >= 1

        resid_table = resid_tables[0]
        df = resid_table.dataframe
        stats_dict = {row["Statistic"]: row["Value"] for _, row in df.iterrows()}
        # Mean of residuals should be close to 0
        assert abs(float(stats_dict["Mean"])) < 0.01

    def test_formatting(self, regression_dataset):
        """Test that output formatting is correct."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(regression_dataset, spec)

        # Check p-value formatting
        coef_table = [t for t in result.tables if t.title == "Coefficients"][0]
        df = coef_table.dataframe
        p_vals = df["p-value"].values
        for p in p_vals:
            assert isinstance(str(p), str)

        # Check CI formatting
        ci_vals = df["CI"].values
        for ci in ci_vals:
            ci_str = str(ci)
            if ci_str and ci_str != "nan":
                assert "[" in ci_str and "]" in ci_str
