"""Tests for descriptive statistics analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.descriptive import run_analysis


@pytest.fixture
def simple_dataset():
    """Create a simple numeric dataset."""
    df = pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 70.0],
        "age": [20, 25, 30, 35, 40, 45, np.nan],
        "group": ["A", "A", "B", "B", "A", "B", "A"],
    })
    ds = Dataset(df, name="Test")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["age"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


class TestDescriptive:
    """Test descriptive statistics analysis."""

    def test_basic_descriptives(self, simple_dataset):
        """Test basic descriptives without grouping."""
        spec = {
            "variables": {"scale": ["score"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(simple_dataset, spec)

        assert result.id == "descriptive_statistics"
        assert len(result.tables) >= 2  # CPS + descriptives

        desc_table = result.tables[1]
        assert desc_table.title == "Descriptive Statistics"
        df = desc_table.dataframe
        assert len(df) == 1
        assert df.iloc[0]["Variable"] == "score"
        assert df.iloc[0]["N"] == 6  # 7 rows minus 1 NaN

    def test_grouped_descriptives(self, simple_dataset):
        """Test descriptives with grouping variable."""
        spec = {
            "variables": {"scale": ["score"], "group": "group"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(simple_dataset, spec)

        desc_table = result.tables[1]
        df = desc_table.dataframe
        assert "Group" in df.columns
        groups = df["Group"].unique()
        assert len(groups) >= 2

    def test_multiple_variables(self, simple_dataset):
        """Test descriptives with multiple variables."""
        spec = {
            "variables": {"scale": ["score", "age"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(simple_dataset, spec)

        desc_table = result.tables[1]
        df = desc_table.dataframe
        assert len(df) == 2
        assert set(df["Variable"]) == {"score", "age"}

    def test_statistical_values(self):
        """Test that computed statistics are correct."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds = Dataset(df, name="Exact")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_table = result.tables[1]
        row = desc_table.dataframe.iloc[0]
        assert row["N"] == 5
        assert float(row["Mean"]) == pytest.approx(3.0, abs=0.01)
        assert float(row["Median"]) == pytest.approx(3.0, abs=0.01)
        assert float(row["Min"]) == pytest.approx(1.0, abs=0.01)
        assert float(row["Max"]) == pytest.approx(5.0, abs=0.01)

    def test_with_missing_data(self):
        """Test handling of missing data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        ds = Dataset(df, name="Missing")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_table = result.tables[1]
        row = desc_table.dataframe.iloc[0]
        assert row["N"] == 4

    def test_formatting(self):
        """Test that output formatting is correct."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds = Dataset(df, name="Format")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_table = result.tables[1]
        row = desc_table.dataframe.iloc[0]
        # CI should be formatted as a string with brackets
        assert "CI" in desc_table.dataframe.columns
        ci_val = row["CI"]
        assert isinstance(ci_val, str)
        assert "[" in ci_val and "]" in ci_val
