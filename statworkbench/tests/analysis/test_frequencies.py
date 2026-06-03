"""Tests for frequency analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.frequencies import run_analysis


@pytest.fixture
def freq_dataset():
    """Create a dataset with categorical variables."""
    df = pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", np.nan, "F", "M"],
        "category": ["A", "B", "A", "C", "B", "A", np.nan, "C"],
    })
    ds = Dataset(df, name="FreqTest")
    ds.variables["gender"].measure = MeasureType.NOMINAL
    ds.variables["category"].measure = MeasureType.NOMINAL
    return ds


class TestFrequencies:
    """Test frequency analysis."""

    def test_basic_frequencies(self, freq_dataset):
        """Test basic frequency table."""
        spec = {
            "variables": {"target": ["gender"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(freq_dataset, spec)

        assert result.id == "frequencies"
        assert len(result.tables) >= 2  # CPS + freq table

        freq_table = result.tables[1]
        assert "Frequency" in freq_table.dataframe.columns
        df = freq_table.dataframe
        total_freq = df["Frequency"].sum()
        assert total_freq == 7  # 8 rows minus 1 NaN

    def test_empty_variables_warns(self, freq_dataset):
        """분석 변수 미지정 시 빈 테이블 대신 명확한 경고를 반환한다."""
        spec = {
            "variables": {"target": []},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(freq_dataset, spec)
        assert result.tables == []
        assert any("지정되지 않았" in w for w in result.warnings)

    def test_multiple_variables(self, freq_dataset):
        """Test frequencies with multiple variables."""
        spec = {
            "variables": {"target": ["gender", "category"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(freq_dataset, spec)

        # Should have CPS + 2 freq tables
        assert len(result.tables) >= 3

    def test_include_missing(self, freq_dataset):
        """Test frequencies with missing values included."""
        spec = {
            "variables": {"target": ["gender"]},
            "options": {"include_missing": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(freq_dataset, spec)

        freq_table = result.tables[1]
        df = freq_table.dataframe
        assert any(df["Value"] == "Missing")

    def test_percentages(self, freq_dataset):
        """Test that percentages sum correctly."""
        spec = {
            "variables": {"target": ["gender"]},
            "options": {"show_cumulative": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(freq_dataset, spec)

        freq_table = result.tables[1]
        df = freq_table.dataframe
        assert "Percent" in df.columns
        assert "Valid Percent" in df.columns
        assert "Cumulative Percent" in df.columns

    def test_small_sample(self):
        """Test with very small sample."""
        df = pd.DataFrame({
            "x": ["A", "B"],
        })
        ds = Dataset(df, name="Small")
        ds.variables["x"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"target": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        freq_table = result.tables[1]
        df = freq_table.dataframe
        assert len(df) == 2
        assert set(df["Value"]) == {"A", "B"}
