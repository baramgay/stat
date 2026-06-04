"""Tests for survival analysis (Kaplan-Meier and log-rank)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.survival_analysis import run_analysis


@pytest.fixture
def survival_dataset():
    """Create a basic survival dataset without groups."""
    rng = np.random.default_rng(42)
    n = 80
    # Exponential survival times
    duration = rng.exponential(scale=10.0, size=n)
    # ~30% censored
    event = (rng.random(n) > 0.3).astype(int)
    df = pd.DataFrame({"time": duration, "event": event})
    ds = Dataset(df, "survival_test")
    ds.variables["time"].measure = MeasureType.SCALE
    ds.variables["event"].measure = MeasureType.BINARY
    return ds


@pytest.fixture
def survival_group_dataset():
    """Create a survival dataset with two groups having different hazard rates."""
    rng = np.random.default_rng(99)
    n_per_group = 50
    # Group A: scale=5 (worse prognosis)
    t_a = rng.exponential(scale=5.0, size=n_per_group)
    e_a = (rng.random(n_per_group) > 0.2).astype(int)
    # Group B: scale=15 (better prognosis)
    t_b = rng.exponential(scale=15.0, size=n_per_group)
    e_b = (rng.random(n_per_group) > 0.2).astype(int)

    df = pd.DataFrame({
        "time": np.concatenate([t_a, t_b]),
        "event": np.concatenate([e_a, e_b]),
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
    })
    ds = Dataset(df, "survival_group_test")
    ds.variables["time"].measure = MeasureType.SCALE
    ds.variables["event"].measure = MeasureType.BINARY
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


class TestSurvivalAnalysis:
    """Tests for Kaplan-Meier and log-rank survival analysis."""

    def test_km_basic(self, survival_dataset):
        """KM analysis without groups should return a non-None result with tables."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
            },
            "options": {"method": "km"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(survival_dataset, spec)

        assert result is not None
        assert result.id == "survival_analysis"
        assert len(result.tables) > 0

    def test_km_result_has_survival_table(self, survival_dataset):
        """KM result must contain a survival function table."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
            },
            "options": {"method": "km"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(survival_dataset, spec)

        survival_tables = [
            t for t in result.tables
            if "생존 함수" in t.title or "KM" in t.title or "Survival" in t.title
        ]
        assert len(survival_tables) >= 1

        sf_df = survival_tables[0].dataframe
        assert len(sf_df) >= 1

    def test_logrank_with_group(self, survival_group_dataset):
        """Log-rank test should be performed when group variable is provided."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "group": "group",
            },
            "options": {"method": "km"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(survival_group_dataset, spec)

        assert result is not None
        logrank_tables = [
            t for t in result.tables
            if "log-rank" in t.title.lower() or "Log-rank" in t.title
        ]
        assert len(logrank_tables) >= 1

    def test_median_survival_time(self, survival_dataset):
        """Median survival time should be reported in the KM summary table."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
            },
            "options": {"method": "km"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(survival_dataset, spec)

        # Find KM summary tables (exclude the case processing summary which has
        # different columns: Total Cases, Valid Cases, etc.)
        km_summary_tables = [
            t for t in result.tables
            if ("요약" in t.title or "Kaplan" in t.title)
            and "통계량" in t.dataframe.columns
        ]
        assert len(km_summary_tables) >= 1, (
            f"No KM summary table found. Tables: {[t.title for t in result.tables]}"
        )

        summary_df = km_summary_tables[0].dataframe
        # Look for the median survival row
        median_rows = summary_df[summary_df["통계량"].str.contains("중앙", na=False)]
        assert len(median_rows) >= 1, (
            f"Median row not found in KM summary. Rows: {summary_df['통계량'].tolist()}"
        )

        # Extract value and check it is non-negative
        val_str = str(median_rows["값"].values[0])
        try:
            val = float(val_str.strip())
            assert val >= 0 or np.isnan(val), f"Median survival must be >= 0, got {val}"
        except (ValueError, TypeError):
            pass  # nan or inf formatted strings are acceptable

    def test_missing_variables_warning(self):
        """Missing duration or event variable should produce a warning."""
        df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1, 0, 1]})
        ds = Dataset(df, "warn_test")
        ds.variables["time"].measure = MeasureType.SCALE
        ds.variables["event"].measure = MeasureType.BINARY

        spec = {
            "variables": {},  # No duration/event specified
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        assert result is not None
        assert len(result.warnings) > 0
