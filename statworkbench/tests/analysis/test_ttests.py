"""Tests for t-test analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.ttests import run_analysis


@pytest.fixture
def independent_dataset():
    """Create dataset for independent t-test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "score": np.concatenate([
            np.random.normal(75, 10, 20),
            np.random.normal(85, 10, 20),
        ]),
        "group": ["A"] * 20 + ["B"] * 20,
    })
    ds = Dataset(df, name="Independent")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def paired_dataset_fixture():
    """Create dataset for paired t-test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "pre": np.random.normal(70, 8, 15),
        "post": np.random.normal(78, 8, 15),
    })
    ds = Dataset(df, name="Paired")
    ds.variables["pre"].measure = MeasureType.SCALE
    ds.variables["post"].measure = MeasureType.SCALE
    return ds


class TestIndependentTTest:
    """Test independent samples t-test."""

    def test_basic_independent_ttest(self, independent_dataset):
        """Test basic independent t-test."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        assert result.id == "t_test"
        assert len(result.tables) >= 3  # CPS + group stats + Levene + t-test

    def test_levene_test(self, independent_dataset):
        """Test that Levene's test is included."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        levene_table = [t for t in result.tables if "Equality of Variances" in t.title][0]
        assert len(levene_table.dataframe) == 1

    def test_both_variants(self, independent_dataset):
        """Test that both equal and unequal variance results are shown."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        ttest_table = [t for t in result.tables if "Independent" in t.title][0]
        df = ttest_table.dataframe
        assert len(df) == 2  # equal var + unequal var

    def test_group_statistics(self, independent_dataset):
        """Test that group statistics are correct."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        group_table = [t for t in result.tables if "Group Statistics" in t.title][0]
        df = group_table.dataframe
        assert len(df) == 2
        assert df.iloc[0]["N"] == 20
        assert df.iloc[1]["N"] == 20

    def test_vs_scipy(self, independent_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        # Manual scipy calculation
        g1 = independent_dataset.data[independent_dataset.data["group"] == "A"]["score"].values
        g2 = independent_dataset.data[independent_dataset.data["group"] == "B"]["score"].values
        t_scipy, p_scipy = stats.ttest_ind(g1, g2)

        ttest_table = [t for t in result.tables if "Independent" in t.title][0]
        df = ttest_table.dataframe
        eq_row = df[df["Variant"] == "Equal variances assumed"].iloc[0]
        assert abs(float(eq_row["t"]) - t_scipy) < 0.01


class TestPairedTTest:
    """Test paired samples t-test."""

    def test_basic_paired_ttest(self, paired_dataset_fixture):
        """Test basic paired t-test."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        assert result.id == "t_test"
        assert len(result.tables) >= 3  # CPS + paired stats + test

    def test_paired_statistics(self, paired_dataset_fixture):
        """Test paired statistics table."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        paired_table = [t for t in result.tables if "Paired Samples Statistics" in t.title][0]
        df = paired_table.dataframe
        assert len(df) == 2

    def test_vs_scipy(self, paired_dataset_fixture):
        """Test that results match scipy."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        pre = paired_dataset_fixture.data["pre"].values
        post = paired_dataset_fixture.data["post"].values
        t_scipy, p_scipy = stats.ttest_rel(pre, post)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        t_row = df[df["Statistic"] == "t"].iloc[0]
        assert abs(float(t_row["Value"]) - t_scipy) < 0.01

    def test_cohens_dz(self, paired_dataset_fixture):
        """Test that Cohen's dz is computed."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        dz_row = df[df["Statistic"] == "Cohen's dz"]
        assert len(dz_row) == 1
        assert abs(float(dz_row.iloc[0]["Value"])) > 0

    def test_with_missing(self):
        """Test paired t-test with missing data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, np.nan, 5.0],
            "y": [2.0, 3.0, np.nan, 5.0, 6.0],
        })
        ds = Dataset(df, name="MissingPaired")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"paired": ["x", "y"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        # N is in Paired Samples Statistics table, not t-Test table
        stats_table = [t for t in result.tables if "Paired Samples Statistics" in t.title][0]
        stats_df = stats_table.dataframe
        n_row = stats_df[stats_df["Variable"] == "x"].iloc[0]
        assert int(n_row["N"]) == 3  # rows 0, 1, 4
