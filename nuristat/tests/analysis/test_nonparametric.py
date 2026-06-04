"""Tests for nonparametric test analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.nonparametric import run_analysis


@pytest.fixture
def mann_whitney_dataset():
    """Create dataset for Mann-Whitney U test."""
    df = pd.DataFrame({
        "group": ["A"] * 8 + ["B"] * 8,
        "score": [12, 15, 18, 14, 16, 19, 13, 17, 25, 28, 22, 27, 24, 26, 23, 29],
    })
    ds = Dataset(df, name="MWTest")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.ORDINAL
    return ds


@pytest.fixture
def wilcoxon_dataset():
    """Create dataset for Wilcoxon test."""
    df = pd.DataFrame({
        "time1": [5, 7, 6, 8, 4, 6, 7, 5],
        "time2": [7, 9, 8, 10, 6, 8, 9, 7],
    })
    ds = Dataset(df, name="WilcoxonTest")
    ds.variables["time1"].measure = MeasureType.ORDINAL
    ds.variables["time2"].measure = MeasureType.ORDINAL
    return ds


@pytest.fixture
def kruskal_dataset():
    """Create dataset for Kruskal-Wallis test."""
    df = pd.DataFrame({
        "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "score": [10, 12, 11, 13, 9, 14, 20, 22, 21, 23, 19, 24, 30, 32, 31, 33, 29, 34],
    })
    ds = Dataset(df, name="KruskalTest")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.ORDINAL
    return ds


@pytest.fixture
def friedman_dataset():
    """Create dataset for Friedman test."""
    df = pd.DataFrame({
        "t1": [5, 7, 6, 8, 4, 6],
        "t2": [7, 9, 8, 10, 6, 8],
        "t3": [9, 11, 10, 12, 8, 10],
    })
    ds = Dataset(df, name="FriedmanTest")
    ds.variables["t1"].measure = MeasureType.ORDINAL
    ds.variables["t2"].measure = MeasureType.ORDINAL
    ds.variables["t3"].measure = MeasureType.ORDINAL
    return ds


class TestMannWhitney:
    """Test Mann-Whitney U test."""

    def test_basic(self, mann_whitney_dataset):
        """Test basic Mann-Whitney U test."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(mann_whitney_dataset, spec)

        assert result.id == "nonparametric_test"
        assert len(result.tables) >= 2

    def test_vs_scipy(self, mann_whitney_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(mann_whitney_dataset, spec)

        g1 = mann_whitney_dataset.data[mann_whitney_dataset.data["group"] == "A"]["score"].values
        g2 = mann_whitney_dataset.data[mann_whitney_dataset.data["group"] == "B"]["score"].values
        u_scipy, p_scipy = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        u_row = df[df["Statistic"] == "Mann-Whitney U"].iloc[0]
        assert abs(float(u_row["Value"]) - u_scipy) < 0.1

    def test_effect_size(self, mann_whitney_dataset):
        """Test that rank-biserial correlation is computed."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(mann_whitney_dataset, spec)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        rbc_row = df[df["Statistic"] == "Rank-Biserial r"]
        assert len(rbc_row) == 1
        assert abs(float(rbc_row.iloc[0]["Value"])) > 0


class TestWilcoxon:
    """Test Wilcoxon signed-rank test."""

    def test_basic(self, wilcoxon_dataset):
        """Test basic Wilcoxon test."""
        spec = {
            "variables": {"paired": ["time1", "time2"]},
            "options": {"test": "wilcoxon"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(wilcoxon_dataset, spec)

        assert result.id == "nonparametric_test"
        assert len(result.tables) >= 2

    def test_vs_scipy(self, wilcoxon_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"paired": ["time1", "time2"]},
            "options": {"test": "wilcoxon"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(wilcoxon_dataset, spec)

        x1 = wilcoxon_dataset.data["time1"].values
        x2 = wilcoxon_dataset.data["time2"].values
        w_scipy, p_scipy = stats.wilcoxon(x1, x2)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        w_row = df[df["Statistic"] == "Wilcoxon W"].iloc[0]
        assert abs(float(w_row["Value"]) - w_scipy) < 0.1

    def test_effect_size(self, wilcoxon_dataset):
        """Test that effect size r is computed."""
        spec = {
            "variables": {"paired": ["time1", "time2"]},
            "options": {"test": "wilcoxon"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(wilcoxon_dataset, spec)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        r_row = df[df["Statistic"] == "Effect Size r"]
        assert len(r_row) == 1


class TestKruskalWallis:
    """Test Kruskal-Wallis H test."""

    def test_basic(self, kruskal_dataset):
        """Test basic Kruskal-Wallis test."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(kruskal_dataset, spec)

        assert result.id == "nonparametric_test"
        assert len(result.tables) >= 2

    def test_vs_scipy(self, kruskal_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(kruskal_dataset, spec)

        groups = [
            kruskal_dataset.data[kruskal_dataset.data["group"] == g]["score"].values
            for g in ["A", "B", "C"]
        ]
        h_scipy, p_scipy = stats.kruskal(*groups)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        h_row = df[df["Statistic"] == "Kruskal-Wallis H"].iloc[0]
        assert abs(float(h_row["Value"]) - h_scipy) < 0.01

    def test_epsilon_squared(self, kruskal_dataset):
        """Test that epsilon-squared is computed."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(kruskal_dataset, spec)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        eps_row = df[df["Statistic"] == "Epsilon-squared"]
        assert len(eps_row) == 1

    def test_less_than_three_groups(self, mann_whitney_dataset):
        """Test that warning is shown for < 3 groups."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(mann_whitney_dataset, spec)

        # The Kruskal-Wallis should still work with 2 groups (it's valid)
        # but should give a warning
        assert len(result.warnings) >= 1


class TestFriedman:
    """Test Friedman test."""

    def test_basic(self, friedman_dataset):
        """Test basic Friedman test."""
        spec = {
            "variables": {"repeated": ["t1", "t2", "t3"]},
            "options": {"test": "friedman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(friedman_dataset, spec)

        assert result.id == "nonparametric_test"
        assert len(result.tables) >= 2

    def test_vs_scipy(self, friedman_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"repeated": ["t1", "t2", "t3"]},
            "options": {"test": "friedman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(friedman_dataset, spec)

        t1 = friedman_dataset.data["t1"].values
        t2 = friedman_dataset.data["t2"].values
        t3 = friedman_dataset.data["t3"].values
        chi2_scipy, p_scipy = stats.friedmanchisquare(t1, t2, t3)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        chi_row = df[df["Statistic"] == "Chi-Square"].iloc[0]
        assert abs(float(chi_row["Value"]) - chi2_scipy) < 0.01

    def test_kendalls_w(self, friedman_dataset):
        """Test that Kendall's W is computed."""
        spec = {
            "variables": {"repeated": ["t1", "t2", "t3"]},
            "options": {"test": "friedman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(friedman_dataset, spec)

        test_table = [t for t in result.tables if "Test Statistics" in t.title][0]
        df = test_table.dataframe
        w_row = df[df["Statistic"] == "Kendall's W"]
        assert len(w_row) == 1
        assert 0 <= float(w_row.iloc[0]["Value"]) <= 1

    def test_less_than_three_conditions(self):
        """Test warning for < 3 conditions."""
        df = pd.DataFrame({
            "t1": [1, 2, 3],
            "t2": [2, 3, 4],
        })
        ds = Dataset(df, name="SmallFriedman")
        ds.variables["t1"].measure = MeasureType.ORDINAL
        ds.variables["t2"].measure = MeasureType.ORDINAL

        spec = {
            "variables": {"repeated": ["t1", "t2"]},
            "options": {"test": "friedman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        assert len(result.warnings) >= 1
        assert "at least 3" in str(result.warnings[0]).lower()
