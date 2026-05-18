"""Tests for ANOVA analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.anova import run_analysis


@pytest.fixture
def anova_test_dataset():
    """Create dataset for ANOVA testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "group": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "score": (
            np.random.normal(75, 8, 12).tolist() +
            np.random.normal(82, 10, 12).tolist() +
            np.random.normal(70, 7, 12).tolist()
        ),
    })
    ds = Dataset(df, name="ANOVATest")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.SCALE
    return ds


class TestANOVA:
    """Test one-way ANOVA analysis."""

    def test_basic_anova(self, anova_test_dataset):
        """Test basic ANOVA."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"], "effect_size": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        assert result.id == "one_way_anova"
        assert len(result.tables) >= 4  # CPS + descriptives + Levene + ANOVA

    def test_descriptives_table(self, anova_test_dataset):
        """Test that descriptives table has correct groups."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        desc_table = [t for t in result.tables if t.title == "Descriptives"][0]
        df = desc_table.dataframe
        assert len(df) == 3
        groups = set(df["Group"])
        assert groups == {"A", "B", "C"}

    def test_anova_table(self, anova_test_dataset):
        """Test ANOVA table structure."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        anova_table = [t for t in result.tables if t.title == "ANOVA"][0]
        df = anova_table.dataframe
        sources = set(df["Source"])
        assert any("C(" in s for s in sources)

    def test_effect_sizes(self, anova_test_dataset):
        """Test that eta-squared and omega-squared are computed."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        anova_table = [t for t in result.tables if t.title == "ANOVA"][0]
        df = anova_table.dataframe
        assert any(df["Source"] == "Eta-squared")
        assert any(df["Source"] == "Omega-squared")

    def test_levene_test(self, anova_test_dataset):
        """Test that Levene's test is included."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        levene_table = [t for t in result.tables if "Homogeneity" in t.title][0]
        assert len(levene_table.dataframe) == 1

    def test_vs_scipy(self, anova_test_dataset):
        """Test that ANOVA results match scipy."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        # Manual scipy
        groups = [
            anova_test_dataset.data[anova_test_dataset.data["group"] == g]["score"].values
            for g in ["A", "B", "C"]
        ]
        f_scipy, p_scipy = stats.f_oneway(*groups)

        anova_table = [t for t in result.tables if t.title == "ANOVA"][0]
        df = anova_table.dataframe
        factor_row = df[df["Source"].str.contains("C(", regex=False)].iloc[0]
        f_val = float(factor_row["F"])
        assert abs(f_val - f_scipy) < 0.01

    def test_tukey_posthoc(self, anova_test_dataset):
        """Test that Tukey HSD is computed."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        tukey_tables = [t for t in result.tables if "Tukey" in t.title]
        assert len(tukey_tables) >= 1

    def test_bonferroni_posthoc(self, anova_test_dataset):
        """Test that Bonferroni post-hoc is computed."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["bonferroni"], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        bonf_tables = [t for t in result.tables if "Bonferroni" in t.title]
        assert len(bonf_tables) >= 1

    def test_welch_anova(self, anova_test_dataset):
        """Test Welch's ANOVA option."""
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": False, "welch": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(anova_test_dataset, spec)

        welch_tables = [t for t in result.tables if "Welch" in t.title]
        assert len(welch_tables) >= 1

    def test_with_missing_data(self):
        """Test ANOVA with missing data."""
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", np.nan],
            "score": [10.0, 20.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        ds = Dataset(df, name="MissingANOVA")
        ds.variables["group"].measure = MeasureType.NOMINAL
        ds.variables["score"].measure = MeasureType.SCALE

        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_table = [t for t in result.tables if t.title == "Descriptives"][0]
        df_desc = desc_table.dataframe
        # Should have 3 groups
        assert len(df_desc) == 3
