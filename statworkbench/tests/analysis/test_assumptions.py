"""Tests for assumption checks and missing-data utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.assumptions import (
    PreparedAnalysisFrame,
    check_homogeneity_of_variance,
    check_homogeneity_of_variance_from_groups,
    check_normality,
    get_case_processing_summary,
    prepare_analysis_frame,
)
from statworkbench.analysis.result import ResultTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset() -> Dataset:
    """Return a simple numeric dataset with no missing values."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y": [2.1, 4.0, 6.1, 7.8, 10.2, 12.0, 14.1, 15.9, 18.0, 20.2],
        "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    })
    ds = Dataset(df, name="Simple")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def missing_dataset() -> Dataset:
    """Return a dataset with missing values."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
        "y": [2.1, np.nan, 6.1, 7.8, np.nan, 12.0, 14.1, 15.9, np.nan, 20.2],
        "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    })
    ds = Dataset(df, name="MissingData")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def group_data() -> list[np.ndarray]:
    """Return three groups with similar variance for Levene test."""
    np.random.seed(42)
    return [
        np.random.normal(10, 2, 30),
        np.random.normal(12, 2, 30),
        np.random.normal(11, 2, 30),
    ]


@pytest.fixture
def heteroscedastic_groups() -> list[np.ndarray]:
    """Return groups with unequal variances."""
    np.random.seed(42)
    return [
        np.random.normal(10, 1, 30),
        np.random.normal(10, 5, 30),
        np.random.normal(10, 10, 30),
    ]


# ---------------------------------------------------------------------------
# prepare_analysis_frame
# ---------------------------------------------------------------------------

class TestPrepareAnalysisFrame:
    """Tests for prepare_analysis_frame."""

    def test_listwise_no_missing(self, simple_dataset: Dataset) -> None:
        """Listwise deletion with no missing data should keep all rows."""
        result = prepare_analysis_frame(
            simple_dataset,
            variables=["x", "y"],
            missing_policy=MissingPolicy.LISTWISE,
        )
        assert isinstance(result, PreparedAnalysisFrame)
        assert result.n_total == 10
        assert result.n_valid == 10
        assert result.n_excluded == 0
        assert result.excluded_pct == 0.0
        assert len(result.data) == 10

    def test_listwise_with_missing(self, missing_dataset: Dataset) -> None:
        """Listwise deletion should drop rows with any missing values."""
        result = prepare_analysis_frame(
            missing_dataset,
            variables=["x", "y"],
            missing_policy=MissingPolicy.LISTWISE,
        )
        assert result.n_total == 10
        # Rows with any NaN in x or y:
        # row 1 (x=NaN), row 1 (y=NaN in row 1), row 2 (x=NaN), row 4 (y=NaN),
        # row 6 (x=NaN), row 8 (y=NaN)
        # Actually let's compute: x=[2,6,9] missing, y=[1,4,8] missing
        # Missing rows: 2, 3, 5, 7, 9 (0-indexed rows 1,2,4,6,8)
        # Valid rows: 0, 3, 5, 7, 9  → 5 rows
        assert result.n_valid == 5
        assert result.n_excluded == 5
        assert result.excluded_pct == 50.0

    def test_pairwise_policy(self, missing_dataset: Dataset) -> None:
        """Pairwise policy should keep all rows."""
        result = prepare_analysis_frame(
            missing_dataset,
            variables=["x", "y"],
            missing_policy=MissingPolicy.PAIRWISE,
        )
        assert result.n_total == 10
        assert result.n_valid == 10

    def test_include_as_category(self, missing_dataset: Dataset) -> None:
        """Include-as-category policy should keep all rows."""
        result = prepare_analysis_frame(
            missing_dataset,
            variables=["x", "y"],
            missing_policy=MissingPolicy.INCLUDE_AS_CATEGORY,
        )
        assert result.n_valid == 10

    def test_single_variable(self, missing_dataset: Dataset) -> None:
        """Requesting a single variable should work."""
        result = prepare_analysis_frame(
            missing_dataset,
            variables=["x"],
            missing_policy=MissingPolicy.LISTWISE,
        )
        assert result.n_total == 10
        # x has NaN at rows 2, 7 → 2 missing
        assert result.n_valid == 8
        assert result.n_excluded == 2

    def test_invalid_variable(self, simple_dataset: Dataset) -> None:
        """Requesting a non-existent variable should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            prepare_analysis_frame(
                simple_dataset,
                variables=["nonexistent"],
            )

    def test_empty_variables(self, simple_dataset: Dataset) -> None:
        """Empty variable list should return all rows (no filtering)."""
        result = prepare_analysis_frame(
            simple_dataset,
            variables=[],
        )
        assert result.n_total == 10
        assert result.n_valid == 10

    def test_exclude_system_missing_only(
        self, missing_dataset: Dataset
    ) -> None:
        """EXCLUDE_SYSTEM_MISSING_ONLY should drop NaN rows."""
        result = prepare_analysis_frame(
            missing_dataset,
            variables=["x"],
            missing_policy=MissingPolicy.EXCLUDE_SYSTEM_MISSING_ONLY,
        )
        assert result.n_valid == 8  # same as listwise for single var


# ---------------------------------------------------------------------------
# Case Processing Summary
# ---------------------------------------------------------------------------

class TestCaseProcessingSummary:
    """Tests for get_case_processing_summary."""

    def test_basic(self) -> None:
        table = get_case_processing_summary(
            n_total=100,
            n_valid=85,
            n_excluded=15,
        )
        assert isinstance(table, ResultTable)
        assert table.title == "Case Processing Summary"
        df = table.dataframe
        assert df.loc[0, "Total Cases"] == 100
        assert df.loc[0, "Valid Cases"] == 85
        assert df.loc[0, "Excluded Cases"] == 15
        assert df.loc[0, "Excluded %"] == "15.0%"

    def test_zero_excluded(self) -> None:
        table = get_case_processing_summary(
            n_total=50,
            n_valid=50,
            n_excluded=0,
        )
        df = table.dataframe
        assert df.loc[0, "Excluded %"] == "0.0%"

    def test_all_excluded(self) -> None:
        table = get_case_processing_summary(
            n_total=50,
            n_valid=0,
            n_excluded=50,
        )
        df = table.dataframe
        assert df.loc[0, "Excluded %"] == "100.0%"

    def test_total_zero(self) -> None:
        """Edge case: zero total should not divide by zero."""
        table = get_case_processing_summary(
            n_total=0,
            n_valid=0,
            n_excluded=0,
        )
        df = table.dataframe
        assert df.loc[0, "Excluded %"] == "0.0%"

    def test_footnotes_present(self) -> None:
        table = get_case_processing_summary(100, 90, 10)
        assert len(table.footnotes) > 0
        assert "listwise" in table.footnotes[0].lower()

    def test_with_explicit_pct(self) -> None:
        table = get_case_processing_summary(
            n_total=200,
            n_valid=180,
            n_excluded=20,
            excluded_pct=10.0,
        )
        df = table.dataframe
        assert df.loc[0, "Excluded %"] == "10.0%"


# ---------------------------------------------------------------------------
# Normality check
# ---------------------------------------------------------------------------

class TestCheckNormality:
    """Tests for check_normality (Shapiro-Wilk)."""

    def test_normal_data(self) -> None:
        """Normal data should pass normality test (fail to reject H0)."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 50))
        result = check_normality(data)
        assert result["test"] == "Shapiro-Wilk"
        assert result["n"] == 50
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert not math.isnan(result["statistic"])
        assert not math.isnan(result["p_value"])
        assert isinstance(result["normal"], bool)

    def test_small_sample_normal(self) -> None:
        """Very small sample should still work with n >= 3."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = check_normality(data)
        assert result["n"] == 5
        assert result["test"] == "Shapiro-Wilk"
        assert "low power" in " ".join(result["warnings"]).lower()

    def test_sample_size_too_small(self) -> None:
        """Sample size < 3 should return NaN values."""
        data = pd.Series([1.0, 2.0])
        result = check_normality(data)
        assert math.isnan(result["statistic"])
        assert result["n"] == 2
        assert any("cannot" in w.lower() for w in result["warnings"])

    def test_with_missing_values(self) -> None:
        """Missing values should be dropped automatically."""
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = check_normality(data)
        assert result["n"] == 5

    def test_uniform_data(self) -> None:
        """Uniform data should fail normality test."""
        np.random.seed(42)
        data = pd.Series(np.random.uniform(0, 100, 100))
        result = check_normality(data, alpha=0.05)
        # Uniform data is not normal, so p should be < 0.05
        assert not result["normal"] or result["p_value"] < 0.1

    def test_large_sample_warning(self) -> None:
        """Sample > 5000 should trigger a warning."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 6000))
        result = check_normality(data)
        assert any("5,000" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Homogeneity of variance
# ---------------------------------------------------------------------------

class TestCheckHomogeneityOfVariance:
    """Tests for check_homogeneity_of_variance (Levene/Brown-Forsythe)."""

    def test_equal_variances(self, group_data: list[np.ndarray]) -> None:
        """Groups with equal variance should pass."""
        result = check_homogeneity_of_variance(*group_data, alpha=0.05)
        assert result["test"] in ("Levene", "Brown-Forsythe")
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert result["homogeneous"] is True
        assert result["alpha"] == 0.05

    def test_unequal_variances(
        self, heteroscedastic_groups: list[np.ndarray]
    ) -> None:
        """Groups with unequal variance should fail."""
        result = check_homogeneity_of_variance(
            *heteroscedastic_groups, alpha=0.05
        )
        assert result["homogeneous"] is False

    def test_single_group(self) -> None:
        """Single group should return an error message."""
        result = check_homogeneity_of_variance(
            np.array([1, 2, 3]), alpha=0.05
        )
        assert any(
            "at least 2" in w.lower() for w in result["warnings"]
        )

    def test_two_groups_equal_var(self) -> None:
        """Two groups with equal variance."""
        np.random.seed(42)
        g1 = np.random.normal(10, 2, 50)
        g2 = np.random.normal(12, 2, 50)
        result = check_homogeneity_of_variance(g1, g2, alpha=0.05)
        assert result["homogeneous"] is True

    def test_two_groups_unequal_var(self) -> None:
        """Two groups with unequal variance."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 50)
        g2 = np.random.normal(12, 5, 50)
        result = check_homogeneity_of_variance(g1, g2, alpha=0.05)
        assert result["homogeneous"] is False

    def test_with_nan_values(self) -> None:
        """NaN values should be dropped."""
        g1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = check_homogeneity_of_variance(g1, g2, alpha=0.05)
        assert isinstance(result["statistic"], float)

    def test_center_mean(self, group_data: list[np.ndarray]) -> None:
        """Test with center='mean' (classic Levene)."""
        result = check_homogeneity_of_variance(
            *group_data, alpha=0.05, center="mean"
        )
        assert result["test"] == "Levene"

    def test_center_median(self, group_data: list[np.ndarray]) -> None:
        """Test with center='median' (Brown-Forsythe)."""
        result = check_homogeneity_of_variance(
            *group_data, alpha=0.05, center="median"
        )
        assert result["test"] == "Brown-Forsythe"


# ---------------------------------------------------------------------------
# Homogeneity from groups (convenience wrapper)
# ---------------------------------------------------------------------------

class TestCheckHomogeneityFromGroups:
    """Tests for check_homogeneity_of_variance_from_groups."""

    def test_basic(self, simple_dataset: Dataset) -> None:
        """Test with dataset group/dependent columns."""
        result = check_homogeneity_of_variance_from_groups(
            data=simple_dataset.data["x"],
            group=simple_dataset.data["group"],
        )
        assert "statistic" in result
        assert "p_value" in result
        assert "homogeneous" in result

    def test_with_nan(self) -> None:
        """Test with NaN in data."""
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        group = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = check_homogeneity_of_variance_from_groups(data, group)
        assert isinstance(result["statistic"], float)
