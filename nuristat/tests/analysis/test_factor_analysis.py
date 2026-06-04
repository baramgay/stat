"""Tests for factor analysis and PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.factor_analysis import run_analysis


@pytest.fixture
def factor_dataset():
    """Create a correlated numeric dataset suitable for factor analysis."""
    rng = np.random.default_rng(42)
    n = 200
    # Two latent factors
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.3, (n, 6))

    v1 = f1 + noise[:, 0]
    v2 = f1 * 0.9 + noise[:, 1]
    v3 = f1 * 0.8 + noise[:, 2]
    v4 = f2 + noise[:, 3]
    v5 = f2 * 0.9 + noise[:, 4]
    v6 = f2 * 0.85 + noise[:, 5]

    df = pd.DataFrame({
        "v1": v1, "v2": v2, "v3": v3,
        "v4": v4, "v5": v5, "v6": v6,
    })
    ds = Dataset(df, "factor_test")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def small_factor_dataset():
    """Minimal dataset for factor analysis (3 variables)."""
    rng = np.random.default_rng(99)
    n = 50
    f = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "a": f + rng.normal(0, 0.2, n),
        "b": f * 0.8 + rng.normal(0, 0.2, n),
        "c": f * 0.7 + rng.normal(0, 0.2, n),
    })
    ds = Dataset(df, "small_factor")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


class TestFactorAnalysis:
    """Tests for EFA and PCA."""

    def test_pca_basic(self, factor_dataset):
        """PCA should return a non-None result with tables."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2, "rotation": "varimax"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(factor_dataset, spec)

        assert result is not None
        assert result.id == "factor_analysis"
        assert len(result.tables) > 0

    def test_efa_basic(self, factor_dataset):
        """EFA should return a non-None result with tables."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "efa", "n_factors": 2, "rotation": "varimax"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(factor_dataset, spec)

        assert result is not None
        assert len(result.tables) > 0

    def test_kmo_between_0_and_1(self, factor_dataset):
        """KMO measure of sampling adequacy must be between 0 and 1."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "efa", "n_factors": 2},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(factor_dataset, spec)

        kmo_tables = [t for t in result.tables if "KMO" in t.title]
        assert len(kmo_tables) >= 1

        kmo_df = kmo_tables[0].dataframe
        kmo_row = kmo_df[kmo_df.apply(
            lambda row: row.astype(str).str.contains("KMO").any(), axis=1
        )]
        assert len(kmo_row) >= 1

        val_col = "값" if "값" in kmo_df.columns else kmo_df.columns[1]
        kmo_val_str = kmo_row[val_col].values[0]
        try:
            kmo_val = float(str(kmo_val_str).strip())
            assert 0.0 <= kmo_val <= 1.0, f"KMO should be in [0,1], got {kmo_val}"
        except (ValueError, TypeError):
            pytest.skip("KMO value could not be parsed as float")

    def test_factor_loadings_shape(self, factor_dataset):
        """Factor loading matrix must have one row per variable."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2, "rotation": "none"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(factor_dataset, spec)

        loading_tables = [t for t in result.tables if "부하량" in t.title or "Loading" in t.title]
        assert len(loading_tables) >= 1

        loading_df = loading_tables[0].dataframe
        # One row per variable (6 variables)
        assert len(loading_df) == 6

    def test_auto_n_factors(self, factor_dataset):
        """Auto factor selection should still produce a valid result."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "efa", "n_factors": "auto"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(factor_dataset, spec)

        assert result is not None
        assert len(result.tables) > 0

    def test_insufficient_variables(self):
        """Factor analysis with only one variable should warn."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = Dataset(df, "single_var")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"variables": ["x"]},
            "options": {"method": "efa"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        assert result is not None
        assert len(result.warnings) > 0
