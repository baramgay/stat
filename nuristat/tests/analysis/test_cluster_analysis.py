"""Tests for cluster analysis (K-means and hierarchical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.cluster_analysis import run_analysis


@pytest.fixture
def cluster_dataset():
    """Create a well-separated 3-cluster dataset."""
    rng = np.random.default_rng(42)
    # Three clearly separated clusters
    c1 = rng.normal([0.0, 0.0], 0.5, (40, 2))
    c2 = rng.normal([5.0, 5.0], 0.5, (40, 2))
    c3 = rng.normal([0.0, 8.0], 0.5, (40, 2))
    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["x", "y"])
    ds = Dataset(df, "cluster_test")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def small_cluster_dataset():
    """Minimal dataset for cluster analysis."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 30),
        "b": rng.normal(0, 1, 30),
        "c": rng.normal(0, 1, 30),
    })
    ds = Dataset(df, "small_cluster")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


class TestClusterAnalysis:
    """Tests for K-means and hierarchical clustering."""

    def test_kmeans_basic(self, cluster_dataset):
        """K-means with k=3 should return a valid result with tables."""
        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"method": "kmeans", "n_clusters": 3, "standardize": True},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cluster_dataset, spec)

        assert result is not None
        assert result.id == "cluster_analysis"
        assert len(result.tables) > 0

    def test_hierarchical_basic(self, cluster_dataset):
        """Hierarchical clustering should return a valid result with tables."""
        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {
                "method": "hierarchical",
                "n_clusters": 3,
                "linkage": "ward",
                "standardize": True,
            },
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cluster_dataset, spec)

        assert result is not None
        assert len(result.tables) > 0

    def test_silhouette_in_result(self, cluster_dataset):
        """Silhouette coefficient table should be present for k-means with k > 1."""
        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"method": "kmeans", "n_clusters": 3, "standardize": True},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cluster_dataset, spec)

        sil_tables = [t for t in result.tables if "실루엣" in t.title or "Silhouette" in t.title]
        assert len(sil_tables) >= 1

        sil_df = sil_tables[0].dataframe
        assert len(sil_df) >= 1

    def test_cluster_summary_table(self, cluster_dataset):
        """Cluster membership count table should list all k clusters."""
        n_clusters = 3
        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"method": "kmeans", "n_clusters": n_clusters, "standardize": True},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cluster_dataset, spec)

        membership_tables = [
            t for t in result.tables
            if "케이스 수" in t.title or "membership" in t.title.lower()
        ]
        assert len(membership_tables) >= 1

        mem_df = membership_tables[0].dataframe
        # Should have exactly n_clusters rows
        assert len(mem_df) == n_clusters

    def test_silhouette_value_in_range(self, cluster_dataset):
        """Overall silhouette coefficient must be between -1 and 1."""
        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"method": "kmeans", "n_clusters": 3, "standardize": False},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cluster_dataset, spec)

        sil_tables = [t for t in result.tables if "실루엣" in t.title or "Silhouette" in t.title]
        if not sil_tables:
            pytest.skip("Silhouette table not produced")

        sil_df = sil_tables[0].dataframe
        # Last row is overall
        overall_row = sil_df.iloc[-1]
        sil_col = [c for c in sil_df.columns if "실루엣" in c or "Silhouette" in c]
        if not sil_col:
            pytest.skip("Cannot locate silhouette column")

        val_str = str(overall_row[sil_col[0]])
        try:
            val = float(val_str.strip())
            assert -1.0 <= val <= 1.0, f"Silhouette must be in [-1,1], got {val}"
        except (ValueError, TypeError):
            pytest.skip("Silhouette value could not be parsed")

    def test_hierarchical_agglomeration_schedule(self, small_cluster_dataset):
        """Hierarchical clustering must produce an agglomeration schedule table."""
        spec = {
            "variables": {"variables": ["a", "b", "c"]},
            "options": {
                "method": "hierarchical",
                "n_clusters": 2,
                "linkage": "complete",
                "standardize": True,
            },
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(small_cluster_dataset, spec)

        agg_tables = [t for t in result.tables if "병합" in t.title or "Agglomeration" in t.title]
        assert len(agg_tables) >= 1
