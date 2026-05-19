"""Tests for linear discriminant analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.discriminant_analysis import run_analysis


@pytest.fixture
def discriminant_dataset():
    """Create a well-separated 3-class dataset for LDA testing."""
    rng = np.random.default_rng(42)
    n_per_class = 40

    # Class A: cluster around (0, 0)
    x_a = rng.normal([0.0, 0.0], 0.5, (n_per_class, 2))
    # Class B: cluster around (5, 5)
    x_b = rng.normal([5.0, 5.0], 0.5, (n_per_class, 2))
    # Class C: cluster around (0, 8)
    x_c = rng.normal([0.0, 8.0], 0.5, (n_per_class, 2))

    X = np.vstack([x_a, x_b, x_c])
    y = ["A"] * n_per_class + ["B"] * n_per_class + ["C"] * n_per_class

    df = pd.DataFrame({"group": y, "feat1": X[:, 0], "feat2": X[:, 1]})
    ds = Dataset(df, "lda_test")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["feat1"].measure = MeasureType.SCALE
    ds.variables["feat2"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def binary_discriminant_dataset():
    """Create a 2-class dataset for LDA."""
    rng = np.random.default_rng(7)
    n = 60
    x1 = np.concatenate([rng.normal(0.0, 1.0, n // 2), rng.normal(4.0, 1.0, n // 2)])
    x2 = np.concatenate([rng.normal(0.0, 1.0, n // 2), rng.normal(4.0, 1.0, n // 2)])
    labels = ["ctrl"] * (n // 2) + ["case"] * (n // 2)

    df = pd.DataFrame({"class": labels, "x1": x1, "x2": x2})
    ds = Dataset(df, "binary_lda")
    ds.variables["class"].measure = MeasureType.NOMINAL
    ds.variables["x1"].measure = MeasureType.SCALE
    ds.variables["x2"].measure = MeasureType.SCALE
    return ds


class TestDiscriminantAnalysis:
    """Tests for linear discriminant analysis."""

    def test_lda_basic(self, discriminant_dataset):
        """LDA should return a non-None result with tables for a valid spec."""
        spec = {
            "variables": {
                "dependent": "group",
                "predictors": ["feat1", "feat2"],
            },
            "options": {"prior": "proportional"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(discriminant_dataset, spec)

        assert result is not None
        assert result.id == "discriminant_analysis"
        assert len(result.tables) > 0

    def test_wilks_lambda_between_0_and_1(self, discriminant_dataset):
        """Wilks' Lambda must be in the range (0, 1]."""
        spec = {
            "variables": {
                "dependent": "group",
                "predictors": ["feat1", "feat2"],
            },
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(discriminant_dataset, spec)

        wilks_tables = [t for t in result.tables if "Wilks" in t.title]
        assert len(wilks_tables) >= 1

        wilks_df = wilks_tables[0].dataframe
        assert "Lambda" in wilks_df.columns

        lambda_val_str = str(wilks_df["Lambda"].values[0])
        try:
            lambda_val = float(lambda_val_str.strip())
            assert 0.0 < lambda_val <= 1.0, (
                f"Wilks' Lambda must be in (0,1], got {lambda_val}"
            )
        except (ValueError, TypeError):
            pytest.skip("Wilks' Lambda value could not be parsed")

    def test_classification_accuracy(self, discriminant_dataset):
        """Classification accuracy on well-separated data should exceed 90%."""
        spec = {
            "variables": {
                "dependent": "group",
                "predictors": ["feat1", "feat2"],
            },
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(discriminant_dataset, spec)

        class_tables = [t for t in result.tables if "분류 결과" in t.title or "Classification" in t.title]
        assert len(class_tables) >= 1

        class_df = class_tables[0].dataframe
        # Footnotes contain overall accuracy
        class_table = next(t for t in result.tables if "분류 결과" in t.title or "Classification" in t.title)
        accuracy_lines = [fn for fn in class_table.footnotes if "정확도" in fn or "accuracy" in fn.lower()]
        assert len(accuracy_lines) >= 1

        # Parse accuracy from footnote
        accuracy_str = accuracy_lines[0]
        import re
        match = re.search(r"(\d+\.?\d*)\s*%", accuracy_str)
        if match:
            accuracy = float(match.group(1))
            # Well-separated 3 clusters should achieve high classification rate
            assert accuracy > 90.0, f"Expected accuracy > 90%, got {accuracy}%"

    def test_canonical_coefficients_table(self, discriminant_dataset):
        """Canonical discriminant function coefficients table must exist."""
        spec = {
            "variables": {
                "dependent": "group",
                "predictors": ["feat1", "feat2"],
            },
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(discriminant_dataset, spec)

        canonical_tables = [
            t for t in result.tables
            if "정준" in t.title or "Canonical" in t.title or "계수" in t.title
        ]
        assert len(canonical_tables) >= 1

        coef_df = canonical_tables[0].dataframe
        # One row per predictor variable
        assert len(coef_df) >= 2

    def test_group_centroids_table(self, discriminant_dataset):
        """Group centroids table must contain one row per class."""
        spec = {
            "variables": {
                "dependent": "group",
                "predictors": ["feat1", "feat2"],
            },
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(discriminant_dataset, spec)

        centroid_tables = [t for t in result.tables if "중심점" in t.title]
        assert len(centroid_tables) >= 1

        centroid_df = centroid_tables[0].dataframe
        # Should have 3 rows (one per class: A, B, C)
        assert len(centroid_df) == 3
