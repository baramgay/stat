"""ml_engine.py 미커버 라인 보강 테스트.

대상 라인:
  68-69   : silhouette_score 예외 → except Exception: pass
  113-114 : object dtype 특성 → LabelEncoder 인코딩 경로
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.ml_engine import (
    kmeans_clustering,
    decision_tree_classifier,
)


# ---------------------------------------------------------------------------
# Lines 68-69: silhouette_score 예외 → except Exception: pass
# ---------------------------------------------------------------------------

class TestKMeansSilhouetteException:

    def test_silhouette_exception_silenced(self):
        """silhouette_score 예외 → except pass(68-69) → result["silhouette"]=None."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 30),
            "b": rng.normal(0, 1, 30),
        })

        with patch(
            "sklearn.metrics.silhouette_score",
            side_effect=RuntimeError("silhouette fail"),
        ):
            result = kmeans_clustering(df, features=["a", "b"], n_clusters=3)

        assert result["silhouette"] is None


# ---------------------------------------------------------------------------
# Lines 113-114: object dtype 특성 → LabelEncoder
# ---------------------------------------------------------------------------

class TestDecisionTreeObjectDtypeFeature:

    def test_object_feature_gets_label_encoded(self):
        """object dtype 특성 → LabelEncoder 적용(113-114)."""
        rng = np.random.default_rng(7)
        n = 60
        df = pd.DataFrame({
            "cat": (["low"] * 20 + ["mid"] * 20 + ["high"] * 20),
            "val": rng.normal(0, 1, n),
            "target": ([0] * 30 + [1] * 30),
        })

        result = decision_tree_classifier(
            df,
            features=["cat", "val"],
            target="target",
            test_size=0.25,
            random_state=42,
        )
        assert "accuracy" in result
        assert result["n_features"] == 2
