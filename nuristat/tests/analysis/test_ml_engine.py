"""ML 엔진 테스트.

검증 항목:
- kmeans_clustering: 반환 키, 레이블 수, 실루엣 점수
- decision_tree_classifier: accuracy 범위, feature_importance 합
- linear_regression_ml: r2_score 범위, 계수 키 일치
- 빈 데이터 / 결측치 → ValueError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.ml_engine import (
    kmeans_clustering,
    decision_tree_classifier,
    linear_regression_ml,
)


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, 200),
        "x2": rng.normal(5, 2, 200),
        "x3": rng.uniform(0, 10, 200),
    })


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 150
    return pd.DataFrame({
        "f1": np.concatenate([rng.normal(0, 1, n // 3), rng.normal(5, 1, n // 3), rng.normal(10, 1, n // 3)]),
        "f2": np.concatenate([rng.normal(0, 1, n // 3), rng.normal(5, 1, n // 3), rng.normal(10, 1, n // 3)]),
        "label": ["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n // 3),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x1 = rng.uniform(0, 10, 100)
    x2 = rng.uniform(0, 5, 100)
    y = 3 * x1 + 2 * x2 + rng.normal(0, 0.5, 100)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ──────────────────────────────────────────────────────────────
# 1. K-Means 군집화
# ──────────────────────────────────────────────────────────────

class TestKMeansClustering:

    def test_returns_dict(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3)
        assert isinstance(result, dict)

    def test_required_keys(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3)
        for key in ["labels", "centers", "inertia", "n_iter", "silhouette"]:
            assert key in result

    def test_labels_length_matches_data(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3)
        assert len(result["labels"]) == len(numeric_df)

    def test_centers_count_matches_k(self, numeric_df):
        for k in [2, 3, 4]:
            result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=k)
            assert len(result["centers"]) == k

    def test_inertia_positive(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3)
        assert result["inertia"] > 0

    def test_silhouette_in_range(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3)
        s = result["silhouette"]
        if s is not None:
            assert -1.0 <= s <= 1.0

    def test_single_cluster_no_silhouette(self, numeric_df):
        result = kmeans_clustering(numeric_df, ["x1"], n_clusters=1)
        assert result["silhouette"] is None

    def test_all_nan_raises(self):
        df = pd.DataFrame({"x": [np.nan] * 10, "y": [np.nan] * 10})
        with pytest.raises(ValueError):
            kmeans_clustering(df, ["x", "y"], n_clusters=2)

    def test_reproducible_with_seed(self, numeric_df):
        r1 = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3, random_state=42)
        r2 = kmeans_clustering(numeric_df, ["x1", "x2"], n_clusters=3, random_state=42)
        assert r1["inertia"] == r2["inertia"]


# ──────────────────────────────────────────────────────────────
# 2. 의사결정나무 분류
# ──────────────────────────────────────────────────────────────

class TestDecisionTreeClassifier:

    def test_returns_dict(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        assert isinstance(result, dict)

    def test_required_keys(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        for key in ["accuracy", "n_features", "n_train", "n_test", "feature_importance", "max_depth"]:
            assert key in result

    def test_accuracy_in_range(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_n_features(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        assert result["n_features"] == 2

    def test_feature_importance_keys(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        assert set(result["feature_importance"].keys()) == {"f1", "f2"}

    def test_feature_importance_sums_to_one(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        total = sum(result["feature_importance"].values())
        assert abs(total - 1.0) < 1e-6

    def test_train_test_split_counts(self, classification_df):
        result = decision_tree_classifier(
            classification_df, ["f1", "f2"], "label", test_size=0.2
        )
        total = result["n_train"] + result["n_test"]
        assert total == len(classification_df)

    def test_empty_data_raises(self):
        df = pd.DataFrame({"f": [np.nan] * 5, "t": [np.nan] * 5})
        with pytest.raises(ValueError):
            decision_tree_classifier(df, ["f"], "t")

    def test_good_separability_high_accuracy(self, classification_df):
        result = decision_tree_classifier(classification_df, ["f1", "f2"], "label")
        assert result["accuracy"] > 0.8


# ──────────────────────────────────────────────────────────────
# 3. 선형 회귀 (ML 버전)
# ──────────────────────────────────────────────────────────────

class TestLinearRegressionML:

    def test_returns_dict(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert isinstance(result, dict)

    def test_required_keys(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        for key in ["r2_score", "mse", "rmse", "coefficients", "intercept", "n_train", "n_test"]:
            assert key in result

    def test_r2_in_range(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert result["r2_score"] <= 1.0

    def test_high_r2_on_clean_data(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert result["r2_score"] > 0.95

    def test_rmse_equals_sqrt_mse(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert abs(result["rmse"] - result["mse"] ** 0.5) < 1e-9

    def test_coefficients_keys(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert set(result["coefficients"].keys()) == {"x1", "x2"}

    def test_train_test_split(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y", test_size=0.3)
        assert result["n_train"] + result["n_test"] == len(regression_df)

    def test_empty_data_raises(self):
        df = pd.DataFrame({"x": [np.nan] * 5, "y": [np.nan] * 5})
        with pytest.raises(ValueError):
            linear_regression_ml(df, ["x"], "y")

    def test_mse_positive(self, regression_df):
        result = linear_regression_ml(regression_df, ["x1", "x2"], "y")
        assert result["mse"] >= 0
