"""ML Engine — 기계학습 기본 기능 엔진.

scikit-learn 기반 기본 ML 기능을 제공합니다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def kmeans_clustering(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """K-Means 군집화를 수행합니다.

    Args:
        df: 데이터프레임
        features: 사용할 특성 변수 목록
        n_clusters: 군집 수
        random_state: 랜덤 시드

    Returns:
        군집화 결과 딕셔너리
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn이 설치되지 않았습니다. 'pip install scikit-learn'로 설치하세요.")

    # 데이터 준비
    X = df[features].dropna()

    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다.")

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 결과
    result = {
        "labels": labels.tolist(),
        "centers": kmeans.cluster_centers_.tolist(),
        "inertia": float(kmeans.inertia_),
        "n_iter": int(kmeans.n_iter_),
        "silhouette": None,
    }

    # 실루엣 점수
    try:
        from sklearn.metrics import silhouette_score
        if n_clusters > 1 and len(X) > n_clusters:
            result["silhouette"] = float(silhouette_score(X_scaled, labels))
    except Exception:
        pass

    return result


def decision_tree_classifier(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """의사결정나무 분류를 수행합니다.

    Args:
        df: 데이터프레임
        features: 특성 변수 목록
        target: 목표 변수
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드

    Returns:
        분류 결과 딕셔너리
    """
    try:
        from sklearn.metrics import accuracy_score, classification_report  # noqa: F401
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        raise ImportError("scikit-learn이 설치되지 않았습니다.")

    # 데이터 준비
    data = df[features + [target]].dropna()

    if len(data) == 0:
        raise ValueError("유효한 데이터가 없습니다.")

    X = data[features]
    y = data[target]

    # 범주형 변수 인코딩
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 모델 학습
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    clf.fit(X_train, y_train)

    # 예측
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": float(accuracy),
        "n_features": len(features),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": dict(zip(features, clf.feature_importances_.tolist())),
        "max_depth": clf.max_depth,
    }


def linear_regression_ml(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """선형 회귀 (ML 버전)을 수행합니다.

    Args:
        df: 데이터프레임
        features: 특성 변수 목록
        target: 목표 변수
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드

    Returns:
        회귀 결과 딕셔너리
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn이 설치되지 않았습니다.")

    # 데이터 준비
    data = df[features + [target]].dropna()

    if len(data) == 0:
        raise ValueError("유효한 데이터가 없습니다.")

    X = data[features]
    y = data[target]

    # 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 모델 학습
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # 예측
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return {
        "r2_score": float(r2),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "coefficients": dict(zip(features, reg.coef_.tolist())),
        "intercept": float(reg.intercept_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
