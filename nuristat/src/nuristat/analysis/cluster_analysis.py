"""Cluster analysis for NuriStat.

Supports:
- K-means clustering (sklearn)
- Hierarchical clustering: Ward, Complete, Average linkage (scipy)
- Dendrogram data generation
- Silhouette coefficient, within/between cluster distances
- Cluster-wise descriptive statistics
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist, pdist

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.formatting import format_number
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run cluster analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "variables" (list[str]).
            - options: dict with:
                "method": "kmeans" | "hierarchical" (default "kmeans")
                "n_clusters": int (default 3)
                "linkage": "ward" | "complete" | "average" (default "ward")
                "standardize": bool (default True)
                "max_k": int for elbow method (default 10)
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with cluster analysis tables.
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    var_list: list[str] = variables.get("variables", [])
    method: str = options.get("method", "kmeans")
    n_clusters: int = int(options.get("n_clusters", 3))
    linkage_method: str = options.get("linkage", "ward")
    standardize: bool = bool(options.get("standardize", True))
    max_k: int = int(options.get("max_k", 10))

    result = AnalysisResult(
        id="cluster_analysis",
        title="K-평균 군집분석" if method == "kmeans" else "계층적 군집분석",
        spec=spec,
    )

    if len(var_list) < 1:
        result.warnings.append("군집분석에는 1개 이상의 변수가 필요합니다.")
        return result

    try:
        prepared = prepare_analysis_frame(dataset, variables=var_list, missing_policy=missing_policy)
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result
    df = prepared.data

    result.add_table(get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded, prepared.excluded_pct
    ))

    if len(df) < n_clusters:
        result.warnings.append(f"관측치 수({len(df)})가 군집 수({n_clusters})보다 적습니다.")
        return result

    X = df[var_list].values.astype(float)

    # Standardize
    if standardize:
        if _SKLEARN_AVAILABLE:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            # Manual z-score standardization
            means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=1)
            stds[stds == 0] = 1.0
            X_scaled = (X - means) / stds
        result.notes.append("변수는 Z-점수로 표준화되었습니다 (평균=0, 표준편차=1).")
    else:
        X_scaled = X

    if method == "kmeans":
        if not _SKLEARN_AVAILABLE:
            result.warnings.append(
                "scikit-learn이 설치되지 않아 K-평균 분석을 실행할 수 없습니다. "
                "'pip install scikit-learn'을 실행하거나 계층적 군집분석을 사용하세요."
            )
            return result
        _run_kmeans(result, X_scaled, df, var_list, n_clusters, max_k)
    else:
        _run_hierarchical(result, X_scaled, df, var_list, n_clusters, linkage_method)

    return result


def _run_kmeans(
    result: AnalysisResult,
    X: np.ndarray,
    df: pd.DataFrame,
    var_list: list[str],
    n_clusters: int,
    max_k: int,
) -> None:
    """Run K-means clustering and generate result tables."""

    # Elbow method: WCSS for k=1..max_k
    max_k = min(max_k, len(X) - 1)
    wcss_rows = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss_rows.append({
            "K (군집 수)": k,
            "WCSS (군집 내 SS)": format_number(float(km.inertia_), 3),
        })
    result.add_table(ResultTable(
        title="엘보우 방법 (Elbow Method) - 최적 K 선택",
        dataframe=pd.DataFrame(wcss_rows),
        footnotes=["WCSS 감소율이 급격히 줄어드는 K를 최적 군집 수로 선택합니다."],
    ))

    # Final K-means
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    centers = km_final.cluster_centers_

    # Cluster membership counts
    unique, counts = np.unique(labels, return_counts=True)
    membership_rows = [
        {"군집": int(c) + 1, "관측치 수": int(cnt), "비율(%)": format_number(cnt / len(labels) * 100, 1)}
        for c, cnt in zip(unique, counts)
    ]
    result.add_table(ResultTable(
        title="군집별 케이스 수",
        dataframe=pd.DataFrame(membership_rows),
    ))

    # Cluster centers
    center_rows = []
    for k in range(n_clusters):
        row: dict = {"군집": k + 1}
        for j, var in enumerate(var_list):
            row[var] = format_number(float(centers[k, j]), 3)
        center_rows.append(row)
    result.add_table(ResultTable(
        title="군집 중심점 (표준화 공간)",
        dataframe=pd.DataFrame(center_rows),
        footnotes=["중심점은 표준화된 공간에서의 값입니다."],
    ))

    # Cluster descriptive statistics (original scale)
    _add_cluster_descriptives(result, df, var_list, labels, n_clusters)

    # Within/Between cluster distances
    _add_distance_summary(result, X, labels, centers, n_clusters)

    # Silhouette coefficient
    if n_clusters > 1 and len(X) > n_clusters:
        _add_silhouette(result, X, labels)

    result.notes.append(
        f"K-평균 군집분석: K={n_clusters}, 반복 횟수={km_final.n_iter_}, "
        f"최종 WCSS={format_number(float(km_final.inertia_), 3)}"
    )


def _run_hierarchical(
    result: AnalysisResult,
    X: np.ndarray,
    df: pd.DataFrame,
    var_list: list[str],
    n_clusters: int,
    linkage_method: str,
) -> None:
    """Run hierarchical clustering and generate result tables."""
    # Compute linkage
    dist_matrix = pdist(X, metric="euclidean")
    Z = hierarchy.linkage(dist_matrix, method=linkage_method)

    # Dendrogram data
    dend_rows = []
    for i, (merge_dist, cluster_i, cluster_j) in enumerate(
        zip(Z[:, 2], Z[:, 0].astype(int), Z[:, 1].astype(int))
    ):
        dend_rows.append({
            "단계": i + 1,
            "군집1": int(cluster_i),
            "군집2": int(cluster_j),
            "병합 거리": format_number(float(merge_dist), 4),
            "새 군집 크기": int(Z[i, 3]),
        })
    result.add_table(ResultTable(
        title=f"병합 계수 (Agglomeration Schedule) - {linkage_method.capitalize()} 연결법",
        dataframe=pd.DataFrame(dend_rows),
        footnotes=["각 단계에서 병합된 두 군집과 병합 거리를 나타냅니다."],
    ))

    # Cut tree at n_clusters
    labels = hierarchy.fcluster(Z, n_clusters, criterion="maxclust") - 1  # 0-based

    unique, counts = np.unique(labels, return_counts=True)
    membership_rows = [
        {"군집": int(c) + 1, "관측치 수": int(cnt), "비율(%)": format_number(cnt / len(labels) * 100, 1)}
        for c, cnt in zip(unique, counts)
    ]
    result.add_table(ResultTable(
        title=f"군집별 케이스 수 (K={n_clusters})",
        dataframe=pd.DataFrame(membership_rows),
    ))

    # Cluster descriptive statistics
    _add_cluster_descriptives(result, df, var_list, labels, n_clusters)

    # Silhouette
    if n_clusters > 1 and len(X) > n_clusters:
        _add_silhouette(result, X, labels)

    result.notes.append(
        f"계층적 군집분석: 연결법={linkage_method}, K={n_clusters}"
    )


def _add_cluster_descriptives(
    result: AnalysisResult,
    df: pd.DataFrame,
    var_list: list[str],
    labels: np.ndarray,
    n_clusters: int,
) -> None:
    """Add cluster-wise descriptive statistics table."""
    rows = []
    for var in var_list:
        if var not in df.columns:
            continue
        row: dict = {"변수": var}
        for k in range(n_clusters):
            mask = labels == k
            vals = df.loc[df.index[mask], var].dropna().values if mask.sum() > 0 else np.array([])
            if len(vals) > 0:
                row[f"군집{k+1} 평균"] = format_number(float(np.mean(vals)), 3)
                row[f"군집{k+1} SD"] = format_number(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else "-"
            else:
                row[f"군집{k+1} 평균"] = ""
                row[f"군집{k+1} SD"] = ""
        rows.append(row)

    result.add_table(ResultTable(
        title="군집별 기술통계 (원래 척도)",
        dataframe=pd.DataFrame(rows),
        footnotes=["평균과 표준편차는 원래 척도 기준입니다."],
    ))


def _add_distance_summary(
    result: AnalysisResult,
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    n_clusters: int,
) -> None:
    """Add within- and between-cluster distance summary."""
    dist_rows = []
    for k in range(n_clusters):
        mask = labels == k
        members = X[mask]
        if len(members) > 1:
            within_dist = float(np.mean(pdist(members, metric="euclidean")))
        elif len(members) == 1:
            within_dist = 0.0
        else:
            within_dist = np.nan
        dist_rows.append({
            "군집": k + 1,
            "군집 내 평균 거리": format_number(within_dist, 4),
            "중심점 거리 (전체 평균)": format_number(
                float(np.mean(cdist(members, centers[k:k+1], metric="euclidean"))) if len(members) > 0 else np.nan,
                4
            ),
        })

    # Between-cluster distances
    between_rows = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            between_rows.append({
                "군집 A": i + 1,
                "군집 B": j + 1,
                "중심점 간 거리": format_number(d, 4),
            })

    result.add_table(ResultTable(
        title="군집 내 거리 요약",
        dataframe=pd.DataFrame(dist_rows),
    ))
    if between_rows:
        result.add_table(ResultTable(
            title="군집 간 중심점 거리",
            dataframe=pd.DataFrame(between_rows),
            footnotes=["거리가 클수록 두 군집이 더 잘 분리되어 있습니다."],
        ))


def _add_silhouette(
    result: AnalysisResult,
    X: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Compute and add silhouette coefficient table."""
    try:
        overall_sil = float(silhouette_score(X, labels, metric="euclidean"))
        sample_sil = silhouette_samples(X, labels, metric="euclidean")

        unique_labels = np.unique(labels)
        sil_rows = [
            {
                "군집": int(k) + 1,
                "케이스 수": int(np.sum(labels == k)),
                "평균 실루엣 계수": format_number(float(np.mean(sample_sil[labels == k])), 3),
            }
            for k in unique_labels
        ]
        sil_rows.append({
            "군집": "전체",
            "케이스 수": len(labels),
            "평균 실루엣 계수": format_number(overall_sil, 3),
        })

        interp = "우수" if overall_sil >= 0.7 else ("양호" if overall_sil >= 0.5 else ("보통" if overall_sil >= 0.25 else "불량"))
        result.add_table(ResultTable(
            title="실루엣 계수 (Silhouette Coefficient)",
            dataframe=pd.DataFrame(sil_rows),
            footnotes=[
                f"전체 실루엣 계수: {format_number(overall_sil, 3)} ({interp})",
                "계수 범위: -1 ~ 1. 1에 가까울수록 군집이 잘 분리됨.",
                ">= 0.7: 우수, 0.5-0.7: 양호, 0.25-0.5: 보통, < 0.25: 불량",
            ],
        ))
    except Exception as e:
        result.warnings.append(f"실루엣 계수 계산 실패: {e}")
