"""Discriminant analysis for NuriStat.

Supports:
- Linear Discriminant Analysis (LDA) via sklearn
- Canonical discriminant function coefficients
- Classification function coefficients
- Group centroids
- Classification result matrix, accuracy, Wilks' Lambda
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run linear discriminant analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with:
                "dependent": str     - grouping variable
                "predictors": list[str] - predictor variables
            - options: dict with:
                "method": "standard" | "stepwise" (default "standard")
                "prior": "equal" | "proportional" (default "proportional")
            - confidence_level: CI level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with discriminant analysis tables.
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    dep_var: str = variables.get("dependent", "")
    predictors: list[str] = variables.get("predictors", variables.get("independent", []))
    prior_type: str = options.get("prior", "proportional")

    result = AnalysisResult(
        id="discriminant_analysis",
        title="선형 판별분석 (Linear Discriminant Analysis)",
        spec=spec,
    )

    if not _SKLEARN_AVAILABLE:
        result.warnings.append(
            "scikit-learn이 설치되지 않아 판별분석을 실행할 수 없습니다. "
            "'pip install scikit-learn'을 실행하세요."
        )
        return result

    if not dep_var or not predictors:
        result.warnings.append("집단 변수(dependent)와 예측변수(predictors)가 필요합니다.")
        return result

    all_vars = [dep_var] + predictors
    try:
        prepared = prepare_analysis_frame(dataset, variables=all_vars, missing_policy=missing_policy)
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result
    df = prepared.data

    result.add_table(get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded, prepared.excluded_pct
    ))

    if len(df) < 2:
        result.warnings.append("유효 관측치가 2개 미만입니다.")
        return result

    y = df[dep_var]
    n_classes = y.nunique()
    if n_classes < 2:
        result.warnings.append("집단 변수에 2개 이상의 범주가 필요합니다.")
        return result

    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    class_labels = le.classes_

    # Build predictor matrix (numeric only)
    X_df = df[predictors].copy()
    for col in predictors:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.dropna()

    # Realign y to rows that survived dropna
    idx = X_df.index
    y_encoded_aligned = le.transform(df.loc[idx, dep_var])
    X = X_df.values.astype(float)

    n, p = X.shape
    k = n_classes

    # Prior probabilities
    if prior_type == "equal":
        priors = np.ones(k) / k
    else:
        counts = np.bincount(y_encoded_aligned, minlength=k)
        priors = counts / counts.sum()

    # Group summary
    group_rows = []
    for i, label in enumerate(class_labels):
        mask = y_encoded_aligned == i
        group_rows.append({
            "집단": str(label),
            "N": int(mask.sum()),
            "사전 확률": format_number(float(priors[i]), 3),
        })
    result.add_table(ResultTable(
        title="집단별 케이스 수 및 사전 확률",
        dataframe=pd.DataFrame(group_rows),
    ))

    # Fit LDA
    lda = LinearDiscriminantAnalysis(solver="svd", priors=priors)
    lda.fit(X, y_encoded_aligned)

    n_components = min(k - 1, p)

    # Wilks' Lambda
    _add_wilks_lambda(result, X, y_encoded_aligned, k, p, n, predictors, class_labels)

    # Canonical discriminant function coefficients (standardized)
    _add_canonical_coefficients(result, lda, predictors, n_components)

    # Group centroids (in discriminant function space)
    centroids = lda.means_ @ lda.scalings_[:, :n_components]
    centroid_rows = []
    for i, label in enumerate(class_labels):
        row: dict = {"집단": str(label)}
        for j in range(n_components):
            row[f"함수{j+1}"] = format_number(float(centroids[i, j]), 4)
        centroid_rows.append(row)
    result.add_table(ResultTable(
        title="집단 중심점 (판별 함수 공간)",
        dataframe=pd.DataFrame(centroid_rows),
        footnotes=["각 집단의 판별 함수 점수 중심점입니다."],
    ))

    # Classification function coefficients (Fisher's linear discriminant functions)
    _add_classification_functions(result, X, y_encoded_aligned, predictors, class_labels, k)

    # Classification results
    y_pred = lda.predict(X)
    _add_classification_table(result, y_encoded_aligned, y_pred, class_labels)

    # Structure matrix (correlations between predictors and discriminant functions)
    _add_structure_matrix(result, X, y_encoded_aligned, lda, predictors, n_components)

    return result


def _add_wilks_lambda(
    result: AnalysisResult,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    p: int,
    n: int,
    predictors: list[str],
    class_labels: np.ndarray,
) -> None:
    """Compute and add Wilks' Lambda test table."""
    try:
        # Between-group scatter matrix
        overall_mean = X.mean(axis=0)
        S_B = np.zeros((p, p))
        S_W = np.zeros((p, p))
        for i in range(k):
            mask = y == i
            ni = mask.sum()
            group_mean = X[mask].mean(axis=0)
            diff = (group_mean - overall_mean).reshape(-1, 1)
            S_B += ni * (diff @ diff.T)
            centered = X[mask] - group_mean
            S_W += centered.T @ centered

        # Wilks' Lambda = det(S_W) / det(S_W + S_B)
        try:
            S_T = S_W + S_B
            det_W = np.linalg.det(S_W)
            det_T = np.linalg.det(S_T)
            if det_T > 0:
                wilks = det_W / det_T
            else:
                wilks = np.nan
        except np.linalg.LinAlgError:
            wilks = np.nan

        if not np.isnan(wilks) and wilks > 0:
            # Approximate chi-square
            df_chi = p * (k - 1)
            chi2 = -(n - 1 - (p + k) / 2) * np.log(wilks)
            p_val = 1 - stats.chi2.cdf(chi2, df=df_chi)
        else:
            chi2 = np.nan
            df_chi = p * (k - 1)
            p_val = np.nan

        wilks_rows = [
            {
                "검정": "Wilks' Lambda",
                "Lambda": format_number(wilks, 4),
                "Chi-square": format_number(chi2, 3),
                "df": str(int(df_chi)),
                "p-value": format_pvalue(p_val),
                "해석": "판별함수 유의" if (not np.isnan(p_val) and p_val < 0.05) else "비유의",
            }
        ]
        result.add_table(ResultTable(
            title="Wilks' Lambda 검정",
            dataframe=pd.DataFrame(wilks_rows),
            footnotes=[
                "Wilks' Lambda는 0에 가까울수록 집단 간 판별이 잘 됨을 의미합니다.",
                "p < .05이면 판별함수가 통계적으로 유의합니다.",
            ],
        ))
    except Exception as e:
        result.warnings.append(f"Wilks' Lambda 계산 실패: {e}")


def _add_canonical_coefficients(
    result: AnalysisResult,
    lda: LinearDiscriminantAnalysis,
    predictors: list[str],
    n_components: int,
) -> None:
    """Add canonical (raw) and standardized discriminant function coefficients."""
    scalings = lda.scalings_[:, :n_components]  # (n_predictors, n_components)

    coef_rows = []
    for i, var in enumerate(predictors):
        row: dict = {"변수": var}
        for j in range(n_components):
            row[f"함수{j+1} (비표준화)"] = format_number(float(scalings[i, j]), 4)
        coef_rows.append(row)

    # Add constant rows
    result.add_table(ResultTable(
        title="정준 판별 함수 계수 (비표준화)",
        dataframe=pd.DataFrame(coef_rows),
        footnotes=["각 함수는 집단을 최대한 분리하는 선형 결합입니다."],
    ))

    # Eigenvalues and explained variance
    eigenvalues = lda.explained_variance_ratio_
    eig_rows = []
    cum_pct = 0.0
    for j in range(n_components):
        pct = float(eigenvalues[j]) * 100 if j < len(eigenvalues) else np.nan
        cum_pct += pct
        eig_rows.append({
            "함수": j + 1,
            "고유값": format_number(float(eigenvalues[j]) if j < len(eigenvalues) else np.nan, 4),
            "분산 설명(%)": format_number(pct, 2),
            "누적(%)": format_number(cum_pct, 2),
        })
    result.add_table(ResultTable(
        title="정준 판별 함수 - 고유값 및 설명 분산",
        dataframe=pd.DataFrame(eig_rows),
    ))


def _add_classification_functions(
    result: AnalysisResult,
    X: np.ndarray,
    y: np.ndarray,
    predictors: list[str],
    class_labels: np.ndarray,
    k: int,
) -> None:
    """Compute Fisher's linear classification function coefficients."""
    try:
        # Pooled within-group covariance
        p = X.shape[1]
        S_W = np.zeros((p, p))
        group_means = []
        group_ns = []

        for i in range(k):
            mask = y == i
            ni = mask.sum()
            gm = X[mask].mean(axis=0)
            group_means.append(gm)
            group_ns.append(ni)
            centered = X[mask] - gm
            S_W += centered.T @ centered

        n_total = X.shape[0]
        S_W_pooled = S_W / (n_total - k)

        try:
            S_W_inv = np.linalg.inv(S_W_pooled)
        except np.linalg.LinAlgError:
            S_W_inv = np.linalg.pinv(S_W_pooled)

        cf_rows = []
        for i, var in enumerate(predictors):
            row: dict = {"변수": var}
            for cls_idx, label in enumerate(class_labels):
                coef = S_W_inv[i, :] @ group_means[cls_idx]
                row[f"집단: {label}"] = format_number(float(coef), 4)
            cf_rows.append(row)

        # Constant terms
        const_row: dict = {"변수": "(상수)"}
        for cls_idx, label in enumerate(class_labels):
            const = -0.5 * group_means[cls_idx] @ S_W_inv @ group_means[cls_idx]
            const_row[f"집단: {label}"] = format_number(float(const), 4)
        cf_rows.append(const_row)

        result.add_table(ResultTable(
            title="분류 함수 계수 (Fisher's Linear Discriminant)",
            dataframe=pd.DataFrame(cf_rows),
            footnotes=["가장 큰 함수 값을 주는 집단으로 케이스가 분류됩니다."],
        ))
    except Exception as e:
        result.warnings.append(f"분류 함수 계수 계산 실패: {e}")


def _add_classification_table(
    result: AnalysisResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: np.ndarray,
) -> None:
    """Add classification result matrix and overall accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(y_true)
    labels_str = [str(lbl) for lbl in class_labels]

    cm_data = []
    for i, label in enumerate(labels_str):
        row: dict = {"실제 집단": label}
        for j, pred_label in enumerate(labels_str):
            row[f"예측: {pred_label}"] = int(cm[i, j]) if i < cm.shape[0] and j < cm.shape[1] else 0
        total_row = int(cm[i, :].sum()) if i < cm.shape[0] else 0
        correct = int(cm[i, i]) if i < cm.shape[0] and i < cm.shape[1] else 0
        row["정확도(행)"] = format_number(correct / total_row * 100 if total_row > 0 else np.nan, 1) + "%"
        cm_data.append(row)

    accuracy = float(accuracy_score(y_true, y_pred)) * 100
    result.add_table(ResultTable(
        title="분류 결과 행렬",
        dataframe=pd.DataFrame(cm_data),
        footnotes=[
            f"전체 정확도: {format_number(accuracy, 1)}%",
            f"정확하게 분류된 케이스: {int(np.diag(cm).sum())} / {n}",
        ],
    ))


def _add_structure_matrix(
    result: AnalysisResult,
    X: np.ndarray,
    y: np.ndarray,
    lda: LinearDiscriminantAnalysis,
    predictors: list[str],
    n_components: int,
) -> None:
    """Add structure matrix: correlations between predictors and discriminant scores."""
    try:
        scores = lda.transform(X)  # (n, n_components)
        struct_rows = []
        for i, var in enumerate(predictors):
            row: dict = {"변수": var}
            for j in range(n_components):
                if scores.shape[1] > j:
                    try:
                        corr, _ = stats.pearsonr(X[:, i], scores[:, j])
                        row[f"함수{j+1}"] = format_number(float(corr), 3)
                    except Exception:
                        row[f"함수{j+1}"] = ""
                else:
                    row[f"함수{j+1}"] = ""
            struct_rows.append(row)
        result.add_table(ResultTable(
            title="구조 행렬 (Structure Matrix)",
            dataframe=pd.DataFrame(struct_rows),
            footnotes=[
                "구조 행렬: 각 예측변수와 판별 점수 간의 상관계수.",
                "절댓값이 클수록 해당 함수에 대한 기여가 큽니다.",
            ],
        ))
    except Exception as e:
        result.warnings.append(f"구조 행렬 계산 실패: {e}")
