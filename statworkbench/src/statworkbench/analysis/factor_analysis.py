"""Factor analysis and PCA for StatWorkbench.

Supports:
- Exploratory Factor Analysis (EFA) with Varimax rotation
- Principal Component Analysis (PCA)
- KMO index and Bartlett's test of sphericity
- Factor loading matrix, communalities, explained variance
- Scree plot data (eigenvalues)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.decomposition import PCA, FactorAnalysis
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run factor analysis or PCA.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "variables" (list[str]).
            - options: dict with:
                "method": "efa" | "pca" (default "efa")
                "n_factors": int or "auto"
                "rotation": "varimax" | "none" (default "varimax")
                "extraction": "ml" | "pa" (default "ml")
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with factor analysis tables.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    var_list: list[str] = variables.get("variables", [])
    method: str = options.get("method", "efa")
    n_factors_opt = options.get("n_factors", "auto")
    rotation: str = options.get("rotation", "varimax")

    result = AnalysisResult(
        id="factor_analysis",
        title="요인분석 / PCA" if method == "efa" else "주성분분석 (PCA)",
        spec=spec,
    )

    if len(var_list) < 2:
        result.warnings.append("요인분석에는 2개 이상의 변수가 필요합니다.")
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

    if len(df) < len(var_list):
        result.warnings.append("유효 관측치 수가 변수 수보다 작습니다. 분석이 불안정할 수 있습니다.")
        return result

    try:
        X = df[var_list].values.astype(float)
        n_obs, n_vars = X.shape

        # KMO and Bartlett's test
        _add_kmo_bartlett(result, X, var_list, n_obs)

        # Determine number of factors/components
        if isinstance(n_factors_opt, int):
            n_factors = min(n_factors_opt, n_vars)
        else:
            n_factors = _auto_n_factors(X)

        n_factors = max(1, min(n_factors, n_vars - 1))

        if not _SKLEARN_AVAILABLE:
            result.warnings.append(
                "scikit-learn이 설치되지 않아 PCA/EFA를 실행할 수 없습니다. "
                "'pip install scikit-learn'을 실행하세요."
            )
            return result

        if method == "pca":
            _run_pca(result, X, var_list, n_factors, rotation)
        else:
            _run_efa(result, X, var_list, n_factors, rotation)
    except Exception as exc:
        result.add_warning(f"요인분석 계산 오류: {exc}")

    return result


def _auto_n_factors(X: np.ndarray) -> int:
    """Automatically determine number of factors via eigenvalue > 1 rule."""
    cov = np.corrcoef(X.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    n = int(np.sum(eigenvalues >= 1.0))
    return max(1, n)


def _add_kmo_bartlett(
    result: AnalysisResult,
    X: np.ndarray,
    var_list: list[str],
    n_obs: int,
) -> None:
    """Compute KMO index and Bartlett's test of sphericity."""
    try:
        corr = np.corrcoef(X.T)
        n_vars = corr.shape[0]

        # Partial correlation matrix for KMO
        try:
            corr_inv = np.linalg.inv(corr)
            # Anti-image correlation matrix
            d = np.sqrt(np.diag(corr_inv))
            partial_corr = -corr_inv / np.outer(d, d)
            np.fill_diagonal(partial_corr, 0.0)
            np.fill_diagonal(corr, 0.0)

            sum_sq_r = np.sum(corr ** 2)
            sum_sq_p = np.sum(partial_corr ** 2)
            np.fill_diagonal(corr, 1.0)

            kmo = sum_sq_r / (sum_sq_r + sum_sq_p) if (sum_sq_r + sum_sq_p) > 0 else np.nan
        except np.linalg.LinAlgError:
            kmo = np.nan

        # Bartlett's test
        corr_full = np.corrcoef(X.T)
        det = np.linalg.det(corr_full)
        det_safe = max(float(det), 1e-300)
        if det > 0:
            chi2_stat = -(n_obs - 1 - (2 * n_vars + 5) / 6) * np.log(det_safe)
        else:
            chi2_stat = np.nan
        df_bartlett = n_vars * (n_vars - 1) / 2
        bartlett_p = 1 - stats.chi2.cdf(chi2_stat, df=df_bartlett) if not np.isnan(chi2_stat) else np.nan

        kmo_interp = ""
        if not np.isnan(kmo):
            if kmo >= 0.9:
                kmo_interp = "탁월(Marvelous)"
            elif kmo >= 0.8:
                kmo_interp = "훌륭(Meritorious)"
            elif kmo >= 0.7:
                kmo_interp = "보통(Middling)"
            elif kmo >= 0.6:
                kmo_interp = "보통 이하(Mediocre)"
            else:
                kmo_interp = "불량(Miserable)"

        rows = [
            {"검정": "KMO 측도", "값": format_number(kmo, 3), "해석": kmo_interp},
            {"검정": "Bartlett 구형성 Chi-square", "값": format_number(chi2_stat, 3), "해석": ""},
            {"검정": "Bartlett df", "값": str(int(df_bartlett)), "해석": ""},
            {"검정": "Bartlett p-value", "값": format_pvalue(bartlett_p), "해석": "유의: 요인분석 적합" if (not np.isnan(bartlett_p) and bartlett_p < 0.05) else "비유의"},
        ]
        result.add_table(ResultTable(
            title="KMO 및 Bartlett 구형성 검정",
            dataframe=pd.DataFrame(rows),
            footnotes=["KMO >= 0.6이면 요인분석 적합. Bartlett p < .05이면 변수 간 상관이 존재함."],
        ))
    except Exception as e:
        result.warnings.append(f"KMO/Bartlett 계산 실패: {e}")


def _run_pca(
    result: AnalysisResult,
    X: np.ndarray,
    var_list: list[str],
    n_components: int,
    rotation: str,
) -> None:
    """Run Principal Component Analysis."""
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Eigenvalues and explained variance
    corr = np.corrcoef(X.T)
    eigenvalues_all = np.linalg.eigvalsh(corr)[::-1]

    scree_rows = []
    cum_pct = 0.0
    total_ev = float(sum(eigenvalues_all))
    for i, ev in enumerate(eigenvalues_all):
        pct = (ev / total_ev * 100) if total_ev > 0 else 0.0
        cum_pct += pct
        scree_rows.append({
            "성분": i + 1,
            "고유값(Eigenvalue)": format_number(float(ev), 4),
            "설명 분산(%)": format_number(pct, 2),
            "누적 분산(%)": format_number(cum_pct, 2),
        })
    result.add_table(ResultTable(
        title="고유값 및 설명 분산 (Scree Plot 데이터)",
        dataframe=pd.DataFrame(scree_rows),
        footnotes=["고유값 >= 1인 성분을 선택하는 Kaiser 기준이 일반적입니다."],
    ))

    # Loading matrix
    loadings = pca.components_.T  # (n_vars, n_components)
    if rotation == "varimax":
        loadings = _varimax_rotation(loadings)

    loading_rows = []
    communalities = []
    for i, var in enumerate(var_list):
        row: dict = {"변수": var}
        for j in range(n_components):
            row[f"성분{j+1}"] = format_number(float(loadings[i, j]), 3)
        h2 = float(np.sum(loadings[i, :] ** 2))
        communalities.append(h2)
        row["공통성(h2)"] = format_number(h2, 3)
        loading_rows.append(row)

    rotation_label = "(Varimax 회전 후)" if rotation == "varimax" else "(회전 없음)"
    result.add_table(ResultTable(
        title=f"성분 부하량 행렬 {rotation_label}",
        dataframe=pd.DataFrame(loading_rows),
        footnotes=["부하량 절댓값 >= 0.4인 경우 해당 성분에 기여함."],
    ))

    result.notes.append(f"PCA: {n_components}개 성분 추출, 회전: {rotation}")


def _run_efa(
    result: AnalysisResult,
    X: np.ndarray,
    var_list: list[str],
    n_factors: int,
    rotation: str,
) -> None:
    """Run Exploratory Factor Analysis."""
    try:
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        fa.fit(X)
        loadings = fa.components_.T  # (n_vars, n_factors)
    except Exception as e:
        result.warnings.append(f"EFA 적합 실패, PCA로 대체합니다: {e}")
        _run_pca(result, X, var_list, n_factors, rotation)
        return

    # Eigenvalues from correlation matrix
    corr = np.corrcoef(X.T)
    eigenvalues_all = np.linalg.eigvalsh(corr)[::-1]

    scree_rows = []
    cum_pct = 0.0
    total_ev = float(sum(eigenvalues_all))
    for i, ev in enumerate(eigenvalues_all):
        pct = (ev / total_ev * 100) if total_ev > 0 else 0.0
        cum_pct += pct
        scree_rows.append({
            "요인": i + 1,
            "고유값": format_number(float(ev), 4),
            "설명 분산(%)": format_number(pct, 2),
            "누적 분산(%)": format_number(cum_pct, 2),
        })
    result.add_table(ResultTable(
        title="고유값 및 설명 분산",
        dataframe=pd.DataFrame(scree_rows),
    ))

    # Apply rotation
    if rotation == "varimax":
        loadings = _varimax_rotation(loadings)

    # Loading matrix + communalities
    loading_rows = []
    for i, var in enumerate(var_list):
        row: dict = {"변수": var}
        for j in range(n_factors):
            row[f"요인{j+1}"] = format_number(float(loadings[i, j]), 3)
        h2 = float(np.sum(loadings[i, :] ** 2))
        row["공통성(h2)"] = format_number(h2, 3)
        row["고유성(u2)"] = format_number(max(0.0, 1.0 - h2), 3)
        loading_rows.append(row)

    rotation_label = "(Varimax 회전 후)" if rotation == "varimax" else "(회전 없음)"
    result.add_table(ResultTable(
        title=f"요인 부하량 행렬 {rotation_label}",
        dataframe=pd.DataFrame(loading_rows),
        footnotes=["공통성(h2): 요인에 의해 설명되는 분산 비율. 고유성(u2) = 1 - h2."],
    ))

    # Factor variance contribution
    factor_var_rows = []
    for j in range(n_factors):
        col_ss = float(np.sum(loadings[:, j] ** 2))
        pct = col_ss / len(var_list) * 100
        factor_var_rows.append({
            "요인": j + 1,
            "SS Loadings": format_number(col_ss, 3),
            "분산 비율(%)": format_number(pct, 2),
            "누적 비율(%)": format_number(sum(
                np.sum(loadings[:, k] ** 2) / len(var_list) * 100
                for k in range(j + 1)
            ), 2),
        })
    result.add_table(ResultTable(
        title="요인별 분산 기여",
        dataframe=pd.DataFrame(factor_var_rows),
    ))

    result.notes.append(f"EFA: {n_factors}개 요인 추출, 회전: {rotation}")


def _varimax_rotation(loadings: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """Apply varimax rotation to a loading matrix.

    Args:
        loadings: Loading matrix of shape (n_vars, n_factors).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Rotated loading matrix of the same shape.
    """
    n_vars, n_factors = loadings.shape
    if n_factors < 2:
        return loadings

    rotation_matrix = np.eye(n_factors)
    var_old = 0.0

    for _ in range(max_iter):
        rotated = loadings @ rotation_matrix
        u, s, vt = np.linalg.svd(
            loadings.T @ (
                rotated ** 3
                - rotated @ np.diag(np.sum(rotated ** 2, axis=0)) / n_vars
            )
        )
        rotation_matrix = u @ vt
        var_new = float(np.sum(s))
        if abs(var_new - var_old) < tol:
            break
        var_old = var_new

    return loadings @ rotation_matrix
