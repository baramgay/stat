"""Correlation analysis for NuriStat."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.analysis.formatting import format_ci, format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def _fisher_z_transform(r: float, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for Pearson correlation using Fisher z-transform.

    Returns (ci_lower, ci_upper).
    """
    if n <= 3 or abs(r) >= 1:
        return np.nan, np.nan

    z = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3)
    alpha = 1 - confidence_level
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return float(ci_lower), float(ci_upper)


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run correlation analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "target" list of variable names.
            - options: dict with "method", "tail", "flag_significant",
              "pairwise".
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with correlation matrices.
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    target_vars: list[str] = variables.get("target", [])
    method: str = options.get("method", "pearson")
    tail: str = options.get("tail", "two-tailed")
    flag_significant = options.get("flag_significant", True)
    use_pairwise = options.get("pairwise", False)

    result = AnalysisResult(
        id="correlation",
        title="Correlation Analysis",
        spec=spec,
    )

    if len(target_vars) < 2:
        result.warnings.append(
            "Correlation analysis requires at least 2 variables."
        )
        return result

    # Prepare data
    try:
        if use_pairwise:
            df = dataset.data[target_vars].copy()
            n_total = len(df)
            n_valid = int(df.dropna().shape[0])
            n_excluded = n_total - n_valid
            excluded_pct = (n_excluded / n_total * 100) if n_total > 0 else 0.0
        else:
            prepared = prepare_analysis_frame(
                dataset, variables=target_vars, missing_policy=missing_policy
            )
            df = prepared.data
            n_total = prepared.n_total
            n_valid = prepared.n_valid
            n_excluded = prepared.n_excluded
            excluded_pct = prepared.excluded_pct
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result

    # Case Processing Summary
    cps = get_case_processing_summary(
        n_total, n_valid, n_excluded, excluded_pct
    )
    result.add_table(cps)

    # Build correlation matrix
    n_vars = len(target_vars)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    n_matrix = np.zeros((n_vars, n_vars), dtype=int)
    ci_low_matrix = np.zeros((n_vars, n_vars))
    ci_high_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i, n_vars):
            var_i = target_vars[i]
            var_j = target_vars[j]

            if use_pairwise:
                x = df[var_i].values
                y = df[var_j].values
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                n_pair = len(x_clean)
            else:
                x_clean = df[var_i].values
                y_clean = df[var_j].values
                n_pair = len(x_clean)

            if n_pair < 2:
                corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                p_matrix[i, j] = p_matrix[j, i] = np.nan
                n_matrix[i, j] = n_matrix[j, i] = 0
                continue

            if method == "pearson":
                r, p = stats.pearsonr(x_clean, y_clean)
                ci_low, ci_high = _fisher_z_transform(r, n_pair, confidence_level)
            elif method == "spearman":
                r, p = stats.spearmanr(x_clean, y_clean)
                ci_low, ci_high = np.nan, np.nan
            elif method == "kendall":
                r, p = stats.kendalltau(x_clean, y_clean)
                ci_low, ci_high = np.nan, np.nan
            else:
                result.warnings.append(f"Unknown method: {method}")
                return result

            if tail == "one-tailed":
                p = p / 2

            corr_matrix[i, j] = corr_matrix[j, i] = r
            p_matrix[i, j] = p_matrix[j, i] = p
            n_matrix[i, j] = n_matrix[j, i] = n_pair
            ci_low_matrix[i, j] = ci_low_matrix[j, i] = ci_low
            ci_high_matrix[i, j] = ci_high_matrix[j, i] = ci_high

    # Correlation matrix — raw floats rounded to 3 dp (SPSS standard for r)
    corr_df = pd.DataFrame(
        np.round(corr_matrix, 3),
        index=target_vars,
        columns=target_vars,
    )
    result.add_table(ResultTable(
        title=f"{method.capitalize()} Correlation Matrix",
        dataframe=corr_df,
    ))

    # P-value matrix — raw floats (formatted at render time)
    p_df = pd.DataFrame(
        p_matrix,
        index=target_vars,
        columns=target_vars,
    )
    result.add_table(ResultTable(
        title="p-value Matrix",
        dataframe=p_df,
    ))

    # N matrix
    n_df = pd.DataFrame(
        n_matrix,
        index=target_vars,
        columns=target_vars,
    )
    result.add_table(ResultTable(
        title="N Matrix",
        dataframe=n_df,
    ))

    # Detailed pairwise table
    detail_rows = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            r = corr_matrix[i, j]
            p = p_matrix[i, j]
            n = n_matrix[i, j]
            ci_low = ci_low_matrix[i, j]
            ci_high = ci_high_matrix[i, j]

            if np.isnan(r):
                continue

            stars = ""
            if flag_significant:
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"

            ci_str = ""
            if method == "pearson" and not np.isnan(ci_low):
                ci_str = format_ci(ci_low, ci_high, level=confidence_level)

            detail_rows.append({
                "Variable Pair": f"{target_vars[i]} — {target_vars[j]}",
                "r": format_number(r, 3),
                "p-value": format_pvalue(p),
                "N": n,
                "CI": ci_str,
                "Significance": stars,
            })

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        result.add_table(ResultTable(
            title="Pairwise Correlations",
            dataframe=detail_df,
        ))

    return result


class CorrelationEngine:
    """AnalysisPlugin wrapper for correlation analysis."""

    id = "pearson_correlation"
    name = "상관분석"
    category = "Correlate"
    description = "Pearson / Spearman / Kendall 상관행렬, Fisher z 변환 CI"
    variable_requirements: list[dict] = [
        {"role": "variables", "measure_types": ["scale"], "min_count": 2, "required": True},
    ]
    implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        return run_analysis(dataset, spec)
