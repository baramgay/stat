"""Crosstab (contingency table) analysis for StatWorkbench."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def _compute_cramers_v(contingency: np.ndarray) -> float:
    """Compute Cramer's V effect size."""
    chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    if n == 0:
        return np.nan
    min_dim = min(contingency.shape) - 1
    if min_dim == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min_dim)))


def _compute_phi(contingency: np.ndarray) -> float:
    """Compute Phi coefficient for 2x2 tables."""
    chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    if n == 0:
        return np.nan
    return float(np.sqrt(chi2 / n))


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run crosstab analysis with chi-square tests.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "row", "column", optional "layer".
            - options: dict with test options.
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with crosstab tables and test statistics.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    row_var: str = variables.get("row", "")
    col_var: str = variables.get("column", "")
    layer_var: str | None = variables.get("layer", None)

    result = AnalysisResult(
        id="crosstab",
        title="Crosstabulation",
        spec=spec,
    )

    all_vars = [row_var, col_var]
    if layer_var is not None:
        all_vars.append(layer_var)

    # Prepare data
    try:
        prepared = prepare_analysis_frame(
            dataset, variables=all_vars, missing_policy=missing_policy
        )
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result

    df = prepared.data

    # Case Processing Summary
    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    if layer_var is not None and layer_var in df.columns:
        layers = df[layer_var].dropna().unique()
        for layer_val in sorted(layers):
            layer_df = df[df[layer_var] == layer_val]
            _build_crosstab(result, layer_df, row_var, col_var, options,
                            confidence_level, layer=f"{layer_var}={layer_val}")
    else:
        _build_crosstab(result, df, row_var, col_var, options, confidence_level)

    return result


def _build_crosstab(
    result: AnalysisResult,
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    options: dict,
    confidence_level: float,
    layer: str | None = None,
) -> None:
    """Build crosstab tables and tests for a single layer."""
    title_prefix = f"Layer: {layer} — " if layer else ""

    # Create contingency table
    contingency = pd.crosstab(df[row_var], df[col_var])
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    grand_total = contingency.sum().sum()

    if grand_total == 0:
        result.warnings.append(f"{title_prefix}No valid data for crosstab.")
        return

    # Count table
    count_table = contingency.copy()
    count_table.loc["Total"] = col_totals
    count_table["Total"] = row_totals.reindex(count_table.index, fill_value=0)
    count_table.loc["Total", "Total"] = grand_total

    result.add_table(ResultTable(
        title=f"{title_prefix}Crosstabulation — Count",
        dataframe=count_table,
    ))

    # Row percentages
    row_pct = contingency.div(row_totals, axis=0) * 100
    row_pct = row_pct.fillna(0).round(1)
    result.add_table(ResultTable(
        title=f"{title_prefix}Row Percentages",
        dataframe=row_pct,
    ))

    # Column percentages
    col_pct = contingency.div(col_totals, axis=1) * 100
    col_pct = col_pct.fillna(0).round(1)
    result.add_table(ResultTable(
        title=f"{title_prefix}Column Percentages",
        dataframe=col_pct,
    ))

    # Total percentages
    total_pct = contingency / grand_total * 100
    total_pct = total_pct.round(1)
    result.add_table(ResultTable(
        title=f"{title_prefix}Total Percentages",
        dataframe=total_pct,
    ))

    # Expected frequencies
    expected = np.outer(row_totals.values, col_totals.values) / grand_total
    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    ).round(2)
    result.add_table(ResultTable(
        title=f"{title_prefix}Expected Frequencies",
        dataframe=expected_df,
    ))

    # Residuals
    residual = contingency.values - expected
    residual_df = pd.DataFrame(
        residual,
        index=contingency.index,
        columns=contingency.columns,
    ).round(2)
    result.add_table(ResultTable(
        title=f"{title_prefix}Residuals",
        dataframe=residual_df,
    ))

    # Standardized residuals
    with np.errstate(divide="ignore", invalid="ignore"):
        std_residual = residual / np.sqrt(expected)
    std_residual_df = pd.DataFrame(
        std_residual,
        index=contingency.index,
        columns=contingency.columns,
    ).round(2)
    result.add_table(ResultTable(
        title=f"{title_prefix}Standardized Residuals",
        dataframe=std_residual_df,
    ))

    # Chi-square tests
    test_rows = []
    contingency_arr = contingency.values

    # Check expected frequencies warning
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    pct_below_5 = (cells_below_5 / total_cells) * 100

    if cells_below_5 > 0:
        result.warnings.append(
            f"{title_prefix}{cells_below_5}/{total_cells} cells ({pct_below_5:.1f}%) "
            f"have expected frequency < 5. "
            f"Consider using Fisher's exact test or merging categories."
        )

    # Pearson chi-square
    try:
        chi2, p, dof, _ = stats.chi2_contingency(contingency_arr, correction=False)
        test_rows.append({
            "Test": "Pearson Chi-Square",
            "Value": format_number(chi2, 3),
            "df": dof,
            "p-value": format_pvalue(p),
        })
    except Exception as e:
        result.warnings.append(f"Pearson Chi-Square could not be computed: {e}")

    # Likelihood ratio
    try:
        chi2_lr, p_lr, dof_lr, _ = stats.chi2_contingency(
            contingency_arr, correction=False, lambda_="log-likelihood"
        )
        test_rows.append({
            "Test": "Likelihood Ratio",
            "Value": format_number(chi2_lr, 3),
            "df": dof_lr,
            "p-value": format_pvalue(p_lr),
        })
    except Exception as e:
        result.warnings.append(f"Likelihood Ratio test could not be computed: {e}")

    # Continuity correction (Yates)
    if contingency_arr.shape == (2, 2):
        try:
            chi2_cc, p_cc, dof_cc, _ = stats.chi2_contingency(
                contingency_arr, correction=True
            )
            test_rows.append({
                "Test": "Continuity Corrected",
                "Value": format_number(chi2_cc, 3),
                "df": dof_cc,
                "p-value": format_pvalue(p_cc),
            })
        except Exception as e:
            result.warnings.append(
                f"Continuity Correction could not be computed: {e}"
            )

    # Fisher's exact test for 2x2
    if contingency_arr.shape == (2, 2):
        try:
            oddsratio, p_fisher = stats.fisher_exact(contingency_arr)
            test_rows.append({
                "Test": "Fisher's Exact Test",
                "Value": format_number(oddsratio, 3),
                "df": "",
                "p-value": format_pvalue(p_fisher),
            })
        except Exception as e:
            result.warnings.append(f"Fisher's Exact test could not be computed: {e}")

    # Effect sizes
    try:
        cramers_v = _compute_cramers_v(contingency_arr)
        test_rows.append({
            "Test": "Cramer's V",
            "Value": format_number(cramers_v, 3),
            "df": "",
            "p-value": "",
        })
    except Exception as exc:
        logger.warning("Cramer's V 계산 실패: %s", exc)

    if contingency_arr.shape == (2, 2):
        try:
            phi = _compute_phi(contingency_arr)
            test_rows.append({
                "Test": "Phi Coefficient",
                "Value": format_number(phi, 3),
                "df": "",
                "p-value": "",
            })
        except Exception as exc:
            logger.warning("Phi Coefficient 계산 실패: %s", exc)

    test_df = pd.DataFrame(test_rows)
    result.add_table(ResultTable(
        title=f"{title_prefix}Chi-Square Tests",
        dataframe=test_df,
    ))
