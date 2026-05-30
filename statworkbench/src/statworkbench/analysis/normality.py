"""Normality test analysis for StatWorkbench."""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import pandas as pd
from scipy import stats

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run normality test (Shapiro-Wilk).

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "target" list of variable names.
            - options: additional options dict.
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with normality test results.
    """
    variables = spec.get("variables", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    target_vars: list[str] = variables.get("target", [])

    result = AnalysisResult(
        id="normality_test",
        title="Normality Test (Shapiro-Wilk)",
        spec=spec,
    )

    # Prepare data
    try:
        prepared = prepare_analysis_frame(
            dataset, variables=target_vars, missing_policy=missing_policy
        )
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result

    df = prepared.data

    try:
        # Case Processing Summary
        cps = get_case_processing_summary(
            prepared.n_total, prepared.n_valid, prepared.n_excluded,
            prepared.excluded_pct
        )
        result.add_table(cps)

        rows = []
        for var_name in target_vars:
            if var_name not in df.columns:
                result.warnings.append(f"Variable '{var_name}' not found.")
                continue

            arr = df[var_name].dropna().values
            n = len(arr)

            if n < 3:
                rows.append({
                    "Variable": var_name,
                    "N": n,
                    "Statistic": "",
                    "df": n,
                    "p-value": "",
                    "Interpretation": "Insufficient data (N < 3)",
                })
                result.warnings.append(
                    f"Variable '{var_name}': Shapiro-Wilk requires at least 3 observations."
                )
                continue

            if n > 5000:
                stat, p = stats.normaltest(arr)
                test_name = "D'Agostino"
                result.warnings.append(
                    f"Variable '{var_name}': N = {n} > 5000. "
                    f"Using D'Agostino's normality test instead of Shapiro-Wilk. "
                    f"With very large samples, even trivial deviations from normality "
                    f"may be statistically significant. Consider visual inspection."
                )
            else:
                stat, p = stats.shapiro(arr)
                test_name = "Shapiro-Wilk"

            alpha = 1 - confidence_level

            if p < alpha:
                interpretation = (
                    f"Data significantly deviates from normal distribution "
                    f"({test_name}, p = {format_pvalue(p)})."
                )
            else:
                interpretation = (
                    f"No significant deviation from normal distribution "
                    f"({test_name}, p = {format_pvalue(p)})."
                )

            if n < 20:
                result.warnings.append(
                    f"Variable '{var_name}': N = {n} is small. "
                    f"Shapiro-Wilk test may have low power to detect non-normality."
                )

            rows.append({
                "Variable": var_name,
                "N": n,
                "Statistic": format_number(float(stat), 4),
                "df": n,
                "p-value": format_pvalue(float(p)),
                "Interpretation": interpretation,
            })

        normality_df = pd.DataFrame(rows)
        normality_table = ResultTable(
            title="Tests of Normality",
            dataframe=normality_df,
        )
        result.add_table(normality_table)

    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")

    return result
