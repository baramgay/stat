"""Descriptive statistics analysis for StatWorkbench."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_pvalue, format_number, format_ci
from statworkbench.analysis.assumptions import (
    prepare_analysis_frame, get_case_processing_summary
)


def _compute_descriptives(
    series: pd.Series,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Compute descriptive statistics for a single series."""
    arr = series.dropna().values
    n = len(arr)
    if n == 0:
        return {
            "N": 0, "Missing": series.isna().sum(),
            "Mean": np.nan, "SD": np.nan,
            "Median": np.nan, "IQR": np.nan,
            "Min": np.nan, "Max": np.nan,
            "Skewness": np.nan, "Kurtosis": np.nan,
            "CI_Lower": np.nan, "CI_Upper": np.nan,
        }

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    skewness = float(stats.skew(arr, bias=False))
    kurtosis = float(stats.kurtosis(arr, bias=False))

    # Confidence interval for mean
    if n > 1:
        se = sd / np.sqrt(n)
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = ci_upper = mean

    return {
        "N": n,
        "Missing": series.isna().sum(),
        "Mean": mean,
        "SD": sd,
        "Median": median,
        "IQR": q3 - q1,
        "Min": min_val,
        "Max": max_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
    }


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run descriptive statistics analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "scale" list of variable names,
              optional "group" for grouping variable.
            - options: additional options dict.
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with descriptive statistics tables.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    scale_vars: list[str] = variables.get("scale", [])
    group_var: Optional[str] = variables.get("group", None)

    result = AnalysisResult(
        id="descriptive_statistics",
        title="Descriptive Statistics",
        spec=spec,
    )

    # Prepare data
    all_vars = list(scale_vars)
    if group_var is not None:
        all_vars.append(group_var)

    prepared = prepare_analysis_frame(
        dataset, variables=all_vars, missing_policy=missing_policy
    )
    df = prepared.data

    # Case Processing Summary
    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    if group_var is not None and group_var in df.columns:
        # Grouped descriptives
        groups = df[group_var].dropna().unique()
        rows = []
        for grp in sorted(groups):
            grp_df = df[df[group_var] == grp]
            for var in scale_vars:
                if var in grp_df.columns:
                    desc = _compute_descriptives(
                        grp_df[var], confidence_level=confidence_level
                    )
                    ci_str = format_ci(
                        desc["CI_Lower"], desc["CI_Upper"], level=confidence_level
                    )
                    rows.append({
                        "Group": grp,
                        "Variable": var,
                        "N": desc["N"],
                        "Missing": desc["Missing"],
                        "Mean": format_number(desc["Mean"], 3),
                        "SD": format_number(desc["SD"], 3),
                        "Median": format_number(desc["Median"], 3),
                        "IQR": format_number(desc["IQR"], 3),
                        "Min": format_number(desc["Min"], 3),
                        "Max": format_number(desc["Max"], 3),
                        "Skewness": format_number(desc["Skewness"], 3),
                        "Kurtosis": format_number(desc["Kurtosis"], 3),
                        "CI": ci_str,
                    })
        desc_df = pd.DataFrame(rows)
    else:
        # Overall descriptives
        rows = []
        for var in scale_vars:
            if var in df.columns:
                desc = _compute_descriptives(
                    df[var], confidence_level=confidence_level
                )
                ci_str = format_ci(
                    desc["CI_Lower"], desc["CI_Upper"], level=confidence_level
                )
                rows.append({
                    "Variable": var,
                    "N": desc["N"],
                    "Missing": desc["Missing"],
                    "Mean": format_number(desc["Mean"], 3),
                    "SD": format_number(desc["SD"], 3),
                    "Median": format_number(desc["Median"], 3),
                    "IQR": format_number(desc["IQR"], 3),
                    "Min": format_number(desc["Min"], 3),
                    "Max": format_number(desc["Max"], 3),
                    "Skewness": format_number(desc["Skewness"], 3),
                    "Kurtosis": format_number(desc["Kurtosis"], 3),
                    "CI": ci_str,
                })
        desc_df = pd.DataFrame(rows)

    desc_table = ResultTable(
        title="Descriptive Statistics",
        dataframe=desc_df,
    )
    result.add_table(desc_table)

    return result
