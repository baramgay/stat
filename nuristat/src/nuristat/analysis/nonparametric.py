"""Nonparametric test analysis for NuriStat."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def _rank_biserial_correlation(u: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U."""
    return (2 * u) / (n1 * n2) - 1


def _epsilon_squared(kruskal_h: float, n: int, k: int) -> float:
    """Compute epsilon-squared effect size for Kruskal-Wallis."""
    if n <= k:
        return 0.0
    return kruskal_h / ((n**2 - 1) / (n + 1))


def _kendalls_w(data: np.ndarray) -> float:
    """Compute Kendall's W for Friedman test.

    Args:
        data: 2D array with shape (subjects, conditions).

    Returns:
        Kendall's W coefficient.
    """
    n_subjects, k_conditions = data.shape
    if n_subjects == 0 or k_conditions <= 1:
        return 0.0

    # Rank data within each subject (row)
    ranks = np.apply_along_axis(lambda x: stats.rankdata(x), 1, data)
    mean_rank = np.mean(ranks, axis=0)
    overall_mean = (k_conditions + 1) / 2

    ss_between = n_subjects * np.sum((mean_rank - overall_mean) ** 2)
    ss_total = n_subjects * k_conditions * (k_conditions**2 - 1) / 12

    return float(ss_between / ss_total) if ss_total > 1e-12 else 0.0


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run nonparametric tests.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with test-specific variable assignments.
            - options: dict with test type.
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with nonparametric test results.
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    test_type = options.get("test", "mann_whitney")

    result = AnalysisResult(
        id="nonparametric_test",
        title="Nonparametric Test",
        spec=spec,
    )

    try:
        if test_type == "mann_whitney":
            return _mann_whitney(dataset, variables, missing_policy, result)
        elif test_type == "wilcoxon":
            return _wilcoxon_test(dataset, variables, missing_policy, result)
        elif test_type == "kruskal_wallis":
            return _kruskal_wallis(dataset, variables, missing_policy, result)
        elif test_type == "friedman":
            return _friedman_test(dataset, variables, missing_policy, result)
        else:
            result.warnings.append(f"Unknown test type: {test_type}")
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
    return result


def _mann_whitney(
    dataset: Dataset,
    variables: dict,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run Mann-Whitney U test."""
    dep_var: str = variables.get("dependent", "")
    group_var: str = variables.get("group", "")

    all_vars = [dep_var, group_var]
    prepared = prepare_analysis_frame(
        dataset, variables=all_vars, missing_policy=missing_policy
    )
    df = prepared.data

    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    groups = df[group_var].dropna().unique()
    if len(groups) != 2:
        result.warnings.append(
            f"Mann-Whitney U requires exactly 2 groups. Found {len(groups)}."
        )
        return result

    g1_val, g2_val = sorted(groups)[:2]
    g1_data = df[df[group_var] == g1_val][dep_var].dropna().values
    g2_data = df[df[group_var] == g2_val][dep_var].dropna().values

    n1, n2 = len(g1_data), len(g2_data)

    # Descriptives
    desc_rows = [
        {
            "Group": g1_val,
            "N": n1,
            "Median": format_number(float(np.median(g1_data)), 3),
        },
        {
            "Group": g2_val,
            "N": n2,
            "Median": format_number(float(np.median(g2_data)), 3),
        },
    ]

    u_stat, p_value = stats.mannwhitneyu(g1_data, g2_data, alternative="two-sided")

    # Rank-biserial correlation
    rbc = _rank_biserial_correlation(u_stat, n1, n2)

    test_rows = [
        {
            "Statistic": "Mann-Whitney U",
            "Value": format_number(u_stat, 1),
        },
        {
            "Statistic": "p-value",
            "Value": format_pvalue(p_value),
        },
        {
            "Statistic": "Rank-Biserial r",
            "Value": format_number(rbc, 3),
        },
        {
            "Statistic": "N (Group 1)",
            "Value": n1,
        },
        {
            "Statistic": "N (Group 2)",
            "Value": n2,
        },
    ]

    result.add_table(ResultTable(
        title="Ranks",
        dataframe=pd.DataFrame(desc_rows),
    ))
    result.add_table(ResultTable(
        title="Test Statistics",
        dataframe=pd.DataFrame(test_rows),
    ))

    return result


def _wilcoxon_test(
    dataset: Dataset,
    variables: dict,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run Wilcoxon signed-rank test."""
    paired_vars: list[str] = variables.get("paired", [])
    if len(paired_vars) != 2:
        result.warnings.append(
            "Wilcoxon test requires exactly 2 paired variables."
        )
        return result

    var1, var2 = paired_vars[0], paired_vars[1]

    prepared = prepare_analysis_frame(
        dataset, variables=paired_vars, missing_policy=missing_policy
    )
    df = prepared.data

    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    x1 = df[var1].values
    x2 = df[var2].values
    n = len(x1)

    diff = x1 - x2
    median1 = float(np.median(x1))
    median2 = float(np.median(x2))
    mean_diff = float(np.mean(diff))

    desc_rows = [
        {
            "Variable": var1,
            "N": n,
            "Median": format_number(median1, 3),
        },
        {
            "Variable": var2,
            "N": n,
            "Median": format_number(median2, 3),
        },
    ]

    # Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(x1, x2)

    # Effect size r
    z_score = (w_stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    r = abs(z_score) / np.sqrt(n) if n > 0 else float("nan")

    test_rows = [
        {
            "Statistic": "Wilcoxon W",
            "Value": format_number(w_stat, 1),
        },
        {
            "Statistic": "p-value",
            "Value": format_pvalue(p_value),
        },
        {
            "Statistic": "Effect Size r",
            "Value": format_number(r, 3),
        },
        {
            "Statistic": "Mean Difference",
            "Value": format_number(mean_diff, 3),
        },
        {
            "Statistic": "N",
            "Value": n,
        },
    ]

    result.add_table(ResultTable(
        title="Ranks",
        dataframe=pd.DataFrame(desc_rows),
    ))
    result.add_table(ResultTable(
        title="Test Statistics",
        dataframe=pd.DataFrame(test_rows),
    ))

    return result


def _kruskal_wallis(
    dataset: Dataset,
    variables: dict,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run Kruskal-Wallis H test."""
    dep_var: str = variables.get("dependent", "")
    group_var: str = variables.get("group", "")

    all_vars = [dep_var, group_var]
    prepared = prepare_analysis_frame(
        dataset, variables=all_vars, missing_policy=missing_policy
    )
    df = prepared.data

    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    groups = sorted(df[group_var].dropna().unique())
    if len(groups) < 3:
        result.warnings.append(
            f"Kruskal-Wallis requires at least 3 groups. Found {len(groups)}."
        )
        return result

    group_data = []
    desc_rows = []
    for grp in groups:
        arr = df[df[group_var] == grp][dep_var].dropna().values
        group_data.append(arr)
        desc_rows.append({
            "Group": grp,
            "N": len(arr),
            "Median": format_number(float(np.median(arr)), 3) if len(arr) > 0 else "",
        })

    h_stat, p_value = stats.kruskal(*group_data)
    n_total = sum(len(g) for g in group_data)
    k = len(group_data)
    eps_sq = _epsilon_squared(h_stat, n_total, k)

    test_rows = [
        {
            "Statistic": "Kruskal-Wallis H",
            "Value": format_number(h_stat, 3),
        },
        {
            "Statistic": "df",
            "Value": k - 1,
        },
        {
            "Statistic": "p-value",
            "Value": format_pvalue(p_value),
        },
        {
            "Statistic": "Epsilon-squared",
            "Value": format_number(eps_sq, 3),
        },
        {
            "Statistic": "N",
            "Value": n_total,
        },
    ]

    result.add_table(ResultTable(
        title="Ranks",
        dataframe=pd.DataFrame(desc_rows),
    ))
    result.add_table(ResultTable(
        title="Test Statistics",
        dataframe=pd.DataFrame(test_rows),
    ))

    return result


def _friedman_test(
    dataset: Dataset,
    variables: dict,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run Friedman test."""
    repeated_vars: list[str] = variables.get("repeated", [])
    if len(repeated_vars) < 3:
        result.warnings.append(
            "Friedman test requires at least 3 repeated measures."
        )
        return result

    prepared = prepare_analysis_frame(
        dataset, variables=repeated_vars, missing_policy=missing_policy
    )
    df = prepared.data

    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    # Get data matrix
    data_matrix = df[repeated_vars].values
    n_subjects = len(data_matrix)

    # Friedman test
    args = [data_matrix[:, i] for i in range(len(repeated_vars))]
    chi2_stat, p_value = stats.friedmanchisquare(*args)

    # Kendall's W
    kendall_w = _kendalls_w(data_matrix)

    # Descriptives
    desc_rows = []
    for i, var in enumerate(repeated_vars):
        arr = data_matrix[:, i]
        desc_rows.append({
            "Condition": var,
            "N": len(arr),
            "Median": format_number(float(np.median(arr)), 3),
            "Mean": format_number(float(np.mean(arr)), 3),
        })

    k = len(repeated_vars)
    test_rows = [
        {
            "Statistic": "Chi-Square",
            "Value": format_number(chi2_stat, 3),
        },
        {
            "Statistic": "df",
            "Value": k - 1,
        },
        {
            "Statistic": "p-value",
            "Value": format_pvalue(p_value),
        },
        {
            "Statistic": "Kendall's W",
            "Value": format_number(kendall_w, 3),
        },
        {
            "Statistic": "N",
            "Value": n_subjects,
        },
    ]

    result.add_table(ResultTable(
        title="Ranks",
        dataframe=pd.DataFrame(desc_rows),
    ))
    result.add_table(ResultTable(
        title="Test Statistics",
        dataframe=pd.DataFrame(test_rows),
    ))

    return result
