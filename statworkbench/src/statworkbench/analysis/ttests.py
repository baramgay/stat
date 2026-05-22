"""t-Test analysis for StatWorkbench."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_pvalue, format_number, format_ci, get_display_decimals
from statworkbench.analysis.assumptions import (
    prepare_analysis_frame, get_case_processing_summary, levene_test
)


def _cohens_d(x1: np.ndarray, x2: np.ndarray, pooled: bool = True) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    sd1 = np.std(x1, ddof=1)
    sd2 = np.std(x2, ddof=1)
    if pooled:
        pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        if pooled_sd == 0:
            return 0.0
        return float((np.mean(x1) - np.mean(x2)) / pooled_sd)
    else:
        avg_sd = (sd1 + sd2) / 2
        if avg_sd == 0:
            return 0.0
        return float((np.mean(x1) - np.mean(x2)) / avg_sd)


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run t-test analysis (independent or paired).

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "dependent" and "group" for independent t-test,
              or "paired" list of two variable names for paired t-test.
            - options: dict with "equal_var" (auto/yes/no).
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with t-test results.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    result = AnalysisResult(
        id="t_test",
        title="t-Test",
        spec=spec,
    )

    paired_vars: list[str] = variables.get("paired", [])
    dep_var: str = variables.get("dependent", "")
    group_var: str = variables.get("group", "")

    if paired_vars and len(paired_vars) == 2:
        return _paired_ttest(
            dataset, paired_vars, confidence_level, missing_policy, result
        )
    elif dep_var and group_var:
        return _independent_ttest(
            dataset, dep_var, group_var, options, confidence_level,
            missing_policy, result
        )
    else:
        result.warnings.append(
            "Invalid variable specification. Need either 'dependent'+'group' "
            "or 'paired' with two variable names."
        )
        return result


def _independent_ttest(
    dataset: Dataset,
    dep_var: str,
    group_var: str,
    options: dict,
    confidence_level: float,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run independent samples t-test."""
    equal_var_option = options.get("equal_var", "auto")

    all_vars = [dep_var, group_var]
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

    groups = df[group_var].dropna().unique()
    if len(groups) != 2:
        result.warnings.append(
            f"Independent t-test requires exactly 2 groups. Found {len(groups)}."
        )
        return result

    g1_val, g2_val = sorted(groups)[:2]
    g1_data = df[df[group_var] == g1_val][dep_var].dropna().values
    g2_data = df[df[group_var] == g2_val][dep_var].dropna().values

    n1, n2 = len(g1_data), len(g2_data)
    mean1, mean2 = float(np.mean(g1_data)), float(np.mean(g2_data))
    sd1 = float(np.std(g1_data, ddof=1))
    sd2 = float(np.std(g2_data, ddof=1))

    def _label(var: str) -> str:
        meta = dataset.variables.get(var) if dataset.variables else None
        return meta.label if (meta and meta.label) else var

    def _val_label(var: str, val) -> str:
        meta = dataset.variables.get(var) if dataset.variables else None
        if meta and meta.value_labels:
            key = int(val) if isinstance(val, float) and val == int(val) else val
            lbl = meta.value_labels.get(key) or meta.value_labels.get(str(key))
            if lbl:
                return lbl
        return str(val)

    # Group statistics table
    d = get_display_decimals(dataset, dep_var)
    group_stats = pd.DataFrame([
        {
            "Group": _val_label(group_var, g1_val),
            "N": n1,
            "Mean": format_number(mean1, d),
            "SD": format_number(sd1, d + 1),
            "SE": format_number(sd1 / np.sqrt(n1), d + 1) if n1 > 0 else "",
        },
        {
            "Group": _val_label(group_var, g2_val),
            "N": n2,
            "Mean": format_number(mean2, d),
            "SD": format_number(sd2, d + 1),
            "SE": format_number(sd2 / np.sqrt(n2), d + 1) if n2 > 0 else "",
        },
    ])
    result.add_table(ResultTable(
        title="Group Statistics",
        dataframe=group_stats,
    ))

    # Levene's test
    levene_stat, levene_p = levene_test(g1_data, g2_data)
    levene_table = pd.DataFrame([
        {
            "Test": "Levene's Test for Equality of Variances",
            "F": format_number(levene_stat, 3),
            "p-value": format_pvalue(levene_p),
        }
    ])
    result.add_table(ResultTable(
        title="Test for Equality of Variances",
        dataframe=levene_table,
    ))

    # Determine equal variance
    if equal_var_option == "auto":
        equal_var = levene_p >= 0.05
    elif equal_var_option == "yes":
        equal_var = True
    else:
        equal_var = False

    alpha = 1 - confidence_level

    # Compute both equal and unequal variance t-tests
    test_rows = []

    # Equal variances assumed
    t_eq, p_eq = stats.ttest_ind(g1_data, g2_data, equal_var=True)
    df_eq = n1 + n2 - 2
    mean_diff = mean1 - mean2
    if df_eq <= 0:
        result.warnings.append("Each group must have at least 2 cases for independent t-test.")
        return result
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df_eq)
    se_diff_eq = pooled_sd * np.sqrt(1 / n1 + 1 / n2)
    t_crit_eq = stats.t.ppf(1 - alpha / 2, df_eq)
    ci_low_eq = mean_diff - t_crit_eq * se_diff_eq
    ci_high_eq = mean_diff + t_crit_eq * se_diff_eq
    d_eq = _cohens_d(g1_data, g2_data, pooled=True)

    test_rows.append({
        "Variant": "Equal variances assumed",
        "t": format_number(t_eq, 3),
        "df": df_eq,
        "p-value": format_pvalue(p_eq),
        "Mean Difference": format_number(mean_diff, 3),
        "SE Difference": format_number(se_diff_eq, 3),
        "95% CI": format_ci(ci_low_eq, ci_high_eq, level=confidence_level),
        "Cohen's d": format_number(d_eq, 3),
    })

    # Unequal variances (Welch)
    t_uneq, p_uneq = stats.ttest_ind(g1_data, g2_data, equal_var=False)
    se1_sq = sd1**2 / n1
    se2_sq = sd2**2 / n2
    _welch_denom = se1_sq**2 / (n1 - 1) + se2_sq**2 / (n2 - 1)
    df_uneq = (se1_sq + se2_sq)**2 / _welch_denom if _welch_denom > 0 else float(n1 + n2 - 2)
    se_diff_uneq = np.sqrt(se1_sq + se2_sq)
    t_crit_uneq = stats.t.ppf(1 - alpha / 2, df_uneq)
    ci_low_uneq = mean_diff - t_crit_uneq * se_diff_uneq
    ci_high_uneq = mean_diff + t_crit_uneq * se_diff_uneq
    d_uneq = _cohens_d(g1_data, g2_data, pooled=False)

    test_rows.append({
        "Variant": "Equal variances not assumed",
        "t": format_number(t_uneq, 3),
        "df": format_number(df_uneq, 1),
        "p-value": format_pvalue(p_uneq),
        "Mean Difference": format_number(mean_diff, 3),
        "SE Difference": format_number(se_diff_uneq, 3),
        "95% CI": format_ci(ci_low_uneq, ci_high_uneq, level=confidence_level),
        "Cohen's d": format_number(d_uneq, 3),
    })

    test_df = pd.DataFrame(test_rows)
    result.add_table(ResultTable(
        title="Independent Samples t-Test",
        dataframe=test_df,
    ))

    if equal_var_option == "auto":
        result.notes.append(
            f"Equal variances {'assumed' if equal_var else 'not assumed'} "
            f"based on Levene's test (p = {format_pvalue(levene_p)})."
        )

    return result


def _paired_ttest(
    dataset: Dataset,
    paired_vars: list[str],
    confidence_level: float,
    missing_policy: MissingPolicy,
    result: AnalysisResult,
) -> AnalysisResult:
    """Run paired samples t-test."""
    var1, var2 = paired_vars[0], paired_vars[1]

    prepared = prepare_analysis_frame(
        dataset, variables=paired_vars, missing_policy=missing_policy
    )
    df = prepared.data

    # Case Processing Summary
    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    x1 = df[var1].values
    x2 = df[var2].values
    diff = x1 - x2
    n = len(diff)

    if n == 0:
        result.warnings.append("No valid paired observations after missing removal.")
        return result

    mean1 = float(np.mean(x1))
    mean2 = float(np.mean(x2))
    sd1 = float(np.std(x1, ddof=1))
    sd2 = float(np.std(x2, ddof=1))
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = sd_diff / np.sqrt(n)

    # Paired statistics — use stricter of the two variables' decimals
    d1 = get_display_decimals(dataset, var1)
    d2 = get_display_decimals(dataset, var2)
    d = max(d1, d2)
    paired_stats = pd.DataFrame([
        {
            "Variable": var1,
            "N": n,
            "Mean": format_number(mean1, d),
            "SD": format_number(sd1, d + 1),
            "SE": format_number(sd1 / np.sqrt(n), d + 1),
        },
        {
            "Variable": var2,
            "N": n,
            "Mean": format_number(mean2, d),
            "SD": format_number(sd2, d + 1),
            "SE": format_number(sd2 / np.sqrt(n), d + 1),
        },
    ])
    result.add_table(ResultTable(
        title="Paired Samples Statistics",
        dataframe=paired_stats,
    ))

    # t-test
    t_stat, p_value = stats.ttest_rel(x1, x2)
    df_val = n - 1
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    # Cohen's dz for paired
    dz = mean_diff / sd_diff if sd_diff > 0 else 0.0

    test_df = pd.DataFrame([
        {
            "Statistic": "Mean Difference",
            "Value": format_number(mean_diff, 3),
        },
        {
            "Statistic": "SD Difference",
            "Value": format_number(sd_diff, 3),
        },
        {
            "Statistic": "SE Difference",
            "Value": format_number(se_diff, 3),
        },
        {
            "Statistic": "t",
            "Value": format_number(t_stat, 3),
        },
        {
            "Statistic": "df",
            "Value": df_val,
        },
        {
            "Statistic": "p-value",
            "Value": format_pvalue(p_value),
        },
        {
            "Statistic": "95% CI",
            "Value": format_ci(ci_low, ci_high, level=confidence_level),
        },
        {
            "Statistic": "Cohen's dz",
            "Value": format_number(dz, 3),
        },
    ])
    result.add_table(ResultTable(
        title="Paired Samples t-Test",
        dataframe=test_df,
    ))

    return result


def run_one_sample_ttest(
    data: pd.DataFrame,
    variable: str,
    test_value: float = 0.0,
    confidence_level: float = 0.95,
) -> AnalysisResult:
    """Run one-sample t-test.

    Args:
        data: DataFrame containing the variable.
        variable: Name of the variable to test.
        test_value: Hypothesized population mean (H0: mu = test_value).
        confidence_level: Confidence level for CI.

    Returns:
        AnalysisResult with one-sample t-test results.
    """
    result = AnalysisResult(
        id="one_sample_ttest",
        title=f"One-Sample t-Test: {variable}",
        spec={"variable": variable, "test_value": test_value},
    )

    col = data[variable].dropna()
    n = len(col)

    if n < 2:
        result.warnings.append("Insufficient valid observations (need at least 2).")
        return result

    arr = col.values.astype(float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    t_stat, p_value = stats.ttest_1samp(arr, test_value)
    df_val = n - 1
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    mean_diff = mean - test_value

    stats_df = pd.DataFrame([{
        "Variable": variable,
        "N": n,
        "Mean": format_number(mean, 3),
        "SD": format_number(sd, 3),
        "SE Mean": format_number(se, 3),
    }])
    result.add_table(ResultTable(title="One-Sample Statistics", dataframe=stats_df))

    test_df = pd.DataFrame([{
        "Test Value": test_value,
        "t": format_number(t_stat, 3),
        "df": df_val,
        "p-value": format_pvalue(p_value),
        "Mean Difference": format_number(mean_diff, 3),
        f"{int(confidence_level*100)}% CI": format_ci(ci_low, ci_high, level=confidence_level),
    }])
    result.add_table(ResultTable(title="One-Sample t-Test", dataframe=test_df))

    return result


def run_one_sample_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """_ModulePlugin 호환 wrapper — one_sample_t_test 플러그인용."""
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    target_vars: list[str] = variables.get("target", [])
    test_value: float = float(options.get("test_value", 0.0))
    confidence_level: float = float(spec.get("confidence_level", 0.95))

    result = AnalysisResult(id="one_sample_t_test", title="One-Sample T Test", spec=spec)
    if not target_vars:
        result.warnings.append("분석 변수가 지정되지 않았습니다.")
        return result

    combined = AnalysisResult(id="one_sample_t_test", title="One-Sample T Test", spec=spec)
    for var in target_vars:
        if var not in dataset.data.columns:
            combined.warnings.append(f"변수를 찾을 수 없습니다: {var}")
            continue
        sub = run_one_sample_ttest(dataset.data, var, test_value, confidence_level)
        combined.tables.extend(sub.tables)
        combined.warnings.extend(sub.warnings)
        combined.notes.extend(sub.notes)
    return combined
