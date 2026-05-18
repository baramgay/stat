"""One-way ANOVA analysis for StatWorkbench."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_pvalue, format_number, format_ci
from statworkbench.analysis.assumptions import (
    prepare_analysis_frame, get_case_processing_summary, levene_test
)


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run one-way ANOVA analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "dependent" and "factor".
            - options: dict with "post_hoc", "welch", "effect_size".
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with ANOVA tables.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    dep_var: str = variables.get("dependent", "")
    factor_var: str = variables.get("factor", "")
    post_hoc = options.get("post_hoc", ["tukey"])
    use_welch = options.get("welch", False)
    show_effect_size = options.get("effect_size", True)

    result = AnalysisResult(
        id="one_way_anova",
        title="One-Way ANOVA",
        spec=spec,
    )

    all_vars = [dep_var, factor_var]

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

    # Group-level descriptives
    groups_list = sorted(df[factor_var].dropna().unique())
    group_rows = []
    group_data = []
    for grp in groups_list:
        grp_df = df[df[factor_var] == grp]
        arr = grp_df[dep_var].dropna().values
        n = len(arr)
        mean = float(np.mean(arr)) if n > 0 else np.nan
        sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 0 else np.nan
        group_rows.append({
            "Group": grp,
            "N": n,
            "Mean": format_number(mean, 3),
            "SD": format_number(sd, 3),
            "SE": format_number(se, 3),
        })
        group_data.append(arr)

    group_stats = pd.DataFrame(group_rows)
    result.add_table(ResultTable(
        title="Descriptives",
        dataframe=group_stats,
    ))

    # Levene's test
    if len(group_data) >= 2 and all(len(g) > 0 for g in group_data):
        levene_stat, levene_p = levene_test(*group_data)
        levene_df = pd.DataFrame([
            {
                "Test": "Levene's Test",
                "F": format_number(levene_stat, 3),
                "p-value": format_pvalue(levene_p),
            }
        ])
        result.add_table(ResultTable(
            title="Test of Homogeneity of Variances",
            dataframe=levene_df,
        ))

    # ANOVA
    clean_df = df[[dep_var, factor_var]].dropna()

    if len(groups_list) < 2:
        result.warnings.append("ANOVA requires at least 2 groups.")
        return result

    try:
        formula = f"{dep_var} ~ C({factor_var})"
        model = ols(formula, data=clean_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        anova_rows = []
        for idx, row in anova_table.iterrows():
            anova_rows.append({
                "Source": idx,
                "SS": format_number(row["sum_sq"], 3),
                "df": int(row["df"]) if not np.isnan(row["df"]) else "",
                "MS": format_number(row["sum_sq"] / row["df"], 3) if row["df"] > 0 else "",
                "F": format_number(row["F"], 3) if not np.isnan(row["F"]) else "",
                "p-value": format_pvalue(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else "",
            })

        # Effect sizes
        ss_between = anova_table.loc[f"C({factor_var})", "sum_sq"]
        ss_total = anova_table["sum_sq"].sum()
        df_between = anova_table.loc[f"C({factor_var})", "df"]
        df_within = anova_table.loc["Residual", "df"]
        ms_within = anova_table.loc["Residual", "sum_sq"] / df_within
        p_val = anova_table.loc[f"C({factor_var})", "PR(>F)"]

        if show_effect_size and ss_total > 0:
            eta_sq = ss_between / ss_total
            omega_sq = max(
                0,
                (ss_between - df_between * ms_within)
                / (ss_total + ms_within),
            )
            anova_rows.append({
                "Source": "Eta-squared",
                "SS": "",
                "df": "",
                "MS": "",
                "F": format_number(eta_sq, 3),
                "p-value": "",
            })
            anova_rows.append({
                "Source": "Omega-squared",
                "SS": "",
                "df": "",
                "MS": "",
                "F": format_number(omega_sq, 3),
                "p-value": "",
            })

        anova_df = pd.DataFrame(anova_rows)
        result.add_table(ResultTable(
            title="ANOVA",
            dataframe=anova_df,
        ))

        if use_welch:
            _run_welch_anova(clean_df, dep_var, factor_var, result)

        # Post-hoc tests
        alpha = 1 - confidence_level
        if p_val < alpha and post_hoc:
            if "tukey" in post_hoc:
                _run_tukey_hsd(clean_df, dep_var, factor_var, confidence_level, result)
            if "bonferroni" in post_hoc:
                _run_bonferroni(clean_df, dep_var, factor_var, confidence_level, result)
            if "scheffe" in post_hoc:
                _run_scheffe(clean_df, dep_var, factor_var, confidence_level, result,
                             anova_table, ms_within, df_within)

    except Exception as e:
        result.warnings.append(f"ANOVA could not be computed: {e}")

    return result


def _run_welch_anova(
    df: pd.DataFrame, dep_var: str, factor_var: str, result: AnalysisResult
) -> None:
    """Run Welch's ANOVA for unequal variances."""
    groups = [g[dep_var].dropna().values for _, g in df.groupby(factor_var)]
    k = len(groups)
    group_means = [np.mean(g) for g in groups]
    group_vars = [np.var(g, ddof=1) for g in groups]
    group_ns = [len(g) for g in groups]

    grand_mean = np.sum([n * m for n, m in zip(group_ns, group_means)]) / sum(group_ns)

    numerator = np.sum([n * (m - grand_mean)**2 for n, m in zip(group_ns, group_means)]) / (k - 1)
    denominator = np.sum([(1 - n / sum(group_ns)) * v for n, v in zip(group_ns, group_vars)]) / (k - 1)

    if denominator > 0:
        F_welch = numerator / denominator
    else:
        F_welch = np.nan

    df1 = k - 1
    L = np.sum([(1 - n / sum(group_ns)) * v for n, v in zip(group_ns, group_vars)])
    denom_df = np.sum(
        [((1 - n / sum(group_ns)) * v)**2 / (n - 1) for n, v in zip(group_ns, group_vars) if n > 1]
    )
    if denom_df > 0:
        df2 = L**2 / denom_df
    else:
        df2 = np.nan

    if not np.isnan(F_welch) and not np.isnan(df2):
        p_welch = 1 - stats.f.cdf(F_welch, df1, df2)
    else:
        p_welch = np.nan

    welch_df = pd.DataFrame([
        {
            "Source": "Welch ANOVA",
            "F": format_number(F_welch, 3),
            "df1": df1,
            "df2": format_number(df2, 1),
            "p-value": format_pvalue(p_welch),
        }
    ])
    result.add_table(ResultTable(
        title="Welch ANOVA (for unequal variances)",
        dataframe=welch_df,
    ))


def _run_tukey_hsd(
    df: pd.DataFrame,
    dep_var: str,
    factor_var: str,
    confidence_level: float,
    result: AnalysisResult,
) -> None:
    """Run Tukey HSD post-hoc test."""
    try:
        tukey = pairwise_tukeyhsd(
            endog=df[dep_var].values,
            groups=df[factor_var].values,
            alpha=1 - confidence_level,
        )
        data = tukey._results_table.data
        tukey_df = pd.DataFrame(data[1:], columns=data[0])
        result.add_table(ResultTable(
            title="Post-Hoc: Tukey HSD",
            dataframe=tukey_df,
        ))
    except Exception as e:
        result.warnings.append(f"Tukey HSD could not be computed: {e}")


def _run_scheffe(
    df: pd.DataFrame,
    dep_var: str,
    factor_var: str,
    confidence_level: float,
    result: AnalysisResult,
    anova_table: "pd.DataFrame",
    ms_within: float,
    df_within: float,
) -> None:
    """Run Scheffe post-hoc test (conservative, controls familywise error rate)."""
    try:
        groups_list = sorted(df[factor_var].unique())
        k = len(groups_list)
        alpha = 1 - confidence_level
        rows = []

        for i in range(k):
            for j in range(i + 1, k):
                g1 = df[df[factor_var] == groups_list[i]][dep_var].dropna().values
                g2 = df[df[factor_var] == groups_list[j]][dep_var].dropna().values
                n1, n2 = len(g1), len(g2)
                if n1 == 0 or n2 == 0:
                    continue
                mean_diff = float(np.mean(g1) - np.mean(g2))
                se_diff = float(np.sqrt(ms_within * (1 / n1 + 1 / n2)))
                # Scheffe F-statistic
                f_scheffe = (mean_diff ** 2) / (ms_within * (1 / n1 + 1 / n2))
                # Critical value: (k-1) * F_crit(alpha, k-1, df_within)
                f_crit = (k - 1) * stats.f.ppf(1 - alpha, dfn=k - 1, dfd=df_within)
                p_scheffe = 1 - stats.f.cdf(f_scheffe / (k - 1), dfn=k - 1, dfd=df_within)
                significant = "Yes" if f_scheffe > f_crit else "No"
                # 95% CI for mean difference
                margin = np.sqrt(f_crit) * se_diff
                ci_lo = mean_diff - margin
                ci_hi = mean_diff + margin
                rows.append({
                    "Group1": groups_list[i],
                    "Group2": groups_list[j],
                    "Mean Difference": format_number(mean_diff, 3),
                    "SE": format_number(se_diff, 3),
                    "F (Scheffe)": format_number(f_scheffe, 3),
                    "p-value": format_pvalue(p_scheffe),
                    f"{int(confidence_level*100)}% CI": format_ci(ci_lo, ci_hi),
                    "Significant": significant,
                })

        scheffe_df = pd.DataFrame(rows)
        result.add_table(ResultTable(
            title="Post-Hoc: Scheffe",
            dataframe=scheffe_df,
            footnotes=[
                "Scheffe 검정은 보수적인 다중비교 방법으로 1종 오류를 엄격하게 통제합니다.",
                f"유의 수준: alpha = {1 - confidence_level:.2f}",
            ],
        ))
    except Exception as e:
        result.warnings.append(f"Scheffe post-hoc 계산 실패: {e}")


def _run_bonferroni(
    df: pd.DataFrame,
    dep_var: str,
    factor_var: str,
    confidence_level: float,
    result: AnalysisResult,
) -> None:
    """Run Bonferroni post-hoc test."""
    try:
        groups_list = sorted(df[factor_var].unique())
        rows = []
        alpha = 1 - confidence_level
        n_comparisons = 0
        for i in range(len(groups_list)):
            for j in range(i + 1, len(groups_list)):
                n_comparisons += 1

        for i in range(len(groups_list)):
            for j in range(i + 1, len(groups_list)):
                g1 = df[df[factor_var] == groups_list[i]][dep_var].dropna().values
                g2 = df[df[factor_var] == groups_list[j]][dep_var].dropna().values
                t_stat, p_val = stats.ttest_ind(g1, g2)
                adj_p = min(p_val * n_comparisons, 1.0)
                mean_diff = np.mean(g1) - np.mean(g2)
                rows.append({
                    "Group1": groups_list[i],
                    "Group2": groups_list[j],
                    "Mean Difference": format_number(mean_diff, 3),
                    "t": format_number(t_stat, 3),
                    "p (uncorrected)": format_pvalue(p_val),
                    "p (Bonferroni)": format_pvalue(adj_p),
                    "Significant": "Yes" if adj_p < alpha else "No",
                })

        bonferroni_df = pd.DataFrame(rows)
        result.add_table(ResultTable(
            title="Post-Hoc: Bonferroni",
            dataframe=bonferroni_df,
        ))
    except Exception as e:
        result.warnings.append(f"Bonferroni post-hoc could not be computed: {e}")
