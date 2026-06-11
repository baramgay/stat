"""Descriptive statistics analysis for NuriStat."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.formatting import format_ci, format_number, get_display_decimals
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset


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
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    scale_vars: list[str] = variables.get("scale", [])
    group_var: str | None = variables.get("group", None)

    result = AnalysisResult(
        id="descriptive_statistics",
        title="Descriptive Statistics",
        spec=spec,
    )

    # 분석 변수 미지정 시 빈 테이블 대신 명확한 경고 반환 (타 분석과 일관)
    if not scale_vars:
        result.warnings.append("분석 변수가 지정되지 않았습니다.")
        return result

    try:
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

        def _var_label(var: str) -> str:
            meta = dataset.variables.get(var) if dataset.variables else None
            return meta.label if (meta and meta.label) else var

        if group_var is not None and group_var in df.columns:
            # Grouped descriptives
            groups = df[group_var].dropna().unique()
            rows = []
            for grp in sorted(groups):
                grp_df = df[df[group_var] == grp]
                for var in scale_vars:
                    if var in grp_df.columns:
                        try:
                            desc = _compute_descriptives(
                                pd.to_numeric(grp_df[var], errors="coerce"),
                                confidence_level=confidence_level,
                            )
                        except Exception as exc:
                            logger.warning("기술통계 계산 실패 (변수=%s, 그룹=%s): %s", var, grp, exc)
                            desc = _compute_descriptives(pd.Series([], dtype=float))
                        d = get_display_decimals(dataset, var)
                        ci_str = format_ci(
                            desc["CI_Lower"], desc["CI_Upper"], decimals=d, level=confidence_level
                        )
                        rows.append({
                            "Group": grp,
                            "Variable": _var_label(var),
                            "N": desc["N"],
                            "Missing": desc["Missing"],
                            "Mean": format_number(desc["Mean"], d),
                            "SD": format_number(desc["SD"], d + 1),
                            "Median": format_number(desc["Median"], d),
                            "IQR": format_number(desc["IQR"], d),
                            "Min": format_number(desc["Min"], d),
                            "Max": format_number(desc["Max"], d),
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
                    try:
                        desc = _compute_descriptives(
                            pd.to_numeric(df[var], errors="coerce"),
                            confidence_level=confidence_level,
                        )
                    except Exception as exc:
                        logger.warning("기술통계 계산 실패 (변수=%s): %s", var, exc)
                        desc = _compute_descriptives(pd.Series([], dtype=float))
                    d = get_display_decimals(dataset, var)
                    ci_str = format_ci(
                        desc["CI_Lower"], desc["CI_Upper"], decimals=d, level=confidence_level
                    )
                    rows.append({
                        "Variable": _var_label(var),
                        "N": desc["N"],
                        "Missing": desc["Missing"],
                        "Mean": format_number(desc["Mean"], d),
                        "SD": format_number(desc["SD"], d + 1),
                        "Median": format_number(desc["Median"], d),
                        "IQR": format_number(desc["IQR"], d),
                        "Min": format_number(desc["Min"], d),
                        "Max": format_number(desc["Max"], d),
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

    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")

    return result


class DescriptiveEngine:
    """AnalysisPlugin wrapper for descriptive statistics."""

    id = "descriptives"
    name = "기술통계"
    category = "Descriptive Statistics"
    description = "척도 변수의 기술통계량 (평균, 표준편차, 사분위수, CI)"
    variable_requirements: list[dict] = [
        {"role": "variables", "measure_types": ["scale", "ordinal"], "min_count": 1, "required": True},
    ]
    implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        return run_analysis(dataset, spec)
