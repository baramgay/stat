"""Frequency analysis for NuriStat."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run frequency analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "target" list of variable names.
            - options: dict with "include_missing", "show_cumulative",
              "show_value_labels", "sort_by".
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with frequency tables.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    target_vars: list[str] = variables.get("target", [])
    include_missing = options.get("include_missing", False)
    show_cumulative = options.get("show_cumulative", True)
    show_value_labels = options.get("show_value_labels", True)
    sort_by = options.get("sort_by", "value")
    weight_var: str | None = spec.get("weight_var")

    result = AnalysisResult(
        id="frequencies",
        title="Frequencies",
        spec=spec,
    )

    # 분석 변수 미지정 시 빈 테이블 대신 명확한 경고 반환 (타 분석과 일관)
    if not target_vars:
        result.warnings.append("분석 변수가 지정되지 않았습니다.")
        return result

    try:
        # Case Processing Summary (filter_$ and weight_var auto-applied)
        prepared = prepare_analysis_frame(
            dataset, variables=target_vars, missing_policy=missing_policy,
            weight_var=weight_var,
        )
        cps = get_case_processing_summary(
            prepared.n_total, prepared.n_valid, prepared.n_excluded,
            prepared.excluded_pct
        )
        result.add_table(cps)

        # Notify if filter or weight is active
        if prepared.n_filtered < prepared.n_total:
            result.warnings.append(
                f"케이스 필터 켜짐: 전체 {prepared.n_total}개 중 {prepared.n_filtered}개 선택됨"
            )
        if prepared.weight_var:
            result.warnings.append(f"가중치 변수 '{prepared.weight_var}' 적용됨")

        for var_name in target_vars:
            if var_name not in prepared.data.columns:
                result.warnings.append(f"Variable '{var_name}' not found in dataset.")
                continue

            series = prepared.data[var_name]
            total_n = len(series)
            valid_series = series.dropna()
            valid_n = len(valid_series)
            missing_n = total_n - valid_n

            # Weighted frequency counts (if weight variable present)
            if prepared.weight_var and prepared.weight_var in prepared.data.columns:
                weights = prepared.data.loc[valid_series.index, prepared.weight_var].fillna(0)
                weights = weights.clip(lower=0)
                wts = pd.Series(weights.values, index=valid_series.index)
                value_counts_raw = (
                    pd.DataFrame({"val": valid_series, "wt": wts})
                    .groupby("val")["wt"]
                    .sum()
                )
                total_n_eff = float(wts.sum())
                valid_n_eff = float(wts.sum())
            else:
                value_counts_raw = valid_series.value_counts(sort=False)
                total_n_eff = float(total_n)
                valid_n_eff = float(valid_n)

            # Sort
            if sort_by == "frequency":
                value_counts = value_counts_raw.sort_values(ascending=False)
            elif sort_by == "label":
                try:
                    value_counts = value_counts_raw.sort_index()
                except TypeError:
                    value_counts = value_counts_raw
            else:
                try:
                    value_counts = value_counts_raw.sort_index()
                except TypeError:
                    value_counts = value_counts_raw

            rows = []
            cumulative = 0.0

            # Get value labels if available
            var_meta = dataset.variables.get(var_name)
            value_labels = (var_meta.value_labels or {}) if var_meta else {}

            for value, freq in value_counts.items():
                pct = (freq / total_n_eff) * 100 if total_n_eff > 0 else 0
                valid_pct = (freq / valid_n_eff) * 100 if valid_n_eff > 0 else 0
                cumulative += valid_pct

                display_value = value_labels.get(str(value), str(value)) if show_value_labels and str(value) in value_labels else value

                row = {
                    "Value": display_value,
                    "Frequency": round(float(freq), 3) if prepared.weight_var else int(freq),
                    "Percent": round(pct, 1),
                    "Valid Percent": round(valid_pct, 1),
                }
                if show_cumulative:
                    row["Cumulative Percent"] = round(cumulative, 1)
                rows.append(row)

            # Add missing row if requested
            if include_missing and missing_n > 0:
                missing_pct = (missing_n / total_n) * 100 if total_n > 0 else 0
                row = {
                    "Value": "Missing",
                    "Frequency": missing_n,
                    "Percent": round(missing_pct, 1),
                    "Valid Percent": 0.0,
                }
                if show_cumulative:
                    row["Cumulative Percent"] = np.nan
                rows.append(row)

            freq_df = pd.DataFrame(rows)
            freq_table = ResultTable(
                title=f"Frequency Table: {var_name}",
                dataframe=freq_df,
            )
            result.add_table(freq_table)

    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")

    return result


class FrequenciesEngine:
    """AnalysisPlugin wrapper for frequency analysis."""

    id = "frequencies"
    name = "빈도분석"
    category = "Descriptive Statistics"
    description = "명목/순서 변수 빈도표: 빈도, 퍼센트, 누적 퍼센트"
    variable_requirements: list[dict] = [
        {"role": "variables", "measure_types": ["nominal", "ordinal", "binary", "text"], "min_count": 1, "required": True},
    ]
    implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        return run_analysis(dataset, spec)
