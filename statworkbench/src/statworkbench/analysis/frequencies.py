"""Frequency analysis for StatWorkbench."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


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

    result = AnalysisResult(
        id="frequencies",
        title="Frequencies",
        spec=spec,
    )

    try:
        # Case Processing Summary
        prepared = prepare_analysis_frame(
            dataset, variables=target_vars, missing_policy=missing_policy
        )
        cps = get_case_processing_summary(
            prepared.n_total, prepared.n_valid, prepared.n_excluded,
            prepared.excluded_pct
        )
        result.add_table(cps)

        for var_name in target_vars:
            if var_name not in dataset.data.columns:
                result.warnings.append(f"Variable '{var_name}' not found in dataset.")
                continue

            series = dataset.data[var_name]
            total_n = len(series)
            valid_series = series.dropna()
            valid_n = len(valid_series)
            missing_n = total_n - valid_n

            # Value counts
            value_counts = valid_series.value_counts(sort=False)

            # Sort
            if sort_by == "frequency":
                value_counts = valid_series.value_counts(sort=True)
            elif sort_by == "label":
                value_counts = valid_series.value_counts(sort=False).sort_index()
            else:
                try:
                    value_counts = valid_series.value_counts(sort=False).sort_index()
                except TypeError:
                    value_counts = valid_series.value_counts(sort=False)

            rows = []
            cumulative = 0

            # Get value labels if available
            var_meta = dataset.variables.get(var_name)
            value_labels = (var_meta.value_labels or {}) if var_meta else {}

            for value, freq in value_counts.items():
                pct = (freq / total_n) * 100 if total_n > 0 else 0
                valid_pct = (freq / valid_n) * 100 if valid_n > 0 else 0
                cumulative += valid_pct

                display_value = value_labels.get(str(value), str(value)) if show_value_labels and str(value) in value_labels else value

                row = {
                    "Value": display_value,
                    "Frequency": freq,
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
