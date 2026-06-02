"""Reliability Analysis — Cronbach's Alpha and item statistics.

SPSS Scale > Reliability Analysis 대응 모듈.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from statworkbench.analysis.assumptions import get_cps_table_kr
from statworkbench.analysis.formatting import format_number
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset


def _cronbach_alpha(data: pd.DataFrame) -> float:
    """Cronbach's alpha 계산.

    Formula: α = (k/(k-1)) * (1 - Σσ²ᵢ / σ²_total)
    """
    k = data.shape[1]
    if k < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def _alpha_if_deleted(data: pd.DataFrame) -> pd.Series:
    """각 항목 제거 시 Cronbach's alpha."""
    result = {}
    cols = data.columns.tolist()
    for col in cols:
        remaining = data.drop(columns=[col])
        result[col] = _cronbach_alpha(remaining)
    return pd.Series(result)


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """신뢰도 분석(Cronbach's Alpha)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.target: 항목 변수 목록
            options.model: 'alpha' (기본)
            options.listwise: True=목록별 결측 제거

    Returns:
        AnalysisResult
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    target_vars: list[str] = variables.get("target", [])
    listwise: bool = options.get("listwise", True)

    result = AnalysisResult(id="reliability", title="Reliability Analysis")

    if len(target_vars) < 2:
        result.warnings.append("신뢰도 분석에는 최소 2개 변수가 필요합니다.")
        return result

    # 데이터 준비
    missing_cols = [v for v in target_vars if v not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    data = dataset.data[target_vars].copy()
    try:
        data = data.apply(pd.to_numeric, errors="coerce")
    except Exception:
        pass

    n_before = len(data)
    if listwise:
        data = data.dropna()
    n_after = len(data)
    n_excluded = n_before - n_after

    if len(data) < 2:
        result.warnings.append("유효한 케이스가 부족합니다.")
        return result

    k = data.shape[1]
    try:
        alpha = _cronbach_alpha(data)
    except Exception as exc:
        result.add_warning(f"Cronbach's Alpha 계산 오류: {exc}")
        return result

    # ── Case Processing Summary ──────────────────────────────
    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    # ── Reliability Statistics ───────────────────────────────
    rel_df = pd.DataFrame({
        "Cronbach's Alpha": [format_number(alpha, 3)],
        "항목 수": [k],
    })
    result.tables.append(ResultTable(title="Reliability Statistics", dataframe=rel_df))

    # ── Item Statistics ──────────────────────────────────────
    item_stats = pd.DataFrame({
        "변수": target_vars,
        "평균": [format_number(data[v].mean(), 3) for v in target_vars],
        "표준편차": [format_number(data[v].std(ddof=1), 3) for v in target_vars],
        "N": [int(data[v].notna().sum()) for v in target_vars],
    })
    result.tables.append(ResultTable(title="Item Statistics", dataframe=item_stats))

    # ── Item-Total Statistics ────────────────────────────────
    total_score = data.sum(axis=1)
    alpha_if_del = _alpha_if_deleted(data)

    item_total_rows = []
    for v in target_vars:
        corrected_r = data[v].corr(total_score - data[v])
        sq_mult_corr = corrected_r ** 2
        item_total_rows.append({
            "변수": v,
            "교정 항목-전체 상관": format_number(corrected_r, 3),
            "제곱 다중 상관": format_number(sq_mult_corr, 3),
            "항목 제거 시 Alpha": format_number(alpha_if_del[v], 3),
        })

    item_total_df = pd.DataFrame(item_total_rows)
    result.tables.append(ResultTable(title="Item-Total Statistics", dataframe=item_total_df))

    # ── Scale Statistics ─────────────────────────────────────
    scale_mean = data.sum(axis=1).mean()
    scale_var = data.sum(axis=1).var(ddof=1)
    scale_sd = np.sqrt(scale_var)

    scale_df = pd.DataFrame({
        "평균": [format_number(scale_mean, 3)],
        "분산": [format_number(scale_var, 3)],
        "표준편차": [format_number(scale_sd, 3)],
        "항목 수": [k],
    })
    result.tables.append(ResultTable(title="Scale Statistics", dataframe=scale_df))

    # ── 해석 메모 ────────────────────────────────────────────
    if not np.isnan(alpha):
        if alpha >= 0.9:
            grade = "우수 (Excellent)"
        elif alpha >= 0.8:
            grade = "양호 (Good)"
        elif alpha >= 0.7:
            grade = "수용 가능 (Acceptable)"
        elif alpha >= 0.6:
            grade = "의심스러움 (Questionable)"
        else:
            grade = "불량 (Poor)"
        result.notes.append(
            f"Cronbach's α = {format_number(alpha, 3)} — {grade} | "
            f"항목 수 = {k}, 유효 케이스 = {n_after}"
        )

    return result
