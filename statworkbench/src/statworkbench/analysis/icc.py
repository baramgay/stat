"""급내상관계수(ICC, Intraclass Correlation Coefficient) 분석 모듈.

SPSS: Analyze > Scale > Reliability Analysis > (Statistics) Intraclass Correlation Coefficient

지원 모델:
  ICC(1,1) — One-Way Random,   Single Measure
  ICC(2,1) — Two-Way Random,   Absolute Agreement, Single Measure
  ICC(3,1) — Two-Way Mixed,    Consistency,        Single Measure  (기본값)

참고 문헌:
  Shrout & Fleiss (1979), Koo & Mae (2016)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.analysis.assumptions import get_cps_table_kr
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset

# ---------------------------------------------------------------------------
# 내부 계산 함수
# ---------------------------------------------------------------------------

def _compute_icc(data_matrix: pd.DataFrame, model: str = "twoway_mixed") -> dict:
    """ICC 핵심 계산.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        행=피험자, 열=평가자. 수치형 데이터만 허용.
    model : str
        "oneway_random"  → ICC(1,1)
        "twoway_random"  → ICC(2,1)
        "twoway_mixed"   → ICC(3,1)  (기본값)

    Returns
    -------
    dict
        icc, f, df1, df2, p, ci_lower, ci_upper, ms_b, ms_w, ms_j, ms_e,
        ss_b, ss_w, ss_j, ss_e, ss_t 포함.
    """
    n = len(data_matrix)      # 피험자 수
    k = data_matrix.shape[1]  # 평가자 수

    if n < 2:
        raise ValueError("피험자 수는 최소 2명 이상이어야 합니다.")
    if k < 2:
        raise ValueError("평가자 수는 최소 2명 이상이어야 합니다.")

    # ── ANOVA 분해 ────────────────────────────────────────────────
    grand_mean = data_matrix.values.mean()
    row_means = data_matrix.mean(axis=1)
    col_means = data_matrix.mean(axis=0)

    # SS 계산
    ss_t = float(((data_matrix.values - grand_mean) ** 2).sum())
    ss_b = float(k * ((row_means - grand_mean) ** 2).sum())          # Between subjects
    ss_w = ss_t - ss_b                                                # Within subjects
    ss_j = float(n * ((col_means - grand_mean) ** 2).sum())          # Between raters (columns)
    ss_e = ss_w - ss_j                                                # Error (residual)

    # MS 계산
    ms_b = ss_b / (n - 1)
    ms_w = ss_w / (n * (k - 1))
    ms_j = ss_j / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))

    # ── 모델별 ICC 계산 ──────────────────────────────────────────
    if model == "oneway_random":
        # ICC(1,1): One-Way Random
        # WMS = 피험자 내 MS (within)
        wms = ms_w
        icc = (ms_b - wms) / (ms_b + (k - 1) * wms)
        f_val = ms_b / wms if wms > 0 else np.nan
        df1 = n - 1
        df2 = n * (k - 1)

    elif model == "twoway_random":
        # ICC(2,1): Two-Way Random, Absolute Agreement
        icc = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e + k * (ms_j - ms_e) / n)
        f_val = ms_b / ms_e if ms_e > 0 else np.nan
        df1 = n - 1
        df2 = (n - 1) * (k - 1)

    elif model == "twoway_mixed":
        # ICC(3,1): Two-Way Mixed, Consistency
        icc = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e)
        f_val = ms_b / ms_e if ms_e > 0 else np.nan
        df1 = n - 1
        df2 = (n - 1) * (k - 1)

    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model!r}. "
                         "oneway_random / twoway_random / twoway_mixed 중 선택하세요.")

    # p값 계산
    if np.isnan(f_val):
        p_val = np.nan
    else:
        p_val = float(1.0 - stats.f.cdf(f_val, df1, df2))

    # ── 95% CI 계산 (F-분포 기반) ────────────────────────────────
    if np.isnan(f_val) or f_val <= 0:
        ci_lower = ci_upper = np.nan
    else:
        fl = f_val / stats.f.ppf(0.975, df1, df2)
        fu = f_val * stats.f.ppf(0.975, df2, df1)
        ci_lower = float((fl - 1.0) / (fl + k - 1.0))
        ci_upper = float((fu - 1.0) / (fu + k - 1.0))

    return {
        "icc": float(icc),
        "f": float(f_val),
        "df1": int(df1),
        "df2": int(df2),
        "p": float(p_val),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        # ANOVA 성분 (테이블 출력용)
        "ms_b": float(ms_b),
        "ms_w": float(ms_w),
        "ms_j": float(ms_j),
        "ms_e": float(ms_e),
        "ss_b": float(ss_b),
        "ss_w": float(ss_w),
        "ss_j": float(ss_j),
        "ss_e": float(ss_e),
        "ss_t": float(ss_t),
        "n": int(n),
        "k": int(k),
    }


# ---------------------------------------------------------------------------
# 해석 함수 (Koo & Mae, 2016)
# ---------------------------------------------------------------------------

def _interpret_icc(icc: float) -> str:
    """ICC값을 Koo & Mae(2016) 기준으로 해석합니다."""
    if icc < 0.5:
        return "불량 (Poor)"
    elif icc < 0.75:
        return "보통 (Moderate)"
    elif icc < 0.90:
        return "양호 (Good)"
    else:
        return "우수 (Excellent)"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """급내상관계수(ICC) 분석을 수행합니다.

    SPSS: Analyze > Scale > Reliability Analysis > (Statistics) ICC

    Parameters
    ----------
    dataset : Dataset
        분석 대상 데이터셋.
    spec : dict
        분석 명세.
        - variables.target : 평가자 변수 목록 (최소 2개)
        - options.model    : "oneway_random" | "twoway_random" | "twoway_mixed" (기본)
        - options.unit     : "single" | "average" (기본 single; 현재 single만 구현)

    Returns
    -------
    AnalysisResult
        4개 테이블 포함:
        1. Case Processing Summary
        2. ICC
        3. ANOVA
        4. Interpretation
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    target_vars: list[str] = variables.get("target", [])
    model: str = options.get("model", "twoway_mixed")

    result = AnalysisResult(id="icc", title="Intraclass Correlation Coefficient")

    # ── 입력 검증 ─────────────────────────────────────────────────
    if len(target_vars) < 2:
        result.warnings.append("ICC 분석에는 최소 2개 변수(평가자)가 필요합니다.")
        return result

    missing_cols = [v for v in target_vars if v not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────
    data = dataset.data[target_vars].copy()
    data = data.apply(pd.to_numeric, errors="coerce")

    n_before = len(data)
    data = data.dropna()
    n_after = len(data)
    n_excluded = n_before - n_after

    if n_after < 2:
        result.warnings.append("유효한 케이스가 부족합니다 (최소 2건 필요).")
        return result

    # ── 계산 ─────────────────────────────────────────────────────
    try:
        r = _compute_icc(data, model=model)
    except ValueError as exc:
        result.warnings.append(str(exc))
        return result

    n = r["n"]
    k = r["k"]
    icc = r["icc"]

    # ── 모델 레이블 ───────────────────────────────────────────────
    model_labels = {
        "oneway_random": "ICC(1,1) — One-Way Random, Single",
        "twoway_random": "ICC(2,1) — Two-Way Random, Absolute Agreement, Single",
        "twoway_mixed":  "ICC(3,1) — Two-Way Mixed, Consistency, Single",
    }
    model_label = model_labels.get(model, model)

    # ── Table 1: Case Processing Summary ─────────────────────────
    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    # ── Table 2: ICC ─────────────────────────────────────────────
    icc_df = pd.DataFrame({
        "모델": [model_label],
        "ICC": [format_number(icc, 3)],
        "95% CI 하한": [format_number(r["ci_lower"], 3)],
        "95% CI 상한": [format_number(r["ci_upper"], 3)],
        "F": [format_number(r["f"], 3)],
        "df1": [r["df1"]],
        "df2": [r["df2"]],
        "p": [format_pvalue(r["p"])],
    })
    result.tables.append(ResultTable(title="ICC", dataframe=icc_df))

    # ── Table 3: ANOVA ───────────────────────────────────────────
    if model == "oneway_random":
        anova_rows = [
            {
                "분산원": "피험자 간 (Between Subjects)",
                "SS": format_number(r["ss_b"], 4),
                "df": n - 1,
                "MS": format_number(r["ms_b"], 4),
                "F": format_number(r["f"], 4),
            },
            {
                "분산원": "피험자 내 (Within Subjects)",
                "SS": format_number(r["ss_w"], 4),
                "df": n * (k - 1),
                "MS": format_number(r["ms_w"], 4),
                "F": "",
            },
        ]
    else:
        anova_rows = [
            {
                "분산원": "피험자 간 (Between Subjects)",
                "SS": format_number(r["ss_b"], 4),
                "df": n - 1,
                "MS": format_number(r["ms_b"], 4),
                "F": format_number(r["f"], 4),
            },
            {
                "분산원": "피험자 내 (Within Subjects)",
                "SS": format_number(r["ss_w"], 4),
                "df": n * (k - 1),
                "MS": format_number(r["ms_w"], 4),
                "F": "",
            },
            {
                "분산원": "평가자 간 (Between Raters)",
                "SS": format_number(r["ss_j"], 4),
                "df": k - 1,
                "MS": format_number(r["ms_j"], 4),
                "F": "",
            },
            {
                "분산원": "잔차 (Error)",
                "SS": format_number(r["ss_e"], 4),
                "df": (n - 1) * (k - 1),
                "MS": format_number(r["ms_e"], 4),
                "F": "",
            },
        ]

    anova_df = pd.DataFrame(anova_rows)
    result.tables.append(ResultTable(title="ANOVA", dataframe=anova_df))

    # ── Table 4: Interpretation ───────────────────────────────────
    grade = _interpret_icc(icc)
    interp_df = pd.DataFrame({
        "ICC": [format_number(icc, 3)],
        "95% CI": [f"[{format_number(r['ci_lower'], 3)}, {format_number(r['ci_upper'], 3)}]"],
        "해석 (Koo & Mae, 2016)": [grade],
        "기준": ["< 0.50 불량 | 0.50–0.74 보통 | 0.75–0.89 양호 | ≥ 0.90 우수"],
    })
    result.tables.append(ResultTable(title="Interpretation", dataframe=interp_df))

    # ── 메모 ─────────────────────────────────────────────────────
    result.notes.append(
        f"{model_label} | ICC = {format_number(icc, 3)} — {grade} | "
        f"n = {n}, k = {k}"
    )

    return result
