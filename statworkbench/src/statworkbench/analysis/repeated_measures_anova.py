"""반복측정 ANOVA(Repeated Measures ANOVA) 분석 모듈.

SPSS: Analyze > General Linear Model > Repeated Measures

지원 기능:
  - 일원 반복측정 ANOVA (1개 within-subjects 요인)
  - 구형성 검정(Mauchly's W) 및 보정
    - Greenhouse-Geisser (ε < 0.75 권장)
    - Huynh-Feldt
    - Lower-bound
  - 기술통계 (반복 수준별)
  - 쌍 비교(Pairwise Comparison): 본페로니 보정 t-검정

참고 문헌:
  Mauchly (1940), Greenhouse & Geisser (1959), Huynh & Feldt (1976)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.analysis.assumptions import get_cps_table_kr
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset


def _mauchly_test(data_matrix: np.ndarray) -> dict:
    """Mauchly 구형성 검정.

    Parameters
    ----------
    data_matrix : ndarray, shape (n_subjects, k_levels)
        반복 측정 데이터 행렬.

    Returns
    -------
    dict with keys: W, chi2, df, p, epsilon_gg, epsilon_hf, epsilon_lb
    """
    n, k = data_matrix.shape
    # 대비 행렬 (k-1 열)
    C = np.zeros((k, k - 1))
    for i in range(k - 1):
        C[i, i] = 1
        C[i + 1, i] = -1

    # 변환 행렬로 공분산 계산
    Y = data_matrix @ C
    S = np.cov(Y, rowvar=False)
    if k == 2:
        # 구형성은 k=2일 때 항상 충족
        return {
            "W": 1.0, "chi2": 0.0, "df": 0, "p": 1.0,
            "epsilon_gg": 1.0, "epsilon_hf": 1.0, "epsilon_lb": 1.0 / (k - 1),
        }

    # Mauchly W
    det_S = np.linalg.det(S)
    tr_S = np.trace(S)
    denom = (tr_S / (k - 1)) ** (k - 1)
    if denom < 1e-15 or det_S <= 0:
        # 공분산 행렬이 특이하거나 분산이 없는 경우 — 검정 불가 처리
        _df = int(k * (k - 1) / 2 - 1)
        return {
            "W": float("nan"), "chi2": float("nan"), "df": _df, "p": float("nan"),
            "epsilon_gg": 1.0 / (k - 1), "epsilon_hf": 1.0 / (k - 1), "epsilon_lb": 1.0 / (k - 1),
        }
    W = det_S / denom
    W = max(W, 1e-15)

    df_mau = int(k * (k - 1) / 2 - 1)
    f_coef = (2 * (k - 1) ** 2 + (k - 1) + 2) / (6 * (k - 1) * (n - 1))
    chi2 = -(n - 1) * (1 - f_coef) * np.log(W)
    p_mau = float(1 - stats.chi2.cdf(chi2, df_mau))

    # Greenhouse-Geisser ε
    tr_S2 = np.trace(S @ S)
    eps_gg = (tr_S ** 2) / ((k - 1) * (tr_S2)) if tr_S2 > 1e-15 else 1.0
    eps_gg = np.clip(eps_gg, 1.0 / (k - 1), 1.0)

    # Huynh-Feldt ε
    eps_hf = (n * (k - 1) * eps_gg - 2) / ((k - 1) * (n - 1 - (k - 1) * eps_gg))
    eps_hf = np.clip(eps_hf, eps_gg, 1.0)

    eps_lb = 1.0 / (k - 1)

    return {
        "W": float(W),
        "chi2": float(chi2),
        "df": df_mau,
        "p": p_mau,
        "epsilon_gg": float(eps_gg),
        "epsilon_hf": float(eps_hf),
        "epsilon_lb": float(eps_lb),
    }


def _rm_anova_one_factor(data_matrix: np.ndarray) -> dict:
    """일원 반복측정 ANOVA 핵심 계산.

    Parameters
    ----------
    data_matrix : ndarray, shape (n, k)
        행=피험자, 열=시점/수준.

    Returns
    -------
    dict: SS_w, SS_b (between subjects), SS_wf (within-factor), SS_err,
          df_wf, df_err, MS_wf, MS_err, F, p
    """
    n, k = data_matrix.shape
    grand_mean = data_matrix.mean()
    subj_means = data_matrix.mean(axis=1)
    cond_means = data_matrix.mean(axis=0)

    SS_total = float(((data_matrix - grand_mean) ** 2).sum())
    SS_bs = float(k * ((subj_means - grand_mean) ** 2).sum())   # between subjects
    SS_ws = SS_total - SS_bs                                      # within subjects
    SS_wf = float(n * ((cond_means - grand_mean) ** 2).sum())    # within-factor (treatment)
    SS_err = SS_ws - SS_wf

    df_bs = n - 1
    df_wf = k - 1
    df_err = (n - 1) * (k - 1)

    MS_wf = SS_wf / df_wf
    MS_err = SS_err / df_err if df_err > 0 else np.nan

    if MS_err is not None and MS_err > 1e-15:
        F_val = MS_wf / MS_err
        p_val = float(1 - stats.f.cdf(F_val, df_wf, df_err))
    elif MS_err is not None and MS_err <= 1e-15 and MS_wf > 1e-15:
        # 오차 분산 0 → 완전한 효과 (p → 0)
        F_val = np.inf
        p_val = 0.0
    else:
        F_val = np.nan
        p_val = np.nan

    return {
        "SS_bs": SS_bs, "SS_ws": SS_ws, "SS_wf": SS_wf, "SS_err": SS_err,
        "SS_total": SS_total,
        "df_bs": df_bs, "df_wf": df_wf, "df_err": df_err,
        "MS_wf": MS_wf, "MS_err": MS_err,
        "F": F_val, "p": p_val,
        "n": n, "k": k,
    }


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """반복측정 ANOVA를 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.measures: 반복 측정 변수 목록 (시점/수준 순서대로, 최소 2개)
            variables.subject: (선택) 피험자 식별 변수 — 없으면 행 순서 사용
            options.within_name: within-subjects 요인 레이블 (기본 "시점")
            options.pairwise: True=쌍 비교 수행 (기본 True)
            options.alpha: 유의 수준 (기본 0.05)
            missing_policy: 결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. Descriptive Statistics
            3. Mauchly's Test of Sphericity
            4. Tests of Within-Subjects Effects (원래·GG보정·HF보정·하한)
            5. Pairwise Comparisons (본페로니)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})

    measures: list[str] = variables.get("measures", [])
    subject_var: str | None = variables.get("subject", None)
    within_name: str = options.get("within_name", "시점")
    do_pairwise: bool = options.get("pairwise", True)
    alpha: float = options.get("alpha", 0.05)

    result = AnalysisResult(id="repeated_measures_anova", title="Repeated Measures ANOVA")

    # ── 입력 검증 ─────────────────────────────────────────────────
    if len(measures) < 2:
        result.warnings.append("반복 측정 변수(measures)가 최소 2개 필요합니다.")
        return result

    missing_cols = [c for c in measures if c not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    k = len(measures)

    # ── 데이터 준비 ───────────────────────────────────────────────
    cols = measures if subject_var is None else [subject_var] + measures
    data = dataset.data[cols].copy()
    data[measures] = data[measures].apply(pd.to_numeric, errors="coerce")

    n_before = len(data)
    data = data.dropna(subset=measures)
    n_after = len(data)
    n_excluded = n_before - n_after

    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    if n_after < 2:
        result.warnings.append("유효한 케이스가 부족합니다 (최소 2건 필요).")
        return result

    mat = data[measures].values.astype(float)
    n = mat.shape[0]

    # ── Table 2: 기술통계 ─────────────────────────────────────────
    desc_rows = []
    for i, var in enumerate(measures):
        col_data = mat[:, i]
        desc_rows.append({
            within_name: var,
            "N": n,
            "평균": format_number(col_data.mean(), 4),
            "표준편차": format_number(col_data.std(ddof=1), 4),
            "표준오차": format_number(col_data.std(ddof=1) / np.sqrt(n), 4),
            "최솟값": format_number(col_data.min(), 4),
            "최댓값": format_number(col_data.max(), 4),
        })
    result.tables.append(ResultTable(
        title="Descriptive Statistics",
        dataframe=pd.DataFrame(desc_rows),
    ))

    # ── Table 3: Mauchly 구형성 검정 ─────────────────────────────
    mau = _mauchly_test(mat)
    mau_df = pd.DataFrame({
        "Mauchly W": [format_number(mau["W"], 4)],
        "근사 χ²": [format_number(mau["chi2"], 4)],
        "df": [mau["df"]],
        "p": [format_pvalue(mau["p"])],
        "Greenhouse-Geisser ε": [format_number(mau["epsilon_gg"], 4)],
        "Huynh-Feldt ε": [format_number(mau["epsilon_hf"], 4)],
        "하한 ε": [format_number(mau["epsilon_lb"], 4)],
    })
    result.tables.append(ResultTable(title="Mauchly's Test of Sphericity", dataframe=mau_df))

    if not np.isnan(mau["p"]) and mau["p"] < alpha:
        result.warnings.append(
            f"구형성 가정이 위반되었습니다 (p = {format_pvalue(mau['p'])}). "
            f"Greenhouse-Geisser 또는 Huynh-Feldt 보정 결과를 사용하세요."
        )

    # ── Table 4: Within-Subjects Effects ─────────────────────────
    r = _rm_anova_one_factor(mat)
    F = r["F"]
    p_raw = r["p"]
    df_wf = r["df_wf"]
    df_err = r["df_err"]
    MS_wf = r["MS_wf"]
    MS_err = r["MS_err"]
    eps_gg = mau["epsilon_gg"]
    eps_hf = mau["epsilon_hf"]
    eps_lb = mau["epsilon_lb"]

    def _corrected_p(eps: float) -> str:
        df1_c = df_wf * eps
        df2_c = df_err * eps
        if np.isnan(F) or df2_c <= 0:
            return "-"
        p_c = float(1 - stats.f.cdf(F, df1_c, df2_c))
        return format_pvalue(p_c)

    within_rows = [
        {
            "소스": within_name,
            "보정": "구형성 가정",
            "SS": format_number(r["SS_wf"], 4),
            "df": format_number(df_wf, 3),
            "MS": format_number(MS_wf, 4),
            "F": format_number(F, 4) if np.isfinite(F) else ("∞" if F == np.inf else "-"),
            "p-value": format_pvalue(p_raw) if not np.isnan(p_raw) else "-",
        },
        {
            "소스": within_name,
            "보정": "Greenhouse-Geisser",
            "SS": format_number(r["SS_wf"], 4),
            "df": format_number(df_wf * eps_gg, 3),
            "MS": format_number(MS_wf, 4),
            "F": format_number(F, 4) if np.isfinite(F) else ("∞" if F == np.inf else "-"),
            "p-value": _corrected_p(eps_gg),
        },
        {
            "소스": within_name,
            "보정": "Huynh-Feldt",
            "SS": format_number(r["SS_wf"], 4),
            "df": format_number(df_wf * eps_hf, 3),
            "MS": format_number(MS_wf, 4),
            "F": format_number(F, 4) if np.isfinite(F) else ("∞" if F == np.inf else "-"),
            "p-value": _corrected_p(eps_hf),
        },
        {
            "소스": within_name,
            "보정": "하한 (Lower-bound)",
            "SS": format_number(r["SS_wf"], 4),
            "df": format_number(df_wf * eps_lb, 3),
            "MS": format_number(MS_wf, 4),
            "F": format_number(F, 4) if np.isfinite(F) else ("∞" if F == np.inf else "-"),
            "p-value": _corrected_p(eps_lb),
        },
        {
            "소스": "오차",
            "보정": "구형성 가정",
            "SS": format_number(r["SS_err"], 4),
            "df": format_number(df_err, 3),
            "MS": format_number(MS_err, 4),
            "F": "",
            "p-value": "",
        },
        {
            "소스": "오차",
            "보정": "Greenhouse-Geisser",
            "SS": format_number(r["SS_err"], 4),
            "df": format_number(df_err * eps_gg, 3),
            "MS": format_number(MS_err, 4),
            "F": "",
            "p-value": "",
        },
    ]

    result.tables.append(ResultTable(
        title="Tests of Within-Subjects Effects",
        dataframe=pd.DataFrame(within_rows),
    ))

    # ── Table 5: 쌍 비교 (본페로니) ──────────────────────────────
    if do_pairwise:
        pairs = list(combinations(range(k), 2))
        n_pairs = len(pairs)
        alpha_bonf = alpha / n_pairs if n_pairs > 0 else alpha

        pair_rows = []
        for i, j in pairs:
            diff = mat[:, i] - mat[:, j]
            t_stat, p_paired = stats.ttest_rel(mat[:, i], mat[:, j])
            p_adj = min(p_paired * n_pairs, 1.0)  # 본페로니 보정
            se = diff.std(ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha_bonf / 2, df=n - 1)
            ci_lo = diff.mean() - t_crit * se
            ci_hi = diff.mean() + t_crit * se

            pair_rows.append({
                f"{within_name} (I)": measures[i],
                f"{within_name} (J)": measures[j],
                "평균차 (I-J)": format_number(diff.mean(), 4),
                "표준오차": format_number(se, 4),
                "p-value": format_pvalue(p_paired),
                "p-adj (본페로니)": format_pvalue(p_adj),
                "CI 하한": format_number(ci_lo, 4),
                "CI 상한": format_number(ci_hi, 4),
            })

        result.tables.append(ResultTable(
            title="Pairwise Comparisons (Bonferroni)",
            dataframe=pd.DataFrame(pair_rows),
        ))

    # ── 해석 메모 ─────────────────────────────────────────────────
    if not np.isnan(F):
        f_str = format_number(F, 3) if np.isfinite(F) else "∞"
        p_str = format_pvalue(p_raw) if not np.isnan(p_raw) else "-"
        result.notes.append(
            f"[{within_name}] F({df_wf}, {df_err}) = {f_str}, "
            f"p = {p_str} (구형성 가정 기준)"
        )
        result.notes.append(
            f"Greenhouse-Geisser ε = {format_number(eps_gg, 3)}, "
            f"Huynh-Feldt ε = {format_number(eps_hf, 3)}"
        )

    return result
