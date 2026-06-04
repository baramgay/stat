"""Bland-Altman 일치도 분석 모듈.

임상·보건 연구에서 두 측정법의 일치도를 평가하는 Bland-Altman 분석.
SPSS에는 직접 메뉴가 없으나 MedCalc·SPSS 매크로로 구현됨.

참조:
  Bland JM, Altman DG (1986). Statistical methods for assessing agreement
  between two methods of clinical measurement. Lancet, 327(8476), 307-310.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.assumptions import get_cps_table_kr
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core.dataset import Dataset


def _compute_bland_altman(method1: np.ndarray, method2: np.ndarray) -> dict:
    """Bland-Altman 일치도 통계량을 계산합니다.

    Args:
        method1: 첫 번째 측정법 값 배열 (NaN 없음, 동일 길이).
        method2: 두 번째 측정법 값 배열 (NaN 없음, 동일 길이).

    Returns:
        dict:
            n                  - 유효 케이스 수
            mean_diff          - 평균 차이 (bias)
            sd_diff            - 차이의 표준편차 (ddof=1)
            loa_upper          - 상한 일치 한계 (mean_diff + 1.96 * sd)
            loa_lower          - 하한 일치 한계 (mean_diff - 1.96 * sd)
            ci_mean_low        - 평균 차이 95% CI 하한
            ci_mean_high       - 평균 차이 95% CI 상한
            ci_loa_upper_low   - 상한 LoA 95% CI 하한
            ci_loa_upper_high  - 상한 LoA 95% CI 상한
            ci_loa_lower_low   - 하한 LoA 95% CI 하한
            ci_loa_lower_high  - 하한 LoA 95% CI 상한
            proportional_bias_r - diff vs mean Pearson r
            proportional_bias_p - 대응 p값

    Raises:
        ValueError: 두 배열의 길이가 다를 때.
        ValueError: 유효 케이스 수 < 2일 때.
    """
    a = np.asarray(method1, dtype=float)
    b = np.asarray(method2, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"두 배열의 길이가 다릅니다: {len(a)} != {len(b)}"
        )

    n = len(a)
    if n < 2:
        raise ValueError("유효 케이스 수가 최소 2개 이상이어야 합니다.")

    diff = a - b
    mean_vals = (a + b) / 2.0

    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # 평균 차이 95% CI — t분포 (SE = sd / sqrt(n))
    se_mean = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_mean_low = mean_diff - t_crit * se_mean
    ci_mean_high = mean_diff + t_crit * se_mean

    # LoA 95% CI — Bland-Altman 1986 원 공식: SE_LoA = sqrt(3 * sd^2 / n)
    se_loa = np.sqrt(3.0 * sd_diff ** 2 / n)
    ci_loa_upper_low = loa_upper - t_crit * se_loa
    ci_loa_upper_high = loa_upper + t_crit * se_loa
    ci_loa_lower_low = loa_lower - t_crit * se_loa
    ci_loa_lower_high = loa_lower + t_crit * se_loa

    # 비례 오차: diff vs mean Pearson 상관
    if sd_diff < 1e-12:
        # 차이가 모두 동일 → 상관 계산 불가
        proportional_bias_r = 0.0
        proportional_bias_p = 1.0
    else:
        proportional_bias_r, proportional_bias_p = stats.pearsonr(diff, mean_vals)

    return {
        "n": n,
        "mean_diff": float(mean_diff),
        "sd_diff": float(sd_diff),
        "loa_upper": float(loa_upper),
        "loa_lower": float(loa_lower),
        "ci_mean_low": float(ci_mean_low),
        "ci_mean_high": float(ci_mean_high),
        "ci_loa_upper_low": float(ci_loa_upper_low),
        "ci_loa_upper_high": float(ci_loa_upper_high),
        "ci_loa_lower_low": float(ci_loa_lower_low),
        "ci_loa_lower_high": float(ci_loa_lower_high),
        "proportional_bias_r": float(proportional_bias_r),
        "proportional_bias_p": float(proportional_bias_p),
    }


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Bland-Altman 일치도 분석을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋.
        spec: 분석 명세.
            variables.method1: 첫 번째 측정법 변수명
            variables.method2: 두 번째 측정법 변수명

    Returns:
        AnalysisResult — 4개 테이블:
            1. Case Processing Summary
            2. Bland-Altman Statistics
            3. Limits of Agreement
            4. Individual Differences
    """
    variables = spec.get("variables", {})
    var1: str | None = variables.get("method1")
    var2: str | None = variables.get("method2")

    result = AnalysisResult(
        id="bland_altman",
        title="Bland-Altman 일치도 분석",
    )

    # ── 변수 유효성 검사 ──────────────────────────────────────────
    if not var1 or not var2:
        result.warnings.append(
            "method1, method2 변수를 모두 지정해야 합니다. "
            "spec.variables.method1, spec.variables.method2를 확인하세요."
        )
        return result

    missing_cols = [v for v in [var1, var2] if v not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(
            f"다음 변수를 데이터셋에서 찾을 수 없습니다: {missing_cols}"
        )
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────
    # 두 열을 별도로 추출 (동일 변수 두 번 지정 시 중복 열 문제 방지)
    s1 = pd.to_numeric(dataset.data[var1], errors="coerce")
    s2 = pd.to_numeric(dataset.data[var2], errors="coerce")
    data = pd.DataFrame({"_m1": s1, "_m2": s2})

    n_total = len(data)
    data_clean = data.dropna()
    n_valid = len(data_clean)
    n_missing = n_total - n_valid

    if n_valid < 2:
        result.warnings.append(
            f"유효한 케이스가 부족합니다 (유효 = {n_valid}). "
            "최소 2개 이상의 완전한 관측값이 필요합니다."
        )
        return result

    a = data_clean["_m1"].values.astype(float)
    b = data_clean["_m2"].values.astype(float)

    # ── 통계 계산 ─────────────────────────────────────────────────
    try:
        r = _compute_bland_altman(a, b)
    except Exception as exc:
        result.add_warning(f"Bland-Altman 계산 오류: {exc}")
        return result

    # ── 테이블 1: Case Processing Summary ────────────────────────
    result.tables.append(get_cps_table_kr(n_total, n_valid, n_missing))

    # ── 테이블 2: Bland-Altman Statistics ────────────────────────
    bias_flag = " *" if r["proportional_bias_p"] < 0.05 else ""
    ba_rows = [
        ("평균 차이 (Bias)", format_number(r["mean_diff"], 3)),
        ("SD (차이)", format_number(r["sd_diff"], 3)),
        ("n", str(r["n"])),
        (
            f"비례 오차 r{bias_flag}",
            format_number(r["proportional_bias_r"], 3),
        ),
        (
            "비례 오차 p값",
            format_pvalue(r["proportional_bias_p"]),
        ),
    ]
    ba_df = pd.DataFrame(ba_rows, columns=["통계량", "값"])
    footnotes_ba: list[str] = []
    if r["proportional_bias_p"] < 0.05:
        footnotes_ba.append(
            "* 비례 오차가 통계적으로 유의합니다 (p < .05). "
            "로그 변환 후 재분석을 권장합니다."
        )
    else:
        footnotes_ba.append(
            "비례 오차 없음 (p ≥ .05) — LoA가 측정 범위 전반에 일정하게 적용됩니다."
        )
    result.tables.append(ResultTable(
        title="Bland-Altman Statistics",
        dataframe=ba_df,
        footnotes=footnotes_ba,
    ))

    # ── 테이블 3: Limits of Agreement ────────────────────────────
    loa_rows = [
        (
            "평균 차이 (Bias)",
            format_number(r["mean_diff"], 3),
            format_number(r["ci_mean_low"], 3),
            format_number(r["ci_mean_high"], 3),
        ),
        (
            "상한 일치 한계 (Upper LoA)",
            format_number(r["loa_upper"], 3),
            format_number(r["ci_loa_upper_low"], 3),
            format_number(r["ci_loa_upper_high"], 3),
        ),
        (
            "하한 일치 한계 (Lower LoA)",
            format_number(r["loa_lower"], 3),
            format_number(r["ci_loa_lower_low"], 3),
            format_number(r["ci_loa_lower_high"], 3),
        ),
    ]
    loa_df = pd.DataFrame(loa_rows, columns=["한계값", "추정치", "95% CI 하한", "95% CI 상한"])
    result.tables.append(ResultTable(
        title="Limits of Agreement",
        dataframe=loa_df,
        footnotes=[
            "LoA = 평균 차이 ± 1.96 × SD. "
            "95% CI: SE_LoA = √(3σ²/n) (Bland & Altman, 1986)."
        ],
    ))

    # ── 테이블 4: Individual Differences ─────────────────────────
    diff_arr = a - b
    mean_arr = (a + b) / 2.0
    sd = r["sd_diff"]
    std_diff_arr = diff_arr / sd if sd > 1e-12 else np.full_like(diff_arr, float("nan"))

    ind_rows = []
    for i in range(n_valid):
        ind_rows.append({
            "케이스": i + 1,
            "평균": format_number(float(mean_arr[i]), 3),
            "차이": format_number(float(diff_arr[i]), 3),
            "표준화 차이": format_number(float(std_diff_arr[i]), 4),
        })
    ind_df = pd.DataFrame(ind_rows)
    result.tables.append(ResultTable(
        title="Individual Differences",
        dataframe=ind_df,
        footnotes=[
            "평균 = (method1 + method2) / 2. "
            "차이 = method1 − method2. "
            "표준화 차이 = 차이 / SD."
        ],
    ))

    # ── 해석 노트 ─────────────────────────────────────────────────
    bias_desc = "편향 없음" if abs(r["mean_diff"]) < r["sd_diff"] * 0.1 else "편향 존재"
    result.notes.append(
        f"Bland-Altman 분석 | "
        f"평균 차이(bias) = {format_number(r['mean_diff'], 3)} | "
        f"SD = {format_number(r['sd_diff'], 3)} | "
        f"LoA: [{format_number(r['loa_lower'], 3)}, {format_number(r['loa_upper'], 3)}] | "
        f"{bias_desc} | "
        f"유효 케이스 = {n_valid}"
    )

    return result
