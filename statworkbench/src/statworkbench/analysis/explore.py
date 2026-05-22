"""탐색적 분석(Explore) 모듈 — SPSS Analyze > Descriptive Statistics > Explore 대응.

SPSS 29/30 호환 출력:
    1. Case Processing Summary — 변수별 유효/결측/합계
    2. Descriptives         — SPSS Explore 스타일 세로형 통계량 테이블
    3. Extreme Values       — 최솟값 5개, 최댓값 5개 (순위 포함)
    4. Tests of Normality   — Shapiro-Wilk W, p
    5. Percentiles          — P5, P10, P25, P50, P75, P90, P95

공개 API:
    from statworkbench.analysis.explore import run_analysis
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.assumptions import prepare_analysis_frame


# ────────────────────────────────────────────────────────────────
# 내부 계산 함수
# ────────────────────────────────────────────────────────────────

def _compute_explore_stats(arr: np.ndarray) -> dict:
    """단일 변수의 탐색적 분석 통계량 계산.

    Parameters
    ----------
    arr : np.ndarray
        결측 제외된 1차원 배열.

    Returns
    -------
    dict
        SPSS Explore > Descriptives 구성 통계량.
    """
    n = len(arr)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean": nan, "trimmed_mean": nan, "median": nan,
            "sd": nan, "variance": nan,
            "min": nan, "max": nan, "range": nan, "iqr": nan,
            "q1": nan, "q3": nan,
            "skewness": nan, "skewness_se": nan,
            "kurtosis": nan, "kurtosis_se": nan,
            "ci_lower": nan, "ci_upper": nan,
            "se_mean": nan,
            "shapiro_w": nan, "shapiro_p": nan,
        }

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    rng = mx - mn

    # 5% 절사평균: scipy.stats.trim_mean(data, 0.05)
    trimmed_mean = float(stats.trim_mean(arr, 0.05))

    # 왜도/첨도 — SPSS bias=False (불편 추정)
    skewness = float(stats.skew(arr, bias=False)) if n >= 3 else float("nan")
    kurtosis = float(stats.kurtosis(arr, bias=False)) if n >= 4 else float("nan")

    # 표준오차 공식: SE_skew = sqrt(6/n), SE_kurt = sqrt(24/n)
    skewness_se = float(np.sqrt(6.0 / n))
    kurtosis_se = float(np.sqrt(24.0 / n))

    # 95% CI: mean ± t(0.025, df=n-1) * SE
    if n > 1:
        se_mean = sd / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean - t_crit * se_mean
        ci_upper = mean + t_crit * se_mean
    else:
        se_mean = 0.0
        ci_lower = ci_upper = mean

    # Shapiro-Wilk
    if n >= 3:
        shapiro_w, shapiro_p = stats.shapiro(arr)
        shapiro_w = float(shapiro_w)
        shapiro_p = float(shapiro_p)
    else:
        shapiro_w = shapiro_p = float("nan")

    return {
        "n": n,
        "mean": mean,
        "trimmed_mean": trimmed_mean,
        "median": median,
        "sd": sd,
        "variance": variance,
        "min": mn,
        "max": mx,
        "range": rng,
        "iqr": iqr,
        "q1": q1,
        "q3": q3,
        "skewness": skewness,
        "skewness_se": skewness_se,
        "kurtosis": kurtosis,
        "kurtosis_se": kurtosis_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se_mean": se_mean,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
    }


def _build_descriptives_rows(
    var_name: str,
    s: dict,
    group_label: Optional[str] = None,
) -> list[dict]:
    """SPSS Explore > Descriptives 세로형 행 목록 생성.

    Parameters
    ----------
    var_name : str
        변수명.
    s : dict
        _compute_explore_stats 반환값.
    group_label : str or None
        그룹값 (factor 분석 시).

    Returns
    -------
    list[dict]
        Descriptives 테이블 행 목록.
    """
    def _val(v):
        """NaN이면 빈 문자열, 아니면 원시 float 반환 (포맷은 테이블 출력 시)."""
        if v is None:
            return float("nan")
        if isinstance(v, float) and np.isnan(v):
            return float("nan")
        return float(v)

    stats_list = [
        ("평균",             _val(s["mean"])),
        ("95% 신뢰구간 하한", _val(s["ci_lower"])),
        ("95% 신뢰구간 상한", _val(s["ci_upper"])),
        ("5% 절사평균",       _val(s["trimmed_mean"])),
        ("중위수",            _val(s["median"])),
        ("표준편차",          _val(s["sd"])),
        ("분산",             _val(s["variance"])),
        ("최솟값",           _val(s["min"])),
        ("최댓값",           _val(s["max"])),
        ("범위",             _val(s["range"])),
        ("IQR",              _val(s["iqr"])),
        ("왜도",             _val(s["skewness"])),
        ("왜도 표준오차",     _val(s["skewness_se"])),
        ("첨도",             _val(s["kurtosis"])),
        ("첨도 표준오차",     _val(s["kurtosis_se"])),
    ]

    rows = []
    for stat_name, val in stats_list:
        row = {"변수": var_name, "통계량": stat_name, "값": val}
        if group_label is not None:
            row["그룹"] = group_label
        rows.append(row)
    return rows


def _build_extreme_values_rows(
    var_name: str,
    arr: np.ndarray,
    group_label: Optional[str] = None,
) -> list[dict]:
    """Extreme Values 행 목록 생성 (최솟값 5개 + 최댓값 5개).

    Parameters
    ----------
    var_name : str
        변수명.
    arr : np.ndarray
        결측 제외 배열.
    group_label : str or None
        그룹값.

    Returns
    -------
    list[dict]
        Extreme Values 테이블 행.
    """
    rows = []
    if len(arr) == 0:
        return rows

    sorted_arr = np.sort(arr)
    n_extreme = min(5, len(sorted_arr))

    # 최솟값 (오름차순 상위 5)
    for rank, val in enumerate(sorted_arr[:n_extreme], start=1):
        row = {
            "변수": var_name,
            "유형": "최솟값",
            "순위": rank,
            "값": float(val),
        }
        if group_label is not None:
            row["그룹"] = group_label
        rows.append(row)

    # 최댓값 (내림차순 상위 5)
    for rank, val in enumerate(sorted_arr[-n_extreme:][::-1], start=1):
        row = {
            "변수": var_name,
            "유형": "최댓값",
            "순위": rank,
            "값": float(val),
        }
        if group_label is not None:
            row["그룹"] = group_label
        rows.append(row)

    return rows


def _build_normality_rows(
    var_name: str,
    s: dict,
    group_label: Optional[str] = None,
) -> list[dict]:
    """Tests of Normality 행 생성.

    Parameters
    ----------
    var_name : str
        변수명.
    s : dict
        _compute_explore_stats 반환값.
    group_label : str or None
        그룹값.

    Returns
    -------
    list[dict]
    """
    row = {
        "변수": var_name,
        "N": s["n"],
        "Shapiro-Wilk W": s["shapiro_w"],
        "Shapiro-Wilk p": s["shapiro_p"],
    }
    if group_label is not None:
        row["그룹"] = group_label
    return [row]


def _build_percentile_rows(
    var_name: str,
    arr: np.ndarray,
    percentiles: list[int],
    group_label: Optional[str] = None,
) -> list[dict]:
    """Percentiles 테이블 행 생성.

    Parameters
    ----------
    var_name : str
        변수명.
    arr : np.ndarray
        결측 제외 배열.
    percentiles : list[int]
        계산할 백분위수 목록 (예: [5, 10, 25, 50, 75, 90, 95]).
    group_label : str or None
        그룹값.

    Returns
    -------
    list[dict]
    """
    row: dict = {"변수": var_name}
    if group_label is not None:
        row["그룹"] = group_label
    if len(arr) == 0:
        for p in percentiles:
            row[f"P{p}"] = float("nan")
    else:
        for p in percentiles:
            row[f"P{p}"] = float(np.percentile(arr, p))
    return [row]


def _build_case_processing_summary(
    dataset: Dataset,
    target_vars: list[str],
    df: pd.DataFrame,
    df_full: pd.DataFrame,
) -> ResultTable:
    """Case Processing Summary 테이블 생성.

    Parameters
    ----------
    dataset : Dataset
        원본 데이터셋 (전체 N 파악용).
    target_vars : list[str]
        분석 대상 변수 목록.
    df : pd.DataFrame
        결측 처리 후 데이터.
    df_full : pd.DataFrame
        결측 처리 전 부분집합.

    Returns
    -------
    ResultTable
    """
    rows = []
    for var in target_vars:
        if var not in df_full.columns:
            continue
        n_total = len(df_full)
        n_missing = int(df_full[var].isna().sum())
        n_valid = n_total - n_missing
        rows.append({
            "변수": var,
            "유효 N": n_valid,
            "결측 N": n_missing,
            "합계 N": n_total,
            "유효 %": f"{(n_valid / n_total * 100) if n_total > 0 else 0:.1f}%",
            "결측 %": f"{(n_missing / n_total * 100) if n_total > 0 else 0:.1f}%",
        })

    return ResultTable(
        title="Case Processing Summary",
        dataframe=pd.DataFrame(rows),
        footnotes=["결측값은 listwise 방식으로 제외됩니다."],
    )


# ────────────────────────────────────────────────────────────────
# 공개 API
# ────────────────────────────────────────────────────────────────

def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """탐색적 분석(Explore) 실행.

    SPSS: Analyze > Descriptive Statistics > Explore 대응.

    Parameters
    ----------
    dataset : Dataset
        분석할 데이터셋.
    spec : dict
        분석 명세. 구조:

        .. code-block:: python

            {
                "variables": {
                    "target": ["변수1", "변수2"],   # 분석 대상 연속형 변수
                    "factor": "그룹변수",           # (선택) 그룹 변수
                },
                "options": {
                    "percentiles": [5, 10, 25, 50, 75, 90, 95],  # 기본
                    "normality": True,              # Shapiro-Wilk 포함 여부
                },
            }

    Returns
    -------
    AnalysisResult
        테이블 5개:
        1. Case Processing Summary
        2. Descriptives
        3. Extreme Values
        4. Tests of Normality
        5. Percentiles

    Raises
    ------
    ValueError
        target 변수가 데이터셋에 존재하지 않을 때.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    target_vars: list[str] = variables.get("target", [])
    factor_var: Optional[str] = variables.get("factor", None)
    percentiles: list[int] = options.get("percentiles", [5, 10, 25, 50, 75, 90, 95])
    include_normality: bool = options.get("normality", True)

    result = AnalysisResult(
        id="explore",
        title="탐색적 분석 (Explore)",
        spec=spec,
    )

    # 변수 존재 확인
    for var in target_vars:
        if var not in dataset.data.columns:
            raise ValueError(f"변수 '{var}'이(가) 데이터셋에 없습니다.")
    if factor_var is not None and factor_var not in dataset.data.columns:
        raise ValueError(f"factor 변수 '{factor_var}'이(가) 데이터셋에 없습니다.")

    # 결측 처리 전 원본 부분집합 (Case Processing Summary용)
    all_vars = list(target_vars)
    if factor_var:
        all_vars.append(factor_var)

    if not target_vars:
        # 빈 target — 5개 빈 테이블 반환
        for title in [
            "Case Processing Summary", "Descriptives",
            "Extreme Values", "Tests of Normality", "Percentiles",
        ]:
            result.add_table(ResultTable(title=title, dataframe=pd.DataFrame()))
        return result

    df_full_subset = dataset.data[all_vars].copy()

    # Case Processing Summary (변수별 개별 결측 집계)
    cps_table = _build_case_processing_summary(
        dataset, target_vars, df=df_full_subset, df_full=df_full_subset
    )
    result.add_table(cps_table)

    # 그룹별 vs 전체 분석
    desc_rows: list[dict] = []
    extreme_rows: list[dict] = []
    norm_rows: list[dict] = []
    pct_rows: list[dict] = []

    if factor_var is not None:
        # 그룹별 분석
        df_factor = df_full_subset.copy()
        groups = sorted(df_factor[factor_var].dropna().unique())
        for grp in groups:
            grp_mask = df_factor[factor_var] == grp
            for var in target_vars:
                arr = df_factor.loc[grp_mask, var].dropna().values.astype(float)
                if len(arr) == 0:
                    result.warnings.append(
                        f"변수 '{var}', 그룹 '{grp}': 유효한 데이터 없음."
                    )
                s = _compute_explore_stats(arr)
                desc_rows.extend(_build_descriptives_rows(var, s, group_label=str(grp)))
                extreme_rows.extend(_build_extreme_values_rows(var, arr, group_label=str(grp)))
                if include_normality:
                    norm_rows.extend(_build_normality_rows(var, s, group_label=str(grp)))
                pct_rows.extend(_build_percentile_rows(var, arr, percentiles, group_label=str(grp)))
    else:
        # 전체 분석
        for var in target_vars:
            arr = df_full_subset[var].dropna().values.astype(float)
            if len(arr) == 0:
                result.warnings.append(f"변수 '{var}': 유효한 데이터가 없습니다.")
            s = _compute_explore_stats(arr)
            desc_rows.extend(_build_descriptives_rows(var, s))
            extreme_rows.extend(_build_extreme_values_rows(var, arr))
            if include_normality:
                norm_rows.extend(_build_normality_rows(var, s))
            pct_rows.extend(_build_percentile_rows(var, arr, percentiles))

    # ── 테이블 2: Descriptives ──
    desc_df = pd.DataFrame(desc_rows)
    # 컬럼 순서 정렬
    cols_order = ["그룹", "변수", "통계량", "값"] if "그룹" in desc_df.columns \
        else ["변수", "통계량", "값"]
    desc_df = desc_df[[c for c in cols_order if c in desc_df.columns]]
    result.add_table(ResultTable(
        title="Descriptives",
        dataframe=desc_df,
        footnotes=[
            "5% 절사평균: scipy.stats.trim_mean(data, 0.05)",
            "왜도 SE = sqrt(6/n), 첨도 SE = sqrt(24/n)",
            "95% CI: mean ± t(0.025, df=n-1) × SE",
        ],
    ))

    # ── 테이블 3: Extreme Values ──
    ext_df = pd.DataFrame(extreme_rows)
    if not ext_df.empty:
        cols_order_ext = ["그룹", "변수", "유형", "순위", "값"] if "그룹" in ext_df.columns \
            else ["변수", "유형", "순위", "값"]
        ext_df = ext_df[[c for c in cols_order_ext if c in ext_df.columns]]
    result.add_table(ResultTable(
        title="Extreme Values",
        dataframe=ext_df,
        footnotes=["최솟값 및 최댓값 각 상위 5개 표시."],
    ))

    # ── 테이블 4: Tests of Normality ──
    if include_normality:
        if norm_rows:
            norm_df = pd.DataFrame(norm_rows)
            cols_order_norm = (
                ["그룹", "변수", "N", "Shapiro-Wilk W", "Shapiro-Wilk p"]
                if "그룹" in norm_df.columns
                else ["변수", "N", "Shapiro-Wilk W", "Shapiro-Wilk p"]
            )
            norm_df = norm_df[[c for c in cols_order_norm if c in norm_df.columns]]
        else:
            norm_df = pd.DataFrame()
        result.add_table(ResultTable(
            title="Tests of Normality",
            dataframe=norm_df,
            footnotes=["* Shapiro-Wilk 검정 기반 정규성 검정."],
        ))

    # ── 테이블 5: Percentiles ──
    pct_df = pd.DataFrame(pct_rows)
    if not pct_df.empty:
        base_cols = ["그룹", "변수"] if "그룹" in pct_df.columns else ["변수"]
        pct_cols = [f"P{p}" for p in percentiles]
        all_pct_cols = base_cols + [c for c in pct_cols if c in pct_df.columns]
        pct_df = pct_df[all_pct_cols]
    result.add_table(ResultTable(
        title="Percentiles",
        dataframe=pct_df,
        footnotes=[
            f"백분위수: {', '.join([f'P{p}' for p in percentiles])}",
            "계산 방법: numpy.percentile (선형 보간, 기본값)",
        ],
    ))

    return result
