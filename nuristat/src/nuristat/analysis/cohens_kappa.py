"""Cohen's Kappa — 평가자 간 일치도 분석.

SPSS: Analyze > Descriptive Statistics > Crosstabs > Statistics > Kappa 대응 모듈.

참고문헌:
    Cohen, J. (1960). A coefficient of agreement for nominal scales.
        Educational and Psychological Measurement, 20(1), 37-46.
    Fleiss, J.L. (1971). Measuring nominal scale agreement among many raters.
        Psychological Bulletin, 76(5), 378-382.
    Landis, J.R. & Koch, G.G. (1977). The measurement of observer agreement
        for categorical data. Biometrics, 33(1), 159-174.
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

# ---------------------------------------------------------------------------
# Landis-Koch 해석 등급
# ---------------------------------------------------------------------------

def _landis_koch_grade(kappa: float) -> str:
    """Landis-Koch (1977) 기준 Cohen's Kappa 해석 등급 반환."""
    if kappa < 0:
        return "없음 (Poor)"
    elif kappa < 0.20:
        return "약함 (Slight)"
    elif kappa < 0.40:
        return "보통 (Fair)"
    elif kappa < 0.60:
        return "중간 (Moderate)"
    elif kappa < 0.80:
        return "상당 (Substantial)"
    else:
        return "거의 완전 (Almost Perfect)"


# ---------------------------------------------------------------------------
# 핵심 계산 함수
# ---------------------------------------------------------------------------

def _compute_kappa(rater1: list | np.ndarray | pd.Series,
                   rater2: list | np.ndarray | pd.Series) -> dict:
    """Cohen's Kappa 계수 및 관련 통계량 계산.

    Args:
        rater1: 평가자1 평가 결과 (1차원 배열)
        rater2: 평가자2 평가 결과 (1차원 배열)

    Returns:
        dict with keys:
            kappa      : Cohen's kappa 계수
            po         : 관찰 일치율 (Observed agreement)
            pe         : 기대 일치율 (Expected agreement by chance)
            se         : 표준오차 (Fleiss 1971)
            z          : z 통계량 (kappa / se)
            p          : 양측 p값
            ci_lower   : 95% 신뢰구간 하한
            ci_upper   : 95% 신뢰구간 상한
            n          : 유효 케이스 수
            categories : 정렬된 범주 목록
    """
    a1 = np.asarray(rater1)
    a2 = np.asarray(rater2)

    if len(a1) != len(a2):
        raise ValueError("두 평가자의 관측치 수가 다릅니다.")
    if len(a1) == 0:
        raise ValueError("데이터가 비어 있습니다.")

    categories = sorted(set(a1.tolist()) | set(a2.tolist()))
    n = len(a1)

    # 관찰 일치율 (Po)
    po = float(np.sum(a1 == a2) / n)

    # 기대 일치율 (Pe)
    pe = 0.0
    for c in categories:
        p_a = float(np.sum(a1 == c)) / n
        p_b = float(np.sum(a2 == c)) / n
        pe += p_a * p_b

    # Kappa
    if abs(1 - pe) < 1e-10:
        # Pe = 1 이면 kappa 정의 불가 (완전 기대 일치)
        kappa = 0.0
    else:
        kappa = (po - pe) / (1 - pe)

    # 표준오차 (Fleiss 1971)
    denominator = n * (1 - pe) ** 2
    if denominator <= 0 or po <= 0 or po >= 1:
        se = np.nan
    else:
        se = float(np.sqrt(po * (1 - po) / denominator))

    # z 통계량 및 p값
    if np.isnan(se) or se == 0:
        z = np.nan
        p = np.nan
    else:
        z = float(kappa / se)
        p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # 95% 신뢰구간
    if np.isnan(se):
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        ci_lower = float(kappa - 1.96 * se)
        ci_upper = float(kappa + 1.96 * se)

    return {
        "kappa": kappa,
        "po": po,
        "pe": pe,
        "se": se,
        "z": z,
        "p": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# 공개 분석 함수
# ---------------------------------------------------------------------------

def run_analysis(dataset: Dataset, spec: object | dict) -> AnalysisResult:
    """Cohen's Kappa 평가자 간 일치도 분석을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세 (dict 또는 .variables 속성 보유 객체)
            variables.rater1: 평가자1 변수명
            variables.rater2: 평가자2 변수명

    Returns:
        AnalysisResult — 4개 테이블 포함:
            1. Case Processing Summary
            2. Crosstabulation
            3. Symmetric Measures
            4. Interpretation
    """
    result = AnalysisResult(id="cohens_kappa", title="Cohen's Kappa")

    # spec 파싱 (dict / 객체 모두 지원)
    if isinstance(spec, dict):
        variables = spec.get("variables", {})
        rater1_var: str = variables.get("rater1", "")
        rater2_var: str = variables.get("rater2", "")
    else:
        variables = getattr(spec, "variables", {}) or {}
        if isinstance(variables, dict):
            rater1_var = variables.get("rater1", "")
            rater2_var = variables.get("rater2", "")
        else:
            rater1_var = getattr(variables, "rater1", "")
            rater2_var = getattr(variables, "rater2", "")

    # 변수 유효성 검사
    if not rater1_var or not rater2_var:
        result.warnings.append("rater1과 rater2 변수를 모두 지정해야 합니다.")
        return result

    if rater1_var not in dataset.data.columns:
        result.warnings.append(f"변수를 찾을 수 없습니다: {rater1_var}")
        return result

    if rater2_var not in dataset.data.columns:
        result.warnings.append(f"변수를 찾을 수 없습니다: {rater2_var}")
        return result

    # 데이터 준비
    df = dataset.data[[rater1_var, rater2_var]].copy()
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    n_excluded = n_before - n_after

    if n_after < 2:
        result.warnings.append("유효한 케이스가 부족합니다 (최소 2개 필요).")
        return result

    # 단일 범주 검사
    unique1 = df[rater1_var].nunique()
    unique2 = df[rater2_var].nunique()
    if unique1 < 2 or unique2 < 2:
        which = []
        if unique1 < 2:
            which.append(rater1_var)
        if unique2 < 2:
            which.append(rater2_var)
        result.warnings.append(
            f"단일 범주 변수가 있습니다: {which}. Kappa를 계산할 수 없습니다."
        )
        return result

    # Kappa 계산
    try:
        stats_dict = _compute_kappa(df[rater1_var].tolist(), df[rater2_var].tolist())
    except Exception as exc:
        result.add_warning(f"Cohen's Kappa 계산 오류: {exc}")
        return result
    kappa = stats_dict["kappa"]
    categories = stats_dict["categories"]

    # ── 테이블 1: Case Processing Summary ────────────────────────
    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    # ── 테이블 2: Crosstabulation ─────────────────────────────────
    ct = pd.crosstab(
        df[rater1_var],
        df[rater2_var],
        margins=True,
        margins_name="합계",
    )
    ct.index.name = rater1_var
    ct.columns.name = rater2_var
    result.tables.append(ResultTable(title="Crosstabulation", dataframe=ct.reset_index()))

    # ── 테이블 3: Symmetric Measures ─────────────────────────────
    measures_df = pd.DataFrame({
        "측도": ["Cohen's Kappa"],
        "값": [format_number(kappa, 3)],
        "표준오차": [format_number(stats_dict["se"], 3)],
        "근사 T": [format_number(stats_dict["z"], 3)],
        "유의확률": [format_pvalue(stats_dict["p"])],
        "95% CI 하한": [format_number(stats_dict["ci_lower"], 3)],
        "95% CI 상한": [format_number(stats_dict["ci_upper"], 3)],
    })
    result.tables.append(ResultTable(title="Symmetric Measures", dataframe=measures_df))

    # ── 테이블 4: Interpretation ──────────────────────────────────
    grade = _landis_koch_grade(kappa)
    interp_df = pd.DataFrame({
        "통계량": ["Cohen's Kappa"],
        "값": [format_number(kappa, 3)],
        "해석 (Landis-Koch)": [grade],
        "관찰 일치율 (Po)": [format_number(stats_dict["po"], 3)],
        "기대 일치율 (Pe)": [format_number(stats_dict["pe"], 3)],
        "케이스 수": [n_after],
    })
    result.tables.append(ResultTable(title="Interpretation", dataframe=interp_df))

    # 해석 메모
    result.notes.append(
        f"Cohen's κ = {format_number(kappa, 3)} — {grade} | "
        f"Po = {format_number(stats_dict['po'], 3)}, "
        f"Pe = {format_number(stats_dict['pe'], 3)}, "
        f"N = {n_after}, 범주 수 = {len(categories)}"
    )

    return result
