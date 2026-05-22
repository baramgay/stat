"""Partial Correlation Analysis — 편상관분석.

SPSS Analyze > Correlate > Partial 대응 모듈.

역행렬법(inverse matrix method):
    전체 상관행렬 R을 역행렬 Ri로 변환 후
    r_partial(i,j) = -Ri[i,j] / sqrt(Ri[i,i] * Ri[j,j])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_number, format_pvalue


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────────────

def _partial_corr_matrix(
    df: pd.DataFrame,
    target: list[str],
    controlling: list[str],
) -> pd.DataFrame:
    """편상관행렬 계산 (역행렬법).

    Args:
        df: 분석 데이터 (listwise 처리 완료)
        target: 편상관 계산 변수 목록
        controlling: 통제 변수 목록

    Returns:
        pd.DataFrame (index=target, columns=target)
        - 대각 원소 = 1.0
        - 통제 변수 없으면 Pearson 상관행렬 반환
    """
    if not controlling:
        # 통제 변수 없음 → Pearson 상관행렬
        corr = df[target].corr(method="pearson")
        return corr.reindex(index=target, columns=target)

    all_vars = target + controlling
    # 전체 상관행렬 R
    R = df[all_vars].corr(method="pearson").values

    # R 역행렬
    try:
        Ri = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # 특이행렬: 무상관 행렬 반환
        result = pd.DataFrame(np.eye(len(target)), index=target, columns=target)
        return result

    # target 변수 인덱스
    n_target = len(target)
    target_idx = list(range(n_target))  # target이 앞에 있음

    # 편상관행렬 계산
    mat = np.zeros((n_target, n_target))
    for i in target_idx:
        for j in target_idx:
            if i == j:
                mat[i, j] = 1.0
            else:
                denom = np.sqrt(Ri[i, i] * Ri[j, j])
                if denom == 0:
                    mat[i, j] = 0.0
                else:
                    mat[i, j] = -Ri[i, j] / denom
                # 범위 클리핑
                mat[i, j] = np.clip(mat[i, j], -1.0, 1.0)

    return pd.DataFrame(mat, index=target, columns=target)


def _calc_pvalue(r: float, df: int) -> float:
    """t 통계량으로 양측 p값 계산."""
    if df <= 0:
        return np.nan
    r_clipped = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
    denom = np.sqrt(1.0 - r_clipped ** 2)
    if denom == 0:
        return 0.0
    t_stat = r_clipped * np.sqrt(df) / denom
    return float(2.0 * stats.t.sf(abs(t_stat), df))


# ──────────────────────────────────────────────────────────────────────────────
# 메인 분석 함수
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """편상관분석을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.target: 분석 변수 목록 (최소 2개)
            variables.controlling: 통제 변수 목록 (0개 이상)
            options.listwise: True(기본) = 목록별 결측 제거

    Returns:
        AnalysisResult (테이블 4개)
            1. Case Processing Summary
            2. Partial Correlation
            3. Significance (2-tailed)
            4. Zero-order Correlations
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    target_vars: list[str] = variables.get("target", [])
    controlling: list[str] = variables.get("controlling", [])
    listwise: bool = options.get("listwise", True)

    result = AnalysisResult(id="partial_correlation", title="Partial Correlation")

    # ── 오류 처리: target 변수 수 확인 ──────────────────────────
    if len(target_vars) < 2:
        result.warnings.append("편상관분석에는 최소 2개 target 변수가 필요합니다.")
        return result

    # ── 오류 처리: 변수 존재 확인 ───────────────────────────────
    all_needed = target_vars + controlling
    missing_vars = [v for v in all_needed if v not in dataset.data.columns]
    if missing_vars:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_vars}")
        return result

    # ── 데이터 준비 ──────────────────────────────────────────────
    data = dataset.data[all_needed].copy()
    try:
        data = data.apply(pd.to_numeric, errors="coerce")
    except Exception:
        pass

    n_before = len(data)

    if listwise:
        data_clean = data.dropna()
    else:
        # pairwise: target+controlling 전부 유효한 행만 (단순화)
        data_clean = data.dropna()

    n_after = len(data_clean)
    n_excluded = n_before - n_after

    # ── 오류 처리: 유효 케이스 부족 ─────────────────────────────
    if n_after < 4:
        result.warnings.append(f"유효한 케이스가 부족합니다 (n={n_after}, 최소 4개 필요).")
        return result

    k = len(controlling)          # 통제변수 수
    df_val = n_after - 2 - k     # 자유도

    # ── 1. Case Processing Summary ───────────────────────────────
    cps_df = pd.DataFrame({
        "구분": ["유효", "제외됨", "합계"],
        "N": [n_after, n_excluded, n_before],
        "%": [
            round(n_after / n_before * 100, 1),
            round(n_excluded / n_before * 100, 1),
            100.0,
        ],
    })
    result.tables.append(ResultTable(title="Case Processing Summary", dataframe=cps_df))

    # ── 편상관행렬 계산 ──────────────────────────────────────────
    partial_mat = _partial_corr_matrix(data_clean, target_vars, controlling)

    # ── 0차 상관행렬 (Pearson, 통제 전) ─────────────────────────
    zero_mat = data_clean[target_vars].corr(method="pearson")
    zero_mat = zero_mat.reindex(index=target_vars, columns=target_vars)

    # ── 2. Partial Correlation 테이블 ────────────────────────────
    # float 값 그대로 저장 (float 비교 테스트 대응; 표시는 format_rules 위임)
    corr_rows = []
    for var_i in target_vars:
        row: dict = {"변수": var_i}
        for var_j in target_vars:
            if var_i == var_j:
                row[var_j] = 1.0
            else:
                row[var_j] = float(partial_mat.loc[var_i, var_j])
        corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows)
    result.tables.append(ResultTable(title="Partial Correlation", dataframe=corr_df))

    # ── 3. Significance (2-tailed) 테이블 ───────────────────────
    sig_rows = []
    for var_i in target_vars:
        row = {"변수": var_i, "df": df_val if df_val > 0 else 0}
        for var_j in target_vars:
            if var_i == var_j:
                row[var_j] = ""   # 대각: 빈 문자열
            else:
                r_val = float(partial_mat.loc[var_i, var_j])
                p_val = _calc_pvalue(r_val, df_val)
                # 테스트에서 float()로 캐스팅하므로 숫자 문자열 필요
                row[var_j] = format_number(p_val, 4)
        sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows)
    result.tables.append(ResultTable(title="Significance (2-tailed)", dataframe=sig_df))

    # ── 4. Zero-order Correlations 테이블 ───────────────────────
    zero_df_val = n_after - 2  # 0차 상관 자유도 (통제 없음)
    zero_rows = []
    for var_i in target_vars:
        row = {"변수": var_i, "df": zero_df_val}
        for var_j in target_vars:
            if var_i == var_j:
                row[var_j] = format_number(1.0, 3)
            else:
                row[var_j] = format_number(float(zero_mat.loc[var_i, var_j]), 3)
        zero_rows.append(row)
    zero_order_df = pd.DataFrame(zero_rows)
    result.tables.append(
        ResultTable(title="Zero-order Correlations", dataframe=zero_order_df)
    )

    # ── 메모 ─────────────────────────────────────────────────────
    result.notes.append(f"통제 변수: {controlling}, n={n_after}")

    return result
