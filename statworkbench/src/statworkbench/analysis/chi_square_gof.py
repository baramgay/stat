"""Chi-Square Goodness-of-Fit Analysis — 카이제곱 적합도 검정.

SPSS: Analyze > Nonparametric Tests > Legacy Dialogs > Chi-Square 대응 모듈.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_number, format_pvalue


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """카이제곱 적합도 검정(Chi-Square Goodness-of-Fit)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.target: 범주형 변수 목록 (1개 이상)
            variables.expected_ratios: dict {범주값: 비율} (없으면 균등 분포)
            options.listwise: True=목록별 결측 제거 (기본값 True)

    Returns:
        AnalysisResult — 4개 테이블 포함:
            1. Case Processing Summary
            2. Frequencies
            3. Test Statistics
            4. Residuals
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})

    target_vars: list[str] = variables.get("target", [])
    expected_ratios: dict | None = variables.get("expected_ratios", None)
    listwise: bool = options.get("listwise", True)

    result = AnalysisResult(id="chi_square_gof", title="Chi-Square Goodness-of-Fit")

    # ── 변수 검증 ────────────────────────────────────────────────
    if len(target_vars) == 0:
        result.warnings.append("분석 변수가 지정되지 않았습니다.")
        return result

    missing_cols = [v for v in target_vars if v not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    data = dataset.data[target_vars].copy()

    # ── 결측 처리 ────────────────────────────────────────────────
    n_before = len(data)
    if listwise:
        data = data.dropna(subset=target_vars)
    n_after = len(data)
    n_excluded = n_before - n_after

    # ── Case Processing Summary 테이블 ───────────────────────────
    cps_df = pd.DataFrame({
        "구분": ["유효", "결측", "합계"],
        "N": [n_after, n_excluded, n_before],
        "%": [
            round(n_after / n_before * 100, 1) if n_before > 0 else 0.0,
            round(n_excluded / n_before * 100, 1) if n_before > 0 else 0.0,
            100.0,
        ],
    })
    result.tables.append(ResultTable(title="Case Processing Summary", dataframe=cps_df))

    if n_after == 0:
        result.warnings.append("유효한 케이스가 없습니다.")
        return result

    # ── 각 변수별 검정 수행 ──────────────────────────────────────
    freq_rows: list[dict] = []
    test_rows: list[dict] = []
    resid_rows: list[dict] = []

    for var in target_vars:
        series = data[var]
        observed_counts = series.value_counts().sort_index()
        categories = observed_counts.index.tolist()
        observed = observed_counts.values.astype(float)
        k = len(categories)

        # 단일 범주 오류 처리
        if k < 2:
            result.warnings.append(
                f"변수 '{var}': 범주가 1개뿐입니다. 검정을 수행할 수 없습니다."
            )
            continue

        # 기대 빈도 계산
        n = observed.sum()
        if expected_ratios is not None:
            # 지정된 비율 사용 — 범주값 기준으로 매핑
            ratios = np.array([expected_ratios.get(cat, 0.0) for cat in categories], dtype=float)
            ratio_sum = ratios.sum()
            if ratio_sum == 0:
                result.warnings.append(
                    f"변수 '{var}': 기대 비율 합계가 0입니다. 균등 분포로 대체합니다."
                )
                expected = np.full(k, n / k)
            else:
                expected = ratios / ratio_sum * n
        else:
            # 균등 분포
            expected = np.full(k, n / k)

        # 기대 빈도 0 검사
        if np.any(expected == 0):
            result.warnings.append(
                f"변수 '{var}': 기대 빈도가 0인 범주가 있습니다. 검정 결과가 유효하지 않을 수 있습니다."
            )
            continue

        # 카이제곱 검정
        chi2_stat, p_val = chisquare(observed, f_exp=expected)
        df = k - 1
        residuals = observed - expected
        std_residuals = residuals / np.sqrt(expected)

        # Frequencies 행 추가
        for cat, obs_val, exp_val, resid_val in zip(categories, observed, expected, residuals):
            freq_rows.append({
                "변수": var,
                "범주": cat,
                "관찰 빈도": int(obs_val),
                "기대 빈도": format_number(exp_val, 2),
                "잔차": format_number(resid_val, 2),
            })

        # Test Statistics 행 추가
        test_rows.append({
            "변수": var,
            "Chi-Square": format_number(chi2_stat, 3),
            "df": df,
            "Asymptotic Significance (p)": format_pvalue(p_val),
        })

        # Residuals 행 추가
        for cat, resid_val, std_resid in zip(categories, residuals, std_residuals):
            resid_rows.append({
                "변수": var,
                "범주": cat,
                "잔차": format_number(resid_val, 2),
                "표준화 잔차": format_number(std_resid, 3),
            })

    # ── 테이블 생성 ──────────────────────────────────────────────
    if freq_rows:
        freq_df = pd.DataFrame(freq_rows)
        result.tables.append(ResultTable(title="Frequencies", dataframe=freq_df))
    else:
        result.tables.append(ResultTable(
            title="Frequencies",
            dataframe=pd.DataFrame(columns=["변수", "범주", "관찰 빈도", "기대 빈도", "잔차"]),
        ))

    if test_rows:
        test_df = pd.DataFrame(test_rows)
        result.tables.append(ResultTable(title="Test Statistics", dataframe=test_df))
    else:
        result.tables.append(ResultTable(
            title="Test Statistics",
            dataframe=pd.DataFrame(
                columns=["변수", "Chi-Square", "df", "Asymptotic Significance (p)"]
            ),
        ))

    if resid_rows:
        resid_df = pd.DataFrame(resid_rows)
        result.tables.append(ResultTable(title="Residuals", dataframe=resid_df))
    else:
        result.tables.append(ResultTable(
            title="Residuals",
            dataframe=pd.DataFrame(columns=["변수", "범주", "잔차", "표준화 잔차"]),
        ))

    # ── 해석 메모 ────────────────────────────────────────────────
    for row in test_rows:
        var = row["변수"]
        chi2_str = row["Chi-Square"]
        df_val = row["df"]
        p_str = row["Asymptotic Significance (p)"]
        result.notes.append(
            f"[{var}] Chi-Square = {chi2_str}, df = {df_val}, "
            f"Asymptotic Significance (p) = {p_str}"
        )

    return result
