"""다변량 분산분석(MANOVA — Multivariate Analysis of Variance) 분석 모듈.

SPSS: Analyze > General Linear Model > Multivariate

지원 기능:
  - 종속변수 2개 이상, 집단 간 요인 1개
  - 다변량 검정: Pillai's Trace, Wilks' Lambda, Hotelling-Lawley Trace, Roy's Largest Root
  - 단변량 후속 검정 (Type III SS, 각 종속변수별 F)
  - 편 η² (Partial Eta Squared)
  - 사후 검정: Tukey HSD / Bonferroni
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

from nuristat.analysis.assumptions import (  # noqa: F401
    get_cps_table_kr,
    prepare_analysis_frame,
)
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """다변량 분산분석(MANOVA)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.dependents:  종속변수 목록 (연속형, 최소 2개)
            variables.factor:      집단 간 요인 변수명 (범주형)
            options.multivariate:  True=다변량 검정 포함 (기본 True)
            options.univariate:    True=단변량 후속 검정 포함 (기본 True)
            options.post_hoc:      True=사후 검정 포함 (기본 True)
            options.post_hoc_method: "tukey" | "bonferroni" (기본 "bonferroni")
            options.effect_size:   True=편 η² 포함 (기본 True)
            confidence_level:      신뢰수준 (기본 0.95)
            missing_policy:        결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. Descriptive Statistics (집단별 평균/SD/N, 각 종속변수)
            3. Multivariate Tests (Pillai/Wilks/Hotelling/Roy)
            4. Tests of Between-Subjects Effects (단변량)
            5. Pairwise Comparisons (사후 검정, 선택)
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    dep_vars: list[str] = variables.get("dependents", [])
    factor_var: str = variables.get("factor", "")

    do_multivariate: bool = options.get("multivariate", True)
    do_univariate: bool = options.get("univariate", True)
    do_post_hoc: bool = options.get("post_hoc", True)
    post_hoc_method: str = options.get("post_hoc_method", "bonferroni").lower()
    do_effect: bool = options.get("effect_size", True)

    result = AnalysisResult(id="manova", title="다변량 분산분석 (MANOVA)")

    # ── 입력 검증 ─────────────────────────────────────────────────────────────
    all_cols = list(dataset.data.columns) if dataset.data is not None else []
    if not factor_var:
        result.add_warning("집단 간 요인 변수를 지정하세요.")
        return result
    if factor_var not in all_cols:
        result.add_warning(f"요인 변수 '{factor_var}'이(가) 데이터셋에 없습니다.")
        return result
    if len(dep_vars) < 2:
        result.add_warning("종속변수를 2개 이상 선택하세요.")
        return result
    missing_dvs = [v for v in dep_vars if v not in all_cols]
    if missing_dvs:
        result.add_warning(f"종속변수 없음: {missing_dvs}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────────────────
    needed = dep_vars + [factor_var]
    paf = prepare_analysis_frame(dataset, needed, missing_policy=missing_policy)
    df = paf.data.copy()
    result.add_table(get_cps_table_kr(paf.n_total, paf.n_valid, paf.n_excluded))

    groups = sorted(df[factor_var].dropna().unique())
    if len(groups) < 2:
        result.add_warning("집단이 2개 이상이어야 합니다.")
        return result
    if len(groups) >= len(df):
        result.add_warning("유효 케이스 수가 집단 수보다 많아야 합니다.")
        return result

    # 최소 케이스 수 확인 (각 집단 n > 종속변수 수)
    p = len(dep_vars)
    for g in groups:
        n_g = int((df[factor_var] == g).sum())
        if n_g <= p:
            result.add_warning(
                f"집단 '{g}'의 케이스 수({n_g})가 종속변수 수({p}) 이하입니다. "
                "MANOVA 수행 불가."
            )
            return result

    # 수치형 변환
    for dv in dep_vars:
        df[dv] = pd.to_numeric(df[dv], errors="coerce")
    df = df.dropna(subset=dep_vars)
    N = len(df)

    # ── 기술통계 ─────────────────────────────────────────────────────────────
    desc_rows: list[dict] = []
    for dv in dep_vars:
        for g in groups:
            sub = df.loc[df[factor_var] == g, dv]
            desc_rows.append({
                "종속변수": dv,
                "집단": str(g),
                "평균": format_number(sub.mean(), 4),
                "표준편차": format_number(sub.std(ddof=1), 4),
                "N": int(len(sub)),
            })
        # 전체
        desc_rows.append({
            "종속변수": dv,
            "집단": "전체",
            "평균": format_number(df[dv].mean(), 4),
            "표준편차": format_number(df[dv].std(ddof=1), 4),
            "N": N,
        })
    desc_table = ResultTable(
        title="기술통계",
        dataframe=pd.DataFrame(desc_rows),
    )
    result.add_table(desc_table)

    # ── 다변량 검정 ──────────────────────────────────────────────────────────
    if do_multivariate:
        mv_rows = _multivariate_tests(df, dep_vars, factor_var, groups, do_effect)
        mv_table = ResultTable(
            title="다변량 검정 (Multivariate Tests)",
            dataframe=pd.DataFrame(mv_rows),
        )
        result.add_table(mv_table)

    # ── 단변량 후속 검정 ──────────────────────────────────────────────────────
    if do_univariate:
        univ_rows = _univariate_tests(df, dep_vars, factor_var, do_effect)
        univ_table = ResultTable(
            title="개체 간 효과 검정 (Tests of Between-Subjects Effects)",
            dataframe=pd.DataFrame(univ_rows),
        )
        result.add_table(univ_table)

    # ── 사후 검정 ────────────────────────────────────────────────────────────
    if do_post_hoc and len(groups) >= 2:
        ph_rows = _post_hoc(df, dep_vars, factor_var, groups, post_hoc_method, confidence_level)
        if ph_rows:
            ph_table = ResultTable(
                title=f"쌍별 비교 ({post_hoc_method.capitalize()})",
                dataframe=pd.DataFrame(ph_rows),
            )
            result.add_table(ph_table)

    # ── 각주 ──────────────────────────────────────────────────────────────────
    notes = [
        f"종속변수: {', '.join(dep_vars)}",
        f"요인: {factor_var} (집단 {len(groups)}개, N={N})",
        "다변량 검정: Pillai's Trace, Wilks' Lambda, Hotelling-Lawley Trace, Roy's Largest Root",
    ]
    if do_effect:
        notes.append("편 η² = SS_효과 / (SS_효과 + SS_오차)")
    for n in notes:
        result.notes.append(n)

    return result


# ─────────────────────────── helpers ────────────────────────────────────────

def _multivariate_tests(
    df: pd.DataFrame,
    dep_vars: list[str],
    factor_var: str,
    groups: list,
    do_effect: bool,
) -> list[dict]:
    """statsmodels MANOVA로 다변량 검정 4종 수행."""
    rows: list[dict] = []
    safe_names = {v: f"_dv{i}" for i, v in enumerate(dep_vars)}
    df2 = df.rename(columns=safe_names)
    safe_factor = "_factor"
    df2[safe_factor] = df[factor_var].astype(str)

    lhs = " + ".join(safe_names[v] for v in dep_vars)
    formula = f"{lhs} ~ C({safe_factor})"

    try:
        maov = MANOVA.from_formula(formula, data=df2)
        mv_res = maov.mv_test()
        # effect key: 'C(_factor)'
        effect_key = None
        for k in mv_res.results:
            if safe_factor in k and "Intercept" not in k:
                effect_key = k
                break
        if effect_key is None and mv_res.results:
            keys = [k for k in mv_res.results if "Intercept" not in k]
            effect_key = keys[0] if keys else list(mv_res.results.keys())[-1]

        if effect_key is None:
            return rows
        stat_df = mv_res.results[effect_key]["stat"]
        # columns: Value, Num DF, Den DF, F Value, Pr > F
        stat_names = [
            "Pillai's Trace",
            "Wilks' Lambda",
            "Hotelling-Lawley Trace",
            "Roy's Largest Root",
        ]
        for i, test_name in enumerate(stat_names):
            if i >= len(stat_df):
                continue
            row = stat_df.iloc[i]
            value = float(row.get("Value", row.iloc[0]))
            f_val = float(row.get("F Value", row.iloc[3]))
            df1 = float(row.get("Num DF", row.iloc[1]))
            df2_val = float(row.get("Den DF", row.iloc[2]))
            p_val = float(row.get("Pr > F", row.iloc[4]))

            r: dict = {
                "검정": test_name,
                "값": format_number(value, 4),
                "F": format_number(f_val, 3),
                "가설 df": format_number(df1, 0) if df1 == int(df1) else format_number(df1, 2),
                "오차 df": format_number(df2_val, 3),
                "p": format_pvalue(p_val),
            }
            if do_effect:
                # partial η² for Pillai/Wilks: ≈ F*df1 / (F*df1 + df2)
                if f_val > 0 and df1 > 0 and df2_val > 0:
                    peta2 = (f_val * df1) / (f_val * df1 + df2_val)
                else:
                    peta2 = float("nan")
                r["편 η²"] = format_number(peta2, 4)
            rows.append(r)
    except Exception as e:
        logger.warning("MANOVA 다변량 검정 실패: %s", e)
        rows.append({"검정": "오류", "값": str(e), "F": "-", "가설 df": "-", "오차 df": "-", "p": "-"})

    return rows


def _univariate_tests(
    df: pd.DataFrame,
    dep_vars: list[str],
    factor_var: str,
    do_effect: bool,
) -> list[dict]:
    """각 종속변수에 대해 일원 분산분석 (Type III SS)."""
    rows: list[dict] = []
    for dv in dep_vars:
        try:
            safe_dv = "_y"
            safe_f = "_f"
            df2 = df[[dv, factor_var]].copy()
            df2.columns = [safe_dv, safe_f]
            df2[safe_f] = df2[safe_f].astype(str)
            model = ols(f"{safe_dv} ~ C({safe_f})", data=df2).fit()
            at = sm.stats.anova_lm(model, typ=3)
            at = at[at.index != "Intercept"]

            factor_row = at[at.index.str.contains(safe_f, regex=False)]
            error_row = at[at.index == "Residual"]
            if factor_row.empty or error_row.empty:
                continue

            ss_factor = float(factor_row["sum_sq"].iloc[0])
            ss_error = float(error_row["sum_sq"].iloc[0])
            df_factor = float(factor_row["df"].iloc[0])
            f_val = float(factor_row["F"].iloc[0])
            p_val = float(factor_row["PR(>F)"].iloc[0])

            r: dict = {
                "종속변수": dv,
                "SS": format_number(ss_factor, 4),
                "df": int(df_factor),
                "MS": format_number(ss_factor / df_factor if df_factor > 0 else float("nan"), 4),
                "F": format_number(f_val, 3),
                "p": format_pvalue(p_val),
            }
            if do_effect:
                peta2 = ss_factor / (ss_factor + ss_error) if (ss_factor + ss_error) > 0 else float("nan")
                r["편 η²"] = format_number(peta2, 4)
            rows.append(r)
        except Exception as e:
            logger.warning("단변량 검정 오류 (%s): %s", dv, e)
    return rows


def _post_hoc(
    df: pd.DataFrame,
    dep_vars: list[str],
    factor_var: str,
    groups: list,
    method: str,
    confidence_level: float,
) -> list[dict]:
    """종속변수별 쌍별 비교."""
    from itertools import combinations
    rows: list[dict] = []
    alpha = 1 - confidence_level
    pairs = list(combinations(groups, 2))
    n_pairs = len(pairs)

    for dv in dep_vars:
        for g1, g2 in pairs:
            x1 = df.loc[df[factor_var] == g1, dv].dropna().values
            x2 = df.loc[df[factor_var] == g2, dv].dropna().values
            if len(x1) < 2 or len(x2) < 2:
                continue

            t_stat, p_raw = stats.ttest_ind(x1, x2, equal_var=False)
            mean_diff = float(x1.mean() - x2.mean())

            if method == "bonferroni":
                p_adj = min(p_raw * n_pairs, 1.0)
            elif method == "tukey":
                # Tukey approximation via studentized range
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                try:
                    all_data = pd.concat([
                        pd.DataFrame({dv: x1, "g": str(g1)}),
                        pd.DataFrame({dv: x2, "g": str(g2)}),
                    ])
                    tukey = pairwise_tukeyhsd(all_data[dv], all_data["g"], alpha=alpha)
                    p_adj = float(tukey.pvalues[0]) if len(tukey.pvalues) > 0 else p_raw
                except Exception:
                    p_adj = min(p_raw * n_pairs, 1.0)
            else:
                p_adj = p_raw

            # SE and CI
            s1, s2 = float(x1.std(ddof=1)), float(x2.std(ddof=1))
            n1, n2 = len(x1), len(x2)
            se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
            df_w = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / (
                (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
            )
            t_crit = float(stats.t.ppf(1 - alpha / 2, df_w))
            ci_lo = mean_diff - t_crit * se
            ci_hi = mean_diff + t_crit * se

            rows.append({
                "종속변수": dv,
                "집단(I)": str(g1),
                "집단(J)": str(g2),
                "평균차 (I-J)": format_number(mean_diff, 4),
                "표준오차": format_number(se, 4),
                "p (조정)": format_pvalue(p_adj),
                f"하한 ({confidence_level*100:.0f}%CI)": format_number(ci_lo, 4),
                f"상한 ({confidence_level*100:.0f}%CI)": format_number(ci_hi, 4),
            })

    return rows
