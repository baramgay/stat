"""공분산분석(ANCOVA — Analysis of Covariance) 분석 모듈.

SPSS: Analyze > General Linear Model > Univariate (공변량 포함)

지원 기능:
  - 1개 요인 + 1~3개 공변량
  - 공변량 조정 후 요인 효과 검정 (Type III SS, Sum 코딩)
  - 동질적 회귀 계수 가정 검정 (요인×공변량 상호작용)
  - 조정된 주변 평균 (Estimated Marginal Means)
  - 효과 크기: η² (Eta-squared)
  - 사후 검정: Bonferroni (조정 평균 기반)
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

from statworkbench.analysis.assumptions import get_cps_table_kr, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """공분산분석(ANCOVA)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.dependent:  종속변수 (연속형)
            variables.factor:     요인 변수 (범주형)
            variables.covariates: 공변량 목록 (연속형, 최대 3개)
            options.homogeneity_test: True=동질적 회귀 계수 가정 검정 (기본 True)
            options.emm:          True=조정된 주변 평균 출력 (기본 True)
            options.post_hoc:     True=Bonferroni 사후 검정 (기본 True)
            options.effect_size:  True=η² 포함 (기본 True)
            confidence_level:     신뢰수준 (기본 0.95)
            missing_policy:       결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. Descriptive Statistics (요인 수준별)
            3. Levene's Test
            4. 동질적 회귀 계수 검정 (요인×공변량 상호작용)
            5. Tests of Between-Subjects Effects (ANCOVA 표)
            6. Estimated Marginal Means
            7. Pairwise Comparisons (Bonferroni)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    alpha = 1 - confidence_level
    missing_policy_str = spec.get("missing_policy", "listwise")

    dep_var: str = variables.get("dependent", "")
    factor: str = variables.get("factor", "")
    covariates: list[str] = variables.get("covariates", [])

    do_homogeneity: bool = options.get("homogeneity_test", True)
    do_emm: bool = options.get("emm", True)
    do_post_hoc: bool = options.get("post_hoc", True)
    do_effect_size: bool = options.get("effect_size", True)

    result = AnalysisResult(id="ancova", title="ANCOVA")

    # ── 입력 검증 ─────────────────────────────────────────────────
    if not dep_var or not factor:
        result.warnings.append("종속변수(dependent)와 요인(factor)을 지정해야 합니다.")
        return result
    if not covariates:
        result.warnings.append("공변량(covariates)이 없으면 일원분산분석(One-Way ANOVA)을 사용하세요.")
        return result
    if len(covariates) > 3:
        result.warnings.append("공변량은 최대 3개까지 지원합니다.")
        covariates = covariates[:3]

    all_vars = [dep_var, factor] + covariates
    missing_cols = [c for c in all_vars if c not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────
    try:
        mp = MissingPolicy(missing_policy_str)
    except ValueError:
        mp = MissingPolicy.LISTWISE

    paf = prepare_analysis_frame(dataset, all_vars, missing_policy=mp)
    data = paf.data.copy()
    n_before, n_after, n_excluded = paf.n_total, paf.n_valid, paf.n_excluded

    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    if n_after < len(covariates) + 4:
        result.warnings.append(f"유효한 케이스가 너무 적습니다 (최소 {len(covariates)+4}건 필요).")
        return result

    # 타입 변환
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")
    for cov in covariates:
        data[cov] = pd.to_numeric(data[cov], errors="coerce")
    data = data.dropna(subset=[dep_var] + covariates)
    data[factor] = data[factor].astype(str)
    levels = sorted(data[factor].unique())

    if len(levels) < 2:
        result.warnings.append(f"요인('{factor}')의 수준이 1개뿐입니다.")
        return result

    # safe 변수명 (formula 파싱 안전)
    dep_s = "dep_var"
    fac_s = "factor_var"
    cov_s = [f"cov_{i}" for i in range(len(covariates))]
    rename_map = {dep_var: dep_s, factor: fac_s}
    rename_map.update({c: cs for c, cs in zip(covariates, cov_s)})
    dm = data.rename(columns=rename_map)

    # ── Table 2: 기술통계 ─────────────────────────────────────────
    desc_rows = []
    for lv in levels:
        grp = data[data[factor] == lv][dep_var]
        desc_rows.append({
            factor: lv,
            "N": len(grp),
            "평균": format_number(grp.mean(), 4),
            "표준편차": format_number(grp.std(ddof=1), 4) if len(grp) > 1 else "-",
        })
    total = data[dep_var]
    desc_rows.append({
        factor: "전체",
        "N": len(total),
        "평균": format_number(total.mean(), 4),
        "표준편차": format_number(total.std(ddof=1), 4),
    })
    result.tables.append(ResultTable(title="Descriptive Statistics", dataframe=pd.DataFrame(desc_rows)))

    # ── Table 3: Levene 등분산 검정 ───────────────────────────────
    cell_groups = [data[data[factor] == lv][dep_var].values for lv in levels]
    cell_groups = [g for g in cell_groups if len(g) > 0]
    if len(cell_groups) >= 2:
        try:
            lev_stat, lev_p = stats.levene(*cell_groups, center="mean")
            lev_df = pd.DataFrame({
                "Levene 통계량": [format_number(lev_stat, 4)],
                "df1": [len(levels) - 1],
                "df2": [n_after - len(levels)],
                "p-value": [format_pvalue(lev_p)],
            })
            result.tables.append(ResultTable(title="Levene's Test of Equality of Error Variances", dataframe=lev_df))
        except Exception as exc:
            result.warnings.append(f"Levene 검정 오류: {exc}")

    # ── Table 4: 동질적 회귀 계수 가정 검정 ──────────────────────
    if do_homogeneity:
        cov_terms = " + ".join(cov_s)
        interaction_terms = " + ".join(f"C({fac_s}, Sum):{cs}" for cs in cov_s)
        homog_formula = (
            f"{dep_s} ~ C({fac_s}, Sum) + {cov_terms} + {interaction_terms}"
        )
        try:
            lm_homog = ols(homog_formula, data=dm).fit()
            at_homog = sm.stats.anova_lm(lm_homog, typ=3)
            homog_rows = []
            for cs, cov_name in zip(cov_s, covariates):
                int_key = f"C({fac_s}, Sum):{cs}"
                if int_key in at_homog.index:
                    row = at_homog.loc[int_key]
                    f_val = float(row.get("F", np.nan))
                    p_val = float(row.get("PR(>F)", np.nan))
                    homog_rows.append({
                        "소스": f"요인 × {cov_name}",
                        "F": format_number(f_val, 4),
                        "df1": int(row["df"]),
                        "df2": int(at_homog.loc["Residual", "df"]) if "Residual" in at_homog.index else "-",
                        "p-value": format_pvalue(p_val),
                        "판정": "위반" if p_val < 0.05 else "충족",
                    })
                    if p_val < 0.05:
                        result.warnings.append(
                            f"동질적 회귀 계수 가정 위반: 요인 × {cov_name} 상호작용이 유의합니다 "
                            f"(p={format_pvalue(p_val)}). ANCOVA 결과 해석에 주의하세요."
                        )
            if homog_rows:
                result.tables.append(ResultTable(
                    title="Test of Homogeneity of Regression Slopes",
                    dataframe=pd.DataFrame(homog_rows),
                ))
        except Exception as exc:
            result.warnings.append(f"동질적 회귀 계수 검정 오류: {exc}")

    # ── Table 5: ANCOVA ──────────────────────────────────────────
    cov_terms = " + ".join(cov_s)
    ancova_formula = f"{dep_s} ~ C({fac_s}, Sum) + {cov_terms}"
    try:
        lm = ols(ancova_formula, data=dm).fit()
        at = sm.stats.anova_lm(lm, typ=3)
    except Exception as exc:
        result.warnings.append(f"ANCOVA 모델 오류: {exc}")
        return result

    N = len(dm)
    ss_total = float(dm[dep_s].var(ddof=1) * (N - 1))
    ss_error = float(at.loc["Residual", "sum_sq"]) if "Residual" in at.index else np.nan
    df_err_val = int(at.loc["Residual", "df"]) if "Residual" in at.index else N - len(levels) - len(covariates) - 1

    ancova_rows = []
    source_map = {
        f"C({fac_s}, Sum)": factor,
        "Residual": "오차 (Error)",
    }
    for cs, cov_name in zip(cov_s, covariates):
        source_map[cs] = cov_name

    display_order = [f"C({fac_s}, Sum)"] + cov_s + ["Residual"]
    for src_key in display_order:
        if src_key not in at.index:
            continue
        src_label = source_map.get(src_key, src_key)
        row = at.loc[src_key]
        ss = float(row["sum_sq"])
        df_v = int(row["df"])
        ms = ss / df_v if df_v > 0 else np.nan
        f_val = row.get("F", np.nan)
        p_val = row.get("PR(>F)", np.nan)

        arow: dict = {
            "소스": src_label,
            "SS": format_number(ss, 4),
            "df": df_v,
            "MS": format_number(ms, 4),
            "F": format_number(float(f_val), 4) if not np.isnan(f_val) else "",
            "p-value": format_pvalue(float(p_val)) if not np.isnan(p_val) else "",
        }
        if do_effect_size and not np.isnan(f_val):
            # SPSS GLM은 편 η² (Partial Eta Squared)를 기본 출력
            partial_eta2 = ss / (ss + ss_error) if (not np.isnan(ss_error) and ss_error > 0) else ss / ss_total
            arow["편 η²"] = format_number(partial_eta2, 4)
        ancova_rows.append(arow)

    ancova_rows.append({
        "소스": "수정 합계 (Corrected Total)",
        "SS": format_number(ss_total, 4),
        "df": N - 1,
        "MS": "",
        "F": "",
        "p-value": "",
        **({"편 η²": ""} if do_effect_size else {}),
    })
    result.tables.append(ResultTable(
        title="Tests of Between-Subjects Effects",
        dataframe=pd.DataFrame(ancova_rows),
        footnotes=["편 η² (Partial Eta Squared) = SS_효과 / (SS_효과 + SS_오차). SPSS GLM 기본 출력과 동일."],
    ))

    # ── Table 6: 조정된 주변 평균 (EMM) ─────────────────────────
    if do_emm:
        # 공변량 전체 평균으로 고정
        cov_means = {cs: float(dm[cs].mean()) for cs in cov_s}
        emm_rows = []
        for lv in levels:
            # EMM = 추정 평균 (공변량을 전체 평균으로 고정)
            predict_row = {fac_s: lv}
            predict_row.update(cov_means)
            pred_df = pd.DataFrame([predict_row])
            try:
                pred = lm.predict(pred_df)
                emm = float(pred.iloc[0])
                # 95% CI
                pred_summary = lm.get_prediction(pred_df)
                ci = pred_summary.conf_int(alpha=alpha)
                ci_lo = float(ci[0][0])
                ci_hi = float(ci[0][1])
            except Exception:
                emm = np.nan
                ci_lo = ci_hi = np.nan
            emm_rows.append({
                factor: lv,
                "조정된 평균": format_number(emm, 4),
                f"CI 하한 ({int(confidence_level*100)}%)": format_number(ci_lo, 4),
                f"CI 상한 ({int(confidence_level*100)}%)": format_number(ci_hi, 4),
            })
        result.tables.append(ResultTable(
            title="Estimated Marginal Means",
            dataframe=pd.DataFrame(emm_rows),
        ))

    # ── Table 7: 사후 검정 — Bonferroni (조정 평균 기반) ─────────
    if do_post_hoc and len(levels) >= 3:
        from itertools import combinations
        n_pairs = len(levels) * (len(levels) - 1) // 2
        pair_rows = []
        for la, lb in combinations(levels, 2):
            # 잔차 표준오차 기반 비교 (lm.resid 사용)
            n_a = int((data[factor] == la).sum())
            n_b = int((data[factor] == lb).sum())
            ms_err = float(lm.mse_resid)
            se = np.sqrt(ms_err * (1 / n_a + 1 / n_b)) if (n_a > 0 and n_b > 0) else np.nan

            # 조정 평균 차이
            emm_a = emm_b = np.nan
            if do_emm:
                ea = [r for r in emm_rows if r[factor] == la]
                eb = [r for r in emm_rows if r[factor] == lb]
                if ea and eb:
                    try:
                        emm_a = float(str(ea[0]["조정된 평균"]).replace(",", ""))
                        emm_b = float(str(eb[0]["조정된 평균"]).replace(",", ""))
                    except ValueError:
                        pass

            diff = emm_a - emm_b if not (np.isnan(emm_a) or np.isnan(emm_b)) else np.nan
            if np.isnan(diff) and np.isnan(emm_a) and np.isnan(emm_b):
                continue

            if not np.isnan(se) and se > 0:
                t_stat = diff / se if not np.isnan(diff) else np.nan
                p_raw = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_err_val))) if not np.isnan(t_stat) else np.nan
                p_adj = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan
                t_crit = stats.t.ppf(1 - alpha / (2 * n_pairs), df=df_err_val)
                ci_lo_p = diff - t_crit * se if not np.isnan(diff) else np.nan
                ci_hi_p = diff + t_crit * se if not np.isnan(diff) else np.nan
            else:
                t_stat = p_raw = p_adj = ci_lo_p = ci_hi_p = np.nan

            pair_rows.append({
                f"{factor} (I)": la,
                f"{factor} (J)": lb,
                "조정 평균차 (I-J)": format_number(diff, 4),
                "표준오차": format_number(se, 4),
                "p-value": format_pvalue(p_raw) if not np.isnan(p_raw) else "-",
                "p-adj (Bonferroni)": format_pvalue(p_adj) if not np.isnan(p_adj) else "-",
                "CI 하한": format_number(ci_lo_p, 4),
                "CI 상한": format_number(ci_hi_p, 4),
            })
        if pair_rows:
            result.tables.append(ResultTable(
                title="Pairwise Comparisons (Bonferroni)",
                dataframe=pd.DataFrame(pair_rows),
            ))

    # ── 해석 메모 ────────────────────────────────────────────────
    fac_key = f"C({fac_s}, Sum)"
    if fac_key in at.index:
        f_fac = float(at.loc[fac_key, "F"]) if not np.isnan(at.loc[fac_key, "F"]) else np.nan
        p_fac = float(at.loc[fac_key, "PR(>F)"]) if not np.isnan(at.loc[fac_key, "PR(>F)"]) else np.nan
        if not np.isnan(f_fac):
            result.notes.append(
                f"[{factor}] F = {format_number(f_fac, 3)}, "
                f"p = {format_pvalue(p_fac)} (공변량 조정 후)"
            )
    for cs, cov_name in zip(cov_s, covariates):
        if cs in at.index:
            f_c = float(at.loc[cs, "F"]) if not np.isnan(at.loc[cs, "F"]) else np.nan
            p_c = float(at.loc[cs, "PR(>F)"]) if not np.isnan(at.loc[cs, "PR(>F)"]) else np.nan
            if not np.isnan(f_c):
                result.notes.append(
                    f"[공변량: {cov_name}] F = {format_number(f_c, 3)}, p = {format_pvalue(p_c)}"
                )

    return result
