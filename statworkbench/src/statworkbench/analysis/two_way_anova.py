"""이원분산분석(Two-Way ANOVA / Factorial ANOVA) 분석 모듈.

SPSS: Analyze > General Linear Model > Univariate (2개 요인)

지원 기능:
  - 주 효과(Main Effects): 요인 A, 요인 B
  - 상호작용 효과(Interaction): A×B
  - 기술통계 (셀 평균, 표준편차)
  - Levene 등분산 검정
  - 효과 크기: η² (Eta-squared), ω² (Omega-squared)
  - 사후 검정: Tukey HSD (각 주 효과별)
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statworkbench.analysis.assumptions import get_cps_table_kr, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """이원분산분석(Two-Way Factorial ANOVA)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.dependent: 종속변수 (연속형)
            variables.factor_a: 요인 A (범주형)
            variables.factor_b: 요인 B (범주형)
            options.post_hoc: True=Tukey HSD 사후 검정 수행 (기본 True)
            options.effect_size: True=η², ω² 포함 (기본 True)
            confidence_level: 신뢰수준 (기본 0.95)
            missing_policy: 결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. Descriptive Statistics (셀별)
            3. Levene's Test
            4. Tests of Between-Subjects Effects (ANOVA 표)
            5. Post-Hoc Tests (Tukey HSD, 주 효과별)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", "listwise")

    dep_var: str = variables.get("dependent", "")
    factor_a: str = variables.get("factor_a", "")
    factor_b: str = variables.get("factor_b", "")
    do_post_hoc: bool = options.get("post_hoc", True)
    post_hoc_method: str = options.get("post_hoc_method", "tukey").lower()  # tukey | scheffe | bonferroni | lsd
    do_effect_size: bool = options.get("effect_size", True)
    do_profile_plot: bool = options.get("profile_plot", True)

    result = AnalysisResult(id="two_way_anova", title="Two-Way ANOVA")

    # ── 입력 검증 ─────────────────────────────────────────────────
    if not dep_var or not factor_a or not factor_b:
        result.warnings.append("종속변수(dependent), 요인A(factor_a), 요인B(factor_b)를 모두 지정해야 합니다.")
        return result
    if factor_a == factor_b:
        result.warnings.append(f"요인 A와 요인 B에 동일한 변수('{factor_a}')가 지정되었습니다. 서로 다른 변수를 선택하세요.")
        return result

    needed = [dep_var, factor_a, factor_b]
    missing_cols = [c for c in needed if c not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────
    if isinstance(missing_policy_str, str):
        try:
            mp = MissingPolicy(missing_policy_str)
        except ValueError:
            mp = MissingPolicy.LISTWISE
    else:
        mp = missing_policy_str

    paf = prepare_analysis_frame(dataset, needed, missing_policy=mp)
    data = paf.data
    n_before = paf.n_total
    n_after = paf.n_valid
    n_excluded = paf.n_excluded

    result.tables.append(get_cps_table_kr(n_before, n_after, n_excluded))

    if n_after < 4:
        result.warnings.append("유효한 케이스가 너무 적습니다 (최소 4건 필요).")
        return result

    # 요인 변환
    data = data.copy()
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")
    data = data.dropna(subset=[dep_var])
    data[factor_a] = data[factor_a].astype(str)
    data[factor_b] = data[factor_b].astype(str)

    levels_a = sorted(data[factor_a].unique())
    levels_b = sorted(data[factor_b].unique())

    if len(levels_a) < 2:
        result.warnings.append(f"요인 A('{factor_a}')의 수준이 1개뿐입니다. 검정을 수행할 수 없습니다.")
        return result
    if len(levels_b) < 2:
        result.warnings.append(f"요인 B('{factor_b}')의 수준이 1개뿐입니다. 검정을 수행할 수 없습니다.")
        return result

    # 빈 셀 또는 단일 관측치 셀 경고
    empty_cells, singleton_cells = [], []
    for la in levels_a:
        for lb in levels_b:
            n_cell = int(((data[factor_a] == la) & (data[factor_b] == lb)).sum())
            if n_cell == 0:
                empty_cells.append(f"({la}, {lb})")
            elif n_cell == 1:
                singleton_cells.append(f"({la}, {lb})")
    if empty_cells:
        result.warnings.append(
            f"다음 셀에 관측치가 없습니다 (불균형 설계): {', '.join(empty_cells)}. "
            "Type III SS 결과가 부정확할 수 있습니다."
        )
    if singleton_cells:
        result.warnings.append(
            f"다음 셀의 관측치가 1건입니다: {', '.join(singleton_cells)}. "
            "분산 추정 및 등분산 검정이 신뢰할 수 없습니다."
        )

    # ── Table 2: 기술통계 (셀별) ──────────────────────────────────
    desc_rows = []
    all_groups = []
    for la in levels_a:
        for lb in levels_b:
            cell = data[(data[factor_a] == la) & (data[factor_b] == lb)][dep_var]
            n_cell = len(cell)
            all_groups.append((la, lb, cell.values))
            desc_rows.append({
                factor_a: la,
                factor_b: lb,
                "N": n_cell,
                "평균": format_number(cell.mean(), 4) if n_cell > 0 else "-",
                "표준편차": format_number(cell.std(ddof=1), 4) if n_cell > 1 else "-",
                "최솟값": format_number(cell.min(), 4) if n_cell > 0 else "-",
                "최댓값": format_number(cell.max(), 4) if n_cell > 0 else "-",
            })
    # 주변 합계 (요인 A 수준별)
    for la in levels_a:
        grp = data[data[factor_a] == la][dep_var]
        desc_rows.append({
            factor_a: la,
            factor_b: "전체",
            "N": len(grp),
            "평균": format_number(grp.mean(), 4),
            "표준편차": format_number(grp.std(ddof=1), 4) if len(grp) > 1 else "-",
            "최솟값": format_number(grp.min(), 4),
            "최댓값": format_number(grp.max(), 4),
        })
    # 전체
    total = data[dep_var]
    desc_rows.append({
        factor_a: "전체",
        factor_b: "전체",
        "N": len(total),
        "평균": format_number(total.mean(), 4),
        "표준편차": format_number(total.std(ddof=1), 4),
        "최솟값": format_number(total.min(), 4),
        "최댓값": format_number(total.max(), 4),
    })
    result.tables.append(ResultTable(
        title="Descriptive Statistics",
        dataframe=pd.DataFrame(desc_rows),
    ))

    # ── Table 3: Levene 등분산 검정 ───────────────────────────────
    cell_groups = [g[2] for g in all_groups if len(g[2]) > 0]
    if len(cell_groups) >= 2:
        try:
            lev_stat, lev_p = stats.levene(*cell_groups, center="mean")
            lev_df1 = len(cell_groups) - 1
            lev_df2 = sum(len(g) for g in cell_groups) - len(cell_groups)
            lev_df = pd.DataFrame({
                "Levene 통계량": [format_number(lev_stat, 4)],
                "df1": [lev_df1],
                "df2": [lev_df2],
                "p-value": [format_pvalue(lev_p)],
            })
            result.tables.append(ResultTable(title="Levene's Test of Equality of Error Variances", dataframe=lev_df))
        except Exception as exc:
            result.warnings.append(f"Levene 검정 오류: {exc}")

    # ── Table 4: ANOVA (statsmodels OLS) ─────────────────────────
    # safe variable names for formula
    dep_safe = "dep_var"
    fa_safe = "factor_a"
    fb_safe = "factor_b"
    df_model = data.rename(columns={dep_var: dep_safe, factor_a: fa_safe, factor_b: fb_safe})

    # Sum contrasts = SPSS-호환 Type III SS (편차 코딩으로 주효과가 올바르게 분해됨)
    formula = (
        f"{dep_safe} ~ C({fa_safe}, Sum) + C({fb_safe}, Sum) "
        f"+ C({fa_safe}, Sum):C({fb_safe}, Sum)"
    )
    try:
        lm = ols(formula, data=df_model).fit()
        anova_tbl = sm.stats.anova_lm(lm, typ=3)
    except Exception as exc:
        result.warnings.append(f"ANOVA 모델 오류: {exc}")
        return result

    N = len(data)
    ss_total = float(data[dep_var].var(ddof=1) * (N - 1))
    ss_error = float(anova_tbl.loc["Residual", "sum_sq"]) if "Residual" in anova_tbl.index else np.nan

    anova_rows = []
    source_map = {
        f"C({fa_safe}, Sum)": factor_a,
        f"C({fb_safe}, Sum)": factor_b,
        f"C({fa_safe}, Sum):C({fb_safe}, Sum)": f"{factor_a} × {factor_b}",
        "Residual": "오차 (Error)",
    }
    for src_key, src_label in source_map.items():
        if src_key not in anova_tbl.index:
            continue
        row = anova_tbl.loc[src_key]
        ss = float(row["sum_sq"])
        df_val = int(row["df"])
        ms = ss / df_val if df_val > 0 else np.nan
        f_val = row.get("F", np.nan)
        p_val = row.get("PR(>F)", np.nan)

        if np.isnan(f_val) or src_key == "Residual":
            anova_row: dict = {
                "소스": src_label,
                "SS": format_number(ss, 4),
                "df": df_val,
                "MS": format_number(ms, 4),
                "F": "",
                "p-value": "",
            }
        else:
            anova_row = {
                "소스": src_label,
                "SS": format_number(ss, 4),
                "df": df_val,
                "MS": format_number(ms, 4),
                "F": format_number(float(f_val), 4),
                "p-value": format_pvalue(float(p_val)),
            }
            if do_effect_size:
                # SPSS GLM은 편 η² (Partial Eta Squared)를 기본 출력
                denom_eta = ss + ss_error
                partial_eta2 = float(ss / denom_eta) if (not np.isnan(ss_error) and ss_error > 0 and denom_eta > 0) else (float(ss / ss_total) if ss_total > 0 else float("nan"))
                anova_row["편 η²"] = format_number(partial_eta2, 4)

        anova_rows.append(anova_row)

    # 수정 합계
    anova_rows.append({
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
        dataframe=pd.DataFrame(anova_rows),
        footnotes=["편 η² (Partial Eta Squared) = SS_효과 / (SS_효과 + SS_오차). SPSS GLM 기본 출력과 동일."],
    ))

    # ── Table 5: 사후 검정 (요인별) ──────────────────────────────
    if do_post_hoc:
        for factor, levels in [(factor_a, levels_a), (factor_b, levels_b)]:
            if len(levels) < 3:
                continue
            try:
                _run_post_hoc(
                    result=result,
                    data=data,
                    dep_var=dep_var,
                    factor=factor,
                    levels=levels,
                    method=post_hoc_method,
                    confidence_level=confidence_level,
                    ss_error=ss_error,
                    df_error=int(anova_tbl.loc["Residual", "df"]) if "Residual" in anova_tbl.index else None,
                )
            except Exception as exc:
                result.warnings.append(f"사후 검정 ({factor}) 오류: {exc}")

    # ── 프로파일 플롯 (상호작용 그래프) ──────────────────────────────
    if do_profile_plot:
        plot_bytes = _profile_plot_two_way(data, dep_var, factor_a, factor_b)
        if plot_bytes:
            result.add_table(ResultTable(
                title="프로파일 플롯 (상호작용 그래프)",
                dataframe=pd.DataFrame([{"image_bytes": plot_bytes}]),
                metadata={"type": "profile_plot"},
            ))

    # ── 해석 메모 ─────────────────────────────────────────────────
    for src_key, src_label in source_map.items():
        if src_key not in anova_tbl.index or src_key == "Residual":
            continue
        row = anova_tbl.loc[src_key]
        f_val = row.get("F", np.nan)
        p_val = row.get("PR(>F)", np.nan)
        if not np.isnan(f_val):
            result.notes.append(
                f"[{src_label}] F = {format_number(float(f_val), 3)}, "
                f"p = {format_pvalue(float(p_val))}"
            )

    return result


def _run_post_hoc(
    result: AnalysisResult,
    data: pd.DataFrame,
    dep_var: str,
    factor: str,
    levels: list,
    method: str,
    confidence_level: float,
    ss_error: float,
    df_error: int | None,
) -> None:
    """요인별 사후 검정을 수행하고 ResultTable을 result에 추가합니다.

    method: "tukey" | "scheffe" | "bonferroni" | "lsd"
    """
    from itertools import combinations

    alpha = 1 - confidence_level
    n_pairs = len(levels) * (len(levels) - 1) // 2
    ms_error = ss_error / df_error if (df_error and df_error > 0) else np.nan

    if method == "tukey":
        tukey = pairwise_tukeyhsd(
            endog=data[dep_var].values,
            groups=data[factor].values,
            alpha=alpha,
        )
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
        tukey_df.columns = ["집단1", "집단2", "평균차", "p-adj", "CI 하한", "CI 상한", "유의"]
        for col in ["평균차", "CI 하한", "CI 상한"]:
            tukey_df[col] = tukey_df[col].apply(lambda v: format_number(float(v), 4))
        tukey_df["p-adj"] = tukey_df["p-adj"].apply(lambda v: format_pvalue(float(v)))
        result.tables.append(ResultTable(
            title=f"Post-Hoc: Tukey HSD — {factor}",
            dataframe=tukey_df,
            footnotes=["Tukey HSD: 집단 간 등분산 가정. p-adj는 Tukey 분포 기반 보정값."],
        ))
        return

    # Scheffe / Bonferroni / LSD (공통 프레임워크)
    rows = []
    label = {"scheffe": "Scheffe", "bonferroni": "Bonferroni", "lsd": "LSD"}.get(method, method)
    for la, lb in combinations(levels, 2):
        grp_a = data[data[factor] == la][dep_var].dropna()
        grp_b = data[data[factor] == lb][dep_var].dropna()
        na, nb = len(grp_a), len(grp_b)
        if na < 1 or nb < 1 or np.isnan(ms_error):
            continue
        mean_diff = float(grp_a.mean() - grp_b.mean())
        se = np.sqrt(ms_error * (1 / na + 1 / nb))
        t_stat = mean_diff / se if se > 0 else np.nan

        if method == "scheffe":
            k = len(levels)
            f_stat = (t_stat ** 2) / (k - 1) if not np.isnan(t_stat) else np.nan
            p_raw = float(1 - stats.f.cdf(f_stat, dfn=k - 1, dfd=df_error)) if not np.isnan(f_stat) else np.nan
            f_crit = stats.f.ppf(1 - alpha, dfn=k - 1, dfd=df_error)
            ci_half = np.sqrt((k - 1) * f_crit * ms_error * (1 / na + 1 / nb))
        elif method == "bonferroni":
            p_raw = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_error))) if not np.isnan(t_stat) else np.nan
            p_raw = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan
            t_crit = stats.t.ppf(1 - alpha / (2 * n_pairs), df=df_error)
            ci_half = t_crit * se
        else:  # lsd
            p_raw = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_error))) if not np.isnan(t_stat) else np.nan
            t_crit = stats.t.ppf(1 - alpha / 2, df=df_error)
            ci_half = t_crit * se

        rows.append({
            "집단1": la,
            "집단2": lb,
            "평균차 (I-J)": format_number(mean_diff, 4),
            "표준오차": format_number(se, 4),
            "p-value": format_pvalue(p_raw) if not np.isnan(p_raw) else "-",
            f"CI 하한 ({int(confidence_level*100)}%)": format_number(mean_diff - ci_half, 4),
            f"CI 상한 ({int(confidence_level*100)}%)": format_number(mean_diff + ci_half, 4),
        })

    if rows:
        result.tables.append(ResultTable(
            title=f"Post-Hoc: {label} — {factor}",
            dataframe=pd.DataFrame(rows),
        ))


def _profile_plot_two_way(
    df: pd.DataFrame,
    dep_var: str,
    factor_a: str,
    factor_b: str,
) -> bytes | None:
    """Factor A × Factor B 셀 평균 상호작용 선 그래프 생성 후 PNG bytes 반환."""
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cell_means = (
            df.groupby([factor_a, factor_b])[dep_var]
            .mean()
            .reset_index()
        )
        levels_b = sorted(cell_means[factor_b].unique(), key=str)
        levels_a = sorted(cell_means[factor_a].unique(), key=str)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = plt.cm.tab10.colors
        for i, lvl_b in enumerate(levels_b):
            sub = cell_means[cell_means[factor_b] == lvl_b].set_index(factor_a)
            y_vals = [float(sub.loc[la, dep_var]) if la in sub.index else float("nan") for la in levels_a]
            ax.plot([str(la) for la in levels_a], y_vals, marker="o",
                    label=str(lvl_b), color=colors[i % len(colors)])

        ax.set_xlabel(factor_a)
        ax.set_ylabel(dep_var)
        ax.set_title(f"프로파일 플롯: {dep_var}")
        ax.legend(title=factor_b)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("프로파일 플롯 생성 실패: %s", e)
        return None
