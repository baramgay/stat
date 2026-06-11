"""혼합 분산분석(Mixed ANOVA — Split-Plot Design) 분석 모듈.

SPSS: Analyze > General Linear Model > Repeated Measures (집단 간 요인 포함)

지원 기능:
  - 집단 간 요인(between-subjects factor) 1개
  - 집단 내 요인(within-subjects factor) 1개, 측정 시점 2~10개
  - Mauchly 구형성 검정 + Greenhouse-Geisser / Huynh-Feldt 보정
  - 집단 간 효과, 집단 내 효과, 상호작용 효과
  - 편 η² (Partial Eta Squared)
  - Bonferroni 사후 검정 (집단 간, 시점 간)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.assumptions import get_cps_table_kr, prepare_analysis_frame
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """혼합 분산분석(Mixed ANOVA)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.between:    집단 간 요인 변수명 (범주형)
            variables.within:     집단 내 측정 변수 목록 (예: ["T1","T2","T3"])
            variables.within_name: 시점 요인 이름 (기본 "시점")
            options.sphericity:   True=구형성 검정 포함 (기본 True)
            options.post_hoc:     True=Bonferroni 사후 검정 (기본 True)
            options.effect_size:  True=편 η² 포함 (기본 True)
            confidence_level:     신뢰수준 (기본 0.95)
            missing_policy:       결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. Descriptive Statistics (집단 × 시점)
            3. Mauchly 구형성 검정 (시점 ≥ 3)
            4. Tests of Within-Subjects Effects (집단 내 + 상호작용)
            5. Tests of Between-Subjects Effects (집단 간)
            6. Pairwise Comparisons (Bonferroni)
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)
    alpha = 1 - confidence_level

    between_var: str = variables.get("between", "")
    within_vars: list[str] = variables.get("within", [])
    within_name: str = variables.get("within_name", "시점")

    do_sphericity: bool = options.get("sphericity", True)
    do_post_hoc: bool = options.get("post_hoc", True)
    do_effect_size: bool = options.get("effect_size", True)
    do_profile_plot: bool = options.get("profile_plot", True)

    result = AnalysisResult(id="mixed_anova", title="혼합 분산분석 (Mixed ANOVA)")

    # ── 입력 검증 ─────────────────────────────────────────────────
    if not between_var:
        result.warnings.append("집단 간 요인(between)을 지정해야 합니다.")
        return result
    if len(within_vars) < 2:
        result.warnings.append("집단 내 측정 변수(within)를 2개 이상 지정해야 합니다.")
        return result
    if len(within_vars) > 10:
        result.warnings.append("집단 내 측정 변수는 최대 10개까지 지원합니다.")
        within_vars = within_vars[:10]

    all_needed = [between_var] + within_vars
    missing_cols = [c for c in all_needed if c not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────
    paf = prepare_analysis_frame(dataset, all_needed, missing_policy=missing_policy)
    data = paf.data.copy()
    result.tables.append(get_cps_table_kr(paf.n_total, paf.n_valid, paf.n_excluded))

    if paf.n_valid < 4:
        result.warnings.append("유효 케이스가 너무 적습니다 (최소 4건 필요).")
        return result

    data[between_var] = data[between_var].astype(str)
    for wv in within_vars:
        data[wv] = pd.to_numeric(data[wv], errors="coerce")
    data = data.dropna(subset=within_vars)

    groups = sorted(data[between_var].unique())
    n_groups = len(groups)
    k = len(within_vars)
    N = len(data)

    if n_groups < 2:
        result.warnings.append(f"집단 간 요인('{between_var}')의 수준이 1개뿐입니다.")
        return result

    # ── Table 2: 기술통계 ─────────────────────────────────────────
    desc_rows = []
    for grp in groups:
        grp_data = data[data[between_var] == grp]
        for wv in within_vars:
            vals = grp_data[wv].dropna()
            desc_rows.append({
                between_var: grp,
                within_name: wv,
                "N": len(vals),
                "평균": format_number(vals.mean(), 4),
                "표준편차": format_number(vals.std(ddof=1), 4) if len(vals) > 1 else "-",
            })
    result.tables.append(ResultTable(title="Descriptive Statistics", dataframe=pd.DataFrame(desc_rows)))

    # ── 핵심 계산: 혼합 ANOVA (SPSS split-plot 공식) ─────────────
    # 각 피험자별 within-subjects 행렬 구성
    Y = data[within_vars].values.astype(float)  # (N, k)
    group_labels = data[between_var].values

    # 집단 크기
    group_ns = {g: int((group_labels == g).sum()) for g in groups}
    group_means = {g: Y[group_labels == g, :].mean(axis=0) for g in groups}  # (k,)
    grand_mean_by_time = Y.mean(axis=0)  # (k,)
    grand_mean = Y.mean()

    # ── SS 계산 ────────────────────────────────────────────────────
    # SS_between (집단 간): k * Σ_g n_g * (mean_g - grand)²
    ss_between = float(k * sum(
        group_ns[g] * (Y[group_labels == g, :].mean() - grand_mean) ** 2
        for g in groups
    ))
    df_between = n_groups - 1

    # SS_subjects_within_groups (피험자 간, 집단 내 오차)
    subj_means = Y.mean(axis=1)  # (N,)
    ss_s_within = float(k * sum(
        float(np.sum((subj_means[group_labels == g] - float(Y[group_labels == g, :].mean())) ** 2))
        for g in groups
    ))
    df_s_within = N - n_groups

    # SS_within (집단 내 — 시점 효과)
    ss_within = float(N * sum((grand_mean_by_time[t] - grand_mean) ** 2 for t in range(k)))
    df_within = k - 1

    # SS_interaction (집단 × 시점)
    ss_interaction = float(sum(
        group_ns[g] * sum(
            (group_means[g][t] - grand_mean_by_time[t] - Y[group_labels == g, :].mean() + grand_mean) ** 2
            for t in range(k)
        )
        for g in groups
    ))
    df_interaction = (n_groups - 1) * (k - 1)

    # SS_error_within (집단 내 오차 — 시점 × 피험자)
    ss_error_within = float(sum(
        (Y[i, t] - group_means[group_labels[i]][t] - subj_means[i] + Y[group_labels == group_labels[i], :].mean()) ** 2
        for i in range(N)
        for t in range(k)
    ))
    df_error_within = (N - n_groups) * (k - 1)

    # ── Table 3: Mauchly 구형성 검정 ─────────────────────────────
    epsilon_gg, epsilon_hf = 1.0, 1.0
    if k >= 3 and do_sphericity:
        epsilon_gg, epsilon_hf = _mauchly_test(result, Y, group_labels, groups, k, N)

    # ── Table 4: Tests of Within-Subjects Effects ─────────────────
    ss_total = float(np.sum((Y - grand_mean) ** 2))
    within_rows = _within_subjects_table(
        ss_within=ss_within, df_within=df_within,
        ss_interaction=ss_interaction, df_interaction=df_interaction,
        ss_error_within=ss_error_within, df_error_within=df_error_within,
        epsilon_gg=epsilon_gg, epsilon_hf=epsilon_hf,
        within_name=within_name, between_var=between_var,
        ss_total=ss_total, do_effect_size=do_effect_size,
    )
    result.tables.append(ResultTable(
        title="Tests of Within-Subjects Effects",
        dataframe=pd.DataFrame(within_rows),
        footnotes=["편 η² = SS_효과 / (SS_효과 + SS_오차_within). GG/HF: 구형성 위반 시 보정값."],
    ))

    # ── Table 5: Tests of Between-Subjects Effects ────────────────
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_s_within = ss_s_within / df_s_within if df_s_within > 0 else np.nan
    f_between = ms_between / ms_s_within if (not np.isnan(ms_s_within) if isinstance(ms_s_within, float) else True) and ms_s_within > 1e-15 else np.nan
    p_between = float(1 - stats.f.cdf(f_between, dfn=df_between, dfd=df_s_within)) if not np.isnan(f_between) else np.nan

    between_rows = [
        {
            "소스": between_var,
            "SS": format_number(ss_between, 4),
            "df": df_between,
            "MS": format_number(ms_between, 4),
            "F": format_number(f_between, 4),
            "p-value": format_pvalue(p_between),
            **({"편 η²": format_number(ss_between / (ss_between + ss_s_within), 4)} if do_effect_size else {}),
        },
        {
            "소스": "오차 (피험자 간)",
            "SS": format_number(ss_s_within, 4),
            "df": df_s_within,
            "MS": format_number(ms_s_within, 4),
            "F": "", "p-value": "",
            **({"편 η²": ""} if do_effect_size else {}),
        },
    ]
    result.tables.append(ResultTable(
        title="Tests of Between-Subjects Effects",
        dataframe=pd.DataFrame(between_rows),
        footnotes=["집단 간 오차 = 피험자 간 분산(집단 내 변동)."],
    ))

    # ── Table 6: Bonferroni 사후 검정 ────────────────────────────
    if do_post_hoc:
        _bonferroni_between(result, data, between_var, groups, within_vars, ms_s_within, df_s_within, k, confidence_level, alpha)
        if k >= 3:
            _bonferroni_within(result, Y, within_vars, within_name, ms_error_within=ss_error_within / df_error_within if df_error_within > 0 else np.nan, df_error=df_error_within, N=N, confidence_level=confidence_level, alpha=alpha)

    # ── 프로파일 플롯 ────────────────────────────────────────────
    if do_profile_plot:
        plot_bytes = _profile_plot_mixed(data, between_var, within_vars, within_name)
        if plot_bytes:
            result.add_table(ResultTable(
                title="프로파일 플롯 (집단 × 시점)",
                dataframe=pd.DataFrame([{"image_bytes": plot_bytes}]),
                metadata={"type": "profile_plot"},
            ))

    # ── 해석 메모 ────────────────────────────────────────────────
    if not np.isnan(f_between):
        result.notes.append(f"[집단 간: {between_var}] F({df_between},{df_s_within}) = {format_number(f_between, 3)}, p = {format_pvalue(p_between)}")
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_wn = ms_within / (ss_error_within / df_error_within) if df_error_within > 0 and ss_error_within > 0 else np.nan
    p_wn = float(1 - stats.f.cdf(f_wn, dfn=df_within, dfd=df_error_within)) if not np.isnan(f_wn) else np.nan
    if not np.isnan(f_wn):
        result.notes.append(f"[집단 내: {within_name}] F({df_within},{df_error_within}) = {format_number(f_wn, 3)}, p = {format_pvalue(p_wn)}")

    return result


def _mauchly_test(
    result: AnalysisResult,
    Y: np.ndarray,
    group_labels: np.ndarray,
    groups: list,
    k: int,
    N: int,
) -> tuple[float, float]:
    """Mauchly 구형성 검정 및 Greenhouse-Geisser / Huynh-Feldt epsilon 계산."""
    try:
        from nuristat.analysis.repeated_measures_anova import _mauchly_test
        mres = _mauchly_test(Y)
        W = mres["W"]
        chi2 = mres["chi2"]
        df_m = mres["df"]
        p_w = mres["p"]
        eps_gg = mres["epsilon_gg"]
        eps_hf = mres["epsilon_hf"]
        eps_lb = mres["epsilon_lb"]

        kmo_interp = "위반 (GG/HF 보정 권장)" if (not np.isnan(p_w) and p_w < 0.05) else "충족"
        rows = [
            {"검정": "Mauchly W", "값": format_number(W, 4), "비고": kmo_interp},
            {"검정": "Chi-square", "값": format_number(chi2, 4), "비고": ""},
            {"검정": "df", "값": str(int(df_m)), "비고": ""},
            {"검정": "p-value", "값": format_pvalue(p_w), "비고": ""},
            {"검정": "Greenhouse-Geisser ε", "값": format_number(eps_gg, 4), "비고": ""},
            {"검정": "Huynh-Feldt ε", "값": format_number(eps_hf, 4), "비고": ""},
            {"검정": "Lower-bound ε", "값": format_number(eps_lb, 4), "비고": "1/(k-1)"},
        ]
        result.tables.append(ResultTable(
            title="Mauchly's Test of Sphericity",
            dataframe=pd.DataFrame(rows),
            footnotes=["p < .05이면 구형성 가정 위반 → GG 또는 HF 보정값 사용."],
        ))
        return float(eps_gg), float(eps_hf)
    except Exception as exc:
        result.warnings.append(f"Mauchly 검정 오류: {exc}")
        return 1.0, 1.0


def _within_subjects_table(
    ss_within: float,
    df_within: int,
    ss_interaction: float,
    df_interaction: int,
    ss_error_within: float,
    df_error_within: int,
    epsilon_gg: float,
    epsilon_hf: float,
    within_name: str,
    between_var: str,
    ss_total: float,
    do_effect_size: bool,
) -> list[dict]:
    """집단 내 효과 테이블 행 목록을 반환합니다."""
    ms_error = ss_error_within / df_error_within if df_error_within > 0 else np.nan
    rows = []

    for label, ss, df, eps_gg, eps_hf in [
        (within_name, ss_within, df_within, epsilon_gg, epsilon_hf),
        (f"{within_name} × {between_var}", ss_interaction, df_interaction, epsilon_gg, epsilon_hf),
        ("오차 (집단 내)", ss_error_within, df_error_within, epsilon_gg, epsilon_hf),
    ]:
        ms = ss / df if df > 0 else np.nan
        is_error = "오차" in label
        f_val = ms / ms_error if (not is_error and not np.isnan(ms_error) and ms_error > 0) else np.nan
        p_val = float(1 - stats.f.cdf(f_val, dfn=df, dfd=df_error_within)) if not np.isnan(f_val) else np.nan

        # GG 보정
        df_gg = df * eps_gg
        df_err_gg = df_error_within * eps_gg
        p_gg = float(1 - stats.f.cdf(f_val, dfn=df_gg, dfd=df_err_gg)) if not np.isnan(f_val) else np.nan

        # HF 보정
        df_hf = min(df * eps_hf, df)
        df_err_hf = min(df_error_within * eps_hf, df_error_within)
        p_hf = float(1 - stats.f.cdf(f_val, dfn=df_hf, dfd=df_err_hf)) if not np.isnan(f_val) else np.nan

        row: dict = {
            "소스": label,
            "가정": "구형성 충족",
            "SS": format_number(ss, 4),
            "df": format_number(df, 4) if not is_error else str(df_error_within),
            "MS": format_number(ms, 4),
            "F": format_number(f_val, 4) if not np.isnan(f_val) else "",
            "p-value": format_pvalue(p_val) if not np.isnan(p_val) else "",
        }
        if do_effect_size and not is_error and not np.isnan(f_val):
            row["편 η²"] = format_number(ss / (ss + ss_error_within), 4)
        rows.append(row)

        if not is_error:
            rows.append({
                "소스": label,
                "가정": f"GG (ε={format_number(eps_gg, 3)})",
                "SS": "",
                "df": format_number(df_gg, 3),
                "MS": "",
                "F": format_number(f_val, 4) if not np.isnan(f_val) else "",
                "p-value": format_pvalue(p_gg) if not np.isnan(p_gg) else "",
                **({"편 η²": ""} if do_effect_size else {}),
            })
            rows.append({
                "소스": label,
                "가정": f"HF (ε={format_number(eps_hf, 3)})",
                "SS": "",
                "df": format_number(df_hf, 3),
                "MS": "",
                "F": format_number(f_val, 4) if not np.isnan(f_val) else "",
                "p-value": format_pvalue(p_hf) if not np.isnan(p_hf) else "",
                **({"편 η²": ""} if do_effect_size else {}),
            })

    return rows


def _bonferroni_between(
    result: AnalysisResult,
    data: pd.DataFrame,
    between_var: str,
    groups: list,
    within_vars: list[str],
    ms_s_within: float,
    df_s_within: int,
    k: int,
    confidence_level: float,
    alpha: float,
) -> None:
    """집단 간 Bonferroni 쌍 비교 (collapsed across within-subjects factor)."""
    from itertools import combinations
    if len(groups) < 3:
        return
    n_pairs = len(groups) * (len(groups) - 1) // 2
    rows = []
    for ga, gb in combinations(groups, 2):
        grp_a = data[data[between_var] == ga][within_vars].values.astype(float)
        grp_b = data[data[between_var] == gb][within_vars].values.astype(float)
        mean_a = float(grp_a.mean())
        mean_b = float(grp_b.mean())
        na, nb = len(grp_a), len(grp_b)
        if np.isnan(ms_s_within) or ms_s_within <= 0:
            continue
        se = np.sqrt(ms_s_within / k * (1 / na + 1 / nb))
        diff = mean_a - mean_b
        t_stat = diff / se if se > 0 else np.nan
        p_raw = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_s_within))) if not np.isnan(t_stat) else np.nan
        p_adj = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan
        t_crit = stats.t.ppf(1 - alpha / (2 * n_pairs), df=df_s_within)
        ci_half = t_crit * se
        rows.append({
            f"{between_var} (I)": ga,
            f"{between_var} (J)": gb,
            "평균차 (I-J)": format_number(diff, 4),
            "표준오차": format_number(se, 4),
            "p-adj (Bonferroni)": format_pvalue(p_adj) if not np.isnan(p_adj) else "-",
            f"CI 하한 ({int(confidence_level*100)}%)": format_number(diff - ci_half, 4),
            f"CI 상한 ({int(confidence_level*100)}%)": format_number(diff + ci_half, 4),
        })
    if rows:
        result.tables.append(ResultTable(
            title=f"Pairwise Comparisons: {between_var} (Bonferroni)",
            dataframe=pd.DataFrame(rows),
        ))


def _bonferroni_within(
    result: AnalysisResult,
    Y: np.ndarray,
    within_vars: list[str],
    within_name: str,
    ms_error_within: float,
    df_error: int,
    N: int,
    confidence_level: float,
    alpha: float,
) -> None:
    """시점 간 Bonferroni 쌍 비교 (collapsed across between-subjects factor)."""
    from itertools import combinations
    k = len(within_vars)
    n_pairs = k * (k - 1) // 2
    rows = []
    for i, j in combinations(range(k), 2):
        diff_col = Y[:, i] - Y[:, j]
        mean_diff = float(diff_col.mean())
        se = np.sqrt(ms_error_within * 2 / N) if (ms_error_within > 0 and N > 0) else np.nan
        t_stat = mean_diff / se if (not np.isnan(se) and se > 0) else np.nan
        p_raw = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_error))) if not np.isnan(t_stat) else np.nan
        p_adj = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan
        t_crit = stats.t.ppf(1 - alpha / (2 * n_pairs), df=df_error)
        ci_half = t_crit * se if not np.isnan(se) else np.nan
        rows.append({
            f"{within_name} (I)": within_vars[i],
            f"{within_name} (J)": within_vars[j],
            "평균차 (I-J)": format_number(mean_diff, 4),
            "표준오차": format_number(se, 4),
            "p-adj (Bonferroni)": format_pvalue(p_adj) if not np.isnan(p_adj) else "-",
            f"CI 하한 ({int(confidence_level*100)}%)": format_number(mean_diff - ci_half, 4) if not np.isnan(ci_half) else "-",
            f"CI 상한 ({int(confidence_level*100)}%)": format_number(mean_diff + ci_half, 4) if not np.isnan(ci_half) else "-",
        })
    if rows:
        result.tables.append(ResultTable(
            title=f"Pairwise Comparisons: {within_name} (Bonferroni)",
            dataframe=pd.DataFrame(rows),
        ))


def _profile_plot_mixed(
    data: pd.DataFrame,
    between_var: str,
    within_vars: list,
    within_name: str,
) -> bytes | None:
    """집단 × 시점 상호작용 선 그래프 생성."""
    try:
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from nuristat.analysis._chart_font import ensure_korean_font
        ensure_korean_font()

        groups = sorted(data[between_var].unique(), key=str)
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, g in enumerate(groups):
            sub = data[data[between_var] == g][within_vars]
            means = sub.mean().values
            ax.plot(within_vars, means, marker="o", label=str(g), color=colors[i % len(colors)])

        ax.set_xlabel(within_name)
        ax.set_ylabel("평균")
        ax.set_title(f"프로파일 플롯: {between_var} × {within_name}")
        ax.legend(title=between_var)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Mixed ANOVA 프로파일 플롯 실패: %s", e)
        return None
