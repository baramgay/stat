"""Survival analysis for NuriStat.

Supports:
- Kaplan-Meier survival estimation (with or without lifelines)
- Log-rank test (scipy.stats)
- Cox Proportional Hazards regression (lifelines or manual)
- Survival function table, median survival time, 95% CI
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from nuristat.analysis.formatting import format_number, format_pvalue
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import parse_common_spec
from nuristat.core.dataset import Dataset

# ---------------------------------------------------------------------------
# Availability flags for optional dependency
# ---------------------------------------------------------------------------

_LIFELINES_AVAILABLE = False
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: F401
    from lifelines.statistics import logrank_test, multivariate_logrank_test  # noqa: F401
    _LIFELINES_AVAILABLE = True
except ImportError:
    pass


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run survival analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with:
                "duration": str   - time-to-event column
                "event": str      - event indicator column (1=event, 0=censored)
                "group": str      - optional grouping variable for log-rank test
                "covariates": list[str] - covariates for Cox regression
            - options: dict with:
                "method": "km" | "cox" | "both" (default "both")
                "alpha": significance level (default 0.05)
                "confidence_level": CI level (default 0.95)
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with survival analysis tables.
    """
    variables, options, confidence_level, missing_policy = parse_common_spec(spec)

    duration_var: str = variables.get("duration", "")
    event_var: str = variables.get("event", "")
    group_var: str | None = variables.get("group", None)
    covariates: list[str] = variables.get("covariates", [])
    method: str = options.get("method", "both")

    result = AnalysisResult(
        id="survival_analysis",
        title="생존분석 (Survival Analysis)",
        spec=spec,
    )

    if not duration_var or not event_var:
        result.warnings.append("생존 시간(duration) 및 사건 변수(event)가 필요합니다.")
        return result

    all_vars = [duration_var, event_var]
    if group_var:
        all_vars.append(group_var)
    all_vars += covariates

    try:
        prepared = prepare_analysis_frame(dataset, variables=all_vars, missing_policy=missing_policy)
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result
    df = prepared.data

    result.add_table(get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded, prepared.excluded_pct
    ))

    if len(df) == 0:
        result.warnings.append("결측 제거 후 유효한 관측치가 없습니다.")
        return result

    # Validate event column (0/1 이진 값만 허용, 정규화 후 확인)
    event_numeric = pd.to_numeric(df[event_var], errors="coerce")
    invalid_mask = ~event_numeric.isin([0, 1])
    if invalid_mask.any():
        bad_vals = df[event_var][invalid_mask].unique().tolist()
        result.warnings.append(
            f"사건 변수({event_var})는 0(중도절단) 또는 1(사건)이어야 합니다. "
            f"유효하지 않은 값: {bad_vals}"
        )
        return result

    T = df[duration_var].astype(float).values
    E = event_numeric.astype(int).values

    if method in ("km", "both"):
        if _LIFELINES_AVAILABLE:
            _run_km_lifelines(result, df, T, E, duration_var, event_var, group_var, confidence_level)
        else:
            _run_km_manual(result, T, E, group_var, df, confidence_level)

    if method in ("cox", "both") and covariates:
        if _LIFELINES_AVAILABLE:
            _run_cox_lifelines(result, df, duration_var, event_var, covariates, confidence_level)
        else:
            _run_cox_manual(result, T, E, df, covariates, confidence_level)

    if not _LIFELINES_AVAILABLE:
        result.notes.append(
            "lifelines 패키지가 설치되어 있지 않아 수동 구현을 사용했습니다. "
            "더 정확한 분석을 위해 'pip install lifelines'를 권장합니다."
        )

    return result


def run_kaplan_meier(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Kaplan-Meier 생존분석 전용 진입점.

    spec.variables: duration, event, group (optional)
    """
    spec = dict(spec)
    spec.setdefault("options", {})
    spec["options"]["method"] = "km"
    result = run_analysis(dataset, spec)
    result.id = "kaplan_meier"
    result.title = "Kaplan-Meier 생존분석"
    return result


def run_cox_regression(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Cox 비례위험 회귀 전용 진입점.

    spec.variables: duration, event, covariates (list)
    """
    spec = dict(spec)
    spec.setdefault("options", {})
    spec["options"]["method"] = "cox"
    result = run_analysis(dataset, spec)
    result.id = "cox_regression"
    result.title = "Cox 비례위험 회귀"
    return result


# ---------------------------------------------------------------------------
# Kaplan-Meier with lifelines
# ---------------------------------------------------------------------------

def _run_km_lifelines(
    result: AnalysisResult,
    df: pd.DataFrame,
    T: np.ndarray,
    E: np.ndarray,
    duration_var: str,
    event_var: str,
    group_var: str | None,
    confidence_level: float,
) -> None:
    """KM analysis using lifelines library."""
    from lifelines import KaplanMeierFitter

    alpha = 1 - confidence_level

    if group_var and group_var in df.columns:
        groups = df[group_var].unique()
        km_results = []
        km_objects = {}
        for grp in sorted(groups):
            mask = df[group_var] == grp
            t_grp = T[mask.values]
            e_grp = E[mask.values]
            kmf = KaplanMeierFitter()
            kmf.fit(t_grp, event_observed=e_grp, label=str(grp), alpha=alpha)
            km_objects[str(grp)] = kmf
            n_events = int(e_grp.sum())
            n_censored = int(len(e_grp) - n_events)
            median_survival = kmf.median_survival_time_
            km_results.append({
                "군집": str(grp),
                "N": len(t_grp),
                "사건 수": n_events,
                "중도절단 수": n_censored,
                "중앙 생존 시간": format_number(float(median_survival) if not np.isinf(median_survival) else np.nan, 3),
            })

        result.add_table(ResultTable(
            title="Kaplan-Meier 요약 (그룹별)",
            dataframe=pd.DataFrame(km_results),
        ))

        # Log-rank test
        from lifelines.statistics import multivariate_logrank_test
        test_result = multivariate_logrank_test(T, df[group_var].values, E)
        lr_df = pd.DataFrame([{
            "검정": "Log-rank (다변량)",
            "Chi-square": format_number(float(test_result.test_statistic), 3),
            "df": str(test_result.degrees_of_freedom),
            "p-value": format_pvalue(float(test_result.p_value)),
            "해석": "그룹 간 생존 분포 차이 유의" if test_result.p_value < 0.05 else "그룹 간 차이 없음",
        }])
        result.add_table(ResultTable(
            title="Log-rank 검정",
            dataframe=lr_df,
            footnotes=["p < .05이면 그룹 간 생존 분포에 통계적으로 유의한 차이가 있습니다."],
        ))

        # Survival function table per group
        for grp_label, kmf in km_objects.items():
            sf_df = kmf.survival_function_.reset_index()
            sf_df.columns = ["시간", "생존 확률"]
            ci_df = kmf.confidence_interval_.reset_index()
            merged = sf_df.copy()
            if ci_df.shape[1] >= 3:
                merged["95% CI 하한"] = ci_df.iloc[:, 1].values
                merged["95% CI 상한"] = ci_df.iloc[:, 2].values
            merged["생존 확률"] = merged["생존 확률"].apply(lambda x: format_number(x, 4))
            result.add_table(ResultTable(
                title=f"생존 함수 (군집: {grp_label})",
                dataframe=merged,
            ))
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, alpha=alpha)
        median_survival = kmf.median_survival_time_
        n_events = int(E.sum())
        n_censored = int(len(E) - n_events)

        summary_rows = [
            {"통계량": "전체 N", "값": str(len(T))},
            {"통계량": "사건 수", "값": str(n_events)},
            {"통계량": "중도절단 수", "값": str(n_censored)},
            {"통계량": "중앙 생존 시간", "값": format_number(float(median_survival) if not np.isinf(median_survival) else np.nan, 3)},
        ]
        result.add_table(ResultTable(title="Kaplan-Meier 요약", dataframe=pd.DataFrame(summary_rows)))

        sf_df = kmf.survival_function_.reset_index()
        sf_df.columns = ["시간", "생존 확률"]
        ci_df = kmf.confidence_interval_.reset_index()
        if ci_df.shape[1] >= 3:
            sf_df["95% CI 하한"] = ci_df.iloc[:, 1].values
            sf_df["95% CI 상한"] = ci_df.iloc[:, 2].values
        sf_df["생존 확률"] = sf_df["생존 확률"].apply(lambda x: format_number(x, 4))
        result.add_table(ResultTable(
            title="생존 함수 (Kaplan-Meier)",
            dataframe=sf_df,
            footnotes=["각 시점의 생존 확률과 95% 신뢰구간입니다."],
        ))


# ---------------------------------------------------------------------------
# Kaplan-Meier manual implementation (no lifelines)
# ---------------------------------------------------------------------------

def _run_km_manual(
    result: AnalysisResult,
    T: np.ndarray,
    E: np.ndarray,
    group_var: str | None,
    df: pd.DataFrame,
    confidence_level: float,
) -> None:
    """Manual KM estimator using only scipy/numpy."""

    def _km_table(t: np.ndarray, e: np.ndarray, confidence_level: float) -> pd.DataFrame:
        """Compute KM survival table."""
        order = np.argsort(t)
        t_sorted = t[order]
        e_sorted = e[order]

        unique_times = np.unique(t_sorted[e_sorted == 1])
        n = len(t_sorted)
        rows = []
        survival = 1.0
        log_var = 0.0  # Greenwood's formula log-variance

        for ti in unique_times:
            at_risk = int(np.sum(t_sorted >= ti))
            events = int(np.sum((t_sorted == ti) & (e_sorted == 1)))
            survival *= (1 - events / at_risk)
            if at_risk > events:
                log_var += events / (at_risk * (at_risk - events))

            # Greenwood's 95% CI on log(-log(S))
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            if survival > 0 and survival < 1:
                se_log = np.sqrt(log_var)
                ci_lo = max(0.0, survival ** np.exp(z * se_log / np.abs(np.log(survival))))
                ci_hi = min(1.0, survival ** np.exp(-z * se_log / np.abs(np.log(survival))))
            else:
                ci_lo = ci_hi = survival

            rows.append({
                "시간": float(ti),
                "위험 집합 크기": at_risk,
                "사건 수": events,
                "생존 확률 S(t)": format_number(survival, 4),
                "95% CI 하한": format_number(ci_lo, 4),
                "95% CI 상한": format_number(ci_hi, 4),
            })

        if not rows:
            rows.append({"시간": 0.0, "위험 집합 크기": n, "사건 수": 0,
                         "생존 확률 S(t)": "1.000", "95% CI 하한": "1.000", "95% CI 상한": "1.000"})

        # Median survival: first time S(t) <= 0.5
        medians = []
        for r in rows:
            try:
                sv = float(str(r["생존 확률 S(t)"]).strip())
                if sv <= 0.5:
                    medians.append(r["시간"])
            except (ValueError, TypeError):  # pragma: no cover
                continue
        median_t = medians[0] if medians else np.nan

        return pd.DataFrame(rows), median_t

    summary_rows_global = []

    if group_var and group_var in df.columns:
        groups = sorted(df[group_var].unique())
        group_data = {}
        for grp in groups:
            mask = (df[group_var] == grp).values
            t_g = T[mask]
            e_g = E[mask]
            km_df, med_t = _km_table(t_g, e_g, confidence_level)
            group_data[str(grp)] = (t_g, e_g, km_df, med_t)
            summary_rows_global.append({
                "그룹": str(grp),
                "N": len(t_g),
                "사건 수": int(e_g.sum()),
                "중도절단": int(len(e_g) - e_g.sum()),
                "중앙 생존 시간": format_number(med_t, 3),
            })
            result.add_table(ResultTable(title=f"KM 생존 함수 (그룹: {grp})", dataframe=km_df))

        result.add_table(ResultTable(
            title="Kaplan-Meier 그룹 요약",
            dataframe=pd.DataFrame(summary_rows_global),
        ))

        # Log-rank test
        _log_rank_test(result, T, E, df, group_var, groups)
    else:
        km_df, med_t = _km_table(T, E, confidence_level)
        summary_rows_global = [
            {"통계량": "전체 N", "값": str(len(T))},
            {"통계량": "사건 수", "값": str(int(E.sum()))},
            {"통계량": "중도절단 수", "값": str(int(len(E) - E.sum()))},
            {"통계량": "중앙 생존 시간", "값": format_number(med_t, 3)},
        ]
        result.add_table(ResultTable(title="Kaplan-Meier 요약", dataframe=pd.DataFrame(summary_rows_global)))
        result.add_table(ResultTable(
            title="KM 생존 함수",
            dataframe=km_df,
            footnotes=["각 사건 시점의 생존 확률과 95% Greenwood CI."],
        ))


def _log_rank_test(
    result: AnalysisResult,
    T: np.ndarray,
    E: np.ndarray,
    df: pd.DataFrame,
    group_var: str,
    groups: list,
) -> None:
    """Manual log-rank test for two or more groups."""
    try:
        event_times = np.unique(T[E == 1])
        n_groups = len(groups)
        group_arrays = {str(g): (T[(df[group_var] == g).values], E[(df[group_var] == g).values]) for g in groups}

        chi2 = 0.0
        df_stat = n_groups - 1
        # Mantel-Cox log-rank
        O_total = np.zeros(n_groups)
        E_total = np.zeros(n_groups)
        for ti in event_times:
            n_j = np.array([int(np.sum(arr[0] >= ti)) for arr in group_arrays.values()])
            d_j = np.array([int(np.sum((arr[0] == ti) & (arr[1] == 1))) for arr in group_arrays.values()])
            n_total = n_j.sum()
            d_total = d_j.sum()
            E_j = n_j * d_total / n_total
            O_total += d_j
            E_total += E_j

        for j in range(n_groups):
            if E_total[j] > 0:
                chi2 += (O_total[j] - E_total[j]) ** 2 / E_total[j]

        p_val = 1 - stats.chi2.cdf(chi2, df=df_stat)

        lr_rows = []
        for j, grp in enumerate(groups):
            lr_rows.append({
                "그룹": str(grp),
                "관측 사건(O)": format_number(float(O_total[j]), 1),
                "기대 사건(E)": format_number(float(E_total[j]), 3),
                "(O-E)^2/E": format_number(float((O_total[j] - E_total[j]) ** 2 / E_total[j]) if E_total[j] > 0 else np.nan, 3),
            })
        lr_rows.append({
            "그룹": "전체 (Log-rank)",
            "관측 사건(O)": "",
            "기대 사건(E)": "",
            "(O-E)^2/E": f"Chi2={format_number(chi2, 3)}, df={df_stat}, p={format_pvalue(p_val)}",
        })
        result.add_table(ResultTable(
            title="Log-rank 검정",
            dataframe=pd.DataFrame(lr_rows),
            footnotes=["p < .05이면 그룹 간 생존 분포에 유의한 차이가 있습니다."],
        ))
    except Exception as e:
        result.warnings.append(f"Log-rank 검정 계산 실패: {e}")


# ---------------------------------------------------------------------------
# Cox regression with lifelines
# ---------------------------------------------------------------------------

def _run_cox_lifelines(
    result: AnalysisResult,
    df: pd.DataFrame,
    duration_var: str,
    event_var: str,
    covariates: list[str],
    confidence_level: float,
) -> None:
    """Cox PH regression using lifelines."""
    from lifelines import CoxPHFitter

    try:
        cox_df = df[[duration_var, event_var] + covariates].copy()
        for col in covariates:
            cox_df[col] = pd.to_numeric(cox_df[col], errors="coerce")
        cox_df = cox_df.dropna()

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=duration_var, event_col=event_var)

        summary = cph.summary
        coef_rows = []
        for var in summary.index:
            row = summary.loc[var]
            coef_rows.append({
                "변수": var,
                "계수(coef)": format_number(float(row.get("coef", np.nan)), 4),
                "SE": format_number(float(row.get("se(coef)", np.nan)), 4),
                "HR (exp(coef))": format_number(float(row.get("exp(coef)", np.nan)), 3),
                "95% CI 하한": format_number(float(row.get("exp(coef) lower 95%", np.nan)), 3),
                "95% CI 상한": format_number(float(row.get("exp(coef) upper 95%", np.nan)), 3),
                "z": format_number(float(row.get("z", np.nan)), 3),
                "p-value": format_pvalue(float(row.get("p", np.nan))),
            })

        result.add_table(ResultTable(
            title="Cox 비례위험 회귀 - 계수표",
            dataframe=pd.DataFrame(coef_rows),
            footnotes=[
                "HR = Hazard Ratio (exp(coef)). HR > 1: 위험 증가, HR < 1: 위험 감소.",
                f"concordance index = {format_number(cph.concordance_index_, 3)}",
            ],
        ))

        # Model fit — semi-parametric Cox has no AIC_, use AIC_partial_ as fallback
        try:
            aic_val = cph.AIC_
        except Exception:
            try:
                aic_val = cph.AIC_partial_
            except Exception:
                aic_val = None
        aic_str = format_number(aic_val, 3) if aic_val is not None else "N/A"

        fit_rows = [
            {"통계량": "log-likelihood", "값": format_number(cph.log_likelihood_, 3)},
            {"통계량": "concordance index", "값": format_number(cph.concordance_index_, 3)},
            {"통계량": "AIC (부분 우도)", "값": aic_str},
        ]
        result.add_table(ResultTable(title="Cox 회귀 모형 적합 요약", dataframe=pd.DataFrame(fit_rows)))

    except Exception as e:
        result.warnings.append(f"Cox 회귀(lifelines) 실패: {e}")


def _run_cox_manual(
    result: AnalysisResult,
    T: np.ndarray,
    E: np.ndarray,
    df: pd.DataFrame,
    covariates: list[str],
    confidence_level: float,
) -> None:
    """Approximate Cox regression using statsmodels PHReg."""
    try:
        import statsmodels.duration.hazard_regression as hr

        X_df = df[covariates].copy()
        for col in covariates:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        valid = X_df.notna().all(axis=1)
        X_vals = X_df[valid].values.astype(float)
        T_val = T[valid.values]
        E_val = E[valid.values]

        model = hr.PHReg(T_val, X_vals, status=E_val)
        result_ph = model.fit(disp=False)

        alpha = 1 - confidence_level
        ci = result_ph.conf_int(alpha=alpha)
        coef_rows = []
        for i, var in enumerate(covariates):
            b = float(result_ph.params[i])
            se = float(result_ph.bse[i])
            z = b / se if se > 0 else np.nan
            p = float(result_ph.pvalues[i])
            hr_val = np.exp(b)
            ci_lo = np.exp(float(ci[i, 0]))
            ci_hi = np.exp(float(ci[i, 1]))
            coef_rows.append({
                "변수": var,
                "계수": format_number(b, 4),
                "SE": format_number(se, 4),
                "HR": format_number(hr_val, 3),
                "95% CI 하한": format_number(ci_lo, 3),
                "95% CI 상한": format_number(ci_hi, 3),
                "z": format_number(z, 3),
                "p-value": format_pvalue(p),
            })

        result.add_table(ResultTable(
            title="Cox 비례위험 회귀 계수표 (statsmodels PHReg)",
            dataframe=pd.DataFrame(coef_rows),
            footnotes=["HR = Hazard Ratio. CI = Confidence Interval."],
        ))
    except Exception as e:
        result.warnings.append(f"Cox 회귀(statsmodels) 실패: {e}")
