"""Linear regression analysis for StatWorkbench."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.formatting import format_ci, format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run linear regression analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "dependent" and "predictors".
            - options: dict with "reference_category".
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with regression output.
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = spec.get("confidence_level", 0.95)
    missing_policy_str = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(missing_policy_str, str):
        missing_policy = MissingPolicy(missing_policy_str)
    else:
        missing_policy = missing_policy_str

    dep_var: str = variables.get("dependent", "")
    predictors: list[str] = variables.get("independent", variables.get("predictors", []))
    reference_categories: dict = options.get("reference_category", {})
    selection_method: str = options.get("selection_method", "enter").lower()  # enter|forward|backward|stepwise
    do_influential: bool = options.get("influential_cases", True)
    do_stepwise_summary: bool = options.get("stepwise_summary", True)

    result = AnalysisResult(
        id="linear_regression",
        title="Linear Regression",
        spec=spec,
    )

    all_vars = [dep_var] + predictors

    try:
        prepared = prepare_analysis_frame(
            dataset, variables=all_vars, missing_policy=missing_policy
        )
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result

    df = prepared.data

    # Case Processing Summary
    cps = get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded,
        prepared.excluded_pct
    )
    result.add_table(cps)

    if len(df) == 0:
        result.warnings.append("No valid observations after missing data removal.")
        return result

    # Build regression model
    try:
        y = df[dep_var].values
        X_df, dummy_info = _build_design_matrix(
            df, predictors, dataset, reference_categories
        )

        if X_df.empty and predictors:
            result.add_warning("예측변수 행렬이 비어 있습니다. 범주형 더미 코딩 후 유효한 열이 없습니다.")
            return result

        # Add constant
        X = sm.add_constant(X_df.values)
        X_cols = ["(Constant)"] + list(X_df.columns)

        # Fit model
        model = sm.OLS(y, X)
        fitted = model.fit()

        # Model Summary
        r = np.sqrt(fitted.rsquared)
        rmse = np.sqrt(fitted.mse_resid)

        summary_rows = [
            {
                "Statistic": "R",
                "Value": format_number(r, 3),
            },
            {
                "Statistic": "R-squared",
                "Value": format_number(fitted.rsquared, 3),
            },
            {
                "Statistic": "Adjusted R-squared",
                "Value": format_number(fitted.rsquared_adj, 3),
            },
            {
                "Statistic": "RMSE (Std. Error of Estimate)",
                "Value": format_number(rmse, 3),
            },
            {
                "Statistic": "F-statistic",
                "Value": format_number(fitted.fvalue, 3),
            },
            {
                "Statistic": "df1",
                "Value": fitted.df_model,
            },
            {
                "Statistic": "df2",
                "Value": fitted.df_resid,
            },
            {
                "Statistic": "p-value (F-test)",
                "Value": format_pvalue(fitted.f_pvalue),
            },
            {
                "Statistic": "N",
                "Value": fitted.nobs,
            },
            {
                "Statistic": "AIC",
                "Value": format_number(fitted.aic, 1),
            },
            {
                "Statistic": "BIC",
                "Value": format_number(fitted.bic, 1),
            },
        ]

        result.add_table(ResultTable(
            title="Model Summary",
            dataframe=pd.DataFrame(summary_rows),
        ))

        # ANOVA table (computed directly from OLS fit attributes)
        ss_reg = fitted.ess
        ss_res = fitted.ssr
        ss_tot = fitted.centered_tss
        df_reg = int(fitted.df_model)
        df_res = int(fitted.df_resid)
        ms_reg = ss_reg / df_reg if df_reg > 0 else 0.0
        anova_rows = [
            {
                "Source": "Regression",
                "SS": format_number(ss_reg, 3),
                "df": df_reg,
                "MS": format_number(ms_reg, 3),
                "F": format_number(fitted.fvalue, 3),
                "p-value": format_pvalue(fitted.f_pvalue),
            },
            {
                "Source": "Residual",
                "SS": format_number(ss_res, 3),
                "df": df_res,
                "MS": format_number(fitted.mse_resid, 3),
                "F": "",
                "p-value": "",
            },
            {
                "Source": "Total",
                "SS": format_number(ss_tot, 3),
                "df": df_reg + df_res,
                "MS": "",
                "F": "",
                "p-value": "",
            },
        ]
        result.add_table(ResultTable(
            title="ANOVA",
            dataframe=pd.DataFrame(anova_rows),
        ))

        # Coefficients
        conf_int = fitted.conf_int(alpha=1 - confidence_level)
        _ci_arr = conf_int.values if hasattr(conf_int, 'values') else conf_int

        def _ci_get(i: int, j: int) -> float:
            return float(_ci_arr[i, j])

        coef_rows = []
        for i, col_name in enumerate(X_cols):
            b = fitted.params[i]
            se = fitted.bse[i]
            t = fitted.tvalues[i]
            p = fitted.pvalues[i]
            ci_low = _ci_get(i, 0)
            ci_high = _ci_get(i, 1)

            # Standardized beta
            beta: float | str
            if i == 0:
                beta = ""
            else:
                x_std = X_df[col_name].std()
                y_std = np.std(y, ddof=1)
                beta = float(b * x_std / y_std) if (x_std > 0 and y_std > 0) else float("nan")

            coef_rows.append({
                "Variable": col_name,
                "B": format_number(b, 4),
                "SE": format_number(se, 4),
                "Beta": format_number(beta, 4) if beta != "" else "",
                "t": format_number(t, 3),
                "p-value": format_pvalue(p),
                "CI": format_ci(ci_low, ci_high, level=confidence_level),
            })

        result.add_table(ResultTable(
            title="Coefficients",
            dataframe=pd.DataFrame(coef_rows),
        ))

        # Diagnostics
        # VIF
        vif_rows = []
        n_predictors = X.shape[1] - 1  # exclude constant
        if n_predictors > 0:
            for i, col_name in enumerate(X_cols[1:], start=1):
                try:
                    vif_val = variance_inflation_factor(X, i)
                    vif_rows.append({
                        "Variable": col_name,
                        "VIF": format_number(vif_val, 3),
                        "Tolerance": format_number(1 / vif_val, 3),
                        "Warning": "High multicollinearity" if vif_val > 10 else (
                            "Moderate multicollinearity" if vif_val > 5 else ""
                        ),
                    })
                except Exception:
                    vif_rows.append({
                        "Variable": col_name,
                        "VIF": "N/A",
                        "Tolerance": "N/A",
                        "Warning": "",
                    })

        if vif_rows:
            result.add_table(ResultTable(
                title="Collinearity Diagnostics (VIF)",
                dataframe=pd.DataFrame(vif_rows),
            ))

        # Durbin-Watson
        dw_stat = sms.durbin_watson(fitted.resid)
        dw_df = pd.DataFrame([
            {
                "Statistic": "Durbin-Watson",
                "Value": format_number(dw_stat, 3),
                "Interpretation": (
                    "No autocorrelation" if 1.5 <= dw_stat <= 2.5 else
                    "Possible positive autocorrelation" if dw_stat < 1.5 else
                    "Possible negative autocorrelation"
                ),
            }
        ])
        result.add_table(ResultTable(
            title="Autocorrelation Test",
            dataframe=dw_df,
        ))

        # Residual summary
        residuals = fitted.resid
        resid_rows = [
            {
                "Statistic": "Mean",
                "Value": format_number(float(np.mean(residuals)), 4),
            },
            {
                "Statistic": "SD",
                "Value": format_number(float(np.std(residuals, ddof=1)), 4),
            },
            {
                "Statistic": "Min",
                "Value": format_number(float(np.min(residuals)), 4),
            },
            {
                "Statistic": "Max",
                "Value": format_number(float(np.max(residuals)), 4),
            },
            {
                "Statistic": "Median",
                "Value": format_number(float(np.median(residuals)), 4),
            },
        ]
        result.add_table(ResultTable(
            title="Residual Summary",
            dataframe=pd.DataFrame(resid_rows),
        ))

        # Influential Cases (Cook's D + Leverage)
        if do_influential:
            try:
                influence = fitted.get_influence()
                cooks_d = influence.cooks_distance[0]
                leverage = influence.hat_matrix_diag
                std_resid = influence.resid_studentized_internal
                n_obs = len(cooks_d)
                k_params = fitted.df_model + 1  # number of predictors + intercept
                cooks_threshold = 4.0 / n_obs if n_obs > 0 else 1.0

                inf_rows = []
                for idx in range(n_obs):
                    cd = float(cooks_d[idx])
                    lev = float(leverage[idx])
                    sr = float(std_resid[idx]) if idx < len(std_resid) else float("nan")
                    flag = "**" if cd > cooks_threshold else ("*" if lev > 2 * k_params / n_obs else "")
                    if flag or (abs(sr) > 2):
                        inf_rows.append({
                            "관측치": idx + 1,
                            "Cook's D": format_number(cd, 4),
                            "레버리지": format_number(lev, 4),
                            "표준화 잔차": format_number(sr, 3),
                            "영향도": flag if flag else ("잔차 큰 경우" if abs(sr) > 2 else ""),
                        })

                if inf_rows:
                    result.add_table(ResultTable(
                        title=f"영향력 케이스 진단 (Cook's D 기준: {format_number(cooks_threshold, 4)})",
                        dataframe=pd.DataFrame(inf_rows),
                    ))
                else:
                    result.add_table(ResultTable(
                        title="영향력 케이스 진단",
                        dataframe=pd.DataFrame([{"결과": f"영향력 큰 케이스 없음 (Cook's D 기준 {format_number(cooks_threshold, 4)})"}]),
                    ))
            except Exception as e:
                logger.warning("영향력 케이스 진단 실패: %s", e)

        # Stepwise summary (if stepwise/forward/backward was used)
        if selection_method in ("forward", "backward", "stepwise") and do_stepwise_summary:
            stepwise_rows = _run_stepwise(
                df, dep_var, predictors, dataset, reference_categories,
                confidence_level, selection_method,
            )
            if stepwise_rows:
                result.add_table(ResultTable(
                    title=f"변수 선택 요약 ({selection_method.capitalize()})",
                    dataframe=pd.DataFrame(stepwise_rows),
                ))

        # Dummy coding info
        if dummy_info:
            dummy_rows = []
            for var_name, info in dummy_info.items():
                dummy_rows.append({
                    "Variable": var_name,
                    "Reference": info.get("reference", ""),
                    "Categories": ", ".join(str(c) for c in info.get("categories", [])),
                    "N dummies": info.get("n_dummies", 0),
                })
            result.add_table(ResultTable(
                title="Dummy Coding",
                dataframe=pd.DataFrame(dummy_rows),
            ))

    except Exception as exc:
        result.add_warning(f"회귀분석 계산 오류: {exc}")

    return result


def _build_design_matrix(
    df: pd.DataFrame,
    predictors: list[str],
    dataset: Dataset,
    reference_categories: dict,
) -> tuple[pd.DataFrame, dict]:
    """Build design matrix with automatic dummy coding.

    Returns (X_df, dummy_info).
    """
    X_parts = []
    dummy_info = {}

    for var_name in predictors:
        if var_name not in df.columns:
            continue

        series = df[var_name]
        meta = dataset.variables.get(var_name)
        measure = meta.measure if meta else None

        if measure in (MeasureType.NOMINAL, MeasureType.BINARY, MeasureType.ORDINAL):
            # Convert to string category for consistent dummy coding
            series_str = series.astype(str)
            unique_vals = sorted(series_str.dropna().unique())
            if len(unique_vals) <= 20:
                ref = reference_categories.get(var_name, unique_vals[0] if unique_vals else None)
                dummies = pd.get_dummies(
                    series_str, prefix=var_name, drop_first=False
                )
                ref_col = f"{var_name}_{ref}"
                if ref_col in dummies.columns:
                    dummies = dummies.drop(columns=[ref_col])
                else:
                    dummies = dummies.iloc[:, 1:]

                dummy_info[var_name] = {
                    "reference": str(ref),
                    "categories": [str(v) for v in unique_vals],
                    "n_dummies": len(dummies.columns),
                }
                X_parts.append(dummies)
            else:
                X_parts.append(pd.DataFrame({var_name: series}))
        else:
            X_parts.append(pd.DataFrame({var_name: series}))

    if X_parts:
        X_df = pd.concat(X_parts, axis=1)
    else:
        X_df = pd.DataFrame(index=df.index)

    # Ensure all columns are numeric for statsmodels
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    # Convert entire DataFrame to float64 to avoid mixed-type issues
    X_df = X_df.astype(float)

    return X_df, dummy_info


class RegressionEngine:
    """AnalysisPlugin wrapper for linear regression."""

    id = "linear_regression"
    name = "선형 회귀"
    category = "Regression"
    description = "선형 회귀분석: ANOVA 테이블, 계수, Beta, VIF, Durbin-Watson"
    variable_requirements: list[dict] = [
        {"role": "dependent", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "predictors", "measure_types": ["scale", "nominal", "ordinal", "binary"], "min_count": 1, "required": True},
    ]
    implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        return run_analysis(dataset, spec)


def _run_stepwise(
    df: pd.DataFrame,
    dep_var: str,
    predictors: list[str],
    dataset: Dataset,
    reference_categories: dict,
    confidence_level: float,
    method: str,
) -> list[dict]:
    """Forward/Backward/Stepwise 변수 선택 요약 테이블 반환."""

    p_enter = 0.05
    p_remove = 0.10
    rows: list[dict] = []

    y = df[dep_var].values
    remaining = [p for p in predictors if p in df.columns]
    selected: list[str] = []

    if method in ("backward", "stepwise"):
        selected = list(remaining)
        remaining = []

    step = 0
    max_steps = len(predictors) * 2 + 1

    for _ in range(max_steps):
        changed = False

        # Forward step: try adding each remaining predictor
        if method in ("forward", "stepwise") and remaining:
            best_p = 1.0
            best_var = None
            for var in remaining:
                trial = selected + [var]
                try:
                    X_df, _ = _build_design_matrix(df, trial, dataset, reference_categories)
                    X = sm.add_constant(X_df.values)
                    fit = sm.OLS(y, X).fit()
                    # p-value for the added variable (last predictor in model)
                    p_val = float(fit.pvalues[-1])
                    if p_val < best_p:
                        best_p = p_val
                        best_var = var
                except Exception:
                    continue
            if best_var is not None and best_p <= p_enter:
                step += 1
                selected.append(best_var)
                remaining.remove(best_var)
                rows.append({
                    "단계": step,
                    "동작": "입력",
                    "변수": best_var,
                    "p": round(best_p, 4),
                    "현재 모델": ", ".join(selected),
                })
                changed = True

        # Backward step: try removing each selected predictor
        if method in ("backward", "stepwise") and len(selected) > 1:
            worst_p = 0.0
            worst_var = None
            try:
                X_df, _ = _build_design_matrix(df, selected, dataset, reference_categories)
                X = sm.add_constant(X_df.values)
                fit = sm.OLS(y, X).fit()
                for i, var in enumerate(selected):
                    p_val = float(fit.pvalues[i + 1]) if i + 1 < len(fit.pvalues) else 1.0
                    if p_val > worst_p:
                        worst_p = p_val
                        worst_var = var
            except Exception:
                pass
            if worst_var is not None and worst_p > p_remove:
                step += 1
                selected.remove(worst_var)
                if method == "stepwise":
                    remaining.append(worst_var)
                rows.append({
                    "단계": step,
                    "동작": "제거",
                    "변수": worst_var,
                    "p": round(worst_p, 4),
                    "현재 모델": ", ".join(selected) if selected else "(없음)",
                })
                changed = True

        if not changed:
            break

    return rows
