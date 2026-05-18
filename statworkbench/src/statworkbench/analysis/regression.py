"""Linear regression analysis for StatWorkbench."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy, MeasureType
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_pvalue, format_number, format_ci
from statworkbench.analysis.assumptions import prepare_analysis_frame, get_case_processing_summary


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

    result = AnalysisResult(
        id="linear_regression",
        title="Linear Regression",
        spec=spec,
    )

    all_vars = [dep_var] + predictors

    prepared = prepare_analysis_frame(
        dataset, variables=all_vars, missing_policy=missing_policy
    )
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
    y = df[dep_var].values
    X_df, dummy_info = _build_design_matrix(
        df, predictors, dataset, reference_categories
    )

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

    # ANOVA table
    try:
        anova_table = sm.stats.anova_lm(fitted, typ=2)
        anova_rows = []
        for idx, row in anova_table.iterrows():
            anova_rows.append({
                "Source": idx,
                "SS": format_number(row["sum_sq"], 3),
                "df": int(row["df"]) if not np.isnan(row["df"]) else "",
                "MS": format_number(row["sum_sq"] / row["df"], 3) if row["df"] > 0 else "",
                "F": format_number(row["F"], 3) if not np.isnan(row["F"]) else "",
                "p-value": format_pvalue(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else "",
            })
        result.add_table(ResultTable(
            title="ANOVA",
            dataframe=pd.DataFrame(anova_rows),
        ))
    except Exception:
        pass

    # Coefficients
    conf_int = fitted.conf_int(alpha=1 - confidence_level)
    # Handle both DataFrame and ndarray return types
    if hasattr(conf_int, 'iloc'):
        _ci_get = lambda i, j: conf_int.iloc[i, j]
    else:
        _ci_get = lambda i, j: conf_int[i, j]
    coef_rows = []
    for i, col_name in enumerate(X_cols):
        b = fitted.params[i]
        se = fitted.bse[i]
        t = fitted.tvalues[i]
        p = fitted.pvalues[i]
        ci_low = _ci_get(i, 0)
        ci_high = _ci_get(i, 1)

        # Standardized beta
        if i == 0:
            beta = ""
        else:
            x_col = col_name
            if x_col in X_df.columns:
                x_std = X_df[x_col].std()
                y_std = np.std(y, ddof=1)
                beta = b * x_std / y_std if y_std > 0 else 0
            else:
                beta = ""

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
