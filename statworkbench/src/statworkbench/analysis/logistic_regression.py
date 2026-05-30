"""Logistic regression analysis for StatWorkbench.

Supports binary and multinomial logistic regression with:
- Odds Ratios and 95% CI
- Wald test statistics
- Hosmer-Lemeshow goodness-of-fit test
- Nagelkerke R2, Cox-Snell R2
- Classification table
- ROC AUC
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: F401
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from statworkbench.analysis.assumptions import get_case_processing_summary, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """Run logistic regression analysis.

    Args:
        dataset: The dataset to analyse.
        spec: Analysis specification with keys:
            - variables: dict with "dependent" (str) and "predictors" (list[str]).
            - options: dict with "method" ("binary" | "multinomial"), "ci_level".
            - confidence_level: confidence level (default 0.95).
            - missing_policy: missing data handling (default LISTWISE).

    Returns:
        AnalysisResult with logistic regression tables.
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
    predictors: list[str] = variables.get("predictors", variables.get("independent", []))
    method: str = options.get("method", "binary")

    result = AnalysisResult(
        id="logistic_regression",
        title="Logistic Regression",
        spec=spec,
    )

    all_vars = [dep_var] + predictors
    try:
        prepared = prepare_analysis_frame(dataset, variables=all_vars, missing_policy=missing_policy)
    except Exception as exc:
        result.add_warning(f"분석 오류: {exc}")
        return result

    df = prepared.data

    # Case Processing Summary
    result.add_table(get_case_processing_summary(
        prepared.n_total, prepared.n_valid, prepared.n_excluded, prepared.excluded_pct
    ))

    if len(df) == 0:
        result.warnings.append("결측 제거 후 유효한 관측치가 없습니다.")
        return result

    y = df[dep_var]
    n_classes = y.nunique()

    if n_classes < 2:
        result.warnings.append("종속변수에 2개 이상의 범주가 필요합니다.")
        return result

    # Encode dependent variable
    y_codes, class_labels = pd.factorize(y)

    # Build design matrix
    X_df = _build_predictor_matrix(df, predictors, dataset)
    if X_df.empty:
        result.warnings.append("예측변수 행렬을 구성할 수 없습니다.")
        return result

    try:
        X = sm.add_constant(X_df.values.astype(float))
        col_names = ["(Constant)"] + list(X_df.columns)

        if n_classes == 2 or method == "binary":
            _run_binary_logistic(
                result, X, y_codes, y, col_names, confidence_level, class_labels
            )
        else:
            _run_multinomial_logistic(
                result, X, y_codes, y, col_names, confidence_level, class_labels
            )
    except Exception as exc:
        result.add_warning(f"로지스틱 회귀 계산 오류: {exc}")

    return result


def _build_predictor_matrix(
    df: pd.DataFrame,
    predictors: list[str],
    dataset: Dataset,
) -> pd.DataFrame:
    """Build numeric design matrix from predictors with dummy coding."""
    parts: list[pd.DataFrame] = []
    for var in predictors:
        if var not in df.columns:
            continue
        meta = dataset.variables.get(var)
        measure = meta.measure if meta else None
        if measure in (MeasureType.NOMINAL, MeasureType.BINARY, MeasureType.ORDINAL):
            series_str = df[var].astype(str)
            unique_vals = sorted(series_str.dropna().unique())
            if len(unique_vals) <= 20:
                dummies = pd.get_dummies(series_str, prefix=var, drop_first=True)
                parts.append(dummies)
            else:
                parts.append(pd.DataFrame({var: df[var]}))
        else:
            parts.append(pd.DataFrame({var: df[var]}))
    if not parts:
        return pd.DataFrame()
    X_df = pd.concat(parts, axis=1)
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    return X_df.astype(float)


def _run_binary_logistic(
    result: AnalysisResult,
    X: np.ndarray,
    y_codes: np.ndarray,
    y_orig: pd.Series,
    col_names: list[str],
    confidence_level: float,
    class_labels: np.ndarray,
) -> None:
    """Fit binary logistic regression and populate result tables."""
    alpha = 1 - confidence_level
    try:
        model = sm.Logit(y_codes, X)
        fitted = model.fit(disp=False, maxiter=200)
    except Exception as e:
        result.warnings.append(f"모델 적합 실패: {e}")
        return

    n = len(y_codes)
    ll_null = fitted.llnull
    ll_model = fitted.llf

    # Cox-Snell R2
    cox_snell = 1 - np.exp(2 * (ll_null - ll_model) / n)
    # Nagelkerke R2
    nagelkerke = cox_snell / (1 - np.exp(2 * ll_null / n))

    # Model fit summary
    model_rows = [
        {"통계량": "-2 Log Likelihood (Null)", "값": format_number(-2 * ll_null, 3)},
        {"통계량": "-2 Log Likelihood (Model)", "값": format_number(-2 * ll_model, 3)},
        {"통계량": "Chi-square", "값": format_number(fitted.llr, 3)},
        {"통계량": "df", "값": str(fitted.df_model)},
        {"통계량": "p-value", "값": format_pvalue(fitted.llr_pvalue)},
        {"통계량": "Cox-Snell R2", "값": format_number(cox_snell, 3)},
        {"통계량": "Nagelkerke R2", "값": format_number(nagelkerke, 3)},
        {"통계량": "N", "값": str(n)},
        {"통계량": "AIC", "값": format_number(fitted.aic, 3)},
        {"통계량": "BIC", "값": format_number(fitted.bic, 3)},
    ]
    result.add_table(ResultTable(
        title="모형 요약 (Model Summary)",
        dataframe=pd.DataFrame(model_rows),
        footnotes=["Cox-Snell 및 Nagelkerke R2는 설명력의 근사치입니다."],
    ))

    # Coefficients table with OR and 95% CI
    conf_int = fitted.conf_int(alpha=alpha)
    coef_rows = []
    for i, name in enumerate(col_names):
        b = fitted.params[i]
        se = fitted.bse[i]
        wald = (b / se) ** 2 if se > 0 else np.nan
        p = fitted.pvalues[i]
        or_val = np.exp(b)
        ci_lo = np.exp(conf_int.iloc[i, 0]) if hasattr(conf_int, "iloc") else np.exp(conf_int[i, 0])
        ci_hi = np.exp(conf_int.iloc[i, 1]) if hasattr(conf_int, "iloc") else np.exp(conf_int[i, 1])
        coef_rows.append({
            "변수": name,
            "B": format_number(b, 4),
            "SE": format_number(se, 4),
            "Wald": format_number(wald, 3),
            "df": "1",
            "p-value": format_pvalue(p),
            "OR (Exp(B))": format_number(or_val, 3),
            "95% CI 하한": format_number(ci_lo, 3),
            "95% CI 상한": format_number(ci_hi, 3),
        })
    result.add_table(ResultTable(
        title="계수표 (Coefficients)",
        dataframe=pd.DataFrame(coef_rows),
        footnotes=["OR = Odds Ratio. CI = Confidence Interval."],
    ))

    # Classification table
    pred_prob = fitted.predict(X)
    pred_class = (pred_prob >= 0.5).astype(int)
    labels = [str(class_labels[i]) for i in range(len(class_labels))]

    if _SKLEARN_AVAILABLE:
        from sklearn.metrics import confusion_matrix as _cm
        from sklearn.metrics import roc_auc_score as _roc_auc
        cm = _cm(y_codes, pred_class)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / n * 100
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else np.nan
        class_rows = [
            {"구분": f"실제 {labels[0] if labels else '0'}", "예측 0": tn, "예측 1": fp, "정확도(행)": f"{tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else ""},
            {"구분": f"실제 {labels[1] if len(labels) > 1 else '1'}", "예측 0": fn, "예측 1": tp, "정확도(행)": f"{tp/(fn+tp)*100:.1f}%" if (fn+tp) > 0 else ""},
            {"구분": "전체 정확도", "예측 0": "", "예측 1": "", "정확도(행)": f"{accuracy:.1f}%"},
        ]
        result.add_table(ResultTable(
            title="분류표 (Classification Table)",
            dataframe=pd.DataFrame(class_rows),
            footnotes=[
                f"민감도(Sensitivity): {format_number(sensitivity, 1)}%",
                f"특이도(Specificity): {format_number(specificity, 1)}%",
                "분류 기준값 = 0.5",
            ],
        ))

        # ROC AUC
        try:
            if len(np.unique(y_codes)) == 2:
                auc = _roc_auc(y_codes, pred_prob)
                auc_df = pd.DataFrame([{
                    "통계량": "ROC AUC",
                    "값": format_number(auc, 3),
                    "해석": "우수" if auc >= 0.8 else ("양호" if auc >= 0.7 else "보통"),
                }])
                result.add_table(ResultTable(
                    title="ROC 분석",
                    dataframe=auc_df,
                    footnotes=["AUC > 0.8: 우수, 0.7-0.8: 양호, 0.6-0.7: 보통"],
                ))
        except Exception as exc:
            logger.warning("ROC/분류표 계산 실패: %s", exc)
    else:
        # Manual classification table without sklearn
        _manual_classification_table(result, y_codes, pred_class, pred_prob, labels, n)

    # Hosmer-Lemeshow test
    _hosmer_lemeshow_test(result, y_codes, pred_prob)


def _manual_classification_table(
    result: AnalysisResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_prob: np.ndarray,
    labels: list[str],
    n: int,
) -> None:
    """Build classification table without sklearn."""
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    rows = []
    for actual in unique_classes:
        row: dict = {"실제": labels[int(actual)] if int(actual) < len(labels) else str(actual)}
        total_actual = int(np.sum(y_true == actual))
        correct = int(np.sum((y_true == actual) & (y_pred == actual)))
        for predicted in unique_classes:
            row[f"예측 {labels[int(predicted)] if int(predicted) < len(labels) else str(predicted)}"] = int(np.sum((y_true == actual) & (y_pred == predicted)))
        row["정확도(행)"] = f"{correct/total_actual*100:.1f}%" if total_actual > 0 else ""
        rows.append(row)
    overall_acc = float(np.sum(y_true == y_pred)) / n * 100
    result.add_table(ResultTable(
        title="분류표 (Classification Table)",
        dataframe=pd.DataFrame(rows),
        footnotes=[f"전체 정확도: {format_number(overall_acc, 1)}%", "분류 기준값 = 0.5"],
    ))

    # Manual AUC via trapezoidal rule
    if len(unique_classes) == 2:
        try:
            thresholds = np.sort(np.unique(pred_prob))[::-1]
            tpr_list, fpr_list = [0.0], [0.0]
            pos = int(np.sum(y_true == 1))
            neg = n - pos
            if pos > 0 and neg > 0:
                for thr in thresholds:
                    pred_thr = (pred_prob >= thr).astype(int)
                    tp = int(np.sum((y_true == 1) & (pred_thr == 1)))
                    fp = int(np.sum((y_true == 0) & (pred_thr == 1)))
                    tpr_list.append(tp / pos)
                    fpr_list.append(fp / neg)
                tpr_list.append(1.0)
                fpr_list.append(1.0)
                auc = float(_trapz(tpr_list, fpr_list))
                auc_df = pd.DataFrame([{
                    "통계량": "ROC AUC (근사)",
                    "값": format_number(abs(auc), 3),
                    "해석": "우수" if abs(auc) >= 0.8 else ("양호" if abs(auc) >= 0.7 else "보통"),
                }])
                result.add_table(ResultTable(title="ROC 분석 (근사)", dataframe=auc_df))
        except Exception as exc:
            logger.warning("ROC 근사 계산 실패: %s", exc)


def _hosmer_lemeshow_test(
    result: AnalysisResult,
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    n_groups: int = 10,
) -> None:
    """Compute Hosmer-Lemeshow goodness-of-fit test."""
    try:
        df_hl = pd.DataFrame({"y": y_true, "prob": pred_prob})
        df_hl["decile"] = pd.qcut(df_hl["prob"], q=n_groups, duplicates="drop", labels=False)

        hl_chi2 = 0.0
        for _, grp in df_hl.groupby("decile"):
            obs_1 = grp["y"].sum()
            obs_0 = len(grp) - obs_1
            exp_1 = grp["prob"].sum()
            exp_0 = len(grp) - exp_1
            if exp_1 >= 1.0:
                hl_chi2 += (obs_1 - exp_1) ** 2 / exp_1
            if exp_0 >= 1.0:
                hl_chi2 += (obs_0 - exp_0) ** 2 / exp_0

        actual_groups = df_hl["decile"].nunique()
        df_hl_stat = actual_groups - 2
        if df_hl_stat > 0:
            hl_p = 1 - stats.chi2.cdf(hl_chi2, df=df_hl_stat)
        else:
            hl_p = np.nan

        hl_rows = [
            {"검정": "Hosmer-Lemeshow", "Chi-square": format_number(hl_chi2, 3),
             "df": str(df_hl_stat), "p-value": format_pvalue(hl_p),
             "해석": "양호 적합" if (not np.isnan(hl_p) and hl_p >= 0.05) else "적합 부족"},
        ]
        result.add_table(ResultTable(
            title="Hosmer-Lemeshow 검정",
            dataframe=pd.DataFrame(hl_rows),
            footnotes=["p >= .05: 모형이 데이터에 잘 적합됨."],
        ))
    except Exception as e:
        result.warnings.append(f"Hosmer-Lemeshow 검정 계산 실패: {e}")


def _run_multinomial_logistic(
    result: AnalysisResult,
    X: np.ndarray,
    y_codes: np.ndarray,
    y_orig: pd.Series,
    col_names: list[str],
    confidence_level: float,
    class_labels: np.ndarray,
) -> None:
    """Fit multinomial logistic regression."""
    try:
        model = sm.MNLogit(y_codes, X)
        fitted = model.fit(disp=False, maxiter=200)
    except Exception as e:
        result.warnings.append(f"다항 로지스틱 모델 적합 실패: {e}")
        return

    n = len(y_codes)
    ll_null = fitted.llnull
    ll_model = fitted.llf
    cox_snell = 1 - np.exp(2 * (ll_null - ll_model) / n)
    nagelkerke = cox_snell / (1 - np.exp(2 * ll_null / n))

    model_rows = [
        {"통계량": "-2 Log Likelihood (Null)", "값": format_number(-2 * ll_null, 3)},
        {"통계량": "-2 Log Likelihood (Model)", "값": format_number(-2 * ll_model, 3)},
        {"통계량": "Chi-square", "값": format_number(fitted.llr, 3)},
        {"통계량": "df", "값": str(int(fitted.df_model))},
        {"통계량": "p-value", "값": format_pvalue(fitted.llr_pvalue)},
        {"통계량": "Cox-Snell R2", "값": format_number(cox_snell, 3)},
        {"통계량": "Nagelkerke R2", "값": format_number(nagelkerke, 3)},
        {"통계량": "N", "값": str(n)},
    ]
    result.add_table(ResultTable(title="모형 요약 (다항 로지스틱)", dataframe=pd.DataFrame(model_rows)))

    # Coefficients per class
    params = fitted.params  # shape: (n_predictors, n_classes-1)
    bse = fitted.bse
    pvalues = fitted.pvalues
    n_outcomes = params.shape[1] if params.ndim > 1 else 1

    for k in range(n_outcomes):
        label = str(class_labels[k + 1]) if k + 1 < len(class_labels) else str(k + 1)
        coef_rows = []
        for i, name in enumerate(col_names):
            b = params.iloc[i, k] if hasattr(params, "iloc") else params[i, k]
            se = bse.iloc[i, k] if hasattr(bse, "iloc") else bse[i, k]
            p = pvalues.iloc[i, k] if hasattr(pvalues, "iloc") else pvalues[i, k]
            wald = (b / se) ** 2 if se > 0 else np.nan
            or_val = np.exp(b)
            coef_rows.append({
                "변수": name,
                "B": format_number(b, 4),
                "SE": format_number(se, 4),
                "Wald": format_number(wald, 3),
                "p-value": format_pvalue(float(p)),
                "OR": format_number(or_val, 3),
            })
        result.add_table(ResultTable(
            title=f"계수표 (범주: {label} vs 기준)",
            dataframe=pd.DataFrame(coef_rows),
        ))

    result.notes.append(f"다항 로지스틱 회귀 - 범주 수: {len(class_labels)}")
