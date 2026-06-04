"""Sensitivity / Specificity Analysis — 진단 정확도 분석.

SPSS Analyze > Descriptive Statistics > Crosstabs (with statistics)
또는 Analyze > ROC Curve 에 해당하는 이진 분류 진단 정확도 측정 모듈.

출력 테이블:
  1. Case Processing Summary   — 전체/유효/제외 케이스 수
  2. 2×2 Contingency Table    — TP, FP, FN, TN
  3. Diagnostic Accuracy      — 민감도, 특이도, PPV, NPV, 정확도, F1
  4. Likelihood Ratios        — LR+, LR-, Odds Ratio
  5. Agreement Statistics     — Youden J, MCC, Cohen's Kappa
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.formatting import format_number
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core.dataset import Dataset

# ──────────────────────────────────────────────────────────────
# 내부 계산 함수
# ──────────────────────────────────────────────────────────────

def _compute_2x2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> dict[str, int]:
    """2×2 분할표 셀 계산 (TP, FP, FN, TN)."""
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _diagnostic_metrics(cells: dict[str, int], confidence_level: float = 0.95) -> dict[str, Any]:
    """분할표 셀에서 진단 정확도 지표를 계산합니다.

    Returns:
        dict with keys:
            sensitivity, specificity, ppv, npv, accuracy, f1
            lr_pos, lr_neg, odds_ratio
            youden_j, mcc, kappa
            각 지표의 _ci_lower, _ci_upper (Wilson score interval)
    """
    tp, fp, fn, tn = cells["TP"], cells["FP"], cells["FN"], cells["TN"]
    n = tp + fp + fn + tn
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    def _wilson_ci(k: int, n_tot: int) -> tuple[float, float]:
        """Wilson score 신뢰구간."""
        if n_tot == 0:
            return (0.0, 0.0)
        p = k / n_tot
        denom = 1 + z**2 / n_tot
        center = (p + z**2 / (2 * n_tot)) / denom
        half = z * math.sqrt(p * (1 - p) / n_tot + z**2 / (4 * n_tot**2)) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    # --- 민감도 (Sensitivity = Recall = TPR) ---
    denom_sens = tp + fn
    sensitivity = tp / denom_sens if denom_sens > 0 else float("nan")
    sens_ci = _wilson_ci(tp, denom_sens)

    # --- 특이도 (Specificity = TNR) ---
    denom_spec = tn + fp
    specificity = tn / denom_spec if denom_spec > 0 else float("nan")
    spec_ci = _wilson_ci(tn, denom_spec)

    # --- PPV (Positive Predictive Value = Precision) ---
    denom_ppv = tp + fp
    ppv = tp / denom_ppv if denom_ppv > 0 else float("nan")
    ppv_ci = _wilson_ci(tp, denom_ppv)

    # --- NPV (Negative Predictive Value) ---
    denom_npv = tn + fn
    npv = tn / denom_npv if denom_npv > 0 else float("nan")
    npv_ci = _wilson_ci(tn, denom_npv)

    # --- Accuracy ---
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    acc_ci = _wilson_ci(tp + tn, n)

    # --- F1 Score ---
    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom > 0 else float("nan")

    # --- Likelihood Ratios ---
    if not math.isnan(sensitivity) and not math.isnan(specificity):
        lr_pos = sensitivity / (1 - specificity) if specificity < 1.0 else float("inf")
        lr_neg = (1 - sensitivity) / specificity if specificity > 0.0 else float("inf")
    else:
        lr_pos = float("nan")
        lr_neg = float("nan")

    # --- Diagnostic Odds Ratio ---
    if fp > 0 and fn > 0:
        odds_ratio = (tp * tn) / (fp * fn)
    else:
        odds_ratio = float("inf")

    # --- Youden's J ---
    youden_j = (
        sensitivity + specificity - 1.0
        if not (math.isnan(sensitivity) or math.isnan(specificity))
        else float("nan")
    )

    # --- Matthews Correlation Coefficient ---
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else float("nan")

    # --- Cohen's Kappa ---
    if n > 0:
        p_obs = (tp + tn) / n
        p_yes = ((tp + fp) / n) * ((tp + fn) / n)
        p_no = ((fn + tn) / n) * ((fp + tn) / n)
        p_exp = p_yes + p_no
        kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1.0 else float("nan")
    else:
        kappa = float("nan")

    return {
        "sensitivity": sensitivity, "sensitivity_ci": sens_ci,
        "specificity": specificity, "specificity_ci": spec_ci,
        "ppv": ppv, "ppv_ci": ppv_ci,
        "npv": npv, "npv_ci": npv_ci,
        "accuracy": accuracy, "accuracy_ci": acc_ci,
        "f1": f1,
        "lr_pos": lr_pos, "lr_neg": lr_neg, "odds_ratio": odds_ratio,
        "youden_j": youden_j, "mcc": mcc, "kappa": kappa,
    }


def _fmt(val: float, decimals: int = 3) -> str:
    if math.isnan(val):
        return "."
    if math.isinf(val):
        return "∞"
    return str(format_number(val, decimals))


# ──────────────────────────────────────────────────────────────
# 공개 분석 함수
# ──────────────────────────────────────────────────────────────

def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """민감도/특이도 분석을 실행합니다.

    Args:
        dataset: Dataset 객체
        spec: 분석 명세
            variables:
                outcome:    실제 이진 결과 변수명 (0=음성, 1=양성)
                predictor:  검사/예측 이진 변수명
            options:
                pos_label:  양성으로 간주할 값 (기본 1)
                confidence_level: 신뢰수준 (기본 0.95)
            missing_policy: "listwise" (기본)

    Returns:
        AnalysisResult with 5개 테이블
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    confidence_level = float(spec.get("confidence_level", 0.95))
    pos_label = int(options.get("pos_label", 1))

    outcome_var: str = variables.get("outcome", "")
    predictor_var: str = variables.get("predictor", "")

    result = AnalysisResult(
        id="sensitivity_specificity",
        title="Sensitivity / Specificity Analysis",
        spec=spec,
    )

    # --- 변수 검증 ---
    if not outcome_var:
        result.warnings.append("결과 변수(outcome)가 지정되지 않았습니다.")
        return result
    if not predictor_var:
        result.warnings.append("예측 변수(predictor)가 지정되지 않았습니다.")
        return result

    for var in [outcome_var, predictor_var]:
        if var not in dataset.data.columns:
            result.warnings.append(f"변수를 찾을 수 없습니다: '{var}'")
            return result

    # --- 결측 처리 (listwise) ---
    df = dataset.data[[outcome_var, predictor_var]].dropna()
    n_total = len(dataset.data)
    n_valid = len(df)
    n_excluded = n_total - n_valid

    if n_valid == 0:
        result.warnings.append("유효한 케이스가 없습니다.")
        return result

    y_true = df[outcome_var].to_numpy()
    y_pred = df[predictor_var].to_numpy()

    # 이진 검증
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    if len(unique_true) > 2:
        result.warnings.append(
            f"'{outcome_var}' 변수에 범주가 2개 초과입니다 ({len(unique_true)}개). "
            "이진 변수만 지원됩니다."
        )
        return result
    if len(unique_pred) > 2:
        result.warnings.append(
            f"'{predictor_var}' 변수에 범주가 2개 초과입니다 ({len(unique_pred)}개). "
            "이진 변수만 지원됩니다."
        )
        return result

    # --- Table 1: Case Processing Summary ---
    n_pos = int(np.sum(y_true == pos_label))
    n_neg = n_valid - n_pos
    cps_df = pd.DataFrame({
        "": ["Positive cases", "Negative cases", "Valid", "Excluded", "Total"],
        "N": [n_pos, n_neg, n_valid, n_excluded, n_total],
        "%": [
            f"{n_pos/n_valid*100:.1f}%" if n_valid > 0 else ".",
            f"{n_neg/n_valid*100:.1f}%" if n_valid > 0 else ".",
            f"{n_valid/n_total*100:.1f}%" if n_total > 0 else ".",
            f"{n_excluded/n_total*100:.1f}%" if n_total > 0 else ".",
            "100.0%",
        ],
    })
    result.add_table(ResultTable(
        title="Case Processing Summary",
        dataframe=cps_df,
        footnotes=[
            f"Outcome variable: {outcome_var}",
            f"Predictor variable: {predictor_var}",
            f"Positive label: {pos_label}",
        ],
    ))

    # --- 2×2 분할표 계산 ---
    try:
        cells = _compute_2x2(y_true, y_pred, pos_label=pos_label)
    except Exception as exc:
        result.add_warning(f"분할표 계산 오류: {exc}")
        return result
    tp, fp, fn, tn = cells["TP"], cells["FP"], cells["FN"], cells["TN"]

    # --- Table 2: 2×2 Contingency Table ---
    neg_label = [v for v in unique_true if v != pos_label]
    neg_label_str = str(neg_label[0]) if neg_label else "0"
    ct_df = pd.DataFrame(
        {
            f"Predicted Positive ({pos_label})": [tp, fp, tp + fp],
            f"Predicted Negative ({neg_label_str})": [fn, tn, fn + tn],
            "Total": [tp + fn, fp + tn, n_valid],
        },
        index=[
            f"Actual Positive ({pos_label})",
            f"Actual Negative ({neg_label_str})",
            "Total",
        ],
    )
    ct_df.index.name = ""
    result.add_table(ResultTable(
        title="2×2 Contingency Table",
        dataframe=ct_df.reset_index(),
        footnotes=["TP=True Positive, FP=False Positive, FN=False Negative, TN=True Negative"],
    ))

    # --- 지표 계산 ---
    try:
        metrics = _diagnostic_metrics(cells, confidence_level=confidence_level)
    except Exception as exc:
        result.add_warning(f"진단 지표 계산 오류: {exc}")
        return result
    ci_pct = int(confidence_level * 100)

    # --- Table 3: Diagnostic Accuracy ---
    acc_df = pd.DataFrame({
        "Measure": [
            "Sensitivity (Recall)", "Specificity",
            "Positive Predictive Value (PPV)", "Negative Predictive Value (NPV)",
            "Accuracy", "F1 Score",
        ],
        "Value": [
            _fmt(metrics["sensitivity"]),
            _fmt(metrics["specificity"]),
            _fmt(metrics["ppv"]),
            _fmt(metrics["npv"]),
            _fmt(metrics["accuracy"]),
            _fmt(metrics["f1"]),
        ],
        f"{ci_pct}% CI Lower": [
            _fmt(metrics["sensitivity_ci"][0]),
            _fmt(metrics["specificity_ci"][0]),
            _fmt(metrics["ppv_ci"][0]),
            _fmt(metrics["npv_ci"][0]),
            _fmt(metrics["accuracy_ci"][0]),
            ".",
        ],
        f"{ci_pct}% CI Upper": [
            _fmt(metrics["sensitivity_ci"][1]),
            _fmt(metrics["specificity_ci"][1]),
            _fmt(metrics["ppv_ci"][1]),
            _fmt(metrics["npv_ci"][1]),
            _fmt(metrics["accuracy_ci"][1]),
            ".",
        ],
    })
    result.add_table(ResultTable(
        title="Diagnostic Accuracy Measures",
        dataframe=acc_df,
        footnotes=[f"95% CI: Wilson score interval. N={n_valid}."],
    ))

    # --- Table 4: Likelihood Ratios ---
    lr_df = pd.DataFrame({
        "Statistic": [
            "Positive Likelihood Ratio (LR+)",
            "Negative Likelihood Ratio (LR-)",
            "Diagnostic Odds Ratio",
        ],
        "Value": [
            _fmt(metrics["lr_pos"], 2),
            _fmt(metrics["lr_neg"], 3),
            _fmt(metrics["odds_ratio"], 2),
        ],
        "Interpretation": [
            "> 10: strong positive evidence" if metrics["lr_pos"] > 10
            else "> 5: moderate positive evidence" if metrics["lr_pos"] > 5
            else "> 2: weak positive evidence" if metrics["lr_pos"] > 2
            else "minimal evidence",
            "< 0.1: strong negative evidence" if metrics["lr_neg"] < 0.1
            else "< 0.2: moderate negative evidence" if metrics["lr_neg"] < 0.2
            else "< 0.5: weak negative evidence" if metrics["lr_neg"] < 0.5
            else "minimal evidence",
            ".",
        ],
    })
    result.add_table(ResultTable(
        title="Likelihood Ratios",
        dataframe=lr_df,
        footnotes=["LR+ = Sensitivity / (1 - Specificity), LR- = (1 - Sensitivity) / Specificity"],
    ))

    # --- Table 5: Agreement Statistics ---
    kappa_interp = (
        "Almost perfect (≥ 0.81)" if not math.isnan(metrics["kappa"]) and metrics["kappa"] >= 0.81
        else "Substantial (0.61–0.80)" if not math.isnan(metrics["kappa"]) and metrics["kappa"] >= 0.61
        else "Moderate (0.41–0.60)" if not math.isnan(metrics["kappa"]) and metrics["kappa"] >= 0.41
        else "Fair (0.21–0.40)" if not math.isnan(metrics["kappa"]) and metrics["kappa"] >= 0.21
        else "Slight (0.00–0.20)" if not math.isnan(metrics["kappa"]) and metrics["kappa"] >= 0.0
        else "Poor (< 0)"
    )
    agree_df = pd.DataFrame({
        "Statistic": ["Youden's J Index", "Matthews Correlation Coefficient (MCC)", "Cohen's Kappa"],
        "Value": [
            _fmt(metrics["youden_j"]),
            _fmt(metrics["mcc"]),
            _fmt(metrics["kappa"]),
        ],
        "Interpretation": [
            "J = Sensitivity + Specificity - 1 (range: -1 to 1)",
            "MCC = balanced measure for binary classification",
            kappa_interp,
        ],
    })
    result.add_table(ResultTable(
        title="Agreement Statistics",
        dataframe=agree_df,
        footnotes=["Kappa interpretation: Landis & Koch (1977)"],
    ))

    result.notes.append(
        f"분석 변수 — 결과: {outcome_var}, 예측: {predictor_var}. "
        f"양성 기준: {pos_label}. 유효 케이스: {n_valid}."
    )
    return result


def validate(dataset: Dataset, spec: dict) -> list[str]:
    """분석 실행 전 변수 및 옵션 유효성 검사."""
    errors: list[str] = []
    variables = spec.get("variables", {})
    outcome_var = variables.get("outcome", "")
    predictor_var = variables.get("predictor", "")
    if not outcome_var:
        errors.append("결과 변수(outcome)가 필요합니다.")
    if not predictor_var:
        errors.append("예측 변수(predictor)가 필요합니다.")
    for var in [outcome_var, predictor_var]:
        if var and var not in dataset.data.columns:
            errors.append(f"변수를 찾을 수 없습니다: '{var}'")
    return errors
