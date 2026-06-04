"""다항 로지스틱 회귀(Multinomial Logistic Regression) 분석 모듈.

SPSS: Analyze > Regression > Multinomial Logistic

지원 기능:
  - MNLogit (statsmodels) 기반 추정
  - 기준 범주(Reference Category) 선택
  - 모형 적합도: 로그우도, Pseudo R² (Nagelkerke/Cox-Snell/McFadden)
  - 우도비 검정 (Likelihood Ratio Test)
  - 모수 추정값: B, SE, Wald, df, p, Exp(B), 95% CI
  - 분류표 (Classification Table)
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
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """다항 로지스틱 회귀를 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.dependent:     종속변수 (범주형, 3개 이상 범주)
            variables.predictors:    예측변수 목록
            options.reference:       기준 범주 (기본: 마지막 범주)
            options.confidence_level: 신뢰수준 (기본 0.95)
            options.classification:  True=분류표 (기본 True)
            missing_policy:          결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. 종속변수 범주 분포
            3. 모형 적합도
            4. 모수 추정값 (범주별)
            5. 분류표 (선택)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    missing_policy_str = spec.get("missing_policy", "listwise")

    dep_var: str = variables.get("dependent", "")
    predictors: list[str] = variables.get("predictors", [])
    ref_cat = options.get("reference", None)
    ci_level: float = float(options.get("confidence_level", 0.95))
    do_classification: bool = options.get("classification", True)

    result = AnalysisResult(id="multinomial_logistic", title="다항 로지스틱 회귀 (Multinomial Logistic Regression)")

    # ── 입력 검증 ─────────────────────────────────────────────────────────────
    if dataset.data is None or not dep_var or not predictors:
        result.add_warning("종속변수와 예측변수를 지정하세요.")
        return result
    missing_cols = [v for v in [dep_var] + predictors if v not in dataset.data.columns]
    if missing_cols:
        result.add_warning(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────────────────
    try:
        mp = MissingPolicy(missing_policy_str) if isinstance(missing_policy_str, str) else missing_policy_str
    except ValueError:
        mp = MissingPolicy.LISTWISE

    all_vars = [dep_var] + predictors
    paf = prepare_analysis_frame(dataset, all_vars, missing_policy=mp)
    df = paf.data[all_vars].copy()
    df = df.dropna()
    N = len(df)

    result.add_table(get_cps_table_kr(paf.n_total, N, paf.n_total - N))

    categories = sorted(df[dep_var].astype(str).unique())
    n_cats = len(categories)

    if n_cats < 3:
        result.add_warning(f"종속변수 범주가 {n_cats}개입니다. 다항 로지스틱은 3개 이상 필요합니다.")
        return result
    if N < len(predictors) * n_cats + 1:
        result.add_warning(f"케이스 수({N})가 너무 적습니다.")
        return result

    # ── 종속변수 범주 분포 ────────────────────────────────────────────────────
    cat_counts = df[dep_var].astype(str).value_counts().sort_index()
    cat_rows = [
        {"범주": cat, "빈도": int(cat_counts.get(cat, 0)),
         "비율 (%)": format_number(cat_counts.get(cat, 0) / N * 100, 2)}
        for cat in categories
    ]
    result.add_table(ResultTable(
        title="종속변수 범주 분포",
        dataframe=pd.DataFrame(cat_rows),
    ))

    # ── MNLogit 추정 ──────────────────────────────────────────────────────────
    try:
        import statsmodels.api as sm

        # 기준 범주 설정
        if ref_cat is None:
            ref_cat = categories[-1]
        ref_cat_str = str(ref_cat)
        if ref_cat_str not in categories:
            ref_cat_str = categories[-1]

        # 종속변수 인코딩 (기준 범주 제외 후 나머지 순서 정렬)
        non_ref = [c for c in categories if c != ref_cat_str]
        y_series = df[dep_var].astype(str)

        # 예측변수 행렬 (연속형은 그대로, 명목형은 더미)
        X_df = df[predictors].copy()
        for col in predictors:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df = X_df.dropna()
        y_series = y_series.loc[X_df.index]
        X = sm.add_constant(X_df.values.astype(float))

        # y를 정수 코드로: 기준=0, 나머지=1,2,...
        cat_order = [ref_cat_str] + non_ref
        y_int = y_series.map({c: i for i, c in enumerate(cat_order)}).values.astype(int)

        from statsmodels.discrete.discrete_model import MNLogit
        model = MNLogit(y_int, X)
        fit = model.fit(method="newton", disp=False, maxiter=200)

    except Exception as exc:
        result.add_warning(f"모형 추정 실패: {exc}")
        return result

    # ── 모형 적합도 ───────────────────────────────────────────────────────────
    ll_full = float(fit.llf)
    ll_null = float(fit.llnull)
    chi2_lr = -2 * (ll_null - ll_full)
    df_lr = fit.df_model
    p_lr = float(1 - stats.chi2.cdf(chi2_lr, df_lr))

    n = float(N)
    cox_snell = 1 - np.exp((2 / n) * (ll_null - ll_full))
    _nag_denom = 1 - np.exp(2 * ll_null / n)
    nagelkerke = cox_snell / _nag_denom if abs(_nag_denom) > 1e-12 else float("nan")
    mcfadden = 1 - ll_full / ll_null if ll_null != 0 else float("nan")

    fit_rows = [
        {"지표": "-2 로그우도 (절편만)", "값": format_number(-2 * ll_null, 3)},
        {"지표": "-2 로그우도 (최종 모형)", "값": format_number(-2 * ll_full, 3)},
        {"지표": "우도비 χ²", "값": format_number(chi2_lr, 3),
         "df": str(int(df_lr)), "p": format_pvalue(p_lr)},
        {"지표": "Cox-Snell R²", "값": format_number(float(cox_snell), 4), "df": "", "p": ""},
        {"지표": "Nagelkerke R²", "값": format_number(float(nagelkerke), 4), "df": "", "p": ""},
        {"지표": "McFadden R²", "값": format_number(float(mcfadden), 4), "df": "", "p": ""},
    ]
    result.add_table(ResultTable(title="모형 적합도", dataframe=pd.DataFrame(fit_rows)))

    # ── 모수 추정값 ───────────────────────────────────────────────────────────
    alpha = 1 - ci_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    pred_names = ["(상수)"] + list(predictors)

    # fit.params: shape (k_vars, n_cats-1); fit.tvalues/pvalues 동일
    params = np.array(fit.params)   # (k, J-1)
    bse = np.array(fit.bse)
    tvals = np.array(fit.tvalues)
    pvals = np.array(fit.pvalues)

    # statsmodels MNLogit columns correspond to non-reference categories
    param_rows = []
    for j, cat in enumerate(non_ref):
        for k, pname in enumerate(pred_names):
            b = float(params[k, j]) if params.ndim == 2 else float(params[k])
            se = float(bse[k, j]) if bse.ndim == 2 else float(bse[k])
            wald = float(tvals[k, j]) ** 2 if tvals.ndim == 2 else float(tvals[k]) ** 2
            pv = float(pvals[k, j]) if pvals.ndim == 2 else float(pvals[k])
            exp_b = np.exp(b)
            lo = np.exp(b - z_crit * se)
            hi = np.exp(b + z_crit * se)
            param_rows.append({
                "비교 범주": f"{cat} vs {ref_cat_str}",
                "변수": pname,
                "B": format_number(b, 4),
                "SE": format_number(se, 4),
                "Wald": format_number(wald, 3),
                "df": "1",
                "p": format_pvalue(pv),
                "Exp(B)": format_number(float(exp_b), 4),
                f"CI 하한 ({int(ci_level*100)}%)": format_number(float(lo), 4),
                f"CI 상한 ({int(ci_level*100)}%)": format_number(float(hi), 4),
            })

    result.add_table(ResultTable(
        title=f"모수 추정값 (기준 범주: {ref_cat_str})",
        dataframe=pd.DataFrame(param_rows),
        footnotes=[f"신뢰수준 {int(ci_level*100)}%, 기준 범주: {ref_cat_str}"],
    ))

    # ── 분류표 ────────────────────────────────────────────────────────────────
    if do_classification:
        pred_proba = np.array(fit.predict())  # (N, n_cats)
        pred_class_idx = np.argmax(pred_proba, axis=1)
        pred_labels = np.array(cat_order)[pred_class_idx]
        actual_labels = y_series.values

        cls_rows = []
        for actual_cat in categories:
            mask = actual_labels == actual_cat
            row: dict = {"실제 범주": actual_cat}
            for pred_cat in categories:
                row[f"예측: {pred_cat}"] = int(np.sum((pred_labels == pred_cat) & mask))
            n_actual = int(np.sum(mask))
            n_correct = int(np.sum((pred_labels == actual_cat) & mask))
            row["정분류율 (%)"] = format_number(n_correct / n_actual * 100 if n_actual > 0 else 0, 1)
            cls_rows.append(row)

        total_correct = int(np.sum(pred_labels == actual_labels))
        overall_row: dict = {"실제 범주": "전체 정분류율"}
        for pred_cat in categories:
            overall_row[f"예측: {pred_cat}"] = ""
        overall_row["정분류율 (%)"] = format_number(total_correct / N * 100, 1)
        cls_rows.append(overall_row)

        result.add_table(ResultTable(title="분류표 (Classification Table)", dataframe=pd.DataFrame(cls_rows)))

    result.notes.extend([
        f"유효 케이스 N={N}, 종속변수 범주 수={n_cats}",
        f"기준 범주: {ref_cat_str}",
        f"예측변수: {', '.join(predictors)}",
    ])
    return result
