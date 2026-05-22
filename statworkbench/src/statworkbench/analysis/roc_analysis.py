"""ROC Curve Analysis — SPSS Analyze > ROC Curve 대응 모듈.

SPSS 29 기준 출력 테이블 4개:
  1. Case Processing Summary  — 양성/음성 케이스 수
  2. Area Under the Curve     — AUC, SE, 95% CI, p-value
  3. Optimal Cutoff           — Youden J 기반 최적 컷오프
  4. ROC Coordinates          — (1-특이도, 민감도) 좌표 (최대 20포인트)

AUC 표준오차: Hanley-McNeil (1982) 공식
p-value: H0: AUC = 0.5 단일 표본 z-검정
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import format_number, format_pvalue

# 최대 ROC 좌표 출력 포인트 수 (SPSS 기본값과 동일)
_MAX_COORD_POINTS = 20


# ──────────────────────────────────────────────────────────────
# 내부 계산 함수
# ──────────────────────────────────────────────────────────────

def _compute_roc(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """ROC 분석 핵심 지표를 계산합니다.

    Args:
        y_true: 이진 실제 레이블 (0/1)
        scores: 예측 점수 (연속형)

    Returns:
        dict with keys:
            fpr, tpr, thresholds (numpy arrays)
            auc (float)
            se (float)           — Hanley-McNeil 표준오차
            ci_lower, ci_upper   — 95% 신뢰구간
            z_stat (float)       — z = (AUC - 0.5) / SE
            p_value (float)      — H0: AUC = 0.5
            optimal_threshold (float)   — Youden J 최대값 임계값
            sensitivity (float)         — 최적 임계값에서의 민감도
            specificity (float)         — 최적 임계값에서의 특이도
            youden_j (float)            — Youden J 지수
            n_pos (int)
            n_neg (int)
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Youden J = sensitivity + specificity - 1 = TPR - FPR
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    optimal_threshold = float(thresholds[best_idx])
    sensitivity = float(tpr[best_idx])
    specificity = float(1.0 - fpr[best_idx])
    youden_j = float(youden[best_idx])

    # Hanley-McNeil (1982) 표준오차
    Q1 = auc / (2.0 - auc)
    Q2 = 2.0 * auc ** 2 / (1.0 + auc)
    se_sq = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (Q1 - auc ** 2)
        + (n_neg - 1) * (Q2 - auc ** 2)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(se_sq, 0.0)))

    ci_lower = float(max(auc - 1.96 * se, 0.0))
    ci_upper = float(min(auc + 1.96 * se, 1.0))

    # H0: AUC = 0.5 단일 표본 z-검정
    if se > 0:
        z_stat = (auc - 0.5) / se
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    else:
        z_stat = np.inf
        p_value = 0.0

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "z_stat": float(z_stat),
        "p_value": p_value,
        "optimal_threshold": optimal_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def _downsample_coords(fpr: np.ndarray, tpr: np.ndarray, max_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """ROC 좌표를 최대 max_pts개로 균등 다운샘플링."""
    n = len(fpr)
    if n <= max_pts:
        return fpr, tpr
    idx = np.round(np.linspace(0, n - 1, max_pts)).astype(int)
    return fpr[idx], tpr[idx]


# ──────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────

def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """ROC 분석을 수행합니다 (SPSS Analyze > ROC Curve 대응).

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.state:          이진 결과 변수명 (0/1)
            variables.test:           검사 점수 변수 목록
            variables.positive_value: 양성 판정값 (기본 1)
            options.max_coords:       ROC 좌표 최대 포인트 수 (기본 20)

    Returns:
        AnalysisResult — 테이블 4개 포함
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})

    state_var: str | None = variables.get("state")
    test_vars: list[str] = variables.get("test", [])
    positive_value = variables.get("positive_value", 1)
    max_coords: int = int(options.get("max_coords", _MAX_COORD_POINTS))

    result = AnalysisResult(id="roc_analysis", title="ROC Curve Analysis")

    # ── 입력 검증 ────────────────────────────────────────────
    if not state_var:
        result.warnings.append("결과 변수(state)를 지정해야 합니다.")
        return result

    if not test_vars:
        result.warnings.append("검사 점수 변수(test)를 최소 1개 지정해야 합니다.")
        return result

    missing_cols = [v for v in [state_var] + test_vars if v not in dataset.data.columns]
    if missing_cols:
        result.warnings.append(f"변수를 찾을 수 없습니다: {missing_cols}")
        return result

    # ── 데이터 준비 ──────────────────────────────────────────
    all_cols = [state_var] + test_vars
    data = dataset.data[all_cols].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) == 0:
        result.warnings.append("결측치 제거 후 유효한 케이스가 없습니다.")
        return result

    # 이진 레이블 변환
    y_raw = data[state_var]
    unique_vals = sorted(y_raw.unique())
    if len(unique_vals) < 2:
        result.warnings.append("결과 변수에 두 가지 이상의 값이 있어야 합니다 (단일 클래스).")
        return result
    if len(unique_vals) > 2:
        result.warnings.append(
            f"결과 변수의 고유값이 2개를 초과합니다: {unique_vals}. "
            "이진 분류만 지원합니다."
        )
        return result

    # positive_value를 1, 나머지를 0으로 변환
    if positive_value not in unique_vals:
        positive_value = unique_vals[1]  # 두 번째 값을 양성으로 간주
        result.warnings.append(
            f"지정한 양성값이 데이터에 없어 {positive_value}를 양성으로 사용합니다."
        )

    y_true = (y_raw == positive_value).astype(int).values
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    # ── Table 1: Case Processing Summary ─────────────────────
    cps_df = pd.DataFrame({
        "구분": ["양성 (Positive)", "음성 (Negative)", "합계"],
        "N": [n_pos, n_neg, n_pos + n_neg],
    })
    result.tables.append(ResultTable(
        title="Case Processing Summary",
        dataframe=cps_df,
        footnotes=[f"양성 판정값: {positive_value}"],
    ))

    # ── 변수별 ROC 계산 ───────────────────────────────────────
    auc_rows: list[dict] = []
    cutoff_rows: list[dict] = []
    coord_frames: list[pd.DataFrame] = []

    for test_var in test_vars:
        scores = data[test_var].values

        try:
            roc = _compute_roc(y_true, scores)
        except ValueError as exc:
            result.warnings.append(f"{test_var}: ROC 계산 오류 — {exc}")
            continue

        # AUC 테이블 행
        auc_rows.append({
            "변수": test_var,
            "AUC": format_number(roc["auc"], 3),
            "표준오차": format_number(roc["se"], 4),
            "95% CI 하한": format_number(roc["ci_lower"], 3),
            "95% CI 상한": format_number(roc["ci_upper"], 3),
            "p값": format_pvalue(roc["p_value"]),
        })

        # 최적 컷오프 테이블 행
        cutoff_rows.append({
            "변수": test_var,
            "최적 컷오프": format_number(roc["optimal_threshold"], 4),
            "민감도": format_number(roc["sensitivity"], 4),
            "특이도": format_number(roc["specificity"], 4),
            "Youden J": format_number(roc["youden_j"], 4),
        })

        # ROC 좌표 (다운샘플링)
        fpr_ds, tpr_ds = _downsample_coords(roc["fpr"], roc["tpr"], max_coords)
        coord_df = pd.DataFrame({
            "변수": test_var,
            "1-특이도": np.round(fpr_ds, 4),
            "민감도": np.round(tpr_ds, 4),
        })
        coord_frames.append(coord_df)

        # 해석 메모
        auc_val = roc["auc"]
        if auc_val >= 0.9:
            grade = "우수 (Excellent)"
        elif auc_val >= 0.8:
            grade = "양호 (Good)"
        elif auc_val >= 0.7:
            grade = "적정 (Fair)"
        elif auc_val >= 0.6:
            grade = "불량 (Poor)"
        else:
            grade = "실패 (Fail — 무작위 수준)"
        result.notes.append(
            f"{test_var}: AUC = {format_number(auc_val, 3)} [{grade}] | "
            f"최적 컷오프 = {format_number(roc['optimal_threshold'], 4)}, "
            f"민감도 = {format_number(roc['sensitivity'], 4)}, "
            f"특이도 = {format_number(roc['specificity'], 4)}"
        )

    # ── Table 2: Area Under the Curve ────────────────────────
    if auc_rows:
        auc_df = pd.DataFrame(auc_rows)
        result.tables.append(ResultTable(
            title="Area Under the Curve",
            dataframe=auc_df,
            footnotes=["AUC 표준오차: Hanley-McNeil(1982). p값: H0: AUC=0.5 단일 표본 z-검정."],
        ))

    # ── Table 3: Optimal Cutoff ──────────────────────────────
    if cutoff_rows:
        cutoff_df = pd.DataFrame(cutoff_rows)
        result.tables.append(ResultTable(
            title="Optimal Cutoff",
            dataframe=cutoff_df,
            footnotes=["최적 컷오프: Youden J(= 민감도 + 특이도 - 1) 최대값 기준."],
        ))

    # ── Table 4: ROC Coordinates ─────────────────────────────
    if coord_frames:
        all_coords = pd.concat(coord_frames, ignore_index=True)
        result.tables.append(ResultTable(
            title="ROC Coordinates",
            dataframe=all_coords,
            footnotes=[f"각 변수별 최대 {max_coords}개 좌표 포인트."],
        ))

    return result
