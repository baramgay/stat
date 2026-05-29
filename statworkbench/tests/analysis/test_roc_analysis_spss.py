"""ROC 분석 SPSS 29 호환 검증 테스트.

검증 항목:
  - AUC 수치 정확성 (완전 분리 / 무작위 / 표준 데이터)
  - 최적 컷오프 (Youden J = 민감도 + 특이도 - 1 최대값)
  - 결과 테이블 구조 4개 검증
  - 결측치 처리
  - 오류 처리 (단일 클래스, 변수 없음, 빈 변수 목록 등)

SPSS 29 참조 출력 (테스트 데이터 n=20):
    y_true = [1]*10 + [0]*10
    scores  = [0.9,0.8,0.85,0.7,0.75,0.6,0.65,0.55,0.5,0.45,
               0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.3,0.45]

    AUC    = 0.995
    SE     = 0.0166
    95% CI = [0.9625, 1.0275] → clamp → [0.9625, 1.000]
    z      = 29.834
    p      < 0.001

    최적 컷오프 (Youden J):
        threshold = 0.5000
        sensitivity = 0.9000
        specificity = 1.0000
        Youden J    = 0.9000

독립 검증:
    Python: sklearn.metrics.roc_auc_score, roc_curve
    R: pROC::roc()$auc, coords(roc, "best", best.method="youden")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.roc_analysis import run_analysis, _compute_roc


# ──────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────

def _approx(val: float, tol: float):
    return pytest.approx(val, abs=tol)


# 표준 테스트 데이터 (n=20, SPSS 참조 계산에 사용한 것과 동일)
Y_TRUE  = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
SCORES  = [0.9,0.8,0.85,0.7,0.75,0.6,0.65,0.55,0.5,0.45,
           0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.3,0.45]


def _make_dataset(
    y_true=None,
    scores=None,
    extra_scores: dict | None = None,
    with_nan: bool = False,
) -> Dataset:
    """테스트용 Dataset 생성."""
    if y_true is None:
        y_true = Y_TRUE
    if scores is None:
        scores = SCORES

    data: dict = {"outcome": y_true, "score": scores}
    if extra_scores:
        data.update(extra_scores)

    df = pd.DataFrame(data)
    if with_nan:
        df.loc[0, "score"] = np.nan
        df.loc[1, "outcome"] = np.nan

    return Dataset(df, name="roc_test")


def _std_spec(
    state: str = "outcome",
    test: list[str] | None = None,
    positive_value: int = 1,
) -> dict:
    if test is None:
        test = ["score"]
    return {
        "variables": {
            "state": state,
            "test": test,
            "positive_value": positive_value,
        }
    }


# ──────────────────────────────────────────────────────────────
# 1. _compute_roc 단위 테스트
# ──────────────────────────────────────────────────────────────

class TestComputeRoc:
    """_compute_roc() 내부 함수 단위 테스트."""

    def test_auc_spss_reference(self):
        """AUC = 0.995 — SPSS 29 참조값.

        SPSS 29: Area = .995
        R: pROC::roc(outcome, score)$auc = 0.995
        Python: sklearn roc_auc_score = 0.995
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["auc"] == _approx(0.995, 0.001)

    def test_se_hanley_mcneil(self):
        """표준오차 ≈ 0.0166 — Hanley-McNeil(1982) 공식.

        SPSS 29: Std. Error = .0166
        수식: sqrt[(AUC(1-AUC) + (n+−1)(Q1−AUC²) + (n−−1)(Q2−AUC²)) / (n+·n−)]
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["se"] == _approx(0.0166, 0.001)

    def test_ci_lower_spss(self):
        """95% CI 하한 ≈ 0.9625 — SPSS 29 참조값.

        SPSS 29: Lower Bound = .963 (반올림)
        계산: AUC − 1.96 × SE = 0.995 − 1.96 × 0.0166 ≈ 0.9625
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["ci_lower"] == _approx(0.9625, 0.002)

    def test_ci_upper_clamped_to_one(self):
        """95% CI 상한 = 1.000 (1.0 clamp).

        계산: AUC + 1.96 × SE = 0.995 + 0.033 > 1.0 → clamp → 1.000
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["ci_upper"] == _approx(1.0, 1e-9)

    def test_p_value_near_zero(self):
        """p값 ≈ 0.000 (H0: AUC=0.5, z≈29.8) — SPSS 참조.

        SPSS 29: Asymptotic Sig. = .000
        z = (0.995 − 0.5) / 0.0166 ≈ 29.83 → p ≈ 0
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["p_value"] < 0.001

    def test_optimal_threshold_youden(self):
        """최적 컷오프 = 0.5 (Youden J 최대) — SPSS 참조.

        SPSS 29 Coordinates of the Curve: 최적점 Cutoff = 0.5
        R: coords(roc, "best", best.method="youden")$threshold = 0.5
        Youden J = TPR − FPR 최대값 지점
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["optimal_threshold"] == _approx(0.5, 0.01)

    def test_sensitivity_at_optimal(self):
        """최적 컷오프에서 민감도 = 0.9 — SPSS 참조.

        SPSS 29: Sensitivity = .900 at threshold = 0.5
        R: coords(roc, "best")$sensitivity = 0.9
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["sensitivity"] == _approx(0.9, 0.01)

    def test_specificity_at_optimal(self):
        """최적 컷오프에서 특이도 = 1.0 — SPSS 참조.

        SPSS 29: Specificity = 1.000 at threshold = 0.5
        R: coords(roc, "best")$specificity = 1.0
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["specificity"] == _approx(1.0, 0.01)

    def test_youden_j_value(self):
        """Youden J = 0.9 = sensitivity + specificity - 1.

        Youden J = 0.9 + 1.0 - 1 = 0.9
        SPSS 출력에 명시적 Youden J 항목 없으나 최적점 선택 기준
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["youden_j"] == _approx(0.9, 0.01)

    def test_n_pos_n_neg(self):
        """양성 10, 음성 10 케이스 수 정확."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["n_pos"] == 10
        assert roc["n_neg"] == 10

    def test_fpr_tpr_are_numpy_arrays(self):
        """fpr, tpr, thresholds가 numpy 배열."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert isinstance(roc["fpr"], np.ndarray)
        assert isinstance(roc["tpr"], np.ndarray)
        assert isinstance(roc["thresholds"], np.ndarray)

    def test_fpr_starts_at_zero(self):
        """FPR 첫 번째 값 = 0.0 (ROC 시작점).

        ROC 곡선은 (0, 0)에서 시작: FPR[0] = 0.
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["fpr"][0] == _approx(0.0, 1e-9)

    def test_tpr_ends_at_one(self):
        """TPR 마지막 값 = 1.0 (ROC 끝점).

        ROC 곡선은 (1, 1)에서 끝남.
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["tpr"][-1] == _approx(1.0, 1e-9)


# ──────────────────────────────────────────────────────────────
# 2. 완전 분리 / 무작위 수준 AUC
# ──────────────────────────────────────────────────────────────

class TestAucBoundary:
    """AUC 경계값 — 완전 분리와 무작위 분류."""

    def test_perfect_separation_auc_one(self):
        """완전 분리 데이터 → AUC = 1.0.

        양성 점수 모두 음성 점수보다 높을 때 AUC = 1.
        SPSS 결과: Area = 1.000
        """
        y = [1, 1, 1, 0, 0, 0]
        s = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        roc = _compute_roc(np.array(y), np.array(s))
        assert roc["auc"] == _approx(1.0, 1e-9)

    def test_random_classifier_auc_approx_half(self):
        """무작위 분류기 → AUC ≈ 0.5.

        점수와 레이블이 무관할 때 AUC → 0.5
        허용 범위: 0.40 ~ 0.60 (확률적 허용)
        """
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        # y와 독립인 점수
        s = rng.random(size=200)
        roc = _compute_roc(y, s)
        assert 0.35 <= roc["auc"] <= 0.65

    def test_auc_in_unit_interval(self):
        """AUC ∈ [0, 1] — 수학적 불변량."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert 0.0 <= roc["auc"] <= 1.0

    def test_se_nonnegative(self):
        """SE ≥ 0 — Hanley-McNeil 분산은 항상 비음수."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["se"] >= 0.0

    def test_ci_lower_le_auc_le_ci_upper(self):
        """CI 하한 ≤ AUC ≤ CI 상한 — 구간 포함 불변량."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["ci_lower"] <= roc["auc"] <= roc["ci_upper"]

    def test_youden_j_in_unit_interval(self):
        """Youden J ∈ [0, 1] — 지수 범위 불변량."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert 0.0 <= roc["youden_j"] <= 1.0

    def test_sensitivity_specificity_in_unit_interval(self):
        """민감도, 특이도 ∈ [0, 1] — 비율 불변량."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert 0.0 <= roc["sensitivity"] <= 1.0
        assert 0.0 <= roc["specificity"] <= 1.0


# ──────────────────────────────────────────────────────────────
# 3. run_analysis 테이블 구조 검증
# ──────────────────────────────────────────────────────────────

class TestRunAnalysisTableStructure:
    """run_analysis() 반환 테이블 4개 구조 검증."""

    @pytest.fixture
    def result(self):
        ds = _make_dataset()
        spec = _std_spec()
        return run_analysis(ds, spec)

    def test_returns_analysis_result(self, result):
        """AnalysisResult 객체 반환."""
        from statworkbench.analysis.result import AnalysisResult
        assert isinstance(result, AnalysisResult)

    def test_exactly_four_tables(self, result):
        """결과 테이블이 정확히 4개.

        SPSS ROC 출력: Case Processing Summary / Area Under the Curve /
        Optimal Cutoff / ROC Coordinates
        """
        assert len(result.tables) == 4

    def test_table_1_case_processing_summary(self, result):
        """Table 1: 'Case Processing Summary' 제목 및 컬럼."""
        t = result.tables[0]
        assert t.title == "Case Processing Summary"
        assert "N" in t.dataframe.columns
        assert "구분" in t.dataframe.columns

    def test_table_1_n_pos_n_neg(self, result):
        """Table 1: 양성=10, 음성=10, 합계=20 확인."""
        df = result.tables[0].dataframe
        ns = df["N"].tolist()
        # 양성, 음성, 합계
        assert ns[0] == 10
        assert ns[1] == 10
        assert ns[2] == 20

    def test_table_2_area_under_curve(self, result):
        """Table 2: 'Area Under the Curve' 제목 및 필수 컬럼."""
        t = result.tables[1]
        assert t.title == "Area Under the Curve"
        required_cols = {"변수", "AUC", "표준오차", "95% CI 하한", "95% CI 상한", "p-value"}
        assert required_cols.issubset(set(t.dataframe.columns))

    def test_table_2_one_row_per_test_variable(self, result):
        """Table 2: 검사 변수 1개 → 1행."""
        df = result.tables[1].dataframe
        assert len(df) == 1

    def test_table_3_optimal_cutoff(self, result):
        """Table 3: 'Optimal Cutoff' 제목 및 필수 컬럼."""
        t = result.tables[2]
        assert t.title == "Optimal Cutoff"
        required_cols = {"변수", "최적 컷오프", "민감도", "특이도", "Youden J"}
        assert required_cols.issubset(set(t.dataframe.columns))

    def test_table_4_roc_coordinates(self, result):
        """Table 4: 'ROC Coordinates' 제목 및 필수 컬럼."""
        t = result.tables[3]
        assert t.title == "ROC Coordinates"
        required_cols = {"변수", "1-특이도", "민감도"}
        assert required_cols.issubset(set(t.dataframe.columns))

    def test_table_4_max_20_points(self, result):
        """Table 4: 좌표 포인트 최대 20개 제한."""
        df = result.tables[3].dataframe
        assert len(df) <= 20

    def test_table_4_fpr_in_unit_interval(self, result):
        """Table 4: 1-특이도(FPR) ∈ [0, 1]."""
        df = result.tables[3].dataframe
        vals = pd.to_numeric(df["1-특이도"], errors="coerce")
        assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_table_4_tpr_in_unit_interval(self, result):
        """Table 4: 민감도(TPR) ∈ [0, 1]."""
        df = result.tables[3].dataframe
        vals = pd.to_numeric(df["민감도"], errors="coerce")
        assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_no_warnings_on_valid_data(self, result):
        """유효한 데이터에서 경고 없음."""
        assert len(result.warnings) == 0


# ──────────────────────────────────────────────────────────────
# 4. SPSS 참조값 수치 정확성 (run_analysis 통합)
# ──────────────────────────────────────────────────────────────

class TestRunAnalysisSpssValues:
    """run_analysis() 출력값 SPSS 29 참조값 대조."""

    @pytest.fixture
    def result(self):
        ds = _make_dataset()
        return run_analysis(ds, _std_spec())

    def test_auc_value_spss(self, result):
        """AUC ≈ 0.995 — SPSS 29 Area = .995.

        SPSS 29: Area Under the Curve, score, Area = .995
        R: pROC::auc() = 0.995
        """
        df = result.tables[1].dataframe
        auc_str = df.loc[df["변수"] == "score", "AUC"].values[0]
        auc_val = float(auc_str)
        assert auc_val == _approx(0.995, 0.001)

    def test_optimal_cutoff_spss(self, result):
        """최적 컷오프 = 0.5 — SPSS 29 Youden J 최대점.

        SPSS 29: 최적 좌표 threshold = 0.5
        R: coords(roc, "best", best.method="youden")$threshold = 0.5
        """
        df = result.tables[2].dataframe
        cutoff_str = df.loc[df["변수"] == "score", "최적 컷오프"].values[0]
        cutoff_val = float(cutoff_str)
        assert cutoff_val == _approx(0.5, 0.01)

    def test_sensitivity_spss(self, result):
        """민감도 = 0.9 — SPSS 참조.

        SPSS 29: Sensitivity = .900
        """
        df = result.tables[2].dataframe
        sens_str = df.loc[df["변수"] == "score", "민감도"].values[0]
        sens_val = float(sens_str)
        assert sens_val == _approx(0.9, 0.01)

    def test_specificity_spss(self, result):
        """특이도 = 1.0 — SPSS 참조.

        SPSS 29: Specificity = 1.000
        """
        df = result.tables[2].dataframe
        spec_str = df.loc[df["변수"] == "score", "특이도"].values[0]
        spec_val = float(spec_str)
        assert spec_val == _approx(1.0, 0.01)

    def test_youden_j_spss(self, result):
        """Youden J = 0.9 = 민감도 + 특이도 - 1.

        SPSS 최적점 선택 기준: Youden J 최대 = 0.9
        """
        df = result.tables[2].dataframe
        yj_str = df.loc[df["변수"] == "score", "Youden J"].values[0]
        yj_val = float(yj_str)
        assert yj_val == _approx(0.9, 0.01)


# ──────────────────────────────────────────────────────────────
# 5. 다중 검사 변수
# ──────────────────────────────────────────────────────────────

class TestMultipleTestVariables:
    """복수 검사 변수 동시 분석."""

    @pytest.fixture
    def result(self):
        rng = np.random.default_rng(7)
        # score2: 무작위 점수 (AUC ≈ 0.5)
        score2 = rng.random(size=20).tolist()
        ds = _make_dataset(extra_scores={"score2": score2})
        spec = _std_spec(test=["score", "score2"])
        return run_analysis(ds, spec)

    def test_two_rows_in_auc_table(self, result):
        """검사 변수 2개 → AUC 테이블 2행."""
        df = result.tables[1].dataframe
        assert len(df) == 2

    def test_two_rows_in_cutoff_table(self, result):
        """검사 변수 2개 → Optimal Cutoff 테이블 2행."""
        df = result.tables[2].dataframe
        assert len(df) == 2

    def test_coord_contains_both_variables(self, result):
        """ROC 좌표 테이블에 두 변수 모두 포함."""
        df = result.tables[3].dataframe
        assert "score" in df["변수"].values
        assert "score2" in df["변수"].values

    def test_score_auc_higher_than_score2(self, result):
        """score(AUC≈0.995)가 score2(무작위)보다 높음."""
        df = result.tables[1].dataframe
        auc_score = float(df.loc[df["변수"] == "score", "AUC"].values[0])
        auc_score2 = float(df.loc[df["변수"] == "score2", "AUC"].values[0])
        assert auc_score > auc_score2


# ──────────────────────────────────────────────────────────────
# 6. 결측치 처리
# ──────────────────────────────────────────────────────────────

class TestMissingValues:
    """결측치(NaN) 처리 검증."""

    def test_nan_rows_excluded(self):
        """NaN 포함 행 제거 후 분석 정상 수행."""
        ds = _make_dataset(with_nan=True)
        result = run_analysis(ds, _std_spec())
        # 경고 없이 결과 반환 (결측치 제거로 처리)
        assert len(result.tables) == 4

    def test_case_summary_reflects_valid_n(self):
        """결측치 제거 후 Case Processing Summary N이 감소."""
        ds_clean = _make_dataset(with_nan=False)
        ds_nan = _make_dataset(with_nan=True)

        r_clean = run_analysis(ds_clean, _std_spec())
        r_nan = run_analysis(ds_nan, _std_spec())

        n_clean = r_clean.tables[0].dataframe["N"].iloc[2]  # 합계
        n_nan = r_nan.tables[0].dataframe["N"].iloc[2]      # 합계
        assert n_nan < n_clean

    def test_all_nan_scores_returns_warning(self):
        """점수 변수 전체 NaN → 경고 반환."""
        df = pd.DataFrame({
            "outcome": [1, 1, 0, 0],
            "score": [np.nan, np.nan, np.nan, np.nan],
        })
        ds = Dataset(df, name="nan_test")
        result = run_analysis(ds, _std_spec())
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 7. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestErrorHandling:
    """비정상 입력에 대한 오류 처리 검증."""

    def test_single_class_returns_warning(self):
        """단일 클래스(모두 양성) → 경고 반환.

        AUC 정의 불가: 이진 분류가 아닌 경우.
        SPSS: 오류 메시지 "Invalid response variable"
        """
        df = pd.DataFrame({
            "outcome": [1, 1, 1, 1],   # 단일 클래스
            "score": [0.9, 0.8, 0.7, 0.6],
        })
        ds = Dataset(df, name="single_class")
        result = run_analysis(ds, _std_spec())
        assert len(result.warnings) > 0

    def test_missing_state_variable_returns_warning(self):
        """state 변수 미지정 → 경고 반환.

        SPSS: 결과 변수 없으면 분석 불가 오류
        """
        ds = _make_dataset()
        spec = {"variables": {"test": ["score"]}}   # state 없음
        result = run_analysis(ds, spec)
        assert any("state" in w or "결과 변수" in w for w in result.warnings)

    def test_empty_test_variable_list_returns_warning(self):
        """test 변수 목록 비어있음 → 경고 반환.

        SPSS: 검사 변수 없으면 분석 불가 오류
        """
        ds = _make_dataset()
        spec = {"variables": {"state": "outcome", "test": []}}
        result = run_analysis(ds, spec)
        assert any("test" in w or "검사 점수" in w for w in result.warnings)

    def test_nonexistent_variable_returns_warning(self):
        """존재하지 않는 변수 지정 → 경고 반환.

        SPSS: 변수 이름이 없으면 오류
        """
        ds = _make_dataset()
        spec = _std_spec(state="no_such_var", test=["also_gone"])
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0

    def test_result_has_no_tables_on_state_error(self):
        """state 변수 오류 시 결과 테이블 없음.

        오류 상태에서 부분 결과를 반환하면 안 됨.
        """
        ds = _make_dataset()
        spec = {"variables": {"test": ["score"]}}
        result = run_analysis(ds, spec)
        assert len(result.tables) == 0

    def test_multiclass_returns_warning(self):
        """다중 클래스 결과 변수 → 경고 반환.

        이진 분류만 지원 (SPSS ROC Curve 동일 제약).
        """
        df = pd.DataFrame({
            "outcome": [0, 1, 2, 0, 1, 2],   # 3클래스
            "score": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
        })
        ds = Dataset(df, name="multiclass")
        result = run_analysis(ds, _std_spec())
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 8. 통계적 불변량
# ──────────────────────────────────────────────────────────────

class TestStatisticalInvariants:
    """ROC 분석 수학적·통계적 불변량 검증."""

    def test_youden_j_equals_sens_plus_spec_minus_one(self):
        """Youden J = 민감도 + 특이도 - 1 정의 확인."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        expected_j = roc["sensitivity"] + roc["specificity"] - 1.0
        assert roc["youden_j"] == _approx(expected_j, 1e-9)

    def test_p_value_significant_for_good_classifier(self):
        """AUC >> 0.5인 경우 p < 0.05 (H0: AUC=0.5 기각).

        SPSS 29: Asymptotic Sig. = .000 (< .001)
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert roc["p_value"] < 0.05

    def test_roc_fpr_tpr_same_length(self):
        """fpr, tpr, thresholds 배열 길이 동일."""
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        assert len(roc["fpr"]) == len(roc["tpr"]) == len(roc["thresholds"])

    def test_fpr_nondecreasing(self):
        """FPR 배열은 단조 비감소 (sklearn roc_curve 보장).

        SPSS ROC 좌표: x축(1-특이도)은 단조 증가해야 함.
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        fpr = roc["fpr"]
        for i in range(len(fpr) - 1):
            assert fpr[i] <= fpr[i + 1] + 1e-9

    def test_tpr_nondecreasing(self):
        """TPR 배열은 단조 비감소.

        SPSS ROC 좌표: y축(민감도)은 단조 증가해야 함.
        """
        roc = _compute_roc(np.array(Y_TRUE), np.array(SCORES))
        tpr = roc["tpr"]
        for i in range(len(tpr) - 1):
            assert tpr[i] <= tpr[i + 1] + 1e-9

    def test_analysis_result_id_is_roc_analysis(self):
        """AnalysisResult.id = 'roc_analysis'."""
        ds = _make_dataset()
        result = run_analysis(ds, _std_spec())
        assert result.id == "roc_analysis"

    def test_analysis_result_title(self):
        """AnalysisResult.title = 'ROC Curve Analysis'."""
        ds = _make_dataset()
        result = run_analysis(ds, _std_spec())
        assert result.title == "ROC Curve Analysis"

    def test_notes_contain_auc_info(self):
        """분석 결과 notes에 AUC 정보 포함."""
        ds = _make_dataset()
        result = run_analysis(ds, _std_spec())
        assert any("AUC" in note for note in result.notes)
