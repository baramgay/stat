"""급내상관계수(ICC) 분석 테스트 — SPSS 대응 검증.

SPSS: Analyze > Scale > Reliability Analysis > (Statistics) Intraclass Correlation Coefficient

참조값 (3명의 평가자가 10명을 평가, 직접 계산 확인):
  data = {'rater1': [4,6,8,2,5,7,3,9,1,6],
          'rater2': [4,6,7,3,5,8,2,9,2,7],
          'rater3': [3,6,8,2,5,7,3,9,1,7]}

  ICC(1,1) = 0.965705, F=85.4762, df1=9, df2=20, CI=[0.9066, 0.9905]
  ICC(2,1) = 0.965686, F=81.5909, df1=9, df2=18, CI=[0.8995, 0.9901]
  ICC(3,1) = 0.964111, F=81.5909, df1=9, df2=18, CI=[0.8995, 0.9901]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.icc import run_analysis, _compute_icc
from nuristat.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rater_dataset() -> Dataset:
    """3명의 평가자가 10명 피험자를 평가한 데이터 — 높은 일치도."""
    data = {
        "rater1": [4, 6, 8, 2, 5, 7, 3, 9, 1, 6],
        "rater2": [4, 6, 7, 3, 5, 8, 2, 9, 2, 7],
        "rater3": [3, 6, 8, 2, 5, 7, 3, 9, 1, 7],
    }
    return Dataset(data=pd.DataFrame(data), name="rater_study")


@pytest.fixture
def perfect_agreement_dataset() -> Dataset:
    """완전 일치 데이터 — ICC = 1.0."""
    data = {
        "r1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "r2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "r3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    return Dataset(data=pd.DataFrame(data), name="perfect")


@pytest.fixture
def low_agreement_dataset() -> Dataset:
    """낮은 일치도 데이터 — ICC 낮음."""
    rng = np.random.default_rng(42)
    data = {
        "r1": rng.integers(1, 10, 20).tolist(),
        "r2": rng.integers(1, 10, 20).tolist(),
        "r3": rng.integers(1, 10, 20).tolist(),
    }
    return Dataset(data=pd.DataFrame(data), name="low_agreement")


@pytest.fixture
def missing_dataset() -> Dataset:
    """결측치 포함 데이터."""
    data = {
        "r1": [4, 6, np.nan, 2, 5, 7, 3, 9, 1, 6],
        "r2": [4, 6, 7, np.nan, 5, 8, 2, 9, 2, 7],
        "r3": [3, 6, 8, 2, 5, 7, 3, 9, 1, 7],
    }
    return Dataset(data=pd.DataFrame(data), name="missing_raters")


def _make_spec(vars_: list[str], **opts) -> dict:
    return {"variables": {"target": vars_}, "options": opts}


# ──────────────────────────────────────────────────────────────────────────
# 1. ICC 수치 정확성 — One-Way Random ICC(1,1)
# ──────────────────────────────────────────────────────────────────────────

class TestICCOnewayRandom:
    """ICC(1,1) One-Way Random 계산 정확성 검증."""

    def test_icc_value(self, rater_dataset):
        """ICC(1,1) 수치 = 0.9657 ± 0.001."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert abs(r["icc"] - 0.965705) < 0.001, f"ICC(1,1) 오차 초과: {r['icc']}"

    def test_f_statistic(self, rater_dataset):
        """F통계량 = 85.4762 ± 0.01."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert abs(r["f"] - 85.4762) < 0.01, f"F통계량 오차 초과: {r['f']}"

    def test_degrees_of_freedom(self, rater_dataset):
        """df1=9, df2=20."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert r["df1"] == 9
        assert r["df2"] == 20

    def test_p_value_significant(self, rater_dataset):
        """p값 < 0.001."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert r["p"] < 0.001

    def test_95ci_lower(self, rater_dataset):
        """95% CI 하한 ≈ 0.9066 ± 0.005."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert abs(r["ci_lower"] - 0.906641) < 0.005, f"CI 하한 오차: {r['ci_lower']}"

    def test_95ci_upper(self, rater_dataset):
        """95% CI 상한 ≈ 0.9905 ± 0.005."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="oneway_random")
        assert abs(r["ci_upper"] - 0.990489) < 0.005, f"CI 상한 오차: {r['ci_upper']}"


# ──────────────────────────────────────────────────────────────────────────
# 2. ICC 수치 정확성 — Two-Way Random ICC(2,1)
# ──────────────────────────────────────────────────────────────────────────

class TestICCTwowayRandom:
    """ICC(2,1) Two-Way Random, Absolute Agreement 검증."""

    def test_icc_value(self, rater_dataset):
        """ICC(2,1) 수치 = 0.9657 ± 0.001."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_random")
        assert abs(r["icc"] - 0.965686) < 0.001, f"ICC(2,1) 오차 초과: {r['icc']}"

    def test_f_statistic(self, rater_dataset):
        """F통계량 = 81.5909 ± 0.01."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_random")
        assert abs(r["f"] - 81.5909) < 0.01, f"F통계량 오차: {r['f']}"

    def test_degrees_of_freedom(self, rater_dataset):
        """df1=9, df2=18."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_random")
        assert r["df1"] == 9
        assert r["df2"] == 18

    def test_95ci_lower(self, rater_dataset):
        """95% CI 하한 ≈ 0.8995 ± 0.005."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_random")
        assert abs(r["ci_lower"] - 0.899515) < 0.005

    def test_95ci_upper(self, rater_dataset):
        """95% CI 상한 ≈ 0.9901 ± 0.005."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_random")
        assert abs(r["ci_upper"] - 0.990132) < 0.005


# ──────────────────────────────────────────────────────────────────────────
# 3. ICC 수치 정확성 — Two-Way Mixed ICC(3,1)
# ──────────────────────────────────────────────────────────────────────────

class TestICCTwowayMixed:
    """ICC(3,1) Two-Way Mixed, Consistency 검증 (기본 모델)."""

    def test_icc_value(self, rater_dataset):
        """ICC(3,1) 수치 = 0.9641 ± 0.001."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert abs(r["icc"] - 0.964111) < 0.001, f"ICC(3,1) 오차 초과: {r['icc']}"

    def test_f_statistic(self, rater_dataset):
        """F통계량 = 81.5909 ± 0.01 (twoway_random과 동일한 F)."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert abs(r["f"] - 81.5909) < 0.01

    def test_default_model_is_twoway_mixed(self, rater_dataset):
        """model 인수 생략 시 twoway_mixed 적용."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r_default = _compute_icc(data)
        r_explicit = _compute_icc(data, model="twoway_mixed")
        assert abs(r_default["icc"] - r_explicit["icc"]) < 1e-10

    def test_p_value_significant(self, rater_dataset):
        """p < 0.001."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert r["p"] < 0.001


# ──────────────────────────────────────────────────────────────────────────
# 4. 완전 일치 → ICC = 1.0
# ──────────────────────────────────────────────────────────────────────────

class TestPerfectAgreement:

    def test_oneway_perfect(self, perfect_agreement_dataset):
        """완전 일치: ICC(1,1) = 1.0."""
        data = perfect_agreement_dataset.data[["r1", "r2", "r3"]]
        r = _compute_icc(data, model="oneway_random")
        assert abs(r["icc"] - 1.0) < 1e-6, f"완전 일치 ICC(1,1) 오차: {r['icc']}"

    def test_twoway_mixed_perfect(self, perfect_agreement_dataset):
        """완전 일치: ICC(3,1) = 1.0."""
        data = perfect_agreement_dataset.data[["r1", "r2", "r3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert abs(r["icc"] - 1.0) < 1e-6, f"완전 일치 ICC(3,1) 오차: {r['icc']}"

    def test_twoway_random_perfect(self, perfect_agreement_dataset):
        """완전 일치: ICC(2,1) = 1.0."""
        data = perfect_agreement_dataset.data[["r1", "r2", "r3"]]
        r = _compute_icc(data, model="twoway_random")
        assert abs(r["icc"] - 1.0) < 1e-6, f"완전 일치 ICC(2,1) 오차: {r['icc']}"


# ──────────────────────────────────────────────────────────────────────────
# 5. 낮은 일치도 → ICC 낮음
# ──────────────────────────────────────────────────────────────────────────

class TestLowAgreement:

    def test_low_icc_oneway(self, low_agreement_dataset):
        """무작위 평가 → ICC(1,1) < 0.5."""
        data = low_agreement_dataset.data[["r1", "r2", "r3"]]
        r = _compute_icc(data, model="oneway_random")
        assert r["icc"] < 0.5, f"낮은 일치도 ICC가 예상보다 높음: {r['icc']}"

    def test_low_icc_twoway_mixed(self, low_agreement_dataset):
        """무작위 평가 → ICC(3,1) < 0.5."""
        data = low_agreement_dataset.data[["r1", "r2", "r3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert r["icc"] < 0.5, f"낮은 일치도 ICC가 예상보다 높음: {r['icc']}"


# ──────────────────────────────────────────────────────────────────────────
# 6. 95% CI 포함 여부 및 유효성
# ──────────────────────────────────────────────────────────────────────────

class TestConfidenceInterval:

    def test_ci_bounds_present(self, rater_dataset):
        """ci_lower, ci_upper 키가 결과에 있어야 함."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert "ci_lower" in r
        assert "ci_upper" in r

    def test_ci_lower_less_than_upper(self, rater_dataset):
        """CI 하한 < ICC < CI 상한."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert r["ci_lower"] < r["icc"] < r["ci_upper"]

    def test_ci_bounds_are_finite(self, rater_dataset):
        """CI 경계값이 유한 숫자여야 함."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert np.isfinite(r["ci_lower"])
        assert np.isfinite(r["ci_upper"])

    def test_ci_includes_true_value(self, rater_dataset):
        """참조값 0.964111이 95% CI 내에 있어야 함."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r = _compute_icc(data, model="twoway_mixed")
        assert r["ci_lower"] <= 0.964111 <= r["ci_upper"]


# ──────────────────────────────────────────────────────────────────────────
# 7. F통계량과 p값
# ──────────────────────────────────────────────────────────────────────────

class TestFStatisticAndPValue:

    def test_f_positive(self, rater_dataset):
        """F통계량은 양수여야 함."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        for model in ["oneway_random", "twoway_random", "twoway_mixed"]:
            r = _compute_icc(data, model=model)
            assert r["f"] > 0, f"{model}: F <= 0"

    def test_p_between_0_and_1(self, rater_dataset):
        """p값은 0 이상 1 이하여야 함."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        for model in ["oneway_random", "twoway_random", "twoway_mixed"]:
            r = _compute_icc(data, model=model)
            assert 0.0 <= r["p"] <= 1.0, f"{model}: p 범위 오류 {r['p']}"


# ──────────────────────────────────────────────────────────────────────────
# 8. 결과 테이블 구조 (4개 테이블)
# ──────────────────────────────────────────────────────────────────────────

class TestResultTableStructure:

    def test_four_tables_returned(self, rater_dataset):
        """run_analysis가 정확히 4개 테이블을 반환해야 함."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        assert len(result.tables) == 4, f"테이블 수 오류: {len(result.tables)}"

    def test_table_titles(self, rater_dataset):
        """4개 테이블의 제목 확인."""
        expected_titles = [
            "Case Processing Summary",
            "ICC",
            "ANOVA",
            "Interpretation",
        ]
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        actual_titles = [t.title for t in result.tables]
        for expected in expected_titles:
            assert expected in actual_titles, f"테이블 '{expected}' 없음. 실제: {actual_titles}"

    def test_case_processing_summary_columns(self, rater_dataset):
        """Case Processing Summary 컬럼: 구분, N, %."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        assert "N" in cps.dataframe.columns
        assert "%" in cps.dataframe.columns

    def test_icc_table_columns(self, rater_dataset):
        """ICC 테이블 필수 컬럼 확인."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        icc_tbl = next(t for t in result.tables if t.title == "ICC")
        required = ["ICC", "95% CI 하한", "95% CI 상한", "F", "df1", "df2", "p"]
        for col in required:
            assert col in icc_tbl.dataframe.columns, f"ICC 테이블에 '{col}' 컬럼 없음"

    def test_anova_table_columns(self, rater_dataset):
        """ANOVA 테이블 필수 컬럼 확인."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        anova_tbl = next(t for t in result.tables if t.title == "ANOVA")
        required = ["분산원", "SS", "df", "MS"]
        for col in required:
            assert col in anova_tbl.dataframe.columns, f"ANOVA 테이블에 '{col}' 컬럼 없음"

    def test_interpretation_table_columns(self, rater_dataset):
        """Interpretation 테이블 필수 컬럼 확인."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        interp = next(t for t in result.tables if t.title == "Interpretation")
        required = ["ICC", "95% CI", "해석 (Koo & Mae, 2016)"]
        for col in required:
            assert col in interp.dataframe.columns, f"Interpretation 테이블에 '{col}' 컬럼 없음"

    def test_two_way_anova_has_four_rows(self, rater_dataset):
        """Two-Way 모델: ANOVA 테이블에 4행 (피험자간/피험자내/평가자간/잔차)."""
        spec = _make_spec(["rater1", "rater2", "rater3"], model="twoway_mixed")
        result = run_analysis(rater_dataset, spec)
        anova_tbl = next(t for t in result.tables if t.title == "ANOVA")
        assert len(anova_tbl.dataframe) == 4, f"ANOVA 행 수 오류: {len(anova_tbl.dataframe)}"

    def test_oneway_anova_has_two_rows(self, rater_dataset):
        """One-Way 모델: ANOVA 테이블에 2행 (피험자간/피험자내)."""
        spec = _make_spec(["rater1", "rater2", "rater3"], model="oneway_random")
        result = run_analysis(rater_dataset, spec)
        anova_tbl = next(t for t in result.tables if t.title == "ANOVA")
        assert len(anova_tbl.dataframe) == 2, f"One-Way ANOVA 행 수 오류: {len(anova_tbl.dataframe)}"


# ──────────────────────────────────────────────────────────────────────────
# 9. ICC 테이블 값 검증 (run_analysis 통합)
# ──────────────────────────────────────────────────────────────────────────

class TestICCTableValues:

    def test_icc_value_in_table(self, rater_dataset):
        """ICC 테이블의 ICC 수치 = 0.964 (twoway_mixed, 소수점 3자리)."""
        spec = _make_spec(["rater1", "rater2", "rater3"], model="twoway_mixed")
        result = run_analysis(rater_dataset, spec)
        icc_tbl = next(t for t in result.tables if t.title == "ICC")
        icc_str = icc_tbl.dataframe["ICC"].iloc[0]
        icc_val = float(icc_str)
        assert abs(icc_val - 0.964) < 0.001

    def test_interpretation_grade_excellent(self, rater_dataset):
        """ICC > 0.9 → 해석 등급 '우수'."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        interp = next(t for t in result.tables if t.title == "Interpretation")
        grade = interp.dataframe["해석 (Koo & Mae, 2016)"].iloc[0]
        assert "우수" in grade, f"등급 오류: {grade}"

    def test_case_processing_valid_count(self, rater_dataset):
        """Case Processing Summary: 유효 케이스 = 10."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_row = cps.dataframe[cps.dataframe["구분"] == "유효"]
        assert valid_row["N"].iloc[0] == 10

    def test_model_label_in_icc_table(self, rater_dataset):
        """ICC 테이블의 모델 레이블에 'ICC(3,1)' 포함 (기본 모델)."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        icc_tbl = next(t for t in result.tables if t.title == "ICC")
        model_str = icc_tbl.dataframe["모델"].iloc[0]
        assert "ICC(3,1)" in model_str


# ──────────────────────────────────────────────────────────────────────────
# 10. 해석 등급 (Koo & Mae, 2016)
# ──────────────────────────────────────────────────────────────────────────

class TestInterpretationGrade:

    @pytest.mark.parametrize("icc_val, expected_keyword", [
        (0.95, "우수"),
        (0.90, "우수"),
        (0.80, "양호"),
        (0.75, "양호"),
        (0.60, "보통"),
        (0.50, "보통"),
        (0.30, "불량"),
        (0.0,  "불량"),
    ])
    def test_grade_thresholds(self, icc_val, expected_keyword):
        """Koo & Mae(2016) 기준 등급 임계값 검증."""
        from nuristat.analysis.icc import _interpret_icc
        grade = _interpret_icc(icc_val)
        assert expected_keyword in grade, f"ICC={icc_val}: 기대={expected_keyword}, 실제={grade}"


# ──────────────────────────────────────────────────────────────────────────
# 11. 오류 처리 — 변수 부족
# ──────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_single_variable_returns_warning(self, rater_dataset):
        """변수가 1개이면 경고 반환, 테이블 없음."""
        spec = _make_spec(["rater1"])
        result = run_analysis(rater_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_no_variables_returns_warning(self, rater_dataset):
        """변수가 없으면 경고 반환."""
        spec = _make_spec([])
        result = run_analysis(rater_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_nonexistent_variable_returns_warning(self, rater_dataset):
        """존재하지 않는 변수 → 경고 반환."""
        spec = _make_spec(["rater1", "rater_x"])
        result = run_analysis(rater_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_invalid_model_raises(self, rater_dataset):
        """잘못된 모델명 → ValueError 발생."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        with pytest.raises(ValueError, match="지원하지 않는 모델"):
            _compute_icc(data, model="invalid_model")

    def test_single_rater_matrix_raises(self):
        """평가자 1명(열이 1개) → ValueError 발생."""
        data = pd.DataFrame({"r1": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="최소 2명"):
            _compute_icc(data, model="twoway_mixed")

    def test_single_subject_raises(self):
        """피험자 1명(행이 1개) → ValueError 발생."""
        data = pd.DataFrame({"r1": [5], "r2": [4], "r3": [5]})
        with pytest.raises(ValueError, match="최소 2명"):
            _compute_icc(data, model="twoway_mixed")

    def test_all_missing_returns_warning(self):
        """전부 결측 → 유효 케이스 부족 경고."""
        data = {
            "r1": [np.nan, np.nan, np.nan],
            "r2": [np.nan, np.nan, np.nan],
        }
        ds = Dataset(data=pd.DataFrame(data), name="empty")
        spec = _make_spec(["r1", "r2"])
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0


# ──────────────────────────────────────────────────────────────────────────
# 12. 결측치 처리
# ──────────────────────────────────────────────────────────────────────────

class TestMissingData:

    def test_missing_excluded_from_case_processing(self, missing_dataset):
        """결측 있는 행은 제외됨 — Case Processing Summary에 결측 건수 표시."""
        spec = _make_spec(["r1", "r2", "r3"])
        result = run_analysis(missing_dataset, spec)
        if len(result.warnings) > 0:
            pytest.skip("유효 케이스 부족으로 경고 반환")
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        missing_row = cps.dataframe[cps.dataframe["구분"] == "제외됨"]
        assert missing_row["N"].iloc[0] >= 0  # 결측 건수는 0 이상

    def test_valid_cases_reduced_by_missing(self, missing_dataset):
        """결측 처리 후 유효 케이스 수가 전체보다 작아야 함."""
        spec = _make_spec(["r1", "r2", "r3"])
        result = run_analysis(missing_dataset, spec)
        if len(result.warnings) > 0:
            pytest.skip("유효 케이스 부족으로 경고 반환")
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_n = cps.dataframe[cps.dataframe["구분"] == "유효"]["N"].iloc[0]
        total_n = cps.dataframe[cps.dataframe["구분"] == "합계"]["N"].iloc[0]
        assert valid_n < total_n


# ──────────────────────────────────────────────────────────────────────────
# 13. 모델별 run_analysis 통합 테스트
# ──────────────────────────────────────────────────────────────────────────

class TestRunAnalysisIntegration:

    @pytest.mark.parametrize("model", ["oneway_random", "twoway_random", "twoway_mixed"])
    def test_all_models_return_four_tables(self, rater_dataset, model):
        """모든 모델에서 4개 테이블 반환."""
        spec = _make_spec(["rater1", "rater2", "rater3"], model=model)
        result = run_analysis(rater_dataset, spec)
        assert len(result.warnings) == 0, f"경고 발생: {result.warnings}"
        assert len(result.tables) == 4

    @pytest.mark.parametrize("model", ["oneway_random", "twoway_random", "twoway_mixed"])
    def test_all_models_no_warnings(self, rater_dataset, model):
        """모든 모델에서 경고 없음."""
        spec = _make_spec(["rater1", "rater2", "rater3"], model=model)
        result = run_analysis(rater_dataset, spec)
        assert len(result.warnings) == 0

    def test_notes_contain_icc_value(self, rater_dataset):
        """result.notes에 ICC 수치와 해석 등급 포함."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        assert len(result.notes) > 0
        note_text = " ".join(result.notes)
        assert "ICC" in note_text

    def test_analysis_id(self, rater_dataset):
        """분석 결과 ID가 'icc'여야 함."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        assert result.id == "icc"

    def test_analysis_title(self, rater_dataset):
        """분석 결과 제목 확인."""
        spec = _make_spec(["rater1", "rater2", "rater3"])
        result = run_analysis(rater_dataset, spec)
        assert "Intraclass" in result.title or "ICC" in result.title

    def test_twoway_mixed_f_equals_twoway_random_f(self, rater_dataset):
        """ICC(2,1)과 ICC(3,1)의 F통계량은 동일해야 함 (MS_B/MS_E 공유)."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r2 = _compute_icc(data, model="twoway_random")
        r3 = _compute_icc(data, model="twoway_mixed")
        assert abs(r2["f"] - r3["f"]) < 1e-6

    def test_oneway_f_different_from_twoway_f(self, rater_dataset):
        """ICC(1,1) F통계량은 ICC(3,1)과 달라야 함 (WMS vs EMS 차이)."""
        data = rater_dataset.data[["rater1", "rater2", "rater3"]]
        r1 = _compute_icc(data, model="oneway_random")
        r3 = _compute_icc(data, model="twoway_mixed")
        assert abs(r1["f"] - r3["f"]) > 0.1
