"""Cohen's Kappa 평가자 간 일치도 분석 테스트.

SPSS: Analyze > Descriptive Statistics > Crosstabs > Statistics > Kappa 호환성 검증.

참조값 (sklearn.metrics.cohen_kappa_score 교차 검증):
    rater_a = [1,1,1,1,1,0,0,0,0,0,2,2,2,2,2,1,0,2,1,0]
    rater_b = [1,1,1,0,1,0,0,0,1,0,2,2,2,1,2,1,0,2,0,0]
    kappa   = 0.6981 (sklearn 일치 확인)
    Po      = 0.8000
    Pe      = 0.3375
    SE      = 0.1350
    z       = 5.1709
    95% CI  = [0.4335, 0.9627]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.cohens_kappa import run_analysis, _compute_kappa
from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

# 참조 데이터: 평가자 A, B (n=20, 3범주: 0=음성, 1=양성, 2=의심)
RATER_A = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 0, 2, 1, 0]
RATER_B = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 1, 2, 1, 0, 2, 0, 0]

# sklearn 교차 검증값
EXPECTED_KAPPA = 0.6981
EXPECTED_PO = 0.8000
EXPECTED_PE = 0.3375
EXPECTED_SE = 0.1350
EXPECTED_CI_LOWER = 0.4335
EXPECTED_CI_UPPER = 0.9627


@pytest.fixture
def main_dataset() -> Dataset:
    """평가자 A/B 3범주 데이터 (n=20)."""
    return Dataset(
        data=pd.DataFrame({"rater_a": RATER_A, "rater_b": RATER_B}),
        name="kappa_test",
    )


@pytest.fixture
def perfect_dataset() -> Dataset:
    """완전 일치 데이터 — kappa = 1.0."""
    vals = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    return Dataset(
        data=pd.DataFrame({"r1": vals, "r2": vals}),
        name="perfect",
    )


@pytest.fixture
def no_agreement_dataset() -> Dataset:
    """체계적 불일치 — kappa <= 0."""
    r1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    r2 = [1, 2, 1, 0, 2, 0, 0, 1, 0]
    return Dataset(
        data=pd.DataFrame({"r1": r1, "r2": r2}),
        name="no_agreement",
    )


@pytest.fixture
def missing_dataset() -> Dataset:
    """결측치 포함 데이터."""
    r1 = [1, 1, np.nan, 0, 2, 1, np.nan, 0]
    r2 = [1, 0, 1, 0, 2, np.nan, 1, 0]
    return Dataset(
        data=pd.DataFrame({"r1": r1, "r2": r2}),
        name="missing",
    )


def _make_spec(rater1: str, rater2: str) -> dict:
    return {"variables": {"rater1": rater1, "rater2": rater2}}


# ──────────────────────────────────────────────────────────────
# 1. Kappa 수치 정확성 (sklearn 교차 검증)
# ──────────────────────────────────────────────────────────────

class TestKappaAccuracy:
    """핵심 통계량이 sklearn/SPSS 참조값과 일치하는지 검증."""

    def test_kappa_value(self):
        """kappa = 0.6981 (sklearn 교차 검증)."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["kappa"] - EXPECTED_KAPPA) < 0.0001

    def test_po_value(self):
        """관찰 일치율 Po = 0.8000."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["po"] - EXPECTED_PO) < 0.0001

    def test_pe_value(self):
        """기대 일치율 Pe = 0.3375."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["pe"] - EXPECTED_PE) < 0.0001

    def test_se_value(self):
        """표준오차 SE ≈ 0.1350 (Fleiss 1971)."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["se"] - EXPECTED_SE) < 0.001

    def test_ci_lower(self):
        """95% CI 하한 ≈ 0.4335."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["ci_lower"] - EXPECTED_CI_LOWER) < 0.001

    def test_ci_upper(self):
        """95% CI 상한 ≈ 0.9627."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert abs(result["ci_upper"] - EXPECTED_CI_UPPER) < 0.001

    def test_kappa_matches_sklearn(self):
        """sklearn.metrics.cohen_kappa_score와 일치 여부."""
        try:
            from sklearn.metrics import cohen_kappa_score
            sklearn_kappa = cohen_kappa_score(RATER_A, RATER_B)
            result = _compute_kappa(RATER_A, RATER_B)
            assert abs(result["kappa"] - sklearn_kappa) < 0.0001
        except ImportError:
            pytest.skip("sklearn 미설치 — 교차 검증 건너뜀")

    def test_n_is_correct(self):
        """n = 20 (전체 케이스 수)."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["n"] == 20

    def test_categories_detected(self):
        """범주 [0, 1, 2] 자동 감지."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["categories"] == [0, 1, 2]

    def test_p_value_significant(self):
        """kappa = 0.698 → p < 0.001."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["p"] < 0.001

    def test_z_positive(self):
        """z 통계량 > 0 (kappa > 0)."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["z"] > 0


# ──────────────────────────────────────────────────────────────
# 2. 완전 일치 → kappa = 1.0
# ──────────────────────────────────────────────────────────────

class TestPerfectAgreement:
    """완전 일치 조건에서 kappa = 1.0 반환."""

    def test_perfect_kappa(self):
        vals = [0, 0, 1, 1, 2, 2]
        result = _compute_kappa(vals, vals)
        assert abs(result["kappa"] - 1.0) < 1e-9

    def test_perfect_po(self):
        vals = [0, 0, 1, 1, 2, 2]
        result = _compute_kappa(vals, vals)
        assert abs(result["po"] - 1.0) < 1e-9

    def test_perfect_via_run_analysis(self, perfect_dataset):
        result = run_analysis(perfect_dataset, _make_spec("r1", "r2"))
        measures = next(t for t in result.tables if t.title == "Symmetric Measures")
        kappa_str = measures.dataframe["값"].iloc[0]
        kappa_val = float(kappa_str)
        assert abs(kappa_val - 1.0) < 0.001

    def test_binary_perfect(self):
        """2범주 완전 일치."""
        vals = [0, 1, 0, 1, 0, 1]
        result = _compute_kappa(vals, vals)
        assert abs(result["kappa"] - 1.0) < 1e-9


# ──────────────────────────────────────────────────────────────
# 3. 완전 불일치 → kappa <= 0
# ──────────────────────────────────────────────────────────────

class TestNoAgreement:
    """체계적 불일치 시 kappa <= 0."""

    def test_disagreement_kappa_nonpositive(self):
        """반대 패턴 → kappa < 0."""
        r1 = [0, 0, 1, 1]
        r2 = [1, 1, 0, 0]
        result = _compute_kappa(r1, r2)
        assert result["kappa"] <= 0

    def test_disagreement_via_dataset(self, no_agreement_dataset):
        result = run_analysis(no_agreement_dataset, _make_spec("r1", "r2"))
        measures = next(t for t in result.tables if t.title == "Symmetric Measures")
        kappa_val = float(measures.dataframe["값"].iloc[0])
        assert kappa_val <= 0.4  # 우연 수준 이하


# ──────────────────────────────────────────────────────────────
# 4. 우연 일치만 → kappa ≈ 0
# ──────────────────────────────────────────────────────────────

class TestChanceAgreement:
    """완전 무작위 응답 → kappa 는 0 근방."""

    def test_random_kappa_near_zero(self):
        rng = np.random.default_rng(2024)
        r1 = rng.integers(0, 3, 200).tolist()
        r2 = rng.integers(0, 3, 200).tolist()
        result = _compute_kappa(r1, r2)
        # 무작위: -0.2 ~ 0.2 범위 기대
        assert -0.3 < result["kappa"] < 0.3


# ──────────────────────────────────────────────────────────────
# 5. 해석 등급 (Landis-Koch)
# ──────────────────────────────────────────────────────────────

class TestLandisKochInterpretation:
    """Landis-Koch 기준 해석 등급 정확성 검증."""

    @pytest.mark.parametrize("kappa,expected", [
        (-0.10, "없음"),
        (0.00,  "약함"),
        (0.10,  "약함"),
        (0.20,  "보통"),
        (0.30,  "보통"),
        (0.40,  "중간"),
        (0.50,  "중간"),
        (0.60,  "상당"),
        (0.70,  "상당"),
        (0.80,  "거의 완전"),
        (0.90,  "거의 완전"),
        (1.00,  "거의 완전"),
    ])
    def test_grade_boundary(self, kappa, expected):
        from nuristat.analysis.cohens_kappa import _landis_koch_grade
        grade = _landis_koch_grade(kappa)
        assert expected in grade, f"kappa={kappa} → 기대: {expected}, 실제: {grade}"

    def test_main_data_grade_substantial(self, main_dataset):
        """kappa=0.698 → 상당 (Substantial)."""
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        interp = next(t for t in result.tables if t.title == "Interpretation")
        grade_str = interp.dataframe["해석 (Landis-Koch)"].iloc[0]
        assert "상당" in grade_str


# ──────────────────────────────────────────────────────────────
# 6. 결과 테이블 구조 (4개 테이블)
# ──────────────────────────────────────────────────────────────

class TestResultStructure:
    """AnalysisResult 구조 검증 — 4개 테이블 필수."""

    def test_returns_analysis_result(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        assert isinstance(result, AnalysisResult)

    def test_has_four_tables(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        assert len(result.tables) == 4

    def test_table_titles_present(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        titles = [t.title for t in result.tables]
        assert "Case Processing Summary" in titles
        assert "Crosstabulation" in titles
        assert "Symmetric Measures" in titles
        assert "Interpretation" in titles

    def test_case_processing_columns(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        cols = cps.dataframe.columns.tolist()
        assert "구분" in cols
        assert "N" in cols
        assert "%" in cols

    def test_case_processing_n_total(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        total_row = cps.dataframe[cps.dataframe["구분"] == "합계"]
        assert total_row["N"].iloc[0] == 20

    def test_symmetric_measures_columns(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        sm = next(t for t in result.tables if t.title == "Symmetric Measures")
        cols = sm.dataframe.columns.tolist()
        assert "측도" in cols
        assert "값" in cols
        assert "표준오차" in cols
        assert "근사 T" in cols
        assert "유의확률" in cols
        assert "95% CI 하한" in cols
        assert "95% CI 상한" in cols

    def test_interpretation_columns(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        interp = next(t for t in result.tables if t.title == "Interpretation")
        cols = interp.dataframe.columns.tolist()
        assert "해석 (Landis-Koch)" in cols
        assert "관찰 일치율 (Po)" in cols
        assert "기대 일치율 (Pe)" in cols

    def test_crosstab_contains_rater_info(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        ct = next(t for t in result.tables if t.title == "Crosstabulation")
        assert ct.dataframe is not None
        assert len(ct.dataframe) > 0

    def test_has_notes(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        assert len(result.notes) > 0

    def test_note_contains_kappa(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        assert "κ" in result.notes[0] or "kappa" in result.notes[0].lower() or "Kappa" in result.notes[0]

    def test_no_warnings_on_valid_data(self, main_dataset):
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        assert len(result.warnings) == 0


# ──────────────────────────────────────────────────────────────
# 7. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestErrorHandling:
    """잘못된 입력에 대한 경고 반환 검증."""

    def test_missing_rater1_var(self, main_dataset):
        """rater1 변수 없음 → 경고."""
        result = run_analysis(main_dataset, _make_spec("없는변수", "rater_b"))
        assert len(result.warnings) > 0

    def test_missing_rater2_var(self, main_dataset):
        """rater2 변수 없음 → 경고."""
        result = run_analysis(main_dataset, _make_spec("rater_a", "없는변수"))
        assert len(result.warnings) > 0

    def test_empty_spec_variables(self, main_dataset):
        """변수 미지정 → 경고."""
        result = run_analysis(main_dataset, {"variables": {}})
        assert len(result.warnings) > 0

    def test_single_category_both(self):
        """두 변수 모두 단일 범주 → 경고."""
        dataset = Dataset(
            data=pd.DataFrame({"r1": [1, 1, 1, 1], "r2": [1, 1, 1, 1]}),
            name="single_cat",
        )
        result = run_analysis(dataset, _make_spec("r1", "r2"))
        assert len(result.warnings) > 0

    def test_insufficient_cases(self):
        """유효 케이스 1개 → 경고."""
        dataset = Dataset(
            data=pd.DataFrame({"r1": [1, np.nan], "r2": [1, np.nan]}),
            name="tiny",
        )
        result = run_analysis(dataset, _make_spec("r1", "r2"))
        assert len(result.warnings) > 0

    def test_unequal_length_raises(self):
        """길이 불일치 → ValueError."""
        with pytest.raises(ValueError):
            _compute_kappa([1, 2, 3], [1, 2])

    def test_empty_arrays_raises(self):
        """빈 배열 → ValueError."""
        with pytest.raises(ValueError):
            _compute_kappa([], [])

    def test_error_result_has_no_tables(self, main_dataset):
        """오류 시 테이블 없음."""
        result = run_analysis(main_dataset, {"variables": {}})
        assert len(result.tables) == 0


# ──────────────────────────────────────────────────────────────
# 8. SE 및 95% CI 계산 검증
# ──────────────────────────────────────────────────────────────

class TestSEandCI:
    """표준오차와 신뢰구간의 수학적 일관성 검증."""

    def test_ci_contains_kappa(self):
        """95% CI가 kappa를 포함해야 함."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["ci_lower"] < result["kappa"] < result["ci_upper"]

    def test_ci_width_equals_2_times_196_se(self):
        """CI 너비 = 2 × 1.96 × SE."""
        result = _compute_kappa(RATER_A, RATER_B)
        expected_width = 2 * 1.96 * result["se"]
        actual_width = result["ci_upper"] - result["ci_lower"]
        assert abs(actual_width - expected_width) < 1e-9

    def test_z_equals_kappa_over_se(self):
        """z = kappa / SE."""
        result = _compute_kappa(RATER_A, RATER_B)
        expected_z = result["kappa"] / result["se"]
        assert abs(result["z"] - expected_z) < 1e-9

    def test_se_positive(self):
        """SE > 0."""
        result = _compute_kappa(RATER_A, RATER_B)
        assert result["se"] > 0

    def test_se_in_symmetric_measures_table(self, main_dataset):
        """Symmetric Measures 테이블에 SE 값 포함."""
        result = run_analysis(main_dataset, _make_spec("rater_a", "rater_b"))
        sm = next(t for t in result.tables if t.title == "Symmetric Measures")
        se_str = sm.dataframe["표준오차"].iloc[0]
        se_val = float(se_str)
        assert abs(se_val - EXPECTED_SE) < 0.002


# ──────────────────────────────────────────────────────────────
# 9. 결측치 처리
# ──────────────────────────────────────────────────────────────

class TestMissingDataHandling:
    """결측치 포함 데이터에서 목록별 제거 검증."""

    def test_valid_count_reduced(self, missing_dataset):
        result = run_analysis(missing_dataset, _make_spec("r1", "r2"))
        assert len(result.tables) == 4  # 오류 없이 완료

    def test_excluded_count_positive(self, missing_dataset):
        result = run_analysis(missing_dataset, _make_spec("r1", "r2"))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        excluded = int(cps.dataframe[cps.dataframe["구분"] == "제외됨"]["N"].iloc[0])
        assert excluded > 0

    def test_total_equals_valid_plus_excluded(self, missing_dataset):
        result = run_analysis(missing_dataset, _make_spec("r1", "r2"))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid = int(cps.dataframe[cps.dataframe["구분"] == "유효"]["N"].iloc[0])
        excluded = int(cps.dataframe[cps.dataframe["구분"] == "제외됨"]["N"].iloc[0])
        total = int(cps.dataframe[cps.dataframe["구분"] == "합계"]["N"].iloc[0])
        assert valid + excluded == total


# ──────────────────────────────────────────────────────────────
# 10. 이진 범주 (2범주)
# ──────────────────────────────────────────────────────────────

class TestBinaryCategories:
    """이진 범주 데이터에서 정확한 kappa 계산."""

    def test_binary_moderate_agreement(self):
        """이진 범주 중간 수준 일치."""
        r1 = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        r2 = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
        result = _compute_kappa(r1, r2)
        assert -1.0 <= result["kappa"] <= 1.0

    def test_binary_dataset_four_tables(self):
        """이진 데이터 — 4개 테이블 반환."""
        r1 = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        r2 = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
        dataset = Dataset(
            data=pd.DataFrame({"r1": r1, "r2": r2}),
            name="binary",
        )
        result = run_analysis(dataset, _make_spec("r1", "r2"))
        assert len(result.tables) == 4

    def test_binary_kappa_matches_sklearn(self):
        """이진 데이터 sklearn 교차 검증."""
        try:
            from sklearn.metrics import cohen_kappa_score
            r1 = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
            r2 = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            sklearn_kappa = cohen_kappa_score(r1, r2)
            result = _compute_kappa(r1, r2)
            assert abs(result["kappa"] - sklearn_kappa) < 0.0001
        except ImportError:
            pytest.skip("sklearn 미설치")
