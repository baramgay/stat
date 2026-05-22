"""카이제곱 적합도 검정(Chi-Square Goodness-of-Fit) 테스트.

SPSS: Analyze > Nonparametric Tests > Legacy Dialogs > Chi-Square 호환성 검증.

참조값 (scipy.stats.chisquare 기반):
  예시1 — 주사위 공정성 (n=60, 균등 기대):
    observed = [8, 12, 9, 11, 10, 10]
    chi2 = 1.0000, df = 5, p = 0.9626

  예시2 — Mendel 완두콩 유전 비율 (9:3:3:1):
    observed = [315, 108, 101, 32]
    expected  = [312.75, 104.25, 104.25, 34.75]
    chi2 = 0.4700, df = 3, p = 0.9254
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.chi_square_gof import run_analysis
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def dice_dataset() -> Dataset:
    """주사위 공정성 검정 — 균등 분포 예시 (n=60)."""
    counts = [8, 12, 9, 11, 10, 10]
    faces = [1, 2, 3, 4, 5, 6]
    rows = []
    for face, count in zip(faces, counts):
        rows.extend([face] * count)
    return Dataset(data=pd.DataFrame({"face": rows}), name="dice")


@pytest.fixture
def mendel_dataset() -> Dataset:
    """Mendel 완두콩 — 9:3:3:1 비율 검정 (n=556)."""
    categories = ["RY", "Ry", "rY", "ry"]
    counts = [315, 108, 101, 32]
    rows = []
    for cat, count in zip(categories, counts):
        rows.extend([cat] * count)
    return Dataset(data=pd.DataFrame({"type": rows}), name="mendel")


@pytest.fixture
def perfect_match_dataset() -> Dataset:
    """완전 일치 — 관찰==기대 (chi2=0, p=1)."""
    rows = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    return Dataset(data=pd.DataFrame({"group": rows}), name="perfect")


@pytest.fixture
def perfect_mismatch_dataset() -> Dataset:
    """완전 불일치 — 한 범주에 집중 (chi2 매우 큼)."""
    rows = ["A"] * 30 + ["B"] * 0 + ["C"] * 0
    # B, C가 0이면 chi2 계산 불가 → B, C 제거 위해 실제 B/C 없는 데이터
    # 대신 B=0 obs는 value_counts에 나타나지 않으므로 multi-cat 유지
    rows = ["A"] * 28 + ["B"] * 1 + ["C"] * 1  # 거의 A에 집중
    return Dataset(data=pd.DataFrame({"group": rows}), name="mismatch")


@pytest.fixture
def missing_dataset() -> Dataset:
    """결측치 포함 데이터."""
    values = ["A", "B", np.nan, "A", "B", np.nan, "C"]
    return Dataset(data=pd.DataFrame({"cat": values}), name="missing")


@pytest.fixture
def two_var_dataset() -> Dataset:
    """복수 변수 — 동시 검정."""
    n = 30
    cat1 = ["X"] * 10 + ["Y"] * 10 + ["Z"] * 10
    cat2 = ["P"] * 20 + ["Q"] * 10
    return Dataset(
        data=pd.DataFrame({"cat1": cat1, "cat2": cat2}),
        name="two_vars",
    )


def _spec(target: list[str], expected_ratios: dict | None = None, **opts) -> dict:
    """spec 딕셔너리 생성 헬퍼."""
    variables: dict = {"target": target}
    if expected_ratios is not None:
        variables["expected_ratios"] = expected_ratios
    return {"variables": variables, "options": opts}


# ──────────────────────────────────────────────────────────────
# 1. chi2 수치 정확성 — 균등 분포
# ──────────────────────────────────────────────────────────────

class TestChiSquareValueUniform:

    def test_chi2_value_dice(self, dice_dataset):
        """주사위 균등 검정: chi2 = 1.0000."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        chi2 = float(test_table.dataframe["Chi-Square"].iloc[0])
        assert abs(chi2 - 1.0000) < 0.001

    def test_df_dice(self, dice_dataset):
        """주사위: df = 6 - 1 = 5."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        df = int(test_table.dataframe["df"].iloc[0])
        assert df == 5

    def test_pvalue_dice(self, dice_dataset):
        """주사위 균등 검정: p = 0.9626 — 유의하지 않음."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        p_str = test_table.dataframe["Asymptotic Significance (p)"].iloc[0]
        # ".963" 또는 "0.963" 형식 모두 허용
        p_val = float(p_str.lstrip(".")) if p_str.startswith(".") else float(p_str)
        if p_str.startswith("."):
            p_val = float("0" + p_str)
        assert abs(p_val - 0.9626) < 0.002


# ──────────────────────────────────────────────────────────────
# 2. chi2 수치 정확성 — 비율 지정 분포
# ──────────────────────────────────────────────────────────────

class TestChiSquareValueRatio:

    def test_chi2_value_mendel(self, mendel_dataset):
        """Mendel 9:3:3:1 검정: chi2 = 0.4700."""
        ratios = {"RY": 9, "Ry": 3, "rY": 3, "ry": 1}
        result = run_analysis(mendel_dataset, _spec(["type"], expected_ratios=ratios))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        chi2 = float(test_table.dataframe["Chi-Square"].iloc[0])
        assert abs(chi2 - 0.4700) < 0.001

    def test_df_mendel(self, mendel_dataset):
        """Mendel: df = 4 - 1 = 3."""
        ratios = {"RY": 9, "Ry": 3, "rY": 3, "ry": 1}
        result = run_analysis(mendel_dataset, _spec(["type"], expected_ratios=ratios))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        df = int(test_table.dataframe["df"].iloc[0])
        assert df == 3

    def test_pvalue_mendel(self, mendel_dataset):
        """Mendel 9:3:3:1: p = 0.9254."""
        ratios = {"RY": 9, "Ry": 3, "rY": 3, "ry": 1}
        result = run_analysis(mendel_dataset, _spec(["type"], expected_ratios=ratios))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        p_str = test_table.dataframe["Asymptotic Significance (p)"].iloc[0]
        if p_str.startswith("."):
            p_val = float("0" + p_str)
        else:
            p_val = float(p_str)
        assert abs(p_val - 0.9254) < 0.002

    def test_expected_frequencies_mendel(self, mendel_dataset):
        """기대 빈도 확인: [312.75, 104.25, 104.25, 34.75]."""
        ratios = {"RY": 9, "Ry": 3, "rY": 3, "ry": 1}
        result = run_analysis(mendel_dataset, _spec(["type"], expected_ratios=ratios))
        freq_table = next(t for t in result.tables if t.title == "Frequencies")
        expected_vals = freq_table.dataframe["기대 빈도"].astype(float).tolist()
        ref = [312.75, 104.25, 104.25, 34.75]
        for ev, rv in zip(sorted(expected_vals), sorted(ref)):
            assert abs(ev - rv) < 0.1


# ──────────────────────────────────────────────────────────────
# 3. df = k - 1 규칙
# ──────────────────────────────────────────────────────────────

class TestDegreesOfFreedom:

    def test_df_equals_k_minus_1_uniform(self, dice_dataset):
        """균등 분포: df = 범주 수 - 1."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        df = int(test_table.dataframe["df"].iloc[0])
        # 주사위: 6 범주 → df=5
        assert df == 5

    def test_df_equals_k_minus_1_ratio(self, mendel_dataset):
        """비율 지정: df = 범주 수 - 1."""
        ratios = {"RY": 9, "Ry": 3, "rY": 3, "ry": 1}
        result = run_analysis(mendel_dataset, _spec(["type"], expected_ratios=ratios))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        df = int(test_table.dataframe["df"].iloc[0])
        # 4 범주 → df=3
        assert df == 3


# ──────────────────────────────────────────────────────────────
# 4. 완전 일치 → chi2=0, p=1.0
# ──────────────────────────────────────────────────────────────

class TestPerfectMatch:

    def test_chi2_zero_on_perfect_match(self, perfect_match_dataset):
        """관찰==기대(균등) → chi2 = 0."""
        result = run_analysis(perfect_match_dataset, _spec(["group"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        chi2 = float(test_table.dataframe["Chi-Square"].iloc[0])
        assert abs(chi2) < 1e-10

    def test_pvalue_one_on_perfect_match(self, perfect_match_dataset):
        """관찰==기대(균등) → p = 1.0."""
        result = run_analysis(perfect_match_dataset, _spec(["group"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        p_str = test_table.dataframe["Asymptotic Significance (p)"].iloc[0]
        # format_pvalue: p>=1 → "1.000"
        assert p_str == "1.000"


# ──────────────────────────────────────────────────────────────
# 5. 완전 불일치 → chi2 매우 큼, p ≈ 0
# ──────────────────────────────────────────────────────────────

class TestPerfectMismatch:

    def test_chi2_large_on_mismatch(self, perfect_mismatch_dataset):
        """한 범주 집중 → chi2 >> 0."""
        result = run_analysis(perfect_mismatch_dataset, _spec(["group"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        chi2 = float(test_table.dataframe["Chi-Square"].iloc[0])
        assert chi2 > 10.0

    def test_pvalue_small_on_mismatch(self, perfect_mismatch_dataset):
        """한 범주 집중 → p < 0.05."""
        result = run_analysis(perfect_mismatch_dataset, _spec(["group"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        p_str = test_table.dataframe["Asymptotic Significance (p)"].iloc[0]
        # "< .001" 또는 매우 작은 p값
        assert p_str == "< .001" or float("0" + p_str if p_str.startswith(".") else p_str) < 0.05


# ──────────────────────────────────────────────────────────────
# 6. 결과 테이블 구조 (4개 테이블)
# ──────────────────────────────────────────────────────────────

class TestResultTableStructure:

    def test_four_tables_returned(self, dice_dataset):
        """결과 테이블이 정확히 4개."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        assert len(result.tables) == 4

    def test_table_titles(self, dice_dataset):
        """테이블 제목 검증."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        titles = [t.title for t in result.tables]
        assert "Case Processing Summary" in titles
        assert "Frequencies" in titles
        assert "Test Statistics" in titles
        assert "Residuals" in titles

    def test_case_processing_summary_columns(self, dice_dataset):
        """Case Processing Summary: 구분, N, % 컬럼."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        assert "구분" in cps.dataframe.columns
        assert "N" in cps.dataframe.columns
        assert "%" in cps.dataframe.columns

    def test_case_processing_summary_rows(self, dice_dataset):
        """Case Processing Summary: 유효/결측/합계 3행."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        assert len(cps.dataframe) == 3
        labels = cps.dataframe["구분"].tolist()
        assert "유효" in labels
        assert "결측" in labels
        assert "합계" in labels

    def test_frequencies_columns(self, dice_dataset):
        """Frequencies: 변수, 범주, 관찰 빈도, 기대 빈도, 잔차 컬럼."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        freq = next(t for t in result.tables if t.title == "Frequencies")
        cols = freq.dataframe.columns.tolist()
        assert "변수" in cols
        assert "범주" in cols
        assert "관찰 빈도" in cols
        assert "기대 빈도" in cols
        assert "잔차" in cols

    def test_test_statistics_columns(self, dice_dataset):
        """Test Statistics: 변수, Chi-Square, df, p값 컬럼."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        cols = test_table.dataframe.columns.tolist()
        assert "변수" in cols
        assert "Chi-Square" in cols
        assert "df" in cols
        assert "Asymptotic Significance (p)" in cols

    def test_residuals_columns(self, dice_dataset):
        """Residuals: 변수, 범주, 잔차, 표준화 잔차 컬럼."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        resid = next(t for t in result.tables if t.title == "Residuals")
        cols = resid.dataframe.columns.tolist()
        assert "변수" in cols
        assert "범주" in cols
        assert "잔차" in cols
        assert "표준화 잔차" in cols

    def test_frequencies_row_count(self, dice_dataset):
        """Frequencies: 범주 수만큼 행."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        freq = next(t for t in result.tables if t.title == "Frequencies")
        # 주사위: 6 범주
        assert len(freq.dataframe) == 6

    def test_residuals_row_count(self, dice_dataset):
        """Residuals: 범주 수만큼 행."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        resid = next(t for t in result.tables if t.title == "Residuals")
        assert len(resid.dataframe) == 6


# ──────────────────────────────────────────────────────────────
# 7. 잔차 및 표준화 잔차 수치
# ──────────────────────────────────────────────────────────────

class TestResiduals:

    def test_residual_values(self, dice_dataset):
        """잔차 = 관찰 - 기대 (주사위 예시)."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        freq_table = next(t for t in result.tables if t.title == "Frequencies")
        obs = freq_table.dataframe["관찰 빈도"].astype(float).values
        exp = freq_table.dataframe["기대 빈도"].astype(float).values
        resid_calc = obs - exp

        resid_table = next(t for t in result.tables if t.title == "Residuals")
        resid_vals = resid_table.dataframe["잔차"].astype(float).values
        np.testing.assert_allclose(resid_vals, resid_calc, atol=0.01)

    def test_standardized_residual_formula(self, dice_dataset):
        """표준화 잔차 = 잔차 / sqrt(기대)."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        freq_table = next(t for t in result.tables if t.title == "Frequencies")
        obs = freq_table.dataframe["관찰 빈도"].astype(float).values
        exp = freq_table.dataframe["기대 빈도"].astype(float).values
        expected_std_resid = (obs - exp) / np.sqrt(exp)

        resid_table = next(t for t in result.tables if t.title == "Residuals")
        actual_std_resid = resid_table.dataframe["표준화 잔차"].astype(float).values
        np.testing.assert_allclose(actual_std_resid, expected_std_resid, atol=0.001)

    def test_residuals_sum_approximately_zero(self, dice_dataset):
        """잔차 합계 ≈ 0 (기대 빈도 합 = 관찰 빈도 합)."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        resid_table = next(t for t in result.tables if t.title == "Residuals")
        resid_sum = resid_table.dataframe["잔차"].astype(float).sum()
        assert abs(resid_sum) < 0.01


# ──────────────────────────────────────────────────────────────
# 8. 결측치 처리
# ──────────────────────────────────────────────────────────────

class TestMissingValues:

    def test_listwise_excludes_missing(self, missing_dataset):
        """listwise=True: 결측 행 제거 후 유효 케이스 수 감소."""
        result = run_analysis(missing_dataset, _spec(["cat"], listwise=True))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_n = int(cps.dataframe.loc[cps.dataframe["구분"] == "유효", "N"].iloc[0])
        excluded_n = int(cps.dataframe.loc[cps.dataframe["구분"] == "결측", "N"].iloc[0])
        assert valid_n == 5   # 7 - 2개 결측
        assert excluded_n == 2

    def test_case_processing_total(self, missing_dataset):
        """합계 N = 유효 + 결측."""
        result = run_analysis(missing_dataset, _spec(["cat"], listwise=True))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_n = int(cps.dataframe.loc[cps.dataframe["구분"] == "유효", "N"].iloc[0])
        excluded_n = int(cps.dataframe.loc[cps.dataframe["구분"] == "결측", "N"].iloc[0])
        total_n = int(cps.dataframe.loc[cps.dataframe["구분"] == "합계", "N"].iloc[0])
        assert total_n == valid_n + excluded_n

    def test_percentage_sums_100(self, missing_dataset):
        """유효% + 결측% = 100.0."""
        result = run_analysis(missing_dataset, _spec(["cat"], listwise=True))
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        pct_sum = cps.dataframe.loc[
            cps.dataframe["구분"].isin(["유효", "결측"]), "%"
        ].sum()
        assert abs(pct_sum - 100.0) < 0.1


# ──────────────────────────────────────────────────────────────
# 9. 복수 변수 동시 검정
# ──────────────────────────────────────────────────────────────

class TestMultipleVariables:

    def test_test_statistics_has_two_rows(self, two_var_dataset):
        """2개 변수 → Test Statistics에 2행."""
        result = run_analysis(two_var_dataset, _spec(["cat1", "cat2"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        assert len(test_table.dataframe) == 2

    def test_var_names_in_test_statistics(self, two_var_dataset):
        """Test Statistics: 변수명 정확히 포함."""
        result = run_analysis(two_var_dataset, _spec(["cat1", "cat2"]))
        test_table = next(t for t in result.tables if t.title == "Test Statistics")
        var_names = test_table.dataframe["변수"].tolist()
        assert "cat1" in var_names
        assert "cat2" in var_names

    def test_four_tables_with_two_vars(self, two_var_dataset):
        """복수 변수도 테이블 수는 4개."""
        result = run_analysis(two_var_dataset, _spec(["cat1", "cat2"]))
        assert len(result.tables) == 4


# ──────────────────────────────────────────────────────────────
# 10. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_no_variables_returns_warning(self):
        """변수 없음 → 경고 반환, 테이블 없음."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}), name="empty_spec")
        result = run_analysis(ds, _spec([]))
        assert len(result.warnings) > 0
        assert any("변수" in w for w in result.warnings)

    def test_missing_variable_returns_warning(self):
        """존재하지 않는 변수명 → 경고 반환."""
        ds = Dataset(data=pd.DataFrame({"x": ["A", "B", "A"]}), name="test")
        result = run_analysis(ds, _spec(["nonexistent"]))
        assert len(result.warnings) > 0
        assert any("nonexistent" in w or "찾을 수 없습니다" in w for w in result.warnings)

    def test_single_category_returns_warning(self):
        """단일 범주 변수 → 경고 (검정 불가)."""
        ds = Dataset(data=pd.DataFrame({"cat": ["A", "A", "A", "A"]}), name="single_cat")
        result = run_analysis(ds, _spec(["cat"]))
        assert any("범주가 1개" in w for w in result.warnings)

    def test_zero_expected_frequency_returns_warning(self):
        """기대 빈도 0인 범주 → 경고."""
        ds = Dataset(
            data=pd.DataFrame({"cat": ["A", "B", "A", "B"]}),
            name="zero_exp",
        )
        # C에 비율 할당하지만 관찰에 C 없음 → value_counts에 C 없으므로
        # 실제로는 관찰에 없는 범주는 무시됨, 기대비율에서 0이 되는 케이스 테스트
        ratios = {"A": 1, "B": 0}  # B 기대 빈도 = 0
        result = run_analysis(ds, _spec(["cat"], expected_ratios=ratios))
        assert any("기대 빈도가 0" in w for w in result.warnings)

    def test_result_is_analysis_result_type(self, dice_dataset):
        """반환 타입이 AnalysisResult."""
        from statworkbench.analysis.result import AnalysisResult
        result = run_analysis(dice_dataset, _spec(["face"]))
        assert isinstance(result, AnalysisResult)

    def test_no_warnings_on_valid_input(self, dice_dataset):
        """정상 입력 → 경고 없음."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        assert len(result.warnings) == 0


# ──────────────────────────────────────────────────────────────
# 11. 분석 결과 노트
# ──────────────────────────────────────────────────────────────

class TestAnalysisNotes:

    def test_notes_contain_chi2_info(self, dice_dataset):
        """노트에 Chi-Square, df, p 정보 포함."""
        result = run_analysis(dice_dataset, _spec(["face"]))
        assert len(result.notes) > 0
        note = result.notes[0]
        assert "Chi-Square" in note
        assert "df" in note

    def test_notes_count_equals_variable_count(self, two_var_dataset):
        """노트 수 = 검정 변수 수."""
        result = run_analysis(two_var_dataset, _spec(["cat1", "cat2"]))
        assert len(result.notes) == 2
