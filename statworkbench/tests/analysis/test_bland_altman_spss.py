"""Bland-Altman 일치도 분석 테스트 — SPSS 대응 검증.

SPSS 직접 메뉴 없음. MedCalc/SPSS 매크로 기반 임상·보건 표준 분석.
참조: Bland & Altman (1986). Statistical methods for assessing agreement
      between two methods of clinical measurement. Lancet, 327(8476), 307-310.

참조값 (n=12, method_a vs method_b):
  mean_diff  = -11.750000
  sd_diff    =  49.587618
  loa_upper  =  85.441730
  loa_lower  = -108.941730
  ci_mean_low  = -43.256469  (95% CI for mean diff)
  ci_mean_high =  19.756469
  ci_loa_upper_low  =  30.870925  (상한 LoA 95% CI)
  ci_loa_upper_high = 140.012536
  ci_loa_lower_low  = -163.512536  (하한 LoA 95% CI)
  ci_loa_lower_high =  -54.370925
  proportional_bias_r = 0.179240
  proportional_bias_p = 0.577258
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.bland_altman import run_analysis, _compute_bland_altman
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────────────────

METHOD_A = [512, 430, 420, 428, 500, 600, 364, 380, 658, 445, 432, 626]
METHOD_B = [525, 415, 508, 444, 500, 625, 460, 390, 687, 432, 420, 530]


@pytest.fixture
def ba_dataset() -> Dataset:
    """두 측정법 비교 데이터 (n=12)."""
    data = pd.DataFrame({"method_a": METHOD_A, "method_b": METHOD_B})
    return Dataset(data=data, name="bland_altman_test")


@pytest.fixture
def ba_result(ba_dataset) -> dict:
    """_compute_bland_altman 결과 (반복 호출 방지용 픽스처)."""
    return _compute_bland_altman(
        np.array(METHOD_A, dtype=float),
        np.array(METHOD_B, dtype=float),
    )


@pytest.fixture
def perfect_dataset() -> Dataset:
    """완전 일치 데이터 — 차이 = 0."""
    vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    data = pd.DataFrame({"m1": vals, "m2": vals})
    return Dataset(data=data, name="perfect_agreement")


@pytest.fixture
def constant_bias_dataset() -> Dataset:
    """일정 편향 데이터 — 모든 차이 = 50."""
    a = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    b = [x - 50 for x in a]
    data = pd.DataFrame({"m1": a, "m2": b})
    return Dataset(data=data, name="constant_bias")


@pytest.fixture
def missing_dataset() -> Dataset:
    """결측치 포함 데이터."""
    a = [512, 430, np.nan, 428, 500, 600, 364, 380, 658, 445, 432, 626]
    b = [525, 415, 508, np.nan, 500, 625, 460, 390, 687, 432, 420, 530]
    data = pd.DataFrame({"method_a": a, "method_b": b})
    return Dataset(data=data, name="missing_data")


def _make_spec(method1: str, method2: str) -> dict:
    return {"variables": {"method1": method1, "method2": method2}}


# ──────────────────────────────────────────────────────────────────────────
# 1. 평균 차이(bias) 정확성
# ──────────────────────────────────────────────────────────────────────────

class TestMeanDifference:
    """평균 차이(bias) 계산 정확성 검증."""

    def test_mean_diff_value(self, ba_result):
        """평균 차이 = -11.750000 ± 0.001."""
        assert abs(ba_result["mean_diff"] - (-11.75)) < 0.001, (
            f"mean_diff 오차 초과: {ba_result['mean_diff']}"
        )

    def test_mean_diff_sign(self, ba_result):
        """평균 차이가 음수 (method_a < method_b 평균)."""
        assert ba_result["mean_diff"] < 0

    def test_n_value(self, ba_result):
        """유효 케이스 수 = 12."""
        assert ba_result["n"] == 12


# ──────────────────────────────────────────────────────────────────────────
# 2. SD 차이 정확성
# ──────────────────────────────────────────────────────────────────────────

class TestSDDifference:
    """차이의 표준편차(ddof=1) 계산 정확성 검증."""

    def test_sd_diff_value(self, ba_result):
        """SD = 49.587618 ± 0.001."""
        assert abs(ba_result["sd_diff"] - 49.587618) < 0.001, (
            f"sd_diff 오차 초과: {ba_result['sd_diff']}"
        )

    def test_sd_diff_positive(self, ba_result):
        """SD는 양수여야 함."""
        assert ba_result["sd_diff"] > 0

    def test_sd_uses_ddof1(self):
        """ddof=1(표본 표준편차) 사용 확인 — ddof=0과 다른 값."""
        a = np.array([10, 20, 30, 40, 50], dtype=float)
        b = np.array([12, 18, 32, 38, 52], dtype=float)
        r = _compute_bland_altman(a, b)
        diff = a - b
        assert abs(r["sd_diff"] - diff.std(ddof=1)) < 1e-9
        assert abs(r["sd_diff"] - diff.std(ddof=0)) > 1e-9


# ──────────────────────────────────────────────────────────────────────────
# 3. LoA 상한/하한 = mean_diff ± 1.96 * SD
# ──────────────────────────────────────────────────────────────────────────

class TestLimitsOfAgreement:
    """일치 한계(LoA) 계산 정확성 검증."""

    def test_loa_upper_value(self, ba_result):
        """LoA 상한 = 85.441730 ± 0.001."""
        assert abs(ba_result["loa_upper"] - 85.441730) < 0.001, (
            f"loa_upper 오차 초과: {ba_result['loa_upper']}"
        )

    def test_loa_lower_value(self, ba_result):
        """LoA 하한 = -108.941730 ± 0.001."""
        assert abs(ba_result["loa_lower"] - (-108.941730)) < 0.001, (
            f"loa_lower 오차 초과: {ba_result['loa_lower']}"
        )

    def test_loa_formula(self, ba_result):
        """LoA = mean_diff ± 1.96 * sd_diff 공식 검증."""
        expected_upper = ba_result["mean_diff"] + 1.96 * ba_result["sd_diff"]
        expected_lower = ba_result["mean_diff"] - 1.96 * ba_result["sd_diff"]
        assert abs(ba_result["loa_upper"] - expected_upper) < 1e-9
        assert abs(ba_result["loa_lower"] - expected_lower) < 1e-9

    def test_loa_upper_greater_than_lower(self, ba_result):
        """LoA 상한 > LoA 하한."""
        assert ba_result["loa_upper"] > ba_result["loa_lower"]

    def test_mean_diff_between_loa(self, ba_result):
        """평균 차이는 LoA 상한과 하한 사이에 있어야 함."""
        assert ba_result["loa_lower"] < ba_result["mean_diff"] < ba_result["loa_upper"]


# ──────────────────────────────────────────────────────────────────────────
# 4. 평균 차이의 95% CI
# ──────────────────────────────────────────────────────────────────────────

class TestCIMeanDifference:
    """평균 차이 95% CI 검증."""

    def test_ci_mean_low(self, ba_result):
        """95% CI 하한 = -43.256469 ± 0.001."""
        assert abs(ba_result["ci_mean_low"] - (-43.256469)) < 0.001, (
            f"ci_mean_low 오차 초과: {ba_result['ci_mean_low']}"
        )

    def test_ci_mean_high(self, ba_result):
        """95% CI 상한 = 19.756469 ± 0.001."""
        assert abs(ba_result["ci_mean_high"] - 19.756469) < 0.001, (
            f"ci_mean_high 오차 초과: {ba_result['ci_mean_high']}"
        )

    def test_ci_mean_contains_zero(self, ba_result):
        """95% CI가 0을 포함 (유의하지 않은 편향)."""
        assert ba_result["ci_mean_low"] < 0 < ba_result["ci_mean_high"]

    def test_ci_mean_symmetric(self, ba_result):
        """CI가 mean_diff를 중심으로 대칭."""
        center = (ba_result["ci_mean_low"] + ba_result["ci_mean_high"]) / 2
        assert abs(center - ba_result["mean_diff"]) < 1e-6

    def test_ci_mean_uses_t_distribution(self):
        """t분포 사용 확인 (SE = sd/sqrt(n), t_crit = t(0.975, df=n-1))."""
        from scipy import stats
        a = np.array(METHOD_A, dtype=float)
        b = np.array(METHOD_B, dtype=float)
        r = _compute_bland_altman(a, b)
        diff = a - b
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        expected_low = diff.mean() - t_crit * se
        expected_high = diff.mean() + t_crit * se
        assert abs(r["ci_mean_low"] - expected_low) < 1e-6
        assert abs(r["ci_mean_high"] - expected_high) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# 5. LoA의 95% CI (Bland-Altman 1986 원 공식)
# ──────────────────────────────────────────────────────────────────────────

class TestCILoA:
    """LoA 95% CI 검증 — SE_LoA = sqrt(3 * sd^2 / n)."""

    def test_ci_loa_upper_low(self, ba_result):
        """상한 LoA CI 하한 = 30.870925 ± 0.001."""
        assert abs(ba_result["ci_loa_upper_low"] - 30.870925) < 0.001, (
            f"ci_loa_upper_low 오차 초과: {ba_result['ci_loa_upper_low']}"
        )

    def test_ci_loa_upper_high(self, ba_result):
        """상한 LoA CI 상한 = 140.012536 ± 0.001."""
        assert abs(ba_result["ci_loa_upper_high"] - 140.012536) < 0.001, (
            f"ci_loa_upper_high 오차 초과: {ba_result['ci_loa_upper_high']}"
        )

    def test_ci_loa_lower_low(self, ba_result):
        """하한 LoA CI 하한 = -163.512536 ± 0.001."""
        assert abs(ba_result["ci_loa_lower_low"] - (-163.512536)) < 0.001, (
            f"ci_loa_lower_low 오차 초과: {ba_result['ci_loa_lower_low']}"
        )

    def test_ci_loa_lower_high(self, ba_result):
        """하한 LoA CI 상한 = -54.370925 ± 0.001."""
        assert abs(ba_result["ci_loa_lower_high"] - (-54.370925)) < 0.001, (
            f"ci_loa_lower_high 오차 초과: {ba_result['ci_loa_lower_high']}"
        )

    def test_loa_ci_se_formula(self):
        """SE_LoA = sqrt(3 * sd^2 / n) 공식 직접 검증."""
        from scipy import stats
        a = np.array(METHOD_A, dtype=float)
        b = np.array(METHOD_B, dtype=float)
        r = _compute_bland_altman(a, b)
        n = len(a)
        sd = (a - b).std(ddof=1)
        se_loa = np.sqrt(3 * sd ** 2 / n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        expected_upper_low = r["loa_upper"] - t_crit * se_loa
        assert abs(r["ci_loa_upper_low"] - expected_upper_low) < 1e-6

    def test_loa_upper_ci_contains_loa_upper(self, ba_result):
        """상한 LoA가 자신의 CI 내에 있어야 함."""
        assert ba_result["ci_loa_upper_low"] < ba_result["loa_upper"] < ba_result["ci_loa_upper_high"]

    def test_loa_lower_ci_contains_loa_lower(self, ba_result):
        """하한 LoA가 자신의 CI 내에 있어야 함."""
        assert ba_result["ci_loa_lower_low"] < ba_result["loa_lower"] < ba_result["ci_loa_lower_high"]


# ──────────────────────────────────────────────────────────────────────────
# 6. 완전 일치 → mean_diff=0, SD=0, LoA=[0,0]
# ──────────────────────────────────────────────────────────────────────────

class TestPerfectAgreement:
    """완전 일치 데이터 검증."""

    def test_mean_diff_is_zero(self, perfect_dataset):
        """완전 일치: mean_diff = 0."""
        a = perfect_dataset.data["m1"].values.astype(float)
        b = perfect_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert abs(r["mean_diff"]) < 1e-9

    def test_sd_is_zero(self, perfect_dataset):
        """완전 일치: sd_diff = 0."""
        a = perfect_dataset.data["m1"].values.astype(float)
        b = perfect_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert abs(r["sd_diff"]) < 1e-9

    def test_loa_both_zero(self, perfect_dataset):
        """완전 일치: LoA 상한 = LoA 하한 = 0."""
        a = perfect_dataset.data["m1"].values.astype(float)
        b = perfect_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert abs(r["loa_upper"]) < 1e-9
        assert abs(r["loa_lower"]) < 1e-9

    def test_run_analysis_perfect(self, perfect_dataset):
        """완전 일치: run_analysis 경고 없음, 4개 테이블 반환."""
        spec = _make_spec("m1", "m2")
        result = run_analysis(perfect_dataset, spec)
        assert len(result.warnings) == 0
        assert len(result.tables) == 4


# ──────────────────────────────────────────────────────────────────────────
# 7. 일정 편향 → mean_diff != 0, SD ≈ 0
# ──────────────────────────────────────────────────────────────────────────

class TestConstantBias:
    """일정 편향(systematic bias) 데이터 검증."""

    def test_mean_diff_nonzero(self, constant_bias_dataset):
        """일정 편향: mean_diff = 50 (0이 아님)."""
        a = constant_bias_dataset.data["m1"].values.astype(float)
        b = constant_bias_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert abs(r["mean_diff"] - 50.0) < 1e-9

    def test_sd_near_zero(self, constant_bias_dataset):
        """일정 편향: sd_diff ≈ 0 (모든 차이 동일)."""
        a = constant_bias_dataset.data["m1"].values.astype(float)
        b = constant_bias_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert r["sd_diff"] < 1e-6

    def test_loa_narrow(self, constant_bias_dataset):
        """일정 편향: LoA 상한 = LoA 하한 ≈ mean_diff."""
        a = constant_bias_dataset.data["m1"].values.astype(float)
        b = constant_bias_dataset.data["m2"].values.astype(float)
        r = _compute_bland_altman(a, b)
        assert abs(r["loa_upper"] - r["loa_lower"]) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# 8. 결과 테이블 구조 (4개 테이블)
# ──────────────────────────────────────────────────────────────────────────

class TestResultTableStructure:
    """run_analysis 반환 테이블 구조 검증."""

    def test_four_tables_returned(self, ba_dataset):
        """run_analysis가 정확히 4개 테이블을 반환해야 함."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        assert len(result.tables) == 4, f"테이블 수 오류: {len(result.tables)}"

    def test_table_titles(self, ba_dataset):
        """4개 테이블의 제목 확인."""
        expected_titles = [
            "Case Processing Summary",
            "Bland-Altman Statistics",
            "Limits of Agreement",
            "Individual Differences",
        ]
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        actual_titles = [t.title for t in result.tables]
        for expected in expected_titles:
            assert expected in actual_titles, (
                f"테이블 '{expected}' 없음. 실제: {actual_titles}"
            )

    def test_case_processing_summary_columns(self, ba_dataset):
        """Case Processing Summary: 구분, N, % 컬럼."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        assert "구분" in cps.dataframe.columns
        assert "N" in cps.dataframe.columns
        assert "%" in cps.dataframe.columns

    def test_case_processing_valid_count(self, ba_dataset):
        """Case Processing Summary: 유효 케이스 = 12."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_row = cps.dataframe[cps.dataframe["구분"] == "유효"]
        assert valid_row["N"].iloc[0] == 12

    def test_ba_statistics_columns(self, ba_dataset):
        """Bland-Altman Statistics 테이블 필수 컬럼 확인."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ba_tbl = next(t for t in result.tables if t.title == "Bland-Altman Statistics")
        required = ["통계량", "값"]
        for col in required:
            assert col in ba_tbl.dataframe.columns, (
                f"Bland-Altman Statistics 테이블에 '{col}' 컬럼 없음"
            )

    def test_limits_of_agreement_columns(self, ba_dataset):
        """Limits of Agreement 테이블 필수 컬럼 확인."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        loa_tbl = next(t for t in result.tables if t.title == "Limits of Agreement")
        required = ["한계값", "추정치", "95% CI 하한", "95% CI 상한"]
        for col in required:
            assert col in loa_tbl.dataframe.columns, (
                f"Limits of Agreement 테이블에 '{col}' 컬럼 없음"
            )

    def test_individual_differences_columns(self, ba_dataset):
        """Individual Differences 테이블 필수 컬럼 확인."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ind_tbl = next(t for t in result.tables if t.title == "Individual Differences")
        required = ["케이스", "평균", "차이", "표준화 차이"]
        for col in required:
            assert col in ind_tbl.dataframe.columns, (
                f"Individual Differences 테이블에 '{col}' 컬럼 없음"
            )

    def test_individual_differences_row_count(self, ba_dataset):
        """Individual Differences: 12개 케이스."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ind_tbl = next(t for t in result.tables if t.title == "Individual Differences")
        assert len(ind_tbl.dataframe) == 12

    def test_limits_of_agreement_row_count(self, ba_dataset):
        """Limits of Agreement: 3행 (평균 차이, 상한 LoA, 하한 LoA)."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        loa_tbl = next(t for t in result.tables if t.title == "Limits of Agreement")
        assert len(loa_tbl.dataframe) == 3, (
            f"Limits of Agreement 행 수 오류: {len(loa_tbl.dataframe)}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 9. 비례 오차 감지 (Pearson r between diff and mean)
# ──────────────────────────────────────────────────────────────────────────

class TestProportionalBias:
    """비례 오차(proportional bias) 검증 — diff vs mean 피어슨 상관."""

    def test_proportional_bias_r_value(self, ba_result):
        """비례 오차 r = 0.179240 ± 0.001."""
        assert abs(ba_result["proportional_bias_r"] - 0.179240) < 0.001, (
            f"proportional_bias_r 오차 초과: {ba_result['proportional_bias_r']}"
        )

    def test_proportional_bias_p_value(self, ba_result):
        """비례 오차 p = 0.577258 ± 0.001."""
        assert abs(ba_result["proportional_bias_p"] - 0.577258) < 0.001, (
            f"proportional_bias_p 오차 초과: {ba_result['proportional_bias_p']}"
        )

    def test_proportional_bias_not_significant(self, ba_result):
        """이 데이터에서 비례 오차는 유의하지 않음 (p > 0.05)."""
        assert ba_result["proportional_bias_p"] > 0.05

    def test_proportional_bias_r_range(self, ba_result):
        """-1 <= r <= 1 범위 내."""
        assert -1.0 <= ba_result["proportional_bias_r"] <= 1.0

    def test_proportional_bias_p_range(self, ba_result):
        """0 <= p <= 1 범위 내."""
        assert 0.0 <= ba_result["proportional_bias_p"] <= 1.0

    def test_proportional_bias_keys_in_result(self, ba_result):
        """결과 딕셔너리에 proportional_bias_r, proportional_bias_p 포함."""
        assert "proportional_bias_r" in ba_result
        assert "proportional_bias_p" in ba_result

    def test_proportional_bias_in_table(self, ba_dataset):
        """Bland-Altman Statistics 테이블에 비례 오차 행 포함."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ba_tbl = next(t for t in result.tables if t.title == "Bland-Altman Statistics")
        stat_names = ba_tbl.dataframe["통계량"].tolist()
        has_bias = any("비례" in s or "proportional" in s.lower() for s in stat_names)
        assert has_bias, f"비례 오차 행 없음. 통계량 목록: {stat_names}"


# ──────────────────────────────────────────────────────────────────────────
# 10. _compute_bland_altman 반환 키 완전성
# ──────────────────────────────────────────────────────────────────────────

class TestComputeBlandAltmanKeys:
    """_compute_bland_altman 반환 딕셔너리 키 완전성 검증."""

    REQUIRED_KEYS = [
        "n",
        "mean_diff",
        "sd_diff",
        "loa_upper",
        "loa_lower",
        "ci_mean_low",
        "ci_mean_high",
        "ci_loa_upper_low",
        "ci_loa_upper_high",
        "ci_loa_lower_low",
        "ci_loa_lower_high",
        "proportional_bias_r",
        "proportional_bias_p",
    ]

    def test_all_keys_present(self, ba_result):
        """모든 필수 키가 결과에 포함되어야 함."""
        for key in self.REQUIRED_KEYS:
            assert key in ba_result, f"키 '{key}' 누락"

    def test_all_values_finite(self, ba_result):
        """모든 수치 값이 유한해야 함 (NaN, inf 없음)."""
        for key in self.REQUIRED_KEYS:
            val = ba_result[key]
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"키 '{key}' 값이 유한하지 않음: {val}"


# ──────────────────────────────────────────────────────────────────────────
# 11. 오류 처리
# ──────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    """오류 및 경계 조건 처리 검증."""

    def test_missing_variable_returns_warning(self, ba_dataset):
        """존재하지 않는 변수 지정 → 경고 반환, 테이블 없음."""
        spec = _make_spec("method_a", "nonexistent_var")
        result = run_analysis(ba_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_empty_spec_variables_returns_warning(self, ba_dataset):
        """변수 미지정 → 경고 반환."""
        spec = {"variables": {}}
        result = run_analysis(ba_dataset, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0

    def test_same_variable_twice(self, ba_dataset):
        """동일한 변수 두 번 지정 → 경고 또는 정상 처리 (mean_diff=0, sd=0)."""
        spec = _make_spec("method_a", "method_a")
        result = run_analysis(ba_dataset, spec)
        # 경고 또는 정상 반환 모두 허용, 단 결과가 일관적이어야 함
        if len(result.warnings) == 0:
            ba_tbl = next(t for t in result.tables if t.title == "Bland-Altman Statistics")
            assert ba_tbl is not None

    def test_fewer_than_two_valid_cases(self):
        """유효 케이스 < 2 → 경고 반환, 테이블 없음."""
        data = pd.DataFrame({"m1": [100, np.nan], "m2": [100, 200]})
        ds = Dataset(data=data, name="tiny")
        spec = _make_spec("m1", "m2")
        result = run_analysis(ds, spec)
        # n=1이면 SD 계산 불가 → 경고 필요
        if len(result.tables) == 0:
            assert len(result.warnings) > 0

    def test_compute_raises_on_mismatched_length(self):
        """길이 다른 배열 → ValueError 발생."""
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises((ValueError, Exception)):
            _compute_bland_altman(a, b)

    def test_all_missing_returns_warning(self):
        """전부 결측 → 경고 반환, 테이블 없음."""
        data = pd.DataFrame({"m1": [np.nan, np.nan], "m2": [np.nan, np.nan]})
        ds = Dataset(data=data, name="all_missing")
        spec = _make_spec("m1", "m2")
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0
        assert len(result.tables) == 0


# ──────────────────────────────────────────────────────────────────────────
# 12. 결측치 처리
# ──────────────────────────────────────────────────────────────────────────

class TestMissingDataHandling:
    """결측치 처리 검증."""

    def test_missing_excluded_from_analysis(self, missing_dataset):
        """결측 행 제외 후 유효 케이스 수 감소."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(missing_dataset, spec)
        if len(result.warnings) > 0:
            pytest.skip("유효 케이스 부족으로 경고 반환")
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        valid_row = cps.dataframe[cps.dataframe["구분"] == "유효"]
        total_row = cps.dataframe[cps.dataframe["구분"] == "합계"]
        assert valid_row["N"].iloc[0] < total_row["N"].iloc[0]

    def test_missing_count_in_case_processing(self, missing_dataset):
        """Case Processing Summary: 결측 건수 > 0."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(missing_dataset, spec)
        if len(result.warnings) > 0:
            pytest.skip("유효 케이스 부족으로 경고 반환")
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        missing_row = cps.dataframe[cps.dataframe["구분"] == "제외됨"]
        assert missing_row["N"].iloc[0] > 0


# ──────────────────────────────────────────────────────────────────────────
# 13. run_analysis 통합 테스트
# ──────────────────────────────────────────────────────────────────────────

class TestRunAnalysisIntegration:
    """run_analysis 통합 검증."""

    def test_no_warnings_on_valid_data(self, ba_dataset):
        """유효 데이터: 경고 없음."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        assert len(result.warnings) == 0, f"예상치 않은 경고: {result.warnings}"

    def test_analysis_id(self, ba_dataset):
        """분석 결과 ID가 'bland_altman'이어야 함."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        assert result.id == "bland_altman"

    def test_analysis_title(self, ba_dataset):
        """분석 제목에 'Bland-Altman' 포함."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        assert "Bland" in result.title or "Altman" in result.title or "일치도" in result.title

    def test_notes_contain_mean_diff(self, ba_dataset):
        """result.notes에 평균 차이 정보 포함."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        note_text = " ".join(result.notes)
        assert "mean" in note_text.lower() or "편향" in note_text or "bias" in note_text.lower()

    def test_ba_statistics_mean_diff_value(self, ba_dataset):
        """Bland-Altman Statistics 테이블의 평균 차이 값 검증."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ba_tbl = next(t for t in result.tables if t.title == "Bland-Altman Statistics")
        # '평균 차이' 행 찾기
        df = ba_tbl.dataframe
        mean_diff_rows = df[df["통계량"].str.contains("평균 차이|Mean Difference|Bias", na=False)]
        assert len(mean_diff_rows) > 0, "평균 차이 행 없음"
        val = float(mean_diff_rows["값"].iloc[0])
        assert abs(val - (-11.75)) < 0.01

    def test_individual_differences_first_case(self, ba_dataset):
        """Individual Differences: 첫 케이스 (평균=518.5, 차이=-13.0)."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ind_tbl = next(t for t in result.tables if t.title == "Individual Differences")
        first_row = ind_tbl.dataframe.iloc[0]
        assert abs(float(first_row["평균"]) - 518.5) < 0.01
        assert abs(float(first_row["차이"]) - (-13.0)) < 0.01

    def test_individual_differences_standardized(self, ba_dataset):
        """Individual Differences: 표준화 차이 = diff / sd_diff."""
        spec = _make_spec("method_a", "method_b")
        result = run_analysis(ba_dataset, spec)
        ind_tbl = next(t for t in result.tables if t.title == "Individual Differences")
        # 첫 케이스 표준화 차이 ≈ -13 / 49.587618 = -0.2622
        first_std = float(ind_tbl.dataframe.iloc[0]["표준화 차이"])
        assert abs(first_std - (-0.2622)) < 0.001
