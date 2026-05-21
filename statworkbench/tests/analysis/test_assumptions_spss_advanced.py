"""가정 검증 고급 SPSS 29/30 호환 테스트.

검증 항목:
- Shapiro-Wilk 정규성 검정 (W 통계량, p-value) — SPSS Explore 출력
- Levene 등분산 검정 (F, p) — center='mean', SPSS 기본값
- Brown-Forsythe 검정 (F, p) — center='median', 왜도 로버스트
- 다중 그룹 등분산 (3개 그룹)
- 소표본 경고 (n<20)
- 준정규 데이터 정규성 통과 판정
- 등분산 데이터 Levene p > 0.05 확인
- 이분산 데이터 Levene p < 0.05 확인
- 수학적 불변량: Levene p ≤ 1, W ∈ [0, 1]

SPSS 29 참조 출력:
    G1 = [2.1,3.4,2.8,3.9,2.5,3.2,2.7,3.6,2.3,3.1] (n=10)
    G2 = [5.1,6.4,5.8,6.9,5.5,6.2,5.7,6.6,5.3,6.1] (n=10)

    Shapiro-Wilk:
        G1: W=0.9794, Sig.=0.962 → 정규성 가정 충족
        G2: W=0.9794, Sig.=0.962 → 정규성 가정 충족

    Levene's Test (mean center):
        F=0.000, Sig.=1.000 → 등분산 가정 충족

    Brown-Forsythe (median center):
        F=0.000, Sig.=1.000 → 등분산 가정 충족

독립 검증:
    Python: scipy.stats.shapiro, scipy.stats.levene
    R: shapiro.test(x), leveneTest(y~group, data=df, center='mean')
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.assumptions import (
    check_normality,
    check_homogeneity_of_variance,
    check_homogeneity_of_variance_from_groups,
    shapiro_test,
    levene_test,
    prepare_analysis_frame,
    get_case_processing_summary,
)


def _approx(val, tol=0.001):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 참조 데이터 (SPSS 29 기준)
# ──────────────────────────────────────────────────────────────

G1 = np.array([2.1, 3.4, 2.8, 3.9, 2.5, 3.2, 2.7, 3.6, 2.3, 3.1], dtype=float)
G2 = np.array([5.1, 6.4, 5.8, 6.9, 5.5, 6.2, 5.7, 6.6, 5.3, 6.1], dtype=float)

# scipy 참조값
_SW1 = stats.shapiro(G1)
SW1_W = float(_SW1.statistic)   # 0.9794
SW1_P = float(_SW1.pvalue)      # 0.962

_SW2 = stats.shapiro(G2)
SW2_W = float(_SW2.statistic)
SW2_P = float(_SW2.pvalue)

_LEV_MEAN = stats.levene(G1, G2, center='mean')
LEV_MEAN_F = float(_LEV_MEAN.statistic)   # ≈ 0.000
LEV_MEAN_P = float(_LEV_MEAN.pvalue)      # ≈ 1.000

_LEV_MED = stats.levene(G1, G2, center='median')
LEV_MED_F = float(_LEV_MED.statistic)
LEV_MED_P = float(_LEV_MED.pvalue)

# 준정규 데이터 (정규성 통과 예상)
NORMAL_DATA = np.array([2.1, 1.8, 2.5, 2.0, 2.3, 1.9, 2.2, 2.4, 2.1, 2.0,
                        1.8, 2.6, 2.1, 1.9, 2.3, 2.0, 2.4, 1.7, 2.2, 2.5], dtype=float)
_SW_N = stats.shapiro(NORMAL_DATA)
NORMAL_W = float(_SW_N.statistic)    # 0.968
NORMAL_P = float(_SW_N.pvalue)       # 0.705 > 0.05

# 이분산 데이터
HET_G1 = np.array([1, 1, 2, 1, 1, 2, 1, 2, 1, 1], dtype=float)    # 분산 작음
HET_G2 = np.array([1, 10, 20, 5, 15, 8, 25, 3, 12, 18], dtype=float)  # 분산 큼
_LEV_HET = stats.levene(HET_G1, HET_G2, center='mean')
LEV_HET_F = float(_LEV_HET.statistic)
LEV_HET_P = float(_LEV_HET.pvalue)


# ──────────────────────────────────────────────────────────────
# 1. Shapiro-Wilk 정규성 검정
# ──────────────────────────────────────────────────────────────

class TestShapiroWilkSPSS:
    """Shapiro-Wilk W 통계량과 p-value가 SPSS 29 출력과 일치."""

    def test_g1_w_statistic(self):
        result = check_normality(pd.Series(G1))
        assert result["statistic"] == _approx(SW1_W)

    def test_g1_p_value(self):
        result = check_normality(pd.Series(G1))
        assert result["p_value"] == _approx(SW1_P, 0.01)

    def test_g1_normal_judgment(self):
        """p > 0.05 → normal=True."""
        result = check_normality(pd.Series(G1))
        assert result["normal"] is True

    def test_g2_w_statistic(self):
        result = check_normality(pd.Series(G2))
        assert result["statistic"] == _approx(SW2_W)

    def test_normal_data_passes(self):
        """준정규 데이터 → normal=True."""
        result = check_normality(pd.Series(NORMAL_DATA))
        assert result["normal"] is True

    def test_normal_data_w(self):
        result = check_normality(pd.Series(NORMAL_DATA))
        assert result["statistic"] == _approx(NORMAL_W, 0.001)

    def test_normal_data_p(self):
        result = check_normality(pd.Series(NORMAL_DATA))
        assert result["p_value"] == _approx(NORMAL_P, 0.01)

    def test_highly_skewed_fails_normality(self):
        """강한 우왜도 데이터 → normal=False."""
        skewed = np.array([1, 1, 1, 1, 1, 1, 1, 2, 5, 100], dtype=float)
        result = check_normality(pd.Series(skewed))
        assert result["normal"] is False

    def test_test_name_field(self):
        result = check_normality(pd.Series(G1))
        assert result["test"] == "Shapiro-Wilk"

    def test_n_field_excludes_nan(self):
        """NaN이 포함된 경우 n이 유효 케이스 수."""
        arr = pd.Series([1, 2, np.nan, 3, 4, np.nan, 5])
        result = check_normality(arr)
        assert result["n"] == 5

    def test_shapiro_test_simple_api(self):
        """shapiro_test() 단순 API: (W, p) 반환."""
        w, p = shapiro_test(G1)
        assert w == _approx(SW1_W)
        assert p == _approx(SW1_P, 0.01)

    def test_small_n_warning(self):
        """n < 20 → 소표본 경고 포함."""
        result = check_normality(pd.Series(G1))  # n=10
        assert any("small" in w.lower() or "20" in w for w in result["warnings"])

    def test_n_less_than_3_returns_nan(self):
        """n < 3 → 검정 불가, statistic=NaN."""
        result = check_normality(pd.Series([1.0, 2.0]))
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])

    def test_n_3_minimum_valid(self):
        """n=3 → 최소 유효 케이스, statistic이 NaN이 아님."""
        result = check_normality(pd.Series([1.0, 2.0, 3.0]))
        assert not np.isnan(result["statistic"])


# ──────────────────────────────────────────────────────────────
# 2. Levene 등분산 검정
# ──────────────────────────────────────────────────────────────

class TestLeveneSPSS:
    """Levene F 통계량과 p-value가 SPSS 29 출력과 일치."""

    def test_equal_var_groups_f_near_zero(self):
        """G1, G2 동일 분산 패턴 → F ≈ 0."""
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert result["statistic"] == _approx(LEV_MEAN_F, 0.001)

    def test_equal_var_groups_p_near_one(self):
        """G1, G2 동일 분산 패턴 → p ≈ 1.0."""
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert result["p_value"] == _approx(LEV_MEAN_P, 0.01)

    def test_equal_var_homogeneous_true(self):
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert result["homogeneous"] is True

    def test_test_name_levene(self):
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert result["test"] == "Levene"

    def test_heterogeneous_data_p_small(self):
        """분산이 크게 다른 두 그룹 → p < 0.05."""
        result = check_homogeneity_of_variance(HET_G1, HET_G2, center='mean')
        assert result["p_value"] < 0.05
        assert result["homogeneous"] is False

    def test_heterogeneous_data_f_large(self):
        result = check_homogeneity_of_variance(HET_G1, HET_G2, center='mean')
        assert result["statistic"] == _approx(LEV_HET_F, 0.01)

    def test_levene_test_simple_api(self):
        """levene_test() 단순 API: (F, p) 반환."""
        f, p = levene_test(G1, G2, center='mean')
        assert f == _approx(LEV_MEAN_F, 0.001)
        assert p == _approx(LEV_MEAN_P, 0.01)

    def test_from_groups_api(self):
        """check_homogeneity_of_variance_from_groups() — Series + group label."""
        data = pd.Series(np.concatenate([G1, G2]))
        group = pd.Series(["G1"] * 10 + ["G2"] * 10)
        result = check_homogeneity_of_variance_from_groups(data, group, center='mean')
        assert result["statistic"] == _approx(LEV_MEAN_F, 0.001)
        assert result["p_value"] == _approx(LEV_MEAN_P, 0.01)

    def test_insufficient_groups_returns_nan(self):
        """그룹이 1개 이하 → statistic=NaN."""
        result = check_homogeneity_of_variance(G1, center='mean')
        assert np.isnan(result["statistic"])


# ──────────────────────────────────────────────────────────────
# 3. Brown-Forsythe 검정 (center='median')
# ──────────────────────────────────────────────────────────────

class TestBrownForsytheSPSS:
    """Brown-Forsythe — 왜도 데이터에 로버스트한 등분산 검정."""

    def test_test_name_is_brown_forsythe(self):
        result = check_homogeneity_of_variance(G1, G2, center='median')
        assert result["test"] == "Brown-Forsythe"

    def test_equal_var_groups_f(self):
        result = check_homogeneity_of_variance(G1, G2, center='median')
        assert result["statistic"] == _approx(LEV_MED_F, 0.001)

    def test_equal_var_groups_p(self):
        result = check_homogeneity_of_variance(G1, G2, center='median')
        assert result["p_value"] == _approx(LEV_MED_P, 0.01)

    def test_homogeneous_true(self):
        result = check_homogeneity_of_variance(G1, G2, center='median')
        assert result["homogeneous"] is True

    def test_bf_and_levene_agree_for_symmetric_data(self):
        """대칭 데이터에서 B-F와 Levene의 결론(homogeneous) 일치."""
        lev = check_homogeneity_of_variance(G1, G2, center='mean')
        bf  = check_homogeneity_of_variance(G1, G2, center='median')
        assert lev["homogeneous"] == bf["homogeneous"]

    def test_levene_simple_api_median(self):
        """levene_test() — center='median'."""
        f, p = levene_test(G1, G2, center='median')
        assert f == _approx(LEV_MED_F, 0.001)


# ──────────────────────────────────────────────────────────────
# 4. 3개 그룹 등분산
# ──────────────────────────────────────────────────────────────

class TestThreeGroupHomogeneity:
    """3개 그룹 Levene 검정 — SPSS One-Way ANOVA 전처리 단계."""

    def test_three_equal_var_groups_homogeneous(self):
        """분산이 동일한 3그룹 → p > 0.05."""
        G_A = np.array([10, 12, 11, 13, 10, 12, 11, 10, 13, 12], dtype=float)
        G_B = G_A.copy()
        G_C = G_A.copy()
        result = check_homogeneity_of_variance(G_A, G_B, G_C, center='mean')
        assert result["homogeneous"] is True

    def test_three_unequal_var_groups_not_homogeneous(self):
        """분산이 크게 다른 3그룹 → p < 0.05."""
        GA = np.array([1.0] * 10)
        GB = np.array([1, 5, 10, 15, 20, 3, 8, 12, 18, 25], dtype=float)
        GC = np.array([1, 100, 200, 5, 50, 150, 10, 80, 120, 30], dtype=float)
        result = check_homogeneity_of_variance(GA, GB, GC, center='mean')
        assert result["homogeneous"] is False

    def test_three_groups_df_structure(self):
        """3그룹 결과 딕셔너리 구조 확인."""
        result = check_homogeneity_of_variance(G1, G2, G1, center='mean')
        assert "test" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "homogeneous" in result
        assert "warnings" in result
        assert "alpha" in result


# ──────────────────────────────────────────────────────────────
# 5. 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestAssumptionInvariants:
    """가정 검증 수학적 불변량."""

    def test_shapiro_w_in_zero_one(self):
        """W ∈ [0, 1]."""
        result = check_normality(pd.Series(G1))
        assert 0 <= result["statistic"] <= 1

    def test_shapiro_p_in_zero_one(self):
        """p ∈ [0, 1]."""
        result = check_normality(pd.Series(NORMAL_DATA))
        assert 0 <= result["p_value"] <= 1

    def test_levene_f_nonneg(self):
        """F ≥ 0."""
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert result["statistic"] >= 0

    def test_levene_p_in_zero_one(self):
        """p ∈ [0, 1]."""
        result = check_homogeneity_of_variance(G1, G2, center='mean')
        assert 0 <= result["p_value"] <= 1

    def test_homogeneous_consistent_with_alpha(self):
        """homogeneous = (p >= alpha)."""
        alpha = 0.05
        result = check_homogeneity_of_variance(G1, G2, alpha=alpha, center='mean')
        expected = result["p_value"] >= alpha
        assert result["homogeneous"] == expected

    def test_normal_consistent_with_alpha(self):
        """normal = (p >= alpha)."""
        alpha = 0.05
        result = check_normality(pd.Series(NORMAL_DATA), alpha=alpha)
        assert result["normal"] == (result["p_value"] >= alpha)

    def test_nan_removed_before_shapiro(self):
        """NaN 포함 시 자동 제거 후 검정."""
        arr_with_nan = pd.Series(np.concatenate([G1, [np.nan, np.nan]]))
        result = check_normality(arr_with_nan)
        assert result["n"] == len(G1)

    def test_nan_removed_before_levene(self):
        """NaN 포함 시 자동 제거 후 Levene."""
        g1_nan = np.concatenate([G1, [np.nan]])
        result = check_homogeneity_of_variance(g1_nan, G2, center='mean')
        assert not np.isnan(result["statistic"])

    def test_identical_groups_levene_f_zero(self):
        """동일한 두 그룹 → F = 0."""
        result = check_homogeneity_of_variance(G1, G1, center='mean')
        assert result["statistic"] == _approx(0.0, 0.001)

    def test_larger_sample_more_power(self):
        """N이 클수록 소표본 경고가 줄어듦."""
        small = check_normality(pd.Series(G1))   # n=10 → 소표본 경고
        large = check_normality(pd.Series(NORMAL_DATA))  # n=20 → 경고 없음
        small_warns = len(small["warnings"])
        large_warns = len(large["warnings"])
        assert small_warns >= large_warns


# ──────────────────────────────────────────────────────────────
# 6. prepare_analysis_frame 결측치 정책
# ──────────────────────────────────────────────────────────────

class TestPrepareAnalysisFrame:
    """prepare_analysis_frame 결측치 처리 SPSS 호환 검증."""

    def _make_ds(self, include_nan: bool = False) -> Dataset:
        if include_nan:
            arr = np.concatenate([G1, [np.nan, np.nan]])
        else:
            arr = G1.copy()
        df = pd.DataFrame({"x": arr})
        ds = Dataset(df, name="Test")
        ds.variables["x"].measure = MeasureType.SCALE
        return ds

    def test_listwise_excludes_nan(self):
        """LISTWISE: NaN이 있는 행 제거."""
        from statworkbench.core.typing import MissingPolicy
        ds = self._make_ds(include_nan=True)
        frame = prepare_analysis_frame(ds, ["x"], MissingPolicy.LISTWISE)
        assert frame.n_valid == len(G1)
        assert frame.n_excluded == 2

    def test_total_n_matches(self):
        """n_total = n_valid + n_excluded."""
        from statworkbench.core.typing import MissingPolicy
        ds = self._make_ds(include_nan=True)
        frame = prepare_analysis_frame(ds, ["x"], MissingPolicy.LISTWISE)
        assert frame.n_total == frame.n_valid + frame.n_excluded

    def test_excluded_pct_calculation(self):
        """excluded_pct = n_excluded / n_total * 100."""
        from statworkbench.core.typing import MissingPolicy
        ds = self._make_ds(include_nan=True)
        frame = prepare_analysis_frame(ds, ["x"], MissingPolicy.LISTWISE)
        expected_pct = frame.n_excluded / frame.n_total * 100
        assert frame.excluded_pct == _approx(expected_pct, 0.01)

    def test_case_processing_summary_table(self):
        """CPS 테이블 구조 검증."""
        cps = get_case_processing_summary(100, 85, 15)
        df = cps.dataframe
        assert df["Total Cases"].iloc[0] == 100
        assert df["Valid Cases"].iloc[0] == 85
        assert df["Excluded Cases"].iloc[0] == 15
        assert "15.0%" in df["Excluded %"].iloc[0]
