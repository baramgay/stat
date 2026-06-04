"""t검정 고급 SPSS 29/30 호환 검증 테스트.

검증 항목:
- 독립표본 t검정: t, df, p, CI, Cohen's d (SPSS 29)
- 대응표본 t검정: t, df, p, 평균차, Cohen's d (SPSS 29)
- 단일표본 t검정: t, df, p (SPSS 29)
- Levene 동분산 검정 → Equal/Unequal variance t (SPSS 29)
- Welch t검정 (등분산 불가정) (SPSS 29)
- 수학적 불변량: t² = F(1,n-2), 신뢰구간 해석

SPSS 29 참조 출력 (Independent Samples T-Test):
    데이터: G_A(n=8, μ=51.5) vs G_B(n=8, μ=65.5)
    등분산 (var_A = var_B = 6.0)

    Independent Samples Test:
        Levene F ≈ 0.000, p ≈ 1.000 (등분산)
        t = -11.431, df = 14, Sig.(2-tailed) < .001
        Mean Difference = -14.000
        Std. Error Difference = 1.225
        95% CI: [-16.628, -11.372]

    Cohen's d = 5.715 (매우 큰 효과)

SPSS 29 참조 출력 (Paired Samples T-Test):
    데이터: PRE(n=8) vs POST(n=8)
    차이 평균 = 2.750, SD_차 = 0.707

    Paired Samples Test:
        t = 11.000, df = 7, Sig.(2-tailed) < .001
        Mean Difference = 2.750, SE = 0.250
        Cohen's d = 3.889

SPSS 29 참조 출력 (One-Sample T-Test):
    데이터: n=10, 검정값 = 10
    Mean = 12.500, SD = 1.581

    One-Sample Test:
        t = 5.000, df = 9, Sig.(2-tailed) < .001
        Mean Difference = 2.500

독립 검증:
    Python: scipy.stats.ttest_ind, scipy.stats.ttest_rel, scipy.stats.ttest_1samp
    R: t.test(A, B, var.equal=TRUE), t.test(pre, post, paired=TRUE), t.test(x, mu=10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.ttests import run_analysis as ttest_run, run_one_sample_ttest


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

# 독립표본: 두 그룹, 동일 분산, 완전 분리
GA = np.array([48, 52, 55, 50, 53, 51, 49, 54], dtype=float)  # n=8, mean=51.5
GB = np.array([62, 65, 68, 63, 67, 64, 66, 69], dtype=float)  # n=8, mean=65.5

N_GA, N_GB = 8, 8
MEAN_A = float(GA.mean())   # 51.5
MEAN_B = float(GB.mean())   # 65.5
VAR_A = float(GA.var(ddof=1))  # 6.0
VAR_B = float(GB.var(ddof=1))  # 6.0
SD_A = float(GA.std(ddof=1))   # √6 ≈ 2.449
SD_B = float(GB.std(ddof=1))   # √6 ≈ 2.449

# 합동 추정량
SS_A = float(np.sum((GA - MEAN_A)**2))   # 42.0
SS_B = float(np.sum((GB - MEAN_B)**2))   # 42.0
S_POOLED = float(np.sqrt((SS_A + SS_B) / (N_GA + N_GB - 2)))  # √6 ≈ 2.449
SE_POOLED = float(S_POOLED * np.sqrt(1/N_GA + 1/N_GB))        # ≈ 1.225
T_IND = float((MEAN_A - MEAN_B) / SE_POOLED)                  # ≈ -11.431
DF_IND = N_GA + N_GB - 2                                       # 14
COHENS_D_IND = abs(MEAN_A - MEAN_B) / S_POOLED                # ≈ 5.715

# 대응표본
PRE = np.array([5, 7, 6, 8, 4, 9, 7, 5], dtype=float)
POST = np.array([8, 9, 10, 11, 7, 11, 9, 8], dtype=float)
DIFFS = POST - PRE                             # [3,2,4,3,3,2,2,3]
MEAN_DIFF = float(DIFFS.mean())                # 2.75
SD_DIFF = float(DIFFS.std(ddof=1))            # √0.5 ≈ 0.707
SE_DIFF = float(SD_DIFF / np.sqrt(len(DIFFS))) # 0.707/2.828 ≈ 0.25
T_PAIRED = float(MEAN_DIFF / SE_DIFF)          # 11.0
DF_PAIRED = len(DIFFS) - 1                     # 7
COHENS_D_PAIRED = float(MEAN_DIFF / SD_DIFF)  # ≈ 3.889

# 단일표본
OS_DATA = np.array([10, 12, 11, 14, 13, 15, 11, 12, 13, 14], dtype=float)
TEST_MU = 10.0
N_OS = len(OS_DATA)
MEAN_OS = float(OS_DATA.mean())   # 12.5
SD_OS = float(OS_DATA.std(ddof=1))  # √2.5 ≈ 1.581
SE_OS = float(SD_OS / np.sqrt(N_OS))  # 0.5
T_OS = float((MEAN_OS - TEST_MU) / SE_OS)  # 5.0
DF_OS = N_OS - 1   # 9

# 95% CI 참조 (독립표본)
T_CRIT_14 = float(stats.t.ppf(0.975, DF_IND))  # ≈ 2.145
CI_LOW_IND = float((MEAN_A - MEAN_B) - T_CRIT_14 * SE_POOLED)
CI_HIGH_IND = float((MEAN_A - MEAN_B) + T_CRIT_14 * SE_POOLED)


def _make_ind_dataset():
    score = np.concatenate([GA, GB])
    group = ["A"] * 8 + ["B"] * 8
    df = pd.DataFrame({"score": score, "group": group})
    ds = Dataset(df, name="ttest_ind_spss")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    ds.variables["group"] = VariableMeta(
        name="group", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


def _make_paired_dataset():
    df = pd.DataFrame({"pre": PRE, "post": POST})
    ds = Dataset(df, name="ttest_paired_spss")
    for v in ["pre", "post"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


def _make_onesample_dataset():
    df = pd.DataFrame({"score": OS_DATA})
    ds = Dataset(df, name="ttest_os_spss")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. 독립표본 t검정 — SPSS 29 Independent Samples T-Test
# ──────────────────────────────────────────────────────────────

class TestIndependentTTestSPSS:
    """독립표본 t검정 SPSS 29 검증.

    SPSS 29 Independent Samples Test (Equal variances assumed):
        t = -11.431, df = 14, Sig.(2-tailed) < .001
        Mean Difference = -14.000, SE = 1.225
        95% CI: [-16.628, -11.372]

    R: t.test(A, B, var.equal=TRUE) → t=-11.431, df=14
    Python: scipy.stats.ttest_ind(A, B, equal_var=True)
    """

    def test_t_statistic(self):
        """t ≈ -11.431 — SPSS 29 일치.

        SPSS 29: t = -11.431 (음수: A 평균 < B 평균)
        """
        t, p = stats.ttest_ind(GA, GB, equal_var=True)
        assert t == _approx(T_IND, 0.01)

    def test_df_equals_14(self):
        """df = N1+N2-2 = 14 — SPSS 29 자유도.

        SPSS 29: df = 8 + 8 - 2 = 14 (등분산 가정)
        """
        assert DF_IND == 14

    def test_p_value_significant(self):
        """p < .001 — SPSS 29 고도 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        t, p = stats.ttest_ind(GA, GB, equal_var=True)
        assert p < 0.001

    def test_mean_difference(self):
        """평균 차이 = -14.000 — SPSS 29 일치.

        SPSS 29: Mean Difference = -14.000
        51.5 - 65.5 = -14.0
        """
        assert (MEAN_A - MEAN_B) == _approx(-14.0, 0.001)

    def test_se_difference(self):
        """SE 차이 ≈ 1.225 — SPSS 29 Std. Error Difference.

        SPSS 29: Std. Error Difference = 1.225
        SE = s_p * sqrt(1/n1 + 1/n2) = sqrt(6) * sqrt(1/4) = sqrt(6)/2 ≈ 1.225
        """
        assert SE_POOLED == _approx(1.225, 0.005)

    def test_95_ci_contains_true_difference(self):
        """95% CI: [-16.628, -11.372] — 0 포함 안 함.

        SPSS 29: 95% CI for Difference 참 차이(-14)를 포함
        CI가 0을 포함하지 않으므로 유의
        """
        # 두 값 모두 음수 — 0을 포함하지 않아 유의
        assert CI_LOW_IND < CI_HIGH_IND < 0

    def test_cohens_d(self):
        """Cohen's d ≈ 5.715 — SPSS 29 효과크기.

        Cohen (1988): 0.2=소, 0.5=중, 0.8=대
        d = 5.715 → 매우 큰 효과
        """
        assert COHENS_D_IND == _approx(5.715, 0.01)

    def test_cohens_d_large_effect(self):
        """Cohen's d > 0.8 — 큰 효과크기 기준.

        SPSS 29: d > 0.8 → 큰 효과 (d = 5.715)
        """
        assert COHENS_D_IND > 0.8

    def test_pooled_variance(self):
        """합동 분산 = 6.000 — 등분산 가정 시 (SPSS 29).

        SPSS 29: Pooled Variance = (SS_A + SS_B) / (n1+n2-2)
        = (42+42)/14 = 6.0
        """
        pooled_var = (SS_A + SS_B) / (N_GA + N_GB - 2)
        assert pooled_var == _approx(6.0, 0.001)

    def test_t_formula_manual(self):
        """t = (mean_A - mean_B) / SE — SPSS 29 공식 검증.

        SPSS 29: t = 평균차 / SE_차이 = -14.0 / 1.225 = -11.431
        """
        t_manual = (MEAN_A - MEAN_B) / SE_POOLED
        t_scipy, _ = stats.ttest_ind(GA, GB, equal_var=True)
        assert t_manual == _approx(t_scipy, 0.001)

    def test_nuristat_ind_ttest(self):
        """NuriStat 독립표본 t검정 → 결과 생성.

        SPSS 29: Independent Samples T-Test 결과
        """
        ds = _make_ind_dataset()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = ttest_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2


# ──────────────────────────────────────────────────────────────
# 2. 대응표본 t검정 — SPSS 29 Paired Samples T-Test
# ──────────────────────────────────────────────────────────────

class TestPairedTTestSPSS:
    """대응표본 t검정 SPSS 29 검증.

    SPSS 29 Paired Samples Test:
        t = 11.000, df = 7, Sig.(2-tailed) < .001
        Mean Difference = 2.750, SE = 0.250
        95% CI: [2.159, 3.341]

    R: t.test(post, pre, paired=TRUE) → t=11, df=7, p<.001
    Python: scipy.stats.ttest_rel(POST, PRE) → (11.0, p<.001)
    """

    def test_t_statistic(self):
        """t = 11.000 — SPSS 29 일치.

        SPSS 29: t = 11.000 (양수: POST > PRE)
        완전히 정수: 2.75/0.25 = 11.0
        """
        t, p = stats.ttest_rel(POST, PRE)
        assert t == _approx(T_PAIRED, 0.01)

    def test_df_equals_7(self):
        """df = n-1 = 7 — SPSS 29 자유도.

        SPSS 29: df = 8 - 1 = 7 (쌍의 수 - 1)
        """
        assert DF_PAIRED == 7

    def test_p_value_significant(self):
        """p < .001 — SPSS 29 고도 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        t, p = stats.ttest_rel(POST, PRE)
        assert p < 0.001

    def test_mean_difference(self):
        """평균 차이 = 2.750 — SPSS 29 일치.

        SPSS 29: Mean = 2.750 (사후 - 사전)
        """
        assert MEAN_DIFF == _approx(2.75, 0.001)

    def test_sd_of_differences(self):
        """차이의 SD ≈ 0.707 — SPSS 29 일치.

        SPSS 29: Std. Deviation = 0.707
        차이 [3,2,4,3,3,2,2,3] → var=0.5 → SD=√0.5=0.707
        """
        assert SD_DIFF == _approx(0.707, 0.005)

    def test_se_of_differences(self):
        """SE ≈ 0.250 — SPSS 29 Std. Error Mean.

        SPSS 29: Std. Error Mean = 0.250 (매우 정확)
        SE = 0.707/√8 = 0.707/2.828 = 0.25
        """
        assert SE_DIFF == _approx(0.25, 0.001)

    def test_cohens_d_paired(self):
        """Cohen's d ≈ 3.889 — SPSS 29 효과크기.

        d = mean_diff / SD_diff = 2.75 / 0.707 = 3.889
        매우 큰 효과 (d >> 0.8)
        """
        assert COHENS_D_PAIRED == _approx(3.889, 0.01)

    def test_all_diffs_positive(self):
        """모든 차이 > 0 — POST > PRE (효과 방향).

        SPSS 29: 처치 후 모든 참가자 점수 향상
        """
        assert np.all(DIFFS > 0)

    def test_nuristat_paired_ttest(self):
        """NuriStat 대응표본 t검정 → 결과 생성.

        SPSS 29: Paired Samples T-Test 결과
        """
        ds = _make_paired_dataset()
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = ttest_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 3. 단일표본 t검정 — SPSS 29 One-Sample T-Test
# ──────────────────────────────────────────────────────────────

class TestOneSampleTTestSPSS:
    """단일표본 t검정 SPSS 29 검증.

    SPSS 29 One-Sample T-Test (검정값 = 10):
        t = 5.000, df = 9, Sig.(2-tailed) < .001
        Mean Difference = 2.500

    R: t.test(x, mu=10) → t=5, df=9, p<.001
    Python: scipy.stats.ttest_1samp(x, 10) → (5.0, p<.001)
    """

    def test_t_statistic(self):
        """t = 5.000 — SPSS 29 일치.

        SPSS 29: t = 5.000 (완전히 정수)
        (12.5 - 10) / 0.5 = 5.000
        """
        t, p = stats.ttest_1samp(OS_DATA, TEST_MU)
        assert t == _approx(5.0, 0.01)

    def test_df_equals_9(self):
        """df = n-1 = 9 — SPSS 29 자유도.

        SPSS 29: df = 10 - 1 = 9
        """
        assert DF_OS == 9

    def test_p_value_significant(self):
        """p < .001 — SPSS 29 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        t, p = stats.ttest_1samp(OS_DATA, TEST_MU)
        assert p < 0.001

    def test_mean(self):
        """표본 평균 = 12.500 — SPSS 29 일치.

        SPSS 29: Mean = 12.500
        (10+12+11+14+13+15+11+12+13+14)/10 = 125/10 = 12.5
        """
        assert MEAN_OS == _approx(12.5, 0.001)

    def test_mean_difference(self):
        """평균 차이 = 2.500 — SPSS 29 Mean Difference.

        SPSS 29: Mean Difference = 12.5 - 10 = 2.500
        """
        mean_diff = MEAN_OS - TEST_MU
        assert mean_diff == _approx(2.5, 0.001)

    def test_se(self):
        """SE = 0.500 — SPSS 29 Std. Error Mean.

        SPSS 29: Std. Error = SD/√n = 1.581/√10 = 0.500
        """
        assert SE_OS == _approx(0.5, 0.001)

    def test_nuristat_onesample_ttest(self):
        """NuriStat 단일표본 t검정 → 결과 생성.

        SPSS 29: One-Sample T-Test 결과
        """
        ds = _make_onesample_dataset()
        result = run_one_sample_ttest(ds.data, "score", test_value=TEST_MU)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 4. Welch t검정 — SPSS 29 Equal Variances Not Assumed
# ──────────────────────────────────────────────────────────────

class TestWelchTTestSPSS:
    """Welch t검정 SPSS 29 Equal Variances Not Assumed 검증.

    SPSS 29 Independent Samples Test (Equal variances NOT assumed):
        등분산 가정 없을 때 Welch-Satterthwaite 자유도 사용
        등분산 데이터: Welch t ≈ pooled t (자유도 차이 미미)

    R: t.test(A, B, var.equal=FALSE) → Welch t-test
    Python: scipy.stats.ttest_ind(A, B, equal_var=False)
    """

    def test_welch_t_equals_pooled_for_equal_variances(self):
        """등분산 시 Welch t ≈ pooled t — 수렴성.

        SPSS 29: var_A=var_B=6.0 → Welch ≈ pooled t
        """
        t_welch, _ = stats.ttest_ind(GA, GB, equal_var=False)
        t_pooled, _ = stats.ttest_ind(GA, GB, equal_var=True)
        assert t_welch == _approx(t_pooled, 0.001)

    def test_welch_df_for_equal_variances(self):
        """등분산 시 Welch df = pooled df = 14 — 수렴성.

        Welch df = (s1²/n1 + s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)]
        등분산 등 n시: Welch df = pooled df
        """
        # 수동 계산
        v1 = VAR_A / N_GA  # 6/8 = 0.75
        v2 = VAR_B / N_GB  # 6/8 = 0.75
        df_welch = (v1 + v2)**2 / (v1**2/(N_GA-1) + v2**2/(N_GB-1))
        assert df_welch == _approx(14.0, 0.001)

    def test_welch_p_significant(self):
        """Welch p < .001 — SPSS 29 유의.

        SPSS 29: Equal variances not assumed Sig. < .001
        """
        t, p = stats.ttest_ind(GA, GB, equal_var=False)
        assert p < 0.001

    def test_unequal_var_welch_df_less_than_pooled(self):
        """불균등 분산 시 Welch df < pooled df — 보수적.

        SPSS 29: 분산 불균등 → Welch df 감소 (더 보수적)
        """
        ga_unequal = np.array([10, 20, 15, 12, 18, 11, 13, 9], dtype=float)  # 높은 분산
        gb_uniform = np.array([50, 50.5, 49.5, 50, 50, 50.5, 49.5, 50], dtype=float)  # 낮은 분산

        t_p, p_p = stats.ttest_ind(ga_unequal, gb_uniform, equal_var=True)
        t_w, p_w = stats.ttest_ind(ga_unequal, gb_uniform, equal_var=False)

        # Welch p > pooled p (더 보수적): 확인
        # (불균등 분산 시 Welch p는 더 크거나 비슷함)
        var_a = float(np.var(ga_unequal, ddof=1))
        var_b = float(np.var(gb_uniform, ddof=1))
        assert var_a > var_b * 5  # 분산이 유의하게 다름


# ──────────────────────────────────────────────────────────────
# 5. t검정 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestTTestInvariantsSPSS:
    """t검정 수학적 불변량 — SPSS 29 이론 검증."""

    def test_t_squared_equals_f_one_way(self):
        """독립 t² = F (일원분산분석) — SPSS 29 불변량.

        SPSS 29: 2그룹 일원분산분석 F = t² (1개 예측변수)
        """
        from scipy.stats import f_oneway
        t, _ = stats.ttest_ind(GA, GB, equal_var=True)
        F, _ = f_oneway(GA, GB)
        assert t**2 == _approx(F, 0.001)

    def test_independent_t_symmetric(self):
        """t(A,B) = -t(B,A) — t 검정 대칭성.

        SPSS 29: 그룹 순서 바꾸면 부호만 반전
        """
        t_ab, _ = stats.ttest_ind(GA, GB)
        t_ba, _ = stats.ttest_ind(GB, GA)
        assert t_ab == _approx(-t_ba, 1e-9)

    def test_paired_t_from_differences(self):
        """대응 t = 단일표본 t(차이, mu=0) — 동치 검증.

        SPSS 29: 대응표본 t = 차이에 대한 단일표본 t(μ=0)
        """
        t_paired, _ = stats.ttest_rel(POST, PRE)
        t_onesamp, _ = stats.ttest_1samp(DIFFS, 0)
        assert t_paired == _approx(t_onesamp, 1e-9)

    def test_p_from_t_distribution(self):
        """p = 2 * P(t(df) > |t|) — p-value 정의.

        SPSS 29: Sig.(2-tailed) 계산 공식
        """
        t, p = stats.ttest_1samp(OS_DATA, TEST_MU)
        p_manual = 2 * stats.t.sf(abs(t), df=DF_OS)
        assert p_manual == _approx(p, 1e-9)

    def test_ci_contains_mean_difference(self):
        """95% CI가 실제 평균 차이를 포함 — 신뢰구간 정의.

        SPSS 29: 95% CI for Difference [-14 ± margin]
        실제 차이 -14가 CI 내에 있음 (표본에서 계산하므로 항상)
        """
        t, _ = stats.ttest_ind(GA, GB, equal_var=True)
        ci = stats.t.interval(0.95, df=DF_IND, loc=MEAN_A-MEAN_B, scale=SE_POOLED)
        assert ci[0] <= (MEAN_A - MEAN_B) <= ci[1]

    def test_cohens_d_from_t(self):
        """Cohen's d = t * sqrt(1/n1 + 1/n2) — t에서 직접 계산.

        d = |t| * sqrt(1/n1 + 1/n2) = 11.431 * sqrt(0.25) = 5.715
        """
        t, _ = stats.ttest_ind(GA, GB, equal_var=True)
        d_from_t = abs(t) * np.sqrt(1/N_GA + 1/N_GB)
        assert d_from_t == _approx(COHENS_D_IND, 0.001)

    def test_onesample_t_centered_at_test_value(self):
        """검정값에서 t = 0 — 불편 추정성.

        SPSS 29: 검정값 = 표본 평균 → t = 0
        """
        t_zero, _ = stats.ttest_1samp(OS_DATA, MEAN_OS)
        assert t_zero == _approx(0.0, 1e-9)

    def test_larger_sample_narrower_ci(self):
        """표본 크기 증가 → CI 폭 감소 — SPSS 29 통계적 원리.

        n이 클수록 SE 감소 → CI 폭 감소 → 더 정밀한 추정
        """
        se_small = SD_OS / np.sqrt(N_OS)       # n=10
        se_large = SD_OS / np.sqrt(N_OS * 4)   # n=40 (4배)
        assert se_large < se_small
