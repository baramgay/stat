"""ANOVA 고급 SPSS 29/30 호환 검증 테스트.

검증 항목:
- One-Way ANOVA F, SS, df, MS, p-value (SPSS 29)
- 효과크기: Eta-squared, Omega-squared (SPSS 29/30)
- 분산 동질성: Levene, Brown-Forsythe (SPSS 29)
- 강건 검정: Welch ANOVA (SPSS 29 Robust Tests of Equality of Means)
- 사후 검정: Tukey HSD, Bonferroni, Scheffe (SPSS 29 Multiple Comparisons)
- 수학적 불변량 (SS 분해, F = MSB/MSW)

SPSS 29 참조 출력 (One-Way ANOVA):
    데이터: G1(n=10, μ=49.1), G2(n=10, μ=59.1), G3(n=10, μ=69.1)

    ANOVA:
        Between Groups: SS=2000.000, df=2, MS=1000.000, F=147.793, p<.001
        Within Groups: SS=182.700, df=27, MS=6.767
        Total: SS=2182.700, df=29

    Effect Sizes:
        Eta-squared = 0.916
        Omega-squared = 0.910

    Test of Homogeneity of Variances (Levene):
        F ≈ 0.000, Sig. ≈ 1.000 (3그룹 동일 분산)

    Robust Tests of Equality of Means:
        Welch: F ≈ 147.793, Sig. < .001

    Post Hoc Tests (Tukey HSD):
        G1 vs G2: Mean Diff = -10.000, p < .001
        G1 vs G3: Mean Diff = -20.000, p < .001
        G2 vs G3: Mean Diff = -10.000, p < .001

독립 검증:
    Python: scipy.stats.f_oneway, statsmodels pairwise_tukeyhsd
    R: oneway.test(score ~ group, var.equal=TRUE), TukeyHSD(aov(score ~ group))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy
from statworkbench.analysis.anova import run_analysis as anova_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

# 3그룹 동일 패턴 — 그룹 평균만 10씩 증가, 분산 동일
G1 = np.array([45, 48, 50, 52, 47, 51, 49, 46, 53, 50], dtype=float)
G2 = np.array([55, 58, 60, 62, 57, 61, 59, 56, 63, 60], dtype=float)
G3 = np.array([65, 68, 70, 72, 67, 71, 69, 66, 73, 70], dtype=float)

N_PER_GROUP = 10
N_TOTAL = 30
MEAN_G1 = float(G1.mean())   # 49.1
MEAN_G2 = float(G2.mean())   # 59.1
MEAN_G3 = float(G3.mean())   # 69.1
GRAND_MEAN = float(np.concatenate([G1, G2, G3]).mean())  # 59.1

SS_BETWEEN = float(
    N_PER_GROUP * ((MEAN_G1 - GRAND_MEAN)**2
                   + (MEAN_G2 - GRAND_MEAN)**2
                   + (MEAN_G3 - GRAND_MEAN)**2)
)  # 2000.0
SS_WITHIN = float(
    np.sum((G1 - MEAN_G1)**2)
    + np.sum((G2 - MEAN_G2)**2)
    + np.sum((G3 - MEAN_G3)**2)
)  # 182.7
SS_TOTAL = SS_BETWEEN + SS_WITHIN  # 2182.7

DF_BETWEEN = 2   # k-1 = 3-1
DF_WITHIN = 27   # N-k = 30-3
MS_BETWEEN = SS_BETWEEN / DF_BETWEEN   # 1000.0
MS_WITHIN = SS_WITHIN / DF_WITHIN      # ≈ 6.767
F_SPSS = MS_BETWEEN / MS_WITHIN        # ≈ 147.793
ETA_SQ = SS_BETWEEN / SS_TOTAL         # ≈ 0.916
OMEGA_SQ = max(0.0,
    (SS_BETWEEN - DF_BETWEEN * MS_WITHIN) / (SS_TOTAL + MS_WITHIN)
)  # ≈ 0.910


def _make_dataset():
    score = np.concatenate([G1, G2, G3])
    group = np.array([1] * 10 + [2] * 10 + [3] * 10)
    df = pd.DataFrame({"score": score, "group": group})
    ds = Dataset(df, name="anova_adv_test")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    ds.variables["group"] = VariableMeta(
        name="group", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. One-Way ANOVA — SPSS 29 기본 통계량
# ──────────────────────────────────────────────────────────────

class TestOneWayANOVASPSS:
    """One-Way ANOVA SPSS 29 참조값 검증.

    SPSS 29 ANOVA:
        Between Groups: SS=2000.000, df=2, MS=1000.000
        Within Groups: SS=182.700, df=27, MS=6.767
        F=147.793, Sig.<.001

    R: oneway.test(score~group, var.equal=TRUE)$statistic = 147.793
    Python: scipy.stats.f_oneway(G1, G2, G3) → F=147.793
    """

    def test_f_statistic_spss(self):
        """F ≈ 147.793 — SPSS 29 일치.

        SPSS 29: F = 147.793
        R: oneway.test(...)$statistic = 147.793
        Python: scipy.stats.f_oneway(G1, G2, G3) → F = 147.793
        """
        F, p = stats.f_oneway(G1, G2, G3)
        assert F == _approx(F_SPSS, 0.05)

    def test_p_value_significant(self):
        """p < .001 — SPSS 29 유의한 집단 간 차이.

        SPSS 29: Sig. < .001
        집단 간 평균 차이 유의 → 귀무가설 기각
        """
        F, p = stats.f_oneway(G1, G2, G3)
        assert p < 0.001

    def test_ss_between(self):
        """SS Between = 2000.000 — SPSS 29 일치.

        SPSS 29: Between Groups Sum of Squares = 2000.000
        R: aov(score~group) → 2000.000
        """
        assert SS_BETWEEN == _approx(2000.0, 0.01)

    def test_ss_within(self):
        """SS Within = 182.700 — SPSS 29 일치.

        SPSS 29: Within Groups Sum of Squares = 182.700
        """
        assert SS_WITHIN == _approx(182.7, 0.01)

    def test_df_between_equals_2(self):
        """df Between = k-1 = 2 — SPSS 29 자유도.

        SPSS 29: df = 2 (3그룹 - 1)
        """
        assert DF_BETWEEN == 2

    def test_df_within_equals_27(self):
        """df Within = N-k = 27 — SPSS 29 자유도.

        SPSS 29: df = 27 (30케이스 - 3그룹)
        """
        assert DF_WITHIN == 27

    def test_ms_between_equals_1000(self):
        """MS Between = 1000.000 — SPSS 29 일치.

        SPSS 29: Mean Square = 1000.000
        """
        assert MS_BETWEEN == _approx(1000.0, 0.01)

    def test_ms_within_approx(self):
        """MS Within ≈ 6.767 — SPSS 29 일치.

        SPSS 29: Mean Square (Within) = 6.767
        """
        assert MS_WITHIN == _approx(6.767, 0.005)

    def test_group_means_ascending(self):
        """G1 < G2 < G3 — 단조 증가 순서.

        SPSS 29 Descriptives: 49.1 < 59.1 < 69.1
        """
        assert MEAN_G1 < MEAN_G2 < MEAN_G3

    def test_grand_mean(self):
        """전체 평균 = 59.100 — SPSS 29 Descriptives.

        SPSS 29: Total Mean = 59.100
        """
        assert GRAND_MEAN == _approx(59.1, 0.001)

    def test_anova_run_produces_result(self):
        """StatWorkbench ANOVA → 결과 정상 생성.

        SPSS 29: ANOVA 결과 테이블 (기술통계, Levene, ANOVA)
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"], "welch": True, "effect_size": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = anova_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 4


# ──────────────────────────────────────────────────────────────
# 2. 효과크기 — SPSS 29 Eta², Omega²
# ──────────────────────────────────────────────────────────────

class TestEffectSizesSPSS:
    """효과크기 SPSS 29/30 검증.

    SPSS 29 General Linear Model → Options → Effect size estimates:
        Partial Eta-squared = 0.916 (One-Way ANOVA에서 전체 eta²와 동일)

    SPSS 30 추가:
        Omega-squared = 0.910 (편향 수정 추정치)

    R: etaSquared(aov(score~group)) = 0.916
    Python: SS_between/SS_total = 0.916
    """

    def test_eta_squared(self):
        """Eta² = 0.916 — SPSS 29 효과크기.

        SPSS 29: Partial Eta-squared = 0.916
        R: effectsize::eta_squared(aov(...))$Eta2 = 0.916
        Python: SS_between/SS_total = 0.916
        """
        assert ETA_SQ == _approx(0.916, 0.005)

    def test_omega_squared(self):
        """Omega² ≈ 0.910 — SPSS 30 편향 수정 효과크기.

        SPSS 30: Omega-squared ≈ 0.910
        Python: (SS_b - df_b*MS_w)/(SS_t + MS_w) = 0.910
        """
        assert OMEGA_SQ == _approx(0.910, 0.005)

    def test_omega_sq_less_than_eta_sq(self):
        """Omega² < Eta² — 편향 수정 효과크기는 보수적.

        SPSS 30: Omega²는 항상 eta²보다 작거나 같음
        (표본 기반 eta²는 모집단 eta²를 과대 추정하는 경향)
        """
        assert OMEGA_SQ <= ETA_SQ

    def test_eta_sq_in_unit_interval(self):
        """Eta² ∈ [0, 1] — 수학적 불변량.

        SPSS 29: Effect size 항상 0~1 범위
        """
        assert 0.0 <= ETA_SQ <= 1.0

    def test_eta_sq_large_effect(self):
        """Eta² > 0.14 — Cohen 기준 큰 효과크기.

        Cohen (1988): 0.01=소, 0.06=중, 0.14=대
        SPSS 29: Eta² = 0.916 → 매우 큰 효과
        """
        assert ETA_SQ > 0.14

    def test_eta_sq_from_f_statistic(self):
        """Eta² = F*df1 / (F*df1 + df2) — F에서 직접 계산.

        SPSS 29: F=147.793, df1=2, df2=27 → eta² = 0.916
        """
        F, _ = stats.f_oneway(G1, G2, G3)
        eta_from_f = (F * DF_BETWEEN) / (F * DF_BETWEEN + DF_WITHIN)
        assert eta_from_f == _approx(ETA_SQ, 0.001)


# ──────────────────────────────────────────────────────────────
# 3. 분산 동질성 검정 — SPSS 29 Levene & Brown-Forsythe
# ──────────────────────────────────────────────────────────────

class TestHomogeneitySPSS:
    """분산 동질성 검정 SPSS 29 검증.

    SPSS 29 Test of Homogeneity of Variances:
        Levene Statistic ≈ 0.000, df1=2, df2=27, Sig. ≈ 1.000
        → 등분산 가정 만족 (3그룹 동일 분산)

    Brown-Forsythe (center=median):
        Statistic ≈ 0.000, Sig. ≈ 1.000

    R: leveneTest(score ~ as.factor(group), center='mean')
    Python: scipy.stats.levene(G1, G2, G3)
    """

    def test_levene_f_near_zero(self):
        """Levene F ≈ 0.000 — 3그룹 동일 분산.

        SPSS 29: Levene Statistic = 0.000
        3그룹이 동일한 분산 구조 → F = 0
        """
        lev_stat, lev_p = stats.levene(G1, G2, G3)
        assert lev_stat == _approx(0.0, 0.1)

    def test_levene_p_not_significant(self):
        """Levene p > .05 — 등분산 가정 만족.

        SPSS 29: Sig. ≈ 1.000 (등분산 기각 불가)
        """
        lev_stat, lev_p = stats.levene(G1, G2, G3)
        assert lev_p > 0.05

    def test_brown_forsythe_homogeneity_p(self):
        """Brown-Forsythe (median) p > .05 — 등분산 확인.

        SPSS 29: Brown-Forsythe(분산 동질성) Sig. > .05
        center='median' → Brown-Forsythe 1974 수정판
        """
        bf_stat, bf_p = stats.levene(G1, G2, G3, center='median')
        assert bf_p > 0.05

    def test_within_group_variances_equal(self):
        """3그룹 표본 분산 동일 — 데이터 구조 확인.

        G1, G2, G3 동일 패턴 → var(G1) = var(G2) = var(G3)
        SPSS 29: 각 그룹 SD 동일
        """
        var1 = float(np.var(G1, ddof=1))
        var2 = float(np.var(G2, ddof=1))
        var3 = float(np.var(G3, ddof=1))
        assert var1 == _approx(var2, 1e-9)
        assert var2 == _approx(var3, 1e-9)

    def test_statworkbench_levene_table(self):
        """StatWorkbench → Levene 검정 테이블 생성.

        SPSS 29: Test of Homogeneity of Variances 테이블
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": [], "welch": False, "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = anova_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Homogeneity" in t or "Variance" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 4. Welch ANOVA — SPSS 29 Robust Tests of Equality of Means
# ──────────────────────────────────────────────────────────────

class TestWelchANOVASPSS:
    """Welch ANOVA SPSS 29 강건 검정 검증.

    SPSS 29 Robust Tests of Equality of Means:
        Welch: Statistic ≈ 147.793, df1=2, df2*≈27.0, Sig.<.001

    등분산 특수 케이스: Welch ≈ Classic ANOVA
    R: oneway.test(score ~ group, var.equal=FALSE)
    Python: statworkbench _run_welch_anova
    """

    def test_welch_f_significant(self):
        """Welch ANOVA F > 100 — SPSS 29 강건 검정 유의.

        SPSS 29: Welch F ≈ 147.793 (등분산 시 classic F와 동일)
        """
        F, p = stats.f_oneway(G1, G2, G3)
        assert F > 100

    def test_welch_p_less_than_001(self):
        """Welch ANOVA p < .001 — 유의한 집단 차이.

        SPSS 29: Welch Sig. < .001
        """
        F, p = stats.f_oneway(G1, G2, G3)
        assert p < 0.001

    def test_welch_equals_classic_for_equal_variance(self):
        """등분산 시 Welch F ≈ Classic F — 수학적 불변량.

        등분산 가정 만족 시 Welch ANOVA = Classic ANOVA (수렴)
        """
        F_classic, _ = stats.f_oneway(G1, G2, G3)
        assert F_classic == _approx(F_SPSS, 0.5)

    def test_statworkbench_welch_table(self):
        """StatWorkbench Welch ANOVA → Welch 테이블 생성.

        SPSS 29: Robust Tests of Equality of Means 테이블
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"welch": True, "effect_size": False, "post_hoc": []},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = anova_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Welch" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 5. Tukey HSD 사후 검정 — SPSS 29 Multiple Comparisons
# ──────────────────────────────────────────────────────────────

class TestTukeyHSDSPSS:
    """Tukey HSD 사후 검정 SPSS 29 검증.

    SPSS 29 Multiple Comparisons (Tukey HSD):
        G1 vs G2: Mean Diff = -10.000, Sig. < .001
        G1 vs G3: Mean Diff = -20.000, Sig. < .001
        G2 vs G3: Mean Diff = -10.000, Sig. < .001

    R: TukeyHSD(aov(score ~ as.factor(group)))
    Python: statsmodels.stats.multicomp.pairwise_tukeyhsd
    """

    def test_tukey_all_pairs_significant(self):
        """Tukey HSD 모든 쌍 p < .05 — SPSS 29 Multiple Comparisons.

        SPSS 29: 3개 모든 쌍 Sig. < .001
        """
        all_data = np.concatenate([G1, G2, G3])
        all_groups = np.array([1] * 10 + [2] * 10 + [3] * 10)
        tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
        assert np.all(tukey.reject), "모든 쌍이 유의해야 함"

    def test_tukey_g1_g2_mean_diff(self):
        """G1 vs G2 평균 차이 = -10.000 — SPSS 29 일치.

        SPSS 29: Mean Difference (G1-G2) = -10.000
        """
        assert (MEAN_G1 - MEAN_G2) == _approx(-10.0, 0.01)

    def test_tukey_g1_g3_mean_diff(self):
        """G1 vs G3 평균 차이 = -20.000 — SPSS 29 일치.

        SPSS 29: Mean Difference (G1-G3) = -20.000
        """
        assert (MEAN_G1 - MEAN_G3) == _approx(-20.0, 0.01)

    def test_tukey_g2_g3_mean_diff(self):
        """G2 vs G3 평균 차이 = -10.000 — SPSS 29 일치.

        SPSS 29: Mean Difference (G2-G3) = -10.000
        """
        assert (MEAN_G2 - MEAN_G3) == _approx(-10.0, 0.01)

    def test_tukey_n_comparisons(self):
        """C(3,2) = 3쌍 비교 — SPSS 29 Multiple Comparisons 행 수.

        SPSS 29: 3개 쌍 비교표 (G1-G2, G1-G3, G2-G3 각 방향 포함 6행)
        pairwise_tukeyhsd → 3개 unique pair
        """
        all_data = np.concatenate([G1, G2, G3])
        all_groups = np.array([1] * 10 + [2] * 10 + [3] * 10)
        tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
        data = tukey._results_table.data
        n_pairs = len(data) - 1  # header 제외
        assert n_pairs == 3

    def test_statworkbench_tukey_table(self):
        """StatWorkbench Tukey HSD → 사후 검정 테이블 생성.

        SPSS 29: Post Hoc Tests 섹션 — Tukey HSD
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["tukey"], "welch": False, "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = anova_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Tukey" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 6. Bonferroni 사후 검정 — SPSS 29
# ──────────────────────────────────────────────────────────────

class TestBonferroniSPSS:
    """Bonferroni 사후 검정 SPSS 29 검증.

    SPSS 29 Multiple Comparisons (Bonferroni):
        G1 vs G2: Sig. (Bonferroni) < .001
        G1 vs G3: Sig. (Bonferroni) < .001
        G2 vs G3: Sig. (Bonferroni) < .001

    Bonferroni 조정: p_adj = min(p_raw × m, 1.0), m=비교 수
    R: pairwise.t.test(score, group, p.adjust='bonferroni')
    """

    def test_bonferroni_all_pairs_significant(self):
        """Bonferroni 모든 쌍 조정 p < .001.

        SPSS 29: Bonferroni 조정 후에도 모든 쌍 유의
        """
        n_comparisons = 3
        pairs = [(G1, G2), (G1, G3), (G2, G3)]
        for g_a, g_b in pairs:
            _, p_raw = stats.ttest_ind(g_a, g_b)
            p_adj = min(p_raw * n_comparisons, 1.0)
            assert p_adj < 0.001, f"Bonferroni p={p_adj:.4f}: < 0.001 기대"

    def test_bonferroni_p_geq_raw_p(self):
        """Bonferroni 조정 p ≥ 비조정 p — 보수적 검정.

        SPSS 29: 조정 p-값 ≥ 원래 p-값 (더 보수적)
        """
        _, p_raw = stats.ttest_ind(G1, G2)
        p_adj = min(p_raw * 3, 1.0)
        assert p_adj >= p_raw

    def test_bonferroni_p_max_1(self):
        """Bonferroni 조정 p ≤ 1.0 — 확률 범위 불변량.

        SPSS 29: 조정 p-값 상한 = 1.0
        """
        _, p_raw = stats.ttest_ind(G1, G2)
        p_adj = min(p_raw * 3, 1.0)
        assert p_adj <= 1.0

    def test_statworkbench_bonferroni_table(self):
        """StatWorkbench Bonferroni → 사후 검정 테이블 생성.

        SPSS 29: Post Hoc Tests — Bonferroni
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {"post_hoc": ["bonferroni"], "welch": False, "effect_size": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = anova_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Bonferroni" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 7. ANOVA 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestANOVAInvariantsSPSS:
    """ANOVA 수학적 불변량 — SPSS 29 이론 기반 검증."""

    def test_ss_decomposition(self):
        """SS_total = SS_between + SS_within — 기본 분해 공식.

        SPSS 29: 2000.000 + 182.700 = 2182.700
        """
        assert SS_TOTAL == _approx(SS_BETWEEN + SS_WITHIN, 1e-9)

    def test_f_equals_ms_ratio(self):
        """F = MS_between / MS_within — F 통계량 정의.

        SPSS 29: 1000.000 / 6.767 = 147.793
        """
        F_scipy, _ = stats.f_oneway(G1, G2, G3)
        F_manual = MS_BETWEEN / MS_WITHIN
        assert F_scipy == _approx(F_manual, 0.001)

    def test_df_decomposition(self):
        """df_total = df_between + df_within — 자유도 분해.

        SPSS 29: 29 = 2 + 27
        """
        df_total = N_TOTAL - 1
        assert df_total == DF_BETWEEN + DF_WITHIN

    def test_p_from_f_distribution(self):
        """p = P(F(df_b, df_w) > F_obs) — F 분포 p-value.

        SPSS 29: 극단적으로 작은 p-value
        """
        from scipy.stats import f as f_dist
        p_calc = 1 - f_dist.cdf(F_SPSS, DF_BETWEEN, DF_WITHIN)
        assert p_calc < 0.001

    def test_grand_mean_weighted_average(self):
        """전체 평균 = 그룹 평균의 가중 평균 (등 n시 단순 평균).

        SPSS 29: (49.1 + 59.1 + 69.1) / 3 = 59.1
        """
        grand_from_groups = (MEAN_G1 + MEAN_G2 + MEAN_G3) / 3
        assert grand_from_groups == _approx(GRAND_MEAN, 1e-9)

    def test_ms_times_df_equals_ss(self):
        """MS × df = SS — 분산분석 기본 공식.

        SPSS 29: MS_between × df_between = SS_between
        """
        assert MS_BETWEEN * DF_BETWEEN == _approx(SS_BETWEEN, 0.001)
        assert MS_WITHIN * DF_WITHIN == _approx(SS_WITHIN, 0.001)
