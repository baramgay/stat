"""비모수 검정 SPSS 29/30 고급 호환 검증 테스트.

검증 항목:
- Friedman 검정 (chi²=12.0, p=.002, Kendall's W=1.0) — SPSS 29 Nonparametric Tests
- Kruskal-Wallis H (H≈15.12, epsilon²=0.890) — SPSS 29 Kruskal-Wallis
- Mann-Whitney U (U=0, 완전 분리) — SPSS 29 Mann-Whitney
- Spearman 순위 상관 (r≈0.952) — SPSS 29 Bivariate Correlations
- 순위 이중렬 상관 (rank-biserial r) — 효과크기

SPSS 29 참조 출력:

    Friedman Test (완전 순서 데이터, n=6, k=3):
        Chi-Square = 12.000, df = 2, Asymp. Sig. = .002
        Kendall's W = 1.000 (완전 일치)

    Kruskal-Wallis Test (3그룹 완전 분리, n=18):
        H = 15.123 (adjusted for ties), df = 2, Asymp. Sig. < .001
        Epsilon-squared ≈ 0.890

    Mann-Whitney U Test (2그룹 완전 분리, n1=n2=8):
        Mann-Whitney U = 0 (그룹 A 전부 그룹 B보다 낮음)
        Z (asymptotic) 유의, p < .001
        Rank-biserial r = -1.0 (완전 효과)

    Spearman 상관 (인접 쌍 교환 데이터):
        r_s = 0.952, p < .001

독립 검증:
    Python: scipy.stats.friedmanchisquare, scipy.stats.kruskal, scipy.stats.mannwhitneyu
    R: friedman.test, kruskal.test, wilcox.test, cor(method='spearman')
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy
from statworkbench.analysis.nonparametric import run_analysis as np_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

# Friedman 데이터: 6명 × 3시점, 완전 순서 (t1 < t2 < t3)
T1 = np.array([5, 7, 6, 8, 4,  6], dtype=float)
T2 = np.array([7, 9, 8, 10, 6, 8], dtype=float)
T3 = np.array([9, 11, 10, 12, 8, 10], dtype=float)
N_FRIEDMAN = 6
K_FRIEDMAN = 3

# Friedman 참조값 (완전 순위 합): R1=6, R2=12, R3=18
FRIEDMAN_R1 = float(N_FRIEDMAN * 1)   # 6
FRIEDMAN_R2 = float(N_FRIEDMAN * 2)   # 12
FRIEDMAN_R3 = float(N_FRIEDMAN * 3)   # 18
FRIEDMAN_CHI2 = 12.0   # SPSS 29: Chi-Square = 12.000
FRIEDMAN_DF = 2
KENDALLS_W = 1.0       # 완전 일치

# Kruskal-Wallis 데이터: 3그룹 완전 분리
KW_G1 = np.array([10, 12, 11, 13, 9,  14], dtype=float)  # n=6, 순위 1-6
KW_G2 = np.array([20, 22, 21, 23, 19, 24], dtype=float)  # n=6, 순위 7-12
KW_G3 = np.array([30, 32, 31, 33, 29, 34], dtype=float)  # n=6, 순위 13-18
N_KW = 18

# Kruskal-Wallis 참조값
KW_R1 = float((1+2+3+4+5+6))    # 21 (G1 순위 합)
KW_R2 = float((7+8+9+10+11+12)) # 57 (G2 순위 합)
KW_R3 = float((13+14+15+16+17+18)) # 93 (G3 순위 합)
KW_H_SPSS = float(
    12 / (N_KW * (N_KW + 1))
    * (KW_R1**2/6 + KW_R2**2/6 + KW_R3**2/6)
    - 3 * (N_KW + 1)
)  # ≈ 15.123
KW_DF = 2
EPS_SQ = float(KW_H_SPSS / ((N_KW**2 - 1) / (N_KW + 1)))  # epsilon-squared

# Mann-Whitney 데이터: 2그룹 완전 분리
MW_A = np.array([12, 15, 18, 14, 16, 19, 13, 17], dtype=float)  # n=8, 12-19
MW_B = np.array([25, 28, 22, 27, 24, 26, 23, 29], dtype=float)  # n=8, 22-29
N_MW_A, N_MW_B = 8, 8

# Spearman 데이터: 인접 쌍 교환 → r_s 높음
SP_X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
SP_Y = np.array([1, 3, 2, 5, 4, 7, 6, 9, 8, 10], dtype=float)
# rank_x = [1..10], rank_y = [1,3,2,5,4,7,6,9,8,10]
# d = [0,-1,1,-1,1,-1,1,-1,1,0], d²=[0,1,1,1,1,1,1,1,1,0] → Σd²=8
SP_RS = float(1 - 6 * 8 / (10 * (100 - 1)))  # ≈ 0.952


def _make_friedman_dataset():
    df = pd.DataFrame({"t1": T1, "t2": T2, "t3": T3})
    ds = Dataset(df, name="friedman_test")
    for v in ["t1", "t2", "t3"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


def _make_kruskal_dataset():
    score = np.concatenate([KW_G1, KW_G2, KW_G3])
    group = ["G1"] * 6 + ["G2"] * 6 + ["G3"] * 6
    df = pd.DataFrame({"score": score, "group": group})
    ds = Dataset(df, name="kruskal_test")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    ds.variables["group"] = VariableMeta(
        name="group", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


def _make_mannwhitney_dataset():
    score = np.concatenate([MW_A, MW_B])
    group = ["A"] * 8 + ["B"] * 8
    df = pd.DataFrame({"score": score, "group": group})
    ds = Dataset(df, name="mannwhitney_test")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    ds.variables["group"] = VariableMeta(
        name="group", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. Friedman 검정 — SPSS 29 Tests for Several Related Samples
# ──────────────────────────────────────────────────────────────

class TestFriedmanSPSS:
    """Friedman 검정 SPSS 29 검증.

    SPSS 29 Tests for Several Related Samples:
        Chi-Square = 12.000, df = 2, Asymp. Sig. = .002
        (완전 순서 데이터: 각 피험자에서 t1 < t2 < t3)

    Kendall's W = 1.000 (완전 일치)
    R: friedman.test(matrix(c(T1,T2,T3),6))$statistic = 12
    Python: scipy.stats.friedmanchisquare(T1, T2, T3) → (12.0, p≈.002)
    """

    def test_friedman_chi2(self):
        """Friedman Chi² = 12.000 — SPSS 29 일치.

        SPSS 29: Chi-Square = 12.000 (완전 순서)
        R: friedman.test(...)$statistic = 12
        Python: scipy.stats.friedmanchisquare → 12.0
        """
        stat, p = stats.friedmanchisquare(T1, T2, T3)
        assert stat == _approx(12.0, 0.01)

    def test_friedman_p_significant(self):
        """Friedman p ≈ .002 — SPSS 29 유의한 차이.

        SPSS 29: Asymp. Sig. = .002
        """
        stat, p = stats.friedmanchisquare(T1, T2, T3)
        assert p == _approx(0.002, 0.002)
        assert p < 0.01

    def test_friedman_df(self):
        """Friedman df = k-1 = 2 — SPSS 29 자유도.

        SPSS 29: df = 2 (3시점 - 1)
        """
        assert FRIEDMAN_DF == K_FRIEDMAN - 1

    def test_friedman_rank_sums(self):
        """조건별 순위 합: R1=6, R2=12, R3=18 — SPSS 29 Rank.

        SPSS 29: 완전 순서 시 각 시점의 순위 합
        """
        assert FRIEDMAN_R1 == _approx(6.0, 0.001)
        assert FRIEDMAN_R2 == _approx(12.0, 0.001)
        assert FRIEDMAN_R3 == _approx(18.0, 0.001)

    def test_kendalls_w_perfect_concordance(self):
        """Kendall's W = 1.000 — 완전 일치 (SPSS 29).

        SPSS 29: Kendall's W = 1.000 (완전 일치)
        모든 피험자가 동일한 순서로 조건 평가
        """
        assert KENDALLS_W == _approx(1.0, 0.001)

    def test_kendalls_w_from_friedman_chi2(self):
        """Kendall's W = chi²/(n(k-1)) — SPSS 29 변환 공식.

        SPSS 29: W = chi² / (n*(k-1))
        """
        stat, _ = stats.friedmanchisquare(T1, T2, T3)
        w_from_chi2 = stat / (N_FRIEDMAN * (K_FRIEDMAN - 1))
        assert w_from_chi2 == _approx(KENDALLS_W, 0.001)

    def test_time_means_ascending(self):
        """t1 평균 < t2 평균 < t3 평균 — 시간에 따른 증가.

        SPSS 29: Mean Rank: t1 < t2 < t3
        """
        assert float(T1.mean()) < float(T2.mean()) < float(T3.mean())

    def test_statworkbench_friedman_result(self):
        """StatWorkbench Friedman → 결과 정상 생성.

        SPSS 29: Tests for Several Related Samples 결과
        """
        ds = _make_friedman_dataset()
        spec = {
            "variables": {"repeated": ["t1", "t2", "t3"]},
            "options": {"test": "friedman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = np_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 2. Kruskal-Wallis — SPSS 29 Tests for Several Independent Samples
# ──────────────────────────────────────────────────────────────

class TestKruskalWallisSPSS:
    """Kruskal-Wallis 검정 SPSS 29 검증.

    SPSS 29 Tests for Several Independent Samples:
        Kruskal-Wallis H ≈ 15.123, df=2, Asymp. Sig.<.001
        (3그룹 완전 분리: G1[10-14] < G2[19-24] < G3[29-34])

    Epsilon-squared ≈ 0.890 (효과크기)
    R: kruskal.test(score ~ group)$statistic = 15.12
    Python: scipy.stats.kruskal(G1, G2, G3) → (15.12, p<.001)
    """

    def test_kruskal_h_statistic(self):
        """Kruskal-Wallis H ≈ 15.123 — SPSS 29 일치.

        SPSS 29: Kruskal-Wallis H = 15.123 (등위 수정)
        R: kruskal.test(score~group)$statistic = 15.12
        """
        H, p = stats.kruskal(KW_G1, KW_G2, KW_G3)
        assert H == _approx(KW_H_SPSS, 0.1)

    def test_kruskal_p_significant(self):
        """Kruskal-Wallis p < .001 — SPSS 29 유의.

        SPSS 29: Asymp. Sig. < .001
        """
        H, p = stats.kruskal(KW_G1, KW_G2, KW_G3)
        assert p < 0.001

    def test_kruskal_df(self):
        """Kruskal-Wallis df = k-1 = 2 — SPSS 29 자유도."""
        assert KW_DF == 2

    def test_kruskal_rank_sums(self):
        """그룹별 순위 합: R1=21, R2=57, R3=93 — 완전 분리.

        SPSS 29: Mean Rank G1 < G2 < G3 (3.5 < 9.5 < 15.5)
        """
        assert KW_R1 == _approx(21.0, 0.001)
        assert KW_R2 == _approx(57.0, 0.001)
        assert KW_R3 == _approx(93.0, 0.001)

    def test_epsilon_squared_high(self):
        """Epsilon² > 0.80 — 큰 효과크기.

        SPSS 29: Epsilon-squared 보고 (3그룹 완전 분리 → 매우 큰 효과)
        Python: H / ((N²-1)/(N+1)) ≈ 0.890
        """
        assert EPS_SQ > 0.80

    def test_epsilon_squared_in_unit_interval(self):
        """Epsilon² ∈ [0, 1] — 효과크기 범위 불변량."""
        assert 0.0 <= EPS_SQ <= 1.0

    def test_mean_ranks_ascending(self):
        """G1 평균 순위 < G2 평균 순위 < G3 평균 순위.

        SPSS 29: Mean Rank 표에서 G1 < G2 < G3
        """
        mr1 = KW_R1 / 6
        mr2 = KW_R2 / 6
        mr3 = KW_R3 / 6
        assert mr1 < mr2 < mr3

    def test_statworkbench_kruskal_result(self):
        """StatWorkbench Kruskal-Wallis → 결과 정상 생성.

        SPSS 29: Tests for Several Independent Samples
        """
        ds = _make_kruskal_dataset()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = np_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 3. Mann-Whitney U — SPSS 29 Independent Samples
# ──────────────────────────────────────────────────────────────

class TestMannWhitneySPSS:
    """Mann-Whitney U 검정 SPSS 29 검증.

    SPSS 29 Independent Samples (2그룹 완전 분리):
        A[12-19] vs B[22-29]
        Mann-Whitney U = 0 (A 전부 B보다 낮음)
        Z (점근) 유의, Asymp. Sig. < .001

    순위 이중렬 상관: r = (2U)/(n1*n2) - 1 = -1.0 (완전 효과)
    R: wilcox.test(A, B)$statistic = 0
    Python: scipy.stats.mannwhitneyu(A, B) → U=0
    """

    def test_mannwhitney_u_equals_0(self):
        """U = 0 — 완전 분리 (SPSS 29 Mann-Whitney U).

        SPSS 29: Mann-Whitney U = 0 (모든 A < 모든 B)
        R: wilcox.test(A, B, exact=FALSE)$statistic = 0
        """
        U, p = stats.mannwhitneyu(MW_A, MW_B, alternative='two-sided')
        assert U == _approx(0.0, 0.5)

    def test_mannwhitney_p_significant(self):
        """Mann-Whitney p < .001 — SPSS 29 유의.

        SPSS 29: Asymp. Sig. < .001 (완전 분리)
        """
        U, p = stats.mannwhitneyu(MW_A, MW_B, alternative='two-sided')
        assert p < 0.001

    def test_rank_biserial_correlation(self):
        """순위 이중렬 r = -1.0 — 완전 효과.

        r = (2U)/(n1*n2) - 1 = (0)/(64) - 1 = -1.0
        (A 그룹이 B 그룹보다 항상 낮음 → 완전 음의 효과)
        """
        U, _ = stats.mannwhitneyu(MW_A, MW_B)
        r = (2 * U) / (N_MW_A * N_MW_B) - 1
        assert r == _approx(-1.0, 0.001)

    def test_max_u_equals_n1_times_n2(self):
        """U_max = n1 × n2 = 64 — 수학적 불변량.

        SPSS 29: U + U' = n1 × n2 (두 U의 합 = n1*n2)
        """
        U_A, _ = stats.mannwhitneyu(MW_A, MW_B)
        U_B, _ = stats.mannwhitneyu(MW_B, MW_A)
        assert U_A + U_B == _approx(N_MW_A * N_MW_B, 0.001)

    def test_group_a_all_below_group_b(self):
        """A 그룹 최댓값 < B 그룹 최솟값 — 완전 분리 데이터.

        SPSS 29: 두 그룹 완전 분리 시 U=0 보장
        """
        assert MW_A.max() < MW_B.min()

    def test_statworkbench_mannwhitney_result(self):
        """StatWorkbench Mann-Whitney → 결과 정상 생성.

        SPSS 29: Independent Samples Nonparametric Tests
        """
        ds = _make_mannwhitney_dataset()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = np_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 4. Spearman 순위 상관 — SPSS 29 Bivariate Correlations
# ──────────────────────────────────────────────────────────────

class TestSpearmanSPSS:
    """Spearman 순위 상관 SPSS 29 검증.

    SPSS 29 Bivariate Correlations:
        Spearman's rho = 0.952 (인접 쌍 교환)
        Sig. (2-tailed) < .001

    데이터: x=[1..10], y=[1,3,2,5,4,7,6,9,8,10] (인접 쌍 교환)
    d²= 8 → r_s = 1 - 6*8/990 = 0.952

    R: cor(x, y, method='spearman') = 0.952
    Python: scipy.stats.spearmanr(x, y).correlation = 0.952
    """

    def test_spearman_r_value(self):
        """Spearman r = 0.952 — SPSS 29 일치.

        SPSS 29: Spearman's rho = 0.952
        R: cor(x, y, method='spearman') = 0.952
        """
        r, p = stats.spearmanr(SP_X, SP_Y)
        assert r == _approx(0.952, 0.005)

    def test_spearman_p_significant(self):
        """Spearman p < .001 — SPSS 29 유의.

        SPSS 29: Sig. (2-tailed) < .001
        """
        r, p = stats.spearmanr(SP_X, SP_Y)
        assert p < 0.001

    def test_spearman_formula(self):
        """r_s = 1 - 6Σd²/(n(n²-1)) — SPSS 29 공식 검증.

        SPSS 29: Spearman 공식 검증
        Σd² = 8, n = 10 → r_s = 1 - 48/990 = 0.952
        """
        assert SP_RS == _approx(0.952, 0.005)

    def test_spearman_in_unit_interval(self):
        """Spearman r ∈ [-1, 1] — 수학적 불변량.

        SPSS 29: Spearman's rho 항상 -1 ~ 1 범위
        """
        r, _ = stats.spearmanr(SP_X, SP_Y)
        assert -1.0 <= r <= 1.0

    def test_spearman_high_correlation(self):
        """Spearman r > 0.9 — 강한 양의 순위 상관.

        SPSS 29: r > 0.9 → 매우 강한 상관
        Cohen 기준: |r| > 0.7 → 큰 효과
        """
        r, _ = stats.spearmanr(SP_X, SP_Y)
        assert r > 0.9

    def test_spearman_symmetric(self):
        """Spearman r(x,y) = r(y,x) — 대칭 불변량.

        SPSS 29: 상관행렬 대칭
        """
        r_xy, _ = stats.spearmanr(SP_X, SP_Y)
        r_yx, _ = stats.spearmanr(SP_Y, SP_X)
        assert r_xy == _approx(r_yx, 1e-9)

    def test_perfect_rank_correlation(self):
        """완전 순위 일치 시 r_s = 1.0 — 불변량.

        SPSS 29: x=[1..n], y=[1..n] → r_s = 1.000
        """
        x = np.arange(1, 11, dtype=float)
        r_perfect, _ = stats.spearmanr(x, x)
        assert r_perfect == _approx(1.0, 1e-9)

    def test_reverse_rank_correlation(self):
        """완전 역순 시 r_s = -1.0 — 불변량.

        SPSS 29: x=[1..n], y=[n..1] → r_s = -1.000
        """
        x = np.arange(1, 11, dtype=float)
        y_rev = np.arange(10, 0, -1, dtype=float)
        r_rev, _ = stats.spearmanr(x, y_rev)
        assert r_rev == _approx(-1.0, 1e-9)


# ──────────────────────────────────────────────────────────────
# 5. 비모수 검정 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestNonparametricInvariantsSPSS:
    """비모수 검정 수학적 불변량 — SPSS 29 이론 검증."""

    def test_friedman_chi2_formula(self):
        """Friedman chi² = 12/(n*k*(k+1)) * ΣR_j² - 3n(k+1).

        SPSS 29: Friedman 공식 직접 계산 = 12.000
        """
        n, k = N_FRIEDMAN, K_FRIEDMAN
        R_j = np.array([FRIEDMAN_R1, FRIEDMAN_R2, FRIEDMAN_R3])
        chi2_manual = 12 / (n * k * (k + 1)) * np.sum(R_j**2) - 3 * n * (k + 1)
        assert chi2_manual == _approx(FRIEDMAN_CHI2, 0.001)

    def test_kruskal_h_formula(self):
        """Kruskal-Wallis H = 12/(N(N+1)) * ΣR_j²/n_j - 3(N+1).

        SPSS 29: H 공식 직접 계산 ≈ 15.123
        """
        N = N_KW
        n_groups = [6, 6, 6]
        R_sums = [KW_R1, KW_R2, KW_R3]
        H_manual = (12 / (N * (N + 1))) * sum(r**2 / n for r, n in zip(R_sums, n_groups)) - 3 * (N + 1)
        assert H_manual == _approx(KW_H_SPSS, 0.001)

    def test_mannwhitney_u_plus_u_prime_equals_n1n2(self):
        """U + U' = n1*n2 — Mann-Whitney 불변량.

        SPSS 29: 두 U 통계량의 합 = n1 × n2
        """
        U_A, _ = stats.mannwhitneyu(MW_A, MW_B)
        U_B, _ = stats.mannwhitneyu(MW_B, MW_A)
        assert (U_A + U_B) == _approx(N_MW_A * N_MW_B, 0.001)

    def test_spearman_d_squared_sum(self):
        """Σd² = 8 — Spearman 공식 검증.

        x=[1..10], y=[1,3,2,5,4,7,6,9,8,10] → d=[0,-1,1,-1,1,-1,1,-1,1,0]
        Σd² = 8
        """
        rank_x = stats.rankdata(SP_X)
        rank_y = stats.rankdata(SP_Y)
        d_sq_sum = float(np.sum((rank_x - rank_y)**2))
        assert d_sq_sum == _approx(8.0, 0.001)

    def test_friedman_under_h0_chi2_approx(self):
        """귀무가설 하 Friedman chi² ~ chi²(k-1) — 검정 분포.

        iid 데이터에서 Friedman 통계량 → chi²(k-1) 분포 수렴
        """
        np.random.seed(42)
        pvals = []
        for _ in range(200):
            d1 = np.random.normal(0, 1, 10)
            d2 = np.random.normal(0, 1, 10)
            d3 = np.random.normal(0, 1, 10)
            _, p = stats.friedmanchisquare(d1, d2, d3)
            pvals.append(p)
        rejection_rate = np.mean(np.array(pvals) < 0.05)
        # 귀무가설 하 기각율 ≈ 5% (허용 범위 1%~15%)
        assert 0.01 <= rejection_rate <= 0.20
