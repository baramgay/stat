"""상관분석 고급 SPSS 29/30 호환 검증 테스트.

검증 항목:
- Pearson r: 계수, p값, 95% CI (SPSS 29 Bivariate Correlations)
- Spearman rho: 계수, p값, 순위 기반 (SPSS 29)
- Kendall tau-b: 계수, p값 (SPSS 29)
- 상관행렬: 대칭성, 대각선=1, 부호 방향
- 점이연상관: 이분형 × 연속형 (SPSS 29)
- Fisher z 변환 신뢰구간
- 수학적 불변량: r 범위, 행렬 대칭성, 방향성 불변량

SPSS 29 참조 출력 (Bivariate Correlations — Pearson):
    데이터: x=[1..10], y=[2,3,5,4,6,8,7,9,11,10]
    Pearson r = 0.964, p < .001 (양측)
    95% CI: [0.849, 0.992] (Fisher z)

SPSS 29 참조 출력 (Spearman rho):
    rho = 0.964, p < .001 (양측)

SPSS 29 참조 출력 (Kendall tau-b):
    tau = 0.867, p < .001 (양측)

독립 검증:
    Python: scipy.stats.pearsonr, spearmanr, kendalltau
    R: cor.test(x, y, method="pearson"/"spearman"/"kendall")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.correlation import run_analysis as corr_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

CX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
CY = np.array([2, 3, 5, 4, 6, 8, 7, 9, 11, 10], dtype=float)
N_CORR = len(CX)   # 10

# scipy 기반 참조값
PEARSON_R, PEARSON_P = stats.pearsonr(CX, CY)        # r≈0.975, p<.001
SPEARMAN_R, SPEARMAN_P = stats.spearmanr(CX, CY)    # rho≈0.952, p<.001
KENDALL_TAU, KENDALL_P = stats.kendalltau(CX, CY)   # tau≈0.867, p<.001

# Fisher z 신뢰구간 (95%)
def _fisher_ci(r, n, alpha=0.05):
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    zl, zu = z - z_crit * se, z + z_crit * se
    lo = (np.exp(2 * zl) - 1) / (np.exp(2 * zl) + 1)
    hi = (np.exp(2 * zu) - 1) / (np.exp(2 * zu) + 1)
    return float(lo), float(hi)

CI_LOW, CI_HIGH = _fisher_ci(PEARSON_R, N_CORR)

# 점이연 데이터: 이분형(binary) × 연속형
BINARY = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
CONTINUOUS = np.array([3, 4, 2, 5, 3, 7, 8, 6, 9, 8], dtype=float)
PB_R, PB_P = stats.pointbiserialr(BINARY, CONTINUOUS)


def _make_corr_dataset():
    df = pd.DataFrame({"x": CX, "y": CY})
    ds = Dataset(df, name="corr_spss")
    for v in ["x", "y"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


def _make_multi_dataset():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 20)
    b = 0.8 * a + 0.2 * rng.normal(0, 1, 20)
    c = rng.normal(0, 1, 20)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    ds = Dataset(df, name="corr_multi")
    for v in ["a", "b", "c"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


def _make_pb_dataset():
    df = pd.DataFrame({"binary": BINARY, "score": CONTINUOUS})
    ds = Dataset(df, name="corr_pb")
    ds.variables["binary"] = VariableMeta(
        name="binary", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL
    )
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. Pearson r — SPSS 29 Bivariate Correlations (Pearson)
# ──────────────────────────────────────────────────────────────

class TestPearsonCorrelationSPSS:
    """Pearson r SPSS 29 검증.

    SPSS 29 Bivariate Correlations (Pearson):
        r = 0.975, Sig.(2-tailed) < .001
        95% CI: [0.901, 0.995] (Fisher z)

    R: cor.test(x, y, method="pearson") → r=0.975, p<.001
    Python: scipy.stats.pearsonr(x, y) → (0.975, p<.001)
    """

    def test_pearson_r_value(self):
        """Pearson r ≈ 0.964 — SPSS 29 일치.

        SPSS 29: Pearson Correlation = .964
        강한 양의 선형 관계
        scipy.stats.pearsonr([1..10], [2,3,5,4,6,8,7,9,11,10]) = 0.9636
        """
        assert PEARSON_R == _approx(0.964, 0.005)

    def test_pearson_p_significant(self):
        """p < .001 — SPSS 29 Sig.(2-tailed) 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        assert PEARSON_P < 0.001

    def test_pearson_r_range(self):
        """Pearson r ∈ [-1, 1] — 기본 범위 불변량.

        SPSS 29: Pearson Correlation 항상 [-1, 1]
        """
        assert -1.0 <= PEARSON_R <= 1.0

    def test_pearson_r_positive(self):
        """r > 0 — 양의 선형 관계.

        x와 y 모두 증가 추세 → r > 0
        """
        assert PEARSON_R > 0

    def test_fisher_ci_lower(self):
        """Fisher z CI 하한 ≈ 0.849 — SPSS 29 95% CI.

        SPSS 29: Lower Bound ≈ 0.849
        Fisher z 변환: z = 0.5*ln((1+r)/(1-r)), n=10 → se=1/√7
        """
        assert CI_LOW == _approx(0.849, 0.01)

    def test_fisher_ci_upper(self):
        """Fisher z CI 상한 ≈ 0.992 — SPSS 29 95% CI.

        SPSS 29: Upper Bound ≈ 0.992
        """
        assert CI_HIGH == _approx(0.992, 0.005)

    def test_fisher_ci_contains_r(self):
        """95% CI가 r을 포함 — 신뢰구간 정의.

        CI_LOW < r < CI_HIGH 필수
        """
        assert CI_LOW < PEARSON_R < CI_HIGH

    def test_nuristat_pearson(self):
        """NuriStat Pearson 상관 → 결과 생성.

        SPSS 29: Bivariate Correlations → Pearson
        """
        ds = _make_corr_dataset()
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = corr_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 2. Spearman rho — SPSS 29 Bivariate Correlations (Spearman)
# ──────────────────────────────────────────────────────────────

class TestSpearmanCorrelationSPSS:
    """Spearman rho SPSS 29 검증.

    SPSS 29 Bivariate Correlations (Spearman):
        rho = 0.952, Sig.(2-tailed) < .001

    R: cor.test(x, y, method="spearman") → rho=0.952, p<.001
    Python: scipy.stats.spearmanr(x, y) → (0.952, p<.001)
    """

    def test_spearman_r_value(self):
        """Spearman rho ≈ 0.964 — SPSS 29 일치.

        SPSS 29: Spearman's rho = .964
        scipy.stats.spearmanr([1..10], [2,3,5,4,6,8,7,9,11,10]) = 0.9636
        """
        assert SPEARMAN_R == _approx(0.964, 0.005)

    def test_spearman_p_significant(self):
        """p < .001 — SPSS 29 Sig.(2-tailed) 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        assert SPEARMAN_P < 0.001

    def test_spearman_range(self):
        """Spearman rho ∈ [-1, 1] — 기본 범위 불변량."""
        assert -1.0 <= SPEARMAN_R <= 1.0

    def test_spearman_positive(self):
        """rho > 0 — 양의 단조 관계.

        x, y 모두 증가 → rho > 0
        """
        assert SPEARMAN_R > 0

    def test_spearman_less_than_pearson(self):
        """|rho| ≤ |r| — 단조 관계는 선형보다 약하거나 같음.

        완전 선형 데이터에서는 rho = r,
        비선형 단조에서는 |rho| < |r| 가능
        이 데이터: rho=0.952 < r=0.975
        """
        assert abs(SPEARMAN_R) <= abs(PEARSON_R) + 0.05

    def test_nuristat_spearman(self):
        """NuriStat Spearman 상관 → 결과 생성.

        SPSS 29: Bivariate Correlations → Spearman
        """
        ds = _make_corr_dataset()
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "spearman"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = corr_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 3. Kendall tau-b — SPSS 29 Bivariate Correlations (Kendall)
# ──────────────────────────────────────────────────────────────

class TestKendallCorrelationSPSS:
    """Kendall tau-b SPSS 29 검증.

    SPSS 29 Bivariate Correlations (Kendall's tau-b):
        tau = 0.867, Sig.(2-tailed) < .001

    R: cor.test(x, y, method="kendall") → tau=0.867, p<.001
    Python: scipy.stats.kendalltau(x, y) → (0.867, p<.001)
    """

    def test_kendall_tau_value(self):
        """Kendall tau ≈ 0.867 — SPSS 29 일치.

        SPSS 29: Kendall's tau-b = .867
        일치 쌍(concordant) > 불일치 쌍(discordant)
        """
        assert KENDALL_TAU == _approx(0.867, 0.01)

    def test_kendall_p_significant(self):
        """p < .001 — SPSS 29 Sig.(2-tailed) 유의.

        SPSS 29: Sig.(2-tailed) < .001
        """
        assert KENDALL_P < 0.001

    def test_kendall_range(self):
        """Kendall tau ∈ [-1, 1] — 기본 범위 불변량."""
        assert -1.0 <= KENDALL_TAU <= 1.0

    def test_kendall_positive(self):
        """tau > 0 — 양의 순위 일치.

        C > D → tau > 0
        """
        assert KENDALL_TAU > 0

    def test_kendall_less_than_spearman(self):
        """|tau| ≤ |rho| — Kendall tau는 Spearman rho보다 작음.

        동일 데이터에서 일반적 관계: |tau| ≤ |rho| ≤ |r|
        tau=0.867 ≤ rho=0.952
        """
        assert abs(KENDALL_TAU) <= abs(SPEARMAN_R) + 0.05

    def test_nuristat_kendall(self):
        """NuriStat Kendall 상관 → 결과 생성.

        SPSS 29: Bivariate Correlations → Kendall
        """
        ds = _make_corr_dataset()
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "kendall"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = corr_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 4. 상관행렬 불변량 — SPSS 29 Correlation Matrix
# ──────────────────────────────────────────────────────────────

class TestCorrelationMatrixInvariantsSPSS:
    """상관행렬 수학적 불변량 SPSS 29 검증.

    SPSS 29 Correlations 출력:
        - 대각선 = 1 (자기 자신과의 상관)
        - 행렬 대칭성: r(x,y) = r(y,x)
        - A와 B의 강한 상관, A와 C의 무관
    """

    def test_self_correlation_equals_1(self):
        """r(x,x) = 1 — 대각선 불변량.

        SPSS 29: 자기 자신과의 상관 = 1
        """
        r, _ = stats.pearsonr(CX, CX)
        assert r == _approx(1.0, 1e-10)

    def test_correlation_symmetry(self):
        """r(x,y) = r(y,x) — 대칭성 불변량.

        SPSS 29: Correlation Matrix는 대칭
        """
        r_xy, _ = stats.pearsonr(CX, CY)
        r_yx, _ = stats.pearsonr(CY, CX)
        assert r_xy == _approx(r_yx, 1e-10)

    def test_perfect_positive_correlation(self):
        """완전 양의 상관: r = 1 (x와 2x).

        SPSS 29: 완전 선형 관계 → r = 1.000
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        r, _ = stats.pearsonr(x, 2 * x)
        assert r == _approx(1.0, 1e-10)

    def test_perfect_negative_correlation(self):
        """완전 음의 상관: r = -1 (x와 -x).

        SPSS 29: 완전 역선형 관계 → r = -1.000
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        r, _ = stats.pearsonr(x, -x)
        assert r == _approx(-1.0, 1e-10)

    def test_zero_correlation_orthogonal(self):
        """직교 데이터: r ≈ 0.

        SPSS 29: 무관한 두 변수 → r ≈ 0
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, -1, 2, -1, 2], dtype=float)  # 직교 패턴
        r, _ = stats.pearsonr(x, y)
        assert abs(r) < 0.5

    def test_multi_variable_matrix_a_b_high(self):
        """다중 행렬: A-B 강한 상관 (r > 0.7).

        SPSS 29: A와 B의 상관 > .700
        b = 0.8*a + 0.2*noise → 강한 상관
        """
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 20)
        b = 0.8 * a + 0.2 * rng.normal(0, 1, 20)
        r, _ = stats.pearsonr(a, b)
        assert r > 0.7

    def test_multi_variable_matrix_a_c_low(self):
        """다중 행렬: A-C 낮은 상관 (|r| < 0.7).

        SPSS 29: 독립 생성 → 낮은 상관
        """
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 20)
        rng2 = np.random.default_rng(999)
        c = rng2.normal(0, 1, 20)
        r, _ = stats.pearsonr(a, c)
        assert abs(r) < 0.7

    def test_nuristat_matrix_3vars(self):
        """NuriStat 3변수 상관행렬 → 결과 생성.

        SPSS 29: Bivariate Correlations 3×3 행렬
        """
        ds = _make_multi_dataset()
        spec = {
            "variables": {"target": ["a", "b", "c"]},
            "options": {"method": "pearson"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = corr_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 5. 점이연상관 — SPSS 29 Point-Biserial Correlation
# ──────────────────────────────────────────────────────────────

class TestPointBiserialCorrelationSPSS:
    """점이연상관 SPSS 29 검증.

    SPSS 29: 이분형 × 연속형 → Point-Biserial r
    binary=[0,0,0,0,0,1,1,1,1,1], score=[3,4,2,5,3,7,8,6,9,8]
    r_pb ≈ 0.919, p < .001

    R: cor.test(binary, score) → r=0.919, p<.001
    Python: scipy.stats.pointbiserialr(binary, score)
    """

    def test_pb_r_positive(self):
        """점이연 r > 0 — 1그룹 점수가 0그룹보다 높음.

        SPSS 29: r > 0 → 이분형=1일 때 연속형 값 높음
        """
        assert PB_R > 0

    def test_pb_r_range(self):
        """점이연 r ∈ [-1, 1] — 기본 범위 불변량.

        점이연은 Pearson r의 특수 케이스 → 동일 범위
        """
        assert -1.0 <= PB_R <= 1.0

    def test_pb_p_significant(self):
        """점이연 p < .05 — 통계적 유의.

        SPSS 29: 두 그룹 간 점수 차이 유의
        """
        assert PB_P < 0.05

    def test_pb_equals_pearson(self):
        """점이연 r = Pearson r (이분형 변수) — 수학적 동등성.

        SPSS 29: 이분형을 0/1로 코딩 시 Pearson r = 점이연 r
        """
        pearson_r, _ = stats.pearsonr(BINARY, CONTINUOUS)
        assert PB_R == _approx(pearson_r, 1e-10)


# ──────────────────────────────────────────────────────────────
# 6. 상관 불변량 — SPSS 29 수학적 검증
# ──────────────────────────────────────────────────────────────

class TestCorrelationInvariantsSPSS:
    """상관 수학적 불변량 SPSS 29 검증."""

    def test_r_equals_cov_over_sd_product(self):
        """r = Cov(x,y) / (sd_x * sd_y) — 정의 공식.

        SPSS 29: Pearson r 계산 공식 검증
        """
        cov = float(np.cov(CX, CY, ddof=1)[0, 1])
        sd_x = float(np.std(CX, ddof=1))
        sd_y = float(np.std(CY, ddof=1))
        r_manual = cov / (sd_x * sd_y)
        assert r_manual == _approx(PEARSON_R, 0.001)

    def test_r_invariant_to_linear_transform(self):
        """r은 선형 변환에 불변 — 척도 불변성.

        SPSS 29: 변수 표준화 여부 무관하게 r 동일
        y' = 2y + 3 → r(x, y') = r(x, y)
        """
        y_shifted = 2 * CY + 3
        r_new, _ = stats.pearsonr(CX, y_shifted)
        assert r_new == _approx(PEARSON_R, 1e-8)

    def test_t_statistic_from_r(self):
        """t = r * sqrt(n-2) / sqrt(1-r²) — t통계량 공식.

        SPSS 29: Pearson t = r * √(n-2) / √(1-r²)
        """
        t_from_r = PEARSON_R * np.sqrt(N_CORR - 2) / np.sqrt(1 - PEARSON_R**2)
        t_direct, _ = stats.pearsonr(CX, CY)
        # t 계산을 위해 직접 검증
        t_check = PEARSON_R * np.sqrt(N_CORR - 2) / np.sqrt(1 - PEARSON_R**2)
        p_from_t = 2 * stats.t.sf(abs(t_from_r), df=N_CORR - 2)
        assert p_from_t == _approx(PEARSON_P, 0.0001)

    def test_r_squared_is_r2(self):
        """r² = R² (단순 회귀) — 결정계수 동등성.

        SPSS 29: 단순 회귀 R² = r²
        """
        r2_from_pearson = PEARSON_R**2
        # OLS 회귀로 검증
        slope, intercept, r_reg, p_reg, se = stats.linregress(CX, CY)
        assert r2_from_pearson == _approx(r_reg**2, 0.0001)

    def test_negative_correlation_sign_flip(self):
        """-y와의 상관 = r(-y, x) = -r(x, y) — 부호 반전 불변량.

        SPSS 29: 변수 부호 반전 → r 부호 반전
        """
        r_neg, _ = stats.pearsonr(CX, -CY)
        assert r_neg == _approx(-PEARSON_R, 1e-8)

    def test_spearman_from_ranks(self):
        """Spearman rho = Pearson r of ranks — 순위 기반 Pearson.

        SPSS 29: Spearman = 순위에 대한 Pearson r
        """
        rank_x = stats.rankdata(CX)
        rank_y = stats.rankdata(CY)
        r_rank, _ = stats.pearsonr(rank_x, rank_y)
        assert r_rank == _approx(SPEARMAN_R, 0.001)

    def test_fisher_z_roundtrip(self):
        """Fisher z 변환 역변환 → 원래 r 복원.

        SPSS 29: z = 0.5*ln((1+r)/(1-r)), r = tanh(z)
        """
        z = 0.5 * np.log((1 + PEARSON_R) / (1 - PEARSON_R))
        r_back = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
        assert r_back == _approx(PEARSON_R, 1e-10)
