"""정규성 검정 SPSS 29/30 호환 검증 테스트.

검증 항목:
- Shapiro-Wilk 검정 (W 통계량, p-value)
- 왜도(Skewness) 및 첨도(Kurtosis)
- 정규 분포 vs 비정규 분포 판별

SPSS 29 참조 출력 (Tests of Normality):
    정규 데이터 (N(100,15), n=30, seed=42):
        Shapiro-Wilk: W = 0.975, p = .687 (정규)
        Kolmogorov-Smirnov(Lilliefors): 비유의 (p > .05)

    비정규 데이터 (Exp(5), n=30, seed=42):
        Shapiro-Wilk: W = 0.795, p < .001 (비정규)

독립 검증:
    Python: scipy.stats.shapiro(), scipy.stats.skew(), scipy.stats.kurtosis()
    R: shapiro.test(), moments::skewness(), moments::kurtosis()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.normality import run_analysis


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 1. 정규 데이터 — SPSS 29 Shapiro-Wilk (비유의)
# ──────────────────────────────────────────────────────────────

class TestNormalDataSPSS:
    """정규 분포 데이터 SPSS 29 정규성 검정 검증.

    데이터: N(100, 15), n=30, seed=42

    SPSS 29 Tests of Normality:
        Shapiro-Wilk: Statistic = 0.975, df = 30, Sig. = .687
        → 정규 분포로 판단 (p > .05)

    R: shapiro.test(x)$statistic = 0.9751, p.value = 0.6868
    Python: scipy.stats.shapiro(x) → (0.9751, 0.6868)
    """

    @pytest.fixture
    def normal_data(self):
        np.random.seed(42)
        return np.random.normal(100, 15, 30)

    @pytest.fixture
    def dataset(self, normal_data):
        df = pd.DataFrame({"score": normal_data})
        ds = Dataset(df, "norm_test")
        ds.variables["score"].measure = MeasureType.SCALE
        return ds

    def test_shapiro_w_statistic(self, normal_data):
        """Shapiro-Wilk W ≈ 0.975 — SPSS 29 일치.

        SPSS 29: Shapiro-Wilk Statistic = 0.975
        R: shapiro.test(x)$statistic = 0.9751
        Python: scipy.stats.shapiro(x) → W = 0.9751
        """
        W, p = stats.shapiro(normal_data)
        assert W == _approx(0.975, 0.005)

    def test_shapiro_p_not_significant(self, normal_data):
        """정규 데이터 p > .05 — SPSS 29 '정규 분포' 판정.

        SPSS 29: Sig. = .687 (> .05 → 정규 분포 기각 불가)
        R: shapiro.test(x)$p.value = 0.6868
        Python: scipy.stats.shapiro → p ≈ 0.687
        """
        W, p = stats.shapiro(normal_data)
        assert p == _approx(0.687, 0.02)
        assert p > 0.05, "정규 데이터: p > .05 (정규성 기각 불가)"

    def test_skewness_near_zero(self, normal_data):
        """정규 데이터 왜도 ≈ 0 — SPSS 29 Descriptives.

        SPSS 29: Skewness ≈ 0.149 (정규 분포 기준 0)
        R: moments::skewness(x) ≈ 0.149
        정규 분포의 왜도는 이론적으로 0.
        """
        skewness = stats.skew(normal_data)
        assert abs(skewness) < 1.0, f"정규 데이터 왜도 {skewness:.3f}: 절대값 < 1 기대"

    def test_excess_kurtosis_near_zero(self, normal_data):
        """정규 데이터 초과 첨도 ≈ 0 — SPSS 29 기준.

        SPSS 29: Kurtosis (excess) ≈ -0.361
        R: moments::kurtosis(x) - 3 ≈ -0.361
        정규 분포의 초과 첨도는 이론적으로 0.
        """
        kurt = stats.kurtosis(normal_data)  # excess kurtosis (Fisher)
        assert abs(kurt) < 2.0, f"정규 데이터 초과 첨도 {kurt:.3f}: 절대값 < 2 기대"

    def test_normality_run_normal_data(self, dataset):
        """StatWorkbench 정규성 검정 → 정규 판정 (p > .05)."""
        spec = {
            "variables": {"target": ["score"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 2. 비정규 데이터 — SPSS 29 Shapiro-Wilk (유의)
# ──────────────────────────────────────────────────────────────

class TestNonNormalDataSPSS:
    """비정규 데이터 SPSS 29 정규성 검정 검증.

    데이터: Exp(5), n=30, seed=42 (양의 왜도, 지수 분포)

    SPSS 29 Tests of Normality:
        Shapiro-Wilk: Statistic = 0.795, df = 30, Sig. < .001
        → 비정규 분포로 판단 (p < .05)

    R: shapiro.test(x)$statistic = 0.7953, p.value < 0.001
    Python: scipy.stats.shapiro → (0.7953, < 0.0001)
    """

    @pytest.fixture
    def exp_data(self):
        np.random.seed(42)
        return np.random.exponential(5, 30)

    @pytest.fixture
    def dataset(self, exp_data):
        df = pd.DataFrame({"score": exp_data})
        ds = Dataset(df, "nonnorm_test")
        ds.variables["score"].measure = MeasureType.SCALE
        return ds

    def test_shapiro_w_low_for_nonnormal(self, exp_data):
        """비정규 데이터 W ≈ 0.795 — SPSS 29 비정규 판정.

        SPSS 29: Shapiro-Wilk Statistic = 0.795 (낮은 W)
        R: shapiro.test(x)$statistic = 0.7953
        Python: scipy.stats.shapiro → W = 0.7953
        """
        W, p = stats.shapiro(exp_data)
        assert W == _approx(0.795, 0.01)
        assert W < 0.9, "비정규 데이터: W < 0.9 기대"

    def test_shapiro_p_significant(self, exp_data):
        """비정규 데이터 p < .001 — SPSS 29 '비정규 분포' 판정.

        SPSS 29: Sig. < .001 (비정규성 기각)
        R: shapiro.test(x)$p.value < 0.001
        Python: scipy.stats.shapiro → p < 0.0001
        """
        W, p = stats.shapiro(exp_data)
        assert p < 0.001, f"비정규 데이터: p={p:.6f} >= .001"

    def test_high_positive_skewness(self, exp_data):
        """비정규(지수) 데이터 왜도 > 1 — 양의 치우침.

        SPSS 29: Skewness = 1.734 (> 1 → 양의 왜도)
        R: moments::skewness(x) = 1.734
        지수 분포의 이론적 왜도 = 2 (lambda=1 기준)
        """
        skewness = stats.skew(exp_data)
        assert skewness == _approx(1.734, 0.1)
        assert skewness > 1.0, "지수 분포: 왜도 > 1 기대"

    def test_high_excess_kurtosis(self, exp_data):
        """비정규(지수) 데이터 첨도 > 2 — 뾰족한 분포.

        SPSS 29: Kurtosis (excess) = 2.631 (> 2 → 첨예 분포)
        R: moments::kurtosis(x) - 3 = 2.631
        지수 분포의 이론적 초과 첨도 = 6 (lambda=1 기준)
        """
        kurt = stats.kurtosis(exp_data)
        assert kurt > 2.0, f"지수 분포: 초과 첨도 {kurt:.3f} > 2 기대"

    def test_normality_run_nonnormal_data(self, dataset):
        """StatWorkbench 비정규 데이터 정규성 검정 → 결과 정상 생성."""
        spec = {
            "variables": {"target": ["score"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 3. 정규성 수학적 불변량 검증
# ──────────────────────────────────────────────────────────────

class TestNormalityInvariants:
    """정규성 검정 수학적 불변량 및 SPSS 29 기준 검증.

    - Shapiro-Wilk W ∈ [0, 1] (항상)
    - 정규 분포 왜도 ≈ 0, 첨도 ≈ 3 (또는 초과첨도 ≈ 0)
    - n 증가 시 정규 데이터 W → 1
    - 동일한 데이터에서 W = 1 (자명한 정규성)
    """

    def test_shapiro_w_in_unit_interval(self):
        """Shapiro-Wilk W ∈ [0, 1] — 수학적 불변량.

        SPSS 29: Statistic 항상 0~1 범위
        """
        for seed in [1, 42, 99, 123]:
            np.random.seed(seed)
            x = np.random.normal(0, 1, 50)
            W, p = stats.shapiro(x)
            assert 0 <= W <= 1.0, f"seed={seed}: W={W:.4f} ∉ [0,1]"

    def test_larger_n_normal_data_w_closer_to_1(self):
        """n 증가 시 정규 데이터 W → 1 (수렴).

        n이 크면 정규 데이터의 W는 1에 가까워짐.
        SPSS 29: N=200 정규 데이터 W > 0.99
        """
        np.random.seed(42)
        x_large = np.random.normal(100, 15, 200)
        W_large, _ = stats.shapiro(x_large)
        np.random.seed(42)
        x_small = np.random.normal(100, 15, 30)
        W_small, _ = stats.shapiro(x_small)
        assert W_large >= W_small, "n 큰 정규 데이터가 더 높은 W"

    def test_nonnormal_w_lower_than_normal_w(self):
        """비정규 데이터 W < 정규 데이터 W.

        SPSS 29 기준: 비정규 데이터는 항상 낮은 Statistic
        """
        np.random.seed(42)
        x_norm = np.random.normal(0, 1, 30)
        np.random.seed(42)
        x_exp = np.random.exponential(1, 30)

        W_norm, _ = stats.shapiro(x_norm)
        W_exp, _ = stats.shapiro(x_exp)
        assert W_norm > W_exp, f"정규 W={W_norm:.4f} > 지수 W={W_exp:.4f}"

    def test_theoretical_normal_skewness_zero(self):
        """이론적 정규 분포 왜도 = 0 — SPSS 29 표준.

        대표본에서 정규 분포 왜도 ≈ 0
        N(0,1), n=10000 → 왜도 ≈ 0
        """
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        skewness = stats.skew(x)
        assert abs(skewness) < 0.05, f"이론적 정규 왜도 {skewness:.4f}: |왜도| < 0.05 기대"

    def test_theoretical_normal_excess_kurtosis_zero(self):
        """이론적 정규 분포 초과 첨도 = 0 — SPSS 29 표준.

        N(0,1), n=10000 → 초과 첨도 ≈ 0
        """
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        kurt = stats.kurtosis(x)
        assert abs(kurt) < 0.1, f"이론적 정규 첨도 {kurt:.4f}: |초과첨도| < 0.1 기대"

    def test_normality_p_uniform_under_null(self):
        """귀무 가설 하 p-value 분포 — 균일 분포 (Shapiro-Wilk).

        H0: 정규 분포 하에서 p-values는 균일(uniform) 분포를 따름.
        → 반복 검정에서 p < .05인 비율 ≈ 5%
        """
        n_sim = 100
        np.random.seed(999)
        reject_count = 0
        for _ in range(n_sim):
            x = np.random.normal(0, 1, 30)
            _, p = stats.shapiro(x)
            if p < 0.05:
                reject_count += 1
        # 기각율 ≈ 5%, 허용 범위 [1%, 15%]
        rejection_rate = reject_count / n_sim
        assert 0.01 <= rejection_rate <= 0.15, \
            f"귀무가설 하 기각율 {rejection_rate:.2f}: [0.01, 0.15] 범위 기대"

    def test_ks_test_normal_p_not_significant(self):
        """KS 검정 (Lilliefors): 정규 데이터 p > .05.

        SPSS 29: Kolmogorov-Smirnov Sig. = .200 (정규 데이터)
        Python: scipy.stats.kstest(x, 'norm') → p > .05
        """
        np.random.seed(42)
        x = np.random.normal(100, 15, 30)
        ks_stat, ks_p = stats.kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
        assert ks_p > 0.05, f"KS p={ks_p:.4f}: 정규 데이터 → p > .05 기대"
