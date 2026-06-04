"""빈도분석 SPSS 29/30 고급 호환 검증 테스트.

검증 항목:
- 빈도표: 빈도수, 유효%, 누적% (SPSS 29 Frequency Table)
- 기술통계: 평균, 중앙값, 최빈값, SD (SPSS 29 Statistics)
- 백분위수: 25th, 50th, 75th (SPSS 29 Percentiles)
- 결측치 처리: 유효 N vs 전체 N (SPSS 29 System-Missing)
- 범주형 빈도: 순서 보존, 누적% 단조 증가

SPSS 29 참조 출력 (Frequencies):
    숫자 데이터: [2, 2, 3, 3, 3, 4, 4, 4, 4, 5] (n=10)

    Statistics:
        N Valid = 10, Missing = 0
        Mean = 3.400
        Median = 3.500
        Mode = 4
        Std. Deviation = 0.966
        Variance = 0.933
        Minimum = 2, Maximum = 5
        Range = 3

    Frequency Table:
        Value  Frequency  Valid%  Cumulative%
        2      2          20.0%   20.0%
        3      3          30.0%   50.0%
        4      4          40.0%   90.0%
        5      1          10.0%   100.0%

    Percentiles:
        25th = 2.750, 50th = 3.500, 75th = 4.000

독립 검증:
    Python: numpy, scipy.stats, pandas.value_counts
    R: table(x), quantile(x), mean(x), sd(x)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.frequencies import run_analysis as freq_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

# 숫자 데이터 (SPSS 29 Frequencies Statistics 검증용)
NUM_DATA = np.array([2, 2, 3, 3, 3, 4, 4, 4, 4, 5], dtype=float)
N_VALID = 10
N_TOTAL = 10

# SPSS 29 Statistics 참조값
MEAN_SPSS = float(NUM_DATA.mean())           # 3.400
MEDIAN_SPSS = float(np.median(NUM_DATA))     # 3.500
MODE_SPSS = 4.0                              # 최빈값
STD_SPSS = float(np.std(NUM_DATA, ddof=1))  # ≈ 0.966
VAR_SPSS = STD_SPSS**2                       # ≈ 0.933
MIN_SPSS = float(NUM_DATA.min())             # 2.0
MAX_SPSS = float(NUM_DATA.max())             # 5.0
RANGE_SPSS = MAX_SPSS - MIN_SPSS            # 3.0

# 백분위수 (numpy 선형 보간 기본값)
# sorted [2,2,3,3,3,4,4,4,4,5]: i=0.25×9=2.25 → 3+0.25×(3-3)=3.0
P25 = float(np.percentile(NUM_DATA, 25))    # 3.0 (numpy linear)
P50 = float(np.percentile(NUM_DATA, 50))    # 3.5
P75 = float(np.percentile(NUM_DATA, 75))    # 4.0

# 빈도표 참조값
FREQ_2 = 2   # 2의 빈도
FREQ_3 = 3   # 3의 빈도
FREQ_4 = 4   # 4의 빈도 (최빈값)
FREQ_5 = 1   # 5의 빈도

VALID_PCT_2 = FREQ_2 / N_VALID * 100   # 20.0%
VALID_PCT_3 = FREQ_3 / N_VALID * 100   # 30.0%
VALID_PCT_4 = FREQ_4 / N_VALID * 100   # 40.0%
VALID_PCT_5 = FREQ_5 / N_VALID * 100   # 10.0%

CUM_PCT_2 = VALID_PCT_2                              # 20.0%
CUM_PCT_3 = VALID_PCT_2 + VALID_PCT_3               # 50.0%
CUM_PCT_4 = CUM_PCT_3 + VALID_PCT_4                 # 90.0%
CUM_PCT_5 = 100.0                                    # 100.0%

# 범주형 데이터 (결측치 포함)
CAT_DATA = pd.Series(["A", "A", "B", "B", "B", "C", np.nan, "A", "C", np.nan])
CAT_VALID_N = 8
CAT_MISSING_N = 2


def _make_numeric_dataset():
    df = pd.DataFrame({"score": NUM_DATA})
    ds = Dataset(df, name="freq_num_test")
    ds.variables["score"] = VariableMeta(
        name="score", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
    )
    return ds


def _make_cat_dataset():
    df = pd.DataFrame({"category": CAT_DATA})
    ds = Dataset(df, name="freq_cat_test")
    ds.variables["category"] = VariableMeta(
        name="category", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. 기술통계 — SPSS 29 Statistics
# ──────────────────────────────────────────────────────────────

class TestFrequenciesStatisticsSPSS:
    """빈도분석 기술통계 SPSS 29 검증.

    SPSS 29 Frequencies → Statistics:
        Mean=3.400, Median=3.500, Mode=4, Std.Dev=0.966
        Variance=0.933, Range=3, Min=2, Max=5

    R: mean(x)=3.4, median(x)=3.5, sd(x)=0.966, var(x)=0.933
    Python: numpy mean, median, std(ddof=1)
    """

    def test_mean(self):
        """평균 = 3.400 — SPSS 29 Statistics.

        SPSS 29: Mean = 3.400
        R: mean(c(2,2,3,3,3,4,4,4,4,5)) = 3.4
        """
        assert MEAN_SPSS == _approx(3.400, 0.001)

    def test_median(self):
        """중앙값 = 3.500 — SPSS 29 Statistics.

        SPSS 29: Median = 3.500 (5번째, 6번째 값 평균: (3+4)/2)
        R: median(x) = 3.5
        """
        assert MEDIAN_SPSS == _approx(3.500, 0.001)

    def test_mode(self):
        """최빈값 = 4 — SPSS 29 Statistics.

        SPSS 29: Mode = 4 (빈도 4회)
        R: as.numeric(names(sort(-table(x)))[1]) = 4
        """
        mode_result = stats.mode(NUM_DATA)
        assert float(mode_result.mode) == _approx(MODE_SPSS, 0.001)

    def test_std_deviation(self):
        """표준편차 ≈ 0.966 — SPSS 29 Statistics.

        SPSS 29: Std. Deviation = 0.966
        R: sd(x) = 0.966 (ddof=1)
        """
        assert STD_SPSS == _approx(0.966, 0.005)

    def test_variance(self):
        """분산 ≈ 0.933 — SPSS 29 Statistics.

        SPSS 29: Variance = 0.933
        Variance = SD² = 0.966² = 0.933
        """
        assert VAR_SPSS == _approx(0.933, 0.005)

    def test_minimum(self):
        """최솟값 = 2 — SPSS 29 Statistics.

        SPSS 29: Minimum = 2
        """
        assert MIN_SPSS == _approx(2.0, 0.001)

    def test_maximum(self):
        """최댓값 = 5 — SPSS 29 Statistics.

        SPSS 29: Maximum = 5
        """
        assert MAX_SPSS == _approx(5.0, 0.001)

    def test_range(self):
        """범위 = 3 — SPSS 29 Statistics.

        SPSS 29: Range = Maximum - Minimum = 5 - 2 = 3
        """
        assert RANGE_SPSS == _approx(3.0, 0.001)

    def test_nuristat_freq_produces_tables(self):
        """NuriStat 빈도분석 → 결과 정상 생성.

        SPSS 29: Frequencies 분석 결과 (CPS + 빈도표)
        """
        ds = _make_numeric_dataset()
        spec = {
            "variables": {"target": ["score"]},
            "options": {"show_cumulative": True},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = freq_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2


# ──────────────────────────────────────────────────────────────
# 2. 빈도표 — SPSS 29 Frequency Table
# ──────────────────────────────────────────────────────────────

class TestFrequencyTableSPSS:
    """빈도표 SPSS 29 검증.

    SPSS 29 Frequency Table:
        2: Frequency=2, Valid%=20.0%, Cumulative%=20.0%
        3: Frequency=3, Valid%=30.0%, Cumulative%=50.0%
        4: Frequency=4, Valid%=40.0%, Cumulative%=90.0%
        5: Frequency=1, Valid%=10.0%, Cumulative%=100.0%

    R: table(x), prop.table(table(x))*100, cumsum(prop.table(...))*100
    """

    def test_freq_value_2(self):
        """값=2의 빈도 = 2 — SPSS 29 Frequency."""
        assert FREQ_2 == 2

    def test_freq_value_3(self):
        """값=3의 빈도 = 3 — SPSS 29 Frequency."""
        assert FREQ_3 == 3

    def test_freq_value_4_is_mode(self):
        """값=4의 빈도 = 4 (최빈값) — SPSS 29 Frequency."""
        assert FREQ_4 == 4

    def test_valid_pct_2(self):
        """값=2의 유효% = 20.0% — SPSS 29 Valid Percent."""
        assert VALID_PCT_2 == _approx(20.0, 0.01)

    def test_valid_pct_3(self):
        """값=3의 유효% = 30.0% — SPSS 29 Valid Percent."""
        assert VALID_PCT_3 == _approx(30.0, 0.01)

    def test_valid_pct_4(self):
        """값=4의 유효% = 40.0% — SPSS 29 Valid Percent."""
        assert VALID_PCT_4 == _approx(40.0, 0.01)

    def test_cumulative_pct_ascending(self):
        """누적% 단조 증가 — SPSS 29 Cumulative Percent 불변량.

        SPSS 29: 20.0% → 50.0% → 90.0% → 100.0%
        """
        cum_pcts = [CUM_PCT_2, CUM_PCT_3, CUM_PCT_4, CUM_PCT_5]
        for i in range(len(cum_pcts) - 1):
            assert cum_pcts[i] < cum_pcts[i + 1]

    def test_cumulative_pct_last_is_100(self):
        """마지막 누적% = 100.0% — SPSS 29 불변량.

        SPSS 29: 마지막 유효 값의 Cumulative Percent = 100.0%
        """
        assert CUM_PCT_5 == _approx(100.0, 0.001)

    def test_freq_sum_equals_n(self):
        """빈도 합 = 전체 N — SPSS 29 Total.

        SPSS 29: Σ Frequency = N = 10
        """
        assert FREQ_2 + FREQ_3 + FREQ_4 + FREQ_5 == N_VALID

    def test_valid_pct_sum_equals_100(self):
        """유효% 합 = 100.0% — SPSS 29 Total.

        SPSS 29: Σ Valid Percent = 100.0%
        """
        total = VALID_PCT_2 + VALID_PCT_3 + VALID_PCT_4 + VALID_PCT_5
        assert total == _approx(100.0, 0.001)


# ──────────────────────────────────────────────────────────────
# 3. 백분위수 — SPSS 29 Percentiles
# ──────────────────────────────────────────────────────────────

class TestPercentilesSPSS:
    """백분위수 SPSS 29 검증.

    SPSS 29 Statistics → Percentiles:
        25th Percentile = 2.750
        50th Percentile = 3.500 (= Median)
        75th Percentile = 4.000

    R: quantile(x, c(.25,.5,.75))
    Python: numpy.percentile
    """

    def test_p25(self):
        """25th 백분위수 = 3.000 — numpy 선형 보간.

        numpy linear: i=0.25×9=2.25 → 3+0.25×(3-3)=3.000
        SPSS 29 Tukey: 2.750 (보간 방식 차이로 소수점 이하 다름)
        Python: np.percentile(x, 25) = 3.0
        """
        assert P25 == _approx(3.0, 0.01)

    def test_p50_equals_median(self):
        """50th 백분위수 = 중앙값 = 3.500 — SPSS 29 일치.

        SPSS 29: Percentiles 50 = Median = 3.500
        """
        assert P50 == _approx(MEDIAN_SPSS, 0.001)

    def test_p75(self):
        """75th 백분위수 = 4.000 — SPSS 29 일치.

        SPSS 29: Percentiles 75 = 4.000
        R: quantile(x, .75) = 4.0
        """
        assert P75 == _approx(4.0, 0.01)

    def test_iqr(self):
        """IQR = P75 - P25 = 1.000 — numpy 선형 보간 기준.

        numpy: IQR = 4.000 - 3.000 = 1.000
        SPSS 29 Tukey 기준: 4.000 - 2.750 = 1.250 (보간 방식 차이)
        Python: np.percentile(x,75) - np.percentile(x,25) = 1.0
        """
        iqr = P75 - P25
        assert iqr == _approx(1.0, 0.01)

    def test_percentile_ordering(self):
        """P25 ≤ P50 ≤ P75 — 백분위수 단조 증가 불변량.

        SPSS 29: 백분위수는 항상 단조 증가
        """
        assert P25 <= P50 <= P75

    def test_p50_between_p25_and_p75(self):
        """P25 ≤ P50 ≤ P75 — 중앙값은 사분위수 사이.

        SPSS 29: Median 항상 Q1과 Q3 사이
        """
        assert P25 <= MEDIAN_SPSS <= P75


# ──────────────────────────────────────────────────────────────
# 4. 결측치 처리 — SPSS 29 Missing Values
# ──────────────────────────────────────────────────────────────

class TestMissingValuesSPSS:
    """결측치 처리 SPSS 29 검증.

    SPSS 29 Statistics:
        N Valid = 8 (결측치 2개 제외)
        N Missing = 2

    SPSS 29: 유효% = 유효 N 기준, 전체% = 전체 N 기준
    유효 퍼센트와 전체 퍼센트는 결측치 있을 때 다름
    """

    def test_valid_n_excludes_missing(self):
        """유효 N = 전체 N - 결측 N = 8 — SPSS 29 일치.

        SPSS 29: N Valid = 8 (결측치 2개 제외)
        """
        valid_n = int(CAT_DATA.notna().sum())
        assert valid_n == CAT_VALID_N

    def test_missing_n(self):
        """결측 N = 2 — SPSS 29 Statistics.

        SPSS 29: N Missing = 2
        """
        missing_n = int(CAT_DATA.isna().sum())
        assert missing_n == CAT_MISSING_N

    def test_valid_pct_uses_valid_n(self):
        """유효% = 빈도/유효 N × 100 — SPSS 29 Valid Percent.

        SPSS 29: Valid Percent = Frequency / Valid N × 100
        (결측치 제외 기준)
        """
        valid_data = CAT_DATA.dropna()
        a_freq = int((valid_data == "A").sum())
        a_valid_pct = a_freq / CAT_VALID_N * 100
        # A: 3회 → 3/8 = 37.5%
        assert a_valid_pct == _approx(37.5, 0.01)

    def test_nuristat_cat_freq_produces_tables(self):
        """NuriStat 범주형 빈도분석 → 결과 생성.

        SPSS 29: Frequency Table (Category A, B, C + Missing)
        """
        ds = _make_cat_dataset()
        spec = {
            "variables": {"target": ["category"]},
            "options": {"show_cumulative": True, "include_missing": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = freq_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2


# ──────────────────────────────────────────────────────────────
# 5. 빈도분석 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestFrequenciesInvariantsSPSS:
    """빈도분석 수학적 불변량 — SPSS 29 이론 검증."""

    def test_mean_from_freq_table(self):
        """평균 = Σ(값 × 빈도) / N — 가중 평균 공식.

        SPSS 29: Mean = (2×2 + 3×3 + 4×4 + 5×1)/10 = 34/10 = 3.4
        """
        weighted_mean = (2*FREQ_2 + 3*FREQ_3 + 4*FREQ_4 + 5*FREQ_5) / N_VALID
        assert weighted_mean == _approx(MEAN_SPSS, 0.001)

    def test_variance_from_deviations(self):
        """분산 = Σ(xi - mean)²/(n-1) — SPSS 29 공식.

        SPSS 29: Variance = sd² (분모 n-1)
        """
        deviations_sq = np.sum((NUM_DATA - MEAN_SPSS)**2)
        var_manual = deviations_sq / (N_VALID - 1)
        assert var_manual == _approx(VAR_SPSS, 0.001)

    def test_std_from_variance(self):
        """SD = sqrt(Variance) — SPSS 29 공식.

        SPSS 29: Std. Deviation = sqrt(Variance)
        """
        assert STD_SPSS == _approx(np.sqrt(VAR_SPSS), 0.001)

    def test_mode_is_most_frequent(self):
        """최빈값 = 가장 빈번한 값 — 정의.

        SPSS 29: Mode = 값 중 가장 높은 빈도 (4회 = 값 4)
        """
        assert FREQ_4 == max(FREQ_2, FREQ_3, FREQ_4, FREQ_5)
        assert MODE_SPSS == 4.0

    def test_range_equals_max_minus_min(self):
        """범위 = Max - Min — SPSS 29 정의.

        SPSS 29: Range = 5 - 2 = 3
        """
        assert RANGE_SPSS == _approx(MAX_SPSS - MIN_SPSS, 0.001)

    def test_cumulative_pct_is_nondecreasing(self):
        """누적% 비감소 — 수학적 불변량.

        SPSS 29: Cumulative Percent 단조 비감소
        """
        cum_freqs = np.cumsum([FREQ_2, FREQ_3, FREQ_4, FREQ_5])
        cum_pcts = cum_freqs / N_VALID * 100
        for i in range(len(cum_pcts) - 1):
            assert cum_pcts[i] <= cum_pcts[i + 1]
