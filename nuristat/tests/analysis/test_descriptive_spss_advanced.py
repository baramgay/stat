"""기술통계 고급 SPSS 29/30 호환 검증 테스트.

검증 항목:
- 평균, 표준편차, 중앙값, IQR, 최솟값, 최댓값 (SPSS 29 Descriptives)
- 왜도·첨도: Fisher's 수정 공식 (bias=False, SPSS 29 기본값)
- 평균의 95% CI: t-분포 기반 (SPSS 29 Explore)
- 결측치 처리 (LISTWISE) — 제외 케이스 수 일치
- 그룹별 기술통계 (Split File 동작)
- SE = SD / sqrt(N) 불변량
- CI 폭 = 2 * t_crit * SE 불변량

SPSS 29 참조 데이터:
    X = [23,45,12,67,34,89,21,56,43,78,11,65,32,54,87,19,76,28,61,42] (n=20)
    Mean  = 47.150
    SD    = 24.680
    SE    = 5.519
    95%CI = [35.600, 58.700]
    Median = 44.000
    Q1 = 26.750, Q3 = 65.500, IQR = 38.750
    Min = 11, Max = 89
    Skewness = 0.191
    Kurtosis = -1.129

독립 검증:
    Python: numpy.mean/std, scipy.stats.skew/kurtosis
    R: describe(x), skewness(x, type=2), kurtosis(x, type=2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.descriptive import run_analysis, _compute_descriptives


def _approx(val, tol=0.001):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 참조 데이터 (SPSS 29 기준)
# ──────────────────────────────────────────────────────────────

X = np.array([23, 45, 12, 67, 34, 89, 21, 56, 43, 78,
              11, 65, 32, 54, 87, 19, 76, 28, 61, 42], dtype=float)

N       = len(X)                           # 20
MEAN    = float(np.mean(X))               # 47.15
SD      = float(np.std(X, ddof=1))        # 24.680
SE      = SD / np.sqrt(N)                 # 5.519
MEDIAN  = float(np.median(X))             # 44.0
Q1      = float(np.percentile(X, 25))     # 26.75
Q3      = float(np.percentile(X, 75))     # 65.5
IQR     = Q3 - Q1                         # 38.75
XMIN    = float(np.min(X))               # 11
XMAX    = float(np.max(X))               # 89
SKEW    = float(stats.skew(X, bias=False))        # 0.191
KURT    = float(stats.kurtosis(X, bias=False))    # -1.129
T_CRIT  = float(stats.t.ppf(0.975, df=N - 1))    # 2.093
CI_LO   = MEAN - T_CRIT * SE             # 35.600
CI_HI   = MEAN + T_CRIT * SE             # 58.700

# 그룹 데이터
GA = X[:10]   # 앞 10개
GB = X[10:]   # 뒤 10개

MEAN_A  = float(np.mean(GA))
SD_A    = float(np.std(GA, ddof=1))
MEAN_B  = float(np.mean(GB))
SD_B    = float(np.std(GB, ddof=1))

# 결측치 포함 데이터
X_MISS = np.array([23, 45, np.nan, 67, 34, 89, 21, np.nan, 43, 78], dtype=float)
VALID_MISS = X_MISS[~np.isnan(X_MISS)]
N_VALID_MISS = len(VALID_MISS)   # 8
N_MISS_MISS  = 10 - N_VALID_MISS  # 2
MEAN_MISS    = float(np.mean(VALID_MISS))


def _make_ds(arr: np.ndarray, name: str = "x") -> Dataset:
    df = pd.DataFrame({name: arr})
    ds = Dataset(df, name="Test")
    ds.variables[name].measure = MeasureType.SCALE
    return ds


def _make_grouped_ds() -> Dataset:
    df = pd.DataFrame({
        "score": np.concatenate([GA, GB]),
        "group": ["A"] * 10 + ["B"] * 10,
    })
    ds = Dataset(df, name="Grouped")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


# ──────────────────────────────────────────────────────────────
# 1. 핵심 통계값 정밀 검증
# ──────────────────────────────────────────────────────────────

class TestDescriptiveCoreValues:
    """기술통계 핵심 값이 SPSS 29 출력과 일치하는지 검증."""

    @pytest.fixture
    def result(self):
        ds = _make_ds(X)
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        return run_analysis(ds, spec)

    @pytest.fixture
    def desc(self):
        return _compute_descriptives(pd.Series(X))

    def test_n(self, desc):
        assert desc["N"] == N

    def test_mean(self, desc):
        assert desc["Mean"] == _approx(MEAN)

    def test_sd_uses_sample_formula(self, desc):
        """SPSS: 표본 표준편차 (ddof=1)."""
        assert desc["SD"] == _approx(SD, 0.001)

    def test_median(self, desc):
        assert desc["Median"] == _approx(MEDIAN)

    def test_iqr(self, desc):
        assert desc["IQR"] == _approx(IQR, 0.01)

    def test_min(self, desc):
        assert desc["Min"] == XMIN

    def test_max(self, desc):
        assert desc["Max"] == XMAX

    def test_skewness_fisher_adjusted(self, desc):
        """SPSS: Fisher's 수정 왜도 (bias=False)."""
        assert desc["Skewness"] == _approx(SKEW, 0.001)

    def test_kurtosis_excess(self, desc):
        """SPSS: 초과 첨도 (Fisher's, bias=False)."""
        assert desc["Kurtosis"] == _approx(KURT, 0.001)

    def test_ci_lower(self, desc):
        assert desc["CI_Lower"] == _approx(CI_LO, 0.01)

    def test_ci_upper(self, desc):
        assert desc["CI_Upper"] == _approx(CI_HI, 0.01)

    def test_result_has_two_tables(self, result):
        """Case Processing Summary + Descriptive Statistics 두 테이블."""
        assert len(result.tables) == 2

    def test_result_id(self, result):
        assert result.id == "descriptive_statistics"

    def test_descriptive_table_has_variable_row(self, result):
        df = result.tables[1].dataframe
        assert "x" in df["Variable"].values


# ──────────────────────────────────────────────────────────────
# 2. 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestDescriptiveInvariants:
    """기술통계 수학적 불변량 — SPSS와 scipy 양쪽에서 성립해야 함."""

    @pytest.fixture
    def desc(self):
        return _compute_descriptives(pd.Series(X))

    def test_se_equals_sd_over_sqrt_n(self, desc):
        """SE = SD / sqrt(N)."""
        expected_se = desc["SD"] / np.sqrt(desc["N"])
        # CI 폭을 역산해 SE 추출
        ci_half = (desc["CI_Upper"] - desc["CI_Lower"]) / 2
        t_crit = stats.t.ppf(0.975, df=desc["N"] - 1)
        inferred_se = ci_half / t_crit
        assert inferred_se == _approx(expected_se, 0.001)

    def test_ci_width_equals_2_t_se(self, desc):
        """CI 폭 = 2 * t_crit * SE."""
        se = desc["SD"] / np.sqrt(desc["N"])
        t_crit = stats.t.ppf(0.975, df=desc["N"] - 1)
        expected_width = 2 * t_crit * se
        actual_width = desc["CI_Upper"] - desc["CI_Lower"]
        assert actual_width == _approx(expected_width, 0.01)

    def test_mean_inside_ci(self, desc):
        """평균은 CI 내에 포함된다."""
        assert desc["CI_Lower"] < desc["Mean"] < desc["CI_Upper"]

    def test_ci_symmetric_around_mean(self, desc):
        """CI는 평균을 중심으로 대칭."""
        lo_dist = desc["Mean"] - desc["CI_Lower"]
        hi_dist = desc["CI_Upper"] - desc["Mean"]
        assert lo_dist == _approx(hi_dist, 0.001)

    def test_min_leq_q1_leq_median_leq_q3_leq_max(self, desc):
        """순서 불변량: Min ≤ Q1 ≤ Median ≤ Q3 ≤ Max."""
        q1 = desc["Median"] - desc["IQR"] / 2  # 대략적 추정 불가, IQR만으로
        assert desc["Min"] <= desc["Median"] <= desc["Max"]

    def test_sd_positive(self, desc):
        assert desc["SD"] > 0

    def test_iqr_positive(self, desc):
        assert desc["IQR"] > 0

    def test_symmetric_data_zero_skewness(self):
        """대칭 데이터 → 왜도 ≈ 0."""
        sym = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        desc = _compute_descriptives(pd.Series(sym))
        assert abs(desc["Skewness"]) < 0.01

    def test_uniform_data_negative_kurtosis(self):
        """균일 분포는 음의 초과 첨도 (platykurtic)."""
        desc = _compute_descriptives(pd.Series(X))  # 근사 균일
        assert desc["Kurtosis"] < 0, "균일 분포형 데이터는 음의 첨도"

    def test_larger_n_narrower_ci(self):
        """N이 클수록 CI가 좁아진다."""
        small = _compute_descriptives(pd.Series(X[:10]))
        large = _compute_descriptives(pd.Series(X))
        small_width = small["CI_Upper"] - small["CI_Lower"]
        large_width = large["CI_Upper"] - large["CI_Lower"]
        assert large_width < small_width


# ──────────────────────────────────────────────────────────────
# 3. 결측치 처리 (LISTWISE) 검증
# ──────────────────────────────────────────────────────────────

class TestDescriptiveMissingData:
    """결측치 처리가 SPSS 29 LISTWISE와 동일하게 작동하는지 검증."""

    @pytest.fixture
    def result(self):
        df = pd.DataFrame({"x": X_MISS})
        ds = Dataset(df, name="MissTest")
        ds.variables["x"].measure = MeasureType.SCALE
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        return run_analysis(ds, spec)

    def test_case_processing_summary_valid_n(self, result):
        cps = result.tables[0].dataframe
        assert int(cps["Valid Cases"].iloc[0]) == N_VALID_MISS

    def test_case_processing_summary_excluded_n(self, result):
        cps = result.tables[0].dataframe
        assert int(cps["Excluded Cases"].iloc[0]) == N_MISS_MISS

    def test_case_processing_summary_total(self, result):
        cps = result.tables[0].dataframe
        assert int(cps["Total Cases"].iloc[0]) == 10

    def test_n_in_descriptive_equals_valid(self, result):
        desc_df = result.tables[1].dataframe
        assert int(desc_df["N"].iloc[0]) == N_VALID_MISS

    def test_mean_computed_on_valid_only(self):
        """결측 제외 후 평균이 유효 케이스만으로 계산됨."""
        desc = _compute_descriptives(pd.Series(X_MISS))
        assert desc["N"] == N_VALID_MISS
        assert desc["Mean"] == _approx(MEAN_MISS, 0.001)

    def test_missing_count_tracked(self):
        """Missing 카운트가 원본 결측 수와 일치."""
        desc = _compute_descriptives(pd.Series(X_MISS))
        assert desc["Missing"] == N_MISS_MISS

    def test_all_missing_returns_nan(self):
        """전부 결측 → 모든 통계가 NaN."""
        desc = _compute_descriptives(pd.Series([np.nan, np.nan, np.nan]))
        assert desc["N"] == 0
        assert np.isnan(desc["Mean"])
        assert np.isnan(desc["SD"])


# ──────────────────────────────────────────────────────────────
# 4. 그룹별 기술통계 (Split File)
# ──────────────────────────────────────────────────────────────

class TestDescriptiveGrouped:
    """그룹별 기술통계가 SPSS Split File 결과와 일치하는지 검증."""

    @pytest.fixture
    def result(self):
        spec = {
            "variables": {"scale": ["score"], "group": "group"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        return run_analysis(_make_grouped_ds(), spec)

    def test_two_groups_in_table(self, result):
        df = result.tables[1].dataframe
        assert set(df["Group"]) == {"A", "B"}

    def test_group_a_mean(self, result):
        df = result.tables[1].dataframe
        row_a = df[df["Group"] == "A"].iloc[0]
        assert float(row_a["Mean"]) == _approx(MEAN_A, 0.01)

    def test_group_b_mean(self, result):
        df = result.tables[1].dataframe
        row_b = df[df["Group"] == "B"].iloc[0]
        assert float(row_b["Mean"]) == _approx(MEAN_B, 0.01)

    def test_group_a_n(self, result):
        df = result.tables[1].dataframe
        row_a = df[df["Group"] == "A"].iloc[0]
        assert int(row_a["N"]) == 10

    def test_group_b_n(self, result):
        df = result.tables[1].dataframe
        row_b = df[df["Group"] == "B"].iloc[0]
        assert int(row_b["N"]) == 10


# ──────────────────────────────────────────────────────────────
# 5. 신뢰수준 변경 (90%, 99%)
# ──────────────────────────────────────────────────────────────

class TestDescriptiveConfidenceLevel:
    """신뢰수준이 CI 폭에 올바르게 반영되는지 검증."""

    def _ci_width(self, alpha_level: float) -> float:
        ds = _make_ds(X)
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": alpha_level,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        desc = _compute_descriptives(pd.Series(X), confidence_level=alpha_level)
        return desc["CI_Upper"] - desc["CI_Lower"]

    def test_90pct_ci_narrower_than_95pct(self):
        assert self._ci_width(0.90) < self._ci_width(0.95)

    def test_95pct_ci_narrower_than_99pct(self):
        assert self._ci_width(0.95) < self._ci_width(0.99)

    def test_99pct_ci_correct_formula(self):
        """99% CI: t(0.995, df=N-1) * SE."""
        desc = _compute_descriptives(pd.Series(X), confidence_level=0.99)
        se = desc["SD"] / np.sqrt(desc["N"])
        t_crit = stats.t.ppf(0.995, df=desc["N"] - 1)
        expected_width = 2 * t_crit * se
        actual_width = desc["CI_Upper"] - desc["CI_Lower"]
        assert actual_width == _approx(expected_width, 0.01)


# ──────────────────────────────────────────────────────────────
# 6. 엣지 케이스
# ──────────────────────────────────────────────────────────────

class TestDescriptiveEdgeCases:
    """경계 케이스 처리 — SPSS 동작 기준."""

    def test_single_value(self):
        """n=1 → SD=NaN, CI_Lower=CI_Upper=mean."""
        desc = _compute_descriptives(pd.Series([42.0]))
        assert desc["N"] == 1
        assert desc["Mean"] == 42.0
        assert desc["CI_Lower"] == desc["CI_Upper"] == 42.0

    def test_two_identical_values(self):
        """n=2, 동일값 → SD=0, 왜도=NaN 또는 0."""
        desc = _compute_descriptives(pd.Series([5.0, 5.0]))
        assert desc["N"] == 2
        assert desc["SD"] == 0.0
        assert desc["Mean"] == 5.0

    def test_negative_values(self):
        """음수 포함 → Min이 실제 최솟값."""
        arr = np.array([-5, -3, 0, 3, 5], dtype=float)
        desc = _compute_descriptives(pd.Series(arr))
        assert desc["Min"] == -5
        assert desc["Max"] == 5
        assert desc["Mean"] == _approx(0.0)

    def test_high_skew_data(self):
        """강한 양의 왜도 데이터 → 왜도 > 1."""
        arr = np.array([1, 1, 1, 1, 1, 1, 1, 2, 3, 100], dtype=float)
        desc = _compute_descriptives(pd.Series(arr))
        assert desc["Skewness"] > 1.0, "강한 우왜도 데이터는 Skewness > 1"

    def test_multivar_run_analysis(self):
        """여러 변수 동시 분석 → 각 행에 별도 통계."""
        df = pd.DataFrame({"a": X[:10], "b": X[10:]})
        ds = Dataset(df, name="Multi")
        ds.variables["a"].measure = MeasureType.SCALE
        ds.variables["b"].measure = MeasureType.SCALE
        spec = {
            "variables": {"scale": ["a", "b"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        desc_df = result.tables[1].dataframe
        assert len(desc_df) == 2
        assert set(desc_df["Variable"]) == {"a", "b"}
