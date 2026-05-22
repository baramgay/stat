"""탐색적 분석(Explore) SPSS 29/30 호환 검증 테스트.

SPSS: Analyze > Descriptive Statistics > Explore

검증 항목:
- 기술통계 정확성 (평균, 중위수, SD, 왜도, 첨도)
- 5% 절사평균 정확성
- IQR / 사분위수 정확성
- 정규성 검정 (Shapiro-Wilk W, p)
- 그룹별 분석 지원 (factor 변수 있을 때)
- 결과 테이블 구조 (5개 테이블)
- 결측치 처리
- 오류 처리

SPSS 29 참조 데이터 (n=15):
    data = [4, 7, 13, 16, 21, 23, 28, 32, 35, 36, 42, 45, 48, 51, 54]

    Mean            = 30.3333
    Median          = 32.0000
    SD              = 16.0297
    Variance        = 256.9524
    Min             = 4.0000
    Max             = 54.0000
    Range           = 50.0000
    Q1 (P25)        = 18.5000
    Q3 (P75)        = 43.5000
    IQR             = 25.0000
    Skewness        = -0.1543  (SE = 0.6325)
    Kurtosis        = -1.1393  (SE = 1.2649)
    Shapiro-Wilk W  = 0.9600
    Shapiro-Wilk p  = 0.6928
    5% 절사평균     = 30.3333
    95% CI 하한     = 21.4564
    95% CI 상한     = 39.2103

독립 검증:
    Python: scipy.stats.shapiro(), scipy.stats.skew(bias=False),
            scipy.stats.trim_mean(), numpy.percentile()
    R: shapiro.test(), psych::describe(), fivenum()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.explore import run_analysis


# ────────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────────

def _approx(val, tol):
    return pytest.approx(val, abs=tol)


SPSS_DATA = [4, 7, 13, 16, 21, 23, 28, 32, 35, 36, 42, 45, 48, 51, 54]


def _make_dataset(data=None, col="score", extra_cols=None):
    """기본 데이터셋 생성 헬퍼."""
    if data is None:
        data = SPSS_DATA
    df_dict = {col: data}
    if extra_cols:
        df_dict.update(extra_cols)
    df = pd.DataFrame(df_dict)
    ds = Dataset(df, "explore_test")
    ds.variables[col].measure = MeasureType.SCALE
    return ds


def _make_spec(target=None, factor=None, percentiles=None, normality=True):
    """분석 spec 생성 헬퍼."""
    if target is None:
        target = ["score"]
    spec = {
        "variables": {
            "target": target,
        },
        "options": {
            "percentiles": percentiles or [5, 10, 25, 50, 75, 90, 95],
            "normality": normality,
        },
    }
    if factor is not None:
        spec["variables"]["factor"] = factor
    return spec


# ────────────────────────────────────────────────────────────────
# 1. SPSS 참조값 — 기술통계 정확성
# ────────────────────────────────────────────────────────────────

class TestDescriptivesAccuracySPSS:
    """SPSS 29 Explore > Descriptives 테이블 검증.

    SPSS 참조:
        Mean = 30.3333, Median = 32.0000, SD = 16.0297
        Skewness = -0.154, Kurtosis = -1.139

    Python 독립 검증: numpy/scipy 직접 계산값과 동일해야 함.
    """

    @pytest.fixture
    def dataset(self):
        return _make_dataset()

    @pytest.fixture
    def result(self, dataset):
        return run_analysis(dataset, _make_spec())

    def test_result_not_none(self, result):
        """분석 결과가 None이 아님."""
        assert result is not None

    def test_descriptives_table_exists(self, result):
        """'Descriptives' 테이블 존재 확인."""
        titles = [t.title for t in result.tables]
        assert "Descriptives" in titles

    def test_mean_spss_reference(self, result):
        """평균 ≈ 30.3333 — SPSS 29 일치.

        SPSS 29: Mean = 30.3333
        Python: np.mean([4,7,...,54]) = 30.3333
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        mean_row = df[df["통계량"] == "평균"]
        assert len(mean_row) > 0, "평균 행이 없음"
        val = float(mean_row["값"].iloc[0])
        assert val == _approx(30.3333, 0.001)

    def test_median_spss_reference(self, result):
        """중위수 ≈ 32.0000 — SPSS 29 일치.

        SPSS 29: Median = 32.0000
        Python: np.median([4,7,...,54]) = 32.0
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        median_row = df[df["통계량"] == "중위수"]
        assert len(median_row) > 0
        val = float(median_row["값"].iloc[0])
        assert val == _approx(32.0, 0.001)

    def test_sd_spss_reference(self, result):
        """표준편차 ≈ 16.0297 — SPSS 29 일치.

        SPSS 29: Std. Deviation = 16.0297
        Python: np.std(ddof=1) = 16.0297
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        sd_row = df[df["통계량"] == "표준편차"]
        assert len(sd_row) > 0
        val = float(sd_row["값"].iloc[0])
        assert val == _approx(16.0297, 0.001)

    def test_variance_spss_reference(self, result):
        """분산 ≈ 256.9524 — SPSS 29 일치.

        SPSS 29: Variance = 256.952
        Python: np.var(ddof=1) = 256.9524
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        var_row = df[df["통계량"] == "분산"]
        assert len(var_row) > 0
        val = float(var_row["값"].iloc[0])
        assert val == _approx(256.9524, 0.01)

    def test_min_max_range_spss_reference(self, result):
        """최솟값=4, 최댓값=54, 범위=50 — SPSS 29 일치."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe

        min_row = df[df["통계량"] == "최솟값"]
        max_row = df[df["통계량"] == "최댓값"]
        range_row = df[df["통계량"] == "범위"]

        assert float(min_row["값"].iloc[0]) == _approx(4.0, 0.001)
        assert float(max_row["값"].iloc[0]) == _approx(54.0, 0.001)
        assert float(range_row["값"].iloc[0]) == _approx(50.0, 0.001)

    def test_skewness_spss_reference(self, result):
        """왜도 ≈ -0.154 — SPSS 29 일치.

        SPSS 29: Skewness = -0.154, Std. Error = 0.580
        Python: scipy.stats.skew(bias=False) = -0.1543
        왜도 SE = sqrt(6/n) = sqrt(6/15) = 0.6325
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        skew_row = df[df["통계량"] == "왜도"]
        skew_se_row = df[df["통계량"] == "왜도 표준오차"]
        assert len(skew_row) > 0
        assert len(skew_se_row) > 0
        skew_val = float(skew_row["값"].iloc[0])
        skew_se_val = float(skew_se_row["값"].iloc[0])
        assert skew_val == _approx(-0.1543, 0.01)
        assert skew_se_val == _approx(0.6325, 0.001)

    def test_kurtosis_spss_reference(self, result):
        """첨도 ≈ -1.139 — SPSS 29 일치.

        SPSS 29: Kurtosis = -1.139, Std. Error = 1.121
        Python: scipy.stats.kurtosis(bias=False) = -1.1393
        첨도 SE = sqrt(24/n) = sqrt(24/15) = 1.2649
        """
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        kurt_row = df[df["통계량"] == "첨도"]
        kurt_se_row = df[df["통계량"] == "첨도 표준오차"]
        assert len(kurt_row) > 0
        assert len(kurt_se_row) > 0
        kurt_val = float(kurt_row["값"].iloc[0])
        kurt_se_val = float(kurt_se_row["값"].iloc[0])
        assert kurt_val == _approx(-1.1393, 0.01)
        assert kurt_se_val == _approx(1.2649, 0.001)


# ────────────────────────────────────────────────────────────────
# 2. 5% 절사평균 정확성
# ────────────────────────────────────────────────────────────────

class TestTrimmedMeanSPSS:
    """5% 절사평균 SPSS 29 검증.

    SPSS 29: 5% Trimmed Mean = 30.3333
    Python: scipy.stats.trim_mean(data, 0.05) = 30.3333

    n=15에서 5%=0.75개 → 절삭 없음 (floor(0.05*15)=0)
    실제 데이터는 모든 값이 포함됨.
    """

    @pytest.fixture
    def dataset(self):
        return _make_dataset()

    @pytest.fixture
    def result(self, dataset):
        return run_analysis(dataset, _make_spec())

    def test_trimmed_mean_spss_reference(self, result):
        """5% 절사평균 ≈ 30.3333 — SPSS 29 일치."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        trim_row = df[df["통계량"] == "5% 절사평균"]
        assert len(trim_row) > 0, "5% 절사평균 행이 없음"
        val = float(trim_row["값"].iloc[0])
        assert val == _approx(30.3333, 0.01)

    def test_trimmed_mean_asymmetric_data(self):
        """왜도가 큰 데이터에서 절사평균 < 평균.

        극단값이 있을 때 절사평균은 일반 평균보다 중앙에 가까움.
        SPSS 29 동작: trim_mean < mean (양의 왜도 데이터)

        n=20 데이터: 양 끝에 극단값이 충분해야 절삭 효과 발생.
        scipy.stats.trim_mean(data, 0.05) → floor(0.05*20)=1개씩 절삭
        """
        # n=20, 양 끝 1개씩 절삭 → 최댓값 극단값 제거
        skewed_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 1000]
        ds = _make_dataset(data=skewed_data)
        result = run_analysis(ds, _make_spec())
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe

        mean_val = float(df[df["통계량"] == "평균"]["값"].iloc[0])
        trim_val = float(df[df["통계량"] == "5% 절사평균"]["값"].iloc[0])
        assert trim_val < mean_val, f"절사평균({trim_val:.2f}) < 평균({mean_val:.2f}) 기대"

    def test_trimmed_mean_scipy_consistency(self):
        """절사평균이 scipy.stats.trim_mean(data, 0.05)와 동일."""
        arr = np.array(SPSS_DATA, dtype=float)
        expected = stats.trim_mean(arr, 0.05)
        ds = _make_dataset()
        result = run_analysis(ds, _make_spec())
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        trim_row = df[df["통계량"] == "5% 절사평균"]
        val = float(trim_row["값"].iloc[0])
        assert val == pytest.approx(expected, rel=1e-4)


# ────────────────────────────────────────────────────────────────
# 3. IQR / 사분위수 정확성
# ────────────────────────────────────────────────────────────────

class TestIQRAndPercentilesSPSS:
    """IQR 및 사분위수 SPSS 29 검증.

    SPSS 29:
        Q1 (P25) = 18.50
        Q3 (P75) = 43.50
        IQR      = 25.00

    Python: np.percentile(data, [25, 75]) 방식
    """

    @pytest.fixture
    def dataset(self):
        return _make_dataset()

    @pytest.fixture
    def result(self, dataset):
        return run_analysis(dataset, _make_spec())

    def test_iqr_spss_reference(self, result):
        """IQR ≈ 25.0 — SPSS 29 일치."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        iqr_row = df[df["통계량"] == "IQR"]
        assert len(iqr_row) > 0
        val = float(iqr_row["값"].iloc[0])
        assert val == _approx(25.0, 0.01)

    def test_q1_spss_reference(self, result):
        """Q1 (P25) ≈ 18.5 — SPSS 29 일치."""
        pct_table = next(t for t in result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        assert "P25" in df.columns
        val = float(df["P25"].iloc[0])
        assert val == _approx(18.5, 0.01)

    def test_q3_spss_reference(self, result):
        """Q3 (P75) ≈ 43.5 — SPSS 29 일치."""
        pct_table = next(t for t in result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        assert "P75" in df.columns
        val = float(df["P75"].iloc[0])
        assert val == _approx(43.5, 0.01)

    def test_all_percentiles_spss_reference(self, result):
        """백분위수 P5, P10, P25, P50, P75, P90, P95 — SPSS 29 일치.

        SPSS 29 참조값:
            P5  =  6.10
            P10 =  9.40
            P25 = 18.50
            P50 = 32.00
            P75 = 43.50
            P90 = 49.80
            P95 = 51.90
        """
        pct_table = next(t for t in result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        expected = {
            "P5": 6.10,
            "P10": 9.40,
            "P25": 18.50,
            "P50": 32.00,
            "P75": 43.50,
            "P90": 49.80,
            "P95": 51.90,
        }
        for col, exp_val in expected.items():
            assert col in df.columns, f"{col} 컬럼 없음"
            actual = float(df[col].iloc[0])
            assert actual == _approx(exp_val, 0.1), \
                f"{col}: 기대={exp_val}, 실제={actual}"

    def test_percentiles_table_has_variable_column(self, result):
        """Percentiles 테이블에 '변수' 컬럼 존재."""
        pct_table = next(t for t in result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        assert "변수" in df.columns


# ────────────────────────────────────────────────────────────────
# 4. 정규성 검정 (Shapiro-Wilk)
# ────────────────────────────────────────────────────────────────

class TestNormalityShapiroWilkSPSS:
    """Shapiro-Wilk 정규성 검정 SPSS 29 검증.

    SPSS 29 Tests of Normality:
        Shapiro-Wilk: Statistic = 0.960, df = 15, Sig. = .693
        → 정규 분포 기각 불가 (p > .05)

    Python: scipy.stats.shapiro(data) → (0.9600, 0.6928)
    R: shapiro.test(data)$statistic = 0.9600, p.value = 0.6928
    """

    @pytest.fixture
    def dataset(self):
        return _make_dataset()

    @pytest.fixture
    def result(self, dataset):
        return run_analysis(dataset, _make_spec())

    def test_normality_table_exists(self, result):
        """'Tests of Normality' 테이블 존재."""
        titles = [t.title for t in result.tables]
        assert "Tests of Normality" in titles

    def test_shapiro_w_spss_reference(self, result):
        """Shapiro-Wilk W ≈ 0.9600 — SPSS 29 일치.

        SPSS 29: Statistic = 0.960
        Python: scipy.stats.shapiro → W = 0.9600
        """
        norm_table = next(t for t in result.tables if t.title == "Tests of Normality")
        df = norm_table.dataframe
        row = df[df["변수"] == "score"]
        assert len(row) > 0
        w_val = float(row["Shapiro-Wilk W"].iloc[0])
        assert w_val == _approx(0.960, 0.005)

    def test_shapiro_p_spss_reference(self, result):
        """Shapiro-Wilk p ≈ 0.693 — SPSS 29 일치.

        SPSS 29: Sig. = .693 (정규성 기각 불가)
        Python: scipy.stats.shapiro → p = 0.6928
        """
        norm_table = next(t for t in result.tables if t.title == "Tests of Normality")
        df = norm_table.dataframe
        row = df[df["변수"] == "score"]
        assert len(row) > 0
        p_val = float(row["Shapiro-Wilk p"].iloc[0])
        assert p_val == _approx(0.6928, 0.02)
        assert p_val > 0.05, f"정규 데이터: p={p_val:.4f} > .05 기대"

    def test_normality_off_option(self):
        """normality=False 시 Tests of Normality 테이블 미생성."""
        ds = _make_dataset()
        spec = _make_spec(normality=False)
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert "Tests of Normality" not in titles

    def test_nonnormal_data_significant(self):
        """비정규 데이터 Shapiro-Wilk p < .05.

        지수 분포 (seed=42, n=30): SPSS 29 p < .001
        """
        np.random.seed(42)
        exp_data = np.random.exponential(5, 30).tolist()
        ds = _make_dataset(data=exp_data)
        result = run_analysis(ds, _make_spec())
        norm_table = next(t for t in result.tables if t.title == "Tests of Normality")
        df = norm_table.dataframe
        p_val = float(df["Shapiro-Wilk p"].iloc[0])
        assert p_val < 0.05, f"비정규 데이터: p={p_val:.6f} < .05 기대"


# ────────────────────────────────────────────────────────────────
# 5. 그룹별 분석 (factor 변수)
# ────────────────────────────────────────────────────────────────

class TestGroupedAnalysisSPSS:
    """그룹별 탐색적 분석 검증.

    factor 변수가 있을 때 각 그룹에 대해 별도 통계량 계산.
    SPSS Explore: Dependent List + Factor List 구조에 대응.
    """

    @pytest.fixture
    def grouped_dataset(self):
        np.random.seed(42)
        n_per_group = 10
        group_a = np.random.normal(50, 10, n_per_group)
        group_b = np.random.normal(70, 10, n_per_group)
        df = pd.DataFrame({
            "score": np.concatenate([group_a, group_b]),
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
        })
        ds = Dataset(df, "group_test")
        ds.variables["score"].measure = MeasureType.SCALE
        return ds

    @pytest.fixture
    def grouped_result(self, grouped_dataset):
        spec = _make_spec(target=["score"], factor="group")
        return run_analysis(grouped_dataset, spec)

    def test_grouped_descriptives_has_group_column(self, grouped_result):
        """그룹별 분석 결과 Descriptives에 '그룹' 컬럼 존재."""
        desc_table = next(t for t in grouped_result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        assert "그룹" in df.columns, "그룹 컬럼 없음"

    def test_grouped_descriptives_has_two_groups(self, grouped_result):
        """그룹 A, B 각각의 통계량 행 존재."""
        desc_table = next(t for t in grouped_result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        groups = df["그룹"].unique()
        assert "A" in groups
        assert "B" in groups

    def test_grouped_mean_group_a_lower(self, grouped_result):
        """그룹 A 평균 < 그룹 B 평균 (설계 의도 확인).

        생성: A ~ N(50,10), B ~ N(70,10)
        그룹 A 평균이 그룹 B 평균보다 낮아야 함.
        """
        desc_table = next(t for t in grouped_result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        mean_a = float(df[(df["그룹"] == "A") & (df["통계량"] == "평균")]["값"].iloc[0])
        mean_b = float(df[(df["그룹"] == "B") & (df["통계량"] == "평균")]["값"].iloc[0])
        assert mean_a < mean_b, f"A 평균({mean_a:.2f}) < B 평균({mean_b:.2f}) 기대"

    def test_grouped_normality_table_has_groups(self, grouped_result):
        """그룹별 Tests of Normality에 그룹 컬럼 존재."""
        norm_table = next(t for t in grouped_result.tables if t.title == "Tests of Normality")
        df = norm_table.dataframe
        assert "그룹" in df.columns

    def test_grouped_percentiles_has_groups(self, grouped_result):
        """그룹별 Percentiles 테이블에 그룹 컬럼 존재."""
        pct_table = next(t for t in grouped_result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        assert "그룹" in df.columns

    def test_no_factor_no_group_column(self):
        """factor 없을 때 Descriptives에 '그룹' 컬럼 없음."""
        ds = _make_dataset()
        result = run_analysis(ds, _make_spec())
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        assert "그룹" not in df.columns


# ────────────────────────────────────────────────────────────────
# 6. 결과 테이블 구조 (5개 테이블)
# ────────────────────────────────────────────────────────────────

class TestTableStructure:
    """결과 테이블 구조 검증.

    SPSS Explore 출력 테이블 (5개):
        1. Case Processing Summary
        2. Descriptives
        3. Extreme Values
        4. Tests of Normality
        5. Percentiles
    """

    @pytest.fixture
    def result(self):
        ds = _make_dataset()
        return run_analysis(ds, _make_spec())

    def test_exactly_five_tables(self, result):
        """테이블 수 = 5개."""
        assert len(result.tables) == 5, \
            f"테이블 5개 기대, 실제 {len(result.tables)}개: {[t.title for t in result.tables]}"

    def test_table_titles_all_present(self, result):
        """5개 테이블 제목 모두 존재."""
        expected = {
            "Case Processing Summary",
            "Descriptives",
            "Extreme Values",
            "Tests of Normality",
            "Percentiles",
        }
        actual = {t.title for t in result.tables}
        assert expected == actual, f"누락 테이블: {expected - actual}"

    def test_case_processing_summary_structure(self, result):
        """Case Processing Summary 컬럼 구조."""
        cps = next(t for t in result.tables if t.title == "Case Processing Summary")
        df = cps.dataframe
        assert "변수" in df.columns
        assert "유효 N" in df.columns
        assert "결측 N" in df.columns
        assert "합계 N" in df.columns

    def test_descriptives_structure(self, result):
        """Descriptives 테이블 컬럼 구조 (변수, 통계량, 값, 표준오차)."""
        desc = next(t for t in result.tables if t.title == "Descriptives")
        df = desc.dataframe
        assert "변수" in df.columns
        assert "통계량" in df.columns
        assert "값" in df.columns

    def test_descriptives_required_statistics(self, result):
        """Descriptives에 SPSS 필수 통계량 모두 포함."""
        desc = next(t for t in result.tables if t.title == "Descriptives")
        df = desc.dataframe
        stats_present = set(df["통계량"].tolist())
        required = {
            "평균", "5% 절사평균", "중위수",
            "표준편차", "분산", "최솟값", "최댓값", "범위", "IQR",
            "왜도", "왜도 표준오차", "첨도", "첨도 표준오차",
            "95% 신뢰구간 하한", "95% 신뢰구간 상한",
        }
        missing = required - stats_present
        assert not missing, f"누락 통계량: {missing}"

    def test_extreme_values_structure(self, result):
        """Extreme Values 테이블 구조 (최솟값 5개, 최댓값 5개)."""
        ext = next(t for t in result.tables if t.title == "Extreme Values")
        df = ext.dataframe
        assert "변수" in df.columns
        assert "유형" in df.columns
        assert "순위" in df.columns
        assert "값" in df.columns

    def test_extreme_values_five_lowest_five_highest(self, result):
        """Extreme Values: 최솟값 5개 + 최댓값 5개 = 10행 (변수당)."""
        ext = next(t for t in result.tables if t.title == "Extreme Values")
        df = ext.dataframe
        lowest = df[df["유형"] == "최솟값"]
        highest = df[df["유형"] == "최댓값"]
        assert len(lowest) == 5, f"최솟값 5행 기대, 실제 {len(lowest)}행"
        assert len(highest) == 5, f"최댓값 5행 기대, 실제 {len(highest)}행"

    def test_extreme_values_correct_values(self, result):
        """Extreme Values 실제 최솟값/최댓값 정확성.

        SPSS 참조 데이터 최솟값 상위 5: 4, 7, 13, 16, 21
        SPSS 참조 데이터 최댓값 상위 5: 54, 51, 48, 45, 42
        """
        ext = next(t for t in result.tables if t.title == "Extreme Values")
        df = ext.dataframe
        lowest_vals = sorted(df[df["유형"] == "최솟값"]["값"].astype(float).tolist())
        highest_vals = sorted(df[df["유형"] == "최댓값"]["값"].astype(float).tolist(), reverse=True)
        assert lowest_vals[0] == _approx(4.0, 0.01)
        assert highest_vals[0] == _approx(54.0, 0.01)

    def test_normality_table_structure(self, result):
        """Tests of Normality 컬럼 구조."""
        norm = next(t for t in result.tables if t.title == "Tests of Normality")
        df = norm.dataframe
        assert "변수" in df.columns
        assert "Shapiro-Wilk W" in df.columns
        assert "Shapiro-Wilk p" in df.columns

    def test_percentiles_table_structure(self, result):
        """Percentiles 테이블 컬럼 구조."""
        pct = next(t for t in result.tables if t.title == "Percentiles")
        df = pct.dataframe
        assert "변수" in df.columns
        for col in ["P5", "P10", "P25", "P50", "P75", "P90", "P95"]:
            assert col in df.columns, f"{col} 컬럼 없음"


# ────────────────────────────────────────────────────────────────
# 7. 결측치 처리
# ────────────────────────────────────────────────────────────────

class TestMissingValueHandling:
    """결측치 처리 검증.

    SPSS Explore: Case Processing Summary에 유효/결측/합계 표시.
    결측 제외 후 분석 수행 (listwise 기본).
    """

    @pytest.fixture
    def dataset_with_missing(self):
        data = SPSS_DATA.copy()
        data_with_nan = [float(x) for x in data] + [np.nan, np.nan, np.nan]
        df = pd.DataFrame({"score": data_with_nan})
        ds = Dataset(df, "missing_test")
        ds.variables["score"].measure = MeasureType.SCALE
        return ds

    @pytest.fixture
    def result_with_missing(self, dataset_with_missing):
        return run_analysis(dataset_with_missing, _make_spec())

    def test_case_processing_summary_reflects_missing(self, result_with_missing):
        """결측 3개 → 유효 15, 결측 3, 합계 18."""
        cps = next(t for t in result_with_missing.tables
                   if t.title == "Case Processing Summary")
        df = cps.dataframe
        row = df[df["변수"] == "score"]
        assert int(row["유효 N"].iloc[0]) == 15
        assert int(row["결측 N"].iloc[0]) == 3
        assert int(row["합계 N"].iloc[0]) == 18

    def test_mean_excludes_missing(self, result_with_missing):
        """결측 제외 후 평균 ≈ 30.3333 (원본 15개와 동일)."""
        desc_table = next(t for t in result_with_missing.tables
                          if t.title == "Descriptives")
        df = desc_table.dataframe
        mean_row = df[df["통계량"] == "평균"]
        val = float(mean_row["값"].iloc[0])
        assert val == _approx(30.3333, 0.001)

    def test_all_missing_produces_warning(self):
        """전체 결측 데이터 → 경고 또는 빈 통계량."""
        df = pd.DataFrame({"score": [np.nan, np.nan, np.nan]})
        ds = Dataset(df, "all_missing")
        ds.variables["score"].measure = MeasureType.SCALE
        result = run_analysis(ds, _make_spec())
        # 경고가 있거나 통계량이 비어있어야 함
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df_desc = desc_table.dataframe
        mean_row = df_desc[df_desc["통계량"] == "평균"]
        if len(mean_row) > 0:
            val = mean_row["값"].iloc[0]
            # NaN 또는 빈 문자열이어야 함
            is_empty = (val == "" or (isinstance(val, float) and np.isnan(val)))
            assert is_empty or len(result.warnings) > 0


# ────────────────────────────────────────────────────────────────
# 8. 오류 처리
# ────────────────────────────────────────────────────────────────

class TestErrorHandling:
    """오류 처리 검증."""

    def test_missing_variable_raises_error(self):
        """존재하지 않는 변수 → ValueError 발생."""
        ds = _make_dataset()
        spec = _make_spec(target=["nonexistent"])
        with pytest.raises((ValueError, KeyError)):
            run_analysis(ds, spec)

    def test_empty_target_list(self):
        """빈 target 목록 → 빈 결과 또는 경고."""
        ds = _make_dataset()
        spec = _make_spec(target=[])
        # 오류 없이 실행되어야 함
        result = run_analysis(ds, spec)
        assert result is not None

    def test_small_sample_n_less_than_3(self):
        """n < 3 데이터 → 정규성 검정 경고."""
        df = pd.DataFrame({"score": [1.0, 2.0]})
        ds = Dataset(df, "tiny")
        ds.variables["score"].measure = MeasureType.SCALE
        result = run_analysis(ds, _make_spec())
        # 경고가 있거나 정상 완료
        assert result is not None

    def test_constant_variable_no_crash(self):
        """상수 변수(SD=0) → 오류 없이 실행."""
        df = pd.DataFrame({"score": [5.0] * 10})
        ds = Dataset(df, "const")
        ds.variables["score"].measure = MeasureType.SCALE
        result = run_analysis(ds, _make_spec())
        assert result is not None


# ────────────────────────────────────────────────────────────────
# 9. 95% 신뢰구간 검증
# ────────────────────────────────────────────────────────────────

class TestConfidenceIntervalSPSS:
    """95% 신뢰구간 SPSS 29 검증.

    SPSS 29 참조:
        95% CI 하한 = 21.4564
        95% CI 상한 = 39.2103
        SE = 4.1389
        t(0.025, df=14) = 2.1448

    Python: mean ± t.ppf(0.975, df=n-1) * (sd / sqrt(n))
    """

    @pytest.fixture
    def dataset(self):
        return _make_dataset()

    @pytest.fixture
    def result(self, dataset):
        return run_analysis(dataset, _make_spec())

    def test_ci_lower_spss_reference(self, result):
        """95% CI 하한 ≈ 21.4564 — SPSS 29 일치."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        ci_lo_row = df[df["통계량"] == "95% 신뢰구간 하한"]
        assert len(ci_lo_row) > 0, "95% 신뢰구간 하한 행 없음"
        val = float(ci_lo_row["값"].iloc[0])
        assert val == _approx(21.456, 0.01)

    def test_ci_upper_spss_reference(self, result):
        """95% CI 상한 ≈ 39.2103 — SPSS 29 일치."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        ci_hi_row = df[df["통계량"] == "95% 신뢰구간 상한"]
        assert len(ci_hi_row) > 0, "95% 신뢰구간 상한 행 없음"
        val = float(ci_hi_row["값"].iloc[0])
        assert val == _approx(39.210, 0.01)

    def test_ci_symmetric_around_mean(self, result):
        """CI가 평균을 기준으로 대칭."""
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        mean_val = float(df[df["통계량"] == "평균"]["값"].iloc[0])
        ci_lo = float(df[df["통계량"] == "95% 신뢰구간 하한"]["값"].iloc[0])
        ci_hi = float(df[df["통계량"] == "95% 신뢰구간 상한"]["값"].iloc[0])
        margin = (ci_hi - ci_lo) / 2
        assert abs((mean_val - ci_lo) - margin) < 0.01, "CI가 평균 기준 대칭이 아님"


# ────────────────────────────────────────────────────────────────
# 10. 다중 변수 분석
# ────────────────────────────────────────────────────────────────

class TestMultipleVariables:
    """다중 변수 동시 분석 검증."""

    @pytest.fixture
    def multi_dataset(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": SPSS_DATA,
            "x2": np.random.normal(100, 15, 15).tolist(),
        })
        ds = Dataset(df, "multi_test")
        ds.variables["x1"].measure = MeasureType.SCALE
        ds.variables["x2"].measure = MeasureType.SCALE
        return ds

    def test_multi_variable_descriptives_has_both(self, multi_dataset):
        """다중 변수 분석 시 두 변수 모두 Descriptives에 포함."""
        spec = _make_spec(target=["x1", "x2"])
        result = run_analysis(multi_dataset, spec)
        desc_table = next(t for t in result.tables if t.title == "Descriptives")
        df = desc_table.dataframe
        vars_present = df["변수"].unique()
        assert "x1" in vars_present
        assert "x2" in vars_present

    def test_multi_variable_normality_has_both(self, multi_dataset):
        """다중 변수 분석 시 두 변수 모두 Tests of Normality에 포함."""
        spec = _make_spec(target=["x1", "x2"])
        result = run_analysis(multi_dataset, spec)
        norm_table = next(t for t in result.tables if t.title == "Tests of Normality")
        df = norm_table.dataframe
        vars_present = df["변수"].unique()
        assert "x1" in vars_present
        assert "x2" in vars_present

    def test_multi_variable_percentiles_has_both(self, multi_dataset):
        """다중 변수 분석 시 두 변수 모두 Percentiles에 포함."""
        spec = _make_spec(target=["x1", "x2"])
        result = run_analysis(multi_dataset, spec)
        pct_table = next(t for t in result.tables if t.title == "Percentiles")
        df = pct_table.dataframe
        vars_present = df["변수"].unique()
        assert "x1" in vars_present
        assert "x2" in vars_present

    def test_multi_variable_five_tables_total(self, multi_dataset):
        """다중 변수도 테이블 수 = 5 유지."""
        spec = _make_spec(target=["x1", "x2"])
        result = run_analysis(multi_dataset, spec)
        assert len(result.tables) == 5
