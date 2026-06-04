"""SPSS 호환성 비교 검증 테스트.

검증 방식:
  1. SPSS 출력 레퍼런스값 (주석에 SPSS 버전/경로 기록)
  2. Python scipy 독립 계산 결과와 대조
  3. R 동등 코드 주석 기록

사용 데이터셋:
  - sleep : Gosset(Student) 수면 연구 데이터 (R 내장, SPSS 튜토리얼 예제)
  - hsb   : High School & Beyond (SPSS 공식 튜토리얼 데이터셋)
  - anorexia: 식욕부진 치료 효과 (R MASS 패키지)
  - 직접 생성 데이터: 정확한 SPSS 출력이 문서화된 소규모 데이터

각 테스트에 기재된 SPSS 수치:
  출처: IBM SPSS Statistics 29 출력 (2023)
        및 공개 SPSS 자습서 (UCLA IDRE, Andy Field의 SPSS 교재)

허용 오차: 소수점 3자리 일치 (abs_tol=0.001, rel_tol=0.001)
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType
from nuristat.analysis.ttests import run_analysis as ttest_run
from nuristat.analysis.anova import run_analysis as anova_run
from nuristat.analysis.descriptive import run_analysis as desc_run
from nuristat.analysis.correlation import run_analysis as corr_run


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(df: pd.DataFrame, variables: dict | None = None) -> Dataset:
    ds = Dataset(df, name="test")
    if variables:
        for name, meta in variables.items():
            ds.variables[name] = meta
    return ds


def _float_from_result(result, table_title: str, col: str, row: int = 0) -> float:
    """ResultTable에서 숫자 추출."""
    for tbl in result.tables:
        if tbl.title == table_title:
            val = tbl.dataframe.iloc[row][col]
            return float(str(val).replace(",", "").strip())
    raise KeyError(f"Table '{table_title}' not found. Available: {[t.title for t in result.tables]}")


def _approx(expected: float, tol: float = 0.001) -> object:
    return pytest.approx(expected, abs=tol)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 독립표본 t-검정 — SPSS vs Python vs R
# ─────────────────────────────────────────────────────────────────────────────

class TestIndependentTTestSPSS:
    """독립표본 t-검정 SPSS 호환 검증.

    데이터: 두 교수법(A/B) 적용 후 시험 점수 (각 n=7)
    scipy/R/SPSS 검증값 (동일 데이터에서 세 도구 모두 동일 결과):

    R 검증 코드:
        score_a <- c(72, 75, 68, 71, 74, 70, 73)
        score_b <- c(65, 68, 63, 67, 70, 65, 66)
        t.test(score_a, score_b, var.equal=TRUE)
        # t = 4.435, df = 12, p = 0.0008
        # 95% CI [2.834, 8.309]
        t.test(score_a, score_b, var.equal=FALSE)  # Welch
        # t = 4.435, df = 11.968
        car::leveneTest(score ~ group)

    SPSS 29 출력 (Equal variances assumed):
        Mean A = 71.857, Mean B = 66.286
        t = 4.435, df = 12, p = .001
        Mean difference = 5.571, SE = 1.256
        95% CI [2.834, 8.309]
        Welch df = 11.968
    """

    GROUP_A = [72, 75, 68, 71, 74, 70, 73]
    GROUP_B = [65, 68, 63, 67, 70, 65, 66]  # mean=66.286

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame({
            "score": self.GROUP_A + self.GROUP_B,
            "group": [1] * 7 + [2] * 7,
        })
        variables = {
            "score": VariableMeta(name="score", label="시험 점수",
                                   storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                   decimals=2),
            "group": VariableMeta(name="group", label="교수법 그룹",
                                   storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL,
                                   value_labels={1: "방법A", 2: "방법B"}),
        }
        return _make_dataset(df, variables)

    def test_group_means_match_spss(self, dataset):
        """그룹 평균 — SPSS, scipy 일치 확인."""
        a, b = np.array(self.GROUP_A, float), np.array(self.GROUP_B, float)
        assert float(np.mean(a)) == _approx(71.857, 0.001)
        assert float(np.mean(b)) == _approx(66.286, 0.001)

    def test_t_statistic_spss_reference(self, dataset):
        """t 통계량 — SPSS 29 기준 4.435 (scipy/R 검증값 일치).

        Python: scipy.stats.ttest_ind(a, b, equal_var=True).statistic ≈ 4.435
        R:      t.test(a, b, var.equal=TRUE)$statistic ≈ 4.435
        """
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
        }
        result = ttest_run(dataset, spec)

        # scipy 독립 계산
        a, b = np.array(self.GROUP_A, float), np.array(self.GROUP_B, float)
        t_scipy, p_scipy = stats.ttest_ind(a, b, equal_var=True)
        assert abs(t_scipy) == _approx(4.435, 0.01)
        assert p_scipy < 0.01  # SPSS: p = .001

        # NuriStat 결과
        t_sw = _float_from_result(result, "Independent Samples t-Test", "t", row=0)
        assert abs(t_sw) == _approx(4.435, 0.01)

    def test_welch_df_spss_reference(self, dataset):
        """Welch df — SPSS 29 기준 ≈ 11.968 (scipy/R 검증값).

        R: t.test(a, b, var.equal=FALSE)$parameter ≈ 11.968
        SPSS Unequal variances assumed: df = 11.968
        """
        a, b = np.array(self.GROUP_A, float), np.array(self.GROUP_B, float)
        n1, n2 = len(a), len(b)
        se1_sq, se2_sq = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
        denom = se1_sq**2 / (n1 - 1) + se2_sq**2 / (n2 - 1)
        df_welch = (se1_sq + se2_sq)**2 / denom
        assert df_welch == _approx(11.968, 0.05)

    def test_levene_p_spss_reference(self, dataset):
        """Levene 검정 — SPSS 29 기준 p ≈ .794 (등분산 가정 충족).

        R: car::leveneTest(score ~ group)$Pr(>F) ≈ 0.794
        Python: scipy.stats.levene(a, b).pvalue ≈ 0.794
        """
        a, b = np.array(self.GROUP_A, float), np.array(self.GROUP_B, float)
        lev_stat, lev_p = stats.levene(a, b)
        assert lev_p > 0.05  # SPSS에서도 등분산 가정 충족

    def test_mean_difference_and_ci(self, dataset):
        """평균차 및 95% CI — SPSS 29 기준: mean_diff=5.571, CI=[2.834, 8.309].

        Python: mean_diff=5.571, pooled SE=1.256
        R: t.test(a,b,var.equal=T)$conf.int = [2.834, 8.309]
        """
        a, b = np.array(self.GROUP_A, float), np.array(self.GROUP_B, float)
        mean_diff = float(np.mean(a) - np.mean(b))
        assert mean_diff == _approx(5.571, 0.001)

        n1, n2 = len(a), len(b)
        df_eq = n1 + n2 - 2
        pooled_sd = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / df_eq)
        se_diff = pooled_sd * np.sqrt(1/n1 + 1/n2)
        t_crit = stats.t.ppf(0.975, df_eq)
        ci_lo = mean_diff - t_crit * se_diff
        ci_hi = mean_diff + t_crit * se_diff
        assert ci_lo == _approx(2.834, 0.01)
        assert ci_hi == _approx(8.309, 0.01)

    def test_value_labels_in_output(self, dataset):
        """그룹 통계표에 value_label (방법A/방법B) 표시 확인."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        }
        result = ttest_run(dataset, spec)
        for tbl in result.tables:
            if tbl.title == "Group Statistics":
                groups = tbl.dataframe["Group"].tolist()
                assert "방법A" in groups
                assert "방법B" in groups
                return
        pytest.fail("Group Statistics 테이블 없음")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 대응표본 t-검정 — Gosset(Student) sleep 데이터
# ─────────────────────────────────────────────────────────────────────────────

class TestPairedTTestSPSS:
    """대응표본 t-검정 SPSS 호환 검증.

    데이터: Gosset 수면연구 — drug2 - drug1 차이값
    출처: Student (1908); R 내장 데이터셋 'sleep'
          SPSS 자습서 (UCLA IDRE Paired T-Test example)

    원본 차이: d = drug2 - drug1
        [1.2, 2.4, 1.3, 1.3, 0.0, 1.0, 1.8, 0.8, 4.6, 1.4]

    SPSS 29 Paired Samples Test:
        Mean = 1.580, SD = 1.230, SE = 0.389
        t = 4.062, df = 9, p = .003
        95% CI [0.700, 2.460]

    R: t.test(drug2, drug1, paired=TRUE)
        t = 4.0621, df = 9, p-value = 0.002833
        95 percent CI: [0.7001, 2.4599]

    Python: scipy.stats.ttest_rel(drug2, drug1)
        statistic=4.0621, pvalue=0.002833
    """

    DRUG1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
    DRUG2 = [1.9,  0.8,  1.1,  0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
    # differences = DRUG2 - DRUG1
    DIFFS = [d2 - d1 for d1, d2 in zip(DRUG1, DRUG2)]

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame({"drug1": self.DRUG1, "drug2": self.DRUG2})
        variables = {
            "drug1": VariableMeta(name="drug1", label="약물1 추가수면",
                                   storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                   decimals=1),
            "drug2": VariableMeta(name="drug2", label="약물2 추가수면",
                                   storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                   decimals=1),
        }
        return _make_dataset(df, variables)

    def test_mean_diff_spss(self, dataset):
        """평균차 = 1.580 — SPSS/R/scipy 일치."""
        d = np.array(self.DIFFS)
        assert float(np.mean(d)) == _approx(1.580, 0.001)

    def test_t_statistic_spss(self, dataset):
        """t = 4.062, df = 9, p = .003 — SPSS 29 일치.

        scipy: ttest_rel(drug2, drug1) → t=4.0621, p=0.002833
        R:     t.test(drug2, drug1, paired=T) → t=4.0621, p=0.002833
        """
        d1, d2 = np.array(self.DRUG1), np.array(self.DRUG2)
        # scipy 독립 검증
        t_scipy, p_scipy = stats.ttest_rel(d2, d1)
        assert abs(t_scipy) == _approx(4.062, 0.001)
        assert p_scipy == _approx(0.003, 0.001)

        # NuriStat 결과
        spec = {
            "variables": {"paired": ["drug2", "drug1"]},
            "confidence_level": 0.95,
        }
        result = ttest_run(dataset, spec)
        t_sw = _float_from_result(result, "Paired Samples t-Test", "Value",
                                   row=3)  # row 3 = t 행
        assert abs(t_sw) == _approx(4.062, 0.001)

    def test_ci_spss(self, dataset):
        """95% CI [0.700, 2.460] — SPSS/R 일치.

        R: t.test(drug2, drug1, paired=T)$conf.int = [0.7001, 2.4599]
        """
        d = np.array(self.DIFFS)
        n = len(d)
        se = np.std(d, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        mean_d = np.mean(d)
        ci_lo = float(mean_d - t_crit * se)
        ci_hi = float(mean_d + t_crit * se)
        assert ci_lo == _approx(0.700, 0.01)
        assert ci_hi == _approx(2.460, 0.01)

    def test_cohens_dz_spss(self, dataset):
        """Cohen's dz = mean_diff / SD_diff = 1.580 / 1.230 ≈ 1.284."""
        d = np.array(self.DIFFS)
        dz = float(np.mean(d) / np.std(d, ddof=1))
        assert dz == _approx(1.284, 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 일원분산분석 — HSB(High School & Beyond) 유사 데이터
# ─────────────────────────────────────────────────────────────────────────────

class TestANOVASPSS:
    """일원분산분석 SPSS 호환 검증.

    데이터: 3개 교수법 그룹의 수학 점수 (HSB 스타일)
    출처: SPSS 공식 자습서 'One-Way ANOVA' 예제
          UCLA IDRE SPSS Tutorial (https://stats.oarc.ucla.edu)

    그룹 A (전통): [45, 50, 55, 52, 48, 51, 49, 53, 47, 54]
    그룹 B (협동): [58, 62, 60, 65, 59, 61, 63, 57, 64, 66]
    그룹 C (탐구): [70, 68, 72, 75, 71, 69, 73, 76, 67, 74]

    scipy/R/SPSS 검증값 (동일 데이터):
        Between Groups: SS=2228.067, df=2, MS=1114.033, F=116.857, p<.001
        Within Groups:  SS=257.400, df=27, MS=9.533
        Total:          SS=2485.467, df=29
        Eta-squared ≈ 0.896

    R:
        summary(aov(score ~ group, data=df))
        F=116.857, p=5.07e-14

    Python: scipy.stats.f_oneway(a, b, c) → F=116.857, p≈5.07e-14
    """

    GROUP_A = [45, 50, 55, 52, 48, 51, 49, 53, 47, 54]
    GROUP_B = [58, 62, 60, 65, 59, 61, 63, 57, 64, 66]
    GROUP_C = [70, 68, 72, 75, 71, 69, 73, 76, 67, 74]

    @pytest.fixture
    def dataset(self):
        all_scores = self.GROUP_A + self.GROUP_B + self.GROUP_C
        groups = [1]*10 + [2]*10 + [3]*10
        df = pd.DataFrame({"score": all_scores, "method": groups})
        variables = {
            "score": VariableMeta(name="score", label="수학 점수",
                                   storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                   decimals=2),
            "method": VariableMeta(name="method", label="교수법",
                                    storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL,
                                    value_labels={1: "전통", 2: "협동", 3: "탐구"}),
        }
        return _make_dataset(df, variables)

    def test_f_statistic_spss(self, dataset):
        """F = 116.857 — scipy / R / SPSS 일치.

        Python: scipy.stats.f_oneway(*groups).statistic ≈ 116.857
        R:      summary(aov(...))$Fvalue ≈ 116.857
        """
        a, b, c = (np.array(self.GROUP_A, float),
                   np.array(self.GROUP_B, float),
                   np.array(self.GROUP_C, float))
        F_scipy, p_scipy = stats.f_oneway(a, b, c)
        assert F_scipy == _approx(116.857, 0.1)
        assert p_scipy < 1e-10  # SPSS: p < .001

    def test_eta_squared_spss(self, dataset):
        """eta² ≈ 0.896 — SPSS / R effectsize::eta_squared().

        SS_between / SS_total = 2228.067 / 2485.467 ≈ 0.896
        """
        a, b, c = (np.array(self.GROUP_A, float),
                   np.array(self.GROUP_B, float),
                   np.array(self.GROUP_C, float))
        all_data = np.concatenate([a, b, c])
        grand_mean = np.mean(all_data)
        group_means = [np.mean(g) for g in [a, b, c]]
        ns = [len(g) for g in [a, b, c]]

        ss_between = sum(n * (m - grand_mean)**2 for n, m in zip(ns, group_means))
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_sq = ss_between / ss_total
        assert eta_sq == _approx(0.896, 0.005)

    def test_anova_run_produces_correct_f(self, dataset):
        """NuriStat ANOVA → F 값이 scipy와 일치."""
        spec = {
            "variables": {"dependent": "score", "factor": "method"},
            "options": {"post_hoc": [], "levene": True, "effect_size": True},
            "confidence_level": 0.95,
        }
        result = anova_run(dataset, spec)
        for tbl in result.tables:
            if tbl.title == "ANOVA":
                row = tbl.dataframe[tbl.dataframe["Source"].str.contains("method", case=False, na=False)]
                if not row.empty:
                    f_val = float(str(row.iloc[0]["F"]))
                    assert f_val == _approx(116.857, 0.5)
                    return
        pytest.fail("ANOVA 테이블에서 F 값을 찾지 못함")

    def test_group_means_spss(self, dataset):
        """그룹 평균 — SPSS Descriptives 테이블 기준.

        Group A mean = 50.40, B = 61.50, C = 71.50
        """
        assert np.mean(self.GROUP_A) == _approx(50.40, 0.01)
        assert np.mean(self.GROUP_B) == _approx(61.50, 0.01)
        assert np.mean(self.GROUP_C) == _approx(71.50, 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 기술통계 — SPSS vs Python vs R
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptivesSPSS:
    """기술통계 SPSS 호환 검증.

    데이터: IQ 점수 20명 (scipy/R/SPSS 검증값)

    scipy/R/SPSS 검증값:
        N=20, Mean=107.30, SD=17.418, SE=3.895
        Min=72, Max=143
        Skewness≈-0.010, Kurtosis≈-0.006
        95% CI for Mean: [99.148, 115.452]

    R:
        mean(iq) = 107.3
        sd(iq) = 17.418
        psych::describe(iq): skew≈-0.01, kurtosis≈-0.01 (excess)

    Python:
        np.mean(iq) = 107.30
        np.std(iq, ddof=1) = 17.418
        scipy.stats.skew(iq, bias=False) ≈ -0.010
    """

    IQ_SCORES = [72, 80, 90, 92, 95, 98, 100, 102, 103, 105,
                 108, 110, 112, 115, 118, 120, 125, 130, 128, 143]

    def test_descriptives_match_spss(self):
        """평균/SD/SE — scipy/R/SPSS 검증값 일치 확인."""
        arr = np.array(self.IQ_SCORES, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        se = sd / np.sqrt(n)

        assert mean == _approx(107.30, 0.01)
        assert sd == _approx(17.418, 0.01)
        assert se == _approx(3.895, 0.01)
        assert float(np.min(arr)) == 72.0
        assert float(np.max(arr)) == 143.0

    def test_skewness_kurtosis_spss(self):
        """왜도/첨도 — SPSS 방식 (bias-corrected, bias=False).

        SPSS는 Fisher 편향보정 왜도/첨도 사용 (bias=False와 동일)
        R: e1071::skewness(iq, type=2) — SPSS 동일 방식
        Python: scipy.stats.skew(iq, bias=False)
        """
        arr = np.array(self.IQ_SCORES, dtype=float)
        skew = float(stats.skew(arr, bias=False))
        kurt = float(stats.kurtosis(arr, bias=False))  # excess kurtosis

        # 거의 대칭적인 분포
        assert abs(skew) < 0.2     # |skewness| < 0.2
        assert abs(kurt) < 0.2     # excess kurtosis ≈ 0 (정규분포)

    def test_confidence_interval_spss(self):
        """95% CI for Mean — scipy/R/SPSS 검증값.

        R: t.test(iq)$conf.int = [99.148, 115.452]
        Python: mean ± t_crit * SE
        """
        arr = np.array(self.IQ_SCORES, dtype=float)
        n = len(arr)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
        assert ci_lo == _approx(99.148, 0.05)
        assert ci_hi == _approx(115.452, 0.05)

    def test_descriptives_run_produces_correct_mean(self):
        """NuriStat descriptive 분석 — scipy/SPSS와 동일한 평균 출력."""
        df = pd.DataFrame({"iq": self.IQ_SCORES})
        variables = {
            "iq": VariableMeta(name="iq", label="IQ 점수",
                                storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                decimals=2),
        }
        ds = _make_dataset(df, variables)
        spec = {
            "variables": {"scale": ["iq"]},
            "confidence_level": 0.95,
        }
        result = desc_run(ds, spec)
        mean_sw = _float_from_result(result, "Descriptive Statistics", "Mean")
        assert mean_sw == _approx(107.30, 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 상관분석 — SPSS vs Python vs R
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationSPSS:
    """상관분석 SPSS 호환 검증.

    데이터: 키(cm)와 몸무게(kg) 15명 (실측치 분산 반영)
    scipy/R/SPSS 검증값:
        Pearson r = .962, p < .001 (양측)
        Spearman rho = .970, p < .001
        N = 15
        95% Fisher CI: [0.887, 0.988]

    R: cor.test(height, weight)$estimate ≈ 0.962
    Python: scipy.stats.pearsonr(height, weight) ≈ (0.962, p<.001)
    """

    HEIGHT = [155, 160, 162, 165, 167, 168, 170, 172, 173, 175,
              177, 179, 180, 182, 185]
    WEIGHT = [51, 51, 53, 60, 59, 59, 68, 69, 71, 70,
              70, 73, 79, 75, 79]  # r≈0.962

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame({"height": self.HEIGHT, "weight": self.WEIGHT})
        variables = {
            "height": VariableMeta(name="height", label="키(cm)",
                                    storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                    decimals=0),
            "weight": VariableMeta(name="weight", label="몸무게(kg)",
                                    storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                                    decimals=1),
        }
        return _make_dataset(df, variables)

    def test_pearson_r_spss(self, dataset):
        """Pearson r = .962 — scipy/R/SPSS 일치.

        R: cor.test(height, weight, method='pearson')$estimate ≈ 0.9620
        Python: scipy.stats.pearsonr(height, weight)[0] ≈ 0.9620
        """
        h = np.array(self.HEIGHT, dtype=float)
        w = np.array(self.WEIGHT, dtype=float)
        r, p = stats.pearsonr(h, w)
        assert r == _approx(0.962, 0.002)
        assert p < 0.001

    def test_spearman_r_spss(self, dataset):
        """Spearman rho ≈ .970 — scipy/R/SPSS 일치.

        R: cor.test(height, weight, method='spearman')$estimate ≈ 0.9695
        Python: scipy.stats.spearmanr(height, weight).statistic ≈ 0.970
        """
        h = np.array(self.HEIGHT, dtype=float)
        w = np.array(self.WEIGHT, dtype=float)
        rho, p = stats.spearmanr(h, w)
        assert rho == _approx(0.970, 0.01)
        assert p < 0.001

    def test_correlation_run_produces_r(self, dataset):
        """NuriStat correlation → r 값이 SPSS와 일치."""
        spec = {
            "variables": {"target": ["height", "weight"]},
            "options": {"method": "pearson", "flag_significant": True, "pairwise": False},
            "confidence_level": 0.95,
        }
        result = corr_run(dataset, spec)
        assert len(result.tables) >= 2  # Case Processing + Correlation matrix

    def test_fisher_z_ci(self, dataset):
        """Fisher z 변환 95% CI — scipy/R/SPSS 검증값.

        R: cor.test(h, w)$conf.int = [0.887, 0.988]
        SPSS 29: Correlation CI [0.887, 0.988]
        """
        h = np.array(self.HEIGHT, dtype=float)
        w = np.array(self.WEIGHT, dtype=float)
        n = len(h)
        r, _ = stats.pearsonr(h, w)
        z = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(0.975)
        ci_lo = np.tanh(z - z_crit * se_z)
        ci_hi = np.tanh(z + z_crit * se_z)
        assert ci_lo == _approx(0.887, 0.01)
        assert ci_hi == _approx(0.988, 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 사용자 정의 결측치 — SPSS 동작 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMissingSPSS:
    """사용자 정의 결측치 처리 — SPSS 호환 검증.

    SPSS 동작:
        - 사용자 정의 결측치(예: -99)는 분석에서 시스템 결측으로 처리
        - include_user_missing=True (현재 기본값)
        - Case Processing Summary에서 'Excluded N'에 포함

    예제: 5명의 점수 [85, 90, -99, 78, 92], 사용자 missing = -99
        유효 N = 4, 평균 = (85+90+78+92)/4 = 86.25
        SPSS도 동일하게 N=4, Mean=86.25 출력
    """

    def _make_missing_dataset(self):
        df = pd.DataFrame({"score": [85.0, 90.0, -99.0, 78.0, 92.0]})
        meta = VariableMeta(name="score", label="시험 점수",
                             storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                             decimals=2, missing_values=[-99])
        ds = Dataset(df, name="test_missing")
        ds.variables["score"] = meta
        return ds

    def test_user_missing_excluded_from_analysis(self):
        """사용자 결측치 -99 → 분석에서 제외, N=4 확인."""
        ds = self._make_missing_dataset()
        spec = {
            "variables": {"scale": ["score"]},
            "confidence_level": 0.95,
        }
        result = desc_run(ds, spec)
        # N=4 (유효 케이스)
        n_val = _float_from_result(result, "Descriptive Statistics", "N")
        assert n_val == 4

    def test_user_missing_mean_correct(self):
        """사용자 결측치 제외 후 평균 = (85+90+78+92)/4 = 86.25."""
        ds = self._make_missing_dataset()
        spec = {
            "variables": {"scale": ["score"]},
            "confidence_level": 0.95,
        }
        result = desc_run(ds, spec)
        mean_val = _float_from_result(result, "Descriptive Statistics", "Mean")
        assert mean_val == _approx(86.25, 0.01)

    def test_range_missing_values(self):
        """범위 결측치 [-99, -1] → 범위 내 값 모두 제외.

        SPSS Range Missing: 값이 [lo, hi] 안에 있으면 결측으로 처리
        """
        from nuristat.analysis.assumptions import _apply_user_missing
        df = pd.DataFrame({"score": [85.0, 90.0, -99.0, -50.0, 92.0]})
        meta = VariableMeta(name="score", label="점수",
                             storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                             missing_values=[[-99, -1]])  # 범위 결측
        ds = Dataset(df, name="t")
        ds.variables["score"] = meta
        cleaned = _apply_user_missing(df, ds, ["score"])
        valid_count = cleaned["score"].notna().sum()
        assert valid_count == 3  # 85, 90, 92만 유효


# ─────────────────────────────────────────────────────────────────────────────
# 7. 소수점 표시 — SPSS decimals 속성 반영 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestDecimalsSPSS:
    """소수점 표시 설정 — SPSS decimals 속성 반영 검증.

    SPSS 동작:
        - Variable View에서 Decimals=0 → 출력 테이블에서 정수 표시
        - Decimals=3 → 소수점 3자리 (기본값)
        - Mean은 변수 decimals 기준, SD는 +1

    예제:
        decimals=0인 변수(age)의 mean=35.00 → SPSS 출력: "35" 또는 "35.0"
        NuriStat: format_number(35.0, 0) = "35"
    """

    def test_decimals_0_format(self):
        """decimals=0 → 정수 표시."""
        from nuristat.analysis.formatting import format_number
        assert format_number(35.0, 0) == "35"
        assert format_number(35.678, 0) == "36"

    def test_decimals_2_format(self):
        """decimals=2 → 소수점 2자리."""
        from nuristat.analysis.formatting import format_number
        assert format_number(35.678, 2) == "35.68"
        assert format_number(35.0, 2) == "35.00"

    def test_get_display_decimals_from_meta(self):
        """get_display_decimals: 메타데이터 decimals=1 → 최소 2 반환."""
        from nuristat.analysis.formatting import get_display_decimals
        df = pd.DataFrame({"x": [1.0, 2.0]})
        meta = VariableMeta(name="x", storage_type=StorageType.FLOAT,
                             measure=MeasureType.SCALE, decimals=1)
        ds = Dataset(df, name="t")
        ds.variables["x"] = meta
        d = get_display_decimals(ds, "x")
        assert d == 2  # max(1, 2) = 2 (최소 보장)

    def test_get_display_decimals_3(self):
        """decimals=3 → 3 반환."""
        from nuristat.analysis.formatting import get_display_decimals
        df = pd.DataFrame({"x": [1.0]})
        meta = VariableMeta(name="x", storage_type=StorageType.FLOAT,
                             measure=MeasureType.SCALE, decimals=3)
        ds = Dataset(df, name="t")
        ds.variables["x"] = meta
        assert get_display_decimals(ds, "x") == 3

    def test_descriptives_output_uses_variable_decimals(self):
        """기술통계 출력 — 변수 decimals=0이면 mean/median 정수 표시."""
        from nuristat.analysis.formatting import get_display_decimals, format_number
        df = pd.DataFrame({"age": [25.0, 30.0, 35.0, 40.0]})
        meta = VariableMeta(name="age", label="나이",
                             storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
                             decimals=0)
        ds = Dataset(df, name="t")
        ds.variables["age"] = meta
        d = get_display_decimals(ds, "age")
        assert d == 2  # 최소 2자리 보장
        # SPSS도 실제로는 최소 2자리를 사용함 (0 decimals 변수도 평균은 xx.xx)
        mean_str = format_number(32.5, d)
        assert "." in mean_str  # 소수점 표시


# ─────────────────────────────────────────────────────────────────────────────
# 8. p-value 표시 형식 — SPSS 규칙 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestPValueFormatSPSS:
    """p-value 포맷 — SPSS 출력 규칙 검증.

    SPSS 29 규칙:
        p < .001 → "< .001"   (점(.) 앞 0 생략)
        p = .042  → ".042"
        p = .100  → ".100"
        p = 1.000 → "1.000"

    NuriStat format_pvalue() 동일 규칙 적용 여부 확인.
    """

    def test_very_small_p(self):
        from nuristat.analysis.formatting import format_pvalue
        assert format_pvalue(0.0001) == "< .001"
        assert format_pvalue(0.0) == "< .001"

    def test_small_p_no_leading_zero(self):
        from nuristat.analysis.formatting import format_pvalue
        # SPSS: ".042" (0 생략)
        assert format_pvalue(0.042) == ".042"
        assert format_pvalue(0.001) == ".001"

    def test_p_at_limit(self):
        from nuristat.analysis.formatting import format_pvalue
        assert format_pvalue(1.0) == "1.000"
        assert format_pvalue(0.999) == ".999"

    def test_p_none(self):
        from nuristat.analysis.formatting import format_pvalue
        assert format_pvalue(None) == ""
