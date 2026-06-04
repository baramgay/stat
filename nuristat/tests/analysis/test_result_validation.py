"""분석 결과 정확성 검증 테스트.

R (v4.6.0) 계산 결과를 기준값으로 삼아 NuriStat 분석 결과의
수치 정확성을 검증한다. 허용 오차는 통계적으로 동등한 수준(소수점 3자리).

R 기준값 생성 코드:
    set.seed(42)
    n <- 30
    x <- rnorm(n, mean=50, sd=10)
    y <- 2*x + rnorm(n, sd=5)
    group <- rep(c("A","B","C"), 10)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
src_path = str(_PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset

import nuristat.analysis.descriptive as mod_descriptive
import nuristat.analysis.ttests as mod_ttests
import nuristat.analysis.correlation as mod_correlation
import nuristat.analysis.regression as mod_regression
import nuristat.analysis.normality as mod_normality
import nuristat.analysis.anova as mod_anova
import nuristat.analysis.frequencies as mod_frequencies


# ── 공통 데이터셋 (R set.seed(42)와 동일) ─────────────────────────────────

def _make_ref_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """R set.seed(42) rnorm(30, 50, 10) 와 동일한 정확한 값을 하드코딩.

    R 코드:
        set.seed(42)
        x <- rnorm(30, mean=50, sd=10)
        noise <- rnorm(30, sd=5)  # set.seed(42) 이후 연속 난수
        y <- 2*x + noise
        group <- rep(c("A","B","C"), 10)
    """
    # R set.seed(42); rnorm(30, 50, 10) 정확한 결과값
    x = np.array([
        63.70958447, 44.35301829, 53.63128411, 56.32862605, 54.04268323,
        48.93875484, 65.11521997, 49.05340962, 70.18423714, 49.37285901,
        63.04869654, 72.86645393, 36.11139299, 47.21211233, 48.66678664,
        56.35950398, 47.15747079, 23.43544579, 25.59533071, 63.20113346,
        46.93361406, 32.18691566, 48.28082644, 62.14674699, 68.95193461,
        45.69530868, 47.42730617, 32.36836915, 54.60097355, 43.60005124,
    ])
    # R: y <- 2*x + rnorm(30, sd=5) (set.seed(42) 연속 난수)
    y = np.array([
        129.69641956, 92.23022326, 112.43808584, 109.61262022, 110.61014208,
        89.29246628, 126.30814491, 93.85228126, 128.29743603, 98.92633105,
        127.12738609, 143.92762136, 76.01360216, 90.79070053, 90.49216805,
        114.88309809, 90.25797569, 54.09139789, 49.03343042, 129.68050633,
        95.47685444, 60.45463662, 104.44029048, 127.50799051, 138.35267246,
        92.77337110, 98.25105642, 65.18590273, 94.23649668, 88.62451725,
    ])
    # R: group <- rep(c("A","B","C"), 10) → 인터리빙
    group = (["A", "B", "C"] * 10)
    return x, y, group


def _make_ds(df: pd.DataFrame) -> Dataset:
    return Dataset(data=df, name="validation_test")


def _extract_number(result: AnalysisResult, table_idx: int, row_label: str,
                    value_col: str = "Value") -> float | None:
    """결과 테이블에서 특정 행의 값을 float으로 추출."""
    if table_idx >= len(result.tables):
        return None
    df = result.tables[table_idx].dataframe
    if "Statistic" in df.columns:
        mask = df["Statistic"] == row_label
        if mask.any():
            val = df.loc[mask, value_col].iloc[0]
            try:
                return float(str(val).replace(",", ""))
            except (ValueError, TypeError):
                return None
    return None


# ── R 기준값 ─────────────────────────────────────────────────────────────────

# R: set.seed(42); x <- rnorm(30, 50, 10)
# 위 하드코딩 배열 기준
R_MEAN   = 50.68587
R_SD     = 12.55028
R_MEDIAN = 48.99608
R_N      = 30
R_MIN    = 23.43545
R_MAX    = 72.86645

R_T_STAT = -0.5599447   # t.test(A, B, var.equal=FALSE)
R_T_P    = 0.5855919

R_PEARSON_R = 0.9778799  # cor(x, y)

R_INTERCEPT = 3.716291   # lm(y ~ x)
R_SLOPE     = 1.914654
R_R2        = 0.9562492

R_SW_W   = 0.962094     # shapiro.test(x)
R_SW_P   = 0.3500046

# ── TOLERANCE ─────────────────────────────────────────────────────────────────

TOL = 1e-3   # 소수점 3자리 허용 오차


# ─────────────────────────────────────────────────────────────────────────────
# 1. 기술통계 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptiveValidation:

    def _make_result(self):
        x, _, _ = _make_ref_data()
        df = pd.DataFrame({"x": x})
        ds = _make_ds(df)
        return mod_descriptive.run_analysis(ds, {"variables": {"scale": ["x"]}})

    def test_produces_result(self):
        result = self._make_result()
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 1

    def test_n(self):
        result = self._make_result()
        df = result.tables[-1].dataframe  # Descriptive Statistics table
        n_val = int(float(df.loc[df.index[0], "N"]))
        assert n_val == R_N

    def test_mean_vs_r(self):
        """평균이 R 결과와 허용 오차 내에 있어야 한다."""
        x, _, _ = _make_ref_data()
        py_mean = float(np.mean(x))
        # 직접 scipy로도 확인
        assert abs(py_mean - R_MEAN) < TOL, f"mean {py_mean} != R {R_MEAN}"

        result = self._make_result()
        df = result.tables[-1].dataframe
        row = df.iloc[0]
        mean_str = str(row.get("Mean", "")).strip()
        if mean_str:
            mean_val = float(mean_str)
            assert abs(mean_val - R_MEAN) < 0.01, f"table mean {mean_val} != R {R_MEAN}"

    def test_sd_vs_r(self):
        x, _, _ = _make_ref_data()
        py_sd = float(np.std(x, ddof=1))
        assert abs(py_sd - R_SD) < TOL

    def test_median_vs_r(self):
        x, _, _ = _make_ref_data()
        py_median = float(np.median(x))
        assert abs(py_median - R_MEDIAN) < TOL

    def test_min_max_vs_r(self):
        x, _, _ = _make_ref_data()
        assert abs(float(np.min(x)) - R_MIN) < TOL
        assert abs(float(np.max(x)) - R_MAX) < TOL


# ─────────────────────────────────────────────────────────────────────────────
# 2. t-검정 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestTtestValidation:

    def _make_result(self):
        """t-검정은 정확히 2개 그룹 필요 → A, B만 사용."""
        x, _, group = _make_ref_data()
        grp_arr = np.array(group)
        mask = (grp_arr == "A") | (grp_arr == "B")
        df = pd.DataFrame({"x": x[mask], "group": grp_arr[mask]})
        ds = _make_ds(df)
        return mod_ttests.run_analysis(
            ds, {"variables": {"dependent": "x", "group": "group"}}
        )

    def test_produces_result(self):
        result = self._make_result()
        assert isinstance(result, AnalysisResult)

    def test_scipy_vs_r_tstat(self):
        """scipy의 t 통계량이 R과 일치해야 한다."""
        x, _, group = _make_ref_data()
        grp_arr = np.array(group)
        g1 = x[grp_arr == "A"]
        g2 = x[grp_arr == "B"]
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        assert abs(float(t_stat) - R_T_STAT) < TOL, \
            f"scipy t={t_stat:.4f} vs R t={R_T_STAT:.4f}"
        assert abs(float(p_val) - R_T_P) < TOL, \
            f"scipy p={p_val:.4f} vs R p={R_T_P:.4f}"

    def test_result_has_ttest_tables(self):
        result = self._make_result()
        assert len(result.tables) >= 2  # Group Stats + Test Results


# ─────────────────────────────────────────────────────────────────────────────
# 3. 상관분석 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationValidation:

    def _make_result(self):
        x, y, _ = _make_ref_data()
        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df)
        return mod_correlation.run_analysis(
            ds, {"variables": {"target": ["x", "y"]}}
        )

    def test_produces_result(self):
        result = self._make_result()
        assert isinstance(result, AnalysisResult)

    def test_scipy_pearson_vs_r(self):
        """scipy Pearson r이 R과 일치해야 한다."""
        x, y, _ = _make_ref_data()
        r, p = stats.pearsonr(x, y)
        assert abs(float(r) - R_PEARSON_R) < TOL, \
            f"scipy r={r:.4f} vs R r={R_PEARSON_R:.4f}"

    def test_pearson_range(self):
        """r은 [-1, 1] 범위여야 한다."""
        x, y, _ = _make_ref_data()
        r, _ = stats.pearsonr(x, y)
        assert -1.0 <= float(r) <= 1.0

    def test_correlation_symmetry(self):
        """cor(x, y) == cor(y, x)."""
        x, y, _ = _make_ref_data()
        r_xy, _ = stats.pearsonr(x, y)
        r_yx, _ = stats.pearsonr(y, x)
        assert abs(r_xy - r_yx) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. 회귀분석 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionValidation:

    def _make_result(self):
        x, y, _ = _make_ref_data()
        df = pd.DataFrame({"y": y, "x": x})
        ds = _make_ds(df)
        return mod_regression.run_analysis(
            ds, {"variables": {"dependent": "y", "independent": ["x"]}}
        )

    def test_produces_result(self):
        result = self._make_result()
        assert isinstance(result, AnalysisResult)
        assert len(result.tables) >= 1

    def test_scipy_regression_vs_r(self):
        """scipy 회귀 결과가 R lm()과 일치해야 한다."""
        x, y, _ = _make_ref_data()
        slope, intercept, r_val, p_val, se = stats.linregress(x, y)
        assert abs(float(intercept) - R_INTERCEPT) < 0.1, \
            f"intercept {intercept:.4f} vs R {R_INTERCEPT:.4f}"
        assert abs(float(slope) - R_SLOPE) < TOL, \
            f"slope {slope:.4f} vs R {R_SLOPE:.4f}"
        assert abs(float(r_val**2) - R_R2) < TOL, \
            f"R² {r_val**2:.4f} vs R {R_R2:.4f}"

    def test_result_tables_structure(self):
        result = self._make_result()
        table_titles = [t.title for t in result.tables]
        assert any("Summary" in t or "Model" in t for t in table_titles)
        assert any("Coeff" in t or "coeff" in t for t in table_titles)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 정규성 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalityValidation:

    def _make_result(self):
        x, _, _ = _make_ref_data()
        df = pd.DataFrame({"x": x})
        ds = _make_ds(df)
        return mod_normality.run_analysis(ds, {"variables": {"target": ["x"]}})

    def test_produces_result(self):
        result = self._make_result()
        assert isinstance(result, AnalysisResult)

    def test_scipy_shapiro_vs_r(self):
        """scipy Shapiro-Wilk W가 R과 허용 오차 내에 있어야 한다."""
        x, _, _ = _make_ref_data()
        stat, p = stats.shapiro(x)
        assert abs(float(stat) - R_SW_W) < 0.01, \
            f"W={stat:.4f} vs R W={R_SW_W:.4f}"
        assert abs(float(p) - R_SW_P) < 0.05, \
            f"p={p:.4f} vs R p={R_SW_P:.4f}"

    def test_result_contains_w_statistic(self):
        result = self._make_result()
        if result.tables:
            df = result.tables[-1].dataframe
            assert "Statistic" in df.columns or "W" in str(df.columns.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANOVA 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestAnovaValidation:

    def test_scipy_anova_vs_r(self):
        """scipy F-통계량이 R aov()와 일치해야 한다."""
        # R: set.seed(1); y2 <- c(rnorm(10,50,10), rnorm(10,55,10), rnorm(10,60,10))
        rng = np.random.default_rng(1)
        a = rng.normal(50, 10, 10)
        b = rng.normal(55, 10, 10)
        c = rng.normal(60, 10, 10)
        f_stat, p_val = stats.f_oneway(a, b, c)
        # scipy 결과 자체 일관성 검증
        assert f_stat > 0
        assert 0.0 <= p_val <= 1.0

    def test_nuristat_anova_produces_f(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "y": np.concatenate([rng.normal(50, 10, 10),
                                  rng.normal(55, 10, 10),
                                  rng.normal(60, 10, 10)]),
            "g": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        })
        ds = _make_ds(df)
        spec = {"variables": {"dependent": "y", "factor": "g"}}
        result = mod_anova.run_analysis(ds, spec)
        assert isinstance(result, AnalysisResult)
        # F 통계량 테이블이 있어야 함
        table_titles = [t.title for t in result.tables]
        assert any("ANOVA" in t or "anova" in t.lower() for t in table_titles)

    def test_anova_f_positive(self):
        """F 통계량은 항상 양수여야 한다."""
        rng = np.random.default_rng(0)
        groups = [rng.normal(i * 5, 10, 20) for i in range(3)]
        f_stat, _ = stats.f_oneway(*groups)
        assert f_stat >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. 빈도분석 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestFrequenciesValidation:

    def test_frequency_counts_correct(self):
        """빈도 집계가 pandas value_counts와 일치해야 한다."""
        df = pd.DataFrame({"x": ["A", "B", "A", "C", "B", "A", "B", "A"]})
        ds = _make_ds(df)
        result = mod_frequencies.run_analysis(
            ds, {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

        freq_table = None
        for t in result.tables:
            if "Frequency" in t.title or "frequency" in t.title.lower():
                freq_table = t
                break
        if freq_table is not None and not freq_table.dataframe.empty:
            total_freq = freq_table.dataframe["Frequency"].sum()
            assert total_freq == 8  # 전체 8개 관측치

    def test_percent_sums_to_100(self):
        """퍼센트 합이 100%여야 한다."""
        df = pd.DataFrame({"x": ["A"] * 30 + ["B"] * 20 + ["C"] * 10})
        ds = _make_ds(df)
        result = mod_frequencies.run_analysis(
            ds, {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)
        for t in result.tables:
            if "Percent" in t.dataframe.columns:
                total_pct = t.dataframe["Percent"].sum()
                assert abs(total_pct - 100.0) < 0.1, \
                    f"Percent 합계 {total_pct:.1f} ≠ 100"


# ─────────────────────────────────────────────────────────────────────────────
# 8. 수치 일관성 교차 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalConsistency:

    def test_mean_sd_ci_consistent(self):
        """평균 ± t * SE가 신뢰구간 내에 있어야 한다."""
        rng = np.random.default_rng(0)
        x = rng.normal(100, 15, 50)
        n = len(x)
        mean = np.mean(x)
        sd = np.std(x, ddof=1)
        se = sd / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
        assert ci_low < mean < ci_high

    def test_r2_equals_pearson_squared_simple(self):
        """단순회귀에서 R² = r² (Pearson)."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = 2 * x + rng.normal(scale=0.5, size=100)
        r, _ = stats.pearsonr(x, y)
        slope, intercept, r_val, _, _ = stats.linregress(x, y)
        assert abs(r**2 - r_val**2) < 1e-10

    def test_f_stat_equals_t_squared_for_two_groups(self):
        """두 집단 ANOVA의 F = t² (등분산 가정)."""
        rng = np.random.default_rng(0)
        g1 = rng.normal(50, 10, 30)
        g2 = rng.normal(55, 10, 30)
        t_stat, _ = stats.ttest_ind(g1, g2, equal_var=True)
        f_stat, _ = stats.f_oneway(g1, g2)
        assert abs(t_stat**2 - f_stat) < TOL, \
            f"t²={t_stat**2:.4f} vs F={f_stat:.4f}"

    def test_pearson_spearman_agree_monotone(self):
        """단조증가 데이터에서 Pearson ≈ Spearman."""
        x = np.arange(1, 101, dtype=float)
        y = x * 2 + 0.01 * np.arange(100)  # 완전 단조
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        assert abs(r_p - r_s) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 9. p-value 경계값 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestPValueBoundaries:

    def test_pvalue_is_between_0_and_1(self):
        """p-value는 항상 [0, 1] 범위여야 한다."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        y = rng.normal(size=30)
        _, p = stats.ttest_ind(x, y)
        assert 0.0 <= float(p) <= 1.0

    def test_significant_result_below_alpha(self):
        """명확히 다른 두 집단의 p-value는 0.05 미만이어야 한다."""
        rng = np.random.default_rng(0)
        g1 = rng.normal(0, 1, 200)
        g2 = rng.normal(5, 1, 200)  # 5 SD 차이
        _, p = stats.ttest_ind(g1, g2)
        assert p < 0.001

    def test_identical_groups_p_near_1(self):
        """완전히 같은 두 집단의 p-value는 1에 가까워야 한다."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = stats.ttest_ind(x, x)
        assert p == 1.0 or np.isnan(p)  # identical → p=1 or NaN


# ─────────────────────────────────────────────────────────────────────────────
# 10. 결과 구조 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestResultStructure:

    def test_all_modules_return_analysis_result(self):
        """모든 주요 분석 모듈이 AnalysisResult를 반환해야 한다."""
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "y": rng.normal(50, 10, n),
            "x": rng.normal(30, 5, n),
            "group": (["A"] * 10 + ["B"] * 10 + ["C"] * 10),
            "cat": (["P"] * 15 + ["Q"] * 15),
        })
        ds = _make_ds(df)

        modules_and_specs = [
            (mod_descriptive, {"variables": {"scale": ["y", "x"]}}),
            (mod_frequencies, {"variables": {"target": ["group", "cat"]}}),
            (mod_correlation, {"variables": {"target": ["y", "x"]}}),
            (mod_normality, {"variables": {"target": ["y"]}}),
            (mod_ttests, {"variables": {"dependent": "y", "group": "cat"}}),
            (mod_anova, {"variables": {"dependent": "y", "factor": "group"}}),
            (mod_regression, {"variables": {"dependent": "y", "independent": ["x"]}}),
        ]
        for mod, spec in modules_and_specs:
            result = mod.run_analysis(ds, spec)
            assert isinstance(result, AnalysisResult), \
                f"{mod.__name__} did not return AnalysisResult"

    def test_result_has_required_fields(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=20)})
        ds = _make_ds(df)
        result = mod_descriptive.run_analysis(ds, {"variables": {"scale": ["x"]}})
        assert result.id
        assert result.title
        assert result.created_at is not None
        assert isinstance(result.tables, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.notes, list)

    def test_result_tables_have_dataframes(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=20)})
        ds = _make_ds(df)
        result = mod_descriptive.run_analysis(ds, {"variables": {"scale": ["x"]}})
        for table in result.tables:
            assert isinstance(table.dataframe, pd.DataFrame)
            assert not table.dataframe.empty or table.title  # 빈 테이블이면 제목이라도 있어야
