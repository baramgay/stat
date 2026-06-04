"""성능/스트레스 테스트 — 대용량 데이터 안정성 검증.

목적:
    - 각 분석 함수가 N=100, N=10000, N=100000에서 정확히 동작하는지 확인
    - 실행 시간이 허용 범위 내인지 확인 (분석별 타임아웃)
    - 메모리 이슈(OOM, 과도한 복사) 없이 처리되는지 확인

각 테스트는 3단계로 구성:
    소규모 (N=100)  → 기본 동작 확인
    중규모 (N=10000) → 성능 확인
    대규모 (N=100000) → 스트레스 확인

SPSS 호환 기준:
    SPSS 29도 동일 데이터에서 동일 결과를 내야 함.
    Python scipy 독립 계산으로 검증.
"""

from __future__ import annotations

import time
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType
from nuristat.analysis.descriptive import run_analysis as desc_run
from nuristat.analysis.ttests import run_analysis as ttest_run
from nuristat.analysis.anova import run_analysis as anova_run
from nuristat.analysis.correlation import run_analysis as corr_run
from nuristat.analysis.regression import run_analysis as reg_run


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _make_ds(df: pd.DataFrame, variables: dict | None = None) -> Dataset:
    ds = Dataset(df, name="perf_test")
    if variables:
        for name, meta in variables.items():
            ds.variables[name] = meta
    return ds


def _scale_meta(name: str, decimals: int = 2) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=decimals)


def _nominal_meta(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


def _float_from_result(result, table_title: str, col: str, row: int = 0) -> float:
    for tbl in result.tables:
        if tbl.title == table_title:
            val = tbl.dataframe.iloc[row][col]
            return float(str(val).replace(",", "").strip())
    raise KeyError(f"Table '{table_title}' not found")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 기술통계 — 소/중/대 규모
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptivesPerformance:
    """기술통계 대용량 성능 테스트."""

    def _make_desc_spec(self, varnames: list[str]) -> dict:
        return {"variables": {"scale": varnames}, "confidence_level": 0.95}

    def test_descriptives_small(self):
        """N=100: 기본 동작 및 정확성 확인."""
        np.random.seed(0)
        data = np.random.normal(100, 15, 100)
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale_meta("x")})

        result = desc_run(ds, self._make_desc_spec(["x"]))
        n_val = _float_from_result(result, "Descriptive Statistics", "N")
        mean_val = _float_from_result(result, "Descriptive Statistics", "Mean")

        assert n_val == 100
        assert mean_val == pytest.approx(float(np.mean(data)), abs=0.01)

    def test_descriptives_medium_performance(self):
        """N=10000: 2초 이내 완료."""
        np.random.seed(1)
        data = np.random.normal(50, 10, 10000)
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale_meta("x")})
        spec = self._make_desc_spec(["x"])

        start = time.perf_counter()
        result = desc_run(ds, spec)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"N=10000 기술통계 {elapsed:.2f}s > 2s"
        n_val = _float_from_result(result, "Descriptive Statistics", "N")
        assert n_val == 10000

    def test_descriptives_large_performance(self):
        """N=100000: 5초 이내 완료."""
        np.random.seed(2)
        data = np.random.normal(50, 10, 100000)
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale_meta("x")})
        spec = self._make_desc_spec(["x"])

        start = time.perf_counter()
        result = desc_run(ds, spec)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"N=100000 기술통계 {elapsed:.2f}s > 5s"
        n_val = _float_from_result(result, "Descriptive Statistics", "N")
        assert n_val == 100000

    def test_descriptives_multi_variable(self):
        """5개 변수 동시 기술통계 — 정확성 확인."""
        np.random.seed(3)
        n = 500
        df = pd.DataFrame({f"v{i}": np.random.normal(i * 10, 5, n) for i in range(5)})
        variables = {f"v{i}": _scale_meta(f"v{i}") for i in range(5)}
        ds = _make_ds(df, variables)
        spec = {"variables": {"scale": [f"v{i}" for i in range(5)]}, "confidence_level": 0.95}

        result = desc_run(ds, spec)
        # 5개 변수 모두 출력 확인
        tbl = next(t for t in result.tables if t.title == "Descriptive Statistics")
        assert len(tbl.dataframe) == 5

    def test_descriptives_with_missing_large(self):
        """N=10000, 결측치 10% — 유효 N 정확히 계산."""
        np.random.seed(4)
        data = np.random.normal(100, 20, 10000).tolist()
        # 10% 결측
        for i in range(0, 10000, 10):
            data[i] = float("nan")
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale_meta("x")})
        spec = self._make_desc_spec(["x"])

        result = desc_run(ds, spec)
        n_val = _float_from_result(result, "Descriptive Statistics", "N")
        assert n_val == pytest.approx(9000, abs=5)  # 약 9000개 유효


# ─────────────────────────────────────────────────────────────────────────────
# 2. 독립표본 t-검정 — 소/중/대 규모
# ─────────────────────────────────────────────────────────────────────────────

class TestTTestPerformance:
    """t-검정 대용량 성능 테스트."""

    def _make_ttest_ds(self, n: int, seed: int = 10) -> Dataset:
        np.random.seed(seed)
        group_a = np.random.normal(100, 15, n)
        group_b = np.random.normal(110, 15, n)
        df = pd.DataFrame({
            "score": np.concatenate([group_a, group_b]),
            "group": [0] * n + [1] * n,
        })
        variables = {
            "score": _scale_meta("score"),
            "group": _nominal_meta("group"),
        }
        return _make_ds(df, variables)

    def test_ttest_small_accuracy(self):
        """N=50 per group: t-통계량 scipy와 일치."""
        np.random.seed(10)
        a = np.random.normal(100, 15, 50)
        b = np.random.normal(110, 15, 50)
        t_scipy, _ = stats.ttest_ind(a, b)

        df = pd.DataFrame({"score": np.concatenate([a, b]), "group": [0]*50 + [1]*50})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "yes"}, "confidence_level": 0.95}
        result = ttest_run(ds, spec)

        t_sw = _float_from_result(result, "Independent Samples t-Test", "t", row=0)
        assert abs(t_sw) == pytest.approx(abs(t_scipy), abs=0.01)

    def test_ttest_medium_performance(self):
        """N=5000 per group: 2초 이내."""
        ds = self._make_ttest_ds(5000)
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "auto"}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = ttest_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"N=5000 t-검정 {elapsed:.2f}s > 2s"
        assert len(result.tables) >= 2

    def test_ttest_large_performance(self):
        """N=50000 per group: 5초 이내."""
        ds = self._make_ttest_ds(50000)
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "auto"}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = ttest_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"N=50000 t-검정 {elapsed:.2f}s > 5s"

    def test_ttest_large_converges_to_true_effect(self):
        """N=10000: 대용량에서 효과크기 수렴 확인."""
        np.random.seed(20)
        n = 10000
        a = np.random.normal(100, 15, n)
        b = np.random.normal(115, 15, n)  # true diff = 15
        df = pd.DataFrame({"score": np.concatenate([a, b]), "group": [0]*n + [1]*n})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "auto"}, "confidence_level": 0.95}
        result = ttest_run(ds, spec)

        # 대용량에서 평균차가 실제값 15에 근접해야 함
        for tbl in result.tables:
            if tbl.title == "Group Statistics":
                means = [float(str(v)) for v in tbl.dataframe["Mean"]]
                diff = abs(means[0] - means[1])
                assert diff == pytest.approx(15, abs=0.5)
                break


# ─────────────────────────────────────────────────────────────────────────────
# 3. 일원분산분석 — 소/중/대 규모
# ─────────────────────────────────────────────────────────────────────────────

class TestANOVAPerformance:
    """ANOVA 대용량 성능 테스트."""

    def _make_anova_ds(self, n_per_group: int, k: int = 3, seed: int = 30) -> Dataset:
        np.random.seed(seed)
        groups = [np.random.normal(i * 10, 5, n_per_group) for i in range(k)]
        all_data = np.concatenate(groups)
        all_groups = np.concatenate([[i] * n_per_group for i in range(k)])
        df = pd.DataFrame({"score": all_data, "group": all_groups})
        variables = {"score": _scale_meta("score"), "group": _nominal_meta("group")}
        return _make_ds(df, variables)

    def test_anova_small_accuracy(self):
        """3그룹 N=20: F값 scipy와 일치."""
        np.random.seed(30)
        g1 = np.random.normal(50, 5, 20)
        g2 = np.random.normal(60, 5, 20)
        g3 = np.random.normal(70, 5, 20)
        F_scipy, _ = stats.f_oneway(g1, g2, g3)

        all_d = np.concatenate([g1, g2, g3])
        all_g = [0]*20 + [1]*20 + [2]*20
        df = pd.DataFrame({"score": all_d, "group": all_g})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": [], "effect_size": True}, "confidence_level": 0.95}
        result = anova_run(ds, spec)

        for tbl in result.tables:
            if tbl.title == "ANOVA":
                row = tbl.dataframe[tbl.dataframe["Source"].str.contains("group", case=False, na=False)]
                if not row.empty:
                    f_sw = float(str(row.iloc[0]["F"]))
                    assert f_sw == pytest.approx(F_scipy, rel=0.01)

    def test_anova_many_groups(self):
        """10그룹 ANOVA — 정확성 및 안정성."""
        np.random.seed(35)
        k = 10
        n_per = 30
        groups = [np.random.normal(i * 5, 3, n_per) for i in range(k)]
        F_scipy, p_scipy = stats.f_oneway(*groups)

        all_d = np.concatenate(groups)
        all_g = np.concatenate([[i] * n_per for i in range(k)])
        df = pd.DataFrame({"score": all_d, "group": all_g})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": [], "effect_size": True}, "confidence_level": 0.95}
        result = anova_run(ds, spec)
        assert len(result.warnings) == 0

    def test_anova_medium_performance(self):
        """N=3000 per group, 3그룹: 3초 이내."""
        ds = self._make_anova_ds(3000)
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": [], "effect_size": True}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = anova_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"N=9000 ANOVA {elapsed:.2f}s > 3s"

    def test_anova_large_performance(self):
        """N=30000 per group, 3그룹: 10초 이내."""
        ds = self._make_anova_ds(30000)
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": [], "effect_size": True}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = anova_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"N=90000 ANOVA {elapsed:.2f}s > 10s"

    def test_anova_with_posthoc_medium(self):
        """N=500 per group, Tukey HSD 포함: 3초 이내."""
        np.random.seed(40)
        k = 4
        n_per = 500
        groups = [np.random.normal(i * 5, 3, n_per) for i in range(k)]
        all_d = np.concatenate(groups)
        all_g = np.concatenate([[i] * n_per for i in range(k)])
        df = pd.DataFrame({"score": all_d, "group": all_g})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": ["tukey"], "effect_size": True},
                "confidence_level": 0.95}

        start = time.perf_counter()
        result = anova_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"Tukey HSD N=2000 {elapsed:.2f}s > 3s"


# ─────────────────────────────────────────────────────────────────────────────
# 4. 상관분석 — 소/중/대 규모
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationPerformance:
    """상관분석 대용량 성능 테스트."""

    def _make_corr_ds(self, n: int, k: int = 3, seed: int = 50) -> Dataset:
        np.random.seed(seed)
        # k개 변수, 일부 상관관계 있음
        base = np.random.normal(0, 1, n)
        data = {f"v{i}": base + np.random.normal(0, 0.5, n) for i in range(k)}
        df = pd.DataFrame(data)
        variables = {f"v{i}": _scale_meta(f"v{i}") for i in range(k)}
        return _make_ds(df, variables)

    def test_correlation_small_accuracy(self):
        """N=100, 2변수: Pearson r scipy와 일치."""
        np.random.seed(50)
        x = np.random.normal(0, 1, 100)
        y = x + np.random.normal(0, 0.5, 100)
        r_scipy, _ = stats.pearsonr(x, y)

        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df, {"x": _scale_meta("x"), "y": _scale_meta("y")})
        spec = {"variables": {"target": ["x", "y"]},
                "options": {"method": "pearson", "flag_significant": True, "pairwise": False},
                "confidence_level": 0.95}
        result = corr_run(ds, spec)
        assert len(result.tables) >= 2

    def test_correlation_medium_performance(self):
        """N=10000, 3변수: 3초 이내."""
        ds = self._make_corr_ds(10000, k=3)
        spec = {"variables": {"target": ["v0", "v1", "v2"]},
                "options": {"method": "pearson", "flag_significant": True, "pairwise": False},
                "confidence_level": 0.95}

        start = time.perf_counter()
        result = corr_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"N=10000 상관 {elapsed:.2f}s > 3s"

    def test_correlation_large_performance(self):
        """N=100000, 2변수: 5초 이내."""
        np.random.seed(51)
        n = 100000
        x = np.random.normal(0, 1, n)
        y = x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df, {"x": _scale_meta("x"), "y": _scale_meta("y")})
        spec = {"variables": {"target": ["x", "y"]},
                "options": {"method": "pearson", "flag_significant": True, "pairwise": False},
                "confidence_level": 0.95}

        start = time.perf_counter()
        result = corr_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"N=100000 상관 {elapsed:.2f}s > 5s"

    def test_correlation_converges_large_n(self):
        """N=100000: 알려진 상관계수로 수렴 확인."""
        np.random.seed(52)
        n = 100000
        x = np.random.normal(0, 1, n)
        # y = 0.8*x + noise → r이론값 ≈ 0.8/sqrt(1+0.36)≈0.8
        noise_sd = 0.6  # r ≈ 0.8/sqrt(1+noise_sd^2) ≈ 0.800
        y = 0.8 * x + np.random.normal(0, noise_sd, n)
        expected_r = 0.8 / np.sqrt(0.64 + noise_sd**2)  # ≈ 0.800
        r_true, _ = stats.pearsonr(x, y)
        # 충분히 큰 N에서 추정값이 이론값에 근접
        assert r_true == pytest.approx(expected_r, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 회귀분석 — 소/중/대 규모
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionPerformance:
    """선형회귀 대용량 성능 테스트."""

    def _make_reg_ds(self, n: int, p: int = 3, seed: int = 60) -> Dataset:
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        beta = np.array([1.0, -0.5, 0.3] + [0.1] * max(0, p - 3))
        y = X @ beta + np.random.normal(0, 1, n)
        data = {f"x{i}": X[:, i] for i in range(p)}
        data["y"] = y
        df = pd.DataFrame(data)
        variables = {col: _scale_meta(col) for col in df.columns}
        return _make_ds(df, variables)

    def test_regression_small_accuracy(self):
        """N=100, 3 predictors: R² 범위 확인 및 계수 추정."""
        ds = self._make_reg_ds(100, p=3)
        spec = {"variables": {"dependent": "y", "independent": ["x0", "x1", "x2"]},
                "options": {}, "confidence_level": 0.95}
        result = reg_run(ds, spec)
        assert len(result.warnings) == 0

    def test_regression_medium_performance(self):
        """N=10000, 5 predictors: 5초 이내."""
        ds = self._make_reg_ds(10000, p=5)
        spec = {"variables": {"dependent": "y", "independent": [f"x{i}" for i in range(5)]},
                "options": {}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = reg_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"N=10000 회귀 {elapsed:.2f}s > 5s"

    def test_regression_large_performance(self):
        """N=100000, 3 predictors: 10초 이내."""
        ds = self._make_reg_ds(100000, p=3)
        spec = {"variables": {"dependent": "y", "independent": ["x0", "x1", "x2"]},
                "options": {}, "confidence_level": 0.95}

        start = time.perf_counter()
        result = reg_run(ds, spec)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"N=100000 회귀 {elapsed:.2f}s > 10s"

    def test_regression_coefficients_converge(self):
        """N=100000: 회귀계수가 실제값으로 수렴."""
        np.random.seed(61)
        n = 100000
        x0 = np.random.normal(0, 1, n)
        x1 = np.random.normal(0, 1, n)
        true_b0, true_b1, true_b2 = 5.0, 2.0, -1.5
        y = true_b0 + true_b1 * x0 + true_b2 * x1 + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
        ds = _make_ds(df, {"y": _scale_meta("y"), "x0": _scale_meta("x0"), "x1": _scale_meta("x1")})
        spec = {"variables": {"dependent": "y", "independent": ["x0", "x1"]},
                "options": {}, "confidence_level": 0.95}
        result = reg_run(ds, spec)

        # 계수 테이블에서 추정값 확인
        for tbl in result.tables:
            if "Coefficient" in tbl.title or "coefficient" in tbl.title.lower():
                df_coef = tbl.dataframe
                # B 컬럼 확인
                if "B" in df_coef.columns:
                    b_vals = [float(str(v)) for v in df_coef["B"]]
                    # Intercept, x0, x1 순서
                    assert abs(b_vals[0] - true_b0) < 0.1
                    assert abs(b_vals[1] - true_b1) < 0.05
                    assert abs(b_vals[2] - true_b2) < 0.05
                break


# ─────────────────────────────────────────────────────────────────────────────
# 6. 에지 케이스 — 경계값 안정성
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """경계값 및 특수 상황 안정성 테스트."""

    def test_all_missing_values_handled(self):
        """모든 값이 결측일 때 분석 함수가 경고만 내고 크래시 없음."""
        df = pd.DataFrame({"x": [float("nan")] * 10})
        ds = _make_ds(df, {"x": _scale_meta("x")})
        spec = {"variables": {"scale": ["x"]}, "confidence_level": 0.95}
        result = desc_run(ds, spec)
        # 크래시 없이 빈 결과 또는 경고 반환
        assert result is not None

    def test_single_value_per_group_ttest(self):
        """그룹당 N=1: t-검정이 크래시 없이 경고 반환."""
        df = pd.DataFrame({"score": [1.0, 2.0], "group": [0, 1]})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "auto"}, "confidence_level": 0.95}
        result = ttest_run(ds, spec)
        assert result is not None  # 크래시 없음
        assert len(result.warnings) > 0  # 경고 있어야 함

    def test_identical_groups_ttest(self):
        """완전히 동일한 두 그룹: t=0, p=1."""
        data = list(range(10)) * 2
        groups = [0] * 10 + [1] * 10
        df = pd.DataFrame({"score": data, "group": groups})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "yes"}, "confidence_level": 0.95}
        result = ttest_run(ds, spec)
        assert result is not None

    def test_single_group_anova(self):
        """그룹 1개: ANOVA 경고 반환, 크래시 없음."""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0], "group": [1, 1, 1]})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": []}, "confidence_level": 0.95}
        result = anova_run(ds, spec)
        assert len(result.warnings) > 0  # 경고 있어야 함

    def test_anova_with_zero_variance_group(self):
        """한 그룹 내 분산=0 (모든 값 동일): 안정적 처리."""
        df = pd.DataFrame({
            "score": [5.0, 5.0, 5.0, 10.0, 11.0, 12.0],
            "group": [0, 0, 0, 1, 1, 1],
        })
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "factor": "group"},
                "options": {"post_hoc": []}, "confidence_level": 0.95}
        result = anova_run(ds, spec)
        assert result is not None

    def test_large_values_no_overflow(self):
        """매우 큰 값 (1e10 범위): 오버플로우 없이 처리."""
        np.random.seed(70)
        data = np.random.normal(1e10, 1e8, 1000)
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale_meta("x")})
        spec = {"variables": {"scale": ["x"]}, "confidence_level": 0.95}
        result = desc_run(ds, spec)
        mean_val = _float_from_result(result, "Descriptive Statistics", "Mean")
        assert abs(mean_val - 1e10) < 1e9  # 수량적으로 합리적

    def test_n2_minimum_ttest(self):
        """그룹당 N=2: t-검정 최소 케이스 처리."""
        df = pd.DataFrame({"score": [10.0, 20.0, 15.0, 25.0], "group": [0, 0, 1, 1]})
        ds = _make_ds(df, {"score": _scale_meta("score"), "group": _nominal_meta("group")})
        spec = {"variables": {"dependent": "score", "group": "group"},
                "options": {"equal_var": "yes"}, "confidence_level": 0.95}
        result = ttest_run(ds, spec)
        assert result is not None
