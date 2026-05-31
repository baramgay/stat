"""Round 11: MissingPolicy 키 수정 + 예외 경로 모킹 커버리지 보완."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


# ─────────────────────────────────────────────────────────────────────────────
# MissingPolicy fallback — top-level spec key (lines 75-76, 83-84, 88-89, 91-92, 91-92, 95-96)
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingPolicyTopLevelKey:
    """spec['missing_policy'] 상위 레벨 키로 잘못된 값 → ValueError → listwise 폴백."""

    def _mnl_ds(self):
        rng = np.random.default_rng(0)
        n = 90
        y = [str(i % 3) for i in range(n)]
        df = pd.DataFrame({"y": y, "x1": rng.normal(0, 1, n)})
        ds = Dataset(df, name="mnl_mp")
        ds.variables["y"] = _nominal("y")
        ds.variables["x1"] = _scale("x1")
        return ds

    def test_multinomial_logistic_missing_policy_fallback(self):
        from statworkbench.analysis.multinomial_logistic import run_analysis
        ds = self._mnl_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def _pca_ds(self):
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(4)})
        ds = Dataset(df, name="pca_mp")
        for i in range(4):
            ds.variables[f"x{i}"] = _scale(f"x{i}")
        return ds

    def test_pca_missing_policy_fallback(self):
        from statworkbench.analysis.pca import run_analysis
        ds = self._pca_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"items": ["x0", "x1", "x2", "x3"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def _twa_ds(self):
        rng = np.random.default_rng(2)
        rows = []
        for a in ["A1", "A2"]:
            for b in ["B1", "B2"]:
                rows.extend({"dep": float(v), "fa": a, "fb": b}
                             for v in rng.normal(0, 1, 10))
        df = pd.DataFrame(rows)
        ds = Dataset(df, name="twa_mp")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"] = _nominal("fa")
        ds.variables["fb"] = _nominal("fb")
        return ds

    def test_two_way_anova_missing_policy_fallback(self):
        from statworkbench.analysis.two_way_anova import run_analysis
        ds = self._twa_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def _mixed_ds(self):
        rng = np.random.default_rng(3)
        n = 20
        df = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10,
            "t1": rng.normal(0, 1, n),
            "t2": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="mixed_mp")
        ds.variables["group"] = _nominal("group")
        ds.variables["t1"] = _scale("t1")
        ds.variables["t2"] = _scale("t2")
        return ds

    def test_mixed_anova_missing_policy_fallback(self):
        from statworkbench.analysis.mixed_anova import run_analysis
        ds = self._mixed_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"between": "group", "within": ["t1", "t2"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def _manova_ds(self):
        rng = np.random.default_rng(4)
        n = 40
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "y1": rng.normal(0, 1, n),
            "y2": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="manova_mp")
        ds.variables["group"] = _nominal("group")
        ds.variables["y1"] = _scale("y1")
        ds.variables["y2"] = _scale("y2")
        return ds

    def test_manova_missing_policy_fallback(self):
        from statworkbench.analysis.manova import run_analysis
        ds = self._manova_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"dependents": ["y1", "y2"], "factor": "group"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def _ancova_ds(self):
        rng = np.random.default_rng(5)
        n = 30
        df = pd.DataFrame({
            "group": ["A"] * 15 + ["B"] * 15,
            "y": rng.normal(0, 1, n),
            "cov": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="ancova_mp")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        ds.variables["cov"] = _scale("cov")
        return ds

    def test_ancova_missing_policy_fallback(self):
        from statworkbench.analysis.ancova import run_analysis
        ds = self._ancova_ds()
        spec = {
            "missing_policy": "invalid_xyz",
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# multinomial_logistic.py L117: ref_cat not in categories → categories[-1]
# ─────────────────────────────────────────────────────────────────────────────

class TestMnlRefCatFallback:
    def test_reference_key_not_in_categories(self):
        """L117: options['reference'] = 'ghost' → 마지막 범주로 폴백."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        rng = np.random.default_rng(10)
        n = 90
        df = pd.DataFrame({
            "y": [str(i % 3) for i in range(n)],
            "x1": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="mnl_ref")
        ds.variables["y"] = _nominal("y")
        ds.variables["x1"] = _scale("x1")
        result = run_analysis(ds, {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {"reference": "ghost_category"},
        })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# pca.py — L205-206 (LinAlgError), L220-221 (Exception), L279-283 (rotation Exception)
#         L312-314 (scree plot Exception)
# ─────────────────────────────────────────────────────────────────────────────

class TestPcaExceptionPaths:
    def _pca_ds(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(4)})
        ds = Dataset(df, name="pca_ex")
        for i in range(4):
            ds.variables[f"x{i}"] = _scale(f"x{i}")
        return ds

    def test_kmo_linalg_error(self):
        """L205-206: np.linalg.inv → LinAlgError → (nan, nan, 0, nan)."""
        from statworkbench.analysis.pca import _kmo_bartlett
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 3))
        N, p = X.shape[0], X.shape[1]
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
            kmo, chi2, df, pval = _kmo_bartlett(X, N, p)
        assert np.isnan(kmo)
        assert np.isnan(chi2)

    def test_kmo_general_exception(self):
        """L220-221: outer try → general Exception → (nan, nan, 0, nan)."""
        from statworkbench.analysis.pca import _kmo_bartlett
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (50, 3))
        N, p = X.shape[0], X.shape[1]
        with patch("numpy.corrcoef", side_effect=ValueError("bad corr")):
            kmo, chi2, df, pval = _kmo_bartlett(X, N, p)
        assert np.isnan(kmo)

    def test_rotation_lstsq_exception_returns_loadings(self):
        """L279-283: np.linalg.lstsq raises → except pass → return loadings."""
        from statworkbench.analysis.pca import _rotate
        rng = np.random.default_rng(3)
        loadings = rng.normal(0, 1, (6, 2))
        with patch("numpy.linalg.lstsq", side_effect=np.linalg.LinAlgError("fail")):
            result = _rotate(loadings, "promax")
        assert result.shape == loadings.shape

    def test_rotation_outer_exception_returns_loadings(self):
        """L282-283: outer exception → return original loadings."""
        from statworkbench.analysis.pca import _rotate
        rng = np.random.default_rng(4)
        loadings = rng.normal(0, 1, (6, 2))
        with patch("numpy.sign", side_effect=RuntimeError("unexpected")):
            result = _rotate(loadings, "promax")
        assert result.shape == loadings.shape

    def test_scree_plot_exception_returns_none(self):
        """L312-314: 스크리 플롯 생성 중 예외 → None 반환."""
        from statworkbench.analysis.pca import _scree_plot
        eigenvalues = np.array([3.0, 1.5, 0.8, 0.3])
        with patch("io.BytesIO", side_effect=RuntimeError("io fail")):
            result = _scree_plot(eigenvalues, 2)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# two_way_anova.py — L201-202 (Levene exception), L461-464 (profile plot exception)
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoWayAnovaExceptions:
    def _ds(self):
        rng = np.random.default_rng(0)
        rows = []
        for a in ["A1", "A2", "A3"]:
            for b in ["B1", "B2"]:
                rows.extend({"dep": float(v), "fa": a, "fb": b}
                             for v in rng.normal(0, 1, 10))
        df = pd.DataFrame(rows)
        ds = Dataset(df, name="twa_ex")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"] = _nominal("fa")
        ds.variables["fb"] = _nominal("fb")
        return ds

    def test_levene_exception_path(self):
        """L201-202: Levene 검정 도중 예외 → 경고만 추가."""
        from statworkbench.analysis.two_way_anova import run_analysis
        from scipy import stats as scipy_stats
        ds = self._ds()
        with patch.object(scipy_stats, "levene", side_effect=RuntimeError("levene fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
                "options": {"levene": True},
            })
        assert any("Levene" in w for w in result.warnings)

    def test_profile_plot_exception_returns_none(self):
        """L461-464: 프로파일 플롯 생성 중 예외 → None 반환."""
        from statworkbench.analysis.two_way_anova import _profile_plot_two_way
        rng = np.random.default_rng(1)
        data = pd.DataFrame({
            "dep": rng.normal(0, 1, 30),
            "fa": ["A1"] * 10 + ["A2"] * 10 + ["A3"] * 10,
            "fb": ["B1", "B2"] * 15,
        })
        with patch("io.BytesIO", side_effect=RuntimeError("io fail")):
            result = _profile_plot_two_way(data, "dep", "fa", "fb")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# mixed_anova.py — L99-100 (too few cases), L296-298 (Mauchly exception)
#                  L402 (ms_s_within <= 0 → continue), L499-502 (plot exception)
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedAnovaExceptions:
    def _ds(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "group": ["A"] * (n // 2) + ["B"] * (n // 2),
            "t1": rng.normal(0, 1, n),
            "t2": rng.normal(0, 1, n),
            "t3": rng.normal(0, 1, n),
        })
        ds = Dataset(df, name="mixed_ex")
        ds.variables["group"] = _nominal("group")
        ds.variables["t1"] = _scale("t1")
        ds.variables["t2"] = _scale("t2")
        ds.variables["t3"] = _scale("t3")
        return ds

    def test_too_few_cases_warning(self):
        """L99-100: n_valid < 4 → 경고 반환."""
        from statworkbench.analysis.mixed_anova import run_analysis
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": ["A", "B", "A"],
            "t1": rng.normal(0, 1, 3),
            "t2": rng.normal(0, 1, 3),
        })
        ds = Dataset(df, name="tiny_mixed")
        ds.variables["group"] = _nominal("group")
        ds.variables["t1"] = _scale("t1")
        ds.variables["t2"] = _scale("t2")
        result = run_analysis(ds, {
            "variables": {"between": "group", "within": ["t1", "t2"]},
            "options": {},
        })
        assert any("적습니다" in w or "케이스" in w for w in result.warnings)

    def test_mauchly_exception_returns_default_eps(self):
        """L296-298: Mauchly 검정 도중 예외 → (1.0, 1.0) 반환."""
        from statworkbench.analysis.mixed_anova import _mauchly_test
        from statworkbench.analysis.result import AnalysisResult
        rng = np.random.default_rng(1)
        result_obj = AnalysisResult(id="test", title="test")
        n, k = 20, 3
        Y = rng.normal(0, 1, (n, k))
        group_labels = np.array(["A"] * 10 + ["B"] * 10)
        groups = ["A", "B"]
        with patch("numpy.cov", side_effect=RuntimeError("cov fail")):
            eps_gg, eps_hf = _mauchly_test(result_obj, Y, group_labels, groups, k, n)
        assert eps_gg == 1.0
        assert eps_hf == 1.0

    def test_bonferroni_between_ms_nan_continue(self):
        """L402: ms_s_within = nan → continue (쌍별 비교 건너뜀)."""
        from statworkbench.analysis.mixed_anova import _bonferroni_between
        from statworkbench.analysis.result import AnalysisResult
        rng = np.random.default_rng(2)
        result_obj = AnalysisResult(id="test", title="test")
        data = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "t1": rng.normal(0, 1, 30),
            "t2": rng.normal(0, 1, 30),
        })
        _bonferroni_between(
            result=result_obj,
            data=data,
            between_var="group",
            groups=["A", "B", "C"],
            within_vars=["t1", "t2"],
            ms_s_within=float("nan"),   # nan → continue 실행
            df_s_within=27,
            k=2,
            confidence_level=0.95,
            alpha=0.05,
        )
        # continue로 인해 테이블이 추가되지 않음
        assert len(result_obj.tables) == 0

    def test_profile_plot_exception_returns_none(self):
        """L499-502: 프로파일 플롯 생성 중 예외 → None 반환."""
        from statworkbench.analysis.mixed_anova import _profile_plot_mixed
        rng = np.random.default_rng(3)
        data = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10,
            "t1": rng.normal(0, 1, 20),
            "t2": rng.normal(0, 1, 20),
        })
        with patch("io.BytesIO", side_effect=RuntimeError("io fail")):
            result = _profile_plot_mixed(data, "group", ["t1", "t2"], "시점")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# ancova.py — L160-161 (Levene exception), L198-199 (Homogeneity exception)
#             L207-209 (ANCOVA model error)
# ─────────────────────────────────────────────────────────────────────────────

class TestAncovaExceptions:
    def _ds(self, n=60, n_groups=3):
        rng = np.random.default_rng(0)
        groups = [str(g % n_groups) for g in range(n)]
        cov = rng.normal(50, 10, n)
        y = [float(int(g) * 2 + 0.3 * cov[i] + rng.normal()) for i, g in enumerate(groups)]
        df = pd.DataFrame({"group": groups, "y": y, "cov": cov})
        ds = Dataset(df, name="ancova_ex")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        ds.variables["cov"] = _scale("cov")
        return ds

    def test_levene_exception_adds_warning(self):
        """L160-161: stats.levene 예외 → 경고 추가."""
        from statworkbench.analysis.ancova import run_analysis
        from scipy import stats as scipy_stats
        ds = self._ds()
        with patch.object(scipy_stats, "levene", side_effect=RuntimeError("levene fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
                "options": {"levene": True},
            })
        assert any("Levene" in w for w in result.warnings)

    def test_homogeneity_exception_adds_warning(self):
        """L198-199: 동질성 검정 OLS 예외 → 경고 추가."""
        from statworkbench.analysis.ancova import run_analysis
        from statsmodels.formula.api import ols
        ds = self._ds()
        call_count = [0]
        original_ols = ols

        def ols_side_effect(formula, data):
            call_count[0] += 1
            if call_count[0] == 1:  # 첫 번째 호출 = 동질성 검정
                raise RuntimeError("homog ols fail")
            return original_ols(formula, data)

        with patch("statworkbench.analysis.ancova.ols", side_effect=ols_side_effect):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
                "options": {"homogeneity_of_regression": True},
            })
        assert result is not None

    def test_ancova_model_error_adds_warning(self):
        """L207-209: ANCOVA OLS 모델 오류 → 경고 후 반환."""
        from statworkbench.analysis.ancova import run_analysis
        ds = self._ds()
        with patch("statworkbench.analysis.ancova.ols", side_effect=RuntimeError("model fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
                "options": {},
            })
        assert any("모델 오류" in w or "오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# manova.py — L335 (continue < 2 obs), L352-353 (Tukey exception fallback)
# ─────────────────────────────────────────────────────────────────────────────

class TestManovaPostHocPaths:
    def _ds_small_group(self):
        """그룹 중 하나가 관측치 1개 (< 2) → post-hoc continue."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 1,
            "y1": np.concatenate([rng.normal(0, 1, 20), rng.normal(1, 1, 20), [5.0]]),
            "y2": np.concatenate([rng.normal(0, 1, 20), rng.normal(1, 1, 20), [5.0]]),
        })
        ds = Dataset(df, name="manova_small")
        ds.variables["group"] = _nominal("group")
        ds.variables["y1"] = _scale("y1")
        ds.variables["y2"] = _scale("y2")
        return ds

    def test_post_hoc_small_group_continue(self):
        """L335: 그룹 관측치 < 2 → continue."""
        from statworkbench.analysis.manova import run_analysis
        ds = self._ds_small_group()
        result = run_analysis(ds, {
            "variables": {"dependents": ["y1", "y2"], "factor": "group"},
            "options": {"post_hoc": True, "post_hoc_method": "bonferroni"},
        })
        assert result is not None

    def test_tukey_exception_fallback(self):
        """L352-353: pairwise_tukeyhsd 예외 → bonferroni 폴백."""
        from statworkbench.analysis.manova import run_analysis
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "y1": rng.normal(0, 1, n),
            "y2": rng.normal(1, 1, n),
        })
        ds = Dataset(df, name="manova_tukey")
        ds.variables["group"] = _nominal("group")
        ds.variables["y1"] = _scale("y1")
        ds.variables["y2"] = _scale("y2")
        with patch("statsmodels.stats.multicomp.pairwise_tukeyhsd",
                   side_effect=RuntimeError("tukey fail")):
            result = run_analysis(ds, {
                "variables": {"dependents": ["y1", "y2"], "factor": "group"},
                "options": {"post_hoc": True, "post_hoc_method": "tukey"},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# explore.py — L461-462 (TypeError in sorted for non-comparable groups)
# ─────────────────────────────────────────────────────────────────────────────

class TestExploreTypeErrorInSorted:
    def test_non_comparable_groups_fallback(self):
        """L461-462: sorted() TypeError → list() fallback."""
        from statworkbench.analysis.explore import run_analysis
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "grp": [1] * 15 + [2] * 15,
        })
        ds = Dataset(df, name="explore_type")
        ds.variables["x"] = _scale("x")
        ds.variables["grp"] = _nominal("grp")
        # sorted()가 TypeError를 발생시키도록 patch
        import builtins
        original_sorted = builtins.sorted

        def sorted_raises_once(iterable, *args, **kwargs):
            items = list(iterable)
            if any(isinstance(v, (int, float)) for v in items):
                raise TypeError("not comparable")
            return original_sorted(iterable, *args, **kwargs)

        with patch("builtins.sorted", side_effect=sorted_raises_once):
            result = run_analysis(ds, {
                "variables": {"targets": ["x"], "factor": "grp"},
                "options": {},
            })
        assert result is not None
