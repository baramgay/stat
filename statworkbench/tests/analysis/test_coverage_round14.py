"""Round 14: 100% 커버리지 달성 — 잔여 미커버 라인 전체 보완.

대상 파일:
  app.py, main.py, visualization.py, regression.py, two_way_anova.py,
  survival_analysis.py, discriminant_analysis.py, result.py,
  multinomial_logistic.py, factor_analysis.py, cohens_kappa.py,
  cluster_analysis.py, io/import_wizard.py, repeated_measures_anova.py,
  registry.py, nonparametric.py, logistic_regression.py,
  ancova.py, text_mining.py, manova.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _binary(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_ds(df: pd.DataFrame, metas: dict) -> Dataset:
    ds = Dataset(df, name="test")
    for col, meta in metas.items():
        ds.variables[col] = meta
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# app.py — StatWorkbenchApp (QApplication + MainWindow 완전 모킹)
# ─────────────────────────────────────────────────────────────────────────────

class TestStatWorkbenchApp:
    def test_run_and_get_window(self):
        """app.py L8-39: StatWorkbenchApp.__init__, run(), get_window()."""
        mock_qapp = MagicMock()
        mock_qapp.return_value.exec.return_value = 0
        mock_window = MagicMock()

        # sys.modules 조작 없이 모듈 속성만 직접 패치 (numpy C 확장 재로딩 방지)
        with patch("statworkbench.app.QApplication", mock_qapp), \
             patch("statworkbench.app.MainWindow", mock_window):
            from statworkbench.app import StatWorkbenchApp
            app = StatWorkbenchApp()
            assert app.get_window() is None
            ret = app.run()
            assert ret == 0
            assert app.get_window() is not None


# ─────────────────────────────────────────────────────────────────────────────
# main.py — main() 성공/실패 + __name__ == "__main__" 블록
# ─────────────────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_success(self):
        """main.py L3,6,8-9,11-12,17-18: main() 정상 경로."""
        mock_app_cls = MagicMock()
        mock_app_cls.return_value.run.return_value = 0
        with patch("statworkbench.app.StatWorkbenchApp", mock_app_cls):
            from statworkbench.main import main
            result = main()
            assert result == 0

    def test_main_import_error(self):
        """main.py L12-15: ImportError → return 1."""
        with patch.dict("sys.modules", {"PySide6.QtWidgets": None}):
            from statworkbench.main import main
            ret = main()
            assert ret == 1


# ─────────────────────────────────────────────────────────────────────────────
# visualization.py L166-171: set_labels
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizationEngineSetLabels:
    def test_set_labels_with_labeled_variables(self):
        """L166-171: 레이블 있는 변수 처리."""
        from statworkbench.analysis.visualization import VisualizationEngine

        class FakeMeta:
            def __init__(self, label):
                self.label = label

        class FakeDataset:
            variables = {
                "height": FakeMeta("키 (cm)"),   # label != col → 저장
                "weight": FakeMeta("weight"),    # label == col → 무시
                "age": FakeMeta(None),           # label is None → 무시
            }

        viz = VisualizationEngine()
        viz.set_labels(FakeDataset())
        assert viz._labels == {"height": "키 (cm)"}

    def test_set_labels_none_dataset(self):
        """dataset=None 시 빈 dict."""
        from statworkbench.analysis.visualization import VisualizationEngine
        viz = VisualizationEngine()
        viz.set_labels(None)
        assert viz._labels == {}


# ─────────────────────────────────────────────────────────────────────────────
# result.py L67-69: ResultTable.to_html() 이미지 렌더링 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestResultTableHtmlException:
    def test_image_render_exception_returns_fallback(self):
        """L67-69: base64 인코딩 예외 → 폴백 HTML 반환."""
        from statworkbench.analysis.result import ResultTable
        bad_bytes = b"\x89PNG_invalid"
        df = pd.DataFrame([{"image_bytes": bad_bytes}])
        rt = ResultTable(
            title="test_img",
            dataframe=df,
            metadata={"type": "profile_plot"},
        )
        with patch("statworkbench.analysis.result.base64.b64encode",
                   side_effect=RuntimeError("encode fail")):
            html = rt.to_html()
        assert "이미지 렌더링 실패" in html


# ─────────────────────────────────────────────────────────────────────────────
# regression.py L346-347: 영향력 케이스 진단 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionInfluenceException:
    def _ds(self):
        rng = np.random.default_rng(42)
        n = 30
        x = rng.normal(0, 1, n)
        y = 2 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"x": x, "y": y})
        return _make_ds(df, {"x": _scale("x"), "y": _scale("y")})

    def test_influence_exception_logged(self):
        """L346-347: get_influence() 예외 → logger.warning만 발생, 결과 반환."""
        from statworkbench.analysis.regression import run_analysis
        from statsmodels.regression.linear_model import OLSResults

        ds = self._ds()
        with patch.object(OLSResults, "get_influence",
                          side_effect=RuntimeError("influence fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "independent": ["x"]},
                "options": {"influential_cases": True},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# regression.py L508-509: forward 단계 개별 변수 OLS 예외 → continue
# regression.py L536-537: backward 단계 OLS 예외 → pass
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionStepwiseException:
    def _ds(self):
        rng = np.random.default_rng(7)
        n = 40
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = x1 + x2 + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        return _make_ds(df, {"x1": _scale("x1"), "x2": _scale("x2"), "y": _scale("y")})

    def test_forward_ols_exception_continue(self):
        """L508-509: forward 선택 중 OLS.fit() 예외 → continue 처리."""
        from statworkbench.analysis.regression import run_analysis
        import statsmodels.api as sm_mod

        ds = self._ds()
        call_count = [0]
        orig_ols = sm_mod.OLS

        def patched_ols(y, X):
            call_count[0] += 1
            # call 1: 메인 회귀 OLS (성공), call 2: forward stepwise 첫 번째 변수 시도 (실패)
            if call_count[0] == 2:
                raise RuntimeError("OLS fail")
            return orig_ols(y, X)

        with patch("statworkbench.analysis.regression.sm.OLS", patched_ols):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "independent": ["x1", "x2"]},
                "options": {"selection_method": "forward", "influential_cases": False},
            })
        assert result is not None

    def test_backward_ols_exception_pass(self):
        """L536-537: backward 선택 중 OLS.fit() 예외 → pass 처리."""
        from statworkbench.analysis.regression import run_analysis
        import statsmodels.api as sm_mod

        ds = self._ds()
        call_count = [0]
        orig_ols = sm_mod.OLS

        def patched_ols(y, X):
            call_count[0] += 1
            # 첫 번째는 Enter 단계(fitted), 그 이후의 stepwise backward 단계에서 예외
            if call_count[0] == 2:
                raise RuntimeError("backward OLS fail")
            return orig_ols(y, X)

        with patch("statworkbench.analysis.regression.sm.OLS", patched_ols):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "independent": ["x1", "x2"]},
                "options": {"selection_method": "backward", "influential_cases": False},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# two_way_anova.py L91: mp = missing_policy_str (비문자열 MissingPolicy 객체)
# two_way_anova.py L236: anova table에 없는 source key → continue
# two_way_anova.py L304-305: 사후검정 예외
# two_way_anova.py L383: 빈 그룹 → continue
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoWayAnovaUncovered:
    def _ds(self, na_group=False):
        rng = np.random.default_rng(1)
        n = 60
        fa = (["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        fb = (["X"] * 10 + ["Y"] * 10) * 3
        y = rng.normal(5, 1, n)
        if na_group:
            # C 그룹의 y값을 모두 NaN → na < 1
            y[40:] = np.nan
        df = pd.DataFrame({"fa": fa, "fb": fb, "y": y})
        ds = Dataset(df, name="twa")
        ds.variables["fa"] = _nominal("fa")
        ds.variables["fb"] = _nominal("fb")
        ds.variables["y"] = _scale("y")
        return ds

    def test_missing_policy_object_passed_directly(self):
        """L91: missing_policy_str가 이미 MissingPolicy 객체일 때."""
        from statworkbench.analysis.two_way_anova import run_analysis
        ds = self._ds()
        result = run_analysis(ds, {
            "missing_policy": MissingPolicy.LISTWISE,
            "variables": {"dependent": "y", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        })
        assert result is not None

    def test_anova_table_missing_key_continue(self):
        """L236: anova_tbl에 interaction key 없을 때 → continue."""
        from statworkbench.analysis.two_way_anova import run_analysis
        import statsmodels.api as sm_api

        ds = self._ds()
        orig_anova = sm_api.stats.anova_lm

        def patched_anova(model, typ=None):
            tbl = orig_anova(model, typ=typ)
            # interaction 행을 제거하여 continue 유발
            drop_keys = [k for k in tbl.index if ":" in k]
            return tbl.drop(index=drop_keys, errors="ignore")

        with patch("statworkbench.analysis.two_way_anova.sm.stats.anova_lm", patched_anova):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor_a": "fa", "factor_b": "fb"},
                "options": {"post_hoc": False},
            })
        assert result is not None

    def test_post_hoc_exception_adds_warning(self):
        """L304-305: _run_post_hoc 예외 → 경고 추가."""
        from statworkbench.analysis.two_way_anova import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.two_way_anova._run_post_hoc",
                   side_effect=RuntimeError("post hoc fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor_a": "fa", "factor_b": "fb"},
                "options": {"post_hoc": True, "post_hoc_method": "bonferroni"},
            })
        assert any("사후 검정" in w and "오류" in w for w in result.warnings)

    def test_post_hoc_empty_group_continue(self):
        """L383: anova_tbl에 Residual 없음 → ms_error=NaN → np.isnan(ms_error) → continue."""
        from statworkbench.analysis.two_way_anova import run_analysis
        import statsmodels.api as sm_api

        ds = self._ds()
        orig_anova = sm_api.stats.anova_lm

        def patched_anova(model, typ=None):
            tbl = orig_anova(model, typ=typ)
            return tbl.drop(index=["Residual"], errors="ignore")

        with patch("statworkbench.analysis.two_way_anova.sm.stats.anova_lm", patched_anova):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor_a": "fa", "factor_b": "fb"},
                "options": {"post_hoc": True, "post_hoc_method": "bonferroni"},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# survival_analysis.py L90-92: prepare_analysis_frame 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestSurvivalAnalysisException:
    def _ds(self):
        rng = np.random.default_rng(5)
        n = 30
        df = pd.DataFrame({"t": rng.exponential(10, n), "e": rng.integers(0, 2, n)})
        return _make_ds(df, {"t": _scale("t"), "e": _binary("e")})

    def test_prepare_frame_exception(self):
        """L90-92: prepare_analysis_frame 예외 → 경고 후 반환."""
        from statworkbench.analysis.survival_analysis import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.survival_analysis.prepare_analysis_frame",
                   side_effect=RuntimeError("frame fail")):
            result = run_analysis(ds, {
                "variables": {"duration": "t", "event": "e"},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# discriminant_analysis.py L85-87: prepare_analysis_frame 예외
# discriminant_analysis.py L402-403: pearsonr 예외 → "" 처리
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscriminantAnalysisUncovered:
    def _ds(self):
        rng = np.random.default_rng(3)
        n = 60
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        grp = (["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        df = pd.DataFrame({"x1": x1, "x2": x2, "grp": grp})
        return _make_ds(df, {"x1": _scale("x1"), "x2": _scale("x2"), "grp": _nominal("grp")})

    def test_prepare_frame_exception(self):
        """L85-87: prepare_analysis_frame 예외 → 경고 후 반환."""
        from statworkbench.analysis.discriminant_analysis import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.discriminant_analysis.prepare_analysis_frame",
                   side_effect=RuntimeError("frame fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "grp", "predictors": ["x1", "x2"]},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)

    def test_pearsonr_exception_in_structure_matrix(self):
        """L402-403: stats.pearsonr 예외 → 빈 문자열 처리."""
        from statworkbench.analysis.discriminant_analysis import run_analysis
        from scipy import stats as scipy_stats

        ds = self._ds()
        with patch.object(scipy_stats, "pearsonr", side_effect=RuntimeError("corr fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "grp", "predictors": ["x1", "x2"]},
                "options": {},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# multinomial_logistic.py L139-141: MNLogit.fit() 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestMultinomialLogisticException:
    def _ds(self):
        rng = np.random.default_rng(2)
        n = 60
        x1 = rng.normal(0, 1, n)
        y = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        df = pd.DataFrame({"x1": x1, "y": y})
        return _make_ds(df, {"x1": _scale("x1"), "y": _nominal("y")})

    def test_mnlogit_fit_exception(self):
        """L139-141: MNLogit.fit() 예외 → 경고 후 반환."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        from statsmodels.discrete.discrete_model import MNLogit

        ds = self._ds()
        with patch.object(MNLogit, "fit", side_effect=RuntimeError("fit fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {},
            })
        assert any("추정 실패" in w or "오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# factor_analysis.py L75-77: prepare_analysis_frame 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorAnalysisException:
    def _ds(self):
        rng = np.random.default_rng(4)
        n = 50
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(4)})
        return _make_ds(df, {f"x{i}": _scale(f"x{i}") for i in range(4)})

    def test_prepare_frame_exception(self):
        """L75-77: prepare_analysis_frame 예외 → 경고 후 반환."""
        from statworkbench.analysis.factor_analysis import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.factor_analysis.prepare_analysis_frame",
                   side_effect=RuntimeError("frame fail")):
            result = run_analysis(ds, {
                "variables": {"variables": ["x0", "x1", "x2", "x3"]},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# cohens_kappa.py L215-217: _compute_kappa 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestCohensKappaException:
    def _ds(self):
        rng = np.random.default_rng(6)
        n = 50
        r1 = rng.choice(["A", "B", "C"], n)
        r2 = rng.choice(["A", "B", "C"], n)
        df = pd.DataFrame({"r1": r1, "r2": r2})
        return _make_ds(df, {"r1": _nominal("r1"), "r2": _nominal("r2")})

    def test_compute_kappa_exception(self):
        """L215-217: _compute_kappa 예외 → 경고 후 반환."""
        from statworkbench.analysis.cohens_kappa import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.cohens_kappa._compute_kappa",
                   side_effect=RuntimeError("kappa fail")):
            result = run_analysis(ds, {
                "variables": {"rater1": "r1", "rater2": "r2"},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# cluster_analysis.py L81-83: prepare_analysis_frame 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterAnalysisException:
    def _ds(self):
        rng = np.random.default_rng(8)
        n = 40
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        return _make_ds(df, {"x": _scale("x"), "y": _scale("y")})

    def test_prepare_frame_exception(self):
        """L81-83: prepare_analysis_frame 예외 → 경고 후 반환."""
        from statworkbench.analysis.cluster_analysis import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.cluster_analysis.prepare_analysis_frame",
                   side_effect=RuntimeError("frame fail")):
            result = run_analysis(ds, {
                "variables": {"variables": ["x", "y"]},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# io/import_wizard.py L136-137: 인코딩 프리뷰에 대체문자(U+FFFD) 포함
# ─────────────────────────────────────────────────────────────────────────────

class TestImportWizardReplacementChar:
    def test_replacement_char_in_preview(self):
        """L135-137: 샘플에 U+FFFD 포함 → preview_ok=False, 경고 추가."""
        from statworkbench.io.import_wizard import ImportWizard

        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", encoding="utf-8", delete=False
        ) as f:
            # U+FFFD 대체 문자를 직접 포함하여 저장
            f.write("col1,col2\n1,�_bad\n2,3\n")
            filepath = f.name

        try:
            wiz = ImportWizard()
            # encoding을 명시적으로 지정하여 auto-detect 우회
            wiz.step_encoding(filepath, encoding="utf-8")
            assert any("깨진 문자" in w for w in wiz._warnings)
        finally:
            os.unlink(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# repeated_measures_anova.py L251: 구형성 위반 경고
# repeated_measures_anova.py L272: _corrected_p → "-" (F=NaN)
# ─────────────────────────────────────────────────────────────────────────────

class TestRepeatedMeasuresUncovered:
    def _ds_sphericity_violated(self):
        """구형성 위반이 예상되는 데이터셋 (3 시점, 차이 분산 불균등)."""
        rng = np.random.default_rng(99)
        n = 20
        t1 = rng.normal(10, 0.1, n)
        t2 = t1 + rng.normal(5, 5.0, n)   # 큰 분산
        t3 = t1 + rng.normal(2, 0.1, n)   # 작은 분산
        subj = [f"s{i}" for i in range(n)]
        df = pd.DataFrame({"subj": subj, "t1": t1, "t2": t2, "t3": t3})
        ds = Dataset(df, name="rm_test")
        ds.variables["t1"] = _scale("t1")
        ds.variables["t2"] = _scale("t2")
        ds.variables["t3"] = _scale("t3")
        return ds

    def test_sphericity_violation_warning(self):
        """L251: Mauchly p < alpha → 구형성 위반 경고 추가."""
        from statworkbench.analysis.repeated_measures_anova import run_analysis
        from statworkbench.analysis import repeated_measures_anova as rma_mod

        ds = self._ds_sphericity_violated()
        orig_mauchly = rma_mod._mauchly_test

        def patched_mauchly(mat):
            res = orig_mauchly(mat)
            res["p"] = 0.001  # 강제로 유의하게 만듦
            return res

        with patch.object(rma_mod, "_mauchly_test", patched_mauchly):
            result = run_analysis(ds, {
                "variables": {"measures": ["t1", "t2", "t3"]},
                "options": {},
            })
        assert any("구형성 가정" in w for w in result.warnings)

    def test_corrected_p_nan_f_returns_dash(self):
        """L272: F=NaN일 때 _corrected_p → '-' 반환."""
        from statworkbench.analysis.repeated_measures_anova import run_analysis
        from statworkbench.analysis import repeated_measures_anova as rma_mod

        ds = self._ds_sphericity_violated()

        def patched_rm_anova(mat):
            return {
                "F": float("nan"), "p": float("nan"),
                "df_wf": 2, "df_err": 10,
                "MS_wf": 1.0, "MS_err": 1.0,
                "SS_bs": 1.0, "SS_ws": 1.0, "SS_wf": 1.0, "SS_err": 0.0,
                "SS_total": 2.0, "df_bs": 19, "n": 20, "k": 3,
            }

        with patch.object(rma_mod, "_rm_anova_one_factor", patched_rm_anova):
            result = run_analysis(ds, {
                "variables": {"measures": ["t1", "t2", "t3"]},
                "options": {},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# registry.py L186: 미구현 분석 등록
# registry.py L328: recommend_analyses에서 implemented=False → continue
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryUncovered:
    def test_registers_planned_analysis(self):
        """L185-190: _BUILTIN_ANALYSES에 implemented=False 항목 → _PlannedAnalysis 등록."""
        import statworkbench.analysis.registry as reg_mod
        from statworkbench.analysis.registry import AnalysisRegistry

        fake_builtin = [
            {
                "id": "planned_test",
                "name": "Planned Test",
                "category": "Test",
                "description": "test only",
                "implemented": False,
            }
        ]
        with patch.object(reg_mod, "_BUILTIN_ANALYSES", fake_builtin):
            registry = AnalysisRegistry()
        assert "planned_test" in registry._plugins
        assert not registry._plugins["planned_test"].implemented

    def test_recommend_skips_unimplemented(self):
        """L327-328: recommend_for_variables 루프에서 implemented=False → continue."""
        import statworkbench.analysis.registry as reg_mod
        from statworkbench.analysis.registry import AnalysisRegistry

        fake_builtin = [
            {
                "id": "planned_test2",
                "name": "Planned Test 2",
                "category": "Test",
                "description": "test only",
                "implemented": False,
            }
        ]
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)})
        ds = Dataset(df, name="rec_test")
        ds.variables["a"] = _scale("a")
        ds.variables["b"] = _scale("b")

        with patch.object(reg_mod, "_BUILTIN_ANALYSES", fake_builtin):
            registry = AnalysisRegistry()
            compatible = registry.recommend_for_variables(ds, ["a", "b"])
        # planned_test2는 recommend 결과에 포함되지 않아야 함
        ids = [p.id for p in compatible]
        assert "planned_test2" not in ids


# ─────────────────────────────────────────────────────────────────────────────
# nonparametric.py L96-97: 검정 함수 예외 → 경고
# ─────────────────────────────────────────────────────────────────────────────

class TestNonparametricException:
    def _ds(self):
        rng = np.random.default_rng(11)
        n = 40
        x = rng.normal(0, 1, n)
        grp = (["A"] * 20 + ["B"] * 20)
        df = pd.DataFrame({"x": x, "grp": grp})
        return _make_ds(df, {"x": _scale("x"), "grp": _nominal("grp")})

    def test_test_function_exception(self):
        """L96-97: _mann_whitney 예외 → 경고 후 반환."""
        from statworkbench.analysis.nonparametric import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.nonparametric._mann_whitney",
                   side_effect=RuntimeError("test fail")):
            result = run_analysis(ds, {
                "variables": {"test_type": "mann_whitney",
                              "dependent": "x", "group": "grp"},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# logistic_regression.py L328-329: ROC 근사 계산 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticRegressionRocException:
    def _ds(self):
        rng = np.random.default_rng(13)
        n = 60
        x = rng.normal(0, 1, n)
        y = (x > 0).astype(int)
        df = pd.DataFrame({"x": x, "y": y})
        return _make_ds(df, {"x": _scale("x"), "y": _binary("y")})

    def test_roc_exception_logged(self):
        """L328-329: manual path(_SKLEARN_AVAILABLE=False)에서 _trapz 예외 → logger.warning."""
        from statworkbench.analysis.logistic_regression import run_analysis

        ds = self._ds()
        with patch("statworkbench.analysis.logistic_regression._SKLEARN_AVAILABLE", False), \
             patch("statworkbench.analysis.logistic_regression._trapz",
                   side_effect=RuntimeError("trapz fail")):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "predictors": ["x"]},
                "options": {},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# ancova.py L227: ANOVA 테이블에 없는 source key → continue
# ancova.py L333: se <= 0 → t_stat = NaN
# ─────────────────────────────────────────────────────────────────────────────

class TestAncovaUncovered:
    def _ds(self):
        rng = np.random.default_rng(15)
        n = 60
        grp = (["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cov = rng.normal(5, 1, n)
        y = rng.normal(10, 2, n) + cov * 0.5
        df = pd.DataFrame({"grp": grp, "cov": cov, "y": y})
        return _make_ds(df, {
            "grp": _nominal("grp"),
            "cov": _scale("cov"),
            "y": _scale("y"),
        })

    def test_anova_table_missing_key_continue(self):
        """L226-227: ANOVA 테이블에 특정 key 없을 때 → continue."""
        from statworkbench.analysis.ancova import run_analysis
        import statsmodels.api as sm_api

        ds = self._ds()
        orig_anova = sm_api.stats.anova_lm

        def patched_anova(model, typ=None):
            tbl = orig_anova(model, typ=typ)
            # 인자 key를 제거하여 continue 유발
            drop_keys = [k for k in tbl.index if k.startswith("C(")]
            return tbl.drop(index=drop_keys, errors="ignore")

        with patch("statworkbench.analysis.ancova.sm.stats.anova_lm", patched_anova):
            result = run_analysis(ds, {
                "variables": {
                    "dependent": "y", "factor": "grp", "covariates": ["cov"]
                },
                "options": {},
            })
        assert result is not None

    def test_post_hoc_se_zero_nan_path(self):
        """L333: lm.mse_resid=0 → se=0 → else 경로."""
        from statworkbench.analysis.ancova import run_analysis
        from statsmodels.regression.linear_model import RegressionResults
        from unittest.mock import PropertyMock

        ds = self._ds()
        with patch.object(RegressionResults, "mse_resid",
                          new_callable=PropertyMock, return_value=0.0):
            result = run_analysis(ds, {
                "variables": {
                    "dependent": "y", "factor": "grp", "covariates": ["cov"]
                },
                "options": {"post_hoc": True},
            })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# text_mining.py L341: _find_korean_font → None (폰트 미존재)
# ─────────────────────────────────────────────────────────────────────────────

class TestTextMiningFontNotFound:
    def test_find_font_returns_none(self):
        """L340-341: 모든 후보 경로 미존재 → None 반환."""
        from statworkbench.analysis import text_mining as tm_mod
        import os

        with patch.object(os.path, "exists", return_value=False):
            result = tm_mod._find_font("ko")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# manova.py L224 (수정 후): effect_key is None → return rows (빈 리스트)
# ─────────────────────────────────────────────────────────────────────────────

class TestManovaEffectKeyNone:
    def test_effect_key_none_returns_empty_rows(self):
        """L223-224 (버그 수정 후): mv_res.results={} → effect_key=None → rows 반환."""
        from statworkbench.analysis.manova import _multivariate_tests

        mock_mv_res = MagicMock()
        mock_mv_res.results = {}  # 빈 dict → effect_key is None 조건 충족

        mock_maov = MagicMock()
        mock_maov.mv_test.return_value = mock_mv_res

        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "_dv0": rng.normal(0, 1, n),
            "_dv1": rng.normal(0, 1, n),
            "_factor": ["A"] * 15 + ["B"] * 15,
        })

        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova_cls:
            mock_manova_cls.from_formula.return_value = mock_maov
            rows = _multivariate_tests(
                df=df,
                dep_vars=["_dv0", "_dv1"],
                factor_var="_factor",
                groups=["A", "B"],
                do_effect=True,
            )
        assert rows == []
