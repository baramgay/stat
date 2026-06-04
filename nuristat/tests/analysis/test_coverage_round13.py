"""Round 13: 소형 모듈 예외 경로 보완 (bland_altman/chi_square_gof/descriptive/normality/partial_correlation/reliability)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


# ─────────────────────────────────────────────────────────────────────────────
# bland_altman.py L175-177: _compute_bland_altman 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestBlandAltmanException:
    def _ds(self):
        rng = np.random.default_rng(0)
        n = 30
        m1 = rng.normal(50, 5, n)
        m2 = m1 + rng.normal(0, 2, n)
        df = pd.DataFrame({"m1": m1, "m2": m2})
        ds = Dataset(df, name="ba_test")
        ds.variables["m1"] = _scale("m1")
        ds.variables["m2"] = _scale("m2")
        return ds

    def test_compute_exception_adds_warning(self):
        """L175-177: _compute_bland_altman 예외 → 경고 후 반환."""
        from nuristat.analysis.bland_altman import run_analysis
        ds = self._ds()
        with patch("nuristat.analysis.bland_altman._compute_bland_altman",
                   side_effect=RuntimeError("compute fail")):
            result = run_analysis(ds, {
                "variables": {"method1": "m1", "method2": "m2"},
                "options": {},
            })
        assert any("오류" in w or "계산" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# chi_square_gof.py L149-151: 변수 검정 중 예외 → continue
# ─────────────────────────────────────────────────────────────────────────────

class TestChiSquareGofException:
    def _ds(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "cat1": np.random.choice(["A", "B", "C"], 50, replace=True),
            "cat2": np.random.choice(["X", "Y"], 50, replace=True),
        })
        ds = Dataset(df, name="chisq_test")
        ds.variables["cat1"] = _nominal("cat1")
        ds.variables["cat2"] = _nominal("cat2")
        return ds

    def test_chisquare_exception_continue(self):
        """L149-151: chisquare 예외 → 경고 후 continue."""
        from nuristat.analysis.chi_square_gof import run_analysis
        ds = self._ds()
        with patch("nuristat.analysis.chi_square_gof.chisquare",
                   side_effect=RuntimeError("chisq fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["cat1", "cat2"]},
                "options": {},
            })
        assert any("검정 오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# descriptive.py L139-141: _compute_descriptives 예외 → logger 경고 + fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptiveException:
    def _ds(self, with_group=True):
        rng = np.random.default_rng(0)
        n = 30
        data = {
            "x": rng.normal(0, 1, n),
        }
        if with_group:
            data["grp"] = ["A"] * 15 + ["B"] * 15
        df = pd.DataFrame(data)
        ds = Dataset(df, name="desc_test")
        ds.variables["x"] = _scale("x")
        if with_group:
            ds.variables["grp"] = _nominal("grp")
        return ds

    def test_compute_exception_with_group_fallback(self):
        """L139-141: _compute_descriptives 예외 → empty stats 폴백."""
        from nuristat.analysis.descriptive import run_analysis
        ds = self._ds(with_group=True)
        call_count = [0]

        from nuristat.analysis.descriptive import _compute_descriptives as orig_compute

        def raise_on_second(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("compute fail")
            return orig_compute(*args, **kwargs)

        with patch("nuristat.analysis.descriptive._compute_descriptives",
                   side_effect=raise_on_second):
            result = run_analysis(ds, {
                "variables": {"scale": ["x"], "group": "grp"},
                "options": {},
            })
        # 예외 발생해도 결과 반환
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# normality.py L138-139: 분석 루프 중 예외 → 경고 추가
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalityException:
    def _ds(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30)})
        ds = Dataset(df, name="norm_test")
        ds.variables["x"] = _scale("x")
        return ds

    def test_shapiro_exception_adds_warning(self):
        """L138-139: stats.shapiro 예외 → 경고 추가."""
        from nuristat.analysis.normality import run_analysis
        from scipy import stats as scipy_stats
        ds = self._ds()
        with patch.object(scipy_stats, "shapiro", side_effect=RuntimeError("shapiro fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["x"]},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# partial_correlation.py L159: df_val <= 0 경고
#                              L171-173: _partial_corr_matrix 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialCorrelationExceptions:
    def _ds(self, n=6, n_ctrl=4):
        rng = np.random.default_rng(0)
        data = {f"x{i}": rng.normal(0, 1, n) for i in range(n_ctrl + 2)}
        df = pd.DataFrame(data)
        ds = Dataset(df, name="pcorr_test")
        for c in df.columns:
            ds.variables[c] = _scale(c)
        return ds, n_ctrl

    def test_df_val_zero_or_negative_warning(self):
        """L159-162: df_val <= 0 → 경고 (n=6, 4 통제변수 → df=6-2-4=0)."""
        from nuristat.analysis.partial_correlation import run_analysis
        ds, n_ctrl = self._ds(n=6, n_ctrl=4)
        ctrl_vars = [f"x{i}" for i in range(n_ctrl)]
        target_vars = [f"x{n_ctrl}", f"x{n_ctrl+1}"]
        result = run_analysis(ds, {
            "variables": {"target": target_vars, "controlling": ctrl_vars},
            "options": {},
        })
        assert any("자유도" in w or "df" in w.lower() for w in result.warnings)

    def test_partial_corr_matrix_exception(self):
        """L171-173: _partial_corr_matrix 예외 → 경고 후 반환."""
        from nuristat.analysis.partial_correlation import run_analysis
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 30),
            "y": rng.normal(0, 1, 30),
            "ctrl": rng.normal(0, 1, 30),
        })
        ds = Dataset(df, name="pcorr_exc")
        for c in ["x", "y", "ctrl"]:
            ds.variables[c] = _scale(c)
        with patch("nuristat.analysis.partial_correlation._partial_corr_matrix",
                   side_effect=RuntimeError("matrix fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["x", "y"], "controlling": ["ctrl"]},
                "options": {},
            })
        assert any("오류" in w or "계산" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# reliability.py L94-96: _cronbach_alpha 예외
# ─────────────────────────────────────────────────────────────────────────────

class TestReliabilityException:
    def _ds(self):
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "item1": rng.normal(3, 1, n),
            "item2": rng.normal(3, 1, n),
            "item3": rng.normal(3, 1, n),
        })
        ds = Dataset(df, name="rel_test")
        for c in ["item1", "item2", "item3"]:
            ds.variables[c] = _scale(c)
        return ds

    def test_cronbach_alpha_exception_adds_warning(self):
        """L94-96: _cronbach_alpha 예외 → 경고 후 반환."""
        from nuristat.analysis.reliability import run_analysis
        ds = self._ds()
        with patch("nuristat.analysis.reliability._cronbach_alpha",
                   side_effect=RuntimeError("alpha fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["item1", "item2", "item3"]},
                "options": {},
            })
        assert any("Alpha" in w or "오류" in w for w in result.warnings)
