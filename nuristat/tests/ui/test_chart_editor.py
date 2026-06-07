"""차트 편집 기능 검증 — 생성 후 사용자 수정 (SPSS 차트 편집기 스타일).

축 제목·글꼴 배율·크기 수정, 격자/범례 토글, 통계 상자(회귀식·상관계수) 제거.

담당 에이전트: visualizer/frontend, tester-unit
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from nuristat.analysis.visualization import VisualizationEngine
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 6], "g": [0, 1, 0, 1, 0]})


# ── 엔진: 통계 상자 토글 ────────────────────────────────────────────────────

class TestShowStats:
    def test_scatter_hides_correlation_box(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y", show_stats=False)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert not any("r =" in t for t in texts)

    def test_scatter_shows_correlation_by_default(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y", show_stats=True)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("r =" in t for t in texts)

    def test_histogram_hides_stats_box(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_histogram(df, "x", show_stats=False)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert not any("왜도" in t for t in texts)


# ── 엔진: apply_edits 후처리 ────────────────────────────────────────────────

class TestApplyEdits:
    def test_axis_titles_overridden(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y")
        eng.apply_edits(fig, title="제목", xlabel="X제목", ylabel="Y제목")
        ax = fig.axes[0]
        assert ax.get_title() == "제목"
        assert ax.get_xlabel() == "X제목"
        assert ax.get_ylabel() == "Y제목"

    def test_empty_strings_keep_auto(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y")
        auto_x = fig.axes[0].get_xlabel()
        eng.apply_edits(fig, xlabel="", ylabel="")
        assert fig.axes[0].get_xlabel() == auto_x   # 빈값이면 자동 유지

    def test_font_scale_enlarges(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y")
        before = fig.axes[0].title.get_fontsize()
        eng.apply_edits(fig, title="T", font_scale=1.5)
        after = fig.axes[0].title.get_fontsize()
        assert after > before

    def test_figsize_changes(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y")
        eng.apply_edits(fig, figsize=(8.0, 5.0))
        w, h = fig.get_size_inches()
        assert abs(w - 8.0) < 0.01 and abs(h - 5.0) < 0.01

    def test_grid_toggle_off(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y")
        eng.apply_edits(fig, show_grid=False)
        # 격자 라인 가시성 off
        assert not any(gl.get_visible() for gl in fig.axes[0].get_xgridlines())

    def test_legend_removed(self, df):
        eng = VisualizationEngine()
        fig = eng.plot_scatter(df, "x", "y", color_var="g")  # 범례 생성
        assert fig.axes[0].get_legend() is not None
        eng.apply_edits(fig, show_legend=False)
        assert fig.axes[0].get_legend() is None


# ── 다이얼로그 통합 ─────────────────────────────────────────────────────────

class TestDialogEditControls:
    def _dialog(self):
        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 6]})
        variables = {
            "x": VariableMeta(name="x", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
            "y": VariableMeta(name="y", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        }
        return VisualizationDialog(Dataset(df, "t", variables))

    def test_edit_controls_exist(self):
        dlg = self._dialog()
        for attr in ("xlabel_edit", "ylabel_edit", "font_scale_spin",
                     "fig_w_spin", "fig_h_spin", "grid_check", "legend_check",
                     "stats_check"):
            assert hasattr(dlg, attr), attr

    def test_build_applies_axis_title_edit(self):
        dlg = self._dialog()
        dlg.xlabel_edit.setText("사용자X")
        fig = dlg._build_figure("scatter", dlg.dataset.data, "x", "y", None, "")
        dlg.engine.apply_edits(fig, xlabel=dlg.xlabel_edit.text().strip())
        assert fig.axes[0].get_xlabel() == "사용자X"

    def test_stats_check_controls_box(self):
        dlg = self._dialog()
        dlg.stats_check.setChecked(False)
        fig = dlg._build_figure("scatter", dlg.dataset.data, "x", "y", None, "")
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert not any("r =" in t for t in texts)
