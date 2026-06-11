"""차트 다이얼로그 디테일 검증 — 제목 적용·x축 무회전·변수 라벨 표시.

사용자 보고 3건:
1. 막대 외 차트에서 옵션 제목이 표시되지 않던 문제
2. x축 눈금 라벨 기울임 제거
3. 변수 선택 콤보에 '변수명 (라벨)' 표시 (라벨이 실질적 변수명)

담당 에이전트: frontend/visualizer, tester-unit
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
def dataset() -> Dataset:
    df = pd.DataFrame({
        "sex": [0, 1, 0, 1, 0, 1],
        "score": [70, 85, 65, 90, 72, 88],
        "age": [20, 35, 22, 40, 25, 38],
    })
    variables = {
        "sex": VariableMeta(name="sex", label="성별", storage_type=StorageType.INTEGER,
                            measure=MeasureType.NOMINAL, value_labels={0: "남", 1: "여"}),
        "score": VariableMeta(name="score", label="시험점수",
                              storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "age": VariableMeta(name="age", storage_type=StorageType.INTEGER,
                            measure=MeasureType.SCALE),   # 라벨 없음
    }
    return Dataset(df, "t", variables)


# ── 1. 제목이 모든 차트 유형에 적용 ────────────────────────────────────────

class TestTitleAllChartTypes:
    def _dialog(self, dataset):
        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        return VisualizationDialog(dataset)

    @pytest.mark.parametrize("chart_type,x,y,hue", [
        ("bar", "sex", "score", None),
        ("hist", "score", "(선택)", None),
        ("scatter", "age", "score", None),
        ("box", "sex", "score", None),
        ("line", "age", "score", None),
        ("violin", "sex", "score", None),
        ("heatmap", "score", "(선택)", None),
    ])
    def test_title_shown(self, dataset, chart_type, x, y, hue):
        dlg = self._dialog(dataset)
        # _build_figure는 None sentinel을 기대 → "(선택)"은 None으로 변환
        yy = None if y == "(선택)" else y
        fig = dlg._build_figure(chart_type, dataset.data, x, yy, hue, "내 제목")
        assert fig.axes[0].get_title() == "내 제목", f"{chart_type} 제목 미적용"

    def test_qq_title_uses_suptitle(self, dataset):
        dlg = self._dialog(dataset)
        fig = dlg._build_figure("qq", dataset.data, "score", None, None, "정규성 제목")
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "정규성 제목"


# ── 2. x축 눈금 라벨 회전 없음 ──────────────────────────────────────────────

class TestNoXAxisRotation:
    def test_bar_xticklabels_not_rotated(self):
        eng = VisualizationEngine()
        df = pd.DataFrame({
            "category_long_name": ["가나다라마바사", "아자차카타파하", "ABCDEFGHIJK"],
            "val": [1, 2, 3],
        })
        fig = eng.plot_bar(df, "category_long_name", y_var="val")
        for lbl in fig.axes[0].get_xticklabels():
            assert lbl.get_rotation() == 0, "x축 라벨이 기울어짐"

    def test_count_bar_not_rotated(self):
        eng = VisualizationEngine()
        df = pd.DataFrame({"g": ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"] * 3})
        fig = eng.plot_bar(df, "g")
        for lbl in fig.axes[0].get_xticklabels():
            assert lbl.get_rotation() == 0


# ── 3. 변수 콤보에 '변수명 (라벨)' 표시 + 실제 컬럼명 보관 ──────────────────

class TestVariableLabelDisplay:
    def _dialog(self, dataset):
        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        return VisualizationDialog(dataset)

    def test_combo_shows_name_with_label(self, dataset):
        dlg = self._dialog(dataset)
        texts = [dlg.x_combo.itemText(i) for i in range(dlg.x_combo.count())]
        assert "sex (성별)" in texts
        assert "score (시험점수)" in texts

    def test_combo_no_label_shows_plain_name(self, dataset):
        dlg = self._dialog(dataset)
        texts = [dlg.x_combo.itemText(i) for i in range(dlg.x_combo.count())]
        assert "age" in texts          # 라벨 없으면 변수명만
        assert "age (" not in " ".join(texts)

    def test_combo_data_is_actual_column(self, dataset):
        dlg = self._dialog(dataset)
        idx = dlg.x_combo.findText("sex (성별)")
        assert idx >= 0
        assert dlg.x_combo.itemData(idx) == "sex"   # 실제 컬럼명 보관

    def test_sentinel_data_is_none(self, dataset):
        dlg = self._dialog(dataset)
        assert dlg.x_combo.itemData(0) is None      # "(선택)"

    def test_chart_builder_combo_shows_label(self, dataset):
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(dataset)
        texts = [dlg._x_combo.itemText(i) for i in range(dlg._x_combo.count())]
        assert "sex (성별)" in texts
        idx = dlg._x_combo.findText("sex (성별)")
        assert dlg._x_combo.itemData(idx) == "sex"


# ── 4. preset_chart_type — 레거시 기존 대화상자 액션 사전 선택 ──────────────

class TestPresetChartType:
    """기존 대화상자 메뉴(막대/선/산점도/히스토그램/상자 그림) 사전 선택 검증."""

    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter", "hist", "box"])
    def test_preset_selects_correct_radio(self, dataset, chart_type):
        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        dlg = VisualizationDialog(dataset, preset_chart_type=chart_type)
        selected = None
        for btn in dlg.chart_type_group.buttons():
            if btn.isChecked():
                selected = btn.property("chart_type")
        assert selected == chart_type, f"preset={chart_type} 이지만 선택된 버튼={selected}"

    def test_no_preset_defaults_to_bar(self, dataset):
        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        dlg = VisualizationDialog(dataset)
        selected = None
        for btn in dlg.chart_type_group.buttons():
            if btn.isChecked():
                selected = btn.property("chart_type")
        assert selected == "bar"
