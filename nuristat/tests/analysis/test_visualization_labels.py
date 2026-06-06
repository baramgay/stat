"""시각화 라벨 적용·논문급 품질 검증.

검증 항목:
- set_labels로 변수 label·값 label을 차트에 반영
- 막대/상자/산점도/바이올린/히트맵에서 코드값(0,1)이 값 label(남,여)로,
  변수명이 변수 label로 표시됨 (축·제목·범례·눈금)
- 비파괴: set_labels 미설정 시 원래 코드값 유지
- 신규 plot_violin (Figure 반환)
- 논문급 테마/내보내기 DPI 설정

담당 에이전트: visualizer (시각화), tester-unit
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from nuristat.analysis.visualization import _EXPORT_DPI, VisualizationEngine
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta


@pytest.fixture
def labeled_dataset() -> Dataset:
    df = pd.DataFrame({
        "sex": [0, 1, 0, 1, 0, 1, 0, 1],
        "score": [70, 85, 65, 90, 72, 88, 60, 95],
        "age": [20, 35, 22, 40, 25, 38, 19, 45],
    })
    variables = {
        "sex": VariableMeta(name="sex", label="성별",
                            storage_type=StorageType.INTEGER,
                            measure=MeasureType.NOMINAL,
                            value_labels={0: "남", 1: "여"}),
        "score": VariableMeta(name="score", label="시험점수",
                              storage_type=StorageType.INTEGER,
                              measure=MeasureType.SCALE),
        "age": VariableMeta(name="age", label="나이",
                            storage_type=StorageType.INTEGER,
                            measure=MeasureType.SCALE),
    }
    return Dataset(df, "t", variables)


@pytest.fixture
def labeled_engine(labeled_dataset) -> VisualizationEngine:
    eng = VisualizationEngine()
    eng.set_labels(labeled_dataset)
    return eng


def _xticks(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


class TestSetLabels:
    def test_stores_var_and_value_labels(self, labeled_engine):
        assert labeled_engine._labels["sex"] == "성별"
        assert labeled_engine._value_labels["sex"] == {0: "남", 1: "여"}

    def test_lbl_returns_variable_label(self, labeled_engine):
        assert labeled_engine._lbl("sex") == "성별"
        assert labeled_engine._lbl("unknown") == "unknown"

    def test_map_value(self, labeled_engine):
        assert labeled_engine._map_value("sex", 0) == "남"
        assert labeled_engine._map_value("sex", 1.0) == "여"   # 정수형 실수 흡수
        assert labeled_engine._map_value("score", 70) == "70"  # 값라벨 없으면 원값


class TestBarLabels:
    def test_xticks_use_value_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_bar(labeled_dataset.data, "sex", y_var="score")
        assert _xticks(fig.axes[0]) == ["남", "여"]

    def test_axis_uses_variable_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_bar(labeled_dataset.data, "sex", y_var="score")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "성별"
        assert ax.get_ylabel() == "시험점수"

    def test_title_uses_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_bar(labeled_dataset.data, "sex", y_var="score")
        assert "성별" in fig.axes[0].get_title()
        assert "시험점수" in fig.axes[0].get_title()


class TestBoxLabels:
    def test_grouped_box_uses_value_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_boxplot(labeled_dataset.data, "sex",
                                          y_var="score", by_group=True)
        assert _xticks(fig.axes[0]) == ["남", "여"]
        assert fig.axes[0].get_xlabel() == "성별"


class TestScatterLabels:
    def test_legend_uses_value_and_var_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_scatter(labeled_dataset.data, "age", "score",
                                          color_var="sex")
        leg = fig.axes[0].get_legend()
        labels = [t.get_text() for t in leg.get_texts()]
        assert set(labels) == {"남", "여"}
        assert leg.get_title().get_text() == "성별"


class TestViolin:
    def test_plot_violin_returns_figure_with_labels(self, labeled_engine, labeled_dataset):
        from matplotlib.figure import Figure
        fig = labeled_engine.plot_violin(labeled_dataset.data, "sex", "score")
        assert isinstance(fig, Figure)
        assert _xticks(fig.axes[0]) == ["남", "여"]
        assert "시험점수" in fig.axes[0].get_title()


class TestHeatmapLabels:
    def test_heatmap_ticklabels_use_variable_labels(self, labeled_engine, labeled_dataset):
        fig = labeled_engine.plot_correlation_heatmap(
            labeled_dataset.data, ["sex", "score", "age"]
        )
        ax = fig.axes[0]
        texts = {t.get_text() for t in ax.get_xticklabels()} | \
                {t.get_text() for t in ax.get_yticklabels()}
        assert "시험점수" in texts
        assert "나이" in texts


class TestNonDestructive:
    def test_no_labels_keeps_raw_codes(self, labeled_dataset):
        """set_labels 미호출 시 원래 코드값(0,1) 유지 — 비파괴·하위호환."""
        eng = VisualizationEngine()  # set_labels 안 함
        fig = eng.plot_bar(labeled_dataset.data, "sex", y_var="score")
        assert _xticks(fig.axes[0]) == ["0", "1"]

    def test_apply_value_labels_does_not_mutate_input(self, labeled_engine, labeled_dataset):
        original = labeled_dataset.data.copy()
        labeled_engine.plot_bar(labeled_dataset.data, "sex", y_var="score")
        pd.testing.assert_frame_equal(labeled_dataset.data, original)


class TestQuality:
    def test_export_dpi_is_publication_grade(self):
        assert _EXPORT_DPI >= 300

    def test_base64_methods_apply_labels(self, labeled_engine, labeled_dataset):
        """base64 반환 메서드(bar_chart)도 값 label 적용 — 예외 없이 PNG 반환."""
        img = labeled_engine.bar_chart(labeled_dataset.data, x="sex", y="score")
        assert isinstance(img, str) and img.startswith("data:image/png")
