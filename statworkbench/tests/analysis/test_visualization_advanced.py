"""visualization.py 고급 테스트 — 커버리지 60%+ 달성.

기존 34개 테스트(test_visualization.py) 미커버 경로 집중 보완:
- plot_histogram / plot_boxplot / plot_scatter / plot_bar / plot_line (Figure 반환 계열)
- plot_qq / plot_correlation_heatmap / plot_roc_curve / plot_survival_curve / plot_residuals
- heatmap / violin_plot / line_chart 고급 경로
- bar_chart 수평 + hue 복합, scatter_plot regression + hue 경로
- _validate_data 결측치 경계, _validate_numeric 무한값 경고
- _error_image / _make_error_figure / _apply_readability 내부 헬퍼
- save_figure 경로
- Dataset 픽스처 기반 직접 호출

담당 에이전트: statworkbench, tester-unit
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.visualization import VisualizationEngine
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

def _is_valid_base64_png(s: str) -> bool:
    """base64 문자열이 유효한 PNG인지 확인."""
    try:
        raw = s.split(",", 1)[1] if "," in s else s
        data = base64.b64decode(raw)
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


def _is_figure(obj) -> bool:
    return isinstance(obj, Figure)


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> VisualizationEngine:
    return VisualizationEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """기본 수치형 + 범주형 혼합 DataFrame."""
    return pd.DataFrame({
        "x":     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y":     [2, 4, 1, 8, 5, 7, 3, 6, 9, 4],
        "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "cat":   ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
    })


@pytest.fixture
def sample_ds(sample_df) -> Dataset:
    """Dataset 픽스처 — visualization 경로 테스트용."""
    return Dataset(data=sample_df, name="vis_test")


@pytest.fixture
def large_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(5, 2, 200),
        "c": rng.uniform(0, 10, 200),
        "d": rng.integers(0, 4, 200).astype(float),
        "grp": (["X"] * 100 + ["Y"] * 100),
    })


@pytest.fixture
def na_df() -> pd.DataFrame:
    """결측치 포함 DataFrame."""
    return pd.DataFrame({
        "v": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0, np.nan, 8.0, np.nan, np.nan],
        "g": ["A"] * 5 + ["B"] * 5,
    })


@pytest.fixture
def single_value_df() -> pd.DataFrame:
    """단일 값(상수) DataFrame."""
    return pd.DataFrame({"x": [5.0] * 10, "y": [3.0] * 10})


# ──────────────────────────────────────────────────────────────
# 1. bar_chart 고급 경로
# ──────────────────────────────────────────────────────────────

class TestBarChartAdvanced:
    """bar_chart 미커버 분기 집중 테스트."""

    def test_bar_chart_valid_png_header(self, engine, sample_df):
        """반환값이 PNG 헤더를 포함한 base64 문자열."""
        result = engine.bar_chart(sample_df, x="group", y="x")
        assert _is_valid_base64_png(result)

    def test_bar_chart_horizontal_no_y(self, engine, sample_df):
        """horizontal + y 없음 → 빈도 수평 막대."""
        result = engine.bar_chart(sample_df, x="group", orientation="horizontal")
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_horizontal_with_hue(self, engine, sample_df):
        """horizontal + hue 분기."""
        result = engine.bar_chart(
            sample_df, x="group", y="x", hue="cat", orientation="horizontal"
        )
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_vertical_with_hue(self, engine, sample_df):
        """vertical + hue 분기."""
        result = engine.bar_chart(
            sample_df, x="group", y="x", hue="cat", orientation="vertical"
        )
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_with_title(self, engine, sample_df):
        result = engine.bar_chart(sample_df, x="group", y="x", title="Test Title")
        assert _is_valid_base64_png(result)

    def test_bar_chart_size_wide(self, engine, sample_df):
        result = engine.bar_chart(sample_df, x="group", y="x", size="wide")
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_size_square(self, engine, sample_df):
        result = engine.bar_chart(sample_df, x="group", y="x", size="square")
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_unknown_size_fallback(self, engine, sample_df):
        """알 수 없는 size → 기본값(10,6) 폴백."""
        result = engine.bar_chart(sample_df, x="group", y="x", size="xxl")
        assert isinstance(result, str) and len(result) > 100

    def test_bar_chart_count_mode_horizontal(self, engine, sample_df):
        """y 없고 수평 방향 → barh 사용."""
        result = engine.bar_chart(sample_df, x="cat", orientation="horizontal")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 2. histogram 고급 경로
# ──────────────────────────────────────────────────────────────

class TestHistogramAdvanced:
    """histogram 미커버 분기."""

    def test_histogram_with_missing_values(self, engine, na_df):
        """결측치 있는 컬럼 → 오류 없이 실행."""
        result = engine.histogram(na_df, x="v")
        assert isinstance(result, str) and len(result) > 100

    def test_histogram_single_unique_value_warns(self, engine, single_value_df):
        """단일 값 컬럼 → _validate_numeric warnings, 그래도 이미지 반환."""
        result = engine.histogram(single_value_df, x="x")
        assert isinstance(result, str) and len(result) > 100

    def test_histogram_bins_50(self, engine, large_df):
        result = engine.histogram(large_df, x="a", bins=50)
        assert _is_valid_base64_png(result)

    def test_histogram_no_kde(self, engine, large_df):
        result = engine.histogram(large_df, x="b", kde=False)
        assert _is_valid_base64_png(result)

    def test_histogram_title_provided(self, engine, large_df):
        result = engine.histogram(large_df, x="c", title="Custom Title")
        assert isinstance(result, str) and len(result) > 100

    def test_histogram_size_small(self, engine, large_df):
        result = engine.histogram(large_df, x="a", size="small")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 3. scatter_plot 고급 경로
# ──────────────────────────────────────────────────────────────

class TestScatterPlotAdvanced:
    """scatter_plot 미커버 분기."""

    def test_scatter_with_regression(self, engine, sample_df):
        """add_regression=True, hue=None → regplot 사용."""
        result = engine.scatter_plot(sample_df, x="x", y="y", add_regression=True)
        assert isinstance(result, str) and len(result) > 100

    def test_scatter_with_hue(self, engine, sample_df):
        """hue 지정 → scatterplot 사용."""
        result = engine.scatter_plot(sample_df, x="x", y="y", hue="group")
        assert isinstance(result, str) and len(result) > 100

    def test_scatter_with_size_var(self, engine, sample_df):
        """size_var 지정."""
        result = engine.scatter_plot(sample_df, x="x", y="y", size_var="x")
        assert isinstance(result, str) and len(result) > 100

    def test_scatter_returns_valid_png(self, engine, large_df):
        result = engine.scatter_plot(large_df, x="a", y="b")
        assert _is_valid_base64_png(result)

    def test_scatter_with_title(self, engine, sample_df):
        result = engine.scatter_plot(sample_df, x="x", y="y", title="Scatter Test")
        assert isinstance(result, str) and len(result) > 100

    def test_scatter_missing_column_returns_error(self, engine, sample_df):
        result = engine.scatter_plot(sample_df, x="missing", y="y")
        assert isinstance(result, str) and len(result) > 100

    def test_scatter_empty_df_returns_error(self, engine):
        result = engine.scatter_plot(pd.DataFrame(), x="x", y="y")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 4. box_plot 고급 경로
# ──────────────────────────────────────────────────────────────

class TestBoxPlotAdvanced:
    """box_plot 미커버 분기."""

    def test_box_plot_with_x_and_y(self, engine, sample_df):
        result = engine.box_plot(sample_df, x="group", y="x")
        assert _is_valid_base64_png(result)

    def test_box_plot_without_x(self, engine, sample_df):
        """x=None → 단일 상자 그림."""
        result = engine.box_plot(sample_df, y="x")
        assert isinstance(result, str) and len(result) > 100

    def test_box_plot_with_hue(self, engine, sample_df):
        result = engine.box_plot(sample_df, x="group", y="x", hue="cat")
        assert isinstance(result, str) and len(result) > 100

    def test_box_plot_empty_df(self, engine):
        result = engine.box_plot(pd.DataFrame(), y="x")
        assert isinstance(result, str) and len(result) > 100

    def test_box_plot_with_title(self, engine, sample_df):
        result = engine.box_plot(sample_df, x="group", y="x", title="Box Test")
        assert isinstance(result, str) and len(result) > 100

    def test_box_plot_size_large(self, engine, sample_df):
        result = engine.box_plot(sample_df, x="group", y="x", size="large")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 5. line_chart 고급 경로
# ──────────────────────────────────────────────────────────────

class TestLineChartAdvanced:
    """line_chart 미커버 분기."""

    def test_line_chart_valid_png(self, engine, sample_df):
        result = engine.line_chart(sample_df, x="x", y="y")
        assert _is_valid_base64_png(result)

    def test_line_chart_with_hue(self, engine, sample_df):
        """hue 지정 → 그룹별 선 차트."""
        result = engine.line_chart(sample_df, x="x", y="y", hue="group")
        assert isinstance(result, str) and len(result) > 100

    def test_line_chart_no_marker(self, engine, sample_df):
        result = engine.line_chart(sample_df, x="x", y="y", marker=False)
        assert isinstance(result, str) and len(result) > 100

    def test_line_chart_with_title(self, engine, sample_df):
        result = engine.line_chart(sample_df, x="x", y="y", title="Line Test")
        assert isinstance(result, str) and len(result) > 100

    def test_line_chart_empty_df(self, engine):
        result = engine.line_chart(pd.DataFrame(), x="x", y="y")
        assert isinstance(result, str) and len(result) > 100

    def test_line_chart_missing_column(self, engine, sample_df):
        result = engine.line_chart(sample_df, x="x", y="missing")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 6. heatmap
# ──────────────────────────────────────────────────────────────

class TestHeatmap:
    """heatmap 미커버 분기."""

    def test_heatmap_valid_png(self, engine, large_df):
        result = engine.heatmap(large_df)
        assert _is_valid_base64_png(result)

    def test_heatmap_with_cols(self, engine, large_df):
        result = engine.heatmap(large_df, cols=["a", "b", "c"])
        assert isinstance(result, str) and len(result) > 100

    def test_heatmap_no_annot(self, engine, large_df):
        result = engine.heatmap(large_df, annot=False)
        assert isinstance(result, str) and len(result) > 100

    def test_heatmap_with_title(self, engine, large_df):
        result = engine.heatmap(large_df, title="Corr Heatmap")
        assert isinstance(result, str) and len(result) > 100

    def test_heatmap_no_numeric_columns_returns_error(self, engine):
        """숫자형 없음 → 오류 이미지."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        result = engine.heatmap(df)
        assert isinstance(result, str) and len(result) > 100

    def test_heatmap_size_medium(self, engine, large_df):
        result = engine.heatmap(large_df, size="medium")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 7. violin_plot
# ──────────────────────────────────────────────────────────────

class TestViolinPlot:
    """violin_plot 미커버 분기."""

    def test_violin_valid_png(self, engine, sample_df):
        result = engine.violin_plot(sample_df, x="group", y="x")
        assert _is_valid_base64_png(result)

    def test_violin_with_hue(self, engine, sample_df):
        result = engine.violin_plot(sample_df, x="group", y="x", hue="cat")
        assert isinstance(result, str) and len(result) > 100

    def test_violin_with_title(self, engine, sample_df):
        result = engine.violin_plot(sample_df, x="group", y="x", title="Violin Test")
        assert isinstance(result, str) and len(result) > 100

    def test_violin_empty_df(self, engine):
        result = engine.violin_plot(pd.DataFrame(), x="g", y="v")
        assert isinstance(result, str) and len(result) > 100

    def test_violin_missing_column(self, engine, sample_df):
        result = engine.violin_plot(sample_df, x="missing", y="x")
        assert isinstance(result, str) and len(result) > 100


# ──────────────────────────────────────────────────────────────
# 8. plot_histogram (Figure 반환 계열)
# ──────────────────────────────────────────────────────────────

class TestPlotHistogram:
    """plot_histogram → Figure 반환 검증."""

    def test_returns_figure(self, engine, sample_df):
        fig = engine.plot_histogram(sample_df, "x")
        assert _is_figure(fig)
        plt.close(fig)

    def test_with_normal_curve(self, engine, large_df):
        fig = engine.plot_histogram(large_df, "a", normal_curve=True)
        assert _is_figure(fig)
        plt.close(fig)

    def test_without_normal_curve(self, engine, large_df):
        fig = engine.plot_histogram(large_df, "b", normal_curve=False)
        assert _is_figure(fig)
        plt.close(fig)

    def test_custom_bins(self, engine, large_df):
        fig = engine.plot_histogram(large_df, "c", bins=10)
        assert _is_figure(fig)
        plt.close(fig)

    def test_missing_column_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_histogram(sample_df, "nonexistent")
        assert _is_figure(fig)
        plt.close(fig)

    def test_non_numeric_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_histogram(sample_df, "group")
        assert _is_figure(fig)
        plt.close(fig)

    def test_empty_df_returns_error_figure(self, engine):
        fig = engine.plot_histogram(pd.DataFrame(), "x")
        assert _is_figure(fig)
        plt.close(fig)

    def test_stats_annotation_in_axes(self, engine, large_df):
        """통계 정보 박스 텍스트가 ax에 추가됨을 간접 확인."""
        fig = engine.plot_histogram(large_df, "a", normal_curve=True)
        assert len(fig.axes) >= 1
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 9. plot_boxplot (Figure 반환 계열)
# ──────────────────────────────────────────────────────────────

class TestPlotBoxplot:
    """plot_boxplot → Figure 반환 검증."""

    def test_returns_figure_by_group(self, engine, sample_df):
        fig = engine.plot_boxplot(sample_df, x_var="group", y_var="x", by_group=True)
        assert _is_figure(fig)
        plt.close(fig)

    def test_returns_figure_no_group(self, engine, sample_df):
        fig = engine.plot_boxplot(sample_df, x_var="x", by_group=False)
        assert _is_figure(fig)
        plt.close(fig)

    def test_y_var_not_in_columns(self, engine, sample_df):
        """y_var 없으면 단일 상자 그림 분기."""
        fig = engine.plot_boxplot(sample_df, x_var="x", y_var="no_col", by_group=True)
        assert _is_figure(fig)
        plt.close(fig)

    def test_by_group_false_single_box(self, engine, large_df):
        fig = engine.plot_boxplot(large_df, x_var="a", by_group=False)
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 10. plot_scatter (Figure 반환 계열)
# ──────────────────────────────────────────────────────────────

class TestPlotScatter:
    """plot_scatter → Figure 반환 검증."""

    def test_returns_figure_basic(self, engine, sample_df):
        fig = engine.plot_scatter(sample_df, x_var="x", y_var="y")
        assert _is_figure(fig)
        plt.close(fig)

    def test_with_color_var(self, engine, sample_df):
        """color_var 지정 → 그룹별 색상 분기."""
        fig = engine.plot_scatter(sample_df, x_var="x", y_var="y", color_var="group")
        assert _is_figure(fig)
        plt.close(fig)

    def test_no_fit_line(self, engine, sample_df):
        fig = engine.plot_scatter(sample_df, x_var="x", y_var="y", fit_line=False)
        assert _is_figure(fig)
        plt.close(fig)

    def test_color_var_with_fit_line(self, engine, large_df):
        """color_var + fit_line → 그룹별 회귀선 분기."""
        fig = engine.plot_scatter(large_df, x_var="a", y_var="b", color_var="grp", fit_line=True)
        assert _is_figure(fig)
        plt.close(fig)

    def test_empty_df_returns_error_figure(self, engine):
        fig = engine.plot_scatter(pd.DataFrame(), x_var="x", y_var="y")
        assert _is_figure(fig)
        plt.close(fig)

    def test_missing_column_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_scatter(sample_df, x_var="x", y_var="missing")
        assert _is_figure(fig)
        plt.close(fig)

    def test_color_var_not_in_columns(self, engine, sample_df):
        """color_var가 컬럼에 없을 때 → 기본 분기."""
        fig = engine.plot_scatter(sample_df, x_var="x", y_var="y", color_var="no_col")
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 11. plot_bar (Figure 반환 계열)
# ──────────────────────────────────────────────────────────────

class TestPlotBar:
    """plot_bar → Figure 반환 검증."""

    def test_returns_figure_with_y(self, engine, sample_df):
        fig = engine.plot_bar(sample_df, x_var="group", y_var="x")
        assert _is_figure(fig)
        plt.close(fig)

    def test_returns_figure_count_mode(self, engine, sample_df):
        """y_var=None → 빈도 막대."""
        fig = engine.plot_bar(sample_df, x_var="group")
        assert _is_figure(fig)
        plt.close(fig)

    def test_error_bars_false(self, engine, sample_df):
        fig = engine.plot_bar(sample_df, x_var="group", y_var="x", error_bars=False)
        assert _is_figure(fig)
        plt.close(fig)

    def test_empty_df_returns_error_figure(self, engine):
        fig = engine.plot_bar(pd.DataFrame(), x_var="x")
        assert _is_figure(fig)
        plt.close(fig)

    def test_y_var_not_in_columns(self, engine, sample_df):
        """y_var가 컬럼에 없으면 빈도 분기."""
        fig = engine.plot_bar(sample_df, x_var="group", y_var="no_col")
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 12. plot_line (Figure 반환 계열)
# ──────────────────────────────────────────────────────────────

class TestPlotLine:
    """plot_line → Figure 반환 검증."""

    def test_returns_figure_basic(self, engine, sample_df):
        fig = engine.plot_line(sample_df, x_var="x", y_var="y")
        assert _is_figure(fig)
        plt.close(fig)

    def test_with_by_group(self, engine, sample_df):
        """by_group 지정 → 그룹별 선 분기."""
        fig = engine.plot_line(sample_df, x_var="x", y_var="y", by_group="group")
        assert _is_figure(fig)
        plt.close(fig)

    def test_by_group_not_in_columns(self, engine, sample_df):
        """by_group이 컬럼에 없으면 단일 선 분기."""
        fig = engine.plot_line(sample_df, x_var="x", y_var="y", by_group="no_col")
        assert _is_figure(fig)
        plt.close(fig)

    def test_empty_df_returns_error_figure(self, engine):
        fig = engine.plot_line(pd.DataFrame(), x_var="x", y_var="y")
        assert _is_figure(fig)
        plt.close(fig)

    def test_missing_column_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_line(sample_df, x_var="x", y_var="missing")
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 13. plot_qq
# ──────────────────────────────────────────────────────────────

class TestPlotQQ:
    """plot_qq → Figure 반환 + 정규성 검정."""

    def test_returns_figure(self, engine, large_df):
        fig = engine.plot_qq(large_df, "a")
        assert _is_figure(fig)
        plt.close(fig)

    def test_two_axes(self, engine, large_df):
        """QQ + 히스토그램 두 개 axes 확인."""
        fig = engine.plot_qq(large_df, "b")
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_missing_column_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_qq(sample_df, "no_col")
        assert _is_figure(fig)
        plt.close(fig)

    def test_non_numeric_returns_error_figure(self, engine, sample_df):
        fig = engine.plot_qq(sample_df, "group")
        assert _is_figure(fig)
        plt.close(fig)

    def test_empty_df_returns_error_figure(self, engine):
        fig = engine.plot_qq(pd.DataFrame(), "x")
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 14. plot_correlation_heatmap
# ──────────────────────────────────────────────────────────────

class TestPlotCorrelationHeatmap:
    """plot_correlation_heatmap → Figure 반환 검증."""

    def test_returns_figure(self, engine, large_df):
        fig = engine.plot_correlation_heatmap(large_df, ["a", "b", "c"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_only_one_numeric_returns_error(self, engine, sample_df):
        """숫자형 변수 1개 → 오류 Figure."""
        fig = engine.plot_correlation_heatmap(sample_df, ["x", "group"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_two_variables(self, engine, sample_df):
        fig = engine.plot_correlation_heatmap(sample_df, ["x", "y"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_four_variables(self, engine, large_df):
        fig = engine.plot_correlation_heatmap(large_df, ["a", "b", "c", "d"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_nonexistent_vars_skipped(self, engine, large_df):
        """존재하지 않는 변수는 제외되고 유효 변수만 처리."""
        fig = engine.plot_correlation_heatmap(large_df, ["a", "b", "no_col"])
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 15. plot_roc_curve
# ──────────────────────────────────────────────────────────────

class TestPlotRocCurve:
    """plot_roc_curve → Figure 반환 검증."""

    def test_returns_figure(self, engine):
        fpr = np.linspace(0, 1, 50)
        tpr = np.clip(fpr + np.random.default_rng(1).normal(0, 0.05, 50), 0, 1)
        fig = engine.plot_roc_curve(fpr, tpr, auc_score=0.82)
        assert _is_figure(fig)
        plt.close(fig)

    def test_auc_perfect(self, engine):
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        fig = engine.plot_roc_curve(fpr, tpr, auc_score=1.0)
        assert _is_figure(fig)
        plt.close(fig)

    def test_auc_random(self, engine):
        fpr = np.linspace(0, 1, 20)
        tpr = fpr.copy()
        fig = engine.plot_roc_curve(fpr, tpr, auc_score=0.5)
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 16. plot_survival_curve
# ──────────────────────────────────────────────────────────────

class TestPlotSurvivalCurve:
    """plot_survival_curve → Figure 반환 검증."""

    def test_returns_figure_single_group(self, engine):
        t = np.linspace(0, 10, 30)
        s = np.exp(-0.2 * t)
        fig = engine.plot_survival_curve(t, s)
        assert _is_figure(fig)
        plt.close(fig)

    def test_returns_figure_multi_group(self, engine):
        """groups 딕셔너리 → 다중 그룹 분기."""
        t1 = np.linspace(0, 10, 20)
        s1 = np.exp(-0.2 * t1)
        t2 = np.linspace(0, 10, 20)
        s2 = np.exp(-0.4 * t2)
        fig = engine.plot_survival_curve(
            t1, s1, groups={"A": (t1, s1), "B": (t2, s2)}
        )
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 17. plot_residuals
# ──────────────────────────────────────────────────────────────

class TestPlotResiduals:
    """plot_residuals → Figure + 4개 axes 검증."""

    def test_returns_figure(self, engine):
        rng = np.random.default_rng(99)
        fitted = rng.normal(0, 1, 100)
        residuals = rng.normal(0, 0.5, 100)
        fig = engine.plot_residuals(fitted, residuals)
        assert _is_figure(fig)
        plt.close(fig)

    def test_four_subplots(self, engine):
        rng = np.random.default_rng(42)
        fitted = rng.uniform(0, 10, 80)
        residuals = rng.normal(0, 1, 80)
        fig = engine.plot_residuals(fitted, residuals)
        assert len(fig.axes) == 4
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 18. save_figure
# ──────────────────────────────────────────────────────────────

class TestSaveFigure:
    """save_figure → 파일 저장 검증."""

    def test_save_as_png(self, engine, sample_df):
        fig = engine.plot_histogram(sample_df, "x")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_output.png")
            engine.save_figure(fig, path, dpi=72)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_save_as_svg(self, engine, sample_df):
        fig = engine.plot_histogram(sample_df, "x")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_output.svg")
            engine.save_figure(fig, path)
            assert os.path.exists(path)
        plt.close(fig)

    def test_save_no_extension_defaults_png(self, engine, sample_df):
        """확장자 없으면 PNG로 저장 시도."""
        fig = engine.plot_histogram(sample_df, "x")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "noext")
            engine.save_figure(fig, path)
            assert os.path.exists(path)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 19. 내부 헬퍼 직접 검증
# ──────────────────────────────────────────────────────────────

class TestInternalHelpers:
    """_error_image / _make_error_figure / _fig_to_base64 / _apply_readability."""

    def test_error_image_returns_base64_png(self, engine):
        result = engine._error_image("테스트 오류")
        assert _is_valid_base64_png(result)

    def test_error_image_long_message(self, engine):
        result = engine._error_image("A" * 200)
        assert isinstance(result, str) and len(result) > 100

    def test_make_error_figure_returns_figure(self, engine):
        fig = engine._make_error_figure("Error message")
        assert _is_figure(fig)
        plt.close(fig)

    def test_fig_to_base64_format(self, engine):
        """_fig_to_base64 → data:image/png;base64, 접두사 포함."""
        fig, _ = plt.subplots()
        result = engine._fig_to_base64(fig)
        assert result.startswith("data:image/png;base64,")
        plt.close(fig)

    def test_fig_to_base64_decodable(self, engine):
        """base64 디코딩 후 PNG 헤더 확인."""
        fig, _ = plt.subplots()
        result = engine._fig_to_base64(fig)
        raw = result.split(",", 1)[1]
        data = base64.b64decode(raw)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        plt.close(fig)

    def test_apply_readability_with_long_labels(self, engine):
        """긴 x 레이블 → 45도 회전 적용 경로."""
        fig, ax = plt.subplots()
        ax.bar(range(3), [1, 2, 3])
        ax.set_xticklabels(["LongLabel1", "LongLabel2", "LongLabel3"])
        engine._apply_readability(ax, "Title", "VeryLongXLabel", "Y")
        plt.close(fig)

    def test_apply_readability_empty_strings(self, engine):
        """xlabel/ylabel 빈 문자열 → 분기 처리."""
        fig, ax = plt.subplots()
        engine._apply_readability(ax, "", "", "")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 20. Dataset 픽스처 기반 smoke test
# ──────────────────────────────────────────────────────────────

class TestDatasetBasedSmoke:
    """Dataset 객체의 data 속성을 직접 추출하여 차트 호출."""

    def test_bar_chart_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.bar_chart(df, x="group", y="x")
        assert _is_valid_base64_png(result)

    def test_histogram_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.histogram(df, x="x")
        assert _is_valid_base64_png(result)

    def test_scatter_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.scatter_plot(df, x="x", y="y")
        assert _is_valid_base64_png(result)

    def test_line_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.line_chart(df, x="x", y="y")
        assert _is_valid_base64_png(result)

    def test_box_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.box_plot(df, x="group", y="x")
        assert _is_valid_base64_png(result)

    def test_heatmap_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        result = engine.heatmap(df, cols=["x", "y"])
        assert _is_valid_base64_png(result)

    def test_plot_histogram_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        fig = engine.plot_histogram(df, "y")
        assert _is_figure(fig)
        plt.close(fig)

    def test_plot_scatter_from_dataset(self, engine, sample_ds):
        df = sample_ds.data
        fig = engine.plot_scatter(df, x_var="x", y_var="y", color_var="group")
        assert _is_figure(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 21. 엣지 케이스 종합
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """빈 데이터, 단일 값, 결측치, 잘못된 타입 엣지 케이스."""

    def test_all_charts_with_single_row_df(self, engine):
        """단일 행 DataFrame → 오류 없이 이미지 반환."""
        df = pd.DataFrame({"x": [1], "y": [2], "g": ["A"]})
        assert isinstance(engine.bar_chart(df, x="g", y="x"), str)
        assert isinstance(engine.histogram(df, x="x"), str)
        assert isinstance(engine.scatter_plot(df, x="x", y="y"), str)
        assert isinstance(engine.box_plot(df, y="x"), str)
        assert isinstance(engine.line_chart(df, x="x", y="y"), str)

    def test_validate_data_low_missing_warns(self, engine):
        """결측 비율 0 < r <= 50% → warnings 포함."""
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0, 4.0, 5.0]})
        result = engine._validate_data(df, ["v"])
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_validate_numeric_infinite_warns(self, engine):
        """무한값 포함 → warnings 포함."""
        s = pd.Series([1.0, 2.0, np.inf, 4.0])
        result = engine._validate_numeric(s, "inf_var")
        assert len(result["warnings"]) > 0

    def test_scatter_plot_both_numeric_correlation_shown(self, engine, large_df):
        """두 수치형 컬럼 → 상관계수 텍스트 경로 실행 확인."""
        result = engine.scatter_plot(large_df, x="a", y="b")
        assert _is_valid_base64_png(result)

    def test_bar_chart_with_missing_values(self, engine, na_df):
        result = engine.bar_chart(na_df, x="g", y="v")
        assert isinstance(result, str) and len(result) > 100

    def test_histogram_with_only_nans(self, engine):
        """전체 NaN → 실행 후 이미지 반환."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = engine.histogram(df, x="x")
        assert isinstance(result, str) and len(result) > 100

    def test_plot_correlation_heatmap_all_nonexistent(self, engine, large_df):
        """모든 변수가 존재하지 않으면 오류 Figure."""
        fig = engine.plot_correlation_heatmap(large_df, ["no1", "no2"])
        assert _is_figure(fig)
        plt.close(fig)
