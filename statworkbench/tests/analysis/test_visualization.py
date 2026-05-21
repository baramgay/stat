"""시각화 엔진(visualization) 테스트 — SW + TU 에이전트 공동 검증.

검증 항목:
- VisualizationEngine 인스턴스 생성
- 막대 차트, 히스토그램, 산점도 등 차트 생성 → base64 문자열 반환
- 검증 메서드: _validate_data, _validate_numeric
- 빈 데이터 / 잘못된 컬럼 → 오류 이미지 반환 (예외 아님)
- 크기 옵션 (small/medium/large/wide/square)
- 색맹 친화 팔레트 정의
- Figure 크기 매핑 정의

담당 에이전트: SW (statworkbench), TU (tester-unit)
"""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.visualization import VisualizationEngine


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> VisualizationEngine:
    return VisualizationEngine()


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """숫자형 컬럼 포함 DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "score": rng.normal(70, 15, 100),
        "age":   rng.integers(20, 60, 100).astype(float),
        "hours": rng.uniform(1, 10, 100),
    })


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """범주형 + 숫자형 혼합 DataFrame."""
    return pd.DataFrame({
        "group":  ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "score":  list(np.random.default_rng(0).normal(70, 10, 30)) +
                  list(np.random.default_rng(1).normal(80, 10, 30)) +
                  list(np.random.default_rng(2).normal(75, 10, 30)),
        "gender": (["M", "F"] * 45),
    })


def _is_valid_base64_png(s: str) -> bool:
    """base64 문자열이 유효한 PNG인지 확인. data URI 접두사 허용."""
    try:
        raw = s
        if "," in s:
            raw = s.split(",", 1)[1]
        data = base64.b64decode(raw)
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# 1. 엔진 초기화 및 설정
# ──────────────────────────────────────────────────────────────

class TestVisualizationEngineInit:
    """VisualizationEngine 초기화 및 상수 검증."""

    def test_instantiation(self, engine):
        assert isinstance(engine, VisualizationEngine)

    def test_color_palette_defined(self, engine):
        """색맹 친화 팔레트가 정의되어 있다."""
        assert hasattr(engine, "COLOR_PALETTE")
        assert len(engine.COLOR_PALETTE) >= 5

    def test_color_palette_hex_format(self, engine):
        """팔레트 색상이 HEX 형식."""
        for color in engine.COLOR_PALETTE:
            assert color.startswith("#"), f"HEX 색상 아님: {color}"
            assert len(color) == 7

    def test_figure_sizes_defined(self, engine):
        """차트 크기 매핑 정의."""
        assert hasattr(engine, "FIGURE_SIZES")
        for key in ["small", "medium", "large", "wide", "square"]:
            assert key in engine.FIGURE_SIZES

    def test_figure_sizes_are_tuples(self, engine):
        """크기 값이 (width, height) 튜플."""
        for key, size in engine.FIGURE_SIZES.items():
            assert isinstance(size, (tuple, list))
            assert len(size) == 2

    def test_figure_count_starts_zero(self, engine):
        assert engine._figure_count == 0


# ──────────────────────────────────────────────────────────────
# 2. 검증 메서드
# ──────────────────────────────────────────────────────────────

class TestValidationMethods:
    """_validate_data, _validate_numeric 검증."""

    def test_validate_data_valid(self, engine, numeric_df):
        result = engine._validate_data(numeric_df, ["score", "age"])
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_data_missing_column(self, engine, numeric_df):
        result = engine._validate_data(numeric_df, ["nonexistent"])
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_data_empty_df(self, engine):
        result = engine._validate_data(pd.DataFrame(), ["x"])
        assert result["valid"] is False

    def test_validate_data_none_df(self, engine):
        result = engine._validate_data(None, ["x"])
        assert result["valid"] is False

    def test_validate_data_high_missing_warns(self, engine):
        """결측 비율 > 50% → warnings 포함."""
        df = pd.DataFrame({"x": [1, np.nan, np.nan, np.nan]})
        result = engine._validate_data(df, ["x"])
        assert len(result["warnings"]) > 0

    def test_validate_numeric_valid(self, engine, numeric_df):
        result = engine._validate_numeric(numeric_df["score"], "score")
        assert result["valid"] is True

    def test_validate_numeric_string_column(self, engine, categorical_df):
        result = engine._validate_numeric(categorical_df["group"], "group")
        assert result["valid"] is False

    def test_validate_numeric_constant_warns(self, engine):
        """모든 값이 동일 → warnings 포함."""
        s = pd.Series([5.0] * 10)
        result = engine._validate_numeric(s, "constant")
        assert len(result["warnings"]) > 0


# ──────────────────────────────────────────────────────────────
# 3. 막대 차트
# ──────────────────────────────────────────────────────────────

class TestBarChart:
    """bar_chart: 반환값이 유효한 PNG base64."""

    def test_returns_string(self, engine, categorical_df):
        result = engine.bar_chart(categorical_df, x="group", y="score")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_valid_png(self, engine, categorical_df):
        result = engine.bar_chart(categorical_df, x="group", y="score")
        assert _is_valid_base64_png(result)

    def test_count_chart_no_y(self, engine, categorical_df):
        """y 없이 빈도 막대 차트."""
        result = engine.bar_chart(categorical_df, x="group")
        assert isinstance(result, str) and len(result) > 0

    def test_horizontal_orientation(self, engine, categorical_df):
        result = engine.bar_chart(categorical_df, x="group", y="score", orientation="horizontal")
        assert isinstance(result, str) and len(result) > 0

    def test_with_hue(self, engine, categorical_df):
        result = engine.bar_chart(categorical_df, x="group", y="score", hue="gender")
        assert isinstance(result, str) and len(result) > 0

    def test_invalid_column_returns_error_image(self, engine, categorical_df):
        """잘못된 컬럼 → base64 오류 이미지 반환 (예외 아님)."""
        result = engine.bar_chart(categorical_df, x="nonexistent", y="score")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_df_returns_error_image(self, engine):
        result = engine.bar_chart(pd.DataFrame(), x="x", y="y")
        assert isinstance(result, str) and len(result) > 0

    def test_size_options(self, engine, categorical_df):
        for sz in ["small", "medium", "large"]:
            result = engine.bar_chart(categorical_df, x="group", y="score", size=sz)
            assert isinstance(result, str) and len(result) > 0


# ──────────────────────────────────────────────────────────────
# 4. 히스토그램
# ──────────────────────────────────────────────────────────────

class TestHistogram:
    """histogram: 반환값이 유효한 PNG base64."""

    def test_returns_valid_png(self, engine, numeric_df):
        result = engine.histogram(numeric_df, x="score")
        assert _is_valid_base64_png(result)

    def test_with_kde(self, engine, numeric_df):
        result = engine.histogram(numeric_df, x="score", kde=True)
        assert isinstance(result, str) and len(result) > 0

    def test_without_kde(self, engine, numeric_df):
        result = engine.histogram(numeric_df, x="score", kde=False)
        assert isinstance(result, str) and len(result) > 0

    def test_custom_bins(self, engine, numeric_df):
        result = engine.histogram(numeric_df, x="score", bins=10)
        assert isinstance(result, str) and len(result) > 0

    def test_non_numeric_returns_error(self, engine, categorical_df):
        """문자형 컬럼 → base64 오류 이미지 반환."""
        result = engine.histogram(categorical_df, x="group")
        assert isinstance(result, str) and len(result) > 0

    def test_empty_df_returns_error(self, engine):
        result = engine.histogram(pd.DataFrame(), x="x")
        assert isinstance(result, str) and len(result) > 0


# ──────────────────────────────────────────────────────────────
# 5. 산점도 (scatter_plot이 있는 경우)
# ──────────────────────────────────────────────────────────────

class TestScatterPlot:
    """scatter_plot 또는 scatter: 반환값 검증."""

    def test_scatter_exists(self, engine):
        """scatter_plot 또는 scatter 메서드 존재."""
        has_scatter = hasattr(engine, "scatter_plot") or hasattr(engine, "scatter")
        assert has_scatter, "scatter_plot 또는 scatter 메서드가 없음"

    def test_scatter_returns_png(self, engine, numeric_df):
        method = getattr(engine, "scatter_plot", None) or getattr(engine, "scatter", None)
        if method is None:
            pytest.skip("산점도 메서드 없음")
        result = method(numeric_df, x="score", y="age")
        assert isinstance(result, str) and len(result) > 0


# ──────────────────────────────────────────────────────────────
# 6. 상자 그림 (box_plot이 있는 경우)
# ──────────────────────────────────────────────────────────────

class TestBoxPlot:
    """box_plot: 반환값 검증."""

    def test_box_plot_exists(self, engine):
        has_box = hasattr(engine, "box_plot") or hasattr(engine, "boxplot")
        assert has_box, "box_plot 메서드가 없음"

    def test_box_plot_returns_png(self, engine, categorical_df):
        method = getattr(engine, "box_plot", None) or getattr(engine, "boxplot", None)
        if method is None:
            pytest.skip("box_plot 메서드 없음")
        result = method(categorical_df, x="group", y="score")
        assert isinstance(result, str) and len(result) > 0


# ──────────────────────────────────────────────────────────────
# 7. 불변량: 결과는 항상 문자열
# ──────────────────────────────────────────────────────────────

class TestVisualizationInvariants:
    """모든 차트 메서드는 항상 str을 반환 (예외를 절대 발생시키지 않음)."""

    def test_bar_chart_never_raises(self, engine):
        """어떤 입력이어도 bar_chart는 str 반환."""
        for args in [
            (pd.DataFrame(), "x", None),
            (pd.DataFrame({"a": [1]}), "missing", None),
            (pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "x", "y"),
        ]:
            df, x, y = args
            result = engine.bar_chart(df, x=x, y=y)
            assert isinstance(result, str)

    def test_histogram_never_raises(self, engine):
        for df, x in [
            (pd.DataFrame(), "x"),
            (pd.DataFrame({"x": ["a", "b"]}), "x"),
            (pd.DataFrame({"x": [1, 2, 3]}), "x"),
        ]:
            result = engine.histogram(df, x=x)
            assert isinstance(result, str)
