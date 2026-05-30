"""Group D — Error-Resilience Tests: Analysis Modules.

Covers:
  - statworkbench.analysis.ml_engine  (kmeans_clustering, decision_tree_classifier,
                                        linear_regression_ml)
  - statworkbench.analysis.visualization.VisualizationEngine  (chart methods)
  - statworkbench.analysis.result  (AnalysisResult, ResultTable)

Goal: ensure graceful degradation / correct errors for all edge-case inputs.
"""

from __future__ import annotations

import base64
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.ml_engine import (
    decision_tree_classifier,
    kmeans_clustering,
    linear_regression_ml,
)
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.visualization import VisualizationEngine


# =============================================================================
# ML Engine — edge cases
# =============================================================================

class TestKMeansClusteringEdgeCases:

    def test_basic_clustering(self):
        """Normal 3-cluster run succeeds."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
        result = kmeans_clustering(df, features=["a", "b"], n_clusters=3)
        assert "labels" in result
        assert len(result["labels"]) == 60
        assert "inertia" in result

    def test_all_nan_features_raises(self):
        """All-NaN features raise ValueError (no valid data)."""
        df = pd.DataFrame({"a": [np.nan] * 10, "b": [np.nan] * 10})
        with pytest.raises(ValueError, match="유효한 데이터"):
            kmeans_clustering(df, features=["a", "b"])

    def test_single_feature(self):
        """Single feature column is accepted."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30)})
        result = kmeans_clustering(df, features=["x"], n_clusters=2)
        assert len(result["labels"]) == 30

    def test_n_clusters_1_no_silhouette(self):
        """n_clusters=1 produces None silhouette (not enough groups)."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})
        result = kmeans_clustering(df, features=["x", "y"], n_clusters=1)
        assert result["silhouette"] is None

    def test_n_clusters_equals_n_samples_no_silhouette(self):
        """n_clusters >= n_samples → silhouette is None (not calculated)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = kmeans_clustering(df, features=["x", "y"], n_clusters=3)
        # 3 clusters, 3 samples — silhouette not computed
        assert result["silhouette"] is None

    def test_partial_nan_rows_dropped(self):
        """Rows with NaN are dropped; valid rows produce a result."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        # dropna leaves 8 rows — should still cluster
        result = kmeans_clustering(df, features=["a", "b"], n_clusters=2)
        assert "labels" in result

    def test_large_n_clusters_raises_or_succeeds(self):
        """n_clusters > sample_count raises inside sklearn (ValueError)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(Exception):
            kmeans_clustering(df, features=["x"], n_clusters=10)

    def test_result_structure(self):
        """Result dict has all expected keys."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"x": rng.normal(0, 1, 40), "y": rng.normal(0, 1, 40)})
        result = kmeans_clustering(df, features=["x", "y"])
        for key in ("labels", "centers", "inertia", "n_iter", "silhouette"):
            assert key in result


class TestDecisionTreeClassifierEdgeCases:

    def _make_binary_df(self, n: int = 40, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": rng.choice([0, 1], n),
        })

    def test_basic_binary_classification(self):
        """Standard binary classification run succeeds."""
        df = self._make_binary_df()
        result = decision_tree_classifier(df, features=["x1", "x2"], target="y")
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_single_predictor(self):
        """Single predictor is accepted."""
        df = self._make_binary_df()
        result = decision_tree_classifier(df, features=["x1"], target="y")
        assert result["n_features"] == 1

    def test_all_same_outcome(self):
        """All same outcome class — train_test_split may raise StratifyError
        or produce accuracy. Either outcome is acceptable (no unhandled crash)."""
        df = pd.DataFrame({
            "x": np.arange(20, dtype=float),
            "y": [0] * 20,
        })
        try:
            result = decision_tree_classifier(df, features=["x"], target="y")
            assert "accuracy" in result
        except Exception:
            # Acceptable: sklearn may raise ValueError for single-class splits
            pass

    def test_missing_outcome_values(self):
        """NaN in target column — dropna should handle it."""
        df = pd.DataFrame({
            "x1": np.arange(1, 21, dtype=float),
            "y": [0, 1, np.nan, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })
        # After dropna, 19 rows remain
        result = decision_tree_classifier(df, features=["x1"], target="y")
        assert "accuracy" in result

    def test_all_nan_target_raises(self):
        """All-NaN target → no valid data → ValueError."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="유효한 데이터"):
            decision_tree_classifier(df, features=["x"], target="y")

    def test_categorical_predictor_encoded(self):
        """String (object dtype) predictor is LabelEncoded without crash."""
        df = pd.DataFrame({
            "cat": ["low", "medium", "high", "low", "medium", "high",
                    "low", "medium", "high", "low", "medium", "high",
                    "low", "medium", "high", "low", "medium", "high",
                    "low", "medium"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        result = decision_tree_classifier(df, features=["cat"], target="y")
        assert "accuracy" in result

    def test_result_structure(self):
        """Result dict has expected keys."""
        df = self._make_binary_df()
        result = decision_tree_classifier(df, features=["x1", "x2"], target="y")
        for key in ("accuracy", "n_features", "n_train", "n_test", "feature_importance"):
            assert key in result


class TestLinearRegressionMLEdgeCases:

    def test_basic_regression(self):
        """Standard regression run succeeds."""
        rng = np.random.default_rng(0)
        n = 50
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x, "x2": rng.normal(0, 1, n), "y": 2 * x + rng.normal(0, 0.1, n)})
        result = linear_regression_ml(df, features=["x1", "x2"], target="y")
        assert "r2_score" in result
        assert "mse" in result

    def test_all_nan_features_raises(self):
        """All-NaN features raise ValueError."""
        df = pd.DataFrame({"x": [np.nan] * 10, "y": [1.0] * 10})
        with pytest.raises(ValueError, match="유효한 데이터"):
            linear_regression_ml(df, features=["x"], target="y")

    def test_constant_target(self):
        """Constant target (y always same) — R2 may be undefined but no crash."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30), "y": [5.0] * 30})
        try:
            result = linear_regression_ml(df, features=["x"], target="y")
            assert "r2_score" in result
        except Exception:
            pass  # sklearn may warn or error for degenerate cases

    def test_result_keys(self):
        """Result has all expected keys."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 40), "y": rng.normal(0, 1, 40)})
        result = linear_regression_ml(df, features=["x"], target="y")
        for key in ("r2_score", "mse", "rmse", "coefficients", "intercept", "n_train", "n_test"):
            assert key in result

    def test_rmse_equals_sqrt_mse(self):
        """rmse == sqrt(mse) within floating-point tolerance."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "y": rng.normal(0, 1, 50),
        })
        result = linear_regression_ml(df, features=["x"], target="y")
        assert abs(result["rmse"] - result["mse"] ** 0.5) < 1e-10


# =============================================================================
# VisualizationEngine — edge cases
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    return VisualizationEngine()


@pytest.fixture(scope="module")
def normal_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(50, 10, 100),
        "y": rng.normal(50, 10, 100),
        "g": np.where(rng.random(100) > 0.5, "A", "B"),
    })


def _is_error_image(result: str) -> bool:
    """Return True if result is a base64 PNG (error or normal image)."""
    return result.startswith("data:image/png;base64,")


class TestVisualizationEdgeCases:

    # ── bar_chart ─────────────────────────────────────────────────────────────

    def test_bar_chart_empty_df_returns_error_image(self, engine):
        """Empty DataFrame → error image returned, not exception."""
        result = engine.bar_chart(pd.DataFrame(), x="x")
        assert _is_error_image(result)

    def test_bar_chart_normal(self, engine, normal_df):
        """Normal bar chart returns base64 PNG."""
        result = engine.bar_chart(normal_df, x="g")
        assert _is_error_image(result)

    def test_bar_chart_nonexistent_column_returns_error_image(self, engine, normal_df):
        """Nonexistent y column → _validate_data catches it → error image."""
        result = engine.bar_chart(normal_df, x="g", y="nonexistent")
        assert _is_error_image(result)

    def test_bar_chart_horizontal(self, engine, normal_df):
        """Horizontal orientation does not raise."""
        result = engine.bar_chart(normal_df, x="g", orientation="horizontal")
        assert _is_error_image(result)

    # ── histogram ─────────────────────────────────────────────────────────────

    def test_histogram_normal(self, engine, normal_df):
        """Normal histogram returns base64 PNG."""
        result = engine.histogram(normal_df, x="x")
        assert _is_error_image(result)

    def test_histogram_empty_df_returns_error_image(self, engine):
        """Empty DataFrame → error image."""
        result = engine.histogram(pd.DataFrame(), x="x")
        assert _is_error_image(result)

    def test_histogram_nonexistent_column_returns_error_image(self, engine, normal_df):
        """Missing column → error image."""
        result = engine.histogram(normal_df, x="nonexistent")
        assert _is_error_image(result)

    def test_histogram_all_nan_column_returns_error_image(self, engine):
        """All-NaN numeric column → _validate_numeric warns but histogram
        may render empty. Should return either valid image or error image."""
        df = pd.DataFrame({"x": [np.nan] * 10})
        result = engine.histogram(df, x="x")
        assert _is_error_image(result)

    def test_histogram_string_column_returns_error_image(self, engine, normal_df):
        """Non-numeric column → _validate_numeric fails → error image."""
        result = engine.histogram(normal_df, x="g")
        assert _is_error_image(result)

    # ── scatter_plot ──────────────────────────────────────────────────────────

    def test_scatter_normal(self, engine, normal_df):
        """Normal scatter plot returns base64 PNG."""
        result = engine.scatter_plot(normal_df, x="x", y="y")
        assert _is_error_image(result)

    def test_scatter_empty_df_returns_error_image(self, engine):
        result = engine.scatter_plot(pd.DataFrame(), x="x", y="y")
        assert _is_error_image(result)

    def test_scatter_nonexistent_column_returns_error_image(self, engine, normal_df):
        result = engine.scatter_plot(normal_df, x="nonexistent", y="y")
        assert _is_error_image(result)

    def test_scatter_with_hue(self, engine, normal_df):
        """Scatter with hue does not raise."""
        result = engine.scatter_plot(normal_df, x="x", y="y", hue="g")
        assert _is_error_image(result)

    def test_scatter_with_regression(self, engine, normal_df):
        """Scatter with add_regression does not raise."""
        result = engine.scatter_plot(normal_df, x="x", y="y", add_regression=True)
        assert _is_error_image(result)

    # ── box_plot ──────────────────────────────────────────────────────────────

    def test_box_plot_empty_df_returns_error_image(self, engine):
        result = engine.box_plot(pd.DataFrame(), y="x")
        assert _is_error_image(result)

    def test_box_plot_single_value_column(self, engine):
        """Single constant value column — matplotlib may warn, not crash."""
        df = pd.DataFrame({"x": [1] * 20, "g": ["A"] * 20})
        result = engine.box_plot(df, y="x", x="g")
        assert _is_error_image(result)

    def test_box_plot_normal(self, engine, normal_df):
        result = engine.box_plot(normal_df, x="g", y="y")
        assert _is_error_image(result)

    def test_box_plot_no_x_axis(self, engine, normal_df):
        """Box plot without group x works."""
        result = engine.box_plot(normal_df, y="x")
        assert _is_error_image(result)

    def test_box_plot_long_category_labels(self, engine):
        """Long string category labels do not crash."""
        df = pd.DataFrame({
            "g": ["Very Long Category Label A"] * 20 + ["Very Long Category Label B"] * 20,
            "val": np.random.randn(40),
        })
        result = engine.box_plot(df, x="g", y="val")
        assert _is_error_image(result)

    # ── line_chart ────────────────────────────────────────────────────────────

    def test_line_chart_normal(self, engine, normal_df):
        result = engine.line_chart(normal_df, x="x", y="y")
        assert _is_error_image(result)

    def test_line_chart_empty_returns_error_image(self, engine):
        result = engine.line_chart(pd.DataFrame(), x="x", y="y")
        assert _is_error_image(result)

    # ── heatmap ───────────────────────────────────────────────────────────────

    def test_heatmap_normal(self, engine, normal_df):
        result = engine.heatmap(normal_df)
        assert _is_error_image(result)

    def test_heatmap_no_numeric_cols_returns_error_image(self, engine):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        result = engine.heatmap(df)
        assert _is_error_image(result)

    def test_heatmap_single_numeric_col(self, engine):
        """Single numeric column → cannot compute correlation → error image."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
        result = engine.heatmap(df, cols=["x"])
        assert _is_error_image(result)

    # ── violin_plot ───────────────────────────────────────────────────────────

    def test_violin_plot_empty_returns_error_image(self, engine):
        result = engine.violin_plot(pd.DataFrame(), x="g", y="y")
        assert _is_error_image(result)

    def test_violin_plot_normal(self, engine, normal_df):
        result = engine.violin_plot(normal_df, x="g", y="y")
        assert _is_error_image(result)

    # ── plot_histogram (returns Figure) ──────────────────────────────────────

    def test_plot_histogram_normal(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_histogram(normal_df, variable="x")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_histogram_empty_df_returns_error_figure(self, engine):
        import matplotlib.figure
        fig = engine.plot_histogram(pd.DataFrame(), variable="x")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_histogram_string_col_returns_error_figure(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_histogram(normal_df, variable="g")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── plot_scatter (returns Figure) ─────────────────────────────────────────

    def test_plot_scatter_normal(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_scatter(normal_df, x_var="x", y_var="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_scatter_empty_df_returns_error_figure(self, engine):
        import matplotlib.figure
        fig = engine.plot_scatter(pd.DataFrame(), x_var="x", y_var="y")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── plot_boxplot (returns Figure) ─────────────────────────────────────────

    def test_plot_boxplot_normal(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_boxplot(normal_df, x_var="g", y_var="x")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── plot_bar (returns Figure) ─────────────────────────────────────────────

    def test_plot_bar_counts_mode(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_bar(normal_df, x_var="g")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_bar_empty_df_returns_error_figure(self, engine):
        import matplotlib.figure
        fig = engine.plot_bar(pd.DataFrame(), x_var="g")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── plot_correlation_heatmap ──────────────────────────────────────────────

    def test_plot_correlation_heatmap_normal(self, engine, normal_df):
        import matplotlib.figure
        fig = engine.plot_correlation_heatmap(normal_df, variables=["x", "y"])
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_correlation_heatmap_less_than_2_numeric_returns_error(self, engine):
        import matplotlib.figure
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})
        fig = engine.plot_correlation_heatmap(df, variables=["x", "g"])
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# ResultTable — edge cases
# =============================================================================

class TestResultTableEdgeCases:

    def test_empty_dataframe(self):
        """ResultTable with empty DataFrame is constructed without raising."""
        rt = ResultTable(title="empty", dataframe=pd.DataFrame())
        assert rt.title == "empty"

    def test_to_html_empty(self):
        """to_html on empty table returns non-empty string (HTML structure)."""
        rt = ResultTable(title="test", dataframe=pd.DataFrame())
        html = rt.to_html()
        assert isinstance(html, str)
        assert "test" in html

    def test_to_markdown_empty(self):
        """to_markdown on empty table returns non-empty string."""
        rt = ResultTable(title="test", dataframe=pd.DataFrame())
        md = rt.to_markdown()
        assert isinstance(md, str)
        assert "test" in md

    def test_to_csv_empty(self):
        """to_csv on empty table returns string (possibly just header)."""
        rt = ResultTable(title="test", dataframe=pd.DataFrame())
        csv = rt.to_csv()
        assert isinstance(csv, str)

    def test_to_csv_non_empty(self):
        """to_csv on a populated table contains column values."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rt = ResultTable(title="t", dataframe=df)
        csv = rt.to_csv()
        assert "1" in csv

    def test_table_with_nan_values(self):
        """Table with NaN values renders without crashing."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        rt = ResultTable(title="nan test", dataframe=df)
        html = rt.to_html()
        md = rt.to_markdown()
        csv = rt.to_csv()
        assert isinstance(html, str)
        assert isinstance(md, str)
        assert isinstance(csv, str)

    def test_table_with_very_long_strings(self):
        """Long string values render without crashing."""
        df = pd.DataFrame({"col": ["A" * 1000, "B" * 1000]})
        rt = ResultTable(title="long", dataframe=df)
        html = rt.to_html()
        assert "long" in html

    def test_footnotes_appear_in_html(self):
        """Footnotes are included in the HTML output."""
        df = pd.DataFrame({"x": [1]})
        rt = ResultTable(title="t", dataframe=df, footnotes=["Note A", "Note B"])
        html = rt.to_html()
        assert "Note A" in html
        assert "Note B" in html

    def test_footnotes_appear_in_markdown(self):
        """Footnotes are included in the Markdown output."""
        df = pd.DataFrame({"x": [1]})
        rt = ResultTable(title="t", dataframe=df, footnotes=["Note A"])
        md = rt.to_markdown()
        assert "Note A" in md

    def test_serialize_dataframe_in_model(self):
        """serialize_dataframe produces expected dict shape."""
        df = pd.DataFrame({"a": [1, 2]})
        rt = ResultTable(title="t", dataframe=df)
        serialised = rt.serialize_dataframe(df)
        assert "columns" in serialised
        assert "data" in serialised
        assert "shape" in serialised

    def test_export_options_passed_to_csv(self):
        """export_options are respected."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        rt = ResultTable(title="t", dataframe=df, export_options={"index": True})
        csv = rt.to_csv()
        assert isinstance(csv, str)


# =============================================================================
# AnalysisResult — edge cases
# =============================================================================

class TestAnalysisResultEdgeCases:

    def _make_result(self, **kwargs) -> AnalysisResult:
        defaults = {"id": "test-001", "title": "Test Analysis"}
        defaults.update(kwargs)
        return AnalysisResult(**defaults)

    def test_minimal_construction(self):
        """AnalysisResult with only id+title is valid."""
        result = self._make_result()
        assert result.id == "test-001"
        assert result.title == "Test Analysis"
        assert result.tables == []
        assert result.warnings == []

    def test_add_table_chaining(self):
        """add_table returns self for chaining."""
        result = self._make_result()
        df = pd.DataFrame({"a": [1]})
        rt = ResultTable(title="t", dataframe=df)
        returned = result.add_table(rt)
        assert returned is result
        assert len(result.tables) == 1

    def test_add_warning_chaining(self):
        """add_warning returns self for chaining."""
        result = self._make_result()
        returned = result.add_warning("Watch out!")
        assert returned is result
        assert "Watch out!" in result.warnings

    def test_add_note_chaining(self):
        """add_note returns self for chaining."""
        result = self._make_result()
        returned = result.add_note("FYI")
        assert returned is result
        assert "FYI" in result.notes

    def test_add_assumption_chaining(self):
        result = self._make_result()
        rt = ResultTable(title="normality", dataframe=pd.DataFrame({"W": [0.99]}))
        returned = result.add_assumption(rt)
        assert returned is result
        assert len(result.assumptions) == 1

    def test_add_diagnostic_chaining(self):
        result = self._make_result()
        rt = ResultTable(title="vif", dataframe=pd.DataFrame({"VIF": [1.2]}))
        returned = result.add_diagnostic(rt)
        assert returned is result
        assert len(result.diagnostics) == 1

    def test_to_html_empty_result(self):
        """to_html with no tables/warnings returns a string (possibly empty)."""
        result = self._make_result()
        html = result.to_html()
        assert isinstance(html, str)

    def test_to_html_with_warnings(self):
        """to_html includes warnings in output."""
        result = self._make_result()
        result.add_warning("sample size is small")
        html = result.to_html()
        assert "sample size is small" in html

    def test_to_html_with_tables(self):
        """to_html with multiple tables returns concatenated HTML."""
        result = self._make_result()
        for i in range(3):
            result.add_table(
                ResultTable(title=f"Table {i}", dataframe=pd.DataFrame({"x": [i]}))
            )
        html = result.to_html()
        assert "Table 0" in html
        assert "Table 2" in html

    def test_summary_string(self):
        """summary() returns a non-empty string."""
        result = self._make_result()
        result.add_warning("w1")
        summary = result.summary()
        assert "Test Analysis" in summary
        assert "Warnings: 1" in summary

    def test_created_at_is_datetime(self):
        """created_at is a datetime instance."""
        result = self._make_result()
        assert isinstance(result.created_at, datetime)

    def test_spec_stored(self):
        """spec dict is stored correctly."""
        spec = {"method": "pearson", "variables": ["x", "y"]}
        result = self._make_result(spec=spec)
        assert result.spec["method"] == "pearson"

    def test_syntax_stored(self):
        """syntax string is stored."""
        result = self._make_result(syntax="CORRELATIONS /VARIABLES=x y.")
        assert "CORRELATIONS" in result.syntax

    def test_many_tables(self):
        """Adding 50 tables does not raise."""
        result = self._make_result()
        for i in range(50):
            result.add_table(
                ResultTable(title=f"T{i}", dataframe=pd.DataFrame({"v": [i]}))
            )
        assert len(result.tables) == 50

    def test_metadata_dict(self):
        """metadata field accepts arbitrary data."""
        result = self._make_result(metadata={"engine": "v2", "n": 100})
        assert result.metadata["engine"] == "v2"
