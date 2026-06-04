"""Advanced tests for AnalysisRegistry and result models.

Targets:
- analysis/registry.py  (currently 78%) -> raise coverage
- analysis/result.py    (currently 93%) -> raise coverage
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
import types

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.analysis.registry import (
    AnalysisRegistry,
    _PlannedAnalysis,
    _ModulePlugin,
    _BUILTIN_ANALYSES,
    _register_new_plugins,
)
from nuristat.analysis.result import AnalysisResult, ResultTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ds() -> Dataset:
    return Dataset(
        data=pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
        name="simple",
    )


@pytest.fixture
def fresh_registry() -> AnalysisRegistry:
    return AnalysisRegistry()


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def basic_table(small_df: pd.DataFrame) -> ResultTable:
    return ResultTable(title="Test Table", dataframe=small_df)


@pytest.fixture
def basic_result() -> AnalysisResult:
    return AnalysisResult(id="r-001", title="Test Analysis")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockPlugin:
    """Generic mock plugin with configurable attributes."""

    def __init__(self, pid: str = "mock_plugin", name: str = "Mock", category: str = "Test", implemented: bool = True) -> None:
        self.id = pid
        self.name = name
        self.category = category
        self.description = "A mock plugin."
        self.variable_requirements: list[dict] = []
        self.implemented = implemented

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        return AnalysisResult(id="mock-001", title=self.name)


# ===========================================================================
# Group A — registry.categories()
# ===========================================================================

class TestCategories:
    """registry.categories() — sorted, unique, strings."""

    def test_categories_returns_list(self, fresh_registry: AnalysisRegistry) -> None:
        cats = fresh_registry.categories()
        assert isinstance(cats, list)

    def test_categories_are_strings(self, fresh_registry: AnalysisRegistry) -> None:
        for c in fresh_registry.categories():
            assert isinstance(c, str)

    def test_categories_are_sorted(self, fresh_registry: AnalysisRegistry) -> None:
        cats = fresh_registry.categories()
        assert cats == sorted(cats)

    def test_categories_no_duplicates(self, fresh_registry: AnalysisRegistry) -> None:
        cats = fresh_registry.categories()
        assert len(cats) == len(set(cats))

    def test_categories_includes_known_builtin(self, fresh_registry: AnalysisRegistry) -> None:
        cats = fresh_registry.categories()
        # built-in planned analyses include these categories
        assert "Descriptive Statistics" in cats

    def test_categories_grows_after_register(self, fresh_registry: AnalysisRegistry) -> None:
        new_cat_plugin = _MockPlugin(pid="new_cat_p", category="Exotic Category")
        before = set(fresh_registry.categories())
        fresh_registry.register(new_cat_plugin)
        after = set(fresh_registry.categories())
        assert "Exotic Category" in after - before


# ===========================================================================
# Group B — list_by_category()
# ===========================================================================

class TestListByCategory:
    """list_by_category() — correct filtering."""

    def test_known_category_returns_nonempty(self, fresh_registry: AnalysisRegistry) -> None:
        result = fresh_registry.list_by_category("Descriptive Statistics")
        assert len(result) >= 1

    def test_all_returned_have_correct_category(self, fresh_registry: AnalysisRegistry) -> None:
        for p in fresh_registry.list_by_category("Nonparametric Tests"):
            assert p.category == "Nonparametric Tests"

    def test_unknown_category_returns_empty_list(self, fresh_registry: AnalysisRegistry) -> None:
        result = fresh_registry.list_by_category("__nonexistent__")
        assert result == []

    def test_results_sorted_by_name(self, fresh_registry: AnalysisRegistry) -> None:
        result = fresh_registry.list_by_category("Descriptive Statistics")
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_multiple_plugins_in_same_category(self, fresh_registry: AnalysisRegistry) -> None:
        p1 = _MockPlugin(pid="alpha_test", name="Alpha", category="CustomCat")
        p2 = _MockPlugin(pid="beta_test", name="Beta", category="CustomCat")
        fresh_registry.register(p1)
        fresh_registry.register(p2)
        result = fresh_registry.list_by_category("CustomCat")
        ids = {p.id for p in result}
        assert {"alpha_test", "beta_test"} == ids


# ===========================================================================
# Group C — get()
# ===========================================================================

class TestGet:
    """get() — found / not found."""

    def test_get_planned_plugin_exists(self, fresh_registry: AnalysisRegistry) -> None:
        # 'kaplan_meier' is a truly unimplemented (planned) analysis
        plugin = fresh_registry.get("kaplan_meier")
        assert plugin.id == "kaplan_meier"

    def test_get_nonexistent_raises_key_error(self, fresh_registry: AnalysisRegistry) -> None:
        with pytest.raises(KeyError, match="not found"):
            fresh_registry.get("does_not_exist_xyz")

    def test_get_after_register(self, fresh_registry: AnalysisRegistry) -> None:
        p = _MockPlugin(pid="new_one")
        fresh_registry.register(p)
        assert fresh_registry.get("new_one") is p

    def test_get_after_unregister_raises(self, fresh_registry: AnalysisRegistry) -> None:
        # unregister a planned plugin that exists
        fresh_registry.unregister("kaplan_meier")
        with pytest.raises(KeyError):
            fresh_registry.get("kaplan_meier")


# ===========================================================================
# Group D — execute (via plugin.run()) + error propagation
# ===========================================================================

class TestExecute:
    """Running a registered plugin and handling failures."""

    def test_run_mock_plugin_returns_result(
        self, fresh_registry: AnalysisRegistry, simple_ds: Dataset
    ) -> None:
        p = _MockPlugin(pid="runnable")
        fresh_registry.register(p)
        result = fresh_registry.get("runnable").run(simple_ds, {})
        assert isinstance(result, AnalysisResult)

    def test_run_planned_plugin_raises_not_implemented(
        self, simple_ds: Dataset
    ) -> None:
        pa = _PlannedAnalysis("planned_stub", "Planned Stub", "Test", "stub")
        with pytest.raises(NotImplementedError):
            pa.run(simple_ds, {})

    def test_run_validate_planned_raises_not_implemented(
        self, simple_ds: Dataset
    ) -> None:
        pa = _PlannedAnalysis("planned_stub2", "Planned Stub2", "Test", "stub")
        with pytest.raises(NotImplementedError):
            pa.validate(simple_ds, {})

    def test_run_plugin_exception_propagates(
        self, fresh_registry: AnalysisRegistry, simple_ds: Dataset
    ) -> None:
        class BrokenPlugin:
            id = "broken"
            name = "Broken"
            category = "Test"
            description = "Explodes on run."
            variable_requirements: list = []
            implemented = True

            def validate(self, dataset: Dataset, spec: dict) -> list[str]:
                return []

            def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
                raise RuntimeError("Deliberate failure")

        fresh_registry.register(BrokenPlugin())
        with pytest.raises(RuntimeError, match="Deliberate failure"):
            fresh_registry.get("broken").run(simple_ds, {})


# ===========================================================================
# Group E — _ModulePlugin load success / failure
# ===========================================================================

class TestModulePlugin:
    """_ModulePlugin lazy loading behaviour."""

    def test_module_plugin_attributes(self) -> None:
        mp = _ModulePlugin(
            plugin_id="test_mp",
            name="Test MP",
            category="Test",
            description="A module plugin.",
            module_path="nuristat.analysis.ttests",
        )
        assert mp.id == "test_mp"
        assert mp.implemented is True
        assert mp._module is None  # not loaded yet

    def test_module_plugin_load_success(self) -> None:
        mp = _ModulePlugin(
            plugin_id="ttests_mp",
            name="TTests MP",
            category="Compare Means",
            description="Wraps ttests.",
            module_path="nuristat.analysis.ttests",
        )
        mp._load()
        assert mp._module is not None

    def test_module_plugin_load_failure_raises_import_error(self) -> None:
        mp = _ModulePlugin(
            plugin_id="bad_mod",
            name="Bad Mod",
            category="Test",
            description="Points to non-existent module.",
            module_path="nuristat.analysis.__nonexistent_module__",
        )
        with pytest.raises(ModuleNotFoundError):
            mp._load()

    def test_module_plugin_validate_no_variables(self, simple_ds: Dataset) -> None:
        mp = _ModulePlugin(
            plugin_id="mp_validate",
            name="MP Validate",
            category="Test",
            description="Validate test.",
            module_path="nuristat.analysis.ttests",
        )
        errors = mp.validate(simple_ds, {})
        assert len(errors) > 0  # no variables specified

    def test_module_plugin_validate_with_variables(self, simple_ds: Dataset) -> None:
        mp = _ModulePlugin(
            plugin_id="mp_val2",
            name="MP Val2",
            category="Test",
            description="Validate test.",
            module_path="nuristat.analysis.ttests",
        )
        errors = mp.validate(simple_ds, {"variables": {"x": "dependent"}})
        assert errors == []

    def test_module_plugin_custom_function_name(self) -> None:
        mp = _ModulePlugin(
            plugin_id="custom_fn",
            name="Custom FN",
            category="Compare Means",
            description="Uses custom function.",
            module_path="nuristat.analysis.ttests",
            function_name="run_one_sample_analysis",
        )
        assert mp._function_name == "run_one_sample_analysis"

    def test_module_plugin_run_calls_correct_function(self, simple_ds: Dataset) -> None:
        fake_result = AnalysisResult(id="fake-001", title="Fake")
        fake_fn = MagicMock(return_value=fake_result)
        fake_module = types.SimpleNamespace(my_fn=fake_fn)

        mp = _ModulePlugin(
            plugin_id="fn_test",
            name="FN Test",
            category="Test",
            description="Tests function dispatch.",
            module_path="nuristat.analysis.ttests",
            function_name="my_fn",
        )
        mp._module = fake_module  # inject fake module
        result = mp.run(simple_ds, {"variables": {}})
        fake_fn.assert_called_once_with(simple_ds, {"variables": {}})
        assert result is fake_result


# ===========================================================================
# Group F — _PlannedAnalysis
# ===========================================================================

class TestPlannedAnalysisAdvanced:
    """_PlannedAnalysis — run/validate raise, attributes correct."""

    def test_run_raises_not_implemented_with_id_in_message(self, simple_ds: Dataset) -> None:
        pa = _PlannedAnalysis(id="roc_analysis", name="ROC", category="Diagnostic Tests", description="ROC.")
        with pytest.raises(NotImplementedError, match="roc_analysis"):
            pa.run(simple_ds, {})

    def test_validate_raises_not_implemented_with_id_in_message(self, simple_ds: Dataset) -> None:
        pa = _PlannedAnalysis(id="roc_analysis", name="ROC", category="Diagnostic Tests", description="ROC.")
        with pytest.raises(NotImplementedError, match="roc_analysis"):
            pa.validate(simple_ds, {})

    def test_planned_implemented_is_false(self) -> None:
        pa = _PlannedAnalysis(id="x", name="X", category="C", description="D")
        assert pa.implemented is False

    def test_planned_variable_requirements_from_template(self) -> None:
        pa = _PlannedAnalysis(id="normality", name="Normality", category="Descriptive Statistics", description=".")
        assert isinstance(pa.variable_requirements, list)
        assert len(pa.variable_requirements) >= 1

    def test_planned_unknown_id_has_empty_requirements(self) -> None:
        pa = _PlannedAnalysis(id="future_magic", name="Magic", category="Other", description=".")
        assert pa.variable_requirements == []


# ===========================================================================
# Group G — all registered analysis IDs + implemented vs planned
# ===========================================================================

class TestRegistryContents:
    """Inspect the full registry contents."""

    def test_all_builtin_planned_ids_present(self, fresh_registry: AnalysisRegistry) -> None:
        """AnalysisRegistry.__init__ registers only planned (implemented=False) entries."""
        all_ids = {p.id for p in fresh_registry.list_all()}
        planned_ids = {info["id"] for info in _BUILTIN_ANALYSES if not info["implemented"]}
        assert planned_ids.issubset(all_ids)

    def test_implemented_ids_present_after_register_new_plugins(self) -> None:
        """After _register_new_plugins, implemented analyses are in the registry."""
        registry = AnalysisRegistry()
        _register_new_plugins(registry)
        actual_ids = {p.id for p in registry.list_implemented()}
        # A sampling of IDs that _register_new_plugins should add
        for expected_id in ["logistic_regression", "factor_analysis", "cluster_analysis"]:
            assert expected_id in actual_ids

    def test_planned_are_only_unimplemented(self, fresh_registry: AnalysisRegistry) -> None:
        for p in fresh_registry.list_planned():
            assert p.implemented is False

    def test_register_new_plugins_replaces_stubs(self) -> None:
        registry = AnalysisRegistry()
        _register_new_plugins(registry)
        # After _register_new_plugins, e.g. logistic_regression should be a _ModulePlugin
        plugin = registry.get("logistic_regression")
        assert isinstance(plugin, _ModulePlugin)
        assert plugin.implemented is True

    def test_contains_operator(self, fresh_registry: AnalysisRegistry) -> None:
        assert "explore" in fresh_registry
        assert "nonexistent_xyz_abc" not in fresh_registry

    def test_len_covers_builtin_planned(self, fresh_registry: AnalysisRegistry) -> None:
        n_planned = len([info for info in _BUILTIN_ANALYSES if not info["implemented"]])
        assert len(fresh_registry) >= n_planned


# ===========================================================================
# Group H — ResultTable
# ===========================================================================

class TestResultTable:
    """ResultTable creation and HTML rendering."""

    def test_to_html_contains_title(self, basic_table: ResultTable) -> None:
        html = basic_table.to_html()
        assert "Test Table" in html

    def test_to_html_contains_data(self, basic_table: ResultTable) -> None:
        html = basic_table.to_html()
        assert "1" in html  # row value

    def test_to_html_no_footnotes(self, basic_table: ResultTable) -> None:
        html = basic_table.to_html()
        assert "<tfoot>" not in html

    def test_to_html_with_footnotes(self, small_df: pd.DataFrame) -> None:
        table = ResultTable(title="Titled", dataframe=small_df, footnotes=["Note A", "Note B"])
        html = table.to_html()
        assert "<tfoot>" in html
        assert "Note A" in html
        assert "Note B" in html

    def test_footnotes_count_in_html(self, small_df: pd.DataFrame) -> None:
        table = ResultTable(title="T", dataframe=small_df, footnotes=["F1", "F2", "F3"])
        html = table.to_html()
        assert html.count("<small>") == 3

    def test_empty_dataframe_renders(self) -> None:
        table = ResultTable(title="Empty", dataframe=pd.DataFrame())
        html = table.to_html()
        assert "Empty" in html

    def test_to_markdown_contains_title(self, basic_table: ResultTable) -> None:
        md = basic_table.to_markdown()
        assert "Test Table" in md

    def test_to_csv_basic(self, basic_table: ResultTable) -> None:
        csv = basic_table.to_csv()
        assert "a" in csv or "b" in csv  # column headers

    def test_serialize_dataframe(self, basic_table: ResultTable) -> None:
        serialized = basic_table.serialize_dataframe(basic_table.dataframe)
        assert "columns" in serialized
        assert "data" in serialized
        assert "shape" in serialized


# ===========================================================================
# Group I — AnalysisResult
# ===========================================================================

class TestAnalysisResult:
    """AnalysisResult construction and methods."""

    def test_empty_result_to_html_is_empty_string(self, basic_result: AnalysisResult) -> None:
        html = basic_result.to_html()
        assert html == ""

    def test_add_table_appends(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_table(basic_table)
        assert len(basic_result.tables) == 1
        assert basic_result.tables[0] is basic_table

    def test_add_table_returns_self(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        returned = basic_result.add_table(basic_table)
        assert returned is basic_result

    def test_to_html_with_one_table(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_table(basic_table)
        html = basic_result.to_html()
        assert "Test Table" in html

    def test_to_html_table_order_preserved(self, small_df: pd.DataFrame) -> None:
        result = AnalysisResult(id="order-test", title="Order")
        t1 = ResultTable(title="First", dataframe=small_df)
        t2 = ResultTable(title="Second", dataframe=small_df)
        t3 = ResultTable(title="Third", dataframe=small_df)
        result.add_table(t1).add_table(t2).add_table(t3)
        html = result.to_html()
        pos1 = html.index("First")
        pos2 = html.index("Second")
        pos3 = html.index("Third")
        assert pos1 < pos2 < pos3

    def test_to_html_with_warnings(self, basic_result: AnalysisResult) -> None:
        basic_result.add_warning("Sample size is small.")
        html = basic_result.to_html()
        assert "Sample size is small." in html
        assert "warnings" in html

    def test_to_html_multiple_warnings(self, basic_result: AnalysisResult) -> None:
        basic_result.add_warning("Warning one")
        basic_result.add_warning("Warning two")
        html = basic_result.to_html()
        assert "Warning one" in html
        assert "Warning two" in html

    def test_to_html_warnings_ul_structure(self, basic_result: AnalysisResult) -> None:
        basic_result.add_warning("W")
        html = basic_result.to_html()
        assert "<ul>" in html and "<li>" in html

    def test_add_note_appends(self, basic_result: AnalysisResult) -> None:
        basic_result.add_note("Informational note.")
        assert "Informational note." in basic_result.notes

    def test_add_note_returns_self(self, basic_result: AnalysisResult) -> None:
        returned = basic_result.add_note("Note.")
        assert returned is basic_result

    def test_notes_not_rendered_in_to_html(self, basic_result: AnalysisResult) -> None:
        """Notes are stored but not part of to_html output (no dedicated section)."""
        basic_result.add_note("Hidden note.")
        html = basic_result.to_html()
        # Notes appear only in summary(), not in to_html()
        assert "Hidden note." not in html

    def test_summary_contains_title(self, basic_result: AnalysisResult) -> None:
        s = basic_result.summary()
        assert "Test Analysis" in s

    def test_summary_contains_counts(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_table(basic_table)
        basic_result.add_warning("W")
        s = basic_result.summary()
        assert "Tables: 1" in s
        assert "Warnings: 1" in s

    def test_summary_lists_warnings(self, basic_result: AnalysisResult) -> None:
        basic_result.add_warning("Alert!")
        s = basic_result.summary()
        assert "Alert!" in s

    def test_add_assumption_appends(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_assumption(basic_table)
        assert len(basic_result.assumptions) == 1

    def test_add_assumption_rendered_in_to_html(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_assumption(basic_table)
        html = basic_result.to_html()
        assert "Test Table" in html

    def test_add_diagnostic_appends(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        basic_result.add_diagnostic(basic_table)
        assert len(basic_result.diagnostics) == 1

    def test_chaining_multiple_adds(self, basic_result: AnalysisResult, basic_table: ResultTable) -> None:
        (
            basic_result
            .add_table(basic_table)
            .add_warning("W1")
            .add_note("N1")
        )
        assert len(basic_result.tables) == 1
        assert len(basic_result.warnings) == 1
        assert len(basic_result.notes) == 1

    def test_created_at_is_datetime(self, basic_result: AnalysisResult) -> None:
        assert isinstance(basic_result.created_at, datetime)

    def test_default_lists_empty(self) -> None:
        r = AnalysisResult(id="empty", title="Empty")
        assert r.tables == []
        assert r.warnings == []
        assert r.notes == []
        assert r.assumptions == []
        assert r.diagnostics == []
        assert r.figures == []
