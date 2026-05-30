"""Tests for AnalysisRegistry."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.base import AnalysisPlugin
from statworkbench.analysis.registry import AnalysisRegistry, _PlannedAnalysis
from statworkbench.analysis.result import AnalysisResult


# ---------------------------------------------------------------------------
# Mock plugin for testing
# ---------------------------------------------------------------------------

class MockDescriptivePlugin:
    """A concrete (non-Protocol) mock plugin for testing."""

    def __init__(self) -> None:
        self.id = "descriptives"
        self.name = "Descriptives"
        self.category = "Descriptive Statistics"
        self.description = "Compute descriptive statistics."
        self.variable_requirements = [
            {"role": "variables", "measure_types": ["scale", "ordinal"], "min_count": 1, "required": True},
        ]
        self.implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        """Mock validation."""
        var_names = spec.get("variables", [])
        if not var_names:
            return ["At least one variable is required."]
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        """Mock run — returns a minimal result."""
        from datetime import datetime
        return AnalysisResult(
            id="mock-desc-001",
            title="Descriptive Statistics",
            created_at=datetime.now(),
        )


class MockTTestPlugin:
    """Mock plugin for independent t-test."""

    def __init__(self) -> None:
        self.id = "independent_t_test"
        self.name = "Independent-Samples T Test"
        self.category = "Compare Means"
        self.description = "Compare two independent group means."
        self.variable_requirements = [
            {"role": "dependent", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
            {"role": "group", "measure_types": ["binary"], "min_count": 1, "max_count": 1, "required": True},
        ]
        self.implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        from datetime import datetime
        return AnalysisResult(
            id="mock-ttest-001",
            title="Independent-Samples T Test",
            created_at=datetime.now(),
        )


class MockCorrelationPlugin:
    """Mock plugin for Pearson correlation."""

    def __init__(self) -> None:
        self.id = "pearson_correlation"
        self.name = "Bivariate (Pearson)"
        self.category = "Correlate"
        self.description = "Pearson correlation matrix."
        self.variable_requirements = [
            {"role": "variables", "measure_types": ["scale"], "min_count": 2, "required": True},
        ]
        self.implemented = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        return []

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        from datetime import datetime
        return AnalysisResult(
            id="mock-corr-001",
            title="Pearson Correlation",
            created_at=datetime.now(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> AnalysisRegistry:
    """Return a fresh registry with built-in entries."""
    return AnalysisRegistry()


@pytest.fixture
def scale_dataset() -> Dataset:
    """Return a dataset with scale variables."""
    df = pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0, 45.0],
        "bp": [120.0, 130.0, 125.0, 140.0, 135.0],
        "score": [80.0, 85.0, 90.0, 88.0, 92.0],
    })
    ds = Dataset(df, name="ScaleData")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def binary_dataset() -> Dataset:
    """Return a dataset with binary group and scale dependent."""
    df = pd.DataFrame({
        "treatment": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
        "outcome": [78.0, 82.0, 75.0, 88.0, 91.0, 85.0, 80.0, 89.0, 77.0, 92.0],
    })
    ds = Dataset(df, name="BinaryGroupData")
    ds.variables["treatment"].measure = MeasureType.BINARY
    ds.variables["outcome"].measure = MeasureType.SCALE
    return ds


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """Tests for plugin registration."""

    def test_register_new_plugin(self, registry: AnalysisRegistry) -> None:
        """Registering a new plugin should succeed."""
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        assert "descriptives" in registry

    def test_register_duplicate_raises(self, registry: AnalysisRegistry) -> None:
        """Registering a duplicate plugin should raise ValueError."""
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(plugin)

    def test_register_no_id_raises(self, registry: AnalysisRegistry) -> None:
        """Registering a plugin without an id should raise ValueError."""
        class BadPlugin:
            pass

        with pytest.raises(ValueError, match="id"):
            registry.register(BadPlugin())  # type: ignore[arg-type]

    def test_unregister_existing(self, registry: AnalysisRegistry) -> None:
        """Unregistering an existing plugin should succeed."""
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        registry.unregister("descriptives")
        assert "descriptives" not in registry

    def test_unregister_nonexistent(self, registry: AnalysisRegistry) -> None:
        """Unregistering a nonexistent plugin should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Tests for plugin retrieval."""

    def test_get_existing(self, registry: AnalysisRegistry) -> None:
        """Getting an existing plugin should succeed."""
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        retrieved = registry.get("descriptives")
        assert retrieved.name == "Descriptives"

    def test_get_nonexistent(self, registry: AnalysisRegistry) -> None:
        """Getting a nonexistent plugin should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_all(self, registry: AnalysisRegistry) -> None:
        """list_all should return both implemented and planned plugins."""
        all_plugins = registry.list_all()
        assert len(all_plugins) > 0
        # Should include planned analyses from built-in list
        assert len(all_plugins) >= len(registry.list_implemented())

    def test_list_by_category(self, registry: AnalysisRegistry) -> None:
        """list_by_category should filter by category."""
        desc_plugins = registry.list_by_category("Descriptive Statistics")
        assert len(desc_plugins) >= 0
        # All returned should have the correct category
        for p in desc_plugins:
            assert p.category == "Descriptive Statistics"

    def test_list_implemented(self, registry: AnalysisRegistry) -> None:
        """list_implemented should only return implemented plugins."""
        impl = registry.list_implemented()
        for p in impl:
            assert getattr(p, "implemented", True) is True

    def test_list_planned(self, registry: AnalysisRegistry) -> None:
        """list_planned should only return non-implemented plugins."""
        planned = registry.list_planned()
        for p in planned:
            assert getattr(p, "implemented", True) is False

    def test_categories(self, registry: AnalysisRegistry) -> None:
        """categories should return a list of unique categories."""
        cats = registry.categories()
        assert isinstance(cats, list)
        assert len(cats) > 0
        # Check for expected categories (planned stubs + registered plugins)
        expected = ["Descriptive Statistics", "Nonparametric Tests"]
        for exp in expected:
            assert exp in cats

    def test_len(self, registry: AnalysisRegistry) -> None:
        """Registry should support len()."""
        n = len(registry)
        assert n > 0  # Built-in planned analyses
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        assert len(registry) == n + 1


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    """Tests for recommend_for_variables."""

    def test_recommend_descriptives(
        self, registry: AnalysisRegistry, scale_dataset: Dataset
    ) -> None:
        """Scale variables should match descriptives."""
        plugin = MockDescriptivePlugin()
        registry.register(plugin)
        recs = registry.recommend_for_variables(
            scale_dataset, ["age", "bp", "score"]
        )
        ids = [p.id for p in recs]
        assert "descriptives" in ids

    def test_recommend_ttest(
        self, registry: AnalysisRegistry, binary_dataset: Dataset
    ) -> None:
        """Binary group + scale dependent should match t-test."""
        # independent_t_test is now properly registered via _register_new_plugins
        recs = registry.recommend_for_variables(
            binary_dataset, ["treatment", "outcome"]
        )
        ids = [p.id for p in recs]
        assert "independent_t_test" in ids

    def test_recommend_correlation(
        self, registry: AnalysisRegistry, scale_dataset: Dataset
    ) -> None:
        """Two+ scale variables should match Pearson correlation."""
        plugin = MockCorrelationPlugin()
        registry.register(plugin)
        recs = registry.recommend_for_variables(
            scale_dataset, ["age", "bp"]
        )
        ids = [p.id for p in recs]
        assert "pearson_correlation" in ids

    def test_no_match_no_variables(
        self, registry: AnalysisRegistry, scale_dataset: Dataset
    ) -> None:
        """Empty variable list should return empty recommendations."""
        recs = registry.recommend_for_variables(scale_dataset, [])
        assert recs == []

    def test_no_match_incompatible(
        self, registry: AnalysisRegistry, scale_dataset: Dataset
    ) -> None:
        """Variables that don't match any requirement should return empty."""
        # Only scale variables, no binary group → t-test shouldn't match
        # independent_t_test is already registered via _register_new_plugins
        recs = registry.recommend_for_variables(
            scale_dataset, ["age", "bp"]
        )
        ids = [p.id for p in recs]
        assert "independent_t_test" not in ids


# ---------------------------------------------------------------------------
# Planned analysis wrapper
# ---------------------------------------------------------------------------

class TestPlannedAnalysis:
    """Tests for the _PlannedAnalysis wrapper."""

    def test_planned_has_required_attributes(self) -> None:
        pa = _PlannedAnalysis(
            id="explore",
            name="Explore",
            category="Descriptive Statistics",
            description="Detailed examination.",
        )
        assert pa.id == "explore"
        assert pa.name == "Explore"
        assert pa.category == "Descriptive Statistics"
        assert pa.description == "Detailed examination."
        assert pa.implemented is False

    def test_planned_validate_raises(self, scale_dataset: Dataset) -> None:
        pa = _PlannedAnalysis(
            id="test",
            name="Test",
            category="Cat",
            description="Desc",
        )
        with pytest.raises(NotImplementedError, match="planned"):
            pa.validate(scale_dataset, {})

    def test_planned_run_raises(self, scale_dataset: Dataset) -> None:
        pa = _PlannedAnalysis(
            id="test",
            name="Test",
            category="Cat",
            description="Desc",
        )
        with pytest.raises(NotImplementedError, match="planned"):
            pa.run(scale_dataset, {})

    def test_planned_in_list_planned(self, registry: AnalysisRegistry) -> None:
        """Planned analyses should appear in list_planned."""
        planned = registry.list_planned()
        ids = [p.id for p in planned]
        # kaplan_meier and cox_regression are truly unimplemented
        assert "kaplan_meier" in ids
        assert "cox_regression" in ids
        # explore, logistic_regression, sensitivity_specificity are now implemented
        assert "logistic_regression" not in ids
        assert "explore" not in ids
        assert "sensitivity_specificity" not in ids

    def test_planned_not_in_list_implemented(
        self, registry: AnalysisRegistry
    ) -> None:
        """Planned analyses should NOT appear in list_implemented."""
        impl = registry.list_implemented()
        ids = [p.id for p in impl]
        # kaplan_meier and cox_regression are truly unimplemented → not in implemented
        assert "kaplan_meier" not in ids
        assert "cox_regression" not in ids
        # explore, logistic_regression, sensitivity_specificity ARE now implemented
        assert "explore" in ids
        assert "logistic_regression" in ids
        assert "sensitivity_specificity" in ids
