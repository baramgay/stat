"""AnalysisRegistry — plugin registry for statistical analyses."""

from __future__ import annotations

from typing import Optional

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.base import AnalysisPlugin


# ---------------------------------------------------------------------------
# Default variable-requirement templates
# ---------------------------------------------------------------------------

_VARIABLE_REQ_TEMPLATES: dict[str, list[dict]] = {
    "frequencies": [
        {"role": "variables", "measure_types": ["nominal", "ordinal", "binary", "text"], "min_count": 1, "required": True},
    ],
    "descriptives": [
        {"role": "variables", "measure_types": ["scale", "ordinal"], "min_count": 1, "required": True},
    ],
    "normality": [
        {"role": "variables", "measure_types": ["scale"], "min_count": 1, "required": True},
        {"role": "group", "measure_types": ["nominal", "ordinal", "binary"], "min_count": 0, "max_count": 1, "required": False},
    ],
    "crosstab": [
        {"role": "row", "measure_types": ["nominal", "ordinal", "binary"], "min_count": 1, "required": True},
        {"role": "column", "measure_types": ["nominal", "ordinal", "binary"], "min_count": 1, "required": True},
    ],
    "independent_t_test": [
        {"role": "dependent", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "group", "measure_types": ["binary"], "min_count": 1, "max_count": 1, "required": True},
    ],
    "paired_t_test": [
        {"role": "paired", "measure_types": ["scale"], "min_count": 2, "required": True},
    ],
    "one_way_anova": [
        {"role": "dependent", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "factor", "measure_types": ["nominal", "ordinal"], "min_count": 1, "max_count": 1, "required": True},
    ],
    "mann_whitney": [
        {"role": "dependent", "measure_types": ["ordinal", "scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "group", "measure_types": ["binary"], "min_count": 1, "max_count": 1, "required": True},
    ],
    "wilcoxon": [
        {"role": "paired", "measure_types": ["ordinal", "scale"], "min_count": 2, "required": True},
    ],
    "kruskal_wallis": [
        {"role": "dependent", "measure_types": ["ordinal", "scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "factor", "measure_types": ["nominal", "ordinal"], "min_count": 1, "max_count": 1, "required": True},
    ],
    "friedman": [
        {"role": "variables", "measure_types": ["ordinal", "scale"], "min_count": 3, "required": True},
    ],
    "pearson_correlation": [
        {"role": "variables", "measure_types": ["scale"], "min_count": 2, "required": True},
    ],
    "spearman_correlation": [
        {"role": "variables", "measure_types": ["ordinal", "scale"], "min_count": 2, "required": True},
    ],
    "linear_regression": [
        {"role": "dependent", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "predictors", "measure_types": ["scale", "nominal", "ordinal", "binary"], "min_count": 1, "required": True},
    ],
}


# ---------------------------------------------------------------------------
# Built-in analysis descriptors (used for planned / not-yet-implemented items)
# ---------------------------------------------------------------------------

_BUILTIN_ANALYSES: list[dict] = [
    {"id": "frequencies", "name": "Frequencies", "category": "Descriptive Statistics", "description": "Frequency tables for categorical variables.", "implemented": True},
    {"id": "descriptives", "name": "Descriptives", "category": "Descriptive Statistics", "description": "Descriptive statistics for scale variables.", "implemented": True},
    {"id": "explore", "name": "Explore", "category": "Descriptive Statistics", "description": "Detailed examination of variables.", "implemented": False},
    {"id": "crosstab", "name": "Crosstabs", "category": "Descriptive Statistics", "description": "Cross-tabulation and chi-square test.", "implemented": True},
    {"id": "one_sample_t_test", "name": "One-Sample T Test", "category": "Compare Means", "description": "Compare a sample mean to a known value.", "implemented": False},
    {"id": "independent_t_test", "name": "Independent-Samples T Test", "category": "Compare Means", "description": "Compare means of two independent groups.", "implemented": True},
    {"id": "paired_t_test", "name": "Paired-Samples T Test", "category": "Compare Means", "description": "Compare means of paired samples.", "implemented": True},
    {"id": "one_way_anova", "name": "One-Way ANOVA", "category": "Compare Means", "description": "Compare means across multiple groups.", "implemented": True},
    {"id": "mann_whitney", "name": "Mann-Whitney U", "category": "Nonparametric Tests", "description": "Non-parametric test for two independent groups.", "implemented": True},
    {"id": "wilcoxon", "name": "Wilcoxon Signed-Rank", "category": "Nonparametric Tests", "description": "Non-parametric test for paired samples.", "implemented": True},
    {"id": "kruskal_wallis", "name": "Kruskal-Wallis", "category": "Nonparametric Tests", "description": "Non-parametric alternative to one-way ANOVA.", "implemented": True},
    {"id": "friedman", "name": "Friedman", "category": "Nonparametric Tests", "description": "Non-parametric test for repeated measures.", "implemented": True},
    {"id": "chi_square_gof", "name": "Chi-Square Goodness-of-Fit", "category": "Nonparametric Tests", "description": "Test whether observed frequencies match expected.", "implemented": False},
    {"id": "pearson_correlation", "name": "Bivariate (Pearson)", "category": "Correlate", "description": "Pearson correlation matrix.", "implemented": True},
    {"id": "spearman_correlation", "name": "Bivariate (Spearman)", "category": "Correlate", "description": "Spearman rank correlation matrix.", "implemented": True},
    {"id": "partial_correlation", "name": "Partial", "category": "Correlate", "description": "Partial correlation.", "implemented": False},
    {"id": "linear_regression", "name": "Linear", "category": "Regression", "description": "Linear regression analysis.", "implemented": True},
    {"id": "logistic_regression", "name": "Logistic", "category": "Regression", "description": "Logistic regression analysis.", "implemented": False},
    {"id": "reliability", "name": "Reliability Analysis", "category": "Scale", "description": "Cronbach's alpha and reliability statistics.", "implemented": False},
    {"id": "roc_analysis", "name": "ROC Analysis", "category": "Diagnostic Tests", "description": "Receiver operating characteristic analysis.", "implemented": False},
    {"id": "sensitivity_specificity", "name": "Sensitivity/Specificity", "category": "Diagnostic Tests", "description": "Diagnostic accuracy measures.", "implemented": False},
    {"id": "kaplan_meier", "name": "Kaplan-Meier", "category": "Survival", "description": "Survival analysis with Kaplan-Meier estimator.", "implemented": False},
    {"id": "cox_regression", "name": "Cox Regression", "category": "Survival", "description": "Cox proportional hazards regression.", "implemented": False},
    {"id": "cohens_kappa", "name": "Cohen's Kappa", "category": "Agreement", "description": "Measure of inter-rater agreement.", "implemented": False},
    {"id": "icc", "name": "ICC", "category": "Agreement", "description": "Intraclass correlation coefficient.", "implemented": False},
    {"id": "bland_altman", "name": "Bland-Altman", "category": "Agreement", "description": "Bland-Altman agreement analysis.", "implemented": False},
    {"id": "normality", "name": "Normality Tests", "category": "Descriptive Statistics", "description": "Shapiro-Wilk normality test.", "implemented": True},
]


# ---------------------------------------------------------------------------
# Lightweight wrapper for planned (not-yet-implemented) analyses
# ---------------------------------------------------------------------------

class _PlannedAnalysis:
    """Lightweight stand-in for an analysis that is planned but not yet implemented.

    Satisfies the ``AnalysisPlugin`` Protocol enough for registry queries.
    Calling ``run`` or ``validate`` raises ``NotImplementedError``.
    """

    def __init__(self, id: str, name: str, category: str, description: str) -> None:  # noqa: A002
        self.id = id  # noqa: A002
        self.name = name
        self.category = category
        self.description = description
        self.variable_requirements: list[dict] = _VARIABLE_REQ_TEMPLATES.get(id, [])
        self.implemented = False

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:  # noqa: ARG002
        """Planned analyses cannot validate."""
        raise NotImplementedError(
            f"Analysis '{self.id}' is planned but not yet implemented."
        )

    def run(self, dataset: Dataset, spec: dict) -> None:  # noqa: ARG002
        """Planned analyses cannot run."""
        raise NotImplementedError(
            f"Analysis '{self.id}' is planned but not yet implemented."
        )


# ---------------------------------------------------------------------------
# AnalysisRegistry
# ---------------------------------------------------------------------------

class AnalysisRegistry:
    """Central registry for analysis plugins.

    Maintains a mapping from plugin ``id`` to :class:`AnalysisPlugin` instances.
    Provides lookup by category, implementation status, and variable-type
    recommendation.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, AnalysisPlugin] = {}
        # Register built-in planned analyses as lightweight wrappers
        for info in _BUILTIN_ANALYSES:
            if not info["implemented"]:
                self._plugins[info["id"]] = _PlannedAnalysis(
                    id=info["id"],
                    name=info["name"],
                    category=info["category"],
                    description=info["description"],
                )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: AnalysisPlugin) -> None:
        """Register an analysis plugin.

        Parameters
        ----------
        plugin : AnalysisPlugin
            Plugin instance to register.

        Raises
        ------
        ValueError
            If a plugin with the same *id* is already registered.
        """
        pid = getattr(plugin, "id", None)
        if not pid:
            raise ValueError("Plugin must have an 'id' attribute.")
        if pid in self._plugins:
            raise ValueError(f"Plugin with id '{pid}' is already registered.")
        self._plugins[pid] = plugin

    def unregister(self, plugin_id: str) -> None:
        """Remove a plugin from the registry.

        Parameters
        ----------
        plugin_id : str
            ID of the plugin to remove.

        Raises
        ------
        KeyError
            If the plugin is not found.
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin '{plugin_id}' not found in registry.")
        del self._plugins[plugin_id]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, plugin_id: str) -> AnalysisPlugin:
        """Retrieve a plugin by its ID.

        Parameters
        ----------
        plugin_id : str
            Plugin identifier.

        Returns
        -------
        AnalysisPlugin

        Raises
        ------
        KeyError
            If the plugin is not found.
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin '{plugin_id}' not found in registry.")
        return self._plugins[plugin_id]

    def list_all(self) -> list[AnalysisPlugin]:
        """Return all registered plugins (implemented + planned), sorted by name."""
        return sorted(self._plugins.values(), key=lambda p: p.name)

    def list_by_category(self, category: str) -> list[AnalysisPlugin]:
        """Return plugins in the given menu category."""
        return sorted(
            [p for p in self._plugins.values() if p.category == category],
            key=lambda p: p.name,
        )

    def list_implemented(self) -> list[AnalysisPlugin]:
        """Return only plugins that have a real implementation."""
        return sorted(
            [p for p in self._plugins.values() if getattr(p, "implemented", True)],
            key=lambda p: p.name,
        )

    def list_planned(self) -> list[AnalysisPlugin]:
        """Return plugins that are planned but not yet implemented."""
        return sorted(
            [p for p in self._plugins.values() if not getattr(p, "implemented", True)],
            key=lambda p: p.name,
        )

    # ------------------------------------------------------------------
    # Recommendation engine
    # ------------------------------------------------------------------

    def recommend_for_variables(
        self,
        dataset: Dataset,
        var_names: list[str],
    ) -> list[AnalysisPlugin]:
        """Recommend analyses that can be applied to the selected variables.

        The matching logic inspects each plugin's ``variable_requirements``
        and checks whether the chosen variables have compatible
        :class:`MeasureType` values.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the variables.
        var_names : list[str]
            Names of the selected variables.

        Returns
        -------
        list[AnalysisPlugin]
            Compatible plugins, sorted by category then name.
        """
        if not var_names:
            return []

        # Gather measure types for selected variables
        var_measures: dict[str, Optional[str]] = {}
        for v in var_names:
            meta = dataset.variables.get(v)
            if meta is not None:
                var_measures[v] = meta.measure.value if meta.measure else None
            else:
                var_measures[v] = None

        compatible: list[AnalysisPlugin] = []
        for plugin in self._plugins.values():
            if not getattr(plugin, "implemented", True):
                continue  # Don't recommend planned analyses
            reqs = getattr(plugin, "variable_requirements", [])
            if not reqs:
                continue
            if self._match_requirements(reqs, var_names, var_measures):
                compatible.append(plugin)

        return sorted(compatible, key=lambda p: (p.category, p.name))

    @staticmethod
    def _match_requirements(
        reqs: list[dict],
        var_names: list[str],
        var_measures: dict[str, Optional[str]],
    ) -> bool:
        """Check whether selected variables satisfy plugin requirements.

        This is a simplified matcher: each requirement must be met by
        at least the minimum count of variables whose measure type is
        in the allowed list.
        """
        available = set(var_names)
        used: set[str] = set()

        for req in reqs:
            allowed_measures = set(req.get("measure_types", []))
            min_count = req.get("min_count", 1)
            max_count = req.get("max_count", None)
            required = req.get("required", True)

            # Find variables that match this requirement
            matching = [
                v for v in available
                if v not in used and var_measures.get(v) in allowed_measures
            ]

            count = len(matching)
            if count < min_count:
                if required:
                    return False
                # Not required: skip this requirement
                continue

            actual_count = count
            if max_count is not None:
                actual_count = min(count, max_count)

            # Mark variables as used
            for v in matching[:actual_count]:
                used.add(v)

        return True

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return a sorted list of all known categories."""
        cats = {p.category for p in self._plugins.values()}
        return sorted(cats)

    def __contains__(self, plugin_id: str) -> bool:
        """Allow ``plugin_id in registry`` syntax."""
        return plugin_id in self._plugins

    def __len__(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)
