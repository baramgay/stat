"""AnalysisRegistry — plugin registry for statistical analyses."""

from __future__ import annotations

from typing import Any

from statworkbench.analysis.base import AnalysisPlugin
from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset

# ---------------------------------------------------------------------------
# Default variable-requirement templates
# ---------------------------------------------------------------------------

_VARIABLE_REQ_TEMPLATES: dict[str, list[dict]] = {
    "logistic_regression": [
        {"role": "dependent", "measure_types": ["nominal", "binary", "ordinal"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "predictors", "measure_types": ["scale", "nominal", "ordinal", "binary"], "min_count": 1, "required": True},
    ],
    "factor_analysis": [
        {"role": "variables", "measure_types": ["scale", "ordinal"], "min_count": 2, "required": True},
    ],
    "cluster_analysis": [
        {"role": "variables", "measure_types": ["scale", "ordinal"], "min_count": 1, "required": True},
    ],
    "survival_analysis": [
        {"role": "duration", "measure_types": ["scale"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "event", "measure_types": ["binary", "nominal"], "min_count": 1, "max_count": 1, "required": True},
    ],
    "discriminant_analysis": [
        {"role": "dependent", "measure_types": ["nominal", "ordinal", "binary"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "predictors", "measure_types": ["scale", "ordinal"], "min_count": 1, "required": True},
    ],
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
    "sensitivity_specificity": [
        {"role": "outcome", "measure_types": ["binary", "nominal"], "min_count": 1, "max_count": 1, "required": True},
        {"role": "predictor", "measure_types": ["binary", "nominal"], "min_count": 1, "max_count": 1, "required": True},
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
    {"id": "explore", "name": "Explore", "category": "Descriptive Statistics", "description": "Detailed examination of variables.", "implemented": True},
    {"id": "crosstab", "name": "Crosstabs", "category": "Descriptive Statistics", "description": "Cross-tabulation and chi-square test.", "implemented": True},
    {"id": "one_sample_t_test", "name": "One-Sample T Test", "category": "Compare Means", "description": "Compare a sample mean to a known value.", "implemented": True},
    {"id": "independent_t_test", "name": "Independent-Samples T Test", "category": "Compare Means", "description": "Compare means of two independent groups.", "implemented": True},
    {"id": "paired_t_test", "name": "Paired-Samples T Test", "category": "Compare Means", "description": "Compare means of paired samples.", "implemented": True},
    {"id": "one_way_anova", "name": "One-Way ANOVA", "category": "Compare Means", "description": "Compare means across multiple groups.", "implemented": True},
    {"id": "mann_whitney", "name": "Mann-Whitney U", "category": "Nonparametric Tests", "description": "Non-parametric test for two independent groups.", "implemented": True},
    {"id": "wilcoxon", "name": "Wilcoxon Signed-Rank", "category": "Nonparametric Tests", "description": "Non-parametric test for paired samples.", "implemented": True},
    {"id": "kruskal_wallis", "name": "Kruskal-Wallis", "category": "Nonparametric Tests", "description": "Non-parametric alternative to one-way ANOVA.", "implemented": True},
    {"id": "friedman", "name": "Friedman", "category": "Nonparametric Tests", "description": "Non-parametric test for repeated measures.", "implemented": True},
    {"id": "chi_square_gof", "name": "Chi-Square Goodness-of-Fit", "category": "Nonparametric Tests", "description": "Test whether observed frequencies match expected.", "implemented": True},
    {"id": "pearson_correlation", "name": "Bivariate (Pearson)", "category": "Correlate", "description": "Pearson correlation matrix.", "implemented": True},
    {"id": "spearman_correlation", "name": "Bivariate (Spearman)", "category": "Correlate", "description": "Spearman rank correlation matrix.", "implemented": True},
    {"id": "partial_correlation", "name": "Partial", "category": "Correlate", "description": "Partial correlation.", "implemented": True},
    {"id": "linear_regression", "name": "Linear", "category": "Regression", "description": "Linear regression analysis.", "implemented": True},
    {"id": "logistic_regression", "name": "Logistic", "category": "Regression", "description": "Binary and multinomial logistic regression with OR, CI, Hosmer-Lemeshow, ROC AUC.", "implemented": True},
    {"id": "factor_analysis", "name": "Factor Analysis / PCA", "category": "Dimension Reduction", "description": "Exploratory Factor Analysis and PCA with Varimax rotation, KMO, Bartlett's test.", "implemented": True},
    {"id": "cluster_analysis", "name": "Cluster Analysis", "category": "Classification", "description": "K-means and hierarchical clustering with silhouette coefficient and dendrogram data.", "implemented": True},
    {"id": "survival_analysis", "name": "Survival Analysis", "category": "Survival", "description": "Kaplan-Meier estimator, log-rank test, and Cox proportional hazards regression.", "implemented": True},
    {"id": "discriminant_analysis", "name": "Discriminant Analysis", "category": "Classification", "description": "Linear Discriminant Analysis with Wilks Lambda, classification matrix, structure matrix.", "implemented": True},
    {"id": "reliability", "name": "Reliability Analysis", "category": "Scale", "description": "Cronbach's alpha and reliability statistics.", "implemented": True},
    {"id": "roc_analysis", "name": "ROC Analysis", "category": "Diagnostic Tests", "description": "Receiver operating characteristic analysis.", "implemented": True},
    {"id": "sensitivity_specificity", "name": "Sensitivity/Specificity", "category": "Diagnostic Tests", "description": "Diagnostic accuracy measures.", "implemented": True},
    {"id": "kaplan_meier", "name": "Kaplan-Meier", "category": "Survival", "description": "Kaplan-Meier 생존곡선, 로그순위 검정, 중앙생존시간, 95% CI.", "implemented": True},
    {"id": "cox_regression", "name": "Cox Regression", "category": "Survival", "description": "Cox 비례위험 회귀: HR, 95% CI, Wald 검정, 비례성 가정 진단.", "implemented": True},
    {"id": "cohens_kappa", "name": "Cohen's Kappa", "category": "Agreement", "description": "Measure of inter-rater agreement.", "implemented": True},
    {"id": "icc", "name": "ICC", "category": "Agreement", "description": "Intraclass correlation coefficient.", "implemented": True},
    {"id": "bland_altman", "name": "Bland-Altman", "category": "Agreement", "description": "Bland-Altman agreement analysis.", "implemented": True},
    {"id": "normality", "name": "Normality Tests", "category": "Descriptive Statistics", "description": "Shapiro-Wilk normality test.", "implemented": True},
    {"id": "two_way_anova", "name": "Two-Way ANOVA", "category": "General Linear Model", "description": "이원분산분석: 주 효과·상호작용, 기술통계, Levene, η², Tukey HSD", "implemented": True},
    {"id": "repeated_measures_anova", "name": "Repeated Measures ANOVA", "category": "General Linear Model", "description": "반복측정 ANOVA: Mauchly 구형성 검정, GG/HF 보정, 본페로니 쌍 비교", "implemented": True},
    {"id": "ancova", "name": "ANCOVA", "category": "General Linear Model", "description": "공분산분석: 공변량 조정 요인 효과, EMM, Bonferroni 사후 검정, η²", "implemented": True},
    {"id": "mixed_anova", "name": "Mixed ANOVA", "category": "General Linear Model", "description": "혼합 분산분석: 집단 간 × 집단 내 혼합 설계, Mauchly 구형성 검정, GG/HF 보정, 편 η², Bonferroni", "implemented": True},
    {"id": "manova", "name": "MANOVA", "category": "General Linear Model", "description": "다변량 분산분석: Pillai/Wilks/Hotelling/Roy 검정, 단변량 후속 검정, 편 η², 사후 검정", "implemented": True},
    {"id": "text_mining", "name": "Text Mining", "category": "Text Analysis", "description": "텍스트 마이닝: 단어 빈도, N-gram, TF-IDF, 워드클라우드", "implemented": True},
    {"id": "pca", "name": "PCA (주성분분석)", "category": "Dimension Reduction", "description": "주성분분석: KMO/Bartlett, 공통성, 설명 분산, Varimax/Promax 회전, 스크리 플롯", "implemented": True},
    {"id": "multinomial_logistic", "name": "Multinomial Logistic Regression", "category": "Regression", "description": "다항 로지스틱 회귀: 기준 범주 선택, Pseudo R², 모수 추정값, 분류표", "implemented": True},
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
        # Replace stubs with actual implementations for all known modules
        _register_new_plugins(self)

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
        var_measures: dict[str, str | None] = {}
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
        var_measures: dict[str, str | None],
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


# ---------------------------------------------------------------------------
# Module-based plugin adapters for new analyses
# ---------------------------------------------------------------------------

class _ModulePlugin:
    """Adapter that wraps a module's run_analysis() as an AnalysisPlugin."""

    def __init__(
        self,
        plugin_id: str,
        name: str,
        category: str,
        description: str,
        module_path: str,
        function_name: str = "run_analysis",
    ) -> None:
        self.id = plugin_id
        self.name = name
        self.category = category
        self.description = description
        self.implemented = True
        self.variable_requirements: list[dict] = _VARIABLE_REQ_TEMPLATES.get(plugin_id, [])
        self._module_path = module_path
        self._function_name = function_name
        self._module: Any = None

    def _load(self) -> None:
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self._module_path)

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        """Basic validation — returns empty list if variables are present."""
        errors: list[str] = []
        variables = spec.get("variables", {})
        if not variables:
            errors.append("분석 변수가 지정되지 않았습니다.")
        return errors

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        """Delegate to the module's designated function."""
        self._load()
        fn = getattr(self._module, self._function_name)
        return fn(dataset, spec)  # type: ignore[union-attr]


def _register_new_plugins(registry: AnalysisRegistry) -> None:
    """Register the five new analysis modules into *registry*.

    This function replaces the _PlannedAnalysis stubs that were created during
    ``__init__`` for these IDs.  Call it once after instantiating the registry.
    """
    new_plugins = [
        _ModulePlugin(
            plugin_id="logistic_regression",
            name="Logistic Regression",
            category="Regression",
            description="이항/다항 로지스틱 회귀: OR, 95% CI, Wald, Hosmer-Lemeshow, ROC AUC",
            module_path="statworkbench.analysis.logistic_regression",
        ),
        _ModulePlugin(
            plugin_id="factor_analysis",
            name="Factor Analysis / PCA",
            category="Dimension Reduction",
            description="탐색적 요인분석(EFA) 및 주성분분석(PCA): Varimax 회전, KMO, Bartlett",
            module_path="statworkbench.analysis.factor_analysis",
        ),
        _ModulePlugin(
            plugin_id="cluster_analysis",
            name="Cluster Analysis",
            category="Classification",
            description="K-평균 및 계층적 군집분석: 실루엣 계수, 덴드로그램 데이터",
            module_path="statworkbench.analysis.cluster_analysis",
        ),
        _ModulePlugin(
            plugin_id="survival_analysis",
            name="Survival Analysis",
            category="Survival",
            description="Kaplan-Meier, Log-rank 검정, Cox 비례위험 회귀",
            module_path="statworkbench.analysis.survival_analysis",
        ),
        _ModulePlugin(
            plugin_id="discriminant_analysis",
            name="Discriminant Analysis",
            category="Classification",
            description="선형 판별분석: Wilks Lambda, 분류 행렬, 구조 행렬",
            module_path="statworkbench.analysis.discriminant_analysis",
        ),
        _ModulePlugin(
            plugin_id="reliability",
            name="Reliability Analysis",
            category="Scale",
            description="Cronbach's alpha, 항목 통계, 항목-전체 상관",
            module_path="statworkbench.analysis.reliability",
        ),
        _ModulePlugin(
            plugin_id="partial_correlation",
            name="Partial Correlation",
            category="Correlate",
            description="편상관분석: 통제 변수 고려한 순수 상관계수, 역행렬법",
            module_path="statworkbench.analysis.partial_correlation",
        ),
        _ModulePlugin(
            plugin_id="roc_analysis",
            name="ROC Analysis",
            category="Diagnostic Tests",
            description="ROC 곡선, AUC, Youden J 최적 컷오프, 민감도/특이도",
            module_path="statworkbench.analysis.roc_analysis",
        ),
        _ModulePlugin(
            plugin_id="explore",
            name="Explore",
            category="Descriptive Statistics",
            description="탐색적 분석: 기술통계, Shapiro-Wilk, 백분위수, 극단값, 그룹별 분석",
            module_path="statworkbench.analysis.explore",
        ),
        _ModulePlugin(
            plugin_id="chi_square_gof",
            name="Chi-Square Goodness-of-Fit",
            category="Nonparametric Tests",
            description="카이제곱 적합도 검정: 관찰/기대 빈도 비교, 균등분포 및 비율 지정",
            module_path="statworkbench.analysis.chi_square_gof",
        ),
        _ModulePlugin(
            plugin_id="cohens_kappa",
            name="Cohen's Kappa",
            category="Agreement",
            description="Cohen's Kappa 평가자 간 일치도: Po, Pe, SE, 95% CI, Landis-Koch 등급",
            module_path="statworkbench.analysis.cohens_kappa",
        ),
        _ModulePlugin(
            plugin_id="icc",
            name="ICC",
            category="Agreement",
            description="급내 상관 계수: ICC(1,1)/ICC(2,1)/ICC(3,1), ANOVA 분해, 95% CI, Koo & Mae 해석",
            module_path="statworkbench.analysis.icc",
        ),
        _ModulePlugin(
            plugin_id="independent_t_test",
            name="Independent-Samples T Test",
            category="Compare Means",
            description="독립 표본 t-검정: Levene 검정, Welch 보정, Cohen's d",
            module_path="statworkbench.analysis.ttests",
        ),
        _ModulePlugin(
            plugin_id="paired_t_test",
            name="Paired-Samples T Test",
            category="Compare Means",
            description="대응 표본 t-검정: 차이 통계량, Cohen's d",
            module_path="statworkbench.analysis.ttests",
        ),
        _ModulePlugin(
            plugin_id="one_sample_t_test",
            name="One-Sample T Test",
            category="Compare Means",
            description="단일 표본 t-검정: 가설 평균 비교, 95% CI",
            module_path="statworkbench.analysis.ttests",
            function_name="run_one_sample_analysis",
        ),
        _ModulePlugin(
            plugin_id="bland_altman",
            name="Bland-Altman",
            category="Agreement",
            description="Bland-Altman 일치도 분석: bias, LoA, 95% CI, 비례 오차 감지",
            module_path="statworkbench.analysis.bland_altman",
        ),
        _ModulePlugin(
            plugin_id="sensitivity_specificity",
            name="Sensitivity/Specificity",
            category="Diagnostic Tests",
            description="민감도/특이도/PPV/NPV/정확도/F1/LR+/LR-/Youden J/MCC/Kappa",
            module_path="statworkbench.analysis.sensitivity_specificity",
        ),
        _ModulePlugin(
            plugin_id="two_way_anova",
            name="Two-Way ANOVA",
            category="General Linear Model",
            description="이원분산분석: 주 효과·상호작용 검정, 기술통계, Levene, η², Tukey HSD",
            module_path="statworkbench.analysis.two_way_anova",
        ),
        _ModulePlugin(
            plugin_id="repeated_measures_anova",
            name="Repeated Measures ANOVA",
            category="General Linear Model",
            description="반복측정 ANOVA: Mauchly 구형성 검정, GG/HF 보정, 본페로니 쌍 비교",
            module_path="statworkbench.analysis.repeated_measures_anova",
        ),
        _ModulePlugin(
            plugin_id="ancova",
            name="ANCOVA",
            category="General Linear Model",
            description="공분산분석: 공변량 조정 후 요인 효과 검정, 조정된 주변 평균(EMM), Bonferroni 사후 검정, η²",
            module_path="statworkbench.analysis.ancova",
        ),
        _ModulePlugin(
            plugin_id="mixed_anova",
            name="Mixed ANOVA",
            category="General Linear Model",
            description="혼합 분산분석: 집단 간 × 집단 내 혼합 설계, Mauchly 구형성 검정, GG/HF 보정, 편 η², Bonferroni",
            module_path="statworkbench.analysis.mixed_anova",
        ),
        _ModulePlugin(
            plugin_id="manova",
            name="MANOVA",
            category="General Linear Model",
            description="다변량 분산분석: Pillai/Wilks/Hotelling/Roy 검정, 단변량 후속 검정, 편 η², 사후 검정",
            module_path="statworkbench.analysis.manova",
        ),
        _ModulePlugin(
            plugin_id="text_mining",
            name="Text Mining",
            category="Text Analysis",
            description="텍스트 마이닝: 단어 빈도, N-gram, TF-IDF, 워드클라우드",
            module_path="statworkbench.analysis.text_mining",
        ),
        _ModulePlugin(
            plugin_id="pca",
            name="PCA (주성분분석)",
            category="Dimension Reduction",
            description="주성분분석: KMO/Bartlett, 공통성, 설명 분산, Varimax/Promax 회전, 스크리 플롯",
            module_path="statworkbench.analysis.pca",
        ),
        _ModulePlugin(
            plugin_id="multinomial_logistic",
            name="Multinomial Logistic Regression",
            category="Regression",
            description="다항 로지스틱 회귀: 기준 범주 선택, Pseudo R², 모수 추정값, 분류표",
            module_path="statworkbench.analysis.multinomial_logistic",
        ),
        _ModulePlugin(
            plugin_id="kaplan_meier",
            name="Kaplan-Meier",
            category="Survival",
            description="Kaplan-Meier 생존곡선, 로그순위 검정, 중앙생존시간, 95% CI",
            module_path="statworkbench.analysis.survival_analysis",
            function_name="run_kaplan_meier",
        ),
        _ModulePlugin(
            plugin_id="cox_regression",
            name="Cox Regression",
            category="Survival",
            description="Cox 비례위험 회귀: HR, 95% CI, Wald 검정, 비례성 가정 진단",
            module_path="statworkbench.analysis.survival_analysis",
            function_name="run_cox_regression",
        ),
    ]

    for plugin in new_plugins:
        # Replace existing planned stub if present
        if plugin.id in registry._plugins:
            del registry._plugins[plugin.id]
        registry._plugins[plugin.id] = plugin
