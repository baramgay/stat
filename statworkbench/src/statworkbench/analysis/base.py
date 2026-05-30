"""AnalysisPlugin Protocol — base interface for all statistical analyses."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset


@runtime_checkable
class AnalysisPlugin(Protocol):
    """Protocol that every analysis plugin must satisfy.

    Attributes
    ----------
    id : str
        Unique identifier for the plugin (e.g. ``"independent_t_test"``).
    name : str
        Human-readable name (e.g. ``"Independent-Samples T Test"``).
    category : str
        Menu category (e.g. ``"Compare Means"``).
    description : str
        Short description of what the analysis does.
    variable_requirements : list[dict]
        List of dicts describing required / optional variables.
        Each dict has keys like ``role``, ``measure_types``, ``min_count``,
        ``max_count``, ``required``.

    Methods
    -------
    validate(dataset, spec) -> list[str]
        Validate whether the chosen variables and options are appropriate.
        Returns a list of warning / error messages (empty = OK).
    run(dataset, spec) -> AnalysisResult
        Execute the analysis and return a structured result.
    """

    id: str
    name: str
    category: str
    description: str
    variable_requirements: list[dict]

    implemented: bool = True

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        """Validate the analysis request.

        Parameters
        ----------
        dataset : Dataset
            The dataset to analyse.
        spec : dict
            Analysis specification (variables, options, etc.).

        Returns
        -------
        list[str]
            Validation messages — empty list means the request is valid.
        """
        ...  # pragma: no cover

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        """Run the analysis.

        Parameters
        ----------
        dataset : Dataset
            The dataset to analyse.
        spec : dict
            Analysis specification.

        Returns
        -------
        AnalysisResult
            Structured analysis result.
        """
        ...  # pragma: no cover
