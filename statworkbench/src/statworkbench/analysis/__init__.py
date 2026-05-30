"""Analysis engine for StatWorkbench.

Provides the statistical analysis framework including plugin architecture,
result formatting, assumption checking, and implementations of standard
statistical procedures.
"""

from statworkbench.analysis.assumptions import (
    check_homogeneity_of_variance,
    check_normality,
    get_case_processing_summary,
    prepare_analysis_frame,
)
from statworkbench.analysis.base import AnalysisPlugin
from statworkbench.analysis.formatting import (
    add_significance_stars,
    format_ci,
    format_number,
    format_percent,
    format_pvalue,
)
from statworkbench.analysis.registry import AnalysisRegistry
from statworkbench.analysis.result import AnalysisResult, ResultTable

__all__ = [
    "AnalysisPlugin",
    "AnalysisRegistry",
    "AnalysisResult",
    "ResultTable",
    "format_pvalue",
    "format_number",
    "format_ci",
    "format_percent",
    "add_significance_stars",
    "prepare_analysis_frame",
    "check_normality",
    "check_homogeneity_of_variance",
    "get_case_processing_summary",
]
