"""Analysis engine for StatWorkbench.

Provides the statistical analysis framework including plugin architecture,
result formatting, assumption checking, and implementations of standard
statistical procedures.
"""

from statworkbench.analysis.base import AnalysisPlugin
from statworkbench.analysis.registry import AnalysisRegistry
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import (
    format_pvalue,
    format_number,
    format_ci,
    format_percent,
    add_significance_stars,
)
from statworkbench.analysis.assumptions import (
    prepare_analysis_frame,
    check_normality,
    check_homogeneity_of_variance,
    get_case_processing_summary,
)

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
