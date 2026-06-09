"""Analysis engine for NuriStat.

Provides the statistical analysis framework including plugin architecture,
result formatting, assumption checking, and implementations of standard
statistical procedures.
"""

from nuristat.analysis.assumptions import (
    check_homogeneity_of_variance,
    check_normality,
    get_case_processing_summary,
    prepare_analysis_frame,
)
from nuristat.analysis.base import AnalysisPlugin
from nuristat.analysis.formatting import (
    add_significance_stars,
    format_ci,
    format_number,
    format_percent,
    format_pvalue,
)
from nuristat.analysis.registry import AnalysisRegistry
from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.analysis.spec_utils import CommonSpec, parse_common_spec

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
    "CommonSpec",
    "parse_common_spec",
]
