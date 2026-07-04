"""Analysis engine for NuriStat.

Provides the statistical analysis framework including plugin architecture,
result formatting, assumption checking, and implementations of standard
statistical procedures.
"""

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

# assumptions.py는 scipy.stats를 임포트하므로(무거운 임포트 체인) 지연 로딩한다.
# PEP 562 모듈 __getattr__ — nuristat.analysis.assumptions 자체를 직접 임포트하는
# 코드에는 영향 없음, 이 패키지의 재노출 심볼에만 적용.
_LAZY_ASSUMPTIONS = {
    "check_homogeneity_of_variance",
    "check_normality",
    "get_case_processing_summary",
    "prepare_analysis_frame",
}


def __getattr__(name: str):
    if name in _LAZY_ASSUMPTIONS:
        from nuristat.analysis import assumptions

        return getattr(assumptions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
