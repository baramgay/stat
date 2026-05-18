"""Formatting utilities for statistical output.

Implements the p-value display rules from HERMES.md §12.2 and §16.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# p-value formatting
# ---------------------------------------------------------------------------

def format_pvalue(p: Union[float, None]) -> str:
    """Format a p-value according to HERMES.md rules.

    Rules
    -----
    * ``p < 0.001``   → ``"< .001"``
    * ``0.001 <= p < 1`` → 3 decimal places (e.g. ``".042"``)
    * ``p >= 1``      → ``"1.000"``
    * ``NaN`` / ``None`` → ``""`` (empty string)

    Parameters
    ----------
    p : float or None
        Raw p-value.

    Returns
    -------
    str
        Formatted p-value string.
    """
    if p is None:
        return ""
    if isinstance(p, float) and (math.isnan(p) or np.isnan(p)):
        return ""
    p = float(p)
    if p < 0.0:
        # Guard against numerical underflow reported as negative
        p = 0.0
    if p < 0.001:
        return "< .001"
    if p >= 1.0:
        return "1.000"
    # 0.001 <= p < 1 → 3 decimal places, drop leading zero
    formatted = f"{p:.3f}"
    if formatted.startswith("0."):
        return formatted[1:]  # ".042"
    return formatted


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def format_number(x: Union[float, None], decimals: int = 3) -> str:
    """Format a number with *decimals* decimal places.

    Parameters
    ----------
    x : float or None
        Number to format.
    decimals : int, default 3
        Number of decimal places.

    Returns
    -------
    str
        Formatted number, or empty string if *x* is None/NaN.
    """
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or np.isnan(x)):
        return ""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(x)


def format_ci(
    lower: Union[float, None],
    upper: Union[float, None],
    decimals: int = 3,
    level: Optional[float] = None,
) -> str:
    """Format a confidence interval as ``[lower, upper]``.

    Parameters
    ----------
    lower, upper : float or None
        CI bounds.
    decimals : int, default 3
        Decimal places for the bounds.
    level : float or None, optional
        Confidence level (e.g. 0.95) for display; not used in formatting.

    Returns
    -------
    str
        Formatted CI string, or ``""`` if either bound is missing/NaN.
    """
    if lower is None or upper is None:
        return ""
    if (isinstance(lower, float) and math.isnan(lower)) or \
       (isinstance(upper, float) and math.isnan(upper)):
        return ""
    lo = format_number(lower, decimals)
    hi = format_number(upper, decimals)
    return f"[{lo}, {hi}]"


def format_percent(x: Union[float, None], decimals: int = 1) -> str:
    """Format a percentage value.

    Parameters
    ----------
    x : float or None
        Percentage value (e.g. 95.5 for 95.5%).
    decimals : int, default 1
        Number of decimal places.

    Returns
    -------
    str
        Formatted percentage string with ``%`` suffix.
    """
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or np.isnan(x)):
        return ""
    return f"{x:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Significance stars
# ---------------------------------------------------------------------------

def add_significance_stars(p: Union[float, None]) -> str:
    """Return significance stars for a p-value.

    * ``p < .001`` → ``"***"``
    * ``p < .01``  → ``"**"``
    * ``p < .05``  → ``"*"``
    * otherwise    → ``""``

    Parameters
    ----------
    p : float or None
        p-value.

    Returns
    -------
    str
        Star string.
    """
    if p is None:
        return ""
    if isinstance(p, float) and (math.isnan(p) or np.isnan(p)):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_pvalue_with_stars(p: Union[float, None]) -> str:
    """Format p-value with appended significance stars.

    Example: ``".042 *"`` for p = 0.042.
    """
    pv = format_pvalue(p)
    stars = add_significance_stars(p)
    if stars:
        return f"{pv} {stars}"
    return pv


# ---------------------------------------------------------------------------
# Convenience: batch-format a DataFrame column
# ---------------------------------------------------------------------------

def format_column(
    values: list[Union[float, None]],
    style: str = "number",
    decimals: int = 3,
) -> list[str]:
    """Format a list of numeric values according to *style*.

    Parameters
    ----------
    values : list[float | None]
        Values to format.
    style : {"number", "pvalue", "percent", "ci"}
        Formatting style.
    decimals : int, default 3
        Decimal places (used by ``"number"`` and ``"percent"``).

    Returns
    -------
    list[str]
        Formatted strings.
    """
    formatters = {
        "number": lambda x: format_number(x, decimals),
        "pvalue": format_pvalue,
        "percent": lambda x: format_percent(x, decimals),
    }
    fmt = formatters.get(style, formatters["number"])
    return [fmt(v) for v in values]
