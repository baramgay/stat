"""Tests for formatting utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from statworkbench.analysis.formatting import (
    add_significance_stars,
    format_ci,
    format_column,
    format_number,
    format_percent,
    format_pvalue,
    format_pvalue_with_stars,
)


class TestFormatPvalue:
    """Tests for format_pvalue — HERMES.md §12.2."""

    def test_very_small_p(self) -> None:
        """p < 0.001 → '< .001'."""
        assert format_pvalue(0.0005) == "< .001"
        assert format_pvalue(0.0001) == "< .001"
        assert format_pvalue(0.0) == "< .001"

    def test_boundary_p_0_001(self) -> None:
        """p exactly 0.001 → '.001' (not < .001)."""
        assert format_pvalue(0.001) == ".001"

    def test_small_p(self) -> None:
        """0.001 < p < 0.05 → decimal without leading zero."""
        assert format_pvalue(0.042) == ".042"
        assert format_pvalue(0.005) == ".005"

    def test_medium_p(self) -> None:
        """Mid-range p values → 3 decimal places."""
        assert format_pvalue(0.123) == ".123"
        assert format_pvalue(0.500) == ".500"
        assert format_pvalue(0.999) == ".999"

    def test_p_exactly_1(self) -> None:
        """p = 1.0 → '1.000'."""
        assert format_pvalue(1.0) == "1.000"

    def test_p_greater_than_1(self) -> None:
        """p > 1 → '1.000'."""
        assert format_pvalue(1.5) == "1.000"
        assert format_pvalue(2.0) == "1.000"

    def test_nan_p(self) -> None:
        """NaN p-value → empty string."""
        assert format_pvalue(float("nan")) == ""
        assert format_pvalue(np.nan) == ""

    def test_none_p(self) -> None:
        """None p-value → empty string."""
        assert format_pvalue(None) == ""

    def test_numpy_scalar_p(self) -> None:
        """numpy scalar p-values should work."""
        assert format_pvalue(np.float64(0.0005)) == "< .001"
        assert format_pvalue(np.float64(0.042)) == ".042"

    def test_negative_p(self) -> None:
        """Negative p (numerical underflow) → treat as 0 → '< .001'."""
        assert format_pvalue(-1e-15) == "< .001"


class TestFormatNumber:
    """Tests for format_number."""

    def test_basic(self) -> None:
        assert format_number(3.14159, 3) == "3.142"
        assert format_number(3.14159, 2) == "3.14"
        assert format_number(3.0, 0) == "3"

    def test_none(self) -> None:
        assert format_number(None, 3) == ""

    def test_nan(self) -> None:
        assert format_number(float("nan"), 3) == ""

    def test_integer(self) -> None:
        assert format_number(42, 3) == "42.000"

    def test_zero(self) -> None:
        assert format_number(0.0, 3) == "0.000"


class TestFormatCI:
    """Tests for format_ci."""

    def test_basic(self) -> None:
        assert format_ci(1.234, 5.678, 3) == "[1.234, 5.678]"

    def test_none_bounds(self) -> None:
        assert format_ci(None, 5.0, 3) == ""
        assert format_ci(1.0, None, 3) == ""

    def test_nan_bounds(self) -> None:
        assert format_ci(float("nan"), 5.0, 3) == ""

    def test_two_decimals(self) -> None:
        assert format_ci(1.2, 5.7, 2) == "[1.20, 5.70]"


class TestFormatPercent:
    """Tests for format_percent."""

    def test_basic(self) -> None:
        assert format_percent(95.5, 1) == "95.5%"
        assert format_percent(100.0, 1) == "100.0%"

    def test_zero(self) -> None:
        assert format_percent(0.0, 1) == "0.0%"

    def test_none(self) -> None:
        assert format_percent(None, 1) == ""

    def test_nan(self) -> None:
        assert format_percent(float("nan"), 1) == ""


class TestSignificanceStars:
    """Tests for add_significance_stars."""

    def test_p_less_than_0_001(self) -> None:
        assert add_significance_stars(0.0005) == "***"

    def test_p_less_than_0_01(self) -> None:
        assert add_significance_stars(0.005) == "**"

    def test_p_less_than_0_05(self) -> None:
        assert add_significance_stars(0.04) == "*"

    def test_p_not_significant(self) -> None:
        assert add_significance_stars(0.10) == ""
        assert add_significance_stars(0.05) == ""

    def test_none(self) -> None:
        assert add_significance_stars(None) == ""

    def test_nan(self) -> None:
        assert add_significance_stars(float("nan")) == ""


class TestFormatPvalueWithStars:
    """Tests for format_pvalue_with_stars."""

    def test_significant(self) -> None:
        assert format_pvalue_with_stars(0.042) == ".042 *"

    def test_highly_significant(self) -> None:
        assert format_pvalue_with_stars(0.0005) == "< .001 ***"

    def test_not_significant(self) -> None:
        assert format_pvalue_with_stars(0.10) == ".100"


class TestFormatColumn:
    """Tests for format_column."""

    def test_number_style(self) -> None:
        result = format_column([1.5, 2.5, None], style="number", decimals=2)
        assert result == ["1.50", "2.50", ""]

    def test_pvalue_style(self) -> None:
        result = format_column([0.042, 0.0005, None], style="pvalue")
        assert result == [".042", "< .001", ""]

    def test_percent_style(self) -> None:
        result = format_column([95.5, 100.0, None], style="percent", decimals=1)
        assert result == ["95.5%", "100.0%", ""]
