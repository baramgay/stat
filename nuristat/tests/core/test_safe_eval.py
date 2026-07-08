"""Tests for validate_expression — dunder-attribute sandbox-escape guard."""

from __future__ import annotations

import pytest

from nuristat.core.exceptions import UnsafeExpressionError
from nuristat.core.safe_eval import validate_expression


class TestValidateExpression:
    """Tests for validate_expression."""

    @pytest.mark.parametrize(
        "expr",
        [
            "Age > 18",
            "(Age - mean(Age)) / std(Age)",
            "Gender == 'Male'",
            "score > 30 & flag == 1",
            "log(Income)",
        ],
    )
    def test_allows_normal_expressions(self, expr: str) -> None:
        validate_expression(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "().__class__.__base__.__subclasses__()",
            "x.__class__.__bases__",
            "().__class__.__mro__[1].__subclasses__()",
        ],
    )
    def test_rejects_dunder_attribute_access(self, expr: str) -> None:
        with pytest.raises(UnsafeExpressionError):
            validate_expression(expr)

    def test_rejects_import(self) -> None:
        with pytest.raises(UnsafeExpressionError):
            validate_expression("__import__('os').system('echo hi')")

    def test_rejects_syntax_error(self) -> None:
        with pytest.raises(UnsafeExpressionError):
            validate_expression("Age >")
