"""Tests for parse_common_spec helper."""

import pytest

from nuristat.analysis.spec_utils import CommonSpec, parse_common_spec
from nuristat.core.typing import MissingPolicy


def test_defaults():
    v, o, cl, mp = parse_common_spec({})
    assert v == {}
    assert o == {}
    assert cl == 0.95
    assert mp == MissingPolicy.LISTWISE


def test_str_to_enum_coercion():
    _, _, _, mp = parse_common_spec({"missing_policy": "pairwise"})
    assert mp == MissingPolicy.PAIRWISE

    _, _, _, mp2 = parse_common_spec({"missing_policy": "listwise"})
    assert mp2 == MissingPolicy.LISTWISE


def test_enum_passthrough():
    _, _, _, mp = parse_common_spec({"missing_policy": MissingPolicy.INCLUDE_AS_CATEGORY})
    assert mp == MissingPolicy.INCLUDE_AS_CATEGORY


def test_values_propagated():
    spec = {
        "variables": {"target": ["x", "y"]},
        "options": {"alpha": 0.01},
        "confidence_level": 0.99,
    }
    v, o, cl, _ = parse_common_spec(spec)
    assert v == {"target": ["x", "y"]}
    assert o == {"alpha": 0.01}
    assert cl == 0.99


def test_returns_namedtuple():
    result = parse_common_spec({})
    assert isinstance(result, CommonSpec)
    assert result.variables == {}
