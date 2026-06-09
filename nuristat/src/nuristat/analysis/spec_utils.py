"""Common spec parsing utilities for NuriStat analysis modules.

Provides a shared helper to extract the standard header fields that every
analysis module needs from the ``spec`` dict, eliminating ~4-8 lines of
boilerplate that was copy-pasted across 20+ modules.

Usage::

    from nuristat.analysis.spec_utils import parse_common_spec

    def run_analysis(dataset, spec):
        v, o, cl, mp = parse_common_spec(spec)
        target_vars = v.get("target", [])
        ...
"""

from __future__ import annotations

from typing import NamedTuple

from nuristat.core.typing import MissingPolicy


class CommonSpec(NamedTuple):
    """Parsed common spec fields shared by all analysis modules.

    Attributes
    ----------
    variables : dict
        The ``spec["variables"]`` sub-dict (default ``{}``).
    options : dict
        The ``spec["options"]`` sub-dict (default ``{}``).
    confidence_level : float
        Confidence level (default 0.95).
    missing_policy : MissingPolicy
        Missing-data policy coerced to enum (default ``LISTWISE``).
    """

    variables: dict
    options: dict
    confidence_level: float
    missing_policy: MissingPolicy


def parse_common_spec(spec: dict) -> CommonSpec:
    """Extract common spec fields, coercing types as needed.

    Parameters
    ----------
    spec : dict
        Raw analysis specification dict.

    Returns
    -------
    CommonSpec
        Named tuple with ``variables``, ``options``, ``confidence_level``,
        ``missing_policy``.

    Examples
    --------
    >>> v, o, cl, mp = parse_common_spec(spec)
    >>> target = v.get("target", [])
    """
    mp_raw = spec.get("missing_policy", MissingPolicy.LISTWISE)
    if isinstance(mp_raw, str):
        mp = MissingPolicy(mp_raw)
    else:
        mp = mp_raw

    return CommonSpec(
        variables=spec.get("variables", {}),
        options=spec.get("options", {}),
        confidence_level=spec.get("confidence_level", 0.95),
        missing_policy=mp,
    )
