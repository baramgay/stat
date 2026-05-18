"""Validation utilities for variable metadata and data constraints.

This module provides functions to validate variable names, check
compatibility between measurement types and storage types, and
verify missing-value rule configurations.
"""

from __future__ import annotations

import re
from typing import Any

from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.core.exceptions import ValidationError

# Variable names may contain English letters, digits, underscores,
# and Korean Hangul characters. They must not start with a digit.
# Auto-generated names (like "0", "1") are prefixed with "var_".
_VARIABLE_NAME_PATTERN = re.compile(r"^(?!\d)[A-Za-z0-9_\uAC00-\uD7A3]+$")

# Mapping of measure types to compatible storage types.
# A ``None`` value means any storage type is acceptable.
_MEASURE_STORAGE_COMPAT: dict[MeasureType, set[StorageType] | None] = {
    MeasureType.NOMINAL: {
        StorageType.STRING,
        StorageType.INTEGER,
        StorageType.CATEGORICAL,
    },
    MeasureType.ORDINAL: {
        StorageType.STRING,
        StorageType.INTEGER,
        StorageType.CATEGORICAL,
    },
    MeasureType.SCALE: {
        StorageType.INTEGER,
        StorageType.FLOAT,
    },
    MeasureType.BINARY: {
        StorageType.BOOLEAN,
        StorageType.INTEGER,
        StorageType.STRING,
        StorageType.CATEGORICAL,
    },
    MeasureType.DATE_TIME: {
        StorageType.DATETIME,
        StorageType.STRING,
    },
    MeasureType.TEXT: {
        StorageType.STRING,
        StorageType.CATEGORICAL,
    },
}


def validate_variable_name(name: str | None) -> str:
    """Validate and sanitize a variable name.

    The name must contain only English letters, digits, underscores, and
    Korean Hangul characters. It must not start with a digit. Whitespace
    characters are replaced with underscores before validation.

    Args:
        name: The proposed variable name. May contain whitespace which
            will be converted to ``_``.

    Returns:
        The sanitized variable name.

    Raises:
        ValidationError: If the name is empty, ``None``, or contains
            characters outside the permitted set.
    """
    if name is None or str(name).strip() == "":
        raise ValidationError(
            "Variable name cannot be empty or None.",
            details={"name": name},
        )

    sanitized = str(name).strip().replace(" ", "_")

    # Replace special characters with underscores
    sanitized = re.sub(r'[^A-Za-z0-9_\uAC00-\uD7A3]', '_', sanitized)

    # Prefix auto-generated numeric names
    if sanitized and sanitized[0].isdigit():
        sanitized = "var_" + sanitized

    if not _VARIABLE_NAME_PATTERN.match(sanitized):
        raise ValidationError(
            f"Invalid variable name '{sanitized}'. "
            "Names must start with a letter or underscore and contain only "
            "letters, digits, underscores, or Korean characters.",
            details={"name": sanitized},
        )

    return sanitized


def validate_measure_storage_compatibility(
    measure: MeasureType,
    storage: StorageType,
) -> bool:
    """Check whether a measurement type is compatible with a storage type.

    Args:
        measure: The statistical measurement level.
        storage: The physical storage type.

    Returns:
        ``True`` if the combination is valid.

    Raises:
        ValidationError: If the measure/storage combination is incompatible.
    """
    allowed = _MEASURE_STORAGE_COMPAT.get(measure)

    if allowed is None:
        # No restrictions registered for this measure type.
        return True

    if storage not in allowed:
        raise ValidationError(
            f"Storage type '{storage.value}' is not compatible with "
            f"measure type '{measure.value}'. "
            f"Allowed storage types: {', '.join(s.value for s in allowed)}.",
            details={
                "measure": measure.value,
                "storage": storage.value,
                "allowed": [s.value for s in allowed],
            },
        )

    return True


def validate_missing_rules(
    missing_values: list[Any] | None,
    allowed_min: float | int | None = None,
    allowed_max: float | int | None = None,
) -> bool:
    """Validate missing-value rules and optional range constraints.

    Ensures that:
    1. ``missing_values`` is a list (or ``None``).
    2. Individual range constraints (``allowed_min``, ``allowed_max``) are
       numeric when provided.
    3. If both ``allowed_min`` and ``allowed_max`` are provided,
       ``allowed_min`` is not greater than ``allowed_max``.

    Args:
        missing_values: A list of values that should be treated as missing,
            or ``None`` if no user-defined missing values exist.
        allowed_min: Optional lower bound for valid values. Values below
            this threshold may be treated as missing.
        allowed_max: Optional upper bound for valid values. Values above
            this threshold may be treated as missing.

    Returns:
        ``True`` if all rules are valid.

    Raises:
        ValidationError: If any rule violates structural or logical constraints.
    """
    if missing_values is not None and not isinstance(missing_values, list):
        raise ValidationError(
            "missing_values must be a list or None.",
            details={"missing_values": missing_values},
        )

    for bound_name, bound_value in [("allowed_min", allowed_min), ("allowed_max", allowed_max)]:
        if bound_value is not None and not isinstance(bound_value, (int, float)):
            raise ValidationError(
                f"{bound_name} must be a numeric value or None.",
                details={bound_name: bound_value},
            )

    if (
        allowed_min is not None
        and allowed_max is not None
        and allowed_min > allowed_max
    ):
        raise ValidationError(
            "allowed_min cannot be greater than allowed_max.",
            details={
                "allowed_min": allowed_min,
                "allowed_max": allowed_max,
            },
        )

    return True
