"""Tests for :class:`VariableMeta`.

Covers construction, field validators, measure/storage compatibility,
serialization round-trips, and public mutation methods.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from statworkbench.core.typing import MeasureType, Role, StorageType
from statworkbench.core.variable import VariableMeta
from statworkbench.core.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_kwargs() -> dict[str, Any]:
    return {"name": "age"}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Tests related to creating VariableMeta instances."""

    def test_minimal_construction(self) -> None:
        """A VariableMeta can be created with just a name."""
        var = VariableMeta(name="x")
        assert var.name == "x"
        assert var.label == ""
        assert var.storage_type == StorageType.FLOAT
        assert var.measure == MeasureType.SCALE
        assert var.role == Role.NONE
        assert var.width == 8
        assert var.decimals == 2

    def test_full_construction(self) -> None:
        """All fields can be provided at construction time."""
        var = VariableMeta(
            name="patient_id",
            label="Patient Identifier",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.NOMINAL,
            role=Role.ID,
            width=12,
            decimals=0,
            value_labels={"1": "Male", "2": "Female"},
            missing_values=[99, 999],
            unit="mmHg",
            allowed_min=0,
            allowed_max=200,
            description="Blood pressure measurement",
        )
        assert var.name == "patient_id"
        assert var.label == "Patient Identifier"
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.NOMINAL
        assert var.role == Role.ID
        assert var.value_labels == {"1": "Male", "2": "Female"}
        assert var.missing_values == [99, 999]
        assert var.unit == "mmHg"

    def test_name_whitespace_replacement(self) -> None:
        """Whitespace in names is converted to underscores."""
        var = VariableMeta(name="blood pressure")
        assert var.name == "blood_pressure"

    def test_name_stripping(self) -> None:
        """Leading and trailing whitespace is stripped."""
        var = VariableMeta(name="  age  ")
        assert var.name == "age"

    def test_korean_name(self) -> None:
        """Korean Hangul characters are accepted in variable names."""
        var = VariableMeta(name="나이")
        assert var.name == "나이"

    def test_korean_name_with_whitespace(self) -> None:
        """Korean name with spaces gets spaces replaced."""
        var = VariableMeta(name="혈압 측정값")
        assert var.name == "혈압_측정값"

    def test_name_none_raises(self) -> None:
        """`None`` as name raises ValueError."""
        with pytest.raises((ValidationError, ValueError)):
            VariableMeta(name=None)

    def test_empty_name_raises(self) -> None:
        """Empty string as name raises ValidationError."""
        with pytest.raises(ValidationError):
            VariableMeta(name="")

    def test_invalid_characters_raise(self) -> None:
        """Names with invalid characters are sanitized."""
        var = VariableMeta(name="age@2024!")
        assert var.name == "age_2024_"

    def test_name_starting_with_digit_raises(self) -> None:
        """Name starting with a digit is auto-prefixed with 'var_'."""
        var = VariableMeta(name="1st_var")
        assert var.name == "var_1st_var"

    def test_missing_values_none_coerced(self) -> None:
        """None` for missing_values is coerced to an empty list."""
        var = VariableMeta(name="x", missing_values=None)
        assert var.missing_values == []

    def test_missing_values_scalar_coerced(self) -> None:
        """A scalar missing_values is wrapped in a list."""
        var = VariableMeta(name="x", missing_values=99)
        assert var.missing_values == [99]

    def test_default_timestamps_set(self) -> None:
        """created_at and updated_at are set automatically."""
        before = datetime.now(timezone.utc)
        var = VariableMeta(name="x")
        after = datetime.now(timezone.utc)
        assert before <= var.created_at <= after
        assert before <= var.updated_at <= after


# ---------------------------------------------------------------------------
# Measure / Storage compatibility
# ---------------------------------------------------------------------------

class TestCompatibility:
    """Tests for measure/storage type validation."""

    def test_valid_combinations(self) -> None:
        """Common valid combinations should succeed."""
        valid_pairs = [
            (MeasureType.SCALE, StorageType.FLOAT),
            (MeasureType.SCALE, StorageType.INTEGER),
            (MeasureType.NOMINAL, StorageType.STRING),
            (MeasureType.NOMINAL, StorageType.INTEGER),
            (MeasureType.BINARY, StorageType.BOOLEAN),
            (MeasureType.BINARY, StorageType.INTEGER),
            (MeasureType.DATE_TIME, StorageType.DATETIME),
            (MeasureType.TEXT, StorageType.STRING),
        ]
        for measure, storage in valid_pairs:
            var = VariableMeta(name="x", measure=measure, storage_type=storage)
            assert var.measure == measure
            assert var.storage_type == storage

    def test_invalid_scale_with_string(self) -> None:
        """SCALE measure with STRING storage is incompatible."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", measure=MeasureType.SCALE, storage_type=StorageType.STRING)

    def test_invalid_nominal_with_float(self) -> None:
        """NOMINAL measure with FLOAT storage is incompatible."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", measure=MeasureType.NOMINAL, storage_type=StorageType.FLOAT)

    def test_invalid_binary_with_float(self) -> None:
        """BINARY measure with FLOAT storage is incompatible."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", measure=MeasureType.BINARY, storage_type=StorageType.FLOAT)

    def test_invalid_datetime_with_integer(self) -> None:
        """DATE_TIME measure with INTEGER storage is incompatible."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", measure=MeasureType.DATE_TIME, storage_type=StorageType.INTEGER)


# ---------------------------------------------------------------------------
# Missing-value rules validation
# ---------------------------------------------------------------------------

class TestMissingRules:
    """Tests for missing-values and range validation."""

    def test_valid_missing_values_list(self) -> None:
        """A list of missing values is accepted."""
        var = VariableMeta(name="x", missing_values=[-1, -2, -3])
        assert var.missing_values == [-1, -2, -3]

    def test_valid_range(self) -> None:
        """allowed_min < allowed_max is accepted."""
        var = VariableMeta(name="x", allowed_min=0, allowed_max=100)
        assert var.allowed_min == 0
        assert var.allowed_max == 100

    def test_min_greater_than_max_raises(self) -> None:
        """allowed_min > allowed_max raises ValidationError."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", allowed_min=100, allowed_max=0)

    def test_equal_min_max_allowed(self) -> None:
        """allowed_min == allowed_max is allowed."""
        var = VariableMeta(name="x", allowed_min=50, allowed_max=50)
        assert var.allowed_min == 50
        assert var.allowed_max == 50

    def test_non_numeric_min_raises(self) -> None:
        """Non-numeric allowed_min raises ValidationError."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", allowed_min="zero")

    def test_non_numeric_max_raises(self) -> None:
        """Non-numeric allowed_max raises ValidationError."""
        with pytest.raises(ValidationError):
            VariableMeta(name="x", allowed_max="hundred")


# ---------------------------------------------------------------------------
# Mutation methods
# ---------------------------------------------------------------------------

class TestMutation:
    """Tests for touch, set_value_label, add_missing_value."""

    def test_touch_updates_timestamp(self) -> None:
        """touch() advances updated_at."""
        var = VariableMeta(name="x")
        old_updated = var.updated_at
        var.touch()
        assert var.updated_at > old_updated

    def test_set_value_label(self) -> None:
        """set_value_label adds a value label and updates timestamp."""
        var = VariableMeta(name="x")
        old_updated = var.updated_at
        var.set_value_label("1", "Male")
        assert var.value_labels["1"] == "Male"
        assert var.updated_at >= old_updated

    def test_add_missing_value(self) -> None:
        """add_missing_value appends a new missing value."""
        var = VariableMeta(name="x", missing_values=[99])
        var.add_missing_value(999)
        assert 999 in var.missing_values
        assert 99 in var.missing_values

    def test_add_missing_value_no_duplicate(self) -> None:
        """add_missing_value does not duplicate existing values."""
        var = VariableMeta(name="x", missing_values=[99])
        var.add_missing_value(99)
        assert var.missing_values.count(99) == 1


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for to_dict / from_dict round-trips."""

    def test_round_trip_preserves_fields(self) -> None:
        """All fields survive a to_dict -> from_dict cycle."""
        original = VariableMeta(
            name="test_var",
            label="Test Variable",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.ORDINAL,
            role=Role.TARGET,
            width=10,
            decimals=0,
            value_labels={"1": "Low", "2": "High"},
            missing_values=[99],
            unit="mg/dL",
            allowed_min=0,
            allowed_max=500,
            format_pattern="F10.2",
            description="A test variable",
            derived=True,
            formula="x * 2",
        )
        d = original.to_dict()
        restored = VariableMeta.from_dict(d)

        assert restored.name == original.name
        assert restored.label == original.label
        assert restored.storage_type == original.storage_type
        assert restored.measure == original.measure
        assert restored.role == original.role
        assert restored.width == original.width
        assert restored.decimals == original.decimals
        assert restored.value_labels == original.value_labels
        assert restored.missing_values == original.missing_values
        assert restored.unit == original.unit
        assert restored.allowed_min == original.allowed_min
        assert restored.allowed_max == original.allowed_max
        assert restored.format_pattern == original.format_pattern
        assert restored.description == original.description
        assert restored.derived == original.derived
        assert restored.formula == original.formula

    def test_serialization_uses_string_enums(self) -> None:
        """to_dict` renders enum members as strings."""
        var = VariableMeta(name="x", storage_type=StorageType.INTEGER)
        d = var.to_dict()
        assert isinstance(d["storage_type"], str)
        assert d["storage_type"] == "integer"
        assert isinstance(d["measure"], str)
        assert isinstance(d["role"], str)

    def test_serialization_uses_iso_datetime(self) -> None:
        """to_dict` renders datetime objects as ISO-8601 strings."""
        var = VariableMeta(name="x")
        d = var.to_dict()
        assert isinstance(d["created_at"], str)
        assert "T" in d["created_at"]

    def test_from_dict_restores_enums(self) -> None:
        """from_dict` restores string enum values back to enum members."""
        d = {
            "name": "x",
            "storage_type": "integer",
            "measure": "nominal",
            "role": "input",
        }
        var = VariableMeta.from_dict(d)
        assert var.storage_type == StorageType.INTEGER
        assert var.measure == MeasureType.NOMINAL
        assert var.role == Role.INPUT

    def test_from_dict_restores_datetimes(self) -> None:
        """from_dict` restores ISO-8601 strings back to datetime objects."""
        d = {
            "name": "x",
            "created_at": "2024-06-15T10:30:00+00:00",
            "updated_at": "2024-06-15T12:00:00+00:00",
        }
        var = VariableMeta.from_dict(d)
        assert isinstance(var.created_at, datetime)
        assert var.created_at.year == 2024
        assert isinstance(var.updated_at, datetime)
        assert var.updated_at.hour == 12

    def test_from_dict_partial_fields(self) -> None:
        """from_dict` works with a sparse dictionary."""
        d = {"name": "y"}
        var = VariableMeta.from_dict(d)
        assert var.name == "y"
        assert var.label == ""
        assert var.storage_type == StorageType.FLOAT


# ---------------------------------------------------------------------------
# Dunder helpers
# ---------------------------------------------------------------------------

class TestDunder:
    """Tests for __repr__, __eq__, and __hash__."""

    def test_repr(self) -> None:
        """__repr__ includes key fields."""
        var = VariableMeta(name="x")
        r = repr(var)
        assert "VariableMeta" in r
        assert "x" in r

    def test_equality_by_name(self) -> None:
        """Two VariableMeta instances with the same name are equal."""
        a = VariableMeta(name="x", label="A")
        b = VariableMeta(name="x", label="B")
        assert a == b

    def test_inequality_by_name(self) -> None:
        """Two VariableMeta instances with different names are not equal."""
        a = VariableMeta(name="x")
        b = VariableMeta(name="y")
        assert a != b

    def test_hash_by_name(self) -> None:
        """Hash is based on the name."""
        a = VariableMeta(name="x")
        b = VariableMeta(name="x")
        assert hash(a) == hash(b)

    def test_eq_with_non_variablemeta(self) -> None:
        """Comparison with non-VariableMeta returns NotImplemented."""
        var = VariableMeta(name="x")
        assert var.__eq__("x") is NotImplemented
