"""Tests for validation utilities.

Covers variable-name validation, measure/storage compatibility checks,
and missing-value rule validation.
"""

from __future__ import annotations

import pytest

from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.core.validation import (
    validate_measure_storage_compatibility,
    validate_missing_rules,
    validate_variable_name,
)
from statworkbench.core.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Variable name validation
# ---------------------------------------------------------------------------

class TestValidateVariableName:
    """Tests for :func:`validate_variable_name`."""

    def test_simple_name(self) -> None:
        """A simple alphabetic name is accepted."""
        assert validate_variable_name("age") == "age"

    def test_name_with_underscore(self) -> None:
        """Underscores are allowed."""
        assert validate_variable_name("blood_pressure") == "blood_pressure"

    def test_name_with_digits(self) -> None:
        """Digits after the first character are allowed."""
        assert validate_variable_name("age_2024") == "age_2024"

    def test_name_with_leading_underscore(self) -> None:
        """A leading underscore is allowed."""
        assert validate_variable_name("_private") == "_private"

    def test_korean_name(self) -> None:
        """Korean Hangul characters are accepted."""
        assert validate_variable_name("나이") == "나이"

    def test_korean_name_with_underscore(self) -> None:
        """Korean characters mixed with underscores."""
        assert validate_variable_name("혈압_값") == "혈압_값"

    def test_mixed_korean_english(self) -> None:
        """Korean and English mixed."""
        assert validate_variable_name("나이_age") == "나이_age"

    def test_whitespace_replaced(self) -> None:
        """Spaces are replaced with underscores."""
        assert validate_variable_name("blood pressure") == "blood_pressure"

    def test_multiple_spaces_replaced(self) -> None:
        """Multiple spaces are all replaced."""
        assert validate_variable_name("systolic bp mmHg") == "systolic_bp_mmHg"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert validate_variable_name("  age  ") == "age"

    def test_none_raises(self) -> None:
        """None raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_variable_name(None)

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_variable_name("")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_variable_name("   ")

    def test_leading_digit_raises(self) -> None:
        """Name starting with a digit is auto-prefixed with 'var_'."""
        assert validate_variable_name("1st_attempt") == "var_1st_attempt"

    def test_hyphen_raises(self) -> None:
        """Hyphen in name is sanitized to underscore."""
        result = validate_variable_name("item-count")
        assert result == "item_count"

    def test_at_sign_raises(self) -> None:
        """At sign in name is sanitized to underscore."""
        result = validate_variable_name("user@id")
        assert result == "user_id"

    def test_exclamation_raises(self) -> None:
        """Exclamation mark is sanitized to underscore."""
        result = validate_variable_name("rate!")
        assert result == "rate_"

    def test_dot_raises(self) -> None:
        """Period in name is sanitized to underscore."""
        result = validate_variable_name("item.1")
        assert result == "item_1"

    def test_slash_raises(self) -> None:
        """Slash in name is sanitized to underscore."""
        result = validate_variable_name("a/b")
        assert result == "a_b"

    def test_single_letter(self) -> None:
        """Single letter name is valid."""
        assert validate_variable_name("x") == "x"

    def test_single_underscore(self) -> None:
        """Single underscore name is valid."""
        assert validate_variable_name("_") == "_"


# ---------------------------------------------------------------------------
# Measure / storage compatibility
# ---------------------------------------------------------------------------

class TestValidateMeasureStorageCompatibility:
    """Tests for :func:`validate_measure_storage_compatibility`."""

    def test_scale_with_integer(self) -> None:
        """SCALE + INTEGER is valid."""
        assert validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.INTEGER)

    def test_scale_with_float(self) -> None:
        """SCALE + FLOAT is valid."""
        assert validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.FLOAT)

    def test_nominal_with_string(self) -> None:
        """NOMINAL + STRING is valid."""
        assert validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.STRING)

    def test_nominal_with_integer(self) -> None:
        """NOMINAL + INTEGER is valid."""
        assert validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.INTEGER)

    def test_nominal_with_categorical(self) -> None:
        """NOMINAL + CATEGORICAL is valid."""
        assert validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.CATEGORICAL)

    def test_ordinal_with_string(self) -> None:
        """ORDINAL + STRING is valid."""
        assert validate_measure_storage_compatibility(MeasureType.ORDINAL, StorageType.STRING)

    def test_binary_with_boolean(self) -> None:
        """BINARY + BOOLEAN is valid."""
        assert validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.BOOLEAN)

    def test_binary_with_integer(self) -> None:
        """BINARY + INTEGER is valid."""
        assert validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.INTEGER)

    def test_binary_with_string(self) -> None:
        """BINARY + STRING is valid."""
        assert validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.STRING)

    def test_datetime_with_datetime(self) -> None:
        """DATE_TIME + DATETIME is valid."""
        assert validate_measure_storage_compatibility(MeasureType.DATE_TIME, StorageType.DATETIME)

    def test_datetime_with_string(self) -> None:
        """DATE_TIME + STRING is valid."""
        assert validate_measure_storage_compatibility(MeasureType.DATE_TIME, StorageType.STRING)

    def test_text_with_string(self) -> None:
        """TEXT + STRING is valid."""
        assert validate_measure_storage_compatibility(MeasureType.TEXT, StorageType.STRING)

    def test_text_with_categorical(self) -> None:
        """TEXT + CATEGORICAL is valid."""
        assert validate_measure_storage_compatibility(MeasureType.TEXT, StorageType.CATEGORICAL)

    # Invalid combinations

    def test_scale_with_string_raises(self) -> None:
        """SCALE + STRING raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.STRING)

    def test_scale_with_boolean_raises(self) -> None:
        """SCALE + BOOLEAN raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.BOOLEAN)

    def test_scale_with_categorical_raises(self) -> None:
        """SCALE + CATEGORICAL raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.CATEGORICAL)

    def test_nominal_with_float_raises(self) -> None:
        """NOMINAL + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.FLOAT)

    def test_nominal_with_boolean_raises(self) -> None:
        """NOMINAL + BOOLEAN raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.BOOLEAN)

    def test_nominal_with_datetime_raises(self) -> None:
        """NOMINAL + DATETIME raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.DATETIME)

    def test_ordinal_with_float_raises(self) -> None:
        """ORDINAL + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.ORDINAL, StorageType.FLOAT)

    def test_binary_with_float_raises(self) -> None:
        """BINARY + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.FLOAT)

    def test_binary_with_datetime_raises(self) -> None:
        """BINARY + DATETIME raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.DATETIME)

    def test_datetime_with_integer_raises(self) -> None:
        """DATE_TIME + INTEGER raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.DATE_TIME, StorageType.INTEGER)

    def test_datetime_with_float_raises(self) -> None:
        """DATE_TIME + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.DATE_TIME, StorageType.FLOAT)

    def test_text_with_integer_raises(self) -> None:
        """TEXT + INTEGER raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.TEXT, StorageType.INTEGER)

    def test_text_with_float_raises(self) -> None:
        """TEXT + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.TEXT, StorageType.FLOAT)

    def test_error_includes_details(self) -> None:
        """Error message includes measure, storage, and allowed types."""
        with pytest.raises(ValidationError) as exc_info:
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.STRING)
        info = exc_info.value.details
        assert info["measure"] == "scale"
        assert info["storage"] == "string"
        assert "allowed" in info


# ---------------------------------------------------------------------------
# Missing-rules validation
# ---------------------------------------------------------------------------

class TestValidateMissingRules:
    """Tests for :func:`validate_missing_rules`."""

    def test_none_missing_values(self) -> None:
        """None missing_values is valid."""
        assert validate_missing_rules(None)

    def test_empty_list(self) -> None:
        """Empty list is valid."""
        assert validate_missing_rules([])

    def test_list_of_integers(self) -> None:
        """A list of integer missing values is valid."""
        assert validate_missing_rules([99, 999, 9999])

    def test_list_of_strings(self) -> None:
        """A list of string missing values is valid."""
        assert validate_missing_rules(["N/A", "Unknown"])

    def test_numeric_min_max(self) -> None:
        """Numeric allowed_min and allowed_max are valid."""
        assert validate_missing_rules(None, allowed_min=0, allowed_max=100)

    def test_only_min(self) -> None:
        """Only allowed_min is valid."""
        assert validate_missing_rules(None, allowed_min=0)

    def test_only_max(self) -> None:
        """Only allowed_max is valid."""
        assert validate_missing_rules(None, allowed_max=100)

    def test_min_equal_max(self) -> None:
        """allowed_min == allowed_max is valid."""
        assert validate_missing_rules(None, allowed_min=50, allowed_max=50)

    def test_negative_min(self) -> None:
        """Negative allowed_min is valid."""
        assert validate_missing_rules(None, allowed_min=-100)

    def test_float_bounds(self) -> None:
        """Float bounds are valid."""
        assert validate_missing_rules(None, allowed_min=0.0, allowed_max=1.0)

    def test_non_list_missing_values_raises(self) -> None:
        """Non-list, non-None missing_values raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_missing_rules("not a list")

    def test_non_numeric_min_raises(self) -> None:
        """Non-numeric allowed_min raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_min="zero")

    def test_non_numeric_max_raises(self) -> None:
        """Non-numeric allowed_max raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_max="hundred")

    def test_min_greater_than_max_raises(self) -> None:
        """allowed_min > allowed_max raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_min=100, allowed_max=0)

    def test_error_details_for_min_max(self) -> None:
        """Error includes min and max values in details."""
        with pytest.raises(ValidationError) as exc_info:
            validate_missing_rules(None, allowed_min=100, allowed_max=0)
        info = exc_info.value.details
        assert info["allowed_min"] == 100
        assert info["allowed_max"] == 0
