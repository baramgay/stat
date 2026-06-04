"""Group D — Error-Resilience Tests: Core Modules.

Covers:
  - nuristat.core.dataset.Dataset
  - nuristat.core.variable.VariableMeta
  - nuristat.core.validation (validate_variable_name,
    validate_measure_storage_compatibility, validate_missing_rules)

Goal: exercise every unhandled-exception edge case so that normal-but-unexpected
usage cannot crash the application.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import DatasetError, ValidationError
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.validation import (
    validate_measure_storage_compatibility,
    validate_missing_rules,
    validate_variable_name,
)
from nuristat.core.variable import VariableMeta


# =============================================================================
# Dataset — edge-case construction
# =============================================================================

class TestDatasetConstruction:
    """Dataset.__init__ with unusual inputs."""

    def test_empty_dataframe(self):
        """Empty DataFrame produces a valid, is_empty Dataset."""
        ds = Dataset(data=pd.DataFrame())
        assert ds.is_empty
        assert ds.n_rows == 0
        assert ds.n_vars == 0

    def test_single_column(self):
        """Single-column DataFrame is handled correctly."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}))
        assert ds.n_vars == 1
        assert ds.n_rows == 3
        assert "x" in ds.variables

    def test_all_nan_dataframe(self):
        """All-NaN DataFrame is accepted without raising."""
        df = pd.DataFrame({"x": [np.nan] * 5, "y": [np.nan] * 5})
        ds = Dataset(data=df)
        assert ds.n_rows == 5
        assert ds.n_vars == 2

    def test_very_wide_dataframe(self):
        """500-column DataFrame is accepted without raising."""
        df = pd.DataFrame(np.random.randn(10, 500))
        ds = Dataset(data=df)
        assert ds.n_vars == 500
        assert ds.n_rows == 10

    def test_very_long_dataframe(self):
        """100,000-row DataFrame is accepted without raising."""
        df = pd.DataFrame({"x": np.random.randn(100_000)})
        ds = Dataset(data=df)
        assert ds.n_rows == 100_000

    def test_special_char_column_names(self):
        """Columns with spaces/dashes/dots are sanitised, not crashed."""
        df = pd.DataFrame({"my score": [1], "x-y": [2], "a.b": [3]})
        ds = Dataset(data=df)
        # Sanitised names exist in variables
        assert ds.n_vars == 3
        for v in ds.var_names:
            # Should not contain raw space, dash, or dot
            assert " " not in v
            assert "." not in v

    def test_custom_name_and_description(self):
        """name and description are stored correctly."""
        df = pd.DataFrame({"val": [1.0, 2.0]})
        ds = Dataset(data=df, name="MyDS", description="A test dataset")
        assert ds.name == "MyDS"
        assert ds.description == "A test dataset"

    def test_source_info_stored(self):
        """source_info dict is stored."""
        df = pd.DataFrame({"a": [1]})
        ds = Dataset(data=df, source_info={"file": "test.csv"})
        assert ds.source_info["file"] == "test.csv"

    def test_default_name_is_untitled(self):
        """Default name is 'Untitled'."""
        ds = Dataset(data=pd.DataFrame({"a": [1]}))
        assert ds.name == "Untitled"

    def test_is_not_dirty_after_init(self):
        """is_dirty is False immediately after construction."""
        ds = Dataset(data=pd.DataFrame({"a": [1]}))
        assert ds.is_dirty is False

    def test_repr_does_not_raise(self):
        """__repr__ never raises."""
        ds = Dataset(data=pd.DataFrame({"a": [1, 2]}))
        r = repr(ds)
        assert "Dataset" in r

    def test_len_returns_row_count(self):
        """len(dataset) returns n_rows."""
        ds = Dataset(data=pd.DataFrame({"a": range(7)}))
        assert len(ds) == 7

    def test_shape_property(self):
        """shape matches underlying DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds = Dataset(data=df)
        assert ds.shape == (3, 2)

    def test_variables_preset(self):
        """Passing variables= dict skips auto-inference."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        meta = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        ds = Dataset(data=df, variables={"x": meta})
        assert ds.variables["x"].measure == MeasureType.SCALE


class TestDatasetWithDuplicateLikeNames:
    """Pandas allows duplicate column names — Dataset must not crash."""

    def test_wide_numeric_columns_auto_named(self):
        """Auto-named columns (0..N) with digit-only names are prefixed."""
        df = pd.DataFrame(np.zeros((3, 4)))
        # columns are 0, 1, 2, 3 (int) — validate_variable_name prefixes them
        ds = Dataset(data=df)
        assert ds.n_vars == 4
        for name in ds.var_names:
            assert not name[0].isdigit()


# =============================================================================
# Dataset — operation edge cases
# =============================================================================

class TestDatasetOperations:

    def test_get_column_existing(self):
        """get_column returns a Series for an existing column."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}))
        col = ds.get_column("x")
        assert isinstance(col, pd.Series)
        assert len(col) == 3

    def test_get_column_missing_raises_keyerror(self):
        """get_column with unknown column raises KeyError (pandas default)."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(KeyError):
            ds.get_column("nonexistent")

    def test_get_variable_missing_raises_dataseterror(self):
        """get_variable raises DatasetError for unknown variable."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(DatasetError):
            ds.get_variable("nonexistent")

    def test_add_variable_existing_raises(self):
        """add_variable raises DatasetError when name already exists."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2]}))
        with pytest.raises(DatasetError):
            ds.add_variable("x")

    def test_add_variable_new_no_data(self):
        """add_variable without data creates an NA-filled column."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}))
        ds.add_variable("y")
        assert "y" in ds.variables
        assert ds.n_vars == 2

    def test_remove_variable_existing(self):
        """remove_variable removes column and meta."""
        ds = Dataset(data=pd.DataFrame({"x": [1], "y": [2]}))
        ds.remove_variable("x")
        assert "x" not in ds.variables
        assert ds.n_vars == 1

    def test_remove_variable_missing_raises(self):
        """remove_variable raises DatasetError for unknown variable."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(DatasetError):
            ds.remove_variable("nonexistent")

    def test_rename_variable_success(self):
        """rename_variable renames both DataFrame column and meta."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2]}))
        ds.rename_variable("x", "new_x")
        assert "new_x" in ds.variables
        assert "x" not in ds.variables

    def test_rename_variable_same_name_noop(self):
        """Renaming to the same name is a no-op without raising."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        ds.rename_variable("x", "x")
        assert "x" in ds.variables

    def test_rename_variable_missing_raises(self):
        """rename_variable raises DatasetError for unknown old_name."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(DatasetError):
            ds.rename_variable("nonexistent", "new")

    def test_rename_variable_duplicate_raises(self):
        """rename_variable raises DatasetError when new_name already exists."""
        ds = Dataset(data=pd.DataFrame({"x": [1], "y": [2]}))
        with pytest.raises(DatasetError):
            ds.rename_variable("x", "y")

    def test_update_variable_meta_valid_field(self):
        """update_variable_meta accepts valid VariableMeta fields."""
        ds = Dataset(data=pd.DataFrame({"x": [1.0, 2.0]}))
        ds.update_variable_meta("x", label="Test Label")
        assert ds.variables["x"].label == "Test Label"

    def test_update_variable_meta_invalid_field_raises(self):
        """update_variable_meta raises DatasetError for an invalid field."""
        ds = Dataset(data=pd.DataFrame({"x": [1.0]}))
        with pytest.raises(DatasetError):
            ds.update_variable_meta("x", nonexistent_field="boom")

    def test_update_variable_meta_missing_raises(self):
        """update_variable_meta raises DatasetError for unknown variable."""
        ds = Dataset(data=pd.DataFrame({"x": [1.0]}))
        with pytest.raises(DatasetError):
            ds.update_variable_meta("nonexistent", label="oops")

    def test_copy_is_independent(self):
        """copy() returns an independent Dataset."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}))
        ds2 = ds.copy()
        ds2.remove_variable("x")
        # Original must be intact
        assert "x" in ds.variables

    def test_data_setter_syncs_variables(self):
        """Assigning new data syncs variable metadata."""
        ds = Dataset(data=pd.DataFrame({"x": [1], "y": [2]}))
        ds.data = pd.DataFrame({"x": [10], "z": [30]})
        assert "y" not in ds.variables
        assert "z" in ds.variables

    def test_data_setter_non_dataframe_raises(self):
        """Assigning non-DataFrame to data raises DatasetError."""
        ds = Dataset(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(DatasetError):
            ds.data = [1, 2, 3]  # type: ignore[assignment]

    def test_variables_setter(self):
        """variables setter stores the provided dict."""
        ds = Dataset(data=pd.DataFrame({"x": [1.0]}))
        meta = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        ds.variables = {"x": meta}
        assert ds.variables["x"] is meta

    def test_is_dirty_after_mutation(self):
        """is_dirty is True after add/remove/rename."""
        ds = Dataset(data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        ds.remove_variable("y")
        assert ds.is_dirty is True


class TestDatasetSerialisation:
    """to_dict / from_dict round-trip."""

    def test_round_trip_numeric(self):
        """Numeric dataset survives round-trip."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ds = Dataset(data=df, name="RT", description="test")
        d = ds.to_dict()
        ds2 = Dataset.from_dict(d)
        assert ds2.name == "RT"
        assert ds2.n_rows == 2
        assert ds2.n_vars == 2

    def test_round_trip_empty(self):
        """Empty dataset survives round-trip."""
        ds = Dataset(data=pd.DataFrame())
        d = ds.to_dict()
        ds2 = Dataset.from_dict(d)
        assert ds2.is_empty

    def test_from_dict_missing_keys(self):
        """from_dict with minimal dict does not raise."""
        ds = Dataset.from_dict({})
        assert ds.is_empty


# =============================================================================
# VariableMeta — edge cases
# =============================================================================

class TestVariableMetaEdgeCases:

    def test_name_starting_with_digit_gets_prefixed(self):
        """VariableMeta name starting with digit is prefixed with var_."""
        vm = VariableMeta(name="1abc", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE)
        assert vm.name.startswith("var_")

    def test_very_long_name(self):
        """500-char name is accepted (after sanitisation)."""
        long_name = "a" * 500
        vm = VariableMeta(name=long_name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        assert len(vm.name) == 500

    def test_korean_name(self):
        """Korean variable name is accepted."""
        vm = VariableMeta(name="나이", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE)
        assert vm.name == "나이"

    def test_name_with_special_chars_sanitised(self):
        """Special chars in name are replaced with underscores."""
        vm = VariableMeta(
            name="my-score",
            storage_type=StorageType.FLOAT,
            measure=MeasureType.SCALE,
        )
        assert "-" not in vm.name
        assert vm.name == "my_score"

    def test_none_missing_values_coerced_to_list(self):
        """Passing missing_values=None is coerced to empty list."""
        vm = VariableMeta(
            name="x",
            storage_type=StorageType.FLOAT,
            measure=MeasureType.SCALE,
            missing_values=None,  # type: ignore[arg-type]
        )
        assert vm.missing_values == []

    def test_scalar_missing_value_coerced_to_list(self):
        """Scalar missing_values is coerced to a single-element list."""
        vm = VariableMeta(
            name="x",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.SCALE,
            missing_values=99,  # type: ignore[arg-type]
        )
        assert vm.missing_values == [99]

    def test_incompatible_measure_storage_raises(self):
        """SCALE + STRING raises ValidationError during __post_init__."""
        with pytest.raises(ValidationError):
            VariableMeta(
                name="bad",
                storage_type=StorageType.STRING,
                measure=MeasureType.SCALE,
            )

    def test_to_dict_and_from_dict_round_trip(self):
        """VariableMeta survives to_dict / from_dict."""
        vm = VariableMeta(
            name="score",
            label="Test Score",
            storage_type=StorageType.FLOAT,
            measure=MeasureType.SCALE,
            missing_values=[99, 999],
        )
        d = vm.to_dict()
        vm2 = VariableMeta.from_dict(d)
        assert vm2.name == "score"
        assert vm2.label == "Test Score"

    def test_set_value_label(self):
        """set_value_label adds label without raising."""
        vm = VariableMeta(
            name="gender",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.BINARY,
        )
        vm.set_value_label(1, "Male")
        vm.set_value_label(2, "Female")
        assert vm.value_labels[1] == "Male"

    def test_add_missing_value(self):
        """add_missing_value appends only unique values."""
        vm = VariableMeta(name="x", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE)
        vm.add_missing_value(99)
        vm.add_missing_value(99)
        assert vm.missing_values.count(99) == 1

    def test_validate_value_below_min(self):
        """validate_value returns warning when value < allowed_min."""
        vm = VariableMeta(
            name="age",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.SCALE,
            allowed_min=0,
            allowed_max=120,
        )
        warnings = vm.validate_value(-1)
        assert len(warnings) == 1

    def test_validate_value_above_max(self):
        """validate_value returns warning when value > allowed_max."""
        vm = VariableMeta(
            name="age",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.SCALE,
            allowed_min=0,
            allowed_max=120,
        )
        warnings = vm.validate_value(200)
        assert len(warnings) == 1

    def test_validate_value_nan_is_ok(self):
        """validate_value treats float NaN as missing — no warnings."""
        vm = VariableMeta(
            name="x",
            storage_type=StorageType.FLOAT,
            measure=MeasureType.SCALE,
            allowed_min=0,
            allowed_max=100,
        )
        warnings = vm.validate_value(float("nan"))
        assert warnings == []

    def test_validate_value_none_is_ok(self):
        """validate_value treats None as missing — no warnings."""
        vm = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        assert vm.validate_value(None) == []

    def test_is_numeric_true_for_float(self):
        """is_numeric is True for FLOAT storage."""
        vm = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        assert vm.is_numeric is True

    def test_is_categorical_true_for_nominal(self):
        """is_categorical is True for NOMINAL measure."""
        vm = VariableMeta(name="g", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL)
        assert vm.is_categorical is True

    def test_has_value_labels_false_by_default(self):
        """has_value_labels is False for a fresh VariableMeta."""
        vm = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        assert vm.has_value_labels is False

    def test_touch_updates_timestamp(self):
        """touch() updates updated_at."""
        vm = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        before = vm.updated_at
        vm.touch()
        assert vm.updated_at >= before

    def test_equality_by_name(self):
        """Two VariableMeta with the same name are equal."""
        a = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        b = VariableMeta(name="x", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE)
        assert a == b

    def test_hash_by_name(self):
        """VariableMeta can be used in a set (hashable)."""
        a = VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        b = VariableMeta(name="y", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)
        assert len({a, b}) == 2

    def test_min_greater_than_max_raises_validation(self):
        """allowed_min > allowed_max raises ValidationError."""
        with pytest.raises(ValidationError):
            VariableMeta(
                name="x",
                storage_type=StorageType.FLOAT,
                measure=MeasureType.SCALE,
                allowed_min=100,
                allowed_max=0,
            )


# =============================================================================
# validate_variable_name — comprehensive edge cases
# =============================================================================

class TestValidateVariableNameEdgeCases:

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            validate_variable_name("")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_variable_name(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            validate_variable_name("   ")

    def test_digit_prefix_gets_var_prefix(self):
        result = validate_variable_name("123abc")
        assert result.startswith("var_")
        assert "123abc" in result

    def test_korean_name_valid(self):
        assert validate_variable_name("한글이름") == "한글이름"

    def test_valid_name_unchanged(self):
        assert validate_variable_name("valid_name") == "valid_name"

    def test_space_replaced_with_underscore(self):
        result = validate_variable_name("my var")
        assert result == "my_var"

    def test_dash_replaced(self):
        result = validate_variable_name("a-b")
        assert result == "a_b"

    def test_dot_replaced(self):
        result = validate_variable_name("a.b")
        assert result == "a_b"

    def test_integer_zero_name(self):
        """Column name '0' (int converted to str) is prefixed."""
        result = validate_variable_name("0")
        assert result.startswith("var_")

    def test_leading_underscore_valid(self):
        assert validate_variable_name("_private") == "_private"

    def test_single_char_valid(self):
        assert validate_variable_name("x") == "x"

    def test_numeric_suffix_valid(self):
        result = validate_variable_name("var_01")
        assert result == "var_01"

    def test_very_long_name(self):
        """500-character name is sanitised without raising."""
        long = "a" * 500
        result = validate_variable_name(long)
        assert len(result) == 500

    def test_at_sign_replaced(self):
        result = validate_variable_name("user@id")
        assert "@" not in result

    def test_unicode_hangul_full_range(self):
        """Various Hangul syllables are all accepted."""
        result = validate_variable_name("가나다라마")
        assert result == "가나다라마"

    def test_mixed_korean_english(self):
        result = validate_variable_name("나이_age")
        assert result == "나이_age"


# =============================================================================
# validate_measure_storage_compatibility — edge cases beyond existing tests
# =============================================================================

class TestValidateMeasureStorageCompatibilityEdgeCases:

    def test_scale_with_string_raises(self):
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.STRING)

    def test_error_details_contain_allowed(self):
        """Error details must have 'allowed' key listing valid storage types."""
        with pytest.raises(ValidationError) as exc_info:
            validate_measure_storage_compatibility(MeasureType.SCALE, StorageType.STRING)
        assert "allowed" in exc_info.value.details

    def test_ordinal_float_valid(self):
        """ORDINAL + FLOAT is allowed (SPSS-compatible)."""
        result = validate_measure_storage_compatibility(MeasureType.ORDINAL, StorageType.FLOAT)
        assert result is True

    def test_binary_float_raises(self):
        """BINARY + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.BINARY, StorageType.FLOAT)

    def test_date_time_integer_raises(self):
        """DATE_TIME + INTEGER raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.DATE_TIME, StorageType.INTEGER)

    def test_text_float_raises(self):
        """TEXT + FLOAT raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.TEXT, StorageType.FLOAT)

    def test_nominal_boolean_raises(self):
        """NOMINAL + BOOLEAN raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_measure_storage_compatibility(MeasureType.NOMINAL, StorageType.BOOLEAN)

    def test_all_valid_combinations_return_true(self):
        """Spot-check that known-valid combinations all return True."""
        valid_pairs = [
            (MeasureType.SCALE, StorageType.INTEGER),
            (MeasureType.SCALE, StorageType.FLOAT),
            (MeasureType.NOMINAL, StorageType.STRING),
            (MeasureType.NOMINAL, StorageType.INTEGER),
            (MeasureType.NOMINAL, StorageType.CATEGORICAL),
            (MeasureType.BINARY, StorageType.BOOLEAN),
            (MeasureType.BINARY, StorageType.INTEGER),
            (MeasureType.DATE_TIME, StorageType.DATETIME),
            (MeasureType.DATE_TIME, StorageType.STRING),
            (MeasureType.TEXT, StorageType.STRING),
        ]
        for measure, storage in valid_pairs:
            assert validate_measure_storage_compatibility(measure, storage) is True


# =============================================================================
# validate_missing_rules — edge cases
# =============================================================================

class TestValidateMissingRulesEdgeCases:

    def test_list_of_ints_valid(self):
        assert validate_missing_rules([1, 2, 3]) is True

    def test_string_missing_values_raises(self):
        """Non-list missing_values raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_missing_rules("not a list")

    def test_dict_missing_values_raises(self):
        with pytest.raises(ValidationError):
            validate_missing_rules({"key": "val"})

    def test_int_missing_values_raises(self):
        with pytest.raises(ValidationError):
            validate_missing_rules(99)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_min=5, allowed_max=3)

    def test_min_equals_max_valid(self):
        """allowed_min == allowed_max is valid."""
        assert validate_missing_rules(None, allowed_min=5, allowed_max=5) is True

    def test_none_returns_true(self):
        assert validate_missing_rules(None) is True

    def test_empty_list_returns_true(self):
        assert validate_missing_rules([]) is True

    def test_float_bounds_valid(self):
        assert validate_missing_rules([], allowed_min=0.0, allowed_max=1.0) is True

    def test_string_min_raises(self):
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_min="zero")

    def test_string_max_raises(self):
        with pytest.raises(ValidationError):
            validate_missing_rules(None, allowed_max="hundred")

    def test_negative_min_valid(self):
        assert validate_missing_rules(None, allowed_min=-999) is True

    def test_error_details_for_inverted_bounds(self):
        """Details include both min and max values."""
        with pytest.raises(ValidationError) as exc_info:
            validate_missing_rules(None, allowed_min=100, allowed_max=0)
        d = exc_info.value.details
        assert d["allowed_min"] == 100
        assert d["allowed_max"] == 0

    def test_large_missing_value_list(self):
        """Very long missing_values list is valid."""
        big_list = list(range(1000))
        assert validate_missing_rules(big_list) is True
