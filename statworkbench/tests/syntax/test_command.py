"""Tests for SyntaxCommand model.

Covers:
- Construction and field access
- JSON Lines serialization / deserialization
- String and repr representations
- Edge cases
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from statworkbench.syntax.command import SyntaxCommand


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_command() -> SyntaxCommand:
    """Return a fully populated SyntaxCommand for testing."""
    return SyntaxCommand(
        command="FREQUENCIES",
        parameters={"variables": ["sex", "diagnosis"], "order": "VALUE"},
        raw_text="FREQUENCIES VARIABLES=sex diagnosis\n  /ORDER=VALUE\n  /MISSING=EXCLUDE.",
        created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        dataset_id="ds-001",
        result_id="res-042",
    )


@pytest.fixture
def minimal_command() -> SyntaxCommand:
    """Return a SyntaxCommand with only required fields."""
    return SyntaxCommand(
        command="TTEST",
        parameters={},
        raw_text="TTEST PAIRS=pre WITH post (PAIRED).",
        created_at=datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        dataset_id="ds-002",
    )


# --------------------------------------------------------------------------- #
# Construction & field access
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_full_construction(self, sample_command: SyntaxCommand) -> None:
        assert sample_command.command == "FREQUENCIES"
        assert sample_command.parameters == {"variables": ["sex", "diagnosis"], "order": "VALUE"}
        assert "FREQUENCIES VARIABLES=sex" in sample_command.raw_text
        assert sample_command.dataset_id == "ds-001"
        assert sample_command.result_id == "res-042"
        assert sample_command.created_at.isoformat() == "2025-01-15T10:30:00+00:00"

    def test_minimal_construction(self, minimal_command: SyntaxCommand) -> None:
        assert minimal_command.command == "TTEST"
        assert minimal_command.parameters == {}
        assert minimal_command.dataset_id == "ds-002"
        assert minimal_command.result_id is None

    def test_default_parameters_is_empty_dict(self) -> None:
        """When parameters is not supplied, it defaults to an empty dict."""
        cmd = SyntaxCommand(
            command="DESCRIPTIVES",
            raw_text="DESCRIPTIVES VARIABLES=age.",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-003",
        )
        assert cmd.parameters == {}

    def test_datetime_with_timezone(self) -> None:
        """created_at should be timezone-aware."""
        now = datetime.now(timezone.utc)
        cmd = SyntaxCommand(
            command="CROSSTABS",
            raw_text="CROSSTABS /TABLES=a BY b.",
            created_at=now,
            dataset_id="ds-004",
        )
        assert cmd.created_at.tzinfo is not None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

class TestSerialization:
    def test_to_json_line(self, sample_command: SyntaxCommand) -> None:
        """to_json_line() should return a valid JSON string."""
        line = sample_command.to_json_line()
        parsed = json.loads(line)
        assert parsed["command"] == "FREQUENCIES"
        assert parsed["parameters"]["variables"] == ["sex", "diagnosis"]
        assert parsed["dataset_id"] == "ds-001"
        assert parsed["result_id"] == "res-042"
        # datetime should be ISO-formatted (Pydantic v2 uses 'Z' for UTC)
        assert parsed["created_at"] in ("2025-01-15T10:30:00Z", "2025-01-15T10:30:00+00:00")

    def test_from_json_line_roundtrip(self, sample_command: SyntaxCommand) -> None:
        """Serialization roundtrip should preserve all fields."""
        line = sample_command.to_json_line()
        restored = SyntaxCommand.from_json_line(line)
        assert restored.command == sample_command.command
        assert restored.parameters == sample_command.parameters
        assert restored.raw_text == sample_command.raw_text
        assert restored.dataset_id == sample_command.dataset_id
        assert restored.result_id == sample_command.result_id
        assert restored.created_at == sample_command.created_at

    def test_from_json_line_without_result_id(self, minimal_command: SyntaxCommand) -> None:
        """Roundtrip with result_id=None should work."""
        line = minimal_command.to_json_line()
        restored = SyntaxCommand.from_json_line(line)
        assert restored.result_id is None

    def test_from_json_line_with_complex_parameters(self) -> None:
        """Parameters containing nested structures should roundtrip."""
        cmd = SyntaxCommand(
            command="REGRESSION",
            parameters={
                "dependent": "y",
                "predictors": ["x1", "x2", "x3"],
                "method": "ENTER",
                "confidence_level": 0.95,
                "nested": {"a": 1, "b": [True, None]},
            },
            raw_text="REGRESSION /DEPENDENT y /METHOD=ENTER x1 x2 x3.",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-005",
        )
        restored = SyntaxCommand.from_json_line(cmd.to_json_line())
        assert restored.parameters["nested"]["a"] == 1
        assert restored.parameters["nested"]["b"] == [True, None]


# --------------------------------------------------------------------------- #
# String representation
# --------------------------------------------------------------------------- #

class TestStringRepresentation:
    def test_str_returns_raw_text(self, sample_command: SyntaxCommand) -> None:
        assert str(sample_command) == sample_command.raw_text

    def test_repr_contains_command_and_dataset(self, sample_command: SyntaxCommand) -> None:
        r = repr(sample_command)
        assert "SyntaxCommand" in r
        assert "FREQUENCIES" in r
        assert "ds-001" in r


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_command_name_with_lowercase(self) -> None:
        """Command names are stored as-is (normalization is writer's job)."""
        cmd = SyntaxCommand(
            command="frequencies",  # lowercase
            raw_text="frequencies variables=age.",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-006",
        )
        assert cmd.command == "frequencies"

    def test_empty_raw_text(self) -> None:
        """Empty raw_text should be accepted (edge case)."""
        cmd = SyntaxCommand(
            command="NOP",
            raw_text="",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-007",
        )
        assert cmd.raw_text == ""

    def test_special_characters_in_raw_text(self) -> None:
        """Unicode and special characters in raw_text should be preserved."""
        text = "FREQUENCIES VARIABLES=\uc131\ubcc4 \ub0a8\uc790 \uc5ec\uc790 /ORDER=VALUE."
        cmd = SyntaxCommand(
            command="FREQUENCIES",
            raw_text=text,
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-008",
        )
        restored = SyntaxCommand.from_json_line(cmd.to_json_line())
        assert restored.raw_text == text

    def test_list_parameters_roundtrip(self) -> None:
        """List parameters should survive JSON roundtrip."""
        cmd = SyntaxCommand(
            command="DESCRIPTIVES",
            parameters={"variables": ["var_a", "var_b", "var_c"]},
            raw_text="DESCRIPTIVES VARIABLES=var_a var_b var_c.",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-009",
        )
        line = cmd.to_json_line()
        restored = SyntaxCommand.from_json_line(line)
        assert restored.parameters["variables"] == ["var_a", "var_b", "var_c"]

    def test_from_invalid_json_line_raises(self) -> None:
        """Invalid JSON should raise a validation error."""
        with pytest.raises(Exception):
            SyntaxCommand.from_json_line("not valid json {{{")

    def test_from_json_line_missing_required_field(self) -> None:
        """JSON missing required fields should raise validation error."""
        bad_json = json.dumps({"command": "ONLY_COMMAND"})
        with pytest.raises(Exception):
            SyntaxCommand.from_json_line(bad_json)
