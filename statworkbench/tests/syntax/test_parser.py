"""Tests for SyntaxParser.

Covers:
- Basic command name extraction
- Parameter extraction from slash-subcommands
- Multi-command parsing
- Edge cases (empty input, no subcommands, etc.)
"""

from __future__ import annotations

import pytest

from statworkbench.syntax.parser import SyntaxParser


@pytest.fixture
def parser() -> SyntaxParser:
    """Return a fresh SyntaxParser instance."""
    return SyntaxParser()


# --------------------------------------------------------------------------- #
# parse_single
# --------------------------------------------------------------------------- #

class TestParseSingle:
    def test_frequencies(self, parser: SyntaxParser) -> None:
        text = (
            "FREQUENCIES VARIABLES=sex diagnosis\n"
            "  /ORDER=VALUE\n"
            "  /MISSING=EXCLUDE."
        )
        cmd = parser.parse_single(text)
        assert cmd.command == "FREQUENCIES"
        assert cmd.parameters.get("ORDER") == "VALUE"
        assert cmd.parameters.get("MISSING") == "EXCLUDE"
        assert cmd.raw_text == text

    def test_ttest_independent(self, parser: SyntaxParser) -> None:
        text = (
            "TTEST GROUPS=treatment(0 1)\n"
            "  /VARIABLES=systolic_bp\n"
            "  /CRITERIA=CI(0.95)."
        )
        cmd = parser.parse_single(text)
        assert cmd.command == "TTEST"
        assert cmd.parameters.get("VARIABLES") == "systolic_bp"
        assert cmd.parameters.get("CRITERIA") == "CI(0.95)"

    def test_regression(self, parser: SyntaxParser) -> None:
        text = (
            "REGRESSION\n"
            "  /DEPENDENT outcome_score\n"
            "  /METHOD=ENTER age sex\n"
            "  /STATISTICS COEFF R ANOVA CI(95)."
        )
        cmd = parser.parse_single(text)
        assert cmd.command == "REGRESSION"
        assert cmd.parameters.get("DEPENDENT") == "outcome_score"
        assert cmd.parameters.get("METHOD") == "ENTER age sex"
        assert cmd.parameters.get("STATISTICS") == "COEFF R ANOVA CI(95)"

    def test_descriptives(self, parser: SyntaxParser) -> None:
        text = "DESCRIPTIVES VARIABLES=age height weight\n  /STATISTICS=MEAN STDDEV MIN MAX."
        cmd = parser.parse_single(text)
        assert cmd.command == "DESCRIPTIVES"
        assert cmd.parameters.get("VARIABLES") == "age height weight"
        assert cmd.parameters.get("STATISTICS") == "MEAN STDDEV MIN MAX"

    def test_crosstabs(self, parser: SyntaxParser) -> None:
        text = (
            "CROSSTABS\n"
            "  /TABLES=sex BY treatment\n"
            "  /STATISTICS=CHISQ\n"
            "  /CELLS=COUNT ROW COLUMN TOTAL."
        )
        cmd = parser.parse_single(text)
        assert cmd.command == "CROSSTABS"
        assert cmd.parameters.get("TABLES") == "sex BY treatment"
        assert cmd.parameters.get("STATISTICS") == "CHISQ"
        assert cmd.parameters.get("CELLS") == "COUNT ROW COLUMN TOTAL"

    def test_crosstabs_with_layer(self, parser: SyntaxParser) -> None:
        text = (
            "CROSSTABS\n"
            "  /TABLES=sex BY treatment BY hospital\n"
            "  /STATISTICS=CHISQ."
        )
        cmd = parser.parse_single(text)
        assert cmd.parameters.get("TABLES") == "sex BY treatment BY hospital"

    def test_single_line_command(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age /ORDER=VALUE."
        cmd = parser.parse_single(text)
        assert cmd.command == "FREQUENCIES"
        assert cmd.parameters.get("VARIABLES") == "age"
        assert cmd.parameters.get("ORDER") == "VALUE"

    def test_no_subcommands(self, parser: SyntaxParser) -> None:
        text = "LIST."
        cmd = parser.parse_single(text)
        assert cmd.command == "LIST"
        assert cmd.parameters == {}


# --------------------------------------------------------------------------- #
# parse (multi-command)
# --------------------------------------------------------------------------- #

class TestParseMulti:
    def test_multiple_commands(self, parser: SyntaxParser) -> None:
        text = (
            "FREQUENCIES VARIABLES=sex\n  /ORDER=VALUE.\n\n"
            "DESCRIPTIVES VARIABLES=age\n  /STATISTICS=MEAN."
        )
        cmds = parser.parse(text)
        assert len(cmds) == 2
        assert cmds[0].command == "FREQUENCIES"
        assert cmds[1].command == "DESCRIPTIVES"

    def test_single_command_in_multi(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=sex\n  /ORDER=VALUE."
        cmds = parser.parse(text)
        assert len(cmds) == 1
        assert cmds[0].command == "FREQUENCIES"

    def test_empty_string(self, parser: SyntaxParser) -> None:
        cmds = parser.parse("")
        assert cmds == []

    def test_whitespace_only(self, parser: SyntaxParser) -> None:
        cmds = parser.parse("   \n\n   ")
        assert cmds == []

    def test_three_commands(self, parser: SyntaxParser) -> None:
        text = (
            "FREQUENCIES VARIABLES=a /ORDER=VALUE.\n\n"
            "TTEST GROUPS=g(0 1) /VARIABLES=v.\n\n"
            "REGRESSION /DEPENDENT y /METHOD=ENTER x."
        )
        cmds = parser.parse(text)
        assert len(cmds) == 3
        assert [c.command for c in cmds] == ["FREQUENCIES", "TTEST", "REGRESSION"]


# --------------------------------------------------------------------------- #
# _extract_command_name
# --------------------------------------------------------------------------- #

class TestExtractCommandName:
    def test_basic_extraction(self, parser: SyntaxParser) -> None:
        assert parser._extract_command_name("FREQUENCIES VARIABLES=age.") == "FREQUENCIES"

    def test_multiline(self, parser: SyntaxParser) -> None:
        text = "REGRESSION\n  /DEPENDENT y\n  /METHOD=ENTER x."
        assert parser._extract_command_name(text) == "REGRESSION"

    def test_lowercase_input(self, parser: SyntaxParser) -> None:
        assert parser._extract_command_name("frequencies variables=age.") == "FREQUENCIES"

    def test_mixed_case(self, parser: SyntaxParser) -> None:
        assert parser._extract_command_name("Regression /dependent y.") == "REGRESSION"

    def test_empty_raises(self, parser: SyntaxParser) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parser._extract_command_name("")

    def test_whitespace_only_raises(self, parser: SyntaxParser) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parser._extract_command_name("   \n  ")


# --------------------------------------------------------------------------- #
# _extract_parameters
# --------------------------------------------------------------------------- #

class TestExtractParameters:
    def test_multiple_parameters(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age\n  /ORDER=VALUE\n  /MISSING=EXCLUDE."
        params = parser._extract_parameters(text)
        assert params["ORDER"] == "VALUE"
        assert params["MISSING"] == "EXCLUDE"

    def test_single_parameter(self, parser: SyntaxParser) -> None:
        text = "DESCRIPTIVES VARIABLES=age\n  /STATISTICS=MEAN."
        params = parser._extract_parameters(text)
        assert params["STATISTICS"] == "MEAN"

    def test_no_parameters(self, parser: SyntaxParser) -> None:
        params = parser._extract_parameters("LIST.")
        assert params == {}

    def test_trailing_period_removed(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age /ORDER=VALUE."
        params = parser._extract_parameters(text)
        assert params["ORDER"] == "VALUE"
        assert not params["ORDER"].endswith(".")

    def test_parameters_case_insensitive_keys(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES variables=age /order=VALUE."
        params = parser._extract_parameters(text)
        # Keys should be upper-cased
        assert "ORDER" in params

    def test_value_with_spaces(self, parser: SyntaxParser) -> None:
        text = "REGRESSION\n  /METHOD=ENTER age sex baseline_score."
        params = parser._extract_parameters(text)
        assert params["METHOD"] == "ENTER age sex baseline_score"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

class TestParserEdgeCases:
    def test_command_with_numbers_in_name(self, parser: SyntaxParser) -> None:
        """Command names can contain numbers (e.g., PHASE2)."""
        text = "PHASE2 /OPTION=1."
        cmd = parser.parse_single(text)
        assert cmd.command == "PHASE2"

    def test_parameter_value_with_parentheses(self, parser: SyntaxParser) -> None:
        text = "TTEST /CRITERIA=CI(0.95)."
        params = parser._extract_parameters(text)
        assert params["CRITERIA"] == "CI(0.95)"

    def test_unicode_in_raw_text(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=\uc131\ubcc4 /ORDER=VALUE."
        cmd = parser.parse_single(text)
        assert cmd.command == "FREQUENCIES"
        assert cmd.parameters["ORDER"] == "VALUE"

    def test_created_at_is_utc(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age."
        cmd = parser.parse_single(text)
        assert cmd.created_at.tzinfo is not None

    def test_dataset_id_defaults_to_unknown(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age."
        cmd = parser.parse_single(text)
        assert cmd.dataset_id == "unknown"

    def test_result_id_is_none(self, parser: SyntaxParser) -> None:
        text = "FREQUENCIES VARIABLES=age."
        cmd = parser.parse_single(text)
        assert cmd.result_id is None
