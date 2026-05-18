"""Tests for SyntaxWriter.

Covers:
- Each analysis type produces correct SPSS-style syntax
- add_command appends to the log
- to_string concatenation
- JSON Lines save / load roundtrip
- clear() empties the log
- Edge cases (empty variable list, custom options, etc.)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from statworkbench.syntax.command import SyntaxCommand
from statworkbench.syntax.writer import SyntaxWriter


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def writer() -> SyntaxWriter:
    """Return a fresh SyntaxWriter with no log path."""
    return SyntaxWriter()


@pytest.fixture
def writer_with_path(tmp_path: Path) -> SyntaxWriter:
    """Return a SyntaxWriter with a temporary log path."""
    log_file = tmp_path / "syntax.jsonl"
    return SyntaxWriter(log_path=str(log_file))


# --------------------------------------------------------------------------- #
# write_frequencies
# --------------------------------------------------------------------------- #

class TestWriteFrequencies:
    def test_basic_syntax(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_frequencies(
            variables=["sex", "diagnosis"],
            dataset_id="ds-001",
        )
        assert cmd.command == "FREQUENCIES"
        assert "FREQUENCIES VARIABLES=sex diagnosis" in cmd.raw_text
        assert "/ORDER=VALUE" in cmd.raw_text
        assert "/MISSING=EXCLUDE" in cmd.raw_text
        assert cmd.raw_text.endswith(".")

    def test_custom_options(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_frequencies(
            variables=["treatment"],
            dataset_id="ds-002",
            options={"order": "FREQUENCY", "missing": "INCLUDE"},
        )
        assert "/ORDER=FREQUENCY" in cmd.raw_text
        assert "/MISSING=INCLUDE" in cmd.raw_text
        assert cmd.parameters["order"] == "FREQUENCY"
        assert cmd.parameters["missing"] == "INCLUDE"

    def test_parameters_include_variables(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_frequencies(
            variables=["a", "b", "c"],
            dataset_id="ds-003",
        )
        assert cmd.parameters["variables"] == ["a", "b", "c"]

    def test_empty_variable_list(self, writer: SyntaxWriter) -> None:
        """Empty variable list should still produce syntactically valid output."""
        cmd = writer.write_frequencies(
            variables=[],
            dataset_id="ds-empty",
        )
        assert "FREQUENCIES VARIABLES=" in cmd.raw_text


# --------------------------------------------------------------------------- #
# write_ttest_independent
# --------------------------------------------------------------------------- #

class TestWriteTTestIndependent:
    def test_basic_syntax(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_independent(
            dependent="systolic_bp",
            group="treatment",
            group_values=(0, 1),
            dataset_id="ds-004",
        )
        assert cmd.command == "TTEST"
        assert "TTEST GROUPS=treatment(0 1)" in cmd.raw_text
        assert "/VARIABLES=systolic_bp" in cmd.raw_text
        assert "/CRITERIA=CI(0.95)" in cmd.raw_text
        assert cmd.raw_text.endswith(".")

    def test_different_group_values(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_independent(
            dependent="score",
            group="gender",
            group_values=("M", "F"),
            dataset_id="ds-005",
        )
        assert "GROUPS=gender(M F)" in cmd.raw_text
        assert cmd.parameters["group_values"] == ["M", "F"]

    def test_custom_confidence(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_independent(
            dependent="outcome",
            group="group_var",
            group_values=(1, 2),
            dataset_id="ds-006",
            options={"confidence_level": 0.99},
        )
        assert "CI(0.99)" in cmd.raw_text
        assert cmd.parameters["confidence_level"] == 0.99

    def test_parameters_structure(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_independent(
            dependent="dep",
            group="grp",
            group_values=(10, 20),
            dataset_id="ds-007",
        )
        assert cmd.parameters["dependent"] == "dep"
        assert cmd.parameters["group"] == "grp"
        assert cmd.parameters["confidence_level"] == 0.95


# --------------------------------------------------------------------------- #
# write_ttest_paired
# --------------------------------------------------------------------------- #

class TestWriteTTestPaired:
    def test_single_pair(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_paired(
            var_pairs=[("pre_score", "post_score")],
            dataset_id="ds-008",
        )
        assert cmd.command == "TTEST"
        assert "PAIRS=pre_score WITH post_score (PAIRED)" in cmd.raw_text
        assert "/CRITERIA=CI(0.95)" in cmd.raw_text

    def test_multiple_pairs(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_paired(
            var_pairs=[("t1", "t2"), ("t3", "t4")],
            dataset_id="ds-009",
        )
        assert "PAIRS=t1 t3 WITH t2 t4 (PAIRED)" in cmd.raw_text

    def test_parameters_contain_pairs(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_paired(
            var_pairs=[("a", "b"), ("c", "d")],
            dataset_id="ds-010",
        )
        assert cmd.parameters["var_pairs"] == [["a", "b"], ["c", "d"]]

    def test_custom_confidence(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_ttest_paired(
            var_pairs=[("v1", "v2")],
            dataset_id="ds-011",
            options={"confidence_level": 0.90},
        )
        assert "CI(0.90)" in cmd.raw_text


# --------------------------------------------------------------------------- #
# write_regression
# --------------------------------------------------------------------------- #

class TestWriteRegression:
    def test_basic_syntax(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_regression(
            dependent="outcome_score",
            predictors=["age", "sex", "baseline_score"],
            dataset_id="ds-012",
        )
        assert cmd.command == "REGRESSION"
        assert "/DEPENDENT outcome_score" in cmd.raw_text
        assert "/METHOD=ENTER age sex baseline_score" in cmd.raw_text
        assert "/STATISTICS COEFF R ANOVA CI(95)" in cmd.raw_text
        assert cmd.raw_text.endswith(".")

    def test_single_predictor(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_regression(
            dependent="y",
            predictors=["x"],
            dataset_id="ds-013",
        )
        assert "/METHOD=ENTER x" in cmd.raw_text

    def test_custom_statistics(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_regression(
            dependent="y",
            predictors=["x1", "x2"],
            dataset_id="ds-014",
            options={"statistics": "COEFF CI(99)"},
        )
        assert "/STATISTICS COEFF CI(99)" in cmd.raw_text

    def test_parameters_structure(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_regression(
            dependent="dep",
            predictors=["p1", "p2"],
            dataset_id="ds-015",
        )
        assert cmd.parameters["dependent"] == "dep"
        assert cmd.parameters["predictors"] == ["p1", "p2"]
        assert cmd.parameters["method"] == "ENTER"


# --------------------------------------------------------------------------- #
# write_descriptives
# --------------------------------------------------------------------------- #

class TestWriteDescriptives:
    def test_basic_syntax(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_descriptives(
            variables=["age", "height", "weight"],
            dataset_id="ds-016",
        )
        assert cmd.command == "DESCRIPTIVES"
        assert "DESCRIPTIVES VARIABLES=age height weight" in cmd.raw_text
        assert "/STATISTICS=MEAN STDDEV MIN MAX" in cmd.raw_text
        assert cmd.raw_text.endswith(".")

    def test_custom_statistics(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_descriptives(
            variables=["income"],
            dataset_id="ds-017",
            options={"statistics": "MEAN MEDIAN"},
        )
        assert "/STATISTICS=MEAN MEDIAN" in cmd.raw_text

    def test_parameters(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_descriptives(
            variables=["var1"],
            dataset_id="ds-018",
        )
        assert cmd.parameters["variables"] == ["var1"]


# --------------------------------------------------------------------------- #
# write_crosstabs
# --------------------------------------------------------------------------- #

class TestWriteCrosstabs:
    def test_basic_without_layer(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_crosstabs(
            row="sex",
            col="treatment",
            dataset_id="ds-019",
        )
        assert cmd.command == "CROSSTABS"
        assert "/TABLES=sex BY treatment" in cmd.raw_text
        assert "/STATISTICS=CHISQ" in cmd.raw_text
        assert "/CELLS=COUNT ROW COLUMN TOTAL" in cmd.raw_text
        assert cmd.raw_text.endswith(".")

    def test_with_layer(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_crosstabs(
            row="sex",
            col="treatment",
            dataset_id="ds-020",
            layer="hospital",
        )
        assert "/TABLES=sex BY treatment BY hospital" in cmd.raw_text

    def test_custom_options(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_crosstabs(
            row="a",
            col="b",
            dataset_id="ds-021",
            options={"statistics": "PHI", "cells": "COUNT EXPECTED"},
        )
        assert "/STATISTICS=PHI" in cmd.raw_text
        assert "/CELLS=COUNT EXPECTED" in cmd.raw_text

    def test_parameters_structure(self, writer: SyntaxWriter) -> None:
        cmd = writer.write_crosstabs(
            row="r",
            col="c",
            dataset_id="ds-022",
            layer="l",
        )
        assert cmd.parameters["row"] == "r"
        assert cmd.parameters["col"] == "c"
        assert cmd.parameters["layer"] == "l"


# --------------------------------------------------------------------------- #
# add_command
# --------------------------------------------------------------------------- #

class TestAddCommand:
    def test_add_command_increases_count(self, writer: SyntaxWriter) -> None:
        assert len(writer) == 0
        writer.add_command(
            SyntaxCommand(
                command="CUSTOM",
                raw_text="CUSTOM /OPTION=1.",
                created_at=datetime.now(timezone.utc),
                dataset_id="ds-add",
            )
        )
        assert len(writer) == 1

    def test_add_command_preserves_command(self, writer: SyntaxWriter) -> None:
        cmd = SyntaxCommand(
            command="CUSTOM",
            raw_text="CUSTOM CMD.",
            created_at=datetime.now(timezone.utc),
            dataset_id="ds-add2",
        )
        writer.add_command(cmd)
        assert writer.commands[0].command == "CUSTOM"


# --------------------------------------------------------------------------- #
# to_string
# --------------------------------------------------------------------------- #

class TestToString:
    def test_empty_writer(self, writer: SyntaxWriter) -> None:
        assert writer.to_string() == ""

    def test_single_command(self, writer: SyntaxWriter) -> None:
        writer.write_frequencies(variables=["x"], dataset_id="ds")
        result = writer.to_string()
        assert "FREQUENCIES" in result

    def test_multiple_commands_separated(self, writer: SyntaxWriter) -> None:
        writer.write_frequencies(variables=["x"], dataset_id="ds1")
        writer.write_descriptives(variables=["y"], dataset_id="ds2")
        result = writer.to_string()
        # Two commands should be separated by a blank line
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert "FREQUENCIES" in parts[0]
        assert "DESCRIPTIVES" in parts[1]


# --------------------------------------------------------------------------- #
# save / load roundtrip
# --------------------------------------------------------------------------- #

class TestSaveLoad:
    def test_save_creates_file(self, writer_with_path: SyntaxWriter) -> None:
        writer_with_path.write_frequencies(variables=["a"], dataset_id="ds")
        writer_with_path.save()
        assert os.path.exists(writer_with_path.log_path)

    def test_save_with_explicit_path(self, writer: SyntaxWriter, tmp_path: Path) -> None:
        path = str(tmp_path / "explicit.jsonl")
        writer.write_ttest_independent(
            dependent="d", group="g", group_values=(0, 1), dataset_id="ds",
        )
        writer.save(path)
        assert os.path.exists(path)

    def test_save_without_path_raises(self, writer: SyntaxWriter) -> None:
        with pytest.raises(ValueError, match="No log path"):
            writer.save()

    def test_load_roundtrip(self, writer_with_path: SyntaxWriter) -> None:
        writer_with_path.write_frequencies(
            variables=["sex", "age"], dataset_id="ds-rt",
        )
        writer_with_path.write_regression(
            dependent="y", predictors=["x1"], dataset_id="ds-rt",
        )
        writer_with_path.save()

        new_writer = SyntaxWriter()
        new_writer.load(writer_with_path.log_path)
        assert len(new_writer) == 2
        assert new_writer.commands[0].command == "FREQUENCIES"
        assert new_writer.commands[1].command == "REGRESSION"

    def test_load_preserves_parameters(self, writer_with_path: SyntaxWriter) -> None:
        writer_with_path.write_frequencies(
            variables=["var1", "var2"],
            dataset_id="ds-params",
            options={"order": "FREQUENCY"},
        )
        writer_with_path.save()

        new_writer = SyntaxWriter()
        new_writer.load(writer_with_path.log_path)
        assert new_writer.commands[0].parameters["variables"] == ["var1", "var2"]
        assert new_writer.commands[0].parameters["order"] == "FREQUENCY"

    def test_load_file_not_found(self, writer: SyntaxWriter) -> None:
        with pytest.raises(FileNotFoundError):
            writer.load("/nonexistent/path/syntax.jsonl")

    def test_load_invalid_json_line(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("this is not json\n")
        writer = SyntaxWriter()
        with pytest.raises(ValueError, match="Failed to parse"):
            writer.load(str(bad_file))

    def test_load_ignores_empty_lines(self, writer_with_path: SyntaxWriter) -> None:
        writer_with_path.write_frequencies(variables=["x"], dataset_id="ds")
        writer_with_path.save()
        # Append an empty line to the file
        with open(writer_with_path.log_path, "a") as fh:
            fh.write("\n\n")
        new_writer = SyntaxWriter()
        new_writer.load(writer_with_path.log_path)
        assert len(new_writer) == 1

    def test_jsonl_format(self, writer_with_path: SyntaxWriter) -> None:
        """Each line should be valid JSON with all expected fields."""
        writer_with_path.write_frequencies(variables=["x"], dataset_id="ds")
        writer_with_path.save()

        with open(writer_with_path.log_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                assert "command" in parsed
                assert "parameters" in parsed
                assert "raw_text" in parsed
                assert "created_at" in parsed
                assert "dataset_id" in parsed
                assert "result_id" in parsed


# --------------------------------------------------------------------------- #
# clear
# --------------------------------------------------------------------------- #

class TestClear:
    def test_clear_removes_all(self, writer: SyntaxWriter) -> None:
        writer.write_frequencies(variables=["x"], dataset_id="ds")
        writer.write_descriptives(variables=["y"], dataset_id="ds")
        assert len(writer) == 2
        writer.clear()
        assert len(writer) == 0
        assert writer.commands == []

    def test_clear_on_empty_writer(self, writer: SyntaxWriter) -> None:
        writer.clear()
        assert len(writer) == 0


# --------------------------------------------------------------------------- #
# iteration
# --------------------------------------------------------------------------- #

class TestIteration:
    def test_iterable(self, writer: SyntaxWriter) -> None:
        writer.write_frequencies(variables=["x"], dataset_id="ds")
        writer.write_descriptives(variables=["y"], dataset_id="ds")
        commands = list(writer)
        assert len(commands) == 2
        assert all(isinstance(c, SyntaxCommand) for c in commands)

    def test_len(self, writer: SyntaxWriter) -> None:
        assert len(writer) == 0
        writer.write_frequencies(variables=["x"], dataset_id="ds")
        assert len(writer) == 1


# --------------------------------------------------------------------------- #
# repr
# --------------------------------------------------------------------------- #

class TestRepr:
    def test_repr(self, writer: SyntaxWriter) -> None:
        r = repr(writer)
        assert "SyntaxWriter" in r
        assert "commands=0" in r


# --------------------------------------------------------------------------- #
# Command ordering
# --------------------------------------------------------------------------- #

class TestCommandOrdering:
    def test_commands_are_ordered(self, writer: SyntaxWriter) -> None:
        """Commands should be appended in the order they are generated."""
        writer.write_frequencies(variables=["a"], dataset_id="ds")
        writer.write_ttest_independent(
            dependent="d", group="g", group_values=(0, 1), dataset_id="ds",
        )
        writer.write_regression(
            dependent="y", predictors=["x"], dataset_id="ds",
        )
        names = [c.command for c in writer]
        assert names == ["FREQUENCIES", "TTEST", "REGRESSION"]
