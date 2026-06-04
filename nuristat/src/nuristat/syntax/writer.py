"""SyntaxWriter — converts analysis specs to SPSS-style syntax and manages logs.

The SyntaxWriter generates SPSS-style syntax strings from analysis specifications
and records them as SyntaxCommand objects. Commands can be persisted to disk in
JSON Lines format for reproducibility.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nuristat.syntax.command import SyntaxCommand


class SyntaxWriter:
    """Converts analysis specifications to SPSS-style syntax and records them.

    The writer maintains an in-memory list of SyntaxCommand objects and can
    persist them to a JSON Lines log file. Each analysis type has a dedicated
    method that generates the appropriate SPSS-style syntax.

    Attributes:
        commands: Ordered list of recorded SyntaxCommand objects.
        log_path: Optional default path for saving/loading the syntax log.
    """

    def __init__(self, log_path: str | None = None) -> None:
        """Initialize the SyntaxWriter.

        Args:
            log_path: Optional default file path for save/load operations.
        """
        self.commands: list[SyntaxCommand] = []
        self.log_path = log_path

    # ------------------------------------------------------------------ #
    # Command management
    # ------------------------------------------------------------------ #

    def add_command(self, cmd: SyntaxCommand) -> None:
        """Add a pre-built SyntaxCommand to the log.

        Args:
            cmd: The SyntaxCommand to append.
        """
        self.commands.append(cmd)

    # ------------------------------------------------------------------ #
    # Analysis-specific syntax generators
    # ------------------------------------------------------------------ #

    def write_frequencies(
        self,
        variables: list[str],
        dataset_id: str,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate FREQUENCIES syntax.

        Example output::

            FREQUENCIES VARIABLES=sex diagnosis
              /ORDER=VALUE
              /MISSING=EXCLUDE.

        Args:
            variables: List of variable names to analyze.
            dataset_id: Identifier of the target dataset.
            options: Optional dict with keys such as ``order``, ``missing``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        order = opts.get("order", "VALUE")
        missing = opts.get("missing", "EXCLUDE")
        var_list = " ".join(variables)

        raw = (
            f"FREQUENCIES VARIABLES={var_list}\n"
            f"  /ORDER={order}\n"
            f"  /MISSING={missing}."
        )

        parameters = {
            "variables": variables,
            "order": order,
            "missing": missing,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="FREQUENCIES",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    def write_ttest_independent(
        self,
        dependent: str,
        group: str,
        group_values: tuple[Any, Any],
        dataset_id: str,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate independent-samples TTEST syntax.

        Example output::

            TTEST GROUPS=treatment(0 1)
              /VARIABLES=systolic_bp
              /CRITERIA=CI(.95).

        Args:
            dependent: Name of the dependent (continuous) variable.
            group: Name of the grouping variable.
            group_values: Tuple of the two group values.
            dataset_id: Identifier of the target dataset.
            options: Optional dict with keys such as ``confidence_level``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        confidence = opts.get("confidence_level", 0.95)
        val1, val2 = group_values

        raw = (
            f"TTEST GROUPS={group}({val1} {val2})\n"
            f"  /VARIABLES={dependent}\n"
            f"  /CRITERIA=CI({confidence:.2f})."
        )

        parameters = {
            "dependent": dependent,
            "group": group,
            "group_values": list(group_values),
            "confidence_level": confidence,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="TTEST",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    def write_ttest_paired(
        self,
        var_pairs: list[tuple[str, str]],
        dataset_id: str,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate paired-samples TTEST syntax.

        Example output::

            TTEST PAIRS=var1 WITH var2 (PAIRED)
              /CRITERIA=CI(.95).

        For multiple pairs::

            TTEST PAIRS=var1 var2 WITH var3 var4 (PAIRED)
              /CRITERIA=CI(.95).

        Args:
            var_pairs: List of (var_before, var_after) tuples.
            dataset_id: Identifier of the target dataset.
            options: Optional dict with keys such as ``confidence_level``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        confidence = opts.get("confidence_level", 0.95)

        left_vars = " ".join(v1 for v1, _ in var_pairs)
        right_vars = " ".join(v2 for _, v2 in var_pairs)

        raw = (
            f"TTEST PAIRS={left_vars} WITH {right_vars} (PAIRED)\n"
            f"  /CRITERIA=CI({confidence:.2f})."
        )

        parameters = {
            "var_pairs": [list(pair) for pair in var_pairs],
            "confidence_level": confidence,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="TTEST",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    def write_regression(
        self,
        dependent: str,
        predictors: list[str],
        dataset_id: str,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate linear REGRESSION syntax.

        Example output::

            REGRESSION
              /DEPENDENT outcome_score
              /METHOD=ENTER age sex baseline_score
              /STATISTICS COEFF R ANOVA CI(95).

        Args:
            dependent: Name of the dependent variable.
            predictors: List of predictor variable names.
            dataset_id: Identifier of the target dataset.
            options: Optional dict with keys such as ``method``, ``statistics``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        method = opts.get("method", "ENTER")
        statistics = opts.get("statistics", "COEFF R ANOVA CI(95)")
        predictor_list = " ".join(predictors)

        raw = (
            f"REGRESSION\n"
            f"  /DEPENDENT {dependent}\n"
            f"  /METHOD={method} {predictor_list}\n"
            f"  /STATISTICS {statistics}."
        )

        parameters = {
            "dependent": dependent,
            "predictors": predictors,
            "method": method,
            "statistics": statistics,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="REGRESSION",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    def write_descriptives(
        self,
        variables: list[str],
        dataset_id: str,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate DESCRIPTIVES syntax.

        Example output::

            DESCRIPTIVES VARIABLES=age height weight
              /STATISTICS=MEAN STDDEV MIN MAX.

        Args:
            variables: List of variable names to describe.
            dataset_id: Identifier of the target dataset.
            options: Optional dict with keys such as ``statistics``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        statistics = opts.get("statistics", "MEAN STDDEV MIN MAX")
        var_list = " ".join(variables)

        raw = (
            f"DESCRIPTIVES VARIABLES={var_list}\n"
            f"  /STATISTICS={statistics}."
        )

        parameters = {
            "variables": variables,
            "statistics": statistics,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="DESCRIPTIVES",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    def write_crosstabs(
        self,
        row: str,
        col: str,
        dataset_id: str,
        layer: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> SyntaxCommand:
        """Generate CROSSTABS syntax.

        Example output (no layer)::

            CROSSTABS
              /TABLES=sex BY treatment
              /STATISTICS=CHISQ
              /CELLS=COUNT ROW COLUMN TOTAL.

        Example output (with layer)::

            CROSSTABS
              /TABLES=sex BY treatment BY hospital
              /STATISTICS=CHISQ
              /CELLS=COUNT ROW COLUMN TOTAL.

        Args:
            row: Row variable name.
            col: Column variable name.
            dataset_id: Identifier of the target dataset.
            layer: Optional layer variable name.
            options: Optional dict with keys such as ``statistics``, ``cells``.

        Returns:
            The generated SyntaxCommand.
        """
        opts = options or {}
        statistics = opts.get("statistics", "CHISQ")
        cells = opts.get("cells", "COUNT ROW COLUMN TOTAL")

        if layer:
            table_line = f"  /TABLES={row} BY {col} BY {layer}"
        else:
            table_line = f"  /TABLES={row} BY {col}"

        raw = (
            f"CROSSTABS\n"
            f"{table_line}\n"
            f"  /STATISTICS={statistics}\n"
            f"  /CELLS={cells}."
        )

        parameters = {
            "row": row,
            "col": col,
            "layer": layer,
            "statistics": statistics,
            "cells": cells,
        }
        parameters.update({k: v for k, v in opts.items() if k not in parameters})

        cmd = SyntaxCommand(
            command="CROSSTABS",
            parameters=parameters,
            raw_text=raw,
            created_at=_now(),
            dataset_id=dataset_id,
        )
        self.commands.append(cmd)
        return cmd

    # ------------------------------------------------------------------ #
    # Output and persistence
    # ------------------------------------------------------------------ #

    def to_string(self) -> str:
        """Return all recorded commands as a single syntax string.

        Commands are separated by a blank line.

        Returns:
            Concatenated SPSS-style syntax text.
        """
        return "\n\n".join(cmd.raw_text for cmd in self.commands)

    def save(self, path: str | None = None) -> None:
        """Save the command log to a JSON Lines file.

        Each line is a JSON-serialized SyntaxCommand.

        Args:
            path: File path to write to. Falls back to ``self.log_path``.

        Raises:
            ValueError: If no path is provided or configured.
        """
        target = path or self.log_path
        if target is None:
            raise ValueError("No log path provided. Set log_path in constructor or pass path to save().")

        # Ensure parent directory exists
        Path(target).parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as fh:
            for cmd in self.commands:
                fh.write(cmd.to_json_line() + "\n")

    def load(self, path: str) -> None:
        """Load commands from a JSON Lines file.

        Replaces any commands currently in memory.

        Args:
            path: File path to read from.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Syntax log file not found: {path}")

        loaded: list[SyntaxCommand] = []
        with open(path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = SyntaxCommand.from_json_line(line)
                    loaded.append(cmd)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse syntax log at line {line_num}: {exc}"
                    ) from exc

        self.commands = loaded

    def clear(self) -> None:
        """Clear all recorded commands from memory."""
        self.commands.clear()

    def __len__(self) -> int:
        """Return the number of recorded commands."""
        return len(self.commands)

    def __iter__(self) -> Iterator[SyntaxCommand]:
        """Iterate over recorded commands."""
        return iter(self.commands)

    def __repr__(self) -> str:
        return f"SyntaxWriter(commands={len(self.commands)}, log_path={self.log_path!r})"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)
