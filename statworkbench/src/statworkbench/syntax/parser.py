"""SyntaxParser — basic SPSS-style syntax parser.

MVP implementation extracts the command name and slash-prefixed subcommands.
Full parsing (values, nested structures) is planned for Phase 2.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from statworkbench.syntax.command import SyntaxCommand


class SyntaxParser:
    """Parse SPSS-style syntax text into SyntaxCommand objects.

    This is an MVP parser that extracts:

    * The first token as the command name.
    * ``/KEY=value`` subcommands as parameters.

    Phase 2 will add full value parsing, nested structures, and comment
    handling.
    """

    # Pattern to split a text block into individual commands.
    # Commands are terminated by a period at end-of-line.
    _COMMAND_SPLIT_RE = re.compile(r"\n*(?=[A-Z])")

    def parse(self, syntax_text: str) -> list[SyntaxCommand]:
        """Parse a multi-command syntax string into SyntaxCommand list.

        Args:
            syntax_text: SPSS-style syntax containing one or more commands.

        Returns:
            List of parsed SyntaxCommand objects.
        """
        commands: list[SyntaxCommand] = []
        raw_blocks = self._split_into_blocks(syntax_text)

        for block in raw_blocks:
            block = block.strip()
            try:
                cmd = self.parse_single(block)
                commands.append(cmd)
            except ValueError:
                # Skip unparsable blocks in MVP
                continue

        return commands

    def parse_single(self, command_text: str) -> SyntaxCommand:
        """Parse a single command text block.

        Args:
            command_text: Raw text of a single SPSS command (may span
                multiple lines).

        Returns:
            A SyntaxCommand representing the parsed command.

        Raises:
            ValueError: If the command text cannot be parsed.
        """
        cleaned = command_text.strip()
        if not cleaned:
            raise ValueError("Empty command text")

        command_name = self._extract_command_name(cleaned)
        parameters = self._extract_parameters(cleaned)

        # Try to infer dataset_id from parameters, otherwise use placeholder
        dataset_id = parameters.get("dataset_id", "unknown")

        return SyntaxCommand(
            command=command_name,
            parameters=parameters,
            raw_text=cleaned,
            created_at=datetime.now(timezone.utc),
            dataset_id=dataset_id,
        )

    def _extract_command_name(self, text: str) -> str:
        """Extract the first whitespace-delimited token as command name.

        Args:
            text: Raw command text.

        Returns:
            Upper-case command name.

        Raises:
            ValueError: If no command name can be extracted.
        """
        # Remove leading whitespace and take first token
        text = text.strip()
        if not text:
            raise ValueError("Empty command text")

        # Handle multi-line: take first line's first token
        first_line = text.splitlines()[0].strip()
        match = re.match(r'^([A-Za-z][A-Za-z0-9_]*)', first_line)
        if not match:
            raise ValueError(f"Cannot extract command name from: {first_line!r}")

        return match.group(1).upper()

    def _extract_parameters(self, text: str) -> dict[str, Any]:
        """Extract parameters from a command text.

        Captures:
        - First-line ``KEY=value`` after the command name (no ``/`` prefix).
        - Slash-prefixed ``/KEY=value`` subcommands.
        - Slash-prefixed ``/KEY value`` subcommands (no ``=``).

        Args:
            text: Raw command text.

        Returns:
            Dictionary of subcommand names to their values.
        """
        parameters: dict[str, Any] = {}

        # --- First-line KEY=value after command name (no / prefix) ---
        # e.g. "FREQUENCIES VARIABLES=age /ORDER=VALUE."
        # e.g. "DESCRIPTIVES VARIABLES=age height /STATISTICS=MEAN."
        # e.g. "TTEST GROUPS=gender(M F) /VARIABLES=score."
        first_line = text.splitlines()[0] if text else ""
        # Remove trailing period from first line
        first_line_clean = first_line.rstrip(". ")
        # Find KEY=value pairs after the first token (command name).
        # Value stops at '/' (next subcommand) or newline or end.
        for match in re.finditer(
            r'([A-Z][A-Z0-9_]*)\s*=\s*([^/\n].*?)(?=\s*/\s*|$)',
            first_line_clean,
            re.IGNORECASE,
        ):
            key = match.group(1).upper()
            value = match.group(2).strip()
            # Remove trailing period
            if value.endswith("."):
                value = value[:-1].strip()
            if value:
                parameters[key] = value

        # --- Slash-prefixed /KEY=value subcommands ---
        for match in re.finditer(r'/\s*([A-Z][A-Z0-9_]*)\s*=\s*([^\n]*)', text, re.IGNORECASE):
            key = match.group(1).upper()
            value = match.group(2).strip()
            # Remove trailing period if present
            if value.endswith("."):
                value = value[:-1].strip()
            parameters[key] = value

        # --- Slash-prefixed /KEY value subcommands (no =) ---
        # e.g. "/DEPENDENT outcome_score" or "/STATISTICS COEFF R"
        # This matches /KEY followed by text until next / or newline or end
        for match in re.finditer(
            r'/\s*([A-Z][A-Z0-9_]*)\s+([^=/\n][^/\n]*?)(?=\s*/\s*|\n|$)',
            text,
            re.IGNORECASE,
        ):
            key = match.group(1).upper()
            value = match.group(2).strip()
            # Remove trailing period
            if value.endswith("."):
                value = value[:-1].strip()
            # Only set if not already captured via = form
            if key not in parameters and value:
                parameters[key] = value

        return parameters

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split syntax text into individual command blocks.

        Commands are separated by a period followed by whitespace/newline.

        Args:
            text: Full syntax text with potentially multiple commands.

        Returns:
            List of individual command text blocks.
        """
        # Split on period followed by whitespace/newline or end of string
        # But be careful not to split inside parentheses
        blocks: list[str] = []
        current: list[str] = []

        for line in text.splitlines():
            current.append(line)
            # Check if line ends with a period (command terminator)
            stripped = line.strip()
            if stripped.endswith('.'):
                # End of a command block
                block = '\n'.join(current).strip()
                if block:
                    blocks.append(block)
                current = []

        # Handle remaining lines (command without trailing period)
        if current:
            block = '\n'.join(current).strip()
            if block:
                blocks.append(block)

        return blocks
