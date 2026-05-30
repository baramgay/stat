"""SyntaxCommand model for recording analysis operations.

Each analysis executed through the GUI is recorded as a SyntaxCommand,
which captures the SPSS-style syntax text, parameters, and metadata needed
for reproducibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SyntaxCommand(BaseModel):
    """A single recorded analysis command in SPSS-style syntax.

    Attributes:
        command: The command name (e.g., FREQUENCIES, TTEST, REGRESSION).
        parameters: Dictionary of parameters and options for the command.
        raw_text: The complete SPSS-style syntax string.
        created_at: Timestamp when the command was created.
        dataset_id: Identifier of the dataset used for the analysis.
        result_id: Optional identifier linking to the analysis result.
    """

    command: str = Field(..., description="Command name (FREQUENCIES, TTEST, REGRESSION, etc.)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    raw_text: str = Field(..., description="Complete SPSS-style syntax string")
    created_at: datetime = Field(..., description="Creation timestamp")
    dataset_id: str = Field(..., description="Dataset identifier")
    result_id: str | None = Field(default=None, description="Optional result identifier")

    model_config = {"populate_by_name": True}

    def to_json_line(self) -> str:
        """Serialize to a JSON Lines string.

        Returns:
            JSON string representation of this command.
        """
        return self.model_dump_json()

    @classmethod
    def from_json_line(cls, line: str) -> SyntaxCommand:
        """Deserialize from a JSON Lines string.

        Args:
            line: JSON string representing a SyntaxCommand.

        Returns:
            A new SyntaxCommand instance.
        """
        return cls.model_validate_json(line)

    def __str__(self) -> str:
        """Return the raw SPSS-style syntax text."""
        return self.raw_text

    def __repr__(self) -> str:
        return (
            f"SyntaxCommand(command={self.command!r}, "
            f"dataset_id={self.dataset_id!r}, "
            f"created_at={self.created_at.isoformat()!r})"
        )
