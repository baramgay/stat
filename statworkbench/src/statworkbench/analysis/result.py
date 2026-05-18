"""Result models for statistical analyses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class ResultTable(BaseModel):
    """A single table inside an analysis result.

    Attributes
    ----------
    title : str
        Table title (e.g. ``"Descriptive Statistics"``).
    dataframe : pd.DataFrame
        The tabular data.
    footnotes : list[str]
        Footnotes to display below the table.
    format_rules : dict
        Column-specific formatting rules.
        Example: ``{"p_value": "pvalue", "mean": ".3f"}``
    export_options : dict
        Export-specific options (e.g. ``{"index": False}``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    dataframe: pd.DataFrame
    footnotes: list[str] = []
    format_rules: dict = {}
    export_options: dict = {}

    @field_serializer("dataframe")
    def serialize_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Serialize DataFrame to a dict for JSON export."""
        return {
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
            "shape": list(df.shape),
        }

    def to_html(self) -> str:
        """Render the table as HTML."""
        html = f'<table class="result-table">\n'
        html += f"<caption>{self.title}</caption>\n"
        html += self.dataframe.to_html(index=True, border=0)
        if self.footnotes:
            html += '<tfoot>\n'
            for note in self.footnotes:
                html += f"<tr><td colspan='{len(self.dataframe.columns)}'>"
                html += f"<small>{note}</small></td></tr>\n"
            html += '</tfoot>\n'
        html += '</table>'
        return html

    def to_markdown(self) -> str:
        """Render the table as Markdown."""
        md = f"### {self.title}\n\n"
        md += self.dataframe.to_markdown(index=True)
        if self.footnotes:
            md += "\n\n"
            for note in self.footnotes:
                md += f"*Note.* {note}\n\n"
        return md

    def to_csv(self, **kwargs: Any) -> str:
        """Render the table as CSV string."""
        opts = {"index": False}
        opts.update(self.export_options)
        opts.update(kwargs)
        return self.dataframe.to_csv(**opts)


class AnalysisResult(BaseModel):
    """Structured result returned by every analysis plugin.

    Attributes
    ----------
    id : str
        Unique result identifier (UUID or timestamp-based).
    title : str
        Analysis title.
    created_at : datetime
        When the analysis was executed.
    spec : dict
        The analysis specification that produced this result.
    notes : list[str]
        Informational notes.
    warnings : list[str]
        Warning messages for the user.
    tables : list[ResultTable]
        Primary result tables.
    text_blocks : list[str]
        Free-form text blocks (interpretation, etc.).
    assumptions : list[ResultTable]
        Assumption-check tables (normality, homogeneity, etc.).
    diagnostics : list[ResultTable]
        Diagnostic tables (VIF, residual summary, etc.).
    figures : list[Any]
        Figure objects (matplotlib Figures, plotly Figures, etc.).
    syntax : str
        Syntax command that reproduces this analysis.
    metadata : dict
        Arbitrary extra metadata.
    """

    id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    spec: dict = {}
    notes: list[str] = []
    warnings: list[str] = []
    tables: list[ResultTable] = []
    text_blocks: list[str] = []
    assumptions: list[ResultTable] = []
    diagnostics: list[ResultTable] = []
    figures: list[Any] = []
    syntax: str = ""
    metadata: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_table(self, table: ResultTable) -> "AnalysisResult":
        """Append a result table and return *self* for chaining."""
        self.tables.append(table)
        return self

    def add_warning(self, message: str) -> "AnalysisResult":
        """Append a warning and return *self* for chaining."""
        self.warnings.append(message)
        return self

    def add_note(self, message: str) -> "AnalysisResult":
        """Append a note and return *self* for chaining."""
        self.notes.append(message)
        return self

    def add_assumption(self, table: ResultTable) -> "AnalysisResult":
        """Append an assumption-check table and return *self*."""
        self.assumptions.append(table)
        return self

    def add_diagnostic(self, table: ResultTable) -> "AnalysisResult":
        """Append a diagnostic table and return *self*."""
        self.diagnostics.append(table)
        return self

    def summary(self) -> str:
        """Return a short text summary of the result."""
        lines = [
            f"Analysis: {self.title}",
            f"  ID: {self.id}",
            f"  Tables: {len(self.tables)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Notes: {len(self.notes)}",
            f"  Assumptions checked: {len(self.assumptions)}",
        ]
        if self.warnings:
            lines.append("  Warning messages:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)
