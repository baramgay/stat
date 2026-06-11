"""Result models for statistical analyses."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from nuristat.core import i18n


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
    metadata: dict = {}

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
        img_type = self.metadata.get("type", "") if self.metadata else ""
        if img_type in ("profile_plot", "wordcloud_image"):
            try:
                if self.dataframe.empty or "image_bytes" not in self.dataframe.columns:
                    return f'<p><em>[이미지 없음: {self.title}]</em></p>'
                img_bytes = self.dataframe.iloc[0]["image_bytes"]
                if not isinstance(img_bytes, (bytes, bytearray)) or len(img_bytes) == 0:
                    return f'<p><em>[이미지 없음: {self.title}]</em></p>'
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                return (
                    f'<div style="margin:8px 0;">'
                    f'<h4 style="margin:4px 0;font-size:13px;">{i18n.tr_title(self.title)}</h4>'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;"/>'
                    f'</div>'
                )
            except Exception:
                pass
            return f'<p><em>[이미지 렌더링 실패: {self.title}]</em></p>'

        # 출력 언어에 따라 제목·컬럼·라벨 번역 (내부 데이터는 불변, 표시용 사본만)
        disp = i18n.tr_frame(self.dataframe)
        html = '<table class="result-table">\n'
        html += f"<caption>{i18n.tr_title(self.title)}</caption>\n"
        html += disp.to_html(index=True, border=0) or ""
        if self.footnotes:
            html += '<tfoot>\n'
            for note in self.footnotes:
                html += f"<tr><td colspan='{len(disp.columns)}'>"
                html += f"<small>{note}</small></td></tr>\n"
            html += '</tfoot>\n'
        html += '</table>'
        return html

    def to_markdown(self) -> str:
        """Render the table as Markdown."""
        md = f"### {i18n.tr_title(self.title)}\n\n"
        md += i18n.tr_frame(self.dataframe).to_markdown(index=True) or ""
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
        return self.dataframe.to_csv(**opts) or ""


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

    def add_table(self, table: ResultTable) -> AnalysisResult:
        """Append a result table and return *self* for chaining."""
        self.tables.append(table)
        return self

    def add_warning(self, message: str) -> AnalysisResult:
        """Append a warning and return *self* for chaining."""
        self.warnings.append(message)
        return self

    def add_note(self, message: str) -> AnalysisResult:
        """Append a note and return *self* for chaining."""
        self.notes.append(message)
        return self

    def add_assumption(self, table: ResultTable) -> AnalysisResult:
        """Append an assumption-check table and return *self*."""
        self.assumptions.append(table)
        return self

    def add_diagnostic(self, table: ResultTable) -> AnalysisResult:
        """Append a diagnostic table and return *self*."""
        self.diagnostics.append(table)
        return self

    def to_html(self) -> str:
        """Render all tables and figures as concatenated HTML."""
        parts = []
        for table in self.tables:
            parts.append(table.to_html())
        for table in self.assumptions:
            parts.append(table.to_html())
        for table in self.diagnostics:
            parts.append(table.to_html())
        # matplotlib Figure objects
        for fig in self.figures:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                            facecolor="white", edgecolor="none")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                parts.append(
                    f'<div style="margin:8px 0;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;height:auto;display:block;"/>'
                    f'</div>'
                )
            except Exception:
                pass
        if self.text_blocks:
            for block in self.text_blocks:
                parts.append(f"<pre style='margin:4px 0'>{block}</pre>")
        if self.notes:
            note_html = "<ul>" + "".join(f"<li>{n}</li>" for n in self.notes) + "</ul>"
            parts.append(f"<div class='notes' style='color:#2874a6'>{note_html}</div>")
        if self.warnings:
            warn_html = "<ul>" + "".join(f"<li>{w}</li>" for w in self.warnings) + "</ul>"
            parts.append(f"<div class='warnings' style='color:#d62728'>{warn_html}</div>")
        return "\n".join(parts)

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
