"""Output View — display analysis results."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QHeaderView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from typing import Optional, List
import base64
import io

import pandas as pd

from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.ui.theme import get_output_html_styles, get_measure_badge_html


class OutputView(QWidget):
    """View for displaying analysis results."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: List[AnalysisResult] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        self.label = QLabel("Analysis Output")
        layout.addWidget(self.label)

        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Analysis Results")
        self.tree.setMaximumWidth(250)
        self.splitter.addWidget(self.tree)

        # Detail view
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.splitter.addWidget(self.detail)

        self.splitter.setSizes([250, 750])

    def add_result(self, result: AnalysisResult) -> None:
        """Add an analysis result to the output."""
        self._results.append(result)
        self._refresh_tree()
        self._display_result(result)

    def _refresh_tree(self) -> None:
        """Refresh the tree widget."""
        self.tree.clear()
        for i, result in enumerate(self._results):
            item = QTreeWidgetItem(self.tree)
            item.setText(0, f"{i+1}. {result.title}")
            item.setData(0, Qt.UserRole, i)

            # Add table children
            for table in result.tables:
                child = QTreeWidgetItem(item)
                child.setText(0, table.title)

        self.tree.expandAll()

    def _display_result(self, result: AnalysisResult) -> None:
        """Display a result in the detail view with themed styling."""
        html_parts = [get_output_html_styles()]
        html_parts.append("<body>")
        html_parts.append(f"<h2>{result.title}</h2>")
        html_parts.append(f"<p class='timestamp'>{result.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>")

        if result.warnings:
            html_parts.append("<div class='warning-box'><b>Warnings:</b><ul>")
            for w in result.warnings:
                html_parts.append(f"<li>{w}</li>")
            html_parts.append("</ul></div>")

        if result.notes:
            html_parts.append("<div class='note-box'><b>Notes:</b><ul>")
            for n in result.notes:
                html_parts.append(f"<li>{n}</li>")
            html_parts.append("</ul></div>")

        for table in result.tables:
            html_parts.append(f"<h3>{table.title}</h3>")
            html_parts.append(self._table_to_html(table))

            if table.footnotes:
                html_parts.append("<div class='footnote'><b>Note.</b> " + " ".join(table.footnotes) + "</div>")

        if result.syntax:
            html_parts.append("<h3>Syntax</h3>")
            html_parts.append(f"<div class='syntax-block'>{result.syntax}</div>")

        html_parts.append("</body></html>")
        self.detail.setHtml("\n".join(html_parts))

    def _table_to_html(self, table: ResultTable) -> str:
        """Convert a ResultTable to styled HTML."""
        df = table.dataframe
        html = ["<table>"]

        # Header
        html.append("<tr>")
        for col in df.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr>")

        # Rows
        for _, row in df.iterrows():
            html.append("<tr>")
            for val in row:
                cell = "" if pd.isna(val) else str(val)
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")

        html.append("</table>")
        return "\n".join(html)

    def add_chart(self, title: str, pixmap: QPixmap) -> None:
        """결과창에 차트 이미지를 삽입합니다.

        Parameters
        ----------
        title:  차트 제목 (트리 및 출력 헤더로 표시)
        pixmap: 표시할 QPixmap 이미지
        """
        if pixmap is None or pixmap.isNull():
            return

        # QPixmap → PNG bytes → base64 → HTML img 태그로 삽입
        buf = io.BytesIO()
        image = pixmap.toImage()
        png_bytes = _qimage_to_png_bytes(image)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'

        # 트리에 항목 추가
        idx = len(self._results)
        item = QTreeWidgetItem(self.tree)
        item.setText(0, f"{idx + 1}. [차트] {title}")
        item.setData(0, Qt.UserRole, f"__chart_{idx}")

        # 결과창에 HTML로 추가
        html_parts = [get_output_html_styles(), "<body>"]
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append(f"<div style='margin:12px 0;'>{img_tag}</div>")
        html_parts.append("</body></html>")

        # 기존 내용에 추가 (append)
        cursor = self.detail.textCursor()
        cursor.movePosition(cursor.End)
        self.detail.setTextCursor(cursor)
        self.detail.append("".join(html_parts))

        self.tree.expandAll()

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()
        self.tree.clear()
        self.detail.clear()

    def export_html(self) -> str:
        """Export all results as HTML."""
        parts = ["<html><body>"]
        for result in self._results:
            parts.append(f"<h2>{result.title}</h2>")
            for table in result.tables:
                parts.append(f"<h3>{table.title}</h3>")
                parts.append(self._table_to_html(table))
            if result.syntax:
                parts.append(f"<pre>{result.syntax}</pre>")
        parts.append("</body></html>")
        return "\n".join(parts)


# ── 모듈 수준 헬퍼 ───────────────────────────────────────────────────────────

def _qimage_to_png_bytes(image: QImage) -> bytes:
    """QImage를 PNG bytes로 변환."""
    from PySide6.QtCore import QBuffer, QIODevice

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(buffer.data())
