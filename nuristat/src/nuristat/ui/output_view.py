"""Output View — display analysis results."""

import base64

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.ui.theme import get_output_html_styles


class OutputView(QWidget):
    """View for displaying analysis results."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: list[AnalysisResult] = []
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

        # Detail view container (toolbar + editor)
        detail_container = QWidget()
        detail_layout = QVBoxLayout(detail_container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(4)

        html_copy_btn = QPushButton("HTML 복사")
        html_copy_btn.setToolTip("현재 결과의 HTML을 클립보드에 복사합니다")
        html_copy_btn.setFixedHeight(24)
        html_copy_btn.clicked.connect(self._copy_html)
        toolbar.addWidget(html_copy_btn)

        text_copy_btn = QPushButton("텍스트 복사")
        text_copy_btn.setToolTip("현재 결과의 일반 텍스트를 클립보드에 복사합니다")
        text_copy_btn.setFixedHeight(24)
        text_copy_btn.clicked.connect(self._copy_text)
        toolbar.addWidget(text_copy_btn)

        export_btn = QPushButton("내보내기")
        export_btn.setToolTip("전체 출력을 HTML 파일로 저장합니다")
        export_btn.setFixedHeight(24)
        export_btn.clicked.connect(self._export_to_file)
        toolbar.addWidget(export_btn)

        toolbar.addStretch()

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        detail_layout.addWidget(toolbar_widget)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        detail_layout.addWidget(self.detail)

        self.splitter.addWidget(detail_container)

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

    def _copy_html(self) -> None:
        """현재 detail 뷰의 HTML을 클립보드에 복사합니다."""
        html = self.detail.toHtml()
        QApplication.clipboard().setText(html)

    def _copy_text(self) -> None:
        """현재 detail 뷰의 일반 텍스트를 클립보드에 복사합니다."""
        text = self.detail.toPlainText()
        QApplication.clipboard().setText(text)

    def _export_to_file(self) -> None:
        """전체 출력을 HTML 파일로 저장합니다."""
        path, _ = QFileDialog.getSaveFileName(
            self, "HTML로 내보내기", "", "HTML 파일 (*.html *.htm)"
        )
        if not path:
            return
        try:
            html_content = self.export_html()
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"파일 저장 실패:\n{exc}")

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()
        self.tree.clear()
        self.detail.clear()

    def export_html(self) -> str:
        """Export all results as a styled standalone HTML report."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_analyses = len(self._results)

        # ── 내비게이션 목차 ───────────────────────────────────────
        toc_items = []
        for i, result in enumerate(self._results):
            anchor = f"result_{i}"
            toc_items.append(f'<li><a href="#{anchor}">{i+1}. {result.title}</a></li>')
        toc_html = "<ul>" + "\n".join(toc_items) + "</ul>" if toc_items else ""

        # ── 각 결과 섹션 ──────────────────────────────────────────
        body_parts = []
        for i, result in enumerate(self._results):
            anchor = f"result_{i}"
            body_parts.append(f'<section id="{anchor}" class="result-section">')
            body_parts.append(f'<h2><span class="result-num">{i+1}</span> {result.title}</h2>')
            ts = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
            body_parts.append(f'<p class="timestamp">분석 시각: {ts}</p>')

            if result.warnings:
                body_parts.append('<div class="warning-box"><strong>경고</strong><ul>')
                for w in result.warnings:
                    body_parts.append(f"<li>{w}</li>")
                body_parts.append("</ul></div>")

            if result.notes:
                body_parts.append('<div class="note-box"><ul>')
                for n in result.notes:
                    body_parts.append(f"<li>{n}</li>")
                body_parts.append("</ul></div>")

            for table in result.tables:
                body_parts.append(f'<h3 class="table-title">{table.title}</h3>')
                body_parts.append(self._table_to_html(table))
                if table.footnotes:
                    body_parts.append(
                        '<p class="footnote"><em>Note.</em> '
                        + " ".join(table.footnotes)
                        + "</p>"
                    )

            if result.syntax:
                body_parts.append(f'<pre class="syntax-block">{result.syntax}</pre>')

            body_parts.append("</section>")

        body_html = "\n".join(body_parts)

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>누리스탯 분석 결과 — {now}</title>
<style>
/* ── 기본 ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.6;
  color: #1a1a2e;
  background: #f5f7fa;
}}
a {{ color: #1a5276; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* ── 레이아웃 ── */
.page-header {{
  background: linear-gradient(135deg, #1a5276 0%, #2e86c1 100%);
  color: #fff;
  padding: 24px 32px;
  border-bottom: 3px solid #154360;
}}
.page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
.page-header .meta {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}

.container {{ max-width: 1100px; margin: 0 auto; padding: 24px 24px 48px; }}

/* ── 목차 ── */
.toc {{
  background: #fff;
  border: 1px solid #d5e8f5;
  border-left: 4px solid #2e86c1;
  border-radius: 6px;
  padding: 16px 20px;
  margin-bottom: 28px;
}}
.toc h4 {{ color: #1a5276; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.toc ul {{ padding-left: 20px; }}
.toc li {{ margin: 3px 0; font-size: 13px; }}

/* ── 결과 섹션 ── */
.result-section {{
  background: #fff;
  border: 1px solid #dde4ec;
  border-radius: 8px;
  padding: 24px 28px;
  margin-bottom: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.result-section h2 {{
  font-size: 17px;
  color: #1a5276;
  border-bottom: 2px solid #2e86c1;
  padding-bottom: 8px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.result-num {{
  background: #2e86c1;
  color: #fff;
  border-radius: 50%;
  width: 24px; height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}}
.timestamp {{ font-size: 11px; color: #7a7a8a; margin-bottom: 14px; }}

/* ── 테이블 제목 ── */
.table-title {{
  font-size: 13px;
  font-weight: 600;
  color: #2c3e50;
  margin: 18px 0 6px;
  padding-left: 8px;
  border-left: 3px solid #2e86c1;
}}

/* ── 테이블 ── */
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 12.5px;
  margin-bottom: 8px;
}}
thead tr {{
  background: #1a5276;
  color: #fff;
}}
thead th {{
  padding: 8px 12px;
  text-align: center;
  font-weight: 600;
  white-space: nowrap;
}}
tbody tr:nth-child(even) {{ background: #f3f8fc; }}
tbody tr:hover {{ background: #e8f4fc; }}
tbody td {{
  padding: 6px 12px;
  border-bottom: 1px solid #e8ecf0;
  text-align: right;
}}
tbody td:first-child {{ text-align: left; font-weight: 500; }}
tfoot td {{
  background: #eaf2f8;
  font-style: italic;
  font-size: 11.5px;
  padding: 4px 12px;
  color: #555;
}}

/* ── 경고 / 노트 ── */
.warning-box {{
  background: #fef9e7;
  border: 1px solid #f39c12;
  border-left: 4px solid #f39c12;
  border-radius: 4px;
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: 12.5px;
}}
.warning-box strong {{ color: #b7770d; }}
.note-box {{
  background: #eaf6fb;
  border: 1px solid #5dade2;
  border-left: 4px solid #2e86c1;
  border-radius: 4px;
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: 12.5px;
}}
.footnote {{ font-size: 11.5px; color: #666; margin-top: 4px; font-style: italic; }}
.syntax-block {{
  background: #1e1e2e;
  color: #cdd6f4;
  border-radius: 6px;
  padding: 14px;
  font-family: 'Consolas', 'D2Coding', monospace;
  font-size: 12px;
  overflow-x: auto;
  margin-top: 12px;
}}

/* ── 인쇄 ── */
@media print {{
  body {{ background: #fff; font-size: 11px; }}
  .page-header {{ background: #1a5276 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .result-section {{ box-shadow: none; page-break-inside: avoid; }}
  .toc {{ page-break-after: always; }}
  a {{ color: inherit; }}
}}
</style>
</head>
<body>
<div class="page-header">
  <h1>누리스탯 분석 결과</h1>
  <div class="meta">생성: {now} &nbsp;|&nbsp; 분석 {n_analyses}건</div>
</div>
<div class="container">
  <nav class="toc">
    <h4>목차</h4>
    {toc_html}
  </nav>
  {body_html}
</div>
</body>
</html>"""


# ── 모듈 수준 헬퍼 ───────────────────────────────────────────────────────────

def _qimage_to_png_bytes(image: QImage) -> bytes:
    """QImage를 PNG bytes로 변환."""
    from PySide6.QtCore import QBuffer, QIODevice

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(buffer.data())
