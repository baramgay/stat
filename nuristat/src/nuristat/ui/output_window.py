"""Output Window — 독립 결과 창.

분석 결과를 누적하여 표시하는 독립 창입니다.
단일 인스턴스로 관리되어 여러 창이 뜨지 않습니다.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class OutputWindow(QMainWindow):
    """독립 결과 창.

    Features:
    - 누적 출력 (새 분석 결과가 계속 추가됨)
    - HTML 렌더링
    - 텍스트/표/차트 표시
    - 저장/납비
    - 지우기
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("📊 누리스탯 결과")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self._setup_ui()
        self._setup_menus()

        # 출력 내용 저장
        self._output_history: list[str] = []

    def _setup_ui(self) -> None:
        """UI 구성."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 상단: 정보 바
        info_layout = QHBoxLayout()

        self.info_label = QLabel("결과가 여기에 누적됩니다")
        self.info_label.setStyleSheet("color: #5d6d7e; font-size: 11px;")
        info_layout.addWidget(self.info_label)

        info_layout.addStretch()

        # 버튼들
        self.btn_clear = QPushButton("🗑️ 지우기")
        self.btn_clear.setToolTip("모든 결과를 지웁니다")
        self.btn_clear.clicked.connect(self.clear_output)
        info_layout.addWidget(self.btn_clear)

        self.btn_save = QPushButton("💾 HTML")
        self.btn_save.setToolTip("결과를 HTML 파일로 저장합니다")
        self.btn_save.clicked.connect(self._save_output)
        info_layout.addWidget(self.btn_save)

        self.btn_word = QPushButton("📄 Word")
        self.btn_word.setToolTip("결과를 Word(.docx) 파일로 저장합니다")
        self.btn_word.clicked.connect(self._save_word)
        info_layout.addWidget(self.btn_word)

        layout.addLayout(info_layout)

        # 출력 영역
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #d5dbdb;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
            """
        )
        layout.addWidget(self.output_text)

        # 상태 표시줄
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("준비됨")

    def _setup_menus(self) -> None:
        """메뉴 구성."""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")

        save_action = QAction("HTML로 저장(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_output)
        file_menu.addAction(save_action)

        word_action = QAction("Word(.docx)로 저장(&W)", self)
        word_action.triggered.connect(self._save_word)
        file_menu.addAction(word_action)

        file_menu.addSeparator()

        close_action = QAction("닫기(&C)", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.hide)
        file_menu.addAction(close_action)

        # 편집 메뉴
        edit_menu = menubar.addMenu("편집(&E)")

        clear_action = QAction("모두 지우기", self)
        clear_action.triggered.connect(self.clear_output)
        edit_menu.addAction(clear_action)

    def add_output(self, content: str, output_type: str = "text") -> None:
        """결과 추가 (누적).

        Args:
            content: 출력 내용
            output_type: 출력 유형 (text, success, error, warning, analysis)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 유형별 스타일
        style_map = {
            "success": "color: #2ca02c;",
            "error": "color: #d62728;",
            "warning": "color: #ff7f0e;",
            "analysis": "color: #1f77b4;",
            "text": "color: #333333;",
        }
        style = style_map.get(output_type, "color: #333333;")

        # 구분선 + 타임스탬프 + 내용
        separator = "─" * 60
        formatted = f"""
<div style="margin: 8px 0;">
<div style="color: #95a5a6; font-size: 10px; margin-bottom: 4px;">
{separator}<br>
🕐 {timestamp}
</div>
<div style="{style}">
{content}
</div>
</div>
"""

        self._output_history.append(formatted)

        # HTML 모드로 설정
        self.output_text.setHtml(self._get_full_html())

        # 스크롤을 맨 아래로
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # 상태 업데이트
        self.statusbar.showMessage(f"출력 항목: {len(self._output_history)}개")

    def _get_full_html(self) -> str:
        """전체 HTML 생성."""
        header = """
        <html>
        <head>
        <style>
        body {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.5;
            color: #333;
        }
        table {
            border-collapse: collapse;
            margin: 8px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        </style>
        </head>
        <body>
        """

        footer = "</body></html>"

        content = "\n".join(self._output_history)
        return header + content + footer

    def clear_output(self) -> None:
        """모든 결과 지우기."""
        self._output_history.clear()
        self.output_text.clear()
        self.statusbar.showMessage("출력이 지워졌습니다")

    def _save_output(self) -> None:
        """결과 저장."""
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "nuristat_output.html", "HTML (*.html)"
        )
        if path:
            try:
                html = self._get_full_html()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
                self.statusbar.showMessage(f"저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")

    def _save_word(self) -> None:
        """결과를 Word(.docx) 파일로 저장."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Word 저장", "nuristat_output.docx", "Word (*.docx)"
        )
        if not path:
            return
        try:
            import re

            from docx import Document
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.shared import Inches

            doc = Document()
            doc.core_properties.title = "누리스탯 분석 결과"

            # 제목
            title = doc.add_heading("누리스탯 분석 결과", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # HTML → Word 변환 (단순 파싱)
            full_html = self._get_full_html()

            # 타임스탬프 섹션별로 분리
            sections = re.split(r'🕐 (\d{2}:\d{2}:\d{2})', full_html)
            for i in range(1, len(sections), 2):
                ts = sections[i]
                content = sections[i + 1] if i + 1 < len(sections) else ""

                doc.add_paragraph(f"── {ts} ──", style="Intense Quote")

                # 테이블 파싱
                tables = re.findall(r'<table[^>]*>(.*?)</table>', content, re.DOTALL)
                for tbl_html in tables:
                    # caption
                    caption_match = re.search(r'<caption[^>]*>(.*?)</caption>', tbl_html, re.DOTALL)
                    if caption_match:
                        cap_text = re.sub(r'<[^>]+>', '', caption_match.group(1)).strip()
                        doc.add_heading(cap_text, level=3)

                    # 행 추출
                    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tbl_html, re.DOTALL)
                    if not rows:
                        continue
                    # 열 수 계산
                    first_row_cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', rows[0], re.DOTALL)
                    n_cols = len(first_row_cells) if first_row_cells else 1
                    if n_cols == 0:
                        continue

                    word_tbl = doc.add_table(rows=0, cols=n_cols)
                    word_tbl.style = "Table Grid"

                    for row_html in rows:
                        cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row_html, re.DOTALL)
                        if not cells:
                            continue
                        row_cells = word_tbl.add_row().cells
                        for j, cell_html in enumerate(cells[:n_cols]):
                            text = re.sub(r'<[^>]+>', '', cell_html).strip()
                            if j < len(row_cells):
                                row_cells[j].text = text

                    doc.add_paragraph()

                # 이미지 파싱 (base64 inline)
                imgs = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', content)
                for b64_str in imgs:
                    try:
                        img_bytes = base64.b64decode(b64_str)
                        img_stream = io.BytesIO(img_bytes)
                        doc.add_picture(img_stream, width=Inches(5.5))
                        doc.add_paragraph()
                    except Exception:
                        pass

            doc.save(path)
            self.statusbar.showMessage(f"Word 저장 완료: {path}")
        except ImportError:
            QMessageBox.critical(self, "오류", "python-docx 패키지가 필요합니다.\npip install python-docx")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"Word 저장 실패:\n{exc}")

    def closeEvent(self, event) -> None:
        """닫기 버튼 → 숨기기."""
        self.hide()
        event.ignore()
