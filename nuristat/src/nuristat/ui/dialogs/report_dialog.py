"""Report Dialog — 보고서 생성 다이얼로그.

HTML/PDF 보고서를 생성하고 저장합니다.
"""

from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from nuristat.core.dataset import Dataset
from nuristat.reporting.report_engine import ReportEngine


class ReportWorker(QThread):
    """보고서 생성 워커 스레드."""

    finished_signal = Signal(bool, str)
    progress_signal = Signal(str)

    def __init__(self, engine: ReportEngine, dataset: Dataset,
                 analyses: list[dict], title: str, author: str, path: str) -> None:
        super().__init__()
        self.engine = engine
        self.dataset = dataset
        self.analyses = analyses
        self.title = title
        self.author = author
        self.path = path

    def run(self) -> None:
        """보고서 생성."""
        try:
            self.progress_signal.emit("보고서 생성 중...")
            html = self.engine.generate_html_report(
                self.dataset, self.analyses, self.title, self.author
            )

            self.progress_signal.emit("파일 저장 중...")
            self.engine.save_html(html, self.path)

            self.finished_signal.emit(True, self.path)
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))


class ReportDialog(QDialog):
    """보고서 생성 다이얼로그."""

    report_generated = Signal(str)

    def __init__(self, dataset: Dataset, analyses: list[dict[str, Any]], parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.analyses = analyses
        self.engine = ReportEngine()
        self._worker: ReportWorker | None = None

        self.setWindowTitle("📄 보고서 생성")
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 정보
        info_group = QGroupBox("📋 보고서 정보")
        info_layout = QVBoxLayout(info_group)

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("제목:"))
        self.title_edit = QLineEdit("누리스탯 분석 보고서")
        title_layout.addWidget(self.title_edit)
        info_layout.addLayout(title_layout)

        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel("작성자:"))
        self.author_edit = QLineEdit()
        author_layout.addWidget(self.author_edit)
        info_layout.addLayout(author_layout)

        layout.addWidget(info_group)

        # 포함 내용
        content_group = QGroupBox("📊 포함 내용")
        content_layout = QVBoxLayout(content_group)

        self.summary_check = QCheckBox("데이터 요약")
        self.summary_check.setChecked(True)
        content_layout.addWidget(self.summary_check)

        self.overview_check = QCheckBox("데이터 개요 (기술통계)")
        self.overview_check.setChecked(True)
        content_layout.addWidget(self.overview_check)

        self.analyses_check = QCheckBox(f"분석 결과 ({len(self.analyses)}개)")
        self.analyses_check.setChecked(True)
        content_layout.addWidget(self.analyses_check)

        layout.addWidget(content_group)

        # 저장 위치
        save_group = QGroupBox("💾 저장 위치")
        save_layout = QHBoxLayout(save_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("저장할 경로를 선택하세요")
        save_layout.addWidget(self.path_edit)

        self.btn_browse = QPushButton("찾아보기...")
        self.btn_browse.clicked.connect(self._browse_path)
        save_layout.addWidget(self.btn_browse)

        layout.addWidget(save_group)

        # 진행 상황
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #3a5068;")
        layout.addWidget(self.status_label)

        # 실행 버튼
        action_layout = QHBoxLayout()

        self.btn_generate = QPushButton("📄 보고서 생성")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_generate.clicked.connect(self._generate_report)
        action_layout.addWidget(self.btn_generate)

        self.btn_close = QPushButton("❌ 닫기")
        self.btn_close.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_close)

        action_layout.addStretch()
        layout.addLayout(action_layout)

    def _browse_path(self) -> None:
        """저장 경로 선택."""
        path, _ = QFileDialog.getSaveFileName(
            self, "보고서 저장", "report.html", "HTML (*.html);;모든 파일 (*.*)"
        )
        if path:
            self.path_edit.setText(path)

    def _generate_report(self) -> None:
        """보고서 생성."""
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "경고", "저장 경로를 선택하세요")
            return

        title = self.title_edit.text().strip()
        author = self.author_edit.text().strip()

        # UI 상태 변경
        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("보고서 생성 중...")

        # 워커 스레드 실행
        self._worker = ReportWorker(self.engine, self.dataset, self.analyses, title, author, path)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, message: str) -> None:
        """진행 상황 업데이트."""
        self.status_label.setText(message)

    def _on_finished(self, success: bool, message: str) -> None:
        """생성 완료."""
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.status_label.setText(f"✅ 저장 완료: {message}")
            self.status_label.setStyleSheet("color: #2ca02c;")
            self.report_generated.emit(message)
            QMessageBox.information(self, "완료", f"보고서가 저장되었습니다.\n{message}")
        else:
            self.status_label.setText(f"❌ 오류: {message}")
            self.status_label.setStyleSheet("color: #d62728;")
            QMessageBox.critical(self, "오류", f"보고서 생성 실패:\n{message}")

    def closeEvent(self, event) -> None:
        """종료 시 워커 정리."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        event.accept()
