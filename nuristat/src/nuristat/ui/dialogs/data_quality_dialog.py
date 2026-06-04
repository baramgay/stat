"""Data Quality Dialog — 데이터 품질 진단 다이얼로그.

결측치, 이상치, 중복, 데이터 타입 문제를 진단합니다.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.reporting.report_engine import ReportEngine


class DataQualityDialog(QDialog):
    """데이터 품질 진단 다이얼로그."""

    quality_report_generated = Signal(str)

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.engine = ReportEngine()

        self.setWindowTitle("🔍 데이터 품질 진단")
        self.setMinimumSize(800, 600)
        self._setup_ui()
        self._run_diagnostics()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 요약 정보
        self.summary_group = QGroupBox("📊 요약")
        summary_layout = QVBoxLayout(self.summary_group)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 12px;"
        )
        summary_layout.addWidget(self.summary_text)

        layout.addWidget(self.summary_group)

        # 탭
        self.tabs = QTabWidget()

        # 결측치 탭
        self.missing_tab = QWidget()
        missing_layout = QVBoxLayout(self.missing_tab)

        self.missing_table = QTableWidget()
        self.missing_table.setColumnCount(5)
        self.missing_table.setHorizontalHeaderLabels([
            "변수", "결측 수", "결측 비율", "데이터 타입", "상태"
        ])
        self.missing_table.horizontalHeader().setStretchLastSection(True)
        missing_layout.addWidget(self.missing_table)

        self.tabs.addTab(self.missing_tab, "⚠️ 결측치")

        # 이상치 탭
        self.outlier_tab = QWidget()
        outlier_layout = QVBoxLayout(self.outlier_tab)

        self.outlier_table = QTableWidget()
        self.outlier_table.setColumnCount(5)
        self.outlier_table.setHorizontalHeaderLabels([
            "변수", "이상치 수", "비율", "IQR 하한", "IQR 상한"
        ])
        outlier_layout.addWidget(self.outlier_table)

        self.tabs.addTab(self.outlier_tab, "🔍 이상치")

        # 중복 탭
        self.duplicate_tab = QWidget()
        dup_layout = QVBoxLayout(self.duplicate_tab)

        self.duplicate_text = QTextEdit()
        self.duplicate_text.setReadOnly(True)
        self.duplicate_text.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 12px;"
        )
        dup_layout.addWidget(self.duplicate_text)

        self.tabs.addTab(self.duplicate_tab, "🔄 중복")

        layout.addWidget(self.tabs)

        # 버튼
        btn_layout = QHBoxLayout()

        self.btn_report = QPushButton("📄 품질 보고서 생성")
        self.btn_report.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_report.clicked.connect(self._generate_report)
        btn_layout.addWidget(self.btn_report)

        self.btn_close = QPushButton("❌ 닫기")
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _run_diagnostics(self) -> None:
        """진단 실행."""
        df = self.dataset.data

        # 요약
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_pct = missing_cells / total_cells * 100

        duplicates = df.duplicated().sum()

        summary = f"""
        총 셀: {total_cells:,} | 결측: {missing_cells:,} ({missing_pct:.2f}%) | 중복 행: {duplicates:,}
        """
        self.summary_text.setText(summary.strip())

        # 결측치
        missing_data = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_data.append({
                    "변수": col,
                    "결측 수": missing_count,
                    "결측 비율": missing_count / len(df) * 100,
                    "데이터 타입": str(df[col].dtype),
                    "상태": "심각" if missing_count / len(df) > 0.5 else "주의" if missing_count / len(df) > 0.1 else "경고"
                })

        self.missing_table.setRowCount(len(missing_data))
        for i, row in enumerate(missing_data):
            self.missing_table.setItem(i, 0, QTableWidgetItem(row["변수"]))
            self.missing_table.setItem(i, 1, QTableWidgetItem(str(row["결측 수"])))
            self.missing_table.setItem(i, 2, QTableWidgetItem(f"{row['결측 비율']:.2f}%"))
            self.missing_table.setItem(i, 3, QTableWidgetItem(row["데이터 타입"]))

            status_item = QTableWidgetItem(row["상태"])
            if row["상태"] == "심각":
                status_item.setBackground(Qt.red)
                status_item.setForeground(Qt.white)
            elif row["상태"] == "주의":
                status_item.setBackground(Qt.yellow)
            self.missing_table.setItem(i, 4, status_item)

        self.missing_table.resizeColumnsToContents()

        # 이상치
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_data = []

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            if len(outliers) > 0:
                outlier_data.append({
                    "변수": col,
                    "이상치 수": len(outliers),
                    "비율": len(outliers) / len(df) * 100,
                    "IQR 하한": lower,
                    "IQR 상한": upper
                })

        self.outlier_table.setRowCount(len(outlier_data))
        for i, row in enumerate(outlier_data):
            self.outlier_table.setItem(i, 0, QTableWidgetItem(row["변수"]))
            self.outlier_table.setItem(i, 1, QTableWidgetItem(str(row["이상치 수"])))
            self.outlier_table.setItem(i, 2, QTableWidgetItem(f"{row['비율']:.2f}%"))
            self.outlier_table.setItem(i, 3, QTableWidgetItem(f"{row['IQR 하한']:.2f}"))
            self.outlier_table.setItem(i, 4, QTableWidgetItem(f"{row['IQR 상한']:.2f}"))

        self.outlier_table.resizeColumnsToContents()

        # 중복
        dup_text = f"중복 행 수: {duplicates}\n\n"
        if duplicates > 0:
            dup_examples = df[df.duplicated(keep=False)].head(10)
            dup_text += "중복 행 예시:\n"
            dup_text += dup_examples.to_string()

        self.duplicate_text.setText(dup_text)

    def _generate_report(self) -> None:
        """품질 보고서 생성."""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "품질 보고서 저장", "data_quality_report.html", "HTML (*.html)"
        )
        if path:
            html = self.engine.generate_data_quality_report(self.dataset)
            self.engine.save_html(html, path)
            self.quality_report_generated.emit(path)
            QMessageBox.information(self, "완료", f"보고서가 저장되었습니다.\n{path}")
