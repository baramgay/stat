"""Pivot Table Dialog — 피벗 테이블 다이얼로그.

pandas pivot_table을 사용하여 교차분석표를 생성합니다.
"""

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from statworkbench.core.dataset import Dataset


class PivotDialog(QDialog):
    """피벗 테이블 다이얼로그."""

    pivot_created = Signal(object)  # pivot_table

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

        self.setWindowTitle("📊 피벗 테이블")
        self.setMinimumSize(700, 550)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 변수 선택
        vars_group = QGroupBox("🔢 변수 선택")
        vars_layout = QVBoxLayout(vars_group)

        # 행 변수
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("행 변수:"))
        self.row_combo = QComboBox()
        self.row_combo.addItems(self.dataset.data.columns)
        row_layout.addWidget(self.row_combo)
        vars_layout.addLayout(row_layout)

        # 열 변수
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("열 변수:"))
        self.col_combo = QComboBox()
        self.col_combo.addItem("(없음)")
        self.col_combo.addItems(self.dataset.data.columns)
        col_layout.addWidget(self.col_combo)
        vars_layout.addLayout(col_layout)

        # 값 변수
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("값 변수:"))
        self.val_combo = QComboBox()
        self.val_combo.addItems(self.dataset.data.columns)
        val_layout.addWidget(self.val_combo)
        vars_layout.addLayout(val_layout)

        layout.addWidget(vars_group)

        # 집계 함수
        agg_group = QGroupBox("⚙️ 집계 함수")
        agg_layout = QHBoxLayout(agg_group)

        self.agg_group = QButtonGroup(self)

        self.count_radio = QRadioButton("빈도 (Count)")
        self.count_radio.setChecked(True)
        self.agg_group.addButton(self.count_radio)
        agg_layout.addWidget(self.count_radio)

        self.sum_radio = QRadioButton("합계 (Sum)")
        self.agg_group.addButton(self.sum_radio)
        agg_layout.addWidget(self.sum_radio)

        self.mean_radio = QRadioButton("평균 (Mean)")
        self.agg_group.addButton(self.mean_radio)
        agg_layout.addWidget(self.mean_radio)

        self.max_radio = QRadioButton("최대 (Max)")
        self.agg_group.addButton(self.max_radio)
        agg_layout.addWidget(self.max_radio)

        self.min_radio = QRadioButton("최소 (Min)")
        self.agg_group.addButton(self.min_radio)
        agg_layout.addWidget(self.min_radio)

        agg_layout.addStretch()
        layout.addWidget(agg_group)

        # 결과
        result_group = QGroupBox("📊 결과")
        result_layout = QVBoxLayout(result_group)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(1)
        self.result_table.setHorizontalHeaderLabels(["결과"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.result_table)

        layout.addWidget(result_group)

        # 버튼
        btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("▶ 생성")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self._generate_pivot)
        btn_layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("💾 납비")
        self.btn_export.clicked.connect(self._export_pivot)
        self.btn_export.setEnabled(False)
        btn_layout.addWidget(self.btn_export)

        self.btn_close = QPushButton("❌ 닫기")
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._current_pivot = None

    def _generate_pivot(self) -> None:
        """피벗 테이블 생성."""
        row_var = self.row_combo.currentText()
        col_var = self.col_combo.currentText()
        val_var = self.val_combo.currentText()

        if col_var == "(없음)":
            col_var = None

        # 집계 함수
        if self.count_radio.isChecked():
            aggfunc = "count"
        elif self.sum_radio.isChecked():
            aggfunc = "sum"
        elif self.mean_radio.isChecked():
            aggfunc = "mean"
        elif self.max_radio.isChecked():
            aggfunc = "max"
        else:
            aggfunc = "min"

        try:
            df = self.dataset.data

            # 피벗 테이블 생성
            pivot = pd.pivot_table(
                df,
                values=val_var if not self.count_radio.isChecked() else None,
                index=row_var,
                columns=col_var,
                aggfunc=aggfunc,
                fill_value=0,
                margins=True,
                margins_name="합계"
            )

            self._current_pivot = pivot

            # 테이블 표시
            rows = len(pivot.index)
            cols = len(pivot.columns)

            self.result_table.setRowCount(rows)
            self.result_table.setColumnCount(cols + 1)

            # 헤더
            headers = [str(pivot.index.name or "Index")]
            if isinstance(pivot.columns, pd.MultiIndex):
                headers.extend([str(c) for c in pivot.columns])
            else:
                headers.extend([str(c) for c in pivot.columns])

            self.result_table.setHorizontalHeaderLabels(headers)

            # 데이터
            for i, idx in enumerate(pivot.index):
                self.result_table.setItem(i, 0, QTableWidgetItem(str(idx)))
                for j, col in enumerate(pivot.columns):
                    val = pivot.loc[idx, col]
                    item = QTableWidgetItem(f"{val:,.0f}" if isinstance(val, (int, float)) else str(val))

                    # 합계 행/열 강조
                    if str(idx) == "합계" or str(col) == "합계":
                        item.setBackground(Qt.lightGray)
                        item.setFont(item.font())
                        item.font().setBold(True)

                    self.result_table.setItem(i, j + 1, item)

            self.result_table.resizeColumnsToContents()
            self.btn_export.setEnabled(True)

        except Exception as exc:
            QMessageBox.critical(self, "오류", f"피벗 테이블 생성 실패:\n{exc}")

    def _export_pivot(self) -> None:
        """피벗 테이블 납비."""
        if self._current_pivot is None:
            return

        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "피벗 테이블 저장", "pivot_table.csv", "CSV (*.csv)"
        )
        if path:
            self._current_pivot.to_csv(path)
            QMessageBox.information(self, "완료", f"저장되었습니다:\n{path}")
