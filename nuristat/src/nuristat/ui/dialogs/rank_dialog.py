"""Rank Cases Dialog — SPSS 스타일 순위 부여 다이얼로그.

변수에 순위를 부여하여 새 변수를 생성합니다.
"""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from nuristat.core.dataset import Dataset


class RankDialog(QDialog):
    """순위 부여 다이얼로그."""

    rank_applied = Signal(str, str, str)  # source, target, method

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

        self.setWindowTitle("🏆 순위 부여")
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 원본 변수 선택
        source_group = QGroupBox("📥 원본 변수 (숫자형)")
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        numeric_cols = self.dataset.data.select_dtypes(include=[np.number]).columns
        self.source_combo.addItems(numeric_cols)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)

        # 통계 정보
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #3a5068; font-size: 11px;")
        source_layout.addWidget(self.stats_label)

        layout.addWidget(source_group)

        # 순위 방법
        method_group = QGroupBox("⚙️ 순위 방법")
        method_layout = QVBoxLayout(method_group)

        self.method_group = QButtonGroup(self)

        self.rank_radio = QRadioButton("표준 순위 (Rank)")
        self.rank_radio.setChecked(True)
        self.method_group.addButton(self.rank_radio)
        method_layout.addWidget(self.rank_radio)

        self.dense_radio = QRadioButton("밀집 순위 (Dense Rank)")
        self.method_group.addButton(self.dense_radio)
        method_layout.addWidget(self.dense_radio)

        self.min_radio = QRadioButton("최소 순위 (Min Rank)")
        self.method_group.addButton(self.min_radio)
        method_layout.addWidget(self.min_radio)

        self.max_radio = QRadioButton("최대 순위 (Max Rank)")
        self.method_group.addButton(self.max_radio)
        method_layout.addWidget(self.max_radio)

        self.first_radio = QRadioButton("첫 번째 순위 (First)")
        self.method_group.addButton(self.first_radio)
        method_layout.addWidget(self.first_radio)

        self.pct_radio = QRadioButton("백분위 순위 (Percentile)")
        self.method_group.addButton(self.pct_radio)
        method_layout.addWidget(self.pct_radio)

        layout.addWidget(method_group)

        # 대상 변수
        target_group = QGroupBox("📤 대상 변수")
        target_layout = QHBoxLayout(target_group)

        target_layout.addWidget(QLabel("새 변수명:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("예: age_rank")
        target_layout.addWidget(self.target_edit)

        layout.addWidget(target_group)

        # 실행 버튼
        action_layout = QHBoxLayout()

        self.btn_apply = QPushButton("✅ 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_rank)
        action_layout.addWidget(self.btn_apply)

        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        # 초기 통계 업데이트
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        """원본 변수 변경 시 통계 업데이트."""
        var_name = self.source_combo.currentText()
        if var_name and var_name in self.dataset.data.columns:
            series = self.dataset.data[var_name]

            stats_text = (
                f"N: {series.count()} | "
                f"결측: {series.isna().sum()} | "
                f"최소: {series.min():.2f} | "
                f"최대: {series.max():.2f}"
            )
            self.stats_label.setText(stats_text)

            # 기본 대상 변수명 제안
            self.target_edit.setText(f"{var_name}_rank")

    def _apply_rank(self) -> None:
        """순위 부여 적용."""
        source_var = self.source_combo.currentText()
        target_var = self.target_edit.text().strip()

        if not target_var:
            QMessageBox.warning(self, "경고", "새 변수명을 입력하세요")
            return

        if target_var in self.dataset.data.columns:
            reply = QMessageBox.question(
                self,
                "확인",
                f"'{target_var}' 변수가 이미 존재합니다. 덮어쓰시겠습니까?"
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 순위 방법 결정
        if self.rank_radio.isChecked():
            method = "average"
        elif self.dense_radio.isChecked():
            method = "dense"
        elif self.min_radio.isChecked():
            method = "min"
        elif self.max_radio.isChecked():
            method = "max"
        elif self.first_radio.isChecked():
            method = "first"
        elif self.pct_radio.isChecked():
            method = "pct"
        else:
            method = "average"

        self.rank_applied.emit(source_var, target_var, method)
        self.accept()
