"""Visual Binning Dialog — SPSS 스타일 시각적 구간화 다이얼로그.

변수를 구간으로 나누어 새 범주형 변수를 생성합니다.
"""

import numpy as np
import pandas as pd
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from statworkbench.core.dataset import Dataset


class BinningDialog(QDialog):
    """시각적 구간화 다이얼로그."""

    binning_applied = Signal(str, str, list, list)  # source, target, cut_points, labels

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

        self.setWindowTitle("📊 시각적 구간화")
        self.setMinimumSize(600, 550)
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
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(80)
        self.stats_text.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 11px;"
        )
        source_layout.addWidget(self.stats_text)

        layout.addWidget(source_group)

        # 대상 변수
        target_group = QGroupBox("📤 대상 변수")
        target_layout = QHBoxLayout(target_group)

        target_layout.addWidget(QLabel("새 변수명:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("예: age_group")
        target_layout.addWidget(self.target_edit)

        layout.addWidget(target_group)

        # 구간화 방법
        method_group = QGroupBox("⚙️ 구간화 방법")
        method_layout = QVBoxLayout(method_group)

        self.method_group = QButtonGroup(self)

        # 동일 너비
        self.equal_width_radio = QRadioButton("동일 너비 구간 (Equal Width)")
        self.equal_width_radio.setChecked(True)
        self.method_group.addButton(self.equal_width_radio)
        method_layout.addWidget(self.equal_width_radio)

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("구간 수:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 50)
        self.bins_spin.setValue(5)
        width_layout.addWidget(self.bins_spin)
        width_layout.addStretch()
        method_layout.addLayout(width_layout)

        # 동일 빈도
        self.equal_freq_radio = QRadioButton("동일 빈도 구간 (Equal Frequency / Quantile)")
        self.method_group.addButton(self.equal_freq_radio)
        method_layout.addWidget(self.equal_freq_radio)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("구간 수:"))
        self.freq_bins_spin = QSpinBox()
        self.freq_bins_spin.setRange(2, 50)
        self.freq_bins_spin.setValue(4)
        freq_layout.addWidget(self.freq_bins_spin)
        freq_layout.addStretch()
        method_layout.addLayout(freq_layout)

        # 수동 구간
        self.manual_radio = QRadioButton("수동 구간 지정")
        self.method_group.addButton(self.manual_radio)
        method_layout.addWidget(self.manual_radio)

        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("구간 경계값 (쉼표 구분):"))
        self.manual_edit = QLineEdit()
        self.manual_edit.setPlaceholderText("예: 0, 18, 35, 50, 65, 100")
        manual_layout.addWidget(self.manual_edit)
        method_layout.addLayout(manual_layout)

        layout.addWidget(method_group)

        # 미리보기
        preview_group = QGroupBox("👁️ 미리보기")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(4)
        self.preview_table.setHorizontalHeaderLabels(["구간", "범위", "빈도", "비율"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        preview_layout.addWidget(self.preview_table)

        self.btn_preview = QPushButton("🔍 미리보기 생성")
        self.btn_preview.clicked.connect(self._generate_preview)
        preview_layout.addWidget(self.btn_preview)

        layout.addWidget(preview_group)

        # 실행 버튼
        action_layout = QHBoxLayout()

        self.btn_apply = QPushButton("✅ 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_binning)
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

            stats = []
            stats.append(f"N: {series.count()}")
            stats.append(f"결측: {series.isna().sum()}")
            stats.append(f"최소: {series.min():.2f}")
            stats.append(f"최대: {series.max():.2f}")
            stats.append(f"평균: {series.mean():.2f}")
            stats.append(f"중위: {series.median():.2f}")
            stats.append(f"표준편차: {series.std():.2f}")

            self.stats_text.setText(" | ".join(stats))

            # 기본 대상 변수명 제안
            self.target_edit.setText(f"{var_name}_group")

    def _generate_preview(self) -> None:
        """미리보기 생성."""
        var_name = self.source_combo.currentText()
        if not var_name:
            return

        series = self.dataset.data[var_name].dropna()

        try:
            if self.equal_width_radio.isChecked():
                n_bins = self.bins_spin.value()
                cut_points = list(np.linspace(series.min(), series.max(), n_bins + 1))
                labels = [f"{cut_points[i]:.1f} - {cut_points[i+1]:.1f}" for i in range(n_bins)]

            elif self.equal_freq_radio.isChecked():
                n_bins = self.freq_bins_spin.value()
                cut_points = list(series.quantile(np.linspace(0, 1, n_bins + 1)))
                labels = [f"Q{i+1}" for i in range(n_bins)]

            else:  # manual
                manual_text = self.manual_edit.text().strip()
                if not manual_text:
                    QMessageBox.warning(self, "경고", "구간 경계값을 입력하세요")
                    return

                cut_points = [float(x.strip()) for x in manual_text.split(",")]
                n_bins = len(cut_points) - 1
                labels = [f"Bin {i+1}" for i in range(n_bins)]

            # 빈도 계산
            binned = pd.cut(series, bins=cut_points, labels=labels, include_lowest=True)
            freq_counts = binned.value_counts().sort_index()

            # 테이블 업데이트
            self.preview_table.setRowCount(len(freq_counts))
            total = len(series)

            for i, (label, count) in enumerate(freq_counts.items()):
                self.preview_table.setItem(i, 0, QTableWidgetItem(str(label)))
                self.preview_table.setItem(i, 1, QTableWidgetItem(
                    f"{cut_points[i]:.1f} - {cut_points[i+1]:.1f}"
                ))
                self.preview_table.setItem(i, 2, QTableWidgetItem(str(count)))
                self.preview_table.setItem(i, 3, QTableWidgetItem(f"{count/total:.1%}"))

        except Exception as exc:
            QMessageBox.warning(self, "오류", f"미리보기 생성 실패:\n{exc}")

    def _apply_binning(self) -> None:
        """구간화 적용."""
        source_var = self.source_combo.currentText()
        target_var = self.target_edit.text().strip()

        if not target_var:
            QMessageBox.warning(self, "경고", "새 변수명을 입력하세요")
            return

        series = self.dataset.data[source_var].dropna()

        try:
            if self.equal_width_radio.isChecked():
                n_bins = self.bins_spin.value()
                cut_points = list(np.linspace(series.min(), series.max(), n_bins + 1))
                labels = [f"{cut_points[i]:.1f}-{cut_points[i+1]:.1f}" for i in range(n_bins)]

            elif self.equal_freq_radio.isChecked():
                n_bins = self.freq_bins_spin.value()
                cut_points = list(series.quantile(np.linspace(0, 1, n_bins + 1)))
                labels = [f"Q{i+1}" for i in range(n_bins)]

            else:  # manual
                manual_text = self.manual_edit.text().strip()
                cut_points = [float(x.strip()) for x in manual_text.split(",")]
                n_bins = len(cut_points) - 1
                labels = [f"Bin{i+1}" for i in range(n_bins)]

            # 시그널 발생
            self.binning_applied.emit(source_var, target_var, cut_points, labels)

            QMessageBox.information(self, "완료", f"구간화가 적용되었습니다.\n{target_var} 변수가 생성되었습니다.")
            self.accept()

        except Exception as exc:
            QMessageBox.critical(self, "오류", f"구간화 적용 실패:\n{exc}")
