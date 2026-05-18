"""One Sample T-Test Dialog — 단일표본 T 검정 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QDialogButtonBox, QDoubleSpinBox, QMessageBox
)
from PySide6.QtCore import Signal

from statworkbench.core.dataset import Dataset


class OneSampleTTestDialog(QDialog):
    """단일표본 T 검정 다이얼로그.

    검정 변수와 검정값(귀무가설 평균)을 선택합니다.
    """

    analysis_requested = Signal(str, dict)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("단일표본 T 검정")
        self.setMinimumSize(420, 280)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 검정 변수
        var_group = QGroupBox("검정 변수 (Test Variable)")
        var_layout = QVBoxLayout(var_group)

        self.var_combo = QComboBox()
        import pandas as pd
        for var in self._dataset.data.columns:
            if pd.api.types.is_numeric_dtype(self._dataset.data[var]):
                self.var_combo.addItem(var)
        var_layout.addWidget(self.var_combo)
        layout.addWidget(var_group)

        # 검정값
        test_group = QGroupBox("검정값 (Test Value / 귀무가설 평균)")
        test_layout = QHBoxLayout(test_group)

        test_layout.addWidget(QLabel("검정값:"))
        self.test_value_spin = QDoubleSpinBox()
        self.test_value_spin.setRange(-1e9, 1e9)
        self.test_value_spin.setValue(0.0)
        self.test_value_spin.setDecimals(4)
        test_layout.addWidget(self.test_value_spin)
        test_layout.addStretch()
        layout.addWidget(test_group)

        # 옵션
        option_group = QGroupBox("옵션")
        option_layout = QHBoxLayout(option_group)
        option_layout.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(80.0, 99.9)
        self.ci_spin.setValue(95.0)
        self.ci_spin.setSuffix("%")
        option_layout.addWidget(self.ci_spin)
        option_layout.addStretch()
        layout.addWidget(option_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_spec(self) -> dict:
        """분석 스펙 반환."""
        return {
            "analysis_id": "one_sample_ttest",
            "variable": self.var_combo.currentText(),
            "test_value": self.test_value_spin.value(),
            "confidence_level": self.ci_spin.value() / 100.0,
        }

    def _on_ok(self):
        var = self.var_combo.currentText()
        if not var:
            QMessageBox.warning(self, "경고", "검정 변수를 선택하세요.")
            return
        self.accept()
