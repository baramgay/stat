"""ROC 분석 다이얼로그 — SPSS Analyze > ROC Curve."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    display_label,
    measure_icon,
    numeric_vars,
)


class ROCDialog(QDialog):
    """SPSS ROC Curve 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("ROC 분석")
        self.setMinimumSize(560, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _num = numeric_vars(self._dataset) or list(self._dataset.data.columns)
        _all = all_vars(self._dataset)

        state_group = QGroupBox("상태 변수 (실제 분류)")
        state_layout = QHBoxLayout(state_group)
        self.state_combo = QComboBox()
        for var in _all:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            self.state_combo.addItem(text, userData=var)
        state_layout.addWidget(self.state_combo)
        state_layout.addWidget(QLabel("양성값:"))
        self.positive_edit = QLineEdit("1")
        self.positive_edit.setMaximumWidth(80)
        state_layout.addWidget(self.positive_edit)
        layout.addWidget(state_group)

        test_group = QGroupBox("검사 변수 (예측 점수)")
        test_layout = QVBoxLayout(test_group)
        self.test_list = QListWidget()
        self.test_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _num:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.test_list.addItem(item)
        test_layout.addWidget(self.test_list)
        layout.addWidget(test_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        state_var = self.state_combo.currentData()
        test_vars = [item.data(0x0100) for item in self.test_list.selectedItems()]

        if not test_vars:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "검사 변수를 하나 이상 선택하세요.")
            return

        pos_text = self.positive_edit.text().strip()
        try:
            positive_value = int(pos_text)
        except ValueError:
            positive_value = pos_text

        spec = {
            "variables": {
                "state": state_var,
                "test_vars": test_vars,
                "positive_value": positive_value,
            },
        }
        try:
            from nuristat.analysis.roc_analysis import run_analysis
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
