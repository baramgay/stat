"""Regression Dialog — SPSS 스타일 회귀분석 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
)


class RegressionDialog(QDialog):
    """SPSS Linear Regression 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("선형 회귀")
        self.setMinimumSize(600, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _vars = scale_vars(self._dataset) or numeric_vars(self._dataset)

        # 종속 변수 — 척도형 우선
        dep_group = QGroupBox("종속 변수 (Dependent) — 척도형")
        dep_layout = QVBoxLayout(dep_group)

        self.dep_combo = QComboBox()
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            self.dep_combo.addItem(text, userData=var)
        dep_layout.addWidget(self.dep_combo)
        layout.addWidget(dep_group)

        # 독립 변수 — 척도형 우선
        ind_group = QGroupBox("독립 변수 (Independent) — 척도형")
        ind_layout = QVBoxLayout(ind_group)

        self.ind_list = QListWidget()
        self.ind_list.setSelectionMode(QListWidget.ExtendedSelection)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.ind_list.addItem(item)

        ind_layout.addWidget(self.ind_list)
        layout.addWidget(ind_group)

        # 방법
        method_group = QGroupBox("방법")
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Enter", "Stepwise", "Forward", "Backward"])
        method_layout.addWidget(self.method_combo)
        layout.addWidget(method_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        dep_var = self.dep_combo.currentData() or self.dep_combo.currentText()
        ind_vars = [item.data(0x0100) or item.text() for item in self.ind_list.selectedItems()]

        if not dep_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "종속 변수를 선택하세요.")
            return

        if not ind_vars:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "독립 변수를 하나 이상 선택하세요.")
            return

        try:
            from statworkbench.analysis.regression import run_linear_regression
            result = run_linear_regression(self._dataset.data, dep_var, ind_vars)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
