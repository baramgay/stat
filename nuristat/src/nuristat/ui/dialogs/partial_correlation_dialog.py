"""편상관 분석 다이얼로그 — SPSS Analyze > Correlate > Partial."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
)


class PartialCorrelationDialog(QDialog, AnalysisDialogMixin):
    """SPSS Partial Correlation 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("편상관")
        self.setMinimumSize(560, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _vars = numeric_vars(self._dataset) or list(self._dataset.data.columns)

        var_group = QGroupBox("분석 변수 (두 개 이상 선택)")
        var_layout = QVBoxLayout(var_group)
        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.var_list.addItem(item)
        var_layout.addWidget(self.var_list)
        layout.addWidget(var_group)

        ctrl_group = QGroupBox("통제 변수 (Controlling for)")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.addWidget(QLabel("통제할 변수를 선택하세요 (선택 없으면 0차 상관)"))
        self.ctrl_list = QListWidget()
        self.ctrl_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.ctrl_list.addItem(item)
        ctrl_layout.addWidget(self.ctrl_list)
        layout.addWidget(ctrl_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        target = [item.data(0x0100) for item in self.var_list.selectedItems()]
        controlling = [item.data(0x0100) for item in self.ctrl_list.selectedItems()]

        if len(target) < 2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "분석 변수를 두 개 이상 선택하세요.")
            return

        spec = {
            "variables": {
                "target": target,
                "controlling": controlling,
            },
        }
        from nuristat.analysis.partial_correlation import run_analysis
        self._start_analysis(run_analysis, self._dataset, spec)
