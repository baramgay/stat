"""신뢰도 분석 다이얼로그 — SPSS Analyze > Scale > Reliability Analysis."""
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
from nuristat.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
)


class ReliabilityDialog(QDialog):
    """SPSS Reliability Analysis (Cronbach Alpha) 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("신뢰도 분석 (Cronbach α)")
        self.setMinimumSize(500, 420)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _vars = numeric_vars(self._dataset) or list(self._dataset.data.columns)

        item_group = QGroupBox("항목 변수 (두 개 이상 선택)")
        item_layout = QVBoxLayout(item_group)
        item_layout.addWidget(QLabel("척도를 구성하는 항목 변수를 선택하세요."))
        self.item_list = QListWidget()
        self.item_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.item_list.addItem(item)
        item_layout.addWidget(self.item_list)
        layout.addWidget(item_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        items = [item.data(0x0100) for item in self.item_list.selectedItems()]
        if len(items) < 2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "항목 변수를 두 개 이상 선택하세요.")
            return

        spec = {"variables": {"items": items}}
        try:
            from nuristat.analysis.reliability import run_analysis
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
