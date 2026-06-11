"""탐색 다이얼로그 — SPSS Analyze > Descriptive Statistics > Explore."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    display_label,
    measure_icon,
    numeric_vars,
)


class ExploreDialog(QDialog, AnalysisDialogMixin):
    """SPSS Explore 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("탐색")
        self.setMinimumSize(520, 480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _num = numeric_vars(self._dataset) or list(self._dataset.data.columns)
        _all = all_vars(self._dataset)

        dep_group = QGroupBox("종속 변수 목록")
        dep_layout = QVBoxLayout(dep_group)
        self.dep_list = QListWidget()
        self.dep_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _num:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.dep_list.addItem(item)
        dep_layout.addWidget(self.dep_list)
        layout.addWidget(dep_group)

        factor_group = QGroupBox("요인 변수 (선택 사항)")
        factor_layout = QVBoxLayout(factor_group)
        self.factor_combo = QComboBox()
        self.factor_combo.addItem("없음", userData=None)
        for var in _all:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            self.factor_combo.addItem(text, userData=var)
        factor_layout.addWidget(self.factor_combo)
        layout.addWidget(factor_group)

        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)
        self.chk_stats = QCheckBox("기술통계량")
        self.chk_stats.setChecked(True)
        opt_layout.addWidget(self.chk_stats)
        self.chk_normality = QCheckBox("정규성 검정 (Shapiro-Wilk)")
        self.chk_normality.setChecked(True)
        opt_layout.addWidget(self.chk_normality)
        layout.addWidget(opt_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        dep_vars = [item.data(0x0100) for item in self.dep_list.selectedItems()]
        if not dep_vars:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "분석할 변수를 하나 이상 선택하세요.")
            return

        spec = {
            "variables": {
                "target": dep_vars,
                "factor": self.factor_combo.currentData(),
            },
            "options": {
                "statistics": self.chk_stats.isChecked(),
                "normality": self.chk_normality.isChecked(),
            },
        }
        from nuristat.analysis.explore import run_analysis
        self._start_analysis(run_analysis, self._dataset, spec)
