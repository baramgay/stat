"""카이제곱 적합도 검정 다이얼로그 — SPSS Analyze > Nonparametric Tests."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    display_label,
    measure_icon,
)


class ChiSquareGOFDialog(QDialog, AnalysisDialogMixin):
    """카이제곱 적합도 검정 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("카이제곱 적합도 검정")
        self.setMinimumSize(500, 420)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _all = all_vars(self._dataset)

        var_group = QGroupBox("검정 변수")
        var_layout = QVBoxLayout(var_group)
        self.var_combo = QComboBox()
        for var in _all:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            self.var_combo.addItem(text, userData=var)
        var_layout.addWidget(self.var_combo)
        layout.addWidget(var_group)

        exp_group = QGroupBox("기대 빈도")
        exp_layout = QVBoxLayout(exp_group)

        self.rb_equal = QRadioButton("균등 분포 (동일 기대 빈도)")
        self.rb_equal.setChecked(True)
        exp_layout.addWidget(self.rb_equal)

        self.rb_custom = QRadioButton("사용자 정의 비율 (쉼표로 구분)")
        exp_layout.addWidget(self.rb_custom)

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.rb_equal)
        self.btn_group.addButton(self.rb_custom)

        self.ratio_edit = QTextEdit()
        self.ratio_edit.setPlaceholderText("예: 1,2,1  또는  0.25,0.5,0.25")
        self.ratio_edit.setMaximumHeight(60)
        self.ratio_edit.setEnabled(False)
        exp_layout.addWidget(self.ratio_edit)

        self.rb_equal.toggled.connect(lambda checked: self.ratio_edit.setEnabled(not checked))
        layout.addWidget(exp_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        target = self.var_combo.currentData()
        expected_ratios = None

        if self.rb_custom.isChecked():
            text = self.ratio_edit.toPlainText().strip()
            if text:
                try:
                    expected_ratios = [float(v.strip()) for v in text.split(",")]
                except ValueError:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "경고", "비율을 올바르게 입력하세요 (숫자, 쉼표 구분).")
                    return

        spec = {
            "variables": {"target": target},
            "options": {"expected_ratios": expected_ratios},
        }
        from nuristat.analysis.chi_square_gof import run_analysis
        self._start_analysis(run_analysis, self._dataset, spec)
