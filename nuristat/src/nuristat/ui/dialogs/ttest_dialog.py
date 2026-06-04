"""T-Test Dialog — SPSS 스타일 T 검정 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


def _combo_add(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    """콤보박스에 변수 추가 — 아이콘 + 라벨(변수명) 형식."""
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        text = f"{icon} {label}" if icon else label
        combo.addItem(text, userData=var)


class IndependentTTestDialog(QDialog):
    """SPSS Independent-Samples T Test 다이얼로그.

    검정 변수: 척도(Scale) 변수만
    그룹 변수: 명목/순서형 우선, 없으면 전체 변수
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("독립표본 T 검정")
        self.setMinimumSize(500, 380)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 검정 변수 (척도)
        test_group = QGroupBox("검정 변수 (Test Variable) — 척도형")
        test_layout = QVBoxLayout(test_group)

        self.test_combo = QComboBox()
        test_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        _combo_add(self.test_combo, self._dataset, test_vars)
        test_layout.addWidget(self.test_combo)
        layout.addWidget(test_group)

        # 그룹 변수 (명목/순서형 우선)
        group_group = QGroupBox("그룹 변수 (Grouping Variable) — 명목/순서형")
        group_layout = QVBoxLayout(group_group)

        self.group_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset)
        group_vars = cat_vars if cat_vars else all_vars(self._dataset)
        _combo_add(self.group_combo, self._dataset, group_vars)
        group_layout.addWidget(self.group_combo)
        layout.addWidget(group_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QHBoxLayout(opt_group)
        opt_layout.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        opt_layout.addWidget(self.ci_spin)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        test_var = self.test_combo.currentData() or var_from_display(self.test_combo.currentText())
        group_var = self.group_combo.currentData() or var_from_display(self.group_combo.currentText())

        if not test_var or not group_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        if test_var == group_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "검정 변수와 그룹 변수가 같을 수 없습니다.")
            return

        try:
            from nuristat.analysis.ttests import run_analysis
            spec = {
                "variables": {"dependent": test_var, "group": group_var},
                "options": {"equal_var": "auto"},
                "confidence_level": self.ci_spin.value(),
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")


class PairedTTestDialog(QDialog):
    """SPSS Paired-Samples T Test 다이얼로그.

    두 변수 모두 척도(Scale) 변수여야 함.
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("대응표본 T 검정")
        self.setMinimumSize(500, 360)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        pair_group = QGroupBox("변수 쌍 (Paired Variables) — 척도형")
        pair_layout = QVBoxLayout(pair_group)

        paired_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)

        pair_layout.addWidget(QLabel("변수 1 (Variable 1):"))
        self.var1_combo = QComboBox()
        _combo_add(self.var1_combo, self._dataset, paired_vars)
        pair_layout.addWidget(self.var1_combo)

        pair_layout.addWidget(QLabel("변수 2 (Variable 2):"))
        self.var2_combo = QComboBox()
        _combo_add(self.var2_combo, self._dataset, paired_vars)
        if len(paired_vars) > 1:
            self.var2_combo.setCurrentIndex(1)
        pair_layout.addWidget(self.var2_combo)

        layout.addWidget(pair_group)

        # 신뢰수준
        opt_group = QGroupBox("옵션")
        opt_layout = QHBoxLayout(opt_group)
        opt_layout.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        opt_layout.addWidget(self.ci_spin)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        var1 = self.var1_combo.currentData() or var_from_display(self.var1_combo.currentText())
        var2 = self.var2_combo.currentData() or var_from_display(self.var2_combo.currentText())

        if not var1 or not var2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        if var1 == var2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "두 변수가 같을 수 없습니다.")
            return

        try:
            from nuristat.analysis.ttests import run_analysis
            spec = {
                "variables": {"paired": [var1, var2]},
                "confidence_level": self.ci_spin.value(),
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
