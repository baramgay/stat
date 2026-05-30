"""일치도 분석 다이얼로그 — Cohen's Kappa / ICC / Bland-Altman."""
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
    all_vars,
    display_label,
    measure_icon,
    numeric_vars,
)


class KappaDialog(QDialog):
    """Cohen's Kappa 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("Cohen's Kappa")
        self.setMinimumSize(480, 360)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _all = all_vars(self._dataset)

        def _build_combo():
            combo = QComboBox()
            for var in _all:
                icon = measure_icon(self._dataset, var)
                label = display_label(self._dataset, var)
                text = f"{icon} {label}" if icon else label
                combo.addItem(text, userData=var)
            return combo

        r1_group = QGroupBox("평가자 1 변수")
        r1_lay = QVBoxLayout(r1_group)
        self.r1_combo = _build_combo()
        r1_lay.addWidget(self.r1_combo)
        layout.addWidget(r1_group)

        r2_group = QGroupBox("평가자 2 변수")
        r2_lay = QVBoxLayout(r2_group)
        self.r2_combo = _build_combo()
        r2_lay.addWidget(self.r2_combo)
        layout.addWidget(r2_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        r1 = self.r1_combo.currentData()
        r2 = self.r2_combo.currentData()
        if r1 == r2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "서로 다른 변수를 선택하세요.")
            return
        spec = {"variables": {"rater1": r1, "rater2": r2}}
        try:
            from statworkbench.analysis.cohens_kappa import run_analysis
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")


class ICCDialog(QDialog):
    """급내 상관계수(ICC) 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("급내 상관계수(ICC)")
        self.setMinimumSize(520, 440)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _num = numeric_vars(self._dataset) or list(self._dataset.data.columns)

        meas_group = QGroupBox("측정 변수 (두 개 이상 선택)")
        meas_layout = QVBoxLayout(meas_group)
        self.meas_list = QListWidget()
        self.meas_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in _num:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.meas_list.addItem(item)
        meas_layout.addWidget(self.meas_list)
        layout.addWidget(meas_group)

        model_group = QGroupBox("모형")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItem("일원 배치 (One-Way Random)", userData="one_way")
        self.model_combo.addItem("이원 혼합 (Two-Way Mixed)", userData="two_way_mixed")
        self.model_combo.addItem("이원 무선 (Two-Way Random)", userData="two_way_random")
        model_layout.addWidget(self.model_combo)
        layout.addWidget(model_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        measurements = [item.data(0x0100) for item in self.meas_list.selectedItems()]
        if len(measurements) < 2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "측정 변수를 두 개 이상 선택하세요.")
            return
        spec = {
            "variables": {"measurements": measurements},
            "options": {"model": self.model_combo.currentData()},
        }
        try:
            from statworkbench.analysis.icc import run_analysis
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")


class BlandAltmanDialog(QDialog):
    """Bland-Altman 분석 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("Bland-Altman 분석")
        self.setMinimumSize(480, 360)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        _num = numeric_vars(self._dataset) or list(self._dataset.data.columns)

        def _build_combo():
            combo = QComboBox()
            for var in _num:
                icon = measure_icon(self._dataset, var)
                label = display_label(self._dataset, var)
                text = f"{icon} {label}" if icon else label
                combo.addItem(text, userData=var)
            return combo

        m1_group = QGroupBox("측정 방법 1")
        m1_lay = QVBoxLayout(m1_group)
        self.m1_combo = _build_combo()
        m1_lay.addWidget(self.m1_combo)
        layout.addWidget(m1_group)

        m2_group = QGroupBox("측정 방법 2")
        m2_lay = QVBoxLayout(m2_group)
        self.m2_combo = _build_combo()
        m2_lay.addWidget(self.m2_combo)
        layout.addWidget(m2_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        m1 = self.m1_combo.currentData()
        m2 = self.m2_combo.currentData()
        if m1 == m2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "서로 다른 변수를 선택하세요.")
            return
        spec = {"variables": {"method1": m1, "method2": m2}}
        try:
            from statworkbench.analysis.bland_altman import run_analysis
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
