"""T-Test Dialog — SPSS 스타일 T 검정 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QDialogButtonBox
)
from PySide6.QtCore import Signal

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult


class IndependentTTestDialog(QDialog):
    """SPSS Independent-Samples T Test 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("독립표본 T 검정")
        self.setMinimumSize(500, 350)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 검정 변수
        test_group = QGroupBox("검정 변수 (Test Variable)")
        test_layout = QVBoxLayout(test_group)

        self.test_combo = QComboBox()
        import pandas as pd
        for var in self._dataset.data.columns:
            if pd.api.types.is_numeric_dtype(self._dataset.data[var]):
                self.test_combo.addItem(var)
        test_layout.addWidget(self.test_combo)
        layout.addWidget(test_group)

        # 그룹 변수
        group_group = QGroupBox("그룹 변수 (Grouping Variable)")
        group_layout = QVBoxLayout(group_group)

        self.group_combo = QComboBox()
        for var in self._dataset.data.columns:
            self.group_combo.addItem(var)
        group_layout.addWidget(self.group_combo)
        layout.addWidget(group_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        test_var = self.test_combo.currentText()
        group_var = self.group_combo.currentText()

        if not test_var or not group_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        try:
            from statworkbench.analysis.ttests import run_independent_ttest
            result = run_independent_ttest(self._dataset.data, test_var, group_var)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")


class PairedTTestDialog(QDialog):
    """SPSS Paired-Samples T Test 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("대응표본 T 검정")
        self.setMinimumSize(500, 350)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 변수 쌍
        pair_group = QGroupBox("변수 쌍 (Paired Variables)")
        pair_layout = QVBoxLayout(pair_group)

        pair_layout.addWidget(QLabel("변수 1:"))
        self.var1_combo = QComboBox()
        import pandas as pd
        for var in self._dataset.data.columns:
            if pd.api.types.is_numeric_dtype(self._dataset.data[var]):
                self.var1_combo.addItem(var)
        pair_layout.addWidget(self.var1_combo)

        pair_layout.addWidget(QLabel("변수 2:"))
        self.var2_combo = QComboBox()
        for var in self._dataset.data.columns:
            if pd.api.types.is_numeric_dtype(self._dataset.data[var]):
                self.var2_combo.addItem(var)
        pair_layout.addWidget(self.var2_combo)

        layout.addWidget(pair_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        var1 = self.var1_combo.currentText()
        var2 = self.var2_combo.currentText()

        if not var1 or not var2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        try:
            from statworkbench.analysis.ttests import run_paired_ttest
            result = run_paired_ttest(self._dataset.data, var1, var2)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
