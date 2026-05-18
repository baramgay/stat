"""Descriptives Dialog — SPSS 스타일 기술통계 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QListWidget, QListWidgetItem,
    QCheckBox, QDialogButtonBox
)
from PySide6.QtCore import Signal

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult


class DescriptivesDialog(QDialog):
    """SPSS Descriptives 다이얼로그."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("기술통계량")
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 변수 목록
        var_group = QGroupBox("변수")
        var_layout = QVBoxLayout(var_group)

        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QListWidget.ExtendedSelection)

        import pandas as pd
        for var in self._dataset.data.columns:
            if pd.api.types.is_numeric_dtype(self._dataset.data[var]):
                item = QListWidgetItem(var)
                self.var_list.addItem(item)

        var_layout.addWidget(self.var_list)
        layout.addWidget(var_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_mean = QCheckBox("평균")
        self.chk_mean.setChecked(True)
        opt_layout.addWidget(self.chk_mean)

        self.chk_std = QCheckBox("표준편차")
        self.chk_std.setChecked(True)
        opt_layout.addWidget(self.chk_std)

        self.chk_minmax = QCheckBox("최소/최대")
        self.chk_minmax.setChecked(True)
        opt_layout.addWidget(self.chk_minmax)

        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        selected = [item.text() for item in self.var_list.selectedItems()]
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        try:
            from statworkbench.analysis.descriptive import run_descriptives
            result = run_descriptives(self._dataset.data, selected)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
