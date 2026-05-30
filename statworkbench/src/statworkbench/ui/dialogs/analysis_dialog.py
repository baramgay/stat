"""Analysis Dialog — SPSS 스타일 분석 다이얼로그 베이스 클래스.

모든 분석 다이얼로그의 공통 기능을 제공합니다.
"""


import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset


class AnalysisDialog(QDialog):
    """분석 다이얼로그 베이스 클래스."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, title: str, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self._title = title
        self.setWindowTitle(title)
        self.setMinimumSize(600, 500)
        self._setup_ui()

    def _setup_ui(self):
        """기본 UI 구성."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 데이터셋 정보
        info = QLabel(f"데이터셋: {self._dataset.name} | "
                      f"케이스: {len(self._dataset.data)} | "
                      f"변수: {len(self._dataset.data.columns)}")
        info.setStyleSheet("color: #5d6d7e; font-size: 12px;")
        layout.addWidget(info)

        # 메인 영역
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_widget)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run_analysis)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _get_variables(self, numeric_only: bool = False) -> list[str]:
        """변수 목록 반환."""
        df = self._dataset.data
        if numeric_only:
            return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return list(df.columns)

    def _run_analysis(self):
        """분석 실행 — 서브클스에서 오버라이드."""
        pass


class VariableSelectorDialog(AnalysisDialog):
    """변수 선택 다이얼로그."""

    def __init__(self, title: str, dataset: Dataset,
                 numeric_only: bool = False, multi_select: bool = True,
                 parent=None):
        self._numeric_only = numeric_only
        self._multi_select = multi_select
        super().__init__(title, dataset, parent)

    def _setup_ui(self):
        super()._setup_ui()

        # 변수 목록
        var_group = QGroupBox("변수 선택")
        var_layout = QVBoxLayout(var_group)

        self.var_list = QListWidget()
        if self._multi_select:
            self.var_list.setSelectionMode(QListWidget.ExtendedSelection)
        else:
            self.var_list.setSelectionMode(QListWidget.SingleSelection)

        for var in self._get_variables(self._numeric_only):
            item = QListWidgetItem(var)
            self.var_list.addItem(item)

        var_layout.addWidget(self.var_list)
        self.main_layout.addWidget(var_group)

    def get_selected_variables(self) -> list[str]:
        """선택된 변수 목록 반환."""
        return [item.text() for item in self.var_list.selectedItems()]

    def _run_analysis(self):
        """분석 실행."""
        selected = self.get_selected_variables()
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return
        self.accept()
