"""Compute Variable Dialog — SPSS 스타일 변수 계산 다이얼로그.

SPSS의 Transform > Compute Variable 기능을 모방합니다.
"""

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)

from nuristat.core.dataset import Dataset


class ComputeVariableDialog(QDialog):
    """SPSS 스타일 변수 계산 다이얼로그."""

    computed = Signal(str, pd.Series)  # 변수명, 계산 결과

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("변수 계산")
        self.setMinimumSize(700, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 타겟 변수
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("타겟 변수:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("새 변수명 또는 기존 변수명")
        target_layout.addWidget(self.target_edit)
        layout.addLayout(target_layout)

        # 수식 입력
        layout.addWidget(QLabel("수식:"))
        self.formula_edit = QTextEdit()
        self.formula_edit.setPlaceholderText(
            "예: (Age - mean(Age)) / std(Age)\n"
            "예: log(Income)\n"
            "예: Gender == 'Male'"
        )
        self.formula_edit.setMaximumHeight(80)
        layout.addWidget(self.formula_edit)

        # 스플리터: 함수 목록 | 변수 목록
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 함수 그룹
        func_group = QGroupBox("함수")
        func_layout = QVBoxLayout(func_group)
        self.func_list = QListWidget()
        self._populate_functions()
        self.func_list.itemDoubleClicked.connect(self._insert_function)
        func_layout.addWidget(self.func_list)
        splitter.addWidget(func_group)

        # 변수 그룹
        var_group = QGroupBox("변수")
        var_layout = QVBoxLayout(var_group)
        self.var_list = QListWidget()
        self._populate_variables()
        self.var_list.itemDoubleClicked.connect(self._insert_variable)
        var_layout.addWidget(self.var_list)
        splitter.addWidget(var_group)

        layout.addWidget(splitter)

        # 계산기 버튼
        calc_group = QGroupBox("계산기")
        calc_layout = QGridLayout(calc_group)

        buttons = [
            ('+', 0, 0), ('-', 0, 1), ('*', 0, 2), ('/', 0, 3),
            ('**', 1, 0), ('(', 1, 1), (')', 1, 2), ('==', 1, 3),
            ('>', 2, 0), ('<', 2, 1), ('>=', 2, 2), ('<=', 2, 3),
            ('&', 3, 0), ('|', 3, 1), ('~', 3, 2), ('%', 3, 3),
        ]
        for text, row, col in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, t=text: self._insert_text(t))
            calc_layout.addWidget(btn, row, col)

        layout.addWidget(calc_group)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        ok_btn = QPushButton("확인")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._compute)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _populate_functions(self):
        """함수 목록 채우기."""
        functions = [
            "abs(x)", "sqrt(x)", "log(x)", "log10(x)", "exp(x)",
            "round(x, n)", "floor(x)", "ceil(x)",
            "mean(x)", "median(x)", "std(x)", "var(x)", "min(x)", "max(x)",
            "sum(x)", "count(x)", "cumsum(x)",
            "ifelse(condition, true_val, false_val)",
        ]
        for func in functions:
            item = QListWidgetItem(func)
            item.setToolTip(f"더블클릭하여 삽입: {func}")
            self.func_list.addItem(item)

    def _populate_variables(self):
        """변수 목록 채우기."""
        if self._dataset and self._dataset.data is not None:
            for col in self._dataset.data.columns:
                item = QListWidgetItem(col)
                item.setToolTip(f"더블클릭하여 삽입: {col}")
                self.var_list.addItem(item)

    def _insert_function(self, item: QListWidgetItem):
        """함수 삽입."""
        func = item.text()
        cursor = self.formula_edit.textCursor()
        cursor.insertText(func)
        self.formula_edit.setFocus()

    def _insert_variable(self, item: QListWidgetItem):
        """변수 삽입."""
        var = item.text()
        cursor = self.formula_edit.textCursor()
        cursor.insertText(var)
        self.formula_edit.setFocus()

    def _insert_text(self, text: str):
        """텍스트 삽입."""
        cursor = self.formula_edit.textCursor()
        cursor.insertText(f" {text} ")
        self.formula_edit.setFocus()

    def _compute(self):
        """변수 계산 실행."""
        target = self.target_edit.text().strip()
        formula = self.formula_edit.toPlainText().strip()

        if not target:
            QMessageBox.warning(self, "경고", "타겟 변수명을 입력하세요.")
            return

        if not formula:
            QMessageBox.warning(self, "경고", "수식을 입력하세요.")
            return

        try:
            df = self._dataset.data

            # 안전한 eval 환경 구성
            safe_dict = {
                'df': df,
                'pd': pd,
                'np': np,
                'abs': np.abs,
                'sqrt': np.sqrt,
                'log': np.log,
                'log10': np.log10,
                'exp': np.exp,
                'round': np.round,
                'floor': np.floor,
                'ceil': np.ceil,
                'mean': lambda x: x.mean(),
                'median': lambda x: x.median(),
                'std': lambda x: x.std(),
                'var': lambda x: x.var(),
                'min': lambda x: x.min(),
                'max': lambda x: x.max(),
                'sum': lambda x: x.sum(),
                'count': lambda x: x.count(),
                'cumsum': lambda x: x.cumsum(),
                'ifelse': lambda c, t, f: np.where(c, t, f),
            }

            # 변수를 DataFrame 컬럼으로 매핑
            for col in df.columns:
                safe_dict[col] = df[col]

            result = eval(formula, {"__builtins__": {}}, safe_dict)

            if isinstance(result, pd.Series):
                self.computed.emit(target, result)
                self.accept()
            else:
                QMessageBox.warning(self, "경고", "수식 결과가 Series가 아닙니다.")

        except Exception as exc:
            QMessageBox.critical(self, "오류", f"계산 실패:\n{exc}")
