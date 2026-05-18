"""Correlation Dialog — 상관분석 다이얼로그.

Pearson, Spearman, Kendall 상관계수를 계산합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit,
    QListWidget, QAbstractItemView
)
from PySide6.QtCore import Signal, Qt
from typing import Optional, List

import pandas as pd
import numpy as np
from scipy import stats

from statworkbench.core.dataset import Dataset


class CorrelationDialog(QDialog):
    """상관분석 다이얼로그."""
    
    analysis_completed = Signal(dict)
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        
        self.setWindowTitle("🔗 상관분석")
        self.setMinimumSize(600, 500)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 상관계수 유형
        type_group = QGroupBox("📈 상관계수 유형")
        type_layout = QHBoxLayout(type_group)
        
        self.type_group = QButtonGroup(self)
        
        self.pearson_radio = QRadioButton("Pearson (적률)")
        self.pearson_radio.setChecked(True)
        self.type_group.addButton(self.pearson_radio)
        type_layout.addWidget(self.pearson_radio)
        
        self.spearman_radio = QRadioButton("Spearman (순위)")
        self.type_group.addButton(self.spearman_radio)
        type_layout.addWidget(self.spearman_radio)
        
        self.kendall_radio = QRadioButton("Kendall's tau")
        self.type_group.addButton(self.kendall_radio)
        type_layout.addWidget(self.kendall_radio)
        
        type_layout.addStretch()
        layout.addWidget(type_group)
        
        # 변수 선택
        vars_group = QGroupBox("🔢 변수 선택 (숫자형)")
        vars_layout = QHBoxLayout(vars_group)
        
        # 사용 가능한 변수
        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        numeric_cols = list(self.dataset.data.select_dtypes(include=[np.number]).columns)
        self.avail_list.addItems(numeric_cols)
        self.avail_list.setSelectionMode(QAbstractItemView.MultiSelection)
        avail_layout.addWidget(self.avail_list)
        
        # 버튼
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.btn_add = QPushButton("▶")
        self.btn_add.clicked.connect(self._add_variables)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton("◀")
        self.btn_remove.clicked.connect(self._remove_variables)
        btn_layout.addWidget(self.btn_remove)
        
        self.btn_add_all = QPushButton("▶▶")
        self.btn_add_all.clicked.connect(self._add_all_variables)
        btn_layout.addWidget(self.btn_add_all)
        
        self.btn_remove_all = QPushButton("◀◀")
        self.btn_remove_all.clicked.connect(self._remove_all_variables)
        btn_layout.addWidget(self.btn_remove_all)
        btn_layout.addStretch()
        
        # 선택된 변수
        selected_layout = QVBoxLayout()
        selected_layout.addWidget(QLabel("선택된 변수:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.MultiSelection)
        selected_layout.addWidget(self.selected_list)
        
        vars_layout.addLayout(avail_layout, 2)
        vars_layout.addLayout(btn_layout)
        vars_layout.addLayout(selected_layout, 2)
        
        layout.addWidget(vars_group)
        
        # 옵션
        options_group = QGroupBox("⚙️ 옵션")
        options_layout = QHBoxLayout(options_group)
        
        self.sig_check = QRadioButton("유의성 표시 (* p<0.05, ** p<0.01)")
        self.sig_check.setChecked(True)
        options_layout.addWidget(self.sig_check)
        
        self.full_check = QRadioButton("모든 값 표시")
        self.full_check.setChecked(False)
        options_layout.addWidget(self.full_check)
        
        options_layout.addStretch()
        layout.addWidget(options_group)
        
        # 결과
        result_group = QGroupBox("📊 결과")
        result_layout = QVBoxLayout(result_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(1)
        self.result_table.setHorizontalHeaderLabels(["변수"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.result_table)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        self.result_text.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas; font-size: 11px;"
        )
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("▶ 분석 실행")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        action_layout.addWidget(self.btn_run)
        
        self.btn_close = QPushButton("❌ 닫기")
        self.btn_close.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_close)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
    
    def _add_variables(self) -> None:
        """선택한 변수 추가."""
        for item in self.avail_list.selectedItems():
            self.selected_list.addItem(item.text())
            self.avail_list.takeItem(self.avail_list.row(item))
    
    def _remove_variables(self) -> None:
        """선택한 변수 제거."""
        for item in self.selected_list.selectedItems():
            self.avail_list.addItem(item.text())
            self.selected_list.takeItem(self.selected_list.row(item))
    
    def _add_all_variables(self) -> None:
        """모든 변수 추가."""
        while self.avail_list.count() > 0:
            item = self.avail_list.takeItem(0)
            self.selected_list.addItem(item.text())
    
    def _remove_all_variables(self) -> None:
        """모든 변수 제거."""
        while self.selected_list.count() > 0:
            item = self.selected_list.takeItem(0)
            self.avail_list.addItem(item.text())
    
    def _run_analysis(self) -> None:
        """상관분석 실행."""
        variables = []
        for i in range(self.selected_list.count()):
            variables.append(self.selected_list.item(i).text())
        
        if len(variables) < 2:
            QMessageBox.warning(self, "경고", "2개 이상의 변수를 선택하세요")
            return
        
        # 상관계수 유형
        if self.pearson_radio.isChecked():
            method = "pearson"
        elif self.spearman_radio.isChecked():
            method = "spearman"
        else:
            method = "kendall"
        
        df = self.dataset.data[variables].dropna()
        
        try:
            # 상관행렬 계산
            corr_matrix = df.corr(method=method)
            
            # p-value 계산
            pvalue_matrix = pd.DataFrame(np.ones((len(variables), len(variables))),
                                        index=variables, columns=variables)
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        if method == "pearson":
                            _, pval = stats.pearsonr(df[var1], df[var2])
                        elif method == "spearman":
                            _, pval = stats.spearmanr(df[var1], df[var2])
                        else:
                            _, pval = stats.kendalltau(df[var1], df[var2])
                        pvalue_matrix.loc[var1, var2] = pval
            
            # 결과 테이블 업데이트
            n_vars = len(variables)
            self.result_table.setColumnCount(n_vars + 1)
            self.result_table.setHorizontalHeaderLabels(["변수"] + variables)
            self.result_table.setRowCount(n_vars)
            
            for i, var in enumerate(variables):
                self.result_table.setItem(i, 0, QTableWidgetItem(var))
                for j, var2 in enumerate(variables):
                    if i == j:
                        item = QTableWidgetItem("1.000")
                        item.setBackground(Qt.lightGray)
                    else:
                        corr = corr_matrix.loc[var, var2]
                        pval = pvalue_matrix.loc[var, var2]
                        
                        if self.sig_check.isChecked():
                            sig = ""
                            if pval < 0.01:
                                sig = "**"
                            elif pval < 0.05:
                                sig = "*"
                            text = f"{corr:.3f}{sig}"
                        else:
                            text = f"{corr:.3f} (p={pval:.3f})"
                        
                        item = QTableWidgetItem(text)
                        
                        # 색상 강조
                        if abs(corr) > 0.7:
                            item.setBackground(Qt.red)
                            item.setForeground(Qt.white)
                        elif abs(corr) > 0.5:
                            item.setBackground(Qt.yellow)
                    
                    self.result_table.setItem(i, j + 1, item)
            
            self.result_table.resizeColumnsToContents()
            
            # 결과 텍스트
            result_lines = []
            result_lines.append("=" * 60)
            result_lines.append(f"상관분석 ({method.capitalize()})")
            result_lines.append("=" * 60)
            result_lines.append(f"변수 수: {n_vars}")
            result_lines.append(f"유효 케이스: {len(df)}")
            result_lines.append("")
            result_lines.append("* p < 0.05, ** p < 0.01")
            
            self.result_text.setText("\n".join(result_lines))
            
            # 시그널 발생
            self.analysis_completed.emit({
                "type": "correlation",
                "method": method,
                "variables": variables,
                "correlation_matrix": corr_matrix.to_dict(),
            })
            
        except Exception as exc:
            self.result_text.setText(f"[오류]\n{exc}")
