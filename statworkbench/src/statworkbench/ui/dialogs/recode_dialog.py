"""Recode Dialog — SPSS 스타일 변수 재코딩 다이얼로그.

Recode into Same/Different Variables 기능을 제공합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from typing import Optional, List, Dict

from statworkbench.core.dataset import Dataset


class RecodeDialog(QDialog):
    """변수 재코딩 다이얼로그."""
    
    recode_applied = Signal(str, str, dict)  # source_var, target_var, recode_map
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self._recode_rules: List[Dict] = []
        
        self.setWindowTitle("🔄 변수 재코딩")
        self.setMinimumSize(600, 500)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 원본 변수 선택
        source_group = QGroupBox("📥 원본 변수")
        source_layout = QVBoxLayout(source_group)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.dataset.data.columns)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        
        # 원본 변수 통계
        self.source_stats = QLabel("")
        self.source_stats.setStyleSheet("color: #5d6d7e; font-size: 11px;")
        source_layout.addWidget(self.source_stats)
        
        layout.addWidget(source_group)
        
        # 출력 옵션
        output_group = QGroupBox("📤 출력 옵션")
        output_layout = QVBoxLayout(output_group)
        
        self.output_group = QButtonGroup(self)
        
        self.same_var_radio = QRadioButton("원본 변수에 재코딩 (Recode into Same Variables)")
        self.same_var_radio.setChecked(True)
        self.output_group.addButton(self.same_var_radio)
        output_layout.addWidget(self.same_var_radio)
        
        diff_layout = QHBoxLayout()
        self.diff_var_radio = QRadioButton("새 변수에 재코딩 (Recode into Different Variables):")
        self.output_group.addButton(self.diff_var_radio)
        diff_layout.addWidget(self.diff_var_radio)
        
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("새 변수명")
        self.target_edit.setEnabled(False)
        diff_layout.addWidget(self.target_edit)
        
        output_layout.addLayout(diff_layout)
        
        self.diff_var_radio.toggled.connect(self.target_edit.setEnabled)
        
        layout.addWidget(output_group)
        
        # 재코딩 규칙
        rules_group = QGroupBox("📋 재코딩 규칙")
        rules_layout = QVBoxLayout(rules_group)
        
        # 규칙 테이블
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(3)
        self.rules_table.setHorizontalHeaderLabels(["원본 값", "->", "새 값"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)
        self.rules_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rules_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.rules_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.rules_table.setColumnWidth(1, 40)
        rules_layout.addWidget(self.rules_table)
        
        # 규칙 추가/삭제 버튼
        btn_layout = QHBoxLayout()
        
        self.btn_add_rule = QPushButton("➕ 규칙 추가")
        self.btn_add_rule.clicked.connect(self._add_rule)
        btn_layout.addWidget(self.btn_add_rule)
        
        self.btn_del_rule = QPushButton("➖ 규칙 삭제")
        self.btn_del_rule.clicked.connect(self._delete_rule)
        btn_layout.addWidget(self.btn_del_rule)
        
        self.btn_add_range = QPushButton("📊 범위 추가")
        self.btn_add_range.clicked.connect(self._add_range_rule)
        btn_layout.addWidget(self.btn_add_range)
        
        btn_layout.addStretch()
        rules_layout.addLayout(btn_layout)
        
        layout.addWidget(rules_group)
        
        # 미리보기
        preview_group = QGroupBox("👁️ 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("재코딩 규칙을 추가하면 미리보기가 표시됩니다")
        self.preview_label.setStyleSheet("color: #7a7a8a; font-style: italic;")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("✅ 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_recode)
        action_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        # 초기 규칙 추가
        self._add_rule()
    
    def _on_source_changed(self) -> None:
        """원본 변수 변경 시."""
        var_name = self.source_combo.currentText()
        if var_name and var_name in self.dataset.data.columns:
            series = self.dataset.data[var_name]
            unique_count = series.nunique()
            na_count = series.isna().sum()
            
            stats_text = f"고유값: {unique_count}개 | 결측치: {na_count}개"
            
            # 값 미리보기 (최대 10개)
            if unique_count <= 20:
                values = series.dropna().unique()
                values_str = ", ".join([str(v) for v in values[:10]])
                if len(values) > 10:
                    values_str += f", ... ({len(values) - 10}개 더)"
                stats_text += f"\n값: {values_str}"
            
            self.source_stats.setText(stats_text)
    
    def _add_rule(self) -> None:
        """규칙 추가."""
        row = self.rules_table.rowCount()
        self.rules_table.insertRow(row)
        
        self.rules_table.setItem(row, 0, QTableWidgetItem(""))
        self.rules_table.setItem(row, 1, QTableWidgetItem("->"))
        self.rules_table.item(row, 1).setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.rules_table.setItem(row, 2, QTableWidgetItem(""))
    
    def _add_range_rule(self) -> None:
        """범위 규칙 추가."""
        row = self.rules_table.rowCount()
        self.rules_table.insertRow(row)
        
        self.rules_table.setItem(row, 0, QTableWidgetItem("lowest thru highest"))
        self.rules_table.setItem(row, 1, QTableWidgetItem("->"))
        self.rules_table.item(row, 1).setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.rules_table.setItem(row, 2, QTableWidgetItem(""))
    
    def _delete_rule(self) -> None:
        """선택한 규칙 삭제."""
        row = self.rules_table.currentRow()
        if row >= 0:
            self.rules_table.removeRow(row)
    
    def _parse_rules(self) -> Dict:
        """규칙 테이블을 파싱합니다."""
        rules = {}
        
        for row in range(self.rules_table.rowCount()):
            old_val_item = self.rules_table.item(row, 0)
            new_val_item = self.rules_table.item(row, 2)
            
            if old_val_item and new_val_item:
                old_val = old_val_item.text().strip()
                new_val = new_val_item.text().strip()
                
                if old_val and new_val:
                    rules[old_val] = new_val
        
        return rules
    
    def _apply_recode(self) -> None:
        """재코딩 적용."""
        source_var = self.source_combo.currentText()
        
        if self.same_var_radio.isChecked():
            target_var = source_var
        else:
            target_var = self.target_edit.text().strip()
            if not target_var:
                QMessageBox.warning(self, "경고", "새 변수명을 입력하세요")
                return
            if target_var in self.dataset.data.columns:
                QMessageBox.warning(self, "경고", f"'{target_var}' 변수가 이미 존재합니다")
                return
        
        rules = self._parse_rules()
        if not rules:
            QMessageBox.warning(self, "경고", "재코딩 규칙을 입력하세요")
            return
        
        # 미리보기 업데이트
        preview_text = f"원본 변수: {source_var}\n"
        preview_text += f"대상 변수: {target_var}\n"
        preview_text += f"규칙 수: {len(rules)}개\n\n"
        
        for old_val, new_val in rules.items():
            preview_text += f"  {old_val} -> {new_val}\n"
        
        self.preview_label.setText(preview_text)
        self.preview_label.setStyleSheet("color: #1a5276; font-family: Consolas;")
        
        # 시그널 발생
        self.recode_applied.emit(source_var, target_var, rules)
        
        QMessageBox.information(self, "완료", f"재코딩이 적용되었습니다.\n{target_var} 변수가 생성/변경되었습니다.")
        self.accept()
