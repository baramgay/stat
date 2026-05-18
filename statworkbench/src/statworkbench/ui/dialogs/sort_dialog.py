"""Sort Cases Dialog — SPSS 스타일 데이터 정렬 다이얼로그.

변수를 기준으로 케이스를 정렬합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Signal
from typing import Optional

from statworkbench.core.dataset import Dataset


class SortDialog(QDialog):
    """데이터 정렬 다이얼로그."""
    
    sort_applied = Signal()
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        
        self.setWindowTitle("🔀 케이스 정렬")
        self.setMinimumSize(500, 400)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 정렬 기준
        sort_group = QGroupBox("📋 정렬 기준")
        sort_layout = QVBoxLayout(sort_group)
        
        # 첫 번째 정렬 변수
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("1순위:"))
        self.sort1_combo = QComboBox()
        self.sort1_combo.addItems(self.dataset.data.columns)
        row1.addWidget(self.sort1_combo, 2)
        
        self.order1_group = QButtonGroup(self)
        self.asc1_radio = QRadioButton("오름차순")
        self.asc1_radio.setChecked(True)
        self.order1_group.addButton(self.asc1_radio)
        row1.addWidget(self.asc1_radio)
        
        self.desc1_radio = QRadioButton("내림차순")
        self.order1_group.addButton(self.desc1_radio)
        row1.addWidget(self.desc1_radio)
        
        sort_layout.addLayout(row1)
        
        # 두 번째 정렬 변수
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("2순위:"))
        self.sort2_combo = QComboBox()
        self.sort2_combo.addItem("(없음)")
        self.sort2_combo.addItems(self.dataset.data.columns)
        row2.addWidget(self.sort2_combo, 2)
        
        self.order2_group = QButtonGroup(self)
        self.asc2_radio = QRadioButton("오름차순")
        self.asc2_radio.setChecked(True)
        self.order2_group.addButton(self.asc2_radio)
        row2.addWidget(self.asc2_radio)
        
        self.desc2_radio = QRadioButton("내림차순")
        self.order2_group.addButton(self.desc2_radio)
        row2.addWidget(self.desc2_radio)
        
        sort_layout.addLayout(row2)
        
        # 세 번째 정렬 변수
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("3순위:"))
        self.sort3_combo = QComboBox()
        self.sort3_combo.addItem("(없음)")
        self.sort3_combo.addItems(self.dataset.data.columns)
        row3.addWidget(self.sort3_combo, 2)
        
        self.order3_group = QButtonGroup(self)
        self.asc3_radio = QRadioButton("오름차순")
        self.asc3_radio.setChecked(True)
        self.order3_group.addButton(self.asc3_radio)
        row3.addWidget(self.asc3_radio)
        
        self.desc3_radio = QRadioButton("내림차순")
        self.order3_group.addButton(self.desc3_radio)
        row3.addWidget(self.desc3_radio)
        
        sort_layout.addLayout(row3)
        
        layout.addWidget(sort_group)
        
        # 미리보기
        preview_group = QGroupBox("👁️ 미리보기 (처음 5행)")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(len(self.dataset.data.columns))
        self.preview_table.setHorizontalHeaderLabels(self.dataset.data.columns)
        self.preview_table.setMaximumHeight(150)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        preview_layout.addWidget(self.preview_table)
        
        self.btn_preview = QPushButton("🔍 미리보기")
        self.btn_preview.clicked.connect(self._update_preview)
        preview_layout.addWidget(self.btn_preview)
        
        layout.addWidget(preview_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("✅ 정렬 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_sort)
        action_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        # 초기 미리보기
        self._update_preview()
    
    def _update_preview(self) -> None:
        """미리보기 업데이트."""
        df = self.dataset.data.copy()
        
        # 정렬 기준 수집
        sort_cols = []
        ascending = []
        
        var1 = self.sort1_combo.currentText()
        if var1:
            sort_cols.append(var1)
            ascending.append(self.asc1_radio.isChecked())
        
        var2 = self.sort2_combo.currentText()
        if var2 and var2 != "(없음)":
            sort_cols.append(var2)
            ascending.append(self.asc2_radio.isChecked())
        
        var3 = self.sort3_combo.currentText()
        if var3 and var3 != "(없음)":
            sort_cols.append(var3)
            ascending.append(self.asc3_radio.isChecked())
        
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=ascending)
        
        # 처음 5행 표시
        preview_df = df.head(5)
        self.preview_table.setRowCount(len(preview_df))
        
        for i, (_, row) in enumerate(preview_df.iterrows()):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.preview_table.setItem(i, j, item)
        
        self.preview_table.resizeColumnsToContents()
    
    def _apply_sort(self) -> None:
        """정렬 적용."""
        df = self.dataset.data
        
        # 정렬 기준 수집
        sort_cols = []
        ascending = []
        
        var1 = self.sort1_combo.currentText()
        if var1:
            sort_cols.append(var1)
            ascending.append(self.asc1_radio.isChecked())
        
        var2 = self.sort2_combo.currentText()
        if var2 and var2 != "(없음)":
            sort_cols.append(var2)
            ascending.append(self.asc2_radio.isChecked())
        
        var3 = self.sort3_combo.currentText()
        if var3 and var3 != "(없음)":
            sort_cols.append(var3)
            ascending.append(self.asc3_radio.isChecked())
        
        if not sort_cols:
            QMessageBox.warning(self, "경고", "정렬 기준 변수를 선택하세요")
            return
        
        try:
            df.sort_values(by=sort_cols, ascending=ascending, inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.sort_applied.emit()
            QMessageBox.information(
                self, "완료",
                f"정렬이 적용되었습니다.\n"
                f"정렬 기준: {', '.join(sort_cols)}"
            )
            self.accept()
        
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"정렬 실패:\n{exc}")
