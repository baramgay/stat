"""Merge Dialog — SPSS 스타일 파일 병합 다이얼로그.

데이터셋을 병합(merge)하거나 추가(append)합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QComboBox, QLineEdit,
    QFileDialog, QMessageBox, QTextEdit, QListWidget, QAbstractItemView
)
from PySide6.QtCore import Signal
from typing import Optional

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.io.csv_reader import read_csv
from statworkbench.io.excel_reader import read_excel


class MergeDialog(QDialog):
    """파일 병합 다이얼로그."""
    
    merge_completed = Signal(object)  # merged_dataset
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.second_dataset: Optional[Dataset] = None
        
        self.setWindowTitle("🔗 파일 병합")
        self.setMinimumSize(600, 500)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 병합 유형
        type_group = QGroupBox("📋 병합 유형")
        type_layout = QVBoxLayout(type_group)
        
        self.type_group = QButtonGroup(self)
        
        self.add_vars_radio = QRadioButton("변수 추가 (Add Variables) - 가로 병합")
        self.add_vars_radio.setChecked(True)
        self.type_group.addButton(self.add_vars_radio)
        type_layout.addWidget(self.add_vars_radio)
        
        self.add_cases_radio = QRadioButton("케이스 추가 (Add Cases) - 세로 병합")
        self.type_group.addButton(self.add_cases_radio)
        type_layout.addWidget(self.add_cases_radio)
        
        layout.addWidget(type_group)
        
        # 두 번째 파일
        file_group = QGroupBox("📂 두 번째 파일")
        file_layout = QVBoxLayout(file_group)
        
        file_btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 파일 불러오기...")
        self.btn_load.clicked.connect(self._load_second_file)
        file_btn_layout.addWidget(self.btn_load)
        
        self.file_label = QLabel("파일을 선택하세요")
        self.file_label.setStyleSheet("color: #5d6d7e;")
        file_btn_layout.addWidget(self.file_label)
        file_btn_layout.addStretch()
        
        file_layout.addLayout(file_btn_layout)
        
        # 파일 정보
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setMaximumHeight(80)
        self.file_info.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 11px;"
        )
        file_layout.addWidget(self.file_info)
        
        layout.addWidget(file_group)
        
        # 키 변수 (변수 추가 시)
        self.key_group = QGroupBox("🔑 키 변수 (병합 기준)")
        key_layout = QVBoxLayout(self.key_group)
        
        key_select_layout = QHBoxLayout()
        key_select_layout.addWidget(QLabel("현재 데이터셋:"))
        self.key1_combo = QComboBox()
        self.key1_combo.addItems(self.dataset.data.columns)
        key_select_layout.addWidget(self.key1_combo)
        
        key_select_layout.addWidget(QLabel("두 번째 데이터셋:"))
        self.key2_combo = QComboBox()
        key_select_layout.addWidget(self.key2_combo)
        
        key_layout.addLayout(key_select_layout)
        layout.addWidget(self.key_group)
        
        # 미리보기
        preview_group = QGroupBox("👁️ 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 11px;"
        )
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_merge = QPushButton("🔗 병합")
        self.btn_merge.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_merge.clicked.connect(self._execute_merge)
        action_layout.addWidget(self.btn_merge)
        
        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
    
    def _load_second_file(self) -> None:
        """두 번째 파일 불러오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "파일 선택", "",
            "CSV (*.csv);;Excel (*.xlsx *.xls);;모든 파일 (*.*)"
        )
        
        if path:
            try:
                if path.endswith('.csv'):
                    self.second_dataset = read_csv(path)
                else:
                    self.second_dataset = read_excel(path)
                
                self.file_label.setText(path.split('/')[-1])
                
                # 정보 표시
                info = f"""
행 수: {len(self.second_dataset.data):,}
열 수: {len(self.second_dataset.data.columns)}
변수: {', '.join(self.second_dataset.data.columns[:5])}{'...' if len(self.second_dataset.data.columns) > 5 else ''}
                """.strip()
                self.file_info.setText(info)
                
                # 키 변수 업데이트
                self.key2_combo.clear()
                self.key2_combo.addItems(self.second_dataset.data.columns)
                
                # 공통 변수 찾아서 기본 선택
                common = set(self.dataset.data.columns) & set(self.second_dataset.data.columns)
                if common:
                    self.key1_combo.setCurrentText(list(common)[0])
                    self.key2_combo.setCurrentText(list(common)[0])
                
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"파일 불러오기 실패:\n{exc}")
    
    def _execute_merge(self) -> None:
        """병합 실행."""
        if self.second_dataset is None:
            QMessageBox.warning(self, "경고", "두 번째 파일을 불러오세요")
            return
        
        try:
            if self.add_vars_radio.isChecked():
                # 변수 추가 (가로 병합)
                key1 = self.key1_combo.currentText()
                key2 = self.key2_combo.currentText()
                
                merged = pd.merge(
                    self.dataset.data,
                    self.second_dataset.data,
                    left_on=key1,
                    right_on=key2,
                    how='outer',
                    suffixes=('', '_y')
                )
                
                # 중복 열 제거
                merged = merged.loc[:, ~merged.columns.str.endswith('_y')]
                
                merge_type = "변수 추가"
            
            else:
                # 케이스 추가 (세로 병합)
                merged = pd.concat(
                    [self.dataset.data, self.second_dataset.data],
                    ignore_index=True
                )
                merge_type = "케이스 추가"
            
            # 결과 데이터셋 생성
            from statworkbench.core.dataset import Dataset
            result_dataset = Dataset(
                name=f"{self.dataset.name}_merged",
                data=merged
            )
            
            self.merge_completed.emit(result_dataset)
            
            QMessageBox.information(
                self, "완료",
                f"병합이 완료되었습니다.\n"
                f"유형: {merge_type}\n"
                f"결과 행 수: {len(merged):,}\n"
                f"결과 열 수: {len(merged.columns)}"
            )
            self.accept()
        
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"병합 실패:\n{exc}")
