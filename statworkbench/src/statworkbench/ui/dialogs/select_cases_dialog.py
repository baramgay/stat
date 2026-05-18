"""Select Cases Dialog — SPSS 스타일 케이스 선택 다이얼로그.

조건에 맞는 케이스만 선택하여 분석 대상으로 지정합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QComboBox, QLineEdit,
    QSpinBox, QMessageBox, QTextEdit
)
from PySide6.QtCore import Signal
from typing import Optional

from statworkbench.core.dataset import Dataset


class SelectCasesDialog(QDialog):
    """케이스 선택 다이얼로그."""
    
    cases_selected = Signal(str, object)  # selection_type, condition
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        
        self.setWindowTitle("🔍 케이스 선택")
        self.setMinimumSize(500, 450)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 선택 방법
        method_group = QGroupBox("📋 선택 방법")
        method_layout = QVBoxLayout(method_group)
        
        self.method_group = QButtonGroup(self)
        
        self.all_radio = QRadioButton("모든 케이스")
        self.all_radio.setChecked(True)
        self.method_group.addButton(self.all_radio)
        method_layout.addWidget(self.all_radio)
        
        self.condition_radio = QRadioButton("조건이 만족하는 경우")
        self.method_group.addButton(self.condition_radio)
        method_layout.addWidget(self.condition_radio)
        
        # 조건 입력
        cond_layout = QHBoxLayout()
        cond_layout.addWidget(QLabel("조건:"))
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("예: age >= 18 AND gender == '남'")
        cond_layout.addWidget(self.condition_edit)
        method_layout.addLayout(cond_layout)
        
        self.random_radio = QRadioButton("무작위 표본")
        self.method_group.addButton(self.random_radio)
        method_layout.addWidget(self.random_radio)
        
        # 무작위 옵션
        random_layout = QHBoxLayout()
        random_layout.addWidget(QLabel("표본 크기:"))
        self.random_spin = QSpinBox()
        self.random_spin.setRange(1, 999999)
        self.random_spin.setValue(100)
        random_layout.addWidget(self.random_spin)
        random_layout.addStretch()
        method_layout.addLayout(random_layout)
        
        self.range_radio = QRadioButton("케이스 범위")
        self.method_group.addButton(self.range_radio)
        method_layout.addWidget(self.range_radio)
        
        # 범위 옵션
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("시작:"))
        self.range_start = QSpinBox()
        self.range_start.setRange(1, 999999)
        self.range_start.setValue(1)
        range_layout.addWidget(self.range_start)
        
        range_layout.addWidget(QLabel("끝:"))
        self.range_end = QSpinBox()
        self.range_end.setRange(1, 999999)
        self.range_end.setValue(100)
        range_layout.addWidget(self.range_end)
        range_layout.addStretch()
        method_layout.addLayout(range_layout)
        
        layout.addWidget(method_group)
        
        # 필터 변수
        filter_group = QGroupBox("📤 필터 변수")
        filter_layout = QHBoxLayout(filter_group)
        
        filter_layout.addWidget(QLabel("필터 변수명:"))
        self.filter_var_edit = QLineEdit("filter_$")
        filter_layout.addWidget(self.filter_var_edit)
        
        layout.addWidget(filter_group)
        
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
        
        self.btn_preview = QPushButton("🔍 미리보기")
        self.btn_preview.clicked.connect(self._preview_selection)
        preview_layout.addWidget(self.btn_preview)
        
        layout.addWidget(preview_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_apply = QPushButton("✅ 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_selection)
        action_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
    
    def _preview_selection(self) -> None:
        """선택 미리보기."""
        df = self.dataset.data
        
        try:
            if self.all_radio.isChecked():
                count = len(df)
                self.preview_text.setText(f"모든 케이스: {count:,}개")
            
            elif self.condition_radio.isChecked():
                condition = self.condition_edit.text().strip()
                if not condition:
                    self.preview_text.setText("조건을 입력하세요")
                    return
                
                # 안전한 조건 평가
                selected = df.query(condition)
                count = len(selected)
                self.preview_text.setText(f"조건 '{condition}': {count:,}개 선택됨")
            
            elif self.random_radio.isChecked():
                n = min(self.random_spin.value(), len(df))
                self.preview_text.setText(f"무작위 {n:,}개 선택 (전체 {len(df):,}개)")
            
            elif self.range_radio.isChecked():
                start = self.range_start.value()
                end = min(self.range_end.value(), len(df))
                count = max(0, end - start + 1)
                self.preview_text.setText(f"범위 {start}~{end}: {count:,}개")
        
        except Exception as exc:
            self.preview_text.setText(f"오류: {exc}")
    
    def _apply_selection(self) -> None:
        """선택 적용."""
        df = self.dataset.data
        filter_var = self.filter_var_edit.text().strip() or "filter_$"
        
        try:
            if self.all_radio.isChecked():
                # 모든 케이스 선택
                df[filter_var] = 1
                selection_type = "all"
                condition = None
            
            elif self.condition_radio.isChecked():
                condition = self.condition_edit.text().strip()
                if not condition:
                    QMessageBox.warning(self, "경고", "조건을 입력하세요")
                    return
                
                # 조건 평가
                df[filter_var] = 0
                mask = df.eval(condition)
                df.loc[mask, filter_var] = 1
                
                selection_type = "condition"
                condition = condition
            
            elif self.random_radio.isChecked():
                n = min(self.random_spin.value(), len(df))
                
                df[filter_var] = 0
                sample_idx = df.sample(n=n).index
                df.loc[sample_idx, filter_var] = 1
                
                selection_type = "random"
                condition = n
            
            elif self.range_radio.isChecked():
                start = self.range_start.value()
                end = min(self.range_end.value(), len(df))
                
                df[filter_var] = 0
                df.iloc[start-1:end, df.columns.get_loc(filter_var)] = 1
                
                selection_type = "range"
                condition = (start, end)
            
            # 시그널 발생
            self.cases_selected.emit(selection_type, condition)
            
            selected_count = df[filter_var].sum()
            QMessageBox.information(
                self, "완료",
                f"케이스 선택이 적용되었습니다.\n"
                f"선택된 케이스: {selected_count:,}개\n"
                f"필터 변수: {filter_var}"
            )
            self.accept()
        
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"선택 적용 실패:\n{exc}")
