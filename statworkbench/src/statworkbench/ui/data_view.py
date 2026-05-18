"""Data View — SPSS 스타일 데이터 편집 화면.

셀 단위 데이터 입력/편집을 지원합니다.
- 기본 1000행 x 100열 격자
- 헤더(변수명) 직접 편집
- 화살표/엔터/탭 키보드 네비게이션
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableView,
    QLabel,
    QHeaderView,
    QAbstractItemView,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QMenu,
    QLineEdit,
    QComboBox,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QKeyEvent
from typing import Optional

from statworkbench.core.dataset import Dataset
from statworkbench.ui.models.spss_grid_model import SPSSGridModel


class DataView(QWidget):
    """SPSS Data View 스타일 데이터 편집 화면."""

    dataset_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._model: Optional[SPSSGridModel] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 정보 바
        self.info_bar = QLabel("데이터 없음 — 셀을 더블클릭하여 데이터 입력")
        self.info_bar.setStyleSheet(
            "font-size: 12px; color: #5d6d7e; padding: 6px 10px; "
            "background-color: #f1f3f4; border-radius: 4px;"
        )
        layout.addWidget(self.info_bar)

        # 도구 버튼
        btn_layout = QHBoxLayout()
        
        self.btn_add_row = QPushButton("+ 행 추가")
        self.btn_add_row.setToolTip("새 행을 추가합니다")
        self.btn_add_row.clicked.connect(self._add_row)
        btn_layout.addWidget(self.btn_add_row)

        self.btn_add_col = QPushButton("+ 변수 추가")
        self.btn_add_col.setToolTip("새 변수(열)를 추가합니다")
        self.btn_add_col.clicked.connect(self._add_column)
        btn_layout.addWidget(self.btn_add_col)

        self.btn_del_row = QPushButton("- 행 삭제")
        self.btn_del_row.setToolTip("선택한 행을 삭제합니다")
        self.btn_del_row.clicked.connect(self._delete_row)
        btn_layout.addWidget(self.btn_del_row)

        self.btn_del_col = QPushButton("- 변수 삭제")
        self.btn_del_col.setToolTip("선택한 변수(열)를 삭제합니다")
        self.btn_del_col.clicked.connect(self._delete_column)
        btn_layout.addWidget(self.btn_del_col)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 검색/필터 바
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("🔍 검색:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("셀 값 검색...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.returnPressed.connect(self._search_next)
        filter_layout.addWidget(self.search_edit, 2)
        
        self.search_btn = QPushButton("다음 찾기")
        self.search_btn.clicked.connect(self._search_next)
        filter_layout.addWidget(self.search_btn)
        
        filter_layout.addSpacing(20)
        
        filter_layout.addWidget(QLabel("🎚️ 필터:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("모든 행")
        self.filter_combo.addItem("빈 셀 제외")
        self.filter_combo.addItem("0 제외")
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_combo, 1)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # 데이터 테이블
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setDefaultSectionSize(26)
        
        # 컨텍스트 메뉴
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        # SPSS 스타일: 항상 편집 가능
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked 
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
        )
        
        # 키보드 네비게이션
        self.table.setTabKeyNavigation(True)
        
        # 엔터 키 → 아래로 이동 (이벤트 필터 설치)
        self.table.installEventFilter(self)
        
        # SPSS 스타일 테이블
        self.table.setStyleSheet("""
            QTableView {
                border: 1px solid #c0c4cc;
                gridline-color: #e0e4e8;
                font-size: 13px;
            }
            QTableView::item {
                padding: 4px 8px;
                border-bottom: 1px solid #e0e4e8;
            }
            QTableView::item:selected {
                background-color: #d4e6f1;
                color: #1a1a2e;
            }
            QHeaderView::section {
                background-color: #e8f0f8;
                color: #1a5276;
                font-weight: bold;
                padding: 6px 10px;
                border: 1px solid #c0c4cc;
            }
            QHeaderView::section:horizontal {
                background-color: #d4e6f1;
                color: #1a5276;
            }
        """)
        
        # 헤더 클릭으로 변수명 편집
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._edit_header)
        
        layout.addWidget(self.table)

        # 안내 문구
        self.help_label = QLabel(
            "💡 팁: 셀을 더블클릭하여 데이터 입력 | "
            "변수명은 헤더를 더블클릭하여 수정 | "
            "화살표/엔터/탭 키로 이동"
        )
        self.help_label.setStyleSheet("color: #7a7a8a; font-size: 11px; padding: 4px;")
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

    def eventFilter(self, obj, event):
        """이벤트 필터: 엔터 키 → 아래로 이동."""
        if obj == self.table and event.type() == QEvent.Type.KeyPress:
            key_event = event
            if key_event.key() in (Qt.Key_Return, Qt.Key_Enter):
                # 현재 셀에서 편집 완료 후 아래로 이동
                current = self.table.currentIndex()
                if current.isValid():
                    # 먼저 현재 셀의 편집을 커밋
                    self.table.commitData(self.table.indexWidget(current))
                    self.table.closePersistentEditor(current)
                    
                    # 아래 셀로 이동
                    next_index = self._model.index(current.row() + 1, current.column())
                    if next_index.isValid():
                        self.table.setCurrentIndex(next_index)
                        self.table.scrollTo(next_index)
                        # 편집 모드로 전환
                        self.table.edit(next_index)
                    return True
        return super().eventFilter(obj, event)

    def set_dataset(self, dataset: Dataset) -> None:
        """데이터셋을 설정합니다."""
        self._dataset = dataset
        self._model = SPSSGridModel(dataset.data)
        self._model.data_changed.connect(self._on_data_changed)
        self._model.variable_added.connect(self._on_variable_added)
        self._model.variable_renamed.connect(self._on_variable_renamed)
        self.table.setModel(self._model)
        self._update_info()

    def refresh(self) -> None:
        """화면을 새로고침합니다."""
        if self._model is not None:
            self._model.beginResetModel()
            self._model.endResetModel()
        self._update_info()

    def get_dataset(self) -> Optional[Dataset]:
        """현재 데이터셋을 반환합니다."""
        return self._dataset

    def _update_info(self) -> None:
        """정보 바를 업데이트합니다."""
        if self._dataset is None or self._model is None:
            self.info_bar.setText("데이터 없음 — 셀을 더블클릭하여 데이터 입력")
            return
        
        # 실제 데이터가 있는 범위
        df = self._model.get_dataframe()
        n_rows = len(df)
        n_cols = len(df.columns)
        
        measures = {}
        for v in self._dataset.variables.values():
            m = v.measure.value if hasattr(v.measure, 'value') else str(v.measure)
            measures[m] = measures.get(m, 0) + 1
        
        measure_str = ", ".join([f"{k}({v})" for k, v in measures.items()]) if measures else "없음"
        self.info_bar.setText(
            f"📊 {self._dataset.name}: {n_rows:,}행 × {n_cols:,}변수 | "
            f"측정: {measure_str}"
        )

    def _on_data_changed(self) -> None:
        """데이터 변경 시 호출됩니다."""
        if self._dataset is not None and self._model is not None:
            self._dataset.data = self._model.get_dataframe()
            self._update_info()
            self.dataset_changed.emit()

    def _on_variable_added(self, var_name: str) -> None:
        """새 변수 추가 시 호출됩니다."""
        if self._dataset is not None:
            from statworkbench.core.variable import VariableMeta, StorageType, MeasureType
            if var_name not in self._dataset.variables:
                var = VariableMeta(
                    name=var_name,
                    label=var_name,
                    storage_type=StorageType.STRING,
                    measure=MeasureType.NOMINAL,
                )
                self._dataset.variables[var_name] = var

    def _on_variable_renamed(self, old_name: str, new_name: str) -> None:
        """변수명 변경 시 호출됩니다."""
        if self._dataset is not None and old_name in self._dataset.variables:
            var = self._dataset.variables.pop(old_name)
            var.name = new_name
            var.label = new_name
            self._dataset.variables[new_name] = var

    def _edit_header(self, section: int) -> None:
        """헤더(변수명)를 편집합니다."""
        if self._model is None:
            return
        
        from PySide6.QtWidgets import QInputDialog
        current_name = self._model.headerData(
            section, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        )
        
        new_name, ok = QInputDialog.getText(
            self, "변수명 변경", "새 변수명:", text=str(current_name)
        )
        
        if ok and new_name.strip():
            self._model.setHeaderData(
                section, Qt.Orientation.Horizontal, new_name.strip()
            )

    def _add_row(self) -> None:
        """행을 추가합니다."""
        if self._model is not None:
            self._model.add_row()

    def _add_column(self) -> None:
        """변수(열)를 추가합니다."""
        if self._model is not None:
            from statworkbench.ui.models.spss_grid_model import generate_var_name
            var_name = generate_var_name(len(self._model.get_full_dataframe().columns) + 1)
            self._model.add_column(var_name)

    def _delete_row(self) -> None:
        """선택한 행을 삭제합니다."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        if index.isValid():
            row = index.row()
            reply = QMessageBox.question(
                self, "행 삭제", f"{row + 1}번 행을 삭제하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._model.remove_row(row)

    def _delete_column(self) -> None:
        """선택한 변수(열)를 삭제합니다."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        if index.isValid():
            col = index.column()
            col_name = self._model.get_full_dataframe().columns[col]
            reply = QMessageBox.question(
                self, "변수 삭제", f"변수 '{col_name}'을(를) 삭제하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._model.remove_column(col)

    def _show_context_menu(self, position):
        """컨텍스트 메뉴를 표시합니다."""
        if self._model is None:
            return
        
        menu = QMenu(self)
        
        # 현재 셀 정보
        index = self.table.indexAt(position)
        
        # 편집 메뉴
        edit_action = menu.addAction("✏️ 편집")
        edit_action.triggered.connect(lambda: self.table.edit(index))
        
        menu.addSeparator()
        
        # 복사/붙여넣기
        copy_action = menu.addAction("📋 복사")
        copy_action.triggered.connect(self._copy_cell)
        
        paste_action = menu.addAction("📋 붙여넣기")
        paste_action.triggered.connect(self._paste_cell)
        
        menu.addSeparator()
        
        # 행/열 조작
        add_row_action = menu.addAction("➕ 행 추가")
        add_row_action.triggered.connect(self._add_row)
        
        add_col_action = menu.addAction("➕ 변수 추가")
        add_col_action.triggered.connect(self._add_column)
        
        if index.isValid():
            del_row_action = menu.addAction("➖ 행 삭제")
            del_row_action.triggered.connect(self._delete_row)
            
            del_col_action = menu.addAction("➖ 변수 삭제")
            del_col_action.triggered.connect(self._delete_column)
        
        menu.addSeparator()
        
        # 정렬
        sort_asc_action = menu.addAction("🔼 오름차순 정렬")
        sort_asc_action.triggered.connect(self._sort_ascending)
        
        sort_desc_action = menu.addAction("🔽 내림차순 정렬")
        sort_desc_action.triggered.connect(self._sort_descending)
        
        menu.exec(self.table.viewport().mapToGlobal(position))
    
    def _copy_cell(self):
        """현재 셀 값을 복사합니다."""
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            value = self._model.data(index, Qt.ItemDataRole.DisplayRole)
            if value:
                from PySide6.QtWidgets import QApplication
                QApplication.clipboard().setText(str(value))
    
    def _paste_cell(self):
        """클립보드 값을 현재 셀에 붙여넣습니다."""
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            from PySide6.QtWidgets import QApplication
            text = QApplication.clipboard().text()
            if text:
                self._model.setData(index, text)
    
    def _sort_ascending(self):
        """오름차순 정렬."""
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            col = index.column()
            df = self._model.get_dataframe()
            col_name = df.columns[col]
            df.sort_values(by=col_name, ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            self._model.beginResetModel()
            self._model._dataframe = df
            self._model.endResetModel()
    
    def _sort_descending(self):
        """내림차순 정렬."""
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            col = index.column()
            df = self._model.get_dataframe()
            col_name = df.columns[col]
            df.sort_values(by=col_name, ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            self._model.beginResetModel()
            self._model._dataframe = df
            self._model.endResetModel()
    
    def _search_next(self):
        """다음 검색 결과로 이동."""
        if not self._model:
            return
        
        search_text = self.search_edit.text().strip()
        if not search_text:
            return
        
        df = self._model.get_dataframe()
        current = self.table.currentIndex()
        start_row = current.row() + 1 if current.isValid() else 0
        
        # 검색
        for row in range(start_row, len(df)):
            for col in range(len(df.columns)):
                value = str(df.iloc[row, col])
                if search_text.lower() in value.lower():
                    index = self._model.index(row, col)
                    self.table.setCurrentIndex(index)
                    self.table.scrollTo(index)
                    return
        
        # 처음부터 다시 검색
        for row in range(start_row):
            for col in range(len(df.columns)):
                value = str(df.iloc[row, col])
                if search_text.lower() in value.lower():
                    index = self._model.index(row, col)
                    self.table.setCurrentIndex(index)
                    self.table.scrollTo(index)
                    return
        
        QMessageBox.information(self, "검색", f"'{search_text}'을(를) 찾을 수 없습니다.")
    
    def _apply_filter(self):
        """필터 적용."""
        if not self._model or not self._dataset:
            return
        
        filter_type = self.filter_combo.currentIndex()
        df = self._dataset.data.copy()
        
        if filter_type == 0:  # 모든 행
            pass
        elif filter_type == 1:  # 빈 셀 제외
            df = df.dropna()
        elif filter_type == 2:  # 0 제외
            df = df[(df != 0).all(axis=1)]
        
        self._model.beginResetModel()
        self._model._dataframe = df
        self._model.endResetModel()
        self._update_info()
