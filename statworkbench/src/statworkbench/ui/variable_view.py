"""Variable View — SPSS 스타일 변수 속성 편집 화면.

SPSS Variable View와 동일한 11개 속성 컬럼:
Name, Type, Width, Decimals, Label, Values, Missing, Columns, Align, Measure, Role
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QLabel, QHeaderView,
    QAbstractItemView, QPushButton, QHBoxLayout, QMessageBox,
    QComboBox, QStyledItemDelegate, QLineEdit, QSpinBox
)
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from typing import Any, Optional, List

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta, StorageType, MeasureType


class VariablePropertiesModel(QAbstractTableModel):
    """SPSS Variable View 모델 (11개 속성)."""
    
    # SPSS Variable View 컬럼 정의
    COLUMNS = [
        ("Name", "변수명"),
        ("Type", "유형"),
        ("Width", "너비"),
        ("Decimals", "소수"),
        ("Label", "라벨"),
        ("Values", "값"),
        ("Missing", "결측"),
        ("Columns", "열"),
        ("Align", "정렬"),
        ("Measure", "측정"),
        ("Role", "역할"),
    ]
    
    # 편집 가능한 컬럼
    EDITABLE_COLS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    
    data_changed = Signal()
    
    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__()
        self._dataset = dataset
        self._variables: List[VariableMeta] = []
        self._update_variables()
    
    def _update_variables(self):
        if self._dataset:
            self._variables = list(self._dataset.variables.values())
        else:
            self._variables = []
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._variables)
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.COLUMNS)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._variables):
            return None
        
        var = self._variables[index.row()]
        col = index.column()
        
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._get_value(var, col)
        
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (2, 3, 7):  # 숫자 컬럼
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        
        if role == Qt.ItemDataRole.BackgroundRole:
            # Measure에 따른 색상
            if col == 9:
                measure = var.measure
                if measure == MeasureType.SCALE:
                    return QColor(232, 245, 233)  # 연한 초록
                elif measure == MeasureType.NOMINAL:
                    return QColor(255, 243, 224)  # 연한 주황
                elif measure == MeasureType.ORDINAL:
                    return QColor(227, 242, 253)  # 연한 파랑
        
        return None
    
    def _get_value(self, var: VariableMeta, col: int) -> str:
        if col == 0:
            return var.name
        elif col == 1:
            return var.storage_type.value if hasattr(var.storage_type, 'value') else str(var.storage_type)
        elif col == 2:
            return str(var.width if hasattr(var, 'width') else 8)
        elif col == 3:
            return str(var.decimals if hasattr(var, 'decimals') else 2)
        elif col == 4:
            return var.label or ""
        elif col == 5:
            if var.value_labels:
                return f"{{{len(var.value_labels)}개}}"
            return ""
        elif col == 6:
            if var.missing_values:
                return f"{{{len(var.missing_values)}개}}"
            return ""
        elif col == 7:
            return str(var.column_width if hasattr(var, 'column_width') else 8)
        elif col == 8:
            return var.align if hasattr(var, 'align') else "Right"
        elif col == 9:
            return var.measure.value if hasattr(var.measure, 'value') else str(var.measure)
        elif col == 10:
            return var.role if hasattr(var, 'role') else "Input"
        return ""
    
    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        
        row, col = index.row(), index.column()
        if row >= len(self._variables):
            return False
        
        var = self._variables[row]
        
        try:
            if col == 0:  # Name
                old_name = var.name
                new_name = str(value).strip()
                if new_name and new_name != old_name:
                    var.name = new_name
                    if self._dataset and old_name in self._dataset.variables:
                        self._dataset.variables[new_name] = self._dataset.variables.pop(old_name)
                        if old_name in self._dataset.data.columns:
                            self._dataset.data.rename(columns={old_name: new_name}, inplace=True)
            elif col == 1:  # Type
                var.storage_type = StorageType(str(value))
            elif col == 2:  # Width
                var.width = int(value)
            elif col == 3:  # Decimals
                var.decimals = int(value)
            elif col == 4:  # Label
                var.label = str(value)
            elif col == 7:  # Columns
                var.column_width = int(value)
            elif col == 8:  # Align
                var.align = str(value)
            elif col == 9:  # Measure
                measure_str = str(value).strip().upper()
                if measure_str == "SCALE":
                    var.measure = MeasureType.SCALE
                elif measure_str == "NOMINAL":
                    var.measure = MeasureType.NOMINAL
                elif measure_str == "ORDINAL":
                    var.measure = MeasureType.ORDINAL
                else:
                    var.measure = MeasureType(str(value))
            elif col == 10:  # Role
                var.role = str(value)
            
            self.dataChanged.emit(index, index)
            self.data_changed.emit()
            return True
        except (ValueError, TypeError):
            return False
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self.COLUMNS[section][1]  # 한글 이름
            return str(section + 1)
        
        if role == Qt.ItemDataRole.ToolTipRole:
            if orientation == Qt.Orientation.Horizontal:
                return self.COLUMNS[section][0]  # 영문 이름
        
        return None
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() in self.EDITABLE_COLS:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags
    
    def set_dataset(self, dataset: Dataset):
        self.beginResetModel()
        self._dataset = dataset
        self._update_variables()
        self.endResetModel()
    
    def add_variable(self, name: str = "VAR00001"):
        if not self._dataset:
            return
        
        from statworkbench.core.variable import VariableMeta
        var = VariableMeta(name=name, label=name)
        
        self.beginInsertRows(QModelIndex(), len(self._variables), len(self._variables))
        self._dataset.variables[name] = var
        if name not in self._dataset.data.columns:
            self._dataset.data[name] = None
        self._variables.append(var)
        self.endInsertRows()
        self.data_changed.emit()
    
    def remove_variable(self, row: int) -> bool:
        if 0 <= row < len(self._variables):
            self.beginRemoveRows(QModelIndex(), row, row)
            var = self._variables.pop(row)
            if self._dataset:
                if var.name in self._dataset.variables:
                    del self._dataset.variables[var.name]
                if var.name in self._dataset.data.columns:
                    self._dataset.data.drop(columns=[var.name], inplace=True)
            self.endRemoveRows()
            self.data_changed.emit()
            return True
        return False
    
    def move_variable(self, from_row: int, to_row: int) -> bool:
        """변수 순서 이동."""
        if 0 <= from_row < len(self._variables) and 0 <= to_row < len(self._variables):
            self.beginResetModel()
            # 변수 목록에서 이동
            var = self._variables.pop(from_row)
            self._variables.insert(to_row, var)
            
            # 데이터셋의 변수 순서도 업데이트
            if self._dataset:
                # OrderedDict로 순서 유지
                from collections import OrderedDict
                new_vars = OrderedDict()
                for v in self._variables:
                    new_vars[v.name] = v
                self._dataset.variables = new_vars
                
                # DataFrame 컬럼 순서도 업데이트
                cols = [v.name for v in self._variables]
                self._dataset.data = self._dataset.data[cols]
            
            self.endResetModel()
            self.data_changed.emit()
            return True
        return False


class VariableView(QWidget):
    """SPSS Variable View 위젯."""
    
    dataset_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._model: Optional[VariablePropertiesModel] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # 정보 바
        self.info_bar = QLabel("변수 없음 — 변수를 추가하세요")
        self.info_bar.setStyleSheet(
            "font-size: 12px; color: #5d6d7e; padding: 6px 10px; "
            "background-color: #f1f3f4; border-radius: 4px;"
        )
        layout.addWidget(self.info_bar)
        
        # 도구 버튼
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("+ 변수 추가")
        self.btn_add.setToolTip("새 변수를 추가합니다 (Ctrl+Insert)")
        self.btn_add.clicked.connect(self._add_variable)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_del = QPushButton("- 변수 삭제")
        self.btn_del.setToolTip("선택한 변수를 삭제합니다 (Delete)")
        self.btn_del.clicked.connect(self._delete_variable)
        btn_layout.addWidget(self.btn_del)
        
        self.btn_up = QPushButton("↑ 위로")
        self.btn_up.setToolTip("변수 순서를 위로 이동")
        self.btn_up.clicked.connect(self._move_up)
        btn_layout.addWidget(self.btn_up)
        
        self.btn_down = QPushButton("↓ 아래로")
        self.btn_down.setToolTip("변수 순서를 아래로 이동")
        self.btn_down.clicked.connect(self._move_down)
        btn_layout.addWidget(self.btn_down)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 변수 속성 테이블
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setDefaultSectionSize(24)
        
        # SPSS 스타일
        self.table.setStyleSheet("""
            QTableView {
                border: 1px solid #c0c4cc;
                gridline-color: #e0e4e8;
                font-size: 12px;
            }
            QTableView::item {
                padding: 2px 6px;
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
                padding: 4px 8px;
                border: 1px solid #c0c4cc;
            }
        """)
        
        layout.addWidget(self.table)
        
        # 안내 문구
        self.help_label = QLabel(
            "💡 팁: 셀을 더블클릭하여 편집 | "
            "변수명, 유형, 측정 척도 등을 설정 | "
            "행을 선택하고 Delete 키로 삭제"
        )
        self.help_label.setStyleSheet("color: #7a7a8a; font-size: 11px; padding: 4px;")
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)
    
    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset
        self._model = VariablePropertiesModel(dataset)
        self._model.data_changed.connect(self._on_data_changed)
        self.table.setModel(self._model)
        
        # 컬럼 너비 설정
        self.table.setColumnWidth(0, 120)  # Name
        self.table.setColumnWidth(1, 100)  # Type
        self.table.setColumnWidth(2, 50)   # Width
        self.table.setColumnWidth(3, 50)   # Decimals
        self.table.setColumnWidth(4, 150)  # Label
        self.table.setColumnWidth(5, 80)   # Values
        self.table.setColumnWidth(6, 80)   # Missing
        self.table.setColumnWidth(7, 50)   # Columns
        self.table.setColumnWidth(8, 60)   # Align
        self.table.setColumnWidth(9, 80)   # Measure
        self.table.setColumnWidth(10, 80)  # Role
        
        self._update_info()
    
    def _update_info(self):
        if self._dataset is None:
            self.info_bar.setText("변수 없음 — 변수를 추가하세요")
            return
        
        n_vars = len(self._dataset.variables)
        
        measures = {}
        for v in self._dataset.variables.values():
            m = v.measure.value if hasattr(v.measure, 'value') else str(v.measure)
            measures[m] = measures.get(m, 0) + 1
        
        measure_str = ", ".join([f"{k}({v})" for k, v in measures.items()]) if measures else "없음"
        self.info_bar.setText(
            f"📊 {self._dataset.name}: {n_vars}개 변수 | "
            f"측정: {measure_str}"
        )
    
    def _on_data_changed(self):
        self._update_info()
        self.dataset_changed.emit()
    
    def _add_variable(self):
        if self._model:
            from statworkbench.ui.models.spss_grid_model import generate_var_name
            name = generate_var_name(len(self._model._variables) + 1)
            self._model.add_variable(name)
    
    def _delete_variable(self):
        if self._model is None:
            return
        index = self.table.currentIndex()
        if index.isValid():
            row = index.row()
            var_name = self._model._variables[row].name
            reply = QMessageBox.question(
                self, "변수 삭제", f"변수 '{var_name}'을(를) 삭제하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._model.remove_variable(row)
    
    def _move_up(self):
        """선택한 변수를 위로 이동."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        if not index.isValid() or index.row() <= 0:
            return
        
        row = index.row()
        self._model.move_variable(row, row - 1)
        self.table.selectRow(row - 1)
    
    def _move_down(self):
        """선택한 변수를 아래로 이동."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        if not index.isValid():
            return
        
        row = index.row()
        if row >= len(self._model._variables) - 1:
            return
        
        self._model.move_variable(row, row + 1)
        self.table.selectRow(row + 1)
