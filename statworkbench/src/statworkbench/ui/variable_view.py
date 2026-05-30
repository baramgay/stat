"""Variable View — SPSS 스타일 변수 속성 편집 화면.

SPSS Variable View와 동일한 11개 속성 컬럼:
Name, Type, Width, Decimals, Label, Values, Missing, Columns, Align, Measure, Role
"""

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import MeasureType, Role, StorageType, VariableMeta

# SPSS 호환 표시 이름
_TYPE_DISPLAY = {
    StorageType.FLOAT:       "숫자형",
    StorageType.INTEGER:     "정수형",
    StorageType.STRING:      "문자열",
    StorageType.DATETIME:    "날짜/시간",
    StorageType.BOOLEAN:     "논리형",
    StorageType.CATEGORICAL: "범주형",
}
_TYPE_REVERSE = {v: k for k, v in _TYPE_DISPLAY.items()}
_TYPE_OPTIONS  = ["숫자형", "정수형", "문자열", "날짜/시간", "논리형", "범주형"]

_MEASURE_DISPLAY = {
    MeasureType.SCALE:     "척도",
    MeasureType.ORDINAL:   "순서형",
    MeasureType.NOMINAL:   "명목형",
    MeasureType.BINARY:    "이분형",
    MeasureType.DATE_TIME: "날짜/시간",
    MeasureType.TEXT:      "텍스트",
}
_MEASURE_REVERSE = {v: k for k, v in _MEASURE_DISPLAY.items()}
_MEASURE_OPTIONS = ["척도", "순서형", "명목형", "이분형"]

_ALIGN_OPTIONS  = ["오른쪽", "왼쪽", "가운데"]
_ALIGN_TO_STR   = {"오른쪽": "right", "왼쪽": "left", "가운데": "center"}
_STR_TO_ALIGN   = {v: k for k, v in _ALIGN_TO_STR.items()}
_STR_TO_ALIGN.update({"Right": "오른쪽", "Left": "왼쪽", "Center": "가운데",
                       "right": "오른쪽", "left": "왼쪽", "center": "가운데"})

_ROLE_DISPLAY = {
    Role.INPUT:     "입력",
    Role.TARGET:    "목표",
    Role.WEIGHT:    "가중치",
    Role.ID:        "ID",
    Role.SPLIT:     "분리",
    Role.FREQUENCY: "빈도",
    Role.NONE:      "없음",
}
_ROLE_REVERSE = {v: k for k, v in _ROLE_DISPLAY.items()}
_ROLE_OPTIONS = ["입력", "목표", "가중치", "ID", "분리", "빈도", "없음"]


class VariableViewDelegate(QStyledItemDelegate):
    """Variable View 전용 delegate: 콤보박스(유형/측정/정렬/역할) + 스핀박스(너비/소수/열)."""

    def createEditor(self, parent, option, index):
        col = index.column()
        if col == 1:   # 유형
            cb = QComboBox(parent)
            cb.addItems(_TYPE_OPTIONS)
            cb.activated.connect(lambda idx, e=cb: (self.commitData.emit(e), self.closeEditor.emit(e, QStyledItemDelegate.EndEditHint.NoHint)))
            return cb
        if col == 8:   # 정렬
            cb = QComboBox(parent)
            cb.addItems(_ALIGN_OPTIONS)
            cb.activated.connect(lambda idx, e=cb: (self.commitData.emit(e), self.closeEditor.emit(e, QStyledItemDelegate.EndEditHint.NoHint)))
            return cb
        if col == 9:   # 측정
            cb = QComboBox(parent)
            cb.addItems(_MEASURE_OPTIONS)
            cb.activated.connect(lambda idx, e=cb: (self.commitData.emit(e), self.closeEditor.emit(e, QStyledItemDelegate.EndEditHint.NoHint)))
            return cb
        if col == 10:  # 역할
            cb = QComboBox(parent)
            cb.addItems(_ROLE_OPTIONS)
            cb.activated.connect(lambda idx, e=cb: (self.commitData.emit(e), self.closeEditor.emit(e, QStyledItemDelegate.EndEditHint.NoHint)))
            return cb
        if col in (2, 7):  # 너비, 열
            sb = QSpinBox(parent)
            sb.setRange(1, 255)
            sb.setFrame(False)
            return sb
        if col == 3:  # 소수
            sb = QSpinBox(parent)
            sb.setRange(0, 16)
            sb.setFrame(False)
            return sb
        if col in (5, 6):  # 값, 결측 → 인라인 편집 불가 (다이얼로그로만)
            return None
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        value = index.data(Qt.ItemDataRole.DisplayRole)
        if isinstance(editor, QComboBox):
            idx = editor.findText(str(value) if value else "")
            if idx >= 0:
                editor.setCurrentIndex(idx)
        elif isinstance(editor, QSpinBox):
            try:
                editor.setValue(int(str(value)) if value else 1)
            except (ValueError, TypeError):
                editor.setValue(1)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        elif isinstance(editor, QSpinBox):
            model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


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

    def __init__(self, dataset: Dataset | None = None):
        super().__init__()
        self._dataset = dataset
        self._variables: list[VariableMeta] = []
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
            return _TYPE_DISPLAY.get(var.storage_type, "숫자형")
        elif col == 2:
            return str(var.width if hasattr(var, 'width') else 8)
        elif col == 3:
            return str(var.decimals if hasattr(var, 'decimals') else 2)
        elif col == 4:
            return var.label or ""
        elif col == 5:
            if var.value_labels:
                return f"{{{len(var.value_labels)}개}}"
            return "없음"
        elif col == 6:
            if var.missing_values:
                return f"{{{len(var.missing_values)}개}}"
            return "없음"
        elif col == 7:
            return str(var.column_width if hasattr(var, 'column_width') else 8)
        elif col == 8:
            align_str = var.align if hasattr(var, 'align') else "right"
            return _STR_TO_ALIGN.get(align_str, "오른쪽")
        elif col == 9:
            return _MEASURE_DISPLAY.get(var.measure, "척도")
        elif col == 10:
            role = var.role if hasattr(var, 'role') else Role.INPUT
            if isinstance(role, str):
                try:
                    role = Role(role)
                except ValueError:
                    return "입력"
            return _ROLE_DISPLAY.get(role, "입력")
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
            elif col == 1:  # Type — SPSS 표시 이름으로 받음
                new_type = _TYPE_REVERSE.get(str(value))
                if new_type is not None:
                    var.storage_type = new_type
                    # 문자열로 변경 시 측정 척도 자동 조정
                    if new_type == StorageType.STRING and var.measure not in (
                        MeasureType.NOMINAL, MeasureType.ORDINAL, MeasureType.TEXT
                    ):
                        var.measure = MeasureType.NOMINAL
                        measure_idx = self.index(row, 9)
                        self.dataChanged.emit(measure_idx, measure_idx)
            elif col == 2:  # Width
                var.width = max(1, int(value))
            elif col == 3:  # Decimals
                var.decimals = max(0, int(value))
            elif col == 4:  # Label
                var.label = str(value)
            elif col == 7:  # Columns (= column_width)
                var.column_width = max(1, int(value))
            elif col == 8:  # Align — SPSS 표시 이름으로 받음
                align_str = _ALIGN_TO_STR.get(str(value), "right")
                var.align = align_str
            elif col == 9:  # Measure — SPSS 표시 이름으로 받음
                new_measure = _MEASURE_REVERSE.get(str(value))
                if new_measure is not None:
                    var.measure = new_measure
            elif col == 10:  # Role — SPSS 표시 이름으로 받음
                new_role = _ROLE_REVERSE.get(str(value))
                if new_role is not None:
                    var.role = new_role

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
        self._dataset: Dataset | None = None
        self._model: VariablePropertiesModel | None = None
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
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
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

        # SPSS 스타일 delegate (콤보박스/스핀박스)
        self._var_delegate = VariableViewDelegate(self.table)
        self.table.setItemDelegate(self._var_delegate)

        # 편집 트리거: 더블클릭 또는 현재 항목 활성화 시
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed
        )

        # Values(5), Missing(6) 셀은 더블클릭 시 전용 다이얼로그로 처리
        self.table.doubleClicked.connect(self._on_cell_double_clicked)

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

    def _on_cell_double_clicked(self, index: QModelIndex) -> None:
        """Values(5) 또는 Missing(6) 셀 더블클릭 시 전용 다이얼로그를 엽니다."""
        col = index.column()
        if col == 5:
            self._show_values_dialog(index.row())
        elif col == 6:
            self._show_missing_dialog(index.row())

    def _show_values_dialog(self, row: int) -> None:
        """값 라벨 편집 다이얼로그를 엽니다."""
        if self._model is None or row >= len(self._model._variables):
            return
        from statworkbench.ui.dialogs.variable_editor import ValueLabelsDialog
        var = self._model._variables[row]
        dlg = ValueLabelsDialog(dict(var.value_labels), self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            var.value_labels = dlg.get_value_labels()
            idx = self._model.index(row, 5)
            self._model.dataChanged.emit(idx, idx)
            self._model.data_changed.emit()

    def _show_missing_dialog(self, row: int) -> None:
        """결측값 규칙 편집 다이얼로그를 엽니다."""
        if self._model is None or row >= len(self._model._variables):
            return
        from statworkbench.ui.dialogs.variable_editor import MissingValuesDialog
        var = self._model._variables[row]
        dlg = MissingValuesDialog(list(var.missing_values), self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            var.missing_values = dlg.get_missing_values()
            idx = self._model.index(row, 6)
            self._model.dataChanged.emit(idx, idx)
            self._model.data_changed.emit()
