"""VariableMeta list backed QAbstractTableModel for StatWorkbench."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, Role


# ── Internationalized column headers ───────────────────────────────────────

#: Variable View column definitions: (header_key, english_label, korean_label)
COLUMN_DEFS: list[tuple[str, str, str]] = [
    ("name", "Name", "이름"),
    ("label", "Label", "라벨"),
    ("type", "Type", "타입"),
    ("measure", "Measure", "척도"),
    ("role", "Role", "역할"),
    ("values", "Values", "값 라벨"),
    ("missing", "Missing", "결측값"),
    ("width", "Width", "폭"),
    ("decimals", "Decimals", "소수점"),
    ("unit", "Unit", "단위"),
    ("range", "Range", "범위"),
]


class VariableTableModel(QAbstractTableModel):
    """VariableMeta 목록을 QTableView에 표시하는 모델.

    SPSS Variable View 스타일로, 행 하나가 변수 하나를 나타냅니다.
    각 열은 변수의 속성(Name, Label, Type, Measure, Role 등)을 표시합니다.
    """

    # Column index constants
    COL_NAME = 0
    COL_LABEL = 1
    COL_TYPE = 2
    COL_MEASURE = 3
    COL_ROLE = 4
    COL_VALUES = 5
    COL_MISSING = 6
    COL_WIDTH = 7
    COL_DECIMALS = 8
    COL_UNIT = 9
    COL_RANGE = 10

    variable_added = Signal(int)  # row index
    variable_removed = Signal(int)  # row index
    variable_updated = Signal(int)  # row index
    variables_changed = Signal()

    def __init__(
        self,
        variables: list[VariableMeta],
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._variables: list[VariableMeta] = list(variables)

    # ── QAbstractTableModel overrides ──────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._variables)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(COLUMN_DEFS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row >= self.rowCount() or col >= self.columnCount():
            return None

        var = self._variables[row]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_data(var, col)

        if role == Qt.ItemDataRole.EditRole:
            return self._edit_data(var, col)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (self.COL_WIDTH, self.COL_DECIMALS):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip_data(var, col)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col == self.COL_NAME:
                # Bold names that have labels
                return None
            if col == self.COL_VALUES and var.has_value_labels:
                from PySide6.QtGui import QColor
                return QColor("#0066CC")
            if col == self.COL_MISSING and var.missing_values:
                from PySide6.QtGui import QColor
                return QColor("#CC6600")

        return None

    def _display_data(self, var: VariableMeta, col: int) -> str:
        """Get display string for a variable attribute."""
        if col == self.COL_NAME:
            return var.name
        if col == self.COL_LABEL:
            return var.label
        if col == self.COL_TYPE:
            return var.storage_type.value
        if col == self.COL_MEASURE:
            return var.measure.value
        if col == self.COL_ROLE:
            return var.role.value
        if col == self.COL_VALUES:
            if var.has_value_labels:
                return f"{{{len(var.value_labels)} items}}"
            return ""
        if col == self.COL_MISSING:
            if var.missing_values:
                return f"[{len(var.missing_values)} rules]"
            return ""
        if col == self.COL_WIDTH:
            return str(var.width)
        if col == self.COL_DECIMALS:
            return str(var.decimals)
        if col == self.COL_UNIT:
            return var.unit
        if col == self.COL_RANGE:
            parts: list[str] = []
            if var.allowed_min is not None:
                parts.append(f"{var.allowed_min}")
            else:
                parts.append("-")
            parts.append("~")
            if var.allowed_max is not None:
                parts.append(f"{var.allowed_max}")
            else:
                parts.append("-")
            return " ".join(parts)
        return ""

    def _edit_data(self, var: VariableMeta, col: int) -> Any:
        """Get editable value for a variable attribute."""
        if col == self.COL_NAME:
            return var.name
        if col == self.COL_LABEL:
            return var.label
        if col == self.COL_TYPE:
            return var.storage_type
        if col == self.COL_MEASURE:
            return var.measure
        if col == self.COL_ROLE:
            return var.role
        if col == self.COL_VALUES:
            return dict(var.value_labels)
        if col == self.COL_MISSING:
            return list(var.missing_values)
        if col == self.COL_WIDTH:
            return var.width
        if col == self.COL_DECIMALS:
            return var.decimals
        if col == self.COL_UNIT:
            return var.unit
        if col == self.COL_RANGE:
            return (var.allowed_min, var.allowed_max)
        return None

    def _tooltip_data(self, var: VariableMeta, col: int) -> str:
        """Get tooltip text for a variable attribute."""
        if col == self.COL_VALUES and var.has_value_labels:
            lines = [f"{k!r} = {v!r}" for k, v in var.value_labels.items()]
            return "Value Labels:\n" + "\n".join(lines[:10])
        if col == self.COL_MISSING and var.missing_values:
            return f"Missing rules: {var.missing_values!r}"
        if col == self.COL_NAME:
            return f"Variable: {var.name}\nLabel: {var.label}\nType: {var.storage_type.value}\nMeasure: {var.measure.value}"
        return ""

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(COLUMN_DEFS):
                # Return English labels (default)
                return COLUMN_DEFS[section][1]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        row, col = index.row(), index.column()
        if row >= len(self._variables):
            return False

        var = self._variables[row]

        try:
            if col == self.COL_NAME:
                var.name = str(value).strip()
            elif col == self.COL_LABEL:
                var.label = str(value).strip()
            elif col == self.COL_TYPE:
                if isinstance(value, StorageType):
                    var.storage_type = value
                else:
                    var.storage_type = StorageType(str(value))
            elif col == self.COL_MEASURE:
                if isinstance(value, MeasureType):
                    var.measure = value
                else:
                    var.measure = MeasureType(str(value))
            elif col == self.COL_ROLE:
                if isinstance(value, Role):
                    var.role = value
                else:
                    var.role = Role(str(value))
            elif col == self.COL_VALUES and isinstance(value, dict):
                var.value_labels = value
            elif col == self.COL_MISSING and isinstance(value, list):
                var.missing_values = value
            elif col == self.COL_WIDTH:
                var.width = int(value)
            elif col == self.COL_DECIMALS:
                var.decimals = int(value)
            elif col == self.COL_UNIT:
                var.unit = str(value)
            elif col == self.COL_RANGE:
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    var.allowed_min = float(value[0]) if value[0] is not None else None
                    var.allowed_max = float(value[1]) if value[1] is not None else None
            else:
                return False

            self.dataChanged.emit(index, index)
            self.variable_updated.emit(row)
            self.variables_changed.emit()
            return True
        except (ValueError, TypeError):
            return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        # All columns are editable
        return base_flags | Qt.ItemFlag.ItemIsEditable

    # ── Variable management ────────────────────────────────────────────────

    def add_variable(self, var: VariableMeta) -> None:
        """변수를 추가합니다."""
        row = len(self._variables)
        self.beginInsertRows(QModelIndex(), row, row)
        self._variables.append(var)
        self.endInsertRows()
        self.variable_added.emit(row)
        self.variables_changed.emit()

    def remove_variable(self, row: int) -> None:
        """지정된 행의 변수를 삭제합니다."""
        if 0 <= row < len(self._variables):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._variables.pop(row)
            self.endRemoveRows()
            self.variable_removed.emit(row)
            self.variables_changed.emit()

    def get_variable(self, row: int) -> VariableMeta:
        """지정된 행의 VariableMeta를 반환합니다."""
        if 0 <= row < len(self._variables):
            return self._variables[row]
        raise IndexError(f"Variable row {row} out of range (0-{len(self._variables) - 1})")

    def update_variable(self, row: int, var: VariableMeta) -> None:
        """지정된 행의 변수를 교체합니다."""
        if 0 <= row < len(self._variables):
            self._variables[row] = var
            top_left = self.index(row, 0)
            bottom_right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right)
            self.variable_updated.emit(row)
            self.variables_changed.emit()

    def get_all_variables(self) -> list[VariableMeta]:
        """모든 변수 목록을 반환합니다."""
        return list(self._variables)

    def set_variables(self, variables: list[VariableMeta]) -> None:
        """변수 목록 전체를 교체합니다."""
        self.beginResetModel()
        self._variables = list(variables)
        self.endResetModel()
        self.variables_changed.emit()

    def clear(self) -> None:
        """모든 변수를 삭제합니다."""
        self.beginResetModel()
        self._variables = []
        self.endResetModel()
        self.variables_changed.emit()

    def get_variable_names(self) -> list[str]:
        """모든 변수명을 반환합니다."""
        return [v.name for v in self._variables]

    def get_variables_by_measure(
        self,
        measures: list[MeasureType],
    ) -> list[VariableMeta]:
        """지정된 척도를 가진 변수들을 필터링합니다."""
        return [v for v in self._variables if v.measure in measures]

    def get_variables_by_role(
        self,
        roles: list[Role],
    ) -> list[VariableMeta]:
        """지정된 역할을 가진 변수들을 필터링합니다."""
        return [v for v in self._variables if v.role in roles]
