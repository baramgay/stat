"""pandas DataFrame backed QAbstractTableModel for StatWorkbench.

SPSS 스타일 데이터 편집을 지원합니다.
- 동적 행/열 추가
- 자동 변수명 생성 (VAR00001)
- 셀 단위 편집
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

logger = logging.getLogger(__name__)


def generate_var_name(index: int) -> str:
    """SPSS 스타일 변수명 생성 (VAR00001, VAR00002, ...)."""
    return f"VAR{index:05d}"


class DataFrameTableModel(QAbstractTableModel):
    """pandas DataFrame을 PySide6 QTableView에 표시하는 모델.

    SPSS Data View 스타일:
    - 셀 단위 편집
    - 빈 셀에 데이터 입력 시 자동으로 행/열 확장
    - 변수명 자동 생성 (VAR00001 형식)
    """

    data_changed = Signal()
    variable_added = Signal(str)  # 새 변수명

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._dataframe: pd.DataFrame = dataframe if dataframe is not None else pd.DataFrame()
        self._var_counter: int = len(self._dataframe.columns) + 1

    # ── QAbstractTableModel overrides ──────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._dataframe)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._dataframe.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row >= self.rowCount() or col >= self.columnCount():
            return None

        # 빈 DataFrame 처리 (SPSS 스타일)
        if len(self._dataframe) == 0 or len(self._dataframe.columns) == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                return ""
            if role == Qt.ItemDataRole.EditRole:
                return ""
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return None

        value = self._dataframe.iloc[row, col]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.isna(value):
                return "" if role == Qt.ItemDataRole.DisplayRole else ""
            # Format floats nicely
            if isinstance(value, float):
                if value != value:  # NaN check
                    return "" if role == Qt.ItemDataRole.DisplayRole else ""
                return f"{value:.4f}".rstrip("0").rstrip(".") if role == Qt.ItemDataRole.DisplayRole else str(value)
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.ToolTipRole:
            col_name = self._dataframe.columns[col]
            return f"{col_name}: 행 {row + 1}"

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self._dataframe.columns):
                    return str(self._dataframe.columns[section])
                return None
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)  # 1-based row numbers
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if orientation == Qt.Orientation.Horizontal:
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
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

        # SPSS 스타일: 빈 셀에 입력 시 자동 확장
        if row >= len(self._dataframe):
            self._add_empty_rows(row - len(self._dataframe) + 1)
        if col >= len(self._dataframe.columns):
            self._add_auto_columns(col - len(self._dataframe.columns) + 1)
            # 재조정 후 인덱스 확인
            if col >= len(self._dataframe.columns):
                return False

        col_name = self._dataframe.columns[col]
        old_value = self._dataframe.iloc[row, col]

        # Try to preserve dtype
        try:
            if value == "" or value is None:
                new_value = pd.NA
            elif isinstance(old_value, (int, float)) and not pd.isna(old_value):
                # Try numeric conversion
                if isinstance(old_value, int):
                    new_value = int(value)
                else:
                    new_value = float(value)
            else:
                # Try numeric first, then string
                try:
                    new_value = float(value)
                    if new_value == int(new_value):
                        new_value = int(new_value)
                except (ValueError, TypeError):
                    new_value = str(value)
            self._dataframe.iloc[row, col] = new_value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            self.data_changed.emit()
            return True
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to set data at (%d, %d): %s", row, col, exc)
            return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    # ── SPSS 스타일 동적 확장 ───────────────────────────────────────────────

    def _add_empty_rows(self, count: int) -> None:
        """빈 행을 추가합니다."""
        if count <= 0:
            return
        # 열이 없으면 먼저 기본 열 하나 생성
        if len(self._dataframe.columns) == 0:
            self._add_auto_columns(1)
        self.beginInsertRows(QModelIndex(), len(self._dataframe), len(self._dataframe) + count - 1)
        for _ in range(count):
            new_row = {col: pd.NA for col in self._dataframe.columns}
            self._dataframe.loc[len(self._dataframe)] = new_row
        self.endInsertRows()
        self.data_changed.emit()

    def _add_auto_columns(self, count: int) -> None:
        """자동 변수명으로 열을 추가합니다 (VAR00001)."""
        if count <= 0:
            return
        
        self.beginResetModel()
        for _ in range(count):
            var_name = generate_var_name(self._var_counter)
            while var_name in self._dataframe.columns:
                self._var_counter += 1
                var_name = generate_var_name(self._var_counter)
            self._var_counter += 1
            
            # 행이 없으면 빈 DataFrame에 열 추가
            if len(self._dataframe) == 0:
                self._dataframe = pd.DataFrame({var_name: [pd.NA]})
            else:
                self._dataframe[var_name] = pd.NA
            self.variable_added.emit(var_name)
        self.endResetModel()
        self.data_changed.emit()

    # ── Public helpers ─────────────────────────────────────────────────────

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """DataFrame을 교체하고 모델을 재설정합니다."""
        self.beginResetModel()
        self._dataframe = dataframe
        self._var_counter = len(dataframe.columns) + 1
        self.endResetModel()
        self.data_changed.emit()

    def get_dataframe(self) -> pd.DataFrame:
        """현재 DataFrame을 반환합니다."""
        return self._dataframe

    def get_cell_value(self, row: int, col: int) -> Any:
        """특정 셀 값을 반환합니다."""
        if 0 <= row < self.rowCount() and 0 <= col < self.columnCount():
            return self._dataframe.iloc[row, col]
        return None

    def set_cell_value(self, row: int, col: int, value: Any) -> bool:
        """특정 셀 값을 설정합니다."""
        if 0 <= row < self.rowCount() and 0 <= col < self.columnCount():
            index = self.index(row, col)
            return self.setData(index, value, Qt.ItemDataRole.EditRole)
        return False

    def add_row(self, values: Optional[dict[str, Any]] = None) -> None:
        """새 행을 추가합니다."""
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        new_row = values if values is not None else {}
        # Fill missing columns with NA
        for col in self._dataframe.columns:
            if col not in new_row:
                new_row[col] = pd.NA
        self._dataframe.loc[len(self._dataframe)] = new_row
        self.endInsertRows()
        self.data_changed.emit()

    def remove_row(self, row: int) -> bool:
        """행을 삭제합니다."""
        if 0 <= row < self.rowCount():
            self.beginRemoveRows(QModelIndex(), row, row)
            self._dataframe = self._dataframe.drop(self._dataframe.index[row]).reset_index(drop=True)
            self.endRemoveRows()
            self.data_changed.emit()
            return True
        return False

    def add_column(self, name: str, values: Optional[list[Any]] = None) -> None:
        """새 열을 추가합니다."""
        self.beginInsertColumns(QModelIndex(), self.columnCount(), self.columnCount())
        if values is not None:
            self._dataframe[name] = values
        else:
            self._dataframe[name] = [pd.NA] * max(1, self.rowCount())
        self.endInsertColumns()
        self.data_changed.emit()

    def remove_column(self, col: int) -> bool:
        """열을 삭제합니다."""
        if 0 <= col < self.columnCount():
            self.beginRemoveColumns(QModelIndex(), col, col)
            col_name = self._dataframe.columns[col]
            self._dataframe = self._dataframe.drop(columns=[col_name])
            self.endRemoveColumns()
            self.data_changed.emit()
            return True
        return False

    def sort_by_column(self, col: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """열 기준으로 데이터를 정렬합니다."""
        if 0 <= col < self.columnCount():
            col_name = self._dataframe.columns[col]
            ascending = order == Qt.SortOrder.AscendingOrder
            self.beginResetModel()
            self._dataframe = self._dataframe.sort_values(
                by=col_name, ascending=ascending, na_position="last"
            ).reset_index(drop=True)
            self.endResetModel()
            self.data_changed.emit()
