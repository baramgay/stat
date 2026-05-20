"""SPSS-style grid data model.

Virtual grid: cells exist conceptually but only store data when actually entered.
- Empty cells: no memory used
- Value cells: stored in DataFrame
- Header (variable name) editable
- Dynamic variable creation on first data entry
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont

from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType

logger = logging.getLogger(__name__)


def generate_var_name(index: int) -> str:
    """Generate SPSS-style variable name (VAR00001, VAR00002, ...)."""
    return f"VAR{index:05d}"


def infer_storage_type(value: Any) -> StorageType:
    """Infer storage type from a value."""
    if isinstance(value, bool):
        return StorageType.INTEGER
    if isinstance(value, int):
        return StorageType.INTEGER
    if isinstance(value, float):
        return StorageType.FLOAT
    if isinstance(value, str):
        try:
            float(value)
            if '.' in value:
                return StorageType.FLOAT
            return StorageType.INTEGER
        except (ValueError, TypeError):
            return StorageType.STRING
    return StorageType.STRING


def infer_measure_type(series: pd.Series) -> MeasureType:
    """Infer measure type from a pandas Series.

    SPSS 호환 규칙:
      - 숫자형 데이터 → SCALE (고유값 수 무관)
      - 문자형 데이터 → NOMINAL
    SPSS는 데이터 입력 시 BINARY/ORDINAL을 자동 설정하지 않는다.
    사용자가 Variable View에서 직접 변경해야 한다.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return MeasureType.NOMINAL

    for val in non_null:
        try:
            float(val)
        except (ValueError, TypeError):
            return MeasureType.NOMINAL

    return MeasureType.SCALE


# SPSS 스타일 색상 상수
_COLOR_MISSING_BG = QColor("#FFF3CD")      # 결측값 배경 (연한 노란색)
_COLOR_ODD_ROW = QColor("#FFFFFF")          # 홀수 행 배경
_COLOR_EVEN_ROW = QColor("#F8F9FA")         # 짝수 행 배경
_COLOR_HEADER_H = QColor("#D4E6F1")         # 수평 헤더 배경
_COLOR_HEADER_V = QColor("#E8EAF6")         # 수직 헤더 배경

# 척도 아이콘 (열 헤더에 표시)
_MEASURE_ICON = {
    "scale": "▪",
    "ordinal": "♦",
    "nominal": "●",
    "binary": "◉",
}


def _get_measure_icon(measure) -> str:
    """MeasureType에 맞는 아이콘 반환."""
    if measure is None:
        return "●"
    key = measure.value.lower() if hasattr(measure, "value") else str(measure).lower()
    return _MEASURE_ICON.get(key, "●")


class SPSSGridModel(QAbstractTableModel):
    """SPSS Data View style grid model.

    Key behaviors (matching SPSS):
    - Empty grid at start (no variables)
    - Variables created automatically when user enters data
    - Variable names: VAR00001, VAR00002, ...
    - Arrow/Enter/Tab navigation moves to adjacent cells
    - Enter on last row creates new row
    - Missing numeric values shown as "."
    - show_value_labels toggle support
    """

    data_changed = Signal()
    variable_added = Signal(str)   # var_name
    variable_renamed = Signal(str, str)  # old_name, new_name

    # Default virtual grid size
    DEFAULT_ROWS = 1000
    DEFAULT_COLS = 100

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        variables: Optional[dict[str, VariableMeta]] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)

        if dataframe is not None and not dataframe.empty:
            self._dataframe = dataframe.copy()
            self._var_counter = len(dataframe.columns) + 1
        else:
            self._dataframe = pd.DataFrame()
            self._var_counter = 1

        self._variables: dict[str, VariableMeta] = variables or {}

        # 값 라벨 표시 모드 (SPSS "View > Value Labels")
        self.show_value_labels: bool = False

        self._last_data_row = -1
        self._update_last_data_row()

    def _update_last_data_row(self) -> None:
        """Update the last row index that contains data."""
        if self._dataframe.empty:
            self._last_data_row = -1
            return

        for row in range(len(self._dataframe) - 1, -1, -1):
            if not self._dataframe.iloc[row].isna().all():
                self._last_data_row = row
                return
        self._last_data_row = -1

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _is_numeric_col(self, col: int) -> bool:
        """숫자형 열인지 확인."""
        if col >= len(self._dataframe.columns):
            return False
        col_name = self._dataframe.columns[col]
        var = self._variables.get(col_name)
        if var is not None:
            return var.storage_type in (StorageType.FLOAT, StorageType.INTEGER)
        # 메타 없으면 데이터로 판단
        series = self._dataframe.iloc[:, col].dropna()
        if series.empty:
            return False
        return pd.api.types.is_numeric_dtype(series)

    def _get_decimals(self, col: int) -> int:
        """열의 소수점 자릿수 반환 (기본 2)."""
        if col >= len(self._dataframe.columns):
            return 2
        col_name = self._dataframe.columns[col]
        var = self._variables.get(col_name)
        if var is not None and hasattr(var, "decimals") and var.decimals is not None:
            return var.decimals
        return 2

    def _format_display(self, value: Any, col: int) -> str:
        """DisplayRole 포맷 적용."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            # 결측값: 숫자형 → ".", 문자형 → ""
            return "." if self._is_numeric_col(col) else ""

        try:
            if pd.isna(value):
                return "." if self._is_numeric_col(col) else ""
        except (TypeError, ValueError):
            pass

        if isinstance(value, float):
            decimals = self._get_decimals(col)
            return f"{value:.{decimals}f}"
        if isinstance(value, int):
            # 정수형이면 소수점 없이
            return str(value)
        return str(value)

    def _get_value_label(self, value: Any, col: int) -> Optional[str]:
        """값 라벨 모드에서 표시할 텍스트 반환 (없으면 None)."""
        if col >= len(self._dataframe.columns):
            return None
        col_name = self._dataframe.columns[col]
        var = self._variables.get(col_name)
        if var is None or not hasattr(var, "value_labels") or not var.value_labels:
            return None
        key = str(value)
        return var.value_labels.get(key) or var.value_labels.get(value)

    # ── QAbstractTableModel overrides ────────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        if self._last_data_row >= 0:
            return max(self.DEFAULT_ROWS, self._last_data_row + 10)
        return self.DEFAULT_ROWS

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        n_cols = len(self._dataframe.columns)
        if n_cols > 0:
            return max(self.DEFAULT_COLS, n_cols + 10)
        return self.DEFAULT_COLS

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        # 실제 데이터 범위 밖 → 빈 셀
        outside = row >= len(self._dataframe) or col >= len(self._dataframe.columns)

        if outside:
            if role == Qt.ItemDataRole.DisplayRole:
                return "." if self._is_numeric_col(col) else ""
            if role == Qt.ItemDataRole.EditRole:
                return ""
            if role == Qt.ItemDataRole.TextAlignmentRole:
                if self._is_numeric_col(col):
                    return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            # 빈 셀 배경: 결측 처리
            if role == Qt.ItemDataRole.BackgroundRole:
                if self._is_numeric_col(col):
                    return QBrush(_COLOR_MISSING_BG)
                return None
            return None

        value = self._dataframe.iloc[row, col]
        is_missing = (value is None or value is pd.NA
                      or (isinstance(value, float) and np.isnan(value)))
        try:
            if pd.isna(value):
                is_missing = True
        except (TypeError, ValueError):
            pass

        if role == Qt.ItemDataRole.DisplayRole:
            if is_missing:
                return "." if self._is_numeric_col(col) else ""
            # 값 라벨 모드
            if self.show_value_labels:
                label = self._get_value_label(value, col)
                if label is not None:
                    return label
            return self._format_display(value, col)

        if role == Qt.ItemDataRole.EditRole:
            if is_missing:
                return ""
            if isinstance(value, float):
                return str(value)
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if self._is_numeric_col(col):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.BackgroundRole:
            if is_missing and self._is_numeric_col(col):
                return QBrush(_COLOR_MISSING_BG)
            # 홀/짝 행 교차색
            if row % 2 == 0:
                return QBrush(_COLOR_ODD_ROW)
            return QBrush(_COLOR_EVEN_ROW)

        if role == Qt.ItemDataRole.ToolTipRole:
            if col < len(self._dataframe.columns):
                col_name = self._dataframe.columns[col]
                var = self._variables.get(col_name)
                label = var.label if var and var.label else col_name
                return f"{label} (행 {row + 1}): {value}"
            return None

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if section < len(self._dataframe.columns):
                    col_name = str(self._dataframe.columns[section])
                    var = self._variables.get(col_name)
                    if var is not None and hasattr(var, "measure"):
                        icon = _get_measure_icon(var.measure)
                        return f"{icon} {col_name}"
                    return col_name
                return ""
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if orientation == Qt.Orientation.Horizontal:
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.BackgroundRole:
            if orientation == Qt.Orientation.Horizontal:
                return QBrush(_COLOR_HEADER_H)
            return QBrush(_COLOR_HEADER_V)

        if role == Qt.ItemDataRole.FontRole:
            font = QFont()
            font.setBold(True)
            return font

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

        if col >= len(self._dataframe.columns):
            self._create_variable_at_col(col)

        if row >= len(self._dataframe):
            self._extend_rows(row + 1)

        col_name = self._dataframe.columns[col]
        old_value = self._dataframe.iloc[row, col]

        try:
            if value == "" or value is None:
                new_value = pd.NA
            else:
                new_value = self._parse_value(value, old_value)

            self._dataframe.iloc[row, col] = new_value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.BackgroundRole])
            self._last_data_row = max(self._last_data_row, row)
            self._update_variable_metadata(col_name)
            self.data_changed.emit()

            return True
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to set data at (%d, %d): %s", row, col, exc)
            return False

    def _parse_value(self, value: Any, old_value: Any) -> Any:
        """Parse input value, trying to maintain consistent typing."""
        str_value = str(value).strip()

        # SPSS 결측 기호 "." → NA
        if str_value == ".":
            return pd.NA

        try:
            return int(str_value)
        except ValueError:
            pass

        try:
            return float(str_value)
        except ValueError:
            pass

        return str_value

    def _create_variable_at_col(self, col: int) -> None:
        """Create a new variable at the specified column index.

        beginResetModel 대신 beginInsertColumns/headerDataChanged를 사용해
        뷰의 현재 인덱스(포커스/커서)를 보존한다.
        """
        while len(self._dataframe.columns) <= col:
            var_name = generate_var_name(self._var_counter)
            while var_name in self._dataframe.columns:
                self._var_counter += 1
                var_name = generate_var_name(self._var_counter)
            self._var_counter += 1

            col_idx = len(self._dataframe.columns)
            old_virtual = self.columnCount()
            # 새 열 추가 후 columnCount가 얼마가 될지 미리 계산
            new_virtual = max(self.DEFAULT_COLS, col_idx + 11)
            needs_insert = new_virtual > old_virtual

            if needs_insert:
                self.beginInsertColumns(QModelIndex(), old_virtual, new_virtual - 1)

            if len(self._dataframe) == 0:
                self._dataframe = pd.DataFrame({var_name: [pd.NA]})
            else:
                self._dataframe[var_name] = pd.NA

            var_meta = VariableMeta(
                name=var_name,
                label=var_name,
                storage_type=StorageType.STRING,
                measure=MeasureType.NOMINAL,
            )
            self._variables[var_name] = var_meta

            if needs_insert:
                self.endInsertColumns()
            else:
                # 헤더 텍스트만 갱신 (가상 그리드 크기 불변)
                self.headerDataChanged.emit(Qt.Orientation.Horizontal, col_idx, col_idx)

            self.variable_added.emit(var_name)
            logger.info("Created variable: %s", var_name)

    def _update_variable_metadata(self, var_name: str) -> None:
        """Update variable metadata based on actual data."""
        if var_name not in self._variables:
            return

        series = self._dataframe[var_name]
        non_null = series.dropna()

        if len(non_null) == 0:
            return

        var_meta = self._variables[var_name]

        first_value = non_null.iloc[0]
        var_meta.storage_type = infer_storage_type(first_value)
        var_meta.measure = infer_measure_type(series)

        if var_meta.storage_type == StorageType.FLOAT:
            max_decimals = 0
            for val in non_null:
                try:
                    str_val = str(val)
                    if '.' in str_val:
                        decimals = len(str_val.split('.')[1])
                        max_decimals = max(max_decimals, decimals)
                except Exception:
                    pass
            var_meta.decimals = min(max_decimals, 4) if max_decimals > 0 else 2

        var_meta.touch()

    def setHeaderData(
        self,
        section: int,
        orientation: Qt.Orientation,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.EditRole:
            return False

        new_name = str(value).strip()
        if not new_name:
            return False

        if section >= len(self._dataframe.columns):
            self._create_variable_at_col(section)

        old_name = self._dataframe.columns[section]
        if old_name == new_name:
            return True

        if new_name in self._dataframe.columns:
            logger.warning("Variable name '%s' already exists", new_name)
            return False

        self._dataframe.rename(columns={old_name: new_name}, inplace=True)

        if old_name in self._variables:
            var_meta = self._variables.pop(old_name)
            var_meta.name = new_name
            var_meta.label = new_name
            self._variables[new_name] = var_meta

        self.headerDataChanged.emit(orientation, section, section)
        self.variable_renamed.emit(old_name, new_name)
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    # ── Row/Column extension ─────────────────────────────────────────────────

    def _extend_rows(self, target_rows: int) -> None:
        """Extend rows to target_rows."""
        current_rows = len(self._dataframe)
        if target_rows <= current_rows:
            return

        n_new = target_rows - current_rows
        self.beginInsertRows(QModelIndex(), current_rows, target_rows - 1)

        new_data = {}
        for col in self._dataframe.columns:
            new_data[col] = [pd.NA] * n_new
        new_df = pd.DataFrame(new_data)
        self._dataframe = pd.concat([self._dataframe, new_df], ignore_index=True)

        self.endInsertRows()

    # ── Public helpers ───────────────────────────────────────────────────────

    def set_dataframe(self, dataframe: pd.DataFrame, variables: Optional[dict[str, VariableMeta]] = None) -> None:
        """Replace DataFrame and optionally variables."""
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self._var_counter = len(dataframe.columns) + 1
        if variables is not None:
            self._variables = variables.copy()
        self._update_last_data_row()
        self.endResetModel()
        self.data_changed.emit()

    def get_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with all columns and data-containing rows.

        dtype은 variables 메타데이터 기준으로 변환 — object 컬럼에 숫자가 담긴
        경우 is_numeric_dtype 체크가 False가 되는 문제를 방지한다.
        """
        if self._dataframe.empty:
            return self._dataframe.copy()

        if self._last_data_row >= 0:
            df = self._dataframe.iloc[:self._last_data_row + 1].copy()
        elif len(self._dataframe.columns) > 0:
            df = self._dataframe.iloc[:0].copy()
        else:
            return pd.DataFrame()

        # variables 메타데이터 기준으로 수치형 컬럼 dtype 변환
        for col_name in df.columns:
            var_meta = self._variables.get(col_name)
            if var_meta is not None and var_meta.storage_type in (
                StorageType.FLOAT, StorageType.INTEGER
            ):
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        return df

    def get_full_dataframe(self) -> pd.DataFrame:
        """Return full DataFrame."""
        return self._dataframe.copy()

    def get_variables(self) -> dict[str, VariableMeta]:
        """Return variable metadata dictionary."""
        return self._variables.copy()

    def toggle_value_labels(self) -> bool:
        """값 라벨 표시 토글. 변경 후 현재 상태 반환."""
        self.show_value_labels = not self.show_value_labels
        # 전체 뷰 갱신
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole])
        return self.show_value_labels

    def add_row(self, values: Optional[dict[str, Any]] = None) -> None:
        """Add a new row."""
        current_rows = len(self._dataframe)
        self.beginInsertRows(QModelIndex(), current_rows, current_rows)

        new_row = values if values is not None else {}
        for col in self._dataframe.columns:
            if col not in new_row:
                new_row[col] = pd.NA

        new_df = pd.DataFrame([new_row])
        self._dataframe = pd.concat([self._dataframe, new_df], ignore_index=True)

        self.endInsertRows()
        self.data_changed.emit()
        self._update_last_data_row()

        if values is None and self._last_data_row < current_rows:
            self._last_data_row = current_rows

    def remove_row(self, row: int) -> bool:
        """Remove a row."""
        if 0 <= row < len(self._dataframe):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._dataframe = self._dataframe.drop(self._dataframe.index[row]).reset_index(drop=True)
            self.endRemoveRows()
            self.data_changed.emit()
            self._update_last_data_row()
            return True
        return False

    def add_column(self, name: str, values: Optional[list[Any]] = None) -> None:
        """Add a new column."""
        col_idx = len(self._dataframe.columns)
        old_virtual = self.columnCount()
        new_virtual = max(self.DEFAULT_COLS, col_idx + 11)
        needs_insert = new_virtual > old_virtual

        if needs_insert:
            self.beginInsertColumns(QModelIndex(), old_virtual, new_virtual - 1)

        if values is not None:
            self._dataframe[name] = values
        else:
            self._dataframe[name] = pd.NA

        if name not in self._variables:
            self._variables[name] = VariableMeta(
                name=name,
                label=name,
                storage_type=StorageType.STRING,
                measure=MeasureType.NOMINAL,
            )

        if needs_insert:
            self.endInsertColumns()
        else:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, col_idx, col_idx)

        self.data_changed.emit()

    def remove_column(self, col: int) -> bool:
        """Remove a column."""
        if 0 <= col < len(self._dataframe.columns):
            col_name = self._dataframe.columns[col]
            old_virtual = self.columnCount()
            # 제거 후 가상 크기 예측 (제거 전 컬럼 수 - 1 기준)
            remaining = len(self._dataframe.columns) - 1
            new_virtual = max(self.DEFAULT_COLS, remaining + 11) if remaining > 0 else self.DEFAULT_COLS

            if new_virtual < old_virtual:
                self.beginRemoveColumns(QModelIndex(), new_virtual, old_virtual - 1)

            self._dataframe = self._dataframe.drop(columns=[col_name])
            if col_name in self._variables:
                del self._variables[col_name]

            if new_virtual < old_virtual:
                self.endRemoveColumns()
            else:
                self.headerDataChanged.emit(Qt.Orientation.Horizontal, col, self.columnCount() - 1)

            self.data_changed.emit()
            return True
        return False
