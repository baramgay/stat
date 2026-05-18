"""SPSS 스타일 격자 데이터 모델.

기본적으로 1000행 x 100열의 가상 격자를 제공합니다.
실제 데이터는 낭비 없이 효율적으로 저장됩니다.
- 빈 셀: 메모리에 저장하지 않음
- 값이 있는 셀만 DataFrame에 저장
- 헤더(변수명) 편집 가능
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Dict

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

logger = logging.getLogger(__name__)


def generate_var_name(index: int) -> str:
    """SPSS 스타일 변수명 생성 (VAR00001, VAR00002, ...)."""
    return f"VAR{index:05d}"


class SPSSGridModel(QAbstractTableModel):
    """SPSS Data View 스타일 격자 모델.
    
    특징:
    - 기본 1000행 x 100열 가상 격자
    - 값이 있는 셀만 실제로 저장
    - 헤더(변수명) 직접 편집 가능
    - 동적 행/열 확장
    """
    
    data_changed = Signal()
    variable_added = Signal(str)
    variable_renamed = Signal(str, str)  # old_name, new_name
    
    # 기본 격자 크기
    DEFAULT_ROWS = 1000
    DEFAULT_COLS = 100
    
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        
        # 실제 데이터 저장소
        if dataframe is not None and not dataframe.empty:
            self._dataframe = dataframe.copy()
            self._var_counter = len(dataframe.columns) + 1
        else:
            # 빈 DataFrame으로 시작 (열 없음)
            self._dataframe = pd.DataFrame()
            self._var_counter = 1
        
        # 실제 데이터가 있는 마지막 행/열 인덱스 (0-based)
        self._last_data_row = -1
        self._last_data_col = -1
        self._update_last_data_indices()
    
    def _update_last_data_indices(self) -> None:
        """실제 데이터가 있는 마지막 인덱스를 업데이트합니다."""
        if self._dataframe.empty:
            self._last_data_row = -1
            self._last_data_col = -1
            return
        
        # 마지막으로 값이 있는 행 찾기
        last_row = -1
        for row in range(len(self._dataframe) - 1, -1, -1):
            if not self._dataframe.iloc[row].isna().all():
                last_row = row
                break
        self._last_data_row = last_row
        
        # 마지막으로 값이 있는 열 찾기
        last_col = -1
        for col in range(len(self._dataframe.columns) - 1, -1, -1):
            if not self._dataframe.iloc[:, col].isna().all():
                last_col = col
                break
        self._last_data_col = last_col
    
    # ── QAbstractTableModel overrides ──────────────────────────────────────
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        # 실제 데이터가 있으면 그것 + 여유, 없으면 기본 크기
        if self._last_data_row >= 0:
            return max(self.DEFAULT_ROWS, self._last_data_row + 10)
        return self.DEFAULT_ROWS
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        # 실제 열이 있으면 그것 + 여유, 없으면 기본 크기
        n_cols = len(self._dataframe.columns)
        if n_cols > 0:
            return max(self.DEFAULT_COLS, n_cols + 10)
        return self.DEFAULT_COLS
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        
        row, col = index.row(), index.column()
        
        # 실제 데이터 범위 밖이면 빈 셀
        if row >= len(self._dataframe) or col >= len(self._dataframe.columns):
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
                return ""
            # Format floats nicely
            if isinstance(value, float):
                if value != value:  # NaN check
                    return ""
                if role == Qt.ItemDataRole.DisplayRole:
                    return f"{value:.4f}".rstrip("0").rstrip(".")
                return str(value)
            return str(value)
        
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        
        if role == Qt.ItemDataRole.ToolTipRole:
            if col < len(self._dataframe.columns):
                col_name = self._dataframe.columns[col]
                return f"{col_name}: 행 {row + 1}"
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
                    return str(self._dataframe.columns[section])
                # 빈 열 헤더
                return f"VAR{section + 1:05d}"
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
        
        # 데이터 확장이 필요한지 확인
        needs_reset = False
        
        # 열이 부족하면 확장
        if col >= len(self._dataframe.columns):
            self._extend_columns(col + 1)
            needs_reset = True
        
        # 행이 부족하면 확장
        if row >= len(self._dataframe):
            self._extend_rows(row + 1)
            needs_reset = True
        
        if needs_reset:
            # 확장 후에도 계속 진행 (뷰가 자동으로 다시 호출)
            pass
        
        col_name = self._dataframe.columns[col]
        old_value = self._dataframe.iloc[row, col]
        
        try:
            if value == "" or value is None:
                new_value = pd.NA
            elif isinstance(old_value, (int, float)) and not pd.isna(old_value):
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
            self._update_last_data_indices()
            return True
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to set data at (%d, %d): %s", row, col, exc)
            return False
    
    def setHeaderData(
        self,
        section: int,
        orientation: Qt.Orientation,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """헤더(변수명) 편집."""
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.EditRole:
            return False
        
        new_name = str(value).strip()
        if not new_name:
            return False
        
        # 열이 없으면 생성
        if section >= len(self._dataframe.columns):
            self._extend_columns(section + 1)
        
        old_name = self._dataframe.columns[section]
        if old_name == new_name:
            return True
        
        # 중복 검사
        if new_name in self._dataframe.columns:
            logger.warning("Variable name '%s' already exists", new_name)
            return False
        
        self._dataframe.rename(columns={old_name: new_name}, inplace=True)
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
    
    def headerFlags(self, section: int, orientation: Qt.Orientation) -> Qt.ItemFlag:
        """헤더 플래그."""
        if orientation == Qt.Orientation.Horizontal:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable
        return Qt.ItemFlag.ItemIsEnabled
    
    # ── 낭비 확장 ──────────────────────────────────────────────────────────
    
    def _extend_rows(self, target_rows: int) -> None:
        """행을 target_rows까지 확장합니다."""
        current_rows = len(self._dataframe)
        if target_rows <= current_rows:
            return
        
        n_new = target_rows - current_rows
        self.beginInsertRows(QModelIndex(), current_rows, target_rows - 1)
        
        # 새 행 추가
        new_data = {}
        for col in self._dataframe.columns:
            new_data[col] = [pd.NA] * n_new
        new_df = pd.DataFrame(new_data)
        self._dataframe = pd.concat([self._dataframe, new_df], ignore_index=True)
        
        self.endInsertRows()
    
    def _extend_columns(self, target_cols: int) -> None:
        """열을 target_cols까지 확장합니다."""
        current_cols = len(self._dataframe.columns)
        if target_cols <= current_cols:
            return
        
        self.beginResetModel()
        
        n_new = target_cols - current_cols
        for i in range(n_new):
            var_name = generate_var_name(self._var_counter)
            while var_name in self._dataframe.columns:
                self._var_counter += 1
                var_name = generate_var_name(self._var_counter)
            self._var_counter += 1
            
            if len(self._dataframe) == 0:
                self._dataframe = pd.DataFrame({var_name: [pd.NA]})
            else:
                self._dataframe[var_name] = pd.NA
            
            self.variable_added.emit(var_name)
        
        self.endResetModel()
    
    # ── Public helpers ─────────────────────────────────────────────────────
    
    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """DataFrame을 교체합니다."""
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self._var_counter = len(dataframe.columns) + 1
        self._update_last_data_indices()
        self.endResetModel()
        self.data_changed.emit()
    
    def get_dataframe(self) -> pd.DataFrame:
        """현재 DataFrame을 반환합니다 (빈 행/열 제거)."""
        # 실제 데이터가 있는 범위만 반환
        if self._dataframe.empty:
            return self._dataframe.copy()
        
        # 마지막 데이터 행/열까지 자르기
        last_row = self._last_data_row
        last_col = self._last_data_col
        
        if last_row < 0 or last_col < 0:
            return pd.DataFrame()
        
        result = self._dataframe.iloc[:last_row + 1, :last_col + 1].copy()
        return result
    
    def get_full_dataframe(self) -> pd.DataFrame:
        """전체 DataFrame을 반환합니다."""
        return self._dataframe.copy()
    
    def add_row(self, values: Optional[Dict[str, Any]] = None) -> None:
        """새 행을 추가합니다."""
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
        self._update_last_data_indices()
    
    def remove_row(self, row: int) -> bool:
        """행을 삭제합니다."""
        if 0 <= row < len(self._dataframe):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._dataframe = self._dataframe.drop(self._dataframe.index[row]).reset_index(drop=True)
            self.endRemoveRows()
            self.data_changed.emit()
            self._update_last_data_indices()
            return True
        return False
    
    def add_column(self, name: str, values: Optional[list[Any]] = None) -> None:
        """새 열을 추가합니다."""
        self.beginResetModel()
        
        if values is not None:
            self._dataframe[name] = values
        else:
            self._dataframe[name] = pd.NA
        
        self.endResetModel()
        self.data_changed.emit()
        self._update_last_data_indices()
    
    def remove_column(self, col: int) -> bool:
        """열을 삭제합니다."""
        if 0 <= col < len(self._dataframe.columns):
            self.beginResetModel()
            col_name = self._dataframe.columns[col]
            self._dataframe = self._dataframe.drop(columns=[col_name])
            self.endResetModel()
            self.data_changed.emit()
            self._update_last_data_indices()
            return True
        return False
