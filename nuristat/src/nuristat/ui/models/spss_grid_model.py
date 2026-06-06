"""SPSS-style grid data model.

Virtual grid: cells exist conceptually but only store data when actually entered.
- Empty cells: no memory used
- Value cells: stored in DataFrame
- Header (variable name) editable
- Dynamic variable creation on first data entry
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont

from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta

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
        dataframe: pd.DataFrame | None = None,
        variables: dict[str, VariableMeta] | None = None,
        parent: Any | None = None,
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

        # 측정 척도 자동 감지 완료 여부 추적:
        #   - 신규 변수(_create_variable_at_col): 첫 입력 시 자동 감지 예정 → set에 없음
        #   - 기존 변수(생성자/set_dataframe): 이미 타입 설정 완료 → set에 있음
        self._measure_initialized: set[str] = set(self._variables.keys())

        self._last_data_row = -1
        self._update_last_data_row()

        # Cache: col index → is_numeric bool. Invalidated on structural changes.
        self._numeric_col_cache: dict[int, bool] = {}

        # 대량 편집(붙여넣기·채우기) 배치 상태:
        #   batch_update() 컨텍스트 동안 셀별 dataChanged/data_changed 방출을 억제하고
        #   더티 영역만 추적했다가 종료 시 1회만 방출한다.
        self._batch_depth: int = 0
        self._batch_min_row = self._batch_min_col = -1
        self._batch_max_row = self._batch_max_col = -1

        # 실행 취소/다시 실행 스택 (SPSS 데이터 편집기 동등 기능)
        #   사용자 액션 1건당 스냅샷 1개를 push한다. 대량 붙여넣기/채우기는
        #   batch_update 진입 시 1회만 push되어 한 번에 되돌려진다.
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []
        self._undo_limit: int = 50

    def _invalidate_col_cache(self) -> None:
        self._numeric_col_cache.clear()

    # ── 실행 취소 / 다시 실행 ────────────────────────────────────────────────

    def _snapshot(self) -> tuple:
        """현재 상태를 되돌릴 수 있는 스냅샷으로 캡처."""
        return (
            self._dataframe.copy(deep=True),
            copy.deepcopy(self._variables),
            self._last_data_row,
            self._var_counter,
            set(self._measure_initialized),
        )

    def _push_undo(self) -> None:
        """변경 직전 상태를 undo 스택에 적재하고 redo 스택을 비운다."""
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore(self, snap: tuple) -> None:
        """스냅샷으로 모델 상태를 복원하고 뷰를 갱신한다."""
        df, variables, ldr, vc, mi = snap
        self.beginResetModel()
        self._dataframe = df
        self._variables = variables
        self._last_data_row = ldr
        self._var_counter = vc
        self._measure_initialized = mi
        self._invalidate_col_cache()
        self.endResetModel()
        self.data_changed.emit()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """마지막 사용자 액션을 취소한다."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        """취소한 액션을 다시 실행한다."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        return True

    @contextmanager
    def batch_update(self):
        """대량 셀 편집 시 셀별 갱신 신호를 억제하고 종료 시 1회만 방출.

        붙여넣기·채우기처럼 다수 셀을 연속 변경할 때 setData를 그대로 호출하되
        ``with model.batch_update():`` 블록으로 감싸면, 셀마다 발생하던
        dataChanged/data_changed(전체 DataFrame 재구축 유발)를 영역 1회로 합쳐
        O(n²) → O(n) 로 줄인다.
        """
        if self._batch_depth == 0:
            self._push_undo()   # 대량 편집을 1건의 실행 취소 단위로 묶음
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._flush_batch()

    def _track_dirty(self, row: int, col: int) -> None:
        """배치 모드에서 변경된 셀 영역(min/max)을 누적 추적."""
        if self._batch_min_row < 0:
            self._batch_min_row = self._batch_max_row = row
            self._batch_min_col = self._batch_max_col = col
        else:
            self._batch_min_row = min(self._batch_min_row, row)
            self._batch_max_row = max(self._batch_max_row, row)
            self._batch_min_col = min(self._batch_min_col, col)
            self._batch_max_col = max(self._batch_max_col, col)

    def _flush_batch(self) -> None:
        """누적된 더티 영역에 대해 dataChanged·data_changed를 1회 방출."""
        if self._batch_min_row < 0:
            return
        top_left = self.index(self._batch_min_row, self._batch_min_col)
        bottom_right = self.index(self._batch_max_row, self._batch_max_col)
        self._batch_min_row = self._batch_min_col = -1
        self._batch_max_row = self._batch_max_col = -1
        self.dataChanged.emit(
            top_left, bottom_right,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.BackgroundRole],
        )
        self.data_changed.emit()

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
        """숫자형 열인지 확인 (캐시 사용)."""
        cached = self._numeric_col_cache.get(col)
        if cached is not None:
            return cached
        if col >= len(self._dataframe.columns):
            return False
        col_name = self._dataframe.columns[col]
        var = self._variables.get(col_name)
        if var is not None:
            result = var.storage_type in (StorageType.FLOAT, StorageType.INTEGER)
        else:
            series = self._dataframe.iloc[:, col].dropna()
            result = bool(series.empty is False and pd.api.types.is_numeric_dtype(series))
        self._numeric_col_cache[col] = result
        return result

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

    def value_labels_for_col(self, col: int) -> dict | None:
        """해당 열 변수의 값 라벨 사전을 반환 (없으면 None).

        데이터 셀 편집기가 범주형 변수에 드롭다운(코드=라벨)을 제공하는 데 사용.
        """
        if col < 0 or col >= len(self._dataframe.columns):
            return None
        col_name = self._dataframe.columns[col]
        var = self._variables.get(col_name)
        labels = getattr(var, "value_labels", None) if var is not None else None
        return labels if labels else None

    def _get_value_label(self, value: Any, col: int) -> str | None:
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
        is_missing = value is None or value is pd.NA
        if not is_missing:
            if isinstance(value, float):
                is_missing = np.isnan(value)
            elif not isinstance(value, (int, str, bool)):
                try:
                    is_missing = bool(pd.isna(value))
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

        # 기존 숫자형 변수에 문자 입력 방지 (SPSS 호환)
        # 변수가 이미 존재하고 numeric 타입으로 확정된 경우에만 검증
        if col < len(self._dataframe.columns) and value not in ("", None, "."):
            existing_var = self._variables.get(self._dataframe.columns[col])
            if (existing_var is not None
                    and existing_var.storage_type in (StorageType.FLOAT, StorageType.INTEGER)):
                try:
                    float(str(value))
                except (ValueError, TypeError):
                    logger.warning(
                        "숫자형 변수 '%s'에 문자 '%s' 입력 거부 (SPSS 호환)",
                        self._dataframe.columns[col], value,
                    )
                    return False

        # 단일 셀 편집을 실행 취소 단위로 기록 (배치 모드는 진입 시 이미 push됨)
        if self._batch_depth == 0:
            self._push_undo()

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
            self._last_data_row = max(self._last_data_row, row)
            self._update_variable_metadata(col_name, new_value=new_value)

            # 배치 모드면 영역만 추적하고 신호는 종료 시 1회 방출
            if self._batch_depth > 0:
                self._track_dirty(row, col)
            else:
                self.dataChanged.emit(
                    index, index,
                    [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.BackgroundRole],
                )
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

            self._invalidate_col_cache()
            self.variable_added.emit(var_name)
            logger.info("Created variable: %s", var_name)

    def _update_variable_metadata(self, var_name: str, new_value: Any = None) -> None:
        """Update variable metadata incrementally (O(1) per cell edit).

        SPSS 호환 규칙:
          - 측정 척도(measure)는 첫 번째 데이터 입력 시에만 자동 감지.
          - _measure_initialized에 등록된 변수는 사용자 설정 보존 — 덮어쓰지 않음.
          - storage_type은 새 값 기준으로 승격(INTEGER→FLOAT→STRING).
        """
        if var_name not in self._variables:
            return

        if new_value is None or new_value is pd.NA:
            return

        var_meta = self._variables[var_name]
        new_st = infer_storage_type(new_value)

        # storage_type: STRING(초기) → INTEGER / FLOAT 로 승격, INTEGER → FLOAT 승격
        # 수치형 변수는 문자 입력 자체가 거부되므로 FLOAT/INTEGER → STRING 하향은 없음
        current_st = var_meta.storage_type
        if current_st == StorageType.STRING:
            if new_st in (StorageType.INTEGER, StorageType.FLOAT):
                var_meta.storage_type = new_st
                cols = self._dataframe.columns
                if var_name in cols:
                    self._numeric_col_cache.pop(cols.get_loc(var_name), None)
        elif current_st == StorageType.INTEGER and new_st == StorageType.FLOAT:
            var_meta.storage_type = StorageType.FLOAT

        # 측정 척도: 최초 데이터 입력 시에만 자동 감지
        if var_name not in self._measure_initialized:
            var_meta.measure = infer_measure_type(self._dataframe[var_name])
            self._measure_initialized.add(var_name)

        # 소수점 자릿수: 새 값 기준으로 현재 최댓값과 비교
        if var_meta.storage_type == StorageType.FLOAT or new_st == StorageType.FLOAT:
            try:
                str_val = str(new_value)
                if '.' in str_val:
                    new_dec = len(str_val.split('.')[1])
                    current_dec = var_meta.decimals if var_meta.decimals is not None else 2
                    var_meta.decimals = min(max(current_dec, new_dec), 4)
            except Exception:
                pass

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

        self._push_undo()
        self._dataframe.rename(columns={old_name: new_name}, inplace=True)

        if old_name in self._variables:
            var_meta = self._variables.pop(old_name)
            var_meta.name = new_name
            var_meta.label = new_name
            self._variables[new_name] = var_meta

        self._numeric_col_cache.pop(section, None)
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

    def set_dataframe(self, dataframe: pd.DataFrame, variables: dict[str, VariableMeta] | None = None) -> None:
        """Replace DataFrame and optionally variables."""
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self._var_counter = len(dataframe.columns) + 1
        if variables is not None:
            self._variables = variables.copy()
        # 새 데이터셋: 모든 변수를 초기화 완료 상태로 표시
        self._measure_initialized = set(self._variables.keys())
        self._invalidate_col_cache()
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

    def mark_measure_initialized(self, var_name: str) -> None:
        """Variable View에서 측정 척도를 사용자가 직접 설정한 변수를 초기화 완료로 표시.

        이후 데이터 입력 시 자동 감지가 사용자 설정을 덮어쓰지 않도록 보호한다.
        """
        self._measure_initialized.add(var_name)

    def sort_by_column(self, col: int, ascending: bool = True) -> None:
        """열 기준 정렬. beginResetModel 없이 dataChanged 시그널로 뷰 갱신해 포커스 보존."""
        if col >= len(self._dataframe.columns):
            return
        self._push_undo()
        col_name = self._dataframe.columns[col]
        self._dataframe.sort_values(by=col_name, ascending=ascending, inplace=True)
        self._dataframe.reset_index(drop=True, inplace=True)
        self._update_last_data_row()
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, len(self._dataframe.columns) - 1)
        self.dataChanged.emit(
            top_left, bottom_right,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.BackgroundRole],
        )
        self.data_changed.emit()

    def toggle_value_labels(self) -> bool:
        """값 라벨 표시 토글. 변경 후 현재 상태 반환."""
        self.show_value_labels = not self.show_value_labels
        # 전체 뷰 갱신
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole])
        return self.show_value_labels

    def add_row(self, values: dict[str, Any] | None = None) -> None:
        """Add a new row."""
        self._push_undo()
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
            self._push_undo()
            self.beginRemoveRows(QModelIndex(), row, row)
            self._dataframe = self._dataframe.drop(self._dataframe.index[row]).reset_index(drop=True)
            self.endRemoveRows()
            self.data_changed.emit()
            self._update_last_data_row()
            return True
        return False

    def add_column(self, name: str, values: list[Any] | None = None) -> None:
        """Add a new column."""
        self._push_undo()
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
            self._push_undo()
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
            self._invalidate_col_cache()

            if new_virtual < old_virtual:
                self.endRemoveColumns()
            else:
                self.headerDataChanged.emit(Qt.Orientation.Horizontal, col, self.columnCount() - 1)

            self.data_changed.emit()
            return True
        return False

    # ── 위치 삽입 (SPSS: 변수/케이스 삽입) ───────────────────────────────────

    def insert_row_at(self, row: int) -> bool:
        """지정 위치에 빈 행(케이스)을 삽입한다 (SPSS '케이스 삽입').

        기존 행들은 아래로 밀린다. 추가(append)만 가능하던 한계를 해소.
        """
        if len(self._dataframe.columns) == 0:
            return False
        self._push_undo()
        row = max(0, min(row, len(self._dataframe)))
        self.beginResetModel()
        blank = pd.DataFrame([{c: pd.NA for c in self._dataframe.columns}])
        self._dataframe = pd.concat(
            [self._dataframe.iloc[:row], blank, self._dataframe.iloc[row:]],
            ignore_index=True,
        )
        self._invalidate_col_cache()
        self._update_last_data_row()
        self.endResetModel()
        self.data_changed.emit()
        return True

    def insert_column_at(self, col: int, name: str | None = None) -> str | None:
        """지정 위치에 빈 변수(열)를 삽입한다 (SPSS '변수 삽입').

        기존 변수들은 오른쪽으로 밀린다. 삽입된 변수명을 반환.
        """
        ncols = len(self._dataframe.columns)
        col = max(0, min(col, ncols))

        if name is None:
            name = generate_var_name(self._var_counter)
            while name in self._dataframe.columns:
                self._var_counter += 1
                name = generate_var_name(self._var_counter)
        self._var_counter += 1
        if name in self._dataframe.columns:
            return None

        self._push_undo()
        self.beginResetModel()
        if ncols == 0:
            self._dataframe = pd.DataFrame({name: [pd.NA]})
        else:
            self._dataframe.insert(col, name, pd.NA)
        self._variables[name] = VariableMeta(
            name=name,
            label=name,
            storage_type=StorageType.STRING,
            measure=MeasureType.NOMINAL,
        )
        self._measure_initialized.discard(name)
        self._invalidate_col_cache()
        self.endResetModel()
        self.variable_added.emit(name)
        self.data_changed.emit()
        return name
