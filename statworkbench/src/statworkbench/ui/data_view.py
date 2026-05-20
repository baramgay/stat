"""Data View — SPSS Data View 스타일 데이터 편집 화면.

SPSS와 동일한 구성:
- 최상단: 이름 상자(Name Box) + 값 입력 바(Formula Bar)
- 그리드: 버튼/검색/도움말 없이 깔끔한 테이블만
- 키보드: F2 편집, Delete 삭제, Ctrl+D 아래 복사, Ctrl+C/V 다중 셀 복사/붙여넣기
- 컨텍스트 메뉴: 행/열 조작, 정렬
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableView,
    QLabel,
    QHeaderView,
    QAbstractItemView,
    QMenu,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QApplication,
    QSizePolicy,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QKeyEvent, QFont, QColor
from typing import Optional

from statworkbench.core.dataset import Dataset
from statworkbench.ui.models.spss_grid_model import SPSSGridModel, generate_var_name
from statworkbench.ui.delegates.cell_delegate import CellDelegate


class DataView(QWidget):
    """SPSS Data View 스타일 데이터 편집 화면."""

    dataset_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._model: Optional[SPSSGridModel] = None
        self._cell_change_connected = False
        self._setup_ui()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 1. 이름 상자 + 값 입력 바 (SPSS Formula Bar) ──────────────────
        formula_frame = QFrame()
        formula_frame.setFrameShape(QFrame.Shape.NoFrame)
        formula_frame.setStyleSheet(
            "QFrame { background-color: #F0F2F5; border-bottom: 1px solid #C0C4CC; }"
        )
        formula_layout = QHBoxLayout(formula_frame)
        formula_layout.setContentsMargins(4, 3, 4, 3)
        formula_layout.setSpacing(4)

        # 이름 상자 (Name Box): "행번호:변수명"
        self.name_box = QLineEdit()
        self.name_box.setReadOnly(True)
        self.name_box.setPlaceholderText("셀 주소")
        self.name_box.setFixedWidth(130)
        self.name_box.setStyleSheet(
            "QLineEdit {"
            "  font-size: 12px; font-weight: bold;"
            "  background: white; border: 1px solid #A0A4AA;"
            "  border-radius: 2px; padding: 2px 6px;"
            "  color: #1A3050;"
            "}"
        )
        formula_layout.addWidget(self.name_box)

        # 구분선
        sep = QLabel("|")
        sep.setStyleSheet("color: #A0A4AA; font-size: 14px;")
        formula_layout.addWidget(sep)

        # 값 입력 바 (Formula Bar): 현재 셀 값 표시/편집
        self.formula_bar = QLineEdit()
        self.formula_bar.setPlaceholderText("셀 값을 입력하고 Enter를 누르세요")
        self.formula_bar.setStyleSheet(
            "QLineEdit {"
            "  font-size: 13px;"
            "  background: white; border: 1px solid #A0A4AA;"
            "  border-radius: 2px; padding: 2px 8px;"
            "}"
            "QLineEdit:focus {"
            "  border: 1px solid #2E86DE;"
            "}"
        )
        self.formula_bar.returnPressed.connect(self._formula_bar_commit)
        formula_layout.addWidget(self.formula_bar, 1)

        layout.addWidget(formula_frame)

        # ── 2. 데이터 테이블 (SPSS Data Grid) ────────────────────────────
        self.table = QTableView()
        self.table.setAlternatingRowColors(False)  # 모델에서 직접 배경색 관리
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.verticalHeader().setMinimumWidth(50)
        self.table.verticalHeader().setMaximumWidth(50)

        # SPSS 스타일 delegate
        self.cell_delegate = CellDelegate(self.table)
        self.table.setItemDelegate(self.cell_delegate)
        self.cell_delegate.closeEditor.connect(self._on_editor_closed)

        # 편집 트리거: 더블클릭, F2만 허용 — 직접타이핑은 eventFilter에서 처리
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )

        # 탭 네비게이션 활성화
        self.table.setTabKeyNavigation(True)

        # 컨텍스트 메뉴
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        # 헤더 더블클릭: 변수명 편집
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._edit_header)

        # SPSS 스타일 테이블 스타일시트
        self.table.setStyleSheet("""
            QTableView {
                border: none;
                gridline-color: #D0D4DA;
                font-size: 13px;
                font-family: "Malgun Gothic", "Arial", sans-serif;
                selection-background-color: #1565C0;
                selection-color: white;
                background-color: white;
            }
            QTableView::item {
                padding: 2px 6px;
            }
            QTableView::item:selected {
                background-color: #1565C0;
                color: white;
            }
            QTableView::item:focus {
                border: 2px solid #1565C0;
                background-color: #E3F2FD;
                color: #0D47A1;
            }
            QHeaderView::section {
                background-color: #D4E6F1;
                color: #1A3050;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 6px;
                border: none;
                border-right: 1px solid #B0BEC5;
                border-bottom: 1px solid #B0BEC5;
            }
            QHeaderView::section:vertical {
                background-color: #E8EAF6;
                color: #3949AB;
                font-size: 11px;
                padding: 2px 4px;
                border: none;
                border-right: 1px solid #B0BEC5;
                border-bottom: 1px solid #D0D4DA;
                text-align: right;
            }
            QHeaderView::section:horizontal:hover {
                background-color: #AED6F1;
            }
            QHeaderView::section:vertical:hover {
                background-color: #C5CAE9;
            }
        """)

        layout.addWidget(self.table)

        # 이벤트 필터: 테이블과 formula bar (setup 완료 후 등록)
        self.table.installEventFilter(self)
        self.formula_bar.installEventFilter(self)

    # ── 이벤트 필터 ─────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """키보드 네비게이션 및 Formula Bar Escape 처리."""
        # setup 완료 전 호출 방어
        if not hasattr(self, 'table') or not hasattr(self, 'formula_bar'):
            return super().eventFilter(obj, event)

        # Formula Bar: Escape → 포커스 반환, 값 원복 / Tab → 커밋 후 오른쪽 이동
        if obj is self.formula_bar and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._formula_bar_cancel()
                return True
            if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                self._formula_bar_commit()
                if event.key() == Qt.Key.Key_Tab:
                    self._navigate(1, 0)
                else:
                    self._navigate(-1, 0)
                return True

        if obj is not self.table or event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(obj, event)

        key_event: QKeyEvent = event
        key = key_event.key()
        modifiers = key_event.modifiers()
        editing = self.table.state() == QAbstractItemView.State.EditingState

        if not editing:
            # F2: 편집 모드 진입
            if key == Qt.Key.Key_F2:
                current = self.table.currentIndex()
                if current.isValid():
                    self.table.edit(current)
                return True

            # Delete / Backspace: 셀 값 지우기
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._clear_selection()
                return True

            # Ctrl+C: 다중 셀 복사
            if key == Qt.Key.Key_C and modifiers & Qt.KeyboardModifier.ControlModifier:
                self._copy_selection()
                return True

            # Ctrl+V: 다중 셀 붙여넣기
            if key == Qt.Key.Key_V and modifiers & Qt.KeyboardModifier.ControlModifier:
                self._paste_selection()
                return True

            # Ctrl+D: 위 셀 값 복사 (Fill Down)
            if key == Qt.Key.Key_D and modifiers & Qt.KeyboardModifier.ControlModifier:
                self._fill_down()
                return True

            # Ctrl+Z: 실행 취소 (향후 Undo 스택 연동 가능)
            if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
                return True  # 현재는 pass-through

            # Enter: 아래로 이동 (auto-repeat 무시 — Windows에서 키 유지 시 다중 이동 방지)
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if key_event.isAutoRepeat():
                    return True
                return self._navigate(0, 1)

            # Tab: 오른쪽으로 이동
            if key == Qt.Key.Key_Tab:
                if key_event.isAutoRepeat():
                    return True
                return self._navigate(1, 0)

            # Shift+Tab: 왼쪽으로 이동
            if key == Qt.Key.Key_Backtab:
                return self._navigate(-1, 0)

            # 화살표 키: 이동만 (편집 진입 없음)
            if key == Qt.Key.Key_Up:
                return self._navigate(0, -1)
            if key == Qt.Key.Key_Down:
                return self._navigate(0, 1)
            if key == Qt.Key.Key_Left:
                return self._navigate(-1, 0)
            if key == Qt.Key.Key_Right:
                return self._navigate(1, 0)

            # 출력 가능한 문자: 즉시 편집 시작 (기존 값 지우고 입력)
            if (key_event.text()
                    and key_event.text().isprintable()
                    and not (modifiers & Qt.KeyboardModifier.ControlModifier)
                    and not (modifiers & Qt.KeyboardModifier.AltModifier)):
                current = self.table.currentIndex()
                if current.isValid():
                    self.cell_delegate.set_initial_text(key_event.text())
                    self.table.edit(current)
                return True

        return super().eventFilter(obj, event)

    def _on_editor_closed(self, editor, hint) -> None:
        """편집기 닫힌 후 delegate의 _pending_navigate로 셀 이동.

        저장된 원래 위치로 무조건 복원 후 이동 — Qt가 closeEditor 처리 중
        currentIndex를 다른 위치로 옮길 수 있어, 조건부 복원만으로는 오이동 발생.
        """
        nav = self.cell_delegate._pending_navigate
        if nav:
            dc, dr, src_row, src_col = nav
            self.cell_delegate._pending_navigate = None
            if src_row >= 0:
                self.table.setCurrentIndex(self._model.index(src_row, src_col))
            self._navigate(dc, dr)

    # ── 셀 선택 변경 → Formula Bar 업데이트 ────────────────────────────────

    def _on_cell_changed(self, current, previous) -> None:
        """셀 선택 변경 시 이름 상자와 값 입력 바 업데이트."""
        if not current.isValid() or self._model is None:
            self.name_box.setText("")
            self.formula_bar.setText("")
            return

        row = current.row()
        col = current.column()

        # 이름 상자: "행번호:변수명"
        var_name = self._model.headerData(
            col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        )
        # 아이콘 제거 (헤더에 아이콘 포함된 경우 "▪ VAR00001" → "VAR00001")
        if var_name and len(var_name) > 2 and var_name[1] == " ":
            var_name_clean = var_name[2:]
        else:
            var_name_clean = var_name or ""

        self.name_box.setText(f"{row + 1}:{var_name_clean}")

        # 값 입력 바: EditRole 기준 실제 값 (포맷 없는 원본)
        value = self._model.data(current, Qt.ItemDataRole.EditRole)
        self.formula_bar.setText(str(value) if value is not None and value != "" else "")

    def _formula_bar_commit(self) -> None:
        """Formula Bar에서 Enter → 현재 셀에 값 반영 후 아래로 이동."""
        if self._model is None:
            return
        current = self.table.currentIndex()
        if not current.isValid():
            return

        text = self.formula_bar.text()
        self._model.setData(current, text, Qt.ItemDataRole.EditRole)
        self.table.setFocus()
        self._navigate(0, 1)

    def _formula_bar_cancel(self) -> None:
        """Formula Bar Escape → 원래 값으로 복원, 포커스 반환."""
        current = self.table.currentIndex()
        if current.isValid() and self._model:
            value = self._model.data(current, Qt.ItemDataRole.EditRole)
            self.formula_bar.setText(str(value) if value else "")
        self.table.setFocus()

    # ── 네비게이션 ───────────────────────────────────────────────────────────

    def _navigate(self, dc: int, dr: int) -> bool:
        """현재 셀에서 (dc열, dr행) 방향으로 이동. 편집 없이 이동만."""
        if self._model is None:
            return False
        current = self.table.currentIndex()
        if not current.isValid():
            return False

        new_row = current.row() + dr
        new_col = current.column() + dc

        # 경계 처리
        if new_row < 0 or new_col < 0:
            return False

        # SPSS 호환: 네비게이션만으로는 변수를 생성하지 않는다.
        # 변수는 실제 데이터 입력(setData) 시에만 생성된다.
        # 가상 그리드(DEFAULT_COLS=100)가 빈 열을 이미 제공하므로
        # _create_variable_at_col 호출 불필요.

        next_index = self._model.index(new_row, new_col)
        if next_index.isValid():
            self.table.setCurrentIndex(next_index)
            self.table.scrollTo(next_index)
        return True

    # ── 편집 기능 ────────────────────────────────────────────────────────────

    def _clear_selection(self) -> None:
        """선택된 셀 값을 지운다."""
        if self._model is None:
            return
        indexes = self.table.selectedIndexes()
        for idx in indexes:
            self._model.setData(idx, "", Qt.ItemDataRole.EditRole)

    def _copy_selection(self) -> None:
        """선택된 영역을 탭/줄바꿈 구분 텍스트로 클립보드에 복사 (Excel 호환)."""
        if self._model is None:
            return
        indexes = self.table.selectedIndexes()
        if not indexes:
            return

        rows = sorted(set(idx.row() for idx in indexes))
        cols = sorted(set(idx.column() for idx in indexes))

        text_rows = []
        for r in rows:
            row_data = []
            for c in cols:
                idx = self._model.index(r, c)
                val = self._model.data(idx, Qt.ItemDataRole.DisplayRole)
                row_data.append(str(val) if val is not None else "")
            text_rows.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(text_rows))

    def _paste_selection(self) -> None:
        """클립보드의 탭/줄바꿈 텍스트를 현재 셀 기준으로 붙여넣기."""
        if self._model is None:
            return
        current = self.table.currentIndex()
        if not current.isValid():
            return

        text = QApplication.clipboard().text()
        if not text:
            return

        rows = text.split("\n")
        for r, row_text in enumerate(rows):
            if not row_text:
                continue
            cells = row_text.split("\t")
            for c, val in enumerate(cells):
                idx = self._model.index(current.row() + r, current.column() + c)
                if idx.isValid():
                    self._model.setData(idx, val.strip(), Qt.ItemDataRole.EditRole)

    def _fill_down(self) -> None:
        """Ctrl+D: 현재 셀 위쪽 값을 선택 영역에 복사 (SPSS Fill Down)."""
        if self._model is None:
            return
        current = self.table.currentIndex()
        if not current.isValid() or current.row() == 0:
            return

        indexes = self.table.selectedIndexes()
        cols = sorted(set(idx.column() for idx in indexes))
        rows = sorted(set(idx.row() for idx in indexes))

        for c in cols:
            source_idx = self._model.index(rows[0] - 1, c)
            source_val = self._model.data(source_idx, Qt.ItemDataRole.EditRole)
            for r in rows:
                target_idx = self._model.index(r, c)
                self._model.setData(target_idx, source_val, Qt.ItemDataRole.EditRole)

    # ── 헤더 편집 ────────────────────────────────────────────────────────────

    def _edit_header(self, section: int) -> None:
        """헤더 더블클릭: 변수명 편집 다이얼로그."""
        if self._model is None:
            return

        raw = self._model.headerData(
            section, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        )
        # 아이콘 제거
        if raw and len(raw) > 2 and raw[1] == " ":
            current_name = raw[2:]
        else:
            current_name = str(raw) if raw else ""

        new_name, ok = QInputDialog.getText(
            self, "변수명 변경", "새 변수명:", text=current_name
        )
        if ok and new_name.strip():
            self._model.setHeaderData(
                section, Qt.Orientation.Horizontal, new_name.strip()
            )

    # ── 컨텍스트 메뉴 ────────────────────────────────────────────────────────

    def _show_context_menu(self, position) -> None:
        if self._model is None:
            return

        index = self.table.indexAt(position)
        menu = QMenu(self)

        # 복사/붙여넣기
        copy_action = menu.addAction("복사 (Ctrl+C)")
        copy_action.triggered.connect(self._copy_selection)

        paste_action = menu.addAction("붙여넣기 (Ctrl+V)")
        paste_action.triggered.connect(self._paste_selection)

        clear_action = menu.addAction("셀 값 지우기 (Delete)")
        clear_action.triggered.connect(self._clear_selection)

        menu.addSeparator()

        fill_down_action = menu.addAction("아래 복사 (Ctrl+D)")
        fill_down_action.triggered.connect(self._fill_down)

        menu.addSeparator()

        add_row_action = menu.addAction("행 추가")
        add_row_action.triggered.connect(self._add_row)

        add_col_action = menu.addAction("변수 추가")
        add_col_action.triggered.connect(self._add_column)

        if index.isValid():
            del_row_action = menu.addAction("행 삭제")
            del_row_action.triggered.connect(self._delete_row)

            del_col_action = menu.addAction("변수 삭제")
            del_col_action.triggered.connect(self._delete_column)

        menu.addSeparator()

        sort_asc_action = menu.addAction("오름차순 정렬")
        sort_asc_action.triggered.connect(self._sort_ascending)

        sort_desc_action = menu.addAction("내림차순 정렬")
        sort_desc_action.triggered.connect(self._sort_descending)

        menu.exec(self.table.viewport().mapToGlobal(position))

    # ── 행/열 조작 ───────────────────────────────────────────────────────────

    def _add_row(self) -> None:
        if self._model is not None:
            self._model.add_row()

    def _add_column(self) -> None:
        if self._model is not None:
            var_name = generate_var_name(len(self._model.get_full_dataframe().columns) + 1)
            self._model.add_column(var_name)

    def _delete_row(self) -> None:
        if self._model is None:
            return
        index = self.table.currentIndex()
        if index.isValid():
            row = index.row()
            reply = QMessageBox.question(
                self, "행 삭제", f"{row + 1}번 행을 삭제하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._model.remove_row(row)

    def _delete_column(self) -> None:
        if self._model is None:
            return
        index = self.table.currentIndex()
        if index.isValid():
            col = index.column()
            df = self._model.get_full_dataframe()
            if col < len(df.columns):
                col_name = df.columns[col]
                reply = QMessageBox.question(
                    self, "변수 삭제", f"변수 '{col_name}'을(를) 삭제하시겠습니까?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._model.remove_column(col)

    # ── 정렬 ────────────────────────────────────────────────────────────────

    def _sort_ascending(self) -> None:
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            col = index.column()
            df = self._model.get_dataframe()
            if col < len(df.columns):
                col_name = df.columns[col]
                df.sort_values(by=col_name, ascending=True, inplace=True)
                df.reset_index(drop=True, inplace=True)
                self._model.beginResetModel()
                self._model._dataframe = df
                self._model.endResetModel()

    def _sort_descending(self) -> None:
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            col = index.column()
            df = self._model.get_dataframe()
            if col < len(df.columns):
                col_name = df.columns[col]
                df.sort_values(by=col_name, ascending=False, inplace=True)
                df.reset_index(drop=True, inplace=True)
                self._model.beginResetModel()
                self._model._dataframe = df
                self._model.endResetModel()

    # ── 공개 인터페이스 (기존 기능 유지) ────────────────────────────────────

    def set_dataset(self, dataset: Dataset) -> None:
        """데이터셋을 설정합니다."""
        self._dataset = dataset
        self._model = SPSSGridModel(dataset.data, dataset.variables)
        self._model.data_changed.connect(self._on_data_changed)
        self._model.variable_added.connect(self._on_variable_added)
        self._model.variable_renamed.connect(self._on_variable_renamed)
        self.table.setModel(self._model)

        # 셀 선택 변경 시 Formula Bar 업데이트
        self._cell_change_connected = False
        if self.table.selectionModel():
            self.table.selectionModel().currentChanged.connect(self._on_cell_changed)
            self._cell_change_connected = True

        # 첫 셀 선택
        first = self._model.index(0, 0)
        if first.isValid():
            self.table.setCurrentIndex(first)

    def refresh(self) -> None:
        """화면을 새로고침합니다."""
        if self._model is not None:
            self._model.beginResetModel()
            self._model.endResetModel()

    def get_dataset(self) -> Optional[Dataset]:
        """현재 데이터셋을 반환합니다."""
        return self._dataset

    def toggle_value_labels(self) -> bool:
        """값 라벨 표시 모드를 토글합니다. 현재 상태를 반환합니다."""
        if self._model is not None:
            return self._model.toggle_value_labels()
        return False

    # ── 내부 콜백 ────────────────────────────────────────────────────────────

    def _on_data_changed(self) -> None:
        if self._dataset is not None and self._model is not None:
            self._dataset.data = self._model.get_dataframe()
            for var_name, var_meta in self._model.get_variables().items():
                self._dataset.variables[var_name] = var_meta
            current = self.table.currentIndex()
            if current.isValid():
                self._on_cell_changed(current, current)
            self.dataset_changed.emit()

    def _on_variable_added(self, var_name: str) -> None:
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
        if self._dataset is not None and old_name in self._dataset.variables:
            var = self._dataset.variables.pop(old_name)
            var.name = new_name
            var.label = new_name
            self._dataset.variables[new_name] = var
