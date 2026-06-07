"""Data View — SPSS Data View 스타일 데이터 편집 화면.

SPSS와 동일한 구성:
- 최상단: 이름 상자(Name Box) + 값 입력 바(Formula Bar)
- 그리드: 버튼/검색/도움말 없이 깔끔한 테이블만
- 키보드: F2 편집, Delete 삭제, Ctrl+D 아래 복사, Ctrl+C/V 다중 셀 복사/붙여넣기
- 컨텍스트 메뉴: 행/열 조작, 정렬
"""


from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.delegates.cell_delegate import CellDelegate
from nuristat.ui.models.spss_grid_model import SPSSGridModel, generate_var_name


class DataView(QWidget):
    """SPSS Data View 스타일 데이터 편집 화면."""

    dataset_changed = Signal()
    selection_info_changed = Signal(str)  # "N행 × M열 선택" 등 상태바용 문자열

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._model: SPSSGridModel | None = None
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

        # 이름 상자 (Name Box): "행번호:변수명" — 행번호 입력 후 Enter로 케이스 이동
        self.name_box = QLineEdit()
        self.name_box.setPlaceholderText("케이스 번호로 이동")
        self.name_box.setToolTip("행 번호를 입력하고 Enter를 누르면 해당 케이스로 이동합니다")
        self.name_box.setFixedWidth(130)
        self.name_box.returnPressed.connect(self._name_box_goto)
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

        # 편집 트리거: 더블클릭·F2·임의 키 입력. AnyKeyPressed가 키 입력 시 편집기를
        # 열고 포커스를 주며 첫 글자를 전달 — 숫자·텍스트 모두 즉시 입력되고 후속 글자가
        # 올바르게 누적된다(커스텀 처리는 포커스 미전달로 두 번째 글자가 값을 대체했음).
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
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

            # Ctrl+X: 잘라내기 (복사 후 지우기)
            if key == Qt.Key.Key_X and modifiers & Qt.KeyboardModifier.ControlModifier:
                self.cut_selection()
                return True

            # Ctrl+D: 위 셀 값 복사 (Fill Down)
            if key == Qt.Key.Key_D and modifiers & Qt.KeyboardModifier.ControlModifier:
                self._fill_down()
                return True

            # Ctrl+Shift+Z / Ctrl+Y: 다시 실행
            if ((key == Qt.Key.Key_Z
                 and modifiers & Qt.KeyboardModifier.ControlModifier
                 and modifiers & Qt.KeyboardModifier.ShiftModifier)
                    or (key == Qt.Key.Key_Y
                        and modifiers & Qt.KeyboardModifier.ControlModifier)):
                self.redo()
                return True

            # Ctrl+Z: 실행 취소
            if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
                self.undo()
                return True

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

            # 출력 가능한 문자: AnyKeyPressed 편집 트리거가 편집 시작·포커스·첫 글자
            # 전달을 처리하도록 가로채지 않고 통과시킨다 (숫자·텍스트 모두 누적 입력).

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

    def _on_selection_changed(self, selected, deselected) -> None:
        """선택 영역 변경 시 상태바용 정보 방출 (N행 × M열 선택)."""
        indexes = self.table.selectedIndexes()
        if not indexes:
            self.selection_info_changed.emit("")
            return
        n_rows = len(set(idx.row() for idx in indexes))
        n_cols = len(set(idx.column() for idx in indexes))
        if n_rows == 1 and n_cols == 1:
            self.selection_info_changed.emit("")
        else:
            self.selection_info_changed.emit(f"{n_rows}행 × {n_cols}열 선택")

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

    def _name_box_goto(self) -> None:
        """이름 상자에 입력한 케이스(행) 번호로 이동 (SPSS Go to Case).

        '12' 또는 '12:변수명' 형식을 허용 — 앞부분 행 번호만 사용한다.
        """
        if self._model is None:
            return
        text = self.name_box.text().strip()
        row_part = text.split(":")[0].strip()
        try:
            row = int(row_part) - 1
        except ValueError:
            self.table.setFocus()
            return
        row = max(0, min(row, self._model.rowCount() - 1))
        cur = self.table.currentIndex()
        col = cur.column() if cur.isValid() else 0
        idx = self._model.index(row, col)
        if idx.isValid():
            self.table.setCurrentIndex(idx)
            self.table.scrollTo(idx)
        self.table.setFocus()

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
        """선택된 셀 값을 지운다 (1회 실행 취소 단위)."""
        if self._model is None:
            return
        indexes = self.table.selectedIndexes()
        if not indexes:
            return
        with self._model.batch_update():
            for idx in indexes:
                self._model.setData(idx, "", Qt.ItemDataRole.EditRole)

    # ── 실행 취소 / 잘라내기 (SPSS 데이터 편집기) ───────────────────────────

    def undo(self) -> None:
        """마지막 데이터 편집을 취소한다."""
        if self._model is not None and self._model.undo():
            self._restore_focus()

    def redo(self) -> None:
        """취소한 편집을 다시 실행한다."""
        if self._model is not None and self._model.redo():
            self._restore_focus()

    def cut_selection(self) -> None:
        """선택 영역을 복사한 뒤 지운다 (Ctrl+X)."""
        if self._model is None:
            return
        self._copy_selection()
        self._clear_selection()

    def _restore_focus(self) -> None:
        """undo/redo(모델 리셋) 후 유효한 현재 셀과 포커스를 복원한다."""
        if self._model is None:
            return
        current = self.table.currentIndex()
        row = current.row() if current.isValid() else 0
        col = current.column() if current.isValid() else 0
        row = max(0, min(row, self._model.rowCount() - 1))
        col = max(0, min(col, self._model.columnCount() - 1))
        idx = self._model.index(row, col)
        if idx.isValid():
            self.table.setCurrentIndex(idx)
        self.table.setFocus()

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
        # 배치 모드: 셀별 전체 재구축 신호를 억제하고 종료 시 1회만 방출 (대량 붙여넣기 최적화)
        with self._model.batch_update():
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

        # 배치 모드: 다중 셀 채우기를 1회 신호로 합침 (SPSS Fill Down 최적화)
        with self._model.batch_update():
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

        # 실행 취소 / 다시 실행
        undo_action = menu.addAction("실행 취소 (Ctrl+Z)")
        undo_action.triggered.connect(self.undo)
        undo_action.setEnabled(self._model.can_undo())

        redo_action = menu.addAction("다시 실행 (Ctrl+Y)")
        redo_action.triggered.connect(self.redo)
        redo_action.setEnabled(self._model.can_redo())

        menu.addSeparator()

        # 복사/잘라내기/붙여넣기
        cut_action = menu.addAction("잘라내기 (Ctrl+X)")
        cut_action.triggered.connect(self.cut_selection)

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

        # 삽입 (커서 위치) — SPSS '케이스 삽입' / '변수 삽입'
        if index.isValid():
            insert_row_action = menu.addAction("위에 행 삽입")
            insert_row_action.triggered.connect(self._insert_row)

            insert_col_action = menu.addAction("앞에 변수 삽입")
            insert_col_action.triggered.connect(self._insert_column)

        add_row_action = menu.addAction("행 추가 (맨 끝)")
        add_row_action.triggered.connect(self._add_row)

        add_col_action = menu.addAction("변수 추가 (맨 끝)")
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

    def _insert_row(self) -> None:
        """현재 행 위에 빈 행 삽입 (SPSS 케이스 삽입)."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        row = index.row() if index.isValid() else 0
        self._model.insert_row_at(row)

    def _insert_column(self) -> None:
        """현재 변수 앞에 빈 변수 삽입 (SPSS 변수 삽입)."""
        if self._model is None:
            return
        index = self.table.currentIndex()
        col = index.column() if index.isValid() else 0
        self._model.insert_column_at(col)

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
            self._model.sort_by_column(index.column(), ascending=True)

    def _sort_descending(self) -> None:
        index = self.table.currentIndex()
        if index.isValid() and self._model:
            self._model.sort_by_column(index.column(), ascending=False)

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
            self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
            self._cell_change_connected = True

        # SPSS column_width 속성 → 실제 열 너비 적용 (문자 단위 × 9px)
        if dataset.variables:
            for i, col_name in enumerate(dataset.data.columns):
                var = dataset.variables.get(col_name)
                if var is not None and hasattr(var, "column_width") and var.column_width:
                    self.table.setColumnWidth(i, max(50, var.column_width * 9))

        # 첫 셀 선택
        first = self._model.index(0, 0)
        if first.isValid():
            self.table.setCurrentIndex(first)

    def refresh(self) -> None:
        """화면을 새로고침합니다. beginResetModel 없이 갱신해 포커스를 보존합니다."""
        if self._model is not None:
            top_left = self._model.index(0, 0)
            bottom_right = self._model.index(
                self._model.rowCount() - 1,
                self._model.columnCount() - 1,
            )
            self._model.dataChanged.emit(
                top_left, bottom_right,
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.BackgroundRole],
            )
            self._model.headerDataChanged.emit(
                Qt.Orientation.Horizontal, 0, self._model.columnCount() - 1
            )

    def get_dataset(self) -> Dataset | None:
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
