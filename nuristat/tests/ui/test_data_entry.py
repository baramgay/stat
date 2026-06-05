"""Data entry integration tests — keyboard interaction with DataView."""
from __future__ import annotations

import sys
import pytest

from PySide6.QtWidgets import QApplication, QLineEdit, QAbstractItemView
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from nuristat.core.dataset import Dataset
from nuristat.ui.data_view import DataView
import pandas as pd


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing:
        yield existing
    else:
        a = QApplication(sys.argv)
        yield a


@pytest.fixture
def view(app):
    """Fresh DataView with empty dataset."""
    ds = Dataset(pd.DataFrame(), name="TestData")
    v = DataView()
    v.set_dataset(ds)
    v.resize(800, 400)
    v.show()
    QApplication.processEvents()
    yield v
    v.hide()
    v.close()


def _current_editor(view: DataView) -> QLineEdit | None:
    """Return the currently open QLineEdit editor, or None."""
    for child in view.table.viewport().findChildren(QLineEdit):
        if child.isVisible():
            return child
    return None


def _select_cell(view: DataView, row: int, col: int) -> None:
    idx = view._model.index(row, col)
    view.table.setCurrentIndex(idx)
    view.table.setFocus()
    QApplication.processEvents()


def _type_char(view: DataView, char: str) -> None:
    """Send a printable character to the table to trigger immediate editing."""
    QTest.keyClicks(view.table, char)
    QApplication.processEvents()


def _wait(ms: int = 50) -> None:
    """Process events for at least ms milliseconds (for QTimer.singleShot to fire)."""
    QTest.qWait(ms)


def _cell_value(view: DataView, row: int, col: int) -> str:
    val = view._model.data(view._model.index(row, col), Qt.ItemDataRole.DisplayRole)
    return str(val) if val is not None else ""


# ─────────────────────────────────────────────────────────
# 1. 즉시 입력 (immediate entry)
# ─────────────────────────────────────────────────────────

class TestImmediateEntry:
    """Typing a character on a selected cell should open editor immediately."""

    def test_printable_key_opens_editor(self, view):
        """Pressing a printable key should open editor with that character."""
        _select_cell(view, 0, 0)
        _type_char(view, '5')

        editor = _current_editor(view)
        assert editor is not None, "Editor should open immediately on keypress"
        assert editor.text() == '5', f"Editor should show '5', got '{editor.text()}'"

        # Escape to cancel
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()

    def test_first_char_not_doubled(self, view):
        """No double-keypress bug: first char appears once, not twice."""
        _select_cell(view, 0, 0)
        _type_char(view, '9')

        editor = _current_editor(view)
        assert editor is not None
        assert editor.text() == '9', f"Got '{editor.text()}' — first char doubled?"
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()

    def test_digit_then_more_typing(self, view):
        """After immediate entry, subsequent keystrokes append to editor."""
        _select_cell(view, 0, 0)
        _type_char(view, '1')

        editor = _current_editor(view)
        assert editor is not None
        assert editor.text() == '1', f"First char should be '1', got '{editor.text()}'"
        editor.deselect()  # 안전 deselect (선택 있으면 해제)
        QTest.keyClicks(editor, '23')
        QApplication.processEvents()
        assert editor.text() == '123', f"Got '{editor.text()}'"
        QTest.keyClick(view.table, Qt.Key.Key_Escape)
        _wait(30)

    def test_editor_has_focus_after_keystroke(self, view):
        """첫 키 입력 후 편집기가 포커스를 가져야 함 — 안 그러면 다음 키가 값을 대체(보고된 버그)."""
        _select_cell(view, 0, 0)
        _type_char(view, '1')

        editor = _current_editor(view)
        assert editor is not None
        assert QApplication.focusWidget() is editor, (
            f"편집기가 포커스를 가져야 함, got {type(QApplication.focusWidget()).__name__}"
        )
        # 포커스된 편집기에 후속 입력 → 누적(대체 아님)
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_2)
        QApplication.processEvents()
        assert editor.text() == '12', f"누적되어 '12'여야 함(기존 숫자 대체 버그), got '{editor.text()}'"
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()

    def test_text_key_opens_editor(self, view):
        """텍스트(문자)도 숫자처럼 키 입력 시 즉시 편집 시작."""
        _select_cell(view, 0, 0)
        _type_char(view, 'a')

        editor = _current_editor(view)
        assert editor is not None, "문자 입력 시 편집기가 즉시 열려야 함"
        assert editor.text() == 'a', f"편집기에 'a' 표시되어야 함, got '{editor.text()}'"
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()


# ─────────────────────────────────────────────────────────
# 2. 커밋 + 네비게이션
# ─────────────────────────────────────────────────────────

class TestCommitAndNavigation:
    """Enter/Tab should commit value and move cursor."""

    def test_enter_commits_and_moves_down(self, view):
        """Enter commits value and moves cursor one row down."""
        _select_cell(view, 0, 0)
        _type_char(view, '7')

        editor = _current_editor(view)
        assert editor is not None
        # SubmitModelCache 방식 → editor에 직접 보내도 안전
        QTest.keyClick(editor, Qt.Key.Key_Return)
        _wait(80)

        # Value committed
        val = _cell_value(view, 0, 0)
        assert val not in ('', '.'), f"Value should be committed, got '{val}'"

        # Cursor moved to row 1
        cur = view.table.currentIndex()
        assert cur.row() == 1, f"Should be on row 1 after Enter, got row {cur.row()}"
        assert cur.column() == 0

    def test_tab_commits_and_moves_right(self, view):
        """Tab commits value and moves cursor one column right."""
        _select_cell(view, 0, 0)
        _type_char(view, '3')

        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Tab)
        _wait(80)

        # Value committed
        val = _cell_value(view, 0, 0)
        assert val not in ('', '.'), f"Value should be committed, got '{val}'"

        # Cursor moved to col 1
        cur = view.table.currentIndex()
        assert cur.column() == 1, f"Should be on col 1 after Tab, got col {cur.column()}"
        assert cur.row() == 0

    def test_shift_tab_moves_left(self, view):
        """Shift+Tab commits and moves cursor one column left."""
        _select_cell(view, 0, 1)
        _type_char(view, '4')

        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        _wait(80)

        cur = view.table.currentIndex()
        assert cur.column() == 0, f"Shift+Tab should move left, got col {cur.column()}"

    def test_escape_reverts_value(self, view):
        """Escape cancels editing and reverts to original value."""
        _select_cell(view, 0, 0)
        # Set an initial value first
        _type_char(view, '5')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

        original = _cell_value(view, 0, 0)

        # Now edit and escape
        _select_cell(view, 0, 0)
        _type_char(view, '9')
        editor2 = _current_editor(view)
        assert editor2 is not None
        assert editor2.text() == '9'
        QTest.keyClick(editor2, Qt.Key.Key_Escape)
        QApplication.processEvents()

        after = _cell_value(view, 0, 0)
        assert after == original, f"Escape should revert to '{original}', got '{after}'"


# ─────────────────────────────────────────────────────────
# 3. 화살표 키 네비게이션 (편집 없이)
# ─────────────────────────────────────────────────────────

class TestArrowNavigation:
    """Arrow keys on a non-editing cell should navigate without opening editor."""

    def test_down_arrow_moves_down(self, view):
        _select_cell(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Down)
        QApplication.processEvents()
        assert _current_editor(view) is None, "Arrow key should not open editor"
        assert view.table.currentIndex().row() == 1

    def test_right_arrow_moves_right(self, view):
        _select_cell(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Right)
        QApplication.processEvents()
        assert _current_editor(view) is None
        assert view.table.currentIndex().column() == 1

    def test_up_arrow_no_negative_row(self, view):
        _select_cell(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Up)
        QApplication.processEvents()
        assert view.table.currentIndex().row() == 0  # stays at top

    def test_left_arrow_no_negative_col(self, view):
        _select_cell(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Left)
        QApplication.processEvents()
        assert view.table.currentIndex().column() == 0  # stays at col 0


# ─────────────────────────────────────────────────────────
# 4. Delete / F2
# ─────────────────────────────────────────────────────────

class TestDeleteAndF2:
    """Delete clears cell; F2 opens editor with existing value."""

    def test_delete_clears_cell(self, view):
        """Delete key should clear the current cell's value."""
        _select_cell(view, 0, 0)
        _type_char(view, '8')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

        _select_cell(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Delete)
        QApplication.processEvents()

        val = _cell_value(view, 0, 0)
        assert val in ('', '.'), f"Delete should clear cell, got '{val}'"

    def test_f2_opens_editor_with_existing_value(self, view):
        """F2 should open editor showing the current cell value."""
        _select_cell(view, 0, 0)
        _type_char(view, '4')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

        stored_val = _cell_value(view, 0, 0)
        _select_cell(view, 0, 0)

        QTest.keyClick(view.table, Qt.Key.Key_F2)
        QApplication.processEvents()

        editor2 = _current_editor(view)
        assert editor2 is not None, "F2 should open editor"
        assert editor2.text() == stored_val, (
            f"F2 editor should show '{stored_val}', got '{editor2.text()}'"
        )
        QTest.keyClick(editor2, Qt.Key.Key_Escape)
        QApplication.processEvents()


# ─────────────────────────────────────────────────────────
# 5. 자동 변수 생성
# ─────────────────────────────────────────────────────────

class TestAutoVariableCreation:
    """SPSS 호환: Tab은 네비게이션만, 데이터 입력 시에만 변수 생성."""

    def test_tab_moves_to_next_virtual_column(self, view):
        """Tab past the last column moves to virtual col (SPSS 호환: 변수 생성 안 함)."""
        _select_cell(view, 0, 0)
        _type_char(view, '1')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Tab)
        _wait(80)

        # 오른쪽 컬럼으로 이동
        cur = view.table.currentIndex()
        assert cur.column() == 1

        # 데이터 입력 없이 Tab만 → VAR00001 1개만 존재 (SPSS 동작)
        df = view._model.get_full_dataframe()
        assert len(df.columns) == 1, (
            f"Tab 네비게이션만으로 새 변수를 생성하면 안 됨. 현재 컬럼: {list(df.columns)}"
        )

    def test_data_entry_creates_variable(self, view):
        """데이터 입력 시 변수 생성 (Tab 후 새 셀에 입력)."""
        _select_cell(view, 0, 0)
        _type_char(view, '1')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Tab)
        _wait(80)

        # col 1으로 이동 후 데이터 입력
        _type_char(view, '2')
        editor2 = _current_editor(view)
        if editor2 is not None:
            QTest.keyClick(editor2, Qt.Key.Key_Return)
            _wait(80)

        df = view._model.get_full_dataframe()
        assert len(df.columns) >= 2, (
            f"데이터 입력 후 변수 생성 필수. 현재: {list(df.columns)}"
        )


# ─────────────────────────────────────────────────────────
# 6. Formula Bar 동기화
# ─────────────────────────────────────────────────────────

class TestFormulaBar:
    """Formula bar should update when cell selection changes."""

    def test_formula_bar_shows_cell_address(self, view):
        _select_cell(view, 2, 0)
        QApplication.processEvents()
        assert '3' in view.name_box.text(), (
            f"Name box should show row 3, got '{view.name_box.text()}'"
        )

    def test_formula_bar_shows_cell_value(self, view):
        _select_cell(view, 0, 0)
        _type_char(view, '6')
        editor = _current_editor(view)
        assert editor is not None
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

        _select_cell(view, 0, 0)
        val = _cell_value(view, 0, 0)
        if val not in ('', '.'):
            assert val in view.formula_bar.text(), (
                f"Formula bar should show '{val}', got '{view.formula_bar.text()}'"
            )
