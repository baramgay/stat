"""변수 보기 편집 UX — 값 라벨 기존값 표시·즉시편집·인접행 이동 검증."""
from __future__ import annotations

import sys

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QLineEdit

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta
from nuristat.ui.dialogs.variable_editor import ValueLabelsDialog
from nuristat.ui.variable_view import VariableView


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def view(app):
    df = pd.DataFrame({"gender": [0, 1, 0, 1, 0], "score": [10, 20, 30, 40, 50]})
    variables = {
        "gender": VariableMeta(name="gender", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL),
        "score": VariableMeta(name="score", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
    }
    ds = Dataset(df, "t", variables)
    v = VariableView()
    v.set_dataset(ds)
    v.resize(900, 300)
    v.show()
    QApplication.processEvents()
    yield v
    v.hide()
    v.close()


def _editor(view):
    for child in view.table.viewport().findChildren(QLineEdit):
        if child.isVisible():
            return child
    return None


# ── 값 라벨: 데이터 기존값 자동 표시 ─────────────────────────────────

class TestValueLabelsExistingValues:

    def test_dialog_shows_existing_data_values(self):
        """라벨 미지정이어도 데이터의 고유값이 행으로 표시됨 (SPSS식)."""
        dlg = ValueLabelsDialog({}, None, existing_values=[0, 1, 2])
        assert dlg.table.rowCount() == 3
        vals = [dlg.table.item(r, 0).text() for r in range(3)]
        assert vals == ["0", "1", "2"]

    def test_existing_labels_prefilled(self):
        """기존 라벨이 있는 값은 라벨이 채워지고, 없는 값은 빈칸."""
        dlg = ValueLabelsDialog({0: "남"}, None, existing_values=[0, 1])
        labels = [dlg.table.item(r, 1).text() for r in range(2)]
        assert labels == ["남", ""]

    def test_float_integer_value_shown_clean(self):
        """1.0 같은 정수형 실수값은 '1'로 표시."""
        dlg = ValueLabelsDialog({}, None, existing_values=[1.0, 2.0])
        vals = [dlg.table.item(r, 0).text() for r in range(2)]
        assert vals == ["1", "2"]

    def test_labels_not_in_data_still_kept(self):
        """데이터에 없지만 기존 라벨이 있는 값도 유지."""
        dlg = ValueLabelsDialog({9: "기타"}, None, existing_values=[0, 1])
        all_vals = [dlg.table.item(r, 0).text() for r in range(dlg.table.rowCount())]
        assert "9" in all_vals

    def test_view_distinct_values_helper(self, view):
        """뷰가 데이터 컬럼의 고유값을 추출."""
        vals = view._distinct_values("gender")
        assert vals == [0, 1]

    def test_distinct_values_skips_continuous(self, view):
        """고유값이 limit 초과(연속형)면 빈 목록 — 과도한 표시 방지."""
        assert view._distinct_values("gender", limit=1) == []


# ── 즉시 편집 (AnyKeyPressed) ────────────────────────────────────────

class TestImmediateEditing:

    def test_edit_triggers_include_anykey(self, view):
        """임의 키 입력으로 편집 시작되도록 트리거 설정됨 (두 번 클릭 불필요)."""
        trig = view.table.editTriggers()
        assert trig & QAbstractItemView.EditTrigger.AnyKeyPressed

    def test_label_cell_opens_inline_editor(self, view):
        """라벨 셀이 인라인 편집기(QLineEdit)를 생성 — 즉시 입력 가능."""
        idx = view._model.index(0, 4)  # Label 열
        view.table.setCurrentIndex(idx)
        view.table.edit(idx)
        QApplication.processEvents()
        assert _editor(view) is not None, "라벨 셀은 인라인 편집기가 열려야 함"
        QApplication.processEvents()

    def test_value_missing_cells_no_inline_editor(self, view):
        """값(5)·결측(6) 셀은 인라인 편집기 없음 — 다이얼로그 전용."""
        assert view._var_delegate.createEditor(view.table, None, view._model.index(0, 5)) is None
        assert view._var_delegate.createEditor(view.table, None, view._model.index(0, 6)) is None


# ── 인접 행 이동 (Enter/아래키) ──────────────────────────────────────

class TestNavigation:

    def test_pending_navigate_moves_to_next_row(self, view):
        """delegate가 설정한 이동 정보로 editor 닫힘 시 아래 행으로 이동."""
        # delegate가 Enter/아래키에서 설정하는 상태를 모사: (dc, dr, src_row, src_col)
        view._var_delegate._pending_navigate = (0, 1, 0, 4)
        view._on_editor_closed(None)
        QApplication.processEvents()
        cur = view.table.currentIndex()
        assert cur.row() == 1, f"아래 행(1)로 이동해야 함, got {cur.row()}"
        assert cur.column() == 4
        QApplication.processEvents()

    def test_pending_navigate_up(self, view):
        """위 방향 이동."""
        view._var_delegate._pending_navigate = (0, -1, 1, 0)
        view._on_editor_closed(None)
        QApplication.processEvents()
        assert view.table.currentIndex().row() == 0
        QApplication.processEvents()

    def test_delegate_eventfilter_sets_navigate_on_enter(self, view):
        """delegate eventFilter가 Enter에서 _pending_navigate(아래 이동)를 설정."""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QStyleOptionViewItem
        idx = view._model.index(0, 4)
        view.table.setCurrentIndex(idx)
        editor = view._var_delegate.createEditor(view.table.viewport(), QStyleOptionViewItem(), idx)
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        # Enter → eventFilter가 커밋+closeEditor → _on_editor_closed가 아래 행으로 이동
        handled = view._var_delegate.eventFilter(editor, ev)
        QApplication.processEvents()
        assert handled is True, "Enter는 eventFilter가 처리해야 함"
        assert view.table.currentIndex().row() == 1, "Enter 후 아래 행으로 이동해야 함"
        QApplication.processEvents()
