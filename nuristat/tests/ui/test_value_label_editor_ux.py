"""ValueLabelsDialog UX 테스트 — 중복 체크, 편집."""

import pytest
from PySide6.QtWidgets import QTableWidgetItem

from nuristat.ui.dialogs.variable_editor import ValueLabelsDialog


def _make_dialog(labels=None, existing_values=None):
    return ValueLabelsDialog(labels or {}, existing_values=existing_values or [])


def test_initial_load_shows_existing_values(qapp):
    """기존 라벨이 테이블에 표시돼야 한다."""
    dlg = _make_dialog({1: "남성", 2: "여성"})
    assert dlg.table.rowCount() == 2


def test_existing_data_values_shown(qapp):
    """데이터의 실제 값이 기존 라벨 없어도 먼저 표시돼야 한다."""
    dlg = _make_dialog({}, existing_values=[1, 2, 3])
    assert dlg.table.rowCount() == 3


def test_get_value_labels_round_trip(qapp):
    """편집 후 get_value_labels()가 올바른 dict를 반환해야 한다."""
    dlg = _make_dialog({1: "남성", 2: "여성"})
    result = dlg.get_value_labels()
    assert result[1] == "남성"
    assert result[2] == "여성"


def test_add_row_button_adds_empty_row(qapp):
    """Add 버튼 클릭 시 빈 행이 추가돼야 한다."""
    dlg = _make_dialog({})
    before = dlg.table.rowCount()
    dlg._add_row()
    assert dlg.table.rowCount() == before + 1


def test_remove_selected_row(qapp):
    """선택한 행 삭제가 동작해야 한다."""
    dlg = _make_dialog({1: "A", 2: "B"})
    dlg.table.setCurrentCell(0, 0)
    dlg.table.selectRow(0)
    dlg._remove_selected()
    assert dlg.table.rowCount() == 1


def test_duplicate_key_detected(qapp, monkeypatch):
    """중복 키 입력 시 경고 다이얼로그가 표시돼야 한다."""
    dlg = _make_dialog({})
    # 테이블에 중복 행 수동 삽입
    dlg.table.insertRow(0)
    dlg.table.setItem(0, 0, QTableWidgetItem("1"))
    dlg.table.setItem(0, 1, QTableWidgetItem("남성"))
    dlg.table.insertRow(1)
    dlg.table.setItem(1, 0, QTableWidgetItem("1"))  # 중복
    dlg.table.setItem(1, 1, QTableWidgetItem("여성"))

    warned = []
    # QMessageBox.warning → Cancel 반환하면 accept()가 호출 안 됨
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **kw: (warned.append(True), QMessageBox.StandardButton.Cancel)[1])
    dlg._on_ok()
    assert warned, "중복 키 경고가 표시돼야 함"
