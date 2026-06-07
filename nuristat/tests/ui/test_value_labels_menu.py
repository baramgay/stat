"""보기 > 값 라벨 표시 토글 검증 (SPSS: View > Value Labels).

값 라벨 표시 기능이 구현돼 있었으나 메뉴/툴바에 노출되지 않아 도달 불가능하던
문제를 해소 — View 메뉴 + Ctrl+L로 코드↔라벨 전환.

담당 에이전트: frontend, tester-unit
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app):
    from nuristat.ui.main_window import MainWindow
    w = MainWindow()
    df = pd.DataFrame({"sex": [0, 1, 0], "score": [10, 20, 30]})
    variables = {
        "sex": VariableMeta(name="sex", storage_type=StorageType.INTEGER,
                            measure=MeasureType.NOMINAL, value_labels={0: "남", 1: "여"}),
        "score": VariableMeta(name="score", storage_type=StorageType.INTEGER,
                              measure=MeasureType.SCALE),
    }
    ds = Dataset(df, "t", variables)
    w.data_view.set_dataset(ds)
    w.current_dataset = ds
    # close()는 호출하지 않는다 — closeEvent가 미저장 확인 모달을 띄워 헤드리스에서
    # 멈추기 때문. 테스트 종료 시 위젯은 가비지 컬렉션에 맡긴다.
    return w


class TestValueLabelsMenu:
    def test_action_exists_and_checkable(self, window):
        assert hasattr(window, "_value_labels_action")
        assert window._value_labels_action.isCheckable()
        assert window._value_labels_action.shortcut().toString() == "Ctrl+L"

    def test_toggle_flips_model_state(self, window):
        model = window.data_view._model
        assert model.show_value_labels is False
        window._toggle_value_labels()
        assert model.show_value_labels is True
        assert window._value_labels_action.isChecked() is True
        window._toggle_value_labels()
        assert model.show_value_labels is False
        assert window._value_labels_action.isChecked() is False

    def test_labels_shown_in_display_role(self, window):
        """토글 ON 시 데이터 셀이 코드 대신 라벨을 표시."""
        from PySide6.QtCore import Qt
        model = window.data_view._model
        idx = model.index(0, 0)   # sex=0
        assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "0"
        window._toggle_value_labels()
        assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "남"
