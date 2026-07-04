"""P1-2: current_dataset 접근 시점에만 지연 동기화되는지 검증.

셀 편집마다 Dataset.data setter(컬럼 메타 재동기화 포함, O(cols))가
즉시 호출되지 않고, current_dataset을 실제로 읽는 시점에만
DataView.sync_dataset()을 경유해 동기화되어야 한다.

담당 에이전트: frontend, tester-unit
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from nuristat.core.dataset import Dataset


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app):
    from nuristat.ui.main_window import MainWindow
    w = MainWindow()
    return w


class TestCurrentDatasetLazySync:
    def _load(self, window):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds = Dataset(df, "t")
        window.current_dataset = ds
        window.data_view.set_dataset(ds)
        return ds

    def test_edit_does_not_call_dataset_data_setter_immediately(self, window):
        ds = self._load(window)
        setter_calls = []
        original_setter = type(ds).data.fset

        def counting_setter(self, value):
            setter_calls.append(value)
            original_setter(self, value)

        type(ds).data = property(type(ds).data.fget, counting_setter)
        try:
            window.data_view._model.setData(
                window.data_view._model.index(0, 0), "99", Qt.ItemDataRole.EditRole
            )
            assert setter_calls == []   # 편집 직후엔 setter 미호출(지연)
        finally:
            type(ds).data = property(type(ds).data.fget, original_setter)

    def test_current_dataset_read_triggers_sync(self, window):
        ds = self._load(window)
        window.data_view._model.setData(
            window.data_view._model.index(0, 0), "99", Qt.ItemDataRole.EditRole
        )
        assert window.data_view._dataset_stale is True
        synced = window.current_dataset   # 읽는 시점에 동기화되어야 함
        assert synced.data.iloc[0, 0] == 99
        assert window.data_view._dataset_stale is False
