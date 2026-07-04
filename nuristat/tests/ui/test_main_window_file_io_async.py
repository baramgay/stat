"""파일 열기/저장 백그라운드 전환 테스트 (P3-2).

read_csv/read_excel/read_sav/load_project/save_project 호출을
AnalysisWorker 경유 백그라운드 스레드에서 실행하도록 전환.
로드 중 재클릭(중복 열기)은 무시되어야 한다.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app):
    from nuristat.ui.main_window import MainWindow
    w = MainWindow()
    w._settings.clear_recent_files()
    return w


def _wait(app, window, timeout_ms: int = 3000) -> None:
    worker = window._file_task_worker
    if worker is not None:
        worker.wait(timeout_ms)
    for _ in range(20):
        app.processEvents()


def test_import_csv_runs_in_background_worker(window, app, monkeypatch, tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(p, index=False)
    monkeypatch.setattr(
        "nuristat.ui.main_window.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(p), ""),
    )

    window._import_csv()

    assert window._file_task_worker is not None
    _wait(app, window)

    assert window.current_dataset is not None
    assert list(window.current_dataset.data.columns) == ["x"]
    assert window._file_task_worker is None


def test_duplicate_open_is_ignored_while_loading(window, app, monkeypatch, tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(p, index=False)
    monkeypatch.setattr(
        "nuristat.ui.main_window.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(p), ""),
    )

    window._import_csv()
    first_worker = window._file_task_worker
    assert first_worker is not None

    window._import_csv()
    assert window._file_task_worker is first_worker

    _wait(app, window)


def test_import_csv_error_shows_critical_message(window, app, monkeypatch):
    monkeypatch.setattr(
        "nuristat.ui.main_window.QFileDialog.getOpenFileName",
        lambda *a, **k: ("/no/such/file.csv", ""),
    )
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    window._import_csv()
    _wait(app, window)

    assert len(errors) == 1
    assert window._file_task_worker is None
