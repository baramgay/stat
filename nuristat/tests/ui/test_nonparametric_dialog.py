"""NonparametricDialog 비동기 전환 테스트 (P2-1)."""

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import AnalysisWorker
from nuristat.ui.dialogs.nonparametric_dialog import NonparametricDialog


@pytest.fixture
def dataset():
    return Dataset(pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "group": ["A", "A", "A", "B", "B", "B"],
    }))


def _wait_for_worker(qapp, worker, timeout_ms: int = 3000) -> None:
    worker.wait(timeout_ms)
    for _ in range(20):
        qapp.processEvents()


def test_run_analysis_disables_button_and_uses_background_worker(qapp, dataset):
    dialog = NonparametricDialog(dataset)
    dialog.test_combo.setCurrentText("value")
    dialog.group_combo.setCurrentText("group")

    dialog._run_analysis()

    # 워커 시작 직후(완료 전) 버튼은 비활성화 상태여야 한다.
    assert isinstance(dialog._analysis_worker, AnalysisWorker)
    assert dialog.btn_run.isEnabled() is False

    _wait_for_worker(qapp, dialog._analysis_worker)

    assert dialog.btn_run.isEnabled() is True
    assert "Mann-Whitney" in dialog.result_text.toPlainText()


def test_run_analysis_emits_analysis_run_after_worker_finishes(qapp, dataset):
    dialog = NonparametricDialog(dataset)
    dialog.test_combo.setCurrentText("value")
    dialog.group_combo.setCurrentText("group")

    received = []
    dialog.analysis_run.connect(received.append)

    dialog._run_analysis()
    _wait_for_worker(qapp, dialog._analysis_worker)

    assert len(received) == 1
    assert "Mann-Whitney" in received[0].title
    assert "Mann-Whitney" in received[0].text_blocks[0]


def test_missing_group_var_warns_without_starting_worker(qapp, dataset, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog = NonparametricDialog(dataset)
    dialog.test_combo.setCurrentText("value")
    dialog.group_combo.setCurrentText("(없음)")

    dialog._run_analysis()

    assert len(warnings) == 1
    assert dialog._analysis_worker is None
