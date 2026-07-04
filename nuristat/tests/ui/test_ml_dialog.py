"""MLDialog 비동기 전환 테스트 (P2-1)."""

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import AnalysisWorker
from nuristat.ui.dialogs.ml_dialog import MLDialog


@pytest.fixture
def dataset():
    return Dataset(pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0],
        "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    }))


def _select_features(dialog, names) -> None:
    for i in range(dialog.feature_list.count()):
        item = dialog.feature_list.item(i)
        item.setSelected(item.text() in names)


def _wait_for_worker(qapp, worker, timeout_ms: int = 5000) -> None:
    worker.wait(timeout_ms)
    for _ in range(20):
        qapp.processEvents()


def test_kmeans_disables_button_and_uses_background_worker(qapp, dataset):
    dialog = MLDialog(dataset)
    dialog.algo_combo.setCurrentIndex(0)  # kmeans
    _select_features(dialog, ["x1", "x2"])
    dialog.k_spin.setValue(2)

    dialog._run_analysis()

    assert isinstance(dialog._analysis_worker, AnalysisWorker)
    assert dialog.run_btn.isEnabled() is False

    _wait_for_worker(qapp, dialog._analysis_worker)

    assert dialog.run_btn.isEnabled() is True
    assert dialog.summary_table.rowCount() == 4


def test_decision_tree_emits_analysis_complete_after_worker_finishes(qapp, dataset):
    dialog = MLDialog(dataset)
    dialog.algo_combo.setCurrentIndex(1)  # decision_tree
    _select_features(dialog, ["x1", "x2"])
    dialog.target_combo.setCurrentText("y")

    received = []
    dialog.analysis_complete.connect(received.append)

    dialog._run_analysis()
    _wait_for_worker(qapp, dialog._analysis_worker)

    assert len(received) == 1
    assert dialog.summary_table.rowCount() == 5


def test_linear_regression_updates_detail_text(qapp, dataset):
    dialog = MLDialog(dataset)
    dialog.algo_combo.setCurrentIndex(2)  # linear_regression
    _select_features(dialog, ["x1", "x2"])
    dialog.target_combo.setCurrentText("y")

    dialog._run_analysis()
    _wait_for_worker(qapp, dialog._analysis_worker)

    assert "선형 회귀" in dialog.detail_text.text()


def test_missing_features_warns_without_starting_worker(qapp, dataset, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog = MLDialog(dataset)
    dialog.algo_combo.setCurrentIndex(0)  # kmeans

    dialog._run_analysis()

    assert len(warnings) == 1
    assert dialog._analysis_worker is None


def test_missing_target_warns_without_starting_worker(qapp, dataset, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog = MLDialog(dataset)
    dialog.algo_combo.setCurrentIndex(1)  # decision_tree
    _select_features(dialog, ["x1", "x2"])
    dialog.target_combo.setCurrentText("(없음)")

    dialog._run_analysis()

    assert len(warnings) == 1
    assert dialog._analysis_worker is None
