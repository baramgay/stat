"""Tests for AnalysisWorker and run_analysis_async."""

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import AnalysisWorker, run_analysis_async


@pytest.fixture
def dataset():
    return Dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


def _wait_for_worker(qapp, worker, timeout_ms: int = 3000) -> None:
    """Wait for thread + pump the event loop so cross-thread signals are delivered."""
    worker.wait(timeout_ms)
    qapp.processEvents()


# ---------------------------------------------------------------------------
# AnalysisWorker unit tests (no-arg closure interface)
# ---------------------------------------------------------------------------

class TestAnalysisWorker:
    def test_result_ready_emitted(self, qapp):
        sentinel = {"value": 42}
        worker = AnalysisWorker(lambda: sentinel)
        received = []
        worker.result_ready.connect(received.append)
        worker.start()
        _wait_for_worker(qapp, worker)
        assert received == [sentinel]

    def test_error_occurred_emitted_on_exception(self, qapp):
        worker = AnalysisWorker(lambda: (_ for _ in ()).throw(RuntimeError("deliberate failure")))
        errors = []
        worker.error_occurred.connect(errors.append)
        worker.start()
        _wait_for_worker(qapp, worker)
        assert len(errors) == 1
        assert "deliberate failure" in errors[0]

    def test_no_result_on_exception(self, qapp):
        worker = AnalysisWorker(lambda: 1 / 0)
        results = []
        worker.result_ready.connect(results.append)
        worker.error_occurred.connect(lambda _: None)
        worker.start()
        _wait_for_worker(qapp, worker)
        assert results == []

    def test_result_value_passed_through(self, qapp):
        worker = AnalysisWorker(lambda: [1, 2, 3])
        received = []
        worker.result_ready.connect(received.append)
        worker.start()
        _wait_for_worker(qapp, worker)
        assert received == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# run_analysis_async helper
# ---------------------------------------------------------------------------

class TestRunAnalysisAsync:
    def test_worker_stored_on_owner(self, qapp, dataset):
        class Owner:
            _analysis_worker = None

        owner = Owner()
        worker = run_analysis_async(
            owner=owner,
            run_fn=lambda ds, spec: "result",
            dataset=dataset,
            spec={},
            on_result=lambda r: None,
            on_error=lambda e: None,
        )
        _wait_for_worker(qapp, worker)
        assert owner._analysis_worker is worker

    def test_on_result_callback_called(self, qapp, dataset):
        class Owner:
            _analysis_worker = None

        received = []
        owner = Owner()
        worker = run_analysis_async(
            owner=owner,
            run_fn=lambda ds, spec: {"status": "ok"},
            dataset=dataset,
            spec={},
            on_result=received.append,
            on_error=lambda e: None,
        )
        _wait_for_worker(qapp, worker)
        assert received == [{"status": "ok"}]

    def test_on_error_callback_called(self, qapp, dataset):
        class Owner:
            _analysis_worker = None

        errors = []
        owner = Owner()

        def failing(ds, spec):
            raise TypeError("bad type")

        worker = run_analysis_async(
            owner=owner,
            run_fn=failing,
            dataset=dataset,
            spec={},
            on_result=lambda r: None,
            on_error=errors.append,
        )
        _wait_for_worker(qapp, worker)
        assert any("bad type" in e for e in errors)

    def test_dataset_and_spec_forwarded_to_run_fn(self, qapp, dataset):
        class Owner:
            _analysis_worker = None

        captured = []

        def capture_fn(ds, spec):
            captured.append((ds, spec))
            return "done"

        spec = {"key": "val"}
        owner = Owner()
        worker = run_analysis_async(
            owner=owner,
            run_fn=capture_fn,
            dataset=dataset,
            spec=spec,
            on_result=lambda r: None,
            on_error=lambda e: None,
        )
        _wait_for_worker(qapp, worker)
        assert len(captured) == 1
        assert captured[0][0] is dataset
        assert captured[0][1] is spec
