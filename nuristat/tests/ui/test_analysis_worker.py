"""Tests for AnalysisWorker and run_analysis_async."""

import time

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import AnalysisWorker, run_analysis_async


@pytest.fixture
def dataset():
    return Dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


# ---------------------------------------------------------------------------
# AnalysisWorker unit tests
# ---------------------------------------------------------------------------

def _wait_for_worker(qapp, worker, timeout_ms: int = 3000) -> None:
    """Wait for thread + pump the event loop so cross-thread signals are delivered."""
    worker.wait(timeout_ms)
    qapp.processEvents()


class TestAnalysisWorker:
    def test_result_ready_emitted(self, qapp, dataset):
        sentinel = {"value": 42}
        run_fn = lambda ds, spec: sentinel  # noqa: E731

        received = []
        worker = AnalysisWorker(run_fn, dataset, {})
        worker.result_ready.connect(lambda r: received.append(r))
        worker.start()
        _wait_for_worker(qapp, worker)

        assert received == [sentinel]

    def test_error_occurred_emitted_on_exception(self, qapp, dataset):
        def bad_fn(ds, spec):
            raise RuntimeError("deliberate failure")

        errors = []
        worker = AnalysisWorker(bad_fn, dataset, {})
        worker.error_occurred.connect(errors.append)
        worker.start()
        _wait_for_worker(qapp, worker)

        assert len(errors) == 1
        assert "deliberate failure" in errors[0]

    def test_no_result_on_exception(self, qapp, dataset):
        def bad_fn(ds, spec):
            raise ValueError("oops")

        results = []
        worker = AnalysisWorker(bad_fn, dataset, {})
        worker.result_ready.connect(results.append)
        worker.error_occurred.connect(lambda _: None)
        worker.start()
        _wait_for_worker(qapp, worker)

        assert results == []


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
