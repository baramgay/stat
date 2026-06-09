"""Async analysis worker — wraps any run_analysis function in a QThread.

Usage::

    run_analysis_async(
        owner=self,
        run_fn=run_analysis,
        dataset=self._dataset,
        spec=spec,
        on_result=self._on_done,
        on_error=self._on_err,
    )

The owner stores a strong reference to the worker so the GC doesn't collect
it before the thread finishes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class AnalysisWorker(QThread):
    """Runs a single run_analysis(dataset, spec) call off the GUI thread."""

    result_ready = Signal(object)   # emits AnalysisResult
    error_occurred = Signal(str)    # emits error message string

    def __init__(
        self,
        run_fn: Callable[..., Any],
        dataset: Any,
        spec: dict,
    ) -> None:
        super().__init__()
        self._run_fn = run_fn
        self._dataset = dataset
        self._spec = spec

    def run(self) -> None:
        try:
            result = self._run_fn(self._dataset, self._spec)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))


def run_analysis_async(
    owner: Any,
    run_fn: Callable[..., Any],
    dataset: Any,
    spec: dict,
    on_result: Callable[[Any], None],
    on_error: Callable[[str], None],
) -> AnalysisWorker:
    """Create, store, connect, and start an AnalysisWorker.

    Stores the worker on *owner._analysis_worker* to prevent GC.
    Returns the worker for callers that need to check isRunning() etc.
    """
    worker = AnalysisWorker(run_fn, dataset, spec)
    owner._analysis_worker = worker   # keep alive until thread ends
    worker.result_ready.connect(on_result)
    worker.error_occurred.connect(on_error)
    worker.start()
    return worker
