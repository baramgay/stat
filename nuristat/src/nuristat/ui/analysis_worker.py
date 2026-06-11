"""분석 QThread 워커 — GUI 프리징 없이 무거운 분석을 백그라운드에서 실행."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class AnalysisWorker(QThread):
    """단일 분석 함수를 백그라운드 스레드에서 실행하는 워커."""

    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, run_fn: Callable[[], Any], parent: Any = None) -> None:
        super().__init__(parent)
        self._run_fn = run_fn

    def run(self) -> None:
        try:
            result = self._run_fn()
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


def run_analysis_async(
    owner: Any,
    run_fn: Callable[..., Any],
    dataset: Any,
    spec: dict,
    on_result: Callable[[Any], None],
    on_error: Callable[[str], None],
) -> AnalysisWorker:
    """run_fn(dataset, spec)을 백그라운드에서 실행하는 헬퍼.

    owner._analysis_worker에 워커를 저장해 GC를 방지한다.
    """
    from PySide6.QtCore import QObject
    _parent = owner if isinstance(owner, QObject) else None
    worker = AnalysisWorker(lambda: run_fn(dataset, spec), parent=_parent)
    owner._analysis_worker = worker
    worker.result_ready.connect(on_result)
    worker.error_occurred.connect(on_error)
    worker.start()
    return worker
