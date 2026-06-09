"""Mixin that adds async analysis execution to analysis dialogs.

Inherit alongside QDialog::

    class FrequenciesDialog(QDialog, AnalysisDialogMixin):
        ...
        def _run(self) -> None:
            spec = self._build_spec()
            self._start_analysis(run_analysis, self._dataset, spec)

The mixin expects the host class to have:
- ``analysis_run`` signal (Signal(object)) — emitted with the AnalysisResult
- ``self._dataset`` — the active Dataset
- Optionally: a QPushButton or QAbstractButton named ``self._run_btn``
  (used to disable it while running; safe to omit)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import QMessageBox

from nuristat.ui.analysis_worker import run_analysis_async


class AnalysisDialogMixin:
    """Mixin: async analysis + uniform OK-disable + error dialog."""

    def _start_analysis(
        self,
        run_fn: Callable[..., Any],
        dataset: Any,
        spec: dict,
    ) -> None:
        self._set_running(True)
        run_analysis_async(
            owner=self,
            run_fn=run_fn,
            dataset=dataset,
            spec=spec,
            on_result=self._on_analysis_done,
            on_error=self._on_analysis_error,
        )

    def _on_analysis_done(self, result: Any) -> None:
        self.analysis_run.emit(result)  # type: ignore[attr-defined]
        self.accept()  # type: ignore[attr-defined]

    def _on_analysis_error(self, message: str) -> None:
        self._set_running(False)
        QMessageBox.critical(
            self,  # type: ignore[arg-type]
            "분석 오류",
            f"분석 중 오류가 발생했습니다:\n{message}",
        )

    def _set_running(self, running: bool) -> None:
        btn = getattr(self, "_run_btn", None)
        if btn is not None:
            btn.setEnabled(not running)
            if running:
                btn.setText("분석 중...")
            else:
                btn.setText("확인")
