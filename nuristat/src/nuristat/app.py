"""Application entry point for NuriStat.

Provides the NuriStatApp class which initializes and runs
the main application window.
"""

import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from nuristat.ui.main_window import MainWindow

# 창 표시 후 유휴 시간에 무거운 모듈(scipy/chardet)을 백그라운드 스레드에서
# 미리 임포트해 둔다. import lock으로 스레드 안전하며, 사용자가 실제 분석/
# 파일 열기를 실행할 때는 이미 sys.modules에 캐시되어 있어 지연이 없다.
_PREWARM_DELAY_MS = 300


def _prewarm_heavy_modules() -> None:
    import chardet  # noqa: F401
    from nuristat.analysis import assumptions  # noqa: F401


class NuriStatApp:
    """Application wrapper for NuriStat.

    Usage:
        app = NuriStatApp()
        app.run()
    """

    def __init__(self) -> None:
        self._app: QApplication | None = None
        self._window: MainWindow | None = None

    def run(self) -> int:
        """Run the application event loop."""
        self._app = QApplication([])
        self._app.setApplicationName("NuriStat")
        self._app.setApplicationVersion("0.1.0")
        self._app.setOrganizationName("NuriStat")

        self._window = MainWindow()
        self._window.show()

        QTimer.singleShot(
            _PREWARM_DELAY_MS,
            lambda: threading.Thread(target=_prewarm_heavy_modules, daemon=True).start(),
        )

        return self._app.exec()

    def get_window(self) -> MainWindow | None:
        """Return the main window instance."""
        return self._window
