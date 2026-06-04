"""Application entry point for NuriStat.

Provides the NuriStatApp class which initializes and runs
the main application window.
"""


from PySide6.QtWidgets import QApplication

from nuristat.ui.main_window import MainWindow


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

        return self._app.exec()

    def get_window(self) -> MainWindow | None:
        """Return the main window instance."""
        return self._window
