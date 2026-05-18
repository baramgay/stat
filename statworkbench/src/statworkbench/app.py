"""Application entry point for StatWorkbench.

Provides the StatWorkbenchApp class which initializes and runs
the main application window.
"""

from PySide6.QtWidgets import QApplication
from typing import Optional

from statworkbench.ui.main_window import MainWindow


class StatWorkbenchApp:
    """Application wrapper for StatWorkbench.

    Usage:
        app = StatWorkbenchApp()
        app.run()
    """

    def __init__(self) -> None:
        self._app: Optional[QApplication] = None
        self._window: Optional[MainWindow] = None

    def run(self) -> int:
        """Run the application event loop."""
        self._app = QApplication([])
        self._app.setApplicationName("StatWorkbench")
        self._app.setApplicationVersion("0.1.0")
        self._app.setOrganizationName("StatWorkbench")

        self._window = MainWindow()
        self._window.show()

        return self._app.exec()

    def get_window(self) -> Optional[MainWindow]:
        """Return the main window instance."""
        return self._window
