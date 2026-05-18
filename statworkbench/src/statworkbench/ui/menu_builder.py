"""Menu builder for StatWorkbench main window."""

from PySide6.QtWidgets import QMenuBar, QMenu, QWidget
from PySide6.QtGui import QAction, QKeySequence
from typing import Callable, Optional


class MenuBuilder:
    """Builds the menu structure for the main window."""

    def __init__(self, menubar: QMenuBar, parent: Optional[QWidget] = None) -> None:
        self.menubar = menubar
        self.parent = parent
        self._actions: dict[str, QAction] = {}

    def add_action(
        self,
        menu: QMenu,
        text: str,
        shortcut: Optional[str] = None,
        triggered: Optional[Callable] = None,
        enabled: bool = True,
    ) -> QAction:
        """Add an action to a menu."""
        action = QAction(text, self.parent)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if triggered:
            action.triggered.connect(triggered)
        action.setEnabled(enabled)
        menu.addAction(action)
        self._actions[text] = action
        return action

    def build_file_menu(self) -> QMenu:
        """Build the File menu."""
        menu = self.menubar.addMenu("&File")
        return menu

    def build_edit_menu(self) -> QMenu:
        """Build the Edit menu."""
        menu = self.menubar.addMenu("&Edit")
        return menu

    def build_data_menu(self) -> QMenu:
        """Build the Data menu."""
        menu = self.menubar.addMenu("&Data")
        return menu

    def build_analyze_menu(self) -> QMenu:
        """Build the Analyze menu."""
        menu = self.menubar.addMenu("&Analyze")
        return menu

    def build_tools_menu(self) -> QMenu:
        """Build the Tools menu."""
        menu = self.menubar.addMenu("&Tools")
        return menu

    def build_help_menu(self) -> QMenu:
        """Build the Help menu."""
        menu = self.menubar.addMenu("&Help")
        return menu

    def get_action(self, name: str) -> Optional[QAction]:
        """Get an action by name."""
        return self._actions.get(name)
