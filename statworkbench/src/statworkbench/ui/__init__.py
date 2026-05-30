"""UI module for StatWorkbench.

Provides PySide6-based GUI components including spreadsheet views, analysis dialogs,
and Qt model implementations for dataset and variable display.
"""

from statworkbench.ui.data_view import DataView
from statworkbench.ui.main_window import MainWindow
from statworkbench.ui.output_view import OutputView
from statworkbench.ui.variable_view import VariableView

__all__ = [
    "MainWindow",
    "DataView",
    "VariableView",
    "OutputView",
]
