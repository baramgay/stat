"""UI module for NuriStat.

Provides PySide6-based GUI components including spreadsheet views, analysis dialogs,
and Qt model implementations for dataset and variable display.
"""

from nuristat.ui.data_view import DataView
from nuristat.ui.main_window import MainWindow
from nuristat.ui.output_view import OutputView
from nuristat.ui.variable_view import VariableView

__all__ = [
    "MainWindow",
    "DataView",
    "VariableView",
    "OutputView",
]
