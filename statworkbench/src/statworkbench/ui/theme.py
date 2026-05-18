"""Theme and styling system for StatWorkbench.

Supports Light (default) and Dark (OLED-optimized) themes.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any


class ThemeMode(str, Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"


class ThemeColors:
    """Color palette for a specific theme."""

    def __init__(self, mode: ThemeMode) -> None:
        self.mode = mode
        if mode == ThemeMode.DARK:
            self._init_dark()
        else:
            self._init_light()

    def _init_light(self) -> None:
        """Initialize light theme colors."""
        # Primary
        self.PRIMARY = "#1a5276"
        self.PRIMARY_LIGHT = "#2e86c1"
        self.PRIMARY_DARK = "#154360"
        self.ACCENT = "#2874a6"

        # Secondary
        self.SECONDARY = "#5dade2"
        self.SECONDARY_LIGHT = "#85c1e9"
        self.SECONDARY_DARK = "#3498db"

        # Background
        self.BG_MAIN = "#f0f2f5"
        self.BG_CARD = "#ffffff"
        self.BG_SIDEBAR = "#e8eaed"
        self.BG_HEADER = "#d6eaf8"
        self.BG_SELECTED = "#aed6f1"
        self.BG_HOVER = "#d6eaf8"
        self.BG_INPUT = "#ffffff"

        # Text - ensure high contrast
        self.TEXT_PRIMARY = "#1a1a2e"
        self.TEXT_SECONDARY = "#4a4a5a"
        self.TEXT_MUTED = "#7a7a8a"
        self.TEXT_ON_PRIMARY = "#ffffff"
        self.TEXT_ON_DARK = "#ffffff"

        # Border
        self.BORDER = "#c0c4cc"
        self.BORDER_LIGHT = "#e0e4e8"
        self.BORDER_FOCUS = "#2e86c1"

        # Status
        self.SUCCESS = "#27ae60"
        self.WARNING = "#f39c12"
        self.ERROR = "#e74c3c"
        self.INFO = "#3498db"

        # Table
        self.TABLE_HEADER_BG = "#d6eaf8"
        self.TABLE_ROW_ALT = "#f5f7fa"
        self.TABLE_BORDER = "#c0c4cc"
        self.TABLE_TEXT = "#1a1a2e"

        # Measure type colors (vibrant for visibility)
        self.MEASURE_SCALE = "#1a5276"
        self.MEASURE_NOMINAL = "#7d3c98"
        self.MEASURE_ORDINAL = "#2874a6"
        self.MEASURE_BINARY = "#239b56"
        self.MEASURE_DATETIME = "#b9770e"
        self.MEASURE_TEXT = "#5d6d7e"

        # Scrollbar
        self.SCROLLBAR_BG = "#e0e4e8"
        self.SCROLLBAR_HANDLE = "#a0a4a8"
        self.SCROLLBAR_HANDLE_HOVER = "#808488"

    def _init_dark(self) -> None:
        """Initialize dark theme colors optimized for OLED."""
        # Primary - brighter for dark background
        self.PRIMARY = "#5dade2"
        self.PRIMARY_LIGHT = "#85c1e9"
        self.PRIMARY_DARK = "#3498db"
        self.ACCENT = "#5dade2"

        # Secondary
        self.SECONDARY = "#2874a6"
        self.SECONDARY_LIGHT = "#5dade2"
        self.SECONDARY_DARK = "#1a5276"

        # Background - true black for OLED
        self.BG_MAIN = "#0a0a0a"
        self.BG_CARD = "#141414"
        self.BG_SIDEBAR = "#0f0f0f"
        self.BG_HEADER = "#1a1a2e"
        self.BG_SELECTED = "#1a3a5c"
        self.BG_HOVER = "#1f1f3a"
        self.BG_INPUT = "#1a1a1a"

        # Text - bright for OLED
        self.TEXT_PRIMARY = "#e8e8f0"
        self.TEXT_SECONDARY = "#b0b0c0"
        self.TEXT_MUTED = "#707080"
        self.TEXT_ON_PRIMARY = "#0a0a0a"
        self.TEXT_ON_DARK = "#e8e8f0"

        # Border
        self.BORDER = "#2a2a3a"
        self.BORDER_LIGHT = "#1a1a2a"
        self.BORDER_FOCUS = "#5dade2"

        # Status - brighter for dark
        self.SUCCESS = "#2ecc71"
        self.WARNING = "#f1c40f"
        self.ERROR = "#e74c3c"
        self.INFO = "#5dade2"

        # Table
        self.TABLE_HEADER_BG = "#1a3a5c"
        self.TABLE_ROW_ALT = "#0f0f1a"
        self.TABLE_BORDER = "#2a2a3a"
        self.TABLE_TEXT = "#e8e8f0"

        # Measure type colors (brighter for dark)
        self.MEASURE_SCALE = "#5dade2"
        self.MEASURE_NOMINAL = "#af7ac5"
        self.MEASURE_ORDINAL = "#5dade2"
        self.MEASURE_BINARY = "#58d68d"
        self.MEASURE_DATETIME = "#f5b041"
        self.MEASURE_TEXT = "#aab7b8"

        # Scrollbar
        self.SCROLLBAR_BG = "#1a1a2a"
        self.SCROLLBAR_HANDLE = "#4a4a5a"
        self.SCROLLBAR_HANDLE_HOVER = "#6a6a7a"


class ThemeManager:
    """Manages application theme."""

    _instance: "ThemeManager | None" = None
    _current_mode: ThemeMode = ThemeMode.LIGHT

    def __new__(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def current_mode(self) -> ThemeMode:
        """Get current theme mode."""
        return self._current_mode

    def set_mode(self, mode: ThemeMode) -> None:
        """Set theme mode."""
        self._current_mode = mode

    def toggle(self) -> ThemeMode:
        """Toggle between light and dark."""
        if self._current_mode == ThemeMode.LIGHT:
            self._current_mode = ThemeMode.DARK
        else:
            self._current_mode = ThemeMode.LIGHT
        return self._current_mode

    def get_colors(self) -> ThemeColors:
        """Get current theme colors."""
        return ThemeColors(self._current_mode)


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    return ThemeManager()


def get_application_stylesheet(mode: ThemeMode | None = None) -> str:
    """Return the application-wide stylesheet.

    Args:
        mode: Theme mode. If None, uses current theme manager setting.
    """
    if mode is None:
        mode = get_theme_manager().current_mode
    t = ThemeColors(mode)

    return f"""
    QMainWindow {{
        background-color: {t.BG_MAIN};
    }}

    QWidget {{
        background-color: {t.BG_MAIN};
        color: {t.TEXT_PRIMARY};
    }}

    QMenuBar {{
        background-color: {t.BG_CARD};
        border-bottom: 1px solid {t.BORDER};
        padding: 2px;
        color: {t.TEXT_PRIMARY};
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 4px 12px;
        border-radius: 3px;
        color: {t.TEXT_PRIMARY};
    }}

    QMenuBar::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QMenu {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        padding: 4px;
        color: {t.TEXT_PRIMARY};
    }}

    QMenu::item {{
        padding: 6px 24px;
        border-radius: 3px;
        color: {t.TEXT_PRIMARY};
    }}

    QMenu::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {t.BORDER};
        margin: 4px 8px;
    }}

    QToolBar {{
        background-color: {t.BG_CARD};
        border-bottom: 1px solid {t.BORDER};
        padding: 4px;
        spacing: 4px;
    }}

    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QToolButton:hover {{
        background-color: {t.BG_HOVER};
        border-color: {t.BORDER};
    }}

    QStatusBar {{
        background-color: {t.BG_SIDEBAR};
        border-top: 1px solid {t.BORDER};
        color: {t.TEXT_SECONDARY};
    }}

    QTabWidget::pane {{
        border: 1px solid {t.BORDER};
        background-color: {t.BG_CARD};
        border-radius: 4px;
    }}

    QTabBar::tab {{
        background-color: {t.BG_SIDEBAR};
        border: 1px solid {t.BORDER};
        border-bottom: none;
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: {t.TEXT_SECONDARY};
    }}

    QTabBar::tab:selected {{
        background-color: {t.BG_CARD};
        border-bottom: 2px solid {t.PRIMARY};
        color: {t.PRIMARY};
        font-weight: bold;
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {t.BG_HOVER};
        color: {t.TEXT_PRIMARY};
    }}

    QGroupBox {{
        font-weight: bold;
        color: {t.TEXT_PRIMARY};
        border: 1px solid {t.BORDER};
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 8px;
        background-color: {t.BG_CARD};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {t.PRIMARY};
    }}

    QPushButton {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px 16px;
        color: {t.TEXT_PRIMARY};
    }}

    QPushButton:hover {{
        background-color: {t.BG_HOVER};
        border-color: {t.BORDER_FOCUS};
    }}

    QPushButton:default {{
        background-color: {t.PRIMARY};
        border-color: {t.PRIMARY_DARK};
        color: {t.TEXT_ON_PRIMARY};
        font-weight: bold;
    }}

    QPushButton:default:hover {{
        background-color: {t.PRIMARY_LIGHT};
    }}

    QPushButton:pressed {{
        background-color: {t.PRIMARY_DARK};
    }}

    QPushButton:disabled {{
        background-color: {t.BG_SIDEBAR};
        color: {t.TEXT_MUTED};
        border-color: {t.BORDER};
    }}

    QLineEdit {{
        background-color: {t.BG_INPUT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 6px;
        color: {t.TEXT_PRIMARY};
        selection-background-color: {t.BG_SELECTED};
    }}

    QLineEdit:focus {{
        border-color: {t.BORDER_FOCUS};
    }}

    QTextEdit, QPlainTextEdit {{
        background-color: {t.BG_INPUT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px;
        color: {t.TEXT_PRIMARY};
        selection-background-color: {t.BG_SELECTED};
    }}

    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {t.BORDER_FOCUS};
    }}

    QListWidget {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 2px;
        alternate-background-color: {t.TABLE_ROW_ALT};
        color: {t.TEXT_PRIMARY};
    }}

    QListWidget::item {{
        padding: 4px 8px;
        border-radius: 2px;
        color: {t.TEXT_PRIMARY};
    }}

    QListWidget::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QListWidget::item:hover {{
        background-color: {t.BG_HOVER};
    }}

    QComboBox {{
        background-color: {t.BG_INPUT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        min-width: 80px;
        color: {t.TEXT_PRIMARY};
    }}

    QComboBox:focus {{
        border-color: {t.BORDER_FOCUS};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        selection-background-color: {t.BG_SELECTED};
        color: {t.TEXT_PRIMARY};
    }}

    QCheckBox {{
        spacing: 6px;
        color: {t.TEXT_PRIMARY};
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid {t.BORDER};
        background-color: {t.BG_INPUT};
    }}

    QCheckBox::indicator:checked {{
        background-color: {t.PRIMARY};
        border-color: {t.PRIMARY};
    }}

    QSplitter::handle {{
        background-color: {t.BORDER};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    QTreeWidget {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        alternate-background-color: {t.TABLE_ROW_ALT};
        color: {t.TEXT_PRIMARY};
    }}

    QTreeWidget::item {{
        padding: 4px 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QTreeWidget::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QHeaderView::section {{
        background-color: {t.TABLE_HEADER_BG};
        color: {t.TABLE_TEXT};
        padding: 6px 8px;
        border: 1px solid {t.BORDER};
        font-weight: bold;
    }}

    QTableView {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        gridline-color: {t.BORDER};
        selection-background-color: {t.BG_SELECTED};
        selection-color: {t.TEXT_PRIMARY};
        color: {t.TEXT_PRIMARY};
    }}

    QTableView::item {{
        padding: 4px 8px;
        color: {t.TEXT_PRIMARY};
    }}

    QTableView::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.TEXT_PRIMARY};
    }}

    QTableView QTableCornerButton::section {{
        background-color: {t.TABLE_HEADER_BG};
        border: 1px solid {t.BORDER};
    }}

    QScrollBar:vertical {{
        background-color: {t.SCROLLBAR_BG};
        width: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {t.SCROLLBAR_HANDLE};
        border-radius: 6px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {t.SCROLLBAR_HANDLE_HOVER};
    }}

    QScrollBar:horizontal {{
        background-color: {t.SCROLLBAR_BG};
        height: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {t.SCROLLBAR_HANDLE};
        border-radius: 6px;
        min-width: 20px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {t.SCROLLBAR_HANDLE_HOVER};
    }}

    QLabel {{
        color: {t.TEXT_PRIMARY};
    }}

    QMessageBox {{
        background-color: {t.BG_CARD};
        color: {t.TEXT_PRIMARY};
    }}

    QDialog {{
        background-color: {t.BG_MAIN};
        color: {t.TEXT_PRIMARY};
    }}

    QSpinBox, QDoubleSpinBox {{
        background-color: {t.BG_INPUT};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px;
        color: {t.TEXT_PRIMARY};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {t.BORDER_FOCUS};
    }}

    QRadioButton {{
        color: {t.TEXT_PRIMARY};
    }}

    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 1px solid {t.BORDER};
        background-color: {t.BG_INPUT};
    }}

    QRadioButton::indicator:checked {{
        background-color: {t.PRIMARY};
        border-color: {t.PRIMARY};
    }}

    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {t.BG_SIDEBAR};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 16px;
        height: 16px;
        background-color: {t.PRIMARY};
        border-radius: 8px;
    }}

    QProgressBar {{
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        text-align: center;
        color: {t.TEXT_PRIMARY};
    }}

    QProgressBar::chunk {{
        background-color: {t.PRIMARY};
        border-radius: 3px;
    }}
"""


def get_output_html_styles(mode: ThemeMode | None = None) -> str:
    """Return HTML/CSS styles for output view tables."""
    if mode is None:
        mode = get_theme_manager().current_mode
    t = ThemeColors(mode)

    return f"""
    <style>
        body {{
            font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            font-size: 13px;
            color: {t.TEXT_PRIMARY};
            background-color: {t.BG_CARD};
            margin: 0;
            padding: 16px;
        }}
        h2 {{
            color: {t.PRIMARY};
            font-size: 18px;
            font-weight: bold;
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid {t.PRIMARY};
        }}
        h3 {{
            color: {t.TEXT_PRIMARY};
            font-size: 14px;
            font-weight: bold;
            margin: 16px 0 8px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 8px 0;
            font-size: 12px;
        }}
        th {{
            background-color: {t.TABLE_HEADER_BG};
            color: {t.TABLE_TEXT};
            font-weight: bold;
            text-align: left;
            padding: 8px 10px;
            border: 1px solid {t.TABLE_BORDER};
        }}
        td {{
            padding: 6px 10px;
            border: 1px solid {t.TABLE_BORDER};
            color: {t.TEXT_PRIMARY};
        }}
        tr:nth-child(even) {{
            background-color: {t.TABLE_ROW_ALT};
        }}
        tr:hover {{
            background-color: {t.BG_HOVER};
        }}
        .warning-box {{
            background-color: {t.BG_CARD};
            border: 1px solid {t.WARNING};
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            color: {t.TEXT_PRIMARY};
        }}
        .note-box {{
            background-color: {t.BG_CARD};
            border: 1px solid {t.INFO};
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            color: {t.TEXT_PRIMARY};
        }}
        .syntax-block {{
            background-color: {t.BG_SIDEBAR};
            border: 1px solid {t.BORDER};
            border-radius: 4px;
            padding: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            color: {t.TEXT_PRIMARY};
        }}
        .footnote {{
            font-size: 11px;
            color: {t.TEXT_SECONDARY};
            margin-top: 4px;
        }}
        .timestamp {{
            font-size: 11px;
            color: {t.TEXT_MUTED};
            font-style: italic;
        }}
        .measure-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            color: white;
            margin-left: 4px;
        }}
        .measure-scale {{ background-color: {t.MEASURE_SCALE}; }}
        .measure-nominal {{ background-color: {t.MEASURE_NOMINAL}; }}
        .measure-ordinal {{ background-color: {t.MEASURE_ORDINAL}; }}
        .measure-binary {{ background-color: {t.MEASURE_BINARY}; }}
        .measure-datetime {{ background-color: {t.MEASURE_DATETIME}; }}
        .measure-text {{ background-color: {t.MEASURE_TEXT}; }}
    </style>
"""


def get_measure_badge_html(measure: str, mode: ThemeMode | None = None) -> str:
    """Get HTML badge for a measure type."""
    measure_class = measure.lower().replace("_", "-")
    return f'<span class="measure-badge measure-{measure_class}">{measure[:3].upper()}</span>'
