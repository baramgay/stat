"""Theme and styling system for NuriStat.

Supports Light (default) and Dark (OLED-optimized) themes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from enum import Enum


def _prepare_arrow_svg() -> str:
    """화살표 SVG를 ASCII 경로 임시 디렉토리에 복사하고 경로 반환.

    Qt QSS의 url()은 유니코드 경로에서 간헐적으로 실패할 수 있으므로
    ASCII 경로의 임시 파일을 사용한다.
    """
    _src = os.path.join(os.path.dirname(__file__), "..", "resources", "arrow_down_dark.svg")
    _src = os.path.normpath(_src)
    if not os.path.exists(_src):
        return ""
    _tmp_dir = tempfile.gettempdir()
    _dst = os.path.join(_tmp_dir, "swb_arrow_down.svg")
    try:
        shutil.copy2(_src, _dst)
        return _dst.replace("\\", "/")
    except Exception:
        return _src.replace("\\", "/")


_ARROW_SVG_PATH = _prepare_arrow_svg()


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
        self.TEXT_MUTED = "#55555f"  # WCAG AA ≥4.5:1 on white background
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
        self.MEASURE_TEXT = "#3a5068"

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
        self.TEXT_MUTED = "#9a9aac"  # WCAG AA ≥3:1 on dark background (#0a0a0a)
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

    _instance: ThemeManager | None = None
    _current_mode: ThemeMode = ThemeMode.LIGHT

    def __new__(cls) -> ThemeManager:
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

    _arrow_svg = _ARROW_SVG_PATH

    return f"""
    /* ── 전역 기반 ─────────────────────────────────────────────────── */
    QMainWindow {{
        background-color: {t.BG_MAIN};
    }}

    QWidget {{
        background-color: {t.BG_MAIN};
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    /* ── 메뉴바 ───────────────────────────────────────────────────── */
    QMenuBar {{
        background-color: {t.BG_CARD};
        border-bottom: 1px solid {t.BORDER};
        padding: 2px 4px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 5px 14px;
        border-radius: 4px;
        color: {t.TEXT_PRIMARY};
    }}

    QMenuBar::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QMenu {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        border-radius: 6px;
        padding: 4px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QMenu::item {{
        padding: 7px 28px 7px 18px;
        border-radius: 4px;
        color: {t.TEXT_PRIMARY};
        min-width: 160px;
    }}

    QMenu::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {t.BORDER};
        margin: 4px 10px;
    }}

    /* ── 툴바 ─────────────────────────────────────────────────────── */
    QToolBar {{
        background-color: {t.BG_CARD};
        border-bottom: 1px solid {t.BORDER};
        padding: 4px 6px;
        spacing: 3px;
    }}

    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 5px 10px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QToolButton:hover {{
        background-color: {t.BG_HOVER};
        border-color: {t.BORDER};
    }}

    QToolButton:pressed {{
        background-color: {t.BG_SELECTED};
    }}

    /* ── 상태바 ───────────────────────────────────────────────────── */
    QStatusBar {{
        background-color: {t.BG_SIDEBAR};
        border-top: 1px solid {t.BORDER};
        color: {t.TEXT_SECONDARY};
        font-size: 12px;
        padding: 2px 8px;
    }}

    /* ── 탭 ───────────────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {t.BORDER};
        background-color: {t.BG_CARD};
        border-radius: 0 4px 4px 4px;
    }}

    QTabBar::tab {{
        background-color: {t.BG_SIDEBAR};
        border: 1px solid {t.BORDER};
        border-bottom: none;
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        color: {t.TEXT_SECONDARY};
        font-size: 13px;
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

    /* ── 그룹박스 ─────────────────────────────────────────────────── */
    QGroupBox {{
        font-weight: bold;
        font-size: 13px;
        color: {t.TEXT_PRIMARY};
        border: 1.5px solid {t.BORDER};
        border-radius: 7px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: {t.BG_CARD};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: {t.PRIMARY};
        background-color: {t.BG_CARD};
        font-size: 12px;
    }}

    /* ── 버튼 ─────────────────────────────────────────────────────── */
    QPushButton {{
        background-color: {t.BG_CARD};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 7px 18px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
        min-height: 28px;
    }}

    QPushButton:hover {{
        background-color: {t.BG_HOVER};
        border-color: {t.PRIMARY_LIGHT};
        color: {t.PRIMARY};
    }}

    QPushButton:default {{
        background-color: {t.PRIMARY};
        border: none;
        color: {t.TEXT_ON_PRIMARY};
        font-weight: bold;
        border-radius: 5px;
    }}

    QPushButton:default:hover {{
        background-color: {t.PRIMARY_LIGHT};
    }}

    QPushButton:pressed {{
        background-color: {t.PRIMARY_DARK};
        color: {t.TEXT_ON_PRIMARY};
    }}

    QPushButton:disabled {{
        background-color: {t.BG_SIDEBAR};
        color: {t.TEXT_MUTED};
        border-color: {t.BORDER_LIGHT};
    }}

    /* ── 입력 필드 ────────────────────────────────────────────────── */
    QLineEdit {{
        background-color: {t.BG_INPUT};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 6px 10px;
        color: {t.TEXT_PRIMARY};
        selection-background-color: {t.BG_SELECTED};
        min-height: 28px;
    }}

    QLineEdit:hover {{
        border-color: {t.SECONDARY_DARK};
    }}

    QLineEdit:focus {{
        border-color: {t.BORDER_FOCUS};
        border-width: 2px;
    }}

    QTextEdit, QPlainTextEdit {{
        background-color: {t.BG_INPUT};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 6px;
        color: {t.TEXT_PRIMARY};
        selection-background-color: {t.BG_SELECTED};
    }}

    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {t.BORDER_FOCUS};
        border-width: 2px;
    }}

    /* ── 리스트 ───────────────────────────────────────────────────── */
    QListWidget {{
        background-color: {t.BG_CARD};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 3px;
        alternate-background-color: {t.TABLE_ROW_ALT};
        color: {t.TEXT_PRIMARY};
        outline: none;
    }}

    QListWidget::item {{
        padding: 5px 10px;
        border-radius: 3px;
        color: {t.TEXT_PRIMARY};
        min-height: 22px;
    }}

    QListWidget::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
        border-left: 3px solid {t.PRIMARY};
        font-weight: bold;
    }}

    QListWidget::item:hover:!selected {{
        background-color: {t.BG_HOVER};
    }}

    /* ── 콤보박스 ─────────────────────────────────────────────────── */
    QComboBox {{
        background-color: {t.BG_INPUT};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 6px 36px 6px 10px;
        min-width: 100px;
        min-height: 28px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QComboBox:hover {{
        border-color: {t.SECONDARY_DARK};
    }}

    QComboBox:focus {{
        border-color: {t.BORDER_FOCUS};
        border-width: 2px;
    }}

    QComboBox::drop-down {{
        subcontrol-origin: border;
        subcontrol-position: center right;
        width: 32px;
        border-left: 1.5px solid {t.BORDER};
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        background-color: {t.BG_SIDEBAR};
    }}

    QComboBox::drop-down:hover {{
        background-color: {t.BG_SELECTED};
    }}

    QComboBox::down-arrow {{
        image: url("{_arrow_svg}");
        width: 12px;
        height: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {t.BG_CARD};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        selection-background-color: {t.BG_SELECTED};
        selection-color: {t.PRIMARY};
        color: {t.TEXT_PRIMARY};
        padding: 2px;
        outline: none;
    }}

    QComboBox QAbstractItemView::item {{
        padding: 6px 10px;
        border-radius: 3px;
        min-height: 24px;
    }}

    QComboBox QAbstractItemView::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    /* ── 체크박스 ─────────────────────────────────────────────────── */
    QCheckBox {{
        spacing: 8px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QCheckBox::indicator {{
        width: 17px;
        height: 17px;
        border-radius: 4px;
        border: 1.5px solid {t.BORDER};
        background-color: {t.BG_INPUT};
    }}

    QCheckBox::indicator:hover {{
        border-color: {t.PRIMARY_LIGHT};
    }}

    QCheckBox::indicator:checked {{
        background-color: {t.PRIMARY};
        border-color: {t.PRIMARY};
    }}

    /* ── 라디오버튼 ───────────────────────────────────────────────── */
    QRadioButton {{
        spacing: 8px;
        color: {t.TEXT_PRIMARY};
        font-size: 13px;
    }}

    QRadioButton::indicator {{
        width: 17px;
        height: 17px;
        border-radius: 9px;
        border: 1.5px solid {t.BORDER};
        background-color: {t.BG_INPUT};
    }}

    QRadioButton::indicator:hover {{
        border-color: {t.PRIMARY_LIGHT};
    }}

    QRadioButton::indicator:checked {{
        background-color: {t.PRIMARY};
        border-color: {t.PRIMARY};
        image: none;
    }}

    /* ── 스핀박스 ─────────────────────────────────────────────────── */
    QSpinBox, QDoubleSpinBox {{
        background-color: {t.BG_INPUT};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        padding: 5px 8px;
        color: {t.TEXT_PRIMARY};
        min-height: 28px;
    }}

    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {t.SECONDARY_DARK};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {t.BORDER_FOCUS};
        border-width: 2px;
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        border: none;
        border-left: 1px solid {t.BORDER};
        border-bottom: 1px solid {t.BORDER};
        background-color: {t.BG_SIDEBAR};
        width: 22px;
        border-top-right-radius: 4px;
    }}

    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        border: none;
        border-left: 1px solid {t.BORDER};
        background-color: {t.BG_SIDEBAR};
        width: 22px;
        border-bottom-right-radius: 4px;
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {t.BG_SELECTED};
    }}

    /* ── 스플리터 ─────────────────────────────────────────────────── */
    QSplitter::handle {{
        background-color: {t.BORDER};
    }}

    QSplitter::handle:horizontal {{
        width: 3px;
    }}

    QSplitter::handle:vertical {{
        height: 3px;
    }}

    QSplitter::handle:hover {{
        background-color: {t.PRIMARY_LIGHT};
    }}

    /* ── 트리 위젯 ────────────────────────────────────────────────── */
    QTreeWidget {{
        background-color: {t.BG_CARD};
        border: 1.5px solid {t.BORDER};
        border-radius: 5px;
        alternate-background-color: {t.TABLE_ROW_ALT};
        color: {t.TEXT_PRIMARY};
        outline: none;
    }}

    QTreeWidget::item {{
        padding: 5px 8px;
        color: {t.TEXT_PRIMARY};
        min-height: 22px;
    }}

    QTreeWidget::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
        border-left: 3px solid {t.PRIMARY};
    }}

    QTreeWidget::item:hover:!selected {{
        background-color: {t.BG_HOVER};
    }}

    /* ── 헤더뷰 ───────────────────────────────────────────────────── */
    QHeaderView::section {{
        background-color: {t.TABLE_HEADER_BG};
        color: {t.TABLE_TEXT};
        padding: 7px 10px;
        border: none;
        border-right: 1px solid {t.BORDER};
        border-bottom: 1.5px solid {t.BORDER};
        font-weight: bold;
        font-size: 12px;
    }}

    QHeaderView::section:hover {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QHeaderView::section:first {{
        border-left: none;
    }}

    /* ── 테이블뷰 ─────────────────────────────────────────────────── */
    QTableView {{
        background-color: {t.BG_CARD};
        border: 1px solid {t.BORDER};
        gridline-color: {t.BORDER_LIGHT};
        selection-background-color: {t.BG_SELECTED};
        selection-color: {t.TEXT_PRIMARY};
        color: {t.TEXT_PRIMARY};
        alternate-background-color: {t.TABLE_ROW_ALT};
        outline: none;
    }}

    QTableView::item {{
        padding: 4px 8px;
        color: {t.TEXT_PRIMARY};
        border: none;
    }}

    QTableView::item:selected {{
        background-color: {t.BG_SELECTED};
        color: {t.PRIMARY};
    }}

    QTableView QTableCornerButton::section {{
        background-color: {t.TABLE_HEADER_BG};
        border-bottom: 1.5px solid {t.BORDER};
        border-right: 1px solid {t.BORDER};
    }}

    /* ── 스크롤바 ─────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {t.SCROLLBAR_HANDLE};
        border-radius: 5px;
        min-height: 24px;
        margin: 1px 1px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {t.SCROLLBAR_HANDLE_HOVER};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
        background: none;
        border: none;
    }}

    QScrollBar:horizontal {{
        background-color: transparent;
        height: 10px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {t.SCROLLBAR_HANDLE};
        border-radius: 5px;
        min-width: 24px;
        margin: 1px 1px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {t.SCROLLBAR_HANDLE_HOVER};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
        background: none;
        border: none;
    }}

    /* ── 레이블 / 다이얼로그 ──────────────────────────────────────── */
    QLabel {{
        color: {t.TEXT_PRIMARY};
        background-color: transparent;
    }}

    QMessageBox {{
        background-color: {t.BG_CARD};
        color: {t.TEXT_PRIMARY};
    }}

    QDialog {{
        background-color: {t.BG_MAIN};
        color: {t.TEXT_PRIMARY};
    }}

    /* ── 슬라이더 ─────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {t.BG_SIDEBAR};
        border-radius: 3px;
        border: 1px solid {t.BORDER};
    }}

    QSlider::sub-page:horizontal {{
        background-color: {t.PRIMARY_LIGHT};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 18px;
        height: 18px;
        margin: -6px 0;
        background-color: {t.PRIMARY};
        border-radius: 9px;
        border: 2px solid {t.BG_CARD};
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {t.PRIMARY_LIGHT};
    }}

    /* ── 프로그레스바 ─────────────────────────────────────────────── */
    QProgressBar {{
        border: 1px solid {t.BORDER};
        border-radius: 5px;
        text-align: center;
        color: {t.TEXT_PRIMARY};
        background-color: {t.BG_SIDEBAR};
        min-height: 16px;
    }}

    QProgressBar::chunk {{
        background-color: {t.PRIMARY};
        border-radius: 4px;
    }}

    /* ── 다이얼로그 버튼박스 ──────────────────────────────────────── */
    QDialogButtonBox QPushButton {{
        min-width: 80px;
    }}

    /* ── 툴팁 ─────────────────────────────────────────────────────── */
    QToolTip {{
        background-color: {t.BG_CARD};
        color: {t.TEXT_PRIMARY};
        border: 1px solid {t.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
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
