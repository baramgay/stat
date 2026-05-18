"""Icon and emoji helpers for StatWorkbench UI.

Unicode emoji와 문자 기반 아이콘을 사용하여 PySide6 애플리케이션에
시각적 표시를 추가합니다. 외부 이미지 파일 의존성 없이 작동합니다.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon, QPixmap, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QSize


# ── Emoji Icons ────────────────────────────────────────────────────────────

class Icons:
    """Unicode emoji 기반 아이콘 모음."""

    # File operations
    NEW = "📄"
    OPEN = "📂"
    SAVE = "💾"
    IMPORT = "📥"
    EXPORT = "📤"
    PRINT = "🖨️"

    # Edit operations
    UNDO = "↩️"
    REDO = "↪️"
    CUT = "✂️"
    COPY = "📋"
    PASTE = "📌"
    DELETE = "🗑️"

    # Data operations
    ADD = "➕"
    REMOVE = "➖"
    EDIT = "✏️"
    FILTER = "🔍"
    SORT = "📊"
    COMPUTE = "🧮"

    # Analysis
    ANALYZE = "📈"
    DESCRIPTIVE = "📊"
    FREQUENCY = "📋"
    CROSSTAB = "🔀"
    TTEST = "🔬"
    ANOVA = "📉"
    CORRELATION = "🔗"
    REGRESSION = "📐"
    NONPARAMETRIC = "🎯"

    # View
    DATA = "📋"
    VARIABLE = "📑"
    OUTPUT = "📤"
    CHART = "📊"
    TABLE = "📋"

    # Navigation
    HOME = "🏠"
    BACK = "◀️"
    FORWARD = "▶️"
    UP = "⬆️"
    DOWN = "⬇️"
    REFRESH = "🔄"

    # Status
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"
    QUESTION = "❓"
    HELP = "❔"

    # Measure types
    MEASURE_SCALE = "🔢"
    MEASURE_NOMINAL = "🏷️"
    MEASURE_ORDINAL = "📶"
    MEASURE_BINARY = "🔘"
    MEASURE_DATETIME = "📅"
    MEASURE_TEXT = "📝"

    # Misc
    SETTINGS = "⚙️"
    ABOUT = "ℹ️"
    EXIT = "🚪"
    SEARCH = "🔎"
    ZOOM_IN = "🔍+"
    ZOOM_OUT = "🔍-"
    FULLSCREEN = "⛶"


# ── Measure Type Icons ─────────────────────────────────────────────────────

MEASURE_ICONS = {
    "scale": Icons.MEASURE_SCALE,
    "nominal": Icons.MEASURE_NOMINAL,
    "ordinal": Icons.MEASURE_ORDINAL,
    "binary": Icons.MEASURE_BINARY,
    "date_time": Icons.MEASURE_DATETIME,
    "text": Icons.MEASURE_TEXT,
}


def get_measure_icon(measure: str) -> str:
    """Get the icon for a measure type."""
    return MEASURE_ICONS.get(measure.lower(), "❓")


# ── QIcon Factory ──────────────────────────────────────────────────────────

def create_text_icon(text: str, size: int = 16, color: str = "#1a5276") -> QIcon:
    """Create a QIcon from text (emoji or character).

    Parameters
    ----------
    text : str
        The text to render as an icon.
    size : int
        Icon size in pixels.
    color : str
        Text color (for non-emoji text).

    Returns
    -------
    QIcon
        The created icon.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Use a font that supports emoji
    font = QFont("Segoe UI Emoji", size - 2)
    painter.setFont(font)
    painter.setPen(QColor(color))

    # Center the text
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return QIcon(pixmap)


def create_circle_icon(color: str, size: int = 12) -> QIcon:
    """Create a circular color indicator icon.

    Parameters
    ----------
    color : str
        Hex color code.
    size : int
        Icon size in pixels.

    Returns
    -------
    QIcon
        The created icon.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()

    return QIcon(pixmap)


# ── Toolbar Action Icons ───────────────────────────────────────────────────

def get_toolbar_icons() -> dict[str, QIcon]:
    """Get a dictionary of common toolbar icons.

    Returns
    -------
    dict[str, QIcon]
        Mapping of action names to QIcon instances.
    """
    return {
        "new": create_text_icon(Icons.NEW),
        "open": create_text_icon(Icons.OPEN),
        "save": create_text_icon(Icons.SAVE),
        "import": create_text_icon(Icons.IMPORT),
        "export": create_text_icon(Icons.EXPORT),
        "undo": create_text_icon(Icons.UNDO),
        "redo": create_text_icon(Icons.REDO),
        "cut": create_text_icon(Icons.CUT),
        "copy": create_text_icon(Icons.COPY),
        "paste": create_text_icon(Icons.PASTE),
        "delete": create_text_icon(Icons.DELETE),
        "analyze": create_text_icon(Icons.ANALYZE),
        "settings": create_text_icon(Icons.SETTINGS),
        "help": create_text_icon(Icons.HELP),
    }
