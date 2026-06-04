"""CSV/Excel data import dialog for NuriStat."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nuristat.ui.models.dataframe_table_model import DataFrameTableModel

logger = logging.getLogger(__name__)

# ── String constants (i18n-ready) ─────────────────────────────────────────

STR_TITLE = "Import Data"
STR_TITLE_KO = "데이터 가져오기"
STR_FILE_SELECT = "Select File..."
STR_FILE_SELECT_KO = "파일 선택..."
STR_ENCODING = "Encoding:"
STR_ENCODING_KO = "인코딩:"
STR_DELIMITER = "Delimiter:"
STR_DELIMITER_KO = "구분자:"
STR_HEADER = "First row as header"
STR_HEADER_KO = "첫 행을 헤더로 사용"
STR_PREVIEW = "Preview (first 20 rows)"
STR_PREVIEW_KO = "미리보기 (처음 20행)"
STR_OK = "OK"
STR_CANCEL = "Cancel"
STR_HELP = "Help"

# Encoding options
ENCODINGS: list[tuple[str, str]] = [
    ("Auto", "Auto-detect"),
    ("utf-8", "UTF-8"),
    ("cp949", "CP949"),
    ("euc-kr", "EUC-KR"),
    ("latin-1", "Latin-1"),
    ("utf-8-sig", "UTF-8-SIG"),
]

# Delimiter options
DELIMITERS: list[tuple[str, str]] = [
    ("auto", "Auto-detect"),
    (",", "Comma (, )"),
    ("\t", "Tab"),
    (";", "Semicolon (; )"),
    ("|", "Pipe (|)"),
    (" ", "Space"),
]


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet if available, else fallback."""
    try:
        import chardet

        with open(file_path, "rb") as f:
            raw = f.read(100000)
            result = chardet.detect(raw)
            if result and result["encoding"]:
                enc = result["encoding"].lower()
                if enc in ("ascii",):
                    return "utf-8"
                return enc
    except ImportError:
        pass
    # Fallback: try common encodings
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            with open(file_path, encoding=enc) as f:
                f.read(1024)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


def _detect_delimiter(file_path: str, encoding: str) -> str:
    """Detect delimiter from first few lines."""
    try:
        with open(file_path, encoding=encoding) as f:
            sample = f.read(8192)
    except Exception:
        return ","

    candidates = [",", "\t", ";", "|"]
    counts: dict[str, int] = {}
    for sep in candidates:
        line_counts = [line.count(sep) for line in sample.splitlines()[:5]]
        if line_counts:
            counts[sep] = max(line_counts)
    if counts:
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            return best
    return ","


class ImportDialog(QDialog):
    """CSV/Excel 파일을 임포트하기 위한 대화상자.

    파일 경로 선택, 인코딩/구분자 설정, 헤더 사용 여부,
    그리고 처음 20행의 미리보기를 제공합니다.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{STR_TITLE} / {STR_TITLE_KO}")
        self.setMinimumSize(800, 600)

        self._file_path: str = ""
        self._preview_df: pd.DataFrame | None = None
        self._preview_model: DataFrameTableModel | None = None

        self._setup_ui()
        self._connect_signals()

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # File selection row
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a CSV, TXT, or Excel file...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit, stretch=1)

        self.browse_btn = QPushButton(STR_FILE_SELECT)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # Options group
        options_group = QGroupBox("Import Options / 가져오기 옵션")
        options_layout = QHBoxLayout(options_group)

        # Encoding
        options_layout.addWidget(QLabel(STR_ENCODING))
        self.encoding_combo = QComboBox()
        for code, label in ENCODINGS:
            self.encoding_combo.addItem(label, code)
        self.encoding_combo.setCurrentIndex(1)  # UTF-8 default
        options_layout.addWidget(self.encoding_combo)

        options_layout.addSpacing(20)

        # Delimiter
        options_layout.addWidget(QLabel(STR_DELIMITER))
        self.delimiter_combo = QComboBox()
        for code, label in DELIMITERS:
            self.delimiter_combo.addItem(label, code)
        self.delimiter_combo.setCurrentIndex(1)  # Comma default
        options_layout.addWidget(self.delimiter_combo)

        options_layout.addSpacing(20)

        # Header checkbox
        self.header_check = QCheckBox(STR_HEADER)
        self.header_check.setChecked(True)
        options_layout.addWidget(self.header_check)

        options_layout.addStretch()
        layout.addWidget(options_group)

        # Preview table
        preview_group = QGroupBox(STR_PREVIEW)
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group, stretch=1)

        # Info label
        self.info_label = QLabel("No file selected.")
        layout.addWidget(self.info_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.help_btn = QPushButton(STR_HELP)
        btn_layout.addWidget(self.help_btn)

        btn_layout.addSpacing(20)

        self.ok_btn = QPushButton(STR_OK)
        self.ok_btn.setDefault(True)
        self.ok_btn.setEnabled(False)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton(STR_CANCEL)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        self.browse_btn.clicked.connect(self._browse_file)
        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn.clicked.connect(self.reject)
        self.help_btn.clicked.connect(self._show_help)

        # Auto-refresh preview when options change
        self.encoding_combo.currentIndexChanged.connect(self._refresh_preview)
        self.delimiter_combo.currentIndexChanged.connect(self._refresh_preview)
        self.header_check.stateChanged.connect(self._refresh_preview)

    # ── File operations ──────────────────────────────────────────────────

    def _browse_file(self) -> None:
        """Open file dialog for CSV/Excel selection."""
        filters = (
            "All Supported (*.csv *.txt *.tsv *.xlsx *.xls);;"
            "CSV Files (*.csv *.txt *.tsv);;"
            "Excel Files (*.xlsx *.xls);;"
            "All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", filters
        )
        if path:
            self._file_path = path
            self.file_path_edit.setText(path)
            self._detect_and_set_encoding(path)
            self._refresh_preview()

    def _detect_and_set_encoding(self, path: str) -> None:
        """Auto-detect encoding and update combo."""
        detected = _detect_encoding(path)
        for i in range(self.encoding_combo.count()):
            if self.encoding_combo.itemData(i) == detected:
                self.encoding_combo.setCurrentIndex(i)
                return
        # If not in list, add it
        self.encoding_combo.addItem(detected.upper(), detected)
        self.encoding_combo.setCurrentIndex(self.encoding_combo.count() - 1)

    def _refresh_preview(self) -> None:
        """Reload preview with current settings."""
        if not self._file_path or not os.path.isfile(self._file_path):
            return

        ext = Path(self._file_path).suffix.lower()

        try:
            if ext in (".xlsx", ".xls"):
                df = self._load_excel_preview()
            else:
                df = self._load_csv_preview()

            if df is None or df.empty:
                self.info_label.setText("No data could be loaded from the file.")
                self.ok_btn.setEnabled(False)
                return

            # Limit to 20 rows for preview
            preview = df.head(20)
            self._preview_df = df

            # Update table model
            if self._preview_model is None:
                self._preview_model = DataFrameTableModel(preview)
                self.preview_table.setModel(self._preview_model)
            else:
                self._preview_model.set_dataframe(preview)

            self.info_label.setText(
                f"Rows: {len(df)} | Columns: {len(df.columns)} | "
                f"Preview: first {min(20, len(df))} rows"
            )
            self.ok_btn.setEnabled(True)

        except Exception as exc:
            logger.error("Preview load error: %s", exc)
            self.info_label.setText(f"Error loading file: {exc}")
            self.ok_btn.setEnabled(False)

    def _load_csv_preview(self) -> pd.DataFrame | None:
        """Load CSV/TXT preview with current settings."""
        encoding = self.encoding_combo.currentData()
        delimiter = self.delimiter_combo.currentData()
        header = 0 if self.header_check.isChecked() else None

        # Auto-detect encoding if needed
        if encoding == "Auto":
            encoding = _detect_encoding(self._file_path)

        # Auto-detect delimiter if needed
        if delimiter == "auto":
            delimiter = _detect_delimiter(self._file_path, encoding)

        kwargs: dict[str, Any] = {
            "encoding": encoding,
            "header": header,
            "nrows": 100,  # Load more than preview for type inference
            "low_memory": False,
        }
        if delimiter != "auto":
            kwargs["sep"] = delimiter

        return pd.read_csv(self._file_path, **kwargs)

    def _load_excel_preview(self) -> pd.DataFrame | None:
        """Load Excel preview."""
        header = 0 if self.header_check.isChecked() else None
        return pd.read_excel(self._file_path, header=header, nrows=100)

    def _on_ok(self) -> None:
        """Validate and accept dialog."""
        if not self._file_path:
            QMessageBox.warning(self, "No File", "Please select a file to import.")
            return
        self.accept()

    def _show_help(self) -> None:
        """Show import dialog help."""
        QMessageBox.information(
            self,
            "Import Help",
            "<b>Import Data Help</b><br><br>"
            "1. Select a CSV, TXT, TSV, or Excel file.<br>"
            "2. Choose the correct character encoding (UTF-8, CP949, etc.)<br>"
            "3. Choose the delimiter for text files (Comma, Tab, etc.)<br>"
            "4. Check 'First row as header' if the first row contains column names.<br>"
            "5. Review the preview and click OK to import.",
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_import_params(self) -> dict[str, Any]:
        """임포트 파라미터를 딕셔너리로 반환합니다.

        Returns:
            {
                'file_path': str,
                'encoding': str,
                'delimiter': str,
                'header': bool,
                'file_type': str,  # 'csv' or 'excel'
            }
        """
        encoding = self.encoding_combo.currentData()
        delimiter = self.delimiter_combo.currentData()

        if encoding == "Auto":
            encoding = _detect_encoding(self._file_path) if self._file_path else "utf-8"
        if delimiter == "auto":
            delimiter = (
                _detect_delimiter(self._file_path, encoding)
                if self._file_path
                else ","
            )

        ext = Path(self._file_path).suffix.lower() if self._file_path else ""
        file_type = "excel" if ext in (".xlsx", ".xls") else "csv"

        return {
            "file_path": self._file_path,
            "encoding": encoding,
            "delimiter": delimiter,
            "header": self.header_check.isChecked(),
            "file_type": file_type,
        }

    def get_file_path(self) -> str:
        """선택된 파일 경로를 반환합니다."""
        return self._file_path

    def get_preview_dataframe(self) -> pd.DataFrame | None:
        """미리보기 DataFrame을 반환합니다."""
        return self._preview_df
