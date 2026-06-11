"""Manual data entry dialog for NuriStat.

Allows users to create a new dataset by manually entering data in a spreadsheet-like interface.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from nuristat.core.dataset import Dataset
from nuristat.core.variable import MeasureType, StorageType


class ManualDataDialog(QDialog):
    """Dialog for manual data entry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Data Entry")
        self.setMinimumSize(800, 600)

        self._dataset: Dataset | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Info label
        info = QLabel(
            "Enter data manually. Add columns with the 'Add Column' button, "
            "then type values directly into cells. Right-click column headers to rename."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #3a5068; font-size: 12px;")
        layout.addWidget(info)

        # Column configuration group
        col_group = QGroupBox("Column Setup")
        col_layout = QHBoxLayout(col_group)

        col_layout.addWidget(QLabel("Column Name:"))
        self.col_name_edit = QLineEdit()
        self.col_name_edit.setPlaceholderText("e.g., Age, Gender, Score")
        col_layout.addWidget(self.col_name_edit)

        col_layout.addWidget(QLabel("Type:"))
        self.col_type_combo = QComboBox()
        self.col_type_combo.addItems(["Numeric", "Text", "Date"])
        col_layout.addWidget(self.col_type_combo)

        add_col_btn = QPushButton("Add Column")
        add_col_btn.setDefault(True)
        add_col_btn.clicked.connect(self._add_column)
        col_layout.addWidget(add_col_btn)

        remove_col_btn = QPushButton("Remove Column")
        remove_col_btn.clicked.connect(self._remove_column)
        col_layout.addWidget(remove_col_btn)

        layout.addWidget(col_group)

        # Data table
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(10)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)

        # Enable editing
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)

        # Header setup
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setDefaultSectionSize(120)

        vheader = self.table.verticalHeader()
        vheader.setDefaultSectionSize(28)

        layout.addWidget(self.table)

        # Row controls
        row_layout = QHBoxLayout()

        add_rows_btn = QPushButton("Add 10 Rows")
        add_rows_btn.clicked.connect(self._add_rows)
        row_layout.addWidget(add_rows_btn)

        row_layout.addStretch()

        rows_label = QLabel("Rows:")
        row_layout.addWidget(rows_label)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 10000)
        self.rows_spin.setValue(10)
        self.rows_spin.valueChanged.connect(self._set_row_count)
        row_layout.addWidget(self.rows_spin)

        layout.addLayout(row_layout)

        # Dialog buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Add default columns
        self._add_default_columns()

    def _add_default_columns(self) -> None:
        """Add default columns for quick start."""
        self._add_column_internal("ID", "Numeric")
        self._add_column_internal("Variable1", "Numeric")
        self._add_column_internal("Variable2", "Numeric")

    def _add_column(self) -> None:
        """Add a new column from user input."""
        name = self.col_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a column name.")
            return

        dtype = self.col_type_combo.currentText()
        self._add_column_internal(name, dtype)
        self.col_name_edit.clear()
        self.col_name_edit.setFocus()

    def _add_column_internal(self, name: str, dtype: str) -> None:
        """Add a column internally."""
        col_idx = self.table.columnCount()
        self.table.insertColumn(col_idx)

        # Set header with type indicator
        type_icon = {"Numeric": "#", "Text": "T", "Date": "D"}.get(dtype, "?")
        header_item = QTableWidgetItem(f"{name} ({type_icon})")
        header_item.setData(Qt.ItemDataRole.UserRole, {"name": name, "type": dtype})
        self.table.setHorizontalHeaderItem(col_idx, header_item)

    def _remove_column(self) -> None:
        """Remove the currently selected column."""
        current_col = self.table.currentColumn()
        if current_col >= 0:
            self.table.removeColumn(current_col)
        else:
            QMessageBox.information(self, "Info", "Please select a column to remove.")

    def _add_rows(self) -> None:
        """Add 10 more rows."""
        current = self.table.rowCount()
        self.table.setRowCount(current + 10)
        self.rows_spin.setValue(current + 10)

    def _set_row_count(self, count: int) -> None:
        """Set the row count."""
        self.table.setRowCount(count)

    def _on_accept(self) -> None:
        """Validate and create dataset."""
        if self.table.columnCount() == 0:
            QMessageBox.warning(self, "Warning", "Please add at least one column.")
            return

        # Build DataFrame from table
        data: dict[str, list] = {}
        col_types: dict[str, str] = {}

        for col in range(self.table.columnCount()):
            header = self.table.horizontalHeaderItem(col)
            if header is None:
                continue

            meta = header.data(Qt.ItemDataRole.UserRole)
            if meta is None:
                continue

            col_name = meta["name"]
            col_type = meta["type"]
            col_types[col_name] = col_type

            # Collect non-empty values
            values = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                if item is not None and item.text().strip():
                    values.append(item.text().strip())
                else:
                    values.append(None)

            data[col_name] = values

        if not data:
            QMessageBox.warning(self, "Warning", "No data entered.")
            return

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert types
        for col, dtype in col_types.items():
            if dtype == "Numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "Date":
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Remove completely empty rows
        df = df.dropna(how="all")

        if len(df) == 0:
            QMessageBox.warning(self, "Warning", "No valid data entered.")
            return

        # Create Dataset
        self._dataset = Dataset(df, name="Manual Entry")

        # Set variable metadata
        for col, dtype in col_types.items():
            if col in self._dataset.variables:
                var = self._dataset.variables[col]
                if dtype == "Numeric":
                    var.storage_type = StorageType.FLOAT
                    var.measure = MeasureType.SCALE
                elif dtype == "Text":
                    var.storage_type = StorageType.STRING
                    var.measure = MeasureType.NOMINAL
                elif dtype == "Date":
                    var.storage_type = StorageType.DATETIME
                    var.measure = MeasureType.ORDINAL

        self.accept()

    def get_dataset(self) -> Dataset | None:
        """Return the created dataset."""
        return self._dataset
