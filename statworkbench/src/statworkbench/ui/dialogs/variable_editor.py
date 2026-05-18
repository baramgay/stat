"""Variable property editor dialog for StatWorkbench."""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, Role, StorageType

logger = logging.getLogger(__name__)

# ── String constants ──────────────────────────────────────────────────────

STR_TITLE_NEW = "New Variable / 새 변수"
STR_TITLE_EDIT = "Edit Variable / 변수 편집"
STR_NAME = "Name:"
STR_LABEL = "Label:"
STR_TYPE = "Type:"
STR_MEASURE = "Measure:"
STR_ROLE = "Role:"
STR_VALUES = "Value Labels:"
STR_MISSING = "Missing Values:"
STR_WIDTH = "Width:"
STR_DECIMALS = "Decimals:"
STR_UNIT = "Unit:"
STR_RANGE_MIN = "Min:"
STR_RANGE_MAX = "Max:"
STR_DESCRIPTION = "Description:"
STR_ADD = "Add"
STR_REMOVE = "Remove"
STR_EDIT = "Edit..."
STR_OK = "OK"
STR_CANCEL = "Cancel"


class ValueLabelsDialog(QDialog):
    """값 라벨 편집 서브 다이얼로그."""

    def __init__(
        self,
        value_labels: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Value Labels / 값 라벨 편집")
        self.setMinimumSize(400, 350)
        self._result: dict = dict(value_labels)
        self._setup_ui()
        self._load_data()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Value", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton(STR_ADD)
        self.remove_btn = QPushButton(STR_REMOVE)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Dialog buttons
        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self._on_ok)
        dialog_btns.rejected.connect(self.reject)
        layout.addWidget(dialog_btns)

        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_selected)

    def _load_data(self) -> None:
        for value, label in self._result.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(value)))
            self.table.setItem(row, 1, QTableWidgetItem(str(label)))

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _on_ok(self) -> None:
        self._result = {}
        for row in range(self.table.rowCount()):
            val_item = self.table.item(row, 0)
            label_item = self.table.item(row, 1)
            if val_item and label_item:
                val_text = val_item.text().strip()
                label_text = label_item.text().strip()
                if val_text and label_text:
                    # Try int conversion for numeric keys
                    try:
                        key = int(val_text)
                    except ValueError:
                        try:
                            key = float(val_text)
                        except ValueError:
                            key = val_text
                    self._result[key] = label_text
        self.accept()

    def get_value_labels(self) -> dict:
        """편집된 값 라벨 딕셔너리를 반환합니다."""
        return dict(self._result)


class MissingValuesDialog(QDialog):
    """결측값 규칙 편집 서브 다이얼로그."""

    def __init__(
        self,
        missing_values: list[Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Missing Values / 결측값 규칙 편집")
        self.setMinimumSize(350, 300)
        self._result: list[Any] = list(missing_values)
        self._setup_ui()
        self._load_data()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Table
        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["Missing Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton(STR_ADD)
        self.remove_btn = QPushButton(STR_REMOVE)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self._on_ok)
        dialog_btns.rejected.connect(self.reject)
        layout.addWidget(dialog_btns)

        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_selected)

    def _load_data(self) -> None:
        for val in self._result:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(val)))

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _on_ok(self) -> None:
        self._result = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                text = item.text().strip()
                if text:
                    # Try numeric conversion
                    try:
                        val = int(text)
                    except ValueError:
                        try:
                            val = float(text)
                        except ValueError:
                            val = text
                    self._result.append(val)
        self.accept()

    def get_missing_values(self) -> list[Any]:
        """편집된 결측값 규칙을 반환합니다."""
        return list(self._result)


class VariableEditorDialog(QDialog):
    """변수 속성 편집 대화상자.

    Name, Label, Type, Measure, Role, Values, Missing,
    Width, Decimals, Unit, Range 등을 편집합니다.
    """

    def __init__(
        self,
        variable: Optional[VariableMeta] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._is_new = variable is None
        self._variable = variable or VariableMeta(name="VAR")
        self._value_labels: dict = dict(self._variable.value_labels)
        self._missing_values: list[Any] = list(self._variable.missing_values)

        self.setWindowTitle(STR_TITLE_NEW if self._is_new else STR_TITLE_EDIT)
        self.setMinimumSize(500, 450)
        self._setup_ui()
        self._load_values()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Main form grid
        form_grid = QGridLayout()
        form_grid.setColumnStretch(1, 1)
        form_grid.setColumnStretch(3, 1)

        row_idx = 0

        # Name
        form_grid.addWidget(QLabel(STR_NAME), row_idx, 0)
        self.name_edit = QLineEdit()
        form_grid.addWidget(self.name_edit, row_idx, 1)

        # Label
        form_grid.addWidget(QLabel(STR_LABEL), row_idx, 2)
        self.label_edit = QLineEdit()
        form_grid.addWidget(self.label_edit, row_idx, 3)
        row_idx += 1

        # Type
        form_grid.addWidget(QLabel(STR_TYPE), row_idx, 0)
        self.type_combo = QComboBox()
        for st in StorageType:
            self.type_combo.addItem(st.value, st)
        form_grid.addWidget(self.type_combo, row_idx, 1)

        # Measure
        form_grid.addWidget(QLabel(STR_MEASURE), row_idx, 2)
        self.measure_combo = QComboBox()
        for mt in MeasureType:
            self.measure_combo.addItem(mt.value, mt)
        form_grid.addWidget(self.measure_combo, row_idx, 3)
        row_idx += 1

        # Role
        form_grid.addWidget(QLabel(STR_ROLE), row_idx, 0)
        self.role_combo = QComboBox()
        for r in Role:
            self.role_combo.addItem(r.value, r)
        form_grid.addWidget(self.role_combo, row_idx, 1)

        # Width
        form_grid.addWidget(QLabel(STR_WIDTH), row_idx, 2)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 999)
        form_grid.addWidget(self.width_spin, row_idx, 3)
        row_idx += 1

        # Decimals
        form_grid.addWidget(QLabel(STR_DECIMALS), row_idx, 0)
        self.decimals_spin = QSpinBox()
        self.decimals_spin.setRange(0, 15)
        form_grid.addWidget(self.decimals_spin, row_idx, 1)

        # Unit
        form_grid.addWidget(QLabel(STR_UNIT), row_idx, 2)
        self.unit_edit = QLineEdit()
        form_grid.addWidget(self.unit_edit, row_idx, 3)
        row_idx += 1

        # Range
        form_grid.addWidget(QLabel("Range:"), row_idx, 0)
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel(STR_RANGE_MIN))
        self.range_min_edit = QLineEdit()
        self.range_min_edit.setPlaceholderText("None")
        range_layout.addWidget(self.range_min_edit)
        range_layout.addWidget(QLabel(STR_RANGE_MAX))
        self.range_max_edit = QLineEdit()
        self.range_max_edit.setPlaceholderText("None")
        range_layout.addWidget(self.range_max_edit)
        range_layout.addStretch()
        form_grid.addLayout(range_layout, row_idx, 1, 1, 3)
        row_idx += 1

        layout.addLayout(form_grid)

        # Values group
        values_group = QGroupBox(STR_VALUES)
        values_layout = QHBoxLayout(values_group)
        self.values_info = QLineEdit()
        self.values_info.setReadOnly(True)
        self.values_info.setPlaceholderText("No value labels defined")
        values_layout.addWidget(self.values_info)
        self.values_edit_btn = QPushButton(STR_EDIT)
        values_layout.addWidget(self.values_edit_btn)
        layout.addWidget(values_group)

        # Missing group
        missing_group = QGroupBox(STR_MISSING)
        missing_layout = QHBoxLayout(missing_group)
        self.missing_info = QLineEdit()
        self.missing_info.setReadOnly(True)
        self.missing_info.setPlaceholderText("No missing value rules defined")
        missing_layout.addWidget(self.missing_info)
        self.missing_edit_btn = QPushButton(STR_EDIT)
        missing_layout.addWidget(self.missing_edit_btn)
        layout.addWidget(missing_group)

        # Description
        desc_group = QGroupBox(STR_DESCRIPTION)
        desc_layout = QVBoxLayout(desc_group)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        desc_layout.addWidget(self.description_edit)
        layout.addWidget(desc_group)

        # Dialog buttons
        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self._on_ok)
        dialog_btns.rejected.connect(self.reject)
        layout.addWidget(dialog_btns)

        # Connect edit buttons
        self.values_edit_btn.clicked.connect(self._edit_value_labels)
        self.missing_edit_btn.clicked.connect(self._edit_missing_values)

    def _load_values(self) -> None:
        """Load variable values into the UI."""
        var = self._variable
        self.name_edit.setText(var.name)
        self.label_edit.setText(var.label)

        # Set combo selections
        idx = self.type_combo.findData(var.storage_type, Qt.ItemDataRole.UserRole)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        else:
            self.type_combo.setCurrentText(var.storage_type.value)

        idx = self.measure_combo.findData(var.measure, Qt.ItemDataRole.UserRole)
        if idx >= 0:
            self.measure_combo.setCurrentIndex(idx)
        else:
            self.measure_combo.setCurrentText(var.measure.value)

        idx = self.role_combo.findData(var.role, Qt.ItemDataRole.UserRole)
        if idx >= 0:
            self.role_combo.setCurrentIndex(idx)
        else:
            self.role_combo.setCurrentText(var.role.value)

        self.width_spin.setValue(var.width)
        self.decimals_spin.setValue(var.decimals)
        self.unit_edit.setText(var.unit)

        if var.allowed_min is not None:
            self.range_min_edit.setText(str(var.allowed_min))
        if var.allowed_max is not None:
            self.range_max_edit.setText(str(var.allowed_max))

        self.description_edit.setPlainText(var.description)
        self._update_value_labels_display()
        self._update_missing_values_display()

    def _update_value_labels_display(self) -> None:
        if self._value_labels:
            self.values_info.setText(f"{len(self._value_labels)} label(s) defined")
        else:
            self.values_info.clear()

    def _update_missing_values_display(self) -> None:
        if self._missing_values:
            self.missing_info.setText(f"{len(self._missing_values)} rule(s) defined")
        else:
            self.missing_info.clear()

    def _edit_value_labels(self) -> None:
        dlg = ValueLabelsDialog(self._value_labels, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._value_labels = dlg.get_value_labels()
            self._update_value_labels_display()

    def _edit_missing_values(self) -> None:
        dlg = MissingValuesDialog(self._missing_values, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._missing_values = dlg.get_missing_values()
            self._update_missing_values_display()

    def _on_ok(self) -> None:
        """Validate and accept."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Variable name cannot be empty.")
            return

        self.accept()

    def get_variable_meta(self) -> VariableMeta:
        """편집된 VariableMeta를 반환합니다."""
        var = self._variable
        var.name = self.name_edit.text().strip()
        var.label = self.label_edit.text().strip()

        var.storage_type = self.type_combo.currentData(Qt.ItemDataRole.UserRole) or StorageType.STRING
        var.measure = self.measure_combo.currentData(Qt.ItemDataRole.UserRole) or MeasureType.NOMINAL
        var.role = self.role_combo.currentData(Qt.ItemDataRole.UserRole) or Role.NONE

        var.width = self.width_spin.value()
        var.decimals = self.decimals_spin.value()
        var.unit = self.unit_edit.text().strip()

        # Parse range
        min_text = self.range_min_edit.text().strip()
        max_text = self.range_max_edit.text().strip()
        var.allowed_min = float(min_text) if min_text else None
        var.allowed_max = float(max_text) if max_text else None

        var.description = self.description_edit.toPlainText().strip()
        var.value_labels = dict(self._value_labels)
        var.missing_values = list(self._missing_values)

        return var
