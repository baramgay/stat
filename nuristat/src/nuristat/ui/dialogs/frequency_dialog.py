"""Frequency analysis dialog for NuriStat."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.ui.dialogs.analysis_dialog_base import AnalysisDialogBase

# ── String constants ──────────────────────────────────────────────────────

STR_DIALOG_TITLE = "Frequencies / 빈도분석"
STR_VARIABLES_LABEL = "Variable(s): / 변수"
STR_OPTIONS_DISPLAY = "Display options"
STR_OPTIONS_SORT = "Sort by:"
STR_OPTIONS_INCLUDE_MISSING = "Include missing in table"
STR_OPTIONS_CUMULATIVE = "Show cumulative percent"
STR_OPTIONS_VALUE_LABELS = "Show value labels"


class FrequencyDialog(AnalysisDialogBase):
    """빈도분석 다이얼로그.

    - Variable(s): Nominal/Ordinal/Binary 변수 (필수, 1개 이상)
    - Options: 누적 퍼센트, 결측 포함, 값 라벨 표시, 정렬 방식
    """

    def __init__(
        self,
        dataset: Dataset,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(dataset, STR_DIALOG_TITLE, parent)
        # Filter available variables: nominal, ordinal, binary
        self._filter_available_by_measure([
            MeasureType.NOMINAL,
            MeasureType.ORDINAL,
            MeasureType.BINARY,
        ])

    def _build_selection_area(self) -> QWidget:
        """Build variable selection area."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        vars_group = QGroupBox(STR_VARIABLES_LABEL)
        vars_layout = QHBoxLayout(vars_group)

        # Selected variables list
        sel_layout = QVBoxLayout()
        sel_layout.addWidget(QLabel("Selected:"))
        self.vars_list = self._create_selection_list("variables", widget)
        sel_layout.addWidget(self.vars_list)
        vars_layout.addLayout(sel_layout, stretch=1)

        # Move buttons
        btn_col = QVBoxLayout()
        btn_col.addStretch()
        from PySide6.QtWidgets import QPushButton

        btn_add = QPushButton(">>")
        btn_add.setMaximumWidth(50)
        btn_add.clicked.connect(lambda: self._move_all_items(self._available_list, self.vars_list))
        btn_col.addWidget(btn_add)

        btn_add_sel = QPushButton(">")
        btn_add_sel.setMaximumWidth(50)
        btn_add_sel.clicked.connect(lambda: self._move_selected_items(self._available_list, self.vars_list))
        btn_col.addWidget(btn_add_sel)

        btn_remove_sel = QPushButton("<")
        btn_remove_sel.setMaximumWidth(50)
        btn_remove_sel.clicked.connect(lambda: self._move_selected_items(self.vars_list, self._available_list))
        btn_col.addWidget(btn_remove_sel)

        btn_remove = QPushButton("<<")
        btn_remove.setMaximumWidth(50)
        btn_remove.clicked.connect(lambda: self._move_all_items(self.vars_list, self._available_list))
        btn_col.addWidget(btn_remove)
        btn_col.addStretch()
        vars_layout.addLayout(btn_col)

        layout.addWidget(vars_group)
        layout.addStretch()
        return widget

    def _build_options_area(self) -> QWidget:
        """Build options area."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: checkboxes
        left = QVBoxLayout()
        left.addWidget(self._add_option_checkbox("include_missing", STR_OPTIONS_INCLUDE_MISSING, False))
        left.addWidget(self._add_option_checkbox("cumulative", STR_OPTIONS_CUMULATIVE, True))
        left.addWidget(self._add_option_checkbox("value_labels", STR_OPTIONS_VALUE_LABELS, True))
        left.addStretch()
        layout.addLayout(left)

        # Right: sort option
        right = QVBoxLayout()
        sort_label, sort_combo = self._add_option_combo(
            "sort_by",
            STR_OPTIONS_SORT,
            [
                ("Value (ascending)", "value_asc"),
                ("Value (descending)", "value_desc"),
                ("Frequency (ascending)", "freq_asc"),
                ("Frequency (descending)", "freq_desc"),
            ],
            0,
        )
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(sort_combo)
        sort_layout.addStretch()
        right.addLayout(sort_layout)
        right.addStretch()
        layout.addLayout(right)

        return widget

    def validate(self) -> list[str]:
        """Validate dialog inputs."""
        errors: list[str] = []

        var_names = self._get_selected_variable_names(self.vars_list)
        if not var_names:
            errors.append("At least one variable must be selected.")

        for name in var_names:
            try:
                var = self.dataset.get_variable(name)
                if var.measure not in (MeasureType.NOMINAL, MeasureType.ORDINAL, MeasureType.BINARY):
                    errors.append(
                        f"Variable '{name}' has measure '{var.measure.value}' - "
                        "Frequencies requires nominal, ordinal, or binary variables."
                    )
                # Warn if too many unique values for a text variable
                if var.measure == MeasureType.TEXT:
                    unique = self.get_variable_unique_count(name)
                    if unique > 50:
                        errors.append(
                            f"Text variable '{name}' has {unique} unique values. "
                            "Frequency table may be very large."
                        )
            except Exception:
                pass

        return errors

    def get_analysis_spec(self) -> dict[str, Any]:
        """Return frequency analysis specification."""
        return {
            "analysis_id": "frequencies",
            "title": "Frequencies",
            "variables": {
                "dependent": self._get_selected_variable_names(self.vars_list),
            },
            "options": {
                "include_missing": self._get_option_value("include_missing"),
                "cumulative_percent": self._get_option_value("cumulative"),
                "show_value_labels": self._get_option_value("value_labels"),
                "sort_by": self._get_option_value("sort_by"),
            },
            "missing_policy": "listwise",
        }
