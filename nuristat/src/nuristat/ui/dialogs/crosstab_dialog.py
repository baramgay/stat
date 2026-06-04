"""Crosstab (Contingency Table) analysis dialog for NuriStat."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.ui.dialogs.analysis_dialog_base import AnalysisDialogBase

# ── String constants ──────────────────────────────────────────────────────

STR_DIALOG_TITLE = "Crosstabs / 교차분석"
STR_ROW_LABEL = "Row(s): / 행 변수"
STR_COL_LABEL = "Column(s): / 열 변수"
STR_LAYER_LABEL = "Layer: / 층 변수 (선택)"
STR_OPTIONS = "Statistics"
STR_OPT_CHISQ = "Chi-square"
STR_OPT_LIKELIHOOD = "Likelihood ratio"
STR_OPT_FISHER = "Fisher's exact test"
STR_OPT_CRAMERS = "Cramer's V"
STR_OPT_CONTINUITY = "Continuity correction"


class CrosstabDialog(AnalysisDialogBase):
    """교차분석 다이얼로그.

    - Row: Nominal/Ordinal/Binary 변수 (필수)
    - Column: Nominal/Ordinal/Binary 변수 (필수)
    - Layer: Nominal/Ordinal/Binary 변수 (선택)
    - Options: 카이제곱, likelihood ratio, Fisher exact, Cramer's V
    """

    def __init__(
        self,
        dataset: Dataset,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(dataset, STR_DIALOG_TITLE, parent)
        self._filter_available_by_measure([
            MeasureType.NOMINAL,
            MeasureType.ORDINAL,
            MeasureType.BINARY,
        ])

    def _build_selection_area(self) -> QWidget:
        """Build Row, Column, Layer selection area."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Row variable
        row_group = QGroupBox(STR_ROW_LABEL)
        row_layout = QHBoxLayout(row_group)
        self.row_list = self._create_selection_list("row", widget)
        self.row_list.setMaximumHeight(60)
        row_layout.addWidget(self.row_list)
        row_btns = self._create_single_move_buttons("row", self.row_list)
        row_layout.addLayout(row_btns)
        layout.addWidget(row_group)

        # Column variable
        col_group = QGroupBox(STR_COL_LABEL)
        col_layout = QHBoxLayout(col_group)
        self.col_list = self._create_selection_list("column", widget)
        self.col_list.setMaximumHeight(60)
        col_layout.addWidget(self.col_list)
        col_btns = self._create_single_move_buttons("column", self.col_list)
        col_layout.addLayout(col_btns)
        layout.addWidget(col_group)

        # Layer variable
        layer_group = QGroupBox(STR_LAYER_LABEL)
        layer_layout = QHBoxLayout(layer_group)
        self.layer_list = self._create_selection_list("layer", widget)
        self.layer_list.setMaximumHeight(60)
        layer_layout.addWidget(self.layer_list)
        layer_btns = self._create_single_move_buttons("layer", self.layer_list)
        layer_layout.addLayout(layer_btns)
        layout.addWidget(layer_group)

        layout.addStretch()
        return widget

    def _create_single_move_buttons(
        self,
        _label: str,
        target: QListWidget,
    ) -> QVBoxLayout:
        """Create single-item move buttons for a selection list."""
        from PySide6.QtWidgets import QPushButton
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()

        btn_add = QPushButton(">")
        btn_add.setMaximumWidth(40)
        btn_add.clicked.connect(lambda: self._move_selected_items(self._available_list, target))
        btn_layout.addWidget(btn_add)

        btn_remove = QPushButton("<")
        btn_remove.setMaximumWidth(40)
        btn_remove.clicked.connect(lambda: self._move_selected_items(target, self._available_list))
        btn_layout.addWidget(btn_remove)

        btn_layout.addStretch()
        return btn_layout

    def _build_options_area(self) -> QWidget:
        """Build statistics options area."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox(STR_OPTIONS)
        stats_layout = QVBoxLayout(stats_group)

        stats_layout.addWidget(self._add_option_checkbox("chisq", STR_OPT_CHISQ, True))
        stats_layout.addWidget(self._add_option_checkbox("likelihood_ratio", STR_OPT_LIKELIHOOD, True))
        stats_layout.addWidget(self._add_option_checkbox("fisher", STR_OPT_FISHER, True))
        stats_layout.addWidget(self._add_option_checkbox("cramers_v", STR_OPT_CRAMERS, True))
        stats_layout.addWidget(self._add_option_checkbox("continuity_correction", STR_OPT_CONTINUITY, False))

        layout.addWidget(stats_group)
        return widget

    def validate(self) -> list[str]:
        """Validate dialog inputs."""
        errors: list[str] = []

        row_names = self._get_selected_variable_names(self.row_list)
        col_names = self._get_selected_variable_names(self.col_list)
        layer_names = self._get_selected_variable_names(self.layer_list)

        if not row_names:
            errors.append("A row variable must be selected.")
        elif len(row_names) > 1:
            errors.append("Only one row variable can be selected.")

        if not col_names:
            errors.append("A column variable must be selected.")
        elif len(col_names) > 1:
            errors.append("Only one column variable can be selected.")

        if len(layer_names) > 1:
            errors.append("Only one layer variable can be selected.")

        # Check measure types
        for names, label in [
            (row_names, "Row"),
            (col_names, "Column"),
            (layer_names, "Layer"),
        ]:
            for name in names:
                try:
                    var = self.dataset.get_variable(name)
                    if var.measure not in (
                        MeasureType.NOMINAL,
                        MeasureType.ORDINAL,
                        MeasureType.BINARY,
                    ):
                        errors.append(
                            f"{label} variable '{name}' should be nominal, ordinal, or binary."
                        )
                except Exception:
                    pass

        # Fisher's exact requires 2x2
        if self._get_option_value("fisher"):
            if row_names and col_names:
                try:
                    row_uniq = self.get_variable_unique_count(row_names[0])
                    col_uniq = self.get_variable_unique_count(col_names[0])
                    if row_uniq > 2 or col_uniq > 2:
                        errors.append(
                            "Fisher's exact test requires 2x2 tables (variables with 2 categories each)."
                        )
                except Exception:
                    pass

        return errors

    def get_analysis_spec(self) -> dict[str, Any]:
        """Return crosstab analysis specification."""
        row_names = self._get_selected_variable_names(self.row_list)
        col_names = self._get_selected_variable_names(self.col_list)
        layer_names = self._get_selected_variable_names(self.layer_list)

        return {
            "analysis_id": "crosstab",
            "title": "Crosstabs",
            "variables": {
                "row": row_names[0] if row_names else None,
                "column": col_names[0] if col_names else None,
                "layer": layer_names[0] if layer_names else None,
            },
            "options": {
                "chi_square": self._get_option_value("chisq"),
                "likelihood_ratio": self._get_option_value("likelihood_ratio"),
                "fisher_exact": self._get_option_value("fisher"),
                "cramers_v": self._get_option_value("cramers_v"),
                "continuity_correction": self._get_option_value("continuity_correction"),
            },
            "missing_policy": "listwise",
        }
