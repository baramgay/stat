"""Descriptive Statistics analysis dialog for StatWorkbench."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.ui.dialogs.analysis_dialog_base import AnalysisDialogBase


# ── String constants ──────────────────────────────────────────────────────

STR_DIALOG_TITLE = "Descriptive Statistics / 기술통계"
STR_VARIABLES_LABEL = "Variable(s): / 변수"
STR_GROUPING_LABEL = "Grouping: / 그룹 변수 (선택)"
STR_OPTIONS_CI = "Confidence intervals"
STR_OPTIONS_CI_LEVEL = "CI level:"
STR_OPTIONS_MISSING = "Exclude missing listwise"
STR_OPTIONS_SKEW = "Include skewness"
STR_OPTIONS_KURT = "Include kurtosis"


class DescriptiveDialog(AnalysisDialogBase):
    """기술통계 분석 다이얼로그.

    - Variable(s): Scale 변수 1개 이상 (필수)
    - Grouping: Nominal/Binary 그룹 변수 0개 또는 1개 (선택)
    - Options: 신뢰구간, 왜도, 첨도, 결측 처리
    """

    def __init__(
        self,
        dataset: Dataset,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(dataset, STR_DIALOG_TITLE, parent)
        # Filter available variables: show only scale variables for main selection
        self._filter_available_by_measure([MeasureType.SCALE])

    def _build_selection_area(self) -> QWidget:
        """Build Variables and Grouping selection area."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Variables selection
        vars_group = QGroupBox(STR_VARIABLES_LABEL)
        vars_layout = QHBoxLayout(vars_group)

        self.vars_list = self._create_selection_list("variables", widget)
        vars_layout.addWidget(self.vars_list)

        # Move buttons between available and variables list
        vars_btn_layout = QVBoxLayout()
        vars_btn_layout.addStretch()

        move_right_btn = QLabel("  >  ")
        move_right_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        move_right_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vars_btn_layout.addWidget(move_right_btn)

        add_btn = self._create_move_btn(">", self._available_list, self.vars_list, False)
        vars_btn_layout.addWidget(add_btn)

        remove_btn = self._create_move_btn("<", self.vars_list, self._available_list, False)
        vars_btn_layout.addWidget(remove_btn)

        move_left_btn = QLabel("  <  ")
        move_left_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        move_left_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vars_btn_layout.addWidget(move_left_btn)
        vars_btn_layout.addStretch()

        # We need a different layout approach
        vars_group_inner = QGroupBox(STR_VARIABLES_LABEL)
        inner_layout = QHBoxLayout(vars_group_inner)

        # Available vars label
        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("Selected:"))
        avail_layout.addWidget(self.vars_list)
        inner_layout.addLayout(avail_layout, stretch=1)

        # Buttons
        btn_col = QVBoxLayout()
        btn_col.addStretch()
        btn_add = self._create_move_btn(">>", self._available_list, self.vars_list, True)
        btn_remove = self._create_move_btn("<<", self.vars_list, self._available_list, True)
        btn_col.addWidget(btn_add)
        btn_col.addWidget(btn_remove)
        btn_col.addStretch()
        inner_layout.addLayout(btn_col)

        layout.addWidget(vars_group_inner)

        # Grouping variable selection
        group_group = QGroupBox(STR_GROUPING_LABEL)
        group_layout = QHBoxLayout(group_group)

        self.group_list = self._create_selection_list("grouping", widget)
        self.group_list.setMaximumHeight(60)
        group_layout.addWidget(self.group_list)

        group_btn_col = QVBoxLayout()
        group_btn_col.addStretch()
        group_btn_add = self._create_move_btn(">", self._available_list, self.group_list, False)
        group_btn_remove = self._create_move_btn("<", self.group_list, self._available_list, False)
        group_btn_col.addWidget(group_btn_add)
        group_btn_col.addWidget(group_btn_remove)
        group_btn_col.addStretch()
        group_layout.addLayout(group_btn_col)

        layout.addWidget(group_group)
        layout.addStretch()

        return widget

    def _create_move_btn(
        self,
        label: str,
        source: QListWidget,
        target: QListWidget,
        all_items: bool,
    ) -> QWidget:
        """Create a move button widget."""
        from PySide6.QtWidgets import QPushButton
        btn = QPushButton(label)
        btn.setMaximumWidth(50)
        if all_items:
            btn.clicked.connect(lambda: self._move_all_items(source, target))
        else:
            btn.clicked.connect(lambda: self._move_selected_items(source, target))
        return btn

    def _build_options_area(self) -> QWidget:
        """Build options area with checkboxes and combos."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left column: checkboxes
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._add_option_checkbox("ci", STR_OPTIONS_CI, True))
        left_layout.addWidget(self._add_option_checkbox("missing_listwise", STR_OPTIONS_MISSING, True))
        left_layout.addWidget(self._add_option_checkbox("skewness", STR_OPTIONS_SKEW, True))
        left_layout.addWidget(self._add_option_checkbox("kurtosis", STR_OPTIONS_KURT, True))
        left_layout.addStretch()
        layout.addLayout(left_layout)

        # Right column: CI level
        right_layout = QVBoxLayout()
        ci_level_label, ci_level_combo = self._add_option_combo(
            "ci_level",
            STR_OPTIONS_CI_LEVEL,
            [
                ("90%", 0.90),
                ("95%", 0.95),
                ("99%", 0.99),
            ],
            1,  # 95% default
        )
        ci_level_layout = QHBoxLayout()
        ci_level_layout.addWidget(ci_level_label)
        ci_level_layout.addWidget(ci_level_combo)
        ci_level_layout.addStretch()
        right_layout.addLayout(ci_level_layout)
        right_layout.addStretch()
        layout.addLayout(right_layout)

        return widget

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the dialog inputs."""
        errors: list[str] = []

        var_names = self._get_selected_variable_names(self.vars_list)
        if not var_names:
            errors.append("At least one variable must be selected.")

        # Check that all selected variables are scale
        for name in var_names:
            try:
                var = self.dataset.get_variable(name)
                if var.measure != MeasureType.SCALE:
                    errors.append(
                        f"Variable '{name}' has measure '{var.measure.value}' - "
                        "Descriptive Statistics requires scale variables."
                    )
            except Exception:
                pass

        # Check grouping variable
        group_names = self._get_selected_variable_names(self.group_list)
        if len(group_names) > 1:
            errors.append("Only one grouping variable can be selected.")
        for name in group_names:
            try:
                var = self.dataset.get_variable(name)
                if var.measure not in (MeasureType.NOMINAL, MeasureType.ORDINAL, MeasureType.BINARY):
                    errors.append(
                        f"Grouping variable '{name}' should be nominal, ordinal, or binary."
                    )
            except Exception:
                pass

        return errors

    # ── Analysis spec ─────────────────────────────────────────────────────

    def get_analysis_spec(self) -> dict[str, Any]:
        """Return the descriptive statistics analysis specification."""
        return {
            "analysis_id": "descriptive_statistics",
            "title": "Descriptive Statistics",
            "variables": {
                "dependent": self._get_selected_variable_names(self.vars_list),
                "grouping": self._get_selected_variable_names(self.group_list),
            },
            "options": {
                "confidence_intervals": self._get_option_value("ci"),
                "confidence_level": self._get_option_value("ci_level"),
                "exclude_missing_listwise": self._get_option_value("missing_listwise"),
                "include_skewness": self._get_option_value("skewness"),
                "include_kurtosis": self._get_option_value("kurtosis"),
            },
            "missing_policy": "listwise" if self._get_option_value("missing_listwise") else "analysis_default",
        }
