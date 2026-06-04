"""Base class for all statistical analysis dialogs in NuriStat."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.core.variable import VariableMeta

logger = logging.getLogger(__name__)

# ── String constants (i18n-ready) ─────────────────────────────────────────

STR_AVAILABLE_VARS = "Available Variables"
STR_AVAILABLE_VARS_KO = "사용 가능한 변수"
STR_SELECTED_VARS = "Selected Variables"
STR_SELECTED_VARS_KO = "선택된 변수"
STR_OPTIONS = "Options / 옵션"
STR_OK = "OK"
STR_PASTE_SYNTAX = "Paste Syntax"
STR_CANCEL = "Cancel"
STR_HELP = "Help"
STR_MOVE_RIGHT = ">"
STR_MOVE_LEFT = "<"
STR_MOVE_ALL_RIGHT = ">>"
STR_MOVE_ALL_LEFT = "<<"
STR_VALIDATE_ERROR = "Validation Error"


def _format_var_display(var: VariableMeta) -> str:
    """Format a variable for display in list widgets.

    Shows variable name, measure icon hint, and optional label.
    Label is shown only when it differs from the variable name.
    Format: "var_name (msr)" or "var_name (msr) [레이블]"
    """
    label_part = f"  [{var.label}]" if var.label and var.label != var.name else ""
    measure_tag = f" ({var.measure.value[:3]})" if var.measure else ""
    return f"{var.name}{measure_tag}{label_part}"


def _get_measure_icon_hint(var: VariableMeta) -> str:
    """Get a short icon hint string for the variable's measure type."""
    icon_map = {
        MeasureType.SCALE: "#",
        MeasureType.NOMINAL: "N",
        MeasureType.ORDINAL: "O",
        MeasureType.BINARY: "B",
        MeasureType.DATE_TIME: "D",
        MeasureType.TEXT: "T",
    }
    return icon_map.get(var.measure, "?")


class AnalysisDialogBase(QDialog):
    """모든 분석 대화상자의 기반 클래스.

    공통 구조:
    - 사용 가능한 변수 목록 (Available Variables)
    - 변수 선택 영역 (Dependent, Grouping, Covariates 등)
    - 옵션 영역
    - OK, Paste Syntax, Cancel, Help 버튼

    하위 클래스는 _build_selection_area()와 _build_options_area()를
    재정의하여 분석별 UI를 구성합니다.
    """

    analysis_completed = Signal(dict)  # spec dict
    syntax_pasted = Signal(dict)  # spec dict without running

    def __init__(
        self,
        dataset: Dataset,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._title = title
        self.setWindowTitle(title)
        self.setMinimumSize(750, 550)

        # UI components
        self._available_list: QListWidget | None = None
        self._selection_lists: dict[str, QListWidget] = {}
        self._option_widgets: dict[str, Any] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the common dialog layout."""
        main_layout = QVBoxLayout(self)

        # Top splitter: Available Variables + Selection Area
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Available variables panel
        available_group = QGroupBox(f"{STR_AVAILABLE_VARS} / {STR_AVAILABLE_VARS_KO}")
        available_layout = QVBoxLayout(available_group)
        self._available_list = QListWidget()
        self._available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._available_list.setToolTip(
            "분석에 사용할 변수를 선택하세요. 더블클릭 또는 화살표 버튼으로 이동할 수 있습니다."
        )
        available_layout.addWidget(self._available_list)
        top_splitter.addWidget(available_group)

        # Selection area (subclass-defined)
        self._selection_widget = self._build_selection_area()
        top_splitter.addWidget(self._selection_widget)

        top_splitter.setSizes([280, 400])
        main_layout.addWidget(top_splitter, stretch=3)

        # Options area (subclass-defined)
        self._options_group = QGroupBox(STR_OPTIONS)
        self._options_layout = QVBoxLayout(self._options_group)
        options_content = self._build_options_area()
        if options_content:
            self._options_layout.addWidget(options_content)
        main_layout.addWidget(self._options_group, stretch=1)

        # Buttons
        self._setup_common_buttons(main_layout)

        # Populate available variables
        self._setup_variable_lists()

    def _build_selection_area(self) -> QWidget:
        """Build the variable selection area.

        Subclasses should override this to create their specific selection
        layout with dependent/grouping/covariate lists and move buttons.

        Default implementation creates a simple placeholder.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Override _build_selection_area() in subclass"))
        return widget

    def _build_options_area(self) -> QWidget | None:
        """Build the analysis options area.

        Subclasses should override this to add checkboxes, combo boxes,
        spin boxes, etc. for analysis-specific options.

        Returns:
            QWidget containing options, or None for no options.
        """
        return None

    def _setup_common_buttons(self, layout: QVBoxLayout) -> None:
        """Set up OK, Paste Syntax, Cancel, Help buttons."""
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.help_btn = QPushButton(STR_HELP)
        self.help_btn.clicked.connect(self._show_help)
        btn_layout.addWidget(self.help_btn)

        btn_layout.addSpacing(20)

        self.paste_btn = QPushButton(STR_PASTE_SYNTAX)
        self.paste_btn.clicked.connect(self._on_paste_syntax)
        btn_layout.addWidget(self.paste_btn)

        self.ok_btn = QPushButton(STR_OK)
        self.ok_btn.setDefault(True)
        self.ok_btn.setToolTip("분석을 실행합니다 (Enter)")
        self.ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton(STR_CANCEL)
        self.cancel_btn.setToolTip("분석을 취소하고 닫습니다 (Esc)")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _setup_variable_lists(self) -> None:
        """Populate the available variables list from the dataset."""
        self._available_list.clear()
        for name in self._dataset.var_names:
            try:
                var = self._dataset.get_variable(name)
                item = QListWidgetItem(_format_var_display(var))
                item.setData(Qt.ItemDataRole.UserRole, var)
                item.setToolTip(f"{var.name}\nLabel: {var.label}\nMeasure: {var.measure.value}\nType: {var.storage_type.value}")
                self._available_list.addItem(item)
            except Exception:
                # Variable not in metadata, create minimal
                var = VariableMeta(name=name, label=name)
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, var)
                self._available_list.addItem(item)

    # ── Variable movement helpers ─────────────────────────────────────────

    def _create_move_buttons(
        self,
        source: QListWidget,
        target: QListWidget,
        allow_multi: bool = True,
        allow_all: bool = True,
    ) -> QVBoxLayout:
        """Create move buttons between two list widgets."""
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()

        if allow_all:
            move_all_right_btn = QPushButton(STR_MOVE_ALL_RIGHT)
            move_all_right_btn.setToolTip("Move all")
            move_all_right_btn.clicked.connect(
                lambda: self._move_all_items(source, target)
            )
            btn_layout.addWidget(move_all_right_btn)

        move_right_btn = QPushButton(STR_MOVE_RIGHT)
        move_right_btn.setToolTip("Move selected")
        move_right_btn.clicked.connect(
            lambda: self._move_selected_items(source, target)
        )
        btn_layout.addWidget(move_right_btn)

        move_left_btn = QPushButton(STR_MOVE_LEFT)
        move_left_btn.setToolTip("Remove selected")
        move_left_btn.clicked.connect(
            lambda: self._move_selected_items(target, source)
        )
        btn_layout.addWidget(move_left_btn)

        if allow_all:
            move_all_left_btn = QPushButton(STR_MOVE_ALL_LEFT)
            move_all_left_btn.setToolTip("Remove all")
            move_all_left_btn.clicked.connect(
                lambda: self._move_all_items(target, source)
            )
            btn_layout.addWidget(move_all_left_btn)

        btn_layout.addStretch()
        return btn_layout

    def _move_selected_items(self, source: QListWidget, target: QListWidget) -> None:
        """Move selected items from source to target list."""
        selected = source.selectedItems()
        for item in selected:
            row = source.row(item)
            taken = source.takeItem(row)
            target.addItem(taken)

    def _move_all_items(self, source: QListWidget, target: QListWidget) -> None:
        """Move all items from source to target list."""
        while source.count() > 0:
            item = source.takeItem(0)
            target.addItem(item)

    def _get_selected_variable_names(self, list_widget: QListWidget) -> list[str]:
        """Get variable names from a list widget."""
        names: list[str] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            var = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(var, VariableMeta):
                names.append(var.name)
            elif isinstance(var, str):
                names.append(var)
            else:
                names.append(item.text().split(" [")[0].split(" (")[0].strip())
        return names

    def _get_selected_variables(self, list_widget: QListWidget) -> list[VariableMeta]:
        """Get VariableMeta objects from a list widget."""
        vars: list[VariableMeta] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            var = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(var, VariableMeta):
                vars.append(var)
        return vars

    def _filter_available_by_measure(self, measures: list[MeasureType]) -> None:
        """Filter the available variables list by measure type."""
        for i in range(self._available_list.count()):
            item = self._available_list.item(i)
            var = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(var, VariableMeta):
                item.setHidden(var.measure not in measures)

    def _create_selection_list(self, label: str, parent: QWidget) -> QListWidget:
        """Create a labeled list widget for selected variables."""
        list_widget = QListWidget(parent)
        list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        list_widget.setMinimumHeight(80)
        list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._selection_lists[label] = list_widget
        return list_widget

    # ── Button handlers ───────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Validate inputs and emit analysis_completed signal."""
        errors = self.validate()
        if errors:
            QMessageBox.warning(
                self,
                STR_VALIDATE_ERROR,
                "\n".join(f"- {e}" for e in errors),
            )
            return

        spec = self.get_analysis_spec()
        self.analysis_completed.emit(spec)
        self.accept()

    def _on_paste_syntax(self) -> None:
        """Emit syntax_pasted signal without running analysis."""
        errors = self.validate()
        if errors:
            QMessageBox.warning(
                self,
                STR_VALIDATE_ERROR,
                "\n".join(f"- {e}" for e in errors),
            )
            return

        spec = self.get_analysis_spec()
        self.syntax_pasted.emit(spec)
        self.accept()

    def _show_help(self) -> None:
        """Show help for this analysis dialog."""
        QMessageBox.information(
            self,
            "Help",
            f"<b>{self._title}</b><br><br>"
            f"Select variables from the available list and move them to "
            f"the appropriate selection areas.<br><br>"
            f"Click <b>OK</b> to run the analysis.<br>"
            f"Click <b>Paste Syntax</b> to add the analysis to the syntax log.<br>"
            f"Click <b>Cancel</b> to close without running.",
        )

    # ── Abstract methods for subclasses ───────────────────────────────────

    @abstractmethod
    def get_analysis_spec(self) -> dict[str, Any]:
        """Return the analysis specification dictionary.

        This dictionary contains all information needed to run the analysis,
        including variable selections and option settings.

        Returns:
            Analysis specification dict with at minimum:
            {
                'analysis_id': str,
                'variables': {...},
                'options': {...},
            }
        """
        ...

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate the current dialog state.

        Returns a list of error/warning messages. An empty list means valid.
        """
        ...

    # ── Common utility methods ────────────────────────────────────────────

    def _add_option_checkbox(
        self,
        key: str,
        label: str,
        default: bool = False,
    ) -> QCheckBox:
        """Add a checkbox option and store reference."""
        cb = QCheckBox(label)
        cb.setChecked(default)
        self._option_widgets[key] = cb
        return cb

    def _add_option_combo(
        self,
        key: str,
        label: str,
        items: list[tuple[str, Any]],
        default_idx: int = 0,
    ) -> tuple[QLabel, QComboBox]:
        """Add a labeled combo box option and store reference."""
        lbl = QLabel(label)
        combo = QComboBox()
        for display, data in items:
            combo.addItem(display, data)
        combo.setCurrentIndex(default_idx)
        self._option_widgets[key] = combo
        return lbl, combo

    def _get_option_value(self, key: str) -> Any:
        """Get the value of an option widget."""
        widget = self._option_widgets.get(key)
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData() or widget.currentText()
        return None

    def _has_option(self, key: str) -> bool:
        """Check if an option exists."""
        return key in self._option_widgets

    @property
    def dataset(self) -> Dataset:
        """Return the current dataset."""
        return self._dataset

    def get_dataset_variable(self, name: str) -> VariableMeta:
        """Get a variable from the dataset by name."""
        return self._dataset.get_variable(name)

    def get_variable_unique_count(self, name: str) -> int:
        """Get the number of unique non-null values for a variable."""
        series = self._dataset.get_column(name)
        return series.dropna().nunique()

    def refresh_variables(self) -> None:
        """Refresh the available variables list."""
        self._setup_variable_lists()
