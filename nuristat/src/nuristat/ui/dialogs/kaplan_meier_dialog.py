"""Kaplan-Meier 생존분석 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


def _add_combo(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    combo.addItem("(선택 안 함)", userData="")
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        combo.addItem(f"{icon} {label}" if icon else label, userData=var)


class KaplanMeierDialog(QDialog, AnalysisDialogMixin):
    """Kaplan-Meier 생존곡선 + 로그순위 검정 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("Kaplan-Meier 생존분석")
        self.setMinimumSize(440, 420)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        num_vars = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        cat_vars = categorical_vars(self._dataset) or all_vars(self._dataset)

        # 생존 시간 변수
        time_group = QGroupBox("생존 시간 변수 (Duration) — 척도형")
        time_layout = QVBoxLayout(time_group)
        self.time_combo = QComboBox()
        for var in num_vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            self.time_combo.addItem(f"{icon} {label}" if icon else label, userData=var)
        time_layout.addWidget(self.time_combo)
        layout.addWidget(time_group)

        # 사건 변수
        event_group = QGroupBox("사건 변수 (Event, 0=중도절단 / 1=사건)")
        event_layout = QVBoxLayout(event_group)
        self.event_combo = QComboBox()
        for var in num_vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            self.event_combo.addItem(f"{icon} {label}" if icon else label, userData=var)
        event_layout.addWidget(self.event_combo)
        layout.addWidget(event_group)

        # 그룹 변수 (로그순위 검정용, 선택)
        group_group = QGroupBox("그룹 변수 — 선택 (로그순위 검정)")
        group_layout = QVBoxLayout(group_group)
        self.group_combo = QComboBox()
        _add_combo(self.group_combo, self._dataset, cat_vars)
        group_layout.addWidget(self.group_combo)
        layout.addWidget(group_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_table = QCheckBox("생존함수 테이블 출력")
        self.chk_table.setChecked(True)
        opt_layout.addWidget(self.chk_table)

        ci_row = QHBoxLayout()
        ci_row.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        ci_row.addWidget(self.ci_spin)
        ci_row.addStretch()
        opt_layout.addLayout(ci_row)
        layout.addWidget(opt_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self) -> None:
        duration_var = self.time_combo.currentData() or var_from_display(self.time_combo.currentText())
        event_var = self.event_combo.currentData() or var_from_display(self.event_combo.currentText())
        group_var = self.group_combo.currentData() or None

        if not duration_var:
            QMessageBox.warning(self, "경고", "생존 시간 변수를 선택하세요.")
            return
        if not event_var:
            QMessageBox.warning(self, "경고", "사건 변수를 선택하세요.")
            return
        if duration_var == event_var:
            QMessageBox.warning(self, "경고", "생존 시간 변수와 사건 변수는 달라야 합니다.")
            return

        from nuristat.analysis.survival_analysis import run_kaplan_meier
        spec = {
            "variables": {
                "duration": duration_var,
                "event": event_var,
                **({"group": group_var} if group_var else {}),
            },
            "options": {
                "method": "km",
            },
            "confidence_level": self.ci_spin.value(),
        }
        self._start_analysis(run_kaplan_meier, self._dataset, spec)
