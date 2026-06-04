"""Cox 비례위험 회귀 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


class CoxRegressionDialog(QDialog):
    """Cox 비례위험 회귀 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("Cox 비례위험 회귀")
        self.setMinimumSize(520, 520)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        num_vars = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)

        # 생존 시간
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

        # 공변량
        cov_group = QGroupBox("공변량 (Covariates) — 최소 1개")
        cov_layout = QHBoxLayout(cov_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능:"))
        self.avail_list = QListWidget()
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        for var in num_vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, var)
            self.avail_list.addItem(item)
        left.addWidget(self.avail_list)
        cov_layout.addLayout(left)

        btn_col = QVBoxLayout()
        btn_col.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_covariate)
        btn_col.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_covariate)
        btn_col.addWidget(rem_btn)
        btn_col.addStretch()
        cov_layout.addLayout(btn_col)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 공변량:"))
        self.cov_list = QListWidget()
        self.cov_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.cov_list)
        cov_layout.addLayout(right)
        layout.addWidget(cov_group)

        # 신뢰수준
        ci_row = QHBoxLayout()
        ci_row.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        ci_row.addWidget(self.ci_spin)
        ci_row.addStretch()
        layout.addLayout(ci_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _add_covariate(self) -> None:
        already = {self.cov_list.item(i).data(0x0100) for i in range(self.cov_list.count())}
        for item in self.avail_list.selectedItems():
            var = item.data(0x0100)
            if var not in already:
                new_item = QListWidgetItem(item.text())
                new_item.setData(0x0100, var)
                self.cov_list.addItem(new_item)
                already.add(var)

    def _remove_covariate(self) -> None:
        for item in self.cov_list.selectedItems():
            self.cov_list.takeItem(self.cov_list.row(item))

    def _run(self) -> None:
        duration_var = self.time_combo.currentData() or var_from_display(self.time_combo.currentText())
        event_var = self.event_combo.currentData() or var_from_display(self.event_combo.currentText())
        covariates = [
            self.cov_list.item(i).data(0x0100) or var_from_display(self.cov_list.item(i).text())
            for i in range(self.cov_list.count())
        ]
        covariates = [v for v in covariates if v]

        if not duration_var:
            QMessageBox.warning(self, "경고", "생존 시간 변수를 선택하세요.")
            return
        if not event_var:
            QMessageBox.warning(self, "경고", "사건 변수를 선택하세요.")
            return
        if duration_var == event_var:
            QMessageBox.warning(self, "경고", "생존 시간 변수와 사건 변수는 달라야 합니다.")
            return
        if not covariates:
            QMessageBox.warning(self, "경고", "공변량을 최소 1개 이상 선택하세요.")
            return
        if duration_var in covariates or event_var in covariates:
            QMessageBox.warning(self, "경고", "생존 시간/사건 변수와 공변량은 달라야 합니다.")
            return

        try:
            from nuristat.analysis.survival_analysis import run_cox_regression
            spec = {
                "variables": {
                    "duration": duration_var,
                    "event": event_var,
                    "covariates": covariates,
                },
                "options": {"method": "cox"},
                "confidence_level": self.ci_spin.value(),
            }
            result = run_cox_regression(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
