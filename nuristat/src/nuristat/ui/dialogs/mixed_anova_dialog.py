"""혼합 분산분석(Mixed ANOVA) 대화상자."""

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
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


def _combo_add(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        combo.addItem(f"{icon} {label}" if icon else label, userData=var)


class MixedAnovaDialog(QDialog):
    """SPSS General Linear Model > Repeated Measures (집단 간 요인 포함) 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("혼합 분산분석 — Mixed ANOVA (Split-Plot)")
        self.setMinimumSize(580, 600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 집단 간 요인
        between_group = QGroupBox("집단 간 요인 (Between-Subjects Factor) — 명목/순서형")
        between_layout = QVBoxLayout(between_group)
        self.between_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset) or all_vars(self._dataset)
        _combo_add(self.between_combo, self._dataset, cat_vars)
        between_layout.addWidget(self.between_combo)
        layout.addWidget(between_group)

        # 집단 내 변수 (측정 시점)
        within_group = QGroupBox("집단 내 측정 변수 (Within-Subjects) — 척도형, 최소 2개")
        within_layout = QHBoxLayout(within_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        avail = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        for var in avail:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, var)
            self.avail_list.addItem(item)
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        left.addWidget(self.avail_list)
        within_layout.addLayout(left)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_within)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_within)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        within_layout.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 시점 변수:"))
        self.within_list = QListWidget()
        self.within_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.within_list)
        within_layout.addLayout(right)
        layout.addWidget(within_group)

        # 시점 요인 이름
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("시점 요인 이름:"))
        self.within_name_combo = QComboBox()
        self.within_name_combo.setEditable(True)
        self.within_name_combo.addItems(["시점", "측정", "조건", "처치", "Time", "Condition"])
        name_row.addWidget(self.within_name_combo)
        name_row.addStretch()
        layout.addLayout(name_row)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_sphericity = QCheckBox("구형성 검정 (Mauchly's Test) + GG/HF 보정")
        self.chk_sphericity.setChecked(True)
        opt_layout.addWidget(self.chk_sphericity)

        self.chk_post_hoc = QCheckBox("Bonferroni 사후 검정 (시점 ≥ 3일 때 시점 간, 집단 ≥ 3일 때 집단 간)")
        self.chk_post_hoc.setChecked(True)
        opt_layout.addWidget(self.chk_post_hoc)

        self.chk_effect = QCheckBox("효과 크기 (편 η² Partial Eta Squared)")
        self.chk_effect.setChecked(True)
        opt_layout.addWidget(self.chk_effect)

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
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _add_within(self) -> None:
        from PySide6.QtWidgets import QListWidgetItem
        already = {self.within_list.item(i).data(0x0100) for i in range(self.within_list.count())}
        for item in self.avail_list.selectedItems():
            var = item.data(0x0100)
            if var not in already:
                new_item = QListWidgetItem(item.text())
                new_item.setData(0x0100, var)
                self.within_list.addItem(new_item)
                already.add(var)

    def _remove_within(self) -> None:
        for item in self.within_list.selectedItems():
            self.within_list.takeItem(self.within_list.row(item))

    def _run(self) -> None:
        between_var = self.between_combo.currentData() or var_from_display(self.between_combo.currentText())
        within_vars = [
            self.within_list.item(i).data(0x0100) or var_from_display(self.within_list.item(i).text())
            for i in range(self.within_list.count())
        ]
        within_vars = [v for v in within_vars if v]

        if not between_var:
            QMessageBox.warning(self, "경고", "집단 간 요인을 선택하세요.")
            return
        if len(within_vars) < 2:
            QMessageBox.warning(self, "경고", "집단 내 측정 변수를 2개 이상 선택하세요.")
            return
        if between_var in within_vars:
            QMessageBox.warning(self, "경고", "집단 간 요인과 집단 내 변수는 서로 달라야 합니다.")
            return

        try:
            from nuristat.analysis.mixed_anova import run_analysis
            spec = {
                "variables": {
                    "between": between_var,
                    "within": within_vars,
                    "within_name": self.within_name_combo.currentText().strip() or "시점",
                },
                "options": {
                    "sphericity": self.chk_sphericity.isChecked(),
                    "post_hoc": self.chk_post_hoc.isChecked(),
                    "effect_size": self.chk_effect.isChecked(),
                },
                "confidence_level": self.ci_spin.value(),
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
