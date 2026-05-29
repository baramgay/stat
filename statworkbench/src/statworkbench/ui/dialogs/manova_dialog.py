"""다변량 분산분석(MANOVA) 대화상자."""

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

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    populate_list_widget,
    scale_vars,
    user_friendly_error,
    var_from_display,
)


def _combo_add(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        combo.addItem(f"{icon} {label}" if icon else label, userData=var)


class ManovaDialog(QDialog):
    """SPSS General Linear Model > Multivariate 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("다변량 분산분석 — MANOVA")
        self.setMinimumSize(580, 580)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 집단 간 요인
        factor_group = QGroupBox("집단 간 요인 (Between-Subjects Factor) — 명목/순서형")
        factor_layout = QVBoxLayout(factor_group)
        self.factor_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset) or all_vars(self._dataset)
        _combo_add(self.factor_combo, self._dataset, cat_vars)
        factor_layout.addWidget(self.factor_combo)
        layout.addWidget(factor_group)

        # 종속변수 선택
        dep_group = QGroupBox("종속변수 (Dependent Variables) — 척도형, 최소 2개")
        dep_layout = QHBoxLayout(dep_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        avail = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        populate_list_widget(self.avail_list, self._dataset, avail, "(척도형 변수 없음)")
        left.addWidget(self.avail_list)
        dep_layout.addLayout(left)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_dep)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_dep)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        dep_layout.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 종속변수:"))
        self.dep_list = QListWidget()
        self.dep_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.dep_list)
        dep_layout.addLayout(right)
        layout.addWidget(dep_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_multivariate = QCheckBox("다변량 검정 (Pillai / Wilks / Hotelling / Roy)")
        self.chk_multivariate.setChecked(True)
        opt_layout.addWidget(self.chk_multivariate)

        self.chk_univariate = QCheckBox("단변량 후속 검정 (각 종속변수별 F)")
        self.chk_univariate.setChecked(True)
        opt_layout.addWidget(self.chk_univariate)

        self.chk_post_hoc = QCheckBox("사후 검정")
        self.chk_post_hoc.setChecked(True)
        opt_layout.addWidget(self.chk_post_hoc)

        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("  사후 검정 방법:"))
        self.post_hoc_combo = QComboBox()
        self.post_hoc_combo.addItems(["Bonferroni", "Tukey HSD"])
        ph_row.addWidget(self.post_hoc_combo)
        ph_row.addStretch()
        opt_layout.addLayout(ph_row)

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

    def _add_dep(self) -> None:
        already = {self.dep_list.item(i).text() for i in range(self.dep_list.count())}
        for item in self.avail_list.selectedItems():
            if item.text() not in already:
                self.dep_list.addItem(item.text())
                already.add(item.text())

    def _remove_dep(self) -> None:
        for item in self.dep_list.selectedItems():
            self.dep_list.takeItem(self.dep_list.row(item))

    def _run(self) -> None:
        factor_var = self.factor_combo.currentData() or var_from_display(
            self.factor_combo.currentText()
        )
        dep_labels = [self.dep_list.item(i).text() for i in range(self.dep_list.count())]
        dep_vars = [var_from_display(s) for s in dep_labels]
        dep_vars = [v for v in dep_vars if v]

        if not factor_var:
            QMessageBox.warning(self, "경고", "집단 간 요인을 선택하세요.")
            return
        if len(dep_vars) < 2:
            QMessageBox.warning(self, "경고", "종속변수를 2개 이상 선택하세요.")
            return
        if factor_var in dep_vars:
            QMessageBox.warning(self, "경고", "요인과 종속변수는 서로 달라야 합니다.")
            return

        ph_text = self.post_hoc_combo.currentText().lower().replace(" hsd", "").replace(" ", "")

        try:
            from statworkbench.analysis.manova import run_analysis
            spec = {
                "variables": {
                    "dependents": dep_vars,
                    "factor": factor_var,
                },
                "options": {
                    "multivariate": self.chk_multivariate.isChecked(),
                    "univariate": self.chk_univariate.isChecked(),
                    "post_hoc": self.chk_post_hoc.isChecked(),
                    "post_hoc_method": ph_text,
                    "effect_size": self.chk_effect.isChecked(),
                },
                "confidence_level": self.ci_spin.value(),
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "분석 오류", user_friendly_error(exc))
