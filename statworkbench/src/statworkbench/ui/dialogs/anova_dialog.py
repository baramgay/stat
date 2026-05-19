"""ANOVA Dialog — SPSS 스타일 분산분석 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QCheckBox, QDialogButtonBox, QDoubleSpinBox
)
from PySide6.QtCore import Signal

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult
from statworkbench.ui.dialogs._dialog_helpers import (
    scale_vars, numeric_vars, categorical_vars, all_vars,
    display_label, var_from_display, measure_icon
)


def _combo_add(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        text = f"{icon} {label}" if icon else label
        combo.addItem(text, userData=var)


class ANOVADialog(QDialog):
    """SPSS One-Way ANOVA 다이얼로그.

    종속 변수: 척도(Scale)
    요인(독립) 변수: 명목/순서형 우선
    사후검정: Tukey HSD, Bonferroni, Scheffe
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("일원분산분석 (One-Way ANOVA)")
        self.setMinimumSize(520, 450)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 종속 변수
        dep_group = QGroupBox("종속 변수 (Dependent Variable) — 척도형")
        dep_layout = QVBoxLayout(dep_group)
        self.dep_combo = QComboBox()
        dep_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        _combo_add(self.dep_combo, self._dataset, dep_vars)
        dep_layout.addWidget(self.dep_combo)
        layout.addWidget(dep_group)

        # 요인 변수
        factor_group = QGroupBox("요인 변수 (Factor) — 명목/순서형")
        factor_layout = QVBoxLayout(factor_group)
        self.factor_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset)
        factor_vars = cat_vars if cat_vars else all_vars(self._dataset)
        _combo_add(self.factor_combo, self._dataset, factor_vars)
        factor_layout.addWidget(self.factor_combo)
        layout.addWidget(factor_group)

        # 사후검정
        posthoc_group = QGroupBox("사후검정 (Post Hoc Tests)")
        posthoc_layout = QVBoxLayout(posthoc_group)
        self.chk_tukey = QCheckBox("Tukey HSD")
        self.chk_tukey.setChecked(True)
        posthoc_layout.addWidget(self.chk_tukey)
        self.chk_bonferroni = QCheckBox("Bonferroni")
        posthoc_layout.addWidget(self.chk_bonferroni)
        self.chk_scheffe = QCheckBox("Scheffe")
        posthoc_layout.addWidget(self.chk_scheffe)
        layout.addWidget(posthoc_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_levene = QCheckBox("등분산성 검정 (Levene's Test)")
        self.chk_levene.setChecked(True)
        opt_layout.addWidget(self.chk_levene)

        self.chk_effect = QCheckBox("효과크기 (Eta squared)")
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

        # 버튼
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        dep_var = self.dep_combo.currentData() or var_from_display(self.dep_combo.currentText())
        factor_var = self.factor_combo.currentData() or var_from_display(self.factor_combo.currentText())

        if not dep_var or not factor_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        if dep_var == factor_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "종속 변수와 요인 변수가 같을 수 없습니다.")
            return

        post_hoc = []
        if self.chk_tukey.isChecked():
            post_hoc.append("tukey")
        if self.chk_bonferroni.isChecked():
            post_hoc.append("bonferroni")
        if self.chk_scheffe.isChecked():
            post_hoc.append("scheffe")

        try:
            from statworkbench.analysis.anova import run_analysis
            spec = {
                "variables": {"dependent": dep_var, "factor": factor_var},
                "options": {
                    "post_hoc": post_hoc,
                    "levene": self.chk_levene.isChecked(),
                    "effect_size": self.chk_effect.isChecked(),
                },
                "confidence_level": self.ci_spin.value(),
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
