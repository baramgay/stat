"""Two-Way ANOVA 대화상자."""

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

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
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
        text = f"{icon} {label}" if icon else label
        combo.addItem(text, userData=var)


class TwoWayAnovaDialog(QDialog):
    """SPSS General Linear Model > Univariate (2-factor) 스타일 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("이원분산분석 — Two-Way ANOVA (Univariate)")
        self.setMinimumSize(520, 480)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 종속변수
        dep_group = QGroupBox("종속 변수 (Dependent Variable) — 척도형")
        dep_layout = QVBoxLayout(dep_group)
        self.dep_combo = QComboBox()
        dep_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        _combo_add(self.dep_combo, self._dataset, dep_vars)
        dep_layout.addWidget(self.dep_combo)
        layout.addWidget(dep_group)

        # 요인 A
        fa_group = QGroupBox("요인 A (Factor A) — 명목/순서형")
        fa_layout = QVBoxLayout(fa_group)
        self.fa_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset) or all_vars(self._dataset)
        _combo_add(self.fa_combo, self._dataset, cat_vars)
        fa_layout.addWidget(self.fa_combo)
        layout.addWidget(fa_group)

        # 요인 B
        fb_group = QGroupBox("요인 B (Factor B) — 명목/순서형")
        fb_layout = QVBoxLayout(fb_group)
        self.fb_combo = QComboBox()
        _combo_add(self.fb_combo, self._dataset, cat_vars)
        fb_layout.addWidget(self.fb_combo)
        layout.addWidget(fb_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_post_hoc = QCheckBox("사후검정 (Tukey HSD) — 수준 ≥ 3인 요인에만 적용")
        self.chk_post_hoc.setChecked(True)
        opt_layout.addWidget(self.chk_post_hoc)

        self.chk_effect = QCheckBox("효과 크기 (η² Eta-squared)")
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

    def _run(self) -> None:
        dep_var = self.dep_combo.currentData() or var_from_display(self.dep_combo.currentText())
        fa_var = self.fa_combo.currentData() or var_from_display(self.fa_combo.currentText())
        fb_var = self.fb_combo.currentData() or var_from_display(self.fb_combo.currentText())

        if not dep_var or not fa_var or not fb_var:
            QMessageBox.warning(self, "경고", "변수를 모두 선택하세요.")
            return
        if len({dep_var, fa_var, fb_var}) < 3:
            QMessageBox.warning(self, "경고", "종속변수, 요인 A, 요인 B는 서로 달라야 합니다.")
            return

        try:
            from statworkbench.analysis.two_way_anova import run_analysis
            spec = {
                "variables": {
                    "dependent": dep_var,
                    "factor_a": fa_var,
                    "factor_b": fb_var,
                },
                "options": {
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
