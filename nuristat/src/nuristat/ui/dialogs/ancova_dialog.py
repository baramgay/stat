"""ANCOVA(공분산분석) 대화상자."""

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
        text = f"{icon} {label}" if icon else label
        combo.addItem(text, userData=var)


class AncovaDialog(QDialog):
    """SPSS General Linear Model > Univariate (공변량 포함) 스타일 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("공분산분석 — ANCOVA (Univariate with Covariate)")
        self.setMinimumSize(560, 580)
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

        # 요인
        fac_group = QGroupBox("요인 (Factor) — 명목/순서형")
        fac_layout = QVBoxLayout(fac_group)
        self.fac_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset) or all_vars(self._dataset)
        _combo_add(self.fac_combo, self._dataset, cat_vars)
        fac_layout.addWidget(self.fac_combo)
        layout.addWidget(fac_group)

        # 공변량 (다중 선택, 최대 3개)
        cov_group = QGroupBox("공변량 (Covariates) — 척도형, 최대 3개")
        cov_layout = QHBoxLayout(cov_group)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        avail = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        self.avail_list.addItems(avail)
        left_layout.addWidget(self.avail_list)
        cov_layout.addLayout(left_layout)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_cov)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_cov)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        cov_layout.addLayout(btn_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("선택된 공변량:"))
        self.cov_list = QListWidget()
        right_layout.addWidget(self.cov_list)
        cov_layout.addLayout(right_layout)
        layout.addWidget(cov_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_homog = QCheckBox("동질적 회귀 계수 가정 검정 (요인×공변량 상호작용)")
        self.chk_homog.setChecked(True)
        opt_layout.addWidget(self.chk_homog)

        self.chk_emm = QCheckBox("조정된 주변 평균 출력 (Estimated Marginal Means)")
        self.chk_emm.setChecked(True)
        opt_layout.addWidget(self.chk_emm)

        self.chk_post_hoc = QCheckBox("사후 검정 — Bonferroni (수준 ≥ 3인 요인에만 적용)")
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

    def _add_cov(self) -> None:
        already = {self.cov_list.item(i).text() for i in range(self.cov_list.count())}
        for item in self.avail_list.selectedItems():
            if item.text() not in already and self.cov_list.count() < 3:
                self.cov_list.addItem(item.text())
                already.add(item.text())

    def _remove_cov(self) -> None:
        for item in self.cov_list.selectedItems():
            self.cov_list.takeItem(self.cov_list.row(item))

    def _run(self) -> None:
        dep_var = self.dep_combo.currentData() or var_from_display(self.dep_combo.currentText())
        fac_var = self.fac_combo.currentData() or var_from_display(self.fac_combo.currentText())
        covariates = [
            self.cov_list.item(i).text()
            for i in range(self.cov_list.count())
        ]

        if not dep_var or not fac_var:
            QMessageBox.warning(self, "경고", "종속변수와 요인을 선택하세요.")
            return
        if not covariates:
            QMessageBox.warning(self, "경고", "공변량을 최소 1개 선택하세요.")
            return
        if dep_var == fac_var or dep_var in covariates:
            QMessageBox.warning(self, "경고", "종속변수, 요인, 공변량은 서로 달라야 합니다.")
            return

        try:
            from nuristat.analysis.ancova import run_analysis
            spec = {
                "variables": {
                    "dependent": dep_var,
                    "factor": fac_var,
                    "covariates": covariates,
                },
                "options": {
                    "homogeneity_test": self.chk_homog.isChecked(),
                    "emm": self.chk_emm.isChecked(),
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
