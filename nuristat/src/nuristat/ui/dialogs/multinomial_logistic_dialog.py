"""다항 로지스틱 회귀 대화상자."""

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
    QListWidgetItem,
    QMessageBox,
    QPushButton,
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
    var_from_display,
)


class MultinomialLogisticDialog(QDialog, AnalysisDialogMixin):
    """SPSS Analyze > Regression > Multinomial Logistic 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("다항 로지스틱 회귀")
        self.setMinimumSize(580, 560)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 종속변수
        dep_group = QGroupBox("종속변수 (범주형, 3개 이상 범주)")
        dep_layout = QVBoxLayout(dep_group)
        self.dep_combo = QComboBox()
        dep_vars = categorical_vars(self._dataset) or all_vars(self._dataset)
        for var in dep_vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            self.dep_combo.addItem(f"{icon} {label}" if icon else label, userData=var)
        self.dep_combo.currentIndexChanged.connect(self._update_reference_combo)
        dep_layout.addWidget(self.dep_combo)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("기준 범주:"))
        self.ref_combo = QComboBox()
        ref_row.addWidget(self.ref_combo)
        ref_row.addStretch()
        dep_layout.addLayout(ref_row)
        layout.addWidget(dep_group)

        # 예측변수
        pred_group = QGroupBox("예측변수 (Predictor Variables)")
        pred_layout = QHBoxLayout(pred_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        avail = numeric_vars(self._dataset) or all_vars(self._dataset)
        if avail:
            for var in avail:
                icon = measure_icon(self._dataset, var)
                label = display_label(self._dataset, var)
                item = QListWidgetItem(f"{icon} {label}" if icon else label)
                item.setData(0x0100, var)
                self.avail_list.addItem(item)
        else:
            from PySide6.QtCore import Qt
            placeholder = QListWidgetItem("(수치형 변수 없음)")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.avail_list.addItem(placeholder)
        left.addWidget(self.avail_list)
        pred_layout.addLayout(left)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_predictors)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_predictors)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        pred_layout.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 예측변수:"))
        self.pred_list = QListWidget()
        self.pred_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.pred_list)
        pred_layout.addLayout(right)
        layout.addWidget(pred_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

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

        self.chk_classification = QCheckBox("분류표 (Classification Table)")
        self.chk_classification.setChecked(True)
        opt_layout.addWidget(self.chk_classification)
        layout.addWidget(opt_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._update_reference_combo()

    def _update_reference_combo(self) -> None:
        dep_var = self.dep_combo.currentData()
        self.ref_combo.clear()
        if dep_var and self._dataset.data is not None and dep_var in self._dataset.data.columns:
            cats = sorted(self._dataset.data[dep_var].astype(str).dropna().unique())
            for c in cats:
                self.ref_combo.addItem(c, userData=c)
            if cats:
                self.ref_combo.setCurrentIndex(len(cats) - 1)

    def _add_predictors(self) -> None:
        already = {self.pred_list.item(i).data(0x0100) for i in range(self.pred_list.count())}
        for item in self.avail_list.selectedItems():
            if item.data(0x0100) not in already:
                new = QListWidgetItem(item.text())
                new.setData(0x0100, item.data(0x0100))
                self.pred_list.addItem(new)

    def _remove_predictors(self) -> None:
        for item in self.pred_list.selectedItems():
            self.pred_list.takeItem(self.pred_list.row(item))

    def _run(self) -> None:
        dep_var = self.dep_combo.currentData()
        ref_cat = self.ref_combo.currentData()
        predictors = [
            self.pred_list.item(i).data(0x0100) or var_from_display(self.pred_list.item(i).text())
            for i in range(self.pred_list.count())
        ]
        predictors = [p for p in predictors if p]

        if not dep_var:
            QMessageBox.warning(self, "경고", "종속변수를 선택하세요.")
            return
        if not predictors:
            QMessageBox.warning(self, "경고", "예측변수를 하나 이상 선택하세요.")
            return
        if dep_var in predictors:
            QMessageBox.warning(self, "경고", "종속변수와 예측변수는 서로 달라야 합니다.")
            return

        from nuristat.analysis.multinomial_logistic import run_analysis
        spec = {
            "variables": {
                "dependent": dep_var,
                "predictors": predictors,
            },
            "options": {
                "reference": ref_cat,
                "confidence_level": self.ci_spin.value(),
                "classification": self.chk_classification.isChecked(),
            },
            "missing_policy": "listwise",
        }
        self._start_analysis(run_analysis, self._dataset, spec)
